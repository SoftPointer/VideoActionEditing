from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import random
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import cage_self_generated_motion_fisher as fisher
else:  # pragma: no cover
    fisher = None


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@unittest.skipIf(torch is None, "torch is unavailable")
class CAGESelfGeneratedMotionFisherTests(unittest.TestCase):
    dimension = 8

    def _registration(
        self,
        *,
        blocks: tuple[int, ...] = (0, 1, 2, 3),
        families: tuple[str, ...] = ("head_turn", "sit_down"),
    ) -> fisher.MotionFisherRegistration:
        return fisher.MotionFisherRegistration(
            candidate_block_indices=blocks,
            required_action_families=families,
            minimum_identity_count=2,
            minimum_seeds_per_identity=2,
            maximum_nuisance_rank=2,
            nuisance_relative_eigenvalue_floor=1.0e-6,
            minimum_nuisance_residual_ratio=0.80,
            minimum_temporal_order_cosine=0.90,
            maximum_reverse_cosine=-0.80,
            minimum_reverse_norm_ratio=0.80,
            maximum_reverse_norm_ratio=1.25,
            minimum_seed_coherence=0.95,
            minimum_identity_coherence=0.95,
            minimum_identity_alignment=0.90,
            maximum_motion_rank=len(families),
            motion_relative_eigenvalue_floor=1.0e-3,
            motion_explained_variance_target=0.99,
            minimum_rank_boundary_relative_gap=0.05,
        )

    def _observations(
        self,
        *,
        blocks: tuple[int, ...] = (0, 1, 2, 3),
        families: tuple[str, ...] = ("head_turn", "sit_down"),
    ) -> list[fisher.MotionGradientObservation]:
        result: list[fisher.MotionGradientObservation] = []
        identities = ("actor_a", "actor_b")
        seeds = ("seed_01", "seed_02")
        for block in blocks:
            coordinate = _sha(f"coordinate-block-{block}")
            for family_index, family in enumerate(families):
                for identity_index, identity in enumerate(identities):
                    for seed_index, seed in enumerate(seeds):
                        direction = torch.zeros(self.dimension, dtype=torch.float32)
                        direction[family_index] = 1.0
                        # A small identity/seed term cancels in each identity's
                        # seed mean and cannot define the fitted family span.
                        direction[4] = (
                            0.02
                            * (1.0 if identity_index == 0 else -1.0)
                            * (1.0 if seed_index == 0 else -1.0)
                        )
                        zero = torch.zeros_like(direction)
                        camera = zero.clone()
                        camera[6] = 1.0
                        appearance = zero.clone()
                        appearance[7] = 1.0
                        gradients = {
                            "action": direction,
                            "reverse": -direction,
                            "freeze": zero,
                            "shuffle": zero,
                            "camera": camera,
                            "appearance": appearance,
                        }
                        event_digest = _sha(f"{family}/{identity}/{seed}/event")
                        for transform in fisher.REQUIRED_TRANSFORMS:
                            result.append(
                                fisher.MotionGradientObservation(
                                    block_index=block,
                                    action_family=family,
                                    identity_key=identity,
                                    seed_key=seed,
                                    transform=transform,
                                    coordinate_digest=coordinate,
                                    event_receipt_digest=event_digest,
                                    event_qualified=True,
                                    origin=fisher.OBSERVATION_ORIGIN,
                                    gradient=gradients[transform].clone(),
                                )
                            )
        return result

    def _rv2v(
        self,
        *,
        blocks: tuple[int, ...],
        families: tuple[str, ...],
        mode_by_block: dict[int, str] | None = None,
    ) -> list[fisher.RV2VGradientObservation]:
        mode_by_block = mode_by_block or {}
        result: list[fisher.RV2VGradientObservation] = []
        for block in blocks:
            for family_index, family in enumerate(families):
                gradient = torch.zeros(self.dimension, dtype=torch.float32)
                mode = mode_by_block.get(block, "aligned")
                if mode == "aligned":
                    gradient[family_index] = 1.0
                    gradient[6] = 0.25
                elif mode == "opposite":
                    gradient[family_index] = -1.0
                elif mode == "orthogonal":
                    gradient[3] = 1.0
                else:  # pragma: no cover - test helper misuse
                    raise AssertionError(mode)
                result.append(
                    fisher.RV2VGradientObservation(
                        block_index=block,
                        action_family=family,
                        coordinate_digest=_sha(f"coordinate-block-{block}"),
                        student_state_receipt_digest=_sha(f"state/{family}"),
                        branch_lock_receipt_digest=_sha("one-branch-lock"),
                        origin=fisher.RV2V_GRADIENT_ORIGIN,
                        gradient=gradient,
                    )
                )
        return result

    def test_fits_orthonormal_nuisance_free_motion_proxy(self) -> None:
        registration = self._registration()
        result = fisher.fit_self_generated_motion_fisher(
            self._observations(), registration
        )
        self.assertEqual(result.qualified_block_indices, (0, 1, 2, 3))
        for block in result.blocks:
            self.assertTrue(block.qualified)
            self.assertEqual(block.motion_rank, 2)
            identity = torch.eye(2, dtype=torch.float64)
            self.assertTrue(
                torch.allclose(
                    block.motion_basis.T @ block.motion_basis,
                    identity,
                    atol=1.0e-10,
                    rtol=0.0,
                )
            )
            self.assertLess(
                float(
                    torch.max(
                        torch.abs(block.nuisance_basis.T @ block.motion_basis)
                    ).item()
                ),
                1.0e-10,
            )
            self.assertFalse(block.receipt["optimizer_update_authorized"])
            self.assertEqual(
                block.receipt["fisher_interpretation"],
                "empirical_contrastive_psd_second_moment_not_statistical_fisher",
            )

    def test_input_order_does_not_change_fit_or_canonical_basis(self) -> None:
        registration = self._registration()
        observations = self._observations()
        shuffled = list(observations)
        random.Random(17).shuffle(shuffled)
        left = fisher.fit_self_generated_motion_fisher(
            observations, registration
        )
        right = fisher.fit_self_generated_motion_fisher(shuffled, registration)
        self.assertEqual(left.receipt, right.receipt)
        for left_block, right_block in zip(left.blocks, right.blocks):
            self.assertTrue(
                torch.equal(left_block.motion_basis, right_block.motion_basis)
            )
            self.assertTrue(
                torch.equal(left_block.nuisance_basis, right_block.nuisance_basis)
            )

    def test_reverse_gate_rejects_same_signed_reverse(self) -> None:
        blocks = (0,)
        families = ("head_turn",)
        observations = self._observations(blocks=blocks, families=families)
        changed: list[fisher.MotionGradientObservation] = []
        for observation in observations:
            if observation.transform == "reverse":
                changed.append(
                    replace(observation, gradient=-observation.gradient)
                )
            else:
                changed.append(observation)
        result = fisher.fit_self_generated_motion_fisher(
            changed, self._registration(blocks=blocks, families=families)
        )
        self.assertEqual(result.qualified_block_indices, ())
        self.assertTrue(
            any(
                "reverse_is_not_opposite_signed" in reason
                for reason in result.blocks[0].rejection_reasons
            )
        )
        self.assertEqual(result.blocks[0].motion_basis.shape, (self.dimension, 0))

    def test_cross_identity_disagreement_rejects_block(self) -> None:
        blocks = (0,)
        families = ("head_turn",)
        observations = self._observations(blocks=blocks, families=families)
        changed: list[fisher.MotionGradientObservation] = []
        for observation in observations:
            if observation.identity_key == "actor_b" and observation.transform in (
                "action",
                "reverse",
            ):
                changed.append(
                    replace(observation, gradient=-observation.gradient)
                )
            else:
                changed.append(observation)
        result = fisher.fit_self_generated_motion_fisher(
            changed, self._registration(blocks=blocks, families=families)
        )
        self.assertFalse(result.blocks[0].qualified)
        self.assertTrue(
            any(
                "zero_cross_identity_consensus" in reason
                for reason in result.blocks[0].rejection_reasons
            )
        )

    def test_action_that_is_camera_nuisance_is_removed_and_rejected(self) -> None:
        blocks = (0,)
        families = ("head_turn",)
        observations = self._observations(blocks=blocks, families=families)
        changed: list[fisher.MotionGradientObservation] = []
        for observation in observations:
            value = observation.gradient
            if observation.transform == "camera":
                value = torch.zeros_like(value)
                value[0] = 1.0
            changed.append(replace(observation, gradient=value))
        result = fisher.fit_self_generated_motion_fisher(
            changed, self._registration(blocks=blocks, families=families)
        )
        self.assertFalse(result.blocks[0].qualified)
        self.assertTrue(
            any(
                "action_contrast_collapses_into_nuisance_span" in reason
                or "zero_nuisance_clean_forward_direction" in reason
                for reason in result.blocks[0].rejection_reasons
            )
        )

    def test_missing_transform_and_coordinate_mismatch_fail_closed(self) -> None:
        blocks = (0,)
        families = ("head_turn",)
        observations = self._observations(blocks=blocks, families=families)
        with self.assertRaisesRegex(fisher.CAGEMotionFisherError, "transform registry"):
            fisher.fit_self_generated_motion_fisher(
                observations[:-1],
                self._registration(blocks=blocks, families=families),
            )
        fit = fisher.fit_self_generated_motion_fisher(
            observations, self._registration(blocks=blocks, families=families)
        )
        rv2v = self._rv2v(blocks=blocks, families=families)[0]
        with self.assertRaisesRegex(fisher.CAGEMotionFisherError, "coordinates"):
            fisher.project_native_rv2v_gradient(
                fit, replace(rv2v, coordinate_digest=_sha("wrong-coordinate"))
            )

    def test_rv2v_projection_filters_nuisance_without_copying_t2v_step(self) -> None:
        blocks = (0,)
        families = ("head_turn",)
        fit = fisher.fit_self_generated_motion_fisher(
            self._observations(blocks=blocks, families=families),
            self._registration(blocks=blocks, families=families),
        )
        observation = self._rv2v(blocks=blocks, families=families)[0]
        result = fisher.project_native_rv2v_gradient(fit, observation)
        self.assertGreater(result.projection_fraction, 0.90)
        self.assertGreater(result.signed_family_alignment, 0.99)
        self.assertAlmostEqual(float(result.projected_gradient[6].item()), 0.0)
        self.assertAlmostEqual(float(result.projected_gradient[0].item()), 1.0)
        self.assertEqual(result.projected_gradient.dtype, torch.float32)
        self.assertFalse(result.receipt["optimizer_update_authorized"])
        self.assertEqual(
            result.receipt["projection_operator"],
            "signless_motion_subspace_projector",
        )

    def test_selects_only_pre_registered_exact_contiguous_band(self) -> None:
        blocks = (0, 1, 2, 3)
        families = ("head_turn", "sit_down")
        fit = fisher.fit_self_generated_motion_fisher(
            self._observations(blocks=blocks, families=families),
            self._registration(blocks=blocks, families=families),
        )
        gradients = self._rv2v(
            blocks=blocks,
            families=families,
            mode_by_block={2: "opposite", 3: "orthogonal"},
        )
        band = fisher.select_registered_contiguous_block_band(
            fit,
            gradients,
            fisher.ContinuousBandRegistration(
                candidate_block_indices=blocks,
                required_action_families=families,
                exact_band_length=2,
                minimum_fit_quality_score=0.80,
                minimum_projection_fraction=0.70,
                minimum_family_alignment=0.80,
                selection_rule=fisher.BAND_SELECTION_RULE,
            ),
        )
        self.assertTrue(band.band_selection_authorized)
        self.assertEqual(band.selected_block_indices, (0, 1))
        self.assertFalse(band.receipt["optimizer_update_authorized"])

    def test_band_selection_returns_null_when_no_window_passes(self) -> None:
        blocks = (0, 1)
        families = ("head_turn",)
        fit = fisher.fit_self_generated_motion_fisher(
            self._observations(blocks=blocks, families=families),
            self._registration(blocks=blocks, families=families),
        )
        gradients = self._rv2v(
            blocks=blocks,
            families=families,
            mode_by_block={0: "opposite", 1: "orthogonal"},
        )
        result = fisher.select_registered_contiguous_block_band(
            fit,
            gradients,
            fisher.ContinuousBandRegistration(
                candidate_block_indices=blocks,
                required_action_families=families,
                exact_band_length=2,
                minimum_fit_quality_score=0.80,
                minimum_projection_fraction=0.70,
                minimum_family_alignment=0.80,
                selection_rule=fisher.BAND_SELECTION_RULE,
            ),
        )
        self.assertFalse(result.band_selection_authorized)
        self.assertEqual(result.selected_block_indices, ())
        self.assertEqual(
            result.block_reason,
            "no_exact_contiguous_band_passes_all_registered_gates",
        )

    def test_band_comparison_requires_same_rv2v_state_across_blocks(self) -> None:
        blocks = (0, 1)
        families = ("head_turn",)
        fit = fisher.fit_self_generated_motion_fisher(
            self._observations(blocks=blocks, families=families),
            self._registration(blocks=blocks, families=families),
        )
        gradients = self._rv2v(blocks=blocks, families=families)
        gradients[1] = replace(
            gradients[1],
            student_state_receipt_digest=_sha("different-student-state"),
        )
        with self.assertRaisesRegex(
            fisher.CAGEMotionFisherError, "same RV2V student state"
        ):
            fisher.select_registered_contiguous_block_band(
                fit,
                gradients,
                fisher.ContinuousBandRegistration(
                    candidate_block_indices=blocks,
                    required_action_families=families,
                    exact_band_length=2,
                    minimum_fit_quality_score=0.80,
                    minimum_projection_fraction=0.70,
                    minimum_family_alignment=0.80,
                    selection_rule=fisher.BAND_SELECTION_RULE,
                ),
            )

    def test_contract_excludes_every_media_carrier_and_update_authority(self) -> None:
        receipt = fisher.contract_receipt()
        self.assertFalse(receipt["optimizer_update_authorized"])
        self.assertIn("t2v_pixel", receipt["forbidden_carriers"])
        self.assertIn("t2v_latent", receipt["forbidden_carriers"])
        self.assertIn("t2v_hidden_state", receipt["forbidden_carriers"])
        self.assertEqual(
            receipt["fisher_interpretation"],
            "empirical_contrastive_psd_second_moment_not_statistical_fisher",
        )


if __name__ == "__main__":
    unittest.main()

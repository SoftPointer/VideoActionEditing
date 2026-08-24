from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "diagnose_starc_core4_hidden_temporal_quotient_v1.py"
)
SPEC = importlib.util.spec_from_file_location("starc_hidden_quotient", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class HiddenTemporalQuotientTest(unittest.TestCase):
    def test_representation_contract_and_names(self) -> None:
        generator = torch.Generator().manual_seed(17)
        value = torch.randn(module.EXPECTED_SHAPE, generator=generator)
        result = module.temporal_representations(value)
        self.assertEqual(
            tuple(result),
            (
                "raw_hidden",
                "centered_hidden",
                "temporal_velocity",
                "endpoint_arrow",
                "phase_energy",
                "velocity_energy",
                "temporal_singular_values",
                "global_temporal_self_similarity",
                "sketch_temporal_self_similarity",
                "centered_sketch_self_similarity",
                "centered_phase_mean",
            ),
        )
        for feature in result.values():
            self.assertEqual(feature.ndim, 1)
            self.assertTrue(torch.isfinite(feature).all())
            self.assertAlmostEqual(float(torch.linalg.vector_norm(feature)), 1.0, places=5)

    def test_global_self_similarity_is_hidden_rotation_invariant(self) -> None:
        generator = torch.Generator().manual_seed(23)
        value = torch.randn(module.EXPECTED_SHAPE, generator=generator)
        sign = torch.where(torch.arange(1536) % 2 == 0, 1.0, -1.0)
        rotated = value * sign.reshape(1, 1, 1, -1)
        left = module.temporal_representations(value)[
            "global_temporal_self_similarity"
        ]
        right = module.temporal_representations(rotated)[
            "global_temporal_self_similarity"
        ]
        self.assertTrue(torch.allclose(left, right, atol=1.0e-6, rtol=1.0e-6))

    def test_fit_only_contrast_and_low_rank_pass_for_shared_arrows(self) -> None:
        generator = torch.Generator().manual_seed(31)
        dimension = 64
        positive = torch.randn(dimension, generator=generator)
        arrows = [torch.randn(dimension, generator=generator) for _ in module.NEGATIVE_ROLES]
        fit = {"positive": positive}
        confirmation = {"positive": positive + 0.02 * torch.randn(dimension, generator=generator)}
        for role, arrow in zip(module.NEGATIVE_ROLES, arrows):
            fit[role] = positive - arrow
            confirmation[role] = confirmation["positive"] - arrow
        fit = {role: fit[role] for role in module.EXPECTED_ROLES}
        confirmation = {role: confirmation[role] for role in module.EXPECTED_ROLES}
        result = module.diagnose_representation(
            fit_by_role=fit, confirmation_by_role=confirmation
        )
        self.assertTrue(result["matched_contrast"]["all_positive"])
        self.assertTrue(
            result["fit_only_role_contrast_subspace"]["4"]["all_positive"]
        )
        rank_one = result["fit_only_role_contrast_subspace"]["1"]
        self.assertGreater(rank_one["explained_fit_contrast_energy"], 0.0)
        self.assertGreater(rank_one["min_confirmation_support_fraction"], 0.0)
        common = result["fit_only_common_direction_ranking"]
        self.assertTrue(common["all_positive"])
        self.assertGreater(common["min_confirmation_margin"], 0.0)

    def test_reversed_confirmation_arrow_fails_closed(self) -> None:
        dimension = 64
        positive = torch.zeros(dimension)
        positive[-1] = 1.0
        fit = {"positive": positive.clone()}
        confirmation = {"positive": -positive.clone()}
        for index, role in enumerate(module.NEGATIVE_ROLES):
            arrow = torch.zeros(dimension)
            arrow[index] = 1.0
            fit[role] = positive - arrow
            confirmation[role] = -(positive - arrow)
        fit = {role: fit[role] for role in module.EXPECTED_ROLES}
        confirmation = {role: confirmation[role] for role in module.EXPECTED_ROLES}
        result = module.diagnose_representation(
            fit_by_role=fit, confirmation_by_role=confirmation
        )
        self.assertEqual(result["matched_contrast"]["positive_count"], 0)
        self.assertFalse(result["matched_contrast"]["all_positive"])
        self.assertFalse(result["fit_only_common_direction_ranking"]["all_positive"])

    def test_nuisance_projection_is_fit_only_and_finite(self) -> None:
        generator = torch.Generator().manual_seed(41)
        values = {
            role: torch.randn(64, generator=generator)
            for role in module.EXPECTED_ROLES
        }
        confirmation = {
            role: value + 0.01 * torch.randn(64, generator=generator)
            for role, value in values.items()
        }
        result = module.nuisance_projected_action_ranking(
            fit_by_role=values,
            confirmation_by_role=confirmation,
            nuisance_roles=("semantic_camera_only", "semantic_appearance_only"),
        )
        self.assertEqual(result["nuisance_effective_rank"], 2)
        self.assertGreater(result["retained_action_norm_fraction"], 0.0)
        self.assertEqual(len(result["rows"]), len(module.NEGATIVE_ROLES))
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(row["confirmation_margin"]))
                for row in result["rows"]
            )
        )


if __name__ == "__main__":
    unittest.main()

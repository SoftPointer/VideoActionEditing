from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from spt_v2 import phase_transport as spt
from spt_v2.oracle_diagnostic import build_parser
from spt_v2.training_objective import SPTLossConfig
from spt_v2 import contracts


def _all_modules() -> list[str]:
    return sorted(
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(30)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    )


class PureContractTests(unittest.TestCase):
    def test_exact_21_phase_contract(self) -> None:
        spt.PhaseTransportConfig().validate()
        self.assertEqual(
            spt.PhaseTransportConfig().max_generate_fraction_per_phase, 0.12
        )
        with self.assertRaises(spt.PhaseTransportError):
            spt.PhaseTransportConfig(latent_phases=20).validate()

    def test_unbounded_generate_requires_explicit_ablation(self) -> None:
        with self.assertRaises(spt.PhaseTransportError):
            spt.PhaseTransportConfig(
                max_generate_fraction_per_phase=None
            ).validate()
        spt.PhaseTransportConfig(
            max_generate_fraction_per_phase=None,
            teacher_allow_unbounded_generate_ablation=True,
        ).validate()

    def test_student_api_cannot_receive_target(self) -> None:
        signature = inspect.signature(spt.PhaseTransportAdapter.forward)
        self.assertEqual(list(signature.parameters), ["self", "source", "instruction_embedding"])

    def test_oracle_cli_requires_only_pair_store_not_external_models(self) -> None:
        args = build_parser().parse_args(
            [
                "--checkpoint", "/checkpoint",
                "--preprocessed-parquet-dir", "/pairs",
                "--output", "/result.json",
            ]
        )
        self.assertEqual(args.row_index, 0)
        self.assertEqual(args.feature_channels, 64)
        self.assertGreater(args.teacher_transport_margin, 0.0)
        self.assertFalse(args.disable_cycle_gate)
        self.assertFalse(args.allow_lossy_projection_ablation)
        forbidden = {"mask", "track", "pose", "flow", "trajectory", "sam"}
        self.assertTrue(forbidden.isdisjoint(vars(args)))

    def test_loss_defaults_favor_late_detail_not_high_noise(self) -> None:
        config = SPTLossConfig()
        config.validate()
        self.assertGreater(config.flow_weight, 0.0)
        self.assertGreater(config.gate_distill_weight, 0.0)
        self.assertGreater(config.low_noise_floor, 0.0)

    def test_v2_scope_adds_only_middle_self_attention_to_cross_attention(self) -> None:
        available = _all_modules()
        cross = contracts.select_spt_lora_scope(available, "cross_q_out")
        cross_mid = contracts.select_spt_lora_scope(available, "cross_mid_q_out")
        all_q_out = contracts.select_spt_lora_scope(available, "q_out")
        self.assertEqual(len(cross), 60)
        self.assertEqual(len(cross_mid), 92)
        self.assertEqual(len(all_q_out), 120)
        self.assertTrue(set(cross) < set(cross_mid) < set(all_q_out))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorContractTests(unittest.TestCase):
    def _video(self, *, channels: int = 4):
        return torch.randn(1, 21, 3, 5, channels)

    def test_noop_is_exact_identity_independent_of_plan(self) -> None:
        source = self._video()
        generated = torch.randn_like(source)
        offsets = torch.randn(1, 3, 21, 3, 5) * 10
        gates = torch.softmax(torch.randn(1, 3, 21, 3, 5), dim=1)
        plan = spt.PhasePlan(offsets, gates, "student")
        actual = spt.execute_clean_plan(source, generated, plan, noop=True)
        self.assertIs(actual, source)
        self.assertTrue(torch.equal(actual, source))

    def test_zero_offset_transport_retrieves_source(self) -> None:
        source = self._video()
        offsets = torch.zeros(1, 3, 21, 3, 5)
        actual = spt.transport_source(source, offsets)
        self.assertTrue(torch.allclose(actual, source, atol=2e-6, rtol=2e-6))

    def test_three_way_executor_is_convex_and_endpoint_exact(self) -> None:
        source = self._video()
        generated = torch.randn_like(source)
        offsets = torch.zeros(1, 3, 21, 3, 5)
        for gate_index, expected in ((0, source), (1, source), (2, generated)):
            gates = torch.zeros(1, 3, 21, 3, 5)
            gates[:, gate_index] = 1
            plan = spt.PhasePlan(offsets, gates, "student")
            actual = spt.execute_clean_plan(source, generated, plan)
            self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))

    def test_packed_boundary_executes_the_same_raw_velocity_it_returns(self) -> None:
        source = self._video()
        generated = torch.randn_like(source)
        sigma = torch.tensor([0.4])
        noisy = generated + sigma.reshape(1, 1, 1, 1, 1) * torch.randn_like(generated)
        base_velocity = (noisy - generated) / sigma.reshape(1, 1, 1, 1, 1)
        offsets = torch.zeros(1, 3, 21, 3, 5)
        gates = torch.zeros(1, 3, 21, 3, 5)
        gates[:, 0] = 1.0
        plan = spt.PhasePlan(offsets, gates, "student")
        actual = spt.execute_packed_velocity(
            source_packed=spt.video_to_packed(source),
            noisy_packed=spt.video_to_packed(noisy),
            base_velocity_packed=spt.video_to_packed(base_velocity),
            sigma=sigma,
            height=3,
            width=5,
            plan=plan,
        )
        expected = spt.video_to_packed((noisy - source) / sigma.reshape(1, 1, 1, 1, 1))
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))

    def test_oracle_recovers_a_known_source_transport_better_than_copy(self) -> None:
        tt = 100.0 * torch.arange(21, dtype=torch.float32).view(1, 21, 1, 1, 1)
        yy = 10.0 * torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1, 1)
        xx = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5, 1)
        source = (tt + yy + xx).expand(1, 21, 3, 5, 2).clone()
        true_offset = torch.zeros(1, 3, 21, 3, 5)
        true_offset[:, 2].fill_(-1.0)
        target = spt.transport_source(source, true_offset)
        # The first column is genuinely new.  This removes the border-clamp
        # ambiguity and gives the cycle check a unique inverse for source x=0.
        target[:, :, :, 0].fill_(-100.0)
        config = spt.PhaseTransportConfig(
            latent_channels=2,
            text_channels=3,
            hidden_channels=8,
            teacher_temporal_offsets=(0,),
            teacher_spatial_offsets=(-1, 0, 1),
            teacher_temperature=0.001,
            teacher_generate_threshold=0.1,
            max_generate_fraction_per_phase=0.5,
        )
        oracle = spt.build_oracle_plan(source, target, config, feature_channels=2)
        proxy = spt.make_proxy_target(source, target, oracle)
        source_error = torch.mean((source - target) ** 2)
        proxy_error = torch.mean((proxy - target) ** 2)
        self.assertLess(float(proxy_error), float(source_error) * 0.2)
        self.assertGreater(float(oracle.gate_probs[:, 1].mean()), 0.5)
        self.assertLess(oracle.diagnostics["hard_executor_candidate_mse"], 1e-10)

    def test_projection_uses_every_packed_channel_and_full_width_preserves_l2(self) -> None:
        projection = spt.fixed_auditable_projection(64, 64)
        self.assertEqual(tuple(projection.shape), (64, 64))
        self.assertTrue(bool((projection != 0).all()))
        identity = projection @ projection.transpose(0, 1)
        self.assertTrue(torch.allclose(identity, torch.eye(64), atol=2e-5, rtol=2e-5))
        metadata = spt.projection_audit_metadata(64, 64)
        self.assertEqual(metadata["covered_input_channels"], 64)
        self.assertEqual(metadata["input_coverage_fraction"], 1.0)
        self.assertTrue(metadata["full_l2_preserving"])
        self.assertEqual(len(metadata["float32_matrix_sha256"]), 64)

    def test_main_oracle_fails_closed_on_lossy_projection(self) -> None:
        source = self._video(channels=4)
        config = spt.PhaseTransportConfig(
            latent_channels=4, text_channels=3, hidden_channels=8
        )
        with self.assertRaises(spt.PhaseTransportError):
            spt.build_oracle_plan(source, source, config, feature_channels=2)
        ablation = spt.PhaseTransportConfig(
            latent_channels=4,
            text_channels=3,
            hidden_channels=8,
            teacher_allow_lossy_projection=True,
        )
        plan = spt.build_oracle_plan(source, source, ablation, feature_channels=2)
        self.assertFalse(plan.diagnostics["projection"]["full_l2_preserving"])

    def test_oracle_hard_offsets_are_valid_candidates_and_executor_consistent(self) -> None:
        source = self._video(channels=4)
        target = torch.roll(source, shifts=1, dims=3)
        config = spt.PhaseTransportConfig(
            latent_channels=4,
            text_channels=3,
            hidden_channels=8,
            teacher_temporal_offsets=(-1, 0, 1),
            teacher_spatial_offsets=(-1, 0, 1),
            teacher_generate_threshold=10.0,
            teacher_transport_margin=1e-4,
        )
        oracle = spt.build_oracle_plan(source, target, config, feature_channels=4)
        candidates = set(spt._candidate_grid(config))
        observed = {
            tuple(int(value) for value in cell)
            for cell in oracle.offsets.permute(0, 2, 3, 4, 1).reshape(-1, 3).tolist()
        }
        self.assertTrue(observed <= candidates)
        self.assertLess(oracle.diagnostics["hard_executor_candidate_mse"], 1e-10)

        dt, dy, dx = oracle.offsets.long().unbind(dim=1)
        tt = torch.arange(21).view(1, 21, 1, 1) + dt
        yy = torch.arange(3).view(1, 1, 3, 1) + dy
        xx = torch.arange(5).view(1, 1, 1, 5) + dx
        self.assertTrue(bool(((tt >= 0) & (tt < 21)).all()))
        self.assertTrue(bool(((yy >= 0) & (yy < 3)).all()))
        self.assertTrue(bool(((xx >= 0) & (xx < 5)).all()))

    def test_transport_requires_explicit_improvement_over_zero(self) -> None:
        source = self._video(channels=4)
        target = torch.roll(source, shifts=1, dims=3)
        config = spt.PhaseTransportConfig(
            latent_channels=4,
            text_channels=3,
            hidden_channels=8,
            teacher_temporal_offsets=(0,),
            teacher_spatial_offsets=(-1, 0, 1),
            teacher_generate_threshold=1e6,
            teacher_transport_margin=1e6,
        )
        oracle = spt.build_oracle_plan(source, target, config, feature_channels=4)
        self.assertEqual(float(oracle.gate_probs[:, 1].sum()), 0.0)

    def test_generate_budget_is_per_phase_topk_and_rejects_to_preserve(self) -> None:
        candidates = torch.ones(1, 21, 2, 5, dtype=torch.bool)
        score = torch.arange(10, dtype=torch.float32).view(1, 1, 2, 5).expand_as(candidates)
        retained, rejected = spt._budget_generate_per_phase(candidates, score, 0.12)
        self.assertTrue(torch.equal(retained.sum(dim=(-2, -1)), torch.ones(1, 21)))
        self.assertTrue(bool(retained[:, :, 1, 4].all()))
        self.assertTrue(torch.equal(rejected, candidates & ~retained))

        source = torch.zeros(1, 21, 3, 5, 4)
        target = torch.ones_like(source) * 10.0
        config = spt.PhaseTransportConfig(
            latent_channels=4,
            text_channels=3,
            hidden_channels=8,
            teacher_temporal_offsets=(0,),
            teacher_spatial_offsets=(-1, 0, 1),
            teacher_generate_threshold=0.01,
            max_generate_fraction_per_phase=0.12,
        )
        oracle = spt.build_oracle_plan(source, target, config, feature_channels=4)
        per_phase = oracle.gate_probs[:, spt.GATE_GENERATE].mean(dim=(-2, -1))
        self.assertLessEqual(float(per_phase.max()), 0.12)
        self.assertAlmostEqual(oracle.diagnostics["prebudget_generate_fraction"], 1.0)
        self.assertAlmostEqual(oracle.diagnostics["budget_reject_fraction"], 14.0 / 15.0)
        self.assertAlmostEqual(
            oracle.diagnostics["observed_max_postbudget_generate_fraction_per_phase"],
            1.0 / 15.0,
        )
        self.assertEqual(float(oracle.gate_probs[:, spt.GATE_TRANSPORT].sum()), 0.0)
        self.assertAlmostEqual(
            float(oracle.gate_probs[:, spt.GATE_PRESERVE].mean()), 14.0 / 15.0
        )

    def test_cycle_proxy_detects_non_closing_reverse_correspondence(self) -> None:
        config = spt.PhaseTransportConfig(
            latent_channels=2,
            text_channels=3,
            hidden_channels=8,
            teacher_temporal_offsets=(0,),
            teacher_spatial_offsets=(-1, 0, 1),
        )
        candidates = spt._candidate_grid(config)
        forward_index = candidates.index((0, 0, 1))
        zero_index = candidates.index((0, 0, 0))
        selected = torch.full((1, 21, 3, 5), forward_index, dtype=torch.long)
        reverse = torch.full((1, 21, 3, 5), zero_index, dtype=torch.long)
        reference = self._video(channels=2)
        valid = torch.stack(
            [spt._valid_mask_for_offset(reference, candidate) for candidate in candidates],
            dim=1,
        )
        cycle = spt._cycle_consistency_for_selected(
            selected_index=selected,
            reverse_best_index=reverse,
            candidates=candidates,
            forward_valid=valid,
        )
        self.assertEqual(float(cycle.float().mean()), 0.0)

    def test_student_plan_is_dense_21_phase_and_source_text_only(self) -> None:
        config = spt.PhaseTransportConfig(
            latent_channels=4, text_channels=6, hidden_channels=8
        )
        adapter = spt.PhaseTransportAdapter(config)
        source = self._video(channels=4)
        instruction = torch.randn(1, 7, 6)
        plan = adapter(source, instruction)
        self.assertEqual(tuple(plan.offsets.shape), (1, 3, 21, 3, 5))
        self.assertEqual(tuple(plan.gate_probs.shape), (1, 3, 21, 3, 5))
        self.assertTrue(torch.allclose(plan.gate_probs.sum(1), torch.ones(1, 21, 3, 5)))


if __name__ == "__main__":
    unittest.main()

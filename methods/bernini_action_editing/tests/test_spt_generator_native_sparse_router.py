from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import tri_branch_unipc as tri  # noqa: E402
from spt_v2 import generator_native_sparse_router as router  # noqa: E402
from spt_v2 import phase_transport as spt  # noqa: E402


class PureGeneratorNativeSparseRouterTests(unittest.TestCase):
    def test_contract_has_no_train_only_or_external_spatial_condition(self) -> None:
        contract = router.runtime_contract()
        self.assertEqual(
            contract["same_state_input"],
            "raw_action_condition_clean_minus_raw_noop_condition_clean",
        )
        self.assertEqual(
            contract["official_apg_role"],
            "parity_certificate_only_not_routed_delta",
        )
        self.assertEqual(contract["generate_fraction_hard_cap"], 0.12)
        self.assertEqual(contract["generate_gate_application_count"], 1)
        self.assertEqual(contract["outside_generate_support"], "bit_exact_source_phase_tensor")
        self.assertEqual(
            contract["external_inference_conditions"],
            ["source_video", "action_instruction"],
        )
        self.assertIn("semantic_noop_instruction", contract["internal_fixed_controls"])
        self.assertEqual(contract["denoise_support_memory"], "causal_saliency_ema")
        self.assertEqual(
            router.GeneratorNativeSparseRouterConfig().static_delta_retention,
            0.0,
        )
        self.assertFalse(contract["learned_parameters"])
        self.assertTrue(
            {
                "target_video",
                "paired_target",
                "mask",
                "track",
                "pose",
                "optical_flow",
                "trajectory",
                "first_frame_anchor",
            }
            <= set(contract["forbidden_conditions"])
        )

    def test_public_inference_apis_cannot_receive_train_time_hints(self) -> None:
        forbidden = {
            "target",
            "target_video",
            "paired_target",
            "mask",
            "track",
            "pose",
            "flow",
            "trajectory",
            "anchor",
        }
        callables = (
            router.generator_native_motion_saliency,
            router.generator_native_phase_plan,
            router.execute_generator_native_sparse_clean,
            router.GeneratorNativeSparseCleanCallback.__init__,
            router.GeneratorNativeSparseCleanCallback.__call__,
        )
        for function in callables:
            with self.subTest(function=function.__qualname__):
                self.assertTrue(
                    forbidden.isdisjoint(inspect.signature(function).parameters)
                )

    def test_config_rejects_any_generate_cap_above_twelve_percent(self) -> None:
        with self.assertRaisesRegex(router.GeneratorNativeSparseRouterError, "0.12"):
            router.GeneratorNativeSparseRouterConfig(
                max_generate_fraction_per_phase=0.1200001
            ).validate()
        for changed in (
            {"static_delta_retention": -0.01},
            {"static_delta_retention": 1.01},
            {"denoise_saliency_ema_decay": -0.01},
            {"denoise_saliency_ema_decay": 1.0},
        ):
            with self.subTest(changed=changed), self.assertRaises(
                router.GeneratorNativeSparseRouterError
            ):
                router.GeneratorNativeSparseRouterConfig(**changed).validate()


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorGeneratorNativeSparseRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = tri.PackedLatentLayout.from_spatial_shape((1, 16, 21, 10, 10))
        self.phase_shape = (1, 21, 5, 5, 64)

    def _phase(self, value: float = 0.0) -> "torch.Tensor":
        return torch.full(self.phase_shape, value, dtype=torch.float32)

    def _fields(
        self, action_phase: "torch.Tensor", noop_phase: "torch.Tensor"
    ) -> tri.CleanFieldStep:
        action = router.phase_video_to_spatial(action_phase, layout=self.layout)
        noop = router.phase_video_to_spatial(noop_phase, layout=self.layout)
        delta = router.phase_video_to_spatial(
            action_phase - noop_phase, layout=self.layout
        )
        zeros = torch.zeros_like(action)
        return tri.CleanFieldStep(
            step_index=0,
            timestep=900.0,
            sigma=0.9,
            model_id="transformer_1",
            noisy=zeros,
            negative_velocity=zeros,
            action_velocity=zeros,
            noop_velocity=zeros,
            negative_clean=zeros,
            action_condition_clean=action,
            noop_condition_clean=noop,
            action_guided_clean=action,
            noop_guided_clean=noop,
            action_delta_clean=delta,
        )

    def _sensitive_config(self, **overrides) -> router.GeneratorNativeSparseRouterConfig:
        values = {
            "activity_energy_floor": 0.0,
            "relative_phase_activity_floor": 0.0,
            "energy_coverage": 1.0,
        }
        values.update(overrides)
        return router.GeneratorNativeSparseRouterConfig(**values)

    def test_spatial_packed_phase_roundtrip_is_exact(self) -> None:
        torch.manual_seed(7)
        layout = tri.PackedLatentLayout.from_spatial_shape((2, 16, 21, 10, 10))
        spatial = torch.randn(2, 16, 21, 10, 10)
        packed = router.spatial_to_packed(spatial, layout=layout)
        phase = router.packed_to_phase_video(packed, layout=layout)
        self.assertEqual(tuple(packed.shape), (2, 21 * 5 * 5, 64))
        self.assertEqual(tuple(phase.shape), (2, 21, 5, 5, 64))
        self.assertTrue(
            torch.equal(
                router.phase_video_to_packed(phase, layout=layout), packed
            )
        )
        self.assertTrue(
            torch.equal(router.packed_to_spatial(packed, layout=layout), spatial)
        )
        self.assertTrue(
            torch.equal(router.phase_video_to_spatial(phase, layout=layout), spatial)
        )
        for source in (spatial, packed, phase):
            with self.subTest(source_ndim=source.ndim):
                self.assertTrue(
                    torch.equal(
                        router.source_to_phase_video(source, layout=layout),
                        phase,
                    )
                )

    def test_action_equal_noop_abstains_with_exact_k_zero(self) -> None:
        action = torch.randn(self.phase_shape)
        delta = action - action.clone()
        plan = router.generator_native_phase_plan(delta)
        self.assertEqual(plan.provenance, "student")
        self.assertEqual(
            int(plan.gate_probs[:, spt.GATE_GENERATE].sum().item()), 0
        )
        self.assertTrue(
            torch.equal(
                plan.gate_probs[:, spt.GATE_PRESERVE],
                torch.ones_like(plan.gate_probs[:, spt.GATE_PRESERVE]),
            )
        )
        self.assertTrue(
            torch.equal(
                plan.diagnostics["selected_counts"],
                torch.zeros_like(plan.diagnostics["selected_counts"]),
            )
        )

    def test_temporally_constant_appearance_delta_is_removed_by_dc(self) -> None:
        delta = self._phase()
        delta[:, :, 2, 3, :] = 4.0
        saliency = router.generator_native_motion_saliency(delta)
        self.assertTrue(torch.equal(saliency, torch.zeros_like(saliency)))
        plan = router.generator_native_phase_plan(delta)
        self.assertEqual(
            int(plan.gate_probs[:, spt.GATE_GENERATE].sum().item()), 0
        )
        quotient = router.temporal_static_quotient(delta)
        self.assertTrue(torch.equal(quotient, torch.zeros_like(quotient)))
        retained = router.temporal_static_quotient(
            delta, static_delta_retention=0.25
        )
        self.assertTrue(torch.equal(retained, 0.25 * delta))

    def test_causal_boundary_projection_preserves_a_persistent_action(self) -> None:
        # Causal-boundary LoRA uses d(t)-d(0), not a temporal mean and not a
        # softly calibrated raw field.  A persistent action is neither weakened
        # after onset nor turned into a negative pre-action ghost.
        delta = self._phase()
        delta += 1.25  # arbitrary time-constant appearance offset
        delta[:, 10:, 2, 3, :] = 4.0
        represented = router.causal_boundary_projection(delta)
        self.assertTrue(
            torch.equal(represented[:, :10], torch.zeros_like(represented[:, :10]))
        )
        expected_terminal = torch.zeros_like(represented[:, 10:])
        expected_terminal[:, :, 2, 3, :] = 2.75
        self.assertTrue(
            torch.equal(represented[:, 10:], expected_terminal)
        )

    def test_causal_ema_projection_is_decay_half_then_exact_q0(self) -> None:
        # EMA is retained only as an explicit ablation.  The formal v4 path
        # uses exact Q0 because EMA attenuates and delays an action onset.
        self.assertIn("causal_ema_boundary_projection", router.__all__)
        delta = self._phase(1.25)
        delta[:, 10:, 2, 3, :] = 5.25
        represented = router.causal_ema_boundary_projection(delta, decay=0.5)

        self.assertTrue(
            torch.equal(represented[:, :10], torch.zeros_like(represented[:, :10]))
        )
        self.assertTrue(
            torch.equal(
                represented[:, 10, 2, 3, :],
                torch.full_like(represented[:, 10, 2, 3, :], 2.0),
            )
        )
        self.assertTrue(
            torch.equal(
                represented[:, 11, 2, 3, :],
                torch.full_like(represented[:, 11, 2, 3, :], 3.0),
            )
        )
        self.assertTrue(
            torch.equal(
                represented[:, 12, 2, 3, :],
                torch.full_like(represented[:, 12, 2, 3, :], 3.5),
            )
        )
        self.assertTrue(
            torch.equal(represented[:, :1], torch.zeros_like(represented[:, :1]))
        )
        with self.assertRaisesRegex(
            router.GeneratorNativeSparseRouterError, "decay"
        ):
            router.causal_ema_boundary_projection(delta, decay=1.0)

    def test_exact_cap_and_canonical_tie_breaking(self) -> None:
        # Every spatial cell has the same non-DC temporal signal.  Stable
        # row-major sorting must choose exactly floor(.12*25)=3 cells.
        temporal = (torch.arange(21, dtype=torch.float32) % 2).reshape(1, 21, 1, 1, 1)
        delta = temporal.expand(self.phase_shape).clone()
        plan = router.generator_native_phase_plan(
            delta, config=self._sensitive_config()
        )
        generate = plan.gate_probs[:, spt.GATE_GENERATE].bool().reshape(1, 21, 25)
        self.assertTrue(torch.equal(generate.sum(dim=-1), torch.full((1, 21), 3)))
        expected = torch.zeros_like(generate)
        expected[..., :3] = True
        self.assertTrue(torch.equal(generate, expected))
        self.assertLessEqual(3 / 25, 0.12)
        self.assertTrue(
            torch.equal(
                plan.gate_probs[:, spt.GATE_TRANSPORT],
                torch.zeros_like(plan.gate_probs[:, spt.GATE_TRANSPORT]),
            )
        )
        self.assertTrue(
            torch.equal(
                plan.gate_probs[:, spt.GATE_PRESERVE]
                + plan.gate_probs[:, spt.GATE_GENERATE],
                torch.ones_like(plan.gate_probs[:, spt.GATE_PRESERVE]),
            )
        )

    def test_multilag_energy_tracks_a_moving_support(self) -> None:
        delta = self._phase()
        moving = {8: (2, 1), 9: (2, 2), 10: (2, 3)}
        for phase, (row, column) in moving.items():
            delta[:, phase, row, column, :] = 3.0
        plan = router.generator_native_phase_plan(
            delta, config=self._sensitive_config(energy_coverage=0.8)
        )
        generate = plan.gate_probs[:, spt.GATE_GENERATE].bool()
        for phase, (row, column) in moving.items():
            with self.subTest(phase=phase):
                self.assertTrue(bool(generate[0, phase, row, column]))
        unrelated = generate.clone()
        for row, column in moving.values():
            unrelated[:, :, row, column] = False
        self.assertFalse(bool(unrelated.any()))

    def test_bridge_preserves_outside_exactly_and_generates_source_plus_delta(self) -> None:
        torch.manual_seed(11)
        source = torch.randn(self.phase_shape)
        noop = torch.randn(self.phase_shape)
        delta = self._phase()
        for phase, column in ((8, 1), (9, 2), (10, 3)):
            delta[:, phase, 2, column, :] = 4.0
        action = noop + delta
        fields = self._fields(action, noop)
        execution = router.execute_generator_native_sparse_clean(
            fields,
            source_clean=source,
            layout=self.layout,
            config=self._sensitive_config(energy_coverage=0.8),
            alpha=0.5,
        )
        support = execution.plan.gate_probs[:, spt.GATE_GENERATE].bool().unsqueeze(-1)
        outside = (~support).expand_as(source)
        inside = support.expand_as(source)
        self.assertTrue(
            torch.equal(execution.executed_clean_phase[outside], source[outside])
        )
        # Execution uses exactly the quotient/multilag representation trained
        # by the primary LoRA arm; the unsupervised temporal DC is not injected.
        represented = router.temporal_static_quotient(action - noop)
        expected_counterfactual = source + 0.5 * represented
        self.assertTrue(
            torch.equal(
                execution.executed_clean_phase[inside],
                expected_counterfactual[inside],
            )
        )
        self.assertGreater(int(support.sum().item()), 0)
        self.assertEqual(router.runtime_contract()["generate_gate_application_count"], 1)
        self.assertTrue(
            torch.equal(
                router.spatial_to_phase_video(
                    execution.executed_clean_spatial, layout=self.layout
                ),
                execution.executed_clean_phase,
            )
        )

    def test_callback_is_a_clean_field_step_bridge_and_records_last_plan(self) -> None:
        source = torch.randn(self.phase_shape)
        noop = torch.randn(self.phase_shape)
        action = noop.clone()
        fields = self._fields(action, noop)
        callback = router.GeneratorNativeSparseCleanCallback(
            source_clean=router.phase_video_to_packed(source, layout=self.layout),
            layout=self.layout,
        )
        actual = callback(fields)
        self.assertIsNotNone(callback.last_execution)
        self.assertTrue(
            torch.equal(
                router.spatial_to_phase_video(actual, layout=self.layout), source
            )
        )
        self.assertEqual(
            int(
                callback.last_execution.plan.gate_probs[
                    :, spt.GATE_GENERATE
                ].sum().item()
            ),
            0,
        )

    def test_callback_uses_causal_denoising_step_saliency_ema(self) -> None:
        source = torch.randn(self.phase_shape)
        noop = torch.zeros(self.phase_shape)
        first_delta = self._phase()
        first_delta[:, 8:11, 2, 1, :] = 3.0
        second_delta = self._phase()
        second_delta[:, 8:11, 2, 4, :] = 3.0
        config = self._sensitive_config(denoise_saliency_ema_decay=0.75)
        callback = router.GeneratorNativeSparseCleanCallback(
            source_clean=source,
            layout=self.layout,
            config=config,
        )
        callback(self._fields(noop + first_delta, noop))
        first = callback.last_execution.plan.diagnostics["motion_saliency"].clone()
        callback(self._fields(noop + second_delta, noop))
        diagnostics = callback.last_execution.plan.diagnostics
        raw_second = diagnostics["raw_motion_saliency"]
        expected = 0.75 * first + 0.25 * raw_second
        self.assertTrue(torch.allclose(diagnostics["motion_saliency"], expected))
        self.assertFalse(torch.equal(diagnostics["motion_saliency"], raw_second))


if __name__ == "__main__":
    unittest.main()

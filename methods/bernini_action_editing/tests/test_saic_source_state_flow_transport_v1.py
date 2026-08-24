#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import saic_source_state_flow_transport_v1 as flow  # noqa: E402


class SourceStateFlowTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (1, 16, 21, 2, 2)
        self.source = torch.ones(self.shape, dtype=torch.float32)
        self.edit = self.source.clone()
        self.noise = torch.zeros(self.shape, dtype=torch.float32)
        self.native = flow.NativeGuidanceBinding(
            model_id="Fudan-FUXI/Bernini-R-1.3B-Diffusers",
            checkpoint_sha256="a" * 64,
            negative_prompt_sha256="b" * 64,
            field_regime="t2v_apg",
            guidance_mode="t2v_apg",
            guidance_contract_sha256="d" * 64,
        )

    def config(
        self,
        *,
        anc: bool = True,
        anchor: bool = False,
        candidate_schedule=None,
        aggregation_mode=None,
        temperature="default",
        noise_schedule=None,
        noise_digest=None,
        native=None,
    ):
        schedule = candidate_schedule or flow.EXACT_CANDIDATE_SCHEDULE
        mode = aggregation_mode or (
            "uniform" if schedule == (1,) * 40 else "source_similarity_softmax"
        )
        if temperature == "default":
            tau = None if mode == "uniform" else 0.01
        else:
            tau = temperature
        if noise_digest is None:
            noise_digest = (
                flow.noise_bank_sha256(
                    noise_schedule, candidate_schedule=schedule
                )
                if noise_schedule is not None
                else "c" * 64
            )
        return flow.FlowTransportRolloutConfig(
            native=self.native if native is None else native,
            anc_enabled=anc,
            noise_generator_id="torch.Generator(device=cpu);native-adapter-v1",
            master_seed=2027,
            noise_bank_sha256=noise_digest,
            sigma_schedule=self.sigmas(),
            candidate_schedule=schedule,
            aggregation_mode=mode,
            temperature=tau,
            anchor_latent_phase_zero=anchor,
        )

    @staticmethod
    def pointer(value):
        if hasattr(value, "untyped_storage"):
            return value.untyped_storage().data_ptr()
        return value.storage().data_ptr()

    def sigmas(self):
        return tuple((40 - i) / 40 for i in range(41))

    def noise_schedule(self, candidate_schedule=None):
        schedule = candidate_schedule or flow.EXACT_CANDIDATE_SCHEDULE
        return tuple(
            tuple(
                torch.full(self.shape, float(step * 10 + candidate), dtype=torch.float32)
                for candidate in range(schedule[step])
            )
            for step in range(flow.EXPECTED_STEPS)
        )

    def step(
        self,
        query,
        *,
        step_index=0,
        config=None,
        edit=None,
        source_caption="A dog is standing in a room.",
        target_caption="The same dog sits down in the same room.",
        previous=None,
        fresh=None,
    ):
        runtime_config = config or self.config()
        if fresh is None:
            count = runtime_config.candidate_schedule[step_index]
            fresh = tuple(self.noise.clone() for _ in range(count))
        sigma = (40 - step_index) / 40
        next_sigma = (39 - step_index) / 40
        return flow.source_state_flow_step(
            config=runtime_config,
            step_index=step_index,
            source_clean=self.source,
            edit_clean=self.edit if edit is None else edit,
            source_caption=source_caption,
            target_caption=target_caption,
            sigma=sigma,
            next_sigma=next_sigma,
            time=sigma,
            next_time=next_sigma,
            fresh_noises=fresh,
            previous_noises=previous,
            velocity_query=query,
        )

    def test_registered_candidate_schedule_is_exact40_and_closed(self) -> None:
        self.assertEqual(len(flow.EXACT_CANDIDATE_SCHEDULE), 40)
        self.assertEqual(flow.EXACT_CANDIDATE_SCHEDULE[:5], (5, 5, 5, 1, 1))
        self.assertEqual(sum(flow.EXACT_CANDIDATE_SCHEDULE), 52)
        self.assertEqual(flow.EXACT_SINGLE_CANDIDATE_SCHEDULE, (1,) * 40)
        self.assertEqual(
            tuple(flow.candidate_count_for_step(i) for i in range(40)),
            flow.EXACT_CANDIDATE_SCHEDULE,
        )
        for invalid in (-1, 40, True):
            with self.assertRaises(flow.SAICSourceStateFlowTransportError):
                flow.candidate_count_for_step(invalid)
        self.assertEqual(
            tuple(
                flow.candidate_count_for_step(
                    i,
                    candidate_schedule=flow.EXACT_SINGLE_CANDIDATE_SCHEDULE,
                )
                for i in range(40)
            ),
            (1,) * 40,
        )

    def test_immutable_config_pins_apg_and_native_noise_closure(self) -> None:
        config = self.config()
        config.validate()
        with self.assertRaises(FrozenInstanceError):
            config.master_seed = 7
        with self.assertRaises(flow.SAICSourceStateFlowTransportError):
            flow.FlowTransportRolloutConfig(
                native=self.native,
                anc_enabled=True,
                noise_generator_id="generator",
                master_seed=2027,
                noise_bank_sha256="c" * 64,
                sigma_schedule=self.sigmas(),
                candidate_schedule=(2,) * 40,
                aggregation_mode="source_similarity_softmax",
                temperature=0.01,
            ).validate()
        self.config(
            candidate_schedule=flow.EXACT_SINGLE_CANDIDATE_SCHEDULE,
            aggregation_mode="uniform",
            temperature=None,
        ).validate()
        with self.assertRaisesRegex(
            flow.SAICSourceStateFlowTransportError,
            "uniform aggregation requires",
        ):
            self.config(aggregation_mode="uniform", temperature=0.01).validate()
        for regime, guidance in flow.GUIDANCE_MODE_BY_FIELD_REGIME.items():
            native_kwargs = {}
            if regime == "r2v_apg_source_i0":
                native_kwargs = {
                    "image_guidance_scale": 4.5,
                    "guidance_chain_scales": (4.5, 4.0),
                    "apg_norm_thresholds": (50.0, 50.0),
                    "apg_momenta": (0.0, 0.0),
                    "branch_order": flow.EXPECTED_R2V_I0_BRANCH_ORDER,
                    "raw_transformer_forwards_per_candidate": 6,
                }
            binding = flow.NativeGuidanceBinding(
                model_id="model",
                checkpoint_sha256="a" * 64,
                negative_prompt_sha256="b" * 64,
                field_regime=regime,
                guidance_mode=guidance,
                guidance_contract_sha256="d" * 64,
                **native_kwargs,
            ).validate()
            self.assertEqual(
                binding.visual_condition_scope,
                flow.VISUAL_CONDITION_SCOPE_BY_FIELD_REGIME[regime],
            )
        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "guidance_mode"):
            flow.NativeGuidanceBinding(
                model_id="model",
                checkpoint_sha256="a" * 64,
                negative_prompt_sha256="b" * 64,
                field_regime="r2v_apg_source_i0",
                guidance_mode="t2v_apg",
                guidance_contract_sha256="d" * 64,
            ).validate()

    def test_true_r2v_i0_chain_binds_three_raw_forwards_per_guided_query(self) -> None:
        native = flow.NativeGuidanceBinding(
            model_id="model",
            checkpoint_sha256="a" * 64,
            negative_prompt_sha256="b" * 64,
            field_regime="r2v_apg_source_i0",
            guidance_mode="r2v_apg",
            guidance_contract_sha256="d" * 64,
            image_guidance_scale=4.5,
            guidance_chain_scales=(4.5, 4.0),
            apg_norm_thresholds=(50.0, 50.0),
            apg_momenta=(0.0, 0.0),
            branch_order=flow.EXPECTED_R2V_I0_BRANCH_ORDER,
            raw_transformer_forwards_per_candidate=6,
        ).validate()
        requests = []

        result = self.step(
            lambda request: requests.append(request) or torch.zeros_like(request.state),
            config=self.config(anc=False, native=native),
        )
        self.assertEqual(len(requests), 10)
        self.assertTrue(
            all(request.expected_raw_transformer_forwards == 3 for request in requests)
        )
        self.assertTrue(
            all(request.step.raw_transformer_forwards_per_candidate == 6 for request in requests)
        )
        self.assertEqual(result.diagnostics.guided_velocity_query_count, 10)
        self.assertEqual(result.diagnostics.raw_transformer_forward_count, 30)
        self.assertFalse(result.diagnostics.raw_transformer_forward_count_verified)

        schedule = flow.EXACT_SINGLE_CANDIDATE_SCHEDULE
        noises = self.noise_schedule(schedule)
        rollout = flow.run_exact40_source_state_flow_transport(
            config=self.config(
                anc=False,
                candidate_schedule=schedule,
                aggregation_mode="uniform",
                temperature=None,
                noise_schedule=noises,
                native=native,
            ),
            source_clean=self.source,
            source_caption="A dog stands.",
            target_caption="The dog sits.",
            sigma_schedule=self.sigmas(),
            fresh_noise_schedule=noises,
            velocity_query=lambda request: torch.zeros_like(request.state),
        )
        self.assertEqual(rollout.diagnostics.guided_velocity_query_count, 80)
        self.assertEqual(rollout.diagnostics.raw_transformer_forward_count, 240)
        self.assertEqual(rollout.diagnostics.image_guidance_scale, 4.5)
        self.assertEqual(
            rollout.diagnostics.guidance_chain_scales, (4.5, 4.0)
        )
        self.assertEqual(
            rollout.diagnostics.apg_norm_thresholds, (50.0, 50.0)
        )
        self.assertEqual(rollout.diagnostics.apg_momenta, (0.0, 0.0))
        self.assertEqual(
            rollout.diagnostics.branch_order,
            flow.EXPECTED_R2V_I0_BRANCH_ORDER,
        )

        with self.assertRaisesRegex(
            flow.SAICSourceStateFlowTransportError,
            "image_guidance_scale",
        ):
            flow.NativeGuidanceBinding(
                model_id="model",
                checkpoint_sha256="a" * 64,
                negative_prompt_sha256="b" * 64,
                field_regime="r2v_apg_source_i0",
                guidance_mode="r2v_apg",
                guidance_contract_sha256="d" * 64,
            ).validate()
        with self.assertRaises(flow.SAICSourceStateFlowTransportError):
            flow.NativeGuidanceBinding(
                model_id="model",
                checkpoint_sha256="a" * 64,
                negative_prompt_sha256="b" * 64,
                field_regime="t2v_apg",
                guidance_mode="t2v_apg",
                guidance_contract_sha256="d" * 64,
                raw_transformer_forwards_per_candidate=2,
            ).validate()

    def test_anc_schedule_and_owned_noise_outputs(self) -> None:
        self.assertEqual(flow.anc_retention(1.0), 0.0)
        self.assertEqual(flow.anc_retention(0.25), 1.0)
        self.assertEqual(flow.anc_retention(0.0), 1.0)
        self.assertAlmostEqual(flow.anc_retention(0.625), 0.5)
        fresh = torch.full(self.shape, 2.0, dtype=torch.float32)
        previous = torch.full(self.shape, 4.0, dtype=torch.float32)
        first = flow.correlate_noise(fresh, previous_noise=None, retention=0.0)
        locked = flow.correlate_noise(fresh, previous_noise=previous, retention=1.0)
        self.assertTrue(torch.equal(first, fresh))
        self.assertTrue(torch.equal(locked, previous))
        self.assertNotEqual(self.pointer(first), self.pointer(fresh))
        self.assertNotEqual(self.pointer(locked), self.pointer(previous))
        mixed = flow.correlate_noise(fresh, previous_noise=previous, retention=0.5)
        expected = (0.5**0.5) * previous + (0.5**0.5) * fresh
        self.assertTrue(torch.equal(mixed, expected))

    def test_query_state_algebra_and_maximum_noise_identity(self) -> None:
        source_state, target_state = flow.build_source_target_query_states(
            self.source, self.edit, self.noise, time=1.0
        )
        self.assertTrue(torch.equal(source_state, self.noise))
        self.assertTrue(torch.equal(target_state, self.noise))
        changed_edit = self.edit + 0.25
        source_state, target_state = flow.build_source_target_query_states(
            self.source, changed_edit, self.noise, time=0.5
        )
        self.assertTrue(torch.equal(target_state - source_state, changed_edit - self.source))

    def test_similarity_guidance_prefers_source_aligned_projection(self) -> None:
        aligned = torch.zeros_like(self.source)
        opposed = torch.full_like(self.source, 2.0)
        aggregate, similarities, weights = flow.similarity_guided_delta(
            self.source,
            self.edit,
            (aligned, opposed),
            time=1.0,
            temperature=0.1,
        )
        self.assertAlmostEqual(similarities[0], 1.0, places=6)
        self.assertAlmostEqual(similarities[1], -1.0, places=6)
        self.assertGreater(weights[0], 0.999)
        self.assertLess(float(aggregate.abs().max().item()), 0.001)

    def test_callback_request_binds_step_schedule_model_guidance_and_counts(self) -> None:
        requests = []

        def query(request):
            requests.append(request)
            return torch.zeros_like(request.state)

        result = self.step(query, config=self.config(anc=False))
        self.assertEqual(len(requests), 10)
        self.assertEqual([item.role for item in requests], ["target", "source"] * 5)
        first = requests[0]
        self.assertIsInstance(first, flow.VelocityQueryRequest)
        self.assertEqual(first.step.step_index, 0)
        self.assertEqual(first.step.candidate_schedule, flow.EXACT_CANDIDATE_SCHEDULE)
        self.assertEqual(first.step.candidate_count, 5)
        self.assertFalse(first.step.anc_enabled)
        self.assertEqual(first.step.candidate_continuation, "candidate_zero")
        self.assertEqual(first.step.sigma, 1.0)
        self.assertEqual(first.step.next_sigma, 0.975)
        self.assertEqual(first.step.time, first.step.sigma)
        self.assertEqual(first.step.native.model_id, self.native.model_id)
        self.assertEqual(first.step.native.field_regime, "t2v_apg")
        self.assertEqual(first.step.native.guidance_mode, "t2v_apg")
        self.assertEqual(first.expected_raw_transformer_forwards, 2)
        self.assertEqual(result.diagnostics.guided_velocity_query_count, 10)
        self.assertEqual(result.diagnostics.raw_transformer_forward_count, 20)
        self.assertFalse(result.diagnostics.optimizer_step_allowed)
        self.assertFalse(result.diagnostics.training_update_allowed)
        self.assertFalse(result.diagnostics.semantic_action_success)

    def test_callback_inputs_are_cloned_and_return_velocity_is_owned(self) -> None:
        seen_pointers = []

        def query(request):
            seen_pointers.append(self.pointer(request.state))
            return torch.zeros_like(request.state)

        result = self.step(query, config=self.config(anc=False))
        source_pointer = self.pointer(self.source)
        self.assertTrue(all(pointer != source_pointer for pointer in seen_pointers))
        self.assertNotEqual(
            self.pointer(result.edit_clean),
            self.pointer(self.edit),
        )

    def test_callback_mutation_alias_and_core_mutation_fail_closed(self) -> None:
        def mutates_request(request):
            request.state.add_(1.0)
            return torch.zeros_like(request.state)

        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "mutated"):
            self.step(mutates_request, config=self.config(anc=False))

        def returns_alias(request):
            return request.state

        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "alias"):
            self.step(returns_alias, config=self.config(anc=False))

        def returns_alias_view(request):
            return request.state.view_as(request.state)

        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "alias"):
            self.step(returns_alias_view, config=self.config(anc=False))

        def mutates_closed_over_source(request):
            self.source.add_(1.0)
            return torch.zeros_like(request.state)

        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "protected"):
            self.step(mutates_closed_over_source, config=self.config(anc=False))

        reusable_config = self.config(anc=False)

        def mutates_binding(request):
            object.__setattr__(request.step.native, "guidance_mode", "v2v_apg")
            return torch.zeros_like(request.state)

        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "immutable request"):
            self.step(mutates_binding, config=reusable_config)
        self.assertEqual(reusable_config.native.guidance_mode, "t2v_apg")

    def test_data_mutation_of_caller_state_cannot_change_owned_core_state(self) -> None:
        original = self.source.clone()

        def mutates_external_data(request):
            self.source.data.fill_(99.0)
            return torch.zeros_like(request.state)

        result = self.step(mutates_external_data, config=self.config(anc=False))
        self.assertTrue(torch.equal(result.edit_clean, original))
        self.assertTrue(torch.equal(self.source, torch.full_like(self.source, 99.0)))
        # The callback is an in-process Python capability and can perform
        # arbitrary external side effects.  The core truthfully does not claim
        # it verified how the request was executed.
        self.assertFalse(result.diagnostics.native_request_execution_verified)
        self.assertFalse(result.diagnostics.model_checkpoint_use_verified)

    def test_core_executes_with_tensors_created_inside_inference_mode(self) -> None:
        config = self.config(
            anc=False,
            candidate_schedule=flow.EXACT_SINGLE_CANDIDATE_SCHEDULE,
            aggregation_mode="uniform",
            temperature=None,
        )
        with torch.inference_mode():
            source = torch.ones(self.shape, dtype=torch.float32)
            noise = (torch.zeros(self.shape, dtype=torch.float32),)
            result = flow.source_state_flow_step(
                config=config,
                step_index=0,
                source_clean=source,
                edit_clean=source.clone(),
                source_caption="A dog stands.",
                target_caption="The dog sits.",
                sigma=1.0,
                next_sigma=0.975,
                time=1.0,
                next_time=0.975,
                fresh_noises=noise,
                previous_noises=None,
                velocity_query=lambda request: torch.zeros_like(request.state),
            )
        self.assertTrue(torch.equal(result.edit_clean, torch.ones_like(result.edit_clean)))
        self.assertEqual(result.diagnostics.guided_velocity_query_count, 2)

    def test_public_global_rebinding_cannot_expand_registered_mechanism(self) -> None:
        saved = {
            name: getattr(flow, name)
            for name in (
                "REGISTERED_FIELD_REGIMES",
                "GUIDANCE_MODE_BY_FIELD_REGIME",
                "VISUAL_CONDITION_SCOPE_BY_FIELD_REGIME",
                "REGISTERED_AGGREGATION_MODES",
                "REGISTERED_CANDIDATE_SCHEDULES",
                "EXACT_CANDIDATE_SCHEDULE",
                "EXPECTED_STEPS",
                "EXPECTED_CHANNELS",
                "EXPECTED_PHASES",
                "EXPECTED_GUIDANCE_SCALE",
                "RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE",
                "NativeGuidanceBinding",
                "FlowTransportRolloutConfig",
                "FlowTransportStepBinding",
                "VelocityQueryRequest",
                "correlate_noise",
                "build_source_target_query_states",
                "similarity_guided_delta",
                "source_state_flow_step",
            )
        }
        noises = self.noise_schedule()
        config = self.config(anc=False, noise_schedule=noises)
        original_binding_type = saved["NativeGuidanceBinding"]
        try:
            flow.REGISTERED_FIELD_REGIMES = ("evil",)
            flow.GUIDANCE_MODE_BY_FIELD_REGIME = {"evil": "evil"}
            flow.VISUAL_CONDITION_SCOPE_BY_FIELD_REGIME = {"evil": "oracle"}
            flow.REGISTERED_AGGREGATION_MODES = ("evil",)
            flow.REGISTERED_CANDIDATE_SCHEDULES = ((9,) * 40,)
            flow.EXACT_CANDIDATE_SCHEDULE = (9,) * 40
            flow.EXPECTED_STEPS = 1
            flow.EXPECTED_CHANNELS = 1
            flow.EXPECTED_PHASES = 1
            flow.EXPECTED_GUIDANCE_SCALE = 999.0
            flow.RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE = 0
            flow.NativeGuidanceBinding = object
            flow.FlowTransportRolloutConfig = object
            flow.FlowTransportStepBinding = object
            flow.VelocityQueryRequest = object
            flow.correlate_noise = lambda *args, **kwargs: torch.full_like(args[0], 123.0)
            flow.build_source_target_query_states = lambda *args, **kwargs: (
                torch.full_like(args[0], 123.0),
                torch.full_like(args[0], 123.0),
            )
            flow.similarity_guided_delta = lambda *args, **kwargs: (
                torch.full_like(args[0], 123.0),
                (),
                (1.0,),
            )
            flow.source_state_flow_step = lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("rebound public source_state_flow_step was used")
            )

            with self.assertRaises(flow.SAICSourceStateFlowTransportError):
                original_binding_type(
                    model_id="model",
                    checkpoint_sha256="a" * 64,
                    negative_prompt_sha256="b" * 64,
                    field_regime="evil",
                    guidance_mode="evil",
                    guidance_contract_sha256="d" * 64,
                    guidance_scale=999.0,
                    raw_transformer_forwards_per_candidate=0,
                ).validate()

            result = flow.run_exact40_source_state_flow_transport(
                config=config,
                source_clean=self.source,
                source_caption="A dog stands.",
                target_caption="The dog sits.",
                sigma_schedule=self.sigmas(),
                fresh_noise_schedule=noises,
                velocity_query=lambda request: torch.zeros_like(request.state),
            )
            self.assertEqual(result.diagnostics.candidate_counts, (5, 5, 5) + (1,) * 37)
            self.assertEqual(result.diagnostics.guided_velocity_query_count, 104)
            self.assertTrue(torch.equal(result.edit_clean, self.source))
        finally:
            for name, value in saved.items():
                setattr(flow, name, value)

    def test_noop_advances_truthful_anc_and_does_not_report_anchor(self) -> None:
        def forbidden(_request):
            raise AssertionError("no-op must not query generator")

        first_fresh = tuple(torch.full(self.shape, 2.0) for _ in range(5))
        first = self.step(
            forbidden,
            source_caption="The dog remains unchanged.",
            target_caption="The dog remains unchanged.",
            fresh=first_fresh,
            config=self.config(anchor=True),
        )
        self.assertTrue(first.diagnostics.exact_caption_noop_bypass)
        self.assertFalse(first.diagnostics.latent_phase_zero_anchored)
        self.assertEqual(first.diagnostics.guided_velocity_query_count, 0)
        self.assertTrue(torch.equal(first.correlated_noises[0], first_fresh[0]))

        second_fresh = tuple(torch.full(self.shape, 4.0) for _ in range(5))
        second = self.step(
            forbidden,
            step_index=1,
            source_caption="The dog remains unchanged.",
            target_caption="The dog remains unchanged.",
            previous=first.correlated_noises,
            fresh=second_fresh,
            config=self.config(anchor=True),
        )
        retained = flow.anc_retention(39 / 40)
        expected = retained**0.5 * first.correlated_noises[0] + (1 - retained) ** 0.5 * second_fresh[0]
        self.assertTrue(torch.allclose(second.correlated_noises[0], expected))
        self.assertTrue(torch.equal(second.edit_clean, self.edit))

    def test_anc_off_rejects_state_and_anc_on_requires_registered_predecessor(self) -> None:
        query = lambda request: torch.zeros_like(request.state)
        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "must not provide"):
            self.step(
                query,
                config=self.config(anc=False),
                previous=tuple(self.noise.clone() for _ in range(5)),
            )
        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "requires previous"):
            self.step(query, step_index=1, config=self.config(anc=True))

    def test_full_exact40_k_transition_uses_candidate_zero_for_anc(self) -> None:
        recorded_step3_source_states = []

        def query(request):
            if request.step.step_index == 3 and request.role == "source":
                recorded_step3_source_states.append(request.state.clone())
            return torch.zeros_like(request.state)

        noises = list(self.noise_schedule())
        # Candidate zero is distinguishable from discarded candidates at step 2.
        noises[2] = tuple(torch.full(self.shape, float(i + 1)) for i in range(5))
        noises[3] = (torch.full(self.shape, 99.0),)
        noises = tuple(noises)
        result = flow.run_exact40_source_state_flow_transport(
            config=self.config(anc=True, noise_schedule=noises),
            source_clean=self.source,
            source_caption="A dog stands.",
            target_caption="The dog sits.",
            sigma_schedule=self.sigmas(),
            fresh_noise_schedule=noises,
            velocity_query=query,
        )
        self.assertEqual(result.diagnostics.candidate_counts, flow.EXACT_CANDIDATE_SCHEDULE)
        self.assertEqual(len(recorded_step3_source_states), 1)
        r1 = flow.anc_retention(39 / 40)
        r2 = flow.anc_retention(38 / 40)
        r3 = flow.anc_retention(37 / 40)
        candidate0_step0 = torch.zeros_like(self.source)
        candidate0_step1 = r1**0.5 * candidate0_step0 + (1 - r1) ** 0.5 * torch.full(self.shape, 10.0)
        candidate0_step2 = r2**0.5 * candidate0_step1 + (1 - r2) ** 0.5 * torch.full(self.shape, 1.0)
        candidate0_step3 = r3**0.5 * candidate0_step2 + (1 - r3) ** 0.5 * torch.full(self.shape, 99.0)
        expected_step3_source = (1 - 37 / 40) * self.source + (37 / 40) * candidate0_step3
        self.assertTrue(torch.allclose(recorded_step3_source_states[0], expected_step3_source))
        self.assertEqual(len(result.final_correlated_noises), 1)
        self.assertEqual(result.diagnostics.guided_velocity_query_count, 104)
        self.assertEqual(result.diagnostics.raw_transformer_forward_count, 208)
        self.assertFalse(result.diagnostics.noise_distribution_verified)
        self.assertEqual(result.diagnostics.candidate_continuation, "candidate_zero")

    def test_exact40_anc_off_arm_and_noop_arm_execute(self) -> None:
        calls = 0

        def query(request):
            nonlocal calls
            calls += 1
            return torch.zeros_like(request.state)

        noises = self.noise_schedule()
        result = flow.run_exact40_source_state_flow_transport(
            config=self.config(anc=False, noise_schedule=noises),
            source_clean=self.source,
            source_caption="A dog stands.",
            target_caption="The dog sits.",
            sigma_schedule=self.sigmas(),
            fresh_noise_schedule=noises,
            velocity_query=query,
        )
        self.assertEqual(calls, 104)
        self.assertFalse(result.diagnostics.anc_enabled)

        noop = flow.run_exact40_source_state_flow_transport(
            config=self.config(anc=True, noise_schedule=noises),
            source_clean=self.source,
            source_caption="No change.",
            target_caption="No change.",
            sigma_schedule=self.sigmas(),
            fresh_noise_schedule=noises,
            velocity_query=lambda _request: (_ for _ in ()).throw(AssertionError()),
        )
        self.assertTrue(torch.equal(noop.edit_clean, self.source))
        self.assertEqual(noop.diagnostics.guided_velocity_query_count, 0)
        self.assertEqual(noop.diagnostics.raw_transformer_forward_count, 0)
        self.assertTrue(all(item.exact_caption_noop_bypass for item in noop.diagnostics.step_diagnostics))

    def test_exact40_k1_uniform_arm_has_80_guided_and_160_raw_contract(self) -> None:
        schedule = flow.EXACT_SINGLE_CANDIDATE_SCHEDULE
        noises = self.noise_schedule(schedule)
        calls = 0

        def query(request):
            nonlocal calls
            calls += 1
            return torch.zeros_like(request.state)

        result = flow.run_exact40_source_state_flow_transport(
            config=self.config(
                anc=False,
                candidate_schedule=schedule,
                aggregation_mode="uniform",
                temperature=None,
                noise_schedule=noises,
            ),
            source_clean=self.source,
            source_caption="A dog stands.",
            target_caption="The dog sits.",
            sigma_schedule=self.sigmas(),
            fresh_noise_schedule=noises,
            velocity_query=query,
        )
        self.assertEqual(calls, 80)
        self.assertEqual(result.diagnostics.candidate_counts, (1,) * 40)
        self.assertEqual(result.diagnostics.guided_velocity_query_count, 80)
        self.assertEqual(result.diagnostics.raw_transformer_forward_count, 160)
        self.assertEqual(result.diagnostics.aggregation_mode, "uniform")
        self.assertIsNone(result.diagnostics.temperature)
        self.assertTrue(result.diagnostics.noise_bank_digest_verified)
        self.assertFalse(result.diagnostics.raw_transformer_forward_count_verified)
        self.assertFalse(result.diagnostics.native_request_execution_verified)
        self.assertFalse(result.diagnostics.model_checkpoint_use_verified)
        self.assertTrue(
            all(item.aggregation_weights == (1.0,) for item in result.diagnostics.step_diagnostics)
        )
        self.assertTrue(
            all(item.source_similarity_by_candidate == () for item in result.diagnostics.step_diagnostics)
        )

    def test_k5_uniform_is_exact_average_without_similarity_scoring(self) -> None:
        def query(request):
            if request.role == "source":
                return torch.zeros_like(request.state)
            return torch.full_like(request.state, float(request.candidate_index + 1))

        result = self.step(
            query,
            config=self.config(
                anc=False,
                aggregation_mode="uniform",
                temperature=None,
            ),
        )
        self.assertEqual(result.diagnostics.aggregation_mode, "uniform")
        self.assertIsNone(result.diagnostics.temperature)
        self.assertEqual(result.diagnostics.source_similarity_by_candidate, ())
        self.assertEqual(result.diagnostics.aggregation_weights, (0.2,) * 5)
        # Mean target-source delta is exactly 3; dt=-1/40.
        torch.testing.assert_close(
            result.edit_clean,
            self.edit - torch.full_like(self.edit, 3.0 / 40.0),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_sigma_and_actual_noise_bank_are_cryptographically_bound(self) -> None:
        noises = self.noise_schedule()
        config = self.config(anc=False, noise_schedule=noises)
        self.assertEqual(
            config.sigma_schedule_sha256,
            flow.sigma_schedule_sha256(self.sigmas()),
        )

        wrong_sigmas = tuple(((40 - i) / 40) ** 2 for i in range(41))
        with self.assertRaisesRegex(
            flow.SAICSourceStateFlowTransportError,
            "does not equal the config-registered",
        ):
            flow.run_exact40_source_state_flow_transport(
                config=config,
                source_clean=self.source,
                source_caption="A dog stands.",
                target_caption="The dog sits.",
                sigma_schedule=wrong_sigmas,
                fresh_noise_schedule=noises,
                velocity_query=lambda request: torch.zeros_like(request.state),
            )

    def test_noise_bank_digest_is_content_order_shape_and_dtype_canonical(self) -> None:
        noises = self.noise_schedule()
        cloned = tuple(tuple(item.clone() for item in cell) for cell in noises)
        digest = flow.noise_bank_sha256(noises)
        self.assertEqual(digest, flow.noise_bank_sha256(cloned))

        reordered = list(cloned)
        reordered[0] = (
            cloned[0][1],
            cloned[0][0],
            *cloned[0][2:],
        )
        self.assertNotEqual(digest, flow.noise_bank_sha256(tuple(reordered)))

        wrong_dtype = list(cloned)
        wrong_dtype[0] = (cloned[0][0].bfloat16(), *cloned[0][1:])
        with self.assertRaisesRegex(
            flow.SAICSourceStateFlowTransportError,
            "detached finite FP32",
        ):
            flow.noise_bank_sha256(tuple(wrong_dtype))

        wrong_shape = list(cloned)
        wrong_shape[0] = (
            torch.zeros((1, 16, 21, 1, 4), dtype=torch.float32),
            *cloned[0][1:],
        )
        with self.assertRaisesRegex(
            flow.SAICSourceStateFlowTransportError,
            "registered geometry",
        ):
            flow.noise_bank_sha256(tuple(wrong_shape))

        if torch.cuda.is_available():  # pragma: no cover - AUH-only assertion
            cuda_bank = tuple(
                tuple(item.to("cuda") for item in cell) for cell in noises
            )
            self.assertEqual(digest, flow.noise_bank_sha256(cuda_bank))

        config = self.config(anc=False, noise_schedule=noises)
        noises[0][0].add_(0.5)
        with self.assertRaisesRegex(
            flow.SAICSourceStateFlowTransportError,
            "actual ordered noise-bank digest",
        ):
            flow.run_exact40_source_state_flow_transport(
                config=config,
                source_clean=self.source,
                source_caption="A dog stands.",
                target_caption="The dog sits.",
                sigma_schedule=self.sigmas(),
                fresh_noise_schedule=noises,
                velocity_query=lambda request: torch.zeros_like(request.state),
            )

    def test_callback_cannot_silently_change_a_future_registered_noise_cell(self) -> None:
        noises = self.noise_schedule()
        config = self.config(anc=False, noise_schedule=noises)
        changed = False

        def query(request):
            nonlocal changed
            if not changed:
                # `.data` bypasses the ordinary tensor version counter.
                noises[5][0].data.add_(1.0)
                changed = True
            return torch.zeros_like(request.state)

        with self.assertRaisesRegex(
            flow.SAICSourceStateFlowTransportError,
            r"\[5\]\[0\] changed after digest registration",
        ):
            flow.run_exact40_source_state_flow_transport(
                config=config,
                source_clean=self.source,
                source_caption="A dog stands.",
                target_caption="The dog sits.",
                sigma_schedule=self.sigmas(),
                fresh_noise_schedule=noises,
                velocity_query=query,
            )

    def test_perfect_linear_field_has_correct_end_to_end_sign(self) -> None:
        def perfect_linear_field(request):
            # Homogeneous linear source field v_s(x)=x and v_t(x)=0.  With
            # zero noise, x_s(t)=(1-t)S.  Backward integration must add the
            # positive left-Riemann integral; a reversed sign subtracts it.
            if request.role == "source":
                return request.state.clone()
            return torch.zeros_like(request.state)

        zero_noise_schedule = tuple(
            tuple(torch.zeros(self.shape) for _ in range(flow.candidate_count_for_step(step)))
            for step in range(flow.EXPECTED_STEPS)
        )

        result = flow.run_exact40_source_state_flow_transport(
            config=self.config(
                anc=False,
                noise_schedule=zero_noise_schedule,
            ),
            source_clean=self.source,
            source_caption="A dog stands.",
            target_caption="The dog sits.",
            sigma_schedule=self.sigmas(),
            fresh_noise_schedule=zero_noise_schedule,
            velocity_query=perfect_linear_field,
        )
        expected_integral = sum((1 / 40) * (step / 40) for step in range(40))
        self.assertTrue(
            torch.allclose(
                result.edit_clean,
                self.source * (1 + expected_integral),
                atol=1e-5,
            )
        )

    def test_invalid_schedule_counts_and_unregistered_k_fail_closed(self) -> None:
        query = lambda request: torch.zeros_like(request.state)
        bad_sigmas = list(self.sigmas())
        bad_sigmas[-1] = 0.001
        noises = self.noise_schedule()
        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "exact zero"):
            flow.run_exact40_source_state_flow_transport(
                config=self.config(noise_schedule=noises),
                source_clean=self.source,
                source_caption="Source.",
                target_caption="Target.",
                sigma_schedule=bad_sigmas,
                fresh_noise_schedule=noises,
                velocity_query=query,
            )
        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "exactly 1"):
            self.step(
                query,
                step_index=3,
                config=self.config(anc=True),
                fresh=tuple(self.noise.clone() for _ in range(5)),
                previous=tuple(self.noise.clone() for _ in range(5)),
            )
        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "predecessor candidates"):
            self.step(
                query,
                step_index=3,
                config=self.config(anc=True),
                fresh=(self.noise.clone(),),
                previous=(self.noise.clone(),),
            )
        with self.assertRaisesRegex(
            flow.SAICSourceStateFlowTransportError,
            r"step_index must be an integer in \[0,39\]",
        ):
            self.step(query, step_index=40, config=self.config(anc=False), fresh=())

    def test_optional_phase_zero_anchor_is_exact_and_truthful(self) -> None:
        edit = self.edit.clone()
        edit[:, :, 0].fill_(7.0)

        def query(request):
            return torch.zeros_like(request.state) if request.role == "source" else torch.ones_like(request.state)

        result = self.step(query, config=self.config(anc=False, anchor=True), edit=edit)
        self.assertTrue(torch.equal(result.edit_clean[:, :, 0], self.source[:, :, 0]))
        self.assertFalse(torch.equal(result.edit_clean[:, :, 1], self.source[:, :, 1]))
        self.assertTrue(result.diagnostics.latent_phase_zero_anchor_requested)
        self.assertTrue(result.diagnostics.latent_phase_zero_anchored)

    def test_invalid_state_velocity_and_runtime_api_fail_closed(self) -> None:
        with self.assertRaises(flow.SAICSourceStateFlowTransportError):
            flow.build_source_target_query_states(self.source.bfloat16(), self.edit, self.noise, time=1.0)

        def wrong_velocity(request):
            return request.state[:, :, :-1].clone()

        with self.assertRaisesRegex(flow.SAICSourceStateFlowTransportError, "guided velocity"):
            self.step(wrong_velocity, config=self.config(anc=False))

        names = set(inspect.signature(flow.source_state_flow_step).parameters)
        self.assertTrue({"source_clean", "edit_clean", "velocity_query", "config", "step_index"}.issubset(names))
        self.assertTrue(
            names.isdisjoint(
                {"target_video", "proposal_video", "donor_video", "mask", "track", "pose", "flow", "trajectory"}
            )
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import pathlib
import inspect
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import anchor_sga_anc_controller as controller  # noqa: E402
import differential_sampler as cdf  # noqa: E402


class AnchorSGAANCControllerTest(unittest.TestCase):
    def test_native_t2v_hard_replacement_algebra_is_exact(self):
        target = torch.full((1, 84, 2), 10.0)
        source = torch.full_like(target, 2.0)
        action = torch.full_like(target, 7.0)
        noop = torch.full_like(target, 3.0)
        replaced_target = controller._apply_native_t2v_hard_replacement(
            target_velocity=target,
            source_velocity=source,
            action_velocity=action,
            noop_velocity=noop,
            transport=controller.FIELD_NATIVE_T2V_TARGET_VELOCITY_REPLACEMENT,
        )
        replaced_delta = controller._apply_native_t2v_hard_replacement(
            target_velocity=target,
            source_velocity=source,
            action_velocity=action,
            noop_velocity=noop,
            transport=controller.FIELD_NATIVE_T2V_DELTA_VELOCITY_REPLACEMENT,
        )
        self.assertTrue(torch.equal(replaced_target, action))
        self.assertTrue(torch.equal(replaced_delta, torch.full_like(target, 6.0)))
        self.assertTrue(torch.equal(replaced_delta - source, action - noop))

    def test_native_t2v_temporal_quotient_replacement_removes_static_basis(self):
        phases = controller.guided.EXPECTED_LATENT_PHASES
        spatial = 4
        target = torch.zeros((1, phases * spatial, 2))
        source = torch.full_like(target, 2.0)
        noop = torch.full_like(target, 3.0)
        action = noop.clone().reshape(1, phases, spatial, 2)
        action += 11.0
        action[:, 1:] += torch.arange(
            1, phases, dtype=action.dtype
        ).reshape(1, phases - 1, 1, 1)
        action = action.reshape_as(target)
        replaced = controller._apply_native_t2v_hard_replacement(
            target_velocity=target,
            source_velocity=source,
            action_velocity=action,
            noop_velocity=noop,
            transport=controller.FIELD_NATIVE_T2V_TEMPORAL_DELTA_REPLACEMENT,
        )
        quotient = (replaced - source).reshape(1, phases, spatial, 2)
        self.assertTrue(torch.equal(quotient[:, 0], torch.zeros_like(quotient[:, 0])))
        expected = torch.arange(1, phases, dtype=quotient.dtype)
        self.assertTrue(torch.equal(quotient[0, 1:, 0, 0], expected))
        self.assertTrue(torch.equal(quotient[0, 1:, 3, 1], expected))

    def test_sga_uses_low_temperature_and_weighted_anc_chain_collapse(self):
        config = controller.AnchorSGAANCConfig(arm="AQK_SGA5").validate()
        self.assertEqual(config.sga_temperature, 0.01)
        sharp = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5", sga_temperature=0.001
        ).validate()
        self.assertEqual(sharp.sga_temperature, 0.001)
        for count in controller.SUPPORTED_EARLY_CANDIDATES:
            expanded = controller.AnchorSGAANCConfig(
                arm="AQK_SGA5", early_candidate_count=count
            ).validate()
            self.assertEqual(expanded.candidate_count(0), count)
            self.assertEqual(expanded.candidate_count(3), 1)
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5", early_candidate_count=7
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_ANC1", early_candidate_count=8
            ).validate()
        for mode in controller.INITIAL_NOISE_PROPOSAL_MODES:
            seeded = controller.AnchorSGAANCConfig(
                arm="AQK_SGA5", initial_noise_proposal_mode=mode
            ).validate()
            self.assertEqual(seeded.initial_noise_proposal_mode, mode)
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_ANC1",
                initial_noise_proposal_mode="anchor_candidate0",
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5", sga_temperature=0.1
            ).validate()
        source = inspect.getsource(controller.sample_anchor_sga_anc)
        self.assertIn("collapse_sga_noise_chains", source)
        self.assertNotIn("[noise_bank[0]]", source)
        self.assertIn("anchor_initial_packed.clone()", source)
        self.assertIn('== "anchor_candidate0_forced"', source)

    def test_transport_steps_accepts_explicit_control_and_early_intervals(self):
        for steps in (0, 3, 8, 40):
            config = controller.AnchorSGAANCConfig(
                arm="AQK_ANC1", transport_steps=steps
            )
            self.assertIs(config.validate(), config)
        for invalid in (-1, 41, True, 3.5):
            with self.assertRaises(controller.AnchorSGAANCError):
                controller.AnchorSGAANCConfig(
                    arm="AQK_ANC1", transport_steps=invalid  # type: ignore[arg-type]
                ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_ANC1", initial_phase_clamp=1  # type: ignore[arg-type]
            ).validate()
        raw_cfg = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            field_model="first_phase_caption_i2v",
            field_guidance="raw_cfg",
            source_cfg_scale=2.5,
            target_cfg_scale=4.5,
        )
        self.assertIs(raw_cfg.validate(), raw_cfg)
        conditional_anchor = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            field_model="first_phase_caption_i2v",
            field_guidance="raw_cfg",
            source_cfg_scale=4.5,
            target_cfg_scale=8.5,
            anchor_cfg_scope="target_conditional_only",
        )
        self.assertIs(conditional_anchor.validate(), conditional_anchor)
        for dual_transport in controller.qk_transport.DUAL_SOURCE_KV_TRANSPORTS:
            dual = controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=dual_transport,
                selected_block_indices=tuple(range(4, 30)),
                transport_steps=3,
                field_model="first_phase_caption_i2v",
                field_guidance="raw_cfg",
                source_cfg_scale=4.5,
                target_cfg_scale=8.5,
                anchor_state_mode="native_t2v_trajectory",
                anchor_cfg_scope="target_conditional_only",
            )
            self.assertIs(dual.validate(), dual)
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=controller.qk_transport.DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_ALL,
                selected_block_indices=tuple(range(4, 30)),
                field_model="source_free_t2v",
                field_guidance="raw_cfg",
                source_cfg_scale=4.5,
                target_cfg_scale=8.5,
            ).validate()
        for replacement in controller.NATIVE_T2V_REPLACEMENT_TRANSPORTS:
            config = controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=replacement,
                transport_steps=3,
                initial_noise_proposal_mode="anchor_candidate0",
                anchor_state_mode="native_t2v_trajectory",
                field_model="first_phase_caption_i2v",
                field_guidance="raw_cfg",
                source_cfg_scale=4.5,
                target_cfg_scale=8.5,
            )
            self.assertIs(config.validate(), config)
        for replacement in controller.TARGETSTATE_HARD_REPLACEMENT_TRANSPORTS:
            config = controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=replacement,
                transport_steps=40,
                transport_strength=1.0,
                initial_noise_proposal_mode="anchor_candidate0_forced",
                anchor_state_mode="native_t2v_trajectory",
                field_model="first_phase_caption_i2v",
                field_guidance="raw_cfg",
                source_cfg_scale=4.5,
                target_cfg_scale=8.5,
            )
            self.assertIs(config.validate(), config)
        for replacement in controller.ROLEWARP_REPLACEMENT_TRANSPORTS:
            for arm in ("AQK_SGA5", "AQK_AVG5"):
                config = controller.AnchorSGAANCConfig(
                    arm=arm,
                    transport=replacement,
                    transport_steps=40,
                    transport_strength=1.0,
                    initial_noise_proposal_mode="keyed_only",
                    anchor_state_mode="native_t2v_trajectory",
                    anchor_contrast_mode="caption_noop_same_video",
                    field_model="first_phase_caption_i2v",
                    field_guidance="raw_cfg",
                    source_cfg_scale=4.5,
                    target_cfg_scale=8.5,
                    preservation_mode="source_motion_support",
                    preservation_start_step=24,
                    preservation_ramp_steps=8,
                    preservation_residual_fraction=0.015,
                    sga_score_mode="background_trust_anchor_envelope_003",
                )
                self.assertIs(config.validate(), config)
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=controller.FIELD_NATIVE_T2V_TARGET_VELOCITY_REPLACEMENT,
                anchor_state_mode="native_t2v_trajectory",
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=controller.FIELD_NATIVE_T2V_DELTA_VELOCITY_REPLACEMENT,
                transport_strength=0.5,
                initial_noise_proposal_mode="anchor_candidate0",
                anchor_state_mode="native_t2v_trajectory",
            ).validate()
        anchor_bank = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5", anchor_candidate_mode="bank_per_candidate"
        )
        self.assertIs(anchor_bank.validate(), anchor_bank)
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_ANC1", anchor_candidate_mode="bank_per_candidate"
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5", anchor_candidate_mode="feature_mean"
            ).validate()
        aligned_bank = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            transport=controller.FIELD_CONTRAST_VELOCITY,
            anchor_contrast_mode="dynamic_static_same_caption",
            preservation_mode="source_motion_support",
            anchor_candidate_mode="bank_per_candidate",
            anchor_spatial_alignment="motion_support_affine",
        )
        self.assertIs(aligned_bank.validate(), aligned_bank)
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                anchor_spatial_alignment="motion_support_affine",
            ).validate()
        dynamic_static = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            transport=controller.FIELD_CONTRAST_VELOCITY,
            anchor_contrast_mode="dynamic_static_same_caption",
            anchor_sigma_cap=0.8,
        )
        self.assertIs(dynamic_static.validate(), dynamic_static)
        for transport, strength in (
            (controller.qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_QK, 0.25),
            (
                controller.qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                0.25,
            ),
            (
                controller.qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                1.0,
            ),
            (
                controller.qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
                1.0,
            ),
            (
                controller.qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
                0.5,
            ),
            (
                controller.qk_transport.TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                0.5,
            ),
            (controller.qk_transport.HARD_PHASE_MEAN_CONTRAST_QK, 1.0),
            (
                controller.qk_transport.HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
                1.0,
            ),
            (
                controller.qk_transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
                1.0,
            ),
            (
                controller.qk_transport.TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
                1.0,
            ),
        ):
            config = controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=transport,
                transport_strength=strength,
                anchor_contrast_mode="dynamic_static_same_caption",
            )
            self.assertIs(config.validate(), config)
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=controller.qk_transport.TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=controller.qk_transport.TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                transport_strength=0.5,
                anchor_contrast_mode="dynamic_static_same_caption",
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=controller.qk_transport.HARD_PHASE_MEAN_CONTRAST_QK,
                transport_strength=0.5,
                anchor_contrast_mode="dynamic_static_same_caption",
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5", anchor_sigma_cap=0.7
            ).validate()
        preservation = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            preservation_mode="source_motion_support",
            preservation_keep_fraction=0.10,
            preservation_outside_scale=0.0,
            preservation_dilation=1,
            preservation_start_step=8,
            preservation_ramp_steps=8,
            sga_score_mode="background_source_cosine",
        )
        self.assertIs(preservation.validate(), preservation)
        snapshot_residual = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            preservation_mode="source_motion_support_snapshot_residual",
            preservation_residual_fraction=0.015,
            sga_score_mode="background_trust_anchor_envelope_003",
        )
        self.assertIs(snapshot_residual.validate(), snapshot_residual)
        corridor = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            preservation_mode="source_motion_support_event01_object1",
            preservation_keep_fraction=0.20,
            preservation_outside_scale=0.05,
            preservation_dilation=1,
            preservation_start_step=24,
            preservation_ramp_steps=8,
            preservation_object_identity_strength=0.025,
            sga_score_mode="background_trust_anchor_envelope_003",
        )
        self.assertIs(corridor.validate(), corridor)
        actor_object = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            preservation_mode="source_motion_support_event01_actor_object",
            preservation_keep_fraction=0.20,
            preservation_outside_scale=0.05,
            preservation_dilation=1,
            preservation_start_step=24,
            preservation_ramp_steps=8,
            sga_score_mode="background_trust_anchor_envelope_003",
        )
        self.assertIs(actor_object.validate(), actor_object)
        actor_object_identity = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            preservation_mode="source_motion_support_event01_actor_object",
            preservation_object_identity_strength=0.025,
            sga_score_mode="background_trust_anchor_envelope_003",
        )
        self.assertIs(actor_object_identity.validate(), actor_object_identity)
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                preservation_mode="source_motion_support",
                preservation_object_identity_strength=0.025,
            ).validate()
        for reward_mode in (
            "background_plus_anchor_action_002",
            "background_trust_anchor_action_003",
            "background_plus_anchor_envelope_005",
            "background_trust_anchor_envelope_003",
        ):
            reward = controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                preservation_mode="source_motion_support",
                sga_score_mode=reward_mode,
                anchor_candidate_mode="single_shared",
            )
            self.assertIs(reward.validate(), reward)
            with self.assertRaises(controller.AnchorSGAANCError):
                controller.AnchorSGAANCConfig(
                    arm="AQK_SGA5",
                    sga_score_mode=reward_mode,
                ).validate()
            with self.assertRaises(controller.AnchorSGAANCError):
                controller.AnchorSGAANCConfig(
                    arm="AQK_SGA5",
                    preservation_mode="source_motion_support",
                    sga_score_mode=reward_mode,
                    anchor_candidate_mode="bank_per_candidate",
                ).validate()
        for key, value in (
            ("preservation_start_step", -1),
            ("preservation_start_step", 40),
            ("preservation_start_step", True),
            ("preservation_ramp_steps", 0),
            ("preservation_ramp_steps", 41),
            ("preservation_ramp_steps", 1.5),
        ):
            with self.assertRaises(controller.AnchorSGAANCError):
                controller.AnchorSGAANCConfig(
                    arm="AQK_SGA5", **{key: value}  # type: ignore[arg-type]
                ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5", sga_score_mode="background_source_cosine"
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5", sga_score_mode="endpoint_action_classifier"
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5", preservation_keep_fraction=0.15
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5", preservation_residual_fraction=0.03
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport="hard_qk",
                anchor_contrast_mode="dynamic_static_same_caption",
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                field_model="first_phase_caption_i2v",
                field_guidance="raw_conditional",
                anchor_cfg_scope="target_conditional_only",
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                field_model="first_phase_caption_i2v",
                field_guidance="raw_cfg",
                source_cfg_scale=1.0,
                target_cfg_scale=4.5,
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_ANC1", field_guidance="cfg7"
            ).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_ANC1", field_model="source_free_t2v", field_guidance="apg"
            ).validate()
        source_free = controller.AnchorSGAANCConfig(
            arm="AQK_ANC1",
            field_model="source_free_t2v",
            field_guidance="raw_conditional",
        )
        self.assertIs(source_free.validate(), source_free)
        first_phase = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            field_model="first_phase_source_rv2v",
            field_guidance="raw_conditional",
        )
        self.assertTrue(first_phase.validate().uses_rv2v_condition)
        caption_i2v = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            field_model="first_phase_caption_i2v",
            field_guidance="raw_conditional",
        )
        self.assertTrue(caption_i2v.validate().uses_rv2v_condition)
        self.assertTrue(caption_i2v.uses_source_target_captions)
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                field_model="first_phase_caption_i2v",
                field_guidance="apg",
            ).validate()

    def test_native_t2v_trajectory_is_not_a_seed_or_clean_endpoint_proxy(self):
        config = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            transport=controller.qk_transport.TARGET_GATED_HARD_KERNEL_TOP10_ATTN_OUTPUT,
            transport_steps=3,
            transport_strength=1.0,
            anchor_state_mode="native_t2v_trajectory",
            anchor_contrast_mode="caption_noop_same_video",
            initial_noise_proposal_mode="keyed_only",
        )
        self.assertIs(config.validate(), config)
        for invalid in (
            {"anchor_candidate_mode": "bank_per_candidate"},
            {"anchor_contrast_mode": "dynamic_static_same_caption"},
            {"anchor_sigma_cap": 0.8},
        ):
            with self.assertRaises(controller.AnchorSGAANCError):
                controller.AnchorSGAANCConfig(
                    arm="AQK_SGA5",
                    anchor_state_mode="native_t2v_trajectory",
                    **invalid,
                ).validate()
        dense_field = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            transport=controller.FIELD_CONTRAST_VELOCITY,
            transport_strength=0.5,
            anchor_state_mode="native_t2v_trajectory",
            anchor_contrast_mode="caption_noop_same_video",
            preservation_mode="source_motion_support",
            anchor_spatial_alignment="motion_support_affine",
        )
        self.assertIs(dense_field.validate(), dense_field)
        for transport in controller.TARGET_STATE_FIELD_TRANSPORTS:
            replacement = transport in (
                *controller.NATIVE_T2V_REPLACEMENT_TRANSPORTS,
                *controller.TARGETSTATE_HARD_REPLACEMENT_TRANSPORTS,
            )
            target_state_field = controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=transport,
                transport_strength=1.0 if replacement else 0.5,
                anchor_state_mode="native_t2v_trajectory",
                anchor_contrast_mode="caption_noop_same_video",
                preservation_mode="source_motion_support",
                anchor_spatial_alignment="none",
                initial_noise_proposal_mode=(
                    "anchor_candidate0" if replacement else "keyed_only"
                ),
            )
            self.assertIs(target_state_field.validate(), target_state_field)
            with self.assertRaises(controller.AnchorSGAANCError):
                controller.AnchorSGAANCConfig(
                    arm="AQK_SGA5",
                    transport=transport,
                    transport_strength=1.0 if replacement else 0.5,
                    initial_noise_proposal_mode=(
                        "anchor_candidate0" if replacement else "keyed_only"
                    ),
                ).validate()
        source = inspect.getsource(controller.sample_anchor_sga_anc)
        self.assertIn("_guided_source_free_apg_velocity", source)
        self.assertIn("_native_unipc_step", source)
        self.assertIn('runtime.initial_noise_proposal_mode != "keyed_only"', source)

    def test_event01_role_graph_requires_native_contrast_and_five_proposals(self):
        for transport in controller.qk_transport.EVENT01_ROLE_GRAPH_TRANSPORTS:
            config = controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=transport,
                transport_strength=1.0,
                transport_steps=40,
                anchor_state_mode="native_t2v_trajectory",
                anchor_contrast_mode="caption_noop_same_video",
                early_candidate_count=5,
            )
            self.assertIs(config.validate(), config)
            forced = controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                transport=transport,
                transport_strength=1.0,
                transport_steps=40,
                anchor_state_mode="native_t2v_trajectory",
                anchor_contrast_mode="caption_noop_same_video",
                early_candidate_count=5,
                event01_forced_role_proposal_index=1,
            )
            self.assertIs(forced.validate(), forced)
            for invalid in (
                {"anchor_state_mode": "clean_noised"},
                {"anchor_contrast_mode": "dynamic_static_same_caption"},
                {"early_candidate_count": 8},
            ):
                with self.assertRaises(controller.AnchorSGAANCError):
                    kwargs = {
                        "arm": "AQK_SGA5",
                        "transport": transport,
                        "transport_strength": 1.0,
                        "transport_steps": 40,
                        "anchor_state_mode": "native_t2v_trajectory",
                        "anchor_contrast_mode": "caption_noop_same_video",
                        "early_candidate_count": 5,
                    }
                    kwargs.update(invalid)
                    controller.AnchorSGAANCConfig(**kwargs).validate()
        with self.assertRaises(controller.AnchorSGAANCError):
            controller.AnchorSGAANCConfig(
                arm="AQK_SGA5",
                event01_forced_role_proposal_index=1,
            ).validate()

    def test_event01_role_graph_receipt_closure_accepts_forced_proposal(self):
        automatic = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            transport=controller.qk_transport.EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT,
            transport_steps=40,
            anchor_state_mode="native_t2v_trajectory",
        ).validate()
        forced = controller.AnchorSGAANCConfig(
            arm="AQK_SGA5",
            transport=controller.qk_transport.EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT,
            transport_steps=40,
            anchor_state_mode="native_t2v_trajectory",
            event01_forced_role_proposal_index=1,
        ).validate()
        self.assertEqual(
            controller._expected_event01_early_role_proposals(automatic),
            list(range(5)) * 3,
        )
        self.assertEqual(
            controller._expected_event01_early_role_proposals(forced),
            [1] * 15,
        )

    def test_native_phase_envelope_gates_target_route_without_copying_space(self):
        phases = controller.guided.EXPECTED_LATENT_PHASES
        target = torch.ones((1, phases * 2, 3), dtype=torch.float32)
        action = torch.zeros_like(target).reshape(1, phases, 2, 3)
        noop = torch.zeros_like(action)
        for phase in range(1, phases):
            action[:, phase] = float(phase)
        routed, envelope = controller._apply_native_phase_envelope(
            target,
            action.reshape_as(target),
            noop.reshape_as(target),
        )
        self.assertEqual(len(envelope), phases)
        self.assertEqual(envelope[0], 0.0)
        self.assertTrue(torch.equal(routed[:, :2], torch.zeros_like(routed[:, :2])))
        self.assertTrue(torch.isfinite(routed).all())
        self.assertGreater(envelope[-1], envelope[1])

    def test_anchor_sigma_cap_uses_a_matched_model_timestep(self):
        outer = torch.tensor(999, dtype=torch.int64)
        capped_08 = controller._model_timestep_for_anchor_sigma(
            outer_timestep=outer,
            outer_sigma=0.9999989867210388,
            anchor_sigma=0.8,
            num_train_timesteps=1000,
        )
        capped_06 = controller._model_timestep_for_anchor_sigma(
            outer_timestep=outer,
            outer_sigma=0.9999989867210388,
            anchor_sigma=0.6,
            num_train_timesteps=1000,
        )
        self.assertEqual(capped_08.dtype, outer.dtype)
        self.assertEqual(capped_08.device, outer.device)
        self.assertEqual(int(capped_08.item()), 800)
        self.assertEqual(int(capped_06.item()), 600)
        outer_fp32 = torch.tensor(803.0, dtype=torch.float32)
        capped_fp32 = controller._model_timestep_for_anchor_sigma(
            outer_timestep=outer_fp32,
            outer_sigma=0.8039219379425049,
            anchor_sigma=0.8,
            num_train_timesteps=1000,
        )
        self.assertEqual(capped_fp32.dtype, torch.float32)
        self.assertEqual(float(capped_fp32.item()), 800.0)
        uncapped = controller._model_timestep_for_anchor_sigma(
            outer_timestep=outer,
            outer_sigma=0.9999989867210388,
            anchor_sigma=0.9999989867210388,
            num_train_timesteps=1000,
        )
        self.assertIs(uncapped, outer)
        with self.assertRaises(controller.AnchorSGAANCError):
            controller._model_timestep_for_anchor_sigma(
                outer_timestep=outer,
                outer_sigma=0.9,
                anchor_sigma=0.8,
                num_train_timesteps=999,
            )
        with self.assertRaises(controller.AnchorSGAANCError):
            controller._model_timestep_for_anchor_sigma(
                outer_timestep=outer,
                outer_sigma=0.9,
                anchor_sigma=0.7505,
                num_train_timesteps=1000,
            )

    def test_dense_anchor_route_affine_alignment_moves_support_not_phase_zero(self):
        layout = cdf.validate_latent_shape((1, 1, 21, 4, 4))
        route = torch.zeros(1, 1, 21, 4, 4, dtype=torch.float32)
        route[:, :, 1:, 0, 0] = 1.0
        source_support = torch.zeros(1, 1, 1, 4, 4, dtype=torch.float32)
        source_support[:, :, :, 3, 3] = 1.0
        aligned, audit = controller._align_packed_route_to_source_motion(
            cdf._pack_spatial_latent(route, layout),
            source_support,
            layout=layout,
            keep_fraction=1.0 / 16.0,
        )
        unpacked = cdf._unpack_spatial_latent(aligned, layout)
        self.assertTrue(
            torch.equal(unpacked[:, :, 0], torch.zeros_like(unpacked[:, :, 0]))
        )
        energy = unpacked[:, :, 1:].square().sum(dim=(1, 2))[0]
        maximum = int(energy.flatten().argmax().item())
        self.assertEqual(divmod(maximum, 4), (3, 3))
        self.assertEqual(audit["anchor_center_xy"], [-0.75, -0.75])
        self.assertEqual(audit["source_center_xy"], [0.75, 0.75])

    def test_initial_phase_clamp_changes_only_first_frame_major_slab(self):
        layout = cdf.LatentLayout(
            batch=1,
            channels=16,
            frames=21,
            height=4,
            width=6,
            tokens=21 * 6,
            packed_channels=64,
        )
        source = torch.arange(
            layout.tokens * layout.packed_channels, dtype=torch.float32
        ).reshape(1, layout.tokens, layout.packed_channels)
        edit = torch.full_like(source, -7.0)
        controller._clamp_initial_latent_phase(edit, source, layout)
        spatial = layout.tokens // layout.frames
        self.assertTrue(torch.equal(edit[:, :spatial], source[:, :spatial]))
        self.assertTrue(
            torch.equal(edit[:, spatial:], torch.full_like(edit[:, spatial:], -7.0))
        )

    def test_full_velocity_residual_is_sparse_temporal_and_frame0_exact(self):
        tokens = 21 * 4
        current = torch.full((1, tokens, 8), 5.0, dtype=torch.float32)
        anchor = torch.zeros_like(current)
        anchor[:, 4:] = 8.0
        routed = controller._sparse_packed_temporal_residual(
            current, anchor, strength=0.25
        ).reshape(1, 21, 4, 8)
        self.assertTrue(torch.equal(routed[:, 0], torch.full_like(routed[:, 0], 5)))
        self.assertGreater(int(torch.count_nonzero(routed[:, 1:] - 5)), 0)
        self.assertLessEqual(
            int(torch.count_nonzero((routed[:, 1:] - 5).sum(dim=-1))),
            20,
        )

    def test_action_noop_velocity_contrast_is_sparse_and_phase0_free(self):
        tokens = 21 * 4
        action = torch.zeros((1, tokens, 8), dtype=torch.float32)
        noop = torch.zeros_like(action)
        action.reshape(1, 21, 4, 8)[:, 1:, 2] = 3.0
        routed = controller._sparse_packed_action_contrast(
            action, noop, strength=0.5
        ).reshape(1, 21, 4, 8)
        self.assertTrue(torch.equal(routed[:, 0], torch.zeros_like(routed[:, 0])))
        self.assertGreater(int(torch.count_nonzero(routed[:, 1:])), 0)
        changed = routed[:, 1:].abs().sum(dim=-1) > 0
        self.assertLessEqual(int(changed.sum()), 20)
        self.assertTrue(torch.allclose(routed[0, 1:, 2], torch.full((20, 8), 1.5)))
        self.assertFalse(changed[..., :2].any())
        self.assertFalse(changed[..., 3:].any())

    def test_raw_action_noop_velocity_contrast_retains_absolute_phase_component(self):
        tokens = 21 * 4
        action = torch.full((1, tokens, 8), 11.0, dtype=torch.float32)
        noop = torch.zeros_like(action)
        action.reshape(1, 21, 4, 8)[:, :, 2] += 3.0
        full = controller._sparse_packed_raw_action_contrast(
            action, noop, strength=1.0, keep_fraction=1.0
        ).reshape(1, 21, 4, 8)
        self.assertTrue(torch.equal(full, action.reshape_as(full)))
        sparse = controller._sparse_packed_raw_action_contrast(
            action, noop, strength=0.5, keep_fraction=0.25
        ).reshape(1, 21, 4, 8)
        self.assertTrue(torch.allclose(sparse[:, :, 2], torch.full_like(sparse[:, :, 2], 7.0)))
        changed = sparse.abs().sum(dim=-1) > 0
        self.assertEqual(int(changed.sum()), 21)
        self.assertFalse(changed[..., :2].any())
        self.assertFalse(changed[..., 3:].any())

    def test_event01_role_warp_moves_actor_and_object_components_independently(self):
        layout = cdf.validate_latent_shape((1, 1, 21, 72, 52))
        route = torch.zeros((1, 1, 21, 72, 52), dtype=torch.float32)
        route[:, :, 1:, 31, 38] = 2.0
        route[:, :, 1:, 43, 14] = 3.0
        packed = cdf._pack_spatial_latent(route, layout)
        warped_packed, audit = controller._event01_role_warp_native_route(
            packed,
            layout=layout,
            proposal_index=2,
            keep_fraction=1.0,
        )
        warped = cdf._unpack_spatial_latent(warped_packed, layout)
        self.assertTrue(torch.equal(warped[:, :, 0], torch.zeros_like(warped[:, :, 0])))
        actor_x, actor_y = controller.EVENT01_ROLE_SOURCE_ACTOR_XY
        object_x, object_y = controller.EVENT01_ROLE_SOURCE_OBJECT_PROPOSALS_XY[2]
        actor_value = float(warped[0, 0, 1, actor_y, actor_x])
        object_value = float(warped[0, 0, 1, object_y, object_x])
        self.assertGreater(actor_value, 1.0)
        self.assertGreater(object_value, 1.5)
        self.assertEqual(audit["source_actor_xy"], [24, 50])
        self.assertEqual(audit["source_object_xy"], [41, 57])
        self.assertEqual(audit["actor_shift_yx"], [19, -14])
        self.assertEqual(audit["object_shift_yx"], [14, 27])

    def test_dense_source_motion_support_preserves_outside_and_frame0(self):
        layout = cdf.LatentLayout(
            batch=1,
            channels=16,
            frames=21,
            height=4,
            width=6,
            tokens=21 * 6,
            packed_channels=64,
        )
        source = torch.zeros((1, 16, 21, 4, 6), dtype=torch.float32)
        source[:, :, 1:, 1, 2] = torch.arange(1, 21).reshape(1, 1, 20)
        support = controller._source_motion_support(
            source, keep_fraction=0.10, dilation=1
        )
        self.assertEqual(tuple(support.shape), (1, 1, 1, 4, 6))
        self.assertGreater(float(support.mean()), 0.10)
        self.assertLess(float(support.mean()), 1.0)
        edit = source + 7.0
        preserved = cdf._unpack_spatial_latent(
            controller._apply_source_motion_preservation(
                cdf._pack_spatial_latent(edit, layout),
                source,
                support,
                layout=layout,
                outside_scale=0.0,
            ),
            layout,
        )
        self.assertTrue(torch.equal(preserved[:, :, 0], source[:, :, 0]))
        outside = (1.0 - support).expand_as(source).bool()
        self.assertTrue(torch.equal(preserved[outside], source[outside]))
        inside_later = support.expand_as(source).bool()
        inside_later[:, :, 0] = False
        self.assertTrue(torch.allclose(preserved[inside_later] - source[inside_later], torch.full_like(preserved[inside_later], 7.0)))
        adaptive = cdf._unpack_spatial_latent(
            controller._apply_source_motion_preservation(
                cdf._pack_spatial_latent(edit, layout),
                source,
                support,
                layout=layout,
                outside_scale=0.0,
                residual_fraction=0.01,
            ),
            layout,
        )
        hard_changed = (preserved[:, :, 1:] != source[:, :, 1:]).any(dim=1)
        adaptive_changed = (adaptive[:, :, 1:] != source[:, :, 1:]).any(dim=1)
        self.assertGreater(int(adaptive_changed.sum()), int(hard_changed.sum()))
        self.assertTrue(torch.equal(adaptive[:, :, 0], source[:, :, 0]))

    def test_event01_object1_preservation_corridor_is_phasewise_and_local(self):
        base = torch.zeros((1, 1, 1, 72, 52), dtype=torch.float32)
        base[..., 50, 24] = 1.0
        corridor = controller._event01_object1_phasewise_preservation_support(base)
        self.assertEqual(tuple(corridor.shape), (1, 1, 21, 72, 52))
        self.assertTrue(torch.equal(corridor[:, :, :, 50, 24], torch.ones(1, 1, 21)))
        proposal_x, proposal_y = controller.qk_transport.EVENT01_SOURCE_OBJECT_PROPOSALS_XY[1]
        start_x, start_y = int(round(2 * proposal_x)), int(round(2 * proposal_y))
        self.assertEqual(float(corridor[0, 0, 0, start_y, start_x]), 1.0)
        terminal = controller.qk_transport._event01_dynamic_target_centers(1)[-1][1]
        terminal_x, terminal_y = int(round(2 * terminal[0])), int(round(2 * terminal[1]))
        self.assertEqual(float(corridor[0, 0, -1, terminal_y, terminal_x]), 1.0)
        self.assertEqual(float(corridor[0, 0, -1, start_y, start_x]), 1.0)
        self.assertEqual(float(corridor[0, 0, :, 0, 0].sum()), 0.0)

        source = torch.zeros((1, 16, 21, 72, 52), dtype=torch.float32)
        edit = torch.ones_like(source)
        layout = cdf.LatentLayout(
            batch=1,
            channels=16,
            frames=21,
            height=72,
            width=52,
            tokens=21 * 72 * 52 // 4,
            packed_channels=64,
        )
        preserved = cdf._unpack_spatial_latent(
            controller._apply_source_motion_preservation(
                cdf._pack_spatial_latent(edit, layout),
                source,
                corridor,
                layout=layout,
                outside_scale=0.0,
            ),
            layout,
        )
        self.assertTrue(torch.equal(preserved[:, :, 0], source[:, :, 0]))
        self.assertGreater(float(preserved[0, :, -1].abs().sum()), 0.0)
        self.assertEqual(float(preserved[0, :, -1, 0, 0].abs().sum()), 0.0)

    def test_snapshot_residual_support_does_not_drop_an_earlier_edit_region(self):
        layout = cdf.LatentLayout(
            batch=1,
            channels=16,
            frames=21,
            height=4,
            width=6,
            tokens=21 * 6,
            packed_channels=64,
        )
        source = torch.zeros((1, 16, 21, 4, 6), dtype=torch.float32)
        base_support = torch.zeros((1, 1, 1, 4, 6), dtype=torch.float32)
        base_support[..., 0, 0] = 1.0
        at_snapshot = source.clone()
        at_snapshot[:, :, 1:, 1, 1] = 10.0
        snapshot = controller._effective_source_edit_support(
            at_snapshot,
            source,
            base_support,
            residual_fraction=0.01,
        )
        later = source.clone()
        later[:, :, 1:, 1, 1] = 3.0
        later[:, :, 1:, 2, 2] = 20.0
        preserved = cdf._unpack_spatial_latent(
            controller._apply_source_motion_preservation(
                cdf._pack_spatial_latent(later, layout),
                source,
                snapshot,
                layout=layout,
                outside_scale=0.0,
                residual_fraction=0.0,
            ),
            layout,
        )
        self.assertTrue(torch.equal(preserved[:, :, 0], source[:, :, 0]))
        self.assertTrue(torch.equal(preserved[:, :, 1:, 1, 1], later[:, :, 1:, 1, 1]))
        self.assertTrue(torch.equal(preserved[:, :, 1:, 2, 2], source[:, :, 1:, 2, 2]))

    def test_event01_object1_identity_projection_translates_phase0_instance(self):
        source = torch.zeros((1, 16, 21, 72, 52), dtype=torch.float32)
        proposal_x, proposal_y = controller.qk_transport.EVENT01_SOURCE_OBJECT_PROPOSALS_XY[1]
        source_x, source_y = int(round(2 * proposal_x)), int(round(2 * proposal_y))
        source[:, :, 0, source_y, source_x] = 7.0
        reference, mask = controller._event01_object1_phasewise_source_reference(source)
        terminal = controller.qk_transport._event01_dynamic_target_centers(1)[-1][1]
        terminal_x, terminal_y = int(round(2 * terminal[0])), int(round(2 * terminal[1]))
        self.assertTrue(
            torch.equal(
                reference[0, :, -1, terminal_y, terminal_x],
                torch.full((16,), 7.0),
            )
        )
        self.assertEqual(float(mask[0, 0, -1, terminal_y, terminal_x]), 1.0)
        self.assertEqual(float(mask[0, 0, -1, 0, 0]), 0.0)

        layout = cdf.LatentLayout(
            batch=1,
            channels=16,
            frames=21,
            height=72,
            width=52,
            tokens=21 * 72 * 52 // 4,
            packed_channels=64,
        )
        edit = torch.zeros_like(source)
        projected = cdf._unpack_spatial_latent(
            controller._apply_event01_object1_identity_projection(
                cdf._pack_spatial_latent(edit, layout),
                source,
                layout=layout,
                strength=1.0,
            ),
            layout,
        )
        self.assertTrue(torch.equal(projected[:, :, 0], source[:, :, 0]))
        self.assertTrue(
            torch.equal(
                projected[0, :, -1, terminal_y, terminal_x],
                torch.full((16,), 7.0),
            )
        )
        self.assertEqual(float(projected[0, :, -1, 0, 0].sum()), 0.0)

    def test_event01_actor_object_support_opens_new_pose_without_global_release(self):
        base = torch.zeros((1, 1, 1, 72, 52), dtype=torch.float32)
        object_only = controller._event01_object1_phasewise_preservation_support(base)
        actor_object = controller._event01_actor_object_phasewise_preservation_support(base)
        self.assertEqual(tuple(actor_object.shape), (1, 1, 21, 72, 52))
        self.assertGreater(float(actor_object.sum()), float(object_only.sum()))
        actor_xy = controller.qk_transport._event01_dynamic_target_centers(1)[-1][0]
        actor_x, actor_y = int(round(2 * actor_xy[0])), int(round(2 * actor_xy[1]))
        self.assertEqual(float(actor_object[0, 0, -1, actor_y, actor_x]), 1.0)
        self.assertEqual(float(actor_object[0, 0, -1, 0, 0]), 0.0)

    def test_event01_sparse_entity_signature_moves_patch_and_vacates_origin(self):
        height, width = 72, 52
        source = torch.zeros((1, 16, 21, height, width), dtype=torch.float32)
        edit = torch.zeros_like(source)
        proposal_x, proposal_y = (
            controller.qk_transport.EVENT01_SOURCE_OBJECT_PROPOSALS_XY[1]
        )
        source_x = 2.0 * proposal_x
        source_y = 2.0 * proposal_y
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        distance = (
            ((xx - source_x) / 2.0).square()
            + ((yy - source_y) / 1.5).square()
        )
        source_flat = torch.topk(
            distance.flatten(), k=16, largest=False, sorted=True
        ).indices
        source_values = torch.arange(1, 17, dtype=torch.float32)
        source[:, :, 0].flatten(2)[:, :, source_flat] = source_values
        edit[:, :, -1].flatten(2)[:, :, source_flat] = 9.0
        layout = cdf.LatentLayout(
            batch=1,
            channels=16,
            frames=21,
            height=height,
            width=width,
            tokens=21 * height * width // 4,
            packed_channels=64,
        )
        full_packed = controller._apply_event01_object1_sparse_signature_projection(
            cdf._pack_spatial_latent(edit, layout),
            source,
            layout=layout,
            strength=1.0,
        )
        half_packed = controller._apply_event01_object1_sparse_signature_projection(
            cdf._pack_spatial_latent(edit, layout),
            source,
            layout=layout,
            strength=0.5,
        )
        full = cdf._unpack_spatial_latent(full_packed, layout)
        half = cdf._unpack_spatial_latent(half_packed, layout)
        self.assertTrue(torch.equal(full[:, :, 0], source[:, :, 0]))
        self.assertTrue(
            torch.equal(
                full[:, :, -1].flatten(2)[:, :, source_flat],
                torch.zeros((1, 16, 16)),
            )
        )
        terminal_x, terminal_y = (
            value * 2.0
            for value in controller.qk_transport._event01_dynamic_target_centers(1)[-1][1]
        )
        source_yx = torch.stack(
            (
                torch.div(source_flat, width, rounding_mode="floor"),
                source_flat.remainder(width),
            ),
            dim=1,
        ).float()
        target_y = torch.floor(source_yx[:, 0] - source_y + terminal_y + 0.5).long()
        target_x = torch.floor(source_yx[:, 1] - source_x + terminal_x + 0.5).long()
        target_flat = target_y * width + target_x
        self.assertEqual(int(target_flat.unique().numel()), 16)
        self.assertTrue(
            torch.equal(
                full[:, :, -1].flatten(2)[:, :, target_flat],
                source_values.reshape(1, 1, 16).expand(1, 16, 16),
            )
        )
        self.assertTrue(torch.equal(half[:, :, 0], source[:, :, 0]))
        self.assertTrue(
            torch.allclose(
                half[:, :, 1:],
                edit[:, :, 1:] + 0.5 * (full[:, :, 1:] - edit[:, :, 1:]),
            )
        )

    def test_background_sga_scores_background_but_aggregates_full_candidates(self):
        layout = cdf.LatentLayout(
            batch=1,
            channels=16,
            frames=21,
            height=2,
            width=2,
            tokens=21,
            packed_channels=64,
        )
        source_clean = torch.ones((1, 16, 21, 2, 2), dtype=torch.float32)
        source_packed = cdf._pack_spatial_latent(source_clean, layout)
        edit_packed = source_packed.clone()
        support = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
        support[..., 0, 0] = 1.0

        projected_a = source_clean.clone()
        projected_a[..., 0, 0] = 20.0
        projected_b = source_clean.clone()
        projected_b[..., 0, 1] = -2.0
        projected_bank = torch.stack(
            (
                cdf._pack_spatial_latent(projected_a, layout),
                cdf._pack_spatial_latent(projected_b, layout),
            )
        )
        sigma = 0.5
        candidate_deltas = (edit_packed.unsqueeze(0) - projected_bank) / sigma
        aggregate, weights, scores = (
            controller._aggregate_candidates_background_source_cosine(
                source_packed=source_packed,
                source_clean=source_clean,
                edit_packed=edit_packed,
                candidate_deltas=candidate_deltas,
                sigma=sigma,
                temperature=0.01,
                layout=layout,
                source_support=support,
                residual_fraction=0.0,
            )
        )
        self.assertEqual(tuple(aggregate.shape), tuple(source_packed.shape))
        self.assertEqual(tuple(weights.shape), (2, 1))
        self.assertEqual(tuple(scores.shape), (2, 1))
        self.assertTrue(torch.allclose(weights.sum(dim=0), torch.ones(1)))
        self.assertGreater(float(weights[0, 0]), 0.999)
        aggregate_projection = edit_packed - sigma * aggregate
        # Candidate A wins because its only change is inside the excluded edit
        # support, and the full candidate (including that large change) survives.
        self.assertGreater(
            float(cdf._unpack_spatial_latent(aggregate_projection, layout)[..., 0, 0].mean()),
            19.9,
        )

    def test_action_temporal_signature_is_spatial_permutation_invariant(self):
        effect = torch.zeros((1, 4, 21, 2, 3), dtype=torch.float32)
        trajectory = torch.arange(21, dtype=torch.float32)
        effect[0, 0, :, 0, 1] = trajectory
        effect[0, 1, :, 0, 1] = trajectory.square() / 20.0
        signature = controller._action_temporal_signature(effect)
        permuted = controller._action_temporal_signature(effect.flip(dims=(3, 4)))
        reversed_time = controller._action_temporal_signature(effect.flip(dims=(2,)))
        self.assertTrue(torch.allclose(signature, permuted, atol=1.0e-6))
        self.assertGreater(float((signature * signature).sum()), 0.99)
        self.assertLess(float((signature * reversed_time).sum()), 0.99)

    def test_motion_envelope_signature_retains_temporal_spatial_direction(self):
        effect = torch.zeros((1, 4, 21, 16, 16), dtype=torch.float32)
        for phase in range(1, 21):
            row = min(14, 2 + phase // 2)
            effect[0, :, phase, row, 6:9] = float(phase)
        signature = controller._action_motion_envelope_signature(effect)
        shifted = torch.roll(effect, shifts=3, dims=4)
        shifted_signature = controller._action_motion_envelope_signature(shifted)
        reversed_signature = controller._action_motion_envelope_signature(
            effect.flip(dims=(2,))
        )
        self.assertGreater(float((signature * signature).sum()), 0.99)
        self.assertGreater(float((signature * shifted_signature).sum()), 0.95)
        self.assertLess(float((signature * reversed_signature).sum()), 0.95)

    def test_v14r2_qk_route_closure_rejects_forward_only_cache_bypass(self):
        transport = (
            controller.qk_transport.TARGET_OWNED_TEMPORAL_KERNEL_ATTN_OUTPUT_V14R2
        )
        trace = {
            "target_owned_qk_route_v14r2": True,
            "anchor_donor_cached_fields": ["query", "key"],
            "anchor_donor_value_hidden_output_or_coordinate_used": False,
        }
        complete = {
            "capture_count": 1320,
            "qk_only_capture_count": 1320,
            "replay_count": 2640,
            "qk_only_replay_count": 2640,
            "pending_entries": 0,
            "qk_only_cached_fields": ["query", "key"],
        }
        controller._validate_target_owned_qk_route_closure(
            transport=transport,
            transport_steps=40,
            expected_anchor_cells=30,
            selected_block_count=22,
            field_guidance="raw_cfg",
            anchor_cfg_scope="shared",
            trace=trace,
            cache_receipt=complete,
        )
        bypass = dict(complete)
        for key in (
            "capture_count",
            "qk_only_capture_count",
            "replay_count",
            "qk_only_replay_count",
        ):
            bypass[key] = 0
        with self.assertRaises(controller.AnchorSGAANCError):
            controller._validate_target_owned_qk_route_closure(
                transport=transport,
                transport_steps=40,
                expected_anchor_cells=30,
                selected_block_count=22,
                field_guidance="raw_cfg",
                anchor_cfg_scope="shared",
                trace=trace,
                cache_receipt=bypass,
            )

        routeoff_trace = {
            "target_owned_qk_route_v14r2": False,
            "anchor_donor_cached_fields": None,
            "anchor_donor_value_hidden_output_or_coordinate_used": None,
        }
        routeoff = dict(complete)
        for key in (
            "capture_count",
            "qk_only_capture_count",
            "replay_count",
            "qk_only_replay_count",
        ):
            routeoff[key] = 0
        controller._validate_target_owned_qk_route_closure(
            transport=transport,
            transport_steps=0,
            expected_anchor_cells=0,
            selected_block_count=22,
            field_guidance="raw_cfg",
            anchor_cfg_scope="shared",
            trace=routeoff_trace,
            cache_receipt=routeoff,
        )


if __name__ == "__main__":
    unittest.main()

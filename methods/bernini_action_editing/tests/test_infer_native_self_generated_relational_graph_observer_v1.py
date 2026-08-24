#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_self_generated_relational_graph_observer_v1 as runner  # noqa: E402
import self_generated_intermediate_action_anchor_v1 as anchor_core  # noqa: E402


def make_authority() -> runner.FourArmForwardAuthority:
    generator = torch.Generator(device="cpu").manual_seed(17)
    prompts = {
        arm: torch.randn((1, 7 + index, 4), generator=generator)
        for index, arm in enumerate(runner.ARMS)
    }
    noisy = torch.randn((1, 2, 3), generator=generator)
    timestep = torch.tensor([18.0])
    rotary = torch.randn((1, 1, 2, 4), generator=generator)

    def shared_step(
        model_id,
        noisy_latents,
        timesteps,
        cond_embeds,
        rotary_embs,
        batch_vae_seqlen,
        batch_text_seqlen,
    ):
        del model_id, noisy_latents, timesteps, rotary_embs
        del batch_vae_seqlen, batch_text_seqlen
        return cond_embeds

    return runner.seal_four_arm_forward(
        appearance_id="appearance_0",
        sigma_cell=runner.SigmaCell("mid", 18, 0.55),
        shared_step=shared_step,
        action_kwargs={
            "model_id": "transformer_1",
            "noisy_latents": noisy,
            "timesteps": timestep,
            "cond_embeds": prompts["action"],
            "rotary_embs": rotary,
            "batch_vae_seqlen": [21],
            "batch_text_seqlen": [int(prompts["action"].shape[1])],
        },
        prompt_embeds=prompts,
        instruction_sha256={
            "action": "a" * 64,
            "noop": anchor_core.CANONICAL_NOOP_SHA256,
            "reverse": "b" * 64,
            "static": "c" * 64,
        },
    )


class NativeRelationalRunnerContractTests(unittest.TestCase):
    def test_role_matched_t2v_authority_replaces_source_video_noop(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(29)
        prompts = {
            arm: torch.randn((1, 9, 4), generator=generator).detach()
            for arm in runner.ARMS
        }
        noisy = torch.randn((1, 2, 3), generator=generator)
        timestep = torch.tensor([32.0])
        rotary = torch.randn((1, 1, 2, 4), generator=generator)

        def shared_step(
            model_id,
            noisy_latents,
            timesteps,
            cond_embeds,
            rotary_embs,
            batch_vae_seqlen,
            batch_text_seqlen,
            **kwargs,
        ):
            self.assertEqual(kwargs, {})
            del model_id, noisy_latents, timesteps, rotary_embs
            del batch_vae_seqlen, batch_text_seqlen
            return cond_embeds.clone()

        role_phrases = {"agent": "the woman", "object": "the mug"}
        instructions = {
            "action": "the woman moves the mug from left to right",
            "noop": "the woman stays still and the mug remains on the left",
            "reverse": "the woman moves the mug from right to left",
            "static": "the woman holds the mug still in the center",
        }
        authority = runner.seal_role_matched_t2v_four_arm_forward(
            appearance_id="appearance_0",
            sigma_cell=runner.SigmaCell("mid", 32, 0.55),
            shared_step=shared_step,
            action_kwargs={
                "model_id": "transformer_1",
                "noisy_latents": noisy,
                "timesteps": timestep,
                "cond_embeds": prompts["action"],
                "rotary_embs": rotary,
                "batch_vae_seqlen": [42],
                "batch_text_seqlen": [9],
            },
            prompt_embeds=prompts,
            instructions=instructions,
            role_phrases=role_phrases,
        )
        receipt = authority.receipt()
        self.assertFalse(receipt["generic_source_video_noop_used"])
        self.assertTrue(receipt["all_role_phrases_retained_in_every_arm"])
        outputs = {arm: authority.call(arm) for arm in runner.ARMS}
        self.assertEqual(len({id(value) for value in outputs.values()}), 4)
        noisy.add_(1.0)
        with self.assertRaises(runner.NativeRelationalObserverError):
            authority.validate()

    def test_role_matched_t2v_authority_rejects_missing_role_in_control(self) -> None:
        prompts = {
            arm: torch.zeros((1, 3, 2)) + index
            for index, arm in enumerate(runner.ARMS)
        }

        def shared_step(
            model_id,
            noisy_latents,
            timesteps,
            cond_embeds,
            rotary_embs,
            batch_vae_seqlen,
            batch_text_seqlen,
        ):
            return cond_embeds

        with self.assertRaisesRegex(
            runner.NativeRelationalObserverError, "lacks role phrase object"
        ):
            runner.seal_role_matched_t2v_four_arm_forward(
                appearance_id="appearance_0",
                sigma_cell=runner.SigmaCell("high", 18, 0.85),
                shared_step=shared_step,
                action_kwargs={
                    "model_id": "transformer_1",
                    "noisy_latents": torch.zeros(1),
                    "timesteps": torch.zeros(1),
                    "cond_embeds": prompts["action"],
                    "rotary_embs": torch.zeros(1),
                    "batch_vae_seqlen": [3],
                    "batch_text_seqlen": [3],
                },
                prompt_embeds=prompts,
                instructions={
                    "action": "the agent moves the object",
                    "noop": "the agent stays still",
                    "reverse": "the agent reverses the object",
                    "static": "the agent holds the object",
                },
                role_phrases={"agent": "the agent", "object": "the object"},
            )

    def test_gpu_contract_is_frozen_observer_only_and_closed_exact144(self) -> None:
        value = runner.native_gpu_launch_contract()
        plan = value["capture_plan"]
        self.assertEqual(plan["native_forward_count"], 36)
        self.assertEqual(plan["block_capture_count"], 144)
        self.assertEqual(plan["appearance_ids"], list(runner.APPEARANCE_IDS))
        self.assertEqual(plan["arms"], list(runner.ARMS))
        self.assertEqual(plan["blocks"], list(runner.BLOCKS))
        self.assertTrue(value["base_frozen"])
        self.assertEqual(value["parameter_updates"], 0)
        self.assertIsNone(value["optimizer"])
        self.assertFalse(value["decoder_available_to_runner"])
        self.assertFalse(value["renderer_available_to_runner"])
        self.assertFalse(value["target_teacher_available_to_runner"])
        self.assertFalse(value["target_inputs_consumed"])
        self.assertFalse(value["gpu_launch_authorized"])
        self.assertFalse(value["scientific_claim_authorized"])

    def test_four_arm_authority_reuses_one_state_and_rejects_mutation(self) -> None:
        authority = make_authority()
        outputs = {arm: authority.call(arm) for arm in runner.ARMS}
        self.assertEqual(len({id(value) for value in outputs.values()}), 4)
        receipt = authority.receipt()
        self.assertTrue(receipt["same_original_shared_step"])
        self.assertTrue(receipt["same_noisy_timestep_rotary_and_nontext_objects"])
        action = runner._bound_arguments(
            authority.action_noop.shared_step,
            authority.action_noop.action_args,
            authority.action_noop.action_kwargs,
        )
        action.arguments["noisy_latents"].add_(1.0)
        with self.assertRaises(runner.NativeRelationalObserverError):
            authority.validate()

    def test_target_named_runtime_argument_fails_before_sealing(self) -> None:
        prompts = {arm: torch.zeros((1, 3, 2)) + index for index, arm in enumerate(runner.ARMS)}

        def shared_step(
            model_id,
            noisy_latents,
            timesteps,
            cond_embeds,
            rotary_embs,
            batch_vae_seqlen,
            batch_text_seqlen,
            target_video=None,
        ):
            del model_id, noisy_latents, timesteps, rotary_embs
            del batch_vae_seqlen, batch_text_seqlen, target_video
            return cond_embeds

        with self.assertRaises(Exception):
            runner.seal_four_arm_forward(
                appearance_id="appearance_0",
                sigma_cell=runner.SigmaCell("high", 5, 0.85),
                shared_step=shared_step,
                action_kwargs={
                    "model_id": "transformer_1",
                    "noisy_latents": torch.zeros(1),
                    "timesteps": torch.zeros(1),
                    "cond_embeds": prompts["action"],
                    "rotary_embs": torch.zeros(1),
                    "batch_vae_seqlen": [3],
                    "batch_text_seqlen": [3],
                    "target_video": torch.zeros(1),
                },
                prompt_embeds=prompts,
                instruction_sha256={
                    "action": "a" * 64,
                    "noop": anchor_core.CANONICAL_NOOP_SHA256,
                    "reverse": "b" * 64,
                    "static": "c" * 64,
                },
            )

    def test_capture_bank_requires_exact_blocks_and_zeroizes(self) -> None:
        authority = make_authority()
        cell = authority.sigma_cell
        invocation = runner.CaptureInvocation(
            "appearance_0",
            "action",
            cell,
            authority.state_tensor_sha256["noisy_latents"],
            authority.state_tensor_sha256["timesteps"],
            authority.state_tensor_sha256["rotary_embs"],
            1,
            5,
        )
        bank = runner.InMemoryNativeCaptureBank()
        with bank.observe(invocation):
            for block in runner.BLOCKS:
                query = torch.randn((1, 21, 5, 2, 3))
                key = torch.randn((1, 21, 5, 2, 3))
                role = torch.softmax(torch.randn((1, 21, 3, 5)), dim=2)
                bank.capture(
                    runner.NativeBlockCapture(
                        runner.CAPTURE_SCHEMA,
                        invocation,
                        block,
                        query.detach().contiguous(),
                        key.detach().contiguous(),
                        role.detach().contiguous(),
                    )
                )
        captures = bank.consume(invocation)
        bank.zeroize(captures)
        self.assertEqual(bank.receipt()["capture_count"], 4)
        self.assertEqual(bank.receipt()["zeroized_count"], 4)
        for capture in captures:
            self.assertEqual(int(torch.count_nonzero(capture.query)), 0)
            self.assertEqual(int(torch.count_nonzero(capture.key)), 0)
            self.assertEqual(
                int(
                    torch.count_nonzero(
                        capture.derived_qk_role_responsibility_proxy
                    )
                ),
                0,
            )

    @unittest.skipUnless(
        importlib.util.find_spec(
            "self_generated_relational_action_graph_observer_v1"
        )
        is not None,
        "relational observer core is being landed independently",
    )
    def test_full_toy_observer_integration_closes_p0_and_raw_clear(self) -> None:
        value = runner._dry_run()
        self.assertTrue(value["frozen_base_unchanged"])
        self.assertTrue(value["p0_exact_replay"]["bit_exact"])
        self.assertTrue(value["observer_output_bit_exact"])
        self.assertTrue(value["raw_qk_and_derived_role_proxies_zeroized"])
        self.assertEqual(value["capture_bank"]["capture_count"], 144)
        relational = value["relational_observer"]
        self.assertEqual(
            [
                (row["source_role"], row["target_role"], row["relation_type"])
                for row in relational["edge_registry"]
            ],
            [
                ("human_agent", "moving_object", "relative_motion"),
                (
                    "moving_object",
                    "support_surface",
                    "approaching_or_receding",
                ),
            ],
        )
        self.assertEqual(
            relational["edge_registry_summary"]["required_edge_count"],
            2,
        )
        self.assertFalse(
            relational["edge_registry_summary"]["default_cartesian_product_used"]
        )
        self.assertFalse(value["target_teacher_consumed"])
        self.assertFalse(value["gpu_launch_authorized"])
        self.assertFalse(value["scientific_claim_authorized"])


if __name__ == "__main__":
    unittest.main()

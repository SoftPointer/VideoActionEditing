#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_self_generated_intermediate_anchor_canary_v1 as runner
import self_generated_intermediate_action_anchor_v1 as core


class NativeIntermediateAnchorCanaryTests(unittest.TestCase):
    def test_368x656_world4_contract_has_explicit_frozen_base(self) -> None:
        config = core.AnchorConfig(patch_height=23, patch_width=41)
        value = runner.native_runtime_contract(config)
        self.assertEqual(value["world4_tensor_geometry"]["target_tokens"], 19_803)
        self.assertEqual(value["world4_tensor_geometry"]["global_tokens"], 43_378)
        self.assertEqual(
            value["world4_tensor_geometry"]["padded_local_hidden_shape"],
            [1, 10_845, 1536],
        )
        self.assertEqual(value["controls"][0], "FROZEN_BASE_P0a")
        self.assertEqual(value["controls"][-1], "FROZEN_BASE_P0b_exact_replay")
        self.assertFalse(value["target_isolation"]["real_target_inputs"])

    def test_paired_state_authority_uses_one_callable_and_rejects_tamper(self) -> None:
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

        noisy = torch.zeros(1)
        timestep = torch.zeros(1)
        rotary = torch.zeros(1)
        action_prompt = torch.ones(1, 5, 3)
        noop_prompt = torch.zeros(1, 4, 3)
        authority = runner.seal_paired_state_forward(
            shared_step=shared_step,
            action_kwargs={
                "model_id": "transformer_1",
                "noisy_latents": noisy,
                "timesteps": timestep,
                "cond_embeds": action_prompt,
                "rotary_embs": rotary,
                "batch_vae_seqlen": [7],
                "batch_text_seqlen": [5],
            },
            canonical_noop_embeds=noop_prompt,
            canonical_noop_instruction=core.CANONICAL_NOOP_INSTRUCTION,
            canonical_noop_instruction_sha256=core.CANONICAL_NOOP_SHA256,
        )
        self.assertIs(authority.action_call(), action_prompt)
        self.assertIs(authority.noop_call(), noop_prompt)
        self.assertTrue(authority.receipt()["same_original_callable"])
        authority.noop_kwargs["noisy_latents"] = noisy.clone()
        with self.assertRaises(runner.NativeIntermediateAnchorCanaryError):
            authority.validate()

    def test_full_cpu_dry_run_closes_freeze_and_p0(self) -> None:
        value = runner._dry_run(0.06)
        self.assertTrue(value["frozen_base_unchanged"])
        self.assertTrue(value["p0_exact_replay"]["bit_exact"])
        self.assertTrue(value["teacher_observer_output_exact"])
        self.assertTrue(value["student_changed"])
        self.assertFalse(value["target_inputs_consumed"])
        self.assertFalse(value["teacher_decode_called"])
        audit = value["teacher_student_bridge"]["injection_audits"][0]
        self.assertTrue(audit["protected_rows_bit_exact"])
        self.assertTrue(audit["phase0_rows_bit_exact"])


if __name__ == "__main__":
    unittest.main()

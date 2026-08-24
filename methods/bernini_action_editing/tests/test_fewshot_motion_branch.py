from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - lightweight local checkout
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import counterfactual_proposal_motion_branch as cpmr  # noqa: E402
import fewshot_motion_branch as fewshot  # noqa: E402


class PassProjection(nn.Module):
    in_features = cpmr.HIDDEN_SIZE
    out_features = cpmr.HIDDEN_SIZE

    def __init__(self, *, bias: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale + self.bias.to(dtype=value.dtype)


class Projector:
    def _project_qkv(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        origin_hidden_states_seq_len,
        is_cross_attn,
    ):
        del rotary_emb, origin_hidden_states_seq_len, is_cross_attn
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        return query, key, value


class DonorAttention(nn.Module):
    def __init__(self, *, output_bias: float = 0.0) -> None:
        super().__init__()
        self.heads = cpmr.ATTENTION_HEADS
        self.inner_dim = cpmr.HIDDEN_SIZE
        self.inner_kv_dim = cpmr.HIDDEN_SIZE
        self.out_dim = cpmr.HIDDEN_SIZE
        self.cross_attention_dim = cpmr.HIDDEN_SIZE
        self.to_q = PassProjection()
        self.to_k = PassProjection()
        self.to_v = PassProjection()
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.to_out = nn.ModuleList(
            [PassProjection(bias=output_bias), nn.Identity()]
        )
        self.processor = Projector()


class BaseProcessor:
    def __call__(self, attn, hidden_states, **kwargs):
        del attn, kwargs
        return hidden_states


class Attn2Box(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.processor = BaseProcessor()

    def set_processor(self, processor) -> None:
        self.processor = processor


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = DonorAttention(output_bias=7.0)
        self.attn2 = Attn2Box()


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [FakeBlock() for _ in range(cpmr.EXPECTED_BLOCK_COUNT)]
        )


class FakeDiffusion(nn.Module):
    transformer_2 = None

    def __init__(self, transformer: FakeTransformer) -> None:
        super().__init__()
        self.transformer = transformer
        self.break_binding_identity = False
        self.calls = 0

    def shared_step(
        self,
        *,
        model_id,
        noisy_latents,
        timesteps,
        cond_embeds,
        rotary_embs,
        batch_vae_seqlen,
        batch_text_seqlen,
    ):
        del model_id, timesteps, rotary_embs, batch_vae_seqlen, batch_text_seqlen
        self.calls += 1
        invocation = cpmr.current_cpmr_motion_invocation()
        if invocation is not None and invocation.routes_motion:
            internal = cond_embeds.clone()
            for index in cpmr.MOTION_BLOCK_INDICES:
                processor = self.transformer.blocks[index].attn2.processor
                observed = internal.clone() if self.break_binding_identity else internal
                invocation.conditioned_encoder_binding.observe(
                    processor._patch_token, index, observed
                )
        return noisy_latents[:, :, : fewshot.OUTPUT_PATCH_WIDTH]


def active_payload(*, dtype=torch.float16):
    phases = torch.zeros((1, cpmr.LATENT_PHASES, 1, 1), dtype=dtype)
    phases[:, 1:] = 1
    carrier = phases.expand(
        1,
        cpmr.LATENT_PHASES,
        cpmr.CARRIER_TOKENS_PER_PHASE,
        cpmr.HIDDEN_SIZE,
    ).reshape(1, cpmr.CARRIER_TOKENS, cpmr.HIDDEN_SIZE)
    activity = torch.zeros((1, cpmr.LATENT_PHASES), dtype=torch.bool)
    activity[:, 1:] = True
    return carrier, activity


class FewShotMotionBranchTests(unittest.TestCase):
    def make_motion(self, *, block_index: int = 0):
        return fewshot.FewShotMotionCrossAttention(
            DonorAttention(output_bias=7.0), block_index=block_index
        )

    def test_01_tied_code_is_36d_and_initializes_to_point_one(self):
        module = fewshot.TiedHeadEpisodicMotionCode()
        self.assertEqual(sum(item.numel() for item in module.parameters()), 36)
        code = module()
        self.assertTrue(torch.equal(code.phase_gates[:, :1], torch.zeros(1, 1)))
        torch.testing.assert_close(
            code.phase_gates[:, 1:], torch.full((1, 20), 0.1)
        )
        torch.testing.assert_close(
            code.block_head_gates, torch.full((1, 16, 12), 0.1)
        )
        for block in range(16):
            for head in range(1, 12):
                self.assertTrue(
                    torch.equal(
                        code.block_head_gates[:, block, 0],
                        code.block_head_gates[:, block, head],
                    )
                )
        self.assertEqual(module.receipt()["trainable_dimension"], 36)

    def test_02_true_heads_give_phase_and_block_gradients_independently(self):
        module = fewshot.TiedHeadEpisodicMotionCode(initial_action_gate=0.0)
        motion = self.make_motion(block_index=3)
        heads = torch.ones((1, 3, 12, 128), dtype=torch.float32)
        phase_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        output = motion.gate_and_merge_projected_heads(
            heads, phase_ids, module()
        )
        loss = output.square().mean() + output.mean()
        phase_grad, block_grad = torch.autograd.grad(
            loss, (module.phase_logits_nonzero, module.block_logits)
        )
        self.assertGreater(float(phase_grad[:, :3].abs().sum()), 0.0)
        self.assertEqual(int(torch.count_nonzero(phase_grad[:, 3:])), 0)
        self.assertGreater(float(block_grad[:, 3].abs().sum()), 0.0)
        inactive_blocks = torch.cat((block_grad[:, :3], block_grad[:, 4:]), dim=1)
        self.assertEqual(int(torch.count_nonzero(inactive_blocks)), 0)

        phase_only = fewshot.TiedHeadEpisodicMotionCode(initial_action_gate=0.0)
        with torch.no_grad():
            phase_only.phase_logits_nonzero[:, 0] = 0.4
        phase_output = motion.gate_and_merge_projected_heads(
            heads[:, :1], phase_ids[:1], phase_only()
        )
        block_only = fewshot.TiedHeadEpisodicMotionCode(initial_action_gate=0.0)
        with torch.no_grad():
            block_only.block_logits[:, 3] = 0.4
        block_output = motion.gate_and_merge_projected_heads(
            heads[:, :1], phase_ids[:1], block_only()
        )
        self.assertGreater(float(phase_output.abs().sum()), 0.0)
        self.assertGreater(float(block_output.abs().sum()), 0.0)

    def test_03_source_and_phase0_have_zero_output_and_zero_gradients(self):
        code_module = fewshot.TiedHeadEpisodicMotionCode(initial_action_gate=0.0)
        motion = self.make_motion()
        heads = torch.randn((1, 2, 12, 128), requires_grad=True)
        phase_ids = torch.tensor([-1, 0], dtype=torch.int64)
        output = motion.gate_and_merge_projected_heads(
            heads, phase_ids, code_module()
        )
        self.assertEqual(int(torch.count_nonzero(output)), 0)
        self.assertEqual(
            int(torch.count_nonzero(output.detach().reshape(-1).view(torch.uint8))),
            0,
        )
        loss = output.sum()
        phase_grad, block_grad, head_grad = torch.autograd.grad(
            loss,
            (
                code_module.phase_logits_nonzero,
                code_module.block_logits,
                heads,
            ),
            allow_unused=True,
        )
        for gradient in (phase_grad, block_grad, head_grad):
            if gradient is not None:
                self.assertEqual(int(torch.count_nonzero(gradient)), 0)

    def test_04_noop_is_exact_zero_even_with_output_projection_bias(self):
        motion = self.make_motion(block_index=2)
        heads = torch.randn((1, 4, 12, 128), dtype=torch.float32)
        phase_ids = torch.tensor([-1, 0, 1, 20], dtype=torch.int64)
        noop = fewshot.canonical_tied_noop_motion_code(device=heads.device)
        output = motion.gate_and_merge_projected_heads(heads, phase_ids, noop)
        self.assertEqual(int(torch.count_nonzero(output)), 0)
        self.assertEqual(
            int(torch.count_nonzero(output.reshape(-1).view(torch.uint8))), 0
        )

    def test_05_full_and_four_way_ids_use_official_pad_slice_order(self):
        events = []

        def pad(value, dim):
            events.append(("pad", tuple(value.shape), dim))
            return value

        def identity(value, dim):
            events.append(("slice", tuple(value.shape), dim, "full"))
            return value

        full = fewshot.local_query_phase_ids(
            device="cpu",
            expected_local_queries=39_060,
            padding_tensor_fn=pad,
            slice_input_tensor_fn=identity,
        )
        self.assertEqual(tuple(full.shape), (39_060,))
        self.assertTrue(torch.equal(full[:19_530], torch.full((19_530,), -1)))
        for phase in range(21):
            start = 19_530 + phase * 930
            self.assertTrue(torch.equal(full[start : start + 930], torch.full((930,), phase)))

        shards = []
        for rank in range(4):
            def shard(value, dim, rank=rank):
                events.append(("slice", tuple(value.shape), dim, rank))
                return value[:, rank * 9_765 : (rank + 1) * 9_765]

            shards.append(
                fewshot.local_query_phase_ids(
                    device="cpu",
                    expected_local_queries=9_765,
                    padding_tensor_fn=pad,
                    slice_input_tensor_fn=shard,
                )
            )
        self.assertTrue(torch.equal(torch.cat(shards), full))
        self.assertEqual(sum(item[0] == "pad" for item in events), 5)
        self.assertEqual(sum(item[0] == "slice" for item in events), 5)

    def test_06_one_call_training_hook_and_noop_unpatched_parity(self):
        transformer = FakeTransformer()
        transformer.requires_grad_(False)
        diffusion = FakeDiffusion(transformer)
        handle = fewshot.install_fewshot_motion_branch(diffusion)
        noisy = torch.zeros((1, 39_060, 1_536), dtype=torch.float16)
        cond = torch.zeros((1, 3, 4_096), dtype=torch.float16)
        timestep = torch.tensor([500], dtype=torch.int64)
        rotary = torch.zeros((1, 1), dtype=torch.float16)
        carrier, activity = active_payload()

        baseline = diffusion.shared_step(
            model_id="transformer_1",
            noisy_latents=noisy,
            timesteps=timestep,
            cond_embeds=cond,
            rotary_embs=rotary,
            batch_vae_seqlen=[39_060],
            batch_text_seqlen=[3],
        )
        result = fewshot.run_training_shared_step(
            diffusion,
            patch_handle=handle,
            motion_code=fewshot.canonical_tied_noop_motion_code(),
            carrier=carrier,
            activity=activity,
            model_id="transformer_1",
            noisy_latents=noisy,
            timesteps=timestep,
            cond_embeds=cond,
            rotary_embs=rotary,
            batch_vae_seqlen=[39_060],
            batch_text_seqlen=[3],
            require_code_grad=False,
        )
        self.assertTrue(torch.equal(result.prediction, baseline))
        self.assertEqual(tuple(result.prediction.shape), (1, 39_060, 64))
        self.assertEqual(diffusion.calls, 2)
        receipt = result.receipt()
        self.assertEqual(receipt["shared_step_calls"], 1)
        self.assertEqual(receipt["source_span"], [0, 19_530])
        self.assertEqual(receipt["target_span"], [19_530, 39_060])
        self.assertEqual(receipt["target_tokens_per_phase"], 930)
        self.assertTrue(receipt["conditioned_encoder_binding"]["completed"])
        self.assertNotIn("target", receipt["inference_conditions"])
        self.assertIn("mask", receipt["forbidden_inference_conditions"])
        handle.restore()

    def test_06b_public_sampling_context_has_no_privileged_inputs(self):
        parameters = tuple(
            inspect.signature(fewshot.fewshot_motion_code_context).parameters
        )
        self.assertEqual(parameters, ("patch_handle", "motion_code"))
        for forbidden in (
            "target",
            "support",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        ):
            self.assertNotIn(forbidden, parameters)

        transformer = FakeTransformer()
        transformer.requires_grad_(False)
        handle = fewshot.install_fewshot_motion_branch(transformer)
        code = fewshot.canonical_tied_noop_motion_code()
        self.assertIsNone(fewshot._CURRENT_CODE.get())
        with fewshot.fewshot_motion_code_context(
            patch_handle=handle, motion_code=code
        ) as bound:
            self.assertIs(bound, code)
            self.assertIs(fewshot._CURRENT_CODE.get().motion_code, code)
            with self.assertRaisesRegex(
                fewshot.FewShotMotionBranchContractError, "nested"
            ):
                with fewshot.fewshot_motion_code_context(
                    patch_handle=handle, motion_code=code
                ):
                    pass
        self.assertIsNone(fewshot._CURRENT_CODE.get())
        handle.restore()
        with self.assertRaisesRegex(
            fewshot.FewShotMotionBranchContractError, "already restored"
        ):
            with fewshot.fewshot_motion_code_context(
                patch_handle=handle, motion_code=code
            ):
                pass

    def test_07_training_hook_rejects_nonexact_binding_and_trainable_base(self):
        transformer = FakeTransformer()
        transformer.requires_grad_(False)
        diffusion = FakeDiffusion(transformer)
        handle = fewshot.install_fewshot_motion_branch(diffusion)
        noisy = torch.zeros((1, 39_060, 1_536), dtype=torch.float16)
        cond = torch.zeros((1, 3, 4_096), dtype=torch.float16)
        carrier, activity = active_payload()
        kwargs = dict(
            diffusion=diffusion,
            patch_handle=handle,
            motion_code=fewshot.TiedHeadEpisodicMotionCode()(),
            carrier=carrier,
            activity=activity,
            model_id="transformer_1",
            noisy_latents=noisy,
            timesteps=torch.tensor([500]),
            cond_embeds=cond,
            rotary_embs=torch.zeros((1, 1), dtype=torch.float16),
            batch_vae_seqlen=[39_060],
            batch_text_seqlen=[3],
        )
        diffusion.break_binding_identity = True
        with self.assertRaisesRegex(cpmr.CPMRMotionBranchContractError, "object changed"):
            fewshot.run_training_shared_step(**kwargs)
        diffusion.break_binding_identity = False
        next(transformer.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(
            fewshot.FewShotMotionBranchContractError, "base/clone parameters"
        ):
            fewshot.run_training_shared_step(**kwargs)
        next(transformer.parameters()).requires_grad_(False)
        transformer.gradient_checkpointing = True
        with self.assertRaisesRegex(
            fewshot.FewShotMotionBranchContractError, "checkpointing disabled"
        ):
            fewshot.run_training_shared_step(**kwargs)
        handle.restore()

    def test_08_training_hook_rejects_no_grad_and_non_transformer1(self):
        transformer = FakeTransformer()
        transformer.requires_grad_(False)
        diffusion = FakeDiffusion(transformer)
        handle = fewshot.install_fewshot_motion_branch(diffusion)
        noisy = torch.zeros((1, 39_060, 1_536), dtype=torch.float16)
        cond = torch.zeros((1, 3, 4_096), dtype=torch.float16)
        carrier, activity = active_payload()
        common = dict(
            diffusion=diffusion,
            patch_handle=handle,
            motion_code=fewshot.TiedHeadEpisodicMotionCode()(),
            carrier=carrier,
            activity=activity,
            noisy_latents=noisy,
            timesteps=torch.tensor([500]),
            cond_embeds=cond,
            rotary_embs=torch.zeros((1, 1), dtype=torch.float16),
            batch_vae_seqlen=[39_060],
            batch_text_seqlen=[3],
        )
        with self.assertRaisesRegex(
            fewshot.FewShotMotionBranchContractError, "transformer_1"
        ):
            fewshot.run_training_shared_step(model_id="transformer_2", **common)
        with torch.no_grad():
            with self.assertRaisesRegex(
                fewshot.FewShotMotionBranchContractError, "autograd enabled"
            ):
                fewshot.run_training_shared_step(model_id="transformer_1", **common)
        handle.restore()


if __name__ == "__main__":
    unittest.main()

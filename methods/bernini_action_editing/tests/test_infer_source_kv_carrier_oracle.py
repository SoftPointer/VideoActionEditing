from __future__ import annotations

from contextlib import contextmanager
import inspect
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_source_kv_carrier_oracle as oracle  # noqa: E402


class _FakeTensor:
    def __init__(self, name: str, shape) -> None:
        self.name = name
        self.shape = tuple(shape)

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != len(self.shape):
            raise AssertionError(f"unexpected fake tensor slice: {key!r}")
        result = []
        for size, selector in zip(self.shape, key):
            if isinstance(selector, slice):
                start, stop, step = selector.indices(size)
                result.append(len(range(start, stop, step)))
            elif isinstance(selector, int):
                continue
            else:
                raise AssertionError(f"unexpected fake selector: {selector!r}")
        return _FakeTensor(f"{self.name}:slice", result)


class _FakePrompt(_FakeTensor):
    def __init__(self, name: str, length: int = 7) -> None:
        super().__init__(name, (1, length, 32))


class _FakeBank:
    def __init__(self) -> None:
        self.identity = None
        self.clear_calls = 0

    def clear(self) -> None:
        if self.identity is None:
            raise AssertionError("fake bank cleared while empty")
        self.identity = None
        self.clear_calls += 1


class _FakeDiffusion:
    transformer = object()
    transformer_2 = None

    def __init__(self) -> None:
        self.forward_calls = []

    def shared_step(
        self,
        model_id,
        noisy_latents,
        timesteps,
        cond_embeds,
        rotary_embs,
        batch_vae_seqlen=None,
        batch_text_seqlen=None,
        **kwargs,
    ):
        del kwargs
        self.forward_calls.append(
            {
                "model_id": model_id,
                "latents": noisy_latents,
                "timestep": timesteps,
                "prompt": cond_embeds,
                "rotary": rotary_embs,
                "vae": batch_vae_seqlen,
                "text": batch_text_seqlen,
            }
        )
        return _FakeTensor("prediction", noisy_latents.shape)

    def sample(
        self,
        prompt_embeds=None,
        prompt_embeds_t2=None,
        uncond_prompt_embeds=None,
        uncond_embeds_t2=None,
        num_frames=81,
        width=496,
        height=480,
        image_vae_latents=None,
        multi_video_vae_latents=None,
        multi_image_vae_latents=None,
        num_inference_steps=40,
        guidance_mode="v2v_apg",
        omega_vid=1.25,
        omega_img=0.0,
        omega_txt=4.0,
        omega_scale=0.8,
        flow_shift=5.0,
        seed=2027,
        device="cuda",
        eta=0.5,
        norm_threshold=(50.0, 50.0),
        momentum=0.0,
    ):
        del (
            prompt_embeds_t2,
            uncond_embeds_t2,
            width,
            height,
            image_vae_latents,
            multi_video_vae_latents,
            multi_image_vae_latents,
            omega_vid,
            omega_img,
            omega_txt,
            omega_scale,
            flow_shift,
            seed,
            device,
            eta,
            norm_threshold,
            momentum,
        )
        if num_frames != 81 or num_inference_steps != 40 or guidance_mode != "v2v_apg":
            raise AssertionError("fake sample contract differs")
        for step in range(num_inference_steps):
            pair = _FakeTensor(f"pair-{step}", (1, 12, 64))
            rotary = _FakeTensor(f"rotary-{step}", (1, 1, 12, 64))
            timestep = float(1000 - step)
            self.shared_step(
                model_id="transformer_1",
                noisy_latents=pair,
                timesteps=timestep,
                cond_embeds=uncond_prompt_embeds,
                rotary_embs=rotary,
                batch_vae_seqlen=[12],
                batch_text_seqlen=[uncond_prompt_embeds.shape[1]],
            )
            self.shared_step(
                model_id="transformer_1",
                noisy_latents=pair,
                timesteps=timestep,
                cond_embeds=prompt_embeds,
                rotary_embs=rotary,
                batch_vae_seqlen=[12],
                batch_text_seqlen=[prompt_embeds.shape[1]],
            )
        return _FakeTensor("generated", (1, 16, 21, 60, 62))


def _trace(rank: int = 0, source_tokens: int = 19_530) -> dict:
    steps = [
        {
            "generation": 0,
            "step_index": step,
            "timestep_token": f"step-{step}:float64-0x1.0p+0",
            "rank": rank,
            "ulysses_size": 4,
            "model_id": "transformer_1",
            "source_tokens_runtime": source_tokens,
            "pair_tokens_runtime": 2 * source_tokens,
            "carrier_forwards": 1,
            "negative_replay_forwards": 1,
            "action_replay_forwards": 1,
            "cleared_after_both_replays": True,
        }
        for step in range(40)
    ]
    return {
        "sample_calls": 1,
        "step_count": 40,
        "unique_identity_count": 40,
        "steps": steps,
    }


def _off_trace(rank: int = 0, source_tokens: int = 19_530) -> dict:
    steps = [
        {
            "generation": 0,
            "step_index": step,
            "timestep_token": f"step-{step}:float64-0x1.0p+0",
            "rank": rank,
            "ulysses_size": 4,
            "model_id": "transformer_1",
            "source_tokens_runtime": source_tokens,
            "pair_tokens_runtime": 2 * source_tokens,
            "official_negative_forwards": 1,
            "official_action_forwards": 1,
        }
        for step in range(40)
    ]
    return {
        "sample_calls": 1,
        "step_count": 40,
        "shared_step_calls": 80,
        "unique_identity_count": 40,
        "steps": steps,
    }


def _core_receipt(selection: str = "all", source_tokens: int = 19_530) -> dict:
    indices = list(oracle.replay_core.resolve_block_indices(30, selection))
    per_block = [
        {
            "block_index": index,
            "capture_calls": 40,
            "replay_calls": 80,
            "branch_counts": {
                "frozen_action": 40,
                "frozen_negative": 40,
                "frozen_noop_carrier": 40,
            },
            "execution_phase_counts": {
                "eager": 120,
                "checkpoint_forward": 0,
                "checkpoint_recompute": 0,
            },
            "verified_post_rope_project_qkv_calls": 120,
            "post_rope_phase_counts": {
                "eager": 120,
                "checkpoint_forward": 0,
                "checkpoint_recompute": 0,
            },
            "last_source_tokens": source_tokens,
            "ulysses_observed": True,
            "rotary_emb_required_non_none": True,
        }
        for index in indices
    ]
    return {
        "block_indices": indices,
        "runtime": {
            "installed_block_count": len(indices),
            "restored": True,
            "cache": {
                "identity": None,
                "captured_blocks": [],
                "entries": [],
                "capture_calls": len(indices) * 40,
                "replay_lookups": len(indices) * 80,
                "replay_branch_counts": {
                    "frozen_action": len(indices) * 40,
                    "frozen_negative": len(indices) * 40,
                },
                "replay_phase_counts": {
                    "eager": len(indices) * 80,
                    "checkpoint_forward": 0,
                    "checkpoint_recompute": 0,
                },
                "checkpoint_context_counts": {
                    "checkpoint_forward": 0,
                    "checkpoint_recompute": 0,
                },
                "retired_identity_count": 40,
            },
            "per_block": per_block,
        },
    }


def _args(**updates):
    values = {
        "checkpoint": "/tmp/base",
        "original_source_path": "/tmp/dog.mp4",
        "expected_source_sha256": "d" * 64,
        "instruction": "Make the dog run.",
        "replay": "on",
        "block_selection": "all",
        "expected_source_tokens": 19_530,
        "num_inference_steps": 40,
        "seed": 2027,
        "expected_bernini_commit": oracle.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
        "expected_veomni_commit": oracle.legacy.trainer.VEOMNI_TESTED_COMMIT,
        "expected_checkpoint_tree_sha256": oracle.legacy.trainer.CHECKPOINT_TREE_SHA256,
        "method_source_revision": "a" * 40,
        "method_source_archive_sha256": "b" * 64,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _source_metadata():
    return {
        "frame_count": 81,
        "fps": 25.0,
        "source_derived_bucket_hw": [480, 496],
    }


class CarrierHookTests(unittest.TestCase):
    def test_exact40_call_order_capture_negative_action_then_clear(self) -> None:
        diffusion = _FakeDiffusion()
        action = _FakePrompt("action")
        negative = _FakePrompt("negative")
        noop = _FakePrompt("noop", length=5)
        source = _FakeTensor("source", (1, 16, 21, 60, 62))
        bank = _FakeBank()
        invocations = []

        @contextmanager
        def fake_invocation(cache_bank, **kwargs):
            self.assertIs(cache_bank, bank)
            identity = (
                kwargs["generation"],
                kwargs["step_index"],
                kwargs["timestep_token"],
                kwargs["rank"],
                kwargs["ulysses_size"],
            )
            if kwargs["mode"] == oracle.replay_core.CAPTURE_MODE:
                self.assertIsNone(bank.identity)
                bank.identity = identity
            else:
                self.assertEqual(bank.identity, identity)
            invocations.append((kwargs["step_index"], kwargs["branch_tag"]))
            yield SimpleNamespace(**kwargs)

        before_sample = diffusion.sample.__func__
        before_shared = diffusion.shared_step.__func__
        with mock.patch.object(
            oracle.replay_core,
            "source_kv_replay_invocation",
            side_effect=fake_invocation,
        ):
            with oracle.source_kv_carrier_hook(
                diffusion,
                cache_bank=bank,
                noop_prompt_embeds=noop,
                rank=0,
                ulysses_size=4,
                expected_steps=40,
                expected_source_tokens=6,
            ) as hook:
                generated = diffusion.sample(
                    prompt_embeds=action,
                    uncond_prompt_embeds=negative,
                    multi_video_vae_latents=[source],
                    num_frames=81,
                    num_inference_steps=40,
                    guidance_mode="v2v_apg",
                    flow_shift=5.0,
                )
            self.assertTrue(hook.restored)

        self.assertEqual(generated.name, "generated")
        self.assertEqual(diffusion.sample.__func__, before_sample)
        self.assertEqual(diffusion.shared_step.__func__, before_shared)
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("shared_step", vars(diffusion))
        self.assertEqual(bank.clear_calls, 40)
        self.assertIsNone(bank.identity)
        self.assertEqual(len(diffusion.forward_calls), 120)
        for step in range(40):
            calls = diffusion.forward_calls[step * 3 : step * 3 + 3]
            self.assertEqual(
                [call["prompt"].name for call in calls],
                ["noop", "negative", "action"],
            )
            self.assertEqual(calls[0]["latents"].shape, (1, 6, 64))
            self.assertEqual(calls[0]["rotary"].shape, (1, 1, 6, 64))
            self.assertEqual(calls[0]["vae"], [6])
            self.assertEqual(calls[1]["latents"].shape, (1, 12, 64))
            self.assertIs(calls[1]["latents"], calls[2]["latents"])
            self.assertIs(calls[1]["rotary"], calls[2]["rotary"])
            self.assertIs(calls[1]["timestep"], calls[2]["timestep"])
            self.assertEqual(
                invocations[step * 3 : step * 3 + 3],
                [
                    (step, "frozen_noop_carrier"),
                    (step, "frozen_negative"),
                    (step, "frozen_action"),
                ],
            )
        trace = hook.trace.as_dict()
        self.assertEqual(trace["step_count"], 40)
        self.assertEqual(trace["unique_identity_count"], 40)
        self.assertTrue(
            all(item["source_tokens_runtime"] == 6 for item in trace["steps"])
        )
        self.assertTrue(
            all(item["pair_tokens_runtime"] == 12 for item in trace["steps"])
        )

    def test_replay_off_never_installs_processor_or_hook(self) -> None:
        model = _FakeDiffusion()
        action = _FakePrompt("action")
        negative = _FakePrompt("negative")
        source = _FakeTensor("source", (1, 16, 21, 2, 2))
        with mock.patch.object(
            oracle.replay_core,
            "source_kv_replay",
            side_effect=AssertionError("off arm touched carrier installer"),
        ), mock.patch.object(
            oracle,
            "source_kv_carrier_hook",
            side_effect=AssertionError("off arm touched carrier hook"),
        ):
            generated, core, certificate = oracle._sample_with_optional_replay(
                model,
                replay="off",
                block_selection="late",
                noop_prompt_embeds=None,
                rank=2,
                source_tokens=6,
                sample_kwargs={
                    "prompt_embeds": action,
                    "uncond_prompt_embeds": negative,
                    "multi_video_vae_latents": [source],
                    "num_frames": 81,
                    "num_inference_steps": 40,
                    "guidance_mode": "v2v_apg",
                    "flow_shift": 5.0,
                    "seed": 2027,
                },
            )
        self.assertEqual(generated.name, "generated")
        self.assertIsNone(core)
        self.assertEqual(certificate["actual_installed_block_indices"], [])
        self.assertEqual(certificate["rank_local_bank_capture_calls"], 0)
        self.assertEqual(certificate["rank_local_bank_replay_lookups"], 0)
        self.assertEqual(certificate["observed_shared_step_calls"], 80)
        self.assertEqual(certificate["unique_step_identities"], 40)
        self.assertEqual(certificate["source_tokens_runtime"], 6)
        self.assertEqual(certificate["pair_tokens_runtime"], 12)
        self.assertTrue(certificate["read_only_observer_installed"])
        self.assertTrue(certificate["observer_hook_restore"])
        self.assertEqual(len(model.forward_calls), 80)

    def test_torch_hook_and_real_cache_bank_compose_to_exact_all30_receipt(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        class Identity:
            def __call__(self, value):
                return value

        class BaseProcessor:
            def _project_qkv(
                self,
                attn,
                hidden_states,
                encoder_hidden_states,
                rotary_emb,
                origin_hidden_states_seq_len,
                is_cross_attn,
            ):
                del (
                    attn,
                    encoder_hidden_states,
                    rotary_emb,
                    origin_hidden_states_seq_len,
                    is_cross_attn,
                )
                value = hidden_states.reshape(1, hidden_states.shape[1], 2, 4)
                return value, value.clone(), value.clone()

        class Attention:
            def __init__(self):
                self.processor = BaseProcessor()
                self.to_out = (Identity(), Identity())

            def set_processor(self, value):
                self.processor = value

        class Block:
            def __init__(self):
                self.attn1 = Attention()

        class Transformer:
            def __init__(self):
                self.blocks = tuple(Block() for _ in range(30))

            def patch_vae_latent(self):
                raise AssertionError("resolver marker only")

        class Diffusion:
            transformer_2 = None

            def __init__(self):
                self.transformer = Transformer()

            def shared_step(
                self,
                model_id,
                noisy_latents,
                timesteps,
                cond_embeds,
                rotary_embs,
                batch_vae_seqlen=None,
                batch_text_seqlen=None,
                **kwargs,
            ):
                del model_id, timesteps, cond_embeds, batch_text_seqlen, kwargs
                hidden = noisy_latents
                sequence = int(hidden.shape[1])
                rotary = rotary_embs.transpose(1, 2)
                cu = torch.tensor([0, sequence], dtype=torch.int32)
                for block in self.transformer.blocks:
                    hidden = block.attn1.processor(
                        block.attn1,
                        hidden,
                        rotary_emb=rotary,
                        batch_image_vae_seqlen=batch_vae_seqlen,
                        origin_hidden_states_seq_len=sequence,
                        cu_seqlens_q_cache=cu,
                        max_seqlen_q_cache=sequence,
                    )
                return hidden

            def sample(
                self,
                prompt_embeds=None,
                prompt_embeds_t2=None,
                uncond_prompt_embeds=None,
                uncond_embeds_t2=None,
                num_frames=81,
                width=16,
                height=16,
                image_vae_latents=None,
                multi_video_vae_latents=None,
                multi_image_vae_latents=None,
                num_inference_steps=40,
                guidance_mode="v2v_apg",
                omega_vid=1.25,
                omega_img=0.0,
                omega_txt=4.0,
                omega_scale=0.8,
                flow_shift=5.0,
                seed=2027,
                device="cpu",
                eta=0.5,
                norm_threshold=(50.0, 50.0),
                momentum=0.0,
            ):
                del (
                    prompt_embeds_t2,
                    uncond_embeds_t2,
                    width,
                    height,
                    image_vae_latents,
                    multi_video_vae_latents,
                    multi_image_vae_latents,
                    omega_vid,
                    omega_img,
                    omega_txt,
                    omega_scale,
                    seed,
                    device,
                    eta,
                    norm_threshold,
                    momentum,
                )
                self.assert_contract = (
                    num_frames,
                    num_inference_steps,
                    guidance_mode,
                    flow_shift,
                )
                for step in range(40):
                    pair = torch.randn(1, 12, 8)
                    rotary = torch.ones(1, 1, 12, 2, dtype=torch.complex64)
                    timestep = torch.tensor([1000.0 - step])
                    for prompt in (uncond_prompt_embeds, prompt_embeds):
                        self.shared_step(
                            model_id="transformer_1",
                            noisy_latents=pair,
                            timesteps=timestep,
                            cond_embeds=prompt,
                            rotary_embs=rotary,
                            batch_vae_seqlen=[12],
                            batch_text_seqlen=[prompt.shape[1]],
                        )
                return torch.zeros(1, 16, 21, 2, 2)

        state = SimpleNamespace(ulysses_enabled=True, ulysses_rank=0, ulysses_size=4)

        def factory(base, index, bank):
            return oracle.replay_core.SourceKVReplaySelfAttnProcessor(
                base,
                block_index=index,
                cache_bank=bank,
                varlen_attention_fn=lambda query, key, value, **kwargs: query,
                get_parallel_state_fn=lambda: state,
                gather_heads_scatter_seq_fn=lambda value, **kwargs: value,
            )

        diffusion = Diffusion()
        patch = oracle.replay_core.install_source_kv_replay(
            diffusion, selection="all", processor_factory=factory
        )
        hook = oracle.InstalledSourceKVCarrierHook(
            diffusion,
            cache_bank=patch.cache_bank,
            noop_prompt_embeds=torch.zeros(1, 5, 16),
            rank=0,
            ulysses_size=4,
            expected_steps=40,
            expected_source_tokens=6,
        )
        hook.install()
        try:
            diffusion.sample(
                prompt_embeds=torch.ones(1, 7, 16),
                uncond_prompt_embeds=torch.zeros(1, 7, 16),
                multi_video_vae_latents=[torch.zeros(1, 16, 21, 2, 2)],
                num_frames=81,
                num_inference_steps=40,
                guidance_mode="v2v_apg",
                flow_shift=5.0,
            )
        finally:
            hook.restore()
            patch.restore()
        value = oracle.validate_enabled_runtime_certificate(
            patch.receipt(),
            hook.trace.as_dict(),
            selection="all",
            expected_source_tokens=6,
            rank=0,
            hook_restored=hook.restored,
        )
        self.assertEqual(value["rank_local_bank_capture_calls"], 1200)
        self.assertEqual(value["rank_local_bank_replay_lookups"], 2400)


class RuntimeCertificateTests(unittest.TestCase):
    def test_all_scope_locks_exact_requested_counts_and_dog_geometry(self) -> None:
        value = oracle.validate_enabled_runtime_certificate(
            _core_receipt("all"),
            _trace(),
            selection="all",
            expected_source_tokens=19_530,
            rank=0,
            hook_restored=True,
        )
        self.assertEqual(value["actual_installed_block_indices"], list(range(30)))
        self.assertEqual(value["per_layer_capture_calls"], 40)
        self.assertEqual(value["per_layer_replay_calls"], 80)
        self.assertEqual(value["per_layer_negative_replays"], 40)
        self.assertEqual(value["per_layer_action_replays"], 40)
        self.assertEqual(value["rank_local_bank_capture_calls"], 1200)
        self.assertEqual(value["rank_local_bank_replay_lookups"], 2400)
        self.assertEqual(value["unique_step_identities"], 40)
        self.assertEqual(value["source_tokens_runtime"], 19_530)
        self.assertEqual(value["pair_tokens_runtime"], 39_060)
        self.assertTrue(value["processor_restore"])
        self.assertTrue(value["sampler_hook_restore"])

    def test_late_ablation_is_20_to_29_with_scope_scaled_bank_counts(self) -> None:
        value = oracle.validate_enabled_runtime_certificate(
            _core_receipt("late"),
            _trace(),
            selection="late",
            expected_source_tokens=19_530,
            rank=0,
            hook_restored=True,
        )
        self.assertEqual(value["actual_installed_block_indices"], list(range(20, 30)))
        self.assertEqual(value["rank_local_bank_capture_calls"], 400)
        self.assertEqual(value["rank_local_bank_replay_lookups"], 800)

    def test_wrong_counts_geometry_or_restore_fail_closed(self) -> None:
        mutations = (
            lambda core, trace: core["runtime"]["cache"].update(capture_calls=1199),
            lambda core, trace: core["runtime"]["per_block"][0].update(replay_calls=79),
            lambda core, trace: trace["steps"][0].update(pair_tokens_runtime=39_059),
            lambda core, trace: core["runtime"].update(restored=False),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                core, trace = _core_receipt(), _trace()
                mutate(core, trace)
                with self.assertRaises(oracle.SourceKVCarrierOracleError):
                    oracle.validate_enabled_runtime_certificate(
                        core,
                        trace,
                        selection="all",
                        expected_source_tokens=19_530,
                        rank=0,
                        hook_restored=True,
                    )

    def test_four_rank_aggregate_distinguishes_per_rank_and_cross_rank(self) -> None:
        certificates = [
            oracle.validate_enabled_runtime_certificate(
                _core_receipt(),
                _trace(rank),
                selection="all",
                expected_source_tokens=19_530,
                rank=rank,
                hook_restored=True,
            )
            for rank in range(4)
        ]
        value = oracle.validate_four_rank_certificates(
            certificates, replay="on"
        )
        self.assertTrue(value["all_four_ranks_exact"])
        self.assertEqual(value["per_rank_capture_calls"], 1200)
        self.assertEqual(value["per_rank_replay_lookups"], 2400)
        self.assertEqual(value["cross_rank_capture_calls"], 4800)
        self.assertEqual(value["cross_rank_replay_lookups"], 9600)


class PinnedBerniniSamplerAPITests(unittest.TestCase):
    def test_pinned_shared_step_and_sample_signatures_match_hook_boundary(self) -> None:
        bernini_root = os.environ.get("BERNINI_OFFICIAL_ROOT")
        veomni_root = os.environ.get("BERNINI_VEOMNI_ROOT")
        if not bernini_root or not veomni_root:
            self.skipTest("pinned Bernini/VeOmni source roots are not set")
        root = Path(bernini_root).resolve(strict=True)
        oracle.legacy.trainer.activate_source_trees(
            root, Path(veomni_root).resolve(strict=True)
        )
        from bernini.models.wan_diffusion import GEN_Wanx22

        shared = list(inspect.signature(GEN_Wanx22.shared_step).parameters)
        sample = list(inspect.signature(GEN_Wanx22.sample).parameters)
        self.assertEqual(
            shared[:8],
            [
                "self",
                "model_id",
                "noisy_latents",
                "timesteps",
                "cond_embeds",
                "rotary_embs",
                "batch_vae_seqlen",
                "batch_text_seqlen",
            ],
        )
        for name in (
            "prompt_embeds",
            "uncond_prompt_embeds",
            "multi_video_vae_latents",
            "num_frames",
            "num_inference_steps",
            "guidance_mode",
            "flow_shift",
            "seed",
        ):
            self.assertIn(name, sample)
        self.assertEqual(
            oracle.legacy.file_sha256(root / "bernini/models/wan_diffusion.py"),
            "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512",
        )


class ReceiptAndCLIContractTests(unittest.TestCase):
    def test_cli_exposes_replay_and_scope_but_no_adapter_argument(self) -> None:
        parser = oracle.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertIn("replay", destinations)
        self.assertIn("block_selection", destinations)
        self.assertIn("checkpoint_content_manifest", destinations)
        self.assertNotIn("adapter_checkpoint", destinations)
        self.assertEqual(
            parser.get_default("block_selection"),
            oracle.replay_core.MAIN_BLOCK_SELECTION,
        )

    def test_source_has_no_legacy_adapter_loader_or_peft_import(self) -> None:
        source = Path(oracle.__file__).read_text(encoding="utf-8")
        self.assertNotIn("_strict_load_and_merge_adapter", source)
        self.assertNotIn("resolve_adapter_bundle", source)
        self.assertNotIn("import peft", source)
        self.assertNotIn("--adapter-checkpoint", source)

    def test_noop_hash_and_dog_token_geometry_are_exact(self) -> None:
        self.assertEqual(
            oracle.route_batches.EXACT_NOOP_INSTRUCTION_SHA256,
            "fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a",
        )
        self.assertEqual(
            oracle.source_tokens_from_vae_latent_shape((1, 16, 21, 60, 62)),
            19_530,
        )
        self.assertEqual(oracle.EXPECTED_DOG_PAIR_TOKENS, 39_060)

    def test_checkpoint_manifest_hashes_actual_complete_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            checkpoint = root_path / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "a.bin").write_bytes(b"a")
            nested = checkpoint / "nested"
            nested.mkdir()
            (nested / "b.bin").write_bytes(b"b")
            entries = []
            for relative in ("a.bin", "nested/b.bin"):
                entries.append(
                    f"{oracle.legacy.file_sha256(checkpoint / relative)}  ./{relative}"
                )
            manifest = root_path / "manifest.sha256"
            manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
            manifest_sha = oracle.legacy.file_sha256(manifest)
            value = oracle.validate_checkpoint_content(
                checkpoint,
                manifest,
                expected_manifest_sha256=manifest_sha,
                expected_file_count=2,
            )
            self.assertTrue(value["every_file_sha256_verified"])
            self.assertEqual(value["verified_file_count"], 2)
            (checkpoint / "a.bin").write_bytes(b"tampered")
            with self.assertRaises(oracle.SourceKVCarrierOracleError):
                oracle.validate_checkpoint_content(
                    checkpoint,
                    manifest,
                    expected_manifest_sha256=manifest_sha,
                    expected_file_count=2,
                )

    def test_source_is_decoded_from_hash_verified_private_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.mp4"
            source.write_bytes(b"immutable-video-bytes")

            def fake_prepare(path):
                self.assertNotEqual(path, source)
                self.assertEqual(path.read_bytes(), source.read_bytes())
                return "tensor", {"source_derived_bucket_hw": [480, 496]}

            with mock.patch.object(
                oracle.legacy, "prepare_exact_source", side_effect=fake_prepare
            ):
                tensor, metadata, digest = oracle.prepare_hashed_source_snapshot(
                    source
                )
            self.assertEqual(tensor, "tensor")
            self.assertEqual(digest, oracle.legacy.file_sha256(source))
            self.assertTrue(metadata["decoded_from_private_byte_snapshot"])
            self.assertEqual(metadata["snapshot_sha256"], digest)

    def test_on_and_off_receipts_bind_same_causal_pairing_digest(self) -> None:
        metadata = _source_metadata()
        pairing = oracle.causal_pairing_contract(
            method_source_revision="a" * 40,
            method_source_archive_sha256="b" * 64,
            bernini_commit=oracle.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
            veomni_commit=oracle.legacy.trainer.VEOMNI_TESTED_COMMIT,
            bernini_inference_files={},
            checkpoint_tree_sha256="c" * 64,
            checkpoint_content_identity={
                "manifest_sha256_computed": "9" * 64,
                "verified_file_count": 23,
                "every_file_sha256_verified": True,
            },
            source_sha256="d" * 64,
            instruction_sha256="e" * 64,
            action_prompt_sha256="f" * 64,
            negative_prompt_sha256="0" * 64,
            source_metadata=metadata,
            steps=40,
            seed=2027,
            runtime_versions={"torch": "test"},
        )
        all_cert = oracle.validate_enabled_runtime_certificate(
            _core_receipt(),
            _trace(),
            selection="all",
            expected_source_tokens=19_530,
            rank=0,
            hook_restored=True,
        )
        on_runtime = oracle.validate_four_rank_certificates(
            [{**all_cert, "rank": rank} for rank in range(4)], replay="on"
        )
        off_certificates = [
            oracle.disabled_runtime_certificate(
                _off_trace(rank),
                selection="all",
                source_tokens_from_input_geometry=19_530,
                rank=rank,
                observer_restored=True,
            )
            for rank in range(4)
        ]
        off_runtime = oracle.validate_four_rank_certificates(
            off_certificates, replay="off"
        )

        common = dict(
            source_path=Path("/tmp/dog.mp4"),
            source_sha256="d" * 64,
            source_metadata=metadata,
            source_tokens=19_530,
            output_path=Path("/tmp/out.mp4"),
            output_sha256="1" * 64,
            bernini_revision=oracle.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
            veomni_revision=oracle.legacy.trainer.VEOMNI_TESTED_COMMIT,
            inference_file_hashes={},
            runtime_versions={"torch": "test"},
            freeze_certificate={
                "base_frozen": True,
                "trainable_parameter_tensors": 0,
                "trainable_parameter_elements": 0,
                "lora_module_count": 0,
            },
            pairing=pairing,
            checkpoint_content_identity={
                "manifest_sha256_computed": "9" * 64,
                "verified_file_count": 23,
                "every_file_sha256_verified": True,
            },
        )
        on = oracle.build_receipt(
            args=_args(replay="on"),
            four_rank_runtime=on_runtime,
            rank0_core_receipt=_core_receipt(),
            **common,
        )
        off = oracle.build_receipt(
            args=_args(replay="off"),
            four_rank_runtime=off_runtime,
            rank0_core_receipt=None,
            **common,
        )
        self.assertEqual(
            on["causal_control"]["causal_pairing_digest"],
            off["causal_control"]["causal_pairing_digest"],
        )
        self.assertNotIn("adapter", on)
        self.assertNotIn("adapter", off)
        for value in (on, off):
            weights = value["weights"]
            self.assertTrue(weights["base_checkpoint_loaded"])
            self.assertTrue(weights["base_frozen"])
            self.assertTrue(weights["base_checkpoint_content_verified"])
            self.assertEqual(weights["base_checkpoint_verified_file_count"], 23)
            self.assertEqual(
                weights["base_checkpoint_content_manifest_sha256"], "9" * 64
            )
            self.assertFalse(weights["adapter_argument_supported"])
            self.assertFalse(weights["legacy_full644_artifact_used"])
            self.assertFalse(weights["adapter_weights_loaded"])
            self.assertFalse(weights["adapter_weights_merged"])
            self.assertFalse(weights["peft_model_constructed"])
            self.assertEqual(weights["lora_module_count"], 0)
            optimization = value["optimization"]
            self.assertTrue(optimization["zero_training"])
            self.assertEqual(optimization["training_steps"], 0)
            self.assertEqual(optimization["backward_calls"], 0)
            self.assertEqual(optimization["trainable_parameter_tensors"], 0)
            stored = value.pop("receipt_digest")
            self.assertEqual(stored, oracle.legacy.object_sha256(value))


class AUHLauncherContractTests(unittest.TestCase):
    def test_launcher_has_four_rank_two_arm_scope_and_no_adapter_dependency(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts/auh_infer_source_kv_carrier_oracle.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("--nproc_per_node=4", launcher)
        self.assertIn('--replay "${replay_mode}"', launcher)
        self.assertIn('--block-selection "${block_selection}"', launcher)
        self.assertIn(
            '--expected-source-tokens "${expected_source_tokens}"', launcher
        )
        self.assertIn('[[ "${expected_source_tokens}" == 19530 ]]', launcher)
        self.assertIn("expected_pair_tokens=39060", launcher)
        self.assertIn("rank_local_bank_capture_calls", launcher)
        self.assertIn("rank_local_bank_replay_lookups", launcher)
        self.assertIn("validate_checkpoint_content", launcher)
        self.assertIn(
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
            launcher,
        )
        self.assertIn('cp -- "${source_video}" "${staged_source_video}"', launcher)
        self.assertIn('--source-video "${staged_source_video}"', launcher)
        self.assertIn('--original-source-path "${source_video}"', launcher)
        self.assertIn('--expected-source-sha256 "${staged_source_sha256}"', launcher)
        self.assertIn("observed_shared_step_calls", launcher)
        self.assertIn("actual_output_sha256", launcher)
        self.assertNotIn("BERNINI_ACTION_ADAPTER_CHECKPOINT", launcher)
        self.assertNotIn("--adapter-checkpoint", launcher)
        self.assertNotIn("full644_adapter_sha256", launcher)


if __name__ == "__main__":
    unittest.main()

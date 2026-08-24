from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import inspect
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    from torch import nn

    import inference_sigma_strata as strata
    import saic_online_motion_native_runtime_v1 as runtime
    import saic_source_anchor_adapter_v1 as source_anchor
    import saic_temporal_action_operator_v2 as operator

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    strata = None  # type: ignore[assignment]
    runtime = None  # type: ignore[assignment]
    operator = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    HIDDEN = 8
    TEXT_TOKENS = 3
    TEXT_DIM = 4


    class _Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.to_k = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.to_v = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.to_out = nn.ModuleList(
                [nn.Linear(HIDDEN, HIDDEN, bias=False), nn.Identity()]
            )


    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn1 = _Attention()
            self.attn2 = _Attention()
            self.gradient_checkpointing = False


    class _Transformer(nn.Module):
        def __init__(self, owner: "GEN_Wanx22") -> None:
            super().__init__()
            self.owner = owner
            self.config = SimpleNamespace(
                num_attention_heads=2,
                attention_head_dim=4,
                in_channels=16,
                out_channels=16,
                patch_size=(1, 2, 2),
                text_dim=TEXT_DIM,
            )
            self.patch_embedding = nn.Conv3d(
                16, HIDDEN, kernel_size=(1, 2, 2), bias=False
            )
            self.blocks = nn.ModuleList(
                [_Block() for _ in range(operator.TOTAL_BLOCKS_1P3B)]
            )
            self.gradient_checkpointing = False
            self.is_gradient_checkpointing = False

        @property
        def dtype(self):
            return self.patch_embedding.weight.dtype

        def patch_vae_latent(
            self, hidden_states: torch.Tensor, source_id=None
        ):
            embedded = self.patch_embedding(hidden_states)
            tokens = embedded.permute(0, 2, 3, 4, 1).reshape(
                hidden_states.shape[0], -1, HIDDEN
            )
            rotary = torch.zeros(
                1, 1, tokens.shape[1], 2, dtype=tokens.dtype, device=tokens.device
            )
            # source_id affects only the test transformer's condition tokens;
            # target source_id=0 remains the raw-state-derived native patch.
            tokens = tokens + float(source_id)
            return tokens, rotary

        def forward(
            self,
            hidden_states,
            timesteps,
            encoder_hidden_states=None,
            rotary_emb=None,
            batch_image_vae_seqlen=None,
            text_features_length=None,
        ):
            del timesteps, rotary_emb, batch_image_vae_seqlen, text_features_length
            active = operator.active_route()
            x = self.blocks[0].attn2.to_q(hidden_states)
            x = torch.tanh(x)
            x = self.blocks[0].attn2.to_out[0](x)
            token_coordinate = torch.arange(
                hidden_states.shape[1],
                dtype=torch.float32,
                device=hidden_states.device,
            ).reshape(1, -1, 1)
            prompt_scale = encoder_hidden_states.float().mean().reshape(1, 1, 1)
            result = x.float().mean(dim=-1, keepdim=True)
            result = result + prompt_scale * (token_coordinate + 1.0) / 32.0
            result = result.expand(-1, -1, 64).contiguous()
            prompt_name = self.owner.prompt_names[id(encoder_hidden_states)]
            if hidden_states.shape[1] == self.owner.target_tokens:
                label = f"teacher-{prompt_name}"
            elif prompt_name == "negative":
                label = "native-negative"
            else:
                label = "native-action"
            self.owner.forward_log.append(
                {
                    "label": label,
                    "active_route": active is not None,
                    "learned_path_active": (
                        bool(active.operator_active) if active is not None else False
                    ),
                    "tokens": int(hidden_states.shape[1]),
                    "output_sum": float(result.sum().item()),
                }
            )
            return SimpleNamespace(sample=result)


    class UniPCMultistepScheduler:
        def __init__(self) -> None:
            self.config = dict(runtime.PINNED_SCHEDULER_CONFIG)
            self.timesteps = torch.tensor(strata.PINNED_TIMESTEPS, dtype=torch.int64)
            self.sigmas = torch.tensor(
                (*strata.PINNED_POSITIVE_SIGMAS, 0.0), dtype=torch.float32
            )
            self.step_index = None

        def set_timesteps(self, count):
            if count != 40:
                raise RuntimeError("test scheduler is exact40 only")
            self.timesteps = torch.tensor(strata.PINNED_TIMESTEPS, dtype=torch.int64)
            self.sigmas = torch.tensor(
                (*strata.PINNED_POSITIVE_SIGMAS, 0.0), dtype=torch.float32
            )
            self.step_index = None

        def step(self, model_output, timestep, sample, return_dict=True):
            if return_dict is not False:
                raise RuntimeError("vendor must pass return_dict=False")
            expected = 0 if self.step_index is None else self.step_index
            if int(timestep.item()) != strata.PINNED_TIMESTEPS[expected]:
                raise RuntimeError("scheduler coordinate differs")
            self.step_index = expected + 1
            return (sample - model_output.float() * 1.0e-5,)


    UniPCMultistepScheduler.__module__ = runtime.PINNED_SCHEDULER_CLASS[0]


    def _pack(value: torch.Tensor) -> torch.Tensor:
        batch, channels, phases, height, width = value.shape
        return (
            value.reshape(batch, channels, phases, height // 2, 2, width // 2, 2)
            .permute(0, 2, 3, 5, 4, 6, 1)
            .reshape(batch, phases * (height // 2) * (width // 2), 64)
            .contiguous()
        )


    def _unpack(value: torch.Tensor, *, phases: int, height: int, width: int):
        return (
            value.reshape(1, phases, height // 2, width // 2, 2, 2, 16)
            .permute(0, 6, 1, 2, 4, 3, 5)
            .reshape(1, 16, phases, height, width)
            .contiguous()
        )


    class GEN_Wanx22:
        use_unipc = True
        transformer_2 = None
        switch_dit_boundary = 0.0
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 8

        def __init__(self, *, corrupt_target_source_id: bool = False) -> None:
            self.corrupt_target_source_id = corrupt_target_source_id
            self.scheduler = UniPCMultistepScheduler()
            self.transformer = _Transformer(self)
            self.prompt_names = {}
            self.forward_log = []
            self.target_tokens = 21

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
            if model_id != "transformer_1":
                raise RuntimeError("single expert only")
            return self.transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=cond_embeds,
                rotary_emb=rotary_embs,
                batch_image_vae_seqlen=batch_vae_seqlen,
                text_features_length=batch_text_seqlen,
            ).sample

        @torch.no_grad()
        def sample(
            self,
            prompt_embeds=None,
            prompt_embeds_t2=None,
            uncond_prompt_embeds=None,
            uncond_embeds_t2=None,
            num_frames=1,
            width=16,
            height=16,
            image_vae_latents=None,
            multi_video_vae_latents=None,
            multi_image_vae_latents=None,
            num_inference_steps=50,
            guidance_mode="rv2v",
            omega_vid=3.0,
            omega_img=3.0,
            omega_txt=4.0,
            omega_scale=0.75,
            flow_shift=5.0,
            seed=42,
            device="cpu",
            eta=1.0,
            norm_threshold=(50.0, 50.0),
            momentum=0.0,
        ):
            del omega_vid, omega_img, omega_scale, seed, eta, norm_threshold, momentum
            if prompt_embeds_t2 is not None or uncond_embeds_t2 is not None:
                raise RuntimeError("test expects one text encoder")
            if image_vae_latents is not None or guidance_mode != "v2v_apg":
                raise RuntimeError("test expects native v2v_apg")
            if flow_shift != 5.0:
                raise RuntimeError("test expects flow shift 5")
            self.scheduler.set_timesteps(num_inference_steps)
            latent_phases = (num_frames - 1) // 4 + 1
            latent_height, latent_width = height // 8, width // 8
            self.target_tokens = latent_phases * (latent_height // 2) * (latent_width // 2)
            raw = torch.linspace(
                -0.8,
                0.8,
                16 * latent_phases * latent_height * latent_width,
                dtype=torch.float32,
                device=device,
            ).reshape(1, 16, latent_phases, latent_height, latent_width)
            noisy = _pack(raw)
            videos = list(multi_video_vae_latents)
            references = [] if multi_image_vae_latents is None else list(
                multi_image_vae_latents
            )
            self.prompt_names[id(prompt_embeds)] = "action"
            self.prompt_names[id(uncond_prompt_embeds)] = "negative"
            for t in self.scheduler.timesteps.to(device):
                vi_latents, vi_rotary = [], []
                source_tokens, source_rotary = self.transformer.patch_vae_latent(
                    videos[0].to(dtype=self.transformer.dtype), source_id=1.0
                )
                vi_latents.append(source_tokens)
                vi_rotary.append(source_rotary)
                for index, reference in enumerate(references):
                    vi_tokens, vi_rope = self.transformer.patch_vae_latent(
                        reference.to(dtype=self.transformer.dtype),
                        source_id=float(index + 2),
                    )
                    vi_latents.append(vi_tokens)
                    vi_rotary.append(vi_rope)
                    # Official image-only axis repatch; its result is not part
                    # of the VI forward but its call order is native state.
                    self.transformer.patch_vae_latent(
                        reference.to(dtype=self.transformer.dtype),
                        source_id=float(index + 1),
                    )
                current_raw = _unpack(
                    noisy,
                    phases=latent_phases,
                    height=latent_height,
                    width=latent_width,
                ).to(self.transformer.dtype)
                target_tokens, target_rotary = self.transformer.patch_vae_latent(
                    current_raw,
                    source_id=0.5 if self.corrupt_target_source_id else 0.0,
                )
                all_tokens = torch.cat((*vi_latents, target_tokens), dim=1)
                all_rotary = torch.cat((*vi_rotary, target_rotary), dim=2)
                target_mask = torch.zeros(
                    all_tokens.shape[1], dtype=torch.bool, device=all_tokens.device
                )
                target_mask[-self.target_tokens :] = True
                timestep = t.expand(1)
                negative = self.shared_step(
                    model_id="transformer_1",
                    noisy_latents=all_tokens,
                    timesteps=timestep,
                    cond_embeds=uncond_prompt_embeds,
                    rotary_embs=all_rotary,
                    batch_vae_seqlen=[all_tokens.shape[1]],
                    batch_text_seqlen=[uncond_prompt_embeds.shape[1]],
                )
                action = self.shared_step(
                    model_id="transformer_1",
                    noisy_latents=all_tokens,
                    timesteps=timestep,
                    cond_embeds=prompt_embeds,
                    rotary_embs=all_rotary,
                    batch_vae_seqlen=[all_tokens.shape[1]],
                    batch_text_seqlen=[prompt_embeds.shape[1]],
                )
                model_output = negative[:, target_mask, :] + omega_txt * (
                    action[:, target_mask, :] - negative[:, target_mask, :]
                )
                noisy = self.scheduler.step(
                    model_output, t, noisy, return_dict=False
                )[0]
            return _unpack(
                noisy,
                phases=latent_phases,
                height=latent_height,
                width=latent_width,
            )


    GEN_Wanx22.__module__ = runtime.PINNED_DIFFUSION_CLASS[0]


    class _ParallelState:
        ulysses_rank = 0
        ulysses_size = 1


def _prompt(value: float) -> "torch.Tensor":
    return torch.full((1, TEXT_TOKENS, TEXT_DIM), value, dtype=torch.float32)


def _sample_kwargs(*, with_references: bool) -> dict:
    source = torch.linspace(-1.0, 1.0, 16 * 21 * 2 * 2).reshape(1, 16, 21, 2, 2)
    references = [
        torch.full((1, 16, 1, 2, 2), float(index) / 8.0)
        for index in range(4)
    ]
    return {
        "prompt_embeds": _prompt(2.0),
        "prompt_embeds_t2": None,
        "uncond_prompt_embeds": _prompt(-1.0),
        "uncond_embeds_t2": None,
        "num_frames": 81,
        "width": 16,
        "height": 16,
        "image_vae_latents": None,
        "multi_video_vae_latents": [source],
        "multi_image_vae_latents": references if with_references else None,
        "num_inference_steps": 40,
        "guidance_mode": "v2v_apg",
        "flow_shift": 5.0,
        "seed": 810,
        "device": "cpu",
    }


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class SAICOnlineMotionNativeRuntimeTests(unittest.TestCase):
    def _run(
        self,
        *,
        with_references: bool,
        nonzero_operator: bool = True,
        corrupt_target_source_id: bool = False,
    ):
        torch.manual_seed(810)
        diffusion = GEN_Wanx22(
            corrupt_target_source_id=corrupt_target_source_id
        )
        diffusion.transformer.requires_grad_(False)
        handle = operator.install_saic_temporal_action_operator(diffusion.transformer)
        if nonzero_operator:
            with torch.no_grad():
                for wrapper in (
                    handle.q_wrappers[0][1],
                    handle.o_wrappers[0][1],
                ):
                    wrapper.state_down.weight.fill_(0.125)
                    wrapper.phase_gate.weight.fill_(0.125)
                    wrapper.output_up.weight.fill_(0.25)
        noop = _prompt(0.5)
        action_t2v = _prompt(3.0)
        diffusion.prompt_names[id(noop)] = "noop"
        diffusion.prompt_names[id(action_t2v)] = "action"
        action_caption = (
            "A brown dog smoothly lowers its body and sits on the floor."
        )
        noop_caption = (
            "A brown dog remains standing still in the same fixed scene."
        )
        parallel = _ParallelState()
        native_runtime = None
        try:
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(runtime, "EXPECTED_HIDDEN_DIM", HIDDEN)
                )
                stack.enter_context(
                    mock.patch.object(runtime, "EXPECTED_TEXT_TOKENS", TEXT_TOKENS)
                )
                stack.enter_context(
                    mock.patch.object(runtime, "EXPECTED_TEXT_DIM", TEXT_DIM)
                )
                stack.enter_context(
                    mock.patch.object(
                        operator, "_get_live_parallel_state", return_value=parallel
                    )
                )
                native_runtime = runtime.SAICOnlineMotionNativeRuntimeV1(
                    diffusion,
                    action_handle=handle,
                    action_prompt=action_caption,
                    noop_prompt=noop_caption,
                    action_t2v_embeds=action_t2v,
                    noop_t2v_embeds=noop,
                )
                native_runtime.install()
                try:
                    output = diffusion.sample(
                        **_sample_kwargs(with_references=with_references)
                    )
                finally:
                    native_runtime.restore()
                receipt = native_runtime.finalize()
            return diffusion, handle, output, receipt
        except Exception:
            if native_runtime is not None and native_runtime.installed:
                native_runtime.restore()
            raise
        finally:
            if not handle.restored and operator.active_route() is None:
                handle.restore()

    def test_teacher_precedes_action_and_only_action_has_operator_route(self) -> None:
        diffusion, _handle, output, receipt = self._run(with_references=True)
        self.assertEqual(tuple(output.shape), (1, 16, 21, 2, 2))
        self.assertEqual(receipt["official_negative_calls"], 40)
        self.assertEqual(receipt["official_action_calls"], 40)
        self.assertEqual(receipt["target_only_t2v_action_calls"], 40)
        self.assertEqual(receipt["target_only_t2v_noop_calls"], 40)
        self.assertEqual(receipt["original_scheduler_calls"], 40)
        self.assertTrue(receipt["native_raw_state_captured_before_official_shared_steps"])
        self.assertTrue(receipt["online_motion_field_built_before_native_action_forward"])
        self.assertTrue(receipt["temporal_operator_only_active_for_native_action_forward"])
        self.assertFalse(receipt["negative_noop_and_t2v_teacher_learned_path_executed"])
        self.assertEqual(receipt["sample_contract"]["native_branch"], "VI")
        self.assertEqual(len(diffusion.forward_log), 160)
        for index in range(40):
            rows = diffusion.forward_log[index * 4 : (index + 1) * 4]
            self.assertEqual(
                [row["label"] for row in rows],
                [
                    "native-negative",
                    "teacher-action",
                    "teacher-noop",
                    "native-action",
                ],
            )
            self.assertEqual(
                [row["active_route"] for row in rows],
                [False, False, False, True],
            )
            self.assertEqual(
                [row["learned_path_active"] for row in rows],
                [False, False, False, index < 38],
            )
            self.assertEqual(
                receipt["trace"][index]["native_patch_source_ids"],
                [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0],
            )

    def test_v_branch_and_target_mask_are_derived_without_caller_metadata(self) -> None:
        diffusion, _handle, _output, receipt = self._run(with_references=False)
        self.assertEqual(receipt["sample_contract"]["native_branch"], "V")
        self.assertTrue(receipt["native_branch_and_target_mask_derived_inside_runtime"])
        self.assertFalse(
            receipt["caller_supplied_mask_branch_code_action_id_index_sigma_or_sp"]
        )
        self.assertTrue(
            all(row["native_branch"] == "V" for row in receipt["trace"])
        )
        self.assertTrue(
            all(
                row["native_patch_source_ids"] == [1.0, 0.0]
                for row in receipt["trace"]
            )
        )
        parameters = set(
            inspect.signature(runtime.SAICOnlineMotionNativeRuntimeV1).parameters
        )
        forbidden = {
            "mask",
            "target_mask",
            "branch",
            "phase_code",
            "action_id",
            "schedule_index",
            "sigma",
            "sequence_parallel_rank",
        }
        self.assertFalse(parameters & forbidden)
        self.assertEqual(
            [row["label"] for row in diffusion.forward_log[:4]],
            [
                "native-negative",
                "teacher-action",
                "teacher-noop",
                "native-action",
            ],
        )

    def test_nonzero_operator_changes_result_but_not_teacher_call_graph(self) -> None:
        active, _active_handle, active_output, active_receipt = self._run(
            with_references=False, nonzero_operator=True
        )
        zero, _zero_handle, zero_output, zero_receipt = self._run(
            with_references=False, nonzero_operator=False
        )
        self.assertGreater(float((active_output - zero_output).abs().sum()), 0.0)
        self.assertEqual(
            [row["label"] for row in active.forward_log],
            [row["label"] for row in zero.forward_log],
        )
        self.assertEqual(
            active_receipt["target_only_t2v_action_calls"],
            zero_receipt["target_only_t2v_action_calls"],
        )
        self.assertEqual(
            active_receipt["target_only_t2v_noop_calls"],
            zero_receipt["target_only_t2v_noop_calls"],
        )

    def test_wrong_target_source_id_fails_before_any_shared_forward_and_restores(self) -> None:
        torch.manual_seed(810)
        diffusion = GEN_Wanx22(corrupt_target_source_id=True)
        diffusion.transformer.requires_grad_(False)
        handle = operator.install_saic_temporal_action_operator(diffusion.transformer)
        noop = _prompt(0.5)
        action_t2v = _prompt(3.0)
        diffusion.prompt_names[id(noop)] = "noop"
        diffusion.prompt_names[id(action_t2v)] = "action"
        native_runtime = None
        original_sample = diffusion.sample
        original_shared = diffusion.shared_step
        original_patch = diffusion.transformer.patch_vae_latent
        original_step = diffusion.scheduler.step
        try:
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(runtime, "EXPECTED_HIDDEN_DIM", HIDDEN))
                stack.enter_context(mock.patch.object(runtime, "EXPECTED_TEXT_TOKENS", TEXT_TOKENS))
                stack.enter_context(mock.patch.object(runtime, "EXPECTED_TEXT_DIM", TEXT_DIM))
                stack.enter_context(
                    mock.patch.object(
                        operator,
                        "_get_live_parallel_state",
                        return_value=_ParallelState(),
                    )
                )
                native_runtime = runtime.SAICOnlineMotionNativeRuntimeV1(
                    diffusion,
                    action_handle=handle,
                    action_prompt=(
                        "A brown dog smoothly lowers its body and sits on the floor."
                    ),
                    noop_prompt=(
                        "A brown dog remains standing still in the same fixed scene."
                    ),
                    action_t2v_embeds=action_t2v,
                    noop_t2v_embeds=noop,
                )
                native_runtime.install()
                with self.assertRaisesRegex(
                    runtime.SAICOnlineMotionNativeRuntimeError,
                    "source-id order",
                ):
                    diffusion.sample(**_sample_kwargs(with_references=False))
                native_runtime.restore()
                self.assertEqual(diffusion.forward_log, [])
                self.assertEqual(native_runtime.official_negative_calls, 0)
                self.assertEqual(native_runtime.teacher_action_calls, 0)
                self.assertEqual(native_runtime.teacher_noop_calls, 0)
            self.assertEqual(diffusion.sample, original_sample)
            self.assertEqual(diffusion.shared_step, original_shared)
            self.assertEqual(diffusion.transformer.patch_vae_latent, original_patch)
            self.assertEqual(diffusion.scheduler.step, original_step)
        finally:
            if native_runtime is not None and native_runtime.installed:
                native_runtime.restore()
            if not handle.restored and operator.active_route() is None:
                handle.restore()

    def test_pinned_runtime_audits_and_no_training_authority_are_receipted(self) -> None:
        _diffusion, _handle, _output, receipt = self._run(with_references=True)
        self.assertEqual(
            receipt["scheduler_config_digest"],
            runtime.PINNED_SCHEDULER_CONFIG_DIGEST,
        )
        self.assertEqual(
            receipt["t_qmosaic_audit_source_schema"],
            runtime.TQMOSAIC_AUDIT_SOURCE_SCHEMA,
        )
        self.assertEqual(receipt["exact40_schedule_sha256"], strata.SCHEDULE_SHA256)
        self.assertIn("NativeRV2VActionFieldPatch", receipt["sgaf_seam_reused"])
        self.assertFalse(receipt["operator_checkpoint_loaded_by_this_runtime"])
        self.assertFalse(receipt["optimizer_created"])
        self.assertFalse(receipt["parameters_updated"])
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["semantic_action_success_claim"])

    def test_source_anchor_and_temporal_handles_fail_closed_under_exclusive_trainable_set_contract(
        self,
    ) -> None:
        """Lock the current, intentionally unsupported joint-handle state.

        Both legacy handles authenticate that the transformer's *complete*
        requires-grad ID set equals their own adapter ID set.  This test is a
        fail-closed specification: the native runtime may expose an inner/outer
        forward seam, but must not imply that the two legacy handles can yet be
        installed or certified together without a separate joint-composition
        contract.
        """

        diffusion = GEN_Wanx22()
        diffusion.transformer.requires_grad_(False)
        source_handle = source_anchor.install_saic_source_anchor_adapter(
            diffusion.transformer
        )
        temporal_handle = None
        try:
            self.assertTrue(source_handle.base_parameters_frozen())
            with self.assertRaisesRegex(
                operator.SAICTemporalActionOperatorError,
                "freeze the complete Bernini transformer",
            ):
                operator.install_saic_temporal_action_operator(
                    diffusion.transformer
                )

            # Even an inference-style freeze followed by installation of the
            # second handle cannot satisfy both legacy receipts: the source
            # handle's own parameters no longer have its required trainable
            # gauge, while the temporal handle exclusively owns requires-grad.
            diffusion.transformer.requires_grad_(False)
            temporal_handle = operator.install_saic_temporal_action_operator(
                diffusion.transformer
            )
            self.assertTrue(temporal_handle.base_parameters_frozen())
            with self.assertRaisesRegex(
                source_anchor.SAICSourceAnchorError,
                "source-anchor parameter gauge differs",
            ):
                source_handle.base_parameters_frozen()
        finally:
            if temporal_handle is not None and not temporal_handle.restored:
                temporal_handle.restore()
            if not source_handle.restored:
                source_handle.restore()


if __name__ == "__main__":
    unittest.main()

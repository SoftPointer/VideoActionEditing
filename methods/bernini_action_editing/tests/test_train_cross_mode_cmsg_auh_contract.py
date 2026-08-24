from __future__ import annotations

import ast
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_cross_mode_cmsg_auh as trainer


SHA1 = "a" * 40
SHA256 = "b" * 64


def _args(*extra: str):
    argv = [
        "--bernini-root",
        "/bernini",
        "--veomni-root",
        "/veomni",
        "--checkpoint",
        "/checkpoint",
        "--preprocessed-parquet-dir",
        "/data",
        "--dataset-summary",
        "/summary.json",
        "--routing-jsonl",
        "/route.jsonl",
        "--output",
        "/output",
        "--method-source-revision",
        SHA1,
        "--method-source-archive-sha256",
        SHA256,
        *extra,
    ]
    return trainer.build_parser().parse_args(argv)


class _Dataset:
    signature = "dataset-signature"
    root = Path("/data")

    def __len__(self):
        return 644


class _Router:
    digest = "9" * 64
    file_sha256 = trainer.v5.STRICT_ROUTING_SHA256

    def receipt(self):
        return {
            "path": "/route.jsonl",
            "default_tier": "reject",
            "file_sha256": self.file_sha256,
            "routing_digest": self.digest,
            "explicit_route_counts": {
                "full_pair": 0,
                "motion_only": 359,
                "reject": 285,
            },
        }


class _Route:
    def __init__(self, index: int):
        self.iid = f"iid-{index}"
        self.tier = "motion_only"
        self.full_target_weight = 0.0


class CrossModeAUHPureContractTests(unittest.TestCase):
    def test_identity_and_literal_six_forward_order_are_pinned(self) -> None:
        self.assertEqual(trainer.NUM_FRAMES, 81)
        self.assertEqual(trainer.LATENT_PHASES, 21)
        self.assertEqual(trainer.TRAINING_BRIDGE_ENDPOINT, "source(beta=0)")
        self.assertEqual(trainer.T2V_GUIDANCE_SCALE, 4.0)
        self.assertEqual(
            trainer.T2V_SYSTEM_PROMPT,
            "You are a helpful assistant specialized in text-to-video generation.",
        )
        self.assertEqual(
            trainer.FORWARD_CELL_ORDER,
            (
                "frozen_editor_negative_full_source",
                "frozen_editor_noop_full_source",
                "frozen_editor_action_full_source",
                "adapted_editor_action_full_source",
                "frozen_generator_negative_target_only",
                "frozen_generator_action_target_only",
            ),
        )

    def test_torch_is_lazy_and_main_has_no_resume_surface(self) -> None:
        tree = ast.parse(Path(trainer.__file__).read_text(encoding="utf-8"))
        eager = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager.extend(alias.name for alias in node.names if alias.name == "torch")
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager.append(node.module)
        self.assertEqual(eager, [])
        options = {
            option
            for action in trainer.build_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--resume", options)
        self.assertNotIn("--init-adapter-checkpoint", options)

    def test_gate_disabled_is_only_a_one_step_canary(self) -> None:
        trainer.validate_cli(_args())
        canary = _args("--max-steps", "1", "--disable-frozen-prior-gate")
        trainer.validate_cli(canary)
        formal = _args("--max-steps", "40", "--disable-frozen-prior-gate")
        with self.assertRaisesRegex(
            trainer.CMSGauhTrainingError, "one-step canary"
        ):
            trainer.validate_cli(formal)

    def test_editor_moves_before_generator_target_views_are_built(self) -> None:
        source = inspect.getsource(trainer._move_candidate_to_device)
        move_position = source.index("editor_action = legacy._move_batch")
        build_position = source.index("branches = core.build_training_branches")
        self.assertLess(move_position, build_position)
        self.assertIn("action_text", source)
        self.assertIn("negative_text", source)

    def test_immutable_contract_binds_deployment_and_rope_parity(self) -> None:
        args = _args()
        targets = trainer.core.select_cmsg_lora_targets(
            trainer.core.canonical_attention_modules()
        )
        routes = [(index, _Route(index)) for index in range(3)]
        contract = trainer._immutable_contract(
            args=args,
            dataset=_Dataset(),
            dataset_summary={"sha256": "c" * 64, "index_sha256": "d" * 64},
            router=_Router(),
            eligible_routes=routes,
            target_modules=targets,
            checkpoint=Path("/checkpoint"),
        )
        value = contract["value"]
        self.assertEqual(contract["digest"], trainer.legacy.object_sha256(value))
        self.assertEqual(value["lora"]["target_module_count"], 46)
        self.assertEqual(value["lora"]["rank"], 8)
        self.assertEqual(value["lora"]["alpha"], 8)
        self.assertEqual(value["training_bridge_endpoint"], "source(beta=0)")
        self.assertTrue(value["target_endpoint_teacher_leakage_forbidden"])
        self.assertEqual(value["forwards_per_candidate"], 6)
        self.assertEqual(value["graph_forwards_per_candidate"], 1)
        self.assertEqual(value["inference_generator_forwards"], 0)
        self.assertEqual(value["sigma_selector"], "absolute_global_step_mod_40")
        self.assertEqual(value["formal_adapter_off_steps"], list(range(32, 40)))
        self.assertIn(31, value["zero_release_steps"])
        self.assertFalse(value["resume_integrated"])
        parity = value["t2v_rope_parity"]
        self.assertEqual(parity["native_t2v_target_source_id"], 0)
        self.assertEqual(parity["mv2v_target_source_id"], 0)
        self.assertTrue(parity["per_candidate_exact_tensor_equality_required"])

    def test_supervision_and_optimizer_state_are_auditable(self) -> None:
        early = trainer._supervision_receipt(global_step=1)
        formal = trainer._supervision_receipt(global_step=40)
        self.assertFalse(early["optimizer_updates_completed"])
        self.assertTrue(formal["optimizer_updates_completed"])
        self.assertTrue(formal["frozen_target_only_generator_teacher"])
        self.assertTrue(formal["generator_teacher_training_only"])
        self.assertFalse(formal["generator_loaded_at_inference"])
        self.assertFalse(formal["paired_target_used_at_inference"])
        self.assertEqual(
            formal["inference_conditions"], ["source_video", "action_instruction"]
        )
        self.assertFalse(formal["resume_integrated"])

        optimizer = SimpleNamespace(state_dict=lambda: {"state": {}})
        audit = [{"attempt_ordinal": 3, "accepted": False}]
        payload = trainer._optimizer_payload(
            optimizer=optimizer,
            global_step=2,
            attempt_ordinal=4,
            rejected_count=2,
            immutable={"digest": SHA256},
            parameter_names=["adapter.a"],
            gate_audit=audit,
        )
        self.assertEqual(payload["accepted_count"], 2)
        self.assertEqual(payload["attempt_ordinal"], 4)
        self.assertEqual(payload["rejected_count"], 2)
        self.assertEqual(payload["gate_audit"], audit)
        self.assertFalse(payload["resume_integrated"])

    def test_step_index_comes_from_sigma_schedule_and_seed_from_attempt(self) -> None:
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        self.assertIn("step_index=selected_stratum.schedule_index", source)
        self.assertIn(
            "legacy.step_seed(args.seed, current_ordinal, row_index)", source
        )
        self.assertIn("selected_stratum = sigma_strata.select_sigma_stratum(global_step)", source)

    def test_real_inference_validator_accepts_a_synthesized_trainer_receipt(self) -> None:
        import infer_cross_mode_cmsg_lora as inference

        args = _args("--max-steps", "40")
        targets = trainer.core.select_cmsg_lora_targets(
            trainer.core.canonical_attention_modules()
        )
        routes = [(index, _Route(index)) for index in range(359)]
        summary = {"sha256": "c" * 64, "index_sha256": "d" * 64}
        immutable = trainer._immutable_contract(
            args=args,
            dataset=_Dataset(),
            dataset_summary=summary,
            router=_Router(),
            eligible_routes=routes,
            target_modules=targets,
            checkpoint=Path("/checkpoint"),
        )
        gate_audit = [
            {"attempt_ordinal": index, "accepted": True} for index in range(40)
        ]
        parameter = SimpleNamespace(numel=lambda: 16)
        with mock.patch.object(
            trainer.v4, "_checkpoint_parameter_digest", return_value="e" * 64
        ), mock.patch.object(
            trainer.v4, "_stable_recursive_digest", return_value="f" * 64
        ):
            receipt = trainer._build_receipt(
                args=args,
                global_step=40,
                attempt_ordinal=40,
                rejected_count=0,
                metrics={"loss": 1.0},
                gate_audit=gate_audit,
                dataset=_Dataset(),
                dataset_summary=summary,
                router=_Router(),
                checkpoint=Path("/checkpoint"),
                bernini_revision=trainer.legacy.BERNINI_OFFICIAL_COMMIT,
                veomni_revision=trainer.legacy.VEOMNI_TESTED_COMMIT,
                distributed=SimpleNamespace(world_size=4, ulysses_size=4),
                backend="nccl/rccl",
                target_modules=targets,
                named_trainable=[("adapter.weight", parameter)],
                initialization_digest="7" * 64,
                transformers_version="test",
                immutable=immutable,
                optimizer_payload={"optimizer": {}, "global_step": 40},
            )
        adapter_config = {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "bias": "none",
            "modules_to_save": None,
            "use_dora": False,
            "use_rslora": False,
            "target_modules": targets,
        }
        identity = inference.validate_training_adapter_contract(
            adapter_config, receipt
        )
        self.assertEqual(identity["global_step"], 40)
        self.assertEqual(identity["targets"], targets)
        self.assertEqual(identity["receipt_digest"], receipt["receipt_digest"])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class CrossModeAUHTensorContractTests(unittest.TestCase):
    def test_checkpoint_receipt_contains_loader_and_sigma_parity(self) -> None:
        args = _args("--max-steps", "40")
        targets = trainer.core.select_cmsg_lora_targets(
            trainer.core.canonical_attention_modules()
        )
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        optimizer_payload = {"optimizer": {}, "global_step": 40}
        receipt = trainer._build_receipt(
            args=args,
            global_step=40,
            attempt_ordinal=43,
            rejected_count=3,
            metrics={"loss": 1.0},
            gate_audit=[{"attempt_ordinal": 42, "accepted": True}],
            dataset=_Dataset(),
            dataset_summary={"sha256": "c" * 64, "index_sha256": "d" * 64},
            router=_Router(),
            checkpoint=Path("/checkpoint"),
            bernini_revision=trainer.legacy.BERNINI_OFFICIAL_COMMIT,
            veomni_revision=trainer.legacy.VEOMNI_TESTED_COMMIT,
            distributed=SimpleNamespace(world_size=4, ulysses_size=4),
            backend="nccl/rccl",
            target_modules=targets,
            named_trainable=[("adapter.weight", parameter)],
            initialization_digest="e" * 64,
            transformers_version="test",
            immutable={"value": {"method": trainer.METHOD_NAME}, "digest": "f" * 64},
            optimizer_payload=optimizer_payload,
        )
        self.assertEqual(receipt["attempt_ordinal"], 43)
        self.assertEqual(receipt["rejected_count"], 3)
        self.assertTrue(receipt["supervision"]["optimizer_updates_completed"])
        self.assertTrue(
            receipt["supervision"][
                "checkpoint_optimizer_inference_receipt_parity"
            ]
        )
        self.assertEqual(
            receipt["inference_sigma_strata"],
            trainer.sigma_strata.build_sigma_strata_receipt(
                completed_optimizer_steps=40
            ),
        )
        self.assertFalse(receipt["resume_integrated"])
        self.assertFalse(receipt["inference_loader_parity_pending"])
        candidate = dict(receipt)
        digest = candidate.pop("receipt_digest")
        self.assertEqual(digest, trainer.legacy.object_sha256(candidate))

    def test_official_t2v_tokenization_uses_cleaned_raw_instruction(self) -> None:
        calls = []

        def tokenizer(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return SimpleNamespace(
                input_ids=torch.tensor([[4, 5, 6]], dtype=torch.long),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
            )

        cleaner_calls = []

        def cleaner(text):
            cleaner_calls.append(text)
            return "cleaned action"

        sample = {
            "inputs": json.dumps(
                [
                    {"type": "video", "has_loss": 0},
                    {"type": "text", "text": "the dog jumps"},
                    {"type": "video_gen", "has_loss": 1},
                ]
            )
        }
        fields, instruction, prompt_sha = trainer._official_t2v_text_fields(
            sample,
            tokenizer=tokenizer,
            prompt_cleaner=cleaner,
            system_prompts={"t2v": trainer.T2V_SYSTEM_PROMPT},
        )
        self.assertEqual(instruction, "the dog jumps")
        self.assertEqual(cleaner_calls, ["the dog jumps"])
        self.assertEqual(calls[0][0], trainer.T2V_SYSTEM_PROMPT + "cleaned action")
        self.assertEqual(
            calls[0][1],
            {
                "add_special_tokens": True,
                "return_attention_mask": True,
                "return_tensors": "pt",
            },
        )
        self.assertEqual(int(fields["t5_input_lens"].item()), 3)
        self.assertEqual(len(prompt_sha), 64)

    def test_text_replacement_updates_packing_geometry_atomically(self) -> None:
        batch = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "t5_input_lens": torch.tensor([[2]], dtype=torch.long),
            "vae_seqlen": torch.tensor([[20]], dtype=torch.long),
            "vlm_seqlen": torch.tensor([[2]], dtype=torch.long),
            "num_tokens": torch.tensor([[22]], dtype=torch.long),
        }
        text = {
            "input_ids": torch.tensor([[7, 8, 9, 10]], dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "t5_input_lens": torch.tensor([[4]], dtype=torch.long),
        }
        result = trainer._bind_text_geometry(batch, text, label="test")
        self.assertEqual(int(result["vae_seqlen"].item()), 20)
        self.assertEqual(int(result["vlm_seqlen"].item()), 4)
        self.assertEqual(int(result["num_tokens"].item()), 24)
        self.assertEqual(int(batch["num_tokens"].item()), 22)

    def test_native_t2v_rope_is_recomputed_at_source_id_zero(self) -> None:
        height, width = 2, 3
        tokens = trainer.LATENT_PHASES * height * width

        class FakeRope:
            use_src_id_rotary_emb = True
            attention_head_dim = 128
            patch_size = (1, 2, 2)
            max_seq_len = 1024

            def __init__(self):
                self.calls = []

            def __call__(self, probe, *, source_id):
                self.calls.append((tuple(probe.shape), source_id))
                values = torch.arange(tokens * 4, dtype=torch.float32)
                return values.reshape(1, 1, tokens, 4)

        rope = FakeRope()
        native = rope(
            torch.empty(1, 16, 21, 2 * height, 2 * width), source_id=0
        ).squeeze(0).permute(1, 0, 2)
        rope.calls.clear()
        editor = {
            "vae_latents_mask": torch.tensor(
                [[False] * tokens + [True] * tokens], dtype=torch.bool
            ),
            "input_vae_rope": torch.cat((torch.zeros_like(native), native), dim=0),
        }
        parity = trainer._validate_native_t2v_rope_parity(
            editor, rope=rope, z_dim=16, spatial_hw=(height, width)
        )
        self.assertEqual(rope.calls, [((1, 16, 21, 4, 6), 0)])
        self.assertTrue(parity["verified"])
        self.assertEqual(parity["target_tokens"], tokens)

        bad = dict(editor)
        bad["input_vae_rope"] = editor["input_vae_rope"].clone()
        bad["input_vae_rope"][-1, 0, 0] += 1
        with self.assertRaisesRegex(
            trainer.CMSGauhTrainingError, "differs from native T2V"
        ):
            trainer._validate_native_t2v_rope_parity(
                bad, rope=rope, z_dim=16, spatial_hw=(height, width)
            )

    def test_plain_cfg_combines_native_bf16_velocities_before_clean(self) -> None:
        shared = torch.zeros(1, 21, 2, dtype=torch.float32)
        sigma = torch.tensor(0.5, dtype=torch.float32)
        negative = torch.full((1, 21, 2), 0.5, dtype=torch.bfloat16)
        action = torch.full((1, 21, 2), 1.25, dtype=torch.bfloat16)
        seen = []

        def reconstruct(noisy, velocity, runtime_sigma):
            self.assertIs(noisy, shared)
            self.assertIs(runtime_sigma, sigma)
            seen.append(velocity)
            return velocity.float()

        with mock.patch.object(
            trainer.v5.tri, "pinned_raw_condition_clean", side_effect=reconstruct
        ), mock.patch.object(trainer.v5, "_as_phase_grid", side_effect=lambda x: x):
            unconditional, guided = trainer._generator_plain_cfg_clean(
                shared_noisy=shared,
                sigma=sigma,
                negative_velocity=negative,
                action_velocity=action,
            )
        expected = negative + 4.0 * (action - negative)
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0], negative)
        self.assertEqual(seen[1].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(seen[1], expected))
        self.assertTrue(torch.equal(unconditional, negative.float()))
        self.assertTrue(torch.equal(guided, expected.float()))

    def test_six_forward_runtime_order_and_only_adapted_action_has_graph(self) -> None:
        labels = []
        editor_action_count = 0

        def velocity(_renderer, batch):
            nonlocal editor_action_count
            label = batch["label"]
            labels.append(label)
            value = torch.full((1, 21, 2), len(labels), dtype=torch.bfloat16)
            if label == "editor_action":
                editor_action_count += 1
                if editor_action_count == 2:
                    value.requires_grad_(True)
            return value

        @contextmanager
        def disabled():
            yield

        controller = SimpleNamespace(disable_adapter=disabled)
        candidate = trainer.MovedCandidate(
            editor_negative={"label": "editor_negative"},
            editor_noop={"label": "editor_noop"},
            editor_action={"label": "editor_action"},
            generator_action={"label": "generator_action"},
            generator_negative={"label": "generator_negative"},
            generator_action_text_fields={},
            generator_negative_text_fields={},
            auxiliary={
                "shared_noisy": torch.zeros(1, 21, 2, dtype=torch.float32),
                "sigma": torch.tensor(0.5, dtype=torch.float32),
                "source_clean": torch.zeros(1, 21, 2, dtype=torch.float32),
                "target_clean": torch.ones(1, 21, 2, dtype=torch.float32),
            },
            spatial_hw=(1, 1),
            instruction_sha256=SHA256,
            t2v_rope_parity={"verified": True},
        )

        def phase_grid(value):
            return value.float().reshape(1, 21, 1, 2)

        def guided_clean(**kwargs):
            return phase_grid(kwargs["negative_velocity"]), phase_grid(
                kwargs["conditional_velocity"]
            )

        generator_uncond = torch.zeros(1, 21, 1, 2, dtype=torch.float32)
        generator_action = torch.ones(1, 21, 1, 2, dtype=torch.float32)
        gate = SimpleNamespace(passed=torch.tensor([True]))
        captured = {}

        def loss(**kwargs):
            captured.update(kwargs)
            total = kwargs["adapted_editor_action_field"].mean()
            zero = total * 0.0
            return SimpleNamespace(
                total=total,
                editor_direction=zero,
                log_amplitude=zero,
                generator_spectral_consistency=zero,
                high_frequency_detail=zero,
                late_frozen_replay=zero,
                rho=trainer.spectrum.release_rho(kwargs["step_index"]),
            )

        with mock.patch.object(
            trainer.motion, "renderer_velocity_prediction", side_effect=velocity
        ), mock.patch.object(trainer.v5, "_guided_clean", side_effect=guided_clean), mock.patch.object(
            trainer.v5, "_as_phase_grid", side_effect=phase_grid
        ), mock.patch.object(
            trainer, "_generator_plain_cfg_clean", return_value=(generator_uncond, generator_action)
        ), mock.patch.object(
            trainer.core, "compute_frozen_prior_gate", return_value=gate
        ), mock.patch.object(
            trainer.core, "compute_cmsg_lora_loss", side_effect=loss
        ), mock.patch.object(
            trainer.motion,
            "clean_field_inverse_sigma_weight",
            return_value=torch.tensor(2.0),
        ):
            result = trainer._run_six_forward_cell(
                renderer=object(),
                adapter_controller=controller,
                candidate=candidate,
                step_index=17,
                enforce_frozen_prior_gate=True,
            )

        self.assertEqual(
            labels,
            [
                "editor_negative",
                "editor_noop",
                "editor_action",
                "editor_action",
                "generator_negative",
                "generator_action",
            ],
        )
        self.assertEqual(captured["step_index"], 17)
        self.assertTrue(captured["adapted_editor_action_field"].requires_grad)
        for name in (
            "frozen_editor_action_field",
            "editor_noop_field",
            "frozen_generator_action_field",
            "generator_uncond_field",
            "target_motion_field",
        ):
            self.assertFalse(captured[name].requires_grad, name)
        self.assertTrue(result.weighted_loss.requires_grad)
        self.assertEqual(float(result.inverse_sigma_weight.item()), 2.0)


if __name__ == "__main__":
    unittest.main()

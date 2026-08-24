from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_prior_tangent_lora as trainer


SHA1 = "1" * 40
SHA256 = "2" * 64


def _args(**overrides):
    values = {
        "num_frames": 81,
        "max_steps": 40,
        "save_every": 40,
        "resume": None,
        "init_adapter_checkpoint": None,
        "allow_incomplete_dataset": False,
        "routing_jsonl": "/strict-359.jsonl",
        "expected_routing_jsonl_sha256": trainer.STRICT_ROUTING_SHA256,
        "unreviewed_tier": "reject",
        "learning_rate": trainer.LEARNING_RATE,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 20260806,
        "lora_scope": "cross_q",
        "branch_state_mode": "source_target_bridge_clean_field",
        "minimum_training_sigma": 0.1,
        "inverse_sigma_weight_floor": (
            trainer.sigma_strata.PINNED_POSITIVE_SIGMAS[-1]
        ),
        "motion_loss_weight": trainer.FIELD_LOSS_WEIGHT,
        "copy_loss_weight": 0.0,
        "boundary_gauge_loss_weight": 0.0,
        "anchor_loss_weight": 0.0,
        "motion_objective": "causal_boundary_charbonnier",
        "bridge_consistency_weight": trainer.BRIDGE_LOSS_WEIGHT,
        "late_replay_loss_weight": trainer.LATE_REPLAY_LOSS_WEIGHT,
        "causal_ema_decay": 0.5,
        "charbonnier_scale": trainer.CHARBONNIER_SCALE,
        "quotient_weight": 0.5,
        "high_noise_floor": 1.0,
        "high_noise_power": 2.0,
        "temporal_lags": [1, 2, 4],
        "noop_instruction": trainer.motion.DEFAULT_NOOP_INSTRUCTION,
        "negative_prompt": trainer.DEFAULT_NEGATIVE_PROMPT,
        "expected_bernini_commit": trainer.legacy.BERNINI_OFFICIAL_COMMIT,
        "expected_veomni_commit": trainer.legacy.VEOMNI_TESTED_COMMIT,
        "method_source_revision": SHA1,
        "expected_checkpoint_tree_sha256": trainer.legacy.CHECKPOINT_TREE_SHA256,
        "method_source_archive_sha256": SHA256,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class CliContractTests(unittest.TestCase):
    def test_defaults_are_exact_v5_main_arm(self) -> None:
        args = trainer.build_parser().parse_args(
            [
                "--bernini-root",
                "/b",
                "--veomni-root",
                "/v",
                "--checkpoint",
                "/c",
                "--preprocessed-parquet-dir",
                "/d",
                "--dataset-summary",
                "/s",
                "--routing-jsonl",
                "/r",
                "--expected-routing-jsonl-sha256",
                trainer.STRICT_ROUTING_SHA256,
                "--output",
                "/o",
                "--method-source-revision",
                SHA1,
                "--method-source-archive-sha256",
                SHA256,
            ]
        )
        self.assertEqual(args.num_frames, 81)
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.lora_scope, "cross_q")
        self.assertEqual(args.branch_state_mode, "source_target_bridge_clean_field")
        self.assertEqual(args.motion_loss_weight, 1.0)
        self.assertEqual(args.bridge_consistency_weight, 0.05)
        self.assertEqual(args.late_replay_loss_weight, 0.10)
        self.assertEqual(args.copy_loss_weight, 0.0)
        self.assertEqual(args.unreviewed_tier, "reject")
        self.assertEqual(args.negative_prompt, trainer.DEFAULT_NEGATIVE_PROMPT)
        trainer.validate_cli(args)

    def test_main_arm_hyperparameters_fail_closed(self) -> None:
        invalid = (
            {"num_frames": 41},
            {"learning_rate": 3e-5},
            {"lora_scope": "cross_q_out"},
            {"branch_state_mode": "shared_noisy_clean_field"},
            {"motion_loss_weight": 0.5},
            {"bridge_consistency_weight": 0.1},
            {"late_replay_loss_weight": 0.0},
            {"copy_loss_weight": 0.1},
            {"boundary_gauge_loss_weight": 0.1},
            {"anchor_loss_weight": 0.1},
            {"charbonnier_scale": 0.2},
            {"high_noise_floor": 0.5},
            {"minimum_training_sigma": 0.2},
            {"inverse_sigma_weight_floor": 0.2},
            {"routing_jsonl": None},
            {"expected_routing_jsonl_sha256": None},
            {"expected_routing_jsonl_sha256": SHA256},
            {"unreviewed_tier": "motion_only"},
            {"negative_prompt": "custom"},
            {"noop_instruction": "custom no-op"},
            {"allow_incomplete_dataset": True},
            {"quotient_weight": 0.0},
            {"causal_ema_decay": 0.2},
            {"high_noise_power": 1.0},
            {"temporal_lags": [1]},
            {"expected_bernini_commit": SHA1},
            {"expected_veomni_commit": SHA1},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(trainer.PriorTangentTrainingError):
                    trainer.validate_cli(_args(**overrides))

    def test_cli_has_no_external_spatial_or_motion_hint(self) -> None:
        destinations = {action.dest for action in trainer.build_parser()._actions}
        for forbidden in (
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "swept_tube",
            "first_frame_anchor",
        ):
            self.assertNotIn(forbidden, destinations)


class PureTrainingContractTests(unittest.TestCase):
    def test_auh_launcher_uses_cluster_compatible_private_scratch(self) -> None:
        launcher = (
            METHOD_ROOT / "scripts" / "auh_train_prior_tangent_lora.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn('scratch_parent="${SLURM_TMPDIR:-/tmp}"', launcher)
        self.assertIn('task_scratch="$(mktemp -d ', launcher)
        self.assertNotIn("SLURM_TMPDIR must be an absolute Slurm-owned", launcher)

    def test_method_schema_and_numeric_constants(self) -> None:
        self.assertEqual(
            trainer.RECEIPT_SCHEMA,
            "bernini-r-1p3b-prior-tangent-lora-receipt-v5",
        )
        self.assertEqual(trainer.METHOD_NAME, trainer.pgt.METHOD_NAME)
        self.assertEqual(trainer.FIELD_LOSS_WEIGHT, 1.0)
        self.assertEqual(trainer.BRIDGE_LOSS_WEIGHT, 0.05)
        self.assertEqual(trainer.LATE_REPLAY_LOSS_WEIGHT, 0.10)
        self.assertEqual(trainer.APG_GUIDANCE_SCALE, 4.0)
        self.assertEqual(trainer.APG_ETA, 0.5)
        self.assertEqual(trainer.APG_NORM_THRESHOLD, 50.0)
        self.assertEqual(trainer.APG_MOMENTUM, 0.0)

    def test_four_forward_code_path_is_explicit_and_does_not_call_v4_main(self) -> None:
        source = inspect.getsource(trainer._endpoint_fields)
        self.assertIn("adapter_controller.disable_adapter()", source)
        self.assertIn("with torch.no_grad()", source)
        self.assertIn("base_negative_velocity", source)
        self.assertIn("base_noop_velocity", source)
        self.assertIn("base_action_velocity", source)
        self.assertIn("adapted_action_velocity", source)
        self.assertEqual(source.count("motion.renderer_velocity_prediction("), 4)
        self.assertNotIn("v4.main", inspect.getsource(trainer.main))

    def test_negative_branch_bypasses_positive_renderer_message_processing(self) -> None:
        source = inspect.getsource(trainer._prepare_prior_bridge_batches)
        tree = ast.parse(source)
        direct_positive_processor_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "process_renderer_sample"
        ]
        self.assertEqual(direct_positive_processor_calls, [])
        self.assertNotIn("replace_edit_instruction", source)
        self.assertIn("_official_negative_text_fields", source)

    def test_main_checks_runtime_bernini_negative_prompt(self) -> None:
        source = inspect.getsource(trainer.main)
        self.assertIn("from bernini.cli import DEFAULT_NEG_PROMPT", source)
        self.assertIn(
            "if DEFAULT_NEG_PROMPT != DEFAULT_NEGATIVE_PROMPT:", source
        )
        self.assertLess(
            source.index("from bernini.cli import DEFAULT_NEG_PROMPT"),
            source.index("BerniniRendererConfig.from_pretrained"),
        )

    def test_temporal_axis_is_materialized_before_shared_operator(self) -> None:
        endpoint = inspect.getsource(trainer._endpoint_fields)
        self.assertIn('_as_phase_grid(auxiliary["source_clean"].float())', endpoint)
        self.assertIn('_as_phase_grid(auxiliary["target_clean"].float())', endpoint)
        self.assertIn("pgt.student_executed_field(", endpoint)
        self.assertIn("pgt.teacher_executed_field(", endpoint)
        tree = ast.parse(Path(trainer.__file__).read_text(encoding="utf-8"))
        eager_torch = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Import)
                and any(alias.name == "torch" for alias in node.names)
            )
            or (isinstance(node, ast.ImportFrom) and node.module == "torch")
        ]
        self.assertEqual(eager_torch, [])

    def test_supervision_receipt_is_inference_closed(self) -> None:
        receipt = trainer._supervision_receipt(_args())
        self.assertTrue(receipt["four_branch_endpoint"])
        self.assertTrue(receipt["base_branches_adapter_disabled"])
        self.assertTrue(receipt["base_branches_no_grad"])
        self.assertTrue(receipt["adapted_action_only_trainable_forward"])
        self.assertEqual(
            receipt["causal_frozen_prior"], "Q0(base_action-base_noop)"
        )
        self.assertEqual(receipt["executed_motion_phase_zero"], "exact_zero")
        self.assertEqual(receipt["official_apg_momentum"], 0.0)
        self.assertEqual(receipt["field_loss_weight"], 1.0)
        self.assertEqual(receipt["bridge_loss_weight"], 0.05)
        self.assertEqual(receipt["late_replay_loss_weight"], 0.10)
        self.assertFalse(receipt["target_used_as_model_condition"])
        self.assertFalse(receipt["external_mask_track_flow_pose_trajectory"])

    def test_immutable_contract_binds_operator_schedule_and_conditions(self) -> None:
        class Dataset:
            signature = "dataset-signature"

        class Router:
            digest = "routing-digest"
            file_sha256 = SHA256

        routes = [
            (
                index,
                trainer.motion.Route(
                    iid=f"iid-{index}",
                    tier="motion_only",
                    full_target_weight=0.0,
                ),
            )
            for index in range(3)
        ]
        targets = [
            f"diff_dec.transformer.blocks.{index}.attn2.to_q"
            for index in range(30)
        ]
        contract = trainer._immutable_contract(
            args=_args(),
            dataset=Dataset(),
            dataset_summary={"sha256": SHA256, "index_sha256": "3" * 64},
            router=Router(),
            eligible_routes=routes,
            target_modules=targets,
            checkpoint=Path("/checkpoint"),
        )
        value = contract["value"]
        self.assertEqual(value["schema_version"], trainer.RECEIPT_SCHEMA)
        self.assertEqual(value["lora_scope"], "cross_q")
        self.assertEqual(value["lora_rank"], 8)
        self.assertEqual(value["target_modules"], targets)
        self.assertEqual(value["forwards_per_endpoint"], 4)
        self.assertEqual(value["forwards_per_optimizer_step"], 8)
        self.assertEqual(value["base_apg"]["momentum"], 0.0)
        self.assertEqual(
            value["raw_prior"],
            "raw_frozen_prior=base_guided_action-base_guided_noop",
        )
        self.assertEqual(
            value["prior"],
            "causal_frozen_prior=Q0(raw_frozen_prior)",
        )
        self.assertEqual(
            value["teacher_correction"],
            "Q0((target-source)-causal_frozen_prior)",
        )
        self.assertEqual(
            value["phase_zero_contract"],
            "executed_motion_exact_zero_source_exactly_preserved",
        )
        self.assertEqual(
            value["noop_instruction_sha256"],
            trainer.hashlib.sha256(
                trainer.motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(value["latent_phases"], 21)
        self.assertEqual(len(value["gamma_schedule"]), 40)
        self.assertEqual(value["gamma_schedule"][24], 1.0)
        self.assertEqual(value["gamma_schedule"][34:], [0.0] * 6)
        self.assertIn("causal frozen prior", value["gamma_contract"])
        self.assertEqual(value["loss"]["bridge"], 0.05)
        self.assertEqual(value["loss"]["late_replay"], 0.10)
        self.assertEqual(
            value["inference_conditions"],
            ["source_video", "action_instruction"],
        )
        self.assertIn("mask", value["forbidden_inference_conditions"])
        self.assertEqual(contract["digest"], trainer.legacy.object_sha256(value))

    def test_cross_q_contract_rejects_any_module_count_other_than_30(self) -> None:
        class Dataset:
            signature = "dataset-signature"

        class Router:
            digest = "routing-digest"
            file_sha256 = SHA256

        with self.assertRaisesRegex(
            trainer.PriorTangentTrainingError, "exactly 30"
        ):
            trainer._immutable_contract(
                args=_args(),
                dataset=Dataset(),
                dataset_summary={"sha256": SHA256, "index_sha256": SHA256},
                router=Router(),
                eligible_routes=[],
                target_modules=["one"],
                checkpoint=Path("/checkpoint"),
            )

    def test_resume_rejects_v4_or_tampered_receipts(self) -> None:
        immutable = {"value": {"method": trainer.METHOD_NAME}, "digest": SHA256}
        valid = {
            "schema_version": trainer.RECEIPT_SCHEMA,
            "method": trainer.METHOD_NAME,
            "global_step": 40,
            "immutable_contract": immutable,
        }
        valid["receipt_digest"] = trainer.legacy.object_sha256(valid)
        self.assertEqual(
            trainer._validate_resume_receipt(valid, immutable=immutable), 40
        )
        wrong = dict(valid, schema_version=trainer.v4.RECEIPT_SCHEMA)
        with self.assertRaises(trainer.PriorTangentTrainingError):
            trainer._validate_resume_receipt(wrong, immutable=immutable)
        tampered = dict(valid, global_step=41)
        with self.assertRaises(trainer.PriorTangentTrainingError):
            trainer._validate_resume_receipt(tampered, immutable=immutable)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorTrainingContractTests(unittest.TestCase):
    def test_packed_phase_apg_is_exact_official_spatial_permutation(self) -> None:
        batch, phases, patch_h, patch_w, channels = 1, 4, 2, 3, 3
        patch_elements = 2 * 2 * channels

        def grid_to_spatial(grid):
            return (
                grid.reshape(
                    batch,
                    phases,
                    patch_h,
                    patch_w,
                    2,
                    2,
                    channels,
                )
                .permute(0, 6, 1, 2, 4, 3, 5)
                .reshape(batch, channels, phases, patch_h * 2, patch_w * 2)
            )

        def spatial_to_grid(spatial):
            return (
                spatial.reshape(
                    batch,
                    channels,
                    phases,
                    patch_h,
                    2,
                    patch_w,
                    2,
                )
                .permute(0, 2, 3, 5, 4, 6, 1)
                .reshape(batch, phases, patch_h * patch_w, patch_elements)
            )

        count = batch * phases * patch_h * patch_w * patch_elements
        negative = (
            torch.arange(count, dtype=torch.float32).reshape(
                batch, phases, patch_h * patch_w, patch_elements
            )
            / 16.0
        )
        conditional = negative + (
            torch.arange(count - 1, -1, -1, dtype=torch.float32).reshape_as(
                negative
            )
            / 32.0
        )
        packed_result = trainer._official_momentum_zero_apg(
            conditional, negative
        )
        official_spatial = trainer.tri._normalized_guidance(
            grid_to_spatial(conditional),
            grid_to_spatial(negative),
            trainer.APG_GUIDANCE_SCALE,
            trainer.tri._MomentumBuffer(0.0, branch="parity"),
            trainer.APG_ETA,
            trainer.APG_NORM_THRESHOLD,
        )
        official_grid = spatial_to_grid(official_spatial)
        self.assertTrue(torch.equal(packed_result, official_grid))

    def test_official_negative_is_verbatim_variable_length_tokenization(self) -> None:
        calls = []

        class Encoded:
            input_ids = torch.arange(7, dtype=torch.long).reshape(1, 7)
            attention_mask = torch.ones(1, 7, dtype=torch.long)

        class Tokenizer:
            def __call__(self, text, **kwargs):
                calls.append((text, kwargs))
                return Encoded()

        fields = trainer._official_negative_text_fields(
            Tokenizer(), trainer.DEFAULT_NEGATIVE_PROMPT
        )
        self.assertEqual(len(calls), 1)
        text, kwargs = calls[0]
        self.assertEqual(text, trainer.DEFAULT_NEGATIVE_PROMPT)
        self.assertNotIn("You are a helpful assistant for editing", text)
        self.assertEqual(
            kwargs,
            {
                "max_length": 512,
                "truncation": True,
                "add_special_tokens": True,
                "return_attention_mask": True,
                "return_tensors": "pt",
            },
        )
        self.assertEqual(tuple(fields["input_ids"].shape), (1, 7))
        self.assertEqual(tuple(fields["attention_mask"].shape), (1, 7))
        self.assertEqual(tuple(fields["t5_input_lens"].shape), (1, 1))
        self.assertEqual(int(fields["t5_input_lens"].item()), 7)
        self.assertEqual(int(fields["attention_mask"].sum()), 7)

    def test_phase_pack_round_trip_and_apg_zero_delta(self) -> None:
        packed = torch.randn(1, 21 * 5, 8, dtype=torch.float32)
        grid = trainer._as_phase_grid(packed)
        self.assertEqual(tuple(grid.shape), (1, 21, 5, 8))
        self.assertTrue(torch.equal(trainer._from_phase_grid(grid), packed))
        guided = trainer._official_momentum_zero_apg(grid, grid)
        self.assertTrue(torch.equal(guided, grid))

    def test_four_branch_forward_contexts_are_enforced(self) -> None:
        class Adapter:
            def __init__(self):
                self.disabled = False

            def disable_adapter(self):
                adapter = self

                class Context:
                    def __enter__(self):
                        adapter.disabled = True

                    def __exit__(self, *_):
                        adapter.disabled = False

                return Context()

        adapter = Adapter()
        calls = []
        velocities = [
            torch.zeros(1, 21, 4, dtype=torch.bfloat16),
            torch.ones(1, 21, 4, dtype=torch.bfloat16),
            torch.full((1, 21, 4), 2.0, dtype=torch.bfloat16),
        ]
        trainable = torch.nn.Parameter(
            torch.full((1, 21, 4), 2.0, dtype=torch.bfloat16)
        )

        def predict(_renderer, batch):
            calls.append((batch["label"], adapter.disabled, torch.is_grad_enabled()))
            if adapter.disabled:
                return velocities[len(calls) - 1]
            return trainable

        state = {
            name: torch.tensor([1]) for name in trainer.SHARED_STATE_FIELDS
        }
        batches = []
        for index, label in enumerate(("negative", "noop", "action")):
            batches.append(
                {
                    **state,
                    "label": label,
                    "input_ids": torch.tensor([[index]]),
                }
            )
        source = torch.zeros(1, 21, 4, dtype=torch.float32)
        auxiliary = {
            "branch_state_mode": "source_target_bridge_clean_field",
            "shared_noisy": torch.zeros_like(source),
            "sigma": torch.tensor(1.0, dtype=torch.float32),
            "source_clean": source,
            "target_clean": source,
        }
        with mock.patch.object(
            trainer.motion, "renderer_velocity_prediction", side_effect=predict
        ):
            result = trainer._endpoint_fields(
                renderer=object(),
                adapter_controller=adapter,
                negative_batch=batches[0],
                noop_batch=batches[1],
                action_batch=batches[2],
                auxiliary=auxiliary,
                step_index=0,
            )
        self.assertEqual(
            calls,
            [
                ("negative", True, False),
                ("noop", True, False),
                ("action", True, False),
                ("action", False, True),
            ],
        )
        result.field_loss.backward()
        self.assertIsNotNone(trainable.grad)

    def test_gamma_zero_total_loss_keeps_effective_adapted_action_gradient(self) -> None:
        base_negative = torch.zeros(1, 21, 2, 2, dtype=torch.float32)
        base_noop = torch.zeros_like(base_negative)
        base_action = torch.linspace(
            0.0, 1.0, base_negative.numel(), dtype=torch.float32
        ).reshape_as(base_negative)
        adapted_action = torch.nn.Parameter(base_action + 0.25)
        prior = trainer.pgt.frozen_prior(base_action, base_noop)
        replay = trainer.motion.charbonnier_distance(
            adapted_action, base_action, scale=trainer.CHARBONNIER_SCALE
        )
        endpoint = trainer.EndpointFields(
            base_negative=base_negative,
            base_noop=base_noop,
            base_action=base_action,
            adapted_action=adapted_action,
            student_executed=prior,
            teacher_executed=prior,
            field_loss=torch.zeros((), dtype=torch.float32),
            replay_loss=replay,
        )
        auxiliary = {"sigma": torch.tensor(0.2, dtype=torch.float32)}
        endpoints = {
            name: ({}, {}, {}, auxiliary) for name in ("source", "target")
        }
        route = trainer.motion.Route(
            iid="late-replay",
            tier="motion_only",
            full_target_weight=0.0,
        )
        with mock.patch.object(
            trainer, "_endpoint_fields", side_effect=(endpoint, endpoint)
        ):
            total, parts = trainer._prior_tangent_bridge_losses(
                renderer=object(),
                adapter_controller=object(),
                endpoints=endpoints,
                route=route,
                step_index=35,
                args=_args(),
            )
        self.assertEqual(float(parts["gamma"]), 0.0)
        self.assertEqual(float(parts["late_replay_gate"]), 1.0)
        self.assertTrue(total.requires_grad)
        total.backward()
        self.assertIsNotNone(adapted_action.grad)
        self.assertGreater(float(adapted_action.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

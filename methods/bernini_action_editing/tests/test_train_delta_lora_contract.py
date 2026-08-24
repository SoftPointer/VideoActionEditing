from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_delta_lora as delta


SHA1 = "1" * 40
SHA256 = "2" * 64


def _args(**overrides):
    values = {
        "num_frames": 81,
        "max_steps": 10,
        "save_every": 1,
        "resume": None,
        "init_adapter_checkpoint": None,
        "routing_jsonl": "/strict-359.jsonl",
        "expected_routing_jsonl_sha256": SHA256,
        "unreviewed_tier": "reject",
        "learning_rate": 3e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 20260806,
        "lora_scope": "cross_q",
        "branch_state_mode": "source_target_bridge_clean_field",
        "minimum_training_sigma": 0.1,
        "inverse_sigma_weight_floor": delta.sigma_strata.PINNED_POSITIVE_SIGMAS[-1],
        "motion_loss_weight": 1.0,
        "copy_loss_weight": 0.0,
        "boundary_gauge_loss_weight": 0.0,
        "motion_objective": "causal_boundary_charbonnier",
        "bridge_consistency_weight": 0.1,
        "causal_ema_decay": 0.5,
        "charbonnier_scale": 0.1,
        "quotient_weight": 0.5,
        "high_noise_floor": 1.0,
        "high_noise_power": 2.0,
        "temporal_lags": [1, 2, 4],
        "noop_instruction": delta.motion.DEFAULT_NOOP_INSTRUCTION,
        "expected_bernini_commit": SHA1,
        "expected_veomni_commit": SHA1,
        "method_source_revision": SHA1,
        "expected_checkpoint_tree_sha256": delta.legacy.CHECKPOINT_TREE_SHA256,
        "method_source_archive_sha256": SHA256,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _v3_args(**overrides):
    """Construct an explicit v3 ablation; bridge loss must never leak into it."""

    values = {
        "branch_state_mode": "shared_noisy_clean_field",
        "motion_objective": "causal_boundary_multilag",
        "bridge_consistency_weight": 0.0,
        "copy_loss_weight": 0.5,
        "boundary_gauge_loss_weight": 0.5,
        "lora_scope": "cross_q_out",
        "learning_rate": 5e-5,
    }
    values.update(overrides)
    return _args(**values)


class CliContractTests(unittest.TestCase):
    def test_defaults_are_v4_bridge_consistent_robust_clean_fields(self) -> None:
        parser = delta.build_parser()
        args = parser.parse_args(
            [
                "--bernini-root", "/b",
                "--veomni-root", "/v",
                "--checkpoint", "/c",
                "--preprocessed-parquet-dir", "/d",
                "--dataset-summary", "/s",
                "--output", "/o",
                "--method-source-revision", SHA1,
                "--method-source-archive-sha256", SHA256,
            ]
        )
        self.assertEqual(args.unreviewed_tier, "reject")
        self.assertEqual(args.learning_rate, 3e-5)
        self.assertEqual(args.lora_scope, "cross_q")
        self.assertEqual(args.branch_state_mode, "source_target_bridge_clean_field")
        self.assertEqual(args.minimum_training_sigma, 0.1)
        self.assertEqual(
            args.inverse_sigma_weight_floor,
            delta.sigma_strata.PINNED_POSITIVE_SIGMAS[-1],
        )
        self.assertEqual(args.copy_loss_weight, 0.0)
        self.assertEqual(args.boundary_gauge_loss_weight, 0.0)
        self.assertEqual(args.motion_objective, "causal_boundary_charbonnier")
        self.assertEqual(args.bridge_consistency_weight, 0.1)
        self.assertEqual(args.causal_ema_decay, 0.5)
        self.assertEqual(args.charbonnier_scale, 0.1)
        self.assertEqual(args.temporal_lags, [1, 2, 4])

    def test_resume_and_initialization_are_mutually_exclusive(self) -> None:
        with self.assertRaises(delta.DeltaTrainingError):
            delta.validate_cli(_args(resume="/r", init_adapter_checkpoint="/i"))

    def test_invalid_loss_or_lag_fails_closed(self) -> None:
        for overrides in (
            {"copy_loss_weight": -0.1},
            {"copy_loss_weight": 0.1},
            {"motion_loss_weight": -0.1},
            {"motion_loss_weight": float("nan")},
            {"boundary_gauge_loss_weight": -0.1},
            {"boundary_gauge_loss_weight": float("nan")},
            {"bridge_consistency_weight": -0.1},
            {"bridge_consistency_weight": float("nan")},
            {"bridge_consistency_weight": 0.0},
            {"causal_ema_decay": -0.1},
            {"causal_ema_decay": 1.0},
            {"charbonnier_scale": 0.0},
            {"high_noise_floor": 1.1},
            {"temporal_lags": [1, 1]},
            {"temporal_lags": [21]},
            {"quotient_weight": -0.1},
            {"minimum_training_sigma": 0.0},
            {"minimum_training_sigma": 1.0},
            {"inverse_sigma_weight_floor": 0.0},
            {"inverse_sigma_weight_floor": 0.05},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(delta.DeltaTrainingError):
                    delta.validate_cli(_args(**overrides))

    def test_zero_weight_v3_causal_ablations_are_explicitly_supported(self) -> None:
        for name in (
            "motion_loss_weight",
            "copy_loss_weight",
            "boundary_gauge_loss_weight",
            "quotient_weight",
        ):
            with self.subTest(name=name):
                delta.validate_cli(_v3_args(**{name: 0.0}))

    def test_unknown_motion_objective_fails_closed(self) -> None:
        with self.assertRaises(delta.DeltaTrainingError):
            delta.validate_cli(_args(motion_objective="unknown"))

    def test_same_state_mode_binds_the_internal_noop_prompt(self) -> None:
        with self.assertRaises(delta.DeltaTrainingError):
            delta.validate_cli(_args(noop_instruction="some other no-op"))
        delta.validate_cli(
            _v3_args(
                branch_state_mode="separate_clean_paths",
                boundary_gauge_loss_weight=0.0,
                bridge_consistency_weight=0.0,
                noop_instruction="legacy ablation prompt",
            )
        )

    def test_boundary_gauge_is_bound_to_same_state_causal_objective(self) -> None:
        for overrides in (
            {"motion_objective": "quotient_multilag"},
            {"motion_objective": "raw_delta"},
            {"branch_state_mode": "separate_clean_paths"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(
                    delta.DeltaTrainingError, "boundary gauge"
                ):
                    delta.validate_cli(
                        _v3_args(
                            boundary_gauge_loss_weight=0.5,
                            bridge_consistency_weight=0.0,
                            **overrides,
                        )
                    )

        delta.validate_cli(
            _v3_args(
                motion_objective="quotient_multilag",
                boundary_gauge_loss_weight=0.0,
                bridge_consistency_weight=0.0,
            )
        )

    def test_v4_requires_pinned_routing_hash_and_rejects_v3_objective(self) -> None:
        for overrides in (
            {"expected_routing_jsonl_sha256": None},
            {"expected_routing_jsonl_sha256": "A" * 64},
            {"expected_routing_jsonl_sha256": "2" * 63},
            {"motion_objective": "causal_boundary_multilag"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(delta.DeltaTrainingError):
                    delta.validate_cli(_args(**overrides))

    def test_auh_launcher_preflights_exact_numeric_and_matched_inference(self) -> None:
        launcher = (
            METHOD_ROOT / "scripts/auh_train_delta_lora.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("test_tri_branch_unipc.py", launcher)
        self.assertIn("test_inference_sigma_strata.py", launcher)
        self.assertIn("test_infer_c2fr_lora_contract.py", launcher)
        self.assertIn("BERNINI_OFFICIAL_ROOT", launcher)
        self.assertIn("--boundary-gauge-loss-weight", launcher)
        self.assertIn("--bridge-consistency-weight", launcher)
        self.assertIn("--expected-routing-jsonl-sha256", launcher)
        self.assertIn("representation=robust_idempotent_Q0", launcher)
        self.assertIn("temporal_ema=disabled", launcher)
        self.assertIn("--high-noise-floor", launcher)
        self.assertNotIn("--causal-ema-decay", launcher)


class ResumeContractTests(unittest.TestCase):
    def test_receipt_digest_and_immutable_contract_are_both_required(self) -> None:
        immutable = {"value": {"seed": 7}, "digest": delta.legacy.object_sha256({"seed": 7})}
        receipt = {
            "schema_version": delta.RECEIPT_SCHEMA,
            "global_step": 3,
            "immutable_contract": immutable,
            "optimizer": {"checkpoint_state_digest": "a" * 64},
        }
        receipt["receipt_digest"] = delta.legacy.object_sha256(receipt)
        self.assertEqual(delta._validate_resume_receipt(receipt, immutable=immutable), 3)

        tampered = dict(receipt)
        tampered["global_step"] = 4
        with self.assertRaises(delta.DeltaTrainingError):
            delta._validate_resume_receipt(tampered, immutable=immutable)

        other = {"value": {"seed": 8}, "digest": delta.legacy.object_sha256({"seed": 8})}
        with self.assertRaises(delta.DeltaTrainingError):
            delta._validate_resume_receipt(receipt, immutable=other)

    def test_optimizer_parameter_order_is_explicit(self) -> None:
        named = [("b", object()), ("a", object())]
        self.assertEqual(delta._optimizer_parameter_names(named), ["b", "a"])

    def test_v3_receipt_cannot_resume_v4_bridge_training(self) -> None:
        immutable = {"value": {}, "digest": delta.legacy.object_sha256({})}
        receipt = {
            "schema_version": "bernini-r-1p3b-c2fr-lora-receipt-v3",
            "global_step": 3,
            "immutable_contract": immutable,
        }
        receipt["receipt_digest"] = delta.legacy.object_sha256(receipt)
        with self.assertRaises(delta.DeltaTrainingError):
            delta._validate_resume_receipt(receipt, immutable=immutable)


class SupervisionReceiptTests(unittest.TestCase):
    def test_v3_noop_field_and_optional_copy_calibration_are_not_conflated(self) -> None:
        control = delta._supervision_receipt(
            _v3_args(
                branch_state_mode="shared_noisy_clean_field",
                motion_objective="raw_delta",
                motion_loss_weight=1.0,
                copy_loss_weight=0.0,
                boundary_gauge_loss_weight=0.0,
                bridge_consistency_weight=0.0,
            )
        )
        self.assertIs(control["counterfactual_noop_forward"], True)
        self.assertIs(control["copy_calibration_enabled"], False)
        self.assertIs(control["temporal_quotient_enabled"], False)
        self.assertIs(control["raw_delta_enabled"], True)
        self.assertIs(control["multiscale_enabled"], False)
        self.assertEqual(control["motion_objective"], "raw_delta")

    def test_v4_receipt_declares_four_branch_bridge_and_robust_field(self) -> None:
        method = delta._supervision_receipt(_args())
        self.assertEqual(method["action_noop_forwards_per_optimizer_step"], 4)
        self.assertIs(method["copy_calibration_enabled"], False)
        self.assertIs(method["temporal_quotient_enabled"], False)
        self.assertIs(method["causal_boundary_quotient_enabled"], True)
        self.assertIs(method["causal_boundary_projection_enabled"], True)
        self.assertIs(method["boundary_gauge_enabled"], False)
        self.assertEqual(method["boundary_gauge_target"], "zero_first_latent_phase")
        self.assertIs(method["boundary_gauge_uses_target_appearance"], False)
        self.assertIs(method["raw_delta_enabled"], False)
        self.assertIs(method["multiscale_enabled"], False)
        self.assertIs(method["exact_same_noisy_query"], True)
        self.assertIs(method["only_text_condition_differs"], True)
        self.assertEqual(method["bridge_endpoints"], [0.0, 1.0])
        self.assertIs(method["bridge_consistency_enabled"], True)
        self.assertEqual(method["bridge_consistency_weight"], 0.1)
        self.assertIs(method["causal_ema_enabled"], False)
        self.assertIsNone(method["causal_ema_decay"])
        self.assertEqual(method["charbonnier_scale"], 0.1)
        self.assertEqual(
            method["inference_sigma_stratification"],
            "exact_40_step_flow_shift_5_cycle",
        )
        self.assertEqual(
            method["inference_sigma_schedule_sha256"],
            delta.sigma_strata.SCHEDULE_SHA256,
        )
        self.assertEqual(
            method["predicted_clean_delta_formula"],
            "-sigma * (v_action - v_noop)",
        )
        self.assertEqual(
            method["clean_reconstruction_numeric_program"],
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity",
        )
        self.assertEqual(method["training_sigma_representation"], "cpu_fp32_0d")
        self.assertEqual(
            method["branch_prediction_dtype_before_clean_reconstruction"],
            "bfloat16",
        )
        self.assertEqual(
            method["copy_boundary_loss_multiplier"],
            "not_enabled",
        )
        self.assertEqual(method["motion_loss_multiplier"], "1 / sigma")
        self.assertEqual(
            method["target_projection"],
            "executable_target=source+Q0(raw_target-source)",
        )
        self.assertIs(method["target_projection_idempotent"], True)


class RoutingStreamTests(unittest.TestCase):
    def test_rejects_do_not_repeat_the_first_later_eligible_row(self) -> None:
        class Router:
            def route(self, iid):
                tier = "reject" if iid in {"0", "2"} else "motion_only"
                return delta.motion.Route(iid, tier, 0.0)

        dataset = [{"iid": str(index)} for index in range(5)]
        eligible = delta._build_eligible_routes(dataset, Router())
        self.assertEqual([index for index, _ in eligible], [1, 3, 4])
        observed = [
            delta._next_routed_row(dataset, eligible, ordinal=ordinal)[0]
            for ordinal in range(7)
        ]
        self.assertEqual(observed, [1, 3, 4, 1, 3, 4, 1])

    def test_zero_loss_rows_fail_but_full_target_control_is_allowed(self) -> None:
        no_auxiliary = _v3_args(
            motion_loss_weight=0.0,
            copy_loss_weight=0.0,
            boundary_gauge_loss_weight=0.0,
            bridge_consistency_weight=0.0,
        )
        motion_only = [(0, delta.motion.Route("a", "motion_only", 0.0))]
        with self.assertRaises(delta.DeltaTrainingError):
            delta._validate_active_supervision(no_auxiliary, motion_only)

        full_only = [(0, delta.motion.Route("a", "full_pair", 1.0))]
        delta._validate_active_supervision(no_auxiliary, full_only)

        mixed = full_only + [(1, delta.motion.Route("b", "motion_only", 0.0))]
        with self.assertRaises(delta.DeltaTrainingError):
            delta._validate_active_supervision(no_auxiliary, mixed)

    def test_v4_strict_router_accepts_only_hash_bound_359_motion_rows(self) -> None:
        class Router:
            def __init__(self, receipt):
                self._receipt = receipt

            def receipt(self):
                return self._receipt

        valid_receipt = {
            "path": "/strict-359.jsonl",
            "default_tier": "reject",
            "file_sha256": SHA256,
            "explicit_route_counts": {
                "full_pair": 0,
                "motion_only": 359,
                "reject": 285,
            },
        }
        eligible = [
            (index, delta.motion.Route(f"iid-{index}", "motion_only", 0.0))
            for index in range(359)
        ]
        delta._validate_v4_strict_router(
            _args(), Router(valid_receipt), eligible
        )

        invalid_cases = (
            (
                _args(expected_routing_jsonl_sha256="3" * 64),
                valid_receipt,
                eligible,
            ),
            (
                _args(),
                {
                    **valid_receipt,
                    "explicit_route_counts": {
                        "full_pair": 0,
                        "motion_only": 358,
                        "reject": 286,
                    },
                },
                eligible,
            ),
            (_args(), valid_receipt, eligible[:-1]),
            (
                _args(),
                valid_receipt,
                [
                    *eligible[:-1],
                    (358, delta.motion.Route("iid-358", "full_pair", 1.0)),
                ],
            ),
            (
                _args(),
                {**valid_receipt, "default_tier": "motion_only"},
                eligible,
            ),
        )
        for args, receipt, routes in invalid_cases:
            with self.subTest(args=args, receipt=receipt, rows=len(routes)):
                with self.assertRaisesRegex(
                    delta.DeltaTrainingError, "strict-359"
                ):
                    delta._validate_v4_strict_router(
                        args, Router(receipt), routes
                    )

    def test_v3_ablation_does_not_apply_the_strict_359_router(self) -> None:
        class Router:
            def receipt(self):
                raise AssertionError("v3 must not inspect the v4 strict receipt")

        delta._validate_v4_strict_router(
            _v3_args(bridge_consistency_weight=0.0),
            Router(),
            [(0, delta.motion.Route("a", "motion_only", 0.0))],
        )

    def test_immutable_contract_binds_motion_ablation(self) -> None:
        class Dataset:
            signature = "dataset"

        class Router:
            digest = "routing"
            file_sha256 = SHA256

        route = delta.motion.Route("a", "motion_only", 0.0)
        common = dict(
            dataset=Dataset(),
            dataset_summary={"sha256": "summary", "index_sha256": "index"},
            router=Router(),
            eligible_routes=[(0, route)],
            target_modules=["module"],
            checkpoint=Path("/checkpoint"),
        )
        baseline = delta._immutable_contract(args=_args(), **common)
        self.assertEqual(
            baseline["value"]["motion_representation"],
            "source-relative-causal-boundary-charbonnier-v1",
        )
        self.assertEqual(
            baseline["value"]["paired_cells"],
            [
                "source_query_action",
                "source_query_noop",
                "executable_target_query_action",
                "executable_target_query_noop",
            ],
        )
        self.assertEqual(
            baseline["value"]["target_projection"],
            "executable_target=source+Q0(raw_target-source)",
        )
        self.assertIs(baseline["value"]["target_projection_idempotent"], True)
        self.assertEqual(baseline["value"]["motion_loss_multiplier"], "1 / sigma")
        self.assertEqual(baseline["value"]["bridge_fractions"], [0.0, 1.0])
        self.assertEqual(
            baseline["value"]["inference_sigma_schedule_sha256"],
            delta.sigma_strata.SCHEDULE_SHA256,
        )
        self.assertEqual(
            baseline["value"]["clean_reconstruction_numeric_program"],
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity",
        )
        self.assertEqual(
            baseline["value"]["training_sigma_representation"], "cpu_fp32_0d"
        )
        self.assertEqual(
            baseline["value"]["branch_prediction_dtype_before_clean_reconstruction"],
            "bfloat16",
        )
        v3 = delta._immutable_contract(
            args=_v3_args(bridge_consistency_weight=0.0), **common
        )
        self.assertEqual(
            delta._immutable_contract(
                args=_v3_args(
                    quotient_weight=0.0,
                    bridge_consistency_weight=0.0,
                ),
                **common,
            )["value"]["motion_representation"],
            "source-relative-multilag-v1",
        )
        for changed in (
            _v3_args(
                motion_objective="raw_delta", bridge_consistency_weight=0.0
            ),
            _v3_args(copy_loss_weight=0.0, bridge_consistency_weight=0.0),
            _v3_args(quotient_weight=0.0, bridge_consistency_weight=0.0),
            _v3_args(
                boundary_gauge_loss_weight=0.0,
                bridge_consistency_weight=0.0,
            ),
        ):
            with self.subTest(changed=changed):
                self.assertNotEqual(
                    v3["digest"],
                    delta._immutable_contract(args=changed, **common)["digest"],
                )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class BridgeLossIntegrationTests(unittest.TestCase):
    def _fixture(self):
        tokens, packed_channels = 21, 4
        sigma = torch.tensor(0.5, dtype=torch.float32)
        source_clean = torch.zeros(1, tokens, packed_channels)
        target_clean = torch.zeros_like(source_clean)
        target_clean[:, 7:] = 0.5

        def endpoint(name, query_value):
            query = torch.full_like(source_clean, query_value)
            action_prediction = ((query - target_clean) / sigma).to(
                torch.bfloat16
            )
            noop_prediction = ((query - source_clean) / sigma).to(
                torch.bfloat16
            )
            renderer_query = torch.full(
                (tokens * 2, 1, 1, 1, packed_channels), query_value
            )
            shared = {
                "input_vae_latents": renderer_query,
                "input_vae_rope": torch.zeros(tokens * 2, 4, 6),
                "vae_latents_mask": torch.tensor(
                    [[False] * tokens + [True] * tokens]
                ),
                "timesteps": torch.tensor([[500]]),
                "vae_seqlen": torch.tensor([[tokens * 2]]),
                "target_lens": torch.tensor([[tokens]]),
            }
            action_batch = {
                **shared,
                "input_ids": torch.tensor([[1, 2, 3]]),
                "target_velocity": torch.zeros(
                    tokens, 1, 1, 1, packed_channels
                ),
            }
            noop_batch = {
                **shared,
                "input_ids": torch.tensor([[4, 5, 6]]),
                "target_velocity": torch.zeros(
                    tokens, 1, 1, 1, packed_channels
                ),
            }
            auxiliary = {
                "branch_state_mode": "source_target_bridge_clean_field",
                "bridge_fraction": 0.0 if name == "source" else 1.0,
                "target_projection": (
                    "executable_target=source+Q0(raw_target-source)"
                ),
                "shared_noisy": query,
                "target_clean": target_clean,
                "source_clean": source_clean,
                "sigma": sigma,
            }
            return (
                (action_batch, noop_batch, auxiliary),
                action_prediction,
                noop_prediction,
            )

        source, source_action, source_noop = endpoint("source", 0.5)
        target, target_action, target_noop = endpoint("target", -0.25)
        # Distinct, exactly representable endpoint errors make both robust
        # motion supervision and source/target bridge consistency observable.
        source_action = source_action.clone()
        target_action = target_action.clone()
        source_action[:, 12:] += 0.125
        target_action[:, 12:] += 0.25
        return (
            {"source": source, "target": target},
            (source_action, source_noop, target_action, target_noop),
        )

    def test_four_branches_and_bridge_consistency_enter_exact_total(self) -> None:
        endpoints, predictions = self._fixture()
        self.assertTrue(
            torch.equal(
                endpoints["source"][0]["input_vae_latents"],
                endpoints["source"][1]["input_vae_latents"],
            )
        )
        self.assertTrue(
            torch.equal(
                endpoints["target"][0]["input_vae_latents"],
                endpoints["target"][1]["input_vae_latents"],
            )
        )
        self.assertFalse(
            torch.equal(
                endpoints["source"][0]["input_vae_latents"],
                endpoints["target"][0]["input_vae_latents"],
            )
        )
        args = _args(boundary_gauge_loss_weight=0.2)
        with mock.patch.object(
            delta.motion,
            "renderer_velocity_prediction",
            side_effect=predictions,
        ) as predict:
            total, parts = delta._bridge_losses(
                renderer=object(),
                endpoints=endpoints,
                route=delta.motion.Route("sample", "motion_only", 0.0),
                args=args,
            )
        self.assertEqual(predict.call_count, 4)
        observed_batches = [call.args[1] for call in predict.call_args_list]
        self.assertEqual(
            observed_batches,
            [
                endpoints["source"][0],
                endpoints["source"][1],
                endpoints["target"][0],
                endpoints["target"][1],
            ],
        )
        expected = parts["clean_field_weight"] * (
            args.motion_loss_weight * parts["high_noise_weight"] * parts["motion"]
            + args.bridge_consistency_weight * parts["bridge_consistency"]
            + args.boundary_gauge_loss_weight * parts["boundary_gauge"]
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertGreater(float(parts["motion"]), 0.0)
        self.assertGreater(float(parts["bridge_consistency"]), 0.0)
        self.assertEqual(float(parts["copy"]), 0.0)
        self.assertEqual(float(parts["full_target"]), 0.0)
        self.assertEqual(float(parts["same_state_exact"]), 1.0)

    def test_bridge_query_or_target_representation_mismatch_fails_closed(self) -> None:
        endpoints, predictions = self._fixture()
        bad_query = {
            name: tuple(values) for name, values in endpoints.items()
        }
        target_action, target_noop, target_auxiliary = bad_query["target"]
        target_noop = dict(target_noop)
        target_noop["input_vae_latents"] = (
            target_noop["input_vae_latents"].clone()
        )
        target_noop["input_vae_latents"][0, 0, 0, 0, 0] += 1.0
        bad_query["target"] = (target_action, target_noop, target_auxiliary)
        with mock.patch.object(
            delta.motion,
            "renderer_velocity_prediction",
            side_effect=predictions,
        ), self.assertRaisesRegex(delta.DeltaTrainingError, "states differ"):
            delta._bridge_losses(
                renderer=object(),
                endpoints=bad_query,
                route=delta.motion.Route("sample", "motion_only", 0.0),
                args=_args(),
            )

        endpoints, predictions = self._fixture()
        action, noop, auxiliary = endpoints["target"]
        auxiliary = dict(auxiliary)
        auxiliary["target_clean"] = auxiliary["target_clean"].clone()
        auxiliary["target_clean"][:, -1] += 0.5
        endpoints["target"] = (action, noop, auxiliary)
        with mock.patch.object(
            delta.motion,
            "renderer_velocity_prediction",
            side_effect=predictions,
        ), self.assertRaisesRegex(
            delta.DeltaTrainingError, "different target representation"
        ):
            delta._bridge_losses(
                renderer=object(),
                endpoints=endpoints,
                route=delta.motion.Route("sample", "motion_only", 0.0),
                args=_args(),
            )

    def test_receipt_embeds_exact_resume_stable_inference_sigma_strata(self) -> None:
        class Dataset:
            root = Path("/dataset")
            signature = "dataset-signature"

            def __len__(self):
                return 644

        class Router:
            def receipt(self):
                return {"strict": True}

        class Distributed:
            world_size = 4
            ulysses_size = 4

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        args = _args(max_steps=80)
        expected_strata = delta.sigma_strata.build_sigma_strata_receipt(
            completed_optimizer_steps=43
        )
        receipt = delta._build_receipt(
            args=args,
            global_step=43,
            metrics={"motion": 1.0},
            dataset=Dataset(),
            dataset_summary={"sha256": "summary", "index_sha256": "index"},
            router=Router(),
            checkpoint=Path("/checkpoint"),
            bernini_revision=SHA1,
            veomni_revision=SHA1,
            distributed=Distributed(),
            backend="nccl",
            target_modules=["module"],
            named_trainable=[("adapter.weight", parameter)],
            initialization_digest="3" * 64,
            transformers_version="test",
            immutable={"value": {"bridge": True}, "digest": "4" * 64},
            optimizer_payload={"global_step": 43},
            resumed_from=None,
            initialized_from=None,
        )
        self.assertEqual(receipt["inference_sigma_strata"], expected_strata)
        strata = receipt["inference_sigma_strata"]
        self.assertEqual(
            strata["schedule"]["schedule_sha256"],
            delta.sigma_strata.SCHEDULE_SHA256,
        )
        self.assertEqual(
            strata["schedule"]["positive_sigmas_float32_be_hex"],
            list(delta.sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX),
        )
        self.assertEqual(strata["histogram_by_schedule_index"][:3], [2, 2, 2])
        self.assertEqual(strata["histogram_by_schedule_index"][3:], [1] * 37)
        signed = dict(receipt)
        declared = signed.pop("receipt_digest")
        self.assertEqual(declared, delta.legacy.object_sha256(signed))


@unittest.skipIf(torch is None, "torch is unavailable")
class SameStateLossIntegrationTests(unittest.TestCase):
    def _fixture(self):
        tokens, packed_channels = 21 * 2, 4
        sigma = torch.tensor(0.5, dtype=torch.float32)
        source = torch.randn(1, tokens, packed_channels)
        target = torch.randn_like(source)
        noisy = torch.randn_like(source)
        action_velocity = ((noisy - target) / sigma).to(torch.bfloat16)
        noop_velocity = ((noisy - source) / sigma).to(torch.bfloat16)

        def patches(value):
            return value.squeeze(0).reshape(tokens, 1, 1, 2, 2)

        shared = {
            "input_vae_latents": torch.randn(tokens * 2, 1, 1, 2, 2),
            "input_vae_rope": torch.randn(tokens * 2, 4, 6),
            "vae_latents_mask": torch.tensor(
                [[False] * tokens + [True] * tokens]
            ),
            "timesteps": torch.tensor([[500]]),
            "vae_seqlen": torch.tensor([[tokens * 2]]),
            "target_lens": torch.tensor([[tokens]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "t5_input_lens": torch.tensor([[3]]),
        }
        action_batch = {
            **shared,
            "input_ids": torch.tensor([[1, 2, 3]]),
            "target_velocity": patches(action_velocity),
        }
        noop_batch = {
            **shared,
            "input_ids": torch.tensor([[4, 5, 6]]),
            "target_velocity": patches(noop_velocity),
        }
        auxiliary = {
            "branch_state_mode": "shared_noisy_clean_field",
            "shared_noisy": noisy,
            "target_clean": target,
            "source_clean": source,
            "sigma": sigma,
        }
        return action_batch, noop_batch, auxiliary, action_velocity, noop_velocity

    def test_total_matches_exact_clean_field_multipliers(self) -> None:
        (
            action_batch,
            noop_batch,
            auxiliary,
            action_target,
            noop_target,
        ) = self._fixture()
        action_prediction = (action_target.float() + 0.1).to(torch.bfloat16)
        noop_prediction = (noop_target.float() - 0.2).to(torch.bfloat16)
        args = _v3_args(
            boundary_gauge_loss_weight=0.2,
            bridge_consistency_weight=0.0,
        )
        route = delta.motion.Route("sample", "full_pair", 0.3)
        with mock.patch.object(
            delta.motion,
            "renderer_velocity_prediction",
            side_effect=(action_prediction, noop_prediction),
        ):
            total, parts = delta._losses(
                renderer=object(),
                action_batch=action_batch,
                copy_batch=noop_batch,
                auxiliary=auxiliary,
                route=route,
                args=args,
            )
        expected = parts["clean_field_weight"] * (
            args.motion_loss_weight * parts["high_noise_weight"] * parts["motion"]
            + args.copy_loss_weight * parts["copy"]
            + args.boundary_gauge_loss_weight * parts["boundary_gauge"]
        ) + route.full_target_weight * parts["full_target"]
        self.assertTrue(torch.allclose(total, expected))
        self.assertEqual(float(parts["clean_field_weight"]), 2.0)
        self.assertEqual(float(parts["same_state_exact"]), 1.0)

    def test_auxiliary_transfer_preserves_cpu_fp32_scalar_sigma(self) -> None:
        auxiliary = {
            "sigma": torch.tensor([0.5], dtype=torch.float32),
            "shared_noisy": torch.ones(1, 2, 3, dtype=torch.float32),
        }
        moved = delta._move_auxiliary_to_device(
            auxiliary,
            device=torch.device("cpu"),
            branch_state_mode="shared_noisy_clean_field",
        )
        self.assertEqual(moved["sigma"].ndim, 0)
        self.assertEqual(moved["sigma"].device.type, "cpu")
        self.assertEqual(moved["sigma"].dtype, torch.float32)
        self.assertTrue(torch.equal(moved["shared_noisy"], auxiliary["shared_noisy"]))

        for invalid in (
            torch.tensor([0.5], dtype=torch.bfloat16),
            torch.tensor([0.5, 0.5], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    delta.DeltaTrainingError, "CPU fp32"
                ):
                    delta._move_auxiliary_to_device(
                        {**auxiliary, "sigma": invalid},
                        device=torch.device("cpu"),
                        branch_state_mode="shared_noisy_clean_field",
                    )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "ROCm/CUDA is unavailable",
    )
    def test_accelerator_scalar_program_has_finite_backward(self) -> None:
        device = torch.device("cuda", 0)
        noisy = torch.randn(1, 42, 8, device=device, dtype=torch.float32)
        velocity = torch.randn(
            1,
            42,
            8,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        sigma = torch.tensor(0.5, dtype=torch.float32)
        clean = delta.motion.same_state_clean_predictions(
            velocity,
            torch.zeros_like(velocity),
            noisy,
            sigma,
        )[0]
        expected = noisy - sigma * velocity
        self.assertTrue(torch.equal(clean, expected))
        clean.square().mean().backward()
        self.assertIsNotNone(velocity.grad)
        self.assertTrue(torch.isfinite(velocity.grad).all())
        self.assertGreater(float(velocity.grad.float().norm().cpu()), 0.0)

    def test_loss_rejects_precast_fp32_branch_predictions(self) -> None:
        action_batch, noop_batch, auxiliary, action_target, noop_target = self._fixture()
        with mock.patch.object(
            delta.motion,
            "renderer_velocity_prediction",
            side_effect=(action_target.float(), noop_target.float()),
        ), self.assertRaisesRegex(
            delta.DeltaTrainingError, "native bf16 branches"
        ):
            delta._losses(
                renderer=object(),
                action_batch=action_batch,
                copy_batch=noop_batch,
                auxiliary=auxiliary,
                route=delta.motion.Route("sample", "motion_only", 0.0),
                args=_v3_args(bridge_consistency_weight=0.0),
            )

    def test_wrong_renderer_state_fails_closed(self) -> None:
        action_batch, noop_batch, auxiliary, action_target, noop_target = self._fixture()
        noop_batch = dict(noop_batch)
        noop_batch["input_vae_rope"] = noop_batch["input_vae_rope"].clone()
        noop_batch["input_vae_rope"][0, 0, 0] += 1.0
        with mock.patch.object(
            delta.motion,
            "renderer_velocity_prediction",
            side_effect=(action_target, noop_target),
        ), self.assertRaisesRegex(delta.DeltaTrainingError, "states differ"):
            delta._losses(
                renderer=object(),
                action_batch=action_batch,
                copy_batch=noop_batch,
                auxiliary=auxiliary,
                route=delta.motion.Route("sample", "motion_only", 0.0),
                args=_v3_args(bridge_consistency_weight=0.0),
            )

    def test_resume_receipt_binds_exact_lora_tensor_bytes(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        named = [("adapter.weight", parameter)]
        digest = delta._checkpoint_parameter_digest(named)
        receipt = {"adapter": {"checkpoint_parameter_digest": digest}}
        self.assertEqual(
            delta._validate_loaded_parameter_digest(receipt, named), digest
        )
        with torch.no_grad():
            parameter.add_(1.0)
        with self.assertRaisesRegex(delta.DeltaTrainingError, "tensors differ"):
            delta._validate_loaded_parameter_digest(receipt, named)

    def test_resume_receipt_binds_exact_recursive_optimizer_state(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = torch.optim.AdamW([parameter], lr=5e-5)
        parameter.grad = torch.tensor([0.25, -0.5])
        optimizer.step()
        payload = delta._optimizer_checkpoint_payload(
            optimizer=optimizer,
            global_step=1,
            immutable={"value": {"branch": "same-state"}, "digest": "d" * 64},
            parameter_names=["adapter.weight"],
        )
        digest = delta._stable_recursive_digest(payload)
        receipt = {"optimizer": {"checkpoint_state_digest": digest}}
        self.assertEqual(
            delta._validate_loaded_optimizer_digest(receipt, payload), digest
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "optimizer.pt"
            torch.save(payload, checkpoint)
            try:
                reloaded = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
            except TypeError:
                reloaded = torch.load(checkpoint, map_location="cpu")
        self.assertEqual(delta._stable_recursive_digest(reloaded), digest)
        self.assertEqual(
            delta._validate_loaded_optimizer_digest(receipt, reloaded), digest
        )

        reordered = dict(reversed(list(payload.items())))
        self.assertEqual(delta._stable_recursive_digest(reordered), digest)

        changed = {
            **payload,
            "optimizer": {
                **payload["optimizer"],
                "state": {
                    key: dict(value) for key, value in payload["optimizer"]["state"].items()
                },
            },
        }
        state_key = next(iter(changed["optimizer"]["state"]))
        changed_state = changed["optimizer"]["state"][state_key]
        changed_state["exp_avg"] = changed_state["exp_avg"].clone()
        changed_state["exp_avg"].reshape(-1)[0] += 1.0
        with self.assertRaisesRegex(delta.DeltaTrainingError, "optimizer state differs"):
            delta._validate_loaded_optimizer_digest(receipt, changed)

        list_reordered = {**payload, "parameter_names": ["b", "a"]}
        self.assertNotEqual(
            delta._stable_recursive_digest(list_reordered), digest
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:  # Documentation/static-only environments.
    torch = None


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "semantic_anchor_action_pca_residual_vae_v3.py"


class SemanticAnchorActionPcaResidualV3StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNTIME_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _definition(self, name: str, kind: type[ast.AST]) -> ast.AST:
        matches = [
            node for node in ast.walk(self.tree)
            if isinstance(node, kind) and getattr(node, "name", None) == name
        ]
        self.assertEqual(len(matches), 1, name)
        return matches[0]

    def test_config_is_exact_preregistered_seed_and_rejects_mutation(self) -> None:
        config = self._definition("Config", ast.ClassDef)
        config_text = ast.get_source_segment(self.source, config) or ""
        self.assertIn("seed: int = 20260819", config_text)
        self.assertIn("if self != Config():", config_text)
        self.assertIn('raise ValueError("v3 hyperparameters are exact-preregistered and immutable")', config_text)

    def test_target_is_anchor_only_and_source_mutation_cannot_enter_target(self) -> None:
        node = self._definition("anchor_action_target", ast.FunctionDef)
        text = ast.get_source_segment(self.source, node) or ""
        # v3 may delegate this exact target helper to the frozen v2 authority;
        # inspect the delegated implementation when the wrapper is used.
        target_text = text
        if "v2.anchor_action_target(pair)" in text:
            target_text += self._v2_target_source()
        self.assertIn("anchor_sequence", target_text)
        self.assertNotIn("source_sequence", target_text)
        self.assertNotIn("source_relative", target_text)
        self.assertNotIn("source_features", target_text)
        self.assertTrue(
            "temporal_center(anchor ordered DINO full768)" in self.source
            or "temporal_center(anchor ordered DINO)" in self.source
        )

    def _v2_target_source(self) -> str:
        v2_path = METHOD_ROOT / "semantic_anchor_action_sequence_vae_v2.py"
        v2_source = v2_path.read_text(encoding="utf-8")
        v2_tree = ast.parse(v2_source)
        node = next(
            item for item in ast.walk(v2_tree)
            if isinstance(item, ast.FunctionDef)
            and item.name == "anchor_action_target"
        )
        return ast.get_source_segment(v2_source, node) or ""

    def test_models_are_frozen_pca_initialized_and_low_capacity(self) -> None:
        for name in (
            "PCAInitializedSequenceCore",
            "PCAInitializedDeterministicAE",
            "PCAInitializedDirectBetaVAE",
        ):
            self._definition(name, ast.ClassDef)
        self.assertIn("nn.Conv1d", self.source)
        self.assertIn("expected_vae_extra", self.source)
        self.assertIn("vae_count - ae_count != expected_vae_extra", self.source)
        self.assertIn('"exact_parameter_count_matched": False', self.source)
        self.assertIn('"zero_initialized_nonlinear_heads": True', self.source)

    def test_initialization_contract_distinguishes_pca_target_from_pca_comparator(self) -> None:
        self.assertIn('"pca_is_model_target": False', self.source)
        contract = self.source[self.source.index('"initialization_contract"'):]
        self.assertIn(
            '"step0_posterior_mean_reconstruction_matches_frame_pca_within_abs_3e_5": True',
            contract,
        )
        self.assertIn('"decoder_nonlinear_output_orthogonal_to_pca": True', contract)
        self.assertIn('"learned_output_increment_entirely_pca_orthogonal": True', contract)
        self.assertIn('"posterior_observes_full_target_residual": False', contract)
        self.assertIn('"raw_identity_skip": False', contract)
        self.assertIn('"frozen_pca_input_output_path": True', contract)
        self.assertIn("reconstruction = linear + correction", self.source)
        self.assertIn("correction = raw_correction - (", self.source)

    def test_kl_schedule_and_post_kl_eligibility_are_source_locked(self) -> None:
        schedule = ast.get_source_segment(
            self.source, self._definition("kl_weight", ast.FunctionDef)
        ) or ""
        self.assertIn("step <= config.kl_zero_steps", schedule)
        self.assertIn("config.beta_kl * min(1.0, progress)", schedule)
        training = ast.get_source_segment(
            self.source,
            self._definition("train_with_preregistered_selection", ast.FunctionDef),
        ) or ""
        self.assertIn("eligibility_step = config.kl_warmup_steps + config.full_beta_plateau_steps", training)
        self.assertIn("arm == \"deterministic_ae\" or step >= eligibility_step", training)
        self.assertIn("for step in range(1, config.max_steps + 1):", training)
        self.assertIn("executed_steps != config.max_steps", training)

    def test_no_source_residual_target_or_family_optimizer_path(self) -> None:
        target = ast.get_source_segment(
            self.source, self._definition("anchor_action_target", ast.FunctionDef)
        ) or ""
        loss = ast.get_source_segment(self.source, self._definition("_loss", ast.FunctionDef)) or ""
        training = ast.get_source_segment(
            self.source,
            self._definition("train_with_preregistered_selection", ast.FunctionDef),
        ) or ""
        self.assertNotIn("source_sequence", target)
        self.assertNotIn("residual_target", target + loss + training)
        self.assertNotIn("family", loss)
        self.assertIn("optimizer = torch.optim.AdamW(", training)
        self.assertIn("model.parameters()", training)
        self.assertNotIn("family_by_iid", training)
        self.assertNotIn("source_features", training)
        self.assertIn('"family_or_transform_labels_consumed_by_model_or_optimizer": False', self.source)
        self.assertIn('"source_subtracted": False', self.source)

    def test_train_and_eval_bundle_key_allowlists_are_exact(self) -> None:
        self.assertIn("if type(bundle) is not dict or set(bundle) != expected_keys:", self.source)
        self.assertIn("TRAIN_KEYS = {", self.source)
        self.assertIn("EVAL_KEYS = {", self.source)
        self.assertIn('source_features_present', self.source)

    def test_all_commands_have_binding_guard_and_receipts_fail_closed(self) -> None:
        for name in (
            "prepare_fold", "train_fold_arm", "compare_fold", "aggregate_oof",
            "prepare_refit", "train_refit",
        ):
            node = self._definition(name, ast.FunctionDef)
            text = ast.get_source_segment(self.source, node) or ""
            self.assertIn("= _binding()", text, name)
            self.assertGreaterEqual(text.count("_assert_binding_unchanged("), 2, name)
        self.assertGreaterEqual(self.source.count('"prior_generation_qualified": False'), 2)
        self.assertGreaterEqual(self.source.count('"inference_authorized": False'), 2)
        self.assertIn('"vae_necessary": None', self.source)
        self.assertIn("UNDETERMINED_SINGLE_EXECUTION", self.source)

    def test_parser_declares_six_commands_including_refit(self) -> None:
        for command in (
            "prepare-fold", "train-fold", "compare-fold", "aggregate-oof",
            "prepare-refit", "train-refit",
        ):
            self.assertIn(f'add_parser("{command}")', self.source)
        self.assertIn("def build_parser", self.source)


@unittest.skipIf(torch is None, "torch is unavailable")
class SemanticAnchorActionPcaResidualV3TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if str(METHOD_ROOT) not in sys.path:
            sys.path.insert(0, str(METHOD_ROOT))
        from methods.bernini_action_editing import semantic_anchor_action_pca_residual_vae_v3

        cls.runtime = semantic_anchor_action_pca_residual_vae_v3

    def _pair(self, iid: str, anchor: torch.Tensor, source: torch.Tensor):
        return self.runtime.authority.PairRecord(
            iid=iid,
            family="family-a",
            instruction_sha256=hashlib.sha256(f"instruction:{iid}".encode()).hexdigest(),
            group_id=hashlib.sha256(f"group:{iid}".encode()).hexdigest(),
            strict=True,
            anchor_sequence=anchor,
            source_sequence=source,
        )

    def _pca(self, latent_dim: int | None = None) -> dict[str, torch.Tensor]:
        latent_dim = self.runtime.Config().latent_dim if latent_dim is None else latent_dim
        return {
            "mean": torch.zeros((1, 768), dtype=torch.float32),
            "basis": torch.eye(768, dtype=torch.float32)[:, :latent_dim].contiguous(),
            "latent_scale": torch.ones((latent_dim,), dtype=torch.float32),
        }

    def test_source_mutation_is_bit_invariant(self) -> None:
        generator = torch.Generator().manual_seed(19)
        anchor = torch.randn((32, 768), generator=generator)
        first = self._pair("iid-1", anchor, torch.zeros_like(anchor))
        second = self._pair("iid-1", anchor, torch.randn(anchor.shape, generator=generator))
        self.assertTrue(torch.equal(
            self.runtime.anchor_action_target(first),
            self.runtime.anchor_action_target(second),
        ))

    def test_models_share_backbone_and_report_vae_variance_head(self) -> None:
        config = self.runtime.Config()
        pca = self._pca()
        ae = self.runtime._make_model("deterministic_ae", config, pca)
        vae = self.runtime._make_model("direct_beta_vae", config, pca)
        ae_count = self.runtime._parameter_count(ae)
        vae_count = self.runtime._parameter_count(vae)
        expected_extra = config.correction_hidden_dim * config.latent_dim + config.latent_dim
        self.assertEqual(vae_count - ae_count, expected_extra)
        self.assertLessEqual(ae_count, 50_000)
        self.assertLessEqual(vae_count, 50_000)

    def test_step0_reconstruction_is_pca_and_vae_residual_mean_is_zero(self) -> None:
        config = self.runtime.Config()
        generator = torch.Generator().manual_seed(701)
        basis = torch.linalg.qr(
            torch.randn((768, config.latent_dim), generator=generator),
            mode="reduced",
        ).Q.contiguous()
        pca = {
            "mean": torch.randn((1, 768), generator=generator) * 0.01,
            "basis": basis,
            "latent_scale": torch.logspace(-4, 2, config.latent_dim),
        }
        value = torch.randn((2, 32, 768), generator=torch.Generator().manual_seed(7))
        value = value - value.mean(dim=1, keepdim=True)
        for arm in ("deterministic_ae", "direct_beta_vae"):
            model = self.runtime._make_model(arm, config, pca)
            expected = self.runtime.v2._reconstruct_frame_pca(value, pca)
            sample_modes = (False, True) if arm == "direct_beta_vae" else (False,)
            for sample in sample_modes:
                result = model(value, sample=sample)
                self.assertLess(
                    float((result["reconstruction"] - expected).abs().max()),
                    1.0e-6,
                )
                if arm == "direct_beta_vae":
                    self.assertTrue(torch.equal(
                        result["mean"], torch.zeros_like(result["mean"])
                    ))
                    self.assertTrue(torch.equal(
                        result["base_latent"], model.pca_encode(value)
                    ))

    def test_complete_learned_output_increment_is_pca_orthogonal(self) -> None:
        config = self.runtime.Config()
        pca = self._pca()
        model = self.runtime.PCAInitializedDeterministicAE(config, pca)
        value = torch.randn((1, 32, 768), generator=torch.Generator().manual_seed(8))
        value = value - value.mean(dim=1, keepdim=True)
        with torch.no_grad():
            model.decoder_output.weight.normal_(mean=0.0, std=0.05)
            model.mean_output.bias[0] = 0.25
        result = model(value, sample=False)
        projection = result["nonlinear_output"] @ model.pca_basis
        self.assertLessEqual(float(projection.abs().max()), 2.0e-5)
        self.assertGreater(float(result["latent_delta"].abs().sum()), 0.0)
        pca_reconstruction = self.runtime.v2._reconstruct_frame_pca(value, pca)
        complete_increment = result["reconstruction"] - pca_reconstruction
        self.assertLessEqual(
            float((complete_increment @ model.pca_basis).abs().max()), 3.0e-5
        )

    def test_zero_residual_mean_cannot_fake_active_units(self) -> None:
        config = self.runtime.Config()
        pca = self._pca()
        model = self.runtime.PCAInitializedDirectBetaVAE(config, pca)
        value = torch.randn((4, 32, 768), generator=torch.Generator().manual_seed(81))
        value = value - value.mean(dim=1, keepdim=True)
        mechanics, reconstruction = self.runtime._vae_mechanics(
            model, value, [f"iid-{index}" for index in range(4)],
            torch.ones((1,), dtype=torch.float32), [-2.0, -1.0, 0.0, 1.0],
            pca, torch.device("cpu"), 17,
        )
        self.assertEqual(tuple(reconstruction.shape), tuple(value.shape))
        self.assertEqual(mechanics["residual_active_unit_count"], 0)
        self.assertGreater(mechanics["residual_only_kl_element_mean"], 0.0)
        self.assertTrue(
            mechanics["full_pca_latent_excluded_from_kl_and_active_unit_metrics"]
        )
        self.assertEqual(
            mechanics["posterior_sample_count"],
            self.runtime.POSTERIOR_MC_SAMPLE_COUNT,
        )
        iids = [f"iid-{index}" for index in range(4)]
        expected_posterior = self.runtime.v2._metric_rows(
            value, reconstruction, iids,
            torch.ones((1,), dtype=torch.float32), [-2.0, -1.0, 0.0, 1.0],
        )["per_iid"]
        analytic = self.runtime.v2._reconstruct_frame_pca(value, pca)
        expected_pca = self.runtime.v2._metric_rows(
            value, analytic, iids,
            torch.ones((1,), dtype=torch.float32), [-2.0, -1.0, 0.0, 1.0],
        )["per_iid"]
        for intervention, posterior, pca_row in zip(
            mechanics["residual_intervention_per_iid"],
            expected_posterior, expected_pca,
        ):
            self.assertEqual(
                intervention["posterior_raw_mse"], posterior["raw_mse"]
            )
            self.assertEqual(
                intervention["analytic_pca_residual_raw_mse"],
                pca_row["raw_mse"],
            )
            self.assertAlmostEqual(
                intervention["normalized_sample_output_variance"],
                intervention["posterior_sample_output_variance_raw"]
                / intervention["analytic_pca_residual_raw_mse"],
                places=10,
            )

    def test_step0_ae_comparison_aliases_exact_analytic_pca_tensor(self) -> None:
        config = self.runtime.Config()
        pca = self._pca()
        model = self.runtime._make_model("deterministic_ae", config, pca)
        value = torch.randn(
            (5, 32, 768), generator=torch.Generator().manual_seed(904)
        )
        value = value - value.mean(dim=1, keepdim=True)
        actual, policy = self.runtime._comparison_reconstruction(
            "deterministic_ae", model, 0, value, pca, torch.device("cpu")
        )
        analytic = self.runtime.v2._reconstruct_frame_pca(value, pca)
        self.assertTrue(torch.equal(actual, analytic))
        self.assertTrue(policy["analytic_frame_pca_alias_used"])
        self.assertLess(
            policy["checkpoint_output_max_abs_vs_analytic_frame_pca"],
            self.runtime.STEP0_MAX_ABS_TOLERANCE,
        )

    def test_step0_metric_alias_uses_pinned_rocm_tolerance(self) -> None:
        config = self.runtime.Config()
        pca = self._pca()
        model = self.runtime._make_model("deterministic_ae", config, pca)
        value = torch.randn(
            (2, 32, 768), generator=torch.Generator().manual_seed(905)
        )
        value = value - value.mean(dim=1, keepdim=True)
        analytic = self.runtime.v2._reconstruct_frame_pca(value, pca)
        with mock.patch.object(
            self.runtime, "_predict", return_value=analytic + 2.0e-5
        ):
            selected, policy = self.runtime._comparison_reconstruction(
                "deterministic_ae", model, 0, value, pca,
                torch.device("cpu"),
            )
        self.assertTrue(torch.equal(selected, analytic))
        self.assertTrue(policy["analytic_frame_pca_alias_used"])
        with mock.patch.object(
            self.runtime, "_predict", return_value=analytic + 4.0e-5
        ):
            with self.assertRaisesRegex(ValueError, "step-zero AE"):
                self.runtime._comparison_reconstruction(
                    "deterministic_ae", model, 0, value, pca,
                    torch.device("cpu"),
                )

    def test_kl_schedule_is_zero_at_100_ramps_and_is_full_at_500(self) -> None:
        config = self.runtime.Config()
        self.assertEqual(self.runtime.kl_weight(0, config), 0.0)
        self.assertEqual(self.runtime.kl_weight(100, config), 0.0)
        self.assertAlmostEqual(self.runtime.kl_weight(300, config), config.beta_kl / 2.0)
        self.assertAlmostEqual(self.runtime.kl_weight(500, config), config.beta_kl)
        self.assertAlmostEqual(self.runtime.kl_weight(700, config), config.beta_kl)

    def test_family_bootstrap_accepts_observed_fewer_than_28_but_expected_28_rejects(self) -> None:
        candidate = [{"iid": f"iid-{i}", "raw_mse": 1.0 + i} for i in range(4)]
        baseline = [{"iid": f"iid-{i}", "raw_mse": 2.0 + i} for i in range(4)]
        families = {f"iid-{i}": f"family-{i % 2}" for i in range(4)}
        result = self.runtime._family_cluster_ratio(
            candidate, baseline, families, seed=3, draws=32
        )
        self.assertEqual(result["family_count"], 2)
        with self.assertRaisesRegex(ValueError, "family closure"):
            self.runtime._family_cluster_ratio(
                candidate, baseline, families, seed=3, draws=32,
                expected_family_count=28,
            )

    def test_train_and_eval_loaders_reject_extra_keys_before_other_validation(self) -> None:
        for expected in (self.runtime.TRAIN_KEYS, self.runtime.EVAL_KEYS):
            with self.assertRaisesRegex(ValueError, "exact-key"):
                self.runtime._validate_common(
                    {key: None for key in (*expected, "unexpected_extra")},
                    {}, expected, {},
                )

    def test_arm_consumer_rejects_less_than_full_budget(self) -> None:
        config = self.runtime.Config()
        pca = self._pca()
        arm = "deterministic_ae"
        implementation = {"implementation_sha256": "a" * 64}
        fold = {"outer_fold": 0}
        receipt = {
            "arm": arm,
            "config": self.runtime.asdict(config),
            "config_sha256": self.runtime._object_sha(
                self.runtime.asdict(config)
            ),
            "fold": fold,
            "prepare_receipt_sha256": "b" * 64,
            "train_bundle_sha256": "c" * 64,
            "pca_initialization_sha256": "d" * 64,
            "implementation": implementation,
            "executed_steps": config.max_steps - 1,
            "selected_checkpoint_selection_eligible": True,
            "executed_minibatch_schedule_sha256": "e" * 64,
            "checkpoint": {
                "path": "/unused", "sha256": "f" * 64, "size_bytes": 1,
            },
            "best_step": 0,
            "parameter_count": self.runtime._parameter_count(
                self.runtime._make_model(arm, config, pca)
            ),
        }
        with mock.patch.object(
            self.runtime, "_load_receipt", return_value=receipt
        ):
            with self.assertRaisesRegex(ValueError, "arm receipt authority"):
                self.runtime._load_arm_model(
                    arm, "/unused", "1" * 64, config, fold,
                    "b" * 64, "c" * 64, pca, "d" * 64,
                    implementation,
                )

    def test_pca_subspace_validation_is_sign_safe_and_foreign_safe(self) -> None:
        pca = self._pca()
        sign_rotated = {
            "mean": pca["mean"].clone(),
            "basis": (pca["basis"] * torch.tensor(
                [(-1.0) ** index for index in range(pca["basis"].shape[1])]
            )).contiguous(),
        }
        self.assertTrue(self.runtime._pca_subspace_equivalent(
            {"mean": pca["mean"], "basis": pca["basis"]}, sign_rotated
        ))
        foreign = {
            "mean": pca["mean"].clone(),
            "basis": torch.eye(768, dtype=torch.float32)[:, 1:33].contiguous(),
        }
        self.assertFalse(self.runtime._pca_subspace_equivalent(
            {"mean": pca["mean"], "basis": pca["basis"]}, foreign
        ))

    def test_parser_rejects_oof_and_config_smuggling_into_train(self) -> None:
        parser = self.runtime.build_parser()
        base = [
            "train-fold", "--prepare-receipt", "p",
            "--expected-prepare-receipt-sha256", "a" * 64,
            "--train-bundle", "t",
            "--expected-train-bundle-sha256", "b" * 64,
            "--fold-index", "0", "--arm", "deterministic_ae",
            "--output", "o",
        ]
        for smuggled in (
            ["--exploratory-oof-bundle", "held.pt"],
            ["--max-steps", "1"],
        ):
            with mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit) as raised:
                    parser.parse_args(base + smuggled)
            self.assertEqual(raised.exception.code, 2)

    def test_binding_guard_rejects_dependency_drift(self) -> None:
        expected = self.runtime._binding()
        changed = {**expected, "implementation_sha256": "f" * 64}
        with mock.patch.object(self.runtime, "_binding", return_value=changed):
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                self.runtime._assert_binding_unchanged(expected)

    def test_parser_exposes_exact_six_commands_and_refit_handlers(self) -> None:
        parser = self.runtime.build_parser()
        subparsers = next(
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(set(subparsers.choices), {
            "prepare-fold", "train-fold", "compare-fold", "aggregate-oof",
            "prepare-refit", "train-refit",
        })
        train_options = {
            option for action in subparsers.choices["train-fold"]._actions
            for option in action.option_strings
        }
        compare_options = {
            option for action in subparsers.choices["compare-fold"]._actions
            for option in action.option_strings
        }
        self.assertNotIn("--exploratory-oof-bundle", train_options)
        self.assertIn("--exploratory-oof-bundle", compare_options)

    def test_aggregate_gate_schema_rejects_deleted_or_resigned_gate(self) -> None:
        ratio = {"ratio_95pct_ci": [1.5, 2.0]}
        improvement = {"improvement_95pct_ci": [-1.0, -0.5]}
        arm_comparison = {
            "vs_zero": {
                "clip_bootstrap": ratio,
                "family_cluster_bootstrap": ratio,
            },
            "vs_frame_pca_rank_l": {
                "clip_bootstrap": ratio,
                "family_cluster_bootstrap": ratio,
            },
            "temporal_delta_vs_frame_pca": {
                "clip_bootstrap": ratio,
                "family_cluster_bootstrap": ratio,
            },
            "cosine_improvement_vs_frame_pca": {
                "clip_bootstrap": improvement,
                "family_cluster_bootstrap": improvement,
            },
        }
        energy = {
            str(index): {"count": 1, "raw_mse": 1.0}
            for index in range(5)
        }
        arm_energy = {
            str(index): {"count": 1, "raw_mse": 2.0}
            for index in range(5)
        }
        fold_ratios = {
            arm: {
                str(fold): {"vs_zero": 2.0, "vs_frame_pca": 2.0}
                for fold in range(self.runtime.OUTER_FOLDS)
            }
            for arm in self.runtime.ARMS
        }
        summaries = {
            str(fold): {
                "selected_step": 700,
                "full_beta_exposure_steps": 200,
                "residual_only_kl_element_mean": 0.0,
                "residual_active_unit_count": 0,
            }
            for fold in range(self.runtime.OUTER_FOLDS)
        }
        effects = {
            "posterior_vs_zero_residual": {
                "clip_bootstrap": ratio,
                "family_cluster_bootstrap": ratio,
            },
            "posterior_vs_shuffled_residual": {
                "clip_bootstrap": ratio,
                "family_cluster_bootstrap": ratio,
            },
            "normalized_posterior_sample_output_variance": {
                "clip_bootstrap": {"improvement_95pct_ci": [0.0, 0.0]},
                "family_cluster_bootstrap": {
                    "improvement_95pct_ci": [0.0, 0.0]
                },
            },
        }
        mc = {
            label: {
                "clip_bootstrap": ratio,
                "family_cluster_bootstrap": ratio,
            }
            for label in ("vs_zero", "vs_frame_pca_rank_l")
        }
        aggregate = {
            "config": self.runtime.asdict(self.runtime.Config()),
            "paired_comparisons": {
                arm: arm_comparison for arm in self.runtime.ARMS
            },
            "aggregate_metrics": {
                "deterministic_ae": {"energy_strata": arm_energy},
                "direct_beta_vae": {"energy_strata": arm_energy},
                "zero_hard_baseline": {"energy_strata": energy},
                "frame_pca_rank_l_hard_baseline": {"energy_strata": energy},
            },
            "fold_point_ratios_by_arm": fold_ratios,
            "vae_fold_mechanics_summary": summaries,
            "vae_residual_effects": effects,
            "vae_posterior_mc8_comparisons": mc,
            "vae_vs_ae_retention": {
                "clip_bootstrap": ratio,
                "family_cluster_bootstrap": ratio,
            },
            "vae_normalized_sample_variance_lcb_floor": (
                self.runtime.VAE_NORMALIZED_SAMPLE_VARIANCE_LCB_FLOOR
            ),
            "selected_steps_by_fold_by_arm": {
                str(fold): {
                    "deterministic_ae": 0, "direct_beta_vae": 700,
                }
                for fold in range(self.runtime.OUTER_FOLDS)
            },
            "full644_refit_step_preregistration": {
                "strategy": "median of exact5 selected fold steps",
                "steps_by_arm": {
                    "deterministic_ae": 0, "direct_beta_vae": 700,
                },
                "frozen_before_refit": True,
            },
            "full644_refit_authorized_arms": [],
            "full644_refit_authorized_by_arm": {
                arm: False for arm in self.runtime.ARMS
            },
            "full644_refit_any_arm_authorized": False,
        }
        evidence = self.runtime._recompute_aggregate_boolean_gates(aggregate)
        gates = {}
        for arm in self.runtime.ARMS:
            gate = dict(evidence[arm])
            gate["fold_point_ratios"] = fold_ratios[arm]
            gate["energy_strata"] = self.runtime._energy_ratio_gates(
                aggregate["aggregate_metrics"], arm
            )["by_bin"]
            gate["aggregate_hard_gate"] = False
            gates[arm] = gate
        aggregate["gates"] = gates
        aggregate["vae_mechanics_gates"] = {
            key: evidence["direct_beta_vae"][key]
            for key in self.runtime.VAE_AGGREGATE_BOOLEAN_GATE_KEYS
        }
        self.assertEqual(self.runtime._authorized_arms_from_aggregate(aggregate), [])
        deleted = {**aggregate, "gates": {**gates}}
        deleted["gates"] = {
            arm: dict(values) for arm, values in gates.items()
        }
        deleted["gates"]["deterministic_ae"].pop(
            "clip_ratio_ucb_lt_1_vs_zero"
        )
        with self.assertRaisesRegex(ValueError, "arm gate"):
            self.runtime._authorized_arms_from_aggregate(deleted)
        subgate = {**aggregate, "gates": {
            arm: dict(values) for arm, values in gates.items()
        }}
        subgate["gates"]["deterministic_ae"][
            "clip_ratio_ucb_lt_1_vs_zero"
        ] = True
        with self.assertRaisesRegex(ValueError, "subgate was re-signed"):
            self.runtime._authorized_arms_from_aggregate(subgate)
        resigned = {**aggregate, "gates": {
            arm: dict(values) for arm, values in gates.items()
        }}
        resigned["gates"]["deterministic_ae"]["aggregate_hard_gate"] = True
        with self.assertRaisesRegex(ValueError, "re-signed"):
            self.runtime._authorized_arms_from_aggregate(resigned)
        step_resigned = {
            **aggregate,
            "full644_refit_step_preregistration": {
                **aggregate["full644_refit_step_preregistration"],
                "steps_by_arm": {
                    "deterministic_ae": 1, "direct_beta_vae": 700,
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "steps were re-signed"):
            self.runtime._authorized_arms_from_aggregate(step_resigned)
        vae_699 = {
            **aggregate,
            "selected_steps_by_fold_by_arm": {
                fold: dict(row)
                for fold, row in aggregate[
                    "selected_steps_by_fold_by_arm"
                ].items()
            },
            "vae_fold_mechanics_summary": {
                fold: dict(row)
                for fold, row in aggregate[
                    "vae_fold_mechanics_summary"
                ].items()
            },
            "gates": {
                arm: dict(row) for arm, row in aggregate["gates"].items()
            },
            "vae_mechanics_gates": dict(
                aggregate["vae_mechanics_gates"]
            ),
        }
        vae_699["selected_steps_by_fold_by_arm"]["0"][
            "direct_beta_vae"
        ] = 699
        vae_699["vae_fold_mechanics_summary"]["0"]["selected_step"] = 699
        vae_699["gates"]["direct_beta_vae"][
            "all_folds_full_beta_eligible"
        ] = False
        vae_699["vae_mechanics_gates"][
            "all_folds_full_beta_eligible"
        ] = False
        with self.assertRaisesRegex(ValueError, "selected-step evidence"):
            self.runtime._authorized_arms_from_aggregate(vae_699)

    def test_refit_step_zero_is_mechanically_valid_for_ae(self) -> None:
        config = self.runtime.Config()
        model = self.runtime._make_model("deterministic_ae", config, self._pca())
        history, digest = self.runtime._train_fixed_steps(
            "deterministic_ae", model,
            torch.zeros((2, 32, 768), dtype=torch.float32),
            0, config, torch.device("cpu"),
        )
        self.assertEqual(history, [])
        self.assertEqual(digest, hashlib.sha256().hexdigest())


if __name__ == "__main__":
    unittest.main()

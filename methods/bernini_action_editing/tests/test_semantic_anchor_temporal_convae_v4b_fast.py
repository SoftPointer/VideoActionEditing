from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "semantic_anchor_temporal_convae_v4b_fast.py"


class TemporalConvAEV4BFastStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNTIME_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _function_source(self, name: str) -> str:
        nodes = [
            node for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        self.assertEqual(len(nodes), 1, name)
        return ast.get_source_segment(self.source, nodes[0]) or ""

    def test_single_exact5_command_and_create_only_artifacts(self) -> None:
        self.assertEqual(self.source.count('add_parser("run-exact5")'), 1)
        self.assertEqual(self.source.count("add_parser("), 1)
        self.assertIn('path.open("xb")', self._function_source("_write_json_create_only"))
        saver = self._function_source("_save_selected_checkpoint_create_only")
        self.assertIn('path.open("xb")', saver)
        self.assertIn("os.chmod(path, 0o444)", saver)
        self.assertIn("fresh_reload_output_bit_exact", saver)
        self.assertIn("_verify_checkpoint_artifacts(checkpoint_artifacts)", self.source)

    def test_pins_and_prior_fixed_comparator_are_literal(self) -> None:
        for value in (
            "568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2",
            "f33d72320905aba135a2bb8729782cf5c89e6eee81fe1bd88aa8d24e1b585a86",
            "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973",
            "46927772a1861354ad5edeb2072ae9b1b505d235de7c2615fb11a6648f2bddca",
            "74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233",
            "tucker_b0384_t04_r096",
        ):
            self.assertIn(value, self.source)
        self.assertIn('"called_best_or_winner": False', self.source)
        self.assertIn('"parameter_or_flop_fairness_claimed": False', self.source)

    def test_training_cannot_reach_source_oof_negative_or_eval_warp(self) -> None:
        train = self._function_source("_train_fold_model")
        loss = self._function_source("_fixed_training_loss")
        for forbidden in (
            "source_sequence", "temporal_variants", "NEGATIVES",
            "exploratory_oof", "_warp_coordinate_tensor",
        ):
            self.assertNotIn(forbidden, train)
            self.assertNotIn(forbidden, loss)
        self.assertIn("training_only", self.source)
        self.assertIn('"negative_views_used_for_training": 0', self.source)
        self.assertIn('"oof_tensors_supplied_to_optimizer_or_checkpoint_selection": False', self.source)

    def test_fixed_budget_and_original_only_checkpoint_selection(self) -> None:
        train = self._function_source("_train_fold_model")
        self.assertIn("range(1, config.max_steps + 1)", train)
        self.assertNotIn("break", train)
        self.assertIn("_validation_original_mse", train)
        self.assertIn("selected_step = min", train)
        self.assertIn('"early_stopped": False', train)
        self.assertIn('checkpoint_steps: tuple[int, ...] = (0, 300, 600, 900, 1200)', self.source)

    def test_exact_b384_sole_decoder_path_and_no_skip(self) -> None:
        self.assertIn("CODE_TIME = 4", self.source)
        self.assertIn("CODE_CHANNELS = 96", self.source)
        self.assertIn("CODE_NUMEL = CODE_TIME * CODE_CHANNELS", self.source)
        self.assertIn("MAX_TRAINABLE_PARAMETERS = 150000", self.source)
        decode = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "decode"
        )
        self.assertEqual([arg.arg for arg in decode.args.args], ["self", "code"])
        decode_source = ast.get_source_segment(self.source, decode) or ""
        self.assertNotIn("value", decode_source)
        loaded_names = {
            node.id for node in ast.walk(decode)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertNotIn("raw_input", loaded_names)
        self.assertNotIn("value", loaded_names)
        self.assertIn("output.mean(dim=1", decode_source)

    def test_all_hard_gates_and_boundaries_are_explicit(self) -> None:
        for value in (
            "both_ucbs_le_1p05",
            "all_five_fold_point_ratios_le_1p05",
            "candidate_minus_0p8_teacher_margin",
            "candidate_minus_fixed_tucker_b384_margin",
            "all_five_fold_point_improvements_strictly_gt_zero",
            "decoded_temporal_codec_development_gate",
            '"latent_metric_qualified": False',
            '"action_representation_qualified": False',
            '"video_editing_qualified": False',
            '"video_model_training_performed": False',
        ):
            self.assertIn(value, self.source)

    def test_driver_calls_train_save_eval_once_in_contract_order(self) -> None:
        run = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_exact5"
        )
        run_calls = [
            node for node in ast.walk(run)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "_run_fold"
        ]
        self.assertEqual(len(run_calls), 1)
        fold = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_run_fold"
        )
        calls = {
            name: [
                node for node in ast.walk(fold)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == name
            ]
            for name in (
                "_train_fold_model", "_save_selected_checkpoint_create_only",
                "_evaluate_fold",
            )
        }
        self.assertTrue(all(len(nodes) == 1 for nodes in calls.values()))
        self.assertLess(calls["_train_fold_model"][0].lineno,
                        calls["_save_selected_checkpoint_create_only"][0].lineno)
        self.assertLess(calls["_save_selected_checkpoint_create_only"][0].lineno,
                        calls["_evaluate_fold"][0].lineno)


@unittest.skipIf(torch is None, "torch is unavailable")
class TemporalConvAEV4BFastDynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if str(METHOD_ROOT.parent.parent) not in sys.path:
            sys.path.insert(0, str(METHOD_ROOT.parent.parent))
        from methods.bernini_action_editing import semantic_anchor_temporal_convae_v4b_fast
        cls.runtime = semantic_anchor_temporal_convae_v4b_fast

    def _fit(self):
        g = torch.Generator().manual_seed(91)
        temporal = torch.linalg.qr(torch.randn(32, 4, generator=g)).Q
        content = torch.linalg.qr(torch.randn(768, 96, generator=g)).Q
        return self.runtime.v4a.FrontierFit(
            frame_mean=torch.randn(1, 768, generator=g) * 0.01,
            frame_basis=content[:, :12],
            clip_mean=torch.empty(0),
            clip_basis=torch.empty(0),
            temporal_basis=temporal,
            content_basis=content,
            fit_iid_digest="fit-iids",
            fit_input_sha256="fit-input",
            diagnostics={},
        )

    def _model(self):
        return self.runtime.TuckerInitializedTemporalConvAE(
            self._fit(), torch.ones(1)
        )

    def _pair(self, iid: str, anchor: torch.Tensor):
        return self.runtime.authority.PairRecord(
            iid=iid,
            family="family-a",
            instruction_sha256=hashlib.sha256(f"instruction:{iid}".encode()).hexdigest(),
            group_id=hashlib.sha256(f"group:{iid}".encode()).hexdigest(),
            strict=True,
            anchor_sequence=anchor,
            source_sequence=torch.full_like(anchor, 12345.0),
        )

    def _rows(self, candidate_margin: float = 1.0, candidate_error: float = 0.5):
        counts = self.runtime.FROZEN_OOF_COUNTS
        rows = []
        index = 0
        for fold, count in enumerate(counts):
            for _ in range(count):
                rows.append({
                    "iid": f"iid-{index:04d}",
                    "family": f"family-{index % 28:02d}",
                    "outer_fold": fold,
                    "teacher_margin_by_negative": {name: 0.1 for name in self.runtime.NEGATIVES},
                    "tucker_b384_margin_by_negative": {name: 0.5 for name in self.runtime.NEGATIVES},
                    "candidate_margin_by_negative": {
                        name: candidate_margin for name in self.runtime.NEGATIVES
                    },
                    "raw_reconstruction_by_view": {
                        view: {
                            "candidate_raw_mse": candidate_error,
                            "tucker_b384_raw_mse": 1.0,
                        }
                        for view in self.runtime.EVAL_VIEWS
                    },
                })
                index += 1
        self.assertEqual(index, 644)
        return rows

    def test_train_and_eval_warp_are_pinned_disjoint(self) -> None:
        coordinates = self.runtime._train_warp_coordinate_tensor()
        self.assertEqual(
            self.runtime._tensor_sha(coordinates),
            "e08c6bb31a0767eaed9f81dd9330f06d8fa7db3453e8afc036e2b3fc6b24c137",
        )
        self.assertNotEqual(
            self.runtime.TRAIN_WARP_COORDINATES_SHA256,
            self.runtime.v4a.PINNED_WARP_COORDINATES_SHA256,
        )
        self.assertFalse(torch.equal(coordinates, self.runtime.v4a._warp_coordinate_tensor()))

    def test_architecture_capacity_step0_and_constant_code_no_bypass(self) -> None:
        fit = self._fit()
        model = self.runtime.TuckerInitializedTemporalConvAE(fit, torch.ones(1))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 124672)
        generator = torch.Generator().manual_seed(92)
        value = torch.randn(7, 32, 768, generator=generator)
        value = value - value.mean(dim=1, keepdim=True)
        code = model.encode(value)
        self.assertEqual(tuple(code.shape), (7, 4, 96))
        self.assertEqual(code[0].numel(), 384)
        actual = model(value)
        expected = self.runtime._analytic_tucker_decode(value, fit)
        self.assertTrue(torch.equal(actual, expected))
        evidence = self.runtime._step0_equivalence(model, value, fit, 3)
        self.assertTrue(evidence["bit_exact"])
        fixed_code = torch.randn(1, 4, 96, generator=generator).expand(2, -1, -1).clone()
        decoded = model.decode(fixed_code)
        self.assertTrue(torch.equal(decoded[0], decoded[1]))

    def test_finite_fail_closed_and_uncentered_input_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.runtime.TuckerInitializedTemporalConvAE(
                self._fit(), torch.tensor([float("nan")])
            )
        model = self._model()
        with self.assertRaises(ValueError):
            model.encode(torch.ones(1, 32, 768))

    def test_full_synthetic_exact5_gate_and_latent_never_qualifies(self) -> None:
        config = self.runtime.Config(bootstrap_draws=64)
        metrics = self.runtime._aggregate(self._rows(), config)
        self.assertTrue(metrics["decoded_temporal_codec_development_gate"])
        self.assertFalse(metrics["latent_metric_qualified"])
        self.assertFalse(metrics["latent_gauge_fixed"])

    def test_garbage_fidelity_cannot_be_hidden_by_large_negative_margin(self) -> None:
        config = self.runtime.Config(bootstrap_draws=64)
        metrics = self.runtime._aggregate(
            self._rows(candidate_margin=100.0, candidate_error=100.0), config
        )
        self.assertTrue(metrics["all_three_decoded_negative_gates"])
        self.assertFalse(metrics["five_view_fidelity_gate"])
        self.assertFalse(metrics["decoded_temporal_codec_development_gate"])

    def test_one_garbage_negative_view_cannot_be_hidden_by_large_margin(self) -> None:
        rows = self._rows(candidate_margin=1.0, candidate_error=0.5)
        for row in rows:
            row["candidate_margin_by_negative"]["reverse"] = 100.0
            row["raw_reconstruction_by_view"]["reverse"]["candidate_raw_mse"] = 100.0
        metrics = self.runtime._aggregate(
            rows, self.runtime.Config(bootstrap_draws=64)
        )
        self.assertTrue(metrics["all_three_decoded_negative_gates"])
        self.assertTrue(metrics["negative_results"]["reverse"]["decoded_negative_gate"])
        self.assertFalse(metrics["five_view_raw_reconstruction_ratio_vs_fixed_tucker_b384"]
                         ["reverse"]["both_ucbs_le_1p05"])
        self.assertFalse(metrics["decoded_temporal_codec_development_gate"])

    def test_single_bad_fold_cannot_be_compensated(self) -> None:
        rows = self._rows(candidate_margin=1.0, candidate_error=0.5)
        for row in rows:
            if row["outer_fold"] == 0:
                row["candidate_margin_by_negative"]["reverse"] = 0.4
        metrics = self.runtime._aggregate(
            rows, self.runtime.Config(bootstrap_draws=64)
        )
        reverse = metrics["negative_results"]["reverse"]
        self.assertTrue(reverse["candidate_minus_fixed_tucker_b384_margin"]
                        ["both_lcbs_strictly_gt_zero"])
        self.assertFalse(reverse["all_five_fold_point_improvements_strictly_gt_zero"])
        self.assertFalse(reverse["decoded_negative_gate"])
        self.assertFalse(metrics["decoded_temporal_codec_development_gate"])

    def test_single_bad_fidelity_fold_cannot_be_compensated(self) -> None:
        rows = self._rows(candidate_margin=1.0, candidate_error=0.5)
        for row in rows:
            if row["outer_fold"] == 0:
                row["raw_reconstruction_by_view"]["original"]["candidate_raw_mse"] = 1.20
        metrics = self.runtime._aggregate(
            rows, self.runtime.Config(bootstrap_draws=64)
        )
        original = metrics["five_view_raw_reconstruction_ratio_vs_fixed_tucker_b384"]["original"]
        self.assertFalse(original["all_five_fold_point_ratios_le_1p05"])
        self.assertGreater(original["per_fold_ratio_of_mean_raw_mses"]["0"], 1.05)
        self.assertFalse(metrics["five_view_fidelity_gate"])
        self.assertFalse(metrics["decoded_temporal_codec_development_gate"])

    def test_selected_step0_alias_strictly_fails_improvement(self) -> None:
        metrics = self.runtime._aggregate(
            self._rows(candidate_margin=0.5, candidate_error=1.0),
            self.runtime.Config(bootstrap_draws=64),
        )
        self.assertTrue(metrics["five_view_fidelity_gate"])
        self.assertFalse(metrics["all_three_decoded_negative_gates"])
        self.assertFalse(metrics["decoded_temporal_codec_development_gate"])

    def test_checkpoint_is_create_only_sealed_reloaded_and_rehashed(self) -> None:
        fit = self._fit()
        model = self.runtime.TuckerInitializedTemporalConvAE(fit, torch.ones(1))
        generator = torch.Generator().manual_seed(93)
        anchors = []
        for index in range(2):
            anchor = torch.randn(32, 768, generator=generator)
            anchor = anchor - anchor.mean(dim=0, keepdim=True)
            anchors.append(self._pair(f"iid-{index}", anchor))
        state_sha = self.runtime._state_sha(self.runtime._state_to_cpu(model))
        audit = {
            "minibatch_schedule_sha256": "a" * 64,
            "fit_only_global_rms_sha256": self.runtime._tensor_sha(torch.ones(1)),
            "model_fit_iid_digest": "b" * 64,
            "inner_validation_iid_digest": "c" * 64,
            "selected_step": 0,
            "selected_state_sha256": state_sha,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "fold0.pt"
            artifact = self.runtime._save_selected_checkpoint_create_only(
                path, model, fit, 0, audit, self.runtime.Config(), 0,
                {"implementation_sha256": "d" * 64}, anchors, torch.device("cpu"),
            )
            self.assertEqual(artifact["file_sha256"], self.runtime._file_sha(path))
            self.assertEqual(artifact["model_state_sha256"], state_sha)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
            self.assertEqual(path.stat().st_nlink, 1)
            with self.assertRaises(ValueError):
                self.runtime._save_selected_checkpoint_create_only(
                    path, model, fit, 0, audit, self.runtime.Config(), 0,
                    {}, anchors, torch.device("cpu"),
                )

    def test_checkpoint_rejects_selected_step_or_state_mismatch(self) -> None:
        fit = self._fit()
        model = self.runtime.TuckerInitializedTemporalConvAE(fit, torch.ones(1))
        anchor = torch.randn(32, 768)
        anchor = anchor - anchor.mean(dim=0, keepdim=True)
        rows = [self._pair("iid", anchor)]
        state_sha = self.runtime._state_sha(self.runtime._state_to_cpu(model))
        base = {
            "minibatch_schedule_sha256": "a" * 64,
            "fit_only_global_rms_sha256": self.runtime._tensor_sha(torch.ones(1)),
            "model_fit_iid_digest": "b" * 64,
            "inner_validation_iid_digest": "c" * 64,
            "selected_step": 0,
            "selected_state_sha256": state_sha,
        }
        with tempfile.TemporaryDirectory() as directory:
            first = dict(base)
            first["selected_step"] = 300
            with self.assertRaises(RuntimeError):
                self.runtime._save_selected_checkpoint_create_only(
                    Path(directory).resolve() / "step.pt", model, fit, 0, first,
                    self.runtime.Config(), 0, {}, rows, torch.device("cpu"),
                )
            second = dict(base)
            second["selected_state_sha256"] = "0" * 64
            with self.assertRaises(RuntimeError):
                self.runtime._save_selected_checkpoint_create_only(
                    Path(directory).resolve() / "state.pt", model, fit, 0, second,
                    self.runtime.Config(), 0, {}, rows, torch.device("cpu"),
                )

    def test_json_receipt_is_create_only_sealed_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "receipt.json"
            sha = self.runtime._write_json_create_only(path, {"value": 7})
            self.assertEqual(sha, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
            self.assertEqual(path.stat().st_nlink, 1)
            with self.assertRaises(ValueError):
                self.runtime._write_json_create_only(path, {"value": 8})


if __name__ == "__main__":
    unittest.main()

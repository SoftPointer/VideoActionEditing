from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py"


class VJepa2TemporalCodecV4DStaticTests(unittest.TestCase):
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
        writer = self._function_source("_write_json_create_only")
        self.assertIn("v4c._write_json_create_only", writer)
        saver = self._function_source("_save_selected_checkpoint_create_only")
        self.assertIn('path.open("xb")', saver)
        self.assertIn("os.chmod(path, 0o444)", saver)
        self.assertIn("written = os.fstat(handle.fileno())", saver)
        self.assertIn('binding["physical_identity"]', saver)
        self.assertIn("fresh_reload_output_bit_exact", saver)
        self.assertIn("_verify_checkpoint_artifacts(checkpoint_artifacts)", self.source)

    def test_checkpoint_loader_is_single_fd_strong_and_semantic(self) -> None:
        loader = self._function_source("_load_selected_checkpoint_sealed")
        for required in (
            "path.is_absolute()", "path.is_symlink()", "path.resolve(strict=True)",
            "os.O_NOFOLLOW", "os.open(", "os.fstat(", "path.lstat()",
            "digest_before", "digest_after", 'weights_only=True',
            "metadata_digest", "model_fit_ordered_iids", "_state_sha(state)",
            "physical_identity",
            "semantic_metadata_state_replay_verified",
        ):
            self.assertIn(required, loader)
        self.assertEqual(loader.count("os.open("), 1)
        self.assertNotIn("torch.load(path", loader)

    def test_pins_and_prior_fixed_comparator_are_literal(self) -> None:
        for value in (
            "568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2",
            "f33d72320905aba135a2bb8729782cf5c89e6eee81fe1bd88aa8d24e1b585a86",
            "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973",
            "d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef",
            "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc",
            "895fd7e9267c82477ffc11fbc1a11fdd89b276687d87c8e82e7d85d7cf62b54a",
            "8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9",
            "376a98dc74e30ab80a277c8866028677d56ba894073d195612a0edb0bbd74f17",
            "tucker_b0384_t04_r096",
        ):
            self.assertIn(value, self.source)
        self.assertIn('"called_best_or_winner": False', self.source)
        self.assertIn('"parameter_or_flop_fairness_claimed": False', self.source)

    def test_training_cannot_reach_source_oof_negative_or_eval_warp(self) -> None:
        train = self._function_source("_train_fold_model")
        loss = self._function_source("_fixed_training_loss")
        for forbidden in ("temporal_variants", "NEGATIVES", "exploratory_oof", "warp"):
            self.assertNotIn(forbidden, train)
            self.assertNotIn(forbidden, loss)
        self.assertNotIn("TRAIN_WARP", self.source)
        self.assertNotIn("training_only_monotone_warp", self.source)
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
        self.assertIn("EXACT_TRAINABLE_PARAMETERS = 143360", self.source)
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
            "all_five_fold_point_means_strictly_gt_zero",
            "all_four_quantities_pass_dual_bootstrap_and_every_fold",
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

    def test_inner_split_is_metadata_only_and_exactly_frozen(self) -> None:
        split = self._function_source("_split_fold")
        for forbidden in (".views", "canonical_action", "torch.", ".square", ".strict"):
            self.assertNotIn(forbidden, split)
        for required in (
            "INNER_SPLIT_NAMESPACE", "_inner_hash", "rank % INNER_FOLDS",
            'inner_assignment[row.iid] == 0', '"split_tensor_values_read": False',
            '"split_vjepa_energy_used": False',
        ):
            self.assertIn(required, split)

    def test_real_exact644_inner_split_literals_are_frozen(self) -> None:
        assignment = next(
            node for node in self.tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "FROZEN_INNER_SPLITS"
        )
        actual = ast.literal_eval(assignment.value)
        expected = (
            (0, 400, 113, 131, 28, 28, 27, ("exit",),
             "b2e3143700ecf10ff54416395267b7cc3c90f33c7acedd402ede8062f374635a",
             "40ce2072cdbc2cded22bd99cb916897011b066fba48d7ebc389bef8efb67dd18",
             "b98b67342049c45898c055546ee9f49bde70c6169e49db7255e5d5f0d03c02aa",
             "4826c85125572c150284d2bfa593848e380e2d907152701b3df9a60148f927d7"),
            (1, 402, 115, 127, 28, 28, 26, ("climb", "exit"),
             "d9431d202451a3ce99d3b7be67806918dc5ed812259c0b06a4deab0ebf7f2a6f",
             "4e45aac2efcb9a2327586860661b1c26be2cd643766307e768c11ba948ba2cd0",
             "e306b191378bf5eff9b3734080c3be036000c03921c3c03ce127ae5733b5e873",
             "ee126c414fdb2e95d4b56f8876e590fbdfbd54bf31a7646a6c88fd3f7f9c8bf4"),
            (2, 401, 115, 128, 28, 28, 26, ("climb", "exit"),
             "10302330a6a4feddece521d12b3e86efac92bb8e5eb5151b631113aeda069f5c",
             "8321d846bb3f98405251580f1088cd6071f3820fcdf0bbe9ac858dc1a2aa7b78",
             "110db33e61e02cccf95424937fa718ab21dee688934b345425fda2bf7fcc5102",
             "608f19e7e05db3202513a23ad5be27c7d78a4a5d9f828abda7da7c9dca77a65a"),
            (3, 403, 112, 129, 27, 27, 27, (),
             "9f5ee3fce90bca584af36b761ed7a9a2d975d8c10270b1ee18adc3b70b42692f",
             "90da16b4d97de006e16fc522d77061b79f68aefaea3599a6a2a4a28659988353",
             "5517ded71818723d464e99ca380bc6cfb34a34f576d2d564235a1d7f3a5c10e4",
             "4235825a4031a247d2bfbba0552596bf7587142be324f2081be77f78bc269997"),
            (4, 403, 112, 129, 28, 28, 27, ("exit",),
             "6b2502ef34eaf4bd81e1abcda313accb13306adb1b1f97f38aea84e97bf1760a",
             "bcd16de76199767e77a889d44b504b96db1703b980facf08b26b50b09730283c",
             "45f7f48fd625941494f88860acc2ed9b81c7035f00950e8304070f7faf919a32",
             "7c7e93a17afec29032b8b7e6948184a43796bb1e8fe680c8591266f3cedab9e8"),
        )
        normalized = tuple(
            (
                item["outer_fold"], item["counts"]["model_fit"],
                item["counts"]["inner_validation"],
                item["counts"]["exploratory_oof"],
                item["outer_train_family_count"],
                item["model_fit_family_count"],
                item["inner_validation_family_count"],
                tuple(item["singleton_families"]),
                item["inner_assignment_digest"], item["model_fit_iid_digest"],
                item["inner_validation_iid_digest"],
                item["partition_iid_digest"],
            )
            for item in actual
        )
        self.assertEqual(normalized, expected)
        split = self._function_source("_split_fold")
        self.assertIn("literal != FROZEN_INNER_SPLITS[outer_fold]", split)
        self.assertIn('"frozen_real_exact644_literal_match"] = True', split)

    def test_fit_reads_only_original_and_eval_reads_stored_five_views(self) -> None:
        fit = self._function_source("_fit_tucker_b384")
        self.assertIn('row.views["original"]', fit)
        for name in ("monotone_warp", "reverse", "block_shuffle", "phase_swap"):
            self.assertNotIn(name, fit)
        evaluate = self._function_source("_evaluate_fold")
        self.assertIn("row.views[name]", evaluate)
        self.assertNotIn("temporal_variants", evaluate)
        self.assertNotIn("index_select", evaluate)
        for forbidden in (
            "tucker_codes", "tucker_code_margin",
            "decoded_vs_code_differences",
            "tucker_b384_code_margin_by_negative",
            "decoded_tucker_vs_code_margin_abs_diff_by_negative",
        ):
            self.assertNotIn(forbidden, evaluate)
        upstream = self._function_source("_verify_v4c_embedded_teacher_evidence")
        self.assertIn("teacher_difference", upstream)
        self.assertNotIn("decoded_difference", upstream)
        self.assertNotIn("max_decoded_code", upstream)

    def test_receipt_postwrite_rechecks_every_input_authority(self) -> None:
        run = self._function_source("run_exact5")
        self.assertIn("output.is_symlink()", run)
        writer = run.rfind("receipt_sha = _write_json_create_only")
        self.assertGreater(writer, 0)
        tail = run[writer:]
        self.assertIn("_verify_checkpoint_artifacts(checkpoint_artifacts)", tail)
        self.assertIn("v4c._assert_input_files_unchanged", tail)
        self.assertIn("_load_v4c_frontier_receipt", tail)
        self.assertIn(
            '"feature_receipt_six_shards_v4a_v4c_reverified_after_receipt_write": True',
            tail,
        )

    def test_receipt_scope_comparator_and_label_contract_are_fail_closed(self) -> None:
        for required in (
            '"v4c_oof_was_burned_before_v4d": True',
            '"clip_pca_b384_was_descriptively_higher_in_v4c": True',
            '"clip_pca_used_to_select_tucker_rank_or_mapping": False',
            '"scientific_confirmation_claimed": False',
            '"identity_disentanglement_qualified": False',
            '"prior_generation_qualified": False',
            '"family_or_transform_labels_enter_loss_or_model_input": False',
            '"family_metadata_used_for_inner_split": True',
            '"transform_metadata_used_for_inner_split": False',
            '"descriptive_scope": {',
            '"model_fit_ordered_iids": fit_iids',
        ):
            self.assertIn(required, self.source)
        qualification = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Dict)
            and any(isinstance(key, ast.Constant)
                    and key.value == "temporal_codec_development_gate"
                    for key in node.keys)
            and any(isinstance(key, ast.Constant)
                    and key.value == "scientific_confirmation_claimed"
                    for key in node.keys)
        )
        keys = [key.value for key in qualification.keys if isinstance(key, ast.Constant)]
        self.assertNotIn("fold_local_model_fit_performed", keys)
        self.assertNotIn("fresh_confirmation_requires_new_external_group_disjoint_data", keys)

    def test_no_duplicate_literal_dictionary_keys(self) -> None:
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            self.assertEqual(len(keys), len(set(keys)), f"duplicate keys at line {node.lineno}")

    def test_parser_requires_all_three_sealed_authorities(self) -> None:
        parser = self._function_source("build_parser")
        for option in (
            "--expected-feature-receipt-sha256", "--v4a-receipt",
            "--expected-v4a-receipt-sha256", "--v4c-frontier-receipt",
            "--expected-v4c-frontier-receipt-sha256",
        ):
            self.assertIn(option, parser)
        self.assertIn('"feature_receipt_path": str(', self.source)
        self.assertIn('(feature_root / "feature_extraction_receipt.json").resolve(strict=True)',
                      self.source)

    def test_no_stale_dino_v4b_or_post_token_runtime_surface(self) -> None:
        for forbidden in (
            "authority.load_exact644_pairs", "v2._split_fold", "chosen_before_v4b",
            "[32,768]", "V4B_FAST", 'f"v4b:', "TRAIN_WARP_COORDINATES",
            "training_only_monotone_warp", "temporal_variants", "anchor_sequence",
        ):
            self.assertNotIn(forbidden, self.source)


@unittest.skipIf(torch is None, "torch is unavailable")
class VJepa2TemporalCodecV4DDynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if str(METHOD_ROOT.parent.parent) not in sys.path:
            sys.path.insert(0, str(METHOD_ROOT.parent.parent))
        from methods.bernini_action_editing import semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d
        cls.runtime = semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d

    def _fit(self):
        g = torch.Generator().manual_seed(91)
        temporal = torch.linalg.qr(torch.randn(32, 4, generator=g)).Q
        content = torch.linalg.qr(torch.randn(1024, 96, generator=g)).Q
        return self.runtime.TuckerFit(
            frame_mean=torch.randn(1, 1024, generator=g) * 0.01,
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
        return self.runtime.v4c.Record(
            iid=iid,
            family="family-a",
            strict=True,
            views={"original": anchor},
        )

    def _checkpoint_audit(self, model, iids=("fit-iid-0", "fit-iid-1")):
        state_sha = self.runtime._state_sha(self.runtime._state_to_cpu(model))
        ordered = list(iids)
        return {
            "minibatch_schedule_sha256": "a" * 64,
            "fit_only_global_rms_sha256": self.runtime._tensor_sha(torch.ones(1)),
            "model_fit_original_count": len(ordered),
            "model_fit_ordered_iids": ordered,
            "model_fit_iid_digest": self.runtime._object_sha(ordered),
            "inner_validation_iid_digest": "c" * 64,
            "selected_step": 0,
            "selected_state_sha256": state_sha,
        }

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

    def test_no_training_warp_surface_exists(self) -> None:
        self.assertFalse(hasattr(self.runtime, "TRAIN_WARP_COORDINATES"))
        self.assertFalse(hasattr(self.runtime, "training_only_monotone_warp"))

    def test_metadata_only_inner_split_has_exact_frozen_fixture(self) -> None:
        records = [
            self.runtime.v4c.Record(
                iid=f"iid-{index:04d}", family=f"family-{index % 28:02d}",
                strict=True, views={},
            )
            for index in range(644)
        ]
        outer_counts = self.runtime.FROZEN_OOF_COUNTS
        outer_assignment = {}
        start = 0
        for fold, count in enumerate(outer_counts):
            for row in records[start:start + count]:
                outer_assignment[row.iid] = fold
            start += count
        expected = (
            (401, 112, "16dc555d803c141055da5444e393ab8a16125c41dc98dc3c8c84423d13b7964c"),
            (405, 112, "43a65fd1dc766c97c10c5452ec9e2743810b082311987776849e1f1045a2e7a3"),
            (404, 112, "1edfc0059ee80bca9d57d41e7083400762cdb4c9be77bb971ccdd544851b8e15"),
            (403, 112, "88aca034a70777ee4eaa5a30e3ea434ed29ec3b9784a8560399e79f5def2f276"),
            (403, 112, "f349cee34efce840ec6caff26f471f1b00970c58f25d4019eee1e13005d4ec66"),
        )
        for fold in range(5):
            groups, split = self.runtime._split_fold(
                records, outer_assignment, fold, self.runtime.Config()
            )
            self.assertEqual(len(groups["model_fit"]), expected[fold][0])
            self.assertEqual(len(groups["inner_validation"]), expected[fold][1])
            self.assertEqual(split["inner_assignment_digest"], expected[fold][2])
            self.assertFalse(split["split_tensor_values_read"])
            self.assertFalse(split["split_vjepa_energy_used"])

    def test_architecture_capacity_step0_and_constant_code_no_bypass(self) -> None:
        fit = self._fit()
        model = self.runtime.TuckerInitializedTemporalConvAE(fit, torch.ones(1))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 143360)
        generator = torch.Generator().manual_seed(92)
        value = torch.randn(7, 32, 1024, generator=generator)
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
            model.encode(torch.ones(1, 32, 1024))

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
        self.assertFalse(
            reverse["candidate_minus_fixed_tucker_b384_margin"]
            ["all_five_fold_point_means_strictly_gt_zero"]
        )
        self.assertFalse(reverse["decoded_negative_gate"])
        self.assertFalse(metrics["decoded_temporal_codec_development_gate"])

    def test_teacher_candidate_and_retention_each_require_every_fold(self) -> None:
        cases = (
            ("teacher_margin_by_negative", -0.01, "teacher_margin"),
            ("candidate_margin_by_negative", -0.01, "candidate_margin"),
            ("teacher_margin_by_negative", 2.0,
             "candidate_minus_0p8_teacher_margin"),
        )
        for row_field, bad_value, metric_field in cases:
            rows = self._rows(candidate_margin=1.0, candidate_error=0.5)
            for row in rows:
                if row["outer_fold"] == 0:
                    row[row_field]["reverse"] = bad_value
            metrics = self.runtime._aggregate(
                rows, self.runtime.Config(bootstrap_draws=64)
            )
            reverse = metrics["negative_results"]["reverse"]
            self.assertFalse(
                reverse[metric_field]
                ["all_five_fold_point_means_strictly_gt_zero"]
            )
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
            anchor = torch.randn(32, 1024, generator=generator)
            anchor = anchor - anchor.mean(dim=0, keepdim=True)
            anchors.append(self._pair(f"iid-{index}", anchor))
        audit = self._checkpoint_audit(model)
        state_sha = audit["selected_state_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "fold0.pt"
            artifact = self.runtime._save_selected_checkpoint_create_only(
                path, model, fit, 0, audit, self.runtime.Config(), 0,
                {"implementation_sha256": "d" * 64}, anchors, torch.device("cpu"),
            )
            self.assertEqual(artifact["file_sha256"], self.runtime._file_sha(path))
            self.assertEqual(artifact["model_state_sha256"], state_sha)
            self.assertTrue(artifact["single_fd_pre_post_sha256_exact"])
            self.assertTrue(artifact["semantic_metadata_state_replay_verified"])
            self.assertEqual(artifact["physical_identity"]["inode"], path.stat().st_ino)
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
        anchor = torch.randn(32, 1024)
        anchor = anchor - anchor.mean(dim=0, keepdim=True)
        rows = [self._pair("iid", anchor)]
        base = self._checkpoint_audit(model, ("fit-iid",))
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

    def test_checkpoint_loader_rejects_symlink_and_path_swap(self) -> None:
        fit = self._fit()
        model = self.runtime.TuckerInitializedTemporalConvAE(fit, torch.ones(1))
        anchor = torch.randn(32, 1024)
        anchor = anchor - anchor.mean(dim=0, keepdim=True)
        rows = [self._pair("iid", anchor)]
        audit = self._checkpoint_audit(model, ("fit-iid",))
        binding = {"implementation_sha256": "d" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            symlink_path = root / "symlink.pt"
            artifact = self.runtime._save_selected_checkpoint_create_only(
                symlink_path, model, fit, 0, audit, self.runtime.Config(), 0,
                binding, rows, torch.device("cpu"),
            )
            target = root / "symlink-target.pt"
            symlink_path.rename(target)
            symlink_path.symlink_to(target)
            with self.assertRaises(ValueError):
                self.runtime._load_selected_checkpoint_sealed(symlink_path, artifact)

            swap_path = root / "swap.pt"
            swap_artifact = self.runtime._save_selected_checkpoint_create_only(
                swap_path, model, fit, 0, audit, self.runtime.Config(), 0,
                binding, rows, torch.device("cpu"),
            )
            original_load = self.runtime.torch.load
            swapped_original = root / "swap-original.pt"

            def swap_during_load(handle, *args, **kwargs):
                swap_path.rename(swapped_original)
                shutil.copyfile(swapped_original, swap_path)
                os.chmod(swap_path, 0o444)
                return original_load(handle, *args, **kwargs)

            with mock.patch.object(
                self.runtime.torch, "load", side_effect=swap_during_load
            ):
                with self.assertRaises(RuntimeError):
                    self.runtime._load_selected_checkpoint_sealed(
                        swap_path, swap_artifact
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

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "semantic_anchor_vjepa2_evaluation_aligned_primal_dual_v4h.py"
V4F_PATH = (
    METHOD_ROOT
    / "semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy.py"
)


class EvaluationAlignedPrimalDualV4HStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNTIME_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def function_source(self, name: str) -> str:
        nodes = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        self.assertEqual(len(nodes), 1, name)
        return ast.get_source_segment(self.source, nodes[0]) or ""

    def class_source(self, name: str) -> str:
        nodes = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef) and node.name == name
        ]
        self.assertEqual(len(nodes), 1, name)
        return ast.get_source_segment(self.source, nodes[0]) or ""

    def test_source_compiles_normal_and_optimized(self) -> None:
        compile(self.source, str(RUNTIME_PATH), "exec")
        compile(self.source, str(RUNTIME_PATH), "exec", optimize=2)

    def test_independent_path_and_frozen_final_v4f_dependency(self) -> None:
        self.assertNotEqual(RUNTIME_PATH, V4F_PATH)
        self.assertTrue(V4F_PATH.is_file())
        self.assertIn(
            'V4F_RUNTIME_DEPENDENCY_SHA256 = "PLACEHOLDER_',
            self.source,
        )
        for pin in (
            "ff5a32ae4047f3408da9954711c4e76069eb06edc60f10078833ba961812fcab",
            "8971b2910262b85948670b5531ad0df30cf8336280342dc8c9245e1110d95a2d",
            "14bf42749c97b20934a2a088a560fa23ed2b1e37555262e9d4c7f2f368e74265",
            "45753aef53fc1a69b46154990b36196be5c2d2e0b9aaf8e60c285b59c9f00cfc",
            "6f501749811e2c57a806bb9cebdd895cfbed285cf64e0ac41eafb484e3042380",
            "96b3638388a21752f5a7ed4a4225d83589df0f0487b948d1a5bc34286cb867b3",
            "5b714ce73c5178f9043e837374487b69e162613f98587194a846b0b9276f8249",
            "affeef9ec5023437a379f00ac424d117a29db796cc3afeb9f086cc83eab1c2fb",
            "a48e4f13f9b85360c5306e274d59cf672549da8adb01aaa452670eeb9807590a",
        ):
            self.assertIn(pin, self.source)
        binding = self.function_source("_binding")
        self.assertIn("dependency_sha != V4F_RUNTIME_DEPENDENCY_SHA256", binding)
        self.assertIn("frozen._binding()", binding)

    def test_unreleased_one_way_release_dag_and_guard_first(self) -> None:
        self.assertIn("RELEASE_SEALED = False", self.source)
        self.assertNotIn("EXPECTED_V4H_RUNTIME_SHA256", self.source)
        self.assertNotIn("EXPECTED_V4H_TEST_SHA256", self.source)
        self.assertNotIn("EXPECTED_V4H_CONTROLLER_SHA256", self.source)
        self.assertNotIn("EXPECTED_V4H_RELEASE_MANIFEST_SHA256", self.source)
        self.assertIn("never reverse-pins those authorities", self.source)
        for name in (
            "run_train_fold", "run_verify_inner_barrier", "run_evaluate_fold",
            "run_aggregate",
            "build_parser", "main",
        ):
            node = next(
                node for node in ast.walk(self.tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            offset = int(
                isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            )
            first = node.body[offset]
            self.assertIsInstance(first, ast.Expr, name)
            self.assertIsInstance(first.value, ast.Call, name)
            self.assertEqual(getattr(first.value.func, "id", None), "_require_release_sealed")

    def test_one_fixed_candidate_and_no_grid_selection(self) -> None:
        for token in (
            "FIXED_SELECTED_STEP = 1200",
            "FIXED_RESIDUAL_SCALE = 1.0",
            "FIXED_CANDIDATE_COUNT = 1",
            '"single_candidate": True',
            '"hyperparameter_selection_performed": False',
        ):
            self.assertIn(token, self.source)
        for forbidden in (
            "RHO_GRID", "RHO_COMPARATOR", "_select_fold_local_rho",
            "selected_rho", "rho_ordinal", "exact7_candidate",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_model_payload_parameter_count_and_fixed_scale(self) -> None:
        model = self.class_source("ClipPCAInitializedVJepaGlobalCodec")
        for token in (
            "CODE_TIME = frozen.CODE_TIME",
            "CODE_CHANNELS = frozen.CODE_CHANNELS",
            "EXACT_TRAINABLE_PARAMETERS = frozen.EXACT_TRAINABLE_PARAMETERS",
        ):
            self.assertIn(token, self.source)
        self.assertIn("count != EXACT_TRAINABLE_PARAMETERS", model)
        self.assertEqual(model.count("FIXED_RESIDUAL_SCALE"), 2)
        self.assertIn("decoder input must be the sole [12,32] code", model)
        self.assertNotIn("set_residual", model)
        self.assertNotIn("raw_input", model)
        self.assertNotIn("MultiheadAttention", model)

    def test_evaluation_aligned_primal_dual_loss_is_exact(self) -> None:
        loss = self.function_source("_evaluation_aligned_primal_dual_loss")
        for token in (
            "candidate_mse.sum(dim=0) / fidelity_denominators",
            '"raw ratio-of-means baseline denominator must be finite and >0"',
            "fidelity_ratios.mean()",
            "fidelity_ratios - config.fidelity_constraint_limit",
            "for left in range(len(EVAL_VIEWS))",
            "for right in range(left + 1, len(EVAL_VIEWS))",
            "len(teacher_pair_distances) != 10",
            "+ config.teacher_pair_scale_epsilon",
            'role_index["original"]',
            'role_index["monotone_warp"]',
            "for negative in NEGATIVES",
            "config.retention_constraint_target * teacher_margins.mean(dim=0)",
            "- candidate_margins.mean(dim=0)",
            ") / teacher_scale_mean",
            "duals[index].detach() * torch.relu(constraints[index])",
        ):
            self.assertIn(token, loss)
        self.assertNotIn("family", loss)
        self.assertNotIn("smooth_l1", loss)
        self.assertNotIn("+ 1.0e-8", loss)

    def test_train_uses_cpu_float64_duals_and_fixed_budget(self) -> None:
        train = self.function_source("_train_fold_model")
        for token in (
            "config.max_steps + 1",
            "torch.optim.AdamW",
            "config.learning_rate",
            "config.weight_decay",
            "config.batch_size",
            "fit_views.flatten(0, 1)",
            "frozen._analytic_clip_pca_decode(",
            "_evaluation_aligned_primal_dual_loss(",
            'dtype=TRACE_DTYPE, device="cpu"',
            "primal_duals = duals.to(device=device, dtype=target.dtype)",
            'constraints.detach().to(\n            dtype=TRACE_DTYPE, device="cpu"',
            '"checkpoint_winner_selection_performed": False',
            '"model_fit_five_view_tensors_used_for_gradient_and_model_input": True',
            '"model_fit_transform_roles_used_for_gradient_loss": True',
            '"model_fit_transform_role_labels_used_as_auxiliary_model_inputs": False',
            '"model_fit_family_metadata_used_for_gradient_or_model_input": False',
        ):
            self.assertIn(token, train)
        self.assertNotIn('"model_fit_transform_roles_used_for_model_input"', train)
        self.assertNotIn('"transform_roles_used_for_model_input"', self.source)
        for token in (
            '"inner_five_view_tensors_used_as_model_inputs_for_gate_evaluation": True',
            '"inner_five_view_tensors_used_for_gradient_or_candidate_checkpoint_selection": False',
            '"all_five_known_view_tensors_used_as_model_inputs": True',
            '"transform_role_labels_used_as_auxiliary_model_inputs": False',
        ):
            self.assertIn(token, self.source)
        self.assertIn("inner_validation_iids: Sequence[str]", train)
        self.assertNotIn("inner_validation: Sequence[v4c.Record]", train)
        self.assertNotIn("checkpoint_scores", train)
        self.assertLess(train.index("optimizer.step()"), train.index(
            "duals = torch.clamp_min("
        ))
        self.assertIn("constraint_trace", train)
        self.assertIn("dual_trace", train)
        self.assertIn("objective_trace", train)

    def test_train_fold_has_absolute_oof_read_barrier(self) -> None:
        train = self.function_source("_train_inner_fold")
        positions = [
            train.index('stage="stage1_model_fit_only"'),
            train.index("_save_primal_dual_trace_create_only("),
            train.index('checkpoint_role="preselection_fixed_step1200"'),
            train.index('checkpoint_role="fixed1200_candidate"'),
            train.index('stage="stage2_post_preselection_seal_inner_five_views"'),
            train.index("_evaluate_fixed_inner_candidate("),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn('stage="stage3_post_selected_seal_oof"', train)
        self.assertNotIn("_evaluate_rows_fixed(\n        [oof", train)
        self.assertIn('"oof_semantic_tensor_read_count_exact0": True', train)

    def test_exact4_float64_trace_is_strongly_sealed_and_fully_threaded(self) -> None:
        save = self.function_source("_save_primal_dual_trace_create_only")
        load = self.function_source("_load_primal_dual_trace_sealed")
        for token in (
            '"metadata", "constraint_trace", "dual_trace", "objective_trace"',
            "set(payload) !=",
            "set(metadata) != TRACE_METADATA_KEYS",
            "tensor.dtype != TRACE_DTYPE",
            "CONSTRAINT_TRACE_SHAPE",
            "DUAL_TRACE_SHAPE",
            "OBJECTIVE_TRACE_SHAPE",
            "torch.equal(dual_trace[1:], replay_dual)",
            "objective trace exact replay differs",
            "expected binding differs before torch parse",
        ):
            self.assertIn(token, load)
        for token in (
            "os.O_CREAT | os.O_EXCL | os.O_RDWR",
            "os.fchmod(handle.fileno(), 0o444)",
            "creation_sha",
            'binding["physical_identity"] != creation_physical',
        ):
            self.assertIn(token, save)
        self.assertGreaterEqual(self.source.count('"primal_dual_trace.pt"'), 3)
        for function in (
            "_load_checkpoint_sealed", "_checkpoint_expectations_from_inner_receipt",
            "_load_inner_receipt_sealed", "_barrier_members",
            "_load_fold_receipt_sealed", "run_aggregate",
        ):
            self.assertIn("primal_dual_trace", self.function_source(function))

    def test_global_barrier_has_no_tensor_loader(self) -> None:
        barrier = self.function_source("_load_all_inner_receipts_or_fail_before_oof")
        self.assertIn("len(fold_roots) != OUTER_FOLDS", barrier)
        self.assertIn("list(range(OUTER_FOLDS))", barrier)
        self.assertIn('row.get("inner_pass") is not True', barrier)
        self.assertIn("GLOBAL_INNER_NO_GO", barrier)
        self.assertNotIn("_selective_materialize_feature_rows", barrier)
        self.assertNotIn("_evaluate_rows_fixed", barrier)

    def test_fixed_candidate_ledger_recomputes_gate_and_seed_ledger(self) -> None:
        replay = self.function_source("_verify_fixed_candidate_ledger")
        inner_gate = self.function_source("_inner_candidate_gate")
        self.assertIn("namespace=INNER_BOOTSTRAP_NAMESPACE", inner_gate)
        self.assertIn('INNER_BOOTSTRAP_NAMESPACE = "v4h"', self.source)
        self.assertIn(
            '"inner_bootstrap_seed_ledger_reused_from_v4f": False',
            self.source,
        )
        for token in (
            "_inner_candidate_gate(",
            'candidate.get("bootstrap_seed_ledger")',
            "_bootstrap_seed_ledger(recomputed_gate)",
            'candidate.get("pass") is not passed',
            'receipt.get("inner_pass") is not passed',
            'receipt.get("fixed1200_checkpoint_artifact", {}).get(',
        ):
            self.assertIn(token, replay)

    def test_barrier_and_aggregate_require_expected_child_hashes(self) -> None:
        barrier = self.function_source("_load_all_inner_receipts_or_fail_before_oof")
        self.assertIn("len(expected_sha256) != OUTER_FOLDS", barrier)
        self.assertIn("_load_inner_receipt_sealed(root, expected, run_binding)", barrier)
        parser = self.function_source("build_parser")
        self.assertEqual(parser.count('"--expected-inner-receipt-sha256"'), 1)
        self.assertEqual(parser.count('"--expected-barrier-receipt-sha256"'), 2)
        self.assertEqual(parser.count('"--expected-fold-receipt-sha256"'), 1)
        fold_loader = self.function_source("_load_fold_receipt_sealed")
        self.assertIn("file_sha != expected_sha256", fold_loader)
        self.assertIn(
            'barrier.get("controller_barrier_receipt_binding")',
            fold_loader,
        )
        self.assertIn("!= barrier_receipt_binding", fold_loader)
        self.assertIn("!= barrier_receipt_digest", fold_loader)

    def test_evaluate_places_global_barrier_before_oof_request(self) -> None:
        evaluate = self.function_source("run_evaluate_fold")
        sealed = evaluate.index("_load_barrier_receipt_sealed(")
        barrier = evaluate.index("_load_all_inner_receipts_or_fail_before_oof(")
        materialize = evaluate.index("frozen._selective_materialize_feature_rows(")
        score = evaluate.index("_evaluate_rows_fixed(")
        self.assertLess(sealed, barrier)
        self.assertLess(barrier, materialize)
        self.assertLess(materialize, score)
        prefix = evaluate[:barrier]
        self.assertNotIn("_selective_materialize_feature_rows", prefix)
        self.assertNotIn("_evaluate_rows_fixed", prefix)
        authority_replay = evaluate.index(
            "_independently_replay_all_inner_gates_before_oof("
        )
        self.assertLess(barrier, authority_replay)
        self.assertLess(authority_replay, materialize)

    def test_independent_replay_reexecutes_all_science_before_oof(self) -> None:
        replay = self.function_source(
            "_independently_replay_all_inner_gates_before_oof"
        )
        for token in (
            "for fold_index, (receipt, receipt_binding) in enumerate(",
            "_recompute_model_fit_provenance_from_authority(",
            "_load_primal_dual_trace_sealed(",
            "_load_checkpoint_sealed(",
            "frozen._selective_materialize_feature_rows(",
            'stage="stage2_post_preselection_seal_inner_five_views"',
            "_evaluate_fixed_inner_candidate(",
            'replay != receipt.get("fixed_candidate")',
            'replay.get("inner_pass") is not True',
            '"oof_semantic_tensor_read_count": 0',
        ):
            self.assertIn(token, replay)
        self.assertNotIn('stage="stage3_post_selected_seal_oof"', replay)

    def test_inner_receipt_records_exact_zero_oof(self) -> None:
        train = self.function_source("_train_inner_fold")
        for token in (
            '"candidate_count": FIXED_CANDIDATE_COUNT',
            '"single_candidate": True',
            '"hyperparameter_selection_performed": False',
            '"oof_used_for_training_checkpoint_or_inner_gate": False',
            '"oof_semantic_tensor_materialized_count": 0',
            '"oof_semantic_tensor_read_count_exact0": True',
            '"global_barrier_required_before_any_fold_oof": True',
        ):
            self.assertIn(token, train)

    def test_checkpoints_are_two_distinct_strongly_reloaded_files(self) -> None:
        save = self.function_source("_save_checkpoint_create_only")
        load = self.function_source("_load_checkpoint_sealed")
        pair = self.function_source("_verify_distinct_checkpoint_pair")
        for token in (
            "os.O_NOFOLLOW", "expected binding differs before torch parse",
            "weights_only=True", "digest_after", "metadata_digest",
            "template.load_state_dict(state, strict=True)",
        ):
            self.assertIn(token, load)
        self.assertLess(
            load.index("expected binding differs before torch parse"),
            load.index("torch.load("),
        )
        self.assertIn('checkpoint_role == "fixed1200_candidate"', save)
        self.assertIn("path.open(\"xb\")", save)
        self.assertIn("os.chmod(path, 0o444)", save)
        self.assertIn('preselection.get("path") == fixed.get("path")', pair)
        self.assertIn('pre_physical.get("inode")', pair)
        self.assertIn('fixed_physical.get("inode")', pair)
        self.assertIn('fixed.get("preselection_checkpoint_binding")', pair)
        self.assertIn('_object_sha(fixed.get("preselection_checkpoint_binding"))', pair)
        self.assertIn('"distinct_device_inode_pair": True', pair)

    def test_aggregate_replays_barrier_before_evidence_union(self) -> None:
        aggregate = self.function_source("run_aggregate")
        self.assertLess(
            aggregate.index("_load_all_inner_receipts_or_fail_before_oof("),
            aggregate.index('evidence = [row for fold in folds'),
        )
        self.assertIn("_verify_fold_inner_oof_join(fold, inner)", aggregate)
        self.assertIn("frozen._aggregate(evidence, config)", aggregate)
        self.assertIn(
            '"final_oof_thresholds_and_seed_namespace_exactly_reused_from_v4f": True',
            aggregate,
        )
        self.assertNotIn('"final_oof_thresholds_unchanged_from_v4e"', aggregate)
        self.assertIn('"inference_authorized": False', aggregate)

    def test_checkpoint_provenance_is_authority_bound_before_forward(self) -> None:
        loader = self.function_source("_load_checkpoint_sealed")
        helper = self.function_source("_checkpoint_expectations_from_inner_receipt")
        replay = self.function_source("_independently_replay_all_inner_gates_before_oof")
        for token in (
            '"outer_fold"', '"model_fit_ordered_iids"',
            '"model_fit_original_count"', '"model_fit_iid_digest"',
            '"inner_validation_iid_digest"',
            '"fixed_clip_pca_fit_input_sha256"',
            '"minibatch_schedule_sha256"', '"runtime_fingerprint"',
            '"primal_dual_trace_artifact"',
            '"fixed_clip_pca_b384_all_five_views_tensor_sha256"',
        ):
            self.assertIn(token, loader)
            self.assertIn(token, helper)
        self.assertIn("expected_runtime_fingerprint=replay_runtime", replay)
        for token in (
            '"checkpoint_outer_fold_authority_join"',
            '"checkpoint_model_fit_ordered_iids_authority_join"',
            '"checkpoint_inner_iid_digest_authority_join"',
            '"checkpoint_pca_fit_input_receipt_training_join"',
            '"checkpoint_minibatch_schedule_receipt_training_join"',
            '"checkpoint_state_receipt_training_inner_join"',
            '"primal_dual_trace_full_binding_receipt_checkpoint_join"',
            '"primal_dual_trace_b384_authority_join"',
            '"training_evaluate_runtime_fingerprint_exact_match"',
        ):
            self.assertIn(token, replay)

    def test_train_and_evaluate_configure_determinism_before_replay(self) -> None:
        for name in (
            "run_train_fold", "run_verify_inner_barrier", "run_evaluate_fold",
        ):
            source = self.function_source(name)
            self.assertIn("torch.set_num_threads(1)", source)
            self.assertIn("_seed_everything(", source)
            self.assertLess(
                source.index("torch.set_num_threads(1)"),
                source.index("frozen._prepare_authorities(args)"),
            )
            self.assertLess(
                source.index("_seed_everything("),
                source.index("frozen._prepare_authorities(args)"),
            )
        evaluate = self.function_source("run_evaluate_fold")
        self.assertLess(
            evaluate.index("_seed_everything("),
            evaluate.index("_load_all_inner_receipts_or_fail_before_oof("),
        )

    def test_qualification_scope_has_complete_false_null_surface(self) -> None:
        scope = self.function_source("_qualification_scope")
        for token in (
            '"identity_disentanglement_qualified": False',
            '"identity_preservation_qualified": False',
            '"vae_necessary": None',
            '"prior_qualified": False',
            '"prior_generation_qualified": False',
            '"generation_qualified": False',
            '"renderer_qualified": False',
            '"video_editing_qualified": False',
            '"inference_authorized": False',
            '"web_evaluation_authorized": False',
            '"full644_refit_authorized": False',
            '"video_model_training_performed": False',
        ):
            self.assertEqual(scope.count(token), 1, token)
        dictionary = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_qualification_scope"
        ).body[1].value
        self.assertIsInstance(dictionary, ast.Dict)
        keys = [key.value for key in dictionary.keys if isinstance(key, ast.Constant)]
        self.assertEqual(len(keys), len(set(keys)))

    def test_cli_has_train_barrier_evaluate_aggregate_only(self) -> None:
        self.assertEqual(self.source.count('add_parser("train-fold")'), 1)
        self.assertEqual(self.source.count('add_parser("verify-inner-barrier")'), 1)
        self.assertEqual(self.source.count('add_parser("evaluate-fold")'), 1)
        self.assertEqual(self.source.count('add_parser("aggregate")'), 1)
        self.assertEqual(self.source.count("add_parser("), 4)

    def test_controller_bound_create_only_barrier_is_the_only_evaluate_input(self) -> None:
        verify = self.function_source("run_verify_inner_barrier")
        evaluate = self.function_source("run_evaluate_fold")
        loader = self.function_source("_load_barrier_receipt_sealed")
        parser = self.function_source("build_parser")
        for token in (
            "_load_all_inner_receipts_or_fail_before_oof(",
            "_independently_replay_all_inner_gates_before_oof(",
            '"oof_semantic_tensor_read_count": 0',
            "_write_json_create_only(output, receipt)",
            '"causal_training_trust_boundary"',
        ):
            self.assertIn(token, verify)
        self.assertIn("_load_barrier_receipt_sealed(", evaluate)
        self.assertNotIn("args.fold_root", evaluate)
        self.assertNotIn("args.expected_inner_receipt_sha256", evaluate)
        self.assertIn("file_sha != expected_sha256", loader)
        self.assertIn("_barrier_replay_semantically_complete(", loader)
        self.assertIn('"--barrier-receipt"', parser)
        self.assertIn('"--expected-barrier-receipt-sha256"', parser)

    def test_barrier_explicitly_binds_model_fit_and_inner_replays(self) -> None:
        members = self.function_source("_barrier_members")
        replay = self.function_source(
            "_independently_replay_all_inner_gates_before_oof"
        )
        semantic = self.function_source("_barrier_replay_semantically_complete")
        for token in (
            '"independent_model_fit_provenance_replay_sha256"',
            '"independent_inner_replay_sha256"',
            '"primal_dual_trace_artifact"',
        ):
            self.assertIn(token, members)
        self.assertIn('"inner_replay_binding"', replay)
        self.assertIn('"inner_replay_sha256"', replay)
        for token in (
            '"checkpoint_model_fit_count_and_digest_authority_join"',
            '"checkpoint_minibatch_schedule_receipt_training_join"',
            '"checkpoint_state_receipt_training_inner_join"',
            '"primal_dual_trace_full_binding_receipt_checkpoint_join"',
            '"preselection_fixed1200_full_binding_reverified"',
            '"optimizer_steps_reexecuted"',
            '"oof_semantic_tensor_read_count"',
        ):
            self.assertIn(token, semantic)

    def test_model_fit_provenance_recomputes_all_required_authority_facts(self) -> None:
        replay = self.function_source(
            "_recompute_model_fit_provenance_from_authority"
        )
        for token in (
            "{iid: EVAL_VIEWS for iid in fit_iids}",
            'stage="stage1_model_fit_only"',
            "frozen._fit_clip_pca_b384(rows)",
            "frozen._fit_only_global_rms(rows, device)",
            "config.seed + 10000 + fold_index",
            "manual_seed(fold_seed + 1)",
            "(config.max_steps, config.batch_size)",
            '"model_fit_original_tensor_sha256"',
            '"model_fit_all_five_views_tensor_sha256"',
            '"fixed_clip_pca_b384_all_five_views_tensor_sha256"',
            '"minibatch_schedule_sha256"',
            '"optimizer_steps_reexecuted": 0',
            '"oof_semantic_tensor_read_count": 0',
        ):
            self.assertIn(token, replay)

    def test_no_assert_statements_guard_contracts(self) -> None:
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(self.tree)))


@unittest.skipIf(torch is None, "PyTorch unavailable")
class EvaluationAlignedPrimalDualV4HDynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from methods.bernini_action_editing import (
            semantic_anchor_vjepa2_evaluation_aligned_primal_dual_v4h as runtime,
        )
        cls.m = runtime

    def make_valid_traces(self, constraints: torch.Tensor | None = None):
        m = self.m
        if constraints is None:
            constraints = torch.zeros(m.CONSTRAINT_TRACE_SHAPE, dtype=torch.float64)
        constraints = constraints.to(torch.float64).contiguous()
        duals = torch.zeros(m.DUAL_TRACE_SHAPE, dtype=torch.float64)
        for step in range(m.FIXED_SELECTED_STEP):
            duals[step + 1] = torch.clamp_min(
                duals[step] + m.DUAL_LEARNING_RATE * constraints[step], 0.0
            )
        objectives = torch.empty(m.OBJECTIVE_TRACE_SHAPE, dtype=torch.float64)
        objectives[:, 0] = 1.0
        for step in range(m.FIXED_SELECTED_STEP):
            primal = objectives[step, 0].to(torch.float32)
            penalty = torch.zeros((), dtype=torch.float32)
            dual32 = duals[step].to(torch.float32)
            constraint32 = constraints[step].to(torch.float32)
            for index in range(m.CONSTRAINT_COUNT):
                penalty = penalty + dual32[index] * torch.relu(
                    constraint32[index]
                )
            objectives[step, 1] = (primal + penalty).to(torch.float64)
        return constraints, duals, objectives

    def test_runtime_model_has_exact_79040_parameters_and_no_scale_buffer(self) -> None:
        m = self.m
        fitted = SimpleNamespace(
            clip_mean=torch.zeros(1, m.FULL_NUMEL),
            clip_basis=torch.zeros(m.FULL_NUMEL, m.CODE_NUMEL),
        )
        model = m.VJepa2GlobalCodec(fitted, torch.ones(1))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 79040)
        self.assertNotIn("residual_gate_rho", model.state_dict())
        self.assertEqual(tuple(model.clip_basis.shape), (32768, 384))

    def test_unreleased_guard_fails_before_parser_or_binding(self) -> None:
        m = self.m
        self.assertIs(m.RELEASE_SEALED, False)
        with mock.patch.object(m, "_binding") as binding:
            with self.assertRaisesRegex(RuntimeError, "UNSEALED v4-H"):
                m.main([])
        binding.assert_not_called()
        origin = m._scientific_design_origin_contract()
        v4f_burned = origin["canonical_v4f_burned_source"]
        burned = origin["canonical_v4g_burned_source"]
        self.assertFalse(v4f_burned["controller_complete"])
        self.assertTrue(v4f_burned["scientific_no_go_valid"])
        self.assertEqual(
            origin["canonical_v4f_burned_source_sha256"],
            m._object_sha(v4f_burned),
        )
        self.assertFalse(burned["controller_complete"])
        self.assertTrue(burned["scientific_all_inner_no_go"])
        self.assertTrue(burned["all_oof_semantic_tensor_read_count_exact0"])
        self.assertEqual(
            origin["canonical_v4g_burned_source_sha256"],
            m._object_sha(burned),
        )
        self.assertGreaterEqual(
            RUNTIME_PATH.read_text(encoding="utf-8").count(
                '"scientific_design_origin_sha256"'
            ),
            3,
        )

    def test_fast_tensor_digest_is_bit_equivalent_to_legacy_storage_bytes(self) -> None:
        m = self.m
        value = torch.tensor([[1.25, -0.0], [3.5, -7.0]], dtype=torch.float32)
        contiguous = value.detach().to(device="cpu").contiguous().clone()
        legacy = hashlib.sha256()
        legacy.update(m._canonical_json({
            "dtype": str(contiguous.dtype), "shape": list(contiguous.shape),
        }))
        legacy.update(bytes(contiguous.untyped_storage()))
        self.assertEqual(m._tensor_sha(value), legacy.hexdigest())

    def test_primal_dual_trace_roundtrip_and_hostile_preparse_guards(self) -> None:
        m = self.m
        constraints = torch.zeros(m.CONSTRAINT_TRACE_SHAPE, dtype=torch.float64)
        constraints[:, 0] = 0.25
        constraints[:, 5] = -0.5
        constraints[17, 7] = 0.75
        constraint_trace, dual_trace, objective_trace = self.make_valid_traces(
            constraints
        )
        training = {
            "runtime_fingerprint": {"device_type": "cpu-test"},
            "minibatch_schedule_sha256": "1" * 64,
            "model_fit_original_count": 2,
            "model_fit_ordered_iids": ["a", "b"],
            "model_fit_iid_digest": m._object_sha(["a", "b"]),
            "model_fit_original_tensor_sha256": "2" * 64,
            "model_fit_all_five_views_tensor_sha256": "3" * 64,
            "fixed_clip_pca_b384_all_five_views_tensor_sha256": "4" * 64,
            "step0_state_sha256": "5" * 64,
            "final_step_state_sha256": "6" * 64,
        }
        binding = {"implementation_sha256": "7" * 64}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "primal_dual_trace.pt"
            artifact = m._save_primal_dual_trace_create_only(
                path, fold_index=2, training=training, run_binding=binding,
                constraint_trace=constraint_trace, dual_trace=dual_trace,
                objective_trace=objective_trace,
            )
            expected = {
                **artifact,
                "outer_fold": 2,
                "implementation_sha256": binding["implementation_sha256"],
                **training,
                "model_state_sha256": training["final_step_state_sha256"],
            }
            metadata, tensors, replay = m._load_primal_dual_trace_sealed(
                path, expected
            )
            self.assertEqual(replay, artifact)
            self.assertEqual(set(tensors), {
                "constraint_trace", "dual_trace", "objective_trace",
            })
            self.assertEqual(set(metadata), m.TRACE_METADATA_KEYS)
            self.assertEqual(metadata["trace_dtype"], "torch.float64")
            self.assertEqual(metadata["objective_trace_columns"], ["P_t", "L_t"])
            self.assertEqual(
                replay["training_trace_summary_sha256"],
                m._object_sha(m._training_trace_summary(
                    constraint_trace, dual_trace, objective_trace
                )),
            )
            hostile_expectations = []
            hostile_sha = copy.deepcopy(expected)
            hostile_sha["file_sha256"] = "0" * 64
            hostile_expectations.append(hostile_sha)
            hostile_size = copy.deepcopy(expected)
            hostile_size["size_bytes"] += 1
            hostile_expectations.append(hostile_size)
            hostile_identity = copy.deepcopy(expected)
            hostile_identity["physical_identity"]["inode"] += 1
            hostile_expectations.append(hostile_identity)
            for hostile in hostile_expectations:
                with mock.patch.object(m.torch, "load") as unsafe_parse:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "expected binding differs before torch parse",
                    ):
                        m._load_primal_dual_trace_sealed(path, hostile)
                unsafe_parse.assert_not_called()

            semantic_expected = {
                "outer_fold": 2,
                "implementation_sha256": binding["implementation_sha256"],
                **training,
                "model_state_sha256": training["final_step_state_sha256"],
            }
            pristine = torch.load(path, map_location="cpu", weights_only=True)

            def write_hostile(name: str, payload: dict) -> Path:
                hostile_path = Path(directory).resolve() / name
                torch.save(payload, hostile_path)
                hostile_path.chmod(0o444)
                return hostile_path

            def resign_trace(payload: dict) -> None:
                metadata = payload["metadata"]
                metadata["constraint_trace_tensor_sha256"] = m._tensor_sha(
                    payload["constraint_trace"]
                )
                metadata["dual_trace_tensor_sha256"] = m._tensor_sha(
                    payload["dual_trace"]
                )
                metadata["objective_trace_tensor_sha256"] = m._tensor_sha(
                    payload["objective_trace"]
                )
                metadata["training_trace_summary_sha256"] = m._object_sha(
                    m._training_trace_summary(
                        payload["constraint_trace"], payload["dual_trace"],
                        payload["objective_trace"],
                    )
                )
                metadata.pop("metadata_digest", None)
                metadata["metadata_digest"] = m._object_sha(metadata)

            extra_envelope = copy.deepcopy(pristine)
            extra_envelope["unexpected"] = None
            with self.assertRaisesRegex(RuntimeError, "exact4 keys"):
                m._load_primal_dual_trace_sealed(
                    write_hostile("extra-envelope.pt", extra_envelope),
                    semantic_expected,
                )

            extra_metadata = copy.deepcopy(pristine)
            extra_metadata["metadata"]["unexpected"] = None
            with self.assertRaisesRegex(RuntimeError, "metadata envelope"):
                m._load_primal_dual_trace_sealed(
                    write_hostile("extra-metadata.pt", extra_metadata),
                    semantic_expected,
                )

            for field, value, name in (
                ("fixed_step", 1199, "bad-fixed-step.pt"),
                ("full_budget_steps_executed", 1199, "bad-budget.pt"),
            ):
                bad_fixed_contract = copy.deepcopy(pristine)
                bad_fixed_contract["metadata"][field] = value
                resign_trace(bad_fixed_contract)
                with self.assertRaisesRegex(
                    RuntimeError, "semantic metadata differs"
                ):
                    m._load_primal_dual_trace_sealed(
                        write_hostile(name, bad_fixed_contract),
                        semantic_expected,
                    )

            wrong_dtype = copy.deepcopy(pristine)
            wrong_dtype["constraint_trace"] = wrong_dtype[
                "constraint_trace"
            ].to(torch.float32)
            resign_trace(wrong_dtype)
            with self.assertRaisesRegex(RuntimeError, "tensor closure differs"):
                m._load_primal_dual_trace_sealed(
                    write_hostile("wrong-dtype.pt", wrong_dtype),
                    semantic_expected,
                )

            bad_recurrence = copy.deepcopy(pristine)
            bad_recurrence["dual_trace"][1, 0] += 1.0
            resign_trace(bad_recurrence)
            with self.assertRaisesRegex(RuntimeError, "exact recurrence differs"):
                m._load_primal_dual_trace_sealed(
                    write_hostile("bad-recurrence.pt", bad_recurrence),
                    semantic_expected,
                )

            bad_objective = copy.deepcopy(pristine)
            bad_objective["objective_trace"][0, 1] += 1.0
            resign_trace(bad_objective)
            with self.assertRaisesRegex(RuntimeError, "exact replay differs"):
                m._load_primal_dual_trace_sealed(
                    write_hostile("bad-objective.pt", bad_objective),
                    semantic_expected,
                )

    def test_real_preselection_fixed1200_create_seal_reload_pair(self) -> None:
        m = self.m
        config = m.Config()
        torch.set_num_threads(1)
        m._seed_everything(config.seed + 10000, torch.device("cpu"))
        fitted = m.ClipPCAFit(
            clip_mean=torch.zeros(1, m.FULL_NUMEL),
            clip_basis=torch.zeros(m.FULL_NUMEL, m.CODE_NUMEL),
            fit_iid_digest=m._object_sha(["fit"]),
            fit_input_sha256="a" * 64,
            diagnostics={},
        )
        model = m.VJepa2GlobalCodec(fitted, torch.ones(1)).eval()
        state_sha = m._state_sha(m._state_to_cpu(model))
        schedule = torch.zeros(config.max_steps, config.batch_size, dtype=torch.int64)
        training = {
            "fixed_step": m.FIXED_SELECTED_STEP,
            "full_budget_steps_executed": config.max_steps,
            "step0_state_sha256": state_sha,
            "final_step_state_sha256": state_sha,
            "selected_state_sha256": state_sha,
            "minibatch_schedule_sha256": m._tensor_sha(schedule),
            "minibatch_schedule_shape": [config.max_steps, config.batch_size],
            "fit_only_global_rms_sha256": m._tensor_sha(torch.ones(1)),
            "runtime_fingerprint": m._runtime_fingerprint(torch.device("cpu")),
            "model_fit_original_count": 1,
            "model_fit_ordered_iids": ["fit"],
            "model_fit_iid_digest": m._object_sha(["fit"]),
            "model_fit_original_tensor_sha256": "c" * 64,
            "model_fit_all_five_views_tensor_sha256": "d" * 64,
            "fixed_clip_pca_b384_all_five_views_tensor_sha256": "e" * 64,
            "inner_validation_iid_digest": m._object_sha(["inner"]),
            "inner_validation_original_count": 1,
            "inner_validation_ordered_iids": ["inner"],
            "fixed_clip_pca_fit_input_sha256": "a" * 64,
        }
        record = m.v4c.Record(
            iid="fit", family="family", strict=False,
            views={"original": torch.zeros(m.TIME_STEPS, m.FEATURE_DIM)},
        )
        binding = {"implementation_sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            pre_path = root / "preselection.pt"
            fixed_path = root / "fixed1200.pt"
            trace_path = root / "primal_dual_trace.pt"
            constraints, duals, objectives = self.make_valid_traces()
            trace = m._save_primal_dual_trace_create_only(
                trace_path, fold_index=0, training=training,
                run_binding=binding, constraint_trace=constraints,
                dual_trace=duals, objective_trace=objectives,
            )
            training.update(m._training_trace_summary(
                constraints, duals, objectives
            ))
            training["primal_dual_trace_artifact"] = trace
            training["primal_dual_trace_binding_sha256"] = m._object_sha(trace)
            pre = m._save_checkpoint_create_only(
                pre_path, model, fitted, training, config, 0, binding,
                [record], torch.device("cpu"),
                checkpoint_role="preselection_fixed_step1200",
                preselection_artifact=None,
            )
            fixed = m._save_checkpoint_create_only(
                fixed_path, model, fitted, training, config, 0, binding,
                [record], torch.device("cpu"),
                checkpoint_role="fixed1200_candidate",
                preselection_artifact=pre,
            )
            pair = m._verify_distinct_checkpoint_pair(pre, fixed)
            self.assertTrue(pair["distinct_device_inode_pair"])
            self.assertTrue(pair["same_model_state_sha256"])
            self.assertNotEqual(pre["physical_identity"]["inode"], fixed["physical_identity"]["inode"])
            self.assertEqual(fixed["preselection_checkpoint_file_sha256"], pre["file_sha256"])
            receipt = {
                "fold_index": 0,
                "implementation": binding,
                "model_fit_original_count": 1,
                "model_fit_ordered_iids": ["fit"],
                "model_fit_iid_digest": m._object_sha(["fit"]),
                "inner_validation_original_count": 1,
                "inner_validation_ordered_iids": ["inner"],
                "inner_validation_iid_digest": m._object_sha(["inner"]),
                "fixed_clip_pca_b384_fit_input_sha256": "a" * 64,
                "fixed_clip_pca_b384_fit_iid_digest": m._object_sha(["fit"]),
                "runtime_fingerprint": training["runtime_fingerprint"],
                "training": training,
                "primal_dual_trace_artifact": trace,
                "primal_dual_trace_binding_sha256": m._object_sha(trace),
                "preselection_checkpoint_artifact": pre,
                "fixed1200_checkpoint_artifact": fixed,
                "fixed_candidate": {
                    "model_state_sha256_before_inner": state_sha,
                    "model_state_sha256_after_inner": state_sha,
                },
            }
            pre_expected, fixed_expected = (
                m._checkpoint_expectations_from_inner_receipt(receipt)
            )
            self.assertEqual(pre_expected["outer_fold"], 0)
            self.assertEqual(fixed_expected["model_fit_ordered_iids"], ["fit"])
            hostile_runtime = dict(training["runtime_fingerprint"])
            hostile_runtime["device_class"] = {
                "name": "AMD Instinct MI210", "gcn_arch_name": "gfx90a",
            }
            with self.assertRaisesRegex(RuntimeError, "runtime and device class differ"):
                m._checkpoint_expectations_from_inner_receipt(
                    receipt, expected_runtime_fingerprint=hostile_runtime,
                )
            hostile_receipt = dict(receipt)
            hostile_receipt["fixed1200_checkpoint_artifact"] = {
                **fixed, "outer_fold": 1,
            }
            with self.assertRaisesRegex(RuntimeError, "provenance join differs"):
                m._checkpoint_expectations_from_inner_receipt(hostile_receipt)
            split_binding_receipt = dict(receipt)
            split_fixed = copy.deepcopy(fixed)
            split_fixed["preselection_checkpoint_binding"]["size_bytes"] += 1
            split_binding_receipt["fixed1200_checkpoint_artifact"] = split_fixed
            with self.assertRaisesRegex(RuntimeError, "provenance join differs"):
                m._checkpoint_expectations_from_inner_receipt(
                    split_binding_receipt
                )
            cross_fold_expected = dict(fixed)
            cross_fold_expected["outer_fold"] = 1
            with self.assertRaisesRegex(
                RuntimeError, "checkpoint semantic metadata replay differs"
            ):
                m._load_checkpoint_sealed(fixed_path, cross_fold_expected)
            split_binding_expected = copy.deepcopy(fixed)
            split_binding_expected["preselection_checkpoint_binding"][
                "size_bytes"
            ] += 1
            with self.assertRaisesRegex(
                RuntimeError, "checkpoint semantic metadata replay differs"
            ):
                m._load_checkpoint_sealed(fixed_path, split_binding_expected)

    def test_forged_all_pass_receipts_are_rejected_by_checkpoint_forward_replay(self) -> None:
        m = self.m
        runtime = m._runtime_fingerprint(torch.device("cpu"))
        forged = [{
            "fold_index": fold,
            "implementation": {"implementation_sha256": "i" * 64},
            "inner_split": {"fold": fold},
            "inner_validation_ordered_iids": [f"inner{fold}"],
            "fixed_clip_pca_b384_diagnostics": {},
            "training": {
                "trainable_parameter_count": 79040,
                "full_budget_steps_executed": 1200,
                "early_stopped": False,
                "checkpoint_winner_selection_performed": False,
                "hyperparameter_selection_performed": False,
                "runtime_fingerprint": runtime,
                "minibatch_schedule_sha256": "m" * 64,
                "final_step_state_sha256": "s" * 64,
                "step0_state_sha256": "z" * 64,
                "model_fit_original_count": 1,
                "model_fit_ordered_iids": [f"fit{fold}"],
                "model_fit_iid_digest": m._object_sha([f"fit{fold}"]),
                "model_fit_original_tensor_sha256": "o" * 64,
                "model_fit_all_five_views_tensor_sha256": "v" * 64,
                "fixed_clip_pca_b384_all_five_views_tensor_sha256": "b" * 64,
            },
            "primal_dual_trace_artifact": {
                "file_sha256": f"{fold + 1}" * 64,
                "metadata_digest": "t" * 64,
            },
            "preselection_checkpoint_artifact": {},
            "preselection_fixed1200_checkpoint_pair_join": {},
            "fixed1200_checkpoint_artifact": {"model_state_sha256": "s" * 64},
            "fixed_candidate": {"inner_pass": True, "pass": True, "forged": True},
        } for fold in range(5)]
        bindings = [{
            "fold_root": f"/forged/fold{fold}",
            "file_sha256": str(fold) * 64,
            "primal_dual_trace_artifact": forged[fold][
                "primal_dual_trace_artifact"
            ],
        } for fold in range(5)]
        fake_state = {
            "clip_mean": torch.zeros(1),
            "clip_basis": torch.zeros(1),
            "fit_only_rms": torch.ones(1),
        }
        model = mock.Mock()
        model.to.return_value = model
        model.eval.return_value = model
        fake_fitted = SimpleNamespace(
            clip_mean=fake_state["clip_mean"],
            clip_basis=fake_state["clip_basis"],
            fit_input_sha256="f" * 64,
            diagnostics={},
        )
        materialize = mock.Mock(side_effect=lambda _index, request, *, stage: (
            {iid: SimpleNamespace(iid=iid) for iid in request},
            {"semantic_tensor_materialized_count": len(request) * 5, "stage": stage},
        ))
        genuine = {
            "inner_pass": True, "pass": True,
            "model_state_sha256_before_inner": "s" * 64,
            "inner_evidence_sha256": "e" * 64,
            "gate": {"complete_candidate_dependent_inner_gate": True},
            "bootstrap_seed_ledger": [],
        }

        def split(_records, _assignment, fold, _config):
            return ({
                "model_fit": [SimpleNamespace(iid=f"fit{fold}")],
                "inner_validation": [SimpleNamespace(iid=f"inner{fold}")],
                "exploratory_oof": [SimpleNamespace(iid=f"oof{fold}")],
            }, {"fold": fold})

        def load_trace(path, _expected):
            fold = int(path.parent.name.removeprefix("fold"))
            artifact = forged[fold]["primal_dual_trace_artifact"]
            return ({
                "fixed_clip_pca_b384_all_five_views_tensor_sha256": "b" * 64,
            }, {
                "constraint_trace": None, "dual_trace": None,
                "objective_trace": None,
            }, artifact)

        with mock.patch.object(m, "_verify_inner_receipt_against_authority"), \
                mock.patch.object(
                    m, "_load_primal_dual_trace_sealed", side_effect=load_trace,
                ), mock.patch.object(m, "_training_trace_summary", return_value={}):
            with mock.patch.object(
                m, "_checkpoint_expectations_from_inner_receipt",
                return_value=({}, {}),
            ):
                with mock.patch.object(
                    m, "_load_checkpoint_sealed",
                    return_value=(
                        {"model_fit_iid_digest": "d", "minibatch_schedule_sha256": "m" * 64,
                         "basis": {
                            "clip_mean_sha256": m._tensor_sha(fake_state["clip_mean"]),
                            "clip_basis_sha256": m._tensor_sha(fake_state["clip_basis"]),
                            "fit_only_global_rms_sha256": m._tensor_sha(fake_state["fit_only_rms"]),
                            "fixed_clip_pca_fit_input_sha256": "f" * 64,
                        }},
                        fake_state,
                        {
                            "file_sha256": "c" * 64,
                            "model_state_sha256": "s" * 64,
                            "primal_dual_trace_artifact": forged[0][
                                "primal_dual_trace_artifact"
                            ],
                        },
                    ),
                ):
                    with mock.patch.object(
                        m, "_recompute_model_fit_provenance_from_authority",
                        return_value=(
                            fake_fitted, torch.ones(1),
                            {
                                "minibatch_schedule_sha256": "m" * 64,
                                "fixed_clip_pca_b384_all_five_views_tensor_sha256": "b" * 64,
                            },
                        ),
                    ):
                        with mock.patch.object(
                            m, "_verify_distinct_checkpoint_pair", return_value={},
                        ):
                            with mock.patch.object(m, "VJepa2GlobalCodec", return_value=model):
                                with mock.patch.object(m.frozen, "_split_fold", side_effect=split):
                                    with mock.patch.object(
                                        m.frozen, "_selective_materialize_feature_rows",
                                        materialize,
                                    ):
                                        with mock.patch.object(
                                            m, "_evaluate_fixed_inner_candidate",
                                            return_value=genuine,
                                        ):
                                            with self.assertRaisesRegex(
                                                RuntimeError,
                                                "checkpoint-forward inner evidence/gate replay differs",
                                            ):
                                                m._independently_replay_all_inner_gates_before_oof(
                                                    forged, bindings,
                                                    {"ordered_records": [], "outer_assignment": {},
                                                     "feature_index": {}},
                                                    m.Config(), torch.device("cpu"),
                                                )
        self.assertGreaterEqual(materialize.call_count, 1)
        self.assertTrue(all(
            call.kwargs["stage"]
                == "stage2_post_preselection_seal_inner_five_views"
            for call in materialize.call_args_list
        ))

    def test_genuine_exact_five_independent_replays_pass(self) -> None:
        m = self.m
        runtime = m._runtime_fingerprint(torch.device("cpu"))
        state_sha = "s" * 64
        receipts = []
        bindings = []
        candidates = []
        for fold in range(5):
            candidate = {
                "inner_pass": True,
                "pass": True,
                "model_state_sha256_before_inner": state_sha,
                "model_state_sha256_after_inner": state_sha,
                "inner_evidence_sha256": f"{fold}" * 64,
                "gate": {"complete_candidate_dependent_inner_gate": True},
                "bootstrap_seed_ledger": [{"fold": fold}],
            }
            candidates.append(candidate)
            receipts.append({
                "fold_index": fold,
                "implementation": {"implementation_sha256": "i" * 64},
                "inner_split": {"fold": fold},
                "inner_validation_ordered_iids": [f"inner{fold}"],
                "fixed_clip_pca_b384_diagnostics": {},
                "fixed_clip_pca_b384_fit_input_sha256": "p" * 64,
                "runtime_fingerprint": runtime,
                "training": {
                    "fixed_clip_pca_fit_input_sha256": "p" * 64,
                    "minibatch_schedule_sha256": "m" * 64,
                    "final_step_state_sha256": state_sha,
                    "step0_state_sha256": "z" * 64,
                    "runtime_fingerprint": runtime,
                    "model_fit_original_count": 1,
                    "model_fit_ordered_iids": [f"fit{fold}"],
                    "model_fit_iid_digest": m._object_sha([f"fit{fold}"]),
                    "model_fit_original_tensor_sha256": "o" * 64,
                    "model_fit_all_five_views_tensor_sha256": "v" * 64,
                    "fixed_clip_pca_b384_all_five_views_tensor_sha256": "b" * 64,
                    "trainable_parameter_count": 79040,
                    "full_budget_steps_executed": 1200,
                    "early_stopped": False,
                    "checkpoint_winner_selection_performed": False,
                    "hyperparameter_selection_performed": False,
                },
                "preselection_checkpoint_artifact": {},
                "primal_dual_trace_artifact": {
                    "file_sha256": f"{fold + 1}" * 64,
                    "metadata_digest": "t" * 64,
                },
                "preselection_fixed1200_checkpoint_pair_join": {},
                "fixed1200_checkpoint_artifact": {
                    "model_state_sha256": state_sha,
                },
                "fixed_candidate": candidate,
            })
            bindings.append({
                "fold_root": f"/fold{fold}",
                "file_sha256": f"{fold}" * 64,
                "primal_dual_trace_artifact": receipts[-1][
                    "primal_dual_trace_artifact"
                ],
            })
        fake_state = {
            "clip_mean": torch.zeros(1),
            "clip_basis": torch.zeros(1),
            "fit_only_rms": torch.ones(1),
        }
        model = mock.Mock()
        model.to.return_value = model
        model.eval.return_value = model
        fake_fitted = SimpleNamespace(
            clip_mean=fake_state["clip_mean"],
            clip_basis=fake_state["clip_basis"],
            fit_input_sha256="p" * 64,
            diagnostics={},
        )

        def split(_records, _assignment, fold, _config):
            return ({
                "model_fit": [SimpleNamespace(iid=f"fit{fold}")],
                "inner_validation": [SimpleNamespace(iid=f"inner{fold}")],
                "exploratory_oof": [SimpleNamespace(iid=f"oof{fold}")],
            }, {"fold": fold})

        def load(path, _expected):
            fold = int(path.parent.name.removeprefix("fold"))
            metadata = {
                "outer_fold": fold,
                "model_fit_original_count": 1,
                "model_fit_ordered_iids": [f"fit{fold}"],
                "model_fit_iid_digest": m._object_sha([f"fit{fold}"]),
                "inner_validation_iid_digest": m._object_sha([f"inner{fold}"]),
                "minibatch_schedule_sha256": "m" * 64,
                "model_state_sha256": state_sha,
                "runtime_fingerprint": runtime,
                "basis": {
                    "clip_mean_sha256": m._tensor_sha(fake_state["clip_mean"]),
                    "clip_basis_sha256": m._tensor_sha(fake_state["clip_basis"]),
                    "fit_only_global_rms_sha256": m._tensor_sha(fake_state["fit_only_rms"]),
                    "fixed_clip_pca_fit_input_sha256": "p" * 64,
                },
            }
            return metadata, fake_state, {
                "file_sha256": "c" * 64,
                "model_state_sha256": state_sha,
                "primal_dual_trace_artifact": receipts[fold][
                    "primal_dual_trace_artifact"
                ],
            }

        def load_trace(path, _expected):
            fold = int(path.parent.name.removeprefix("fold"))
            artifact = receipts[fold]["primal_dual_trace_artifact"]
            return ({
                "fixed_clip_pca_b384_all_five_views_tensor_sha256": "b" * 64,
            }, {
                "constraint_trace": None, "dual_trace": None,
                "objective_trace": None,
            }, artifact)

        materialize = mock.Mock(side_effect=lambda _index, request, *, stage: (
            {iid: SimpleNamespace(iid=iid) for iid in request},
            {"semantic_tensor_materialized_count": len(request) * 5, "stage": stage},
        ))
        with mock.patch.object(m, "_verify_inner_receipt_against_authority"), \
                mock.patch.object(
                    m, "_load_primal_dual_trace_sealed", side_effect=load_trace,
                ), mock.patch.object(m, "_training_trace_summary", return_value={}):
            with mock.patch.object(
                m, "_checkpoint_expectations_from_inner_receipt",
                return_value=({}, {}),
            ):
                with mock.patch.object(m, "_load_checkpoint_sealed", side_effect=load):
                    with mock.patch.object(
                        m, "_recompute_model_fit_provenance_from_authority",
                        return_value=(
                            fake_fitted, torch.ones(1),
                            {
                                "minibatch_schedule_sha256": "m" * 64,
                                "fixed_clip_pca_b384_all_five_views_tensor_sha256": "b" * 64,
                            },
                        ),
                    ):
                        with mock.patch.object(
                            m, "_verify_distinct_checkpoint_pair", return_value={},
                        ):
                            with mock.patch.object(m, "VJepa2GlobalCodec", return_value=model):
                                with mock.patch.object(m.frozen, "_split_fold", side_effect=split):
                                    with mock.patch.object(
                                        m.frozen, "_selective_materialize_feature_rows",
                                        materialize,
                                    ):
                                        with mock.patch.object(
                                            m, "_evaluate_fixed_inner_candidate",
                                            side_effect=candidates,
                                        ):
                                            ledger, digest = (
                                                m._independently_replay_all_inner_gates_before_oof(
                                                    receipts, bindings,
                                                    {"ordered_records": [], "outer_assignment": {},
                                                     "feature_index": {}},
                                                    m.Config(), torch.device("cpu"),
                                                )
                                            )
        self.assertEqual(len(ledger), 5)
        self.assertEqual(digest, m._object_sha(ledger))
        self.assertTrue(all(row["inner_pass"] for row in ledger))
        self.assertTrue(all(
            row["checkpoint_outer_fold_authority_join"] for row in ledger
        ))

    def test_runtime_setup_closes_default_flag_and_device_class_mismatch(self) -> None:
        m = self.m
        device = torch.device("cpu")
        prior_threads = torch.get_num_threads()
        prior_deterministic = torch.are_deterministic_algorithms_enabled()
        prior_benchmark = torch.backends.cudnn.benchmark
        prior_cudnn_deterministic = torch.backends.cudnn.deterministic
        prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        prior_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            before = m._runtime_fingerprint(device)
            torch.set_num_threads(1)
            m._seed_everything(m.Config().seed + 30000, device)
            after = m._runtime_fingerprint(device)
            self.assertNotEqual(before, after)
            self.assertEqual(after["torch_num_threads"], 1)
            self.assertTrue(after["deterministic_algorithms_enabled"])
            self.assertFalse(after["cudnn_benchmark"])
            self.assertTrue(after["cudnn_deterministic"])
            self.assertFalse(after["cudnn_allow_tf32"])
            self.assertFalse(after["matmul_allow_tf32"])
            hostile = dict(after)
            hostile["device_class"] = {"name": "AMD Instinct MI210", "gcn_arch_name": "gfx90a"}
            self.assertNotEqual(hostile, after)
        finally:
            torch.set_num_threads(prior_threads)
            torch.use_deterministic_algorithms(prior_deterministic)
            torch.backends.cudnn.benchmark = prior_benchmark
            torch.backends.cudnn.deterministic = prior_cudnn_deterministic
            torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32

    def test_model_fit_provenance_recompute_rejects_each_forged_fact(self) -> None:
        m = self.m
        config = m.Config()
        fold = 0
        views = {
            view: torch.full(
                (m.TIME_STEPS, m.FEATURE_DIM), float(index + 1),
                dtype=torch.float32,
            )
            for index, view in enumerate(m.EVAL_VIEWS)
        }
        row = m.v4c.Record(iid="fit", family="family", strict=False, views=views)
        all_five = torch.stack([
            torch.stack([m.v4c.canonical_action(row.views[view]) for view in m.EVAL_VIEWS])
        ])
        originals = all_five[:, m.EVAL_VIEWS.index("original")]
        baseline_all_five = all_five + 0.25
        fit_input_sha = "p" * 64
        diagnostics = {"basis": "authority"}
        fitted = SimpleNamespace(
            clip_mean=torch.zeros(1), clip_basis=torch.ones(1),
            fit_iid_digest=m._object_sha(["fit"]),
            fit_input_sha256=fit_input_sha, diagnostics=diagnostics,
        )
        rms = torch.tensor([2.0])
        seed = config.seed + 10000 + fold
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        schedule = torch.randint(
            1, (config.max_steps, config.batch_size), generator=generator,
        )
        receipt = {
            "model_fit_ordered_iids": ["fit"],
            "fixed_clip_pca_b384_fit_input_sha256": fit_input_sha,
            "fixed_clip_pca_b384_diagnostics": diagnostics,
            "training": {
                "fold_seed": seed,
                "model_fit_original_tensor_sha256": m._tensor_sha(originals),
                "model_fit_all_five_views_tensor_sha256": m._tensor_sha(all_five),
                "fixed_clip_pca_b384_all_five_views_tensor_sha256": m._tensor_sha(
                    baseline_all_five
                ),
                "fit_only_global_rms_sha256": m._tensor_sha(rms),
                "fit_only_global_rms": 2.0,
                "minibatch_schedule_sha256": m._tensor_sha(schedule),
            },
            "fixed1200_checkpoint_artifact": {
                "minibatch_schedule_sha256": m._tensor_sha(schedule),
            },
        }
        audit = {
            "semantic_tensor_materialized_count": len(m.EVAL_VIEWS),
            "semantic_tensor_materialized_count_by_view": {
                view: 1 for view in m.EVAL_VIEWS
            },
        }
        materialize = mock.Mock(return_value=({"fit": row}, audit))
        groups = {
            "model_fit": [SimpleNamespace(iid="fit")],
            "exploratory_oof": [SimpleNamespace(iid="oof")],
        }
        with mock.patch.object(
            m.frozen, "_selective_materialize_feature_rows", materialize,
        ):
            with mock.patch.object(m.frozen, "_fit_clip_pca_b384", return_value=fitted):
                with mock.patch.object(
                    m.frozen, "_analytic_clip_pca_decode",
                    return_value=baseline_all_five.flatten(0, 1),
                ):
                    with mock.patch.object(m.frozen, "_fit_only_global_rms", return_value=rms):
                        _, _, ledger = m._recompute_model_fit_provenance_from_authority(
                            {"feature_index": {}}, groups, receipt, config, fold,
                            torch.device("cpu"),
                        )
                        self.assertEqual(ledger["optimizer_steps_reexecuted"], 0)
                        self.assertEqual(ledger["oof_semantic_tensor_read_count"], 0)
                        hostile_paths = (
                        ("training", "model_fit_original_tensor_sha256"),
                        ("training", "model_fit_all_five_views_tensor_sha256"),
                        ("training", "fixed_clip_pca_b384_all_five_views_tensor_sha256"),
                        ("fixed_clip_pca_b384_fit_input_sha256",),
                        ("fixed_clip_pca_b384_diagnostics",),
                        ("training", "fit_only_global_rms_sha256"),
                        ("training", "fit_only_global_rms"),
                        ("training", "fold_seed"),
                        ("training", "minibatch_schedule_sha256"),
                        ("fixed1200_checkpoint_artifact", "minibatch_schedule_sha256"),
                        )
                        for path in hostile_paths:
                            with self.subTest(path=path):
                                hostile = copy.deepcopy(receipt)
                                target = hostile
                                for key in path[:-1]:
                                    target = target[key]
                                target[path[-1]] = "forged"
                                with self.assertRaisesRegex(
                                    RuntimeError, "PCA/RMS/schedule receipt join differs"
                                ):
                                    m._recompute_model_fit_provenance_from_authority(
                                        {"feature_index": {}}, groups, hostile,
                                        config, fold, torch.device("cpu"),
                                    )
        self.assertTrue(all(
            call.kwargs["stage"] == "stage1_model_fit_only"
            and call.args[1] == {"fit": m.EVAL_VIEWS}
            for call in materialize.call_args_list
        ))

    def test_step0_and_one_step_primal_dual_formula_and_nonfinite_guards(self) -> None:
        m = self.m
        generator = torch.Generator().manual_seed(5)
        target = torch.randn(
            2, len(m.EVAL_VIEWS), m.TIME_STEPS, m.FEATURE_DIM,
            generator=generator,
        )
        baseline = target + 0.1 * torch.randn(target.shape, generator=generator)
        zero_duals = torch.zeros(m.CONSTRAINT_COUNT, dtype=torch.float32)
        step0_loss, step0_constraints, step0_components = (
            m._evaluation_aligned_primal_dual_loss(
                baseline, target, baseline, zero_duals, m.Config()
            )
        )
        self.assertEqual(step0_components["fidelity_ratio_of_means"], [1.0] * 5)
        self.assertTrue(torch.equal(
            step0_constraints[:5], torch.ones(5) - 1.02
        ))
        self.assertEqual(float(step0_loss), 1.0)

        prediction = (baseline + 0.03 * torch.randn(
            target.shape, generator=generator
        )).requires_grad_(True)
        duals = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            dtype=torch.float32,
        )
        total, constraints, components = m._evaluation_aligned_primal_dual_loss(
            prediction, target, baseline, duals, m.Config()
        )
        candidate_mse = (prediction - target).square().mean(dim=(2, 3))
        baseline_mse = (baseline - target).square().mean(dim=(2, 3))
        ratios = candidate_mse.sum(dim=0) / baseline_mse.sum(dim=0)
        index = {name: m.EVAL_VIEWS.index(name) for name in m.EVAL_VIEWS}

        def margins(value):
            query = value[:, index["original"]]
            positive = value[:, index["monotone_warp"]]
            return torch.stack([
                (query - value[:, index[name]]).square().mean(dim=(1, 2))
                - (query - positive).square().mean(dim=(1, 2))
                for name in m.NEGATIVES
            ], dim=1)
        pair_distances = torch.stack([
            (target[:, left] - target[:, right]).square().mean(dim=(1, 2))
            for left in range(5) for right in range(left + 1, 5)
        ], dim=1)
        scale_mean = (pair_distances.mean(dim=1) + 1.0e-8).mean()
        expected_constraints = torch.cat((
            ratios - 1.02,
            (
                0.85 * margins(target).mean(dim=0)
                - margins(prediction).mean(dim=0)
            ) / scale_mean,
        ))
        manual = ratios.mean()
        for constraint_index in range(m.CONSTRAINT_COUNT):
            manual = manual + duals[constraint_index] * torch.relu(
                expected_constraints[constraint_index]
            )
        self.assertTrue(torch.equal(constraints, expected_constraints))
        self.assertTrue(torch.equal(total, manual))
        self.assertEqual(
            components["constraint_order"], list(m.CONSTRAINT_ORDER)
        )
        total.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        with self.assertRaisesRegex(RuntimeError, "denominator must be finite and >0"):
            m._evaluation_aligned_primal_dual_loss(
                target.clone(), target, target.clone(), zero_duals, m.Config()
            )
        nonfinite = prediction.detach().clone()
        nonfinite[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "input is non-finite"):
            m._evaluation_aligned_primal_dual_loss(
                nonfinite, target, baseline, zero_duals, m.Config()
            )

    def test_exact_one_inner_candidate_calls_evaluator_once(self) -> None:
        m = self.m
        evidence = [{
            "iid": "i", "family": "f",
            "teacher_margin_by_negative": {name: 2.0 for name in m.NEGATIVES},
            "clip_pca_b384_margin_by_negative": {name: 0.5 for name in m.NEGATIVES},
            "candidate_margin_by_negative": {name: 2.0 for name in m.NEGATIVES},
            "raw_reconstruction_by_view": {
                view: {"candidate_raw_mse": 0.5, "clip_pca_b384_raw_mse": 1.0}
                for view in m.EVAL_VIEWS
            },
            "fixed_step": 1200, "fixed_residual_scale": 1.0,
            "single_fixed_candidate": True,
        }]
        model = SimpleNamespace()
        with mock.patch.object(
            m, "_state_to_cpu", return_value={"x": torch.ones(1)}
        ):
            with mock.patch.object(
                m, "_evaluate_rows_fixed", return_value=evidence
            ) as evaluate:
                with mock.patch.object(
                    m, "_inner_candidate_gate",
                    return_value={"complete_candidate_dependent_inner_gate": True},
                ) as gate:
                    result = m._evaluate_fixed_inner_candidate(
                        [], model, SimpleNamespace(), m.Config(), 0,
                        torch.device("cpu"),
                    )
        evaluate.assert_called_once()
        gate.assert_called_once()
        self.assertEqual(result["candidate_count"], 1)
        self.assertTrue(result["single_candidate"])
        self.assertFalse(result["hyperparameter_selection_performed"])
        self.assertTrue(result["inner_pass"])

    def test_global_barrier_failure_returns_before_any_downstream_work(self) -> None:
        m = self.m
        roots = [f"/fold/{index}" for index in range(5)]

        def load(root, _expected, _binding):
            index = int(root.rsplit("/", 1)[1])
            return ({
                "fold_index": index,
                "inner_pass": index != 3,
                "fixed1200_checkpoint_artifact": {"file_sha256": "f" * 64},
            }, {
                "fold_root": root,
                "file_sha256": str(index) * 64,
                "receipt_digest": "d" * 64,
            })

        with mock.patch.object(m, "_load_inner_receipt_sealed", side_effect=load):
            with self.assertRaisesRegex(RuntimeError, "GLOBAL_INNER_NO_GO"):
                m._load_all_inner_receipts_or_fail_before_oof(
                    roots, [str(index) * 64 for index in range(5)], {}
                )

    def test_sealed_barrier_rejects_semantic_forgery_even_with_caller_sha(self) -> None:
        m = self.m
        run_binding = {"implementation_sha256": "1" * 64}
        runtime_fingerprint = {"device_class": "cpu-test", "torch": "test"}
        authority_binding = {"authority": "test"}
        replay = []
        members = []
        required_true = (
            "checkpoint_outer_fold_authority_join",
            "checkpoint_model_fit_ordered_iids_authority_join",
            "checkpoint_model_fit_count_and_digest_authority_join",
            "checkpoint_inner_iid_digest_authority_join",
            "checkpoint_pca_fit_input_receipt_training_join",
            "checkpoint_minibatch_schedule_receipt_training_join",
            "checkpoint_state_receipt_training_inner_join",
            "primal_dual_trace_full_binding_receipt_checkpoint_join",
            "primal_dual_trace_b384_authority_join",
            "primal_dual_trace_formula_recurrence_strongly_replayed",
            "training_evaluate_runtime_fingerprint_exact_match",
            "checkpoint_clip_mean_equals_authority_recomputed_pca",
            "checkpoint_clip_basis_equals_authority_recomputed_pca",
            "checkpoint_fit_only_rms_equals_authority_recomputed_rms",
            "checkpoint_schema_and_exact79040_strict_loaded",
            "preselection_fixed1200_full_binding_reverified",
            "full1200_causal_weights_trusted_only_to_controller_pinned_sealed_training_execution",
            "authority_inner_five_views_re_materialized",
            "checkpoint_forward_reexecuted",
            "full_candidate_ledger_exact_match",
            "inner_pass",
        )
        for fold_index in range(m.OUTER_FOLDS):
            fit_iids = [f"fit-{fold_index}-0", f"fit-{fold_index}-1"]
            fit_digest = m._object_sha(fit_iids)
            checkpoint_file_sha = f"{fold_index + 1:x}" * 64
            state_sha = f"{fold_index + 6:x}" * 64
            inner_file_sha = str(fold_index) * 64
            pca_sha = f"{fold_index + 11:x}"[-1] * 64
            schedule_sha = f"{fold_index + 2:x}" * 64
            candidate_sha = f"{fold_index + 3:x}" * 64
            trace = {
                "file_sha256": f"{fold_index + 12:x}"[-1] * 64,
                "metadata_digest": f"{fold_index + 13:x}"[-1] * 64,
            }
            provenance = {
                "fold_index": fold_index,
                "model_fit_original_count": len(fit_iids),
                "model_fit_ordered_iids": fit_iids,
                "model_fit_iid_digest": fit_digest,
                "model_fit_original_tensor_sha256": "a" * 64,
                "model_fit_all_five_views_tensor_sha256": "b" * 64,
                "fixed_clip_pca_b384_all_five_views_tensor_sha256": "0" * 64,
                "clip_pca_fit_input_sha256": pca_sha,
                "clip_pca_fit_iid_digest": fit_digest,
                "clip_pca_clip_mean_sha256": "c" * 64,
                "clip_pca_clip_basis_sha256": "d" * 64,
                "clip_pca_diagnostics_sha256": "e" * 64,
                "fit_only_global_rms": 1.0,
                "fit_only_global_rms_sha256": "f" * 64,
                "fold_seed": m.Config().seed + 10000 + fold_index,
                "minibatch_generator_seed": m.Config().seed + 10001 + fold_index,
                "minibatch_schedule_shape": [
                    m.Config().max_steps, m.Config().batch_size,
                ],
                "minibatch_schedule_sha256": schedule_sha,
                "authority_model_fit_all_five_views_re_materialized": True,
                "pca_and_rms_recomputed_from_original_only": True,
                "minibatch_schedule_regenerated_without_training": True,
                "optimizer_steps_reexecuted": 0,
                "oof_semantic_tensor_read_count": 0,
                "materialization_audit_sha256": "9" * 64,
            }
            inner_replay = {
                "fold_index": fold_index,
                "fixed_candidate_ledger_sha256": candidate_sha,
                "inner_iid_digest": "8" * 64,
                "inner_evidence_sha256": "7" * 64,
                "complete_gate_sha256": "6" * 64,
                "bootstrap_seed_ledger_sha256": "5" * 64,
                "fixed1200_checkpoint_file_sha256": checkpoint_file_sha,
                "fixed1200_model_state_sha256": state_sha,
                "primal_dual_trace_file_sha256": trace["file_sha256"],
                "primal_dual_trace_metadata_digest": trace["metadata_digest"],
            }
            pair = {"fold_index": fold_index, "distinct": True}
            row = {
                "fold_index": fold_index,
                "inner_receipt_file_sha256": inner_file_sha,
                "fixed1200_checkpoint_file_sha256": checkpoint_file_sha,
                "fixed1200_model_state_sha256": state_sha,
                "primal_dual_trace_file_sha256": trace["file_sha256"],
                "primal_dual_trace_metadata_digest": trace["metadata_digest"],
                "checkpoint_outer_fold": fold_index,
                "runtime_fingerprint": runtime_fingerprint,
                "model_fit_provenance_replay": provenance,
                "model_fit_provenance_replay_sha256": m._object_sha(provenance),
                "inner_replay_binding": inner_replay,
                "inner_replay_sha256": m._object_sha(inner_replay),
                "preselection_fixed1200_pair_join_sha256": m._object_sha(pair),
                "full1200_optimizer_trajectory_reexecuted": False,
                "duplicate_training_performed": False,
                "inner_iid_digest": inner_replay["inner_iid_digest"],
                "inner_evidence_sha256": inner_replay["inner_evidence_sha256"],
                "complete_gate_sha256": inner_replay["complete_gate_sha256"],
                "bootstrap_seed_ledger_sha256": inner_replay[
                    "bootstrap_seed_ledger_sha256"
                ],
                "oof_semantic_tensor_read_count": 0,
            }
            row.update({key: True for key in required_true})
            fixed = {
                "file_sha256": checkpoint_file_sha,
                "model_state_sha256": state_sha,
                "model_fit_original_count": len(fit_iids),
                "model_fit_ordered_iids": fit_iids,
                "model_fit_iid_digest": fit_digest,
                "fixed_clip_pca_fit_input_sha256": pca_sha,
                "minibatch_schedule_sha256": schedule_sha,
                "primal_dual_trace_artifact": trace,
            }
            member = {
                "fold_index": fold_index,
                "fold_root": f"/sealed/fold-{fold_index}",
                "inner_receipt_binding": {
                    "file_sha256": inner_file_sha,
                    "primal_dual_trace_artifact": trace,
                },
                "preselection_checkpoint_artifact": {"fold": fold_index},
                "fixed1200_checkpoint_artifact": fixed,
                "primal_dual_trace_artifact": trace,
                "preselection_fixed1200_checkpoint_pair_join": pair,
                "fixed_candidate_ledger_sha256": candidate_sha,
                "independent_model_fit_provenance_replay_sha256": row[
                    "model_fit_provenance_replay_sha256"
                ],
                "independent_inner_replay_sha256": row["inner_replay_sha256"],
                "inner_pass": True,
                "oof_semantic_tensor_read_count_exact0": True,
            }
            replay.append(row)
            members.append(member)

        def barrier_value() -> dict:
            value = {
                "schema_version": m.BARRIER_SCHEMA,
                "status": m.BARRIER_PASS_STATUS,
                "authority": "burned_exposed_known_transform_development_only",
                "implementation": run_binding,
                "config": m._config_value(m.Config()),
                "config_sha256": m._object_sha(m._config_value(m.Config())),
                "authority_binding": authority_binding,
                "runtime_fingerprint": runtime_fingerprint,
                "replay_seed": m.Config().seed + 30000,
                "members": copy.deepcopy(members),
                "members_sha256": m._object_sha(members),
                "inner_transport_barrier_sha256": "4" * 64,
                "independent_replay_ledger": copy.deepcopy(replay),
                "independent_replay_sha256": m._object_sha(replay),
                "all_five_exact_one_full_gates_pass": True,
                "all_five_authority_model_fit_provenances_recomputed": True,
                "all_five_authority_inner_checkpoint_forwards_reexecuted": True,
                "all_five_primal_dual_trace_full_bindings_reverified": True,
                "oof_semantic_tensor_read_count": 0,
                "oof_semantic_tensor_read_count_exact0": True,
                "evaluate_fold_accepts_only_this_barrier_path_and_controller_expected_sha": True,
                "arbitrary_evaluate_fold_child_roots_or_inner_shas_accepted": False,
                "causal_training_trust_boundary": {
                    "full1200_optimizer_trajectory_reexecuted_by_verifier": False,
                    "duplicate_training_performed": False,
                    "causal_weights_trusted_only_to_official_controller_pinned_runtime_execution": True,
                    "barrier_expected_sha_must_be_supplied_by_detached_controller_not_untrusted_caller": True,
                    "inference_or_refit_authorized": False,
                },
                "qualification_scope": {
                    **m._qualification_scope(None),
                    "all_five_inner_fixed_candidate_gates_passed": True,
                    "aggregate_gate_evaluated": False,
                },
            }
            value["receipt_digest"] = m._object_sha(value)
            return value

        with mock.patch.object(
            m, "_barrier_authority_binding", return_value=authority_binding,
        ):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp).resolve() / "barrier.json"
                valid = barrier_value()
                file_sha = m._write_json_create_only(path, valid)
                loaded, binding = m._load_barrier_receipt_sealed(
                    str(path), file_sha, run_binding, {},
                )
                self.assertEqual(loaded, valid)
                self.assertEqual(binding["file_sha256"], file_sha)
                with self.assertRaisesRegex(RuntimeError, "expected SHA differs"):
                    m._load_barrier_receipt_sealed(
                        str(path), "0" * 64, run_binding, {},
                    )
            with tempfile.TemporaryDirectory() as tmp:
                forged = barrier_value()
                forged["independent_replay_ledger"][2][
                    "checkpoint_state_receipt_training_inner_join"
                ] = False
                forged["independent_replay_sha256"] = m._object_sha(
                    forged["independent_replay_ledger"]
                )
                unsigned = dict(forged)
                unsigned.pop("receipt_digest")
                forged["receipt_digest"] = m._object_sha(unsigned)
                path = Path(tmp).resolve() / "barrier.json"
                forged_sha = m._write_json_create_only(path, forged)
                with self.assertRaisesRegex(RuntimeError, "barrier replay differs"):
                    m._load_barrier_receipt_sealed(
                        str(path), forged_sha, run_binding, {},
                    )

    def test_fold_inner_oof_join_rejects_equal_size_cross_fold_swap(self) -> None:
        m = self.m
        count = m.FROZEN_OOF_COUNTS[3]
        self.assertEqual(count, m.FROZEN_OOF_COUNTS[4])
        fold3_iids = [f"fold3-{index:03d}" for index in range(count)]
        fold4_iids = [f"fold4-{index:03d}" for index in range(count)]

        def ledger(iids: list[str]) -> dict:
            return {
                "fold_index": 3,
                "oof_original_count": len(iids),
                "oof_ordered_iids": iids,
                "oof_iid_digest": m._object_sha(iids),
            }

        authoritative_inner = ledger(fold3_iids)
        m._verify_fold_inner_oof_join(
            ledger(fold3_iids), authoritative_inner
        )
        with self.assertRaisesRegex(
            RuntimeError, "authority-verified inner receipt"
        ):
            m._verify_fold_inner_oof_join(
                ledger(fold4_iids), authoritative_inner
            )
        hostile_count = ledger(fold3_iids)
        hostile_count["oof_original_count"] -= 1
        with self.assertRaisesRegex(
            RuntimeError, "authority-verified inner receipt"
        ):
            m._verify_fold_inner_oof_join(
                hostile_count, authoritative_inner
            )
        hostile_digest = ledger(fold3_iids)
        hostile_digest["oof_iid_digest"] = "0" * 64
        with self.assertRaisesRegex(
            RuntimeError, "authority-verified inner receipt"
        ):
            m._verify_fold_inner_oof_join(
                hostile_digest, authoritative_inner
            )

    def test_evaluate_one_failed_inner_keeps_oof_materialization_exact_zero(self) -> None:
        m = self.m
        args = SimpleNamespace(
            fold_index=0, device="cpu",
            barrier_receipt="/controller/barrier.json",
            expected_barrier_receipt_sha256="b" * 64,
        )
        barrier = {"members": [{
            "fold_root": f"/fold/{index}",
            "inner_receipt_binding": {"file_sha256": str(index) * 64},
        } for index in range(5)]}
        no_oof = mock.Mock()
        no_score = mock.Mock()
        no_replay = mock.Mock()
        with mock.patch.object(m, "_require_release_sealed"):
            with mock.patch.object(m, "_binding", return_value={}):
                with mock.patch.object(m.torch, "__version__", "2.7.1+rocm6.3"):
                    with mock.patch.object(m.frozen, "_prepare_authorities", return_value={}):
                        with mock.patch.object(
                            m, "_load_barrier_receipt_sealed",
                            return_value=(barrier, {"file_sha256": "b" * 64}),
                        ):
                            with mock.patch.object(
                                m, "_load_all_inner_receipts_or_fail_before_oof",
                                side_effect=RuntimeError("GLOBAL_INNER_NO_GO"),
                            ):
                                with mock.patch.object(
                                    m, "_independently_replay_all_inner_gates_before_oof",
                                    no_replay,
                                ):
                                    with mock.patch.object(
                                        m.frozen, "_selective_materialize_feature_rows", no_oof,
                                    ):
                                        with mock.patch.object(m, "_evaluate_rows_fixed", no_score):
                                            with self.assertRaisesRegex(
                                                RuntimeError, "GLOBAL_INNER_NO_GO"
                                            ):
                                                m.run_evaluate_fold(args)
        no_replay.assert_not_called()
        no_oof.assert_not_called()
        no_score.assert_not_called()


if __name__ == "__main__":
    unittest.main()

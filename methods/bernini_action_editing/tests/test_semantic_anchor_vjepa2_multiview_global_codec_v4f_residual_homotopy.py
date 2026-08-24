from __future__ import annotations

import ast
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
RUNTIME_PATH = (
    METHOD_ROOT
    / "semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy.py"
)
V4E_PATH = METHOD_ROOT / "semantic_anchor_vjepa2_multiview_global_codec_v4e_alt.py"


class ResidualHomotopyV4FTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNTIME_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def function_source(self, name: str) -> str:
        nodes = [
            node for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
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

    def test_source_compiles_normally_and_optimized(self) -> None:
        compile(self.source, str(RUNTIME_PATH), "exec")
        compile(self.source, str(RUNTIME_PATH), "exec", optimize=2)

    def test_is_independent_and_binds_exact_burned_v4e(self) -> None:
        self.assertNotEqual(RUNTIME_PATH, V4E_PATH)
        self.assertTrue(V4E_PATH.is_file())
        for digest in (
            "4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a",
            "76d5aaf4667ac7a99f26788faa3f205c360479836200f5abc4715d3a9afd7cee",
            "bd30bf71b509154847bba3a7a474a9e2ecfd38c13b04687970be6adec82b0d67",
            "10b5c8d2271353baf94633eeda9e359ef765271234d7d69486506fd32abdc25f",
            "c1639cee6151e7ad28adaecd27186c3dc70fe241cdb19c3cf905541648dc8d0d",
            "9dbd57b84b8e3498315536c8aff6d19123add300d1cb12cba134e073215cf33d",
        ):
            self.assertIn(digest, self.source)

    def test_release_is_sealed_and_public_entries_guard_first(self) -> None:
        self.assertIn("RELEASE_SEALED = True", self.source)
        self.assertNotIn("EXPECTED_V4F_CONTROLLER_SHA256", self.source)
        self.assertNotIn("EXPECTED_V4F_RELEASE_MANIFEST_SHA256", self.source)
        self.assertNotIn('"controller_sha256"', self.source)
        self.assertNotIn('"release_manifest_sha256"', self.source)
        guard = self.function_source("_require_release_sealed")
        self.assertIn("RELEASE_SEALED is not True", guard)
        self.assertIn("must not reverse-pin its controller", guard)
        for name in ("run_train_fold", "run_aggregate", "build_parser", "main"):
            node = next(
                item for item in ast.walk(self.tree)
                if isinstance(item, ast.FunctionDef) and item.name == name
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

    def test_exact7_rho_contract_is_literal(self) -> None:
        for token in (
            "1.0 / 64.0", "1.0 / 32.0", "1.0 / 16.0", "1.0 / 8.0",
            "1.0 / 4.0", "1.0 / 2.0", "1.0)",
            "RHO_COMPARATOR = 0.0", "TRAINING_RHO = 1.0",
            "FIXED_SELECTED_STEP = 1200",
        ):
            self.assertIn(token, self.source)
        validate = self.function_source("validate")
        self.assertIn("len(RHO_GRID) != 7", validate)
        self.assertIn("len(set(RHO_GRID)) != len(RHO_GRID)", validate)
        self.assertIn("any(type(rho) is not float for rho in RHO_GRID)", validate)
        self.assertIn("for left, right in zip(RHO_GRID, RHO_GRID[1:])", validate)
        self.assertIn("math.frexp(rho)[0] != 0.5", validate)
        self.assertIn("dtype=torch.float32", validate)
        self.assertIn("float(torch.tensor(rho", validate)
        self.assertIn("RHO_COMPARATOR in RHO_GRID", validate)

    def test_model_uses_fp32_rho_on_both_residuals_and_no_raw_blend(self) -> None:
        model = self.class_source("ClipPCAInitializedVJepaGlobalCodec")
        self.assertIn('"residual_gate_rho", torch.tensor([TRAINING_RHO], dtype=torch.float32)', model)
        self.assertEqual(model.count("* self.residual_gate_rho"), 2)
        self.assertIn("self.encoder_delta(self.encoder_norm(attended))", model)
        self.assertIn("residual * self.fit_only_rms * self.residual_gate_rho", model)
        self.assertIn("code.flatten(1) @ self.clip_basis.T", model)
        self.assertNotIn("raw_input", model)
        self.assertNotIn("raw_output", model)
        self.assertNotIn("MultiheadAttention", model)

    def test_rho_setter_accepts_only_exact_preregistered_floats(self) -> None:
        setter = self.function_source("set_residual_gate_rho")
        self.assertIn("type(rho) is not float", setter)
        self.assertIn("rho not in (RHO_COMPARATOR, *RHO_GRID)", setter)
        self.assertIn("dtype=torch.float32", setter)
        self.assertIn("float(fp32.item()) != rho", setter)

    def test_code_and_parameter_geometry_remain_exact(self) -> None:
        for token in (
            "CODE_TIME = 12", "CODE_CHANNELS = 32",
            "CODE_NUMEL = CODE_TIME * CODE_CHANNELS",
            "EXACT_TRAINABLE_PARAMETERS = 79040",
            "MAX_TRAINABLE_PARAMETERS = 150000",
        ):
            self.assertIn(token, self.source)
        model = self.class_source("ClipPCAInitializedVJepaGlobalCodec")
        self.assertIn("count != EXACT_TRAINABLE_PARAMETERS", model)
        self.assertIn("decoder input must be the sole [12,32] code", model)

    def test_training_is_rho1_fixed_step1200_without_inner_tensor(self) -> None:
        train = self.function_source("_train_fold_model")
        self.assertIn("inner_validation_iids: Sequence[str]", train)
        self.assertNotIn("inner_validation: Sequence[v4c.Record]", train)
        self.assertNotIn("validation_original =", train)
        self.assertNotIn("checkpoint_scores", train)
        self.assertNotIn("min(config.checkpoint_steps", train)
        self.assertIn("model.set_residual_gate_rho(TRAINING_RHO)", train)
        self.assertIn("selected_step = FIXED_SELECTED_STEP", train)
        self.assertIn('"checkpoint_winner_selection_performed": False', train)
        self.assertIn('"inner_validation_tensor_count_before_preselection_checkpoint_seal": 0', train)
        self.assertIn("fit_views.flatten(0, 1)", train)
        self.assertIn(
            '"model_fit_five_view_tensors_used_for_gradient_and_model_input": True',
            train,
        )
        self.assertIn(
            '"inner_validation_derived_view_tensor_count_used_during_training": 0',
            train,
        )
        self.assertNotIn('"inner_validation_derived_views_used": 0', train)

    def test_five_view_loss_has_per_iid_ten_pair_geometry(self) -> None:
        loss = self.function_source("_multiview_training_loss")
        for token in (
            "for left in range(len(EVAL_VIEWS))",
            "for right in range(left + 1, len(EVAL_VIEWS))",
            "len(teacher_distances) != 10",
            "teacher_geometry.detach().mean(dim=1, keepdim=True) + 1.0e-8",
            "(candidate_geometry - teacher_geometry) / per_iid_scale",
            "beta=0.1", "geometry_weight != 0.25",
        ):
            self.assertIn(token, loss)
        self.assertIn(
            "torch.sort(per_view).values.mean()",
            self.function_source("_single_view_reconstruction_loss"),
        )
        for forbidden in ("family", "NEGATIVES", "monotone_warp"):
            self.assertNotIn(forbidden, loss)

    def test_three_stage_read_barrier_is_lexically_ordered(self) -> None:
        run = self.function_source("_run_fold")
        positions = [
            run.index('stage="stage1_model_fit_only"'),
            run.index('checkpoint_role="preselection_fixed_step1200"'),
            run.index('stage="stage2_post_preselection_seal_inner_five_views"'),
            run.index("_select_fold_local_rho("),
            run.index('checkpoint_role="selected_fold_local_rho"'),
            run.index('stage="stage3_post_selected_seal_oof"'),
            run.index("_evaluate_fold("),
        ]
        self.assertEqual(positions, sorted(positions))
        stage1_request = run[run.index("stage1_request ="):positions[0]]
        self.assertIn("for iid in fit_iids", stage1_request)
        self.assertNotIn("validation_iids", stage1_request)
        self.assertNotIn("oof_iids", stage1_request)

    def test_no_pass_has_no_selected_save_oof_read_or_evaluation(self) -> None:
        run = self.function_source("_run_fold")
        start = run.index("if selected_rho is None:")
        end = run.index("else:", start)
        branch = run[start:end]
        self.assertNotIn("_save_checkpoint_create_only", branch)
        self.assertNotIn("_selective_materialize_feature_rows", branch)
        self.assertNotIn("_evaluate_fold", branch)
        self.assertIn('"semantic_tensor_materialized_count": 0', branch)
        self.assertIn('"oof_semantic_tensor_read_count_exact0": True', branch)
        self.assertIn("fold_status = INNER_NO_GO_STATUS", branch)

    def test_inner_full_gate_has_all_required_strict_terms(self) -> None:
        gate = self.function_source("_inner_candidate_gate")
        for token in (
            "_paired_ratio_ucb", '"both_point_ratios_le_1p05"',
            '"both_ucbs_le_1p05"', "teacher_values", "candidate_values",
            "candidate - config.teacher_retention * teacher", "candidate - baseline",
            "_positive_point_and_lcb_gate(teacher)",
            "_positive_point_and_lcb_gate(candidate)",
            "_positive_point_and_lcb_gate(retention)",
            "_positive_point_and_lcb_gate(improvement)",
            '"teacher_fixed_gate_included": True',
        ):
            self.assertIn(token, gate)
        self.assertIn("teacher-fixed", gate)

    def test_exact7_scan_evaluates_all_then_selects_first_pass(self) -> None:
        scan = self.function_source("_select_fold_local_rho")
        self.assertIn("for ordinal, rho in enumerate(RHO_GRID)", scan)
        self.assertIn("if selected_rho is None and passed", scan)
        self.assertIn("len(candidates) != len(RHO_GRID)", scan)
        self.assertIn("fixed_teacher_and_clip_pca_reference_sha256", scan)
        for literal in (
            '"rho_candidate_count": len(RHO_GRID)', '"single_candidate": False',
            '"rho0_selectable": False',
            '"monotonic_metric_behavior_assumed": False',
            '"smallest_rho_minimizes_distortion_claimed": False',
            '"cross_fold_inner_metric_aggregation_or_global_rho_selection": False',
            '"transform_role_and_family_metadata_used_for_hyperparameter_selection": True',
            '"transform_role_and_family_metadata_used_for_gradient": False',
            '"transform_role_and_family_metadata_used_for_model_input": False',
        ):
            self.assertIn(literal, scan)

    def test_rho_ledger_recomputes_evidence_gates_seeds_and_fixed_reference(self) -> None:
        verify = self.function_source("_verify_rho_selection_ledger")
        for token in (
            "_inner_candidate_gate(", "candidate.get(\"gate\") != recomputed_gate",
            "_bootstrap_seed_ledger(recomputed_gate)",
            "len(candidate.get(\"bootstrap_seed_ledger\", [])) != 34",
            "expected_selected = pass_rhos[0] if pass_rhos else None",
            "_fixed_inner_reference_projection(evidence)",
            "rho0_inner_evidence_sha256", "list(range(len(RHO_GRID)))",
        ):
            self.assertIn(token, verify)

    def test_checkpoint_binds_role_rho_base_and_preselection_file(self) -> None:
        save = self.function_source("_save_checkpoint_create_only")
        loader = self.function_source("_load_selected_checkpoint_sealed")
        combined = save + loader
        for token in (
            "CHECKPOINT_SCHEMA", '"checkpoint_role"', '"deployment_rho"',
            '"preselection_base_state_sha256"', '"preselection_checkpoint_binding"',
            "_base_state_sha(state)", "template.load_state_dict(state, strict=True)",
            '"residual_gate_rho"',
        ):
            self.assertIn(token, combined)
        self.assertIn('checkpoint_role == "selected_fold_local_rho"', save)
        self.assertIn('checkpoint_role == "preselection_fixed_step1200"', save)

    def test_pass_branch_hard_gates_distinct_checkpoint_pair(self) -> None:
        pair = self.function_source("_verify_distinct_checkpoint_pair")
        run = self.function_source("_run_fold")
        for token in (
            'preselection.get("path") == selected.get("path")',
            'pre_physical.get("device")', 'pre_physical.get("inode")',
            'selected_physical.get("device")', 'selected_physical.get("inode")',
            'preselection.get("preselection_base_state_sha256")',
            'selected.get("preselection_base_state_sha256")',
            'selected.get("deployment_rho") != selected_rho',
            '"fresh_reload_strict_state_verified"',
            '"fresh_reload_output_bit_exact"',
            '"distinct_device_inode_pair": True',
            '"selected_rho_strict_reload_verified": True',
        ):
            self.assertIn(token, pair)
        self.assertIn("checkpoint_pair_join = _verify_distinct_checkpoint_pair(", run)
        self.assertLess(
            run.index("checkpoint_pair_join = _verify_distinct_checkpoint_pair("),
            run.index('stage="stage3_post_selected_seal_oof"'),
        )

    def test_checkpoint_bytes_are_bound_before_torch_parse(self) -> None:
        loader = self.function_source("_load_selected_checkpoint_sealed")
        for token in (
            "os.O_NOFOLLOW", "digest_before", "expected_file_sha",
            "checkpoint expected binding differs before torch parse",
            "weights_only=True", "digest_after", "physical_identity",
        ):
            self.assertIn(token, loader)
        self.assertLess(
            loader.index("checkpoint expected binding differs before torch parse"),
            loader.index("payload = torch.load"),
        )
        self.assertEqual(loader.count("os.open("), 1)

    def test_artifact_names_include_preselection_and_selected(self) -> None:
        roots = self.function_source("_resolve_fold_root")
        for token in ('root / "fold.json"', 'root / "preselection.pt"', 'root / "selected.pt"'):
            self.assertIn(token, roots)
        train = self.function_source("run_train_fold")
        self.assertIn("_load_selected_checkpoint_sealed(\n        preselection_path", train)
        self.assertIn("if inner_pass:", train)

    def test_selective_loader_allows_only_three_stages(self) -> None:
        loader = self.function_source("_selective_materialize_feature_rows")
        for token in (
            '"stage1_model_fit_only"',
            '"stage2_post_preselection_seal_inner_five_views"',
            '"stage3_post_selected_seal_oof"', "FakeTensorMode",
            "weights_only=True", "_checkpoint_offset", "os.pread",
            "FULL_NUMEL * 4", "digest_before", "digest_after",
            "all_fake_tensor_offsets_unique_nonoverlapping_in_file",
            "unrequested_tensor_storage_materialized_count",
        ):
            self.assertIn(token, loader)
        self.assertNotIn("untyped_storage().nbytes()", loader)

    def test_three_stage_ledger_replays_labels_counts_maps_and_reload_flags(self) -> None:
        verify = self.function_source("_verify_fold_selective_materialization_ledger")
        for token in (
            "stage1_request = {iid: list(EVAL_VIEWS) for iid in fit_iids}",
            "stage2_request = {iid: list(EVAL_VIEWS) for iid in inner_iids}",
            "if inner_pass else {}", "expected_stage3_counts",
            "stage1_model_fit_only", "stage2_post_preselection_seal_inner_five_views",
            "stage3_post_selected_seal_oof", "stage3_inner_no_go_oof_unread",
            "caller_model_reloaded_from_sealed_artifact_before_next_stage",
            "inner_no_go_oof_semantic_tensor_read_count_exact0",
            "_verify_rho_selection_ledger(fold)",
        ):
            self.assertIn(token, verify)

    def test_fold_loader_replays_no_go_and_oof_rho_join(self) -> None:
        loader = self.function_source("_load_fold_receipt_sealed")
        for token in (
            "object_pairs_hook=_reject_duplicate_json_pairs",
            "parse_constant=_reject_nonfinite_json", "INNER_NO_GO_STATUS",
            "not checkpoint_path.exists()", "not checkpoint_path.is_symlink()",
            'row.get("residual_gate_rho") != fold.get("selected_rho")',
            'row.get("rho0_exact_clip_pca_alias_used") is not False',
            "_verify_fold_selective_materialization_ledger(fold)",
        ):
            self.assertIn(token, loader)

    def test_aggregate_fails_before_oof_union_on_any_inner_no_go(self) -> None:
        aggregate = self.function_source("run_aggregate")
        self.assertLess(
            aggregate.index("aggregate fail-closed"),
            aggregate.index('evidence = [row for receipt in receipts'),
        )
        self.assertIn('receipt.get("status") != STATUS', aggregate)
        self.assertIn('get("inner_pass") is not True', aggregate)
        for forbidden in ("_train_fold_model", "_run_fold(", "_resolve_device"):
            self.assertNotIn(forbidden, aggregate)
        self.assertIn('"aggregate_device": "cpu"', aggregate)

    def test_aggregate_replays_both_checkpoint_sets_and_authoritative_splits(self) -> None:
        aggregate = self.function_source("run_aggregate")
        self.assertGreaterEqual(aggregate.count("_verify_checkpoint_artifacts("), 6)
        self.assertIn('expected_role="preselection_fixed_step1200"', aggregate)
        self.assertIn('expected_role="selected_fold_local_rho"', aggregate)
        self.assertIn("_verify_fold_split_against_authority", aggregate)
        self.assertIn("_verify_fold_selective_materialization_ledger", aggregate)
        self.assertGreaterEqual(aggregate.count("_load_fold_receipt_sealed"), 2)
        replay = self.function_source("_verify_fold_split_against_authority")
        self.assertIn("inner_iid_family", replay)
        self.assertIn("embedded_inner_populations", replay)

    def test_final_oof_bootstrap_defaults_to_v4e_seed_namespace(self) -> None:
        lcb = self.function_source("_paired_lcb")
        ratio = self.function_source("_paired_ratio_ucb")
        self.assertIn('namespace: str = "v4e"', lcb)
        self.assertIn('namespace: str = "v4e"', ratio)
        self.assertIn('f"{namespace}:{label}"', lcb)
        self.assertIn('config, namespace, label, "ratio", "clip"', ratio)
        aggregate = self.function_source("_aggregate")
        self.assertNotIn("namespace=", aggregate)

    def test_receipts_are_honest_about_exact7_known_exposure_and_scope(self) -> None:
        aggregate = self.function_source("run_aggregate")
        for literal in (
            '"rho_candidate_count": len(RHO_GRID)', '"single_candidate": False',
            '"model_fit_five_view_tensors_used_for_gradient_and_model_input": True',
            '"inner_five_view_tensors_used_for_hyperparameter_selection": True',
            '"inner_five_view_tensors_used_for_gradient_or_model_input": False',
            '"transform_role_and_family_metadata_used_for_hyperparameter_selection": True',
            '"transform_role_and_family_metadata_used_for_gradient": False',
            '"transform_role_and_family_metadata_used_for_model_input": False',
            '"teacher_and_fixed_pca_metadata_used_for_hyperparameter_selection": True',
            '"teacher_and_fixed_pca_metadata_used_for_gradient_or_model_input": False',
            '"each_outer_fold_selected_rho_independently": True',
            '"cross_fold_inner_aggregation_or_global_rho_selection": False',
            '"known_exposed_transform_families_only": True',
            '"unseen_hostile_transform_gate": False',
            '"unseen_hostile_transform_gate_evaluated": False',
            '"action_representation_qualified": False',
            '"identity_preservation_qualified": False',
            '"generation_qualified": False', '"renderer_qualified": False',
            '"video_editing_qualified": False', '"inference_authorized": False',
            '"full644_refit_authorized": False',
        ):
            self.assertIn(literal, aggregate)

    def test_only_train_fold_and_aggregate_cli_commands_exist(self) -> None:
        self.assertEqual(self.source.count('add_parser("train-fold")'), 1)
        self.assertEqual(self.source.count('add_parser("aggregate")'), 1)
        self.assertEqual(self.source.count("add_parser("), 2)
        self.assertNotIn("run-exact5", self.source)

    def test_torch_dynamic_no_pass_and_pair_spies_are_statically_pinned(self) -> None:
        test_source = Path(__file__).read_text(encoding="utf-8")
        test_tree = ast.parse(test_source)

        def local_function(name: str) -> str:
            nodes = [
                node for node in ast.walk(test_tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            ]
            self.assertEqual(len(nodes), 1, name)
            return ast.get_source_segment(test_source, nodes[0]) or ""

        no_pass = local_function("test_no_pass_event_spy_has_exact7_and_zero_oof_calls")
        for token in (
            "_run_fold(", "_verify_rho_selection_ledger(fold)",
            "stage3_post_selected_seal_oof", "selected_path.exists()",
            "selected_path.is_symlink()", "evaluate_oof.assert_not_called()",
            "[m.RHO_COMPARATOR, *m.RHO_GRID]", "list(range(7)) * 2",
        ):
            self.assertIn(token, no_pass)
        pair = local_function(
            "test_distinct_checkpoint_pair_requires_inode_base_rho_and_reload"
        )
        self.assertGreaterEqual(pair.count("_verify_distinct_checkpoint_pair("), 4)
        self.assertIn('wrong_base["preselection_base_state_sha256"]', pair)
        self.assertIn('not_reloaded["fresh_reload_strict_state_verified"]', pair)

    def test_no_assert_statements_guard_contracts(self) -> None:
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(self.tree)))


@unittest.skipIf(torch is None, "PyTorch is unavailable in the local interpreter")
class ResidualHomotopyV4FDynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from methods.bernini_action_editing import (
            semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy
            as runtime,
        )
        cls.m = runtime

    def _evidence(self, iid: str, family: str, rho: float):
        alias = rho == self.m.RHO_COMPARATOR
        return [{
            "iid": iid,
            "family": family,
            "teacher_margin_by_negative": {
                name: 1.0 for name in self.m.NEGATIVES
            },
            "clip_pca_b384_margin_by_negative": {
                name: 0.5 for name in self.m.NEGATIVES
            },
            "candidate_margin_by_negative": {
                name: 0.25 for name in self.m.NEGATIVES
            },
            "raw_reconstruction_by_view": {
                view: {
                    "candidate_raw_mse": 0.5 if alias else 0.4,
                    "clip_pca_b384_raw_mse": 0.5,
                }
                for view in self.m.EVAL_VIEWS
            },
            "residual_gate_rho": rho,
            "rho0_exact_clip_pca_alias_used": alias,
        }]

    def test_no_pass_event_spy_has_exact7_and_zero_oof_calls(self) -> None:
        m = self.m
        fit = SimpleNamespace(iid="fit", family="family", strict=False, views={})
        inner = SimpleNamespace(iid="inner", family="family", strict=False, views={})
        oof = SimpleNamespace(iid="oof", family="family", strict=False, views={})
        fit_iids = [fit.iid]
        inner_iids = [inner.iid]
        oof_iids = [oof.iid]
        split = {
            "outer_assignment_digest": "outer",
            "outer_oof_iid_digest": m._object_sha(oof_iids),
            "model_fit_iid_digest": m._object_sha(fit_iids),
            "inner_validation_iid_digest": m._object_sha(inner_iids),
        }
        groups = {
            "model_fit": [fit], "inner_validation": [inner],
            "exploratory_oof": [oof],
        }
        events = []

        def materialize(_index, request, *, stage):
            events.append(("materialize", stage))
            if stage == "stage3_post_selected_seal_oof":
                self.fail("OOF materializer was called on INNER_NO_GO")
            rows = {iid: {"fit": fit, "inner": inner}[iid] for iid in request}
            counts = {
                view: sum(view in tuple(views) for views in request.values())
                for view in m.EVAL_VIEWS
            }
            return rows, {
                "stage": stage,
                "requested_iid_cluster_count": len(request),
                "semantic_tensor_materialized_count": sum(counts.values()),
                "semantic_tensor_materialized_count_by_view": counts,
                "requested_iid_view_map_sha256": m._object_sha({
                    iid: list(request[iid]) for iid in sorted(request)
                }),
                "unrequested_tensor_storage_materialized_count": 0,
            }

        fitted = SimpleNamespace(
            fit_iid_digest=m._object_sha(fit_iids),
            fit_input_sha256="f" * 64,
            diagnostics={},
        )
        model = SimpleNamespace()
        training = {
            "selected_step": m.FIXED_SELECTED_STEP,
            "selected_state_sha256": "s" * 64,
            "final_step_base_state_sha256": "b" * 64,
            "model_fit_ordered_iids": fit_iids,
            "model_fit_iid_digest": m._object_sha(fit_iids),
            "inner_validation_ordered_iids": inner_iids,
            "inner_validation_original_count": 1,
            "inner_validation_iid_digest": m._object_sha(inner_iids),
        }

        def train(*_args, **_kwargs):
            events.append(("train", "rho1_step1200"))
            return model, m.FIXED_SELECTED_STEP, training

        preselection = {
            "checkpoint_role": "preselection_fixed_step1200",
            "deployment_rho": m.TRAINING_RHO,
            "preselection_base_state_sha256": "b" * 64,
            "caller_model_reloaded_from_sealed_artifact_before_next_stage": True,
        }

        def save(*_args, **kwargs):
            events.append(("save", kwargs["checkpoint_role"]))
            if kwargs["checkpoint_role"] != "preselection_fixed_step1200":
                self.fail("selected checkpoint save was called on INNER_NO_GO")
            events.append(("seal_reload", kwargs["checkpoint_role"]))
            return preselection

        gate = {
            "complete_candidate_dependent_inner_gate": False,
            "fixed_seeds": {
                f"seed_{index:02d}": {"seed": index} for index in range(34)
            },
        }

        def evaluate(_rows, _model, rho, *_args):
            events.append(("rho", rho))
            return self._evidence(inner.iid, inner.family, rho)

        def inner_gate(*_args, **_kwargs):
            events.append(("gate", _kwargs["rho_ordinal"]))
            return gate

        evaluate_oof = mock.Mock(side_effect=AssertionError("OOF evaluation called"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            preselection_path = root / "preselection.pt"
            selected_path = root / "selected.pt"
            with (
                mock.patch.object(m, "_split_fold", return_value=(groups, split)),
                mock.patch.object(m.v4c, "OUTER_ASSIGNMENT_DIGEST", "outer"),
                mock.patch.object(m.v4c, "FOLD_IID_DIGESTS", {0: "fold"}),
                mock.patch.object(m, "FROZEN_OOF_COUNTS", (1, 1, 1, 1, 1)),
                mock.patch.object(m, "_selective_materialize_feature_rows", side_effect=materialize),
                mock.patch.object(m, "_fit_clip_pca_b384", return_value=fitted),
                mock.patch.object(m, "_train_fold_model", side_effect=train),
                mock.patch.object(m, "_save_checkpoint_create_only", side_effect=save),
                mock.patch.object(m, "_state_to_cpu", return_value={}),
                mock.patch.object(m, "_base_state_sha", return_value="b" * 64),
                mock.patch.object(m, "_evaluate_rows_at_rho", side_effect=evaluate),
                mock.patch.object(m, "_inner_candidate_gate", side_effect=inner_gate),
                mock.patch.object(m, "_evaluate_fold", evaluate_oof),
            ):
                fold, evidence = m._run_fold(
                    [fit, inner, oof], {},
                    {"folds": [{"oof_iid_digest": m._object_sha(oof_iids)}]},
                    0, m.Config(), torch.device("cpu"), preselection_path,
                    selected_path, {"implementation_sha256": "i" * 64}, {},
                )
                m._verify_rho_selection_ledger(fold)

            labels = [event for event in events if event[0] in {"materialize", "save"}]
            self.assertEqual(labels[:3], [
                ("materialize", "stage1_model_fit_only"),
                ("save", "preselection_fixed_step1200"),
                ("materialize", "stage2_post_preselection_seal_inner_five_views"),
            ])
            self.assertLess(
                events.index(("seal_reload", "preselection_fixed_step1200")),
                events.index((
                    "materialize",
                    "stage2_post_preselection_seal_inner_five_views",
                )),
            )
            self.assertEqual(
                [value for kind, value in events if kind == "rho"],
                [m.RHO_COMPARATOR, *m.RHO_GRID],
            )
            self.assertEqual(
                [value for kind, value in events if kind == "gate"],
                list(range(7)) * 2,
            )
            self.assertEqual(len(fold["rho_selection"]["candidates"]), 7)
            self.assertIsNone(fold["selected_checkpoint_artifact"])
            self.assertIsNone(fold["preselection_selected_checkpoint_pair_join"])
            self.assertEqual(fold["oof_semantic_tensor_materialized_count"], 0)
            self.assertTrue(fold["oof_semantic_tensor_read_count_exact0_on_inner_no_go"])
            stage3 = fold["selective_feature_materialization"][
                "stage3_only_after_selected_checkpoint_strong_seal_reload_or_no_go"
            ]
            self.assertEqual(stage3["stage"], "stage3_inner_no_go_oof_unread")
            self.assertEqual(stage3["requested_iid_cluster_count"], 0)
            self.assertEqual(stage3["semantic_tensor_materialized_count"], 0)
            self.assertEqual(stage3["requested_iid_view_map_sha256"], m._object_sha({}))
            self.assertTrue(stage3["oof_semantic_tensor_read_count_exact0"])
            self.assertEqual(evidence, [])
            self.assertFalse(selected_path.exists())
            self.assertFalse(selected_path.is_symlink())
            evaluate_oof.assert_not_called()

    def test_distinct_checkpoint_pair_requires_inode_base_rho_and_reload(self) -> None:
        m = self.m
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            pre_path = root / "preselection.pt"
            selected_path = root / "selected.pt"
            pre_path.write_bytes(b"pre")
            selected_path.write_bytes(b"selected")
            pre_stat = pre_path.stat()
            selected_stat = selected_path.stat()

            def artifact(path, value_stat, role, rho):
                return {
                    "path": str(path),
                    "file_sha256": ("a" if role.startswith("pre") else "b") * 64,
                    "checkpoint_role": role,
                    "deployment_rho": rho,
                    "preselection_base_state_sha256": "c" * 64,
                    "preselection_checkpoint_file_sha256": (
                        None if role.startswith("pre") else "a" * 64
                    ),
                    "physical_identity": {
                        "device": value_stat.st_dev, "inode": value_stat.st_ino,
                        "size_bytes": value_stat.st_size,
                    },
                    "semantic_metadata_state_replay_verified": True,
                    "fresh_reload_strict_state_verified": True,
                    "fresh_reload_output_bit_exact": True,
                    "caller_model_reloaded_from_sealed_artifact_before_next_stage": True,
                }

            pre = artifact(
                pre_path, pre_stat, "preselection_fixed_step1200", m.TRAINING_RHO
            )
            selected = artifact(
                selected_path, selected_stat, "selected_fold_local_rho", m.RHO_GRID[0]
            )
            replay = m._verify_distinct_checkpoint_pair(pre, selected, m.RHO_GRID[0])
            self.assertTrue(replay["distinct_device_inode_pair"])
            self.assertTrue(replay["same_preselection_base_state_sha256"])
            self.assertTrue(replay["selected_rho_strict_reload_verified"])

            same_inode = dict(selected)
            same_inode["physical_identity"] = dict(pre["physical_identity"])
            with self.assertRaises(RuntimeError):
                m._verify_distinct_checkpoint_pair(pre, same_inode, m.RHO_GRID[0])
            wrong_base = dict(selected)
            wrong_base["preselection_base_state_sha256"] = "d" * 64
            with self.assertRaises(RuntimeError):
                m._verify_distinct_checkpoint_pair(pre, wrong_base, m.RHO_GRID[0])
            not_reloaded = dict(selected)
            not_reloaded["fresh_reload_strict_state_verified"] = False
            with self.assertRaises(RuntimeError):
                m._verify_distinct_checkpoint_pair(pre, not_reloaded, m.RHO_GRID[0])


if __name__ == "__main__":
    unittest.main()

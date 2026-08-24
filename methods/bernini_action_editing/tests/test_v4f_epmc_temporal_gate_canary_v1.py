#!/usr/bin/env python3
"""Static and optional tensor tests for the unsealed v4-F/EPMC canary core."""

from __future__ import annotations

import ast
import argparse
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
METHOD_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_PATH = METHOD_ROOT / "materialize_v4f_epmc_gate_state_v1.py"
RUNTIME_PATH = METHOD_ROOT / "infer_v4f_epmc_temporal_gate_canary_v1.py"
BUILDER_PATH = METHOD_ROOT / "tools/build_v4f_epmc_temporal_gate_review_v1.py"
CONTROLLER_PATH = (
    METHOD_ROOT / "scripts/auh_v4f_epmc_temporal_gate_video_canary_v1.sh"
)
PRODUCTION_PATHS = (MATERIALIZER_PATH, RUNTIME_PATH, BUILDER_PATH)

try:
    import torch
except ImportError:  # The local static audit environment intentionally lacks torch.
    torch = None


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one function {name}, found {len(matches)}")
    return matches[0]


def _first_statement_calls_guard(function: ast.FunctionDef) -> bool:
    if not function.body:
        return False
    first = function.body[0]
    if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Call):
        # Assignment from the guard is also allowed when its return is needed.
        if not isinstance(first, (ast.Assign, ast.AnnAssign)):
            return False
        value = first.value
        if not isinstance(value, ast.Call):
            return False
    else:
        value = first.value
    return isinstance(value.func, ast.Name) and value.func.id == "_require_release_sealed"


class V4FEPMCTemporalGateStaticTests(unittest.TestCase):
    def test_future_use_files_parse_and_controller_is_absent(self) -> None:
        for path in PRODUCTION_PATHS:
            self.assertTrue(path.is_file(), path)
            _tree(path)
        self.assertFalse(
            CONTROLLER_PATH.exists(),
            "controller must remain absent until final pins exist",
        )

    def test_release_placeholders_fail_closed(self) -> None:
        for path in PRODUCTION_PATHS:
            source = _source(path)
            self.assertIn('PIN_PLACEHOLDER = "TO_BE_PINNED"', source)
            self.assertIn("RELEASE_SEALED = False", source)
            self.assertIn("def _require_release_sealed", source)
        materializer = _source(MATERIALIZER_PATH)
        for name in (
            "EXPECTED_V4F_RECEIPT_SCHEMA",
            "EXPECTED_V4F_RECEIPT_STATUS",
            "EXPECTED_V4F_FOLD_RECEIPT_SCHEMA",
            "EXPECTED_V4F_FOLD_RECEIPT_STATUS",
            "EXPECTED_V4F_CHECKPOINT_SCHEMA",
            "EXPECTED_V4F_IMPLEMENTATION_SHA256",
            "EXPECTED_V4F_RECEIPT_FILE_SHA256",
            "EXPECTED_V4F_RECEIPT_SELF_DIGEST",
            "EXPECTED_FOLD1_RECEIPT_SHA256",
            "EXPECTED_FOLD1_RECEIPT_SELF_DIGEST",
            "EXPECTED_FOLD1_PRESELECTION_CHECKPOINT_SHA256",
            "EXPECTED_FOLD1_PRESELECTION_CHECKPOINT_METADATA_DIGEST",
            "EXPECTED_FOLD1_PRESELECTION_MODEL_STATE_SHA256",
            "EXPECTED_FOLD1_PRESELECTION_BASE_STATE_SHA256",
            "EXPECTED_FOLD1_CHECKPOINT_SHA256",
            "EXPECTED_FOLD1_CHECKPOINT_METADATA_DIGEST",
            "EXPECTED_FOLD1_MODEL_STATE_SHA256",
        ):
            self.assertIn(f"{name} = PIN_PLACEHOLDER", materializer)
        self.assertIn(
            "EXPECTED_FOLD1_SELECTED_RHO: Any = PIN_PLACEHOLDER", materializer
        )
        runtime = _source(RUNTIME_PATH)
        for name in (
            "EXPECTED_GATE_STATE_SCHEMA",
            "EXPECTED_GATE_STATE_STATUS",
            "EXPECTED_GATE_STATE_FILE_SHA256",
            "EXPECTED_GATE_STATE_SELF_DIGEST",
            "EXPECTED_GATE_MATERIALIZER_SHA256",
            "EXPECTED_V4B_RUNTIME_IMPLEMENTATION_SHA256",
        ):
            self.assertIn(f"{name} = PIN_PLACEHOLDER", runtime)
        self.assertIn(
            "EXPECTED_V4B_RUNTIME_IMPLEMENTATION_SHA256 = PIN_PLACEHOLDER",
            runtime,
        )
        builder = _source(BUILDER_PATH)
        for name in (
            "EXPECTED_RUNTIME_IMPLEMENTATION_SHA256",
            "EXPECTED_RUNTIME_RECEIPT_SCHEMA",
            "EXPECTED_RUNTIME_RECEIPT_STATUS",
        ):
            self.assertIn(f"{name} = PIN_PLACEHOLDER", builder)

    def test_all_cli_and_run_entries_guard_first(self) -> None:
        required = {
            MATERIALIZER_PATH: ("build_parser", "run", "main"),
            RUNTIME_PATH: (
                "load_gate_state",
                "build_video_receipt",
                "build_parser",
                "validate_cli",
                "run",
                "main",
            ),
            BUILDER_PATH: ("_strict_receipt", "build_parser", "run", "main"),
        }
        for path, names in required.items():
            tree = _tree(path)
            for name in names:
                self.assertTrue(
                    _first_statement_calls_guard(_function(tree, name)),
                    f"{path.name}:{name} must guard as its first statement",
                )

    def test_no_optimized_mode_assert_contracts(self) -> None:
        for path in PRODUCTION_PATHS:
            asserts = [node for node in ast.walk(_tree(path)) if isinstance(node, ast.Assert)]
            self.assertEqual(asserts, [], f"{path.name} contains optimized-away assert")

    def test_future_canary_contains_no_launcher_or_remote_execution_surface(self) -> None:
        for path in PRODUCTION_PATHS:
            source = _source(path)
            for forbidden in (
                "subprocess.", "os.system(", "ssh ", "sbatch ", "srun ",
                "torchrun ",
            ):
                self.assertNotIn(forbidden, source, f"{path.name}: {forbidden}")

    def test_exact_oof_row_and_authorities_are_frozen(self) -> None:
        source = _source(MATERIALIZER_PATH)
        for fragment in (
            'EXPECTED_IID = "7b88a1ca1f804f41"',
            'EXPECTED_FAMILY = "sit_down"',
            "EXPECTED_OUTER_FOLD = 1",
            "EXPECTED_FEATURE_ORDINAL = 465",
            "EXPECTED_FEATURE_SHARD = 3",
            "18c7ad8a24f678ea93cc9d16365fcba0cb8d101667eed9542618240f3ed9c13f",
            "8e0034fcc4a53c8220390df08e2361080dcd918990629a05c297211dd2bb6637",
            "5ab9704f456768b440c966a53328de0c1a67836548f8f8ebd92e50d21846ab5f",
            "678985f2cf0cd0244c707e9ab7f9c6b8116aac5ede6f0a56fe78b92eeb582400",
            "568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2",
            "8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9",
            "loaded_original_sequence_sha256_verified",
        ):
            self.assertIn(fragment, source)

    def test_only_true_aggregate_gate_can_reach_materialization(self) -> None:
        materializer = _source(MATERIALIZER_PATH)
        runtime = _source(RUNTIME_PATH)
        for fragment in (
            'metrics.get("exposed_five_view_codec_development_gate") is not True',
            'scope.get("exposed_five_view_codec_development_gate") is True',
            'scope.get("unseen_hostile_transform_gate") is False',
            'scope.get("unseen_hostile_transform_gate_evaluated") is False',
            '"v4c_burned_oof_informed_clip_pca_b384_choice"',
            '"v4e_burned_oof_informed_v4f_homotopy_choice"',
            '"one_predeclared_fold_local_selection_algorithm"',
            '"rho_candidate_count"',
            '"single_candidate"',
            '"v4f_aggregate_gate_verified_true": True',
            '"v4f_exact5_all_fold_inner_gates_passed": True',
        ):
            self.assertIn(fragment, materializer)
        self.assertIn(
            'payload.get("v4f_aggregate_gate_verified_true") is not True', runtime
        )
        self.assertIn("expected_sha256 != EXPECTED_GATE_STATE_FILE_SHA256", runtime)
        self.assertIn("digest != EXPECTED_GATE_STATE_SELF_DIGEST", runtime)
        self.assertIn('"v4f_aggregate_gate_verified_true": True', runtime)

    def test_exact5_no_go_and_false_gate_are_fail_closed_before_html(self) -> None:
        materializer = _source(MATERIALIZER_PATH)
        runtime = _source(RUNTIME_PATH)
        builder = _source(BUILDER_PATH)
        for fragment in (
            'metrics.get("exposed_five_view_codec_development_gate") is not True',
            'fold.get("rho_selection", {}).get("inner_pass") is not True',
            'fold.get("fold_status") != EXPECTED_V4F_FOLD_RECEIPT_STATUS',
            'len(folds) != 5',
            'selected.get("count") != 5',
            'preselected.get("count") != 5',
        ):
            self.assertIn(fragment, materializer)
        self.assertIn(
            'payload.get("v4f_exact5_all_fold_inner_gates_passed") is not True',
            runtime,
        )
        self.assertIn(
            'value.get("v4f_exact5_all_fold_inner_gates_passed") is not True',
            builder,
        )
        self.assertIn("任一 INNER_NO_GO", builder)

    def test_decoded_residual_and_fit_only_temporal_mapping_are_explicit(self) -> None:
        source = _source(MATERIALIZER_PATH)
        for fragment in (
            "R=C(D(E(C(anchor))))-C(D(0))",
            "TIME_STEPS = 32",
            "FEATURE_DIM = 1024",
            "CODE_TIME = 12",
            "CODE_CHANNELS = 32",
            "P95_QUANTILE = 0.95",
            "fit_iids = list(fold_contract[\"model_fit_ordered_iids\"])",
            "inner_validation_or_oof_values_used_for_scale",
            '"torch linear size=20 align_corners=True"',
            '"0.10*0.5*profile20=0.05*profile20"',
        ):
            self.assertIn(fragment, source)
        self.assertIn("align_corners=True", source)
        self.assertIn("EXPECTED_IID in fit_iids", source)

    def test_v4f_global_codec_and_three_way_authority_join_are_explicit(self) -> None:
        source = _source(MATERIALIZER_PATH)
        for fragment in (
            "semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy as v4f",
            "v4f.ClipPCAFit(",
            "v4f.VJepa2GlobalCodec(",
            "sum(parameter.numel() for parameter in model.parameters()) != 79040",
            'parser.add_argument("--fold1-receipt", required=True)',
            'parser.add_argument("--fold1-preselection-checkpoint", required=True)',
            "def validate_v4f_fold_receipt(",
            '"aggregate_binding_exact": True',
            '"fold_receipt_artifact_join_exact": True',
            '"single_fd_preparse_sha_and_postparse_identity_verified": True',
            '"basis_metadata_state_hash_join_verified": True',
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("Tucker", source)
        self.assertNotIn("tucker", source)
        self.assertNotIn("nonlinear_temporal_codec_v4f", source)

    def test_selected_rho_preselection_state_and_base_join_are_explicit(self) -> None:
        materializer = _source(MATERIALIZER_PATH)
        runtime = _source(RUNTIME_PATH)
        for fragment in (
            "EXPECTED_FOLD1_SELECTED_RHO",
            "EXPECTED_FOLD1_PRESELECTION_CHECKPOINT_SHA256",
            "EXPECTED_FOLD1_PRESELECTION_BASE_STATE_SHA256",
            'expected_role="preselection_fixed_step1200"',
            'expected_role="selected_fold_local_rho"',
            '"residual_gate_rho"',
            "v4f._base_state_sha(preselection_state)",
            "v4f._base_state_sha(state)",
            '"exact_trainable_parameter_count": 79040',
            '"sole_code_shape": [CODE_TIME, CODE_CHANNELS]',
            '"selected_rho_state_buffer_verified": True',
            '"selected_preselection_base_state_join_exact": True',
        ):
            self.assertIn(fragment, materializer)
        for fragment in (
            "fold_preselection_checkpoint",
            "EXPECTED_FOLD1_PRESELECTION_CHECKPOINT_SHA256",
            'fold_checkpoint.get("selected_rho")',
            '"v4f_exact5_all_fold_inner_gates_passed": True',
        ):
            self.assertIn(fragment, runtime)

    def test_arm_permutations_seeds_and_heavy_path_are_frozen(self) -> None:
        materializer = _source(MATERIALIZER_PATH)
        runtime = _source(RUNTIME_PATH)
        self.assertIn(
            'ARM_ORDER = ("zero", "correct", "reverse", "shuffle")', materializer
        )
        self.assertIn(
            'ARM_ORDER = ("B0", "zero", "correct", "reverse", "shuffle")',
            runtime,
        )
        self.assertIn("RENDER_SEEDS = (2028, 2029)", runtime)
        self.assertIn("PROPOSAL_SEED = 2027", runtime)
        self.assertIn("epmc.REVERSE_PHASE_INDICES", materializer)
        self.assertIn("epmc.SHUFFLE_PHASE_INDICES", materializer)
        self.assertIn("infer_fewshot_motion_code as epmc_runner", runtime)
        self.assertIn("v4b_runtime._save_arm_outputs", runtime)
        self.assertIn("epmc_runner.main(_runner_argv(args))", runtime)

    def test_runner_defers_heavy_imports_until_after_release_guard(self) -> None:
        for path in (MATERIALIZER_PATH, RUNTIME_PATH):
            tree = _tree(path)
            top_import_names = {
                alias.name
                for node in tree.body
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            self.assertNotIn("torch", top_import_names)
        source = _source(RUNTIME_PATH)
        guard = source.index("def _require_release_sealed")
        torch_import = source.index("    import torch", guard)
        bernini_import = source.index("infer_fewshot_motion_code as epmc_runner", guard)
        self.assertGreater(torch_import, guard)
        self.assertGreater(bernini_import, guard)

    def test_authority_receipt_checkpoint_and_gate_reads_are_single_fd(self) -> None:
        materializer = _source(MATERIALIZER_PATH)
        runtime = _source(RUNTIME_PATH)
        for fragment in (
            'os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)',
            "before = os.fstat(handle.fileno())",
            '"st_ino"',
            "torch.load(handle, map_location=\"cpu\", weights_only=True)",
            "changed across single-FD read",
            "changed across single-FD load",
        ):
            self.assertIn(fragment, materializer)
        for fragment in (
            'os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)',
            "before = os.fstat(handle.fileno())",
            '"st_ino"',
            "gate-state changed across single-FD read",
        ):
            self.assertIn(fragment, runtime)
        builder = _source(BUILDER_PATH)
        for fragment in (
            'descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))',
            "hashlib.sha256(raw).hexdigest() != expected_sha256",
            "receipt changed across single-FD read",
        ):
            self.assertIn(fragment, builder)

    def test_known_exposed_boundary_never_becomes_unseen_or_action_qualification(self) -> None:
        for path in PRODUCTION_PATHS:
            source = _source(path)
            self.assertIn("unseen_hostile_transform_gate", source)
            self.assertIn("unseen_hostile_transform_gate_evaluated", source)
            self.assertIn("action_representation_qualified", source)
        materializer = _source(MATERIALIZER_PATH)
        runtime = _source(RUNTIME_PATH)
        builder = _source(BUILDER_PATH)
        self.assertIn('"known_exposed_transform_families_only": True', materializer)
        self.assertIn('"unseen_hostile_transform_gate": False', materializer)
        self.assertIn('"unseen_action_qualification": False', runtime)
        self.assertIn('"unseen_action_qualification": False', builder)
        self.assertIn("not an unseen-transform or unseen-action result", runtime)

    def test_html_has_all_columns_and_nonclaim_warning(self) -> None:
        source = _source(BUILDER_PATH)
        for fragment in (
            '"Source"',
            '"Anchor"',
            '"B0"',
            '"Zero"',
            '"Correct"',
            '"Reverse"',
            '"Shuffle"',
            "PRIVILEGED OOF TEMPORAL-GATING DIAGNOSTIC ONLY",
            "source+instruction-only",
            "Anchor RGB",
            "0.05×profile",
            "final latent byte-exact",
            "[12,32]",
            "known transform families",
            "unseen-transform / unseen-action qualification",
            '"vae_necessary": None',
        ):
            self.assertIn(fragment, source)

    def test_unsealed_runtime_and_html_fail_before_io(self) -> None:
        materializer = importlib.import_module(
            "methods.bernini_action_editing.materialize_v4f_epmc_gate_state_v1"
        )
        runtime = importlib.import_module(
            "methods.bernini_action_editing.infer_v4f_epmc_temporal_gate_canary_v1"
        )
        builder = importlib.import_module(
            "methods.bernini_action_editing.tools."
            "build_v4f_epmc_temporal_gate_review_v1"
        )
        with mock.patch.object(materializer, "_load_tensor_runtime") as tensor_runtime:
            with mock.patch.object(materializer, "_plain_absolute_file") as plain:
                with self.assertRaisesRegex(
                    materializer.V4FEPMCGateStateError, "UNSEALED"
                ):
                    materializer.run(object())
                tensor_runtime.assert_not_called()
                plain.assert_not_called()
        with mock.patch.object(argparse, "ArgumentParser") as parser:
            with self.assertRaisesRegex(
                materializer.V4FEPMCGateStateError, "UNSEALED"
            ):
                materializer.build_parser()
            parser.assert_not_called()
        with mock.patch.object(Path, "lstat") as lstat:
            with self.assertRaisesRegex(runtime.V4FEPMCVideoCanaryError, "UNSEALED"):
                runtime.load_gate_state("/definitely/not/read.json", expected_sha256="x")
            lstat.assert_not_called()
        with mock.patch.object(Path, "lstat") as lstat:
            with self.assertRaisesRegex(builder.V4FEPMCReviewError, "UNSEALED"):
                builder.run(object())
            lstat.assert_not_called()
        with self.assertRaisesRegex(runtime.V4FEPMCVideoCanaryError, "UNSEALED"):
            runtime.main([])
        with self.assertRaisesRegex(builder.V4FEPMCReviewError, "UNSEALED"):
            builder.main([])
        with self.assertRaisesRegex(materializer.V4FEPMCGateStateError, "UNSEALED"):
            materializer.main([])

    def test_scope_helper_accepts_only_known_exposed_fail_closed_contract(self) -> None:
        materializer = importlib.import_module(
            "methods.bernini_action_editing.materialize_v4f_epmc_gate_state_v1"
        )
        closed = {
            "exposed_five_view_codec_development_gate": True,
            "unseen_hostile_transform_gate": False,
            "unseen_hostile_transform_gate_evaluated": False,
            "latent_metric_qualified": False,
            "action_representation_qualified": False,
            "identity_disentanglement_qualified": False,
            "identity_preservation_qualified": False,
            "prior_qualified": False,
            "prior_generation_qualified": False,
            "generation_qualified": False,
            "renderer_qualified": False,
            "video_editing_qualified": False,
            "inference_authorized": False,
            "web_evaluation_authorized": False,
            "full644_refit_authorized": False,
            "video_model_training_performed": False,
            "vae_necessary": None,
        }
        self.assertTrue(materializer._scope_is_fail_closed(closed, aggregate=True))
        for key, hostile in (
            ("exposed_five_view_codec_development_gate", False),
            ("unseen_hostile_transform_gate", True),
            ("unseen_hostile_transform_gate_evaluated", True),
            ("action_representation_qualified", True),
            ("vae_necessary", False),
        ):
            changed = dict(closed)
            changed[key] = hostile
            self.assertFalse(
                materializer._scope_is_fail_closed(changed, aggregate=True), key
            )


@unittest.skipIf(torch is None, "torch unavailable; static fail-close audit still runs")
class V4FEPMCTemporalGateTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.materializer = importlib.import_module(
            "methods.bernini_action_editing.materialize_v4f_epmc_gate_state_v1"
        )

    def test_decoded_residual_uses_decode_zero_subtraction(self) -> None:
        materializer = self.materializer

        class DummyCodec:
            def __call__(self, value: object) -> object:
                return value * 3.0 + 7.0

        centered = torch.arange(32 * 1024, dtype=torch.float32).reshape(1, 32, 1024)
        zero_decode = torch.full((1, 32, 1024), 7.0, dtype=torch.float32)
        residual = materializer.decoded_residual(DummyCodec(), centered, zero_decode)
        self.assertTrue(torch.equal(residual, centered * 3.0))
        self.assertEqual(tuple(residual.shape), (1, 32, 1024))

    def test_fit_only_p95_and_32_to_20_mapping(self) -> None:
        materializer = self.materializer
        held = torch.ones(1, 32, 1024, dtype=torch.float32)
        fit = torch.stack(
            (
                torch.ones(32, 1024, dtype=torch.float32),
                torch.full((32, 1024), 2.0, dtype=torch.float32),
            )
        )
        p95, profile32, profile20 = materializer.scaled_profile_32_to_20(
            held, fit
        )
        expected_p95 = torch.quantile(
            materializer.residual_rms_profile(fit).reshape(-1).to(torch.float64),
            0.95,
            interpolation="linear",
        ).to(torch.float32)
        self.assertTrue(torch.equal(p95.reshape(()), expected_p95))
        self.assertEqual(tuple(profile32.shape), (1, 32))
        self.assertEqual(tuple(profile20.shape), (1, 20))
        self.assertTrue(bool((profile20 >= 0.0).all()))
        self.assertTrue(bool((profile20 <= 1.0).all()))

    def test_codes_have_zero_phase0_heads_and_frozen_permutations(self) -> None:
        materializer = self.materializer
        profile20 = torch.linspace(0.05, 1.0, 20, dtype=torch.float32)[None]
        codes = materializer.build_motion_codes(profile20)
        self.assertEqual(tuple(codes), materializer.ARM_ORDER)
        self.assertEqual(
            int(torch.count_nonzero(codes["zero"].phase_gates).item()), 0
        )
        for code in codes.values():
            self.assertEqual(
                int(torch.count_nonzero(code.phase_gates[:, :1]).item()), 0
            )
            self.assertEqual(int(torch.count_nonzero(code.block_head_gates).item()), 0)
            self.assertEqual(
                int(torch.count_nonzero(code.block_head_gates.view(torch.uint8)).item()),
                0,
            )
        epmc = importlib.import_module(
            "methods.bernini_action_editing.fewshot_privileged_motion_code"
        )
        self.assertTrue(
            torch.equal(
                codes["reverse"].phase_gates,
                codes["correct"].phase_gates[:, list(epmc.REVERSE_PHASE_INDICES)],
            )
        )
        self.assertTrue(
            torch.equal(
                codes["shuffle"].phase_gates,
                codes["correct"].phase_gates[:, list(epmc.SHUFFLE_PHASE_INDICES)],
            )
        )
        reference = torch.sort(codes["correct"].phase_gates[:, 1:], dim=1).values
        self.assertTrue(
            torch.equal(
                reference,
                torch.sort(codes["reverse"].phase_gates[:, 1:], dim=1).values,
            )
        )
        self.assertTrue(
            torch.equal(
                reference,
                torch.sort(codes["shuffle"].phase_gates[:, 1:], dim=1).values,
            )
        )

    def test_materializer_unsealed_guard_precedes_input_io(self) -> None:
        materializer = self.materializer
        with tempfile.TemporaryDirectory() as directory:
            args = type(
                "Args",
                (),
                {
                    "v4f_receipt": str(Path(directory) / "receipt.json"),
                    "fold1_receipt": str(Path(directory) / "fold.json"),
                    "fold1_preselection_checkpoint": str(
                        Path(directory) / "preselection.pt"
                    ),
                    "fold1_checkpoint": str(Path(directory) / "fold1.pt"),
                    "feature_root": str(Path(directory) / "features"),
                    "output": str(Path(directory) / "gate.json"),
                    "batch_size": 32,
                },
            )()
            with mock.patch.object(materializer, "_plain_absolute_file") as plain:
                with self.assertRaisesRegex(
                    materializer.V4FEPMCGateStateError, "UNSEALED"
                ):
                    materializer.run(args)
                plain.assert_not_called()


if __name__ == "__main__":
    unittest.main()

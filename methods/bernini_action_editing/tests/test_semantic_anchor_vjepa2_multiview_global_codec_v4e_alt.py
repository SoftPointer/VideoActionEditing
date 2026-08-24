from __future__ import annotations

import ast
import hashlib
import os
import tempfile
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "semantic_anchor_vjepa2_multiview_global_codec_v4e_alt.py"


class VJepa2MultiviewGlobalCodecV4EAltStaticTests(unittest.TestCase):
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

    def test_only_parallel_fold_and_cpu_aggregate_commands_exist(self) -> None:
        self.assertEqual(self.source.count('add_parser("train-fold")'), 1)
        self.assertEqual(self.source.count('add_parser("aggregate")'), 1)
        self.assertEqual(self.source.count("add_parser("), 2)
        self.assertNotIn("run-exact5", self.source)
        parser = self.function_source("build_parser")
        for token in (
            '"--fold-index"', '"--fold-root"', 'action="append"',
            '"--v4d-receipt"', '"--expected-v4d-receipt-sha256"',
        ):
            self.assertIn(token, parser)

    def test_frozen_v4d_burned_motivation_is_full_literal(self) -> None:
        for value in (
            "20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc",
            "53910bcb71ce02a193bd47e44c3a97de0ee24f431576db64a763637447720b6f",
            "45d2ae7c45f1db8ccee9b14ba8a7543cfd1ff0d311128472ae116d6befa92f9c",
            "/semantic_anchor_vjepa2_nonlinear_codec_v4d_20260820/",
            "runs/exact5_20934925_v2/receipt.json",
        ):
            self.assertIn(value, self.source)
        loader = self.function_source("_load_v4d_burned_receipt")
        self.assertIn("str(path) != V4D_RECEIPT_PATH", loader)
        self.assertIn("V4D_RECEIPT_DIGEST", loader)
        self.assertIn("decoded_temporal_codec_development_gate", loader)

    def test_clip_pca_is_only_v4e_comparator(self) -> None:
        self.assertIn('BASELINE_NAME = "clip_pca_b0384_t01_r384"', self.source)
        self.assertNotIn("Tucker", self.source)
        self.assertNotIn("tucker_b384", self.source)
        fit = self.function_source("_fit_clip_pca_b384")
        self.assertIn("v4a._fit_clip_basis", fit)
        self.assertIn("CODE_NUMEL", fit)
        self.assertIn('row.views["original"]', fit)
        self.assertNotIn("EVAL_VIEWS", fit)

    def test_global_codec_is_sole_code_cross_attention(self) -> None:
        model = self.class_source("ClipPCAInitializedVJepaGlobalCodec")
        for required in (
            "input_projection", "input_position", "code_queries",
            "encoder_attention", "encoder_delta", "time_queries",
            "code_position", "decoder_attention", "decoder_output",
            "self.clip_mean", "self.clip_basis",
        ):
            self.assertIn(required, model)
        self.assertIn("code.flatten(1) @ self.clip_basis.T", model)
        self.assertIn("self.decoder_attention(queries, context)", model)
        self.assertNotIn("Conv1d", model)
        self.assertNotIn("MultiheadAttention", model)
        decoder = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "decode"
        ]
        self.assertEqual(len(decoder), 1)
        decoder_text = ast.get_source_segment(self.source, decoder[0]) or ""
        self.assertNotIn("value", decoder_text)
        self.assertNotIn("input_projection", decoder_text)

    def test_code_and_parameter_closure_are_literal(self) -> None:
        for literal in (
            "CODE_TIME = 12", "CODE_CHANNELS = 32",
            "CODE_NUMEL = CODE_TIME * CODE_CHANNELS",
            "EXACT_TRAINABLE_PARAMETERS = 79040",
            "MAX_TRAINABLE_PARAMETERS = 150000",
        ):
            self.assertIn(literal, self.source)
        model = self.class_source("ClipPCAInitializedVJepaGlobalCodec")
        self.assertIn("count != EXACT_TRAINABLE_PARAMETERS", model)
        self.assertIn("count >= MAX_TRAINABLE_PARAMETERS", model)

    def test_step0_is_zero_delta_clip_pca_and_all_five_fit_views_checked(self) -> None:
        model = self.class_source("ClipPCAInitializedVJepaGlobalCodec")
        self.assertIn("nn.init.zeros_(self.encoder_delta.weight)", model)
        self.assertIn("nn.init.zeros_(self.decoder_output.weight)", model)
        step0 = self.function_source("_step0_equivalence")
        self.assertIn("torch.equal(actual_code, reference_code)", step0)
        self.assertIn("torch.equal(actual_all, reference_all)", step0)
        train = self.function_source("_train_fold_model")
        self.assertIn("fit_views.flatten(0, 1)", train)
        self.assertIn('"step0_model_fit_all_five_views_equivalence"', train)

    def test_multiview_loss_uses_per_iid_stopgrad_scale_and_ten_pairs(self) -> None:
        loss = self.function_source("_multiview_training_loss")
        for required in (
            "for left in range(len(EVAL_VIEWS))",
            "for right in range(left + 1, len(EVAL_VIEWS))",
            "len(teacher_distances) != 10",
            "teacher_geometry.detach().mean(dim=1, keepdim=True) + 1.0e-8",
            "(candidate_geometry - teacher_geometry) / per_iid_scale",
            "beta=0.1", "geometry_weight != 0.25",
        ):
            self.assertIn(required, loss)
        for forbidden in ("family", "NEGATIVES", "monotone_warp", "view_names"):
            self.assertNotIn(forbidden, loss)

    def test_equal_view_reconstruction_is_permutation_canonicalized(self) -> None:
        rec = self.function_source("_single_view_reconstruction_loss")
        for required in (
            'beta=0.1, reduction="none"', "torch.sort(per_view).values.mean()",
            "for stride in (1, 2, 4)", "prediction[:, :, -1]",
        ):
            self.assertIn(required, rec)

    def test_training_reads_all_five_by_iid_but_no_role_labels(self) -> None:
        train = self.function_source("_train_fold_model")
        self.assertIn("fit_views", train)
        self.assertIn("target.flatten(0, 1)", train)
        self.assertIn("_multiview_training_loss", train)
        self.assertIn('"view_axis_permutation_invariant_loss": True', train)
        self.assertIn('"view_name_or_positive_negative_role_labels_used": 0', train)
        for forbidden in ("row.family", "row.strict", "NEGATIVES", "outer_assignment"):
            self.assertNotIn(forbidden, train)

    def test_pretraining_authority_does_not_bulk_load_feature_values(self) -> None:
        prepare = self.function_source("_prepare_authorities")
        self.assertIn("_load_feature_metadata_authority", prepare)
        self.assertNotIn("load_v4c_features", prepare)
        metadata = self.function_source("_load_feature_metadata_authority")
        self.assertNotIn("torch.load", metadata)
        self.assertNotIn("_load_sealed_shard", metadata)
        self.assertIn("views={}", metadata)

    def test_selective_loader_uses_fake_offsets_and_exact_pread(self) -> None:
        loader = self.function_source("_selective_materialize_feature_rows")
        for required in (
            "FakeTensorMode", "weights_only=True", "_checkpoint_offset",
            "os.pread", "FULL_NUMEL * 4", "torch.frombuffer",
            "_tensor_sha(value)", "digest_before", "digest_after",
            "os.O_NOFOLLOW", "stat.S_IMODE(before.st_mode) != 0o444",
            "all_fake_tensor_offsets_unique_nonoverlapping_in_file",
            "unrequested_tensor_storage_materialized_count",
        ):
            self.assertIn(required, loader)
        self.assertNotIn("untyped_storage().nbytes()", loader)

    def test_two_stage_read_barrier_is_after_checkpoint_seal(self) -> None:
        run = self.function_source("_run_fold")
        stage1 = run.index('stage="stage1_model_fit_and_inner"')
        save = run.index("_save_selected_checkpoint_create_only")
        stage2 = run.index('stage="stage2_postseal_oof"')
        evaluate = run.index("_evaluate_fold")
        self.assertLess(stage1, save)
        self.assertLess(save, stage2)
        self.assertLess(stage2, evaluate)
        self.assertIn('"stage1_oof_semantic_tensor_count": 0', run)
        self.assertIn('"stage1_inner_derived_semantic_tensor_count": 0', run)
        self.assertIn('"stage2_model_fit_or_inner_semantic_tensor_count": 0', run)

    def test_checkpoint_loader_is_single_fd_strong_and_semantic(self) -> None:
        loader = self.function_source("_load_selected_checkpoint_sealed")
        for required in (
            "path.is_absolute()", "path.is_symlink()", "path.resolve(strict=True)",
            "os.O_NOFOLLOW", "os.open(", "os.fstat(", "path.lstat()",
            "digest_before", "digest_after", "weights_only=True",
            "metadata_digest", "_state_sha(state)", "physical_identity",
        ):
            self.assertIn(required, loader)
        self.assertEqual(loader.count("os.open("), 1)
        self.assertNotIn("torch.load(path", loader)
        for required in (
            'state["clip_mean"]', 'state["clip_basis"]', 'state["fit_only_rms"]',
            "VJepa2GlobalCodec(template_fit", "template.load_state_dict(state, strict=True)",
            "EXACT_TRAINABLE_PARAMETERS", '"model_forward_executed_by_loader": False',
        ):
            self.assertIn(required, loader)

    def test_fold_receipt_loader_rejects_duplicates_nonfinite_and_replays(self) -> None:
        loader = self.function_source("_load_fold_receipt_sealed")
        for required in (
            "object_pairs_hook=_reject_duplicate_json_pairs",
            "parse_constant=_reject_nonfinite_json", "os.O_NOFOLLOW",
            "single_fd_pre_post_bytes_and_identity_exact", "receipt_digest",
        ):
            self.assertIn(required, loader)

    def test_train_fold_writes_and_strongly_self_reads_fixed_artifacts(self) -> None:
        train = self.function_source("run_train_fold")
        self.assertIn("_write_json_create_only(fold_path, receipt)", train)
        self.assertIn("_load_fold_receipt_sealed", train)
        self.assertIn("_load_selected_checkpoint_sealed", train)
        roots = self.function_source("_resolve_fold_root")
        self.assertIn('root / "fold.json"', roots)
        self.assertIn('root / "selected.pt"', roots)

    def test_aggregate_is_cpu_only_and_never_constructs_or_trains_model(self) -> None:
        aggregate = self.function_source("run_aggregate")
        for forbidden in (
            "_resolve_device", "_train_fold_model", "_run_fold(",
            "VJepa2GlobalCodec(", "ClipPCAInitializedVJepaGlobalCodec(",
            ".backward(", "optimizer",
        ):
            self.assertNotIn(forbidden, aggregate)
        self.assertIn('"aggregate_device": "cpu"', aggregate)
        self.assertIn('"model_schema_reconstructed_and_strict_loaded": True', aggregate)
        self.assertIn('"model_forward_executed": False', aggregate)
        self.assertIn('"model_trained_or_recomputed": False', aggregate)
        self.assertGreaterEqual(aggregate.count("_load_fold_receipt_sealed"), 2)
        self.assertGreaterEqual(aggregate.count("_verify_checkpoint_artifacts"), 2)
        self.assertIn("v4c._load_json_sealed(output, receipt_sha)", aggregate)
        self.assertIn("_verify_fold_split_against_authority", aggregate)
        replay = self.function_source("_verify_fold_split_against_authority")
        self.assertIn("_split_fold(records, outer_assignment", replay)
        self.assertIn('fold.get("inner_split") != split', replay)
        self.assertIn('fold.get("model_fit_ordered_iids") != fit_iids', replay)
        self.assertIn('fold.get("oof_ordered_iids") != oof_iids', replay)

    def test_comparator_choice_ledger_and_fail_closed_scope(self) -> None:
        aggregate = self.function_source("run_aggregate")
        for literal in (
            '"v4c_burned_oof_informed_clip_pca_b384_choice": True',
            '"v4e_oof_used_to_select_comparator": False',
            '"single_v4e_candidate": True',
            '"unseen_hostile_transform_gate": False',
            '"unseen_hostile_transform_gate_evaluated": False',
            '"latent_metric_qualified": False',
            '"action_representation_qualified": False',
            '"identity_disentanglement_qualified": False',
            '"identity_preservation_qualified": False',
            '"vae_necessary": None',
            '"generation_qualified": False',
            '"prior_generation_qualified": False',
            '"renderer_qualified": False',
            '"video_editing_qualified": False',
            '"inference_authorized": False',
            '"web_evaluation_authorized": False',
            '"full644_refit_authorized": False',
        ):
            self.assertIn(literal, aggregate)

    def test_no_assert_statements_guard_runtime_contracts(self) -> None:
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(self.tree)))


@unittest.skipIf(torch is None, "PyTorch is unavailable in the local test interpreter")
class VJepa2MultiviewGlobalCodecV4EAltDynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from methods.bernini_action_editing import (
            semantic_anchor_vjepa2_multiview_global_codec_v4e_alt as runtime,
        )
        cls.m = runtime
        cls.fit = runtime.ClipPCAFit(
            clip_mean=torch.zeros(1, runtime.FULL_NUMEL),
            clip_basis=torch.zeros(runtime.FULL_NUMEL, runtime.CODE_NUMEL),
            fit_iid_digest="0" * 64,
            fit_input_sha256="1" * 64,
            diagnostics={},
        )

    def model(self):
        torch.manual_seed(7)
        return self.m.VJepa2GlobalCodec(self.fit, torch.ones(1))

    def centered(self, count: int) -> torch.Tensor:
        torch.manual_seed(11 + count)
        value = torch.randn(count, self.m.TIME_STEPS, self.m.FEATURE_DIM)
        return (value - value.mean(dim=1, keepdim=True)).contiguous()

    def test_exact_parameter_count_and_code_geometry(self) -> None:
        model = self.model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 79040)
        value = self.centered(2)
        code = model.encode(value)
        self.assertEqual(tuple(code.shape), (2, 12, 32))
        self.assertEqual(code[0].numel(), 384)
        self.assertEqual(tuple(model.decode(code).shape), tuple(value.shape))

    def test_step0_is_bit_exact_clip_pca(self) -> None:
        model = self.model().eval()
        value = self.centered(2)
        with torch.no_grad():
            code = model.encode(value)
            expected_code = self.m._analytic_clip_pca_encode(value, self.fit)
            actual = model.decode(code)
            expected = self.m._analytic_clip_pca_decode(value, self.fit)
        self.assertTrue(torch.equal(code, expected_code))
        self.assertTrue(torch.equal(actual, expected))

    def test_decoder_accepts_only_code_geometry(self) -> None:
        model = self.model()
        with self.assertRaises(ValueError):
            model.decode(torch.zeros(1, 384))
        with self.assertRaises(ValueError):
            model.decode(torch.zeros(1, 12, 31))

    def test_multiview_loss_and_components_are_permutation_invariant(self) -> None:
        torch.manual_seed(29)
        target = torch.randn(3, 5, 32, 1024)
        prediction = target + 0.2 * torch.randn_like(target)
        first, first_parts = self.m._multiview_training_loss(prediction, target)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        second, second_parts = self.m._multiview_training_loss(
            prediction[:, permutation], target[:, permutation]
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(set(first_parts), set(second_parts))
        for key in first_parts:
            self.assertEqual(first_parts[key], second_parts[key], key)

    def test_multiview_loss_has_finite_gradients(self) -> None:
        torch.manual_seed(31)
        target = torch.randn(2, 5, 32, 1024)
        prediction = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
        loss, parts = self.m._multiview_training_loss(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(parts["geometry_weight"], 0.25)

    def test_selective_pread_events_straddle_checkpoint_reload(self) -> None:
        view_names = tuple(self.m.EVAL_VIEWS)
        anchors = {
            f"i{ordinal:03d}": SimpleNamespace(
                ordinal=ordinal, iid=f"i{ordinal:03d}",
                family=f"f{ordinal % 28}", strict=False,
            )
            for ordinal in range(644)
        }
        ordinals = {iid: row.ordinal for iid, row in anchors.items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            shards = []
            for shard_index in range(6):
                shard_ordinals = [value for value in range(644) if value % 6 == shard_index]
                records = []
                for ordinal in shard_ordinals:
                    iid = f"i{ordinal:03d}"
                    sequences = {
                        name: torch.tensor([[float(ordinal * 10 + index)]])
                        for index, name in enumerate(view_names)
                    }
                    records.append({
                        "ordinal": ordinal, "iid": iid,
                        "family": anchors[iid].family,
                        "strict_selection_gates_all_true": False,
                        "role": "action_anchor", "view_order": list(view_names),
                        "view_sequences": sequences,
                        "view_receipts": {
                            name: {
                                "ordered_contextual_sequence_sha256":
                                    self.m._tensor_sha(sequences[name])
                            }
                            for name in view_names
                        },
                    })
                payload = {
                    "schema_version": self.m.features.FEATURE_SCHEMA,
                    "shard_index": shard_index, "num_shards": 6,
                    "global_anchor_ordinals": shard_ordinals,
                    "records": records,
                }
                path = root / f"shard{shard_index}.pt"
                torch.save(payload, path)
                os.chmod(path, 0o444)
                raw = path.read_bytes()
                shards.append({
                    "index": shard_index, "path": str(path),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw), "mode": 0o444, "nlink": 1,
                    "record_count": len(records),
                })
            feature_index = {
                "anchors_by_iid": anchors,
                "ordinal_by_iid": ordinals,
                "receipt": {"shards": shards},
            }
            events = []
            stage = ["stage1"]
            real_pread = os.pread

            def recording_pread(fd, count, offset):
                events.append((stage[0], "pread", offset, count))
                return real_pread(fd, count, offset)

            with (
                mock.patch.object(self.m, "TIME_STEPS", 1),
                mock.patch.object(self.m, "FEATURE_DIM", 1),
                mock.patch.object(self.m, "FULL_NUMEL", 1),
                mock.patch.object(self.m.os, "pread", side_effect=recording_pread),
            ):
                _, first = self.m._selective_materialize_feature_rows(
                    feature_index,
                    {"i000": view_names, "i001": ("original",)},
                    stage="stage1_model_fit_and_inner",
                )
                with mock.patch.object(
                    self.m, "_load_selected_checkpoint_sealed",
                    side_effect=lambda *args, **kwargs: events.append(
                        ("checkpoint", "strong_reload", 0, 0)
                    ),
                ):
                    self.m._load_selected_checkpoint_sealed(None, None)
                stage[0] = "stage2"
                _, second = self.m._selective_materialize_feature_rows(
                    feature_index, {"i002": view_names},
                    stage="stage2_postseal_oof",
                )
            labels = [event[0] for event in events]
            checkpoint_index = labels.index("checkpoint")
            self.assertTrue(all(label == "stage1" for label in labels[:checkpoint_index]))
            self.assertTrue(all(label == "stage2" for label in labels[checkpoint_index + 1:]))
            self.assertEqual(first["semantic_tensor_materialized_count"], 6)
            self.assertEqual(first["semantic_tensor_materialized_count_by_view"]["original"], 2)
            self.assertTrue(all(
                first["semantic_tensor_materialized_count_by_view"][name] == 1
                for name in view_names if name != "original"
            ))
            self.assertEqual(second["semantic_tensor_materialized_count"], 5)

    def test_fresh_checkpoint_save_and_strong_semantic_reload(self) -> None:
        model = self.model().eval()
        state_sha = self.m._state_sha(self.m._state_to_cpu(model))
        fit_iids = ["fit-0"]
        inner_iids = ["inner-0"]
        audit = {
            "selected_step": 0,
            "selected_state_sha256": state_sha,
            "minibatch_schedule_sha256": "3" * 64,
            "fit_only_global_rms_sha256": self.m._tensor_sha(model.fit_only_rms),
            "model_fit_original_count": len(fit_iids),
            "model_fit_ordered_iids": fit_iids,
            "model_fit_iid_digest": self.m._object_sha(fit_iids),
            "inner_validation_iid_digest": self.m._object_sha(inner_iids),
        }
        validation = [self.m.v4c.Record(
            iid=inner_iids[0], family="family", strict=False,
            views={"original": self.centered(1)[0]},
        )]
        binding = {"implementation_sha256": "2" * 64}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "selected.pt"
            artifact = self.m._save_selected_checkpoint_create_only(
                path, model, self.fit, 0, audit, self.m.Config(), 0,
                binding, validation, torch.device("cpu"),
            )
            metadata, state, replay = self.m._load_selected_checkpoint_sealed(
                path, artifact
            )
            self.assertEqual(metadata["model_state_sha256"], state_sha)
            self.assertEqual(self.m._state_sha(state), state_sha)
            self.assertTrue(replay["model_schema_reconstructed_and_strict_loaded"])
            self.assertFalse(replay["model_forward_executed_by_loader"])

    def test_checkpoint_wrong_file_sha_fails_before_torch_parse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "hostile.pt"
            path.write_bytes(b"not a torch checkpoint")
            os.chmod(path, 0o444)
            with mock.patch.object(self.m.torch, "load") as loader:
                with self.assertRaisesRegex(RuntimeError, "before torch parse"):
                    self.m._load_selected_checkpoint_sealed(
                        path, {"file_sha256": "0" * 64}
                    )
                loader.assert_not_called()

    def test_config_is_immutable(self) -> None:
        self.m.Config().validate()
        with self.assertRaises(ValueError):
            self.m.Config(geometry_weight=0.2).validate()


if __name__ == "__main__":
    unittest.main()

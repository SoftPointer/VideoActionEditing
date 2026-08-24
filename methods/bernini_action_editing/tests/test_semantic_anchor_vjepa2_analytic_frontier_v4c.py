#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from methods.bernini_action_editing import semantic_anchor_vjepa2_analytic_frontier_v4c as runtime


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _assignment_and_evidence():
    assignment = {}
    evidence = []
    ordinal = 0
    for fold, count in enumerate(runtime.OOF_COUNTS):
        for _ in range(count):
            iid = f"iid-{ordinal:04d}"
            family = f"family-{ordinal % 28:02d}"
            assignment[iid] = fold
            evidence.append({"iid": iid, "family": family, "outer_fold": fold})
            ordinal += 1
    return assignment, evidence


def _records(evidence):
    shared = torch.zeros((runtime.TIME_STEPS, runtime.FEATURE_DIM), dtype=torch.float32)
    return [
        runtime.Record(
            iid=row["iid"], family=row["family"], strict=index < 359,
            views={name: shared for name in runtime.features.VIEW_NAMES},
        )
        for index, row in enumerate(evidence)
    ]


class VJepa2FrontierV4CSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(runtime.__file__).resolve(strict=True)
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _function(self, name: str) -> str:
        node = next(
            item for item in self.tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
        )
        return ast.get_source_segment(self.source, node) or ""

    def test_pins_and_exact5_budget_contract(self):
        self.assertEqual(runtime.OUTER_ASSIGNMENT_DIGEST, "5ab9704f456768b440c966a53328de0c1a67836548f8f8ebd92e50d21846ab5f")
        self.assertEqual(runtime.OOF_COUNTS, (131, 127, 128, 129, 129))
        self.assertEqual(runtime.PAYLOAD_BUDGETS, (32, 64, 128, 256, 384))
        self.assertEqual(runtime.V4A_RECEIPT_FILE_SHA256, "568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2")
        self.assertEqual(runtime.V4A_EVIDENCE_SHA256, "f1d34d9ade4e36200f5dbd0da277cf8cf1221482f66c76d42c168a984a0cf123")
        self.assertEqual(runtime.EXTRACTOR_IMPLEMENTATION_SHA256, "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc")

    def test_no_vjepa_energy_resplit_and_fit_reads_original_only(self):
        self.assertNotIn("v2._split_fold(", self.source)
        self.assertNotIn("stratified_fold_assignment(", self.source)
        fitted = self._function("_fit_frontier")
        self.assertIn('row.views["original"]', fitted)
        for name in ("monotone_warp", "reverse", "block_shuffle", "phase_swap"):
            self.assertNotIn(f'row.views["{name}"]', fitted)
        fold = self._function("_run_fold")
        self.assertLess(fold.index("_fit_frontier("), fold.index("_evaluate_fold("))
        self.assertIn('"oof_used_for_projection_rank_or_winner_selection": False', fold)

    def test_bootstrap_reuses_v4a_abi_and_receipt_is_fail_closed(self):
        bootstrap = self._function("_bootstrap")
        self.assertIn("v4a._paired_bootstrap_lcbs", bootstrap)
        run = self._function("run_exact5")
        for field in (
            '"training_authorized": False',
            '"action_representation_qualified": False',
            '"scientific_confirmation_claimed": False',
            '"identity_disentanglement_qualified": False',
            '"identity_preservation_qualified": False',
            '"prior_generation_qualified": False',
            '"generation_qualified": False',
            '"renderer_qualified": False',
            '"video_editing_qualified": False',
            '"inference_authorized": False',
            '"web_evaluation_authorized": False',
            '"full644_refit_authorized": False',
            '"vae_necessary": None',
        ):
            self.assertIn(field, run)


class VJepa2FrontierV4CMathTests(unittest.TestCase):
    def test_canonicalization_distance_and_payloads(self):
        value = torch.arange(
            runtime.TIME_STEPS * runtime.FEATURE_DIM, dtype=torch.float32
        ).reshape(runtime.TIME_STEPS, runtime.FEATURE_DIM)
        canonical = runtime.canonical_action(value)
        self.assertEqual(tuple(canonical.shape), (32, 1024))
        self.assertLess(float(canonical.mean(dim=0).abs().max()), 3.0e-6)
        shifted = canonical + 2.0
        self.assertAlmostEqual(
            runtime.normalized_squared_distance(canonical, shifted), 4.0, places=6
        )
        specs = runtime.candidate_specs(runtime.Config())
        self.assertEqual(len(specs), 15)
        for budget in runtime.PAYLOAD_BUDGETS:
            selected = [row for row in specs if row["payload_numel"] == budget]
            self.assertEqual([row["kind"] for row in selected], ["frame_pca", "clip_pca", "tucker"])

    def test_canonicalization_preserves_fp32_result_at_real_scale(self):
        # Frozen values from the coordinate that triggered the first real
        # exact644 run's fail-closed FP32-centering check.  The old one-pass
        # FP32 implementation leaves abs(mean)==3.248453e-6 (>3e-6).
        coordinate = torch.tensor([
            -18.603256225585938, -18.250633239746094, -16.765487670898438,
            -17.879440307617188, -18.04193115234375, -18.70024871826172,
            -16.00243377685547, -16.30375099182129, -17.899837493896484,
            -13.388456344604492, -18.098861694335938, -15.34324836730957,
            -15.576608657836914, -15.860183715820312, -20.210037231445312,
            -13.688745498657227, -15.64361572265625, -18.75749969482422,
            -15.060007095336914, -16.221572875976562, -16.14737892150879,
            -18.99261474609375, -18.128257751464844, -19.29755401611328,
            -16.665428161621094, -17.096172332763672, -17.337886810302734,
            -15.779948234558105, -15.381729125976562, -16.16427230834961,
            -18.53766632080078, -18.79390525817871,
        ], dtype=torch.float32)
        value = coordinate[:, None].repeat(1, runtime.FEATURE_DIM)
        old_fp32 = value - value.mean(dim=0, keepdim=True)
        self.assertGreater(float(old_fp32.mean(dim=0).abs().max()), 3.0e-6)
        canonical = runtime.canonical_action(value)
        self.assertTrue(torch.equal(canonical, old_fp32))
        residual = float(canonical.to(torch.float64).mean(dim=0).abs().max())
        bound = (
            (runtime.TIME_STEPS + 2)
            * torch.finfo(torch.float32).eps
            * max(1.0, float(value.abs().max()))
        )
        self.assertLess(residual, bound)
        self.assertEqual(canonical.dtype, torch.float32)

        hostile = value.clone()
        hostile[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "canonicalization differs"):
            runtime.canonical_action(hostile)
        hostile[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "canonicalization differs"):
            runtime.canonical_action(hostile)
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            runtime.canonical_action(value[:31])

    def test_each_candidate_family_has_exact_actual_b32_payload(self):
        fitted = runtime.FrontierFit(
            frame_mean=torch.zeros((1, 1024)),
            frame_basis=torch.eye(1024)[:, :1].contiguous(),
            clip_mean=torch.zeros((1, 32 * 1024)),
            clip_basis=torch.eye(32 * 1024, 32).contiguous(),
            temporal_basis=torch.eye(32)[:, :4].contiguous(),
            content_basis=torch.eye(1024)[:, :8].contiguous(),
            fit_iid_digest=_digest("fit"), fit_input_sha256=_digest("input"),
            diagnostics={},
        )
        value = torch.arange(32 * 1024, dtype=torch.float32).reshape(32, 1024)
        value = runtime.canonical_action(value)
        specs = [row for row in runtime.candidate_specs(runtime.Config()) if row["payload_numel"] == 32]
        for spec in specs:
            with self.subTest(kind=spec["kind"]):
                self.assertEqual(runtime._encode(value, spec, fitted).numel(), 32)


class VJepa2FrontierV4CSplitTests(unittest.TestCase):
    def _receipt(self):
        assignment, evidence = _assignment_and_evidence()
        fold_pins = {fold: _digest(f"fold-pin:{fold}") for fold in range(5)}
        receipt = {
            "schema_version": runtime.v4a.SCHEMA,
            "status": "V4_FAST_EXACT5_LINEAR_FRONTIER_COMPLETE_BURNED_DEVELOPMENT",
            "implementation": {"implementation_sha256": _digest("v4a-implementation")},
            "frozen_split": {
                "outer_assignment_digest": runtime._object_sha(assignment),
                "fold_iid_digests": {str(key): value for key, value in fold_pins.items()},
            },
            "folds": [
                {
                    "fold_index": fold,
                    "frozen_v2_outer_assignment_digest": runtime._object_sha(assignment),
                    "frozen_v2_fold_iid_digest": fold_pins[fold],
                    "oof_original_count": runtime.OOF_COUNTS[fold],
                    "oof_iid_digest": runtime._object_sha([
                        row["iid"] for row in evidence if row["outer_fold"] == fold
                    ]),
                }
                for fold in range(5)
            ],
            "oof_closure": {
                "embedded_paired_margin_evidence_count": 644,
                "embedded_paired_margin_evidence_sha256": runtime._object_sha(evidence),
                "embedded_paired_margin_evidence": evidence,
            },
        }
        receipt["receipt_digest"] = runtime._object_sha(receipt)
        return assignment, evidence, fold_pins, receipt

    def _write(self, root: Path, receipt):
        path = root / "receipt.json"
        raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("ascii")
        path.write_bytes(raw)
        os.chmod(path, 0o444)
        return path, hashlib.sha256(raw).hexdigest()

    def test_consumes_exact_pinned_embedded_assignment_without_resplit(self):
        assignment, evidence, fold_pins, receipt = self._receipt()
        with tempfile.TemporaryDirectory() as temporary:
            path, file_sha = self._write(Path(temporary).resolve(), receipt)
            with mock.patch.multiple(
                runtime,
                V4A_RECEIPT_FILE_SHA256=file_sha,
                V4A_RECEIPT_DIGEST=receipt["receipt_digest"],
                V4A_EVIDENCE_SHA256=runtime._object_sha(evidence),
                V4A_IMPLEMENTATION_SHA256=_digest("v4a-implementation"),
                OUTER_ASSIGNMENT_DIGEST=runtime._object_sha(assignment),
                FOLD_IID_DIGESTS=fold_pins,
            ):
                observed, loaded = runtime.load_frozen_v4a_split(path, file_sha)
        self.assertEqual(observed, assignment)
        self.assertEqual(loaded["receipt_digest"], receipt["receipt_digest"])

    def test_rejects_semantically_wrong_fold_digest_even_when_all_outer_hashes_match(self):
        assignment, evidence, fold_pins, receipt = self._receipt()
        receipt["folds"][2]["oof_iid_digest"] = "0" * 64
        receipt.pop("receipt_digest")
        receipt["receipt_digest"] = runtime._object_sha(receipt)
        with tempfile.TemporaryDirectory() as temporary:
            path, file_sha = self._write(Path(temporary).resolve(), receipt)
            with mock.patch.multiple(
                runtime,
                V4A_RECEIPT_FILE_SHA256=file_sha,
                V4A_RECEIPT_DIGEST=receipt["receipt_digest"],
                V4A_EVIDENCE_SHA256=runtime._object_sha(evidence),
                V4A_IMPLEMENTATION_SHA256=_digest("v4a-implementation"),
                OUTER_ASSIGNMENT_DIGEST=runtime._object_sha(assignment),
                FOLD_IID_DIGESTS=fold_pins,
            ), self.assertRaises(ValueError):
                runtime.load_frozen_v4a_split(path, file_sha)


class VJepa2FrontierV4CFeatureLoaderTests(unittest.TestCase):
    def _authority(self):
        shared = torch.zeros((32, 1024), dtype=torch.float32)
        anchors = [
            runtime.features.AnchorItem(
                ordinal=index, iid=f"iid-{index:04d}", family=f"family-{index % 28:02d}",
                group_id=_digest(f"group:{index}"),
                instruction_sha256=_digest(f"instruction:{index}"),
                strict=index < 359, path=Path(f"/sealed/{index:04d}.mp4"),
                media_sha256=_digest(f"media:{index}"),
            )
            for index in range(644)
        ]
        implementation = {"sha256": runtime.EXTRACTOR_IMPLEMENTATION_SHA256}
        model_files = [
            {
                "relative_path": name, "logical_path": f"/model/{name}",
                "realpath": f"/model/{name}", "sha256": expected["sha256"],
                "size_bytes": expected["size_bytes"], "mode": 0o444, "nlink": 1,
                "device": 1, "inode": index + 10,
            }
            for index, (name, expected) in enumerate(runtime.features.MODEL_FILES.items())
        ]
        module_rows = [
            {
                "module": name, "source_path": f"/modules/{name}.py",
                "source_realpath": f"/modules/{name}.py", "sha256": expected,
                "size_bytes": 100 + index,
            }
            for index, (name, expected) in enumerate(
                runtime.features.TRANSFORMERS_MODULES.items()
            )
        ]
        model_closure = {
            "model_files_before_and_after_exact": True,
            "transformers_modules_before_and_after_exact": True,
            "model": {
                "model_repo": runtime.features.MODEL_REPO,
                "model_revision": runtime.features.MODEL_REVISION,
                "root": "/model", "root_realpath": "/model", "root_mode": 0o555,
                "exact_top_level_regular_file_count": 3, "files": model_files,
                "closure_sha256": runtime.features.object_sha256(model_files),
            },
            "transformers": {
                "transformers_version": runtime.features.TRANSFORMERS_VERSION,
                "modules": module_rows,
                "closure_sha256": runtime.features.object_sha256(module_rows),
            },
        }
        transform_abi = {
            "exact81_to_base64_formula": "floor(80*k/63), k=0..63",
            "exact81_to_base64_indices_sha256": runtime.features.BASE64_INDICES_SHA256,
            "warp64_formula": "coord[2*i+j]=2*float32(WARP32[i])+j",
            "warp64_coordinates_sha256": runtime.features.WARP64_COORDINATES_SHA256,
            "phase32_block_permutation": list(runtime.features.PHASE_BLOCK_PERMUTATION),
            "transform_axis": "pixel_values_videos temporal dim 1",
            "post_backbone_token_permutation_used": False,
        }
        payloads = []
        bindings = []
        for shard_index in range(6):
            ordinals = [index for index in range(644) if index % 6 == shard_index]
            rows = [
                {
                    "ordinal": index, "iid": anchors[index].iid,
                    "family": anchors[index].family,
                    "strict_selection_gates_all_true": anchors[index].strict,
                    "view_sequences": {
                        name: shared for name in runtime.features.VIEW_NAMES
                    },
                }
                for index in ordinals
            ]
            payloads.append({
                "schema_version": runtime.features.FEATURE_SCHEMA,
                "status": "V4C_VJEPA2_ORDERED_CONTEXTUAL_SHARD_COMPLETE_BURNED_DEVELOPMENT",
                "authority": "feature_mechanics_diagnostic_only",
                "formal_training_authorized": False,
                "paired_ground_truth_claimed": False,
                "implementation": implementation,
                "manifest_sha256": runtime.features.FEATURE_MANIFEST_SHA256,
                "manifest_digest": runtime.features.FEATURE_MANIFEST_DIGEST,
                "source_manifest_sha256": runtime.features.SOURCE_MANIFEST_FILE_SHA256,
                "source_manifest_digest": runtime.features.SOURCE_MANIFEST_DIGEST,
                "shard_index": shard_index, "num_shards": 6,
                "global_anchor_ordinals": ordinals, "record_count": len(rows),
                "processor_call_count": len(rows),
                "frozen_backbone_forward_count": 5 * len(rows),
                "one_processor_then_exact5_separate_forwards_per_anchor": True,
                "model_forward_batching_across_views": False,
                "model_repo": runtime.features.MODEL_REPO,
                "model_revision": runtime.features.MODEL_REVISION,
                "model_dtype": "torch.float16", "skip_predictor": True,
                "model_and_source_closure": model_closure,
                "sampling_and_transform_abi": transform_abi,
                "records": rows,
            })
            bindings.append({
                "path": f"/sealed/shard-{shard_index}.pt",
                "sha256": _digest(f"shard:{shard_index}"),
                "size_bytes": 1000 + shard_index, "mode": 0o444, "nlink": 1,
                "semantic_sha256": _digest(f"semantic:{shard_index}"),
                "single_fd_pre_post_sha256_exact": True,
            })
        semantic = runtime._object_sha([
            {
                "iid": row.iid, "ordinal": row.ordinal,
                "view_sequence_sha256": {
                    name: "a" * 64 for name in runtime.features.VIEW_NAMES
                },
            }
            for row in anchors
        ])
        receipt = {
            "schema_version": runtime.features.RECEIPT_SCHEMA,
            "status": "FEATURES_EXTRACTED_NOT_REPRESENTATION_QUALIFIED",
            "formal_training_authorized": False,
            "paired_ground_truth_claimed": False,
            "burned_development_only": True,
            "implementation": implementation,
            "population": {
                "unique_base_clips": 644, "action_anchor_records": 644,
                "source_records": 0, "total_feature_records": 644,
                "view_evaluations_per_anchor": 5,
                "derived_views_are_independent_samples": False,
                "family_count": 28, "strict_true": 359, "strict_false": 285,
            },
            "exact6_shards": True,
            "each_anchor_processor_call_count": 1,
            "each_anchor_independent_backbone_forward_count": 5,
            "feature_geometry": {
                "views": list(runtime.features.VIEW_NAMES),
                "stored_sequence_per_view": [32, 1024],
                "teacher": "V-JEPA2 ViT-L fpc64 256 frozen FP16 skip_predictor",
                "post_backbone_token_permutation_used": False,
            },
            "exact644_ordered_iid_digest": runtime._object_sha([row.iid for row in anchors]),
            "exact644_record_semantic_sha256": semantic,
            "manifest": {
                "path": "/sealed/manifest.json",
                "sha256": runtime.features.FEATURE_MANIFEST_SHA256,
                "manifest_digest": runtime.features.FEATURE_MANIFEST_DIGEST,
                "source_manifest_sha256": runtime.features.SOURCE_MANIFEST_FILE_SHA256,
                "source_manifest_digest": runtime.features.SOURCE_MANIFEST_DIGEST,
            },
            "model_and_source_closure": model_closure,
            "sampling_and_transform_abi": transform_abi,
            "shards": [
                {"index": index, **binding, "record_count": payloads[index]["record_count"]}
                for index, binding in enumerate(bindings)
            ],
            "action_representation_qualified": False,
            "scientific_confirmation_claimed": False,
            "identity_disentanglement_qualified": False,
            "identity_preservation_qualified": False,
            "prior_generation_qualified": False,
            "generation_qualified": False,
            "video_editing_qualified": False,
            "full644_refit_authorized": False,
            "renderer_authorized": False,
            "inference_authorized": False,
            "web_evaluation_authorized": False,
            "vae_necessary": None,
        }
        receipt["receipt_digest"] = runtime._object_sha(receipt)
        return anchors, payloads, bindings, receipt

    def test_loader_replays_exact6_partitions_and_record_validation(self):
        anchors, payloads, bindings, receipt = self._authority()
        validator = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runtime, "_load_json_sealed", return_value=receipt
        ), mock.patch.object(
            runtime.features, "load_anchor_manifest", return_value=(anchors, {})
        ), mock.patch.object(
            runtime.features, "_load_sealed_shard",
            side_effect=list(zip(payloads, bindings)),
        ), mock.patch.object(
            runtime.features, "_validate_record", validator
        ), mock.patch.object(runtime, "_tensor_sha", return_value="a" * 64):
            records, loaded = runtime.load_v4c_features(
                Path(temporary).resolve(), "1" * 64
            )
        self.assertEqual(len(records), 644)
        self.assertEqual(validator.call_count, 644)
        self.assertEqual(loaded["receipt_digest"], receipt["receipt_digest"])

    def test_loader_rejects_record_swapped_into_wrong_shard(self):
        anchors, payloads, bindings, receipt = self._authority()
        payloads[0]["records"][0]["ordinal"] = 6
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runtime, "_load_json_sealed", return_value=receipt
        ), mock.patch.object(
            runtime.features, "load_anchor_manifest", return_value=(anchors, {})
        ), mock.patch.object(
            runtime.features, "_load_sealed_shard",
            side_effect=list(zip(payloads, bindings)),
        ), mock.patch.object(runtime, "_tensor_sha", return_value="a" * 64), self.assertRaises(ValueError):
            runtime.load_v4c_features(Path(temporary).resolve(), "1" * 64)


class VJepa2FrontierV4COOFTests(unittest.TestCase):
    def test_fold_fit_and_oof_are_disjoint_and_fit_completes_first(self):
        assignment, evidence = _assignment_and_evidence()
        records = _records(evidence)
        events = []

        def fake_fit(rows, config):
            events.append(("fit", {row.iid for row in rows}))
            return SimpleNamespace(
                fit_iid_digest=runtime._object_sha([row.iid for row in rows]),
                fit_input_sha256=_digest("fit-input"), diagnostics={"orthogonal": True},
            )

        def fake_evaluate(rows, fitted, config):
            events.append(("evaluate", {row.iid for row in rows}))
            return [{"iid": row.iid, "family": row.family} for row in rows]

        with mock.patch.object(runtime, "_fit_frontier", side_effect=fake_fit), mock.patch.object(
            runtime, "_evaluate_fold", side_effect=fake_evaluate
        ):
            receipt, evaluated = runtime._run_fold(records, assignment, 0, runtime.Config())
        self.assertEqual([event[0] for event in events], ["fit", "evaluate"])
        self.assertFalse(events[0][1] & events[1][1])
        self.assertEqual(len(evaluated), runtime.OOF_COUNTS[0])
        self.assertTrue(receipt["fit_oof_iid_disjoint"])
        self.assertFalse(receipt["oof_used_for_projection_rank_or_winner_selection"])

    def test_aggregate_lists_same_payload_candidates_without_oof_winner(self):
        _, evidence = _assignment_and_evidence()
        specs = runtime.candidate_specs(runtime.Config())
        rows = []
        for source in evidence:
            margins = {
                negative: {"margin": 1.0, "positive_distance": 0.5, "negative_distance": 1.5}
                for negative in runtime.NEGATIVES
            }
            rows.append({
                "iid": source["iid"], "family": source["family"],
                "teacher": margins,
                "candidates": {spec["name"]: margins for spec in specs},
            })

        def fake_bootstrap(values, families, config, label):
            mean = sum(values) / len(values)
            return {
                "clip_paired_bootstrap": {"lcb": mean},
                "family_cluster_paired_bootstrap": {"lcb": mean},
                "both_lcbs_strictly_gt_zero": mean > 0,
            }

        with mock.patch.object(runtime, "_bootstrap", side_effect=fake_bootstrap):
            metrics, frontier = runtime._aggregate(rows, runtime.Config())
        self.assertEqual(len(metrics["candidates"]), 15)
        for budget in runtime.PAYLOAD_BUDGETS:
            entry = frontier[str(budget)]
            self.assertEqual(len(entry["candidate_names"]), 3)
            self.assertFalse(entry["oof_winner_selected"])
            self.assertTrue(entry["candidates_listed_without_ranking"])


class VJepa2FrontierV4CReceiptAndCLITests(unittest.TestCase):
    def test_run_receipt_is_exact5_unranked_and_fail_closed(self):
        assignment, evidence = _assignment_and_evidence()
        records = _records(evidence)
        v4a_receipt = {
            "receipt_digest": runtime.V4A_RECEIPT_DIGEST,
            "folds": [
                {
                    "oof_original_count": runtime.OOF_COUNTS[fold],
                    "oof_iid_digest": runtime._object_sha([
                        row["iid"] for row in evidence if row["outer_fold"] == fold
                    ]),
                }
                for fold in range(5)
            ],
            "oof_closure": {"embedded_paired_margin_evidence": evidence},
        }
        feature_receipt = {"receipt_digest": _digest("feature-receipt"), "shards": []}
        specs = runtime.candidate_specs(runtime.Config())
        metrics = {
            "candidates": {
                spec["name"]: {"temporal_mechanics_gate": index == 0}
                for index, spec in enumerate(specs)
            }
        }
        frontier = {
            str(budget): {
                "oof_winner_selected": False,
                "candidates_listed_without_ranking": True,
            }
            for budget in runtime.PAYLOAD_BUDGETS
        }
        captured = []

        def fake_fold(all_records, observed_assignment, fold, config):
            selected = [row for row in evidence if row["outer_fold"] == fold]
            evaluated = [
                {"iid": row["iid"], "family": row["family"], "outer_fold": fold}
                for row in selected
            ]
            return {
                "fold_index": fold,
                "oof_original_count": len(selected),
                "oof_iid_digest": runtime._object_sha([row["iid"] for row in selected]),
            }, evaluated

        def fake_writer(path, receipt):
            captured.append(receipt)
            return {
                "path": str(path), "sha256": _digest("output"), "size_bytes": 1,
                "mode": 0o444, "nlink": 1,
                "single_fd_pre_post_sha256_exact": True,
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            v4a_path = root / "v4a.json"
            v4a_path.write_text("{}", encoding="ascii")
            args = argparse.Namespace(
                feature_root=str(root), expected_feature_receipt_sha256="1" * 64,
                v4a_receipt=str(v4a_path),
                expected_v4a_receipt_sha256=runtime.V4A_RECEIPT_FILE_SHA256,
                output=str(root / "frontier.json"),
            )
            with mock.patch.object(
                runtime, "_binding", return_value={"implementation": {"sha256": _digest("frontier")}}
            ), mock.patch.object(
                runtime, "load_frozen_v4a_split", return_value=(assignment, v4a_receipt)
            ), mock.patch.object(
                runtime, "load_v4c_features", return_value=(records, feature_receipt)
            ), mock.patch.object(
                runtime, "_run_fold", side_effect=fake_fold
            ), mock.patch.object(
                runtime, "_aggregate", return_value=(metrics, frontier)
            ), mock.patch.object(
                runtime, "_compact_evidence",
                side_effect=lambda rows: [
                    {"iid": row["iid"], "outer_fold": row["outer_fold"]} for row in rows
                ],
            ), mock.patch.object(
                runtime, "_assert_input_files_unchanged"
            ), mock.patch.object(
                runtime, "_write_json_create_only", side_effect=fake_writer
            ), mock.patch.object(
                runtime, "OUTER_ASSIGNMENT_DIGEST", runtime._object_sha(assignment)
            ), mock.patch.object(runtime.torch, "__version__", "2.7.1+rocm6.3"):
                result = runtime.run_exact5(args)

        receipt = captured[0]
        self.assertEqual(receipt["frozen_split"]["oof_counts"], list(runtime.OOF_COUNTS))
        self.assertFalse(receipt["frozen_split"]["split_recomputed_from_vjepa_feature_values"])
        self.assertFalse(receipt["projection_contract"]["oof_winner_selected"])
        self.assertEqual(
            receipt["qualified_temporal_mechanics_candidates"], [specs[0]["name"]]
        )
        self.assertTrue(
            receipt["qualification_scope"]["candidates_listed_without_ranking_or_selection"]
        )
        for key in (
            "training_authorized", "action_representation_qualified",
            "scientific_confirmation_claimed", "identity_disentanglement_qualified",
            "identity_preservation_qualified", "prior_generation_qualified",
            "generation_qualified", "renderer_qualified", "video_editing_qualified",
            "inference_authorized", "web_evaluation_authorized", "full644_refit_authorized",
        ):
            self.assertFalse(receipt["qualification_scope"][key])
        self.assertIsNone(receipt["qualification_scope"]["vae_necessary"])
        self.assertTrue(result["qualified_candidates_are_not_selected_winners"])

    def test_receipt_writer_is_create_only_sealed_and_single_fd_read_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "frontier.json"
            binding = runtime._write_json_create_only(path, {"finite": 1.0})
            info = path.lstat()
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o444)
            self.assertEqual(info.st_nlink, 1)
            self.assertEqual(binding["sha256"], runtime._file_sha(path))
            self.assertTrue(binding["single_fd_pre_post_sha256_exact"])
            with self.assertRaises(ValueError):
                runtime._write_json_create_only(path, {"finite": 2.0})

    def test_parser_and_main_expose_only_run_exact5(self):
        parser = runtime.build_parser()
        parsed = parser.parse_args([
            "run-exact5", "--feature-root", "/features",
            "--expected-feature-receipt-sha256", "1" * 64,
            "--v4a-receipt", "/v4a.json",
            "--expected-v4a-receipt-sha256", runtime.V4A_RECEIPT_FILE_SHA256,
            "--output", "/out.json",
        ])
        self.assertEqual(parsed.command, "run-exact5")
        self.assertEqual(parsed.handler, runtime.run_exact5)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["unknown"])

        expected = {"receipt": "/out.json"}
        with mock.patch.object(runtime, "run_exact5", return_value=expected) as handler:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = runtime.main([
                    "run-exact5", "--feature-root", "/features",
                    "--expected-feature-receipt-sha256", "1" * 64,
                    "--v4a-receipt", "/v4a.json",
                    "--expected-v4a-receipt-sha256", runtime.V4A_RECEIPT_FILE_SHA256,
                    "--output", "/out.json",
                ])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout.getvalue()), expected)
            handler.assert_called_once()


if __name__ == "__main__":
    unittest.main()

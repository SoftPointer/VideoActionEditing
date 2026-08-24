from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_registered_source8_dinov2_pair_matrix_raw_v1 as diagnostic


def _sources() -> dict[str, dict]:
    order = [
        diagnostic.EXPECTED_MISSING_CORRECT_SOURCE_IID,
        "dog-1", "human-0", "dog-2", "human-1", "dog-3", "human-2", "human-3",
    ]
    return {
        iid: {
            "iid": iid,
            "row_id": f"row-{ordinal}",
            "analysis_split": "fit" if ordinal < 4 else "confirmation",
            "actor_family": "dog" if iid.startswith("dog") or iid == diagnostic.EXPECTED_MISSING_CORRECT_SOURCE_IID else "human",
            "actor_group_id": f"actor-{ordinal}",
            "scene_group_id": f"scene-{ordinal}",
            "source_video": f"/sealed/source-{ordinal}.mp4",
            "source_video_sha256": f"{ordinal + 1:064x}",
        }
        for ordinal, iid in enumerate(order)
    }


def _manifest() -> dict:
    sources = _sources()
    return {
        "source_manifest_order": list(sources),
        "sources": list(sources.values()),
        "matrix_registration": diagnostic._manifest_design(sources),
    }


def _cache(manifest: dict) -> dict:
    import torch

    cache = {}
    for ordinal, binding in enumerate(manifest["sources"]):
        base = torch.arange(17 * 8, dtype=torch.float32).reshape(17, 8) + ordinal + 1
        global_feature = torch.nn.functional.normalize(base, dim=-1)
        dense_feature = global_feature[:, None, :].repeat(1, 4, 1)
        cache[binding["iid"]] = {
            "global": global_feature,
            "dense": dense_feature,
            "entry": {
                "ordinal": ordinal,
                "iid": binding["iid"],
                "actor_family": binding["actor_family"],
                "source_video": binding["source_video"],
                "source_video_sha256": binding["source_video_sha256"],
                "global_feature_sha256": f"{ordinal + 20:064x}",
                "dense_feature_sha256": f"{ordinal + 40:064x}",
                "decode": {
                    "artifact_sha256": binding["source_video_sha256"],
                    "decoded_rgb_sha256": f"{ordinal + 60:064x}",
                    "frame_count": 81,
                    "fps_numerator": 25,
                    "fps_denominator": 1,
                    "time_base_numerator": 1,
                    "time_base_denominator": 12800,
                    "pts_step": 512,
                    "pts_sha256": f"{ordinal + 80:064x}",
                    "width": 736,
                    "height": 704,
                    "selected_frame_indices": list(range(0, 81, 5)),
                    "selected_rgb_sha256": f"{ordinal + 100:064x}",
                    "preprocessed_tensor_sha256": f"{ordinal + 120:064x}",
                },
                "feature_geometry": dict(diagnostic.EXPECTED_FEATURE_GEOMETRY),
            },
        }
    return cache


def _cache_summary(manifest: dict, cache: dict) -> dict:
    entries = [cache[iid]["entry"] for iid in manifest["source_manifest_order"]]
    hashes = [{
        "iid": row["iid"],
        "source_video_sha256": row["source_video_sha256"],
        "global_feature_sha256": row["global_feature_sha256"],
        "dense_feature_sha256": row["dense_feature_sha256"],
    } for row in entries]
    return {
        "cache_scope": "one_rank_process_exact8",
        "source_count": 8,
        "source_manifest_order": manifest["source_manifest_order"],
        "all_exact8_sources_warmed_before_pair_computation": True,
        "source_features_held_in_cpu_memory_until_worker_exit": True,
        "source_files_retained_open_until_worker_exit": False,
        "entries": entries,
        "entries_sha256": diagnostic.core.object_sha256(entries),
        "feature_hash_map": hashes,
        "feature_hash_map_sha256": diagnostic.core.object_sha256(hashes),
    }


def _frozen_pins() -> dict:
    return {
        "checkpoint_root": diagnostic.EXPECTED_CHECKPOINT_ROOT,
        "checkpoint_manifest_path": diagnostic.EXPECTED_CHECKPOINT_MANIFEST_PATH,
        "evaluator_spec_path": diagnostic.EXPECTED_EVALUATOR_SPEC_PATH,
        "visual_scorer_path": diagnostic.EXPECTED_VISUAL_SCORER_PATH,
        "visual_contract_path": diagnostic.EXPECTED_VISUAL_CONTRACT_PATH,
        "evaluator_spec_sha256": diagnostic.EXPECTED_EVALUATOR_SPEC_SHA256,
        "visual_scorer_sha256": diagnostic.EXPECTED_VISUAL_SCORER_SHA256,
        "visual_contract_sha256": diagnostic.EXPECTED_VISUAL_CONTRACT_SHA256,
        "runtime_versions": dict(diagnostic.EXPECTED_RUNTIME_VERSIONS),
        "checkpoint_manifest_sha256": diagnostic.EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_verified_entries_digest": "1" * 64,
        "model_adapter_id": "transformers-dinov2-image-processor-v1",
        "model_architecture_id": "dinov2",
        "checkpoint_config_sha256": "2" * 64,
        "preprocessor_config_sha256": "3" * 64,
        "checkpoint_file_count": 4,
        "num_register_tokens": 0,
        "model_image_size": 518,
        "patch_size": 14,
        "preprocessor_golden_input_sha256": (
            diagnostic.EXPECTED_PREPROCESSOR_GOLDEN_INPUT_SHA256
        ),
        "preprocessor_golden_output_sha256": (
            diagnostic.EXPECTED_PREPROCESSOR_GOLDEN_OUTPUT_SHA256
        ),
        "preprocessor_golden_output_shape": [1, 3, 224, 224],
        "selected_frame_indices": list(diagnostic.EVAL_FRAME_INDICES),
        "feature_geometry": dict(diagnostic.EXPECTED_FEATURE_GEOMETRY),
    }


def _visual_evidence() -> dict:
    pins = _frozen_pins()
    model = {
        "adapter_id": pins["model_adapter_id"],
        "architecture_id": pins["model_architecture_id"],
        "checkpoint_manifest_sha256": diagnostic.EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_config_sha256": pins["checkpoint_config_sha256"],
        "preprocessor_config_sha256": pins["preprocessor_config_sha256"],
        "checkpoint_file_count": pins["checkpoint_file_count"],
        "verified_entries_digest": pins["checkpoint_verified_entries_digest"],
        "preprocessor_golden_input_sha256": (
            diagnostic.EXPECTED_PREPROCESSOR_GOLDEN_INPUT_SHA256
        ),
        "preprocessor_golden_output_sha256": (
            diagnostic.EXPECTED_PREPROCESSOR_GOLDEN_OUTPUT_SHA256
        ),
        "preprocessor_golden_output_shape": [1, 3, 224, 224],
        "every_checkpoint_file_verified": True,
        "all_parameters_frozen": True,
        "trainable_parameter_tensors": 0,
        "parameter_tensor_count": 12,
        "parameter_element_count": 345,
        "parameter_metadata_digest": "4" * 64,
        "missing_key_count": 0,
        "unexpected_key_count": 0,
        "mismatched_key_count": 0,
        "loading_error_count": 0,
        "runtime_versions": dict(diagnostic.EXPECTED_RUNTIME_VERSIONS),
    }
    return {
        "checkpoint_root": diagnostic.EXPECTED_CHECKPOINT_ROOT,
        "checkpoint_manifest_path": diagnostic.EXPECTED_CHECKPOINT_MANIFEST_PATH,
        "evaluator_spec_path": diagnostic.EXPECTED_EVALUATOR_SPEC_PATH,
        "visual_scorer_path": diagnostic.EXPECTED_VISUAL_SCORER_PATH,
        "visual_contract_path": diagnostic.EXPECTED_VISUAL_CONTRACT_PATH,
        "evaluator_spec_sha256": diagnostic.EXPECTED_EVALUATOR_SPEC_SHA256,
        "visual_scorer_sha256": diagnostic.EXPECTED_VISUAL_SCORER_SHA256,
        "visual_contract_sha256": diagnostic.EXPECTED_VISUAL_CONTRACT_SHA256,
        "checkpoint_manifest_raw_sha256": (
            diagnostic.EXPECTED_CHECKPOINT_MANIFEST_SHA256
        ),
        "model_evidence": model,
        "model_evidence_sha256": diagnostic.object_sha256(model),
        "candidate_or_proposal_media_consulted": False,
        "candidate_metric_fields_queried": False,
        "candidate_metric_values_used": False,
        "identity_authority": False,
        "scientific_claim_authorized": False,
    }


class RegisteredSource8PairMatrixContractTest(unittest.TestCase):
    def test_hard_pins_and_authority_are_closed(self) -> None:
        self.assertEqual(
            diagnostic.EXPECTED_SOURCE_MANIFEST_SHA256,
            "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9",
        )
        self.assertEqual(
            diagnostic.EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256,
            "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f",
        )
        self.assertEqual(
            diagnostic.EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256,
            "257d3aafaaee126ff2c1a061413d01bd0457676eb5d1ee027671221a5a794218",
        )
        self.assertEqual(diagnostic.EXPECTED_RUNTIME_VERSIONS["transformers_version"], "4.53.2")
        self.assertEqual(diagnostic.EXPECTED_RUNTIME_VERSIONS, {
            "python_version": "3.12.13",
            "torch_version": "2.7.1+rocm6.3",
            "torch_hip_version": "6.3.42131-fa1d09cbd",
            "transformers_version": "4.53.2",
            "safetensors_version": "0.8.0rc0",
            "av_version": "13.1.0",
            "numpy_version": "1.26.4",
            "pillow_version": "11.3.0",
        })
        self.assertEqual(
            diagnostic.EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "b61f251411f0d8f6a617b67d0b903c333d16c77fb6b3f49507225884d4aed0ea",
        )
        self.assertEqual(
            diagnostic.EXPECTED_PREPROCESSOR_GOLDEN_INPUT_SHA256,
            "d8217ce3a86de051a4affd701c965befd12584cce51902c9f266fff952ebd18a",
        )
        self.assertEqual(
            diagnostic.EXPECTED_PREPROCESSOR_GOLDEN_OUTPUT_SHA256,
            "b5ef31a8754b854ce64dcf49a79949e22ff9219a7db5d2dfd5fec1ed0602fb6a",
        )
        self.assertEqual(
            diagnostic.EXPECTED_EVALUATOR_SPEC_PATH,
            diagnostic.EXPECTED_EXPERIMENT_ROOT
            + "/inputs/pair_v5_source_bound_preservation_evaluator_7c4c837_v1.json",
        )
        self.assertEqual(
            diagnostic.EXPECTED_VISUAL_SCORER_SHA256,
            "9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39",
        )
        self.assertEqual(
            diagnostic.EXPECTED_VISUAL_CONTRACT_SHA256,
            "183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a",
        )
        self.assertTrue(diagnostic.AUTHORITY)
        self.assertTrue(all(value is False for value in diagnostic.AUTHORITY.values()))
        self.assertFalse(diagnostic.LIMITATION["candidate_or_proposal_media_consulted"])
        self.assertFalse(diagnostic.LIMITATION["candidate_metric_fields_queried"])
        self.assertFalse(diagnostic.LIMITATION["candidate_metric_values_used"])
        self.assertFalse(diagnostic.LIMITATION["formal_retained_source_fd_closure_satisfied"])

    def test_registration_is_manifest_order_exact64_and_actor_closed(self) -> None:
        sources = _sources()
        design = diagnostic._manifest_design(sources)
        self.assertEqual(design["source_manifest_order"], list(sources))
        self.assertEqual(design["matrix_shape"], [8, 8])
        self.assertEqual(len(design["cells"]), 64)
        self.assertEqual(sum(row["relationship"] == "same_actor" for row in design["cells"]), 32)
        self.assertEqual(sum(row["relationship"] == "cross_actor" for row in design["cells"]), 32)
        self.assertEqual(sum(row["diagonal"] for row in design["cells"]), 8)
        self.assertEqual(sum(row["registered_all3_directed_pair"] for row in design["cells"]), 24)
        self.assertFalse(design["candidate_or_proposal_media_or_metrics_consulted_during_registration"])
        for ordinal, row in enumerate(design["cells"]):
            self.assertEqual(row["matrix_ordinal"], ordinal)
            self.assertEqual(row["row_ordinal"], ordinal // 8)
            self.assertEqual(row["column_ordinal"], ordinal % 8)

    def test_registration_rejects_non_exact8(self) -> None:
        sources = _sources()
        sources.pop(next(iter(sources)))
        with self.assertRaises(diagnostic.Source8MatrixError):
            diagnostic._manifest_design(sources)

    def test_all_rows_form_symmetric_exact64_with_exact_self(self) -> None:
        manifest = _manifest()
        cache = _cache(manifest)
        rows = [diagnostic.pair_row(rank, manifest, cache) for rank in range(8)]
        cells = [cell for row in rows for cell in row["cells"]]
        self.assertEqual(len(cells), 64)
        by_coordinate = {(row["row_ordinal"], row["column_ordinal"]): row for row in cells}
        for left in range(8):
            self.assertEqual(by_coordinate[left, left]["global_mean_mapped_cosine"], 1.0)
            self.assertEqual(by_coordinate[left, left]["dense_median_mapped_cosine"], 1.0)
            for right in range(8):
                self.assertAlmostEqual(
                    by_coordinate[left, right]["global_mean_mapped_cosine"],
                    by_coordinate[right, left]["global_mean_mapped_cosine"],
                    places=7,
                )
                self.assertAlmostEqual(
                    by_coordinate[left, right]["dense_median_mapped_cosine"],
                    by_coordinate[right, left]["dense_median_mapped_cosine"],
                    places=7,
                )

    def test_row_validator_rejects_sha_or_ordinal_rebinding(self) -> None:
        manifest = _manifest()
        row = diagnostic.pair_row(0, manifest, _cache(manifest))
        self.assertEqual(len(diagnostic._validate_row(row, rank=0, manifest=manifest)), 8)
        for key, replacement in (
            ("matrix_ordinal", 63),
            ("column_source_video_sha256", "f" * 64),
            ("column_source_iid", "rebound"),
        ):
            hostile = copy.deepcopy(row)
            hostile["cells"][0][key] = replacement
            with self.assertRaises(diagnostic.Source8MatrixError):
                diagnostic._validate_row(hostile, rank=0, manifest=manifest)

    def test_cache_validator_binds_all8_iid_sha_ordinal_and_hashes(self) -> None:
        manifest = _manifest()
        cache = _cache(manifest)
        summary = _cache_summary(manifest, cache)
        hashes = diagnostic._validate_cache(summary, manifest=manifest)
        self.assertEqual(len(hashes), 8)
        hostile = copy.deepcopy(summary)
        hostile["entries"][3]["ordinal"] = 4
        hostile["entries_sha256"] = diagnostic.core.object_sha256(hostile["entries"])
        with self.assertRaises(diagnostic.Source8MatrixError):
            diagnostic._validate_cache(hostile, manifest=manifest)
        hostile = copy.deepcopy(summary)
        hostile["feature_hash_map"][1]["dense_feature_sha256"] = "f" * 64
        hostile["feature_hash_map_sha256"] = diagnostic.core.object_sha256(hostile["feature_hash_map"])
        with self.assertRaises(diagnostic.Source8MatrixError):
            diagnostic._validate_cache(hostile, manifest=manifest)
        hostile = copy.deepcopy(summary)
        hostile["entries"][2]["decode"]["fps_numerator"] = 24
        hostile["entries_sha256"] = diagnostic.core.object_sha256(hostile["entries"])
        with self.assertRaises(diagnostic.Source8MatrixError):
            diagnostic._validate_cache(hostile, manifest=manifest)
        hostile = copy.deepcopy(summary)
        hostile["entries"][2]["decode"]["preprocessed_tensor_sha256"] = "0" * 64
        hostile["entries_sha256"] = diagnostic.core.object_sha256(hostile["entries"])
        with self.assertRaises(diagnostic.Source8MatrixError):
            diagnostic._validate_cache(hostile, manifest=manifest)
        hostile = copy.deepcopy(summary)
        hostile["entries"][2]["decode"]["selected_rgb_sha256"] = "0" * 64
        hostile["entries_sha256"] = diagnostic.core.object_sha256(hostile["entries"])
        with self.assertRaises(diagnostic.Source8MatrixError):
            diagnostic._validate_cache(hostile, manifest=manifest)
        hostile = copy.deepcopy(summary)
        hostile["entries"][2]["source_video"] = "/sealed/rebound.mp4"
        hostile["entries_sha256"] = diagnostic.core.object_sha256(hostile["entries"])
        with self.assertRaises(diagnostic.Source8MatrixError):
            diagnostic._validate_cache(hostile, manifest=manifest)

    def test_visual_evidence_is_deep_closed_and_digest_bound(self) -> None:
        pins = _frozen_pins()
        evidence = _visual_evidence()
        self.assertEqual(
            diagnostic._validate_visual_evidence(evidence, frozen_pins=pins),
            evidence,
        )
        hostile_cases = []
        hostile = copy.deepcopy(evidence)
        hostile["extra"] = False
        hostile_cases.append(hostile)
        hostile = copy.deepcopy(evidence)
        hostile["model_evidence"]["runtime_versions"]["av_version"] = "14.4.0"
        hostile["model_evidence_sha256"] = diagnostic.object_sha256(
            hostile["model_evidence"]
        )
        hostile_cases.append(hostile)
        hostile = copy.deepcopy(evidence)
        hostile["model_evidence"]["missing_key_count"] = 1
        hostile["model_evidence_sha256"] = diagnostic.object_sha256(
            hostile["model_evidence"]
        )
        hostile_cases.append(hostile)
        hostile = copy.deepcopy(evidence)
        hostile["checkpoint_root"] += "-rebound"
        hostile_cases.append(hostile)
        hostile = copy.deepcopy(evidence)
        hostile["model_evidence"]["parameter_metadata_digest"] = "5" * 64
        hostile_cases.append(hostile)
        for hostile in hostile_cases:
            with self.subTest(hostile=hostile):
                with self.assertRaises(diagnostic.Source8MatrixError):
                    diagnostic._validate_visual_evidence(hostile, frozen_pins=pins)

    def test_source_has_standalone_import_and_preprocess_hash_closure(self) -> None:
        source = Path(diagnostic.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnose_saic_partial", source)
        self.assertNotIn("all3.", source)
        self.assertNotIn("sys.path.insert", source)
        self.assertIn("spec_from_file_location", source)
        self.assertIn("getattr(module, \"__file__\", None)", source)
        self.assertIn("scorer.tensor_sha256(normalized)", source)
        self.assertIn("list(global_feature.shape) != [17, 768]", source)
        self.assertIn("list(dense_feature.shape) != [17, 256, 768]", source)

    def test_exact_module_loader_rejects_preload_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "sealed_module.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            hostile_name = "_source8_hostile_preloaded_module"
            sys.modules[hostile_name] = object()
            try:
                with self.assertRaises(diagnostic.Source8MatrixError):
                    diagnostic._load_exact_module(
                        hostile_name,
                        source,
                        expected_path=str(source),
                        expected_sha256=digest,
                    )
            finally:
                sys.modules.pop(hostile_name, None)
            symlink = root / "rebound.py"
            symlink.symlink_to(source)
            with self.assertRaises(diagnostic.Source8MatrixError):
                diagnostic._load_exact_module(
                    "_source8_hostile_symlink_module",
                    symlink,
                    expected_path=str(symlink),
                    expected_sha256=digest,
                )

    def test_legacy_regression_uses_only_hashes_and_pair_bindings_after_matrix(self) -> None:
        manifest = _manifest()
        cache = _cache(manifest)
        feature_map = _cache_summary(manifest, cache)["feature_hash_map"]
        design = manifest["matrix_registration"]["cells"]
        registered = {
            (row["row_source_iid"], row["column_source_iid"])
            for row in design if row["registered_all3_directed_pair"]
        }
        executed = sorted(
            pair for pair in registered
            if pair[0] != diagnostic.EXPECTED_MISSING_CORRECT_SOURCE_IID
        )
        unsigned = {
            "schema_version": diagnostic.LEGACY_ALL3_AGGREGATE_SCHEMA,
            "diagnostic_source_sha256": diagnostic.LEGACY_ALL3_SOURCE_SHA256,
            "world_size": 8,
            "candidate_count": diagnostic.LEGACY_ALL3_CANDIDATE_COUNT,
            "executed_directed_source_pair_count": 21,
            "executed_directed_source_pairs": [
                {"correct_source_iid": left, "negative_source_iid": right}
                for left, right in executed
            ],
            "executed_directed_source_pairs_sha256": diagnostic.object_sha256([
                {"correct_source_iid": left, "negative_source_iid": right}
                for left, right in executed
            ]),
            "cross_rank_source_feature_cache_consistency": {
                "per_source_feature_hashes": feature_map,
            },
            # Present and integrity-bound, but the regression must not query/use it.
            "candidate_results": [{"hostile_metric_sentinel": 0.123456789}],
        }
        value = {**unsigned, "receipt_digest": diagnostic.core.object_sha256(unsigned)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.touch()
            with mock.patch.object(
                diagnostic.core,
                "_strict_json",
                return_value=(value, diagnostic.EXPECTED_LEGACY_ALL3_AGGREGATE_SHA256),
            ), mock.patch.object(diagnostic.core, "_plain_file", return_value=path):
                result = diagnostic._legacy_regression(
                    diagnostic.EXPECTED_LEGACY_ALL3_AGGREGATE_PATH,
                    diagnostic.EXPECTED_LEGACY_ALL3_AGGREGATE_SHA256,
                    feature_map=feature_map,
                    registered_pairs=registered,
                )
        self.assertTrue(result["matrix_was_registered_computed_and_sharded_before_legacy_aggregate_open"])
        self.assertTrue(result["legacy_aggregate_bytes_parsed"])
        self.assertFalse(result["candidate_metric_fields_queried"])
        self.assertFalse(result["candidate_metric_values_used"])
        self.assertEqual(result["executed_same_actor_directed_pair_binding_match_count"], 21)
        self.assertEqual(result["new_missing_correct_directed_pair_count"], 3)
        hostile_unsigned = copy.deepcopy(unsigned)
        hostile_unsigned["executed_directed_source_pairs"][0][
            "negative_source_iid"
        ] = "cross-actor-rebind"
        hostile_unsigned["executed_directed_source_pairs_sha256"] = (
            diagnostic.object_sha256(hostile_unsigned["executed_directed_source_pairs"])
        )
        hostile_value = {
            **hostile_unsigned,
            "receipt_digest": diagnostic.object_sha256(hostile_unsigned),
        }
        with mock.patch.object(
            diagnostic.core,
            "_strict_json",
            return_value=(hostile_value, diagnostic.EXPECTED_LEGACY_ALL3_AGGREGATE_SHA256),
        ):
            with self.assertRaises(diagnostic.Source8MatrixError):
                diagnostic._legacy_regression(
                    diagnostic.EXPECTED_LEGACY_ALL3_AGGREGATE_PATH,
                    diagnostic.EXPECTED_LEGACY_ALL3_AGGREGATE_SHA256,
                    feature_map=feature_map,
                    registered_pairs=registered,
                )

    def test_parser_and_launcher_are_fail_closed(self) -> None:
        parser = diagnostic.build_parser()
        subcommands = next(
            action for action in parser._actions if getattr(action, "choices", None)
        ).choices
        self.assertEqual(set(subcommands), {"build-manifest", "preflight", "worker", "aggregate"})
        launcher = (
            METHOD_ROOT / "scripts"
            / "auh_diagnose_saic_registered_source8_dinov2_pair_matrix_raw_v1.sh"
        ).read_text(encoding="utf-8")
        source_path = Path(diagnostic.__file__)
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        self.assertIn('[[ "$#" -ne 20 ]]', launcher)
        self.assertIn('"${SLURM_JOB_ID:-}" == "$allocation_job_id"', launcher)
        self.assertIn("runtime/venv-transformers-4.53.2/bin/python", launcher)
        self.assertIn("for rank in 0 1 2 3 4 5 6 7", launcher)
        self.assertIn('readonly expected_diagnostic_sha256=' + source_sha256, launcher)
        self.assertIn('diagnostic caller substitution is forbidden', launcher)
        self.assertIn('"$env_bin" -i', launcher)
        self.assertGreaterEqual(launcher.count('"$python_bin" -I -B'), 4)
        self.assertIn('"ROCR_VISIBLE_DEVICES=$rank"', launcher)
        self.assertIn('MIOPEN_USER_DB_PATH="$cache/miopen-user"', launcher)
        self.assertIn('MIOPEN_CUSTOM_CACHE_DIR="$cache/miopen-custom"', launcher)
        self.assertIn('XDG_CACHE_HOME="$cache/xdg"', launcher)
        self.assertIn('remove_runtime_scratch_exact', launcher)
        self.assertIn('trap cleanup_on_exit EXIT', launcher)
        self.assertIn('validate_output_set final', launcher)
        self.assertIn('top_log="$output_root/top-$mode.log"', launcher)
        self.assertIn('"$chmod_bin" 0400 -- "$path"', launcher)
        self.assertIn(diagnostic.EXPECTED_CHECKPOINT_MANIFEST_PATH, launcher)
        self.assertIn(diagnostic.EXPECTED_EVALUATOR_SPEC_PATH, launcher)
        self.assertIn(diagnostic.EXPECTED_VISUAL_SCORER_PATH, launcher)
        self.assertIn(diagnostic.EXPECTED_VISUAL_CONTRACT_PATH, launcher)
        self.assertIn(diagnostic.EXPECTED_LEGACY_ALL3_AGGREGATE_SHA256, launcher)
        self.assertNotIn('sha256sum "$diagnostic_source"', launcher)
        self.assertNotIn('env -u ', launcher)
        self.assertNotIn("sbatch", launcher)
        self.assertNotIn("salloc", launcher)


if __name__ == "__main__":
    unittest.main()

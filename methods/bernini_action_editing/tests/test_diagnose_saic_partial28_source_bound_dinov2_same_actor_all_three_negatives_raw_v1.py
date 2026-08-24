from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_partial28_source_bound_dinov2_same_actor_all_three_negatives_raw_v1 as diagnostic


def _seal(value: dict) -> dict:
    unsigned = copy.deepcopy(value)
    unsigned.pop("receipt_digest", None)
    return {**unsigned, "receipt_digest": diagnostic.core.object_sha256(unsigned)}


def _synthetic_input_manifest() -> dict:
    source_order = [
        "7b88a1ca1f804f41",
        "841b5e0080a1441d",
        "a35b590961d24694",
        "31c34509415745ca",
        "99cde432839f4240",
        "6ea45d35943742bb",
        "311c82f83eca4a7f",
        "6d346c38cf504493",
    ]
    dog_iids = {
        "6ea45d35943742bb", "7b88a1ca1f804f41",
        "841b5e0080a1441d", "99cde432839f4240",
    }
    sources = {
        iid: {
            "iid": iid,
            "actor_family": "dog" if iid in dog_iids else "human",
            "source_video_sha256": f"{index + 1:064x}",
        }
        for index, iid in enumerate(source_order)
    }
    design = diagnostic.negative_design(sources)
    executed_iids = [
        iid for iid in source_order
        if iid not in diagnostic.EXPECTED_MISSING_CORRECT_SOURCE_IIDS
    ]
    attempts = []
    for index in range(diagnostic.EXPECTED_ATTEMPT_COUNT):
        iid = executed_iids[index % len(executed_iids)]
        attempts.append({
            "candidate_id": f"candidate-{index:02d}",
            "correct_source": copy.deepcopy(sources[iid]),
            "negative_sources": [
                copy.deepcopy(sources[negative_iid])
                for negative_iid in design["negative_iids_by_correct_iid"][iid]
            ],
            "legacy_cyclic_negative_iid": design[
                "legacy_cyclic_negative_iid_by_correct_iid"
            ][iid],
        })
    source_sha = "a" * 64
    unsigned = {
        "schema_version": diagnostic.INPUT_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "attempts_root": "/synthetic/attempts",
        "root_spec_raw_sha256": "b" * 64,
        "attempt_count": diagnostic.EXPECTED_ATTEMPT_COUNT,
        "world_size": diagnostic.EXPECTED_WORLD_SIZE,
        "partition_rule": "candidate_order_index_modulo_world_size",
        "selected_frame_indices": list(diagnostic.core.EVAL_FRAME_INDICES),
        "source_manifest": {
            "raw_sha256": diagnostic.EXPECTED_SOURCE_MANIFEST_SHA256,
            "bound_files_verified": True,
            "negative_source_policy": diagnostic.NEGATIVE_SOURCE_POLICY,
            "source_manifest_order": source_order,
            "negative_registration_sha256": diagnostic.core.object_sha256(design),
        },
        "negative_design": design,
        "legacy_cyclic_regression": {
            "path": "/synthetic/legacy-aggregate.json",
            "raw_sha256": diagnostic.EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256,
            "receipt_digest": diagnostic.EXPECTED_LEGACY_CYCLIC_AGGREGATE_RECEIPT_DIGEST,
            "schema_version": diagnostic.FROZEN_CYCLIC_AGGREGATE_SCHEMA,
            "diagnostic_source_sha256": diagnostic.FROZEN_CYCLIC_SOURCE_SHA256,
            "input_manifest_sha256": diagnostic.EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256,
            "candidate_count": diagnostic.EXPECTED_ATTEMPT_COUNT,
            "required_for_aggregate_regression": True,
        },
        "attempts": attempts,
        "operational_limitation": dict(diagnostic.OPERATIONAL_LIMITATION),
        "authority": dict(diagnostic.AUTHORITY_CLOSURE),
    }
    return _seal(unsigned)


def _roundtrip_input_manifest(value: dict) -> dict:
    raw = diagnostic.core.canonical_json_bytes(value)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "input-manifest.json"
        path.write_bytes(raw)
        loaded, actual_sha = diagnostic.load_input_manifest(
            path,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_sha256=value["diagnostic_source_sha256"],
        )
    if actual_sha != hashlib.sha256(raw).hexdigest():
        raise AssertionError("round-trip raw SHA differs")
    return loaded


def _raw_metrics() -> dict:
    global_correct, global_wrong, global_self = 0.8, 0.6, 1.0
    dense_correct, dense_wrong, dense_self = 0.7, 0.5, 1.0
    global_margin = global_correct - global_wrong
    dense_margin = dense_correct - dense_wrong
    global_denominator = global_self - global_wrong
    dense_denominator = dense_self - dense_wrong
    return {
        "measurement_label": "frozen_dinov2_source_bound_raw_proxy_only",
        "global_candidate_correct": global_correct,
        "global_candidate_wrong": global_wrong,
        "global_correct_minus_wrong_margin": global_margin,
        "global_source_self_upper_bound": global_self,
        "dense_candidate_correct": dense_correct,
        "dense_candidate_wrong": dense_wrong,
        "dense_correct_minus_wrong_margin": dense_margin,
        "dense_source_self_upper_bound": dense_self,
        "thresholds": None,
        **diagnostic.AUTHORITY_CLOSURE,
        "global_wrong_normalized_contrast_denominator": global_denominator,
        "global_wrong_normalized_contrast": global_margin / global_denominator,
        "dense_wrong_normalized_contrast_denominator": dense_denominator,
        "dense_wrong_normalized_contrast": dense_margin / dense_denominator,
        "normalized_contrast_zero_when_denominator_nonpositive": True,
        "descriptive_only": True,
        "operational_diagnostic_only": True,
        "multi_negative_proxy_authority": False,
        "formal_retained_source_fd_authority": False,
    }


def _candidate_fixture() -> tuple[dict, dict]:
    correct = {"iid": "dog-0", "source_video_sha256": "1" * 64}
    negatives = [
        {"iid": f"dog-{index}", "source_video_sha256": f"{index + 1:064x}"}
        for index in (1, 2, 3)
    ]
    expected = {
        "candidate_id": "candidate-00",
        "correct_source": correct,
        "negative_sources": negatives,
        "legacy_cyclic_negative_iid": negatives[0]["iid"],
    }
    negative_results = []
    for ordinal, negative in enumerate(negatives):
        negative_results.append({
            "negative_ordinal_in_manifest_order": ordinal,
            "correct_source_iid": correct["iid"],
            "correct_source_video_sha256": correct["source_video_sha256"],
            "negative_source_iid": negative["iid"],
            "negative_source_video_sha256": negative["source_video_sha256"],
            "is_legacy_cyclic_negative": ordinal == 0,
            "raw_metrics": _raw_metrics(),
            "authority": dict(diagnostic.AUTHORITY_CLOSURE),
        })
    result = {
        "candidate_id": expected["candidate_id"],
        "candidate_binding": copy.deepcopy(expected),
        "source_features_served_from_exact8_rank_cache": True,
        "negative_results": negative_results,
        "candidate_descriptive_summary": diagnostic.descriptive_candidate_summary(
            [row["raw_metrics"] for row in negative_results]
        ),
        "operational_limitation": dict(diagnostic.OPERATIONAL_LIMITATION),
        "authority": dict(diagnostic.AUTHORITY_CLOSURE),
    }
    return expected, result


class R6Exact28AllThreeSameActorRawContractTest(unittest.TestCase):
    def test_partition_is_exactly_once_and_balanced(self) -> None:
        partitions = [diagnostic.partition_indices(28, rank, 8) for rank in range(8)]
        self.assertEqual([len(row) for row in partitions], [4, 4, 4, 4, 3, 3, 3, 3])
        flattened = [index for row in partitions for index in row]
        self.assertEqual(sorted(flattened), list(range(28)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_main_restores_both_nested_module_layers_and_self_identity(self) -> None:
        source_sha256 = (
            "6e25e0ec0a170a816bff50a4565b302704ca3e42213de2d89f103469930ed1c3"
        )
        for module in (diagnostic.frozen, diagnostic.core):
            module.__file__ = "/tmp/polluted-r6-source.py"
            module.SCHEMA_VERSION = "polluted"
            module.INPUT_SCHEMA = "polluted-input"
            module.SHARD_SCHEMA = "polluted-shard"
            module.AGGREGATE_SCHEMA = "polluted-aggregate"
            module.PREFLIGHT_SCHEMA = "polluted-preflight"
            module.EXPECTED_ATTEMPT_COUNT = 47
            module.EXPECTED_WORLD_SIZE = 1
            module.AUTHORITY_CLOSURE = {"scientific_claim_authorized": True}
            module.partition_indices = lambda *_args: (999,)

        with self.assertRaises(SystemExit) as raised:
            diagnostic.main(["--help"])
        self.assertEqual(raised.exception.code, 0)

        expected_file = diagnostic.__file__
        for module in (diagnostic.frozen, diagnostic.core):
            self.assertEqual(module.__file__, expected_file)
            self.assertEqual(module.SCHEMA_VERSION, diagnostic.SCHEMA_VERSION)
            self.assertEqual(module.INPUT_SCHEMA, diagnostic.INPUT_SCHEMA)
            self.assertEqual(module.SHARD_SCHEMA, diagnostic.SHARD_SCHEMA)
            self.assertEqual(module.AGGREGATE_SCHEMA, diagnostic.AGGREGATE_SCHEMA)
            self.assertEqual(module.PREFLIGHT_SCHEMA, diagnostic.PREFLIGHT_SCHEMA)
            self.assertEqual(module.EXPECTED_ATTEMPT_COUNT, 28)
            self.assertEqual(module.EXPECTED_WORLD_SIZE, 8)
            self.assertEqual(module.AUTHORITY_CLOSURE, diagnostic.AUTHORITY_CLOSURE)
            self.assertIs(module.partition_indices, diagnostic.partition_indices)
        self.assertEqual(diagnostic.core._verify_self(source_sha256), source_sha256)

    def test_negative_map_uses_actor_and_manifest_order_only(self) -> None:
        order = ["dog-z", "human-b", "dog-a", "human-d", "dog-m", "human-a", "dog-b", "human-c"]
        sources = {
            iid: {
                "iid": iid,
                "actor_family": "dog" if iid.startswith("dog-") else "human",
                "source_video_sha256": f"{index + 1:064x}",
            }
            for index, iid in enumerate(order)
        }
        design = diagnostic.negative_design(sources)
        self.assertEqual(design["source_manifest_order"], order)
        self.assertEqual(
            design["negative_iids_by_correct_iid"]["dog-z"],
            ["dog-a", "dog-m", "dog-b"],
        )
        self.assertEqual(
            design["negative_iids_by_correct_iid"]["human-d"],
            ["human-b", "human-a", "human-c"],
        )
        self.assertEqual(design["registered_directed_source_pair_count"], 24)
        self.assertEqual(len(design["directed_source_pairs"]), 24)
        self.assertFalse(design["candidate_metrics_consulted_during_registration"])

    def test_pins_counts_policy_and_required_legacy_source(self) -> None:
        self.assertEqual(
            diagnostic.FROZEN_CYCLIC_SOURCE_SHA256,
            "e732c37353eb53f6ada5ab493ce00bc53a2f984eadca1d69d5d34702cbbe1521",
        )
        self.assertEqual(
            diagnostic.NEGATIVE_SOURCE_POLICY,
            "same_actor_family_sealed_manifest_order_all_other_three_v1",
        )
        self.assertEqual(diagnostic.EXPECTED_SOURCE_COUNT, 8)
        self.assertEqual(diagnostic.EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT, 84)
        self.assertEqual(diagnostic.EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT, 24)
        self.assertEqual(diagnostic.EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT, 15)
        self.assertEqual(diagnostic.EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT, 5)
        self.assertEqual(
            diagnostic.EXPECTED_MISSING_CORRECT_SOURCE_IIDS,
            {"6ea45d35943742bb", "841b5e0080a1441d", "99cde432839f4240"},
        )
        self.assertEqual(
            diagnostic.EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256,
            "8f00dc5e9650fab76f93e28c7129544a732fe925782af7c83cd4d4cee06cff96",
        )
        self.assertEqual(
            diagnostic.EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256,
            "cb474ebca8c1aa0cb8c5443c40cf778e9643fe3ee67164886828dd3fec91ac29",
        )
        parser = diagnostic.build_parser()
        build = next(
            action for action in parser._actions
            if getattr(action, "choices", None)
        ).choices["build-manifest"]
        required = {action.dest for action in build._actions if action.required}
        self.assertIn("legacy_cyclic_aggregate", required)
        self.assertIn("expected_legacy_cyclic_aggregate_sha256", required)

    def test_raw_pair_preserves_every_frozen_cyclic_field_exactly(self) -> None:
        import torch

        candidate_global = torch.nn.functional.normalize(
            torch.arange(24, dtype=torch.float32).reshape(3, 8) + 1,
            dim=-1,
        )
        correct_global = candidate_global.clone()
        wrong_global = torch.flip(candidate_global, dims=(0,))
        candidate_dense = candidate_global[:, None, :].repeat(1, 4, 1)
        correct_dense = candidate_dense.clone()
        wrong_dense = torch.flip(candidate_dense, dims=(0,))
        frozen_metrics = diagnostic.frozen.raw_metrics(
            candidate_global,
            candidate_dense,
            correct_global,
            correct_dense,
            wrong_global,
            wrong_dense,
        )
        result = diagnostic.raw_pair_metrics(
            candidate_global,
            candidate_dense,
            correct_global,
            correct_dense,
            wrong_global,
            wrong_dense,
        )
        self.assertEqual({key: result[key] for key in frozen_metrics}, frozen_metrics)
        self.assertIn("global_wrong_normalized_contrast_denominator", result)
        self.assertIn("global_wrong_normalized_contrast", result)
        self.assertIn("dense_wrong_normalized_contrast_denominator", result)
        self.assertIn("dense_wrong_normalized_contrast", result)
        self.assertIsNone(result["thresholds"])

    def test_candidate_worst_and_median_are_descriptive_only(self) -> None:
        rows = [
            {
                "global_correct_minus_wrong_margin": value,
                "global_wrong_normalized_contrast": value + 0.1,
                "dense_correct_minus_wrong_margin": value + 0.2,
                "dense_wrong_normalized_contrast": value + 0.3,
            }
            for value in (0.7, -0.2, 0.4)
        ]
        result = diagnostic.descriptive_candidate_summary(rows)
        self.assertEqual(
            result["statistics"]["global_correct_minus_wrong_margin"],
            {"worst_minimum": -0.2, "median": 0.4},
        )
        self.assertTrue(result["descriptive_only"])
        self.assertIsNone(result["thresholds"])
        self.assertFalse(result["ranking_authorized"])
        self.assertFalse(result["selection_authorized"])
        self.assertFalse(result["training_target_authorized"])

    def test_real_schema_manifest_roundtrip_distinguishes_15_executed_from_24_registered(self) -> None:
        loaded = _roundtrip_input_manifest(_synthetic_input_manifest())
        design = loaded["negative_design"]
        self.assertEqual(design["registered_directed_source_pair_count"], 24)
        correct_iids = {row["correct_source"]["iid"] for row in loaded["attempts"]}
        observed_pairs = {
            (row["correct_source"]["iid"], negative["iid"])
            for row in loaded["attempts"]
            for negative in row["negative_sources"]
        }
        self.assertEqual(len(loaded["attempts"]), 28)
        self.assertEqual(sum(len(row["negative_sources"]) for row in loaded["attempts"]), 84)
        self.assertEqual(len(correct_iids), 5)
        self.assertTrue(diagnostic.EXPECTED_MISSING_CORRECT_SOURCE_IIDS.isdisjoint(correct_iids))
        self.assertEqual(len(observed_pairs), 15)

    def test_manifest_roundtrip_rejects_nested_authority_tamper(self) -> None:
        hostile = _synthetic_input_manifest()
        hostile["authority"]["selection_authorized"] = True
        hostile = _seal(hostile)
        with self.assertRaises(diagnostic.AllThreeNegativeRawError):
            _roundtrip_input_manifest(hostile)

    def test_candidate_result_rejects_nested_authority_arithmetic_sha_and_marker_tamper(self) -> None:
        expected, result = _candidate_fixture()
        diagnostic._validate_candidate_result(result, expected=expected)
        hostile_cases = {}
        hostile = copy.deepcopy(result)
        hostile["negative_results"][0]["raw_metrics"]["selection_authorized"] = True
        hostile_cases["nested_authority"] = hostile
        hostile = copy.deepcopy(result)
        hostile["negative_results"][0]["raw_metrics"]["invented_authority"] = True
        hostile_cases["unknown_nested_authority"] = hostile
        hostile = copy.deepcopy(result)
        hostile["negative_results"][0]["raw_metrics"]["global_correct_minus_wrong_margin"] += 0.01
        hostile_cases["margin_arithmetic"] = hostile
        hostile = copy.deepcopy(result)
        hostile["negative_results"][0]["raw_metrics"]["dense_wrong_normalized_contrast_denominator"] += 0.01
        hostile_cases["denominator_arithmetic"] = hostile
        hostile = copy.deepcopy(result)
        hostile["negative_results"][0]["raw_metrics"]["global_wrong_normalized_contrast"] += 0.01
        hostile_cases["contrast_arithmetic"] = hostile
        hostile = copy.deepcopy(result)
        hostile["negative_results"][0]["negative_source_video_sha256"] = "f" * 64
        hostile_cases["negative_source_sha"] = hostile
        hostile = copy.deepcopy(result)
        hostile["negative_results"][0]["correct_source_video_sha256"] = "e" * 64
        hostile_cases["correct_source_sha"] = hostile
        hostile = copy.deepcopy(result)
        hostile["negative_results"][0]["is_legacy_cyclic_negative"] = False
        hostile["negative_results"][1]["is_legacy_cyclic_negative"] = True
        hostile_cases["legacy_marker_rebound"] = hostile
        for label, hostile in hostile_cases.items():
            with self.subTest(label=label):
                with self.assertRaises(diagnostic.AllThreeNegativeRawError):
                    diagnostic._validate_candidate_result(hostile, expected=expected)

    def test_cache_rejects_scope_cpu_and_manifest_source_tamper(self) -> None:
        source_order = [f"iid-{index}" for index in range(8)]
        expected_sources = {
            iid: {
                "iid": iid,
                "actor_family": "dog" if index < 4 else "human",
                "source_video_sha256": f"{index + 1:064x}",
            }
            for index, iid in enumerate(source_order)
        }
        entries = [{
            "iid": iid,
            "actor_family": expected_sources[iid]["actor_family"],
            "source_video_sha256": expected_sources[iid]["source_video_sha256"],
            "global_feature_sha256": f"{index + 11:064x}",
            "dense_feature_sha256": f"{index + 21:064x}",
        } for index, iid in enumerate(source_order)]
        hash_map = [{
            "iid": row["iid"],
            "source_video_sha256": row["source_video_sha256"],
            "global_feature_sha256": row["global_feature_sha256"],
            "dense_feature_sha256": row["dense_feature_sha256"],
        } for row in entries]
        summary = {
            "cache_scope": "one_rank_process",
            "source_count": 8,
            "source_manifest_order": source_order,
            "all_sources_warmed_before_candidate_decode": True,
            "cache_reused_for_all_candidate_measurements": True,
            "source_features_held_in_cpu_memory_until_worker_exit": True,
            "source_files_retained_open_until_worker_exit": False,
            "entries": entries,
            "feature_hash_map_sha256": diagnostic.core.object_sha256(hash_map),
            "operational_limitation": dict(diagnostic.OPERATIONAL_LIMITATION),
        }
        diagnostic._cache_hash_map(
            summary,
            expected_order=source_order,
            expected_sources=expected_sources,
        )
        hostile_cases = {}
        hostile = copy.deepcopy(summary)
        hostile["cache_scope"] = "shared_across_ranks"
        hostile_cases["scope"] = hostile
        hostile = copy.deepcopy(summary)
        hostile["source_features_held_in_cpu_memory_until_worker_exit"] = False
        hostile_cases["cpu_held"] = hostile
        hostile = copy.deepcopy(summary)
        hostile["entries"][0]["source_video_sha256"] = "f" * 64
        tampered_map = [{
            "iid": row["iid"],
            "source_video_sha256": row["source_video_sha256"],
            "global_feature_sha256": row["global_feature_sha256"],
            "dense_feature_sha256": row["dense_feature_sha256"],
        } for row in hostile["entries"]]
        hostile["feature_hash_map_sha256"] = diagnostic.core.object_sha256(tampered_map)
        hostile_cases["source_sha_even_with_resealed_map"] = hostile
        for label, hostile in hostile_cases.items():
            with self.subTest(label=label):
                with self.assertRaises(diagnostic.AllThreeNegativeRawError):
                    diagnostic._cache_hash_map(
                        hostile,
                        expected_order=source_order,
                        expected_sources=expected_sources,
                    )

    def test_caller_legacy_aggregate_sha_must_equal_hard_pin(self) -> None:
        with self.assertRaisesRegex(
            diagnostic.AllThreeNegativeRawError,
            "caller legacy cyclic aggregate SHA-256 differs from hard pin",
        ):
            diagnostic._legacy_aggregate("/does/not/exist", "f" * 64)

    def test_exact8_cache_and_operational_fd_gap_are_explicit(self) -> None:
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn("all_sources_warmed_before_candidate_decode", source)
        self.assertIn("source feature hashes differ across ranks", source)
        self.assertIn("pair_count != EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT", source)
        self.assertIn("len(observed_pairs) != EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT", source)
        self.assertIn("all_28_legacy_cyclic_raw_metrics_exact", source)
        self.assertTrue(diagnostic.OPERATIONAL_LIMITATION["operational_diagnostic_only"])
        self.assertFalse(
            diagnostic.OPERATIONAL_LIMITATION[
                "exact8_source_files_retained_open_for_full_process_lifetime"
            ]
        )
        self.assertFalse(
            diagnostic.OPERATIONAL_LIMITATION["formal_source_retained_fd_closure_satisfied"]
        )
        self.assertFalse(
            diagnostic.OPERATIONAL_LIMITATION["formal_or_training_admission_authorized"]
        )
        for key, value in diagnostic.AUTHORITY_CLOSURE.items():
            if key.endswith("authority") or key.endswith("authorized"):
                self.assertFalse(value, key)

    def test_launcher_is_owned_exact8_and_requires_regression_input(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts"
            / "auh_diagnose_saic_partial28_source_bound_dinov2_same_actor_all_three_negatives_raw_v1.sh"
        ).read_text("utf-8")
        self.assertIn('if [[ "$#" -ne 20 ]]', launcher)
        self.assertIn('--legacy-cyclic-aggregate "$legacy_cyclic_aggregate"', launcher)
        self.assertIn(
            '--expected-legacy-cyclic-aggregate-sha256 "$legacy_cyclic_aggregate_sha256"',
            launcher,
        )
        self.assertIn("for rank in 0 1 2 3 4 5 6 7", launcher)
        self.assertIn('ROCR_VISIBLE_DEVICES="$rank"', launcher)
        self.assertIn("env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL", launcher)
        self.assertNotIn('HIP_VISIBLE_DEVICES="$rank"', launcher)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="$rank"', launcher)
        self.assertIn("aggregate is forbidden", launcher)
        self.assertIn("portable ffprobe identity differs", launcher)


if __name__ == "__main__":
    unittest.main()

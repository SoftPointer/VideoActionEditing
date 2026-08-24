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

import diagnose_saic_r8_exact60_source_bound_dinov2_same_actor_all_three_negatives_raw_v1 as diagnostic


def _seal(value: dict) -> dict:
    unsigned = copy.deepcopy(value)
    unsigned.pop("receipt_digest", None)
    return {**unsigned, "receipt_digest": diagnostic.core.object_sha256(unsigned)}


def _sources() -> tuple[list[str], dict[str, dict]]:
    order = [
        "7b88a1ca1f804f41", "841b5e0080a1441d",
        "a35b590961d24694", "31c34509415745ca",
        "99cde432839f4240", "6ea45d35943742bb",
        "311c82f83eca4a7f", "6d346c38cf504493",
    ]
    dog = {
        "6ea45d35943742bb", "7b88a1ca1f804f41",
        "841b5e0080a1441d", "99cde432839f4240",
    }
    sources = {
        iid: {
            "iid": iid,
            "actor_family": "dog" if iid in dog else "human",
            "source_video_sha256": f"{index + 1:064x}",
        }
        for index, iid in enumerate(order)
    }
    return order, sources


def _manifest() -> dict:
    order, sources = _sources()
    design = diagnostic.negative_design(sources)
    attempts = []
    for index in range(diagnostic.EXPECTED_ATTEMPT_COUNT):
        iid = order[index % len(order)]
        attempts.append({
            "candidate_id": f"candidate-{index:02d}",
            "correct_source": copy.deepcopy(sources[iid]),
            "negative_sources": [
                copy.deepcopy(sources[negative])
                for negative in design["negative_iids_by_correct_iid"][iid]
            ],
            "legacy_cyclic_negative_iid":
                design["legacy_cyclic_negative_iid_by_correct_iid"][iid],
        })
    unsigned = {
        "schema_version": diagnostic.INPUT_SCHEMA,
        "diagnostic_source_sha256": "a" * 64,
        "attempts_root": diagnostic.EXPECTED_ATTEMPTS_ROOT,
        "root_spec_raw_sha256": diagnostic.EXPECTED_ROOT_SPEC_SHA256,
        "attempt_count": diagnostic.EXPECTED_ATTEMPT_COUNT,
        "world_size": diagnostic.EXPECTED_WORLD_SIZE,
        "partition_rule": "candidate_order_index_modulo_world_size",
        "selected_frame_indices": list(diagnostic.core.EVAL_FRAME_INDICES),
        "source_manifest": {
            "path": diagnostic.EXPECTED_SOURCE_MANIFEST_PATH,
            "raw_sha256": diagnostic.EXPECTED_SOURCE_MANIFEST_SHA256,
            "content_sha256": diagnostic.EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256,
            "validator_summary_sha256":
                diagnostic.EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256,
            "bound_files_verified": True,
            "negative_source_policy": diagnostic.NEGATIVE_SOURCE_POLICY,
            "source_manifest_order": order,
            "negative_registration_sha256": diagnostic.core.object_sha256(design),
        },
        "negative_design": design,
        "legacy_cyclic_regression": {"synthetic": True},
        "attempts": attempts,
        "operational_limitation": dict(diagnostic.OPERATIONAL_LIMITATION),
        "authority": dict(diagnostic.AUTHORITY_CLOSURE),
    }
    return _seal(unsigned)


def _raw_metrics() -> dict:
    values = {
        "measurement_label": "frozen_dinov2_source_bound_raw_proxy_only",
        "global_candidate_correct": 0.8,
        "global_candidate_wrong": 0.6,
        "global_correct_minus_wrong_margin": 0.2,
        "global_source_self_upper_bound": 1.0,
        "dense_candidate_correct": 0.7,
        "dense_candidate_wrong": 0.5,
        "dense_correct_minus_wrong_margin": 0.2,
        "dense_source_self_upper_bound": 1.0,
        "thresholds": None,
        **diagnostic.AUTHORITY_CLOSURE,
        "global_wrong_normalized_contrast_denominator": 0.4,
        "global_wrong_normalized_contrast": 0.5,
        "dense_wrong_normalized_contrast_denominator": 0.5,
        "dense_wrong_normalized_contrast": 0.4,
        "normalized_contrast_zero_when_denominator_nonpositive": True,
        "descriptive_only": True,
        "operational_diagnostic_only": True,
        "multi_negative_proxy_authority": False,
        "formal_retained_source_fd_authority": False,
    }
    return values


def _decode(sha256: str) -> dict:
    return {
        "artifact_sha256": sha256,
        "decoded_rgb_sha256": "1" * 64,
        "frame_count": 81, "fps_numerator": 25, "fps_denominator": 1,
        "time_base_numerator": 1, "time_base_denominator": 12800,
        "pts_step": 512, "pts_sha256": "2" * 64,
        "width": 736, "height": 704,
        "selected_frame_indices": list(range(0, 81, 5)),
        "selected_rgb_sha256": "3" * 64,
        "preprocessed_tensor_sha256": "0" * 64,
    }


def _features() -> dict:
    return {
        "global_feature_sha256": "4" * 64,
        "dense_feature_sha256": "5" * 64,
        "selected_frame_count": 17,
        "dense_grid_height": 16,
        "dense_grid_width": 16,
        "feature_dimension": 768,
    }


def _candidate() -> tuple[dict, dict]:
    correct = {"iid": "dog-0", "source_video_sha256": "6" * 64}
    negatives = [
        {"iid": f"dog-{index}", "source_video_sha256": f"{index + 7:064x}"}
        for index in (1, 2, 3)
    ]
    expected = {
        "candidate_id": "candidate-00",
        "mp4_sha256": "f" * 64,
        "correct_source": correct,
        "negative_sources": negatives,
        "legacy_cyclic_negative_iid": negatives[0]["iid"],
    }
    rows = [{
        "negative_ordinal_in_manifest_order": ordinal,
        "correct_source_iid": correct["iid"],
        "correct_source_video_sha256": correct["source_video_sha256"],
        "negative_source_iid": negative["iid"],
        "negative_source_video_sha256": negative["source_video_sha256"],
        "is_legacy_cyclic_negative": ordinal == 0,
        "raw_metrics": _raw_metrics(),
        "authority": dict(diagnostic.AUTHORITY_CLOSURE),
    } for ordinal, negative in enumerate(negatives)]
    result = {
        "candidate_id": expected["candidate_id"],
        "candidate_binding": copy.deepcopy(expected),
        "candidate_decode": _decode(expected["mp4_sha256"]),
        "candidate_features": _features(),
        "source_features_served_from_exact8_rank_cache": True,
        "negative_results": rows,
        "candidate_descriptive_summary": diagnostic.algorithm.descriptive_candidate_summary(
            [row["raw_metrics"] for row in rows]
        ),
        "operational_limitation": dict(diagnostic.OPERATIONAL_LIMITATION),
        "authority": dict(diagnostic.AUTHORITY_CLOSURE),
    }
    return expected, result


class R8Exact60AllThreeContractTest(unittest.TestCase):
    def test_partition_and_180_24_24_0_geometry(self) -> None:
        partitions = [diagnostic.partition_indices(60, rank, 8) for rank in range(8)]
        self.assertEqual([len(row) for row in partitions], [8, 8, 8, 8, 7, 7, 7, 7])
        flattened = [index for row in partitions for index in row]
        self.assertEqual(sorted(flattened), list(range(60)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(diagnostic.EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT, 180)
        self.assertEqual(diagnostic.EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT, 24)
        self.assertEqual(diagnostic.EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT, 24)
        self.assertEqual(diagnostic.EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT, 8)
        self.assertEqual(diagnostic.EXPECTED_MISSING_CORRECT_SOURCE_IIDS, frozenset())

    def test_manifest_shape_recomputes_all8_all24_not_assumed(self) -> None:
        value = _manifest()
        design = value["negative_design"]
        correct_iids = {row["correct_source"]["iid"] for row in value["attempts"]}
        pairs = {
            (row["correct_source"]["iid"], negative["iid"])
            for row in value["attempts"] for negative in row["negative_sources"]
        }
        self.assertEqual(len(value["attempts"]), 60)
        self.assertEqual(sum(len(row["negative_sources"]) for row in value["attempts"]), 180)
        self.assertEqual(len(correct_iids), 8)
        self.assertEqual(len(pairs), 24)
        self.assertEqual(design["registered_directed_source_pair_count"], 24)

    def test_dependency_artifact_and_visual_pins_are_exact(self) -> None:
        self.assertEqual(diagnostic.CYCLIC_SOURCE_SHA256, "2839641766b4605311f0c4e7a3ff41ea9a322ad44cf1d12d21fb7f0cca5ab24e")
        self.assertEqual(diagnostic.EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256, "28ff1e40f4dd314548616050013afdfb5e2a2a768aba9f0cbd4f00c9f6718c62")
        self.assertEqual(diagnostic.EXPECTED_LEGACY_CYCLIC_INPUT_RECEIPT_DIGEST, "a4ad17a09e46d549089356ecf86e21d5d8a6da2f41aaa92d218b058a9e28f378")
        self.assertEqual(diagnostic.EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256, "2e10fd8539d42aecb8872bba3e504e26d7e2dfb9a5120b1145080f8b463dc7fb")
        self.assertEqual(diagnostic.EXPECTED_LEGACY_CYCLIC_AGGREGATE_RECEIPT_DIGEST, "68c1670836f01ff2d147237ca70ee03914b4c8735438a5ffa9295621de4161e1")
        self.assertEqual(diagnostic.EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256, "6a9232bdb17703747c76cd6eb9a5e7c92aa4fbcb4a0e85e77bd3cd960230dbaa")
        self.assertEqual(diagnostic.EXPECTED_SOURCE_MANIFEST_PATH, "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/t2v-events-topup-r8-ddc8a79-r1/sealed-saic-source-manifest.json")
        self.assertEqual(diagnostic.EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256, "257d3aafaaee126ff2c1a061413d01bd0457676eb5d1ee027671221a5a794218")

    def test_real_import_and_help_twice_restore_nested_globals_and_callables(self) -> None:
        polluted = lambda *_args, **_kwargs: None
        for module in (diagnostic.algorithm, diagnostic.frozen, diagnostic.core):
            module.__file__ = "/tmp/hostile.py"
            module.SCHEMA_VERSION = "hostile"
            module.INPUT_SCHEMA = "hostile"
            module.SHARD_SCHEMA = "hostile"
            module.AGGREGATE_SCHEMA = "hostile"
            module.PREFLIGHT_SCHEMA = "hostile"
            module.EXPECTED_ATTEMPT_COUNT = 1
            module.EXPECTED_WORLD_SIZE = 1
            module.AUTHORITY_CLOSURE = {"training_target_authorized": True}
            module.partition_indices = polluted
        for name in (
            "build_manifest", "load_input_manifest", "_worker_common",
            "aggregate", "preflight", "worker",
        ):
            setattr(diagnostic.algorithm, name, polluted)
            setattr(diagnostic.frozen, name, polluted)
        diagnostic.algorithm._source_closure = polluted
        diagnostic.algorithm._cache_hash_map = polluted
        diagnostic.algorithm._validate_candidate_result = polluted
        for _ in range(2):
            with self.assertRaises(SystemExit) as raised:
                diagnostic.main(["--help"])
            self.assertEqual(raised.exception.code, 0)
        for module in (diagnostic.algorithm, diagnostic.frozen, diagnostic.core):
            self.assertEqual(module.__file__, diagnostic.__file__)
            self.assertEqual(module.SCHEMA_VERSION, diagnostic.SCHEMA_VERSION)
            self.assertEqual(module.EXPECTED_ATTEMPT_COUNT, 60)
            self.assertEqual(module.EXPECTED_WORLD_SIZE, 8)
            self.assertEqual(module.AUTHORITY_CLOSURE, diagnostic.AUTHORITY_CLOSURE)
            self.assertIs(module.partition_indices, diagnostic.partition_indices)
        self.assertIs(diagnostic.algorithm.build_manifest, diagnostic.build_manifest)
        self.assertIs(diagnostic.algorithm.load_input_manifest, diagnostic.load_input_manifest)
        self.assertIs(diagnostic.algorithm._worker_common, diagnostic._worker_common)
        self.assertIs(diagnostic.algorithm.aggregate, diagnostic.aggregate)
        self.assertIs(diagnostic.algorithm._source_closure, diagnostic._source_closure)
        self.assertIs(diagnostic.algorithm._cache_hash_map, diagnostic._cache_hash_map)
        self.assertIs(diagnostic.algorithm._validate_candidate_result, diagnostic._validate_candidate_result)

    def test_candidate_deep_validation_rejects_binding_metrics_and_fields(self) -> None:
        expected, valid = _candidate()
        diagnostic._validate_candidate_result(valid, expected=expected)
        hostile = []
        value = copy.deepcopy(valid); value["extra"] = False; hostile.append(value)
        value = copy.deepcopy(valid); value["candidate_binding"]["candidate_id"] = "other"; hostile.append(value)
        value = copy.deepcopy(valid); value["candidate_decode"]["frame_count"] = 80; hostile.append(value)
        value = copy.deepcopy(valid); value["candidate_features"]["feature_dimension"] = 1; hostile.append(value)
        value = copy.deepcopy(valid); value["negative_results"][0]["extra"] = False; hostile.append(value)
        value = copy.deepcopy(valid); value["negative_results"][0]["authority"]["extra"] = False; hostile.append(value)
        value = copy.deepcopy(valid); value["negative_results"][0]["raw_metrics"]["global_correct_minus_wrong_margin"] = 0.3; hostile.append(value)
        value = copy.deepcopy(valid); value["negative_results"][0]["raw_metrics"]["dense_candidate_correct"] = float("nan"); hostile.append(value)
        for mutated in hostile:
            with self.assertRaises(diagnostic.AllThreeNegativeRawError):
                diagnostic._validate_candidate_result(mutated, expected=expected)

    def test_visual_object_and_candidate_projection_are_present(self) -> None:
        checked = diagnostic.cyclic._validate_visual_evaluator(
            copy.deepcopy(diagnostic.EXPECTED_VISUAL_EVALUATOR), rank=0,
        )
        self.assertEqual(diagnostic.core.object_sha256(checked), diagnostic.EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256)
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn('"visual_evaluator_evidence_projection"', source)
        self.assertIn('"all8_visual_evaluator_projections_identical": True', source)
        self.assertIn('"legacy_cyclic_projection_sha256"', source)
        self.assertIn("all8_shards_and_all60_candidate_results_deep_validated", source)

    def test_caller_path_and_hash_are_both_hard_pinned(self) -> None:
        for path, sha in (
            ("/wrong/path", diagnostic.EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256),
            (diagnostic.EXPECTED_LEGACY_CYCLIC_AGGREGATE_PATH, "f" * 64),
        ):
            with self.assertRaisesRegex(
                diagnostic.AllThreeNegativeRawError,
                "caller legacy cyclic aggregate path/SHA-256 differs",
            ):
                diagnostic._legacy_aggregate(path, sha)

    def test_launcher_pins_all_inputs_and_rank_local_cache_cleanup(self) -> None:
        launcher = (METHOD_ROOT / "scripts" / "auh_diagnose_saic_r8_exact60_source_bound_dinov2_same_actor_all_three_negatives_raw_v1.sh").read_text("utf-8")
        self.assertIn('if [[ "$#" -ne 21 ]]', launcher)
        self.assertIn("readonly expected_diagnostic_sha=", launcher)
        self.assertIn(diagnostic.CYCLIC_SOURCE_SHA256, launcher)
        self.assertIn(diagnostic.CYCLIC_SOURCE_SHA256, Path(diagnostic.__file__).read_text("utf-8"))
        self.assertIn(diagnostic.EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256, launcher)
        self.assertIn("b61f251411f0d8f6a617b67d0b903c333d16c77fb6b3f49507225884d4aed0ea", launcher)
        self.assertIn("6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736", launcher)
        self.assertIn("9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39", launcher)
        self.assertIn("183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a", launcher)
        for variable in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR", "TMPDIR", "TORCH_EXTENSIONS_DIR"):
            self.assertIn(f'{variable}="$cache/', launcher)
        self.assertIn("validate_cache_roots()", launcher)
        self.assertIn("remove_runtime_scratch_exact()", launcher)
        self.assertIn('find "$runtime_scratch" -xdev -depth -mindepth 1 -delete', launcher)
        self.assertIn("trap cleanup_on_exit EXIT", launcher)

    def test_actual_source_sha_is_cascaded(self) -> None:
        actual = hashlib.sha256(Path(diagnostic.__file__).read_bytes()).hexdigest()
        launcher = (METHOD_ROOT / "scripts" / "auh_diagnose_saic_r8_exact60_source_bound_dinov2_same_actor_all_three_negatives_raw_v1.sh").read_text("utf-8")
        self.assertIn(f"readonly expected_diagnostic_sha={actual}", launcher)
        self.assertNotIn("SOURCE_SHA256_PLACEHOLDER", launcher)


if __name__ == "__main__":
    unittest.main()

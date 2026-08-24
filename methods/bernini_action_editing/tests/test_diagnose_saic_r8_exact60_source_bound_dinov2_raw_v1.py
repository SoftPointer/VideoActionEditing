from __future__ import annotations

import copy
from pathlib import Path
import hashlib
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_r8_exact60_source_bound_dinov2_raw_v1 as diagnostic


def _decode(artifact_sha256: str) -> dict:
    return {
        "artifact_sha256": artifact_sha256,
        "decoded_rgb_sha256": "1" * 64,
        "frame_count": 81,
        "fps_numerator": 25,
        "fps_denominator": 1,
        "time_base_numerator": 1,
        "time_base_denominator": 12800,
        "pts_step": 512,
        "pts_sha256": "2" * 64,
        "width": 736,
        "height": 704,
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


def _valid_result() -> tuple[dict, dict]:
    correct = {"iid": "correct", "source_video_sha256": "6" * 64}
    wrong = {"iid": "wrong", "source_video_sha256": "7" * 64}
    expected = {
        "candidate_id": "candidate-0",
        "mp4_sha256": "8" * 64,
        "correct_source": correct,
        "wrong_source": wrong,
    }
    metrics = {
        "measurement_label": "frozen_dinov2_source_bound_raw_proxy_only",
        "global_candidate_correct": 0.75,
        "global_candidate_wrong": 0.25,
        "global_correct_minus_wrong_margin": 0.5,
        "global_source_self_upper_bound": 1.0,
        "dense_candidate_correct": 0.6,
        "dense_candidate_wrong": 0.4,
        "dense_correct_minus_wrong_margin": 0.2,
        "dense_source_self_upper_bound": 1.0,
        "thresholds": None,
        **diagnostic.RAW_METRIC_AUTHORITY,
    }
    result = {
        "candidate_id": expected["candidate_id"],
        "candidate_binding": copy.deepcopy(expected),
        "candidate_decode": _decode(expected["mp4_sha256"]),
        "candidate_features": _features(),
        "correct_source_evidence": {
            "decode": _decode(correct["source_video_sha256"]),
            "features": _features(),
        },
        "wrong_source_evidence": {
            "decode": _decode(wrong["source_video_sha256"]),
            "features": _features(),
        },
        "raw_metrics": metrics,
        "authority": dict(diagnostic.AUTHORITY_CLOSURE),
    }
    return result, expected


def _visual_evaluator() -> dict:
    return copy.deepcopy(diagnostic.EXPECTED_VISUAL_EVALUATOR)


class R8Exact60SourceBoundRawContractTest(unittest.TestCase):
    def test_partition_is_exactly_once_and_balanced(self) -> None:
        partitions = [diagnostic.partition_indices(60, rank, 8) for rank in range(8)]
        self.assertEqual([len(row) for row in partitions], [8, 8, 8, 8, 7, 7, 7, 7])
        flattened = [index for row in partitions for index in row]
        self.assertEqual(sorted(flattened), list(range(60)))
        self.assertEqual(len(flattened), len(set(flattened)))
        with self.assertRaises(diagnostic.SourceBoundRaw60Error):
            diagnostic.partition_indices(28, 0, 8)

    def test_nested_self_verification_targets_exact60_source(self) -> None:
        source = Path(diagnostic.__file__)
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        diagnostic._install_specialization()
        self.assertEqual(Path(diagnostic.core.__file__), source)
        self.assertEqual(Path(diagnostic.core.core.__file__), source)
        self.assertEqual(diagnostic.core.core._verify_self(source_sha256), source_sha256)
        self.assertEqual(diagnostic.core.core.EXPECTED_ATTEMPT_COUNT, 60)
        self.assertEqual(diagnostic.core.core.EXPECTED_WORLD_SIZE, 8)
        self.assertIs(diagnostic.core.partition_indices, diagnostic.partition_indices)
        self.assertIs(diagnostic.core._base_partition_indices, diagnostic.partition_indices)
        self.assertIs(diagnostic.core.core.partition_indices, diagnostic.partition_indices)

    def test_main_parser_can_be_installed_twice_without_recursion(self) -> None:
        for _ in range(2):
            with self.assertRaises(SystemExit) as caught:
                diagnostic.main(["--help"])
            self.assertEqual(caught.exception.code, 0)

    def test_exact47_algorithm_and_r8_bank_are_hard_pinned(self) -> None:
        self.assertEqual(
            diagnostic._BASE_SHA256,
            "ffbc9ba149d1ddadf704dd8258678a8893235e328da4c7601e98d63ba37aa7a2",
        )
        self.assertEqual(diagnostic.EXPECTED_JOB_ID, "135056")
        self.assertEqual(
            diagnostic.EXPECTED_SOURCE_REVISION,
            "ddc8a79199aed1391cf089f51835c2bbfa74ae28",
        )
        self.assertEqual(
            diagnostic.EXPECTED_SOURCE_ARCHIVE_SHA256,
            "4038100b86655e5ea3e9a32432dc619c4b8d1a5d7859703c4cf06b77de0b934b",
        )
        self.assertIn("t2v-events-topup-r8-ddc8a79-r1/attempts", diagnostic.EXPECTED_ATTEMPTS_ROOT)
        self.assertEqual(
            diagnostic.EXPECTED_ROOT_SPEC_SHA256,
            "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145",
        )
        self.assertEqual(
            diagnostic.EXPECTED_SOURCE_MANIFEST_SHA256,
            "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9",
        )

    def test_terminal_authority_is_narrow_and_replayed_on_load(self) -> None:
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertEqual(
            diagnostic.TERMINAL_EVIDENCE_STATUS,
            "terminal_technical_full60_complete_pending_detached_semantic_review",
        )
        self.assertEqual(
            diagnostic.EXPECTED_TERMINAL_AUTHORITY,
            {
                "detached_decoded_event_review_input": True,
                "data_selection": False,
                "human_review": False,
                "optimizer": False,
                "scientific_action_editing_success_claim": False,
                "training": False,
                "training_target_admission": False,
            },
        )
        missing = dict(diagnostic.EXPECTED_TERMINAL_AUTHORITY)
        missing.pop("training")
        extra = {**diagnostic.EXPECTED_TERMINAL_AUTHORITY, "extra_false": False}
        self.assertNotEqual(missing, diagnostic.EXPECTED_TERMINAL_AUTHORITY)
        self.assertNotEqual(extra, diagnostic.EXPECTED_TERMINAL_AUTHORITY)
        self.assertIn("authority != EXPECTED_TERMINAL_AUTHORITY", source)
        self.assertIn("_validate_terminal_evidence(EXPECTED_TERMINAL_EVIDENCE_PATH)", source)
        self.assertIn('parsed.get("State") != "COMPLETED"', source)
        self.assertIn('parsed.get("ExitCode") != "0:0"', source)
        self.assertIn("r8 master/deep artifact binding differs", source)

    def test_deep_audit_authority_is_exact_five_key_closure(self) -> None:
        self.assertEqual(
            diagnostic.EXPECTED_DEEP_AUDIT_AUTHORITY,
            {
                "detached_decoded_event_review_input": False,
                "merge_or_partial_reuse": False,
                "scientific_selection": False,
                "training": False,
                "optimizer": False,
            },
        )
        missing = dict(diagnostic.EXPECTED_DEEP_AUDIT_AUTHORITY)
        missing.pop("training")
        extra = {**diagnostic.EXPECTED_DEEP_AUDIT_AUTHORITY, "extra_false": False}
        self.assertNotEqual(missing, diagnostic.EXPECTED_DEEP_AUDIT_AUTHORITY)
        self.assertNotEqual(extra, diagnostic.EXPECTED_DEEP_AUDIT_AUTHORITY)
        self.assertIn(
            'audit.get("authority") != EXPECTED_DEEP_AUDIT_AUTHORITY',
            Path(diagnostic.__file__).read_text("utf-8"),
        )

    def test_candidate_result_deep_closure_rejects_hostile_mutations(self) -> None:
        valid, expected = _valid_result()
        diagnostic._validate_candidate_result(valid, expected=expected)
        mutations = []
        top_extra = copy.deepcopy(valid)
        top_extra["extra_false"] = False
        mutations.append(top_extra)
        binding = copy.deepcopy(valid)
        binding["candidate_binding"]["candidate_id"] = "substituted"
        mutations.append(binding)
        authority = copy.deepcopy(valid)
        authority["authority"]["extra_false"] = False
        mutations.append(authority)
        metric = copy.deepcopy(valid)
        metric["raw_metrics"]["global_correct_minus_wrong_margin"] = 0.4
        mutations.append(metric)
        nonfinite = copy.deepcopy(valid)
        nonfinite["raw_metrics"]["dense_candidate_correct"] = float("nan")
        mutations.append(nonfinite)
        evidence = copy.deepcopy(valid)
        evidence["wrong_source_evidence"]["decode"]["artifact_sha256"] = "9" * 64
        mutations.append(evidence)
        for mutated in mutations:
            with self.assertRaises(diagnostic.SourceBoundRaw60Error):
                diagnostic._validate_candidate_result(mutated, expected=expected)

    def test_visual_evaluator_is_exact_and_cross_rank_identical(self) -> None:
        valid = _visual_evaluator()
        checked = diagnostic._validate_visual_evaluator(valid, rank=0)
        self.assertEqual(
            diagnostic.core.core.object_sha256(checked),
            diagnostic.EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256,
        )
        hostile = []
        missing = _visual_evaluator()
        missing.pop("frozen_eval")
        hostile.append(missing)
        extra = _visual_evaluator()
        extra["extra_false"] = False
        hostile.append(extra)
        manifest = _visual_evaluator()
        manifest["checkpoint_manifest_sha256"] = "0" * 64
        hostile.append(manifest)
        authority = _visual_evaluator()
        authority["scientific_claim_authorized"] = True
        hostile.append(authority)
        loading = _visual_evaluator()
        loading["loading_counts"]["missing_key_count"] = 1
        hostile.append(loading)
        for mutated in hostile:
            with self.assertRaises(diagnostic.SourceBoundRaw60Error):
                diagnostic._validate_visual_evaluator(mutated, rank=7)

        divergent = _visual_evaluator()
        divergent["adapter_id"] = "hostile-cross-rank-substitution"
        with self.assertRaises(diagnostic.SourceBoundRaw60Error):
            diagnostic._require_identical_visual_evaluator(
                valid, divergent, rank=7,
            )
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn('"visual_evaluator_evidence_projection"', source)
        self.assertIn('"visual_evaluator_evidence_projection_sha256"', source)
        self.assertIn('"per_rank_visual_evaluator_projection_receipts"', source)
        self.assertIn('"all8_visual_evaluator_projections_identical": True', source)

    def test_registered_visual_inputs_wrong_policy_and_authority_are_closed(self) -> None:
        self.assertEqual(diagnostic.core.EXPECTED_EVALUATOR_SPEC_SHA256, "6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736")
        self.assertEqual(diagnostic.core.EXPECTED_VISUAL_SCORER_SHA256, "9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39")
        self.assertEqual(diagnostic.core.EXPECTED_VISUAL_CONTRACT_SHA256, "183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a")
        self.assertEqual(diagnostic.core.WRONG_SOURCE_POLICY, "same_actor_family_iid_lexical_cyclic_next_v1")
        self.assertEqual(set(diagnostic.EXPECTED_WRONG_IID_BY_IID), set(diagnostic.EXPECTED_SOURCE_SHA256_BY_IID))
        for field in (
            "absolute_preservation_authority", "source_bound_proxy_authority",
            "identity_authority", "event_authority", "scientific_claim_authorized",
            "ranking_authorized", "selection_authorized", "training_target_authorized",
        ):
            self.assertFalse(diagnostic.AUTHORITY_CLOSURE[field])

    def test_schema_and_aggregate_are_exact60_raw_only(self) -> None:
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertIn("r8-exact60-source-bound-dinov2-raw", diagnostic.SCHEMA_VERSION)
        self.assertIn('"coverage": "exactly_once_complete_r8_exact60_source_bound_raw"', source)
        self.assertNotIn("EXPECTED_RECEIPT_BINDING_SHA256", source)
        self.assertNotIn("SourceBoundRaw28Error", source)

    def test_launcher_has_19_args_fixed_runtime_terminal_and_only_rocr_mapping(self) -> None:
        launcher = (METHOD_ROOT / "scripts" / "auh_diagnose_saic_r8_exact60_source_bound_dinov2_raw_v1.sh").read_text("utf-8")
        self.assertIn('if [[ "$#" -ne 19 ]]', launcher)
        self.assertIn("runtime/venv-transformers-4.53.2/bin/python", launcher)
        self.assertIn(
            "readonly expected_diagnostic_sha=2839641766b4605311f0c4e7a3ff41ea9a322ad44cf1d12d21fb7f0cca5ab24e",
            launcher,
        )
        self.assertEqual(
            hashlib.sha256(Path(diagnostic.__file__).read_bytes()).hexdigest(),
            "2839641766b4605311f0c4e7a3ff41ea9a322ad44cf1d12d21fb7f0cca5ab24e",
        )
        self.assertIn('[[ "$diagnostic_sha256" != "$expected_diagnostic_sha" ]]', launcher)
        self.assertIn("ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime", launcher)
        self.assertIn("356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5", launcher)
        self.assertIn("saic-exact60-terminal-evidence-135056.json", launcher)
        self.assertIn("--terminal-evidence", launcher)
        self.assertIn("for rank in 0 1 2 3 4 5 6 7", launcher)
        self.assertIn('run_isolated "$rank_cache" "$rank"', launcher)
        self.assertIn('ROCR_VISIBLE_DEVICES="$rocr_device"', launcher)
        self.assertIn("env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL", launcher)
        self.assertNotIn('HIP_VISIBLE_DEVICES="$rank"', launcher)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="$rank"', launcher)
        self.assertIn("aggregate is forbidden", launcher)
        self.assertIn("b61f251411f0d8f6a617b67d0b903c333d16c77fb6b3f49507225884d4aed0ea", launcher)

    def test_launcher_uses_fresh_exactly_cleaned_rank_local_caches(self) -> None:
        launcher = (METHOD_ROOT / "scripts" / "auh_diagnose_saic_r8_exact60_source_bound_dinov2_raw_v1.sh").read_text("utf-8")
        self.assertIn('scratch_parent="${SLURM_TMPDIR:-/tmp}"', launcher)
        self.assertIn('runtime_scratch="$(mktemp -d -- "${runtime_prefix}XXXXXXXX")"', launcher)
        self.assertIn('rank_cache="$runtime_scratch/rank-$rank_label"', launcher)
        self.assertIn('"$(stat -c \'%u:%a\' -- "$path")" != "$current_uid:700"', launcher)
        for variable in (
            "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "MIOPEN_USER_DB_PATH",
            "MIOPEN_CUSTOM_CACHE_DIR", "TMPDIR", "TORCH_EXTENSIONS_DIR",
        ):
            self.assertIn(f'{variable}="$cache/', launcher)
        self.assertIn("validate_cache_roots()", launcher)
        self.assertIn("remove_runtime_scratch_exact()", launcher)
        self.assertIn('find "$runtime_scratch" -xdev -depth -mindepth 1 -delete', launcher)
        self.assertIn("trap cleanup_on_exit EXIT", launcher)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import author_pair_v5_source_bound_preservation_spec_v1 as author  # noqa: E402
import pair_v5_source_bound_preservation_evaluator_v1 as evaluator  # noqa: E402
import score_pair_v5_source_bound_preservation_v1 as scorer  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


ROLLOUT_SPEC = METHOD_ROOT / "assets/pair_v5_native_rv2v4_core4_action_population_v1.json"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _payload() -> dict:
    return json.loads(ROLLOUT_SPEC.read_text(encoding="utf-8"))


def _runtime() -> dict[str, str]:
    return {
        "python_version": "3.12.9", "torch_version": "2.7.1+rocm6.3",
        "torch_hip_version": "6.3.42131", "transformers_version": "4.53.2",
        "safetensors_version": "0.5.3", "av_version": "14.4.0",
        "numpy_version": "2.2.6", "pillow_version": "11.2.1",
    }


def _native_runtime() -> dict[str, str]:
    return {"torch": "2.7.1", "torch_hip": "6.3", "transformers": "4.53.2", "diffusers": "0.34.0"}


def _generation() -> dict:
    unsigned = {
        "reference_native_receipt_file_sha256": _digest("reference-native"),
        "native_schema_version": evaluator.EXPECTED_NATIVE_SCHEMA,
        "native_method": evaluator.EXPECTED_NATIVE_METHOD,
        "method_source_revision": "b" * 40,
        "method_source_archive_sha256": _digest("generation-source"),
        "bernini_commit": "c" * 40, "veomni_commit": "d" * 40,
        "bernini_inference_files": {"bernini/pipeline.py": _digest("pipeline")},
        "checkpoint_tree_sha256": _digest("bernini-tree"),
        "checkpoint_manifest_sha256": _digest("bernini-manifest"),
        "checkpoint_file_count": 23,
        "checkpoint_entries_digest": _digest("bernini-entries"),
        "runtime_versions": _native_runtime(),
    }
    return evaluator.validate_generation_provenance(
        {**unsigned, "provenance_digest": evaluator.object_sha256(unsigned)}
    )


def _spec(generation: dict | None = None) -> dict:
    return evaluator.make_evaluator_spec(
        _payload(),
        rollout_spec_raw_sha256=hashlib.sha256(ROLLOUT_SPEC.read_bytes()).hexdigest(),
        implementation_sha256=_digest("implementation"), contract_sha256=_digest("contract"),
        method_source_revision="a" * 40, method_source_archive_sha256=_digest("source-archive"),
        architecture_id="dinov2", checkpoint_manifest_sha256=_digest("checkpoint-manifest"),
        checkpoint_config_sha256=_digest("checkpoint-config"),
        preprocessor_config_sha256=_digest("preprocessor-config"), checkpoint_file_count=3,
        num_register_tokens=0, image_size=518, patch_size=14,
        preprocessor_golden_input_sha256=_digest("golden-input"),
        preprocessor_golden_output_sha256=_digest("golden-output"),
        preprocessor_golden_output_shape=[1, 3, 224, 224],
        generation_provenance=generation or _generation(), runtime_versions=_runtime(),
    )


def _decode(sha: str) -> dict:
    return {
        "artifact_sha256": sha, "decoded_rgb_sha256": _digest("decoded" + sha),
        "frame_count": 81, "fps_numerator": 25, "fps_denominator": 1,
        "time_base_numerator": 1, "time_base_denominator": 25, "pts_step": 1,
        "pts_sha256": _digest("pts" + sha), "width": 832, "height": 480,
        "selected_frame_indices": list(evaluator.EVAL_FRAME_INDICES),
        "selected_rgb_sha256": _digest("selected" + sha),
        "preprocessed_tensor_sha256": _digest("pixels" + sha),
    }


def _feature(role: str) -> dict:
    return {
        "global_feature_sha256": _digest("global" + role),
        "dense_feature_sha256": _digest("dense" + role),
        "selected_frame_count": len(evaluator.EVAL_FRAME_INDICES),
        "dense_grid_height": 16, "dense_grid_width": 16, "feature_dimension": 384,
    }


def _metrics(correct: float = .8, wrong: float = .4) -> dict[str, float]:
    upper, dense_correct, dense_wrong = 1.0, .75, .35
    return {
        "source_identity_appearance_proxy": correct,
        "source_identity_appearance_wrong_source_proxy": wrong,
        "source_identity_appearance_correct_minus_wrong_margin": correct - wrong,
        "source_identity_appearance_source_self_upper_bound": upper,
        "source_identity_appearance_upper_bound_minus_correct_headroom": upper - correct,
        "source_identity_appearance_wrong_normalized_contrast": (correct - wrong) / (upper - wrong),
        "background_appearance_fixed_grid_proxy": dense_correct,
        "background_appearance_wrong_source_fixed_grid_proxy": dense_wrong,
        "background_appearance_correct_minus_wrong_margin": dense_correct - dense_wrong,
        "non_target_temporal_consistency_proxy": .7,
        "non_target_temporal_consistency_wrong_source_proxy": .5,
        "source_bound_spatial_layout_viewpoint_proxy": dense_correct,
        "source_bound_spatial_layout_wrong_source_proxy": dense_wrong,
        "source_bound_spatial_layout_correct_minus_wrong_margin": dense_correct - dense_wrong,
        "source_bound_spatial_layout_wrong_normalized_contrast_proxy": (dense_correct - dense_wrong) / (1.0 - dense_wrong),
        "temporal_global_translation_agreement_diagnostic": .6,
        "decode_video_quality_diagnostic": .65, "quality_sharpness_retention": .7,
        "quality_exposure_score": .9, "quality_nonfreeze_score": .8, "quality_flicker_score": .75,
    }


def _model_evidence(spec: dict) -> dict:
    model = spec["model"]
    return {
        "adapter_id": model["adapter_id"], "architecture_id": model["architecture_id"],
        "checkpoint_manifest_sha256": model["checkpoint_manifest_sha256"],
        "checkpoint_config_sha256": model["checkpoint_config_sha256"],
        "preprocessor_config_sha256": model["preprocessor_config_sha256"],
        "checkpoint_file_count": model["checkpoint_file_count"],
        "verified_entries_digest": _digest("verified"),
        "preprocessor_golden_input_sha256": model["preprocessor_golden_input_sha256"],
        "preprocessor_golden_output_sha256": model["preprocessor_golden_output_sha256"],
        "preprocessor_golden_output_shape": model["preprocessor_golden_output_shape"],
        "every_checkpoint_file_verified": True, "all_parameters_frozen": True,
        "trainable_parameter_tensors": 0, "parameter_tensor_count": 12,
        "parameter_element_count": 3840, "parameter_metadata_digest": _digest("parameters"),
        "missing_key_count": 0, "unexpected_key_count": 0, "mismatched_key_count": 0,
        "loading_error_count": 0, "runtime_versions": spec["runtime_versions"],
    }


def _receipt(spec: dict, ordinal: int = 0, correct: float = .8, wrong: float = .4) -> dict:
    candidate = evaluator._validate_current_family_rollout_spec(_payload())["candidates"][ordinal]
    correct_sha = candidate["source_video_sha256"]
    wrong_sha = spec["wrong_source_by_source_sha256"][correct_sha]
    mp4_sha = _digest("mp4" + candidate["candidate_id"])
    roles = {"candidate": mp4_sha, "correct_source": correct_sha, "wrong_source": wrong_sha}
    return evaluator.make_candidate_receipt(
        evaluator_spec=spec, evaluator_spec_raw_sha256=_digest("spec-raw"),
        candidate_id=candidate["candidate_id"], candidate_ordinal=ordinal,
        group_id=candidate["group_id"], candidate_envelope_sha256=_digest("envelope"),
        rollout_receipt_digest=_digest("pair-object"), rollout_receipt_file_sha256=_digest("pair-file"),
        native_rollout_receipt_digest=_digest("native-object"),
        native_rollout_receipt_file_sha256=_digest("native-file"),
        native_generation_provenance_digest=spec["generation_provenance"]["provenance_digest"],
        candidate_mp4_sha256=mp4_sha, predecode_clean_latent_sha256=_digest("clean"),
        official_initial_gaussian_sha256=_digest("gaussian"),
        correct_source_video_sha256=correct_sha, wrong_source_video_sha256=wrong_sha,
        decode_evidence_by_role={role: _decode(sha) for role, sha in roles.items()},
        feature_evidence_by_role={role: _feature(role) for role in roles},
        model_evidence=_model_evidence(spec), metrics=_metrics(correct, wrong),
    )


def _resign(row: dict, field: str) -> None:
    unsigned = dict(row); unsigned.pop(field, None); row[field] = evaluator.object_sha256(unsigned)


def _native_sign(row: dict) -> None:
    unsigned = dict(row); unsigned.pop("receipt_digest", None)
    raw = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    row["receipt_digest"] = hashlib.sha256(raw).hexdigest()


def _artifact(path: Path, content: bytes, **extra: object) -> dict:
    path.write_bytes(content)
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), **extra}


def _native_fixture(root: Path) -> tuple[dict, dict, Path]:
    normalized = evaluator._validate_current_family_rollout_spec(_payload())
    candidate = dict(normalized["candidates"][0])
    source = root / "source.mp4"; source.write_bytes(b"source")
    candidate["source_video"] = str(source)
    candidate["source_video_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    candidate_dir = root / candidate["candidate_id"]; candidate_dir.mkdir()
    shape = [1, 16, 21, 60, 104]
    source_artifact = _artifact(
        candidate_dir / "source.safetensors", b"source-latent",
        tensor_key="normalized_clean_latent", shape=shape, stored_dtype="torch.float32",
        sampler_return_dtype="torch.float32", coordinate="bernini_normalized_clean_vae_latent",
        artifact_role="source_video_condition", source_video_vae_encode_before_any_decode=True,
        native_sampler_before_vae_decode=False, mp4_decode_reencode_used=False,
        roundtrip_byte_exact_fp32=True, origin="source_video_vae_encode_before_any_decode",
    )
    clean = _artifact(
        candidate_dir / "clean.safetensors", b"clean",
        tensor_key="normalized_clean_latent", shape=shape, stored_dtype="torch.float32",
        sampler_return_dtype="torch.bfloat16", coordinate="bernini_normalized_clean_vae_latent",
        artifact_role="native_sampler_proposal", source_video_vae_encode_before_any_decode=False,
        native_sampler_before_vae_decode=True, mp4_decode_reencode_used=False,
        roundtrip_byte_exact_fp32=True, origin="native_sampler_before_vae_decode",
    )
    gaussian = _artifact(
        candidate_dir / "noise.safetensors", b"noise",
        tensor_key="official_initial_gaussian", tensor_value_sha256=_digest("raw-noise"),
        raw_value_sha256=_digest("raw-noise"), content_sha256=_digest("content-noise"),
        shape=shape, dtype="torch.float32", stored_dtype="torch.float32",
        original_device="cpu", stored_device="cpu", numel=1 * 16 * 21 * 60 * 104,
        byte_count=4 * 1 * 16 * 21 * 60 * 104, randn_tensor_call_count=1,
        generator_initial_seed=candidate["seed"], captured_from_native_sampler=True,
        requested_device="cpu", requested_dtype="torch.float32", generator_device="cpu",
        all_rank_identity={}, coordinate="bernini_native_target_latent_before_rearrange",
        origin="observed_return_of_official_module_global_randn_tensor",
        observer_only=True, external_initial_noise_injection=False, source_or_target_derived=False,
        observer_changed_return_value=False, official_randn_tensor_call_count=1,
        observer_added_device_to_cpu_readback=True,
        official_module_global_symbol="bernini.models.wan_diffusion.randn_tensor",
        original_callable_invoked_once_with_unchanged_arguments=True,
        original_return_tensor_forwarded_by_identity=True, sampler_noise_replacement=False,
        roundtrip_raw_value_exact=True,
    )
    output = _artifact(
        candidate_dir / "rv2v.mp4", b"video", frame_count=81, fps=25, height=480,
        width=832, normalized_clean_latent=clean,
    )
    generation = _generation()
    native = {
        "schema_version": evaluator.EXPECTED_NATIVE_SCHEMA, "method": evaluator.EXPECTED_NATIVE_METHOD,
        "method_source_revision": generation["method_source_revision"],
        "method_source_archive_sha256": generation["method_source_archive_sha256"],
        "bernini_commit": generation["bernini_commit"], "veomni_commit": generation["veomni_commit"],
        "bernini_inference_files": generation["bernini_inference_files"],
        "checkpoint": {"path": str(root / "checkpoint"), "tree_sha256": generation["checkpoint_tree_sha256"], "content": {
            "manifest_path": str(root / "checkpoint-manifest"),
            "manifest_sha256_computed": generation["checkpoint_manifest_sha256"],
            "manifest_sha256_expected": generation["checkpoint_manifest_sha256"],
            "verified_file_count": generation["checkpoint_file_count"],
            "every_file_sha256_verified": True,
            "verified_entries_digest": generation["checkpoint_entries_digest"],
        }},
        "arms": ["rv2v"],
        "input": {
            "source_video_path": candidate["source_video"], "source_video_sha256": candidate["source_video_sha256"],
            "action_prompt_utf8_sha256": candidate["complete_caption_sha256"],
            "action_prompt_utf8_bytes": len(candidate["complete_caption"].encode()),
            "accepted_external_conditions": ["source_video", "action_prompt"],
            "target_video": False, "external_reference_image_or_video": False,
            "external_mask_flow_pose_track_trajectory": False, "external_first_frame_anchor": False,
        },
        "preprocessing": {
            "frame_count": 81, "fps": 25, "reported_fps": 25.0, "source_input_hw": [480, 832],
            "source_derived_bucket_hw": [480, 832], "max_pixels": 245760, "stride": 16,
            "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
            "spatial_policy": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
            "resize": "torchvision_bicubic_antialias_true", "external_shared_i0": False,
            "decoded_from_private_byte_snapshot": True, "snapshot_sha256": candidate["source_video_sha256"],
            "original_pre_snapshot_sha256": candidate["source_video_sha256"],
            "original_post_snapshot_sha256": candidate["source_video_sha256"],
            "original_stat_identity_stable": True,
        },
        "prompt_contract": {"rv2v": {
            "training_task_name": "vr2v", "inference_arm": "rv2v", "guidance_mode": "rv2v",
            "system_prompt_sha256": _digest("system"), "binding_clause_sha256": _digest("binding"),
            "full_prompt_sha256": _digest("full"),
            "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenizer_fix_mistral_regex": True,
        }},
        "conditioning": {"rv2v": {
            "full_source_video_count": 1, "source_derived_reference_count": 4,
            "source_frame_indices": [0, 27, 53, 80],
            "reference_encoding": "independent_rgb_frame_to_wan_vae_[1,C,1,H,W]",
            "reference_from_temporal_video_latent_slice": False,
            "source_ids": {"target_source_id": 0, "video_source_ids": [1],
                "reference_source_ids": [2, 3, 4, 5], "conditioning_source_count": 5,
                "max_conditioning_source_id": 5, "within_pretrained_source_ids_1_through_5": True,
                "source_id_interpolation_required": False},
        }},
        "sampling": {"rv2v": {
            "num_frames": 81, "num_inference_steps": 40, "guidance_mode": "rv2v",
            "omega_vid": 1.25, "omega_img": 4.5, "omega_txt": 4.0, "omega_scale": .8,
            "flow_shift": 5.0, "seed": candidate["seed"], "eta": .5,
            "norm_threshold": [50.0, 50.0], "momentum": 0.0,
            "target_initialization": evaluator.EXPECTED_TARGET_INITIALIZATION,
            "target_mixed_with_source_latent": False, "custom_sampler_or_scheduler": False,
            "same_seed_and_target_shape_across_arms": True, "single_expert": "transformer_1",
            "ulysses_size": 4,
        }},
        "latent_geometry": {"video_latent_shape": shape, "reference_latent_shape": [1, 16, 1, 60, 104],
            "target_patch_tokens": 21 * 30 * 52, "one_reference_patch_tokens": 30 * 52,
            "per_arm_total_visual_tokens": {"t2v": 21 * 30 * 52, "r2v": 26 * 30 * 52, "rv2v": 46 * 30 * 52}},
        "condition_identities": {"rank_zero_broadcasts": {}, "references": {str(i): {} for i in (0, 27, 53, 80)}, "full_source_video": {}},
        "source_condition_artifact": source_artifact, "initial_noise_artifacts": {"rv2v": gaussian},
        "generated_identities": {"rv2v": {}}, "outputs": {"rv2v": output},
        "freeze_certificate": {"base_frozen": True, "trainable_parameter_tensors": 0, "trainable_parameter_elements": 0, "lora_module_count": 0},
        "runtime_versions": generation["runtime_versions"],
        "interpretation": {"purpose": "test_native_identity_conditioned_generation_before_training", "quality_claim": False, "training_performed": False, "best_arm_selected": False},
        "experimental_canary": True, "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    _native_sign(native)
    return native, candidate, candidate_dir


class ContractTests(unittest.TestCase):
    def test_spec_seals_official_processor_and_generation_provenance(self) -> None:
        spec = evaluator.validate_evaluator_spec(_spec())
        self.assertEqual(spec["model"]["preprocessor_golden_output_shape"], [1, 3, 224, 224])
        self.assertEqual(spec["model"]["image_size"], 518)
        self.assertEqual(spec["model"]["patch_size"], 14)
        self.assertEqual(spec["generation_provenance"]["native_method"], evaluator.EXPECTED_NATIVE_METHOD)
        self.assertFalse(spec["scientific_claims"]["absolute_viewpoint_or_camera_decomposition"])

    def test_negative_margin_is_valid_and_eligible(self) -> None:
        spec = _spec(); row = _receipt(spec, correct=.3, wrong=.6)
        checked = evaluator.validate_candidate_receipt(row, evaluator_spec=spec)
        self.assertTrue(checked["evidence_valid"])
        self.assertTrue(checked["eligible_for_downstream_calibration"])
        self.assertFalse(checked["binding_probes"]["diagnostic_ordering_holds"])
        self.assertLess(checked["metrics"]["source_identity_appearance_correct_minus_wrong_margin"], 0)

    def test_source_self_is_upper_bound_not_a_fake_noop_arm(self) -> None:
        spec = _spec(); row = _receipt(spec)
        checked = evaluator.validate_candidate_receipt(row, evaluator_spec=spec)
        self.assertEqual(checked["binding_probes"]["source_self_similarity_upper_bound"], 1.0)
        self.assertNotIn("noop", json.dumps(checked).lower())
        forged = copy.deepcopy(row)
        forged["eligible_for_downstream_calibration"] = False
        _resign(forged, "receipt_digest")
        with self.assertRaises(evaluator.PairV5SourceBoundEvaluationError):
            evaluator.validate_candidate_receipt(forged, evaluator_spec=spec)

    def test_root_completion_requires_two_groups_and_all_eight_eligible(self) -> None:
        spec = _spec(); rows = [_receipt(spec, ordinal=i) for i in range(8)]
        groups = []
        group_files = {}
        for group_id in evaluator.EXPECTED_GROUPS:
            selected = [row for row in rows if row["group_id"] == group_id]
            group = evaluator.make_group_receipt(
                evaluator_spec=spec, evaluator_spec_raw_sha256=_digest("spec-raw"),
                group_id=group_id, candidate_receipts=selected,
                candidate_receipt_file_sha256_by_id={row["candidate_id"]: _digest("file" + row["candidate_id"]) for row in selected},
            )
            groups.append(group); group_files[group_id] = _digest("group" + group_id)
        topology = {
            "group_world_size": 4, "group_ulysses_size": 4,
            "groups": {key: evaluator.EXPECTED_GROUP_GPUS[key] for key in evaluator.EXPECTED_GROUPS},
            "total_physical_gpus": 8, "concurrent_disjoint_groups": True,
        }
        root = evaluator.make_root_receipt(
            evaluator_spec=spec, evaluator_spec_raw_sha256=_digest("spec-raw"),
            group_receipts=groups, group_receipt_file_sha256_by_id=group_files,
            candidate_receipts=rows,
            candidate_receipt_file_sha256_by_id={row["candidate_id"]: _digest("file" + row["candidate_id"]) for row in rows},
            topology=topology,
        )
        self.assertTrue(evaluator.validate_root_receipt(root, evaluator_spec=spec)["complete"])
        # JSON object members are emitted in canonical lexical order, while the
        # explicit candidate_order array retains the preregistered experiment
        # order.  Exercise the durable write/read boundary that the AUH launcher
        # uses; validating only the in-memory object misses ordering drift from
        # canonical_json_bytes(sort_keys=True).
        self.assertNotEqual(spec["candidate_order"], sorted(spec["candidate_order"]))
        self.assertEqual(root["candidate_order"], spec["candidate_order"])
        self.assertEqual(
            list(root["candidate_receipt_digest_by_id"]),
            sorted(spec["candidate_order"]),
        )
        serialized = evaluator.canonical_json_bytes(root)
        durable = json.loads(serialized)
        self.assertTrue(
            evaluator.validate_root_receipt(durable, evaluator_spec=spec)["complete"]
        )
        self.assertEqual(root["eligible_for_downstream_calibration_count"], 8)
        self.assertTrue(root["exploratory_dev_only"])
        self.assertFalse(root["action_score_dependency"])

    def test_generation_provenance_mutation_fails_even_if_spec_resigned(self) -> None:
        spec = copy.deepcopy(_spec())
        spec["generation_provenance"]["native_method"] = "forged-native-method"
        _resign(spec["generation_provenance"], "provenance_digest")
        _resign(spec, "spec_digest")
        with self.assertRaises(evaluator.PairV5SourceBoundEvaluationError):
            evaluator.validate_evaluator_spec(spec)


class NativeProvenanceAuditTests(unittest.TestCase):
    @staticmethod
    def _write_utf8_json(path: Path, value: dict) -> str:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_full_native_receipt_accepts_only_frozen_unprivileged_exact81_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            native, candidate, candidate_dir = _native_fixture(Path(tmp).resolve())
            spec = _spec()
            checked = scorer._verify_native_receipt(
                native, candidate=candidate, candidate_dir=candidate_dir,
                evaluator_spec=spec, native_file_sha256=_digest("native-file"),
            )
            self.assertEqual(checked["native_generation_provenance_digest"], spec["generation_provenance"]["provenance_digest"])
            mutations = (
                lambda row: row["input"].__setitem__("target_video", True),
                lambda row: row["sampling"]["rv2v"].__setitem__("ulysses_size", 2),
                lambda row: row["freeze_certificate"].__setitem__("base_frozen", False),
                lambda row: row["interpretation"].__setitem__("training_performed", True),
            )
            for mutate in mutations:
                forged = copy.deepcopy(native); mutate(forged); _native_sign(forged)
                with self.subTest(mutation=mutate), self.assertRaises(scorer.PairV5SourceBoundScoringError):
                    scorer._verify_native_receipt(
                        forged, candidate=candidate, candidate_dir=candidate_dir,
                        evaluator_spec=spec, native_file_sha256=_digest("native-file"),
                    )

    def test_native_artifact_path_cannot_escape_candidate_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); native, candidate, candidate_dir = _native_fixture(root)
            escaped = root / "escaped.mp4"; escaped.write_bytes(b"video")
            native["outputs"]["rv2v"]["path"] = str(escaped); _native_sign(native)
            with self.assertRaisesRegex(scorer.PairV5SourceBoundScoringError, "escaped candidate directory"):
                scorer._verify_native_receipt(
                    native, candidate=candidate, candidate_dir=candidate_dir,
                    evaluator_spec=_spec(), native_file_sha256=_digest("native-file"),
                )

    def test_pair_envelope_topology_and_native_artifact_equality_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rollout_root = Path(tmp).resolve(); native, candidate, candidate_dir = _native_fixture(rollout_root)
            native_sha = self._write_utf8_json(candidate_dir / "receipt.json", native)
            raw_candidate = {key: value for key, value in candidate.items() if key != "group_id"}
            envelope = {
                "schema_version": "pair-v5-native-rv2v4-candidate-v1",
                "root_spec_raw_sha256": evaluator.CURRENT_FAMILY_ROLLOUT_SPEC_RAW_SHA256,
                "group_id": "sp4-a", "visible_gpus": [0, 1, 2, 3], "ordinal": 0,
                "sampling_contract": scorer._PAIR_SAMPLING,
                "semantic_input_closure": scorer._PAIR_SEMANTIC_CLOSURE,
                "candidate": raw_candidate,
            }
            envelope_sha = hashlib.sha256(json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n").hexdigest()
            pair = {
                "schema_version": evaluator.EXPECTED_ROLLOUT_RECEIPT_SCHEMA,
                "root_spec_raw_sha256": evaluator.CURRENT_FAMILY_ROLLOUT_SPEC_RAW_SHA256,
                "candidate_envelope_sha256": envelope_sha, "group_id": "sp4-a",
                "visible_gpus": [0, 1, 2, 3],
                "runtime_topology": {"world_size": 4, "ulysses_size": 4, "rocr_visible_devices": "0,1,2,3"},
                "ordinal": 0, "candidate": raw_candidate,
                "sampling_contract": scorer._PAIR_SAMPLING,
                "semantic_input_closure": scorer._PAIR_SEMANTIC_CLOSURE,
                "native_receipt_path": str(candidate_dir / "receipt.json"),
                "native_receipt_sha256": native_sha, "native_receipt_digest": native["receipt_digest"],
                "artifacts": {"mp4": native["outputs"]["rv2v"],
                    "predecode_clean_latent": native["outputs"]["rv2v"]["normalized_clean_latent"],
                    "official_initial_gaussian": native["initial_noise_artifacts"]["rv2v"]},
            }
            _native_sign(pair)
            # _native_sign uses the same receipt_digest canonicalizer as the PAIR wrapper.
            self._write_utf8_json(candidate_dir / "pair-v5-rollout-receipt.json", pair)
            normalized = evaluator._validate_current_family_rollout_spec(_payload())
            audited = scorer.audit_rollout_candidate(
                candidate=candidate, normalized_rollout=normalized,
                rollout_root=rollout_root,
                rollout_spec_raw_sha256=evaluator.CURRENT_FAMILY_ROLLOUT_SPEC_RAW_SHA256,
                evaluator_spec=_spec(),
            )
            self.assertEqual(audited["candidate_envelope_sha256"], envelope_sha)
            pair["candidate_envelope_sha256"] = _digest("forged-envelope"); _native_sign(pair)
            self._write_utf8_json(candidate_dir / "pair-v5-rollout-receipt.json", pair)
            with self.assertRaisesRegex(scorer.PairV5SourceBoundScoringError, "envelope/topology"):
                scorer.audit_rollout_candidate(
                    candidate=candidate, normalized_rollout=normalized,
                    rollout_root=rollout_root,
                    rollout_spec_raw_sha256=evaluator.CURRENT_FAMILY_ROLLOUT_SPEC_RAW_SHA256,
                    evaluator_spec=_spec(),
                )


@unittest.skipIf(torch is None, "torch unavailable")
class MetricTests(unittest.TestCase):
    def test_native_model_geometry_is_distinct_from_evaluation_patch_grid(self) -> None:
        class Model:
            def __call__(self, *, pixel_values):
                del pixel_values
                output = type("Output", (), {})()
                output.last_hidden_state = torch.ones(
                    (len(evaluator.EVAL_FRAME_INDICES), 1 + 16 * 16, 8),
                    dtype=torch.float32,
                )
                return output

        global_feature, dense_feature, evidence = scorer.extract_features(
            Model(),
            torch.zeros(
                (len(evaluator.EVAL_FRAME_INDICES), 3, 224, 224),
                dtype=torch.float32,
            ),
            device=torch.device("cpu"),
            num_register_tokens=0,
        )
        self.assertEqual(tuple(global_feature.shape), (17, 8))
        self.assertEqual(tuple(dense_feature.shape), (17, 256, 8))
        self.assertEqual(evidence["dense_grid_height"], 16)
        self.assertEqual(evaluator.MODEL_NATIVE_IMAGE_SIZE, 518)

    def test_layout_proxy_penalizes_flip_zoom_and_perspective_like_warps(self) -> None:
        side = 4
        base = torch.eye(side * side).reshape(1, side, side, side * side).repeat(3, 1, 1, 1)
        base = base.reshape(3, side * side, side * side)
        flip = base.reshape(3, side, side, -1).flip(2).reshape_as(base)
        zoom = torch.roll(base.reshape(3, side, side, -1), shifts=1, dims=1).reshape_as(base)
        perspective = torch.roll(base.reshape(3, side, side, -1), shifts=1, dims=2).reshape_as(base)
        identical = scorer.spatial_layout_viewpoint_similarity(base, base)
        self.assertGreater(identical, scorer.spatial_layout_viewpoint_similarity(flip, base))
        self.assertGreater(identical, scorer.spatial_layout_viewpoint_similarity(zoom, base))
        self.assertGreater(identical, scorer.spatial_layout_viewpoint_similarity(perspective, base))

    def test_temporal_translation_is_diagnostic_only_and_static_quality_is_not_zero(self) -> None:
        frames = torch.full((len(evaluator.EVAL_FRAME_INDICES), 3, 224, 224), .5)
        quality = scorer.quality_diagnostics(frames, frames)
        self.assertEqual(quality["quality_sharpness_retention"], 1.0)
        self.assertEqual(quality["quality_nonfreeze_score"], 1.0)
        # FFT translation cannot distinguish a static viewpoint warp; the spatial proxy above can.
        self.assertEqual(scorer.temporal_translation_agreement(frames, frames), 1.0)


class AuthorTests(unittest.TestCase):
    def test_checkpoint_seals_preprocessor_and_model_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); checkpoint = root / "checkpoint"; checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps({"model_type": "dinov2", "num_register_tokens": 0, "image_size": 518, "patch_size": 14}))
            (checkpoint / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "BitImageProcessor"}))
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            manifest = root / "SHA256SUMS"
            manifest.write_text("\n".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.name}" for path in sorted(checkpoint.iterdir())) + "\n")
            with mock.patch.object(scorer, "inspect_official_processor", return_value={
                "processor": object(), "preprocessor_golden_input_sha256": _digest("golden-input"),
                "preprocessor_golden_output_sha256": _digest("golden-output"),
                "preprocessor_golden_output_shape": [1, 3, 224, 224],
            }):
                checked = author.inspect_checkpoint(checkpoint, manifest, expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(checked["image_size"], 518)
            self.assertEqual(checked["patch_size"], 14)
            self.assertEqual(checked["preprocessor_config_sha256"], hashlib.sha256((checkpoint / "preprocessor_config.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Sealed contract for PAIR-v5 source-bound preservation evidence.

This evaluator is intentionally *post-video* and action agnostic.  It compares
one rendered native RV2V candidate with its own sealed source and a sealed,
source-disjoint wrong source.  It emits raw visual-feature and deterministic
video diagnostics for later calibration/Pareto selection; it does not claim
that actor identity, background, camera, or action editing succeeded.

No target video, proposal, donor, mask, flow, pose, track, or trajectory is an
input.  Dense metrics use every token on a fixed square feature grid.  The
``non_target_temporal_consistency_proxy`` name records the intended use, but
the receipt explicitly states that there is no target localization and that
the quantity is only a whole-grid, background-dominant proxy.

The spec binds the exact eight-candidate current-family rollout order, wrong
source permutation, evaluator implementation, DINO-style checkpoint content,
runtime versions, and deterministic preprocessing.  Receipts bind every media
artifact, decode/preprocess/feature digest, metric, probe, and ordering field.
Any malformed, non-finite, hash-unbound, or decode-invalid row is ineligible.
Every provenance-valid row remains eligible even when a diagnostic contrast
is negative.  No absolute visual threshold is defined here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


SPEC_SCHEMA = "bernini-pair-v5-source-bound-preservation-evaluator-spec-v1"
RECEIPT_SCHEMA = "bernini-pair-v5-source-bound-preservation-candidate-v1"
GROUP_SCHEMA = "bernini-pair-v5-source-bound-preservation-group-v1"
ROOT_SCHEMA = "bernini-pair-v5-source-bound-preservation-root-v1"
FAILURE_SCHEMA = "bernini-pair-v5-source-bound-preservation-failure-v1"

EXPECTED_ROLLOUT_SCHEMA = "pair-v5-native-rv2v4-rollout-spec-v1"
EXPECTED_ROLLOUT_RECEIPT_SCHEMA = "pair-v5-native-rv2v4-rollout-receipt-v1"
CURRENT_FAMILY_ROLLOUT_SPEC_RAW_SHA256 = (
    "525d727951ee05d7aac27f47d294e3604996781106dfc710087d4029a1bbd8f0"
)
EXPECTED_CANDIDATE_COUNT = 8
EXPECTED_SOURCE_COUNT = 4
EXPECTED_GROUPS = ("sp4-a", "sp4-b")
EXPECTED_GROUP_GPUS = {"sp4-a": [0, 1, 2, 3], "sp4-b": [4, 5, 6, 7]}
EXPECTED_SEEDS = (2026080901, 2026080902)
EXPECTED_TARGET_INITIALIZATION = "official_gen_wanx22_fresh_gaussian"
FRAME_COUNT = 81
FPS = 25
EVAL_FRAME_INDICES = tuple(range(0, FRAME_COUNT, 5))
REFERENCE_FRAME_INDICES = (0, 27, 53, 80)

MODEL_ADAPTER_ID = "hf-dinov2-last-hidden-state-square-patch-grid-v1"
SUPPORTED_ARCHITECTURES = ("dinov2",)
MODEL_NATIVE_IMAGE_SIZE = 518
MODEL_PATCH_SIZE = 14
EVALUATION_IMAGE_SIZE = 224
EXPECTED_NATIVE_SCHEMA = "bernini-native-identity-generation-canary-v1"
EXPECTED_NATIVE_METHOD = "frozen-bernini-native-identity-generation-canary"
FEATURE_ORDER = ("global_cls", "dense_patch")
METRIC_ORDER = (
    "source_identity_appearance_proxy",
    "background_appearance_fixed_grid_proxy",
    "non_target_temporal_consistency_proxy",
    "source_bound_spatial_layout_viewpoint_proxy",
    "source_bound_spatial_layout_wrong_normalized_contrast_proxy",
    "temporal_global_translation_agreement_diagnostic",
    "decode_video_quality_diagnostic",
)
PROBE_ORDER = (
    "correct_source",
    "wrong_source",
    "source_self_upper_bound",
    "reference_off",
)

PREPROCESS_CONTRACT = {
    "decoder": "pyav-video-stream-0-rgb24-presentation-order-v1",
    "decode_thread_count": 1,
    "required_frame_count": FRAME_COUNT,
    "required_fps_numerator": FPS,
    "required_fps_denominator": 1,
    "selected_frame_indices": list(EVAL_FRAME_INDICES),
    "rollout_reference_frame_indices_provenance_only": list(REFERENCE_FRAME_INDICES),
    "input_range": "uint8_rgb_0_255",
    "image_processor": "transformers.AutoImageProcessor-use_fast_false",
    "image_processor_type": "BitImageProcessor",
    "processor_backend": "PIL",
    "preprocessor_config_hash_bound": True,
    "resize": {
        "mode": "PIL.Image.Resampling.BICUBIC",
        "shorter_side": 256,
        "long_edge_rounding": "floor_int_matching_transformers_4_53_2",
    },
    "crop": {"kind": "integer_center_crop", "height": 224, "width": 224},
    "processor_rescaled_range": "float32_rgb_0_1_before_normalization",
    "model_input_range": "float32_imagenet_mean_std_normalized",
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "feature_normalization": "float32_l2_last_dimension_epsilon_1e-12",
    "dense_token_rule": (
        "drop_cls_then_config_num_register_tokens_then_require_square_patch_count"
    ),
    "batch_order": "candidate_then_correct_source_then_wrong_source",
    "autocast": False,
    "torch_deterministic_algorithms": True,
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "transformers_attention_implementation": "eager",
}

METRIC_CONTRACT = {
    "wrong_source_policy": (
        "registered_current_family_half_cycle_same_actor_class_dog_to_dog_human_to_human"
    ),
    "global_similarity": (
        "mean_over_selected_aligned_frames_of_unit_mapped_cosine_cls"
    ),
    "dense_similarity": (
        "median_over_selected_frames_and_all_fixed_grid_tokens_of_unit_mapped_cosine"
    ),
    "temporal_consistency": (
        "exp_negative_median_absolute_difference_of_adjacent_dense_cosine_change"
    ),
    "temporal_translation_diagnostic": (
        "exp_negative_mean_l1_difference_of_normalized_fft_phase_correlation_steps"
    ),
    "temporal_translation_phase_correlation_config": {
        "grayscale_weights": [0.2989, 0.587, 0.114],
        "working_height": 96,
        "working_width": 96,
        "window": "nonperiodic_hann_outer_product",
        "cross_power_epsilon": 1.0e-12,
        "minimum_valid_frequency_bins": 96,
    },
    "quality_diagnostic": (
        "geometric_mean_of_finite_sharpness_exposure_nonfreeze_and_flicker_terms"
    ),
    "quality_diagnostic_config": {
        "exposure_black_threshold": 2.0 / 255.0,
        "exposure_white_threshold": 253.0 / 255.0,
        "sharpness_kind": "mean_squared_first_spatial_difference_ratio_clipped_to_one_with_both_zero_equal_one",
        "nonfreeze_kind": "mean_absolute_frame_step_ratio_clipped_to_one_with_static_source_equal_one",
        "flicker_kind": "exp_negative_scale_times_global_rgb_mean_second_difference_error",
        "flicker_scale": 10.0,
        "ratio_denominator_epsilon": 1.0e-12,
        "geometric_mean_epsilon": 1.0e-12,
    },
    "correct_wrong_margin": "correct_source_global_similarity_minus_wrong_source_global_similarity",
    "source_self_upper_bound_headroom": (
        "source_self_similarity_upper_bound_minus_correct_source_global_similarity"
    ),
    "wrong_normalized_contrast": (
        "piecewise_(correct-wrong)/(source_self_upper_bound-wrong)_when_upper_bound_greater_than_wrong_else_zero"
    ),
    "spatial_layout_viewpoint_proxy": (
        "same_position_dense_patch_cosine_to_correct_source_with_wrong_source_normalized_contrast"
    ),
    "eligibility_policy": "provenance_decode_model_and_metric_evidence_valid_only",
    "absolute_acceptance_thresholds": None,
    "calibration_required_before_selection": True,
}

INPUT_CLOSURE = {
    "accepted_media": [
        "candidate_own_rendered_rv2v_mp4",
        "candidate_own_sealed_source_mp4",
        "sealed_source_disjoint_wrong_source_mp4",
    ],
    "accepted_semantics": [],
    "target_video": False,
    "paired_target": False,
    "t2v_proposal": False,
    "donor": False,
    "external_reference": False,
    "mask": False,
    "flow": False,
    "pose": False,
    "track": False,
    "trajectory": False,
    "caption_consumed_by_visual_evaluator": False,
    "caption_and_action_text_hash_bound_as_rollout_provenance_only": True,
    "action_success_scored": False,
    "training_performed": False,
}

SCIENTIFIC_CLAIMS = {
    "actor_identity_isolated": False,
    "background_isolated": False,
    "target_or_non_target_localized": False,
    "camera_motion_estimated_without_scene_motion_confounding": False,
    "absolute_viewpoint_or_camera_decomposition": False,
    "spatial_layout_viewpoint_proxy_is_whole_grid": True,
    "quality_is_human_preference": False,
    "action_editing_success": False,
    "raw_evidence_only": True,
    "requires_sealed_downstream_calibration_and_pareto_policy": True,
}

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

_MODEL_FIELDS = frozenset(
    {
        "adapter_id",
        "architecture_id",
        "checkpoint_manifest_sha256",
        "checkpoint_config_sha256",
        "preprocessor_config_sha256",
        "checkpoint_file_count",
        "num_register_tokens",
        "image_size",
        "patch_size",
        "preprocessor_golden_input_sha256",
        "preprocessor_golden_output_sha256",
        "preprocessor_golden_output_shape",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "python_version",
        "torch_version",
        "torch_hip_version",
        "transformers_version",
        "safetensors_version",
        "av_version",
        "numpy_version",
        "pillow_version",
    }
)
_GENERATION_FIELDS = frozenset(
    {
        "reference_native_receipt_file_sha256",
        "native_schema_version",
        "native_method",
        "method_source_revision",
        "method_source_archive_sha256",
        "bernini_commit",
        "veomni_commit",
        "bernini_inference_files",
        "checkpoint_tree_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_file_count",
        "checkpoint_entries_digest",
        "runtime_versions",
        "provenance_digest",
    }
)
_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "rollout_spec_raw_sha256",
        "candidate_order",
        "candidate_group_by_id",
        "correct_source_sha256_by_candidate_id",
        "source_order",
        "wrong_source_by_source_sha256",
        "implementation_sha256",
        "contract_sha256",
        "method_source_revision",
        "method_source_archive_sha256",
        "model",
        "generation_provenance",
        "runtime_versions",
        "preprocess_contract",
        "metric_order",
        "metric_contract",
        "probe_order",
        "input_closure",
        "scientific_claims",
        "spec_digest",
    }
)
_DECODE_FIELDS = frozenset(
    {
        "artifact_sha256",
        "decoded_rgb_sha256",
        "frame_count",
        "fps_numerator",
        "fps_denominator",
        "time_base_numerator",
        "time_base_denominator",
        "pts_step",
        "pts_sha256",
        "width",
        "height",
        "selected_frame_indices",
        "selected_rgb_sha256",
        "preprocessed_tensor_sha256",
    }
)
_FEATURE_FIELDS = frozenset(
    {
        "global_feature_sha256",
        "dense_feature_sha256",
        "selected_frame_count",
        "dense_grid_height",
        "dense_grid_width",
        "feature_dimension",
    }
)
_MODEL_EVIDENCE_FIELDS = frozenset(
    {
        "adapter_id",
        "architecture_id",
        "checkpoint_manifest_sha256",
        "checkpoint_config_sha256",
        "preprocessor_config_sha256",
        "checkpoint_file_count",
        "verified_entries_digest",
        "preprocessor_golden_input_sha256",
        "preprocessor_golden_output_sha256",
        "preprocessor_golden_output_shape",
        "every_checkpoint_file_verified",
        "all_parameters_frozen",
        "trainable_parameter_tensors",
        "parameter_tensor_count",
        "parameter_element_count",
        "parameter_metadata_digest",
        "missing_key_count",
        "unexpected_key_count",
        "mismatched_key_count",
        "loading_error_count",
        "runtime_versions",
    }
)
_METRIC_FIELDS = frozenset(
    {
        "source_identity_appearance_proxy",
        "source_identity_appearance_wrong_source_proxy",
        "source_identity_appearance_correct_minus_wrong_margin",
        "source_identity_appearance_source_self_upper_bound",
        "source_identity_appearance_upper_bound_minus_correct_headroom",
        "source_identity_appearance_wrong_normalized_contrast",
        "background_appearance_fixed_grid_proxy",
        "background_appearance_wrong_source_fixed_grid_proxy",
        "background_appearance_correct_minus_wrong_margin",
        "non_target_temporal_consistency_proxy",
        "non_target_temporal_consistency_wrong_source_proxy",
        "source_bound_spatial_layout_viewpoint_proxy",
        "source_bound_spatial_layout_wrong_source_proxy",
        "source_bound_spatial_layout_correct_minus_wrong_margin",
        "source_bound_spatial_layout_wrong_normalized_contrast_proxy",
        "temporal_global_translation_agreement_diagnostic",
        "decode_video_quality_diagnostic",
        "quality_sharpness_retention",
        "quality_exposure_score",
        "quality_nonfreeze_score",
        "quality_flicker_score",
    }
)
_PROBE_FIELDS = frozenset(
    {
        "order",
        "correct_source_sha256",
        "wrong_source_sha256",
        "source_disjoint",
        "correct_source_global_similarity",
        "wrong_source_global_similarity",
        "correct_minus_wrong_margin",
        "source_self_similarity_upper_bound",
        "upper_bound_minus_correct_headroom",
        "upper_bound_minus_wrong_denominator",
        "wrong_normalized_contrast",
        "contrast_denominator_positive",
        "reference_off_applicable",
        "reference_off_reason",
        "strict_correct_greater_than_wrong",
        "upper_bound_not_below_correct",
        "diagnostic_ordering_holds",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "evaluator_spec_digest",
        "evaluator_spec_raw_sha256",
        "rollout_spec_raw_sha256",
        "candidate_id",
        "candidate_ordinal",
        "group_id",
        "candidate_order_digest",
        "candidate_envelope_sha256",
        "rollout_receipt_digest",
        "rollout_receipt_file_sha256",
        "native_rollout_receipt_digest",
        "native_rollout_receipt_file_sha256",
        "native_generation_provenance_digest",
        "candidate_mp4_sha256",
        "predecode_clean_latent_sha256",
        "official_initial_gaussian_sha256",
        "correct_source_video_sha256",
        "wrong_source_video_sha256",
        "decode_evidence_by_role",
        "feature_evidence_by_role",
        "model_evidence",
        "metrics",
        "binding_probes",
        "metric_order",
        "input_closure",
        "scientific_claims",
        "evidence_valid",
        "eligible_for_downstream_calibration",
        "absolute_source_preservation_pass_claim",
        "receipt_digest",
    }
)
_GROUP_FIELDS = frozenset(
    {
        "schema_version",
        "evaluator_spec_digest",
        "evaluator_spec_raw_sha256",
        "rollout_spec_raw_sha256",
        "group_id",
        "visible_gpus",
        "candidate_order",
        "candidate_receipt_digest_by_id",
        "candidate_receipt_file_sha256_by_id",
        "candidate_count",
        "eligible_for_downstream_calibration_count",
        "all_evidence_valid",
        "all_candidates_eligible_for_downstream_calibration",
        "group_digest",
    }
)
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "evaluator_spec_digest",
        "evaluator_spec_raw_sha256",
        "rollout_spec_raw_sha256",
        "method_source_revision",
        "method_source_archive_sha256",
        "model_checkpoint_manifest_sha256",
        "model_checkpoint_config_sha256",
        "model_preprocessor_config_sha256",
        "generation_checkpoint_tree_sha256",
        "generation_checkpoint_manifest_sha256",
        "generation_provenance_digest",
        "runtime_versions",
        "topology",
        "group_order",
        "group_receipt_digest_by_id",
        "group_receipt_file_sha256_by_id",
        "candidate_order",
        "candidate_receipt_digest_by_id",
        "candidate_receipt_file_sha256_by_id",
        "candidate_count",
        "eligible_for_downstream_calibration_count",
        "all_evidence_valid",
        "complete",
        "exploratory_dev_only",
        "action_score_dependency",
        "absolute_source_preservation_pass_claims",
        "root_digest",
    }
)
_FAILURE_FIELDS = frozenset(
    {
        "schema_version",
        "evaluator_spec_digest",
        "evaluator_spec_raw_sha256",
        "rollout_spec_raw_sha256",
        "candidate_id",
        "candidate_ordinal",
        "group_id",
        "failure_stage",
        "error_class",
        "error_message_sha256",
        "evidence_valid",
        "eligible_for_downstream_calibration",
        "absolute_source_preservation_pass_claim",
        "receipt_digest",
    }
)


class PairV5SourceBoundEvaluationError(RuntimeError):
    """A sealed evaluator input, evidence row, or receipt is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV5SourceBoundEvaluationError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    value = Path(path)
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise PairV5SourceBoundEvaluationError("hash target must be an absolute plain file")
    before = value.stat()
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = value.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PairV5SourceBoundEvaluationError("file changed while hashing")
    return digest.hexdigest()


def _closed(value: Any, fields: frozenset[str] | set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise PairV5SourceBoundEvaluationError(
            f"{label} closure differs; missing={sorted(set(fields) - actual)}, "
            f"extra={sorted(actual - set(fields))}"
        )
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PairV5SourceBoundEvaluationError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PairV5SourceBoundEvaluationError(f"{label} must be a safe identifier")
    return value


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise PairV5SourceBoundEvaluationError(f"{label} must be boolean")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PairV5SourceBoundEvaluationError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairV5SourceBoundEvaluationError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PairV5SourceBoundEvaluationError(f"{label} must be finite numeric")
    return result


def _unit(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise PairV5SourceBoundEvaluationError(f"{label} must lie in [0,1]")
    return result


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise PairV5SourceBoundEvaluationError(f"{label} must be lowercase full SHA-1")
    return value


def _version_map(value: Any, *, label: str) -> dict[str, str]:
    fields = ("torch", "torch_hip", "transformers", "diffusers")
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise PairV5SourceBoundEvaluationError(f"{label} closure/order differs")
    result: dict[str, str] = {}
    for key in fields:
        item = value[key]
        if not isinstance(item, str) or not item or "\x00" in item or len(item) > 128:
            raise PairV5SourceBoundEvaluationError(f"{label} {key} differs")
        result[key] = item
    return result


def validate_generation_provenance(value: Any) -> dict[str, Any]:
    """Validate the path-free, action-agnostic native-generation seal."""

    row = _closed(value, _GENERATION_FIELDS, label="native generation provenance")
    inference_files = row["bernini_inference_files"]
    if not isinstance(inference_files, Mapping) or not inference_files:
        raise PairV5SourceBoundEvaluationError("Bernini inference-file seal is empty")
    checked_files: dict[str, str] = {}
    for path, digest in inference_files.items():
        if (
            not isinstance(path, str)
            or not path
            or "\x00" in path
            or path in checked_files
        ):
            raise PairV5SourceBoundEvaluationError("Bernini inference-file key differs")
        checked_files[path] = _sha256(digest, label=f"Bernini inference file {path}")
    normalized = {
        "reference_native_receipt_file_sha256": _sha256(
            row["reference_native_receipt_file_sha256"],
            label="reference native receipt file SHA-256",
        ),
        "native_schema_version": row["native_schema_version"],
        "native_method": row["native_method"],
        "method_source_revision": _sha1(
            row["method_source_revision"], label="generation method source revision"
        ),
        "method_source_archive_sha256": _sha256(
            row["method_source_archive_sha256"],
            label="generation method source archive SHA-256",
        ),
        "bernini_commit": _sha1(row["bernini_commit"], label="Bernini commit"),
        "veomni_commit": _sha1(row["veomni_commit"], label="VeOmni commit"),
        "bernini_inference_files": checked_files,
        "checkpoint_tree_sha256": _sha256(
            row["checkpoint_tree_sha256"], label="generation checkpoint tree SHA-256"
        ),
        "checkpoint_manifest_sha256": _sha256(
            row["checkpoint_manifest_sha256"],
            label="generation checkpoint manifest SHA-256",
        ),
        "checkpoint_file_count": _integer(
            row["checkpoint_file_count"], label="generation checkpoint file count", minimum=1
        ),
        "checkpoint_entries_digest": _sha256(
            row["checkpoint_entries_digest"],
            label="generation checkpoint entries digest",
        ),
        "runtime_versions": _version_map(
            row["runtime_versions"], label="generation runtime versions"
        ),
        "provenance_digest": _sha256(
            row["provenance_digest"], label="generation provenance digest"
        ),
    }
    if (
        normalized["native_schema_version"] != EXPECTED_NATIVE_SCHEMA
        or normalized["native_method"] != EXPECTED_NATIVE_METHOD
    ):
        raise PairV5SourceBoundEvaluationError("native generation schema/method differs")
    unsigned = dict(normalized)
    declared = unsigned.pop("provenance_digest")
    if object_sha256(unsigned) != declared:
        raise PairV5SourceBoundEvaluationError("generation provenance digest differs")
    return normalized


def generation_provenance_from_native_receipt(
    value: Any, *, reference_file_sha256: str
) -> dict[str, Any]:
    """Extract the static generation seal from one fully digested native receipt."""

    if not isinstance(value, Mapping):
        raise PairV5SourceBoundEvaluationError("reference native receipt root differs")
    checkpoint = value.get("checkpoint")
    content = checkpoint.get("content") if isinstance(checkpoint, Mapping) else None
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"path", "tree_sha256", "content"}:
        raise PairV5SourceBoundEvaluationError("reference generation checkpoint closure differs")
    if not isinstance(content, Mapping) or set(content) != {
        "manifest_path",
        "manifest_sha256_computed",
        "manifest_sha256_expected",
        "verified_file_count",
        "every_file_sha256_verified",
        "verified_entries_digest",
    }:
        raise PairV5SourceBoundEvaluationError("reference checkpoint content closure differs")
    manifest = _sha256(
        content["manifest_sha256_computed"], label="reference checkpoint manifest SHA-256"
    )
    if (
        content["manifest_sha256_expected"] != manifest
        or content["every_file_sha256_verified"] is not True
    ):
        raise PairV5SourceBoundEvaluationError("reference checkpoint content did not close")
    unsigned = {
        "reference_native_receipt_file_sha256": _sha256(
            reference_file_sha256, label="reference native receipt file SHA-256"
        ),
        "native_schema_version": value.get("schema_version"),
        "native_method": value.get("method"),
        "method_source_revision": value.get("method_source_revision"),
        "method_source_archive_sha256": value.get("method_source_archive_sha256"),
        "bernini_commit": value.get("bernini_commit"),
        "veomni_commit": value.get("veomni_commit"),
        "bernini_inference_files": value.get("bernini_inference_files"),
        "checkpoint_tree_sha256": checkpoint.get("tree_sha256"),
        "checkpoint_manifest_sha256": manifest,
        "checkpoint_file_count": content.get("verified_file_count"),
        "checkpoint_entries_digest": content.get("verified_entries_digest"),
        "runtime_versions": value.get("runtime_versions"),
    }
    return validate_generation_provenance(
        {**unsigned, "provenance_digest": object_sha256(unsigned)}
    )


def _strict_json(
    path: str | Path, *, expected_sha256: str | None, label: str
) -> tuple[dict[str, Any], str]:
    value = Path(path)
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise PairV5SourceBoundEvaluationError(f"{label} must be an absolute plain file")
    raw = value.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _sha256(
        expected_sha256, label=f"{label} expected SHA-256"
    ):
        raise PairV5SourceBoundEvaluationError(f"{label} raw SHA-256 differs")

    def reject_constant(token: str) -> None:
        raise PairV5SourceBoundEvaluationError(f"{label} contains {token}")

    def reject_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PairV5SourceBoundEvaluationError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = item
        return result

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5SourceBoundEvaluationError(f"{label} is invalid JSON") from error
    if not isinstance(decoded, dict):
        raise PairV5SourceBoundEvaluationError(f"{label} root must be an object")
    return decoded, digest


def _validate_current_family_rollout_spec(value: Any) -> dict[str, Any]:
    root = _closed(
        value,
        {"schema_version", "sampling_contract", "semantic_input_closure", "groups"},
        label="rollout spec",
    )
    if root["schema_version"] != EXPECTED_ROLLOUT_SCHEMA:
        raise PairV5SourceBoundEvaluationError("rollout schema differs")
    sampling = root["sampling_contract"]
    if not isinstance(sampling, Mapping) or (
        sampling.get("num_frames") != FRAME_COUNT
        or sampling.get("latent_frames") != 21
        or sampling.get("fps") != FPS
        or sampling.get("num_inference_steps") != 40
        or sampling.get("source_reference_indices") != list(REFERENCE_FRAME_INDICES)
        or sampling.get("condition_mode") != "rv2v4"
        or sampling.get("target_initialization") != EXPECTED_TARGET_INITIALIZATION
    ):
        raise PairV5SourceBoundEvaluationError("rollout is not native RV2V-4 exact81/40")
    closure = root["semantic_input_closure"]
    if not isinstance(closure, Mapping) or closure.get("accepted") != [
        "source_video",
        "complete_caption",
    ]:
        raise PairV5SourceBoundEvaluationError("rollout semantic input closure differs")
    if any(closure.get(key) is not False for key in closure if key != "accepted"):
        raise PairV5SourceBoundEvaluationError("rollout contains privileged semantic input")
    groups = root["groups"]
    if not isinstance(groups, list) or len(groups) != 2:
        raise PairV5SourceBoundEvaluationError("rollout requires exactly two groups")
    candidates: list[dict[str, Any]] = []
    for group, expected_group in zip(groups, EXPECTED_GROUPS):
        group = _closed(
            group, {"group_id", "visible_gpus", "candidates"}, label="rollout group"
        )
        if (
            group["group_id"] != expected_group
            or group["visible_gpus"] != EXPECTED_GROUP_GPUS[expected_group]
            or not isinstance(group["candidates"], list)
            or len(group["candidates"]) != 4
        ):
            raise PairV5SourceBoundEvaluationError("rollout group topology differs")
        for candidate in group["candidates"]:
            expected_fields = {
                "candidate_id",
                "source_video",
                "source_video_sha256",
                "complete_caption",
                "complete_caption_sha256",
                "caption_contract",
                "seed",
                "guidance",
            }
            row = _closed(candidate, expected_fields, label="rollout candidate")
            candidate_id = _safe_id(row["candidate_id"], label="candidate_id")
            lowered = candidate_id.lower()
            if not candidate_id.startswith("pair5-native-core4-v1-") or "-action-s" not in lowered:
                raise PairV5SourceBoundEvaluationError(
                    "candidate is not in the current action-only core4 family"
                )
            if any(
                token in lowered
                for token in ("noop", "reverse", "shuffle", "wrong", "negative")
            ):
                raise PairV5SourceBoundEvaluationError("non-action candidate entered population")
            source = Path(str(row["source_video"]))
            if not source.is_absolute() or source == Path("/"):
                raise PairV5SourceBoundEvaluationError("source path must be absolute non-root")
            source_sha = _sha256(row["source_video_sha256"], label="source SHA-256")
            caption = row["complete_caption"]
            if not isinstance(caption, str) or not caption.strip() or "\x00" in caption:
                raise PairV5SourceBoundEvaluationError("candidate caption differs")
            if hashlib.sha256(caption.encode("utf-8")).hexdigest() != _sha256(
                row["complete_caption_sha256"], label="caption SHA-256"
            ):
                raise PairV5SourceBoundEvaluationError("candidate caption hash differs")
            if type(row["seed"]) is not int or row["seed"] not in EXPECTED_SEEDS:
                raise PairV5SourceBoundEvaluationError("current-family seed differs")
            candidates.append(
                {
                    **dict(row),
                    "source_video": str(source),
                    "source_video_sha256": source_sha,
                    "group_id": expected_group,
                }
            )
    ids = [row["candidate_id"] for row in candidates]
    if len(ids) != EXPECTED_CANDIDATE_COUNT or len(set(ids)) != len(ids):
        raise PairV5SourceBoundEvaluationError("candidate identity closure differs")
    source_order = list(dict.fromkeys(row["source_video_sha256"] for row in candidates))
    if len(source_order) != EXPECTED_SOURCE_COUNT:
        raise PairV5SourceBoundEvaluationError("current family must cover four sources")
    for source_sha in source_order:
        rows = [row for row in candidates if row["source_video_sha256"] == source_sha]
        if len(rows) != 2 or tuple(sorted(row["seed"] for row in rows)) != EXPECTED_SEEDS:
            raise PairV5SourceBoundEvaluationError("each source requires both registered seeds")
    return {
        "groups": groups,
        "candidates": candidates,
        "candidate_order": ids,
        "source_order": source_order,
    }


def load_current_family_rollout_spec(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    if expected_sha256 != CURRENT_FAMILY_ROLLOUT_SPEC_RAW_SHA256:
        raise PairV5SourceBoundEvaluationError(
            "expected rollout SHA-256 is not the registered current family"
        )
    value, digest = _strict_json(
        path, expected_sha256=expected_sha256, label="current-family rollout spec"
    )
    return _validate_current_family_rollout_spec(value), digest


def make_evaluator_spec(
    rollout_spec: Mapping[str, Any],
    *,
    rollout_spec_raw_sha256: str,
    implementation_sha256: str,
    contract_sha256: str,
    method_source_revision: str,
    method_source_archive_sha256: str,
    architecture_id: str,
    checkpoint_manifest_sha256: str,
    checkpoint_config_sha256: str,
    preprocessor_config_sha256: str,
    checkpoint_file_count: int,
    num_register_tokens: int,
    image_size: int,
    patch_size: int,
    preprocessor_golden_input_sha256: str,
    preprocessor_golden_output_sha256: str,
    preprocessor_golden_output_shape: Sequence[int],
    generation_provenance: Mapping[str, Any],
    runtime_versions: Mapping[str, str],
) -> dict[str, Any]:
    if rollout_spec_raw_sha256 != CURRENT_FAMILY_ROLLOUT_SPEC_RAW_SHA256:
        raise PairV5SourceBoundEvaluationError(
            "evaluator authoring requires the registered current-family rollout"
        )
    normalized_rollout = _validate_current_family_rollout_spec(rollout_spec)
    source_order = normalized_rollout["source_order"]
    # Current-family source order is dog-fit, human-confirmation, dog-
    # confirmation, human-fit.  A half-cycle pairs dog<->dog and human<->human,
    # making wrong-source identity/appearance evidence materially harder than
    # a trivial cross-species negative.
    wrong = {
        source: source_order[(index + 2) % len(source_order)]
        for index, source in enumerate(source_order)
    }
    model = {
        "adapter_id": MODEL_ADAPTER_ID,
        "architecture_id": architecture_id,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_config_sha256": checkpoint_config_sha256,
        "preprocessor_config_sha256": preprocessor_config_sha256,
        "checkpoint_file_count": checkpoint_file_count,
        "num_register_tokens": num_register_tokens,
        "image_size": image_size,
        "patch_size": patch_size,
        "preprocessor_golden_input_sha256": preprocessor_golden_input_sha256,
        "preprocessor_golden_output_sha256": preprocessor_golden_output_sha256,
        "preprocessor_golden_output_shape": list(preprocessor_golden_output_shape),
    }
    unsigned = {
        "schema_version": SPEC_SCHEMA,
        "rollout_spec_raw_sha256": rollout_spec_raw_sha256,
        "candidate_order": normalized_rollout["candidate_order"],
        "candidate_group_by_id": {
            row["candidate_id"]: row["group_id"]
            for row in normalized_rollout["candidates"]
        },
        "correct_source_sha256_by_candidate_id": {
            row["candidate_id"]: row["source_video_sha256"]
            for row in normalized_rollout["candidates"]
        },
        "source_order": source_order,
        "wrong_source_by_source_sha256": wrong,
        "implementation_sha256": implementation_sha256,
        "contract_sha256": contract_sha256,
        "method_source_revision": method_source_revision,
        "method_source_archive_sha256": method_source_archive_sha256,
        "model": model,
        "generation_provenance": dict(generation_provenance),
        "runtime_versions": dict(runtime_versions),
        "preprocess_contract": PREPROCESS_CONTRACT,
        "metric_order": list(METRIC_ORDER),
        "metric_contract": METRIC_CONTRACT,
        "probe_order": list(PROBE_ORDER),
        "input_closure": INPUT_CLOSURE,
        "scientific_claims": SCIENTIFIC_CLAIMS,
    }
    return validate_evaluator_spec({**unsigned, "spec_digest": object_sha256(unsigned)})


def validate_evaluator_spec(
    value: Any, *, normalized_rollout: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    row = _closed(value, _SPEC_FIELDS, label="evaluator spec")
    if row["schema_version"] != SPEC_SCHEMA:
        raise PairV5SourceBoundEvaluationError("evaluator spec schema differs")
    rollout_sha = _sha256(row["rollout_spec_raw_sha256"], label="rollout spec SHA-256")
    if rollout_sha != CURRENT_FAMILY_ROLLOUT_SPEC_RAW_SHA256:
        raise PairV5SourceBoundEvaluationError("evaluator rollout family differs")
    implementation_sha = _sha256(
        row["implementation_sha256"], label="implementation SHA-256"
    )
    contract_sha = _sha256(row["contract_sha256"], label="contract SHA-256")
    method_revision = row["method_source_revision"]
    if not isinstance(method_revision, str) or _SHA1.fullmatch(method_revision) is None:
        raise PairV5SourceBoundEvaluationError("method source revision must be full SHA-1")
    method_archive_sha = _sha256(
        row["method_source_archive_sha256"], label="method source archive SHA-256"
    )
    candidate_order = row["candidate_order"]
    if (
        not isinstance(candidate_order, list)
        or len(candidate_order) != EXPECTED_CANDIDATE_COUNT
        or len(set(candidate_order)) != len(candidate_order)
    ):
        raise PairV5SourceBoundEvaluationError("evaluator candidate order differs")
    for candidate_id in candidate_order:
        _safe_id(candidate_id, label="evaluator candidate ID")
    group_by_id = row["candidate_group_by_id"]
    if not isinstance(group_by_id, Mapping) or set(group_by_id) != set(candidate_order):
        raise PairV5SourceBoundEvaluationError("candidate group closure differs")
    if any(group_by_id[candidate] not in EXPECTED_GROUPS for candidate in candidate_order):
        raise PairV5SourceBoundEvaluationError("candidate group value differs")
    correct_source_by_id = row["correct_source_sha256_by_candidate_id"]
    if (
        not isinstance(correct_source_by_id, Mapping)
        or set(correct_source_by_id) != set(candidate_order)
    ):
        raise PairV5SourceBoundEvaluationError("candidate correct-source closure differs")
    for candidate in candidate_order:
        _sha256(
            correct_source_by_id[candidate],
            label=f"{candidate} correct source SHA-256",
        )
    source_order = row["source_order"]
    if (
        not isinstance(source_order, list)
        or len(source_order) != EXPECTED_SOURCE_COUNT
        or len(set(source_order)) != len(source_order)
    ):
        raise PairV5SourceBoundEvaluationError("source order differs")
    for source in source_order:
        _sha256(source, label="source order SHA-256")
    if any(correct_source_by_id[candidate] not in source_order for candidate in candidate_order):
        raise PairV5SourceBoundEvaluationError("candidate source is outside source order")
    if any(
        sum(correct_source_by_id[candidate] == source for candidate in candidate_order) != 2
        for source in source_order
    ):
        raise PairV5SourceBoundEvaluationError("each source must bind exactly two candidates")
    wrong = row["wrong_source_by_source_sha256"]
    expected_wrong = {
        source: source_order[(index + 2) % len(source_order)]
        for index, source in enumerate(source_order)
    }
    if wrong != expected_wrong or any(source == target for source, target in wrong.items()):
        raise PairV5SourceBoundEvaluationError("wrong-source permutation differs")
    model = _closed(row["model"], _MODEL_FIELDS, label="model binding")
    if model["adapter_id"] != MODEL_ADAPTER_ID:
        raise PairV5SourceBoundEvaluationError("model adapter differs")
    architecture_id = _safe_id(model["architecture_id"], label="architecture_id")
    if architecture_id not in SUPPORTED_ARCHITECTURES:
        raise PairV5SourceBoundEvaluationError(
            "v1 supports only frozen DINOv2 visual checkpoints"
        )
    checked_model = {
        "adapter_id": MODEL_ADAPTER_ID,
        "architecture_id": architecture_id,
        "checkpoint_manifest_sha256": _sha256(
            model["checkpoint_manifest_sha256"], label="checkpoint manifest SHA-256"
        ),
        "checkpoint_config_sha256": _sha256(
            model["checkpoint_config_sha256"], label="checkpoint config SHA-256"
        ),
        "preprocessor_config_sha256": _sha256(
            model["preprocessor_config_sha256"], label="preprocessor config SHA-256"
        ),
        "checkpoint_file_count": _integer(
            model["checkpoint_file_count"], label="checkpoint file count", minimum=1
        ),
        "num_register_tokens": _integer(
            model["num_register_tokens"], label="num register tokens", minimum=0
        ),
        "image_size": _integer(model["image_size"], label="model image size", minimum=1),
        "patch_size": _integer(model["patch_size"], label="model patch size", minimum=1),
        "preprocessor_golden_input_sha256": _sha256(
            model["preprocessor_golden_input_sha256"], label="processor golden input SHA-256"
        ),
        "preprocessor_golden_output_sha256": _sha256(
            model["preprocessor_golden_output_sha256"], label="processor golden output SHA-256"
        ),
        "preprocessor_golden_output_shape": model["preprocessor_golden_output_shape"],
    }
    if (
        checked_model["image_size"] != MODEL_NATIVE_IMAGE_SIZE
        or checked_model["patch_size"] != MODEL_PATCH_SIZE
        or checked_model["num_register_tokens"] != 0
        or checked_model["preprocessor_golden_output_shape"]
        != [1, 3, EVALUATION_IMAGE_SIZE, EVALUATION_IMAGE_SIZE]
    ):
        raise PairV5SourceBoundEvaluationError("sealed model/processor geometry differs")
    generation = validate_generation_provenance(row["generation_provenance"])
    versions = _closed(row["runtime_versions"], _RUNTIME_FIELDS, label="runtime versions")
    checked_versions: dict[str, str] = {}
    for key in _RUNTIME_FIELDS:
        version = versions[key]
        if not isinstance(version, str) or not version or "\x00" in version or len(version) > 128:
            raise PairV5SourceBoundEvaluationError(f"runtime version {key} differs")
        checked_versions[key] = version
    if checked_versions["transformers_version"] != "4.53.2":
        raise PairV5SourceBoundEvaluationError("v1 requires Transformers 4.53.2 exactly")
    if row["preprocess_contract"] != PREPROCESS_CONTRACT:
        raise PairV5SourceBoundEvaluationError("preprocess contract differs")
    if row["metric_order"] != list(METRIC_ORDER) or row["metric_contract"] != METRIC_CONTRACT:
        raise PairV5SourceBoundEvaluationError("metric contract/order differs")
    if row["probe_order"] != list(PROBE_ORDER):
        raise PairV5SourceBoundEvaluationError("probe order differs")
    if row["input_closure"] != INPUT_CLOSURE or row["scientific_claims"] != SCIENTIFIC_CLAIMS:
        raise PairV5SourceBoundEvaluationError("input/claim closure differs")
    if normalized_rollout is not None:
        if candidate_order != normalized_rollout["candidate_order"]:
            raise PairV5SourceBoundEvaluationError("evaluator/rollout candidate order differs")
        expected_groups = {
            item["candidate_id"]: item["group_id"]
            for item in normalized_rollout["candidates"]
        }
        if dict(group_by_id) != expected_groups:
            raise PairV5SourceBoundEvaluationError("evaluator/rollout groups differ")
        expected_sources = {
            item["candidate_id"]: item["source_video_sha256"]
            for item in normalized_rollout["candidates"]
        }
        if dict(correct_source_by_id) != expected_sources:
            raise PairV5SourceBoundEvaluationError(
                "evaluator/rollout candidate source binding differs"
            )
        if source_order != normalized_rollout["source_order"]:
            raise PairV5SourceBoundEvaluationError("evaluator/rollout source order differs")
    normalized = {
        "schema_version": SPEC_SCHEMA,
        "rollout_spec_raw_sha256": rollout_sha,
        "candidate_order": list(candidate_order),
        "candidate_group_by_id": dict(group_by_id),
        "correct_source_sha256_by_candidate_id": dict(correct_source_by_id),
        "source_order": list(source_order),
        "wrong_source_by_source_sha256": dict(wrong),
        "implementation_sha256": implementation_sha,
        "contract_sha256": contract_sha,
        "method_source_revision": method_revision,
        "method_source_archive_sha256": method_archive_sha,
        "model": checked_model,
        "generation_provenance": generation,
        "runtime_versions": checked_versions,
        "preprocess_contract": PREPROCESS_CONTRACT,
        "metric_order": list(METRIC_ORDER),
        "metric_contract": METRIC_CONTRACT,
        "probe_order": list(PROBE_ORDER),
        "input_closure": INPUT_CLOSURE,
        "scientific_claims": SCIENTIFIC_CLAIMS,
        "spec_digest": _sha256(row["spec_digest"], label="spec digest"),
    }
    unsigned = dict(normalized)
    declared = unsigned.pop("spec_digest")
    if object_sha256(unsigned) != declared:
        raise PairV5SourceBoundEvaluationError("evaluator embedded spec digest differs")
    return normalized


def load_evaluator_spec(
    path: str | Path,
    expected_sha256: str,
    *,
    normalized_rollout: Mapping[str, Any],
    rollout_spec_raw_sha256: str,
    implementation_path: str | Path,
    contract_path: str | Path,
) -> tuple[dict[str, Any], str]:
    value, raw_sha = _strict_json(path, expected_sha256=expected_sha256, label="evaluator spec")
    spec = validate_evaluator_spec(value, normalized_rollout=normalized_rollout)
    if spec["rollout_spec_raw_sha256"] != _sha256(
        rollout_spec_raw_sha256, label="rollout spec raw SHA-256"
    ):
        raise PairV5SourceBoundEvaluationError("evaluator rollout binding differs")
    implementation = Path(implementation_path).resolve(strict=True)
    if file_sha256(implementation) != spec["implementation_sha256"]:
        raise PairV5SourceBoundEvaluationError("evaluator implementation hash differs")
    contract_file = Path(contract_path).resolve(strict=True)
    if file_sha256(contract_file) != spec["contract_sha256"]:
        raise PairV5SourceBoundEvaluationError("evaluator contract hash differs")
    return spec, raw_sha


def _validate_decode_evidence(value: Any, *, role: str, artifact_sha256: str) -> dict[str, Any]:
    row = _closed(value, _DECODE_FIELDS, label=f"{role} decode evidence")
    if _sha256(row["artifact_sha256"], label=f"{role} artifact SHA-256") != artifact_sha256:
        raise PairV5SourceBoundEvaluationError(f"{role} artifact/decode binding differs")
    result = {
        "artifact_sha256": artifact_sha256,
        "decoded_rgb_sha256": _sha256(
            row["decoded_rgb_sha256"], label=f"{role} decoded RGB SHA-256"
        ),
        "frame_count": _integer(row["frame_count"], label=f"{role} frame count", minimum=1),
        "fps_numerator": _integer(row["fps_numerator"], label=f"{role} FPS numerator", minimum=1),
        "fps_denominator": _integer(row["fps_denominator"], label=f"{role} FPS denominator", minimum=1),
        "time_base_numerator": _integer(
            row["time_base_numerator"], label=f"{role} time-base numerator", minimum=1
        ),
        "time_base_denominator": _integer(
            row["time_base_denominator"], label=f"{role} time-base denominator", minimum=1
        ),
        "pts_step": _integer(row["pts_step"], label=f"{role} PTS step", minimum=1),
        "pts_sha256": _sha256(row["pts_sha256"], label=f"{role} PTS SHA-256"),
        "width": _integer(row["width"], label=f"{role} width", minimum=1),
        "height": _integer(row["height"], label=f"{role} height", minimum=1),
        "selected_frame_indices": row["selected_frame_indices"],
        "selected_rgb_sha256": _sha256(
            row["selected_rgb_sha256"], label=f"{role} selected RGB SHA-256"
        ),
        "preprocessed_tensor_sha256": _sha256(
            row["preprocessed_tensor_sha256"], label=f"{role} preprocessed SHA-256"
        ),
    }
    if (
        result["frame_count"] != FRAME_COUNT
        or (result["fps_numerator"], result["fps_denominator"]) != (FPS, 1)
        or result["selected_frame_indices"] != list(EVAL_FRAME_INDICES)
    ):
        raise PairV5SourceBoundEvaluationError(f"{role} decode geometry differs")
    if (
        result["time_base_numerator"] * result["pts_step"] * FPS
        != result["time_base_denominator"]
    ):
        raise PairV5SourceBoundEvaluationError(f"{role} PTS cadence/FPS differs")
    return result


def _validate_feature_evidence(value: Any, *, role: str) -> dict[str, Any]:
    row = _closed(value, _FEATURE_FIELDS, label=f"{role} feature evidence")
    result = {
        "global_feature_sha256": _sha256(
            row["global_feature_sha256"], label=f"{role} global feature SHA-256"
        ),
        "dense_feature_sha256": _sha256(
            row["dense_feature_sha256"], label=f"{role} dense feature SHA-256"
        ),
        "selected_frame_count": _integer(
            row["selected_frame_count"], label=f"{role} selected frame count", minimum=1
        ),
        "dense_grid_height": _integer(
            row["dense_grid_height"], label=f"{role} dense grid height", minimum=1
        ),
        "dense_grid_width": _integer(
            row["dense_grid_width"], label=f"{role} dense grid width", minimum=1
        ),
        "feature_dimension": _integer(
            row["feature_dimension"], label=f"{role} feature dimension", minimum=1
        ),
    }
    if result["selected_frame_count"] != len(EVAL_FRAME_INDICES):
        raise PairV5SourceBoundEvaluationError(f"{role} feature frame count differs")
    return result


def _validate_model_evidence(value: Any, *, spec: Mapping[str, Any]) -> dict[str, Any]:
    row = _closed(value, _MODEL_EVIDENCE_FIELDS, label="model evidence")
    model = spec["model"]
    checked = {
        "adapter_id": row["adapter_id"],
        "architecture_id": row["architecture_id"],
        "checkpoint_manifest_sha256": _sha256(
            row["checkpoint_manifest_sha256"], label="observed checkpoint manifest SHA-256"
        ),
        "checkpoint_config_sha256": _sha256(
            row["checkpoint_config_sha256"], label="observed checkpoint config SHA-256"
        ),
        "preprocessor_config_sha256": _sha256(
            row["preprocessor_config_sha256"], label="observed preprocessor config SHA-256"
        ),
        "checkpoint_file_count": _integer(
            row["checkpoint_file_count"], label="observed checkpoint file count", minimum=1
        ),
        "verified_entries_digest": _sha256(
            row["verified_entries_digest"], label="verified checkpoint entries digest"
        ),
        "preprocessor_golden_input_sha256": _sha256(
            row["preprocessor_golden_input_sha256"], label="observed processor golden input SHA-256"
        ),
        "preprocessor_golden_output_sha256": _sha256(
            row["preprocessor_golden_output_sha256"], label="observed processor golden SHA-256"
        ),
        "preprocessor_golden_output_shape": row["preprocessor_golden_output_shape"],
        "every_checkpoint_file_verified": _strict_bool(
            row["every_checkpoint_file_verified"], label="every checkpoint file verified"
        ),
        "all_parameters_frozen": _strict_bool(
            row["all_parameters_frozen"], label="all parameters frozen"
        ),
        "trainable_parameter_tensors": _integer(
            row["trainable_parameter_tensors"], label="trainable parameter tensors"
        ),
        "parameter_tensor_count": _integer(
            row["parameter_tensor_count"], label="parameter tensor count", minimum=1
        ),
        "parameter_element_count": _integer(
            row["parameter_element_count"], label="parameter element count", minimum=1
        ),
        "parameter_metadata_digest": _sha256(
            row["parameter_metadata_digest"], label="parameter metadata digest"
        ),
        "missing_key_count": _integer(
            row["missing_key_count"], label="missing key count"
        ),
        "unexpected_key_count": _integer(
            row["unexpected_key_count"], label="unexpected key count"
        ),
        "mismatched_key_count": _integer(
            row["mismatched_key_count"], label="mismatched key count"
        ),
        "loading_error_count": _integer(
            row["loading_error_count"], label="loading error count"
        ),
        "runtime_versions": dict(
            _closed(row["runtime_versions"], _RUNTIME_FIELDS, label="observed runtime versions")
        ),
    }
    if (
        checked["adapter_id"] != model["adapter_id"]
        or checked["architecture_id"] != model["architecture_id"]
        or checked["checkpoint_manifest_sha256"] != model["checkpoint_manifest_sha256"]
        or checked["checkpoint_config_sha256"] != model["checkpoint_config_sha256"]
        or checked["preprocessor_config_sha256"] != model["preprocessor_config_sha256"]
        or checked["checkpoint_file_count"] != model["checkpoint_file_count"]
        or checked["preprocessor_golden_input_sha256"]
        != model["preprocessor_golden_input_sha256"]
        or checked["preprocessor_golden_output_sha256"]
        != model["preprocessor_golden_output_sha256"]
        or checked["preprocessor_golden_output_shape"]
        != model["preprocessor_golden_output_shape"]
        or checked["runtime_versions"] != spec["runtime_versions"]
        or not checked["every_checkpoint_file_verified"]
        or not checked["all_parameters_frozen"]
        or checked["trainable_parameter_tensors"] != 0
        or checked["missing_key_count"] != 0
        or checked["unexpected_key_count"] != 0
        or checked["mismatched_key_count"] != 0
        or checked["loading_error_count"] != 0
    ):
        raise PairV5SourceBoundEvaluationError("model evidence/spec binding differs")
    return checked


def make_candidate_receipt(
    *,
    evaluator_spec: Mapping[str, Any],
    evaluator_spec_raw_sha256: str,
    candidate_id: str,
    candidate_ordinal: int,
    group_id: str,
    candidate_envelope_sha256: str,
    rollout_receipt_digest: str,
    rollout_receipt_file_sha256: str,
    native_rollout_receipt_digest: str,
    native_rollout_receipt_file_sha256: str,
    native_generation_provenance_digest: str,
    candidate_mp4_sha256: str,
    predecode_clean_latent_sha256: str,
    official_initial_gaussian_sha256: str,
    correct_source_video_sha256: str,
    wrong_source_video_sha256: str,
    decode_evidence_by_role: Mapping[str, Mapping[str, Any]],
    feature_evidence_by_role: Mapping[str, Mapping[str, Any]],
    model_evidence: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    spec = validate_evaluator_spec(evaluator_spec)
    correct = float(metrics["source_identity_appearance_proxy"])
    wrong = float(metrics["source_identity_appearance_wrong_source_proxy"])
    upper = float(metrics["source_identity_appearance_source_self_upper_bound"])
    denominator = upper - wrong
    calibrated_margin = (correct - wrong) / denominator if denominator > 0.0 else 0.0
    probes = {
        "order": list(PROBE_ORDER),
        "correct_source_sha256": correct_source_video_sha256,
        "wrong_source_sha256": wrong_source_video_sha256,
        "source_disjoint": correct_source_video_sha256 != wrong_source_video_sha256,
        "correct_source_global_similarity": correct,
        "wrong_source_global_similarity": wrong,
        "correct_minus_wrong_margin": correct - wrong,
        "source_self_similarity_upper_bound": upper,
        "upper_bound_minus_correct_headroom": upper - correct,
        "upper_bound_minus_wrong_denominator": denominator,
        "wrong_normalized_contrast": calibrated_margin,
        "contrast_denominator_positive": denominator > 0.0,
        "reference_off_applicable": False,
        "reference_off_reason": (
            "not_applicable_post_video_scorer_has_no_conditional_reference_branch"
        ),
        "strict_correct_greater_than_wrong": correct > wrong,
        "upper_bound_not_below_correct": upper >= correct,
        "diagnostic_ordering_holds": correct > wrong and upper >= correct and denominator > 0.0,
    }
    unsigned = {
        "schema_version": RECEIPT_SCHEMA,
        "evaluator_spec_digest": spec["spec_digest"],
        "evaluator_spec_raw_sha256": evaluator_spec_raw_sha256,
        "rollout_spec_raw_sha256": spec["rollout_spec_raw_sha256"],
        "candidate_id": candidate_id,
        "candidate_ordinal": candidate_ordinal,
        "group_id": group_id,
        "candidate_order_digest": object_sha256(spec["candidate_order"]),
        "candidate_envelope_sha256": candidate_envelope_sha256,
        "rollout_receipt_digest": rollout_receipt_digest,
        "rollout_receipt_file_sha256": rollout_receipt_file_sha256,
        "native_rollout_receipt_digest": native_rollout_receipt_digest,
        "native_rollout_receipt_file_sha256": native_rollout_receipt_file_sha256,
        "native_generation_provenance_digest": native_generation_provenance_digest,
        "candidate_mp4_sha256": candidate_mp4_sha256,
        "predecode_clean_latent_sha256": predecode_clean_latent_sha256,
        "official_initial_gaussian_sha256": official_initial_gaussian_sha256,
        "correct_source_video_sha256": correct_source_video_sha256,
        "wrong_source_video_sha256": wrong_source_video_sha256,
        "decode_evidence_by_role": dict(decode_evidence_by_role),
        "feature_evidence_by_role": dict(feature_evidence_by_role),
        "model_evidence": dict(model_evidence),
        "metrics": dict(metrics),
        "binding_probes": probes,
        "metric_order": list(METRIC_ORDER),
        "input_closure": INPUT_CLOSURE,
        "scientific_claims": SCIENTIFIC_CLAIMS,
        "evidence_valid": True,
        "eligible_for_downstream_calibration": True,
        "absolute_source_preservation_pass_claim": False,
    }
    return validate_candidate_receipt(
        {**unsigned, "receipt_digest": object_sha256(unsigned)},
        evaluator_spec=spec,
        evaluator_spec_raw_sha256=evaluator_spec_raw_sha256,
    )


def validate_candidate_receipt(
    value: Any,
    *,
    evaluator_spec: Mapping[str, Any],
    evaluator_spec_raw_sha256: str | None = None,
) -> dict[str, Any]:
    spec = validate_evaluator_spec(evaluator_spec)
    row = _closed(value, _RECEIPT_FIELDS, label="candidate evidence receipt")
    if row["schema_version"] != RECEIPT_SCHEMA:
        raise PairV5SourceBoundEvaluationError("candidate receipt schema differs")
    candidate_id = _safe_id(row["candidate_id"], label="receipt candidate_id")
    ordinal = _integer(row["candidate_ordinal"], label="candidate ordinal")
    if ordinal >= len(spec["candidate_order"]) or spec["candidate_order"][ordinal] != candidate_id:
        raise PairV5SourceBoundEvaluationError("candidate order binding differs")
    group_id = row["group_id"]
    if group_id != spec["candidate_group_by_id"][candidate_id]:
        raise PairV5SourceBoundEvaluationError("candidate group binding differs")
    correct_sha = _sha256(
        row["correct_source_video_sha256"], label="correct source video SHA-256"
    )
    wrong_sha = _sha256(
        row["wrong_source_video_sha256"], label="wrong source video SHA-256"
    )
    if spec["wrong_source_by_source_sha256"].get(correct_sha) != wrong_sha:
        raise PairV5SourceBoundEvaluationError("wrong source mapping differs")
    if spec["correct_source_sha256_by_candidate_id"].get(candidate_id) != correct_sha:
        raise PairV5SourceBoundEvaluationError("candidate correct-source binding differs")
    candidate_mp4_sha = _sha256(row["candidate_mp4_sha256"], label="candidate MP4 SHA-256")
    roles = ("candidate", "correct_source", "wrong_source")
    decode = row["decode_evidence_by_role"]
    features = row["feature_evidence_by_role"]
    if not isinstance(decode, Mapping) or tuple(decode) != roles:
        raise PairV5SourceBoundEvaluationError("decode role order/closure differs")
    if not isinstance(features, Mapping) or tuple(features) != roles:
        raise PairV5SourceBoundEvaluationError("feature role order/closure differs")
    expected_artifacts = {
        "candidate": candidate_mp4_sha,
        "correct_source": correct_sha,
        "wrong_source": wrong_sha,
    }
    checked_decode = {
        role: _validate_decode_evidence(
            decode[role], role=role, artifact_sha256=expected_artifacts[role]
        )
        for role in roles
    }
    checked_features = {
        role: _validate_feature_evidence(features[role], role=role) for role in roles
    }
    shapes = {
        (
            item["selected_frame_count"],
            item["dense_grid_height"],
            item["dense_grid_width"],
            item["feature_dimension"],
        )
        for item in checked_features.values()
    }
    if len(shapes) != 1:
        raise PairV5SourceBoundEvaluationError("feature geometry differs across roles")
    metrics = _closed(row["metrics"], _METRIC_FIELDS, label="metric evidence")
    checked_metrics = {key: _finite(metrics[key], label=f"metric {key}") for key in _METRIC_FIELDS}
    unit_metrics = set(_METRIC_FIELDS) - {
        "source_identity_appearance_correct_minus_wrong_margin",
        "source_identity_appearance_upper_bound_minus_correct_headroom",
        "source_identity_appearance_wrong_normalized_contrast",
        "background_appearance_correct_minus_wrong_margin",
        "source_bound_spatial_layout_correct_minus_wrong_margin",
        "source_bound_spatial_layout_wrong_normalized_contrast_proxy",
    }
    for key in unit_metrics:
        _unit(checked_metrics[key], label=f"metric {key}")
    expected_formulas = {
        "source_identity_appearance_correct_minus_wrong_margin": (
            checked_metrics["source_identity_appearance_proxy"]
            - checked_metrics["source_identity_appearance_wrong_source_proxy"]
        ),
        "source_identity_appearance_upper_bound_minus_correct_headroom": (
            checked_metrics["source_identity_appearance_source_self_upper_bound"]
            - checked_metrics["source_identity_appearance_proxy"]
        ),
        "background_appearance_correct_minus_wrong_margin": (
            checked_metrics["background_appearance_fixed_grid_proxy"]
            - checked_metrics["background_appearance_wrong_source_fixed_grid_proxy"]
        ),
        "source_bound_spatial_layout_correct_minus_wrong_margin": (
            checked_metrics["source_bound_spatial_layout_viewpoint_proxy"]
            - checked_metrics["source_bound_spatial_layout_wrong_source_proxy"]
        ),
    }
    upper_denominator = (
        checked_metrics["source_identity_appearance_source_self_upper_bound"]
        - checked_metrics["source_identity_appearance_wrong_source_proxy"]
    )
    expected_formulas["source_identity_appearance_wrong_normalized_contrast"] = (
        (
            checked_metrics["source_identity_appearance_proxy"]
            - checked_metrics["source_identity_appearance_wrong_source_proxy"]
        )
        / upper_denominator
        if upper_denominator > 0.0
        else 0.0
    )
    layout_upper = 1.0
    layout_denominator = (
        layout_upper - checked_metrics["source_bound_spatial_layout_wrong_source_proxy"]
    )
    expected_formulas["source_bound_spatial_layout_wrong_normalized_contrast_proxy"] = (
        checked_metrics["source_bound_spatial_layout_correct_minus_wrong_margin"]
        / layout_denominator
        if layout_denominator > 0.0
        else 0.0
    )
    for key, expected in expected_formulas.items():
        if checked_metrics[key] != expected:
            raise PairV5SourceBoundEvaluationError(f"metric formula differs: {key}")
    probes = _closed(row["binding_probes"], _PROBE_FIELDS, label="binding probes")
    if probes["order"] != list(PROBE_ORDER):
        raise PairV5SourceBoundEvaluationError("binding probe order differs")
    if (
        probes["correct_source_sha256"] != correct_sha
        or probes["wrong_source_sha256"] != wrong_sha
        or probes["reference_off_applicable"] is not False
        or probes["reference_off_reason"]
        != "not_applicable_post_video_scorer_has_no_conditional_reference_branch"
    ):
        raise PairV5SourceBoundEvaluationError("binding probe provenance differs")
    correct = checked_metrics["source_identity_appearance_proxy"]
    wrong = checked_metrics["source_identity_appearance_wrong_source_proxy"]
    upper = checked_metrics["source_identity_appearance_source_self_upper_bound"]
    expected_probe = {
        "source_disjoint": correct_sha != wrong_sha,
        "correct_source_global_similarity": correct,
        "wrong_source_global_similarity": wrong,
        "correct_minus_wrong_margin": correct - wrong,
        "source_self_similarity_upper_bound": upper,
        "upper_bound_minus_correct_headroom": upper - correct,
        "upper_bound_minus_wrong_denominator": upper - wrong,
        "wrong_normalized_contrast": (
            (correct - wrong) / (upper - wrong) if upper > wrong else 0.0
        ),
        "contrast_denominator_positive": upper > wrong,
        "strict_correct_greater_than_wrong": correct > wrong,
        "upper_bound_not_below_correct": upper >= correct,
        "diagnostic_ordering_holds": correct > wrong and upper >= correct and upper > wrong,
    }
    for key, expected in expected_probe.items():
        actual = probes[key]
        if isinstance(expected, bool):
            _strict_bool(actual, label=f"probe {key}")
        else:
            actual = _finite(actual, label=f"probe {key}")
        if actual != expected:
            raise PairV5SourceBoundEvaluationError(f"binding probe formula differs: {key}")
    evidence_valid = _strict_bool(row["evidence_valid"], label="evidence_valid")
    eligible = _strict_bool(
        row["eligible_for_downstream_calibration"],
        label="eligible_for_downstream_calibration",
    )
    if not evidence_valid or not eligible:
        raise PairV5SourceBoundEvaluationError("valid evidence must remain calibration eligible")
    if _strict_bool(
        row["absolute_source_preservation_pass_claim"],
        label="absolute source preservation pass claim",
    ):
        raise PairV5SourceBoundEvaluationError("v1 cannot make an absolute pass claim")
    if (
        row["evaluator_spec_digest"] != spec["spec_digest"]
        or _sha256(row["rollout_spec_raw_sha256"], label="receipt rollout SHA-256")
        != spec["rollout_spec_raw_sha256"]
        or _sha256(row["candidate_order_digest"], label="candidate order digest")
        != object_sha256(spec["candidate_order"])
        or row["metric_order"] != list(METRIC_ORDER)
        or row["input_closure"] != INPUT_CLOSURE
        or row["scientific_claims"] != SCIENTIFIC_CLAIMS
    ):
        raise PairV5SourceBoundEvaluationError("candidate receipt/spec closure differs")
    normalized = dict(row)
    for key in (
        "evaluator_spec_raw_sha256",
        "candidate_envelope_sha256",
        "rollout_receipt_digest",
        "rollout_receipt_file_sha256",
        "native_rollout_receipt_digest",
        "native_rollout_receipt_file_sha256",
        "native_generation_provenance_digest",
        "predecode_clean_latent_sha256",
        "official_initial_gaussian_sha256",
        "receipt_digest",
    ):
        normalized[key] = _sha256(row[key], label=key)
    if normalized["native_generation_provenance_digest"] != spec[
        "generation_provenance"
    ]["provenance_digest"]:
        raise PairV5SourceBoundEvaluationError("native generation provenance/spec binding differs")
    if (
        evaluator_spec_raw_sha256 is not None
        and normalized["evaluator_spec_raw_sha256"]
        != _sha256(
            evaluator_spec_raw_sha256, label="expected evaluator spec raw SHA-256"
        )
    ):
        raise PairV5SourceBoundEvaluationError("candidate evaluator-spec raw binding differs")
    normalized["decode_evidence_by_role"] = checked_decode
    normalized["feature_evidence_by_role"] = checked_features
    normalized["model_evidence"] = _validate_model_evidence(row["model_evidence"], spec=spec)
    normalized["metrics"] = checked_metrics
    unsigned = dict(normalized)
    declared = unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        raise PairV5SourceBoundEvaluationError("candidate receipt digest differs")
    return normalized


def make_group_receipt(
    *,
    evaluator_spec: Mapping[str, Any],
    evaluator_spec_raw_sha256: str,
    group_id: str,
    candidate_receipts: Sequence[Mapping[str, Any]],
    candidate_receipt_file_sha256_by_id: Mapping[str, str],
) -> dict[str, Any]:
    spec = validate_evaluator_spec(evaluator_spec)
    if group_id not in EXPECTED_GROUPS:
        raise PairV5SourceBoundEvaluationError("group ID differs")
    order = [
        candidate
        for candidate in spec["candidate_order"]
        if spec["candidate_group_by_id"][candidate] == group_id
    ]
    checked = [
        validate_candidate_receipt(
            value,
            evaluator_spec=spec,
            evaluator_spec_raw_sha256=evaluator_spec_raw_sha256,
        )
        for value in candidate_receipts
    ]
    if [item["candidate_id"] for item in checked] != order:
        raise PairV5SourceBoundEvaluationError("group candidate receipt order differs")
    if set(candidate_receipt_file_sha256_by_id) != set(order):
        raise PairV5SourceBoundEvaluationError("group receipt file closure differs")
    file_hashes = {
        candidate: _sha256(
            candidate_receipt_file_sha256_by_id[candidate],
            label=f"{candidate} receipt file SHA-256",
        )
        for candidate in order
    }
    eligible_count = sum(
        bool(item["eligible_for_downstream_calibration"]) for item in checked
    )
    unsigned = {
        "schema_version": GROUP_SCHEMA,
        "evaluator_spec_digest": spec["spec_digest"],
        "evaluator_spec_raw_sha256": evaluator_spec_raw_sha256,
        "rollout_spec_raw_sha256": spec["rollout_spec_raw_sha256"],
        "group_id": group_id,
        "visible_gpus": EXPECTED_GROUP_GPUS[group_id],
        "candidate_order": order,
        "candidate_receipt_digest_by_id": {
            item["candidate_id"]: item["receipt_digest"] for item in checked
        },
        "candidate_receipt_file_sha256_by_id": file_hashes,
        "candidate_count": len(checked),
        "eligible_for_downstream_calibration_count": eligible_count,
        "all_evidence_valid": all(bool(item["evidence_valid"]) for item in checked),
        "all_candidates_eligible_for_downstream_calibration": eligible_count == len(checked),
    }
    return validate_group_receipt(
        {**unsigned, "group_digest": object_sha256(unsigned)},
        evaluator_spec=spec,
        evaluator_spec_raw_sha256=evaluator_spec_raw_sha256,
    )


def validate_group_receipt(
    value: Any,
    *,
    evaluator_spec: Mapping[str, Any],
    evaluator_spec_raw_sha256: str | None = None,
) -> dict[str, Any]:
    spec = validate_evaluator_spec(evaluator_spec)
    row = _closed(value, _GROUP_FIELDS, label="group receipt")
    if row["schema_version"] != GROUP_SCHEMA or row["group_id"] not in EXPECTED_GROUPS:
        raise PairV5SourceBoundEvaluationError("group receipt schema/group differs")
    group_id = row["group_id"]
    expected_order = [
        candidate
        for candidate in spec["candidate_order"]
        if spec["candidate_group_by_id"][candidate] == group_id
    ]
    if row["candidate_order"] != expected_order or row["visible_gpus"] != EXPECTED_GROUP_GPUS[group_id]:
        raise PairV5SourceBoundEvaluationError("group topology/order differs")
    digest_map = row["candidate_receipt_digest_by_id"]
    file_map = row["candidate_receipt_file_sha256_by_id"]
    if not isinstance(digest_map, Mapping) or tuple(digest_map) != tuple(expected_order):
        raise PairV5SourceBoundEvaluationError("group candidate digest order differs")
    if not isinstance(file_map, Mapping) or tuple(file_map) != tuple(expected_order):
        raise PairV5SourceBoundEvaluationError("group candidate file order differs")
    for candidate in expected_order:
        _sha256(digest_map[candidate], label=f"{candidate} receipt digest")
        _sha256(file_map[candidate], label=f"{candidate} receipt file SHA-256")
    count = _integer(row["candidate_count"], label="group candidate count")
    eligible = _integer(
        row["eligible_for_downstream_calibration_count"], label="group eligible count"
    )
    if count != len(expected_order) or eligible != count:
        raise PairV5SourceBoundEvaluationError("group counts differ")
    all_valid = _strict_bool(row["all_evidence_valid"], label="all evidence valid")
    all_eligible = _strict_bool(
        row["all_candidates_eligible_for_downstream_calibration"],
        label="all candidates eligible",
    )
    if not all_valid or not all_eligible:
        raise PairV5SourceBoundEvaluationError("group aggregate flags differ")
    if (
        row["evaluator_spec_digest"] != spec["spec_digest"]
        or row["rollout_spec_raw_sha256"] != spec["rollout_spec_raw_sha256"]
    ):
        raise PairV5SourceBoundEvaluationError("group spec binding differs")
    normalized = dict(row)
    normalized["evaluator_spec_raw_sha256"] = _sha256(
        row["evaluator_spec_raw_sha256"], label="group evaluator spec raw SHA-256"
    )
    if (
        evaluator_spec_raw_sha256 is not None
        and normalized["evaluator_spec_raw_sha256"]
        != _sha256(
            evaluator_spec_raw_sha256, label="expected evaluator spec raw SHA-256"
        )
    ):
        raise PairV5SourceBoundEvaluationError("group evaluator-spec raw binding differs")
    normalized["group_digest"] = _sha256(row["group_digest"], label="group digest")
    unsigned = dict(normalized)
    declared = unsigned.pop("group_digest")
    if object_sha256(unsigned) != declared:
        raise PairV5SourceBoundEvaluationError("group receipt digest differs")
    return normalized


def make_root_receipt(
    *,
    evaluator_spec: Mapping[str, Any],
    evaluator_spec_raw_sha256: str,
    group_receipts: Sequence[Mapping[str, Any]],
    group_receipt_file_sha256_by_id: Mapping[str, str],
    candidate_receipts: Sequence[Mapping[str, Any]],
    candidate_receipt_file_sha256_by_id: Mapping[str, str],
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the durable, path-free two-group/eight-candidate completion fact."""

    spec = validate_evaluator_spec(evaluator_spec)
    checked_groups = [
        validate_group_receipt(
            item,
            evaluator_spec=spec,
            evaluator_spec_raw_sha256=evaluator_spec_raw_sha256,
        )
        for item in group_receipts
    ]
    if [item["group_id"] for item in checked_groups] != list(EXPECTED_GROUPS):
        raise PairV5SourceBoundEvaluationError("root group order differs")
    checked_candidates = [
        validate_candidate_receipt(
            item,
            evaluator_spec=spec,
            evaluator_spec_raw_sha256=evaluator_spec_raw_sha256,
        )
        for item in candidate_receipts
    ]
    if [item["candidate_id"] for item in checked_candidates] != spec["candidate_order"]:
        raise PairV5SourceBoundEvaluationError("root candidate order differs")
    if set(group_receipt_file_sha256_by_id) != set(EXPECTED_GROUPS):
        raise PairV5SourceBoundEvaluationError("root group file closure differs")
    if set(candidate_receipt_file_sha256_by_id) != set(spec["candidate_order"]):
        raise PairV5SourceBoundEvaluationError("root candidate file closure differs")
    expected_topology = {
        "group_world_size": 4,
        "group_ulysses_size": 4,
        "groups": {key: EXPECTED_GROUP_GPUS[key] for key in EXPECTED_GROUPS},
        "total_physical_gpus": 8,
        "concurrent_disjoint_groups": True,
    }
    if topology != expected_topology:
        raise PairV5SourceBoundEvaluationError("root runtime topology differs")
    unsigned = {
        "schema_version": ROOT_SCHEMA,
        "evaluator_spec_digest": spec["spec_digest"],
        "evaluator_spec_raw_sha256": _sha256(
            evaluator_spec_raw_sha256, label="root evaluator spec raw SHA-256"
        ),
        "rollout_spec_raw_sha256": spec["rollout_spec_raw_sha256"],
        "method_source_revision": spec["method_source_revision"],
        "method_source_archive_sha256": spec["method_source_archive_sha256"],
        "model_checkpoint_manifest_sha256": spec["model"]["checkpoint_manifest_sha256"],
        "model_checkpoint_config_sha256": spec["model"]["checkpoint_config_sha256"],
        "model_preprocessor_config_sha256": spec["model"]["preprocessor_config_sha256"],
        "generation_checkpoint_tree_sha256": spec["generation_provenance"]["checkpoint_tree_sha256"],
        "generation_checkpoint_manifest_sha256": spec["generation_provenance"]["checkpoint_manifest_sha256"],
        "generation_provenance_digest": spec["generation_provenance"]["provenance_digest"],
        "runtime_versions": spec["runtime_versions"],
        "topology": dict(topology),
        "group_order": list(EXPECTED_GROUPS),
        "group_receipt_digest_by_id": {
            item["group_id"]: item["group_digest"] for item in checked_groups
        },
        "group_receipt_file_sha256_by_id": {
            key: _sha256(group_receipt_file_sha256_by_id[key], label=f"{key} group file SHA-256")
            for key in EXPECTED_GROUPS
        },
        "candidate_order": list(spec["candidate_order"]),
        "candidate_receipt_digest_by_id": {
            item["candidate_id"]: item["receipt_digest"]
            for item in sorted(checked_candidates, key=lambda row: row["candidate_id"])
        },
        "candidate_receipt_file_sha256_by_id": {
            key: _sha256(candidate_receipt_file_sha256_by_id[key], label=f"{key} candidate file SHA-256")
            for key in sorted(spec["candidate_order"])
        },
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "eligible_for_downstream_calibration_count": EXPECTED_CANDIDATE_COUNT,
        "all_evidence_valid": True,
        "complete": True,
        "exploratory_dev_only": True,
        "action_score_dependency": False,
        "absolute_source_preservation_pass_claims": 0,
    }
    return validate_root_receipt(
        {**unsigned, "root_digest": object_sha256(unsigned)},
        evaluator_spec=spec,
        evaluator_spec_raw_sha256=evaluator_spec_raw_sha256,
    )


def validate_root_receipt(
    value: Any,
    *,
    evaluator_spec: Mapping[str, Any],
    evaluator_spec_raw_sha256: str | None = None,
) -> dict[str, Any]:
    spec = validate_evaluator_spec(evaluator_spec)
    row = _closed(value, _ROOT_FIELDS, label="root completion receipt")
    expected_topology = {
        "group_world_size": 4,
        "group_ulysses_size": 4,
        "groups": {key: EXPECTED_GROUP_GPUS[key] for key in EXPECTED_GROUPS},
        "total_physical_gpus": 8,
        "concurrent_disjoint_groups": True,
    }
    expected_static = {
        "schema_version": ROOT_SCHEMA,
        "evaluator_spec_digest": spec["spec_digest"],
        "rollout_spec_raw_sha256": spec["rollout_spec_raw_sha256"],
        "method_source_revision": spec["method_source_revision"],
        "method_source_archive_sha256": spec["method_source_archive_sha256"],
        "model_checkpoint_manifest_sha256": spec["model"]["checkpoint_manifest_sha256"],
        "model_checkpoint_config_sha256": spec["model"]["checkpoint_config_sha256"],
        "model_preprocessor_config_sha256": spec["model"]["preprocessor_config_sha256"],
        "generation_checkpoint_tree_sha256": spec["generation_provenance"]["checkpoint_tree_sha256"],
        "generation_checkpoint_manifest_sha256": spec["generation_provenance"]["checkpoint_manifest_sha256"],
        "generation_provenance_digest": spec["generation_provenance"]["provenance_digest"],
        "runtime_versions": spec["runtime_versions"],
        "topology": expected_topology,
        "group_order": list(EXPECTED_GROUPS),
        "candidate_order": list(spec["candidate_order"]),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "eligible_for_downstream_calibration_count": EXPECTED_CANDIDATE_COUNT,
        "all_evidence_valid": True,
        "complete": True,
        "exploratory_dev_only": True,
        "action_score_dependency": False,
        "absolute_source_preservation_pass_claims": 0,
    }
    for key, expected in expected_static.items():
        if row[key] != expected:
            raise PairV5SourceBoundEvaluationError(f"root completion binding differs: {key}")
    for field, order in (
        ("group_receipt_digest_by_id", EXPECTED_GROUPS),
        ("group_receipt_file_sha256_by_id", EXPECTED_GROUPS),
        ("candidate_receipt_digest_by_id", sorted(spec["candidate_order"])),
        ("candidate_receipt_file_sha256_by_id", sorted(spec["candidate_order"])),
    ):
        mapping = row[field]
        if not isinstance(mapping, Mapping) or list(mapping) != list(order):
            raise PairV5SourceBoundEvaluationError(f"root map order differs: {field}")
        for key in order:
            _sha256(mapping[key], label=f"root {field} {key}")
    normalized = dict(row)
    normalized["evaluator_spec_raw_sha256"] = _sha256(
        row["evaluator_spec_raw_sha256"], label="root evaluator spec raw SHA-256"
    )
    if evaluator_spec_raw_sha256 is not None and normalized[
        "evaluator_spec_raw_sha256"
    ] != _sha256(evaluator_spec_raw_sha256, label="expected evaluator spec raw SHA-256"):
        raise PairV5SourceBoundEvaluationError("root evaluator-spec raw binding differs")
    normalized["root_digest"] = _sha256(row["root_digest"], label="root digest")
    unsigned = dict(normalized)
    declared = unsigned.pop("root_digest")
    if object_sha256(unsigned) != declared:
        raise PairV5SourceBoundEvaluationError("root completion digest differs")
    return normalized


def make_failure_receipt(
    *,
    evaluator_spec: Mapping[str, Any],
    evaluator_spec_raw_sha256: str,
    candidate_id: str,
    candidate_ordinal: int,
    group_id: str,
    failure_stage: str,
    error: Exception,
) -> dict[str, Any]:
    """Seal a path-free fail-closed row when candidate evidence cannot exist."""

    spec = validate_evaluator_spec(evaluator_spec)
    unsigned = {
        "schema_version": FAILURE_SCHEMA,
        "evaluator_spec_digest": spec["spec_digest"],
        "evaluator_spec_raw_sha256": evaluator_spec_raw_sha256,
        "rollout_spec_raw_sha256": spec["rollout_spec_raw_sha256"],
        "candidate_id": candidate_id,
        "candidate_ordinal": candidate_ordinal,
        "group_id": group_id,
        "failure_stage": failure_stage,
        "error_class": type(error).__name__,
        "error_message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        "evidence_valid": False,
        "eligible_for_downstream_calibration": False,
        "absolute_source_preservation_pass_claim": False,
    }
    return validate_failure_receipt(
        {**unsigned, "receipt_digest": object_sha256(unsigned)},
        evaluator_spec=spec,
        evaluator_spec_raw_sha256=evaluator_spec_raw_sha256,
    )


def validate_failure_receipt(
    value: Any,
    *,
    evaluator_spec: Mapping[str, Any],
    evaluator_spec_raw_sha256: str | None = None,
) -> dict[str, Any]:
    spec = validate_evaluator_spec(evaluator_spec)
    row = _closed(value, _FAILURE_FIELDS, label="failure receipt")
    if row["schema_version"] != FAILURE_SCHEMA:
        raise PairV5SourceBoundEvaluationError("failure receipt schema differs")
    candidate = _safe_id(row["candidate_id"], label="failure candidate ID")
    ordinal = _integer(row["candidate_ordinal"], label="failure candidate ordinal")
    if ordinal >= len(spec["candidate_order"]) or spec["candidate_order"][ordinal] != candidate:
        raise PairV5SourceBoundEvaluationError("failure candidate order differs")
    if row["group_id"] != spec["candidate_group_by_id"][candidate]:
        raise PairV5SourceBoundEvaluationError("failure candidate group differs")
    for key in ("failure_stage", "error_class"):
        value = row[key]
        if not isinstance(value, str) or not value or "\x00" in value or len(value) > 128:
            raise PairV5SourceBoundEvaluationError(f"failure {key} differs")
    if (
        row["evaluator_spec_digest"] != spec["spec_digest"]
        or row["rollout_spec_raw_sha256"] != spec["rollout_spec_raw_sha256"]
        or _strict_bool(row["evidence_valid"], label="failure evidence_valid")
        or _strict_bool(
            row["eligible_for_downstream_calibration"],
            label="failure eligibility",
        )
        or _strict_bool(
            row["absolute_source_preservation_pass_claim"],
            label="failure absolute pass claim",
        )
    ):
        raise PairV5SourceBoundEvaluationError("failure receipt is not fail closed")
    normalized = dict(row)
    for key in (
        "evaluator_spec_raw_sha256",
        "error_message_sha256",
        "receipt_digest",
    ):
        normalized[key] = _sha256(row[key], label=f"failure {key}")
    if (
        evaluator_spec_raw_sha256 is not None
        and normalized["evaluator_spec_raw_sha256"]
        != _sha256(
            evaluator_spec_raw_sha256, label="expected evaluator spec raw SHA-256"
        )
    ):
        raise PairV5SourceBoundEvaluationError("failure evaluator-spec raw binding differs")
    unsigned = dict(normalized)
    declared = unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        raise PairV5SourceBoundEvaluationError("failure receipt digest differs")
    return normalized


__all__ = [
    "EVAL_FRAME_INDICES",
    "CURRENT_FAMILY_ROLLOUT_SPEC_RAW_SHA256",
    "EXPECTED_GROUPS",
    "EXPECTED_GROUP_GPUS",
    "FEATURE_ORDER",
    "FAILURE_SCHEMA",
    "FRAME_COUNT",
    "GROUP_SCHEMA",
    "ROOT_SCHEMA",
    "INPUT_CLOSURE",
    "METRIC_CONTRACT",
    "METRIC_ORDER",
    "MODEL_ADAPTER_ID",
    "MODEL_NATIVE_IMAGE_SIZE",
    "MODEL_PATCH_SIZE",
    "EVALUATION_IMAGE_SIZE",
    "PREPROCESS_CONTRACT",
    "PROBE_ORDER",
    "RECEIPT_SCHEMA",
    "SCIENTIFIC_CLAIMS",
    "SUPPORTED_ARCHITECTURES",
    "SPEC_SCHEMA",
    "PairV5SourceBoundEvaluationError",
    "canonical_json_bytes",
    "file_sha256",
    "load_current_family_rollout_spec",
    "load_evaluator_spec",
    "make_candidate_receipt",
    "make_evaluator_spec",
    "make_failure_receipt",
    "make_group_receipt",
    "make_root_receipt",
    "object_sha256",
    "validate_candidate_receipt",
    "validate_evaluator_spec",
    "validate_failure_receipt",
    "validate_group_receipt",
    "validate_root_receipt",
    "generation_provenance_from_native_receipt",
    "validate_generation_provenance",
]

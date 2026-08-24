#!/usr/bin/env python3
"""Sealed two-seed CAPER object-grounding canary for the CDF dog example.

This runner deliberately reuses the committed frozen pure-T2V to full-source-
V2V exact81 implementation.  It only specializes the sealed population and
adds a non-automatic object-event/correspondence review contract to each child
receipt.  The two cells are same-source seed replicates, never independent
identities.  No generated output is automatically declared successful.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_t2v_v2v_branch_homotopy_canary as base  # noqa: E402


METHOD = "frozen-bernini-caper-cdf-dog-object-grounding-exact81-canary"
SCHEMA_VERSION = "bernini-caper-cdf-dog-object-grounding-exact81-receipt-v1"
REGISTRY_SCHEMA_VERSION = "bernini-caper-cdf-dog-object-grounding-exact81-v1"
CANONICAL_REGISTRY_RELATIVE = (
    "assets/caper_cdf_dog_object_grounding_exact81_v1.json"
)
CANONICAL_REGISTRY_SHA256 = (
    "f91327227384d4d29308d43895fe71d2fa9b4666438b9ec99bf6c65e7b7283c8"
)
CELL_ORDER = (
    "cdf-dog-historical-seed-2027",
    "cdf-dog-fresh-seed-2026081701",
)
WAVE_ORDER = ("wave1-same-source-seed-replication",)
WAVE_CELLS = {WAVE_ORDER[0]: CELL_ORDER}
COHORT_BY_WAVE = {WAVE_ORDER[0]: "cdf_dog_same_source_seed_replication"}
SOURCE_VIDEO = (
    "/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/"
    "288545b9c031491a/source.mp4"
)
SOURCE_VIDEO_SHA256 = (
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
)
TARGET_ACTION_CAPTION = "Make the dog pick up the bone and hold it in its mouth."
TARGET_ACTION_CAPTION_SHA256 = (
    "84df12ede824d239a4c7c3d21dccdf22663535d1e504e7b280544c8a9be0fd5d"
)
SEEDS = (2027, 2026081701)
BUCKET_HW = (496, 480)
LATENT_SHAPE = (1, 16, 21, 62, 60)
REQUIRED_EVENT_STAGES = ("approach", "contact", "grip", "lift", "hold")
REQUIRED_CORRESPONDENCES = (
    "source_dog_identity",
    "source_bone_identity",
    "source_dog_mouth_anatomical_identity",
)

ARM_ORDER = base.ARM_ORDER
GUIDANCE_BY_ARM = base.GUIDANCE_BY_ARM
HOMOTOPY_ARM = base.HOMOTOPY_ARM
SCHEDULE_SHA256 = base.SCHEDULE_SHA256
HIGH_ENDPOINT_STEP_INDICES = base.HIGH_ENDPOINT_STEP_INDICES
TRANSITION_STEP_INDICES = base.TRANSITION_STEP_INDICES
LOW_ENDPOINT_STEP_INDICES = base.LOW_ENDPOINT_STEP_INDICES
NATIVE_UNIPC40_SIGMAS = base.NATIVE_UNIPC40_SIGMAS


class CaperCDFDogObjectGroundingCanaryError(RuntimeError):
    """Raised before ambiguous same-source or object-grounding evidence ships."""


def _canonical_registry() -> Mapping[str, Any]:
    path = METHOD_ROOT / CANONICAL_REGISTRY_RELATIVE
    if (
        not path.is_file()
        or path.is_symlink()
        or base.native.legacy.file_sha256(path) != CANONICAL_REGISTRY_SHA256
    ):
        raise CaperCDFDogObjectGroundingCanaryError(
            "sealed CDF dog registry bytes differ"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CaperCDFDogObjectGroundingCanaryError(
            "sealed CDF dog registry JSON differs"
        ) from error
    if not isinstance(value, Mapping):
        raise CaperCDFDogObjectGroundingCanaryError("registry root differs")
    return value


def _validate_scientific_contract(registry: Mapping[str, Any]) -> None:
    contract = registry.get("contract")
    low = contract.get("native_source_video_only_v2v_endpoint") if isinstance(contract, Mapping) else None
    high = contract.get("pure_target_only_t2v_endpoint") if isinstance(contract, Mapping) else None
    hom = contract.get("homotopy") if isinstance(contract, Mapping) else None
    prompts = contract.get("prompt_homotopy_disclosure") if isinstance(contract, Mapping) else None
    schedule = contract.get("apg_and_scheduler") if isinstance(contract, Mapping) else None
    conditions = contract.get("condition_contract") if isinstance(contract, Mapping) else None
    if (
        not isinstance(low, Mapping)
        or low.get("guidance_mode") != "v2v_apg"
        or low.get("positive_task") != "mv2v"
        or low.get("full_source_video_count") != 1
        or low.get("source_reference_count") != 0
        or low.get("first_frame_condition_count") != 0
        or low.get("mask_track_pose_flow_count") != 0
        or low.get("transformer_forwards_per_step") != 2
        or low.get("pure_source_video_only") is not True
        or not isinstance(high, Mapping)
        or high.get("guidance_mode") != "t2v_apg"
        or high.get("positive_task") != "t2v"
        or high.get("full_source_video_count") != 0
        or high.get("source_reference_count") != 0
        or high.get("first_frame_condition_count") != 0
        or high.get("mask_track_pose_flow_count") != 0
        or high.get("target_only_visual_tokens") is not True
        or high.get("source_object_passed_to_sampler") is not False
        or high.get("transformer_forwards_per_step") != 2
        or not isinstance(hom, Mapping)
        or hom.get("high_sigma") != 0.95
        or hom.get("low_sigma") != 0.75
        or hom.get("endpoint_velocities_measured_at_same_x_t_timestep_and_sigma") is not True
        or hom.get("branch_apg_completed_independently_before_interpolation") is not True
        or hom.get("fp32_interpolation_before_one_official_scheduler_step") is not True
        or hom.get("hard_switch") is not False
        or not isinstance(prompts, Mapping)
        or prompts.get("same_target_action_caption_body") is not True
        or prompts.get("same_renderer_negative_embedding_object") is not True
        or prompts.get("task_prefix_and_visual_regime_change_together") is not True
        or prompts.get("shared_positive_embedding_across_endpoints") is not False
        or not isinstance(schedule, Mapping)
        or schedule.get("flow_shift_from_renderer_config") != 5.0
        or schedule.get("omega_text") != 4.0
        or schedule.get("eta") != 0.5
        or schedule.get("norm_threshold") != 50.0
        or schedule.get("momentum") != 0.0
        or schedule.get("unipc_steps") != 40
        or schedule.get("exact40_shift5_schedule_sha256") != SCHEDULE_SHA256
        or not isinstance(conditions, Mapping)
        or conditions.get("full_source_video_is_independently_vae_encoded_from_all_81_rgb_frames") is not True
        or conditions.get("pure_t2v_sampler_visual_conditions_all_none") is not True
        or conditions.get("source_v2v_sampler_has_exactly_one_full_source_video") is not True
        or conditions.get("source_references") is not False
        or conditions.get("first_frame_anchor") is not False
        or conditions.get("mask_track_pose_flow") is not False
        or contract.get("frame_count") != 81
        or contract.get("latent_phases") != 21
        or contract.get("num_inference_steps") != 40
        or contract.get("fps") != 25
        or contract.get("frozen_model") is not True
        or contract.get("training") is not False
        or contract.get("optimizer") is not False
        or contract.get("parameter_update") is not False
        or contract.get("target_video") is not False
        or contract.get("custom_initial_noise") is not False
    ):
        raise CaperCDFDogObjectGroundingCanaryError(
            "frozen exact81/exact40 three-arm contract differs"
        )
    regions = schedule.get("homotopy_regions")
    if (
        not isinstance(regions, Mapping)
        or regions.get("high_pure_t2v_weight_one_step_indices") != list(range(9))
        or regions.get("strict_transition_step_indices") != list(range(9, 26))
        or regions.get("low_source_v2v_weight_one_step_indices") != list(range(26, 40))
    ):
        raise CaperCDFDogObjectGroundingCanaryError(
            "exact40 homotopy regions differ"
        )


def _validate_object_grounding_contract(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation = registry.get("object_grounding_evaluation_contract")
    event = evaluation.get("ordered_object_event_gate") if isinstance(evaluation, Mapping) else None
    correspondence = evaluation.get("source_correspondence_gate") if isinstance(evaluation, Mapping) else None
    if (
        not isinstance(event, Mapping)
        or event.get("required_stage_order") != list(REQUIRED_EVENT_STAGES)
        or event.get("all_stages_required_in_order") is not True
        or event.get("automatic_adjudication_performed") is not False
        or event.get("automatic_success_claim_authorized") is not False
        or event.get("manual_review_required") is not True
        or not isinstance(event.get("stage_semantics"), Mapping)
        or set(event["stage_semantics"]) != set(REQUIRED_EVENT_STAGES)
        or not isinstance(correspondence, Mapping)
        or correspondence.get("required_correspondences") != list(REQUIRED_CORRESPONDENCES)
        or correspondence.get("dog_must_correspond_to_source_dog") is not True
        or correspondence.get("bone_must_correspond_to_source_bone") is not True
        or correspondence.get("mouth_must_correspond_to_source_dog_mouth") is not True
        or correspondence.get("replacement_dog_does_not_pass") is not True
        or correspondence.get("replacement_bone_does_not_pass") is not True
        or correspondence.get("detached_or_anatomically_incoherent_mouth_does_not_pass") is not True
        or correspondence.get("automatic_adjudication_performed") is not False
        or correspondence.get("automatic_success_claim_authorized") is not False
        or correspondence.get("manual_review_required") is not True
        or evaluation.get("conjunctive_pass_requires_both_gates") is not True
        or evaluation.get("receipt_records_contract_not_outcome") is not True
        or evaluation.get("automatic_success_claim_authorized") is not False
        or evaluation.get("single_example_success_claim_authorized") is not False
    ):
        raise CaperCDFDogObjectGroundingCanaryError(
            "object-event or source-correspondence gate differs"
        )
    return evaluation


def _registry_cell(registry: Mapping[str, Any], *, cell_id: str) -> Mapping[str, Any]:
    canonical = _canonical_registry()
    if registry != canonical:
        raise CaperCDFDogObjectGroundingCanaryError(
            "registry differs from sealed CDF dog authority"
        )
    if (
        registry.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or registry.get("method") != METHOD
        or registry.get("arm_order") != list(ARM_ORDER)
        or cell_id not in CELL_ORDER
    ):
        raise CaperCDFDogObjectGroundingCanaryError("registry root differs")
    _validate_scientific_contract(registry)
    _validate_object_grounding_contract(registry)
    population = registry.get("population_design")
    waves = population.get("waves") if isinstance(population, Mapping) else None
    if (
        population.get("wave_order") != list(WAVE_ORDER)
        or waves != [{
            "wave_id": WAVE_ORDER[0],
            "cohort": COHORT_BY_WAVE[WAVE_ORDER[0]],
            "cell_ids": list(CELL_ORDER),
        }]
        or population.get("same_source_cells_are_seed_replicates_not_independent_identities") is not True
        or population.get("aggregate_as_independent_identities_authorized") is not False
        or population.get("single_example_conclusion_authorized") is not False
        or population.get("automatic_success_claim_authorized") is not False
    ):
        raise CaperCDFDogObjectGroundingCanaryError(
            "same-source replication boundary differs"
        )
    rows = registry.get("cells")
    if not isinstance(rows, list) or len(rows) != 2:
        raise CaperCDFDogObjectGroundingCanaryError("cell population differs")
    for index, row in enumerate(rows):
        expected_role = (
            "historical_comparison_seed" if index == 0 else "fresh_preregistered_seed"
        )
        if (
            not isinstance(row, Mapping)
            or row.get("cell_id") != CELL_ORDER[index]
            or row.get("wave_id") != WAVE_ORDER[0]
            or row.get("cohort") != COHORT_BY_WAVE[WAVE_ORDER[0]]
            or row.get("replicate_role") != expected_role
            or row.get("actor_kind") != "dog"
            or row.get("source_iid") != "cdf-dog-288545b9c031491a"
            or row.get("source_video") != SOURCE_VIDEO
            or row.get("source_video_sha256") != SOURCE_VIDEO_SHA256
            or row.get("target_action_caption") != TARGET_ACTION_CAPTION
            or row.get("target_action_caption_sha256") != TARGET_ACTION_CAPTION_SHA256
            or row.get("seed") != SEEDS[index]
            or row.get("bucket_hw") != list(BUCKET_HW)
            or row.get("latent_shape") != list(LATENT_SHAPE)
            or row.get("selected_before_generation") is not True
        ):
            raise CaperCDFDogObjectGroundingCanaryError(
                f"sealed cell {index} differs"
            )
    return rows[CELL_ORDER.index(cell_id)]


def _specialize_base() -> Mapping[str, Any]:
    names = (
        "METHOD", "SCHEMA_VERSION", "REGISTRY_SCHEMA_VERSION",
        "CANONICAL_REGISTRY_RELATIVE", "CANONICAL_REGISTRY_SHA256",
        "CELL_ORDER", "WAVE_ORDER", "WAVE_CELLS", "COHORT_BY_WAVE",
        "_registry_cell",
    )
    previous = {name: getattr(base, name) for name in names}
    values = {
        "METHOD": METHOD,
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "REGISTRY_SCHEMA_VERSION": REGISTRY_SCHEMA_VERSION,
        "CANONICAL_REGISTRY_RELATIVE": CANONICAL_REGISTRY_RELATIVE,
        "CANONICAL_REGISTRY_SHA256": CANONICAL_REGISTRY_SHA256,
        "CELL_ORDER": CELL_ORDER,
        "WAVE_ORDER": WAVE_ORDER,
        "WAVE_CELLS": WAVE_CELLS,
        "COHORT_BY_WAVE": COHORT_BY_WAVE,
        "_registry_cell": _registry_cell,
    }
    for name, value in values.items():
        setattr(base, name, value)
    return previous


def _restore_base(previous: Mapping[str, Any]) -> None:
    for name, value in previous.items():
        setattr(base, name, value)


def _augment_rank0_receipt(*, output_dir: Path, cell: Mapping[str, Any]) -> None:
    receipt_path = output_dir / "receipt.json"
    if (
        not receipt_path.is_file()
        or receipt_path.is_symlink()
        or receipt_path.resolve(strict=True) != receipt_path
    ):
        raise CaperCDFDogObjectGroundingCanaryError("child receipt path differs")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CaperCDFDogObjectGroundingCanaryError("child receipt JSON differs") from error
    unsigned = dict(receipt) if isinstance(receipt, Mapping) else {}
    declared = unsigned.pop("receipt_digest", None)
    if (
        base._object_sha256(unsigned) != declared
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("method") != METHOD
        or receipt.get("cell_id") != cell["cell_id"]
        or receipt.get("input", {}).get("source_video_sha256") != SOURCE_VIDEO_SHA256
        or receipt.get("input", {}).get("target_action_caption") != TARGET_ACTION_CAPTION
        or receipt.get("sampling", {}).get("seed") != cell["seed"]
        or receipt.get("training_performed") is not False
        or receipt.get("optimizer_created") is not False
        or receipt.get("parameter_update") is not False
    ):
        raise CaperCDFDogObjectGroundingCanaryError(
            "base child receipt failed pre-augmentation postflight"
        )
    registry = _canonical_registry()
    receipt = dict(receipt)
    receipt.pop("receipt_digest", None)
    receipt["replication_design"] = {
        "replicate_role": cell["replicate_role"],
        "same_source_iid": cell["source_iid"],
        "same_source_cells": list(CELL_ORDER),
        "same_source_cells_are_seed_replicates_not_independent_identities": True,
        "aggregate_as_independent_identities_authorized": False,
        "seed_search_or_posthoc_selection_authorized": False,
    }
    receipt["object_grounding_evaluation"] = dict(
        registry["object_grounding_evaluation_contract"]
    )
    receipt["object_grounding_evaluation"]["gate_status"] = (
        "not_automatically_adjudicated"
    )
    receipt["object_grounding_evaluation"]["observed_outcome_recorded"] = False
    receipt["object_grounding_evaluation"]["automatic_success_claim"] = False
    receipt["object_grounding_evaluation"]["manual_review_required"] = True
    receipt["automatic_object_event_success_claim_authorized"] = False
    receipt["same_source_replication_identity_aggregation_authorized"] = False
    receipt["single_example_conclusion_authorized"] = False
    receipt["receipt_digest"] = base._object_sha256(receipt)
    base.value_audit.write_receipt_atomically(receipt_path, receipt)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    previous = _specialize_base()
    try:
        args = base.build_parser().parse_args(argv_list)
        _, registry = base.branch_base._plain_json(args.registry, label="registry")
        cell = _registry_cell(registry, cell_id=args.cell_id)
        result = base.main(argv_list)
        if result != 0:
            raise CaperCDFDogObjectGroundingCanaryError("base canary returned nonzero")
        if int(os.environ.get("RANK", "-1")) == 0:
            _augment_rank0_receipt(
                output_dir=Path(args.output_dir).resolve(strict=True), cell=cell
            )
        return 0
    finally:
        _restore_base(previous)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "BUCKET_HW",
    "CANONICAL_REGISTRY_SHA256",
    "CELL_ORDER",
    "CaperCDFDogObjectGroundingCanaryError",
    "LATENT_SHAPE",
    "METHOD",
    "REQUIRED_CORRESPONDENCES",
    "REQUIRED_EVENT_STAGES",
    "SCHEMA_VERSION",
    "SEEDS",
    "SOURCE_VIDEO",
    "SOURCE_VIDEO_SHA256",
    "TARGET_ACTION_CAPTION",
    "_registry_cell",
    "main",
]

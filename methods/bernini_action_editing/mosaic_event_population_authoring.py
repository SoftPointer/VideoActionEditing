#!/usr/bin/env python3
"""Closed authoring and audit contracts for the MOSAIC event population.

This module expands a compact, hand-reviewed registry into the existing
PAIR-v5 authoring schema.  It deliberately reuses one video only as an
exact-81 spatial-bucket probe; no pixels or latents from that video may enter
Bernini's T2V transformer.  Every synthetic actor/scene is therefore free of
the paired action-editing dataset's identity count.

The requested semantic branch is a rendering intervention, never a critic
label.  A candidate becomes eligible only after a detached, prompt-blind
post-render event-audit sidecar is authenticated.  A critic episode becomes
eligible only when all ten requested branches in the same seed cell pass the
audit.  Generated media remains critic-only evidence and can never become an
editor target, donor, condition, reference, or initial noise.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import pair_v5_t2v_calibration_bank_spec as pair_contract


REGISTRY_SCHEMA = "bernini-mosaic-event-population-registry-v1"
TOPUP_AUTHORING_SCHEMA = "bernini-mosaic-composition-topup-authoring-v1"
COMPOSITION_SCHEMA = "bernini-mosaic-event-population-composition-plan-v1"
AUDIT_REQUEST_SCHEMA = "bernini-mosaic-event-audit-request-manifest-v1"
AUDIT_SIDECAR_SCHEMA = "bernini-mosaic-detached-event-audit-v1"
ELIGIBILITY_SCHEMA = "bernini-mosaic-event-eligibility-index-v1"
COST_SCHEMA = "bernini-mosaic-event-population-cost-v1"

STAGE_IDS = ("pilot_topup12", "holdout_topup8", "full_topup20")
FAMILY_ROLES = (
    "pilot_development",
    "locked_action_family_validation",
    "locked_action_family_test",
)
SEMANTIC_BRANCHES = pair_contract.MACE_BRANCH_ORDER
DERIVED_SAME_VIDEO_NEGATIVES = (
    "same_video_reverse_chronology",
    "same_video_phase_shuffle",
    "same_video_freeze_first",
    "same_video_terminal_only",
    "same_video_transition_loop",
    "same_video_truncate_hold",
)
ANALYSIS_SPLITS = pair_contract.ANALYSIS_SPLITS

AUTHORING_CONTRACT = {
    "frame_count": 81,
    "latent_phase_count": 21,
    "inference_step_count": 40,
    "semantic_branches": list(SEMANTIC_BRANCHES),
    "derived_same_video_negatives": list(DERIVED_SAME_VIDEO_NEGATIVES),
    "generated_media_role": "text_conditioned_event_critic_evidence_only",
    "generated_media_may_train_editor": False,
    "generated_media_may_be_editor_target": False,
    "generated_media_may_be_editor_condition": False,
    "generated_media_may_be_editor_donor": False,
    "generated_media_may_be_editor_initial_noise": False,
    "geometry_source_role": "exact81_bucket_shape_probe_only",
    "geometry_source_pixels_enter_transformer": False,
    "geometry_source_vae_latent_created": False,
    "prompt_semantic_branch_is_critic_label": False,
    "detached_event_audit_required_before_candidate_eligibility": True,
    "complete_ten_branch_cell_required_before_feature_extraction": True,
    "generated_cell_count_never_implies_eligible_cell_count": True,
}

STAGE_POLICY = {
    "atomic_action_family_count": 6,
    "target_identity_scene_groups_per_family": 3,
    "official_gaussian_seeds_per_identity_scene": 2,
    "inherited_seed_cells": 16,
    "new_topup_seed_cells": 20,
    "composed_seed_cells": 36,
    "inherited_generated_clips": 160,
    "new_generated_clips": 200,
    "composed_generated_clips": 360,
    "pilot_topup_seed_cells": 12,
    "locked_holdout_topup_seed_cells": 8,
    "all_nine_semantic_negative_types_rendered_per_episode": True,
    "family_metrics_must_not_pool_distinct_action_events": True,
    "fit_confirmation_actor_scene_action_seed_disjoint": True,
    "locked_family_roles_excluded_from_critic_fit": True,
    "compact_three_identity_population_is_scientific_claim_authority": False,
    "compact_three_identity_population_is_editor_optimizer_authority": False,
}

INHERITED_BANK_PROFILES = [
    {
        "profile_id": "core4-v2",
        "seed1_root_spec_raw_sha256": (
            "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95"
        ),
        "seed2_root_spec_raw_sha256": (
            "900c0dece65ee2f075765571b39d62e45ceb1b3c8b5c883443ea09d1876e18f3"
        ),
        "seed1_candidate_prefix": "pair5-t2v-core4-v2-",
        "seed2_candidate_prefix": "pair5-t2v-core4-seed2-",
        "seed_map": {
            "2026080825": 2026080925,
            "2026080826": 2026080926,
            "2026080827": 2026080927,
            "2026080828": 2026080928,
        },
        "identity_scene_count": 4,
        "seed_cell_count": 8,
        "generated_clip_count": 80,
    },
    {
        "profile_id": "reserve4-v1",
        "seed1_root_spec_raw_sha256": (
            "2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab"
        ),
        "seed2_root_spec_raw_sha256": (
            "0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e"
        ),
        "seed1_candidate_prefix": "pair5-t2v-reserve4-v1-",
        "seed2_candidate_prefix": "pair5-t2v-reserve4-seed2-",
        "seed_map": {
            "2026080821": 2026080921,
            "2026080822": 2026080922,
            "2026080823": 2026080923,
            "2026080824": 2026080924,
        },
        "identity_scene_count": 4,
        "seed_cell_count": 8,
        "generated_clip_count": 80,
    },
]

AUDIT_POLICY = {
    "requested_branch_is_render_intervention_only": True,
    "requested_branch_is_label": False,
    "auditor_receives_render_and_sealed_family_rubric_only": True,
    "auditor_blinded_to_generation_prompt_and_requested_branch": True,
    "entire_exact81_video_must_be_viewed": True,
    "audit_sealed_before_hidden_feature_extraction": True,
    "one_detached_sidecar_per_candidate_required": True,
    "ambiguous_or_low_quality_render_is_rejection_not_negative": True,
    "negative_may_not_be_labeled_from_prompt_intent": True,
    "positive_requires_start_transition_terminal_hold_conjunction": True,
    "negative_requires_full_target_event_false_confirmation": True,
    "complete_ten_branch_cell_required": True,
    "allowed_auditor_methods": [
        "human_blind_video_review_v1",
        "fixed_vlm_blind_video_review_v1",
        "dual_consensus_blind_video_review_v1",
    ],
}

REFERENCE_COST = {
    "hardware": "one_AUH_node_8xMI210_dual_concurrent_SP4",
    "reference_generated_clips": 40,
    "reference_wall_minutes": 49.0,
    "estimate_kind": "planning_only_from_observed_current_serial_reload_launcher",
    "persistent_worker_speedup_assumed": False,
}

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "registry_id",
        "geometry_source_video",
        "geometry_contract",
        "authoring_contract",
        "stage_policy",
        "audit_policy",
        "reference_cost",
        "inherited_bank_profiles",
        "action_families",
    }
)
_FAMILY_FIELDS = frozenset(
    {
        "family_id",
        "family_code",
        "population_role",
        "event_definition",
        "branch_descriptions",
        "inherited_identity_scenes",
        "topup_identity_scenes",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "subject_kind",
        "start_state",
        "ordered_transition",
        "terminal_state",
        "minimum_terminal_hold_frames",
        "actor_binding_rule",
        "object_binding_required",
        "object_binding_rule",
        "camera_background_rule",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "identity_scene_id",
        "analysis_split",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "scene_caption",
        "camera_caption",
        "seeds",
    }
)
_INHERITED_IDENTITY_FIELDS = frozenset(
    {
        "identity_scene_id",
        "analysis_split",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "source_iid",
        "source_bank_profile",
        "seeds",
    }
)
_TOPUP_AUTHORING_FIELDS = frozenset(
    {
        "schema_version",
        "registry_id",
        "stage_id",
        "bank_id",
        "expected_cell_count",
        "expected_candidate_count",
        "semantic_branch_order",
        "geometry_contract",
        "geometry_source_pixels_enter_transformer",
        "geometry_source_vae_latent_created",
        "generated_media_editor_use_authorized",
        "cells",
    }
)
_TOPUP_CELL_FIELDS = frozenset(
    {
        "iid",
        "analysis_split",
        "action_family_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "execution_group",
        "geometry_source_video",
        "seed",
        "scene_caption",
        "branch_descriptions",
        "camera_caption",
    }
)
_AUDIT_REQUEST_FIELDS = frozenset(
    {
        "candidate_id",
        "cell_id",
        "analysis_split",
        "critic_partition",
        "action_family_id",
        "population_role",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "seed",
        "requested_semantic_branch",
        "full_t2v_caption_utf8_sha256",
        "event_definition",
        "event_definition_sha256",
        "audit_protocol_sha256",
        "audit_sidecar_basename",
        "authenticated_generation_receipt_digest",
        "candidate_eligible_before_audit",
        "critic_label_before_audit",
        "request_digest",
    }
)
_COMPOSITION_FIELDS = frozenset(
    {
        "schema_version",
        "registry_id",
        "stage_id",
        "inherited_bank_profiles",
        "inherited_cell_count",
        "inherited_candidate_count",
        "inherited_cells",
        "new_topup_cell_count",
        "new_topup_candidate_count",
        "new_topup_cells",
        "composed_cell_count_after_stage",
        "composed_candidate_count_after_stage",
        "full_compact_target_cell_count",
        "full_compact_target_candidate_count",
        "identity_scene_groups_per_family_at_full_target",
        "seeds_per_identity_scene_at_full_target",
        "prompt_or_requested_branch_used_as_label",
        "event_audit_required_for_every_row",
        "compact_population_is_pilot_only",
        "scientific_claim_authorized",
        "editor_optimizer_authorized",
        "composition_plan_digest",
    }
)
_AUDIT_CHECK_FIELDS = frozenset(
    {
        "video_quality_pass",
        "continuous_no_cut",
        "primary_actor_trackable",
        "family_start_state_observed",
        "family_transition_observed",
        "family_terminal_state_observed",
        "family_terminal_hold_observed",
        "full_target_event_observed",
        "full_target_event_false_confirmed",
        "requested_branch_mechanism_observed",
        "actor_binding",
        "object_binding",
        "camera_class",
        "appearance_only_observed",
    }
)
_AUDIT_EVIDENCE_FIELDS = frozenset(
    {
        "start_frames",
        "transition_frames",
        "terminal_frames",
        "terminal_hold_frames",
        "branch_mechanism_frames",
        "written_observation",
    }
)
_AUDIT_SIDECAR_FIELDS = frozenset(
    {
        "schema_version",
        "request_digest",
        "candidate_id",
        "rendered_media_path",
        "rendered_media_sha256",
        "generation_receipt_path",
        "generation_receipt_sha256",
        "generation_receipt_digest",
        "audit_evidence_path",
        "audit_evidence_sha256",
        "audit_protocol_sha256",
        "auditor_id",
        "auditor_method",
        "audited_at_utc",
        "generation_prompt_or_requested_branch_disclosed",
        "entire_video_viewed",
        "hidden_feature_extraction_started_before_audit",
        "observed_class",
        "checks",
        "evidence",
        "eligibility_decision",
        "rejection_reasons",
        "sidecar_digest",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_BROAD_FAMILY_IDS = frozenset(
    {
        "action",
        "motion",
        "human-motion",
        "articulated-pose-transition",
        "pose-transition",
    }
)
_OBSERVED_CLASSES = frozenset((*SEMANTIC_BRANCHES, "ambiguous", "unusable"))
_ACTOR_BINDINGS = frozenset(("primary", "secondary", "none", "ambiguous"))
_OBJECT_BINDINGS = frozenset(
    ("target", "distractor", "none", "not_applicable", "ambiguous")
)
_CAMERA_CLASSES = frozenset(
    ("locked_or_natural", "camera_only_motion", "unexpected_motion", "ambiguous")
)


class MosaicEventPopulationError(ValueError):
    """The population, audit, or eligibility evidence is not closed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise MosaicEventPopulationError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MosaicEventPopulationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _closed(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise MosaicEventPopulationError(
            f"{label} field closure differs; expected={sorted(fields)!r}, actual={actual!r}"
        )
    return dict(value)


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise MosaicEventPopulationError(f"{label} must be path-safe")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MosaicEventPopulationError(f"{label} must be lowercase SHA-256")
    return value


def _text(value: Any, *, label: str, minimum_words: int) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise MosaicEventPopulationError(f"{label} must be text without NUL")
    result = value.strip()
    if len(result.split()) < minimum_words or "{" in result or "}" in result:
        raise MosaicEventPopulationError(
            f"{label} is incomplete or contains an unexpanded placeholder"
        )
    return result


def _absolute_path_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise MosaicEventPopulationError(f"{label} must be path text")
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise MosaicEventPopulationError(f"{label} must be absolute and non-root")
    return str(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(path_text: Any, *, label: str) -> Path:
    path = Path(_absolute_path_text(path_text, label=label))
    if not path.is_file() or path.is_symlink():
        raise MosaicEventPopulationError(f"{label} must be an absolute plain file")
    return path


def load_sealed_registry(
    path: str | Path, expected_raw_sha256: str
) -> tuple[dict[str, Any], str]:
    source = _plain_file(str(path), label="sealed population registry")
    expected = _sha256(expected_raw_sha256, label="expected registry SHA-256")
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise MosaicEventPopulationError("sealed registry raw SHA-256 differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MosaicEventPopulationError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MosaicEventPopulationError("sealed registry is not UTF-8 JSON") from error
    return validate_registry(value), actual


def validate_registry(value: Any) -> dict[str, Any]:
    root = _closed(value, _ROOT_FIELDS, label="population registry")
    if root["schema_version"] != REGISTRY_SCHEMA:
        raise MosaicEventPopulationError("population registry schema differs")
    registry_id = _safe_id(root["registry_id"], label="registry_id")
    geometry_path = _absolute_path_text(
        root["geometry_source_video"], label="geometry_source_video"
    )
    if root["geometry_contract"] != pair_contract.GEOMETRY_CONTRACT:
        raise MosaicEventPopulationError("geometry probe contract differs")
    if root["authoring_contract"] != AUTHORING_CONTRACT:
        raise MosaicEventPopulationError("authoring contract differs")
    if root["stage_policy"] != STAGE_POLICY:
        raise MosaicEventPopulationError("stage policy differs")
    if root["audit_policy"] != AUDIT_POLICY:
        raise MosaicEventPopulationError("audit policy differs")
    if root["reference_cost"] != REFERENCE_COST:
        raise MosaicEventPopulationError("reference cost contract differs")
    if root["inherited_bank_profiles"] != INHERITED_BANK_PROFILES:
        raise MosaicEventPopulationError("inherited bank profile contract differs")

    raw_families = root["action_families"]
    if not isinstance(raw_families, list) or len(raw_families) != 6:
        raise MosaicEventPopulationError("registry requires exactly six atomic families")
    families: list[dict[str, Any]] = []
    family_ids: set[str] = set()
    family_codes: set[str] = set()
    event_digests: set[str] = set()
    all_actor_groups: set[str] = set()
    all_scene_groups: set[str] = set()
    all_action_groups: set[str] = set()
    all_identity_scene_ids: set[str] = set()
    all_seeds: set[int] = set()
    roles = Counter()

    for family_index, raw_family in enumerate(raw_families):
        family = _closed(
            raw_family, _FAMILY_FIELDS, label=f"action_families[{family_index}]"
        )
        family_id = _safe_id(
            family["family_id"], label=f"family[{family_index}].family_id"
        )
        if family_id in _BROAD_FAMILY_IDS or family_id in family_ids:
            raise MosaicEventPopulationError(
                "family IDs must be unique atomic events, never broad pose/motion buckets"
            )
        family_ids.add(family_id)
        family_code = _safe_id(
            family["family_code"], label=f"{family_id}.family_code"
        )
        if len(family_code) > 12 or family_code in family_codes:
            raise MosaicEventPopulationError("family_code must be unique and at most 12 chars")
        family_codes.add(family_code)
        role = family["population_role"]
        if role not in FAMILY_ROLES:
            raise MosaicEventPopulationError(f"{family_id} population role differs")
        roles[role] += 1

        event = _closed(
            family["event_definition"], _EVENT_FIELDS, label=f"{family_id}.event"
        )
        normalized_event = {
            "subject_kind": _safe_id(
                event["subject_kind"], label=f"{family_id}.subject_kind"
            ),
            "start_state": _text(
                event["start_state"], label=f"{family_id}.start_state", minimum_words=6
            ),
            "ordered_transition": _text(
                event["ordered_transition"],
                label=f"{family_id}.ordered_transition",
                minimum_words=8,
            ),
            "terminal_state": _text(
                event["terminal_state"],
                label=f"{family_id}.terminal_state",
                minimum_words=6,
            ),
            "minimum_terminal_hold_frames": event["minimum_terminal_hold_frames"],
            "actor_binding_rule": _text(
                event["actor_binding_rule"],
                label=f"{family_id}.actor_binding_rule",
                minimum_words=6,
            ),
            "object_binding_required": event["object_binding_required"],
            "object_binding_rule": _text(
                event["object_binding_rule"],
                label=f"{family_id}.object_binding_rule",
                minimum_words=5,
            ),
            "camera_background_rule": _text(
                event["camera_background_rule"],
                label=f"{family_id}.camera_background_rule",
                minimum_words=6,
            ),
        }
        if (
            type(normalized_event["minimum_terminal_hold_frames"]) is not int
            or not 4 <= normalized_event["minimum_terminal_hold_frames"] <= 24
            or type(normalized_event["object_binding_required"]) is not bool
        ):
            raise MosaicEventPopulationError(f"{family_id} event scalar contract differs")
        event_digest = object_sha256(normalized_event)
        if event_digest in event_digests:
            raise MosaicEventPopulationError(
                "distinct action_family_id values may not alias one event definition"
            )
        event_digests.add(event_digest)

        descriptions = _closed(
            family["branch_descriptions"],
            frozenset(SEMANTIC_BRANCHES),
            label=f"{family_id}.branch_descriptions",
        )
        normalized_descriptions = {
            branch: _text(
                descriptions[branch],
                label=f"{family_id}.{branch}",
                minimum_words=10,
            )
            for branch in SEMANTIC_BRANCHES
        }
        if len(set(normalized_descriptions.values())) != len(SEMANTIC_BRANCHES):
            raise MosaicEventPopulationError(f"{family_id} branch descriptions collide")

        raw_inherited_scenes = family["inherited_identity_scenes"]
        raw_topup_scenes = family["topup_identity_scenes"]
        if (
            not isinstance(raw_inherited_scenes, list)
            or not isinstance(raw_topup_scenes, list)
            or len(raw_inherited_scenes) not in (1, 2)
            or len(raw_topup_scenes) != 3 - len(raw_inherited_scenes)
        ):
            raise MosaicEventPopulationError(
                f"{family_id} inherited+topup identity topology must close at three"
            )
        split_counts = Counter()
        normalized_inherited_scenes = []
        normalized_topup_scenes = []
        profile_by_id = {
            profile["profile_id"]: profile for profile in INHERITED_BANK_PROFILES
        }

        def register_groups(scene: Mapping[str, Any]) -> tuple[str, str, str]:
            actor_group = _safe_id(
                scene["actor_group_id"], label=f"{family_id}.actor_group_id"
            )
            scene_group = _safe_id(
                scene["scene_group_id"], label=f"{family_id}.scene_group_id"
            )
            action_group = _safe_id(
                scene["action_group_id"], label=f"{family_id}.action_group_id"
            )
            for value_set, group_value, group_label in (
                (all_actor_groups, actor_group, "actor_group_id"),
                (all_scene_groups, scene_group, "scene_group_id"),
                (all_action_groups, action_group, "action_group_id"),
            ):
                if group_value in value_set:
                    raise MosaicEventPopulationError(
                        f"{group_label} must be globally identity/split disjoint"
                    )
                value_set.add(group_value)
            return actor_group, scene_group, action_group

        def register_identity_and_seeds(
            scene: Mapping[str, Any], *, inherited: bool
        ) -> tuple[str, str, list[int], str, str, str]:
            identity_scene_id = _safe_id(
                scene["identity_scene_id"],
                label=f"{family_id}.identity_scene_id",
            )
            composite_identity = f"{family_code}-{identity_scene_id}"
            if composite_identity in all_identity_scene_ids:
                raise MosaicEventPopulationError("identity_scene_id values must be global")
            all_identity_scene_ids.add(composite_identity)
            split = scene["analysis_split"]
            if split not in ANALYSIS_SPLITS:
                raise MosaicEventPopulationError(f"{family_id} split differs")
            split_counts[split] += 1
            actor_group, scene_group, action_group = register_groups(scene)
            seeds = scene["seeds"]
            if (
                not isinstance(seeds, list)
                or len(seeds) != 2
                or len(set(seeds)) != 2
                or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
            ):
                raise MosaicEventPopulationError(
                    f"{family_id}/{identity_scene_id} requires two valid seeds"
                )
            if any(seed in all_seeds for seed in seeds):
                raise MosaicEventPopulationError(
                    "official Gaussian seeds must be globally split-disjoint"
                )
            all_seeds.update(seeds)
            return (
                identity_scene_id,
                split,
                list(seeds),
                actor_group,
                scene_group,
                action_group,
            )

        for scene_index, raw_scene in enumerate(raw_inherited_scenes):
            scene = _closed(
                raw_scene,
                _INHERITED_IDENTITY_FIELDS,
                label=f"{family_id}.inherited_identity_scenes[{scene_index}]",
            )
            (
                identity_scene_id,
                split,
                seeds,
                actor_group,
                scene_group,
                action_group,
            ) = register_identity_and_seeds(scene, inherited=True)
            source_iid = _safe_id(scene["source_iid"], label=f"{family_id}.source_iid")
            profile_id = scene["source_bank_profile"]
            if profile_id not in profile_by_id:
                raise MosaicEventPopulationError(f"{family_id} inherited profile differs")
            profile = profile_by_id[profile_id]
            seed_map = profile["seed_map"]
            if str(seeds[0]) not in seed_map or seed_map[str(seeds[0])] != seeds[1]:
                raise MosaicEventPopulationError(
                    f"{family_id}/{source_iid} inherited seed pair differs"
                )
            normalized_inherited_scenes.append(
                {
                    "identity_scene_id": identity_scene_id,
                    "analysis_split": split,
                    "actor_group_id": actor_group,
                    "scene_group_id": scene_group,
                    "action_group_id": action_group,
                    "source_iid": source_iid,
                    "source_bank_profile": profile_id,
                    "seeds": seeds,
                }
            )

        for scene_index, raw_scene in enumerate(raw_topup_scenes):
            scene = _closed(
                raw_scene,
                _IDENTITY_FIELDS,
                label=f"{family_id}.topup_identity_scenes[{scene_index}]",
            )
            (
                identity_scene_id,
                split,
                seeds,
                actor_group,
                scene_group,
                action_group,
            ) = register_identity_and_seeds(scene, inherited=False)
            normalized_topup_scenes.append(
                {
                    "identity_scene_id": identity_scene_id,
                    "analysis_split": split,
                    "actor_group_id": actor_group,
                    "scene_group_id": scene_group,
                    "action_group_id": action_group,
                    "scene_caption": _text(
                        scene["scene_caption"],
                        label=f"{family_id}.scene_caption",
                        minimum_words=18,
                    ),
                    "camera_caption": _text(
                        scene["camera_caption"],
                        label=f"{family_id}.camera_caption",
                        minimum_words=10,
                    ),
                    "seeds": seeds,
                }
            )
        if set(split_counts) != set(ANALYSIS_SPLITS) or sum(split_counts.values()) != 3:
            raise MosaicEventPopulationError(
                f"{family_id} compact pilot requires three identities across both splits"
            )
        families.append(
            {
                "family_id": family_id,
                "family_code": family_code,
                "population_role": role,
                "event_definition": normalized_event,
                "branch_descriptions": normalized_descriptions,
                "inherited_identity_scenes": normalized_inherited_scenes,
                "topup_identity_scenes": normalized_topup_scenes,
            }
        )
    if roles != Counter(
        {
            "pilot_development": 4,
            "locked_action_family_validation": 1,
            "locked_action_family_test": 1,
        }
    ):
        raise MosaicEventPopulationError("family population roles must be 4/1/1")
    inherited_identity_count = sum(
        len(family["inherited_identity_scenes"]) for family in families
    )
    topup_identity_count = sum(
        len(family["topup_identity_scenes"]) for family in families
    )
    if inherited_identity_count != 8 or topup_identity_count != 10:
        raise MosaicEventPopulationError("population must reuse 8 and add 10 identities")
    profile_counts = Counter(
        scene["source_bank_profile"]
        for family in families
        for scene in family["inherited_identity_scenes"]
    )
    if profile_counts != Counter({"core4-v2": 4, "reserve4-v1": 4}):
        raise MosaicEventPopulationError("inherited core4/reserve4 identity mapping differs")
    return {
        "schema_version": REGISTRY_SCHEMA,
        "registry_id": registry_id,
        "geometry_source_video": geometry_path,
        "geometry_contract": pair_contract.GEOMETRY_CONTRACT,
        "authoring_contract": dict(AUTHORING_CONTRACT),
        "stage_policy": dict(STAGE_POLICY),
        "audit_policy": dict(AUDIT_POLICY),
        "reference_cost": dict(REFERENCE_COST),
        "inherited_bank_profiles": json.loads(json.dumps(INHERITED_BANK_PROFILES)),
        "action_families": families,
    }


def _families_for_stage(
    registry: Mapping[str, Any], stage_id: str
) -> list[Mapping[str, Any]]:
    if stage_id not in STAGE_IDS:
        raise MosaicEventPopulationError(f"stage_id must be one of {STAGE_IDS!r}")
    families = registry["action_families"]
    if stage_id == "pilot_topup12":
        return [row for row in families if row["population_role"] == "pilot_development"]
    if stage_id == "holdout_topup8":
        return [row for row in families if row["population_role"] != "pilot_development"]
    return list(families)


def _critic_partition(population_role: str, analysis_split: str) -> str:
    if population_role == "pilot_development":
        return (
            "critic_fit"
            if analysis_split == "fit"
            else "identity_scene_seed_confirmation"
        )
    if population_role == "locked_action_family_validation":
        return "locked_action_family_validation"
    if population_role == "locked_action_family_test":
        return "locked_action_family_test"
    raise MosaicEventPopulationError("unknown critic partition")


def _candidate_id(bank_id: str, iid: str, branch: str) -> str:
    candidate_id = f"{bank_id}-{iid}-{branch}"
    if len(candidate_id) > 96:
        candidate_id = f"{iid}-{branch}"
    return _safe_id(candidate_id, label="candidate_id")


def build_stage_bundle(registry_value: Any, *, stage_id: str) -> dict[str, Any]:
    """Build a composition-aware top-up fragment, audit requests, and cost.

    The fragment intentionally is not mislabeled as the old standalone
    PAIR-v5 authoring schema: dog-sit and human-rise each need only one new
    identity, so their top-up fragment contains one split while the inherited
    cells supply the other.  Scientific split closure is checked on the
    composed 16+20 population, not by forcing redundant regeneration.
    """

    registry = validate_registry(registry_value)
    families = _families_for_stage(registry, stage_id)
    bank_id = f"mosaic-{stage_id}-v1"
    cells: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    audit_protocol_digest = object_sha256(AUDIT_POLICY)
    ordinal = 0
    for family in families:
        for scene in family["topup_identity_scenes"]:
            for seed_slot, seed in enumerate(scene["seeds"]):
                iid = f"{family['family_code']}-{scene['identity_scene_id']}-z{seed_slot}"
                _safe_id(iid, label="cell IID")
                execution_group = "sp4-a" if ordinal % 2 == 0 else "sp4-b"
                ordinal += 1
                cell = {
                    "iid": iid,
                    "analysis_split": scene["analysis_split"],
                    "action_family_id": family["family_id"],
                    "actor_group_id": scene["actor_group_id"],
                    "scene_group_id": scene["scene_group_id"],
                    "action_group_id": scene["action_group_id"],
                    "execution_group": execution_group,
                    "geometry_source_video": registry["geometry_source_video"],
                    "seed": seed,
                    "scene_caption": scene["scene_caption"],
                    "branch_descriptions": dict(family["branch_descriptions"]),
                    "camera_caption": scene["camera_caption"],
                }
                cells.append(cell)
                event_definition = dict(family["event_definition"])
                event_digest = object_sha256(event_definition)
                for branch in SEMANTIC_BRANCHES:
                    caption = " ".join(
                        (
                            scene["scene_caption"],
                            family["branch_descriptions"][branch],
                            scene["camera_caption"],
                        )
                    )
                    candidate_id = _candidate_id(bank_id, iid, branch)
                    unsigned_request = {
                        "candidate_id": candidate_id,
                        "cell_id": f"cell-{iid}-s{seed}",
                        "analysis_split": scene["analysis_split"],
                        "critic_partition": _critic_partition(
                            family["population_role"], scene["analysis_split"]
                        ),
                        "action_family_id": family["family_id"],
                        "population_role": family["population_role"],
                        "actor_group_id": scene["actor_group_id"],
                        "scene_group_id": scene["scene_group_id"],
                        "action_group_id": scene["action_group_id"],
                        "seed": seed,
                        "requested_semantic_branch": branch,
                        "full_t2v_caption_utf8_sha256": hashlib.sha256(
                            caption.encode("utf-8")
                        ).hexdigest(),
                        "event_definition": event_definition,
                        "event_definition_sha256": event_digest,
                        "audit_protocol_sha256": audit_protocol_digest,
                        "audit_sidecar_basename": f"{candidate_id}.event-audit.json",
                        "authenticated_generation_receipt_digest": None,
                        "candidate_eligible_before_audit": False,
                        "critic_label_before_audit": None,
                    }
                    requests.append(
                        {
                            **unsigned_request,
                            "request_digest": object_sha256(unsigned_request),
                        }
                    )

    authoring = {
        "schema_version": TOPUP_AUTHORING_SCHEMA,
        "registry_id": registry["registry_id"],
        "stage_id": stage_id,
        "bank_id": bank_id,
        "expected_cell_count": len(cells),
        "expected_candidate_count": len(cells) * len(SEMANTIC_BRANCHES),
        "semantic_branch_order": list(SEMANTIC_BRANCHES),
        "geometry_contract": pair_contract.GEOMETRY_CONTRACT,
        "geometry_source_pixels_enter_transformer": False,
        "geometry_source_vae_latent_created": False,
        "generated_media_editor_use_authorized": False,
        "cells": cells,
    }

    profile_by_id = {
        profile["profile_id"]: profile
        for profile in registry["inherited_bank_profiles"]
    }
    inherited_cells = []
    for family in registry["action_families"]:
        for scene in family["inherited_identity_scenes"]:
            profile = profile_by_id[scene["source_bank_profile"]]
            legacy_family = (
                family["family_id"]
                if scene["source_bank_profile"] == "core4-v2"
                else "articulated-pose-transition"
            )
            for seed_slot, seed in enumerate(scene["seeds"]):
                prefix = profile[
                    "seed1_candidate_prefix" if seed_slot == 0 else "seed2_candidate_prefix"
                ]
                spec_digest = profile[
                    "seed1_root_spec_raw_sha256"
                    if seed_slot == 0
                    else "seed2_root_spec_raw_sha256"
                ]
                inherited_cells.append(
                    {
                        "population_source": "existing_or_inflight_reuse_no_regeneration",
                        "source_bank_profile": scene["source_bank_profile"],
                        "source_root_spec_raw_sha256": spec_digest,
                        "source_root_spec_hash_must_be_bound_before_use": spec_digest is None,
                        "source_iid": scene["source_iid"],
                        "cell_id": f"cell-{scene['source_iid']}-s{seed}",
                        "analysis_split": scene["analysis_split"],
                        "legacy_action_family_id": legacy_family,
                        "atomic_action_family_id": family["family_id"],
                        "atomic_family_relabel_is_metadata_only": True,
                        "actor_group_id": scene["actor_group_id"],
                        "scene_group_id": scene["scene_group_id"],
                        "action_group_id": scene["action_group_id"],
                        "seed": seed,
                        "candidate_ids": [
                            f"{prefix}{scene['source_iid']}-{branch}"
                            for branch in SEMANTIC_BRANCHES
                        ],
                        "semantic_branch_order": list(SEMANTIC_BRANCHES),
                        "requested_branch_is_label": False,
                        "detached_event_audit_required": True,
                    }
                )
    new_topup_cells = []
    requests_by_cell: dict[str, list[str]] = defaultdict(list)
    for request in requests:
        requests_by_cell[request["cell_id"]].append(request["candidate_id"])
    selected_family_by_id = {family["family_id"]: family for family in families}
    for cell in cells:
        cell_id = f"cell-{cell['iid']}-s{cell['seed']}"
        family = selected_family_by_id[cell["action_family_id"]]
        new_topup_cells.append(
            {
                "population_source": "new_topup_generation",
                "topup_bank_id": bank_id,
                "cell_id": cell_id,
                "analysis_split": cell["analysis_split"],
                "atomic_action_family_id": cell["action_family_id"],
                "population_role": family["population_role"],
                "actor_group_id": cell["actor_group_id"],
                "scene_group_id": cell["scene_group_id"],
                "action_group_id": cell["action_group_id"],
                "seed": cell["seed"],
                "candidate_ids": requests_by_cell[cell_id],
                "semantic_branch_order": list(SEMANTIC_BRANCHES),
                "requested_branch_is_label": False,
                "detached_event_audit_required": True,
            }
        )
    composition_unsigned = {
        "schema_version": COMPOSITION_SCHEMA,
        "registry_id": registry["registry_id"],
        "stage_id": stage_id,
        "inherited_bank_profiles": registry["inherited_bank_profiles"],
        "inherited_cell_count": len(inherited_cells),
        "inherited_candidate_count": len(inherited_cells) * len(SEMANTIC_BRANCHES),
        "inherited_cells": inherited_cells,
        "new_topup_cell_count": len(new_topup_cells),
        "new_topup_candidate_count": len(new_topup_cells) * len(SEMANTIC_BRANCHES),
        "new_topup_cells": new_topup_cells,
        "composed_cell_count_after_stage": len(inherited_cells) + len(new_topup_cells),
        "composed_candidate_count_after_stage": (
            len(inherited_cells) + len(new_topup_cells)
        ) * len(SEMANTIC_BRANCHES),
        "full_compact_target_cell_count": 36,
        "full_compact_target_candidate_count": 360,
        "identity_scene_groups_per_family_at_full_target": 3,
        "seeds_per_identity_scene_at_full_target": 2,
        "prompt_or_requested_branch_used_as_label": False,
        "event_audit_required_for_every_row": True,
        "compact_population_is_pilot_only": True,
        "scientific_claim_authorized": False,
        "editor_optimizer_authorized": False,
    }
    composition = {
        **composition_unsigned,
        "composition_plan_digest": object_sha256(composition_unsigned),
    }
    request_unsigned = {
        "schema_version": AUDIT_REQUEST_SCHEMA,
        "registry_id": registry["registry_id"],
        "stage_id": stage_id,
        "pair_v5_bank_id": bank_id,
        "authoring_object_sha256": object_sha256(authoring),
        "audit_policy": dict(AUDIT_POLICY),
        "audit_protocol_sha256": audit_protocol_digest,
        "candidate_count": len(requests),
        "candidate_requests": requests,
        "generated_media_editor_use_authorized": False,
        "critic_feature_extraction_authorized_before_audits": False,
    }
    audit_requests = {
        **request_unsigned,
        "manifest_digest": object_sha256(request_unsigned),
    }
    group_cell_counts = Counter(cell["execution_group"] for cell in cells)
    group_clip_counts = {
        group: group_cell_counts[group] * len(SEMANTIC_BRANCHES)
        for group in ("sp4-a", "sp4-b")
    }
    clip_count = len(requests)
    estimated_wall_minutes = (
        clip_count
        * REFERENCE_COST["reference_wall_minutes"]
        / REFERENCE_COST["reference_generated_clips"]
    )
    composed_cells = len(inherited_cells) + len(cells)
    cost = {
        "schema_version": COST_SCHEMA,
        "stage_id": stage_id,
        "new_topup_action_family_count": len(families),
        "new_topup_identity_scene_group_count": sum(
            len(family["topup_identity_scenes"]) for family in families
        ),
        "inherited_seed_cell_count": len(inherited_cells),
        "new_topup_seed_cell_count": len(cells),
        "composed_seed_cell_count_after_stage": composed_cells,
        "generated_positive_clip_count": len(cells),
        "generated_semantic_negative_clip_count": len(cells)
        * (len(SEMANTIC_BRANCHES) - 1),
        "generated_clip_count": clip_count,
        "derived_same_video_negative_count": len(cells)
        * len(DERIVED_SAME_VIDEO_NEGATIVES),
        "critic_state_count_after_all_audits_pass": len(cells)
        * (len(SEMANTIC_BRANCHES) + len(DERIVED_SAME_VIDEO_NEGATIVES)),
        "paired_action_noop_hidden_queries_per_sigma": len(cells)
        * (len(SEMANTIC_BRANCHES) + len(DERIVED_SAME_VIDEO_NEGATIVES))
        * 2,
        "sp4_generated_clip_counts": group_clip_counts,
        "requested_gpu_count": 8,
        "concurrent_sequence_parallel_groups": 2,
        "gpus_per_sequence_parallel_group": 4,
        "estimated_wall_minutes_at_reference_rate": estimated_wall_minutes,
        "estimated_eight_gpu_hours_at_reference_rate": estimated_wall_minutes
        * 8
        / 60,
        "reference_cost": dict(REFERENCE_COST),
        "inherited_clips_are_reused_not_regenerated": True,
        "compact_population_is_pilot_only": True,
        "scientific_claim_authorized": False,
        "editor_optimizer_authorized": False,
        "cost_is_training_or_success_authority": False,
    }
    validate_composition_plan(composition)
    validate_topup_authoring(authoring, composition)
    validate_audit_request_manifest(audit_requests)
    return {
        "authoring": authoring,
        "composition": composition,
        "audit_requests": audit_requests,
        "cost": cost,
    }


def validate_topup_authoring(value: Any, composition_value: Any) -> dict[str, Any]:
    """Validate a nonredundant fragment against its inherited composition."""

    root = _closed(value, _TOPUP_AUTHORING_FIELDS, label="topup authoring")
    composition = validate_composition_plan(composition_value)
    if (
        root["schema_version"] != TOPUP_AUTHORING_SCHEMA
        or root["registry_id"] != composition["registry_id"]
        or root["stage_id"] != composition["stage_id"]
        or root["semantic_branch_order"] != list(SEMANTIC_BRANCHES)
        or root["geometry_contract"] != pair_contract.GEOMETRY_CONTRACT
        or root["geometry_source_pixels_enter_transformer"] is not False
        or root["geometry_source_vae_latent_created"] is not False
        or root["generated_media_editor_use_authorized"] is not False
    ):
        raise MosaicEventPopulationError("topup authoring contract differs")
    _safe_id(root["bank_id"], label="topup bank_id")
    cells = root["cells"]
    if (
        not isinstance(cells, list)
        or type(root["expected_cell_count"]) is not int
        or len(cells) != root["expected_cell_count"]
        or root["expected_candidate_count"] != len(cells) * len(SEMANTIC_BRANCHES)
        or len(cells) != composition["new_topup_cell_count"]
    ):
        raise MosaicEventPopulationError("topup authoring count differs")
    expected_by_cell = {
        row["cell_id"]: row for row in composition["new_topup_cells"]
    }
    seen_iids: set[str] = set()
    seen_seeds: set[int] = set()
    geometry_paths: set[str] = set()
    group_counts = Counter()
    normalized_cells = []
    for index, raw_cell in enumerate(cells):
        cell = _closed(raw_cell, _TOPUP_CELL_FIELDS, label=f"topup cell[{index}]")
        iid = _safe_id(cell["iid"], label="topup IID")
        if iid in seen_iids:
            raise MosaicEventPopulationError("topup IIDs repeat")
        seen_iids.add(iid)
        seed = cell["seed"]
        if type(seed) is not int or not 0 <= seed < 2**63 or seed in seen_seeds:
            raise MosaicEventPopulationError("topup seeds must be unique valid integers")
        seen_seeds.add(seed)
        cell_id = f"cell-{iid}-s{seed}"
        expected = expected_by_cell.get(cell_id)
        if expected is None:
            raise MosaicEventPopulationError("topup cell is absent from composition")
        for key, expected_key in (
            ("analysis_split", "analysis_split"),
            ("action_family_id", "atomic_action_family_id"),
            ("actor_group_id", "actor_group_id"),
            ("scene_group_id", "scene_group_id"),
            ("action_group_id", "action_group_id"),
            ("seed", "seed"),
        ):
            if cell[key] != expected[expected_key]:
                raise MosaicEventPopulationError(f"topup {key} composition binding differs")
        if cell["analysis_split"] not in ANALYSIS_SPLITS:
            raise MosaicEventPopulationError("topup split differs")
        for name in ("action_family_id", "actor_group_id", "scene_group_id", "action_group_id"):
            _safe_id(cell[name], label=name)
        execution_group = cell["execution_group"]
        if execution_group not in ("sp4-a", "sp4-b"):
            raise MosaicEventPopulationError("topup execution group differs")
        group_counts[execution_group] += 1
        geometry_paths.add(
            _absolute_path_text(cell["geometry_source_video"], label="geometry source")
        )
        descriptions = _closed(
            cell["branch_descriptions"],
            frozenset(SEMANTIC_BRANCHES),
            label="topup branch descriptions",
        )
        _text(cell["scene_caption"], label="topup scene caption", minimum_words=18)
        _text(cell["camera_caption"], label="topup camera caption", minimum_words=10)
        for branch in SEMANTIC_BRANCHES:
            _text(descriptions[branch], label=f"topup {branch}", minimum_words=10)
        normalized_cells.append(dict(cell))
    if set(expected_by_cell) != {f"cell-{cell['iid']}-s{cell['seed']}" for cell in cells}:
        raise MosaicEventPopulationError("topup composition cells are incomplete")
    if abs(group_counts["sp4-a"] - group_counts["sp4-b"]) > 1:
        raise MosaicEventPopulationError("topup dual-SP4 work is imbalanced")
    if len(geometry_paths) != 1:
        raise MosaicEventPopulationError(
            "all topup cells must reuse exactly one bucket-shape geometry probe"
        )
    return {**dict(root), "cells": normalized_cells}


def validate_composition_plan(value: Any) -> dict[str, Any]:
    root = _closed(value, _COMPOSITION_FIELDS, label="composition plan")
    unsigned = dict(root)
    digest = _sha256(
        unsigned.pop("composition_plan_digest"), label="composition plan digest"
    )
    if object_sha256(unsigned) != digest:
        raise MosaicEventPopulationError("composition plan digest differs")
    if (
        root["schema_version"] != COMPOSITION_SCHEMA
        or root["stage_id"] not in STAGE_IDS
        or root["inherited_bank_profiles"] != INHERITED_BANK_PROFILES
        or root["prompt_or_requested_branch_used_as_label"] is not False
        or root["event_audit_required_for_every_row"] is not True
        or root["compact_population_is_pilot_only"] is not True
        or root["scientific_claim_authorized"] is not False
        or root["editor_optimizer_authorized"] is not False
    ):
        raise MosaicEventPopulationError("composition authority differs")
    inherited = root["inherited_cells"]
    topup = root["new_topup_cells"]
    expected_topup = {
        "pilot_topup12": 12,
        "holdout_topup8": 8,
        "full_topup20": 20,
    }[root["stage_id"]]
    if (
        not isinstance(inherited, list)
        or not isinstance(topup, list)
        or len(inherited) != 16
        or root["inherited_cell_count"] != 16
        or root["inherited_candidate_count"] != 160
        or len(topup) != expected_topup
        or root["new_topup_cell_count"] != expected_topup
        or root["new_topup_candidate_count"] != expected_topup * 10
        or root["composed_cell_count_after_stage"] != 16 + expected_topup
        or root["composed_candidate_count_after_stage"] != (16 + expected_topup) * 10
        or root["full_compact_target_cell_count"] != 36
        or root["full_compact_target_candidate_count"] != 360
        or root["identity_scene_groups_per_family_at_full_target"] != 3
        or root["seeds_per_identity_scene_at_full_target"] != 2
    ):
        raise MosaicEventPopulationError("composition population counts differ")
    candidate_ids: list[str] = []
    cell_ids: set[str] = set()
    for cell in [*inherited, *topup]:
        if not isinstance(cell, Mapping):
            raise MosaicEventPopulationError("composition cell must be an object")
        cell_id = _safe_id(cell.get("cell_id"), label="composition cell_id")
        if cell_id in cell_ids:
            raise MosaicEventPopulationError("composition cell IDs repeat")
        cell_ids.add(cell_id)
        branches = cell.get("semantic_branch_order")
        ids = cell.get("candidate_ids")
        if (
            branches != list(SEMANTIC_BRANCHES)
            or not isinstance(ids, list)
            or len(ids) != len(SEMANTIC_BRANCHES)
            or cell.get("requested_branch_is_label") is not False
            or cell.get("detached_event_audit_required") is not True
        ):
            raise MosaicEventPopulationError("composition cell branch/audit closure differs")
        candidate_ids.extend(_safe_id(item, label="candidate_id") for item in ids)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise MosaicEventPopulationError("composition candidate IDs repeat")
    return dict(root)


def build_inherited_audit_requests_from_authenticated_rows(
    registry_value: Any,
    composition_value: Any,
    authenticated_bound_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Profile arbitrary core4/reserve/seed2 rows without prefix/count hacks.

    ``authenticated_bound_rows`` must come from the existing strong bank
    verifier.  This function verifies their exact candidate identity and
    population coordinates, atomically maps the reserve bank's historical
    ``articulated-pose-transition`` bucket to its sealed event family, and
    emits null-label audit requests.  It never copies ``semantic_branch`` into
    a critic label.
    """

    registry = validate_registry(registry_value)
    composition = validate_composition_plan(composition_value)
    expected_cells = composition["inherited_cells"]
    expected_order = [
        (cell, branch, candidate_id)
        for cell in expected_cells
        for branch, candidate_id in zip(SEMANTIC_BRANCHES, cell["candidate_ids"])
    ]
    if (
        not isinstance(authenticated_bound_rows, Sequence)
        or isinstance(authenticated_bound_rows, (str, bytes, bytearray))
        or len(authenticated_bound_rows) != len(expected_order)
    ):
        raise MosaicEventPopulationError("inherited audit profile requires exact160 rows")
    bound_by_id: dict[str, Mapping[str, Any]] = {}
    for bound in authenticated_bound_rows:
        candidate = bound.get("candidate") if isinstance(bound, Mapping) else None
        candidate_id = candidate.get("candidate_id") if isinstance(candidate, Mapping) else None
        if not isinstance(candidate_id, str) or candidate_id in bound_by_id:
            raise MosaicEventPopulationError(
                "authenticated inherited candidate IDs are missing or repeated"
            )
        bound_by_id[candidate_id] = bound
    expected_ids = {candidate_id for _, _, candidate_id in expected_order}
    if set(bound_by_id) != expected_ids:
        raise MosaicEventPopulationError("authenticated inherited candidate set differs")
    family_by_id = {
        family["family_id"]: family for family in registry["action_families"]
    }
    audit_protocol_digest = object_sha256(AUDIT_POLICY)
    requests = []
    for ordinal, (cell, branch, candidate_id) in enumerate(expected_order):
        bound = bound_by_id[candidate_id]
        if not isinstance(bound, Mapping) or not isinstance(bound.get("candidate"), Mapping):
            raise MosaicEventPopulationError(f"bound row {ordinal} lacks candidate")
        candidate = bound["candidate"]
        expected_candidate_surface = {
            "candidate_id": candidate_id,
            "analysis_split": cell["analysis_split"],
            "action_family_id": cell["legacy_action_family_id"],
            "calibration_group_id": cell["cell_id"],
            "actor_group_id": cell["actor_group_id"],
            "scene_group_id": cell["scene_group_id"],
            "action_group_id": cell["action_group_id"],
            "semantic_branch": branch,
            "seed": cell["seed"],
        }
        if any(candidate.get(key) != value for key, value in expected_candidate_surface.items()):
            raise MosaicEventPopulationError(
                f"bound inherited candidate identity/order differs at ordinal {ordinal}"
            )
        caption = candidate.get("full_t2v_caption")
        caption_digest = candidate.get("full_t2v_caption_utf8_sha256")
        if (
            not isinstance(caption, str)
            or len(caption.split()) < 12
            or hashlib.sha256(caption.encode("utf-8")).hexdigest() != caption_digest
        ):
            raise MosaicEventPopulationError("bound inherited caption identity differs")
        family = family_by_id[cell["atomic_action_family_id"]]
        event_definition = dict(family["event_definition"])
        unsigned_request = {
            "candidate_id": candidate_id,
            "cell_id": cell["cell_id"],
            "analysis_split": cell["analysis_split"],
            "critic_partition": _critic_partition(
                family["population_role"], cell["analysis_split"]
            ),
            "action_family_id": family["family_id"],
            "population_role": family["population_role"],
            "actor_group_id": cell["actor_group_id"],
            "scene_group_id": cell["scene_group_id"],
            "action_group_id": cell["action_group_id"],
            "seed": cell["seed"],
            "requested_semantic_branch": branch,
            "full_t2v_caption_utf8_sha256": caption_digest,
            "event_definition": event_definition,
            "event_definition_sha256": object_sha256(event_definition),
            "audit_protocol_sha256": audit_protocol_digest,
            "audit_sidecar_basename": f"{candidate_id}.event-audit.json",
            "authenticated_generation_receipt_digest": _sha256(
                bound.get("generation_receipt_digest"),
                label="authenticated generation receipt digest",
            ),
            "candidate_eligible_before_audit": False,
            "critic_label_before_audit": None,
        }
        requests.append(
            {**unsigned_request, "request_digest": object_sha256(unsigned_request)}
        )
    root_unsigned = {
        "schema_version": AUDIT_REQUEST_SCHEMA,
        "registry_id": registry["registry_id"],
        "stage_id": composition["stage_id"],
        "pair_v5_bank_id": "mosaic-inherited-core4-reserve4",
        "authoring_object_sha256": composition["composition_plan_digest"],
        "audit_policy": dict(AUDIT_POLICY),
        "audit_protocol_sha256": audit_protocol_digest,
        "candidate_count": len(requests),
        "candidate_requests": requests,
        "generated_media_editor_use_authorized": False,
        "critic_feature_extraction_authorized_before_audits": False,
    }
    return {**root_unsigned, "manifest_digest": object_sha256(root_unsigned)}


def merge_composed_audit_request_manifests(
    composition_value: Any,
    inherited_manifest_value: Any,
    topup_manifest_value: Any,
) -> dict[str, Any]:
    """Merge authenticated inherited and new requests in composition order."""

    composition = validate_composition_plan(composition_value)
    inherited = validate_audit_request_manifest(inherited_manifest_value)
    topup = validate_audit_request_manifest(topup_manifest_value)
    if (
        inherited["registry_id"] != composition["registry_id"]
        or topup["registry_id"] != composition["registry_id"]
        or inherited["stage_id"] != composition["stage_id"]
        or topup["stage_id"] != composition["stage_id"]
    ):
        raise MosaicEventPopulationError("audit manifests belong to another composition")
    inherited_by_id = {
        row["candidate_id"]: row for row in inherited["candidate_requests"]
    }
    topup_by_id = {row["candidate_id"]: row for row in topup["candidate_requests"]}
    expected_inherited = [
        candidate_id
        for cell in composition["inherited_cells"]
        for candidate_id in cell["candidate_ids"]
    ]
    expected_topup = [
        candidate_id
        for cell in composition["new_topup_cells"]
        for candidate_id in cell["candidate_ids"]
    ]
    if set(inherited_by_id) != set(expected_inherited):
        raise MosaicEventPopulationError("inherited audit request set differs")
    if set(topup_by_id) != set(expected_topup):
        raise MosaicEventPopulationError("topup audit request set differs")
    rows = [
        *(inherited_by_id[candidate_id] for candidate_id in expected_inherited),
        *(topup_by_id[candidate_id] for candidate_id in expected_topup),
    ]
    unsigned = {
        "schema_version": AUDIT_REQUEST_SCHEMA,
        "registry_id": composition["registry_id"],
        "stage_id": composition["stage_id"],
        "pair_v5_bank_id": "mosaic-composed-compact6",
        "authoring_object_sha256": composition["composition_plan_digest"],
        "audit_policy": dict(AUDIT_POLICY),
        "audit_protocol_sha256": object_sha256(AUDIT_POLICY),
        "candidate_count": len(rows),
        "candidate_requests": rows,
        "generated_media_editor_use_authorized": False,
        "critic_feature_extraction_authorized_before_audits": False,
    }
    merged = {**unsigned, "manifest_digest": object_sha256(unsigned)}
    validate_audit_request_manifest(merged)
    return merged


def bind_audit_requests_to_authenticated_rows(
    provisional_manifest_value: Any,
    authenticated_bound_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach strong generation-receipt identities after a bank audit.

    The provisional request seals prompts and rubrics before rendering but is
    intentionally ineligible.  This second seal requires the exact candidate
    set returned by a strong bank verifier and binds each generation receipt
    digest before any event sidecar can authorize feature extraction.
    """

    manifest = validate_audit_request_manifest(provisional_manifest_value)
    if (
        not isinstance(authenticated_bound_rows, Sequence)
        or isinstance(authenticated_bound_rows, (str, bytes, bytearray))
        or len(authenticated_bound_rows) != manifest["candidate_count"]
    ):
        raise MosaicEventPopulationError(
            "strong request binding requires the exact authenticated candidate population"
        )
    bound_by_id: dict[str, Mapping[str, Any]] = {}
    for bound in authenticated_bound_rows:
        candidate = bound.get("candidate") if isinstance(bound, Mapping) else None
        candidate_id = candidate.get("candidate_id") if isinstance(candidate, Mapping) else None
        if not isinstance(candidate_id, str) or candidate_id in bound_by_id:
            raise MosaicEventPopulationError(
                "authenticated candidate IDs are missing or repeated"
            )
        bound_by_id[candidate_id] = bound
    expected_ids = {row["candidate_id"] for row in manifest["candidate_requests"]}
    if set(bound_by_id) != expected_ids:
        raise MosaicEventPopulationError("authenticated candidate set differs")
    rows = []
    for request in manifest["candidate_requests"]:
        bound = bound_by_id[request["candidate_id"]]
        candidate = bound["candidate"]
        expected = {
            "candidate_id": request["candidate_id"],
            "analysis_split": request["analysis_split"],
            "action_family_id": request["action_family_id"],
            "calibration_group_id": request["cell_id"],
            "actor_group_id": request["actor_group_id"],
            "scene_group_id": request["scene_group_id"],
            "action_group_id": request["action_group_id"],
            "semantic_branch": request["requested_semantic_branch"],
            "seed": request["seed"],
            "full_t2v_caption_utf8_sha256": request[
                "full_t2v_caption_utf8_sha256"
            ],
        }
        if any(candidate.get(key) != value for key, value in expected.items()):
            raise MosaicEventPopulationError(
                f"authenticated candidate surface differs: {request['candidate_id']}"
            )
        generation_digest = _sha256(
            bound.get("generation_receipt_digest"),
            label="authenticated generation receipt digest",
        )
        prior = request["authenticated_generation_receipt_digest"]
        if prior is not None and prior != generation_digest:
            raise MosaicEventPopulationError(
                "request was already bound to another generation receipt"
            )
        unsigned = dict(request)
        unsigned.pop("request_digest")
        unsigned["authenticated_generation_receipt_digest"] = generation_digest
        rows.append({**unsigned, "request_digest": object_sha256(unsigned)})
    root_unsigned = dict(manifest)
    root_unsigned.pop("manifest_digest")
    root_unsigned["candidate_requests"] = rows
    result = {**root_unsigned, "manifest_digest": object_sha256(root_unsigned)}
    validate_audit_request_manifest(result)
    return result


def validate_audit_request_manifest(value: Any) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "registry_id",
        "stage_id",
        "pair_v5_bank_id",
        "authoring_object_sha256",
        "audit_policy",
        "audit_protocol_sha256",
        "candidate_count",
        "candidate_requests",
        "generated_media_editor_use_authorized",
        "critic_feature_extraction_authorized_before_audits",
        "manifest_digest",
    }
    root = _closed(value, frozenset(expected_fields), label="audit request manifest")
    unsigned = dict(root)
    digest = _sha256(unsigned.pop("manifest_digest"), label="manifest_digest")
    if object_sha256(unsigned) != digest:
        raise MosaicEventPopulationError("audit request manifest digest differs")
    if (
        root["schema_version"] != AUDIT_REQUEST_SCHEMA
        or root["stage_id"] not in STAGE_IDS
        or root["audit_policy"] != AUDIT_POLICY
        or root["audit_protocol_sha256"] != object_sha256(AUDIT_POLICY)
        or root["generated_media_editor_use_authorized"] is not False
        or root["critic_feature_extraction_authorized_before_audits"] is not False
    ):
        raise MosaicEventPopulationError("audit request manifest contract differs")
    _safe_id(root["registry_id"], label="registry_id")
    _safe_id(root["pair_v5_bank_id"], label="pair_v5_bank_id")
    _sha256(root["authoring_object_sha256"], label="authoring_object_sha256")
    rows = root["candidate_requests"]
    if (
        type(root["candidate_count"]) is not int
        or not isinstance(rows, list)
        or len(rows) != root["candidate_count"]
    ):
        raise MosaicEventPopulationError("audit request candidate count differs")
    seen: set[str] = set()
    normalized_rows = []
    for index, raw_row in enumerate(rows):
        row = _closed(
            raw_row, _AUDIT_REQUEST_FIELDS, label=f"audit request[{index}]"
        )
        unsigned_row = dict(row)
        request_digest = _sha256(
            unsigned_row.pop("request_digest"), label="request_digest"
        )
        if object_sha256(unsigned_row) != request_digest:
            raise MosaicEventPopulationError("audit request digest differs")
        candidate_id = _safe_id(row["candidate_id"], label="candidate_id")
        if candidate_id in seen:
            raise MosaicEventPopulationError("audit request candidate IDs repeat")
        seen.add(candidate_id)
        for name in (
            "cell_id",
            "action_family_id",
            "actor_group_id",
            "scene_group_id",
            "action_group_id",
        ):
            _safe_id(row[name], label=name)
        if (
            row["analysis_split"] not in ANALYSIS_SPLITS
            or row["population_role"] not in FAMILY_ROLES
            or row["requested_semantic_branch"] not in SEMANTIC_BRANCHES
            or row["candidate_eligible_before_audit"] is not False
            or row["critic_label_before_audit"] is not None
            or row["audit_protocol_sha256"] != object_sha256(AUDIT_POLICY)
        ):
            raise MosaicEventPopulationError("audit request label/role contract differs")
        _sha256(
            row["full_t2v_caption_utf8_sha256"], label="caption SHA-256"
        )
        authenticated_receipt = row["authenticated_generation_receipt_digest"]
        if authenticated_receipt is not None:
            _sha256(
                authenticated_receipt,
                label="authenticated generation receipt digest",
            )
        if object_sha256(row["event_definition"]) != row["event_definition_sha256"]:
            raise MosaicEventPopulationError("event definition digest differs")
        if row["audit_sidecar_basename"] != f"{candidate_id}.event-audit.json":
            raise MosaicEventPopulationError("audit sidecar basename differs")
        normalized_rows.append(dict(row))
    return {**dict(root), "candidate_requests": normalized_rows}


def _load_and_verify_receipt(
    path: Path, expected_raw: str, expected_digest: str
) -> dict[str, Any]:
    if _file_sha256(path) != expected_raw:
        raise MosaicEventPopulationError("generation receipt raw SHA-256 differs")
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MosaicEventPopulationError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MosaicEventPopulationError("generation receipt is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise MosaicEventPopulationError("generation receipt must be an object")
    unsigned = dict(value)
    declared = _sha256(unsigned.pop("receipt_digest", None), label="receipt digest")
    if (
        declared != expected_digest
        or pair_contract.sha256_bytes(pair_contract.canonical_json_bytes(unsigned))
        != declared
    ):
        raise MosaicEventPopulationError("generation receipt digest differs")
    return dict(value)


def _frame_list(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(set(value)) != len(value)
        or any(type(frame) is not int or not 0 <= frame < 81 for frame in value)
    ):
        raise MosaicEventPopulationError(f"{label} must contain unique exact81 frames")
    return list(value)


def _logical_audit_acceptance(
    request: Mapping[str, Any], checks: Mapping[str, Any], evidence: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    branch = request["requested_semantic_branch"]
    failures: list[str] = []
    for name in (
        "video_quality_pass",
        "continuous_no_cut",
        "primary_actor_trackable",
        "requested_branch_mechanism_observed",
    ):
        if checks[name] is not True:
            failures.append(name)
    if not evidence["start_frames"] or not evidence["branch_mechanism_frames"]:
        failures.append("missing_frame_evidence")
    if branch == "action":
        for name in (
            "family_start_state_observed",
            "family_transition_observed",
            "family_terminal_state_observed",
            "family_terminal_hold_observed",
            "full_target_event_observed",
        ):
            if checks[name] is not True:
                failures.append(name)
        if checks["full_target_event_false_confirmed"] is not False:
            failures.append("positive_false_confirmation_conflict")
        if checks["actor_binding"] != "primary":
            failures.append("positive_actor_binding")
        event = request["event_definition"]
        expected_objects = (
            {"target"} if event["object_binding_required"] else {"not_applicable", "target"}
        )
        if checks["object_binding"] not in expected_objects:
            failures.append("positive_object_binding")
        if checks["camera_class"] != "locked_or_natural":
            failures.append("positive_camera")
        if checks["appearance_only_observed"] is not False:
            failures.append("positive_appearance_conflict")
        temporal = [
            evidence["start_frames"],
            evidence["transition_frames"],
            evidence["terminal_frames"],
            evidence["terminal_hold_frames"],
        ]
        if any(not frames for frames in temporal):
            failures.append("positive_temporal_evidence_missing")
        elif not (
            max(temporal[0]) < max(temporal[1]) <= min(temporal[2])
            <= min(temporal[3])
        ):
            failures.append("positive_temporal_evidence_order")
        hold_span = (
            max(evidence["terminal_hold_frames"])
            - min(evidence["terminal_hold_frames"])
            + 1
        ) if evidence["terminal_hold_frames"] else 0
        if hold_span < request["event_definition"]["minimum_terminal_hold_frames"]:
            failures.append("positive_terminal_hold_too_short")
    else:
        if checks["full_target_event_observed"] is not False:
            failures.append("negative_contains_target_event")
        if checks["full_target_event_false_confirmed"] is not True:
            failures.append("negative_not_false_confirmed")
        if all(
            checks[name] is True
            for name in (
                "family_start_state_observed",
                "family_transition_observed",
                "family_terminal_state_observed",
                "family_terminal_hold_observed",
            )
        ):
            failures.append("negative_has_full_event_conjunction")
        if branch == "wrong_actor" and checks["actor_binding"] != "secondary":
            failures.append("wrong_actor_mechanism")
        if branch == "wrong_object" and checks["object_binding"] != "distractor":
            failures.append("wrong_object_mechanism")
        if branch == "camera_only" and checks["camera_class"] != "camera_only_motion":
            failures.append("camera_only_mechanism")
        if branch != "camera_only" and checks["camera_class"] != "locked_or_natural":
            failures.append("negative_camera_confounded")
        if branch == "appearance_only" and checks["appearance_only_observed"] is not True:
            failures.append("appearance_only_mechanism")
        if branch != "appearance_only" and checks["appearance_only_observed"] is not False:
            failures.append("negative_appearance_confounded")
    return not failures, failures


def validate_event_audit(value: Any, request_value: Any) -> dict[str, Any]:
    """Authenticate one detached audit and independently recompute eligibility."""

    request = _closed(request_value, _AUDIT_REQUEST_FIELDS, label="audit request")
    # Reuse the manifest row's own seal even when called independently.
    unsigned_request = dict(request)
    request_digest = _sha256(
        unsigned_request.pop("request_digest"), label="request_digest"
    )
    if object_sha256(unsigned_request) != request_digest:
        raise MosaicEventPopulationError("audit request digest differs")
    row = _closed(value, _AUDIT_SIDECAR_FIELDS, label="event audit sidecar")
    unsigned = dict(row)
    sidecar_digest = _sha256(
        unsigned.pop("sidecar_digest"), label="event audit sidecar digest"
    )
    if object_sha256(unsigned) != sidecar_digest:
        raise MosaicEventPopulationError("event audit sidecar digest differs")
    if (
        row["schema_version"] != AUDIT_SIDECAR_SCHEMA
        or row["request_digest"] != request_digest
        or row["candidate_id"] != request["candidate_id"]
        or row["audit_protocol_sha256"] != request["audit_protocol_sha256"]
    ):
        raise MosaicEventPopulationError("event audit request binding differs")
    media = _plain_file(row["rendered_media_path"], label="rendered media")
    evidence_path = _plain_file(row["audit_evidence_path"], label="audit evidence")
    receipt_path = _plain_file(
        row["generation_receipt_path"], label="generation receipt"
    )
    for name in (
        "rendered_media_sha256",
        "generation_receipt_sha256",
        "generation_receipt_digest",
        "audit_evidence_sha256",
        "audit_protocol_sha256",
    ):
        _sha256(row[name], label=name)
    if _file_sha256(media) != row["rendered_media_sha256"]:
        raise MosaicEventPopulationError("rendered media SHA-256 differs")
    if _file_sha256(evidence_path) != row["audit_evidence_sha256"]:
        raise MosaicEventPopulationError("audit evidence SHA-256 differs")
    generation_receipt = _load_and_verify_receipt(
        receipt_path,
        row["generation_receipt_sha256"],
        row["generation_receipt_digest"],
    )
    receipt_candidate = generation_receipt.get("candidate")
    receipt_artifacts = generation_receipt.get("artifacts")
    receipt_mp4 = (
        receipt_artifacts.get("mp4")
        if isinstance(receipt_artifacts, Mapping)
        else None
    )
    if (
        not isinstance(receipt_candidate, Mapping)
        or receipt_candidate.get("candidate_id") != request["candidate_id"]
        or receipt_candidate.get("semantic_branch")
        != request["requested_semantic_branch"]
        or receipt_candidate.get("full_t2v_caption_utf8_sha256")
        != request["full_t2v_caption_utf8_sha256"]
        or not isinstance(receipt_mp4, Mapping)
        or receipt_mp4.get("path") != str(media)
        or receipt_mp4.get("sha256") != row["rendered_media_sha256"]
    ):
        raise MosaicEventPopulationError(
            "generation receipt does not bind the requested candidate and review media"
        )
    _safe_id(row["auditor_id"], label="auditor_id")
    if row["auditor_method"] not in AUDIT_POLICY["allowed_auditor_methods"]:
        raise MosaicEventPopulationError("auditor method differs")
    if (
        row["generation_prompt_or_requested_branch_disclosed"] is not False
        or row["entire_video_viewed"] is not True
        or row["hidden_feature_extraction_started_before_audit"] is not False
    ):
        raise MosaicEventPopulationError("event audit was not detached and prompt-blind")
    try:
        audited_at = datetime.fromisoformat(row["audited_at_utc"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise MosaicEventPopulationError("audited_at_utc is invalid") from error
    if not row["audited_at_utc"].endswith("Z") or audited_at.tzinfo != timezone.utc:
        raise MosaicEventPopulationError("audited_at_utc must be explicit UTC Z time")
    if row["observed_class"] not in _OBSERVED_CLASSES:
        raise MosaicEventPopulationError("observed event class differs")
    checks = _closed(row["checks"], _AUDIT_CHECK_FIELDS, label="audit checks")
    for name in _AUDIT_CHECK_FIELDS - {
        "actor_binding",
        "object_binding",
        "camera_class",
    }:
        if type(checks[name]) is not bool:
            raise MosaicEventPopulationError(f"audit check {name} must be bool")
    if (
        checks["actor_binding"] not in _ACTOR_BINDINGS
        or checks["object_binding"] not in _OBJECT_BINDINGS
        or checks["camera_class"] not in _CAMERA_CLASSES
    ):
        raise MosaicEventPopulationError("audit categorical check differs")
    evidence = _closed(row["evidence"], _AUDIT_EVIDENCE_FIELDS, label="audit evidence")
    normalized_evidence = {
        name: _frame_list(evidence[name], label=name)
        for name in (
            "start_frames",
            "transition_frames",
            "terminal_frames",
            "terminal_hold_frames",
            "branch_mechanism_frames",
        )
    }
    normalized_evidence["written_observation"] = _text(
        evidence["written_observation"],
        label="written_observation",
        minimum_words=12,
    )
    logical_accept, logical_failures = _logical_audit_acceptance(
        request, checks, normalized_evidence
    )
    observed_matches = row["observed_class"] == request["requested_semantic_branch"]
    expected_accept = logical_accept and observed_matches
    if row["eligibility_decision"] not in ("accept", "reject"):
        raise MosaicEventPopulationError("eligibility decision differs")
    rejection_reasons = row["rejection_reasons"]
    if (
        not isinstance(rejection_reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in rejection_reasons)
        or len(rejection_reasons) != len(set(rejection_reasons))
    ):
        raise MosaicEventPopulationError("rejection reasons differ")
    if row["eligibility_decision"] == "accept":
        if not expected_accept or rejection_reasons:
            raise MosaicEventPopulationError(
                "declared audit acceptance does not satisfy the blinded rubric"
            )
    else:
        if expected_accept or not rejection_reasons:
            raise MosaicEventPopulationError(
                "declared audit rejection is inconsistent or unexplained"
            )
    return {
        **dict(row),
        "checks": checks,
        "evidence": normalized_evidence,
        "candidate_eligible": row["eligibility_decision"] == "accept",
        "critic_label": (
            1
            if row["eligibility_decision"] == "accept"
            and request["requested_semantic_branch"] == "action"
            else 0 if row["eligibility_decision"] == "accept" else None
        ),
        "recomputed_failure_reasons": sorted(
            set(
                logical_failures
                + ([] if observed_matches else ["observed_class_mismatch"])
            )
        ),
    }


def build_eligibility_index(
    request_manifest_value: Any,
    audit_sidecars_by_candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Create a fail-closed cell index; missing audits never imply negatives."""

    manifest = validate_audit_request_manifest(request_manifest_value)
    if not isinstance(audit_sidecars_by_candidate, Mapping):
        raise MosaicEventPopulationError("audit_sidecars_by_candidate must be a mapping")
    expected_ids = {row["candidate_id"] for row in manifest["candidate_requests"]}
    unknown = set(audit_sidecars_by_candidate) - expected_ids
    if unknown:
        raise MosaicEventPopulationError(f"unknown audit candidates: {sorted(unknown)!r}")
    rows = []
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in manifest["candidate_requests"]:
        sidecar = audit_sidecars_by_candidate.get(request["candidate_id"])
        if sidecar is None:
            result = {
                "candidate_id": request["candidate_id"],
                "cell_id": request["cell_id"],
                "requested_semantic_branch": request["requested_semantic_branch"],
                "audit_state": "missing",
                "candidate_eligible": False,
                "critic_label": None,
                "audit_sidecar_digest": None,
            }
        else:
            audit = validate_event_audit(sidecar, request)
            authenticated_digest = request[
                "authenticated_generation_receipt_digest"
            ]
            if (
                authenticated_digest is not None
                and authenticated_digest != audit["generation_receipt_digest"]
            ):
                raise MosaicEventPopulationError(
                    "detached audit belongs to a different strongly authenticated generation receipt"
                )
            bank_authenticated = authenticated_digest is not None
            candidate_eligible = audit["candidate_eligible"] and bank_authenticated
            result = {
                "candidate_id": request["candidate_id"],
                "cell_id": request["cell_id"],
                "requested_semantic_branch": request["requested_semantic_branch"],
                "audit_state": (
                    audit["eligibility_decision"]
                    if bank_authenticated
                    else "bank_authentication_missing"
                ),
                "candidate_eligible": candidate_eligible,
                "critic_label": audit["critic_label"] if candidate_eligible else None,
                "audit_sidecar_digest": audit["sidecar_digest"],
            }
        rows.append(result)
        by_cell[request["cell_id"]].append(result)
    eligible_cells = []
    for cell_id, cell_rows in by_cell.items():
        branches = [row["requested_semantic_branch"] for row in cell_rows]
        if branches != list(SEMANTIC_BRANCHES):
            raise MosaicEventPopulationError(f"cell {cell_id} branch order differs")
        if all(row["candidate_eligible"] for row in cell_rows):
            eligible_cells.append(cell_id)
    eligible_cell_set = set(eligible_cells)
    authorized_candidates = [
        row["candidate_id"] for row in rows if row["cell_id"] in eligible_cell_set
    ]
    unsigned = {
        "schema_version": ELIGIBILITY_SCHEMA,
        "audit_request_manifest_digest": manifest["manifest_digest"],
        "expected_candidate_count": manifest["candidate_count"],
        "received_audit_count": len(audit_sidecars_by_candidate),
        "accepted_candidate_count": sum(row["candidate_eligible"] for row in rows),
        "strongly_bank_authenticated_request_count": sum(
            request["authenticated_generation_receipt_digest"] is not None
            for request in manifest["candidate_requests"]
        ),
        "audit_accept_but_bank_authentication_missing_count": sum(
            row["audit_state"] == "bank_authentication_missing" for row in rows
        ),
        "rejected_candidate_count": sum(row["audit_state"] == "reject" for row in rows),
        "missing_candidate_count": sum(row["audit_state"] == "missing" for row in rows),
        "eligible_cell_ids": eligible_cells,
        "feature_extraction_authorized_candidate_ids": authorized_candidates,
        "candidate_rows": rows,
        "prompt_branch_used_as_label": False,
        "strong_bank_authentication_required_before_eligibility": True,
        "generated_media_editor_use_authorized": False,
        "editor_optimizer_authorized": False,
    }
    return {**unsigned, "index_digest": object_sha256(unsigned)}


def write_stage_bundle(
    *, registry_path: str | Path, expected_registry_sha256: str, stage_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    registry, registry_digest = load_sealed_registry(
        registry_path, expected_registry_sha256
    )
    bundle = build_stage_bundle(registry, stage_id=stage_id)
    output = Path(output_dir)
    if (
        not output.is_absolute()
        or output == Path("/")
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
    ):
        raise MosaicEventPopulationError(
            "output_dir must be a fresh absolute directory in a plain parent"
        )
    output.mkdir()
    paths = {
        "topup_authoring": output / "mosaic-topup-authoring.json",
        "composition": output / "population-composition-plan.json",
        "audit_requests": output / "event-audit-requests.json",
        "cost": output / "population-cost.json",
    }
    hashes = {}
    for name, path in paths.items():
        raw = canonical_json_bytes(
            bundle[name if name != "topup_authoring" else "authoring"]
        ) + b"\n"
        path.write_bytes(raw)
        path.chmod(0o400)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {
        "registry_raw_sha256": registry_digest,
        "stage_id": stage_id,
        "output_dir": str(output),
        "artifact_paths": {name: str(path) for name, path in paths.items()},
        "artifact_raw_sha256": hashes,
        "generated_media_editor_use_authorized": False,
        "critic_feature_extraction_authorized_before_audits": False,
    }


__all__ = [
    "ANALYSIS_SPLITS",
    "AUDIT_POLICY",
    "AUDIT_REQUEST_SCHEMA",
    "AUDIT_SIDECAR_SCHEMA",
    "AUTHORING_CONTRACT",
    "COMPOSITION_SCHEMA",
    "COST_SCHEMA",
    "DERIVED_SAME_VIDEO_NEGATIVES",
    "ELIGIBILITY_SCHEMA",
    "FAMILY_ROLES",
    "MosaicEventPopulationError",
    "REFERENCE_COST",
    "REGISTRY_SCHEMA",
    "SEMANTIC_BRANCHES",
    "STAGE_IDS",
    "STAGE_POLICY",
    "TOPUP_AUTHORING_SCHEMA",
    "bind_audit_requests_to_authenticated_rows",
    "build_eligibility_index",
    "build_inherited_audit_requests_from_authenticated_rows",
    "build_stage_bundle",
    "canonical_json_bytes",
    "load_sealed_registry",
    "merge_composed_audit_request_manifests",
    "object_sha256",
    "validate_audit_request_manifest",
    "validate_composition_plan",
    "validate_event_audit",
    "validate_registry",
    "validate_topup_authoring",
    "write_stage_bundle",
]

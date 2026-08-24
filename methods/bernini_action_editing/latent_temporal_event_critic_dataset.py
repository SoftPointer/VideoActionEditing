#!/usr/bin/env python3
"""Fail-closed data contracts for a Bernini latent temporal event critic.

Generated T2V clips are *state owners* for critic supervision.  They are never
an RGB/latent target, donor, condition, reference, or initial noise for the
RV2V editor.  Every owner is queried twice at the same noisy latent state by a
frozen Bernini model (target action versus scene-matched no-op).  A small head
may then learn from the resulting hidden-state residual.

This module is standard-library only.  It validates provenance, event labels,
episode closure, split isolation, and pilot/scientific decision authority.  It
does not open tensors, instantiate an optimizer, or authorize an editor.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-latent-temporal-event-critic-dataset-v1"
EPISODE_SCHEMA = "bernini-latent-temporal-event-critic-episode-plan-v1"
USAGE_AUTHORITY_SCHEMA = "bernini-self-generated-t2v-critic-only-use-v1"
HIDDEN_QUERY_SCHEMA = "bernini-same-state-action-noop-hidden-query-v1"
PILOT_AUDIT_SCHEMA = "bernini-core4-hidden-event-critic-pilot-audit-v1"
POPULATION_AUDIT_SCHEMA = "bernini-hidden-event-critic-population-audit-v1"

SOURCE_KIND = "frozen_bernini_self_generated_t2v"
FRAME_COUNT = 81
LATENT_PHASES = 21
LATENT_CHANNELS = 16
HIDDEN_SIZE = 1536
LATENT_SPATIAL_PATCH_SIZE = (2, 2)
CORE4_NATIVE_LATENT_SHAPES = (
    (1, 16, 21, 60, 62),
    (1, 16, 21, 64, 58),
    (1, 16, 21, 68, 54),
)
PILOT_SPLITS = ("fit", "confirmation")
SCIENTIFIC_SPLITS = ("train", "validation", "test")

ACTION_BRANCH = "action"
SEMANTIC_NEGATIVE_BRANCHES = (
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)
SEMANTIC_BRANCHES = (ACTION_BRANCH, *SEMANTIC_NEGATIVE_BRANCHES)

DERIVED_NEGATIVE_ROLES = (
    "same_video_reverse",
    "same_video_freeze_first",
    "same_video_phase_shuffle",
)
SEMANTIC_NEGATIVE_ROLES = tuple(
    f"semantic_{branch}" for branch in SEMANTIC_NEGATIVE_BRANCHES
)
NEGATIVE_ROLES = (*DERIVED_NEGATIVE_ROLES, *SEMANTIC_NEGATIVE_ROLES)
ARM_ROLES = ("positive", *NEGATIVE_ROLES)

TEMPORAL_TRANSFORM_BY_ROLE = {
    "positive": "chronological",
    "same_video_reverse": "reverse",
    "same_video_freeze_first": "freeze_first",
    "same_video_phase_shuffle": "phase_shuffle",
    **{role: "chronological" for role in SEMANTIC_NEGATIVE_ROLES},
}
SOURCE_BRANCH_BY_ROLE = {
    "positive": ACTION_BRANCH,
    **{role: ACTION_BRANCH for role in DERIVED_NEGATIVE_ROLES},
    **{
        f"semantic_{branch}": branch
        for branch in SEMANTIC_NEGATIVE_BRANCHES
    },
}

# The first pilot uses one coordinate fixed before hidden materialization.  A
# later coordinate scan is a new registered experiment, not an implicit knob.
PILOT_HIDDEN_QUERY = {
    "schema_version": HIDDEN_QUERY_SCHEMA,
    "condition_mode": "t2v_same_state_target_tail",
    "native_schedule_index": 33,
    "sigma": 0.5161304473876953,
    "native_timestep": 516,
    "hook_coordinate": "block.15.output",
    "hidden_size": HIDDEN_SIZE,
    "latent_phases": LATENT_PHASES,
    "native_geometry_mode": (
        "derive_per_episode_from_authenticated_exact81_clean_latent"
    ),
    "spatial_patch_size_height_width": list(LATENT_SPATIAL_PATCH_SIZE),
    "patch_positions_are_episode_specific": True,
    "action_and_noop_share_exact_x_sigma": True,
    "action_and_noop_share_exact_noise": True,
    "source_condition_consumed": False,
    "mask_flow_pose_track_consumed": False,
    "frozen_checkpoint_and_adapter_off_required": True,
}

CRITIC_ONLY_USE = {
    "generated_clean_latent_may_create_frozen_hidden_queries": True,
    "generated_hidden_may_train_critic_head": True,
    "generated_rgb_or_latent_may_train_editor": False,
    "generated_rgb_or_latent_may_be_editor_target": False,
    "generated_rgb_or_latent_may_be_editor_condition": False,
    "generated_rgb_or_latent_may_be_editor_initial_noise": False,
    "generated_rgb_or_latent_may_be_editor_donor": False,
    "generated_hidden_may_be_editor_feature_target": False,
    "critic_score_may_be_used_only_after_heldout_gates": True,
}

STRICT_SPLIT_AXES = (
    "actor_group_id",
    "scene_group_id",
    "action_group_id",
    "seed_key",
    "action_family_id",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")

_USAGE_FIELDS = frozenset(
    {
        "schema_version",
        "bank_receipt_digest",
        "authorized_use",
        "authorization_source",
        "authorization_evidence_sha256",
        "receipt_digest",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "bank_receipt_digest",
        "cell_id",
        "analysis_split",
        "action_family_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "seed",
        "official_gaussian_tensor_sha256",
        "semantic_branch",
        "full_t2v_caption",
        "full_t2v_caption_utf8_sha256",
        "clean_latent_artifact_path",
        "clean_latent_artifact_sha256",
        "clean_latent_tensor_sha256",
        "clean_latent_shape",
        "generation_receipt_digest",
        "event_audit_artifact_sha256",
        "complete_target_transition_observed",
        "terminal_hold_observed",
        "full_target_action_observed",
        "full_target_action_false_confirmed",
    }
)


class LatentTemporalEventDatasetError(ValueError):
    """The critic dataset is incomplete, ambiguous, or exceeds its authority."""


def derive_native_geometry(shape: Any) -> dict[str, Any]:
    """Derive Bernini's native patch geometry from an exact81 latent shape.

    This helper is intentionally shape-generic: scientific top-ups may use a
    native resolution not present in core4.  The core4 materializer separately
    requires one of :data:`CORE4_NATIVE_LATENT_SHAPES` after authenticating the
    artifact bytes and receipt.  No resize, crop, or orientation swap is
    permitted here.
    """

    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 5
        or any(type(item) is not int or item <= 0 for item in shape)
        or tuple(shape[:3]) != (1, LATENT_CHANNELS, LATENT_PHASES)
    ):
        raise LatentTemporalEventDatasetError(
            "clean latent must be exact81 [1,16,21,H,W]"
        )
    latent_height, latent_width = int(shape[3]), int(shape[4])
    patch_height, patch_width = LATENT_SPATIAL_PATCH_SIZE
    if latent_height % patch_height or latent_width % patch_width:
        raise LatentTemporalEventDatasetError(
            "clean latent H/W must divide the native 2x2 spatial patch"
        )
    grid_height = latent_height // patch_height
    grid_width = latent_width // patch_width
    return {
        "latent_shape": [int(item) for item in shape],
        "latent_height_width": [latent_height, latent_width],
        "spatial_patch_size_height_width": [patch_height, patch_width],
        "patch_grid_height_width": [grid_height, grid_width],
        "patch_positions": grid_height * grid_width,
        "patch_flatten_order": "patch-y-x",
        "resize_or_crop_applied": False,
    }


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
        raise LatentTemporalEventDatasetError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise LatentTemporalEventDatasetError(
            f"{label} field closure differs; expected={sorted(fields)!r}, actual={actual!r}"
        )
    return dict(value)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LatentTemporalEventDatasetError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise LatentTemporalEventDatasetError(f"{label} must be a path-safe identifier")
    return value


def _absolute_plain_path_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise LatentTemporalEventDatasetError(f"{label} must be path text")
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise LatentTemporalEventDatasetError(f"{label} must be absolute and non-root")
    return str(path)


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(unsigned)
    return {**row, "receipt_digest": object_sha256(row)}


def validate_critic_usage_authority(value: Any) -> dict[str, Any]:
    """Validate a new sidecar; the old calibration-only receipt is insufficient."""

    row = _closed(value, _USAGE_FIELDS, label="critic-use authority")
    digest = _sha256(row["receipt_digest"], label="critic-use receipt digest")
    unsigned = dict(row)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != digest:
        raise LatentTemporalEventDatasetError("critic-use authority digest differs")
    if row["schema_version"] != USAGE_AUTHORITY_SCHEMA:
        raise LatentTemporalEventDatasetError("critic-use authority schema differs")
    _sha256(row["bank_receipt_digest"], label="bank receipt digest")
    _sha256(
        row["authorization_evidence_sha256"],
        label="authorization evidence SHA-256",
    )
    if (
        not isinstance(row["authorization_source"], str)
        or not row["authorization_source"].strip()
        or row["authorized_use"] != CRITIC_ONLY_USE
    ):
        raise LatentTemporalEventDatasetError(
            "critic-use authority is not the exact critic-only/no-editor contract"
        )
    return row


def make_critic_usage_authority(
    *,
    bank_receipt_digest: str,
    authorization_source: str,
    authorization_evidence_sha256: str,
) -> dict[str, Any]:
    """Create a sealable sidecar after an external authorization is archived."""

    _sha256(bank_receipt_digest, label="bank receipt digest")
    _sha256(authorization_evidence_sha256, label="authorization evidence SHA-256")
    if not isinstance(authorization_source, str) or not authorization_source.strip():
        raise LatentTemporalEventDatasetError("authorization_source must be nonempty")
    return _seal(
        {
            "schema_version": USAGE_AUTHORITY_SCHEMA,
            "bank_receipt_digest": bank_receipt_digest,
            "authorized_use": dict(CRITIC_ONLY_USE),
            "authorization_source": authorization_source.strip(),
            "authorization_evidence_sha256": authorization_evidence_sha256,
        }
    )


def validate_candidate_evidence(value: Any) -> dict[str, Any]:
    row = _closed(value, _CANDIDATE_FIELDS, label="candidate evidence")
    for name in (
        "candidate_id",
        "cell_id",
        "action_family_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
    ):
        _safe_id(row[name], label=name)
    if row["analysis_split"] not in (*PILOT_SPLITS, *SCIENTIFIC_SPLITS):
        raise LatentTemporalEventDatasetError("candidate split is not registered")
    if row["semantic_branch"] not in SEMANTIC_BRANCHES:
        raise LatentTemporalEventDatasetError("candidate semantic branch is not registered")
    for name in (
        "bank_receipt_digest",
        "official_gaussian_tensor_sha256",
        "full_t2v_caption_utf8_sha256",
        "clean_latent_artifact_sha256",
        "clean_latent_tensor_sha256",
        "generation_receipt_digest",
        "event_audit_artifact_sha256",
    ):
        _sha256(row[name], label=name)
    if type(row["seed"]) is not int or not 0 <= row["seed"] < 2**63:
        raise LatentTemporalEventDatasetError("seed must be an integer in [0,2^63)")
    caption = row["full_t2v_caption"]
    if (
        not isinstance(caption, str)
        or len(caption.strip().split()) < 12
        or hashlib.sha256(caption.encode("utf-8")).hexdigest()
        != row["full_t2v_caption_utf8_sha256"]
    ):
        raise LatentTemporalEventDatasetError("caption is incomplete or its hash differs")
    _absolute_plain_path_text(
        row["clean_latent_artifact_path"], label="clean latent artifact path"
    )
    shape = row["clean_latent_shape"]
    if not isinstance(shape, list):
        raise LatentTemporalEventDatasetError("clean latent shape must be a JSON list")
    derive_native_geometry(shape)
    for name in (
        "complete_target_transition_observed",
        "terminal_hold_observed",
        "full_target_action_observed",
        "full_target_action_false_confirmed",
    ):
        if type(row[name]) is not bool:
            raise LatentTemporalEventDatasetError(f"{name} must be an external bool label")
    if row["full_target_action_observed"] and row[
        "full_target_action_false_confirmed"
    ]:
        raise LatentTemporalEventDatasetError("event audit is contradictory")
    return row


def _event_positive(row: Mapping[str, Any]) -> bool:
    return all(
        row[name] is True
        for name in (
            "complete_target_transition_observed",
            "terminal_hold_observed",
            "full_target_action_observed",
        )
    ) and row["full_target_action_false_confirmed"] is False


def _event_negative(row: Mapping[str, Any]) -> bool:
    return (
        row["full_target_action_observed"] is False
        and row["full_target_action_false_confirmed"] is True
    )


def build_episode_plan(
    candidates: Sequence[Mapping[str, Any]],
    *,
    usage_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one positive-plus-12-negative hidden-query episode.

    The three same-video negatives reuse the positive clean-latent artifact but
    apply a registered phase transform *before* noising and frozen-model
    cross-query.  Semantic negatives use their own generated clean latent.  In
    every arm the model condition remains the cell's target-action/no-op pair.
    """

    authority = validate_critic_usage_authority(usage_authority)
    rows = [validate_candidate_evidence(row) for row in candidates]
    if len(rows) != len(SEMANTIC_BRANCHES):
        raise LatentTemporalEventDatasetError("one episode requires exactly ten owners")
    by_branch = {row["semantic_branch"]: row for row in rows}
    if len(by_branch) != len(rows) or tuple(by_branch) != SEMANTIC_BRANCHES:
        raise LatentTemporalEventDatasetError(
            "candidate order must be exact action plus nine semantic negatives"
        )
    shared_fields = (
        "bank_receipt_digest",
        "cell_id",
        "analysis_split",
        "action_family_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "seed",
        "official_gaussian_tensor_sha256",
        "clean_latent_shape",
    )
    for field in shared_fields:
        values = {json.dumps(row[field], sort_keys=True) for row in rows}
        if len(values) != 1:
            raise LatentTemporalEventDatasetError(f"cell candidates disagree on {field}")
    first = rows[0]
    if authority["bank_receipt_digest"] != first["bank_receipt_digest"]:
        raise LatentTemporalEventDatasetError("critic-use authority belongs to another bank")
    if not _event_positive(by_branch[ACTION_BRANCH]):
        raise LatentTemporalEventDatasetError(
            "positive owner is not externally event-qualified with transition and hold"
        )
    for branch in SEMANTIC_NEGATIVE_BRANCHES:
        if not _event_negative(by_branch[branch]):
            raise LatentTemporalEventDatasetError(
                f"semantic negative {branch} is ambiguous or contains the target event"
            )

    action_owner = by_branch[ACTION_BRANCH]
    target_action = {
        "caption": action_owner["full_t2v_caption"],
        "caption_utf8_sha256": action_owner["full_t2v_caption_utf8_sha256"],
    }
    noop_owner = by_branch["noop"]
    target_noop = {
        "caption": noop_owner["full_t2v_caption"],
        "caption_utf8_sha256": noop_owner["full_t2v_caption_utf8_sha256"],
    }
    if target_action["caption_utf8_sha256"] == target_noop["caption_utf8_sha256"]:
        raise LatentTemporalEventDatasetError("action/no-op captions collide")

    arms = []
    for role in ARM_ROLES:
        owner = by_branch[SOURCE_BRANCH_BY_ROLE[role]]
        transform = TEMPORAL_TRANSFORM_BY_ROLE[role]
        arms.append(
            {
                "role": role,
                "label": 1 if role == "positive" else 0,
                "source_candidate_id": owner["candidate_id"],
                "source_semantic_branch": owner["semantic_branch"],
                "clean_latent_artifact_path": owner["clean_latent_artifact_path"],
                "clean_latent_artifact_sha256": owner[
                    "clean_latent_artifact_sha256"
                ],
                "clean_latent_tensor_sha256": owner["clean_latent_tensor_sha256"],
                "temporal_transform": transform,
                "transform_applied_before_x_sigma_and_hidden_query": True,
                "action_hidden_value_digest_required": True,
                "noop_hidden_value_digest_required": True,
                "full_hidden_artifacts_may_be_discarded_after_hashing": True,
                "fixed_spatial_sketched_action_minus_noop_residual_artifact_required": True,
                "same_state_x_sigma_proof_required": True,
            }
        )
    seed_key = (
        f"seed-{first['seed']}-gaussian-{first['official_gaussian_tensor_sha256']}"
    )
    native_geometry = derive_native_geometry(first["clean_latent_shape"])
    unsigned = {
        "schema_version": EPISODE_SCHEMA,
        "dataset_schema_version": SCHEMA_VERSION,
        "episode_id": first["cell_id"],
        "split": first["analysis_split"],
        "source_kind": SOURCE_KIND,
        "bank_receipt_digest": first["bank_receipt_digest"],
        "critic_usage_authority_digest": authority["receipt_digest"],
        "action_family_id": first["action_family_id"],
        "actor_group_id": first["actor_group_id"],
        "scene_group_id": first["scene_group_id"],
        "action_group_id": first["action_group_id"],
        "seed": first["seed"],
        "seed_key": seed_key,
        "official_gaussian_tensor_sha256": first[
            "official_gaussian_tensor_sha256"
        ],
        "clean_latent_shape": first["clean_latent_shape"],
        "native_geometry": native_geometry,
        "target_action_condition": target_action,
        "target_noop_condition": target_noop,
        "hidden_query_contract": dict(PILOT_HIDDEN_QUERY),
        "arm_order": list(ARM_ROLES),
        "arms": arms,
        "positive_event_audit_artifact_sha256": action_owner[
            "event_audit_artifact_sha256"
        ],
        "all_semantic_negatives_false_confirmed": True,
        "mask_flow_pose_track_or_trajectory_used": False,
        "generated_media_editor_use_authorized": False,
    }
    return _seal(unsigned)


def _episode_identity(episode: Mapping[str, Any], field: str) -> str:
    value = episode.get(field)
    return _safe_id(value, label=f"episode {field}") if field != "seed_key" else str(value)


def _validate_episode_surface(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LatentTemporalEventDatasetError("episode must be a mapping")
    row = dict(value)
    digest = _sha256(row.pop("receipt_digest", None), label="episode digest")
    if object_sha256(row) != digest:
        raise LatentTemporalEventDatasetError("episode digest differs")
    if row.get("schema_version") != EPISODE_SCHEMA:
        raise LatentTemporalEventDatasetError("episode schema differs")
    if row.get("source_kind") != SOURCE_KIND:
        raise LatentTemporalEventDatasetError("episode source kind differs")
    try:
        expected_geometry = derive_native_geometry(row.get("clean_latent_shape"))
    except LatentTemporalEventDatasetError as error:
        raise LatentTemporalEventDatasetError(
            "episode native latent geometry is invalid"
        ) from error
    if row.get("arm_order") != list(ARM_ROLES):
        raise LatentTemporalEventDatasetError("episode arm order differs")
    arms = row.get("arms")
    if (
        not isinstance(arms, list)
        or [arm.get("role") for arm in arms] != list(ARM_ROLES)
        or any(arm.get("label") != (1 if index == 0 else 0) for index, arm in enumerate(arms))
    ):
        raise LatentTemporalEventDatasetError("episode arm closure differs")
    if (
        row.get("all_semantic_negatives_false_confirmed") is not True
        or row.get("mask_flow_pose_track_or_trajectory_used") is not False
        or row.get("generated_media_editor_use_authorized") is not False
        or row.get("hidden_query_contract") != PILOT_HIDDEN_QUERY
        or row.get("native_geometry") != expected_geometry
    ):
        raise LatentTemporalEventDatasetError("episode interpretation exceeds authority")
    return {**row, "receipt_digest": digest}


def _axis_overlaps(
    episodes: Sequence[Mapping[str, Any]], splits: Sequence[str]
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for axis in STRICT_SPLIT_AXES:
        memberships = {
            split: {
                str(row[axis]) for row in episodes if row["split"] == split
            }
            for split in splits
        }
        overlaps: dict[str, list[str]] = {}
        for left_index, left in enumerate(splits):
            for right in splits[left_index + 1 :]:
                overlap = sorted(memberships[left] & memberships[right])
                if overlap:
                    overlaps[f"{left}__{right}"] = overlap
        result[axis] = overlaps
    return result


def audit_episode_population(
    episodes: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    minimum_families_per_scientific_split: int = 2,
    minimum_identities_per_family: int = 3,
    minimum_seeds_per_identity: int = 3,
) -> dict[str, Any]:
    """Audit pilot or publishable population support without training a head."""

    checked = [_validate_episode_surface(row) for row in episodes]
    if len({row["episode_id"] for row in checked}) != len(checked):
        raise LatentTemporalEventDatasetError("episode IDs must be unique")
    if protocol not in ("core4_pilot", "scientific_family_holdout"):
        raise LatentTemporalEventDatasetError("population protocol is not registered")
    splits = PILOT_SPLITS if protocol == "core4_pilot" else SCIENTIFIC_SPLITS
    failures: list[str] = []
    present = {row["split"] for row in checked}
    if present != set(splits):
        failures.append("split_coverage")
    overlaps = _axis_overlaps(checked, splits)

    if protocol == "core4_pilot":
        # Core4 intentionally shares families between fit and confirmation;
        # every other group axis, including the exact Gaussian seed key, must
        # remain disjoint.  This is a go/no-go for top-up only.
        for axis in STRICT_SPLIT_AXES[:-1]:
            if overlaps[axis]:
                failures.append(f"pilot_split_overlap:{axis}")
        family_sets = {
            split: {row["action_family_id"] for row in checked if row["split"] == split}
            for split in splits
        }
        if family_sets["fit"] != family_sets["confirmation"]:
            failures.append("pilot_family_coverage_mismatch")
        if any(sum(row["split"] == split for row in checked) < 2 for split in splits):
            failures.append("pilot_requires_two_cells_per_split")
        population_eligible = not failures
        return {
            "schema_version": POPULATION_AUDIT_SCHEMA,
            "protocol": protocol,
            "episode_count": len(checked),
            "split_counts": Counter(row["split"] for row in checked),
            "action_families_by_split": {
                split: sorted(family_sets[split]) for split in splits
            },
            "strict_axis_overlaps": overlaps,
            "population_eligible": population_eligible,
            "critic_head_pilot_training_authorized": population_eligible,
            "worth_topup_evaluation_authorized": population_eligible,
            "scientific_critic_claim_authorized": False,
            "editor_optimizer_authorized": False,
            "failure_reasons": failures,
        }

    for axis in STRICT_SPLIT_AXES:
        if overlaps[axis]:
            failures.append(f"scientific_split_overlap:{axis}")
    family_sets = {
        split: {row["action_family_id"] for row in checked if row["split"] == split}
        for split in splits
    }
    for split in splits:
        if len(family_sets[split]) < minimum_families_per_scientific_split:
            failures.append(f"too_few_action_families:{split}")
    for split in splits:
        for family in family_sets[split]:
            family_rows = [
                row
                for row in checked
                if row["split"] == split and row["action_family_id"] == family
            ]
            actors = {row["actor_group_id"] for row in family_rows}
            if len(actors) < minimum_identities_per_family:
                failures.append(f"too_few_identities:{split}:{family}")
            for actor in actors:
                seeds = {
                    row["seed_key"]
                    for row in family_rows
                    if row["actor_group_id"] == actor
                }
                if len(seeds) < minimum_seeds_per_identity:
                    failures.append(f"too_few_seeds:{split}:{family}:{actor}")
    population_eligible = not failures
    return {
        "schema_version": POPULATION_AUDIT_SCHEMA,
        "protocol": protocol,
        "episode_count": len(checked),
        "split_counts": Counter(row["split"] for row in checked),
        "action_families_by_split": {
            split: sorted(family_sets[split]) for split in splits
        },
        "strict_axis_overlaps": overlaps,
        "population_eligible": population_eligible,
        "critic_head_training_authorized": population_eligible,
        "scientific_critic_claim_authorized": False,
        "editor_optimizer_authorized": False,
        "failure_reasons": failures,
    }


def audit_core4_spec_inventory(
    root_spec: Mapping[str, Any],
    *,
    detached_event_labels: Mapping[str, Any] | None = None,
    hidden_pair_manifest_present: bool = False,
    critic_use_sidecar_present: bool = False,
) -> dict[str, Any]:
    """Describe what the existing core4 bank can and cannot support.

    This intentionally does not authenticate files; the existing bank/label
    tools remain the authority for byte-level verification.  It audits sample
    topology and use scope after those tools have succeeded.
    """

    if not isinstance(root_spec, Mapping) or not isinstance(root_spec.get("groups"), list):
        raise LatentTemporalEventDatasetError("core4 root spec is malformed")
    rows = [candidate for group in root_spec["groups"] for candidate in group["candidates"]]
    cells: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[str(row["calibration_group_id"])].append(row)
    families = {str(row["action_family_id"]) for row in rows}
    actors = {str(row["actor_group_id"]) for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    split_cells = Counter(cell[0]["analysis_split"] for cell in cells.values())
    event_positive_count: int | None = None
    semantic_false_count: int | None = None
    if detached_event_labels is not None:
        label_rows = detached_event_labels.get("rows")
        if not isinstance(label_rows, list):
            raise LatentTemporalEventDatasetError("detached label rows are malformed")
        event_positive_count = sum(
            row.get("full_target_action_observed") is True
            and row.get("complete_target_transition_observed") is True
            and row.get("terminal_hold_observed") is True
            and row.get("full_target_action_false_confirmed") is False
            for row in label_rows
        )
        semantic_false_count = sum(
            row.get("semantic_branch") != ACTION_BRANCH
            and row.get("full_target_action_observed") is False
            and row.get("full_target_action_false_confirmed") is True
            for row in label_rows
        )
    artifact_use = root_spec.get("artifact_use_contract", {})
    old_scope_is_calibration_only = (
        artifact_use.get("predecode_clean_latent") == "calibration_evidence_only"
        and artifact_use.get("training_donor") is False
    )
    failures = []
    if len(rows) != 40 or len(cells) != 4:
        failures.append("not_exact40_four_cells")
    if len(families) < 3:
        failures.append("fewer_than_three_action_families_for_three_way_holdout")
    if any(len(cell) != 10 for cell in cells.values()):
        failures.append("incomplete_ten_branch_cell")
    if len(seeds) != len(cells):
        failures.append("not_one_distinct_registered_seed_per_cell")
    if any(count != 2 for count in split_cells.values()) or set(split_cells) != set(PILOT_SPLITS):
        failures.append("core4_pilot_split_topology_differs")
    if event_positive_count is None:
        failures.append("detached_event_labels_not_supplied_to_inventory_audit")
    elif event_positive_count != 4 or semantic_false_count != 36:
        failures.append("detached_event_label_closure_differs")
    if not critic_use_sidecar_present:
        failures.append("critic_only_use_sidecar_missing")
    if not hidden_pair_manifest_present:
        failures.append("same_state_action_noop_hidden_pairs_not_materialized")
    return {
        "schema_version": PILOT_AUDIT_SCHEMA,
        "candidate_count": len(rows),
        "cell_count": len(cells),
        "action_family_count": len(families),
        "actor_group_count": len(actors),
        "registered_seed_count": len(seeds),
        "split_cell_counts": dict(split_cells),
        "event_qualified_positive_count": event_positive_count,
        "false_confirmed_semantic_negative_count": semantic_false_count,
        "old_artifact_scope_is_calibration_only": old_scope_is_calibration_only,
        "critic_use_sidecar_present": critic_use_sidecar_present,
        "same_state_action_noop_hidden_pairs_present": hidden_pair_manifest_present,
        "core4_geometry_can_support_fit_confirmation_pilot": (
            len(rows) == 40
            and len(cells) == 4
            and set(split_cells) == set(PILOT_SPLITS)
            and all(count == 2 for count in split_cells.values())
            and event_positive_count in (None, 4)
        ),
        "scientific_three_way_family_holdout_possible": False,
        "current_critic_training_authorized": not failures,
        "scientific_critic_claim_authorized": False,
        "editor_optimizer_authorized": False,
        "failure_reasons": failures,
    }


def evaluate_core4_pilot_gate(
    confirmation_scores: Mapping[str, Mapping[str, float]],
    *,
    expected_confirmation_episode_ids: Sequence[str],
    minimum_margin: float = 0.20,
    input_gradient_audit_passed: bool,
) -> dict[str, Any]:
    """Require every negative margin on both confirmation cells.

    Passing only recommends a preregistered top-up.  It can never authorize an
    RV2V optimizer, because two confirmation positives and two action families
    cannot establish nuisance or family generalization.
    """

    if (
        isinstance(minimum_margin, bool)
        or not isinstance(minimum_margin, (int, float))
        or not math.isfinite(float(minimum_margin))
        or float(minimum_margin) <= 0.0
    ):
        raise LatentTemporalEventDatasetError("minimum_margin must be positive finite")
    episode_ids = tuple(expected_confirmation_episode_ids)
    if len(episode_ids) != 2 or len(set(episode_ids)) != 2:
        raise LatentTemporalEventDatasetError("core4 pilot requires exactly two confirmation cells")
    failures: list[str] = []
    margins: dict[str, dict[str, float]] = {}
    for episode_id in episode_ids:
        scores = confirmation_scores.get(episode_id)
        if not isinstance(scores, Mapping) or set(scores) != set(ARM_ROLES):
            failures.append(f"score_closure:{episode_id}")
            continue
        numeric = {}
        for role, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise LatentTemporalEventDatasetError("pilot scores must be finite numbers")
            numeric[role] = float(value)
        margins[episode_id] = {
            role: numeric["positive"] - numeric[role] for role in NEGATIVE_ROLES
        }
        for role, margin in margins[episode_id].items():
            if margin < float(minimum_margin):
                failures.append(f"margin:{episode_id}:{role}")
    if input_gradient_audit_passed is not True:
        failures.append("current_rv2v_clean_latent_gradient_audit")
    passed = not failures
    return {
        "schema_version": "bernini-core4-hidden-event-critic-pilot-gate-v1",
        "minimum_margin": float(minimum_margin),
        "confirmation_margins": margins,
        "all_confirmation_hard_negative_margins_passed": passed,
        "input_gradient_audit_passed": input_gradient_audit_passed is True,
        "worth_fixed_topup_generation": passed,
        "scientific_critic_claim_authorized": False,
        "editor_optimizer_authorized": False,
        "failure_reasons": failures,
    }

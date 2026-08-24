#!/usr/bin/env python3
"""Preregister the branch-trajectory, canonical-role T2V probe.

V2 evaluated four text prompts on an action-dominated shared latent.  This
registry fixes that confound: every arm owns a complete frozen T2V trajectory
from the same source-independent Gaussian and the same exact40 scheduler.
Only intermediate transformer states may be observed; decoding, targets,
training and generator routing remain unavailable.

Appearance words are deliberately outside the role vocabulary.  The four
literal aliases (``the agent``, ``the moving object``, ``the start support``,
``the end support``) are identical in every appearance and every arm.  They
provide a canonical object-slot interface while all actor/object/material
descriptors are assigned to ``null_context`` by the runtime partition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


METHOD = "bernini-self-generated-relational-t2v-probe-registry-v3"
SCHEMA_VERSION = "bernini-self-generated-relational-t2v-probe-registry-v3"
APPEARANCE_IDS = ("appearance_0", "appearance_1", "appearance_2")
ARMS = ("action", "noop", "reverse", "static")
ROLE_IDS = (
    "agent",
    "moving_object",
    "start_support",
    "end_support",
    "null_context",
)
BLOCKS = (6, 12, 18, 24)
SIGMA_CELL_INDICES = MappingProxyType({"high": 18, "mid": 32, "mid_low": 38})
ROLE_PHRASES = MappingProxyType(
    {
        "agent": "the agent",
        "moving_object": "the moving object",
        "start_support": "the start support",
        "end_support": "the end support",
    }
)
PUBLIC_FEATURES = (
    "object_axis_position",
    "object_cross_axis_position",
    "agent_object_distance",
    "object_start_distance",
    "object_end_distance",
    "agent_object_soft_edge",
    "object_start_soft_edge",
    "object_end_soft_edge",
)
ADMISSION_THRESHOLDS = MappingProxyType(
    {
        "role_mass_min": 1.0e-6,
        "role_localization_confidence_min": 0.01,
        "support_frame_scale_min": 0.02,
        "forward_progress_min": 0.08,
        "reverse_progress_max": -0.08,
        "per_cell_forward_progress_strictly_greater_than": 0.0,
        "dynamic_over_null_ratio_min": 1.50,
        "null_transition_ratio_max": 0.60,
        # Preserve the v2 transfer bar.  V3 changes the representation, not
        # the definition of "stable" after seeing a rejected receipt.
        "reverse_cycle_cosine_min": 0.95,
        "reverse_cycle_distance_max": 0.15,
        "appearance_cosine_min": 0.95,
        "appearance_distance_max": 0.15,
        "positive_progress_cell_fraction_min": 0.75,
        "support_signed_distance_change_min": 0.03,
        "soft_edge_switch_min": 0.02,
        "reverse_endpoint_topology_rms_max": 0.15,
        "reverse_endpoint_topology_max_abs_max": 0.15,
    }
)


class RelationalT2VProbeRegistryV3Error(ValueError):
    """Raised before an ambiguous or target-leaking v3 probe can run."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RelationalT2VProbeRegistryV3Error(
            "registry is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise RelationalT2VProbeRegistryV3Error("text digest requires a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AppearanceProbe:
    appearance_id: str
    appearance_bindings: Mapping[str, str]
    role_phrases: Mapping[str, str]
    captions: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.appearance_id not in APPEARANCE_IDS:
            raise RelationalT2VProbeRegistryV3Error("appearance id differs")
        expected_roles = set(ROLE_IDS) - {"null_context"}
        if set(self.appearance_bindings) != expected_roles:
            raise RelationalT2VProbeRegistryV3Error(
                "appearance binding registry differs"
            )
        if dict(self.role_phrases) != dict(ROLE_PHRASES):
            raise RelationalT2VProbeRegistryV3Error(
                "canonical role phrase registry differs"
            )
        if set(self.captions) != set(ARMS) or len(set(self.captions.values())) != len(ARMS):
            raise RelationalT2VProbeRegistryV3Error("arm registry differs")
        for descriptor in self.appearance_bindings.values():
            if not isinstance(descriptor, str) or descriptor != descriptor.strip() or not descriptor:
                raise RelationalT2VProbeRegistryV3Error("appearance descriptor is malformed")
            if any(phrase.casefold() in descriptor.casefold() for phrase in ROLE_PHRASES.values()):
                raise RelationalT2VProbeRegistryV3Error(
                    "appearance descriptor aliases a canonical role"
                )
        for arm, caption in self.captions.items():
            if (
                not isinstance(caption, str)
                or caption != caption.strip()
                or not caption
                or "\x00" in caption
            ):
                raise RelationalT2VProbeRegistryV3Error(f"{arm} caption is malformed")
            for phrase in ROLE_PHRASES.values():
                if phrase.casefold() not in caption.casefold():
                    raise RelationalT2VProbeRegistryV3Error(
                        f"{arm} lacks canonical role phrase {phrase}"
                    )

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "appearance_id": self.appearance_id,
            "appearance_bindings": dict(self.appearance_bindings),
            "role_phrases": dict(self.role_phrases),
            "captions": dict(self.captions),
            "caption_sha256": {
                arm: text_sha256(self.captions[arm]) for arm in ARMS
            },
            "appearance_descriptors_owned_by_null_context": True,
            "canonical_role_aliases_identical_across_appearances": True,
        }
        return {**value, "digest": object_sha256(value)}


def _appearance(
    appearance_id: str,
    *,
    agent: str,
    moving_object: str,
    start_support: str,
    end_support: str,
) -> AppearanceProbe:
    bindings = {
        "agent": agent,
        "moving_object": moving_object,
        "start_support": start_support,
        "end_support": end_support,
    }
    glossary = (
        f"Scene appearance: {agent} is the agent; {moving_object} is the moving object; "
        f"{start_support} is the start support; {end_support} is the end support. "
    )
    locked = " The camera remains locked and the background remains still."
    captions = {
        "action": (
            glossary
            + "The agent picks up the moving object from the start support, carries "
            "the moving object across the frame, and sets the moving object down on "
            "the end support."
            + locked
        ),
        "noop": (
            glossary
            + "The agent remains motionless beside the start support; the moving object "
            "stays unchanged on the start support, while the end support remains empty."
            + locked
        ),
        "reverse": (
            glossary
            + "The agent picks up the moving object from the end support, carries the "
            "moving object back across the frame, and sets the moving object down on "
            "the start support."
            + locked
        ),
        "static": (
            glossary
            + "The agent holds the moving object completely still midway between the "
            "start support and the end support for the entire shot."
            + locked
        ),
    }
    return AppearanceProbe(appearance_id, bindings, ROLE_PHRASES, captions)


APPEARANCES = (
    _appearance(
        "appearance_0",
        agent="A young woman in a green jacket",
        moving_object="a red ceramic mug",
        start_support="a dark wooden table on the left",
        end_support="a white metal shelf on the right",
    ),
    _appearance(
        "appearance_1",
        agent="An older man in a blue sweater",
        moving_object="a yellow rubber ball",
        start_support="a gray stone bench on the left",
        end_support="a blue plastic bin on the right",
    ),
    _appearance(
        "appearance_2",
        agent="A small silver robot",
        moving_object="a purple wooden block",
        start_support="a black steel platform on the left",
        end_support="an orange tray on the right",
    ),
)


def registry_receipt() -> Mapping[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "appearance_ids": list(APPEARANCE_IDS),
        "arms": list(ARMS),
        "role_ids": list(ROLE_IDS),
        "canonical_role_phrases": dict(ROLE_PHRASES),
        "blocks": list(BLOCKS),
        "sigma_cell_indices": dict(SIGMA_CELL_INDICES),
        "appearances": [dict(item.receipt()) for item in APPEARANCES],
        "trajectory_authority": {
            "one_complete_native_t2v_trajectory_per_appearance_arm": True,
            "same_initial_source_independent_gaussian": True,
            "same_exact40_unipc_schedule": True,
            "branch_states_expected_to_differ_after_step_zero": True,
            "same_state_prompt_overlay_used": False,
        },
        "object_centric_representation": {
            "support_frame": ["start_support", "end_support"],
            "translation_invariant": True,
            "normalized_unit_square_rotation_invariant_above_cutoff": True,
            "physical_pixel_rotation_invariant": False,
            "representation_scale_invariant_above_degeneracy_cutoff": True,
            "whole_admission_receipt_scale_invariant": False,
            "soft_edge_temperature": 0.25,
            "public_features": list(PUBLIC_FEATURES),
            "appearance_descriptors_in_public_features": False,
            "raw_qk_in_public_features": False,
            "absolute_coordinates_in_public_features": False,
        },
        "interaction_graph": {
            "context_edge": ["start_support", "end_support"],
            "candidate_dynamic_edges": [
                ["agent", "moving_object"],
                ["moving_object", "start_support"],
                ["moving_object", "end_support"],
            ],
            "soft_phase_varying_edges": True,
            "default_cartesian_product_used": False,
            "physical_contact_truth_claimed": False,
        },
        "admission_thresholds": dict(ADMISSION_THRESHOLDS),
        "thresholds_preregistered_before_v3_r2_auditable_observation": True,
        "earlier_v3_r1_diagnostic_excluded_from_admission": True,
        "frozen_base_off_branch_required_per_appearance_arm_sigma": True,
        "frozen_base_can_supply_graph_success": False,
        "target_inputs_authorized": False,
        "final_anchor_video_decode_authorized": False,
        "training_or_parameter_updates_authorized": False,
        "routing_or_injection_authorized": False,
        "scientific_claim_authorized": False,
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "ADMISSION_THRESHOLDS",
    "APPEARANCE_IDS",
    "APPEARANCES",
    "ARMS",
    "BLOCKS",
    "METHOD",
    "PUBLIC_FEATURES",
    "ROLE_IDS",
    "ROLE_PHRASES",
    "SCHEMA_VERSION",
    "SIGMA_CELL_INDICES",
    "AppearanceProbe",
    "RelationalT2VProbeRegistryV3Error",
    "canonical_json_bytes",
    "object_sha256",
    "registry_receipt",
    "text_sha256",
]

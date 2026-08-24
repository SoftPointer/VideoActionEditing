#!/usr/bin/env python3
"""Preregister the real self-generated relational T2V representation probe.

The registry deliberately owns no model and no media path.  It defines three
appearance substitutions of one typed interaction, four role-matched prompt
controls, and the exact three cells selected from Bernini's 40-step UniPC
schedule.  Runtime code may encode these strings and observe the frozen
transformer's intermediate Q/K tensors, but it may not decode a video, read a
target, train a parameter, or use the resulting representation for routing.

All object nodes have ``self_generated_anchor_owned`` provenance.  They are
not source identities.  The two support edges are kinematic hypotheses only;
neither is evidence of physical support or contact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


METHOD = "bernini-self-generated-relational-t2v-probe-registry-v2"
SCHEMA_VERSION = "bernini-self-generated-relational-t2v-probe-registry-v2"
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


class RelationalT2VProbeRegistryError(ValueError):
    """Raised before an ambiguous or target-leaking probe can be launched."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RelationalT2VProbeRegistryError("registry is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise RelationalT2VProbeRegistryError("text digest requires a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AppearanceProbe:
    appearance_id: str
    role_phrases: Mapping[str, str]
    captions: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.appearance_id not in APPEARANCE_IDS:
            raise RelationalT2VProbeRegistryError("appearance id differs")
        expected_roles = set(ROLE_IDS) - {"null_context"}
        if set(self.role_phrases) != expected_roles:
            raise RelationalT2VProbeRegistryError("appearance role phrase registry differs")
        if set(self.captions) != set(ARMS):
            raise RelationalT2VProbeRegistryError("appearance arm registry differs")
        phrases = tuple(self.role_phrases.values())
        if (
            len(set(phrases)) != len(phrases)
            or any(not isinstance(item, str) or not item.strip() for item in phrases)
        ):
            raise RelationalT2VProbeRegistryError("role phrases must be distinct text")
        if len(set(self.captions.values())) != len(ARMS):
            raise RelationalT2VProbeRegistryError("four controls must be distinct")
        for arm, caption in self.captions.items():
            if (
                not isinstance(caption, str)
                or caption != caption.strip()
                or not caption
                or "\x00" in caption
            ):
                raise RelationalT2VProbeRegistryError(f"{arm} caption is malformed")
            for role, phrase in self.role_phrases.items():
                if phrase not in caption:
                    raise RelationalT2VProbeRegistryError(
                        f"{arm} lacks role-matched phrase {role}"
                    )

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "appearance_id": self.appearance_id,
            "role_phrases": dict(self.role_phrases),
            "captions": dict(self.captions),
            "caption_sha256": {
                arm: text_sha256(self.captions[arm]) for arm in ARMS
            },
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
    roles = {
        "agent": agent,
        "moving_object": moving_object,
        "start_support": start_support,
        "end_support": end_support,
    }
    locked = " The camera remains locked and the background remains still."
    captions = {
        "action": (
            f"{agent} picks up {moving_object} from {start_support}, carries "
            f"{moving_object} across the frame, and sets it down on {end_support}."
            + locked
        ),
        "noop": (
            f"{agent} remains motionless beside {start_support}; {moving_object} "
            f"stays unchanged on {start_support}, while {end_support} remains empty."
            + locked
        ),
        "reverse": (
            f"{agent} picks up {moving_object} from {end_support}, carries "
            f"{moving_object} back across the frame, and sets it down on {start_support}."
            + locked
        ),
        "static": (
            f"{agent} holds {moving_object} completely still midway between "
            f"{start_support} and {end_support} for the entire shot."
            + locked
        ),
    }
    return AppearanceProbe(appearance_id, roles, captions)


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
        "blocks": list(BLOCKS),
        "sigma_cell_indices": dict(SIGMA_CELL_INDICES),
        "sigma_selection_preregistered_before_representation_observation": True,
        "appearances": [dict(item.receipt()) for item in APPEARANCES],
        "node_provenance": "self_generated_anchor_owned_not_source_identity",
        "typed_edges": [
            {
                "source_role": "agent",
                "target_role": "moving_object",
                "relation_type": "relative_motion",
                "first_applicable_phase": 0,
                "last_applicable_phase": 20,
                "applicability": "required",
            },
            {
                "source_role": "moving_object",
                "target_role": "start_support",
                "relation_type": "approaching_or_receding",
                "first_applicable_phase": 0,
                "last_applicable_phase": 12,
                "applicability": "required",
            },
            {
                "source_role": "moving_object",
                "target_role": "end_support",
                "relation_type": "approaching_or_receding",
                "first_applicable_phase": 8,
                "last_applicable_phase": 20,
                "applicability": "required",
            },
        ],
        "default_cartesian_graph_used": False,
        "physical_contact_or_support_truth_claimed": False,
        "same_initial_gaussian_across_appearances": True,
        "same_action_trajectory_state_across_four_arms_within_cell": True,
        "frozen_base_off_branch_required_per_appearance_sigma": True,
        "frozen_base_can_supply_graph_success": False,
        "target_inputs_authorized": False,
        "final_anchor_video_decode_authorized": False,
        "training_or_parameter_updates_authorized": False,
        "routing_or_injection_authorized": False,
        "scientific_claim_authorized": False,
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "APPEARANCE_IDS",
    "APPEARANCES",
    "ARMS",
    "BLOCKS",
    "METHOD",
    "ROLE_IDS",
    "SCHEMA_VERSION",
    "SIGMA_CELL_INDICES",
    "AppearanceProbe",
    "RelationalT2VProbeRegistryError",
    "object_sha256",
    "registry_receipt",
    "text_sha256",
]

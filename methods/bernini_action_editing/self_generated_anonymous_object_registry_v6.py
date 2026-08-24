#!/usr/bin/env python3
"""Frozen anonymous-object preregistry for the V6 same-state probe.

V6 is intentionally not a revision of the V4 text-role observer.  The prompt
strings remain ordinary T2V controls, but this registry exposes no semantic
role names, token spans, token owners, or fixed object-slot inventory.  The
observer may create only cell-local anonymous hypotheses from visual
intermediates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


METHOD = "bernini-self-generated-anonymous-object-probe-v6"
SCHEMA_VERSION = "bernini-self-generated-anonymous-object-registry-v6"
APPEARANCE_IDS = ("appearance_0", "appearance_1", "appearance_2")
ARMS = (
    "action",
    "noop",
    "reverse",
    "static",
    "neutral",
    "paraphrase",
    "lexical_placebo",
    "source_swap",
)
BLOCKS = (6, 12, 18, 24)
SIGMA_CELL_INDICES = MappingProxyType({"high": 18, "mid": 32, "low": 38})
PHASES = 21
PATCH_HEIGHT = 37
PATCH_WIDTH = 25
PATCHES = PATCH_HEIGHT * PATCH_WIDTH
PREREG_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "self_generated_anonymous_object_probe_prereg_v6.json"
)


class AnonymousObjectRegistryV6Error(ValueError):
    """Raised before accepting a mutable or semantically slotted registry."""


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
        raise AnonymousObjectRegistryV6Error(
            "value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise AnonymousObjectRegistryV6Error("text digest requires a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnonymousAppearanceV6:
    """Opaque prompt quartet; no phrase is declared to be an object role."""

    appearance_id: str
    captions: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.appearance_id not in APPEARANCE_IDS:
            raise AnonymousObjectRegistryV6Error("appearance id differs")
        if set(self.captions) != set(ARMS):
            raise AnonymousObjectRegistryV6Error("prompt arm registry differs")
        if len(set(self.captions.values())) != len(ARMS):
            raise AnonymousObjectRegistryV6Error("prompt arms must be distinct")
        for arm, caption in self.captions.items():
            if (
                not isinstance(caption, str)
                or caption != caption.strip()
                or not caption
                or "\x00" in caption
            ):
                raise AnonymousObjectRegistryV6Error(
                    f"{arm} caption is malformed"
                )

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "appearance_id": self.appearance_id,
            "captions": dict(self.captions),
            "caption_sha256": {
                arm: text_sha256(self.captions[arm]) for arm in ARMS
            },
            "semantic_role_inventory_exposed": False,
            "caption_token_partition_exposed": False,
        }
        return {**value, "digest": object_sha256(value)}


def _appearance(
    appearance_id: str,
    *,
    subject: str,
    item: str,
    origin: str,
    destination: str,
    swapped_action: str,
) -> AnonymousAppearanceV6:
    scene = (
        f"A locked-camera scene contains {subject}, {item}, {origin}, and "
        f"{destination}. "
    )
    still = " The camera and background remain completely still."
    captions = {
        "action": _action_caption(subject, item, origin, destination),
        "noop": (
            scene
            + f"{subject} remains motionless beside {origin}; {item} stays "
            f"unchanged on {origin}, while {destination} remains empty."
            + still
        ),
        "reverse": (
            scene
            + f"{subject} picks up {item} from {destination}, carries it back "
            f"across the frame, and sets it down on {origin}."
            + still
        ),
        "static": (
            scene
            + f"{subject} holds {item} completely still midway between "
            f"{origin} and {destination} for the entire shot."
            + still
        ),
        "neutral": (
            "A locked-camera scene with an unspecified foreground and a still "
            "background. Nothing is named or assigned a semantic role."
        ),
        "paraphrase": (
            scene
            + f"Beginning at {origin}, {item} is lifted by {subject}, transported "
            f"laterally through the shot, then released at {destination}."
            + still
        ),
        "lexical_placebo": (
            scene
            + "The written words pick up, carry, and set down appear on a small "
            "stationary sign, but every physical thing remains motionless for the "
            "entire shot."
            + still
        ),
        "source_swap": swapped_action,
    }
    return AnonymousAppearanceV6(appearance_id, captions)


def _action_caption(subject: str, item: str, origin: str, destination: str) -> str:
    return (
        f"A locked-camera scene contains {subject}, {item}, {origin}, and "
        f"{destination}. {subject} picks up {item} from {origin}, carries it "
        f"across the frame, and sets it down on {destination}. The camera and "
        "background remain completely still."
    )


APPEARANCES = (
    _appearance(
        "appearance_0",
        subject="a young woman in a green jacket",
        item="a red ceramic mug",
        origin="a dark wooden table on the left",
        destination="a white metal shelf on the right",
        swapped_action=_action_caption(
            "an older man in a blue sweater",
            "a yellow rubber ball",
            "a gray stone bench on the left",
            "a blue plastic bin on the right",
        ),
    ),
    _appearance(
        "appearance_1",
        subject="an older man in a blue sweater",
        item="a yellow rubber ball",
        origin="a gray stone bench on the left",
        destination="a blue plastic bin on the right",
        swapped_action=_action_caption(
            "a small silver robot",
            "a purple wooden block",
            "a black steel platform on the left",
            "an orange tray on the right",
        ),
    ),
    _appearance(
        "appearance_2",
        subject="a small silver robot",
        item="a purple wooden block",
        origin="a black steel platform on the left",
        destination="an orange tray on the right",
        swapped_action=_action_caption(
            "a young woman in a green jacket",
            "a red ceramic mug",
            "a dark wooden table on the left",
            "a white metal shelf on the right",
        ),
    ),
)

# A source-swap control changes appearance nouns only: it is byte-for-byte the
# next registered appearance's action caption, with a closed three-cycle.
for index, appearance in enumerate(APPEARANCES):
    expected = APPEARANCES[(index + 1) % len(APPEARANCES)].captions["action"]
    if appearance.captions["source_swap"] != expected:
        raise AnonymousObjectRegistryV6Error(
            "source-swap prompt is not the exact next-appearance action caption"
        )


def load_preregistration() -> Mapping[str, Any]:
    if not PREREG_PATH.is_file() or PREREG_PATH.is_symlink():
        raise AnonymousObjectRegistryV6Error(
            "V6 preregistration must be a plain file"
        )
    try:
        value = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnonymousObjectRegistryV6Error(
            "V6 preregistration cannot be read"
        ) from error
    if not isinstance(value, dict):
        raise AnonymousObjectRegistryV6Error("V6 preregistration differs")
    if value.get("method") != METHOD or value.get("frozen_before_gpu_execution") is not True:
        raise AnonymousObjectRegistryV6Error("V6 preregistration authority differs")
    capture = value.get("capture")
    if not isinstance(capture, dict) or any(
        capture.get(name) is not False
        for name in (
            "text_key_or_value_used",
            "caption_token_offsets_used",
            "caption_role_partition_used",
            "fixed_semantic_role_inventory_used",
        )
    ):
        raise AnonymousObjectRegistryV6Error(
            "V6 preregistration permits text-role localization"
        )
    claims = value.get("claims")
    if not isinstance(claims, dict) or (
        claims.get("representation_admission_hard_false") is not True
        or claims.get("gpu_launch_authorized") is not True
        or claims.get("launch_blocked_pending_independent_audit") is not False
    ):
        raise AnonymousObjectRegistryV6Error("V6 claim boundary differs")
    if (
        capture.get("blocks") != list(BLOCKS)
        or capture.get("sigma_cells") != dict(SIGMA_CELL_INDICES)
        or capture.get("patch_height") != PATCH_HEIGHT
        or capture.get("patch_width") != PATCH_WIDTH
    ):
        raise AnonymousObjectRegistryV6Error("V6 capture matrix differs")
    cross_fit = value.get("cross_fit")
    expected_a_to_b = [[phase, phase + 1] for phase in range(0, 20, 2)]
    expected_b_to_a = [[phase, phase + 1] for phase in range(1, 20, 2)]
    if (
        not isinstance(cross_fit, dict)
        or cross_fit.get("A_to_B_phase_pairs") != expected_a_to_b
        or cross_fit.get("B_to_A_phase_pairs") != expected_b_to_a
        or cross_fit.get("phase_pair_count_per_branch") != 10
        or cross_fit.get("branchwise_and_required") is not True
        or cross_fit.get("evaluation_uses_action_noop_residual") is not False
    ):
        raise AnonymousObjectRegistryV6Error("V6 cross-fit authority differs")
    gates = value.get("branchwise_diagnostic_gates")
    tracking = value.get("tracking")
    dynamic_edges = value.get("dynamic_edges")
    if (
        not isinstance(gates, dict)
        or gates.get("R0_or_proposal_score_compensation") is not False
        or gates.get("static_zero_track_displacement_definition") != 0.0
        or gates.get("source_swap_dynamic_edge_lifecycle_max") != 0
    ):
        raise AnonymousObjectRegistryV6Error("V6 diagnostic gates differ")
    if (
        not isinstance(tracking, dict)
        or tracking.get("minimum_track_observed_phases") != 3
        or tracking.get(
            "only_qualified_tracks_contribute_to_graph_or_control_metrics"
        )
        is not True
        or not isinstance(dynamic_edges, dict)
        or dynamic_edges.get("qualified_tracks_only") is not True
    ):
        raise AnonymousObjectRegistryV6Error("V6 qualified-track authority differs")
    overall = value.get("overall_diagnostic_aggregation")
    if (
        not isinstance(overall, dict)
        or overall.get("expected_cell_count") != 9
        or overall.get("cell_selection_or_compensation_permitted") is not False
    ):
        raise AnonymousObjectRegistryV6Error("V6 aggregation authority differs")
    return value


def registry_receipt() -> Mapping[str, Any]:
    prereg = dict(load_preregistration())
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "appearance_ids": list(APPEARANCE_IDS),
        "arms": list(ARMS),
        "blocks": list(BLOCKS),
        "sigma_cell_indices": dict(SIGMA_CELL_INDICES),
        "phases": PHASES,
        "patch_grid": [PATCH_HEIGHT, PATCH_WIDTH],
        "appearances": [dict(item.receipt()) for item in APPEARANCES],
        "source_swap_exact_next_appearance_action": True,
        "source_swap_cycle": [
            [
                appearance.appearance_id,
                APPEARANCES[(index + 1) % len(APPEARANCES)].appearance_id,
            ]
            for index, appearance in enumerate(APPEARANCES)
        ],
        "preregistration_sha256": hashlib.sha256(
            PREREG_PATH.read_bytes()
        ).hexdigest(),
        "preregistration_object_digest": object_sha256(prereg),
        "anonymous_object_hypotheses": True,
        "semantic_role_ids": None,
        "canonical_role_phrases": None,
        "token_to_role": None,
        "variable_cardinality": True,
        "unrestricted_dustbin": True,
        "representation_admission_hard_false": True,
        "gpu_launch_authorized": prereg["claims"]["gpu_launch_authorized"],
        "launch_blocked_pending_independent_audit": prereg["claims"][
            "launch_blocked_pending_independent_audit"
        ],
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "APPEARANCES",
    "APPEARANCE_IDS",
    "ARMS",
    "BLOCKS",
    "METHOD",
    "PATCHES",
    "PATCH_HEIGHT",
    "PATCH_WIDTH",
    "PHASES",
    "PREREG_PATH",
    "SCHEMA_VERSION",
    "SIGMA_CELL_INDICES",
    "AnonymousAppearanceV6",
    "AnonymousObjectRegistryV6Error",
    "canonical_json_bytes",
    "load_preregistration",
    "object_sha256",
    "registry_receipt",
    "text_sha256",
]

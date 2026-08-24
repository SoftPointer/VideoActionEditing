#!/usr/bin/env python3
"""Frozen registry for the V7 factorial compatibility tube diagnostic.

V7 keeps the V6 projected visual-capture ABI, but expands the prompt authority
to the complete three visual states by three action captions.  The registry
contains no semantic object names, role-token partitions, learned parameters,
or launch authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import self_generated_anonymous_object_registry_v6 as v6_registry


METHOD = "bernini-self-generated-factorial-compatibility-tube-graph-probe-v7"
SCHEMA_VERSION = "bernini-self-generated-factorial-compatibility-registry-v7"
APPEARANCE_IDS = v6_registry.APPEARANCE_IDS
BLOCKS = v6_registry.BLOCKS
SIGMA_CELL_INDICES = v6_registry.SIGMA_CELL_INDICES
PHASES = v6_registry.PHASES
PATCH_HEIGHT = v6_registry.PATCH_HEIGHT
PATCH_WIDTH = v6_registry.PATCH_WIDTH
PATCHES = PATCH_HEIGHT * PATCH_WIDTH
CONTROL_ARMS = (
    "noop",
    "reverse",
    "static",
    "neutral",
    "paraphrase",
    "lexical_placebo",
)
BRANCHES = ("A_to_B", "B_to_A")
PREREG_PATH = (
    Path(__file__).absolute().parent
    / "assets"
    / "self_generated_factorial_compatibility_tube_graph_prereg_v7.json"
)

ACTION_CAPTIONS = MappingProxyType(
    {
        row.appearance_id: row.captions["action"]
        for row in v6_registry.APPEARANCES
    }
)
CONTROL_CAPTIONS = MappingProxyType(
    {
        row.appearance_id: MappingProxyType(
            {arm: row.captions[arm] for arm in CONTROL_ARMS}
        )
        for row in v6_registry.APPEARANCES
    }
)
_NEUTRAL_CAPTIONS = tuple(
    CONTROL_CAPTIONS[state_id]["neutral"] for state_id in APPEARANCE_IDS
)
if len(set(_NEUTRAL_CAPTIONS)) != 1:
    raise RuntimeError("V7 requires one byte-identical prompt-neutral caption")
IDENTICAL_NEUTRAL_CAPTION = _NEUTRAL_CAPTIONS[0]


class FactorialCompatibilityRegistryV7Error(ValueError):
    """A prompt-matrix, preregistration, or hard-claim authority differs."""


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
        raise FactorialCompatibilityRegistryV7Error(
            "value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise FactorialCompatibilityRegistryV7Error(
            "text digest requires a string"
        )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _index(appearance_id: str) -> int:
    try:
        return APPEARANCE_IDS.index(appearance_id)
    except ValueError as error:
        raise FactorialCompatibilityRegistryV7Error(
            "appearance id differs"
        ) from error


def factorial_relation(state_id: str, caption_id: str) -> str:
    """Return the frozen diagonal/clockwise/anti-clockwise matrix relation."""

    state_index = _index(state_id)
    caption_index = _index(caption_id)
    if state_index == caption_index:
        return "matched"
    if caption_index == (state_index + 1) % len(APPEARANCE_IDS):
        return "clockwise"
    if caption_index == (state_index - 1) % len(APPEARANCE_IDS):
        return "anti_clockwise"
    raise FactorialCompatibilityRegistryV7Error("factorial relation differs")


CLOCKWISE_OFF_DIAGONAL = tuple(
    (state_id, APPEARANCE_IDS[(index + 1) % len(APPEARANCE_IDS)])
    for index, state_id in enumerate(APPEARANCE_IDS)
)
ANTI_CLOCKWISE_OFF_DIAGONAL = tuple(
    (state_id, APPEARANCE_IDS[(index - 1) % len(APPEARANCE_IDS)])
    for index, state_id in enumerate(APPEARANCE_IDS)
)
FACTORIAL_KEYS = tuple(
    (state_id, caption_id)
    for state_id in APPEARANCE_IDS
    for caption_id in APPEARANCE_IDS
)
BRANCH_OFF_DIAGONAL_FOLDS = MappingProxyType(
    {
        "A_to_B": MappingProxyType(
            {
                "nuisance": CLOCKWISE_OFF_DIAGONAL,
                "heldout": ANTI_CLOCKWISE_OFF_DIAGONAL,
            }
        ),
        "B_to_A": MappingProxyType(
            {
                "nuisance": ANTI_CLOCKWISE_OFF_DIAGONAL,
                "heldout": CLOCKWISE_OFF_DIAGONAL,
            }
        ),
    }
)


@dataclass(frozen=True)
class FactorialPromptV7:
    state_appearance_id: str
    caption_appearance_id: str
    caption: str
    relation: str

    def __post_init__(self) -> None:
        expected_relation = factorial_relation(
            self.state_appearance_id, self.caption_appearance_id
        )
        expected_caption = ACTION_CAPTIONS[self.caption_appearance_id]
        if self.relation != expected_relation or self.caption != expected_caption:
            raise FactorialCompatibilityRegistryV7Error(
                "factorial prompt differs"
            )
        if self.caption != self.caption.strip() or not self.caption:
            raise FactorialCompatibilityRegistryV7Error(
                "factorial prompt is malformed"
            )

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "state_appearance_id": self.state_appearance_id,
            "caption_appearance_id": self.caption_appearance_id,
            "relation": self.relation,
            "caption": self.caption,
            "caption_sha256": text_sha256(self.caption),
            "semantic_role_inventory_exposed": False,
            "caption_token_partition_exposed": False,
        }
        return {**value, "digest": object_sha256(value)}


FACTORIAL_PROMPTS = tuple(
    FactorialPromptV7(
        state_id,
        caption_id,
        ACTION_CAPTIONS[caption_id],
        factorial_relation(state_id, caption_id),
    )
    for state_id, caption_id in FACTORIAL_KEYS
)


def nuisance_state_for_caption(branch: str, caption_id: str) -> str:
    """State carrying the same caption in the branch's nuisance fold."""

    if branch not in BRANCHES:
        raise FactorialCompatibilityRegistryV7Error("branch differs")
    rows = BRANCH_OFF_DIAGONAL_FOLDS[branch]["nuisance"]
    matches = [state_id for state_id, candidate in rows if candidate == caption_id]
    if len(matches) != 1:
        raise FactorialCompatibilityRegistryV7Error(
            "nuisance caption authority differs"
        )
    return matches[0]


def heldout_caption_for_state(branch: str, state_id: str) -> str:
    """Caption assigned to a state in the disjoint held-out off-diagonal."""

    if branch not in BRANCHES:
        raise FactorialCompatibilityRegistryV7Error("branch differs")
    rows = BRANCH_OFF_DIAGONAL_FOLDS[branch]["heldout"]
    matches = [caption_id for candidate, caption_id in rows if candidate == state_id]
    if len(matches) != 1:
        raise FactorialCompatibilityRegistryV7Error(
            "held-out caption authority differs"
        )
    return matches[0]


_NUMERIC_GATE_KEYS = (
    "primary_minimum_track_count",
    "primary_minimum_track_coverage",
    "primary_minimum_dynamic_edge_lifecycle_events",
    "noop_maximum_component_count",
    "static_to_primary_displacement_ratio_max",
    "static_zero_track_displacement_definition",
    "reverse_endpoint_direction_cosine_max",
    "phase_shuffle_to_primary_acceleration_ratio_min",
    "phase_shuffle_absolute_acceleration_floor",
    "paraphrase_support_iou_min",
    "paraphrase_endpoint_direction_cosine_min",
    "lexical_placebo_to_primary_component_ratio_max",
    "source_swap_to_primary_support_iou_max",
    "source_swap_evaluated_track_coverage_max",
    "source_swap_dynamic_edge_lifecycle_max",
)


def _plain_prereg_path() -> Path:
    original = PREREG_PATH.absolute()
    if not original.is_file() or original.is_symlink():
        raise FactorialCompatibilityRegistryV7Error(
            "V7 preregistration must be a plain file"
        )
    canonical = original.resolve(strict=True)
    if original != canonical:
        raise FactorialCompatibilityRegistryV7Error(
            "V7 preregistration path is not canonical"
        )
    return canonical


def load_preregistration() -> Mapping[str, Any]:
    path = _plain_prereg_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorialCompatibilityRegistryV7Error(
            "V7 preregistration cannot be read"
        ) from error
    if not isinstance(value, dict):
        raise FactorialCompatibilityRegistryV7Error(
            "V7 preregistration differs"
        )
    if (
        value.get("method") != METHOD
        or value.get("frozen_before_gpu_execution") is not True
    ):
        raise FactorialCompatibilityRegistryV7Error(
            "V7 preregistration authority differs"
        )
    scope = value.get("implementation_scope")
    claims = value.get("claims")
    if (
        not isinstance(scope, dict)
        or scope.get("registry_and_cpu_reducer_only") is not True
        or scope.get("gpu_runner_implemented") is not False
        or scope.get("factorial_prompt_embedding_runtime_binding_implemented")
        is not False
        or scope.get("gpu_launch_authorized") is not False
        or not isinstance(claims, dict)
        or claims.get("representation_admission_hard_false") is not True
        or claims.get("stable_transferable_action_representation_claimed")
        is not False
        or claims.get("scientific_claim_authorized") is not False
        or claims.get("renderer_or_decoder_authorized") is not False
        or claims.get("renderer_called") is not False
        or claims.get("decoder_called") is not False
        or claims.get("training_or_parameter_updates_authorized") is not False
        or claims.get("optimizer_created") is not False
        or claims.get("parameter_updates") != 0
        or claims.get("route_or_injection_authorized") is not False
        or claims.get("route_or_injection_called") is not False
        or claims.get("prompt_shuffle_control_executed") is not False
        or claims.get("heldout_transfer_control_executed") is not False
        or claims.get("gpu_launch_authorized") is not False
        or claims.get("launch_blocked_pending_independent_audit") is not True
    ):
        raise FactorialCompatibilityRegistryV7Error(
            "V7 hard claim boundary differs"
        )
    capture = value.get("capture")
    if (
        not isinstance(capture, dict)
        or capture.get("reuses_v6_projected_capture_abi") is not True
        or capture.get("blocks") != list(BLOCKS)
        or capture.get("sigma_cells") != dict(SIGMA_CELL_INDICES)
        or capture.get("phases") != PHASES
        or capture.get("patch_height") != PATCH_HEIGHT
        or capture.get("patch_width") != PATCH_WIDTH
        or capture.get("factorial_action_state_caption_shape") != [3, 3]
        or capture.get("control_arms_per_state") != list(CONTROL_ARMS)
        or capture.get("identical_prompt_neutral_caption_required_across_states")
        is not True
        or any(
            capture.get(name) is not False
            for name in (
                "text_key_or_value_used",
                "caption_token_offsets_used",
                "caption_role_partition_used",
                "fixed_semantic_role_inventory_used",
            )
        )
    ):
        raise FactorialCompatibilityRegistryV7Error(
            "V7 capture matrix differs"
        )
    factorial = value.get("factorial_interaction")
    if (
        not isinstance(factorial, dict)
        or factorial.get("full_three_by_three_required") is not True
        or factorial.get("clockwise_off_diagonal")
        != [list(row) for row in CLOCKWISE_OFF_DIAGONAL]
        or factorial.get("anti_clockwise_off_diagonal")
        != [list(row) for row in ANTI_CLOCKWISE_OFF_DIAGONAL]
        or factorial.get("nuisance_and_heldout_sets_disjoint") is not True
        or factorial.get("interaction_residual_stop_gradient_proposal_only")
        is not True
        or factorial.get("interaction_residual_used_as_descriptor") is not False
        or factorial.get("interaction_residual_used_as_reward") is not False
    ):
        raise FactorialCompatibilityRegistryV7Error(
            "V7 factorial interaction authority differs"
        )
    expected_folds = {
        "A_to_B": {
            "nuisance_off_diagonal": "clockwise",
            "heldout_source_swap_off_diagonal": "anti_clockwise",
        },
        "B_to_A": {
            "nuisance_off_diagonal": "anti_clockwise",
            "heldout_source_swap_off_diagonal": "clockwise",
        },
    }
    if factorial.get("branch_folds") != expected_folds:
        raise FactorialCompatibilityRegistryV7Error(
            "V7 off-diagonal cross-fit differs"
        )
    cross_fit = value.get("cross_fit")
    expected_a = [[phase, phase + 1] for phase in range(0, 20, 2)]
    expected_b = [[phase, phase + 1] for phase in range(1, 20, 2)]
    if (
        not isinstance(cross_fit, dict)
        or cross_fit.get("A_to_B_phase_pairs") != expected_a
        or cross_fit.get("B_to_A_phase_pairs") != expected_b
        or cross_fit.get("proposal_and_evaluation_layers_disjoint") is not True
        or cross_fit.get("proposal_and_evaluation_times_disjoint") is not True
        or cross_fit.get("evaluation_uses_interaction_residual") is not False
        or cross_fit.get("branchwise_and_required") is not True
    ):
        raise FactorialCompatibilityRegistryV7Error(
            "V7 cross-fit authority differs"
        )
    tubes = value.get("space_time_tubes")
    if (
        not isinstance(tubes, dict)
        or tubes.get("construction_domain") != [PHASES, PATCH_HEIGHT, PATCH_WIDTH]
        or tubes.get("branch_active_phase_mask")
        != "V6 disjoint time fold; A union B is all 21 phases"
        or tubes.get("joint_space_time_connected_components") is not True
        or tubes.get("independent_per_phase_slot_finalization_permitted")
        is not False
        or tubes.get("v6_unbalanced_ot_thresholds_reused_unchanged") is not True
        or tubes.get("unassigned_or_rejected_voxels_go_to_dustbin") is not True
        or tubes.get("variable_cardinality") is not True
        or tubes.get("no_forced_tube") is not True
    ):
        raise FactorialCompatibilityRegistryV7Error(
            "V7 space-time tube authority differs"
        )
    controls = value.get("controls")
    if (
        not isinstance(controls, dict)
        or controls.get("prompt_shuffle_executed") is not False
        or controls.get("heldout_transfer_executed") is not False
    ):
        raise FactorialCompatibilityRegistryV7Error(
            "V7 future-control boundary differs"
        )
    v6 = v6_registry.load_preregistration()
    gates = value.get("branchwise_diagnostic_gates")
    if not isinstance(gates, dict) or any(
        gates.get(key) != v6["branchwise_diagnostic_gates"].get(key)
        for key in _NUMERIC_GATE_KEYS
    ):
        raise FactorialCompatibilityRegistryV7Error(
            "V7 changed a V6 diagnostic threshold"
        )
    for section, keys in (
        (
            "discovery",
            (
                "component_seed_z",
                "component_soft_temperature_z",
                "absolute_energy_floor",
                "minimum_component_soft_mass",
                "minimum_component_support_patches",
                "maximum_components_per_phase_computational_cap",
                "spatial_concentration_top_fraction",
                "spatial_concentration_min",
            ),
        ),
        (
            "unbalanced_ot",
            tuple(v6["unbalanced_ot"]),
        ),
        (
            "dynamic_edges",
            ("soft_distance_temperature", "activation_affinity"),
        ),
        (
            "tracking",
            (
                "maximum_occlusion_gap",
                "minimum_track_observed_phases",
                "minimum_primary_track_coverage",
            ),
        ),
        (
            "cross_fit",
            (
                "correspondence_softmax_temperature",
                "correspondence_spatial_sigma",
                "neutral_visual_cosine_top_vs_median_margin_min",
                "correspondence_top_vs_median_margin_min",
                "correspondence_top10_mass_fraction_min",
            ),
        ),
    ):
        current = value.get(section)
        previous = v6.get(section)
        if not isinstance(current, dict) or any(
            current.get(key) != previous.get(key) for key in keys
        ):
            raise FactorialCompatibilityRegistryV7Error(
                f"V7 changed a V6 {section} threshold"
            )
    overall = value.get("overall_diagnostic_aggregation")
    if (
        not isinstance(overall, dict)
        or overall.get("expected_cell_count") != 9
        or overall.get("cell_selection_or_compensation_permitted") is not False
        or overall.get("missing_cell_fails") is not True
    ):
        raise FactorialCompatibilityRegistryV7Error(
            "V7 aggregation authority differs"
        )
    return value


def registry_receipt() -> Mapping[str, Any]:
    prereg = dict(load_preregistration())
    prompts = [dict(row.receipt()) for row in FACTORIAL_PROMPTS]
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "appearance_ids": list(APPEARANCE_IDS),
        "blocks": list(BLOCKS),
        "sigma_cell_indices": dict(SIGMA_CELL_INDICES),
        "phases": PHASES,
        "patch_grid": [PATCH_HEIGHT, PATCH_WIDTH],
        "factorial_shape": [len(APPEARANCE_IDS), len(APPEARANCE_IDS)],
        "factorial_prompts": prompts,
        "factorial_prompt_count": len(prompts),
        "clockwise_off_diagonal": [list(row) for row in CLOCKWISE_OFF_DIAGONAL],
        "anti_clockwise_off_diagonal": [
            list(row) for row in ANTI_CLOCKWISE_OFF_DIAGONAL
        ],
        "branch_off_diagonal_folds": {
            branch: {
                name: [list(row) for row in rows]
                for name, rows in BRANCH_OFF_DIAGONAL_FOLDS[branch].items()
            }
            for branch in BRANCHES
        },
        "control_arms": list(CONTROL_ARMS),
        "identical_neutral_caption_sha256": text_sha256(
            IDENTICAL_NEUTRAL_CAPTION
        ),
        "preregistration_sha256": hashlib.sha256(
            _plain_prereg_path().read_bytes()
        ).hexdigest(),
        "preregistration_object_digest": object_sha256(prereg),
        "reuses_v6_projected_capture_abi": True,
        "joint_space_time_tubes": True,
        "semantic_role_ids": None,
        "caption_token_partition": None,
        "fixed_slot_cardinality": None,
        "variable_cardinality": True,
        "unrestricted_dustbin": True,
        "representation_admission_hard_false": True,
        "prompt_shuffle_control_executed": False,
        "heldout_transfer_control_executed": False,
        "scientific_claim_authorized": False,
        "training_authorized": False,
        "renderer_or_decoder_authorized": False,
        "route_or_injection_authorized": False,
        "gpu_launch_authorized": False,
        "gpu_runner_implemented": False,
        "factorial_prompt_embedding_runtime_binding_implemented": False,
        "launch_blocked_pending_independent_audit": True,
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "ACTION_CAPTIONS",
    "ANTI_CLOCKWISE_OFF_DIAGONAL",
    "APPEARANCE_IDS",
    "BLOCKS",
    "BRANCHES",
    "BRANCH_OFF_DIAGONAL_FOLDS",
    "CLOCKWISE_OFF_DIAGONAL",
    "CONTROL_ARMS",
    "CONTROL_CAPTIONS",
    "FACTORIAL_KEYS",
    "FACTORIAL_PROMPTS",
    "IDENTICAL_NEUTRAL_CAPTION",
    "METHOD",
    "PATCHES",
    "PATCH_HEIGHT",
    "PATCH_WIDTH",
    "PHASES",
    "PREREG_PATH",
    "SCHEMA_VERSION",
    "SIGMA_CELL_INDICES",
    "FactorialCompatibilityRegistryV7Error",
    "FactorialPromptV7",
    "canonical_json_bytes",
    "factorial_relation",
    "heldout_caption_for_state",
    "load_preregistration",
    "nuisance_state_for_caption",
    "object_sha256",
    "registry_receipt",
    "text_sha256",
]

"""Immutable checkpoint identities used by the OmniVideo2-1.3B wrapper."""

from __future__ import annotations


OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID = (
    "omnivideo2-1.3b-adcee0a4-f269fe8c-72129ce9-v1"
)
OMNIVIDEO2_1_3B_UPSTREAM_REVISION = "adcee0a4a5b439ad3615f825298221b21177d4e3"
OMNIVIDEO2_1_3B_TRANSFORMER_SHA256 = (
    "f269fe8c6b35993bbb4ea340c535ee9893928ea215fb8c4be3d5e9f122d844d6"
)
OMNIVIDEO2_1_3B_SPECIAL_TOKENS_SHA256 = (
    "72129ce9ade25aa0fbf738c005d3dc090c1b6c45918580e1c683b6ecef726ad4"
)

# The official file serializes all six tensors.  The current unified forward
# consumes img/prp delimiters and does not consume the two ipl tensors.
OMNIVIDEO2_1_3B_SPECIAL_TOKEN_LAYOUT = (
    ("<img_st>", (6, 4096), True),
    ("<img_ed>", (6, 4096), True),
    ("<ipl_st>", (7, 4096), False),
    ("<ipl_ed>", (7, 4096), False),
    ("<prp_st>", (7, 4096), True),
    ("<prp_ed>", (7, 4096), True),
)
OMNIVIDEO2_1_3B_ACTIVE_SPECIAL_TOKEN_ROWS = sum(
    shape[0]
    for _key, shape, active in OMNIVIDEO2_1_3B_SPECIAL_TOKEN_LAYOUT
    if active
)
OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS = sum(
    shape[0] for _key, shape, _active in OMNIVIDEO2_1_3B_SPECIAL_TOKEN_LAYOUT
)

ACTION_ADAPTER_CHECKPOINT_FIELDS = frozenset(
    {
        "format",
        "step",
        "validated_config",
        "config_sha256",
        "manifest_sha256",
        "base_checkpoint_sha256",
        "checkpoint_contract_id",
        "special_tokens_sha256",
        "special_token_rows",
        "special_token_serialized_rows",
        "special_token_layout",
        "encoder_contract_sha256",
        "world_size",
        "preview_only",
        "temporal_smoke_only",
        "production_claim_forbidden",
        "source_revision",
        "source_archive_sha256",
        "activation_contract",
        "target_motion_tokens_used_by_renderer",
        "base_weights_saved",
        "lora_modules",
        "lora_state_dict",
        "motion_planner_state_dict",
        "rank0_cpu_rng_state",
        "rank0_device_rng_state",
    }
)


def special_token_layout_record() -> list[dict[str, object]]:
    """Return a JSON-safe copy of the pinned serialized/active layout."""

    return [
        {"key": key, "shape": list(shape), "active_in_unified_forward": active}
        for key, shape, active in OMNIVIDEO2_1_3B_SPECIAL_TOKEN_LAYOUT
    ]


def action_activation_contract_record() -> dict[str, str]:
    """Return the exact adapter/planner gate contract saved in every adapter."""

    return {
        "action_edit": "action_lora_on_predicted_plan_on",
        "identity_reconstruction": "action_lora_on_predicted_plan_on",
        "native_replay": "action_lora_on_predicted_plan_absent",
        "native_isolation_probe": "action_lora_zero_predicted_plan_absent",
        "non_action_deployment": "do_not_load_action_adapter",
    }

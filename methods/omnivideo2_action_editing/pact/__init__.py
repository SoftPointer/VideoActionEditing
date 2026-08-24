"""PACT package with lazy tensor exports.

Keeping package initialization dependency-free lets the signed-release and
manifest tools run without importing PyTorch. Tensor modules are imported only
when one of their public symbols is actually requested.
"""

from importlib import import_module


_EXPORT_MODULES = {
    "checkpoint": {
        "AdapterCheckpointError",
        "LoadedPactAdapters",
        "load_pact_adapter_bundle",
    },
    "conditioning": {
        "SourceLatentBudgetMetadata",
        "budget_source_latent",
        "erase_source_motion",
    },
    "flow": {
        "SharedNoiseSplice",
        "flow_noisy_latent",
        "reconstruct_x0",
        "shared_noise_local_latent_splice",
        "shared_noise_local_splice",
        "velocity_target",
    },
    "guidance": {
        "anchor_to_source_noisy",
        "gate_keep_edit_deltas",
        "source_noisy_anchor",
        "spatially_gated_guidance",
    },
    "lora": {
        "LoRALinear",
        "expected_lora_module_count",
        "inject_lora",
        "load_lora_weights",
        "load_lora_state_dict",
        "lora_config",
        "lora_scope_target_regex",
        "lora_state_dict",
        "save_lora_weights",
    },
    "losses": {
        "area_normalized_masked_loss",
        "boundary_consistency_loss",
        "edit_preserve_losses",
        "outside_temporal_difference_loss",
        "pact_reconstruction_losses",
    },
    "masks": {
        "boundary_ring",
        "dilate_and_feather",
        "dilate_mask",
        "erode_mask",
        "source_target_tube_union",
        "validate_video_mask",
    },
    "router": {
        "PromptConditionedMaskRouter",
        "bce_dice_loss",
        "router_loss_components",
    },
    "sampling": {
        "AnchoredEulerStep",
        "anchored_euler_flow_step",
        "euler_flow_step",
        "sample_anchored_flow",
        "validate_inference_sigmas",
        "wan_rational_shifted_sigmas",
    },
    "training": {
        "DiffSynthWanTrainingSample",
        "DiffSynthWanTrainingScheduler",
    },
}
_EXPORTS = {
    symbol: module_name
    for module_name, symbols in _EXPORT_MODULES.items()
    for symbol in symbols
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "AdapterCheckpointError",
    "AnchoredEulerStep",
    "DiffSynthWanTrainingSample",
    "DiffSynthWanTrainingScheduler",
    "LoRALinear",
    "LoadedPactAdapters",
    "PromptConditionedMaskRouter",
    "SharedNoiseSplice",
    "SourceLatentBudgetMetadata",
    "anchor_to_source_noisy",
    "area_normalized_masked_loss",
    "bce_dice_loss",
    "boundary_consistency_loss",
    "boundary_ring",
    "budget_source_latent",
    "dilate_and_feather",
    "dilate_mask",
    "edit_preserve_losses",
    "erase_source_motion",
    "erode_mask",
    "euler_flow_step",
    "expected_lora_module_count",
    "flow_noisy_latent",
    "gate_keep_edit_deltas",
    "inject_lora",
    "load_lora_weights",
    "load_pact_adapter_bundle",
    "load_lora_state_dict",
    "lora_config",
    "lora_scope_target_regex",
    "lora_state_dict",
    "outside_temporal_difference_loss",
    "pact_reconstruction_losses",
    "reconstruct_x0",
    "router_loss_components",
    "sample_anchored_flow",
    "save_lora_weights",
    "shared_noise_local_latent_splice",
    "shared_noise_local_splice",
    "source_noisy_anchor",
    "source_target_tube_union",
    "spatially_gated_guidance",
    "validate_video_mask",
    "validate_inference_sigmas",
    "velocity_target",
    "wan_rational_shifted_sigmas",
    "anchored_euler_flow_step",
]

if set(__all__) != set(_EXPORTS):
    raise RuntimeError("PACT lazy export table and __all__ differ")

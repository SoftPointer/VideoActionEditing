"""Mask-free, full-target action-editing core with lazy public exports."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "config": {
        "ACTION_CONFIG_FORMAT",
        "ACTION_TASK_TYPES",
        "CONTEXT_PADDING_MODES",
        "SPATIAL_PROFILES",
        "TEMPORAL_MODES",
        "LORA_SCOPES",
        "MOTION_TOKEN_DIM",
        "VLM_DIM",
        "ActionConfig",
        "ActionConfigError",
        "DataConfig",
        "FlowConfig",
        "LoraConfig",
        "ModelConfig",
        "OptimizerConfig",
        "PlannerConfig",
        "TrainingConfig",
        "load_action_config",
        "validate_action_config",
    },
    "dataset": {
        "ACTION_MANIFEST_FORMAT",
        "ACTION_MANIFEST_ROW_FIELDS",
        "ACTION_PAYLOAD_FIELDS",
        "ACTION_PAYLOAD_FORMAT",
        "ACTION_PROVENANCE_FIELDS",
        "ACTION_PROVENANCE_FORMAT",
        "ACTION_PROVENANCE_PRODUCTION_FIELDS",
        "ACTION_TRAINING_RELEASE_FIELDS",
        "ACTION_TRAINING_RELEASE_FORMAT",
        "ACTION_TRAINING_RELEASE_ROW_FIELDS",
        "ACTION_TRAINING_RELEASE_VERIFICATION_FIELDS",
        "ACTION_TRAINING_RELEASE_VERIFICATION_FORMAT",
        "ActionDatasetError",
        "ActionLatentDataset",
        "action_tensor_sha256",
        "collate_action_latents",
        "validate_action_payload",
    },
    "flow": {
        "DiffSynthWanTrainingSample",
        "DiffSynthWanTrainingScheduler",
        "FullTargetFlowBatch",
        "flow_noisy_latent",
        "full_target_flow_loss",
        "prepare_full_target_flow",
        "reconstruct_x0",
        "shifted_rectified_flow_sigma",
        "velocity_target",
    },
    "planner": {"TemporalMotionPlanPredictor", "motion_plan_loss"},
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


__all__ = sorted(_EXPORTS)

#!/usr/bin/env python3
"""Frozen exact81 pure-T2V to source-only-V2V Bernini canary.

The three matched arms share one official initial Gaussian and compare the
stock source-video-only ``v2v_apg`` endpoint, the stock target-only
``t2v_apg`` endpoint, and a same-state velocity homotopy from vendor-native
pure-T2V APG at high sigma to stock source-only V2V APG at low sigma.  The
pure T2V sampler receives no visual condition object.  The V2V sampler receives
exactly one full 81-frame source latent and no reference, first-frame, mask,
track, pose, or flow condition.  No optimizer or parameter update is allowed.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_branch_homotopy_canary as branch_base  # noqa: E402
import infer_native_identity_generation_canary as native  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import infer_source_value_residual_oracle as value_audit  # noqa: E402
from infer_native_self_guided_action_field_canary import (  # noqa: E402
    _strong_model_freeze_certificate,
)
from t2v_v2v_branch_homotopy_runtime_v1 import (  # noqa: E402
    T2VV2VBranchHomotopyRuntimeConfig,
    T2VV2VBranchHomotopyRuntimePatch,
)
import tri_branch_unipc as sampler_contract  # noqa: E402


METHOD = "frozen-bernini-pure-t2v-source-v2v-branch-homotopy-canary"
SCHEMA_VERSION = "bernini-t2v-v2v-branch-homotopy-canary-receipt-v1"
REGISTRY_SCHEMA_VERSION = "bernini-t2v-v2v-branch-homotopy-core4-v1"
CANONICAL_REGISTRY_RELATIVE = "assets/t2v_v2v_branch_homotopy_core4_v1.json"
CANONICAL_REGISTRY_SHA256 = (
    "dc0088fd3e43b7667a0f2bce7bb55e867553897bdddc8fd737d589b62fd84e43"
)
CELL_AUTHORITY_RELATIVE = "assets/native_branch_homotopy_core4_v1.json"
CELL_AUTHORITY_SHA256 = (
    "55d8f3800cc3088e0d9c22a0e5e6546a1eec8d629944f820eb12c295b0f71d13"
)
FRAME_COUNT = 81
LATENT_PHASES = 21
NUM_INFERENCE_STEPS = 40
FPS = 25
ULYSSES_SIZE = 4
ARM_ORDER = (
    "native-source-video-only-v2v-endpoint",
    "pure-target-only-t2v-endpoint",
    "t2v-v2v-branch-homotopy-095-075",
)
CELL_ORDER = branch_base.CELL_ORDER
WAVE_ORDER = branch_base.WAVE_ORDER
WAVE_CELLS = branch_base.WAVE_CELLS
COHORT_BY_WAVE = branch_base.COHORT_BY_WAVE
HOMOTOPY_ARM = ARM_ORDER[2]
GUIDANCE_BY_ARM = {
    ARM_ORDER[0]: "v2v_apg",
    ARM_ORDER[1]: "t2v_apg",
    ARM_ORDER[2]: "v2v_apg",
}
EXPECTED_FORWARD_COUNT_PER_STEP = {
    ARM_ORDER[0]: 2,
    ARM_ORDER[1]: 2,
    ARM_ORDER[2]: 4,
}
SCHEDULE_SHA256 = branch_base.SCHEDULE_SHA256
NATIVE_UNIPC40_SIGMAS = branch_base.NATIVE_UNIPC40_SIGMAS
HIGH_ENDPOINT_STEP_INDICES = tuple(range(0, 9))
TRANSITION_STEP_INDICES = tuple(range(9, 26))
LOW_ENDPOINT_STEP_INDICES = tuple(range(26, 40))

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class T2VV2VBranchHomotopyCanaryError(RuntimeError):
    """Raised before incomplete or ambiguous canary evidence is published."""


def _object_sha256(value: Any) -> str:
    return branch_base._object_sha256(value)


def _sha256_text(value: str) -> str:
    return branch_base._sha256_text(value)


def _load_canonical_registry(relative: str, expected_sha256: str) -> Mapping[str, Any]:
    path = METHOD_ROOT / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or native.legacy.file_sha256(path) != expected_sha256
    ):
        raise T2VV2VBranchHomotopyCanaryError(
            f"canonical registry authority differs: {relative}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise T2VV2VBranchHomotopyCanaryError(
            f"canonical registry authority is invalid: {relative}"
        ) from error
    if not isinstance(value, Mapping):
        raise T2VV2VBranchHomotopyCanaryError("canonical registry root differs")
    return value


def _registry_cell(registry: Mapping[str, Any], *, cell_id: str) -> Mapping[str, Any]:
    """Validate the complete new registry and its exact reused-cell authority."""

    canonical = _load_canonical_registry(
        CANONICAL_REGISTRY_RELATIVE, CANONICAL_REGISTRY_SHA256
    )
    if registry != canonical:
        raise T2VV2VBranchHomotopyCanaryError("registry differs from sealed authority")
    if (
        registry.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or registry.get("method") != METHOD
        or registry.get("arm_order") != list(ARM_ORDER)
    ):
        raise T2VV2VBranchHomotopyCanaryError("registry root differs")

    cell_authority = _load_canonical_registry(
        CELL_AUTHORITY_RELATIVE, CELL_AUTHORITY_SHA256
    )
    if (
        registry.get("population_design") != cell_authority.get("population_design")
        or registry.get("cells") != cell_authority.get("cells")
    ):
        raise T2VV2VBranchHomotopyCanaryError("reused four-cell authority differs")
    try:
        authoritative = branch_base._registry_cell(cell_authority, cell_id=cell_id)
    except Exception as error:
        raise T2VV2VBranchHomotopyCanaryError(str(error)) from error

    contract = registry.get("contract")
    low = contract.get("native_source_video_only_v2v_endpoint") if isinstance(contract, Mapping) else None
    high = contract.get("pure_target_only_t2v_endpoint") if isinstance(contract, Mapping) else None
    hom = contract.get("homotopy") if isinstance(contract, Mapping) else None
    prompt = contract.get("prompt_homotopy_disclosure") if isinstance(contract, Mapping) else None
    schedule = contract.get("apg_and_scheduler") if isinstance(contract, Mapping) else None
    embeddings = contract.get("embedding_contract") if isinstance(contract, Mapping) else None
    conditions = contract.get("condition_contract") if isinstance(contract, Mapping) else None
    if (
        not isinstance(low, Mapping)
        or low.get("guidance_mode") != "v2v_apg"
        or low.get("positive_task") != "mv2v"
        or low.get("full_source_video_count") != 1
        or low.get("source_reference_count") != 0
        or low.get("first_frame_condition_count") != 0
        or low.get("mask_track_pose_flow_count") != 0
        or low.get("forward_order") != ["V_negative", "V_action"]
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
        or high.get("forward_order") != ["none_negative", "none_action"]
        or high.get("transformer_forwards_per_step") != 2
        or not isinstance(hom, Mapping)
        or hom.get("high_sigma") != 0.95
        or hom.get("low_sigma") != 0.75
        or hom.get("endpoint_velocities_measured_at_same_x_t_timestep_and_sigma") is not True
        or hom.get("branch_apg_completed_independently_before_interpolation") is not True
        or hom.get("fp32_interpolation_before_one_official_scheduler_step") is not True
        or hom.get("hard_switch") is not False
        or not isinstance(prompt, Mapping)
        or prompt.get("same_target_action_caption_body") is not True
        or prompt.get("same_renderer_negative_embedding_object") is not True
        or prompt.get("task_prefix_and_visual_regime_change_together") is not True
        or prompt.get("shared_positive_embedding_across_endpoints") is not False
        or not isinstance(schedule, Mapping)
        or schedule.get("flow_shift_from_renderer_config") != 5.0
        or schedule.get("omega_text") != 4.0
        or schedule.get("eta") != 0.5
        or schedule.get("norm_threshold") != 50.0
        or schedule.get("momentum") != 0.0
        or schedule.get("unipc_steps") != NUM_INFERENCE_STEPS
        or schedule.get("exact40_shift5_schedule_sha256") != SCHEDULE_SHA256
        or not isinstance(embeddings, Mapping)
        or embeddings.get("source_v2v_positive_shape") != [1, 512, 4096]
        or embeddings.get("pure_t2v_positive_shape") != [1, 512, 4096]
        or embeddings.get("negative_shape") != [1, 512, 4096]
        or not isinstance(conditions, Mapping)
        or conditions.get("full_source_video_is_independently_vae_encoded_from_all_81_rgb_frames") is not True
        or conditions.get("pure_t2v_sampler_visual_conditions_all_none") is not True
        or conditions.get("source_v2v_sampler_has_exactly_one_full_source_video") is not True
        or conditions.get("source_references") is not False
        or conditions.get("first_frame_anchor") is not False
        or conditions.get("mask_track_pose_flow") is not False
        or contract.get("frame_count") != FRAME_COUNT
        or contract.get("latent_phases") != LATENT_PHASES
        or contract.get("num_inference_steps") != NUM_INFERENCE_STEPS
        or contract.get("frozen_model") is not True
        or contract.get("training") is not False
        or contract.get("optimizer") is not False
        or contract.get("parameter_update") is not False
    ):
        raise T2VV2VBranchHomotopyCanaryError("registry scientific contract differs")
    regions = schedule.get("homotopy_regions")
    if (
        not isinstance(regions, Mapping)
        or regions.get("high_pure_t2v_weight_one_step_indices") != list(HIGH_ENDPOINT_STEP_INDICES)
        or regions.get("strict_transition_step_indices") != list(TRANSITION_STEP_INDICES)
        or regions.get("low_source_v2v_weight_one_step_indices") != list(LOW_ENDPOINT_STEP_INDICES)
    ):
        raise T2VV2VBranchHomotopyCanaryError("registry exact40 regions differ")
    return authoritative


def build_mode_native_prompt(
    mode: str, caption: str, *, prompt_cleaner: Any
) -> str:
    if mode == "source-mv2v":
        try:
            return native.legacy.build_training_prompt(
                caption, prompt_cleaner=prompt_cleaner
            )
        except Exception as error:
            raise T2VV2VBranchHomotopyCanaryError(str(error)) from error
    if mode == "pure-t2v":
        try:
            return native.build_task_prompt(
                "t2v", caption, prompt_cleaner=prompt_cleaner
            )
        except Exception as error:
            raise T2VV2VBranchHomotopyCanaryError(str(error)) from error
    raise T2VV2VBranchHomotopyCanaryError("prompt mode differs")


def sampling_contract(arm: str, *, seed: int) -> Mapping[str, Any]:
    if arm not in ARM_ORDER or type(seed) is not int or not 0 <= seed < 2**63:
        raise T2VV2VBranchHomotopyCanaryError("sampling arm/seed differs")
    if arm == ARM_ORDER[1]:
        value = native.native_sampling_contract(
            "t2v", steps=NUM_INFERENCE_STEPS, seed=seed
        )
    else:
        value = native.legacy.sampler_contract(
            steps=NUM_INFERENCE_STEPS, seed=seed
        )
    if (
        value.get("guidance_mode") != GUIDANCE_BY_ARM[arm]
        or value.get("num_frames") != FRAME_COUNT
        or value.get("num_inference_steps") != NUM_INFERENCE_STEPS
        or value.get("flow_shift") != 5.0
        or value.get("omega_txt") != 4.0
        or value.get("eta") != 0.5
        or value.get("norm_threshold") != (50.0, 50.0)
        or value.get("momentum") != 0.0
    ):
        raise T2VV2VBranchHomotopyCanaryError("native sampling contract differs")
    return value


def conditions_for_arm(arm: str, *, source_latent: Any) -> Mapping[str, Any]:
    if arm not in ARM_ORDER:
        raise T2VV2VBranchHomotopyCanaryError("condition arm differs")
    if arm == ARM_ORDER[1]:
        return {
            "image_vae_latents": None,
            "multi_video_vae_latents": None,
            "multi_image_vae_latents": None,
        }
    if source_latent is None:
        raise T2VV2VBranchHomotopyCanaryError("source-only V2V latent is absent")
    return {
        "image_vae_latents": None,
        "multi_video_vae_latents": [source_latent],
        "multi_image_vae_latents": None,
    }


def validate_homotopy_runtime_trace(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    schedule = branch_base.pinned_exact40_schedule_receipt()
    rows = trace.get("trace") if isinstance(trace, Mapping) else None
    if (
        schedule.get("digest") != SCHEDULE_SHA256
        or trace.get("steps") != NUM_INFERENCE_STEPS
        or trace.get("transformer_forwards") != 4 * NUM_INFERENCE_STEPS
        or trace.get("low_source_v2v_forwards") != 2 * NUM_INFERENCE_STEPS
        or trace.get("high_pure_t2v_forwards") != 2 * NUM_INFERENCE_STEPS
        or trace.get("patch_vae_latent_calls") != 2 * NUM_INFERENCE_STEPS
        or trace.get("original_scheduler_calls") != NUM_INFERENCE_STEPS
        or trace.get("low_stock_apg_exact_parity_all_steps") is not True
        or trace.get("smoothstep_sigma_low") != 0.75
        or trace.get("smoothstep_sigma_high") != 0.95
        or trace.get("exact40_shift5_schedule_digest") != SCHEDULE_SHA256
        or trace.get("scheduler_mutation_surface") != "model_output_argument_only"
        or trace.get("runtime_source_identity_enforcement") != "external_canary_required"
        or trace.get("optimizer_created") is not False
        or trace.get("parameters_updated") is not False
        or not isinstance(rows, list)
        or len(rows) != NUM_INFERENCE_STEPS
    ):
        raise T2VV2VBranchHomotopyCanaryError("homotopy runtime trace root differs")
    branch_apg = trace.get("branch_apg")
    if (
        not isinstance(branch_apg, Mapping)
        or branch_apg.get("function")
        != "bernini.models.wan_diffusion.normalized_guidance"
        or branch_apg.get("one_condition_per_branch") is not True
        or branch_apg.get("omega_text") != 4.0
        or branch_apg.get("eta") != 0.5
        or branch_apg.get("norm_threshold") != 50.0
        or branch_apg.get("independent_momentum") != 0.0
    ):
        raise T2VV2VBranchHomotopyCanaryError("homotopy vendor APG differs")
    for index, (row, timestep, sigma) in enumerate(
        zip(rows, schedule["timesteps"], NATIVE_UNIPC40_SIGMAS)
    ):
        endpoint = (
            "high_pure_t2v_apg"
            if index in HIGH_ENDPOINT_STEP_INDICES
            else "transition" if index in TRANSITION_STEP_INDICES else "low_source_v2v_apg"
        )
        weight = row.get("high_pure_t2v_weight") if isinstance(row, Mapping) else None
        weight_ok = (
            weight == 1.0
            if index in HIGH_ENDPOINT_STEP_INDICES
            else isinstance(weight, float) and 0.0 < weight < 1.0
            if index in TRANSITION_STEP_INDICES
            else weight == 0.0
        )
        if (
            not isinstance(row, Mapping)
            or row.get("step_index") != index
            or row.get("timestep") != float(timestep)
            or row.get("sigma") != sigma
            or row.get("endpoint") != endpoint
            or not weight_ok
            or row.get("transformer_forwards") != 4
            or row.get("low_source_v2v_forwards") != 2
            or row.get("high_pure_t2v_forwards") != 2
            or row.get("original_scheduler_calls") != 1
            or row.get("patch_call_count") != 2
            or row.get("patch_source_ids") != [1.0, 0.0]
            or row.get("low_stock_apg_exact_parity") is not True
            or row.get("vendor_apg_function")
            != "bernini.models.wan_diffusion.normalized_guidance"
            or row.get("freeze_safe_no_grad_outputs") is not True
        ):
            raise T2VV2VBranchHomotopyCanaryError(
                f"homotopy exact40 step {index} differs"
            )
    return schedule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--cell-id", choices=CELL_ORDER, required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--runtime-source-closure-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit",
        default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=native.legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=native.legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_registry_sha256",
        "runtime_source_archive_sha256",
        "runtime_source_closure_sha256",
        "launcher_source_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise T2VV2VBranchHomotopyCanaryError(f"{name} differs")
    for name in (
        "runtime_source_revision",
        "expected_bernini_commit",
        "expected_veomni_commit",
    ):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            raise T2VV2VBranchHomotopyCanaryError(f"{name} differs")
    if args.expected_registry_sha256 != CANONICAL_REGISTRY_SHA256:
        raise T2VV2VBranchHomotopyCanaryError("canonical registry digest differs")
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise T2VV2VBranchHomotopyCanaryError("Bernini revision differs")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise T2VV2VBranchHomotopyCanaryError("VeOmni revision differs")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise T2VV2VBranchHomotopyCanaryError("checkpoint tree differs")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    try:
        registry_path, registry = branch_base._plain_json(args.registry, label="registry")
    except Exception as error:
        raise T2VV2VBranchHomotopyCanaryError(str(error)) from error
    if native.legacy.file_sha256(registry_path) != args.expected_registry_sha256:
        raise T2VV2VBranchHomotopyCanaryError("registry file digest differs")
    cell = _registry_cell(registry, cell_id=args.cell_id)
    output_dir = native._resolve_fresh_output_dir(args.output_dir)

    source_requested = Path(str(cell["source_video"])).expanduser()
    if not source_requested.is_absolute() or source_requested.is_symlink():
        raise T2VV2VBranchHomotopyCanaryError("source path differs")
    source_path = source_requested.resolve(strict=True)
    if (
        source_path != source_requested
        or not source_path.is_file()
        or native.legacy.file_sha256(source_path) != cell["source_video_sha256"]
    ):
        raise T2VV2VBranchHomotopyCanaryError("source bytes differ")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise T2VV2VBranchHomotopyCanaryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise T2VV2VBranchHomotopyCanaryError("attention heads do not divide Ulysses4")
    inference_file_hashes = native.legacy.validate_inference_source_files(bernini_root)
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("t2v") != native.TASK_SYSTEM_PROMPTS["t2v"]:
        raise T2VV2VBranchHomotopyCanaryError("runtime T2V system prompt differs")
    if SYSTEM_PROMPTS.get("mv2v") != native.legacy.MV2V_SYSTEM_PROMPT:
        raise T2VV2VBranchHomotopyCanaryError("runtime MV2V system prompt differs")
    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise T2VV2VBranchHomotopyCanaryError("runtime negative prompt differs")

    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != ULYSSES_SIZE
        or distributed.ulysses_size != ULYSSES_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise T2VV2VBranchHomotopyCanaryError("runtime requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=ULYSSES_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
        raise T2VV2VBranchHomotopyCanaryError(
            f"checkpoint validation failed: {checkpoint_rows[0]}"
        )
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    source_tensor, source_metadata, source_sha = source_audit.prepare_hashed_source_snapshot(
        source_path
    )
    bucket_hw = tuple(int(item) for item in cell["bucket_hw"])
    latent_shape = tuple(int(item) for item in cell["latent_shape"])
    if (
        source_sha != cell["source_video_sha256"]
        or source_metadata.get("frame_count") != FRAME_COUNT
        or tuple(source_metadata.get("source_derived_bucket_hw", ())) != bucket_hw
    ):
        raise T2VV2VBranchHomotopyCanaryError("source exact81 geometry differs")

    caption = str(cell["target_action_caption"])
    low_prompt = build_mode_native_prompt(
        "source-mv2v", caption, prompt_cleaner=prompt_clean
    )
    high_prompt = build_mode_native_prompt(
        "pure-t2v", caption, prompt_cleaner=prompt_clean
    )
    if low_prompt == high_prompt:
        raise T2VV2VBranchHomotopyCanaryError("mode-native prompts unexpectedly alias")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    low_ids, low_mask = native.legacy._tokenize_training_prompt(tokenizer, low_prompt)
    high_ids, high_mask = native.legacy._tokenize_training_prompt(tokenizer, high_prompt)
    negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
        tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    if float(config.shift) != 5.0 or config.use_unipc is not True:
        raise T2VV2VBranchHomotopyCanaryError("renderer exact40 shift5 UniPC differs")
    model = BerniniRendererModel(config)
    model.eval().requires_grad_(False)
    freeze_before = _strong_model_freeze_certificate(model)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    if distributed.rank == 0:
        source_pixels = source_tensor.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            source_latent = _vae_encode(vae, source_pixels).contiguous()
        del source_pixels
    else:
        source_latent = torch.empty(latent_shape, device=device, dtype=torch.float32)
    dist.broadcast(source_latent, src=0)
    if tuple(source_latent.shape) != latent_shape:
        raise T2VV2VBranchHomotopyCanaryError("full source latent geometry differs")
    condition_identity = native._all_rank_tensor_identity(
        source_latent,
        label=f"{args.cell_id}_full_source_video",
        world_size=ULYSSES_SIZE,
    )
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    model.to(device)
    model.t5_text_encoder.to(device)
    with torch.inference_mode():
        low_embeds = model.encode_prompt(low_ids.to(device), low_mask.to(device)).detach()
        high_embeds = model.encode_prompt(high_ids.to(device), high_mask.to(device)).detach()
        uncond_embeds = model.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
    expected_embed_shape = (1, 512, 4096)
    if any(
        tuple(value.shape) != expected_embed_shape
        for value in (low_embeds, high_embeds, uncond_embeds)
    ):
        raise T2VV2VBranchHomotopyCanaryError("mode-native embedding geometry differs")
    if torch.equal(low_embeds, high_embeds):
        raise T2VV2VBranchHomotopyCanaryError("T2V and MV2V positive embeddings alias")
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()

    diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
    wan_source_sha = sampler_contract.validate_runtime_source_identity(
        bernini_commit=bernini_revision,
        wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
    )
    sampler_contract._validate_scheduler_contract(
        diffusion.scheduler, expected_flow_shift=5.0
    )
    if diffusion.transformer_2 is not None:
        raise T2VV2VBranchHomotopyCanaryError(
            "canary requires Bernini-R 1.3B single DiT"
        )

    target_patch_tokens = LATENT_PHASES * (bucket_hw[0] // 16) * (bucket_hw[1] // 16)
    generated: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    initial_noise: dict[str, Any] = {}
    initial_noise_identities: dict[str, Any] = {}
    runtime_traces: dict[str, Any] = {}
    live_schedule_receipt: Optional[Mapping[str, Any]] = None
    with torch.inference_mode():
        for arm in ARM_ORDER:
            patch = None
            if arm == HOMOTOPY_ARM:
                patch = T2VV2VBranchHomotopyRuntimePatch(
                    diffusion,
                    t2v_action_prompt_embeds=high_embeds,
                    config=T2VV2VBranchHomotopyRuntimeConfig(
                        target_latent_shape=latent_shape,
                        expected_steps=NUM_INFERENCE_STEPS,
                        expected_num_frames=FRAME_COUNT,
                        expected_flow_shift=5.0,
                        omega_text=4.0,
                        eta=0.5,
                        norm_threshold=50.0,
                        momentum=0.0,
                    ),
                    expected_bernini_commit=bernini_revision,
                    observed_wan_diffusion_sha256=wan_source_sha,
                )
                patch.install()
            prompt_embeds = high_embeds if arm == ARM_ORDER[1] else low_embeds
            # Do not even hand the source object to the pure-T2V condition
            # constructor.  The source stays outside that sampler call graph.
            arm_source_latent = None if arm == ARM_ORDER[1] else source_latent
            sample_kwargs = {
                "prompt_embeds": prompt_embeds,
                "uncond_prompt_embeds": uncond_embeds,
                **conditions_for_arm(arm, source_latent=arm_source_latent),
                "width": bucket_hw[1],
                "height": bucket_hw[0],
                "device": device,
                **sampling_contract(arm, seed=int(cell["seed"])),
            }
            try:
                result, capture = native._sample_with_native_initial_noise_observer(
                    sample_fn=lambda kwargs=sample_kwargs: diffusion.sample(**kwargs),
                    wan_diffusion_module=wan_diffusion,
                    expected_shape=latent_shape,
                    expected_device=device,
                    expected_seed=int(cell["seed"]),
                )
            finally:
                if patch is not None:
                    patch.restore()
            if patch is not None:
                runtime_traces[arm] = dict(patch.finalize())
                live_schedule_receipt = validate_homotopy_runtime_trace(
                    runtime_traces[arm]
                )
            else:
                runtime_traces[arm] = {
                    "native_endpoint": True,
                    "guidance_mode": GUIDANCE_BY_ARM[arm],
                    "expected_transformer_forwards_per_step_from_pinned_vendor": EXPECTED_FORWARD_COUNT_PER_STEP[arm],
                    "runtime_hook_installed": False,
                    "vendor_source_sha256": wan_source_sha,
                    "source_video_count": 0 if arm == ARM_ORDER[1] else 1,
                    "source_reference_count": 0,
                }
            if (
                not isinstance(result, torch.Tensor)
                or tuple(result.shape) != latent_shape
                or result.dtype != torch.float32
                or result.requires_grad
                or result.grad_fn is not None
                or not bool(torch.isfinite(result).all().item())
            ):
                raise T2VV2VBranchHomotopyCanaryError("native sampler result differs")
            stored = result.detach().to(device="cpu").contiguous()
            generated[arm] = stored
            generated_identities[arm] = native._all_rank_tensor_identity(
                stored, label=f"{args.cell_id}_{arm}", world_size=ULYSSES_SIZE
            )
            initial_noise[arm] = capture
            initial_noise_identities[arm] = native._all_rank_tensor_identity(
                capture.tensor,
                label=f"{args.cell_id}_{arm}_official_initial_gaussian",
                world_size=ULYSSES_SIZE,
            )

    noise_hashes = {capture.raw_value_sha256 for capture in initial_noise.values()}
    if len(noise_hashes) != 1 or live_schedule_receipt is None:
        raise T2VV2VBranchHomotopyCanaryError(
            "arms did not share one official Gaussian and exact40 schedule"
        )
    trace_digest = _object_sha256(runtime_traces)
    trace_rows: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(trace_rows, trace_digest)
    if len(set(trace_rows)) != 1:
        raise T2VV2VBranchHomotopyCanaryError("branch traces differ across SP4 ranks")
    freeze_after = _strong_model_freeze_certificate(model)
    if freeze_after != freeze_before or any(p.requires_grad for p in model.parameters()):
        raise T2VV2VBranchHomotopyCanaryError("frozen model changed")
    model.to("cpu")
    torch.cuda.empty_cache()

    after_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            after_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            after_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(after_rows, src=0)
    if not isinstance(after_rows[0], Mapping) or after_rows[0].get("identity") != checkpoint_identity:
        raise T2VV2VBranchHomotopyCanaryError("checkpoint content changed")

    if distributed.rank == 0:
        output_dir.mkdir(parents=False, exist_ok=False)
        noise_artifacts = {
            arm: native._save_initial_noise_atomically(
                output_dir / f"{arm}.official-initial-gaussian.safetensors",
                initial_noise[arm],
                all_rank_identity=initial_noise_identities[arm],
            )
            for arm in ARM_ORDER
        }
        generated_for_decode = {
            arm: value.to(device=device).contiguous() for arm, value in generated.items()
        }
        try:
            outputs = native._save_outputs(
                output_dir=output_dir,
                generated=generated_for_decode,
                vae=vae,
                bucket_hw=bucket_hw,
                device=device,
                save_output_fn=save_output,
            )
        finally:
            generated_for_decode.clear()
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "cell_id": args.cell_id,
            "wave_id": cell["wave_id"],
            "cohort": cell["cohort"],
            "actor_kind": cell["actor_kind"],
            "input": {
                "source_video": str(source_path),
                "source_video_sha256": source_sha,
                "target_action_caption": caption,
                "target_action_caption_sha256": _sha256_text(caption),
                "seed_collision_evidence": cell.get("seed_collision_evidence"),
                "target_video": False,
                "custom_initial_noise": False,
                "generated_owner_media": False,
                "mask_track_pose_flow": False,
                "first_frame_anchor": False,
                "source_reference": False,
            },
            "prompts": {
                "same_target_action_caption_body": True,
                "same_verbatim_negative_prompt_all_arms": True,
                "same_negative_embedding_object_all_branches": True,
                "source_mv2v_full_prompt_sha256": _sha256_text(low_prompt),
                "pure_t2v_full_prompt_sha256": _sha256_text(high_prompt),
                "positive_task_prefix_and_visual_regime_change_together": True,
                "shared_positive_embedding_across_endpoints": False,
                "embedding_shape": list(expected_embed_shape),
            },
            "conditions": {
                "source_v2v_full_source_video_count": 1,
                "source_v2v_source_reference_count": 0,
                "pure_t2v_full_source_video_count": 0,
                "pure_t2v_source_reference_count": 0,
                "pure_t2v_visual_conditions_all_none": True,
                "homotopy_host_full_source_video_count": 1,
                "homotopy_host_source_reference_count": 0,
                "full_source_identity": condition_identity,
            },
            "sampling": {
                "frame_count": FRAME_COUNT,
                "latent_phases": LATENT_PHASES,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "fps": FPS,
                "seed": int(cell["seed"]),
                "arm_order": list(ARM_ORDER),
                "guidance_by_arm": dict(GUIDANCE_BY_ARM),
                "same_official_gaussian_all_arms": True,
                "official_gaussian_raw_sha256": next(iter(noise_hashes)),
                "target_patch_tokens": target_patch_tokens,
                "flow_shift_from_renderer_config": float(config.shift),
                "omega_text": 4.0,
                "eta": 0.5,
                "norm_threshold": 50.0,
                "momentum": 0.0,
                "schedule_sha256": SCHEDULE_SHA256,
                "schedule_receipt": live_schedule_receipt,
                "high_endpoint_step_indices": list(HIGH_ENDPOINT_STEP_INDICES),
                "transition_step_indices": list(TRANSITION_STEP_INDICES),
                "low_endpoint_step_indices": list(LOW_ENDPOINT_STEP_INDICES),
                "native_target_initialization": native.TARGET_INITIALIZATION,
            },
            "apg_parity_disclosure": {
                "standalone_source_v2v_endpoint_uses_stock_vendor_v2v_apg": True,
                "standalone_pure_t2v_endpoint_uses_stock_vendor_t2v_apg": True,
                "homotopy_low_reconstruction_exact_to_stock_all_steps": True,
                "homotopy_high_uses_stock_vendor_normalized_guidance": True,
                "homotopy_high_same_state_target_only_t2v_apg_contract_certified": True,
                "formal_official_high_apg_operator_claim_authorized": True,
                "standalone_and_homotopy_trajectory_equality_claim_authorized": False,
            },
            "runtime_traces": runtime_traces,
            "runtime_trace_digest": trace_digest,
            "generated_identities": generated_identities,
            "initial_noise_artifacts": noise_artifacts,
            "outputs": outputs,
            "checkpoint": checkpoint_identity,
            "freeze_certificate": freeze_after,
            "source_revisions": {
                "bernini": bernini_revision,
                "veomni": veomni_revision,
                "wan_diffusion_sha256": wan_source_sha,
                "runtime_method": args.runtime_source_revision,
                "runtime_source_archive_sha256": args.runtime_source_archive_sha256,
                "runtime_source_closure_sha256": args.runtime_source_closure_sha256,
                "launcher_source_sha256": args.launcher_source_sha256,
                "inference_files": inference_file_hashes,
            },
            "runtime_versions": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "diffusers": diffusers_version,
                "transformers": transformers_version,
            },
            "training_performed": False,
            "optimizer_created": False,
            "parameter_update": False,
            "scientific_or_action_editing_claim_authorized": False,
            "fit_and_confirmation_never_aggregated": True,
            "single_example_conclusion_authorized": False,
        }
        receipt["receipt_digest"] = _object_sha256(receipt)
        value_audit.write_receipt_atomically(output_dir / "receipt.json", receipt)
        print(native.legacy.canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    del source_latent, generated, initial_noise
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "CANONICAL_REGISTRY_SHA256",
    "CELL_ORDER",
    "COHORT_BY_WAVE",
    "GUIDANCE_BY_ARM",
    "HIGH_ENDPOINT_STEP_INDICES",
    "HOMOTOPY_ARM",
    "LOW_ENDPOINT_STEP_INDICES",
    "METHOD",
    "SCHEDULE_SHA256",
    "SCHEMA_VERSION",
    "T2VV2VBranchHomotopyCanaryError",
    "TRANSITION_STEP_INDICES",
    "WAVE_CELLS",
    "WAVE_ORDER",
    "_registry_cell",
    "build_mode_native_prompt",
    "build_parser",
    "conditions_for_arm",
    "sampling_contract",
    "validate_homotopy_runtime_trace",
]

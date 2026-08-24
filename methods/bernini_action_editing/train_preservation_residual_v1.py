#!/usr/bin/env python3
"""Train a source-only preservation residual on exact no-op reconstruction.

The Bernini/RV2V base is frozen.  For every registered UniPC coordinate this
entry point constructs

    x_target(sigma) = (1 - sigma) * z_clean + sigma * epsilon
    x_source(sigma) = (1 - sigma) * z_clean + sigma * epsilon
    v*              = epsilon - z_clean

with the *same epsilon object* for the target and source condition.  Three
clean source references remain independently VAE-encoded RGB-frame
conditions.  A disabled-adapter forward supplies the frozen no-op base field
``v0`` and the adapter is optimized with the single objective

    MSE((v_phi - v0), stopgrad(v* - v0)).

This is algebraically equal to no-op flow matching while making the scientific
contract explicit: the action model is not retrained and the only learned
quantity is the correction to the frozen base's source reconstruction field.
No synthetic target, action anchor, action score, VLM score, or feature reward
is consumed.

The physical profile is single-node WORLD8 DP2 x Ulysses-SP4.  The two logical
source rows execute concurrently as the two DP replicas.  Every positive coordinate of the exact40
scheduler receives one optimizer step.  Rank 8 is the
main arm and rank 2 is the preregistered low-capacity variant.  This pretext
alone makes no decoded action-preservation or method-success claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import inference_sigma_strata as exact40  # noqa: E402
import source_noised_ladder_v1 as source_ladder  # noqa: E402


METHOD_NAME = "bernini-preservation-only-base-prior-residual-v1"
RUN_RECEIPT_SCHEMA = "bernini-preservation-residual-training-receipt-v1"
HISTORY_SCHEMA = "bernini-preservation-residual-step-history-v1"
ADAPTER_FILE_SCHEMA = "bernini-preservation-residual-adapter-file-v1"
MODE = "preservation-residual-v1"
FRAME_COUNT = 81
LATENT_PHASES = 21
REFERENCE_PHASES = 1
LOGICAL_ARM_COUNT = 2
OPTIMIZER_STEPS = 40
CHECKPOINT_INTERVAL = 20
SUPPORTED_OPTIMIZER_STEPS = (CHECKPOINT_INTERVAL, OPTIMIZER_STEPS)
LOADABLE_CHECKPOINT_STEPS = (0, CHECKPOINT_INTERVAL, OPTIMIZER_STEPS)
SP_SIZE = 4
PRESERVATION_TOPOLOGY_PROFILE = "world8-dp2-sp4"
MAIN_LORA_RANK = 8
LOW_CAPACITY_LORA_RANK = 2
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 20260813
REGISTERED_SCHEDULE_INDICES = tuple(range(OPTIMIZER_STEPS))
EXPECTED_EXACT40_SCHEDULE_SHA256 = (
    "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
)
EXACT_NOOP_INSTRUCTION = (
    "Keep the source video exactly unchanged, including every subject, "
    "appearance, action, camera motion, background, timing, and composition."
)
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SourceNoisedCarrierTrainingError(RuntimeError):
    """Raised before an ambiguous update or artifact publication."""


@dataclass(frozen=True)
class PreservationTopology:
    profile: str = PRESERVATION_TOPOLOGY_PROFILE
    world_size: int = 8
    sp_size: int = 4
    dp_size: int = 2
    sp_group_ranks: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3), (4, 5, 6, 7))
    dp_group_ranks: tuple[tuple[int, ...], ...] = (
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )


def fail(message: str) -> NoReturn:
    raise SourceNoisedCarrierTrainingError(message)


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
        raise SourceNoisedCarrierTrainingError(
            "value is not canonical JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if type(value) is not str or pattern.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-{1 if length == 40 else 256}")
    return value


@dataclass(frozen=True)
class RegisteredCarrierCoordinate:
    optimizer_step_zero_based: int
    schedule_index: int
    timestep: int
    sigma: float
    sigma_float32_be_hex: str

    def receipt(self) -> dict[str, Any]:
        return {
            "optimizer_step_zero_based": self.optimizer_step_zero_based,
            "optimizer_step_one_based": self.optimizer_step_zero_based + 1,
            "schedule_index": self.schedule_index,
            "timestep_int64": self.timestep,
            "sigma": self.sigma,
            "sigma_float32_be_hex": self.sigma_float32_be_hex,
        }


def validate_registered_schedule(
    schedule_indices: Any = REGISTERED_SCHEDULE_INDICES,
) -> tuple[RegisteredCarrierCoordinate, ...]:
    """Return the only optimizer schedule authorized for Stage-B v1."""

    if exact40.SCHEDULE_SHA256 != EXPECTED_EXACT40_SCHEDULE_SHA256:
        fail("exact40 schedule SHA differs from the Stage-B pin")
    if (
        type(schedule_indices) is not tuple
        or schedule_indices != REGISTERED_SCHEDULE_INDICES
    ):
        fail("custom or reordered Stage-B schedule is forbidden")
    if (
        len(exact40.PINNED_TIMESTEPS) != OPTIMIZER_STEPS
        or len(exact40.PINNED_POSITIVE_SIGMAS) != OPTIMIZER_STEPS
        or len(exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX) != OPTIMIZER_STEPS
    ):
        fail("exact40 positive schedule closure differs")
    coordinates = tuple(
        RegisteredCarrierCoordinate(
            optimizer_step_zero_based=step,
            schedule_index=index,
            timestep=exact40.PINNED_TIMESTEPS[index],
            sigma=exact40.PINNED_POSITIVE_SIGMAS[index],
            sigma_float32_be_hex=exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                index
            ],
        )
        for step, index in enumerate(schedule_indices)
    )
    if (
        len(coordinates) != OPTIMIZER_STEPS
        or any(
            left.sigma <= right.sigma
            for left, right in zip(coordinates, coordinates[1:])
        )
    ):
        fail("preservation coordinates must be exact40 decreasing positive sigmas")
    return coordinates


def training_coordinates(optimizer_steps: Any) -> tuple[RegisteredCarrierCoordinate, ...]:
    """Return an exact20 prefix checkpoint or the full exact40 schedule.

    The 20-step arm exists only to expose training dynamics under the same
    deterministic prefix as the formal 40-step run.  It has no authority to
    claim Stage-B or action-editing success.
    """

    if (
        type(optimizer_steps) is not int
        or optimizer_steps not in SUPPORTED_OPTIMIZER_STEPS
    ):
        fail("preservation v1 supports only the exact20 prefix or full exact40")
    return validate_registered_schedule()[:optimizer_steps]


def validate_coordinate(coordinate: Any) -> RegisteredCarrierCoordinate:
    if not isinstance(coordinate, RegisteredCarrierCoordinate):
        fail("preservation coordinate type differs")
    if not 0 <= coordinate.optimizer_step_zero_based < OPTIMIZER_STEPS:
        fail("preservation optimizer coordinate lies outside exact40")
    expected = validate_registered_schedule()[coordinate.optimizer_step_zero_based]
    if (
        coordinate.schedule_index != expected.schedule_index
        or coordinate.timestep != expected.timestep
        or coordinate.sigma != expected.sigma
        or coordinate.sigma_float32_be_hex != expected.sigma_float32_be_hex
    ):
        fail("preservation coordinate is outside exact40")
    return coordinate


def validate_scientific_claim_contract(
    *,
    inversion_claimed: Any = False,
    exact_roundtrip_claimed: Any = False,
    semantic_method_success_claimed: Any = False,
) -> None:
    if (
        inversion_claimed is not False
        or exact_roundtrip_claimed is not False
        or semantic_method_success_claimed is not False
    ):
        fail("forward-noised source pretext cannot claim inversion or method success")


def register_preservation_topology(runtime: Any) -> Any:
    """Describe the stable runtime's native WORLD8 DP2/SP4 default."""

    candidate = PreservationTopology()
    if (
        candidate.profile != PRESERVATION_TOPOLOGY_PROFILE
        or candidate.world_size != 8
        or candidate.sp_size != SP_SIZE
        or candidate.dp_size != LOGICAL_ARM_COUNT
        or getattr(runtime, "WORLD_SIZE", None) != candidate.world_size
        or getattr(runtime, "SP_SIZE", None) != candidate.sp_size
        or getattr(runtime, "DP_SIZE", None) != candidate.dp_size
        or getattr(runtime, "SP_GROUP_RANKS", None) != candidate.sp_group_ranks
        or getattr(runtime, "DP_GROUP_RANKS", None) != candidate.dp_group_ranks
    ):
        fail("WORLD8 preservation topology differs")
    return candidate


def authorize_optimizer_step(
    *, completed_control_arms: Any, completed_backward_arms: Any
) -> bool:
    """Gate only on complete arms, never on sigma or block depth."""

    expected_options = ((0,), (1,))
    if (
        type(completed_control_arms) is not tuple
        or completed_control_arms not in expected_options
        or type(completed_backward_arms) is not tuple
        or completed_backward_arms != completed_control_arms
    ):
        fail("optimizer step requires one complete DP-owned logical arm")
    return True


def publish_preservation_output(
    runtime: Any,
    output: Path,
    stage: Path,
    receipt: Optional[Mapping[str, Any]],
    rank: int,
    world_group: Any,
    *,
    rank_zero_error: Optional[str] = None,
) -> None:
    """Publish one create-only bundle with the stable source-self runtime."""

    import torch.distributed as dist

    publication: list[Any] = [None]
    if rank == 0:
        try:
            if rank_zero_error is not None:
                fail(rank_zero_error)
            if receipt is None:
                fail("rank-zero publication receipt is missing")
            if output.exists() or output.is_symlink():
                fail("training output appeared before publication")
            runtime.verify_staged_run_bundle(stage, receipt)
            runtime.fsync_directory(stage)
            os.rename(stage, output)
            runtime.fsync_directory(output.parent)
            runtime.verify_staged_run_bundle(output, receipt)
            publication[0] = {
                "ok": True,
                "output": str(output),
                "receipt_digest": receipt.get("receipt_digest"),
            }
        except Exception as error:
            publication[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(publication, src=0, group=world_group)
    result = publication[0]
    if (
        not isinstance(result, Mapping)
        or result.get("ok") is not True
        or result.get("output") != str(output)
    ):
        fail(f"cannot publish preservation output: {result!r}")


def fixed_plan_receipt(optimizer_steps: int = OPTIMIZER_STEPS) -> dict[str, Any]:
    validate_scientific_claim_contract()
    coordinates = training_coordinates(optimizer_steps)
    value = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "exact40_schedule_sha256": EXPECTED_EXACT40_SCHEDULE_SHA256,
        "registered_schedule_indices": list(REGISTERED_SCHEDULE_INDICES),
        "training_schedule_indices": [
            item.schedule_index for item in coordinates
        ],
        "registered_coordinates": [item.receipt() for item in validate_registered_schedule()],
        "training_coordinates": [item.receipt() for item in coordinates],
        "optimizer_steps": optimizer_steps,
        "optimizer_step_per_training_coordinate": True,
        "optimizer_step_per_exact40_coordinate": optimizer_steps == OPTIMIZER_STEPS,
        "all_registered_strata_optimizer_authorized": optimizer_steps == OPTIMIZER_STEPS,
        "formal_exact40_complete": optimizer_steps == OPTIMIZER_STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "late_or_low_sigma_zero_update_gate_present": False,
        "schedule_customization_authorized": False,
        "schedule_policy": "one_update_at_each_exact40_positive_coordinate",
        "target_equation": "x_target_sigma=(1-sigma)*z_clean+sigma*epsilon",
        "source_condition_equation": "x_source_sigma=(1-sigma)*z_clean+sigma*epsilon",
        "target_velocity_equation": "v*=epsilon-z_clean",
        "same_epsilon_target_source_required": True,
        "clean_source_references_required": True,
        "forward_noising_only": True,
        "inversion_claimed": False,
        "reverse_ode_executed": False,
        "solver_state_replayed": False,
        "exact_roundtrip_claimed": False,
        "semantic_method_success_claimed": False,
        "synthetic_target_consumed": False,
        "action_reward_consumed": False,
    }
    return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class PreparedCondition:
    input_patches: Any
    rotary: Any
    layout: Any
    coordinate: RegisteredCarrierCoordinate


@dataclass(frozen=True)
class PreparedCarrierStep:
    main: PreparedCondition
    controls: Mapping[str, PreparedCondition]
    target_velocity: Any
    epsilon_sha256: str
    tensor_identities: Mapping[str, str]
    shared_noise_binding: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedLogicalArm:
    logical_arm: int
    row_index: int
    iid: str
    wrong_ref_iid: str
    noise_seed: int
    main: PreparedCondition
    target_velocity: Any
    frozen_base_prediction: Any
    residual_teacher: Any
    epsilon_sha256: str
    tensor_identities: Mapping[str, str]
    shared_noise_binding: Mapping[str, Any]
    control_metrics: Mapping[str, float]
    control_names: tuple[str, ...]
    absent_reference_layout: Mapping[str, Any]


def _condition(
    *,
    base: Any,
    role: Any,
    donor: Any,
    references: Sequence[Any],
    noisy_target: Any,
    coordinate: RegisteredCarrierCoordinate,
    rope: Any,
    device: Any,
) -> PreparedCondition:
    import torch

    if len(references) not in {0, role.REFERENCE_COUNT}:
        fail("image references must be truly absent or exactly three modes")
    validate_coordinate(coordinate)
    donor_patches = base.pack_latent_patches(donor, phases=LATENT_PHASES)
    reference_patches = [
        base.pack_latent_patches(item, phases=REFERENCE_PHASES)
        for item in references
    ]
    target_patches = base.pack_latent_patches(
        noisy_target, phases=LATENT_PHASES
    )
    layout = role.TokenRoleLayout.contiguous(
        donor_tokens=int(donor_patches.shape[0]),
        reference_tokens=[int(item.shape[0]) for item in reference_patches],
        target_tokens=int(target_patches.shape[0]),
    )
    patches = torch.cat(
        (donor_patches, *reference_patches, target_patches), dim=0
    ).to(device)
    donor_rope = rope(donor.unsqueeze(0).to(device), source_id=1)
    reference_rope = [
        rope(item.unsqueeze(0).to(device), source_id=index + 2)
        for index, item in enumerate(references)
    ]
    target_rope = rope(noisy_target.unsqueeze(0).to(device), source_id=0)
    rotary = torch.cat((donor_rope, *reference_rope, target_rope), dim=2)
    rotary = rotary.squeeze(0).permute(1, 0, 2).contiguous()
    if (
        int(patches.shape[0]) != layout.total_tokens
        or int(rotary.shape[0]) != layout.total_tokens
    ):
        fail("donor+refs+target pack geometry differs")
    return PreparedCondition(patches, rotary, layout, coordinate)


def _shared_noise_state(source: Any, epsilon: Any, sigma: float) -> Any:
    try:
        return source_ladder.shared_noise_source_state(source, epsilon, sigma)
    except source_ladder.SourceNoisedLadderError as error:
        raise SourceNoisedCarrierTrainingError(str(error)) from error


def prepare_carrier_step(
    *,
    base: Any,
    role: Any,
    runtime: Any,
    clean: Any,
    wrong_clean: Any,
    correct_refs: Sequence[Any],
    wrong_refs: Sequence[Any],
    epsilon: Any,
    coordinate: RegisteredCarrierCoordinate,
    rope: Any,
    device: Any,
) -> PreparedCarrierStep:
    if len(correct_refs) != 3 or len(wrong_refs) != 3:
        fail("correct/wrong reference triplets are required")
    validate_coordinate(coordinate)

    noisy_target = _shared_noise_state(clean, epsilon, coordinate.sigma)
    velocity = (epsilon - clean).detach().contiguous()
    target_velocity = base.packed_output_field(
        base.pack_latent_patches(velocity, phases=LATENT_PHASES)
    ).to(device)
    source_state = _shared_noise_state(clean, epsilon, coordinate.sigma)
    wrong_source_state = _shared_noise_state(
        wrong_clean, epsilon, coordinate.sigma
    )
    condition_kwargs = {
        "base": base,
        "role": role,
        "noisy_target": noisy_target,
        "coordinate": coordinate,
        "rope": rope,
        "device": device,
    }
    controls: dict[str, PreparedCondition] = {
        "source_correct_refs": _condition(
            donor=source_state,
            references=correct_refs,
            **condition_kwargs,
        ),
        "wrong_source_wrong_refs": _condition(
            donor=wrong_source_state,
            references=wrong_refs,
            **condition_kwargs,
        ),
        "source_refs_absent": _condition(
            donor=source_state,
            references=(),
            **condition_kwargs,
        ),
    }
    main = controls["source_correct_refs"]
    for name, item in controls.items():
        if name != "source_refs_absent" and item.layout.receipt() != main.layout.receipt():
            fail("three-reference control token layouts differ")
    absent = controls["source_refs_absent"].layout
    if (
        absent.reference_tokens != ()
        or absent.reference_token_total != 0
        or absent.condition_tokens != absent.donor_tokens
        or absent.total_tokens >= main.layout.total_tokens
    ):
        fail("absent-reference control still contains reference tokens")

    identities = {
        "clean_target": runtime.tensor_sha256(clean),
        "wrong_clean_source": runtime.tensor_sha256(wrong_clean),
        "epsilon": runtime.tensor_sha256(epsilon),
        "noisy_target": runtime.tensor_sha256(noisy_target),
        "target_velocity": runtime.tensor_sha256(velocity),
        "noised_source_condition": runtime.tensor_sha256(source_state),
        "noised_wrong_source_condition": runtime.tensor_sha256(wrong_source_state),
        **{
            f"correct_clean_ref_{index}": runtime.tensor_sha256(value)
            for index, value in enumerate(correct_refs)
        },
        **{
            f"wrong_clean_ref_{index}": runtime.tensor_sha256(value)
            for index, value in enumerate(wrong_refs)
        },
    }
    if (
        identities["clean_target"] == identities["wrong_clean_source"]
        or identities["noised_source_condition"]
        == identities["noised_wrong_source_condition"]
    ):
        fail("correct and wrong source identities are degenerate")
    target_recheck = _shared_noise_state(clean, epsilon, coordinate.sigma)
    source_recheck = _shared_noise_state(clean, epsilon, coordinate.sigma)
    binding_value = {
        "equation": "x=(1-sigma)*z+sigma*epsilon",
        "schedule_index": coordinate.schedule_index,
        "timestep_int64": coordinate.timestep,
        "sigma_float32_be_hex": coordinate.sigma_float32_be_hex,
        "epsilon_sha256": identities["epsilon"],
        "target_state_sha256": identities["noisy_target"],
        "source_condition_state_sha256": identities["noised_source_condition"],
        "same_epsilon_object_reused_during_target_and_source_construction": True,
        "target_formula_recomputed_and_equal": runtime.tensor_sha256(
            target_recheck
        )
        == identities["noisy_target"],
        "source_formula_recomputed_and_equal": runtime.tensor_sha256(
            source_recheck
        )
        == identities["noised_source_condition"],
        "same_sigma_registered_coordinate_reused": True,
        "clean_source_references_routed": True,
        "references_independently_encoded_from_source_rgb": True,
        "reference_from_video_posterior_slice": False,
        "forward_noising_only": True,
        "inversion_claimed": False,
        "reverse_ode_executed": False,
        "solver_state_replayed": False,
        "exact_roundtrip_claimed": False,
    }
    if not all(
        binding_value[name] is True
        for name in (
            "same_epsilon_object_reused_during_target_and_source_construction",
            "target_formula_recomputed_and_equal",
            "source_formula_recomputed_and_equal",
            "same_sigma_registered_coordinate_reused",
            "clean_source_references_routed",
        )
    ):
        fail("same-epsilon source-carrier binding could not be verified")
    binding = {**binding_value, "digest": object_sha256(binding_value)}
    return PreparedCarrierStep(
        main=main,
        controls=controls,
        target_velocity=target_velocity,
        epsilon_sha256=identities["epsilon"],
        tensor_identities=identities,
        shared_noise_binding=binding,
    )


def _prediction(
    *,
    role: Any,
    renderer: Any,
    transformer: Any,
    condition: PreparedCondition,
    text_lens: Any,
    text_embs: Any,
) -> Any:
    import torch

    coordinate = validate_coordinate(condition.coordinate)
    embedded = transformer.patch_embedding(condition.input_patches).flatten(1).unsqueeze(0)
    rotary = condition.rotary.permute(1, 0, 2).unsqueeze(0)
    value = renderer.diff_dec.shared_step(
        model_id="transformer_1",
        noisy_latents=embedded,
        timesteps=embedded.new_tensor([coordinate.timestep], dtype=torch.int64),
        cond_embeds=text_embs,
        rotary_embs=rotary,
        batch_vae_seqlen=[condition.layout.total_tokens],
        batch_text_seqlen=text_lens,
    )
    start = condition.layout.condition_tokens
    target = value[:, start : start + condition.layout.target_tokens, :]
    if tuple(target.shape) != (
        1,
        condition.layout.target_tokens,
        role.PATCH_VALUES,
    ):
        fail("target-row prediction geometry differs")
    return target


def _relative_l2(value: Any, reference: Any) -> float:
    numerator = (value.float() - reference.float()).square().mean().sqrt()
    denominator = reference.float().square().mean().sqrt().clamp_min(1.0e-12)
    result = float((numerator / denominator).item())
    if not math.isfinite(result):
        fail("causal control relative L2 is non-finite")
    return result


def _noise_seed(base_seed: int, optimizer_step: int, logical_arm: int) -> int:
    raw = (
        f"{base_seed}\0preservation-residual-v1\0{optimizer_step}\0{logical_arm}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31)


def _atomic_adapter_safetensors(
    path: Path, adapter: Any, runtime: Any, *, lora_rank: int
) -> None:
    import torch
    from safetensors.torch import save_file

    tensors = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, parameter in adapter.trainable_named_parameters()
    }
    metadata = {
        "schema_version": ADAPTER_FILE_SCHEMA,
        "role_adapter_schema_version": str(adapter.receipt()["schema_version"]),
        "block_indices_json": canonical_json_bytes(
            list(adapter.block_indices)
        ).decode("ascii"),
        "projections_json": canonical_json_bytes(
            ["attn1.to_q", "attn1.to_out.0"]
        ).decode("ascii"),
        "target_row_only": "true",
        "role_embedding": "donor_reference_target",
        "lora_rank": str(lora_rank),
        "lora_alpha_hex": float(lora_rank).hex(),
        "exact40_schedule_sha256": EXPECTED_EXACT40_SCHEDULE_SHA256,
        "registered_schedule_indices_json": canonical_json_bytes(
            list(REGISTERED_SCHEDULE_INDICES)
        ).decode("ascii"),
        "target_and_source_same_epsilon": "true",
        "forward_noising_only": "true",
        "inversion_claimed": "false",
        "matched_carrier_runtime_required": "true",
        "objective": "mse_adapted_minus_base_vs_target_minus_base",
        "action_reward_consumed": "false",
        "synthetic_target_consumed": "false",
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".safetensors",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        runtime.durable_file_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _checkpoint_receipt(
    *,
    optimizer_step: int,
    trajectory_optimizer_steps: int,
    adapter_rank: int,
    initial_digest: str,
    final_digest: str,
    dataset: Any,
    args: argparse.Namespace,
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    if optimizer_step not in LOADABLE_CHECKPOINT_STEPS:
        fail("checkpoint step is outside the registered cadence")
    if trajectory_optimizer_steps != OPTIMIZER_STEPS:
        fail("immutable cadence checkpoints require one continuous exact40 run")
    receipt: dict[str, Any] = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "complete": True,
        "mode": MODE,
        "checkpoint_bundle": True,
        "continuous_trajectory": True,
        "trajectory_optimizer_steps": trajectory_optimizer_steps,
        "optimizer_steps": optimizer_step,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "registered_schedule_indices": list(REGISTERED_SCHEDULE_INDICES),
        "training_schedule_indices": list(
            REGISTERED_SCHEDULE_INDICES[:optimizer_step]
        ),
        "base_frozen": True,
        "frozen_base_action_prior_not_retrained": True,
        "adapter_rank": adapter_rank,
        "initial_adapter_sha256": initial_digest,
        "final_adapter_sha256": final_digest,
        "dataset": {
            "root": str(dataset.root),
            "rows": len(dataset.rows),
            "parquet_sha256": dataset.parquet_sha256,
            "receipt_sha256": dataset.receipt_sha256,
            "receipt_digest": dataset.receipt_digest,
            "synthetic_target_consumed": False,
            "target_is_same_real_source": True,
        },
        "objective": {
            "name": "single_preservation_residual_mse",
            "weighted_multihead_objective": False,
            "feature_reward": False,
            "vlm_reward": False,
            "action_reward": False,
            "synthetic_target": False,
        },
        "artifacts": dict(artifacts),
        "method_source_revision": args.method_source_revision,
        "method_source_revision_kind": args.method_source_revision_kind,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "method_source_manifest_sha256": args.method_source_manifest_sha256,
        "scientific_claim_authorized": False,
        "action_editing_claim_authorized": False,
        "method_success_claimed": False,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    return receipt


def publish_cadence_checkpoint(
    *,
    runtime: Any,
    checkpoint_root: Path,
    optimizer_step: int,
    trajectory_optimizer_steps: int,
    adapter: Any,
    optimizer: Any,
    adapter_rank: int,
    initial_digest: str,
    final_digest: str,
    history_steps: Sequence[Mapping[str, Any]],
    dataset: Any,
    args: argparse.Namespace,
    rank: int,
    world_group: Any,
) -> Mapping[str, Any]:
    """Publish one immutable 0/20 checkpoint from a live exact40 trajectory."""

    import torch.distributed as dist

    publication: list[Any] = [None]
    if rank == 0:
        try:
            name = f"step-{optimizer_step:08d}"
            output = checkpoint_root / name
            stage = checkpoint_root / f".{name}.staging"
            if output.exists() or output.is_symlink() or stage.exists() or stage.is_symlink():
                fail("cadence checkpoint path is not fresh")
            os.mkdir(stage, mode=0o750)
            adapter_path = stage / "adapter.safetensors"
            optimizer_path = stage / "optimizer.pt"
            history_path = stage / "history.json"
            _atomic_adapter_safetensors(
                adapter_path, adapter, runtime, lora_rank=adapter_rank
            )
            runtime.atomic_torch_save(
                optimizer_path,
                {
                    "schema_version": RUN_RECEIPT_SCHEMA,
                    "optimizer": optimizer.state_dict(),
                    "global_step": optimizer_step,
                    "trajectory_optimizer_steps": trajectory_optimizer_steps,
                    "adapter_sha256": final_digest,
                    "adapter_rank": adapter_rank,
                    "registered_schedule_indices": list(
                        REGISTERED_SCHEDULE_INDICES
                    ),
                    "training_schedule_indices": list(
                        REGISTERED_SCHEDULE_INDICES[:optimizer_step]
                    ),
                },
            )
            history_value = {
                "schema_version": HISTORY_SCHEMA,
                "optimizer_steps": optimizer_step,
                "trajectory_optimizer_steps": trajectory_optimizer_steps,
                "continuous_trajectory": True,
                "exact40_schedule_sha256": EXPECTED_EXACT40_SCHEDULE_SHA256,
                "registered_schedule_indices": list(REGISTERED_SCHEDULE_INDICES),
                "training_schedule_indices": list(
                    REGISTERED_SCHEDULE_INDICES[:optimizer_step]
                ),
                "steps": list(history_steps),
            }
            runtime.atomic_json(history_path, history_value)
            artifacts = {
                "adapter.safetensors": runtime.file_sha256(adapter_path),
                "optimizer.pt": runtime.file_sha256(optimizer_path),
                "history.json": runtime.file_sha256(history_path),
            }
            receipt = _checkpoint_receipt(
                optimizer_step=optimizer_step,
                trajectory_optimizer_steps=trajectory_optimizer_steps,
                adapter_rank=adapter_rank,
                initial_digest=initial_digest,
                final_digest=final_digest,
                dataset=dataset,
                args=args,
                artifacts=artifacts,
            )
            runtime.atomic_json(stage / "receipt.json", receipt)
            runtime.verify_staged_run_bundle(stage, receipt)
            runtime.fsync_directory(stage)
            os.rename(stage, output)
            runtime.fsync_directory(checkpoint_root)
            runtime.verify_staged_run_bundle(output, receipt)
            publication[0] = {
                "ok": True,
                "optimizer_step": optimizer_step,
                "path": str(output),
                "adapter_parameter_sha256": final_digest,
                "adapter_file_sha256": artifacts["adapter.safetensors"],
                "receipt_sha256": runtime.file_sha256(output / "receipt.json"),
                "receipt_digest": receipt["receipt_digest"],
            }
        except Exception as error:
            publication[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(publication, src=0, group=world_group)
    result = publication[0]
    if (
        not isinstance(result, Mapping)
        or result.get("ok") is not True
        or result.get("optimizer_step") != optimizer_step
    ):
        fail(f"cannot publish cadence checkpoint: {result!r}")
    return dict(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--expected-materialization-spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-output-root")
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument(
        "--parallel-topology",
        required=True,
        choices=(PRESERVATION_TOPOLOGY_PROFILE,),
    )
    parser.add_argument(
        "--adapter-block-scope",
        choices=("early-mid-0-22",),
        default="early-mid-0-22",
    )
    parser.add_argument(
        "--adapter-rank",
        type=int,
        choices=(LOW_CAPACITY_LORA_RANK, MAIN_LORA_RANK),
        default=MAIN_LORA_RANK,
    )
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        choices=SUPPORTED_OPTIMIZER_STEPS,
        default=OPTIMIZER_STEPS,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument(
        "--method-source-revision-kind",
        choices=("git-commit", "content-closure-sha1"),
        required=True,
    )
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--method-source-manifest-sha256", required=True)
    parser.add_argument("--ack-upstream-training-use-forbidden", action="store_true")
    parser.add_argument(
        "--ack-forward-noising-is-not-inversion", action="store_true"
    )
    parser.add_argument("--num-frames", type=int, choices=(FRAME_COUNT,), default=FRAME_COUNT)
    return parser


def validate_cli(args: argparse.Namespace, *, legacy: Any, role: Any, runtime: Any) -> tuple[int, ...]:
    validate_registered_schedule()
    validate_scientific_claim_contract()
    if (
        args.mode != MODE
        or args.parallel_topology != PRESERVATION_TOPOLOGY_PROFILE
        or args.num_frames != FRAME_COUNT
    ):
        fail("preservation residual v1 requires exact81 WORLD8 DP2 x SP4")
    if args.ack_upstream_training_use_forbidden is not True:
        fail("--ack-upstream-training-use-forbidden is mandatory")
    if args.ack_forward_noising_is_not_inversion is not True:
        fail("--ack-forward-noising-is-not-inversion is mandatory")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        fail("learning rate must be finite and positive")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.0:
        fail("max grad norm must be finite and positive")
    training_coordinates(args.optimizer_steps)
    if args.optimizer_steps == OPTIMIZER_STEPS:
        if not args.checkpoint_output_root:
            fail("continuous exact40 requires --checkpoint-output-root")
        output_path = Path(args.output).expanduser()
        checkpoint_path = Path(args.checkpoint_output_root).expanduser()
        if (
            not output_path.is_absolute()
            or not checkpoint_path.is_absolute()
            or checkpoint_path.name != "checkpoints"
            or checkpoint_path.parent != output_path.parent
        ):
            fail("checkpoint root must be the output sibling named checkpoints")
    elif args.checkpoint_output_root is not None:
        fail("exact20 replicate cannot publish continuous-trajectory checkpoints")
    if args.adapter_rank not in {LOW_CAPACITY_LORA_RANK, MAIN_LORA_RANK}:
        fail("adapter rank must be the registered rank-8 main or rank-2 variant")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or not 0 <= args.seed < 2**63:
        fail("seed must lie in [0,2^63)")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        _sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_materialization_spec_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
        "method_source_manifest_sha256",
    ):
        _sha(getattr(args, name), length=64, label=name)
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        fail("checkpoint tree differs from audited Bernini 1.3B")
    if role.TRAINABLE_BLOCK_INDICES != tuple(range(23)):
        fail("Stage-B early/mid adapter block pin differs")
    return role.TRAINABLE_BLOCK_INDICES


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Torch-dependent imports are intentionally lazy so the fixed scientific
    # contract can be audited in a model-free environment.
    import preservation_source_role_v1 as role
    import source_self_runtime as runtime
    import train_lora as legacy
    import train_source_self_role_repaint as base

    preservation_topology = register_preservation_topology(runtime)
    args = build_parser().parse_args(argv)
    block_indices = validate_cli(args, legacy=legacy, role=role, runtime=runtime)
    dataset = base._load_dataset(
        args.dataset_root, args.expected_materialization_spec_sha256
    )
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except (ImportError, AttributeError):
        pass
    gc.collect()
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(
            args.checkpoint
        )
    except legacy.TrainingContractError as error:
        raise SourceNoisedCarrierTrainingError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        fail("pinned Bernini attention-head count differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import (
        UniPCMultistepScheduler,
        __version__ as diffusers_version,
    )
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state

    topology = preservation_topology
    contract = runtime.distributed_contract()
    if (
        contract.world_size != topology.world_size
        or contract.local_world_size != topology.world_size
        or contract.rank != contract.local_rank
    ):
        fail("preservation residual physical topology must be WORLD8 DP2 x SP4")
    device = runtime.initialise_distributed(contract)
    parallel = runtime.validate_parallel_state(
        contract, init_parallel_state(ulysses_size=SP_SIZE)
    )
    output, stage = runtime.prepare_output_transaction(
        args.output, contract.rank, parallel.world_group
    )
    checkpoint_root: Optional[Path] = None
    if args.checkpoint_output_root is not None:
        checkpoint_root = Path(args.checkpoint_output_root)
        checkpoint_reservation: list[Any] = [None]
        if contract.rank == 0:
            try:
                if checkpoint_root.exists() or checkpoint_root.is_symlink():
                    fail("checkpoint root must be fresh")
                os.mkdir(checkpoint_root, mode=0o750)
                runtime.fsync_directory(checkpoint_root.parent)
                checkpoint_reservation[0] = {
                    "ok": True,
                    "path": str(checkpoint_root),
                }
            except Exception as error:
                checkpoint_reservation[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(
            checkpoint_reservation, src=0, group=parallel.world_group
        )
        if checkpoint_reservation[0] != {
            "ok": True,
            "path": str(checkpoint_root),
        }:
            fail(f"cannot reserve checkpoint root: {checkpoint_reservation[0]!r}")
    local_logical_arms = (contract.arm_index,)
    if local_logical_arms not in ((0,), (1,)):
        fail("WORLD8 DP replica does not own exactly one logical source row")
    # synchronize_gradients averages the two DP replicas after SP reduction.
    loss_scale = 1.0
    physical_placement = {
        "nodes": 1,
        "local_world_size": contract.local_world_size,
        "ranks_per_node": contract.local_world_size,
        "sp4_crosses_nodes": False,
        "single_node_world8": True,
        "logical_arm_per_dp_replica": True,
    }
    runtime.digest_consensus(
        object_sha256(physical_placement),
        group=parallel.world_group,
        expected_count=topology.world_size,
        label="source-carrier physical placement",
    )

    legacy.seed_same_sample(args.seed)
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    # The holder allocates 64 GiB host RAM.  Eight simultaneous checkpoint
    # deserializations exceed that cgroup even though the resident GPU model
    # fits comfortably.  Admit one rank at a time, move it to its own GPU, and
    # only then let the next rank deserialize.  This changes neither weights
    # nor distributed math.
    renderer = None
    for loading_rank in range(topology.world_size):
        if contract.rank == loading_rank:
            renderer = BerniniRendererModel(config)
            renderer.requires_grad_(False)
            renderer.eval()
            renderer.t5_text_encoder.eval()
            renderer.to(device)
        dist.barrier(group=parallel.world_group)
    if renderer is None:
        fail("rank-serialized renderer load did not materialize every rank")
    transformer = renderer.diff_dec.transformer
    if transformer is None or renderer.diff_dec.transformer_2 is not None:
        fail("Stage-B requires only Bernini transformer_1")
    renderer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": role.checkpoint_route_context_fn,
        }
    )
    if not bool(getattr(transformer, "gradient_checkpointing", False)):
        fail("Stage-B requires non-reentrant checkpointing")
    adapter = role.install_source_self_adapter(
        transformer,
        rank=args.adapter_rank,
        alpha=float(args.adapter_rank),
        block_indices=block_indices,
    )
    trainable = adapter.trainable_named_parameters()
    if not adapter.base_parameters_frozen():
        fail("Bernini base changed trainability after adapter install")
    initial_digest = runtime.synchronize_initial_parameters(
        trainable,
        parallel.world_group,
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    checkpoint_records: list[Mapping[str, Any]] = []
    if checkpoint_root is not None:
        checkpoint_records.append(
            publish_cadence_checkpoint(
                runtime=runtime,
                checkpoint_root=checkpoint_root,
                optimizer_step=0,
                trajectory_optimizer_steps=args.optimizer_steps,
                adapter=adapter,
                optimizer=optimizer,
                adapter_rank=args.adapter_rank,
                initial_digest=initial_digest,
                final_digest=initial_digest,
                history_steps=(),
                dataset=dataset,
                args=args,
                rank=contract.rank,
                world_group=parallel.world_group,
            )
        )

    inference_scheduler = UniPCMultistepScheduler.from_pretrained(
        str(checkpoint),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=exact40.FLOW_SHIFT,
    )
    runtime_schedule_audit = exact40.audit_runtime_unipc_schedule(
        inference_scheduler
    )
    if runtime_schedule_audit.get("schedule_sha256") != EXPECTED_EXACT40_SCHEDULE_SHA256:
        fail("runtime UniPC schedule SHA differs")
    del inference_scheduler

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    text = runtime.tokenize_generic_instruction(
        tokenizer, EXACT_NOOP_INSTRUCTION, device
    )
    text_digest = object_sha256(
        {
            name: runtime.tensor_sha256(value)
            for name, value in sorted(text.items())
        }
    )
    runtime.digest_consensus(
        text_digest,
        group=parallel.world_group,
        expected_count=topology.world_size,
        label="source-carrier generic text",
    )
    with torch.inference_mode():
        text_lens, text_embs = renderer.get_t5_text_embeddings(
            text["input_ids"], text["attention_mask"], text["t5_input_lens"]
        )
    if getattr(text_embs, "requires_grad", False):
        fail("frozen T5 embeddings unexpectedly require gradients")
    renderer.t5_text_encoder = None
    del tokenizer, text
    gc.collect()
    torch.cuda.empty_cache()
    if renderer.t5_text_encoder is not None:
        fail("frozen T5 encoder was not released")

    vae_mean, vae_std, _ = legacy._vae_statistics(checkpoint)
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    previous_digest = initial_digest
    history_steps: list[dict[str, Any]] = []
    positive_gradient_steps = 0

    for coordinate in training_coordinates(args.optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        prepared_arms: list[PreparedLogicalArm] = []
        completed_control_arms: list[int] = []
        # Both arms' controls precede every graph-bearing forward at this sigma.
        for logical_arm in local_logical_arms:
            local_index = logical_arm % len(dataset.rows)
            wrong_index = (local_index + 1) % len(dataset.rows)
            local_row = dataset.rows[local_index]
            wrong_row = dataset.rows[wrong_index]
            if local_row.iid == wrong_row.iid or local_row.clean_shape != wrong_row.clean_shape:
                fail("wrong-reference row must be distinct in the same bucket")
            clean = base._posterior_mode(
                local_row.clean_blob,
                vae_mean,
                vae_std,
                phases=LATENT_PHASES,
                label=f"{local_row.iid} clean",
            )
            wrong_clean = base._posterior_mode(
                wrong_row.clean_blob,
                vae_mean,
                vae_std,
                phases=LATENT_PHASES,
                label=f"{wrong_row.iid} wrong-clean",
            )
            if clean.data_ptr() == wrong_clean.data_ptr():
                fail("correct and wrong clean source must be independent tensors")
            correct_ref_map = {
                index: base._posterior_mode(
                    local_row.refs[index],
                    vae_mean,
                    vae_std,
                    phases=REFERENCE_PHASES,
                    label=f"{local_row.iid} clean-ref{index}",
                )
                for index in role.REFERENCE_RGB_INDICES
            }
            wrong_ref_map = {
                index: base._posterior_mode(
                    wrong_row.refs[index],
                    vae_mean,
                    vae_std,
                    phases=REFERENCE_PHASES,
                    label=f"{wrong_row.iid} wrong-clean-ref{index}",
                )
                for index in role.REFERENCE_RGB_INDICES
            }
            correct_refs = [
                correct_ref_map[index] for index in local_row.reference_order
            ]
            wrong_refs = [
                wrong_ref_map[index] for index in local_row.reference_order
            ]
            if any(
                left.data_ptr() == right.data_ptr()
                for left in correct_refs
                for right in wrong_refs
            ):
                fail("correct and wrong references alias storage")
            seed = _noise_seed(
                args.seed, coordinate.optimizer_step_zero_based, logical_arm
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            epsilon = torch.randn(
                tuple(clean.shape), generator=generator, dtype=torch.float32
            ).contiguous()
            prepared = prepare_carrier_step(
                base=base,
                role=role,
                runtime=runtime,
                clean=clean,
                wrong_clean=wrong_clean,
                correct_refs=correct_refs,
                wrong_refs=wrong_refs,
                epsilon=epsilon,
                coordinate=coordinate,
                rope=rope,
                device=device,
            )
            control_predictions: dict[str, Any] = {}
            for name, condition in prepared.controls.items():
                invocation = role.RouteInvocation(
                    condition.layout,
                    sequence_parallel_rank=contract.sp_rank,
                    sequence_parallel_size=SP_SIZE,
                )
                with adapter.route(invocation):
                    with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        control_predictions[name] = _prediction(
                            role=role,
                            renderer=renderer,
                            transformer=transformer,
                            condition=condition,
                            text_lens=text_lens,
                            text_embs=text_embs,
                        ).detach()
                if any(parameter.grad is not None for _, parameter in trainable):
                    fail("no-gradient control touched adapter gradients")
            base_invocation = role.RouteInvocation(
                prepared.main.layout,
                sequence_parallel_rank=contract.sp_rank,
                sequence_parallel_size=SP_SIZE,
                enabled=False,
            )
            with adapter.route(base_invocation):
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    frozen_base_prediction = _prediction(
                        role=role,
                        renderer=renderer,
                        transformer=transformer,
                        condition=prepared.main,
                        text_lens=text_lens,
                        text_embs=text_embs,
                    ).detach()
            baseline = control_predictions["source_correct_refs"]
            control_metrics = {
                name: _relative_l2(value, baseline)
                for name, value in control_predictions.items()
                if name != "source_correct_refs"
            }
            control_metrics = {
                **control_metrics,
                "adapted_vs_frozen_base_relative_l2": _relative_l2(
                    baseline, frozen_base_prediction
                ),
                "correct_source_flow_mse": float(
                    torch.nn.functional.mse_loss(
                        baseline.float(), prepared.target_velocity.float()
                    ).item()
                ),
                "wrong_source_flow_mse": float(
                    torch.nn.functional.mse_loss(
                        control_predictions["wrong_source_wrong_refs"].float(),
                        prepared.target_velocity.float(),
                    ).item()
                ),
                "source_binding_gap_wrong_minus_correct": float(
                    (
                        torch.nn.functional.mse_loss(
                            control_predictions["wrong_source_wrong_refs"].float(),
                            prepared.target_velocity.float(),
                        )
                        - torch.nn.functional.mse_loss(
                            baseline.float(), prepared.target_velocity.float()
                        )
                    ).item()
                ),
            }
            residual_teacher = (
                prepared.target_velocity.float()
                - frozen_base_prediction.float()
            ).detach().contiguous()
            prepared_arms.append(
                PreparedLogicalArm(
                    logical_arm=logical_arm,
                    row_index=local_index,
                    iid=local_row.iid,
                    wrong_ref_iid=wrong_row.iid,
                    noise_seed=seed,
                    main=prepared.main,
                    target_velocity=prepared.target_velocity,
                    frozen_base_prediction=frozen_base_prediction,
                    residual_teacher=residual_teacher,
                    epsilon_sha256=prepared.epsilon_sha256,
                    tensor_identities=prepared.tensor_identities,
                    shared_noise_binding=prepared.shared_noise_binding,
                    control_metrics=control_metrics,
                    control_names=tuple(sorted(prepared.controls)),
                    absent_reference_layout={
                        **prepared.controls[
                            "source_refs_absent"
                        ].layout.receipt(),
                        "reference_source_ids": [],
                    },
                )
            )
            completed_control_arms.append(logical_arm)
            del (
                control_predictions,
                baseline,
                prepared,
                condition,
                invocation,
                base_invocation,
                residual_teacher,
                frozen_base_prediction,
            )

        if tuple(completed_control_arms) != local_logical_arms:
            fail("DP-owned logical-arm controls are incomplete")
        if any(parameter.grad is not None for _, parameter in trainable):
            fail("control phase did not leave a clean gradient state")
        # Both DP arms must finish all no-grad controls before either enters
        # its graph-bearing forward/backward.
        dist.barrier(group=parallel.world_group)

        losses: dict[int, float] = {}
        completed_backward_arms: list[int] = []
        for logical in prepared_arms:
            invocation = role.RouteInvocation(
                logical.main.layout,
                sequence_parallel_rank=contract.sp_rank,
                sequence_parallel_size=SP_SIZE,
            )
            with adapter.route(invocation):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = _prediction(
                        role=role,
                        renderer=renderer,
                        transformer=transformer,
                        condition=logical.main,
                        text_lens=text_lens,
                        text_embs=text_embs,
                    )
                    predicted_residual = (
                        prediction.float()
                        - logical.frozen_base_prediction.float()
                    )
                    loss = torch.nn.functional.mse_loss(
                        predicted_residual,
                        logical.residual_teacher.float(),
                        reduction="mean",
                    )
                if not runtime.world_all_true(
                    bool(torch.isfinite(loss.detach()).item()),
                    group=parallel.world_group,
                ):
                    fail("non-finite source-carrier loss blocked update")
                (loss * loss_scale).backward()
                losses[logical.logical_arm] = float(loss.detach().item())
            completed_backward_arms.append(logical.logical_arm)
        del prediction, predicted_residual, loss, invocation

        authorize_optimizer_step(
            completed_control_arms=tuple(completed_control_arms),
            completed_backward_arms=tuple(completed_backward_arms),
        )
        preclip_norm = runtime.synchronize_gradients(trainable, parallel)
        if not math.isfinite(preclip_norm) or preclip_norm <= 0.0:
            fail("Stage-B requires a finite positive synchronized gradient norm")
        positive_gradient_steps += 1
        clipped = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], args.max_grad_norm
        )
        if not math.isfinite(float(clipped)):
            fail("gradient clipping produced a non-finite norm")
        optimizer.step()
        final_digest = runtime.parameter_consensus(
            trainable,
            parallel.world_group,
            f"source-carrier step {coordinate.optimizer_step_zero_based + 1}",
        )
        if final_digest == previous_digest:
            fail("optimizer step did not change adapter parameters")

        local_records: list[dict[str, Any]] = []
        for logical in prepared_arms:
            local_record = {
                "schema_version": HISTORY_SCHEMA,
                **coordinate.receipt(),
                "logical_arm": logical.logical_arm,
                "physical_dp_rank": contract.arm_index,
                "sp_rank": contract.sp_rank,
                "row_index": logical.row_index,
                "iid": logical.iid,
                "wrong_ref_iid": logical.wrong_ref_iid,
                "noise_seed": logical.noise_seed,
                "epsilon_sha256": logical.epsilon_sha256,
                "tensor_identities": dict(logical.tensor_identities),
                "shared_noise_binding": dict(logical.shared_noise_binding),
                "clean_source_references_routed": True,
                "preservation_residual_mse_unscaled": losses[logical.logical_arm],
                "objective_equation": (
                    "mse((v_adapted-v_frozen_noop),"
                    "stopgrad(v_target-v_frozen_noop))"
                ),
                "logical_loss_scale": loss_scale,
                "preclip_gradient_norm_logical_two_arm_mean": preclip_norm,
                "control_names": list(logical.control_names),
                "source_binding_and_route_diagnostics": dict(
                    logical.control_metrics
                ),
                "all_controls_no_gradient": True,
                "both_logical_controls_preceded_any_backward": True,
                "optimizer_update_authorized_for_this_sigma": True,
                "parameter_sha256_before_step": previous_digest,
                "parameter_sha256_after_step": final_digest,
            }
            projection = {
                key: value for key, value in local_record.items() if key != "sp_rank"
            }
            runtime.digest_consensus(
                object_sha256(projection),
                group=parallel.sp_group,
                expected_count=SP_SIZE,
                label=(
                    f"source-carrier step {coordinate.optimizer_step_zero_based + 1} "
                    f"arm {logical.logical_arm}"
                ),
            )
            local_records.append(local_record)
        del logical, local_record, projection
        gathered: list[Any] = [None] * topology.world_size
        dist.all_gather_object(gathered, local_records, group=parallel.world_group)
        step_records = []
        for sp_members in topology.sp_group_ranks:
            leader_records = gathered[sp_members[0]]
            if not isinstance(leader_records, list) or len(leader_records) != 1:
                fail("SP leader history is not one DP-owned record")
            step_records.extend(leader_records)
        step_records.sort(key=lambda item: item["logical_arm"])
        if (
            len(step_records) != LOGICAL_ARM_COUNT
            or tuple(item["logical_arm"] for item in step_records) != (0, 1)
            or len({item["iid"] for item in step_records}) != LOGICAL_ARM_COUNT
            or len({item["epsilon_sha256"] for item in step_records})
            != LOGICAL_ARM_COUNT
            or any(
                item["shared_noise_binding"].get(
                    "same_epsilon_object_reused_during_target_and_source_construction"
                )
                is not True
                for item in step_records
            )
        ):
            fail("two-arm same-noise history is incomplete")
        history_steps.append(
            {
                "schema_version": HISTORY_SCHEMA,
                **coordinate.receipt(),
                "logical_records": step_records,
                "optimizer_step_executed": True,
                "parameter_sha256_before_step": previous_digest,
                "parameter_sha256_after_step": final_digest,
            }
        )
        if (
            checkpoint_root is not None
            and coordinate.optimizer_step_zero_based + 1 == CHECKPOINT_INTERVAL
        ):
            checkpoint_records.append(
                publish_cadence_checkpoint(
                    runtime=runtime,
                    checkpoint_root=checkpoint_root,
                    optimizer_step=coordinate.optimizer_step_zero_based + 1,
                    trajectory_optimizer_steps=args.optimizer_steps,
                    adapter=adapter,
                    optimizer=optimizer,
                    adapter_rank=args.adapter_rank,
                    initial_digest=initial_digest,
                    final_digest=final_digest,
                    history_steps=history_steps,
                    dataset=dataset,
                    args=args,
                    rank=contract.rank,
                    world_group=parallel.world_group,
                )
            )
        previous_digest = final_digest
        del (
            prepared_arms,
            clean,
            wrong_clean,
            correct_ref_map,
            wrong_ref_map,
            correct_refs,
            wrong_refs,
            epsilon,
        )
        gc.collect()
        torch.cuda.empty_cache()

    final_digest = previous_digest
    if (
        len(history_steps) != args.optimizer_steps
        or positive_gradient_steps != args.optimizer_steps
        or tuple(item["schedule_index"] for item in history_steps)
        != REGISTERED_SCHEDULE_INDICES[: args.optimizer_steps]
        or any(item["optimizer_step_executed"] is not True for item in history_steps)
        or len(
            {
                logical["epsilon_sha256"]
                for item in history_steps
                for logical in item["logical_records"]
            }
        )
        != args.optimizer_steps * LOGICAL_ARM_COUNT
    ):
        fail("preservation residual optimizer history is incomplete")
    if checkpoint_root is not None and [
        item.get("optimizer_step") for item in checkpoint_records
    ] != [0, CHECKPOINT_INTERVAL]:
        fail("continuous exact40 checkpoint cadence is incomplete")

    dist.barrier(group=parallel.world_group)
    receipt: Optional[dict[str, Any]] = None
    rank_zero_publication_error: Optional[str] = None
    if contract.rank == 0:
        try:
            adapter_path = stage / "adapter.safetensors"
            optimizer_path = stage / "optimizer.pt"
            history_path = stage / "history.json"
            _atomic_adapter_safetensors(
                adapter_path, adapter, runtime, lora_rank=args.adapter_rank
            )
            runtime.atomic_torch_save(
                optimizer_path,
                {
                    "schema_version": RUN_RECEIPT_SCHEMA,
                    "optimizer": optimizer.state_dict(),
                    "global_step": args.optimizer_steps,
                    "adapter_sha256": final_digest,
                    "adapter_rank": args.adapter_rank,
                    "registered_schedule_indices": list(
                        REGISTERED_SCHEDULE_INDICES
                    ),
                    "training_schedule_indices": list(
                        REGISTERED_SCHEDULE_INDICES[: args.optimizer_steps]
                    ),
                },
            )
            history_value = {
                "schema_version": HISTORY_SCHEMA,
                "optimizer_steps": args.optimizer_steps,
                "trajectory_optimizer_steps": args.optimizer_steps,
                "continuous_trajectory": args.optimizer_steps == OPTIMIZER_STEPS,
                "exact40_schedule_sha256": EXPECTED_EXACT40_SCHEDULE_SHA256,
                "registered_schedule_indices": list(REGISTERED_SCHEDULE_INDICES),
                "training_schedule_indices": list(
                    REGISTERED_SCHEDULE_INDICES[: args.optimizer_steps]
                ),
                "steps": history_steps,
            }
            runtime.atomic_json(history_path, history_value)
            receipt = {
                "schema_version": RUN_RECEIPT_SCHEMA,
                "method": METHOD_NAME,
                "complete": True,
                "mode": MODE,
                "fixed_plan": fixed_plan_receipt(args.optimizer_steps),
                "optimizer_steps": args.optimizer_steps,
                "trajectory_optimizer_steps": args.optimizer_steps,
                "continuous_trajectory": args.optimizer_steps == OPTIMIZER_STEPS,
                "positive_gradient_steps": positive_gradient_steps,
                "optimizer_step_per_training_sigma": True,
                "optimizer_step_per_registered_sigma": (
                    args.optimizer_steps == OPTIMIZER_STEPS
                ),
                "all_registered_strata_optimizer_authorized": (
                    args.optimizer_steps == OPTIMIZER_STEPS
                ),
                "formal_exact40_complete": args.optimizer_steps == OPTIMIZER_STEPS,
                "checkpoint_interval": CHECKPOINT_INTERVAL,
                "checkpoint_steps": (
                    list(LOADABLE_CHECKPOINT_STEPS)
                    if args.optimizer_steps == OPTIMIZER_STEPS
                    else [args.optimizer_steps]
                ),
                "checkpoint_bundles": [dict(item) for item in checkpoint_records],
                "late_or_low_sigma_zero_update_gate_present": False,
                "frame_count": FRAME_COUNT,
                "latent_phases": LATENT_PHASES,
                "exact40_schedule_sha256": EXPECTED_EXACT40_SCHEDULE_SHA256,
                "runtime_unipc_schedule_audit": runtime_schedule_audit,
                "registered_schedule_indices": list(REGISTERED_SCHEDULE_INDICES),
                "training_schedule_indices": list(
                    REGISTERED_SCHEDULE_INDICES[: args.optimizer_steps]
                ),
                "registered_coordinates": [item.receipt() for item in validate_registered_schedule()],
                "training_coordinates": [
                    item.receipt() for item in training_coordinates(args.optimizer_steps)
                ],
                "forward_noising": {
                    "target_equation": "x_target_sigma=(1-sigma)*z_clean+sigma*epsilon",
                    "source_condition_equation": "x_source_sigma=(1-sigma)*z_clean+sigma*epsilon",
                    "target_velocity_equation": "v*=epsilon-z_clean",
                    "same_epsilon_target_and_source_required": True,
                    "same_epsilon_target_and_source_verified_every_logical_record": True,
                    "same_sigma_target_and_source_verified_every_logical_record": True,
                    "different_epsilon_across_logical_step_samples": True,
                    "clean_source_references_routed_every_logical_record": True,
                    "forward_noising_only": True,
                    "inversion_claimed": False,
                    "reverse_ode_executed": False,
                    "solver_state_replayed": False,
                    "exact_roundtrip_claimed": False,
                },
                "dataset": {
                    "root": str(dataset.root),
                    "parquet_sha256": dataset.parquet_sha256,
                    "receipt_sha256": dataset.receipt_sha256,
                    "receipt_digest": dataset.receipt_digest,
                    "rows": len(dataset.rows),
                    "positive_target_role": (
                        "independent_pinned_vae_encode_of_same_raw_clean_source_rgb"
                    ),
                    "paired_dataset_accessed": False,
                    "prior_posterior_accessed": False,
                    "edited_target_accessed": False,
                    "action_supervision_present": False,
                    "synthetic_target_consumed": False,
                    "target_is_same_real_source": True,
                },
                "visual_pack": {
                    "source_ids": {
                        "forward_noised_source_condition": 1,
                        "clean_reference_in_preregistered_order": [2, 3, 4],
                        "forward_noised_target": 0,
                    },
                    "reference_rgb_indices": list(role.REFERENCE_RGB_INDICES),
                    "reference_order_per_iid": True,
                    "reference_from_video_posterior_slice": False,
                    "clean_target_is_not_model_input": True,
                },
                "adapter": dict(adapter.receipt()),
                "adapter_rank": args.adapter_rank,
                "experiment_arm": (
                    "main_rank8" if args.adapter_rank == MAIN_LORA_RANK
                    else "low_capacity_rank2"
                ),
                "adapter_file_schema": ADAPTER_FILE_SCHEMA,
                "matched_same-noise_carrier_inference_runtime_required": True,
                "existing_sigma1_conditional_base_loader_compatible": False,
                "initial_adapter_sha256": initial_digest,
                "final_adapter_sha256": final_digest,
                "adapter_changed_each_optimizer_step": True,
                "base_frozen": True,
                "key_value_frozen": True,
                "cross_attention_frozen": True,
                "late_blocks_23_29_frozen": True,
                "vae_frozen_and_absent_from_training_process": True,
                "t5_frozen": True,
                "t5_released_after_one_frozen_embedding": True,
                "distributed": {
                    "profile": topology.profile,
                    "world_size": topology.world_size,
                    "physical_data_parallel_size": topology.dp_size,
                    "ulysses_sequence_parallel_size": topology.sp_size,
                    "sp_groups": [list(item) for item in topology.sp_group_ranks],
                    "dp_groups": [list(item) for item in topology.dp_group_ranks],
                    "placement": physical_placement,
                    "logical_arm_count": LOGICAL_ARM_COUNT,
                    "logical_arms_per_physical_dp_rank": [[0], [1]],
                    "serial_logical_arm_accumulation": False,
                    "parallel_logical_arms_across_dp_replicas": True,
                    "rank_serialized_cpu_checkpoint_load": True,
                    "rank_serialized_load_changes_weights_or_objective": False,
                    "both_logical_controls_precede_any_backward_per_sigma": True,
                    "logical_loss_scale_per_backward": loss_scale,
                    "logical_objective": "DP-mean(logical_arm_0,logical_arm_1)",
                    "dp_all_reduce_executed_for_dp2": True,
                    "parameter_consensus_across_physical_world": True,
                    "gradient_checkpointing_non_reentrant": True,
                    "checkpoint_route_context_fn": (
                        "source_self_role_repaint.checkpoint_route_context_fn"
                    ),
                    "checkpoint_recomputation_route_context_replayed": True,
                },
                "controls": {
                    "optimizer_supervision": "none",
                    "cells": history_steps[0]["logical_records"][0][
                        "control_names"
                    ],
                    "same_sigma_noisy_target_within_logical_sample": True,
                    "same_epsilon_target_and_source_condition": True,
                    "wrong_source_wrong_references": True,
                    "references_truly_absent_control": True,
                    "metrics_in_history": True,
                    "no_fixed_binding_margin": True,
                },
                "objective": {
                    "name": "single_preservation_residual_mse",
                    "frozen_base_prompt": EXACT_NOOP_INSTRUCTION,
                    "residual_teacher": "stopgrad(v_target-v_frozen_noop)",
                    "prediction": "v_adapted-v_frozen_noop",
                    "weighted_multihead_objective": False,
                    "feature_reward": False,
                    "vlm_reward": False,
                    "action_reward": False,
                    "synthetic_target": False,
                },
                "runtime": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "transformers": transformers_version,
                    "diffusers": diffusers_version,
                },
                "model": {
                    "bernini_commit": bernini_revision,
                    "veomni_commit": veomni_revision,
                    "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                },
                "artifacts": {
                    "adapter.safetensors": runtime.file_sha256(adapter_path),
                    "optimizer.pt": runtime.file_sha256(optimizer_path),
                    "history.json": runtime.file_sha256(history_path),
                },
                "upstream_training_use_forbidden_acknowledged": True,
                "forward_noising_not_inversion_acknowledged": True,
                "pretext_training_only": True,
                "frozen_base_action_prior_not_retrained": True,
                "semantic_motion_preservation_claimed": False,
                "natural_semantic_action_learned": False,
                "action_editing_claim_authorized": False,
                "video_quality_claim_authorized": False,
                "scientific_claim_authorized": False,
                "method_success_claimed": False,
                "long_training_scientific_gate_passed": False,
                "long_training_automatically_submitted": False,
                "method_source_revision": args.method_source_revision,
                "method_source_revision_kind": args.method_source_revision_kind,
                "method_source_archive_sha256": args.method_source_archive_sha256,
                "method_source_manifest_sha256": args.method_source_manifest_sha256,
            }
            receipt["receipt_digest"] = object_sha256(receipt)
            runtime.atomic_json(stage / "receipt.json", receipt)
            runtime.verify_staged_run_bundle(stage, receipt)
            runtime.fsync_directory(stage)
        except Exception as error:
            rank_zero_publication_error = f"{type(error).__name__}: {error}"
    publish_preservation_output(
        runtime,
        output,
        stage,
        receipt,
        contract.rank,
        parallel.world_group,
        rank_zero_error=rank_zero_publication_error,
    )
    if contract.rank == 0:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "adapter_sha256": final_digest,
                    "optimizer_steps": args.optimizer_steps,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    adapter.restore()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

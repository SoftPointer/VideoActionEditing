#!/usr/bin/env python3
"""Invert an exact K=2 Bernini EPMC prototype on 81-frame supports.

This is a fail-closed engineering canary, not Bernini fine-tuning and not a
video-quality claim.  The only optimized state is a replicated 36D motion
code (20 non-boundary phase logits plus 16 block logits tied over 12 heads).
The Bernini base, UMT5, and cloned CPMR Q/K/V/O branches remain frozen.

Each carrier is inference-available: frozen Bernini generates an action and a
semantic-no-op proposal from the *same source* and *same proposal seed*, then
``fewshot_proposal_motion_carrier`` builds a motion carrier on the true 30x31
patch grid.  No target, mask, flow, pose, track, trajectory, or edited frame
enters that path.  A paired target is used only by the training loss.

The loss is a single-noisy-state surrogate for Bernini's released momentum-zero
APG program.  Negative, raw no-op, and EPMC-conditioned no-op velocities share
one state; the APG clean-field difference is compared to the Q0 executable
target motion.  Full-target flow matching is structurally weighted zero.

The episode loader hashes and ffprobes all three selected videos and hashes all
three parquets up front.  Before prototype freeze, however, only support
parquets are deserialized.  Held-out target latent decoding is possible only
with ``--posthoc-heldout-eval`` after the support prototype is immutable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import fewshot_episode_io as episode_io  # noqa: E402
import fewshot_episode_parallel as episode_parallel  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import motion_residual as motion  # noqa: E402
import train_lora as legacy  # noqa: E402


METHOD_NAME = "k2-privileged-epmc-code-inversion"
RUN_RECEIPT_SCHEMA = "bernini-epmc-k2-code-inversion-receipt-v2"
GO_RECEIPT_SCHEMA = "bernini-epmc-k2-representability-gate-v1"
PROTOTYPE_RECEIPT_SCHEMA = (
    "bernini-epmc-v12-tied-prototype-training-receipt-v1"
)
PROTOTYPE_STATE_SCHEMA = "bernini-epmc-v12-tied-prototype-state-v1"
DIAGNOSTIC_STATE_SCHEMA = "bernini-epmc-code-diagnostics-v1"

NUM_FRAMES = 81
LATENT_PHASES = 21
K_SHOT = 2
GLOBAL_TOKENS = 39_060
TARGET_TOKENS = 19_530
OUTPUT_PATCH_WIDTH = 64
RAW_TEXT_WIDTH = 4_096
TRAINABLE_CODE_DIMENSION = 36
PATCH_GRID_YX = (30, 31)
LATENT_SHAPE = (1, 16, 21, 60, 62)
EPMC_WORLD_SIZE = episode_parallel.WORLD_SIZE
ULYSSES_SIZE = episode_parallel.ULYSSES_SIZE
DATA_PARALLEL_SIZE = episode_parallel.DATA_PARALLEL_SIZE
SP_GROUP_RANKS = tuple(
    group.ranks for group in episode_parallel.DEFAULT_TOPOLOGY.ulysses_groups
)
DP_GROUP_RANKS = tuple(
    group.ranks for group in episode_parallel.DEFAULT_TOPOLOGY.data_parallel_groups
)
WITHIN_SUPPORT_GRADIENT_SYNC = (
    "ulysses_group_all_reduce_sum_then_divide_by_4"
)
K2_EXCHANGE = "data_parallel_lane_all_gather_then_world_digest_consensus"
FULL_TARGET_FM_WEIGHT = 0.0
CHARBONNIER_SCALE = 0.1
BRIDGE_FRACTION = 0.0
PROPOSAL_STEPS = 40
DEFAULT_PROPOSAL_SEED = 2027
DEFAULT_STEPS_PER_SUPPORT = 50
DEFAULT_LEARNING_RATE = 5.0e-2
DEFAULT_FIXED_SIGMA_INDEX = 20
DEFAULT_HELD_SIGMA_INDEX = 32
DEFAULT_SEED = 20260808
EXACT_NOOP_INSTRUCTION = motion.DEFAULT_NOOP_INSTRUCTION

FORBIDDEN_INFERENCE_CONDITIONS = (
    "target",
    "support",
    "mask",
    "flow",
    "pose",
    "track",
    "trajectory",
    "edited_first_frame",
    "reference_image",
    "reference_video",
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class FewShotCodeTrainingError(RuntimeError):
    """Raised before an ambiguous optimizer step or artifact publication."""


@dataclass(frozen=True)
class EPMCDistributedContract:
    world_size: int
    global_rank: int
    local_rank: int
    local_world_size: int
    ulysses_size: int
    data_parallel_size: int

    @property
    def support_index(self) -> int:
        return self.global_rank // self.ulysses_size

    @property
    def ulysses_rank(self) -> int:
        return self.global_rank % self.ulysses_size


@dataclass(frozen=True)
class EPMCParallelContext:
    contract: EPMCDistributedContract
    world_group: Any
    ulysses_group: Any
    dp_group: Any

    @property
    def support_index(self) -> int:
        return self.contract.support_index

    @property
    def ulysses_rank(self) -> int:
        return self.contract.ulysses_rank


def _environment_integer(
    environment: Mapping[str, str], name: str
) -> int:
    raw = environment.get(name)
    if raw is None or not raw.isdecimal():
        raise FewShotCodeTrainingError(f"{name} must be a decimal integer")
    return int(raw)


def epmc_distributed_contract(
    environment: Mapping[str, str] = os.environ,
) -> EPMCDistributedContract:
    """Require one eight-GPU node arranged as DP=2 by Ulysses-SP=4."""

    world_size = _environment_integer(environment, "WORLD_SIZE")
    global_rank = _environment_integer(environment, "RANK")
    local_rank = _environment_integer(environment, "LOCAL_RANK")
    local_world_size = _environment_integer(environment, "LOCAL_WORLD_SIZE")
    if world_size != EPMC_WORLD_SIZE or local_world_size != EPMC_WORLD_SIZE:
        raise FewShotCodeTrainingError(
            "EPMC support parallelism requires exact single-node WORLD_SIZE=8"
        )
    if not 0 <= global_rank < world_size or not 0 <= local_rank < local_world_size:
        raise FewShotCodeTrainingError(
            "torchrun rank environment is outside the exact eight-rank topology"
        )
    if global_rank != local_rank:
        raise FewShotCodeTrainingError(
            "single-node EPMC requires RANK to equal LOCAL_RANK"
        )
    return EPMCDistributedContract(
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        local_world_size=local_world_size,
        ulysses_size=ULYSSES_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )


def initialise_epmc_distributed(
    contract: EPMCDistributedContract,
) -> tuple[Any, str]:
    """Initialize the global RCCL world before Bernini creates SP/DP groups."""

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise FewShotCodeTrainingError("eight-rank EPMC requires ROCm-visible GPUs")
    if torch.cuda.device_count() != contract.local_world_size:
        raise FewShotCodeTrainingError(
            "visible accelerator count differs from LOCAL_WORLD_SIZE=8"
        )
    torch.cuda.set_device(contract.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    if (
        dist.get_world_size() != contract.world_size
        or dist.get_rank() != contract.global_rank
    ):
        raise FewShotCodeTrainingError(
            "initialized RCCL world differs from the torchrun environment"
        )
    return torch.device("cuda", contract.local_rank), "nccl/rccl"


def _group_members(
    group: Any, *, expected_size: int, dist_module: Optional[Any] = None
) -> tuple[int, ...]:
    if dist_module is None:
        import torch.distributed as dist_module

    if group is None or dist_module.get_world_size(group) != expected_size:
        raise FewShotCodeTrainingError("distributed subgroup size differs")
    members: list[Any] = [None] * expected_size
    dist_module.all_gather_object(
        members, int(dist_module.get_rank()), group=group
    )
    if any(type(item) is not int for item in members):
        raise FewShotCodeTrainingError("distributed subgroup member audit failed")
    return tuple(members)


def validate_epmc_parallel_state(
    contract: EPMCDistributedContract,
    parallel_state: Any,
    *,
    dist_module: Optional[Any] = None,
) -> EPMCParallelContext:
    """Bind Bernini's two SP4 groups and four cross-support DP2 lanes."""

    if dist_module is None:
        import torch.distributed as dist_module

    expected_scalars = (
        getattr(parallel_state, "world_size", None) == EPMC_WORLD_SIZE,
        getattr(parallel_state, "ulysses_size", None) == ULYSSES_SIZE,
        getattr(parallel_state, "dp_size", None) == DATA_PARALLEL_SIZE,
        getattr(parallel_state, "rank", None) == contract.global_rank,
        getattr(parallel_state, "ulysses_rank", None) == contract.ulysses_rank,
        getattr(parallel_state, "dp_rank", None) == contract.support_index,
    )
    if not all(expected_scalars):
        raise FewShotCodeTrainingError("Bernini DP2xSP4 parallel state differs")
    world_group = dist_module.group.WORLD
    if (
        dist_module.get_world_size(world_group) != EPMC_WORLD_SIZE
        or dist_module.get_rank(world_group) != contract.global_rank
    ):
        raise FewShotCodeTrainingError("default process group is not exact WORLD8")
    ulysses_group = getattr(parallel_state, "ulysses_group", None)
    dp_group = getattr(parallel_state, "dp_group", None)
    sp_members = _group_members(
        ulysses_group, expected_size=ULYSSES_SIZE, dist_module=dist_module
    )
    dp_members = _group_members(
        dp_group, expected_size=DATA_PARALLEL_SIZE, dist_module=dist_module
    )
    if sp_members != SP_GROUP_RANKS[contract.support_index]:
        raise FewShotCodeTrainingError("Ulysses group rank membership differs")
    if dp_members != DP_GROUP_RANKS[contract.ulysses_rank]:
        raise FewShotCodeTrainingError("data-parallel lane rank membership differs")
    if dist_module.get_rank(ulysses_group) != contract.ulysses_rank:
        raise FewShotCodeTrainingError("local Ulysses rank differs")
    if dist_module.get_rank(dp_group) != contract.support_index:
        raise FewShotCodeTrainingError("DP lane rank does not select its support")
    return EPMCParallelContext(
        contract=contract,
        world_group=world_group,
        ulysses_group=ulysses_group,
        dp_group=dp_group,
    )


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FewShotCodeTrainingError(
            f"value is not canonical JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    import torch

    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k2-config", required=True)
    parser.add_argument("--expected-k2-config-sha256", required=True)
    parser.add_argument("--preview-manifest", required=True)
    parser.add_argument("--vae-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--num-frames", type=int, choices=(NUM_FRAMES,), default=NUM_FRAMES
    )
    parser.add_argument("--k-shot", type=int, choices=(K_SHOT,), default=K_SHOT)
    parser.add_argument(
        "--steps-per-support", type=int, default=DEFAULT_STEPS_PER_SUPPORT
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--proposal-seed", type=int, default=DEFAULT_PROPOSAL_SEED)
    parser.add_argument(
        "--fixed-sigma-index", type=int, default=DEFAULT_FIXED_SIGMA_INDEX
    )
    parser.add_argument(
        "--held-sigma-index", type=int, default=DEFAULT_HELD_SIGMA_INDEX
    )
    parser.add_argument(
        "--full-target-fm-weight", type=float, default=FULL_TARGET_FM_WEIGHT
    )
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--posthoc-heldout-eval", action="store_true")
    parser.add_argument(
        "--ack-preview-experimental-only",
        action="store_true",
        help=(
            "acknowledge upstream training_authorized=false and "
            "training_use_forbidden=true for this engineering canary"
        ),
    )
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.ack_preview_experimental_only is not True:
        raise FewShotCodeTrainingError(
            "--ack-preview-experimental-only is mandatory"
        )
    if args.num_frames != NUM_FRAMES or args.k_shot != K_SHOT:
        raise FewShotCodeTrainingError("runner is frozen to exact 81-frame K=2")
    if type(args.steps_per_support) is not int or args.steps_per_support <= 0:
        raise FewShotCodeTrainingError("steps-per-support must be positive")
    if args.engineering_smoke and args.posthoc_heldout_eval:
        raise FewShotCodeTrainingError(
            "engineering smoke cannot deserialize the heldout parquet"
        )
    for name in ("learning_rate", "max_grad_norm"):
        value = getattr(args, name)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise FewShotCodeTrainingError(f"{name} must be finite and positive")
    if args.full_target_fm_weight != FULL_TARGET_FM_WEIGHT:
        raise FewShotCodeTrainingError(
            "full-target flow-matching weight is structurally fixed to zero"
        )
    for name in ("seed", "proposal_seed"):
        value = getattr(args, name)
        if type(value) is not int or not 0 <= value < 2**63:
            raise FewShotCodeTrainingError(f"{name} must lie in [0,2^63)")
    for name in ("fixed_sigma_index", "held_sigma_index"):
        value = getattr(args, name)
        if type(value) is not int or not 0 <= value < sigma_strata.NUM_INFERENCE_STEPS:
            raise FewShotCodeTrainingError(f"{name} must lie in [0,39]")
    if args.fixed_sigma_index == args.held_sigma_index:
        raise FewShotCodeTrainingError("fixed and held-noise sigma indices must differ")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
            raise FewShotCodeTrainingError(f"{name} must be a lowercase full SHA-1")
    for name in (
        "expected_k2_config_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise FewShotCodeTrainingError(f"{name} must be a lowercase SHA-256")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise FewShotCodeTrainingError("checkpoint tree differs from the audited 1.3B tree")
    output = Path(args.output).expanduser()
    if not output.is_absolute() or output.suffix:
        raise FewShotCodeTrainingError("output must be an absolute directory path")


def load_audited_episode(args: argparse.Namespace) -> episode_io.AuditedFewShotEpisode:
    """Use the sole checked-in role-row loader; do not reinterpret its schema."""

    try:
        return episode_io.load_epmc_k2_canary(
            args.k2_config,
            args.preview_manifest,
            args.vae_index,
            experimental_training_acknowledged=True,
            expected_config_sha256=args.expected_k2_config_sha256,
        )
    except episode_io.FewShotEpisodeIOError as error:
        raise FewShotCodeTrainingError(str(error)) from error


def _artifact_snapshot(artifact: episode_io.BoundArtifact) -> tuple[int, int, int, int]:
    return (
        artifact.device,
        artifact.inode,
        artifact.size_bytes,
        artifact.mtime_ns,
    )


def _read_episode_parquet(
    row: episode_io.AuditedEpisodeRow,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deserialize exactly one already audited row with a second TOCTOU check."""

    artifact = row.vae_parquet
    before = artifact.path.stat()
    observed_snapshot = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if observed_snapshot != _artifact_snapshot(artifact):
        raise FewShotCodeTrainingError(f"VAE parquet changed after audit: {row.iid}")
    if file_sha256(artifact.path) != artifact.sha256:
        raise FewShotCodeTrainingError(f"VAE parquet hash changed: {row.iid}")
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(artifact.path).to_pylist()
    except Exception as error:
        raise FewShotCodeTrainingError(
            f"cannot deserialize selected VAE parquet {row.iid}: {error}"
        ) from error
    after = artifact.path.stat()
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != _artifact_snapshot(artifact):
        raise FewShotCodeTrainingError(f"VAE parquet changed while reading: {row.iid}")
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise FewShotCodeTrainingError(
            f"selected parquet must contain exactly one row: {row.iid}"
        )
    raw = rows[0]
    if raw.get("iid", raw.get("id")) != row.iid:
        raise FewShotCodeTrainingError(f"selected parquet IID differs: {row.iid}")
    return raw, {
        "path": str(artifact.path),
        "sha256": artifact.sha256,
        "bytes": artifact.size_bytes,
        "deserialized_after_episode_audit": True,
    }


def _seed_for_iid(base_seed: int, iid: str, namespace: str) -> int:
    payload = f"{int(base_seed)}\0{namespace}\0{iid}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _move_batch(value: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: item.to(device, non_blocking=True)
        if isinstance(item, torch.Tensor)
        else item
        for key, item in value.items()
    }


def _sigma_for_batch(noise_scheduler: Any, batch: Mapping[str, Any]) -> Any:
    task = legacy.TASK_SOURCE_NAME.split("$", 1)[0].lower()
    shift_name = task if task in noise_scheduler.shift_config else "default"
    shift = noise_scheduler.shift_config[shift_name]
    scheduler = noise_scheduler.flow_scheduler[shift]["scheduler"]
    timestep = batch["timesteps"].reshape(-1)
    if int(timestep.numel()) != 1:
        raise FewShotCodeTrainingError("one row must contain one upstream timestep")
    sigma = scheduler.get_noise_sigma(timestep).reshape(-1)
    if int(sigma.numel()) != 1:
        raise FewShotCodeTrainingError("upstream timestep did not resolve one sigma")
    return sigma


def _normalized_source_latent(
    source_blob: Any, vae_mean: Any, vae_std: Any
) -> Any:
    """Decode only the source posterior mode to model.sample's 5-D layout."""

    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    distribution = DiagonalGaussianDistribution(legacy._load_tensor_blob(source_blob))
    source = distribution.mode().squeeze(0)
    source = ((source - vae_mean) / vae_std)[:, :LATENT_PHASES]
    source = source.unsqueeze(0).contiguous().float()
    if tuple(int(item) for item in source.shape) != LATENT_SHAPE:
        raise FewShotCodeTrainingError(
            f"source proposal latent geometry differs: {tuple(source.shape)}"
        )
    if not bool(torch.isfinite(source).all().item()):
        raise FewShotCodeTrainingError("source proposal latent is non-finite")
    return source


@dataclass
class TeacherCell:
    iid: str
    instruction: str
    action_batch: Mapping[str, Any]
    noop_batch: Mapping[str, Any]
    negative_batch: Mapping[str, Any]
    auxiliary: Mapping[str, Any]
    source_latent_cpu: Any
    parquet_receipt: Mapping[str, Any]
    noise_seed: int
    sigma_stratum: Mapping[str, Any]


@dataclass
class TrainingCell:
    teacher: TeacherCell
    carrier: Any
    activity: Any
    carrier_receipt: Mapping[str, Any]
    proposal_receipt: Mapping[str, Any]


def _prepare_teacher_cell(
    *,
    episode_row: episode_io.AuditedEpisodeRow,
    raw_row: Mapping[str, Any],
    parquet_receipt: Mapping[str, Any],
    tokenizer: Any,
    rope: Any,
    vae_mean: Any,
    vae_std: Any,
    z_dim: int,
    noise_scheduler: Any,
    process_renderer_sample: Any,
    device: Any,
    noise_seed: int,
    sigma_index: int,
) -> TeacherCell:
    """Create the target-privileged loss cell, never a carrier."""

    import torch
    import train_prior_tangent_lora as prior

    sample = legacy.sanitize_preprocessed_row(raw_row)
    source_shape, target_shape = legacy.validate_81_frame_latents(
        sample, expected_parameter_channels=2 * z_dim
    )
    if source_shape != target_shape or source_shape != episode_row.posterior_parameters_shape:
        raise FewShotCodeTrainingError("audited and materialized posterior shapes differ")
    messages = json.loads(sample["inputs"])
    if messages[1].get("text", "").strip() != episode_row.edit_instruction.strip():
        raise FewShotCodeTrainingError(
            "audited and materialized edit instructions differ"
        )
    noop_sample = motion.replace_edit_instruction(sample, EXACT_NOOP_INSTRUCTION)
    transform_kwargs = dict(
        tokenizer=tokenizer,
        vae_rope_func=rope,
        vae_latent_mean=vae_mean,
        vae_latent_std=vae_std,
        noise_scheduler=noise_scheduler,
        text_dropout_rate=0.0,
        img_dropout_rate=0.0,
        video_dropout_rate=0.0,
        max_vae_frames=LATENT_PHASES,
        source_name=legacy.TASK_SOURCE_NAME,
    )
    legacy.seed_same_sample(noise_seed)
    action = legacy.collate_single_renderer_sample(
        process_renderer_sample(sample, **transform_kwargs)
    )
    legacy.seed_same_sample(noise_seed)
    noop = legacy.collate_single_renderer_sample(
        process_renderer_sample(noop_sample, **transform_kwargs)
    )
    legacy.validate_collated_supervision(action)
    legacy.validate_collated_supervision(noop)
    if not torch.equal(action["input_vae_latents"], noop["input_vae_latents"]):
        raise FewShotCodeTrainingError("fixed-seed action/no-op latent draws differ")
    if torch.equal(action["input_ids"], noop["input_ids"]):
        raise FewShotCodeTrainingError("action and semantic-no-op text are identical")

    old_sigma = _sigma_for_batch(noise_scheduler, action)
    selector = action["vae_latents_mask"].squeeze(0).bool()
    target_noisy_old = action["input_vae_latents"][selector]
    target_velocity_old = action["target_velocity"]
    sigma_shape = [1] * target_noisy_old.ndim
    epsilon = target_noisy_old.float() + (
        1.0 - old_sigma.float().reshape(sigma_shape)
    ) * target_velocity_old.float()
    source_mode = motion.unpack_clean_mode(
        sample["video_vae_latents"][0], vae_mean, vae_std, max_frames=LATENT_PHASES
    )
    raw_target_mode = motion.unpack_clean_mode(
        sample["video_vae_latents"][1], vae_mean, vae_std, max_frames=LATENT_PHASES
    )
    target_mode = motion.project_executable_target_mode(
        source_mode, raw_target_mode, latent_frames=LATENT_PHASES
    )
    selected = sigma_strata.select_sigma_stratum(sigma_index)
    sigma_cpu = torch.tensor(selected.sigma, dtype=torch.float32, device="cpu")
    timestep_cpu = torch.tensor(selected.timestep, dtype=torch.int64, device="cpu")
    sigma_strata.assert_selected_timestep_sigma(
        timestep=timestep_cpu, sigma=sigma_cpu, selected=selected
    )
    rebuilt_action, rebuilt_noop, auxiliary = (
        motion.rebuild_bridge_state_batches_from_modes(
            action,
            noop,
            source_mode=source_mode,
            target_mode=target_mode,
            epsilon=epsilon,
            sigma=sigma_cpu,
            timestep=timestep_cpu,
            bridge_fraction=BRIDGE_FRACTION,
            minimum_sigma=float(sigma_strata.PINNED_POSITIVE_SIGMAS[-1]),
        )
    )
    negative = dict(rebuilt_noop)
    negative.update(
        prior._official_negative_text_fields(tokenizer, prior.DEFAULT_NEGATIVE_PROMPT)
    )
    for field in prior.SHARED_STATE_FIELDS:
        if not (
            torch.equal(negative[field], rebuilt_noop[field])
            and torch.equal(negative[field], rebuilt_action[field])
        ):
            raise FewShotCodeTrainingError(
                f"negative/no-op/action fixed state differs at {field}"
            )
    if any(
        torch.equal(left, right)
        for left, right in (
            (negative["input_ids"], rebuilt_noop["input_ids"]),
            (negative["input_ids"], rebuilt_action["input_ids"]),
            (rebuilt_noop["input_ids"], rebuilt_action["input_ids"]),
        )
    ):
        raise FewShotCodeTrainingError("negative/no-op/action texts must differ")

    # APG's pinned numerical program intentionally consumes a CPU fp32 scalar.
    # Moving this value to the accelerator is a train/inference mismatch.
    compact_auxiliary = {
        "source_clean": auxiliary["source_clean"].to(device),
        "target_clean": auxiliary["target_clean"].to(device),
        "shared_noisy": auxiliary["shared_noisy"].to(device),
        "sigma": auxiliary["sigma"],
        "timestep": auxiliary["timestep"],
        "branch_state_mode": auxiliary["branch_state_mode"],
        "target_projection": "source+Q0(raw_target-source)",
    }
    if (
        compact_auxiliary["sigma"].device.type != "cpu"
        or compact_auxiliary["sigma"].dtype != torch.float32
        or compact_auxiliary["sigma"].ndim != 0
    ):
        raise FewShotCodeTrainingError("APG sigma must remain CPU fp32 scalar")
    source_latent = _normalized_source_latent(
        sample["video_vae_latents"][0], vae_mean, vae_std
    )
    return TeacherCell(
        iid=episode_row.iid,
        instruction=episode_row.edit_instruction,
        action_batch=_move_batch(rebuilt_action, device),
        noop_batch=_move_batch(rebuilt_noop, device),
        negative_batch=_move_batch(negative, device),
        auxiliary=compact_auxiliary,
        source_latent_cpu=source_latent,
        parquet_receipt=parquet_receipt,
        noise_seed=noise_seed,
        sigma_stratum=selected.as_dict(),
    )


def _proposal_sample_kwargs(
    *,
    input_ids: Any,
    attention_mask: Any,
    negative_ids: Any,
    negative_mask: Any,
    source_latent: Any,
    device: Any,
    seed: int,
) -> dict[str, Any]:
    import infer_lora as inference

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "uncond_input_ids": negative_ids.to(device),
        "uncond_attention_mask": negative_mask.to(device),
        "image_vae_latents": None,
        "multi_video_vae_latents": [source_latent],
        "multi_image_vae_latents": None,
        "width": episode_io.EXPECTED_BUCKET_HW[1],
        "height": episode_io.EXPECTED_BUCKET_HW[0],
        "device": device,
        **inference.sampler_contract(steps=PROPOSAL_STEPS, seed=seed),
    }


def _restore_frozen_text_encoder(renderer: Any, device: Any) -> dict[str, Any]:
    """Undo Bernini.sample's deliberate post-sampling T5 CPU offload."""

    import torch

    encoder = getattr(renderer, "t5_text_encoder", None)
    if encoder is None or not callable(getattr(encoder, "to", None)):
        raise FewShotCodeTrainingError("renderer has no movable T5 text encoder")
    encoder.to(device)
    encoder.requires_grad_(False)
    encoder.eval()
    tensors = [*encoder.parameters(), *encoder.buffers()]
    if not tensors:
        raise FewShotCodeTrainingError("T5 text encoder has no state tensors")
    expected = torch.device(device)
    if any(item.device != expected for item in tensors):
        raise FewShotCodeTrainingError("T5 restoration left state on another device")
    if encoder.training or any(item.requires_grad for item in encoder.parameters()):
        raise FewShotCodeTrainingError("T5 restoration changed frozen eval state")
    return {
        "restored_after_sample": True,
        # Receipts cross the DP lanes and are then hashed across WORLD8.  The
        # local CUDA index is execution-local (cuda:0 ... cuda:7), not part of
        # the support semantics, so publishing it would make an otherwise
        # identical distributed receipt disagree on every lane.
        "device_type": expected.type,
        "eval": True,
        "frozen": True,
        "state_tensor_count": len(tensors),
    }


def _build_source_proposal_carrier(
    *,
    renderer: Any,
    transformer: Any,
    tokenizer: Any,
    source_latent_cpu: Any,
    instruction: str,
    source_video_sha256: str,
    instruction_sha256: str,
    proposal_seed: int,
    prompt_cleaner: Any,
    device: Any,
    ulysses_group: Any,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Build a carrier from source-conditioned proposals; target is impossible."""

    import torch
    import torch.distributed as dist
    import counterfactual_proposal_motion_rebinding as cpmr
    import fewshot_proposal_motion_carrier as proposal_carrier
    import infer_lora as inference

    action_prompt = inference.build_training_prompt(
        instruction, prompt_cleaner=prompt_cleaner
    )
    noop_prompt = inference.build_training_prompt(
        EXACT_NOOP_INSTRUCTION, prompt_cleaner=prompt_cleaner
    )
    action_ids, action_mask = inference._tokenize_training_prompt(
        tokenizer, action_prompt
    )
    noop_ids, noop_mask = inference._tokenize_training_prompt(tokenizer, noop_prompt)
    negative_ids, negative_mask = inference._tokenize_renderer_negative(
        tokenizer, inference.DEFAULT_NEGATIVE_PROMPT
    )
    source_latent = source_latent_cpu.to(device=device, dtype=torch.float32)
    common = dict(
        negative_ids=negative_ids,
        negative_mask=negative_mask,
        source_latent=source_latent,
        device=device,
        seed=proposal_seed,
    )
    action_latent = None
    noop_latent = None
    restoration_receipt: Mapping[str, Any] = {}
    try:
        with torch.no_grad():
            action_latent = renderer.sample(
                **_proposal_sample_kwargs(
                    input_ids=action_ids, attention_mask=action_mask, **common
                )
            )
        # Official sample offloads UMT5 after every trajectory.  Restore it
        # before the paired proposal and again before returning to training.
        _restore_frozen_text_encoder(renderer, device)
        with torch.no_grad():
            noop_latent = renderer.sample(
                **_proposal_sample_kwargs(
                    input_ids=noop_ids, attention_mask=noop_mask, **common
                )
            )
    finally:
        restoration_receipt = _restore_frozen_text_encoder(renderer, device)
    if action_latent is None or noop_latent is None:
        raise FewShotCodeTrainingError("Bernini proposal sampling returned no latent")
    if tuple(action_latent.shape) != LATENT_SHAPE or tuple(noop_latent.shape) != LATENT_SHAPE:
        raise FewShotCodeTrainingError("Bernini proposal latent geometry differs")
    action_sha = cpmr.tensor_sha256(action_latent)
    noop_sha = cpmr.tensor_sha256(noop_latent)
    if action_sha == noop_sha:
        raise FewShotCodeTrainingError("action/no-op proposal latents are identical")
    if dist.get_world_size(ulysses_group) != ULYSSES_SIZE:
        raise FewShotCodeTrainingError("proposal audit requires one SP4 group")
    identities: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(
        identities, (action_sha, noop_sha), group=ulysses_group
    )
    if any(item != identities[0] for item in identities[1:]):
        raise FewShotCodeTrainingError("proposal latents differ across Ulysses ranks")
    built = proposal_carrier.build_carrier_from_proposal_latents(
        transformer,
        action_latent,
        noop_latent,
        expected_patch_grid=PATCH_GRID_YX,
    )
    carrier = built.flattened(dtype=torch.bfloat16).detach()
    activity = built.activity.detach()
    proposal_receipt = {
        "source_video_sha256": source_video_sha256,
        "instruction_sha256": instruction_sha256,
        "proposal_seed": proposal_seed,
        "action_and_noop_same_source": True,
        "action_and_noop_same_seed": True,
        "proposal_steps": PROPOSAL_STEPS,
        "proposal_guidance": "v2v_apg",
        "action_latent_sha256": action_sha,
        "noop_latent_sha256": noop_sha,
        "text_encoder_after_proposals": dict(restoration_receipt),
        "target_input": False,
        "forbidden_inputs": list(FORBIDDEN_INFERENCE_CONDITIONS),
    }
    del action_latent, noop_latent, source_latent
    return carrier, activity, built.audit_receipt(), proposal_receipt


def _renderer_velocity(renderer: Any, batch: Mapping[str, Any]) -> Any:
    with __import__("torch").no_grad():
        return motion.renderer_velocity_prediction(renderer, batch)


def _coded_velocity(
    renderer: Any,
    batch: Mapping[str, Any],
    *,
    patch_handle: Any,
    motion_code: Any,
    carrier: Any,
    activity: Any,
    require_code_grad: bool,
) -> tuple[Any, Mapping[str, Any]]:
    """Mirror renderer_velocity_prediction at the authenticated shared_step."""

    import fewshot_motion_branch as fewshot_branch

    text_lens, text_embs = renderer.get_t5_text_embeddings(
        batch["input_ids"], batch["attention_mask"], batch["t5_input_lens"]
    )
    valid_samples = len(text_lens)
    vae_seqlen = batch["vae_seqlen"].squeeze(0)
    vae_seqlen = vae_seqlen[vae_seqlen > 0].unsqueeze(0)
    timesteps = batch["timesteps"].squeeze(0)[:valid_samples].unsqueeze(0)
    decoder = renderer.diff_dec
    if decoder.transformer is None or decoder.transformer_2 is not None:
        raise FewShotCodeTrainingError("EPMC requires transformer_1 only")
    inputs = batch["input_vae_latents"].unsqueeze(0)
    inputs = decoder.transformer.patch_embedding(inputs.squeeze(0)).flatten(1).unsqueeze(0)
    rope = batch["input_vae_rope"].permute(1, 0, 2).unsqueeze(0)
    target_indices = batch["vae_latents_mask"].squeeze(0).nonzero().squeeze(-1)
    result = fewshot_branch.run_training_shared_step(
        decoder,
        patch_handle=patch_handle,
        motion_code=motion_code,
        carrier=carrier,
        activity=activity,
        model_id="transformer_1",
        noisy_latents=inputs,
        timesteps=timesteps.squeeze(0),
        cond_embeds=text_embs,
        rotary_embs=rope,
        batch_vae_seqlen=vae_seqlen.squeeze(0).tolist(),
        batch_text_seqlen=text_lens,
        require_code_grad=require_code_grad,
    )
    prediction = result.prediction[:, target_indices, :]
    if tuple(prediction.shape) != (1, TARGET_TOKENS, OUTPUT_PATCH_WIDTH):
        raise FewShotCodeTrainingError("coded target velocity geometry differs")
    return prediction, result.receipt()


@dataclass(frozen=True)
class BaseAPGFields:
    negative_velocity: Any
    noop_velocity: Any
    guided_noop_clean: Any


def _base_apg_fields(renderer: Any, cell: TrainingCell) -> BaseAPGFields:
    import torch
    import train_prior_tangent_lora as prior

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        negative_velocity = _renderer_velocity(renderer, cell.teacher.negative_batch)
        noop_velocity = _renderer_velocity(renderer, cell.teacher.noop_batch)
    if negative_velocity.dtype != torch.bfloat16 or noop_velocity.dtype != torch.bfloat16:
        raise FewShotCodeTrainingError("Bernini raw APG velocities must remain bfloat16")
    with torch.no_grad():
        _, guided_noop = prior._guided_clean(
            shared_noisy=cell.teacher.auxiliary["shared_noisy"],
            sigma=cell.teacher.auxiliary["sigma"],
            negative_velocity=negative_velocity,
            conditional_velocity=noop_velocity,
        )
    return BaseAPGFields(negative_velocity, noop_velocity, guided_noop)


def _motion_loss(
    renderer: Any,
    cell: TrainingCell,
    base: BaseAPGFields,
    *,
    patch_handle: Any,
    code: Any,
    require_code_grad: bool,
) -> tuple[Any, dict[str, Any], Mapping[str, Any], Any]:
    import fewshot_teacher_objective as teacher_objective
    import train_prior_tangent_lora as prior

    coded_velocity, branch_receipt = _coded_velocity(
        renderer,
        cell.teacher.noop_batch,
        patch_handle=patch_handle,
        motion_code=code,
        carrier=cell.carrier,
        activity=cell.activity,
        require_code_grad=require_code_grad,
    )
    _, guided_coded = prior._guided_clean(
        shared_noisy=cell.teacher.auxiliary["shared_noisy"],
        sigma=cell.teacher.auxiliary["sigma"],
        negative_velocity=base.negative_velocity,
        conditional_velocity=coded_velocity,
    )
    predicted_delta = (guided_coded - base.guided_noop_clean).reshape(
        1, TARGET_TOKENS, OUTPUT_PATCH_WIDTH
    )
    source_clean = _packed_clean_video(cell.teacher.auxiliary["source_clean"])
    target_clean = _packed_clean_video(cell.teacher.auxiliary["target_clean"])
    predicted_clean = source_clean + _packed_clean_video(predicted_delta)
    objective = teacher_objective.fewshot_teacher_objective(
        source_clean,
        predicted_clean,
        target_clean,
        code.phase_gates,
        code.block_head_gates[:, :, 0],
    )
    return (
        objective.total,
        objective.detached_statistics(),
        branch_receipt,
        coded_velocity,
    )


def _packed_clean_video(packed: Any) -> Any:
    """Invert Wan's exact ``(t,h,w),(pt,ph,pw,c)`` packed token order."""

    import torch

    if (
        not isinstance(packed, torch.Tensor)
        or packed.dtype != torch.float32
        or tuple(packed.shape) != (1, TARGET_TOKENS, OUTPUT_PATCH_WIDTH)
    ):
        raise FewShotCodeTrainingError(
            "packed clean field must be fp32 [1,19530,64]"
        )
    if not bool(torch.isfinite(packed).all().item()):
        raise FewShotCodeTrainingError("packed clean field is non-finite")
    # N=(t,h,w)=(21,30,31); D=(pt,ph,pw,c)=(1,2,2,16).
    patches = packed.reshape(1, 21, 30, 31, 1, 2, 2, 16)
    video = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
        1, 16, 21, 60, 62
    )
    return video.contiguous()


def _all_reduce_code_gradients(module: Any, *, ulysses_group: Any) -> float:
    """Explicitly synchronize replicated 36D gradients over the Ulysses group."""

    import torch
    import torch.distributed as dist

    if (
        not dist.is_initialized()
        or ulysses_group is None
        or dist.get_world_size(ulysses_group) != ULYSSES_SIZE
    ):
        raise FewShotCodeTrainingError("code gradients require one exact SP4 group")
    parameters = [item for item in module.parameters() if item.requires_grad]
    local_ready = all(
        item.grad is not None and bool(torch.isfinite(item.grad).all().item())
        for item in parameters
    )
    if not _all_group_true(local_ready, group=ulysses_group):
        raise FewShotCodeTrainingError(
            "at least one rank has a missing/non-finite code gradient"
        )
    squared = torch.zeros((), dtype=torch.float32, device="cuda")
    count = 0
    for parameter in parameters:
        assert parameter.grad is not None
        dist.all_reduce(
            parameter.grad, op=dist.ReduceOp.SUM, group=ulysses_group
        )
        parameter.grad.div_(float(ULYSSES_SIZE))
        squared.add_(parameter.grad.float().square().sum())
        count += int(parameter.numel())
    if count != TRAINABLE_CODE_DIMENSION:
        raise FewShotCodeTrainingError("gradient synchronization scope is not 36D")
    norm = float(squared.sqrt().item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise FewShotCodeTrainingError("synchronized code gradient is zero/non-finite")
    return norm


def _all_group_true(value: bool, *, group: Any) -> bool:
    import torch
    import torch.distributed as dist

    probe = torch.tensor(int(value), dtype=torch.int32, device="cuda")
    if group is None:
        raise FewShotCodeTrainingError("boolean consensus requires an explicit group")
    dist.all_reduce(probe, op=dist.ReduceOp.MIN, group=group)
    return bool(probe.item())


def _tied_code_36d(code: Any) -> Any:
    import torch

    code.validate()
    value = torch.cat(
        (code.phase_gates[:, 1:], code.block_head_gates[:, :, 0]), dim=1
    ).detach().float().contiguous()
    if tuple(value.shape) != (1, TRAINABLE_CODE_DIMENSION):
        raise FewShotCodeTrainingError("derived tied motion code is not [1,36]")
    return value


def _code_hash(code: Any) -> str:
    import counterfactual_proposal_motion_rebinding as cpmr

    return cpmr.tensor_sha256(_tied_code_36d(code))


def _assert_code_group_exact(code: Any, *, ulysses_group: Any) -> str:
    import torch.distributed as dist

    digest = _code_hash(code)
    if dist.get_world_size(ulysses_group) != ULYSSES_SIZE:
        raise FewShotCodeTrainingError("code identity requires one exact SP4 group")
    values: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(values, digest, group=ulysses_group)
    if any(item != digest for item in values):
        raise FewShotCodeTrainingError("replicated code differs across ranks")
    return digest


def _motion_code_from_tied_36d(value: Any) -> Any:
    import torch
    import fewshot_privileged_motion_code as epmc

    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or tuple(value.shape) != (1, TRAINABLE_CODE_DIMENSION)
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise FewShotCodeTrainingError("exchanged tied motion code is not finite fp32 [1,36]")
    phase = torch.zeros((1, LATENT_PHASES), dtype=torch.float32, device=value.device)
    phase[:, 1:] = value[:, :20]
    blocks = (
        value[:, 20:]
        .reshape(1, 16, 1)
        .expand(1, 16, 12)
        .clone()
        .contiguous()
    )
    code = epmc.MotionCode(phase.contiguous(), blocks)
    code.validate()
    if not bool(torch.equal(_tied_code_36d(code), value)):
        raise FewShotCodeTrainingError("K=2 code reconstruction changed tied values")
    return code


def _digest_projection(value: Any) -> Any:
    """Convert diagnostics containing tensors into canonical digest material."""

    import torch
    import counterfactual_proposal_motion_rebinding as cpmr

    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous()
        return {
            "tensor_sha256": cpmr.tensor_sha256(tensor),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise FewShotCodeTrainingError("consensus mapping keys must be strings")
        return {key: _digest_projection(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_digest_projection(item) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise FewShotCodeTrainingError(
        f"unsupported consensus value type: {type(value).__name__}"
    )


def _world_digest_consensus(
    value: Any,
    *,
    world_group: Any,
    context: str,
) -> str:
    import torch.distributed as dist

    if dist.get_world_size(world_group) != EPMC_WORLD_SIZE:
        raise FewShotCodeTrainingError(f"{context} consensus requires WORLD8")
    digest = object_sha256(_digest_projection(value))
    gathered: list[Any] = [None] * EPMC_WORLD_SIZE
    dist.all_gather_object(gathered, digest, group=world_group)
    if any(item != digest for item in gathered):
        raise FewShotCodeTrainingError(f"{context} differs across WORLD8")
    return digest


def _exchange_k2_codes(
    local_code: Any,
    *,
    support_index: int,
    support_iids: Sequence[str],
    dp_group: Any,
    world_group: Any,
) -> tuple[list[Any], str]:
    """Exchange one SP-replicated code along each DP2 lane in support order."""

    import torch
    import torch.distributed as dist

    if len(support_iids) != K_SHOT or support_index not in range(K_SHOT):
        raise FewShotCodeTrainingError("K=2 code exchange support identity differs")
    if (
        dist.get_world_size(dp_group) != DATA_PARALLEL_SIZE
        or dist.get_rank(dp_group) != support_index
    ):
        raise FewShotCodeTrainingError("K=2 code exchange DP lane ordering differs")
    local_tied = _tied_code_36d(local_code)
    gathered = [torch.empty_like(local_tied) for _ in range(K_SHOT)]
    dist.all_gather(gathered, local_tied, group=dp_group)
    if any(
        tuple(item.shape) != (1, TRAINABLE_CODE_DIMENSION)
        or item.dtype != torch.float32
        or not bool(torch.isfinite(item).all().item())
        for item in gathered
    ):
        raise FewShotCodeTrainingError("K=2 code exchange returned invalid tensors")
    codes = [_motion_code_from_tied_36d(item.contiguous()) for item in gathered]
    digest = _world_digest_consensus(
        {
            "support_iids": list(support_iids),
            "ordered_tied_codes": gathered,
        },
        world_group=world_group,
        context="ordered K=2 code exchange",
    )
    return codes, digest


def _exchange_k2_objects(
    local_value: Any,
    *,
    support_index: int,
    local_iid: str,
    support_iids: Sequence[str],
    dp_group: Any,
    world_group: Any,
    context: str,
) -> tuple[list[Any], str]:
    """Exchange support metadata on every DP lane and audit WORLD8 equality."""

    import torch.distributed as dist

    if len(support_iids) != K_SHOT or support_index not in range(K_SHOT):
        raise FewShotCodeTrainingError(f"{context} support identity differs")
    if local_iid != support_iids[support_index]:
        raise FewShotCodeTrainingError(f"{context} local IID is assigned incorrectly")
    if (
        dist.get_world_size(dp_group) != DATA_PARALLEL_SIZE
        or dist.get_rank(dp_group) != support_index
    ):
        raise FewShotCodeTrainingError(f"{context} DP lane ordering differs")
    envelope = {
        "support_index": support_index,
        "iid": local_iid,
        "value": local_value,
    }
    gathered: list[Any] = [None] * K_SHOT
    dist.all_gather_object(gathered, envelope, group=dp_group)
    ordered: list[Any] = [None] * K_SHOT
    seen: set[int] = set()
    for item in gathered:
        if not isinstance(item, Mapping):
            raise FewShotCodeTrainingError(f"{context} exchange envelope is invalid")
        index = item.get("support_index")
        iid = item.get("iid")
        if type(index) is not int or index not in range(K_SHOT) or index in seen:
            raise FewShotCodeTrainingError(f"{context} exchange ordering is invalid")
        if iid != support_iids[index]:
            raise FewShotCodeTrainingError(f"{context} exchange IID differs")
        seen.add(index)
        ordered[index] = item.get("value")
    if seen != set(range(K_SHOT)):
        raise FewShotCodeTrainingError(f"{context} exchange is incomplete")
    digest = _world_digest_consensus(
        {"support_iids": list(support_iids), "ordered_values": ordered},
        world_group=world_group,
        context=context,
    )
    return ordered, digest


def _gradient_probe(
    renderer: Any,
    cell: TrainingCell,
    base: BaseAPGFields,
    *,
    patch_handle: Any,
    family: str,
    ulysses_group: Any,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    import counterfactual_proposal_motion_rebinding as cpmr
    import fewshot_motion_branch as fewshot_branch

    if family not in ("phase_only", "block_only"):
        raise FewShotCodeTrainingError("unknown gradient-probe family")
    module = fewshot_branch.TiedHeadEpisodicMotionCode().to("cuda")
    if family == "phase_only":
        module.block_logits.requires_grad_(False)
        selected = module.phase_logits_nonzero
        excluded = module.block_logits
    else:
        module.phase_logits_nonzero.requires_grad_(False)
        selected = module.block_logits
        excluded = module.phase_logits_nonzero
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _, branch_receipt, _ = _motion_loss(
            renderer,
            cell,
            base,
            patch_handle=patch_handle,
            code=module(),
            require_code_grad=True,
        )
    loss.backward()
    routing_ok = selected.grad is not None and excluded.grad is None
    if not _all_group_true(routing_ok, group=ulysses_group):
        raise FewShotCodeTrainingError(f"{family} gradient routing failed")
    assert selected.grad is not None
    if dist.get_world_size(ulysses_group) != ULYSSES_SIZE:
        raise FewShotCodeTrainingError("gradient probe requires one exact SP4 group")
    dist.all_reduce(
        selected.grad, op=dist.ReduceOp.SUM, group=ulysses_group
    )
    selected.grad.div_(float(ULYSSES_SIZE))
    gradient_rms = float(selected.grad.float().square().mean().sqrt().item())
    local_ok = bool(torch.isfinite(selected.grad).all().item()) and gradient_rms > 1.0e-8
    if not _all_group_true(local_ok, group=ulysses_group):
        raise FewShotCodeTrainingError(f"{family} gradient probe failed")
    gradient_sha = cpmr.tensor_sha256(selected.grad)
    gathered: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(gathered, gradient_sha, group=ulysses_group)
    if any(item != gradient_sha for item in gathered):
        raise FewShotCodeTrainingError(f"{family} synchronized gradient differs")
    return {
        "iid": cell.teacher.iid,
        "family": family,
        "passed": True,
        "selected_gradient_l1": float(selected.grad.abs().sum().item()),
        "selected_gradient_rms": gradient_rms,
        "selected_gradient_sha256": gradient_sha,
        "gradient_sync": WITHIN_SUPPORT_GRADIENT_SYNC,
        "excluded_gradient_absent": True,
        "single_step_apg_surrogate": True,
        "branch": dict(branch_receipt),
    }


def _invert_support(
    renderer: Any,
    cell: TrainingCell,
    *,
    patch_handle: Any,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
    ulysses_group: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch
    import fewshot_motion_branch as fewshot_branch

    code_module = fewshot_branch.TiedHeadEpisodicMotionCode().to("cuda")
    optimizer = torch.optim.Adam(
        tuple(code_module.parameters()), lr=learning_rate, weight_decay=0.0
    )
    base = _base_apg_fields(renderer, cell)
    history: list[dict[str, Any]] = []
    last_branch: Mapping[str, Any] = {}
    last_objective: Mapping[str, Any] = {}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, last_objective, last_branch, _ = _motion_loss(
                renderer,
                cell,
                base,
                patch_handle=patch_handle,
                code=code_module(),
                require_code_grad=True,
            )
        if not _all_group_true(
            bool(torch.isfinite(loss.detach()).item()), group=ulysses_group
        ):
            raise FewShotCodeTrainingError("support inversion loss is non-finite")
        loss.backward()
        preclip_norm = _all_reduce_code_gradients(
            code_module, ulysses_group=ulysses_group
        )
        torch.nn.utils.clip_grad_norm_(tuple(code_module.parameters()), max_grad_norm)
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().item()),
                "preclip_gradient_norm": preclip_norm,
                "objective": dict(last_objective),
            }
        )
    code = code_module()
    code.validate()
    detached = type(code)(
        code.phase_gates.detach().float().clone(),
        code.block_head_gates.detach().float().clone(),
    )
    code_hash = _assert_code_group_exact(
        detached, ulysses_group=ulysses_group
    )
    diagnostic = {
        "iid": cell.teacher.iid,
        "steps": steps,
        "history": history,
        "phase_logits_nonzero": code_module.phase_logits_nonzero.detach().cpu(),
        "block_logits": code_module.block_logits.detach().cpu(),
    }
    receipt = {
        "iid": cell.teacher.iid,
        "tied_36d_code_sha256": code_hash,
        "final_loss": history[-1]["loss"],
        "steps": steps,
        "fixed_noise_seed": cell.teacher.noise_seed,
        "sigma_stratum": dict(cell.teacher.sigma_stratum),
        "gradient_sync": WITHIN_SUPPORT_GRADIENT_SYNC,
        "full_target_flow_matching_weight": FULL_TARGET_FM_WEIGHT,
        "single_step_apg_surrogate": True,
        "final_teacher_objective": dict(last_objective),
        "last_branch": dict(last_branch),
        "carrier": dict(cell.carrier_receipt),
        "proposal": dict(cell.proposal_receipt),
        "parquet": dict(cell.teacher.parquet_receipt),
    }
    return detached, receipt, diagnostic


def _tensor_bytes_equal(left: Any, right: Any) -> bool:
    import torch

    return (
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and bool(
            torch.equal(
                left.detach().contiguous().reshape(-1).view(torch.uint8),
                right.detach().contiguous().reshape(-1).view(torch.uint8),
            )
        )
    )


def _evaluate_controls(
    renderer: Any,
    cell: TrainingCell,
    code: Any,
    *,
    patch_handle: Any,
) -> dict[str, Any]:
    import torch
    import fewshot_motion_branch as fewshot_branch
    import fewshot_privileged_motion_code as epmc

    base = _base_apg_fields(renderer, cell)
    controls = {
        "correct": code,
        "zero": fewshot_branch.canonical_tied_noop_motion_code(device="cuda"),
        "reverse": epmc.permute_motion_code_phases(
            code, epmc.REVERSE_PHASE_INDICES
        ),
        "shuffle": epmc.permute_motion_code_phases(
            code, epmc.SHUFFLE_PHASE_INDICES
        ),
    }
    losses: dict[str, float] = {}
    objective_statistics: dict[str, Mapping[str, Any]] = {}
    zero_velocity = None
    # The authenticated shared-step contract requires grad mode even for
    # evaluation.  Renderer/clone parameters and these detached control codes
    # are frozen, so keeping grad mode enabled does not make them trainable.
    for name, control in controls.items():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, parts, _, velocity = _motion_loss(
                renderer,
                cell,
                base,
                patch_handle=patch_handle,
                code=control,
                require_code_grad=False,
            )
        losses[name] = float(loss.item())
        objective_statistics[name] = dict(parts)
        if name == "zero":
            zero_velocity = velocity
    assert zero_velocity is not None
    zero_base_parity = _tensor_bytes_equal(zero_velocity, base.noop_velocity)
    if not zero_base_parity:
        raise FewShotCodeTrainingError(
            "zero EPMC code is not byte-identical to raw semantic no-op"
        )
    passed = all(
        losses["correct"] < losses[name]
        for name in ("zero", "reverse", "shuffle")
    )
    return {
        "iid": cell.teacher.iid,
        "held_noise": True,
        "sigma_stratum": dict(cell.teacher.sigma_stratum),
        "noise_seed": cell.teacher.noise_seed,
        "losses": losses,
        "objective_statistics": objective_statistics,
        "correct_strictly_better_than_zero_reverse_shuffle": passed,
        "zero_code_raw_noop_byte_exact": True,
        "full_target_flow_matching_weight": FULL_TARGET_FM_WEIGHT,
        "single_step_apg_surrogate": True,
    }


def validate_prototype_tensors(phase_gates: Any, block_head_gates: Any) -> None:
    import torch

    expected = {
        "phase_gates": ((1, 21), phase_gates),
        "block_head_gates": ((1, 16, 12), block_head_gates),
    }
    for name, (shape, value) in expected.items():
        if not isinstance(value, torch.Tensor):
            raise FewShotCodeTrainingError(f"prototype {name} must be a tensor")
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise FewShotCodeTrainingError(f"prototype {name} must be CPU float32")
        if tuple(value.shape) != shape or not value.is_contiguous():
            raise FewShotCodeTrainingError(f"prototype {name} geometry differs")
        if not bool(torch.isfinite(value).all().item()) or bool((value.abs() > 1).any()):
            raise FewShotCodeTrainingError(f"prototype {name} escaped finite [-1,1]")
    phase_zero_bytes = phase_gates[:, 0].reshape(-1).repeat(1).view(torch.uint8)
    if int(torch.count_nonzero(phase_zero_bytes).item()) != 0:
        raise FewShotCodeTrainingError("prototype phase 0 is not byte-positive-zero")
    reference = block_head_gates[:, :, :1].expand_as(block_head_gates).contiguous()
    if not _tensor_bytes_equal(block_head_gates, reference):
        raise FewShotCodeTrainingError("prototype heads are not byte-identically tied")


def _atomic_save_prototype(path: Path, code: Any) -> dict[str, Any]:
    import torch
    import counterfactual_proposal_motion_rebinding as cpmr
    from safetensors import safe_open
    from safetensors.torch import save_file

    phase = code.phase_gates.detach().cpu().float().contiguous()
    blocks = code.block_head_gates.detach().cpu().float().contiguous()
    validate_prototype_tensors(phase, blocks)
    tensors = {"phase_gates": phase, "block_head_gates": blocks}
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".safetensors", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(tensors, str(temporary))
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            keys = sorted(opened.keys())
            if keys != ["block_head_gates", "phase_gates"]:
                raise FewShotCodeTrainingError("prototype safetensors keys differ")
            loaded_phase = opened.get_tensor("phase_gates").contiguous()
            loaded_blocks = opened.get_tensor("block_head_gates").contiguous()
        validate_prototype_tensors(loaded_phase, loaded_blocks)
        if not _tensor_bytes_equal(phase, loaded_phase) or not _tensor_bytes_equal(
            blocks, loaded_blocks
        ):
            raise FewShotCodeTrainingError("prototype safetensors round-trip differs")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "tensor_keys": ["block_head_gates", "phase_gates"],
        "phase_gates_sha256": cpmr.tensor_sha256(phase),
        "block_head_gates_sha256": cpmr.tensor_sha256(blocks),
        "phase_gates_shape": [1, 21],
        "block_head_gates_shape": [1, 16, 12],
        "dtype": "torch.float32",
        "device": "cpu",
        "phase0_exact_positive_zero": True,
        "heads_tied_byte_exact": True,
    }


def representability_decision(
    probes: Sequence[Mapping[str, Any]],
    teacher_go: Any,
    *,
    support_iids: Sequence[str],
) -> str:
    if len(support_iids) != K_SHOT or len(set(support_iids)) != K_SHOT:
        raise FewShotCodeTrainingError("representability requires two distinct supports")
    expected = {
        (support_iids[0], family)
        for family in episode_parallel.REFERENCE_PROBE_FAMILIES
    }
    passed = {
        (str(item.get("iid")), str(item.get("family")))
        for item in probes
        if item.get("passed") is True
    }
    teacher_passed = getattr(teacher_go, "go", None) is True
    return (
        "GO"
        if len(probes) == len(expected) and passed == expected and teacher_passed
        else "NO_GO"
    )


def _support_assignments(
    supports: Sequence[episode_io.AuditedEpisodeRow],
) -> list[dict[str, Any]]:
    if len(supports) != K_SHOT:
        raise FewShotCodeTrainingError("support-parallel runner requires exact K=2")
    iids = [row.iid for row in supports]
    if len(set(iids)) != K_SHOT:
        raise FewShotCodeTrainingError("support-parallel IIDs must be distinct")
    return [
        {
            "support_index": index + 1,
            "iid": iid,
            "dp_rank": index,
            "sp_ranks": list(SP_GROUP_RANKS[index]),
        }
        for index, iid in enumerate(iids)
    ]


def _distributed_receipt(
    *, support_assignments: Sequence[Mapping[str, Any]], backend: str
) -> dict[str, Any]:
    return {
        "world_size": EPMC_WORLD_SIZE,
        "ulysses_size": ULYSSES_SIZE,
        "data_parallel_size": DATA_PARALLEL_SIZE,
        "support_parallel": True,
        "sp_groups": [list(group) for group in SP_GROUP_RANKS],
        "dp_groups": [list(group) for group in DP_GROUP_RANKS],
        "support_assignments": [dict(item) for item in support_assignments],
        "within_support_gradient_sync": WITHIN_SUPPORT_GRADIENT_SYNC,
        "cross_support_gradient_sync": False,
        "k2_exchange": K2_EXCHANGE,
        "backend": backend,
    }


def _make_training_cell(
    *,
    episode_row: episode_io.AuditedEpisodeRow,
    raw_row: Mapping[str, Any],
    parquet_receipt: Mapping[str, Any],
    renderer: Any,
    transformer: Any,
    tokenizer: Any,
    rope: Any,
    vae_mean: Any,
    vae_std: Any,
    z_dim: int,
    scheduler: Any,
    process_renderer_sample: Any,
    prompt_cleaner: Any,
    device: Any,
    noise_seed: int,
    sigma_index: int,
    proposal_seed: int,
    ulysses_group: Any,
) -> TrainingCell:
    teacher = _prepare_teacher_cell(
        episode_row=episode_row,
        raw_row=raw_row,
        parquet_receipt=parquet_receipt,
        tokenizer=tokenizer,
        rope=rope,
        vae_mean=vae_mean,
        vae_std=vae_std,
        z_dim=z_dim,
        noise_scheduler=scheduler,
        process_renderer_sample=process_renderer_sample,
        device=device,
        noise_seed=noise_seed,
        sigma_index=sigma_index,
    )
    carrier, activity, carrier_receipt, proposal_receipt = (
        _build_source_proposal_carrier(
            renderer=renderer,
            transformer=transformer,
            tokenizer=tokenizer,
            source_latent_cpu=teacher.source_latent_cpu,
            instruction=episode_row.edit_instruction,
            source_video_sha256=episode_row.source_video_sha256,
            instruction_sha256=episode_row.edit_instruction_sha256,
            proposal_seed=proposal_seed,
            prompt_cleaner=prompt_cleaner,
            device=device,
            ulysses_group=ulysses_group,
        )
    )
    return TrainingCell(
        teacher=teacher,
        carrier=carrier,
        activity=activity,
        carrier_receipt=carrier_receipt,
        proposal_receipt=proposal_receipt,
    )


def _create_output_directory(
    path: Path, *, rank: int, world_group: Any
) -> None:
    import torch.distributed as dist

    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise FewShotCodeTrainingError("output parent is not a directory")
    if path.exists() or path.is_symlink():
        raise FewShotCodeTrainingError(f"refusing to overwrite output: {path}")
    dist.barrier(group=world_group)
    if rank == 0:
        path.mkdir(mode=0o750)
    dist.barrier(group=world_group)
    if not path.is_dir():
        raise FewShotCodeTrainingError("rank zero did not create output directory")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    episode = load_audited_episode(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise FewShotCodeTrainingError(str(error)) from error
    if transformer_config["num_attention_heads"] % 4:
        raise FewShotCodeTrainingError("12 Bernini heads must divide across Ulysses=4")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, __version__ as transformers_version
    from diffusers import __version__ as diffusers_version
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.training.data import NoiseScheduler, process_renderer_sample
    import counterfactual_proposal_motion_branch as cpmr_branch
    import counterfactual_proposal_motion_rebinding as cpmr
    import fewshot_motion_branch as fewshot_branch
    import fewshot_privileged_motion_code as epmc
    import fewshot_teacher_objective as teacher_objective
    import infer_lora as inference

    contract = epmc_distributed_contract()
    device, backend = initialise_epmc_distributed(contract)
    parallel_state = init_parallel_state(ulysses_size=ULYSSES_SIZE)
    parallel = validate_epmc_parallel_state(contract, parallel_state)
    support_assignments = _support_assignments(episode.supports)
    support_iids = [row.iid for row in episode.supports]
    assigned_row = episode.supports[parallel.support_index]
    distributed_receipt = _distributed_receipt(
        support_assignments=support_assignments, backend=backend
    )
    output_dir = Path(args.output).expanduser()
    _create_output_directory(
        output_dir,
        rank=contract.global_rank,
        world_group=parallel.world_group,
    )
    legacy.seed_same_sample(args.seed)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **inference.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False)
    renderer.eval()
    renderer.t5_text_encoder.eval()
    renderer.to(device)
    transformer = cpmr_branch.resolve_wan_transformer(renderer)
    if bool(getattr(transformer, "gradient_checkpointing", False)):
        raise FewShotCodeTrainingError("gradient checkpointing must remain disabled")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**legacy.noise_scheduler_kwargs())

    raw, parquet_receipt = _read_episode_parquet(assigned_row)
    fixed_cell = _make_training_cell(
        episode_row=assigned_row,
        raw_row=raw,
        parquet_receipt=parquet_receipt,
        renderer=renderer,
        transformer=transformer,
        tokenizer=tokenizer,
        rope=rope,
        vae_mean=vae_mean,
        vae_std=vae_std,
        z_dim=z_dim,
        scheduler=scheduler,
        process_renderer_sample=process_renderer_sample,
        prompt_cleaner=prompt_clean,
        device=device,
        noise_seed=_seed_for_iid(args.seed, assigned_row.iid, "fixed"),
        sigma_index=args.fixed_sigma_index,
        proposal_seed=args.proposal_seed,
        ulysses_group=parallel.ulysses_group,
    )

    patch_handle = fewshot_branch.install_fewshot_motion_branch(renderer)
    local_probes: list[dict[str, Any]] = []
    # Preserve the preregistered four-rank serial gate exactly: phase/block
    # routing probes are reference evidence from support 1 only.  The second
    # SP4 group may begin its independent inversion and wait at the later DP
    # exchange; no collective crosses the two Ulysses groups here.
    if not args.engineering_smoke and parallel.support_index == 0:
        probe_base = _base_apg_fields(renderer, fixed_cell)
        local_probes = [
            _gradient_probe(
                renderer,
                fixed_cell,
                probe_base,
                patch_handle=patch_handle,
                family=family,
                ulysses_group=parallel.ulysses_group,
            )
            for family in episode_parallel.REFERENCE_PROBE_FAMILIES
        ]

    steps = 1 if args.engineering_smoke else args.steps_per_support
    local_code, local_support_receipt, local_diagnostic = _invert_support(
        renderer,
        fixed_cell,
        patch_handle=patch_handle,
        steps=steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        ulysses_group=parallel.ulysses_group,
    )
    local_assignment = support_assignments[parallel.support_index]
    local_support_receipt = {
        **local_support_receipt,
        "support_index": local_assignment["support_index"],
        "dp_rank": local_assignment["dp_rank"],
        "sp_ranks": list(local_assignment["sp_ranks"]),
    }
    local_diagnostic = {
        **local_diagnostic,
        "support_index": local_assignment["support_index"],
        "dp_rank": local_assignment["dp_rank"],
        "sp_ranks": list(local_assignment["sp_ranks"]),
    }
    local_probes = [
        {
            **probe,
            "support_index": local_assignment["support_index"],
            "dp_rank": local_assignment["dp_rank"],
            "sp_ranks": list(local_assignment["sp_ranks"]),
        }
        for probe in local_probes
    ]
    support_codes, code_exchange_digest = _exchange_k2_codes(
        local_code,
        support_index=parallel.support_index,
        support_iids=support_iids,
        dp_group=parallel.dp_group,
        world_group=parallel.world_group,
    )
    support_receipts, receipt_exchange_digest = _exchange_k2_objects(
        local_support_receipt,
        support_index=parallel.support_index,
        local_iid=assigned_row.iid,
        support_iids=support_iids,
        dp_group=parallel.dp_group,
        world_group=parallel.world_group,
        context="ordered K=2 support receipts",
    )
    support_diagnostics, diagnostic_exchange_digest = _exchange_k2_objects(
        local_diagnostic,
        support_index=parallel.support_index,
        local_iid=assigned_row.iid,
        support_iids=support_iids,
        dp_group=parallel.dp_group,
        world_group=parallel.world_group,
        context="ordered K=2 support diagnostics",
    )
    probe_bundles, probe_exchange_digest = _exchange_k2_objects(
        local_probes,
        support_index=parallel.support_index,
        local_iid=assigned_row.iid,
        support_iids=support_iids,
        dp_group=parallel.dp_group,
        world_group=parallel.world_group,
        context="ordered K=2 gradient probes",
    )
    probes = [probe for bundle in probe_bundles for probe in bundle]
    for index, (code, receipt) in enumerate(zip(support_codes, support_receipts)):
        if not isinstance(receipt, Mapping) or (
            receipt.get("iid") != support_iids[index]
            or receipt.get("tied_36d_code_sha256") != _code_hash(code)
        ):
            raise FewShotCodeTrainingError(
                "exchanged support receipt does not bind its ordered motion code"
            )

    if args.engineering_smoke:
        if contract.global_rank == 0:
            diagnostic_path = output_dir / "diagnostics.pt"
            _atomic_torch_save(
                diagnostic_path,
                {
                    "schema_version": DIAGNOSTIC_STATE_SCHEMA,
                    "engineering_smoke": True,
                    "supports": support_diagnostics,
                },
            )
            receipt = {
                "schema_version": RUN_RECEIPT_SCHEMA,
                "method": METHOD_NAME,
                "engineering_smoke": True,
                "support_count": K_SHOT,
                "optimizer_steps": K_SHOT,
                "optimizer_steps_per_support": 1,
                "representability_gate": "NOT_EVALUATED_ENGINEERING_SMOKE",
                "prototype_published": False,
                "single_step_apg_surrogate": True,
                "full_trajectory_render": False,
                "full_target_flow_matching_weight": FULL_TARGET_FM_WEIGHT,
                "episode_audit": episode.audit_receipt(),
                "support": support_receipts,
                "distributed": distributed_receipt,
                "k2_exchange_consensus": {
                    "code": code_exchange_digest,
                    "receipt": receipt_exchange_digest,
                    "diagnostic": diagnostic_exchange_digest,
                    "probe": probe_exchange_digest,
                },
                "diagnostics_sha256": file_sha256(diagnostic_path),
            }
            receipt["receipt_sha256"] = object_sha256(receipt)
            _atomic_write_json(output_dir / "smoke.receipt.json", receipt)
        dist.barrier(group=parallel.world_group)
        patch_handle.restore()
        dist.destroy_process_group()
        return 0

    raw, parquet_receipt = _read_episode_parquet(assigned_row)
    held_teacher = _prepare_teacher_cell(
        episode_row=assigned_row,
        raw_row=raw,
        parquet_receipt=parquet_receipt,
        tokenizer=tokenizer,
        rope=rope,
        vae_mean=vae_mean,
        vae_std=vae_std,
        z_dim=z_dim,
        noise_scheduler=scheduler,
        process_renderer_sample=process_renderer_sample,
        device=device,
        noise_seed=_seed_for_iid(args.seed, assigned_row.iid, "held"),
        sigma_index=args.held_sigma_index,
    )
    held_cell = TrainingCell(
        teacher=held_teacher,
        carrier=fixed_cell.carrier,
        activity=fixed_cell.activity,
        carrier_receipt=fixed_cell.carrier_receipt,
        proposal_receipt=fixed_cell.proposal_receipt,
    )
    local_held_control = _evaluate_controls(
        renderer,
        held_cell,
        support_codes[parallel.support_index],
        patch_handle=patch_handle,
    )
    held_controls, held_control_exchange_digest = _exchange_k2_objects(
        local_held_control,
        support_index=parallel.support_index,
        local_iid=assigned_row.iid,
        support_iids=support_iids,
        dp_group=parallel.dp_group,
        world_group=parallel.world_group,
        context="ordered K=2 held-noise controls",
    )

    stacked = epmc.MotionCode(
        torch.cat([item.phase_gates for item in support_codes], dim=0),
        torch.cat([item.block_head_gates for item in support_codes], dim=0),
    )
    prototype_result = epmc.build_training_support_prototype(stacked)
    prototype = prototype_result.code
    prototype_hash = _assert_code_group_exact(
        prototype, ulysses_group=parallel.ulysses_group
    )
    prototype_consensus_digest = _world_digest_consensus(
        {"prototype_sha256": prototype_hash, "prototype": _tied_code_36d(prototype)},
        world_group=parallel.world_group,
        context="K=2 prototype",
    )
    held_controls_for_aggregation = sorted(
        held_controls, key=lambda row: str(row["iid"])
    )
    if (
        len(held_controls_for_aggregation) != K_SHOT
        or {str(row["iid"]) for row in held_controls_for_aggregation}
        != set(support_iids)
    ):
        raise FewShotCodeTrainingError(
            "held-noise controls do not bind the exact two support IIDs"
        )
    mean_control_losses = {
        name: sum(
            float(row["losses"][name])
            for row in held_controls_for_aggregation
        )
        / K_SHOT
        for name in ("zero", "correct", "reverse", "shuffle")
    }
    support_tied_codes = torch.cat(
        [_tied_code_36d(item) for item in support_codes], dim=0
    )
    held_statistics = teacher_objective.build_held_noise_statistics(
        zero_loss=mean_control_losses["zero"],
        correct_loss=mean_control_losses["correct"],
        reverse_loss=mean_control_losses["reverse"],
        shuffle_loss=mean_control_losses["shuffle"],
        phase_gates=prototype.phase_gates,
        block_gates=prototype.block_head_gates[:, :, 0],
        support_codes=support_tied_codes,
    )
    teacher_go = teacher_objective.evaluate_teacher_go(held_statistics)
    representability = representability_decision(
        probes, teacher_go, support_iids=support_iids
    )
    teacher_go_receipt = {
        "go": teacher_go.go,
        "checks": dict(teacher_go.checks),
        "failed_checks": list(teacher_go.failed_checks),
        "thresholds": dict(teacher_go.thresholds),
        "statistics": held_statistics.as_dict(),
        "loss_aggregation": (
            "sort_by_iid_then_arithmetic_mean_across_two_support_held_noise_cells_"
            "before_ratios"
        ),
        "loss_aggregation_iids": [
            str(row["iid"]) for row in held_controls_for_aggregation
        ],
    }
    decision_digest = _world_digest_consensus(
        {"teacher_go": teacher_go_receipt, "representability": representability},
        world_group=parallel.world_group,
        context="teacher GO evidence",
    )

    # This flag becomes true before any optional heldout parquet deserialization.
    prototype_frozen = True
    posthoc: Optional[dict[str, Any]] = None
    if args.posthoc_heldout_eval:
        if not prototype_frozen:
            raise FewShotCodeTrainingError("heldout target opened before prototype freeze")
        raw, parquet_receipt = _read_episode_parquet(episode.heldout)
        heldout_cell = _make_training_cell(
            episode_row=episode.heldout,
            raw_row=raw,
            parquet_receipt=parquet_receipt,
            renderer=renderer,
            transformer=transformer,
            tokenizer=tokenizer,
            rope=rope,
            vae_mean=vae_mean,
            vae_std=vae_std,
            z_dim=z_dim,
            scheduler=scheduler,
            process_renderer_sample=process_renderer_sample,
            prompt_cleaner=prompt_clean,
            device=device,
            noise_seed=_seed_for_iid(args.seed, episode.heldout.iid, "posthoc"),
            sigma_index=args.held_sigma_index,
            proposal_seed=args.proposal_seed,
            ulysses_group=parallel.ulysses_group,
        )
        posthoc = _evaluate_controls(
            renderer, heldout_cell, prototype, patch_handle=patch_handle
        )
        posthoc.update(
            {
                "posthoc_only": True,
                "target_privileged": True,
                "affected_optimizer": False,
                "affected_prototype": False,
                "affected_representability_gate": False,
            }
        )
        _world_digest_consensus(
            posthoc,
            world_group=parallel.world_group,
            context="posthoc heldout controls",
        )

    if contract.global_rank == 0:
        diagnostic_path = output_dir / "diagnostics.pt"
        _atomic_torch_save(
            diagnostic_path,
            {
                "schema_version": DIAGNOSTIC_STATE_SCHEMA,
                "engineering_smoke": False,
                "support_logits_and_history": support_diagnostics,
                "teacher_go": teacher_go_receipt,
                "posthoc": posthoc,
            },
        )
        go_receipt: dict[str, Any] = {
            "schema_version": GO_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "representability_gate": representability,
            "support_iids": [row.iid for row in episode.supports],
            "support_tied_36d_code_sha256": [
                item["tied_36d_code_sha256"] for item in support_receipts
            ],
            "gradient_probes": probes,
            "held_noise_controls": held_controls,
            "teacher_go": teacher_go_receipt,
            "teacher_objective_contract": teacher_objective.objective_contract(),
            "decision_rule": (
                "support-1 phase/block reference probes pass; mean held-noise improvement is "
                ">=15% vs zero and >=5% vs reverse/shuffle; unique-36D gate "
                "saturation is <25%; support-code cosine is >=0.60"
            ),
            "reference_gradient_probe_support_index": 1,
            "single_step_apg_surrogate": True,
            "full_trajectory_render": False,
            "video_quality_claim": False,
            "full_target_flow_matching_weight": FULL_TARGET_FM_WEIGHT,
            "distributed": distributed_receipt,
            "k2_exchange_consensus": {
                "code": code_exchange_digest,
                "receipt": receipt_exchange_digest,
                "diagnostic": diagnostic_exchange_digest,
                "probe": probe_exchange_digest,
                "held_control": held_control_exchange_digest,
                "prototype": prototype_consensus_digest,
                "decision": decision_digest,
            },
        }
        go_receipt["receipt_sha256"] = object_sha256(go_receipt)
        go_path = output_dir / "training_go_receipt.json"
        _atomic_write_json(go_path, go_receipt)
        prototype_path = output_dir / "prototype.safetensors"
        prototype_artifact = _atomic_save_prototype(prototype_path, prototype)
        diagnostic_sha = file_sha256(diagnostic_path)
        import infer_fewshot_motion_code as fewshot_inference

        if (
            fewshot_inference.PROTOTYPE_STATE_SCHEMA != PROTOTYPE_STATE_SCHEMA
            or fewshot_inference.PROTOTYPE_RECEIPT_SCHEMA
            != PROTOTYPE_RECEIPT_SCHEMA
        ):
            raise FewShotCodeTrainingError("training/inference prototype schemas differ")
        tied_prototype = _tied_code_36d(prototype).cpu()
        companion = fewshot_inference.build_prototype_training_receipt(
            state_filename=prototype_path.name,
            state_file_sha256=prototype_artifact["file_sha256"],
            motion_code=epmc.MotionCode(
                prototype.phase_gates.detach().cpu().float().contiguous(),
                prototype.block_head_gates.detach().cpu().float().contiguous(),
            ),
            tied_code_36d=tied_prototype,
            support_tied_code_36d_sha256=[
                item["tied_36d_code_sha256"] for item in support_receipts
            ],
            training_gate_receipt_sha256=file_sha256(go_path),
            representability_gate=representability,
        )
        companion_path = output_dir / "prototype.receipt.json"
        _atomic_write_json(companion_path, companion)
        run_receipt: dict[str, Any] = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "engineering_smoke": False,
            "representability_gate": representability,
            "prototype_frozen_before_heldout_deserialization": True,
            "heldout_pre_run_audit": (
                "content-opaque hash and media-metadata audit only"
            ),
            "heldout_target_latent_deserialized": bool(args.posthoc_heldout_eval),
            "posthoc_heldout": posthoc,
            "support": support_receipts,
            "episode_audit": episode.audit_receipt(),
            "prototype_companion_file_sha256": file_sha256(companion_path),
            "training_go_receipt_file_sha256": file_sha256(go_path),
            "diagnostics_file_sha256": diagnostic_sha,
            "distributed": distributed_receipt,
            "k2_exchange_consensus": {
                "code": code_exchange_digest,
                "receipt": receipt_exchange_digest,
                "diagnostic": diagnostic_exchange_digest,
                "probe": probe_exchange_digest,
                "held_control": held_control_exchange_digest,
                "prototype": prototype_consensus_digest,
                "decision": decision_digest,
            },
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                "base_frozen": True,
                "cpmr_clone_frozen": True,
                "gradient_checkpointing": False,
                "trainable_dimension_per_support": TRAINABLE_CODE_DIMENSION,
            },
            "runtime_versions": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
            },
            "objective": {
                "name": "momentum_zero_apg_clean_field_fewshot_teacher_objective",
                "single_step_apg_surrogate": True,
                "full_trajectory_render": False,
                "full_target_flow_matching_weight": FULL_TARGET_FM_WEIGHT,
                "negative_prompt_used": True,
                "conditional_prompt": "semantic_noop_plus_source_proposal_epmc",
                "teacher_contract": teacher_objective.objective_contract(),
            },
            "inference_conditions": ["source_video", "edit_instruction"],
            "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_CONDITIONS),
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
        }
        run_receipt["receipt_sha256"] = object_sha256(run_receipt)
        _atomic_write_json(output_dir / "receipt.json", run_receipt)
        print(canonical_json_bytes(run_receipt).decode("utf-8"), flush=True)

    dist.barrier(group=parallel.world_group)
    patch_handle.restore()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

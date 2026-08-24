#!/usr/bin/env python3
"""WORLD4 generic source-anchored action trainer.

The first executable closure in this file is the approved real-source carrier
segment: ``smoke-r`` performs one disposable update and ``stage-r64`` performs
the exact 64-row Stage R pass.  Stage R reuses the audited physical index-0
source posterior store and Bernini's native ``[source; noisy target]`` packed
forward.  The carrier memory and target use the exact same Gaussian and sigma.

The R64 checkpoint also seals the still-untrained planner/operator bytes and a
single AdamW state so a later ``resume-po40`` implementation can be bound to
the exact checkpoint.  It is deliberately marked ``complete_action_result =
false``.  This runner never reads generated video, latent, noise, velocity, or
action-family IDs.
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
import sys
import time
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_adapter_v1 as visual  # noqa: E402
import clean_source_visual_context_stage_b_contract_v1 as old_contract  # noqa: E402
import clean_source_visual_context_training_v1 as source_data  # noqa: E402
import generic_source_anchored_action_v1 as core  # noqa: E402
import inference_sigma_strata as exact40  # noqa: E402
import train_clean_source_visual_context_stage_b_v1 as native_r  # noqa: E402


METHOD = "bernini-generic-source-anchored-action-v1"
RUN_RECEIPT_SCHEMA = core.TRAINING_RECEIPT_SCHEMA
CHECKPOINT_SCHEMA = "bernini-generic-source-anchored-action-checkpoint-v1"
HISTORY_SCHEMA = "bernini-generic-source-anchored-action-history-v1"
EXECUTION_PROFILES = (
    "smoke-r",
    "stage-r64",
    "smoke-p",
    "smoke-o",
    "resume-po40",
    "action-only40",
)
IMPLEMENTED_EXECUTION_PROFILES = ("smoke-r", "stage-r64")
R_CHECKPOINT_NAME = "stage_r_composite_checkpoint.pt"
R_MIDPOINT_CHECKPOINT_NAME = "stage_r_u032_composite_checkpoint.pt"
SMOKE_CHECKPOINT_NAME = "disposable_smoke_r_checkpoint.pt"
EXPECTED_VEOMNI_COMMIT = native_r.EXPECTED_VEOMNI_COMMIT
EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    native_r.EXPECTED_CHECKPOINT_MANIFEST_SHA256
)


class GenericSourceAnchoredTrainingError(RuntimeError):
    """Raised before an ambiguous update or incomplete artifact publication."""


def fail(message: str) -> NoReturn:
    raise GenericSourceAnchoredTrainingError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GenericSourceAnchoredTrainingError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class RCoordinate:
    optimizer_step_zero_based: int
    schedule_index: int
    timestep: int
    sigma: float
    sigma_float32_be_hex: str

    def receipt(self) -> Mapping[str, Any]:
        return {
            "optimizer_step_zero_based": self.optimizer_step_zero_based,
            "optimizer_step_one_based": self.optimizer_step_zero_based + 1,
            "schedule_index": self.schedule_index,
            "timestep_int64": self.timestep,
            "sigma": self.sigma,
            "sigma_float32_be_hex": self.sigma_float32_be_hex,
        }


def r_coordinates(profile: str) -> tuple[RCoordinate, ...]:
    if profile not in IMPLEMENTED_EXECUTION_PROFILES:
        fail("execution profile does not have an implemented runtime")
    schedule = core.fixed_sigma_schedule("R")
    if profile == "smoke-r":
        schedule = schedule[:1]
    return tuple(
        RCoordinate(
            optimizer_step_zero_based=position,
            schedule_index=index,
            timestep=exact40.PINNED_TIMESTEPS[index],
            sigma=exact40.PINNED_POSITIVE_SIGMAS[index],
            sigma_float32_be_hex=exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index],
        )
        for position, index in enumerate(schedule)
    )


def _noise_seed(base_seed: int, row_sha256: str, schedule_index: int) -> int:
    _require_sha256(row_sha256, label="source row canonical SHA")
    payload = (
        f"{base_seed}\0generic-source-anchored-action-v1\0stage-r\0"
        f"{row_sha256}\0{schedule_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach().contiguous().reshape(-1)
    return value.view(torch.uint8).cpu().numpy().tobytes(order="C")


def named_parameters_sha256(named: Sequence[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in named:
        tensor = parameter.detach().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _parameter_consensus(
    named: Sequence[tuple[str, Any]], *, world_group: Any, label: str
) -> str:
    import torch.distributed as dist

    digest = named_parameters_sha256(named)
    gathered: list[Any] = [None] * core.WORLD_SIZE
    dist.all_gather_object(gathered, digest, group=world_group)
    if gathered != [digest] * core.WORLD_SIZE:
        fail(f"{label} parameters differ across WORLD4")
    return digest


def _rank0_call(
    *, rank: int, world_group: Any, label: str, callback: Any
) -> Any:
    """Execute one filesystem transaction and propagate failure to all ranks."""

    import torch.distributed as dist

    box: list[Any] = [None]
    if rank == 0:
        try:
            box[0] = {"ok": True, "result": callback()}
        except Exception as error:  # noqa: BLE001 - serialized fail-closed boundary
            box[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(box, src=0, group=world_group)
    result = box[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        fail(f"rank0 {label} failed: {result!r}")
    return result.get("result")


def _carrier_gradient_group_norms(
    active: Sequence[tuple[str, Any]],
) -> Mapping[str, float]:
    if not active or any(not name.startswith("carrier.") for name, _ in active):
        fail("Stage R active parameter names are not carrier-only")
    return native_r.grouped_gradient_norms(
        tuple((name.removeprefix("carrier."), parameter) for name, parameter in active)
    )


def _logical_route_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove only SP-rank-local evidence from a verified route receipt."""

    expected = {
        "total_tokens",
        "condition_tokens",
        "target_tokens",
        "sequence_parallel_rank",
        "sequence_parallel_size",
        "enabled",
        "memory_digest",
        "query_rows",
        "key_value_rows",
        "digest",
    }
    if set(receipt) != expected:
        fail("visual-context route receipt schema differs")
    rank = receipt.get("sequence_parallel_rank")
    size = receipt.get("sequence_parallel_size")
    if (
        type(rank) is not int
        or type(size) is not int
        or size != core.SP_SIZE
        or not 0 <= rank < size
    ):
        fail("visual-context route SP coordinate differs")
    logical = {
        key: value
        for key, value in receipt.items()
        if key not in {"sequence_parallel_rank", "digest"}
    }
    return {**logical, "logical_digest": object_sha256(logical)}


def _broadcast_composite(handle: core.CompositeHandle, world_group: Any) -> Mapping[str, str]:
    import torch.distributed as dist

    result: dict[str, str] = {}
    for stage, rows in handle.named_parameter_groups().items():
        for _, parameter in rows:
            dist.broadcast(parameter.data, src=0, group=world_group)
        result[stage] = _parameter_consensus(
            rows, world_group=world_group, label=f"initial {stage}"
        )
    return result


def _source_row_order(manifest: Any) -> tuple[tuple[str, Any], ...]:
    rows = manifest.rows_for_split("train")
    if len(rows) != core.STAGE_UPDATES["R"]:
        fail("Stage R requires exactly 64 real-source train rows")
    keyed = tuple(
        sorted(
            ((object_sha256(row.receipt()), row) for row in rows),
            key=lambda item: item[0],
        )
    )
    if len({sha for sha, _ in keyed}) != len(keyed):
        fail("Stage R source row canonical SHA is not unique")
    return keyed


def _read_decimal_file(path: Path) -> Optional[int]:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return int(value) if value.isdecimal() else None


def _cgroup_memory_receipt() -> Mapping[str, float]:
    """Read the live process cgroup's current/peak memory, fail-closed."""

    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise GenericSourceAnchoredTrainingError(
            f"cannot read process cgroup: {error}"
        ) from error
    unified = None
    memory_v1 = None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, relative = fields
        if hierarchy == "0" and controllers == "":
            unified = Path("/sys/fs/cgroup") / relative.lstrip("/")
        elif "memory" in controllers.split(","):
            memory_v1 = Path("/sys/fs/cgroup/memory") / relative.lstrip("/")
    current: Optional[int] = None
    peak: Optional[int] = None
    if unified is not None:
        current = _read_decimal_file(unified / "memory.current")
        peak = _read_decimal_file(unified / "memory.peak")
    if current is None and memory_v1 is not None:
        current = _read_decimal_file(memory_v1 / "memory.usage_in_bytes")
        peak = _read_decimal_file(memory_v1 / "memory.max_usage_in_bytes")
    if current is None:
        fail("live cgroup memory.current is unavailable")
    if peak is None:
        peak = current
    gib = float(1024**3)
    return {
        "current_gib": float(current) / gib,
        "peak_gib": float(peak) / gib,
    }


def _local_resource(stage: str, rank: int, device: Any) -> Mapping[str, Any]:
    import torch

    torch.cuda.synchronize(device)
    gib = float(1024**3)
    rss = native_r.linux_host_memory_receipt()
    cgroup = _cgroup_memory_receipt()
    return {
        "stage": stage,
        "rank": rank,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "gpu_allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "gpu_reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
        "host_current_rss_gib": rss["current_rss_gib"],
        "host_peak_rss_gib": rss["peak_rss_gib"],
        "host_cgroup_current_gib": cgroup["current_gib"],
        "host_cgroup_peak_gib": cgroup["peak_gib"],
    }


def collect_and_gate_resources(
    *,
    stage: str,
    rank: int,
    device: Any,
    world_group: Any,
    gpu_limit_gib: float,
    host_limit_gib: float,
) -> tuple[Mapping[str, Any], ...]:
    import torch.distributed as dist

    local = _local_resource(stage, rank, device)
    gathered: list[Any] = [None] * core.WORLD_SIZE
    dist.all_gather_object(gathered, local, group=world_group)
    if [item.get("rank") for item in gathered] != list(range(core.WORLD_SIZE)):
        fail(f"{stage} resource WORLD4 closure differs")
    if any(
        float(item["gpu_peak_reserved_gib"]) >= gpu_limit_gib
        or float(item["host_peak_rss_gib"]) >= host_limit_gib
        or float(item["host_cgroup_current_gib"]) >= host_limit_gib
        or float(item["host_cgroup_peak_gib"]) >= host_limit_gib
        for item in gathered
    ):
        fail(
            f"{stage} crossed strict GPU<{gpu_limit_gib}GiB or "
            f"host<{host_limit_gib}GiB memory gate"
        )
    return tuple(dict(item) for item in gathered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=core.EXPERIMENTS, required=True)
    parser.add_argument(
        "--execution-profile", choices=EXECUTION_PROFILES, required=True
    )
    parser.add_argument("--parallel-topology", choices=(core.TOPOLOGY,), required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--source-manifest", "--source-only-manifest", dest="source_manifest", required=True
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        "--expected-source-only-manifest-sha256",
        dest="expected_source_manifest_sha256",
        default=core.EXPECTED_SOURCE_ONLY_MANIFEST_SHA256,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--learning-rate", type=float, default=core.DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=core.DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=core.DEFAULT_SEED)
    parser.add_argument(
        "--gpu-memory-limit-gib", type=float, default=core.GPU_MEMORY_LIMIT_GIB
    )
    parser.add_argument(
        "--host-memory-limit-gib", type=float, default=core.HOST_MEMORY_LIMIT_GIB
    )
    parser.add_argument(
        "--expected-bernini-commit",
        default=visual.PINNED_BERNINI_SOURCE_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit", default=EXPECTED_VEOMNI_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=old_contract.EXPECTED_CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument("--ack-upstream-training-use-forbidden", action="store_true")
    parser.add_argument("--ack-user-authorized-exploratory-training", action="store_true")
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.execution_profile not in IMPLEMENTED_EXECUTION_PROFILES:
        fail(
            f"execution profile {args.execution_profile!r} is registered but not "
            "implemented in this source revision"
        )
    if args.experiment != "joint_source_anchored_v1":
        fail("Stage R profiles require joint_source_anchored_v1")
    if (
        args.parallel_topology != core.TOPOLOGY
        or args.learning_rate != core.DEFAULT_LEARNING_RATE
        or args.max_grad_norm != core.DEFAULT_MAX_GRAD_NORM
        or args.gpu_memory_limit_gib != core.GPU_MEMORY_LIMIT_GIB
        or args.host_memory_limit_gib != core.HOST_MEMORY_LIMIT_GIB
    ):
        fail("topology/optimizer/resource limits are fixed by the reviewed contract")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        fail("seed must lie in [0,2^63)")
    if (
        args.ack_upstream_training_use_forbidden is not True
        or args.ack_user_authorized_exploratory_training is not True
    ):
        fail("both exploratory source-data acknowledgements are required")
    for name in (
        "expected_source_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_checkpoint_content_manifest_sha256",
    ):
        _require_sha256(getattr(args, name), label=name)
    if (
        args.expected_source_manifest_sha256
        != core.EXPECTED_SOURCE_ONLY_MANIFEST_SHA256
        or args.expected_bernini_commit != visual.PINNED_BERNINI_SOURCE_COMMIT
        or args.expected_veomni_commit != EXPECTED_VEOMNI_COMMIT
        or args.expected_checkpoint_tree_sha256
        != old_contract.EXPECTED_CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        fail("pinned source/model/runtime identity differs")
    output = Path(args.output).expanduser()
    if (
        not output.is_absolute()
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
    ):
        fail("output must be a fresh absolute directory")


def _component_cpu_state(
    handle: core.CompositeHandle,
) -> Mapping[str, Mapping[str, Any]]:
    return {
        stage: {
            name: parameter.detach().float().cpu().contiguous().clone()
            for name, parameter in rows
        }
        for stage, rows in handle.named_parameter_groups().items()
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch
    import torch.distributed as dist
    import source_self_runtime as runtime
    import train_lora as legacy

    args = build_parser().parse_args(argv)
    validate_cli(args)
    source_manifest_path = Path(args.source_manifest).expanduser()
    checkpoint_manifest_path = Path(args.checkpoint_content_manifest).expanduser()
    if (
        not source_manifest_path.is_absolute()
        or file_sha256(source_manifest_path) != args.expected_source_manifest_sha256
    ):
        fail("source-only manifest path/SHA differs")
    if (
        not checkpoint_manifest_path.is_absolute()
        or file_sha256(checkpoint_manifest_path)
        != args.expected_checkpoint_content_manifest_sha256
    ):
        fail("checkpoint content manifest path/SHA differs")
    manifest = source_data.load_source_only_split_manifest(
        source_manifest_path, verify_files=True
    )
    dataset_authorization = source_data.authorize_exploratory_training(
        manifest,
        ack_upstream_training_use_forbidden=args.ack_upstream_training_use_forbidden,
        ack_user_authorized_exploratory_training=(
            args.ack_user_authorized_exploratory_training
        ),
    )
    keyed_rows = _source_row_order(manifest)
    source_row_order_sha256 = object_sha256([sha for sha, _ in keyed_rows])
    coordinates = r_coordinates(args.execution_profile)

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
        raise GenericSourceAnchoredTrainingError(str(error)) from error
    if (
        transformer_config.get("num_attention_heads") != 12
        or legacy.CHECKPOINT_TREE_SHA256 != args.expected_checkpoint_tree_sha256
    ):
        fail("pinned Bernini transformer/checkpoint differs")
    packed_sp_audit = native_r.audit_packed_sp_sources(bernini_root, veomni_root)
    legacy.activate_source_trees(bernini_root, veomni_root)

    from diffusers import UniPCMultistepScheduler, __version__ as diffusers_version
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state

    topology = runtime.parallel_topology(core.TOPOLOGY)
    distributed = runtime.distributed_contract(topology=topology)
    if (
        distributed.world_size != core.WORLD_SIZE
        or distributed.topology.dp_size != core.DP_SIZE
        or distributed.topology.sp_size != core.SP_SIZE
        or distributed.arm_index != 0
    ):
        fail("trainer requires one shared WORLD4 DP1xSP4 model")
    device = runtime.initialise_distributed(distributed)
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=core.SP_SIZE)
    )
    torch.cuda.reset_peak_memory_stats(device)

    checkpoint_content_box: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_content_box[0] = {
                "ok": True,
                "identity": native_r.validate_checkpoint_content(
                    checkpoint,
                    checkpoint_manifest_path,
                    expected_manifest_sha256=(
                        args.expected_checkpoint_content_manifest_sha256
                    ),
                ),
            }
        except Exception as error:
            checkpoint_content_box[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_content_box, src=0)
    checkpoint_result = checkpoint_content_box[0]
    if (
        not isinstance(checkpoint_result, Mapping)
        or checkpoint_result.get("ok") is not True
        or not isinstance(checkpoint_result.get("identity"), Mapping)
    ):
        fail(f"checkpoint content validation failed: {checkpoint_result!r}")
    checkpoint_content_identity = dict(checkpoint_result["identity"])

    legacy.seed_same_sample(args.seed)
    renderer_config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    renderer_config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(renderer_config.to_dict(), checkpoint)
    renderer = None
    for loading_rank in range(core.WORLD_SIZE):
        if distributed.rank == loading_rank:
            renderer = BerniniRendererModel(renderer_config)
            renderer.requires_grad_(False)
            renderer.eval()
            renderer.t5_text_encoder.eval()
            renderer.to(device)
        dist.barrier(group=parallel.world_group)
    if renderer is None:
        fail("rank-serialized renderer load failed")
    transformer = renderer.diff_dec.transformer
    if transformer is None or renderer.diff_dec.transformer_2 is not None:
        fail("generic trainer requires only frozen transformer_1")
    base_transformer_named = tuple(transformer.named_parameters())
    if not base_transformer_named or any(
        parameter.requires_grad for _, parameter in base_transformer_named
    ):
        fail("loaded base transformer is absent or not frozen")
    base_transformer_sha256 = _parameter_consensus(
        base_transformer_named,
        world_group=parallel.world_group,
        label="loaded frozen base transformer",
    )
    renderer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": visual.checkpoint_route_context_fn,
        }
    )
    if not bool(getattr(transformer, "gradient_checkpointing", False)):
        fail("Stage R requires non-reentrant carrier route checkpoint replay")
    composite = core.install_composite_v1(
        transformer,
        experiment=args.experiment,
        runtime_source_commit=bernini_revision,
        model_revision=visual.PINNED_BERNINI_MODEL_REVISION,
        checkpoint_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        initialization_seed=args.seed,
    )
    if composite.carrier is None:
        fail("joint Stage R did not install the source carrier")
    initial_component_sha256 = _broadcast_composite(
        composite, parallel.world_group
    )
    parameter_counts = composite.parameter_count_receipt()
    optimizer_controller = core.StageOptimizerController(
        composite, learning_rate=args.learning_rate
    )
    active = optimizer_controller.activate("R")
    optimizer = optimizer_controller.optimizer

    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(checkpoint),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=5.0,
    )
    schedule_audit = exact40.audit_runtime_unipc_schedule(scheduler)
    if schedule_audit.get("schedule_sha256") != old_contract.EXPECTED_SCHEDULE_SHA256:
        fail("runtime UniPC exact40 schedule differs")
    del scheduler

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    tokenized = runtime.tokenize_generic_instruction(
        tokenizer, core.EXACT_NOOP_INSTRUCTION, device
    )
    with torch.inference_mode():
        text_lens, text_embs = renderer.get_t5_text_embeddings(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            tokenized["t5_input_lens"],
        )
    if text_embs.requires_grad:
        fail("frozen no-op UMT5 states require gradients")
    renderer.t5_text_encoder = None
    del tokenizer, tokenized
    gc.collect()
    torch.cuda.empty_cache()

    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    if z_dim != 16:
        fail("Wan VAE latent dimension differs")
    store = source_data.PinnedPhysicalSourceOnlyPosteriorStore(
        manifest,
        vae_latents_mean=vae_mean.unsqueeze(0).float().contiguous(),
        vae_latents_std=vae_std.unsqueeze(0).float().contiguous(),
        verify_files_on_first_access=True,
    )
    index_by_iid = {row.iid: index for index, row in enumerate(manifest.rows)}
    selected_rows = keyed_rows[: len(coordinates)]
    preload = store.preload(tuple(index_by_iid[row.iid] for _, row in selected_rows))
    if (
        preload.get("preloaded_rows") != len(coordinates)
        or preload.get("legacy_parquet_opened") is not False
        or preload.get("synthetic_target_index1_bytes_read") is not False
    ):
        fail("source-only physical preload differs")
    runtime.digest_consensus(
        str(preload["digest"]),
        group=parallel.world_group,
        expected_count=core.WORLD_SIZE,
        label="source-only physical preload",
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)

    output = Path(args.output)

    def create_private_output() -> bool:
        os.mkdir(output, mode=0o700)
        os.chmod(output, 0o700)
        if output.stat().st_mode & 0o777 != 0o700:
            fail("output directory mode is not exact 0700")
        return True

    _rank0_call(
        rank=distributed.rank,
        world_group=parallel.world_group,
        label="private output creation",
        callback=create_private_output,
    )
    resource_milestones: dict[str, tuple[Mapping[str, Any], ...]] = {}
    resource_milestones["model_load"] = collect_and_gate_resources(
        stage="model_load",
        rank=distributed.rank,
        device=device,
        world_group=parallel.world_group,
        gpu_limit_gib=args.gpu_memory_limit_gib,
        host_limit_gib=args.host_memory_limit_gib,
    )

    history: list[Mapping[str, Any]] = []
    intermediate_checkpoints: list[Mapping[str, Any]] = []
    previous_carrier_sha = initial_component_sha256["R"]
    started = time.monotonic()
    for step_zero_based, (coordinate, selected) in enumerate(
        zip(coordinates, selected_rows)
    ):
        row_sha, row = selected
        sample = store.load(index_by_iid[row.iid])
        seed = _noise_seed(args.seed, row_sha, coordinate.schedule_index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        epsilon = torch.randn(
            tuple(sample.clean_noop_target.shape),
            generator=generator,
            dtype=torch.float32,
        ).contiguous()
        packed = native_r.prepare_noop_condition(
            clean_source=sample.clean_noop_target,
            epsilon=epsilon,
            coordinate=coordinate,
            memory_input_kind="same_noise_forward_noised_source",
            rope=rope,
            device=device,
            runtime=runtime,
        )
        memory_input = packed.memory_input.to(
            device=device, dtype=torch.float32
        ).detach().contiguous()
        memory = composite.carrier.build_memory(
            memory_input,
            source_video_sha256=sample.source_video_sha256,
            memory_input_latent_sha256=packed.tensor_identities[
                "visual_context_input"
            ],
            input_kind="same_noise_forward_noised_source",
        )
        route = visual.VisualContextRoute(
            packed.total_tokens,
            packed.condition_tokens,
            distributed.sp_rank,
            core.SP_SIZE,
            memory,
        )
        inactive_before = core.frozen_inactive_snapshot(composite, "R")
        optimizer.zero_grad(set_to_none=True)
        with composite.carrier.route(route):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = native_r._prediction(
                    renderer=renderer,
                    transformer=transformer,
                    condition=packed,
                    coordinate=coordinate,
                    text_lens=text_lens,
                    text_embs=text_embs,
                )
                prediction = native_r._zero_dependency(prediction, active)
                loss = visual.no_op_flow_matching_loss(
                    prediction=prediction,
                    target_velocity=packed.target_velocity,
                )
        if step_zero_based == 0:
            resource_milestones["first_forward"] = collect_and_gate_resources(
                stage="first_forward",
                rank=distributed.rank,
                device=device,
                world_group=parallel.world_group,
                gpu_limit_gib=args.gpu_memory_limit_gib,
                host_limit_gib=args.host_memory_limit_gib,
            )
        if not runtime.world_all_true(
            bool(torch.isfinite(loss.detach()).item()), group=parallel.world_group
        ):
            fail("Stage R produced non-finite no-op flow loss")
        loss_value = float(loss.detach().item())
        loss.backward()
        preclip_norm = runtime.synchronize_gradients(active, parallel)
        component_gradient_norms = _carrier_gradient_group_norms(active)
        if component_gradient_norms["output"] <= 0.0:
            fail(
                "same-noise no-op objective produced no real carrier output gradient"
            )
        if step_zero_based >= 1 and any(
            value <= 0.0 for value in component_gradient_norms.values()
        ):
            fail(
                "same-noise carrier encoder/Q/K/V/output/gate path is inactive"
            )
        if step_zero_based == 0:
            resource_milestones["first_backward"] = collect_and_gate_resources(
                stage="first_backward",
                rank=distributed.rank,
                device=device,
                world_group=parallel.world_group,
                gpu_limit_gib=args.gpu_memory_limit_gib,
                host_limit_gib=args.host_memory_limit_gib,
            )
        clipped = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in active], args.max_grad_norm
        )
        if not math.isfinite(float(clipped)) or preclip_norm <= 0.0:
            fail("Stage R synchronized gradient norm differs")
        optimizer.step()
        optimizer_controller.assert_inactive_unchanged(inactive_before)
        carrier_sha = _parameter_consensus(
            active,
            world_group=parallel.world_group,
            label=f"Stage R update {step_zero_based + 1}",
        )
        if carrier_sha == previous_carrier_sha:
            fail("Stage R optimizer update did not change carrier parameters")
        previous_carrier_sha = carrier_sha
        if step_zero_based == 0:
            resource_milestones["first_optimizer_step"] = collect_and_gate_resources(
                stage="first_optimizer_step",
                rank=distributed.rank,
                device=device,
                world_group=parallel.world_group,
                gpu_limit_gib=args.gpu_memory_limit_gib,
                host_limit_gib=args.host_memory_limit_gib,
            )
        local_step = {
            "rank": distributed.rank,
            "sp_rank": distributed.sp_rank,
            "iid": sample.iid,
            "source_video_sha256": sample.source_video_sha256,
            "source_row_canonical_sha256": row_sha,
            "noise_seed": seed,
            **coordinate.receipt(),
            "memory_input_kind": "same_noise_forward_noised_source",
            "tensor_identities": dict(packed.tensor_identities),
            "memory_receipt": dict(memory.receipt()),
            "route_receipt": dict(route.receipt()),
            "logical_route_receipt": _logical_route_receipt(route.receipt()),
            "loss": loss_value,
            "preclip_gradient_norm_sp4_mean": preclip_norm,
            "component_gradient_norms": dict(component_gradient_norms),
            "real_output_gradient_positive": True,
            "carrier_parameter_sha256_after": carrier_sha,
        }
        gathered_step: list[Any] = [None] * core.WORLD_SIZE
        dist.all_gather_object(gathered_step, local_step, group=parallel.world_group)
        projection = []
        for item in gathered_step:
            projected = {
                key: value
                for key, value in item.items()
                if key not in {"rank", "sp_rank", "route_receipt"}
            }
            projection.append(projected)
        if any(item != projection[0] for item in projection[1:]):
            fail("SP4 ranks did not execute the same logical Stage R row")
        if [
            item["route_receipt"]["sequence_parallel_rank"]
            for item in gathered_step
        ] != list(range(core.SP_SIZE)):
            fail("SP4 route receipt rank order differs")
        history.append(
            {
                "schema_version": HISTORY_SCHEMA,
                **projection[0],
                "route_receipts_by_rank": [
                    dict(item["route_receipt"]) for item in gathered_step
                ],
                "world4_same_logical_row": True,
                "optimizer_step_executed": True,
            }
        )
        if args.execution_profile == "stage-r64" and step_zero_based + 1 == 32:
            midpoint_value = {
                "schema_version": CHECKPOINT_SCHEMA,
                "method": METHOD,
                "experiment": args.experiment,
                "execution_profile": "stage-r64",
                "completed_stages": [],
                "in_progress_stage": "R",
                "completed_stage_updates": {"R": 32},
                "resume_po40_authorized": False,
                "complete_action_result": False,
                "initial_component_sha256": initial_component_sha256,
                "current_carrier_sha256": carrier_sha,
                "component_state": _component_cpu_state(composite),
                "optimizer_state": optimizer.state_dict(),
                "source_manifest_digest": manifest.manifest_digest,
                "source_row_order_sha256": source_row_order_sha256,
                "history_sha256": object_sha256(history),
            }
            midpoint_path = output / R_MIDPOINT_CHECKPOINT_NAME

            def write_midpoint() -> str:
                runtime.atomic_torch_save(midpoint_path, midpoint_value)
                return file_sha256(midpoint_path)

            midpoint_sha256 = _rank0_call(
                rank=distributed.rank,
                world_group=parallel.world_group,
                label="Stage R u032 checkpoint write",
                callback=write_midpoint,
            )
            intermediate_checkpoints.append(
                {
                    "stage": "R",
                    "stage_update": 32,
                    "path": str(midpoint_path),
                    "file_sha256": midpoint_sha256,
                    "resume_po40_authorized": False,
                }
            )
        del (
            sample,
            epsilon,
            packed,
            memory_input,
            memory,
            route,
            prediction,
            loss,
            local_step,
            gathered_step,
            projection,
            inactive_before,
        )
        gc.collect()

    resource_milestones["final"] = collect_and_gate_resources(
        stage="final",
        rank=distributed.rank,
        device=device,
        world_group=parallel.world_group,
        gpu_limit_gib=args.gpu_memory_limit_gib,
        host_limit_gib=args.host_memory_limit_gib,
    )
    elapsed = time.monotonic() - started
    final_component_sha256 = {
        stage: _parameter_consensus(
            rows, world_group=parallel.world_group, label=f"final {stage}"
        )
        for stage, rows in composite.named_parameter_groups().items()
    }
    if (
        final_component_sha256["P"] != initial_component_sha256["P"]
        or final_component_sha256["O"] != initial_component_sha256["O"]
        or final_component_sha256["R"] == initial_component_sha256["R"]
    ):
        fail("Stage R parameter-scope closure differs")
    terminal_base_transformer_sha256 = _parameter_consensus(
        base_transformer_named,
        world_group=parallel.world_group,
        label="terminal frozen base transformer",
    )
    if terminal_base_transformer_sha256 != base_transformer_sha256:
        fail("frozen base transformer changed during Stage R")

    def terminal_toctou_audit() -> Mapping[str, Any]:
        if file_sha256(source_manifest_path) != args.expected_source_manifest_sha256:
            fail("source manifest changed during Stage R")
        if (
            file_sha256(checkpoint_manifest_path)
            != args.expected_checkpoint_content_manifest_sha256
        ):
            fail("checkpoint manifest changed during Stage R")
        terminal_checkpoint_content_identity = native_r.validate_checkpoint_content(
            checkpoint,
            checkpoint_manifest_path,
            expected_manifest_sha256=(
                args.expected_checkpoint_content_manifest_sha256
            ),
        )
        if dict(terminal_checkpoint_content_identity) != checkpoint_content_identity:
            fail("checkpoint content changed during Stage R")
        try:
            end_bernini, end_veomni, end_bernini_revision, end_veomni_revision = (
                legacy.validate_source_trees(
                    args.bernini_root,
                    args.veomni_root,
                    expected_bernini_commit=args.expected_bernini_commit,
                    expected_veomni_commit=args.expected_veomni_commit,
                )
            )
        except legacy.TrainingContractError as error:
            raise GenericSourceAnchoredTrainingError(str(error)) from error
        end_packed_sp_audit = native_r.audit_packed_sp_sources(
            end_bernini, end_veomni
        )
        if (
            end_bernini != bernini_root
            or end_veomni != veomni_root
            or end_bernini_revision != bernini_revision
            or end_veomni_revision != veomni_revision
            or end_packed_sp_audit != packed_sp_audit
        ):
            fail("model/runtime source trees changed during Stage R")
        return {
            "source_manifest_sha256_reverified": args.expected_source_manifest_sha256,
            "checkpoint_manifest_sha256_reverified": (
                args.expected_checkpoint_content_manifest_sha256
            ),
            "checkpoint_content_identity_reverified": object_sha256(
                terminal_checkpoint_content_identity
            ),
            "bernini_commit_reverified": end_bernini_revision,
            "veomni_commit_reverified": end_veomni_revision,
            "packed_sp_audit_digest_reverified": end_packed_sp_audit["digest"],
            "unchanged": True,
        }

    terminal_audit = _rank0_call(
        rank=distributed.rank,
        world_group=parallel.world_group,
        label="terminal TOCTOU audit",
        callback=terminal_toctou_audit,
    )

    pair_invariants = {
        "status": "partial_r_only_action_manifest_fields_deferred",
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_content_manifest_sha256": (
            args.expected_checkpoint_content_manifest_sha256
        ),
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "representation_manifest_sha256": None,
        "source_pair_manifest_sha256": None,
        "action_row_order_sha256": None,
        "stage_r_source_row_order_sha256": source_row_order_sha256,
        "gaussian_base_seed": args.seed,
        "gaussian_seed_derivation": (
            "sha256(base_seed,method,stage_r,row_sha,schedule_index)"
        ),
        "r_sigma_mapping": [coordinate.schedule_index for coordinate in coordinates],
        "o_sigma_mapping": list(core.fixed_sigma_schedule("O")),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "max_grad_norm": args.max_grad_norm,
        },
        "planner_initial_sha256": initial_component_sha256["P"],
        "operator_initial_sha256": initial_component_sha256["O"],
    }
    checkpoint_value = {
        "schema_version": CHECKPOINT_SCHEMA,
        "method": METHOD,
        "experiment": args.experiment,
        "execution_profile": args.execution_profile,
        "completed_stages": (
            ["R"] if args.execution_profile == "stage-r64" else []
        ),
        "incomplete_stages": (
            [] if args.execution_profile == "stage-r64" else ["R"]
        ),
        "completed_stage_updates": {"R": len(coordinates)},
        "resume_po40_authorized": args.execution_profile == "stage-r64",
        "complete_action_result": False,
        "pair_invariants": pair_invariants,
        "initial_component_sha256": initial_component_sha256,
        "final_component_sha256": final_component_sha256,
        "component_state": _component_cpu_state(composite),
        "optimizer_state": optimizer.state_dict(),
        "source_manifest_digest": manifest.manifest_digest,
        "history_sha256": object_sha256(history),
        "terminal_toctou_audit": terminal_audit,
    }
    checkpoint_name = (
        R_CHECKPOINT_NAME
        if args.execution_profile == "stage-r64"
        else SMOKE_CHECKPOINT_NAME
    )
    checkpoint_path = output / checkpoint_name

    def write_final_checkpoint() -> str:
        runtime.atomic_torch_save(checkpoint_path, checkpoint_value)
        return file_sha256(checkpoint_path)

    checkpoint_file_sha256 = _rank0_call(
        rank=distributed.rank,
        world_group=parallel.world_group,
        label="final Stage R checkpoint write",
        callback=write_final_checkpoint,
    )

    gpu_peak_reserved_by_rank = [
        max(
            float(records[rank]["gpu_peak_reserved_gib"])
            for records in resource_milestones.values()
        )
        for rank in range(core.WORLD_SIZE)
    ]
    host_peak_rss_by_rank = [
        max(
            float(records[rank]["host_peak_rss_gib"])
            for records in resource_milestones.values()
        )
        for rank in range(core.WORLD_SIZE)
    ]
    host_cgroup_peak_by_rank = [
        max(
            float(records[rank]["host_cgroup_peak_gib"])
            for records in resource_milestones.values()
        )
        for rank in range(core.WORLD_SIZE)
    ]
    unsigned_receipt = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "method": METHOD,
        "complete": True,
        "experiment": args.experiment,
        "execution_profile": args.execution_profile,
        "complete_action_result": False,
        "stage_r_complete": args.execution_profile == "stage-r64",
        "stage_r_updates": len(coordinates),
        "planner_updates": 0,
        "operator_updates": 0,
        "resume_po40_authorized": args.execution_profile == "stage-r64",
        "checkpoint": {
            "path": str(checkpoint_path),
            "file_sha256": checkpoint_file_sha256,
            "schema_version": CHECKPOINT_SCHEMA,
        },
        "intermediate_checkpoints": intermediate_checkpoints,
        "contract": core.training_contract_receipt(args.experiment),
        "pair_invariants": pair_invariants,
        "parameter_counts": parameter_counts,
        "initial_component_sha256": initial_component_sha256,
        "final_component_sha256": final_component_sha256,
        "data": {
            **manifest.receipt(),
            "manifest_path": str(source_manifest_path),
            "manifest_file_sha256": args.expected_source_manifest_sha256,
            "source_row_order_sha256": source_row_order_sha256,
            "optimizer_rows_read": len(coordinates),
            "legacy_parquet_opened": False,
            "synthetic_target_index1_bytes_read": False,
            "generated_media_read": False,
            "action_family_used_for_routing": False,
            "authorization": dict(dataset_authorization),
        },
        "objective": {
            "name": "real_source_same_noise_noop_flow_matching",
            "memory_input_kind": "same_noise_forward_noised_source",
            "same_epsilon_and_sigma_target_memory": True,
            "synthetic_target": False,
            "reward": False,
        },
        "history": history,
        "history_sha256": object_sha256(history),
        "resources": {
            "strict_gpu_peak_reserved_limit_gib": args.gpu_memory_limit_gib,
            "strict_host_limit_gib": args.host_memory_limit_gib,
            "gpu_peak_reserved_gib_by_rank": gpu_peak_reserved_by_rank,
            "host_peak_rss_gib_by_rank": host_peak_rss_by_rank,
            "host_cgroup_peak_gib_by_rank": host_cgroup_peak_by_rank,
            "all_strictly_below_limits": True,
            "milestones": {
                key: list(value) for key, value in resource_milestones.items()
            },
        },
        "distributed": {
            "topology": core.TOPOLOGY,
            "world_size": core.WORLD_SIZE,
            "dp_size": core.DP_SIZE,
            "sp_size": core.SP_SIZE,
            "one_shared_model": True,
            "same_logical_row_on_all_ranks": True,
            "rank_action_family_partition": False,
            "gradient_sync": "SP4_mean",
            "packed_sp_source_audit": packed_sp_audit,
        },
        "model": {
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
            "checkpoint_content_identity": checkpoint_content_identity,
            "frozen_base_transformer_sha256_initial": base_transformer_sha256,
            "frozen_base_transformer_sha256_terminal": (
                terminal_base_transformer_sha256
            ),
        },
        "runtime": {
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
            "transformers": transformers_version,
            "diffusers": diffusers_version,
            "elapsed_seconds": elapsed,
        },
        "terminal_toctou_audit": terminal_audit,
        "scientific_success_claimed": False,
        "parent_allocation_released": False,
    }
    receipt = {
        **unsigned_receipt,
        "receipt_digest": object_sha256(unsigned_receipt),
    }
    def publish_receipts() -> bool:
        runtime.atomic_json(
            output / "history.json",
            {
                "schema_version": HISTORY_SCHEMA,
                "steps": history,
                "digest": object_sha256(history),
            },
        )
        runtime.atomic_json(output / "run_receipt.json", receipt)
        return True

    _rank0_call(
        rank=distributed.rank,
        world_group=parallel.world_group,
        label="final Stage R receipt publication",
        callback=publish_receipts,
    )
    if distributed.rank == 0:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "execution_profile": args.execution_profile,
                    "stage_r_updates": len(coordinates),
                    "complete_action_result": False,
                    "resume_po40_authorized": (
                        args.execution_profile == "stage-r64"
                    ),
                    "checkpoint_sha256": checkpoint_file_sha256,
                    "receipt_digest": receipt["receipt_digest"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

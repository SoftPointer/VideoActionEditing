#!/usr/bin/env python3
"""WORLD8 same-process short -> Field14 -> active14 optimizer transaction.

The Slurm dependency and sealed upstream Field14 parent are qualification
evidence only.  This process reloads the base checkpoint, reruns the two-update
short protocol, executes the complete no-grad exact40 Field14 sweep, starts a
fresh AdamW state, and applies one update at each active index 26..39.

Downstream work uses a two-phase callback: ``prepare`` may create only staging;
after this runner mints the active14 commit receipt, ``finalize`` may atomically
publish a result bound to that receipt.  Any pre-finalize failure rolls the
active14 trainables back.  No checkpoint API or scientific authority exists.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

import graft_phase_a_field14_exact40_v1 as field14
import identity_rebinder_v1 as rebinder
import infer_lora as legacy
import infer_native_identity_generation_canary as native_generation
import infer_source_kv_carrier_oracle as source_audit
import run_graft_phase_a_a_lite_short_gpu_v1 as short_runner
import run_graft_phase_a_field14_exact40_gpu_v1 as field_runner
import train_graft_phase_a_a_lite_short_v1 as short_trainer
import train_graft_phase_a_active14_transaction_v1 as active14
import tri_branch_unipc as sampler_contract


SCHEMA_VERSION = "bernini-graft-phase-a-active14-gpu-runner-v1"
WORLD8_SCHEMA_VERSION = "bernini-graft-phase-a-active14-world8-result-v1"
PLAN_SCHEMA_VERSION = "bernini-graft-phase-a-active14-world8-plan-v1"
RUNTIME_CLOSURE_SCHEMA_VERSION = (
    "bernini-graft-phase-a-active14-runtime-python-closure-v1"
)

# Repinned only after the active14 core is frozen.  The runner's own SHA is a
# sealed CLI/plan input because self-hashing source constants are circular.
PINNED_ACTIVE14_CORE_SHA256 = (
    "83290f94d92b66b8c5d15dc516dcec1ba3a492d03c7ed901d8d215cbcaec2244"
)
MAX_WORLD8_PACKET_BYTES = 48 * 1024 * 1024


class Active14GPUError(RuntimeError):
    """Reject a live transaction without a checkpoint or success claim."""


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Active14GPUError(f"{label} must be lowercase SHA256")
    return value


def _false_authority() -> dict[str, bool]:
    return {name: False for name in active14.AUTHORITY_FIELDS}


def assert_pinned_dependencies() -> None:
    field_runner.assert_pinned_dependencies()
    if (
        PINNED_ACTIVE14_CORE_SHA256.startswith("__")
        or file_sha256(active14.__file__) != PINNED_ACTIVE14_CORE_SHA256
        or tuple(active14.ACTIVE_INDICES) != tuple(range(26, 40))
        or active14.TRAINING_REGIME != "post_bootstrap"
        or active14.WORLD_SIZE != 8
        or active14.DP_SIZE != 2
        or active14.SP_SIZE != 4
    ):
        raise Active14GPUError("active14 pinned dependency differs")


def build_parser() -> argparse.ArgumentParser:
    parser = field_runner.build_parser()
    parser.description = __doc__
    parser.add_argument("--active14-plan-path", required=True)
    parser.add_argument("--expected-active14-plan-sha256", required=True)
    parser.add_argument("--upstream-field14-receipt-path", required=True)
    parser.add_argument("--expected-upstream-field14-receipt-sha256", required=True)
    parser.add_argument("--expected-upstream-field14-job-id", required=True)
    parser.add_argument("--expected-active14-core-sha256", required=True)
    parser.add_argument("--expected-active14-runner-sha256", required=True)
    parser.add_argument(
        "--ack-active14-fresh-optimizer-14-update-transaction-no-checkpoint-no-scientific-claim",
        action="store_true",
    )
    return parser


def _read_upstream_qualification(args: argparse.Namespace) -> Mapping[str, Any]:
    raw = short_runner._read_exact_0444_file(  # noqa: SLF001
        args.upstream_field14_receipt_path,
        expected_sha256=args.expected_upstream_field14_receipt_sha256,
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Active14GPUError("upstream Field14 receipt is not ASCII JSON") from error
    if raw != active14.canonical_json_bytes(value) + b"\n":
        raise Active14GPUError("upstream Field14 receipt is not canonical newline JSON")
    runtime = value.get("runtime")
    expected_runtime_keys = {
        "field14_core_sha256",
        "field14_runner_sha256",
        "field14_source_commit",
        "short_source_commit",
        "plan_sha256",
        "source_archive_sha256",
        "runtime_closure_manifest_sha256",
        "launcher_sha256",
        "checkpoint_content_pre_sha256",
        "checkpoint_content_post_sha256",
    }
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != expected_runtime_keys
        or runtime.get("field14_core_sha256")
        != args.expected_field14_core_sha256
        or runtime.get("field14_runner_sha256")
        != args.expected_field14_runner_sha256
        or runtime.get("field14_source_commit")
        != args.expected_field14_source_commit
        or runtime.get("short_source_commit")
        != field_runner.PINNED_SHORT_SOURCE_COMMIT
        or runtime.get("plan_sha256") != args.expected_plan_sha256
        or runtime.get("checkpoint_content_pre_sha256")
        != runtime.get("checkpoint_content_post_sha256")
    ):
        raise Active14GPUError("upstream Field14 runtime binding differs")
    for name in (
        "field14_core_sha256",
        "field14_runner_sha256",
        "plan_sha256",
        "source_archive_sha256",
        "runtime_closure_manifest_sha256",
        "launcher_sha256",
        "checkpoint_content_pre_sha256",
        "checkpoint_content_post_sha256",
    ):
        _require_sha256(runtime.get(name), label=f"upstream Field14 {name}")
    return active14.validate_upstream_field14_parent(
        value,
        expected_job_id=args.expected_upstream_field14_job_id,
    )


def _read_active14_plan(args: argparse.Namespace) -> Mapping[str, Any]:
    raw = short_runner._read_exact_0444_file(  # noqa: SLF001
        args.active14_plan_path,
        expected_sha256=args.expected_active14_plan_sha256,
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Active14GPUError("active14 plan is not ASCII JSON") from error
    if raw != active14.canonical_json_bytes(value) + b"\n":
        raise Active14GPUError("active14 plan is not canonical newline JSON")
    required = {
        "schema_version",
        "field14_dependency",
        "dependency_is_queue_gate_only",
        "inherits_weights_from_dependency",
        "same_process_stages",
        "active_indices",
        "optimizer",
        "topology",
        "resources",
        "runtime",
        "checkpoint",
        "claim_scope",
        "authority",
    }
    dependency = value.get("field14_dependency")
    runtime = value.get("runtime")
    authority = value.get("authority")
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("dependency_is_queue_gate_only") is not True
        or value.get("inherits_weights_from_dependency") is not False
        or value.get("same_process_stages")
        != [
            "base-checkpoint-load",
            "short-updates-29-38-and-confirmation",
            "field14-exact40-no-grad",
            "active14-fresh-adamw-updates-26-39",
        ]
        or value.get("active_indices") != list(range(26, 40))
        or dependency
        != {
            "job_id": args.expected_upstream_field14_job_id,
            "kind": "afterok",
            "receipt_path": args.upstream_field14_receipt_path,
            "receipt_sha256_policy": (
                "derive-from-stable-sealed-file-after-afterok"
            ),
        }
        or not args.expected_upstream_field14_job_id.isdecimal()
        or value.get("optimizer")
        != {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "moments": "fresh-zero-state-after-short-and-field14",
            "steps": 14,
            "training_regime": "post_bootstrap",
            "weight_decay": 0.0,
        }
        or value.get("topology")
        != {
            "allocation": "single-node-8xMI210",
            "dp_size": 2,
            "sp_size": 4,
            "world_size": 8,
        }
        or value.get("resources")
        != {
            "cpus_per_task": 64,
            "gpus": 8,
            "memory_gib": 256,
            "nodes": 1,
            "ntasks": 1,
            "time_limit_hours": 72,
        }
        or value.get("claim_scope")
        != "operational-same-process-active14-transaction-no-checkpoint-no-training-or-scientific-authority"
        or not isinstance(runtime, Mapping)
        or runtime.get("active14_core_sha256")
        != args.expected_active14_core_sha256
        or runtime.get("active14_runner_sha256")
        != args.expected_active14_runner_sha256
        or runtime.get("field14_core_sha256")
        != args.expected_field14_core_sha256
        or runtime.get("field14_runner_sha256")
        != args.expected_field14_runner_sha256
        or runtime.get("field14_source_commit")
        != args.expected_field14_source_commit
        or runtime.get("short_runner_sha256")
        != field_runner.PINNED_SHORT_RUNNER_SHA256
        or not isinstance(authority, Mapping)
        or set(authority) != set(active14.AUTHORITY_FIELDS)
        or any(authority.get(name) is not False for name in active14.AUTHORITY_FIELDS)
    ):
        raise Active14GPUError("active14 plan contract differs")
    checkpoint = value.get("checkpoint")
    if checkpoint != {
        "content_manifest_sha256": args.expected_checkpoint_content_manifest_sha256,
        "tree_sha256": args.expected_checkpoint_tree_sha256,
        "written_by_this_job": False,
    }:
        raise Active14GPUError("active14 plan checkpoint differs")
    return value


def validate_cli(args: argparse.Namespace) -> argparse.Namespace:
    field_runner.validate_cli(args)
    for name in (
        "expected_active14_plan_sha256",
        "expected_upstream_field14_receipt_sha256",
        "expected_active14_core_sha256",
        "expected_active14_runner_sha256",
    ):
        _require_sha256(getattr(args, name), label=name)
    if (
        args.expected_active14_core_sha256 != PINNED_ACTIVE14_CORE_SHA256
        or file_sha256(__file__) != args.expected_active14_runner_sha256
        or not Path(args.active14_plan_path).is_absolute()
        or not Path(args.upstream_field14_receipt_path).is_absolute()
        or not args.expected_upstream_field14_job_id.isdecimal()
        or args.ack_active14_fresh_optimizer_14_update_transaction_no_checkpoint_no_scientific_claim
        is not True
    ):
        raise Active14GPUError("active14 CLI source/path/acknowledgement differs")
    assert_pinned_dependencies()
    _read_active14_plan(args)
    _read_upstream_qualification(args)
    return args


class Active14AtlasRouteFactory(short_runner.ShortTrainingAtlasRouteFactory):
    """The frozen four-forward atlas route extended to exact active14 order."""

    def begin(
        self,
        *,
        update_number: int,
        schedule_index: int,
        row_iid: str,
        row_source_sha256: str,
        source_frames: torch.Tensor,
    ) -> None:
        if (
            self._pending is not None  # noqa: SLF001
            or update_number not in range(1, 15)
            or schedule_index != active14.ACTIVE_INDICES[update_number - 1]
        ):
            raise Active14GPUError("active14 route transaction differs")
        _require_sha256(row_source_sha256, label="active14 route source SHA256")
        if (
            type(source_frames) is not torch.Tensor
            or source_frames.dtype != torch.float32
            or tuple(source_frames.shape[:3]) != (1, short_runner.FRAME_COUNT, 3)
            or source_frames.requires_grad
            or not source_frames.is_contiguous()
            or not bool(torch.isfinite(source_frames).all().item())
        ):
            raise Active14GPUError("active14 route source frames differ")
        self._pending = {  # noqa: SLF001
            "update_number": update_number,
            "schedule_index": schedule_index,
            "row_iid": row_iid,
            "row_source_sha256": row_source_sha256,
            "source_frames": source_frames,
            "source_frames_identity": short_runner.tensor_identity(source_frames),
            "rows": [],
            "atlas_objects": [],
        }

    def finish(
        self,
        *,
        plan: active14.Active14CellPlan,
        update_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        pending = self._pending  # noqa: SLF001
        if pending is None:
            raise Active14GPUError("active14 route has no transaction")
        rows = pending["rows"]
        observed = [
            (row["phase"], row["role"], row["graph_expected"]) for row in rows
        ]
        expected = list(self._EXPECTED)  # noqa: SLF001
        if (
            observed != expected
            or plan.update_number != pending["update_number"]
            or plan.schedule_index != pending["schedule_index"]
            or plan.row_iid != pending["row_iid"]
            or update_receipt.get("schedule_index") != pending["schedule_index"]
            or len({row["atlas_tokens"]["content_sha256"] for row in rows}) != 1
        ):
            raise Active14GPUError("active14 route completion differs")
        receipt = active14.seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-active14-update-route-v1",
                "update_number": pending["update_number"],
                "schedule_index": pending["schedule_index"],
                "row_iid": pending["row_iid"],
                "row_source_sha256": pending["row_source_sha256"],
                "fit_row_only": True,
                "exact_four_native_forwards": True,
                "forward_order": [[row[0], row[1]] for row in self._EXPECTED],  # noqa: SLF001
                "fresh_atlas_per_forward": True,
                "measurement_atlas_detached": True,
                "replay_atlas_graph_bearing_only_on_target_owner": True,
                "rows": [dict(row) for row in rows],
                "checkpoint_written": False,
                "publication_performed": False,
                **_false_authority(),
            }
        )
        self._completed.append(receipt)  # noqa: SLF001
        self._pending = None  # noqa: SLF001
        return receipt


def _active14_forward_route_receipt() -> Mapping[str, Any]:
    return active14.seal_mapping(
        {
            "schema_version": short_runner.native_v2.FORWARD_ROUTE_SCHEMA_VERSION,
            "route_kind": "identity_rebinder_v1",
            "phase_a_active_schedule_indices": list(
                short_runner.native_v2.PHASE_A_ACTIVE_SCHEDULE_INDICES
            ),
            "inactive_schedule_policy": "exact_zero_update_not_trained",
            "target_queries_only": True,
            "condition_rows_written": False,
            "external_oracle_inputs": False,
            "factory": "fresh-fit-atlas-per-active14-native-forward",
            "update_schedule_indices": list(active14.ACTIVE_INDICES),
        }
    )


class OfficialActive14Runtime:
    """Own active14 cell inputs while reusing the live post-Field14 model."""

    def __init__(
        self,
        *,
        short_runtime: short_runner.OfficialShortRuntime,
        bindings: short_runner.native_v2.AuthenticatedNativeBindings,
        route_factory: Active14AtlasRouteFactory,
    ) -> None:
        if (
            type(short_runtime) is not short_runner.OfficialShortRuntime
            or type(route_factory) is not Active14AtlasRouteFactory
            or tuple(short_runtime._confirmation_seen)  # noqa: SLF001
            != tuple(short_runner.CONFIRMATION_INDICES)
        ):
            raise Active14GPUError("active14 live runtime input differs")
        self.short_runtime = short_runtime
        self.bindings = bindings
        self.route_factory = route_factory
        self._inputs: dict[int, Mapping[str, Any]] = {}

    def make_update_cell(
        self, plan: active14.Active14CellPlan
    ) -> short_runner.native_v2.PhaseANativeTrainingClosure:
        runtime = self.short_runtime
        if (
            plan.row_iid != runtime.local.fit_iid
            or plan.row_source_sha256 != runtime.fit.row.source_sha256
            or plan.schedule_index not in active14.ACTIVE_INDICES
            or plan.schedule_index in self._inputs
        ):
            raise Active14GPUError("active14 update plan differs")
        coordinate = runtime.schedule.coordinate(plan.schedule_index)
        noise = short_runner.keyed_fresh_gaussian(
            shape=runtime.fit.source_latent.shape,
            device=runtime.fit.source_latent.device,
            source_sha256=runtime.fit.row.source_sha256,
            purpose="active14-optimizer-update",
            schedule_index=plan.schedule_index,
        )
        noisy, noisy_receipt = short_runner.native_runner_v1.build_noisy_target(
            runtime.fit.source_latent,
            noise.epsilon,
            sigma=coordinate.sigma,
        )
        self.route_factory.begin(
            update_number=plan.update_number,
            schedule_index=plan.schedule_index,
            row_iid=plan.row_iid,
            row_source_sha256=plan.row_source_sha256,
            source_frames=runtime.fit.atlas_frames,
        )
        self._inputs[plan.schedule_index] = active14.seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-active14-update-input-v1",
                "update_number": plan.update_number,
                "schedule_index": plan.schedule_index,
                "row_iid": plan.row_iid,
                "source_state_receipt_digest": runtime.fit.receipt["digest"],
                "coordinate_digest": coordinate.receipt["digest"],
                "epsilon_receipt_digest": noise.receipt["digest"],
                "noisy_target_receipt_digest": noisy_receipt["digest"],
                "positive_condition_role": "canonical_source_noop_r2v",
                "negative_condition_role": "pinned_renderer_negative",
                "target_video_used": False,
            }
        )
        return short_runner.native_v2.PhaseANativeTrainingClosure(
            bindings=self.bindings,
            source_video=runtime.fit.source_latent,
            noisy_target=noisy,
            negative_condition=runtime.negative_condition,
            positive_condition=runtime.noop_condition,
            schedule_index=plan.schedule_index,
            sigma=coordinate.sigma,
            timestep=coordinate.timestep,
        )

    def after_update(
        self,
        plan: active14.Active14CellPlan,
        admission: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        route = self.route_factory.finish(plan=plan, update_receipt=admission)
        plain = dict(route)
        plain.pop("digest")
        plain["update_input_digest"] = self._inputs[plan.schedule_index]["digest"]
        return active14.seal_mapping(plain)


@dataclass(frozen=True)
class Active14ContinuationContext:
    topology: short_runner.DistributedTopology
    backend: short_trainer.AuthenticatedDP2SP4Backend
    renderer: torch.nn.Module
    diffusion: torch.nn.Module
    transformer: torch.nn.Module
    handle: rebinder.IdentityRebinderHandle
    bindings: short_runner.native_v2.AuthenticatedNativeBindings
    schedule: short_runner.Exact40CoordinateRegistry
    fit: short_runner.SourceState
    confirmation: short_runner.SourceState
    negative_condition: torch.Tensor
    noop_condition: torch.Tensor
    action_condition: torch.Tensor
    device: torch.device
    local_rank: int
    bernini_revision: str
    checkpoint_root: Path
    checkpoint_content_identity: Mapping[str, Any]
    checkpoint_manifest_sha256: str
    short_receipt: Mapping[str, Any]
    field14_receipt: Mapping[str, Any]
    active14_precommit_receipt: Mapping[str, Any]
    trainable_final_digest: str
    frozen_base_final_digest: str


PrepareCallback = Callable[
    [Active14ContinuationContext, Mapping[str, Any]], Mapping[str, Any]
]
FinalizeCallback = Callable[
    [
        Active14ContinuationContext,
        Mapping[str, Any],
        Mapping[str, Any],
    ],
    Mapping[str, Any],
]


def _assemble_world8_packets(
    packets: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not isinstance(packets, (list, tuple)) or len(packets) != 8:
        raise Active14GPUError("active14 WORLD8 packet coverage differs")
    rows = []
    for rank, packet in enumerate(packets):
        if (
            not isinstance(packet, Mapping)
            or packet.get("global_rank") != rank
            or len(active14.canonical_json_bytes(packet)) >= MAX_WORLD8_PACKET_BYTES
        ):
            raise Active14GPUError("active14 WORLD8 packet differs")
        commit = active14.validate_sealed_mapping(
            packet.get("active14_commit_receipt"),
            label=f"active14 rank{rank} commit",
        )
        outer = active14.validate_sealed_mapping(
            packet.get("transaction_receipt"),
            label=f"active14 rank{rank} transaction",
        )
        if (
            packet.get("active14_commit_receipt_digest") != commit["digest"]
            or packet.get("transaction_receipt_digest") != outer["digest"]
            or commit.get("all_fourteen_updates_completed") is not True
            or commit.get("transaction_committed_in_memory") is not True
            or commit.get("optimizer_contract", {}).get("schedule_indices")
            != list(range(26, 40))
            or commit.get("checkpoint_written") is not False
            or commit.get("publication_performed") is not False
        ):
            raise Active14GPUError("active14 WORLD8 commit differs")
        active14.assert_no_authority(packet)
        rows.append(
            {
                "global_rank": rank,
                "dp_arm": rank // 4,
                "sp_rank": rank % 4,
                "family": commit["family"],
                "active14_commit_receipt_digest": commit["digest"],
                "transaction_receipt_digest": outer["digest"],
                "initial_trainable_digest": commit["initial_trainable_digest"],
                "final_trainable_digest": commit["final_trainable_digest"],
                "frozen_base_digest": commit["final_frozen_base_digest"],
            }
        )
    for arm in range(2):
        arm_rows = rows[arm * 4 : (arm + 1) * 4]
        for key in (
            "family",
            "initial_trainable_digest",
            "final_trainable_digest",
            "frozen_base_digest",
        ):
            if len({row[key] for row in arm_rows}) != 1:
                raise Active14GPUError(f"active14 arm{arm} SP4 {key} differs")
    return active14.seal_mapping(
        {
            "schema_version": WORLD8_SCHEMA_VERSION,
            "rank_order": list(range(8)),
            "dp2_family_order": list(short_runner.FAMILY_BY_DP_ARM),
            "rows": rows,
            "all_eight_active14_transactions_completed": True,
            "both_sp4_arms_parameter_consensus": True,
            "frozen_base_unchanged": True,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


def replay_active14_for_downstream(
    args: argparse.Namespace,
    routing: short_runner.source_consumer.TrainerRouting,
    *,
    prepare: PrepareCallback,
    finalize: FinalizeCallback,
) -> Mapping[str, Any]:
    """Keep the official model/process groups live across both callbacks."""

    if not callable(prepare) or not callable(finalize):
        raise Active14GPUError("active14 downstream callbacks differ")
    plan = _read_active14_plan(args)
    upstream_qualification = _read_upstream_qualification(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise Active14GPUError(str(error)) from error
    if int(transformer_config.get("num_attention_heads", -1)) != 12:
        raise Active14GPUError("active14 checkpoint head count differs")
    manifest_path = Path(args.checkpoint_content_manifest).resolve(strict=True)
    if file_sha256(manifest_path) != args.expected_checkpoint_content_manifest_sha256:
        raise Active14GPUError("active14 checkpoint manifest differs")
    inference_hashes = legacy.validate_inference_source_files(bernini_root)
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    import bernini.models.transformer_wan as transformer_wan
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if (
        SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT
        or DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT
    ):
        raise Active14GPUError("active14 official prompt constants differ")
    try:
        topology = short_runner._initialize_world8_dp2sp4(  # noqa: SLF001
            init_parallel_state
        )
    except Exception:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    local = short_runner.route_local_family(routing, dp_arm=topology.dp_arm)
    device = torch.device("cuda", topology.local_rank)
    handle: Optional[rebinder.IdentityRebinderHandle] = None
    try:
        field_plan = field_runner._read_plan(args)  # noqa: SLF001
        source_binding = active14.seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-active14-source-binding-v1",
                "active14_runner_sha256": args.expected_active14_runner_sha256,
                "active14_core_sha256": args.expected_active14_core_sha256,
                "field14_runner_sha256": args.expected_field14_runner_sha256,
                "field14_core_sha256": args.expected_field14_core_sha256,
                "field14_source_commit": args.expected_field14_source_commit,
                "short_runner_sha256": field_runner.PINNED_SHORT_RUNNER_SHA256,
                "plan_sha256": args.expected_active14_plan_sha256,
                "field14_plan_sha256": args.expected_plan_sha256,
                "upstream_qualification_digest": upstream_qualification["digest"],
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "bernini_inference_files": inference_hashes,
            }
        )
        short_runner._gather_equal(  # noqa: SLF001
            json.loads(active14.canonical_json_bytes(source_binding)),
            group=topology.world_group,
            count=8,
            label="active14 source binding",
        )

        checkpoint_rows: list[Any] = [None]
        if topology.global_rank == 0:
            try:
                checkpoint_rows[0] = {
                    "ok": True,
                    "identity": source_audit.validate_checkpoint_content(
                        checkpoint,
                        manifest_path,
                        expected_manifest_sha256=(
                            args.expected_checkpoint_content_manifest_sha256
                        ),
                    ),
                }
            except Exception as error:
                checkpoint_rows[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(checkpoint_rows, src=0)
        checkpoint_result = checkpoint_rows[0]
        if (
            not isinstance(checkpoint_result, Mapping)
            or checkpoint_result.get("ok") is not True
        ):
            raise Active14GPUError(
                f"active14 checkpoint validation failed: {checkpoint_result}"
            )
        checkpoint_identity = active14.seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-active14-checkpoint-v1",
                "identity": dict(checkpoint_result["identity"]),
            }
        )
        short_runner._gather_equal(  # noqa: SLF001
            json.loads(active14.canonical_json_bytes(checkpoint_identity)),
            group=topology.world_group,
            count=8,
            label="active14 checkpoint content",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint),
            subfolder="tokenizer",
            **legacy.tokenizer_load_kwargs(),
        )
        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        if float(config.shift) != native_generation.FLOW_SHIFT or config.use_unipc is not True:
            raise Active14GPUError("active14 renderer is not pinned UniPC shift5")
        renderer = BerniniRendererModel(config)
        renderer.eval().requires_grad_(False)
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval().requires_grad_(False).to(device)
        fit = short_runner._encode_source_state(  # noqa: SLF001
            row=local.fit_row,
            vae=vae,
            vae_encode=_vae_encode,
            device=device,
            sp_group=topology.sp_group,
        )
        confirmation = short_runner._encode_source_state(  # noqa: SLF001
            row=local.confirmation_row,
            vae=vae,
            vae_encode=_vae_encode,
            device=device,
            sp_group=topology.sp_group,
        )
        vae.to("cpu")
        del vae
        torch.cuda.empty_cache()

        renderer.to(device)
        diffusion = source_audit.resolve_diffusion_core(renderer)
        transformer = diffusion.transformer
        if transformer is None or getattr(diffusion, "transformer_2", None) is not None:
            raise Active14GPUError("active14 requires one pinned transformer_1")
        renderer.eval().requires_grad_(False)
        wan_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
        )
        schedule = short_runner.Exact40CoordinateRegistry(
            diffusion.scheduler, device=device
        )
        short_runner._gather_equal(  # noqa: SLF001
            schedule.receipt,
            group=topology.world_group,
            count=8,
            label="active14 exact40 schedule",
        )
        negative, noop, action, condition_receipt = short_runner._encode_conditions(  # noqa: SLF001
            tokenizer=tokenizer,
            renderer=renderer,
            prompt_cleaner=prompt_clean,
            device=device,
            local=local,
        )
        short_runner._gather_equal(  # noqa: SLF001
            condition_receipt,
            group=topology.sp_group,
            count=4,
            label="active14 family conditions",
        )
        renderer.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()

        preinstall_baseline = field_runner._capture_full_preinstall_baseline(  # noqa: SLF001
            diffusion=diffusion,
            transformer=transformer,
            schedule=schedule,
            confirmation=confirmation,
            negative_condition=negative,
            noop_condition=noop,
            action_condition=action,
        )
        short_runner._gather_equal(  # noqa: SLF001
            preinstall_baseline,
            group=topology.sp_group,
            count=4,
            label="active14 preinstall exact40 baseline",
        )
        short_baseline = field_runner._short_baseline_projection(  # noqa: SLF001
            preinstall_baseline
        )
        base_rows = short_runner.native_runner_v1._base_parameter_rows(  # noqa: SLF001
            transformer
        )
        base_before = short_runner.short_chunked_parameter_registry_digest(base_rows)
        handle = rebinder.install_identity_rebinder_v1(
            transformer,
            runtime_source_commit=bernini_revision,
            model_revision=rebinder.PINNED_BERNINI_MODEL_REVISION,
            checkpoint_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        )
        transformer.eval()
        handle.atlas_encoder.eval()
        trainable_rows = handle.trainable_named_parameters()
        initialization = short_runner._broadcast_initial_trainables(  # noqa: SLF001
            trainable_rows, world_group=topology.world_group
        )
        short_route_factory = short_runner.ShortTrainingAtlasRouteFactory(
            handle=handle,
            sp_rank=topology.sp_rank,
            sp_group=topology.sp_group,
        )
        short_bindings = short_runner.native_v2.authenticate_pinned_native_bindings(
            diffusion=diffusion,
            transformer=transformer,
            named_trainable_parameters=trainable_rows,
            external_trainable_owner_modules={"atlas_encoder": handle.atlas_encoder},
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
            transformer_wan_path=Path(transformer_wan.__file__).resolve(strict=True),
            bernini_commit=bernini_revision,
            forward_context_factory=short_route_factory,
            forward_route_receipt=short_runner._official_forward_route_receipt(),  # noqa: SLF001
        )
        backend = short_trainer.authenticate_torch_distributed_world8_dp2sp4(
            world_group=topology.world_group,
            sp_group=topology.sp_group,
            dp_group=topology.dp_group,
        )
        short_runtime = short_runner.OfficialShortRuntime(
            local=local,
            bindings=short_bindings,
            diffusion=diffusion,
            transformer=transformer,
            handle=handle,
            route_factory=short_route_factory,
            schedule=schedule,
            fit=fit,
            confirmation=confirmation,
            negative_condition=negative,
            noop_condition=noop,
            action_condition=action,
            sp_rank=topology.sp_rank,
            sp_group=topology.sp_group,
            adapter_off_baseline=short_baseline,
        )
        short_result = short_runner.execute_authenticated_short_run(
            routing=routing,
            bindings=short_bindings,
            collectives=backend,
            services=short_runner.authenticate_official_services(short_runtime),
        )
        trainable_before_sweep = short_runner.short_chunked_parameter_registry_digest(
            trainable_rows
        )
        field_runtime = field_runner.OfficialField14Runtime(
            short_runtime=short_runtime,
            preinstall_baseline=preinstall_baseline,
        )
        with torch.no_grad():
            field_result = field14.execute_exact40_sweep(
                family=local.family,
                confirmation_iid=local.confirmation_iid,
                confirmation_source_sha256=confirmation.row.source_sha256,
                wrong_owner_iid=local.fit_iid,
                wrong_owner_source_sha256=fit.row.source_sha256,
                short_result_digest=short_result.receipt["digest"],
                preinstall_baseline_digest=preinstall_baseline["digest"],
                measure_index=field_runtime.measure_index,
                release_index=field_runtime.release_index,
            )
        field_runtime.assert_complete()
        if (
            short_runner.short_chunked_parameter_registry_digest(trainable_rows)
            != trainable_before_sweep
        ):
            raise Active14GPUError("Field14 changed trainables before active14")

        active_route_factory = Active14AtlasRouteFactory(
            handle=handle,
            sp_rank=topology.sp_rank,
            sp_group=topology.sp_group,
        )
        active_bindings = short_runner.native_v2.authenticate_pinned_native_bindings(
            diffusion=diffusion,
            transformer=transformer,
            named_trainable_parameters=trainable_rows,
            external_trainable_owner_modules={"atlas_encoder": handle.atlas_encoder},
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
            transformer_wan_path=Path(transformer_wan.__file__).resolve(strict=True),
            bernini_commit=bernini_revision,
            forward_context_factory=active_route_factory,
            forward_route_receipt=_active14_forward_route_receipt(),
        )
        active_runtime = OfficialActive14Runtime(
            short_runtime=short_runtime,
            bindings=active_bindings,
            route_factory=active_route_factory,
        )
        active_services = active14.authenticate_official_services(
            make_update_cell=active_runtime.make_update_cell,
            after_update=active_runtime.after_update,
            assert_schedule_unchanged=schedule.assert_unchanged,
        )
        context_slot: list[Optional[Active14ContinuationContext]] = [None]

        def prepare_bridge(precommit: Mapping[str, Any]) -> Mapping[str, Any]:
            context = Active14ContinuationContext(
                topology=topology,
                backend=backend,
                renderer=renderer,
                diffusion=diffusion,
                transformer=transformer,
                handle=handle,
                bindings=active_bindings,
                schedule=schedule,
                fit=fit,
                confirmation=confirmation,
                negative_condition=negative,
                noop_condition=noop,
                action_condition=action,
                device=device,
                local_rank=topology.local_rank,
                bernini_revision=bernini_revision,
                checkpoint_root=checkpoint,
                checkpoint_content_identity=checkpoint_identity,
                checkpoint_manifest_sha256=(
                    args.expected_checkpoint_content_manifest_sha256
                ),
                short_receipt=short_result.receipt,
                field14_receipt=field_result,
                active14_precommit_receipt=precommit,
                trainable_final_digest=precommit["final_trainable_digest"],
                frozen_base_final_digest=precommit["frozen_base_digest"],
            )
            context_slot[0] = context
            return prepare(context, precommit)

        def finalize_bridge(
            commit: Mapping[str, Any], preparation: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            context = context_slot[0]
            if context is None:
                raise Active14GPUError("active14 finalize lacks prepared context")
            return finalize(context, commit, preparation)

        transaction = active14.execute_active14_transaction(
            upstream_qualification=upstream_qualification,
            local_field14_receipt=field_result,
            short_result_digest=short_result.receipt["digest"],
            family=local.family,
            row_iid=local.fit_iid,
            row_source_sha256=fit.row.source_sha256,
            bindings=active_bindings,
            backend=backend,
            services=active_services,
            prepare=prepare_bridge,
            finalize=finalize_bridge,
        )
        active_commit_plain = json.loads(
            active14.canonical_json_bytes(transaction.active14_commit_receipt)
        )
        outer_plain = json.loads(active14.canonical_json_bytes(transaction.receipt))
        local_packet = {
            "global_rank": topology.global_rank,
            "active14_commit_receipt_digest": active_commit_plain["digest"],
            "active14_commit_receipt": active_commit_plain,
            "transaction_receipt_digest": outer_plain["digest"],
            "transaction_receipt": outer_plain,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
        if (
            len(active14.canonical_json_bytes(local_packet))
            >= MAX_WORLD8_PACKET_BYTES
            or pickle.loads(pickle.dumps(local_packet, protocol=5)) != local_packet
        ):
            raise Active14GPUError("active14 local packet is not bounded/pickle-safe")
        packets: list[Any] = [None] * 8
        dist.all_gather_object(packets, local_packet)
        world8 = _assemble_world8_packets(packets)
        base_after = short_runner.short_chunked_parameter_registry_digest(base_rows)
        if base_after != base_before or any(
            parameter.grad is not None for _, parameter in base_rows
        ):
            raise Active14GPUError("active14 frozen base final invariant differs")
        assembled = active14.seal_mapping(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed_same_process_short_field14_active14_no_checkpoint",
                "world8": world8,
                "topology_receipt": topology.receipt,
                "source_binding": source_binding,
                "active14_plan": plan,
                "field14_plan": field_plan,
                "upstream_qualification": upstream_qualification,
                "checkpoint_identity": checkpoint_identity,
                "initialization": initialization,
                "local_short_receipt": short_result.receipt,
                "local_field14_receipt": field_result,
                "local_active14_commit_receipt": transaction.active14_commit_receipt,
                "local_transaction_receipt": transaction.receipt,
                "base_sha256_before": base_before,
                "base_sha256_after": base_after,
                "base_unchanged": True,
                "wan_diffusion_sha256": wan_sha,
                "transformer_wan_sha256": file_sha256(transformer_wan.__file__),
                "runtime_versions": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "diffusers": diffusers_version,
                    "transformers": transformers_version,
                },
                "dependency_afterok_is_queue_gate_only": True,
                "weights_inherited_from_dependency_job": False,
                "checkpoint_written": False,
                "publication_performed": False,
                **_false_authority(),
            }
        )
        active14.assert_no_authority(assembled)
        dist.barrier()
        if topology.global_rank == 0:
            print(
                active14.canonical_json_bytes(assembled).decode("ascii"),
                flush=True,
            )
        dist.barrier()
        return assembled
    finally:
        if (
            handle is not None
            and not handle.restored
            and rebinder.active_route() is None
        ):
            handle.restore()
        if dist.is_initialized():
            dist.destroy_process_group()


def _standalone_prepare(
    _context: Active14ContinuationContext,
    _precommit: Mapping[str, Any],
) -> Mapping[str, Any]:
    return active14.seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-active14-standalone-prepare-v1",
            "preparation_completed": True,
            "kind": "no-downstream-staging",
            "published": False,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


def _standalone_finalize(
    _context: Active14ContinuationContext,
    commit: Mapping[str, Any],
    preparation: Mapping[str, Any],
) -> Mapping[str, Any]:
    return active14.seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-active14-standalone-finalize-v1",
            "finalize_completed": True,
            "active14_commit_receipt_digest": commit["digest"],
            "preparation_receipt_digest": preparation["digest"],
            "published": False,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    routing = short_runner.consume_authenticated_source_routing(args)
    replay_active14_for_downstream(
        args,
        routing,
        prepare=_standalone_prepare,
        finalize=_standalone_finalize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Active14AtlasRouteFactory",
    "Active14ContinuationContext",
    "Active14GPUError",
    "OfficialActive14Runtime",
    "PINNED_ACTIVE14_CORE_SHA256",
    "PLAN_SCHEMA_VERSION",
    "RUNTIME_CLOSURE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "WORLD8_SCHEMA_VERSION",
    "_assemble_world8_packets",
    "assert_pinned_dependencies",
    "build_parser",
    "file_sha256",
    "main",
    "replay_active14_for_downstream",
    "validate_cli",
]

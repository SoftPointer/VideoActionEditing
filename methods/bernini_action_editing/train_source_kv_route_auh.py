#!/usr/bin/env python3
"""Independent exact-40 AUH trainer for Bernini CSV-ART V9.

This runner owns the complete optimizer-step lifetime.  It deliberately does
not install a strategy into the V7/V8 loop: source-only K/V capture, five pair
forwards, the two adapted checkpoint graphs, ``loss.backward()``, replay-cache
audit, cache retirement, and AdamW update all occur in this file.

The paired target is used only to construct the detached executable teacher.
Every renderer call receives only the pinned model-field view of a beta-zero
source-endpoint query.  No mask, track, flow, pose, trajectory, swept tube, or
first-frame anchor is accepted.
"""

from __future__ import annotations

import argparse
import bisect
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import inference_sigma_strata as sigma_strata  # noqa: E402
import motion_residual as motion  # noqa: E402
import source_kv_replay as replay  # noqa: E402
import source_kv_route_batches as route_batches  # noqa: E402
import source_kv_route_objective as objective  # noqa: E402
import source_kv_route_scope as route_scope  # noqa: E402
import train_cross_mode_cmsg_auh as v6_runtime  # noqa: E402
import train_delta_lora as v4  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_prior_tangent_lora as v5  # noqa: E402


METHOD_NAME = "bernini-frozen-source-kv-csv-art-v9-auh"
RECEIPT_SCHEMA = "bernini-source-kv-route-auh-training-receipt-v9"
OPTIMIZER_SCHEMA = "bernini-source-kv-route-auh-optimizer-v9"
ARTIFACT_VALIDATION_SCHEMA = "bernini-source-kv-route-artifact-validation-v9"

NUM_FRAMES = 81
LATENT_PHASES = 21
MAX_STEPS = 40
SAVE_EVERY = 40
LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
METRICS_TIMING = "post_backward_pre_optimizer_for_loss_and_cache"
FORWARD_ORDER = objective.FORWARD_BRANCH_ORDER

PINNED_DATASET_SHARDS = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/data/"
    "vae_full_81f_4d41e4c/shards"
)
PINNED_DATASET_SUMMARY = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/data/"
    "vae_full_81f_4d41e4c/dataset_summary.json"
)
PINNED_DATASET_SUMMARY_FILE_SHA256 = (
    "5dc45b4a6d700b3cd0108e941242ae364396458f20f41249744e74e00acc02dd"
)
PINNED_DATASET_INDEX = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/data/"
    "vae_full_81f_4d41e4c/dataset_index.jsonl"
)
PINNED_DATASET_INDEX_SHA256 = (
    "d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b"
)
PINNED_ROUTING_JSONL = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_c2fr_20260806/routing/strict359.jsonl"
)
PINNED_ROUTING_SHA256 = (
    "0da09787889687726d9161b0c74b8df5d58226f6e431632b317891d630ef49eb"
)
PINNED_DATASET_ROWS = 644
PINNED_ELIGIBLE_ROWS = 359
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
CHECKPOINT_CONTENT_FILE_COUNT = 23

# The pre-registered V9 pilot is deliberately a source-tangent experiment.  It
# is not a claim that training queries match every state visited by a scheduler
# at deployment.  Keep endpoint selection behind one narrow helper so a later
# target-free stop-gradient self-rollout / DAgger arm can replace this policy
# without admitting the paired target into the renderer input.
QUERY_STATE_POLICY = "offline_source_tangent_beta0"
QUERY_STATE_TRAIN_FORMULA = "y0=(1-sigma)*source+sigma*epsilon"
QUERY_STATE_DEPLOYMENT_FORMULA = (
    "deployment_step_k_gt_0_tail=scheduler_evolving_self_generated_edit_state"
)
QUERY_STATE_FOLLOWUP_ARM = "target_free_stop_gradient_self_rollout_or_dagger"


class SourceKVRouteAUHError(RuntimeError):
    """Raised before an invalid V9 step can mutate the adapter."""


@dataclass(frozen=True)
class PreparedSourceKVRouteCandidate:
    editor_negative: Mapping[str, Any]
    editor_noop: Mapping[str, Any]
    editor_action: Mapping[str, Any]
    auxiliary: Mapping[str, Any]
    spatial_hw: tuple[int, int]
    instruction_sha256: str


@dataclass(frozen=True)
class MovedSourceKVRouteCandidate:
    editor_negative: Mapping[str, Any]
    editor_noop: Mapping[str, Any]
    editor_action: Mapping[str, Any]
    carrier: route_batches.SourceOnlyCarrierBatch
    auxiliary: Mapping[str, Any]
    spatial_hw: tuple[int, int]
    instruction_sha256: str


@dataclass(frozen=True)
class SixForwardCellResult:
    loss_result: objective.SourceKVRouteLossResult
    forward_order: tuple[str, ...]
    fresh_parity_checked: bool
    fresh_noop_exact: bool
    fresh_action_exact: bool
    carrier_output_shape: tuple[int, ...]


@dataclass(frozen=True)
class OptimizerStepResult:
    cell: SixForwardCellResult
    record: Mapping[str, Any]


class AccessedShardIntegrityTracker:
    """Hash-close each optimizer-accessed parquet read against the pinned index."""

    def __init__(
        self,
        *,
        dataset: Any,
        expected_shard_sha256: Mapping[Path, str],
        index_path: Path,
    ) -> None:
        self.dataset = dataset
        self.index_path = Path(index_path).resolve()
        self.expected_shard_sha256 = {
            Path(path).resolve(): digest
            for path, digest in expected_shard_sha256.items()
        }
        self.accesses: list[dict[str, Any]] = []

    @classmethod
    def from_pinned_index(
        cls, *, dataset: Any, index_path: str | Path = PINNED_DATASET_INDEX
    ) -> "AccessedShardIntegrityTracker":
        path = Path(index_path).resolve(strict=True)
        if (
            not path.is_file()
            or path.is_symlink()
            or str(path) != PINNED_DATASET_INDEX
            or _sha256_file(path) != PINNED_DATASET_INDEX_SHA256
        ):
            raise SourceKVRouteAUHError("pinned dataset index changed before tracking")
        expected: dict[Path, str] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise SourceKVRouteAUHError(
                            f"dataset index row is not an object at line {line_number}"
                        )
                    shard = Path(str(row.get("parquet_path"))).resolve(strict=True)
                    digest = row.get("parquet_sha256")
                    if (
                        not isinstance(digest, str)
                        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                        or shard in expected
                        or shard.parent != dataset.root
                        or shard.is_symlink()
                    ):
                        raise SourceKVRouteAUHError(
                            f"dataset index shard contract differs at line {line_number}"
                        )
                    expected[shard] = digest
        except (OSError, json.JSONDecodeError) as error:
            raise SourceKVRouteAUHError(
                f"cannot construct accessed-shard integrity tracker: {error}"
            ) from error
        dataset_files = {Path(value).resolve() for value in dataset.files}
        if (
            len(expected) != PINNED_DATASET_ROWS
            or set(expected) != dataset_files
            or _sha256_file(path) != PINNED_DATASET_INDEX_SHA256
        ):
            raise SourceKVRouteAUHError("pinned dataset index membership changed")
        return cls(
            dataset=dataset,
            expected_shard_sha256=expected,
            index_path=path,
        )

    def _shard_for_index(self, index: int) -> Path:
        length = len(self.dataset)
        normalized = index + length if index < 0 else index
        if normalized < 0 or normalized >= length:
            raise IndexError(index)
        group_index = bisect.bisect_right(self.dataset._ends, normalized)
        _, _, path, _ = self.dataset._groups[group_index]
        resolved = Path(path).resolve(strict=True)
        if resolved not in self.expected_shard_sha256:
            raise SourceKVRouteAUHError("optimizer row maps outside the pinned index")
        return resolved

    def read(self, index: int, *, access_ordinal: int) -> tuple[Mapping[str, Any], dict[str, Any]]:
        if access_ordinal != len(self.accesses):
            raise SourceKVRouteAUHError("accessed-shard ordinal is non-contiguous")
        shard = self._shard_for_index(index)
        expected = self.expected_shard_sha256[shard]
        before = _sha256_file(shard)
        if before != expected:
            raise SourceKVRouteAUHError(
                f"optimizer-accessed shard changed before read: {shard.name}"
            )
        # The routing scan leaves one row-group cached.  Invalidate it so the
        # optimizer consumes bytes inside this before/after hash closure.
        self.dataset._cached_key = None
        self.dataset._cached_rows = None
        row = self.dataset[index]
        after = _sha256_file(shard)
        if after != expected or before != after:
            raise SourceKVRouteAUHError(
                f"optimizer-accessed shard changed during read: {shard.name}"
            )
        audit = {
            "access_ordinal": access_ordinal,
            "row_index": int(index),
            "shard_path": str(shard),
            "expected_sha256": expected,
            "before_read_sha256": before,
            "after_read_sha256": after,
            "cache_invalidated_before_read": True,
            "hash_closed_read": True,
        }
        self.accesses.append(audit)
        return row, dict(audit)

    def finalize(
        self,
        *,
        summary_path: str | Path,
        routing_path: str | Path,
    ) -> dict[str, Any]:
        if len(self.accesses) != MAX_STEPS:
            raise SourceKVRouteAUHError("input-integrity audit requires exact40 reads")
        summary_sha = _sha256_file(summary_path)
        index_sha = _sha256_file(self.index_path)
        routing_sha = _sha256_file(routing_path)
        if (
            summary_sha != PINNED_DATASET_SUMMARY_FILE_SHA256
            or index_sha != PINNED_DATASET_INDEX_SHA256
            or routing_sha != PINNED_ROUTING_SHA256
        ):
            raise SourceKVRouteAUHError(
                "summary, index, or routing changed during training"
            )
        final_shards = []
        for shard in sorted({Path(item["shard_path"]) for item in self.accesses}):
            expected = self.expected_shard_sha256[shard]
            actual = _sha256_file(shard)
            if actual != expected:
                raise SourceKVRouteAUHError(
                    f"optimizer-accessed shard changed after training: {shard.name}"
                )
            final_shards.append(
                {"shard_path": str(shard), "expected_sha256": expected, "final_sha256": actual}
            )
        value = {
            "validated": True,
            "policy": "pinned_index_hash_before_and_after_each_optimizer_read",
            "access_count": len(self.accesses),
            "unique_accessed_shard_count": len(final_shards),
            "accesses": list(self.accesses),
            "accesses_sha256": route_scope.object_sha256(self.accesses),
            "final_accessed_shards": final_shards,
            "dataset_summary_final_sha256": summary_sha,
            "dataset_index_final_sha256": index_sha,
            "routing_final_sha256": routing_sha,
        }
        value["audit_sha256"] = route_scope.object_sha256(value)
        return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train exact-40 81f Bernini CSV-ART V9 on four AUH GPUs"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--preprocessed-parquet-dir", default=PINNED_DATASET_SHARDS
    )
    parser.add_argument("--dataset-summary", default=PINNED_DATASET_SUMMARY)
    parser.add_argument("--routing-jsonl", default=PINNED_ROUTING_JSONL)
    parser.add_argument(
        "--expected-routing-jsonl-sha256", default=PINNED_ROUTING_SHA256
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, choices=(81,), default=81)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--noop-instruction", default=route_batches.EXACT_NOOP_INSTRUCTION
    )
    parser.add_argument("--negative-prompt", default=v5.DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--block-selection", choices=replay.BLOCK_SELECTIONS, default="all"
    )
    parser.add_argument(
        "--experimental-block-ablation",
        action="store_true",
        help="required for non-all carrier scopes; never used by V9-main",
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


def loss_config_from_args(
    args: argparse.Namespace,
) -> objective.SourceKVRouteLossConfig:
    del args
    return objective.SourceKVRouteLossConfig()


def validate_cli(args: argparse.Namespace) -> None:
    if (
        args.num_frames != NUM_FRAMES
        or legacy.LATENT_FRAMES != LATENT_PHASES
        or args.max_steps != MAX_STEPS
        or args.save_every != SAVE_EVERY
        or float(args.learning_rate) != LEARNING_RATE
        or float(args.weight_decay) != WEIGHT_DECAY
        or float(args.max_grad_norm) != MAX_GRAD_NORM
    ):
        raise SourceKVRouteAUHError(
            "V9 pilot fixes 81f/21 phases, exact40/save40, lr=1e-5, "
            "weight_decay=0, and max_grad_norm=1"
        )
    if args.noop_instruction != route_batches.EXACT_NOOP_INSTRUCTION:
        raise SourceKVRouteAUHError("V9 semantic no-op text differs")
    route_batches.validate_noop_instruction(args.noop_instruction)
    if args.negative_prompt != v5.DEFAULT_NEGATIVE_PROMPT:
        raise SourceKVRouteAUHError("V9 negative prompt differs")
    if args.block_selection == "all":
        if args.experimental_block_ablation:
            raise SourceKVRouteAUHError(
                "all-block V9-main cannot be labelled an experimental block ablation"
            )
    elif not args.experimental_block_ablation:
        raise SourceKVRouteAUHError(
            "mid/late carrier scopes require --experimental-block-ablation"
        )
    pinned_paths = {
        "preprocessed_parquet_dir": PINNED_DATASET_SHARDS,
        "dataset_summary": PINNED_DATASET_SUMMARY,
        "routing_jsonl": PINNED_ROUTING_JSONL,
    }
    for name, expected in pinned_paths.items():
        if str(getattr(args, name)) != expected:
            raise SourceKVRouteAUHError(f"V9 pinned path differs: {name}")
    if args.expected_routing_jsonl_sha256 != PINNED_ROUTING_SHA256:
        raise SourceKVRouteAUHError("V9 strict359 routing SHA differs")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if re.fullmatch(r"[0-9a-fA-F]{40}", str(getattr(args, name))) is None:
            raise SourceKVRouteAUHError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
        "expected_routing_jsonl_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))) is None:
            raise SourceKVRouteAUHError(f"{name} must be lowercase SHA-256")
    if args.expected_bernini_commit.lower() != legacy.BERNINI_OFFICIAL_COMMIT:
        raise SourceKVRouteAUHError("Bernini revision differs")
    if args.expected_veomni_commit.lower() != legacy.VEOMNI_TESTED_COMMIT:
        raise SourceKVRouteAUHError("VeOmni revision differs")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise SourceKVRouteAUHError("checkpoint tree differs")
    route_scope.validate_lora_hyperparameters(
        rank=8, alpha=8, hidden_size=1536, dropout=0.0, bias="none"
    )
    loss_config_from_args(args).validate()


def configure_source_kv_gradient_checkpointing(model: Any) -> Mapping[str, Any]:
    """Install the only checkpoint mode that can restore branch invocations."""

    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise SourceKVRouteAUHError("renderer lacks gradient_checkpointing_enable")
    kwargs = {
        "use_reentrant": False,
        "context_fn": replay.source_kv_replay_checkpoint_context_fn,
    }
    enable(gradient_checkpointing_kwargs=kwargs)
    return kwargs


def select_training_query_endpoint(
    endpoints: Mapping[
        str,
        tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    ],
    *,
    query_state_policy: str = QUERY_STATE_POLICY,
) -> tuple[
    Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]
]:
    """Select the pre-registered beta-zero query without touching target beta1.

    This is intentionally isolated from candidate preparation.  V9-main only
    accepts the offline source-tangent arm.  Future self-rollout experiments
    must add a new explicit policy and must remain target-free; silently
    selecting the paired-target endpoint is always forbidden here.
    """

    if query_state_policy != QUERY_STATE_POLICY:
        raise SourceKVRouteAUHError(
            f"unregistered V9 query-state policy: {query_state_policy!r}"
        )
    if set(endpoints) != {"source", "target"}:
        raise SourceKVRouteAUHError("bridge endpoint inventory differs")
    source = endpoints["source"]
    if not isinstance(source, tuple) or len(source) != 4:
        raise SourceKVRouteAUHError("source endpoint payload differs")
    auxiliary = source[3]
    if float(auxiliary.get("bridge_fraction", -1.0)) != 0.0:
        raise SourceKVRouteAUHError("V9 query endpoint is not beta=0")
    return source


def _prepare_source_endpoint_candidate_cpu(
    *,
    raw_row: Mapping[str, Any],
    tokenizer: Any,
    prompt_cleaner: Any,
    system_prompts: Mapping[str, str],
    rope: Any,
    vae_mean: Any,
    vae_std: Any,
    z_dim: int,
    scheduler: Any,
    noop_instruction: str,
    negative_prompt: str,
    process_renderer_sample: Any,
    selected_stratum: Any,
) -> PreparedSourceKVRouteCandidate:
    """Build one beta-zero pair; target clean remains an offline label."""

    try:
        sample = legacy.sanitize_preprocessed_row(raw_row)
        spatial_hw = v6_runtime._spatial_hw_from_sample(sample, z_dim=z_dim)
        _, instruction, _ = v6_runtime._official_t2v_text_fields(
            sample,
            tokenizer=tokenizer,
            prompt_cleaner=prompt_cleaner,
            system_prompts=system_prompts,
        )
        endpoints = v5._prepare_prior_bridge_batches(
            raw_row=raw_row,
            tokenizer=tokenizer,
            rope=rope,
            vae_mean=vae_mean,
            vae_std=vae_std,
            z_dim=z_dim,
            scheduler=scheduler,
            noop_instruction=noop_instruction,
            negative_prompt=negative_prompt,
            minimum_training_sigma=0.1,
            process_renderer_sample=process_renderer_sample,
            selected_stratum=selected_stratum,
        )
    except Exception as error:
        raise SourceKVRouteAUHError(
            f"cannot prepare beta-zero V9 candidate: {error}"
        ) from error
    editor_negative, editor_noop, editor_action, auxiliary = (
        select_training_query_endpoint(endpoints)
    )
    if float(auxiliary.get("bridge_fraction", -1.0)) != 0.0:
        raise SourceKVRouteAUHError("V9 training query must be beta=0 source endpoint")
    negative_text = {
        field: editor_negative[field]
        for field in v6_runtime.branch_geometry.TEXT_FIELDS
    }
    editor_negative = v6_runtime._bind_text_geometry(
        editor_negative, negative_text, label="V9 full-pair negative"
    )
    return PreparedSourceKVRouteCandidate(
        editor_negative=editor_negative,
        editor_noop=editor_noop,
        editor_action=editor_action,
        auxiliary=auxiliary,
        spatial_hw=spatial_hw,
        instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
    )


def _move_source_endpoint_candidate_to_device(
    candidate: PreparedSourceKVRouteCandidate,
    *,
    device: Any,
    noop_instruction: str,
) -> MovedSourceKVRouteCandidate:
    import torch

    try:
        negative = legacy._move_batch(candidate.editor_negative, device)
        noop = legacy._move_batch(candidate.editor_noop, device)
        action = legacy._move_batch(candidate.editor_action, device)
        auxiliary = v4._move_auxiliary_to_device(
            candidate.auxiliary,
            device=device,
            branch_state_mode="source_target_bridge_clean_field",
        )
        v5._assert_same_endpoint_state(negative, noop, action)
        carrier = route_batches.build_source_only_carrier_batch(
            action_pair_batch=action,
            noop_pair_batch=noop,
            noop_instruction=noop_instruction,
        )
    except Exception as error:
        raise SourceKVRouteAUHError(
            f"cannot move/validate V9 candidate: {error}"
        ) from error
    selector = action["vae_latents_mask"].squeeze(0).bool()
    target_tokens = int(selector.sum().item())
    shared_noisy = auxiliary.get("shared_noisy")
    if (
        target_tokens <= 0
        or int((~selector).sum().item()) != target_tokens
        or target_tokens != LATENT_PHASES * math.prod(candidate.spatial_hw)
        or not isinstance(shared_noisy, torch.Tensor)
        or shared_noisy.dtype != torch.float32
        or tuple(shared_noisy.shape[:2]) != (1, target_tokens)
        or float(auxiliary.get("bridge_fraction", -1.0)) != 0.0
    ):
        raise SourceKVRouteAUHError("V9 beta-zero pair geometry differs")
    packed_tail = motion.flatten_velocity_patches(
        action["input_vae_latents"][selector].unsqueeze(0)
    ).float()
    if not torch.equal(packed_tail, shared_noisy):
        raise SourceKVRouteAUHError(
            "beta-zero renderer target-query tail differs from shared noisy state"
        )
    return MovedSourceKVRouteCandidate(
        editor_negative=negative,
        editor_noop=noop,
        editor_action=action,
        carrier=carrier,
        auxiliary=auxiliary,
        spatial_hw=candidate.spatial_hw,
        instruction_sha256=candidate.instruction_sha256,
    )


def _model_forward_view(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Remove selectors and every teacher-only field before renderer access."""

    view = {name: batch[name] for name in route_batches.CARRIER_MODEL_FIELDS}
    forbidden = set(view).intersection(route_batches.FORBIDDEN_EXTERNAL_FIELDS)
    teacher = set(view).intersection(
        {"target_velocity", "target_clean", "paired_target_video", "target_lens"}
    )
    if forbidden or teacher or set(view) != set(route_batches.CARRIER_MODEL_FIELDS):
        raise SourceKVRouteAUHError("renderer model-field view leaked a teacher")
    return view


def canonical_timestep_token(*, timestep: Any, sigma: Any) -> str:
    try:
        timestep_value = float(
            timestep.detach().float().cpu().item()
            if hasattr(timestep, "detach")
            else timestep
        )
        sigma_value = float(
            sigma.detach().float().cpu().item()
            if hasattr(sigma, "detach")
            else sigma
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise SourceKVRouteAUHError("cannot canonicalize timestep/sigma") from error
    if not math.isfinite(timestep_value) or not math.isfinite(sigma_value):
        raise SourceKVRouteAUHError("timestep/sigma token must be finite")
    return f"t={timestep_value:.17g},sigma={sigma_value:.17g}"


def next_integrity_tracked_routed_row(
    *,
    tracker: AccessedShardIntegrityTracker,
    eligible_routes: Sequence[tuple[int, Any]],
    ordinal: int,
) -> tuple[int, Mapping[str, Any], Any, Mapping[str, Any]]:
    """Read the deterministic routed row inside a pinned shard-hash closure."""

    if not eligible_routes:
        raise SourceKVRouteAUHError("eligible route stream is empty")
    row_index, route = eligible_routes[ordinal % len(eligible_routes)]
    row, shard_audit = tracker.read(row_index, access_ordinal=ordinal)
    try:
        iid = v4._iid(row)
    except Exception as error:
        raise SourceKVRouteAUHError(f"optimizer row IID is invalid: {error}") from error
    if iid != route.iid:
        raise SourceKVRouteAUHError(
            "dataset/routing membership changed inside integrity-tracked read"
        )
    return row_index, row, route, shard_audit


def _invoke_full_prediction(
    *,
    renderer: Any,
    batch: Mapping[str, Any],
    cache_bank: replay.SourceKVCacheBank,
    mode: str,
    branch_tag: str,
    generation: int,
    step_index: int,
    timestep_token: str,
    rank: int,
    ulysses_size: int,
    full_velocity_fn: Callable[[Any, Mapping[str, Any]], Any],
) -> Any:
    with replay.source_kv_replay_invocation(
        cache_bank,
        mode=mode,
        branch_tag=branch_tag,
        generation=generation,
        step_index=step_index,
        timestep_token=timestep_token,
        rank=rank,
        ulysses_size=ulysses_size,
    ):
        return full_velocity_fn(renderer, _model_forward_view(batch))


def _run_six_forward_cell(
    *,
    renderer: Any,
    adapter_controller: Any,
    candidate: MovedSourceKVRouteCandidate,
    cache_bank: replay.SourceKVCacheBank,
    generation: int,
    step_index: int,
    timestep_token: str,
    rank: int,
    ulysses_size: int,
    loss_config: objective.SourceKVRouteLossConfig,
    require_fresh_parity: bool,
    full_velocity_fn: Callable[[Any, Mapping[str, Any]], Any] = (
        route_batches.renderer_full_velocity_prediction
    ),
) -> SixForwardCellResult:
    """Run exactly one carrier capture, three frozen, and two adapted pairs."""

    import torch

    auxiliary = candidate.auxiliary
    shared_noisy = auxiliary["shared_noisy"]
    sigma = auxiliary["sigma"]
    observed: list[str] = []
    with torch.no_grad():
        with adapter_controller.disable_adapter():
            carrier_output = _invoke_full_prediction(
                renderer=renderer,
                batch=candidate.carrier.batch,
                cache_bank=cache_bank,
                mode=replay.CAPTURE_MODE,
                branch_tag=replay.CAPTURE_BRANCH_TAG,
                generation=generation,
                step_index=step_index,
                timestep_token=timestep_token,
                rank=rank,
                ulysses_size=ulysses_size,
                full_velocity_fn=full_velocity_fn,
            )
            observed.append(FORWARD_ORDER[0])
            frozen_full: dict[str, Any] = {}
            for name, tag, batch in (
                ("negative", "frozen_negative", candidate.editor_negative),
                ("noop", "frozen_noop", candidate.editor_noop),
                ("action", "frozen_action", candidate.editor_action),
            ):
                frozen_full[name] = _invoke_full_prediction(
                    renderer=renderer,
                    batch=batch,
                    cache_bank=cache_bank,
                    mode=replay.REPLAY_MODE,
                    branch_tag=tag,
                    generation=generation,
                    step_index=step_index,
                    timestep_token=timestep_token,
                    rank=rank,
                    ulysses_size=ulysses_size,
                    full_velocity_fn=full_velocity_fn,
                )
                observed.append(f"frozen_{name}_full_pair")

    adapted_full: dict[str, Any] = {}
    for name, tag, batch in (
        ("noop", "adapted_noop", candidate.editor_noop),
        ("action", "adapted_action", candidate.editor_action),
    ):
        adapted_full[name] = _invoke_full_prediction(
            renderer=renderer,
            batch=batch,
            cache_bank=cache_bank,
            mode=replay.REPLAY_MODE,
            branch_tag=tag,
            generation=generation,
            step_index=step_index,
            timestep_token=timestep_token,
            rank=rank,
            ulysses_size=ulysses_size,
            full_velocity_fn=full_velocity_fn,
        )
        observed.append(f"adapted_{name}_full_pair")

    if tuple(observed) != FORWARD_ORDER:
        raise SourceKVRouteAUHError(f"six-forward order differs: {observed}")
    if (
        not isinstance(carrier_output, torch.Tensor)
        or carrier_output.dtype != torch.bfloat16
        or tuple(carrier_output.shape[:2])
        != (1, candidate.carrier.source_tokens)
        or carrier_output.requires_grad
        or not bool(torch.isfinite(carrier_output).all())
    ):
        raise SourceKVRouteAUHError("source-only carrier forward output differs")

    branch_batches = {
        "negative": candidate.editor_negative,
        "noop": candidate.editor_noop,
        "action": candidate.editor_action,
    }
    frozen_velocity = {
        name: route_batches.select_target_velocity(value, branch_batches[name])
        for name, value in frozen_full.items()
    }
    adapted_velocity = {
        name: route_batches.select_target_velocity(value, branch_batches[name])
        for name, value in adapted_full.items()
    }
    velocities = (
        frozen_velocity["negative"],
        frozen_velocity["noop"],
        frozen_velocity["action"],
        adapted_velocity["noop"],
        adapted_velocity["action"],
    )
    if any(
        value.dtype != torch.bfloat16
        or tuple(value.shape) != tuple(shared_noisy.shape)
        or not bool(torch.isfinite(value).all())
        for value in velocities
    ):
        raise SourceKVRouteAUHError(
            "five pair branches must be finite native-BF16 target fields"
        )
    if any(value.requires_grad for value in velocities[:3]):
        raise SourceKVRouteAUHError("frozen replay branch retained autograd")
    if not velocities[3].requires_grad or not velocities[4].requires_grad:
        raise SourceKVRouteAUHError("both adapted replay branches require graphs")

    fresh_noop_exact = torch.equal(velocities[1], velocities[3])
    fresh_action_exact = torch.equal(velocities[2], velocities[4])
    if require_fresh_parity and not (fresh_noop_exact and fresh_action_exact):
        raise SourceKVRouteAUHError(
            "fresh LoRA step-zero output differs from adapter-off base"
        )

    try:
        frozen_negative, frozen_noop = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=velocities[0],
            conditional_velocity=velocities[1],
        )
        negative_action, frozen_action = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=velocities[0],
            conditional_velocity=velocities[2],
        )
        negative_adapted_noop, adapted_noop = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=velocities[0],
            conditional_velocity=velocities[3],
        )
        negative_adapted_action, adapted_action = v5._guided_clean(
            shared_noisy=shared_noisy,
            sigma=sigma,
            negative_velocity=velocities[0],
            conditional_velocity=velocities[4],
        )
        if not all(
            torch.equal(frozen_negative, value)
            for value in (
                negative_action,
                negative_adapted_noop,
                negative_adapted_action,
            )
        ):
            raise SourceKVRouteAUHError("APG negative clean field differs by branch")
        fields = objective.RouteCleanFields(
            frozen_noop=frozen_noop.detach().float(),
            frozen_action=frozen_action.detach().float(),
            adapted_noop=adapted_noop.float(),
            adapted_action=adapted_action.float(),
            source_clean=v5._as_phase_grid(auxiliary["source_clean"].float()).detach(),
            target_clean=v5._as_phase_grid(auxiliary["target_clean"].float()).detach(),
        )
        loss_result = objective.compute_source_kv_route_objective(
            fields, sigma=sigma, config=loss_config
        )
    except Exception as error:
        if isinstance(error, SourceKVRouteAUHError):
            raise
        raise SourceKVRouteAUHError(f"V9 clean-field objective failed: {error}") from error
    return SixForwardCellResult(
        loss_result=loss_result,
        forward_order=tuple(observed),
        fresh_parity_checked=bool(require_fresh_parity),
        fresh_noop_exact=bool(fresh_noop_exact),
        fresh_action_exact=bool(fresh_action_exact),
        carrier_output_shape=tuple(int(value) for value in carrier_output.shape),
    )


def _counter_delta(
    after: Mapping[str, Any], before: Mapping[str, Any], key: str
) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def audit_cache_after_backward(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    selected_blocks: Sequence[int],
) -> dict[str, Any]:
    blocks = list(selected_blocks)
    count = len(blocks)
    expected_branch = {
        "frozen_negative": count,
        "frozen_noop": count,
        "frozen_action": count,
        "adapted_noop": 2 * count,
        "adapted_action": 2 * count,
    }
    branch_delta = {
        key: int(after["replay_branch_counts"].get(key, 0))
        - int(before["replay_branch_counts"].get(key, 0))
        for key in expected_branch
    }
    phase_expected = {
        replay.EAGER_EXECUTION: 3 * count,
        replay.CHECKPOINT_FORWARD: 2 * count,
        replay.CHECKPOINT_RECOMPUTE: 2 * count,
    }
    phase_delta = {
        key: int(after["replay_phase_counts"].get(key, 0))
        - int(before["replay_phase_counts"].get(key, 0))
        for key in phase_expected
    }
    # Bernini wraps every one of its 30 transformer blocks in a checkpoint.
    # Processor replay lookups occur only in selected attention blocks, but the
    # context_fn is invoked for all checkpointed blocks.  These quantities are
    # equal only in the all30 main arm; keep them separate for explicit
    # mid/late ablations.
    checkpoint_block_count = replay.EXPECTED_BLOCK_COUNT
    checkpoint_delta = {
        key: int(after["checkpoint_context_counts"].get(key, 0))
        - int(before["checkpoint_context_counts"].get(key, 0))
        for key in (replay.CHECKPOINT_FORWARD, replay.CHECKPOINT_RECOMPUTE)
    }
    audit = {
        "selected_block_count": count,
        "selected_blocks": blocks,
        "capture_calls_delta": _counter_delta(after, before, "capture_calls"),
        "replay_lookups_delta": _counter_delta(after, before, "replay_lookups"),
        "replay_branch_count_delta": branch_delta,
        "replay_phase_count_delta": phase_delta,
        "checkpoint_context_count_delta": checkpoint_delta,
        "cache_complete_before_clear": bool(after.get("complete")),
        "captured_blocks_before_clear": list(after.get("captured_blocks", [])),
        "all_entries_detached": all(
            bool(entry.get("detached")) for entry in after.get("entries", [])
        ),
        "backward_recompute_observed": checkpoint_delta[
            replay.CHECKPOINT_RECOMPUTE
        ]
        == 2 * checkpoint_block_count,
        "checkpointed_transformer_block_count": checkpoint_block_count,
    }
    expected = {
        "capture": count,
        "replay": 7 * count,
        "branch": expected_branch,
        "phase": phase_expected,
        "checkpoint": {
            replay.CHECKPOINT_FORWARD: 2 * checkpoint_block_count,
            replay.CHECKPOINT_RECOMPUTE: 2 * checkpoint_block_count,
        },
    }
    if (
        audit["capture_calls_delta"] != expected["capture"]
        or audit["replay_lookups_delta"] != expected["replay"]
        or branch_delta != expected["branch"]
        or phase_delta != expected["phase"]
        or checkpoint_delta != expected["checkpoint"]
        or audit["cache_complete_before_clear"] is not True
        or audit["captured_blocks_before_clear"] != blocks
        or audit["all_entries_detached"] is not True
    ):
        raise SourceKVRouteAUHError(f"source K/V lifecycle audit differs: {audit}")
    return audit


def audit_cache_after_clear(
    *, before_clear: Mapping[str, Any], after_clear: Mapping[str, Any]
) -> dict[str, Any]:
    unchanged = all(
        before_clear.get(name) == after_clear.get(name)
        for name in (
            "capture_calls",
            "replay_lookups",
            "replay_branch_counts",
            "replay_phase_counts",
            "checkpoint_context_counts",
        )
    )
    audit = {
        "cleared_after_backward": True,
        "identity_after_clear": after_clear.get("identity"),
        "complete_after_clear": bool(after_clear.get("complete")),
        "captured_blocks_after_clear": list(after_clear.get("captured_blocks", [])),
        "counters_preserved": unchanged,
        "retired_identity_delta": int(after_clear.get("retired_identity_count", 0))
        - int(before_clear.get("retired_identity_count", 0)),
    }
    if audit != {
        "cleared_after_backward": True,
        "identity_after_clear": None,
        "complete_after_clear": False,
        "captured_blocks_after_clear": [],
        "counters_preserved": True,
        "retired_identity_delta": 1,
    }:
        raise SourceKVRouteAUHError(f"cache clear audit differs: {audit}")
    return audit


def audit_trainable_gradients(
    named_trainable: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    import torch

    if len(named_trainable) != route_scope.EXPECTED_ADAPTER_TENSOR_COUNT:
        raise SourceKVRouteAUHError("V9 requires exactly 184 trainable tensors")
    names = [name for name, _ in named_trainable]
    if len(set(names)) != len(names):
        raise SourceKVRouteAUHError("trainable parameter names are duplicated")
    parameter_count = sum(int(parameter.numel()) for _, parameter in named_trainable)
    if parameter_count != route_scope.EXPECTED_TRAINABLE_PARAMETER_COUNT:
        raise SourceKVRouteAUHError("V9 trainable parameter count differs")
    missing = [name for name, parameter in named_trainable if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in named_trainable
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all())
    ]
    if missing or nonfinite:
        raise SourceKVRouteAUHError(
            f"V9 gradient audit failed: missing={missing[:4]} nonfinite={nonfinite[:4]}"
        )
    nonzero = sum(
        int(bool(parameter.grad.detach().abs().max() > 0))
        for _, parameter in named_trainable
    )
    return {
        "trainable_tensor_count": len(named_trainable),
        "trainable_parameter_count": parameter_count,
        "all_gradients_present": True,
        "all_gradients_finite": True,
        "nonzero_gradient_tensor_count": nonzero,
        "parameter_names_sha256": route_scope.object_sha256(names),
    }


def audit_optimizer_state(
    *,
    optimizer: Any,
    named_trainable: Sequence[tuple[str, Any]],
    expected_step: int,
) -> dict[str, Any]:
    import torch

    parameters = [parameter for _, parameter in named_trainable]
    missing = [parameter for parameter in parameters if parameter not in optimizer.state]
    if missing or len(optimizer.state) != route_scope.EXPECTED_ADAPTER_TENSOR_COUNT:
        raise SourceKVRouteAUHError("AdamW does not hold exactly 184 parameter states")
    observed_steps: set[int] = set()
    for parameter in parameters:
        state = optimizer.state[parameter]
        step = state.get("step")
        step_value = int(step.detach().cpu().item() if hasattr(step, "detach") else step)
        observed_steps.add(step_value)
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(parameter.shape)
                or not bool(torch.isfinite(value).all())
            ):
                raise SourceKVRouteAUHError(f"AdamW state {name} differs")
    if observed_steps != {int(expected_step)}:
        raise SourceKVRouteAUHError(
            f"AdamW state steps {sorted(observed_steps)} != {expected_step}"
        )
    return {
        "state_parameter_count": len(optimizer.state),
        "state_step_values": sorted(observed_steps),
        "no_moment_reset": True,
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
    }


def execute_source_kv_route_optimizer_step(
    *,
    renderer: Any,
    adapter_controller: Any,
    candidate: MovedSourceKVRouteCandidate,
    cache_bank: replay.SourceKVCacheBank,
    named_trainable: Sequence[tuple[str, Any]],
    optimizer: Any,
    generation: int,
    step_index: int,
    timestep_token: str,
    rank: int,
    ulysses_size: int,
    loss_config: objective.SourceKVRouteLossConfig,
    max_grad_norm: float = MAX_GRAD_NORM,
    require_fresh_parity: bool = False,
    full_velocity_fn: Callable[[Any, Mapping[str, Any]], Any] = (
        route_batches.renderer_full_velocity_prediction
    ),
    gradient_reduce_fn: Callable[[Sequence[tuple[str, Any]]], float] = (
        legacy.all_reduce_lora_gradients
    ),
) -> OptimizerStepResult:
    """Own forward/backward/cache retirement/optimizer as one auditable unit."""

    import torch

    if cache_bank.identity is not None:
        raise SourceKVRouteAUHError("cache must be empty before an optimizer candidate")
    cache_before = cache_bank.receipt()
    optimizer.zero_grad(set_to_none=True)
    cell = _run_six_forward_cell(
        renderer=renderer,
        adapter_controller=adapter_controller,
        candidate=candidate,
        cache_bank=cache_bank,
        generation=generation,
        step_index=step_index,
        timestep_token=timestep_token,
        rank=rank,
        ulysses_size=ulysses_size,
        loss_config=loss_config,
        require_fresh_parity=require_fresh_parity,
        full_velocity_fn=full_velocity_fn,
    )
    if not bool(torch.isfinite(cell.loss_result.total.detach())):
        raise SourceKVRouteAUHError("V9 loss is non-finite before backward")

    # Critical ordering: checkpoint recompute still needs the live bank here.
    cell.loss_result.total.backward()
    try:
        gradient_norm = float(gradient_reduce_fn(named_trainable))
    except Exception as error:
        raise SourceKVRouteAUHError(f"LoRA gradient reduction failed: {error}") from error
    gradient_audit = audit_trainable_gradients(named_trainable)
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise SourceKVRouteAUHError("V9 requires a finite positive gradient norm")
    gradient_audit = {
        **gradient_audit,
        "global_l2_norm": gradient_norm,
        "positive_global_l2_norm": True,
    }

    cache_after_backward = cache_bank.receipt()
    cache_backward_audit = audit_cache_after_backward(
        before=cache_before,
        after=cache_after_backward,
        selected_blocks=cache_bank.selected_block_indices,
    )
    cache_bank.clear()
    cache_clear_audit = audit_cache_after_clear(
        before_clear=cache_after_backward,
        after_clear=cache_bank.receipt(),
    )

    clipped_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in named_trainable], float(max_grad_norm)
    )
    if not math.isfinite(float(clipped_norm)):
        raise SourceKVRouteAUHError("gradient clipping returned non-finite norm")
    optimizer.step()
    optimizer_audit = audit_optimizer_state(
        optimizer=optimizer,
        named_trainable=named_trainable,
        expected_step=generation + 1,
    )
    metrics = objective.detached_objective_metrics(cell.loss_result)
    target_q = cell.loss_result.diagnostics.target_quotient.detach()
    record = {
        "optimizer_step": generation + 1,
        "sigma_schedule_index": step_index,
        "timestep_token": timestep_token,
        "forward_order": list(cell.forward_order),
        "forwards_per_candidate": 6,
        "graph_forwards_per_candidate": 2,
        "source_only_carrier_forwards": 1,
        "paired_target_model_forward_access": False,
        "paired_target_role": "detached_objective_label_only",
        "model_forward_fields": sorted(route_batches.CARRIER_MODEL_FIELDS),
        "fresh_parity_checked": cell.fresh_parity_checked,
        "fresh_noop_exact": cell.fresh_noop_exact,
        "fresh_action_exact": cell.fresh_action_exact,
        "preclip_gradient_norm": gradient_norm,
        "gradient_audit": gradient_audit,
        "optimizer_audit": optimizer_audit,
        "cache_after_backward_audit": cache_backward_audit,
        "cache_after_clear_audit": cache_clear_audit,
        "target_quotient_rms": float(target_q.float().square().mean().sqrt().cpu()),
        "target_energy_retention": metrics["target_energy_retention"],
        "target_clipped_fraction": metrics["target_clipped_fraction"],
        "loss_metrics": metrics,
    }
    return OptimizerStepResult(cell=cell, record=record)


def validate_exact40_step_audit(
    step_audit: Sequence[Mapping[str, Any]], *, block_selection: str
) -> dict[str, Any]:
    if len(step_audit) != MAX_STEPS:
        raise SourceKVRouteAUHError("formal V9 receipt requires exactly 40 steps")
    expected_blocks = replay.resolve_block_indices(30, block_selection)
    expected_indices = list(range(MAX_STEPS))
    actual_indices = [int(record["sigma_schedule_index"]) for record in step_audit]
    if actual_indices != expected_indices:
        raise SourceKVRouteAUHError("exact40 sigma schedule indices differ")
    for index, record in enumerate(step_audit):
        cache = record.get("cache_after_backward_audit", {})
        clear = record.get("cache_after_clear_audit", {})
        gradients = record.get("gradient_audit", {})
        optimizer = record.get("optimizer_audit", {})
        shard = record.get("input_shard_integrity", {})
        if (
            record.get("forward_order") != list(FORWARD_ORDER)
            or record.get("forwards_per_candidate") != 6
            or record.get("graph_forwards_per_candidate") != 2
            or record.get("paired_target_model_forward_access") is not False
            or gradients.get("trainable_tensor_count") != 184
            or gradients.get("all_gradients_finite") is not True
            or gradients.get("positive_global_l2_norm") is not True
            or float(gradients.get("global_l2_norm", 0.0)) <= 0.0
            or optimizer.get("state_parameter_count") != 184
            or optimizer.get("state_step_values") != [index + 1]
            or optimizer.get("no_moment_reset") is not True
            or record.get("target_energy_retention") != 1.0
            or record.get("target_clipped_fraction") != 0.0
            or cache.get("selected_blocks") != list(expected_blocks)
            or cache.get("capture_calls_delta") != len(expected_blocks)
            or cache.get("replay_lookups_delta") != 7 * len(expected_blocks)
            or cache.get("backward_recompute_observed") is not True
            or clear.get("cleared_after_backward") is not True
            or clear.get("identity_after_clear") is not None
            or shard.get("access_ordinal") != index
            or shard.get("row_index") != record.get("row_index")
            or shard.get("hash_closed_read") is not True
            or shard.get("cache_invalidated_before_read") is not True
            or shard.get("before_read_sha256") != shard.get("expected_sha256")
            or shard.get("after_read_sha256") != shard.get("expected_sha256")
        ):
            raise SourceKVRouteAUHError(f"formal V9 step audit failed at {index}")
    if step_audit[0].get("fresh_parity_checked") is not True:
        raise SourceKVRouteAUHError("step zero lacks fresh-init parity evidence")
    if any(
        record.get("fresh_parity_checked") is True for record in step_audit[1:]
    ):
        raise SourceKVRouteAUHError("fresh parity may only be claimed at step zero")
    return {
        "validated": True,
        "step_count": 40,
        "sigma_schedule_indices": expected_indices,
        "all30_main": block_selection == "all",
        "selected_block_count": len(expected_blocks),
        "step_audit_sha256": route_scope.object_sha256(list(step_audit)),
    }


def _immutable_contract(
    *,
    args: argparse.Namespace,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    router: Any,
    eligible_routes: Sequence[tuple[int, Any]],
    scope_manifest: Mapping[str, Any],
    checkpoint: Path,
    loss_config: objective.SourceKVRouteLossConfig,
) -> dict[str, Any]:
    blocks = replay.resolve_block_indices(30, args.block_selection)
    value = {
        "method": METHOD_NAME,
        "schema_version": RECEIPT_SCHEMA,
        "run_role": (
            "v9_main" if args.block_selection == "all" else "experimental_ablation"
        ),
        "method_source_revision": args.method_source_revision.lower(),
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "bernini_commit": args.expected_bernini_commit.lower(),
        "veomni_commit": args.expected_veomni_commit.lower(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_content_manifest_sha256": CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "checkpoint_content_file_count": CHECKPOINT_CONTENT_FILE_COUNT,
        "dataset": {
            "shards_path": PINNED_DATASET_SHARDS,
            "summary_path": PINNED_DATASET_SUMMARY,
            "summary_file_sha256": PINNED_DATASET_SUMMARY_FILE_SHA256,
            "index_path": PINNED_DATASET_INDEX,
            "index_sha256": PINNED_DATASET_INDEX_SHA256,
            "signature": dataset.signature,
            "summary_digest": dataset_summary["summary_digest"],
            "rows": PINNED_DATASET_ROWS,
            "eligible_rows": PINNED_ELIGIBLE_ROWS,
            "routing_path": PINNED_ROUTING_JSONL,
            "routing_sha256": PINNED_ROUTING_SHA256,
            "routing_digest": router.digest,
            "eligible_route_stream_sha256": legacy.object_sha256(
                [
                    {
                        "row_index": row_index,
                        "iid": route.iid,
                        "tier": route.tier,
                        "full_target_weight": route.full_target_weight,
                    }
                    for row_index, route in eligible_routes
                ]
            ),
            "runtime_integrity_policy": {
                "routing_scan_cache_is_not_training_input": True,
                "invalidate_parquet_row_group_cache_before_each_optimizer_read": True,
                "hash_each_accessed_shard_before_and_after_read": True,
                "bind_shard_hash_to_pinned_index": True,
                "rehash_all_accessed_shards_after_exact40": True,
                "rehash_summary_index_routing_after_exact40": True,
            },
        },
        "frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "max_steps": MAX_STEPS,
        "save_every": SAVE_EVERY,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "lora_scope_manifest": dict(scope_manifest),
        "fresh_initialization": route_scope.fresh_initialization_declaration(),
        "checkpointing": {
            "use_reentrant": False,
            "context_fn": "source_kv_replay_checkpoint_context_fn",
            "cache_lifetime": "capture_through_loss_backward_then_audit_and_clear",
        },
        "carrier": {
            "selection": args.block_selection,
            "selected_blocks": list(blocks),
            "selected_block_count": len(blocks),
            "main_requires_all30": args.block_selection == "all",
            "experimental_ablation": bool(args.experimental_block_ablation),
            "source_only": True,
            "post_rope": True,
            "decoded_output": False,
        },
        "forward_order": list(FORWARD_ORDER),
        "forwards_per_candidate": 6,
        "graph_forwards_per_candidate": 2,
        "expected_all30_counts": {
            "capture_calls": 30,
            "forward_replay_lookups": 150,
            "backward_recompute_replay_lookups": 60,
            "total_replay_lookups": 210,
        },
        "training_diffusion_query": "source(beta=0)",
        "query_state_policy": {
            "name": QUERY_STATE_POLICY,
            "pre_registered_main_arm": True,
            "training_endpoint": "source(beta=0)",
            "training_tail_formula": QUERY_STATE_TRAIN_FORMULA,
            "deployment_tail_after_first_step": QUERY_STATE_DEPLOYMENT_FORMULA,
            "query_state_train_test_matched": False,
            "exposure_gap": (
                "offline beta0 queries do not cover scheduler states produced by "
                "the edited trajectory after deployment step zero"
            ),
            "paired_target_tail_used": False,
            "future_ablation": QUERY_STATE_FOLLOWUP_ARM,
            "future_ablation_trigger": (
                "deployment dog identity-consistency gate fails after exact40 pilot"
            ),
        },
        "paired_target_role": "detached_executable_objective_label_only",
        "paired_target_used_as_model_condition": False,
        "model_forward_fields": sorted(route_batches.CARRIER_MODEL_FIELDS),
        "objective_contract": objective.objective_contract(loss_config),
        "optimizer_contract": {
            "type": "AdamW",
            "trainable_tensor_count": 184,
            "trainable_parameter_count": 2_260_992,
            "state_count_at_step40": 184,
            "state_step_at_step40": 40,
            "moment_reset": False,
        },
        "target_clipping": False,
        "target_energy_retention": 1.0,
        "inference_conditions": ["source_video", "action_instruction"],
        "training_only_conditions": ["paired_target_video"],
        "forbidden_inference_conditions": list(
            objective.FORBIDDEN_INFERENCE_CONDITIONS
        ),
        "external_mask_track_flow_pose_trajectory": False,
        "first_frame_anchor": False,
        "generator_forwards": 0,
        "sigma_schedule": "exact_40_step_flow_shift_5",
        "sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "resume_integrated": False,
        "production_claim_forbidden": True,
        "limitations": [
            "exact40 is a low-cost engineering pilot, not deployment efficacy evidence",
            "beta0 source-tangent training has an explicit autoregressive exposure gap",
            "paired-target labels do not close the deployment query-state gap",
        ],
    }
    return {"value": value, "digest": route_scope.object_sha256(value)}


def _optimizer_payload(
    *,
    optimizer: Any,
    global_step: int,
    immutable: Mapping[str, Any],
    parameter_names: Sequence[str],
    step_audit: Sequence[Mapping[str, Any]],
    optimizer_audit: Mapping[str, Any],
    input_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": OPTIMIZER_SCHEMA,
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "immutable_contract": dict(immutable),
        "parameter_names": list(parameter_names),
        "optimizer_audit": dict(optimizer_audit),
        "input_integrity": dict(input_integrity),
        "step_audit": list(step_audit),
        "step_audit_sha256": route_scope.object_sha256(list(step_audit)),
        "resume_integrated": False,
    }


def _build_receipt(
    *,
    args: argparse.Namespace,
    global_step: int,
    step_audit: Sequence[Mapping[str, Any]],
    exact40_audit: Mapping[str, Any],
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    router: Any,
    checkpoint: Path,
    bernini_revision: str,
    veomni_revision: str,
    distributed: Any,
    backend: str,
    scope_manifest: Mapping[str, Any],
    named_trainable: Sequence[tuple[str, Any]],
    initialization_digest: str,
    immutable: Mapping[str, Any],
    optimizer_payload: Mapping[str, Any],
    input_integrity: Mapping[str, Any],
    transformers_version: str,
) -> dict[str, Any]:
    final_optimizer = step_audit[-1]["optimizer_audit"]
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "global_step": global_step,
        "formal_exact40_complete": global_step == 40,
        "exact40_audit": dict(exact40_audit),
        "step_audit": list(step_audit),
        "step_audit_sha256": route_scope.object_sha256(list(step_audit)),
        "immutable_contract": dict(immutable),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint": {
            "path": str(checkpoint),
            "tree_sha256": args.expected_checkpoint_tree_sha256,
        },
        "dataset": {
            "path": str(dataset.root),
            "rows": len(dataset),
            "signature": dataset.signature,
            "summary": dict(dataset_summary),
            "routing": router.receipt(),
            "input_integrity": dict(input_integrity),
        },
        "adapter": {
            "scope_manifest": dict(scope_manifest),
            "target_module_count": 92,
            "adapter_tensor_count": 184,
            "trainable_parameter_count": sum(
                int(parameter.numel()) for _, parameter in named_trainable
            ),
            "initialization_digest": initialization_digest,
            "checkpoint_parameter_digest": v4._checkpoint_parameter_digest(
                named_trainable
            ),
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_gradient_norm": MAX_GRAD_NORM,
            "state_parameter_count": final_optimizer["state_parameter_count"],
            "state_step_values": final_optimizer["state_step_values"],
            "no_moment_reset": final_optimizer["no_moment_reset"],
            "checkpoint_state_digest": v4._stable_recursive_digest(
                optimizer_payload
            ),
        },
        "distributed": {
            "world_size": distributed.world_size,
            "ulysses_size": distributed.ulysses_size,
            "backend": backend,
            "same_pair_all_ranks": True,
            "explicit_lora_gradient_all_reduce": True,
        },
        "transformers_version": transformers_version,
        "inference_conditions": ["source_video", "action_instruction"],
        "training_only_conditions": ["paired_target_video"],
        "paired_target_model_forward_access": False,
        "external_mask_track_flow_pose_trajectory": False,
        "first_frame_anchor": False,
        "experimental_training": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "query_state_policy": immutable["value"]["query_state_policy"],
        "limitations": immutable["value"]["limitations"],
        "resume_integrated": False,
        "artifact_validation": {
            "schema_version": ARTIFACT_VALIDATION_SCHEMA,
            "verified": False,
            "status": "pending_atomic_artifact_roundtrip_validation",
            "runtime_adapter_loader_verified": False,
        },
    }
    receipt["receipt_digest"] = route_scope.object_sha256(receipt)
    return receipt


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_torch_artifact(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch <2.6 compatibility
        return torch.load(path, map_location="cpu")


def validate_optimizer_artifact_roundtrip(
    *,
    optimizer_path: Path,
    optimizer_payload: Mapping[str, Any],
    named_trainable: Sequence[tuple[str, Any]],
    expected_step: int,
) -> dict[str, Any]:
    """Deserialize optimizer.pt and exercise a fresh AdamW load_state_dict."""

    import torch

    loaded = _load_torch_artifact(optimizer_path)
    if not isinstance(loaded, Mapping):
        raise SourceKVRouteAUHError("optimizer artifact is not a mapping")
    expected_names = [name for name, _ in named_trainable]
    if (
        loaded.get("schema_version") != OPTIMIZER_SCHEMA
        or loaded.get("global_step") != expected_step
        or loaded.get("parameter_names") != expected_names
    ):
        raise SourceKVRouteAUHError("deserialized optimizer identity differs")
    expected_payload_digest = v4._stable_recursive_digest(optimizer_payload)
    loaded_payload_digest = v4._stable_recursive_digest(loaded)
    if loaded_payload_digest != expected_payload_digest:
        raise SourceKVRouteAUHError(
            "deserialized optimizer payload differs from pre-save logical state"
        )
    loaded_state = loaded.get("optimizer")
    if not isinstance(loaded_state, Mapping):
        raise SourceKVRouteAUHError("optimizer artifact lacks optimizer state_dict")

    fresh_named = [
        (
            name,
            torch.nn.Parameter(
                torch.zeros_like(parameter.detach(), device="cpu"),
                requires_grad=True,
            ),
        )
        for name, parameter in named_trainable
    ]
    reloaded_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in fresh_named],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    try:
        reloaded_optimizer.load_state_dict(loaded_state)
    except Exception as error:
        raise SourceKVRouteAUHError(
            f"fresh AdamW load_state_dict failed: {error}"
        ) from error
    reload_audit = audit_optimizer_state(
        optimizer=reloaded_optimizer,
        named_trainable=fresh_named,
        expected_step=expected_step,
    )
    reloaded_state_digest = v4._stable_recursive_digest(
        reloaded_optimizer.state_dict()
    )
    serialized_state_digest = v4._stable_recursive_digest(loaded_state)
    if reloaded_state_digest != serialized_state_digest:
        raise SourceKVRouteAUHError(
            "fresh AdamW state_dict differs after load_state_dict"
        )
    return {
        "torch_deserialize_verified": True,
        "fresh_optimizer_load_state_dict_verified": True,
        "optimizer_state_logical_equality_verified": True,
        "optimizer_payload_logical_digest": loaded_payload_digest,
        "serialized_optimizer_state_logical_digest": serialized_state_digest,
        "reloaded_optimizer_state_logical_digest": reloaded_state_digest,
        "state_parameter_count": reload_audit["state_parameter_count"],
        "state_step_values": reload_audit["state_step_values"],
        "trainable_parameter_count": sum(
            int(parameter.numel()) for _, parameter in fresh_named
        ),
    }


def _save_checkpoint_with_artifact_roundtrip(
    *,
    model: Any,
    optimizer_payload: Mapping[str, Any],
    named_trainable: Sequence[tuple[str, Any]],
    output: Path,
    global_step: int,
    receipt: Mapping[str, Any],
    rank: int,
) -> Path:
    import torch
    import torch.distributed as dist

    final = output / f"checkpoint-{global_step:08d}"
    if rank == 0:
        if final.exists():
            raise SourceKVRouteAUHError(f"refusing to overwrite checkpoint: {final}")
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / f".{final.name}.tmp-{os.getpid()}"
        if temporary.exists():
            raise SourceKVRouteAUHError(f"stale temporary checkpoint: {temporary}")
        temporary.mkdir()
        model.save_pretrained(temporary / "adapter", safe_serialization=True)
        torch.save(optimizer_payload, temporary / "optimizer.pt")

        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file

        current = {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(model).items()
        }
        saved_path = temporary / "adapter" / "adapter_model.safetensors"
        saved = load_file(str(saved_path), device="cpu")
        saved_validation = route_scope.validate_adapter_state(saved)
        if set(saved) != set(current) or any(
            not torch.equal(saved[key], current[key]) for key in saved
        ):
            raise SourceKVRouteAUHError(
                "saved adapter tensor file differs from runtime PEFT state"
            )
        current_parameter_digest = v4._checkpoint_parameter_digest(named_trainable)
        if current_parameter_digest != receipt["adapter"]["checkpoint_parameter_digest"]:
            raise SourceKVRouteAUHError(
                "runtime adapter parameters changed before artifact validation"
            )
        optimizer_path = temporary / "optimizer.pt"
        optimizer_roundtrip = validate_optimizer_artifact_roundtrip(
            optimizer_path=optimizer_path,
            optimizer_payload=optimizer_payload,
            named_trainable=named_trainable,
            expected_step=global_step,
        )
        if (
            optimizer_roundtrip["optimizer_payload_logical_digest"]
            != receipt["optimizer"]["checkpoint_state_digest"]
        ):
            raise SourceKVRouteAUHError(
                "optimizer artifact digest differs from receipt declaration"
            )
        adapter_config_path = temporary / "adapter" / "adapter_config.json"
        if not adapter_config_path.is_file() or adapter_config_path.is_symlink():
            raise SourceKVRouteAUHError("saved adapter config is not a plain file")
        completed = json.loads(json.dumps(receipt))
        completed.pop("receipt_digest", None)
        completed["artifact_validation"] = {
            "schema_version": ARTIFACT_VALIDATION_SCHEMA,
            "verified": True,
            "status": (
                "post_save_adapter_tensor_file_roundtrip_and_"
                "optimizer_load_state_dict_complete"
            ),
            "validation_scope": (
                "adapter_safetensors_file_roundtrip_plus_fresh_AdamW_"
                "deserialize_and_load_state_dict"
            ),
            "adapter_tensor_file_roundtrip_verified": True,
            "adapter_tensor_file_runtime_equality": True,
            "runtime_adapter_loader_verified": False,
            "fresh_base_peft_from_pretrained_verified": False,
            "deployment_loader_claim_forbidden": True,
            "adapter_tensor_count": saved_validation["adapter_tensor_count"],
            "trainable_parameter_count": saved_validation[
                "trainable_parameter_count"
            ],
            "adapter_model_sha256": _sha256_file(saved_path),
            "adapter_config_sha256": _sha256_file(adapter_config_path),
            "optimizer_checkpoint_sha256": _sha256_file(optimizer_path),
            **optimizer_roundtrip,
        }
        completed["receipt_digest"] = route_scope.object_sha256(completed)
        _atomic_write_json(temporary / "receipt.json", completed)
        os.replace(temporary, final)
        _atomic_write_json(
            output / "latest.json",
            {
                "checkpoint": str(final),
                "global_step": global_step,
                "receipt_digest": completed["receipt_digest"],
            },
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return final


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    loss_config = loss_config_from_args(args)
    if (
        _sha256_file(args.dataset_summary) != PINNED_DATASET_SUMMARY_FILE_SHA256
        or _sha256_file(PINNED_DATASET_INDEX) != PINNED_DATASET_INDEX_SHA256
        or _sha256_file(args.routing_jsonl) != PINNED_ROUTING_SHA256
    ):
        raise SourceKVRouteAUHError("pinned dataset summary or routing bytes differ")
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
        raise SourceKVRouteAUHError(str(error)) from error
    if transformer_config["num_attention_heads"] % 4:
        raise SourceKVRouteAUHError("Bernini heads must divide Ulysses=4")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import UniPCMultistepScheduler
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.training.data import NoiseScheduler, SYSTEM_PROMPTS, process_renderer_sample

    if DEFAULT_NEG_PROMPT != v5.DEFAULT_NEGATIVE_PROMPT:
        raise SourceKVRouteAUHError("runtime negative prompt differs")
    distributed = legacy.distributed_contract()
    if distributed.world_size != 4 or distributed.ulysses_size != 4:
        raise SourceKVRouteAUHError("formal V9 AUH pilot requires four ranks")
    device, backend = legacy.initialise_distributed(distributed)
    from bernini.parallel import init_parallel_state

    init_parallel_state(ulysses_size=4)
    legacy.seed_same_sample(args.seed)
    output = Path(args.output).expanduser().resolve()
    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = legacy.validate_preprocessed_dataset_summary(
        args.dataset_summary, dataset, allow_incomplete=False
    )
    if (
        len(dataset) != PINNED_DATASET_ROWS
        or str(dataset.root) != PINNED_DATASET_SHARDS
        or dataset_summary.get("path") != PINNED_DATASET_SUMMARY
        or dataset_summary.get("sha256")
        != PINNED_DATASET_SUMMARY_FILE_SHA256
        or dataset_summary.get("index_path") != PINNED_DATASET_INDEX
        or dataset_summary.get("index_sha256") != PINNED_DATASET_INDEX_SHA256
        or dataset_summary.get("expected_rows") != PINNED_DATASET_ROWS
        or dataset_summary.get("materialized_rows") != PINNED_DATASET_ROWS
        or dataset_summary.get("complete") is not True
    ):
        raise SourceKVRouteAUHError("formal V9 dataset identity differs")
    try:
        router = motion.ReviewRouter.load(args.routing_jsonl, default_tier="reject")
        eligible_routes = v4._build_eligible_routes(dataset, router)
        v5._strict_router(args, router, eligible_routes, dataset)
    except Exception as error:
        raise SourceKVRouteAUHError(f"strict359 routing differs: {error}") from error
    input_tracker = AccessedShardIntegrityTracker.from_pinned_index(dataset=dataset)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    base_model = BerniniRendererModel(config)
    base_model.requires_grad_(False)
    base_model.t5_text_encoder.eval()
    configure_source_kv_gradient_checkpointing(base_model)
    # Preserve the frozen pre-injection affine inventory.  PEFT mutates the
    # module tree in place by wrapping targets; receipt geometry must describe
    # the actual frozen base modules that were validated before that mutation.
    runtime_module_inventory = dict(base_model.named_modules())
    target_modules = route_scope.validate_runtime_target_modules(
        runtime_module_inventory
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            target_modules=target_modules,
        ),
    )
    model.to(device)
    model.eval()
    renderer = model.get_base_model()
    renderer.t5_text_encoder.eval()
    named_trainable = legacy.trainable_lora_parameters(model)
    initialization_digest = legacy.synchronize_trainable_parameters(
        named_trainable, source_rank=0
    )
    adapter_state = get_peft_model_state_dict(model)
    scope_manifest = route_scope.build_receipt_manifest(
        runtime_module_inventory=runtime_module_inventory,
        adapter_state=adapter_state,
        initialization=route_scope.fresh_initialization_declaration(),
    )
    if (
        len(named_trainable) != 184
        or sum(int(parameter.numel()) for _, parameter in named_trainable)
        != 2_260_992
    ):
        raise SourceKVRouteAUHError("runtime PEFT scope differs from exact92/184")
    parameter_names = [name for name, _ in named_trainable]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    replay_handle = replay.install_source_kv_replay(
        renderer, selection=args.block_selection
    )

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
    scheduler_kwargs = legacy.noise_scheduler_kwargs()
    scheduler_kwargs["noise_tmin"] = 0.1
    scheduler = NoiseScheduler(**scheduler_kwargs)
    inference_scheduler = UniPCMultistepScheduler.from_pretrained(
        str(checkpoint),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=sigma_strata.FLOW_SHIFT,
    )
    sigma_strata.audit_runtime_unipc_schedule(inference_scheduler)
    immutable = _immutable_contract(
        args=args,
        dataset=dataset,
        dataset_summary=dataset_summary,
        router=router,
        eligible_routes=eligible_routes,
        scope_manifest=scope_manifest,
        checkpoint=checkpoint,
        loss_config=loss_config,
    )

    step_audit: list[dict[str, Any]] = []
    try:
        for global_step in range(MAX_STEPS):
            selected = sigma_strata.select_sigma_stratum(global_step)
            row_index, raw_row, route, shard_integrity = (
                next_integrity_tracked_routed_row(
                    tracker=input_tracker,
                    eligible_routes=eligible_routes,
                    ordinal=global_step,
                )
            )
            identity = legacy.dataset_identity(raw_row, row_index)
            legacy.assert_identical_row(identity)
            current_seed = legacy.step_seed(args.seed, global_step, row_index)
            legacy.seed_same_sample(current_seed)
            prepared = _prepare_source_endpoint_candidate_cpu(
                raw_row=raw_row,
                tokenizer=tokenizer,
                prompt_cleaner=prompt_clean,
                system_prompts=SYSTEM_PROMPTS,
                rope=rope,
                vae_mean=vae_mean,
                vae_std=vae_std,
                z_dim=z_dim,
                scheduler=scheduler,
                noop_instruction=args.noop_instruction,
                negative_prompt=args.negative_prompt,
                process_renderer_sample=process_renderer_sample,
                selected_stratum=selected,
            )
            moved = _move_source_endpoint_candidate_to_device(
                prepared, device=device, noop_instruction=args.noop_instruction
            )
            timestep_token = canonical_timestep_token(
                timestep=selected.timestep, sigma=moved.auxiliary["sigma"]
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                step = execute_source_kv_route_optimizer_step(
                    renderer=renderer,
                    adapter_controller=model,
                    candidate=moved,
                    cache_bank=replay_handle.cache_bank,
                    named_trainable=named_trainable,
                    optimizer=optimizer,
                    generation=global_step,
                    step_index=selected.schedule_index,
                    timestep_token=timestep_token,
                    rank=distributed.rank,
                    ulysses_size=distributed.ulysses_size,
                    loss_config=loss_config,
                    require_fresh_parity=global_step == 0,
                )
            record = {
                **dict(step.record),
                "row_index": row_index,
                "iid": route.iid,
                "seed": current_seed,
                "instruction_sha256": moved.instruction_sha256,
                "sigma_timestep": selected.timestep,
                "block_selection": args.block_selection,
                "input_shard_integrity": dict(shard_integrity),
            }
            v6_runtime._assert_gate_record_equal_across_ranks(record)
            step_audit.append(record)
            if distributed.rank == 0:
                print(
                    json.dumps({"event": "optimizer_step", **record}, sort_keys=True),
                    flush=True,
                )

        exact40 = validate_exact40_step_audit(
            step_audit, block_selection=args.block_selection
        )
        final_optimizer = audit_optimizer_state(
            optimizer=optimizer,
            named_trainable=named_trainable,
            expected_step=40,
        )
        input_integrity = input_tracker.finalize(
            summary_path=args.dataset_summary,
            routing_path=args.routing_jsonl,
        )
        v6_runtime._assert_gate_record_equal_across_ranks(input_integrity)
        optimizer_payload = _optimizer_payload(
            optimizer=optimizer,
            global_step=40,
            immutable=immutable,
            parameter_names=parameter_names,
            step_audit=step_audit,
            optimizer_audit=final_optimizer,
            input_integrity=input_integrity,
        )
        receipt = _build_receipt(
            args=args,
            global_step=40,
            step_audit=step_audit,
            exact40_audit=exact40,
            dataset=dataset,
            dataset_summary=dataset_summary,
            router=router,
            checkpoint=checkpoint,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            distributed=distributed,
            backend=backend,
            scope_manifest=scope_manifest,
            named_trainable=named_trainable,
            initialization_digest=initialization_digest,
            immutable=immutable,
            optimizer_payload=optimizer_payload,
            input_integrity=input_integrity,
            transformers_version=transformers_version,
        )
        _save_checkpoint_with_artifact_roundtrip(
            model=model,
            optimizer_payload=optimizer_payload,
            named_trainable=named_trainable,
            output=output,
            global_step=40,
            receipt=receipt,
            rank=distributed.rank,
        )
    finally:
        replay_handle.restore()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    return 0


__all__ = [
    "AccessedShardIntegrityTracker",
    "ARTIFACT_VALIDATION_SCHEMA",
    "FORWARD_ORDER",
    "LEARNING_RATE",
    "MAX_STEPS",
    "METHOD_NAME",
    "MovedSourceKVRouteCandidate",
    "OptimizerStepResult",
    "OPTIMIZER_SCHEMA",
    "PINNED_DATASET_INDEX_SHA256",
    "PINNED_DATASET_SHARDS",
    "PINNED_DATASET_SUMMARY",
    "PINNED_DATASET_SUMMARY_FILE_SHA256",
    "PINNED_ELIGIBLE_ROWS",
    "PINNED_ROUTING_JSONL",
    "PINNED_ROUTING_SHA256",
    "PreparedSourceKVRouteCandidate",
    "QUERY_STATE_DEPLOYMENT_FORMULA",
    "QUERY_STATE_FOLLOWUP_ARM",
    "QUERY_STATE_POLICY",
    "QUERY_STATE_TRAIN_FORMULA",
    "RECEIPT_SCHEMA",
    "SAVE_EVERY",
    "SixForwardCellResult",
    "SourceKVRouteAUHError",
    "audit_cache_after_backward",
    "audit_cache_after_clear",
    "audit_optimizer_state",
    "audit_trainable_gradients",
    "build_parser",
    "canonical_timestep_token",
    "configure_source_kv_gradient_checkpointing",
    "execute_source_kv_route_optimizer_step",
    "loss_config_from_args",
    "main",
    "next_integrity_tracked_routed_row",
    "select_training_query_endpoint",
    "validate_optimizer_artifact_roundtrip",
    "validate_cli",
    "validate_exact40_step_audit",
]


if __name__ == "__main__":
    raise SystemExit(main())

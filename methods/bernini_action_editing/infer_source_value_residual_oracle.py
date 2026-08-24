#!/usr/bin/env python3
"""Frozen Bernini-R dog oracle for seven source-value residual arms.

The only method-level CLI choice is ``--arm``.  Each arm resolves to one
pre-registered operator, block scope, and fixed gate.  No adapter, paired
target, mask, flow, pose, track, trajectory, or first-frame condition is
accepted by this entry point.

Most of the model, source, checkpoint, prompt, sampler, and reversible
source-only carrier plumbing is shared with
``infer_source_kv_carrier_oracle.py``.  The Z0 arm additionally executes an
untouched official O0 sample in the same four-rank process before executing
the fixed-zero V10 wrapper.  Every rank must prove byte-identical generated
latents before rank zero may decode or write the V10 output.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as legacy  # noqa: E402
import infer_source_kv_carrier_oracle as carrier_oracle  # noqa: E402
import source_kv_replay as replay_core  # noqa: E402
import source_kv_route_batches as route_batches  # noqa: E402
import source_value_residual as value_core  # noqa: E402


RECEIPT_SCHEMA = "bernini-r-1p3b-frozen-source-value-residual-oracle-v10-v1"
EXPECTED_STEPS = carrier_oracle.EXPECTED_STEPS
EXPECTED_FRAMES = carrier_oracle.EXPECTED_FRAMES
EXPECTED_ULYSSES_SIZE = carrier_oracle.EXPECTED_ULYSSES_SIZE
EXPECTED_SOURCE_TOKENS = carrier_oracle.EXPECTED_DOG_SOURCE_TOKENS
EXPECTED_PAIR_TOKENS = carrier_oracle.EXPECTED_DOG_PAIR_TOKENS
EXPECTED_BUCKET_HW = (496, 480)
EXPECTED_SEED = 2027
EXPECTED_INSTRUCTION = "Make the dog pick up the bone and hold it in its mouth."
EXPECTED_SOURCE_SHA256 = (
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
)
EXPECTED_ORIGINAL_SOURCE_PATH = (
    "/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/"
    "videos/288545b9c031491a/source.mp4"
)
PINNED_HISTORICAL_O0_MP4_SHA256 = (
    "980b0daa85a15feac427f6f2611985d3e33896a0ac45e1ee57938e146ac987c9"
)
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SourceValueResidualOracleError(RuntimeError):
    """Raised before an ambiguous or incomplete V10 oracle can be emitted."""


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    operator: str
    block_selection: str
    gate: float
    decision_role: str

    @property
    def block_indices(self) -> tuple[int, ...]:
        return replay_core.resolve_block_indices(
            replay_core.EXPECTED_BLOCK_COUNT, self.block_selection
        )

    @property
    def residual_varlen_calls_per_layer(self) -> int:
        if self.gate == 0.0:
            return 0
        return (
            2
            if self.operator == value_core.CACHED_KV_DIAGNOSTIC
            else 1
        )


_ARM_SPECS = (
    ArmSpec("Z0", value_core.MAIN_OPERATOR, "late", 0.0, "engineering_parity_gate"),
    ArmSpec("F25", value_core.MAIN_OPERATOR, "late", 0.25, "main_decision_arm"),
    ArmSpec("F50", value_core.MAIN_OPERATOR, "late", 0.50, "main_decision_arm"),
    ArmSpec("F100", value_core.MAIN_OPERATOR, "late", 1.0, "main_decision_arm"),
    ArmSpec("FA25", value_core.MAIN_OPERATOR, "all", 0.25, "depth_accumulation_diagnostic"),
    ArmSpec(
        "SN10",
        value_core.SOURCE_NORMALIZED_DIAGNOSTIC,
        "late",
        0.10,
        "source_renormalization_diagnostic",
    ),
    ArmSpec(
        "CK10",
        value_core.CACHED_KV_DIAGNOSTIC,
        "late",
        0.10,
        "cached_key_diagnostic",
    ),
)
ARM_SPECS = {spec.arm: spec for spec in _ARM_SPECS}
ARM_NAMES = tuple(spec.arm for spec in _ARM_SPECS)
ARM_TABLE = [asdict(spec) for spec in _ARM_SPECS]
ARM_TABLE_SHA256 = legacy.object_sha256(ARM_TABLE)


def arm_spec(arm: str) -> ArmSpec:
    try:
        return ARM_SPECS[arm]
    except (KeyError, TypeError) as error:
        raise SourceValueResidualOracleError(
            f"arm must be one of {ARM_NAMES}, got {arm!r}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one pre-registered frozen Bernini V10 dog oracle arm"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--original-source-path", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument(
        "--expected-bernini-commit",
        default=legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> ArmSpec:
    spec = arm_spec(args.arm)
    if args.instruction != EXPECTED_INSTRUCTION:
        raise SourceValueResidualOracleError(
            "dog oracle instruction differs from the pre-registered action"
        )
    if args.original_source_path != EXPECTED_ORIGINAL_SOURCE_PATH:
        raise SourceValueResidualOracleError(
            "dog oracle original source path differs"
        )
    if args.expected_source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SourceValueResidualOracleError(
            "dog oracle source SHA-256 differs"
        )
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA1_RE.fullmatch(value.lower()) is None:
            raise SourceValueResidualOracleError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
        "expected_source_sha256",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise SourceValueResidualOracleError(
                f"{name} must be a lowercase SHA-256"
            )
    # Reuse every stable frozen-oracle source/checkpoint/sampler validation.
    shared = argparse.Namespace(
        **vars(args),
        replay="on",
        block_selection=spec.block_selection,
        expected_source_tokens=EXPECTED_SOURCE_TOKENS,
        num_inference_steps=EXPECTED_STEPS,
        seed=EXPECTED_SEED,
    )
    try:
        carrier_oracle.validate_cli(shared)
    except carrier_oracle.SourceKVCarrierOracleError as error:
        raise SourceValueResidualOracleError(str(error)) from error
    return spec


def tensor_identity(value: Any, *, label: str) -> dict[str, Any]:
    """Hash tensor metadata and exact storage bytes independent of device."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise SourceValueResidualOracleError(
            "tensor identity requires PyTorch"
        ) from error
    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise SourceValueResidualOracleError(f"{label} must be a non-empty tensor")
    detached = value.detach().contiguous()
    if not bool(torch.isfinite(detached).all()):
        raise SourceValueResidualOracleError(f"{label} contains non-finite values")
    cpu = detached.cpu().contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes(order="C")
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "byte_count": len(raw),
    }
    payload = legacy.canonical_json_bytes(metadata) + b"\0" + raw
    return {
        **metadata,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_storage_sha256": hashlib.sha256(raw).hexdigest(),
        "finite": True,
        "label": label,
    }


def save_video_atomically(
    decoded: Any,
    output_path: Path,
    *,
    fps: int,
    save_output_fn: Any,
) -> None:
    """Write one fresh MP4 and always unlink this process's hidden temporary."""

    token = output_transaction_token()
    temporary = output_path.with_name(
        f".{output_path.stem}.tmp-{token}{output_path.suffix}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise SourceValueResidualOracleError("stale temporary output exists")
    try:
        save_output_fn(decoded, str(temporary), fps=int(fps))
        if temporary.is_symlink() or not temporary.is_file():
            raise SourceValueResidualOracleError(
                "video encoder did not create one plain temporary output"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        # ``os.replace`` removes the source name on success.  On encoder error
        # (including a partially written file), remove only the exact hidden
        # path whose absence was established above.  This also unlinks a
        # maliciously substituted symlink rather than following it.
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def output_transaction_token() -> str:
    """Resolve a launcher-visible, path-safe token for hidden output files."""

    configured = os.environ.get("BERNINI_OUTPUT_TRANSACTION_ID")
    if configured is None:
        return f"pid-{os.getpid()}"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", configured) is None:
        raise SourceValueResidualOracleError(
            "BERNINI_OUTPUT_TRANSACTION_ID is not a path-safe token"
        )
    return configured


def write_receipt_atomically(
    receipt_path: Path, receipt: Mapping[str, Any]
) -> None:
    """Write one canonical receipt with a deterministic, always-cleaned temp."""

    token = output_transaction_token()
    temporary = receipt_path.with_name(f".{receipt_path.name}.tmp-{token}")
    if temporary.exists() or temporary.is_symlink():
        raise SourceValueResidualOracleError("stale temporary receipt exists")
    payload = legacy.canonical_json_bytes(receipt) + b"\n"
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, receipt_path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def unlink_fresh_artifact(path: Path) -> None:
    """Unlink one exact fresh file/symlink without following or recursing."""

    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        raise SourceValueResidualOracleError(
            f"refusing to recursively remove substituted artifact directory: {path}"
        )
    path.unlink()


def _validate_trace(
    trace: Mapping[str, Any], *, rank: int, source_tokens: int
) -> str:
    steps = trace.get("steps") if isinstance(trace, Mapping) else None
    if (
        trace.get("sample_calls") != 1
        or trace.get("step_count") != EXPECTED_STEPS
        or trace.get("unique_identity_count") != EXPECTED_STEPS
        or not isinstance(steps, list)
        or len(steps) != EXPECTED_STEPS
    ):
        raise SourceValueResidualOracleError(
            "carrier trace lacks 40 unique completed step identities"
        )
    identities = []
    for step_index, item in enumerate(steps):
        if (
            not isinstance(item, Mapping)
            or item.get("step_index") != step_index
            or item.get("rank") != rank
            or item.get("ulysses_size") != EXPECTED_ULYSSES_SIZE
            or item.get("model_id") != "transformer_1"
            or item.get("source_tokens_runtime") != source_tokens
            or item.get("pair_tokens_runtime") != 2 * source_tokens
            or item.get("carrier_forwards") != 1
            or item.get("negative_replay_forwards") != 1
            or item.get("action_replay_forwards") != 1
            or item.get("cleared_after_both_replays") is not True
        ):
            raise SourceValueResidualOracleError(
                f"carrier step {step_index} execution evidence differs"
            )
        identities.append(
            (
                item.get("generation"),
                item.get("step_index"),
                item.get("timestep_token"),
                item.get("rank"),
                item.get("ulysses_size"),
            )
        )
    if len(set(identities)) != EXPECTED_STEPS:
        raise SourceValueResidualOracleError("carrier identities are not unique")
    return legacy.object_sha256(trace)


def validate_value_runtime_certificate(
    core_receipt: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    spec: ArmSpec,
    expected_source_tokens: int,
    rank: int,
    hook_restored: bool,
) -> dict[str, Any]:
    """Validate exact per-rank V10 capture/replay/operator counts."""

    indices = list(spec.block_indices)
    contract = value_core.source_value_residual_contract(
        selection=spec.block_selection,
        operator=spec.operator,
        gate=spec.gate,
    )
    for key in (
        "schema_version",
        "operator",
        "fixed_gate",
        "block_selection",
        "block_indices",
        "contract_digest",
    ):
        if core_receipt.get(key) != contract.get(key):
            raise SourceValueResidualOracleError(
                f"source-value core receipt differs at {key}"
            )
    runtime = core_receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise SourceValueResidualOracleError("source-value runtime is missing")
    if (
        runtime.get("restored") is not True
        or hook_restored is not True
        or runtime.get("installed_block_count") != len(indices)
    ):
        raise SourceValueResidualOracleError(
            "source-value processor/hook installation or restore differs"
        )
    cache = runtime.get("cache")
    if not isinstance(cache, Mapping):
        raise SourceValueResidualOracleError("source-value cache receipt is missing")
    expected_capture = len(indices) * EXPECTED_STEPS
    expected_lookup = 0 if spec.gate == 0.0 else len(indices) * EXPECTED_STEPS * 2
    expected_cache_branches = (
        {}
        if spec.gate == 0.0
        else {
            "frozen_action": len(indices) * EXPECTED_STEPS,
            "frozen_negative": len(indices) * EXPECTED_STEPS,
        }
    )
    if (
        cache.get("identity") is not None
        or cache.get("captured_blocks") != []
        or cache.get("entries") != []
        or cache.get("capture_calls") != expected_capture
        or cache.get("replay_lookups") != expected_lookup
        or cache.get("replay_branch_counts") != expected_cache_branches
        or cache.get("retired_identity_count") != EXPECTED_STEPS
        or cache.get("checkpoint_context_counts")
        != {
            replay_core.CHECKPOINT_FORWARD: 0,
            replay_core.CHECKPOINT_RECOMPUTE: 0,
        }
    ):
        raise SourceValueResidualOracleError(
            "rank-local cache capture/lookup/clear counts differ"
        )
    per_block = runtime.get("per_block")
    if not isinstance(per_block, list) or len(per_block) != len(indices):
        raise SourceValueResidualOracleError("per-block V10 evidence differs")
    expected_branches = {
        replay_core.CAPTURE_BRANCH_TAG: EXPECTED_STEPS,
        "frozen_action": EXPECTED_STEPS,
        "frozen_negative": EXPECTED_STEPS,
    }
    residual_calls = EXPECTED_STEPS * 2 * spec.residual_varlen_calls_per_layer
    for index, item in zip(indices, per_block):
        metrics = item.get("metrics") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or item.get("block_index") != index
            or item.get("operator") != spec.operator
            or item.get("fixed_gate") != spec.gate
            or item.get("capture_calls") != EXPECTED_STEPS
            or item.get("replay_calls") != EXPECTED_STEPS * 2
            or item.get("zero_gate_delegations")
            != (EXPECTED_STEPS * 2 if spec.gate == 0.0 else 0)
            or item.get("residual_varlen_calls") != residual_calls
            or item.get("branch_counts") != expected_branches
            or item.get("execution_phase_counts")
            != {
                replay_core.EAGER_EXECUTION: EXPECTED_STEPS * 3,
                replay_core.CHECKPOINT_FORWARD: 0,
                replay_core.CHECKPOINT_RECOMPUTE: 0,
            }
            or item.get("last_source_tokens") != expected_source_tokens
            or item.get("ulysses_observed") is not True
            or not isinstance(metrics, Mapping)
            or metrics.get("calls")
            != (0 if spec.gate == 0.0 else EXPECTED_STEPS * 2)
            or metrics.get("projected_output_all_finite") is not True
        ):
            raise SourceValueResidualOracleError(
                f"block {index} lacks exact V10 frozen-oracle evidence"
            )
        if spec.gate == 0.0:
            if any(
                metrics.get(name) is not None
                for name in (
                    "all_finite",
                    "combined_attention_output_all_finite",
                    "base_target_rms",
                    "source_value_delta_rms",
                    "delta_memory_rms",
                    "gated_delta_rms",
                    "gated_to_base_rms_ratio",
                )
            ):
                raise SourceValueResidualOracleError(
                    f"zero-gate block {index} unexpectedly computed a residual"
                )
        else:
            numeric_names = (
                "base_target_rms",
                "source_value_delta_rms",
                "delta_memory_rms",
                "gated_delta_rms",
                "gated_to_base_rms_ratio",
            )
            if (
                metrics.get("all_finite") is not True
                or metrics.get("combined_attention_output_all_finite") is not True
                or any(
                    isinstance(metrics.get(name), bool)
                    or not isinstance(metrics.get(name), (int, float))
                    or not math.isfinite(float(metrics[name]))
                    or float(metrics[name]) < 0.0
                    for name in numeric_names
                )
            ):
                raise SourceValueResidualOracleError(
                    f"block {index} residual diagnostics are invalid"
                )
    trace_digest = _validate_trace(
        trace, rank=rank, source_tokens=expected_source_tokens
    )
    return {
        "validated": True,
        "arm": spec.arm,
        "operator": spec.operator,
        "fixed_gate": spec.gate,
        "decision_role": spec.decision_role,
        "rank": rank,
        "ulysses_size": EXPECTED_ULYSSES_SIZE,
        "block_selection": spec.block_selection,
        "actual_installed_block_indices": indices,
        "selected_block_count": len(indices),
        "per_layer_capture_calls": EXPECTED_STEPS,
        "per_layer_replay_calls": EXPECTED_STEPS * 2,
        "per_layer_residual_varlen_calls": residual_calls,
        "per_layer_zero_gate_delegations": (
            EXPECTED_STEPS * 2 if spec.gate == 0.0 else 0
        ),
        "rank_local_bank_capture_calls": expected_capture,
        "rank_local_bank_replay_lookups": expected_lookup,
        "unique_step_identities": EXPECTED_STEPS,
        "source_tokens_runtime": expected_source_tokens,
        "pair_tokens_runtime": expected_source_tokens * 2,
        "cache_empty_after_each_step": True,
        "processor_restore": True,
        "sampler_hook_restore": True,
        "trace_digest": trace_digest,
        "core_runtime_digest": runtime.get("runtime_digest", core_receipt.get("runtime_digest")),
        "per_block_metrics": [dict(item["metrics"]) for item in per_block],
    }


def validate_four_rank_runtime(
    rows: Sequence[Mapping[str, Any]], *, spec: ArmSpec
) -> dict[str, Any]:
    if len(rows) != EXPECTED_ULYSSES_SIZE:
        raise SourceValueResidualOracleError("exactly four rank rows are required")
    if sorted(row.get("rank") for row in rows) != list(range(EXPECTED_ULYSSES_SIZE)):
        raise SourceValueResidualOracleError("rank rows are incomplete")
    reference = rows[0]
    invariant = (
        "arm",
        "operator",
        "fixed_gate",
        "decision_role",
        "ulysses_size",
        "block_selection",
        "actual_installed_block_indices",
        "selected_block_count",
        "per_layer_capture_calls",
        "per_layer_replay_calls",
        "per_layer_residual_varlen_calls",
        "per_layer_zero_gate_delegations",
        "rank_local_bank_capture_calls",
        "rank_local_bank_replay_lookups",
        "unique_step_identities",
        "source_tokens_runtime",
        "pair_tokens_runtime",
    )
    if any(
        row.get("validated") is not True
        or row.get("arm") != spec.arm
        or row.get("ulysses_size") != EXPECTED_ULYSSES_SIZE
        for row in rows
    ):
        raise SourceValueResidualOracleError("one rank V10 certificate differs")
    if any(
        row.get(name) != reference.get(name)
        for row in rows[1:]
        for name in invariant
    ):
        raise SourceValueResidualOracleError("V10 runtime counts differ across ranks")
    identities = [row.get("generated_latent") for row in rows]
    if not all(isinstance(item, Mapping) for item in identities):
        raise SourceValueResidualOracleError("generated latent identity is missing")
    latent_invariant = ("shape", "dtype", "numel", "byte_count", "content_sha256", "raw_storage_sha256", "finite")
    if any(
        item.get(name) != identities[0].get(name)
        for item in identities[1:]
        for name in latent_invariant
    ):
        raise SourceValueResidualOracleError(
            "full generated latent differs across Ulysses ranks"
        )
    z0_rows = [row.get("z0_control") for row in rows]
    if spec.arm == "Z0":
        if not all(isinstance(item, Mapping) for item in z0_rows):
            raise SourceValueResidualOracleError("Z0 baseline evidence is missing")
        if any(
            item.get("byte_exact") is not True
            or item.get("official_o0_latent") != row.get("generated_latent")
            or item.get("historical_o0_mp4_sha256")
            != PINNED_HISTORICAL_O0_MP4_SHA256
            or not isinstance(item.get("official_o0_runtime"), Mapping)
            or item["official_o0_runtime"].get("validated") is not True
            or item["official_o0_runtime"].get("replay") != "off"
            or item["official_o0_runtime"].get("rank") != row.get("rank")
            or item["official_o0_runtime"].get("ulysses_size")
            != EXPECTED_ULYSSES_SIZE
            for row, item in zip(rows, z0_rows)
        ):
            raise SourceValueResidualOracleError(
                "Z0 wrapper is not byte-identical to same-job official O0"
            )
        if any(
            item.get("official_o0_latent") != z0_rows[0].get("official_o0_latent")
            for item in z0_rows[1:]
        ):
            raise SourceValueResidualOracleError(
                "same-job official O0 latent differs across ranks"
            )
    elif any(item is not None for item in z0_rows):
        raise SourceValueResidualOracleError("non-Z0 arm contains Z0 evidence")
    canonical_rows = [dict(row) for row in rows]
    all_rank_digest = legacy.object_sha256(canonical_rows)
    return {
        "validated": True,
        "all_four_ranks_exact": True,
        "arm": spec.arm,
        "operator": spec.operator,
        "fixed_gate": spec.gate,
        "block_selection": spec.block_selection,
        "actual_block_indices": list(spec.block_indices),
        "per_rank_capture_calls": reference["rank_local_bank_capture_calls"],
        "per_rank_replay_lookups": reference["rank_local_bank_replay_lookups"],
        "per_rank_residual_varlen_calls": (
            reference["per_layer_residual_varlen_calls"]
            * reference["selected_block_count"]
        ),
        "full_generated_latent": dict(identities[0]),
        "full_generated_latent_sha256": identities[0]["content_sha256"],
        "all_rank_generated_latent_exact": True,
        "z0_same_job_official_byte_exact": spec.arm == "Z0",
        "per_rank": canonical_rows,
        "all_rank_certificate_digest": all_rank_digest,
    }


def _sample_value_arm(
    model: Any,
    *,
    spec: ArmSpec,
    noop_prompt_embeds: Any,
    rank: int,
    source_tokens: int,
    sample_kwargs: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    patch: Optional[value_core.SourceValueResidualPatchHandle] = None
    hook: Optional[carrier_oracle.InstalledSourceKVCarrierHook] = None
    with value_core.source_value_residual(
        model,
        selection=spec.block_selection,
        operator=spec.operator,
        gate=spec.gate,
    ) as installed_patch:
        patch = installed_patch
        with carrier_oracle.source_kv_carrier_hook(
            model,
            cache_bank=installed_patch.cache_bank,
            noop_prompt_embeds=noop_prompt_embeds,
            rank=rank,
            ulysses_size=EXPECTED_ULYSSES_SIZE,
            expected_steps=EXPECTED_STEPS,
            expected_source_tokens=source_tokens,
        ) as installed_hook:
            hook = installed_hook
            generated = model.sample(**dict(sample_kwargs))
    if patch is None or hook is None:
        raise SourceValueResidualOracleError("V10 carrier contexts did not install")
    core_receipt = patch.receipt()
    certificate = validate_value_runtime_certificate(
        core_receipt,
        hook.trace.as_dict(),
        spec=spec,
        expected_source_tokens=source_tokens,
        rank=rank,
        hook_restored=hook.restored,
    )
    return generated, core_receipt, certificate


def _sample_same_job_official_o0(
    model: Any,
    *,
    spec: ArmSpec,
    rank: int,
    source_tokens: int,
    sample_kwargs: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    observer: Optional[carrier_oracle.InstalledReplayOffAuditHook] = None
    with carrier_oracle.replay_off_audit_hook(
        model,
        rank=rank,
        ulysses_size=EXPECTED_ULYSSES_SIZE,
        expected_steps=EXPECTED_STEPS,
        expected_source_tokens=source_tokens,
    ) as installed:
        observer = installed
        generated = model.sample(**dict(sample_kwargs))
    if observer is None:
        raise SourceValueResidualOracleError("official O0 observer did not install")
    certificate = carrier_oracle.disabled_runtime_certificate(
        observer.trace.as_dict(),
        selection=spec.block_selection,
        source_tokens_from_input_geometry=source_tokens,
        rank=rank,
        observer_restored=observer.restored,
    )
    return generated, certificate


def build_receipt(
    *,
    args: argparse.Namespace,
    spec: ArmSpec,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    source_tokens: int,
    output_path: Path,
    output_sha256: str,
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
    freeze_certificate: Mapping[str, Any],
    four_rank_runtime: Mapping[str, Any],
    rank0_core_receipt: Mapping[str, Any],
    pairing: Mapping[str, Any],
    checkpoint_content_identity: Mapping[str, Any],
) -> dict[str, Any]:
    pairing_payload = dict(pairing)
    pairing_digest = pairing_payload.pop("causal_pairing_digest", None)
    if (
        not isinstance(pairing_digest, str)
        or pairing_digest != legacy.object_sha256(pairing_payload)
    ):
        raise SourceValueResidualOracleError(
            "causal pairing digest does not bind its canonical payload"
        )
    shared_args = argparse.Namespace(
        **vars(args),
        replay="on",
        block_selection=spec.block_selection,
        expected_source_tokens=EXPECTED_SOURCE_TOKENS,
        num_inference_steps=EXPECTED_STEPS,
        seed=EXPECTED_SEED,
    )
    receipt = carrier_oracle.build_receipt(
        args=shared_args,
        source_path=source_path,
        source_sha256=source_sha256,
        source_metadata=source_metadata,
        source_tokens=source_tokens,
        output_path=output_path,
        output_sha256=output_sha256,
        bernini_revision=bernini_revision,
        veomni_revision=veomni_revision,
        inference_file_hashes=inference_file_hashes,
        runtime_versions=runtime_versions,
        freeze_certificate=freeze_certificate,
        four_rank_runtime=four_rank_runtime,
        rank0_core_receipt=rank0_core_receipt,
        pairing=pairing,
        checkpoint_content_identity=checkpoint_content_identity,
    )
    receipt.pop("receipt_digest", None)
    receipt["schema_version"] = RECEIPT_SCHEMA
    receipt["arm_registry"] = {
        "allowed_arms": list(ARM_NAMES),
        "arm_table_sha256": ARM_TABLE_SHA256,
        "selected": asdict(spec),
        "method_cli_controls": ["arm"],
        "operator_scope_gate_override_supported": False,
    }
    receipt["causal_control"] = {
        "arm": spec.arm,
        "operator": spec.operator,
        "fixed_gate": spec.gate,
        "block_selection": spec.block_selection,
        "decision_role": spec.decision_role,
        "pairing_contract": dict(pairing),
        "causal_pairing_digest": pairing["causal_pairing_digest"],
        "arm_excluded_from_pairing_digest": True,
        "excluded_fields": [
            "arm",
            "operator",
            "fixed_gate",
            "block_selection",
            "output_path",
        ],
        "same_source_instruction_seed_scheduler_across_arms": True,
    }
    receipt["oracle"] = {
        "zero_training": True,
        "base_frozen": True,
        "integration_status": "integrated_frozen_source_value_runtime_certificate",
        "arm": spec.arm,
        "operator": spec.operator,
        "fixed_gate": spec.gate,
        "requested_block_selection": spec.block_selection,
        "actual_block_indices": list(spec.block_indices),
        "runtime_execution_certificate": dict(four_rank_runtime),
        "rank0_source_value_core_receipt": dict(rank0_core_receipt),
        "main_operator": value_core.MAIN_OPERATOR,
        "operator_is_main": spec.operator == value_core.MAIN_OPERATOR,
        "training_decision_arm": spec.decision_role == "main_decision_arm",
    }
    receipt["z0_control"] = (
        {
            "same_job_official_o0_executed_first": True,
            "same_job_generated_latent_byte_exact": four_rank_runtime[
                "z0_same_job_official_byte_exact"
            ],
            "same_job_generated_latent_sha256": four_rank_runtime[
                "full_generated_latent_sha256"
            ],
            "historical_o0_mp4_sha256_binding": PINNED_HISTORICAL_O0_MP4_SHA256,
            "historical_o0_file_reverified_this_job": False,
            "historical_binding_role": "previously_closed_mp4_comparator_only",
        }
        if spec.arm == "Z0"
        else None
    )
    receipt["input"]["accepted_model_conditions"] = [
        "source_video",
        "edit_instruction",
    ]
    receipt["input"]["paired_target_video_available_to_runner"] = False
    receipt["input"]["operator_external_condition_count"] = 0
    receipt["output"]["generated_latent_sha256"] = four_rank_runtime[
        "full_generated_latent_sha256"
    ]
    receipt["output"]["all_rank_generated_latent_exact"] = True
    receipt["experimental_inference"] = True
    receipt["production_claim_forbidden"] = True
    receipt["scientific_claim_authorized"] = False
    if "adapter" in receipt:
        raise SourceValueResidualOracleError("adapter object is forbidden")
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    spec = validate_cli(args)
    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute():
        raise SourceValueResidualOracleError("source video must be absolute")
    source_path = legacy._plain_file(
        source_requested.resolve(strict=True), label="source video"
    )
    output_path, receipt_path = legacy._resolve_output(args.output)

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
    except legacy.trainer.TrainingContractError as error:
        raise SourceValueResidualOracleError(str(error)) from error
    if transformer_config["num_attention_heads"] % EXPECTED_ULYSSES_SIZE:
        raise SourceValueResidualOracleError(
            "attention heads are not divisible by Ulysses=4"
        )
    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    if not checkpoint_manifest.is_absolute():
        raise SourceValueResidualOracleError(
            "checkpoint content manifest must be absolute"
        )
    inference_file_hashes = legacy.validate_inference_source_files(bernini_root)
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT:
        raise SourceValueResidualOracleError("runtime mv2v prompt differs")
    if DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT:
        raise SourceValueResidualOracleError("runtime negative prompt differs")
    route_batches.validate_noop_instruction(route_batches.EXACT_NOOP_INSTRUCTION)
    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise SourceValueResidualOracleError(
            "four-rank oracle requires AUH ROCm GPUs"
        )
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_validation: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_validation[0] = {
                "ok": True,
                "identity": carrier_oracle.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_validation[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_validation, src=0)
    checkpoint_result = checkpoint_validation[0]
    if (
        not isinstance(checkpoint_result, Mapping)
        or checkpoint_result.get("ok") is not True
        or not isinstance(checkpoint_result.get("identity"), Mapping)
    ):
        detail = (
            checkpoint_result.get("error")
            if isinstance(checkpoint_result, Mapping)
            else "missing rank-zero checkpoint audit"
        )
        raise SourceValueResidualOracleError(
            f"rank-zero checkpoint validation failed: {detail}"
        )
    checkpoint_content_identity = dict(checkpoint_result["identity"])

    source_tensor, source_metadata, source_sha256 = (
        carrier_oracle.prepare_hashed_source_snapshot(source_path)
    )
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SourceValueResidualOracleError("staged dog source SHA-256 differs")
    full_prompt = legacy.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = legacy.build_training_prompt(
        route_batches.EXACT_NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )
    negative_prompt = legacy.DEFAULT_NEGATIVE_PROMPT

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        legacy.trainer.validate_renderer_config_mapping(
            config.to_dict(), checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise SourceValueResidualOracleError(str(error)) from error
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    freeze = carrier_oracle.model_freeze_certificate(model)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    input_ids, attention_mask = legacy._tokenize_training_prompt(
        tokenizer, full_prompt
    )
    negative_ids, negative_mask = legacy._tokenize_renderer_negative(
        tokenizer, negative_prompt
    )
    noop_ids, noop_mask = legacy._tokenize_training_prompt(tokenizer, noop_prompt)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    with torch.no_grad():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        )
    bucket = source_metadata["source_derived_bucket_hw"]
    if tuple(int(item) for item in bucket) != EXPECTED_BUCKET_HW:
        raise SourceValueResidualOracleError(
            "canonical dog source bucket must be H=496,W=480"
        )
    expected_latent_shape = (
        1,
        int(vae.config.z_dim),
        legacy.LATENT_FRAME_COUNT,
        int(bucket[0]) // 8,
        int(bucket[1]) // 8,
    )
    if tuple(int(item) for item in source_latent.shape) != expected_latent_shape:
        raise SourceValueResidualOracleError("source VAE latent shape differs")
    source_tokens = carrier_oracle.source_tokens_from_vae_latent_shape(
        source_latent.shape
    )
    if source_tokens != EXPECTED_SOURCE_TOKENS:
        raise SourceValueResidualOracleError("dog source token geometry differs")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    model.t5_text_encoder.to(device)
    with torch.no_grad():
        noop_prompt_embeds = model.encode_prompt(
            noop_ids.to(device), noop_mask.to(device)
        )
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()

    sample_kwargs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "uncond_input_ids": negative_ids.to(device),
        "uncond_attention_mask": negative_mask.to(device),
        "image_vae_latents": None,
        "multi_video_vae_latents": [source_latent],
        "multi_image_vae_latents": None,
        "width": int(bucket[1]),
        "height": int(bucket[0]),
        "device": device,
        **legacy.sampler_contract(steps=EXPECTED_STEPS, seed=EXPECTED_SEED),
    }

    official_o0_latent = None
    official_o0_certificate = None
    with torch.no_grad():
        if spec.arm == "Z0":
            official_o0_latent, official_o0_certificate = (
                _sample_same_job_official_o0(
                    model,
                    spec=spec,
                    rank=distributed.rank,
                    source_tokens=source_tokens,
                    sample_kwargs=sample_kwargs,
                )
            )
        generated_latent, core_receipt, local_certificate = _sample_value_arm(
            model,
            spec=spec,
            noop_prompt_embeds=noop_prompt_embeds,
            rank=distributed.rank,
            source_tokens=source_tokens,
            sample_kwargs=sample_kwargs,
        )
    if tuple(int(item) for item in generated_latent.shape) != expected_latent_shape:
        raise SourceValueResidualOracleError("generated latent shape differs")
    generated_identity = tensor_identity(
        generated_latent, label="full generated latent"
    )
    local_certificate["generated_latent"] = generated_identity
    local_certificate["z0_control"] = None
    if spec.arm == "Z0":
        if official_o0_latent is None or official_o0_certificate is None:
            raise SourceValueResidualOracleError("Z0 official O0 result is missing")
        official_identity = tensor_identity(
            official_o0_latent, label="full generated latent"
        )
        byte_exact = bool(torch.equal(official_o0_latent, generated_latent))
        if not byte_exact or official_identity != generated_identity:
            raise SourceValueResidualOracleError(
                "Z0 fixed-zero wrapper differs bytewise from same-job official O0"
            )
        local_certificate["z0_control"] = {
            "byte_exact": True,
            "official_o0_latent": official_identity,
            "official_o0_runtime": official_o0_certificate,
            "historical_o0_mp4_sha256": PINNED_HISTORICAL_O0_MP4_SHA256,
        }
        del official_o0_latent
    freeze_after = carrier_oracle.model_freeze_certificate(model)
    if freeze_after != freeze:
        raise SourceValueResidualOracleError("model freeze certificate changed")

    rank_rows: list[Any] = [None] * EXPECTED_ULYSSES_SIZE
    dist.all_gather_object(rank_rows, local_certificate)
    four_rank_runtime = validate_four_rank_runtime(rank_rows, spec=spec)
    model.to("cpu")
    del source_latent
    torch.cuda.empty_cache()

    runtime_versions = {
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "transformers": transformers_version,
        "diffusers": diffusers_version,
    }
    pairing = carrier_oracle.causal_pairing_contract(
        method_source_revision=args.method_source_revision,
        method_source_archive_sha256=args.method_source_archive_sha256,
        bernini_commit=bernini_revision,
        veomni_commit=veomni_revision,
        bernini_inference_files=inference_file_hashes,
        checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        checkpoint_content_identity=checkpoint_content_identity,
        source_sha256=source_sha256,
        instruction_sha256=hashlib.sha256(args.instruction.encode("utf-8")).hexdigest(),
        action_prompt_sha256=hashlib.sha256(full_prompt.encode("utf-8")).hexdigest(),
        negative_prompt_sha256=hashlib.sha256(negative_prompt.encode("utf-8")).hexdigest(),
        source_metadata=source_metadata,
        steps=EXPECTED_STEPS,
        seed=EXPECTED_SEED,
        runtime_versions=runtime_versions,
    )

    if distributed.rank == 0:
        receipt_published = False
        try:
            vae.to(device)
            with torch.no_grad():
                decoded = _vae_decode(vae, generated_latent)
            vae.to("cpu")
            if tuple(int(item) for item in decoded.shape) != (
                EXPECTED_FRAMES,
                int(bucket[0]),
                int(bucket[1]),
                3,
            ):
                raise SourceValueResidualOracleError("decoded output shape differs")
            save_video_atomically(
                decoded,
                output_path,
                fps=int(legacy.FPS),
                save_output_fn=save_output,
            )
            from tools import materialize_vae

            encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(
                output_path
            )
            legacy.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
            if tuple(encoded_hw) != tuple(bucket):
                raise SourceValueResidualOracleError("encoded output geometry differs")
            receipt = build_receipt(
                args=args,
                spec=spec,
                source_path=source_path,
                source_sha256=source_sha256,
                source_metadata=source_metadata,
                source_tokens=source_tokens,
                output_path=output_path,
                output_sha256=legacy.file_sha256(output_path),
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                inference_file_hashes=inference_file_hashes,
                runtime_versions=runtime_versions,
                freeze_certificate=freeze_after,
                four_rank_runtime=four_rank_runtime,
                rank0_core_receipt=core_receipt,
                pairing=pairing,
                checkpoint_content_identity=checkpoint_content_identity,
            )
            write_receipt_atomically(receipt_path, receipt)
            receipt_published = True
            print(legacy.canonical_json_bytes(receipt).decode("utf-8"), flush=True)
        except BaseException:
            # Both final paths were proven fresh by ``_resolve_output``.  Until
            # the canonical receipt is durable, treat MP4+receipt as one
            # transaction and leave neither an orphan nor a false success.
            if not receipt_published:
                unlink_fresh_artifact(receipt_path)
                unlink_fresh_artifact(output_path)
            raise

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_NAMES",
    "ARM_SPECS",
    "ARM_TABLE_SHA256",
    "ArmSpec",
    "EXPECTED_INSTRUCTION",
    "EXPECTED_BUCKET_HW",
    "EXPECTED_ORIGINAL_SOURCE_PATH",
    "EXPECTED_SEED",
    "EXPECTED_SOURCE_SHA256",
    "PINNED_HISTORICAL_O0_MP4_SHA256",
    "RECEIPT_SCHEMA",
    "SourceValueResidualOracleError",
    "arm_spec",
    "build_parser",
    "build_receipt",
    "main",
    "output_transaction_token",
    "save_video_atomically",
    "tensor_identity",
    "unlink_fresh_artifact",
    "validate_cli",
    "validate_four_rank_runtime",
    "validate_value_runtime_certificate",
    "write_receipt_atomically",
]

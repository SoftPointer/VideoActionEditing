#!/usr/bin/env python3
"""Create-only real-checkpoint WORLD4 parity smoke for the relational hook.

Live mode is intentionally tiny: it reuses the byte-pinned E00 source/runtime
binding from :mod:`auh_source_owned_role_locator_v15_adapter`, materializes one
official frozen Bernini-R transformer input, and executes exactly two forwards
on the same prepared tensor objects:

1. observer OFF (all processors official),
2. observer ON at blocks 6/12/18/24.

The second output must be bit-exact to the first and every prepared-state hash
and parameter/buffer version marker must remain unchanged.  Only after the ON
forward has returned are the hook's Q/K and derived role-proxy shards gathered
with the WORLD4 group.  The gathered tensors are validated, committed to the
ephemeral ``InMemoryNativeCaptureBank``, and zeroized.  Rank zero writes one
new JSON receipt with ``open(..., 'x')``; no tensor artifact is persisted.

This smoke has no decode, target-video, optimizer, training, adapter, routing,
or injection path.  Import/contract mode does not initialize distributed state
or CUDA.  Live mode is deliberately not launched by this module itself; it
must be invoked by an explicit four-rank ``torch.distributed.run`` command.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.distributed as dist


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import auh_source_owned_role_locator_v15_adapter as site  # noqa: E402
import infer_native_self_generated_relational_graph_observer_v1 as native  # noqa: E402
import native_relational_attention_hook_v1 as attention_hook  # noqa: E402
import source_owned_role_locator_v15 as locator  # noqa: E402
import source_owned_role_locator_v15b_e00_asset as role_asset  # noqa: E402


METHOD = "bernini-auh-native-relational-attention-parity-smoke-v1"
SCHEMA_VERSION = "bernini-auh-native-relational-attention-parity-smoke-v1"
CONTRACT_SCHEMA_VERSION = "bernini-auh-native-relational-attention-parity-contract-v1"
WORLD_SIZE = 4
BLOCKS = attention_hook.BLOCKS
NULL_ROLE = "null_context"
RESPONSIBILITY_KIND = attention_hook.RESPONSIBILITY_KIND
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AUHNativeRelationalParityError(RuntimeError):
    """Fail-closed real-checkpoint parity or authority violation."""


def parity_smoke_contract() -> Mapping[str, Any]:
    value = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "method": METHOD,
        "checkpoint_tree_sha256": site.CHECKPOINT_TREE_SHA256,
        "official_transformer_source_sha256": (
            attention_hook.OFFICIAL_TRANSFORMER_SOURCE_SHA256
        ),
        "source_video_sha256": site.SOURCE_VIDEO_SHA256,
        "source_manifest_sha256": site.SOURCE_MANIFEST_SHA256,
        "world_size": WORLD_SIZE,
        "blocks": list(BLOCKS),
        "scheduler": {
            "name": "UniPCMultistepScheduler",
            "steps": 40,
            "index": 37,
            "timestep": site.TIMESTEP_VALUE,
            "sigma": site.SCHEDULE_SIGMA,
        },
        "forward_order": ["observer_off", "observer_on"],
        "same_prepared_tensor_objects": True,
        "output_bit_exact_required": True,
        "output_sha256_equal_required": True,
        "prepared_state_sha256_equal_required": True,
        "module_state_version_digest_equal_required": True,
        "added_collective_location": "after_observer_on_transformer_forward_returned",
        "added_collectives_inside_attention": 0,
        "world4_tensor_collectives": [
            "attn1_post_rope_qk_rank_major",
            RESPONSIBILITY_KIND + "_rank_major",
        ],
        "responsibility_kind": RESPONSIBILITY_KIND,
        "backend_attention_weights_observed": False,
        "persistent_tensor_artifact_authorized": False,
        "receipt_write_mode": "rank0_create_only_after_all_rank_success",
        "base_frozen": True,
        "adapter_or_lora_loaded": False,
        "parameter_updates": 0,
        "candidate_output_modified": False,
        "renderer_or_decoder_called": False,
        "optimizer_created": False,
        "route_or_injection_called": False,
        "target_inputs_consumed": False,
        "training_authorized": False,
        "gpu_launch_authorized_by_contract_print": False,
        "scientific_claim_authorized": False,
    }
    return {**value, "digest": locator.object_sha256(value)}


def remote_launch_template() -> Mapping[str, Any]:
    """Return a non-executing torchrun template for an already allocated node."""

    value = {
        "python": "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12",
        "launcher": "python -m torch.distributed.run",
        "nproc_per_node": WORLD_SIZE,
        "required_environment": {
            "MODELING_BACKEND": "hf",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        "entrypoint": Path(__file__).name,
        "arguments": ["--run", "--output", "ABSENT_CREATE_ONLY_JSON_PATH"],
        "launch_executed": False,
        "gpu_launch_authorized": False,
    }
    return {**value, "digest": locator.object_sha256(value)}


def _all_rank_rows(value: Any) -> list[Any]:
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, value)
    return rows


def _require_all_rank_equal(value: Any, *, label: str) -> None:
    if _all_rank_rows(value) != [value] * WORLD_SIZE:
        raise AUHNativeRelationalParityError(f"{label} differs across WORLD4")


def _role_partition_from_e00_event(event_spec: Any) -> attention_hook.ExhaustiveTextRolePartition:
    if (
        not isinstance(event_spec, locator.SourceRoleEventSpec)
        or event_spec.event_sha256 != role_asset.EVENT_SHA256
        or event_spec.role_names != role_asset.ROLE_NAMES
    ):
        raise AUHNativeRelationalParityError("E00 instance-role event authority differs")
    role_names = tuple(event_spec.role_names) + (NULL_ROLE,)
    null_index = len(role_names) - 1
    owner = [null_index] * site.RENDERER_TEXT_LENGTH
    for role_index, role in enumerate(event_spec.roles):
        if not 0 <= role.token_start < role.token_end <= site.RENDERER_TEXT_LENGTH:
            raise AUHNativeRelationalParityError("E00 role token span is outside text K")
        for token_index in range(role.token_start, role.token_end):
            if owner[token_index] != null_index:
                raise AUHNativeRelationalParityError("E00 role token spans overlap")
            owner[token_index] = role_index
    try:
        return attention_hook.ExhaustiveTextRolePartition(
            role_names=role_names,
            token_to_role=tuple(owner),
        )
    except attention_hook.NativeRelationalAttentionHookError as error:
        raise AUHNativeRelationalParityError("cannot form exhaustive E00 role partition") from error


def _tensor_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise AUHNativeRelationalParityError(f"{label} is not a material tensor")
    # The pinned locator predates scalar-tensor byte views on torch 2.7:
    # ``scalar.view(torch.uint8)`` is rejected when element sizes differ.
    # Reshaping to one logical element preserves every scalar byte while also
    # keeping its original shape in the surrounding prepared-state receipt.
    logical = value.detach().reshape(1) if value.ndim == 0 else value
    digest = locator.tensor_sha256(logical)
    if _SHA256_RE.fullmatch(digest) is None:
        raise AUHNativeRelationalParityError(f"{label} digest differs")
    return digest


def _prepared_state_receipt(prepared: Any) -> Mapping[str, Any]:
    if not isinstance(prepared, site._PreparedForward):
        raise AUHNativeRelationalParityError("prepared forward ABI differs")
    tensor_rows = {
        "hidden_states": _tensor_digest(prepared.hidden_states, label="hidden states"),
        "encoder_hidden_states": _tensor_digest(
            prepared.encoder_hidden_states, label="encoder hidden states"
        ),
        "timestep_proj": _tensor_digest(prepared.timestep_proj, label="timestep projection"),
        "temb": _tensor_digest(prepared.temb, label="time embedding"),
        "rotary_emb": _tensor_digest(prepared.rotary_emb, label="rotary embedding"),
    }
    kwargs = {}
    for name, value in sorted(prepared.kwargs.items()):
        if isinstance(value, torch.Tensor):
            kwargs[name] = {
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
                "sha256": _tensor_digest(value, label=name),
            }
        elif isinstance(value, (int, float, str, bool)) or value is None:
            if isinstance(value, float) and not math.isfinite(value):
                raise AUHNativeRelationalParityError("prepared scalar is non-finite")
            kwargs[name] = value
        else:
            raise AUHNativeRelationalParityError(f"prepared kwarg {name} is unsupported")
    value = {
        "tensor_sha256": tensor_rows,
        "tensor_object_ids": {
            "hidden_states": id(prepared.hidden_states),
            "encoder_hidden_states": id(prepared.encoder_hidden_states),
            "timestep_proj": id(prepared.timestep_proj),
            "temb": id(prepared.temb),
            "rotary_emb": id(prepared.rotary_emb),
        },
        "batch_image_vae_seqlen": list(prepared.batch_image_vae_seqlen),
        "text_features_length": list(prepared.text_features_length),
        "kwargs": kwargs,
    }
    return {**value, "digest": locator.object_sha256(value)}


def _module_state_version_receipt(module: torch.nn.Module) -> Mapping[str, Any]:
    if not isinstance(module, torch.nn.Module):
        raise AUHNativeRelationalParityError("transformer is not a torch module")
    rows = []
    for kind, iterator in (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, value in iterator:
            if not isinstance(value, torch.Tensor):
                raise AUHNativeRelationalParityError("module state contains a non-tensor")
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": [int(item) for item in value.shape],
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "data_ptr": int(value.data_ptr()),
                    "logical_bytes": int(value.numel() * value.element_size()),
                    "version": int(value._version),
                    "requires_grad": bool(value.requires_grad),
                    "grad_is_none": value.grad is None,
                }
            )
    value = {
        "training": bool(module.training),
        "row_count": len(rows),
        "rows": rows,
    }
    return {"digest": locator.object_sha256(value), "row_count": len(rows)}


def _check_runtime_world4() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise AUHNativeRelationalParityError("WORLD4 process group is not initialized")
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        env_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        env_world = int(os.environ["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as error:
        raise AUHNativeRelationalParityError("torchrun rank environment differs") from error
    if world != WORLD_SIZE or env_world != WORLD_SIZE or rank != env_rank:
        raise AUHNativeRelationalParityError("runtime is not exact WORLD4")
    if not 0 <= local_rank < WORLD_SIZE:
        raise AUHNativeRelationalParityError("LOCAL_RANK lies outside WORLD4")
    return rank, local_rank


def _gather_one_block_after_forward(
    shard: attention_hook.World4BlockRankShard,
    *,
    invocation: native.CaptureInvocation,
    role_partition: attention_hook.ExhaustiveTextRolePartition,
) -> tuple[attention_hook.World4BlockRankShard, ...]:
    """Run the only added collectives; caller invokes this post-forward only."""

    qk_local: torch.Tensor | None = None
    proxy_local: torch.Tensor | None = None
    qk_flat: torch.Tensor | None = None
    proxy_flat: torch.Tensor | None = None
    rebuilt: tuple[attention_hook.World4BlockRankShard, ...] = ()
    succeeded = False
    try:
        qk_local, proxy_local, metadata_local = (
            shard.collective_payload_and_zeroize()
        )
        qk_flat = torch.empty(
            (WORLD_SIZE * int(qk_local.shape[0]), *tuple(qk_local.shape[1:])),
            dtype=qk_local.dtype,
            device=qk_local.device,
        )
        proxy_flat = torch.empty(
            (WORLD_SIZE * int(proxy_local.shape[0]), *tuple(proxy_local.shape[1:])),
            dtype=proxy_local.dtype,
            device=proxy_local.device,
        )
        metadata_rows: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_into_tensor(qk_flat, qk_local)
        dist.all_gather_into_tensor(proxy_flat, proxy_local)
        dist.all_gather_object(metadata_rows, metadata_local)
        qk_rank_major = qk_flat.reshape(
            WORLD_SIZE, *tuple(qk_local.shape)
        ).contiguous()
        proxy_rank_major = proxy_flat.reshape(
            WORLD_SIZE, *tuple(proxy_local.shape)
        ).contiguous()
        # Reconstruction consumes and clears both rank-major copies.
        rebuilt = attention_hook.reconstruct_world4_block_from_collectives(
            invocation=invocation,
            role_partition=role_partition,
            block_index=shard.block_index,
            qk_rank_major=qk_rank_major,
            proxy_rank_major=proxy_rank_major,
            rank_metadata=metadata_rows,
        )
        succeeded = True
        return rebuilt
    finally:
        with torch.inference_mode():
            for value in (qk_local, proxy_local, qk_flat, proxy_flat):
                if isinstance(value, torch.Tensor):
                    value.zero_()
            if not succeeded:
                for row in rebuilt:
                    row.zeroize()


def _capture_summary_and_zeroize(
    captures: Sequence[native.NativeBlockCapture],
    *,
    bank: native.InMemoryNativeCaptureBank,
) -> Sequence[Mapping[str, Any]]:
    rows = []
    for capture in captures:
        proxy = capture.derived_qk_role_responsibility_proxy
        mass_error = float((proxy.float().sum(dim=2) - 1.0).abs().max().item())
        if mass_error > 2.0e-4:
            raise AUHNativeRelationalParityError("assembled derived role proxy mass differs")
        rows.append(
            {
                "block_index": capture.block_index,
                "query_shape": [int(item) for item in capture.query.shape],
                "key_shape": [int(item) for item in capture.key.shape],
                RESPONSIBILITY_KIND + "_shape": [int(item) for item in proxy.shape],
                RESPONSIBILITY_KIND + "_max_mass_error": mass_error,
                "query_dtype": str(capture.query.dtype),
                RESPONSIBILITY_KIND + "_dtype": str(proxy.dtype),
            }
        )
    bank.zeroize(captures)
    if any(
        int(torch.count_nonzero(tensor).item()) != 0
        for capture in captures
        for tensor in (
            capture.query,
            capture.key,
            capture.derived_qk_role_responsibility_proxy,
        )
    ):
        raise AUHNativeRelationalParityError("native capture zeroization failed")
    return rows


def run_real_world4_parity_smoke(output: Path) -> Mapping[str, Any]:
    rank, _local_rank = _check_runtime_world4()
    if not isinstance(output, Path) or not output.is_absolute() or output.is_symlink():
        raise AUHNativeRelationalParityError("output must be an absolute absent plain path")
    if output.exists() or not output.parent.is_dir() or output.parent.is_symlink():
        raise AUHNativeRelationalParityError("output must be absent in a plain existing directory")

    runtime = site.create_auh_bernini_source_role_adapter({})
    event_spec, _raw_asset = role_asset.load_e00_v15b_asset()
    source = runtime.materialize_source(
        event_spec=event_spec, rank=rank, world_size=WORLD_SIZE
    )
    conditioned = source["derive_conditioned_source_text"](
        source["raw_source_text_hidden_states"]
    )
    source_authority = SimpleNamespace(
        source_receipt_sha256=source["source_receipt_sha256"]
    )
    prepared = runtime.prepare_inputs_for_sp(
        conditioned_source_text_hidden_states=conditioned,
        source=source_authority,
        event_spec=event_spec,
        rank=rank,
        world_size=WORLD_SIZE,
    )
    role_partition = _role_partition_from_e00_event(event_spec)
    prepared_before = _prepared_state_receipt(prepared)
    state_before = _module_state_version_receipt(runtime._transformer)

    with torch.inference_mode():
        observer_off = runtime.run_frozen_forward(prepared=prepared)
    off_sha256 = _tensor_digest(observer_off, label="observer OFF output")

    capture_invocation = native.CaptureInvocation(
        "appearance_0",
        "action",
        native.SigmaCell("mid_low", 37, site.SCHEDULE_SIGMA),
        site.NOISY_SOURCE_TENSOR_SHA256,
        site.TIMESTEP_TENSOR_SHA256,
        _tensor_digest(prepared.rotary_emb, label="capture rotary"),
        runtime.source_geometry.height,
        runtime.source_geometry.width,
    )
    rank_invocation = attention_hook.RankCaptureInvocation(
        capture_invocation,
        attention_hook.World4RankLayout(
            rank, runtime.source_geometry.height, runtime.source_geometry.width
        ),
        role_partition,
    )
    rank_bank = attention_hook.InMemoryWorld4RankShardBank()
    handle = attention_hook.install_native_relational_attention_hook(
        runtime._transformer, rank_bank=rank_bank
    )
    try:
        with torch.inference_mode(), rank_bank.observe(rank_invocation):
            observer_on = runtime.run_frozen_forward(prepared=prepared)
    finally:
        handle.restore()
    on_sha256 = _tensor_digest(observer_on, label="observer ON output")
    output_bit_exact = bool(torch.equal(observer_off, observer_on))
    if not output_bit_exact or off_sha256 != on_sha256:
        raise AUHNativeRelationalParityError("observer ON changed frozen Bernini output")
    _require_all_rank_equal(off_sha256, label="observer OFF/ON output SHA-256")
    if any(
        wrapper.base_calls != 1 or wrapper.observer_calls != 1
        for wrapper in (*handle.attn1_wrappers, *handle.attn2_wrappers)
    ):
        raise AUHNativeRelationalParityError("hook did not observe exactly eight official calls")

    prepared_after_forward = _prepared_state_receipt(prepared)
    if prepared_after_forward != prepared_before:
        raise AUHNativeRelationalParityError("prepared state changed across OFF/ON forwards")

    # All additional observer collectives begin strictly after observer_on was
    # materialized, hashed, and checked bit-exact above.
    local_shards = rank_bank.take_rank(rank_invocation)
    gathered_shards: list[attention_hook.World4BlockRankShard] = []
    for shard in local_shards:
        gathered_shards.extend(
            _gather_one_block_after_forward(
                shard,
                invocation=capture_invocation,
                role_partition=role_partition,
            )
        )
    native_bank = native.InMemoryNativeCaptureBank()
    commit_receipt = attention_hook.commit_world4_shards_to_native_bank(
        native_bank=native_bank,
        invocation=capture_invocation,
        rank_shards=gathered_shards,
    )
    captures = native_bank.consume(capture_invocation)
    capture_summary = _capture_summary_and_zeroize(captures, bank=native_bank)
    bank_receipt = native_bank.receipt()
    rank_bank_receipt = rank_bank.receipt()
    state_after = _module_state_version_receipt(runtime._transformer)
    prepared_after = _prepared_state_receipt(prepared)
    if state_after != state_before or prepared_after != prepared_before:
        raise AUHNativeRelationalParityError("frozen module/prepared state changed")
    if (
        bank_receipt["capture_count"] != len(BLOCKS)
        or bank_receipt["zeroized_count"] != len(BLOCKS)
        or bank_receipt["resident_invocation_count"] != 0
        or rank_bank_receipt["resident_rank_invocations"] != 0
    ):
        raise AUHNativeRelationalParityError("ephemeral capture banks did not close")

    local_receipt = {
        "rank": rank,
        "device": str(runtime.device),
        "observer_off_output_sha256": off_sha256,
        "observer_on_output_sha256": on_sha256,
        "output_bit_exact": output_bit_exact,
        "prepared_state_digest": prepared_before["digest"],
        "module_state_version_digest": state_before["digest"],
        "hook_calls": {
            "attn1": [wrapper.observer_calls for wrapper in handle.attn1_wrappers],
            "attn2": [wrapper.observer_calls for wrapper in handle.attn2_wrappers],
        },
        "hook_restored": handle.restored,
        "rank_bank": dict(rank_bank_receipt),
        "native_bank": dict(bank_receipt),
        "commit": dict(commit_receipt),
        "capture_summary": list(capture_summary),
    }
    local_receipt = {
        **local_receipt,
        "digest": locator.object_sha256(local_receipt),
    }
    rank_receipts = _all_rank_rows(local_receipt)
    if sorted(row.get("rank", -1) for row in rank_receipts) != list(range(WORLD_SIZE)):
        raise AUHNativeRelationalParityError("WORLD4 final receipt rank registry differs")
    if len({row.get("observer_on_output_sha256") for row in rank_receipts}) != 1:
        raise AUHNativeRelationalParityError("WORLD4 final output hashes differ")

    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": dict(parity_smoke_contract()),
        "site_binding": runtime.binding_receipt(),
        "role_asset_sha256": role_asset.ASSET_SHA256,
        "role_event_sha256": role_asset.EVENT_SHA256,
        "role_names": list(role_partition.role_names),
        "role_partition_sha256": role_partition.digest,
        "rank_receipts": rank_receipts,
        "observer_off_on_output_bit_exact": True,
        "observer_off_on_output_sha256_equal": True,
        "same_prepared_tensor_objects_and_hashes": True,
        "module_state_version_digest_equal": True,
        "official_output_forwarded_same_object_inside_each_wrapped_processor": True,
        "added_collectives_started_only_after_observer_on_forward_returned": True,
        "added_collectives_inside_attention": 0,
        "backend_attention_weights_observed": False,
        "responsibility_kind": RESPONSIBILITY_KIND,
        "all_raw_rank_and_global_captures_zeroized": True,
        "persistent_tensor_artifact_created": False,
        "output_video_created": False,
        "candidate_output_modified": False,
        "parameter_updates": 0,
        "target_inputs_consumed": False,
        "scientific_claim_authorized": False,
        "gpu_launch_authorized_for_followup": False,
        "status": "MECHANICAL_NATIVE_PARITY_PASS_NOT_REPRESENTATION_EVIDENCE",
    }
    value = {**value, "digest": locator.object_sha256(value)}

    status: list[Any] = [None]
    if rank == 0:
        try:
            encoded = (
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            )
            with output.open("x", encoding="utf-8") as handle_out:
                handle_out.write(encoded)
                handle_out.flush()
                os.fsync(handle_out.fileno())
            status[0] = {"ok": True, "receipt_sha256": value["digest"]}
        except Exception as error:
            status[0] = {
                "ok": False,
                "type": type(error).__name__,
                "message": str(error),
            }
    dist.broadcast_object_list(status, src=0)
    if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
        raise AUHNativeRelationalParityError(f"create-only receipt write failed: {status[0]}")
    dist.barrier()
    return value


def _initialize_world4() -> None:
    if dist.is_initialized():
        raise AUHNativeRelationalParityError("process group was initialized before live entry")
    try:
        world = int(os.environ.get("WORLD_SIZE", ""))
    except ValueError as error:
        raise AUHNativeRelationalParityError("WORLD_SIZE is invalid") from error
    if world != WORLD_SIZE:
        raise AUHNativeRelationalParityError("live smoke requires torchrun WORLD4")
    dist.init_process_group(backend="nccl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contract", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_contract:
        if args.output is not None:
            raise AUHNativeRelationalParityError("contract print does not accept output")
        value = {
            "contract": parity_smoke_contract(),
            "remote_launch_template": remote_launch_template(),
        }
        sys.stdout.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        )
        return 0
    if args.output is None:
        raise AUHNativeRelationalParityError("live smoke requires --output")
    _initialize_world4()
    try:
        run_real_world4_parity_smoke(args.output)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUHNativeRelationalParityError",
    "BLOCKS",
    "CONTRACT_SCHEMA_VERSION",
    "METHOD",
    "SCHEMA_VERSION",
    "parity_smoke_contract",
    "remote_launch_template",
    "run_real_world4_parity_smoke",
]

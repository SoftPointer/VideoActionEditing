#!/usr/bin/env python3
"""Frozen E00 SP4 observer probe with 64 explicit null-span maps.

This harness runs adapter OFF once and ON twice.  It performs no optimization,
route, decode, or action-success test.  The only distributed operation added
by the observer is an explicit post-attn2 gather of a field-closed 75-channel
diagnostic shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


_miopen_root = os.environ.get("V15B_MIOPEN_CACHE_ROOT")
if _miopen_root is not None:
    _local_rank = os.environ.get("LOCAL_RANK")
    if _local_rank not in {"0", "1", "2", "3"}:
        raise RuntimeError("r6 MIOpen cache requires LOCAL_RANK 0..3")
    _rank_cache = Path(_miopen_root) / f"rank_{_local_rank}"
    _user_db = _rank_cache / "miopen-user"
    _kernel_cache = _rank_cache / "miopen-custom"
    if (
        not Path(_miopen_root).is_absolute()
        or not _user_db.is_dir()
        or not _kernel_cache.is_dir()
        or _user_db.is_symlink()
        or _kernel_cache.is_symlink()
    ):
        raise RuntimeError("r6 rank-local MIOpen cache directories are unsafe")
    os.environ["MIOPEN_USER_DB_PATH"] = str(_user_db)
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(_kernel_cache)

import numpy as np
import torch
import torch.distributed as dist


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import probe_source_owned_role_locator_v15_sp4 as base  # noqa: E402
import source_owned_role_locator_v15 as locator  # noqa: E402
import source_owned_role_locator_v15b_e00_asset as asset_v15b  # noqa: E402
import source_owned_role_mask_calibration_v15b_r5 as calibration  # noqa: E402
import source_owned_role_null_bank_observer_v15b_r6 as observer_r6  # noqa: E402
import source_owned_role_null_registry_v15b_r6 as registry_r6  # noqa: E402


PROBE_SCHEMA_VERSION = "bernini-source-owned-instance-role-null64-sp4-probe-v15b-r6"
DIAGNOSTIC_SCHEMA_VERSION = "bernini-source-owned-instance-role-null64-affinity-v15b-r6"
SP_SIZE = 4
BLOCKS = (4, 9, 14, 19, 24)


class SourceRoleV15BR6ProbeError(RuntimeError):
    """Fail-closed r6 probe violation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _explicit_gather_block(
    local: observer_r6.NullBankAffinityShardV15BR6,
    *,
    rank: int,
) -> observer_r6.GlobalNullBankAffinityV15BR6 | None:
    tensor = local.padded_collective_tensor()
    gathered_tensors = [torch.empty_like(tensor) for _ in range(SP_SIZE)]
    dist.all_gather(gathered_tensors, tensor)
    gathered_metadata: list[Any] = [None for _ in range(SP_SIZE)]
    dist.all_gather_object(gathered_metadata, local.collective_metadata())
    if rank != 0:
        return None
    shards = tuple(
        observer_r6.NullBankAffinityShardV15BR6.from_collective(value, metadata)
        for value, metadata in zip(gathered_tensors, gathered_metadata)
    )
    return observer_r6.assemble_global_null_bank_affinity_v15b_r6(shards)


def _tensor_hashes(tensors: Mapping[str, torch.Tensor]) -> Mapping[str, str]:
    return {name: locator.tensor_sha256(value) for name, value in tensors.items()}


def _save_diagnostics_create_only(
    path: Path,
    *,
    globals_by_block: Sequence[observer_r6.GlobalNullBankAffinityV15BR6],
    spec: locator.SourceRoleEventSpec,
    runtime_registry_receipt: Mapping[str, Any],
    source_binding_receipt_sha256: str,
    source_text_provenance_sha256: str,
) -> Mapping[str, Any]:
    if path.exists():
        raise SourceRoleV15BR6ProbeError("r6 diagnostic path already exists")
    if (
        len(globals_by_block) != len(BLOCKS)
        or tuple(item.block_index for item in globals_by_block) != BLOCKS
    ):
        raise SourceRoleV15BR6ProbeError("r6 diagnostic block registry differs")
    real = torch.stack([item.affinity.cpu() for item in globals_by_block], dim=0)
    legacy = torch.stack(
        [item.legacy_null_affinity.cpu() for item in globals_by_block], dim=0
    )
    shuffled = torch.stack(
        [item.shuffled_affinity.cpu() for item in globals_by_block], dim=0
    )
    null_spans = torch.stack(
        [item.null_span_affinity.cpu() for item in globals_by_block], dim=0
    )
    if (
        tuple(real.shape) != (5, 5, 21, 37, 25)
        or tuple(null_spans.shape) != (5, 64, 21, 37, 25)
    ):
        raise SourceRoleV15BR6ProbeError("r6 global diagnostic geometry differs")
    calibrated = calibration.calibrate_source_role_maps(
        np.ascontiguousarray(real.numpy()),
        null_span_maps=np.ascontiguousarray(null_spans.numpy()),
        null_registry_sha256=registry_r6.REGISTRY_SHA256,
    )
    tensors: dict[str, torch.Tensor] = {
        "aggregate_affinity": real.mean(dim=0).contiguous(),
        "aggregate_legacy_null_affinity": legacy.mean(dim=0).contiguous(),
        "aggregate_shuffled_affinity": shuffled.mean(dim=0).contiguous(),
        "aggregate_null_span_affinity": null_spans.mean(dim=0).contiguous(),
        "calibration_standardized_role_maps": torch.from_numpy(
            calibrated.standardized_role_maps
        ).contiguous(),
        "calibration_exploratory_track_masks_u8": torch.from_numpy(
            calibrated.exploratory_track_masks.astype(np.uint8)
        ).contiguous(),
        "calibration_strict_block_masks_u8": torch.from_numpy(
            calibrated.strict_block_masks.astype(np.uint8)
        ).contiguous(),
        "calibration_strict_aggregate_masks_u8": torch.from_numpy(
            calibrated.strict_aggregate_masks.astype(np.uint8)
        ).contiguous(),
    }
    for offset, block in enumerate(BLOCKS):
        tensors[f"block_{block:02d}_affinity"] = real[offset].contiguous()
        tensors[f"block_{block:02d}_legacy_null_affinity"] = legacy[
            offset
        ].contiguous()
        tensors[f"block_{block:02d}_shuffled_affinity"] = shuffled[
            offset
        ].contiguous()
        tensors[f"block_{block:02d}_null_span_affinity"] = null_spans[
            offset
        ].contiguous()
    metadata_payload = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "event_id": spec.event_id,
        "role_asset_sha256": asset_v15b.ASSET_SHA256,
        "role_event_sha256": spec.event_sha256,
        "role_names": list(spec.role_names),
        "null_registry_sha256": registry_r6.REGISTRY_SHA256,
        "null_span_count": registry_r6.SPAN_COUNT,
        "runtime_null_registry_receipt_sha256": runtime_registry_receipt[
            "receipt_sha256"
        ],
        "source_binding_receipt_sha256": source_binding_receipt_sha256,
        "source_text_provenance_sha256": source_text_provenance_sha256,
        "geometry": [21, 37, 25],
        "selected_block_indices": list(BLOCKS),
        "tensor_sha256": _tensor_hashes(tensors),
        "calibration_receipt": dict(calibrated.receipt),
        "observer_only": True,
        "localization_semantically_certified": False,
        "action_success_certified": False,
        "route_authorized": False,
        "training_authorized": False,
        "decode_authorized": False,
    }
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(path),
        metadata={
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "metadata_sha256": locator.object_sha256(metadata_payload),
        },
    )
    return {
        **metadata_payload,
        "metadata_sha256": locator.object_sha256(metadata_payload),
        "path": str(path.resolve(strict=True)),
        "file_sha256": _sha256_file(path),
    }


def run_probe(args: argparse.Namespace) -> Mapping[str, Any] | None:
    rank, world, device = base._distributed_preflight()
    spec, asset_raw = asset_v15b.load_e00_v15b_asset(args.role_asset)
    registry = registry_r6.load_null_registry_v15b_r6(args.null_registry)
    if args.event_id != spec.event_id or tuple(args.block_indices) != BLOCKS:
        raise SourceRoleV15BR6ProbeError("r6 event/block lock differs")
    factory = base._load_factory(args.runtime_adapter)
    adapter_config = json.loads(args.adapter_config)
    if not isinstance(adapter_config, Mapping):
        raise SourceRoleV15BR6ProbeError("adapter config is not one object")
    adapter = factory(dict(adapter_config))
    adapter_contract = base.validate_adapter_contract(adapter, BLOCKS)
    binding_method = getattr(adapter, "binding_receipt", None)
    if not callable(binding_method):
        raise SourceRoleV15BR6ProbeError("adapter lacks source binding receipt")
    binding = binding_method()
    if (
        not isinstance(binding, Mapping)
        or binding.get("role_asset_sha256") != asset_v15b.ASSET_SHA256
        or binding.get("role_event_sha256") != asset_v15b.EVENT_SHA256
        or binding.get("source_video_sha256") != asset_v15b.SOURCE_VIDEO_SHA256
        or any(
            binding.get(name) is not False
            for name in (
                "action_success_authorized",
                "route_authorized",
                "training_authorized",
                "decode_authorized",
            )
        )
    ):
        raise SourceRoleV15BR6ProbeError("adapter/source binding differs")
    geometry = getattr(adapter, "source_geometry", None)
    if not isinstance(geometry, locator.SourceVisualGeometry):
        raise SourceRoleV15BR6ProbeError("adapter lacks source geometry")
    layout = locator.UlyssesVisualShard(geometry=geometry, rank=rank, size=world)
    source = base.SourceMaterialization.from_adapter(
        adapter.materialize_source(event_spec=spec, rank=rank, world_size=world)
    )
    runtime_registry_receipt = registry_r6.validate_runtime_null_registry_v15b_r6(
        source.tokenizer, registry
    )
    gathered_registry_receipts: list[Any] = [None for _ in range(world)]
    dist.all_gather_object(gathered_registry_receipts, runtime_registry_receipt)
    if any(item != runtime_registry_receipt for item in gathered_registry_receipts):
        raise SourceRoleV15BR6ProbeError("runtime null registry differs across SP4")
    bank = observer_r6.NullBankCaptureBankV15BR6(BLOCKS, registry=registry)
    patch_handle = observer_r6.install_source_owned_role_null_bank_observer_v15b_r6(
        adapter.model, capture_bank=bank
    )
    initial_state = base._module_state_receipt(adapter.model)
    initial_rng = base._rng_snapshot(device)
    try:
        off = base._run_arm(
            adapter=adapter,
            spec=spec,
            capture_bank=bank,
            layout=layout,
            initial_rng=initial_rng,
            observer_enabled=False,
            step_index=0,
            device=device,
        )
        if bank.capture_count != 0:
            raise SourceRoleV15BR6ProbeError("observer OFF captured r6 maps")
        on = base._run_arm(
            adapter=adapter,
            spec=spec,
            capture_bank=bank,
            layout=layout,
            initial_rng=initial_rng,
            observer_enabled=True,
            step_index=0,
            device=device,
        )
        repeat = base._run_arm(
            adapter=adapter,
            spec=spec,
            capture_bank=bank,
            layout=layout,
            initial_rng=initial_rng,
            observer_enabled=True,
            step_index=1,
            device=device,
        )
    finally:
        patch_handle.restore()
    if not torch.equal(off.output, on.output) or off.output_sha256 != on.output_sha256:
        raise SourceRoleV15BR6ProbeError("r6 observer changed frozen output")
    if not base._rng_equal(off.post_rng, on.post_rng):
        raise SourceRoleV15BR6ProbeError("r6 observer changed RNG")
    if (
        not torch.equal(on.output, repeat.output)
        or on.output_sha256 != repeat.output_sha256
        or not base._rng_equal(on.post_rng, repeat.post_rng)
    ):
        raise SourceRoleV15BR6ProbeError("r6 repeated output/RNG differs")
    if not (
        off.state_receipt == initial_state
        and on.state_receipt == initial_state
        and repeat.state_receipt == initial_state
    ):
        raise SourceRoleV15BR6ProbeError("r6 changed frozen model state")
    if not (
        off.provenance_sha256 == on.provenance_sha256 == repeat.provenance_sha256
        and off.source_receipt_sha256
        == on.source_receipt_sha256
        == repeat.source_receipt_sha256
        == binding["binding_receipt_sha256"]
    ):
        raise SourceRoleV15BR6ProbeError("r6 source/token provenance differs")
    if bank.capture_count != 2 * len(BLOCKS):
        raise SourceRoleV15BR6ProbeError("r6 capture cardinality differs")

    globals_by_block: list[observer_r6.GlobalNullBankAffinityV15BR6] = []
    repeat_globals: list[observer_r6.GlobalNullBankAffinityV15BR6] = []
    for block in BLOCKS:
        local = bank.shards_for(event_id=spec.event_id, step_index=0, block_index=block)
        repeated = bank.shards_for(
            event_id=spec.event_id, step_index=1, block_index=block
        )
        if len(local) != 1 or len(repeated) != 1:
            raise SourceRoleV15BR6ProbeError("r6 rank-local capture differs")
        first, second = local[0], repeated[0]
        if not all(
            torch.equal(getattr(first, name), getattr(second, name))
            for name in (
                "affinity",
                "legacy_null_affinity",
                "shuffled_affinity",
                "null_span_affinity",
            )
        ):
            raise SourceRoleV15BR6ProbeError("r6 rank-local repeat differs")
        first_global = _explicit_gather_block(first, rank=rank)
        second_global = _explicit_gather_block(second, rank=rank)
        if first_global is not None:
            globals_by_block.append(first_global)
            if second_global is None:
                raise SourceRoleV15BR6ProbeError("r6 repeat gather missing")
            repeat_globals.append(second_global)
    dist.barrier()
    if rank != 0:
        return None
    for first, second in zip(globals_by_block, repeat_globals):
        if not all(
            torch.equal(getattr(first, name), getattr(second, name))
            for name in (
                "affinity",
                "legacy_null_affinity",
                "shuffled_affinity",
                "null_span_affinity",
            )
        ):
            raise SourceRoleV15BR6ProbeError("r6 global repeat differs")
    diagnostics = _save_diagnostics_create_only(
        Path(args.diagnostics_output),
        globals_by_block=globals_by_block,
        spec=spec,
        runtime_registry_receipt=runtime_registry_receipt,
        source_binding_receipt_sha256=on.source_receipt_sha256,
        source_text_provenance_sha256=on.provenance_sha256,
    )
    receipt_payload = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "observer_only_sp4_null64_mechanical_pass_overlay_pending",
        "event_id": spec.event_id,
        "role_asset_sha256": asset_v15b.ASSET_SHA256,
        "role_event_sha256": asset_v15b.EVENT_SHA256,
        "role_names": list(spec.role_names),
        "semantic_contract": asset_raw["semantic_contract"],
        "competition_groups": asset_raw["competition_groups"],
        "independent_roles": asset_raw["independent_roles"],
        "null_registry_sha256": registry.registry_sha256,
        "null_span_count": registry_r6.SPAN_COUNT,
        "runtime_null_registry_receipt": dict(runtime_registry_receipt),
        "adapter_contract": adapter_contract,
        "source_binding": dict(binding),
        "world_size": world,
        "selected_block_indices": list(BLOCKS),
        "geometry": [geometry.phases, geometry.height, geometry.width],
        "source_text_provenance_sha256": on.provenance_sha256,
        "frozen_output_sha256": on.output_sha256,
        "diagnostics": diagnostics,
        "mechanical_gates": {
            "off_on_output_torch_equal": True,
            "off_on_output_byte_sha_equal": True,
            "off_on_rng_equal": True,
            "parameter_buffer_grad_state_unchanged": True,
            "one_capture_per_selected_block_rank": True,
            "repeat_output_rng_role_and_64null_bit_deterministic": True,
            "explicit_75_channel_collective_outside_attn2": True,
            "exact_source_checkpoint_latent_noise_timestep_token_embedding_binding": True,
            "exact_preregistered_64_null_token_span_binding": True,
        },
        "localization_semantically_certified": False,
        "action_success_certified": False,
        "route_authorized": False,
        "training_authorized": False,
        "decode_authorized": False,
    }
    receipt = {**receipt_payload, "receipt_sha256": locator.object_sha256(receipt_payload)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    except FileExistsError as error:
        raise SourceRoleV15BR6ProbeError("r6 receipt already exists") from error
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-adapter", required=True)
    parser.add_argument("--adapter-config", default="{}")
    parser.add_argument("--role-asset", type=Path, default=asset_v15b.DEFAULT_ASSET)
    parser.add_argument("--null-registry", type=Path, default=registry_r6.DEFAULT_ASSET)
    parser.add_argument("--event-id", default="pour-liquid-into-cup")
    parser.add_argument("--block-indices", type=int, nargs="+", default=list(BLOCKS))
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _initialize_torchrun_group() -> bool:
    if dist.is_initialized():
        return False
    try:
        world = int(os.environ["WORLD_SIZE"])
        int(os.environ["RANK"])
        int(os.environ["LOCAL_RANK"])
    except (KeyError, TypeError, ValueError) as error:
        raise SourceRoleV15BR6ProbeError("torchrun rank environment is missing") from error
    if world != SP_SIZE:
        raise SourceRoleV15BR6ProbeError("r6 WORLD_SIZE must be four")
    dist.init_process_group(backend="nccl", init_method="env://")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    initialized_here = False
    try:
        initialized_here = _initialize_torchrun_group()
        run_probe(build_parser().parse_args(argv))
        dist.barrier()
        return 0
    except (
        SourceRoleV15BR6ProbeError,
        base.SourceRoleSP4ProbeError,
        locator.SourceOwnedRoleLocatorError,
        asset_v15b.E00V15BAssetError,
        registry_r6.NullRegistryV15BR6Error,
        observer_r6.NullBankObserverV15BR6Error,
        calibration.V15BR5CalibrationError,
    ) as error:
        rank = dist.get_rank() if dist.is_initialized() else "?"
        print(f"FAIL-CLOSED rank={rank}: {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

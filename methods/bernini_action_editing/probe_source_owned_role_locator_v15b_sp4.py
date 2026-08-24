#!/usr/bin/env python3
"""E00 instance-level observer-only SP4 probe with create-only diagnostics.

This is a diagnostic harness, not a trainer, controller, route, or decoder.
It runs the frozen adapter OFF once and ON twice, preserves the v15 bitwise
checks, and exports rank-zero source-Q/source-caption-K affinities.  Diagnostic
masks use mutual exclusion only among the three peer vessel instances;
``agent`` and ``support`` never enter that winner-take-all competition.

The raw maps and masks are candidates for visual audit.  They do not certify
the requested action graph and cannot be consumed by a model call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# ROCm evaluates these paths lazily at the first convolution, but they must be
# selected before importing torch.  The launcher creates four disjoint,
# writable directories and supplies only their common absolute root.  Import
# preflight omits the variable and remains read-only.
_miopen_root = os.environ.get("V15B_MIOPEN_CACHE_ROOT")
if _miopen_root is not None:
    _local_rank = os.environ.get("LOCAL_RANK")
    if _local_rank not in {"0", "1", "2", "3"}:
        raise RuntimeError("V15B MIOpen cache requires LOCAL_RANK 0..3")
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
        raise RuntimeError("rank-local MIOpen cache directories are absent/unsafe")
    os.environ["MIOPEN_USER_DB_PATH"] = str(_user_db)
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(_kernel_cache)

import torch
import torch.distributed as dist


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import probe_source_owned_role_locator_v15_sp4 as base  # noqa: E402
import source_owned_role_locator_v15 as locator  # noqa: E402
import source_owned_role_locator_v15b_e00_asset as asset_v15b  # noqa: E402


PROBE_SCHEMA_VERSION = "bernini-source-owned-instance-role-sp4-probe-v15b"
DIAGNOSTIC_SCHEMA_VERSION = "bernini-source-owned-instance-role-affinity-v15b"
SP_SIZE = 4


class SourceRoleV15BProbeError(RuntimeError):
    """Fail-closed v15b SP4 or diagnostic-export violation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _diagnostic_group_masks(
    affinity: torch.Tensor,
    null_affinity: torch.Tensor,
    shuffled_affinity: torch.Tensor,
    *,
    role_names: tuple[str, ...],
    keep_fraction: float = 0.08,
    minimum_spatial_std: float = 0.01,
    minimum_absolute_affinity: float = 0.05,
    minimum_null_margin: float = 0.03,
    minimum_shuffled_margin: float = 0.03,
    minimum_peer_margin: float = 0.01,
    minimum_confident_phases: int = 3,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Build non-route candidates without cross-layer WTA or forced pixels."""

    if role_names != asset_v15b.ROLE_NAMES:
        raise SourceRoleV15BProbeError("diagnostic role order differs")
    if (
        affinity.dtype != torch.float32
        or shuffled_affinity.dtype != torch.float32
        or null_affinity.dtype != torch.float32
        or tuple(affinity.shape) != (5, 21, 37, 25)
        or tuple(shuffled_affinity.shape) != tuple(affinity.shape)
        or tuple(null_affinity.shape) != (21, 37, 25)
    ):
        raise SourceRoleV15BProbeError("diagnostic affinity geometry differs")
    scores = affinity.detach().cpu().contiguous()
    null = null_affinity.detach().cpu().contiguous()
    shuffled = shuffled_affinity.detach().cpu().contiguous()
    roles, phases, height, width = scores.shape
    spatial = height * width
    flat = scores.reshape(roles, phases, spatial)
    null_flat = null.reshape(phases, spatial)
    shuffled_flat = shuffled.reshape(roles, phases, spatial)
    masks = torch.zeros_like(flat, dtype=torch.bool)
    std = flat.std(dim=2, unbiased=False)
    quota = max(1, min(spatial, int((spatial * keep_fraction) + 0.999999)))
    vessel_indices = torch.tensor(
        [role_names.index(name) for name in asset_v15b.VESSEL_COMPETITION_GROUP],
        dtype=torch.int64,
    )
    for phase in range(phases):
        vessel_scores = flat[vessel_indices, phase]
        vessel_winners = vessel_scores.argmax(dim=0)
        for role_index in range(roles):
            real = flat[role_index, phase]
            allowed = (
                (std[role_index, phase] >= minimum_spatial_std)
                & (real >= minimum_absolute_affinity)
                & (real - null_flat[phase] >= minimum_null_margin)
                & (
                    real - shuffled_flat[role_index, phase]
                    >= minimum_shuffled_margin
                )
            )
            role_name = role_names[role_index]
            if role_name in asset_v15b.VESSEL_COMPETITION_GROUP:
                group_offset = asset_v15b.VESSEL_COMPETITION_GROUP.index(role_name)
                peers = torch.cat(
                    (
                        vessel_scores[:group_offset],
                        vessel_scores[group_offset + 1 :],
                    ),
                    dim=0,
                ).max(dim=0).values
                allowed &= vessel_winners.eq(group_offset)
                allowed &= real - peers >= minimum_peer_margin
            candidates = torch.nonzero(allowed, as_tuple=False).flatten()
            if not int(candidates.numel()):
                continue
            take = min(quota, int(candidates.numel()))
            chosen = torch.topk(real[candidates], k=take, largest=True, sorted=False).indices
            masks[role_index, phase, candidates[chosen]] = True
    masks = masks.reshape(roles, phases, height, width).contiguous()
    counts = masks.reshape(roles, phases, spatial).sum(dim=2)
    confident = counts.ge(1)
    vessel_counts = {
        name: int(confident[role_names.index(name)].sum().item())
        for name in asset_v15b.VESSEL_COMPETITION_GROUP
    }
    candidate_qualified = all(
        count >= minimum_confident_phases for count in vessel_counts.values()
    )
    policy = {
        "keep_fraction": keep_fraction,
        "minimum_spatial_std": minimum_spatial_std,
        "minimum_absolute_affinity": minimum_absolute_affinity,
        "minimum_null_margin": minimum_null_margin,
        "minimum_shuffled_margin": minimum_shuffled_margin,
        "minimum_peer_margin": minimum_peer_margin,
        "minimum_confident_phases": minimum_confident_phases,
        "vessel_competition_group": list(asset_v15b.VESSEL_COMPETITION_GROUP),
        "independent_roles": list(asset_v15b.INDEPENDENT_ROLES),
        "cross_layer_winner_take_all": False,
        "forced_nonempty": False,
    }
    receipt = {
        "policy": policy,
        "phase_role_pixel_counts": counts.tolist(),
        "vessel_confident_phase_counts": vessel_counts,
        "mechanical_candidate_qualified": candidate_qualified,
        "semantic_localization_certified": False,
        "action_success_certified": False,
        "status": "requires_heatmap_overlay_and_instance_ROI_audit",
    }
    return masks, receipt


def _save_diagnostics_create_only(
    path: Path,
    *,
    globals_by_block: Sequence[locator.GlobalRoleAffinity],
    spec: locator.SourceRoleEventSpec,
    source_binding_receipt_sha256: str,
    source_text_provenance_sha256: str,
) -> Mapping[str, Any]:
    if path.exists():
        raise SourceRoleV15BProbeError("diagnostic tensor path already exists")
    if len(globals_by_block) != 5:
        raise SourceRoleV15BProbeError("diagnostic block cardinality differs")
    real = torch.stack([item.affinity.cpu() for item in globals_by_block], dim=0)
    null = torch.stack([item.null_affinity.cpu() for item in globals_by_block], dim=0)
    shuffled = torch.stack(
        [item.shuffled_affinity.cpu() for item in globals_by_block], dim=0
    )
    aggregate_real = real.mean(dim=0).contiguous()
    aggregate_null = null.mean(dim=0).contiguous()
    aggregate_shuffled = shuffled.mean(dim=0).contiguous()
    masks, mask_receipt = _diagnostic_group_masks(
        aggregate_real,
        aggregate_null,
        aggregate_shuffled,
        role_names=spec.role_names,
    )
    tensors: dict[str, torch.Tensor] = {
        "aggregate_affinity": aggregate_real,
        "aggregate_null_affinity": aggregate_null,
        "aggregate_shuffled_affinity": aggregate_shuffled,
        "aggregate_group_masks_u8": masks.to(torch.uint8).contiguous(),
    }
    for index, item in enumerate(globals_by_block):
        if item.block_index not in (4, 9, 14, 19, 24):
            raise SourceRoleV15BProbeError("unregistered diagnostic block")
        tensors[f"block_{item.block_index:02d}_affinity"] = real[index].contiguous()
        tensors[f"block_{item.block_index:02d}_null_affinity"] = null[index].contiguous()
        tensors[f"block_{item.block_index:02d}_shuffled_affinity"] = shuffled[
            index
        ].contiguous()
    metadata_payload = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "event_id": spec.event_id,
        "role_asset_sha256": asset_v15b.ASSET_SHA256,
        "role_event_sha256": spec.event_sha256,
        "role_names": list(spec.role_names),
        "source_binding_receipt_sha256": source_binding_receipt_sha256,
        "source_text_provenance_sha256": source_text_provenance_sha256,
        "geometry": [21, 37, 25],
        "selected_block_indices": [item.block_index for item in globals_by_block],
        "tensor_sha256": {
            name: locator.tensor_sha256(value) for name, value in tensors.items()
        },
        "mask_diagnostic": mask_receipt,
        "observer_only": True,
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
    if args.event_id != spec.event_id:
        raise SourceRoleV15BProbeError("event-id differs from v15b E00 lock")
    blocks = tuple(args.block_indices)
    if blocks != (4, 9, 14, 19, 24):
        raise SourceRoleV15BProbeError("v15b selected blocks differ")
    factory = base._load_factory(args.runtime_adapter)
    adapter_config = json.loads(args.adapter_config)
    if not isinstance(adapter_config, Mapping):
        raise SourceRoleV15BProbeError("adapter-config is not one object")
    adapter = factory(dict(adapter_config))
    adapter_contract = base.validate_adapter_contract(adapter, blocks)
    binding_method = getattr(adapter, "binding_receipt", None)
    if not callable(binding_method):
        raise SourceRoleV15BProbeError("adapter lacks source binding receipt")
    binding = binding_method()
    if (
        not isinstance(binding, Mapping)
        or binding.get("role_asset_sha256") != asset_v15b.ASSET_SHA256
        or binding.get("role_event_sha256") != asset_v15b.EVENT_SHA256
        or binding.get("source_video_sha256") != asset_v15b.SOURCE_VIDEO_SHA256
        or binding.get("action_success_authorized") is not False
        or binding.get("route_authorized") is not False
        or binding.get("training_authorized") is not False
        or binding.get("decode_authorized") is not False
    ):
        raise SourceRoleV15BProbeError("adapter/source/role binding receipt differs")

    geometry = getattr(adapter, "source_geometry", None)
    if not isinstance(geometry, locator.SourceVisualGeometry):
        raise SourceRoleV15BProbeError("adapter lacks source geometry")
    layout = locator.UlyssesVisualShard(geometry=geometry, rank=rank, size=world)
    bank = locator.SourceRoleCaptureBank(blocks)
    patch_handle = locator.install_source_owned_role_observer(
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
            raise SourceRoleV15BProbeError("observer OFF captured affinity")
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
        raise SourceRoleV15BProbeError("observer ON changed frozen output")
    if not base._rng_equal(off.post_rng, on.post_rng):
        raise SourceRoleV15BProbeError("observer ON changed RNG")
    if (
        not torch.equal(on.output, repeat.output)
        or on.output_sha256 != repeat.output_sha256
        or not base._rng_equal(on.post_rng, repeat.post_rng)
    ):
        raise SourceRoleV15BProbeError("observer repeat is not bit deterministic")
    if not (
        off.state_receipt == initial_state
        and on.state_receipt == initial_state
        and repeat.state_receipt == initial_state
    ):
        raise SourceRoleV15BProbeError("observer changed frozen model state")
    if not (
        off.provenance_sha256 == on.provenance_sha256 == repeat.provenance_sha256
        and off.source_receipt_sha256
        == on.source_receipt_sha256
        == repeat.source_receipt_sha256
        == binding["binding_receipt_sha256"]
    ):
        raise SourceRoleV15BProbeError("source/token/binding provenance differs")
    if bank.capture_count != 2 * len(blocks):
        raise SourceRoleV15BProbeError("selected-block capture cardinality differs")

    globals_by_block: list[locator.GlobalRoleAffinity] = []
    repeats_by_block: list[locator.GlobalRoleAffinity] = []
    for block_index in blocks:
        local = bank.shards_for(
            event_id=spec.event_id, step_index=0, block_index=block_index
        )
        repeat_local = bank.shards_for(
            event_id=spec.event_id, step_index=1, block_index=block_index
        )
        if len(local) != 1 or len(repeat_local) != 1:
            raise SourceRoleV15BProbeError("rank-local capture cardinality differs")
        if not (
            torch.equal(local[0].affinity, repeat_local[0].affinity)
            and torch.equal(local[0].null_affinity, repeat_local[0].null_affinity)
            and torch.equal(
                local[0].shuffled_affinity, repeat_local[0].shuffled_affinity
            )
        ):
            raise SourceRoleV15BProbeError("rank-local affinity repeat differs")
        first_global = base._explicit_gather_block(local[0], rank=rank)
        repeat_global = base._explicit_gather_block(repeat_local[0], rank=rank)
        if first_global is not None:
            globals_by_block.append(first_global)
            repeats_by_block.append(repeat_global)
    dist.barrier()
    if rank != 0:
        return None
    for first, second in zip(globals_by_block, repeats_by_block):
        if second is None or not (
            torch.equal(first.affinity, second.affinity)
            and torch.equal(first.null_affinity, second.null_affinity)
            and torch.equal(first.shuffled_affinity, second.shuffled_affinity)
        ):
            raise SourceRoleV15BProbeError("global affinity repeat differs")

    diagnostics = _save_diagnostics_create_only(
        Path(args.diagnostics_output),
        globals_by_block=globals_by_block,
        spec=spec,
        source_binding_receipt_sha256=on.source_receipt_sha256,
        source_text_provenance_sha256=on.provenance_sha256,
    )
    receipt_payload = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "observer_only_sp4_mechanical_pass_localization_pending_overlay",
        "event_id": spec.event_id,
        "role_asset_sha256": asset_v15b.ASSET_SHA256,
        "role_event_sha256": asset_v15b.EVENT_SHA256,
        "role_names": list(spec.role_names),
        "semantic_contract": asset_raw["semantic_contract"],
        "competition_groups": asset_raw["competition_groups"],
        "independent_roles": asset_raw["independent_roles"],
        "adapter_contract": adapter_contract,
        "source_binding": dict(binding),
        "world_size": world,
        "selected_block_indices": list(blocks),
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
            "repeat_output_rng_affinity_bit_deterministic": True,
            "explicit_collective_outside_attn2": True,
            "exact_source_checkpoint_latent_noise_timestep_token_embedding_binding": True,
        },
        "localization_semantically_certified": False,
        "action_success_certified": False,
        "route_authorized": False,
        "training_authorized": False,
        "decode_authorized": False,
    }
    receipt = {
        **receipt_payload,
        "receipt_sha256": locator.object_sha256(receipt_payload),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    except FileExistsError as error:
        raise SourceRoleV15BProbeError("probe receipt path already exists") from error
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-adapter", required=True)
    parser.add_argument("--adapter-config", default="{}")
    parser.add_argument("--role-asset", type=Path, default=asset_v15b.DEFAULT_ASSET)
    parser.add_argument("--event-id", default="pour-liquid-into-cup")
    parser.add_argument(
        "--block-indices", type=int, nargs="+", default=[4, 9, 14, 19, 24]
    )
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
        raise SourceRoleV15BProbeError("torchrun rank environment is missing") from error
    if world != SP_SIZE:
        raise SourceRoleV15BProbeError("torchrun WORLD_SIZE must be four")
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
        SourceRoleV15BProbeError,
        base.SourceRoleSP4ProbeError,
        locator.SourceOwnedRoleLocatorError,
        asset_v15b.E00V15BAssetError,
    ) as error:
        rank = dist.get_rank() if dist.is_initialized() else "?"
        print(f"FAIL-CLOSED rank={rank}: {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

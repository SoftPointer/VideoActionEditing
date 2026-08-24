#!/usr/bin/env python3
"""Observer-only SP4 GPU harness for ``source_owned_role_locator_v15``.

This file is intentionally not a Bernini trainer, decoder, or action route.  A
site-specific runtime adapter supplies the frozen official renderer seam while
this harness owns the audit order:

1. authenticate source tokenization and build pre-SP text provenance;
2. enter the observer context *before* ``prepare_inputs_for_sp``;
3. run the same frozen forward once OFF and twice ON from one RNG state;
4. prove output/RNG/model state are unchanged; and
5. explicitly gather rank-local diagnostic affinities outside attn2.

It must be launched with exactly four torchrun processes.  Merely importing
this module never initializes distributed state or touches a GPU.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_owned_role_locator_v15 as locator  # noqa: E402


PROBE_SCHEMA_VERSION = "bernini-source-owned-role-locator-sp4-probe-v15"
ADAPTER_SCHEMA_VERSION = "bernini-source-owned-role-locator-sp4-adapter-v15"
SP_SIZE = 4


class SourceRoleSP4ProbeError(RuntimeError):
    """Fail-closed probe contract violation."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SourceRoleSP4ProbeError(f"{label} must be a lowercase SHA-256")
    return value


def _load_factory(reference: str) -> Callable[[Mapping[str, Any]], Any]:
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise SourceRoleSP4ProbeError("runtime adapter must be module:factory")
    module_name, symbol = reference.split(":", 1)
    if not module_name or not symbol or symbol.startswith("_"):
        raise SourceRoleSP4ProbeError("runtime adapter reference is invalid")
    try:
        factory = getattr(importlib.import_module(module_name), symbol)
    except (ImportError, AttributeError) as error:
        raise SourceRoleSP4ProbeError("cannot import runtime adapter factory") from error
    if not callable(factory):
        raise SourceRoleSP4ProbeError("runtime adapter factory is not callable")
    return factory


def validate_adapter_contract(adapter: Any, selected_blocks: Sequence[int]) -> Mapping[str, Any]:
    contract_method = getattr(adapter, "observer_contract", None)
    if not callable(contract_method):
        raise SourceRoleSP4ProbeError("adapter lacks observer_contract()")
    contract = contract_method()
    required = {
        "schema_version",
        "checkpoint_sha256",
        "source_manifest_sha256",
        "source_is_real_video",
        "frozen_base",
        "eval_mode",
        "adapters_disabled",
        "ulysses_group_is_world",
        "world_size",
        "selected_block_indices",
        "observer_only",
        "training_authorized",
        "route_authorized",
    }
    if not isinstance(contract, Mapping) or set(contract) != required:
        raise SourceRoleSP4ProbeError("adapter observer contract fields differ")
    if contract["schema_version"] != ADAPTER_SCHEMA_VERSION:
        raise SourceRoleSP4ProbeError("adapter observer schema differs")
    for label in ("checkpoint_sha256", "source_manifest_sha256"):
        _exact_sha(contract[label], label)
    exact_true = (
        "source_is_real_video",
        "frozen_base",
        "eval_mode",
        "adapters_disabled",
        "ulysses_group_is_world",
        "observer_only",
    )
    if any(contract[label] is not True for label in exact_true):
        raise SourceRoleSP4ProbeError("adapter is not a frozen source-only observer")
    if contract["training_authorized"] is not False or contract["route_authorized"] is not False:
        raise SourceRoleSP4ProbeError("adapter authorizes training or routing")
    if contract["world_size"] != SP_SIZE:
        raise SourceRoleSP4ProbeError("adapter is not registered for SP4")
    if tuple(contract["selected_block_indices"]) != tuple(selected_blocks):
        raise SourceRoleSP4ProbeError("adapter selected blocks differ")
    for method_name in (
        "materialize_source",
        "prepare_inputs_for_sp",
        "run_frozen_forward",
    ):
        if not callable(getattr(adapter, method_name, None)):
            raise SourceRoleSP4ProbeError(f"adapter lacks {method_name}()")
    model = getattr(adapter, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise SourceRoleSP4ProbeError("adapter model must be a torch module")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise SourceRoleSP4ProbeError("adapter model is not frozen/eval")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise SourceRoleSP4ProbeError("adapter model has pre-existing gradients")
    return dict(contract)


@dataclass(frozen=True)
class SourceMaterialization:
    tokenizer: Any
    tokenizer_dir: Path
    raw_source_text_hidden_states: torch.Tensor
    derive_conditioned_source_text: Callable[[torch.Tensor], torch.Tensor]
    renderer_text_length: int
    geometry: locator.SourceVisualGeometry
    source_receipt_sha256: str

    @classmethod
    def from_adapter(cls, value: Any) -> "SourceMaterialization":
        required = {
            "tokenizer",
            "tokenizer_dir",
            "raw_source_text_hidden_states",
            "derive_conditioned_source_text",
            "renderer_text_length",
            "geometry",
            "source_receipt_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise SourceRoleSP4ProbeError("materialized source fields differ")
        if not callable(value["derive_conditioned_source_text"]):
            raise SourceRoleSP4ProbeError("conditioned-text derivation is not callable")
        if not isinstance(value["geometry"], locator.SourceVisualGeometry):
            raise SourceRoleSP4ProbeError("materialized source geometry differs")
        _exact_sha(value["source_receipt_sha256"], "source_receipt_sha256")
        return cls(
            tokenizer=value["tokenizer"],
            tokenizer_dir=Path(value["tokenizer_dir"]),
            raw_source_text_hidden_states=value["raw_source_text_hidden_states"],
            derive_conditioned_source_text=value["derive_conditioned_source_text"],
            renderer_text_length=value["renderer_text_length"],
            geometry=value["geometry"],
            source_receipt_sha256=value["source_receipt_sha256"],
        )


def _rng_snapshot(device: torch.device) -> dict[str, torch.Tensor]:
    result = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device).clone()
    return result


def _restore_rng(snapshot: Mapping[str, torch.Tensor], device: torch.device) -> None:
    torch.set_rng_state(snapshot["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(snapshot["cuda"], device)


def _rng_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def _module_state_receipt(model: torch.nn.Module) -> dict[str, Any]:
    rows = []
    for category, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in values:
            rows.append(
                {
                    "category": category,
                    "name": name,
                    "requires_grad": bool(tensor.requires_grad),
                    "grad_is_none": tensor.grad is None,
                    "sha256": locator.tensor_sha256(tensor),
                }
            )
    payload = {"rows": rows, "training": bool(model.training)}
    return {**payload, "sha256": locator.object_sha256(payload)}


@dataclass(frozen=True)
class ArmResult:
    output: torch.Tensor
    output_sha256: str
    provenance_sha256: str
    source_receipt_sha256: str
    post_rng: Mapping[str, torch.Tensor]
    state_receipt: Mapping[str, Any]


def _run_arm(
    *,
    adapter: Any,
    spec: locator.SourceRoleEventSpec,
    capture_bank: locator.SourceRoleCaptureBank,
    layout: locator.UlyssesVisualShard,
    initial_rng: Mapping[str, torch.Tensor],
    observer_enabled: bool,
    step_index: int,
    device: torch.device,
) -> ArmResult:
    _restore_rng(initial_rng, device)
    source = SourceMaterialization.from_adapter(
        adapter.materialize_source(event_spec=spec, rank=layout.rank, world_size=layout.size)
    )
    if source.geometry != layout.geometry:
        raise SourceRoleSP4ProbeError("adapter source geometry changed")
    provenance, conditioned = locator.bind_source_text_provenance(
        tokenizer=source.tokenizer,
        tokenizer_dir=source.tokenizer_dir,
        transformers_version=locator.PINNED_TRANSFORMERS_VERSION,
        event_spec=spec,
        raw_source_text_hidden_states=source.raw_source_text_hidden_states,
        derive_conditioned_source_text=source.derive_conditioned_source_text,
        renderer_text_length=source.renderer_text_length,
    )
    invocation = locator.SourceRoleObserverInvocation(
        capture_bank=capture_bank,
        event_spec=spec,
        geometry=source.geometry,
        source_text_provenance=provenance,
        step_index=step_index,
        ulysses=layout,
    )
    context = locator.observe_source_roles(invocation) if observer_enabled else nullcontext()
    # Deliberately enter before prepare_inputs_for_sp.  Official Bernini shards
    # visual Q but leaves full text K replicated.  The returned text tensor
    # must therefore be the authenticated root allocation (or a sharing
    # logical prefix); equal-valued clones fail on every selected block.
    with torch.inference_mode(), context:
        prepared = adapter.prepare_inputs_for_sp(
            conditioned_source_text_hidden_states=conditioned,
            source=source,
            event_spec=spec,
            rank=layout.rank,
            world_size=layout.size,
        )
        output = adapter.run_frozen_forward(prepared=prepared)
    if not isinstance(output, torch.Tensor) or output.device != device:
        raise SourceRoleSP4ProbeError("adapter forward output contract differs")
    if output.requires_grad or output.grad_fn is not None or not bool(torch.isfinite(output).all().item()):
        raise SourceRoleSP4ProbeError("adapter output is not detached finite")
    return ArmResult(
        output=output.detach().clone(memory_format=torch.contiguous_format),
        output_sha256=locator.tensor_sha256(output),
        provenance_sha256=provenance.receipt_sha256,
        source_receipt_sha256=source.source_receipt_sha256,
        post_rng=_rng_snapshot(device),
        state_receipt=_module_state_receipt(adapter.model),
    )


def _explicit_gather_block(
    local: locator.RoleAffinityShard,
    *,
    rank: int,
) -> locator.GlobalRoleAffinity | None:
    tensor = local.padded_collective_tensor()
    gathered_tensors = [torch.empty_like(tensor) for _ in range(SP_SIZE)]
    dist.all_gather(gathered_tensors, tensor)
    gathered_metadata: list[Any] = [None for _ in range(SP_SIZE)]
    dist.all_gather_object(gathered_metadata, local.collective_metadata())
    if rank != 0:
        return None
    shards = tuple(
        locator.RoleAffinityShard.from_collective(value, metadata)
        for value, metadata in zip(gathered_tensors, gathered_metadata)
    )
    return locator.assemble_global_role_affinity(shards)


def _distributed_preflight() -> tuple[int, int, torch.device]:
    if not dist.is_available() or not dist.is_initialized():
        raise SourceRoleSP4ProbeError("launch with torchrun; process group is not initialized")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != SP_SIZE:
        raise SourceRoleSP4ProbeError("observer probe requires exactly four ranks")
    if not torch.cuda.is_available():
        raise SourceRoleSP4ProbeError("observer probe requires four GPU ranks")
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, TypeError, ValueError) as error:
        raise SourceRoleSP4ProbeError("LOCAL_RANK is missing or invalid") from error
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    backend = str(dist.get_backend()).lower()
    if "nccl" not in backend:
        raise SourceRoleSP4ProbeError("SP4 observer probe requires NCCL/RCCL backend")
    return rank, world, device


def run_probe(args: argparse.Namespace) -> Mapping[str, Any] | None:
    rank, world, device = _distributed_preflight()
    asset_specs = locator.load_role_span_asset(args.role_asset)
    matches = [item for item in asset_specs if item.event_id == args.event_id]
    if len(matches) != 1:
        raise SourceRoleSP4ProbeError("event-id is absent or ambiguous")
    spec = matches[0]
    blocks = tuple(args.block_indices)
    factory = _load_factory(args.runtime_adapter)
    adapter_config = json.loads(args.adapter_config)
    if not isinstance(adapter_config, Mapping):
        raise SourceRoleSP4ProbeError("adapter-config must decode to one JSON object")
    adapter = factory(dict(adapter_config))
    adapter_contract = validate_adapter_contract(adapter, blocks)

    geometry = getattr(adapter, "source_geometry", None)
    if not isinstance(geometry, locator.SourceVisualGeometry):
        raise SourceRoleSP4ProbeError("adapter lacks exact source_geometry")
    layout = locator.UlyssesVisualShard(geometry=geometry, rank=rank, size=world)
    bank = locator.SourceRoleCaptureBank(blocks)
    patch_handle = locator.install_source_owned_role_observer(adapter.model, capture_bank=bank)
    initial_state = _module_state_receipt(adapter.model)
    initial_rng = _rng_snapshot(device)
    try:
        off = _run_arm(
            adapter=adapter, spec=spec, capture_bank=bank, layout=layout,
            initial_rng=initial_rng, observer_enabled=False, step_index=0, device=device,
        )
        if bank.capture_count != 0:
            raise SourceRoleSP4ProbeError("observer OFF arm captured affinity")
        on = _run_arm(
            adapter=adapter, spec=spec, capture_bank=bank, layout=layout,
            initial_rng=initial_rng, observer_enabled=True, step_index=0, device=device,
        )
        on_repeat = _run_arm(
            adapter=adapter, spec=spec, capture_bank=bank, layout=layout,
            initial_rng=initial_rng, observer_enabled=True, step_index=1, device=device,
        )
    finally:
        patch_handle.restore()

    if not torch.equal(off.output, on.output) or off.output_sha256 != on.output_sha256:
        raise SourceRoleSP4ProbeError("observer ON changed frozen output")
    if not _rng_equal(off.post_rng, on.post_rng):
        raise SourceRoleSP4ProbeError("observer ON changed post-forward RNG state")
    if (
        not torch.equal(on.output, on_repeat.output)
        or on.output_sha256 != on_repeat.output_sha256
        or not _rng_equal(on.post_rng, on_repeat.post_rng)
    ):
        raise SourceRoleSP4ProbeError("repeated observer ON arm is not bit deterministic")
    if (
        off.state_receipt != initial_state
        or on.state_receipt != initial_state
        or on_repeat.state_receipt != initial_state
    ):
        raise SourceRoleSP4ProbeError("observer probe changed parameter/buffer/grad state")
    if not (
        off.provenance_sha256
        == on.provenance_sha256
        == on_repeat.provenance_sha256
    ):
        raise SourceRoleSP4ProbeError("OFF/ON source-text provenance differs")
    if not (
        off.source_receipt_sha256
        == on.source_receipt_sha256
        == on_repeat.source_receipt_sha256
    ):
        raise SourceRoleSP4ProbeError("OFF/ON real-source receipt differs")
    if bank.capture_count != 2 * len(blocks):
        raise SourceRoleSP4ProbeError("ON repeats did not capture every selected block exactly once")

    globals_by_block = []
    repeat_globals_by_block = []
    for block_index in blocks:
        local_rows = bank.shards_for(
            event_id=spec.event_id, step_index=0, block_index=block_index
        )
        repeat_rows = bank.shards_for(
            event_id=spec.event_id, step_index=1, block_index=block_index
        )
        if len(local_rows) != 1 or local_rows[0].layout.rank != rank:
            raise SourceRoleSP4ProbeError("rank-local capture cardinality differs")
        if len(repeat_rows) != 1 or repeat_rows[0].layout.rank != rank:
            raise SourceRoleSP4ProbeError("repeated rank-local capture cardinality differs")
        if not (
            torch.equal(local_rows[0].affinity, repeat_rows[0].affinity)
            and torch.equal(local_rows[0].null_affinity, repeat_rows[0].null_affinity)
            and torch.equal(local_rows[0].shuffled_affinity, repeat_rows[0].shuffled_affinity)
        ):
            raise SourceRoleSP4ProbeError("repeated rank-local affinity is not bit deterministic")
        global_value = _explicit_gather_block(local_rows[0], rank=rank)
        repeat_global = _explicit_gather_block(repeat_rows[0], rank=rank)
        if global_value is not None:
            globals_by_block.append(global_value)
            repeat_globals_by_block.append(repeat_global)
    dist.barrier()
    if rank != 0:
        return None
    for first, second in zip(globals_by_block, repeat_globals_by_block):
        if second is None or not (
            torch.equal(first.affinity, second.affinity)
            and torch.equal(first.null_affinity, second.null_affinity)
            and torch.equal(first.shuffled_affinity, second.shuffled_affinity)
        ):
            raise SourceRoleSP4ProbeError("repeated global affinity is not bit deterministic")

    receipt_payload = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "observer_only_sp4_probe_pass",
        "event_id": spec.event_id,
        "adapter_contract": adapter_contract,
        "world_size": world,
        "selected_block_indices": list(blocks),
        "geometry": [geometry.phases, geometry.height, geometry.width],
        "ulysses_intervals": [
            [
                locator.UlyssesVisualShard(geometry, candidate_rank, world).global_start,
                locator.UlyssesVisualShard(geometry, candidate_rank, world).global_stop,
            ]
            for candidate_rank in range(world)
        ],
        "source_receipt_sha256": on.source_receipt_sha256,
        "source_text_provenance_sha256": on.provenance_sha256,
        "output_sha256": on.output_sha256,
        "global_affinity_sha256_by_block": {
            str(item.block_index): locator.object_sha256(
                {
                    "real": locator.tensor_sha256(item.affinity),
                    "null": locator.tensor_sha256(item.null_affinity),
                    "shuffled": locator.tensor_sha256(item.shuffled_affinity),
                }
            )
            for item in globals_by_block
        },
        "mechanical_gates": {
            "off_on_output_torch_equal": True,
            "off_on_output_byte_sha_equal": True,
            "off_on_rng_equal": True,
            "parameter_buffer_grad_state_unchanged": True,
            "one_capture_per_selected_block_rank": True,
            "repeated_on_output_rng_and_affinity_bit_deterministic": True,
            "explicit_collective_outside_attn2": True,
        },
        "training_authorized": False,
        "route_authorized": False,
        "decode_authorized": False,
    }
    receipt = {
        **receipt_payload,
        "receipt_sha256": locator.object_sha256(receipt_payload),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    except FileExistsError as error:
        raise SourceRoleSP4ProbeError("probe receipt path already exists") from error
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-adapter", required=True, help="module:factory")
    parser.add_argument("--adapter-config", default="{}", help="strict JSON object")
    parser.add_argument(
        "--role-asset",
        type=Path,
        default=METHOD_ROOT / "assets" / "interaction_complex4_source_role_token_spans_v15.json",
    )
    parser.add_argument("--event-id", default="pour-liquid-into-cup")
    parser.add_argument("--block-indices", type=int, nargs="+", default=[4, 9, 14, 19, 24])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_probe(args)
    except (SourceRoleSP4ProbeError, locator.SourceOwnedRoleLocatorError) as error:
        print(f"FAIL-CLOSED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

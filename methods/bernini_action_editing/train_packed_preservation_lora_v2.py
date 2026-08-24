#!/usr/bin/env python3
"""Train the strong packed-token Bernini preservation LoRA v2.

The runner intentionally has only two execution scopes: a real two-update
optimizer canary and one fresh, continuous 80-update trajectory.  Both use the
official Bernini packed ``[source; noisy target]`` path under WORLD8 DP2xSP4.
The 64 train records are physical source-posterior index-0 files; legacy
Parquet and synthetic target index 1 are unreachable from this process.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

METHOD = "bernini-packed-preservation-lora-v2"
RECEIPT_SCHEMA = "bernini-packed-preservation-training-v2"
EXECUTION_SCOPES = ("optimizer-canary-2", "exact80")
LORA_SCOPES = ("all-attention", "self-attention")
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
GRADIENT_ACCUMULATION = 4
GLOBAL_BATCH = 8
FRAME_COUNT = 81
LATENT_PHASES = 21
PATCH_VALUES = 64
TOPOLOGY = "world8-dp2-sp4"
DEFAULT_SEED = 20260814
DEFAULT_LR = 1.0e-4
DEFAULT_MAX_GRAD_NORM = 1.0
SOURCE_MANIFEST_SHA256 = (
    "128064fd335c4e48c567217c6e7bae43555a904875625c9d1e21178e6f7fcc3d"
)
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)


class PackedPreservationTrainingError(RuntimeError):
    """Raised before publishing an invalid update or checkpoint."""


def fail(message: str) -> NoReturn:
    raise PackedPreservationTrainingError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(named: Sequence[tuple[str, Any]]) -> str:
    """Full parameter digest used only at P0/P1/P2 and saved checkpoints."""

    import torch

    digest = hashlib.sha256()
    for name, parameter in named:
        tensor = parameter.detach().contiguous()
        metadata = core.canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite JSON artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(core.canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def deterministic_seed(base: int, *coordinates: Any) -> int:
    payload = "\0".join(str(value) for value in (base, METHOD, *coordinates)).encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


@contextmanager
def serialized_model_load() -> Iterator[None]:
    """Serialize eight checkpoint loads through one shared node-local flock."""

    job = os.environ.get("SLURM_JOB_ID", "no-slurm")
    step = os.environ.get("SLURM_STEP_ID", "no-step")
    path = Path(f"/tmp/bernini-preservation-v2-{job}-{step}.model-load.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _pack_latent_patches(latent: Any) -> Any:
    import torch

    if (
        not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float32
        or latent.device.type != "cpu"
        or latent.ndim != 4
        or tuple(int(value) for value in latent.shape[:2]) != (16, LATENT_PHASES)
        or int(latent.shape[2]) % 2
        or int(latent.shape[3]) % 2
        or not latent.is_contiguous()
    ):
        fail("latent patch input must be CPU FP32 [16,21,evenH,evenW]")
    channels, phases, height, width = (int(value) for value in latent.shape)
    return (
        latent.reshape(channels, phases, height // 2, 2, width // 2, 2)
        .permute(1, 2, 4, 0, 3, 5)
        .reshape(phases * (height // 2) * (width // 2), channels, 1, 2, 2)
        .contiguous()
    )


def _packed_output_field(patches: Any) -> Any:
    return patches.permute(0, 2, 3, 4, 1).reshape(1, int(patches.shape[0]), 64)


def prepare_restoration_pair(
    *, clean: Any, corrupted_source: Any, epsilon: Any, coordinate: Any, rope: Any, device: Any
) -> Mapping[str, Any]:
    import torch

    for value, label in ((clean, "clean"), (corrupted_source, "corrupted"), (epsilon, "epsilon")):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.shape != clean.shape
            or value.ndim != 5
            or tuple(int(item) for item in value.shape[:3]) != (1, 16, 21)
        ):
            fail(f"{label} restoration latent differs")
    target_clean = clean.squeeze(0).contiguous()
    source_clean = corrupted_source.squeeze(0).contiguous()
    eps = epsilon.squeeze(0).contiguous()
    sigma = float(coordinate.sigma)
    noisy_target = ((1.0 - sigma) * target_clean + sigma * eps).contiguous()
    target_velocity = (eps - target_clean).contiguous()
    source_patches = _pack_latent_patches(source_clean)
    target_patches = _pack_latent_patches(noisy_target)
    velocity_patches = _pack_latent_patches(target_velocity)
    source_tokens = int(source_patches.shape[0])
    target_tokens = int(target_patches.shape[0])
    if source_tokens != target_tokens or source_tokens <= 0:
        fail("source/target packed token geometry differs")
    input_patches = torch.cat((source_patches, target_patches), dim=0).to(device)
    source_rope = rope(corrupted_source.to(device), source_id=1)
    target_rope = rope(noisy_target.unsqueeze(0).to(device), source_id=0)
    rotary = torch.cat((source_rope, target_rope), dim=2)
    rotary = rotary.squeeze(0).permute(1, 0, 2).contiguous()
    return {
        "input_patches": input_patches,
        "rotary": rotary,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "total_tokens": source_tokens + target_tokens,
        "target_velocity": _packed_output_field(velocity_patches).to(device),
    }


def predict_target(
    *, renderer: Any, transformer: Any, packed: Mapping[str, Any], coordinate: Any,
    text_lens: Any, text_embs: Any
) -> Any:
    import torch

    with core.packed_role_layout(packed["source_tokens"], packed["target_tokens"]):
        embedded = transformer.patch_embedding(packed["input_patches"]).flatten(1).unsqueeze(0)
    rotary = packed["rotary"].permute(1, 0, 2).unsqueeze(0)
    value = renderer.diff_dec.shared_step(
        model_id="transformer_1",
        noisy_latents=embedded,
        timesteps=embedded.new_tensor([coordinate.timestep], dtype=torch.int64),
        cond_embeds=text_embs,
        rotary_embs=rotary,
        batch_vae_seqlen=[packed["total_tokens"]],
        batch_text_seqlen=text_lens,
    )
    target = value[:, packed["source_tokens"] :, :]
    if tuple(target.shape) != (1, packed["target_tokens"], PATCH_VALUES):
        fail("official shared_step target prediction geometry differs")
    return target


def gradient_groups(named: Sequence[tuple[str, Any]]) -> Mapping[str, float]:
    import torch

    names = ("lora_A", "lora_B", "source_patch", "target_patch", "role")
    squared = {name: torch.zeros((), dtype=torch.float64, device="cuda") for name in names}
    counts = {name: 0 for name in names}
    for name, parameter in named:
        if ".lora_A." in name:
            group = "lora_A"
        elif ".lora_B." in name:
            group = "lora_B"
        elif ".source_delta." in name:
            group = "source_patch"
        elif ".target_delta." in name:
            group = "target_patch"
        elif ".role_embedding" in name:
            group = "role"
        else:
            fail(f"unclassified trainable gradient: {name}")
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
            fail(f"missing/non-finite trainable gradient: {name}")
        gradient = parameter.grad.detach().to(dtype=torch.float64)
        squared[group].add_(gradient.square().sum())
        counts[group] += 1
    if any(count <= 0 for count in counts.values()):
        fail("gradient group inventory is incomplete")
    return {name: float(torch.sqrt(value).item()) for name, value in squared.items()}


def lora_projection_gradient_groups(
    named: Sequence[tuple[str, Any]], specs: Sequence[core.ProjectionSpec]
) -> Mapping[str, float]:
    """Norm every attention/projection/A-or-B group, not merely aggregate LoRA."""

    import torch

    parameters = dict(named)
    squared: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for spec in specs:
        for factor in ("A", "B"):
            matches = [
                (name, parameter)
                for name, parameter in parameters.items()
                if f"{spec.name}.lora_{factor}." in name
            ]
            if len(matches) != 1:
                fail(f"LoRA projection gradient owner differs: {spec.name}/{factor}")
            _, parameter = matches[0]
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
                fail(f"LoRA projection gradient is absent/non-finite: {spec.name}/{factor}")
            key = f"attn{spec.attention}.{spec.projection}.lora_{factor}"
            squared.setdefault(key, torch.zeros((), dtype=torch.float64, device=parameter.device))
            squared[key].add_(parameter.grad.detach().to(torch.float64).square().sum())
            counts[key] = counts.get(key, 0) + 1
    expected_groups = (8 if any(spec.attention == 2 for spec in specs) else 4) * 2
    if len(squared) != expected_groups or any(count != 30 for count in counts.values()):
        fail("LoRA projection gradient group coverage differs")
    return {key: float(torch.sqrt(value).item()) for key, value in sorted(squared.items())}


def lora_affine_gradient_audit(
    named: Sequence[tuple[str, Any]], specs: Sequence[core.ProjectionSpec], step: int
) -> Mapping[str, Any]:
    """Reject a silent individual block/projection, not only a silent sum."""

    import torch

    parameters = dict(named)
    factors: dict[str, list[Any]] = {"A": [], "B": []}
    for spec in specs:
        for factor in factors:
            matches = [
                parameter
                for name, parameter in parameters.items()
                if f"{spec.name}.lora_{factor}." in name
            ]
            if len(matches) != 1 or matches[0].grad is None:
                fail(f"per-affine gradient owner differs: {spec.name}/{factor}")
            factors[factor].append(
                torch.sqrt(matches[0].grad.detach().to(torch.float64).square().sum())
            )
    result: dict[str, Any] = {}
    for factor, values in factors.items():
        stacked = torch.stack(values)
        if not bool(torch.isfinite(stacked).all().item()):
            fail(f"per-affine LoRA-{factor} gradient is non-finite")
        positive = int((stacked > 0).sum().item())
        expected = len(specs)
        required = factor == "B" or step >= 2
        if required and positive != expected:
            fail(
                f"step {step} has silent LoRA-{factor} affine gradients: "
                f"{positive}/{expected}"
            )
        result[f"lora_{factor}"] = {
            "expected_affines": expected,
            "positive_affines": positive,
            "all_positive_required": required,
            "min_norm": float(stacked.min().item()),
            "max_norm": float(stacked.max().item()),
        }
    return result


def gradient_digest(named: Sequence[tuple[str, Any]]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in named:
        if parameter.grad is None:
            fail(f"gradient digest missing parameter: {name}")
        gradient = parameter.grad.detach().contiguous()
        metadata = core.canonical_json_bytes(
            {"name": name, "shape": list(gradient.shape), "dtype": str(gradient.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(gradient.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def synchronize_gradients_bucketed(
    named: Sequence[tuple[str, Any]], parallel: Any, *, bucket_bytes: int = 64 * 1024 * 1024
) -> float:
    """SP4 mean then DP2 mean with bounded flat buckets (not ~1k collectives)."""

    import torch
    import torch.distributed as dist

    if not named or bucket_bytes <= 0:
        fail("bucketed gradient scope differs")
    ready = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for _, parameter in named
    )
    import source_self_runtime as runtime

    if not runtime.world_all_true(ready, group=parallel.world_group):
        fail("at least one WORLD8 gradient is absent/non-finite")
    buckets: list[list[Any]] = []
    current: list[Any] = []
    current_bytes = 0
    for _, parameter in named:
        assert parameter.grad is not None
        item_bytes = parameter.grad.numel() * parameter.grad.element_size()
        if current and current_bytes + item_bytes > bucket_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(parameter)
        current_bytes += item_bytes
    if current:
        buckets.append(current)
    squared = torch.zeros((), dtype=torch.float64, device=named[0][1].device)
    for bucket in buckets:
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=parallel.sp_group)
        flat.div_(float(SP_SIZE))
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=parallel.dp_group)
        flat.div_(float(DP_SIZE))
        offset = 0
        for parameter in bucket:
            assert parameter.grad is not None
            count = parameter.grad.numel()
            parameter.grad.copy_(flat[offset : offset + count].view_as(parameter.grad))
            squared.add_(parameter.grad.detach().to(torch.float64).square().sum())
            offset += count
        if offset != flat.numel():
            fail("gradient bucket scatter closure differs")
    norm = float(torch.sqrt(squared).item())
    if not math.isfinite(norm) or norm <= 0.0:
        fail("bucketed synchronized gradient norm is zero/non-finite")
    return norm


def validate_gradient_gate(
    step: int,
    groups: Mapping[str, float],
    projection_groups: Mapping[str, float],
) -> None:
    if any(not math.isfinite(value) or value < 0 for value in groups.values()):
        fail("gradient group norm is invalid")
    required_positive = {"lora_B", "source_patch", "target_patch", "role"}
    if step >= 2:
        required_positive.add("lora_A")
    if any(groups[name] <= 0.0 for name in required_positive):
        fail(f"step {step} required gradient group is zero: {groups}")
    if any(
        not math.isfinite(value)
        or value < 0.0
        or (key.endswith("lora_B") and value <= 0.0)
        or (step >= 2 and key.endswith("lora_A") and value <= 0.0)
        for key, value in projection_groups.items()
    ):
        fail(f"step {step} per-projection LoRA gradient gate failed")


def per_rank_memory(device: Any, distributed: Any) -> Mapping[str, Any]:
    import torch

    gib = float(1024**3)
    return {
        "world_rank": distributed.rank,
        "dp_arm": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "device": torch.cuda.get_device_name(device),
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "step_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "step_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }


def _cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def save_checkpoint(
    *, root: Path, step: int, model: Any, optimizer: Any, metadata: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Rank-zero create-only inference adapter + separate optimizer state."""

    import torch

    final = root / f"checkpoint-{step:08d}"
    if final.exists() or final.is_symlink():
        fail(f"refusing to overwrite checkpoint {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=root))
    try:
        adapter_path = temporary / "adapter.pt"
        optimizer_path = temporary / "optimizer.pt"
        metadata_path = temporary / "metadata.json"
        torch.save(dict(core.export_trainable_state(model)), adapter_path)
        torch.save(_cpu_tree(optimizer.state_dict()), optimizer_path)
        # This is a real runtime round-trip, not a metadata assertion.  It
        # exercises the same strict loader used by later official RV2V
        # inference, restores the optimizer schema, and proves the live
        # parameter bytes are unchanged by serialization.
        loaded_adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
        core.load_trainable_state_strict(model, loaded_adapter)
        loaded_optimizer = torch.load(
            optimizer_path, map_location="cpu", weights_only=True
        )
        if (
            not isinstance(loaded_optimizer, Mapping)
            or set(loaded_optimizer) != {"state", "param_groups"}
            or len(loaded_optimizer["param_groups"])
            != len(optimizer.state_dict()["param_groups"])
            or len(loaded_optimizer["state"]) != len(optimizer.state_dict()["state"])
        ):
            fail("checkpoint optimizer round-trip schema differs")
        optimizer.load_state_dict(loaded_optimizer)
        roundtrip_parameter_sha = tensor_digest(core.trainable_named_parameters(model))
        if roundtrip_parameter_sha != metadata.get("parameter_sha256"):
            fail("checkpoint adapter round-trip changed parameter bytes")
        inventory = list(core.trainable_inventory(model))
        payload = {
            **dict(metadata),
            "step": step,
            "adapter_file": adapter_path.name,
            "adapter_sha256": file_sha256(adapter_path),
            "optimizer_file": optimizer_path.name,
            "optimizer_sha256": file_sha256(optimizer_path),
            "trainable_inventory": inventory,
            "trainable_inventory_sha256": core.object_sha256(inventory),
            "strict_loader": "packed_preservation_lora_v2.load_trainable_state_strict",
            "adapter_reload_verified": True,
            "optimizer_reload_verified": True,
            "roundtrip_parameter_sha256": roundtrip_parameter_sha,
            "same_architecture_strict_reload_verified": True,
            "fresh_official_rv2v_inference_process_verified": False,
        }
        metadata_path.write_bytes(core.canonical_json_bytes(payload) + b"\n")
        with metadata_path.open("rb") as handle:
            os.fsync(handle.fileno())
        fsync_directory(temporary)
        os.rename(temporary, final)
        fsync_directory(root)
    except Exception:
        # Leave a private, visibly incomplete temp directory for diagnosis.
        raise
    return {
        "step": step,
        "path": str(final),
        "adapter_sha256": payload["adapter_sha256"],
        "optimizer_sha256": payload["optimizer_sha256"],
        "metadata_sha256": file_sha256(final / "metadata.json"),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--checkpoint-content-manifest", required=True)
    value.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    value.add_argument("--source-only-manifest", required=True)
    value.add_argument("--expected-source-only-manifest-sha256", default=SOURCE_MANIFEST_SHA256)
    value.add_argument("--output", required=True)
    value.add_argument("--execution-scope", choices=EXECUTION_SCOPES, required=True)
    value.add_argument("--lora-scope", choices=LORA_SCOPES, required=True)
    value.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    value.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    value.add_argument("--seed", type=int, default=DEFAULT_SEED)
    value.add_argument("--expected-bernini-commit", default=BERNINI_COMMIT)
    value.add_argument("--expected-veomni-commit", default=VEOMNI_COMMIT)
    value.add_argument("--expected-checkpoint-tree-sha256", default=CHECKPOINT_TREE_SHA256)
    value.add_argument("--method-source-revision", required=True)
    value.add_argument("--method-source-archive", required=True)
    value.add_argument("--expected-method-source-archive-sha256", required=True)
    value.add_argument("--method-source-manifest", required=True)
    value.add_argument("--expected-method-source-manifest-sha256", required=True)
    value.add_argument("--ack-source-release-is-exploratory", action="store_true")
    value.add_argument("--ack-fresh-base-not-canary-resume", action="store_true")
    return value


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.expected_source_only_manifest_sha256 != SOURCE_MANIFEST_SHA256
        or args.expected_checkpoint_tree_sha256 != CHECKPOINT_TREE_SHA256
        or args.expected_bernini_commit != BERNINI_COMMIT
        or args.expected_veomni_commit != VEOMNI_COMMIT
        or args.expected_checkpoint_content_manifest_sha256
        != CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        fail("pinned data/model/source identity differs")
    if not args.ack_source_release_is_exploratory or not args.ack_fresh_base_not_canary_resume:
        fail("source-release and fresh-base acknowledgements are mandatory")
    if args.learning_rate != DEFAULT_LR or args.max_grad_norm != DEFAULT_MAX_GRAD_NORM:
        fail("v2 optimizer hyperparameters are fixed")
    if not 0 <= args.seed < 2**63:
        fail("seed differs")
    if (
        len(args.method_source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.method_source_revision)
        or len(args.expected_method_source_archive_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.expected_method_source_archive_sha256
        )
    ):
        fail("method source identity differs")
    if (
        len(args.expected_method_source_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.expected_method_source_manifest_sha256
        )
    ):
        fail("method source manifest identity differs")
    archive = Path(args.method_source_archive).resolve(strict=True)
    if not archive.is_file() or file_sha256(archive) != args.expected_method_source_archive_sha256:
        fail("sealed method source archive differs")
    output = Path(args.output).expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("output must be one fresh absolute path")
    core.optimizer_steps(args.execution_scope)
    core.checkpoint_steps(args.execution_scope)
    if args.execution_scope == "exact80" and "canary" in output.name.lower():
        fail("exact80 output may not masquerade as/resume a canary")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)

    # The launcher materializes METHOD_ROOT from the authenticated archive.  At
    # process entry, import only the small release validator; all model, data,
    # schedule, runtime, and legacy training modules remain unreachable until
    # the archive/manifest/executed-root closure has passed.
    import packed_preservation_release_v2 as release_contract

    method_release = release_contract.validate_executed_release(
        method_root=METHOD_ROOT,
        archive=args.method_source_archive,
        manifest=args.method_source_manifest,
        expected_archive_sha256=args.expected_method_source_archive_sha256,
        expected_manifest_sha256=args.expected_method_source_manifest_sha256,
        method_revision=args.method_source_revision,
    )

    global core, runtime
    import clean_source_visual_context_stage_b_contract_v1 as schedule_contract
    import clean_source_visual_context_training_v1 as source_data
    import packed_preservation_lora_v2 as core
    import source_self_runtime as runtime
    import train_lora as legacy

    validate_args(args)
    manifest_path = Path(args.source_only_manifest).resolve(strict=True)
    if file_sha256(manifest_path) != SOURCE_MANIFEST_SHA256:
        fail("source-only v3 manifest SHA differs")
    manifest = source_data.load_source_only_split_manifest(manifest_path, verify_files=True)
    source_data.authorize_exploratory_training(
        manifest,
        ack_upstream_training_use_forbidden=True,
        ack_user_authorized_exploratory_training=True,
    )
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=BERNINI_COMMIT,
            expected_veomni_commit=VEOMNI_COMMIT,
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise PackedPreservationTrainingError(str(error)) from error
    if transformer_config.get("num_layers") != 30 or transformer_config.get("attention_head_dim") != 128:
        fail("Bernini-R 1.3B transformer geometry differs")
    checkpoint_content_manifest = Path(args.checkpoint_content_manifest).resolve(strict=True)
    if file_sha256(checkpoint_content_manifest) != CHECKPOINT_CONTENT_MANIFEST_SHA256:
        fail("checkpoint content manifest SHA differs")
    checkpoint_content = release_contract.validate_checkpoint_content(
        checkpoint,
        checkpoint_content_manifest,
        expected_manifest_sha256=CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state

    # ``source_self_runtime`` already fail-closes this path to one-node
    # WORLD8 DP2 x SP4.  Keep the runner compatible with that committed API;
    # alternate-topology helpers in unrelated worktree changes are not part of
    # this sealed release.
    distributed = runtime.distributed_contract()
    if distributed.world_size != WORLD_SIZE or distributed.local_world_size != WORLD_SIZE:
        fail("v2 requires one-node WORLD8 DP2xSP4")
    device = runtime.initialise_distributed(distributed)
    parallel = runtime.validate_parallel_state(distributed, init_parallel_state(ulysses_size=SP_SIZE))
    seed_everything(args.seed)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    with serialized_model_load():
        renderer = BerniniRendererModel(config)
        renderer.requires_grad_(False)
        renderer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        specs = core.select_projection_specs(renderer, args.lora_scope)
        model = get_peft_model(
            renderer,
            LoraConfig(
                r=core.LORA_RANK,
                lora_alpha=core.LORA_ALPHA,
                lora_dropout=0.0,
                bias="none",
                target_modules=[item.name for item in specs],
            ),
        )
        transformer = model.get_base_model().diff_dec.transformer
        core.install_typed_patch_embedding(transformer)
        model.to(device)
    model.train()
    base_renderer = model.get_base_model()
    base_renderer.t5_text_encoder.eval()
    named_trainable = core.trainable_named_parameters(model)
    trainable_count = core.verify_trainable_parameter_count(model, args.lora_scope)
    lora_installation = core.validate_lora_installation(model, specs)
    if any(
        parameter.device != device or parameter.dtype != torch.float32
        for _, parameter in named_trainable
    ):
        fail("all LoRA/typed patch trainables must be FP32 on the local GPU")
    architecture = core.architecture_receipt(args.lora_scope, specs)
    inventory = list(core.trainable_inventory(model))
    inventory_sha = core.object_sha256(inventory)
    runtime.digest_consensus(
        inventory_sha,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="v2 trainable inventory",
    )
    runtime.synchronize_initial_parameters(named_trainable, parallel.world_group)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    text_conditions: dict[str, tuple[Any, Any]] = {}
    with torch.inference_mode():
        for objective, instruction in core.PRETEXT_INSTRUCTIONS.items():
            tokenized = runtime.tokenize_generic_instruction(tokenizer, instruction, device)
            text_conditions[objective] = base_renderer.get_t5_text_embeddings(
                tokenized["input_ids"], tokenized["attention_mask"], tokenized["t5_input_lens"]
            )
    base_renderer.t5_text_encoder = None
    del tokenizer, tokenized
    gc.collect()
    torch.cuda.empty_cache()
    if base_renderer.t5_text_encoder is not None:
        fail("frozen T5 was not released after four instruction embeddings")

    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    if z_dim != 16:
        fail("Wan VAE latent width differs")
    store = source_data.PinnedPhysicalSourceOnlyPosteriorStore(
        manifest,
        vae_latents_mean=vae_mean.unsqueeze(0).float().contiguous(),
        vae_latents_std=vae_std.unsqueeze(0).float().contiguous(),
        verify_files_on_first_access=True,
    )
    train_rows = manifest.rows_for_split("train")
    if len(train_rows) != 64:
        fail("optimizer train split differs from 64 real sources")
    index_by_iid = {row.iid: index for index, row in enumerate(manifest.rows)}
    train_indices = [index_by_iid[row.iid] for row in train_rows]
    preload = store.preload(train_indices)
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)

    output = Path(args.output)
    checkpoints = output / "checkpoints"
    if distributed.rank == 0:
        output.mkdir(mode=0o700)
        checkpoints.mkdir(mode=0o700)
    dist.barrier(group=parallel.world_group)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=DEFAULT_LR,
        weight_decay=0.0,
    )
    steps = core.optimizer_steps(args.execution_scope)
    required_checkpoints = core.checkpoint_steps(args.execution_scope)
    common_checkpoint_metadata = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD,
        "execution_scope": args.execution_scope,
        "fresh_official_base": True,
        "resume_consumed": False,
        "lora_scope": args.lora_scope,
        "architecture": architecture,
        "lora_installation": lora_installation,
        "target_module_names": [item.name for item in specs],
        "rank": core.LORA_RANK,
        "source_only_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_only_manifest_digest": manifest.manifest_digest,
        "synthetic_target_accessed": False,
        "reward_used": False,
        "vlm_used": False,
        "official_bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
        "checkpoint_content": checkpoint_content,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.expected_method_source_archive_sha256,
        "method_release": method_release,
    }
    checkpoint_records: list[Mapping[str, Any]] = []
    parameter_digests: dict[int, str] = {}

    initial_digest = tensor_digest(named_trainable)
    runtime.digest_consensus(
        initial_digest,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="v2 P0",
    )
    parameter_digests[0] = initial_digest
    if distributed.rank == 0:
        checkpoint_records.append(
            save_checkpoint(
                root=checkpoints,
                step=0,
                model=model,
                optimizer=optimizer,
                metadata={**common_checkpoint_metadata, "parameter_sha256": initial_digest},
            )
        )
    dist.barrier(group=parallel.world_group)

    history: list[Mapping[str, Any]] = []
    lifetime_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    lifetime_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    started = time.monotonic()
    for step_zero in range(steps):
        completed_step = step_zero + 1
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        micro_records: list[Mapping[str, Any]] = []
        for coordinate in schedule_contract.coordinates_for_optimizer_step(step_zero):
            row_position = schedule_contract.train_row_position(
                optimizer_step_zero_based=step_zero,
                microbatch_index=coordinate.microbatch_index,
                dp_arm=distributed.arm_index,
            )
            row = train_rows[row_position]
            sample = store.load(index_by_iid[row.iid])
            logical_record = (
                step_zero * GLOBAL_BATCH
                + coordinate.microbatch_index * DP_SIZE
                + distributed.arm_index
            )
            objective = core.objective_for_logical_record(logical_record)
            corruption_seed = deterministic_seed(args.seed, "corrupt", logical_record)
            corrupted, corruption = core.restoration_source(
                sample.source_condition, objective, corruption_seed
            )
            noise_seed = deterministic_seed(args.seed, "flow", step_zero, coordinate.microbatch_index, distributed.arm_index)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(noise_seed)
            epsilon = torch.randn(
                tuple(sample.clean_noop_target.shape), generator=generator, dtype=torch.float32
            ).contiguous()
            packed = prepare_restoration_pair(
                clean=sample.clean_noop_target,
                corrupted_source=corrupted,
                epsilon=epsilon,
                coordinate=coordinate,
                rope=rope,
                device=device,
            )
            text_lens, text_embs = text_conditions[objective]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = predict_target(
                    renderer=base_renderer,
                    transformer=transformer,
                    packed=packed,
                    coordinate=coordinate,
                    text_lens=text_lens,
                    text_embs=text_embs,
                )
                raw_loss = torch.nn.functional.mse_loss(
                    prediction.float(), packed["target_velocity"].float()
                )
                scaled_loss = raw_loss / float(GRADIENT_ACCUMULATION)
            if not runtime.world_all_true(
                bool(torch.isfinite(scaled_loss.detach()).item()), group=parallel.world_group
            ):
                fail("non-finite v2 flow-matching loss")
            scaled_loss.backward()
            micro_records.append(
                {
                    "microbatch": coordinate.microbatch_index,
                    "logical_record": logical_record,
                    "row_position": row_position,
                    "iid": row.iid,
                    "objective": objective,
                    "instruction": core.PRETEXT_INSTRUCTIONS[objective],
                    "corruption": corruption,
                    "noise_seed": noise_seed,
                    "schedule_index": coordinate.schedule_index,
                    "sigma": coordinate.sigma,
                    "loss": float(raw_loss.detach().item()),
                    "sample_receipt": dict(sample.receipt()),
                    "source_condition_sha256": runtime.tensor_sha256(
                        sample.source_condition
                    ),
                    "clean_target_sha256": runtime.tensor_sha256(
                        sample.clean_noop_target
                    ),
                    "corrupted_source_sha256": runtime.tensor_sha256(corrupted),
                    "source_condition_is_clean_target_same_object": (
                        sample.source_condition is sample.clean_noop_target
                    ),
                    "source_is_real_index0": True,
                    "target_is_original_same_source": True,
                }
            )
            del corrupted, epsilon, packed, prediction, raw_loss, scaled_loss

        synchronized_norm = synchronize_gradients_bucketed(named_trainable, parallel)
        groups = gradient_groups(named_trainable)
        projection_groups = lora_projection_gradient_groups(named_trainable, specs)
        validate_gradient_gate(completed_step, groups, projection_groups)
        affine_gradient_audit = lora_affine_gradient_audit(
            named_trainable, specs, completed_step
        )
        synchronized_gradient_sha: Optional[str] = None
        if args.execution_scope == "optimizer-canary-2":
            synchronized_gradient_sha = gradient_digest(named_trainable)
            runtime.digest_consensus(
                synchronized_gradient_sha,
                group=parallel.world_group,
                expected_count=WORLD_SIZE,
                label=f"v2 synchronized gradient step {completed_step}",
            )
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named_trainable], DEFAULT_MAX_GRAD_NORM
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        local_memory = per_rank_memory(device, distributed)
        memory_world: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(memory_world, local_memory, group=parallel.world_group)
        if [item["world_rank"] for item in memory_world] != list(range(WORLD_SIZE)):
            fail("per-rank memory receipt closure differs")

        full_digest: Optional[str] = None
        if completed_step in required_checkpoints:
            full_digest = tensor_digest(named_trainable)
            runtime.digest_consensus(
                full_digest,
                group=parallel.world_group,
                expected_count=WORLD_SIZE,
                label=f"v2 P{completed_step}",
            )
            parameter_digests[completed_step] = full_digest
            previous_saved_step = max(step for step in parameter_digests if step < completed_step)
            if full_digest == parameter_digests[previous_saved_step]:
                fail("saved checkpoint parameters did not change")
        local_step = {
            "step": completed_step,
            "optimizer_step_executed": True,
            "microbatches_per_dp_arm": GRADIENT_ACCUMULATION,
            "synchronized_gradient_norm": synchronized_norm,
            "gradient_groups": groups,
            "lora_projection_gradient_groups": projection_groups,
            "lora_affine_gradient_audit": affine_gradient_audit,
            "synchronized_gradient_sha256": synchronized_gradient_sha,
            "parameter_sha256": full_digest,
            "memory_world8": memory_world,
            "dp_arm": distributed.arm_index,
            "microbatches": micro_records,
        }
        gathered_steps: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_steps, local_step, group=parallel.world_group)
        for arm_start in (0, 4):
            sp_records = [
                gathered_steps[rank]["microbatches"]
                for rank in range(arm_start, arm_start + SP_SIZE)
            ]
            if len({core.object_sha256(value) for value in sp_records}) != 1:
                fail(f"step {completed_step} SP4 microbatch evidence differs")
        leaders = [gathered_steps[0], gathered_steps[4]]
        observed_logical_records = sorted(
            record["logical_record"]
            for leader in leaders
            for record in leader["microbatches"]
        )
        expected_logical_records = list(
            range(step_zero * GLOBAL_BATCH, completed_step * GLOBAL_BATCH)
        )
        if observed_logical_records != expected_logical_records:
            fail(f"step {completed_step} DP2 logical-record closure differs")
        step_record = {
            "step": completed_step,
            "optimizer_step_executed": True,
            "logical_records": [record for leader in leaders for record in leader["microbatches"]],
            "synchronized_gradient_norm": synchronized_norm,
            "gradient_groups": groups,
            "lora_projection_gradient_groups": projection_groups,
            "lora_affine_gradient_audit": affine_gradient_audit,
            "synchronized_gradient_sha256": synchronized_gradient_sha,
            "parameter_sha256": full_digest,
            "memory_world8": memory_world,
        }
        if distributed.rank == 0:
            history.append(step_record)
            print(json.dumps(step_record, sort_keys=True), flush=True)
            with (output / "history.jsonl").open("ab") as handle:
                handle.write(core.canonical_json_bytes(step_record) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            if completed_step in required_checkpoints:
                assert full_digest is not None
                checkpoint_records.append(
                    save_checkpoint(
                        root=checkpoints,
                        step=completed_step,
                        model=model,
                        optimizer=optimizer,
                        metadata={**common_checkpoint_metadata, "parameter_sha256": full_digest},
                    )
                )
        dist.barrier(group=parallel.world_group)
        lifetime_peak_allocated = max(
            lifetime_peak_allocated, int(torch.cuda.max_memory_allocated(device))
        )
        lifetime_peak_reserved = max(
            lifetime_peak_reserved, int(torch.cuda.max_memory_reserved(device))
        )

    if tuple(sorted(parameter_digests)) != required_checkpoints:
        fail("checkpoint/digest cadence differs")
    if args.execution_scope == "optimizer-canary-2" and len(set(parameter_digests.values())) != 3:
        fail("optimizer canary requires P0 != P1 != P2")
    gib = float(1024**3)
    local_lifetime_memory = {
        "world_rank": distributed.rank,
        "dp_arm": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "lifetime_peak_allocated_gib": lifetime_peak_allocated / gib,
        "lifetime_peak_reserved_gib": lifetime_peak_reserved / gib,
        "covers_model_load_t5_encode_and_all_training_steps": True,
        "step_peaks_recorded_separately": True,
        "memory_pass_threshold_gib": None,
    }
    lifetime_memory_world: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        lifetime_memory_world, local_lifetime_memory, group=parallel.world_group
    )
    if [item["world_rank"] for item in lifetime_memory_world] != list(range(WORLD_SIZE)):
        fail("lifetime per-rank memory closure differs")
    if distributed.rank == 0:
        unsigned = {
            "schema_version": RECEIPT_SCHEMA,
            "method": METHOD,
            "complete": True,
            "execution_scope": args.execution_scope,
            "optimizer_steps": steps,
            "continuous_single_process_trajectory": True,
            "fresh_official_base": True,
            "resume_consumed": False,
            "architecture": architecture,
            "lora_installation": lora_installation,
            "trainable_parameter_count": trainable_count,
            "trainable_inventory_sha256": inventory_sha,
            "parameter_digests": {str(key): value for key, value in parameter_digests.items()},
            "checkpoints": checkpoint_records,
            "checkpoint_steps": list(required_checkpoints),
            "history_steps": len(history),
            "lifetime_memory_world8": lifetime_memory_world,
            "dataset": {
                **manifest.receipt(),
                "physical_index0_train_rows_preloaded": preload["preloaded_rows"],
                "legacy_parquet_opened": False,
                "synthetic_target_index1_bytes_read": False,
            },
            "objective": {
                "preregistered_exact80_mixture": {
                    "noop": 0.4,
                    "cube": 0.2,
                    "speed": 0.2,
                    "tube": 0.2,
                },
                "realized_histogram": core.objective_histogram(steps * GLOBAL_BATCH),
                "realized_fractions": {
                    name: count / float(steps * GLOBAL_BATCH)
                    for name, count in core.objective_histogram(
                        steps * GLOBAL_BATCH
                    ).items()
                },
                "exact_registered_mixture_realized": (
                    args.execution_scope == "exact80"
                ),
                "target_always_original_real_source": True,
                "reward": False,
                "vlm": False,
            },
            "distributed": {
                "world_size": WORLD_SIZE,
                "dp_size": DP_SIZE,
                "sp_size": SP_SIZE,
                "gradient_sync": "SP4_mean_then_DP2_mean",
                "lora_applied_to_all_local_packed_tokens": True,
                "targetless_early_return": False,
            },
            "elapsed_seconds": time.monotonic() - started,
            "parent_allocation_released": False,
        }
        receipt = {**unsigned, "receipt_digest": core.object_sha256(unsigned)}
        atomic_json(output / "receipt.json", receipt)
        print(
            json.dumps(
                {
                    "complete": True,
                    "output": str(output),
                    "scope": args.execution_scope,
                    "lora_scope": args.lora_scope,
                    "optimizer_steps": steps,
                    "parent_allocation_released": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=parallel.world_group)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

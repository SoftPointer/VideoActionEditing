#!/usr/bin/env python3
"""Matched native-vs-preservation exact81 Bernini action canary.

The native arm is unchanged RV2V.  The preservation arm adds, at every exact40
scheduler boundary, the unit-gain source-only residual learned from exact no-op
reconstruction.  The adapter sees the no-op prompt, never the action prompt.
No action reward, feature reward, VLM score, synthetic target, scale sweep,
sigma gate, or clipping is used.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import infer_source_value_residual_oracle as value_audit  # noqa: E402
import load_preservation_residual_v1 as preservation_load  # noqa: E402
import preservation_source_role_v1 as preservation_role  # noqa: E402
import source_self_runtime as preservation_runtime  # noqa: E402
import train_preservation_residual_v1 as preservation_train  # noqa: E402
from preservation_residual_action_patch_v1 import (  # noqa: E402
    NativeRV2VPreservationResidualPatch,
    PreservationPatchConfig,
)
import tri_branch_unipc as sampler_contract  # noqa: E402


METHOD = "bernini-preservation-residual-action-canary-v1"
SCHEMA_VERSION = "bernini-preservation-residual-action-canary-receipt-v1"
NATIVE_GUIDANCE_MODE = "v2v_apg"
FRAME_COUNT = 81
LATENT_PHASES = 21
NATIVE_REFERENCE_INDICES = (0, 27, 53, 80)
PRESERVATION_REFERENCE_INDICES = (0, 40, 80)
ALL_REFERENCE_INDICES = tuple(sorted(set(NATIVE_REFERENCE_INDICES + PRESERVATION_REFERENCE_INDICES)))
NUM_INFERENCE_STEPS = 40
ULYSSES_SIZE = 4
FPS = 25
ARMS = ("native-rv2v", "preservation-residual")

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ActionFieldCanaryError(RuntimeError):
    """Raised before incomplete or ambiguous evidence is published."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_digest(value: Any) -> str:
    return _sha256_bytes(native.legacy.canonical_json_bytes(value))


@contextmanager
def _serialized_host_checkpoint_load() -> Any:
    """Limit an all8/64-GiB holder to one CPU deserializer at a time."""

    value = os.environ.get("PRESERVATION_INFER_LOAD_LOCK")
    if value is None:
        yield
        return
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ActionFieldCanaryError("serialized checkpoint-load lock differs")
    with path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _strong_model_freeze_certificate(model: Any) -> dict[str, Any]:
    """Hash every frozen parameter and buffer, not only ``requires_grad``.

    The older shared oracle certificate proves that an optimizer cannot see a
    trainable tensor, but it does not prove that an inference hook left tensor
    values unchanged.  This certificate deliberately excludes device/storage
    addresses so it remains stable when Bernini moves T5 and the transformer
    between CPU and GPU.  Names, module topology, dtype, shape and exact raw
    bytes are all included in the digest.
    """

    try:
        import torch
    except Exception as error:  # pragma: no cover - AUH runtime dependency
        raise ActionFieldCanaryError(
            "strong freeze certificate requires PyTorch"
        ) from error

    if not isinstance(model, torch.nn.Module) or bool(model.training):
        raise ActionFieldCanaryError("frozen model must be one eval torch module")
    modules = [
        (
            str(name),
            f"{type(module).__module__}.{type(module).__qualname__}",
        )
        for name, module in model.named_modules()
    ]
    module_names = [name for name, _ in modules]
    if len(module_names) != len(set(module_names)):
        raise ActionFieldCanaryError("frozen model module names repeat")
    adapter_modules = [
        name
        for name, class_name in modules
        if "lora" in name.lower() or "lora" in class_name.lower()
    ]
    if adapter_modules:
        raise ActionFieldCanaryError("frozen model contains LoRA/adapter modules")

    named_state = [
        *(('parameter', name, value) for name, value in model.named_parameters()),
        *(('buffer', name, value) for name, value in model.named_buffers()),
    ]
    qualified_names = [f"{kind}.{name}" for kind, name, _ in named_state]
    if len(qualified_names) != len(set(qualified_names)):
        raise ActionFieldCanaryError("frozen model state names repeat")
    content = hashlib.sha256()
    metadata_rows: list[dict[str, Any]] = []
    counts = {"parameter": 0, "buffer": 0}
    elements = {"parameter": 0, "buffer": 0}
    byte_counts = {"parameter": 0, "buffer": 0}
    for kind, name, value in named_state:
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise ActionFieldCanaryError(
                f"frozen model {kind} {name} is not materialized"
            )
        if kind == "parameter" and (
            bool(value.requires_grad) or value.grad is not None
        ):
            raise ActionFieldCanaryError(
                f"frozen model parameter {name} is trainable or retains a gradient"
            )
        detached = value.detach().to(device="cpu").contiguous()
        # ``Tensor.view(dtype)`` rejects a zero-dimensional tensor even though
        # scalar buffers are perfectly valid module state.  Flatten first so
        # parameters and buffers of every rank share one exact raw-byte path.
        raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        row = {
            "kind": kind,
            "name": str(name),
            "shape": [int(item) for item in detached.shape],
            "dtype": str(detached.dtype),
            "numel": int(detached.numel()),
            "byte_count": len(raw),
            "raw_storage_sha256": _sha256_bytes(raw),
        }
        encoded = native.legacy.canonical_json_bytes(row)
        content.update(encoded)
        content.update(b"\0")
        content.update(raw)
        counts[kind] += 1
        elements[kind] += int(detached.numel())
        byte_counts[kind] += len(raw)
        metadata_rows.append(row)

    topology_digest = _canonical_digest(modules)
    metadata_digest = _canonical_digest(metadata_rows)
    return {
        "base_frozen": True,
        "model_eval": True,
        "adapter_modules_absent": True,
        "module_count": len(modules),
        "module_topology_sha256": topology_digest,
        "parameter_tensor_count": counts["parameter"],
        "parameter_element_count": elements["parameter"],
        "parameter_byte_count": byte_counts["parameter"],
        "buffer_tensor_count": counts["buffer"],
        "buffer_element_count": elements["buffer"],
        "buffer_byte_count": byte_counts["buffer"],
        "state_metadata_sha256": metadata_digest,
        "state_content_sha256": content.hexdigest(),
        "device_and_storage_address_excluded": True,
        "exact_parameter_and_buffer_bytes_hashed": True,
    }


def _model_mutation_guard(model: Any) -> dict[str, Any]:
    """Record storage/version metadata without copying model bytes to host.

    The guard is process-local by design: storage addresses differ across
    Ulysses ranks, but every rank compares its own token before and after the
    relevant operation.  Any in-place tensor update, storage replacement,
    shape/dtype change, or trainability change invalidates the token.
    """

    try:
        import torch
    except Exception as error:  # pragma: no cover - AUH runtime dependency
        raise ActionFieldCanaryError("model mutation guard requires PyTorch") from error
    if not isinstance(model, torch.nn.Module) or bool(model.training):
        raise ActionFieldCanaryError("mutation-guard model must be frozen eval")
    rows: list[dict[str, Any]] = []
    values = [
        *(("parameter", name, value) for name, value in model.named_parameters()),
        *(("buffer", name, value) for name, value in model.named_buffers()),
    ]
    for kind, name, value in values:
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise ActionFieldCanaryError("mutation-guard tensor is not materialized")
        if kind == "parameter" and (value.requires_grad or value.grad is not None):
            raise ActionFieldCanaryError("mutation-guard parameter is not frozen")
        rows.append(
            {
                "kind": kind,
                "name": str(name),
                "shape": [int(item) for item in value.shape],
                "stride": [int(item) for item in value.stride()],
                "dtype": str(value.dtype),
                "device": str(value.device),
                "data_ptr": int(value.data_ptr()),
                "storage_offset": int(value.storage_offset()),
                "version": int(value._version),
                "requires_grad": bool(value.requires_grad),
                "gradient_absent": value.grad is None,
            }
        )
    return {
        "schema_version": "bernini-model-mutation-guard-v1",
        "state_tensor_count": len(rows),
        "process_local_storage_and_version_sha256": _canonical_digest(rows),
        "no_parameter_or_buffer_bytes_copied_to_host": True,
    }


def _rank_zero_strong_model_freeze_certificate(model: Any, *, rank: int) -> dict[str, Any]:
    """Hash exact model bytes once, then publish the certificate to all ranks."""

    import torch.distributed as dist

    payload: list[Any] = [None]
    if rank == 0:
        try:
            payload[0] = {"ok": True, "certificate": _strong_model_freeze_certificate(model)}
        except Exception as error:
            payload[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        raise ActionFieldCanaryError(f"rank-zero freeze certificate failed: {result!r}")
    certificate = result.get("certificate")
    if not isinstance(certificate, Mapping):
        raise ActionFieldCanaryError("rank-zero freeze certificate differs")
    return dict(certificate)


def _trim_host_allocator() -> bool:
    """Return unused deserialization arenas to the Linux cgroup."""

    import ctypes

    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    malloc_trim = libc.malloc_trim
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)
    return True


def _plain_json(path_value: str | Path, *, label: str) -> tuple[Path, Mapping[str, Any]]:
    requested = Path(path_value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise ActionFieldCanaryError(f"{label} path differs")
    path = requested.resolve(strict=True)
    if path != requested or not path.is_file() or path.is_symlink():
        raise ActionFieldCanaryError(f"{label} must be a plain absolute file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ActionFieldCanaryError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ActionFieldCanaryError(f"{label} root differs")
    return path, value


def _registry_cell(
    registry: Mapping[str, Any], *, cell_id: str
) -> Mapping[str, Any]:
    if registry.get("schema_version") != "bernini-self-guided-action-field-core2-v1":
        raise ActionFieldCanaryError("registry schema differs")
    contract = registry.get("contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("native_guidance_mode") != NATIVE_GUIDANCE_MODE
    ):
        raise ActionFieldCanaryError("registry native guidance mode differs")
    cells = registry.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise ActionFieldCanaryError("registry cells differ")
    matches = [row for row in cells if isinstance(row, Mapping) and row.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise ActionFieldCanaryError("registry cell lookup differs")
    cell = matches[0]
    for key in (
        "source_video",
        "source_video_sha256",
        "source_action_caption",
        "source_action_caption_sha256",
        "target_action_caption",
        "target_action_caption_sha256",
        "seed",
        "bucket_hw",
        "latent_shape",
    ):
        if key not in cell:
            raise ActionFieldCanaryError(f"registry cell lacks {key}")
    if _sha256_text(str(cell["source_action_caption"])) != cell["source_action_caption_sha256"]:
        raise ActionFieldCanaryError("source action caption digest differs")
    if _sha256_text(str(cell["target_action_caption"])) != cell["target_action_caption_sha256"]:
        raise ActionFieldCanaryError("target action caption digest differs")
    if not isinstance(cell["seed"], int) or not 0 <= cell["seed"] < 2**63:
        raise ActionFieldCanaryError("cell seed differs")
    if len(cell["bucket_hw"]) != 2 or len(cell["latent_shape"]) != 5:
        raise ActionFieldCanaryError("cell geometry differs")
    if list(cell["latent_shape"])[0:3] != [1, 16, LATENT_PHASES]:
        raise ActionFieldCanaryError("cell latent phase geometry differs")
    return cell


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--cell-id", choices=("dog", "human"), required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--training-bundle", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=native.legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=native.legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_registry_sha256",
        "runtime_source_archive_sha256",
        "launcher_source_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_adapter_sha256",
        "expected_training_receipt_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise ActionFieldCanaryError(f"{name} differs")
    for name in (
        "runtime_source_revision",
        "expected_bernini_commit",
        "expected_veomni_commit",
    ):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            raise ActionFieldCanaryError(f"{name} differs")
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise ActionFieldCanaryError("Bernini revision differs")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise ActionFieldCanaryError("VeOmni revision differs")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise ActionFieldCanaryError("checkpoint tree differs")


def _prompt_tokens(tokenizer: Any, prompt: str) -> tuple[Any, Any]:
    return native.legacy._tokenize_training_prompt(tokenizer, prompt)


def _action_field_sampling_contract(*, steps: int, seed: int) -> dict[str, Any]:
    """Pin VR2V task conditioning to Bernini's two-forward APG sampler.

    ``native_sampling_contract("rv2v")`` intentionally selects the separate
    four-forward linear ``rv2v`` guidance mode.  SGAF observes exactly the
    two calls made by ``v2v_apg`` while retaining the full video/reference
    conditions and the VR2V task prompt, so the override must be explicit and
    independently testable.
    """

    contract = native.native_sampling_contract("rv2v", steps=steps, seed=seed)
    if contract.get("guidance_mode") != "rv2v":
        raise ActionFieldCanaryError("native RV2V sampling contract changed")
    contract["guidance_mode"] = NATIVE_GUIDANCE_MODE
    return contract


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    registry_path, registry = _plain_json(args.registry, label="registry")
    if native.legacy.file_sha256(registry_path) != args.expected_registry_sha256:
        raise ActionFieldCanaryError("registry file digest differs")
    cell = _registry_cell(registry, cell_id=args.cell_id)
    bundle = preservation_load.resolve_bundle(
        args.training_bundle,
        expected_adapter_sha256=args.expected_adapter_sha256,
        expected_receipt_sha256=args.expected_training_receipt_sha256,
    )
    output_dir = native._resolve_fresh_output_dir(args.output_dir)
    source_requested = Path(str(cell["source_video"])).expanduser()
    if not source_requested.is_absolute() or source_requested.is_symlink():
        raise ActionFieldCanaryError("source path differs")
    source_path = source_requested.resolve(strict=True)
    if (
        source_path != source_requested
        or not source_path.is_file()
        or native.legacy.file_sha256(source_path) != cell["source_video_sha256"]
    ):
        raise ActionFieldCanaryError("source bytes differ")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise ActionFieldCanaryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise ActionFieldCanaryError("attention heads do not divide Ulysses4")
    inference_file_hashes = native.legacy.validate_inference_source_files(bernini_root)
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("vr2v") != native.TASK_SYSTEM_PROMPTS["vr2v"]:
        raise ActionFieldCanaryError("runtime VR2V system prompt differs")
    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise ActionFieldCanaryError("runtime negative prompt differs")
    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != ULYSSES_SIZE
        or distributed.ulysses_size != ULYSSES_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise ActionFieldCanaryError("runtime requires AUH WORLD4/Ulysses4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=ULYSSES_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
        raise ActionFieldCanaryError(f"checkpoint validation failed: {checkpoint_rows[0]}")
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    source_tensor, source_metadata, source_sha = source_audit.prepare_hashed_source_snapshot(
        source_path
    )
    bucket_hw = tuple(int(item) for item in cell["bucket_hw"])
    latent_shape = tuple(int(item) for item in cell["latent_shape"])
    if (
        source_sha != cell["source_video_sha256"]
        or source_metadata.get("frame_count") != FRAME_COUNT
        or tuple(source_metadata.get("source_derived_bucket_hw", ())) != bucket_hw
    ):
        raise ActionFieldCanaryError("source exact81 geometry differs")

    target_caption = str(cell["target_action_caption"])
    source_caption = str(cell["source_action_caption"])
    rv2v_target_prompt = native.build_task_prompt(
        "rv2v", target_caption, prompt_cleaner=prompt_clean
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    rv2v_ids, rv2v_mask = _prompt_tokens(tokenizer, rv2v_target_prompt)
    noop_text = preservation_runtime.tokenize_generic_instruction(
        tokenizer, preservation_train.EXACT_NOOP_INSTRUCTION, device
    )
    negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
        tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
        raise ActionFieldCanaryError("renderer scheduler differs")
    with _serialized_host_checkpoint_load():
        model = BerniniRendererModel(config)
        model.eval().requires_grad_(False)
        model.to(device)
        host_allocator_trim_invoked = _trim_host_allocator()
    full_freeze_certificate = _rank_zero_strong_model_freeze_certificate(
        model, rank=distributed.rank
    )
    full_guard_before_prompt = _model_mutation_guard(model)

    # Prompt embeddings are the only sampling-time product of the frozen T5.
    # Certify the complete model before and after prompt encoding, then retire
    # the encoder instead of keeping four CPU copies resident under a 60-GiB
    # Slurm cgroup.  This changes resource lifetime, not the sampled equation.
    model.t5_text_encoder.to(device)
    with torch.inference_mode():
        rv2v_target_embeds = model.encode_prompt(
            rv2v_ids.to(device), rv2v_mask.to(device)
        ).detach()
        noop_text_lens, noop_embeds = model.get_t5_text_embeddings(
            noop_text["input_ids"],
            noop_text["attention_mask"],
            noop_text["t5_input_lens"],
        )
        noop_embeds = noop_embeds.detach()
        uncond_embeds = model.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
    full_guard_after_prompt = _model_mutation_guard(model)
    if full_guard_after_prompt != full_guard_before_prompt:
        raise ActionFieldCanaryError("frozen model changed during prompt encoding")
    retired_text_encoder = model.t5_text_encoder
    model.t5_text_encoder = None
    del retired_text_encoder
    gc.collect()
    _trim_host_allocator()
    torch.cuda.empty_cache()

    # Only rank zero encodes/decodes pixels.  Other Ulysses ranks receive the
    # exact latent tensors by broadcast and never need a private VAE copy.
    vae = None
    reference_shape = (1, 16, 1, latent_shape[3], latent_shape[4])
    if distributed.rank == 0:
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval().requires_grad_(False)
        vae.to(device)
        source_pixels = source_tensor.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            source_latent = _vae_encode(vae, source_pixels).contiguous()
            references = {
                index: _vae_encode(
                    vae, source_pixels[:, :, index : index + 1].contiguous()
                ).contiguous()
                for index in ALL_REFERENCE_INDICES
            }
        del source_pixels
        vae.to("cpu")
    else:
        source_latent = torch.empty(latent_shape, device=device, dtype=torch.float32)
        references = {
            index: torch.empty(reference_shape, device=device, dtype=torch.float32)
            for index in ALL_REFERENCE_INDICES
        }
    dist.broadcast(source_latent, src=0)
    for index in ALL_REFERENCE_INDICES:
        dist.broadcast(references[index], src=0)
    if tuple(source_latent.shape) != latent_shape or any(
        tuple(value.shape) != reference_shape for value in references.values()
    ):
        raise ActionFieldCanaryError("source condition geometry differs")
    condition_identities = {
        "source_video": native._all_rank_tensor_identity(
            source_latent, label=f"{args.cell_id}_source_video", world_size=ULYSSES_SIZE
        ),
        "references": {
            str(index): native._all_rank_tensor_identity(
                value,
                label=f"{args.cell_id}_source_reference_{index}",
                world_size=ULYSSES_SIZE,
            )
            for index, value in references.items()
        },
    }
    del source_tensor
    gc.collect()
    torch.cuda.empty_cache()
    diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
    sampler_contract._validate_scheduler_contract(
        diffusion.scheduler, expected_flow_shift=native.FLOW_SHIFT
    )
    if diffusion.transformer_2 is not None:
        raise ActionFieldCanaryError(
            "canary is pinned to the single-expert Bernini-R 1.3B checkpoint"
        )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    # Certify the native transformer topology on both sides of the temporary
    # adapter installation.  A second guard, recorded after strict_load, then
    # proves that neither the frozen base nor the loaded adapter tensors change
    # while sampling.  Comparing the adapted topology before sampling with the
    # restored native topology afterwards is invalid by construction: restore()
    # deliberately removes the wrapper modules and their adapter parameters.
    native_sampling_guard_before = _model_mutation_guard(model)
    adapter, adapter_load_receipt = preservation_load.strict_load(
        diffusion.transformer, bundle
    )
    adapted_sampling_guard_before = _model_mutation_guard(model)

    generated: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    initial_noise: dict[str, Any] = {}
    initial_noise_identities: dict[str, Any] = {}
    preservation_receipts: dict[str, Any] = {}
    with torch.inference_mode():
        for arm in ARMS:
            patch = None
            if arm == "preservation-residual":
                patch = NativeRV2VPreservationResidualPatch(
                    diffusion,
                    adapter=adapter,
                    noop_prompt_embeds=noop_embeds,
                    noop_text_lens=noop_text_lens,
                    source_latent=source_latent,
                    source_references=[
                        references[index] for index in PRESERVATION_REFERENCE_INDICES
                    ],
                    rope=rope,
                    config=PreservationPatchConfig(
                        target_latent_shape=latent_shape,
                        sequence_parallel_size=ULYSSES_SIZE,
                        expected_steps=NUM_INFERENCE_STEPS,
                    ),
                )
                patch.install()
            sample_kwargs = {
                "prompt_embeds": rv2v_target_embeds,
                "uncond_prompt_embeds": uncond_embeds,
                "image_vae_latents": None,
                "multi_video_vae_latents": [source_latent],
                "multi_image_vae_latents": [
                    references[index] for index in NATIVE_REFERENCE_INDICES
                ],
                "width": bucket_hw[1],
                "height": bucket_hw[0],
                "device": device,
                **_action_field_sampling_contract(
                    steps=NUM_INFERENCE_STEPS, seed=int(cell["seed"])
                ),
            }
            try:
                result, capture = native._sample_with_native_initial_noise_observer(
                    sample_fn=lambda kw=sample_kwargs: diffusion.sample(**kw),
                    wan_diffusion_module=wan_diffusion,
                    expected_shape=latent_shape,
                    expected_device=device,
                    expected_seed=int(cell["seed"]),
                )
            finally:
                if patch is not None and patch.installed and not patch.restored:
                    patch.restore()
            if patch is not None:
                preservation_receipts[arm] = dict(patch.finalize())
            else:
                preservation_receipts[arm] = {
                    "native_baseline": True,
                    "preservation_noop_forwards": 0,
                    "preservation_residual_applied": False,
                }
            if (
                not isinstance(result, torch.Tensor)
                or tuple(result.shape) != latent_shape
                or result.dtype != torch.float32
                or result.requires_grad
                or result.grad_fn is not None
                or not bool(torch.isfinite(result).all().item())
            ):
                raise ActionFieldCanaryError("native sampler result differs")
            stored = result.detach().to(device="cpu").contiguous()
            generated[arm] = stored
            generated_identities[arm] = native._all_rank_tensor_identity(
                stored, label=f"{args.cell_id}_{arm}", world_size=ULYSSES_SIZE
            )
            initial_noise[arm] = capture
            initial_noise_identities[arm] = native._all_rank_tensor_identity(
                capture.tensor,
                label=f"{args.cell_id}_{arm}_official_initial_gaussian",
                world_size=ULYSSES_SIZE,
            )

    if len({capture.raw_value_sha256 for capture in initial_noise.values()}) != 1:
        raise ActionFieldCanaryError("arms did not share one official Gaussian")
    local_trace_digest = _canonical_digest(preservation_receipts)
    trace_rows: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(trace_rows, local_trace_digest)
    if len(set(trace_rows)) != 1:
        raise ActionFieldCanaryError("action-field traces differ across SP4 ranks")
    adapted_sampling_guard_after = _model_mutation_guard(model)
    if adapted_sampling_guard_after != adapted_sampling_guard_before or any(
        p.requires_grad for p in model.parameters()
    ):
        raise ActionFieldCanaryError("frozen base or adapter changed during sampling")
    adapter.restore()
    native_sampling_guard_after = _model_mutation_guard(model)
    if native_sampling_guard_after != native_sampling_guard_before or any(
        p.requires_grad for p in model.parameters()
    ):
        raise ActionFieldCanaryError("frozen native model changed after adapter restore")
    # Do not create four new CPU transformer copies merely to free VRAM.  All
    # sampling evidence is detached on CPU and the exact model certificate is
    # already complete, so destroy the sampling graph before rank-zero decode.
    patch = None
    del adapter, diffusion, model, result
    del rv2v_target_embeds, noop_embeds, uncond_embeds
    del source_latent, references, rope
    gc.collect()
    torch.cuda.empty_cache()

    if distributed.rank == 0:
        output_dir.mkdir(parents=False, exist_ok=False)
        noise_artifacts = {
            arm: native._save_initial_noise_atomically(
                output_dir / f"{arm}.official-initial-gaussian.safetensors",
                initial_noise[arm],
                all_rank_identity=initial_noise_identities[arm],
            )
            for arm in ARMS
        }
        generated_for_decode = {
            arm: value.to(device=device).contiguous() for arm, value in generated.items()
        }
        try:
            if vae is None:
                raise ActionFieldCanaryError("rank-zero VAE lifetime differs")
            outputs = native._save_outputs(
                output_dir=output_dir,
                generated=generated_for_decode,
                vae=vae,
                bucket_hw=bucket_hw,
                device=device,
                save_output_fn=save_output,
            )
        finally:
            generated_for_decode.clear()
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "cell_id": args.cell_id,
            "input": {
                "source_video": str(source_path),
                "source_video_sha256": source_sha,
                "source_action_caption": source_caption,
                "source_action_caption_sha256": _sha256_text(source_caption),
                "target_action_caption": target_caption,
                "target_action_caption_sha256": _sha256_text(target_caption),
                "mask_track_pose_flow": False,
                "generated_owner_media": False,
                "target_video": False,
            },
            "prompts": {
                "rv2v_target_full_prompt_sha256": _sha256_text(rv2v_target_prompt),
                "preservation_exact_noop_instruction": preservation_train.EXACT_NOOP_INSTRUCTION,
                "preservation_exact_noop_instruction_sha256": _sha256_text(
                    preservation_train.EXACT_NOOP_INSTRUCTION
                ),
                "adapter_receives_action_text": False,
            },
            "sampling": {
                "frame_count": FRAME_COUNT,
                "latent_phases": LATENT_PHASES,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_mode": NATIVE_GUIDANCE_MODE,
                "seed": int(cell["seed"]),
                "arms": list(ARMS),
                "same_official_gaussian_all_arms": True,
                "native_source_reference_indices": list(NATIVE_REFERENCE_INDICES),
                "preservation_source_reference_indices": list(
                    PRESERVATION_REFERENCE_INDICES
                ),
                "source_rich_initial_noise": False,
                "native_target_initialization": native.TARGET_INITIALIZATION,
            },
            "preservation_residual": preservation_receipts,
            "training_bundle": {
                "root": str(bundle.root),
                "adapter_sha256": bundle.adapter_sha256,
                "receipt_sha256": bundle.receipt_sha256,
                "receipt_digest": bundle.receipt_digest,
                "adapter_rank": bundle.adapter_rank,
                "strict_load": adapter_load_receipt,
            },
            "condition_identities": condition_identities,
            "generated_identities": generated_identities,
            "initial_noise_artifacts": noise_artifacts,
            "outputs": outputs,
            "checkpoint": checkpoint_identity,
            "freeze_certificate": {
                "rank_zero_exact_full_model": full_freeze_certificate,
                "exact_full_model_bytes_hashed_on_rank_zero_only": True,
                "all_ranks_prompt_mutation_guard_before": full_guard_before_prompt,
                "all_ranks_prompt_mutation_guard_after": full_guard_after_prompt,
                "all_ranks_model_unchanged_during_prompt_encoding": True,
                "text_encoder_retired_before_vae_and_sampling": True,
                "all_ranks_native_mutation_guard_before_adapter": native_sampling_guard_before,
                "all_ranks_native_mutation_guard_after_restore": native_sampling_guard_after,
                "all_ranks_adapted_mutation_guard_before_sampling": adapted_sampling_guard_before,
                "all_ranks_adapted_mutation_guard_after_sampling": adapted_sampling_guard_after,
                "all_ranks_sampling_model_unchanged": True,
            },
            "source_revisions": {
                "bernini": bernini_revision,
                "veomni": veomni_revision,
                "runtime_method": args.runtime_source_revision,
                "runtime_source_archive_sha256": args.runtime_source_archive_sha256,
                "launcher_source_sha256": args.launcher_source_sha256,
                "inference_files": inference_file_hashes,
            },
            "runtime_versions": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "diffusers": diffusers_version,
                "transformers": transformers_version,
            },
            "rank_serialized_cpu_checkpoint_load": bool(
                os.environ.get("PRESERVATION_INFER_LOAD_LOCK")
            ),
            "host_allocator_trim_invoked_after_checkpoint_load": host_allocator_trim_invoked,
            "rank_zero_only_vae": True,
            "sampling_model_destroyed_without_cpu_offload_before_decode": True,
            "training_performed": False,
            "parameter_update": False,
            "objective": "preservation_only_exact_noop_residual",
            "action_reward_consumed": False,
            "feature_reward_consumed": False,
            "vlm_reward_consumed": False,
            "synthetic_target_consumed": False,
            "scientific_or_action_editing_claim_authorized": False,
        }
        receipt["receipt_digest"] = _canonical_digest(receipt)
        value_audit.write_receipt_atomically(output_dir / "receipt.json", receipt)
        print(native.legacy.canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    del generated, initial_noise
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "ActionFieldCanaryError",
    "METHOD",
    "NATIVE_GUIDANCE_MODE",
    "_action_field_sampling_contract",
    "_registry_cell",
    "build_parser",
]

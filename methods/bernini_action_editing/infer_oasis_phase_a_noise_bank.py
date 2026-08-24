#!/usr/bin/env python3
"""Generate one OASIS family noise bank with frozen native Bernini RV2V-4.

This is a dedicated external-initial-noise schema.  It must not be presented
as a legacy PAIR-v5 native rollout: that schema requires
``external_initial_noise_injection == false``.  For each of the two source
cells and two registered seeds in one family, this runner executes three
independent full exact40/exact81 rollouts:

* the official Gaussian, forwarded by exact tensor-object identity;
* an unordered source-appearance-set carrier at rho=0.05; and
* the same carrier at rho=0.10.

The active carrier receives only four independently VAE-encoded T=1 frames
derived from the source.  Rank zero broadcasts every frame bit-exactly across
SP4 before the frame set is copied to standalone CPU storage.  The motion-null
operator consumes neither temporal order nor a full-video latent.  All arms
first call Bernini's original ``randn_tensor`` with unchanged arguments; each
active receipt binds its returned official Gaussian as its parent.

The generated endpoints are candidates only.  This runner performs no action
or source score, selection, optimizer step, training, or success claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native  # noqa: E402
import motion_null_appearance_noise as appearance_noise  # noqa: E402
import oasis_phase_a_core as oasis_core  # noqa: E402
import oasis_phase_a_manifest as oasis_manifest  # noqa: E402


SCHEMA_VERSION = "bernini-oasis-phase-a-source-set-noise-family-bank-v2"
ROLLOUT_SCHEMA = oasis_core.ROLLOUT_SCHEMA
METHOD = "frozen-bernini-oasis-source-set-noise-bank"
WORLD_SIZE = 4
SP_SIZE = 4
FRAME_COUNT = 81
LATENT_PHASES = 21
NUM_INFERENCE_STEPS = 40
GUIDANCE_POLICY = "fixed_native_rv2v_no_ablation"
NOISE_OPERATOR_CALLABLE = oasis_core.NOISE_OPERATOR_CALLABLE
TEMPORARY_MUTATION_SURFACE = (
    "bernini.models.wan_diffusion.randn_tensor",
)
NOISE_ARM_ORDER = oasis_core.NOISE_ARM_ORDER
NOISE_RHO_BY_ARM = oasis_core.NOISE_RHO_BY_ARM
if NOISE_ARM_ORDER != tuple(oasis_manifest.ARM_ORDER):
    raise RuntimeError("OASIS noise arm/rho registry differs from the sealed manifest")

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class OASISNoiseBankError(RuntimeError):
    """The candidate bank is not native, matched, source-only, or hash-closed."""


@dataclass(frozen=True)
class NoiseInjectionCapture:
    noise_arm: str
    rho: float
    baseline_tensor: Any
    injected_tensor: Any
    baseline_raw_value_sha256: str
    injected_raw_value_sha256: str
    baseline_content_sha256: str
    injected_content_sha256: str
    requested_shape: tuple[int, ...]
    requested_dtype: str
    requested_device: str
    generator_device: str
    generator_initial_seed: int
    original_randn_call_count: int
    external_initial_noise_injection: bool
    original_return_object_forwarded: bool
    source_frame_set_digest: str
    operator_receipt: Mapping[str, Any]
    operator_receipt_digest: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--family", choices=oasis_manifest.FAMILY_ORDER, required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _canonical(value: Any) -> bytes:
    return oasis_manifest.canonical_json_bytes(value)


def _object_sha(value: Any) -> str:
    return oasis_manifest.object_sha256(value)


def _sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1_RE if length == 40 else _SHA256_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise OASISNoiseBankError(f"{label} must be lowercase SHA-{length}")
    return value


def _fresh_output(value: str) -> Path:
    requested = Path(value)
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or _SAFE_NAME_RE.fullmatch(requested.name) is None
        or requested.exists()
        or requested.is_symlink()
    ):
        raise OASISNoiseBankError("output must be a fresh absolute safe non-root path")
    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink() or requested != parent / requested.name:
        raise OASISNoiseBankError("output parent/path is not canonical")
    return requested


def _output_staging_directory(final: Path) -> Path:
    """Create a private sibling and publish it only after receipt closure."""

    staging = Path(
        tempfile.mkdtemp(dir=final.parent, prefix=f".{final.name}.partial-")
    )
    staging.chmod(0o755)
    if staging.parent != final.parent or staging.is_symlink():
        raise OASISNoiseBankError("output staging directory escaped its parent")
    return staging


def _rebase_artifact_paths(value: Any, *, old_root: Path, new_root: Path) -> Any:
    old = str(old_root)
    new = str(new_root)
    if isinstance(value, str):
        if value == old:
            return new
        if value.startswith(old + os.sep):
            return new + value[len(old) :]
        return value
    if isinstance(value, Mapping):
        return {
            key: _rebase_artifact_paths(item, old_root=old_root, new_root=new_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rebase_artifact_paths(item, old_root=old_root, new_root=new_root)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _rebase_artifact_paths(item, old_root=old_root, new_root=new_root)
            for item in value
        )
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_output_transaction(*, staging: Path, final: Path) -> None:
    if final.exists() or final.is_symlink():
        raise OASISNoiseBankError("refusing to replace an output directory")
    if staging.parent != final.parent or not staging.is_dir() or staging.is_symlink():
        raise OASISNoiseBankError("output staging directory is invalid")
    _fsync_directory(staging)
    os.replace(staging, final)
    _fsync_directory(final.parent)


def _tensor_identity(value: Any, *, label: str) -> Mapping[str, Any]:
    try:
        result = native.value_audit.tensor_identity(value, label=label)
    except Exception as error:
        raise OASISNoiseBankError(str(error)) from error
    digest = result.get("raw_storage_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise OASISNoiseBankError(f"{label} tensor identity differs")
    return dict(result)


def source_frame_set_digest(frames: Sequence[Any]) -> str:
    """Hash an unordered set of four standalone T=1 source latents."""

    if isinstance(frames, (str, bytes)) or len(frames) != len(native.RV2V_REFERENCE_INDICES):
        raise OASISNoiseBankError("source appearance set requires exactly four T=1 frames")
    rows = []
    for ordinal, frame in enumerate(frames):
        identity = _tensor_identity(frame, label=f"source_T1_frame_{ordinal}")
        if int(frame.shape[2]) != 1:
            raise OASISNoiseBankError("source appearance set contains a temporal latent")
        rows.append(
            {
                "shape": [int(item) for item in frame.shape],
                "dtype": str(frame.dtype),
                "raw_value_sha256": identity["raw_storage_sha256"],
            }
        )
    rows.sort(key=lambda row: row["raw_value_sha256"])
    if len({row["raw_value_sha256"] for row in rows}) != len(rows):
        raise OASISNoiseBankError("source appearance frame set aliases one T=1 value")
    return _object_sha(
        {
            "frame_set": rows,
            "frame_order_consumed": False,
            "source_temporal_phase_consumed": False,
        }
    )


def carrier_seed_for(*, sample_digest: str, seed: int) -> int:
    try:
        return oasis_core.carrier_seed_for(sample_digest=sample_digest, seed=seed)
    except oasis_core.OASISPhaseAError as error:
        raise OASISNoiseBankError(str(error)) from error


def _sample_with_oasis_noise_arm(
    *,
    sample_fn: Callable[[], Any],
    wan_diffusion_module: Any,
    noise_arm: str,
    independent_frame_latents_cpu: Sequence[Any],
    expected_shape: Sequence[int],
    expected_device: Any,
    expected_seed: int,
    carrier_seed: int,
    canonical_randn_tensor: Optional[Callable[..., Any]] = None,
) -> tuple[Any, NoiseInjectionCapture]:
    """Compose with native sampling by replacing only ``randn_tensor``.

    The wrapper does not wrap or replace ``sample``, ``sample_one_step``, the
    scheduler, or any CFG/APG implementation.  This narrow boundary is kept
    deliberately independent so a later guidance ablation can compose with it
    without changing the source-set candidate bank defined here.
    """

    try:
        import torch
        if canonical_randn_tensor is None:
            from diffusers.utils.torch_utils import randn_tensor as canonical
        else:
            canonical = canonical_randn_tensor
    except ImportError as error:  # pragma: no cover - AUH runtime supplies dependencies
        raise OASISNoiseBankError("noise injection requires PyTorch and Diffusers") from error
    if not callable(sample_fn):
        raise OASISNoiseBankError("sample_fn must be callable")
    if noise_arm not in NOISE_ARM_ORDER:
        raise OASISNoiseBankError("noise arm is outside OASIS registry")
    rho = NOISE_RHO_BY_ARM[noise_arm]
    expected = tuple(int(item) for item in expected_shape)
    if (
        len(expected) != 5
        or tuple(expected[:3]) != (1, 16, LATENT_PHASES)
        or expected[3] < appearance_noise.MIN_SPATIAL_EXTENT
        or expected[4] < appearance_noise.MIN_SPATIAL_EXTENT
    ):
        raise OASISNoiseBankError("noise injection requires exact81 [1,16,21,H,W]")
    if type(expected_seed) is not int or not 0 <= expected_seed < 2**63:
        raise OASISNoiseBankError("expected seed differs")
    frame_set_digest = source_frame_set_digest(independent_frame_latents_cpu)
    original = getattr(wan_diffusion_module, "randn_tensor", None)
    if original is not canonical:
        raise OASISNoiseBankError("pinned randn_tensor is already replaced or differs")
    calls: list[dict[str, Any]] = []

    def injected_randn_tensor(*call_args: Any, **call_kwargs: Any) -> Any:
        shape_value = call_args[0] if call_args else call_kwargs.get("shape")
        try:
            requested_shape = tuple(int(item) for item in shape_value)
        except Exception as error:
            raise OASISNoiseBankError("native randn_tensor shape differs") from error
        generator = call_kwargs.get("generator")
        if not isinstance(generator, torch.Generator):
            raise OASISNoiseBankError("native noise must use one torch.Generator")
        baseline_native = original(*call_args, **call_kwargs)
        if (
            not isinstance(baseline_native, torch.Tensor)
            or tuple(int(item) for item in baseline_native.shape) != expected
            or baseline_native.dtype != torch.float32
            or baseline_native.device != torch.device(expected_device)
            or not baseline_native.is_contiguous()
            or baseline_native.requires_grad
            or not bool(torch.isfinite(baseline_native).all().item())
        ):
            raise OASISNoiseBankError("official native Gaussian contract differs")
        baseline_cpu = baseline_native.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        operator = appearance_noise.build_motion_null_appearance_noise(
            canonical_gaussian=baseline_cpu,
            independent_frame_latents=independent_frame_latents_cpu,
            rho=rho,
            carrier_seed=carrier_seed,
        )
        if rho == 0.0:
            if operator.initial_noise is not baseline_cpu:
                raise OASISNoiseBankError("rho0 operator lost exact Gaussian CPU alias")
            injected_native = baseline_native
        else:
            injected_native = operator.initial_noise.to(
                device=baseline_native.device, dtype=baseline_native.dtype
            ).contiguous()
            if injected_native is baseline_native or torch.equal(injected_native, baseline_native):
                raise OASISNoiseBankError("active source-set arm did not alter initial noise")
        injected_cpu = injected_native.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        baseline_identity = _tensor_identity(baseline_cpu, label="official_parent_gaussian")
        injected_identity = _tensor_identity(injected_cpu, label="sampler_initial_noise")
        calls.append(
            {
                "requested_shape": requested_shape,
                "requested_dtype": str(call_kwargs.get("dtype")),
                "requested_device": str(call_kwargs.get("device")),
                "generator_device": str(generator.device),
                "generator_initial_seed": int(generator.initial_seed()),
                "baseline_tensor": baseline_cpu,
                "injected_tensor": injected_cpu,
                "baseline_identity": baseline_identity,
                "injected_identity": injected_identity,
                "original_return_object_forwarded": injected_native is baseline_native,
                "operator_receipt": dict(operator.receipt),
            }
        )
        return injected_native

    setattr(injected_randn_tensor, "_oasis_source_set_noise_injector", True)
    setattr(wan_diffusion_module, "randn_tensor", injected_randn_tensor)
    wrapper_unchanged = True
    try:
        result = sample_fn()
    finally:
        wrapper_unchanged = (
            getattr(wan_diffusion_module, "randn_tensor", None)
            is injected_randn_tensor
        )
        setattr(wan_diffusion_module, "randn_tensor", original)
    if not wrapper_unchanged or getattr(wan_diffusion_module, "randn_tensor", None) is not original:
        raise OASISNoiseBankError("randn_tensor injector was not restored exactly")
    if len(calls) != 1:
        raise OASISNoiseBankError("native sampler must call official randn_tensor exactly once")
    call = calls[0]
    expected_device_text = str(torch.device(expected_device))
    if (
        call["requested_shape"] != expected
        or call["requested_dtype"] != str(torch.float32)
        or call["requested_device"] != expected_device_text
        or call["generator_device"] != "cpu"
        or call["generator_initial_seed"] != expected_seed
    ):
        raise OASISNoiseBankError("official native RNG request/seed differs")
    baseline_identity = call["baseline_identity"]
    injected_identity = call["injected_identity"]
    active = rho > 0.0
    if (
        bool(call["original_return_object_forwarded"]) is active
        or (not active and baseline_identity["raw_storage_sha256"] != injected_identity["raw_storage_sha256"])
    ):
        raise OASISNoiseBankError("rho0/active injection identity contract differs")
    operator_receipt = call["operator_receipt"]
    return result, NoiseInjectionCapture(
        noise_arm=noise_arm,
        rho=rho,
        baseline_tensor=call["baseline_tensor"],
        injected_tensor=call["injected_tensor"],
        baseline_raw_value_sha256=str(baseline_identity["raw_storage_sha256"]),
        injected_raw_value_sha256=str(injected_identity["raw_storage_sha256"]),
        baseline_content_sha256=str(baseline_identity["content_sha256"]),
        injected_content_sha256=str(injected_identity["content_sha256"]),
        requested_shape=expected,
        requested_dtype=call["requested_dtype"],
        requested_device=call["requested_device"],
        generator_device=call["generator_device"],
        generator_initial_seed=call["generator_initial_seed"],
        original_randn_call_count=1,
        external_initial_noise_injection=active,
        original_return_object_forwarded=bool(call["original_return_object_forwarded"]),
        source_frame_set_digest=frame_set_digest,
        operator_receipt=operator_receipt,
        operator_receipt_digest=_object_sha(operator_receipt),
    )


def _save_tensor(path: Path, tensor: Any, *, key: str, metadata: Mapping[str, str]) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if path.exists() or path.is_symlink() or path.suffix != ".safetensors":
        raise OASISNoiseBankError("tensor artifact path must be fresh safetensors")
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if (
        value.ndim != 5
        or tuple(int(item) for item in value.shape[:3]) != (1, 16, LATENT_PHASES)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise OASISNoiseBankError("noise artifact is not finite exact81 FP32")
    before = _tensor_identity(value, label=f"{key}_before_save")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file({key: value}, str(temporary), metadata=dict(metadata))
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [key]:
                raise OASISNoiseBankError("tensor artifact key differs")
            restored = opened.get_tensor(key).contiguous()
            restored_metadata = dict(opened.metadata() or {})
        if not torch.equal(restored, value):
            raise OASISNoiseBankError("tensor artifact round trip differs")
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return {
        "path": str(path),
        "file_sha256": oasis_manifest.file_sha256(path),
        "tensor_key": key,
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "raw_value_sha256": before["raw_storage_sha256"],
        "content_sha256": before["content_sha256"],
        "metadata": restored_metadata,
        "roundtrip_byte_exact": True,
    }


def _save_source_frame_set(path: Path, frames: Sequence[Any]) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if path.exists() or path.is_symlink():
        raise OASISNoiseBankError("source frame-set artifact path must be fresh")
    tensors = {
        f"source_ref_{index:03d}": frame.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for index, frame in zip(native.RV2V_REFERENCE_INDICES, frames)
    }
    save_file(
        tensors,
        str(path),
        metadata={
            "coordinate": "four_standalone_rank0_broadcast_wan_T1_latents",
            "frame_order_consumed_by_carrier": "false",
            "full_video_latent_consumed_by_carrier": "false",
        },
    )
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        keys = list(opened.keys())
        restored = [opened.get_tensor(key).contiguous() for key in keys]
    if keys != sorted(tensors) or any(
        not torch.equal(restored[index], tensors[key]) for index, key in enumerate(keys)
    ):
        raise OASISNoiseBankError("source frame-set artifact round trip differs")
    return {
        "path": str(path),
        "file_sha256": oasis_manifest.file_sha256(path),
        "tensor_keys": keys,
        "reference_indices_for_provenance_only": list(native.RV2V_REFERENCE_INDICES),
        "frame_set_digest": source_frame_set_digest(frames),
        "frame_order_consumed_by_carrier": False,
        "full_video_latent_consumed_by_carrier": False,
        "rank0_broadcast_before_cpu_copy": True,
    }


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise OASISNoiseBankError("refusing to overwrite receipt")
    payload = _canonical(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _all_rank_exact(value: Any, *, label: str) -> Mapping[str, Any]:
    try:
        return native._all_rank_tensor_identity(value, label=label, world_size=WORLD_SIZE)
    except Exception as error:
        raise OASISNoiseBankError(str(error)) from error


def _sampling_contract(seed: int) -> Mapping[str, Any]:
    value = native.native_sampling_contract("rv2v", steps=NUM_INFERENCE_STEPS, seed=seed)
    if value["num_frames"] != FRAME_COUNT or value["num_inference_steps"] != NUM_INFERENCE_STEPS:
        raise OASISNoiseBankError("native exact81/exact40 sampling contract differs")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = _fresh_output(args.output_dir)
    manifest = oasis_manifest.load_phase_a_manifest(
        args.manifest, args.expected_manifest_sha256, verify_files=True
    )
    if manifest.checkpoint_tree_sha256 != _sha(
        args.expected_checkpoint_tree_sha256, length=64, label="checkpoint tree"
    ):
        raise OASISNoiseBankError("manifest/checkpoint tree differs")
    cells = manifest.cells_for_family(args.family)
    if tuple(cell.analysis_split for cell in cells) != oasis_manifest.SPLIT_ORDER:
        raise OASISNoiseBankError("family source cells do not close fit/confirmation")
    checkpoint_manifest = Path(args.checkpoint_content_manifest)
    if (
        not checkpoint_manifest.is_absolute()
        or not checkpoint_manifest.is_file()
        or checkpoint_manifest.is_symlink()
        or oasis_manifest.file_sha256(checkpoint_manifest)
        != _sha(
            args.expected_checkpoint_content_manifest_sha256,
            length=64,
            label="checkpoint content manifest SHA",
        )
    ):
        raise OASISNoiseBankError("checkpoint content manifest binding differs")
    source_revision = _sha(args.method_source_revision, length=40, label="method source revision")
    source_archive_sha = _sha(
        args.method_source_archive_sha256, length=64, label="method source archive SHA"
    )
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=(
                    native.legacy.trainer.BERNINI_OFFICIAL_COMMIT
                ),
                expected_veomni_commit=(
                    native.legacy.trainer.VEOMNI_TESTED_COMMIT
                ),
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise OASISNoiseBankError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % SP_SIZE:
        raise OASISNoiseBankError("checkpoint attention heads are not SP4-compatible")
    inference_hashes = native.legacy.validate_inference_source_files(bernini_root)
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
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode

    distributed = native.legacy.inference_distributed_contract()
    if distributed.world_size != WORLD_SIZE or distributed.ulysses_size != SP_SIZE:
        raise OASISNoiseBankError("noise bank requires WORLD4/Ulysses-SP4")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise OASISNoiseBankError("noise bank requires AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
        raise OASISNoiseBankError(f"checkpoint validation failed: {checkpoint_rows[0]}")
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
        tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
    )
    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise OASISNoiseBankError("native negative prompt differs")
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except Exception as error:
        raise OASISNoiseBankError(str(error)) from error
    if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
        raise OASISNoiseBankError("renderer is not pinned native UniPC shift5")
    model = BerniniRendererModel(config)
    model.eval().requires_grad_(False)
    freeze_before = native.source_audit.model_freeze_certificate(model)
    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    vae.eval().requires_grad_(False)

    generated: dict[tuple[str, int, str], Any] = {}
    captures: dict[tuple[str, int, str], NoiseInjectionCapture] = {}
    cell_runtime: dict[str, Mapping[str, Any]] = {}
    try:
        for cell in cells:
            source_tensor, source_metadata, source_sha = (
                native.source_audit.prepare_hashed_source_snapshot(cell.source_video)
            )
            if source_sha != cell.source_video_sha256:
                raise OASISNoiseBankError(f"source snapshot changed: {cell.sample_id}")
            bucket_hw = tuple(int(item) for item in source_metadata["source_derived_bucket_hw"])
            prompt = native.build_task_prompt(
                "rv2v", cell.edit_instruction, prompt_cleaner=prompt_clean
            )
            positive_ids, positive_mask = native.legacy._tokenize_training_prompt(
                tokenizer, prompt
            )
            prompt_sha = oasis_manifest.text_sha256(prompt)
            positive_ids_identity = _tensor_identity(
                positive_ids.detach().to(device="cpu").contiguous(),
                label=f"{cell.sample_id}_positive_input_ids",
            )
            positive_mask_identity = _tensor_identity(
                positive_mask.detach().to(device="cpu").contiguous(),
                label=f"{cell.sample_id}_positive_attention_mask",
            )
            model.to("cpu")
            vae.to(device)
            source_pixels = source_tensor.to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                full_source_latent = _vae_encode(vae, source_pixels).contiguous()
                reference_latents = {
                    index: _vae_encode(
                        vae,
                        source_pixels[:, :, index : index + 1].contiguous(),
                    ).contiguous()
                    for index in native.RV2V_REFERENCE_INDICES
                }
            full_broadcast = native._broadcast_condition_from_rank_zero(
                full_source_latent,
                label=f"{cell.sample_id}_full_source",
                world_size=WORLD_SIZE,
            )
            reference_broadcasts = {
                str(index): native._broadcast_condition_from_rank_zero(
                    latent,
                    label=f"{cell.sample_id}_source_ref_{index}",
                    world_size=WORLD_SIZE,
                )
                for index, latent in reference_latents.items()
            }
            full_identity = _all_rank_exact(
                full_source_latent, label=f"{cell.sample_id}_full_source"
            )
            reference_identities = {
                str(index): _all_rank_exact(
                    latent, label=f"{cell.sample_id}_source_ref_{index}"
                )
                for index, latent in reference_latents.items()
            }
            independent_frames_cpu = tuple(
                reference_latents[index]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .clone()
                for index in native.RV2V_REFERENCE_INDICES
            )
            frame_set_digest = source_frame_set_digest(independent_frames_cpu)
            expected_shape = tuple(int(item) for item in full_source_latent.shape)
            if tuple(expected_shape[:3]) != (1, 16, LATENT_PHASES):
                raise OASISNoiseBankError("source-derived target geometry is not exact81")
            source_conditioning_payload = {
                "sample_digest": cell.sample_digest,
                "source_video_sha256": cell.source_video_sha256,
                "edit_instruction_sha256": cell.edit_instruction_sha256,
                "source_instruction_binding_digest": (
                    cell.source_instruction_binding_digest
                ),
                "task_prompt_sha256": prompt_sha,
                "positive_input_ids_raw_value_sha256": positive_ids_identity[
                    "raw_storage_sha256"
                ],
                "positive_attention_mask_raw_value_sha256": positive_mask_identity[
                    "raw_storage_sha256"
                ],
                "bucket_hw": list(bucket_hw),
                "full_source_latent_raw_value_sha256": full_identity["identity"][
                    "raw_storage_sha256"
                ],
                "source_reference_raw_value_sha256_by_frame": {
                    index: value["identity"]["raw_storage_sha256"]
                    for index, value in reference_identities.items()
                },
                "source_frame_set_digest": frame_set_digest,
            }
            source_conditioning_digest = _object_sha(source_conditioning_payload)
            cell_runtime[cell.sample_id] = {
                "sample_id": cell.sample_id,
                "family": cell.family,
                "analysis_split": cell.analysis_split,
                "source_video_path": str(cell.source_video),
                "source_video_sha256": cell.source_video_sha256,
                "edit_instruction_sha256": cell.edit_instruction_sha256,
                "source_instruction_binding_digest": (
                    cell.source_instruction_binding_digest
                ),
                "sample_digest": cell.sample_digest,
                "source_conditioning": source_conditioning_payload,
                "source_conditioning_digest": source_conditioning_digest,
                "bucket_hw": list(bucket_hw),
                "source_frame_set_digest": frame_set_digest,
                "full_source_rank0_broadcast": full_broadcast,
                "source_refs_rank0_broadcast": reference_broadcasts,
                "full_source_all_rank_identity": full_identity,
                "source_ref_all_rank_identities": reference_identities,
                "source_frame_set_artifact_pending": True,
                "independent_frames_cpu": independent_frames_cpu,
            }
            vae.to("cpu")
            del source_tensor, source_pixels
            torch.cuda.empty_cache()
            model.to(device)
            condition_kwargs = {
                "image_vae_latents": None,
                "multi_video_vae_latents": [full_source_latent],
                "multi_image_vae_latents": [
                    reference_latents[index] for index in native.RV2V_REFERENCE_INDICES
                ],
            }
            for seed in manifest.seed_order:
                arm_captures: dict[str, NoiseInjectionCapture] = {}
                for arm in NOISE_ARM_ORDER:
                    carrier_seed = carrier_seed_for(
                        sample_digest=cell.sample_digest, seed=seed
                    )
                    with torch.inference_mode():
                        endpoint, capture = _sample_with_oasis_noise_arm(
                            sample_fn=lambda: model.sample(
                                input_ids=positive_ids.to(device),
                                attention_mask=positive_mask.to(device),
                                uncond_input_ids=negative_ids.to(device),
                                uncond_attention_mask=negative_mask.to(device),
                                **condition_kwargs,
                                width=bucket_hw[1],
                                height=bucket_hw[0],
                                device=device,
                                **_sampling_contract(seed),
                            ),
                            wan_diffusion_module=wan_diffusion,
                            noise_arm=arm,
                            independent_frame_latents_cpu=independent_frames_cpu,
                            expected_shape=expected_shape,
                            expected_device=device,
                            expected_seed=seed,
                            carrier_seed=carrier_seed,
                        )
                    if (
                        not isinstance(endpoint, torch.Tensor)
                        or endpoint.dtype != torch.float32
                        or endpoint.device != device
                        or endpoint.requires_grad
                        or tuple(int(item) for item in endpoint.shape) != expected_shape
                        or not bool(torch.isfinite(endpoint).all().item())
                    ):
                        raise OASISNoiseBankError("native endpoint tensor contract differs")
                    endpoint_cpu = endpoint.detach().to(device="cpu").contiguous()
                    _all_rank_exact(
                        endpoint_cpu,
                        label=f"{cell.sample_id}_{seed}_{arm}_endpoint",
                    )
                    _all_rank_exact(
                        capture.baseline_tensor,
                        label=f"{cell.sample_id}_{seed}_{arm}_parent_gaussian",
                    )
                    _all_rank_exact(
                        capture.injected_tensor,
                        label=f"{cell.sample_id}_{seed}_{arm}_sampler_noise",
                    )
                    generated[(cell.sample_id, seed, arm)] = endpoint_cpu
                    captures[(cell.sample_id, seed, arm)] = capture
                    arm_captures[arm] = capture
                parent_hashes = {
                    capture.baseline_raw_value_sha256 for capture in arm_captures.values()
                }
                if len(parent_hashes) != 1:
                    raise OASISNoiseBankError("matched arms do not share one official Gaussian")
                official = arm_captures["official_gaussian"]
                if (
                    official.external_initial_noise_injection
                    or not official.original_return_object_forwarded
                    or official.injected_raw_value_sha256
                    != official.baseline_raw_value_sha256
                ):
                    raise OASISNoiseBankError("rho0 control is not bit-exact native")
                for arm in NOISE_ARM_ORDER[1:]:
                    active = arm_captures[arm]
                    if (
                        not active.external_initial_noise_injection
                        or active.original_return_object_forwarded
                        or active.baseline_raw_value_sha256
                        != official.baseline_raw_value_sha256
                    ):
                        raise OASISNoiseBankError("active arm lost parent-official binding")
            del full_source_latent, reference_latents
            torch.cuda.empty_cache()
        freeze_after = native.source_audit.model_freeze_certificate(model)
        if freeze_after != freeze_before or any(parameter.requires_grad for parameter in model.parameters()):
            raise OASISNoiseBankError("frozen Bernini model changed during candidate bank")
        model.to("cpu")
        torch.cuda.empty_cache()
        checkpoint_after: list[Any] = [None]
        if distributed.rank == 0:
            try:
                checkpoint_after[0] = {
                    "ok": True,
                    "identity": native.source_audit.validate_checkpoint_content(
                        checkpoint, checkpoint_manifest
                    ),
                }
            except Exception as error:
                checkpoint_after[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(checkpoint_after, src=0)
        if (
            not isinstance(checkpoint_after[0], Mapping)
            or checkpoint_after[0].get("ok") is not True
            or checkpoint_after[0].get("identity") != checkpoint_identity
        ):
            raise OASISNoiseBankError("checkpoint changed during frozen bank")

        if distributed.rank == 0:
            manifest.assert_unchanged()
            staging = _output_staging_directory(output_dir)
            rollout_rows: list[Mapping[str, Any]] = []
            cell_rows: list[Mapping[str, Any]] = []
            matched_triplet_audits: list[Mapping[str, Any]] = []
            for cell in cells:
                runtime = dict(cell_runtime[cell.sample_id])
                frames = runtime.pop("independent_frames_cpu")
                runtime.pop("source_frame_set_artifact_pending")
                cell_root = staging / cell.sample_id
                cell_root.mkdir()
                frame_artifact = _save_source_frame_set(
                    cell_root / "source-frame-set.safetensors", frames
                )
                runtime["source_frame_set_artifact"] = frame_artifact
                cell_rows.append(runtime)
                for seed in manifest.seed_order:
                    triplet_start = len(rollout_rows)
                    seed_root = cell_root / f"seed-{seed}"
                    seed_root.mkdir()
                    generated_for_decode = {
                        arm: generated[(cell.sample_id, seed, arm)]
                        .to(device=device)
                        .contiguous()
                        for arm in NOISE_ARM_ORDER
                    }
                    try:
                        outputs = native._save_outputs(
                            output_dir=seed_root,
                            generated=generated_for_decode,
                            vae=vae,
                            bucket_hw=runtime["bucket_hw"],
                            device=device,
                            save_output_fn=save_output,
                        )
                    finally:
                        generated_for_decode.clear()
                    official_parent = captures[
                        (cell.sample_id, seed, "official_gaussian")
                    ].baseline_raw_value_sha256
                    for arm in NOISE_ARM_ORDER:
                        capture = captures[(cell.sample_id, seed, arm)]
                        baseline_artifact = _save_tensor(
                            seed_root / f"{arm}.parent-official-gaussian.safetensors",
                            capture.baseline_tensor,
                            key="parent_official_gaussian",
                            metadata={
                                "coordinate": "bernini_native_target_latent_before_rearrange",
                                "role": "official_parent_gaussian_before_oasis_injection",
                                "external_initial_noise_injection": "false",
                            },
                        )
                        sampler_noise_artifact = _save_tensor(
                            seed_root / f"{arm}.sampler-initial-noise.safetensors",
                            capture.injected_tensor,
                            key="sampler_initial_noise",
                            metadata={
                                "coordinate": "bernini_native_target_latent_before_rearrange",
                                "noise_arm": arm,
                                "source_carrier_rho": str(capture.rho),
                                "external_initial_noise_injection": str(
                                    capture.external_initial_noise_injection
                                ).lower(),
                            },
                        )
                        unsigned = {
                            "schema_version": ROLLOUT_SCHEMA,
                            "candidate_id": oasis_core.candidate_id_for(
                                sample_id=cell.sample_id,
                                seed=seed,
                                noise_arm=arm,
                            ),
                            "sample_id": cell.sample_id,
                            "sample_digest": cell.sample_digest,
                            "source_video_sha256": cell.source_video_sha256,
                            "edit_instruction_sha256": (
                                cell.edit_instruction_sha256
                            ),
                            "source_instruction_binding_digest": (
                                cell.source_instruction_binding_digest
                            ),
                            "source_conditioning_digest": runtime[
                                "source_conditioning_digest"
                            ],
                            "family": cell.family,
                            "analysis_split": cell.analysis_split,
                            "seed": seed,
                            "noise_arm": arm,
                            "source_carrier_rho": capture.rho,
                            "carrier_seed": carrier_seed_for(
                                sample_digest=cell.sample_digest, seed=seed
                            ),
                            "source_frame_set_digest": capture.source_frame_set_digest,
                            "source_frame_order_consumed": False,
                            "full_video_latent_consumed_by_carrier": False,
                            "operator_receipt": dict(capture.operator_receipt),
                            "operator_receipt_digest": capture.operator_receipt_digest,
                            "operator_runtime_binding": {
                                "callable": NOISE_OPERATOR_CALLABLE,
                                "integration_owner": (
                                    "infer_oasis_phase_a_noise_bank."
                                    "_sample_with_oasis_noise_arm"
                                ),
                                "official_randn_called_first": True,
                                "inference_integration_executed": True,
                                "operator_self_registered_sampler_hook": False,
                            },
                            "parent_official_gaussian_raw_value_sha256": official_parent,
                            "baseline_artifact": baseline_artifact,
                            "sampler_initial_noise_artifact": sampler_noise_artifact,
                            "external_initial_noise_injection": capture.external_initial_noise_injection,
                            "rho_zero_exact_native_object_forwarded": (
                                arm == "official_gaussian"
                                and capture.original_return_object_forwarded
                                and capture.baseline_raw_value_sha256
                                == capture.injected_raw_value_sha256
                            ),
                            "active_noise_parent_matches_official_control": (
                                capture.baseline_raw_value_sha256 == official_parent
                            ),
                            "native_sampling": {
                                **_sampling_contract(seed),
                                "condition_mode": "rv2v4",
                                "guidance_policy": GUIDANCE_POLICY,
                                "guidance_implementation_replaced": False,
                                "sample_one_step_replaced": False,
                                "scheduler_replaced": False,
                                "temporary_mutation_surface": list(
                                    TEMPORARY_MUTATION_SURFACE
                                ),
                                "target_initialization": (
                                    native.TARGET_INITIALIZATION
                                    if arm == "official_gaussian"
                                    else "oasis_source_set_external_initial_noise_injection"
                                ),
                                "exact81": True,
                                "exact40": True,
                            },
                            "endpoint": outputs[arm],
                            "endpoint_candidate_only": True,
                            "legacy_pair_v5_native_rollout_schema_compatible": False,
                            "external_action_scorer_consumed": False,
                            "action_source_scoring_performed": False,
                            "endpoint_selection_performed": False,
                            "optimizer_or_training_authorized": False,
                            "training_performed": False,
                            "scientific_action_editing_success_claim": False,
                        }
                        rollout_rows.append(
                            {**unsigned, "rollout_digest": _object_sha(unsigned)}
                        )
                    try:
                        triplet_audit = oasis_core.validate_matched_rollout_triplet(
                            rollout_rows[triplet_start:],
                            sample_id=cell.sample_id,
                            sample_digest=cell.sample_digest,
                            source_video_sha256=cell.source_video_sha256,
                            edit_instruction_sha256=cell.edit_instruction_sha256,
                            source_conditioning_digest=runtime[
                                "source_conditioning_digest"
                            ],
                            source_frame_set_digest=runtime[
                                "source_frame_set_digest"
                            ],
                            family=cell.family,
                            analysis_split=cell.analysis_split,
                            seed=seed,
                        )
                    except oasis_core.OASISPhaseAError as error:
                        raise OASISNoiseBankError(str(error)) from error
                    matched_triplet_audits.append(
                        {
                            "sample_id": triplet_audit.sample_id,
                            "sample_digest": triplet_audit.sample_digest,
                            "seed": triplet_audit.seed,
                            "candidate_ids": list(triplet_audit.candidate_ids),
                            "audit_digest": triplet_audit.audit_digest,
                        }
                    )
            if len(rollout_rows) != 12:
                raise OASISNoiseBankError("family bank must contain exact12 rollouts")
            runtime_versions = {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
            }
            unsigned_receipt = {
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "manifest_path": str(manifest.path),
                "manifest_file_sha256": manifest.file_sha256,
                "manifest_digest": manifest.manifest_digest,
                "family": args.family,
                "analysis_splits": list(oasis_manifest.SPLIT_ORDER),
                "seed_order": list(manifest.seed_order),
                "noise_arm_order": list(NOISE_ARM_ORDER),
                "source_cell_count": len(cell_rows),
                "rollout_count": len(rollout_rows),
                "topology": {
                    "world_size": WORLD_SIZE,
                    "ulysses_sequence_parallel_size": SP_SIZE,
                    "all_four_ranks_reported_bit_exact_tensors": True,
                },
                "source_cells": cell_rows,
                "rollouts": rollout_rows,
                "matched_triplet_audits": matched_triplet_audits,
                "method_source_revision": source_revision,
                "method_source_archive_sha256": source_archive_sha,
                "bernini_revision": bernini_revision,
                "veomni_revision": veomni_revision,
                "bernini_inference_files": inference_hashes,
                "checkpoint": {
                    "path": str(checkpoint),
                    "tree_sha256": manifest.checkpoint_tree_sha256,
                    "content_manifest_path": str(checkpoint_manifest),
                    "content_manifest_file_sha256": (
                        args.expected_checkpoint_content_manifest_sha256
                    ),
                    "content_before_and_after": checkpoint_identity,
                    "frozen_unchanged": True,
                },
                "runtime_versions": runtime_versions,
                "information_flow": {
                    "external_media_inputs_source_video_only": True,
                    "text_input": "complete_action_caption",
                    "source_frame_order_consumed_by_carrier": False,
                    "full_video_latent_consumed_by_carrier": False,
                    "target_mask_flow_pose_track_trajectory_consumed": False,
                    "proposal_media_latent_or_motion_donor_consumed": False,
                    "external_action_scorer_or_calibration_consumed": False,
                    "active_rho_is_external_initial_noise_injection": True,
                    "rho_zero_is_bit_exact_native_control": True,
                },
                "composable_sampling_boundary": {
                    "temporary_mutation_surface": list(TEMPORARY_MUTATION_SURFACE),
                    "official_randn_tensor_always_called_first": True,
                    "sample_method_replaced": False,
                    "sample_one_step_replaced": False,
                    "native_cfg_or_apg_replaced": False,
                    "native_scheduler_replaced": False,
                    "guidance_policy": GUIDANCE_POLICY,
                    "future_guidance_ablation_composes_outside_this_runner": True,
                },
                "noise_operator_contract": {
                    "runtime_callable": NOISE_OPERATOR_CALLABLE,
                    "same_callable_required_for_any_future_training": True,
                    "alternate_training_noise_builder_authorized": False,
                    "operator_self_registers_sampler_hook": False,
                    "operator_self_registers_launcher": False,
                    "dedicated_frozen_inference_integration_executed": True,
                    "training_integration_executed": False,
                },
                "legacy_pair_v5_native_bank_authorized": False,
                "endpoint_action_source_scoring_authorized": False,
                "endpoint_selection_performed": False,
                "external_action_scorer_consumed": False,
                "optimizer_or_training_authorized": False,
                "training_performed": False,
                "scientific_action_editing_success_claim": False,
            }
            rebased = _rebase_artifact_paths(
                unsigned_receipt, old_root=staging, new_root=output_dir
            )
            receipt = {**rebased, "receipt_digest": _object_sha(rebased)}
            _write_receipt(staging / "receipt.json", receipt)
            _commit_output_transaction(staging=staging, final=output_dir)
            print(_canonical(receipt).decode("ascii"), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METHOD",
    "NOISE_ARM_ORDER",
    "NOISE_RHO_BY_ARM",
    "NoiseInjectionCapture",
    "OASISNoiseBankError",
    "ROLLOUT_SCHEMA",
    "SCHEMA_VERSION",
    "carrier_seed_for",
    "main",
    "source_frame_set_digest",
]

#!/usr/bin/env python3
"""Materialize one source-only Bernini/Wan condition latent.

The executable accepts exactly one raw source MP4.  It decodes all 81 integer
frames, applies the same aspect bucket used by the Bernini exact81 pipeline,
and encodes the pixels with the pinned checkpoint's Wan VAE.  The VAE
posterior *mode* is normalized with the checkpoint's 16-channel mean/std and
stored as FP32 safetensors before any decode.

There is deliberately no parquet, target, edited-video, paired-posterior, or
directory-discovery input.  Target media, target paths, target columns, and
target posterior parameters are neither exposed by the CLI nor accessed by
the implementation.  The receipt records this closed input surface; it does
not authorize training or a scientific source-reward claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as legacy  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402


METHOD_NAME = "bernini-dclr-source-only-condition-materializer-v1"
# This value is consumed verbatim by infer_dclr_reward_runtime_smoke.py.
RECEIPT_SCHEMA = "bernini-source-only-vae-materialization-v1"
ARTIFACT_NAME = "source.normalized-clean-latent.safetensors"
RECEIPT_NAME = "receipt.json"
FRAME_COUNT = 81
FPS = 25.0
LATENT_PHASES = 21
VAE_CHANNELS = 16
SPATIAL_COMPRESSION = 8
EXPECTED_DIFFUSERS_VERSION = "0.38.0"
EXPECTED_CHECKPOINT_TREE_SHA256 = legacy.trainer.CHECKPOINT_TREE_SHA256
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
)
EXPECTED_CHECKPOINT_FILE_COUNT = source_audit.CHECKPOINT_CONTENT_FILE_COUNT
EXPECTED_VAE_CONFIG_SHA256 = (
    "f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527"
)
ARTIFACT_METADATA = {
    "coordinate": "bernini_normalized_clean_vae_latent",
    "frame_contract": "exact81_latent21",
    "artifact_role": "source_video_condition",
    "source": "source_video_vae_encode_before_any_decode",
}
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class SourceConditionMaterializationError(RuntimeError):
    """Raised before ambiguous source-only evidence can be emitted."""


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if _SHA256_RE.fullmatch(text) is None:
        raise SourceConditionMaterializationError(
            f"{label} must be a lowercase SHA-256"
        )
    return text


def _require_sha1(value: Any, *, label: str) -> str:
    text = str(value).lower()
    if _SHA1_RE.fullmatch(text) is None:
        raise SourceConditionMaterializationError(
            f"{label} must be a full SHA-1"
        )
    return text


def _plain_absolute_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SourceConditionMaterializationError(f"{label} must be absolute")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise SourceConditionMaterializationError(
            f"{label} is unavailable: {error}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise SourceConditionMaterializationError(
            f"{label} must be a plain file"
        )
    return path


def _fresh_output_path(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise SourceConditionMaterializationError("output-dir must be absolute")
    if requested.name in ("", ".", "..") or _IID_RE.fullmatch(requested.name) is None:
        raise SourceConditionMaterializationError("output-dir basename is unsafe")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise SourceConditionMaterializationError(
            f"output parent is unavailable: {error}"
        ) from error
    path = parent / requested.name
    if path.exists() or path.is_symlink():
        raise SourceConditionMaterializationError(
            "output-dir must be fresh and absent"
        )
    if not parent.is_dir() or parent.is_symlink():
        raise SourceConditionMaterializationError(
            "output parent must be a plain directory"
        )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode one exact81 raw source MP4 into a Wan source latent"
    )
    parser.add_argument("--iid", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument(
        "--expected-bucket-hw", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), required=True
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=EXPECTED_CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=EXPECTED_CHECKPOINT_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-vae-config-sha256", default=EXPECTED_VAE_CONFIG_SHA256
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(args.iid, str) or _IID_RE.fullmatch(args.iid) is None:
        raise SourceConditionMaterializationError("iid is not a safe identifier")
    source = _plain_absolute_file(args.source_video, label="source-video")
    if source.suffix.lower() != ".mp4":
        raise SourceConditionMaterializationError("source-video must be one MP4")
    source_sha = _require_sha256(
        args.expected_source_sha256, label="expected source SHA-256"
    )
    if legacy.file_sha256(source) != source_sha:
        raise SourceConditionMaterializationError("source-video SHA-256 differs")
    if (
        not isinstance(args.expected_bucket_hw, list)
        and not isinstance(args.expected_bucket_hw, tuple)
    ) or len(args.expected_bucket_hw) != 2:
        raise SourceConditionMaterializationError("expected-bucket-hw must be H W")
    bucket = tuple(args.expected_bucket_hw)
    if any(
        isinstance(value, bool)
        or type(value) is not int
        or value <= 0
        or value % legacy.SPATIAL_STRIDE
        for value in bucket
    ):
        raise SourceConditionMaterializationError(
            "expected bucket dimensions must be positive multiples of 16"
        )
    if bucket[0] * bucket[1] > legacy.MAX_PIXELS:
        raise SourceConditionMaterializationError(
            "expected bucket exceeds the pinned Bernini pixel budget"
        )
    checkpoint_tree_sha = _require_sha256(
        args.expected_checkpoint_tree_sha256,
        label="expected checkpoint tree SHA-256",
    )
    manifest_sha = _require_sha256(
        args.expected_checkpoint_content_manifest_sha256,
        label="expected checkpoint manifest SHA-256",
    )
    vae_config_sha = _require_sha256(
        args.expected_vae_config_sha256,
        label="expected VAE config SHA-256",
    )
    if checkpoint_tree_sha != EXPECTED_CHECKPOINT_TREE_SHA256:
        raise SourceConditionMaterializationError(
            "checkpoint tree differs from pinned Bernini-R"
        )
    if manifest_sha != EXPECTED_CHECKPOINT_MANIFEST_SHA256:
        raise SourceConditionMaterializationError(
            "checkpoint content manifest differs from pinned Bernini-R"
        )
    if vae_config_sha != EXPECTED_VAE_CONFIG_SHA256:
        raise SourceConditionMaterializationError(
            "VAE config differs from pinned Wan VAE"
        )
    manifest = _plain_absolute_file(
        args.checkpoint_content_manifest,
        label="checkpoint-content-manifest",
    )
    if legacy.file_sha256(manifest) != manifest_sha:
        raise SourceConditionMaterializationError(
            "checkpoint content manifest SHA-256 differs"
        )
    method_revision = _require_sha1(
        args.method_source_revision, label="method source revision"
    )
    method_archive_sha = _require_sha256(
        args.method_source_archive_sha256,
        label="method source archive SHA-256",
    )
    output = _fresh_output_path(args.output_dir)
    return {
        "iid": args.iid,
        "source": source,
        "source_sha256": source_sha,
        "expected_bucket_hw": bucket,
        "checkpoint_manifest": manifest,
        "checkpoint_tree_sha256": checkpoint_tree_sha,
        "checkpoint_manifest_sha256": manifest_sha,
        "vae_config_sha256": vae_config_sha,
        "method_source_revision": method_revision,
        "method_source_archive_sha256": method_archive_sha,
        "output": output,
    }


def validate_vae_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceConditionMaterializationError("Wan VAE config must be an object")
    mean = value.get("latents_mean")
    std = value.get("latents_std")
    if (
        value.get("_class_name") != "AutoencoderKLWan"
        or value.get("z_dim") != VAE_CHANNELS
        or value.get("temperal_downsample") != [False, True, True]
        or not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != VAE_CHANNELS
        or len(std) != VAE_CHANNELS
    ):
        raise SourceConditionMaterializationError(
            "checkpoint does not expose the pinned Wan VAE geometry"
        )
    try:
        mean_values = tuple(float(item) for item in mean)
        std_values = tuple(float(item) for item in std)
    except (TypeError, ValueError, OverflowError) as error:
        raise SourceConditionMaterializationError(
            "Wan VAE normalization statistics are not numeric"
        ) from error
    if any(not math.isfinite(item) for item in mean_values + std_values) or any(
        item <= 0.0 for item in std_values
    ):
        raise SourceConditionMaterializationError(
            "Wan VAE normalization statistics are invalid"
        )
    return {
        "class_name": "AutoencoderKLWan",
        "z_dim": VAE_CHANNELS,
        "temperal_downsample": [False, True, True],
        "latents_mean": list(mean_values),
        "latents_std": list(std_values),
    }


def normalized_wan_vae_mode(vae: Any, source_pixels: Any) -> Any:
    """Return exact normalized posterior mode in FP32 source coordinates."""

    import torch

    if (
        not isinstance(source_pixels, torch.Tensor)
        or source_pixels.dtype != torch.float32
        or source_pixels.ndim != 5
        or tuple(int(item) for item in source_pixels.shape[:3])
        != (1, 3, FRAME_COUNT)
        or source_pixels.requires_grad
        or not bool(torch.isfinite(source_pixels).all().item())
    ):
        raise SourceConditionMaterializationError(
            "source pixels must be finite detached FP32 [1,3,81,H,W]"
        )
    config = validate_vae_config(dict(vae.config))
    encoded = vae.encode(source_pixels)
    distribution = getattr(encoded, "latent_dist", None)
    mode_fn = getattr(distribution, "mode", None)
    if not callable(mode_fn):
        raise SourceConditionMaterializationError(
            "Wan VAE encode did not return a posterior distribution"
        )
    latent = mode_fn()
    if (
        not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float32
        or latent.ndim != 5
        or tuple(int(item) for item in latent.shape[:3])
        != (1, VAE_CHANNELS, LATENT_PHASES)
        or latent.requires_grad
        or not bool(torch.isfinite(latent).all().item())
    ):
        raise SourceConditionMaterializationError(
            "source posterior mode must be finite FP32 [1,16,21,H,W]"
        )
    mean = torch.tensor(
        config["latents_mean"], dtype=latent.dtype, device=latent.device
    ).view(1, VAE_CHANNELS, 1, 1, 1)
    std = torch.tensor(
        config["latents_std"], dtype=latent.dtype, device=latent.device
    ).view(1, VAE_CHANNELS, 1, 1, 1)
    normalized = ((latent - mean) / std).detach().contiguous()
    if normalized.dtype != torch.float32 or not bool(
        torch.isfinite(normalized).all().item()
    ):
        raise SourceConditionMaterializationError(
            "normalized source latent must remain finite FP32"
        )
    return normalized


def _tensor_identity(value: Any) -> dict[str, Any]:
    import torch

    if not isinstance(value, torch.Tensor):
        raise SourceConditionMaterializationError("artifact tensor is absent")
    cpu = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    raw = bytes(cpu.untyped_storage())
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "byte_count": len(raw),
    }
    if metadata["byte_count"] != metadata["numel"] * cpu.element_size():
        raise SourceConditionMaterializationError("tensor storage byte count differs")
    payload = legacy.canonical_json_bytes(metadata) + b"\0" + raw
    metadata.update(
        {
            "content_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_storage_sha256": hashlib.sha256(raw).hexdigest(),
            "finite": bool(torch.isfinite(cpu).all().item()),
        }
    )
    return metadata


def save_source_artifact(path: Path, latent: Any) -> dict[str, Any]:
    """Atomically persist and byte-verify the runtime-compatible artifact."""

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if path.exists() or path.is_symlink() or path.name != ARTIFACT_NAME:
        raise SourceConditionMaterializationError(
            "source artifact path must be fresh and canonical"
        )
    stored = latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if (
        tuple(int(item) for item in stored.shape[:3])
        != (1, VAE_CHANNELS, LATENT_PHASES)
        or stored.ndim != 5
        or not bool(torch.isfinite(stored).all().item())
    ):
        raise SourceConditionMaterializationError(
            "source artifact must be finite FP32 [1,16,21,H,W]"
        )
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".safetensors",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(
            {"normalized_clean_latent": stored},
            str(temporary),
            metadata=ARTIFACT_METADATA,
        )
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != ["normalized_clean_latent"]:
                raise SourceConditionMaterializationError(
                    "source artifact tensor key differs"
                )
            restored = opened.get_tensor("normalized_clean_latent").contiguous()
            metadata = dict(opened.metadata() or {})
        if metadata != ARTIFACT_METADATA or not torch.equal(restored, stored):
            raise SourceConditionMaterializationError(
                "source artifact safetensors round trip differs"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": legacy.file_sha256(path),
        "tensor_key": "normalized_clean_latent",
        "shape": [int(item) for item in stored.shape],
        "artifact_role": ARTIFACT_METADATA["artifact_role"],
        "coordinate": ARTIFACT_METADATA["coordinate"],
        "frame_contract": ARTIFACT_METADATA["frame_contract"],
        "metadata": dict(ARTIFACT_METADATA),
        "tensor_identity": _tensor_identity(stored),
        "stored_dtype": "torch.float32",
        "source_video_vae_encode_before_any_decode": True,
        "mp4_decode_reencode_used": False,
        "roundtrip_tensor_exact": True,
    }


def build_receipt(
    *,
    contract: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    checkpoint: Path,
    checkpoint_identity: Mapping[str, Any],
    vae_config_identity: Mapping[str, Any],
    artifact: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    bucket = list(contract["expected_bucket_hw"])
    expected_shape = [
        1,
        VAE_CHANNELS,
        LATENT_PHASES,
        bucket[0] // SPATIAL_COMPRESSION,
        bucket[1] // SPATIAL_COMPRESSION,
    ]
    tensor_identity = artifact.get("tensor_identity")
    if (
        not isinstance(tensor_identity, Mapping)
        or tensor_identity.get("shape") != expected_shape
        or tensor_identity.get("dtype") != "torch.float32"
        or tensor_identity.get("finite") is not True
    ):
        raise SourceConditionMaterializationError(
            "artifact geometry differs from source-derived bucket"
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "method_source_revision": contract["method_source_revision"],
        "method_source_archive_sha256": contract[
            "method_source_archive_sha256"
        ],
        "source_only": True,
        "input": {
            "source_iid": contract["iid"],
            "source_video_path": str(contract["source"]),
            "source_video_sha256": contract["source_sha256"],
            "input_kind": "raw_source_mp4",
            "media": dict(source_metadata),
        },
        "access_audit": {
            # These names are a schema-level declaration of the only logical
            # source fields supplied by the raw-MP4 CLI.  No parquet is read.
            "source_columns_accessed": [
                "iid",
                "source_video",
                "source_video_sha256",
            ],
            "target_columns_accessed": [],
            "target_media_accessed": False,
            "paired_target_accessed": False,
            "source_video_argument_exposed": True,
            "source_path_accessed": str(contract["source"]),
            "source_media_decoded": True,
            "source_vae_posterior_mode_accessed": True,
            "source_directory_enumerated": False,
            "parquet_argument_exposed": False,
            "parquet_path_accessed": None,
            "parquet_columns_accessed": [],
            "target_path_argument_exposed": False,
            "target_path_accessed": None,
            "target_vae_posterior_accessed": False,
            "edited_sibling_path_inferred_or_accessed": False,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "tree_sha256": contract["checkpoint_tree_sha256"],
            "content": dict(checkpoint_identity),
            "vae_config_sha256": contract["vae_config_sha256"],
            "vae_config": dict(vae_config_identity),
        },
        "preprocessing": {
            "num_frames": FRAME_COUNT,
            "fps": FPS,
            "expected_bucket_hw": bucket,
            "all_integer_frames_0_through_80": True,
            "temporal_subsampling": False,
            "spatial_policy": (
                "sqrt_max_pixels_then_floor_each_dimension_to_stride"
            ),
        },
        "encoding": {
            "vae_class": "AutoencoderKLWan",
            "posterior_statistic": "mode",
            "normalization": "(posterior_mode-latents_mean)/latents_std",
            "latent_coordinate": "bernini_normalized_clean_vae_latent",
            "latent_shape": expected_shape,
            "latent_dtype": "torch.float32",
            "before_any_vae_decode": True,
            "target_posterior_used": False,
        },
        "source_condition_artifact": dict(artifact),
        "runtime": dict(runtime),
        "single_gpu": True,
        "distributed": False,
        "training_performed": False,
        "optimizer_constructed": False,
        "engineering_materialization_only": True,
        "training_pair_authorized": False,
        "source_reward_calibration_authorized": False,
        "scientific_claim_authorized": False,
        "production_claim_forbidden": True,
    }
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def _write_receipt_atomically(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink() or path.name != RECEIPT_NAME:
        raise SourceConditionMaterializationError(
            "receipt path must be fresh and canonical"
        )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise SourceConditionMaterializationError("stale receipt temporary exists")
    payload = legacy.canonical_json_bytes(dict(receipt)) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    contract = validate_cli(args)
    checkpoint_value = Path(args.checkpoint).expanduser()
    try:
        checkpoint, _ = legacy.trainer.validate_checkpoint(checkpoint_value)
        checkpoint_identity = source_audit.validate_checkpoint_content(
            checkpoint,
            contract["checkpoint_manifest"],
            expected_manifest_sha256=contract["checkpoint_manifest_sha256"],
            expected_file_count=EXPECTED_CHECKPOINT_FILE_COUNT,
        )
    except Exception as error:
        raise SourceConditionMaterializationError(
            f"checkpoint validation failed: {error}"
        ) from error
    vae_config_path = checkpoint / "vae/config.json"
    if legacy.file_sha256(vae_config_path) != contract["vae_config_sha256"]:
        raise SourceConditionMaterializationError("pinned VAE config hash differs")
    vae_config_identity = validate_vae_config(
        legacy._read_json(vae_config_path, label="Wan VAE config")
    )

    import torch
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan

    if diffusers_version != EXPECTED_DIFFUSERS_VERSION:
        raise SourceConditionMaterializationError(
            f"diffusers version differs: {diffusers_version}"
        )
    if (
        not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
        or torch.cuda.device_count() != 1
    ):
        raise SourceConditionMaterializationError(
            "materializer requires exactly one visible AUH ROCm GPU"
        )
    world_size = os.environ.get("WORLD_SIZE", "1")
    if world_size != "1":
        raise SourceConditionMaterializationError(
            "source materialization is single-process, not distributed"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    source_pixels, source_metadata, source_sha256 = (
        source_audit.prepare_hashed_source_snapshot(contract["source"])
    )
    if source_sha256 != contract["source_sha256"]:
        raise SourceConditionMaterializationError(
            "source changed before private snapshot decode"
        )
    actual_bucket = tuple(source_metadata.get("source_derived_bucket_hw", ()))
    if actual_bucket != contract["expected_bucket_hw"]:
        raise SourceConditionMaterializationError(
            f"source-derived bucket differs: {actual_bucket}"
        )

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    if vae.training or any(parameter.requires_grad for parameter in vae.parameters()):
        raise SourceConditionMaterializationError("Wan VAE is not frozen in eval mode")
    vae.to(device)
    source_gpu = source_pixels.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        normalized = normalized_wan_vae_mode(vae, source_gpu)
    expected_shape = (
        1,
        VAE_CHANNELS,
        LATENT_PHASES,
        actual_bucket[0] // SPATIAL_COMPRESSION,
        actual_bucket[1] // SPATIAL_COMPRESSION,
    )
    if tuple(int(item) for item in normalized.shape) != expected_shape:
        raise SourceConditionMaterializationError(
            f"source latent shape differs: {tuple(normalized.shape)}"
        )
    normalized_cpu = normalized.to(device="cpu", dtype=torch.float32).contiguous()
    del normalized, source_gpu
    vae.to("cpu")
    del vae
    torch.cuda.empty_cache()

    output: Path = contract["output"]
    output.mkdir(mode=0o750, parents=False, exist_ok=False)
    artifact = save_source_artifact(output / ARTIFACT_NAME, normalized_cpu)
    receipt = build_receipt(
        contract=contract,
        source_metadata=source_metadata,
        checkpoint=checkpoint,
        checkpoint_identity=checkpoint_identity,
        vae_config_identity=vae_config_identity,
        artifact=artifact,
        runtime={
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
            "diffusers": diffusers_version,
            "device_count": torch.cuda.device_count(),
            "visible_rocr_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
    )
    _write_receipt_atomically(output / RECEIPT_NAME, receipt)
    print(legacy.canonical_json_bytes(receipt).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

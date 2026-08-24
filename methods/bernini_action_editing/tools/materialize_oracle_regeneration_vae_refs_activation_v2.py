#!/usr/bin/env python3
"""Materialize source-only VAE/ref provenance for activation-v2 authority.

This is an authoring tool, not a sampler.  On one WORLD4 allocation, rank zero
alone decodes the bound source, loads the VAE, encodes the full 81-frame source,
and independently calls the official VAE encoder once for each RGB frame at
indices 0/27/53/80.  The five FP32 latents are then broadcast byte-exactly.
No transformer, scheduler, target, anchor tensor, optimizer, or training code is
loaded.  The output receipt remains diagnostic material until a separate model
review and a later activation core compile its enclosing packet SHA.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Optional


METHOD_ROOT = Path(__file__).resolve().parents[1]
while str(METHOD_ROOT) in sys.path:
    sys.path.remove(str(METHOD_ROOT))
sys.path.insert(0, str(METHOD_ROOT))
_existing_activation = sys.modules.get("oracle_regeneration_activation_v2")
if _existing_activation is not None and Path(
    str(getattr(_existing_activation, "__file__", ""))
).resolve(strict=True) != (METHOD_ROOT / "oracle_regeneration_activation_v2.py").resolve(
    strict=True
):
    raise RuntimeError("preloaded activation-v2 origin differs")
_activation_spec = importlib.util.find_spec("oracle_regeneration_activation_v2")
if (
    _activation_spec is None
    or not isinstance(_activation_spec.origin, str)
    or Path(_activation_spec.origin).resolve(strict=True)
    != (METHOD_ROOT / "oracle_regeneration_activation_v2.py").resolve(strict=True)
):
    raise RuntimeError("activation-v2 import origin differs")

import oracle_regeneration_activation_v2 as activation  # noqa: E402


SCHEMA_VERSION = "bernini-oracle-regeneration-activation-v2-vae-authoring-run-v1"
WORLD_SIZE = 4
EXPECTED_PYTHON_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
EXPECTED_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)


class VaeReferenceMaterializationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_rgb_sha256(frame: Any) -> str:
    dtype = getattr(frame, "dtype", None)
    shape = tuple(getattr(frame, "shape", ()))
    if str(dtype) != "uint8" or shape[-1:] != (3,):
        raise VaeReferenceMaterializationError("decoded reference must be uint8 RGB")
    header = activation.safe_core.canonical_json_bytes_v1(
        {
            "schema_version": "decoded-uint8-rgb-frame-v1",
            "shape": [int(item) for item in frame.shape],
            "dtype": "uint8",
            "channel_order": "RGB",
        }
    )
    return hashlib.sha256(header + b"\x00" + frame.tobytes(order="C")).hexdigest()


def _tensor_identity(value: Any) -> Mapping[str, Any]:
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "content_sha256": activation.safe_core.tensor_content_sha256_v1(value),
    }


def _all_rank_sha(value: Any, *, dist: Any) -> list[str]:
    local = activation.safe_core.tensor_content_sha256_v1(value)
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, local)
    if any(row != local for row in rows):
        raise VaeReferenceMaterializationError("broadcast condition differs across WORLD4")
    return [str(row) for row in rows]


def _fresh_output_dir(value: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise VaeReferenceMaterializationError("output-dir must be absolute and non-root")
    parent = requested.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise VaeReferenceMaterializationError("output parent must be a plain directory")
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise VaeReferenceMaterializationError("refusing to overwrite output-dir")
    return output


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    if path.name in ("", ".", "..") or "/" in path.name:
        raise VaeReferenceMaterializationError("receipt basename differs")
    directory = path.parent.resolve(strict=True)
    if directory != path.parent or directory.is_symlink():
        raise VaeReferenceMaterializationError("receipt directory differs")
    temporary_name = f".{path.name}.tmp-pid-{os.getpid()}"
    payload = activation.safe_core.canonical_json_bytes_v1(value)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(str(directory), directory_flags)
    descriptor: Optional[int] = None
    published = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
        os.fsync(directory_fd)
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        check_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(check_fd)
            restored = b""
            while True:
                chunk = os.read(check_fd, 1024 * 1024)
                if not chunk:
                    break
                restored += chunk
            after = os.fstat(check_fd)
        finally:
            os.close(check_fd)
        if (
            restored != payload
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise VaeReferenceMaterializationError("published receipt identity differs")
    except FileExistsError as error:
        raise VaeReferenceMaterializationError(
            "receipt destination appeared; refusing overwrite"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", choices=activation.ALLOWED_CASES, required=True)
    parser.add_argument("--source-iid", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case_binding = activation.EXPECTED_CASE_BINDINGS[args.case_id]
    if args.source_iid != case_binding["source_iid"]:
        raise VaeReferenceMaterializationError("source iid differs from compiled case")
    source_path = activation._plain_absolute_file(args.source_video, label="source video")
    source_before, source_bytes = activation._seal_plain_file_v2(
        source_path, label="source video", retain_bytes=True
    )
    if source_before.sha256 != case_binding["source_sha256"] or source_bytes is None:
        raise VaeReferenceMaterializationError("source bytes differ from compiled case")
    output = _fresh_output_dir(args.output_dir)
    bernini_root = Path(args.bernini_root).expanduser().resolve(strict=True)
    veomni_root = Path(args.veomni_root).expanduser().resolve(strict=True)
    checkpoint_manifest = activation._plain_absolute_file(
        args.checkpoint_content_manifest, label="checkpoint content manifest"
    )
    tool_path = Path(__file__).resolve(strict=True)
    tool_sha = _sha256_file(tool_path)
    python_path = Path(sys.executable).resolve(strict=True)
    if (
        python_path != EXPECTED_PYTHON_PATH
        or python_path.is_symlink()
        or _sha256_file(python_path) != EXPECTED_PYTHON_SHA256
    ):
        raise VaeReferenceMaterializationError("Python runtime identity differs")
    activation.verify_frozen_dependency_pins_v2()

    import infer_native_identity_generation_canary as native
    import infer_source_kv_carrier_oracle as source_audit

    try:
        (
            checked_bernini_root,
            checked_veomni_root,
            bernini_revision,
            veomni_revision,
        ) = native.legacy.trainer.validate_source_trees(
            bernini_root,
            veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, _ = native.legacy.trainer.validate_checkpoint(args.checkpoint)
        inference_hashes = native.legacy.validate_inference_source_files(bernini_root)
        checkpoint_identity = source_audit.validate_checkpoint_content(
            checkpoint, checkpoint_manifest
        )
    except Exception as error:
        raise VaeReferenceMaterializationError(str(error)) from error
    pipeline_path = activation._plain_absolute_file(
        str(bernini_root / "bernini/pipeline.py"), label="Bernini VAE pipeline code"
    )
    vae_config_path = activation._plain_absolute_file(
        str(checkpoint / "vae/config.json"), label="VAE config"
    )
    decode_code_path = activation._plain_absolute_file(
        str(METHOD_ROOT / "tools/materialize_vae.py"), label="decode/resize code"
    )
    source_prepare_code_path = activation._plain_absolute_file(
        str(Path(native.legacy.__file__).resolve(strict=True)),
        label="source prepare code",
    )
    checkpoint_identity_sha = activation._canonical_object_sha256(
        checkpoint_identity
    )
    pre_model_pins = {
        "materializer_code_sha256": tool_sha,
        "pipeline_sha256": _sha256_file(pipeline_path),
        "vae_config_sha256": _sha256_file(vae_config_path),
        "decode_code_sha256": _sha256_file(decode_code_path),
        "source_prepare_code_sha256": _sha256_file(source_prepare_code_path),
        "checkpoint_manifest_sha256": _sha256_file(checkpoint_manifest),
    }
    native.legacy.trainer.activate_source_trees(
        checked_bernini_root, checked_veomni_root
    )

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from tools import materialize_vae

    autoencoder_module_path = activation._plain_absolute_file(
        str(Path(inspect.getfile(AutoencoderKLWan)).resolve(strict=True)),
        label="Diffusers AutoencoderKLWan implementation",
    )
    pre_model_pins = {
        **pre_model_pins,
        "autoencoder_module_sha256": _sha256_file(autoencoder_module_path),
        "python_executable_sha256": EXPECTED_PYTHON_SHA256,
    }

    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != WORLD_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise VaeReferenceMaterializationError("materializer requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=WORLD_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    latent_shape = activation.EXPECTED_LATENT_GEOMETRY[args.case_id]
    ref_shape = activation.EXPECTED_REFERENCE_GEOMETRY[args.case_id]
    bucket_hw = (latent_shape[-2] * 8, latent_shape[-1] * 8)

    rank_zero_metadata: list[Any] = [None]
    if distributed.rank == 0:
        try:
            with tempfile.TemporaryDirectory(prefix="activation-v2-source-snapshot-") as temp:
                snapshot = Path(temp) / "source.mp4"
                snapshot.write_bytes(source_bytes)
                if _sha256_file(snapshot) != source_before.sha256:
                    raise VaeReferenceMaterializationError("private source snapshot differs")
                raw_frames, reported_fps, input_hw = materialize_vae._decode_exact_video(
                    snapshot
                )
                source_pixels, metadata = native.legacy.prepare_exact_source(snapshot)
            if tuple(int(item) for item in source_pixels.shape) != (
                1,
                3,
                81,
                *bucket_hw,
            ):
                raise VaeReferenceMaterializationError("preprocessed source geometry differs")
            if (
                len(raw_frames) != 81
                or float(reported_fps) != 25.0
                or tuple(int(item) for item in input_hw)
                != activation.EXPECTED_SOURCE_INPUT_HW[args.case_id]
                or metadata.get("frame_count") != 81
                or float(metadata.get("fps", -1.0)) != 25.0
                or tuple(metadata.get("source_derived_bucket_hw", ())) != bucket_hw
                or tuple(metadata.get("source_input_hw", ()))
                != activation.EXPECTED_SOURCE_INPUT_HW[args.case_id]
            ):
                raise VaeReferenceMaterializationError(
                    "decoded source fps/frame/input geometry differs"
                )
            raw_rgb_sha = [
                _raw_rgb_sha256(raw_frames[index])
                for index in activation.REFERENCE_RGB_INDICES
            ]
            preprocessed_sha = [
                activation.safe_core.tensor_content_sha256_v1(
                    source_pixels[:, :, index : index + 1].contiguous()
                )
                for index in activation.REFERENCE_RGB_INDICES
            ]
            rank_zero_metadata[0] = {
                "ok": True,
                "reported_fps": float(reported_fps),
                "input_hw": [int(item) for item in input_hw],
                "source_metadata": dict(metadata),
                "raw_rgb_sha256": raw_rgb_sha,
                "preprocessed_sha256": preprocessed_sha,
                "full_preprocessed_identity": _tensor_identity(source_pixels),
            }
        except Exception as error:
            rank_zero_metadata[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(rank_zero_metadata, src=0)
    if not isinstance(rank_zero_metadata[0], Mapping) or rank_zero_metadata[0].get("ok") is not True:
        raise VaeReferenceMaterializationError(
            f"rank-zero source decode failed: {rank_zero_metadata[0]}"
        )

    encode_status: list[Any] = [None]
    if distributed.rank == 0:
        try:
            vae = AutoencoderKLWan.from_pretrained(
                str(checkpoint),
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            vae.eval().requires_grad_(False).to(device)
            pixels_device = source_pixels.to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                source_latent = _vae_encode(vae, pixels_device).float().contiguous()
                references = tuple(
                    _vae_encode(
                        vae, pixels_device[:, :, index : index + 1].contiguous()
                    ).float().contiguous()
                    for index in activation.REFERENCE_RGB_INDICES
                )
            del pixels_device, vae, source_pixels
            torch.cuda.empty_cache()
            encode_status[0] = {"ok": True}
        except Exception as error:
            encode_status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    else:
        source_latent = torch.empty(latent_shape, device=device, dtype=torch.float32)
        references = tuple(
            torch.empty(ref_shape, device=device, dtype=torch.float32) for _ in range(4)
        )
    dist.broadcast_object_list(encode_status, src=0)
    if not isinstance(encode_status[0], Mapping) or encode_status[0].get("ok") is not True:
        raise VaeReferenceMaterializationError(
            f"rank-zero VAE encode failed: {encode_status[0]}"
        )
    vae_load_rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        vae_load_rows,
        {"rank": distributed.rank, "vae_loaded": distributed.rank == 0},
    )
    if vae_load_rows != [
        {"rank": rank, "vae_loaded": rank == 0} for rank in range(WORLD_SIZE)
    ]:
        raise VaeReferenceMaterializationError("WORLD4 VAE load closure differs")
    dist.broadcast(source_latent, src=0)
    for value in references:
        dist.broadcast(value, src=0)
    tensors = (source_latent, *references)
    if (
        tuple(source_latent.shape) != latent_shape
        or source_latent.dtype != torch.float32
        or any(tuple(value.shape) != ref_shape or value.dtype != torch.float32 for value in references)
        or any(not value.is_contiguous() or not bool(torch.isfinite(value).all()) for value in tensors)
    ):
        raise VaeReferenceMaterializationError("broadcast VAE condition contract differs")
    try:
        activation.safe_core._require_pairwise_storage_disjoint_v1(tensors)
    except Exception as error:
        raise VaeReferenceMaterializationError(str(error)) from error
    source_identity = _tensor_identity(source_latent)
    reference_identities = tuple(_tensor_identity(value) for value in references)
    reference_sha = tuple(row["content_sha256"] for row in reference_identities)
    source_slice_sha = tuple(
        activation.safe_core.tensor_content_sha256_v1(
            source_latent[:, :, phase : phase + 1].contiguous()
        )
        for phase in activation.REFERENCE_LATENT_PHASES
    )
    if len(set(reference_sha)) != 4 or set(reference_sha).intersection(source_slice_sha):
        raise VaeReferenceMaterializationError(
            "independent reference content is duplicate or equals a full-latent slice"
        )
    source_rank_sha = _all_rank_sha(source_latent, dist=dist)
    reference_rank_rows = [_all_rank_sha(value, dist=dist) for value in references]
    rank_major_refs = [
        [reference_rank_rows[position][rank] for position in range(4)]
        for rank in range(WORLD_SIZE)
    ]
    source_after, _ = activation._seal_plain_file_v2(
        source_path, label="source video after VAE encode", retain_bytes=False
    )
    if source_after != source_before:
        raise VaeReferenceMaterializationError("source media changed during materialization")
    if any(_sha256_file(Path(path)) != digest for path, digest in (
        (tool_path, tool_sha),
        (pipeline_path, pre_model_pins["pipeline_sha256"]),
        (vae_config_path, pre_model_pins["vae_config_sha256"]),
        (decode_code_path, pre_model_pins["decode_code_sha256"]),
        (
            source_prepare_code_path,
            pre_model_pins["source_prepare_code_sha256"],
        ),
        (checkpoint_manifest, pre_model_pins["checkpoint_manifest_sha256"]),
        (
            autoencoder_module_path,
            pre_model_pins["autoencoder_module_sha256"],
        ),
        (python_path, pre_model_pins["python_executable_sha256"]),
    )):
        raise VaeReferenceMaterializationError("materializer dependency changed during run")

    checkpoint_after_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_after_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_after_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_after_rows, src=0)
    checkpoint_after = checkpoint_after_rows[0]
    if (
        not isinstance(checkpoint_after, Mapping)
        or checkpoint_after.get("ok") is not True
        or activation._canonical_object_sha256(checkpoint_after.get("identity"))
        != checkpoint_identity_sha
    ):
        raise VaeReferenceMaterializationError(
            "checkpoint content changed during VAE materialization"
        )

    if distributed.rank == 0:
        output.mkdir(mode=0o755)
        metadata = rank_zero_metadata[0]
        raw_sha = list(metadata["raw_rgb_sha256"])
        pre_sha = list(metadata["preprocessed_sha256"])
        receipt = {
            "schema_version": activation.REFERENCE_RECEIPT_SCHEMA_VERSION,
            "case_id": args.case_id,
            "source_iid": args.source_iid,
            "source_video_sha256": source_before.sha256,
            "source_frame_count": 81,
            "source_fps_numerator": 25,
            "source_fps_denominator": 1,
            "source_input_frame_geometry": [*metadata["input_hw"], 3],
            "source_bucket_hw": list(bucket_hw),
            "reference_rgb_indices": list(activation.REFERENCE_RGB_INDICES),
            "reference_raw_rgb_sha256": raw_sha,
            "full_preprocessed_source_identity": dict(metadata["full_preprocessed_identity"]),
            "reference_preprocessed_rgb_sha256": pre_sha,
            "preprocess_contract": {
                "frame_decode_backend": "decord_cpu0_num_threads1_private_source_snapshot",
                "frame_decode_code_path": str(decode_code_path),
                "frame_decode_code_sha256": pre_model_pins["decode_code_sha256"],
                "source_prepare_code_path": str(source_prepare_code_path),
                "source_prepare_code_sha256": pre_model_pins[
                    "source_prepare_code_sha256"
                ],
                "rgb_dtype": "uint8",
                "rgb_channel_order": "RGB",
                "resize_policy": "torchvision_bicubic_antialias_true_source_aspect_bucket",
                "normalization": "uint8_div255_mul2_sub1_float32",
            },
            "vae_contract": {
                "checkpoint_content_manifest_path": str(checkpoint_manifest),
                "checkpoint_content_manifest_sha256": pre_model_pins[
                    "checkpoint_manifest_sha256"
                ],
                "checkpoint_content_identity_sha256": checkpoint_identity_sha,
                "config_path": str(vae_config_path),
                "config_sha256": pre_model_pins["vae_config_sha256"],
                "vae_code_path": str(pipeline_path),
                "vae_code_sha256": pre_model_pins["pipeline_sha256"],
                "autoencoder_class_module_path": str(autoencoder_module_path),
                "autoencoder_class_module_sha256": pre_model_pins[
                    "autoencoder_module_sha256"
                ],
                "diffusers_version": str(diffusers_version),
                "torch_version": str(torch.__version__),
                "python_executable_path": str(python_path),
                "python_executable_sha256": pre_model_pins[
                    "python_executable_sha256"
                ],
                "python_version": str(sys.version),
                "rocm_version": str(torch.version.hip),
                "encode_function": "bernini.pipeline._vae_encode",
                "encode_dtype": "torch.float32",
                "latent_coordinate": "official_bernini_vae_encode_output",
            },
            "full_source_latent_identity": source_identity,
            "reference_latent_identities": [
                {
                    "frame_index": frame_index,
                    "raw_rgb_sha256": raw_sha[position],
                    "preprocessed_rgb_sha256": pre_sha[position],
                    **reference_identities[position],
                    "independently_vae_encoded": True,
                }
                for position, frame_index in enumerate(activation.REFERENCE_RGB_INDICES)
            ],
            "materializer_code_path": str(tool_path),
            "materializer_code_sha256": tool_sha,
            "rank_world_receipt": {
                "world_size": WORLD_SIZE,
                "sequence_parallel_size": WORLD_SIZE,
                "rank0_only_vae_encode": True,
                "all_rank_vae_load_roles": vae_load_rows,
                "broadcast_exact": True,
                "all_rank_full_source_latent_sha256": source_rank_sha,
                "all_rank_reference_latent_sha256": rank_major_refs,
            },
            "references_encoded_as_four_independent_rgb_frames": True,
            "references_not_sliced_from_full_source_latent": True,
            "source_reference_storage_alias_rejected": True,
            "reference_content_duplicates_rejected": True,
            "target_video_or_latent_used": False,
            "self_generated_anchor_tensor_used": False,
            "materialization_checks_passed": True,
        }
        _write_receipt(output / "vae-reference-receipt.json", receipt)
        run_receipt = {
            "schema_version": SCHEMA_VERSION,
            "receipt_sha256": _sha256_file(output / "vae-reference-receipt.json"),
            "source_tree": {
                "bernini_root": str(checked_bernini_root),
                "veomni_root": str(checked_veomni_root),
                "bernini_revision": str(bernini_revision),
                "veomni_revision": str(veomni_revision),
            },
            "inference_source_hashes": inference_hashes,
            "checkpoint_identity": checkpoint_identity,
            "full_model_or_sampler_loaded": False,
            "transformer_loaded": False,
            "scheduler_loaded": False,
            "training": False,
            "optimizer": False,
            "diagnostic_authoring_material_only": True,
        }
        _write_receipt(output / "run-receipt.json", run_receipt)
        output.chmod(0o555)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

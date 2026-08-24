#!/usr/bin/env python3
"""Decode native and step-0/20/40/60/80 packed-preservation RV2V sentinels.

One WORLD4/SP4 process loads the official Bernini-R 1.3B renderer once, uses
the four preregistered real source-only-v3 heldout sentinels, and decodes the
same complete forward instruction and seed for native plus every exact80
checkpoint.  Checkpoints are loaded strictly into the same PEFT architecture
and typed patch components used by training.  The output is an unranked media
receipt for manual review; there is no score, reward, VLM, or selection path.
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
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

SCHEMA_VERSION = "bernini-packed-preservation-checkpoint-decode-shard-v2"
METHOD = "bernini-packed-preservation-checkpoint-review-v2"
SENTINEL_ORDER = (
    "animal-dog-pick",
    "human-runner-jump",
    "hand-object-blueprint-roll",
    "emitter-fireworks-explode",
)
LORA_SCOPES = ("all-attention", "self-attention")
FORMAL_CHECKPOINT_STEPS = (0, 20, 40, 60, 80)
FRAME_COUNT = 81
FPS = 25
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
PYAV_VERSION = "13.1.0"
IMAGEIO_FFMPEG_SHA256 = (
    "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
)
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")


class PackedPreservationDecodeError(RuntimeError):
    """Raised before partial or semantically ambiguous media is published."""


def fail(message: str) -> NoReturn:
    raise PackedPreservationDecodeError(message)


def _sha(value: Any, *, label: str, length: int = 64) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        fail(f"{label} must be lowercase {'SHA-1' if length == 40 else 'SHA-256'}")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PackedPreservationDecodeError(f"{label} is unavailable") from error
    if resolved != path or not path.is_file() or path.is_symlink():
        fail(f"{label} must be one canonical plain file")
    return path


def _fresh_output(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or _SAFE.fullmatch(path.name) is None
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
    ):
        fail("output-dir must be one fresh safe absolute child")
    return path


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PackedPreservationDecodeError(f"{label} is unavailable") from error
    if resolved != path or not path.is_dir() or path.is_symlink():
        fail(f"{label} must be one canonical plain directory")
    return path


@contextmanager
def _serialized_adapter_load() -> Iterator[None]:
    raw = os.environ.get("PACKED_PRESERVATION_REVIEW_LOAD_LOCK")
    if not raw:
        fail("PACKED_PRESERVATION_REVIEW_LOAD_LOCK is required")
    path = _plain_file(raw, label="checkpoint load lock")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--expected-review-manifest-sha256", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--training-receipt")
    parser.add_argument("--expected-training-receipt-sha256")
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument(
        "--execution-scope",
        choices=("exact80", "optimizer-canary-2"),
        default="exact80",
    )
    parser.add_argument(
        "--smoke-sentinel", choices=SENTINEL_ORDER, default=None
    )
    parser.add_argument("--lora-scope", choices=LORA_SCOPES, required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--expected-bernini-commit", default=BERNINI_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=VEOMNI_COMMIT)
    parser.add_argument("--runtime-source-manifest", required=True)
    parser.add_argument("--expected-runtime-source-manifest-sha256", required=True)
    parser.add_argument("--runtime-source-launcher", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> Path:
    for name in (
        "runtime_source_revision",
        "expected_bernini_commit",
        "expected_veomni_commit",
    ):
        _sha(getattr(args, name), label=name, length=40)
    for name in (
        "expected_review_manifest_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_runtime_source_manifest_sha256",
        "runtime_source_archive_sha256",
        "launcher_source_sha256",
    ):
        _sha(getattr(args, name), label=name)
    if (
        args.expected_bernini_commit != BERNINI_COMMIT
        or args.expected_veomni_commit != VEOMNI_COMMIT
        or args.expected_checkpoint_tree_sha256 != CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        fail("pinned Bernini/VeOmni/base-checkpoint identity differs")
    if args.execution_scope == "exact80":
        if (
            args.checkpoint_step not in FORMAL_CHECKPOINT_STEPS
            or args.smoke_sentinel is not None
            or not args.training_receipt
            or not args.expected_training_receipt_sha256
        ):
            fail("formal exact80 checkpoint/sentinel scope differs")
        _sha(args.expected_training_receipt_sha256, label="training receipt SHA")
    elif args.checkpoint_step != 2 or args.smoke_sentinel is None:
        fail("optimizer canary decode is restricted to one explicit P2 sentinel")
    elif args.training_receipt is not None or args.expected_training_receipt_sha256 is not None:
        fail("optimizer canary cannot claim a terminal exact80 receipt")
    return _fresh_output(args.output_dir)


def _atomic_receipt(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        fail("refusing to overwrite receipt")
    payload = review.canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pyav_exact81(path: Path) -> Mapping[str, Any]:
    """Fully decode one video with the in-process, pinned PyAV runtime."""

    try:
        import av
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise PackedPreservationDecodeError("PyAV is required for media admission") from error
    if av.__version__ != PYAV_VERSION:
        fail("pinned PyAV version differs")
    try:
        with av.open(str(path), mode="r") as container:
            streams = tuple(container.streams.video)
            if len(streams) != 1:
                fail(f"{path} must contain exactly one video stream")
            stream = streams[0]
            rate = stream.average_rate
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            codec = str(stream.codec_context.name)
            decoded = 0
            for frame in container.decode(video=0):
                if int(frame.width) != width or int(frame.height) != height:
                    fail(f"{path} changes frame geometry during full decode")
                decoded += 1
    except PackedPreservationDecodeError:
        raise
    except Exception as error:
        raise PackedPreservationDecodeError(f"cannot fully decode {path} with PyAV") from error
    try:
        fps = float(rate)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise PackedPreservationDecodeError(f"invalid PyAV frame rate for {path}") from error
    if (
        decoded != FRAME_COUNT
        or abs(fps - float(FPS)) > 1.0e-9
        or width <= 0
        or height <= 0
        or not codec
    ):
        fail(f"{path} must be fully decodable 81-frame 25-fps media")
    return {
        "frame_count": decoded,
        "fps": FPS,
        "codec": codec,
        "width": width,
        "height": height,
        "probe_backend": f"pyav-{PYAV_VERSION}-full-decode",
    }


def _media_runtime_preflight() -> Mapping[str, Any]:
    """Fail before model load if decode or ImageIO's encoder backend is absent."""

    try:
        import av
        import imageio_ffmpeg
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise PackedPreservationDecodeError("pinned media runtime is unavailable") from error
    if av.__version__ != PYAV_VERSION:
        fail("pinned PyAV version differs")
    ffmpeg = _plain_file(imageio_ffmpeg.get_ffmpeg_exe(), label="ImageIO FFmpeg")
    ffmpeg_sha = review.file_sha256(ffmpeg)
    if ffmpeg_sha != IMAGEIO_FFMPEG_SHA256:
        fail("ImageIO FFmpeg binary bytes differ")
    try:
        completed = subprocess.run(
            [str(ffmpeg), "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PackedPreservationDecodeError("ImageIO FFmpeg execution preflight failed") from error
    lines = completed.stdout.splitlines()
    first_line = lines[0] if lines else ""
    if not first_line.startswith("ffmpeg version 7.0.2-static"):
        fail("ImageIO FFmpeg version banner differs")
    return {
        "pyav_version": av.__version__,
        "probe_backend": "pyav-full-decode",
        "imageio_ffmpeg_path": str(ffmpeg),
        "imageio_ffmpeg_sha256": ffmpeg_sha,
        "imageio_ffmpeg_version": first_line,
        "external_ffprobe_required": False,
        "preflight_before_model_load": True,
    }


def _media_record(*, root: Path, path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file() or root not in path.resolve(strict=True).parents:
        fail("published media must be one plain file below staging root")
    probe = _pyav_exact81(path)
    return {
        "relative_mp4": path.relative_to(root).as_posix(),
        "mp4_sha256": review.file_sha256(path),
        **dict(probe),
    }


def _sample(
    *,
    diffusion: Any,
    prompt_embeds: Any,
    negative_embeds: Any,
    source_latent: Any,
    references: Mapping[int, Any],
    bucket_hw: tuple[int, int],
    latent_shape: tuple[int, ...],
    seed: int,
    device: Any,
    wan_diffusion: Any,
) -> tuple[Any, Mapping[str, Any]]:
    import torch

    endpoint, capture = native._sample_with_native_initial_noise_observer(
        sample_fn=lambda: diffusion.sample(
            prompt_embeds=prompt_embeds,
            uncond_prompt_embeds=negative_embeds,
            image_vae_latents=None,
            multi_video_vae_latents=[source_latent],
            multi_image_vae_latents=[
                references[index] for index in native.RV2V_REFERENCE_INDICES
            ],
            width=bucket_hw[1],
            height=bucket_hw[0],
            device=device,
            **native.native_sampling_contract(
                "rv2v", steps=review.NUM_INFERENCE_STEPS, seed=seed
            ),
        ),
        wan_diffusion_module=wan_diffusion,
        expected_shape=latent_shape,
        expected_device=device,
        expected_seed=seed,
    )
    if (
        not isinstance(endpoint, torch.Tensor)
        or tuple(int(value) for value in endpoint.shape) != latent_shape
        or endpoint.device != device
        or endpoint.dtype != torch.float32
        or endpoint.requires_grad
        or endpoint.grad_fn is not None
        or not endpoint.is_contiguous()
        or not bool(torch.isfinite(endpoint).all().item())
    ):
        fail("official native RV2V endpoint differs")
    endpoint_identity = native._all_rank_tensor_identity(
        endpoint, label=f"packed_preservation_endpoint_{seed}", world_size=review.WORLD_SIZE
    )
    gaussian_identity = native._all_rank_tensor_identity(
        capture.tensor,
        label=f"packed_preservation_gaussian_{seed}",
        world_size=review.WORLD_SIZE,
    )
    return endpoint, {
        "endpoint_identity": endpoint_identity,
        "initial_gaussian_sha256": capture.raw_value_sha256,
        "initial_gaussian_identity": gaussian_identity,
        "initial_gaussian_call_count": capture.call_count,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output = _validate_args(args)

    # No model/data/review implementation is imported before these executable
    # bytes have been rebound to the adjacent canonical manifest, archive, and
    # exact extracted root.  The launcher materializes that root from archive
    # bytes; callers cannot supply METHOD_ROOT.
    import packed_preservation_checkpoint_review_release_v2 as release_contract

    runtime_authority = release_contract.validate_executed_release(
        executed_file=Path(__file__).resolve(),
        executed_launcher=args.runtime_source_launcher,
        manifest=args.runtime_source_manifest,
        expected_manifest_sha256=args.expected_runtime_source_manifest_sha256,
        expected_archive_sha256=args.runtime_source_archive_sha256,
        expected_method_revision=args.runtime_source_revision,
    )

    global authoring, native, lifetime, review, core
    import clean_source_visual_context_checkpoint_review_contract_v1 as authoring
    import infer_native_identity_generation_canary as native
    import infer_native_v_axis_exact81_probe_v1 as lifetime
    import packed_preservation_checkpoint_review_v2 as review
    import packed_preservation_lora_v2 as core

    if authoring.SENTINEL_ORDER != SENTINEL_ORDER or core.LORA_SCOPES != LORA_SCOPES:
        fail("sealed parser constants differ from authenticated runtime modules")
    manifest_path = _plain_file(args.review_manifest, label="fixed review manifest")
    manifest = authoring.load_manifest(
        manifest_path,
        expected_file_sha256=args.expected_review_manifest_sha256,
        verify_files=True,
    )
    sentinel_order = (
        (args.smoke_sentinel,)
        if args.execution_scope == "optimizer-canary-2"
        else authoring.SENTINEL_ORDER
    )
    training_run = _plain_directory(args.training_run, label="continuous exact80 run")
    incremental_checkpoint = review.load_checkpoint_authority(
        training_run / "checkpoints" / f"checkpoint-{args.checkpoint_step:08d}",
        expected_step=args.checkpoint_step,
        expected_lora_scope=args.lora_scope,
        expected_execution_scope=args.execution_scope,
        verify_files=True,
    )
    terminal_training: Optional[review.TrainingAuthority] = None
    if args.execution_scope == "exact80":
        terminal_training = review.load_training_authority(
            args.training_receipt,
            expected_file_sha256=args.expected_training_receipt_sha256,
            expected_lora_scope=args.lora_scope,
            verify_files=True,
        )
        if terminal_training.receipt.parent != training_run:
            fail("terminal receipt does not belong to the requested training run")
        terminal_checkpoint = terminal_training.checkpoint(args.checkpoint_step)
        if dict(terminal_checkpoint.receipt()) != dict(incremental_checkpoint.receipt()):
            fail("checkpoint does not reverse-bind to the terminal exact80 receipt")
    checkpoint = incremental_checkpoint
    source_authority = manifest.get("source_only_manifest")
    if (
        not isinstance(source_authority, Mapping)
        or source_authority.get("file_sha256")
        != review.SOURCE_ONLY_MANIFEST_SHA256
    ):
        fail("four-sentinel authoring is not bound to the training source-only-v3 file")
    checkpoint_manifest = _plain_file(
        args.checkpoint_content_manifest, label="base checkpoint content manifest"
    )
    if review.file_sha256(checkpoint_manifest) != CHECKPOINT_CONTENT_MANIFEST_SHA256:
        fail("base checkpoint content manifest bytes differ")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=BERNINI_COMMIT,
                expected_veomni_commit=VEOMNI_COMMIT,
            )
        )
        base_checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.base_checkpoint
        )
    except Exception as error:
        raise PackedPreservationDecodeError(str(error)) from error
    if (
        transformer_config.get("num_layers") != core.BERNINI_BLOCKS
        or int(transformer_config["num_attention_heads"]) % review.SP_SIZE
    ):
        fail("official Bernini-R 1.3B WORLD4/SP4 geometry differs")
    inference_hashes = native.legacy.validate_inference_source_files(bernini_root)
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode

    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        fail("official Bernini negative prompt differs")
    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != review.WORLD_SIZE
        or distributed.ulysses_size != review.SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        fail("checkpoint review requires one AUH WORLD4/SP4 ROCm process group")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=720),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=review.SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    model: Any = None
    route: Any = None
    vae: Any = None
    try:
        media_payload: list[Any] = [None]
        if distributed.rank == 0:
            try:
                media_payload[0] = {
                    "ok": True,
                    "runtime": dict(_media_runtime_preflight()),
                }
            except Exception as error:
                media_payload[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(media_payload, src=0)
        if not isinstance(media_payload[0], Mapping) or media_payload[0].get("ok") is not True:
            fail(f"rank-zero media runtime preflight failed: {media_payload[0]!r}")
        media_runtime = dict(media_payload[0]["runtime"])

        checkpoint_payload: list[Any] = [None]
        if distributed.rank == 0:
            try:
                checkpoint_payload[0] = {
                    "ok": True,
                    "identity": native.source_audit.validate_checkpoint_content(
                        base_checkpoint,
                        checkpoint_manifest,
                        expected_manifest_sha256=CHECKPOINT_CONTENT_MANIFEST_SHA256,
                    ),
                }
            except Exception as error:
                checkpoint_payload[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(checkpoint_payload, src=0)
        if not isinstance(checkpoint_payload[0], Mapping) or checkpoint_payload[0].get("ok") is not True:
            fail(f"base checkpoint admission failed: {checkpoint_payload[0]!r}")
        checkpoint_identity = dict(checkpoint_payload[0]["identity"])

        sentinel_by_id = {row["sentinel_id"]: row for row in manifest["sentinels"]}
        raw_pixels: dict[str, Any] = {}
        source_metadata: dict[str, Any] = {}
        source_payload: list[Any] = [None]
        if distributed.rank == 0:
            try:
                for sentinel_id in sentinel_order:
                    sentinel = sentinel_by_id[sentinel_id]
                    path = _plain_file(sentinel["source_video"], label=f"{sentinel_id} source")
                    media_probe = _pyav_exact81(path)
                    expected_media = sentinel["source_media"]
                    if any(
                        media_probe[key] != expected_media[key]
                        for key in ("frame_count", "fps", "codec", "width", "height")
                    ):
                        fail(f"{sentinel_id} PyAV source preflight differs")
                    pixels, metadata, digest = native.source_audit.prepare_hashed_source_snapshot(path)
                    latent_shape = tuple(int(item) for item in sentinel["latent_shape"])
                    expected_bucket = (latent_shape[3] * 8, latent_shape[4] * 8)
                    if (
                        digest != sentinel["source_video_sha256"]
                        or tuple(metadata["source_derived_bucket_hw"]) != expected_bucket
                        or tuple(int(item) for item in pixels.shape)
                        != (1, 3, review.FRAME_COUNT, *expected_bucket)
                    ):
                        fail(f"{sentinel_id} raw source geometry/identity differs")
                    raw_pixels[sentinel_id] = pixels
                    source_metadata[sentinel_id] = dict(metadata)
                source_payload[0] = {"ok": True, "metadata": source_metadata}
            except Exception as error:
                source_payload[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(source_payload, src=0)
        if not isinstance(source_payload[0], Mapping) or source_payload[0].get("ok") is not True:
            fail(f"rank-zero source admission failed: {source_payload[0]!r}")
        source_metadata = dict(source_payload[0]["metadata"])

        # Rank zero is the only VAE encoder.  Exact tensors are broadcast to
        # the other SP ranks before the model is loaded.
        if distributed.rank == 0:
            vae = AutoencoderKLWan.from_pretrained(
                str(base_checkpoint),
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            vae.eval().requires_grad_(False).to(device)
        source_latents: dict[str, Any] = {}
        reference_latents: dict[str, dict[int, Any]] = {}
        for sentinel_id in sentinel_order:
            latent_shape = tuple(int(item) for item in sentinel_by_id[sentinel_id]["latent_shape"])
            reference_shape = (1, 16, 1, latent_shape[3], latent_shape[4])
            if distributed.rank == 0:
                pixels = raw_pixels[sentinel_id].to(device=device, dtype=torch.float32)
                with torch.inference_mode():
                    source = _vae_encode(vae, pixels).float().contiguous()
                    references = {
                        index: _vae_encode(
                            vae, pixels[:, :, index : index + 1].contiguous()
                        ).float().contiguous()
                        for index in native.RV2V_REFERENCE_INDICES
                    }
                del pixels
            else:
                source = torch.empty(latent_shape, device=device, dtype=torch.float32)
                references = {
                    index: torch.empty(reference_shape, device=device, dtype=torch.float32)
                    for index in native.RV2V_REFERENCE_INDICES
                }
            if tuple(source.shape) != latent_shape or any(
                tuple(value.shape) != reference_shape for value in references.values()
            ):
                fail(f"{sentinel_id} independently encoded RV2V condition geometry differs")
            dist.broadcast(source, src=0)
            native._all_rank_tensor_identity(
                source, label=f"{sentinel_id}_full_source", world_size=review.WORLD_SIZE
            )
            for index in native.RV2V_REFERENCE_INDICES:
                dist.broadcast(references[index], src=0)
                native._all_rank_tensor_identity(
                    references[index],
                    label=f"{sentinel_id}_reference_{index}",
                    world_size=review.WORLD_SIZE,
                )
            source_latents[sentinel_id] = source
            reference_latents[sentinel_id] = references
        raw_pixels.clear()
        if distributed.rank == 0:
            vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(
            str(base_checkpoint),
            subfolder="tokenizer",
            **native.legacy.tokenizer_load_kwargs(),
        )
        tokenized: dict[str, tuple[Any, Any]] = {}
        prompt_records: dict[str, Any] = {}
        for sentinel_id in sentinel_order:
            sentinel = sentinel_by_id[sentinel_id]
            instruction = sentinel["instructions"]["forward"]
            prompt = native.build_task_prompt("rv2v", instruction, prompt_cleaner=prompt_clean)
            tokenized[sentinel_id] = native.legacy._tokenize_training_prompt(tokenizer, prompt)
            prompt_records[sentinel_id] = {
                "full_instruction": instruction,
                "instruction_utf8_sha256": sentinel["instruction_sha256"]["forward"],
                "native_prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
            tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
        )
        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **native.legacy.inference_renderer_config_overrides(base_checkpoint),
        )
        config.dtype = torch.bfloat16
        native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), base_checkpoint)
        if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
            fail("renderer is not official exact40 UniPC flow-shift5")
        with lifetime._serialized_host_checkpoint_load():
            base_renderer = BerniniRendererModel(config)
            base_renderer.eval().requires_grad_(False).to(device)
        if base_renderer.diff_dec.transformer_2 is not None:
            fail("packed preservation review requires the 1.3B single transformer")
        specs = core.select_projection_specs(base_renderer, args.lora_scope)
        with torch.inference_mode():
            positive_embeds = {
                sentinel_id: base_renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
                for sentinel_id, (ids, mask) in tokenized.items()
            }
            negative_embeds = base_renderer.encode_prompt(
                negative_ids.to(device), negative_mask.to(device)
            ).detach()
        base_renderer.t5_text_encoder = None
        del tokenizer, tokenized, negative_ids, negative_mask
        gc.collect()
        torch.cuda.empty_cache()

        diffusion = base_renderer.diff_dec
        generated: dict[tuple[str, str], Any] = {}
        decode_records: dict[tuple[str, str], Mapping[str, Any]] = {}
        native_endpoints: dict[str, Any] = {}
        gaussian_by_sentinel: dict[str, set[str]] = {
            sentinel_id: set() for sentinel_id in sentinel_order
        }
        if args.checkpoint_step == 0:
            for sentinel_id in sentinel_order:
                sentinel = sentinel_by_id[sentinel_id]
                latent_shape = tuple(int(item) for item in sentinel["latent_shape"])
                bucket_hw = (latent_shape[3] * 8, latent_shape[4] * 8)
                with torch.inference_mode():
                    endpoint, evidence = _sample(
                        diffusion=diffusion,
                        prompt_embeds=positive_embeds[sentinel_id],
                        negative_embeds=negative_embeds,
                        source_latent=source_latents[sentinel_id],
                        references=reference_latents[sentinel_id],
                        bucket_hw=bucket_hw,
                        latent_shape=latent_shape,
                        seed=int(sentinel["seed"]),
                        device=device,
                        wan_diffusion=wan_diffusion,
                    )
                native_endpoints[sentinel_id] = endpoint.detach().cpu().contiguous()
                if distributed.rank == 0:
                    generated[(sentinel_id, "native")] = native_endpoints[sentinel_id]
                gaussian_by_sentinel[sentinel_id].add(evidence["initial_gaussian_sha256"])
                decode_records[(sentinel_id, "native")] = {
                    **dict(evidence),
                    "sentinel_id": sentinel_id,
                    "arm": "native",
                    "checkpoint_step": None,
                    "adapter_loaded": False,
                    "native_patch_route": None,
                }
                del endpoint
                torch.cuda.empty_cache()

        model = get_peft_model(
            base_renderer,
            LoraConfig(
                r=core.LORA_RANK,
                lora_alpha=core.LORA_ALPHA,
                lora_dropout=core.LORA_DROPOUT,
                bias="none",
                target_modules=[item.name for item in specs],
            ),
        )
        base_renderer = model.get_base_model()
        transformer = base_renderer.diff_dec.transformer
        core.install_typed_patch_embedding(transformer)
        model.to(device)
        core.verify_trainable_parameter_count(model, args.lora_scope)
        installation = core.validate_lora_installation(model, specs)
        architecture = core.architecture_receipt(args.lora_scope, specs)
        if any(
            parameter.device != device or parameter.dtype != torch.float32
            for _, parameter in core.trainable_named_parameters(model)
        ):
            fail("inference adapter parameters must be FP32 on the local GPU")
        route = review.NativePatchRoute(transformer)
        route.install()
        diffusion = base_renderer.diff_dec
        step = args.checkpoint_step
        with _serialized_adapter_load():
            loaded = review.strict_load_adapter(model, checkpoint)
        checkpoint_parameter_fp32 = review.trainable_parameter_digest(model)
        if checkpoint_parameter_fp32 != checkpoint.parameter_sha256:
            fail(f"checkpoint {step} FP32 parameter digest differs before inference cast")
        if step == 0 and not review.zero_effect_adapter(model):
            fail("step-0 adapter is not an exact zero-effect initialization")
        # The checkpoint authority and byte digest are verified while every
        # trainable remains FP32, exactly as saved by training.  Native
        # sample() has no outer autocast, and an FP32 direct parameter makes
        # diffusers report ``transformer.dtype == float32`` even though the
        # frozen renderer is BF16.  Cast only the already-authenticated
        # inference adapter to BF16: this is the effective matmul/conv dtype
        # used by training autocast and leaves the native FP32 scheduler alone.
        trainables = tuple(core.trainable_named_parameters(model))
        with torch.no_grad():
            for _, parameter in trainables:
                parameter.data = parameter.data.to(dtype=torch.bfloat16)
        if (
            not trainables
            or any(
                parameter.device != device or parameter.dtype != torch.bfloat16
                for _, parameter in trainables
            )
            or transformer.dtype != torch.bfloat16
        ):
            fail("inference-only BF16 adapter cast did not restore native renderer dtype")
        inference_parameter_before = review.trainable_parameter_digest(model)
        inference_cast = {
            "checkpoint_fp32_parameter_sha256": checkpoint_parameter_fp32,
            "inference_bfloat16_parameter_sha256": inference_parameter_before,
            "trainable_tensor_count": len(trainables),
            "checkpoint_dtype": "torch.float32",
            "inference_dtype": "torch.bfloat16",
            "transformer_dtype": str(transformer.dtype),
            "cast_after_strict_checkpoint_digest": True,
            "native_scheduler_outside_autocast": True,
        }
        for sentinel_id in sentinel_order:
            sentinel = sentinel_by_id[sentinel_id]
            latent_shape = tuple(int(item) for item in sentinel["latent_shape"])
            bucket_hw = (latent_shape[3] * 8, latent_shape[4] * 8)
            with torch.inference_mode():
                endpoint, evidence = _sample(
                    diffusion=diffusion,
                    prompt_embeds=positive_embeds[sentinel_id],
                    negative_embeds=negative_embeds,
                    source_latent=source_latents[sentinel_id],
                    references=reference_latents[sentinel_id],
                    bucket_hw=bucket_hw,
                    latent_shape=latent_shape,
                    seed=int(sentinel["seed"]),
                    device=device,
                    wan_diffusion=wan_diffusion,
                )
            trace = route.trace(clear=True)
            if (
                trace["calls"] != review.NUM_INFERENCE_STEPS * 10
                or trace["source_calls"] != review.NUM_INFERENCE_STEPS * 9
                or trace["target_calls"] != review.NUM_INFERENCE_STEPS
            ):
                fail("native RV2V typed patch source/target call closure differs")
            if step == 0 and not torch.equal(
                endpoint.detach().cpu(), native_endpoints[sentinel_id]
            ):
                fail(f"{sentinel_id} step-0 endpoint is not byte-exact native")
            arm = f"step-{step}"
            if distributed.rank == 0:
                generated[(sentinel_id, arm)] = endpoint.detach().cpu().contiguous()
            gaussian_by_sentinel[sentinel_id].add(evidence["initial_gaussian_sha256"])
            decode_records[(sentinel_id, arm)] = {
                **dict(evidence),
                "sentinel_id": sentinel_id,
                "arm": arm,
                "checkpoint_step": step,
                "adapter_loaded": True,
                "checkpoint_parameter_sha256": checkpoint.parameter_sha256,
                "native_patch_route": trace,
                "step_zero_native_endpoint_equal": step == 0,
            }
            del endpoint
            torch.cuda.empty_cache()
        inference_parameter_after = review.trainable_parameter_digest(model)
        if inference_parameter_after != inference_parameter_before:
            fail(f"checkpoint {step} adapter changed during inference")
        checkpoint_load = {
            **dict(loaded),
            "inference_cast": inference_cast,
            "parameter_unchanged_during_inference": True,
        }
        if any(len(values) != 1 for values in gaussian_by_sentinel.values()):
            fail("same sentinel seed/shape did not reuse one exact official Gaussian")
        route.restore()
        route = None
        del diffusion, transformer, model, base_renderer, positive_embeds, negative_embeds
        model = None
        gc.collect()
        torch.cuda.empty_cache()

        if distributed.rank != 0:
            generated.clear()
            native_endpoints.clear()
        if distributed.rank == 0:
            stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
            media_dir = stage / "media"
            media_dir.mkdir(mode=0o700)
            source_records: list[Mapping[str, Any]] = []
            for sentinel_id in sentinel_order:
                sentinel = sentinel_by_id[sentinel_id]
                destination = media_dir / f"{sentinel_id}__source.mp4"
                shutil.copyfile(Path(sentinel["source_video"]), destination)
                media = _media_record(root=stage, path=destination)
                if media["mp4_sha256"] != sentinel["source_video_sha256"]:
                    fail("self-contained source snapshot changed bytes")
                source_records.append(
                    {
                        "sentinel_id": sentinel_id,
                        "iid": sentinel["iid"],
                        "diversity_role": sentinel["diversity_role"],
                        "source_entity_type": sentinel["source_entity_type"],
                        "source_caption": sentinel["source_caption"],
                        "source_video_sha256": sentinel["source_video_sha256"],
                        "full_instruction": sentinel["instructions"]["forward"],
                        "instruction_utf8_sha256": sentinel["instruction_sha256"]["forward"],
                        "seed": sentinel["seed"],
                        **dict(media),
                    }
                )
            if vae is None:
                fail("rank-zero VAE lifetime differs")
            for sentinel_id in sentinel_order:
                sentinel = sentinel_by_id[sentinel_id]
                latent_shape = tuple(int(item) for item in sentinel["latent_shape"])
                bucket_hw = (latent_shape[3] * 8, latent_shape[4] * 8)
                keys = (["native"] if args.checkpoint_step == 0 else []) + [
                    f"step-{args.checkpoint_step}"
                ]
                save_keys = ["native"] if args.checkpoint_step == 0 else keys
                device_outputs = {
                    f"{sentinel_id}__{key}": generated[(sentinel_id, key)].to(device).contiguous()
                    for key in save_keys
                }
                saved = native._save_outputs(
                    output_dir=media_dir,
                    generated=device_outputs,
                    vae=vae,
                    bucket_hw=bucket_hw,
                    device=device,
                    save_output_fn=save_output,
                )
                if args.checkpoint_step == 0:
                    native_key = f"{sentinel_id}__native"
                    adapted_key = f"{sentinel_id}__step-0"
                    native_path = Path(saved[native_key]["path"])
                    adapted_path = media_dir / f"{adapted_key}.mp4"
                    if adapted_path.exists() or adapted_path.is_symlink():
                        fail("step-0 byte-copy destination is not fresh")
                    shutil.copyfile(native_path, adapted_path)
                    saved[adapted_key] = {"path": str(adapted_path)}
                for key in keys:
                    record_key = (sentinel_id, key)
                    media = _media_record(
                        root=stage, path=Path(saved[f"{sentinel_id}__{key}"]["path"])
                    )
                    decode_records[record_key] = {
                        **dict(decode_records[record_key]),
                        **dict(media),
                    }
                del device_outputs, saved
                torch.cuda.empty_cache()
            record_keys = (["native"] if args.checkpoint_step == 0 else []) + [
                f"step-{args.checkpoint_step}"
            ]
            ordered_records = [
                dict(decode_records[(sentinel_id, key)])
                for sentinel_id in sentinel_order
                for key in record_keys
            ]
            unsigned = {
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "complete": True,
                "lora_scope": args.lora_scope,
                "execution_scope": args.execution_scope,
                "checkpoint_step": args.checkpoint_step,
                "sentinel_order": list(sentinel_order),
                "smoke_only": args.execution_scope == "optimizer-canary-2",
                "review_manifest": {
                    "path": str(manifest_path),
                    "file_sha256": args.expected_review_manifest_sha256,
                    "manifest_digest": manifest["manifest_digest"],
                },
                "checkpoint_authority": dict(checkpoint.receipt()),
                "continuous_training_run": str(training_run),
                "terminal_training_receipt_bound": terminal_training is not None,
                "terminal_training_authority": (
                    dict(terminal_training.as_receipt())
                    if terminal_training is not None
                    else None
                ),
                "base_checkpoint": {
                    "path": str(base_checkpoint),
                    "tree_sha256": CHECKPOINT_TREE_SHA256,
                    "content_manifest_sha256": CHECKPOINT_CONTENT_MANIFEST_SHA256,
                    "content_identity": checkpoint_identity,
                    "opened_read_only": True,
                },
                "runtime_source": {
                    **dict(runtime_authority),
                    "launcher_sha256": args.launcher_source_sha256,
                },
                "pinned_sources": {
                    "bernini_commit": bernini_revision,
                    "veomni_commit": veomni_revision,
                    "inference_files": inference_hashes,
                },
                "architecture": architecture,
                "lora_installation": installation,
                "checkpoint_load": checkpoint_load,
                "source_preprocessing": source_metadata,
                "prompts": prompt_records,
                "source_records": source_records,
                "decode_records": ordered_records,
                "execution": {
                    "world_size": review.WORLD_SIZE,
                    "sequence_parallel_size": review.SP_SIZE,
                    "num_inference_steps": review.NUM_INFERENCE_STEPS,
                    "frame_count": review.FRAME_COUNT,
                    "fps": review.FPS,
                    "native_rv2v_four_references": True,
                    "same_source_instruction_seed_all_columns": True,
                    "step_zero_endpoint_byte_exact_native": (
                        args.checkpoint_step == 0
                    ),
                    "step_zero_media_byte_copy_from_native": (
                        args.checkpoint_step == 0
                    ),
                    "parent_allocation_released": False,
                },
                "runtime_versions": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "transformers": transformers_version,
                    "diffusers": diffusers_version,
                    "media": media_runtime,
                },
                "optimizer_present": False,
                "backward_performed": False,
                "parameter_update_performed": False,
                "feature_evaluator_present": False,
                "vlm_evaluator_present": False,
                "automatic_ranking_present": False,
                "candidate_selection_present": False,
                "quality_claimed": False,
                "scientific_claim_authorized": False,
                "manual_review_pending": True,
            }
            receipt = {**unsigned, "receipt_digest": review.object_sha256(unsigned)}
            _atomic_receipt(stage / "receipt.json", receipt)
            os.rename(stage, output)
            print(review.canonical_json_bytes(receipt).decode("ascii"), flush=True)
        dist.barrier()
    finally:
        if route is not None and route.installed:
            try:
                route.restore()
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METHOD",
    "PackedPreservationDecodeError",
    "SCHEMA_VERSION",
    "main",
]

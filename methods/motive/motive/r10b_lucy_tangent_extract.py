"""Extract a non-promotional R10B tangent smoke artifact from frozen Lucy.

This backend intentionally bypasses ``LucyEditPipeline.__call__`` because that
inference entrypoint is decorated with ``torch.no_grad``.  It reuses the
pipeline's VAE/text components and calls the frozen transformer directly.

Safety contract:

* no optimizer is constructed;
* no parameter update or scheduler step is performed;
* no video is decoded, rendered, or copied;
* the selected checkpoint weights are hashed before and after extraction;
* all representation, renderer, and training permission fields stay false.

Lucy is a failed 5B editing baseline in this project.  Its R10B output can
validate the measurement implementation, but can never promote a
representation.  The primary scientific R10B model remains a smaller
primary-eligible instruction editor such as Bernini-R 1.3B.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from .attribution import (
    CountSketchProjector,
    factorial_edit_tangents,
    normalize_projected_tangent,
    project_parameter_gradients,
    selected_parameter_manifest,
)
from .r10b_tangent_core import (
    R10BTangentError,
    canonical_json,
    file_digest,
    motion_x0_measurement_loss,
    object_digest,
    read_jsonl,
    resolve_lucy_attention_roles,
    set_only_parameters_trainable,
    temporal_broadcast_noise,
    track_delta_saliency,
    validate_smoke_rows,
)


EXTRACT_SCHEMA = "motive-r10b-lucy-frozen-tangent-extract-v1"
ROW_SCHEMA = "motive-r10b-lucy-frozen-tangent-row-v1"
DONE_SCHEMA = "motive-r10b-lucy-frozen-tangent-done-v1"
FEATURES_NAME = "features.npz"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
PAYLOAD_NAMES = (FEATURES_NAME, ROWS_NAME, SUMMARY_NAME)
OUTPUT_NAMES = (*PAYLOAD_NAMES, DONE_NAME)
CELL_NAMES = ("tc", "sc", "t0", "s0")
ROLE_NAMES = ("self_motion", "cross_instruction")
TANGENT_NAMES = (
    "target_only",
    "paired_delta",
    "factorial_did",
    "cross_cell_control",
    "noop_target_delta",
)


class R10BLucyExtractError(RuntimeError):
    """The Lucy tangent backend cannot satisfy its frozen measurement contract."""


def _torch() -> Any:
    import torch

    return torch


def _pretty_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _weight_digest(model: Any, names: set[str]) -> str:
    torch = _torch()
    digest = hashlib.sha256()
    found = set()
    for name, parameter in model.named_parameters():
        if str(name) not in names:
            continue
        found.add(str(name))
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_json(list(parameter.shape)).encode("utf-8"))
        digest.update(b"\0")
        values = parameter.detach().to(device="cpu").contiguous()
        if values.dtype == torch.bfloat16:
            values = values.view(torch.uint16)
        digest.update(values.numpy().tobytes(order="C"))
    missing = sorted(names - found)
    if missing:
        raise R10BLucyExtractError(f"weight digest missed selected names: {missing}")
    return digest.hexdigest()


def _encode_lucy_latents(
    pipe: Any,
    frames: list[Any],
    *,
    height: int,
    width: int,
    device: Any,
    output_dtype: Any,
) -> Any:
    """Use Lucy's official deterministic posterior and latent normalization."""

    torch = _torch()
    video = pipe.video_processor.preprocess_video(
        frames,
        height=height,
        width=width,
    ).to(device=device, dtype=pipe.vae.dtype)
    if video.ndim != 5:
        raise R10BLucyExtractError(
            f"Lucy video processor returned unexpected shape {tuple(video.shape)}"
        )
    encoded = pipe.vae.encode(video)
    if hasattr(encoded, "latent_dist"):
        latents = encoded.latent_dist.mode()
    elif hasattr(encoded, "latents"):
        latents = encoded.latents
    else:
        raise R10BLucyExtractError(
            "Lucy VAE output exposes neither latent_dist nor latents"
        )
    mean_config = getattr(pipe.vae.config, "latents_mean", None)
    std_config = getattr(pipe.vae.config, "latents_std", None)
    z_dim = int(getattr(pipe.vae.config, "z_dim", latents.shape[1]))
    if mean_config is None or std_config is None:
        raise R10BLucyExtractError(
            "Lucy VAE config lacks required latents_mean/latents_std"
        )
    mean = torch.tensor(
        mean_config,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(
        std_config,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, z_dim, 1, 1, 1)
    if tuple(mean.shape[:2]) != tuple(latents.shape[:2]):
        raise R10BLucyExtractError(
            f"Lucy latent normalization shape differs: {tuple(mean.shape)} "
            f"vs {tuple(latents.shape)}"
        )
    if not bool(torch.isfinite(std).all()) or float(std.abs().min()) <= 0:
        raise R10BLucyExtractError("Lucy latent std is non-finite or zero")
    normalized = (latents - mean) / std
    if not bool(torch.isfinite(normalized).all()):
        raise R10BLucyExtractError("Lucy normalized latents contain non-finite values")
    return normalized.to(dtype=output_dtype)


def _encode_prompt(
    pipe: Any,
    prompt: str,
    *,
    device: Any,
    dtype: Any,
    max_sequence_length: int,
) -> Any:
    encoded = pipe.encode_prompt(
        prompt=[prompt],
        negative_prompt=None,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
    )
    embeddings = encoded[0] if isinstance(encoded, tuple) else encoded
    if embeddings.shape[0] != 1 or not bool(_torch().isfinite(embeddings).all()):
        raise R10BLucyExtractError("Lucy prompt embeddings are invalid")
    return embeddings


def _scheduler_midpoint(
    pipe: Any,
    *,
    steps: int,
    device: Any,
) -> tuple[Any, float, int]:
    if steps < 3:
        raise R10BLucyExtractError("scheduler_steps must be at least three")
    pipe.scheduler.set_timesteps(steps, device=device)
    index = len(pipe.scheduler.timesteps) // 2
    timestep = pipe.scheduler.timesteps[index]
    sigmas = getattr(pipe.scheduler, "sigmas", None)
    if sigmas is None or len(sigmas) <= index:
        raise R10BLucyExtractError("Lucy scheduler exposes no aligned sigma")
    sigma = float(sigmas[index])
    if not math.isfinite(sigma) or not 0.0 < sigma < 1.0:
        raise R10BLucyExtractError(
            f"Lucy native midpoint sigma must lie in (0,1), got {sigma}"
        )
    return timestep, sigma, index


def _expanded_timestep(
    timestep: Any,
    latents: Any,
    transformer: Any,
    *,
    expand_timesteps: bool,
) -> Any:
    torch = _torch()
    if not expand_timesteps:
        return timestep.expand(latents.shape[0])
    patch_size = tuple(getattr(transformer.config, "patch_size", (1, 2, 2)))
    if len(patch_size) != 3:
        raise R10BLucyExtractError("Lucy transformer patch_size must be length three")
    latent_frames = latents.shape[2] // int(patch_size[0])
    latent_height = latents.shape[3] // int(patch_size[1])
    latent_width = latents.shape[4] // int(patch_size[2])
    sequence_length = int(latent_frames * latent_height * latent_width)
    if sequence_length <= 0:
        raise R10BLucyExtractError("expanded timestep sequence length is zero")
    return torch.ones(
        (latents.shape[0], sequence_length),
        device=latents.device,
        # Lucy's official path multiplies an FP32 mask by the scheduler
        # timestep before flattening, so expanded timesteps are FP32.
        dtype=torch.float32,
    ) * timestep


def _transformer_prediction(
    transformer: Any,
    *,
    noisy_latents: Any,
    source_latents: Any,
    prompt_embeds: Any,
    timestep: Any,
    expand_timesteps: bool,
) -> Any:
    torch = _torch()
    model_input = torch.cat((noisy_latents, source_latents), dim=1).to(
        dtype=transformer.dtype
    )
    expected_channels = int(getattr(transformer.config, "in_channels", -1))
    if model_input.shape[1] != expected_channels:
        raise R10BLucyExtractError(
            f"Lucy source concatenation produced {model_input.shape[1]} channels; "
            f"checkpoint expects {expected_channels}"
        )
    token_timestep = _expanded_timestep(
        timestep,
        noisy_latents,
        transformer,
        expand_timesteps=expand_timesteps,
    )
    cache_context = (
        transformer.cache_context("cond")
        if hasattr(transformer, "cache_context")
        else nullcontext()
    )
    with cache_context:
        output = transformer(
            hidden_states=model_input,
            timestep=token_timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=None,
            return_dict=False,
        )
    prediction = output[0] if isinstance(output, (tuple, list)) else output.sample
    prediction = prediction[:, : noisy_latents.shape[1]]
    if prediction.shape != noisy_latents.shape:
        raise R10BLucyExtractError(
            f"Lucy prediction shape differs: {tuple(prediction.shape)} vs "
            f"{tuple(noisy_latents.shape)}"
        )
    if not bool(torch.isfinite(prediction).all()):
        raise R10BLucyExtractError("Lucy prediction contains non-finite values")
    return prediction


def _uniform_x0_loss(
    prediction: Any,
    noisy_latents: Any,
    clean_latents: Any,
    *,
    sigma: float,
) -> Any:
    x0_hat = noisy_latents.float() - float(sigma) * prediction.float()
    return (x0_hat - clean_latents.float()).pow(2).mean()


def _project_current_gradients(
    transformer: Any,
    parameter_roles: Mapping[str, Any],
    *,
    projection_seeds: tuple[int, ...],
    projection_dim: int,
) -> tuple[dict[tuple[str, int], Any], dict[str, Any]]:
    outputs = {}
    diagnostics = {}
    for role_name in ROLE_NAMES:
        names = tuple(parameter_roles["roles"][role_name]["names"])
        for projection_seed in projection_seeds:
            projected, detail = project_parameter_gradients(
                transformer,
                projector=CountSketchProjector(
                    output_dim=projection_dim,
                    seed=projection_seed,
                ),
                trainable_only=True,
                name_contains=names,
                normalize=False,
            )
            if (
                not detail["all_present_gradients_finite"]
                or detail["gradient_tensors_nonzero"] <= 0
                or detail["raw_projection_l2"] <= 0
            ):
                raise R10BLucyExtractError(
                    f"invalid gradient for role={role_name}, "
                    f"projection_seed={projection_seed}: {detail}"
                )
            outputs[(role_name, projection_seed)] = projected.detach().float().cpu()
            diagnostics[f"{role_name}:{projection_seed}"] = detail
    return outputs, diagnostics


def _one_gradient_cell(
    transformer: Any,
    parameter_roles: Mapping[str, Any],
    *,
    clean_latents: Any,
    source_latents: Any,
    prompt_embeds: Any,
    motion_mask: Any,
    timestep: Any,
    sigma: float,
    diffusion_noise_seed: int,
    expand_timesteps: bool,
    projection_seeds: tuple[int, ...],
    projection_dim: int,
    objective: str,
) -> tuple[dict[tuple[str, int], Any], dict[str, Any]]:
    torch = _torch()
    transformer.zero_grad(set_to_none=True)
    noise = temporal_broadcast_noise(
        clean_latents,
        seed=diffusion_noise_seed,
    )
    noisy = (1.0 - sigma) * clean_latents + sigma * noise
    prediction = _transformer_prediction(
        transformer,
        noisy_latents=noisy,
        source_latents=source_latents,
        prompt_embeds=prompt_embeds,
        timestep=timestep,
        expand_timesteps=expand_timesteps,
    )
    if objective == "motion_x0":
        loss, loss_metrics = motion_x0_measurement_loss(
            prediction,
            noisy,
            clean_latents,
            motion_mask,
            sigma=sigma,
        )
    elif objective == "uniform_x0":
        loss = _uniform_x0_loss(
            prediction,
            noisy,
            clean_latents,
            sigma=sigma,
        )
        loss_metrics = {"combined_loss": float(loss.detach())}
    else:
        raise R10BLucyExtractError(f"unknown objective: {objective}")
    if not bool(torch.isfinite(loss)) or float(loss.detach()) <= 0:
        raise R10BLucyExtractError(
            f"measurement loss is non-positive or non-finite: {loss_metrics}"
        )
    loss.backward()
    projections, gradient_metrics = _project_current_gradients(
        transformer,
        parameter_roles,
        projection_seeds=projection_seeds,
        projection_dim=projection_dim,
    )
    transformer.zero_grad(set_to_none=True)
    del prediction, noisy, noise, loss
    return projections, {
        "loss": loss_metrics,
        "gradients": gradient_metrics,
    }


def _load_track_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {
        "input_indices",
        "source_stabilized_tracks",
        "target_stabilized_tracks",
        "source_visibility",
        "target_visibility",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise R10BLucyExtractError(f"track cache is missing {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    input_indices = arrays["input_indices"]
    if (
        input_indices.ndim != 1
        or not np.issubdtype(input_indices.dtype, np.integer)
        or np.any(input_indices < 0)
        or len(set(input_indices.tolist())) != len(input_indices)
    ):
        raise R10BLucyExtractError(
            "track cache input_indices must be unique nonnegative integers"
        )
    expected_rows = len(input_indices)
    if any(
        values.ndim < 1 or len(values) != expected_rows
        for values in arrays.values()
    ):
        raise R10BLucyExtractError("track cache arrays have different row counts")
    source_tracks = arrays["source_stabilized_tracks"]
    target_tracks = arrays["target_stabilized_tracks"]
    source_visibility = arrays["source_visibility"]
    target_visibility = arrays["target_visibility"]
    if (
        source_tracks.shape != target_tracks.shape
        or source_tracks.ndim != 4
        or source_tracks.shape[1] < 2
        or source_tracks.shape[2] < 1
        or source_tracks.shape[3] != 2
        or source_visibility.shape != source_tracks.shape[:3]
        or target_visibility.shape != source_tracks.shape[:3]
    ):
        raise R10BLucyExtractError("track cache geometry contract differs")
    if not all(
        np.isfinite(values).all()
        for values in (
            source_tracks,
            target_tracks,
            source_visibility,
            target_visibility,
        )
    ):
        raise R10BLucyExtractError("track cache geometry is non-finite")
    return arrays


def _preencode_cases(
    pipe: Any,
    rows: list[dict[str, Any]],
    track_arrays: Mapping[str, np.ndarray],
    *,
    width: int,
    height: int,
    num_frames: int,
    device: Any,
    dtype: Any,
    max_sequence_length: int,
) -> list[dict[str, Any]]:
    from lucy.video import load_video_frames

    encoded_cases = []
    for row in rows:
        try:
            data_root = Path(str(row["data_root"])).expanduser().resolve(strict=True)
        except OSError as error:
            raise R10BLucyExtractError(
                f"data_root is unavailable for iid={row['iid']}"
            ) from error
        cache_index = int(row["track_cache_index"])
        if cache_index >= len(track_arrays["input_indices"]):
            raise R10BLucyExtractError(
                f"track_cache_index is out of range for iid={row['iid']}"
            )
        if int(track_arrays["input_indices"][cache_index]) != int(
            row["track_input_index"]
        ):
            raise R10BLucyExtractError(
                f"track input/cache binding differs for iid={row['iid']}"
            )
        media_paths = {}
        for name, relative_path in (
            ("source", row["src_video"]),
            ("target", row["tgt_video"]),
        ):
            path = (data_root / str(relative_path)).resolve()
            try:
                path.relative_to(data_root)
            except ValueError as error:
                raise R10BLucyExtractError(
                    f"{name} media escapes data_root for iid={row['iid']}"
                ) from error
            media_paths[name] = path
        source_path = media_paths["source"]
        target_path = media_paths["target"]
        for name, path, expected_digest in (
            ("source", source_path, row["src_video_sha256"]),
            ("target", target_path, row["tgt_video_sha256"]),
        ):
            if not path.is_file():
                raise R10BLucyExtractError(f"{name} media is missing: {path}")
            observed_digest = file_digest(path)
            if observed_digest != expected_digest:
                raise R10BLucyExtractError(
                    f"{name} media digest differs for iid={row['iid']}"
                )
        source_frames = load_video_frames(
            source_path,
            width=width,
            height=height,
            num_frames=num_frames,
            sample_mode="uniform",
            short_video_mode="error",
        )
        target_frames = load_video_frames(
            target_path,
            width=width,
            height=height,
            num_frames=num_frames,
            sample_mode="uniform",
            short_video_mode="error",
        )
        with _torch().no_grad():
            source_latents = _encode_lucy_latents(
                pipe,
                source_frames,
                height=height,
                width=width,
                device=device,
                output_dtype=dtype,
            )
            target_latents = _encode_lucy_latents(
                pipe,
                target_frames,
                height=height,
                width=width,
                device=device,
                output_dtype=dtype,
            )
            prompt_embeds = _encode_prompt(
                pipe,
                str(row["prompt"]),
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
            )
            noop_embeds = _encode_prompt(
                pipe,
                str(row["noop_prompt"]),
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
            )
        if source_latents.shape != target_latents.shape:
            raise R10BLucyExtractError(
                f"source/target latent shape differs for iid={row['iid']}"
            )
        motion_mask, saliency_metrics = track_delta_saliency(
            track_arrays["source_stabilized_tracks"][cache_index],
            track_arrays["target_stabilized_tracks"][cache_index],
            track_arrays["source_visibility"][cache_index],
            track_arrays["target_visibility"][cache_index],
        )
        if saliency_metrics["normalized_active_fraction"] <= 0:
            raise R10BLucyExtractError(
                f"motion saliency is empty for iid={row['iid']}"
            )
        encoded_cases.append(
            {
                "row": row,
                "source_latents": source_latents.detach().cpu(),
                "target_latents": target_latents.detach().cpu(),
                "prompt_embeds": prompt_embeds.detach().cpu(),
                "noop_embeds": noop_embeds.detach().cpu(),
                "motion_mask": motion_mask,
                "saliency": saliency_metrics,
            }
        )
    return encoded_cases


def _move_frozen_encoders_to_cpu(pipe: Any) -> None:
    for name in ("vae", "text_encoder", "text_encoder_2", "image_encoder"):
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, "to"):
            module.to("cpu")
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _source_implementation_digest() -> dict[str, Any]:
    files = {}
    for path in (
        Path(__file__),
        Path(__file__).with_name("r10b_tangent_core.py"),
        Path(__file__).with_name("attribution.py"),
    ):
        files[path.name] = file_digest(path)
    return {
        "files": dict(sorted(files.items())),
        "bundle_sha256": object_digest(dict(sorted(files.items()))),
    }


def extract(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.width % 32 or args.height % 32:
        raise R10BLucyExtractError("Lucy width/height must be divisible by 32")
    if (args.num_frames - 1) % 4:
        raise R10BLucyExtractError("Lucy num_frames must satisfy (F-1) % 4 == 0")
    if len(set(args.projection_seeds)) != len(args.projection_seeds):
        raise R10BLucyExtractError("projection seeds must be unique")
    if len(args.projection_seeds) < 2:
        raise R10BLucyExtractError(
            "engineering smoke requires at least two projection seeds"
        )
    rows = read_jsonl(args.manifest)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    try:
        validate_smoke_rows(rows)
    except R10BTangentError as error:
        raise R10BLucyExtractError(str(error)) from error

    from diffusers import AutoencoderKLWan, LucyEditPipeline

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise R10BLucyExtractError("R10B Lucy extraction requires one CUDA/ROCm GPU")
    dtype = torch.bfloat16
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.reset_peak_memory_stats(device)

    checkpoint_file = args.model_path / "transformer" / "diffusion_pytorch_model.safetensors"
    checkpoint_digest = file_digest(checkpoint_file)
    # Lucy's model card and the project's verified inference entrypoint use
    # an FP32 VAE.  Avoid adding a backend-dependent BF16 perturbation before
    # measuring tangent directions.
    vae_dtype = torch.float32
    vae = AutoencoderKLWan.from_pretrained(
        str(args.model_path),
        subfolder="vae",
        torch_dtype=vae_dtype,
    )
    pipe = LucyEditPipeline.from_pretrained(
        str(args.model_path),
        vae=vae,
        torch_dtype=dtype,
    )
    pipe.to(device)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    transformer = pipe.transformer
    transformer.eval()
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)

    parameter_roles = resolve_lucy_attention_roles(
        transformer.named_parameters(),
        block_index=args.block_index,
        include_cross_kv=args.include_cross_kv,
    )
    set_only_parameters_trainable(transformer, parameter_roles)
    selected_names = {
        name
        for role in parameter_roles["roles"].values()
        for name in role["names"]
    }
    parameter_manifest = selected_parameter_manifest(
        transformer,
        trainable_only=True,
    )
    parameter_manifest_digest = object_digest(parameter_manifest)
    selected_weight_digest_before = _weight_digest(transformer, selected_names)

    track_arrays = _load_track_arrays(args.track_cache)
    encoded_cases = _preencode_cases(
        pipe,
        rows,
        track_arrays,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        device=device,
        dtype=dtype,
        max_sequence_length=args.max_sequence_length,
    )
    _move_frozen_encoders_to_cpu(pipe)
    timestep, sigma, scheduler_index = _scheduler_midpoint(
        pipe,
        steps=args.scheduler_steps,
        device=device,
    )
    expand_timesteps = bool(pipe.config.expand_timesteps)

    feature_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    row_outputs = []
    for case_index, case in enumerate(encoded_cases):
        row = case["row"]
        source_latents = case["source_latents"].to(device=device, dtype=dtype)
        target_latents = case["target_latents"].to(device=device, dtype=dtype)
        prompt_embeds = case["prompt_embeds"].to(device=device, dtype=dtype)
        noop_embeds = case["noop_embeds"].to(device=device, dtype=dtype)
        motion_mask = torch.from_numpy(case["motion_mask"]).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        )
        cell_inputs = {
            "tc": (target_latents, prompt_embeds),
            "sc": (source_latents, prompt_embeds),
            "t0": (target_latents, noop_embeds),
            "s0": (source_latents, noop_embeds),
        }
        raw_cells: dict[tuple[str, int], dict[str, Any]] = {
            (role, seed): {}
            for role in ROLE_NAMES
            for seed in args.projection_seeds
        }
        cell_metrics = {}
        for cell_name in CELL_NAMES:
            clean_latents, embeddings = cell_inputs[cell_name]
            projections, metrics = _one_gradient_cell(
                transformer,
                parameter_roles,
                clean_latents=clean_latents,
                source_latents=source_latents,
                prompt_embeds=embeddings,
                motion_mask=motion_mask,
                timestep=timestep,
                sigma=sigma,
                diffusion_noise_seed=args.diffusion_noise_seed,
                expand_timesteps=expand_timesteps,
                projection_seeds=tuple(args.projection_seeds),
                projection_dim=args.projection_dim,
                objective="motion_x0",
            )
            cell_metrics[cell_name] = metrics
            for key, value in projections.items():
                raw_cells[key][cell_name] = value

        uniform_projections, uniform_metrics = _one_gradient_cell(
            transformer,
            parameter_roles,
            clean_latents=target_latents,
            source_latents=source_latents,
            prompt_embeds=prompt_embeds,
            motion_mask=motion_mask,
            timestep=timestep,
            sigma=sigma,
            diffusion_noise_seed=args.diffusion_noise_seed,
            expand_timesteps=expand_timesteps,
            projection_seeds=tuple(args.projection_seeds),
            projection_dim=args.projection_dim,
            objective="uniform_x0",
        )
        for role_name in ROLE_NAMES:
            for projection_seed in args.projection_seeds:
                tangents = factorial_edit_tangents(
                    raw_cells[(role_name, projection_seed)]
                )
                for tangent_name in TANGENT_NAMES:
                    feature_rows[
                        f"{role_name}__{tangent_name}__p{projection_seed}"
                    ].append(tangents[tangent_name].numpy())
                uniform = normalize_projected_tangent(
                    uniform_projections[(role_name, projection_seed)]
                )
                feature_rows[
                    f"{role_name}__uniform_target__p{projection_seed}"
                ].append(uniform.numpy())
                for cell_name in CELL_NAMES:
                    feature_rows[
                        f"raw__{role_name}__{cell_name}__p{projection_seed}"
                    ].append(raw_cells[(role_name, projection_seed)][cell_name].numpy())

        for projection_seed in args.projection_seeds:
            for tangent_name in TANGENT_NAMES:
                self_value = feature_rows[
                    f"self_motion__{tangent_name}__p{projection_seed}"
                ][-1]
                cross_value = feature_rows[
                    f"cross_instruction__{tangent_name}__p{projection_seed}"
                ][-1]
                combined = np.concatenate((self_value, cross_value), axis=0)
                combined_norm = float(np.linalg.norm(combined))
                if not math.isfinite(combined_norm) or combined_norm <= 0:
                    raise R10BLucyExtractError("combined tangent is zero/non-finite")
                feature_rows[
                    f"combined_balanced__{tangent_name}__p{projection_seed}"
                ].append((combined / combined_norm).astype(np.float32))

        row_outputs.append(
            {
                "schema_version": ROW_SCHEMA,
                "iid": row["iid"],
                "family": row["family"],
                "component_id": row["component_id"],
                "case_index": int(case_index),
                "source_split": row["source_split"],
                "saliency": case["saliency"],
                "cells": cell_metrics,
                "uniform_control": uniform_metrics,
                "projection_seed_coordinates_comparable": False,
                "formal_evidence": False,
                "representation_gate_passed": False,
                "renderer_probe_authorized": False,
                "editor_training_authorized": False,
            }
        )
        del source_latents, target_latents, prompt_embeds, noop_embeds, motion_mask
        torch.cuda.empty_cache()

    selected_weight_digest_after = _weight_digest(transformer, selected_names)
    weights_unchanged = selected_weight_digest_after == selected_weight_digest_before
    if not weights_unchanged:
        raise R10BLucyExtractError("selected frozen checkpoint weights changed")

    features = {
        name: np.stack(values).astype(np.float32)
        for name, values in sorted(feature_rows.items())
    }
    identifiers = np.asarray([str(row["iid"]) for row in rows])
    for name, values in features.items():
        if values.ndim != 2 or len(values) != len(rows):
            raise R10BLucyExtractError(f"invalid feature array shape for {name}")
        if not np.isfinite(values).all():
            raise R10BLucyExtractError(f"feature array contains non-finite values: {name}")

    summary = {
        "schema_version": EXTRACT_SCHEMA,
        "status": "complete",
        "artifact_kind": "immutable_engineering_smoke",
        "scope": (
            "frozen Lucy 5B measurement-chain validation only; not Motive "
            "paper reproduction and never representation promotion evidence"
        ),
        "model": {
            "id": "lucy_edit_1_1",
            "registry_role": "failed_baseline",
            "primary_eligible": False,
            "model_path": str(args.model_path.resolve()),
            "transformer_checkpoint_sha256": checkpoint_digest,
            "selected_weight_sha256_before": selected_weight_digest_before,
            "selected_weight_sha256_after": selected_weight_digest_after,
            "selected_weights_unchanged": weights_unchanged,
        },
        "measurement": {
            "backend": "restricted_full_parameter_gradient_countsketch",
            "paper_reproduction": False,
            "paper_difference": [
                "restricted late attention parameter subspaces, not full generator",
                "CountSketch, not Fastfood",
                "paired and factorial action-editing quotient are project extensions",
            ],
            "factorial_cells": list(CELL_NAMES),
            "tangents": list(TANGENT_NAMES),
            "objective": "0.25 motion-weighted x0 level + 0.75 motion-weighted temporal x0 difference",
            "source_condition": "channel-concatenated clean source latent",
            "saliency_use": "loss weighting only; never model input or user mask",
            "vae_posterior_mode": "deterministic_argmax",
            "vae_normalization": "(latent-latents_mean)/latents_std",
            "vae_dtype": str(vae_dtype),
            "noise_mode": "same spatial Gaussian broadcast over latent time",
            "diffusion_noise_seed": int(args.diffusion_noise_seed),
            "scheduler_class": pipe.scheduler.__class__.__name__,
            "scheduler_steps": int(args.scheduler_steps),
            "scheduler_midpoint_index": int(scheduler_index),
            "scheduler_timestep": float(timestep),
            "scheduler_sigma": float(sigma),
            "expand_timesteps": expand_timesteps,
            "projection_backend": "countsketch_streaming_raw_before_quotient",
            "projection_dimension_per_role": int(args.projection_dim),
            "projection_seeds": [int(value) for value in args.projection_seeds],
        },
        "parameter_subspaces": parameter_roles,
        "parameter_manifest": parameter_manifest,
        "parameter_manifest_sha256": parameter_manifest_digest,
        "data": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_digest(args.manifest),
            "track_cache": str(args.track_cache.resolve()),
            "track_cache_sha256": file_digest(args.track_cache),
            "rows": len(rows),
            "families": sorted({str(row["family"]) for row in rows}),
            "unique_components": len({str(row["component_id"]) for row in rows}),
            "videos_read": 2 * len(rows),
            "videos_copied": 0,
            "video_outputs_created": 0,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "dtype": str(dtype),
            "width": int(args.width),
            "height": int(args.height),
            "num_frames": int(args.num_frames),
        },
        "implementation": _source_implementation_digest(),
        "source_tree_sha256": str(args.source_tree_sha256),
        "safety": {
            "optimizer_created": False,
            "optimizer_steps": 0,
            "scheduler_steps_executed": 0,
            "renderer_calls": 0,
            "videos_decoded_for_measurement": 2 * len(rows),
            "videos_rendered": 0,
            "videos_copied": 0,
            "checkpoint_mutated": False,
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "editor_training_authorized": False,
        },
        "decision": {
            "technical_smoke_passed": True,
            "representation_gate_passed": False,
            "next_if_valid": (
                "inspect controls and repeat on primary-eligible Bernini-R 1.3B; "
                "Lucy output alone can never trigger expansion or promotion"
            ),
        },
    }

    output_parent = args.output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.work.",
            dir=output_parent,
        )
    )
    try:
        with (work / FEATURES_NAME).open("xb") as handle:
            np.savez_compressed(
                handle,
                ids=identifiers,
                metadata_json=np.asarray(canonical_json(summary["measurement"])),
                **features,
            )
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_write_text(
            work / ROWS_NAME,
            "".join(canonical_json(row) + "\n" for row in row_outputs),
        )
        _atomic_write_text(work / SUMMARY_NAME, _pretty_json(summary))
        payloads = {
            name: {
                "sha256": file_digest(work / name),
                "bytes": int((work / name).stat().st_size),
            }
            for name in PAYLOAD_NAMES
        }
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "complete",
            "payloads": payloads,
            "payload_closure": sorted(PAYLOAD_NAMES),
            "artifact_digest": object_digest(payloads),
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "editor_training_authorized": False,
        }
        _atomic_write_text(work / DONE_NAME, _pretty_json(done))
        _fsync_directory(work)
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        os.replace(work, args.output_dir)
        work = None
        for path in args.output_dir.iterdir():
            path.chmod(0o444)
        args.output_dir.chmod(0o555)
        _fsync_directory(output_parent)
    finally:
        if work is not None and work.exists():
            shutil.rmtree(work)
    return summary


def validate_published_extract(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    if not output.is_dir():
        raise R10BLucyExtractError(f"extract directory is missing: {output}")
    observed_names = sorted(path.name for path in output.iterdir())
    if observed_names != sorted(OUTPUT_NAMES):
        raise R10BLucyExtractError(
            f"extract artifact closure differs: {observed_names}"
        )
    done = json.loads((output / DONE_NAME).read_text(encoding="utf-8"))
    if done.get("schema_version") != DONE_SCHEMA or done.get("status") != "complete":
        raise R10BLucyExtractError("done marker schema/status differs")
    if done.get("payload_closure") != sorted(PAYLOAD_NAMES):
        raise R10BLucyExtractError("done payload closure differs")
    payloads = done.get("payloads")
    if not isinstance(payloads, dict) or set(payloads) != set(PAYLOAD_NAMES):
        raise R10BLucyExtractError("done payload records differ")
    for name in PAYLOAD_NAMES:
        path = output / name
        record = payloads[name]
        if (
            record.get("sha256") != file_digest(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise R10BLucyExtractError(f"payload digest/size differs: {name}")
    if done.get("artifact_digest") != object_digest(payloads):
        raise R10BLucyExtractError("artifact digest differs")
    summary = json.loads((output / SUMMARY_NAME).read_text(encoding="utf-8"))
    if summary.get("schema_version") != EXTRACT_SCHEMA:
        raise R10BLucyExtractError("summary schema differs")
    if summary.get("model", {}).get("selected_weights_unchanged") is not True:
        raise R10BLucyExtractError("summary does not prove unchanged weights")
    safety = summary.get("safety", {})
    required_false = (
        "optimizer_created",
        "checkpoint_mutated",
        "representation_gate_passed",
        "renderer_probe_authorized",
        "editor_training_authorized",
    )
    if any(safety.get(name) is not False for name in required_false):
        raise R10BLucyExtractError("safety false gate differs")
    if safety.get("optimizer_steps") != 0 or safety.get("renderer_calls") != 0:
        raise R10BLucyExtractError("forbidden optimizer/renderer activity recorded")
    with np.load(output / FEATURES_NAME, allow_pickle=False) as archive:
        if "ids" not in archive.files or "metadata_json" not in archive.files:
            raise R10BLucyExtractError("features archive metadata closure differs")
        ids = np.asarray(archive["ids"]).astype(str)
        feature_names = sorted(set(archive.files) - {"ids", "metadata_json"})
        if not feature_names:
            raise R10BLucyExtractError("features archive contains no representations")
        for name in feature_names:
            values = np.asarray(archive[name])
            if values.ndim != 2 or len(values) != len(ids):
                raise R10BLucyExtractError(f"feature shape differs: {name}")
            if not np.isfinite(values).all():
                raise R10BLucyExtractError(f"feature is non-finite: {name}")
    rows = read_jsonl(output / ROWS_NAME)
    if len(rows) != len(ids) or [row["iid"] for row in rows] != ids.tolist():
        raise R10BLucyExtractError("row/id order differs")
    return {
        "status": "VALID",
        "output_dir": str(output.resolve()),
        "artifact_digest": done["artifact_digest"],
        "rows": len(rows),
        "feature_arrays": len(feature_names),
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract or validate frozen Lucy R10B tangent smoke."
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--track-cache", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--block-index", type=int)
    parser.add_argument("--include-cross-kv", action="store_true")
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument(
        "--projection-seeds",
        type=int,
        nargs="+",
        default=[260108847, 260108848],
    )
    parser.add_argument("--diffusion-noise-seed", type=int, default=260108849)
    parser.add_argument("--torch-seed", type=int, default=260108850)
    parser.add_argument("--scheduler-steps", type=int, default=50)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-tree-sha256", default="")
    args = parser.parse_args()
    if not args.validate_only:
        for name in ("model_path", "manifest", "track_cache"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        if (
            len(args.source_tree_sha256) != 64
            or any(character not in "0123456789abcdef" for character in args.source_tree_sha256)
        ):
            parser.error("--source-tree-sha256 must be one lowercase SHA-256")
    return args


def main() -> None:
    args = _parse_args()
    if args.validate_only:
        result = validate_published_extract(args.output_dir)
    else:
        extract(args)
        result = validate_published_extract(args.output_dir)
    print(canonical_json(result))


if __name__ == "__main__":
    main()

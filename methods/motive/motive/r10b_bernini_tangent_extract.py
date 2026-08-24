"""Extract Motive-aligned action-editing tangents from Bernini-R 1.3B.

This backend follows Bernini's native V2V conditioning contract:

* the clean source video is patch-embedded as a source-id-1 token segment;
* the noisy measurement target is patch-embedded as source-id 0;
* source and target token segments are concatenated for self-attention;
* the user instruction is the only external edit control.

The motion saliency is used only to weight the attribution loss.  It is never
passed to Bernini as a mask or other model input.  No optimizer, parameter
update, denoising loop, decoder, renderer, or video output exists here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import html
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import types
from typing import Any, Mapping

import numpy as np

from .attribution import (
    factorial_edit_tangents,
    normalize_projected_tangent,
    selected_parameter_manifest,
)
from .r10b_lucy_tangent_extract import (
    _atomic_write_text,
    _fsync_directory,
    _load_track_arrays,
    _project_current_gradients,
    _uniform_x0_loss,
    _weight_digest,
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


# The immutable v1 engineering smoke used the tokenizer's legacy regex.
# Corrected scientific-pilot artifacts use v2 and must never reinterpret v1.
EXTRACT_SCHEMA = "motive-r10b-bernini-r13-frozen-tangent-extract-v2"
ROW_SCHEMA = "motive-r10b-bernini-r13-frozen-tangent-row-v2"
DONE_SCHEMA = "motive-r10b-bernini-r13-frozen-tangent-done-v2"
FEATURES_NAME = "features.npz"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
PAYLOAD_NAMES = (FEATURES_NAME, ROWS_NAME, SUMMARY_NAME)
OUTPUT_NAMES = (*PAYLOAD_NAMES, DONE_NAME)
CELL_NAMES = ("tc", "sc", "t0", "s0")
ROLE_NAMES = ("self_motion", "cross_instruction")
ARTIFACT_KINDS = ("engineering_smoke", "controlled_retrieval_pilot")
NOISE_MODES = ("temporal_broadcast", "iid_spatiotemporal")
RESIZE_MODES = ("exact_technical", "aspect_preserving_center_crop")
DEFAULT_SCHEDULER_STEPS = 50
DEFAULT_SCHEDULER_INDEX = 25
TANGENT_NAMES = (
    "target_only",
    "paired_delta",
    "factorial_did",
    "cross_cell_control",
    "noop_target_delta",
)
EXPECTED_REPO_REVISION = "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
EXPECTED_SOURCE_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "39336cadc06b68dd04d9c218f4a6465779055c1b85945ae1dcaef95f818eeb08"
)
PROMPT_MODE = (
    "official_generic_v2v_system_prefix_no_prompt_enhancement_"
    "tokenizer_regex_fixed_v2"
)
V2V_SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in video editing."
)
TOKENIZER_FIX_MISTRAL_REGEX = True
TOKENIZER_FIX_RATIONALE = (
    "Transformers 5.5.4 warned that the bundled tokenizer used the legacy "
    "incorrect Mistral regex, and the default versus corrected tokenizer "
    "produced different token IDs for every audited smoke/canonical prompt."
)
TOKENIZER_FIX_COMPATIBILITY = (
    "Pass fix_mistral_regex=True explicitly on every load; tokenizer "
    "implementations that do not accept or apply this option must fail "
    "instead of silently falling back."
)
TOKENIZER_CONTRACT = {
    "fix_mistral_regex": TOKENIZER_FIX_MISTRAL_REGEX,
    "rationale": TOKENIZER_FIX_RATIONALE,
    "version_compatibility": TOKENIZER_FIX_COMPATIBILITY,
}
# Exact sizes and Git-blob/LFS identities returned by the Hugging Face model
# API for EXPECTED_REPO_REVISION.  This pins the downloaded bytes rather than
# trusting mutable local ``.metadata`` revision labels.
EXPECTED_CHECKPOINT_FILES = {
    "config.json": (347, "d26094dce449fa4a83a22b2f79600a9881962b11"),
    "scheduler/scheduler_config.json": (
        751,
        "7c6d4f1916fe490af29216f633d38cfa3028fd66",
    ),
    "text_encoder/config.json": (
        854,
        "2fd01c57dddd8ae386d518c69c087c8ba8c73804",
    ),
    "text_encoder/model-00001-of-00005.safetensors": (
        4_972_389_712,
        "c0ef3a140898e228a3520c9adec60743d2e8e5b3d229651bb37f1a3921919f99",
    ),
    "text_encoder/model-00002-of-00005.safetensors": (
        4_899_225_672,
        "481c7b2b39771c44df6dd8d13ee12ed072d731b4a650bd092885d4d52db229ad",
    ),
    "text_encoder/model-00003-of-00005.safetensors": (
        4_966_309_504,
        "f93148bcc04052a169e1e49bfcf6125df6cf9bf243cb9c627da75266cf8e35c3",
    ),
    "text_encoder/model-00004-of-00005.safetensors": (
        4_999_880_704,
        "a451792c739c05bca4606190cc2dd16731411bac03b4cf6aacc5767321f857c9",
    ),
    "text_encoder/model-00005-of-00005.safetensors": (
        2_885_866_152,
        "7e76e18d224531b8197a46231cb53daf7f2f6ca707130252becf933026ac4eea",
    ),
    "text_encoder/model.safetensors.index.json": (
        22_476,
        "60ece61b46ecb3e6a5b705ea304bc97535317c2a",
    ),
    "tokenizer/special_tokens_map.json": (
        7_079,
        "2ed25bf989a28d20b5d4b5822fbc24666d12a6f7",
    ),
    "tokenizer/spiece.model": (
        4_548_313,
        "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
    ),
    "tokenizer/tokenizer.json": (
        16_837_459,
        "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef",
    ),
    "tokenizer/tokenizer_config.json": (
        61_758,
        "09d434f9457238f697f4c208aab47f58caa15bfe",
    ),
    "transformer/config.json": (
        494,
        "992181cd29bd54f9fce3e7b760f69e16f9afde64",
    ),
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": (
        5_387_151_416,
        "4170c80a25e7e0f58a6728cfdc33bba84033090196841c09656e6935afb8642f",
    ),
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": (
        288_919_192,
        "7d5bef54ec13a33391ecd3c82dcca2c6b71610cda7e8a7485417e30d111f6406",
    ),
    "transformer/diffusion_pytorch_model.safetensors.index.json": (
        76_607,
        "e37efe5d56e320c3dc4cb93c83352439f9c410cc",
    ),
    "vae/config.json": (
        724,
        "fe988ee53511225fb2fd0a01004d6e19524df75f",
    ),
    "vae/diffusion_pytorch_model.safetensors": (
        507_591_892,
        "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793",
    ),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class R10BBerniniExtractError(RuntimeError):
    """Bernini cannot satisfy the frozen tangent measurement contract."""


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


def _tokenizer_provenance(transformers_version: str) -> dict[str, Any]:
    if not isinstance(transformers_version, str) or not transformers_version:
        raise R10BBerniniExtractError(
            "transformers version is unavailable for tokenizer provenance"
        )
    return {
        **TOKENIZER_CONTRACT,
        "contract_sha256": object_digest(TOKENIZER_CONTRACT),
        "transformers_version": transformers_version,
    }


def _load_fixed_tokenizer(
    auto_tokenizer_class: Any,
    model_path: Path,
) -> Any:
    return auto_tokenizer_class.from_pretrained(
        str(model_path),
        subfolder="tokenizer",
        local_files_only=True,
        fix_mistral_regex=TOKENIZER_FIX_MISTRAL_REGEX,
    )


def _validate_tokenizer_provenance(
    summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    runtime = summary.get("runtime", {})
    transformers_version = runtime.get("transformers_version")
    expected = _tokenizer_provenance(transformers_version)
    prompt_conditioning = summary.get("measurement", {}).get(
        "prompt_conditioning",
        {},
    )
    if (
        prompt_conditioning.get("mode") != PROMPT_MODE
        or prompt_conditioning.get("tokenizer") != expected
    ):
        raise R10BBerniniExtractError(
            "Bernini tokenizer provenance differs"
        )
    expected_digest = object_digest(TOKENIZER_CONTRACT)
    for row in rows:
        conditioning = row.get("prompt_conditioning", {})
        if (
            conditioning.get("tokenizer_fix_mistral_regex") is not True
            or conditioning.get("tokenizer_contract_sha256")
            != expected_digest
        ):
            raise R10BBerniniExtractError(
                "Bernini row tokenizer provenance differs"
            )


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean_prompt(value: str) -> str:
    """Apply the released Bernini renderer's prompt cleaning."""

    try:
        import ftfy
    except ImportError as error:
        raise R10BBerniniExtractError(
            "ftfy is required for official Bernini prompt cleaning"
        ) from error
    return re.sub(
        r"\s+",
        " ",
        html.unescape(html.unescape(ftfy.fix_text(str(value)))),
    ).strip()


def _effective_v2v_prompt(value: str) -> str:
    cleaned = _clean_prompt(value)
    if not cleaned:
        raise R10BBerniniExtractError("Bernini prompt must be non-empty")
    # This intentionally matches the released renderer pipeline's direct
    # ``system_prompt + cleaned_prompt`` concatenation (without PE).
    return V2V_SYSTEM_PROMPT + cleaned


def _load_root_config(model_path: Path) -> dict[str, Any]:
    path = model_path / "config.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R10BBerniniExtractError(
            f"cannot read Bernini root config: {path}"
        ) from error
    required = {
        "model_type": "bernini_renderer",
        "skip_transformer_1": False,
        "skip_transformer_2": True,
        "use_src_id_rotary_emb": True,
    }
    differences = {
        key: {"expected": expected, "observed": value.get(key)}
        for key, expected in required.items()
        if value.get(key) != expected
    }
    if differences:
        raise R10BBerniniExtractError(
            f"Bernini root config differs from the 1.3B V2V contract: {differences}"
        )
    if float(value.get("shift", -1.0)) != 3.0:
        raise R10BBerniniExtractError("Bernini scheduler shift must be 3.0")
    return value


def _git_blob_digest(path: Path) -> str:
    """Return the SHA-1 object identity used by Git/Hugging Face for a file."""

    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_manifest(model_path: Path) -> dict[str, Any]:
    """Hash and pin every self-contained model file used by measurement."""

    roots = ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")
    paths = [model_path / "config.json"]
    for root in roots:
        component = model_path / root
        if not component.is_dir():
            raise R10BBerniniExtractError(
                f"Bernini checkpoint component is missing: {component}"
            )
        paths.extend(
            path
            for path in sorted(component.rglob("*"))
            if path.is_file()
        )
    observed = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise R10BBerniniExtractError(
                f"checkpoint entry must be a regular file: {path}"
            )
        relative = path.relative_to(model_path).as_posix()
        observed[relative] = path
    expected_names = set(EXPECTED_CHECKPOINT_FILES)
    observed_names = set(observed)
    if observed_names != expected_names:
        raise R10BBerniniExtractError(
            "checkpoint file closure differs: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )

    rows = []
    for relative in sorted(expected_names):
        path = observed[relative]
        expected_bytes, expected_identity = EXPECTED_CHECKPOINT_FILES[relative]
        observed_bytes = int(path.stat().st_size)
        if observed_bytes != expected_bytes:
            raise R10BBerniniExtractError(
                f"checkpoint size differs: {relative}"
            )
        sha256 = file_digest(path)
        observed_identity = (
            sha256
            if len(expected_identity) == 64
            else _git_blob_digest(path)
        )
        if observed_identity != expected_identity:
            raise R10BBerniniExtractError(
                f"checkpoint content differs: {relative}"
            )
        rows.append(
            {
                "path": relative,
                "bytes": observed_bytes,
                "sha256": sha256,
                "huggingface_identity": expected_identity,
            }
        )
    return {
        "files": rows,
        "file_count": len(rows),
        "total_bytes": int(sum(row["bytes"] for row in rows)),
        "tree_sha256": object_digest(rows),
    }


def _checkpoint_revision_metadata(model_path: Path) -> dict[str, Any]:
    """Bind local download metadata to every pinned checkpoint file."""

    metadata_root = model_path / ".cache/huggingface/download"
    rows = []
    for relative, (_expected_bytes, expected_identity) in sorted(
        EXPECTED_CHECKPOINT_FILES.items()
    ):
        path = metadata_root / f"{relative}.metadata"
        if path.is_symlink() or not path.is_file():
            raise R10BBerniniExtractError(
                f"Hugging Face revision metadata is missing: {path}"
            )
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise R10BBerniniExtractError(
                f"cannot read checkpoint metadata: {path}"
            ) from error
        if len(lines) < 2 or _GIT_COMMIT_RE.fullmatch(lines[0]) is None:
            raise R10BBerniniExtractError(
                f"checkpoint metadata has no valid revision/identity: {path}"
            )
        revision = lines[0]
        identity = lines[1]
        if revision != EXPECTED_REPO_REVISION or identity != expected_identity:
            raise R10BBerniniExtractError(
                f"checkpoint metadata differs: {relative}"
            )
        rows.append(
            {
                "path": relative,
                "revision": revision,
                "huggingface_identity": identity,
            }
        )
    return {
        "metadata_files": len(rows),
        "revision": EXPECTED_REPO_REVISION,
        "manifest_sha256": object_digest(rows),
    }


def _bernini_source_manifest(repo: Path) -> dict[str, Any]:
    """Bind the official source bytes imported by the backend."""

    required = (
        "bernini/__init__.py",
        "bernini/attention.py",
        "bernini/cli.py",
        "bernini/data/bernini_template.py",
        "bernini/models/__init__.py",
        "bernini/models/transformer_wan.py",
        "bernini/parallel/__init__.py",
        "bernini/parallel/ops.py",
        "bernini/parallel/state.py",
        "bernini/pipeline.py",
        "bernini/prompt_enhancer.py",
    )
    rows = []
    for relative in required:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise R10BBerniniExtractError(
                f"official Bernini source is missing: {path}"
            )
        rows.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": file_digest(path),
            }
        )
    manifest = {
        "files": rows,
        "bundle_sha256": object_digest(rows),
    }
    if manifest["bundle_sha256"] != EXPECTED_SOURCE_BUNDLE_SHA256:
        raise R10BBerniniExtractError(
            "official Bernini source bytes differ from the preregistered commit"
        )
    return manifest


def _install_bernini_import_path(repo: Path) -> None:
    repo = repo.resolve()
    if not (repo / "bernini/models/transformer_wan.py").is_file():
        raise R10BBerniniExtractError(
            f"official Bernini source tree is invalid: {repo}"
        )
    repo_string = str(repo)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)


def _load_official_transformer_class(repo: Path) -> Any:
    """Load the official transformer without importing unrelated Bernini-LLM.

    The upstream ``bernini.models.__init__`` eagerly imports the unified
    Qwen/VeOmni model even though the released Diffusers renderer only needs
    ``transformer_wan.py``.  That makes the direct submodule import depend on
    VeOmni.  We leave the official tree untouched and create only the package
    shell needed for Python's relative imports; all executed implementation
    files are bound by ``_bernini_source_manifest``.
    """

    _install_bernini_import_path(repo)
    root_package = importlib.import_module("bernini")
    expected_root = (repo.resolve() / "bernini/__init__.py").resolve()
    observed_root = Path(str(getattr(root_package, "__file__", ""))).resolve()
    if observed_root != expected_root:
        raise R10BBerniniExtractError(
            "imported bernini root does not come from the fixed vendor tree: "
            f"{observed_root} vs {expected_root}"
        )
    forbidden = (
        "bernini.models",
        "bernini.models.bernini",
        "bernini.models.modeling_qwen2_5_vl",
        "veomni",
    )
    preloaded = [name for name in forbidden if name in sys.modules]
    if preloaded:
        raise R10BBerniniExtractError(
            "unrelated Bernini/VeOmni modules were imported before the "
            f"restricted official loader: {preloaded}"
        )
    package_path = repo.resolve() / "bernini/models"
    package = types.ModuleType("bernini.models")
    package.__package__ = "bernini.models"
    package.__path__ = [str(package_path)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        name="bernini.models",
        loader=None,
        is_package=True,
    )
    package.__spec__.submodule_search_locations = [str(package_path)]
    sys.modules["bernini.models"] = package
    setattr(sys.modules["bernini"], "models", package)

    module_name = "bernini.models.transformer_wan"
    module_path = package_path / "transformer_wan.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise R10BBerniniExtractError(
            f"cannot construct official transformer loader: {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    transformer_class = getattr(module, "WanTransformer3DModel", None)
    if transformer_class is None:
        raise R10BBerniniExtractError(
            "official source exposes no WanTransformer3DModel"
        )
    accidentally_loaded = [
        name for name in forbidden[1:] if name in sys.modules
    ]
    if accidentally_loaded:
        raise R10BBerniniExtractError(
            "restricted official loader imported forbidden modules: "
            f"{accidentally_loaded}"
        )
    attention = importlib.import_module("bernini.attention")
    attention._BACKEND = "sdpa"
    attention._flash_varlen = None
    return transformer_class


def _encode_vae_latents(
    vae: Any,
    processor: Any,
    frames: list[Any],
    *,
    height: int,
    width: int,
    device: Any,
    output_dtype: Any,
) -> Any:
    torch = _torch()
    video = processor.preprocess_video(
        frames,
        height=height,
        width=width,
    ).to(device=device, dtype=vae.dtype)
    if video.ndim != 5:
        raise R10BBerniniExtractError(
            f"Bernini video processor returned {tuple(video.shape)}"
        )
    encoded = vae.encode(video)
    if not hasattr(encoded, "latent_dist"):
        raise R10BBerniniExtractError("Bernini VAE exposes no latent_dist")
    latents = encoded.latent_dist.mode()
    z_dim = int(getattr(vae.config, "z_dim", latents.shape[1]))
    mean = torch.tensor(
        vae.config.latents_mean,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(
        vae.config.latents_std,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, z_dim, 1, 1, 1)
    if mean.shape[1] != latents.shape[1] or float(std.abs().min()) <= 0:
        raise R10BBerniniExtractError("Bernini latent normalization is invalid")
    normalized = (latents - mean) / std
    if not bool(torch.isfinite(normalized).all()):
        raise R10BBerniniExtractError("Bernini VAE produced non-finite latents")
    return normalized.to(dtype=output_dtype)


def _encode_prompt(
    tokenizer: Any,
    text_encoder: Any,
    prompt: str,
    *,
    device: Any,
    dtype: Any,
    max_sequence_length: int,
) -> Any:
    torch = _torch()
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    hidden = text_encoder(input_ids, attention_mask).last_hidden_state
    valid = min(int(attention_mask[0].sum()), max_sequence_length)
    embedding = hidden[:, :valid]
    if valid < max_sequence_length:
        embedding = torch.cat(
            (
                embedding,
                embedding.new_zeros(
                    1,
                    max_sequence_length - valid,
                    embedding.shape[-1],
                ),
            ),
            dim=1,
        )
    embedding = embedding.to(dtype=dtype)
    if embedding.shape[1] != max_sequence_length:
        raise R10BBerniniExtractError("Bernini prompt length differs")
    if not bool(torch.isfinite(embedding).all()):
        raise R10BBerniniExtractError("Bernini prompt embedding is non-finite")
    return embedding


def _scheduler_point(
    scheduler: Any,
    *,
    steps: int,
    index: int,
    device: Any,
) -> tuple[Any, float, int]:
    if steps < 3:
        raise R10BBerniniExtractError("scheduler_steps must be at least three")
    if isinstance(index, bool) or not isinstance(index, int):
        raise R10BBerniniExtractError("scheduler_index must be one integer")
    scheduler.set_timesteps(steps, device=device)
    if len(scheduler.timesteps) != steps:
        raise R10BBerniniExtractError(
            "Bernini scheduler timestep count differs from scheduler_steps"
        )
    if index < 0 or index >= len(scheduler.timesteps):
        raise R10BBerniniExtractError(
            "scheduler_index is out of range: "
            f"{index} for {len(scheduler.timesteps)} timesteps"
        )
    timestep = scheduler.timesteps[index]
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or len(sigmas) <= index:
        raise R10BBerniniExtractError("Bernini scheduler has no aligned sigma")
    sigma = float(sigmas[index])
    if not math.isfinite(sigma) or not 0.0 < sigma < 1.0:
        raise R10BBerniniExtractError(
            f"Bernini native scheduler sigma must be in (0,1), got {sigma}"
        )
    return timestep, sigma, index


def _noise_for_mode(
    reference: Any,
    *,
    seed: int,
    mode: str,
) -> Any:
    """Generate a deterministic noise realization under an explicit mode."""

    torch = _torch()
    if reference.ndim != 5 or reference.shape[2] < 2:
        raise R10BBerniniExtractError(
            "latent reference must have shape [B,C,T,H,W]"
        )
    if mode == "temporal_broadcast":
        noise = temporal_broadcast_noise(reference, seed=seed)
    elif mode == "iid_spatiotemporal":
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            tuple(reference.shape),
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        raise R10BBerniniExtractError(f"unknown noise_mode: {mode}")
    if noise.shape != reference.shape or not bool(torch.isfinite(noise).all()):
        raise R10BBerniniExtractError(
            "Bernini noise shape/finite contract differs"
        )
    return noise


def _resize_transform(
    input_width: int,
    input_height: int,
    *,
    output_width: int,
    output_height: int,
    mode: str,
) -> dict[str, Any]:
    """Return the exact deterministic resize/crop geometry for one frame."""

    values = (input_width, input_height, output_width, output_height)
    if any(isinstance(value, bool) or int(value) <= 0 for value in values):
        raise R10BBerniniExtractError("frame dimensions must be positive")
    if mode == "exact_technical":
        return {
            "resized_width": int(output_width),
            "resized_height": int(output_height),
            "crop_left": 0,
            "crop_top": 0,
            "crop_right": int(output_width),
            "crop_bottom": int(output_height),
        }
    if mode != "aspect_preserving_center_crop":
        raise R10BBerniniExtractError(f"unknown resize_mode: {mode}")
    scale = max(
        float(output_width) / float(input_width),
        float(output_height) / float(input_height),
    )
    resized_width = max(int(output_width), int(math.ceil(input_width * scale)))
    resized_height = max(
        int(output_height),
        int(math.ceil(input_height * scale)),
    )
    crop_left = (resized_width - int(output_width)) // 2
    crop_top = (resized_height - int(output_height)) // 2
    return {
        "resized_width": resized_width,
        "resized_height": resized_height,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_right": crop_left + int(output_width),
        "crop_bottom": crop_top + int(output_height),
    }


def _resize_frame(
    frame: Any,
    *,
    width: int,
    height: int,
    mode: str,
) -> tuple[Any, dict[str, Any]]:
    from PIL import Image

    if not isinstance(frame, Image.Image):
        frame = Image.fromarray(np.asarray(frame))
    frame = frame.convert("RGB")
    input_width, input_height = frame.size
    transform = _resize_transform(
        input_width,
        input_height,
        output_width=width,
        output_height=height,
        mode=mode,
    )
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = frame.resize(
        (transform["resized_width"], transform["resized_height"]),
        resampling,
    )
    if mode == "aspect_preserving_center_crop":
        resized = resized.crop(
            (
                transform["crop_left"],
                transform["crop_top"],
                transform["crop_right"],
                transform["crop_bottom"],
            )
        )
    if resized.size != (width, height):
        raise R10BBerniniExtractError(
            f"resized frame shape differs: {resized.size}"
        )
    return resized, {
        "input_width": int(input_width),
        "input_height": int(input_height),
        **transform,
    }


def _decode_video_frames(path: Path) -> list[Any]:
    try:
        from diffusers.utils import load_video

        frames = list(load_video(str(path)))
    except Exception:
        import imageio.v3 as iio

        frames = list(iio.imiter(path))
    if not frames:
        raise R10BBerniniExtractError(f"video has no decodable frames: {path}")
    return frames


def _uniform_frame_indices(frame_count: int, num_frames: int) -> list[int]:
    if (
        isinstance(frame_count, bool)
        or isinstance(num_frames, bool)
        or frame_count < num_frames
        or num_frames < 2
    ):
        raise R10BBerniniExtractError(
            f"cannot sample {num_frames} frames from {frame_count}"
        )
    if frame_count == num_frames:
        return list(range(num_frames))
    return [
        round(index * (frame_count - 1) / (num_frames - 1))
        for index in range(num_frames)
    ]


def _load_paired_video_frames(
    source_path: Path,
    target_path: Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    resize_mode: str,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Decode a pair and apply one shared list of absolute frame indices."""

    source_raw = _decode_video_frames(source_path)
    target_raw = _decode_video_frames(target_path)
    shared_frame_count = min(len(source_raw), len(target_raw))
    indices = _uniform_frame_indices(shared_frame_count, num_frames)
    source_frames = []
    target_frames = []
    source_transforms = []
    target_transforms = []
    for index in indices:
        source_frame, source_transform = _resize_frame(
            source_raw[index],
            width=width,
            height=height,
            mode=resize_mode,
        )
        target_frame, target_transform = _resize_frame(
            target_raw[index],
            width=width,
            height=height,
            mode=resize_mode,
        )
        source_frames.append(source_frame)
        target_frames.append(target_frame)
        source_transforms.append(source_transform)
        target_transforms.append(target_transform)
    return source_frames, target_frames, {
        "resize_mode": resize_mode,
        "source_decoded_frame_count": int(len(source_raw)),
        "target_decoded_frame_count": int(len(target_raw)),
        "shared_sampling_frame_count": int(shared_frame_count),
        "selected_frame_indices": [int(value) for value in indices],
        "source_target_frame_indices_identical": True,
        "output_width": int(width),
        "output_height": int(height),
        "source_spatial_transforms": source_transforms,
        "target_spatial_transforms": target_transforms,
        "source_spatial_transforms_sha256": object_digest(
            source_transforms
        ),
        "target_spatial_transforms_sha256": object_digest(
            target_transforms
        ),
    }


def _pack_source_and_target(
    transformer: Any,
    source_latents: Any,
    noisy_target_latents: Any,
) -> tuple[Any, Any, Any, list[int]]:
    """Build Bernini's native source-id-1 + target-id-0 token sequence."""

    torch = _torch()
    source_tokens, source_rope = transformer.patch_vae_latent(
        source_latents.to(dtype=transformer.dtype),
        source_id=1,
    )
    target_tokens, target_rope = transformer.patch_vae_latent(
        noisy_target_latents.to(dtype=transformer.dtype),
        source_id=0,
    )
    if source_tokens.shape[0] != 1 or target_tokens.shape[0] != 1:
        raise R10BBerniniExtractError("Bernini smoke requires batch size one")
    hidden = torch.cat((source_tokens, target_tokens), dim=1)
    rotary = torch.cat((source_rope, target_rope), dim=2)
    target_mask = torch.cat(
        (
            torch.zeros(
                source_tokens.shape[1],
                device=hidden.device,
                dtype=torch.bool,
            ),
            torch.ones(
                target_tokens.shape[1],
                device=hidden.device,
                dtype=torch.bool,
            ),
        )
    )
    return hidden, rotary, target_mask, [int(hidden.shape[1])]


def _unpack_target_prediction(
    packed: Any,
    reference: Any,
    *,
    patch_size: tuple[int, int, int],
) -> Any:
    """Invert Bernini's ``(t h w) × (pt ph pw c)`` output packing."""

    if packed.ndim != 3 or reference.ndim != 5:
        raise R10BBerniniExtractError("packed/reference tensor ranks differ")
    batch, channels, frames, height, width = reference.shape
    pt, ph, pw = (int(value) for value in patch_size)
    if frames % pt or height % ph or width % pw:
        raise R10BBerniniExtractError("reference shape is not patch divisible")
    tp, hp, wp = frames // pt, height // ph, width // pw
    expected_tokens = tp * hp * wp
    expected_features = pt * ph * pw * channels
    if tuple(packed.shape) != (batch, expected_tokens, expected_features):
        raise R10BBerniniExtractError(
            "Bernini packed prediction shape differs: "
            f"{tuple(packed.shape)} vs "
            f"{(batch, expected_tokens, expected_features)}"
        )
    values = packed.reshape(batch, tp, hp, wp, pt, ph, pw, channels)
    values = values.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return values.reshape(batch, channels, frames, height, width)


def _transformer_prediction(
    transformer: Any,
    *,
    noisy_latents: Any,
    source_latents: Any,
    prompt_embeds: Any,
    timestep: Any,
) -> Any:
    hidden, rotary, target_mask, batch_lengths = _pack_source_and_target(
        transformer,
        source_latents,
        noisy_latents,
    )
    output = transformer(
        hidden,
        timestep.expand(1),
        encoder_hidden_states=prompt_embeds,
        rotary_emb=rotary,
        batch_image_vae_seqlen=batch_lengths,
        text_features_length=[int(prompt_embeds.shape[1])],
        return_dict=False,
    )
    packed = output[0] if isinstance(output, (tuple, list)) else output.sample
    packed = packed[:, target_mask, :]
    prediction = _unpack_target_prediction(
        packed,
        noisy_latents,
        patch_size=tuple(transformer.config.patch_size),
    )
    if not bool(_torch().isfinite(prediction).all()):
        raise R10BBerniniExtractError(
            "Bernini transformer prediction is non-finite"
        )
    return prediction


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
    noise_mode: str,
    projection_seeds: tuple[int, ...],
    projection_dim: int,
    objective: str,
) -> tuple[dict[tuple[str, int], Any], dict[str, Any]]:
    torch = _torch()
    transformer.zero_grad(set_to_none=True)
    noise = _noise_for_mode(
        clean_latents,
        seed=diffusion_noise_seed,
        mode=noise_mode,
    )
    noisy = (1.0 - sigma) * clean_latents + sigma * noise
    prediction = _transformer_prediction(
        transformer,
        noisy_latents=noisy,
        source_latents=source_latents,
        prompt_embeds=prompt_embeds,
        timestep=timestep,
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
        raise R10BBerniniExtractError(f"unknown objective: {objective}")
    if not bool(torch.isfinite(loss)) or float(loss.detach()) <= 0:
        raise R10BBerniniExtractError(
            f"measurement loss is invalid: {loss_metrics}"
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


def _track_saliency_for_row(
    row: Mapping[str, Any],
    track_arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_index = int(row["track_cache_index"])
    if cache_index >= len(track_arrays["input_indices"]):
        raise R10BBerniniExtractError(
            f"track_cache_index is out of range for iid={row['iid']}"
        )
    if int(track_arrays["input_indices"][cache_index]) != int(
        row["track_input_index"]
    ):
        raise R10BBerniniExtractError(
            f"track input/cache binding differs for iid={row['iid']}"
        )
    motion_mask, saliency_metrics = track_delta_saliency(
        track_arrays["source_stabilized_tracks"][cache_index],
        track_arrays["target_stabilized_tracks"][cache_index],
        track_arrays["source_visibility"][cache_index],
        track_arrays["target_visibility"][cache_index],
    )
    if saliency_metrics["normalized_active_fraction"] <= 0:
        raise R10BBerniniExtractError(
            f"motion saliency is empty for iid={row['iid']}"
        )
    return motion_mask, saliency_metrics


def _preencode_cases(
    *,
    vae: Any,
    video_processor: Any,
    tokenizer: Any,
    text_encoder: Any,
    rows: list[dict[str, Any]],
    track_arrays: Mapping[str, np.ndarray],
    width: int,
    height: int,
    num_frames: int,
    device: Any,
    dtype: Any,
    max_sequence_length: int,
    resize_mode: str,
) -> list[dict[str, Any]]:
    media = []
    for row in rows:
        try:
            data_root = Path(str(row["data_root"])).expanduser().resolve(
                strict=True
            )
        except OSError as error:
            raise R10BBerniniExtractError(
                f"data_root is unavailable for iid={row['iid']}"
            ) from error
        motion_mask, saliency_metrics = _track_saliency_for_row(
            row,
            track_arrays,
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
                raise R10BBerniniExtractError(
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
                raise R10BBerniniExtractError(f"{name} media missing: {path}")
            if file_digest(path) != expected_digest:
                raise R10BBerniniExtractError(
                    f"{name} media digest differs for iid={row['iid']}"
                )
        (
            source_frames,
            target_frames,
            media_preprocessing,
        ) = _load_paired_video_frames(
            source_path,
            target_path,
            width=width,
            height=height,
            num_frames=num_frames,
            resize_mode=resize_mode,
        )
        media.append(
            (
                row,
                source_frames,
                target_frames,
                motion_mask,
                saliency_metrics,
                media_preprocessing,
            )
        )

    torch = _torch()
    vae.to(device)
    vae.eval()
    encoded = []
    with torch.no_grad():
        for (
            row,
            source_frames,
            target_frames,
            motion_mask,
            saliency_metrics,
            media_preprocessing,
        ) in media:
            source_latents = _encode_vae_latents(
                vae,
                video_processor,
                source_frames,
                height=height,
                width=width,
                device=device,
                output_dtype=dtype,
            )
            target_latents = _encode_vae_latents(
                vae,
                video_processor,
                target_frames,
                height=height,
                width=width,
                device=device,
                output_dtype=dtype,
            )
            if source_latents.shape != target_latents.shape:
                raise R10BBerniniExtractError(
                    f"source/target latent shape differs for iid={row['iid']}"
                )
            encoded.append(
                {
                    "row": row,
                    "source_latents": source_latents.detach().cpu(),
                    "target_latents": target_latents.detach().cpu(),
                    "motion_mask": motion_mask,
                    "saliency": saliency_metrics,
                    "media_preprocessing": media_preprocessing,
                }
            )
    vae.to("cpu")
    torch.cuda.empty_cache()

    text_encoder.to(device)
    text_encoder.eval()
    with torch.no_grad():
        for case in encoded:
            row = case["row"]
            effective_prompt = _effective_v2v_prompt(str(row["prompt"]))
            effective_noop_prompt = _effective_v2v_prompt(
                str(row["noop_prompt"])
            )
            case["prompt_embeds"] = _encode_prompt(
                tokenizer,
                text_encoder,
                effective_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
            ).detach().cpu()
            case["noop_embeds"] = _encode_prompt(
                tokenizer,
                text_encoder,
                effective_noop_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
            ).detach().cpu()
            case["prompt_conditioning"] = {
                "mode": PROMPT_MODE,
                "tokenizer_fix_mistral_regex": TOKENIZER_FIX_MISTRAL_REGEX,
                "tokenizer_contract_sha256": object_digest(
                    TOKENIZER_CONTRACT
                ),
                "raw_prompt_sha256": _text_digest(str(row["prompt"])),
                "raw_noop_prompt_sha256": _text_digest(
                    str(row["noop_prompt"])
                ),
                "effective_prompt_sha256": _text_digest(effective_prompt),
                "effective_noop_prompt_sha256": _text_digest(
                    effective_noop_prompt
                ),
            }
    text_encoder.to("cpu")
    torch.cuda.empty_cache()
    return encoded


def _source_implementation_digest() -> dict[str, Any]:
    files = {}
    for path in (
        Path(__file__),
        Path(__file__).with_name("r10b_lucy_tangent_extract.py"),
        Path(__file__).with_name("r10b_tangent_core.py"),
        Path(__file__).with_name("attribution.py"),
    ):
        files[path.name] = file_digest(path)
    return {
        "files": dict(sorted(files.items())),
        "bundle_sha256": object_digest(dict(sorted(files.items()))),
    }


def _build_run_contract(
    *,
    artifact_kind: str,
    scheduler_class: str,
    scheduler_steps: int,
    scheduler_index: int,
    scheduler_timestep: float,
    scheduler_sigma: float,
    noise_mode: str,
    diffusion_noise_seed: int,
    resize_mode: str,
    width: int,
    height: int,
    num_frames: int,
) -> dict[str, Any]:
    if artifact_kind not in ARTIFACT_KINDS:
        raise R10BBerniniExtractError(
            f"unknown artifact_kind: {artifact_kind}"
        )
    if noise_mode not in NOISE_MODES:
        raise R10BBerniniExtractError(f"unknown noise_mode: {noise_mode}")
    if resize_mode not in RESIZE_MODES:
        raise R10BBerniniExtractError(f"unknown resize_mode: {resize_mode}")
    if (
        artifact_kind == "controlled_retrieval_pilot"
        and resize_mode != "aspect_preserving_center_crop"
    ):
        raise R10BBerniniExtractError(
            "controlled_retrieval_pilot requires "
            "aspect_preserving_center_crop"
        )
    integer_values = {
        "scheduler_steps": scheduler_steps,
        "scheduler_index": scheduler_index,
        "diffusion_noise_seed": diffusion_noise_seed,
        "width": width,
        "height": height,
        "num_frames": num_frames,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_values.values()
    ):
        raise R10BBerniniExtractError(
            "run-contract integer fields must be integers"
        )
    if scheduler_steps < 3 or not 0 <= scheduler_index < scheduler_steps:
        raise R10BBerniniExtractError(
            "run-contract scheduler index/steps are invalid"
        )
    if not isinstance(scheduler_class, str) or not scheduler_class:
        raise R10BBerniniExtractError(
            "run-contract scheduler class is empty"
        )
    if (
        isinstance(scheduler_timestep, bool)
        or not isinstance(scheduler_timestep, (int, float))
        or not math.isfinite(float(scheduler_timestep))
        or isinstance(scheduler_sigma, bool)
        or not isinstance(scheduler_sigma, (int, float))
        or not math.isfinite(float(scheduler_sigma))
        or not 0.0 < float(scheduler_sigma) < 1.0
    ):
        raise R10BBerniniExtractError(
            "run-contract scheduler timestep/sigma are invalid"
        )
    if (
        width <= 0
        or height <= 0
        or width % 16
        or height % 16
        or num_frames < 2
        or (num_frames - 1) % 4
    ):
        raise R10BBerniniExtractError(
            "run-contract spatial/temporal dimensions are invalid"
        )
    temporal_behavior = {
        "temporal_broadcast": (
            "sample_B_C_1_H_W_then_expand_without_new_random_draws"
        ),
        "iid_spatiotemporal": "sample_full_B_C_T_H_W",
    }[noise_mode]
    return {
        "artifact_kind": artifact_kind,
        "scheduler": {
            "class": scheduler_class,
            "steps": int(scheduler_steps),
            "index": int(scheduler_index),
            "selection": "explicit_cli_index",
            "timestep": float(scheduler_timestep),
            "sigma": float(scheduler_sigma),
        },
        "noise": {
            "mode": noise_mode,
            "seed": int(diffusion_noise_seed),
            "generator": "torch.Generator(device=reference.device)",
            "deterministic_seeded": True,
            "output_shape_matches_reference": True,
            "temporal_behavior": temporal_behavior,
        },
        "resize": {
            "mode": resize_mode,
            "width": int(width),
            "height": int(height),
            "num_frames": int(num_frames),
            "sampling": "shared_uniform_absolute_frame_indices",
            "shared_sampling_frame_count_policy": (
                "minimum_source_target_decoded_frame_count"
            ),
            "source_target_frame_indices_identical": True,
            "aspect_ratio_preserved": (
                resize_mode == "aspect_preserving_center_crop"
            ),
            "center_crop": (
                resize_mode == "aspect_preserving_center_crop"
            ),
        },
    }


def _validate_recorded_spatial_transforms(
    values: Any,
    *,
    expected_frames: int,
    resize: Mapping[str, Any],
) -> None:
    if not isinstance(values, list) or len(values) != expected_frames:
        raise R10BBerniniExtractError(
            "Bernini recorded spatial-transform count differs"
        )
    for record in values:
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "input_width",
                "input_height",
                "resized_width",
                "resized_height",
                "crop_left",
                "crop_top",
                "crop_right",
                "crop_bottom",
            }
            or isinstance(record.get("input_width"), bool)
            or not isinstance(record.get("input_width"), int)
            or isinstance(record.get("input_height"), bool)
            or not isinstance(record.get("input_height"), int)
        ):
            raise R10BBerniniExtractError(
                "Bernini recorded spatial-transform schema differs"
            )
        expected = {
            "input_width": record["input_width"],
            "input_height": record["input_height"],
            **_resize_transform(
                record["input_width"],
                record["input_height"],
                output_width=resize["width"],
                output_height=resize["height"],
                mode=resize["mode"],
            ),
        }
        if record != expected:
            raise R10BBerniniExtractError(
                "Bernini recorded spatial-transform geometry differs"
            )


def _validate_run_contract(
    summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
    done: Mapping[str, Any],
) -> dict[str, Any]:
    measurement = summary.get("measurement", {})
    contract = measurement.get("run_contract")
    if not isinstance(contract, dict):
        raise R10BBerniniExtractError("Bernini run contract is missing")
    if set(contract) != {"artifact_kind", "scheduler", "noise", "resize"}:
        raise R10BBerniniExtractError(
            "Bernini run-contract top-level closure differs"
        )
    scheduler = contract.get("scheduler")
    noise = contract.get("noise")
    resize = contract.get("resize")
    if (
        not isinstance(scheduler, dict)
        or set(scheduler)
        != {"class", "steps", "index", "selection", "timestep", "sigma"}
        or not isinstance(noise, dict)
        or set(noise)
        != {
            "mode",
            "seed",
            "generator",
            "deterministic_seeded",
            "output_shape_matches_reference",
            "temporal_behavior",
        }
        or not isinstance(resize, dict)
        or set(resize)
        != {
            "mode",
            "width",
            "height",
            "num_frames",
            "sampling",
            "shared_sampling_frame_count_policy",
            "source_target_frame_indices_identical",
            "aspect_ratio_preserved",
            "center_crop",
        }
    ):
        raise R10BBerniniExtractError(
            "Bernini run-contract nested closure differs"
        )
    expected = _build_run_contract(
        artifact_kind=contract.get("artifact_kind"),
        scheduler_class=scheduler.get("class"),
        scheduler_steps=scheduler.get("steps"),
        scheduler_index=scheduler.get("index"),
        scheduler_timestep=scheduler.get("timestep"),
        scheduler_sigma=scheduler.get("sigma"),
        noise_mode=noise.get("mode"),
        diffusion_noise_seed=noise.get("seed"),
        resize_mode=resize.get("mode"),
        width=resize.get("width"),
        height=resize.get("height"),
        num_frames=resize.get("num_frames"),
    )
    if contract != expected:
        raise R10BBerniniExtractError(
            "Bernini run-contract values differ"
        )
    digest = object_digest(contract)
    if (
        measurement.get("run_contract_sha256") != digest
        or summary.get("artifact_kind") != contract["artifact_kind"]
        or done.get("artifact_kind") != contract["artifact_kind"]
        or done.get("run_contract_sha256") != digest
    ):
        raise R10BBerniniExtractError(
            "Bernini run-contract artifact binding differs"
        )
    runtime = summary.get("runtime", {})
    if (
        runtime.get("width") != resize["width"]
        or runtime.get("height") != resize["height"]
        or runtime.get("num_frames") != resize["num_frames"]
    ):
        raise R10BBerniniExtractError(
            "Bernini runtime/run-contract dimensions differ"
        )
    flat_values = {
        "noise_mode": noise["mode"],
        "diffusion_noise_seed": noise["seed"],
        "scheduler_class": scheduler["class"],
        "scheduler_steps": scheduler["steps"],
        "scheduler_index": scheduler["index"],
        "scheduler_timestep": scheduler["timestep"],
        "scheduler_sigma": scheduler["sigma"],
    }
    if any(measurement.get(name) != value for name, value in flat_values.items()):
        raise R10BBerniniExtractError(
            "Bernini flat measurement/run-contract values differ"
        )
    resize_policy = measurement.get("resize_policy")
    if not isinstance(resize_policy, dict):
        raise R10BBerniniExtractError("Bernini resize policy is missing")
    expected_resize_policy = {
        **resize,
        "technical_smoke_only": (
            resize["mode"] == "exact_technical"
        ),
        "scientific_retrieval_promotion_eligible": False,
    }
    if resize_policy != expected_resize_policy:
        raise R10BBerniniExtractError(
            "Bernini resize policy/run-contract differs"
        )
    for row in rows:
        if (
            row.get("artifact_kind") != contract["artifact_kind"]
            or row.get("run_contract_sha256") != digest
        ):
            raise R10BBerniniExtractError(
                "Bernini row/run-contract binding differs"
            )
        media = row.get("media_preprocessing")
        if not isinstance(media, dict):
            raise R10BBerniniExtractError(
                "Bernini row media preprocessing is missing"
            )
        required_media = {
            "resize_mode",
            "source_decoded_frame_count",
            "target_decoded_frame_count",
            "shared_sampling_frame_count",
            "selected_frame_indices",
            "source_target_frame_indices_identical",
            "output_width",
            "output_height",
            "source_spatial_transforms",
            "target_spatial_transforms",
            "source_spatial_transforms_sha256",
            "target_spatial_transforms_sha256",
        }
        if set(media) != required_media:
            raise R10BBerniniExtractError(
                "Bernini row media-preprocessing closure differs"
            )
        indices = media.get("selected_frame_indices")
        source_count = media.get("source_decoded_frame_count")
        target_count = media.get("target_decoded_frame_count")
        shared_count = media.get("shared_sampling_frame_count")
        source_transforms = media.get("source_spatial_transforms")
        target_transforms = media.get("target_spatial_transforms")
        if (
            media.get("resize_mode") != resize["mode"]
            or media.get("output_width") != resize["width"]
            or media.get("output_height") != resize["height"]
            or media.get("source_target_frame_indices_identical") is not True
            or isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or isinstance(target_count, bool)
            or not isinstance(target_count, int)
            or isinstance(shared_count, bool)
            or not isinstance(shared_count, int)
            or shared_count != min(source_count, target_count)
            or shared_count < resize["num_frames"]
            or not isinstance(indices, list)
            or len(indices) != resize["num_frames"]
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in indices
            )
            or indices != sorted(set(indices))
            or not indices
            or indices[0] < 0
            or indices[-1] >= shared_count
            or indices != _uniform_frame_indices(
                shared_count,
                resize["num_frames"],
            )
            or media.get("source_spatial_transforms_sha256")
            != object_digest(source_transforms)
            or media.get("target_spatial_transforms_sha256")
            != object_digest(target_transforms)
        ):
            raise R10BBerniniExtractError(
                "Bernini row paired media preprocessing differs"
            )
        _validate_recorded_spatial_transforms(
            source_transforms,
            expected_frames=resize["num_frames"],
            resize=resize,
        )
        _validate_recorded_spatial_transforms(
            target_transforms,
            expected_frames=resize["num_frames"],
            resize=resize,
        )
    return contract


def extract(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.artifact_kind not in ARTIFACT_KINDS:
        raise R10BBerniniExtractError(
            f"unknown artifact_kind: {args.artifact_kind}"
        )
    if args.noise_mode not in NOISE_MODES:
        raise R10BBerniniExtractError(
            f"unknown noise_mode: {args.noise_mode}"
        )
    if args.resize_mode not in RESIZE_MODES:
        raise R10BBerniniExtractError(
            f"unknown resize_mode: {args.resize_mode}"
        )
    if (
        args.artifact_kind == "controlled_retrieval_pilot"
        and args.resize_mode != "aspect_preserving_center_crop"
    ):
        raise R10BBerniniExtractError(
            "controlled_retrieval_pilot requires "
            "aspect_preserving_center_crop"
        )
    if args.width % 16 or args.height % 16:
        raise R10BBerniniExtractError(
            "Bernini width/height must be divisible by 16"
        )
    if (args.num_frames - 1) % 4:
        raise R10BBerniniExtractError(
            "Bernini num_frames must satisfy (F-1) % 4 == 0"
        )
    if len(set(args.projection_seeds)) != len(args.projection_seeds):
        raise R10BBerniniExtractError("projection seeds must be unique")
    if len(args.projection_seeds) < 2:
        raise R10BBerniniExtractError(
            "engineering smoke requires at least two projection seeds"
        )
    rows = read_jsonl(args.manifest)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    try:
        validate_smoke_rows(rows)
    except R10BTangentError as error:
        raise R10BBerniniExtractError(str(error)) from error
    for row in rows:
        for field in ("prompt", "noop_prompt"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise R10BBerniniExtractError(
                    f"{field} is empty for iid={row.get('iid')}"
                )

    root_config = _load_root_config(args.model_path)
    if (
        int(root_config.get("max_sequence_length", -1)) != 512
        or int(args.max_sequence_length) != 512
    ):
        raise R10BBerniniExtractError(
            "Bernini prompt length must match the fixed 512-token contract"
        )
    if args.model_revision != EXPECTED_REPO_REVISION:
        raise R10BBerniniExtractError(
            "Bernini model revision differs from the preregistered release"
        )
    if args.bernini_source_commit != EXPECTED_SOURCE_COMMIT:
        raise R10BBerniniExtractError(
            "Bernini source commit differs from the preregistered commit"
        )
    vendor_manifest = _bernini_source_manifest(args.bernini_repo)

    WanTransformer3DModel = _load_official_transformer_class(
        args.bernini_repo
    )
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import (
        UniPCMultistepScheduler,
    )
    from diffusers.video_processor import VideoProcessor
    from transformers import AutoTokenizer, UMT5EncoderModel

    transformers_version = importlib.metadata.version("transformers")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise R10BBerniniExtractError(
            "R10B Bernini extraction requires one CUDA/ROCm GPU"
        )
    dtype = torch.bfloat16
    vae_dtype = torch.float32
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.reset_peak_memory_stats(device)

    checkpoint_revision_metadata = _checkpoint_revision_metadata(
        args.model_path
    )
    checkpoint_manifest = _checkpoint_manifest(args.model_path)
    track_arrays = _load_track_arrays(args.track_cache)

    tokenizer = _load_fixed_tokenizer(AutoTokenizer, args.model_path)
    vae = AutoencoderKLWan.from_pretrained(
        str(args.model_path),
        subfolder="vae",
        torch_dtype=vae_dtype,
        local_files_only=True,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(args.model_path),
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=True,
    )
    video_processor = VideoProcessor(
        vae_scale_factor=2 ** len(vae.temperal_downsample)
    )
    encoded_cases = _preencode_cases(
        vae=vae,
        video_processor=video_processor,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        rows=rows,
        track_arrays=track_arrays,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        device=device,
        dtype=vae_dtype,
        max_sequence_length=args.max_sequence_length,
        resize_mode=args.resize_mode,
    )
    del vae, text_encoder, tokenizer, video_processor
    gc.collect()
    torch.cuda.empty_cache()

    transformer = WanTransformer3DModel.from_pretrained(
        str(args.model_path),
        subfolder="transformer",
        use_src_id_rotary_emb=True,
        torch_dtype=dtype,
        local_files_only=True,
    )
    transformer.to(device)
    transformer.eval()
    if (
        int(transformer.config.in_channels) != 16
        or int(transformer.config.out_channels) != 16
        or not bool(transformer.rope.use_src_id_rotary_emb)
    ):
        raise R10BBerniniExtractError(
            "Bernini transformer/source-id contract differs"
        )
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)

    parameter_roles = resolve_lucy_attention_roles(
        transformer.named_parameters(),
        block_index=args.block_index,
        include_cross_kv=args.include_cross_kv,
    )
    parameter_roles = dict(parameter_roles)
    parameter_roles["resolver"] = "wan_bernini_last_block_attention_v1"
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
    selected_weight_digest_before = _weight_digest(
        transformer,
        selected_names,
    )

    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(args.model_path),
        subfolder="scheduler",
        flow_shift=float(root_config["shift"]),
        local_files_only=True,
    )
    timestep, sigma, scheduler_index = _scheduler_point(
        scheduler,
        steps=args.scheduler_steps,
        index=args.scheduler_index,
        device=device,
    )
    run_contract = _build_run_contract(
        artifact_kind=args.artifact_kind,
        scheduler_class=scheduler.__class__.__name__,
        scheduler_steps=args.scheduler_steps,
        scheduler_index=scheduler_index,
        scheduler_timestep=float(timestep),
        scheduler_sigma=sigma,
        noise_mode=args.noise_mode,
        diffusion_noise_seed=args.diffusion_noise_seed,
        resize_mode=args.resize_mode,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
    )
    run_contract_digest = object_digest(run_contract)

    feature_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    row_outputs = []
    for case_index, case in enumerate(encoded_cases):
        row = case["row"]
        source_latents = case["source_latents"].to(
            device=device,
            dtype=vae_dtype,
        )
        target_latents = case["target_latents"].to(
            device=device,
            dtype=vae_dtype,
        )
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
                noise_mode=args.noise_mode,
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
            noise_mode=args.noise_mode,
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
                    ].append(
                        raw_cells[(role_name, projection_seed)][cell_name].numpy()
                    )

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
                    raise R10BBerniniExtractError(
                        "combined Bernini tangent is zero/non-finite"
                    )
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
                "artifact_kind": args.artifact_kind,
                "run_contract_sha256": run_contract_digest,
                "media_preprocessing": case["media_preprocessing"],
                "saliency": case["saliency"],
                "prompt_conditioning": case["prompt_conditioning"],
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

    selected_weight_digest_after = _weight_digest(
        transformer,
        selected_names,
    )
    weights_unchanged = (
        selected_weight_digest_after == selected_weight_digest_before
    )
    if not weights_unchanged:
        raise R10BBerniniExtractError(
            "selected frozen Bernini weights changed"
        )

    features = {
        name: np.stack(values).astype(np.float32)
        for name, values in sorted(feature_rows.items())
    }
    identifiers = np.asarray([str(row["iid"]) for row in rows])
    for name, values in features.items():
        if values.ndim != 2 or len(values) != len(rows):
            raise R10BBerniniExtractError(
                f"invalid Bernini feature shape: {name}"
            )
        if not np.isfinite(values).all():
            raise R10BBerniniExtractError(
                f"non-finite Bernini feature: {name}"
            )

    summary = {
        "schema_version": EXTRACT_SCHEMA,
        "status": "complete",
        "artifact_kind": args.artifact_kind,
        "scope": (
            "frozen Bernini-R 1.3B controlled representation measurement; "
            "not Motive paper reproduction, representation promotion, "
            "rendering, generation, or training"
        ),
        "model": {
            "id": "bernini_r_1_3b",
            "registry_role": "primary_instruction_editor",
            "primary_eligible": True,
            "model_path": str(args.model_path.resolve()),
            "huggingface_repo": "ByteDance/Bernini-R-1.3B-Diffusers",
            "huggingface_revision": args.model_revision,
            "huggingface_download_metadata": checkpoint_revision_metadata,
            "checkpoint_manifest": checkpoint_manifest,
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
            "objective": (
                "0.25 motion-weighted x0 level + "
                "0.75 motion-weighted temporal x0 difference"
            ),
            "source_condition": (
                "native Bernini source-id-1 clean source tokens concatenated "
                "with source-id-0 noisy target tokens"
            ),
            "saliency_use": "loss weighting only; never model input or user mask",
            "vae_posterior_mode": "deterministic_argmax",
            "vae_normalization": "(latent-latents_mean)/latents_std",
            "vae_dtype": str(vae_dtype),
            "latent_storage_dtype": str(vae_dtype),
            "noising_dtype": str(vae_dtype),
            "transformer_input_dtype": str(dtype),
            "run_contract": run_contract,
            "run_contract_sha256": run_contract_digest,
            "noise_mode": args.noise_mode,
            "noise_matches_bernini_training": (
                args.noise_mode == "iid_spatiotemporal"
            ),
            "noise_promotion_requirement": (
                "repeat at multiple sigma values with both iid "
                "spatiotemporal and temporal-broadcast noise"
            ),
            "diffusion_noise_seed": int(args.diffusion_noise_seed),
            "scheduler_class": scheduler.__class__.__name__,
            "scheduler_steps": int(args.scheduler_steps),
            "scheduler_index": int(scheduler_index),
            "scheduler_timestep": float(timestep),
            "scheduler_sigma": float(sigma),
            "projection_backend": "countsketch_streaming_raw_before_quotient",
            "projection_dimension_per_role": int(args.projection_dim),
            "projection_seeds": [
                int(value) for value in args.projection_seeds
            ],
            "prompt_conditioning": {
                "mode": PROMPT_MODE,
                "system_prefix": V2V_SYSTEM_PROMPT,
                "system_prefix_sha256": _text_digest(V2V_SYSTEM_PROMPT),
                "concatenation": "system_prefix_plus_cleaned_raw_without_separator",
                "prompt_enhancement_used": False,
                "mv2v_appearance_biased_prefix_used": False,
                "max_sequence_length": int(args.max_sequence_length),
                "tokenizer": _tokenizer_provenance(transformers_version),
            },
            "resize_policy": {
                **run_contract["resize"],
                "technical_smoke_only": (
                    args.resize_mode == "exact_technical"
                ),
                "scientific_retrieval_promotion_eligible": False,
            },
            "feature_promotion_policy": {
                "primary_candidate": "self_motion__factorial_did",
                "cross_instruction": "separate_diagnostic",
                "combined_balanced": (
                    "diagnostic_only; must not promote a weak text residual"
                ),
                "mandatory_controls": [
                    "prompt-only baseline",
                    "action paraphrase lexical holdout",
                    "action-token shuffle with fixed target",
                    "canonical versus original prompt",
                    "cross-subject same-action positives",
                    "same-subject different-action negatives",
                    "similar-background different-action negatives",
                    "appearance readout must fail",
                    "temporal-only loss ablation",
                    "multiple no-op prompts",
                    "multiple sigma and noise modes",
                    "higher-dimensional CountSketch confirmation",
                ],
            },
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
            "unique_components": len(
                {str(row["component_id"]) for row in rows}
            ),
            "videos_read": 2 * len(rows),
            "videos_copied": 0,
            "video_outputs_created": 0,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_hip_version": getattr(torch.version, "hip", None),
            "numpy_version": np.__version__,
            "diffusers_version": importlib.metadata.version("diffusers"),
            "transformers_version": transformers_version,
            "ftfy_version": importlib.metadata.version("ftfy"),
            "device_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "dtype": str(dtype),
            "width": int(args.width),
            "height": int(args.height),
            "num_frames": int(args.num_frames),
        },
        "implementation": _source_implementation_digest(),
        "official_bernini_source": {
            "repo": "https://github.com/bytedance/Bernini",
            "commit": args.bernini_source_commit,
            "loader": "restricted_official_transformer_submodule_v1",
            "attention_backend": "pytorch_sdpa_for_mi210_backward_smoke",
            "unrelated_bernini_llm_or_veomni_imported": False,
            **vendor_manifest,
        },
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
            "artifact_contract_passed": True,
            "technical_smoke_passed": (
                args.artifact_kind == "engineering_smoke"
            ),
            "controlled_retrieval_pilot_extraction_complete": (
                args.artifact_kind == "controlled_retrieval_pilot"
            ),
            "representation_gate_passed": False,
            "next_if_valid": (
                (
                    "run a preregistered small atomic Bernini retrieval pilot "
                    "with temporal, instruction, appearance, and nuisance controls"
                )
                if args.artifact_kind == "engineering_smoke"
                else (
                    "analyze preregistered retrieval/control metrics; all "
                    "representation, renderer, generation, and training gates "
                    "remain false"
                )
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
                metadata_json=np.asarray(
                    canonical_json(summary["measurement"])
                ),
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
            "artifact_kind": args.artifact_kind,
            "run_contract_sha256": run_contract_digest,
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
        raise R10BBerniniExtractError(
            f"Bernini extract directory missing: {output}"
        )
    observed_names = sorted(path.name for path in output.iterdir())
    if observed_names != sorted(OUTPUT_NAMES):
        raise R10BBerniniExtractError(
            f"Bernini artifact closure differs: {observed_names}"
        )
    done = json.loads((output / DONE_NAME).read_text(encoding="utf-8"))
    if done.get("schema_version") != DONE_SCHEMA or done.get("status") != "complete":
        raise R10BBerniniExtractError("Bernini done schema/status differs")
    if done.get("payload_closure") != sorted(PAYLOAD_NAMES):
        raise R10BBerniniExtractError("Bernini payload closure differs")
    payloads = done.get("payloads")
    if not isinstance(payloads, dict) or set(payloads) != set(PAYLOAD_NAMES):
        raise R10BBerniniExtractError("Bernini payload records differ")
    for name in PAYLOAD_NAMES:
        path = output / name
        record = payloads[name]
        if (
            record.get("sha256") != file_digest(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise R10BBerniniExtractError(
                f"Bernini payload digest/size differs: {name}"
            )
    if done.get("artifact_digest") != object_digest(payloads):
        raise R10BBerniniExtractError("Bernini artifact digest differs")
    for field in (
        "representation_gate_passed",
        "renderer_probe_authorized",
        "editor_training_authorized",
    ):
        if done.get(field) is not False:
            raise R10BBerniniExtractError(
                f"Bernini done false gate differs: {field}"
            )
    summary = json.loads((output / SUMMARY_NAME).read_text(encoding="utf-8"))
    if (
        summary.get("schema_version") != EXTRACT_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("artifact_kind") not in ARTIFACT_KINDS
    ):
        raise R10BBerniniExtractError("Bernini summary schema differs")
    model = summary.get("model", {})
    if (
        model.get("selected_weights_unchanged") is not True
        or model.get("selected_weight_sha256_before")
        != model.get("selected_weight_sha256_after")
        or model.get("huggingface_revision") != EXPECTED_REPO_REVISION
    ):
        raise R10BBerniniExtractError(
            "Bernini summary does not prove fixed unchanged weights"
        )
    checkpoint = model.get("checkpoint_manifest", {})
    checkpoint_files = checkpoint.get("files")
    if not isinstance(checkpoint_files, list):
        raise R10BBerniniExtractError(
            "Bernini checkpoint manifest is missing"
        )
    checkpoint_identities = {}
    for record in checkpoint_files:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or _SHA256_RE.fullmatch(str(record.get("sha256"))) is None
            or not isinstance(record.get("huggingface_identity"), str)
        ):
            raise R10BBerniniExtractError(
                "Bernini checkpoint file record differs"
            )
        checkpoint_identities[record["path"]] = (
            record["bytes"],
            record["huggingface_identity"],
        )
    if (
        checkpoint_identities != EXPECTED_CHECKPOINT_FILES
        or len(checkpoint_files) != len(EXPECTED_CHECKPOINT_FILES)
        or checkpoint.get("file_count") != len(EXPECTED_CHECKPOINT_FILES)
        or checkpoint.get("total_bytes")
        != sum(size for size, _identity in EXPECTED_CHECKPOINT_FILES.values())
        or checkpoint.get("tree_sha256") != object_digest(checkpoint_files)
    ):
        raise R10BBerniniExtractError(
            "Bernini checkpoint provenance differs"
        )
    expected_metadata_rows = [
        {
            "path": relative,
            "revision": EXPECTED_REPO_REVISION,
            "huggingface_identity": identity,
        }
        for relative, (_size, identity) in sorted(
            EXPECTED_CHECKPOINT_FILES.items()
        )
    ]
    download_metadata = model.get("huggingface_download_metadata", {})
    if (
        download_metadata.get("metadata_files")
        != len(EXPECTED_CHECKPOINT_FILES)
        or download_metadata.get("revision") != EXPECTED_REPO_REVISION
        or download_metadata.get("manifest_sha256")
        != object_digest(expected_metadata_rows)
    ):
        raise R10BBerniniExtractError(
            "Bernini download metadata provenance differs"
        )
    official_source = summary.get("official_bernini_source", {})
    if (
        official_source.get("commit") != EXPECTED_SOURCE_COMMIT
        or official_source.get("bundle_sha256")
        != EXPECTED_SOURCE_BUNDLE_SHA256
        or official_source.get("attention_backend")
        != "pytorch_sdpa_for_mi210_backward_smoke"
        or official_source.get(
            "unrelated_bernini_llm_or_veomni_imported"
        )
        is not False
    ):
        raise R10BBerniniExtractError(
            "Bernini official source provenance differs"
        )
    safety = summary.get("safety", {})
    required_false = (
        "optimizer_created",
        "checkpoint_mutated",
        "representation_gate_passed",
        "renderer_probe_authorized",
        "editor_training_authorized",
    )
    if any(safety.get(name) is not False for name in required_false):
        raise R10BBerniniExtractError("Bernini safety false gate differs")
    if safety.get("optimizer_steps") != 0 or safety.get("renderer_calls") != 0:
        raise R10BBerniniExtractError(
            "forbidden Bernini optimizer/renderer activity recorded"
        )
    if (
        safety.get("scheduler_steps_executed") != 0
        or safety.get("videos_rendered") != 0
        or safety.get("videos_copied") != 0
    ):
        raise R10BBerniniExtractError(
            "forbidden Bernini sampling/media activity recorded"
        )
    decision = summary.get("decision", {})
    artifact_kind = summary.get("artifact_kind")
    if (
        decision.get("artifact_contract_passed") is not True
        or decision.get("technical_smoke_passed")
        is not (artifact_kind == "engineering_smoke")
        or decision.get("controlled_retrieval_pilot_extraction_complete")
        is not (artifact_kind == "controlled_retrieval_pilot")
        or decision.get("representation_gate_passed") is not False
    ):
        raise R10BBerniniExtractError("Bernini decision gates differ")
    measurement = summary.get("measurement", {})
    projection_seeds = measurement.get("projection_seeds")
    projection_dim = measurement.get("projection_dimension_per_role")
    if (
        not isinstance(projection_seeds, list)
        or len(projection_seeds) < 2
        or len(set(projection_seeds)) != len(projection_seeds)
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in projection_seeds
        )
        or isinstance(projection_dim, bool)
        or not isinstance(projection_dim, int)
        or projection_dim <= 0
    ):
        raise R10BBerniniExtractError(
            "Bernini projection schema differs"
        )
    expected_dimensions = {}
    for role_name in ROLE_NAMES:
        for projection_seed in projection_seeds:
            for tangent_name in TANGENT_NAMES:
                expected_dimensions[
                    f"{role_name}__{tangent_name}__p{projection_seed}"
                ] = projection_dim
            expected_dimensions[
                f"{role_name}__uniform_target__p{projection_seed}"
            ] = projection_dim
            for cell_name in CELL_NAMES:
                expected_dimensions[
                    f"raw__{role_name}__{cell_name}__p{projection_seed}"
                ] = projection_dim
    for projection_seed in projection_seeds:
        for tangent_name in TANGENT_NAMES:
            expected_dimensions[
                f"combined_balanced__{tangent_name}__p{projection_seed}"
            ] = 2 * projection_dim
    with np.load(output / FEATURES_NAME, allow_pickle=False) as archive:
        if "ids" not in archive.files or "metadata_json" not in archive.files:
            raise R10BBerniniExtractError(
                "Bernini features metadata closure differs"
            )
        ids = np.asarray(archive["ids"]).astype(str)
        feature_names = sorted(
            set(archive.files) - {"ids", "metadata_json"}
        )
        if set(feature_names) != set(expected_dimensions):
            raise R10BBerniniExtractError(
                "Bernini feature-name closure differs"
            )
        metadata_json = np.asarray(archive["metadata_json"])
        if (
            metadata_json.ndim != 0
            or str(metadata_json.item()) != canonical_json(measurement)
        ):
            raise R10BBerniniExtractError(
                "Bernini embedded measurement metadata differs"
            )
        for name in feature_names:
            values = np.asarray(archive[name])
            if values.shape != (len(ids), expected_dimensions[name]):
                raise R10BBerniniExtractError(
                    f"Bernini feature shape differs: {name}"
                )
            if not np.isfinite(values).all():
                raise R10BBerniniExtractError(
                    f"Bernini feature is non-finite: {name}"
                )
    rows = read_jsonl(output / ROWS_NAME)
    if (
        len(rows) != len(ids)
        or len(set(ids.tolist())) != len(ids)
        or len(rows) != summary.get("data", {}).get("rows")
        or [row["iid"] for row in rows] != ids.tolist()
    ):
        raise R10BBerniniExtractError("Bernini row/id order differs")
    run_contract = _validate_run_contract(summary, rows, done)
    _validate_tokenizer_provenance(summary, rows)
    for row in rows:
        if row.get("schema_version") != ROW_SCHEMA:
            raise R10BBerniniExtractError("Bernini row schema differs")
        for field in (
            "formal_evidence",
            "representation_gate_passed",
            "renderer_probe_authorized",
            "editor_training_authorized",
        ):
            if row.get(field) is not False:
                raise R10BBerniniExtractError(
                    f"Bernini row false gate differs: {field}"
                )
        conditioning = row.get("prompt_conditioning", {})
        if conditioning.get("mode") != PROMPT_MODE:
            raise R10BBerniniExtractError(
                "Bernini row prompt mode differs"
            )
        for field in (
            "raw_prompt_sha256",
            "raw_noop_prompt_sha256",
            "effective_prompt_sha256",
            "effective_noop_prompt_sha256",
        ):
            if _SHA256_RE.fullmatch(str(conditioning.get(field))) is None:
                raise R10BBerniniExtractError(
                    f"Bernini row prompt digest differs: {field}"
                )
    return {
        "status": "VALID",
        "output_dir": str(output.resolve()),
        "artifact_digest": done["artifact_digest"],
        "artifact_kind": run_contract["artifact_kind"],
        "scheduler_index": run_contract["scheduler"]["index"],
        "scheduler_sigma": run_contract["scheduler"]["sigma"],
        "noise_mode": run_contract["noise"]["mode"],
        "resize_mode": run_contract["resize"]["mode"],
        "rows": len(rows),
        "feature_arrays": len(feature_names),
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract or validate a Bernini-R 1.3B R10B frozen tangent artifact."
        )
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-revision", default=EXPECTED_REPO_REVISION)
    parser.add_argument("--bernini-repo", type=Path)
    parser.add_argument("--bernini-source-commit")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--track-cache", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--artifact-kind",
        choices=ARTIFACT_KINDS,
        default="engineering_smoke",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--block-index", type=int)
    parser.add_argument("--include-cross-kv", action="store_true")
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument(
        "--projection-seeds",
        type=int,
        nargs="+",
        default=[260108851, 260108852],
    )
    parser.add_argument("--diffusion-noise-seed", type=int, default=260108853)
    parser.add_argument(
        "--noise-mode",
        choices=NOISE_MODES,
        default="temporal_broadcast",
    )
    parser.add_argument(
        "--resize-mode",
        choices=RESIZE_MODES,
        default="exact_technical",
    )
    parser.add_argument("--torch-seed", type=int, default=260108854)
    parser.add_argument(
        "--scheduler-steps",
        type=int,
        default=DEFAULT_SCHEDULER_STEPS,
    )
    parser.add_argument(
        "--scheduler-index",
        type=int,
        default=DEFAULT_SCHEDULER_INDEX,
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-tree-sha256", default="")
    args = parser.parse_args()
    if not args.validate_only:
        required = (
            "model_path",
            "bernini_repo",
            "bernini_source_commit",
            "manifest",
            "track_cache",
        )
        for name in required:
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        if _GIT_COMMIT_RE.fullmatch(args.bernini_source_commit) is None:
            parser.error("--bernini-source-commit must be one lowercase commit")
        for name in ("source_tree_sha256",):
            if _SHA256_RE.fullmatch(str(getattr(args, name))) is None:
                parser.error(
                    f"--{name.replace('_', '-')} must be one lowercase SHA-256"
                )
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

#!/usr/bin/env python3
"""Mask-free Bernini-R 1.3B renderer LoRA training on exact 81-frame pairs.

This is a deliberately small training harness around the *official* Bernini
implementation.  It neither copies nor reimplements the renderer.  The source
tree is pinned to the public Bernini training release, and the model, sample
transform, flow scheduler, and rotary embedding are imported from that tree at
runtime.

The only conditioning available to the renderer is the source video latent and
the natural-language edit instruction.  The target video latent participates
only in the standard flow-matching construction (x_t and velocity target).
External spatial/segmentation masks, tracks, and swept tubes are rejected.  The
``vae_latents_mask`` produced by Bernini internally is merely a packed-token
loss selector; it is not an edit mask or an inference-time input.

For the intended AUH launch use four ROCm processes::

    torchrun --standalone --nproc_per_node=4 train_lora.py \
      --bernini-root /abs/Bernini \
      --veomni-root /abs/VeOmni \
      --checkpoint /abs/Bernini-R-1.3B-Diffusers \
      --preprocessed-parquet-dir /abs/vae_parquet \
      --output /abs/run --num-frames 81 --max-steps 400

Each rank intentionally sees the same sample and the same stochastic seed.
Bernini Ulysses splits its token sequence over the four ranks, after which this
script explicitly all-reduces the replicated LoRA gradients before stepping.
"""

from __future__ import annotations

import argparse
import bisect
import ctypes
from dataclasses import dataclass
from datetime import timedelta
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence


BERNINI_OFFICIAL_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_TESTED_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
BERNINI_PINNED_FILE_HASHES = {
    "bernini/models/renderer.py": "fec319f3ede3482b28873dc55622208f1242ecba0caedea8e710093748dc7159",
    "bernini/models/wan_diffusion.py": "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512",
    "bernini/models/transformer_wan.py": "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223",
    "bernini/models/scheduler.py": "b6d729187fd784bf66831d5260a5c9482d89c452881d2f700c8887278f52ef97",
    "bernini/training/data.py": "29aa4f89579c7771cb9f78706fde4f0dca0de954fdb2f5e2de1abacd8a0d6c65",
    "bernini/attention.py": "e3986d1e5ba2e70f5244f53e77adbec705720be5cd2e9dbbde92f5aec1f99055",
    "bernini/parallel/state.py": "32d784e7193297a599569da07c091b8d0a51ab08ad319ee2cfc0e495921db3aa",
    "bernini/parallel/ops.py": "c264f28b7b011ce01204ec5b0f11acd08adb6568a9855108b866fb9ce1a2ce30",
    "configs/bernini_renderer_wan21_1p3b/config.json": "4659e97bbb09f6c9baa3528dcdbb23064998e2f92aace8e8fd4b02776c529496",
}
RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-receipt-v2"
VAE_DATASET_SUMMARY_SCHEMA = "bernini-r-action-vae-dataset-summary-v2"
VAE_DATASET_INDEX_ROW_SCHEMA = "bernini-r-action-vae-index-row-v2"
REWARD_SELECTED_DATASET_SUMMARY_SCHEMA = (
    "bernini-reward-selected-synthetic-target-dataset-summary-v1"
)
EXPECTED_DATASET_ROWS = 644
EXPECTED_STRICT_ROWS = 359
EXPECTED_NON_STRICT_ROWS = 285
EXPECTED_INCLUSION_POLICY = "natural_release_all"
TOKENIZER_FIX_MISTRAL_REGEX = True
TASK_SOURCE_NAME = "mv2v$action_editing_81f"
NUM_FRAMES = 81
LATENT_FRAMES = 21  # (81 - 1) / Wan temporal compression 4 + 1
LORA_RANK = 8
LORA_ALPHA = 8
FULL644_EXPLORATORY_PROFILE = "full644-r64-reference-dpo-preservation-one-pass-v1"
FULL644_EXPLORATORY_STEPS = 644
FULL644_EXPLORATORY_RANK = 64
FULL644_EXPLORATORY_ALPHA = 64
FULL644_EXPLORATORY_SEED = 20260817
FULL644_PEFT_VERSION = "0.19.1"
# 240 audited Wan projection routes, each with rank-64 A/B matrices over
# 1536-dimensional inputs/outputs: 240 * 64 * (1536 + 1536).
FULL644_EXPLORATORY_TRAINABLE_PARAMETER_COUNT = 47_185_920
FULL644_DATASET_SUMMARY_SHA256 = (
    "5dc45b4a6d700b3cd0108e941242ae364396458f20f41249744e74e00acc02dd"
)
FULL644_DATASET_SUMMARY_DIGEST = (
    "29e2341f09d58289590ae48d17d02f2299bac3201df772584b6269bec0dbbe82"
)
FULL644_DATASET_INDEX_SHA256 = (
    "d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b"
)
FULL644_SOURCE_AUTHORITY_SHA256 = (
    "0bcf24ce8aafabb37cf38eafe9da6b13c70043bb0f4c3146f16dc0bafd35618f"
)
# Exact ``LoraConfig.to_dict()`` closure for PEFT 0.19.1.  In particular,
# LoraConfig deliberately removes its runtime-only ``runtime_config`` field
# before returning this mapping, so this is 38 fields (not 39).
FULL644_PEFT_LORA_CONFIG_FIELDS = frozenset(
    {
        "alora_invocation_tokens",
        "alpha_pattern",
        "arrow_config",
        "auto_mapping",
        "base_model_name_or_path",
        "bias",
        "corda_config",
        "ensure_weight_tying",
        "eva_config",
        "exclude_modules",
        "fan_in_fan_out",
        "inference_mode",
        "init_lora_weights",
        "layer_replication",
        "layers_pattern",
        "layers_to_transform",
        "loftq_config",
        "lora_alpha",
        "lora_bias",
        "lora_dropout",
        "lora_ga_config",
        "megatron_config",
        "megatron_core",
        "modules_to_save",
        "peft_type",
        "peft_version",
        "qalora_group_size",
        "r",
        "rank_pattern",
        "revision",
        "target_modules",
        "target_parameters",
        "task_type",
        "trainable_token_indices",
        "use_bdlora",
        "use_dora",
        "use_qalora",
        "use_rslora",
    }
)
EXPECTED_LORA_TARGET_MODULES = 30 * 2 * 4
TRAINING_OBJECTIVES = (
    "sft",
    "sft_preservation",
    "high_contrast_margin",
    "detached_margin",
    "detached_margin_preservation",
    "reference_dpo",
    "reference_dpo_preservation",
)
CONTRASTIVE_NEGATIVE_KINDS = ("noop", "reverse", "incomplete")
CONTRASTIVE_NEGATIVE_SCHEDULES = (
    "rotate",
    "noop_incomplete",
) + CONTRASTIVE_NEGATIVE_KINDS
IDENTITY_PRESERVATION_INSTRUCTION = (
    "Keep the source video unchanged, preserving the subject, appearance, "
    "background, camera, and motion."
)

# These are sequence-packing fields in the official renderer collator.  For a
# single sample they gain a leading packed-batch dimension.  Latent sequences,
# by contrast, stay concatenated along dimension zero.
PACKING_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "t5_input_lens",
        "vae_latents_mask",
        "vae_seqlen",
        "timesteps",
        "num_tokens",
        "vlm_seqlen",
        "target_lens",
    }
)
CONCAT_KEYS = frozenset(
    {"input_vae_latents", "input_vae_rope", "target_velocity"}
)
REQUIRED_MODEL_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "t5_input_lens",
        "input_vae_latents",
        "input_vae_rope",
        "vae_latents_mask",
        "vae_seqlen",
        "timesteps",
        "target_velocity",
        "target_lens",
    }
)

# Dataset-provided spatial hints are forbidden.  This list is intentionally
# explicit so provenance columns such as an ordinary boolean quality gate are
# not mistaken for model conditioning.
FORBIDDEN_SPATIAL_CONDITIONING_KEYS = frozenset(
    {
        "mask",
        "edit_mask",
        "spatial_mask",
        "segmentation_mask",
        "motion_mask",
        "tracking_mask",
        "track_mask",
        "tube_mask",
        "swept_tube",
        "swept_tube_mask",
        "bounding_boxes",
        "bboxes",
        "trajectories",
    }
)

_ATTENTION_PROJECTION = re.compile(
    r"^(?P<prefix>.+\.blocks\.\d+\.attn[12])\."
    r"(?P<projection>to_q|to_k|to_v|to_out\.0)$"
)
_EXPECTED_PROJECTIONS = frozenset({"to_q", "to_k", "to_v", "to_out.0"})
_LORA_PARAMETER_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
)


class TrainingContractError(RuntimeError):
    """Raised when a model, data, or distributed invariant is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TrainingContractError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise TrainingContractError(f"JSON object required: {path}")
    return value


def _absolute_existing_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise TrainingContractError(f"{label} must be an absolute local path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise TrainingContractError(f"{label} is unavailable: {path}: {error}") from error
    if not resolved.is_dir():
        raise TrainingContractError(f"{label} is not a directory: {resolved}")
    return resolved


def git_revision(root: Path, *, required: bool = True) -> Optional[str]:
    """Return the exact source revision without importing repository code."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        if required:
            raise TrainingContractError(
                f"cannot prove source revision for {root}: {error}"
            ) from error
        return None
    revision = result.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise TrainingContractError(f"invalid git revision for {root}: {revision!r}")
    return revision


def require_tracked_clean(root: Path, *, label: str) -> None:
    """Reject modifications to versioned source while ignoring runtime caches."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise TrainingContractError(f"cannot audit {label} tracked state: {error}") from error
    if result.stdout.strip():
        raise TrainingContractError(f"{label} has modified tracked files")


def validate_source_trees(
    bernini_root: str | Path,
    veomni_root: str | Path,
    *,
    expected_bernini_commit: str = BERNINI_OFFICIAL_COMMIT,
    expected_veomni_commit: str = VEOMNI_TESTED_COMMIT,
) -> tuple[Path, Path, str, str]:
    bernini = _absolute_existing_directory(bernini_root, label="bernini_root")
    veomni = _absolute_existing_directory(veomni_root, label="veomni_root")
    required_bernini = (
        "bernini/models/renderer.py",
        "bernini/models/transformer_wan.py",
        "bernini/training/data.py",
        "configs/bernini_renderer_wan21_1p3b/config.json",
    )
    for relative in required_bernini:
        if not (bernini / relative).is_file():
            raise TrainingContractError(f"incomplete Bernini tree: missing {relative}")
    if not (veomni / "veomni/distributed/sequence_parallel").is_dir():
        raise TrainingContractError("incomplete VeOmni tree: sequence_parallel is missing")
    if (bernini / ".git").is_dir():
        bernini_revision = git_revision(bernini)
        if bernini_revision != expected_bernini_commit.lower():
            raise TrainingContractError(
                "Bernini source revision mismatch: "
                f"expected {expected_bernini_commit}, got {bernini_revision}"
            )
        require_tracked_clean(bernini, label="Bernini source")
    else:
        # AUH already holds the official release as a hash-bound, read-only tar
        # extraction without .git.  Verify every file on the training path and
        # bind the recorded revision to those immutable bytes.
        for relative, expected_sha in BERNINI_PINNED_FILE_HASHES.items():
            path = bernini / relative
            if not path.is_file() or path.is_symlink():
                raise TrainingContractError(f"pinned Bernini file is missing: {relative}")
            actual_sha = file_sha256(path)
            if actual_sha != expected_sha:
                raise TrainingContractError(
                    f"pinned Bernini file hash mismatch: {relative}: {actual_sha}"
                )
        bernini_revision = expected_bernini_commit.lower()
    veomni_revision = git_revision(veomni)
    if veomni_revision != expected_veomni_commit.lower():
        raise TrainingContractError(
            "VeOmni source revision mismatch: "
            f"expected {expected_veomni_commit}, got {veomni_revision}"
        )
    require_tracked_clean(veomni, label="VeOmni source")
    assert veomni_revision is not None
    return bernini, veomni, bernini_revision, veomni_revision


def activate_source_trees(bernini_root: Path, veomni_root: Path) -> None:
    """Put pinned source trees first; no installed Bernini fork may win."""

    roots = [str(bernini_root), str(veomni_root)]
    for root in roots:
        while root in sys.path:
            sys.path.remove(root)
    sys.path[0:0] = roots


def validate_renderer_config_mapping(
    config: Mapping[str, Any], checkpoint: str | Path
) -> None:
    """Validate the Bernini-R 1.3B single-expert renderer configuration."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        raise TrainingContractError("renderer wan22_base must be an absolute path")
    if config.get("model_type") != "bernini_renderer":
        raise TrainingContractError("model_type must be bernini_renderer")
    skip_1 = config.get("skip_transformer_1")
    skip_2 = config.get("skip_transformer_2")
    if skip_1 is not False or skip_2 is not True:
        raise TrainingContractError(
            "Bernini-R 1.3B must train transformer_1 only "
            "(skip_transformer_1=false, skip_transformer_2=true)"
        )
    wan_base = config.get("wan22_base")
    if wan_base is None or Path(str(wan_base)) != checkpoint_path:
        raise TrainingContractError(
            f"wan22_base must equal the local checkpoint: {checkpoint_path}"
        )
    if config.get("use_src_id_rotary_emb") is not True:
        raise TrainingContractError("source-id rotary embeddings must stay enabled")
    if int(config.get("max_sequence_length", 0)) != 512:
        raise TrainingContractError("max_sequence_length must be 512")


def validate_checkpoint(checkpoint: str | Path) -> tuple[Path, dict[str, Any]]:
    """Prove that ``checkpoint`` is a local Wan/Bernini-R 1.3B single DiT."""

    root = _absolute_existing_directory(checkpoint, label="checkpoint")
    required_dirs = ("transformer", "text_encoder", "tokenizer", "vae", "scheduler")
    for name in required_dirs:
        if not (root / name).is_dir():
            raise TrainingContractError(f"checkpoint is incomplete: missing {name}/")
    if (root / "transformer_2").exists():
        raise TrainingContractError(
            "checkpoint contains transformer_2; a single-expert 1.3B checkpoint is required"
        )
    transformer_config_path = root / "transformer/config.json"
    transformer_config = _read_json(transformer_config_path)
    expected = {
        "num_layers": 30,
        "num_attention_heads": 12,
        "attention_head_dim": 128,
        "in_channels": 16,
        "out_channels": 16,
    }
    for key, wanted in expected.items():
        if transformer_config.get(key) != wanted:
            raise TrainingContractError(
                f"checkpoint is not the expected 1.3B transformer: "
                f"{key}={transformer_config.get(key)!r}, expected {wanted!r}"
            )
    weight_files = tuple((root / "transformer").glob("*.safetensors"))
    index_files = tuple((root / "transformer").glob("*.safetensors.index.json"))
    if not weight_files and not index_files:
        raise TrainingContractError("transformer has no local safetensors weights")
    return root, transformer_config


def renderer_config_overrides(checkpoint: Path) -> dict[str, Any]:
    """The immutable 1.3B config overrides applied to the official config."""

    if not checkpoint.is_absolute():
        raise TrainingContractError("checkpoint override must be absolute")
    return {
        "wan22_base": str(checkpoint),
        "diff_dec_config_path": str(checkpoint),
        "skip_transformer_1": False,
        "skip_transformer_2": True,
        "switch_dit_boundary": 0.0,
        "max_sequence_length": 512,
        "shift": 3.0,
        "use_src_id_rotary_emb": True,
        "scratch": False,
        "ema_decay": None,
    }


def noise_scheduler_kwargs() -> dict[str, Any]:
    """Full-flow noise schedule for the single 1.3B expert."""

    return {
        # The checkpoint scheduler default is shift=3, but Bernini's official
        # renderer-training config assigns video editing / mv2v shift=5.
        "shift_config": {"default": 3.0, "mv2v": 5.0},
        "weighting_scheme_config": {"default": "mode", "mv2v": "mode"},
        "noise_tmin": 0.0,
        "noise_tmax": 1.0,
    }


def select_attention_projection_names(model: Any) -> list[str]:
    """Select every q/k/v/out Linear under active Wan self/cross attention.

    Exact fully-qualified module names are returned.  This avoids suffix-based
    PEFT matches accidentally adapting the frozen UMT5 encoder.
    """

    if not hasattr(model, "named_modules"):
        raise TrainingContractError("model does not expose named_modules()")
    by_attention: dict[str, set[str]] = {}
    names: list[str] = []
    for name, module in model.named_modules():
        match = _ATTENTION_PROJECTION.fullmatch(name)
        if match is None:
            continue
        # All four targets are affine projections.  A duck-typed weight check
        # keeps this helper model-load-free in unit tests.
        if not hasattr(module, "weight"):
            raise TrainingContractError(f"attention target is not affine: {name}")
        prefix = match.group("prefix")
        projection = match.group("projection")
        by_attention.setdefault(prefix, set()).add(projection)
        names.append(name)
    if not by_attention:
        raise TrainingContractError("no Wan attention projections were found")
    incomplete = {
        prefix: sorted(_EXPECTED_PROJECTIONS - projections)
        for prefix, projections in by_attention.items()
        if projections != _EXPECTED_PROJECTIONS
    }
    if incomplete:
        raise TrainingContractError(f"incomplete attention projection set: {incomplete}")
    block_ids = {
        int(re.search(r"\.blocks\.(\d+)\.", prefix).group(1))  # type: ignore[union-attr]
        for prefix in by_attention
    }
    if len(by_attention) != 2 * len(block_ids):
        raise TrainingContractError(
            "each Wan block must expose exactly attn1 and attn2 projections"
        )
    return sorted(names)


def is_lora_parameter_name(name: str) -> bool:
    return any(marker in name for marker in _LORA_PARAMETER_MARKERS)


def trainable_lora_parameters(model: Any) -> list[tuple[str, Any]]:
    """Return trainable parameters and fail if any frozen-base invariant broke."""

    if not hasattr(model, "named_parameters"):
        raise TrainingContractError("model does not expose named_parameters()")
    selected = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not selected:
        raise TrainingContractError("LoRA injection produced no trainable parameters")
    leaked = [name for name, _ in selected if not is_lora_parameter_name(name)]
    if leaked:
        raise TrainingContractError(
            f"base/T5 parameters unexpectedly trainable: {leaked[:8]}"
        )
    return selected


def _normalise_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_")


def _parse_inputs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise TrainingContractError(f"inputs is invalid JSON: {error}") from error
    if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
        raise TrainingContractError("inputs must be a JSON list of message objects")
    return value


def _validate_message_contract(messages: Sequence[Mapping[str, Any]]) -> None:
    if len(messages) != 3:
        raise TrainingContractError(
            "renderer row must contain exactly source video, instruction, target video_gen"
        )
    expected = (("video", 0), ("text", 0), ("video_gen", 1))
    for index, (message, (wanted_type, wanted_loss)) in enumerate(zip(messages, expected)):
        if message.get("type") != wanted_type or message.get("has_loss") != wanted_loss:
            raise TrainingContractError(
                f"message {index} must be type={wanted_type!r}, has_loss={wanted_loss}"
            )
    text = messages[1].get("text")
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise TrainingContractError("edit instruction must be non-empty text")
    for index in (0, 2):
        extra_text = messages[index].get("text")
        if extra_text not in (None, ""):
            raise TrainingContractError("video messages may not carry target captions/text")


def _as_list(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    raise TrainingContractError(f"{label} must be a sequence")


def sanitize_preprocessed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and reduce a parquet row to inputs consumed by Bernini.

    VIT/Qwen features and all provenance/path columns are deliberately omitted.
    Thus no target-derived caption, embedding, tracking result, or spatial mask
    can enter the renderer call.
    """

    forbidden = sorted(
        key
        for key in row
        if _normalise_key(key) in FORBIDDEN_SPATIAL_CONDITIONING_KEYS
        and row[key] not in (None, [], "")
    )
    if forbidden:
        raise TrainingContractError(
            f"external spatial conditioning is forbidden: {forbidden}"
        )
    messages = _parse_inputs(row.get("inputs"))
    _validate_message_contract(messages)
    video_latents = _as_list(
        row.get("video_vae_latents"), label="video_vae_latents"
    )
    if len(video_latents) != 2:
        raise TrainingContractError(
            "exactly two VAE distributions are required (source then target)"
        )
    return {
        "inputs": canonical_json_bytes(messages).decode("utf-8"),
        "video_vae_latents": video_latents,
        "source_name": TASK_SOURCE_NAME,
    }


def _load_tensor_blob(blob: Any) -> Any:
    import torch

    if isinstance(blob, torch.Tensor):
        return blob
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TrainingContractError(
            f"VAE distribution must be a torch tensor blob, got {type(blob).__name__}"
        )
    buffer = io.BytesIO(bytes(blob))
    try:
        return torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility for pure-contract tests.
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")
    except Exception as error:
        raise TrainingContractError(f"cannot decode VAE distribution: {error}") from error


def _tensor_blob(tensor: Any) -> bytes:
    """Serialize a CPU tensor without introducing a filesystem dependency."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TrainingContractError("only torch tensors can be serialized as VAE blobs")
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu().contiguous(), buffer)
    return buffer.getvalue()


def contrastive_negative_kind(global_step: int, *, schedule: str = "rotate") -> str:
    """Rotate the three preregistered high-contrast failures deterministically."""

    if global_step < 0:
        raise TrainingContractError("global_step must be non-negative")
    if schedule not in CONTRASTIVE_NEGATIVE_SCHEDULES:
        raise TrainingContractError(f"unsupported contrastive schedule: {schedule}")
    if schedule == "noop_incomplete":
        return ("noop", "incomplete")[global_step % 2]
    if schedule != "rotate":
        return schedule
    return CONTRASTIVE_NEGATIVE_KINDS[global_step % len(CONTRASTIVE_NEGATIVE_KINDS)]


def build_contrastive_sample(
    sample: Mapping[str, Any], *, negative_kind: str
) -> dict[str, Any]:
    """Replace the chosen target with a deterministic action failure.

    ``noop`` uses the source distribution as target, ``reverse`` reverses the
    chosen target along latent time, and ``incomplete`` freezes the chosen
    target after its midpoint.  No external scorer or caption enters training.
    """

    import torch

    if negative_kind not in CONTRASTIVE_NEGATIVE_KINDS:
        raise TrainingContractError(f"unsupported contrastive negative: {negative_kind}")
    validate_81_frame_latents(sample)
    source_blob, target_blob = _as_list(
        sample.get("video_vae_latents"), label="video_vae_latents"
    )
    source = _load_tensor_blob(source_blob)
    target = _load_tensor_blob(target_blob)
    if negative_kind == "noop":
        rejected = source.clone()
    elif negative_kind == "reverse":
        rejected = target.flip(dims=(2,)).contiguous()
    else:
        rejected = target.clone()
        midpoint = LATENT_FRAMES // 2
        rejected[:, :, midpoint + 1 :] = rejected[:, :, midpoint : midpoint + 1]
    if not bool(torch.isfinite(rejected).all().item()):
        raise TrainingContractError("contrastive target contains non-finite values")
    return {
        "inputs": sample["inputs"],
        "video_vae_latents": [source_blob, _tensor_blob(rejected)],
        "source_name": TASK_SOURCE_NAME,
    }


def build_identity_preservation_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Build a source-as-target conditional identity example."""

    messages = _parse_inputs(sample.get("inputs"))
    _validate_message_contract(messages)
    identity_messages = [dict(message) for message in messages]
    identity_messages[1]["text"] = IDENTITY_PRESERVATION_INSTRUCTION
    source_blob = _as_list(
        sample.get("video_vae_latents"), label="video_vae_latents"
    )[0]
    result = {
        "inputs": canonical_json_bytes(identity_messages).decode("utf-8"),
        "video_vae_latents": [source_blob, source_blob],
        "source_name": TASK_SOURCE_NAME,
    }
    validate_81_frame_latents(result)
    return result


def high_contrast_preference_loss(
    chosen_loss: Any,
    rejected_loss: Any,
    *,
    margin: float,
    temperature: float,
) -> Any:
    """Smooth hinge requiring rejected flow error to exceed chosen error."""

    import torch.nn.functional as functional

    gap = rejected_loss - chosen_loss
    return functional.softplus(temperature * (margin - gap)) / temperature


def detached_rejected_preference_loss(
    chosen_loss: Any,
    rejected_loss: Any,
    *,
    margin: float,
    temperature: float,
) -> Any:
    """Use rejected difficulty without optimizing the model away from it.

    The rejected scalar is a stop-gradient gate.  This can strengthen chosen
    fitting while preventing the easy reward-hacking route of deliberately
    increasing rejected flow error.
    """

    import torch.nn.functional as functional

    gap = rejected_loss.detach() - chosen_loss
    return functional.softplus(temperature * (margin - gap)) / temperature


def reference_dpo_loss(
    chosen_loss: Any,
    rejected_loss: Any,
    reference_chosen_loss: Any,
    reference_rejected_loss: Any,
    *,
    beta: float,
) -> Any:
    """Reference-corrected flow-error preference objective.

    Positive advantage means the adapter increased the rejected-vs-chosen
    error gap relative to the frozen base renderer.
    """

    import torch.nn.functional as functional

    student_gap = rejected_loss - chosen_loss
    reference_gap = reference_rejected_loss - reference_chosen_loss
    advantage = student_gap - reference_gap
    return functional.softplus(-beta * advantage) / beta


def validate_81_frame_latents(
    sample: Mapping[str, Any], *, expected_parameter_channels: int = 32
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate two Wan VAE distribution tensors for an exact 81-frame pair."""

    blobs = _as_list(sample.get("video_vae_latents"), label="video_vae_latents")
    if len(blobs) != 2:
        raise TrainingContractError("81-frame validation requires source and target")
    shapes: list[tuple[int, ...]] = []
    for role, blob in zip(("source", "target"), blobs):
        tensor = _load_tensor_blob(blob)
        shape = tuple(int(x) for x in getattr(tensor, "shape", ()))
        if len(shape) != 5:
            raise TrainingContractError(
                f"{role} VAE distribution must be [B,2C,T,H,W], got {shape}"
            )
        if shape[0] != 1 or shape[1] != expected_parameter_channels:
            raise TrainingContractError(
                f"{role} VAE distribution has wrong batch/channels: {shape}"
            )
        if shape[2] != LATENT_FRAMES:
            raise TrainingContractError(
                f"{role} has {shape[2]} latent frames; exact {NUM_FRAMES}f requires {LATENT_FRAMES}"
            )
        if shape[3] <= 0 or shape[4] <= 0 or shape[3] % 2 or shape[4] % 2:
            raise TrainingContractError(
                f"{role} latent spatial dimensions must be positive/even: {shape[3:]}"
            )
        shapes.append(shape)
    if shapes[0] != shapes[1]:
        raise TrainingContractError(
            f"paired source/target latent geometry differs: {shapes[0]} vs {shapes[1]}"
        )
    return shapes[0], shapes[1]


def collate_single_renderer_sample(sample: Any) -> dict[str, Any]:
    """Apply Bernini's packing contract to exactly one transformed sample."""

    import torch

    if isinstance(sample, list):
        if len(sample) != 1 or not isinstance(sample[0], Mapping):
            raise TrainingContractError("official transform must return one sample")
        sample = sample[0]
    if not isinstance(sample, Mapping):
        raise TrainingContractError("collate input must be a transformed mapping")
    missing = sorted(REQUIRED_MODEL_KEYS - sample.keys())
    if missing:
        raise TrainingContractError(f"transformed sample is missing: {missing}")
    batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in PACKING_KEYS:
            if not isinstance(value, torch.Tensor):
                raise TrainingContractError(f"packing field {key} must be a tensor")
            batch[key] = value.unsqueeze(0)
        elif key in CONCAT_KEYS:
            if not isinstance(value, torch.Tensor):
                raise TrainingContractError(f"concat field {key} must be a tensor")
            # Crucial: do not introduce a batch dimension here.  Official
            # GEN_Wanx22 expects the already-concatenated token dimension.
            batch[key] = value
    missing_after = sorted(REQUIRED_MODEL_KEYS - batch.keys())
    if missing_after:
        raise TrainingContractError(f"collated sample is missing: {missing_after}")
    return batch


def validate_collated_supervision(batch: Mapping[str, Any]) -> None:
    """Check clean-source/noisy-target packing without any spatial edit mask."""

    mask = batch["vae_latents_mask"]
    if getattr(mask, "ndim", None) != 2 or int(mask.shape[0]) != 1:
        raise TrainingContractError("internal target selector must be packed as [1,N]")
    selector = mask.squeeze(0).bool()
    total = int(selector.numel())
    target_count = int(selector.sum().item())
    source_count = total - target_count
    if source_count <= 0 or target_count <= 0 or source_count != target_count:
        raise TrainingContractError(
            "paired 81f geometry requires equal non-empty source/target token spans"
        )
    # The official ordering is clean source first and supervised target second.
    if bool(selector[:source_count].any()) or not bool(selector[source_count:].all()):
        raise TrainingContractError("target selector is not source-then-target contiguous")
    latents = batch["input_vae_latents"]
    rope = batch["input_vae_rope"]
    target = batch["target_velocity"]
    if int(latents.shape[0]) != total or int(rope.shape[0]) != total:
        raise TrainingContractError("latent/rope length disagrees with packed selector")
    if int(target.shape[0]) != target_count:
        raise TrainingContractError("velocity target length disagrees with target span")
    target_lens = batch["target_lens"].reshape(-1)
    positive_target_lens = target_lens[target_lens > 0]
    if positive_target_lens.numel() != 1 or int(positive_target_lens[0]) != target_count:
        raise TrainingContractError("single target_lens entry must cover the target span")


def step_seed(base_seed: int, global_step: int, row_index: int) -> int:
    payload = f"{int(base_seed)}:{int(global_step)}:{int(row_index)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def seed_same_sample(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DistributedContract:
    world_size: int
    rank: int
    local_rank: int
    ulysses_size: int


def distributed_contract(environment: Mapping[str, str] = os.environ) -> DistributedContract:
    world = int(environment.get("WORLD_SIZE", "1"))
    rank = int(environment.get("RANK", "0"))
    local_rank = int(environment.get("LOCAL_RANK", "0"))
    if world not in (1, 4):
        raise TrainingContractError(
            f"supported world sizes are 1 (debug) and 4 (AUH RCCL), got {world}"
        )
    if not 0 <= rank < world or not 0 <= local_rank < world:
        raise TrainingContractError(
            f"invalid torchrun ranks: rank={rank}, local_rank={local_rank}, world={world}"
        )
    return DistributedContract(world, rank, local_rank, 4 if world == 4 else 1)


def initialise_distributed(contract: DistributedContract) -> tuple[Any, str]:
    """Initialize NCCL, which is the PyTorch frontend for RCCL on ROCm."""

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise TrainingContractError("Bernini training requires a ROCm-visible GPU")
    if contract.world_size == 4 and getattr(torch.version, "hip", None) is None:
        raise TrainingContractError(
            "four-rank production launch requires ROCm; NCCL must resolve to RCCL"
        )
    torch.cuda.set_device(contract.local_rank)
    if contract.world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    if contract.world_size > 1:
        if dist.get_world_size() != 4 or dist.get_rank() != contract.rank:
            raise TrainingContractError("initialized process group violates torchrun contract")
    return torch.device("cuda", contract.local_rank), (
        "nccl/rccl" if getattr(torch.version, "hip", None) is not None else "nccl"
    )


def _distributed_boolean(value: bool, *, op: str) -> bool:
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(int(value), dtype=torch.int32, device="cuda")
    reduce_op = dist.ReduceOp.MIN if op == "all" else dist.ReduceOp.MAX
    dist.all_reduce(tensor, op=reduce_op)
    return bool(tensor.item())


def assert_identical_row(identity: str) -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    gathered: list[Optional[str]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, identity)
    if len(set(gathered)) != 1:
        raise TrainingContractError(f"ranks selected different rows: {gathered}")


def trainable_parameters_digest(
    named_parameters: Sequence[tuple[str, Any]],
) -> str:
    """Hash names, tensor metadata, and exact bytes for a replicated adapter."""

    import torch

    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        tensor = parameter.detach().contiguous()
        byte_view = tensor.view(torch.uint8).cpu()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(byte_view.numpy().tobytes(order="C"))
    return digest.hexdigest()


def synchronize_trainable_parameters(
    named_parameters: Sequence[tuple[str, Any]],
    *,
    source_rank: int = 0,
    dist_module: Optional[Any] = None,
    digest_function: Optional[Any] = None,
) -> str:
    """Make every replicated LoRA tensor bit-identical before optimizer setup.

    PEFT initializes LoRA-A randomly.  Independent torchrun processes therefore
    cannot rely on construction order alone, even when they later consume the
    same sample and seed.  Ulysses shards activations rather than parameters, so
    all replicated adapter tensors must start from one common rank-0 state.
    """

    if not named_parameters:
        raise TrainingContractError("cannot synchronize an empty LoRA parameter set")
    if dist_module is None:
        import torch.distributed as dist_module
    if digest_function is None:
        digest_function = trainable_parameters_digest

    if not dist_module.is_available() or not dist_module.is_initialized():
        return str(digest_function(named_parameters))
    world_size = int(dist_module.get_world_size())
    if source_rank < 0 or source_rank >= world_size:
        raise TrainingContractError(
            f"LoRA broadcast source rank {source_rank} is outside world size {world_size}"
        )
    for name, parameter in named_parameters:
        if not bool(getattr(parameter, "requires_grad", False)):
            raise TrainingContractError(
                f"non-trainable parameter included in LoRA synchronization: {name}"
            )
        dist_module.broadcast(parameter.data, src=source_rank)
    # Keep model construction, adapter broadcast, and optimizer construction in
    # a strict global order.  This is also required on resume before optimizer
    # state is restored independently on each rank.
    dist_module.barrier()
    local_digest = str(digest_function(named_parameters))
    gathered: list[Optional[str]] = [None] * world_size
    dist_module.all_gather_object(gathered, local_digest)
    if len(set(gathered)) != 1:
        raise TrainingContractError(
            f"LoRA tensors differ after rank-0 broadcast: {gathered}"
        )
    return local_digest


_SERIALIZED_CONSTRUCTION_STATUS_SCHEMA = (
    "bernini-r-world4-rank-serialized-construction-status-v1"
)


def _trainable_metadata(
    named_parameters: Sequence[tuple[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, parameter in named_parameters:
        shape = [int(value) for value in tuple(getattr(parameter, "shape", ()))]
        numel = int(parameter.numel())
        dtype = str(getattr(parameter, "dtype", ""))
        if (
            not name
            or not shape
            or numel <= 0
            or not dtype
            or not bool(getattr(parameter, "requires_grad", False))
        ):
            raise TrainingContractError(
                "serialized construction trainable metadata differs"
            )
        rows.append(
            {"name": name, "shape": shape, "dtype": dtype, "numel": numel}
        )
    names = [str(row["name"]) for row in rows]
    if not rows or len(set(names)) != len(names):
        raise TrainingContractError(
            "serialized construction trainable parameter names differ"
        )
    return rows


def _trim_host_allocator_after_model_move(
    *, torch_module: Any, gc_module: Any = gc, ctypes_module: Any = ctypes
) -> None:
    gc_module.collect()
    torch_module.cuda.empty_cache()
    try:
        libc = ctypes_module.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = (ctypes_module.c_size_t,)
        malloc_trim.restype = ctypes_module.c_int
        result = malloc_trim(0)
    except (AttributeError, OSError, TypeError) as error:
        raise TrainingContractError(
            f"glibc malloc_trim is unavailable for serialized construction: {error}"
        ) from error
    if type(result) is not int:
        raise TrainingContractError(
            "glibc malloc_trim returned a non-integer construction result"
        )


def _serialized_construction_success_status(
    *,
    active_rank: int,
    device: Any,
    target_modules: Sequence[str],
    named_trainable: Sequence[tuple[str, Any]],
    trainable_count: int,
) -> dict[str, Any]:
    metadata = _trainable_metadata(named_trainable)
    if (
        type(active_rank) is not int
        or active_rank not in range(4)
        or list(target_modules) != sorted(target_modules)
        or len(set(target_modules)) != len(target_modules)
        or type(trainable_count) is not int
        or trainable_count != sum(int(row["numel"]) for row in metadata)
    ):
        raise TrainingContractError(
            "serialized construction success metadata differs"
        )
    return {
        "schema_version": _SERIALIZED_CONSTRUCTION_STATUS_SCHEMA,
        "active_rank": active_rank,
        "ok": True,
        "device": str(device),
        "target_module_count": len(target_modules),
        "target_modules_sha256": object_sha256(list(target_modules)),
        "trainable_parameter_count": trainable_count,
        "trainable_metadata_sha256": object_sha256(metadata),
        "host_allocator_trim_called": True,
    }


def _validate_serialized_construction_status(
    value: Any, *, active_rank: int
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingContractError(
            "serialized construction rank status is not one mapping"
        )
    common = {
        "schema_version",
        "active_rank",
        "ok",
        "device",
    }
    if (
        value.get("schema_version") != _SERIALIZED_CONSTRUCTION_STATUS_SCHEMA
        or value.get("active_rank") != active_rank
        or value.get("device") != f"cuda:{active_rank}"
        or type(value.get("ok")) is not bool
    ):
        raise TrainingContractError(
            "serialized construction rank status identity differs"
        )
    if value.get("ok") is False:
        if set(value) != common | {"error_type", "error_message_sha256"} or (
            not isinstance(value.get("error_type"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("error_message_sha256")))
            is None
        ):
            raise TrainingContractError(
                "serialized construction failure status schema differs"
            )
        return value
    expected = common | {
        "target_module_count",
        "target_modules_sha256",
        "trainable_parameter_count",
        "trainable_metadata_sha256",
        "host_allocator_trim_called",
    }
    if (
        set(value) != expected
        or type(value.get("target_module_count")) is not int
        or int(value["target_module_count"]) <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("target_modules_sha256")))
        is None
        or type(value.get("trainable_parameter_count")) is not int
        or int(value["trainable_parameter_count"]) <= 0
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("trainable_metadata_sha256"))
        )
        is None
        or value.get("host_allocator_trim_called") is not True
    ):
        raise TrainingContractError(
            "serialized construction success status schema differs"
        )
    return value


def world4_rank_serialized_model_construction(
    *,
    contract: DistributedContract,
    device: Any,
    build_function: Any,
    torch_module: Any,
    dist_module: Any,
    trim_function: Any = _trim_host_allocator_after_model_move,
) -> tuple[Any, list[Mapping[str, Any]]]:
    """Construct one full renderer at a time to bound the WORLD4 host peak."""

    if contract.world_size == 1:
        return build_function(), []
    if (
        contract.world_size != 4
        or not dist_module.is_available()
        or not dist_module.is_initialized()
        or int(dist_module.get_world_size()) != 4
        or int(dist_module.get_rank()) != contract.rank
    ):
        raise TrainingContractError(
            "serialized construction requires the exact WORLD4 process group"
        )
    local_result: Any = None
    statuses: list[Mapping[str, Any]] = []
    for active_rank in range(4):
        local_status: Optional[dict[str, Any]] = None
        if contract.rank == active_rank:
            try:
                local_result = build_function()
                model, target_modules, named_trainable, trainable_count = local_result
                torch_module.cuda.synchronize(device)
                trim_function(torch_module=torch_module)
                local_status = _serialized_construction_success_status(
                    active_rank=active_rank,
                    device=device,
                    target_modules=target_modules,
                    named_trainable=named_trainable,
                    trainable_count=trainable_count,
                )
            except Exception as error:
                local_status = {
                    "schema_version": _SERIALIZED_CONSTRUCTION_STATUS_SCHEMA,
                    "active_rank": active_rank,
                    "ok": False,
                    "device": str(device),
                    "error_type": type(error).__name__,
                    "error_message_sha256": hashlib.sha256(
                        str(error).encode("utf-8", errors="replace")
                    ).hexdigest(),
                }
        payload: list[Any] = [local_status]
        dist_module.broadcast_object_list(payload, src=active_rank)
        status = _validate_serialized_construction_status(
            payload[0], active_rank=active_rank
        )
        statuses.append(status)
        if status.get("ok") is not True:
            raise TrainingContractError(
                "serialized model construction failed on rank "
                f"{active_rank}: {status.get('error_type')} "
                f"{status.get('error_message_sha256')}"
            )
        dist_module.barrier()
    if local_result is None:
        raise TrainingContractError(
            "serialized construction did not build this rank's model"
        )
    comparable = {
        (
            status.get("target_module_count"),
            status.get("target_modules_sha256"),
            status.get("trainable_parameter_count"),
            status.get("trainable_metadata_sha256"),
        )
        for status in statuses
    }
    if len(comparable) != 1:
        raise TrainingContractError(
            "WORLD4 serialized model construction metadata differs"
        )
    return local_result, statuses


def all_reduce_lora_gradients(
    named_parameters: Sequence[tuple[str, Any]], *, bucket_bytes: int = 64 * 1024 * 1024
) -> float:
    """Finite-gate and explicitly average replicated LoRA gradients."""

    import torch
    import torch.distributed as dist

    missing = [name for name, param in named_parameters if param.grad is None]
    if missing:
        raise TrainingContractError(f"LoRA parameters have no gradient: {missing[:8]}")
    local_finite = all(
        bool(torch.isfinite(param.grad).all().item()) for _, param in named_parameters
    )
    if not _distributed_boolean(local_finite, op="all"):
        raise TrainingContractError("non-finite LoRA gradient; optimizer step blocked")

    # Group only identical dtype/device tensors.  A moderate flat bucket avoids
    # hundreds of tiny RCCL collectives without an excessive temporary tensor.
    grouped: dict[tuple[Any, Any], list[Any]] = {}
    for _, param in named_parameters:
        grouped.setdefault((param.grad.device, param.grad.dtype), []).append(param.grad)
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    for gradients in grouped.values():
        bucket: list[Any] = []
        size = 0

        def flush() -> None:
            nonlocal bucket, size
            if not bucket:
                return
            flat = torch.cat([gradient.reshape(-1) for gradient in bucket])
            if world > 1:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(world)
            offset = 0
            for gradient in bucket:
                count = gradient.numel()
                gradient.copy_(flat[offset : offset + count].view_as(gradient))
                offset += count
            bucket = []
            size = 0

        for gradient in gradients:
            nbytes = gradient.numel() * gradient.element_size()
            if bucket and size + nbytes > bucket_bytes:
                flush()
            bucket.append(gradient)
            size += nbytes
        flush()

    post_finite = all(
        bool(torch.isfinite(param.grad).all().item()) for _, param in named_parameters
    )
    if not _distributed_boolean(post_finite, op="all"):
        raise TrainingContractError("non-finite all-reduced LoRA gradient")
    squared = torch.zeros((), dtype=torch.float64, device=named_parameters[0][1].grad.device)
    for _, param in named_parameters:
        squared += param.grad.detach().double().pow(2).sum()
    norm = math.sqrt(float(squared.item()))
    if not math.isfinite(norm):
        raise TrainingContractError("non-finite LoRA gradient norm")
    return norm


class ParquetRowStore:
    """Deterministic random access over parquet row groups with one-group cache."""

    def __init__(self, directory: str | Path):
        root = _absolute_existing_directory(directory, label="preprocessed_parquet_dir")
        files = self._snapshot_parquet_files(root)
        if not files:
            raise TrainingContractError(f"no parquet files in {root}")
        self.root = root
        self.files = files
        self._groups: list[tuple[int, int, Path, int]] = []
        self._ends: list[int] = []
        # Do not inspect row groups or schemas from mutable pathnames here.
        # ``bind_indexed_file_hashes`` rebuilds all layout metadata from the
        # exact index-authorized bytes held in memory.
        self._length: Optional[int] = None
        self._schema: Optional[str] = None
        self._layout_signature: Optional[str] = None
        self._cached_key: Optional[tuple[Path, int]] = None
        self._cached_rows: Optional[list[dict[str, Any]]] = None
        self._expected_file_sha256: Optional[dict[Path, str]] = None
        self.content_signature: Optional[str] = None
        self.signature: Optional[str] = None

    @staticmethod
    def _snapshot_parquet_files(root: Path) -> tuple[Path, ...]:
        """Return a stable exact set of regular ``*.parquet`` children."""

        try:
            root_before = root.lstat()
            children = tuple(root.iterdir())
            root_after = root.lstat()
        except OSError as error:
            raise TrainingContractError(
                f"cannot enumerate preprocessed parquet directory: {error}"
            ) from error
        root_identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_mtime_ns,
        )
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or stat.S_ISLNK(root_before.st_mode)
            or root_identity(root_before) != root_identity(root_after)
        ):
            raise TrainingContractError("preprocessed parquet directory changed")
        files: list[Path] = []
        for path in children:
            if path.suffix != ".parquet":
                continue
            try:
                info = path.lstat()
            except OSError as error:
                raise TrainingContractError(
                    f"cannot inspect parquet child {path}: {error}"
                ) from error
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise TrainingContractError(
                    f"parquet child is not a plain regular file: {path}"
                )
            files.append(path)
        return tuple(sorted(files))

    def __len__(self) -> int:
        if self._length is None:
            raise TrainingContractError(
                "dataset length is unavailable before index hashes are bound"
            )
        return self._length

    def bind_indexed_file_hashes(self, expected: Mapping[Path, str]) -> None:
        """Bind all subsequent reads to the exact index-authorized shard bytes."""

        normalized: dict[Path, str] = {}
        for path_value, sha256 in expected.items():
            path = Path(path_value)
            if (
                path.parent != self.root
                or path not in self.files
                or type(sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            ):
                raise TrainingContractError("indexed shard hash map differs")
            normalized[path] = sha256
        if set(normalized) != set(self.files):
            raise TrainingContractError("indexed shard hash map membership differs")
        if self._expected_file_sha256 is not None and normalized != self._expected_file_sha256:
            raise TrainingContractError("indexed shard hash map changed")

        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise TrainingContractError("pyarrow is required to read parquet") from error
        groups: list[tuple[int, int, Path, int]] = []
        ends: list[int] = []
        total = 0
        schemas: set[str] = set()
        layout_rows: list[dict[str, Any]] = []
        for path in sorted(normalized):
            raw = self._stable_plain_file_bytes(path, normalized[path])
            try:
                parquet = pq.ParquetFile(io.BytesIO(raw))
            except Exception as error:
                raise TrainingContractError(
                    f"cannot parse index-authorized parquet bytes: {path}: {error}"
                ) from error
            schema = str(parquet.schema_arrow)
            schemas.add(schema)
            row_group_sizes: list[int] = []
            for row_group in range(parquet.metadata.num_row_groups):
                rows = int(parquet.metadata.row_group(row_group).num_rows)
                if rows <= 0:
                    raise TrainingContractError(
                        f"indexed parquet contains an empty row group: {path}"
                    )
                start, end = total, total + rows
                groups.append((start, end, path, row_group))
                ends.append(end)
                row_group_sizes.append(rows)
                total = end
            layout_rows.append(
                {
                    "name": path.name,
                    "sha256": normalized[path],
                    "row_groups": row_group_sizes,
                }
            )
        if total <= 0:
            raise TrainingContractError("preprocessed parquet dataset is empty")
        if len(schemas) != 1:
            raise TrainingContractError("all parquet shards must have identical schemas")
        schema = next(iter(schemas))
        for required in ("inputs", "video_vae_latents"):
            if required not in schema:
                raise TrainingContractError(f"parquet schema is missing {required}")
        layout_signature = object_sha256(
            {"files": layout_rows, "rows": total, "schema": schema}
        )
        if (
            self._layout_signature is not None
            and layout_signature != self._layout_signature
        ):
            raise TrainingContractError("indexed parquet layout changed")

        self._expected_file_sha256 = dict(normalized)
        self._groups = groups
        self._ends = ends
        self._length = total
        self._schema = schema
        self._layout_signature = layout_signature
        self._cached_key = None
        self._cached_rows = None
        self.content_signature = object_sha256(
            {
                "root": str(self.root),
                "layout_signature": layout_signature,
                "files": layout_rows,
                "rows": total,
            }
        )
        self.signature = self.content_signature

    @staticmethod
    def _stable_plain_file_bytes(
        path: Path, expected_sha256: Optional[str] = None
    ) -> bytes:
        """Read one regular non-symlink file and prove path/FD stability."""

        try:
            path_before = path.lstat()
        except OSError as error:
            raise TrainingContractError(
                f"indexed shard is unavailable while reading: {path}: {error}"
            ) from error
        if not stat.S_ISREG(path_before.st_mode) or stat.S_ISLNK(path_before.st_mode):
            raise TrainingContractError(f"indexed shard file type differs: {path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise TrainingContractError(
                f"cannot open indexed shard without following links: {path}: {error}"
            ) from error
        try:
            fd_before = os.fstat(descriptor)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                raw = handle.read()
            fd_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        try:
            path_after = path.lstat()
        except OSError as error:
            raise TrainingContractError(
                f"indexed shard disappeared while reading: {path}: {error}"
            ) from error

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
            )

        observed = identity(path_before)
        if (
            observed != identity(fd_before)
            or observed != identity(fd_after)
            or observed != identity(path_after)
            or len(raw) != path_before.st_size
            or (
                expected_sha256 is not None
                and hashlib.sha256(raw).hexdigest() != expected_sha256
            )
        ):
            raise TrainingContractError(
                f"indexed shard identity changed or hash differs: {path}"
            )
        return raw

    def revalidate_bound_files(self) -> str:
        """Re-read every shard for the terminal exact-input closure."""

        if self._expected_file_sha256 is None or self.content_signature is None:
            raise TrainingContractError("dataset shard hashes were not bound")
        if self._snapshot_parquet_files(self.root) != self.files:
            raise TrainingContractError("parquet directory membership changed")
        # Rebuild row-group/schema metadata from the same stable authorized
        # bytes; this detects both byte replacement and layout drift.
        self.bind_indexed_file_hashes(self._expected_file_sha256)
        return self.content_signature

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        group_index = bisect.bisect_right(self._ends, index)
        start, _, path, row_group = self._groups[group_index]
        key = (path, row_group)
        if self._cached_key != key:
            import pyarrow.parquet as pq

            if self._expected_file_sha256 is None:
                raise TrainingContractError(
                    "dataset rows cannot be read before index hashes are bound"
                )
            raw = self._stable_plain_file_bytes(
                path, self._expected_file_sha256[path]
            )
            self._cached_rows = (
                pq.ParquetFile(io.BytesIO(raw)).read_row_group(row_group).to_pylist()
            )
            self._cached_key = key
        assert self._cached_rows is not None
        return self._cached_rows[index - start]


def validate_preprocessed_dataset_summary(
    summary_value: str | Path,
    dataset: ParquetRowStore,
    *,
    allow_incomplete: bool,
    allow_reward_selected_synthetic_targets: bool = False,
) -> dict[str, Any]:
    """Bind training to the finalized 644-row release and every indexed shard."""

    summary_path = Path(summary_value).expanduser()
    if not summary_path.is_absolute():
        raise TrainingContractError("dataset summary must be an absolute local path")
    try:
        summary_path = summary_path.resolve(strict=True)
    except OSError as error:
        raise TrainingContractError(f"dataset summary is unavailable: {error}") from error
    if not summary_path.is_file() or summary_path.is_symlink():
        raise TrainingContractError("dataset summary must be a plain file")
    summary_raw = ParquetRowStore._stable_plain_file_bytes(summary_path)
    summary_sha256 = hashlib.sha256(summary_raw).hexdigest()
    try:
        summary = json.loads(summary_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TrainingContractError(
            f"cannot parse stable dataset summary bytes: {error}"
        ) from error
    if not isinstance(summary, dict):
        raise TrainingContractError("dataset summary must be a JSON object")
    candidate = dict(summary)
    declared_digest = candidate.pop("summary_digest", None)
    schema_version = summary.get("schema_version")
    if object_sha256(candidate) != declared_digest:
        raise TrainingContractError("dataset summary schema or digest differs")
    reward_selected = schema_version == REWARD_SELECTED_DATASET_SUMMARY_SCHEMA
    if reward_selected and not allow_reward_selected_synthetic_targets:
        raise TrainingContractError(
            "reward-selected synthetic targets require the explicit experimental flag"
        )
    if schema_version not in {
        VAE_DATASET_SUMMARY_SCHEMA,
        REWARD_SELECTED_DATASET_SUMMARY_SCHEMA,
    }:
        raise TrainingContractError("dataset summary schema or digest differs")
    if (
        summary.get("preview_only") is not True
        or summary.get("training_authorized") is not False
        or summary.get("training_use_forbidden") is not True
        or summary.get("experimental_training_acknowledged") is not True
        or summary.get("production_claim_forbidden") is not True
        or summary.get("scientific_claim_authorized") is not False
    ):
        raise TrainingContractError("dataset summary authorization state differs")
    expected_rows = summary.get("expected_sample_count")
    materialized_rows = summary.get("materialized_sample_count")
    missing_rows = summary.get("missing_sample_count")
    if reward_selected:
        if (
            summary.get("reward_selected_synthetic_target") is not True
            or summary.get("same_source_instruction_rows_across_arms") is not True
            or summary.get("arm")
            not in {"baseline", "action_only", "preservation_only", "composite"}
            or expected_rows != 4
            or materialized_rows != 4
            or missing_rows != 0
            or summary.get("complete") is not True
            or summary.get("frame_count") != NUM_FRAMES
            or float(summary.get("fps", -1.0)) != 25.0
            or summary.get("latent_frame_count") != LATENT_FRAMES
        ):
            raise TrainingContractError("reward-selected dataset contract differs")
    elif summary.get("experimental_inclusion_policy") != EXPECTED_INCLUSION_POLICY:
        raise TrainingContractError("dataset summary inclusion policy differs")
    if not reward_selected and (
        expected_rows != EXPECTED_DATASET_ROWS
        or type(materialized_rows) is not int
        or type(missing_rows) is not int
        or missing_rows != expected_rows - materialized_rows
        or materialized_rows <= 0
    ):
        raise TrainingContractError("dataset summary row counts differ")
    complete = summary.get("complete")
    if type(complete) is not bool or complete != (missing_rows == 0):
        raise TrainingContractError("dataset summary completion state differs")
    if not complete and not allow_incomplete:
        raise TrainingContractError(
            "dataset is incomplete; only an explicit smoke/canary launch may allow it"
        )
    if not reward_selected and (
        summary.get("raw_strict_selection_rows") != EXPECTED_STRICT_ROWS
        or summary.get("raw_non_strict_selection_rows") != EXPECTED_NON_STRICT_ROWS
        or summary.get("materialized_strict_selection_rows", -1)
        + summary.get("materialized_non_strict_selection_rows", -1)
        != materialized_rows
        or summary.get("frame_count") != NUM_FRAMES
        or float(summary.get("fps", -1.0)) != 25.0
        or summary.get("latent_frame_count") != LATENT_FRAMES
    ):
        raise TrainingContractError("dataset summary cohort or media contract differs")
    bucket_counts = summary.get("bucket_counts")
    if (
        not isinstance(bucket_counts, Mapping)
        or any(type(value) is not int or value <= 0 for value in bucket_counts.values())
        or sum(bucket_counts.values()) != materialized_rows
    ):
        raise TrainingContractError("dataset summary bucket counts differ")
    try:
        shards_directory = Path(str(summary.get("shards_directory"))).resolve(strict=True)
    except OSError as error:
        raise TrainingContractError(f"dataset shard directory is unavailable: {error}") from error
    if shards_directory != dataset.root:
        raise TrainingContractError("dataset summary points to a different shard directory")

    try:
        index_path = Path(str(summary.get("index_path"))).resolve(strict=True)
    except OSError as error:
        raise TrainingContractError(f"dataset index is unavailable: {error}") from error
    if not index_path.is_file() or index_path.is_symlink():
        raise TrainingContractError("dataset index must be a plain file")
    index_raw = ParquetRowStore._stable_plain_file_bytes(index_path)
    index_sha256 = hashlib.sha256(index_raw).hexdigest()
    if index_sha256 != summary.get("index_sha256"):
        raise TrainingContractError("dataset index hash differs")
    index_rows: list[dict[str, Any]] = []
    try:
        index_text = index_raw.decode("utf-8")
        for line_number, line in enumerate(index_text.splitlines(), 1):
            if not line.strip():
                raise TrainingContractError(
                    f"blank dataset index row at line {line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TrainingContractError(
                    f"dataset index row {line_number} is not an object"
                )
            index_rows.append(value)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TrainingContractError(f"cannot parse dataset index: {error}") from error
    if len(index_rows) != materialized_rows:
        raise TrainingContractError("dataset index row count differs")

    indexed_files: list[Path] = []
    indexed_hashes: dict[Path, str] = {}
    seen_iids: set[str] = set()
    for row in index_rows:
        iid = row.get("iid")
        if (
            row.get("schema_version") != VAE_DATASET_INDEX_ROW_SCHEMA
            or type(iid) is not str
            or not iid
            or iid in seen_iids
        ):
            raise TrainingContractError("dataset index schema or IID differs")
        seen_iids.add(iid)
        try:
            shard = Path(str(row.get("parquet_path"))).resolve(strict=True)
        except OSError as error:
            raise TrainingContractError(f"indexed shard is unavailable: {iid}: {error}") from error
        expected_shard_sha256 = row.get("parquet_sha256")
        if (
            not shard.is_file()
            or shard.is_symlink()
            or shard.parent != dataset.root
            or shard.name != f"{iid}.parquet"
            or type(expected_shard_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_shard_sha256) is None
        ):
            raise TrainingContractError(f"indexed shard identity differs: {iid}")
        indexed_files.append(shard)
        indexed_hashes[shard] = expected_shard_sha256
    if tuple(sorted(indexed_files)) != tuple(sorted(dataset.files)):
        raise TrainingContractError("dataset index membership differs from parquet directory")
    if isinstance(dataset, ParquetRowStore):
        dataset.bind_indexed_file_hashes(indexed_hashes)
    else:
        for path in sorted(indexed_hashes):
            ParquetRowStore._stable_plain_file_bytes(path, indexed_hashes[path])
    if materialized_rows != len(dataset):
        raise TrainingContractError("dataset summary row counts differ")

    return {
        "path": str(summary_path),
        "sha256": summary_sha256,
        "summary_digest": declared_digest,
        "complete": complete,
        "allow_incomplete": bool(allow_incomplete),
        "expected_rows": expected_rows,
        "materialized_rows": materialized_rows,
        "index_path": str(index_path),
        "index_sha256": index_sha256,
        "indexed_shards_sha256": object_sha256(
            [
                {"name": path.name, "sha256": indexed_hashes[path]}
                for path in sorted(indexed_hashes)
            ]
        ),
        "dataset_content_signature": (
            dataset.content_signature
            if isinstance(dataset, ParquetRowStore)
            else None
        ),
        "reward_selected_synthetic_targets": reward_selected,
        "arm": summary.get("arm") if reward_selected else None,
    }


def dataset_identity(row: Mapping[str, Any], row_index: int) -> str:
    iid = row.get("iid", row.get("id", ""))
    return object_sha256(
        {"row_index": int(row_index), "iid": str(iid), "inputs": row.get("inputs")}
    )


def full644_one_pass_row_index(global_step: int, dataset_rows: int) -> int:
    """Return the exact no-replacement row for the exploratory one-pass profile."""

    if type(global_step) is not int or type(dataset_rows) is not int:
        raise TrainingContractError("full644 row selection requires exact integers")
    if dataset_rows != FULL644_EXPLORATORY_STEPS:
        raise TrainingContractError("full644 row selection requires exact644 rows")
    if not 0 <= global_step < FULL644_EXPLORATORY_STEPS:
        raise TrainingContractError("full644 row selection escaped steps 0..643")
    return global_step


def validate_full644_trainable_parameter_count(
    value: int, *, profile_enabled: bool
) -> None:
    """Reject a changed R64 route before the first optimizer construction."""

    if type(profile_enabled) is not bool or type(value) is not int or value <= 0:
        raise TrainingContractError("trainable parameter count contract differs")
    if (
        profile_enabled
        and value != FULL644_EXPLORATORY_TRAINABLE_PARAMETER_COUNT
    ):
        raise TrainingContractError(
            "full644 R64 trainable parameter count differs: "
            f"{value} != {FULL644_EXPLORATORY_TRAINABLE_PARAMETER_COUNT}"
        )


def validate_full644_peft_construction(
    config: Mapping[str, Any],
    *,
    peft_version: str,
    expected_target_modules: Sequence[str],
    profile_enabled: bool,
) -> None:
    """Close every PEFT 0.19.1 LoRA semantic before model/optimizer setup."""

    if type(profile_enabled) is not bool:
        raise TrainingContractError("full644 PEFT profile flag must be bool")
    if not profile_enabled:
        return
    if type(peft_version) is not str or peft_version != FULL644_PEFT_VERSION:
        raise TrainingContractError(
            "full644 training requires runtime PEFT "
            f"{FULL644_PEFT_VERSION}, got {peft_version!r}"
        )
    if not isinstance(config, Mapping):
        raise TrainingContractError("full644 PEFT LoraConfig mapping differs")
    if set(config) != FULL644_PEFT_LORA_CONFIG_FIELDS:
        raise TrainingContractError(
            "full644 PEFT 0.19.1 LoraConfig field closure differs"
        )
    if (
        isinstance(expected_target_modules, (str, bytes))
        or len(expected_target_modules) != EXPECTED_LORA_TARGET_MODULES
        or not all(type(name) is str and name for name in expected_target_modules)
        or len(set(expected_target_modules)) != EXPECTED_LORA_TARGET_MODULES
        or list(expected_target_modules) != sorted(expected_target_modules)
    ):
        raise TrainingContractError("full644 expected PEFT target scope differs")
    observed_targets = config.get("target_modules")
    # PEFT 0.19.1 converts a list passed to LoraConfig into a set in
    # __post_init__, and dataclasses.asdict preserves that set in to_dict().
    if (
        type(observed_targets) is not set
        or len(observed_targets) != EXPECTED_LORA_TARGET_MODULES
        or not all(type(name) is str and name for name in observed_targets)
        or observed_targets != set(expected_target_modules)
    ):
        raise TrainingContractError("full644 PEFT target_modules differ")

    expected_without_targets: dict[str, Any] = {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": None,
        "base_model_name_or_path": None,
        "bias": "none",
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": FULL644_EXPLORATORY_ALPHA,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": FULL644_PEFT_VERSION,
        "qalora_group_size": 16,
        "r": FULL644_EXPLORATORY_RANK,
        "rank_pattern": {},
        "revision": None,
        "target_parameters": None,
        "task_type": None,
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }

    def same_exact_value(observed: Any, expected: Any) -> bool:
        if expected is None:
            return observed is None
        if type(expected) is bool:
            return type(observed) is bool and observed is expected
        if type(expected) is int:
            return type(observed) is int and observed == expected
        if type(expected) is float:
            return type(observed) is float and observed == expected
        if type(expected) is dict:
            return type(observed) is dict and observed == expected
        if type(expected) is str:
            # PeftType.LORA is a str Enum in PEFT 0.19.1.
            return isinstance(observed, str) and observed == expected
        return type(observed) is type(expected) and observed == expected

    for name, expected in expected_without_targets.items():
        if not same_exact_value(config.get(name), expected):
            raise TrainingContractError(
                f"full644 PEFT LoraConfig semantic differs: {name}"
            )


def validate_full644_source_authority(
    authority_value: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Bind the exploratory pass to the existing 644-source uniqueness proof."""

    if expected_sha256 != FULL644_SOURCE_AUTHORITY_SHA256:
        raise TrainingContractError("full644 source authority expected SHA differs")
    requested = Path(authority_value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise TrainingContractError(
            "full644 source authority must be one absolute non-symlink file"
        )
    try:
        path = requested.resolve(strict=True)
    except OSError as error:
        raise TrainingContractError(
            f"full644 source authority is unavailable: {error}"
        ) from error
    if path != requested or not path.is_file() or path.is_symlink():
        raise TrainingContractError("full644 source authority file type differs")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise TrainingContractError("full644 source authority changed while reading")
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise TrainingContractError("full644 source authority SHA differs")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TrainingContractError(
            "full644 source authority is not ASCII JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise TrainingContractError("full644 source authority root differs")
    unsigned = dict(value)
    declared_digest = unsigned.pop("receipt_digest", None)
    if declared_digest != object_sha256(unsigned):
        raise TrainingContractError("full644 source authority receipt seal differs")
    data = value.get("data")
    source = data.get("source_dataset") if isinstance(data, Mapping) else None
    expected = {
        "membership_rows": EXPECTED_DATASET_ROWS,
        "action_family_count": 28,
        "unique_group_id": EXPECTED_DATASET_ROWS,
        "unique_source_video_sha256": EXPECTED_DATASET_ROWS,
        "raw_parquet_sha256": (
            "706d835a8cdf924776000d69b229c272fd434a91abc8942c67dc6fd7732b7d1b"
        ),
        "vae_index_sha256": FULL644_DATASET_INDEX_SHA256,
        "vae_summary_sha256": FULL644_DATASET_SUMMARY_SHA256,
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "scientific_claim_authorized": False,
    }
    if not isinstance(source, Mapping) or any(
        source.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise TrainingContractError("full644 source authority claims differ")
    return {
        "path": str(path),
        "sha256": observed_sha256,
        "membership_rows": source["membership_rows"],
        "action_family_count": source["action_family_count"],
        "unique_group_id": source["unique_group_id"],
        "unique_source_video_sha256": source["unique_source_video_sha256"],
        "raw_parquet_sha256": source["raw_parquet_sha256"],
        "vae_index_sha256": source["vae_index_sha256"],
        "vae_summary_sha256": source["vae_summary_sha256"],
        "role": "historical_exposed_train_debug_not_heldout",
        "historical_receipt_user_authorization_is_not_current_launch_authority": True,
    }


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _vae_statistics(checkpoint: Path) -> tuple[Any, Any, int]:
    import torch

    config = _read_json(checkpoint / "vae/config.json")
    z_dim = int(config.get("z_dim", 0))
    means = config.get("latents_mean")
    stds = config.get("latents_std")
    if z_dim != 16 or not isinstance(means, list) or not isinstance(stds, list):
        raise TrainingContractError("unexpected Wan VAE latent statistics")
    if len(means) != z_dim or len(stds) != z_dim:
        raise TrainingContractError("Wan VAE mean/std dimensions are inconsistent")
    mean = torch.tensor(means, device="cpu").view(z_dim, 1, 1, 1)
    std = torch.tensor(stds, device="cpu").view(z_dim, 1, 1, 1)
    if bool((std <= 0).any()):
        raise TrainingContractError("Wan VAE latent std must be positive")
    return mean, std, z_dim


def _adapter_paths(resume: Path) -> tuple[Path, Path, Path]:
    root = resume.expanduser().resolve(strict=True)
    adapter = root / "adapter" if (root / "adapter").is_dir() else root
    state = root / "optimizer.pt"
    receipt = root / "receipt.json"
    if not (adapter / "adapter_config.json").is_file():
        raise TrainingContractError(f"resume adapter is missing: {adapter}")
    if not state.is_file() or not receipt.is_file():
        raise TrainingContractError(f"resume checkpoint is incomplete: {root}")
    return adapter, state, receipt


def _optimizer_to(optimizer: Any, device: Any) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _checkpoint_fingerprint(checkpoint: Path) -> dict[str, Any]:
    files = [
        checkpoint / "model_index.json",
        checkpoint / "transformer/config.json",
        checkpoint / "vae/config.json",
    ]
    return {
        "path": str(checkpoint),
        "configs": {
            str(path.relative_to(checkpoint)): file_sha256(path)
            for path in files
            if path.is_file()
        },
    }


def build_receipt(
    *,
    args: argparse.Namespace,
    global_step: int,
    last_loss: Optional[float],
    gradient_norm: Optional[float],
    dataset: ParquetRowStore,
    dataset_summary: Mapping[str, Any],
    checkpoint: Path,
    bernini_revision: str,
    veomni_revision: str,
    distributed: DistributedContract,
    backend: str,
    target_modules: Sequence[str],
    trainable_parameter_count: int,
    lora_initialization_digest: str,
    peft_version: str,
    transformers_version: str,
    resumed_from: Optional[str],
    full644_source_authority: Optional[Mapping[str, Any]] = None,
    terminal_dataset_reverified: bool = False,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "global_step": int(global_step),
        "max_steps": int(args.max_steps),
        "last_loss": last_loss,
        "last_preclip_gradient_norm": gradient_norm,
        "bernini_commit": bernini_revision,
        "bernini_training_files_index_sha256": object_sha256(
            BERNINI_PINNED_FILE_HASHES
        ),
        "veomni_commit": veomni_revision,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "checkpoint": _checkpoint_fingerprint(checkpoint),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "dataset": {
            "path": str(dataset.root),
            "rows": len(dataset),
            "signature": dataset.signature,
            "content_signature": dataset.content_signature,
            "summary": dict(dataset_summary),
        },
        "training_contract": {
            "model": "Bernini-R-1.3B-Diffusers renderer-only",
            "single_expert": "transformer_1",
            "noise_tmin": 0.0,
            "noise_tmax": 1.0,
            "mv2v_flow_shift": 5.0,
            "num_frames": NUM_FRAMES,
            "latent_frames": LATENT_FRAMES,
            "task_source_name": TASK_SOURCE_NAME,
            "external_spatial_mask": False,
            "external_tracking_or_swept_tube": False,
            "conditioning": ["clean_source_video_vae", "edit_instruction"],
            "supervision": ["noisy_target_video_vae", "target_velocity"],
            "target_embedding_or_caption_conditioning": False,
            "lora_rank": int(getattr(args, "lora_rank", LORA_RANK)),
            "lora_alpha": int(getattr(args, "lora_alpha", LORA_ALPHA)),
            "lora_scope": "all Wan attn1/attn2 q,k,v,out projections",
            "tokenizer_fix_mistral_regex": TOKENIZER_FIX_MISTRAL_REGEX,
            "peft_version": peft_version,
            "transformers_version": transformers_version,
            "gradient_checkpointing": True,
            "objective": args.objective,
            "preference_weight": float(args.preference_weight),
            "preference_margin": float(args.preference_margin),
            "preference_temperature": float(args.preference_temperature),
            "dpo_beta": float(args.dpo_beta),
            "preservation_weight": float(args.preservation_weight),
            "contrastive_negative_kinds": list(CONTRASTIVE_NEGATIVE_KINDS),
            "contrastive_negative_schedule": args.contrastive_negative_schedule,
            "preservation_branch": (
                "source_as_target_conditional_identity"
                if args.objective in (
                    "sft_preservation",
                    "detached_margin_preservation",
                    "reference_dpo_preservation",
                )
                else None
            ),
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "max_gradient_norm": float(args.max_grad_norm),
        },
        "distributed": {
            "world_size": distributed.world_size,
            "ulysses_size": distributed.ulysses_size,
            "backend": backend,
            "same_sample_all_ranks": True,
            "same_seed_all_ranks": True,
            "lora_initialization_seeded_all_ranks": True,
            "lora_parameters_broadcast_from_rank": 0,
            "lora_initialization_digest": lora_initialization_digest,
            "explicit_lora_gradient_all_reduce": distributed.world_size > 1,
        },
        "seed": int(args.seed),
        "target_module_count": len(target_modules),
        "target_modules_sha256": object_sha256(list(target_modules)),
        "trainable_parameter_count": int(trainable_parameter_count),
        "resumed_from": resumed_from,
        # The upstream preview authorization remains unchanged; this run is an
        # explicit experimental training operation and cannot establish a
        # production/scientific dataset claim.
        "experimental_training": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    if getattr(args, "exploratory_full644_one_pass", False) is True:
        if not 0 <= global_step <= FULL644_EXPLORATORY_STEPS:
            raise TrainingContractError("full644 receipt step is outside one pass")
        if (
            distributed.world_size != 4
            or distributed.ulysses_size != 4
            or backend != "nccl/rccl"
            or not isinstance(full644_source_authority, Mapping)
            or full644_source_authority.get("sha256")
            != FULL644_SOURCE_AUTHORITY_SHA256
            or full644_source_authority.get(
                "historical_receipt_user_authorization_is_not_current_launch_authority"
            )
            is not True
            or peft_version != FULL644_PEFT_VERSION
            or trainable_parameter_count
            != FULL644_EXPLORATORY_TRAINABLE_PARAMETER_COUNT
            or dataset.content_signature is None
            or type(terminal_dataset_reverified) is not bool
        ):
            raise TrainingContractError("full644 receipt authority differs")
        receipt["exploratory_full644"] = {
            "profile": FULL644_EXPLORATORY_PROFILE,
            "historical_train_debug_rows": EXPECTED_DATASET_ROWS,
            "optimizer_rows_consumed": global_step,
            "next_row_index": (
                global_step if global_step < FULL644_EXPLORATORY_STEPS else None
            ),
            "row_sequence_prefix": f"0..{global_step - 1}" if global_step else "empty",
            "row_sequence_sha256": object_sha256(list(range(global_step))),
            "no_replacement_within_pass": True,
            "complete_one_pass": global_step == FULL644_EXPLORATORY_STEPS,
            "historical_dataset_exists": True,
            "historical_optimizer_contribution_rows": EXPECTED_DATASET_ROWS,
            "historical_source_receipt_is_not_current_launch_authority": True,
            "runtime_data_integrity_validated": True,
            "dataset_quality_accepted_under_0817": False,
            "formal_training_dataset_authorized": False,
            "formal_heldout_contribution": 0,
            "target_scientific_qualification_complete": False,
            "matched_frozen_evaluation_required_before_claim": True,
            "resume_policy": "forbidden_for_this_profile",
            "intermediate_checkpoints_archival_only": True,
            "interrupted_run_requires_fresh_step0_restart": True,
            "dataset_summary_sha256": FULL644_DATASET_SUMMARY_SHA256,
            "dataset_summary_digest": FULL644_DATASET_SUMMARY_DIGEST,
            "dataset_index_sha256": FULL644_DATASET_INDEX_SHA256,
            "dataset_content_signature": dataset.content_signature,
            "source_authority": dict(full644_source_authority),
            "indexed_source_and_target_vae_shards_verified_before_training": True,
            "indexed_source_and_target_vae_shards_reverified_after_training": (
                terminal_dataset_reverified
            ),
        }
    elif full644_source_authority is not None or terminal_dataset_reverified is not False:
        raise TrainingContractError(
            "non-full644 receipt cannot claim full644 terminal authority"
        )
    receipt["receipt_digest"] = object_sha256(receipt)
    return receipt


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_checkpoint_content_manifest(
    root: Path, *, global_step: int, receipt_digest: str
) -> dict[str, Any]:
    """Seal every checkpoint payload file except the manifest itself."""

    if (
        type(global_step) is not int
        or global_step <= 0
        or re.fullmatch(r"[0-9a-f]{64}", receipt_digest) is None
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise TrainingContractError("checkpoint manifest input differs")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == "checkpoint_manifest.json":
            continue
        if path.is_symlink():
            raise TrainingContractError(f"checkpoint contains symlink: {relative}")
        if path.is_dir():
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TrainingContractError(
                f"checkpoint member is not one single-link regular file: {relative}"
            )
        entries.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "size": info.st_size,
            }
        )
    paths = {entry["path"] for entry in entries}
    if not {
        "adapter/adapter_config.json",
        "optimizer.pt",
        "receipt.json",
    }.issubset(paths) or not any(
        path.startswith("adapter/adapter_model") and path.endswith(".safetensors")
        for path in paths
    ):
        raise TrainingContractError("checkpoint payload closure is incomplete")
    manifest: dict[str, Any] = {
        "schema_version": "bernini-r-action-lora-checkpoint-manifest-v1",
        "global_step": global_step,
        "receipt_digest": receipt_digest,
        "file_count": len(entries),
        "entries": entries,
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    return manifest


def _atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish a checkpoint through create-only links and a negative marker.

    ``renameat2(RENAME_NOREPLACE)`` is not implemented consistently by the
    NFS/Lustre mounts used for training.  This protocol uses only atomic
    create-only namespace operations: reserve the final directory with
    ``mkdir``, keep ``.INCOMPLETE`` visible while manifest-bound regular files
    are linked into it, and remove that marker only after the stage has been
    consumed and the final tree has been replayed exactly.  Any exception
    deliberately preserves the stage/final partial state; a fresh run must use
    a fresh output namespace.
    """

    source = Path(source)
    destination = Path(destination)
    if (
        not source.is_absolute()
        or not destination.is_absolute()
        or source.parent != destination.parent
        or source.name in ("", ".", "..")
        or destination.name in ("", ".", "..")
        or "/" in source.name
        or "/" in destination.name
    ):
        raise TrainingContractError(
            "checkpoint no-replace publication requires sibling absolute paths"
        )
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise TrainingContractError(
            "checkpoint create-only publication directory flags are unavailable"
        )
    parent_fd = -1
    source_fd = -1
    destination_fd = -1
    marker_fd = -1

    def fd_digest(descriptor: int) -> str:
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return digest.hexdigest()

    def open_relative_directory(root_fd: int, relative: str) -> int:
        return os.open(
            relative if relative else ".",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )

    def fsync_relative_directory(root_fd: int, relative: str) -> None:
        descriptor = open_relative_directory(root_fd, relative)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def named_stat(root_fd: int, relative: str) -> os.stat_result:
        return os.stat(relative, dir_fd=root_fd, follow_symlinks=False)

    try:
        parent_lstat = source.parent.lstat()
        if (
            not stat.S_ISDIR(parent_lstat.st_mode)
            or source.parent.is_symlink()
            or source.parent.resolve(strict=True) != source.parent
        ):
            raise TrainingContractError(
                "checkpoint publication parent identity differs"
            )
        parent_fd = os.open(
            source.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except TrainingContractError:
        raise
    except OSError as error:
        raise TrainingContractError(
            f"cannot open checkpoint publication parent: {error}"
        ) from error
    try:
        opened_parent = os.fstat(parent_fd)
        if (
            opened_parent.st_dev != parent_lstat.st_dev
            or opened_parent.st_ino != parent_lstat.st_ino
            or not stat.S_ISDIR(opened_parent.st_mode)
        ):
            raise TrainingContractError(
                "checkpoint publication parent changed before rename"
            )
        source_fd = os.open(
            source.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        source_stat = os.fstat(source_fd)
        source_named = named_stat(parent_fd, source.name)
        if (
            not stat.S_ISDIR(source_stat.st_mode)
            or (source_stat.st_dev, source_stat.st_ino)
            != (source_named.st_dev, source_named.st_ino)
            or source_stat.st_dev != opened_parent.st_dev
        ):
            raise TrainingContractError(
                "checkpoint temporary publication source is not a directory"
            )

        manifest_path = source / "checkpoint_manifest.json"
        try:
            manifest_raw = manifest_path.read_bytes()
            manifest = json.loads(manifest_raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TrainingContractError(
                f"checkpoint publication manifest is unavailable: {error}"
            ) from error
        if not isinstance(manifest, dict):
            raise TrainingContractError("checkpoint publication manifest root differs")
        expected_manifest_keys = {
            "schema_version",
            "global_step",
            "receipt_digest",
            "file_count",
            "entries",
            "manifest_digest",
        }
        unsigned_manifest = dict(manifest)
        declared_manifest_digest = unsigned_manifest.pop("manifest_digest", None)
        global_step = manifest.get("global_step")
        if (
            set(manifest) != expected_manifest_keys
            or manifest.get("schema_version")
            != "bernini-r-action-lora-checkpoint-manifest-v1"
            or type(global_step) is not int
            or global_step <= 0
            or destination.name != f"checkpoint-{global_step:08d}"
            or declared_manifest_digest != object_sha256(unsigned_manifest)
            or manifest_raw != canonical_json_bytes(manifest) + b"\n"
        ):
            raise TrainingContractError("checkpoint publication manifest contract differs")
        expected_manifest = build_checkpoint_content_manifest(
            source,
            global_step=global_step,
            receipt_digest=str(manifest.get("receipt_digest", "")),
        )
        if manifest != expected_manifest:
            raise TrainingContractError(
                "checkpoint stage changed after its manifest was sealed"
            )
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise TrainingContractError("checkpoint publication manifest entries differ")
        relative_files: list[str] = []
        expected_directories: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
                raise TrainingContractError("checkpoint publication manifest row differs")
            relative = entry.get("path")
            if (
                not isinstance(relative, str)
                or not relative
                or relative.startswith("/")
                or any(part in ("", ".", "..") for part in relative.split("/"))
            ):
                raise TrainingContractError("checkpoint publication relative path differs")
            relative_files.append(relative)
            parts = relative.split("/")[:-1]
            for index in range(1, len(parts) + 1):
                expected_directories.add("/".join(parts[:index]))
        if relative_files != sorted(relative_files) or len(set(relative_files)) != len(
            relative_files
        ):
            raise TrainingContractError("checkpoint publication manifest order differs")
        actual_stage_files: set[str] = set()
        actual_stage_directories: set[str] = set()
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise TrainingContractError(
                    f"checkpoint stage contains a symlink: {relative}"
                )
            if stat.S_ISDIR(info.st_mode):
                if info.st_dev != source_stat.st_dev:
                    raise TrainingContractError(
                        f"checkpoint stage directory device differs: {relative}"
                    )
                actual_stage_directories.add(relative)
            elif (
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and info.st_dev == source_stat.st_dev
            ):
                actual_stage_files.add(relative)
            else:
                raise TrainingContractError(
                    f"checkpoint stage contains a non-single-link file: {relative}"
                )
        if actual_stage_directories != expected_directories or actual_stage_files != set(
            relative_files
        ) | {"checkpoint_manifest.json"}:
            raise TrainingContractError("checkpoint stage membership differs from manifest")

        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise TrainingContractError(
                f"cannot inspect checkpoint publication destination: {error}"
            ) from error
        else:
            raise TrainingContractError(
                f"refusing to overwrite checkpoint: {destination}"
            )
        try:
            os.mkdir(destination.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise TrainingContractError(
                f"refusing to overwrite checkpoint: {destination}"
            ) from error
        except OSError as error:
            raise TrainingContractError(
                f"cannot reserve checkpoint destination: {error}"
            ) from error
        destination_fd = os.open(
            destination.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        os.fchmod(destination_fd, 0o700)
        destination_stat = os.fstat(destination_fd)
        destination_named = named_stat(parent_fd, destination.name)
        if (
            not stat.S_ISDIR(destination_stat.st_mode)
            or stat.S_IMODE(destination_stat.st_mode) != 0o700
            or (destination_stat.st_dev, destination_stat.st_ino)
            != (destination_named.st_dev, destination_named.st_ino)
            or destination_stat.st_dev != source_stat.st_dev
            or os.listdir(destination_fd)
        ):
            raise TrainingContractError("reserved checkpoint destination identity differs")

        marker_value = {
            "schema_version": "bernini-r-action-lora-checkpoint-incomplete-v1",
            "global_step": global_step,
            "stage_name": source.name,
            "destination_name": destination.name,
            "checkpoint_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }
        marker_raw = canonical_json_bytes(marker_value) + b"\n"
        marker_fd = os.open(
            ".INCOMPLETE",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=destination_fd,
        )
        os.fchmod(marker_fd, 0o400)
        marker_view = memoryview(marker_raw)
        while marker_view:
            written = os.write(marker_fd, marker_view)
            if written <= 0:
                raise TrainingContractError("checkpoint incomplete marker write stalled")
            marker_view = marker_view[written:]
        os.fsync(marker_fd)
        marker_stat = os.fstat(marker_fd)
        marker_named = named_stat(destination_fd, ".INCOMPLETE")
        if (
            not stat.S_ISREG(marker_stat.st_mode)
            or marker_stat.st_nlink != 1
            or stat.S_IMODE(marker_stat.st_mode) != 0o400
            or (marker_stat.st_dev, marker_stat.st_ino)
            != (marker_named.st_dev, marker_named.st_ino)
            or marker_stat.st_size != len(marker_raw)
            or fd_digest(marker_fd) != hashlib.sha256(marker_raw).hexdigest()
        ):
            raise TrainingContractError("checkpoint incomplete marker identity differs")
        os.fsync(destination_fd)
        os.fsync(parent_fd)

        for relative in sorted(expected_directories, key=lambda item: (item.count("/"), item)):
            try:
                os.mkdir(relative, 0o700, dir_fd=destination_fd)
            except OSError as error:
                raise TrainingContractError(
                    f"cannot create checkpoint destination directory {relative}: {error}"
                ) from error
            destination_directory_fd = open_relative_directory(
                destination_fd, relative
            )
            try:
                os.fchmod(destination_directory_fd, 0o700)
                destination_directory_opened = os.fstat(destination_directory_fd)
            finally:
                os.close(destination_directory_fd)
            source_directory = named_stat(source_fd, relative)
            destination_directory = named_stat(destination_fd, relative)
            if (
                not stat.S_ISDIR(source_directory.st_mode)
                or not stat.S_ISDIR(destination_directory.st_mode)
                or stat.S_IMODE(destination_directory.st_mode) != 0o700
                or (
                    destination_directory.st_dev,
                    destination_directory.st_ino,
                )
                != (
                    destination_directory_opened.st_dev,
                    destination_directory_opened.st_ino,
                )
                or source_directory.st_dev != source_stat.st_dev
                or destination_directory.st_dev != destination_stat.st_dev
            ):
                raise TrainingContractError(
                    f"checkpoint directory identity differs: {relative}"
                )

        entry_by_path = {str(entry["path"]): entry for entry in entries}

        def move_regular(relative: str, expected_sha256: str, expected_size: int) -> None:
            source_file_fd = -1
            destination_file_fd = -1
            try:
                source_file_fd = os.open(
                    relative,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=source_fd,
                )
                source_file = os.fstat(source_file_fd)
                source_named_file = named_stat(source_fd, relative)
                if (
                    not stat.S_ISREG(source_file.st_mode)
                    or source_file.st_nlink != 1
                    or source_file.st_size != expected_size
                    or (source_file.st_dev, source_file.st_ino)
                    != (source_named_file.st_dev, source_named_file.st_ino)
                    or fd_digest(source_file_fd) != expected_sha256
                ):
                    raise TrainingContractError(
                        f"checkpoint staged file identity differs: {relative}"
                    )
                os.fsync(source_file_fd)
                link_error: Optional[OSError] = None
                try:
                    os.link(
                        relative,
                        relative,
                        src_dir_fd=source_fd,
                        dst_dir_fd=destination_fd,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    link_error = error
                try:
                    destination_file_fd = os.open(
                        relative,
                        os.O_RDONLY | os.O_NOFOLLOW,
                        dir_fd=destination_fd,
                    )
                except OSError as open_error:
                    if link_error is not None:
                        raise TrainingContractError(
                            f"checkpoint create-only link failed for {relative}: {link_error}"
                        ) from link_error
                    raise TrainingContractError(
                        f"checkpoint linked destination disappeared: {relative}: {open_error}"
                    ) from open_error
                destination_file = os.fstat(destination_file_fd)
                source_after_link = os.fstat(source_file_fd)
                if (
                    not stat.S_ISREG(destination_file.st_mode)
                    or (destination_file.st_dev, destination_file.st_ino)
                    != (source_file.st_dev, source_file.st_ino)
                    or destination_file.st_size != expected_size
                    or source_after_link.st_nlink != 2
                    or fd_digest(destination_file_fd) != expected_sha256
                ):
                    raise TrainingContractError(
                        f"checkpoint create-only link replay differs: {relative}"
                    )
                destination_parent = relative.rpartition("/")[0]
                # NFS silly-renames an unlinked name while any client file
                # descriptor for that inode remains open.  The destination
                # link and its retained descriptor now bind the exact inode
                # and bytes, so close the source descriptor before consuming
                # the source name.  Otherwise a transient `.nfs*` name keeps
                # the link count at two and makes publication unreachable on
                # the production mount.
                os.close(source_file_fd)
                source_file_fd = -1
                fsync_relative_directory(destination_fd, destination_parent)
                unlink_error: Optional[OSError] = None
                try:
                    os.unlink(relative, dir_fd=source_fd)
                except OSError as error:
                    unlink_error = error
                try:
                    named_stat(source_fd, relative)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    raise TrainingContractError(
                        f"cannot replay checkpoint source unlink {relative}: {error}"
                    ) from error
                else:
                    if unlink_error is not None:
                        raise TrainingContractError(
                            f"checkpoint source unlink failed for {relative}: {unlink_error}"
                        ) from unlink_error
                    raise TrainingContractError(
                        f"checkpoint source remained after unlink: {relative}"
                    )
                destination_after_unlink = os.fstat(destination_file_fd)
                destination_named_file = named_stat(destination_fd, relative)
                if (
                    not stat.S_ISREG(destination_after_unlink.st_mode)
                    or (destination_after_unlink.st_dev, destination_after_unlink.st_ino)
                    != (source_file.st_dev, source_file.st_ino)
                    or (destination_named_file.st_dev, destination_named_file.st_ino)
                    != (source_file.st_dev, source_file.st_ino)
                    or destination_named_file.st_nlink != 1
                    or destination_after_unlink.st_size != expected_size
                    or destination_named_file.st_size != expected_size
                    or (destination_after_unlink.st_dev, destination_after_unlink.st_ino)
                    != (destination_named_file.st_dev, destination_named_file.st_ino)
                    or fd_digest(destination_file_fd) != expected_sha256
                ):
                    raise TrainingContractError(
                        f"checkpoint linked file final identity differs: {relative}"
                    )
                fsync_relative_directory(source_fd, destination_parent)
                fsync_relative_directory(destination_fd, destination_parent)
            finally:
                if destination_file_fd >= 0:
                    os.close(destination_file_fd)
                if source_file_fd >= 0:
                    os.close(source_file_fd)

        for relative in relative_files:
            entry = entry_by_path[relative]
            move_regular(relative, str(entry["sha256"]), int(entry["size"]))
        move_regular(
            "checkpoint_manifest.json",
            hashlib.sha256(manifest_raw).hexdigest(),
            len(manifest_raw),
        )

        for relative in sorted(
            expected_directories, key=lambda item: (item.count("/"), item), reverse=True
        ):
            source_directory_fd = open_relative_directory(source_fd, relative)
            try:
                if os.listdir(source_directory_fd):
                    raise TrainingContractError(
                        f"checkpoint source directory is not empty: {relative}"
                    )
                os.fsync(source_directory_fd)
            finally:
                os.close(source_directory_fd)
            try:
                os.rmdir(relative, dir_fd=source_fd)
            except OSError as error:
                raise TrainingContractError(
                    f"cannot remove consumed checkpoint source directory {relative}: {error}"
                ) from error
        if os.listdir(source_fd):
            raise TrainingContractError("checkpoint stage was not consumed exactly")
        os.fsync(source_fd)
        try:
            os.rmdir(source.name, dir_fd=parent_fd)
        except OSError as error:
            try:
                named_stat(parent_fd, source.name)
            except FileNotFoundError:
                pass
            else:
                raise TrainingContractError(
                    f"cannot remove consumed checkpoint stage: {error}"
                ) from error
        try:
            named_stat(parent_fd, source.name)
        except FileNotFoundError:
            pass
        else:
            raise TrainingContractError("checkpoint stage name remained after publication")
        os.fsync(parent_fd)

        expected_final_files = set(relative_files) | {
            "checkpoint_manifest.json",
            ".INCOMPLETE",
        }
        actual_final_files: set[str] = set()
        actual_final_directories: set[str] = set()
        for path in sorted(destination.rglob("*")):
            relative = path.relative_to(destination).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise TrainingContractError(
                    f"published checkpoint contains a symlink: {relative}"
                )
            if stat.S_ISDIR(info.st_mode):
                actual_final_directories.add(relative)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                actual_final_files.add(relative)
            else:
                raise TrainingContractError(
                    f"published checkpoint contains a hostile member: {relative}"
                )
        if (
            actual_final_directories != expected_directories
            or actual_final_files != expected_final_files
        ):
            raise TrainingContractError("published checkpoint closure differs")
        final_expectations = {
            relative: (
                str(entry_by_path[relative]["sha256"]),
                int(entry_by_path[relative]["size"]),
            )
            for relative in relative_files
        }
        final_expectations["checkpoint_manifest.json"] = (
            hashlib.sha256(manifest_raw).hexdigest(),
            len(manifest_raw),
        )
        for relative, (expected_sha256, expected_size) in sorted(
            final_expectations.items()
        ):
            descriptor = os.open(
                relative,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=destination_fd,
            )
            try:
                info = os.fstat(descriptor)
                named = named_stat(destination_fd, relative)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_size != expected_size
                    or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
                    or fd_digest(descriptor) != expected_sha256
                ):
                    raise TrainingContractError(
                        f"published checkpoint final replay differs: {relative}"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for relative in sorted(
            expected_directories, key=lambda item: item.count("/"), reverse=True
        ):
            fsync_relative_directory(destination_fd, relative)
        os.fsync(destination_fd)
        os.fsync(parent_fd)

        parent_replay = os.fstat(parent_fd)
        destination_replay = os.fstat(destination_fd)
        destination_named = named_stat(parent_fd, destination.name)
        if (
            (parent_replay.st_dev, parent_replay.st_ino)
            != (opened_parent.st_dev, opened_parent.st_ino)
            or (destination_replay.st_dev, destination_replay.st_ino)
            != (destination_named.st_dev, destination_named.st_ino)
            or stat.S_IMODE(destination_replay.st_mode) != 0o700
        ):
            raise TrainingContractError(
                "checkpoint destination identity changed before commit"
            )
        marker_named = named_stat(destination_fd, ".INCOMPLETE")
        marker_opened = os.fstat(marker_fd)
        if (
            (marker_named.st_dev, marker_named.st_ino)
            != (marker_opened.st_dev, marker_opened.st_ino)
            or marker_opened.st_nlink != 1
            or stat.S_IMODE(marker_opened.st_mode) != 0o400
            or marker_opened.st_size != len(marker_raw)
            or fd_digest(marker_fd) != hashlib.sha256(marker_raw).hexdigest()
        ):
            raise TrainingContractError("checkpoint incomplete marker changed before commit")
        # NFS silly-renames an unlinked file that is still open.  Close the
        # retained marker descriptor before the namespace commit so a
        # successful unlink cannot leave a transient `.nfs*` member that
        # violates the exact checkpoint closure.
        os.close(marker_fd)
        marker_fd = -1
        marker_unlink_error: Optional[OSError] = None
        try:
            os.unlink(".INCOMPLETE", dir_fd=destination_fd)
        except OSError as error:
            marker_unlink_error = error
        try:
            named_stat(destination_fd, ".INCOMPLETE")
        except FileNotFoundError:
            pass
        except OSError as error:
            raise TrainingContractError(
                f"cannot replay checkpoint marker commit: {error}"
            ) from error
        else:
            if marker_unlink_error is not None:
                raise TrainingContractError(
                    f"checkpoint marker commit failed: {marker_unlink_error}"
                ) from marker_unlink_error
            raise TrainingContractError("checkpoint incomplete marker remained after commit")
        committed_files: set[str] = set()
        committed_directories: set[str] = set()
        for path in sorted(destination.rglob("*")):
            relative = path.relative_to(destination).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise TrainingContractError(
                    f"committed checkpoint contains a symlink: {relative}"
                )
            if stat.S_ISDIR(info.st_mode):
                committed_directories.add(relative)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                committed_files.add(relative)
            else:
                raise TrainingContractError(
                    f"committed checkpoint contains a hostile member: {relative}"
                )
        if committed_directories != expected_directories or committed_files != set(
            relative_files
        ) | {"checkpoint_manifest.json"}:
            raise TrainingContractError(
                "committed checkpoint namespace closure differs"
            )
        os.fsync(destination_fd)
        os.fsync(parent_fd)
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_fd >= 0:
            os.close(source_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def save_training_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    output: Path,
    global_step: int,
    receipt: Mapping[str, Any],
    dataset_signature: str,
    rank: int,
) -> Path:
    import torch
    import torch.distributed as dist

    final = output / f"checkpoint-{global_step:08d}"
    if rank == 0:
        if final.exists() or final.is_symlink():
            raise TrainingContractError(f"refusing to overwrite checkpoint: {final}")
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / f".{final.name}.tmp-{os.getpid()}"
        if temporary.exists() or temporary.is_symlink():
            raise TrainingContractError(f"stale temporary checkpoint exists: {temporary}")
        temporary.mkdir(parents=False)
        model.save_pretrained(temporary / "adapter", safe_serialization=True)
        torch.save(
            {
                "schema_version": RECEIPT_SCHEMA,
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "dataset_signature": dataset_signature,
            },
            temporary / "optimizer.pt",
        )
        _atomic_write_json(temporary / "receipt.json", receipt)
        manifest = build_checkpoint_content_manifest(
            temporary,
            global_step=global_step,
            receipt_digest=str(receipt.get("receipt_digest", "")),
        )
        _atomic_write_json(temporary / "checkpoint_manifest.json", manifest)
        _atomic_rename_directory_noreplace(temporary, final)
        _atomic_write_json(
            output / "latest.json",
            {
                "checkpoint": str(final),
                "global_step": global_step,
                "checkpoint_manifest_path": str(final / "checkpoint_manifest.json"),
                "checkpoint_manifest_sha256": file_sha256(
                    final / "checkpoint_manifest.json"
                ),
                "checkpoint_receipt_sha256": file_sha256(final / "receipt.json"),
            },
        )
        output_fd = os.open(
            output,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an exact-81f mask-free LoRA for Bernini-R 1.3B"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    parser.add_argument(
        "--allow-reward-selected-synthetic-targets",
        action="store_true",
        help="Explicitly opt into the four-row reward-selected exploratory dataset.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, choices=(NUM_FRAMES,), default=NUM_FRAMES)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--objective", choices=TRAINING_OBJECTIVES, default="sft")
    parser.add_argument(
        "--contrastive-negative-schedule",
        choices=CONTRASTIVE_NEGATIVE_SCHEDULES,
        default="rotate",
    )
    parser.add_argument("--preference-weight", type=float, default=1.0)
    parser.add_argument("--preference-margin", type=float, default=0.05)
    parser.add_argument("--preference-temperature", type=float, default=20.0)
    parser.add_argument("--dpo-beta", type=float, default=10.0)
    parser.add_argument("--preservation-weight", type=float, default=0.25)
    parser.add_argument("--lora-rank", type=int, choices=(8, 64, 256), default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, choices=(8, 64, 256), default=LORA_ALPHA)
    parser.add_argument(
        "--exploratory-full644-one-pass",
        action="store_true",
        help=(
            "Consume the sealed historical full644 rows exactly once as exposed "
            "train/debug data. This is not formal held-out or scientific promotion."
        ),
    )
    parser.add_argument(
        "--full644-source-authority-receipt",
        default=None,
        help=(
            "Absolute exact receipt proving the historical full644 source/group "
            "uniqueness and raw/VAE identities; required only by the full644 profile."
        ),
    )
    parser.add_argument(
        "--expected-full644-source-authority-sha256",
        default=None,
        help="Expected SHA-256 of --full644-source-authority-receipt.",
    )
    parser.add_argument(
        "--expected-bernini-commit", default=BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument("--expected-veomni-commit", default=VEOMNI_TESTED_COMMIT)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.num_frames != NUM_FRAMES:
        raise TrainingContractError(f"only exact {NUM_FRAMES}-frame training is supported")
    if args.max_steps <= 0:
        raise TrainingContractError("max_steps must be positive")
    if args.save_every < 0:
        raise TrainingContractError("save_every must be non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise TrainingContractError("learning_rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise TrainingContractError("weight_decay must be finite and non-negative")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise TrainingContractError("max_grad_norm must be finite and positive")
    objective = getattr(args, "objective", "sft")
    preference_defaults = {
        "preference_weight": 1.0,
        "preference_temperature": 20.0,
        "dpo_beta": 10.0,
    }
    for name, default in preference_defaults.items():
        value = getattr(args, name, default)
        if not math.isfinite(value) or value <= 0:
            raise TrainingContractError(f"{name} must be finite and positive")
    preference_margin = getattr(args, "preference_margin", 0.05)
    preservation_weight = getattr(args, "preservation_weight", 0.25)
    if not math.isfinite(preference_margin) or preference_margin < 0:
        raise TrainingContractError("preference_margin must be finite and non-negative")
    if not math.isfinite(preservation_weight) or preservation_weight < 0:
        raise TrainingContractError("preservation_weight must be finite and non-negative")
    if objective in (
        "sft_preservation",
        "detached_margin_preservation",
        "reference_dpo_preservation",
    ) and preservation_weight <= 0:
        raise TrainingContractError("preservation objective requires positive preservation_weight")
    full644_profile = getattr(args, "exploratory_full644_one_pass", False)
    if type(full644_profile) is not bool:
        raise TrainingContractError("exploratory full644 profile flag must be bool")
    lora_rank = getattr(args, "lora_rank", LORA_RANK)
    lora_alpha = getattr(args, "lora_alpha", LORA_ALPHA)
    if type(lora_rank) is not int or lora_rank not in {8, 64, 256}:
        raise TrainingContractError("LoRA rank differs")
    if type(lora_alpha) is not int or lora_alpha not in {8, 64, 256}:
        raise TrainingContractError("LoRA alpha differs")
    if objective != "sft" and not (
        getattr(args, "allow_reward_selected_synthetic_targets", False)
        or full644_profile is True
    ):
        raise TrainingContractError(
            "preference objectives require --allow-reward-selected-synthetic-targets"
        )
    if full644_profile is True:
        exact = {
            "max_steps": FULL644_EXPLORATORY_STEPS,
            "save_every": 64,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "seed": FULL644_EXPLORATORY_SEED,
            "objective": "reference_dpo_preservation",
            "contrastive_negative_schedule": "rotate",
            "preference_weight": 1.0,
            "preference_margin": 0.05,
            "preference_temperature": 20.0,
            "dpo_beta": 10.0,
            "preservation_weight": 0.25,
            "lora_rank": FULL644_EXPLORATORY_RANK,
            "lora_alpha": FULL644_EXPLORATORY_ALPHA,
        }
        for name, expected in exact.items():
            if getattr(args, name, None) != expected:
                raise TrainingContractError(
                    f"full644 exploratory profile requires {name}={expected!r}"
                )
        if (
            getattr(args, "allow_incomplete_dataset", False) is not False
            or getattr(args, "allow_reward_selected_synthetic_targets", False)
            is not False
            or getattr(args, "resume", None) is not None
        ):
            raise TrainingContractError(
                "full644 exploratory profile requires complete natural data and fresh state"
            )
        authority_path = getattr(args, "full644_source_authority_receipt", None)
        authority_sha256 = getattr(
            args, "expected_full644_source_authority_sha256", None
        )
        if (
            type(authority_path) is not str
            or not authority_path
            or not Path(authority_path).expanduser().is_absolute()
            or authority_sha256 != FULL644_SOURCE_AUTHORITY_SHA256
        ):
            raise TrainingContractError(
                "full644 exploratory profile requires the exact source authority"
            )
    elif (
        getattr(args, "full644_source_authority_receipt", None) is not None
        or getattr(args, "expected_full644_source_authority_sha256", None) is not None
    ):
        raise TrainingContractError(
            "full644 source authority arguments require the exploratory profile"
        )
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if not re.fullmatch(r"[0-9a-fA-F]{40}", getattr(args, name)):
            raise TrainingContractError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", getattr(args, name)):
            raise TrainingContractError(f"{name} must be a lowercase SHA-256")
    if args.expected_checkpoint_tree_sha256 != CHECKPOINT_TREE_SHA256:
        raise TrainingContractError("checkpoint tree identity differs from the audited 1.3B tree")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    full644_profile = getattr(args, "exploratory_full644_one_pass", False) is True
    contract = distributed_contract()
    if full644_profile and (
        contract.world_size != 4 or contract.ulysses_size != 4
    ):
        raise TrainingContractError(
            "full644 exploratory training requires exact torchrun world_size=4/Ulysses=4"
        )
    output_requested = Path(args.output).expanduser()
    if full644_profile:
        try:
            output_parent = output_requested.parent.resolve(strict=True)
        except OSError as error:
            raise TrainingContractError(
                f"full644 exploratory output parent is unavailable: {error}"
            ) from error
        if (
            not output_requested.is_absolute()
            or output_requested != output_parent / output_requested.name
            or not output_parent.is_dir()
            or output_parent.is_symlink()
            or output_requested.exists()
            or output_requested.is_symlink()
        ):
            raise TrainingContractError(
                "full644 exploratory output must be one fresh canonical path"
            )
    output_dir = output_requested.resolve()
    bernini_root, veomni_root, bernini_revision, veomni_revision = validate_source_trees(
        args.bernini_root,
        args.veomni_root,
        expected_bernini_commit=args.expected_bernini_commit,
        expected_veomni_commit=args.expected_veomni_commit,
    )
    checkpoint, transformer_config = validate_checkpoint(args.checkpoint)
    if transformer_config["num_attention_heads"] % 4:
        raise TrainingContractError("1.3B attention heads must be divisible by Ulysses=4")
    dataset = ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = validate_preprocessed_dataset_summary(
        args.dataset_summary,
        dataset,
        allow_incomplete=args.allow_incomplete_dataset,
        allow_reward_selected_synthetic_targets=(
            args.allow_reward_selected_synthetic_targets
        ),
    )
    full644_source_authority: Optional[dict[str, Any]] = None
    if full644_profile:
        full644_source_authority = validate_full644_source_authority(
            args.full644_source_authority_receipt,
            expected_sha256=args.expected_full644_source_authority_sha256,
        )
        if (
            len(dataset) != FULL644_EXPLORATORY_STEPS
            or dataset_summary.get("sha256") != FULL644_DATASET_SUMMARY_SHA256
            or dataset_summary.get("summary_digest")
            != FULL644_DATASET_SUMMARY_DIGEST
            or dataset_summary.get("index_sha256") != FULL644_DATASET_INDEX_SHA256
            or dataset_summary.get("complete") is not True
            or dataset_summary.get("materialized_rows")
            != FULL644_EXPLORATORY_STEPS
            or dataset_summary.get("reward_selected_synthetic_targets") is not False
            or dataset_summary.get("dataset_content_signature")
            != dataset.content_signature
            or dataset.content_signature is None
        ):
            raise TrainingContractError(
                "full644 exploratory dataset authority differs from sealed exact644"
            )
    if args.objective != "sft" and not full644_profile and (
        not dataset_summary.get("reward_selected_synthetic_targets")
        or dataset_summary.get("arm") != "action_only"
    ):
        raise TrainingContractError(
            "preference objectives require the audited action_only synthetic-target arm"
        )
    activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from peft import (
        LoraConfig,
        PeftModel,
        __version__ as peft_version,
        get_peft_model,
    )
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.training.data import NoiseScheduler, process_renderer_sample

    device, backend = initialise_distributed(contract)
    from bernini.parallel import init_parallel_state as init_bernini_parallel_state

    init_bernini_parallel_state(ulysses_size=contract.ulysses_size)
    if full644_profile:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if contract.rank == 0:
            try:
                output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
            except OSError as error:
                raise TrainingContractError(
                    f"cannot create fresh full644 output root: {error}"
                ) from error
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise TrainingContractError("fresh full644 output root identity differs")
    # Deterministic construction is useful for single-rank debug.  Four-rank
    # correctness is enforced independently by the explicit broadcast below.
    seed_same_sample(args.seed)

    config_dir = bernini_root / "configs/bernini_renderer_wan21_1p3b"
    config = BerniniRendererConfig.from_pretrained(
        str(config_dir), local_files_only=True, **renderer_config_overrides(checkpoint)
    )
    # Loading bf16 pretrained modules directly avoids an unnecessary fp32 peak.
    config.dtype = torch.bfloat16
    validate_renderer_config_mapping(config.to_dict(), checkpoint)
    resumed_from: Optional[str] = None
    resume_state_path: Optional[Path] = None
    if args.resume:
        adapter_path, resume_state_path, resume_receipt_path = _adapter_paths(Path(args.resume))
        adapter_config = _read_json(adapter_path / "adapter_config.json")
        if int(adapter_config.get("r", -1)) != args.lora_rank:
            raise TrainingContractError("resume adapter rank differs")
        prior_receipt = _read_json(resume_receipt_path)
        if prior_receipt.get("schema_version") != RECEIPT_SCHEMA:
            raise TrainingContractError("resume receipt schema mismatch")
        resumed_from = str(Path(args.resume).expanduser().resolve())

    def build_rank_local_model() -> tuple[Any, list[str], list[tuple[str, Any]], int]:
        base_model = BerniniRendererModel(config)
        base_model.requires_grad_(False)
        base_model.t5_text_encoder.eval()
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        target_modules = select_attention_projection_names(base_model)
        if len(target_modules) != EXPECTED_LORA_TARGET_MODULES:
            raise TrainingContractError(
                "Bernini-R 1.3B LoRA target count differs: "
                f"expected {EXPECTED_LORA_TARGET_MODULES}, got {len(target_modules)}"
            )
        if args.resume:
            model = PeftModel.from_pretrained(
                base_model, adapter_path, is_trainable=True
            )
        else:
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=target_modules,
            )
            # This must run before get_peft_model can inject a changed adapter
            # and before AdamW is constructed or any optimizer update is possible.
            validate_full644_peft_construction(
                lora_config.to_dict(),
                peft_version=peft_version,
                expected_target_modules=target_modules,
                profile_enabled=full644_profile,
            )
            model = get_peft_model(base_model, lora_config)
        model.to(device)
        named_trainable = trainable_lora_parameters(model)
        trainable_count = sum(int(param.numel()) for _, param in named_trainable)
        validate_full644_trainable_parameter_count(
            trainable_count,
            profile_enabled=bool(args.exploratory_full644_one_pass),
        )
        return model, target_modules, named_trainable, trainable_count

    construction, construction_statuses = world4_rank_serialized_model_construction(
        contract=contract,
        device=device,
        build_function=build_rank_local_model,
        torch_module=torch,
        dist_module=dist,
    )
    model, target_modules, named_trainable, trainable_count = construction
    if contract.rank == 0 and construction_statuses:
        print(
            json.dumps(
                {"serialized_model_construction": construction_statuses},
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    model.train()
    model.get_base_model().t5_text_encoder.eval()
    lora_initialization_digest = synchronize_trainable_parameters(
        named_trainable, source_rank=0
    )
    optimizer = torch.optim.AdamW(
        [param for _, param in named_trainable],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    global_step = 0
    if resume_state_path is not None:
        try:
            state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(resume_state_path, map_location="cpu")
        if state.get("schema_version") != RECEIPT_SCHEMA:
            raise TrainingContractError("resume optimizer schema mismatch")
        if state.get("dataset_signature") != dataset.signature:
            raise TrainingContractError("resume dataset signature mismatch")
        optimizer.load_state_dict(state["optimizer"])
        _optimizer_to(optimizer, device)
        global_step = int(state["global_step"])
        if global_step < 0 or global_step > args.max_steps:
            raise TrainingContractError("resume global_step is outside requested run")

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(
        128, (1, 2, 2), 1024, use_src_id_rotary_emb=True
    )
    vae_mean, vae_std, z_dim = _vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**noise_scheduler_kwargs())

    last_loss: Optional[float] = None
    last_grad_norm: Optional[float] = None
    last_saved = global_step if global_step and args.resume else -1

    def transform_sample(sample_to_transform: Mapping[str, Any], seed: int) -> dict[str, Any]:
        seed_same_sample(seed)
        transformed_sample = process_renderer_sample(
            sample_to_transform,
            tokenizer=tokenizer,
            vae_rope_func=rope,
            vae_latent_mean=vae_mean,
            vae_latent_std=vae_std,
            noise_scheduler=scheduler,
            text_dropout_rate=0.0,
            img_dropout_rate=0.0,
            video_dropout_rate=0.0,
            max_vae_frames=LATENT_FRAMES,
            source_name=TASK_SOURCE_NAME,
        )
        collated = collate_single_renderer_sample(transformed_sample)
        validate_collated_supervision(collated)
        return _move_batch(collated, device)

    while global_step < args.max_steps:
        row_index = (
            full644_one_pass_row_index(global_step, len(dataset))
            if full644_profile
            else global_step % len(dataset)
        )
        raw_row = dataset[row_index]
        identity = dataset_identity(raw_row, row_index)
        assert_identical_row(identity)
        sample = sanitize_preprocessed_row(raw_row)
        validate_81_frame_latents(sample, expected_parameter_channels=2 * z_dim)
        current_seed = step_seed(args.seed, global_step, row_index)
        chosen_batch = transform_sample(sample, current_seed)

        optimizer.zero_grad(set_to_none=True)
        negative_kind: Optional[str] = None
        chosen_loss: Any
        rejected_loss: Optional[Any] = None
        reference_chosen_loss: Optional[Any] = None
        reference_rejected_loss: Optional[Any] = None
        preference_loss: Optional[Any] = None
        preservation_loss: Optional[Any] = None
        if args.objective in ("sft", "sft_preservation"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chosen_loss = model(**chosen_batch, use_cache=False).diff_loss.float().mean()
            loss = chosen_loss
            loss.backward()
            if args.objective == "sft_preservation":
                identity_sample = build_identity_preservation_sample(sample)
                identity_batch = transform_sample(identity_sample, current_seed)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    seed_same_sample(current_seed)
                    preservation_loss = model(
                        **identity_batch, use_cache=False
                    ).diff_loss.float().mean()
                    weighted_preservation = args.preservation_weight * preservation_loss
                weighted_preservation.backward()
                loss = loss.detach() + weighted_preservation.detach()
        else:
            negative_kind = contrastive_negative_kind(
                global_step, schedule=args.contrastive_negative_schedule
            )
            rejected_sample = build_contrastive_sample(
                sample, negative_kind=negative_kind
            )
            rejected_batch = transform_sample(rejected_sample, current_seed)
            if args.objective in ("reference_dpo", "reference_dpo_preservation"):
                with torch.no_grad(), model.disable_adapter(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    seed_same_sample(current_seed)
                    reference_chosen_loss = (
                        model(**chosen_batch, use_cache=False).diff_loss.float().mean()
                    )
                    seed_same_sample(current_seed)
                    reference_rejected_loss = (
                        model(**rejected_batch, use_cache=False).diff_loss.float().mean()
                    )
            detached_objective = args.objective in (
                "detached_margin",
                "detached_margin_preservation",
            )
            if detached_objective:
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    seed_same_sample(current_seed)
                    rejected_loss = model(
                        **rejected_batch, use_cache=False
                    ).diff_loss.float().mean()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                seed_same_sample(current_seed)
                chosen_loss = model(
                    **chosen_batch, use_cache=False
                ).diff_loss.float().mean()
                if not detached_objective:
                    seed_same_sample(current_seed)
                    rejected_loss = model(
                        **rejected_batch, use_cache=False
                    ).diff_loss.float().mean()
                if args.objective == "high_contrast_margin":
                    preference_loss = high_contrast_preference_loss(
                        chosen_loss,
                        rejected_loss,
                        margin=args.preference_margin,
                        temperature=args.preference_temperature,
                    )
                elif detached_objective:
                    preference_loss = detached_rejected_preference_loss(
                        chosen_loss,
                        rejected_loss,
                        margin=args.preference_margin,
                        temperature=args.preference_temperature,
                    )
                else:
                    assert reference_chosen_loss is not None
                    assert reference_rejected_loss is not None
                    preference_loss = reference_dpo_loss(
                        chosen_loss,
                        rejected_loss,
                        reference_chosen_loss,
                        reference_rejected_loss,
                        beta=args.dpo_beta,
                    )
                objective_loss = chosen_loss + args.preference_weight * preference_loss
            objective_loss.backward()
            loss = objective_loss.detach()
            if args.objective in (
                "detached_margin_preservation",
                "reference_dpo_preservation",
            ):
                identity_sample = build_identity_preservation_sample(sample)
                identity_batch = transform_sample(identity_sample, current_seed)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    seed_same_sample(current_seed)
                    preservation_loss = model(
                        **identity_batch, use_cache=False
                    ).diff_loss.float().mean()
                    weighted_preservation = args.preservation_weight * preservation_loss
                weighted_preservation.backward()
                loss = loss + weighted_preservation.detach()
        finite_loss = bool(torch.isfinite(loss.detach()).item())
        if not _distributed_boolean(finite_loss, op="all"):
            raise TrainingContractError(
                f"non-finite loss at step {global_step + 1}; optimizer step blocked"
            )
        last_grad_norm = all_reduce_lora_gradients(named_trainable)
        torch.nn.utils.clip_grad_norm_(
            [param for _, param in named_trainable], args.max_grad_norm
        )
        optimizer.step()
        global_step += 1
        last_loss = float(loss.detach().item())
        if contract.rank == 0:
            print(
                json.dumps(
                    {
                        "step": global_step,
                        "loss": last_loss,
                        "objective": args.objective,
                        "chosen_loss": float(chosen_loss.detach().item()),
                        "rejected_loss": (
                            float(rejected_loss.detach().item())
                            if rejected_loss is not None
                            else None
                        ),
                        "preference_loss": (
                            float(preference_loss.detach().item())
                            if preference_loss is not None
                            else None
                        ),
                        "student_gap": (
                            float((rejected_loss - chosen_loss).detach().item())
                            if rejected_loss is not None
                            else None
                        ),
                        "reference_gap": (
                            float(
                                (
                                    reference_rejected_loss - reference_chosen_loss
                                ).detach().item()
                            )
                            if reference_rejected_loss is not None
                            and reference_chosen_loss is not None
                            else None
                        ),
                        "preservation_loss": (
                            float(preservation_loss.detach().item())
                            if preservation_loss is not None
                            else None
                        ),
                        "negative_kind": negative_kind,
                        "preclip_grad_norm": last_grad_norm,
                        "row": row_index,
                        "seed": current_seed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        should_save = args.save_every > 0 and global_step % args.save_every == 0
        if should_save:
            receipt = build_receipt(
                args=args,
                global_step=global_step,
                last_loss=last_loss,
                gradient_norm=last_grad_norm,
                dataset=dataset,
                dataset_summary=dataset_summary,
                checkpoint=checkpoint,
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                distributed=contract,
                backend=backend,
                target_modules=target_modules,
                trainable_parameter_count=trainable_count,
                lora_initialization_digest=lora_initialization_digest,
                peft_version=peft_version,
                transformers_version=transformers_version,
                resumed_from=resumed_from,
                full644_source_authority=full644_source_authority,
                terminal_dataset_reverified=False,
            )
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                output=output_dir,
                global_step=global_step,
                receipt=receipt,
                dataset_signature=(
                    dataset.content_signature if full644_profile else dataset.signature
                ),
                rank=contract.rank,
            )
            last_saved = global_step

    terminal_dataset_reverified = False
    if full644_profile:
        terminal_summary = validate_preprocessed_dataset_summary(
            args.dataset_summary,
            dataset,
            allow_incomplete=False,
            allow_reward_selected_synthetic_targets=False,
        )
        terminal_source_authority = validate_full644_source_authority(
            args.full644_source_authority_receipt,
            expected_sha256=args.expected_full644_source_authority_sha256,
        )
        if (
            terminal_summary != dataset_summary
            or terminal_source_authority != full644_source_authority
            or dataset.revalidate_bound_files() != dataset.content_signature
        ):
            raise TrainingContractError(
                "full644 data authority changed before terminal checkpoint"
            )
        terminal_dataset_reverified = True

    if last_saved != global_step:
        receipt = build_receipt(
            args=args,
            global_step=global_step,
            last_loss=last_loss,
            gradient_norm=last_grad_norm,
            dataset=dataset,
            dataset_summary=dataset_summary,
            checkpoint=checkpoint,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            distributed=contract,
            backend=backend,
            target_modules=target_modules,
            trainable_parameter_count=trainable_count,
            lora_initialization_digest=lora_initialization_digest,
            peft_version=peft_version,
            transformers_version=transformers_version,
            resumed_from=resumed_from,
            full644_source_authority=full644_source_authority,
            terminal_dataset_reverified=terminal_dataset_reverified,
        )
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            output=output_dir,
            global_step=global_step,
            receipt=receipt,
            dataset_signature=(
                dataset.content_signature if full644_profile else dataset.signature
            ),
            rank=contract.rank,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

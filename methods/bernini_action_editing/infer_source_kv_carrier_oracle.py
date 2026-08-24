#!/usr/bin/env python3
"""Frozen-base Bernini source-K/V carrier oracle and replay-off control.

The ``on`` arm adds exactly one source-only semantic-noop forward before the
two official ``v2v_apg`` forwards at every solver step.  All selected Bernini
``attn1`` processors capture post-RoPE source K/V in that forward; the
official negative and action pair queries then replay the same carrier.  The
one-step bank is cleared only after both queries have completed.

The ``off`` arm is the same frozen base, source, prompt, seed, scheduler and
four-rank Ulysses execution without installing the carrier processor or hook.
Both receipts contain the same causal-pairing digest.  This file deliberately
does not accept, construct, load, or merge a PEFT/LoRA adapter.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import timedelta
import hashlib
import inspect
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as legacy  # noqa: E402
import source_kv_replay as replay_core  # noqa: E402
import source_kv_route_batches as route_batches  # noqa: E402


RECEIPT_SCHEMA = "bernini-r-1p3b-frozen-source-kv-carrier-oracle-v1"
EXPECTED_STEPS = 40
EXPECTED_FRAMES = 81
EXPECTED_ULYSSES_SIZE = 4
EXPECTED_DOG_SOURCE_TOKENS = 19_530
EXPECTED_DOG_PAIR_TOKENS = 39_060
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
CHECKPOINT_CONTENT_FILE_COUNT = 23
REPLAY_MODES = ("off", "on")
FLOW_SHIFT = 5.0
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SourceKVCarrierOracleError(RuntimeError):
    """Raised before emitting an unsupported oracle result."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Bernini-R source-K/V carrier inference or its "
            "same-seed replay-off control"
        )
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--original-source-path", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay", required=True, choices=REPLAY_MODES)
    parser.add_argument(
        "--block-selection",
        default=replay_core.MAIN_BLOCK_SELECTION,
        choices=replay_core.BLOCK_SELECTIONS,
    )
    parser.add_argument(
        "--expected-source-tokens", type=int, required=True
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=EXPECTED_STEPS
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--expected-bernini-commit",
        default=legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if (
        not isinstance(args.instruction, str)
        or not args.instruction.strip()
        or "\x00" in args.instruction
    ):
        raise SourceKVCarrierOracleError(
            "instruction must be non-empty text without NUL"
        )
    if args.replay not in REPLAY_MODES:
        raise SourceKVCarrierOracleError(f"replay must be one of {REPLAY_MODES}")
    if args.block_selection not in replay_core.BLOCK_SELECTIONS:
        raise SourceKVCarrierOracleError(
            "block_selection must be all, mid, or late"
        )
    if args.num_inference_steps != EXPECTED_STEPS:
        raise SourceKVCarrierOracleError(
            f"oracle is fixed to {EXPECTED_STEPS} inference steps"
        )
    if (
        type(args.expected_source_tokens) is not int
        or args.expected_source_tokens <= 0
    ):
        raise SourceKVCarrierOracleError(
            "expected_source_tokens must be a positive integer"
        )
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise SourceKVCarrierOracleError("seed must be in [0,2^63)")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA1_RE.fullmatch(value.lower()) is None:
            raise SourceKVCarrierOracleError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
        "expected_source_sha256",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise SourceKVCarrierOracleError(
                f"{name} must be a lowercase SHA-256"
            )
    if args.expected_bernini_commit.lower() != legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise SourceKVCarrierOracleError(
            "only the audited Bernini source commit is supported"
        )
    if args.expected_veomni_commit.lower() != legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise SourceKVCarrierOracleError(
            "only the tested VeOmni source commit is supported"
        )
    if args.expected_checkpoint_tree_sha256 != legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise SourceKVCarrierOracleError(
            "only the audited Bernini-R 1.3B checkpoint is supported"
        )


def _bind_call(function: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        bound = inspect.signature(function).bind(*args, **dict(kwargs))
        bound.apply_defaults()
    except (TypeError, ValueError) as error:
        raise SourceKVCarrierOracleError(
            "pinned Bernini call signature differs"
        ) from error
    return dict(bound.arguments)


def _replace_argument(
    function: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    try:
        bound = inspect.signature(function).bind(*args, **dict(kwargs))
        bound.apply_defaults()
    except (TypeError, ValueError) as error:
        raise SourceKVCarrierOracleError(
            "cannot replace a pinned Bernini shared_step argument"
        ) from error
    if name not in bound.arguments:
        raise SourceKVCarrierOracleError(
            f"pinned Bernini shared_step lacks {name}"
        )
    bound.arguments[name] = value
    return tuple(bound.args), dict(bound.kwargs)


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise SourceKVCarrierOracleError(f"{label} has no tensor shape")
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise SourceKVCarrierOracleError(
            f"{label} has a non-integral tensor shape"
        ) from error


def _metadata_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SourceKVCarrierOracleError(f"{label} must be a sequence")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SourceKVCarrierOracleError(f"{label} must contain integers") from error
    if any(item <= 0 for item in result):
        raise SourceKVCarrierOracleError(f"{label} must contain positive lengths")
    return result


def _canonical_timestep_token(value: Any, *, step_index: int) -> str:
    scalar = value
    if hasattr(scalar, "detach"):
        scalar = scalar.detach()
    if hasattr(scalar, "numel"):
        if int(scalar.numel()) != 1:
            raise SourceKVCarrierOracleError("timestep must contain one scalar")
        scalar = scalar.item()
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError) as error:
        raise SourceKVCarrierOracleError("timestep must be numeric") from error
    if not math.isfinite(number):
        raise SourceKVCarrierOracleError("timestep must be finite")
    return f"step-{step_index}:float64-{number.hex()}"


def validate_checkpoint_content(
    checkpoint: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str = CHECKPOINT_CONTENT_MANIFEST_SHA256,
    expected_file_count: int = CHECKPOINT_CONTENT_FILE_COUNT,
) -> dict[str, Any]:
    """Hash every non-cache checkpoint file against one pinned manifest."""

    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise SourceKVCarrierOracleError(
            "checkpoint content root must be a non-symlink directory"
        )
    manifest = legacy._plain_file(
        manifest_path.resolve(strict=True), label="checkpoint content manifest"
    )
    computed_manifest_sha256 = legacy.file_sha256(manifest)
    if computed_manifest_sha256 != expected_manifest_sha256:
        raise SourceKVCarrierOracleError(
            "checkpoint content manifest SHA-256 differs"
        )
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise SourceKVCarrierOracleError(
            "cannot read checkpoint content manifest"
        ) from error
    if len(lines) != expected_file_count:
        raise SourceKVCarrierOracleError(
            "checkpoint content manifest file count differs"
        )
    expected: dict[str, str] = {}
    pattern = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            raise SourceKVCarrierOracleError(
                "checkpoint manifest line is not canonical sha256sum syntax"
            )
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise SourceKVCarrierOracleError(
                "checkpoint manifest contains an unsafe path"
            )
        normalized = PurePosixPath(*(
            part for part in relative.parts if part not in ("", ".")
        )).as_posix()
        if not normalized or normalized in expected:
            raise SourceKVCarrierOracleError(
                "checkpoint manifest contains an empty/duplicate path"
            )
        expected[normalized] = digest

    actual_paths: set[str] = set()
    for path in checkpoint.rglob("*"):
        relative = path.relative_to(checkpoint)
        if ".cache" in relative.parts:
            continue
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise SourceKVCarrierOracleError(
                "cannot stat checkpoint content"
            ) from error
        if stat.S_ISLNK(mode):
            raise SourceKVCarrierOracleError(
                "checkpoint contains a non-cache symlink"
            )
        if stat.S_ISREG(mode):
            actual_paths.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise SourceKVCarrierOracleError(
                "checkpoint contains a non-regular filesystem entry"
            )
    if actual_paths != set(expected):
        raise SourceKVCarrierOracleError(
            "checkpoint non-cache file set differs from pinned manifest"
        )
    verified_entries = []
    for relative in sorted(expected):
        path = legacy._plain_file(
            checkpoint / relative, label=f"checkpoint file {relative}"
        )
        actual = legacy.file_sha256(path)
        if actual != expected[relative]:
            raise SourceKVCarrierOracleError(
                f"checkpoint content hash differs: {relative}"
            )
        verified_entries.append({"path": relative, "sha256": actual})
    return {
        "manifest_path": str(manifest),
        "manifest_sha256_computed": computed_manifest_sha256,
        "manifest_sha256_expected": expected_manifest_sha256,
        "verified_file_count": len(verified_entries),
        "every_file_sha256_verified": True,
        "verified_entries_digest": legacy.object_sha256(verified_entries),
    }


def prepare_hashed_source_snapshot(
    source_path: Path,
) -> tuple[Any, dict[str, Any], str]:
    """Decode a private byte snapshot and prove it equals the named input."""

    before = source_path.stat()
    source_sha256 = legacy.file_sha256(source_path)
    with tempfile.TemporaryDirectory(prefix="bernini-source-snapshot-") as root:
        snapshot = Path(root) / "source.mp4"
        shutil.copyfile(source_path, snapshot)
        snapshot_sha256 = legacy.file_sha256(snapshot)
        if snapshot_sha256 != source_sha256:
            raise SourceKVCarrierOracleError(
                "private source snapshot digest differs during copy"
            )
        source_tensor, metadata = legacy.prepare_exact_source(snapshot)
    after = source_path.stat()
    after_sha256 = legacy.file_sha256(source_path)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or after_sha256 != source_sha256:
        raise SourceKVCarrierOracleError(
            "source video changed while its private decode snapshot was made"
        )
    value = dict(metadata)
    value.update(
        {
            "decoded_from_private_byte_snapshot": True,
            "snapshot_sha256": snapshot_sha256,
            "original_pre_snapshot_sha256": source_sha256,
            "original_post_snapshot_sha256": after_sha256,
            "original_stat_identity_stable": True,
        }
    )
    return source_tensor, value, source_sha256


def _no_grad_context() -> Any:
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.no_grad()


def resolve_diffusion_core(renderer_or_diffusion: Any) -> Any:
    """Resolve pinned ``GEN_Wanx22`` without assuming one wrapper layout."""

    queue = [renderer_or_diffusion]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if all(
            callable(getattr(candidate, name, None))
            for name in ("sample", "shared_step")
        ) and getattr(candidate, "transformer", None) is not None:
            return candidate
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        for name in ("diff_dec", "base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
    raise SourceKVCarrierOracleError(
        "could not resolve the pinned Bernini diffusion core"
    )


@dataclass(frozen=True)
class CarrierStepRecord:
    generation: int
    step_index: int
    timestep_token: str
    rank: int
    ulysses_size: int
    model_id: str
    source_tokens_runtime: int
    pair_tokens_runtime: int
    carrier_forwards: int = 1
    negative_replay_forwards: int = 1
    action_replay_forwards: int = 1
    cleared_after_both_replays: bool = True

    @property
    def identity(self) -> tuple[int, int, str, int, int]:
        return (
            self.generation,
            self.step_index,
            self.timestep_token,
            self.rank,
            self.ulysses_size,
        )


@dataclass
class CarrierTrace:
    records: list[CarrierStepRecord] = field(default_factory=list)
    sample_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        identities = [record.identity for record in self.records]
        return {
            "sample_calls": self.sample_calls,
            "step_count": len(self.records),
            "unique_identity_count": len(set(identities)),
            "steps": [asdict(record) for record in self.records],
        }


@dataclass(frozen=True)
class ReplayOffStepRecord:
    generation: int
    step_index: int
    timestep_token: str
    rank: int
    ulysses_size: int
    model_id: str
    source_tokens_runtime: int
    pair_tokens_runtime: int
    official_negative_forwards: int = 1
    official_action_forwards: int = 1

    @property
    def identity(self) -> tuple[int, int, str, int, int]:
        return (
            self.generation,
            self.step_index,
            self.timestep_token,
            self.rank,
            self.ulysses_size,
        )


@dataclass
class ReplayOffTrace:
    records: list[ReplayOffStepRecord] = field(default_factory=list)
    sample_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        identities = [record.identity for record in self.records]
        return {
            "sample_calls": self.sample_calls,
            "step_count": len(self.records),
            "shared_step_calls": 2 * len(self.records),
            "unique_identity_count": len(set(identities)),
            "steps": [asdict(record) for record in self.records],
        }


@dataclass
class _PendingPair:
    model_id: str
    noisy_latents: Any
    timesteps: Any
    rotary_embs: Any
    batch_vae_seqlen: tuple[int, ...]
    source_tokens: int
    pair_tokens: int
    timestep_token: str


@dataclass
class _ActiveSample:
    generation: int
    action_prompt: Any
    negative_prompt: Any
    completed_steps: int = 0
    pending: Optional[_PendingPair] = None


class InstalledReplayOffAuditHook:
    """Observe the untouched official two-forward sampler without K/V replay."""

    def __init__(
        self,
        renderer_or_diffusion: Any,
        *,
        rank: int,
        ulysses_size: int,
        expected_steps: int,
        expected_source_tokens: int,
    ) -> None:
        self.diffusion = resolve_diffusion_core(renderer_or_diffusion)
        self.rank = int(rank)
        self.ulysses_size = int(ulysses_size)
        self.expected_steps = int(expected_steps)
        self.expected_source_tokens = int(expected_source_tokens)
        self.trace = ReplayOffTrace()
        self.restored = False
        self._active: Optional[_ActiveSample] = None
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._original_sample = getattr(self.diffusion, "sample", None)
        self._original_shared_step = getattr(self.diffusion, "shared_step", None)
        if not callable(self._original_sample) or not callable(
            self._original_shared_step
        ):
            raise SourceKVCarrierOracleError(
                "replay-off observer requires callable sample/shared_step"
            )
        if getattr(self.diffusion, "transformer_2", None) is not None:
            raise SourceKVCarrierOracleError(
                "replay-off observer supports only the audited 1.3B expert"
            )
        if (
            self.ulysses_size != EXPECTED_ULYSSES_SIZE
            or not 0 <= self.rank < self.ulysses_size
            or self.expected_steps != EXPECTED_STEPS
            or self.expected_source_tokens <= 0
        ):
            raise SourceKVCarrierOracleError(
                "replay-off observer rank/step/geometry contract differs"
            )
        for name in ("sample", "shared_step"):
            if name in vars(self.diffusion):
                raise SourceKVCarrierOracleError(
                    f"refusing to stack replay-off observer on {name} override"
                )

    def _set_patch(self, name: str, value: Any) -> None:
        owner = self.diffusion
        instance = vars(owner)
        had_instance = name in instance
        previous = instance.get(name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous))

    def install(self) -> None:
        if self._patches:
            raise SourceKVCarrierOracleError(
                "replay-off observer is already installed"
            )

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        setattr(sample_wrapper, "_bernini_replay_off_audit", self)
        setattr(shared_wrapper, "_bernini_replay_off_audit", self)
        try:
            self._set_patch("shared_step", shared_wrapper)
            self._set_patch("sample", sample_wrapper)
        except Exception:
            self.restore()
            raise
        self.restored = False

    def restore(self) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had_instance, previous = self._patches.pop()
            try:
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except Exception as error:
                errors.append(error)
        self._active = None
        self.restored = not errors
        if errors:
            raise SourceKVCarrierOracleError(
                f"failed to restore {len(errors)} replay-off observer hook(s)"
            ) from errors[0]

    def _validate_sample_call(self, values: Mapping[str, Any]) -> None:
        if (
            values.get("guidance_mode") != legacy.GUIDANCE_MODE
            or int(values.get("num_inference_steps")) != self.expected_steps
            or int(values.get("num_frames")) != EXPECTED_FRAMES
            or not math.isclose(
                float(values.get("flow_shift")),
                FLOW_SHIFT,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            raise SourceKVCarrierOracleError(
                "replay-off sample parameters differ from the on arm"
            )
        if (
            values.get("image_vae_latents") is not None
            or values.get("multi_image_vae_latents") is not None
        ):
            raise SourceKVCarrierOracleError(
                "replay-off observer saw a forbidden image reference"
            )
        videos = values.get("multi_video_vae_latents")
        if not isinstance(videos, (list, tuple)) or len(videos) != 1:
            raise SourceKVCarrierOracleError(
                "replay-off observer requires exactly one source video"
            )
        if values.get("prompt_embeds") is None or values.get(
            "uncond_prompt_embeds"
        ) is None:
            raise SourceKVCarrierOracleError(
                "replay-off action/negative prompts are required"
            )

    def _pair_geometry(self, values: Mapping[str, Any]) -> tuple[int, int]:
        noisy_shape = _shape(values.get("noisy_latents"), label="off paired latents")
        if len(noisy_shape) != 3 or noisy_shape[0] != 1 or noisy_shape[1] % 2:
            raise SourceKVCarrierOracleError(
                "replay-off paired latents must have shape [1,2N,D]"
            )
        pair_tokens = noisy_shape[1]
        if pair_tokens <= 0:
            raise SourceKVCarrierOracleError("replay-off pair cannot be empty")
        lengths = _metadata_tuple(
            values.get("batch_vae_seqlen"), label="off batch_vae_seqlen"
        )
        rotary_shape = _shape(values.get("rotary_embs"), label="off paired rotary")
        if lengths != (pair_tokens,) or (
            len(rotary_shape) != 4
            or rotary_shape[0] != 1
            or rotary_shape[1] != 1
            or rotary_shape[2] != pair_tokens
            or rotary_shape[3] <= 0
        ):
            raise SourceKVCarrierOracleError(
                "replay-off pair metadata/rotary geometry differs"
            )
        source_tokens = pair_tokens // 2
        if source_tokens != self.expected_source_tokens:
            raise SourceKVCarrierOracleError(
                "replay-off runtime source boundary differs from input geometry"
            )
        return source_tokens, pair_tokens

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.trace.sample_calls:
            raise SourceKVCarrierOracleError(
                "replay-off observer permits exactly one sample call"
            )
        values = _bind_call(self._original_sample, args, kwargs)
        self._validate_sample_call(values)
        self._active = _ActiveSample(
            generation=0,
            action_prompt=values["prompt_embeds"],
            negative_prompt=values["uncond_prompt_embeds"],
        )
        try:
            result = self._original_sample(*args, **kwargs)
            state = self._active
            if (
                state is None
                or state.pending is not None
                or state.completed_steps != self.expected_steps
                or len(self.trace.records) != self.expected_steps
            ):
                raise SourceKVCarrierOracleError(
                    "replay-off sample lacked 40 complete negative/action pairs"
                )
            self.trace.sample_calls = 1
            return result
        finally:
            self._active = None

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise SourceKVCarrierOracleError(
                "replay-off shared_step ran outside the observed sample"
            )
        values = _bind_call(self._original_shared_step, args, kwargs)
        model_id = str(values.get("model_id"))
        if model_id != "transformer_1":
            raise SourceKVCarrierOracleError(
                "replay-off observed a non-1.3B transformer route"
            )
        source_tokens, pair_tokens = self._pair_geometry(values)
        step_index = state.completed_steps
        timestep_token = _canonical_timestep_token(
            values.get("timesteps"), step_index=step_index
        )
        prompt = values.get("cond_embeds")
        if state.pending is None:
            if prompt is not state.negative_prompt:
                raise SourceKVCarrierOracleError(
                    "replay-off first per-step call is not negative"
                )
            prediction = self._original_shared_step(*args, **kwargs)
            state.pending = _PendingPair(
                model_id=model_id,
                noisy_latents=values["noisy_latents"],
                timesteps=values["timesteps"],
                rotary_embs=values["rotary_embs"],
                batch_vae_seqlen=(pair_tokens,),
                source_tokens=source_tokens,
                pair_tokens=pair_tokens,
                timestep_token=timestep_token,
            )
            return prediction
        pending = state.pending
        if prompt is not state.action_prompt:
            raise SourceKVCarrierOracleError(
                "replay-off second per-step call is not action"
            )
        if model_id != pending.model_id:
            raise SourceKVCarrierOracleError(
                "replay-off negative/action model route differs"
            )
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            if values.get(name) is not getattr(pending, name):
                raise SourceKVCarrierOracleError(
                    f"replay-off negative/action {name} identity differs"
                )
        if (source_tokens, pair_tokens, timestep_token) != (
            pending.source_tokens,
            pending.pair_tokens,
            pending.timestep_token,
        ):
            raise SourceKVCarrierOracleError(
                "replay-off negative/action geometry or timestep differs"
            )
        prediction = self._original_shared_step(*args, **kwargs)
        self.trace.records.append(
            ReplayOffStepRecord(
                generation=state.generation,
                step_index=step_index,
                timestep_token=timestep_token,
                rank=self.rank,
                ulysses_size=self.ulysses_size,
                model_id=model_id,
                source_tokens_runtime=source_tokens,
                pair_tokens_runtime=pair_tokens,
            )
        )
        state.completed_steps += 1
        state.pending = None
        return prediction


@contextmanager
def replay_off_audit_hook(
    renderer_or_diffusion: Any,
    *,
    rank: int,
    ulysses_size: int,
    expected_steps: int,
    expected_source_tokens: int,
) -> Iterator[InstalledReplayOffAuditHook]:
    observer = InstalledReplayOffAuditHook(
        renderer_or_diffusion,
        rank=rank,
        ulysses_size=ulysses_size,
        expected_steps=expected_steps,
        expected_source_tokens=expected_source_tokens,
    )
    observer.install()
    try:
        yield observer
    finally:
        observer.restore()


class InstalledSourceKVCarrierHook:
    """Reversible instance hook around the two official per-step calls."""

    def __init__(
        self,
        renderer_or_diffusion: Any,
        *,
        cache_bank: replay_core.SourceKVCacheBank,
        noop_prompt_embeds: Any,
        rank: int,
        ulysses_size: int,
        expected_steps: int,
        expected_source_tokens: int,
    ) -> None:
        self.diffusion = resolve_diffusion_core(renderer_or_diffusion)
        self.cache_bank = cache_bank
        self.noop_prompt_embeds = noop_prompt_embeds
        self.rank = int(rank)
        self.ulysses_size = int(ulysses_size)
        self.expected_steps = int(expected_steps)
        self.expected_source_tokens = int(expected_source_tokens)
        self.trace = CarrierTrace()
        self.restored = False
        self._active: Optional[_ActiveSample] = None
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._original_sample = getattr(self.diffusion, "sample", None)
        self._original_shared_step = getattr(self.diffusion, "shared_step", None)
        if not callable(self._original_sample) or not callable(
            self._original_shared_step
        ):
            raise SourceKVCarrierOracleError(
                "diffusion sample/shared_step must be callable"
            )
        if getattr(self.diffusion, "transformer_2", None) is not None:
            raise SourceKVCarrierOracleError(
                "carrier oracle supports only the audited single 1.3B expert"
            )
        if self.ulysses_size != EXPECTED_ULYSSES_SIZE:
            raise SourceKVCarrierOracleError("carrier oracle requires Ulysses=4")
        if not 0 <= self.rank < self.ulysses_size:
            raise SourceKVCarrierOracleError("invalid Ulysses rank")
        if self.expected_steps != EXPECTED_STEPS:
            raise SourceKVCarrierOracleError("carrier hook is fixed to 40 steps")
        if self.expected_source_tokens <= 0:
            raise SourceKVCarrierOracleError(
                "expected source-token count must be positive"
            )
        noop_shape = _shape(noop_prompt_embeds, label="no-op prompt embeddings")
        if len(noop_shape) != 3 or noop_shape[0] != 1 or noop_shape[1] <= 0:
            raise SourceKVCarrierOracleError(
                "no-op prompt embeddings must have shape [1,L,D]"
            )
        for name in ("sample", "shared_step"):
            if name in vars(self.diffusion):
                raise SourceKVCarrierOracleError(
                    f"refusing to stack on an instance-level {name} override"
                )

    def _set_patch(self, name: str, value: Any) -> None:
        owner = self.diffusion
        instance = vars(owner)
        had_instance = name in instance
        previous = instance.get(name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous))

    def install(self) -> None:
        if self._patches:
            raise SourceKVCarrierOracleError("carrier hook is already installed")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        setattr(sample_wrapper, "_bernini_source_kv_carrier", self)
        setattr(shared_wrapper, "_bernini_source_kv_carrier", self)
        try:
            self._set_patch("shared_step", shared_wrapper)
            self._set_patch("sample", sample_wrapper)
        except Exception:
            self.restore()
            raise
        self.restored = False

    def restore(self) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had_instance, previous = self._patches.pop()
            try:
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except Exception as error:
                errors.append(error)
        self._active = None
        self.restored = not errors
        if errors:
            raise SourceKVCarrierOracleError(
                f"failed to restore {len(errors)} carrier hook(s)"
            ) from errors[0]

    def _clear_bank(self) -> None:
        if self.cache_bank.identity is not None:
            self.cache_bank.clear()

    def _validate_sample_call(self, values: Mapping[str, Any]) -> None:
        if values.get("guidance_mode") != legacy.GUIDANCE_MODE:
            raise SourceKVCarrierOracleError(
                "carrier hook requires guidance_mode='v2v_apg'"
            )
        if int(values.get("num_inference_steps")) != self.expected_steps:
            raise SourceKVCarrierOracleError(
                "sample did not request exactly 40 steps"
            )
        if int(values.get("num_frames")) != EXPECTED_FRAMES:
            raise SourceKVCarrierOracleError(
                "sample did not request exactly 81 frames"
            )
        if not math.isclose(
            float(values.get("flow_shift")), FLOW_SHIFT, rel_tol=0.0, abs_tol=1e-8
        ):
            raise SourceKVCarrierOracleError("sample flow shift differs from 5")
        if values.get("image_vae_latents") is not None:
            raise SourceKVCarrierOracleError("reference images are forbidden")
        if values.get("multi_image_vae_latents") is not None:
            raise SourceKVCarrierOracleError("multi-image references are forbidden")
        videos = values.get("multi_video_vae_latents")
        if not isinstance(videos, (list, tuple)) or len(videos) != 1:
            raise SourceKVCarrierOracleError(
                "carrier oracle requires exactly one source video latent"
            )
        if values.get("prompt_embeds") is None or values.get(
            "uncond_prompt_embeds"
        ) is None:
            raise SourceKVCarrierOracleError(
                "action and negative prompt embeddings are required"
            )

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.trace.sample_calls:
            raise SourceKVCarrierOracleError(
                "carrier oracle permits exactly one non-nested sample call"
            )
        values = _bind_call(self._original_sample, args, kwargs)
        self._validate_sample_call(values)
        self._active = _ActiveSample(
            generation=0,
            action_prompt=values["prompt_embeds"],
            negative_prompt=values["uncond_prompt_embeds"],
        )
        try:
            result = self._original_sample(*args, **kwargs)
            state = self._active
            if state is None:
                raise SourceKVCarrierOracleError("carrier sample state disappeared")
            if state.pending is not None:
                raise SourceKVCarrierOracleError(
                    "sample returned after negative but before action replay"
                )
            if state.completed_steps != self.expected_steps:
                raise SourceKVCarrierOracleError(
                    "sample did not execute exactly 40 carrier triplets"
                )
            if len(self.trace.records) != self.expected_steps:
                raise SourceKVCarrierOracleError(
                    "carrier trace and completed-step counts differ"
                )
            self.trace.sample_calls = 1
            return result
        finally:
            self._active = None
            self._clear_bank()

    def _pair_geometry(self, values: Mapping[str, Any]) -> tuple[int, int]:
        noisy_shape = _shape(values.get("noisy_latents"), label="paired latents")
        if len(noisy_shape) != 3 or noisy_shape[0] != 1:
            raise SourceKVCarrierOracleError(
                "paired latents must have shape [1,2N,D]"
            )
        pair_tokens = noisy_shape[1]
        if pair_tokens <= 0 or pair_tokens % 2:
            raise SourceKVCarrierOracleError(
                "paired latent sequence must contain equal source/target spans"
            )
        lengths = _metadata_tuple(
            values.get("batch_vae_seqlen"), label="batch_vae_seqlen"
        )
        if lengths != (pair_tokens,):
            raise SourceKVCarrierOracleError(
                "shared_step batch_vae_seqlen differs from paired tensor"
            )
        rotary_shape = _shape(values.get("rotary_embs"), label="paired rotary")
        if (
            len(rotary_shape) != 4
            or rotary_shape[0] != 1
            or rotary_shape[1] != 1
            or rotary_shape[2] != pair_tokens
            or rotary_shape[3] <= 0
        ):
            raise SourceKVCarrierOracleError(
                "paired rotary must have official shape [1,1,2N,D/2]"
            )
        source_tokens = pair_tokens // 2
        if source_tokens != self.expected_source_tokens:
            raise SourceKVCarrierOracleError(
                "runtime paired source boundary differs from the source latent "
                f"geometry: {source_tokens} != {self.expected_source_tokens}"
            )
        return source_tokens, pair_tokens

    def _carrier_call(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        *,
        source_tokens: int,
        state: _ActiveSample,
        step_index: int,
        timestep_token: str,
    ) -> None:
        values = _bind_call(self._original_shared_step, args, kwargs)
        source_latents = values["noisy_latents"][:, :source_tokens, :]
        source_rotary = values["rotary_embs"][:, :, :source_tokens, :]
        replacements = {
            "noisy_latents": source_latents,
            "rotary_embs": source_rotary,
            "cond_embeds": self.noop_prompt_embeds,
            "batch_vae_seqlen": [source_tokens],
            "batch_text_seqlen": [
                _shape(
                    self.noop_prompt_embeds,
                    label="no-op prompt embeddings",
                )[1]
            ],
        }
        call_args, call_kwargs = tuple(args), dict(kwargs)
        for name, value in replacements.items():
            call_args, call_kwargs = _replace_argument(
                self._original_shared_step,
                call_args,
                call_kwargs,
                name=name,
                value=value,
            )
        with _no_grad_context(), replay_core.source_kv_replay_invocation(
            self.cache_bank,
            mode=replay_core.CAPTURE_MODE,
            branch_tag=replay_core.CAPTURE_BRANCH_TAG,
            generation=state.generation,
            step_index=step_index,
            timestep_token=timestep_token,
            rank=self.rank,
            ulysses_size=self.ulysses_size,
        ):
            self._original_shared_step(*call_args, **call_kwargs)

    def _replay_call(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        *,
        branch_tag: str,
        state: _ActiveSample,
        step_index: int,
        timestep_token: str,
    ) -> Any:
        with _no_grad_context(), replay_core.source_kv_replay_invocation(
            self.cache_bank,
            mode=replay_core.REPLAY_MODE,
            branch_tag=branch_tag,
            generation=state.generation,
            step_index=step_index,
            timestep_token=timestep_token,
            rank=self.rank,
            ulysses_size=self.ulysses_size,
        ):
            return self._original_shared_step(*args, **dict(kwargs))

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise SourceKVCarrierOracleError(
                "shared_step ran outside the validated sample call"
            )
        values = _bind_call(self._original_shared_step, args, kwargs)
        model_id = str(values.get("model_id"))
        if model_id != "transformer_1":
            raise SourceKVCarrierOracleError(
                "carrier oracle observed a non-1.3B transformer route"
            )
        prompt = values.get("cond_embeds")
        source_tokens, pair_tokens = self._pair_geometry(values)
        step_index = state.completed_steps
        timestep_token = _canonical_timestep_token(
            values.get("timesteps"), step_index=step_index
        )

        if state.pending is None:
            if prompt is not state.negative_prompt:
                raise SourceKVCarrierOracleError(
                    "first official per-step call is not the exact negative prompt"
                )
            try:
                self._carrier_call(
                    args,
                    kwargs,
                    source_tokens=source_tokens,
                    state=state,
                    step_index=step_index,
                    timestep_token=timestep_token,
                )
                prediction = self._replay_call(
                    args,
                    kwargs,
                    branch_tag="frozen_negative",
                    state=state,
                    step_index=step_index,
                    timestep_token=timestep_token,
                )
            except Exception:
                self._clear_bank()
                raise
            state.pending = _PendingPair(
                model_id=model_id,
                noisy_latents=values["noisy_latents"],
                timesteps=values["timesteps"],
                rotary_embs=values["rotary_embs"],
                batch_vae_seqlen=(pair_tokens,),
                source_tokens=source_tokens,
                pair_tokens=pair_tokens,
                timestep_token=timestep_token,
            )
            return prediction

        pending = state.pending
        if prompt is not state.action_prompt:
            raise SourceKVCarrierOracleError(
                "second official per-step call is not the exact action prompt"
            )
        if model_id != pending.model_id:
            raise SourceKVCarrierOracleError("negative/action model_id differs")
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            if values.get(name) is not getattr(pending, name):
                raise SourceKVCarrierOracleError(
                    f"negative/action {name} object identity differs"
                )
        if (source_tokens, pair_tokens, timestep_token) != (
            pending.source_tokens,
            pending.pair_tokens,
            pending.timestep_token,
        ):
            raise SourceKVCarrierOracleError(
                "negative/action sequence or timestep identity differs"
            )
        try:
            prediction = self._replay_call(
                args,
                kwargs,
                branch_tag="frozen_action",
                state=state,
                step_index=step_index,
                timestep_token=timestep_token,
            )
        finally:
            self._clear_bank()
        self.trace.records.append(
            CarrierStepRecord(
                generation=state.generation,
                step_index=step_index,
                timestep_token=timestep_token,
                rank=self.rank,
                ulysses_size=self.ulysses_size,
                model_id=model_id,
                source_tokens_runtime=source_tokens,
                pair_tokens_runtime=pair_tokens,
            )
        )
        state.completed_steps += 1
        state.pending = None
        return prediction


@contextmanager
def source_kv_carrier_hook(
    renderer_or_diffusion: Any,
    *,
    cache_bank: replay_core.SourceKVCacheBank,
    noop_prompt_embeds: Any,
    rank: int,
    ulysses_size: int,
    expected_steps: int,
    expected_source_tokens: int,
) -> Iterator[InstalledSourceKVCarrierHook]:
    bridge = InstalledSourceKVCarrierHook(
        renderer_or_diffusion,
        cache_bank=cache_bank,
        noop_prompt_embeds=noop_prompt_embeds,
        rank=rank,
        ulysses_size=ulysses_size,
        expected_steps=expected_steps,
        expected_source_tokens=expected_source_tokens,
    )
    bridge.install()
    try:
        yield bridge
    finally:
        bridge.restore()


def source_tokens_from_vae_latent_shape(shape: Sequence[int]) -> int:
    values = tuple(int(item) for item in shape)
    if len(values) != 5 or values[0] != 1 or min(values) <= 0:
        raise SourceKVCarrierOracleError(
            "source VAE latent must have shape [1,C,T,H,W]"
        )
    if values[3] % 2 or values[4] % 2:
        raise SourceKVCarrierOracleError(
            "source VAE latent H/W must be divisible by Bernini's 2x2 patch"
        )
    return values[2] * (values[3] // 2) * (values[4] // 2)


def _expected_indices(selection: str) -> list[int]:
    return list(
        replay_core.resolve_block_indices(
            replay_core.EXPECTED_BLOCK_COUNT, selection
        )
    )


def validate_enabled_runtime_certificate(
    core_receipt: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    selection: str,
    expected_source_tokens: int,
    rank: int,
    hook_restored: bool,
) -> dict[str, Any]:
    """Fail closed unless one rank executed the exact carrier schedule."""

    indices = _expected_indices(selection)
    if core_receipt.get("block_indices") != indices:
        raise SourceKVCarrierOracleError(
            "source-K/V receipt has the wrong installed block indices"
        )
    runtime = core_receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise SourceKVCarrierOracleError("source-K/V runtime receipt is missing")
    if runtime.get("restored") is not True or hook_restored is not True:
        raise SourceKVCarrierOracleError(
            "processor and sampler hooks must be restored before certification"
        )
    if runtime.get("installed_block_count") != len(indices):
        raise SourceKVCarrierOracleError("installed block count differs")
    cache = runtime.get("cache")
    if not isinstance(cache, Mapping):
        raise SourceKVCarrierOracleError("source-K/V cache receipt is missing")
    expected_capture = len(indices) * EXPECTED_STEPS
    expected_replay = len(indices) * EXPECTED_STEPS * 2
    if (
        cache.get("identity") is not None
        or cache.get("captured_blocks") != []
        or cache.get("entries") != []
        or cache.get("capture_calls") != expected_capture
        or cache.get("replay_lookups") != expected_replay
        or cache.get("replay_branch_counts")
        != {
            "frozen_action": len(indices) * EXPECTED_STEPS,
            "frozen_negative": len(indices) * EXPECTED_STEPS,
        }
        or cache.get("replay_phase_counts")
        != {
            replay_core.EAGER_EXECUTION: expected_replay,
            replay_core.CHECKPOINT_FORWARD: 0,
            replay_core.CHECKPOINT_RECOMPUTE: 0,
        }
        or cache.get("checkpoint_context_counts")
        != {
            replay_core.CHECKPOINT_FORWARD: 0,
            replay_core.CHECKPOINT_RECOMPUTE: 0,
        }
        or cache.get("retired_identity_count") != EXPECTED_STEPS
    ):
        raise SourceKVCarrierOracleError(
            "rank-local cache did not execute exact capture/replay/clear counts"
        )
    per_block = runtime.get("per_block")
    if not isinstance(per_block, list) or len(per_block) != len(indices):
        raise SourceKVCarrierOracleError("per-block source-K/V evidence differs")
    required_branches = {
        replay_core.CAPTURE_BRANCH_TAG: EXPECTED_STEPS,
        "frozen_negative": EXPECTED_STEPS,
        "frozen_action": EXPECTED_STEPS,
    }
    for index, item in zip(indices, per_block):
        if (
            not isinstance(item, Mapping)
            or item.get("block_index") != index
            or item.get("capture_calls") != EXPECTED_STEPS
            or item.get("replay_calls") != EXPECTED_STEPS * 2
            or item.get("branch_counts") != required_branches
            or item.get("execution_phase_counts")
            != {
                replay_core.EAGER_EXECUTION: EXPECTED_STEPS * 3,
                replay_core.CHECKPOINT_FORWARD: 0,
                replay_core.CHECKPOINT_RECOMPUTE: 0,
            }
            or item.get("verified_post_rope_project_qkv_calls")
            != EXPECTED_STEPS * 3
            or item.get("post_rope_phase_counts")
            != {
                replay_core.EAGER_EXECUTION: EXPECTED_STEPS * 3,
                replay_core.CHECKPOINT_FORWARD: 0,
                replay_core.CHECKPOINT_RECOMPUTE: 0,
            }
            or item.get("last_source_tokens") != expected_source_tokens
            or item.get("ulysses_observed") is not True
            or item.get("rotary_emb_required_non_none") is not True
        ):
            raise SourceKVCarrierOracleError(
                f"block {index} lacks exact 40-capture/80-replay post-RoPE evidence"
            )
    steps = trace.get("steps") if isinstance(trace, Mapping) else None
    if (
        trace.get("sample_calls") != 1
        or trace.get("step_count") != EXPECTED_STEPS
        or trace.get("unique_identity_count") != EXPECTED_STEPS
        or not isinstance(steps, list)
        or len(steps) != EXPECTED_STEPS
    ):
        raise SourceKVCarrierOracleError(
            "carrier hook lacks 40 unique completed step identities"
        )
    identities = []
    for step_index, item in enumerate(steps):
        if (
            not isinstance(item, Mapping)
            or item.get("step_index") != step_index
            or item.get("rank") != rank
            or item.get("ulysses_size") != EXPECTED_ULYSSES_SIZE
            or item.get("source_tokens_runtime") != expected_source_tokens
            or item.get("pair_tokens_runtime") != expected_source_tokens * 2
            or item.get("carrier_forwards") != 1
            or item.get("negative_replay_forwards") != 1
            or item.get("action_replay_forwards") != 1
            or item.get("cleared_after_both_replays") is not True
        ):
            raise SourceKVCarrierOracleError(
                f"carrier step {step_index} has invalid execution evidence"
            )
        identities.append(
            (
                item.get("generation"),
                item.get("step_index"),
                item.get("timestep_token"),
                item.get("rank"),
                item.get("ulysses_size"),
            )
        )
    if len(set(identities)) != EXPECTED_STEPS:
        raise SourceKVCarrierOracleError("carrier identities are not unique")
    return {
        "validated": True,
        "replay": "on",
        "rank": rank,
        "ulysses_size": EXPECTED_ULYSSES_SIZE,
        "block_selection": selection,
        "actual_installed_block_indices": indices,
        "selected_block_count": len(indices),
        "per_layer_capture_calls": EXPECTED_STEPS,
        "per_layer_replay_calls": EXPECTED_STEPS * 2,
        "per_layer_negative_replays": EXPECTED_STEPS,
        "per_layer_action_replays": EXPECTED_STEPS,
        "rank_local_bank_capture_calls": expected_capture,
        "rank_local_bank_replay_lookups": expected_replay,
        "unique_step_identities": EXPECTED_STEPS,
        "source_tokens_runtime": expected_source_tokens,
        "pair_tokens_runtime": expected_source_tokens * 2,
        "post_rope_non_none_rotary_verified": True,
        "eager_no_grad_only": True,
        "checkpoint_contexts": 0,
        "cache_empty_after_each_step": True,
        "processor_restore": True,
        "sampler_hook_restore": True,
        "trace_digest": legacy.object_sha256(trace),
    }


def disabled_runtime_certificate(
    trace: Mapping[str, Any],
    *,
    selection: str,
    source_tokens_from_input_geometry: int,
    rank: int,
    observer_restored: bool,
) -> dict[str, Any]:
    steps = trace.get("steps") if isinstance(trace, Mapping) else None
    if (
        observer_restored is not True
        or trace.get("sample_calls") != 1
        or trace.get("step_count") != EXPECTED_STEPS
        or trace.get("shared_step_calls") != EXPECTED_STEPS * 2
        or trace.get("unique_identity_count") != EXPECTED_STEPS
        or not isinstance(steps, list)
        or len(steps) != EXPECTED_STEPS
    ):
        raise SourceKVCarrierOracleError(
            "replay-off observer lacks exact 40-step/80-forward evidence"
        )
    identities = []
    for step_index, item in enumerate(steps):
        if (
            not isinstance(item, Mapping)
            or item.get("step_index") != step_index
            or item.get("rank") != rank
            or item.get("ulysses_size") != EXPECTED_ULYSSES_SIZE
            or item.get("model_id") != "transformer_1"
            or item.get("source_tokens_runtime")
            != source_tokens_from_input_geometry
            or item.get("pair_tokens_runtime")
            != 2 * source_tokens_from_input_geometry
            or item.get("official_negative_forwards") != 1
            or item.get("official_action_forwards") != 1
        ):
            raise SourceKVCarrierOracleError(
                f"replay-off step {step_index} evidence differs"
            )
        identities.append(
            (
                item.get("generation"),
                item.get("step_index"),
                item.get("timestep_token"),
                item.get("rank"),
                item.get("ulysses_size"),
            )
        )
    if len(set(identities)) != EXPECTED_STEPS:
        raise SourceKVCarrierOracleError(
            "replay-off observed step identities are not unique"
        )
    return {
        "validated": True,
        "replay": "off",
        "rank": rank,
        "ulysses_size": EXPECTED_ULYSSES_SIZE,
        "block_selection": selection,
        "requested_block_indices": _expected_indices(selection),
        "actual_installed_block_indices": [],
        "selected_block_count": 0,
        "per_layer_capture_calls": 0,
        "per_layer_replay_calls": 0,
        "official_negative_forwards": EXPECTED_STEPS,
        "official_action_forwards": EXPECTED_STEPS,
        "observed_shared_step_calls": EXPECTED_STEPS * 2,
        "rank_local_bank_capture_calls": 0,
        "rank_local_bank_replay_lookups": 0,
        "unique_step_identities": EXPECTED_STEPS,
        "source_tokens_runtime": source_tokens_from_input_geometry,
        "pair_tokens_runtime": 2 * source_tokens_from_input_geometry,
        "processor_patch_installed": False,
        "carrier_sampler_hook_installed": False,
        "read_only_observer_installed": True,
        "processor_restore": True,
        "observer_hook_restore": True,
        "trace_digest": legacy.object_sha256(trace),
    }


def validate_four_rank_certificates(
    certificates: Sequence[Mapping[str, Any]], *, replay: str
) -> dict[str, Any]:
    if len(certificates) != EXPECTED_ULYSSES_SIZE:
        raise SourceKVCarrierOracleError("exactly four rank certificates required")
    if sorted(item.get("rank") for item in certificates) != list(
        range(EXPECTED_ULYSSES_SIZE)
    ):
        raise SourceKVCarrierOracleError("rank certificates are incomplete")
    for item in certificates:
        if (
            item.get("validated") is not True
            or item.get("replay") != replay
            or item.get("ulysses_size") != EXPECTED_ULYSSES_SIZE
        ):
            raise SourceKVCarrierOracleError("one rank certificate differs")
    invariant_fields = (
        "block_selection",
        "actual_installed_block_indices",
        "selected_block_count",
        "per_layer_capture_calls",
        "per_layer_replay_calls",
        "rank_local_bank_capture_calls",
        "rank_local_bank_replay_lookups",
        "unique_step_identities",
        "source_tokens_runtime",
        "pair_tokens_runtime",
        "observed_shared_step_calls",
        "official_negative_forwards",
        "official_action_forwards",
    )
    reference = certificates[0]
    if any(
        item.get(name) != reference.get(name)
        for item in certificates[1:]
        for name in invariant_fields
    ):
        raise SourceKVCarrierOracleError(
            "source-K/V runtime counts differ across Ulysses ranks"
        )
    return {
        "validated": True,
        "all_four_ranks_exact": True,
        "replay": replay,
        "block_selection": reference.get("block_selection"),
        "actual_block_indices": reference.get(
            "actual_installed_block_indices"
        ),
        "per_rank": [dict(item) for item in certificates],
        "per_rank_capture_calls": reference.get(
            "rank_local_bank_capture_calls"
        ),
        "per_rank_replay_lookups": reference.get(
            "rank_local_bank_replay_lookups"
        ),
        "cross_rank_capture_calls": sum(
            int(item.get("rank_local_bank_capture_calls", 0))
            for item in certificates
        ),
        "cross_rank_replay_lookups": sum(
            int(item.get("rank_local_bank_replay_lookups", 0))
            for item in certificates
        ),
        "certificates_digest": legacy.object_sha256(
            [dict(item) for item in certificates]
        ),
    }


def model_freeze_certificate(model: Any) -> dict[str, Any]:
    trainable = [
        (name, int(parameter.numel()))
        for name, parameter in model.named_parameters()
        if bool(parameter.requires_grad)
    ]
    lora_names = sorted(
        name
        for name, _ in model.named_modules()
        if "lora_" in name.lower() or ".lora" in name.lower()
    )
    if trainable:
        raise SourceKVCarrierOracleError(
            f"base model has {len(trainable)} trainable parameter tensors"
        )
    if lora_names:
        raise SourceKVCarrierOracleError(
            "LoRA/adapter modules are present in the frozen-base oracle"
        )
    return {
        "base_frozen": True,
        "trainable_parameter_tensors": 0,
        "trainable_parameter_elements": 0,
        "lora_module_count": 0,
    }


def causal_pairing_contract(
    *,
    method_source_revision: str,
    method_source_archive_sha256: str,
    bernini_commit: str,
    veomni_commit: str,
    bernini_inference_files: Mapping[str, str],
    checkpoint_tree_sha256: str,
    checkpoint_content_identity: Mapping[str, Any],
    source_sha256: str,
    instruction_sha256: str,
    action_prompt_sha256: str,
    negative_prompt_sha256: str,
    source_metadata: Mapping[str, Any],
    steps: int,
    seed: int,
    runtime_versions: Mapping[str, str],
) -> dict[str, Any]:
    stable_checkpoint_identity = {
        key: checkpoint_content_identity.get(key)
        for key in (
            "manifest_sha256_computed",
            "manifest_sha256_expected",
            "verified_file_count",
            "every_file_sha256_verified",
            "verified_entries_digest",
        )
    }
    value = {
        "base": "frozen_bernini_r_1p3b",
        "method_source_revision": method_source_revision,
        "method_source_archive_sha256": method_source_archive_sha256,
        "bernini_commit": bernini_commit,
        "veomni_commit": veomni_commit,
        "bernini_inference_files": dict(bernini_inference_files),
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "checkpoint_content_identity": stable_checkpoint_identity,
        "source_video_sha256": source_sha256,
        "instruction_utf8_sha256": instruction_sha256,
        "action_prompt_utf8_sha256": action_prompt_sha256,
        "negative_prompt_utf8_sha256": negative_prompt_sha256,
        "source_derived_bucket_hw": list(
            source_metadata["source_derived_bucket_hw"]
        ),
        "num_frames": EXPECTED_FRAMES,
        "num_inference_steps": int(steps),
        "seed": int(seed),
        "sampler": legacy.sampler_contract(steps=steps, seed=seed),
        "ulysses_size": EXPECTED_ULYSSES_SIZE,
        "rank0_decode_and_save_only": True,
        "compute_dtype": "torch.bfloat16",
        "runtime_versions": dict(runtime_versions),
    }
    value["causal_pairing_digest"] = legacy.object_sha256(value)
    return value


def build_receipt(
    *,
    args: argparse.Namespace,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    source_tokens: int,
    output_path: Path,
    output_sha256: str,
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
    freeze_certificate: Mapping[str, Any],
    four_rank_runtime: Mapping[str, Any],
    rank0_core_receipt: Optional[Mapping[str, Any]],
    pairing: Mapping[str, Any],
    checkpoint_content_identity: Mapping[str, Any],
) -> dict[str, Any]:
    instruction_bytes = args.instruction.encode("utf-8")
    noop_bytes = route_batches.EXACT_NOOP_INSTRUCTION.encode("utf-8")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "bernini_inference_files": dict(inference_file_hashes),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_content_identity": dict(checkpoint_content_identity),
        "weights": {
            "base_checkpoint_path": str(Path(args.checkpoint).resolve()),
            "base_checkpoint_loaded": True,
            "base_checkpoint_content_verified": (
                checkpoint_content_identity.get("every_file_sha256_verified")
                is True
            ),
            "base_checkpoint_verified_file_count": (
                checkpoint_content_identity.get("verified_file_count")
            ),
            "base_checkpoint_content_manifest_sha256": (
                checkpoint_content_identity.get("manifest_sha256_computed")
            ),
            "base_frozen": True,
            "adapter_argument_supported": False,
            "legacy_full644_artifact_used": False,
            "adapter_weights_loaded": False,
            "adapter_weights_merged": False,
            "peft_model_constructed": False,
            "lora_module_count": 0,
        },
        "optimization": {
            "zero_training": True,
            "training_steps": 0,
            "optimizer_constructed": False,
            "backward_calls": 0,
            **dict(freeze_certificate),
        },
        "input": {
            "source_video_path": args.original_source_path,
            "staged_source_video_path": str(source_path),
            "source_video_sha256": source_sha256,
            "expected_source_video_sha256": args.expected_source_sha256,
            "staged_source_hash_verified": (
                source_sha256 == args.expected_source_sha256
            ),
            "instruction_utf8_sha256": hashlib.sha256(
                instruction_bytes
            ).hexdigest(),
            "instruction_utf8_bytes": len(instruction_bytes),
            "accepted_model_conditions": [
                "source_video",
                "edit_instruction",
            ],
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "external_mask_or_swept_tube": False,
            "external_tracking_pose_or_trajectory": False,
            "optical_flow": False,
            "reference_image_or_video": False,
            "first_frame_anchor": False,
            "external_shared_i0": False,
        },
        "preprocessing": {
            **dict(source_metadata),
            "source_tokens_from_input_geometry": source_tokens,
            "pair_tokens_from_input_geometry": source_tokens * 2,
        },
        "prompt_contract": {
            "task": "mv2v",
            "semantic_noop_instruction_sha256": hashlib.sha256(
                noop_bytes
            ).hexdigest(),
            "semantic_noop_sha_pinned": (
                hashlib.sha256(noop_bytes).hexdigest()
                == route_batches.EXACT_NOOP_INSTRUCTION_SHA256
            ),
            "negative_prompt_utf8_sha256": hashlib.sha256(
                legacy.DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
            ).hexdigest(),
            "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenizer_fix_mistral_regex": True,
            "max_sequence_length": 512,
            "prompt_enhancer": False,
        },
        "sampling": {
            **legacy.sampler_contract(
                steps=args.num_inference_steps, seed=args.seed
            ),
            "single_expert": "transformer_1",
            "ulysses_size": EXPECTED_ULYSSES_SIZE,
            "rank0_decode_and_save_only": True,
        },
        "causal_control": {
            "replay": args.replay,
            "block_selection": args.block_selection,
            "main_arm": (
                args.replay == "on"
                and args.block_selection == replay_core.MAIN_BLOCK_SELECTION
            ),
            "same_seed_replay_off_control_available": True,
            "control_generated_in_this_run": args.replay == "off",
            "paired_counterpart_receipt_required_for_causal_claim": True,
            "pairing_contract": dict(pairing),
            "causal_pairing_digest": pairing["causal_pairing_digest"],
            "arm_excluded_from_pairing_digest": [
                "replay",
                "block_selection",
                "output_path",
            ],
        },
        "oracle": {
            "zero_training": True,
            "base_frozen": True,
            "integration_status": "integrated_frozen_base_runtime_certificate",
            "replay": args.replay,
            "requested_block_selection": args.block_selection,
            "actual_block_indices": four_rank_runtime[
                "actual_block_indices"
            ],
            "runtime_execution_certificate": dict(four_rank_runtime),
            "rank0_source_kv_core_receipt": (
                None
                if rank0_core_receipt is None
                else dict(rank0_core_receipt)
            ),
        },
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
            "frame_count": EXPECTED_FRAMES,
            "fps": legacy.FPS,
            "height": source_metadata["source_derived_bucket_hw"][0],
            "width": source_metadata["source_derived_bucket_hw"][1],
            "audio_preserved": False,
        },
        "runtime_versions": dict(runtime_versions),
        "experimental_inference": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    if "adapter" in receipt:
        raise SourceKVCarrierOracleError(
            "frozen-base receipt must not contain a legacy adapter object"
        )
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def _sample_with_optional_replay(
    model: Any,
    *,
    replay: str,
    block_selection: str,
    noop_prompt_embeds: Any,
    rank: int,
    source_tokens: int,
    sample_kwargs: Mapping[str, Any],
) -> tuple[Any, Optional[dict[str, Any]], dict[str, Any]]:
    """Execute one arm; the off arm never touches the carrier installer."""

    if replay == "off":
        observer: Optional[InstalledReplayOffAuditHook] = None
        with replay_off_audit_hook(
            model,
            rank=rank,
            ulysses_size=EXPECTED_ULYSSES_SIZE,
            expected_steps=EXPECTED_STEPS,
            expected_source_tokens=source_tokens,
        ) as installed_observer:
            observer = installed_observer
            generated = model.sample(**dict(sample_kwargs))
        if observer is None:
            raise SourceKVCarrierOracleError(
                "replay-off observer did not install"
            )
        return (
            generated,
            None,
            disabled_runtime_certificate(
                observer.trace.as_dict(),
                selection=block_selection,
                source_tokens_from_input_geometry=source_tokens,
                rank=rank,
                observer_restored=observer.restored,
            ),
        )
    if replay != "on":
        raise SourceKVCarrierOracleError("unknown replay arm")
    hook: Optional[InstalledSourceKVCarrierHook] = None
    patch: Optional[replay_core.SourceKVReplayPatchHandle] = None
    with replay_core.source_kv_replay(
        model, selection=block_selection
    ) as patch_handle:
        patch = patch_handle
        with source_kv_carrier_hook(
            model,
            cache_bank=patch_handle.cache_bank,
            noop_prompt_embeds=noop_prompt_embeds,
            rank=rank,
            ulysses_size=EXPECTED_ULYSSES_SIZE,
            expected_steps=EXPECTED_STEPS,
            expected_source_tokens=source_tokens,
        ) as installed:
            hook = installed
            generated = model.sample(**dict(sample_kwargs))
    if patch is None or hook is None:
        raise SourceKVCarrierOracleError("carrier contexts did not install")
    core_receipt = patch.receipt()
    trace = hook.trace.as_dict()
    certificate = validate_enabled_runtime_certificate(
        core_receipt,
        trace,
        selection=block_selection,
        expected_source_tokens=source_tokens,
        rank=rank,
        hook_restored=hook.restored,
    )
    return generated, core_receipt, certificate


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute():
        raise SourceKVCarrierOracleError("source video must be absolute")
    source_path = legacy._plain_file(
        source_requested.resolve(strict=True), label="source video"
    )
    original_source = Path(args.original_source_path).expanduser()
    if not original_source.is_absolute():
        raise SourceKVCarrierOracleError(
            "original source path must be absolute"
        )
    output_path, receipt_path = legacy._resolve_output(args.output)

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise SourceKVCarrierOracleError(str(error)) from error
    if transformer_config["num_attention_heads"] % EXPECTED_ULYSSES_SIZE:
        raise SourceKVCarrierOracleError(
            "attention heads are not divisible by Ulysses=4"
        )
    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    if not checkpoint_manifest.is_absolute():
        raise SourceKVCarrierOracleError(
            "checkpoint content manifest must be an absolute path"
        )
    inference_file_hashes = legacy.validate_inference_source_files(bernini_root)
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT:
        raise SourceKVCarrierOracleError("runtime mv2v prompt differs")
    if DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT:
        raise SourceKVCarrierOracleError("runtime negative prompt differs")
    route_batches.validate_noop_instruction(
        route_batches.EXACT_NOOP_INSTRUCTION
    )
    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise SourceKVCarrierOracleError(
            "four-rank oracle requires AUH ROCm GPUs"
        )
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_validation: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_validation[0] = {
                "ok": True,
                "identity": validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_validation[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_validation, src=0)
    checkpoint_result = checkpoint_validation[0]
    if (
        not isinstance(checkpoint_result, Mapping)
        or checkpoint_result.get("ok") is not True
        or not isinstance(checkpoint_result.get("identity"), Mapping)
    ):
        detail = (
            checkpoint_result.get("error")
            if isinstance(checkpoint_result, Mapping)
            else "missing rank-zero checkpoint audit"
        )
        raise SourceKVCarrierOracleError(
            f"rank-zero checkpoint content validation failed: {detail}"
        )
    checkpoint_content_identity = dict(checkpoint_result["identity"])

    source_tensor, source_metadata, source_sha256 = prepare_hashed_source_snapshot(
        source_path
    )
    if source_sha256 != args.expected_source_sha256:
        raise SourceKVCarrierOracleError(
            "staged source SHA-256 differs from the launcher-bound original"
        )
    full_prompt = legacy.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = legacy.build_training_prompt(
        route_batches.EXACT_NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )
    negative_prompt = legacy.DEFAULT_NEGATIVE_PROMPT

    config_dir = bernini_root / "configs/bernini_renderer_wan21_1p3b"
    config = BerniniRendererConfig.from_pretrained(
        str(config_dir),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        legacy.trainer.validate_renderer_config_mapping(
            config.to_dict(), checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise SourceKVCarrierOracleError(str(error)) from error
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    freeze = model_freeze_certificate(model)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        **legacy.tokenizer_load_kwargs(),
    )
    input_ids, attention_mask = legacy._tokenize_training_prompt(
        tokenizer, full_prompt
    )
    negative_ids, negative_mask = legacy._tokenize_renderer_negative(
        tokenizer, negative_prompt
    )
    noop_ids, noop_mask = legacy._tokenize_training_prompt(tokenizer, noop_prompt)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    with torch.no_grad():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        )
    expected_bucket = source_metadata["source_derived_bucket_hw"]
    expected_latent_shape = (
        1,
        int(vae.config.z_dim),
        legacy.LATENT_FRAME_COUNT,
        int(expected_bucket[0]) // 8,
        int(expected_bucket[1]) // 8,
    )
    if tuple(int(item) for item in source_latent.shape) != expected_latent_shape:
        raise SourceKVCarrierOracleError("source VAE latent shape differs")
    source_tokens = source_tokens_from_vae_latent_shape(source_latent.shape)
    if source_tokens != args.expected_source_tokens:
        raise SourceKVCarrierOracleError(
            "source token geometry differs from --expected-source-tokens: "
            f"{source_tokens} != {args.expected_source_tokens}"
        )
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    noop_prompt_embeds = None
    if args.replay == "on":
        model.t5_text_encoder.to(device)
        with torch.no_grad():
            noop_prompt_embeds = model.encode_prompt(
                noop_ids.to(device), noop_mask.to(device)
            )
        model.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()

    sample_kwargs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "uncond_input_ids": negative_ids.to(device),
        "uncond_attention_mask": negative_mask.to(device),
        "image_vae_latents": None,
        "multi_video_vae_latents": [source_latent],
        "multi_image_vae_latents": None,
        "width": int(expected_bucket[1]),
        "height": int(expected_bucket[0]),
        "device": device,
        **legacy.sampler_contract(
            steps=args.num_inference_steps, seed=args.seed
        ),
    }
    with torch.no_grad():
        generated_latent, core_receipt, local_certificate = (
            _sample_with_optional_replay(
                model,
                replay=args.replay,
                block_selection=args.block_selection,
                noop_prompt_embeds=noop_prompt_embeds,
                rank=distributed.rank,
                source_tokens=source_tokens,
                sample_kwargs=sample_kwargs,
            )
        )
    if tuple(int(item) for item in generated_latent.shape) != expected_latent_shape:
        raise SourceKVCarrierOracleError("generated latent shape differs")
    freeze_after = model_freeze_certificate(model)
    if freeze_after != freeze:
        raise SourceKVCarrierOracleError("model freeze certificate changed")

    rank_certificates: list[Any] = [None] * EXPECTED_ULYSSES_SIZE
    dist.all_gather_object(rank_certificates, local_certificate)
    four_rank_runtime = validate_four_rank_certificates(
        rank_certificates, replay=args.replay
    )
    model.to("cpu")
    del source_latent
    torch.cuda.empty_cache()

    instruction_sha = hashlib.sha256(args.instruction.encode("utf-8")).hexdigest()
    runtime_versions = {
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "transformers": transformers_version,
        "diffusers": diffusers_version,
    }
    pairing = causal_pairing_contract(
        method_source_revision=args.method_source_revision,
        method_source_archive_sha256=args.method_source_archive_sha256,
        bernini_commit=bernini_revision,
        veomni_commit=veomni_revision,
        bernini_inference_files=inference_file_hashes,
        checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        checkpoint_content_identity=checkpoint_content_identity,
        source_sha256=source_sha256,
        instruction_sha256=instruction_sha,
        action_prompt_sha256=hashlib.sha256(full_prompt.encode("utf-8")).hexdigest(),
        negative_prompt_sha256=hashlib.sha256(
            negative_prompt.encode("utf-8")
        ).hexdigest(),
        source_metadata=source_metadata,
        steps=args.num_inference_steps,
        seed=args.seed,
        runtime_versions=runtime_versions,
    )
    if distributed.rank == 0:
        vae.to(device)
        with torch.no_grad():
            output = _vae_decode(vae, generated_latent)
        vae.to("cpu")
        if tuple(int(item) for item in output.shape) != (
            EXPECTED_FRAMES,
            int(expected_bucket[0]),
            int(expected_bucket[1]),
            3,
        ):
            raise SourceKVCarrierOracleError("decoded output shape differs")
        temporary = output_path.with_name(
            f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
        )
        if temporary.exists() or temporary.is_symlink():
            raise SourceKVCarrierOracleError("stale temporary output exists")
        save_output(output, str(temporary), fps=int(legacy.FPS))
        os.replace(temporary, output_path)
        from tools import materialize_vae

        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(
            output_path
        )
        legacy.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
        if tuple(encoded_hw) != tuple(expected_bucket):
            raise SourceKVCarrierOracleError("encoded output geometry differs")
        receipt = build_receipt(
            args=args,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            source_tokens=source_tokens,
            output_path=output_path,
            output_sha256=legacy.file_sha256(output_path),
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            inference_file_hashes=inference_file_hashes,
            runtime_versions=runtime_versions,
            freeze_certificate=freeze_after,
            four_rank_runtime=four_rank_runtime,
            rank0_core_receipt=core_receipt,
            pairing=pairing,
            checkpoint_content_identity=checkpoint_content_identity,
        )
        legacy._atomic_write_json(receipt_path, receipt)
        print(legacy.canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CarrierStepRecord",
    "CarrierTrace",
    "EXPECTED_DOG_PAIR_TOKENS",
    "EXPECTED_DOG_SOURCE_TOKENS",
    "EXPECTED_STEPS",
    "InstalledSourceKVCarrierHook",
    "RECEIPT_SCHEMA",
    "SourceKVCarrierOracleError",
    "build_parser",
    "build_receipt",
    "causal_pairing_contract",
    "disabled_runtime_certificate",
    "main",
    "model_freeze_certificate",
    "resolve_diffusion_core",
    "source_kv_carrier_hook",
    "source_tokens_from_vae_latent_shape",
    "validate_cli",
    "validate_enabled_runtime_certificate",
    "validate_four_rank_certificates",
]

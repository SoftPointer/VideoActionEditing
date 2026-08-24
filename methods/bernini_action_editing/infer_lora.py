#!/usr/bin/env python3
"""Source-only Bernini-R 1.3B base/LoRA inference on exact 81-frame clips.

The model-facing inputs are deliberately closed: one source video and one edit
instruction.  There is no target-video, mask, track, swept-tube, pose,
trajectory, reference-image, or shared-I0 argument.  Ground-truth targets, if
used for evaluation, belong in a separate post-generation scorer process.

This harness keeps the official Bernini renderer and sampler, but owns the
small amount of glue required for training/inference parity:

* the exact training ``mv2v`` system prefix and Wan prompt cleaner;
* ``fix_mistral_regex=True`` for the frozen T5 tokenizer;
* all 81 integer frames at 25 fps and the training area/aspect bucket;
* the single Wan2.1 1.3B transformer, flow shift 5 and ``v2v_apg``;
* either a frozen-base control or strict PEFT reload before safe LoRA merge; and
* four-rank Ulysses on AUH ROCm, with decode/save on rank zero only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
from datetime import timedelta
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_lora as trainer  # noqa: E402
import action_preservation_decoded_eval_model_authority_v2 as model_authority  # noqa: E402
import exact_local_video_materializer_v1 as exact_video_materializer  # noqa: E402

if (
    Path(exact_video_materializer.__file__).resolve(strict=True)
    != METHOD_ROOT / "exact_local_video_materializer_v1.py"
):
    raise RuntimeError("exact local video materializer import escaped method root")


INFERENCE_RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-inference-receipt-v5"
FRAME_COUNT = 81
FPS = 25.0
LATENT_FRAME_COUNT = 21
MAX_PIXELS = 245_760
SPATIAL_STRIDE = 16
FLOW_SHIFT = 5.0
GUIDANCE_MODE = "v2v_apg"
NUM_INFERENCE_STEPS_DEFAULT = 40
OMEGA_TEXT = 4.0
ETA = 0.5
NORM_THRESHOLD = (50.0, 50.0)
MOMENTUM = 0.0
ULYSSES_SIZE = 4
EXPECTED_ADAPTER_TENSOR_COUNT = 2 * trainer.EXPECTED_LORA_TARGET_MODULES
PEFT_COMPACT_TARGET_MODULES = frozenset(("to_q", "to_k", "to_v", "to_out.0"))
FULL644_PEFT_VERSION = "0.19.1"
FULL644_PEFT_CONFIG_FIELDS = frozenset(
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
V2_PEFT_COMPACT_TARGET_MODULES = {
    "all_attention": PEFT_COMPACT_TARGET_MODULES,
    # PEFT 0.19's minimal-target algorithm must distinguish the selected
    # attn2 Q/O modules from the unselected attn1 Q/O modules.  Bare to_q and
    # to_out.0 suffixes would silently broaden the route during a raw config
    # load, so only these attn2-qualified canonical suffixes are accepted.
    "cross_attn2_qo": frozenset(("attn2.to_q", "attn2.to_out.0")),
}
V2_TRAINING_RECEIPT_SCHEMA = (
    "bernini-r-1p3b-action-preservation-lora-receipt-v2"
)
V4_FULLFIELD_TRAINING_RECEIPT_SCHEMA = (
    "bernini-r-1p3b-self-generated-fullfield-lora-receipt-v4"
)
V4_FULLFIELD_METHOD = "bernini-self-generated-action-fullfield-v4"
V4_FULLFIELD_LORA_RANK = 256
V4_FULLFIELD_LORA_ALPHA = 256
V4_FULLFIELD_TRAINABLE_PARAMETERS = 188_743_680
V2_OBJECTIVE = "bernini-self-generated-action-preservation-canary-v2"
V2_CANARY_SEED = 20260818
V2_SAVE_STEPS = (0, 5, 10, 20)
V2_SIGMA_BINS = (
    (0.0, 0.20),
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 0.80),
    (0.80, 1.0001),
)

# Populated only by the exact15 decoder after a canonical consumption-input
# receipt and both capture receipts have been replayed.  Legacy callers leave
# this empty and retain the original plain-file-only behavior.
_AUTHORIZED_FD_VIEW_FILES: dict[str, Mapping[str, Any]] = {}
_AUTHORIZED_FD_VIEW_DIRECTORIES: dict[str, Mapping[str, Any]] = {}
_ACTIVE_INHERITED_FDS: Mapping[str, Any] | None = None
SOURCE_ONSET_POLICIES = {
    "none": (),
    "hard1": (1.0,),
    "ramp3": (1.0, 0.5, 0.25),
    # Unlike hard1/ramp3, this policy is applied at the native packed UniPC
    # boundary on every solver step.  Its empty post-denoise tuple is
    # intentional: the terminal sigma=+0 projection already makes phase zero
    # exactly source-derived, without a second clone after model.sample.
    "hard1_every_step": (),
}
EVERY_STEP_SOURCE_ONSET_POLICY = "hard1_every_step"
SOURCE_TRAJECTORY_CLAMP_SCHEMA = "bernini-source-phase0-unipc-clamp-v1"
MV2V_SYSTEM_PROMPT = (
    "You are a helpful assistant for editing. You might need to adjust the "
    "video's style, lighting, colors, textures, and the subject's pose or action."
)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# The training validator already pins renderer/training/parallel files.  These
# additional files are imported only by inference and therefore need their own
# byte identities for the hash-bound AUH tar extraction (which has no .git).
BERNINI_INFERENCE_FILE_HASHES = {
    "bernini/pipeline.py": "c6acf05c01a637d9bce69e8160eb6eb4260ff4ec798fd990de8e5aa73999ab40",
    "bernini/cli.py": "26949fbf246003403ed0cca1ec1bbb62c2099fc9740bb17ba5a1e7c86fbc0edf",
    "bernini/io_utils.py": "233541373746f5d97e1cb3680d3c2a41d5d212b797eefb97693afa6e3ab5f30a",
}
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class InferenceContractError(RuntimeError):
    """Raised before generation when a source-only invariant is violated."""


def trim_process_heap() -> None:
    """Return freed glibc arenas after serialized checkpoint/PEFT loading."""

    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


@contextmanager
def serialized_model_load() -> Any:
    """Bound four-rank checkpoint/adapter host-memory peaks inside small cgroups."""

    path = Path(
        f"/tmp/bernini-infer-{os.environ.get('SLURM_JOB_ID', 'none')}-"
        f"{os.environ.get('SLURM_STEP_ID', 'none')}.model-load.lock"
    )
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True)
class AdapterBundle:
    checkpoint_root: Path
    adapter_dir: Path
    adapter_config_path: Path
    adapter_model_path: Path
    training_receipt_path: Path


@dataclass(frozen=True)
class InferenceDistributedContract:
    world_size: int
    rank: int
    local_rank: int
    ulysses_size: int = ULYSSES_SIZE


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
        raise InferenceContractError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    if str(path) in _AUTHORIZED_FD_VIEW_FILES and _ACTIVE_INHERITED_FDS is not None:
        try:
            raw, _ = model_authority.stable_inherited_view_file(
                path,
                inherited_fd_binding=_ACTIVE_INHERITED_FDS,
                label="authorized FD-view file",
            )
        except model_authority.ModelConsumptionAuthorityError as error:
            raise InferenceContractError(str(error)) from error
        return hashlib.sha256(raw).hexdigest()
    if _ACTIVE_INHERITED_FDS is not None:
        try:
            task_root = Path(model_authority.inherited_proc_root(
                _ACTIVE_INHERITED_FDS,
                scope="task",
                role="publication_root",
            ))
            if path.parent == task_root:
                raw, _ = model_authority.stable_inherited_task_file(
                    path,
                    inherited_fd_binding=_ACTIVE_INHERITED_FDS,
                    label="inherited task file",
                )
                return hashlib.sha256(raw).hexdigest()
        except model_authority.ModelConsumptionAuthorityError as error:
            raise InferenceContractError(str(error)) from error
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        if str(path) in _AUTHORIZED_FD_VIEW_FILES and _ACTIVE_INHERITED_FDS is not None:
            raw, _ = model_authority.stable_inherited_view_file(
                path,
                inherited_fd_binding=_ACTIVE_INHERITED_FDS,
                label=label,
            )
            value = json.loads(raw.decode("utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except model_authority.ModelConsumptionAuthorityError as error:
        raise InferenceContractError(str(error)) from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceContractError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise InferenceContractError(f"{label} must contain one JSON object: {path}")
    return value


def _plain_file(path: Path, *, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise InferenceContractError(f"missing {label}: {path}") from error
    if path.is_symlink():
        authorized = _AUTHORIZED_FD_VIEW_FILES.get(str(path))
        if authorized is None:
            raise InferenceContractError(
                f"{label} is an unauthorized symlink: {path}"
            )
        try:
            target = os.readlink(path)
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                observed = model_authority._identity(os.fstat(descriptor))
            finally:
                os.close(descriptor)
        except OSError as error:
            raise InferenceContractError(
                f"cannot replay authorized {label}: {path}"
            ) from error
        if (
            target != authorized["proc_fd_path"]
            or observed != authorized["identity"]
        ):
            raise InferenceContractError(
                f"authorized {label} FD-view identity differs: {path}"
            )
        return path
    if not stat.S_ISREG(mode):
        raise InferenceContractError(f"{label} is not a plain file: {path}")
    return path


def _stable_plain_file_bytes(path: Path, *, label: str) -> bytes:
    """Read one path once while proving its regular-file identity is stable."""

    if str(path) in _AUTHORIZED_FD_VIEW_FILES and _ACTIVE_INHERITED_FDS is not None:
        try:
            raw, _ = model_authority.stable_inherited_view_file(
                path,
                inherited_fd_binding=_ACTIVE_INHERITED_FDS,
                label=label,
            )
        except model_authority.ModelConsumptionAuthorityError as error:
            raise InferenceContractError(str(error)) from error
        return raw
    path = _plain_file(path, label=label)
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fd_before = os.fstat(descriptor)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                raw = handle.read()
            fd_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = path.lstat()
    except OSError as error:
        raise InferenceContractError(f"cannot stably read {label}: {path}: {error}") from error

    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )
    if (
        identity(before) != identity(fd_before)
        or identity(before) != identity(fd_after)
        or identity(before) != identity(after)
        or len(raw) != before.st_size
    ):
        raise InferenceContractError(f"{label} changed while reading: {path}")
    return raw


def _absolute_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise InferenceContractError(f"{label} must be an absolute path: {path}")
    authorized = _AUTHORIZED_FD_VIEW_DIRECTORIES.get(str(path))
    if authorized is not None:
        try:
            direct_fd = authorized.get("direct_fd")
            if direct_fd is None:
                descriptor = os.open(
                    path,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                )
            elif type(direct_fd) is int:
                descriptor = os.dup(direct_fd)
                os.set_inheritable(descriptor, False)
            else:
                raise InferenceContractError(
                    f"authorized {label} direct FD differs: {path}"
                )
            try:
                before = model_authority._identity(os.fstat(descriptor))
                middle = model_authority._identity(os.fstat(descriptor))
                after = model_authority._identity(os.fstat(descriptor))
            finally:
                os.close(descriptor)
        except OSError as error:
            raise InferenceContractError(
                f"cannot replay authorized {label}: {path}"
            ) from error
        if (
            before != authorized["identity"]
            or middle != before
            or after != before
        ):
            raise InferenceContractError(
                f"authorized {label} FD-view identity differs: {path}"
            )
        return path
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise InferenceContractError(f"cannot resolve {label}: {path}: {error}") from error
    if not resolved.is_dir():
        raise InferenceContractError(f"{label} is not a directory: {resolved}")
    return resolved


def expected_lora_target_modules() -> list[str]:
    names = [
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(30)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    ]
    return sorted(names)


def expected_adapter_state_keys(
    targets: Optional[Sequence[str]] = None,
) -> list[str]:
    """Exact PEFT safetensors keys for one receipt-bound Wan route.

    Omitting ``targets`` intentionally retains the legacy 240-module result.
    Preservation-v2 callers must pass the explicit target list reconstructed
    from their signed receipt.
    """

    exact_targets = (
        expected_lora_target_modules() if targets is None else list(targets)
    )
    if (
        not exact_targets
        or len(exact_targets) != len(set(exact_targets))
        or not all(isinstance(module, str) for module in exact_targets)
    ):
        raise InferenceContractError(
            "adapter state-key targets must be unique non-empty strings"
        )

    return sorted(
        f"base_model.model.{module}.lora_{factor}.weight"
        for module in exact_targets
        for factor in ("A", "B")
    )


def apply_source_onset_policy(generated: Any, source: Any, policy: str) -> Any:
    """Anchor early generated latent phases to the source after denoising.

    This is a cheap inference-time boundary-condition ablation, not a training
    reward and not a claim that later identity is preserved.  ``hard1`` makes
    latent phase zero exactly source-derived; ``ramp3`` additionally blends
    the next two phases to reduce an abrupt onset transition.
    """

    try:
        weights = SOURCE_ONSET_POLICIES[policy]
    except KeyError as error:
        raise InferenceContractError(f"unknown source onset policy: {policy!r}") from error
    if not weights:
        return generated
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production runtime always has torch
        raise InferenceContractError("source onset policy requires torch") from error
    if (
        not isinstance(generated, torch.Tensor)
        or not isinstance(source, torch.Tensor)
        or generated.shape != source.shape
        or generated.ndim != 5
        or int(generated.shape[2]) != LATENT_FRAME_COUNT
        or generated.device != source.device
    ):
        raise InferenceContractError(
            "source onset policy requires matching [B,C,21,H,W] latents"
        )
    result = generated.clone()
    for phase, weight in enumerate(weights):
        result[:, :, phase] = (
            (1.0 - weight) * generated[:, :, phase].float()
            + weight * source[:, :, phase].float()
        ).to(dtype=generated.dtype)
    return result


def _pack_wan_source_latent(source: Any) -> Any:
    """Pack ``[B,16,21,H,W]`` in Wan's native ``(1,2,2)`` token order."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - production always has torch
        raise InferenceContractError("source trajectory clamp requires torch") from error
    if (
        not isinstance(source, torch.Tensor)
        or source.ndim != 5
        or int(source.shape[1]) != 16
        or int(source.shape[2]) != LATENT_FRAME_COUNT
        or int(source.shape[3]) <= 0
        or int(source.shape[4]) <= 0
        or int(source.shape[3]) % 2
        or int(source.shape[4]) % 2
        or source.requires_grad
        or not bool(torch.isfinite(source).all().item())
    ):
        raise InferenceContractError(
            "source trajectory clamp requires finite detached [B,16,21,evenH,evenW] latents"
        )
    batch, channels, phases, height, width = map(int, source.shape)
    # [B,C,T,Hg,ph,Wg,pw] -> [B,T,Hg,Wg,ph,pw,C].  This is the
    # patch_vae_latent / scheduler ordering, not an arbitrary flatten.
    return (
        source.reshape(batch, channels, phases, height // 2, 2, width // 2, 2)
        .permute(0, 2, 3, 5, 4, 6, 1)
        .reshape(batch, phases * (height // 2) * (width // 2), 64)
        .detach()
        .contiguous()
    )


def _scheduler_config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        if name not in config:
            raise InferenceContractError(
                f"source trajectory clamp scheduler config lacks {name}"
            )
        return config[name]
    if not hasattr(config, name):
        raise InferenceContractError(
            f"source trajectory clamp scheduler config lacks {name}"
        )
    return getattr(config, name)


def _scalar_float(value: Any, *, label: str) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel") and int(value.numel()) != 1:
            raise InferenceContractError(f"{label} must be scalar")
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        result = float(value)
    except InferenceContractError:
        raise
    except Exception as error:
        raise InferenceContractError(f"{label} must be a numeric scalar") from error
    if not math.isfinite(result):
        raise InferenceContractError(f"{label} must be finite")
    return result


def _extract_scheduler_step_argument(
    args: Sequence[Any], kwargs: Mapping[str, Any], *, index: int, name: str
) -> Any:
    positional = len(args) > index
    keyword = name in kwargs
    if positional and keyword:
        raise InferenceContractError(f"scheduler.step received duplicate {name}")
    if positional:
        return args[index]
    if keyword:
        return kwargs[name]
    raise InferenceContractError(f"scheduler.step is missing {name}")


def _replace_scheduler_model_output(
    args: Sequence[Any], kwargs: Mapping[str, Any], replacement: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    new_args = list(args)
    new_kwargs = dict(kwargs)
    if new_args:
        if "model_output" in new_kwargs:
            raise InferenceContractError(
                "scheduler.step received duplicate model_output"
            )
        new_args[0] = replacement
    elif "model_output" in new_kwargs:
        new_kwargs["model_output"] = replacement
    else:
        raise InferenceContractError("scheduler.step is missing model_output")
    return tuple(new_args), new_kwargs


@dataclass
class SourceTrajectoryClampTrace:
    """Tensor-free audit of a successful phase-0 source trajectory clamp."""

    expected_steps: int
    records: list[dict[str, Any]] = field(default_factory=list)
    initial_packed_noise_captured: bool = False
    finalized: bool = False

    def as_dict(self) -> dict[str, Any]:
        if not self.finalized:
            raise InferenceContractError(
                "source trajectory clamp trace is not finalized"
            )
        return {
            "schema_version": SOURCE_TRAJECTORY_CLAMP_SCHEMA,
            "policy": EVERY_STEP_SOURCE_ONSET_POLICY,
            "integrator": "original_unipc_scheduler_step",
            "prediction_type": "flow_prediction",
            "phase": 0,
            "latent_phases": LATENT_FRAME_COUNT,
            "initial_packed_noise_captured": self.initial_packed_noise_captured,
            "step_count": len(self.records),
            "expected_steps": self.expected_steps,
            "steps": list(self.records),
            "target_video_accessed": False,
            "identity_or_background_claim": False,
        }


class _InstalledSourceTrajectoryClamp:
    """Reversible instance wrapper around one native UniPC scheduler."""

    def __init__(
        self, diffusion: Any, source_latent: Any, *, expected_steps: int
    ) -> None:
        if type(expected_steps) is not int or expected_steps <= 0:
            raise InferenceContractError(
                "source trajectory clamp expected_steps must be positive"
            )
        if getattr(diffusion, "use_unipc", None) is not True:
            raise InferenceContractError(
                "source trajectory clamp requires diffusion.use_unipc=True"
            )
        scheduler = getattr(diffusion, "scheduler", None)
        if scheduler is None:
            raise InferenceContractError(
                "source trajectory clamp requires diffusion.scheduler"
            )
        if scheduler.__class__.__name__ != "UniPCMultistepScheduler":
            raise InferenceContractError(
                "source trajectory clamp requires UniPCMultistepScheduler"
            )
        config = getattr(scheduler, "config", None)
        if config is None:
            raise InferenceContractError(
                "source trajectory clamp requires a scheduler config"
            )
        required = {
            "_class_name": "UniPCMultistepScheduler",
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "predict_x0": True,
            "final_sigmas_type": "zero",
        }
        for name, expected in required.items():
            actual = _scheduler_config_value(config, name)
            if type(expected) is bool:
                matches = actual is expected
            else:
                matches = actual == expected
            if not matches:
                raise InferenceContractError(
                    f"source trajectory clamp scheduler {name} differs: "
                    f"expected {expected!r}, got {actual!r}"
                )
        flow_shift = _scalar_float(
            _scheduler_config_value(config, "flow_shift"),
            label="scheduler flow_shift",
        )
        if flow_shift != FLOW_SHIFT:
            raise InferenceContractError(
                f"source trajectory clamp requires flow_shift={FLOW_SHIFT}, got {flow_shift}"
            )

        self.diffusion = diffusion
        self.scheduler = scheduler
        self.source_packed = _pack_wan_source_latent(source_latent)
        self.phase_tokens = int(self.source_packed.shape[1]) // LATENT_FRAME_COUNT
        self.expected_steps = expected_steps
        self.trace = SourceTrajectoryClampTrace(expected_steps=expected_steps)
        self._epsilon_phase0: Any = None
        self._runtime_sigmas: Optional[tuple[float, ...]] = None
        try:
            instance_dict = vars(scheduler)
        except TypeError as error:
            raise InferenceContractError(
                "scheduler must permit a reversible instance step wrapper"
            ) from error
        self._had_instance_step = "step" in instance_dict
        self._old_instance_step = instance_dict.get("step")
        self._original_step = getattr(scheduler, "step", None)
        if not callable(self._original_step):
            raise InferenceContractError("scheduler.step must be callable")
        if getattr(
            self._original_step, "_bernini_source_phase0_trajectory_clamp", False
        ):
            raise InferenceContractError(
                "scheduler.step already has a source trajectory clamp"
            )
        self._installed = False

    def _audit_runtime_schedule(self) -> tuple[float, ...]:
        if self._runtime_sigmas is not None:
            return self._runtime_sigmas
        sigmas = getattr(self.scheduler, "sigmas", None)
        if sigmas is None:
            raise InferenceContractError(
                "UniPC must expose runtime sigmas before its first step"
            )
        try:
            values = tuple(
                _scalar_float(value, label=f"scheduler sigma {index}")
                for index, value in enumerate(sigmas)
            )
        except TypeError as error:
            raise InferenceContractError("scheduler.sigmas must be iterable") from error
        if len(values) != self.expected_steps + 1:
            raise InferenceContractError(
                "source trajectory clamp runtime sigma count differs: "
                f"{len(values)} != {self.expected_steps + 1}"
            )
        if (
            values[0] <= 0.0
            or values[-1] != 0.0
            or math.copysign(1.0, values[-1]) != 1.0
            or any(right >= left for left, right in zip(values, values[1:]))
        ):
            raise InferenceContractError(
                "source trajectory clamp requires strictly descending sigmas ending in +0"
            )
        self._runtime_sigmas = values
        return values

    def _validate_step_timestep(self, index: int, timestep: Any) -> float:
        timesteps = getattr(self.scheduler, "timesteps", None)
        if timesteps is None:
            raise InferenceContractError(
                "UniPC must expose runtime timesteps before its first step"
            )
        try:
            if len(timesteps) != self.expected_steps:
                raise InferenceContractError(
                    "source trajectory clamp runtime timestep count differs"
                )
            expected = _scalar_float(
                timesteps[index], label=f"scheduler timestep {index}"
            )
        except IndexError as error:
            raise InferenceContractError(
                "source trajectory clamp timestep index is out of range"
            ) from error
        actual = _scalar_float(timestep, label="scheduler.step timestep")
        if actual != expected:
            raise InferenceContractError(
                f"scheduler.step timestep differs at {index}: {actual} != {expected}"
            )
        return actual

    def _wrapped_step(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        model_output = _extract_scheduler_step_argument(
            args, kwargs, index=0, name="model_output"
        )
        timestep = _extract_scheduler_step_argument(
            args, kwargs, index=1, name="timestep"
        )
        sample = _extract_scheduler_step_argument(
            args, kwargs, index=2, name="sample"
        )
        return_dict = _extract_scheduler_step_argument(
            args, kwargs, index=3, name="return_dict"
        )
        if return_dict is not False:
            raise InferenceContractError(
                "source trajectory clamp requires explicit return_dict=False"
            )
        if (
            not isinstance(model_output, torch.Tensor)
            or not isinstance(sample, torch.Tensor)
            or model_output.ndim != 3
            or sample.ndim != 3
            or tuple(model_output.shape) != tuple(sample.shape)
            or tuple(sample.shape) != tuple(self.source_packed.shape)
            or model_output.device != sample.device
            or sample.device != self.source_packed.device
            or not bool(torch.isfinite(model_output).all().item())
            or not bool(torch.isfinite(sample).all().item())
        ):
            raise InferenceContractError(
                "source trajectory clamp requires matching finite packed [B,N,64] tensors"
            )

        sigmas = self._audit_runtime_schedule()
        index = len(self.trace.records)
        if index >= self.expected_steps:
            raise InferenceContractError(
                "source trajectory clamp observed too many scheduler steps"
            )
        cursor = getattr(self.scheduler, "step_index", None)
        if cursor is None:
            cursor = getattr(self.scheduler, "_step_index", None)
        if index == 0:
            if cursor is not None:
                raise InferenceContractError(
                    "source trajectory clamp requires a fresh UniPC cursor"
                )
        elif cursor is None or int(cursor) != index:
            raise InferenceContractError(
                "source trajectory clamp scheduler cursor differs before step"
            )
        timestep_float = self._validate_step_timestep(index, timestep)

        source_phase0 = self.source_packed[:, : self.phase_tokens, :]
        sample_phase0 = sample[:, : self.phase_tokens, :]
        if self._epsilon_phase0 is None:
            # Bernini initializes the packed target state from one noise draw.
            # Capture those exact phase-0 bytes before the first native step;
            # no new RNG call is introduced by this policy.
            self._epsilon_phase0 = sample_phase0.detach().clone()
            self.trace.initial_packed_noise_captured = True
        epsilon_phase0 = self._epsilon_phase0
        if (
            tuple(epsilon_phase0.shape) != tuple(source_phase0.shape)
            or epsilon_phase0.device != sample.device
        ):
            raise InferenceContractError(
                "captured phase-0 noise no longer matches the sampler state"
            )

        forced_velocity = model_output.clone()
        forced_velocity[:, : self.phase_tokens, :] = (
            epsilon_phase0.float() - source_phase0.float()
        ).to(dtype=model_output.dtype)
        if not bool(torch.isfinite(forced_velocity).all().item()):
            raise InferenceContractError(
                "source trajectory clamp produced non-finite flow velocity"
            )
        call_args, call_kwargs = _replace_scheduler_model_output(
            args, kwargs, forced_velocity
        )
        result = self._original_step(*call_args, **call_kwargs)
        if type(result) is not tuple or len(result) != 1:
            raise InferenceContractError(
                "return_dict=False UniPC result must be one built-in tuple"
            )
        previous = result[0]
        if (
            not isinstance(previous, torch.Tensor)
            or tuple(previous.shape) != tuple(sample.shape)
            or previous.device != sample.device
            or not bool(torch.isfinite(previous).all().item())
        ):
            raise InferenceContractError(
                "UniPC previous sample differs from the packed sampler state"
            )
        cursor_after = getattr(self.scheduler, "step_index", None)
        if cursor_after is None:
            cursor_after = getattr(self.scheduler, "_step_index", None)
        if cursor_after is None or int(cursor_after) != index + 1:
            raise InferenceContractError(
                "UniPC scheduler cursor did not advance exactly once"
            )

        next_sigma = sigmas[index + 1]
        projected_previous = previous.clone()
        projected_previous[:, : self.phase_tokens, :] = (
            (1.0 - next_sigma) * source_phase0.float()
            + next_sigma * epsilon_phase0.float()
        ).to(dtype=previous.dtype)
        self.trace.records.append(
            {
                "step_index": index,
                "timestep": timestep_float,
                "sigma": sigmas[index],
                "next_sigma": next_sigma,
                "phase0_velocity": "captured_epsilon_minus_clean_source",
                "phase0_post_step": "source_noise_flow_trajectory_projection",
                "other_phases_projected": False,
                "original_scheduler_step_calls": 1,
            }
        )
        return (projected_previous,)

    def install(self) -> None:
        if self._installed:
            raise InferenceContractError("source trajectory clamp is already installed")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_step(*args, **kwargs)

        setattr(wrapped, "_bernini_source_phase0_trajectory_clamp", True)
        setattr(self.scheduler, "step", wrapped)
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        if self._had_instance_step:
            setattr(self.scheduler, "step", self._old_instance_step)
        else:
            delattr(self.scheduler, "step")
        self._installed = False

    def finalize(self) -> None:
        sigmas = self._audit_runtime_schedule()
        if (
            len(self.trace.records) != self.expected_steps
            or not self.trace.initial_packed_noise_captured
            or sigmas[-1] != 0.0
            or self.trace.records[-1]["next_sigma"] != 0.0
        ):
            raise InferenceContractError(
                "source trajectory clamp did not complete the full terminal-zero schedule"
            )
        self.trace.finalized = True


@contextmanager
def hard_phase0_source_trajectory_clamp(
    diffusion: Any, source_latent: Any, *, expected_steps: int
) -> Any:
    """Clamp phase zero to one fixed source/noise flow path at every UniPC step."""

    installed = _InstalledSourceTrajectoryClamp(
        diffusion, source_latent, expected_steps=expected_steps
    )
    installed.install()
    try:
        yield installed.trace
        installed.finalize()
    finally:
        installed.restore()


def inference_renderer_config_overrides(checkpoint: Path) -> dict[str, Any]:
    """Return the training mapping with the sampler's actual mv2v shift."""

    overrides = trainer.renderer_config_overrides(checkpoint)
    overrides.update(
        {
            "shift": FLOW_SHIFT,
            "use_unipc": True,
            "switch_dit_boundary": 0.0,
        }
    )
    return overrides


def tokenizer_load_kwargs() -> dict[str, Any]:
    return {
        "padding_side": "right",
        "trust_remote_code": True,
        "local_files_only": True,
        "fix_mistral_regex": True,
    }


def sampler_contract(*, steps: int, seed: int) -> dict[str, Any]:
    return {
        "num_frames": FRAME_COUNT,
        "num_inference_steps": int(steps),
        "guidance_mode": GUIDANCE_MODE,
        "omega_vid": 1.25,
        "omega_img": 0.0,
        "omega_txt": OMEGA_TEXT,
        "omega_scale": 0.8,
        "flow_shift": FLOW_SHIFT,
        "seed": int(seed),
        "eta": ETA,
        "norm_threshold": NORM_THRESHOLD,
        "momentum": MOMENTUM,
    }


def build_training_prompt(
    instruction: str, *, prompt_cleaner: Callable[[str], str]
) -> str:
    if not isinstance(instruction, str) or not instruction.strip() or "\x00" in instruction:
        raise InferenceContractError("edit instruction must be non-empty text without NUL")
    cleaned = prompt_cleaner(instruction)
    if not isinstance(cleaned, str) or not cleaned.strip():
        raise InferenceContractError("Wan prompt cleaner produced an empty instruction")
    # This intentionally has no inserted delimiter: it is byte-for-byte the
    # concatenation in bernini.training.data.encode_renderer_messages.
    return MV2V_SYSTEM_PROMPT + cleaned


def validate_exact_video_metadata(frame_count: Any, fps: Any) -> None:
    if type(frame_count) is not int or frame_count != FRAME_COUNT:
        raise InferenceContractError(
            f"source video must decode to exactly {FRAME_COUNT} frames, got {frame_count!r}"
        )
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise InferenceContractError("source video fps is not numeric")
    value = float(fps)
    if not math.isfinite(value) or abs(value - FPS) > 1e-3:
        raise InferenceContractError(f"source video must report {FPS} fps, got {fps!r}")


def inference_distributed_contract(
    environment: Mapping[str, str] = os.environ,
) -> InferenceDistributedContract:
    try:
        world = int(environment.get("WORLD_SIZE", ""))
        rank = int(environment.get("RANK", ""))
        local_rank = int(environment.get("LOCAL_RANK", ""))
    except ValueError as error:
        raise InferenceContractError("invalid torchrun rank environment") from error
    if world != ULYSSES_SIZE:
        raise InferenceContractError(
            f"inference requires exactly {ULYSSES_SIZE} ranks/Ulysses shards, got {world}"
        )
    if not 0 <= rank < world or not 0 <= local_rank < world:
        raise InferenceContractError(
            f"invalid torchrun ranks: rank={rank}, local_rank={local_rank}, world={world}"
        )
    return InferenceDistributedContract(world, rank, local_rank)


def resolve_adapter_bundle(value: str | Path) -> AdapterBundle:
    requested = _absolute_directory(value, label="adapter checkpoint")
    if (requested / "adapter").is_dir():
        checkpoint_root = requested
        adapter_dir = requested / "adapter"
    elif requested.name == "adapter" and (requested.parent / "receipt.json").is_file():
        checkpoint_root = requested.parent
        adapter_dir = requested
    else:
        raise InferenceContractError(
            "adapter must be a training checkpoint root containing adapter/ and receipt.json, "
            "or that exact adapter/ directory"
        )
    adapter_dir = _absolute_directory(
        adapter_dir, label="adapter checkpoint directory"
    )
    config = _plain_file(adapter_dir / "adapter_config.json", label="adapter config")
    model = _plain_file(adapter_dir / "adapter_model.safetensors", label="adapter weights")
    receipt = _plain_file(checkpoint_root / "receipt.json", label="training receipt")
    return AdapterBundle(checkpoint_root, adapter_dir, config, model, receipt)


def validate_training_checkpoint_manifest(
    adapter: AdapterBundle,
    *,
    expected_sha256: str,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the terminal full644 checkpoint's complete physical closure."""

    if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise InferenceContractError("checkpoint manifest expected SHA differs")
    manifest_requested = (
        adapter.checkpoint_root / "checkpoint_manifest.json"
        if manifest_path is None
        else Path(manifest_path).expanduser()
    )
    if not manifest_requested.is_absolute():
        raise InferenceContractError("checkpoint manifest path must be absolute")
    manifest_path = _plain_file(
        manifest_requested,
        label="checkpoint manifest",
    )
    physical_root = manifest_path.parent
    manifest_raw = _stable_plain_file_bytes(
        manifest_path, label="checkpoint manifest"
    )
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha256 != expected_sha256:
        raise InferenceContractError("checkpoint manifest SHA differs")
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InferenceContractError("checkpoint manifest is not UTF-8 JSON") from error
    if not isinstance(manifest, dict):
        raise InferenceContractError("checkpoint manifest root differs")
    unsigned = dict(manifest)
    declared_digest = unsigned.pop("manifest_digest", None)
    if (
        manifest.get("schema_version")
        != "bernini-r-action-lora-checkpoint-manifest-v1"
        or manifest.get("global_step") != trainer.FULL644_EXPLORATORY_STEPS
        or not isinstance(declared_digest, str)
        or declared_digest != object_sha256(unsigned)
    ):
        raise InferenceContractError("checkpoint manifest semantic seal differs")
    entries = manifest.get("entries")
    if (
        not isinstance(entries, list)
        or manifest.get("file_count") != len(entries)
        or not entries
    ):
        raise InferenceContractError("checkpoint manifest entry count differs")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in entries:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
            raise InferenceContractError("checkpoint manifest row schema differs")
        relative = row.get("path")
        sha256 = row.get("sha256")
        size = row.get("size")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in seen
            or not isinstance(sha256, str)
            or _SHA256_RE.fullmatch(sha256) is None
            or type(size) is not int
            or size <= 0
        ):
            raise InferenceContractError("checkpoint manifest row value differs")
        consumed_paths = {
            "adapter/adapter_config.json": adapter.adapter_config_path,
            "adapter/adapter_model.safetensors": adapter.adapter_model_path,
            "receipt.json": adapter.training_receipt_path,
        }
        path = consumed_paths.get(relative, physical_root / relative)
        raw = _stable_plain_file_bytes(path, label=f"checkpoint member {relative}")
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha256:
            raise InferenceContractError(
                f"checkpoint manifest member bytes differ: {relative}"
            )
        seen.add(relative)
        normalized.append({"path": relative, "sha256": sha256, "size": size})
    if normalized != sorted(normalized, key=lambda row: row["path"]):
        raise InferenceContractError("checkpoint manifest rows are not sorted")
    physical: list[str] = []
    for path in sorted(physical_root.rglob("*")):
        relative = path.relative_to(physical_root).as_posix()
        if path.is_symlink():
            raise InferenceContractError(
                f"checkpoint physical closure contains a symlink: {relative}"
            )
        if path.is_file() and relative != "checkpoint_manifest.json":
            physical.append(relative)
    if physical != [row["path"] for row in normalized]:
        raise InferenceContractError("checkpoint physical closure differs from manifest")
    required = {
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "optimizer.pt",
        "receipt.json",
    }
    if not required.issubset(seen):
        raise InferenceContractError("checkpoint manifest lacks required payloads")
    by_path = {row["path"]: row for row in normalized}
    try:
        manifest_receipt = json.loads(
            _stable_plain_file_bytes(
                adapter.training_receipt_path, label="manifest training receipt"
            ).decode("utf-8")
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InferenceContractError(
            "manifest training receipt is not UTF-8 JSON"
        ) from error
    if (
        not isinstance(manifest_receipt, dict)
        or manifest.get("receipt_digest") != manifest_receipt.get("receipt_digest")
    ):
        raise InferenceContractError("checkpoint manifest receipt digest differs")
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "manifest_digest": declared_digest,
        "global_step": manifest["global_step"],
        "receipt_digest": manifest["receipt_digest"],
        "file_count": manifest["file_count"],
        "adapter_config_sha256": by_path["adapter/adapter_config.json"]["sha256"],
        "adapter_model_sha256": by_path["adapter/adapter_model.safetensors"]["sha256"],
        "training_receipt_sha256": by_path["receipt.json"]["sha256"],
        "optimizer_sha256": by_path["optimizer.pt"]["sha256"],
    }


def validate_inference_checkpoint(
    checkpoint: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Validate the exact model without resolving an inherited proc-FD root."""

    requested = Path(checkpoint)
    if str(requested) not in _AUTHORIZED_FD_VIEW_DIRECTORIES:
        try:
            return trainer.validate_checkpoint(checkpoint)
        except trainer.TrainingContractError as error:
            raise InferenceContractError(str(error)) from error
    root = _absolute_directory(requested, label="authorized checkpoint")
    for name in ("transformer", "text_encoder", "tokenizer", "vae", "scheduler"):
        _absolute_directory(root / name, label=f"checkpoint {name}")
    if (root / "transformer_2").exists():
        raise InferenceContractError(
            "checkpoint contains forbidden transformer_2"
        )
    transformer_config = _read_json(
        root / "transformer/config.json", label="transformer config"
    )
    expected = {
        "num_layers": 30,
        "num_attention_heads": 12,
        "attention_head_dim": 128,
        "in_channels": 16,
        "out_channels": 16,
    }
    if any(transformer_config.get(key) != value for key, value in expected.items()):
        raise InferenceContractError(
            "authorized checkpoint transformer semantic contract differs"
        )
    weights = tuple((root / "transformer").glob("*.safetensors"))
    indices = tuple((root / "transformer").glob("*.safetensors.index.json"))
    if not weights and not indices:
        raise InferenceContractError(
            "authorized checkpoint transformer has no local weights"
        )
    return root, transformer_config


def _finite_number(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise InferenceContractError(f"{label} must be {qualifier}")
    return result


def _strict_json_scalar_equal(value: Any, expected: Any) -> bool:
    """Compare receipt scalars without Python's bool/int numeric aliasing."""

    return type(value) is type(expected) and value == expected


def _strict_float(value: Any, expected: float) -> bool:
    return type(value) is float and math.isfinite(value) and value == expected


def _load_preservation_v2_contract() -> Any:
    """Load the v2 selector only for v2 receipts, preserving legacy closure."""

    try:
        import self_generated_action_preservation_v2 as module
    except ImportError as error:
        raise InferenceContractError(
            "preservation-v2 receipt requires self_generated_action_preservation_v2.py"
        ) from error
    if (
        module.SCHEMA != V2_OBJECTIVE
        or tuple(module.ROUTE_SCOPES) != tuple(V2_PEFT_COMPACT_TARGET_MODULES)
    ):
        raise InferenceContractError(
            "preservation-v2 selector constants differ from the inference contract"
        )
    return module


def expected_v2_lora_target_modules(scope: str) -> list[str]:
    """Rebuild one v2 route only from the audited full Wan registry."""

    preservation_v2 = _load_preservation_v2_contract()
    try:
        selected = preservation_v2.select_projection_scope(
            expected_lora_target_modules(), scope=scope
        )
    except preservation_v2.ActionPreservationV2Error as error:
        raise InferenceContractError(f"invalid preservation-v2 route scope: {error}") from error
    return list(selected)


def _validate_v2_peft_config(
    adapter_config: Mapping[str, Any], *, route_scope: str
) -> None:
    targets = adapter_config.get("target_modules")
    canonical = V2_PEFT_COMPACT_TARGET_MODULES.get(route_scope)
    if (
        not isinstance(targets, list)
        or not all(isinstance(item, str) for item in targets)
        or len(targets) != len(set(targets))
        or canonical is None
        or set(targets) != set(canonical)
    ):
        raise InferenceContractError(
            "preservation-v2 adapter target_modules are not the canonical "
            f"PEFT serialization for {route_scope}"
        )
    if adapter_config.get("peft_type") != "LORA":
        raise InferenceContractError("adapter peft_type must be LORA")
    if adapter_config.get("r") != trainer.LORA_RANK:
        raise InferenceContractError("adapter LoRA rank must be 8")
    if (
        _finite_number(
            adapter_config.get("lora_alpha"), label="adapter lora_alpha"
        )
        != trainer.LORA_ALPHA
    ):
        raise InferenceContractError("adapter lora_alpha must be 8")
    if (
        _finite_number(
            adapter_config.get("lora_dropout"), label="adapter lora_dropout"
        )
        != 0.0
    ):
        raise InferenceContractError("adapter lora_dropout must be zero")
    if adapter_config.get("bias") != "none":
        raise InferenceContractError("adapter bias must be none")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise InferenceContractError("adapter may not contain modules_to_save")
    if adapter_config.get("use_dora") not in (None, False):
        raise InferenceContractError("DoRA adapters are outside the training contract")
    if adapter_config.get("use_rslora") not in (None, False):
        raise InferenceContractError("RS-LoRA adapters are outside the training contract")
def _validate_v2_adapter_contract(
    adapter_config: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    """Validate a signed preservation-v2 receipt and derive its exact route."""

    receipt = dict(training_receipt)
    stored_digest = receipt.pop("receipt_digest", None)
    if not isinstance(stored_digest, str) or not _SHA256_RE.fullmatch(stored_digest):
        raise InferenceContractError("training receipt has no valid receipt_digest")
    if object_sha256(receipt) != stored_digest:
        raise InferenceContractError("training receipt digest mismatch")
    if (
        training_receipt.get("schema_version")
        != V2_TRAINING_RECEIPT_SCHEMA
    ):
        raise InferenceContractError("preservation-v2 training receipt schema mismatch")
    if training_receipt.get("bernini_commit") != trainer.BERNINI_OFFICIAL_COMMIT:
        raise InferenceContractError("training receipt Bernini commit mismatch")
    if training_receipt.get("veomni_commit") != trainer.VEOMNI_TESTED_COMMIT:
        raise InferenceContractError("training receipt VeOmni commit mismatch")
    if training_receipt.get("checkpoint_tree_sha256") != expected_checkpoint_tree_sha256:
        raise InferenceContractError("training receipt checkpoint tree mismatch")
    if training_receipt.get("bernini_training_files_index_sha256") != object_sha256(
        trainer.BERNINI_PINNED_FILE_HASHES
    ):
        raise InferenceContractError("training receipt Bernini file index mismatch")

    contract = training_receipt.get("training_contract")
    if not isinstance(contract, dict):
        raise InferenceContractError("training receipt lacks training_contract")
    arm = contract.get("arm")
    if not isinstance(arm, str):
        raise InferenceContractError(
            "preservation-v2 training receipt arm must be a string"
        )
    preservation_v2 = _load_preservation_v2_contract()
    try:
        spec = preservation_v2.arm_spec(arm)
    except preservation_v2.ActionPreservationV2Error as error:
        raise InferenceContractError(
            f"preservation-v2 training receipt arm mismatch: {arm!r}"
        ) from error
    route_scope = contract.get("lora_route_scope")
    if route_scope != spec.route_scope:
        raise InferenceContractError(
            "preservation-v2 receipt arm/route scope mismatch"
        )
    expected_targets = expected_v2_lora_target_modules(route_scope)
    explicit_targets = training_receipt.get("target_modules")
    if explicit_targets != expected_targets:
        raise InferenceContractError(
            "preservation-v2 explicit target_modules differ from reconstructed route"
        )
    if training_receipt.get("target_module_count") != len(expected_targets):
        raise InferenceContractError(
            "preservation-v2 training receipt target_module_count mismatch"
        )
    expected_target_digest = object_sha256(expected_targets)
    if training_receipt.get("target_modules_sha256") != expected_target_digest:
        raise InferenceContractError(
            "preservation-v2 training receipt target module digest mismatch"
        )
    _validate_v2_peft_config(adapter_config, route_scope=route_scope)

    expected_contract = {
        "model": "Bernini-R-1.3B-Diffusers renderer-only",
        "single_expert": "transformer_1",
        "mv2v_flow_shift": FLOW_SHIFT,
        "num_frames": FRAME_COUNT,
        "latent_frames": LATENT_FRAME_COUNT,
        "task_source_name": trainer.TASK_SOURCE_NAME,
        "external_spatial_mask": False,
        "external_tracking_or_swept_tube": False,
        "conditioning": ["clean_source_video_vae", "edit_instruction"],
        "target_embedding_or_caption_conditioning": False,
        "lora_rank": trainer.LORA_RANK,
        "lora_alpha": trainer.LORA_ALPHA,
        "tokenizer_fix_mistral_regex": True,
        "objective": V2_OBJECTIVE,
        "objective_family": "preservation_v2",
        "arm": arm,
        "weights": {
            "noop": spec.noop_weight,
            "onset": spec.onset_weight,
            "nuisance": spec.nuisance_weight,
            "functional": spec.functional_weight,
        },
        "onset_latent_phase_weights": list(preservation_v2.ONSET_WEIGHTS),
        "functional_components": [
            "teacher_direction_exempt_post_head_orthogonal_drift",
            "post_onset_temporal_dc_clean_latent_drift",
        ],
        "lora_route_scope": route_scope,
        "lora_route_scope_semantics": (
            "observable Wan attention topology only; no temporal-only or "
            "source-only route claim"
        ),
        "sigma_bins": [list(item) for item in V2_SIGMA_BINS],
        "checkpoint_updates": list(V2_SAVE_STEPS),
        "rv2v_supervision_target": "source_video_only",
        "self_generated_anchor_role": (
            "detached_post_head_action_phase_code_only"
        ),
        "historical_selected_target_reachable": False,
        "decoded_identity_background_camera_claim_authorized": False,
        "post_decode_gate_schema": "bernini-action-preservation-decision-v1",
        "blind_full_video_review_required_for_promotion": True,
    }
    for key, wanted in expected_contract.items():
        if contract.get(key) != wanted:
            raise InferenceContractError(
                f"preservation-v2 training receipt contract mismatch for {key}: "
                f"{contract.get(key)!r}"
            )
    runtime_transformers = contract.get("transformers_version")
    if not isinstance(runtime_transformers, str) or not runtime_transformers:
        raise InferenceContractError("training receipt lacks Transformers version")

    global_step = training_receipt.get("global_step")
    if (
        type(global_step) is not int
        or global_step not in V2_SAVE_STEPS
    ):
        raise InferenceContractError(
            "preservation-v2 global_step is not a registered canary checkpoint"
        )
    if training_receipt.get("max_steps") != 20:
        raise InferenceContractError("preservation-v2 max_steps must be 20")
    last_loss = _finite_number(
        training_receipt.get("last_loss"), label="last training loss"
    )
    grad_norm = _finite_number(
        training_receipt.get("last_preclip_gradient_norm"),
        label="last preclip gradient norm",
        positive=global_step > 0,
    )
    component_names = {
        "action",
        "onset",
        "nuisance",
        "noop",
        "functional_code",
        "functional_temporal_dc",
        "functional_total",
    }
    components = training_receipt.get("last_loss_components")
    if not isinstance(components, dict) or set(components) != component_names:
        raise InferenceContractError(
            "preservation-v2 loss component closure mismatch"
        )
    normalized_components = {
        key: _finite_number(value, label=f"last loss component {key}")
        for key, value in components.items()
    }
    if last_loss < 0.0 or any(
        value < 0.0 for value in normalized_components.values()
    ):
        raise InferenceContractError(
            "preservation-v2 losses must be non-negative"
        )
    frozen_baseline = global_step == 0
    if frozen_baseline and (
        last_loss != 0.0
        or grad_norm != 0.0
        or any(value != 0.0 for value in normalized_components.values())
    ):
        raise InferenceContractError(
            "preservation-v2 checkpoint 0 must be an untouched zero-loss/zero-grad baseline"
        )

    if training_receipt.get("objective_family") != "preservation_v2":
        raise InferenceContractError("preservation-v2 objective_family mismatch")
    if training_receipt.get("initialization_seed") != V2_CANARY_SEED:
        raise InferenceContractError("preservation-v2 initialization seed mismatch")
    if training_receipt.get("teacher_cache_seed") != V2_CANARY_SEED:
        raise InferenceContractError("preservation-v2 teacher cache seed mismatch")
    optimizer = training_receipt.get("optimizer")
    if optimizer != {
        "type": "AdamW",
        "learning_rate": spec.learning_rate,
        "weight_decay": 0.0,
    }:
        raise InferenceContractError("preservation-v2 optimizer contract mismatch")
    trainable_count = training_receipt.get("trainable_parameter_count")
    if type(trainable_count) is not int or trainable_count <= 0:
        raise InferenceContractError(
            "preservation-v2 trainable_parameter_count must be positive"
        )

    distributed = training_receipt.get("distributed")
    if not isinstance(distributed, dict):
        raise InferenceContractError("preservation-v2 distributed contract is missing")
    expected_distributed = {
        "world_size": ULYSSES_SIZE,
        "ulysses_size": ULYSSES_SIZE,
        "backend": "nccl/rccl",
        "same_sample_all_ranks": True,
        "same_seed_all_ranks": True,
        "explicit_lora_gradient_all_reduce": True,
    }
    for key, wanted in expected_distributed.items():
        if distributed.get(key) != wanted:
            raise InferenceContractError(
                f"preservation-v2 distributed contract mismatch for {key}"
            )
    sha_fields = (
        (distributed.get("lora_initialization_digest"), "LoRA initialization digest"),
        (training_receipt.get("source_manifest_digest"), "source manifest digest"),
        (training_receipt.get("source_manifest_sha256"), "source manifest SHA-256"),
        (training_receipt.get("teacher_cache_sha256"), "teacher cache SHA-256"),
    )
    for value, label in sha_fields:
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise InferenceContractError(f"preservation-v2 receipt has invalid {label}")
    if training_receipt.get("decoded_preservation_evidence_present") is not False:
        raise InferenceContractError(
            "preservation-v2 receipt cannot claim decoded preservation evidence"
        )
    if training_receipt.get("automatic_scientific_promotion_authorized") is not False:
        raise InferenceContractError(
            "preservation-v2 receipt cannot authorize automatic scientific promotion"
        )
    if training_receipt.get("experimental_training") is not True:
        raise InferenceContractError("preservation-v2 receipt lost experimental status")
    if training_receipt.get("production_claim_forbidden") is not True:
        raise InferenceContractError("training receipt lost production-claim restriction")
    if training_receipt.get("scientific_claim_authorized") is not False:
        raise InferenceContractError("training receipt has an unsupported scientific claim")
    for key in ("method_source_revision", "method_source_archive_sha256"):
        value = training_receipt.get(key)
        pattern = _SHA1_RE if key.endswith("revision") else _SHA256_RE
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise InferenceContractError(f"training receipt has invalid {key}")
    return {
        "global_step": global_step,
        "receipt_digest": stored_digest,
        "target_modules_sha256": expected_target_digest,
        "target_modules": expected_targets,
        "lora_route_scope": route_scope,
        "frozen_baseline": frozen_baseline,
        "transformers_version": runtime_transformers,
        "method_source_revision": training_receipt["method_source_revision"],
        "method_source_archive_sha256": training_receipt[
            "method_source_archive_sha256"
        ],
    }


def validate_adapter_contract(
    adapter_config: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Validate the closed PEFT/training receipt before model construction."""

    if (
        training_receipt.get("schema_version")
        == V2_TRAINING_RECEIPT_SCHEMA
    ):
        return _validate_v2_adapter_contract(
            adapter_config,
            training_receipt,
            expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
        )
    if (
        training_receipt.get("schema_version")
        == V4_FULLFIELD_TRAINING_RECEIPT_SCHEMA
    ):
        return _validate_v4_fullfield_adapter_contract(
            adapter_config,
            training_receipt,
            expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
        )

    full644_contract = training_receipt.get("exploratory_full644")
    if full644_contract is not None and (
        not isinstance(full644_contract, Mapping)
        or full644_contract.get("profile") != trainer.FULL644_EXPLORATORY_PROFILE
    ):
        raise InferenceContractError("full644 exploratory adapter profile differs")
    is_full644 = full644_contract is not None
    expected_rank = trainer.FULL644_EXPLORATORY_RANK if is_full644 else trainer.LORA_RANK
    expected_alpha = (
        trainer.FULL644_EXPLORATORY_ALPHA if is_full644 else trainer.LORA_ALPHA
    )
    expected_targets = expected_lora_target_modules()
    targets = adapter_config.get("target_modules")
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise InferenceContractError("adapter target_modules must be an explicit string list")
    serialized_targets = set(targets)
    # PEFT 0.19 deliberately compacts a full list of matching qualified names
    # to the smallest common suffix set when it writes adapter_config.json.
    # Accept only that exact canonical four-suffix form (or an un-compacted
    # exact list), then override it with all 240 qualified Wan names before
    # constructing PeftModel below.  The receipt and tensor-key checks retain
    # the complete scope identity.
    if len(targets) != len(serialized_targets) or serialized_targets not in (
        set(expected_targets),
        set(PEFT_COMPACT_TARGET_MODULES),
    ):
        raise InferenceContractError(
            "adapter target_modules differ from the audited Wan projection scope"
        )
    if adapter_config.get("peft_type") != "LORA":
        raise InferenceContractError("adapter peft_type must be LORA")
    if adapter_config.get("r") != expected_rank:
        raise InferenceContractError(f"adapter LoRA rank must be {expected_rank}")
    if (
        _finite_number(adapter_config.get("lora_alpha"), label="adapter lora_alpha")
        != expected_alpha
    ):
        raise InferenceContractError(f"adapter lora_alpha must be {expected_alpha}")
    if _finite_number(adapter_config.get("lora_dropout"), label="adapter lora_dropout") != 0.0:
        raise InferenceContractError("adapter lora_dropout must be zero")
    if adapter_config.get("bias") != "none":
        raise InferenceContractError("adapter bias must be none")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise InferenceContractError("adapter may not contain modules_to_save")
    if adapter_config.get("use_dora") not in (None, False):
        raise InferenceContractError("DoRA adapters are outside the training contract")
    if adapter_config.get("use_rslora") not in (None, False):
        raise InferenceContractError("RS-LoRA adapters are outside the training contract")

    if is_full644:
        # This is the complete PEFT 0.19.1 LoraConfig serialization emitted by
        # the audited BerniniRendererModel path.  Exact field closure matters:
        # optional aLoRA/Arrow/BD-LoRA/LoRA-GA/LoftQ/PiSSA/weight-tying/runtime
        # values can alter application or even base weights while preserving
        # the same tensor keys and shapes.  Only target_modules is checked
        # separately above because PEFT may serialize either the audited full
        # 240-name scope or its exact four-suffix compaction.
        if set(adapter_config) != FULL644_PEFT_CONFIG_FIELDS:
            raise InferenceContractError(
                "full644 adapter PEFT 0.19.1 field closure differs"
            )
        expected_peft_without_targets = {
            "alora_invocation_tokens": None,
            "alpha_pattern": {},
            "arrow_config": None,
            "auto_mapping": {
                "base_model_class": "BerniniRendererModel",
                "parent_library": "bernini.models.renderer",
            },
            "base_model_name_or_path": "",
            "bias": "none",
            "corda_config": None,
            "ensure_weight_tying": False,
            "eva_config": None,
            "exclude_modules": None,
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layer_replication": None,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": expected_alpha,
            "lora_bias": False,
            "lora_dropout": 0.0,
            "lora_ga_config": None,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": None,
            "peft_type": "LORA",
            "peft_version": FULL644_PEFT_VERSION,
            "qalora_group_size": 16,
            "r": expected_rank,
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
        observed_peft_without_targets = dict(adapter_config)
        observed_peft_without_targets.pop("target_modules")
        if canonical_json_bytes(observed_peft_without_targets) != canonical_json_bytes(
            expected_peft_without_targets
        ):
            raise InferenceContractError(
                "full644 adapter PEFT 0.19.1 semantic closure differs"
            )

    receipt = dict(training_receipt)
    stored_digest = receipt.pop("receipt_digest", None)
    if not isinstance(stored_digest, str) or not _SHA256_RE.fullmatch(stored_digest):
        raise InferenceContractError("training receipt has no valid receipt_digest")
    if object_sha256(receipt) != stored_digest:
        raise InferenceContractError("training receipt digest mismatch")
    if training_receipt.get("schema_version") != trainer.RECEIPT_SCHEMA:
        raise InferenceContractError("training receipt schema mismatch")
    if training_receipt.get("bernini_commit") != trainer.BERNINI_OFFICIAL_COMMIT:
        raise InferenceContractError("training receipt Bernini commit mismatch")
    if training_receipt.get("veomni_commit") != trainer.VEOMNI_TESTED_COMMIT:
        raise InferenceContractError("training receipt VeOmni commit mismatch")
    if training_receipt.get("checkpoint_tree_sha256") != expected_checkpoint_tree_sha256:
        raise InferenceContractError("training receipt checkpoint tree mismatch")
    if training_receipt.get("bernini_training_files_index_sha256") != object_sha256(
        trainer.BERNINI_PINNED_FILE_HASHES
    ):
        raise InferenceContractError("training receipt Bernini file index mismatch")
    global_step = training_receipt.get("global_step")
    if type(global_step) is not int or global_step <= 0:
        raise InferenceContractError("training receipt global_step must be positive")
    if is_full644 and global_step != trainer.FULL644_EXPLORATORY_STEPS:
        raise InferenceContractError("full644 adapter is not the terminal one-pass checkpoint")
    _finite_number(training_receipt.get("last_loss"), label="last training loss")
    _finite_number(
        training_receipt.get("last_preclip_gradient_norm"),
        label="last preclip gradient norm",
        positive=True,
    )
    contract = training_receipt.get("training_contract")
    if not isinstance(contract, dict):
        raise InferenceContractError("training receipt lacks training_contract")
    expected_contract = {
        "model": "Bernini-R-1.3B-Diffusers renderer-only",
        "single_expert": "transformer_1",
        "mv2v_flow_shift": FLOW_SHIFT,
        "num_frames": FRAME_COUNT,
        "latent_frames": LATENT_FRAME_COUNT,
        "task_source_name": trainer.TASK_SOURCE_NAME,
        "external_spatial_mask": False,
        "external_tracking_or_swept_tube": False,
        "conditioning": ["clean_source_video_vae", "edit_instruction"],
        "target_embedding_or_caption_conditioning": False,
        "lora_rank": expected_rank,
        "lora_alpha": expected_alpha,
        "tokenizer_fix_mistral_regex": True,
    }
    for key, wanted in expected_contract.items():
        if contract.get(key) != wanted:
            raise InferenceContractError(
                f"training receipt contract mismatch for {key}: {contract.get(key)!r}"
            )
    if is_full644:
        dataset = training_receipt.get("dataset")
        summary = dataset.get("summary") if isinstance(dataset, Mapping) else None
        source_authority = full644_contract.get("source_authority")
        optimizer = training_receipt.get("optimizer")
        distributed = training_receipt.get("distributed")
        expected_full644 = {
            "profile": trainer.FULL644_EXPLORATORY_PROFILE,
            "historical_train_debug_rows": trainer.EXPECTED_DATASET_ROWS,
            "optimizer_rows_consumed": trainer.FULL644_EXPLORATORY_STEPS,
            "next_row_index": None,
            "row_sequence_prefix": "0..643",
            "row_sequence_sha256": object_sha256(
                list(range(trainer.FULL644_EXPLORATORY_STEPS))
            ),
            "no_replacement_within_pass": True,
            "complete_one_pass": True,
            "historical_dataset_exists": True,
            "historical_optimizer_contribution_rows": trainer.EXPECTED_DATASET_ROWS,
            "runtime_data_integrity_validated": True,
            "dataset_quality_accepted_under_0817": False,
            "formal_training_dataset_authorized": False,
            "formal_heldout_contribution": 0,
            "target_scientific_qualification_complete": False,
            "matched_frozen_evaluation_required_before_claim": True,
            "resume_policy": "forbidden_for_this_profile",
            "intermediate_checkpoints_archival_only": True,
            "interrupted_run_requires_fresh_step0_restart": True,
            "dataset_summary_sha256": trainer.FULL644_DATASET_SUMMARY_SHA256,
            "dataset_summary_digest": trainer.FULL644_DATASET_SUMMARY_DIGEST,
            "dataset_index_sha256": trainer.FULL644_DATASET_INDEX_SHA256,
            "indexed_source_and_target_vae_shards_verified_before_training": True,
            "indexed_source_and_target_vae_shards_reverified_after_training": True,
        }
        if any(
            not _strict_json_scalar_equal(full644_contract.get(key), value)
            for key, value in expected_full644.items()
        ):
            raise InferenceContractError("full644 terminal one-pass receipt differs")
        if (
            not isinstance(dataset, Mapping)
            or dataset.get("rows") != trainer.FULL644_EXPLORATORY_STEPS
            or type(dataset.get("content_signature")) is not str
            or full644_contract.get("dataset_content_signature")
            != dataset.get("content_signature")
            or not isinstance(summary, Mapping)
            or summary.get("sha256") != trainer.FULL644_DATASET_SUMMARY_SHA256
            or summary.get("summary_digest") != trainer.FULL644_DATASET_SUMMARY_DIGEST
            or summary.get("index_sha256") != trainer.FULL644_DATASET_INDEX_SHA256
            or summary.get("complete") is not True
            or summary.get("materialized_rows")
            != trainer.FULL644_EXPLORATORY_STEPS
            or not isinstance(source_authority, Mapping)
            or source_authority.get("sha256")
            != trainer.FULL644_SOURCE_AUTHORITY_SHA256
            or source_authority.get("membership_rows")
            != trainer.FULL644_EXPLORATORY_STEPS
            or source_authority.get("unique_group_id")
            != trainer.FULL644_EXPLORATORY_STEPS
            or source_authority.get("unique_source_video_sha256")
            != trainer.FULL644_EXPLORATORY_STEPS
            or source_authority.get("action_family_count") != 28
            or training_receipt.get("max_steps")
            != trainer.FULL644_EXPLORATORY_STEPS
            or training_receipt.get("seed") != trainer.FULL644_EXPLORATORY_SEED
            or training_receipt.get("resumed_from") is not None
            or contract.get("objective") != "reference_dpo_preservation"
            or contract.get("contrastive_negative_schedule") != "rotate"
            or not _strict_float(contract.get("preference_weight"), 1.0)
            or not _strict_float(contract.get("preference_margin"), 0.05)
            or not _strict_float(contract.get("preference_temperature"), 20.0)
            or not _strict_float(contract.get("dpo_beta"), 10.0)
            or not _strict_float(contract.get("preservation_weight"), 0.25)
            or contract.get("preservation_branch")
            != "source_as_target_conditional_identity"
            or contract.get("peft_version") != FULL644_PEFT_VERSION
            or not isinstance(optimizer, Mapping)
            or set(optimizer)
            != {"type", "learning_rate", "weight_decay", "max_gradient_norm"}
            or optimizer.get("type") != "AdamW"
            or not _strict_float(optimizer.get("learning_rate"), 1.0e-4)
            or not _strict_float(optimizer.get("weight_decay"), 0.0)
            or not _strict_float(optimizer.get("max_gradient_norm"), 1.0)
            or not isinstance(distributed, Mapping)
            or type(distributed.get("world_size")) is not int
            or distributed.get("world_size") != 4
            or type(distributed.get("ulysses_size")) is not int
            or distributed.get("ulysses_size") != 4
            or distributed.get("backend") != "nccl/rccl"
            or distributed.get("same_sample_all_ranks") is not True
            or distributed.get("same_seed_all_ranks") is not True
            or distributed.get("lora_initialization_seeded_all_ranks") is not True
            or type(distributed.get("lora_parameters_broadcast_from_rank"))
            is not int
            or distributed.get("lora_parameters_broadcast_from_rank") != 0
            or distributed.get("explicit_lora_gradient_all_reduce") is not True
            or not isinstance(
                distributed.get("lora_initialization_digest"), str
            )
            or _SHA256_RE.fullmatch(
                distributed.get("lora_initialization_digest", "")
            )
            is None
            or type(training_receipt.get("trainable_parameter_count")) is not int
            or training_receipt.get("trainable_parameter_count")
            != trainer.FULL644_EXPLORATORY_TRAINABLE_PARAMETER_COUNT
        ):
            raise InferenceContractError("full644 adapter data/objective closure differs")
    runtime_transformers = contract.get("transformers_version")
    if not isinstance(runtime_transformers, str) or not runtime_transformers:
        raise InferenceContractError("training receipt lacks Transformers version")
    if training_receipt.get("target_module_count") != len(expected_targets):
        raise InferenceContractError("training receipt target_module_count mismatch")
    if training_receipt.get("target_modules_sha256") != object_sha256(expected_targets):
        raise InferenceContractError("training receipt target module digest mismatch")
    distributed = training_receipt.get("distributed")
    if not isinstance(distributed, dict) or distributed.get("world_size") != ULYSSES_SIZE:
        raise InferenceContractError("adapter was not trained with the four-rank contract")
    if distributed.get("ulysses_size") != ULYSSES_SIZE:
        raise InferenceContractError("adapter training Ulysses size mismatch")
    if training_receipt.get("production_claim_forbidden") is not True:
        raise InferenceContractError("training receipt lost production-claim restriction")
    if training_receipt.get("scientific_claim_authorized") is not False:
        raise InferenceContractError("training receipt has an unsupported scientific claim")
    for key in ("method_source_revision", "method_source_archive_sha256"):
        value = training_receipt.get(key)
        pattern = _SHA1_RE if key.endswith("revision") else _SHA256_RE
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise InferenceContractError(f"training receipt has invalid {key}")
    result = {
        "global_step": global_step,
        "receipt_digest": stored_digest,
        "target_modules_sha256": object_sha256(expected_targets),
        "transformers_version": runtime_transformers,
        "method_source_revision": training_receipt["method_source_revision"],
        "method_source_archive_sha256": training_receipt[
            "method_source_archive_sha256"
        ],
    }
    if is_full644:
        result.update(
            lora_rank=expected_rank,
            lora_alpha=expected_alpha,
            target_module_count=len(expected_targets),
            exploratory_full644=True,
            peft_version=FULL644_PEFT_VERSION,
        )
    return result


def _validate_v4_fullfield_adapter_contract(
    adapter_config: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    """Validate the full30/rank256 V4 adapter without weakening old schemas."""

    expected_targets = expected_lora_target_modules()
    targets = adapter_config.get("target_modules")
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise InferenceContractError(
            "V4 adapter target_modules must be an explicit string list"
        )
    serialized_targets = set(targets)
    if len(targets) != len(serialized_targets) or serialized_targets not in (
        set(expected_targets),
        set(PEFT_COMPACT_TARGET_MODULES),
    ):
        raise InferenceContractError("V4 adapter is not exact full30 q/k/v/out")
    if adapter_config.get("peft_type") != "LORA":
        raise InferenceContractError("V4 adapter peft_type must be LORA")
    if adapter_config.get("r") != V4_FULLFIELD_LORA_RANK:
        raise InferenceContractError("V4 adapter LoRA rank must be 256")
    if (
        _finite_number(adapter_config.get("lora_alpha"), label="V4 adapter lora_alpha")
        != V4_FULLFIELD_LORA_ALPHA
    ):
        raise InferenceContractError("V4 adapter lora_alpha must be 256")
    if _finite_number(
        adapter_config.get("lora_dropout"), label="V4 adapter lora_dropout"
    ) != 0.0:
        raise InferenceContractError("V4 adapter lora_dropout must be zero")
    if adapter_config.get("bias") != "none":
        raise InferenceContractError("V4 adapter bias must be none")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise InferenceContractError("V4 adapter may not contain modules_to_save")
    if adapter_config.get("use_dora") not in (None, False) or adapter_config.get(
        "use_rslora"
    ) not in (None, False):
        raise InferenceContractError("V4 forbids DoRA and RS-LoRA")

    receipt = dict(training_receipt)
    stored_digest = receipt.pop("receipt_digest", None)
    if not isinstance(stored_digest, str) or not _SHA256_RE.fullmatch(stored_digest):
        raise InferenceContractError("V4 training receipt has no valid digest")
    if object_sha256(receipt) != stored_digest:
        raise InferenceContractError("V4 training receipt digest mismatch")
    if training_receipt.get("bernini_commit") != trainer.BERNINI_OFFICIAL_COMMIT:
        raise InferenceContractError("V4 training Bernini commit mismatch")
    if training_receipt.get("veomni_commit") != trainer.VEOMNI_TESTED_COMMIT:
        raise InferenceContractError("V4 training VeOmni commit mismatch")
    if training_receipt.get("checkpoint_tree_sha256") != expected_checkpoint_tree_sha256:
        raise InferenceContractError("V4 base checkpoint identity mismatch")
    if training_receipt.get("bernini_training_files_index_sha256") != object_sha256(
        trainer.BERNINI_PINNED_FILE_HASHES
    ):
        raise InferenceContractError("V4 Bernini training file index mismatch")
    global_step = training_receipt.get("global_step")
    if type(global_step) is not int or global_step <= 0:
        raise InferenceContractError("V4 global_step must be positive")
    _finite_number(training_receipt.get("last_loss"), label="V4 last loss")
    _finite_number(
        training_receipt.get("last_preclip_gradient_norm"),
        label="V4 preclip gradient norm",
        positive=True,
    )
    contract = training_receipt.get("training_contract")
    if not isinstance(contract, dict):
        raise InferenceContractError("V4 training contract is absent")
    exact_contract = {
        "method": V4_FULLFIELD_METHOD,
        "model": "Bernini-R-1.3B-Diffusers renderer-only",
        "single_expert": "transformer_1",
        "mv2v_flow_shift": FLOW_SHIFT,
        "num_frames": FRAME_COUNT,
        "latent_frames": LATENT_FRAME_COUNT,
        "task_source_name": trainer.TASK_SOURCE_NAME,
        "conditioning": ["clean_source_video_vae", "edit_instruction"],
        "target_embedding_or_caption_conditioning": False,
        "external_spatial_mask": False,
        "external_tracking_or_swept_tube": False,
        "lora_rank": V4_FULLFIELD_LORA_RANK,
        "lora_alpha": V4_FULLFIELD_LORA_ALPHA,
        "lora_scope": "all_30_blocks_attn1_attn2_qkvo",
        "gradient_checkpointing": "selective_nonreentrant_stride4",
        "selective_checkpoint_blocks": [0, 4, 8, 12, 16, 20, 24, 28],
        "full_field_shape": "[B,16,21,H,W]",
        "frozen_rv2v_action_target": False,
        "frozen_relative_band_or_trust_radius": False,
        "pooled_or_32d_representation": False,
        "phase0_action_teacher_exact_zero": True,
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            raise InferenceContractError(
                f"V4 training contract mismatch for {key}: {contract.get(key)!r}"
            )
    runtime_transformers = contract.get("transformers_version")
    if not isinstance(runtime_transformers, str) or not runtime_transformers:
        raise InferenceContractError("V4 training Transformers version is absent")
    explicit_targets = training_receipt.get("target_modules")
    if explicit_targets != expected_targets:
        raise InferenceContractError("V4 receipt full target registry differs")
    if training_receipt.get("target_module_count") != len(expected_targets):
        raise InferenceContractError("V4 target module count differs")
    expected_digest = object_sha256(expected_targets)
    if training_receipt.get("target_modules_sha256") != expected_digest:
        raise InferenceContractError("V4 target module digest differs")
    if (
        training_receipt.get("trainable_parameter_count")
        != V4_FULLFIELD_TRAINABLE_PARAMETERS
    ):
        raise InferenceContractError("V4 trainable capacity differs")
    distributed = training_receipt.get("distributed")
    if not isinstance(distributed, dict) or (
        distributed.get("world_size"), distributed.get("ulysses_size")
    ) != (ULYSSES_SIZE, ULYSSES_SIZE):
        raise InferenceContractError("V4 SP4 training contract differs")
    memory = training_receipt.get("memory_gate")
    if (
        not isinstance(memory, dict)
        or memory.get("passed") is not True
        or memory.get("dummy_or_padding_allocations") is not False
        or _finite_number(
            memory.get("minimum_reserved_fraction"),
            label="V4 minimum reserved fraction",
        )
        <= 0.5
    ):
        raise InferenceContractError("V4 >50% real-memory gate is absent")
    if training_receipt.get("production_claim_forbidden") is not True or (
        training_receipt.get("scientific_claim_authorized") is not False
    ):
        raise InferenceContractError("V4 experimental claim boundary differs")
    for key in ("method_source_revision", "method_source_archive_sha256"):
        value = training_receipt.get(key)
        pattern = _SHA1_RE if key.endswith("revision") else _SHA256_RE
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise InferenceContractError(f"V4 receipt has invalid {key}")
    return {
        "global_step": global_step,
        "receipt_digest": stored_digest,
        "target_modules_sha256": expected_digest,
        "target_modules": expected_targets,
        "transformers_version": runtime_transformers,
        "method_source_revision": training_receipt["method_source_revision"],
        "method_source_archive_sha256": training_receipt[
            "method_source_archive_sha256"
        ],
        "v4_fullfield": True,
    }


def validate_adapter_state_dicts(
    saved: Mapping[str, Any],
    loaded: Mapping[str, Any],
    *,
    tensor_equal: Callable[[Any, Any], bool],
    expected_targets: Optional[Sequence[str]] = None,
    require_zero_effect: bool = False,
    tensor_is_zero: Optional[Callable[[Any], bool]] = None,
) -> int:
    saved_keys, loaded_keys = set(saved), set(loaded)
    expected_keys = set(expected_adapter_state_keys(expected_targets))
    if saved_keys != expected_keys or loaded_keys != expected_keys:
        missing = sorted(expected_keys - loaded_keys)
        unexpected = sorted(loaded_keys - expected_keys)
        malformed_saved = sorted(saved_keys ^ expected_keys)
        raise InferenceContractError(
            "strict adapter reload key mismatch: "
            f"missing={missing[:4]} unexpected={unexpected[:4]} "
            f"saved_scope_difference={malformed_saved[:4]}"
        )
    expected_count = 2 * len(
        expected_lora_target_modules()
        if expected_targets is None
        else list(expected_targets)
    )
    if len(saved_keys) != expected_count:
        raise InferenceContractError(
            f"adapter tensor count must be {expected_count}, got {len(saved_keys)}"
        )
    unequal = [key for key in sorted(saved_keys) if not tensor_equal(saved[key], loaded[key])]
    if unequal:
        raise InferenceContractError(f"strict adapter reload tensor mismatch: {unequal[:4]}")
    if require_zero_effect:
        if tensor_is_zero is None:
            raise InferenceContractError(
                "checkpoint-0 baseline validation requires a zero-tensor predicate"
            )
        nonzero_b = [
            key
            for key in sorted(saved_keys)
            if ".lora_B.weight" in key and not tensor_is_zero(saved[key])
        ]
        if nonzero_b:
            raise InferenceContractError(
                "checkpoint-0 adapter is not a frozen zero-effect baseline: "
                f"nonzero_lora_B={nonzero_b[:4]}"
            )
    return len(saved_keys)


def validate_inference_source_files(bernini_root: Path) -> dict[str, str]:
    identities: dict[str, str] = {}
    for relative, expected in BERNINI_INFERENCE_FILE_HASHES.items():
        path = _plain_file(bernini_root / relative, label=f"pinned Bernini {relative}")
        actual = file_sha256(path)
        if actual != expected:
            raise InferenceContractError(
                f"pinned Bernini inference file hash mismatch: {relative}: {actual}"
            )
        identities[relative] = actual
    return identities


def prepare_exact_source(source_path: Path) -> tuple[Any, dict[str, Any]]:
    """Decode only the user source and apply the training spatial bucket."""

    materialize_vae = (
        exact_video_materializer.install_exact_local_video_materializer()
    )

    frames, reported_fps, source_hw = materialize_vae._decode_exact_video(source_path)
    validate_exact_video_metadata(int(frames.shape[0]), reported_fps)
    bucket_hw = materialize_vae.source_aspect_bucket(
        *source_hw,
        max_pixels=MAX_PIXELS,
        stride=SPATIAL_STRIDE,
    )
    source = materialize_vae._resize_video(frames, bucket_hw, None)
    if tuple(int(value) for value in source.shape) != (
        3,
        FRAME_COUNT,
        bucket_hw[0],
        bucket_hw[1],
    ):
        raise InferenceContractError(f"source preprocessing shape mismatch: {tuple(source.shape)}")
    return source.unsqueeze(0), {
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "reported_fps": float(reported_fps),
        "source_input_hw": list(source_hw),
        "source_derived_bucket_hw": list(bucket_hw),
        "max_pixels": MAX_PIXELS,
        "stride": SPATIAL_STRIDE,
        "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
        "spatial_policy": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
        "resize": "torchvision_bicubic_antialias_true",
        "external_shared_i0": False,
    }


_SOURCE_FILE_AUTHORITY_FIELDS = frozenset(
    {
        "path", "sha256", "size", "mode", "device", "inode", "uid",
        "gid", "nlink", "rdev", "blocks", "mtime_ns", "ctime_ns",
    }
)


def _source_file_authority(
    raw: str, *, path: Path, expected_sha256: str
) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise InferenceContractError(
                    "source video authority contains a duplicate key"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise InferenceContractError(
            "source video authority literal is not JSON"
        ) from error
    if (
        type(value) is not dict
        or set(value) != _SOURCE_FILE_AUTHORITY_FIELDS
        or canonical_json_bytes(value).decode("utf-8") != raw
        or value["path"] != str(path)
        or value["sha256"] != expected_sha256
    ):
        raise InferenceContractError(
            "source video authority literal differs"
        )
    for field in _SOURCE_FILE_AUTHORITY_FIELDS - {"path", "sha256"}:
        if type(value[field]) is not int or value[field] < 0:
            raise InferenceContractError(
                "source video authority identity differs"
            )
    if (
        value["nlink"] != 1
        or value["size"] <= 0
        or value["mode"] & ~0o7777
    ):
        raise InferenceContractError(
            "source video authority file policy differs"
        )
    return value


def _source_observation(path: Path, info: os.stat_result, sha256: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256,
        "size": int(info.st_size),
        "mode": stat.S_IMODE(info.st_mode),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "nlink": int(info.st_nlink),
        "rdev": int(info.st_rdev),
        "blocks": int(getattr(info, "st_blocks", 0)),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _retained_source_parent_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_rdev, info.st_size,
        getattr(info, "st_blocks", 0), info.st_mtime_ns, info.st_ctime_ns,
    )


def _open_retained_source(
    path: Path, *, expected_sha256: str, authority_raw: str
) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or path.name in ("", ".", "..")
        or path.resolve(strict=True) != path
    ):
        raise InferenceContractError("source video path is not canonical")
    expected_sha256 = expected_sha256.lower()
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise InferenceContractError("source video expected SHA differs")
    authority = _source_file_authority(
        authority_raw, path=path, expected_sha256=expected_sha256
    )
    parent_fd = source_fd = None
    try:
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        os.set_inheritable(parent_fd, False)
        parent_identity = _retained_source_parent_identity(
            os.fstat(parent_fd)
        )
        source_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        os.set_inheritable(source_fd, False)
        before = os.fstat(source_fd)
        first_sha, first_size = _hash_descriptor(source_fd)
        middle = os.fstat(source_fd)
        second_sha, second_size = _hash_descriptor(source_fd)
        after = os.fstat(source_fd)
        named = os.stat(
            path.name, dir_fd=parent_fd, follow_symlinks=False
        )
        named_parent = path.parent.lstat()
        observed = _source_observation(path, before, first_sha)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or first_sha != expected_sha256
            or second_sha != expected_sha256
            or first_size != authority["size"]
            or second_size != authority["size"]
            or observed != authority
            or _source_observation(path, middle, second_sha) != authority
            or _source_observation(path, after, second_sha) != authority
            or _source_observation(path, named, second_sha) != authority
            or _retained_source_parent_identity(named_parent)
            != parent_identity
            or os.get_inheritable(source_fd)
            or os.get_inheritable(parent_fd)
        ):
            raise InferenceContractError(
                "source video retained authority differs"
            )
        return {
            "path": path,
            "authority": authority,
            "parent_fd": parent_fd,
            "source_fd": source_fd,
            "parent_identity": parent_identity,
            "consumer_path": Path(f"/proc/self/fd/{source_fd}"),
            "sha256": expected_sha256,
        }
    except BaseException:
        if source_fd is not None:
            os.close(source_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        raise


def _replay_retained_source(state: Mapping[str, Any]) -> None:
    source_fd = state["source_fd"]
    parent_fd = state["parent_fd"]
    before = os.fstat(source_fd)
    first_sha, first_size = _hash_descriptor(source_fd)
    middle = os.fstat(source_fd)
    second_sha, second_size = _hash_descriptor(source_fd)
    after = os.fstat(source_fd)
    named = os.stat(
        state["path"].name, dir_fd=parent_fd, follow_symlinks=False
    )
    named_parent = state["path"].parent.lstat()
    authority = state["authority"]
    if (
        _source_observation(state["path"], before, first_sha) != authority
        or _source_observation(state["path"], middle, second_sha) != authority
        or _source_observation(state["path"], after, second_sha) != authority
        or _source_observation(state["path"], named, second_sha) != authority
        or first_sha != state["sha256"]
        or second_sha != state["sha256"]
        or first_size != authority["size"]
        or second_size != authority["size"]
        or _retained_source_parent_identity(os.fstat(parent_fd))
        != state["parent_identity"]
        or _retained_source_parent_identity(named_parent)
        != state["parent_identity"]
        or os.get_inheritable(source_fd)
        or os.get_inheritable(parent_fd)
    ):
        raise InferenceContractError("source video final retained replay differs")


def _close_retained_source(state: Mapping[str, Any]) -> None:
    for field in ("source_fd", "parent_fd"):
        try:
            os.close(state[field])
        except OSError:
            pass


_RETAINED_SOURCE_LIFETIMES: list[Mapping[str, Any]] = []


def _close_registered_retained_sources(
    function: Callable[..., int]
) -> Callable[..., int]:
    """Give main() a deterministic failure-finally without a path reopen."""
    def wrapped(*args: Any, **kwargs: Any) -> int:
        if _RETAINED_SOURCE_LIFETIMES:
            raise InferenceContractError(
                "retained source lifetime registry is not empty"
            )
        try:
            return function(*args, **kwargs)
        finally:
            while _RETAINED_SOURCE_LIFETIMES:
                _close_retained_source(
                    _RETAINED_SOURCE_LIFETIMES.pop()
                )
    return wrapped


def training_prompt_tokenizer_kwargs() -> dict[str, Any]:
    """The exact unpadded/untruncated call used by renderer training."""

    return {
        "add_special_tokens": True,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }


def renderer_negative_tokenizer_kwargs() -> dict[str, Any]:
    """The fixed-length call used by the official renderer inference path."""

    return {
        "padding": "max_length",
        "max_length": 512,
        "truncation": True,
        "add_special_tokens": True,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }


def _tokenize_training_prompt(tokenizer: Any, text: str) -> tuple[Any, Any]:
    """Match encode_renderer_messages + get_t5_text_embeddings exactly.

    Training tokenizes without truncation and then slices the resulting IDs to
    512 inside the renderer.  Asking the tokenizer to truncate is not
    equivalent for long prompts because Transformers forces EOS into the last
    retained position.  Pad short prompts with the same zero ID/mask values as
    the training renderer and manually slice long prompts without rewriting
    their final token.
    """

    import torch

    encoded = tokenizer(text, **training_prompt_tokenizer_kwargs())
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    if (
        input_ids.ndim != 2
        or attention_mask.ndim != 2
        or input_ids.shape[0] != 1
        or tuple(input_ids.shape) != tuple(attention_mask.shape)
        or input_ids.shape[1] <= 0
    ):
        raise InferenceContractError("tokenizer produced invalid renderer training inputs")
    length = int(input_ids.shape[1])
    if length >= 512:
        input_ids = input_ids[:, :512]
        attention_mask = attention_mask[:, :512]
    else:
        padding = 512 - length
        input_ids = torch.cat(
            [input_ids, input_ids.new_zeros((1, padding))], dim=1
        )
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((1, padding))], dim=1
        )
    return input_ids, attention_mask


def _tokenize_renderer_negative(tokenizer: Any, text: str) -> tuple[Any, Any]:
    encoded = tokenizer(
        text,
        **renderer_negative_tokenizer_kwargs(),
    )
    if tuple(encoded.input_ids.shape) != (1, 512) or tuple(encoded.attention_mask.shape) != (1, 512):
        raise InferenceContractError("tokenizer did not produce the fixed [1,512] renderer inputs")
    return encoded.input_ids, encoded.attention_mask


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if _ACTIVE_INHERITED_FDS is not None:
        try:
            inherited = model_authority.validate_inherited_fd_binding(
                _ACTIVE_INHERITED_FDS,
                verify_open_fds=True,
                expected_inheritable=False,
            )
            task = model_authority.inherited_fd_row(
                inherited, scope="task", role="publication_root"
            )
        except model_authority.ModelConsumptionAuthorityError as error:
            raise InferenceContractError(str(error)) from error
        expected_parent = Path(f"/proc/self/fd/{task['fd']}")
        if (
            not path.is_absolute()
            or path.parent != expected_parent
            or path.name in ("", ".", "..")
            or "/" in path.name
            or "\x00" in path.name
        ):
            raise InferenceContractError(
                "inference receipt is outside inherited task root"
            )
        descriptor: int | None = None
        try:
            descriptor = os.open(
                path.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o400,
                dir_fd=task["fd"],
            )
            os.set_inheritable(descriptor, False)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise InferenceContractError(
                        "inference receipt write made no progress"
                    )
                offset += written
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            before = model_authority._identity(os.fstat(descriptor))
            first = model_authority._read_fd(descriptor)
            middle = model_authority._identity(os.fstat(descriptor))
            second = model_authority._read_fd(descriptor)
            after = model_authority._identity(os.fstat(descriptor))
            named = model_authority._identity(os.stat(
                path.name, dir_fd=task["fd"], follow_symlinks=False
            ))
            if (
                not stat.S_ISREG(before["mode"])
                or before["nlink"] != 1
                or stat.S_IMODE(before["mode"]) != 0o400
                or before != middle
                or before != after
                or before != named
                or first != payload
                or second != payload
                or len(first) != before["size"]
                or os.get_inheritable(descriptor)
            ):
                raise InferenceContractError(
                    "inference receipt same-FD publication differs"
                )
            os.fsync(task["fd"])
        except FileExistsError as error:
            raise InferenceContractError(
                f"refusing to overwrite inference receipt: {path}"
            ) from error
        except OSError as error:
            raise InferenceContractError(
                f"cannot publish inference receipt: {path}: {error}"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        try:
            replayed, replayed_identity = (
                model_authority.stable_inherited_task_file(
                    path,
                    inherited_fd_binding=inherited,
                    label="published inference receipt",
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
            model_authority.validate_inherited_fd_binding(
                inherited,
                verify_open_fds=True,
                expected_inheritable=False,
            )
        except model_authority.ModelConsumptionAuthorityError as error:
            raise InferenceContractError(str(error)) from error
        if (
            replayed != payload
            or stat.S_IMODE(replayed_identity["mode"]) != 0o400
            or replayed_identity["nlink"] != 1
        ):
            raise InferenceContractError(
                "inference receipt post-close publication differs"
            )
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return digest.hexdigest(), size
        digest.update(block)
        size += len(block)


def _create_retained_encoded_output(
    path: Path, *, production_mode: bool = True
) -> dict[str, Any]:
    """Create an unpublished inode for the production MP4 encoder.

    Linux production requires a sealable memfd.  The final staging basename
    therefore does not exist while ffmpeg is writing and while the trusted
    rank seals, decodes, and hashes the encoded bytes.  The non-Linux path is
    an explicit unlinked-file fallback used only by local Darwin tests; it
    must never be selected by the verified release.
    """

    if _ACTIVE_INHERITED_FDS is None:
        raise InferenceContractError(
            "retained encoded output requires model-consumption authority"
        )
    try:
        inherited = model_authority.validate_inherited_fd_binding(
            _ACTIVE_INHERITED_FDS,
            verify_open_fds=True,
            expected_inheritable=False,
        )
        task = model_authority.inherited_fd_row(
            inherited, scope="task", role="publication_root"
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise InferenceContractError(str(error)) from error
    expected_parent = Path(f"/proc/self/fd/{task['fd']}")
    if (
        not path.is_absolute()
        or path.parent != expected_parent
        or path.suffix.lower() != ".mp4"
        or path.name in ("", ".", "..")
        or "/" in path.name
        or "\x00" in path.name
    ):
        raise InferenceContractError(
            "encoded output is outside inherited task root"
        )
    descriptor: int | None = None
    private_name: str | None = None
    try:
        if production_mode:
            if (
                not sys.platform.startswith("linux")
                or not hasattr(os, "memfd_create")
                or not hasattr(os, "MFD_ALLOW_SEALING")
            ):
                raise InferenceContractError(
                    "production anonymous output requires a sealable Linux memfd"
                )
            descriptor = os.memfd_create(
                "apv2-anonymous-encoded-output",
                os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
            )
            os.fchmod(descriptor, 0o600)
            creation_method = "linux-sealed-memfd-v1"
            expected_initial_nlink = 0
        else:
            private_name = (
                f".{path.stem}.injected-anonymous-{os.getpid()}-"
                f"{os.urandom(8).hex()}"
            )
            descriptor = os.open(
                private_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=task["fd"],
            )
            os.unlink(private_name, dir_fd=task["fd"])
            private_name = None
            creation_method = "injected-unlinked-inode-v1"
            expected_initial_nlink = 0
        os.set_inheritable(descriptor, False)
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_nlink != expected_initial_nlink
            or initial.st_size != 0
            or stat.S_IMODE(initial.st_mode) != 0o600
        ):
            raise InferenceContractError(
                "encoded output initial inode differs"
            )
        try:
            os.stat(path.name, dir_fd=task["fd"], follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise InferenceContractError(
                "encoded output final basename is not fresh"
            )
        os.fsync(task["fd"])
        return {
            "binding": inherited,
            "task_fd": task["fd"],
            "descriptor": descriptor,
            "path": path,
            "writer_path": Path(f"/proc/self/fd/{descriptor}.mp4"),
            "encoded_fd_path": Path(f"/proc/self/fd/{descriptor}"),
            "creation_method": creation_method,
            "production_mode": production_mode,
            "private_name": private_name,
            "expected_unpublished_nlink": expected_initial_nlink,
            "initial_inode_identity": {
                key: model_authority._identity(initial)[key]
                for key in ("device", "inode", "uid", "gid", "rdev")
            },
        }
    except (OSError, model_authority.ModelConsumptionAuthorityError) as error:
        if private_name is not None:
            try:
                os.unlink(private_name, dir_fd=task["fd"])
            except OSError:
                pass
        if descriptor is not None:
            os.close(descriptor)
        if isinstance(error, InferenceContractError):
            raise
        raise InferenceContractError(
            "cannot create retained encoded output"
        ) from error
    except BaseException:
        if private_name is not None:
            try:
                os.unlink(private_name, dir_fd=task["fd"])
            except OSError:
                pass
        if descriptor is not None:
            os.close(descriptor)
        raise


def _save_output_with_exact_ffmpeg_fds(
    save_function: Callable[..., Any],
    output: Any,
    state: Mapping[str, Any],
    *,
    fps: int,
) -> None:
    """Run the official writer with only the anonymous output FD inherited."""

    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise InferenceContractError(
            "retained output requires imageio-ffmpeg"
        ) from error
    write_frames = getattr(imageio_ffmpeg, "write_frames", None)
    owner = (
        sys.modules.get(getattr(write_frames, "__module__", ""))
        if callable(write_frames) else None
    )
    original_subprocess = getattr(owner, "subprocess", None)
    if owner is None or original_subprocess is None:
        raise InferenceContractError(
            "imageio-ffmpeg subprocess authority differs"
        )
    pass_fds = (state["descriptor"],)
    if (
        len(pass_fds) != 1
        or any(os.get_inheritable(descriptor) for descriptor in pass_fds)
    ):
        raise InferenceContractError(
            "encoded output FD allowlist differs"
        )
    writer_path = str(state["writer_path"])
    encoded_fd_path = str(state["encoded_fd_path"])
    calls: list[list[str]] = []

    class _ScopedSubprocess:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_subprocess, name)

        def Popen(self, *args: Any, **kwargs: Any) -> Any:
            if (
                not args
                or not isinstance(args[0], (list, tuple))
                or not all(isinstance(item, str) for item in args[0])
                or args[0][-1] != writer_path
                or "-i" not in args[0]
                or "-vcodec" not in args[0]
                or kwargs.get("shell") not in (None, False)
                or "pass_fds" in kwargs
                or "close_fds" in kwargs
            ):
                raise InferenceContractError(
                    "imageio-ffmpeg launch contract differs"
                )
            command = list(args[0])
            command[-1:] = ["-f", "mp4", encoded_fd_path]
            calls.append(command)
            kwargs["close_fds"] = True
            kwargs["pass_fds"] = pass_fds
            return original_subprocess.Popen(command, *args[1:], **kwargs)

    scoped = _ScopedSubprocess()
    owner.subprocess = scoped
    try:
        save_function(output, writer_path, fps=fps)
    finally:
        if getattr(owner, "subprocess", None) is scoped:
            owner.subprocess = original_subprocess
        else:
            owner.subprocess = original_subprocess
            raise InferenceContractError(
                "imageio-ffmpeg subprocess hook changed during encoding"
            )
    if (
        len(calls) != 1
        or calls[0][-3:] != ["-f", "mp4", encoded_fd_path]
        or any(os.get_inheritable(descriptor) for descriptor in pass_fds)
    ):
        raise InferenceContractError(
            "imageio-ffmpeg exact FD launch differs"
        )


def _finalize_retained_encoded_output(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal and hash the still-unpublished encoded inode."""
    descriptor = state["descriptor"]
    task_fd = state["task_fd"]
    seal_mask: int | None = None
    if state["production_mode"]:
        seal_mask = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        try:
            observed_seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
            if observed_seals == 0:
                fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seal_mask)
                observed_seals = fcntl.fcntl(
                    descriptor, fcntl.F_GET_SEALS
                )
        except OSError as error:
            raise InferenceContractError(
                "cannot seal anonymous encoded output"
            ) from error
        if observed_seals != seal_mask:
            raise InferenceContractError(
                "anonymous encoded output seal set differs"
            )
    os.fsync(descriptor)
    os.fsync(task_fd)
    before = os.fstat(descriptor)
    first_sha, first_size = _hash_descriptor(descriptor)
    middle = os.fstat(descriptor)
    second_sha, second_size = _hash_descriptor(descriptor)
    after = os.fstat(descriptor)
    identity = model_authority._identity(before)
    try:
        os.stat(state["path"].name, dir_fd=task_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise InferenceContractError(
            "encoded output was published before trusted decode"
        )
    private_name = state.get("private_name")
    if private_name is not None:
        private = os.stat(
            private_name, dir_fd=task_fd, follow_symlinks=False
        )
        if model_authority._identity(private) != identity:
            raise InferenceContractError(
                "injected anonymous output name differs"
            )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != state["expected_unpublished_nlink"]
        or stat.S_IMODE(before.st_mode) != 0o600
        or first_size <= 0
        or first_sha != second_sha
        or first_size != second_size
        or identity != model_authority._identity(middle)
        or identity != model_authority._identity(after)
        or any(os.get_inheritable(fd) for fd in (descriptor, task_fd))
    ):
        raise InferenceContractError(
            "anonymous encoded output replay differs"
        )
    model_authority.validate_inherited_fd_binding(
        state["binding"],
        verify_open_fds=True,
        expected_inheritable=False,
    )
    return {
        "sha256": first_sha,
        "size": first_size,
        "unpublished_identity": identity,
        "seal_mask": seal_mask,
        "read_path": Path(f"/proc/self/fd/{descriptor}"),
    }


def _publish_retained_encoded_output(
    state: dict[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy a sealed source into one create-only held publication inode."""
    replay = _finalize_retained_encoded_output(state)
    if replay != dict(evidence):
        raise InferenceContractError(
            "anonymous encoded output changed before publication"
        )
    source_fd = state["descriptor"]
    task_fd = state["task_fd"]
    name = state["path"].name
    final_fd: int | None = None
    try:
        final_fd = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=task_fd,
        )
        os.set_inheritable(final_fd, False)
    except OSError as error:
        raise InferenceContractError(
            "cannot create-only publish anonymous encoded output"
        ) from error
    try:
        source_before = model_authority._identity(os.fstat(source_fd))
        os.lseek(source_fd, 0, os.SEEK_SET)
        copy_digest = hashlib.sha256()
        copy_size = 0
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            copy_digest.update(block)
            copy_size += len(block)
            view = memoryview(block)
            while view:
                written = os.write(final_fd, view)
                if written <= 0:
                    raise InferenceContractError(
                        "anonymous output copy made no progress"
                    )
                view = view[written:]
        source_middle = model_authority._identity(os.fstat(source_fd))
        source_after_sha, source_after_size = _hash_descriptor(source_fd)
        source_after = model_authority._identity(os.fstat(source_fd))
        final_before = model_authority._identity(os.fstat(final_fd))
        copied_sha, copied_size = _hash_descriptor(final_fd)
        os.fchmod(final_fd, 0o444)
        os.fsync(final_fd)
        os.fsync(task_fd)
        before = os.fstat(final_fd)
        first_sha, first_size = _hash_descriptor(final_fd)
        middle = os.fstat(final_fd)
        second_sha, second_size = _hash_descriptor(final_fd)
        after = os.fstat(final_fd)
        named = os.stat(name, dir_fd=task_fd, follow_symlinks=False)
        identity = model_authority._identity(before)
        if (
            source_before != evidence["unpublished_identity"]
            or source_middle != evidence["unpublished_identity"]
            or source_after != evidence["unpublished_identity"]
            or copy_digest.hexdigest() != evidence["sha256"]
            or copy_size != evidence["size"]
            or source_after_sha != evidence["sha256"]
            or source_after_size != evidence["size"]
            or copied_sha != evidence["sha256"]
            or copied_size != evidence["size"]
            or final_before["size"] != evidence["size"]
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or first_sha != evidence["sha256"]
            or second_sha != evidence["sha256"]
            or first_size != evidence["size"]
            or second_size != evidence["size"]
            or identity != model_authority._identity(middle)
            or identity != model_authority._identity(after)
            or identity != model_authority._identity(named)
            or any(os.get_inheritable(fd) for fd in (source_fd, final_fd, task_fd))
        ):
            raise InferenceContractError(
                "sealed anonymous source/final publication differs"
            )
        model_authority.validate_inherited_fd_binding(
            state["binding"], verify_open_fds=True,
            expected_inheritable=False,
        )
        state["publication_descriptor"] = final_fd
        state["published"] = True
        final_fd = None
        return {
            "sha256": first_sha,
            "size": first_size,
            "identity": identity,
            "prepublication_identity": dict(
                evidence["unpublished_identity"]
            ),
            "seal_mask": evidence["seal_mask"],
            "creation_method": state["creation_method"],
            "published_via_create_only_copy": True,
            "read_path": Path(
                f"/proc/self/fd/{state['publication_descriptor']}"
            ),
        }
    finally:
        if final_fd is not None:
            os.close(final_fd)


def _replay_retained_encoded_output(
    state: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    descriptor = state.get("publication_descriptor")
    if type(descriptor) is not int:
        raise InferenceContractError(
            "retained encoded output was not published"
        )
    source_fd = state["descriptor"]
    before = os.fstat(descriptor)
    first_sha, first_size = _hash_descriptor(descriptor)
    middle = os.fstat(descriptor)
    second_sha, second_size = _hash_descriptor(descriptor)
    after = os.fstat(descriptor)
    named = os.stat(
        state["path"].name,
        dir_fd=state["task_fd"],
        follow_symlinks=False,
    )
    source_before = model_authority._identity(os.fstat(source_fd))
    source_sha, source_size = _hash_descriptor(source_fd)
    source_after = model_authority._identity(os.fstat(source_fd))
    if (
        model_authority._identity(before) != evidence["identity"]
        or model_authority._identity(middle) != evidence["identity"]
        or model_authority._identity(after) != evidence["identity"]
        or model_authority._identity(named) != evidence["identity"]
        or first_sha != evidence["sha256"]
        or second_sha != evidence["sha256"]
        or first_size != evidence["size"]
        or second_size != evidence["size"]
        or source_before != evidence["prepublication_identity"]
        or source_after != evidence["prepublication_identity"]
        or source_sha != evidence["sha256"]
        or source_size != evidence["size"]
        or (
            state["production_mode"]
            and fcntl.fcntl(source_fd, fcntl.F_GET_SEALS)
            != evidence["seal_mask"]
        )
    ):
        raise InferenceContractError(
            "retained encoded output final replay differs"
        )
    model_authority.validate_inherited_fd_binding(
        state["binding"],
        verify_open_fds=True,
        expected_inheritable=False,
    )


def _close_retained_encoded_output(state: Mapping[str, Any]) -> None:
    try:
        private_name = state.get("private_name")
        if private_name is not None:
            try:
                observed = os.stat(
                    private_name,
                    dir_fd=state["task_fd"],
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                observed = None
            if (
                observed is not None
                and observed.st_ino
                == os.fstat(state["descriptor"]).st_ino
                and observed.st_dev
                == os.fstat(state["descriptor"]).st_dev
            ):
                os.unlink(private_name, dir_fd=state["task_fd"])
                os.fsync(state["task_fd"])
    finally:
        publication_descriptor = state.get("publication_descriptor")
        if type(publication_descriptor) is int:
            try:
                os.close(publication_descriptor)
            except OSError:
                pass
        try:
            os.close(state["descriptor"])
        except OSError:
            pass


def build_inference_receipt(
    *,
    args: argparse.Namespace,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    output_path: Path,
    output_sha256: str,
    adapter: Optional[AdapterBundle],
    adapter_sha256: Optional[str],
    adapter_identity: Optional[Mapping[str, Any]],
    adapter_tensor_count: int,
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
    source_onset_solver_trace: Optional[Mapping[str, Any]] = None,
    model_consumption_evidence: Optional[Mapping[str, Any]] = None,
    output_identity: Optional[Mapping[str, Any]] = None,
    output_publication_evidence: Optional[Mapping[str, Any]] = None,
    source_file_authority: Optional[Mapping[str, Any]] = None,
    adapter_manifest_identity: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    instruction_bytes = args.instruction.encode("utf-8")
    if adapter is None:
        if (
            adapter_sha256 is not None
            or adapter_identity is not None
            or adapter_manifest_identity is not None
            or adapter_tensor_count != 0
        ):
            raise InferenceContractError("base-only receipt may not claim an adapter")
        adaptation_receipt: dict[str, Any] = {
            "enabled": False,
            "mode": "frozen_base_no_adapter",
            "strictly_reloaded": False,
            "safe_merged_for_inference": False,
            "tensor_count": 0,
        }
    else:
        if adapter_sha256 is None or adapter_identity is None:
            raise InferenceContractError("adapter receipt lacks its byte identity")
        adaptation_receipt = {
            "enabled": True,
            "mode": "lora_safe_merge",
            "checkpoint_root": str(adapter.checkpoint_root),
            "adapter_model_path": str(adapter.adapter_model_path),
            "adapter_model_sha256": adapter_sha256,
            "training_receipt_path": str(adapter.training_receipt_path),
            "training_receipt_digest": adapter_identity["receipt_digest"],
            "training_global_step": adapter_identity["global_step"],
            "strictly_reloaded": True,
            "safe_merged_for_inference": True,
            "tensor_count": adapter_tensor_count,
            "target_modules_sha256": adapter_identity["target_modules_sha256"],
        }
        if adapter_identity.get("exploratory_full644") is True:
            if not isinstance(adapter_manifest_identity, Mapping):
                raise InferenceContractError(
                    "full644 adapter receipt lacks checkpoint manifest identity"
                )
            if adapter_tensor_count != EXPECTED_ADAPTER_TENSOR_COUNT:
                raise InferenceContractError(
                    "full644 adapter receipt tensor count differs"
                )
            adaptation_receipt.update(
                {
                    "profile": trainer.FULL644_EXPLORATORY_PROFILE,
                    "lora_rank": adapter_identity["lora_rank"],
                    "lora_alpha": adapter_identity["lora_alpha"],
                    "target_module_count": adapter_identity[
                        "target_module_count"
                    ],
                }
            )
            adaptation_receipt["checkpoint_manifest"] = dict(
                adapter_manifest_identity
            )
        elif adapter_manifest_identity is not None:
            adaptation_receipt["checkpoint_manifest"] = dict(
                adapter_manifest_identity
            )
        if "lora_route_scope" in adapter_identity:
            adaptation_receipt.update(
                {
                    "lora_route_scope": adapter_identity["lora_route_scope"],
                    "target_module_count": len(adapter_identity["target_modules"]),
                    "frozen_baseline": adapter_identity["frozen_baseline"],
                }
            )
    output_receipt: dict[str, Any] = {
        "path": str(output_path),
        "sha256": output_sha256,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "height": source_metadata["source_derived_bucket_hw"][0],
        "width": source_metadata["source_derived_bucket_hw"][1],
        "audio_preserved": False,
    }
    if output_identity is not None:
        identity_fields = {
            "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
            "size", "blocks", "mtime_ns", "ctime_ns",
        }
        if (
            type(output_identity) is not dict
            or set(output_identity) != identity_fields
            or any(type(output_identity[field]) is not int
                   for field in identity_fields)
            or not stat.S_ISREG(output_identity["mode"])
            or stat.S_IMODE(output_identity["mode"]) != 0o444
            or output_identity["nlink"] != 1
            or output_identity["size"] <= 0
        ):
            raise InferenceContractError(
                "encoded output publication identity differs"
            )
        if not isinstance(output_publication_evidence, Mapping):
            raise InferenceContractError(
                "encoded output lacks anonymous publication evidence"
            )
        prepublication_identity = output_publication_evidence.get(
            "prepublication_identity"
        )
        expected_seal_mask = (
            fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        )
        if (
            type(prepublication_identity) is not dict
            or set(prepublication_identity) != identity_fields
            or any(type(prepublication_identity[field]) is not int
                   for field in identity_fields)
            or not stat.S_ISREG(prepublication_identity["mode"])
            or stat.S_IMODE(prepublication_identity["mode"]) != 0o600
            or prepublication_identity["nlink"] != 0
            or prepublication_identity["size"] != output_identity["size"]
            or output_publication_evidence.get("creation_method")
            != "linux-sealed-memfd-v1"
            or output_publication_evidence.get("seal_mask")
            != expected_seal_mask
            or output_publication_evidence.get(
                "published_via_create_only_copy"
            ) is not True
            or output_publication_evidence.get("sha256") != output_sha256
            or output_publication_evidence.get("size") != output_identity["size"]
        ):
            raise InferenceContractError(
                "encoded output anonymous publication evidence differs"
            )
        output_receipt.update(
            size=output_identity["size"],
            publication_identity=dict(output_identity),
            prepublication_identity=dict(prepublication_identity),
            anonymous_creation_method="linux-sealed-memfd-v1",
            anonymous_seal_mask=expected_seal_mask,
            sealed_source_sha256=output_sha256,
            sealed_source_size=output_identity["size"],
            anonymous_inode_encoded_and_decoded_before_publication=True,
            create_only_copy_publication_after_decode=True,
            sealed_source_and_publication_bytes_equal=True,
            retained_inode_encoded_and_replayed=True,
            named_output_never_replaced=True,
        )
    elif output_publication_evidence is not None:
        raise InferenceContractError(
            "anonymous publication evidence lacks a publication identity"
        )
    input_receipt: dict[str, Any] = {
        "source_video_path": str(source_path),
        "source_video_sha256": source_sha256,
        "instruction_utf8_sha256": hashlib.sha256(instruction_bytes).hexdigest(),
        "instruction_utf8_bytes": len(instruction_bytes),
        "accepted_model_conditions": ["source_video", "edit_instruction"],
        "target_video_argument": False,
        "target_accessed_by_inference": False,
        "external_mask_or_swept_tube": False,
        "external_tracking_pose_or_trajectory": False,
        "reference_image_or_video": False,
        "external_shared_i0": False,
    }
    if source_file_authority is not None:
        authority = dict(source_file_authority)
        if (
            set(authority) != _SOURCE_FILE_AUTHORITY_FIELDS
            or authority.get("path") != str(source_path)
            or authority.get("sha256") != source_sha256
        ):
            raise InferenceContractError(
                "receipt source retained authority differs"
            )
        input_receipt.update(
            source_video_physical_authority=authority,
            source_video_physical_authority_digest=object_sha256(authority),
            retained_source_fd_consumed=True,
            source_video_pre_and_post_decode_rehashed=True,
        )
    receipt: dict[str, Any] = {
        "schema_version": INFERENCE_RECEIPT_SCHEMA,
        "infer_lora_source_sha256": hashlib.sha256(
            _stable_plain_file_bytes(
                Path(__file__).resolve(), label="infer_lora source"
            )
        ).hexdigest(),
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "bernini_inference_files": dict(inference_file_hashes),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "adapter": adaptation_receipt,
        "input": input_receipt,
        "preprocessing": dict(source_metadata),
        "prompt_contract": {
            "task": "mv2v",
            "system_prompt_sha256": hashlib.sha256(MV2V_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenizer_fix_mistral_regex": True,
            "tokenizer_padding_side": "right",
            "max_sequence_length": 512,
            "prompt_enhancer": False,
        },
        "sampling": {
            **sampler_contract(steps=args.num_inference_steps, seed=args.seed),
            "single_expert": "transformer_1",
            "ulysses_size": ULYSSES_SIZE,
            "rank0_decode_and_save_only": True,
            "source_onset_policy": getattr(args, "source_onset_policy", "none"),
        },
        "output": output_receipt,
        "runtime_versions": dict(runtime_versions),
        "experimental_inference": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    # Keep the existing none/hard1/ramp3 receipt payload unchanged.  The new
    # field exists only when a native every-step intervention actually ran.
    if source_onset_solver_trace is not None:
        receipt["sampling"]["source_onset_solver_trace"] = dict(
            source_onset_solver_trace
        )
    if model_consumption_evidence is not None:
        evidence = dict(model_consumption_evidence)
        required = {
            "consumption_input_digest", "task_input_digest",
            "model_capture_digest", "model_view_root",
            "adapter_capture_digest", "adapter_view_root",
            "fd_view_files_authorized", "inherited_fd_binding_digest",
            "inherited_fd_count", "ptrace_authorization_used",
            "source_video_sha256",
            "source_video_physical_authority_digest",
            "all_ranks_use_retained_source_fd",
            "four_rank_attestation",
        }
        if set(evidence) != required:
            raise InferenceContractError(
                "model consumption receipt evidence closure differs"
            )
        if source_file_authority is None:
            raise InferenceContractError(
                "model consumption receipt lacks retained source authority"
            )
        if (
            evidence.get("source_video_sha256") != source_sha256
            or evidence.get("source_video_physical_authority_digest")
            != object_sha256(dict(source_file_authority))
            or evidence.get("all_ranks_use_retained_source_fd") is not True
        ):
            raise InferenceContractError(
                "model consumption retained source evidence differs"
            )
        receipt["consumption_input_digest"] = evidence[
            "consumption_input_digest"
        ]
        receipt["task_input_digest"] = evidence["task_input_digest"]
        receipt["model_consumption"] = evidence
    receipt["receipt_digest"] = object_sha256(receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run source-video + instruction only Bernini-R 1.3B base/LoRA inference"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    adaptation = parser.add_mutually_exclusive_group(required=True)
    adaptation.add_argument("--adapter-checkpoint")
    parser.add_argument(
        "--adapter-checkpoint-manifest-sha256",
        help=(
            "Expected SHA-256 of checkpoint_manifest.json. Required for the "
            "terminal full644 exploratory adapter."
        ),
    )
    adaptation.add_argument(
        "--base-only",
        action="store_true",
        help="Use the frozen Bernini checkpoint without loading or merging an adapter",
    )
    parser.add_argument(
        "--adapter-checkpoint-manifest",
        help=(
            "Absolute original checkpoint_manifest.json path. Full644 retained-FD "
            "inference uses this authority while consuming adapter bytes via the FD view."
        ),
    )
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--source-video-sha256")
    parser.add_argument("--source-video-authority")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS_DEFAULT
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-onset-policy",
        choices=tuple(SOURCE_ONSET_POLICIES),
        default="none",
        help=(
            "Optional post-denoise source-latent onset boundary: none, exact "
            "phase-0 hard1, three-phase 1/.5/.25 ramp3, or native UniPC "
            "phase-0 hard1_every_step"
        ),
    )
    parser.add_argument(
        "--expected-bernini-commit", default=trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=trainer.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--model-consumption-input")
    parser.add_argument("--model-consumption-input-sha256")
    parser.add_argument("--model-consumption-input-digest")
    parser.add_argument("--task-input-digest")
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    base_only = bool(getattr(args, "base_only", False))
    adapter_checkpoint = getattr(args, "adapter_checkpoint", None)
    if base_only:
        if (
            adapter_checkpoint is not None
            or getattr(args, "adapter_checkpoint_manifest", None) is not None
            or getattr(args, "adapter_checkpoint_manifest_sha256", None) is not None
        ):
            raise InferenceContractError("base-only mode may not accept an adapter checkpoint")
    elif not isinstance(adapter_checkpoint, str) or not adapter_checkpoint.strip():
        raise InferenceContractError("exactly one of base-only or adapter checkpoint is required")
    manifest_sha256 = getattr(args, "adapter_checkpoint_manifest_sha256", None)
    manifest_path = getattr(args, "adapter_checkpoint_manifest", None)
    if (manifest_path is None) is not (manifest_sha256 is None):
        raise InferenceContractError(
            "adapter checkpoint manifest path/SHA must be supplied together"
        )
    if manifest_path is not None and (
        not isinstance(manifest_path, str)
        or not Path(manifest_path).expanduser().is_absolute()
    ):
        raise InferenceContractError("adapter checkpoint manifest path differs")
    if manifest_sha256 is not None and (
        not isinstance(manifest_sha256, str)
        or _SHA256_RE.fullmatch(manifest_sha256) is None
    ):
        raise InferenceContractError("adapter checkpoint manifest SHA differs")
    if not isinstance(args.instruction, str) or not args.instruction.strip() or "\x00" in args.instruction:
        raise InferenceContractError("instruction must be non-empty text without NUL")
    if type(args.num_inference_steps) is not int or args.num_inference_steps <= 0:
        raise InferenceContractError("num_inference_steps must be positive")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise InferenceContractError("seed must be in [0, 2^63)")
    if getattr(args, "source_onset_policy", "none") not in SOURCE_ONSET_POLICIES:
        raise InferenceContractError("source_onset_policy is unsupported")
    for key in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        value = getattr(args, key)
        if not isinstance(value, str) or _SHA1_RE.fullmatch(value.lower()) is None:
            raise InferenceContractError(f"{key} must be a full SHA-1")
    for key in ("expected_checkpoint_tree_sha256", "method_source_archive_sha256"):
        value = getattr(args, key)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise InferenceContractError(f"{key} must be a lowercase SHA-256")
    if args.expected_bernini_commit.lower() != trainer.BERNINI_OFFICIAL_COMMIT:
        raise InferenceContractError("only the audited Bernini source commit is supported")
    if args.expected_veomni_commit.lower() != trainer.VEOMNI_TESTED_COMMIT:
        raise InferenceContractError("only the tested VeOmni commit is supported")
    if args.expected_checkpoint_tree_sha256 != trainer.CHECKPOINT_TREE_SHA256:
        raise InferenceContractError("only the audited Bernini-R 1.3B checkpoint is supported")
    consumption_values = (
        getattr(args, "model_consumption_input", None),
        getattr(args, "model_consumption_input_sha256", None),
        getattr(args, "model_consumption_input_digest", None),
        getattr(args, "task_input_digest", None),
    )
    if any(value is None for value in consumption_values) and not all(
        value is None for value in consumption_values
    ):
        raise InferenceContractError("model consumption arguments are incomplete")
    if consumption_values[0] is not None and _SHA256_RE.fullmatch(
        args.task_input_digest
    ) is None:
        raise InferenceContractError("task input digest differs")
    source_authority_values = (
        getattr(args, "source_video_sha256", None),
        getattr(args, "source_video_authority", None),
    )
    if any(value is None for value in source_authority_values) and not all(
        value is None for value in source_authority_values
    ):
        raise InferenceContractError(
            "source video SHA and physical authority must be supplied together"
        )
    if consumption_values[0] is not None and source_authority_values[0] is None:
        raise InferenceContractError(
            "model-consumption inference lacks source video authority"
        )
    if (
        source_authority_values[0] is not None
        and _SHA256_RE.fullmatch(source_authority_values[0]) is None
    ):
        raise InferenceContractError("source video expected SHA differs")


def activate_model_consumption_authority(
    args: argparse.Namespace,
) -> Mapping[str, Any] | None:
    """Replay the exact15 pre-use receipt and authorize only its FD leaves."""

    global _AUTHORIZED_FD_VIEW_FILES
    global _AUTHORIZED_FD_VIEW_DIRECTORIES
    global _ACTIVE_INHERITED_FDS
    path = getattr(args, "model_consumption_input", None)
    if path is None:
        _AUTHORIZED_FD_VIEW_FILES = {}
        _AUTHORIZED_FD_VIEW_DIRECTORIES = {}
        _ACTIVE_INHERITED_FDS = None
        return None
    try:
        consumption, model_capture, adapter_capture = (
            model_authority.load_consumption_input(
                path,
                expected_sha256=args.model_consumption_input_sha256,
                expected_digest=args.model_consumption_input_digest,
                verify_views=True,
            )
        )
        inherited_fds = model_authority.load_inherited_fd_environment(
            model_capture=model_capture,
            adapter_capture=adapter_capture,
            verify_open_fds=True,
            expected_inheritable=False,
        )
        if inherited_fds != consumption["inherited_fds"]:
            raise model_authority.ModelConsumptionAuthorityError(
                "rank environment/consumption FD binding differs"
            )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise InferenceContractError(str(error)) from error
    if Path(args.checkpoint) != Path(consumption["model"]["view_root"]):
        raise InferenceContractError("base model was not routed through its FD view")
    base_only = bool(getattr(args, "base_only", False))
    if base_only is not (adapter_capture is None):
        raise InferenceContractError("adapter consumption/base-only closure differs")
    if adapter_capture is not None and Path(args.adapter_checkpoint) != Path(
        consumption["adapter"]["view_root"]
    ):
        raise InferenceContractError("adapter was not routed through its FD view")
    authorized: dict[str, Mapping[str, Any]] = {}
    authorized_directories: dict[str, Mapping[str, Any]] = {}
    capture_rows = [
        (model_capture, consumption["model"]["view_root"], "model")
    ]
    if adapter_capture is not None:
        capture_rows.append(
            (
                adapter_capture,
                consumption["adapter"]["view_root"],
                "adapter",
            )
        )
    for capture, inherited_root, scope in capture_rows:
        view_root = Path(inherited_root)
        for item in capture["files"]:
            authorized[str(view_root / item["relative_path"])] = item
        namespace = model_authority.inherited_fd_row(
            inherited_fds, scope=scope, role="namespace_root"
        )
        for directory in capture["view_directories"]:
            relative = directory["relative_path"]
            path_value = view_root if relative == "." else view_root / relative
            identity = (
                namespace["identity"]
                if relative == "."
                else directory["identity"]
            )
            authorized_directories[str(path_value)] = {
                "identity": identity,
                "direct_fd": namespace["fd"] if relative == "." else None,
            }
    task_root = model_authority.inherited_fd_row(
        inherited_fds, scope="task", role="publication_root"
    )
    authorized_directories[f"/proc/self/fd/{task_root['fd']}"] = {
        "identity": model_authority._identity(os.fstat(task_root["fd"])),
        "direct_fd": task_root["fd"],
    }
    _AUTHORIZED_FD_VIEW_FILES = authorized
    _AUTHORIZED_FD_VIEW_DIRECTORIES = authorized_directories
    _ACTIVE_INHERITED_FDS = inherited_fds
    return {
        "consumption_input_digest": consumption["consumption_input_digest"],
        "task_input_digest": args.task_input_digest,
        "model_capture_digest": model_capture["capture_digest"],
        "model_view_root": consumption["model"]["view_root"],
        "adapter_capture_digest": (
            adapter_capture["capture_digest"] if adapter_capture is not None else None
        ),
        "adapter_view_root": (
            consumption["adapter"]["view_root"]
            if adapter_capture is not None else None
        ),
        "fd_view_files_authorized": len(authorized),
        "inherited_fd_binding_digest": inherited_fds["fd_binding_digest"],
        "inherited_fd_count": inherited_fds["fd_count"],
        "ptrace_authorization_used": False,
    }


def _resolve_output(value: str | Path) -> tuple[Path, Path]:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise InferenceContractError(f"output must be an absolute path: {requested}")
    if requested.suffix.lower() != ".mp4":
        raise InferenceContractError("output must end in .mp4")
    parent = _absolute_directory(requested.parent, label="output parent")
    output = parent / requested.name
    receipt = output.with_name(f"{output.name}.receipt.json")
    for path, label in ((output, "output"), (receipt, "inference receipt")):
        if path.exists() or path.is_symlink():
            raise InferenceContractError(f"refusing to overwrite existing {label}: {path}")
    return output, receipt


def _strict_load_and_merge_adapter(
    base_model: Any,
    adapter: AdapterBundle,
    expected_targets: Sequence[str],
    *,
    route_scope: Optional[str] = None,
    require_zero_effect: bool = False,
) -> tuple[Any, int]:
    import torch
    from peft import LoraConfig, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors import safe_open

    class _StreamingSafeTensorMapping(Mapping[str, Any]):
        """Expose one safetensor at a time to the strict reload validator."""

        def __init__(self, handle: Any) -> None:
            self._handle = handle
            self._keys = tuple(handle.keys())

        def __iter__(self) -> Any:
            return iter(self._keys)

        def __len__(self) -> int:
            return len(self._keys)

        def __getitem__(self, key: str) -> Any:
            if key not in self._keys:
                raise KeyError(key)
            return self._handle.get_tensor(key)

    actual_full_targets = trainer.select_attention_projection_names(base_model)
    if list(actual_full_targets) != expected_lora_target_modules():
        raise InferenceContractError(
            "runtime Bernini full target registry differs from the audited Wan model"
        )
    if route_scope is None:
        actual_targets = list(actual_full_targets)
    else:
        preservation_v2 = _load_preservation_v2_contract()
        try:
            actual_targets = preservation_v2.select_projection_scope(
                actual_full_targets, scope=route_scope
            )
        except preservation_v2.ActionPreservationV2Error as error:
            raise InferenceContractError(
                f"runtime preservation-v2 route scope differs: {error}"
            ) from error
    if list(actual_targets) != list(expected_targets):
        raise InferenceContractError(
            "runtime Bernini target module set differs from adapter contract"
        )
    peft_config = LoraConfig.from_pretrained(
        str(adapter.adapter_dir), local_files_only=True
    )
    peft_config.target_modules = set(expected_targets)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter.adapter_dir),
        is_trainable=False,
        config=peft_config,
        local_files_only=True,
    )
    loaded = get_peft_model_state_dict(peft_model, adapter_name="default")
    # Do not materialize the complete 721-MiB rank-256 safetensors file on
    # every Ulysses rank merely to verify the already-loaded PEFT state.  The
    # Mapping wrapper preserves exact key/tensor equality while reading only
    # one saved tensor at a time; this removes a multi-GiB four-rank host peak.
    with safe_open(
        str(adapter.adapter_model_path), framework="pt", device="cpu"
    ) as saved_handle:
        count = validate_adapter_state_dicts(
            _StreamingSafeTensorMapping(saved_handle),
            loaded,
            tensor_equal=lambda left, right: bool(
                torch.equal(left.cpu(), right.cpu())
            ),
            expected_targets=expected_targets,
            require_zero_effect=require_zero_effect,
            tensor_is_zero=lambda value: bool(
                torch.count_nonzero(value.detach()).item() == 0
            ),
        )
    del loaded
    gc.collect()
    trim_process_heap()
    merged = peft_model.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in merged.named_modules()):
        raise InferenceContractError("LoRA modules remain after merge_and_unload")
    merged.requires_grad_(False)
    merged.eval()
    return merged, count


@_close_registered_retained_sources
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    consumption_evidence = activate_model_consumption_authority(args)
    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute():
        raise InferenceContractError(f"source video must be absolute: {source_requested}")
    retained_source: dict[str, Any] | None = None
    if consumption_evidence is None:
        source_path = _plain_file(
            source_requested.resolve(strict=True), label="source video"
        )
        source_consumer_path = source_path
    else:
        source_path = source_requested
        retained_source = _open_retained_source(
            source_path,
            expected_sha256=args.source_video_sha256,
            authority_raw=args.source_video_authority,
        )
        _RETAINED_SOURCE_LIFETIMES.append(retained_source)
        consumption_evidence = {
            **dict(consumption_evidence),
            "source_video_sha256": retained_source["sha256"],
            "source_video_physical_authority_digest": object_sha256(
                retained_source["authority"]
            ),
            "all_ranks_use_retained_source_fd": True,
        }
        source_consumer_path = retained_source["consumer_path"]
    output_path, inference_receipt_path = _resolve_output(args.output)
    base_only = bool(args.base_only)
    adapter: Optional[AdapterBundle] = None
    adapter_identity: Optional[dict[str, Any]] = None
    adapter_manifest_identity: Optional[dict[str, Any]] = None
    if not base_only:
        adapter = resolve_adapter_bundle(args.adapter_checkpoint)
        adapter_config = _read_json(adapter.adapter_config_path, label="adapter config")
        training_receipt = _read_json(adapter.training_receipt_path, label="training receipt")
        adapter_identity = validate_adapter_contract(
            adapter_config,
            training_receipt,
            expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        )
        expected_manifest_sha256 = getattr(
            args, "adapter_checkpoint_manifest_sha256", None
        )
        if adapter_identity.get("exploratory_full644") is True:
            if (
                expected_manifest_sha256 is None
                or getattr(args, "adapter_checkpoint_manifest", None) is None
                or consumption_evidence is None
            ):
                raise InferenceContractError(
                    "full644 adapter requires retained-FD consumption and its exact original manifest"
                )
            adapter_manifest_identity = validate_training_checkpoint_manifest(
                adapter,
                expected_sha256=expected_manifest_sha256,
                manifest_path=args.adapter_checkpoint_manifest,
            )
            if (
                adapter_manifest_identity["receipt_digest"]
                != adapter_identity["receipt_digest"]
            ):
                raise InferenceContractError(
                    "full644 manifest and training receipt identity differ"
                )
        elif expected_manifest_sha256 is not None:
            adapter_manifest_identity = validate_training_checkpoint_manifest(
                adapter,
                expected_sha256=expected_manifest_sha256,
                manifest_path=getattr(args, "adapter_checkpoint_manifest", None),
            )

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = trainer.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = validate_inference_checkpoint(
            args.checkpoint
        )
    except trainer.TrainingContractError as error:
        raise InferenceContractError(str(error)) from error
    if transformer_config["num_attention_heads"] % ULYSSES_SIZE:
        raise InferenceContractError("1.3B attention heads are not divisible by Ulysses=4")
    inference_file_hashes = validate_inference_source_files(bernini_root)
    trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from peft import __version__ as peft_version
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != MV2V_SYSTEM_PROMPT:
        raise InferenceContractError("runtime Bernini mv2v system prompt differs from training")
    if DEFAULT_NEG_PROMPT != DEFAULT_NEGATIVE_PROMPT:
        raise InferenceContractError("runtime Bernini default negative prompt differs")
    if adapter_identity is not None and transformers_version != adapter_identity["transformers_version"]:
        raise InferenceContractError(
            "Transformers version differs from adapter training: "
            f"training={adapter_identity['transformers_version']} runtime={transformers_version}"
        )
    if adapter_identity is not None and adapter_identity.get(
        "exploratory_full644"
    ) is True and (
        peft_version != FULL644_PEFT_VERSION
        or adapter_identity.get("peft_version") != FULL644_PEFT_VERSION
    ):
        raise InferenceContractError(
            "full644 runtime/training PEFT version must both be 0.19.1"
        )
    distributed = inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise InferenceContractError("four-rank inference requires AUH ROCm-visible GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    if consumption_evidence is not None:
        local_consumption_digest = object_sha256(consumption_evidence)
        gathered_consumption: list[Any] = [None] * distributed.world_size
        dist.all_gather_object(gathered_consumption, local_consumption_digest)
        if gathered_consumption != [
            local_consumption_digest
        ] * distributed.world_size:
            raise InferenceContractError(
                "four-rank model-consumption authority differs"
            )
        consumption_evidence = {
            **dict(consumption_evidence),
            "four_rank_attestation": {
                "world_size": distributed.world_size,
                "all_ranks_replayed_exact_fd_views": True,
                "rank_evidence_digest": local_consumption_digest,
                "ordered_rank_evidence_digests": gathered_consumption,
            },
        }
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    source_tensor, source_metadata = prepare_exact_source(
        source_consumer_path
    )
    if retained_source is None:
        source_sha256 = file_sha256(source_path)
    else:
        _replay_retained_source(retained_source)
        source_sha256 = retained_source["sha256"]
    adapter_sha256 = file_sha256(adapter.adapter_model_path) if adapter is not None else None
    if (
        adapter_manifest_identity is not None
        and adapter_sha256 != adapter_manifest_identity["adapter_model_sha256"]
    ):
        raise InferenceContractError(
            "adapter model bytes differ from checkpoint manifest before loading"
        )
    full_prompt = build_training_prompt(args.instruction, prompt_cleaner=prompt_clean)
    # Official BerniniRendererPipeline cleans the positive edit prompt but
    # tokenizes DEFAULT_NEG_PROMPT verbatim.  In particular, Wan prompt_clean
    # rewrites the Chinese full-width punctuation and changes token IDs, so do
    # not apply it to the unconditional branch.
    clean_negative = DEFAULT_NEGATIVE_PROMPT

    config_dir = bernini_root / "configs/bernini_renderer_wan21_1p3b"
    config = BerniniRendererConfig.from_pretrained(
        str(config_dir),
        local_files_only=True,
        **inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except trainer.TrainingContractError as error:
        raise InferenceContractError(str(error)) from error
    if float(config.shift) != FLOW_SHIFT or config.use_unipc is not True:
        raise InferenceContractError("renderer sampler was not constructed with UniPC shift 5")
    with serialized_model_load():
        base_model = BerniniRendererModel(config)
        base_model.requires_grad_(False)
        base_model.eval()
        if adapter is None:
            model = base_model
            adapter_tensor_count = 0
        else:
            if adapter_identity is None:
                raise InferenceContractError(
                    "adapter identity disappeared before strict model loading"
                )
            expected_targets = adapter_identity.get(
                "target_modules", expected_lora_target_modules()
            )
            model, adapter_tensor_count = _strict_load_and_merge_adapter(
                base_model,
                adapter,
                expected_targets,
                route_scope=adapter_identity.get("lora_route_scope"),
                require_zero_effect=bool(
                    adapter_identity.get("frozen_baseline", False)
                ),
            )
            if adapter_manifest_identity is not None:
                observed_manifest = validate_training_checkpoint_manifest(
                    adapter,
                    expected_sha256=adapter_manifest_identity["sha256"],
                    manifest_path=adapter_manifest_identity["path"],
                )
                if observed_manifest != adapter_manifest_identity:
                    raise InferenceContractError(
                        "adapter checkpoint changed during strict model loading"
                    )
                if (
                    file_sha256(adapter.adapter_model_path)
                    != adapter_manifest_identity["adapter_model_sha256"]
                ):
                    raise InferenceContractError(
                        "adapter model bytes changed during strict model loading"
                    )
        model.to(device)
        gc.collect()
        torch.cuda.empty_cache()
        trim_process_heap()
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **tokenizer_load_kwargs()
    )
    if tokenizer.padding_side != "right" or tokenizer.init_kwargs.get("fix_mistral_regex") is not True:
        raise InferenceContractError("tokenizer did not retain fix_mistral_regex/right-padding contract")
    input_ids, attention_mask = _tokenize_training_prompt(tokenizer, full_prompt)
    negative_ids, negative_mask = _tokenize_renderer_negative(
        tokenizer, clean_negative
    )

    vae_config = AutoencoderKLWan.load_config(
        str(checkpoint), subfolder="vae", local_files_only=True
    )
    expected_bucket = source_metadata["source_derived_bucket_hw"]
    expected_latent_shape = (
        1,
        int(vae_config["z_dim"]),
        LATENT_FRAME_COUNT,
        int(expected_bucket[0]) // 8,
        int(expected_bucket[1]) // 8,
    )
    vae = None
    if distributed.rank == 0:
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
        if tuple(int(value) for value in source_latent.shape) != expected_latent_shape:
            raise InferenceContractError(
                "source VAE latent shape mismatch: "
                f"{tuple(source_latent.shape)} != {expected_latent_shape}"
            )
        if source_latent.dtype != torch.float32:
            raise InferenceContractError("source VAE latent must be float32")
        vae.to("cpu")
    else:
        source_latent = torch.empty(expected_latent_shape, device=device, dtype=torch.float32)
    dist.broadcast(source_latent, src=0)
    del source_tensor
    gc.collect()
    torch.cuda.empty_cache()
    trim_process_heap()

    sampling = sampler_contract(steps=args.num_inference_steps, seed=args.seed)
    source_onset_solver_trace: Optional[dict[str, Any]] = None
    with torch.no_grad():
        if args.source_onset_policy == EVERY_STEP_SOURCE_ONSET_POLICY:
            with hard_phase0_source_trajectory_clamp(
                model.diff_dec,
                source_latent,
                expected_steps=args.num_inference_steps,
            ) as clamp_trace:
                generated_latent = model.sample(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    uncond_input_ids=negative_ids.to(device),
                    uncond_attention_mask=negative_mask.to(device),
                    image_vae_latents=None,
                    multi_video_vae_latents=[source_latent],
                    multi_image_vae_latents=None,
                    width=int(expected_bucket[1]),
                    height=int(expected_bucket[0]),
                    device=device,
                    **sampling,
                )
            source_onset_solver_trace = clamp_trace.as_dict()
        else:
            # Deliberately retain the pre-existing call path for none, hard1,
            # and ramp3.  In particular, policy=none does not inspect or wrap
            # the scheduler and returns the same model.sample object.
            generated_latent = model.sample(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                uncond_input_ids=negative_ids.to(device),
                uncond_attention_mask=negative_mask.to(device),
                image_vae_latents=None,
                multi_video_vae_latents=[source_latent],
                multi_image_vae_latents=None,
                width=int(expected_bucket[1]),
                height=int(expected_bucket[0]),
                device=device,
                **sampling,
            )
    if tuple(int(value) for value in generated_latent.shape) != expected_latent_shape:
        raise InferenceContractError(
            f"generated latent shape mismatch: {tuple(generated_latent.shape)} != {expected_latent_shape}"
        )
    generated_latent = apply_source_onset_policy(
        generated_latent,
        source_latent,
        args.source_onset_policy,
    )
    # Do not move four full renderer replicas back to host memory.  The AUH
    # allocation has 64 GiB per node; a simultaneous device->CPU copy on all
    # ranks creates a needless post-sampling host peak and can be OOM-killed
    # after a completely successful denoise.  The renderer is no longer used,
    # so release it in place and leave only rank 0's VAE decode below.
    del model
    del base_model
    del source_latent
    gc.collect()
    torch.cuda.empty_cache()

    if distributed.rank == 0:
        if vae is None:
            raise InferenceContractError("rank 0 VAE was not retained for decode")
        vae.to(device)
        with torch.no_grad():
            output = _vae_decode(vae, generated_latent)
        vae.to("cpu")
        if tuple(int(value) for value in output.shape) != (
            FRAME_COUNT,
            int(expected_bucket[0]),
            int(expected_bucket[1]),
            3,
        ):
            raise InferenceContractError(f"decoded output shape mismatch: {tuple(output.shape)}")
        retained_output: dict[str, Any] | None = None
        retained_evidence: dict[str, Any] | None = None
        anonymous_evidence: dict[str, Any] | None = None
        try:
            if _ACTIVE_INHERITED_FDS is None:
                temporary_output = output_path.with_name(
                    f".{output_path.stem}.tmp-{os.getpid()}"
                    f"{output_path.suffix}"
                )
                if temporary_output.exists() or temporary_output.is_symlink():
                    raise InferenceContractError(
                        f"stale temporary output exists: {temporary_output}"
                    )
                save_output(output, str(temporary_output), fps=int(FPS))
                os.replace(temporary_output, output_path)
                encoded_path = output_path
                output_sha256 = file_sha256(output_path)
                output_identity = None
            else:
                retained_output = _create_retained_encoded_output(output_path)
                _save_output_with_exact_ffmpeg_fds(
                    save_output,
                    output,
                    retained_output,
                    fps=int(FPS),
                )
                anonymous_evidence = _finalize_retained_encoded_output(
                    retained_output
                )
                encoded_path = anonymous_evidence["read_path"]
                output_sha256 = anonymous_evidence["sha256"]
                output_identity = None

            # Decode the held encoded inode so the receipt binds the actual
            # container bytes, not only the in-memory VAE result.
            materialize_vae = (
                exact_video_materializer.install_exact_local_video_materializer()
            )

            encoded_frames, encoded_fps, encoded_hw = (
                materialize_vae._decode_exact_video(encoded_path)
            )
            validate_exact_video_metadata(
                int(encoded_frames.shape[0]), encoded_fps
            )
            if tuple(encoded_hw) != tuple(expected_bucket):
                raise InferenceContractError(
                    f"encoded output geometry mismatch: {encoded_hw} != "
                    f"{tuple(expected_bucket)}"
                )
            if retained_output is not None:
                if anonymous_evidence is None:
                    raise InferenceContractError(
                        "retained output lacks anonymous publication evidence"
                    )
                retained_evidence = _publish_retained_encoded_output(
                    retained_output, anonymous_evidence
                )
                output_sha256 = retained_evidence["sha256"]
                output_identity = retained_evidence["identity"]
            receipt = build_inference_receipt(
                args=args,
                source_path=source_path,
                source_sha256=source_sha256,
                source_metadata=source_metadata,
                output_path=output_path,
                output_sha256=output_sha256,
                adapter=adapter,
                adapter_sha256=adapter_sha256,
                adapter_identity=adapter_identity,
                adapter_manifest_identity=adapter_manifest_identity,
                adapter_tensor_count=adapter_tensor_count,
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                inference_file_hashes=inference_file_hashes,
                runtime_versions={
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "transformers": transformers_version,
                    "diffusers": diffusers_version,
                    # Both matched arms record the same installed PEFT
                    # producer version.  Omitting it from the frozen-base arm
                    # would make an otherwise matched runtime comparison
                    # structurally unequal.
                    "peft": peft_version,
                },
                source_onset_solver_trace=source_onset_solver_trace,
                model_consumption_evidence=consumption_evidence,
                output_identity=output_identity,
                output_publication_evidence=retained_evidence,
                source_file_authority=(
                    retained_source["authority"]
                    if retained_source is not None else None
                ),
            )
            _atomic_write_json(inference_receipt_path, receipt)
            print(canonical_json_bytes(receipt).decode("utf-8"), flush=True)
            if retained_output is not None:
                if retained_evidence is None:
                    raise InferenceContractError(
                        "retained output lacks terminal publication evidence"
                    )
                _replay_retained_encoded_output(
                    retained_output, retained_evidence
                )
        finally:
            if retained_output is not None:
                _close_retained_encoded_output(retained_output)

    if retained_source is not None:
        try:
            _replay_retained_source(retained_source)
        finally:
            _close_retained_source(retained_source)
            _RETAINED_SOURCE_LIFETIMES.remove(retained_source)
        retained_source = None
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

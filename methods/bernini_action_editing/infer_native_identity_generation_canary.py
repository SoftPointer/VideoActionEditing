#!/usr/bin/env python3
"""Frozen Bernini-R native identity-conditioned generation canary.

This runner deliberately does *not* modify the existing action-editing or
training paths.  It compares three native Bernini-R 1.3B sampling contracts on
the same exact-81-frame source, action description, spatial bucket, Gaussian
seed, and 40-step UniPC schedule:

``t2v``
    Text-only generation (``t2v_apg``).  This is the action-prior arm and sees
    no source-content latent.

``r2v``
    Reference-to-video generation (``r2v_apg``) from five frames derived
    inside the runner from source indices ``0,20,40,60,80``.  Every frame is
    independently encoded by the Wan VAE as ``[1,C,1,H,W]``; no temporal-video
    latent is sliced to manufacture an image reference.

``rv2v``
    Reference + video generation (``rv2v``) from the full source video and
    four independently encoded source frames at ``0,27,53,80``.

All three targets start from fresh Gaussian noise inside the unmodified
``GEN_Wanx22.sample`` implementation.  The only external semantic inputs are
the source video and one complete source-content caption containing the new
action.  There is no target, mask, flow, pose, track, trajectory, first-frame
anchor, external reference, or custom target-noise input.

For provenance, each sampling call is surrounded by a read-only observer on
the pinned ``bernini.models.wan_diffusion.randn_tensor`` module global.  The
observer forwards the original call unchanged, returns the original tensor
object to the sampler, and saves an exact CPU copy plus raw-value hash.  It
never injects or replaces sampler noise and always restores the module global
in ``finally``.

The runner accepts an arm subset so one eight-GPU node can execute
``t2v+r2v`` and ``rv2v`` concurrently as two legal Ulysses-4 jobs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as legacy  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import infer_source_value_residual_oracle as value_audit  # noqa: E402


SCHEMA_VERSION = "bernini-native-identity-generation-canary-v2"
METHOD = "frozen-bernini-native-identity-generation-canary"
ARM_ORDER = ("t2v", "r2v", "rv2v")
ARM_GUIDANCE_MODES = {
    "t2v": "t2v_apg",
    "r2v": "r2v_apg",
    "rv2v": "rv2v",
}
ARM_TRAINING_TASK_NAMES = {
    "t2v": "t2v",
    "r2v": "r2v",
    # Bernini exposes this arm as ``rv2v`` at inference, but the renderer was
    # trained with the task key ``vr2v``.  The guidance-mode and task-prefix
    # namespaces are deliberately different; collapsing them caused the first
    # AUH canary to fail closed before model loading.
    "rv2v": "vr2v",
}
ARM_REFERENCE_COUNTS = {"t2v": 0, "r2v": 5, "rv2v": 4}
ARM_VIDEO_COUNTS = {"t2v": 0, "r2v": 0, "rv2v": 1}

TASK_SYSTEM_PROMPTS = {
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "vr2v": "You are a helpful assistant specialized in video editing with reference.",
}
TASK_BINDING_CLAUSES = {
    "t2v": "",
    "r2v": (
        "The same main subject is shown across image0, image1, image2, image3, "
        "and image4. Preserve that subject's identity across those viewpoints. "
    ),
    "rv2v": (
        "image0, image1, image2, and image3 are frames sampled from the source "
        "video. Preserve the source subject's identity, background, and camera "
        "while applying the requested new action. "
    ),
}

FRAME_COUNT = 81
LATENT_FRAME_COUNT = 21
FPS = 25
NUM_INFERENCE_STEPS = 40
DEFAULT_SEED = 2027
ULYSSES_SIZE = 4
FLOW_SHIFT = 5.0
OMEGA_VIDEO = 1.25
OMEGA_IMAGE = 4.5
OMEGA_TEXT = 4.0
OMEGA_SCALE = 0.8
ETA = 0.5
NORM_THRESHOLD = (50.0, 50.0)
MOMENTUM = 0.0
TARGET_INITIALIZATION = "official_gen_wanx22_fresh_gaussian"
T2V_RESOURCE_LIFECYCLE_CONTRACT = {
    "schema_version": "bernini-native-t2v-resource-lifecycle-v4",
    "serialized_host_checkpoint_load_required": True,
    "renderer_deserialized_and_moved_to_rank_gpu_under_lock": True,
    "host_allocator_trim_called_before_load_lock_release": True,
    "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup": True,
    "world4_load_completion_receipt_before_native_sampling": True,
    "renderer_retired_before_rank_zero_vae_load": True,
    "world4_renderer_retirement_barrier_before_rank_zero_vae_load": True,
    "t2v_vae_weights_loaded_before_sampling": False,
    "t2v_vae_decode_rank": 0,
    "sampling_model_and_vae_not_host_resident_concurrently_for_t2v": True,
    "t2v_text_encoder_gpu_residency_required": True,
    "t2v_text_encoder_cpu_offload_bypass_active": True,
    "t2v_text_encoder_retired_only_with_renderer": True,
}
WORLD4_LOAD_COMPLETION_GATE_SCHEMA = (
    "bernini-native-world4-renderer-load-completion-gate-v1"
)
_WORLD4_LOAD_COMPLETION_GATE_KEYS = {
    "schema_version",
    "world_size",
    "hostname",
    "ranks",
    "renderer_gpu_resident_trimmed_monotonic_ns_by_rank",
    "load_completion_barrier_returned_monotonic_ns_by_rank",
    "source_tokenizer_setup_entered_monotonic_ns_by_rank",
    "native_sampling_entered_monotonic_ns_by_rank",
    "world4_barrier_completed_before_source_tokenizer_setup",
    "all_four_renderer_loads_complete_before_any_source_tokenizer_setup",
    "all_four_renderer_loads_complete_before_first_native_sampling",
}
T2V_TEXT_ENCODER_GPU_RESIDENCY_GATE_SCHEMA = (
    "bernini-native-t2v-text-encoder-gpu-residency-gate-v2"
)
T2V_GPU_MEMORY_LIMIT_GIB = 52
T2V_GPU_MEMORY_LIMIT_BYTES = T2V_GPU_MEMORY_LIMIT_GIB * 1024**3
_T2V_TEXT_ENCODER_GPU_RESIDENCY_GATE_KEYS = {
    "schema_version",
    "world_size",
    "hostname",
    "ranks",
    "module_path",
    "rank_evidence",
    "all_rank_exactly_one_cpu_offload_request_suppressed",
    "all_rank_zero_successful_cpu_materializations",
    "all_rank_gpu_resident_before_and_after_sampling",
    "all_rank_storage_fingerprint_unchanged",
    "all_rank_guard_method_restored",
    "all_rank_peak_reserved_within_52_gib",
}

_SAFE_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class NativeIdentityCanaryError(RuntimeError):
    """Raised before an ambiguous or non-native canary artifact is published."""


@contextmanager
def _serialized_host_checkpoint_load() -> Any:
    """Serialize WORLD4 host deserialization until weights reside on the GPU.

    The generic-action r11 release sets both environment variables below and
    authenticates the lock as an empty, read-only file in its node-local task
    scratch.  Other historical callers retain their former behaviour unless
    they explicitly opt into the required-lock contract.
    """

    required = os.environ.get("NATIVE_SERIALIZED_HOST_LOAD_REQUIRED")
    if required not in (None, "1"):
        raise NativeIdentityCanaryError(
            "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED differs"
        )
    value = os.environ.get("NATIVE_V_AXIS_LOAD_LOCK")
    if value is None:
        if required == "1":
            raise NativeIdentityCanaryError(
                "NATIVE_V_AXIS_LOAD_LOCK is required for serialized host load"
            )
        yield
        return
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise NativeIdentityCanaryError("serialized checkpoint-load lock differs")
    rank = os.environ.get("RANK", "unknown")
    with path.open("rb") as handle:
        print(f"[native-load-lock] rank={rank} waiting", flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            print(f"[native-load-lock] rank={rank} acquired", flush=True)
            yield
            print(f"[native-load-lock] rank={rank} gpu-resident-trimmed", flush=True)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _trim_host_allocator() -> bool:
    """Return unused deserialization arenas to the Slurm memory cgroup."""

    import ctypes

    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    malloc_trim = libc.malloc_trim
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)
    return True


def _load_frozen_renderer_gpu_resident_serialized(
    model_factory: Callable[[Any], Any], config: Any, device: Any
) -> Any:
    """Deserialize one renderer and retire host arenas inside the shared lock."""

    with _serialized_host_checkpoint_load():
        model = model_factory(config)
        model.requires_grad_(False)
        model.eval()
        model.to(device)
        if not _trim_host_allocator():
            raise NativeIdentityCanaryError(
                "host allocator trim after model load failed"
            )
    return model


def _module_storage_fingerprint(module: Any) -> str:
    """Hash tensor metadata and storage identity without reading tensor bytes."""

    rows: list[dict[str, Any]] = []
    for kind, iterator in (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, value in iterator:
            rows.append(
                {
                    "kind": kind,
                    "name": str(name),
                    "shape": [int(item) for item in value.shape],
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "numel": int(value.numel()),
                    "data_ptr": int(value.data_ptr()),
                }
            )
    if not rows:
        raise NativeIdentityCanaryError("T2V text encoder has no tensor storage")
    return legacy.object_sha256(rows)


def _module_parameter_device(module: Any) -> str:
    devices = {str(value.device) for value in module.parameters()}
    if len(devices) != 1:
        raise NativeIdentityCanaryError("T2V text encoder parameter device differs")
    return next(iter(devices))


def _linux_process_memory_kib() -> dict[str, int]:
    """Read the current process RSS/HWM used by the Slurm memory-cgroup audit."""

    status = Path("/proc/self/status")
    if not status.is_file() or status.is_symlink():
        raise NativeIdentityCanaryError("Linux process memory status is unavailable")
    values: dict[str, int] = {}
    for line in status.read_text(encoding="ascii").splitlines():
        if line.startswith("VmRSS:") or line.startswith("VmHWM:"):
            key, amount, unit = line.split()
            if unit != "kB" or not amount.isdigit():
                raise NativeIdentityCanaryError("Linux process memory status differs")
            values[key[:-1].lower() + "_kib"] = int(amount)
    if set(values) != {"vmrss_kib", "vmhwm_kib"} or any(
        value <= 0 for value in values.values()
    ):
        raise NativeIdentityCanaryError("Linux process memory evidence is incomplete")
    return values


def _requested_to_device(args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
    if "device" in kwargs:
        return kwargs["device"]
    return args[0] if args else None


def _reset_cuda_peak_memory(device: Any) -> None:
    import torch

    torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_memory_bytes(device: Any) -> dict[str, int]:
    import torch

    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu_total_memory_bytes": int(properties.total_memory),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


@contextmanager
def _t2v_text_encoder_rank_gpu_residency(
    model: Any,
    *,
    arm: str,
    device: Any,
    memory_reader: Callable[[], Mapping[str, int]] = _linux_process_memory_kib,
    cuda_peak_reset: Callable[[Any], None] = _reset_cuda_peak_memory,
    cuda_memory_reader: Callable[[Any], Mapping[str, int]] = (
        _cuda_peak_memory_bytes
    ),
) -> Any:
    """Keep only the T2V text encoder on its rank GPU during native sampling.

    Bernini's official ``model.sample`` first moves the encoder to the rank GPU,
    computes the positive/negative embeddings, and then calls the exact
    positional ``t5_text_encoder.to("cpu")`` before denoising.  In the r10
    formal log, early ranks had already entered native sampling while the last
    rank was still deserializing and was then cgroup-OOM-killed; that ordering
    proves an overlap, not a unique causal attribution.  Avoiding four T5 host
    materializations removes one large deterministic contributor to that
    overlap.  The r11 release keeps the official sampler and intercepts only
    that one known offload request.  Every other ``to`` call is delegated
    unchanged.
    """

    required = os.environ.get("NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED")
    if required not in (None, "1"):
        raise NativeIdentityCanaryError(
            "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED differs"
        )
    evidence: dict[str, Any] = {
        "guard_required": required == "1",
        "guard_active": False,
        "module_path": "model.t5_text_encoder",
    }
    if required is None:
        yield evidence
        return
    if arm != "t2v":
        raise NativeIdentityCanaryError(
            "T2V text-encoder residency guard cannot wrap a non-T2V arm"
        )
    encoder = getattr(model, "t5_text_encoder", None)
    if encoder is None or not callable(getattr(encoder, "to", None)):
        raise NativeIdentityCanaryError("T2V text encoder is unavailable")
    if vars(encoder).get("_native_t2v_gpu_residency_guard_active") is not None:
        raise NativeIdentityCanaryError("T2V text-encoder residency guard is reentered")

    expected_device = str(device)
    before_device = _module_parameter_device(encoder)
    before_fingerprint = _module_storage_fingerprint(encoder)
    if before_device != expected_device:
        raise NativeIdentityCanaryError(
            "T2V text encoder is not rank-GPU resident before sampling"
        )

    sentinel = object()
    previous_instance_to = vars(encoder).get("to", sentinel)
    original_to = encoder.to
    calls = {"cpu": 0, "delegated": 0}
    cuda_peak_reset(device)

    def guarded_to(*args: Any, **kwargs: Any) -> Any:
        requested = _requested_to_device(args, kwargs)
        if args == ("cpu",) and not kwargs:
            calls["cpu"] += 1
            return encoder
        if requested is not None and str(requested) == "cpu":
            raise NativeIdentityCanaryError(
                "unexpected T2V text-encoder CPU offload signature"
            )
        calls["delegated"] += 1
        return original_to(*args, **kwargs)

    setattr(encoder, "_native_t2v_gpu_residency_guard_active", True)
    setattr(encoder, "to", guarded_to)
    evidence["guard_active"] = True
    completed = False
    try:
        yield evidence
        completed = True
    finally:
        if previous_instance_to is sentinel:
            delattr(encoder, "to")
        else:
            setattr(encoder, "to", previous_instance_to)
        delattr(encoder, "_native_t2v_gpu_residency_guard_active")

    if not completed:
        return
    after_device = _module_parameter_device(encoder)
    after_fingerprint = _module_storage_fingerprint(encoder)
    memory = dict(memory_reader())
    gpu_memory = dict(cuda_memory_reader(device))
    method_restored = (
        "to" not in vars(encoder)
        if previous_instance_to is sentinel
        else vars(encoder).get("to") is previous_instance_to
    )
    if (
        calls["cpu"] != 1
        or calls["delegated"] < 1
        or after_device != expected_device
        or after_fingerprint != before_fingerprint
        or not method_restored
        or set(memory) != {"vmrss_kib", "vmhwm_kib"}
        or any(type(value) is not int or value <= 0 for value in memory.values())
        or set(gpu_memory)
        != {
            "gpu_total_memory_bytes",
            "gpu_peak_allocated_bytes",
            "gpu_peak_reserved_bytes",
        }
        or any(type(value) is not int or value <= 0 for value in gpu_memory.values())
        or gpu_memory["gpu_total_memory_bytes"] < T2V_GPU_MEMORY_LIMIT_BYTES
        or gpu_memory["gpu_peak_allocated_bytes"]
        > gpu_memory["gpu_peak_reserved_bytes"]
        or gpu_memory["gpu_peak_reserved_bytes"] >= T2V_GPU_MEMORY_LIMIT_BYTES
    ):
        raise NativeIdentityCanaryError(
            "T2V text-encoder GPU-residency evidence differs"
        )
    evidence.update(
        {
            "exact_positional_cpu_offload_request_only": True,
            "cpu_offload_requests_observed": calls["cpu"],
            "cpu_offload_requests_suppressed": calls["cpu"],
            "successful_cpu_materializations": 0,
            "delegated_to_requests": calls["delegated"],
            "parameter_device_before": before_device,
            "parameter_device_after": after_device,
            "storage_fingerprint_before": before_fingerprint,
            "storage_fingerprint_after": after_fingerprint,
            "guard_method_restored": method_restored,
            "gpu_memory_limit_gib": T2V_GPU_MEMORY_LIMIT_GIB,
            "gpu_memory_limit_bytes": T2V_GPU_MEMORY_LIMIT_BYTES,
            "gpu_peak_reserved_within_limit": True,
            **memory,
            **gpu_memory,
        }
    )


def _world4_phase_rows(
    dist_module: Any,
    *,
    rank: int,
    world_size: int,
    phase: str,
    monotonic_ns: int,
    hostname: str,
) -> list[Mapping[str, Any]]:
    """Gather one internally timestamped lifecycle phase from every rank."""

    if (
        world_size != 4
        or rank not in range(world_size)
        or not isinstance(phase, str)
        or not phase
        or type(monotonic_ns) is not int
        or monotonic_ns <= 0
        or not isinstance(hostname, str)
        or not hostname
    ):
        raise NativeIdentityCanaryError("WORLD4 lifecycle phase differs")
    local = {
        "rank": rank,
        "hostname": hostname,
        "phase": phase,
        "monotonic_ns": monotonic_ns,
    }
    rows: list[Any] = [None] * world_size
    dist_module.all_gather_object(rows, local)
    if (
        any(not isinstance(row, Mapping) for row in rows)
        or sorted(int(row.get("rank", -1)) for row in rows) != list(range(4))
        or any(row.get("phase") != phase for row in rows)
        or any(row.get("hostname") != hostname for row in rows)
        or any(
            type(row.get("monotonic_ns")) is not int
            or int(row["monotonic_ns"]) <= 0
            for row in rows
        )
    ):
        raise NativeIdentityCanaryError(f"WORLD4 {phase} evidence differs")
    return sorted((dict(row) for row in rows), key=lambda row: int(row["rank"]))


def _world4_renderer_load_completion_barrier(
    dist_module: Any,
    *,
    rank: int,
    world_size: int,
    renderer_gpu_resident_trimmed_monotonic_ns: int,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    hostname: Optional[str] = None,
) -> dict[str, Any]:
    """Block every rank after load/GPU/trim and before source/tokenizer setup.

    The literal WORLD4 barrier is the first distributed operation after the
    caller records completion of the serialized load.  No source video,
    tokenizer, or sampler setup is allowed until this helper returns.
    """

    host = socket.gethostname() if hostname is None else hostname
    if world_size != 4 or rank not in range(world_size):
        raise NativeIdentityCanaryError("renderer load gate requires WORLD4")
    if (
        type(renderer_gpu_resident_trimmed_monotonic_ns) is not int
        or renderer_gpu_resident_trimmed_monotonic_ns <= 0
    ):
        raise NativeIdentityCanaryError("renderer load completion time differs")

    # This must remain the first distributed call after model load -> GPU ->
    # malloc_trim.  Early ranks wait here while the final rank deserializes.
    dist_module.barrier()
    barrier_returned_ns = monotonic_ns()
    load_rows = _world4_phase_rows(
        dist_module,
        rank=rank,
        world_size=world_size,
        phase="renderer_gpu_resident_trimmed",
        monotonic_ns=renderer_gpu_resident_trimmed_monotonic_ns,
        hostname=host,
    )
    barrier_rows = _world4_phase_rows(
        dist_module,
        rank=rank,
        world_size=world_size,
        phase="load_completion_barrier_returned",
        monotonic_ns=barrier_returned_ns,
        hostname=host,
    )
    # This marker is the entry boundary.  The caller performs no actual source
    # or tokenizer work until all four markers have been gathered.
    setup_entered_ns = monotonic_ns()
    setup_rows = _world4_phase_rows(
        dist_module,
        rank=rank,
        world_size=world_size,
        phase="source_tokenizer_setup_entered",
        monotonic_ns=setup_entered_ns,
        hostname=host,
    )
    evidence = {
        "schema_version": WORLD4_LOAD_COMPLETION_GATE_SCHEMA,
        "world_size": 4,
        "hostname": host,
        "ranks": [0, 1, 2, 3],
        "renderer_gpu_resident_trimmed_monotonic_ns_by_rank": [
            int(row["monotonic_ns"]) for row in load_rows
        ],
        "load_completion_barrier_returned_monotonic_ns_by_rank": [
            int(row["monotonic_ns"]) for row in barrier_rows
        ],
        "source_tokenizer_setup_entered_monotonic_ns_by_rank": [
            int(row["monotonic_ns"]) for row in setup_rows
        ],
        "native_sampling_entered_monotonic_ns_by_rank": None,
        "world4_barrier_completed_before_source_tokenizer_setup": True,
        "all_four_renderer_loads_complete_before_any_source_tokenizer_setup": True,
        "all_four_renderer_loads_complete_before_first_native_sampling": False,
    }
    _validate_world4_load_completion_gate(evidence, sampling_required=False)
    return evidence


def _complete_world4_load_completion_gate_before_sampling(
    dist_module: Any,
    evidence: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> dict[str, Any]:
    """Seal all-rank sampling-entry evidence before the native sampler runs."""

    _validate_world4_load_completion_gate(evidence, sampling_required=False)
    sampling_rows = _world4_phase_rows(
        dist_module,
        rank=rank,
        world_size=world_size,
        phase="native_sampling_entered",
        monotonic_ns=monotonic_ns(),
        hostname=str(evidence["hostname"]),
    )
    completed = dict(evidence)
    completed["native_sampling_entered_monotonic_ns_by_rank"] = [
        int(row["monotonic_ns"]) for row in sampling_rows
    ]
    completed["all_four_renderer_loads_complete_before_first_native_sampling"] = True
    _validate_world4_load_completion_gate(completed, sampling_required=True)
    return completed


def _validate_world4_load_completion_gate(
    value: Mapping[str, Any], *, sampling_required: bool
) -> dict[str, Any]:
    """Reject stale schemas, missing ranks, and impossible lifecycle ordering."""

    if not isinstance(value, Mapping) or set(value) != _WORLD4_LOAD_COMPLETION_GATE_KEYS:
        raise NativeIdentityCanaryError("WORLD4 load-completion gate closure differs")
    loads = value.get("renderer_gpu_resident_trimmed_monotonic_ns_by_rank")
    barriers = value.get("load_completion_barrier_returned_monotonic_ns_by_rank")
    setups = value.get("source_tokenizer_setup_entered_monotonic_ns_by_rank")
    samplings = value.get("native_sampling_entered_monotonic_ns_by_rank")
    if (
        value.get("schema_version") != WORLD4_LOAD_COMPLETION_GATE_SCHEMA
        or value.get("world_size") != 4
        or value.get("ranks") != [0, 1, 2, 3]
        or not isinstance(value.get("hostname"), str)
        or not value.get("hostname")
        or any(
            not isinstance(rows, list)
            or len(rows) != 4
            or any(type(item) is not int or item <= 0 for item in rows)
            for rows in (loads, barriers, setups)
        )
        or value.get("world4_barrier_completed_before_source_tokenizer_setup")
        is not True
        or value.get(
            "all_four_renderer_loads_complete_before_any_source_tokenizer_setup"
        )
        is not True
        or max(loads) > min(barriers)
        or max(loads) >= min(setups)
        or max(barriers) > min(setups)
    ):
        raise NativeIdentityCanaryError("WORLD4 load-completion ordering differs")
    if sampling_required:
        if (
            not isinstance(samplings, list)
            or len(samplings) != 4
            or any(type(item) is not int or item <= 0 for item in samplings)
            or value.get(
                "all_four_renderer_loads_complete_before_first_native_sampling"
            )
            is not True
            or max(loads) >= min(samplings)
            or max(setups) > min(samplings)
        ):
            raise NativeIdentityCanaryError(
                "WORLD4 load completion was not proven before native sampling"
            )
    elif (
        samplings is not None
        or value.get("all_four_renderer_loads_complete_before_first_native_sampling")
        is not False
    ):
        raise NativeIdentityCanaryError("unsealed WORLD4 sampling evidence differs")
    return dict(value)


def _validate_world4_t2v_text_encoder_gpu_residency_gate(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that every T2V rank suppressed exactly one host offload."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _T2V_TEXT_ENCODER_GPU_RESIDENCY_GATE_KEYS
        or value.get("schema_version")
        != T2V_TEXT_ENCODER_GPU_RESIDENCY_GATE_SCHEMA
        or value.get("world_size") != 4
        or value.get("ranks") != [0, 1, 2, 3]
        or value.get("module_path") != "model.t5_text_encoder"
        or not isinstance(value.get("hostname"), str)
        or not value.get("hostname")
    ):
        raise NativeIdentityCanaryError(
            "WORLD4 T2V text-encoder residency closure differs"
        )
    rows = value.get("rank_evidence")
    if (
        not isinstance(rows, list)
        or len(rows) != 4
        or any(not isinstance(row, Mapping) for row in rows)
        or sorted(int(row.get("rank", -1)) for row in rows) != list(range(4))
    ):
        raise NativeIdentityCanaryError(
            "WORLD4 T2V text-encoder residency ranks differ"
        )
    for row in rows:
        rank = int(row["rank"])
        expected_device = f"cuda:{rank}"
        if (
            set(row)
            != {
                "rank",
                "local_rank",
                "hostname",
                "guard_required",
                "guard_active",
                "module_path",
                "exact_positional_cpu_offload_request_only",
                "cpu_offload_requests_observed",
                "cpu_offload_requests_suppressed",
                "successful_cpu_materializations",
                "delegated_to_requests",
                "parameter_device_before",
                "parameter_device_after",
                "storage_fingerprint_before",
                "storage_fingerprint_after",
                "guard_method_restored",
                "vmrss_kib",
                "vmhwm_kib",
                "gpu_memory_limit_gib",
                "gpu_memory_limit_bytes",
                "gpu_total_memory_bytes",
                "gpu_peak_allocated_bytes",
                "gpu_peak_reserved_bytes",
                "gpu_peak_reserved_within_limit",
            }
            or row.get("local_rank") != rank
            or row.get("hostname") != value["hostname"]
            or row.get("guard_required") is not True
            or row.get("guard_active") is not True
            or row.get("module_path") != "model.t5_text_encoder"
            or row.get("exact_positional_cpu_offload_request_only") is not True
            or row.get("cpu_offload_requests_observed") != 1
            or row.get("cpu_offload_requests_suppressed") != 1
            or row.get("successful_cpu_materializations") != 0
            or type(row.get("delegated_to_requests")) is not int
            or int(row["delegated_to_requests"]) < 1
            or row.get("parameter_device_before") != expected_device
            or row.get("parameter_device_after") != expected_device
            or not isinstance(row.get("storage_fingerprint_before"), str)
            or _SHA256.fullmatch(str(row["storage_fingerprint_before"])) is None
            or row.get("storage_fingerprint_after")
            != row.get("storage_fingerprint_before")
            or row.get("guard_method_restored") is not True
            or type(row.get("vmrss_kib")) is not int
            or int(row["vmrss_kib"]) <= 0
            or type(row.get("vmhwm_kib")) is not int
            or int(row["vmhwm_kib"]) < int(row["vmrss_kib"])
            or row.get("gpu_memory_limit_gib") != T2V_GPU_MEMORY_LIMIT_GIB
            or row.get("gpu_memory_limit_bytes") != T2V_GPU_MEMORY_LIMIT_BYTES
            or type(row.get("gpu_total_memory_bytes")) is not int
            or int(row["gpu_total_memory_bytes"]) < T2V_GPU_MEMORY_LIMIT_BYTES
            or type(row.get("gpu_peak_allocated_bytes")) is not int
            or int(row["gpu_peak_allocated_bytes"]) <= 0
            or type(row.get("gpu_peak_reserved_bytes")) is not int
            or int(row["gpu_peak_reserved_bytes"])
            < int(row["gpu_peak_allocated_bytes"])
            or int(row["gpu_peak_reserved_bytes"]) >= T2V_GPU_MEMORY_LIMIT_BYTES
            or row.get("gpu_peak_reserved_within_limit") is not True
        ):
            raise NativeIdentityCanaryError(
                f"WORLD4 T2V text-encoder residency rank {rank} differs"
            )
    for field in (
        "all_rank_exactly_one_cpu_offload_request_suppressed",
        "all_rank_zero_successful_cpu_materializations",
        "all_rank_gpu_resident_before_and_after_sampling",
        "all_rank_storage_fingerprint_unchanged",
        "all_rank_guard_method_restored",
        "all_rank_peak_reserved_within_52_gib",
    ):
        if value.get(field) is not True:
            raise NativeIdentityCanaryError(
                f"WORLD4 T2V text-encoder residency proof differs at {field}"
            )
    return {**dict(value), "rank_evidence": [dict(row) for row in rows]}


def _world4_t2v_text_encoder_gpu_residency_gate(
    dist_module: Any,
    local_evidence: Mapping[str, Any],
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    hostname: Optional[str] = None,
) -> dict[str, Any]:
    """Gather the process-local T5 residency proof from all four ranks."""

    host = socket.gethostname() if hostname is None else hostname
    if world_size != 4 or rank not in range(4) or local_rank not in range(4):
        raise NativeIdentityCanaryError("T2V residency gate requires WORLD4")
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": host,
        **dict(local_evidence),
    }
    rows: list[Any] = [None] * world_size
    dist_module.all_gather_object(rows, local)
    if any(not isinstance(row, Mapping) for row in rows):
        raise NativeIdentityCanaryError("T2V residency evidence gather differs")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["rank"]))
    gate = {
        "schema_version": T2V_TEXT_ENCODER_GPU_RESIDENCY_GATE_SCHEMA,
        "world_size": 4,
        "hostname": host,
        "ranks": [0, 1, 2, 3],
        "module_path": "model.t5_text_encoder",
        "rank_evidence": ordered,
        "all_rank_exactly_one_cpu_offload_request_suppressed": True,
        "all_rank_zero_successful_cpu_materializations": True,
        "all_rank_gpu_resident_before_and_after_sampling": True,
        "all_rank_storage_fingerprint_unchanged": True,
        "all_rank_guard_method_restored": True,
        "all_rank_peak_reserved_within_52_gib": True,
    }
    return _validate_world4_t2v_text_encoder_gpu_residency_gate(gate)


def _resource_lifecycle_receipt(
    *,
    t2v_vae_deferred_until_post_sampling: bool,
    world4_load_completion_gate: Mapping[str, Any],
    world4_t2v_text_encoder_gpu_residency_gate: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report only lifecycle facts established by the active environment."""

    load_lock_configured = os.environ.get("NATIVE_V_AXIS_LOAD_LOCK") is not None
    gate = _validate_world4_load_completion_gate(
        world4_load_completion_gate, sampling_required=True
    )
    residency_required = (
        os.environ.get("NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED") == "1"
    )
    residency_gate = (
        _validate_world4_t2v_text_encoder_gpu_residency_gate(
            world4_t2v_text_encoder_gpu_residency_gate
        )
        if residency_required
        else None
    )
    return {
        "schema_version": "bernini-native-t2v-resource-lifecycle-v4",
        "serialized_host_checkpoint_load_required": (
            os.environ.get("NATIVE_SERIALIZED_HOST_LOAD_REQUIRED") == "1"
        ),
        "renderer_deserialized_and_moved_to_rank_gpu_under_lock": (
            load_lock_configured
        ),
        "host_allocator_trim_called_before_load_lock_release": (
            load_lock_configured
        ),
        "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup": True,
        "world4_load_completion_receipt_before_native_sampling": True,
        "renderer_retired_before_rank_zero_vae_load": True,
        "world4_renderer_retirement_barrier_before_rank_zero_vae_load": True,
        "t2v_vae_weights_loaded_before_sampling": (
            not t2v_vae_deferred_until_post_sampling
        ),
        "t2v_vae_decode_rank": 0,
        "sampling_model_and_vae_not_host_resident_concurrently_for_t2v": (
            t2v_vae_deferred_until_post_sampling
        ),
        "t2v_text_encoder_gpu_residency_required": residency_required,
        "t2v_text_encoder_cpu_offload_bypass_active": residency_required,
        "t2v_text_encoder_retired_only_with_renderer": residency_required,
        "world4_load_completion_gate": gate,
        "world4_t2v_text_encoder_gpu_residency_gate": residency_gate,
    }


def validate_t2v_resource_lifecycle(
    value: Mapping[str, Any], *, require_serialized_load: bool
) -> dict[str, Any]:
    """Validate the fixed v4 contract and its dynamic WORLD4 ordering proof."""

    if not isinstance(value, Mapping) or set(value) != (
        set(T2V_RESOURCE_LIFECYCLE_CONTRACT)
        | {
            "world4_load_completion_gate",
            "world4_t2v_text_encoder_gpu_residency_gate",
        }
    ):
        raise NativeIdentityCanaryError("T2V resource lifecycle closure differs")
    for field, expected in T2V_RESOURCE_LIFECYCLE_CONTRACT.items():
        if value.get(field) != expected and (
            require_serialized_load
            or field
            not in {
                "serialized_host_checkpoint_load_required",
                "renderer_deserialized_and_moved_to_rank_gpu_under_lock",
                "host_allocator_trim_called_before_load_lock_release",
                "t2v_text_encoder_gpu_residency_required",
                "t2v_text_encoder_cpu_offload_bypass_active",
                "t2v_text_encoder_retired_only_with_renderer",
            }
        ):
            raise NativeIdentityCanaryError(
                f"T2V resource lifecycle differs at {field}"
            )
    gate = _validate_world4_load_completion_gate(
        value["world4_load_completion_gate"], sampling_required=True
    )
    residency_value = value["world4_t2v_text_encoder_gpu_residency_gate"]
    if require_serialized_load:
        residency_gate = _validate_world4_t2v_text_encoder_gpu_residency_gate(
            residency_value
        )
    elif residency_value is None:
        residency_gate = None
    else:
        residency_gate = _validate_world4_t2v_text_encoder_gpu_residency_gate(
            residency_value
        )
    return {
        **dict(value),
        "world4_load_completion_gate": gate,
        "world4_t2v_text_encoder_gpu_residency_gate": residency_gate,
    }


@dataclass(frozen=True)
class NativeInitialNoiseCapture:
    """Read-only copy and provenance of Bernini's actual initial Gaussian.

    ``tensor`` is a detached CPU clone made immediately after the pinned
    module-global ``randn_tensor`` returns.  The native sampler receives the
    original return object, not this clone.
    """

    tensor: Any
    call_count: int
    requested_shape: tuple[int, ...]
    requested_dtype: str
    requested_device: str
    returned_dtype: str
    returned_device: str
    generator_device: str
    generator_initial_seed: int
    raw_value_sha256: str
    content_sha256: str
    numel: int
    byte_count: int


def canonical_reference_indices(frame_count: int, count: int) -> tuple[int, ...]:
    """Return endpoint-inclusive integer samples with deterministic half-up rounding."""

    if type(frame_count) is not int or frame_count <= 0:
        raise NativeIdentityCanaryError("frame_count must be a positive integer")
    if type(count) is not int or not 1 <= count <= frame_count:
        raise NativeIdentityCanaryError("reference count must be in [1, frame_count]")
    if count == 1:
        return ((frame_count - 1) // 2,)
    denominator = count - 1
    indices = tuple(
        (index * (frame_count - 1) + denominator // 2) // denominator
        for index in range(count)
    )
    if len(set(indices)) != count or indices[0] != 0 or indices[-1] != frame_count - 1:
        raise NativeIdentityCanaryError("reference index construction lost coverage")
    return indices


R2V_REFERENCE_INDICES = canonical_reference_indices(FRAME_COUNT, 5)
RV2V_REFERENCE_INDICES = canonical_reference_indices(FRAME_COUNT, 4)


def reference_indices_for_arm(arm: str) -> tuple[int, ...]:
    if arm == "t2v":
        return ()
    if arm == "r2v":
        return R2V_REFERENCE_INDICES
    if arm == "rv2v":
        return RV2V_REFERENCE_INDICES
    raise NativeIdentityCanaryError(f"unknown arm: {arm!r}")


def normalize_arms(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise NativeIdentityCanaryError("at least one native arm is required")
    names = tuple(str(value).lower() for value in values)
    unknown = sorted(set(names) - set(ARM_ORDER))
    if unknown:
        raise NativeIdentityCanaryError(f"unknown native arms: {unknown}")
    if len(set(names)) != len(names):
        raise NativeIdentityCanaryError("native arm names must be unique")
    return tuple(arm for arm in ARM_ORDER if arm in names)


def build_task_prompt(
    arm: str,
    action_prompt: str,
    *,
    prompt_cleaner: Callable[[str], str],
) -> str:
    """Match Bernini training: task system prefix + cleaned task body."""

    if arm not in ARM_ORDER:
        raise NativeIdentityCanaryError(f"unknown prompt arm: {arm!r}")
    if not isinstance(action_prompt, str) or not action_prompt.strip() or "\x00" in action_prompt:
        raise NativeIdentityCanaryError(
            "action_prompt must be non-empty source-content caption text without NUL"
        )
    body = TASK_BINDING_CLAUSES[arm] + action_prompt
    cleaned = prompt_cleaner(body)
    if not isinstance(cleaned, str) or not cleaned.strip():
        raise NativeIdentityCanaryError("Wan prompt cleaner produced an empty prompt")
    return TASK_SYSTEM_PROMPTS[ARM_TRAINING_TASK_NAMES[arm]] + cleaned


def native_sampling_contract(arm: str, *, steps: int, seed: int) -> dict[str, Any]:
    if arm not in ARM_ORDER:
        raise NativeIdentityCanaryError(f"unknown sampling arm: {arm!r}")
    return {
        "num_frames": FRAME_COUNT,
        "num_inference_steps": int(steps),
        "guidance_mode": ARM_GUIDANCE_MODES[arm],
        "omega_vid": OMEGA_VIDEO,
        "omega_img": OMEGA_IMAGE,
        "omega_txt": OMEGA_TEXT,
        "omega_scale": OMEGA_SCALE,
        "flow_shift": FLOW_SHIFT,
        "seed": int(seed),
        "eta": ETA,
        "norm_threshold": NORM_THRESHOLD,
        "momentum": MOMENTUM,
    }


def source_id_contract(arm: str) -> dict[str, Any]:
    """Describe the source-id axis consumed by the unmodified native sampler."""

    refs = ARM_REFERENCE_COUNTS.get(arm)
    videos = ARM_VIDEO_COUNTS.get(arm)
    if refs is None or videos is None:
        raise NativeIdentityCanaryError(f"unknown source-id arm: {arm!r}")
    conditioning_count = int(refs + videos)
    return {
        "target_source_id": 0,
        "video_source_ids": list(range(1, videos + 1)),
        "reference_source_ids": list(
            range(videos + 1, videos + refs + 1)
        ),
        "conditioning_source_count": conditioning_count,
        "max_conditioning_source_id": conditioning_count,
        "within_pretrained_source_ids_1_through_5": conditioning_count <= 5,
        "source_id_interpolation_required": False,
    }


def select_native_conditions(
    arm: str,
    *,
    full_source_latent: Any,
    reference_latents: Mapping[int, Any],
) -> dict[str, Any]:
    """Select only source-derived conditions required by one native arm."""

    indices = reference_indices_for_arm(arm)
    missing = [index for index in indices if index not in reference_latents]
    if missing:
        raise NativeIdentityCanaryError(f"missing independently encoded refs: {missing}")
    if arm == "rv2v" and full_source_latent is None:
        raise NativeIdentityCanaryError("rv2v requires the full source video latent")
    return {
        "image_vae_latents": None,
        "multi_video_vae_latents": [full_source_latent] if arm == "rv2v" else None,
        "multi_image_vae_latents": [reference_latents[index] for index in indices]
        if indices
        else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument(
        "--action-prompt",
        required=True,
        help=(
            "complete source-content generation caption containing the desired new "
            "action; do not pass a bare imperative edit command"
        ),
    )
    parser.add_argument("--expected-action-prompt-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARM_ORDER), choices=ARM_ORDER)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> tuple[str, ...]:
    arms = normalize_arms(args.arms)
    if not isinstance(args.action_prompt, str) or not args.action_prompt.strip() or "\x00" in args.action_prompt:
        raise NativeIdentityCanaryError("action_prompt must be non-empty text without NUL")
    action_digest = hashlib.sha256(args.action_prompt.encode("utf-8")).hexdigest()
    if action_digest != args.expected_action_prompt_sha256:
        raise NativeIdentityCanaryError("action prompt SHA-256 differs")
    if args.num_inference_steps != NUM_INFERENCE_STEPS:
        raise NativeIdentityCanaryError("native canary is fixed to 40 UniPC steps")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise NativeIdentityCanaryError("seed must be in [0,2^63)")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
            raise NativeIdentityCanaryError(f"{name} must be a full lowercase SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "expected_source_sha256",
        "expected_action_prompt_sha256",
        "method_source_archive_sha256",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise NativeIdentityCanaryError(f"{name} must be a lowercase SHA-256")
    if args.expected_bernini_commit != legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise NativeIdentityCanaryError("unsupported Bernini source revision")
    if args.expected_veomni_commit != legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise NativeIdentityCanaryError("unsupported VeOmni source revision")
    if args.expected_checkpoint_tree_sha256 != legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise NativeIdentityCanaryError("unsupported Bernini-R checkpoint tree")
    output = Path(args.output_dir).expanduser()
    if not output.is_absolute() or output == Path("/"):
        raise NativeIdentityCanaryError("output-dir must be an absolute non-root path")
    if _SAFE_BASENAME.fullmatch(output.name) is None:
        raise NativeIdentityCanaryError("output-dir basename is not path-safe")
    return arms


def _resolve_fresh_output_dir(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise NativeIdentityCanaryError("output-dir must be absolute and non-root")
    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise NativeIdentityCanaryError("output-dir parent must be a plain directory")
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise NativeIdentityCanaryError("refusing to reuse output-dir")
    return output


def _all_rank_tensor_identity(
    value: Any,
    *,
    label: str,
    world_size: int,
) -> dict[str, Any]:
    import torch.distributed as dist

    local = value_audit.tensor_identity(value, label=label)
    rows: list[Any] = [None] * world_size
    dist.all_gather_object(rows, local)
    if any(row != rows[0] for row in rows[1:]):
        raise NativeIdentityCanaryError(f"{label} differs across Ulysses ranks")
    return {"all_rank_exact": True, "identity": dict(rows[0])}


def _broadcast_condition_from_rank_zero(
    value: Any,
    *,
    label: str,
    world_size: int,
) -> dict[str, Any]:
    """Make a locally encoded VAE condition bit-identical on every SP rank.

    Wan VAE convolutions can select different ROCm kernels on different
    devices.  Even with ``latent_dist.mode()`` this can produce harmless
    floating-point round-off differences, but Ulysses requires replicated
    conditioning tensors to be exactly identical.  Rank zero is therefore the
    sole authoritative value after local encoding; the broadcast happens
    before the tensor is ever passed to the renderer.
    """

    import torch
    import torch.distributed as dist

    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise NativeIdentityCanaryError(f"{label} must be a non-empty tensor")
    if (
        value.requires_grad
        or not value.is_floating_point()
        or not value.is_contiguous()
    ):
        raise NativeIdentityCanaryError(
            f"{label} must be a detached contiguous floating tensor"
        )
    if world_size != dist.get_world_size():
        raise NativeIdentityCanaryError(f"{label} distributed world size differs")
    local_contract = {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device_type": value.device.type,
        "finite": bool(torch.isfinite(value).all().item()),
    }
    contracts: list[Any] = [None] * world_size
    dist.all_gather_object(contracts, local_contract)
    if not local_contract["finite"] or any(
        row != contracts[0] for row in contracts[1:]
    ):
        raise NativeIdentityCanaryError(
            f"{label} metadata differs across Ulysses ranks before broadcast"
        )
    dist.broadcast(value, src=0)
    return {
        "authoritative_rank": 0,
        "broadcast_before_renderer": True,
        "prebroadcast_metadata_all_rank_exact": True,
        "contract": dict(contracts[0]),
    }


def _latent_geometry_receipt(
    *,
    bucket_hw: Sequence[int],
    z_dim: int,
) -> dict[str, Any]:
    height, width = (int(bucket_hw[0]), int(bucket_hw[1]))
    video_shape = (1, int(z_dim), LATENT_FRAME_COUNT, height // 8, width // 8)
    image_shape = (1, int(z_dim), 1, height // 8, width // 8)
    spatial_tokens = (height // 16) * (width // 16)
    target_tokens = LATENT_FRAME_COUNT * spatial_tokens
    return {
        "video_latent_shape": list(video_shape),
        "reference_latent_shape": list(image_shape),
        "target_patch_tokens": target_tokens,
        "one_reference_patch_tokens": spatial_tokens,
        "per_arm_total_visual_tokens": {
            arm: (
                target_tokens
                + ARM_VIDEO_COUNTS[arm] * target_tokens
                + ARM_REFERENCE_COUNTS[arm] * spatial_tokens
            )
            for arm in ARM_ORDER
        },
    }


def _tensor_raw_value_sha256(value: Any, *, label: str) -> str:
    """Hash only exact contiguous tensor bytes, independent of Python object id."""

    try:
        identity = value_audit.tensor_identity(value, label=label)
    except Exception as error:
        raise NativeIdentityCanaryError(str(error)) from error
    digest = identity.get("raw_storage_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise NativeIdentityCanaryError(f"{label} raw tensor digest is invalid")
    return digest


def _sample_with_native_initial_noise_observer(
    *,
    sample_fn: Callable[[], Any],
    wan_diffusion_module: Any,
    expected_shape: Sequence[int],
    expected_device: Any,
    expected_seed: int,
    canonical_randn_tensor: Optional[Callable[..., Any]] = None,
) -> tuple[Any, NativeInitialNoiseCapture]:
    """Run one native sample while observing its real initial Gaussian.

    The observer temporarily replaces only the symbol looked up by the pinned
    ``GEN_Wanx22.sample`` function.  Its wrapper forwards the original
    positional arguments, keyword arguments, and generator object unchanged,
    returns the *same* tensor object, and keeps a detached CPU clone solely for
    provenance.  It is therefore an observer, not an initial-noise injection.

    The symbol is restored in ``finally`` on both success and failure.  A
    successful native sample is rejected unless exactly one call matches the
    official CPU-generator, FP32, target-shape contract.
    """

    try:
        import torch
        if canonical_randn_tensor is None:
            from diffusers.utils.torch_utils import randn_tensor as canonical
        else:
            canonical = canonical_randn_tensor
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise NativeIdentityCanaryError(
            "native initial-noise observation requires PyTorch and Diffusers"
        ) from error

    if not callable(sample_fn):
        raise NativeIdentityCanaryError("sample_fn must be callable")
    expected = tuple(int(item) for item in expected_shape)
    if not expected or any(item <= 0 for item in expected):
        raise NativeIdentityCanaryError("expected initial-noise shape is invalid")
    if type(expected_seed) is not int or not 0 <= expected_seed < 2**63:
        raise NativeIdentityCanaryError("expected initial-noise seed is invalid")

    original = getattr(wan_diffusion_module, "randn_tensor", None)
    if original is not canonical:
        raise NativeIdentityCanaryError(
            "pinned wan_diffusion.randn_tensor is already replaced or differs"
        )

    calls: list[dict[str, Any]] = []

    def observed_randn_tensor(*call_args: Any, **call_kwargs: Any) -> Any:
        shape_value = (
            call_args[0]
            if call_args
            else call_kwargs.get("shape")
        )
        try:
            requested_shape = tuple(int(item) for item in shape_value)
        except Exception as error:
            raise NativeIdentityCanaryError(
                "official randn_tensor call has no valid shape"
            ) from error
        generator = call_kwargs.get("generator")
        requested_device = call_kwargs.get("device")
        requested_dtype = call_kwargs.get("dtype")
        if not isinstance(generator, torch.Generator):
            raise NativeIdentityCanaryError(
                "official initial Gaussian must use one torch.Generator"
            )
        generator_device = str(generator.device)
        generator_seed = int(generator.initial_seed())

        # The original callable sees the exact objects supplied by Bernini.
        returned = original(*call_args, **call_kwargs)
        if not isinstance(returned, torch.Tensor):
            raise NativeIdentityCanaryError(
                "official randn_tensor did not return a tensor"
            )
        captured = returned.detach().to(device="cpu").contiguous().clone()
        identity = value_audit.tensor_identity(
            captured, label="official_native_initial_gaussian"
        )
        calls.append(
            {
                "requested_shape": requested_shape,
                "requested_device": str(requested_device),
                "requested_dtype": str(requested_dtype),
                "generator_device": generator_device,
                "generator_initial_seed": generator_seed,
                "returned_shape": tuple(int(item) for item in returned.shape),
                "returned_device": str(returned.device),
                "returned_dtype": str(returned.dtype),
                "tensor": captured,
                "identity": identity,
            }
        )
        # Crucial: do not return ``captured`` (or any transformed tensor).
        return returned

    setattr(wan_diffusion_module, "randn_tensor", observed_randn_tensor)
    observer_symbol_unchanged = True
    try:
        sample_result = sample_fn()
    finally:
        observer_symbol_unchanged = (
            getattr(wan_diffusion_module, "randn_tensor", None)
            is observed_randn_tensor
        )
        setattr(wan_diffusion_module, "randn_tensor", original)

    if not observer_symbol_unchanged:
        raise NativeIdentityCanaryError(
            "wan_diffusion.randn_tensor changed while observer was active"
        )
    if getattr(wan_diffusion_module, "randn_tensor", None) is not original:
        raise NativeIdentityCanaryError(
            "wan_diffusion.randn_tensor restoration failed"
        )
    if len(calls) != 1:
        raise NativeIdentityCanaryError(
            "native sampler must make exactly one initial randn_tensor call; "
            f"observed {len(calls)}"
        )

    call = calls[0]
    if call["requested_shape"] != expected or call["returned_shape"] != expected:
        raise NativeIdentityCanaryError("native initial Gaussian shape differs")
    if call["requested_dtype"] != str(torch.float32) or call["returned_dtype"] != str(
        torch.float32
    ):
        raise NativeIdentityCanaryError("native initial Gaussian is not FP32")
    expected_device_text = str(torch.device(expected_device))
    if (
        call["requested_device"] != expected_device_text
        or call["returned_device"] != expected_device_text
    ):
        raise NativeIdentityCanaryError("native initial Gaussian device differs")
    if call["generator_device"] != "cpu":
        raise NativeIdentityCanaryError(
            "native initial Gaussian generator is not on CPU"
        )
    if call["generator_initial_seed"] != expected_seed:
        raise NativeIdentityCanaryError("native initial Gaussian seed differs")

    identity = call["identity"]
    return sample_result, NativeInitialNoiseCapture(
        tensor=call["tensor"],
        call_count=1,
        requested_shape=call["requested_shape"],
        requested_dtype=call["requested_dtype"],
        requested_device=call["requested_device"],
        returned_dtype=call["returned_dtype"],
        returned_device=call["returned_device"],
        generator_device=call["generator_device"],
        generator_initial_seed=call["generator_initial_seed"],
        raw_value_sha256=str(identity["raw_storage_sha256"]),
        content_sha256=str(identity["content_sha256"]),
        numel=int(identity["numel"]),
        byte_count=int(identity["byte_count"]),
    )


def _save_initial_noise_atomically(
    path: Path,
    capture: NativeInitialNoiseCapture,
    *,
    all_rank_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the exact observed FP32 Gaussian with a byte-exact round trip."""

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if path.exists() or path.is_symlink() or path.suffix != ".safetensors":
        raise NativeIdentityCanaryError(
            "initial-noise path must be a fresh safetensors file"
        )
    stored = capture.tensor
    if (
        not isinstance(stored, torch.Tensor)
        or stored.device.type != "cpu"
        or stored.dtype != torch.float32
        or stored.requires_grad
        or not stored.is_contiguous()
        or tuple(int(item) for item in stored.shape) != capture.requested_shape
    ):
        raise NativeIdentityCanaryError(
            "captured initial Gaussian storage contract differs"
        )
    if _tensor_raw_value_sha256(stored, label="initial_gaussian_before_save") != (
        capture.raw_value_sha256
    ):
        raise NativeIdentityCanaryError(
            "captured initial Gaussian changed before artifact save"
        )
    gathered_identity = all_rank_identity.get("identity")
    if (
        all_rank_identity.get("all_rank_exact") is not True
        or not isinstance(gathered_identity, Mapping)
        or gathered_identity.get("raw_storage_sha256")
        != capture.raw_value_sha256
        or gathered_identity.get("shape") != list(capture.requested_shape)
        or gathered_identity.get("dtype") != capture.returned_dtype
    ):
        raise NativeIdentityCanaryError(
            "captured initial Gaussian lacks exact all-rank identity"
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
            {"official_initial_gaussian": stored},
            str(temporary),
            metadata={
                "coordinate": "bernini_native_target_latent_before_rearrange",
                "source": "observed_return_of_official_module_global_randn_tensor",
                "observer_only": "true",
                "external_initial_noise_injection": "false",
            },
        )
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != ["official_initial_gaussian"]:
                raise NativeIdentityCanaryError(
                    "initial-noise safetensors key differs"
                )
            restored = opened.get_tensor("official_initial_gaussian").contiguous()
            metadata = dict(opened.metadata() or {})
        if (
            restored.dtype != stored.dtype
            or tuple(restored.shape) != tuple(stored.shape)
            or not torch.equal(restored, stored)
            or _tensor_raw_value_sha256(
                restored, label="initial_gaussian_after_roundtrip"
            )
            != capture.raw_value_sha256
        ):
            raise NativeIdentityCanaryError(
                "initial-noise safetensors round trip differs"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()

    return {
        "path": str(path),
        "sha256": legacy.file_sha256(path),
        "tensor_key": "official_initial_gaussian",
        "tensor_value_sha256": capture.raw_value_sha256,
        "raw_value_sha256": capture.raw_value_sha256,
        "content_sha256": capture.content_sha256,
        "shape": list(capture.requested_shape),
        "dtype": capture.returned_dtype,
        "stored_dtype": str(stored.dtype),
        "original_device": capture.returned_device,
        "stored_device": "cpu",
        "numel": capture.numel,
        "byte_count": capture.byte_count,
        "randn_tensor_call_count": capture.call_count,
        "official_randn_tensor_call_count": capture.call_count,
        "requested_device": capture.requested_device,
        "requested_dtype": capture.requested_dtype,
        "generator_device": capture.generator_device,
        "generator_initial_seed": capture.generator_initial_seed,
        "all_rank_identity": dict(all_rank_identity),
        "coordinate": metadata["coordinate"],
        "origin": metadata["source"],
        "observer_only": True,
        "captured_from_native_sampler": True,
        "observer_changed_return_value": False,
        "source_or_target_derived": False,
        "observer_added_device_to_cpu_readback": True,
        "official_module_global_symbol": (
            "bernini.models.wan_diffusion.randn_tensor"
        ),
        "original_callable_invoked_once_with_unchanged_arguments": True,
        "original_return_tensor_forwarded_by_identity": True,
        "external_initial_noise_injection": False,
        "sampler_noise_replacement": False,
        "roundtrip_raw_value_exact": True,
    }


def _build_receipt(
    *,
    args: argparse.Namespace,
    arms: Sequence[str],
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    prompts: Mapping[str, str],
    checkpoint_identity: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
    freeze_certificate: Mapping[str, Any],
    latent_geometry: Mapping[str, Any],
    condition_identities: Mapping[str, Any],
    source_condition_artifact: Optional[Mapping[str, Any]],
    initial_noise_artifacts: Mapping[str, Any],
    generated_identities: Mapping[str, Any],
    outputs: Mapping[str, Any],
    resource_lifecycle: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_bytes = args.action_prompt.encode("utf-8")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "bernini_inference_files": dict(inference_file_hashes),
        "checkpoint": {
            "path": str(Path(args.checkpoint).expanduser().resolve()),
            "tree_sha256": args.expected_checkpoint_tree_sha256,
            "content": dict(checkpoint_identity),
        },
        "arms": list(arms),
        "input": {
            "source_video_path": str(source_path),
            "source_video_sha256": source_sha256,
            "action_prompt_utf8_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "action_prompt_utf8_bytes": len(prompt_bytes),
            "accepted_external_conditions": ["source_video", "action_prompt"],
            "target_video": False,
            "external_reference_image_or_video": False,
            "external_mask_flow_pose_track_trajectory": False,
            "external_first_frame_anchor": False,
        },
        "preprocessing": dict(source_metadata),
        "prompt_contract": {
            arm: {
                "training_task_name": ARM_TRAINING_TASK_NAMES[arm],
                "inference_arm": arm,
                "guidance_mode": ARM_GUIDANCE_MODES[arm],
                "system_prompt_sha256": hashlib.sha256(
                    TASK_SYSTEM_PROMPTS[ARM_TRAINING_TASK_NAMES[arm]].encode("utf-8")
                ).hexdigest(),
                "binding_clause_sha256": hashlib.sha256(
                    TASK_BINDING_CLAUSES[arm].encode("utf-8")
                ).hexdigest(),
                "full_prompt_sha256": hashlib.sha256(
                    prompts[arm].encode("utf-8")
                ).hexdigest(),
                "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
                "tokenizer_fix_mistral_regex": True,
            }
            for arm in arms
        },
        "conditioning": {
            arm: {
                "full_source_video_count": ARM_VIDEO_COUNTS[arm],
                "source_derived_reference_count": ARM_REFERENCE_COUNTS[arm],
                "source_frame_indices": list(reference_indices_for_arm(arm)),
                "reference_encoding": (
                    "independent_rgb_frame_to_wan_vae_[1,C,1,H,W]"
                    if ARM_REFERENCE_COUNTS[arm]
                    else "none"
                ),
                "reference_from_temporal_video_latent_slice": False,
                "source_ids": source_id_contract(arm),
            }
            for arm in arms
        },
        "sampling": {
            arm: {
                **native_sampling_contract(
                    arm, steps=args.num_inference_steps, seed=args.seed
                ),
                "target_initialization": TARGET_INITIALIZATION,
                "target_mixed_with_source_latent": False,
                "custom_sampler_or_scheduler": False,
                "same_seed_and_target_shape_across_arms": True,
                "single_expert": "transformer_1",
                "ulysses_size": ULYSSES_SIZE,
            }
            for arm in arms
        },
        "latent_geometry": dict(latent_geometry),
        "condition_identities": dict(condition_identities),
        "source_condition_artifact": (
            dict(source_condition_artifact)
            if source_condition_artifact is not None
            else None
        ),
        "initial_noise_artifacts": dict(initial_noise_artifacts),
        "generated_identities": dict(generated_identities),
        "outputs": dict(outputs),
        "freeze_certificate": dict(freeze_certificate),
        "runtime_versions": dict(runtime_versions),
        "resource_lifecycle": dict(resource_lifecycle),
        "interpretation": {
            "purpose": "test_native_identity_conditioned_generation_before_training",
            "quality_claim": False,
            "training_performed": False,
            "best_arm_selected": False,
        },
        "experimental_canary": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def _save_outputs(
    *,
    output_dir: Path,
    generated: Mapping[str, Any],
    vae: Any,
    bucket_hw: Sequence[int],
    device: Any,
    save_output_fn: Any,
) -> dict[str, Any]:
    import torch
    from bernini.pipeline import _vae_decode
    from tools import materialize_vae

    expected_hw = (int(bucket_hw[0]), int(bucket_hw[1]))
    outputs: dict[str, Any] = {}
    vae.to(device)
    for arm, latent in generated.items():
        clean_latent = _save_normalized_clean_latent_atomically(
            output_dir / f"{arm}.normalized-clean-latent.safetensors",
            latent,
        )
        with torch.no_grad():
            decoded = _vae_decode(vae, latent)
        if tuple(int(item) for item in decoded.shape) != (
            FRAME_COUNT,
            expected_hw[0],
            expected_hw[1],
            3,
        ):
            raise NativeIdentityCanaryError(f"{arm} decoded shape differs")
        path = output_dir / f"{arm}.mp4"
        value_audit.save_video_atomically(
            decoded,
            path,
            fps=FPS,
            save_output_fn=save_output_fn,
        )
        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(path)
        legacy.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
        if tuple(encoded_hw) != expected_hw:
            raise NativeIdentityCanaryError(f"{arm} encoded geometry differs")
        outputs[arm] = {
            "path": str(path),
            "sha256": legacy.file_sha256(path),
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "height": expected_hw[0],
            "width": expected_hw[1],
            "normalized_clean_latent": clean_latent,
        }
    vae.to("cpu")
    return outputs


def _save_normalized_clean_latent_atomically(
    path: Path,
    latent: Any,
    *,
    artifact_role: str = "native_sampler_proposal",
) -> dict[str, Any]:
    """Persist the pre-decode clean proposal without an MP4 round trip.

    The stored tensor is an FP32 copy of the exact normalized latent returned
    by Bernini's native sampler.  It is the only legal candidate state for a
    later DCLR reward query; decoding and re-encoding the companion MP4 would
    change that state.
    """

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    roles = {
        "native_sampler_proposal": "native_sampler_before_vae_decode",
        "source_video_condition": "source_video_vae_encode_before_any_decode",
    }
    if artifact_role not in roles:
        raise NativeIdentityCanaryError("unsupported normalized latent artifact role")
    if path.exists() or path.is_symlink() or path.suffix != ".safetensors":
        raise NativeIdentityCanaryError("clean-latent path must be fresh safetensors")
    if (
        not isinstance(latent, torch.Tensor)
        or latent.requires_grad
        or not latent.is_floating_point()
        or latent.ndim != 5
        or tuple(int(item) for item in latent.shape[:3])
        != (1, 16, LATENT_FRAME_COUNT)
    ):
        raise NativeIdentityCanaryError(
            "generated clean latent must be detached [1,16,21,H,W]"
        )
    stored = latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(stored).all().item()):
        raise NativeIdentityCanaryError("generated clean latent is non-finite")
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
            metadata={
                "coordinate": "bernini_normalized_clean_vae_latent",
                "frame_contract": "exact81_latent21",
                "artifact_role": artifact_role,
                "source": roles[artifact_role],
            },
        )
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != ["normalized_clean_latent"]:
                raise NativeIdentityCanaryError("clean-latent safetensors key differs")
            restored = opened.get_tensor("normalized_clean_latent").contiguous()
            metadata = dict(opened.metadata() or {})
        if (
            restored.dtype != torch.float32
            or tuple(restored.shape) != tuple(stored.shape)
            or not torch.equal(restored, stored)
        ):
            raise NativeIdentityCanaryError("clean-latent safetensors round trip differs")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": legacy.file_sha256(path),
        "tensor_key": "normalized_clean_latent",
        "shape": [int(item) for item in stored.shape],
        "stored_dtype": str(stored.dtype),
        "sampler_return_dtype": str(latent.dtype),
        "coordinate": metadata["coordinate"],
        "artifact_role": artifact_role,
        "origin": roles[artifact_role],
        "native_sampler_before_vae_decode": artifact_role == "native_sampler_proposal",
        "source_video_vae_encode_before_any_decode": artifact_role
        == "source_video_condition",
        "mp4_decode_reencode_used": False,
        "roundtrip_byte_exact_fp32": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    arms = validate_cli(args)
    t2v_gpu_residency_required = os.environ.get(
        "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED"
    )
    if t2v_gpu_residency_required not in (None, "1"):
        raise NativeIdentityCanaryError(
            "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED differs"
        )
    if t2v_gpu_residency_required == "1" and arms != ("t2v",):
        raise NativeIdentityCanaryError(
            "rank-GPU T5 residency is sealed only for the T2V-only worker"
        )
    output_dir = _resolve_fresh_output_dir(args.output_dir)
    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute():
        raise NativeIdentityCanaryError("source-video must be absolute")
    source_path = legacy._plain_file(
        source_requested.resolve(strict=True), label="source video"
    )
    manifest_path = Path(args.checkpoint_content_manifest).expanduser()
    if not manifest_path.is_absolute():
        raise NativeIdentityCanaryError("checkpoint-content-manifest must be absolute")

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
        raise NativeIdentityCanaryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise NativeIdentityCanaryError(
            "Bernini-R 1.3B heads are not divisible by Ulysses=4"
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
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    for task_name, expected in TASK_SYSTEM_PROMPTS.items():
        if SYSTEM_PROMPTS.get(task_name) != expected:
            raise NativeIdentityCanaryError(
                f"runtime Bernini {task_name} system prompt differs"
            )
    if DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT:
        raise NativeIdentityCanaryError("runtime Bernini negative prompt differs")

    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise NativeIdentityCanaryError("canary requires four AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, manifest_path
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_result = checkpoint_rows[0]
    if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get("ok") is not True:
        raise NativeIdentityCanaryError(
            f"rank-zero checkpoint validation failed: {checkpoint_result}"
        )
    checkpoint_identity = dict(checkpoint_result["identity"])

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise NativeIdentityCanaryError(str(error)) from error
    if float(config.shift) != FLOW_SHIFT or config.use_unipc is not True:
        raise NativeIdentityCanaryError("renderer is not pinned to UniPC shift 5")

    # Hold the authenticated node-local flock until the temporary host
    # checkpoint state has been moved to the rank GPU and glibc arenas have
    # been returned to the cgroup.  The r10 formal log proves that early ranks
    # sampled while the final rank deserialized and was then cgroup-OOM-killed;
    # it does not prove one unique cause.  This is the same serialized load
    # boundary used by the R64 heldout renderer.
    model = _load_frozen_renderer_gpu_resident_serialized(
        BerniniRendererModel, config, device
    )
    renderer_gpu_resident_trimmed_ns = time.monotonic_ns()
    world4_load_completion_gate = _world4_renderer_load_completion_barrier(
        dist,
        rank=distributed.rank,
        world_size=distributed.world_size,
        renderer_gpu_resident_trimmed_monotonic_ns=(
            renderer_gpu_resident_trimmed_ns
        ),
    )
    freeze_before = source_audit.model_freeze_certificate(model)

    # Source decoding and tokenizer construction intentionally begin only
    # after all four renderers are GPU-resident and their host load arenas have
    # been trimmed.  This prevents the observed r10 load-versus-sampling
    # overlap without assigning the OOM to one unique contributor.
    source_tensor, source_metadata, source_sha256 = (
        source_audit.prepare_hashed_source_snapshot(source_path)
    )
    if source_sha256 != args.expected_source_sha256:
        raise NativeIdentityCanaryError("source video SHA-256 differs")
    bucket_hw = tuple(int(item) for item in source_metadata["source_derived_bucket_hw"])

    prompts = {
        arm: build_task_prompt(arm, args.action_prompt, prompt_cleaner=prompt_clean)
        for arm in arms
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    if (
        tokenizer.padding_side != "right"
        or tokenizer.init_kwargs.get("fix_mistral_regex") is not True
    ):
        raise NativeIdentityCanaryError("tokenizer contract differs")
    positive_tokens = {
        arm: legacy._tokenize_training_prompt(tokenizer, prompts[arm])
        for arm in arms
    }
    negative_ids, negative_mask = legacy._tokenize_renderer_negative(
        tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT
    )

    required_reference_indices = sorted(
        {
            index
            for arm in arms
            for index in reference_indices_for_arm(arm)
        }
    )
    # The released fit40 worker is T2V-only: its source snapshot supplies
    # geometry but never a source/reference latent.  Do not materialize four
    # unused VAE weight copies before sampling.  Historical R2V/RV2V callers
    # retain their exact independently-encoded source-conditioning path.
    vae = None
    if required_reference_indices or "rv2v" in arms:
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval().requires_grad_(False)
        vae.to(device)
        source_pixels = source_tensor.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            full_source_latent = (
                _vae_encode(vae, source_pixels).contiguous()
                if "rv2v" in arms
                else None
            )
            # This is intentionally RGB-frame -> VAE for every reference.  Do
            # not replace it with a temporal slice of ``full_source_latent``.
            reference_latents = {
                index: _vae_encode(
                    vae,
                    source_pixels[:, :, index : index + 1, :, :].contiguous(),
                ).contiguous()
                for index in required_reference_indices
            }
        vae_z_dim = int(vae.config.z_dim)
        del source_pixels
    else:
        vae_config = AutoencoderKLWan.load_config(
            str(checkpoint), subfolder="vae", local_files_only=True
        )
        vae_z_dim = int(vae_config["z_dim"])
        full_source_latent = None
        reference_latents = {}

    # The VAE posterior mode is deterministic mathematically, but independent
    # ROCm convolution kernels can differ by a few low bits across devices.
    # Ulysses conditioning is replicated, so establish one exact authoritative
    # tensor on rank zero before any native renderer forward.
    condition_broadcasts = {
        "references": {
            str(index): _broadcast_condition_from_rank_zero(
                latent,
                label=f"source_reference_{index}",
                world_size=distributed.world_size,
            )
            for index, latent in reference_latents.items()
        }
    }
    if full_source_latent is not None:
        condition_broadcasts["full_source_video"] = (
            _broadcast_condition_from_rank_zero(
                full_source_latent,
                label="full_source_video",
                world_size=distributed.world_size,
            )
        )
    else:
        condition_broadcasts["full_source_video"] = None

    latent_geometry = _latent_geometry_receipt(bucket_hw=bucket_hw, z_dim=vae_z_dim)
    expected_video_shape = tuple(latent_geometry["video_latent_shape"])
    expected_reference_shape = tuple(latent_geometry["reference_latent_shape"])
    if full_source_latent is not None and tuple(full_source_latent.shape) != expected_video_shape:
        raise NativeIdentityCanaryError("full source latent shape differs")
    for index, latent in reference_latents.items():
        if tuple(latent.shape) != expected_reference_shape:
            raise NativeIdentityCanaryError(f"reference {index} latent shape differs")

    condition_identities: dict[str, Any] = {
        "rank_zero_broadcasts": condition_broadcasts,
        "references": {
            str(index): _all_rank_tensor_identity(
                latent,
                label=f"source_reference_{index}",
                world_size=distributed.world_size,
            )
            for index, latent in reference_latents.items()
        }
    }
    if full_source_latent is not None:
        condition_identities["full_source_video"] = _all_rank_tensor_identity(
            full_source_latent,
            label="full_source_video",
            world_size=distributed.world_size,
        )
    else:
        condition_identities["full_source_video"] = None

    if vae is not None:
        vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    world4_load_completion_gate = (
        _complete_world4_load_completion_gate_before_sampling(
            dist,
            world4_load_completion_gate,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
    )

    generated: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    initial_noise_captures: dict[str, NativeInitialNoiseCapture] = {}
    initial_noise_rank_identities: dict[str, Any] = {}
    local_t2v_text_encoder_residency: Optional[Mapping[str, Any]] = None
    with torch.no_grad():
        for arm in arms:
            input_ids, attention_mask = positive_tokens[arm]
            condition_kwargs = select_native_conditions(
                arm,
                full_source_latent=full_source_latent,
                reference_latents=reference_latents,
            )
            with _t2v_text_encoder_rank_gpu_residency(
                model, arm=arm, device=device
            ) as residency_evidence:
                generated_latent, noise_capture = (
                    _sample_with_native_initial_noise_observer(
                        sample_fn=lambda: model.sample(
                            input_ids=input_ids.to(device),
                            attention_mask=attention_mask.to(device),
                            uncond_input_ids=negative_ids.to(device),
                            uncond_attention_mask=negative_mask.to(device),
                            **condition_kwargs,
                            width=bucket_hw[1],
                            height=bucket_hw[0],
                            device=device,
                            **native_sampling_contract(
                                arm, steps=args.num_inference_steps, seed=args.seed
                            ),
                        ),
                        wan_diffusion_module=wan_diffusion,
                        expected_shape=expected_video_shape,
                        expected_device=device,
                        expected_seed=args.seed,
                    )
                )
            if arm == "t2v":
                local_t2v_text_encoder_residency = dict(residency_evidence)
            if tuple(generated_latent.shape) != expected_video_shape:
                raise NativeIdentityCanaryError(f"{arm} generated latent shape differs")
            generated[arm] = generated_latent
            initial_noise_captures[arm] = noise_capture
            initial_noise_rank_identities[arm] = _all_rank_tensor_identity(
                noise_capture.tensor,
                label=f"official_initial_gaussian_{arm}",
                world_size=distributed.world_size,
            )
            generated_identities[arm] = _all_rank_tensor_identity(
                generated_latent,
                label=f"generated_{arm}",
                world_size=distributed.world_size,
            )

    initial_noise_hashes = {
        capture.raw_value_sha256 for capture in initial_noise_captures.values()
    }
    if len(initial_noise_hashes) != 1:
        raise NativeIdentityCanaryError(
            "same seed and target shape did not yield one exact Gaussian across arms"
        )

    world4_t2v_text_encoder_gpu_residency_gate = None
    if t2v_gpu_residency_required == "1":
        if local_t2v_text_encoder_residency is None:
            raise NativeIdentityCanaryError(
                "T2V text-encoder residency evidence is absent"
            )
        world4_t2v_text_encoder_gpu_residency_gate = (
            _world4_t2v_text_encoder_gpu_residency_gate(
                dist,
                local_t2v_text_encoder_residency,
                rank=distributed.rank,
                local_rank=distributed.local_rank,
                world_size=distributed.world_size,
            )
        )

    freeze_after = source_audit.model_freeze_certificate(model)
    if freeze_after != freeze_before:
        raise NativeIdentityCanaryError("frozen model certificate changed")
    # Never materialize four full CPU model replicas after sampling.  Retire
    # every rank's renderer, return its host arenas, release its device cache,
    # and close one WORLD4 barrier before rank zero is allowed to deserialize
    # the post-sampling decode-only VAE.
    t2v_vae_deferred_until_post_sampling = vae is None
    del model
    _trim_host_allocator()
    torch.cuda.empty_cache()
    dist.barrier()

    resource_lifecycle = _resource_lifecycle_receipt(
        t2v_vae_deferred_until_post_sampling=(
            t2v_vae_deferred_until_post_sampling
        ),
        world4_load_completion_gate=world4_load_completion_gate,
        world4_t2v_text_encoder_gpu_residency_gate=(
            world4_t2v_text_encoder_gpu_residency_gate
        ),
    )

    runtime_versions = {
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "transformers": transformers_version,
        "diffusers": diffusers_version,
    }
    if distributed.rank == 0:
        if vae is None:
            # T2V needs the VAE only once, after the renderer has been retired,
            # to decode the proposal on rank zero.
            vae = AutoencoderKLWan.from_pretrained(
                str(checkpoint),
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            vae.eval().requires_grad_(False)
        output_dir.mkdir(parents=False, exist_ok=False)
        initial_noise_artifacts = {
            arm: _save_initial_noise_atomically(
                output_dir / f"{arm}.official-initial-gaussian.safetensors",
                initial_noise_captures[arm],
                all_rank_identity=initial_noise_rank_identities[arm],
            )
            for arm in arms
        }
        source_condition_artifact = (
            _save_normalized_clean_latent_atomically(
                output_dir / "source.normalized-clean-latent.safetensors",
                full_source_latent,
                artifact_role="source_video_condition",
            )
            if full_source_latent is not None
            else None
        )
        outputs = _save_outputs(
            output_dir=output_dir,
            generated=generated,
            vae=vae,
            bucket_hw=bucket_hw,
            device=device,
            save_output_fn=save_output,
        )
        receipt = _build_receipt(
            args=args,
            arms=arms,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            prompts=prompts,
            checkpoint_identity=checkpoint_identity,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            inference_file_hashes=inference_file_hashes,
            runtime_versions=runtime_versions,
            freeze_certificate=freeze_after,
            latent_geometry=latent_geometry,
            condition_identities=condition_identities,
            source_condition_artifact=source_condition_artifact,
            initial_noise_artifacts=initial_noise_artifacts,
            generated_identities=generated_identities,
            outputs=outputs,
            resource_lifecycle=resource_lifecycle,
        )
        value_audit.write_receipt_atomically(output_dir / "receipt.json", receipt)
        print(legacy.canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier()
    del full_source_latent, reference_latents, initial_noise_captures
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_GUIDANCE_MODES",
    "ARM_ORDER",
    "ARM_REFERENCE_COUNTS",
    "ARM_VIDEO_COUNTS",
    "FRAME_COUNT",
    "LATENT_FRAME_COUNT",
    "METHOD",
    "NativeIdentityCanaryError",
    "R2V_REFERENCE_INDICES",
    "RV2V_REFERENCE_INDICES",
    "SCHEMA_VERSION",
    "T2V_GPU_MEMORY_LIMIT_BYTES",
    "T2V_GPU_MEMORY_LIMIT_GIB",
    "T2V_RESOURCE_LIFECYCLE_CONTRACT",
    "T2V_TEXT_ENCODER_GPU_RESIDENCY_GATE_SCHEMA",
    "TARGET_INITIALIZATION",
    "WORLD4_LOAD_COMPLETION_GATE_SCHEMA",
    "build_parser",
    "build_task_prompt",
    "canonical_reference_indices",
    "main",
    "native_sampling_contract",
    "normalize_arms",
    "reference_indices_for_arm",
    "select_native_conditions",
    "source_id_contract",
    "validate_cli",
    "validate_t2v_resource_lifecycle",
]

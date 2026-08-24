#!/usr/bin/env python3
"""Frozen native RV2V same-process P0/P1/P2 formal prompt matrix.

One WORLD4 process materializes the exact source-video latent and the four
source-derived reference latents once, then calls the unmodified Bernini
``model.sample`` in the fixed order P0a, P1, P2, P0b.  Every call uses native
RV2V, the same source-condition tensor objects, the same seed, and the same
official UniPC scheduler object.  P0b is a bit-exact replay gate for hidden
sampler state.

All hooks in this file are read-only forwarding observers.  They record the
official fresh Gaussian, effective ``set_timesteps`` reset, ``encode_prompt``
inputs/returns, condition storage, rope values, RNG, freeze/eval state, and
memory evidence; they never inject noise or assign model/scheduler state.

The only semantic inputs are the source video and three sealed positive prompt
strings.  Target media/action JSON, RGB or hidden anchors, masks, tracks, flow,
features, embeddings, latents, Q/K/V, and external Gaussian controls are not
accepted by the CLI or read by the generator.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import fcntl
import gc
import hashlib
import json
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


SCHEMA_VERSION = "mev840-native-rv2v-paired-prompt-matrix-formal-v1"
METHOD = "frozen-bernini-native-rv2v-paired-prompt-matrix-formal"
AUTHORITY_SCHEMA = "mev840-native-rv2v-same-process-formal-v1"
PROMPT_MATRIX_SCHEMA = "mev840-native-rv2v-incremental-prompt-matrix-v1"
EXECUTION_ORDER = ("p0a", "p1", "p2", "p0b")
PROMPT_LABEL_BY_CELL = {
    "p0a": "P0",
    "p1": "P1",
    "p2": "P2",
    "p0b": "P0",
}
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
UNIPC_SOURCE_SHA256 = "5bfe1dcf55ebea6dbbf624d3af676b2529b81fbcaf493150d562ec9e1aba3872"
UNIPC_SOURCE_PATH = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/"
    "diffusers/schedulers/scheduling_unipc_multistep.py"
)
FORMAL_CGROUP_LIMIT_BYTES = 64 * 1024**3
FORMAL_CGROUP_MIN_HEADROOM_BYTES = 64 * 1024**2
FORMAL_SLURM_BY_SEED = {
    2027: {"job_id": "143808", "node": "auh7-1b-gpu-292"},
    2028: {"job_id": "147873", "node": "auh7-1b-gpu-284"},
}
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
    parser.add_argument("--prompt-matrix-authority", required=True)
    parser.add_argument(
        "--expected-prompt-matrix-authority-sha256", required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--skip-video-decode", action="store_true")
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


def _plain_authority_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise NativeIdentityCanaryError(f"{label} must be absolute")
    resolved = requested.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise NativeIdentityCanaryError(f"{label} must be a plain file")
    return resolved


def load_prompt_matrix_authority(
    value: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load the sealed same-process authority and its sibling prompt matrix."""

    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise NativeIdentityCanaryError("prompt matrix authority SHA-256 is invalid")
    path = _plain_authority_file(value, label="prompt-matrix-authority")
    if legacy.file_sha256(path) != expected_sha256:
        raise NativeIdentityCanaryError("prompt matrix authority SHA-256 differs")
    try:
        authority = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeIdentityCanaryError("prompt matrix authority is not canonical ASCII JSON") from error
    if authority.get("schema") != AUTHORITY_SCHEMA:
        raise NativeIdentityCanaryError("prompt matrix authority schema differs")
    pairing = authority.get("same_process_pairing")
    if not isinstance(pairing, Mapping):
        raise NativeIdentityCanaryError("same-process pairing authority is absent")
    required_pairing = {
        "execution_order": list(EXECUTION_ORDER),
        "prompt_label_by_execution_cell": dict(PROMPT_LABEL_BY_CELL),
        "source_decode_count": 1,
        "full_source_vae_encode_count": 1,
        "reference_frame_indices": list(RV2V_REFERENCE_INDICES),
        "each_reference_vae_encode_count": 1,
        "rank_zero_condition_broadcast_before_first_sample": True,
        "same_condition_tensor_objects_for_all_four_calls": True,
        "same_seed_for_all_four_calls": True,
        "official_fresh_gaussian_per_call": True,
        "external_gaussian_injection": False,
        "custom_sampler_or_scheduler": False,
        "same_scheduler_object_all_calls": True,
        "no_manual_model_or_scheduler_state_reset_between_calls": True,
        "rope_unregistered_state_observed_not_mutated": True,
        "p0a_p0b_generated_latent_bit_exact_required": True,
    }
    for key, expected in required_pairing.items():
        if pairing.get(key) != expected:
            raise NativeIdentityCanaryError(f"same-process authority differs: {key}")
    runtime_authority = authority.get("runtime_authority")
    if runtime_authority != {
        "unipc_source": {
            "path": UNIPC_SOURCE_PATH,
            "sha256": UNIPC_SOURCE_SHA256,
        },
        "formal_slurm_by_seed": {
            str(seed): {**row, "world_size": ULYSSES_SIZE}
            for seed, row in FORMAL_SLURM_BY_SEED.items()
        },
        "nearest_finite_cgroup_limit_bytes": FORMAL_CGROUP_LIMIT_BYTES,
        "minimum_cgroup_headroom_bytes": FORMAL_CGROUP_MIN_HEADROOM_BYTES,
    }:
        raise NativeIdentityCanaryError("runtime authority differs")
    if authority.get("execution_mode") != {
        "seeds": [2027, 2028],
        "num_inference_steps": 40,
        "decode_cells": ["p0a", "p1", "p2"],
        "latent_only_replay_cells": ["p0b"],
        "exact_regular_file_count_per_seed": 13,
        "scientific_candidate": True,
        "requires_separate_independent_launch_go": True,
        "mechanical_gate_step": "147873.10",
        "mechanical_gate_receipt_sha256": "40d5124281472eb89c7cb9bc8ee9b6436892ee4ec40c3f3af22698ddf5f43172",
        "mechanical_gate_receipt_digest": "98e053ad26144498e66e661088d4aaddf5f67cd969997c37240abda3d0eee2d8",
    }:
        raise NativeIdentityCanaryError("formal execution authority differs")
    gates = authority.get("dynamic_observer_gates")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or gates.get("encode_prompt_calls_per_cell") != 2
        or any(
            value is not True
            for key, value in gates.items()
            if key != "encode_prompt_calls_per_cell"
        )
    ):
        raise NativeIdentityCanaryError("dynamic observer gates are not all required")
    generator = authority.get("generator_contract")
    if (
        not isinstance(generator, Mapping)
        or generator.get("guidance_mode") != "rv2v"
        or generator.get("accepted_external_conditions")
        != ["source_video", "positive_prompt_matrix"]
        or generator.get("target_video_read") is not False
        or generator.get("target_action_json_read") is not False
        or generator.get(
            "target_rgb_mask_box_xy_flow_feature_embedding_latent_qkv_gaussian_read"
        )
        is not False
        or generator.get("anchor_rgb_kv_latent_gaussian_read") is not False
        or generator.get("legacy_activity25_qk_read") is not False
    ):
        raise NativeIdentityCanaryError("generator input authority differs")
    matrix_ref = authority.get("prompt_matrix")
    if not isinstance(matrix_ref, Mapping):
        raise NativeIdentityCanaryError("prompt matrix reference is absent")
    basename = matrix_ref.get("basename")
    matrix_sha = matrix_ref.get("sha256")
    if (
        not isinstance(basename, str)
        or _SAFE_BASENAME.fullmatch(basename) is None
        or not isinstance(matrix_sha, str)
        or _SHA256.fullmatch(matrix_sha) is None
        or matrix_ref.get("labels") != ["P0", "P1", "P2"]
        or matrix_ref.get("only_registered_design_variable")
        != "positive_prompt_utf8"
    ):
        raise NativeIdentityCanaryError("prompt matrix reference differs")
    matrix_path = _plain_authority_file(path.parent / basename, label="prompt matrix")
    if matrix_path.parent != path.parent or legacy.file_sha256(matrix_path) != matrix_sha:
        raise NativeIdentityCanaryError("prompt matrix file identity differs")
    try:
        matrix = json.loads(matrix_path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeIdentityCanaryError("prompt matrix is not canonical ASCII JSON") from error
    if matrix.get("schema") != PROMPT_MATRIX_SCHEMA:
        raise NativeIdentityCanaryError("prompt matrix schema differs")
    rows = matrix.get("prompts")
    if not isinstance(rows, Mapping) or set(rows) != {"P0", "P1", "P2"}:
        raise NativeIdentityCanaryError("prompt matrix labels differ")
    authority_rows = authority.get("prompts")
    if not isinstance(authority_rows, Mapping) or set(authority_rows) != {"P0", "P1", "P2"}:
        raise NativeIdentityCanaryError("same-process prompt rows differ")
    prompts: dict[str, str] = {}
    prompt_rows: dict[str, Any] = {}
    for label in ("P0", "P1", "P2"):
        row = rows[label]
        sealed_row = authority_rows[label]
        if not isinstance(row, Mapping) or not isinstance(sealed_row, Mapping):
            raise NativeIdentityCanaryError(f"prompt row differs: {label}")
        prompt = row.get("full_prompt_utf8")
        if not isinstance(prompt, str) or not prompt.strip() or "\x00" in prompt:
            raise NativeIdentityCanaryError(f"prompt text differs: {label}")
        payload = prompt.encode("utf-8")
        if (
            len(payload) != row.get("full_prompt_utf8_bytes")
            or hashlib.sha256(payload).hexdigest()
            != row.get("full_prompt_utf8_sha256")
        ):
            raise NativeIdentityCanaryError(f"prompt identity differs: {label}")
        for key in (
            "full_prompt_utf8",
            "full_prompt_utf8_bytes",
            "full_prompt_utf8_sha256",
        ):
            if sealed_row.get(key) != row.get(key):
                raise NativeIdentityCanaryError(
                    f"same-process prompt row disagrees with matrix: {label} {key}"
                )
        if (
            type(sealed_row.get("final_task_prompt_utf8_bytes")) is not int
            or not isinstance(sealed_row.get("final_task_prompt_utf8_sha256"), str)
            or _SHA256.fullmatch(sealed_row["final_task_prompt_utf8_sha256"])
            is None
            or type(sealed_row.get("untruncated_token_count")) is not int
            or not 1 <= sealed_row["untruncated_token_count"] <= 512
            or sealed_row.get("terminal_token_id") != 1
        ):
            raise NativeIdentityCanaryError(f"sealed task prompt contract differs: {label}")
        prompts[label] = prompt
        prompt_rows[label] = dict(sealed_row)
    return {
        "authority": authority,
        "authority_path": str(path),
        "authority_sha256": expected_sha256,
        "prompt_matrix": matrix,
        "prompt_matrix_path": str(matrix_path),
        "prompt_matrix_sha256": matrix_sha,
        "prompts": prompts,
        "prompt_rows": prompt_rows,
    }


def validate_cli(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_inference_steps != NUM_INFERENCE_STEPS:
        raise NativeIdentityCanaryError("formal paired runner requires exactly 40 UniPC steps")
    if bool(args.skip_video_decode):
        raise NativeIdentityCanaryError("formal paired runner requires P0a/P1/P2 MP4 decode")
    if type(args.seed) is not int or args.seed not in FORMAL_SLURM_BY_SEED:
        raise NativeIdentityCanaryError("formal seed must be exactly 2027 or 2028")
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
        "expected_prompt_matrix_authority_sha256",
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
    return load_prompt_matrix_authority(
        args.prompt_matrix_authority,
        expected_sha256=args.expected_prompt_matrix_authority_sha256,
    )


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


def _identity_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("shape", "dtype", "numel", "byte_count", "content_sha256", "raw_storage_sha256")
    }


def _tensor_runtime_state(value: Any, *, label: str) -> dict[str, Any]:
    import torch

    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise NativeIdentityCanaryError(f"{label} must be a non-empty tensor")
    identity = value_audit.tensor_identity(value, label=label)
    return {
        "identity": _identity_core(identity),
        "device": str(value.device),
        "data_ptr": int(value.data_ptr()),
        "version": int(value._version),
        "requires_grad": bool(value.requires_grad),
        "contiguous": bool(value.is_contiguous()),
    }


def _condition_runtime_state(
    full_source_latent: Any,
    reference_latents: Mapping[int, Any],
) -> dict[str, Any]:
    return {
        "full_source_video": _tensor_runtime_state(
            full_source_latent, label="full_source_video_runtime"
        ),
        "references": {
            str(index): _tensor_runtime_state(
                reference_latents[index], label=f"source_reference_{index}_runtime"
            )
            for index in RV2V_REFERENCE_INDICES
        },
    }


def _all_rank_object(value: Any, *, world_size: int, label: str) -> list[Any]:
    import torch.distributed as dist

    rows: list[Any] = [None] * world_size
    dist.all_gather_object(rows, value)
    if len(rows) != world_size or any(row is None for row in rows):
        raise NativeIdentityCanaryError(f"{label} all-rank gather differs")
    return rows


def _assert_condition_state_unchanged(
    expected_rows: Sequence[Any],
    current_rows: Sequence[Any],
    *,
    cell: str,
) -> None:
    if list(current_rows) != list(expected_rows):
        raise NativeIdentityCanaryError(
            f"source condition tensor storage or bytes changed in {cell}"
        )


def _torch_rng_identity(device: Any) -> dict[str, Any]:
    import torch

    return {
        "cpu": _identity_core(
            value_audit.tensor_identity(torch.get_rng_state(), label="torch_cpu_rng")
        ),
        "rank_cuda": _identity_core(
            value_audit.tensor_identity(
                torch.cuda.get_rng_state(device), label="torch_rank_cuda_rng"
            )
        ),
    }


def _model_eval_certificate(model: Any) -> dict[str, Any]:
    training = sorted(name for name, module in model.named_modules() if module.training)
    gradients = sorted(
        name for name, parameter in model.named_parameters() if parameter.grad is not None
    )
    if training or gradients:
        raise NativeIdentityCanaryError("model eval/gradient state differs")
    return {
        "all_modules_eval": True,
        "parameter_gradients_present": False,
        "training_module_count": 0,
        "parameter_gradient_count": 0,
    }


def _unique_rope_modules(model: Any) -> list[tuple[str, Any]]:
    rows = [("transformer_1.rope", model.diff_dec.transformer.rope)]
    for name, module in rows:
        if not hasattr(module, "freqs"):
            raise NativeIdentityCanaryError(f"{name} has no mutable freqs tensor")
    return rows


def _capture_rope_pristine(model: Any) -> dict[str, Any]:
    pristine: dict[str, Any] = {}
    for name, module in _unique_rope_modules(model):
        state = _tensor_runtime_state(module.freqs, label=f"{name}.freqs_pristine")
        pristine[name] = {
            "module": module,
            "module_object_id": id(module),
            "identity": state["identity"],
            "initial_state": state,
        }
    return pristine


def _rope_post_state(
    model: Any, rope_pristine: Mapping[str, Any], *, cell: str
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    current_modules = dict(_unique_rope_modules(model))
    if set(current_modules) != set(rope_pristine):
        raise NativeIdentityCanaryError("rope module closure changed")
    for name, row in rope_pristine.items():
        if current_modules[name] is not row["module"]:
            raise NativeIdentityCanaryError(f"{name} module object changed in {cell}")
        current = _tensor_runtime_state(
            row["module"].freqs, label=f"{name}.freqs_after_{cell}"
        )
        if current["identity"] != row["identity"]:
            raise NativeIdentityCanaryError(f"{name} values changed in {cell}")
        current["module_object_id"] = id(current_modules[name])
        rows[name] = current
    return rows


def _scheduler_config_identity(scheduler: Any) -> str:
    return legacy.object_sha256(dict(scheduler.config))


def _scheduler_state(scheduler: Any, *, label: str) -> dict[str, Any]:
    model_outputs = list(getattr(scheduler, "model_outputs", []))
    timestep_list = list(getattr(scheduler, "timestep_list", []))
    return {
        "label": label,
        "class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        "config_sha256": _scheduler_config_identity(scheduler),
        "solver_order": int(scheduler.config.solver_order),
        "num_inference_steps": getattr(scheduler, "num_inference_steps", None),
        "timesteps": _identity_core(
            value_audit.tensor_identity(scheduler.timesteps, label=f"{label}_timesteps")
        ),
        "sigmas": _identity_core(
            value_audit.tensor_identity(scheduler.sigmas, label=f"{label}_sigmas")
        ),
        "model_outputs_none": [item is None for item in model_outputs],
        "timestep_list": [
            None if item is None else int(item.detach().cpu().item())
            for item in timestep_list
        ],
        "lower_order_nums": int(getattr(scheduler, "lower_order_nums", -1)),
        "last_sample_present": getattr(scheduler, "last_sample", None) is not None,
        "step_index": getattr(scheduler, "step_index", None),
        "begin_index": getattr(scheduler, "begin_index", None),
    }


@contextmanager
def _observe_native_scheduler(scheduler: Any):
    """Observe one official sample call without changing scheduler state.

    Diffusers 0.38 deliberately leaves stale ``timestep_list`` entries in
    ``set_timesteps``.  The observer therefore records them instead of clearing
    them.  Effective reset is established from the official mutable fields and
    the P0 replay, never by assigning model or scheduler state here.
    """

    original_set_timesteps = scheduler.set_timesteps
    original_step = scheduler.step
    evidence: dict[str, list[Any]] = {"set_timesteps": [], "steps": []}

    def observed_set_timesteps(*args: Any, **kwargs: Any) -> Any:
        result = original_set_timesteps(*args, **kwargs)
        evidence["set_timesteps"].append(
            _scheduler_state(scheduler, label="after_set_timesteps")
        )
        return result

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        try:
            timestep_value = int(timestep.detach().cpu().item())
        except Exception:
            timestep_value = int(timestep)
        before = _scheduler_state(
            scheduler, label=f"before_step_{len(evidence['steps'])}"
        )
        result = original_step(*args, **kwargs)
        after = _scheduler_state(
            scheduler, label=f"after_step_{len(evidence['steps'])}"
        )
        evidence["steps"].append(
            {"timestep": timestep_value, "before": before, "after": after}
        )
        return result

    scheduler.set_timesteps = observed_set_timesteps
    scheduler.step = observed_step
    try:
        yield evidence
    finally:
        if (
            scheduler.set_timesteps is not observed_set_timesteps
            or scheduler.step is not observed_step
        ):
            raise NativeIdentityCanaryError("scheduler observer changed while active")
        del scheduler.set_timesteps
        del scheduler.step


@contextmanager
def _observe_encode_prompt(model: Any, *, cell: str):
    """Forward exact encode_prompt calls and record their inputs and return."""

    original = model.encode_prompt
    calls: list[dict[str, Any]] = []

    def observed(*args: Any, **kwargs: Any) -> Any:
        input_ids = args[0] if args else kwargs.get("input_ids")
        attention_mask = args[1] if len(args) > 1 else kwargs.get("attention_mask")
        result = original(*args, **kwargs)
        calls.append(
            {
                "input_ids": _identity_core(
                    value_audit.tensor_identity(
                        input_ids, label=f"{cell}_encode_prompt_ids"
                    )
                ),
                "attention_mask": _identity_core(
                    value_audit.tensor_identity(
                        attention_mask, label=f"{cell}_encode_prompt_mask"
                    )
                ),
                "embedding": _identity_core(
                    value_audit.tensor_identity(
                        result, label=f"{cell}_encode_prompt_return"
                    )
                ),
                "nonpadding_token_count": [
                    int(item)
                    for item in attention_mask.gt(0).sum(dim=1).detach().cpu().tolist()
                ],
            }
        )
        return result

    model.encode_prompt = observed
    try:
        yield calls
    finally:
        if model.encode_prompt is not observed:
            raise NativeIdentityCanaryError("encode_prompt observer changed while active")
        del model.encode_prompt


def _scheduler_effective_reset_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "class",
            "config_sha256",
            "solver_order",
            "num_inference_steps",
            "timesteps",
            "sigmas",
            "model_outputs_none",
            "lower_order_nums",
            "last_sample_present",
            "step_index",
            "begin_index",
        )
    }


def _validate_scheduler_observations(
    per_cell: Mapping[str, Mapping[str, Any]], *, expected_steps: int
) -> dict[str, Any]:
    if list(per_cell) != list(EXECUTION_ORDER):
        raise NativeIdentityCanaryError("scheduler cell order differs")
    reset_cores: list[dict[str, Any]] = []
    stale_lists: dict[str, Any] = {}
    for cell in EXECUTION_ORDER:
        row = per_cell[cell]
        set_calls = row.get("set_timesteps")
        step_calls = row.get("steps")
        if not isinstance(set_calls, list) or len(set_calls) != 1:
            raise NativeIdentityCanaryError(
                f"scheduler set_timesteps count differs in {cell}"
            )
        if not isinstance(step_calls, list) or len(step_calls) != expected_steps:
            raise NativeIdentityCanaryError(f"scheduler step count differs in {cell}")
        reset = set_calls[0]
        if (
            reset.get("num_inference_steps") != expected_steps
            or reset.get("solver_order") != 2
            or reset.get("lower_order_nums") != 0
            or reset.get("last_sample_present") is not False
            or reset.get("step_index") is not None
            or reset.get("begin_index") is not None
            or not reset.get("model_outputs_none")
            or any(value is not True for value in reset["model_outputs_none"])
        ):
            raise NativeIdentityCanaryError(
                f"scheduler effective reset differs in {cell}"
            )
        first = step_calls[0]
        if (
            first["before"]["lower_order_nums"] != 0
            or any(value is not True for value in first["before"]["model_outputs_none"])
        ):
            raise NativeIdentityCanaryError(
                f"scheduler first step did not enter first-order mode in {cell}"
            )
        if expected_steps >= 2:
            second = step_calls[1]
            first_timestep = first["timestep"]
            before_second = second["before"]
            if (
                before_second["lower_order_nums"] != 1
                or before_second["timestep_list"][-1:] != [first_timestep]
                or before_second["model_outputs_none"][-1:] != [False]
            ):
                raise NativeIdentityCanaryError(
                    f"scheduler order-2 fresh predecessor proof differs in {cell}"
                )
        reset_cores.append(_scheduler_effective_reset_core(reset))
        stale_lists[cell] = list(reset["timestep_list"])
    if any(row != reset_cores[0] for row in reset_cores[1:]):
        raise NativeIdentityCanaryError(
            "effective scheduler reset or schedule differs across cells"
        )
    return {
        "same_scheduler_object_all_calls": True,
        "set_timesteps_once_per_call": True,
        "step_count_per_call": expected_steps,
        "effective_reset_fields_exact_across_calls": True,
        "stale_timestep_list_recorded": True,
        "stale_timestep_list_by_cell": stale_lists,
        "stale_timestep_list_inactive_on_first_order_step": True,
        "fresh_predecessor_present_before_order2_step": expected_steps >= 2,
        "no_manual_scheduler_state_reset": True,
        "effective_reset_core": reset_cores[0],
        "per_cell": {key: dict(per_cell[key]) for key in EXECUTION_ORDER},
    }


def _linux_cgroup_memory_evidence(
    *, baseline: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Find the nearest finite cgroup-v2 limit and prove no new OOM events.

    Slurm job cgroups can outlive an individual step, so their event counters
    are cumulative.  A formal run therefore seals an early per-rank
    baseline and later requires a zero delta, rather than assuming the inherited
    counters began at zero.
    """

    cgroup = Path("/proc/self/cgroup")
    if not cgroup.is_file() or cgroup.is_symlink():
        raise NativeIdentityCanaryError("process cgroup evidence is unavailable")
    rows = [line.split(":", 2) for line in cgroup.read_text(encoding="ascii").splitlines()]
    matches = [row[2] for row in rows if len(row) == 3 and row[0] == "0" and row[1] == ""]
    if len(matches) != 1 or not matches[0].startswith("/"):
        raise NativeIdentityCanaryError("cgroup-v2 path differs")
    filesystem_root = Path("/sys/fs/cgroup")
    current_path = filesystem_root / matches[0].lstrip("/")
    ancestors: list[dict[str, Any]] = []
    while True:
        paths = {name: current_path / name for name in ("memory.current", "memory.max", "memory.events")}
        if all(path.is_file() and not path.is_symlink() for path in paths.values()):
            maximum_text = paths["memory.max"].read_text(encoding="ascii").strip()
            current_text = paths["memory.current"].read_text(encoding="ascii").strip()
            events: dict[str, int] = {}
            for line in paths["memory.events"].read_text(encoding="ascii").splitlines():
                key, amount = line.split()
                events[key] = int(amount)
            relative = current_path.relative_to(filesystem_root).as_posix()
            row = {
                "relative_path": "/" if relative == "." else "/" + relative,
                "memory_current": int(current_text),
                "memory_max": maximum_text if maximum_text == "max" else int(maximum_text),
                "memory_events": events,
            }
            ancestors.append(row)
            if maximum_text != "max":
                break
        if current_path == filesystem_root:
            raise NativeIdentityCanaryError("no finite cgroup-v2 memory limit found")
        current_path = current_path.parent
    effective = ancestors[-1]
    if effective["memory_max"] != FORMAL_CGROUP_LIMIT_BYTES:
        raise NativeIdentityCanaryError("nearest finite cgroup limit is not 64 GiB")
    headroom = FORMAL_CGROUP_LIMIT_BYTES - int(effective["memory_current"])
    if headroom < FORMAL_CGROUP_MIN_HEADROOM_BYTES:
        raise NativeIdentityCanaryError("formal cgroup headroom is below 64 MiB")
    result = {
        "leaf_relative_path": matches[0],
        "ancestors_through_nearest_finite": ancestors,
        "nearest_finite_relative_path": effective["relative_path"],
        "nearest_finite_limit_bytes": effective["memory_max"],
        "nearest_finite_current_bytes": effective["memory_current"],
        "headroom_bytes": headroom,
        "minimum_required_headroom_bytes": FORMAL_CGROUP_MIN_HEADROOM_BYTES,
        "headroom_gate_passed": True,
        "effective_64_gib_limit": True,
        "oom_event_baseline_by_path": None,
        "oom_event_delta_by_path": None,
        "oom_oom_kill_oom_group_kill_delta_zero": None,
    }
    if baseline is None:
        return result
    baseline_rows = baseline.get("ancestors_through_nearest_finite")
    if not isinstance(baseline_rows, list):
        raise NativeIdentityCanaryError("cgroup OOM baseline is absent")
    baseline_by_path = {
        row.get("relative_path"): row for row in baseline_rows if isinstance(row, Mapping)
    }
    current_by_path = {row["relative_path"]: row for row in ancestors}
    if (
        set(baseline_by_path) != set(current_by_path)
        or baseline.get("nearest_finite_relative_path")
        != result["nearest_finite_relative_path"]
        or baseline.get("nearest_finite_limit_bytes")
        != FORMAL_CGROUP_LIMIT_BYTES
    ):
        raise NativeIdentityCanaryError("cgroup hierarchy changed after baseline")
    baseline_events: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for path in sorted(current_by_path):
        prior_events = baseline_by_path[path].get("memory_events")
        current_events = current_by_path[path].get("memory_events")
        if not isinstance(prior_events, Mapping) or not isinstance(
            current_events, Mapping
        ):
            raise NativeIdentityCanaryError("cgroup memory.events evidence differs")
        baseline_events[path] = {
            key: int(prior_events.get(key, 0))
            for key in ("oom", "oom_kill", "oom_group_kill")
        }
        delta = {
            key: int(current_events.get(key, 0)) - baseline_events[path][key]
            for key in ("oom", "oom_kill", "oom_group_kill")
        }
        if any(value != 0 for value in delta.values()):
            raise NativeIdentityCanaryError("cgroup OOM event delta is nonzero")
        deltas[path] = delta
    result["oom_event_baseline_by_path"] = baseline_events
    result["oom_event_delta_by_path"] = deltas
    result["oom_oom_kill_oom_group_kill_delta_zero"] = True
    return result


def _formal_slurm_context(
    args: argparse.Namespace, authority_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind one formal seed to its authorized holder job, node, and exact step."""

    expected = FORMAL_SLURM_BY_SEED[args.seed]
    sealed = authority_bundle["authority"].get("runtime_authority", {}).get(
        "formal_slurm_by_seed", {}
    ).get(str(args.seed))
    if sealed != {**expected, "world_size": ULYSSES_SIZE}:
        raise NativeIdentityCanaryError("formal Slurm authority differs")
    job_id = os.environ.get("SLURM_JOB_ID")
    step_id = os.environ.get("SLURM_STEP_ID")
    node = socket.gethostname().split(".", 1)[0]
    world_size_text = os.environ.get("WORLD_SIZE")
    if (
        job_id != expected["job_id"]
        or not isinstance(step_id, str)
        or not step_id.isdigit()
        or node != expected["node"]
        or world_size_text != str(ULYSSES_SIZE)
    ):
        raise NativeIdentityCanaryError("formal Slurm job/node/step/WORLD4 differs")
    return {
        "job_id": job_id,
        "step_id": step_id,
        "job_step_id": f"{job_id}.{step_id}",
        "node": node,
        "world_size": int(world_size_text),
    }


def _tokenize_positive_prompt_untruncated(
    tokenizer: Any, prompt: str, *, label: str
) -> tuple[Any, Any, dict[str, Any]]:
    encoded = tokenizer(prompt, **legacy.training_prompt_tokenizer_kwargs())
    if (
        encoded.input_ids.ndim != 2
        or tuple(encoded.input_ids.shape) != tuple(encoded.attention_mask.shape)
        or encoded.input_ids.shape[0] != 1
    ):
        raise NativeIdentityCanaryError(f"raw tokenizer output differs: {label}")
    raw_length = int(encoded.input_ids.shape[1])
    if raw_length > 512:
        raise NativeIdentityCanaryError(f"positive prompt would be truncated: {label}")
    input_ids, attention_mask = legacy._tokenize_training_prompt(tokenizer, prompt)
    if tuple(input_ids.shape) != (1, 512) or tuple(attention_mask.shape) != (1, 512):
        raise NativeIdentityCanaryError(f"padded prompt tokens differ: {label}")
    return input_ids, attention_mask, {
        "raw_token_count_including_special_tokens": raw_length,
        "untruncated": True,
        "padded_shape": [1, 512],
        "eos_token_id": int(tokenizer.eos_token_id),
        "input_ids": _identity_core(
            value_audit.tensor_identity(input_ids, label=f"{label}_positive_ids")
        ),
        "attention_mask": _identity_core(
            value_audit.tensor_identity(
                attention_mask, label=f"{label}_positive_mask"
            )
        ),
    }


def _save_paired_outputs(
    *,
    output_dir: Path,
    generated: Mapping[str, Any],
    vae: Any,
    bucket_hw: Sequence[int],
    device: Any,
    save_output_fn: Any,
    skip_video_decode: bool,
) -> dict[str, Any]:
    import torch
    from bernini.pipeline import _vae_decode
    from tools import materialize_vae

    if skip_video_decode:
        raise NativeIdentityCanaryError("formal runner cannot skip candidate MP4 decode")
    if set(generated) != set(EXECUTION_ORDER):
        raise NativeIdentityCanaryError("formal generated-cell closure differs")
    expected_hw = (int(bucket_hw[0]), int(bucket_hw[1]))
    outputs: dict[str, Any] = {}
    vae.to(device)
    for cell in EXECUTION_ORDER:
        latent = generated[cell]
        clean_latent = _save_normalized_clean_latent_atomically(
            output_dir / f"{cell}.normalized-clean-latent.safetensors",
            latent,
        )
        if cell == "p0b":
            outputs[cell] = {
                "path": None,
                "sha256": None,
                "video_decode_skipped": True,
                "replay_gate_only": True,
                "normalized_clean_latent": clean_latent,
            }
            continue
        with torch.no_grad():
            decoded = _vae_decode(vae, latent)
        if tuple(int(item) for item in decoded.shape) != (
            FRAME_COUNT,
            expected_hw[0],
            expected_hw[1],
            3,
        ):
            raise NativeIdentityCanaryError(f"{cell} decoded shape differs")
        path = output_dir / f"{cell}.mp4"
        value_audit.save_video_atomically(
            decoded,
            path,
            fps=FPS,
            save_output_fn=save_output_fn,
        )
        del decoded
        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(path)
        legacy.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
        if tuple(encoded_hw) != expected_hw:
            raise NativeIdentityCanaryError(f"{cell} encoded geometry differs")
        del encoded
        outputs[cell] = {
            "path": str(path),
            "sha256": legacy.file_sha256(path),
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "height": expected_hw[0],
            "width": expected_hw[1],
            "video_decode_skipped": False,
            "normalized_clean_latent": clean_latent,
        }
        _trim_host_allocator()
        torch.cuda.empty_cache()
    return outputs


def _build_paired_receipt(
    *,
    args: argparse.Namespace,
    authority_bundle: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
    freeze_certificate: Mapping[str, Any],
    latent_geometry: Mapping[str, Any],
    condition_identities: Mapping[str, Any],
    source_condition_artifact: Mapping[str, Any],
    initial_noise_artifacts: Mapping[str, Any],
    generated_identities: Mapping[str, Any],
    outputs: Mapping[str, Any],
    resource_lifecycle: Mapping[str, Any],
    paired_contract: Mapping[str, Any],
) -> dict[str, Any]:
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
        "arms": ["rv2v"],
        "execution_cells": list(EXECUTION_ORDER),
        "input": {
            "source_video_path": str(source_path),
            "source_video_sha256": source_sha256,
            "prompt_matrix_authority_path": authority_bundle["authority_path"],
            "prompt_matrix_authority_sha256": authority_bundle["authority_sha256"],
            "prompt_matrix_path": authority_bundle["prompt_matrix_path"],
            "prompt_matrix_sha256": authority_bundle["prompt_matrix_sha256"],
            "accepted_external_conditions": [
                "source_video",
                "positive_prompt_matrix",
            ],
            "target_video": False,
            "target_action_json": False,
            "target_rgb_mask_box_xy_flow_feature_embedding_latent_qkv_gaussian": False,
            "anchor_rgb_kv_latent_gaussian": False,
            "legacy_activity25_qk": False,
            "external_reference_image_or_video": False,
            "external_mask_flow_pose_track_trajectory": False,
            "external_first_frame_anchor": False,
        },
        "preprocessing": dict(source_metadata),
        "prompt_contract": dict(prompt_contract),
        "conditioning": {
            "rv2v": {
                "full_source_video_count": 1,
                "source_derived_reference_count": 4,
                "source_frame_indices": list(RV2V_REFERENCE_INDICES),
                "reference_encoding": "independent_rgb_frame_to_wan_vae_[1,C,1,H,W]",
                "reference_from_temporal_video_latent_slice": False,
                "source_ids": source_id_contract("rv2v"),
            }
        },
        "sampling": {
            cell: {
                **native_sampling_contract(
                    "rv2v", steps=args.num_inference_steps, seed=args.seed
                ),
                "prompt_label": PROMPT_LABEL_BY_CELL[cell],
                "target_initialization": TARGET_INITIALIZATION,
                "target_mixed_with_source_latent": False,
                "custom_sampler_or_scheduler": False,
                "official_fresh_gaussian_per_call": True,
                "external_initial_noise_injection": False,
                "single_expert": "transformer_1",
                "ulysses_size": ULYSSES_SIZE,
            }
            for cell in EXECUTION_ORDER
        },
        "latent_geometry": dict(latent_geometry),
        "condition_identities": dict(condition_identities),
        "source_condition_artifact": dict(source_condition_artifact),
        "initial_noise_artifacts": dict(initial_noise_artifacts),
        "generated_identities": dict(generated_identities),
        "outputs": dict(outputs),
        "freeze_certificate": dict(freeze_certificate),
        "runtime_versions": dict(runtime_versions),
        "resource_lifecycle": dict(resource_lifecycle),
        "paired_same_process_contract": dict(paired_contract),
        "execution_mode": {
            "num_inference_steps": int(args.num_inference_steps),
            "formal_generation": True,
            "seed": int(args.seed),
            "decoded_cells": ["p0a", "p1", "p2"],
            "latent_only_replay_cells": ["p0b"],
            "scientific_candidate": True,
        },
        "interpretation": {
            "purpose": "paired_native_rv2v_prompt_ablation_with_one_materialized_source_condition",
            "training_performed": False,
            "formal_generation_proves_video_quality_before_visual_review": False,
            "formal_generation_proves_action_gain_before_observer_scoring": False,
            "best_cell_selected": False,
        },
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def _legacy_single_prompt_main_unused(argv: Optional[Sequence[str]] = None) -> int:
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run P0a/P1/P2/P0b in one WORLD4 process with one source condition."""

    args = build_parser().parse_args(argv)
    authority_bundle = validate_cli(args)
    if os.environ.get("NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED") is not None:
        raise NativeIdentityCanaryError(
            "T2V-only text-encoder residency mode is forbidden for paired RV2V"
        )
    local_slurm_context = _formal_slurm_context(args, authority_bundle)
    cgroup_memory_baseline = _linux_cgroup_memory_evidence()
    output_dir = _resolve_fresh_output_dir(args.output_dir)
    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute():
        raise NativeIdentityCanaryError("source-video must be absolute")
    source_path = legacy._plain_file(
        source_requested.resolve(strict=True), label="source video"
    )
    manifest_path = Path(args.checkpoint_content_manifest).expanduser()
    if not manifest_path.is_absolute():
        raise NativeIdentityCanaryError(
            "checkpoint-content-manifest must be absolute"
        )

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
        raise NativeIdentityCanaryError("paired canary requires four AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)
    slurm_rows = _all_rank_object(
        local_slurm_context,
        world_size=distributed.world_size,
        label="formal_slurm_context",
    )
    if any(row != slurm_rows[0] for row in slurm_rows[1:]):
        raise NativeIdentityCanaryError("formal Slurm binding differs across WORLD4")
    slurm_context = {**dict(slurm_rows[0]), "all_rank_exact": True}
    cgroup_baseline_rows = _all_rank_object(
        cgroup_memory_baseline,
        world_size=distributed.world_size,
        label="cgroup_memory_baseline",
    )

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
    if (
        not isinstance(checkpoint_result, Mapping)
        or checkpoint_result.get("ok") is not True
    ):
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
    eval_before = _model_eval_certificate(model)

    source_tensor, source_metadata, source_sha256 = (
        source_audit.prepare_hashed_source_snapshot(source_path)
    )
    if source_sha256 != args.expected_source_sha256:
        raise NativeIdentityCanaryError("source video SHA-256 differs")
    bucket_hw = tuple(
        int(item) for item in source_metadata["source_derived_bucket_hw"]
    )

    raw_prompts = dict(authority_bundle["prompts"])
    prompts = {
        cell: build_task_prompt(
            "rv2v",
            raw_prompts[PROMPT_LABEL_BY_CELL[cell]],
            prompt_cleaner=prompt_clean,
        )
        for cell in EXECUTION_ORDER
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    if (
        tokenizer.padding_side != "right"
        or tokenizer.init_kwargs.get("fix_mistral_regex") is not True
    ):
        raise NativeIdentityCanaryError("tokenizer contract differs")
    authority_prompt_rows = authority_bundle["authority"].get("prompts")
    if not isinstance(authority_prompt_rows, Mapping):
        raise NativeIdentityCanaryError("authority prompt token pins are absent")
    positive_tokens: dict[str, tuple[Any, Any]] = {}
    prompt_contract: dict[str, Any] = {}
    for cell in EXECUTION_ORDER:
        label = PROMPT_LABEL_BY_CELL[cell]
        input_ids, attention_mask, token_evidence = (
            _tokenize_positive_prompt_untruncated(
                tokenizer, prompts[cell], label=cell
            )
        )
        pin = authority_prompt_rows.get(label)
        prompt_bytes = prompts[cell].encode("utf-8")
        if (
            not isinstance(pin, Mapping)
            or len(prompt_bytes) != pin.get("final_task_prompt_utf8_bytes")
            or hashlib.sha256(prompt_bytes).hexdigest()
            != pin.get("final_task_prompt_utf8_sha256")
            or token_evidence["raw_token_count_including_special_tokens"]
            != pin.get("untruncated_token_count")
            or token_evidence["eos_token_id"] != pin.get("terminal_token_id")
        ):
            raise NativeIdentityCanaryError(f"sealed prompt/token pin differs: {cell}")
        positive_tokens[cell] = (input_ids, attention_mask)
        prompt_contract[cell] = {
            "prompt_label": label,
            "raw_prompt_utf8_sha256": authority_bundle["prompt_rows"][label][
                "full_prompt_utf8_sha256"
            ],
            "final_task_prompt_utf8_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "final_task_prompt_utf8_bytes": len(prompt_bytes),
            "training_task_name": "vr2v",
            "guidance_mode": "rv2v",
            "tokenizer": token_evidence,
        }
    if prompts["p0a"] != prompts["p0b"]:
        raise NativeIdentityCanaryError("P0 replay prompt bytes differ")
    negative_ids, negative_mask = legacy._tokenize_renderer_negative(
        tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT
    )
    negative_token_core = {
        "input_ids": _identity_core(
            value_audit.tensor_identity(negative_ids, label="negative_input_ids")
        ),
        "attention_mask": _identity_core(
            value_audit.tensor_identity(negative_mask, label="negative_attention_mask")
        ),
    }

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
        full_source_latent = _vae_encode(vae, source_pixels).contiguous()
        reference_latents = {
            index: _vae_encode(
                vae,
                source_pixels[:, :, index : index + 1, :, :].contiguous(),
            ).contiguous()
            for index in RV2V_REFERENCE_INDICES
        }
    vae_z_dim = int(vae.config.z_dim)
    del source_pixels, source_tensor

    condition_broadcasts = {
        "references": {
            str(index): _broadcast_condition_from_rank_zero(
                latent,
                label=f"source_reference_{index}",
                world_size=distributed.world_size,
            )
            for index, latent in reference_latents.items()
        },
        "full_source_video": _broadcast_condition_from_rank_zero(
            full_source_latent,
            label="full_source_video",
            world_size=distributed.world_size,
        ),
    }
    latent_geometry = _latent_geometry_receipt(bucket_hw=bucket_hw, z_dim=vae_z_dim)
    expected_video_shape = tuple(latent_geometry["video_latent_shape"])
    expected_reference_shape = tuple(latent_geometry["reference_latent_shape"])
    if tuple(full_source_latent.shape) != expected_video_shape:
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
        },
        "full_source_video": _all_rank_tensor_identity(
            full_source_latent,
            label="full_source_video",
            world_size=distributed.world_size,
        ),
    }
    condition_baseline_rows = _all_rank_object(
        _condition_runtime_state(full_source_latent, reference_latents),
        world_size=distributed.world_size,
        label="source_condition_runtime_baseline",
    )
    condition_kwargs = select_native_conditions(
        "rv2v",
        full_source_latent=full_source_latent,
        reference_latents=reference_latents,
    )
    condition_object_ids = {
        "multi_video_list_id": id(condition_kwargs["multi_video_vae_latents"]),
        "multi_image_list_id": id(condition_kwargs["multi_image_vae_latents"]),
        "full_source_tensor_id": id(full_source_latent),
        "reference_tensor_ids": {
            str(index): id(reference_latents[index])
            for index in RV2V_REFERENCE_INDICES
        },
    }

    # The source VAE is no longer needed by any sampler call.  Retiring all four
    # replicas here is necessary under the 64-GiB node cgroup; rank zero reloads
    # a decode-only VAE only after the renderer is retired in formal mode.
    del vae
    _trim_host_allocator()
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
    condition_evidence: dict[str, Any] = {}
    scheduler_evidence: dict[str, Any] = {}
    prompt_encoder_evidence: dict[str, Any] = {}
    rng_evidence: dict[str, Any] = {}
    rope_evidence: dict[str, Any] = {}
    model_evidence: dict[str, Any] = {}
    memory_evidence: dict[str, Any] = {}
    noise_raw_hashes: dict[str, str] = {}
    scheduler = model.diff_dec.scheduler
    scheduler_object_id = id(scheduler)
    import inspect

    scheduler_source_value = inspect.getsourcefile(type(scheduler))
    if not isinstance(scheduler_source_value, str):
        raise NativeIdentityCanaryError("UniPC scheduler source path is unavailable")
    scheduler_source_path = legacy._plain_file(
        Path(scheduler_source_value).resolve(strict=True),
        label="official UniPC scheduler source",
    )
    if (
        str(scheduler_source_path) != UNIPC_SOURCE_PATH
        or legacy.file_sha256(scheduler_source_path) != UNIPC_SOURCE_SHA256
    ):
        raise NativeIdentityCanaryError("official UniPC scheduler source SHA differs")
    rope_pristine = _capture_rope_pristine(model)

    with torch.no_grad():
        for cell in EXECUTION_ORDER:
            if id(model.diff_dec.scheduler) != scheduler_object_id:
                raise NativeIdentityCanaryError("native scheduler object changed")
            input_ids, attention_mask = positive_tokens[cell]
            before_conditions = _all_rank_object(
                _condition_runtime_state(full_source_latent, reference_latents),
                world_size=distributed.world_size,
                label=f"source_condition_before_{cell}",
            )
            _assert_condition_state_unchanged(
                condition_baseline_rows, before_conditions, cell=f"before_{cell}"
            )
            rng_before = _torch_rng_identity(device)
            with _observe_native_scheduler(scheduler) as observed_scheduler:
                with _observe_encode_prompt(model, cell=cell) as observed_prompts:
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
                                    "rv2v",
                                    steps=args.num_inference_steps,
                                    seed=args.seed,
                                ),
                            ),
                            wan_diffusion_module=wan_diffusion,
                            expected_shape=expected_video_shape,
                            expected_device=device,
                            expected_seed=args.seed,
                        )
                    )
            if id(model.diff_dec.scheduler) != scheduler_object_id:
                raise NativeIdentityCanaryError(
                    "model.sample replaced the native scheduler object"
                )
            if tuple(generated_latent.shape) != expected_video_shape:
                raise NativeIdentityCanaryError(
                    f"{cell} generated latent shape differs"
                )
            rng_after = _torch_rng_identity(device)
            if rng_after != rng_before:
                raise NativeIdentityCanaryError(
                    f"global torch RNG state changed in {cell}"
                )
            after_conditions = _all_rank_object(
                _condition_runtime_state(full_source_latent, reference_latents),
                world_size=distributed.world_size,
                label=f"source_condition_after_{cell}",
            )
            _assert_condition_state_unchanged(
                condition_baseline_rows, after_conditions, cell=f"after_{cell}"
            )
            condition_evidence[cell] = {
                "before": before_conditions,
                "after": after_conditions,
                "raw_bytes_data_ptr_version_exact": True,
            }

            scheduler_rows = _all_rank_object(
                observed_scheduler,
                world_size=distributed.world_size,
                label=f"scheduler_observer_{cell}",
            )
            if any(row != scheduler_rows[0] for row in scheduler_rows[1:]):
                raise NativeIdentityCanaryError(
                    f"scheduler evidence differs across ranks in {cell}"
                )
            scheduler_evidence[cell] = dict(scheduler_rows[0])

            if len(observed_prompts) != 2:
                raise NativeIdentityCanaryError(
                    f"encode_prompt call count differs in {cell}"
                )
            positive_core = {
                "input_ids": prompt_contract[cell]["tokenizer"]["input_ids"],
                "attention_mask": prompt_contract[cell]["tokenizer"][
                    "attention_mask"
                ],
            }
            classified: dict[str, Any] = {}
            for call in observed_prompts:
                token_core = {
                    "input_ids": call["input_ids"],
                    "attention_mask": call["attention_mask"],
                }
                if token_core == positive_core and "positive" not in classified:
                    classified["positive"] = call
                elif token_core == negative_token_core and "negative" not in classified:
                    classified["negative"] = call
                else:
                    raise NativeIdentityCanaryError(
                        f"encode_prompt input identity differs in {cell}"
                    )
            if set(classified) != {"positive", "negative"}:
                raise NativeIdentityCanaryError(
                    f"positive/negative encode_prompt calls differ in {cell}"
                )
            prompt_rows = _all_rank_object(
                classified,
                world_size=distributed.world_size,
                label=f"encode_prompt_observer_{cell}",
            )
            if any(row != prompt_rows[0] for row in prompt_rows[1:]):
                raise NativeIdentityCanaryError(
                    f"T5 token or embedding differs across ranks in {cell}"
                )
            prompt_encoder_evidence[cell] = dict(prompt_rows[0])

            noise_rank_identity = _all_rank_tensor_identity(
                noise_capture.tensor,
                label=f"official_initial_gaussian_{cell}",
                world_size=distributed.world_size,
            )
            generated_identity = _all_rank_tensor_identity(
                generated_latent,
                label=f"generated_{cell}",
                world_size=distributed.world_size,
            )
            generated_identities[cell] = generated_identity
            initial_noise_rank_identities[cell] = noise_rank_identity
            noise_raw_hashes[cell] = noise_capture.raw_value_sha256
            if distributed.rank == 0:
                generated[cell] = generated_latent
                initial_noise_captures[cell] = noise_capture
            rope_evidence[cell] = _rope_post_state(model, rope_pristine, cell=cell)
            model_evidence[cell] = {
                "freeze_certificate": source_audit.model_freeze_certificate(model),
                "eval_certificate": _model_eval_certificate(model),
            }
            if model_evidence[cell]["freeze_certificate"] != freeze_before:
                raise NativeIdentityCanaryError(
                    f"frozen model certificate changed in {cell}"
                )
            rng_evidence[cell] = {
                "before": rng_before,
                "after": rng_after,
                "unchanged": True,
            }
            del generated_latent
            if distributed.rank != 0:
                del noise_capture
            _trim_host_allocator()
            torch.cuda.empty_cache()
            local_memory = {
                "process": _linux_process_memory_kib(),
                "cgroup": _linux_cgroup_memory_evidence(
                    baseline=cgroup_memory_baseline
                ),
                "host_allocator_trim_called": True,
                "torch_cuda_empty_cache_called": True,
            }
            memory_evidence[cell] = _all_rank_object(
                local_memory,
                world_size=distributed.world_size,
                label=f"memory_after_{cell}",
            )

    if len(set(noise_raw_hashes.values())) != 1:
        raise NativeIdentityCanaryError(
            "same seed and target shape did not yield one exact Gaussian"
        )
    p0a_identity = _identity_core(generated_identities["p0a"]["identity"])
    p0b_identity = _identity_core(generated_identities["p0b"]["identity"])
    if p0a_identity != p0b_identity:
        raise NativeIdentityCanaryError("P0 native replay latent is not bit exact")
    negative_rows = [
        prompt_encoder_evidence[cell]["negative"] for cell in EXECUTION_ORDER
    ]
    if any(row != negative_rows[0] for row in negative_rows[1:]):
        raise NativeIdentityCanaryError(
            "negative tokens or embedding differ across cells"
        )
    if (
        prompt_encoder_evidence["p0a"]["positive"]
        != prompt_encoder_evidence["p0b"]["positive"]
    ):
        raise NativeIdentityCanaryError("P0 positive embedding replay differs")
    scheduler_contract = _validate_scheduler_observations(
        scheduler_evidence, expected_steps=args.num_inference_steps
    )
    scheduler_contract["official_source_path"] = str(scheduler_source_path)
    scheduler_contract["official_source_sha256"] = UNIPC_SOURCE_SHA256
    p0a_rope_state = rope_evidence["p0a"]
    if any(
        rope_evidence[cell] != p0a_rope_state
        for cell in ("p1", "p2", "p0b")
    ):
        raise NativeIdentityCanaryError(
            "single-expert rope runtime state differs after P0a"
        )
    freeze_after = source_audit.model_freeze_certificate(model)
    eval_after = _model_eval_certificate(model)
    if freeze_after != freeze_before or eval_after != eval_before:
        raise NativeIdentityCanaryError("final frozen/eval certificate changed")

    paired_contract = {
        "schema": "mev840-native-rv2v-paired-same-process-contract-v1",
        "execution_order": list(EXECUTION_ORDER),
        "same_process_pid": os.getpid(),
        "same_process_hostname": socket.gethostname(),
        "source_materialization": {
            "source_decode_count_per_rank": 1,
            "full_source_vae_encode_count_per_rank": 1,
            "reference_vae_encode_count_per_rank": {
                str(index): 1 for index in RV2V_REFERENCE_INDICES
            },
            "one_logical_rank_zero_authoritative_condition": True,
            "rank_zero_broadcast_before_first_sample": True,
            "source_vae_retired_before_first_sample": True,
        },
        "condition_runtime": {
            "baseline_all_rank": condition_baseline_rows,
            "same_condition_tensor_objects_for_all_calls": True,
            "local_object_ids": condition_object_ids,
            "per_cell": condition_evidence,
        },
        "official_initial_gaussian": {
            "four_call_raw_value_exact": True,
            "all_rank_exact_per_call": True,
            "raw_value_sha256": next(iter(noise_raw_hashes.values())),
        },
        "p0_replay": {
            "generated_latent_bit_exact": True,
            "positive_tokens_and_embedding_bit_exact": True,
            "identity": p0a_identity,
        },
        "scheduler": scheduler_contract,
        "rope": {
            "observer_only": True,
            "manual_state_assignment": False,
            "observed_module_names": ["transformer_1.rope"],
            "unregistered_freq_values_unchanged_after_each_call": True,
            "p1_p2_p0b_full_state_exact_to_p0a": True,
            "initial": {
                name: {
                    **row["initial_state"],
                    "module_object_id": row["module_object_id"],
                }
                for name, row in rope_pristine.items()
            },
            "per_cell": rope_evidence,
        },
        "prompt_encoder": {
            "encode_prompt_calls_per_cell": 2,
            "observer_changed_return_value": False,
            "positive_tokens_and_embedding_world4_exact_per_cell": True,
            "negative_tokens_and_embedding_world4_exact_across_cells": True,
            "p0_positive_replay_exact": True,
            "per_cell": prompt_encoder_evidence,
        },
        "rng": {"global_torch_rng_unchanged_each_call": True, "per_cell": rng_evidence},
        "model": {
            "freeze_and_eval_unchanged_each_call": True,
            "no_manual_model_state_reset_between_calls": True,
            "per_cell": model_evidence,
        },
        "memory": {
            "host_allocator_trim_after_each_call": True,
            "torch_cuda_empty_cache_after_each_call": True,
            "cgroup_baseline_all_rank": cgroup_baseline_rows,
            "per_cell_all_rank": memory_evidence,
        },
        "slurm": slurm_context,
        "current_authorized_overlay_runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": legacy.file_sha256(Path(__file__).resolve()),
            "upstream_release_entrypoint_authorized": False,
        },
        "target_media_or_action_json_read": False,
        "external_hidden_qkv_latent_gaussian_control": False,
    }

    del model, scheduler, rope_pristine
    _trim_host_allocator()
    torch.cuda.empty_cache()
    dist.barrier()
    terminal_memory_rows = _all_rank_object(
        {
            "process": _linux_process_memory_kib(),
            "cgroup": _linux_cgroup_memory_evidence(
                baseline=cgroup_memory_baseline
            ),
            "host_allocator_trim_called": True,
            "torch_cuda_empty_cache_called": True,
        },
        world_size=distributed.world_size,
        label="terminal_memory_after_renderer_retirement",
    )
    paired_contract["memory"][
        "terminal_after_renderer_retirement_all_rank"
    ] = terminal_memory_rows
    resource_lifecycle = _resource_lifecycle_receipt(
        t2v_vae_deferred_until_post_sampling=True,
        world4_load_completion_gate=world4_load_completion_gate,
        world4_t2v_text_encoder_gpu_residency_gate=None,
    )
    runtime_versions = {
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "transformers": transformers_version,
        "diffusers": diffusers_version,
    }
    initial_noise_artifacts = None
    source_condition_artifact = None
    outputs = None
    if distributed.rank == 0:
        output_dir.mkdir(parents=False, exist_ok=False)
        initial_noise_artifacts = {
            cell: _save_initial_noise_atomically(
                output_dir / f"{cell}.official-initial-gaussian.safetensors",
                initial_noise_captures[cell],
                all_rank_identity=initial_noise_rank_identities[cell],
            )
            for cell in EXECUTION_ORDER
        }
        source_condition_artifact = _save_normalized_clean_latent_atomically(
            output_dir / "source.normalized-clean-latent.safetensors",
            full_source_latent,
            artifact_role="source_video_condition",
        )

    # Persist source/noise evidence, then retire every renderer alias and source
    # condition before rank zero is allowed to construct the decode-only VAE.
    dist.barrier()
    del full_source_latent, reference_latents, condition_kwargs
    del positive_tokens, negative_ids, negative_mask, input_ids, attention_mask
    if distributed.rank == 0:
        del noise_capture
    del initial_noise_captures
    _trim_host_allocator()
    torch.cuda.empty_cache()
    dist.barrier()
    predecode_retirement_rows = _all_rank_object(
        {
            "process": _linux_process_memory_kib(),
            "cgroup": _linux_cgroup_memory_evidence(
                baseline=cgroup_memory_baseline
            ),
            "host_allocator_trim_called": True,
            "torch_cuda_empty_cache_called": True,
            "renderer_scheduler_and_rope_retired": True,
            "source_conditions_and_noise_captures_retired": True,
            "rank_zero_proposal_latents_retained_only_for_decode": True,
        },
        world_size=distributed.world_size,
        label="formal_predecode_renderer_and_condition_retirement",
    )
    paired_contract["memory"][
        "predecode_renderer_and_condition_retirement_all_rank"
    ] = predecode_retirement_rows
    resource_lifecycle.update(
        {
            "renderer_scheduler_and_rope_aliases_retired_before_rank_zero_decode_vae_load": True,
            "source_conditions_and_noise_captures_retired_before_rank_zero_decode_vae_load": True,
            "world4_predecode_retirement_barrier_completed": True,
        }
    )

    if distributed.rank == 0:
        decode_vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        decode_vae.eval().requires_grad_(False)
        outputs = _save_paired_outputs(
            output_dir=output_dir,
            generated=generated,
            vae=decode_vae,
            bucket_hw=bucket_hw,
            device=device,
            save_output_fn=save_output,
            skip_video_decode=bool(args.skip_video_decode),
        )
        del decode_vae

    # The formal terminal gate is intentionally later than decoding.  Retire
    # the rank-zero decode VAE and all four proposal tensors before taking the
    # final all-rank cgroup snapshot used by the formal postflight.
    del generated
    _trim_host_allocator()
    torch.cuda.empty_cache()
    dist.barrier()
    post_decode_terminal_memory_rows = _all_rank_object(
        {
            "process": _linux_process_memory_kib(),
            "cgroup": _linux_cgroup_memory_evidence(
                baseline=cgroup_memory_baseline
            ),
            "host_allocator_trim_called": True,
            "torch_cuda_empty_cache_called": True,
            "decoder_retired": True,
            "source_conditions_and_held_latents_retired": True,
        },
        world_size=distributed.world_size,
        label="formal_terminal_memory_after_decode_and_tensor_retirement",
    )
    paired_contract["memory"][
        "terminal_after_decode_and_all_tensors_retired_all_rank"
    ] = post_decode_terminal_memory_rows
    paired_contract["memory"]["decode_completed_before_terminal_gate"] = True
    paired_contract["memory"]["all_held_latents_retired_before_terminal_gate"] = True
    resource_lifecycle.update(
        {
            "rank_zero_decode_vae_loaded_only_after_renderer_retirement": True,
            "rank_zero_decode_vae_cpu_materialization_count_after_decode": 0,
            "rank_zero_decode_vae_retired_before_final_memory_gate": True,
            "all_rank_conditions_and_held_latents_retired_before_final_memory_gate": True,
            "world4_post_decode_retirement_barrier_completed": True,
        }
    )

    if distributed.rank == 0:
        if not isinstance(initial_noise_artifacts, Mapping):
            raise NativeIdentityCanaryError("formal noise artifact closure differs")
        if not isinstance(source_condition_artifact, Mapping):
            raise NativeIdentityCanaryError("formal source artifact closure differs")
        if not isinstance(outputs, Mapping):
            raise NativeIdentityCanaryError("formal output closure differs")
        receipt = _build_paired_receipt(
            args=args,
            authority_bundle=authority_bundle,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            prompt_contract=prompt_contract,
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
            paired_contract=paired_contract,
        )
        value_audit.write_receipt_atomically(output_dir / "receipt.json", receipt)
        print(legacy.canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier()
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

#!/usr/bin/env python3
"""First real-checkpoint GRAFT Phase-A native single-cell GPU canary.

This runner is intentionally independent from the historical SSFT programs and
from ``run_identity_rebinder_gpu_structural_canary_v1.py``.  It executes one
active cell of the committed Phase-A closure at exact40 schedule index 33:

* one exact81 source is encoded once to the frozen Bernini VAE latent ``z_S``;
* a deterministically keyed fresh Gaussian ``eps`` is materialized;
* ``x = (1 - sigma) * z_S + sigma * eps`` is built in FP32;
* the canonical semantic no-op R2V positive condition and pinned renderer
  negative condition share one native source1-prefix/target0-suffix V pack;
* four real ``shared_step`` calls perform detached measurement, vendor APG leaf
  VJP, then negative/positive graph replay and backward;
* a fresh IdentityRebinder atlas is rebuilt inside every forward context, with
  the two replay atlases retaining graphs into the external atlas owner.

No optimizer is constructed and no parameter update is performed.  The only
authority of a passing receipt is real-checkpoint wiring and flow-matching
gradient reachability at one scheduler cell.  It is not evidence of semantic
action success, visual quality, beneficial training, or full-sampler parity.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

import graft_phase_a_native_training_closure_v1 as phase_core
import identity_rebinder_v1 as rebinder
import infer_lora as legacy
import infer_native_i_axis_exact81_canary as cell_registry
import infer_native_identity_generation_canary as native
import infer_source_kv_carrier_oracle as source_audit
import inference_sigma_strata as sigma_strata
import source_kv_route_batches as route_batches
import tri_branch_unipc as sampler_contract


SCHEMA_VERSION = "bernini-graft-phase-a-native-gpu-canary-v1"
METHOD = "graft-phase-a-native-source-only-v2v-apg-fm-gradient-canary"
NOISE_SCHEMA_VERSION = "bernini-graft-phase-a-keyed-fresh-gaussian-v1"
LOCAL_RESULT_SCHEMA_VERSION = "bernini-graft-phase-a-native-local-result-v1"
ROUTE_FACTORY_SCHEMA_VERSION = "bernini-graft-phase-a-fresh-atlas-route-factory-v1"
ACTIVE_SCHEDULE_INDEX = 33
WORLD_SIZE = 4
SP_SIZE = 4
FRAME_COUNT = 81
LATENT_PHASES = 21
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class GraftPhaseANativeGPUCanaryError(RuntimeError):
    """Raised instead of publishing ambiguous Phase-A GPU evidence."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return finite, deterministic ASCII JSON bytes."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise GraftPhaseANativeGPUCanaryError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_mapping(
    value: Mapping[str, Any], *, digest_field: str = "digest"
) -> Mapping[str, Any]:
    """Seal a mapping and immediately JSON-own/reverify the result."""

    if not isinstance(value, Mapping) or digest_field in value:
        raise GraftPhaseANativeGPUCanaryError("seal input or digest field differs")
    unsigned = json.loads(canonical_json_bytes(dict(value)).decode("ascii"))
    digest = object_sha256(unsigned)
    sealed = {**unsigned, digest_field: digest}
    serialized = canonical_json_bytes(sealed)
    owned = json.loads(serialized.decode("ascii"))
    claimed = owned.pop(digest_field, None)
    if claimed != object_sha256(owned):
        raise GraftPhaseANativeGPUCanaryError(
            "newly sealed canonical receipt digest does not recompute"
        )
    owned[digest_field] = claimed
    if canonical_json_bytes(owned) != serialized:
        raise GraftPhaseANativeGPUCanaryError(
            "newly sealed receipt changed during immediate canonical roundtrip"
        )
    return owned


def own_and_verify_receipt(
    value: Mapping[str, Any], *, digest_field: str = "digest"
) -> tuple[Mapping[str, Any], bytes]:
    """Immediately serialize a producer receipt and recompute its digest."""

    if not isinstance(value, Mapping):
        raise GraftPhaseANativeGPUCanaryError("receipt must be a mapping")
    serialized = canonical_json_bytes(dict(value))
    owned = json.loads(serialized.decode("ascii"))
    claimed = owned.pop(digest_field, None)
    if (
        not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or object_sha256(owned) != claimed
    ):
        raise GraftPhaseANativeGPUCanaryError(
            f"authenticated receipt {digest_field} differs"
        )
    owned[digest_field] = claimed
    if canonical_json_bytes(owned) != serialized:
        raise GraftPhaseANativeGPUCanaryError(
            "authenticated receipt canonical roundtrip differs"
        )
    return owned, serialized


def file_sha256(path: Path | str) -> str:
    candidate = Path(path)
    descriptor = os.open(
        candidate,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GraftPhaseANativeGPUCanaryError(f"not a plain file: {candidate}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise GraftPhaseANativeGPUCanaryError(
                f"file changed while hashing: {candidate}"
            )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def tensor_identity(value: torch.Tensor) -> Mapping[str, Any]:
    """Own and hash exactly the logical tensor bytes without NumPy/ctypes."""

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise GraftPhaseANativeGPUCanaryError(
            "tensor identity requires a materialized torch.Tensor"
        )
    detached = value.detach()
    if type(detached) is not torch.Tensor:
        raise GraftPhaseANativeGPUCanaryError(
            "tensor identity rejects tensor-subclass detach hooks"
        )
    owned = detached.cpu().contiguous().clone()
    if type(owned) is not torch.Tensor:
        raise GraftPhaseANativeGPUCanaryError(
            "tensor identity did not obtain an exact owned tensor"
        )
    storage = owned.untyped_storage()
    expected = int(owned.numel()) * int(owned.element_size())
    if int(storage.nbytes()) != expected:
        raise GraftPhaseANativeGPUCanaryError(
            "owned tensor storage exceeds its logical bytes"
        )
    raw = bytes(storage)
    if len(raw) != expected:
        raise GraftPhaseANativeGPUCanaryError(
            "owned tensor byte length differs"
        )
    return {
        "shape": [int(item) for item in owned.shape],
        "dtype": str(owned.dtype),
        "device_type_at_observation": value.device.type,
        "finite": bool(torch.isfinite(owned).all().item()),
        "byte_count": expected,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": hashlib.sha256(
            canonical_json_bytes(
                {"shape": [int(item) for item in owned.shape], "dtype": str(owned.dtype)}
            )
            + b"\0"
            + raw
        ).hexdigest(),
    }


def parameter_registry_digest(
    rows: Sequence[tuple[str, torch.nn.Parameter]],
) -> str:
    digest = hashlib.sha256()
    for name, parameter in rows:
        if not isinstance(name, str) or not isinstance(parameter, torch.nn.Parameter):
            raise GraftPhaseANativeGPUCanaryError("parameter registry row differs")
        identity = tensor_identity(parameter)
        payload = canonical_json_bytes({"name": name, "tensor": identity})
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def require_equal_rows(
    rows: Sequence[Any], *, label: str, expected_count: int = WORLD_SIZE
) -> list[Any]:
    owned = list(rows)
    if len(owned) != expected_count or not owned:
        raise GraftPhaseANativeGPUCanaryError(
            f"{label} parity row count differs"
        )
    reference = canonical_json_bytes(owned[0])
    if any(canonical_json_bytes(row) != reference for row in owned[1:]):
        raise GraftPhaseANativeGPUCanaryError(f"{label} differs across ranks")
    return owned


def _all_gather_equal(value: Any, *, label: str) -> list[Any]:
    import torch.distributed as dist

    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, value)
    return require_equal_rows(rows, label=label)


def _require_sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise GraftPhaseANativeGPUCanaryError(
            f"{label} must be a full lowercase {length * 4}-bit digest"
        )
    return value


def _fresh_output_path(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or requested.suffix
        or _SAFE_NAME.fullmatch(requested.name) is None
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "output-dir must be an absolute safe suffix-free non-root path"
        )
    parent = requested.parent.resolve(strict=True)
    before = parent.stat()
    if parent.is_symlink() or not parent.is_dir() or requested != parent / requested.name:
        raise GraftPhaseANativeGPUCanaryError("output parent/path is not canonical")
    if requested.exists() or requested.is_symlink():
        raise GraftPhaseANativeGPUCanaryError("output-dir must be fresh")
    after = parent.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise GraftPhaseANativeGPUCanaryError(
            "output parent changed during validation"
        )
    return requested


def _open_directory_no_follow(path: Path) -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise GraftPhaseANativeGPUCanaryError(
            "safe publication requires O_DIRECTORY and O_NOFOLLOW"
        )
    return os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )


def _assert_directory_identity(
    path: Path, descriptor: int, expected: tuple[int, int]
) -> None:
    descriptor_info = os.fstat(descriptor)
    path_info = path.lstat()
    if (
        not stat.S_ISDIR(descriptor_info.st_mode)
        or not stat.S_ISDIR(path_info.st_mode)
        or path.is_symlink()
        or (int(descriptor_info.st_dev), int(descriptor_info.st_ino)) != expected
        or (int(path_info.st_dev), int(path_info.st_ino)) != expected
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "publication directory descriptor/path identity differs"
        )


def create_output_directory(path: Path) -> tuple[int, tuple[int, int]]:
    parent_fd = _open_directory_no_follow(path.parent)
    output_fd: Optional[int] = None
    try:
        parent_info = os.fstat(parent_fd)
        parent_identity = (int(parent_info.st_dev), int(parent_info.st_ino))
        _assert_directory_identity(path.parent, parent_fd, parent_identity)
        os.mkdir(path.name, mode=0o750, dir_fd=parent_fd)
        output_fd = os.open(
            path.name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        info = os.fstat(output_fd)
        identity = (int(info.st_dev), int(info.st_ino))
        _assert_directory_identity(path, output_fd, identity)
        _assert_directory_identity(path.parent, parent_fd, parent_identity)
        return output_fd, identity
    except Exception:
        if output_fd is not None:
            os.close(output_fd)
        raise
    finally:
        os.close(parent_fd)


def write_receipt_create_only(
    path: Path,
    value: Mapping[str, Any],
    *,
    directory_fd: Optional[int] = None,
    expected_directory_identity: Optional[tuple[int, int]] = None,
) -> None:
    """Publish canonical immutable bytes with retained-directory openat."""

    if path.name in ("", ".", ".."):
        raise GraftPhaseANativeGPUCanaryError("receipt basename differs")
    owns_fd = directory_fd is None
    if directory_fd is None:
        directory_fd = _open_directory_no_follow(path.parent)
    try:
        if expected_directory_identity is None:
            info = os.fstat(directory_fd)
            expected_directory_identity = (int(info.st_dev), int(info.st_ino))
        _assert_directory_identity(
            path.parent, directory_fd, expected_directory_identity
        )
        payload = canonical_json_bytes(dict(value)) + b"\n"
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o444,
            dir_fd=directory_fd,
        )
        try:
            offset = 0
            while offset < len(payload):
                count = os.write(descriptor, payload[offset:])
                if count <= 0:
                    raise GraftPhaseANativeGPUCanaryError(
                        "receipt publication made no progress"
                    )
                offset += count
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            info = os.fstat(descriptor)
            identity = (int(info.st_dev), int(info.st_ino))
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o444
                or int(info.st_size) != len(payload)
            ):
                raise GraftPhaseANativeGPUCanaryError(
                    "published receipt mode/size differs"
                )
        finally:
            os.close(descriptor)
        published = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(published.st_mode)
            or (int(published.st_dev), int(published.st_ino)) != identity
            or stat.S_IMODE(published.st_mode) != 0o444
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "published receipt path identity differs"
            )
        verify_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        try:
            verify_info = os.fstat(verify_fd)
            observed = bytearray()
            while True:
                block = os.read(verify_fd, 1024 * 1024)
                if not block:
                    break
                observed.extend(block)
            if (
                (int(verify_info.st_dev), int(verify_info.st_ino)) != identity
                or bytes(observed) != payload
            ):
                raise GraftPhaseANativeGPUCanaryError(
                    "published receipt reread bytes/identity differ"
                )
        finally:
            os.close(verify_fd)
        _assert_directory_identity(
            path.parent, directory_fd, expected_directory_identity
        )
        os.fsync(directory_fd)
    finally:
        if owns_fd:
            os.close(directory_fd)


def canonical_noop_prompt_contract(
    *, prompt_cleaner: Callable[[str], str]
) -> tuple[str, str, Mapping[str, Any]]:
    """Bind the exact source-only no-op positive and renderer negative text."""

    noop_sha = route_batches.validate_noop_instruction(
        route_batches.EXACT_NOOP_INSTRUCTION
    )
    positive = legacy.build_training_prompt(
        route_batches.EXACT_NOOP_INSTRUCTION, prompt_cleaner=prompt_cleaner
    )
    negative = legacy.DEFAULT_NEGATIVE_PROMPT
    if not positive or not negative or positive == negative:
        raise GraftPhaseANativeGPUCanaryError("canonical no-op prompts differ")
    receipt = seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-noop-r2v-prompts-v1",
            "guidance_mode": phase_core.GUIDANCE_MODE,
            "positive_role": "canonical_semantic_noop_r2v_condition",
            "negative_role": "pinned_renderer_negative_condition",
            "noop_instruction_utf8_sha256": noop_sha,
            "positive_full_prompt_utf8_sha256": hashlib.sha256(
                positive.encode("utf-8")
            ).hexdigest(),
            "negative_prompt_utf8_sha256": hashlib.sha256(
                negative.encode("utf-8")
            ).hexdigest(),
            "training_system_prompt_utf8_sha256": hashlib.sha256(
                legacy.MV2V_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "source_only": True,
            "action_instruction_used": False,
        }
    )
    return positive, negative, receipt


@dataclass(frozen=True)
class KeyedGaussian:
    epsilon: torch.Tensor = field(repr=False, compare=False)
    receipt: Mapping[str, Any]


def keyed_fresh_gaussian(
    *,
    shape: Sequence[int],
    device: torch.device | str,
    source_video_sha256: str,
    cell_id: str,
    base_seed: int,
    schedule_index: int = ACTIVE_SCHEDULE_INDEX,
) -> KeyedGaussian:
    """Materialize one CPU-generator Gaussian keyed to the sealed cell."""

    dimensions = tuple(int(item) for item in shape)
    _require_sha(source_video_sha256, length=64, label="source video SHA-256")
    if (
        not dimensions
        or any(item <= 0 for item in dimensions)
        or not isinstance(cell_id, str)
        or _SAFE_NAME.fullmatch(cell_id) is None
        or type(base_seed) is not int
        or not 0 <= base_seed < 2**63
        or schedule_index != ACTIVE_SCHEDULE_INDEX
    ):
        raise GraftPhaseANativeGPUCanaryError("fresh Gaussian key fields differ")
    key = {
        "schema_version": NOISE_SCHEMA_VERSION,
        "source_video_sha256": source_video_sha256,
        "cell_id": cell_id,
        "base_seed": base_seed,
        "schedule_index": ACTIVE_SCHEDULE_INDEX,
        "shape": list(dimensions),
        "dtype": "torch.float32",
        "generator_device": "cpu",
    }
    key_digest = object_sha256(key)
    derived_seed = int.from_bytes(bytes.fromhex(key_digest[:16]), "big") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)
    epsilon_cpu = torch.randn(
        dimensions, generator=generator, dtype=torch.float32, device="cpu"
    ).contiguous()
    epsilon = epsilon_cpu.to(device=torch.device(device)).contiguous()
    if (
        epsilon.dtype != torch.float32
        or epsilon.requires_grad
        or epsilon.grad_fn is not None
        or not bool(torch.isfinite(epsilon).all().item())
    ):
        raise GraftPhaseANativeGPUCanaryError("fresh Gaussian tensor differs")
    receipt = seal_mapping(
        {
            **key,
            "key_digest": key_digest,
            "derived_seed": derived_seed,
            "algorithm": "torch_cpu_generator_manual_seed_then_randn_fp32",
            "fresh_per_key": True,
            "source_or_target_derived": False,
            "tensor": tensor_identity(epsilon),
        }
    )
    return KeyedGaussian(epsilon=epsilon, receipt=receipt)


def build_noisy_target(
    source_latent: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Build the exact detached FP32 Phase-A noisy source state."""

    if (
        not isinstance(source_latent, torch.Tensor)
        or not isinstance(epsilon, torch.Tensor)
        or source_latent.dtype != torch.float32
        or epsilon.dtype != torch.float32
        or source_latent.shape != epsilon.shape
        or source_latent.device != epsilon.device
        or source_latent.ndim != 5
        or tuple(source_latent.shape[:3]) != (1, 16, LATENT_PHASES)
        or source_latent.requires_grad
        or epsilon.requires_grad
        or not isinstance(sigma, torch.Tensor)
        or sigma.dtype != torch.float32
        or sigma.device.type != "cpu"
        or sigma.ndim != 0
        or sigma.requires_grad
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "noisy target requires detached same-device FP32 zS/eps and CPU FP32 sigma"
        )
    sigma_device = sigma.to(device=source_latent.device, dtype=torch.float32)
    noisy = (
        (torch.ones_like(sigma_device) - sigma_device) * source_latent
        + sigma_device * epsilon
    ).detach().contiguous()
    if (
        noisy.dtype != torch.float32
        or noisy.shape != source_latent.shape
        or noisy.requires_grad
        or noisy.grad_fn is not None
        or not bool(torch.isfinite(noisy).all().item())
    ):
        raise GraftPhaseANativeGPUCanaryError("noisy target construction differs")
    receipt = seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-noisy-source-state-v1",
            "formula": "x=(1-sigma)*zS+sigma*eps",
            "arithmetic_dtype": "torch.float32",
            "sigma_float32_be_hex": struct.pack(
                ">f", float(sigma.item())
            ).hex(),
            "source_latent": tensor_identity(source_latent),
            "epsilon": tensor_identity(epsilon),
            "noisy_target": tensor_identity(noisy),
            "source_epsilon_storage_disjoint": (
                source_latent.untyped_storage().data_ptr()
                != epsilon.untyped_storage().data_ptr()
            ),
            "source_noisy_storage_disjoint": (
                source_latent.untyped_storage().data_ptr()
                != noisy.untyped_storage().data_ptr()
            ),
            "target_video_used": False,
        }
    )
    return noisy, receipt


@dataclass(frozen=True)
class ActiveCoordinate:
    schedule_index: int
    sigma: torch.Tensor = field(repr=False, compare=False)
    timestep: torch.Tensor = field(repr=False, compare=False)
    schedule_receipt: Mapping[str, Any]
    coordinate_receipt: Mapping[str, Any]
    scheduler_state_before: Mapping[str, Any]


def bind_active_index33_coordinate(
    scheduler: Any, *, device: torch.device | str
) -> ActiveCoordinate:
    """Audit the real UniPC schedule and select only live index 33."""

    schedule_receipt = seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-runtime-unipc-audit-v1",
            "audit": sigma_strata.audit_runtime_unipc_schedule(
                scheduler, initialize=True
            ),
        }
    )
    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if (
        not isinstance(timesteps, torch.Tensor)
        or timesteps.dtype != torch.int64
        or timesteps.device.type != "cpu"
        or tuple(timesteps.shape) != (40,)
        or not isinstance(sigmas, torch.Tensor)
        or sigmas.dtype != torch.float32
        or sigmas.device.type != "cpu"
        or tuple(sigmas.shape) != (41,)
    ):
        raise GraftPhaseANativeGPUCanaryError("audited scheduler storage differs")
    sigma = sigmas[ACTIVE_SCHEDULE_INDEX].detach().clone().reshape(())
    timestep = timesteps[ACTIVE_SCHEDULE_INDEX : ACTIVE_SCHEDULE_INDEX + 1].to(
        device=torch.device(device)
    )
    if (
        timestep.dtype != torch.int64
        or tuple(timestep.shape) != (1,)
        or timestep.device != torch.device(device)
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "selected timestep is not device-local INT64 [1]"
        )
    expected_sigma_hex = sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
        ACTIVE_SCHEDULE_INDEX
    ]
    if (
        int(timestep.item())
        != sigma_strata.PINNED_TIMESTEPS[ACTIVE_SCHEDULE_INDEX]
        or struct.pack(">f", float(sigma.item())).hex() != expected_sigma_hex
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "selected live scheduler coordinate differs from pinned index33"
        )
    state = {
        "timesteps": tensor_identity(timesteps),
        "sigmas": tensor_identity(sigmas),
        "step_index": getattr(scheduler, "step_index", None),
    }
    if state["step_index"] is not None:
        raise GraftPhaseANativeGPUCanaryError(
            "fresh audited scheduler already has a live solver step index"
        )
    coordinate = seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-active-coordinate-v1",
            "schedule_index": ACTIVE_SCHEDULE_INDEX,
            "schedule_steps": 40,
            "timestep": int(timestep.item()),
            "timestep_dtype": "torch.int64",
            "timestep_device_type": timestep.device.type,
            "sigma_dtype": "torch.float32",
            "sigma_device_type": "cpu",
            "sigma_float32_be_hex": expected_sigma_hex,
            "schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "scheduler_step_called": False,
            "only_active_index_executed": True,
        }
    )
    return ActiveCoordinate(
        schedule_index=ACTIVE_SCHEDULE_INDEX,
        sigma=sigma,
        timestep=timestep,
        schedule_receipt=schedule_receipt,
        coordinate_receipt=coordinate,
        scheduler_state_before=state,
    )


def assert_scheduler_unchanged(scheduler: Any, coordinate: ActiveCoordinate) -> None:
    current = {
        "timesteps": tensor_identity(getattr(scheduler, "timesteps", None)),
        "sigmas": tensor_identity(getattr(scheduler, "sigmas", None)),
        "step_index": getattr(scheduler, "step_index", None),
    }
    if current != dict(coordinate.scheduler_state_before):
        raise GraftPhaseANativeGPUCanaryError(
            "scheduler state changed despite the no-step canary"
        )


def route_capability_receipt() -> Mapping[str, Any]:
    """Mint the static route receipt consumed by the official authenticator."""

    return seal_mapping(
        {
            "schema_version": phase_core.FORWARD_ROUTE_SCHEMA_VERSION,
            "route_kind": "identity_rebinder_v1",
            "phase_a_active_schedule_indices": list(
                phase_core.PHASE_A_ACTIVE_SCHEDULE_INDICES
            ),
            "canary_executed_schedule_indices": [ACTIVE_SCHEDULE_INDEX],
            "other_active_envelope_indices_executed": False,
            "inactive_schedule_policy": "exact_zero_update_not_trained",
            "target_queries_only": True,
            "condition_rows_written": False,
            "external_oracle_inputs": False,
            "factory": "fresh_identity_atlas_per_forward",
            "replay_atlas_graph_rebuilt_inside_enable_grad": True,
            "branch_name": "V",
        }
    )


class FreshAtlasRouteFactory:
    """Build and activate one fresh source-memory V route for every forward."""

    _EXPECTED_TRACE = (
        ("measurement", "negative", False),
        ("measurement", "positive", False),
        ("replay", "negative", True),
        ("replay", "positive", True),
    )

    def __init__(
        self,
        *,
        handle: Any,
        source_frames: torch.Tensor,
        source_video_sha256: str,
        sequence_parallel_rank: int,
        sequence_parallel_size: int,
        tensor_parity: Optional[Callable[..., Any]] = None,
    ) -> None:
        if (
            not callable(getattr(handle, "build_atlas", None))
            or not callable(getattr(handle, "route", None))
            or not isinstance(source_frames, torch.Tensor)
            or source_frames.dtype != torch.float32
            or source_frames.ndim != 5
            or tuple(source_frames.shape[:3]) != (1, FRAME_COUNT, 3)
            or not source_frames.is_contiguous()
            or source_frames.requires_grad
            or not bool(torch.isfinite(source_frames).all().item())
            or float(source_frames.amin().item()) < -1.0
            or float(source_frames.amax().item()) > 1.0
            or sequence_parallel_size not in (1, SP_SIZE)
            or type(sequence_parallel_rank) is not int
            or not 0 <= sequence_parallel_rank < sequence_parallel_size
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "fresh atlas route factory inputs differ"
            )
        _require_sha(source_video_sha256, length=64, label="source video SHA-256")
        self.handle = handle
        self.source_frames = source_frames
        self.source_video_sha256 = source_video_sha256
        self.sequence_parallel_rank = sequence_parallel_rank
        self.sequence_parallel_size = sequence_parallel_size
        self.tensor_parity = tensor_parity
        self._source_identity = tensor_identity(source_frames)
        self._rows: list[Mapping[str, Any]] = []
        # Retain the small atlas outputs until audit.  Keeping the objects
        # alive makes the freshness proof immune to CPython reusing an id
        # after an earlier no-grad atlas falls out of scope.
        self._token_objects: list[torch.Tensor] = []

    def __call__(
        self, *, request: phase_core.NativeForwardContextRequest
    ) -> AbstractContextManager[Any]:
        if not isinstance(request, phase_core.NativeForwardContextRequest):
            raise GraftPhaseANativeGPUCanaryError(
                "route factory received a non-native request"
            )
        position = len(self._rows)
        if position >= len(self._EXPECTED_TRACE):
            raise GraftPhaseANativeGPUCanaryError(
                "route factory received more than four forwards"
            )
        phase, role, graph_expected = self._EXPECTED_TRACE[position]
        if (
            (request.phase, request.role) != (phase, role)
            or torch.is_grad_enabled() is not graph_expected
            or request.schedule_index != ACTIVE_SCHEDULE_INDEX
            or request.condition_tokens != request.target_tokens
            or request.total_tokens
            != request.condition_tokens + request.target_tokens
            or request.timestep.dtype != torch.int64
            or tuple(request.timestep.shape) != (1,)
            or tensor_identity(self.source_frames) != self._source_identity
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "fresh atlas route request/graph/source differs"
            )
        atlas = self.handle.build_atlas(
            self.source_frames, source_video_sha256=self.source_video_sha256
        )
        if not isinstance(atlas, rebinder.IdentityAtlas):
            raise GraftPhaseANativeGPUCanaryError(
                "route factory did not receive an IdentityAtlas"
            )
        if (
            any(atlas.tokens is prior for prior in self._token_objects)
            or atlas.tokens.requires_grad is not graph_expected
            or (atlas.tokens.grad_fn is not None) is not graph_expected
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "fresh atlas graph/object contract differs"
            )
        self._token_objects.append(atlas.tokens)
        atlas_receipt, _ = own_and_verify_receipt(atlas.receipt())
        token_identity = tensor_identity(atlas.tokens)
        if self.tensor_parity is not None:
            self.tensor_parity(
                token_identity,
                label=f"fresh atlas {phase} {role}",
            )
        route = rebinder.IdentityRebinderRoute(
            total_tokens=request.total_tokens,
            condition_tokens=request.condition_tokens,
            sequence_parallel_rank=self.sequence_parallel_rank,
            sequence_parallel_size=self.sequence_parallel_size,
            branch_name="V",
            sigma=request.sigma,
            atlas=atlas,
            enabled=True,
        )
        if route.gate <= 0.0 or route.target_tokens != request.target_tokens:
            raise GraftPhaseANativeGPUCanaryError(
                "active index33 route gate/target suffix differs"
            )
        local_selector = route.local_target_selector(device=atlas.tokens.device)
        local_target_rows = int(torch.count_nonzero(local_selector).item())
        adapter_graph_bearing = graph_expected and local_target_rows > 0
        observation = phase_core.build_native_forward_context_observation(
            request=request,
            sequence_parallel_rank=self.sequence_parallel_rank,
            sequence_parallel_size=self.sequence_parallel_size,
            local_target_selector=local_selector,
            route_gate=route.gate,
            adapter_graph_bearing=adapter_graph_bearing,
        )
        self._rows.append(
            {
                "ordinal": position,
                "phase": phase,
                "role": role,
                "graph_bearing_atlas": graph_expected,
                "fresh_atlas_token_object": True,
                "atlas_tokens": token_identity,
                "atlas_receipt_digest": atlas_receipt["digest"],
                "branch_name": route.branch_name,
                "gate_hex": float(route.gate).hex(),
                "total_tokens": route.total_tokens,
                "condition_tokens": route.condition_tokens,
                "target_tokens": route.target_tokens,
                "global_total_tokens": observation.global_total_tokens,
                "local_shard_start": observation.local_shard_start,
                "local_shard_stop_exclusive": (
                    observation.local_shard_stop_exclusive
                ),
                "local_shard_rows": observation.local_shard_rows,
                "local_valid_rows": observation.local_valid_rows,
                "local_padding_rows": observation.local_padding_rows,
                "local_target_rows": observation.local_target_rows,
                "local_target_selector_sha256": (
                    observation.local_target_selector_sha256
                ),
                "adapter_graph_bearing": observation.adapter_graph_bearing,
                "sp_rank": self.sequence_parallel_rank,
                "sp_size": self.sequence_parallel_size,
                "target_queries_only": True,
            }
        )

        @contextmanager
        def observed_route() -> Any:
            with self.handle.route(route):
                yield observation

        return observed_route()

    def audit_receipt(self) -> Mapping[str, Any]:
        if len(self._rows) != 4:
            raise GraftPhaseANativeGPUCanaryError(
                "route factory did not service exactly four forwards"
            )
        observed = tuple(
            (row["phase"], row["role"], row["graph_bearing_atlas"])
            for row in self._rows
        )
        if observed != self._EXPECTED_TRACE:
            raise GraftPhaseANativeGPUCanaryError(
                "route factory call trace differs"
            )
        local_target_rows = int(self._rows[0]["local_target_rows"])
        if any(
            row["local_target_rows"] != local_target_rows
            or row["local_shard_rows"] != self._rows[0]["local_shard_rows"]
            or row["local_target_selector_sha256"]
            != self._rows[0]["local_target_selector_sha256"]
            or row["adapter_graph_bearing"]
            is not (
                row["phase"] == "replay" and local_target_rows > 0
            )
            for row in self._rows
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "route factory local ownership/adapter graph observation differs"
            )
        identities = [row["atlas_tokens"] for row in self._rows]
        if any(identity != identities[0] for identity in identities[1:]):
            raise GraftPhaseANativeGPUCanaryError(
                "fresh atlas recomputation changed parameter-identical bytes"
            )
        return seal_mapping(
            {
                "schema_version": ROUTE_FACTORY_SCHEMA_VERSION,
                "call_count": 4,
                "call_trace": [
                    [phase, role] for phase, role, _ in self._EXPECTED_TRACE
                ],
                "measurement_atlases_graph_bearing": False,
                "replay_atlases_graph_bearing": True,
                "replay_adapter_graph_bearing": local_target_rows > 0,
                "fresh_atlas_per_forward": True,
                "source_frames_unchanged": True,
                "source_frames": self._source_identity,
                "all_four_atlas_values_byte_exact": True,
                "external_atlas_owner_required": True,
                "sp_rank": self.sequence_parallel_rank,
                "sp_size": self.sequence_parallel_size,
                "rows": list(self._rows),
            }
        )


@dataclass(frozen=True)
class LocalCanaryResult:
    noisy_target: torch.Tensor = field(repr=False, compare=False)
    guided_clean: torch.Tensor = field(repr=False, compare=False)
    flow_matching_loss: torch.Tensor = field(repr=False, compare=False)
    receipt: Mapping[str, Any]


def local_gradient_receipt(
    bindings: phase_core.AuthenticatedNativeBindings,
    *,
    local_target_rows: int,
) -> Mapping[str, Any]:
    if type(local_target_rows) is not int or local_target_rows < 0:
        raise GraftPhaseANativeGPUCanaryError("local target-row count differs")
    rows: list[Mapping[str, Any]] = []
    total_squared = 0.0
    for name, parameter in bindings.named_trainable_parameters:
        gradient = parameter.grad
        if gradient is None:
            identity = None
            norm = 0.0
            finite = True
        else:
            finite = bool(torch.isfinite(gradient).all().item())
            if not finite:
                raise GraftPhaseANativeGPUCanaryError(
                    f"non-finite local gradient: {name}"
                )
            norm = float(gradient.detach().float().norm().item())
            identity = tensor_identity(gradient)
        total_squared += norm * norm
        rows.append(
            {
                "name": name,
                "present": gradient is not None,
                "finite": finite,
                "l2_float64_hex": norm.hex(),
                "tensor": identity,
                "external_atlas_owner": name.startswith("atlas_encoder."),
            }
        )
    total = math.sqrt(total_squared)
    if not math.isfinite(total) or (total > 0.0) is not (local_target_rows > 0):
        raise GraftPhaseANativeGPUCanaryError(
            "local registry gradient differs from SP4 target ownership"
        )
    return seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-local-gradients-v1",
            "rows": rows,
            "local_target_rows": local_target_rows,
            "total_l2_float64_hex": total.hex(),
            "at_least_one_nonzero": total > 0.0,
            "matches_local_target_ownership": True,
            "optimizer_step_applied": False,
        }
    )


def execute_authenticated_local_cell(
    *,
    bindings: phase_core.AuthenticatedNativeBindings,
    route_factory: FreshAtlasRouteFactory,
    source_latent: torch.Tensor,
    epsilon: torch.Tensor,
    negative_condition: torch.Tensor,
    positive_condition: torch.Tensor,
    sigma: torch.Tensor,
    timestep: torch.Tensor,
    noise_receipt: Mapping[str, Any],
    pre_backward_observer: Optional[Callable[[Mapping[str, Any]], Any]] = None,
) -> LocalCanaryResult:
    """Execute the common authenticated CPU/GPU single-cell outer protocol."""

    if (
        type(bindings) is not phase_core.AuthenticatedNativeBindings
        or bindings.forward_context_factory is not route_factory
        or tuple(name for name, _ in bindings.external_trainable_owner_modules)
        != ("atlas_encoder",)
        or ACTIVE_SCHEDULE_INDEX not in bindings.active_schedule_indices
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "authenticated bindings do not close the external atlas owner/route"
        )
    owned_noise, _ = own_and_verify_receipt(noise_receipt)
    parameter_before = parameter_registry_digest(bindings.named_trainable_parameters)
    noisy_target, noisy_receipt = build_noisy_target(
        source_latent, epsilon, sigma=sigma
    )
    session = phase_core.PhaseANativeTrainingClosure(
        bindings=bindings,
        source_video=source_latent,
        noisy_target=noisy_target,
        negative_condition=negative_condition,
        positive_condition=positive_condition,
        schedule_index=ACTIVE_SCHEDULE_INDEX,
        sigma=sigma,
        timestep=timestep,
    )
    session.measure()
    pre_backward_context, _ = own_and_verify_receipt(
        session.forward_context_observation_receipt()
    )
    if pre_backward_observer is not None:
        if not callable(pre_backward_observer):
            raise GraftPhaseANativeGPUCanaryError(
                "pre-backward observer is not callable"
            )
        pre_backward_observer(pre_backward_context)
    session.derive_phase_a_flow_matching_vjp()
    core_result = session.replay_and_backward()
    core_receipt, core_bytes = own_and_verify_receipt(core_result.receipt)
    route_receipt = route_factory.audit_receipt()
    parameter_after = parameter_registry_digest(bindings.named_trainable_parameters)
    if parameter_after != parameter_before:
        raise GraftPhaseANativeGPUCanaryError(
            "no-step canary changed trainable parameter bytes"
        )
    local_target_rows = int(core_receipt["local_target_rows"])
    if (
        route_receipt["rows"][0]["local_target_rows"] != local_target_rows
        or core_receipt["local_shard_rows"]
        != route_receipt["rows"][0]["local_shard_rows"]
        or core_receipt["local_adapter_graph_bearing"]
        is not (local_target_rows > 0)
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "core/route local target ownership evidence differs"
        )
    gradients = local_gradient_receipt(
        bindings, local_target_rows=local_target_rows
    )
    expected_core = {
        "schedule_index": ACTIVE_SCHEDULE_INDEX,
        "schedule_cell_active_for_training": True,
        "schedule_cell_counted_as_trained": True,
        "scheduler_step_called": False,
        "optimizer_created": False,
        "parameters_updated": False,
        "target_video_used": False,
        "guidance_mode": "v2v_apg",
        "external_trainable_owner_names": ["atlas_encoder"],
        "replay_visual_pack_detached_leaf": True,
        "replay_pack_gradient_cleared_after_each_branch": True,
        "local_trainable_registry_final_gradient_matches_target_ownership": True,
    }
    if any(core_receipt.get(key) != expected for key, expected in expected_core.items()):
        raise GraftPhaseANativeGPUCanaryError(
            "Phase-A core receipt exceeds or differs from runner authority"
        )
    authority = {
        "wiring_canary": True,
        "flow_matching_gradient_canary": True,
        "single_active_exact40_cell_only": True,
        "semantic_success": False,
        "action_success": False,
        "quality_success": False,
        "semantic_action_success": False,
        "visual_quality_success": False,
        "beneficial_training_evidence": False,
        "training_positive": False,
        "training_run": False,
        "optimizer_step": False,
        "parameters_updated": False,
        "scientific_claim_authorized": False,
        "production_claim_authorized": False,
        "full_sampler_parity": False,
    }
    local_receipt = seal_mapping(
        {
            "schema_version": LOCAL_RESULT_SCHEMA_VERSION,
            "complete": True,
            "pass": True,
            "schedule_index": ACTIVE_SCHEDULE_INDEX,
            "source_only": True,
            "sequence_parallel_rank": core_receipt[
                "local_sequence_parallel_rank"
            ],
            "sequence_parallel_size": core_receipt[
                "local_sequence_parallel_size"
            ],
            "local_shard_rows": core_receipt["local_shard_rows"],
            "local_target_rows": local_target_rows,
            "local_adapter_graph_bearing": core_receipt[
                "local_adapter_graph_bearing"
            ],
            "source_retelling_used": False,
            "proposal_selection_used": False,
            "phase_b_only": True,
            "phase_b_deferred_features": {
                "source_retelling_paired_captions": "reserved_not_executed",
                "action_first_proposal_selection": "reserved_not_executed",
            },
            "noisy_state": noisy_receipt,
            "noise": owned_noise,
            "phase_core_receipt": core_receipt,
            "pre_backward_context": pre_backward_context,
            "phase_core_canonical_bytes_sha256": hashlib.sha256(
                core_bytes
            ).hexdigest(),
            "route_factory": route_receipt,
            "gradients_before_rank_sync": gradients,
            "source_latent": tensor_identity(source_latent),
            "epsilon": tensor_identity(epsilon),
            "noisy_target": tensor_identity(noisy_target),
            "guided_clean": tensor_identity(core_result.guided_clean),
            "flow_matching_loss": tensor_identity(
                core_result.flow_matching_loss
            ),
            "trainable_parameter_sha256_before": parameter_before,
            "trainable_parameter_sha256_after": parameter_after,
            "trainable_parameter_bytes_unchanged": True,
            "external_atlas_owner_in_authenticated_closure": True,
            "authority": authority,
        }
    )
    return LocalCanaryResult(
        noisy_target=noisy_target,
        guided_clean=core_result.guided_clean,
        flow_matching_loss=core_result.flow_matching_loss,
        receipt=local_receipt,
    )


def validate_world4_pre_backward_contexts(
    rank_rows: Sequence[Mapping[str, Any]],
    *,
    cell_id: str,
) -> Mapping[str, Any]:
    """Validate and seal SP4 locality before any native replay/backward."""

    if cell_id not in ("dog", "human") or len(rank_rows) != WORLD_SIZE:
        raise GraftPhaseANativeGPUCanaryError(
            "pre-backward WORLD4 locality requires dog/human rank rows"
        )
    contexts: list[Mapping[str, Any]] = []
    for rank, row in enumerate(rank_rows):
        if (
            not isinstance(row, Mapping)
            or row.get("global_rank") != rank
            or not isinstance(row.get("pre_backward_context"), Mapping)
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "pre-backward WORLD4 context order/schema differs"
            )
        contexts.append(row["pre_backward_context"])
    shard_rows = [context.get("local_shard_rows") for context in contexts]
    if (
        any(type(value) is not int or value <= 0 for value in shard_rows)
        or len(set(shard_rows)) != 1
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "pre-backward WORLD4 shard symbol N differs"
        )
    shard_n = int(shard_rows[0])
    expected_ownership = [0, 0, shard_n, shard_n]
    ownership = [context.get("local_target_rows") for context in contexts]
    selector_digests = [
        context.get("local_target_selector_sha256") for context in contexts
    ]
    route_gates = [context.get("route_gate_float64_hex") for context in contexts]
    if ownership != expected_ownership:
        raise GraftPhaseANativeGPUCanaryError(
            "pre-backward dog/human target ownership is not [0,0,N,N]"
        )
    if (
        selector_digests[0] != selector_digests[1]
        or selector_digests[2] != selector_digests[3]
        or selector_digests[0] == selector_digests[2]
        or len(set(route_gates)) != 1
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "pre-backward selector/gate pattern differs"
        )
    try:
        route_gate = float.fromhex(route_gates[0])
    except (TypeError, ValueError) as error:
        raise GraftPhaseANativeGPUCanaryError(
            "pre-backward route gate is not hexadecimal float64"
        ) from error
    if not math.isfinite(route_gate) or route_gate <= 0.0:
        raise GraftPhaseANativeGPUCanaryError(
            "pre-backward route gate is not finite positive"
        )
    for rank, context in enumerate(contexts):
        if (
            context.get("measurement_complete") is not True
            or context.get("backward_started") is not False
            or context.get("measurement_grad_enabled") is not False
            or context.get("adapter_graph_bearing") is not False
            or context.get("sequence_parallel_rank") != rank
            or context.get("sequence_parallel_size") != SP_SIZE
            or context.get("global_total_tokens") != shard_n * WORLD_SIZE
            or context.get("global_condition_tokens") != shard_n * 2
            or context.get("global_target_tokens") != shard_n * 2
            or context.get("local_shard_start") != rank * shard_n
            or context.get("local_shard_stop_exclusive") != (rank + 1) * shard_n
            or context.get("local_valid_rows") != shard_n
            or context.get("local_padding_rows") != 0
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "pre-backward WORLD4 context geometry/state differs"
            )
    return seal_mapping(
        {
            "schema_version": (
                "bernini-graft-phase-a-world4-pre-backward-locality-v1"
            ),
            "cell_id": cell_id,
            "gathered_before_backward": True,
            "measurement_no_grad_all_ranks": True,
            "adapter_graph_absent_during_measurement_all_ranks": True,
            "local_shard_rows_N": shard_n,
            "expected_local_target_rows": expected_ownership,
            "observed_local_target_rows": ownership,
            "selector_sha256_by_rank": selector_digests,
            "route_gate_float64_hex": route_gates[0],
            "rank_contexts": [dict(context) for context in contexts],
        }
    )


def validate_world4_locality_receipts(
    rank_local_rows: Sequence[Mapping[str, Any]],
    *,
    cell_id: str,
    pre_backward_world4: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Close the dog/human SP4 source-prefix/target-suffix locality proof."""

    if cell_id not in ("dog", "human") or len(rank_local_rows) != WORLD_SIZE:
        raise GraftPhaseANativeGPUCanaryError(
            "WORLD4 locality requires one dog/human row per rank"
        )
    if (
        not isinstance(pre_backward_world4, Mapping)
        or pre_backward_world4.get("cell_id") != cell_id
        or pre_backward_world4.get("gathered_before_backward") is not True
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "post-backward locality lacks its pre-backward WORLD4 authority"
        )
    local_receipts: list[Mapping[str, Any]] = []
    for rank, row in enumerate(rank_local_rows):
        if (
            not isinstance(row, Mapping)
            or row.get("global_rank") != rank
            or not isinstance(row.get("local_receipt"), Mapping)
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "WORLD4 rank-local receipt order/schema differs"
            )
        local_receipts.append(row["local_receipt"])
    shard_rows = [int(row["local_shard_rows"]) for row in local_receipts]
    if len(set(shard_rows)) != 1 or shard_rows[0] <= 0:
        raise GraftPhaseANativeGPUCanaryError(
            "WORLD4 local shard sizes are not one positive N"
        )
    shard_n = shard_rows[0]
    expected_ownership = [0, 0, shard_n, shard_n]
    ownership = [int(row["local_target_rows"]) for row in local_receipts]
    if ownership != expected_ownership:
        raise GraftPhaseANativeGPUCanaryError(
            "dog/human SP4 target ownership is not [0,0,N,N]"
        )
    if (
        pre_backward_world4.get("local_shard_rows_N") != shard_n
        or pre_backward_world4.get("observed_local_target_rows") != ownership
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "pre/post-backward WORLD4 ownership differs"
        )
    selector_digests: list[str] = []
    route_gate_hexes: list[str] = []
    rank_evidence: list[Mapping[str, Any]] = []
    for rank, receipt in enumerate(local_receipts):
        core = receipt.get("phase_core_receipt")
        route = receipt.get("route_factory")
        gradients = receipt.get("gradients_before_rank_sync")
        if not all(isinstance(value, Mapping) for value in (core, route, gradients)):
            raise GraftPhaseANativeGPUCanaryError(
                "WORLD4 locality nested receipt schema differs"
            )
        observations = core.get("forward_context_observations")
        pack_gradients = core.get("per_branch_replay_pack_leaf_gradient")
        local_gates = core.get("per_branch_local_trainable_gradient_gate")
        if (
            not isinstance(observations, list)
            or len(observations) != 4
            or not isinstance(pack_gradients, list)
            or len(pack_gradients) != 2
            or not isinstance(local_gates, list)
            or len(local_gates) != 2
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "WORLD4 replay observation inventory differs"
            )
        expected_graph = ownership[rank] > 0
        if any(
            observation.get("sequence_parallel_rank") != rank
            or observation.get("sequence_parallel_size") != SP_SIZE
            or observation.get("global_total_tokens") != shard_n * WORLD_SIZE
            or observation.get("global_condition_tokens") != shard_n * 2
            or observation.get("global_target_tokens") != shard_n * 2
            or observation.get("local_shard_rows") != shard_n
            or observation.get("local_target_rows") != ownership[rank]
            or observation.get("local_shard_start") != rank * shard_n
            or observation.get("local_shard_stop_exclusive")
            != (rank + 1) * shard_n
            or observation.get("local_valid_rows") != shard_n
            or observation.get("local_padding_rows") != 0
            for observation in observations
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "WORLD4 forward-context geometry differs"
            )
        if [row.get("adapter_graph_bearing") for row in observations] != [
            False,
            False,
            expected_graph,
            expected_graph,
        ]:
            raise GraftPhaseANativeGPUCanaryError(
                "WORLD4 adapter graph-bearing ownership differs"
            )
        selector_digest = observations[0].get("local_target_selector_sha256")
        route_gate_hex = observations[0].get("route_gate_float64_hex")
        try:
            route_gate = float.fromhex(route_gate_hex)
        except (TypeError, ValueError) as error:
            raise GraftPhaseANativeGPUCanaryError(
                "WORLD4 route gate is not hexadecimal float64"
            ) from error
        if (
            not isinstance(selector_digest, str)
            or not math.isfinite(route_gate)
            or route_gate <= 0.0
            or any(
                row.get("local_target_selector_sha256") != selector_digest
                or row.get("route_gate_float64_hex") != route_gate_hex
                for row in observations
            )
            or any(
                row.get("gradient_finite_nonzero") is not True
                or row.get("cleared_after_branch") is not True
                for row in pack_gradients
            )
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "WORLD4 selector or replay pack-leaf evidence differs"
            )
        expected_gate = (
            "target_rows_output_projection_only_nonzero"
            if expected_graph
            else "zero_target_rows_adapter_absent_or_zero"
        )
        if (
            any(row.get("gate") != expected_gate for row in local_gates)
            or gradients.get("local_target_rows") != ownership[rank]
            or gradients.get("at_least_one_nonzero") is not expected_graph
            or gradients.get("matches_local_target_ownership") is not True
            or route.get("sp_rank") != rank
            or route.get("sp_size") != SP_SIZE
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "WORLD4 local adapter-gradient gate differs"
            )
        selector_digests.append(selector_digest)
        route_gate_hexes.append(route_gate_hex)
        rank_evidence.append(
            {
                "global_rank": rank,
                "local_shard_rows": shard_n,
                "local_target_rows": ownership[rank],
                "selector_sha256": selector_digest,
                "route_gate_float64_hex": route_gate_hex,
                "adapter_graph_bearing_on_replay": expected_graph,
                "pack_leaf_gradient_finite_nonzero_both_branches": True,
                "local_gradient_gate": expected_gate,
            }
        )
    if len(set(selector_digests)) != 2:
        raise GraftPhaseANativeGPUCanaryError(
            "WORLD4 selector digests do not form source/source/target/target ownership"
        )
    if len(set(route_gate_hexes)) != 1:
        raise GraftPhaseANativeGPUCanaryError(
            "WORLD4 route gate differs across ranks"
        )
    return seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-world4-locality-v1",
            "cell_id": cell_id,
            "pre_backward_world4_digest": pre_backward_world4.get("digest"),
            "global_pack_layout": "source_prefix_2N_then_target_suffix_2N",
            "local_shard_symbol": "N",
            "local_shard_rows_N": shard_n,
            "expected_local_target_rows": expected_ownership,
            "observed_local_target_rows": ownership,
            "selector_digest_pattern": ["source", "source", "target", "target"],
            "route_gate_float64_hex": route_gate_hexes[0],
            "measurement_no_grad_all_ranks": True,
            "replay_native_pack_leaf_all_ranks": True,
            "zero_target_ranks_adapter_absent_or_zero": True,
            "target_ranks_output_projection_local_grad_nonzero": True,
            "target_ranks_qkv_and_atlas_local_grad_exact_zero": True,
            "rank_evidence": rank_evidence,
        }
    )


def synchronize_gradients(
    bindings: phase_core.AuthenticatedNativeBindings,
) -> Mapping[str, Any]:
    """Average the replicated registry gradients and prove all-rank parity."""

    import torch.distributed as dist

    local_presence = {
        name: parameter.grad is not None
        for name, parameter in bindings.named_trainable_parameters
    }
    presence_by_rank: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(presence_by_rank, local_presence)
    if any(
        not isinstance(row, Mapping) or tuple(row) != tuple(local_presence)
        for row in presence_by_rank
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "pre-sync gradient-presence registry differs across ranks"
        )
    rows: list[Mapping[str, Any]] = []
    total_squared = 0.0
    for name, parameter in bindings.named_trainable_parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise GraftPhaseANativeGPUCanaryError(
                f"non-finite pre-sync gradient: {name}"
            )
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(WORLD_SIZE))
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise GraftPhaseANativeGPUCanaryError(
                f"non-finite synchronized gradient: {name}"
            )
        norm = float(parameter.grad.detach().float().norm().item())
        total_squared += norm * norm
        rows.append(
            {
                "name": name,
                "local_gradient_present_by_global_rank": [
                    bool(row[name]) for row in presence_by_rank
                ],
                "averaged_across_world4": True,
                "l2_float64_hex": norm.hex(),
                "tensor": tensor_identity(parameter.grad),
                "external_atlas_owner": name.startswith("atlas_encoder."),
            }
        )
    total = math.sqrt(total_squared)
    if not math.isfinite(total) or total <= 0.0:
        raise GraftPhaseANativeGPUCanaryError(
            "synchronized active-cell gradient is zero/non-finite"
        )
    receipt = seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-world4-gradients-v1",
            "reduction": "SUM_then_divide_by_WORLD4",
            "world_size": WORLD_SIZE,
            "rows": rows,
            "total_l2_float64_hex": total.hex(),
            "all_rank_exact_after_sync": True,
            "optimizer_step_applied": False,
        }
    )
    _all_gather_equal(receipt, label="synchronized trainable gradients")
    return receipt


def prepare_atlas_source_frames(
    source_tensor: torch.Tensor, *, device: torch.device | str
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Create an independent bounded RGB atlas view without changing VAE input."""

    if (
        not isinstance(source_tensor, torch.Tensor)
        or source_tensor.dtype != torch.float32
        or source_tensor.ndim != 5
        or tuple(source_tensor.shape[:3]) != (1, 3, FRAME_COUNT)
        or not source_tensor.is_contiguous()
        or not bool(torch.isfinite(source_tensor).all().item())
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "atlas source requires contiguous finite FP32 [1,3,81,H,W]"
        )
    before = tensor_identity(source_tensor)
    raw = source_tensor.permute(0, 2, 1, 3, 4).to(
        device=torch.device(device), dtype=torch.float32
    ).contiguous()
    below = int(torch.count_nonzero(raw < -1.0).item())
    above = int(torch.count_nonzero(raw > 1.0).item())
    frames = raw.clamp(-1.0, 1.0).contiguous()
    after = tensor_identity(source_tensor)
    if (
        before != after
        or tuple(frames.shape[:3]) != (1, FRAME_COUNT, 3)
        or frames.requires_grad
        or float(frames.amin().item()) < -1.0
        or float(frames.amax().item()) > 1.0
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "atlas-only RGB view mutated source or escaped its range"
        )
    return frames, seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-atlas-rgb-view-v1",
            "input_layout": "B_C_T_H_W",
            "output_layout": "B_T_C_H_W",
            "source_tensor_unchanged": True,
            "vae_source_tensor_clamped_or_replaced": False,
            "atlas_view_clamped_to_closed_minus1_plus1": True,
            "below_minus_one_count": below,
            "above_plus_one_count": above,
            "clipped_element_count": below + above,
            "source_tensor": before,
            "atlas_frames": tensor_identity(frames),
        }
    )


def _live_sp4() -> tuple[int, Mapping[str, Any]]:
    import torch.distributed as dist
    from bernini.parallel import get_parallel_state

    state = get_parallel_state()
    group = getattr(state, "ulysses_group", None)
    rank = getattr(state, "ulysses_rank", None)
    if (
        getattr(state, "ulysses_enabled", None) is not True
        or getattr(state, "ulysses_size", None) != SP_SIZE
        or type(rank) is not int
        or not 0 <= rank < SP_SIZE
        or dist.get_world_size(group) != SP_SIZE
        or dist.get_rank(group) != rank
        or str(dist.get_backend(group)).lower() != "nccl"
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "live Bernini parallel state is not WORLD4/SP4 NCCL"
        )
    members: list[Any] = [None] * SP_SIZE
    dist.all_gather_object(members, int(dist.get_rank()), group=group)
    if members != list(range(SP_SIZE)):
        raise GraftPhaseANativeGPUCanaryError("SP4 rank order differs")
    receipt = seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-live-sp4-v1",
            "world_size": WORLD_SIZE,
            "ulysses_size": SP_SIZE,
            "ulysses_rank": rank,
            "backend": "nccl",
            "ordered_global_ranks": members,
        }
    )
    return rank, receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=EXPECTED_CHECKPOINT_MANIFEST_SHA256,
    )
    parser.add_argument("--cell-spec", required=True)
    parser.add_argument("--expected-cell-spec-sha256", required=True)
    parser.add_argument("--cell-id", required=True, choices=("dog", "human"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-phase-a-closure-sha256", required=True)
    parser.add_argument("--expected-rebinder-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=rebinder.PINNED_BERNINI_SOURCE_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument(
        "--ack-wiring-fm-gradient-only-no-training-claim", action="store_true"
    )
    return parser


def validate_cli(args: argparse.Namespace) -> Path:
    if args.ack_wiring_fm_gradient_only_no_training_claim is not True:
        raise GraftPhaseANativeGPUCanaryError(
            "--ack-wiring-fm-gradient-only-no-training-claim is mandatory"
        )
    for name in ("expected_bernini_commit", "expected_veomni_commit"):
        _require_sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_checkpoint_content_manifest_sha256",
        "expected_cell_spec_sha256",
        "expected_runner_sha256",
        "expected_phase_a_closure_sha256",
        "expected_rebinder_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        _require_sha(getattr(args, name), length=64, label=name)
    if args.expected_bernini_commit != rebinder.PINNED_BERNINI_SOURCE_COMMIT:
        raise GraftPhaseANativeGPUCanaryError("Bernini source commit differs")
    if args.expected_veomni_commit != legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise GraftPhaseANativeGPUCanaryError("VeOmni source commit differs")
    if args.expected_checkpoint_tree_sha256 != legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise GraftPhaseANativeGPUCanaryError("checkpoint tree differs")
    if (
        args.expected_checkpoint_content_manifest_sha256
        != EXPECTED_CHECKPOINT_MANIFEST_SHA256
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "checkpoint manifest authority differs"
        )
    sources = {
        Path(__file__).resolve(): args.expected_runner_sha256,
        Path(phase_core.__file__).resolve(): args.expected_phase_a_closure_sha256,
        Path(rebinder.__file__).resolve(): args.expected_rebinder_sha256,
    }
    for path, expected in sources.items():
        if file_sha256(path) != expected:
            raise GraftPhaseANativeGPUCanaryError(
                f"authenticated runner/core source bytes differ: {path.name}"
            )
    return _fresh_output_path(args.output_dir)


def _base_parameter_rows(
    transformer: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    rows = tuple(
        (f"transformer.{name}", parameter)
        for name, parameter in transformer.named_parameters()
    )
    if not rows or any(
        parameter.requires_grad or parameter.grad is not None
        for _, parameter in rows
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "complete pre-install transformer base is not frozen/grad-free"
        )
    return rows


def _broadcast_trainable_parameters(
    rows: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    import torch.distributed as dist

    for _, parameter in rows:
        dist.broadcast(parameter.data, src=0)
    output_rows = [
        (name, parameter)
        for name, parameter in rows
        if name.endswith(".identity_rebinder.output.weight")
    ]
    if not output_rows or any(
        int(torch.count_nonzero(parameter.detach()).item()) != 0
        for _, parameter in output_rows
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "broadcast rebinder output projections are not exact zero"
        )
    digest = parameter_registry_digest(rows)
    _all_gather_equal(digest, label="initialized trainable registry bytes")
    return seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-initial-registry-v1",
            "rank0_broadcast_before_forward": True,
            "parameter_count": len(rows),
            "parameter_sha256": digest,
            "zero_initialized_output_projection_count": len(output_rows),
            "zero_initialized_output_projections_exact_zero": True,
            "optimizer_created": False,
        }
    )


def _gradient_owner_summary(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    atlas_squared = 0.0
    transformer_squared = 0.0
    for row in receipt["rows"]:
        norm = float.fromhex(row["l2_float64_hex"])
        if row["external_atlas_owner"]:
            atlas_squared += norm * norm
        else:
            transformer_squared += norm * norm
    return {
        "atlas_encoder_l2_float64_hex": math.sqrt(atlas_squared).hex(),
        "transformer_adapter_l2_float64_hex": math.sqrt(transformer_squared).hex(),
        "transformer_adapter_nonzero": transformer_squared > 0.0,
        "atlas_nonzero_not_required_at_zero_output_initialization": True,
    }


def zero_initialized_gradient_gate(
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Verify the only legal no-step gradient category at zero initialization."""

    category_squared = {
        "atlas_encoder": 0.0,
        "query_key_value": 0.0,
        "output_projection": 0.0,
    }
    category_counts = {name: 0 for name in category_squared}
    for row in receipt.get("rows", ()):
        name = row.get("name")
        norm_hex = row.get("l2_float64_hex")
        if not isinstance(name, str) or not isinstance(norm_hex, str):
            raise GraftPhaseANativeGPUCanaryError(
                "synchronized gradient row schema differs"
            )
        try:
            norm = float.fromhex(norm_hex)
        except ValueError as error:
            raise GraftPhaseANativeGPUCanaryError(
                "synchronized gradient norm is not hexadecimal float64"
            ) from error
        if not math.isfinite(norm) or norm < 0.0:
            raise GraftPhaseANativeGPUCanaryError(
                "synchronized gradient norm is non-finite/negative"
            )
        if name.startswith("atlas_encoder."):
            category = "atlas_encoder"
        elif name.endswith(".identity_rebinder.output.weight"):
            category = "output_projection"
        elif any(
            name.endswith(f".identity_rebinder.{projection}.weight")
            for projection in ("query", "key", "value")
        ):
            category = "query_key_value"
        else:
            raise GraftPhaseANativeGPUCanaryError(
                f"unclassified synchronized trainable: {name}"
            )
        category_counts[category] += 1
        category_squared[category] += norm * norm
    category_l2 = {
        name: math.sqrt(value) for name, value in category_squared.items()
    }
    if (
        any(count <= 0 for count in category_counts.values())
        or category_l2["output_projection"] <= 0.0
        or category_l2["atlas_encoder"] != 0.0
        or category_l2["query_key_value"] != 0.0
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "zero-initialized no-step gradient gate differs from output-only"
        )
    return seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-zero-init-gradient-gate-v1",
            "initialization": "identity_rebinder_output_projection_exact_zero",
            "optimizer_step_before_observation": False,
            "category_parameter_counts": category_counts,
            "category_l2_float64_hex": {
                name: value.hex() for name, value in category_l2.items()
            },
            "output_projection_nonzero": True,
            "query_key_value_exact_zero": True,
            "external_atlas_encoder_exact_zero": True,
            "gate": "output_projection_only_nonzero",
        }
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = validate_cli(args)
    root_spec, cell, spec_path, spec_sha = cell_registry.load_cell_spec(
        args.cell_spec,
        expected_file_sha256=args.expected_cell_spec_sha256,
        cell_id=args.cell_id,
    )
    del root_spec
    manifest_path = Path(args.checkpoint_content_manifest).expanduser().resolve(
        strict=True
    )
    if file_sha256(manifest_path) != args.expected_checkpoint_content_manifest_sha256:
        raise GraftPhaseANativeGPUCanaryError(
            "checkpoint content manifest bytes differ"
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
    except Exception as error:
        raise GraftPhaseANativeGPUCanaryError(str(error)) from error
    if int(transformer_config.get("num_attention_heads", -1)) != 12:
        raise GraftPhaseANativeGPUCanaryError("checkpoint head count differs")
    inference_hashes = legacy.validate_inference_source_files(bernini_root)
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    import bernini.models.transformer_wan as transformer_wan
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT:
        raise GraftPhaseANativeGPUCanaryError("MV2V system prompt differs")
    if DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT:
        raise GraftPhaseANativeGPUCanaryError("renderer negative prompt differs")
    positive_prompt, negative_prompt, prompt_contract = canonical_noop_prompt_contract(
        prompt_cleaner=prompt_clean
    )
    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise GraftPhaseANativeGPUCanaryError(
            "runner requires AUH ROCm WORLD4/SP4"
        )
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    sp_rank, sp_receipt = _live_sp4()
    sp_rank_rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        sp_rank_rows,
        {"global_rank": distributed.rank, "sp_receipt": sp_receipt},
    )
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("sp_receipt"), Mapping)
        for row in sp_rank_rows
    ) or (
        [row["global_rank"] for row in sp_rank_rows] != list(range(WORLD_SIZE))
        or [row["sp_receipt"]["ulysses_rank"] for row in sp_rank_rows]
        != list(range(SP_SIZE))
    ):
        raise GraftPhaseANativeGPUCanaryError("WORLD4/SP4 receipt coverage differs")
    sp_receipt_set = seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-world4-sp4-receipt-set-v1",
            "ordered_by_global_rank": True,
            "rows": sp_rank_rows,
            "assembled_receipt_exact_across_all_ranks": True,
        }
    )
    _all_gather_equal(sp_receipt_set, label="assembled WORLD4/SP4 receipt set")
    device = torch.device("cuda", distributed.local_rank)
    handle: Optional[rebinder.IdentityRebinderHandle] = None
    try:
        runner_source_binding = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-runner-source-binding-v1",
                "runner_sha256": args.expected_runner_sha256,
                "phase_a_closure_sha256": args.expected_phase_a_closure_sha256,
                "identity_rebinder_sha256": args.expected_rebinder_sha256,
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "bernini_inference_files": inference_hashes,
            }
        )
        _all_gather_equal(
            runner_source_binding, label="runner/vendor source binding"
        )
        checkpoint_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                checkpoint_rows[0] = {
                    "ok": True,
                    "identity": source_audit.validate_checkpoint_content(
                        checkpoint,
                        manifest_path,
                        expected_manifest_sha256=(
                            args.expected_checkpoint_content_manifest_sha256
                        ),
                    ),
                }
            except Exception as error:
                checkpoint_rows[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(checkpoint_rows, src=0)
        checkpoint_result = checkpoint_rows[0]
        if (
            not isinstance(checkpoint_result, Mapping)
            or checkpoint_result.get("ok") is not True
        ):
            raise GraftPhaseANativeGPUCanaryError(
                f"checkpoint content validation failed: {checkpoint_result}"
            )
        checkpoint_identity = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-checkpoint-content-v1",
                "identity": dict(checkpoint_result["identity"]),
            }
        )
        _all_gather_equal(checkpoint_identity, label="checkpoint content receipt")

        source_path = cell_registry._plain_file(
            cell["source_video"], label="source video"
        )
        source_tensor, source_metadata, source_sha = (
            source_audit.prepare_hashed_source_snapshot(source_path)
        )
        if (
            source_sha != cell["source_video_sha256"]
            or tuple(source_tensor.shape[:3]) != (1, 3, FRAME_COUNT)
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "exact81 source identity/geometry differs"
            )
        _all_gather_equal(
            tensor_identity(source_tensor), label="decoded exact81 source RGB"
        )
        bucket_hw = tuple(int(item) for item in source_metadata["source_derived_bucket_hw"])

        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
        )
        positive_ids, positive_mask = legacy._tokenize_training_prompt(
            tokenizer, positive_prompt
        )
        negative_ids, negative_mask = legacy._tokenize_renderer_negative(
            tokenizer, negative_prompt
        )
        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
            raise GraftPhaseANativeGPUCanaryError(
                "renderer is not pinned UniPC flow-shift5"
            )
        renderer = BerniniRendererModel(config)
        renderer.eval().requires_grad_(False)

        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval().requires_grad_(False).to(device)
        pixels = source_tensor.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            source_latent = _vae_encode(vae, pixels).float().contiguous()
        expected_shape = (
            1,
            16,
            LATENT_PHASES,
            bucket_hw[0] // 8,
            bucket_hw[1] // 8,
        )
        if (
            tuple(source_latent.shape) != expected_shape
            or source_latent.requires_grad
            or not bool(torch.isfinite(source_latent).all().item())
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "frozen source VAE latent zS differs"
            )
        # The VAE is evaluated independently on every rank as a runtime
        # check, but rank0 owns the one canonical zS consumed by Phase A.
        # This avoids silently treating implementation-level cross-device
        # drift as four different training examples.
        dist.broadcast(source_latent, src=0)
        source_latent_identity = tensor_identity(source_latent)
        _all_gather_equal(source_latent_identity, label="source VAE latent zS")
        source_latent_receipt = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-source-vae-zs-v1",
                "source_frame_count": FRAME_COUNT,
                "latent_phase_count": LATENT_PHASES,
                "full_exact81_encode_calls_per_rank": 1,
                "frozen_vae_eval": True,
                "vae_encode_grad_enabled": False,
                "rank0_tensor_broadcast_after_runtime_encode": True,
                "all_rank_byte_exact_after_broadcast": True,
                "tensor": source_latent_identity,
            }
        )
        _all_gather_equal(source_latent_receipt, label="source VAE zS receipt")
        vae.to("cpu")
        del vae, pixels
        torch.cuda.empty_cache()

        renderer.to(device)
        diffusion = source_audit.resolve_diffusion_core(renderer)
        transformer = diffusion.transformer
        if transformer is None or getattr(diffusion, "transformer_2", None) is not None:
            raise GraftPhaseANativeGPUCanaryError(
                "runner requires the pinned single transformer_1"
            )
        renderer.eval().requires_grad_(False)
        wan_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
        )
        coordinate = bind_active_index33_coordinate(
            diffusion.scheduler, device=device
        )
        _all_gather_equal(
            coordinate.coordinate_receipt, label="active index33 coordinate"
        )

        with torch.no_grad():
            positive_condition = renderer.encode_prompt(
                positive_ids.to(device), positive_mask.to(device)
            ).detach().contiguous()
            negative_condition = renderer.encode_prompt(
                negative_ids.to(device), negative_mask.to(device)
            ).detach().contiguous()
        if (
            tuple(positive_condition.shape) != (1, 512, 4096)
            or positive_condition.shape != negative_condition.shape
            or positive_condition.dtype != torch.bfloat16
            or negative_condition.dtype != torch.bfloat16
            or torch.equal(positive_condition, negative_condition)
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "canonical no-op positive/negative embedding contract differs"
            )
        prompt_tensor_receipt = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-noop-r2v-embeddings-v1",
                "prompt_contract_digest": prompt_contract["digest"],
                "positive": tensor_identity(positive_condition),
                "negative": tensor_identity(negative_condition),
                "distinct": True,
                "all_rank_exact": True,
            }
        )
        _all_gather_equal(prompt_tensor_receipt, label="canonical no-op embeddings")
        renderer.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()

        noise = keyed_fresh_gaussian(
            shape=expected_shape,
            device=device,
            source_video_sha256=source_sha,
            cell_id=cell["cell_id"],
            base_seed=int(cell["seeds"][0]),
        )
        _all_gather_equal(noise.receipt, label="keyed fresh Gaussian epsilon")
        source_frames, atlas_rgb_receipt = prepare_atlas_source_frames(
            source_tensor, device=device
        )
        _all_gather_equal(atlas_rgb_receipt, label="atlas RGB source view")

        base_rows = _base_parameter_rows(transformer)
        base_before = parameter_registry_digest(base_rows)
        _all_gather_equal(base_before, label="pre-install frozen base bytes")
        handle = rebinder.install_identity_rebinder_v1(
            transformer,
            runtime_source_commit=bernini_revision,
            model_revision=rebinder.PINNED_BERNINI_MODEL_REVISION,
            checkpoint_manifest_sha256=(
                args.expected_checkpoint_content_manifest_sha256
            ),
        )
        transformer.eval()
        handle.atlas_encoder.eval()
        install_receipt, _ = own_and_verify_receipt(handle.receipt())
        _all_gather_equal(install_receipt, label="identity rebinder install receipt")
        trainable_rows = handle.trainable_named_parameters()
        initialization_receipt = _broadcast_trainable_parameters(trainable_rows)
        route_factory = FreshAtlasRouteFactory(
            handle=handle,
            source_frames=source_frames,
            source_video_sha256=source_sha,
            sequence_parallel_rank=sp_rank,
            sequence_parallel_size=SP_SIZE,
            tensor_parity=_all_gather_equal,
        )
        route_capability = route_capability_receipt()
        _all_gather_equal(route_capability, label="Phase-A route capability")
        bindings = phase_core.authenticate_pinned_native_bindings(
            diffusion=diffusion,
            transformer=transformer,
            named_trainable_parameters=trainable_rows,
            external_trainable_owner_modules={
                "atlas_encoder": handle.atlas_encoder
            },
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
            transformer_wan_path=Path(transformer_wan.__file__).resolve(
                strict=True
            ),
            bernini_commit=bernini_revision,
            forward_context_factory=route_factory,
            forward_route_receipt=route_capability,
        )
        binding_receipt, _ = own_and_verify_receipt(bindings.receipt())
        _all_gather_equal(binding_receipt, label="authenticated native bindings")

        pre_backward_holder: dict[str, Mapping[str, Any]] = {}

        def gather_pre_backward_context(local_context: Mapping[str, Any]) -> None:
            rows: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(
                rows,
                {
                    "global_rank": distributed.rank,
                    "pre_backward_context": local_context,
                },
            )
            world4 = validate_world4_pre_backward_contexts(
                rows,
                cell_id=cell["cell_id"],
            )
            _all_gather_equal(
                world4, label="pre-backward dog/human WORLD4 locality"
            )
            pre_backward_holder["world4"] = world4

        local = execute_authenticated_local_cell(
            bindings=bindings,
            route_factory=route_factory,
            source_latent=source_latent,
            epsilon=noise.epsilon,
            negative_condition=negative_condition,
            positive_condition=positive_condition,
            sigma=coordinate.sigma,
            timestep=coordinate.timestep,
            noise_receipt=noise.receipt,
            pre_backward_observer=gather_pre_backward_context,
        )
        pre_backward_world4 = pre_backward_holder.get("world4")
        if not isinstance(pre_backward_world4, Mapping):
            raise GraftPhaseANativeGPUCanaryError(
                "pre-backward WORLD4 locality receipt was not retained"
            )
        local_receipt, local_bytes = own_and_verify_receipt(local.receipt)
        rank_local_rows: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(
            rank_local_rows,
            {
                "global_rank": distributed.rank,
                "local_receipt": local_receipt,
                "canonical_bytes_sha256": hashlib.sha256(local_bytes).hexdigest(),
            },
        )
        if any(not isinstance(row, Mapping) for row in rank_local_rows) or [
            row["global_rank"] for row in rank_local_rows
        ] != list(range(WORLD_SIZE)):
            raise GraftPhaseANativeGPUCanaryError(
                "rank-local Phase-A receipt coverage/order differs"
            )
        world4_locality = validate_world4_locality_receipts(
            rank_local_rows,
            cell_id=cell["cell_id"],
            pre_backward_world4=pre_backward_world4,
        )
        _all_gather_equal(
            world4_locality, label="dog/human WORLD4 target locality"
        )
        local_receipt_set = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-rank-local-receipt-set-v1",
                "ordered_by_global_rank": True,
                "rank_count": WORLD_SIZE,
                "rows": rank_local_rows,
                "pre_backward_world4": pre_backward_world4,
                "world4_locality": world4_locality,
                "assembled_receipt_exact_across_all_ranks": True,
                "rank_local_gradient_metrics_may_differ_before_sync": True,
            }
        )
        _all_gather_equal(
            local_receipt_set, label="assembled Phase-A rank-local receipt set"
        )
        synchronized_gradients = synchronize_gradients(bindings)
        gradient_owner_summary = _gradient_owner_summary(synchronized_gradients)
        gradient_gate = zero_initialized_gradient_gate(synchronized_gradients)
        assert_scheduler_unchanged(diffusion.scheduler, coordinate)

        if any(parameter.grad is not None for _, parameter in base_rows):
            raise GraftPhaseANativeGPUCanaryError(
                "Phase-A backward populated frozen base gradients"
            )
        base_after = parameter_registry_digest(base_rows)
        if base_after != base_before:
            raise GraftPhaseANativeGPUCanaryError(
                "Phase-A no-step canary changed frozen base bytes"
            )
        _all_gather_equal(base_after, label="post-backward frozen base bytes")
        trainable_after = parameter_registry_digest(trainable_rows)
        if trainable_after != initialization_receipt["parameter_sha256"]:
            raise GraftPhaseANativeGPUCanaryError(
                "Phase-A no-step canary changed trainable parameter bytes"
            )
        _all_gather_equal(
            trainable_after, label="post-backward trainable parameter bytes"
        )

        parity_payload = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-world4-parity-v1",
                "rank_local_receipt_set": local_receipt_set,
                "pre_backward_world4": pre_backward_world4,
                "world4_locality": world4_locality,
                "synchronized_gradients": synchronized_gradients,
                "source_latent": source_latent_identity,
                "positive_condition": tensor_identity(positive_condition),
                "negative_condition": tensor_identity(negative_condition),
                "epsilon": tensor_identity(noise.epsilon),
                "noisy_target": tensor_identity(local.noisy_target),
                "guided_clean": tensor_identity(local.guided_clean),
                "flow_matching_loss": tensor_identity(local.flow_matching_loss),
                "base_sha256_before": base_before,
                "base_sha256_after": base_after,
                "trainable_sha256_before": initialization_receipt[
                    "parameter_sha256"
                ],
                "trainable_sha256_after": trainable_after,
                "all_rank_tensor_exact": True,
                "all_rank_gradient_exact_after_sync": True,
                "all_rank_assembled_receipt_exact": True,
                "rank_local_receipt_equality_before_gradient_sync_claimed": False,
            }
        )
        parity_rows = _all_gather_equal(
            parity_payload, label="complete Phase-A rank parity payload"
        )
        rank_metadata = {
            "global_rank": distributed.rank,
            "local_rank": distributed.local_rank,
            "sp_rank": sp_rank,
            "parity_payload_digest": parity_payload["digest"],
        }
        gathered_rank_metadata: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_rank_metadata, rank_metadata)
        if sorted(row["global_rank"] for row in gathered_rank_metadata) != list(
            range(WORLD_SIZE)
        ):
            raise GraftPhaseANativeGPUCanaryError(
                "WORLD4 rank metadata coverage differs"
            )
        dist.barrier()

        if distributed.rank == 0:
            directory_fd, directory_identity = create_output_directory(output_dir)
            try:
                unsigned = {
                    "schema_version": SCHEMA_VERSION,
                    "method": METHOD,
                    "complete": True,
                    "pass": True,
                    "authority": {
                        "wiring_canary": True,
                        "flow_matching_gradient_canary": True,
                        "single_active_exact40_index33": True,
                        "semantic_success": False,
                        "action_success": False,
                        "quality_success": False,
                        "semantic_action_success": False,
                        "visual_quality_success": False,
                        "beneficial_training_evidence": False,
                        "training_positive": False,
                        "training_run": False,
                        "optimizer_created": False,
                        "optimizer_step": False,
                        "parameters_updated": False,
                        "scientific_claim_authorized": False,
                        "production_claim_authorized": False,
                        "full_sampler_executed": False,
                        "full_sampler_parity": False,
                    },
                    "cell": {
                        "cell_id": cell["cell_id"],
                        "source_iid": cell["source_iid"],
                        "source_video_sha256": source_sha,
                        "action_caption_present_in_registry_but_unused": True,
                        "action_caption_sha256": cell[
                            "action_caption_utf8_sha256"
                        ],
                        "noise_base_seed": int(cell["seeds"][0]),
                        "frame_count": FRAME_COUNT,
                        "latent_phases": LATENT_PHASES,
                        "bucket_hw": list(bucket_hw),
                        "registry_path": str(spec_path),
                        "registry_sha256": spec_sha,
                        "source_metadata": source_metadata,
                    },
                    "source_only_phase_a": {
                        "source_video_used": True,
                        "target_video_used": False,
                        "generated_proposal_used": False,
                        "source_retelling_used": False,
                        "proposal_selection_used": False,
                        "phase_b_only": True,
                        "phase_b_deferred_features": {
                            "source_retelling_paired_captions": (
                                "reserved_not_executed"
                            ),
                            "action_first_proposal_selection": (
                                "reserved_not_executed"
                            ),
                        },
                        "mask_pose_flow_track_or_donor_used": False,
                        "canonical_noop_r2v": prompt_contract,
                        "prompt_tensors": prompt_tensor_receipt,
                        "source_vae_latent_zS": source_latent_receipt,
                        "keyed_fresh_gaussian": noise.receipt,
                        "noisy_state_formula": "x=(1-sigma)*zS+sigma*eps",
                    },
                    "schedule": {
                        "runtime_exact40": coordinate.schedule_receipt,
                        "selected_coordinate": coordinate.coordinate_receipt,
                        "scheduler_state_unchanged": True,
                        "scheduler_step_called": False,
                    },
                    "identity_rebinder": {
                        "install_receipt": install_receipt,
                        "binding_receipt": binding_receipt,
                        "route_capability": route_capability,
                        "fresh_route_factory": local_receipt["route_factory"],
                        "atlas_rgb": atlas_rgb_receipt,
                        "external_atlas_owner_in_closure": True,
                    },
                    "native_phase_a": {
                        "four_forward_order": [
                            ["measurement", "negative"],
                            ["measurement", "positive"],
                            ["replay", "negative"],
                            ["replay", "positive"],
                        ],
                        "vendor_apg_leaf_vjp": True,
                        "flow_matching_objective": (
                            phase_core.FLOW_MATCHING_OBJECTIVE
                        ),
                        "local_receipt": local_receipt,
                        "pre_backward_world4": pre_backward_world4,
                        "world4_locality": world4_locality,
                        "synchronized_gradients": synchronized_gradients,
                        "gradient_owner_summary": gradient_owner_summary,
                        "zero_initialized_gradient_gate": gradient_gate,
                        "optimizer_created": False,
                        "parameters_updated": False,
                    },
                    "distributed": {
                        "topology": "WORLD4/SP4",
                        "rank0_parallel_receipt": sp_receipt,
                        "all_rank_parallel_receipt_set": sp_receipt_set,
                        "rank_metadata": gathered_rank_metadata,
                        "parity_payload": parity_payload,
                        "pre_backward_world4": pre_backward_world4,
                        "world4_locality": world4_locality,
                        "parity_row_count": len(parity_rows),
                        "all_rank_tensor_exact": True,
                        "all_rank_gradient_exact_after_sync": True,
                        "all_rank_assembled_receipt_exact": True,
                        "rank_local_receipt_equality_before_gradient_sync_claimed": False,
                    },
                    "parameter_closure": {
                        "initial_trainable_registry": initialization_receipt,
                        "trainable_bytes_unchanged": True,
                        "frozen_base_sha256_before": base_before,
                        "frozen_base_sha256_after": base_after,
                        "frozen_base_bytes_unchanged": True,
                        "frozen_base_gradients_all_none": True,
                    },
                    "provenance": {
                        "runner_source_binding": runner_source_binding,
                        "runner_sha256": args.expected_runner_sha256,
                        "phase_a_closure_sha256": (
                            args.expected_phase_a_closure_sha256
                        ),
                        "identity_rebinder_sha256": args.expected_rebinder_sha256,
                        "bernini_commit": bernini_revision,
                        "veomni_commit": veomni_revision,
                        "checkpoint_tree_sha256": (
                            args.expected_checkpoint_tree_sha256
                        ),
                        "checkpoint_content": checkpoint_identity,
                        "bernini_inference_files": inference_hashes,
                        "wan_diffusion_sha256": wan_sha,
                        "transformer_wan_sha256": file_sha256(
                            Path(transformer_wan.__file__).resolve(strict=True)
                        ),
                        "runtime_versions": {
                            "torch": torch.__version__,
                            "torch_hip": str(torch.version.hip),
                            "diffusers": diffusers_version,
                            "transformers": transformers_version,
                        },
                    },
                    "output_directory_identity": {
                        "st_dev": directory_identity[0],
                        "st_ino": directory_identity[1],
                        "created_fresh": True,
                    },
                    "receipt_publication": {
                        "canonical_immediate_serialization": True,
                        "digest_immediately_recomputed": True,
                        "create_only_O_EXCL": True,
                        "directory_fd_O_DIRECTORY_O_NOFOLLOW_retained": True,
                        "receipt_openat_directory_fd": True,
                        "mode": "0444",
                        "post_write_reread_byte_exact": True,
                        "atomic_replace_used": False,
                    },
                }
                receipt = seal_mapping(unsigned, digest_field="receipt_digest")
                owned_final, final_bytes = own_and_verify_receipt(
                    receipt, digest_field="receipt_digest"
                )
                if canonical_json_bytes(owned_final) != final_bytes:
                    raise GraftPhaseANativeGPUCanaryError(
                        "final canonical receipt bytes changed before publication"
                    )
                write_receipt_create_only(
                    output_dir / "receipt.json",
                    owned_final,
                    directory_fd=directory_fd,
                    expected_directory_identity=directory_identity,
                )
                print(final_bytes.decode("ascii"), flush=True)
            finally:
                os.close(directory_fd)
        dist.barrier()
        return 0
    finally:
        if handle is not None and not handle.restored and rebinder.active_route() is None:
            handle.restore()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_SCHEDULE_INDEX",
    "FreshAtlasRouteFactory",
    "GraftPhaseANativeGPUCanaryError",
    "KeyedGaussian",
    "LocalCanaryResult",
    "SCHEMA_VERSION",
    "bind_active_index33_coordinate",
    "build_noisy_target",
    "build_parser",
    "canonical_json_bytes",
    "canonical_noop_prompt_contract",
    "create_output_directory",
    "execute_authenticated_local_cell",
    "keyed_fresh_gaussian",
    "main",
    "object_sha256",
    "own_and_verify_receipt",
    "prepare_atlas_source_frames",
    "require_equal_rows",
    "route_capability_receipt",
    "seal_mapping",
    "tensor_identity",
    "validate_cli",
    "validate_world4_pre_backward_contexts",
    "validate_world4_locality_receipts",
    "write_receipt_create_only",
    "zero_initialized_gradient_gate",
]

#!/usr/bin/env python3
"""Seven-scenario four-rank torchrun admission for the trajectory tensor ABI.

The controller runs one happy and six hostile real four-process scenarios. It
never calls Bernini, a VAE, a renderer, or a media writer. The only optional
named output is the create-only admission receipt.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import re
import types
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-world4-admission-v5"
WORKER_SCHEMA = "case01-object-trajectory-exact5-world4-worker-v5"
SCENARIOS = (
    "happy", "hostile_rank0_tensor", "hostile_rank2_tensor",
    "hostile_rank0_aux", "hostile_rank2_abi", "hostile_rank1_row_build",
    "hostile_rank3_final_scheduler",
)
EXPECTED_WRAPPER_SHA256 = (
    "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9"
)
EXPECTED_PROJECTION_SHA256 = (
    "a851afd220d770e6c8082ff8e7f0a0c2b5a5be894bcbf9feeaf8bca4cc6c9e7e"
)
EXPECTED_SCAFFOLD_MODULE_SHA256 = (
    "a7d4e008e78d373370b832c0754e5d7420b504fa8b2022eef9a9bb899added8a"
)
EXPECTED_SCAFFOLD_SHA256 = (
    "7b1bec6e9764a1297bb0029f8fea01ebe4b2deab0acc2c7f07fdee96bc0a098a"
)
EXPECTED_SCAFFOLD_DIGEST = (
    "5e6156909d8261a23c3add3134059bec20505b682ca0eb13dc88fa8512eeace1"
)
PACKED_TOKENS = 19_530
PACKED_CHANNELS = 64
STEPS = 40
SCENARIO_TIMEOUT_SECONDS = 30
PROCESS_GROUP_REAP_SECONDS = 2
SHA_RE = re.compile(r"[0-9a-f]{64}")
VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}")
NO_HIP_SENTINEL = "NONE"
UNSET_ENV_SENTINEL = "__UNSET__"
GPU_VISIBILITY_KEYS = (
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
)
TORCH_SOURCE_ARGUMENTS = (
    "torchrun_source", "torchrun_handler_source", "torch_local_agent_source",
    "torch_dynamic_rendezvous_source", "torch_multiprocessing_api_source",
)
CPU_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class World4ProbeError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_rdev, info.st_size,
        getattr(info, "st_blocks", 0), info.st_mtime_ns, info.st_ctime_ns,
    )


def _read_pinned(
    path_value: str, expected: str, *, executable: bool = False,
    require_single_link: bool = True,
) -> tuple[Path, bytes]:
    path = Path(path_value)
    if (
        SHA_RE.fullmatch(expected) is None or not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
    ):
        raise World4ProbeError(f"incomplete/noncanonical source: {path}")
    try:
        named = os.lstat(path)
    except OSError as error:
        raise World4ProbeError(f"missing source: {path}") from error
    if (
        not stat.S_ISREG(named.st_mode)
        or (require_single_link and named.st_nlink != 1)
        or (not require_single_link and named.st_nlink < 1)
        or (executable and not named.st_mode & 0o111)
        or path.resolve(strict=True) != path
    ):
        raise World4ProbeError(f"source is not one regular file: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (require_single_link and before.st_nlink != 1)
            or (not require_single_link and before.st_nlink < 1)
            or _identity(before) != _identity(named)
            or (executable and not before.st_mode & 0o111)
        ):
            raise World4ProbeError(f"opened source differs before read: {path}")
        blocks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor, min(1_048_576, before.st_size - offset), offset,
            )
            if not block:
                break
            blocks.append(block); offset += len(block)
        raw = b"".join(blocks)
        middle = os.fstat(descriptor)
        replay = b"".join(
            os.pread(descriptor, min(1_048_576, before.st_size - at), at)
            for at in range(0, before.st_size, 1_048_576)
        )
        eof = os.pread(descriptor, 1, before.st_size)
        after = os.fstat(descriptor); named_after = os.lstat(path)
    finally:
        os.close(descriptor)
    if (
        len(raw) != before.st_size or raw != replay or eof != b""
        or _identity(named) != _identity(before)
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
        or hashlib.sha256(raw).hexdigest() != expected
    ):
        raise World4ProbeError(f"source identity/SHA differs: {path}")
    return path, raw


def _load(path_value: str, expected: str, name: str) -> Any:
    path, raw = _read_pinned(path_value, expected)
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    if spec is None:
        raise World4ProbeError(f"cannot create module spec: {name}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(raw.decode("utf-8", "strict"), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_scaffold(path_value: str) -> dict[str, Any]:
    _, raw = _read_pinned(path_value, EXPECTED_SCAFFOLD_SHA256)
    value = json.loads(raw)
    if (
        type(value) is not dict or raw != canonical(value) + b"\n"
        or value.get("artifact_digest") != EXPECTED_SCAFFOLD_DIGEST
    ):
        raise World4ProbeError("scaffold canonical authority differs")
    return value


def _publication_empty(path_value: str) -> Path:
    root = Path(path_value)
    if not root.is_absolute() or os.path.normpath(str(root)) != str(root):
        raise World4ProbeError("publication root path is not canonical")
    info = os.lstat(root)
    if (
        not stat.S_ISDIR(info.st_mode) or root.resolve(strict=True) != root
        or any(root.iterdir())
    ):
        raise World4ProbeError("publication root must be an empty real directory")
    return root


def _process_group_state(process_group: int) -> str:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return "absent"
    except PermissionError:
        return "inaccessible"
    return "present"


def _signal_live_process_group(
    process: subprocess.Popen[str], process_group: int, signal_number: int,
) -> str:
    """Signal only while the original session leader still anchors its PGID."""

    if process.poll() is not None:
        return "leader_reaped"
    try:
        observed_group = os.getpgid(process.pid)
    except ProcessLookupError:
        return "leader_missing"
    except PermissionError:
        return "leader_inaccessible"
    if observed_group != process_group or process_group != process.pid:
        return "identity_changed"
    state = _process_group_state(process_group)
    if state != "present":
        return state
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        return "absent"
    except PermissionError:
        return "inaccessible"
    return "signaled"


def _process_group_absent(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        state = _process_group_state(process_group)
        if state == "absent":
            return True
        # Darwin can transiently report EPERM for a dying group after SIGKILL.
        # It is never evidence of absence, but bounded polling may still reach
        # the only accepted terminal observation: ESRCH.
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _terminate_and_reap_process_group(
    process: subprocess.Popen[str], process_group: int,
) -> tuple[bool, bool]:
    """Boundedly terminate the full torchrun session and reap its leader.

    The two booleans independently attest that the direct child was reaped and
    that the kernel no longer resolves the process-group id.  A caller must not
    turn either false value into a successful admission or a timeout receipt.
    """

    _signal_live_process_group(process, process_group, signal.SIGTERM)
    try:
        process.communicate(timeout=PROCESS_GROUP_REAP_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_live_process_group(process, process_group, signal.SIGKILL)
        try:
            process.communicate(timeout=PROCESS_GROUP_REAP_SECONDS)
        except subprocess.TimeoutExpired:
            # Reassert both group and leader kills: a hostile descendant may
            # still hold the captured pipes even after the leader has exited.
            _signal_live_process_group(process, process_group, signal.SIGKILL)
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=PROCESS_GROUP_REAP_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
    if process.poll() is None:
        _signal_live_process_group(process, process_group, signal.SIGKILL)
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=PROCESS_GROUP_REAP_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    direct_child_reaped = process.poll() is not None
    # Once the leader is reaped its numeric PID/PGID may be reused.  Never
    # signal that naked number again; only ESRCH from a bounded query proves
    # absence, while EPERM remains an explicit fail-closed result.
    process_group_absent = _process_group_absent(
        process_group, PROCESS_GROUP_REAP_SECONDS,
    )
    return direct_child_reaped, process_group_absent


def _receipt_target_fresh(path_value: str | None) -> None:
    if path_value is None:
        return
    target = Path(path_value)
    try:
        parent = os.lstat(target.parent)
    except OSError as error:
        raise World4ProbeError("world4 receipt parent is missing") from error
    if (
        not target.is_absolute() or os.path.normpath(str(target)) != str(target)
        or os.path.lexists(target) or not stat.S_ISDIR(parent.st_mode)
        or target.parent.resolve(strict=True) != target.parent
    ):
        raise World4ProbeError("world4 receipt target is not fresh")


def _scheduler_class(torch: Any, step_count: int = STEPS) -> type:
    class UniPCMultistepScheduler:
        def __init__(self) -> None:
            self.config = {
                "_class_name": "UniPCMultistepScheduler",
                "prediction_type": "flow_prediction",
                "use_flow_sigmas": True,
                "predict_x0": True,
                "final_sigmas_type": "zero",
                "flow_shift": 5.0,
            }
            self.sigmas = torch.linspace(1.0, 0.0, step_count + 1)
            self.timesteps = torch.linspace(1000.0, 25.0, step_count)
            self.step_index = None
            self.calls = 0

        def step(self, model_output: Any, timestep: Any, sample: Any,
                 return_dict: bool = True) -> tuple[Any]:
            if return_dict is not False:
                raise World4ProbeError("synthetic scheduler requires return_dict=False")
            index = 0 if self.step_index is None else int(self.step_index)
            self.calls += 1
            # The admission is about the projection wrapper, not a synthetic
            # numerical integrator.  A distinct clone satisfies the native
            # UniPC return/storage ABI while keeping all 40 steps bounded.
            previous = sample.clone().contiguous()
            self.step_index = index + 1
            return (previous,)

    return UniPCMultistepScheduler


def _configure_cpu_thread_contract(torch: Any) -> dict[str, Any]:
    observed_environment = {
        key: os.environ.get(key) for key in CPU_THREAD_ENVIRONMENT
    }
    if observed_environment != CPU_THREAD_ENVIRONMENT:
        raise World4ProbeError(
            "worker CPU thread environment differs: "
            + canonical(observed_environment).decode("utf-8")
        )
    required = (
        "set_num_threads", "set_num_interop_threads",
        "get_num_threads", "get_num_interop_threads",
    )
    if any(not callable(getattr(torch, name, None)) for name in required):
        raise World4ProbeError("Torch CPU thread-control API is incomplete")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        num_threads = int(torch.get_num_threads())
        num_interop_threads = int(torch.get_num_interop_threads())
    except Exception as error:
        raise World4ProbeError("Torch CPU thread-control call failed") from error
    if num_threads != 1 or num_interop_threads != 1:
        raise World4ProbeError("Torch CPU reported thread counts differ")
    return {
        "environment": dict(CPU_THREAD_ENVIRONMENT),
        "torch_num_threads": num_threads,
        "torch_num_interop_threads": num_interop_threads,
    }


def _expected_runtime_versions(args: argparse.Namespace) -> dict[str, Any]:
    torch_version = args.expected_torch_version
    hip_argument = args.expected_hip_version
    if (
        type(torch_version) is not str
        or VERSION_RE.fullmatch(torch_version) is None
        or type(hip_argument) is not str
        or (
            hip_argument != NO_HIP_SENTINEL
            and VERSION_RE.fullmatch(hip_argument) is None
        )
    ):
        raise World4ProbeError("expected Torch/HIP version contract differs")
    return {
        "torch": torch_version,
        "hip": None if hip_argument == NO_HIP_SENTINEL else hip_argument,
    }


def _expected_gpu_spec(args: argparse.Namespace) -> dict[str, Any]:
    count = args.expected_gpu_count
    if type(count) is not int or not (0 <= count <= 64):
        raise World4ProbeError("expected GPU count differs")
    visibility: dict[str, str | None] = {}
    for key in GPU_VISIBILITY_KEYS:
        value = getattr(args, "expected_" + key.lower())
        if type(value) is not str or len(value) > 256:
            raise World4ProbeError("expected GPU visibility contract differs")
        visibility[key] = None if value == UNSET_ENV_SENTINEL else value
    return {"device_count": count, "visibility_environment": visibility}


def _configure_gpu_contract(torch: Any, args: argparse.Namespace) -> dict[str, Any]:
    expected = _expected_gpu_spec(args)
    observed = {key: os.environ.get(key) for key in GPU_VISIBILITY_KEYS}
    try:
        device_count = int(torch.cuda.device_count())
    except Exception as error:
        raise World4ProbeError("Torch GPU device-count query failed") from error
    if (
        observed != expected["visibility_environment"]
        or device_count != expected["device_count"]
    ):
        raise World4ProbeError(
            "worker GPU visibility/device-count contract differs: "
            + canonical({
                "expected": expected,
                "observed": {
                    "device_count": device_count,
                    "visibility_environment": observed,
                },
            }).decode("utf-8")
        )
    return {
        "expected_device_count": expected["device_count"],
        "device_count": device_count,
        "visibility_environment": observed,
    }


def worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    cpu_thread_contract = _configure_cpu_thread_contract(torch)
    gpu_contract = _configure_gpu_contract(torch, args)
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != 4 or rank not in range(4) or local_rank not in range(4):
        raise World4ProbeError("worker is not an exact torchrun world4 rank")
    expected_runtime_versions = _expected_runtime_versions(args)
    torch_version = str(torch.__version__)
    torch_hip_version = getattr(torch.version, "hip", None)
    if (
        torch_version != expected_runtime_versions["torch"]
        or torch_hip_version != expected_runtime_versions["hip"]
    ):
        raise World4ProbeError(
            "worker Torch/HIP runtime version differs: "
            + canonical({
                "expected": expected_runtime_versions,
                "observed": {"torch": torch_version, "hip": torch_hip_version},
            }).decode("utf-8")
        )
    python_path, python_raw = _read_pinned(
        args.python, args.python_sha256, executable=True,
    )
    if Path(sys.executable).resolve() != python_path:
        raise World4ProbeError("worker Python executable identity differs")
    torch_root = Path(torch.__file__).resolve().parent
    expected_torch_paths = {
        "torchrun_source": torch_root / "distributed/run.py",
        "torchrun_handler_source": torch_root
        / "distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py",
        "torch_local_agent_source": torch_root
        / "distributed/elastic/agent/server/local_elastic_agent.py",
        "torch_dynamic_rendezvous_source": torch_root
        / "distributed/elastic/rendezvous/dynamic_rendezvous.py",
        "torch_multiprocessing_api_source": torch_root
        / "distributed/elastic/multiprocessing/api.py",
    }
    runtime_rows: dict[str, dict[str, Any]] = {
        "python": {
            "path": str(python_path), "sha256": args.python_sha256,
            "size": len(python_raw),
        }
    }
    for role, expected_path in expected_torch_paths.items():
        path_value = getattr(args, role)
        sha256 = getattr(args, role + "_sha256")
        path, raw = _read_pinned(path_value, sha256)
        loaded_path, loaded_raw = _read_pinned(
            str(expected_path), sha256, require_single_link=False,
        )
        if loaded_path != expected_path or loaded_raw != raw:
            raise World4ProbeError(f"loaded Torch source bytes differ: {role}")
        runtime_rows[role] = {
            "path": str(path), "sha256": sha256, "size": len(raw),
        }
    runtime_digest = digest(runtime_rows)
    publication = _publication_empty(args.publication_root)
    wrapper = _load(args.wrapper, EXPECTED_WRAPPER_SHA256,
                    f"_case01_world4_wrapper_rank{rank}")
    projection = _load(args.projection, EXPECTED_PROJECTION_SHA256,
                       f"_case01_world4_projection_rank{rank}")
    scaffold_module = _load(
        args.scaffold_module, EXPECTED_SCAFFOLD_MODULE_SHA256,
        f"_case01_world4_scaffold_rank{rank}",
    )
    scaffold = _load_scaffold(args.scaffold)
    scaffold_module._validate_artifact(scaffold)
    if dist.is_initialized():
        raise World4ProbeError("distributed group was initialized before admission")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        if dist.get_backend() != "gloo":
            raise World4ProbeError("world4 CPU admission requires gloo")
        broadcast = (
            torch.zeros(
                1, PACKED_TOKENS, PACKED_CHANNELS, dtype=torch.float16,
            )
            if rank == 0
            else torch.zeros(
                1, PACKED_TOKENS, PACKED_CHANNELS, dtype=torch.float16,
            )
        )
        dist.broadcast(broadcast, src=0)
        broadcast_calls = 1
        source = broadcast.detach().contiguous()
        aux = source.clone().contiguous()
        token_plan = wrapper.compile_scaffold_token_plan(scaffold)
        effective = sorted({
            index for phase in token_plan["phases"]
            for index in phase["effective_origin"]
        })
        if not effective:
            raise World4ProbeError("effective origin support is empty")
        aux[:, effective, :] += 1.0
        tensor_hostile_rank = {
            "hostile_rank0_tensor": 0, "hostile_rank2_tensor": 2,
        }.get(args.scenario)
        if tensor_hostile_rank == rank:
            source[0, 0, 0] += 17.0
        arm = (
            "trajectory_bone_only"
            if args.scenario in {"hostile_rank0_tensor", "hostile_rank0_aux"}
            else "trajectory_dog_bone"
        )
        rows, row_evidence = wrapper.build_projection_rows(
            arm=arm, scaffold=scaffold,
            source_packed=source, aux_packed=aux, projection_module=projection,
        )
        expected_names = (
            ["legacy_phase0_hard1_every_step", "bone_conservation_all_sigma"]
            if arm == "trajectory_bone_only"
            else [
                "legacy_phase0_hard1_every_step",
                "bone_conservation_all_sigma",
                "dog_core_low_mid",
            ]
        )
        if (
            [row.name for row in rows] != expected_names
            or row_evidence["dog_row_consumed"]
            is not (arm == "trajectory_dog_bone")
        ):
            raise World4ProbeError("active row closure differs")
        contract = wrapper._projection_contract(
            arm=arm, expected_steps=STEPS, row_evidence=row_evidence,
        )
        consensus_failed = False
        try:
            consensus = wrapper._four_rank_projection_consensus(contract)
        except wrapper.ObjectOracleWrapperError as error:
            if (
                "four-rank arm/row/gate/plan/tensor projection contracts differ"
                not in str(error)
            ):
                raise
            consensus_failed = True
            consensus = None
        expected_consensus_failure = tensor_hostile_rank is not None
        if consensus_failed != expected_consensus_failure:
            raise World4ProbeError("hostile consensus disposition differs")
        stage_gate_failed = False
        stage_name: str | None = None
        operational_path = "projection_contract_consensus"
        operational_aux_gate_count = 0
        operational_projection_gate_count = 0
        operational_wrapper_trace_steps = 0
        aux_broadcast_calls = 0
        trace_steps = 0
        scheduler_calls = 0
        scheduler_token_count = 0

        def make_state() -> Any:
            legacy = types.SimpleNamespace(
                _pack_wan_source_latent=lambda value: value,
            )
            assets = types.SimpleNamespace(
                cli=types.SimpleNamespace(arm=arm),
                scaffold=scaffold, projection_module=projection,
            )
            state = wrapper.OracleExecutionState(legacy=legacy, assets=assets)
            if rank == 0:
                state.source_vae_encode_calls = 1
                state.aux_vae_encode_attempts = 1
                state.aux_vae_encode_calls = 1
                state.aux_latent = aux.clone().contiguous()
            return state

        if args.scenario in {"hostile_rank0_aux", "hostile_rank2_abi"}:
            stage_name = "aux_readiness"
            operational_path = "oracle_execution_state.distributed_aux"
            state = make_state()
            original_empty_like = torch.empty_like
            if args.scenario == "hostile_rank0_aux" and rank == 0:
                state.aux_vae_encode_calls = 0
                state.aux_latent = None
                state.aux_encode_error = RuntimeError(
                    "injected rank0 aux encode failure"
                )
            if args.scenario == "hostile_rank2_abi" and rank == 2:
                torch.empty_like = lambda _value: torch.empty(
                    1, dtype=source.dtype, device=source.device,
                )
            try:
                state.distributed_aux(source)
            except wrapper.ObjectOracleWrapperError as error:
                if f"one or more ranks failed {stage_name}" not in str(error):
                    raise
                stage_gate_failed = True
            finally:
                torch.empty_like = original_empty_like
            operational_aux_gate_count = len(state.aux_collective_gates)
            aux_broadcast_calls = state.aux_broadcast_calls
            if not stage_gate_failed:
                raise World4ProbeError(f"{stage_name} hostile gate did not fail")
        elif args.scenario == "hostile_rank1_row_build":
            stage_name = "projection_row_build"
            operational_path = "oracle_execution_state.clamp_row_build"
            state = make_state()
            scheduler_for_gate = _scheduler_class(torch, 1)()
            diffusion = types.SimpleNamespace(
                use_unipc=True, scheduler=scheduler_for_gate,
            )
            original_tensor_authority = wrapper._tensor_byte_authority
            if rank == 1:
                def reject_tensor_authority(*_values: Any, **_named: Any) -> Any:
                    raise RuntimeError("injected row-build tensor authority failure")
                wrapper._tensor_byte_authority = reject_tensor_authority
            try:
                with state.clamp(diffusion, source, expected_steps=1):
                    raise World4ProbeError("row-build hostile unexpectedly yielded")
            except wrapper.ObjectOracleWrapperError as error:
                if f"one or more ranks failed {stage_name}" not in str(error):
                    raise
                stage_gate_failed = True
            finally:
                wrapper._tensor_byte_authority = original_tensor_authority
            operational_aux_gate_count = len(state.aux_collective_gates)
            operational_projection_gate_count = len(
                state.projection_collective_gates
            )
            aux_broadcast_calls = state.aux_broadcast_calls
            if not stage_gate_failed:
                raise World4ProbeError(f"{stage_name} hostile gate did not fail")

        if args.scenario in {"happy", "hostile_rank3_final_scheduler"}:
            state = make_state()
            operational_scheduler = _scheduler_class(torch, STEPS)()
            diffusion = types.SimpleNamespace(
                use_unipc=True, scheduler=operational_scheduler,
            )
            operational_path = "oracle_execution_state.clamp_full_path"
            if args.scenario == "hostile_rank3_final_scheduler":
                stage_name = "projection_final_validation"
            facade = None
            sample = torch.zeros_like(source)
            model_output = torch.zeros_like(source)
            try:
                with state.clamp(diffusion, source, expected_steps=STEPS) as facade:
                    count = (
                        STEPS - 1
                        if args.scenario == "hostile_rank3_final_scheduler"
                        and rank == 3 else STEPS
                    )
                    for index in range(count):
                        sample = operational_scheduler.step(
                            model_output, operational_scheduler.timesteps[index],
                            sample, return_dict=False,
                        )[0]
            except wrapper.ObjectOracleWrapperError as error:
                if (
                    args.scenario != "hostile_rank3_final_scheduler"
                    or f"one or more ranks failed {stage_name}" not in str(error)
                ):
                    raise
                stage_gate_failed = True
            operational_aux_gate_count = len(state.aux_collective_gates)
            operational_projection_gate_count = len(
                state.projection_collective_gates
            )
            aux_broadcast_calls = state.aux_broadcast_calls
            if facade is not None and facade.core_trace is not None:
                operational_wrapper_trace_steps = len(facade.core_trace.records)
            trace_steps = operational_wrapper_trace_steps
            scheduler_calls = operational_scheduler.calls
            scheduler_token_count = PACKED_TOKENS
            if args.scenario == "happy" and (
                operational_wrapper_trace_steps != STEPS
                or scheduler_calls != STEPS
                or operational_aux_gate_count != 2
                or operational_projection_gate_count != 7
            ):
                raise World4ProbeError("full wrapper happy path differs")
            if (
                args.scenario == "hostile_rank3_final_scheduler"
                and not stage_gate_failed
            ):
                raise World4ProbeError(
                    "actual wrapper final-scheduler gate did not fail"
                )
            if args.scenario == "hostile_rank3_final_scheduler" and (
                trace_steps != (STEPS - 1 if rank == 3 else STEPS)
                or scheduler_calls != (STEPS - 1 if rank == 3 else STEPS)
                or operational_projection_gate_count != 6
            ):
                raise World4ProbeError("actual final scheduler trace differs")
        expected_failure = args.scenario != "happy"
        if not expected_failure and (consensus_failed or stage_gate_failed):
            raise World4ProbeError("happy scenario reported a hostile failure")
        if expected_failure and not (consensus_failed or stage_gate_failed):
            raise World4ProbeError("hostile scenario did not fail closed")
        row = {
            "rank": rank, "local_rank": local_rank,
            "scenario": args.scenario, "world_size": world_size,
            "python_optimize_level": sys.flags.optimize,
            "torch_version": torch_version, "distributed_backend": "gloo",
            "torch_hip_version": torch_hip_version,
            "expected_torch_version": expected_runtime_versions["torch"],
            "expected_hip_version": expected_runtime_versions["hip"],
            "gpu_visibility_environment": gpu_contract["visibility_environment"],
            "expected_gpu_count": gpu_contract["expected_device_count"],
            "torch_visible_gpu_count": gpu_contract["device_count"],
            "cpu_thread_environment": cpu_thread_contract["environment"],
            "torch_num_threads": cpu_thread_contract["torch_num_threads"],
            "torch_num_interop_threads": (
                cpu_thread_contract["torch_num_interop_threads"]
            ),
            "source_broadcast_calls": broadcast_calls,
            "aux_broadcast_calls": aux_broadcast_calls,
            "active_arm": arm, "row_count": len(rows),
            "consensus_failed": consensus_failed,
            "stage_gate_failed": stage_gate_failed,
            "failure_stage": (
                "projection_contract_consensus" if consensus_failed else stage_name
            ),
            "trace_steps": trace_steps, "scheduler_calls": scheduler_calls,
            "scheduler_token_count": scheduler_token_count,
            "operational_path": operational_path,
            "operational_aux_gate_count": operational_aux_gate_count,
            "operational_projection_gate_count": (
                operational_projection_gate_count
            ),
            "operational_wrapper_trace_steps": (
                operational_wrapper_trace_steps
            ),
            "publication_empty": not any(publication.iterdir()),
            "scaffold_digest": scaffold["artifact_digest"],
            "runtime_identity_digest": runtime_digest,
        }
        row["row_digest"] = digest(row)
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, row)
        if any(item.get("rank") != index for index, item in enumerate(gathered)):
            raise World4ProbeError("ordered rank evidence differs")
        result = {
            "schema_version": WORKER_SCHEMA,
            "scenario": args.scenario,
            "status": "PASS_EXPECTED_HOSTILE" if expected_failure else "PASS_HAPPY",
            "world_size": world_size,
            "cpu_thread_contract": cpu_thread_contract,
            "expected_runtime_versions": expected_runtime_versions,
            "expected_gpu_contract": {
                "device_count": gpu_contract["expected_device_count"],
                "visibility_environment": gpu_contract["visibility_environment"],
            },
            "rank_rows": gathered,
            "publication_performed": False,
        }
        result["result_digest"] = digest(result)
        if rank == 0:
            encoded = base64.b64encode(canonical(result)).decode("ascii")
            print("WORLD4_RESULT_B64=" + encoded, flush=True)
        return result
    finally:
        dist.destroy_process_group()


def _run_scenario(args: argparse.Namespace, scenario: str) -> dict[str, Any]:
    optimize_flag = (
        ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
    )
    command = [
        args.python, *optimize_flag,
        "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=4", "--no-python",
        args.python, *optimize_flag,
        str(Path(__file__).resolve()), "worker",
        "--scenario", scenario,
        "--wrapper", args.wrapper,
        "--projection", args.projection,
        "--scaffold-module", args.scaffold_module,
        "--scaffold", args.scaffold,
        "--publication-root", args.publication_root,
        "--python", args.python,
        "--python-sha256", args.python_sha256,
        "--expected-torch-version", args.expected_torch_version,
        "--expected-hip-version", args.expected_hip_version,
        "--expected-gpu-count", str(args.expected_gpu_count),
    ]
    for key in GPU_VISIBILITY_KEYS:
        command.extend([
            "--expected-" + key.lower().replace("_", "-"),
            getattr(args, "expected_" + key.lower()),
        ])
    for role in TORCH_SOURCE_ARGUMENTS:
        command.extend([
            "--" + role.replace("_", "-"), getattr(args, role),
            "--" + role.replace("_", "-") + "-sha256",
            getattr(args, role + "_sha256"),
        ])
    started = time.monotonic()
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
        env={
            **os.environ, **CPU_THREAD_ENVIRONMENT,
            "PYTHONUNBUFFERED": "1",
        },
    )
    try:
        process_group = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError) as error:
        try:
            process.communicate(timeout=PROCESS_GROUP_REAP_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                process.kill(); process.wait(timeout=PROCESS_GROUP_REAP_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass
        raise World4ProbeError(
            f"torchrun session leader disappeared before PGID pin: {scenario}"
        ) from error
    if process_group != process.pid:
        try:
            process.kill(); process.wait(timeout=PROCESS_GROUP_REAP_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass
        raise World4ProbeError(
            f"torchrun start_new_session PGID differs: {scenario}"
        )
    try:
        stdout, stderr = process.communicate(timeout=SCENARIO_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        direct_child_reaped, group_absent = _terminate_and_reap_process_group(
            process, process_group,
        )
        publication = _publication_empty(args.publication_root)
        _receipt_target_fresh(args.output)
        if not direct_child_reaped or not group_absent:
            raise World4ProbeError(
                "torchrun process group could not be reaped: "
                f"{scenario}; direct_child_reaped={direct_child_reaped}; "
                f"process_group_absent={group_absent}"
            ) from error
        if any(publication.iterdir()):
            raise World4ProbeError(
                f"timeout path created publication: {scenario}"
            ) from error
        raise World4ProbeError(
            f"torchrun exceeded {SCENARIO_TIMEOUT_SECONDS} seconds; "
            f"process group reaped and publication remained empty: {scenario}"
        ) from error
    elapsed = time.monotonic() - started
    process_group_reaped = _process_group_absent(
        process_group, PROCESS_GROUP_REAP_SECONDS,
    )
    publication = _publication_empty(args.publication_root)
    if not process_group_reaped:
        publication = _publication_empty(args.publication_root)
        _receipt_target_fresh(args.output)
        raise World4ProbeError(
            "torchrun residual/inaccessible process group after leader reap; "
            f"refusing unsafe numeric-PGID signal: {scenario}"
        )
    if process.returncode != 0:
        raise World4ProbeError(
            f"torchrun failed for {scenario}: rc={process.returncode}: "
            + stderr[-12000:]
        )
    markers = [
        line.split("=", 1)[1] for line in stdout.splitlines()
        if line.startswith("WORLD4_RESULT_B64=")
    ]
    if len(markers) != 1:
        raise World4ProbeError(f"torchrun result marker differs: {scenario}")
    try:
        worker_raw = base64.b64decode(markers[0], validate=True)
        result = json.loads(worker_raw)
    except (ValueError, json.JSONDecodeError) as error:
        raise World4ProbeError(f"invalid worker result: {scenario}") from error
    unsigned = dict(result) if type(result) is dict else {}
    claimed = unsigned.pop("result_digest", None)
    if (
        worker_raw != canonical(result)
        or claimed != digest(unsigned)
        or set(result) != {
            "schema_version", "scenario", "status", "world_size",
            "cpu_thread_contract", "expected_runtime_versions", "rank_rows",
            "expected_gpu_contract",
            "publication_performed",
            "result_digest",
        }
        or result.get("schema_version") != WORKER_SCHEMA
        or result.get("scenario") != scenario
        or result.get("world_size") != 4
        or result.get("publication_performed") is not False
        or result.get("cpu_thread_contract") != {
            "environment": CPU_THREAD_ENVIRONMENT,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
        }
        or result.get("expected_runtime_versions")
        != _expected_runtime_versions(args)
        or result.get("expected_gpu_contract") != _expected_gpu_spec(args)
        or type(result.get("rank_rows")) is not list
        or len(result["rank_rows"]) != 4
    ):
        raise World4ProbeError(f"worker result closure differs: {scenario}")
    return {
        "scenario": scenario, "worker_result": result,
        "elapsed_milliseconds": int(elapsed * 1000),
        "timeout_seconds": SCENARIO_TIMEOUT_SECONDS,
        "process_group_id": process_group,
        "process_group_reaped": process_group_reaped,
        "publication_empty_after_scenario": not any(publication.iterdir()),
        "worker_optimize_level": sys.flags.optimize,
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
    }


def _controller_preflight(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if SHA_RE.fullmatch(EXPECTED_WRAPPER_SHA256) is None:
        raise World4ProbeError("HOLD: final wrapper pin is blocked")
    _expected_runtime_versions(args)
    _expected_gpu_spec(args)
    rows: dict[str, dict[str, Any]] = {}
    sources = {
        "python": (args.python, args.python_sha256, True),
        "wrapper": (args.wrapper, EXPECTED_WRAPPER_SHA256, False),
        "projection": (args.projection, EXPECTED_PROJECTION_SHA256, False),
        "scaffold_module": (
            args.scaffold_module, EXPECTED_SCAFFOLD_MODULE_SHA256, False,
        ),
        "scaffold": (args.scaffold, EXPECTED_SCAFFOLD_SHA256, False),
    }
    for role in TORCH_SOURCE_ARGUMENTS:
        sources[role] = (
            getattr(args, role), getattr(args, role + "_sha256"), False,
        )
    for role, (path_value, pin, executable) in sources.items():
        path, raw = _read_pinned(path_value, pin, executable=executable)
        rows[role] = {
            "path": str(path), "sha256": pin, "size": len(raw),
        }
    _load_scaffold(args.scaffold)
    _publication_empty(args.publication_root)
    _receipt_target_fresh(args.output)
    return rows


def controller(args: argparse.Namespace) -> dict[str, Any]:
    publication = _publication_empty(args.publication_root)
    runtime_identities = _controller_preflight(args)
    results = [_run_scenario(args, scenario) for scenario in SCENARIOS]
    if any(publication.iterdir()):
        raise World4ProbeError("world4 admission created a publication leaf")
    if (
        [row["scenario"] for row in results] != list(SCENARIOS)
        or results[0]["worker_result"]["status"] != "PASS_HAPPY"
        or any(
            row["worker_result"]["status"] != "PASS_EXPECTED_HOSTILE"
            for row in results[1:]
        )
        or any(row["worker_result"]["world_size"] != 4 for row in results)
        or any(
            set(row) != {
                "scenario", "worker_result", "elapsed_milliseconds",
                "timeout_seconds", "process_group_id", "process_group_reaped",
                "publication_empty_after_scenario", "worker_optimize_level",
                "stdout_sha256", "stderr_sha256",
            }
            or row["timeout_seconds"] != SCENARIO_TIMEOUT_SECONDS
            or type(row["process_group_id"]) is not int
            or row["process_group_id"] <= 1
            or row["process_group_reaped"] is not True
            or row["publication_empty_after_scenario"] is not True
            or row["worker_optimize_level"] != sys.flags.optimize
            or not (0 <= row["elapsed_milliseconds"]
                    < SCENARIO_TIMEOUT_SECONDS * 1000)
            for row in results
        )
        or any(
            rank["torch_version"] != args.expected_torch_version
            or rank["torch_hip_version"]
            != _expected_runtime_versions(args)["hip"]
            for scenario in results
            for rank in scenario["worker_result"]["rank_rows"]
        )
    ):
        raise World4ProbeError("world4 scenario closure differs")
    expected_failure_stage = {
        "happy": None,
        "hostile_rank0_tensor": "projection_contract_consensus",
        "hostile_rank2_tensor": "projection_contract_consensus",
        "hostile_rank0_aux": "aux_readiness",
        "hostile_rank2_abi": "aux_readiness",
        "hostile_rank1_row_build": "projection_row_build",
        "hostile_rank3_final_scheduler": "projection_final_validation",
    }
    for scenario in results:
        name = scenario["scenario"]
        rows = scenario["worker_result"]["rank_rows"]
        for rank_index, row in enumerate(rows):
            unsigned = dict(row); claimed = unsigned.pop("row_digest", None)
            expected_arm = (
                "trajectory_bone_only"
                if name in {"hostile_rank0_tensor", "hostile_rank0_aux"}
                else "trajectory_dog_bone"
            )
            expected_count = 2 if expected_arm == "trajectory_bone_only" else 3
            expected_operational = {
                "happy": (
                    "oracle_execution_state.clamp_full_path", 2, 7, STEPS, 1,
                ),
                "hostile_rank0_tensor": (
                    "projection_contract_consensus", 0, 0, 0, 0,
                ),
                "hostile_rank2_tensor": (
                    "projection_contract_consensus", 0, 0, 0, 0,
                ),
                "hostile_rank0_aux": (
                    "oracle_execution_state.distributed_aux", 0, 0, 0, 0,
                ),
                "hostile_rank2_abi": (
                    "oracle_execution_state.distributed_aux", 0, 0, 0, 0,
                ),
                "hostile_rank1_row_build": (
                    "oracle_execution_state.clamp_row_build", 2, 1, 0, 1,
                ),
                "hostile_rank3_final_scheduler": (
                    "oracle_execution_state.clamp_full_path", 2, 6,
                    STEPS - 1 if rank_index == 3 else STEPS, 1,
                ),
            }[name]
            if (
                set(row) != {
                    "rank", "local_rank", "scenario", "world_size",
                    "python_optimize_level",
                    "torch_version", "distributed_backend",
                    "torch_hip_version", "expected_torch_version",
                    "expected_hip_version",
                    "gpu_visibility_environment", "expected_gpu_count",
                    "torch_visible_gpu_count",
                    "cpu_thread_environment", "torch_num_threads",
                    "torch_num_interop_threads",
                    "source_broadcast_calls", "aux_broadcast_calls",
                    "active_arm", "row_count", "consensus_failed",
                    "stage_gate_failed", "failure_stage", "trace_steps",
                    "scheduler_calls", "scheduler_token_count",
                    "operational_path", "operational_aux_gate_count",
                    "operational_projection_gate_count",
                    "operational_wrapper_trace_steps",
                    "publication_empty", "scaffold_digest",
                    "runtime_identity_digest", "row_digest",
                }
                or row.get("rank") != rank_index or row.get("local_rank") != rank_index
                or row.get("world_size") != 4
                or row.get("python_optimize_level") != sys.flags.optimize
                or row.get("scenario") != name
                or row.get("distributed_backend") != "gloo"
                or row.get("torch_version") != args.expected_torch_version
                or row.get("torch_hip_version")
                != _expected_runtime_versions(args)["hip"]
                or row.get("expected_torch_version")
                != args.expected_torch_version
                or row.get("expected_hip_version")
                != _expected_runtime_versions(args)["hip"]
                or row.get("gpu_visibility_environment")
                != _expected_gpu_spec(args)["visibility_environment"]
                or row.get("expected_gpu_count")
                != _expected_gpu_spec(args)["device_count"]
                or row.get("torch_visible_gpu_count")
                != _expected_gpu_spec(args)["device_count"]
                or row.get("cpu_thread_environment") != CPU_THREAD_ENVIRONMENT
                or row.get("torch_num_threads") != 1
                or row.get("torch_num_interop_threads") != 1
                or row.get("source_broadcast_calls") != 1
                or row.get("operational_path") != expected_operational[0]
                or row.get("operational_aux_gate_count") != expected_operational[1]
                or row.get("operational_projection_gate_count")
                != expected_operational[2]
                or row.get("operational_wrapper_trace_steps")
                != expected_operational[3]
                or row.get("aux_broadcast_calls") != expected_operational[4]
                or row.get("active_arm") != expected_arm
                or row.get("row_count") != expected_count
                or row.get("failure_stage") != expected_failure_stage[name]
                or row.get("consensus_failed")
                is (name not in {
                    "hostile_rank0_tensor", "hostile_rank2_tensor"
                })
                or row.get("stage_gate_failed")
                is (name not in {
                    "hostile_rank0_aux", "hostile_rank2_abi",
                    "hostile_rank1_row_build",
                    "hostile_rank3_final_scheduler",
                })
                or row.get("publication_empty") is not True
                or row.get("runtime_identity_digest")
                != digest({
                    key: runtime_identities[key]
                    for key in ("python", *TORCH_SOURCE_ARGUMENTS)
                })
                or claimed != digest(unsigned)
            ):
                raise World4ProbeError(f"rank row closure differs: {name}/r{rank_index}")
        if name == "happy" and any(
            row["trace_steps"] != STEPS or row["scheduler_calls"] != STEPS
            or row["scheduler_token_count"] != PACKED_TOKENS
            for row in rows
        ):
            raise World4ProbeError("happy scenario lacks exact 40-step closure")
        if name == "hostile_rank3_final_scheduler" and any(
            row["trace_steps"] != (STEPS - 1 if index == 3 else STEPS)
            or row["scheduler_calls"] != (STEPS - 1 if index == 3 else STEPS)
            or row["scheduler_token_count"] != PACKED_TOKENS
            for index, row in enumerate(rows)
        ):
            raise World4ProbeError("final scheduler failpoint trace differs")
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "ADMITTED_WORLD4_TENSOR_ABI_HOLD_ONLY",
        "launch_allowed": False,
        "scenario_order": list(SCENARIOS),
        "scenarios": results,
        "runtime_identities": runtime_identities,
        "runtime_identity_digest": digest(runtime_identities),
        "expected_runtime_versions": _expected_runtime_versions(args),
        "expected_gpu_contract": _expected_gpu_spec(args),
        "cpu_thread_contract": {
            "environment": dict(CPU_THREAD_ENVIRONMENT),
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
        },
        "active_row_counts_admitted": [2, 3],
        "happy_scheduler_steps": 40,
        "real_torchrun_process_count_per_scenario": 4,
        "controller_python_optimize_level": sys.flags.optimize,
        "timeout_seconds_per_scenario": SCENARIO_TIMEOUT_SECONDS,
        "timeout_cleanup_policy": "new_session_sigterm_then_sigkill_bounded_reap",
        "publication_performed": False,
        "renderer_or_vae_loaded": False,
        "scope": "distributed_tensor_projection_abi_not_renderer_integration",
    }
    result["receipt_digest"] = digest(result)
    if args.output:
        target = Path(args.output)
        _receipt_target_fresh(args.output)
        raw = canonical(result) + b"\n"
        fd = os.open(
            target, os.O_RDWR | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0,
        )
        try:
            offset = 0
            while offset < len(raw):
                count = os.write(fd, raw[offset:])
                if count <= 0:
                    raise World4ProbeError("world4 receipt write made no progress")
                offset += count
            os.fsync(fd)
            os.fchmod(fd, 0o400)
            os.fsync(fd)
            before = os.fstat(fd); named = os.lstat(target)
            replay = os.pread(fd, len(raw), 0)
        finally:
            os.close(fd)
        if (
            replay != raw or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o400
            or _identity(before) != _identity(named)
        ):
            raise World4ProbeError("world4 receipt replay differs")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    for name in ("wrapper", "projection", "scaffold-module", "scaffold",
                 "publication-root"):
        worker_parser.add_argument("--" + name, required=True)
    worker_parser.add_argument("--python", required=True)
    worker_parser.add_argument("--python-sha256", required=True)
    worker_parser.add_argument("--expected-torch-version", required=True)
    worker_parser.add_argument("--expected-hip-version", required=True)
    worker_parser.add_argument("--expected-gpu-count", required=True, type=int)
    for key in GPU_VISIBILITY_KEYS:
        worker_parser.add_argument(
            "--expected-" + key.lower().replace("_", "-"), required=True,
        )
    for role in TORCH_SOURCE_ARGUMENTS:
        option = role.replace("_", "-")
        worker_parser.add_argument("--" + option, required=True)
        worker_parser.add_argument("--" + option + "-sha256", required=True)
    run = sub.add_parser("run")
    run.add_argument("--python", required=True)
    run.add_argument("--python-sha256", required=True)
    run.add_argument("--expected-torch-version", required=True)
    run.add_argument("--expected-hip-version", required=True)
    run.add_argument("--expected-gpu-count", required=True, type=int)
    for key in GPU_VISIBILITY_KEYS:
        run.add_argument(
            "--expected-" + key.lower().replace("_", "-"), required=True,
        )
    for name in ("wrapper", "projection", "scaffold-module", "scaffold",
                 "publication-root"):
        run.add_argument("--" + name, required=True)
    for role in TORCH_SOURCE_ARGUMENTS:
        option = role.replace("_", "-")
        run.add_argument("--" + option, required=True)
        run.add_argument("--" + option + "-sha256", required=True)
    run.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = worker(args) if args.command == "worker" else controller(args)
    except (OSError, ValueError, KeyError, ImportError, World4ProbeError) as error:
        print(f"world4 probe refused: {error}", file=sys.stderr)
        return 88 if str(error).startswith("HOLD:") else 96
    if args.command == "run":
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

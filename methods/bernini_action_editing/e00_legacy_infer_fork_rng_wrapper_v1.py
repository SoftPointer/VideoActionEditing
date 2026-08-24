#!/usr/bin/env python3
"""Run one immutable legacy E00 arm under per-rank RNG restoration.

This wrapper does not make the old ABI anchor-free.  It records that the
target process reads/decodes the anchor, captures every raw keyed-noise hash,
and restores the caller's CPU plus rank-owned CUDA RNG state around the whole
legacy inference entrypoint.  The native receipt remains the authority for
the frozen/no-adapter certificate and the attention data flow.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence


SCHEMA = "bernini-e00-legacy-infer-fork-rng-audit-v1"
ARM_TRANSPORTS = {
    "pure_noobserver_output_routeoff": "self_target_owned_temporal_kernel_attn_output_v14r2",
    "observer_matched_output_routeoff": "self_target_owned_temporal_kernel_attn_output_v14r2",
    "old_pureqk_temporal_routeon": "self_target_owned_temporal_kernel_attn_output_v14r2",
}


class E00LegacyWrapperError(RuntimeError):
    pass


def _state_sha256(state: Any) -> str:
    values = state.detach().cpu().contiguous().reshape(-1).tolist()
    return hashlib.sha256(bytes(int(value) for value in values)).hexdigest()


def _snapshot_rng(torch: Any, cuda_device: Optional[int]) -> dict[str, Any]:
    return {
        "cpu_sha256": _state_sha256(torch.random.get_rng_state()),
        "cuda_sha256": (
            _state_sha256(torch.cuda.get_rng_state(cuda_device))
            if cuda_device is not None
            else None
        ),
    }


def run_with_rng_fork(
    torch: Any,
    callback: Callable[[], Any],
    *,
    cuda_device: Optional[int],
) -> tuple[Any, dict[str, Any]]:
    """Run ``callback`` and prove that global RNG state is restored."""

    before = _snapshot_rng(torch, cuda_device)
    devices = [] if cuda_device is None else [cuda_device]
    with torch.random.fork_rng(devices=devices, enabled=True):
        result = callback()
    after = _snapshot_rng(torch, cuda_device)
    proof = {
        "enabled": True,
        "scope": "entire_legacy_inference_entrypoint_per_rank",
        "owned_cuda_device": cuda_device,
        "before": before,
        "after": after,
        "cpu_state_restored": before["cpu_sha256"] == after["cpu_sha256"],
        "owned_cuda_state_restored": before["cuda_sha256"] == after["cuda_sha256"],
    }
    if not proof["cpu_state_restored"] or not proof["owned_cuda_state_restored"]:
        raise E00LegacyWrapperError("fork_rng did not restore the rank RNG state")
    return result, proof


def _flag(argv: Sequence[str], name: str) -> Optional[str]:
    positions = [index for index, value in enumerate(argv) if value == name]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise E00LegacyWrapperError(f"legacy argument {name} is duplicated or valueless")
    return argv[positions[0] + 1]


def validate_legacy_argv(argv: Sequence[str], *, arm_role: str) -> dict[str, str]:
    if arm_role not in ARM_TRANSPORTS:
        raise E00LegacyWrapperError("unsupported diagnostic arm role")
    transport = _flag(argv, "--transport")
    steps = _flag(argv, "--transport-steps")
    arm = _flag(argv, "--arm")
    initial_noise = _flag(argv, "--initial-noise-proposal-mode")
    anchor_state = _flag(argv, "--anchor-state-mode")
    source = _flag(argv, "--source-video")
    anchor = _flag(argv, "--anchor-video")
    output = _flag(argv, "--output")
    no_observer = arm_role == "pure_noobserver_output_routeoff"
    expected = {
        "transport": ARM_TRANSPORTS[arm_role],
        "steps": "0" if no_observer else "40",
        "arm": "AQK_IID1",
        "initial_noise": "keyed_only",
        "anchor_state": "clean_noised",
    }
    actual = {
        "transport": transport,
        "steps": steps,
        "arm": arm,
        "initial_noise": initial_noise,
        "anchor_state": anchor_state,
    }
    if actual != expected:
        raise E00LegacyWrapperError(
            f"legacy diagnostic arguments differ: {actual!r} != {expected!r}"
        )
    if not source or not anchor or not output:
        raise E00LegacyWrapperError("source, anchor, and output paths are required by legacy ABI")
    if no_observer and source == anchor:
        raise E00LegacyWrapperError(
            "pure no-observer arm must use a distinct pinned source-static placeholder"
        )
    if not no_observer and source == anchor:
        raise E00LegacyWrapperError("observer arm must read the registered action anchor")
    if _flag(argv, "--anchor-initial-gaussian") is not None:
        raise E00LegacyWrapperError("fresh keyed-only diagnostic forbids anchor Gaussian")
    if _flag(argv, "--trained-attention-checkpoint") is not None:
        raise E00LegacyWrapperError("zero-update diagnostic forbids a trained adapter")
    if "--no-initial-phase-clamp" in argv:
        raise E00LegacyWrapperError("corrected E00 diagnostic requires the initial phase clamp")
    return {"source": source, "anchor": anchor, "output": output, **expected}


@contextmanager
def capture_keyed_noise_rows() -> Iterator[list[dict[str, Any]]]:
    """Wrap the pinned draw function without changing its returned tensors."""

    guided = importlib.import_module("guided_source_aligned_controller")
    original = guided._draw_keyed_packed_noise
    rows: list[dict[str, Any]] = []

    def audited(*args: Any, **kwargs: Any) -> Any:
        value, digest = original(*args, **kwargs)
        try:
            seed = int(kwargs["seed"])
            step = int(kwargs["step"])
            candidate = int(kwargs["candidate"])
        except (KeyError, TypeError, ValueError) as error:
            raise E00LegacyWrapperError("keyed-noise call coordinates are unreadable") from error
        rows.append(
            {
                "master_seed": seed,
                "step": step,
                "candidate": candidate,
                "derived_seed": int(guided.keyed_noise_seed(seed, step, candidate)),
                "raw_noise_sha256": str(digest),
            }
        )
        return value, digest

    guided._draw_keyed_packed_noise = audited
    try:
        yield rows
    finally:
        guided._draw_keyed_packed_noise = original


@contextmanager
def audit_route_application(*, mode: str) -> Iterator[dict[str, Any]]:
    """Gate only the final pure-QK route addition, leaving observer Q/K intact."""

    transport = importlib.import_module("anchor_qk_transport")
    name = "_qk_only_temporal_kernel_contrast_output"
    original = getattr(transport, name, None)
    if not callable(original):
        raise E00LegacyWrapperError("pinned pure-QK route function is absent")
    if mode not in ("inactive_noobserver", "identity_observer", "enabled"):
        raise E00LegacyWrapperError("unknown route-application gate mode")
    enabled = mode == "enabled"
    identity_gate = mode == "identity_observer"
    audit = {
        "mode": mode,
        "enabled": enabled,
        "exact_identity_gate": identity_gate,
        "function": name,
        "call_count": 0,
        "observer_qk_capture_and_consumption_unchanged": True,
    }

    def routed(*args: Any, **kwargs: Any) -> Any:
        audit["call_count"] += 1
        if enabled:
            return original(*args, **kwargs)
        if mode == "inactive_noobserver":
            raise E00LegacyWrapperError("pure no-observer arm unexpectedly called route application")
        current_output = args[0] if args else kwargs.get("current_output")
        if current_output is None:
            raise E00LegacyWrapperError("pure-QK route call has no current output")
        # Exact tensor identity: no clone, arithmetic, donor access, or dtype
        # conversion is allowed after the same Q/K entries have been consumed.
        return current_output

    setattr(transport, name, routed)
    try:
        yield audit
    finally:
        setattr(transport, name, original)


@contextmanager
def capture_predecode_latent() -> Iterator[dict[str, Any]]:
    """Hash the controller result before VAE decode without changing it."""

    controller = importlib.import_module("anchor_sga_anc_controller")
    original = controller.sample_anchor_sga_anc
    audit: dict[str, Any] = {}

    def captured(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        latent = result[0] if isinstance(result, tuple) else result
        if not hasattr(latent, "detach"):
            raise E00LegacyWrapperError("controller result is not a tensor")
        cpu = latent.detach().cpu().contiguous()
        try:
            raw = cpu.numpy().tobytes(order="C")
        except (TypeError, RuntimeError) as error:
            raise E00LegacyWrapperError("predecode latent cannot expose canonical bytes") from error
        audit.update(
            {
                "raw_storage_sha256": hashlib.sha256(raw).hexdigest(),
                "dtype": str(cpu.dtype),
                "shape": [int(value) for value in cpu.shape],
                "finite": bool(cpu.isfinite().all().item()),
            }
        )
        return result

    controller.sample_anchor_sga_anc = captured
    try:
        yield audit
    finally:
        controller.sample_anchor_sga_anc = original


def _native_adapter_proof(output: str) -> Optional[dict[str, Any]]:
    receipt_path = Path(output + ".receipt.json")
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return None
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    before = value.get("freeze_before")
    after = value.get("freeze_after")
    if not isinstance(before, Mapping) or before != after:
        raise E00LegacyWrapperError("native frozen certificate is absent or changed")
    proof = {
        "native_receipt_path": str(receipt_path),
        "native_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "freeze_before_equals_after": True,
        "base_frozen": before.get("base_frozen"),
        "trainable_parameter_tensors": before.get("trainable_parameter_tensors"),
        "trainable_parameter_elements": before.get("trainable_parameter_elements"),
        "lora_module_count": before.get("lora_module_count"),
        "adapter_modules_absent": before.get("adapter_modules_absent"),
    }
    expected = {
        "base_frozen": True,
        "trainable_parameter_tensors": 0,
        "trainable_parameter_elements": 0,
        "lora_module_count": 0,
        "adapter_modules_absent": True,
    }
    if any(proof[key] != expected[key] for key in expected):
        raise E00LegacyWrapperError("native receipt does not prove adapter-off frozen base")
    return proof


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise E00LegacyWrapperError(f"refusing to overwrite wrapper audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-pid-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise E00LegacyWrapperError(f"wrapper temporary already exists: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-arm-role", required=True, choices=tuple(ARM_TRANSPORTS))
    parser.add_argument("--wrapper-audit-prefix", required=True)
    parser.add_argument("--legacy-entrypoint", default="infer_anchor_sga_anc_event_v1")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, legacy_argv = build_parser().parse_known_args(argv)
    if legacy_argv and legacy_argv[0] == "--":
        legacy_argv = legacy_argv[1:]
    contract = validate_legacy_argv(legacy_argv, arm_role=args.wrapper_arm_role)

    import torch

    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 4 or rank not in range(4) or local_rank not in range(4):
        raise E00LegacyWrapperError("production diagnostic requires exactly four ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise E00LegacyWrapperError("production diagnostic requires four visible GPUs")
    torch.cuda.set_device(local_rank)

    entrypoint = importlib.import_module(args.legacy_entrypoint)
    if not callable(getattr(entrypoint, "main", None)):
        raise E00LegacyWrapperError("legacy entrypoint has no callable main")
    route_mode = {
        "pure_noobserver_output_routeoff": "inactive_noobserver",
        "observer_matched_output_routeoff": "identity_observer",
        "old_pureqk_temporal_routeon": "enabled",
    }[args.wrapper_arm_role]
    with capture_keyed_noise_rows() as noise_rows, audit_route_application(
        mode=route_mode
    ) as route_audit, capture_predecode_latent() as latent_audit:
        return_code, rng_proof = run_with_rng_fork(
            torch,
            lambda: entrypoint.main(legacy_argv),
            cuda_device=local_rank,
        )
    if return_code not in (None, 0):
        raise E00LegacyWrapperError(f"legacy entrypoint returned {return_code!r}")
    expected_rows = [
        (2027, step, 0) for step in range(40)
    ]
    actual_rows = [
        (row["master_seed"], row["step"], row["candidate"]) for row in noise_rows
    ]
    if actual_rows != expected_rows:
        raise E00LegacyWrapperError("runtime keyed-noise call closure differs")
    expected_route_calls = (
        0 if args.wrapper_arm_role == "pure_noobserver_output_routeoff" else 2 * 40 * 22
    )
    if route_audit["call_count"] != expected_route_calls:
        raise E00LegacyWrapperError("pure-QK route-application call closure differs")
    if not latent_audit or latent_audit.get("finite") is not True:
        raise E00LegacyWrapperError("predecode latent audit is absent or non-finite")
    native_proof = _native_adapter_proof(contract["output"]) if rank == 0 else None
    if rank == 0 and native_proof is None:
        raise E00LegacyWrapperError("rank 0 native receipt is absent")
    receipt = {
        "schema_version": SCHEMA,
        "complete": True,
        "arm_role": args.wrapper_arm_role,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "fork_rng": rng_proof,
        "legacy_abi": {
            "offline_anchor_graph_split_supported": False,
            "target_process_reads_anchor_video_path": True,
            "target_process_decodes_anchor_video": True,
            "self_generated_anchor_video_read": (
                args.wrapper_arm_role != "pure_noobserver_output_routeoff"
            ),
            "required_anchor_path_is_source_placeholder": (
                False
            ),
            "required_anchor_path_is_source_frame0_static_placeholder": (
                args.wrapper_arm_role == "pure_noobserver_output_routeoff"
            ),
            "output_path": contract["output"],
            "anchor_path_sha256": hashlib.sha256(contract["anchor"].encode("utf-8")).hexdigest(),
            "honest_control_name": (
                "pure no-observer route-off; pinned source-frame0 static legacy placeholder"
                if args.wrapper_arm_role == "pure_noobserver_output_routeoff"
                else "observer-matched output-route-off K0; not anchor-free"
                if args.wrapper_arm_role == "observer_matched_output_routeoff"
                else "old dfix2 pure-QK temporal route-on; not v15b"
            ),
        },
        "runtime_noise": {
            "scheme": "sha256_keyed_cpu_torch_generator_v1",
            "master_seed": 2027,
            "rows": noise_rows,
        },
        "route_application": route_audit,
        "predecode_latent": latent_audit,
        "native_adapter_off_proof_rank0": native_proof,
    }
    output = Path(f"{args.wrapper_audit_prefix}.rank{rank}.json")
    _write_atomic(output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

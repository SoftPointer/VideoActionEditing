#!/usr/bin/env python3
"""Run one E00 legacy arm from an explicit, cross-arm RNG initial state.

This is an independent diagnostic wrapper.  It does not modify the pinned
legacy inference/controller/transport sources.  Each distributed rank enters
``fork_rng``, explicitly seeds the CPU generator and only its rank-owned CUDA
generator, records the resulting state bytes, and then calls the complete
legacy inference entrypoint.  The caller RNG bytes are restored on exit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import e00_legacy_infer_fork_rng_wrapper_v1 as legacy_audit


SCHEMA = "bernini-e00-legacy-infer-fixed-initial-rng-audit-v2"
REVISION_TAG = "E00_DFIX2_CLEAN_DIAG_R2_FIXED_RNG_TWO_PHASE_20260821"
EXPECTED_LEGACY_AUDIT_SHA256 = (
    "42cb90d2e05ce7f14d7b44e1b930e2133564343d5ef8492367a8af87b882d5c7"
)


class E00FixedRNGWrapperError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_rng(torch: Any, cuda_device: Optional[int]) -> dict[str, Any]:
    def state_sha256(value: Any) -> str:
        raw = value.detach().cpu().contiguous().numpy().tobytes(order="C")
        return hashlib.sha256(raw).hexdigest()

    return {
        "cpu_sha256": state_sha256(torch.random.get_rng_state()),
        "cuda_sha256": (
            state_sha256(torch.cuda.get_rng_state(cuda_device))
            if cuda_device is not None
            else None
        ),
    }


def run_with_fixed_initial_rng(
    torch: Any,
    callback: Callable[[], Any],
    *,
    cuda_device: Optional[int],
    cpu_seed: int,
    cuda_seed: Optional[int],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Execute one callback from fixed per-rank CPU/CUDA generator states."""

    if isinstance(cpu_seed, bool) or not isinstance(cpu_seed, int) or cpu_seed < 0:
        raise E00FixedRNGWrapperError("CPU seed must be a non-negative integer")
    if cuda_device is None:
        if cuda_seed is not None:
            raise E00FixedRNGWrapperError("CPU-only execution cannot accept a CUDA seed")
    elif isinstance(cuda_seed, bool) or not isinstance(cuda_seed, int) or cuda_seed < 0:
        raise E00FixedRNGWrapperError("CUDA execution requires a non-negative CUDA seed")

    caller_before = _snapshot_rng(torch, cuda_device)
    devices = [] if cuda_device is None else [cuda_device]
    with torch.random.fork_rng(devices=devices, enabled=True):
        # Seed the CPU generator directly; do not use torch.manual_seed because
        # it may seed CUDA generators outside the rank-owned device namespace.
        torch.random.default_generator.manual_seed(cpu_seed)
        if cuda_device is not None:
            torch.cuda.default_generators[cuda_device].manual_seed(cuda_seed)
        seeded_initial = _snapshot_rng(torch, cuda_device)
        result = callback()
        terminal_before_restore = _snapshot_rng(torch, cuda_device)
    caller_after = _snapshot_rng(torch, cuda_device)

    fork_proof = {
        "enabled": True,
        "scope": "entire_legacy_inference_entrypoint_per_rank",
        "owned_cuda_device": cuda_device,
        "before": caller_before,
        "after": caller_after,
        "cpu_state_restored": caller_before["cpu_sha256"]
        == caller_after["cpu_sha256"],
        "owned_cuda_state_restored": caller_before["cuda_sha256"]
        == caller_after["cuda_sha256"],
    }
    fixed_proof = {
        "enabled": True,
        "scheme": "explicit_rank_owned_cpu_cuda_manual_seed_v2",
        "scope": (
            "inside_fork_rng_immediately_before_entire_legacy_inference_entrypoint"
        ),
        "cpu_seed": cpu_seed,
        "cuda_seed": cuda_seed,
        "seeded_initial": seeded_initial,
        "terminal_before_restore": terminal_before_restore,
    }
    if (
        fork_proof["cpu_state_restored"] is not True
        or fork_proof["owned_cuda_state_restored"] is not True
    ):
        raise E00FixedRNGWrapperError("fork_rng did not restore caller RNG bytes")
    return result, fork_proof, fixed_proof


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wrapper-arm-role",
        required=True,
        choices=tuple(legacy_audit.ARM_TRANSPORTS),
    )
    parser.add_argument("--wrapper-audit-prefix", required=True)
    parser.add_argument("--wrapper-protocol", required=True)
    parser.add_argument("--legacy-entrypoint", default="infer_anchor_sga_anc_event_v1")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, legacy_argv = build_parser().parse_known_args(argv)
    if legacy_argv and legacy_argv[0] == "--":
        legacy_argv = legacy_argv[1:]
    contract = legacy_audit.validate_legacy_argv(
        legacy_argv, arm_role=args.wrapper_arm_role
    )

    legacy_path = Path(legacy_audit.__file__).resolve()
    if file_sha256(legacy_path) != EXPECTED_LEGACY_AUDIT_SHA256:
        raise E00FixedRNGWrapperError("sealed legacy audit wrapper bytes differ")

    protocol_validator = importlib.import_module(
        "validate_e00_three_vessel_fresh_keyed_two_phase_diagnostic_v2"
    )
    protocol_path = Path(args.wrapper_protocol)
    protocol = protocol_validator.load_protocol(protocol_path)
    protocol_identity = protocol_validator.protocol_identity(protocol, protocol_path)
    if protocol.get("revision_tag") != REVISION_TAG:
        raise E00FixedRNGWrapperError("protocol revision differs")

    import torch

    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 4 or rank not in range(4) or local_rank not in range(4):
        raise E00FixedRNGWrapperError("production diagnostic requires exactly four ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise E00FixedRNGWrapperError("production diagnostic requires four visible GPUs")
    torch.cuda.set_device(local_rank)

    seed_rows = protocol["fixed_initial_rng"]["per_rank"]
    selected = [row for row in seed_rows if row.get("rank") == rank]
    if len(selected) != 1:
        raise E00FixedRNGWrapperError("rank seed lookup differs")
    seed_row = selected[0]

    entrypoint = importlib.import_module(args.legacy_entrypoint)
    if not callable(getattr(entrypoint, "main", None)):
        raise E00FixedRNGWrapperError("legacy entrypoint has no callable main")
    route_mode = {
        "pure_noobserver_output_routeoff": "inactive_noobserver",
        "observer_matched_output_routeoff": "identity_observer",
        "old_pureqk_temporal_routeon": "enabled",
    }[args.wrapper_arm_role]

    with legacy_audit.capture_keyed_noise_rows() as noise_rows, legacy_audit.audit_route_application(
        mode=route_mode
    ) as route_audit, legacy_audit.capture_predecode_latent() as latent_audit:
        return_code, fork_proof, fixed_proof = run_with_fixed_initial_rng(
            torch,
            lambda: entrypoint.main(legacy_argv),
            cuda_device=local_rank,
            cpu_seed=int(seed_row["cpu_seed"]),
            cuda_seed=int(seed_row["cuda_seed"]),
        )
    if return_code not in (None, 0):
        raise E00FixedRNGWrapperError(
            f"legacy entrypoint returned {return_code!r}"
        )
    expected_rows = [(2027, step, 0) for step in range(40)]
    actual_rows = [
        (row["master_seed"], row["step"], row["candidate"])
        for row in noise_rows
    ]
    if actual_rows != expected_rows:
        raise E00FixedRNGWrapperError("runtime keyed-noise call closure differs")
    expected_route_calls = (
        0
        if args.wrapper_arm_role == "pure_noobserver_output_routeoff"
        else 2 * 40 * 22
    )
    if route_audit["call_count"] != expected_route_calls:
        raise E00FixedRNGWrapperError("pure-QK route call closure differs")
    if not latent_audit or latent_audit.get("finite") is not True:
        raise E00FixedRNGWrapperError("predecode latent audit is absent or non-finite")

    native_proof = (
        legacy_audit._native_adapter_proof(contract["output"]) if rank == 0 else None
    )
    if rank == 0 and native_proof is None:
        raise E00FixedRNGWrapperError("rank-zero native receipt is absent")
    receipt: Mapping[str, Any] = {
        "schema_version": SCHEMA,
        "revision_tag": REVISION_TAG,
        "complete": True,
        "arm_role": args.wrapper_arm_role,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "protocol": protocol_identity,
        "fork_rng": fork_proof,
        "fixed_initial_rng": fixed_proof,
        "legacy_abi": {
            "offline_anchor_graph_split_supported": False,
            "target_process_reads_anchor_video_path": True,
            "target_process_decodes_anchor_video": True,
            "self_generated_anchor_video_read": (
                args.wrapper_arm_role != "pure_noobserver_output_routeoff"
            ),
            "required_anchor_path_is_source_placeholder": False,
            "required_anchor_path_is_source_frame0_static_placeholder": (
                args.wrapper_arm_role == "pure_noobserver_output_routeoff"
            ),
            "output_path": contract["output"],
            "anchor_path_sha256": hashlib.sha256(
                contract["anchor"].encode("utf-8")
            ).hexdigest(),
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
    legacy_audit._write_atomic(output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

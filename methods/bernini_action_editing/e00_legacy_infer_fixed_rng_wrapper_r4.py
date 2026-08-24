#!/usr/bin/env python3
"""R4 receipt wrapper for one fixed-RNG E00 legacy inference arm."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import e00_legacy_infer_fork_rng_wrapper_v1 as legacy_audit
import e00_legacy_infer_fixed_rng_wrapper_r3 as fixed_rng_core


SCHEMA = "bernini-e00-legacy-fixed-rng-r4-receipt-v4"
REVISION_TAG = "E00_DFIX2_CLEAN_DIAG_R4_OVERLAY_CACHE_CLOSURE_20260821"
EXPECTED_LEGACY_AUDIT_SHA256 = "42cb90d2e05ce7f14d7b44e1b930e2133564343d5ef8492367a8af87b882d5c7"


class E00R4WrapperError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


run_with_fixed_initial_rng = fixed_rng_core.run_with_fixed_initial_rng


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-arm-role", required=True, choices=tuple(legacy_audit.ARM_TRANSPORTS))
    parser.add_argument("--wrapper-audit-prefix", required=True)
    parser.add_argument("--wrapper-protocol", required=True)
    parser.add_argument("--legacy-entrypoint", default="infer_anchor_sga_anc_event_v1")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, legacy_argv = build_parser().parse_known_args(argv)
    if legacy_argv and legacy_argv[0] == "--":
        legacy_argv = legacy_argv[1:]
    contract = legacy_audit.validate_legacy_argv(legacy_argv, arm_role=args.wrapper_arm_role)
    if file_sha256(Path(legacy_audit.__file__).resolve()) != EXPECTED_LEGACY_AUDIT_SHA256:
        raise E00R4WrapperError("sealed legacy audit wrapper bytes differ")
    protocol_validator = importlib.import_module("validate_e00_three_vessel_clean_diag_r4")
    protocol_path = Path(args.wrapper_protocol)
    protocol = protocol_validator.load_protocol(protocol_path)
    protocol_id = protocol_validator.protocol_identity(protocol, protocol_path)
    if protocol.get("revision_tag") != REVISION_TAG:
        raise E00R4WrapperError("R4 protocol revision differs")

    import torch

    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 4 or rank not in range(4) or local_rank not in range(4):
        raise E00R4WrapperError("R4 diagnostic requires exactly four ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise E00R4WrapperError("R4 diagnostic requires four visible GPUs")
    torch.cuda.set_device(local_rank)
    seed_rows = [row for row in protocol["fixed_initial_rng"]["per_rank"] if row.get("rank") == rank]
    if len(seed_rows) != 1:
        raise E00R4WrapperError("R4 rank seed lookup differs")
    seed_row = seed_rows[0]
    entrypoint = importlib.import_module(args.legacy_entrypoint)
    if not callable(getattr(entrypoint, "main", None)):
        raise E00R4WrapperError("legacy entrypoint has no callable main")
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
        raise E00R4WrapperError(f"legacy entrypoint returned {return_code!r}")
    if [(row["master_seed"], row["step"], row["candidate"]) for row in noise_rows] != [(2027, step, 0) for step in range(40)]:
        raise E00R4WrapperError("runtime keyed-noise call closure differs")
    expected_route_calls = 0 if args.wrapper_arm_role == "pure_noobserver_output_routeoff" else 2 * 40 * 22
    if route_audit["call_count"] != expected_route_calls:
        raise E00R4WrapperError("pure-QK route call closure differs")
    if not latent_audit or latent_audit.get("finite") is not True:
        raise E00R4WrapperError("predecode latent audit is absent or non-finite")
    native_proof = legacy_audit._native_adapter_proof(contract["output"]) if rank == 0 else None
    if rank == 0 and native_proof is None:
        raise E00R4WrapperError("rank-zero native receipt is absent")
    receipt: Mapping[str, Any] = {
        "schema_version": SCHEMA,
        "revision_tag": REVISION_TAG,
        "complete": True,
        "arm_role": args.wrapper_arm_role,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "protocol": protocol_id,
        "fork_rng": fork_proof,
        "fixed_initial_rng": fixed_proof,
        "legacy_abi": {
            "offline_anchor_graph_split_supported": False,
            "target_process_reads_anchor_video_path": True,
            "target_process_decodes_anchor_video": True,
            "self_generated_anchor_video_read": args.wrapper_arm_role != "pure_noobserver_output_routeoff",
            "required_anchor_path_is_source_placeholder": False,
            "required_anchor_path_is_source_frame0_static_placeholder": args.wrapper_arm_role == "pure_noobserver_output_routeoff",
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
        "runtime_noise": {"scheme": "sha256_keyed_cpu_torch_generator_v1", "master_seed": 2027, "rows": noise_rows},
        "route_application": route_audit,
        "predecode_latent": latent_audit,
        "native_adapter_off_proof_rank0": native_proof,
    }
    legacy_audit._write_atomic(Path(f"{args.wrapper_audit_prefix}.rank{rank}.json"), receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

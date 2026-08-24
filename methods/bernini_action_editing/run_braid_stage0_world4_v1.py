#!/usr/bin/env python3
"""Run the first BRAID Stage-0 structural canary on real WORLD4/SP4.

This adapter intentionally implements exactly one preregistered arm:
``parity-reset-off-reference-4f-a``.  It authenticates the frozen Bernini
source/checkpoint, Q-MOSAIC owner and signed editor packet, reconstructs the
pinned negative condition, and executes one exact81/exact40 source-video-only
``v2v_apg`` sample under ``BraidDualNativeAPGRuntimePatch``.

The action and no-op conditions are the same exact c0 tensor object.  Thus the
arm can establish four-forward dual-APG parity and fresh state separation, but
it cannot establish action capacity or decoded quality.  It consumes no image
references, performs no decode/backward/optimizer/update/checkpoint write, and
emits one create-only engineering receipt only after all four ranks agree.
"""

from __future__ import annotations

import argparse
import base64
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import braid_stage0_all8_orchestrator_v1 as stage0  # noqa: E402


SUPPORTED_ARM_ID = "parity-reset-off-reference-4f-a"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
CHECKPOINT_CONTENT_FILE_COUNT = 23
WORLD4_RECEIPT_FILENAME = "world4.receipt.json"
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class BraidStage0World4Error(RuntimeError):
    """An authenticated input, real forward, or receipt boundary differed."""


def _sha1(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA1.fullmatch(value) is None:
        raise BraidStage0World4Error(f"{label} must be full lowercase SHA-1")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise BraidStage0World4Error(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise BraidStage0World4Error(f"{label} must be one canonical absolute plain file")
    return path


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or not path.is_dir()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise BraidStage0World4Error(
            f"{label} must be one canonical absolute plain directory"
        )
    return path


def _checked_file(
    value: str | Path, *, expected_sha256: str, label: str
) -> Path:
    path = _plain_file(value, label=label)
    expected = _sha256(expected_sha256, label=f"{label} expected SHA-256")
    if stage0.file_sha256(path) != expected:
        raise BraidStage0World4Error(f"{label} SHA-256 differs")
    return path


def _fresh_output(value: str | Path, *, expected: Path) -> Path:
    requested = Path(value)
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or requested != expected
        or requested != requested.resolve(strict=False)
        or requested.exists()
        or requested.is_symlink()
        or not requested.parent.is_dir()
        or requested.parent.is_symlink()
        or requested.parent.resolve(strict=True) != requested.parent
    ):
        raise BraidStage0World4Error(
            "output directory must be the fresh plan-bound cell/arm directory"
        )
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-world4", help="run the sole implemented WORLD4 arm")
    run.add_argument("--plan", required=True)
    run.add_argument("--expected-plan-file-sha256", required=True)
    run.add_argument("--cell-id", choices=tuple(stage0.CELL_BY_ID), required=True)
    run.add_argument("--query-seed", type=int, required=True)
    run.add_argument("--arm-id", choices=tuple(stage0.ARM_BY_ID), required=True)
    run.add_argument("--query-registry", required=True)
    run.add_argument("--expected-query-registry-sha256", required=True)
    run.add_argument("--braid-arm-registry", required=True)
    run.add_argument("--expected-braid-arm-registry-sha256", required=True)
    run.add_argument("--dual-runtime-source", required=True)
    run.add_argument("--expected-dual-runtime-source-sha256", required=True)
    run.add_argument("--bernini-root", required=True)
    run.add_argument("--veomni-root", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--checkpoint-content-manifest", required=True)
    run.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    run.add_argument("--expected-checkpoint-tree-sha256", required=True)
    run.add_argument("--expected-bernini-commit", required=True)
    run.add_argument("--expected-veomni-commit", required=True)
    run.add_argument("--owner-root", required=True)
    run.add_argument("--owner-master-receipt", required=True)
    run.add_argument("--expected-owner-master-receipt-sha256", required=True)
    run.add_argument("--owner-audit-sidecar", required=True)
    run.add_argument("--expected-owner-audit-sidecar-sha256", required=True)
    run.add_argument("--owner-audit-evidence", required=True)
    run.add_argument("--owner-audit-public-key", required=True)
    run.add_argument("--expected-owner-audit-public-key-sha256", required=True)
    run.add_argument("--owner-cell-root", required=True)
    run.add_argument("--owner-cell-receipt", required=True)
    run.add_argument("--expected-owner-cell-receipt-sha256", required=True)
    run.add_argument("--editor-receipt", required=True)
    run.add_argument("--expected-editor-receipt-sha256", required=True)
    run.add_argument("--editor-public-key", required=True)
    run.add_argument("--expected-editor-public-key-sha256", required=True)
    run.add_argument("--editor-artifact-root", required=True)
    run.add_argument("--execution-private-key", required=True)
    run.add_argument("--execution-public-key", required=True)
    run.add_argument("--expected-execution-public-key-sha256", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument(
        "--ack-forward-only-no-decode-backward-optimizer-update",
        action="store_true",
    )
    return parser


def validate_live_device_environment(
    cell: Mapping[str, Any], *, environment: Optional[Mapping[str, str]] = None
) -> dict[str, Any]:
    """Observe the exact torchrun/ROCm visibility contract before torch import."""

    live = os.environ if environment is None else environment
    expected_rocr = ",".join(str(item) for item in cell["visible_devices"])
    if live.get("ROCR_VISIBLE_DEVICES") != expected_rocr:
        raise BraidStage0World4Error(
            "live ROCR_VISIBLE_DEVICES differs from the plan cell"
        )
    polluted = [
        name
        for name in (
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
        )
        if name in live
    ]
    if polluted:
        raise BraidStage0World4Error(
            f"live device environment contains forbidden aliases: {polluted}"
        )

    parsed: dict[str, int] = {}
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        raw = live.get(name)
        if type(raw) is not str or not raw.isascii() or not raw.isdigit():
            raise BraidStage0World4Error(f"live torchrun {name} differs")
        value = int(raw)
        if str(value) != raw:
            raise BraidStage0World4Error(f"live torchrun {name} is not canonical")
        parsed[name] = value
    if (
        parsed["WORLD_SIZE"] != stage0.WORLD_SIZE
        or parsed["RANK"] not in range(stage0.WORLD_SIZE)
        or parsed["LOCAL_RANK"] != parsed["RANK"]
    ):
        raise BraidStage0World4Error("live torchrun rank topology differs")
    unsigned = {
        "schema_version": stage0.DEVICE_ENVIRONMENT_SCHEMA,
        "sp_rank": parsed["RANK"],
        "rank": parsed["RANK"],
        "local_rank": parsed["LOCAL_RANK"],
        "world_size": parsed["WORLD_SIZE"],
        "rocr_visible_devices": expected_rocr,
        "physical_visible_devices": list(cell["visible_devices"]),
        "hip_visible_devices_unset": True,
        "cuda_visible_devices_unset": True,
        "gpu_device_ordinal_unset": True,
        "observed_before_torch_import": True,
    }
    return {**unsigned, "environment_digest": stage0.object_sha256(unsigned)}


def _load_execution_signer(
    *,
    private_key_path: str | Path,
    public_key_path: str | Path,
    expected_public_key_file_sha256: str,
) -> Any:
    """Load one private job key and prove it matches the plan-bound public key."""

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as error:  # pragma: no cover
        raise BraidStage0World4Error(
            "cryptography Ed25519 support is required"
        ) from error
    private_path = _plain_file(private_key_path, label="execution private key")
    public_path = _checked_file(
        public_key_path,
        expected_sha256=expected_public_key_file_sha256,
        label="execution public key",
    )
    if stat.S_IMODE(private_path.stat().st_mode) & 0o077:
        raise BraidStage0World4Error(
            "execution private key must not be group/world accessible"
        )
    try:
        private = serialization.load_pem_private_key(
            private_path.read_bytes(), password=None
        )
        public = serialization.load_pem_public_key(public_path.read_bytes())
    except (OSError, ValueError, TypeError) as error:
        raise BraidStage0World4Error("execution key pair cannot be loaded") from error
    if not isinstance(private, Ed25519PrivateKey) or not isinstance(
        public, Ed25519PublicKey
    ):
        raise BraidStage0World4Error("execution key pair is not Ed25519")
    private_raw = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    public_raw = public.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    if private_raw != public_raw:
        raise BraidStage0World4Error("execution private/public keys do not match")
    return private


def _sign_world4_payload(
    payload: Mapping[str, Any], *, signer: Any, public_key_file_sha256: str
) -> dict[str, Any]:
    """Sign the already sealed, schema-closed WORLD4 payload on rank zero."""

    stage0.validate_sealed_receipt(
        dict(payload),
        schema=stage0.WORLD4_SCHEMA,
        required_keys=stage0._WORLD4_KEYS,
        label="rank-zero WORLD4 payload",
    )
    signature = signer.sign(stage0.canonical_json_bytes(dict(payload)))
    if type(signature) is not bytes or len(signature) != 64:
        raise BraidStage0World4Error("execution Ed25519 signature length differs")
    return {
        **dict(payload),
        "execution_signature_scheme": stage0.EXECUTION_SIGNATURE_SCHEME,
        "execution_public_key_file_sha256": _sha256(
            public_key_file_sha256,
            label="execution public key file",
        ),
        "execution_signature_ed25519_base64": base64.b64encode(signature).decode(
            "ascii"
        ),
    }


def validate_cli(args: argparse.Namespace) -> Mapping[str, Any]:
    """Close all caller-selectable coordinates before importing PyTorch."""

    if args.ack_forward_only_no_decode_backward_optimizer_update is not True:
        raise BraidStage0World4Error("explicit forward-only authority acknowledgement is required")
    if args.arm_id != SUPPORTED_ARM_ID:
        raise BraidStage0World4Error(
            "only parity-reset-off-reference-4f-a is implemented; all other arms fail closed"
        )
    if args.arm_id not in stage0.IMPLEMENTED_WORLD4_ARM_IDS:
        raise BraidStage0World4Error("runner/orchestrator implemented-arm closure differs")

    for name in ("expected_bernini_commit", "expected_veomni_commit"):
        _sha1(getattr(args, name), label=name)
    for name in (
        "expected_plan_file_sha256",
        "expected_query_registry_sha256",
        "expected_braid_arm_registry_sha256",
        "expected_dual_runtime_source_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_owner_master_receipt_sha256",
        "expected_owner_audit_sidecar_sha256",
        "expected_owner_audit_public_key_sha256",
        "expected_owner_cell_receipt_sha256",
        "expected_editor_receipt_sha256",
        "expected_editor_public_key_sha256",
        "expected_execution_public_key_sha256",
    ):
        _sha256(getattr(args, name), label=name)

    plan_path = _checked_file(
        args.plan,
        expected_sha256=args.expected_plan_file_sha256,
        label="Stage-0 plan",
    )
    plan = stage0.validate_plan(stage0.load_json(plan_path, label="Stage-0 plan"))
    cell = stage0.plan_cell(plan, args.cell_id)
    arm = stage0.ARM_BY_ID[args.arm_id]
    if args.query_seed != cell["query_seed"]:
        raise BraidStage0World4Error("cell/query seed differs from preregistration")

    provenance = plan["provenance"]
    if (
        args.expected_bernini_commit != stage0.PINNED_BERNINI_REVISION
        or args.expected_veomni_commit != stage0.PINNED_VEOMNI_REVISION
        or args.expected_bernini_commit != provenance["bernini_revision"]
        or args.expected_veomni_commit != provenance["veomni_revision"]
        or args.expected_query_registry_sha256
        != stage0.PINNED_QUERY_REGISTRY_SHA256
        or args.expected_query_registry_sha256
        != provenance["query_registry_sha256"]
        or args.expected_braid_arm_registry_sha256
        != stage0.PINNED_BRAID_ARM_REGISTRY_SHA256
        or args.expected_braid_arm_registry_sha256
        != provenance["braid_arm_registry_sha256"]
        or args.expected_checkpoint_content_manifest_sha256
        != stage0.PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != provenance["checkpoint_content_manifest_sha256"]
        or args.expected_checkpoint_tree_sha256 != CHECKPOINT_TREE_SHA256
        or args.expected_editor_public_key_sha256
        != stage0.PINNED_EDITOR_PUBLIC_KEY_SHA256
        or args.expected_editor_public_key_sha256
        != provenance["editor_public_key_file_sha256"]
        or args.expected_editor_receipt_sha256
        != cell["editor_receipt_file_sha256"]
        or args.expected_execution_public_key_sha256
        != plan["execution_authentication"]["public_key_file_sha256"]
    ):
        raise BraidStage0World4Error("pinned source/checkpoint/registry coordinate differs")

    query_registry = _checked_file(
        args.query_registry,
        expected_sha256=args.expected_query_registry_sha256,
        label="query registry",
    )
    arm_registry = _checked_file(
        args.braid_arm_registry,
        expected_sha256=args.expected_braid_arm_registry_sha256,
        label="BRAID arm registry",
    )
    runtime_source = _checked_file(
        args.dual_runtime_source,
        expected_sha256=args.expected_dual_runtime_source_sha256,
        label="dual runtime source",
    )
    if args.expected_dual_runtime_source_sha256 != provenance["runtime_source_sha256"]:
        raise BraidStage0World4Error("dual runtime source differs from sealed plan")
    running_source = Path(__file__).resolve(strict=True)
    if stage0.file_sha256(running_source) != provenance["runner_source_sha256"]:
        raise BraidStage0World4Error("running WORLD4 source differs from sealed plan")

    checkpoint_manifest = _checked_file(
        args.checkpoint_content_manifest,
        expected_sha256=args.expected_checkpoint_content_manifest_sha256,
        label="checkpoint content manifest",
    )
    owner_master = _checked_file(
        args.owner_master_receipt,
        expected_sha256=args.expected_owner_master_receipt_sha256,
        label="owner master receipt",
    )
    owner_sidecar = _checked_file(
        args.owner_audit_sidecar,
        expected_sha256=args.expected_owner_audit_sidecar_sha256,
        label="owner audit sidecar",
    )
    owner_public_key = _checked_file(
        args.owner_audit_public_key,
        expected_sha256=args.expected_owner_audit_public_key_sha256,
        label="owner audit public key",
    )
    owner_cell_receipt = _checked_file(
        args.owner_cell_receipt,
        expected_sha256=args.expected_owner_cell_receipt_sha256,
        label="owner cell receipt",
    )
    editor_receipt = _checked_file(
        args.editor_receipt,
        expected_sha256=args.expected_editor_receipt_sha256,
        label="editor receipt",
    )
    editor_public_key = _checked_file(
        args.editor_public_key,
        expected_sha256=args.expected_editor_public_key_sha256,
        label="editor public key",
    )
    expected_execution_public_key = stage0.execution_public_key_path(plan)
    if Path(args.execution_public_key) != expected_execution_public_key:
        raise BraidStage0World4Error(
            "execution public key path differs from the sealed plan"
        )
    execution_signer = _load_execution_signer(
        private_key_path=args.execution_private_key,
        public_key_path=args.execution_public_key,
        expected_public_key_file_sha256=(
            args.expected_execution_public_key_sha256
        ),
    )
    device_environment = validate_live_device_environment(cell)
    owner_evidence = _plain_file(args.owner_audit_evidence, label="owner audit evidence")
    roots = {
        "bernini_root": _plain_directory(args.bernini_root, label="Bernini root"),
        "veomni_root": _plain_directory(args.veomni_root, label="VeOmni root"),
        "checkpoint": _plain_directory(args.checkpoint, label="checkpoint"),
        "owner_root": _plain_directory(args.owner_root, label="owner root"),
        "owner_cell_root": _plain_directory(args.owner_cell_root, label="owner cell root"),
        "editor_artifact_root": _plain_directory(
            args.editor_artifact_root, label="editor artifact root"
        ),
    }
    expected_output = (
        Path(plan["output_root"])
        / "evidence"
        / args.cell_id
        / args.arm_id
    )
    output = _fresh_output(args.output_dir, expected=expected_output)
    return {
        "plan_path": plan_path,
        "plan": plan,
        "cell": dict(cell),
        "arm": arm,
        "query_registry": query_registry,
        "arm_registry": arm_registry,
        "runtime_source": runtime_source,
        "checkpoint_manifest": checkpoint_manifest,
        "owner_master": owner_master,
        "owner_sidecar": owner_sidecar,
        "owner_evidence": owner_evidence,
        "owner_public_key": owner_public_key,
        "owner_cell_receipt": owner_cell_receipt,
        "editor_receipt": editor_receipt,
        "editor_public_key": editor_public_key,
        "execution_signer": execution_signer,
        "device_environment": device_environment,
        "output": output,
        **roots,
    }


def _sampling_contract(native: Any, *, seed: int) -> dict[str, Any]:
    """Use VR2V numeric pins with the source-video-only two-forward APG mode."""

    contract = dict(native.native_sampling_contract("rv2v", steps=40, seed=seed))
    expected = {
        "num_frames": 81,
        "num_inference_steps": 40,
        "guidance_mode": "rv2v",
        "omega_vid": 1.25,
        "omega_img": 4.5,
        "omega_txt": 4.0,
        "omega_scale": 0.8,
        "flow_shift": 5.0,
        "seed": seed,
        "eta": 0.5,
        "norm_threshold": (50.0, 50.0),
        "momentum": 0.0,
    }
    if contract != expected:
        raise BraidStage0World4Error("native RV2V sampling pins changed")
    contract.update(
        {
            "guidance_mode": "v2v_apg",
            "omega_img": 0.0,
            "omega_scale": 0.75,
            "norm_threshold": (50.0, 50.0),
        }
    )
    return contract


def _process_start_identity(
    *, proc_root: Path = Path("/proc")
) -> str:
    """Hash Linux boot+PID+kernel start time+argv for fresh-process evidence."""

    boot = proc_root / "sys/kernel/random/boot_id"
    stat_path = proc_root / "self/stat"
    cmdline_path = proc_root / "self/cmdline"
    try:
        boot_id = boot.read_text(encoding="ascii").strip()
        stat_line = stat_path.read_text(encoding="ascii").strip()
        cmdline = cmdline_path.read_bytes()
        after_name = stat_line.rsplit(")", 1)[1].strip().split()
        start_ticks = int(after_name[19])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise BraidStage0World4Error("cannot establish Linux process start identity") from error
    if not boot_id or not cmdline or start_ticks <= 0:
        raise BraidStage0World4Error("Linux process start identity is incomplete")
    return stage0.object_sha256(
        {
            "schema_version": "bernini-braid-linux-process-start-v1",
            "boot_id": boot_id,
            "pid": os.getpid(),
            "kernel_start_ticks": start_ticks,
            "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
        }
    )


def _object_identity(
    value: Any, *, role: str, process_start_identity_sha256: str
) -> str:
    return stage0.object_sha256(
        {
            "schema_version": "bernini-braid-process-object-v1",
            "process_start_identity_sha256": process_start_identity_sha256,
            "role": role,
            "object_id": id(value),
            "exact_type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    )


def _validate_runtime_schedule(runtime_receipt: Mapping[str, Any], schedule: Any) -> None:
    timesteps = tuple(schedule.NATIVE_UNIPC40_TIMESTEPS)
    sigmas = tuple(schedule.NATIVE_UNIPC40_SIGMAS)
    trace = runtime_receipt.get("trace")
    if (
        not isinstance(trace, list)
        or len(trace) != 40
        or len(timesteps) != 40
        or len(sigmas) != 40
    ):
        raise BraidStage0World4Error("exact40 runtime schedule closure differs")
    for index, row in enumerate(trace):
        if (
            not isinstance(row, Mapping)
            or row.get("step_index") != index
            or float(row.get("timestep")) != float(timesteps[index])
            or float(row.get("sigma")) != float(sigmas[index])
        ):
            raise BraidStage0World4Error("dual runtime trace differs from exact40 pin")


def _read_negative_fp32_hash(
    *, runtime_inputs: Any, materializer: Any
) -> str:
    artifact = runtime_inputs.payload.get("materialization_receipt_artifact")
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "path",
        "file_sha256",
        "receipt_digest",
    }:
        raise BraidStage0World4Error("signed materialization binding differs")
    value, path, observed_sha = materializer._strict_json_file(
        Path(artifact["path"]),
        expected_sha256=artifact["file_sha256"],
        label="signed editor materialization receipt",
    )
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    prompts = value.get("prompts")
    text_encoder = prompts.get("text_encoder_receipt") if isinstance(prompts, Mapping) else None
    expected = (
        text_encoder.get("negative_condition_tensor_sha256")
        if isinstance(text_encoder, Mapping)
        else None
    )
    if (
        path != Path(artifact["path"])
        or observed_sha != artifact["file_sha256"]
        or digest != artifact["receipt_digest"]
        or digest != materializer.object_sha256(unsigned)
        or type(expected) is not str
        or _SHA256.fullmatch(expected) is None
    ):
        raise BraidStage0World4Error("negative-condition materialization seal differs")
    return expected


def _all_equal(rows: Sequence[Mapping[str, Any]], key: str) -> bool:
    if len(rows) != 4:
        return False
    values = [row.get(key) for row in rows]
    return all(value == values[0] for value in values[1:])


def _build_world4_receipt(
    *,
    contract: Mapping[str, Any],
    gathered: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Build one sealed payload from four real-forward rank packets."""

    if (
        len(gathered) != 4
        or [row.get("sp_rank") for row in gathered] != [0, 1, 2, 3]
    ):
        raise BraidStage0World4Error("WORLD4 rank packet order differs")
    runtime_receipts = [dict(row["runtime_receipt"]) for row in gathered]
    process_rows = [dict(row["process_evidence"]) for row in gathered]
    coordinates = [dict(row["coordinate_evidence"]) for row in gathered]
    collectives = [dict(row["collective_receipt"]) for row in gathered]
    device_rows = [dict(row["device_environment_evidence"]) for row in gathered]
    if (
        not all(value == coordinates[0] for value in coordinates[1:])
        or len({row["process_start_identity_sha256"] for row in process_rows}) != 4
        or [row.get("sp_rank") for row in process_rows] != [0, 1, 2, 3]
        or [row.get("sp_rank") for row in collectives] != [0, 1, 2, 3]
        or [row.get("sp_rank") for row in device_rows] != [0, 1, 2, 3]
        or [row.get("rank") for row in device_rows] != [0, 1, 2, 3]
        or [row.get("local_rank") for row in device_rows] != [0, 1, 2, 3]
        or not _all_equal(collectives, "group_contract_digest")
    ):
        raise BraidStage0World4Error("WORLD4 coordinate/process/collective consensus failed")

    arm = contract["arm"]
    if arm.arm_id != SUPPORTED_ARM_ID:
        raise BraidStage0World4Error("receipt builder received an unsupported arm")
    for rank, runtime_receipt in enumerate(runtime_receipts):
        stage0._validate_runtime_receipt(runtime_receipt, rank=rank, arm=arm)
    off_off = all(
        all(
            trace["negative_repeat_exact_parity"] is True
            and trace["negative_repeat_mismatch_bytes"] == 0
            and trace["action_base_velocity_delta_rms"] == 0.0
            and trace["base_stock_apg_exact_parity"] is True
            and trace["base_stock_apg_parity_max_abs"] == 0.0
            and trace["base_stock_apg_parity_rms"] == 0.0
            for trace in runtime["trace"]
        )
        for runtime in runtime_receipts
    )
    projection_exact = all(
        all(
            record["target_post_reset_mismatch_bytes"] == 0
            and record["padding_post_reset_mismatch_bytes"] == 0
            and record["reset_off_returned_original_object"] is True
            and record["reset_returned_new_object"] is False
            for record in runtime["block15"]["records"]
        )
        for runtime in runtime_receipts
    )
    if not off_off or not projection_exact:
        raise BraidStage0World4Error("real forward did not establish off/off structural parity")

    coordinate = dict(coordinates[0])
    coordinate.update(
        {
            "source_and_noise_byte_identity_revalidated": True,
            "prompt_byte_identity_revalidated": True,
            "all_rank_coordinate_consensus": True,
        }
    )
    metric_packet = {
        "schema_version": "bernini-braid-stage0-reference4f-a-real-forward-metrics-v1",
        "runtime_digests": [row["runtime_digest"] for row in runtime_receipts],
        "endpoint_latent_sha256": coordinate["endpoint_latent_sha256"],
        "process_evidence": process_rows,
        "collective_receipts": collectives,
        "device_environment_evidence": device_rows,
        "off_off_path_structural_pass": off_off,
        "projection_local_zero_residual_exact": projection_exact,
    }
    plan = contract["plan"]
    unsigned = {
        "schema_version": stage0.WORLD4_SCHEMA,
        "method": stage0.METHOD,
        "plan_receipt_digest": plan["receipt_digest"],
        "cell_id": contract["cell"]["cell_id"],
        "query_seed": contract["cell"]["query_seed"],
        "source_iid": contract["cell"]["source_iid"],
        "arm_id": arm.arm_id,
        "arm_contract": stage0.asdict(arm),
        "topology": {
            "world_size": 4,
            "sequence_parallel_size": 4,
            "rank_order": [0, 1, 2, 3],
            "visible_devices": contract["cell"]["visible_devices"],
        },
        "provenance": plan["provenance"],
        "coordinate_evidence": coordinate,
        "mechanism_evidence": {
            "visual_pack_mode": stage0.VISUAL_PACK_MODE,
            "sp4_collective_receipt_digest": stage0.object_sha256(collectives),
            "source_bias_mode": "none",
            "source_bias_operator_digest": None,
            "source_bias_read_only": True,
            "source_bias_parameter_mutation": False,
            "comparison_evaluator_source_sha256": plan["provenance"][
                "runner_source_sha256"
            ],
            "comparison_threshold_registry_sha256": stage0.PINNED_BRAID_ARM_REGISTRY_SHA256,
            "all_rank_metric_packet_digest": stage0.object_sha256(metric_packet),
            "all_rank_mechanism_consensus": True,
        },
        "runtime_receipts": runtime_receipts,
        "fresh_process_evidence": process_rows,
        "device_environment_evidence": device_rows,
        "measurements": {
            "runtime_finalize_passed": True,
            "projection_local_zero_residual_exact": projection_exact,
            "off_off_path_structural_pass": off_off,
            "reset_on_off_path_structural_pass": None,
            "old_motion_axis_observed": None,
            "desired_action_capacity_axis_observed": None,
            "old_motion_action_capacity_non_regression_pass": None,
            "scheduler_steps_observed": 40,
            "scheduler_advances_per_step": 1,
            "exact81_latent_rollout_observed": True,
            "decoded_video_observed": False,
        },
        "execution_authority": dict(stage0.EXECUTION_AUTHORITY),
        "result": {
            "status": "PASS",
            "classification": "ENGINEERING_FORWARD_PATH_ONLY",
            "semantic_authority": False,
            "decoded_quality_authority": False,
            "stage0_training_authority": False,
        },
    }
    return stage0.seal_receipt(unsigned)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    contract = dict(validate_cli(args))

    # The method-owned validator imports no Bernini package.  Authenticate and
    # activate both source trees before importing any official model symbols.
    import infer_lora as legacy

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                contract["bernini_root"],
                contract["veomni_root"],
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            contract["checkpoint"]
        )
    except legacy.trainer.TrainingContractError as error:
        raise BraidStage0World4Error(str(error)) from error
    if (
        bernini_revision != stage0.PINNED_BERNINI_REVISION
        or veomni_revision != stage0.PINNED_VEOMNI_REVISION
        or args.expected_checkpoint_tree_sha256 != legacy.trainer.CHECKPOINT_TREE_SHA256
        or transformer_config.get("num_attention_heads") != 12
        or transformer_config.get("attention_head_dim") != 128
    ):
        raise BraidStage0World4Error("authenticated Bernini/checkpoint geometry differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.parallel import get_parallel_state, init_parallel_state

    import braid_dual_native_apg_runtime_v1 as dual
    import infer_native_identity_generation_canary as native
    import materialize_qmosaic_editor_runtime_v1 as materializer
    import self_imagined_native_rv2v_hidden_vjp_v1 as qmosaic
    import source_self_native_ref_contrastive_v3 as schedule_contract
    import tri_branch_unipc as sampler_contract

    if (
        Path(dual.__file__).resolve(strict=True) != contract["runtime_source"]
        or stage0.file_sha256(Path(dual.__file__).resolve(strict=True))
        != args.expected_dual_runtime_source_sha256
    ):
        raise BraidStage0World4Error("imported dual runtime differs from authenticated source")

    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != 4
        or distributed.rank != contract["device_environment"]["rank"]
        or distributed.local_rank
        != contract["device_environment"]["local_rank"]
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise BraidStage0World4Error("runner requires one AUH WORLD4/SP4 ROCm group")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=4)
    device = torch.device("cuda", distributed.local_rank)
    output = contract["output"]
    patch: Any = None
    try:
        create_result: list[Any] = [None]
        if distributed.rank == 0:
            try:
                output.mkdir(mode=0o700)
                create_result[0] = {"ok": True}
            except BaseException as error:
                create_result[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(create_result, src=0)
        if not isinstance(create_result[0], Mapping) or create_result[0].get("ok") is not True:
            raise BraidStage0World4Error(f"rank-zero output creation failed: {create_result[0]}")

        checkpoint_packet = qmosaic.load_validated_checkpoint_content_manifest(
            checkpoint_root=checkpoint,
            content_manifest_path=contract["checkpoint_manifest"],
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
            expected_file_count=CHECKPOINT_CONTENT_FILE_COUNT,
        )
        owner = qmosaic.load_authenticated_owner_quotient_packet(
            registry=contract["query_registry"],
            expected_registry_sha256=args.expected_query_registry_sha256,
            owner_root=contract["owner_root"],
            owner_master_receipt=contract["owner_master"],
            expected_owner_master_receipt_sha256=args.expected_owner_master_receipt_sha256,
            audit_sidecar=contract["owner_sidecar"],
            expected_audit_sidecar_sha256=args.expected_owner_audit_sidecar_sha256,
            audit_evidence=contract["owner_evidence"],
            audit_public_key=contract["owner_public_key"],
            expected_audit_public_key_sha256=args.expected_owner_audit_public_key_sha256,
            cell_root=contract["owner_cell_root"],
            receipt_path=contract["owner_cell_receipt"],
            expected_receipt_file_sha256=args.expected_owner_cell_receipt_sha256,
            query_seed=args.query_seed,
        )
        if (
            owner.cell_id != args.cell_id
            or owner.query_seed != args.query_seed
            or owner.source_iid != contract["cell"]["source_iid"]
        ):
            raise BraidStage0World4Error("owner packet differs from Stage-0 cell")
        runtime_inputs = qmosaic.load_authenticated_editor_runtime_input_packet(
            receipt_path=contract["editor_receipt"],
            expected_receipt_file_sha256=args.expected_editor_receipt_sha256,
            public_key_path=contract["editor_public_key"],
            expected_public_key_file_sha256=args.expected_editor_public_key_sha256,
            artifact_root=contract["editor_artifact_root"],
            owner=owner,
            checkpoint=checkpoint_packet,
        )
        editor_binding = runtime_inputs.receipt()
        editor_source_revision = _sha1(
            editor_binding.get("method_source_revision"),
            label="signed editor method source revision",
        )
        editor_source_archive_sha256 = _sha256(
            editor_binding.get("method_source_archive_sha256"),
            label="signed editor method source archive",
        )

        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        if float(config.shift) != 5.0 or config.use_unipc is not True:
            raise BraidStage0World4Error("renderer is not pinned native UniPC shift5")
        renderer = BerniniRendererModel(config).eval().requires_grad_(False).to(device)
        diffusion = sampler_contract.resolve_diffusion_core(renderer.diff_dec)
        transformer = diffusion.transformer
        if (
            transformer is None
            or diffusion.transformer_2 is not None
            or any(parameter.requires_grad for parameter in renderer.parameters())
            or any("lora" in name.lower() for name, _ in renderer.named_parameters())
        ):
            raise BraidStage0World4Error("runner requires one frozen adapter-free transformer")
        wan_source_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
        )
        collective = qmosaic.authenticate_live_bernini_sp4_collective(
            parallel_state=get_parallel_state()
        )

        expected_negative_fp32_sha = _read_negative_fp32_hash(
            runtime_inputs=runtime_inputs, materializer=materializer
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint),
            subfolder="tokenizer",
            **legacy.tokenizer_load_kwargs(),
        )
        if (
            tokenizer.padding_side != "right"
            or tokenizer.init_kwargs.get("fix_mistral_regex") is not True
        ):
            raise BraidStage0World4Error("pinned tokenizer contract differs")
        negative_ids, negative_mask = legacy._tokenize_renderer_negative(
            tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT
        )
        with torch.inference_mode():
            negative_fp32 = renderer.encode_prompt(
                negative_ids.to(device), negative_mask.to(device)
            ).detach().float().contiguous()
        dist.broadcast(negative_fp32, src=0)
        if (
            tuple(negative_fp32.shape) != (1, 512, 4096)
            or materializer.tensor_sha256(
                negative_fp32, label="reconstructed negative FP32 condition"
            )
            != expected_negative_fp32_sha
        ):
            raise BraidStage0World4Error("reconstructed negative condition differs")
        renderer.t5_text_encoder.to("cpu")
        del tokenizer, negative_ids, negative_mask
        torch.cuda.empty_cache()

        source_latent = (
            runtime_inputs.tensors["source_latent"]
            .detach()
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )
        packet_source_sha = qmosaic.tensor_sha256(
            runtime_inputs.tensors["source_latent"], label="signed source latent"
        )
        if qmosaic.tensor_sha256(source_latent, label="live source latent") != packet_source_sha:
            raise BraidStage0World4Error("live source latent differs from signed packet")
        vendor_noop = materializer._vendor_condition_bf16(
            runtime_inputs.tensors["noop_condition"],
            transformer=transformer,
            label="signed c0 no-op condition",
        )
        # This is the parity arm: action c0 is the same Python tensor object,
        # not a separately cast equal-value copy.
        vendor_action = vendor_noop
        vendor_negative = materializer._vendor_condition_bf16(
            negative_fp32,
            transformer=transformer,
            label="reconstructed negative condition",
        )
        if vendor_action is not vendor_noop:
            raise BraidStage0World4Error("c0/c0 prompt object identity differs")
        editor_noise_seed = runtime_inputs.payload.get("editor_noise_seed")
        if type(editor_noise_seed) is not int:
            raise BraidStage0World4Error("signed editor noise seed differs")
        sample_values = _sampling_contract(native, seed=editor_noise_seed)
        process_identity = _process_start_identity()
        model_identity = _object_identity(
            renderer,
            role="BerniniRendererModel",
            process_start_identity_sha256=process_identity,
        )
        scheduler_identity = _object_identity(
            diffusion.scheduler,
            role="UniPCMultistepScheduler",
            process_start_identity_sha256=process_identity,
        )

        patch = dual.BraidDualNativeAPGRuntimePatch(
            diffusion,
            action_prompt_embeds=vendor_action,
            config=dual.BraidDualNativeAPGConfig(
                target_latent_shape=tuple(int(value) for value in source_latent.shape),
                sp_rank=collective.sp_rank,
                reset_source_costate=False,
                forward_mode="reference_4f",
                allow_shared_negative_diagnostic=False,
                expected_steps=40,
                expected_num_frames=81,
                block_index=15,
            ),
            expected_bernini_commit=bernini_revision,
            observed_wan_diffusion_sha256=wan_source_sha,
        )
        patch.install()
        try:
            with torch.inference_mode():
                endpoint, noise_capture = native._sample_with_native_initial_noise_observer(
                    sample_fn=lambda: diffusion.sample(
                        prompt_embeds=vendor_noop,
                        uncond_prompt_embeds=vendor_negative,
                        image_vae_latents=None,
                        multi_video_vae_latents=[source_latent],
                        multi_image_vae_latents=None,
                        width=int(source_latent.shape[4]) * 8,
                        height=int(source_latent.shape[3]) * 8,
                        device=device,
                        **sample_values,
                    ),
                    wan_diffusion_module=wan_diffusion,
                    expected_shape=tuple(int(value) for value in source_latent.shape),
                    expected_device=device,
                    expected_seed=editor_noise_seed,
                )
        finally:
            if patch is not None and patch.installed and not patch.restored:
                patch.restore()
        runtime_receipt = dict(patch.finalize())
        _validate_runtime_schedule(runtime_receipt, schedule_contract)
        live_schedule = materializer._capture_live_exact40_schedule(
            diffusion.scheduler, schedule_contract=schedule_contract
        )
        if live_schedule.get("static_schedule_digest") != stage0.PINNED_NATIVE_SCHEDULE_DIGEST:
            raise BraidStage0World4Error("live exact40 schedule digest differs")

        signed_noise = runtime_inputs.tensors["official_initial_noise"]
        if (
            not torch.equal(noise_capture.tensor, signed_noise)
            or tuple(endpoint.shape) != tuple(source_latent.shape)
            or endpoint.dtype != torch.float32
            or endpoint.requires_grad
            or endpoint.grad_fn is not None
            or not bool(torch.isfinite(endpoint).all().item())
        ):
            raise BraidStage0World4Error("native endpoint/noise closure differs")

        runtime_inputs.assert_live(owner, checkpoint_packet)
        owner.assert_live()
        checkpoint_packet.assert_live()
        collective.assert_live()
        if qmosaic.tensor_sha256(
            source_latent, label="terminal live source latent"
        ) != packet_source_sha:
            raise BraidStage0World4Error("live source latent changed during sampling")
        coordinate = {
            "editor_runtime_input_receipt_digest": runtime_inputs.payload[
                "receipt_digest"
            ],
            "editor_runtime_input_receipt_file_sha256": (
                args.expected_editor_receipt_sha256
            ),
            "editor_public_key_file_sha256": (
                args.expected_editor_public_key_sha256
            ),
            "editor_method_source_revision": editor_source_revision,
            "editor_method_source_archive_sha256": editor_source_archive_sha256,
            "source_latent_sha256": packet_source_sha,
            "official_initial_noise_sha256": qmosaic.tensor_sha256(
                noise_capture.tensor, label="observed official initial noise"
            ),
            "endpoint_latent_sha256": qmosaic.tensor_sha256(
                endpoint, label="exact81 latent endpoint"
            ),
            "noop_prompt_tensor_sha256": qmosaic.tensor_sha256(
                vendor_noop, label="vendor c0 condition"
            ),
            "action_prompt_tensor_sha256": qmosaic.tensor_sha256(
                vendor_action, label="vendor action c0 condition"
            ),
            "negative_prompt_tensor_sha256": qmosaic.tensor_sha256(
                vendor_negative, label="vendor negative condition"
            ),
            "exact40_timestep_sigma_digest": live_schedule[
                "static_schedule_digest"
            ],
        }
        base_binding = runtime_receipt["base_apg_binding"]
        action_binding = runtime_receipt["action_apg_binding"]
        process_evidence = {
            "sp_rank": collective.sp_rank,
            "process_start_identity_sha256": process_identity,
            "model_object_identity_sha256": model_identity,
            "scheduler_object_identity_sha256": scheduler_identity,
            "noop_apg_state_identity_sha256": stage0.apg_state_identity_sha256(
                process_start_identity_sha256=process_identity,
                binding=base_binding,
            ),
            "action_apg_state_identity_sha256": stage0.apg_state_identity_sha256(
                process_start_identity_sha256=process_identity,
                binding=action_binding,
            ),
            "model_construct_count": 1,
            "scheduler_construct_count": 1,
            "sample_call_count": 1,
        }
        local_packet = {
            "sp_rank": collective.sp_rank,
            "runtime_receipt": runtime_receipt,
            "process_evidence": process_evidence,
            "coordinate_evidence": coordinate,
            "collective_receipt": dict(collective.receipt()),
            "device_environment_evidence": dict(
                contract["device_environment"]
            ),
        }
        if collective.sp_rank != contract["device_environment"]["sp_rank"]:
            raise BraidStage0World4Error(
                "live SP rank differs from pre-torch device evidence"
            )
        gathered = collective.all_gather_object(local_packet)
        payload = _build_world4_receipt(contract=contract, gathered=gathered)

        publish: list[Any] = [None]
        if collective.sp_rank == 0:
            try:
                receipt = _sign_world4_payload(
                    payload,
                    signer=contract["execution_signer"],
                    public_key_file_sha256=(
                        args.expected_execution_public_key_sha256
                    ),
                )
                receipt = stage0.validate_world4_receipt(
                    receipt,
                    plan=contract["plan"],
                    expected_cell=args.cell_id,
                    expected_arm=args.arm_id,
                )
                target = output / WORLD4_RECEIPT_FILENAME
                stage0.write_create_only_json(target, receipt)
                reopened = stage0.validate_world4_receipt(
                    stage0.load_json(target, label="published WORLD4 receipt"),
                    plan=contract["plan"],
                    expected_cell=args.cell_id,
                    expected_arm=args.arm_id,
                )
                publish[0] = {
                    "ok": True,
                    "receipt_digest": reopened["receipt_digest"],
                    "receipt_file_sha256": stage0.file_sha256(target),
                }
            except BaseException as error:
                publish[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(publish, src=0)
        if (
            not isinstance(publish[0], Mapping)
            or publish[0].get("ok") is not True
            or publish[0].get("receipt_digest") != payload["receipt_digest"]
        ):
            raise BraidStage0World4Error(f"rank-zero receipt publication failed: {publish[0]}")
        collective.assert_live()
        target = output / WORLD4_RECEIPT_FILENAME
        reopened_all_ranks = stage0.validate_world4_receipt(
            stage0.load_json(target, label="published WORLD4 receipt on every rank"),
            plan=contract["plan"],
            expected_cell=args.cell_id,
            expected_arm=args.arm_id,
        )
        if (
            stage0.file_sha256(target)
            != publish[0].get("receipt_file_sha256")
        ):
            raise BraidStage0World4Error(
                "published WORLD4 file differs across ranks"
            )
        return dict(reopened_all_ranks)
    finally:
        if patch is not None and patch.installed and not patch.restored:
            patch.restore()
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run(args)
    if os.environ.get("RANK", "0") == "0":
        print(stage0.canonical_json_bytes(receipt).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "BraidStage0World4Error",
    "CHECKPOINT_TREE_SHA256",
    "SUPPORTED_ARM_ID",
    "WORLD4_RECEIPT_FILENAME",
    "_build_world4_receipt",
    "_process_start_identity",
    "_sampling_contract",
    "_validate_runtime_schedule",
    "build_parser",
    "main",
    "run",
    "validate_cli",
]

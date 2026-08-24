#!/usr/bin/env python3
"""WORLD4 native-only e02 regeneration diagnostic runner.

The entrypoint is intentionally inert while the independent execution
runtime/preflight
compiled trust anchors are unset.  A later exact-byte release may run only the
two preregistered arms for e02 (official V2V base and scheduled source-reference
R2V-4 inside the reviewed exact D/C/K union).  e03 is policy-only: its existing
frozen base is hash-bound and retained without materialization or sampling.
The self-generated anchor is review context only; no anchor tensor, target,
FlowEdit, connected route, training, optimizer, or automatic selection enters
this runner.  e03 is always recorded as ``ABSTAIN_KEEP_BASE``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
PREFLIGHT_PATH = (
    METHOD_ROOT / "preflight_oracle_regeneration_native_activation_v2_r2.py"
).resolve(strict=True)
while str(METHOD_ROOT) in sys.path:
    sys.path.remove(str(METHOD_ROOT))
sys.path.insert(0, str(METHOD_ROOT))
_preloaded = sys.modules.get("preflight_oracle_regeneration_native_activation_v2_r2")
if _preloaded is not None and Path(
    str(getattr(_preloaded, "__file__", ""))
).resolve(strict=True) != PREFLIGHT_PATH:
    raise RuntimeError("preloaded activation-v2 preflight origin differs")
_preflight_spec = importlib.util.find_spec(
    "preflight_oracle_regeneration_native_activation_v2_r2"
)
if (
    _preflight_spec is None
    or not isinstance(_preflight_spec.origin, str)
    or Path(_preflight_spec.origin).resolve(strict=True) != PREFLIGHT_PATH
):
    raise RuntimeError("activation-v2 preflight import origin differs")
import preflight_oracle_regeneration_native_activation_v2_r2 as release_preflight  # noqa: E402


SCHEMA_VERSION = "bernini-oracle-regeneration-native-activation-v2-run-r11"
METHOD = "round37-native-source-reference-r2v4-local-regeneration-diagnostic-r11"
WORLD_SIZE = 4
NUM_INFERENCE_STEPS = 40
FRAME_COUNT = 81
FPS = 25
AUTHORIZED_SLURM_JOB_ID = "141620"
AUTHORIZED_HOSTNAME = "auh7-1b-gpu-226"
ARM_OFFICIAL = "official-v2v-base"
ARM_LOCAL = "local-source-reference-r2v4-in-manual-G"
CASE_ORDER = ("e02", "e03")
EXECUTION_CASES = ("e02",)
ARM_ORDER_BY_CASE = {
    "e02": (ARM_OFFICIAL, ARM_LOCAL),
    "e03": (),
}
E03_FROZEN_BASE_PATH = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/"
    "interaction_complex8_rv2v_candidates_v1/complex8-e03-rv2v-s0/rv2v.mp4"
)
E03_FROZEN_BASE_SHA256 = (
    "d75bbafbbc225ea3935c2d149be8b3969fffd6d8b645c5ec9edb5968bf25f654"
)
MIOPEN_CACHE_SUFFIX = ".miopen-cache-r9"
MIOPEN_CACHE_ROOT_ENV = "ORACLE_ACTIVATION_V2_MIOPEN_CACHE_ROOT"
MIOPEN_LOCAL_TMP_ROOT_ENV = "ORACLE_ACTIVATION_V2_MIOPEN_LOCAL_TMP_ROOT"
MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV = (
    "ORACLE_ACTIVATION_V2_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_SHA256"
)
MIOPEN_USER_DB_ENV = "MIOPEN_USER_DB_PATH"
MIOPEN_KERNEL_CACHE_ENV = "MIOPEN_CUSTOM_CACHE_DIR"
MIOPEN_CACHE_RECEIPT_NAME = "miopen-cache-receipt.json"
MIOPEN_LOCAL_TMP_RECEIPT_NAME = "local-tmp-receipt.json"
MIOPEN_TEMP_ENV_NAMES = ("TMPDIR", "TMP", "TEMP", "TEMPDIR")
SCHEDULER_TMPDIR_NORMALIZATION_ENV = {
    "job_id": "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_JOB_ID",
    "step_id": "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_STEP_ID",
    "hostname": "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_HOSTNAME",
    "observed_tmpdir": "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_OBSERVED",
    "action": "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_ACTION",
}
SCHEDULER_TMPDIR_NORMALIZATION_ACTION = "UNSET_BEFORE_ANY_PYTHON"
MIOPEN_BOOTSTRAP_ROLES = ("user-db", "kernel-cache")
MIOPEN_RANK_ROLES = ("user-db", "kernel-cache")
MIOPEN_LOCAL_TMP_PARENT = Path("/tmp")
MIOPEN_LOCAL_TMP_DOMAIN = (
    "bernini-oracle-regeneration-native-activation-v2-local-tmp-r9"
)
MIOPEN_USER_DB_MAIN_BASENAMES = frozenset(
    {
        "gfx90a68.HIP.3_3_0_a85ca8a54-dirty.udb.txt",
        "gfx90a68.HIP.3_3_0_a85ca8a54-dirty.ufdb.txt",
    }
)
MIOPEN_USER_DB_TIME_BASENAMES = frozenset(
    f"{name}.time" for name in MIOPEN_USER_DB_MAIN_BASENAMES
)
MIOPEN_KERNEL_CACHE_BASENAMES = frozenset(
    {"gfx90a68.ukdb", "gfx90a68.ukdb-shm", "gfx90a68.ukdb-wal"}
)
MIOPEN_LIBRARY_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/torch/lib/libMIOpen.so"
)
MIOPEN_LIBRARY_SHA256 = (
    "1e6cc33ca21951dce12795e6c5d99578e8f2f1754b84a703508df44426b44b52"
)
MIOPEN_LIBRARY_SIZE = 690355265
_LOCAL_RELEASE_PATHS = {
    "spec": "assets/oracle_regeneration_native_activation_v2_r2_spec.json",
    "runtime": "oracle_regeneration_native_runtime_activation_v2.py",
    "runner": "infer_oracle_regeneration_native_activation_v2_r2.py",
    "vae_tool": "tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py",
    "prompt_tool": "tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py",
    "dependency:oracle_regeneration_canary_v1.py": "oracle_regeneration_canary_v1.py",
    "dependency:native_branch_homotopy_runtime_v1.py": "native_branch_homotopy_runtime_v1.py",
    "dependency:native_branch_homotopy_v1.py": "native_branch_homotopy_v1.py",
    "dependency:self_guided_action_field_v1.py": "self_guided_action_field_v1.py",
    "dependency:tri_branch_unipc.py": "tri_branch_unipc.py",
    "dependency:infer_native_identity_generation_canary.py": "infer_native_identity_generation_canary.py",
    "dependency:infer_native_branch_homotopy_canary.py": "infer_native_branch_homotopy_canary.py",
    "dependency:infer_source_kv_carrier_oracle.py": "infer_source_kv_carrier_oracle.py",
    "dependency:infer_source_value_residual_oracle.py": "infer_source_value_residual_oracle.py",
    "dependency:infer_native_self_guided_action_field_canary.py": "infer_native_self_guided_action_field_canary.py",
    "dependency:infer_lora.py": "infer_lora.py",
    "dependency:tools/materialize_vae.py": "tools/materialize_vae.py",
    "dependency:tools/build_renderer_dataset.py": "tools/build_renderer_dataset.py",
    "dependency:tools/__init__.py": "tools/__init__.py",
    "dependency:train_lora.py": "train_lora.py",
    "dependency:action_preservation_decoded_eval_model_authority_v2.py": "action_preservation_decoded_eval_model_authority_v2.py",
    "dependency:source_kv_replay.py": "source_kv_replay.py",
    "dependency:source_kv_route_batches.py": "source_kv_route_batches.py",
    "dependency:source_self_native_ref_contrastive_v3.py": "source_self_native_ref_contrastive_v3.py",
    "dependency:source_value_residual.py": "source_value_residual.py",
}
_COPIED_LOCAL_PROMPT_ROLE_BINDINGS = {
    "tokenizer_code": {
        "receipt_path": (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "action_flow_noise_stage0_job140846_v1/stage1/"
            "oracle_activation_v2_materializer_r2_ef97259d_07b40cdd_"
            "5bba9f97_6da8c414/methods/bernini_action_editing/infer_lora.py"
        ),
        "release_relative_path": "infer_lora.py",
        "sha256": (
            "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
        ),
    },
    "prompt_builder_code": {
        "receipt_path": (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "action_flow_noise_stage0_job140846_v1/stage1/"
            "oracle_activation_v2_materializer_r2_ef97259d_07b40cdd_"
            "5bba9f97_6da8c414/methods/bernini_action_editing/"
            "infer_native_branch_homotopy_canary.py"
        ),
        "release_relative_path": "infer_native_branch_homotopy_canary.py",
        "sha256": (
            "d6dab735ce52da151848c96f9e00775994dc281ace20afa6dcb9fb64709e5983"
        ),
    },
    "native_prompt_code": {
        "receipt_path": (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "action_flow_noise_stage0_job140846_v1/stage1/"
            "oracle_activation_v2_materializer_r2_ef97259d_07b40cdd_"
            "5bba9f97_6da8c414/methods/bernini_action_editing/"
            "infer_native_identity_generation_canary.py"
        ),
        "release_relative_path": "infer_native_identity_generation_canary.py",
        "sha256": (
            "bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42"
        ),
    },
}
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NativeActivationV2RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class _CallSurfaceSnapshot:
    tokens: tuple[Any, ...]
    receipt: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _checkpoint_content_identity_sha256(
    value: Any, *, activation: Any
) -> str:
    """Hash checkpoint identities in the materializer receipt byte domain.

    Checkpoint materialization receipts use the frozen activation canonical
    JSON encoder, whose payload has one trailing LF.  Other runner receipts
    intentionally remain in the legacy no-LF ``_canonical_sha256`` domain.
    """

    raw = activation.safe_core.canonical_json_bytes_v1(value)
    digest = activation._canonical_object_sha256(value)
    if (
        not isinstance(raw, bytes)
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        raise NativeActivationV2RunnerError(
            "checkpoint canonical identity digest differs"
        )
    return digest


def _raw_rgb_sha256(frame: Any, *, expected_hw: tuple[int, int]) -> str:
    """Hash one decoded RGB frame with the authoring-receipt byte domain."""

    if (
        str(getattr(frame, "dtype", "")) != "uint8"
        or tuple(getattr(frame, "shape", ())) != (*expected_hw, 3)
        or not bool(getattr(getattr(frame, "flags", None), "c_contiguous", False))
    ):
        raise NativeActivationV2RunnerError("decoded source RGB frame differs")
    header = json.dumps(
        {
            "schema_version": "decoded-uint8-rgb-frame-v1",
            "shape": [*expected_hw, 3],
            "dtype": "uint8",
            "channel_order": "RGB",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\x00" + frame.tobytes(order="C")).hexdigest()


def cpu_preflight(*, authority_packet: Path, external_ledger: Path) -> Mapping[str, Any]:
    """Complete the exact CPU trust gate before Torch or model imports."""

    if "torch" in sys.modules:
        raise NativeActivationV2RunnerError("Torch was imported before CPU preflight")
    result = release_preflight.validate_release(
        packet_path=authority_packet,
        ledger_path=external_ledger,
    )
    if (
        result.get("ready") is not True
        or result.get("cpu_only") is not True
        or result.get("torch_imported") is not False
        or result.get("distributed_initialized") is not False
        or result.get("model_loaded") is not False
        or result.get("training") is not False
        or result.get("optimizer") is not False
        or result.get("flowedit") is not False
        or result.get("connected_route") is not False
        or result.get("automatic_replacement") is not False
        or result.get("selection_authority") is not None
    ):
        raise NativeActivationV2RunnerError("CPU activation preflight differs")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-packet", required=True)
    parser.add_argument("--external-ledger", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--miopen-cache-root", required=True)
    parser.add_argument("--miopen-local-tmp-root", required=True)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    return parser


def _fresh_output_dir(value: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise NativeActivationV2RunnerError("output directory must be absolute/non-root")
    parent = requested.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise NativeActivationV2RunnerError("output parent differs")
    output = parent / requested.name
    if (
        output != requested
        or output.exists()
        or output.is_symlink()
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", output.name) is None
    ):
        raise NativeActivationV2RunnerError("refusing ambiguous/overwriting output")
    try:
        output.relative_to(METHOD_ROOT.parents[1])
    except ValueError:
        pass
    else:
        raise NativeActivationV2RunnerError(
            "output/cache must be outside the frozen release tree"
        )
    return output


def _private_directory_identity(path: Path, *, label: str) -> Mapping[str, Any]:
    """Return a no-symlink, effective-user-owned private directory identity."""

    if not path.is_absolute() or path == Path("/") or path.is_symlink():
        raise NativeActivationV2RunnerError(f"{label} path differs")
    try:
        resolved = path.resolve(strict=True)
        row = path.lstat()
    except OSError as error:
        raise NativeActivationV2RunnerError(f"{label} stat failed") from error
    mode = stat.S_IMODE(row.st_mode)
    if (
        resolved != path
        or not stat.S_ISDIR(row.st_mode)
        or int(row.st_uid) != os.geteuid()
        or mode != 0o700
    ):
        raise NativeActivationV2RunnerError(
            f"{label} must be canonical/private/effective-user-owned"
        )
    return {
        "path": str(path),
        "mode": mode,
        "uid": int(row.st_uid),
        "gid": int(row.st_gid),
        "device": int(row.st_dev),
        "inode": int(row.st_ino),
    }


def _node_local_tmp_parent_identity() -> Mapping[str, Any]:
    """Capture and validate the live public ``/tmp`` mount-point identity.

    The inode is deliberately learned at runtime rather than compiled.  Every
    later cache snapshot compares it with this owned initial observation.
    """

    path = MIOPEN_LOCAL_TMP_PARENT
    if not path.is_absolute() or path.is_symlink():
        raise NativeActivationV2RunnerError("node-local tmp parent path differs")
    try:
        resolved = path.resolve(strict=True)
        row = path.lstat()
    except OSError as error:
        raise NativeActivationV2RunnerError(
            "node-local tmp parent stat failed"
        ) from error
    mode = stat.S_IMODE(row.st_mode)
    if (
        resolved != path
        or not stat.S_ISDIR(row.st_mode)
        or mode != 0o1777
        or int(row.st_uid) != 0
    ):
        raise NativeActivationV2RunnerError(
            "node-local tmp parent must be canonical root-owned mode 1777"
        )
    return {
        "path": str(path),
        "mode": mode,
        "uid": int(row.st_uid),
        "gid": int(row.st_gid),
        "device": int(row.st_dev),
        "inode": int(row.st_ino),
    }


def _rank_cache_paths(
    root: Path, local_tmp_root: Path, rank: int
) -> tuple[Path, Path, Path]:
    if type(rank) is not int or rank not in range(WORLD_SIZE):
        raise NativeActivationV2RunnerError("MIOpen cache local rank differs")
    return (
        root / f"rank-{rank}-user-db",
        root / f"rank-{rank}-kernel-cache",
        local_tmp_root / f"rank-{rank}",
    )


def _expected_miopen_cache_directory_names() -> set[str]:
    rank_names = {
        f"rank-{rank}-{role}"
        for rank in range(WORLD_SIZE)
        for role in MIOPEN_RANK_ROLES
    }
    return rank_names | {f"launcher-bootstrap-{role}" for role in MIOPEN_BOOTSTRAP_ROLES}


def _expected_miopen_local_tmp_directory_names() -> set[str]:
    return {f"rank-{rank}" for rank in range(WORLD_SIZE)}


def _launcher_local_tmp_empty_proof(
    root_identity: Mapping[str, Any],
    directory_identities: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Recompute the pinned launcher's exact fresh/empty layout attestation.

    This is not an independent signature.  The frozen launcher emits the
    digest only after creating the fresh layout and observing all four rank
    directories empty.  CPU preflight and every worker independently rebuild
    the same payload from their no-symlink identities before Torch.
    """

    identity_keys = {"path", "mode", "uid", "gid", "device", "inode"}
    names = _expected_miopen_local_tmp_directory_names()
    if (
        not isinstance(root_identity, Mapping)
        or set(root_identity) != identity_keys
        or not isinstance(directory_identities, Mapping)
        or set(directory_identities) != names
        or any(
            not isinstance(directory_identities[name], Mapping)
            or set(directory_identities[name]) != identity_keys
            for name in names
        )
    ):
        raise NativeActivationV2RunnerError(
            "launcher node-local tmp empty proof identity differs"
        )

    def line(kind: str, name: str, row: Mapping[str, Any], empty: str) -> str:
        return "\t".join(
            (
                kind,
                name,
                str(row["path"]),
                format(int(row["mode"]), "o"),
                str(int(row["uid"])),
                str(int(row["gid"])),
                str(int(row["device"])),
                str(int(row["inode"])),
                empty,
            )
        )

    payload_lines = [
        "bernini-launcher-node-local-tmp-fresh-empty-proof-r11",
        line("root", "root", root_identity, "roles-only"),
        *(
            line("role", name, directory_identities[name], "empty")
            for name in sorted(names)
        ),
    ]
    payload = "\n".join(payload_lines).encode("utf-8")
    return {
        "schema_version": (
            "bernini-launcher-node-local-tmp-fresh-empty-proof-r11"
        ),
        "environment_key": MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "fresh_root_created_by_frozen_launcher": True,
        "exact_rank_directory_set_observed_by_frozen_launcher": True,
        "all_rank_directories_observed_empty_by_frozen_launcher": True,
        "not_an_independent_signature": True,
    }


def _expected_miopen_local_tmp_root(
    output_dir: Path, scheduler_tmpdir_normalization: Mapping[str, Any]
) -> Path:
    job_id = str(scheduler_tmpdir_normalization.get("slurm_job_id", ""))
    step_id = str(scheduler_tmpdir_normalization.get("slurm_step_id", ""))
    uid = str(os.geteuid())
    if (
        job_id != AUTHORIZED_SLURM_JOB_ID
        or re.fullmatch(r"0|[1-9][0-9]*", step_id) is None
        or not output_dir.is_absolute()
    ):
        raise NativeActivationV2RunnerError(
            "MIOpen node-local tmp domain binding differs"
        )
    payload = b"\x00".join(
        (
            MIOPEN_LOCAL_TMP_DOMAIN.encode("utf-8"),
            job_id.encode("ascii"),
            step_id.encode("ascii"),
            uid.encode("ascii"),
            str(output_dir).encode("utf-8"),
        )
    )
    output_id = hashlib.sha256(payload).hexdigest()
    return MIOPEN_LOCAL_TMP_PARENT / (
        f"oracle-regeneration-native-v2-r9-u{uid}-j{job_id}-s{step_id}-o{output_id}"
    )


def _certify_miopen_library_file() -> Mapping[str, Any]:
    requested = MIOPEN_LIBRARY_PATH
    if requested.is_symlink() or requested.resolve(strict=True) != requested:
        raise NativeActivationV2RunnerError("MIOpen library path differs")
    identity = _owned_file_identity(requested, label="Torch-bundled MIOpen library")
    if (
        identity["sha256"] != MIOPEN_LIBRARY_SHA256
        or identity["size"] != MIOPEN_LIBRARY_SIZE
        or identity["mode"] != 0o755
        or identity["nlink"] != 1
    ):
        raise NativeActivationV2RunnerError("MIOpen library identity differs")
    return {
        "path": str(requested),
        **dict(identity),
    }


def _certify_scheduler_tmpdir_normalization() -> Mapping[str, Any]:
    """Revalidate the launcher-authenticated Slurm ``TMPDIR`` normalization."""

    prefix = "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_"
    observed_environment = {
        key: value for key, value in os.environ.items() if key.startswith(prefix)
    }
    expected_keys = set(SCHEDULER_TMPDIR_NORMALIZATION_ENV.values())
    if set(observed_environment) != expected_keys:
        raise NativeActivationV2RunnerError(
            "scheduler TMPDIR normalization environment differs"
        )
    values = {
        role: observed_environment[name]
        for role, name in SCHEDULER_TMPDIR_NORMALIZATION_ENV.items()
    }
    step_id = values["step_id"]
    if (
        values["job_id"] != AUTHORIZED_SLURM_JOB_ID
        or os.environ.get("SLURM_JOB_ID") != values["job_id"]
        or not isinstance(step_id, str)
        or re.fullmatch(r"0|[1-9][0-9]*", step_id) is None
        or os.environ.get("SLURM_STEP_ID") != step_id
        or values["hostname"] != AUTHORIZED_HOSTNAME
        or socket.gethostname().split(".", 1)[0] != values["hostname"]
        or values["observed_tmpdir"] != "/tmp"
        or values["action"] != SCHEDULER_TMPDIR_NORMALIZATION_ACTION
    ):
        raise NativeActivationV2RunnerError(
            "scheduler TMPDIR normalization binding differs"
        )
    return {
        "schema_version": "bernini-slurm-tmpdir-normalization-r9",
        "slurm_job_id": values["job_id"],
        "slurm_step_id": step_id,
        "hostname": values["hostname"],
        "scheduler_observed_tmpdir": values["observed_tmpdir"],
        "normalization_action": values["action"],
        "normalized_before_any_python_or_torch_import": True,
        "launcher_receipt_environment_keys": sorted(expected_keys),
    }


def _activate_miopen_cache_pre_torch(
    *, output_dir: Path, cache_root_value: str, local_tmp_root_value: str
) -> Mapping[str, Any]:
    """Bind persistent MIOpen caches plus one node-local rank temp before Torch.

    The launcher creates two persistent bootstrap roles, eight persistent
    rank DB/cache roles, and four rank-private directories on exact ``/tmp``.
    This function selects the three directories for ``LOCAL_RANK``.  It never
    creates or disables a solver/cache path and runs before Torch/MIOpen.
    """

    if "torch" in sys.modules:
        raise NativeActivationV2RunnerError("Torch loaded before MIOpen cache activation")
    requested = Path(cache_root_value).expanduser()
    expected = Path(f"{output_dir}{MIOPEN_CACHE_SUFFIX}")
    if (
        not requested.is_absolute()
        or requested != expected
        or os.environ.get(MIOPEN_CACHE_ROOT_ENV) != str(requested)
    ):
        raise NativeActivationV2RunnerError("MIOpen cache root/output binding differs")
    scheduler_tmpdir_normalization = _certify_scheduler_tmpdir_normalization()
    local_tmp_requested = Path(local_tmp_root_value).expanduser()
    expected_local_tmp = _expected_miopen_local_tmp_root(
        output_dir, scheduler_tmpdir_normalization
    )
    if (
        not local_tmp_requested.is_absolute()
        or local_tmp_requested != expected_local_tmp
        or os.environ.get(MIOPEN_LOCAL_TMP_ROOT_ENV)
        != str(local_tmp_requested)
    ):
        raise NativeActivationV2RunnerError(
            "MIOpen node-local tmp root binding differs"
        )
    root_identity = _private_directory_identity(requested, label="MIOpen cache root")
    local_tmp_root_identity = _private_directory_identity(
        local_tmp_requested, label="MIOpen node-local tmp root"
    )
    parent_identity = _node_local_tmp_parent_identity()
    if (
        local_tmp_root_identity["device"] != int(parent_identity["device"])
        or local_tmp_root_identity["device"] == root_identity["device"]
    ):
        raise NativeActivationV2RunnerError(
            "MIOpen tmp/cache filesystem separation differs"
        )
    names = {entry.name for entry in requested.iterdir()}
    if names != _expected_miopen_cache_directory_names():
        raise NativeActivationV2RunnerError("initial MIOpen cache root entries differ")
    role_directory_identities = {
        name: _private_directory_identity(
            requested / name, label=f"MIOpen cache {name}"
        )
        for name in sorted(names)
    }
    for name in sorted(names):
        if any((requested / name).iterdir()):
            raise NativeActivationV2RunnerError(
                f"initial MIOpen cache {name} is not empty"
            )
    local_tmp_names = {entry.name for entry in local_tmp_requested.iterdir()}
    if local_tmp_names != _expected_miopen_local_tmp_directory_names():
        raise NativeActivationV2RunnerError(
            "initial MIOpen node-local tmp entries differ"
        )
    local_tmp_directory_identities = {
        name: _private_directory_identity(
            local_tmp_requested / name, label=f"MIOpen node-local tmp {name}"
        )
        for name in sorted(local_tmp_names)
    }
    for name in sorted(local_tmp_names):
        if any((local_tmp_requested / name).iterdir()):
            raise NativeActivationV2RunnerError(
                f"initial MIOpen node-local tmp {name} is not empty"
            )
    launcher_local_tmp_empty_proof = _launcher_local_tmp_empty_proof(
        local_tmp_root_identity, local_tmp_directory_identities
    )
    if os.environ.get(MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV) != str(
        launcher_local_tmp_empty_proof["sha256"]
    ):
        raise NativeActivationV2RunnerError(
            "launcher node-local tmp empty proof differs"
        )
    bootstrap_user_db = requested / "launcher-bootstrap-user-db"
    bootstrap_kernel_cache = requested / "launcher-bootstrap-kernel-cache"
    inherited_miopen = {
        key: value for key, value in os.environ.items() if key.startswith("MIOPEN_")
    }
    if inherited_miopen != {
        MIOPEN_USER_DB_ENV: str(bootstrap_user_db),
        MIOPEN_KERNEL_CACHE_ENV: str(bootstrap_kernel_cache),
    } or any(os.environ.get(name) is not None for name in MIOPEN_TEMP_ENV_NAMES):
        raise NativeActivationV2RunnerError(
            "inherited MIOpen bootstrap environment differs"
        )
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as error:
        raise NativeActivationV2RunnerError("MIOpen cache WORLD4 env differs") from error
    if (
        local_rank not in range(WORLD_SIZE)
        or global_rank != local_rank
        or world_size != WORLD_SIZE
        or local_world_size != WORLD_SIZE
    ):
        raise NativeActivationV2RunnerError("MIOpen cache WORLD4 rank binding differs")
    user_db, kernel_cache, rank_tmp = _rank_cache_paths(
        requested, local_tmp_requested, local_rank
    )
    for role, path in (
        ("user-db", user_db),
        ("kernel-cache", kernel_cache),
        ("tmp", rank_tmp),
    ):
        _private_directory_identity(path, label=f"rank {local_rank} MIOpen {role}")
        if any(path.iterdir()):
            raise NativeActivationV2RunnerError(
                f"rank {local_rank} initial MIOpen {role} is not empty"
            )
    library = _certify_miopen_library_file()
    os.environ[MIOPEN_USER_DB_ENV] = str(user_db)
    os.environ[MIOPEN_KERNEL_CACHE_ENV] = str(kernel_cache)
    os.environ["TMPDIR"] = str(rank_tmp)
    return {
        "schema_version": "bernini-miopen-rank-private-cache-activation-r11",
        "root": dict(root_identity),
        "role_directory_identities": role_directory_identities,
        "local_tmp_root": dict(local_tmp_root_identity),
        "local_tmp_parent_identity": dict(parent_identity),
        "local_tmp_role_directory_identities": (
            local_tmp_directory_identities
        ),
        "launcher_local_tmp_fresh_empty_proof": dict(
            launcher_local_tmp_empty_proof
        ),
        "worker_pre_torch_local_tmp_empty_proof": {
            "root_identity": dict(local_tmp_root_identity),
            "directory_identities": local_tmp_directory_identities,
            "all_rank_directories_empty": True,
        },
        "persistent_and_local_tmp_devices_are_distinct": True,
        "rank": global_rank,
        "local_rank": local_rank,
        "user_db_path": str(user_db),
        "kernel_cache_path": str(kernel_cache),
        "tmp_path": str(rank_tmp),
        "launcher_bootstrap": {
            "user_db_path": str(bootstrap_user_db),
            "kernel_cache_path": str(bootstrap_kernel_cache),
            "torchrun_parent_tmpdir": None,
            "workers_use_bootstrap_namespace": False,
        },
        "scheduler_tmpdir_normalization": dict(
            scheduler_tmpdir_normalization
        ),
        "official_environment": {
            MIOPEN_USER_DB_ENV: str(user_db),
            MIOPEN_KERNEL_CACHE_ENV: str(kernel_cache),
            "TMPDIR": str(rank_tmp),
            "TMP": None,
            "TEMP": None,
            "TEMPDIR": None,
        },
        "exact_miopen_environment_key_allowlist": [
            MIOPEN_KERNEL_CACHE_ENV,
            MIOPEN_USER_DB_ENV,
        ],
        "initial_rank_directories_empty": True,
        "library": dict(library),
        "explicit_solver_control_environment_override_used": False,
        "matched_arms_share_cache_lookup_state": True,
        "engineering_evidence_only": True,
    }


def _certify_active_miopen_environment(initial: Mapping[str, Any]) -> None:
    if (
        _certify_scheduler_tmpdir_normalization()
        != initial.get("scheduler_tmpdir_normalization")
    ):
        raise NativeActivationV2RunnerError(
            "scheduler TMPDIR normalization changed"
        )
    expected = initial.get("official_environment")
    if not isinstance(expected, Mapping) or set(expected) != {
        MIOPEN_USER_DB_ENV,
        MIOPEN_KERNEL_CACHE_ENV,
        *MIOPEN_TEMP_ENV_NAMES,
    }:
        raise NativeActivationV2RunnerError("MIOpen initial environment receipt differs")
    if any(os.environ.get(name) != value for name, value in expected.items()):
        raise NativeActivationV2RunnerError("MIOpen cache environment changed")
    if {
        key: value for key, value in os.environ.items() if key.startswith("MIOPEN_")
    } != {
        MIOPEN_USER_DB_ENV: expected[MIOPEN_USER_DB_ENV],
        MIOPEN_KERNEL_CACHE_ENV: expected[MIOPEN_KERNEL_CACHE_ENV],
    }:
        raise NativeActivationV2RunnerError("unexpected MIOpen solver environment appeared")
    root_receipt = initial.get("root")
    if not isinstance(root_receipt, Mapping):
        raise NativeActivationV2RunnerError("MIOpen cache root receipt differs")
    root = Path(str(root_receipt.get("path")))
    if _private_directory_identity(root, label="live MIOpen cache root") != root_receipt:
        raise NativeActivationV2RunnerError("MIOpen cache root identity changed")
    expected_roles = initial.get("role_directory_identities")
    if (
        not isinstance(expected_roles, Mapping)
        or set(expected_roles) != _expected_miopen_cache_directory_names()
    ):
        raise NativeActivationV2RunnerError("MIOpen role identity receipt differs")
    for name, expected_role in expected_roles.items():
        observed_role = _private_directory_identity(
            root / name, label=f"live MIOpen cache {name}"
        )
        if observed_role != expected_role:
            raise NativeActivationV2RunnerError(
                f"MIOpen cache {name} identity changed"
            )
    local_root_receipt = initial.get("local_tmp_root")
    if not isinstance(local_root_receipt, Mapping):
        raise NativeActivationV2RunnerError(
            "MIOpen node-local tmp root receipt differs"
        )
    local_root = Path(str(local_root_receipt.get("path")))
    observed_local_root = _private_directory_identity(
        local_root, label="live MIOpen node-local tmp root"
    )
    parent_receipt = initial.get("local_tmp_parent_identity")
    observed_parent = _node_local_tmp_parent_identity()
    if (
        not isinstance(parent_receipt, Mapping)
        or observed_parent != parent_receipt
        or observed_local_root != local_root_receipt
        or observed_local_root["device"]
        != int(observed_parent["device"])
        or observed_local_root["device"] == root_receipt.get("device")
    ):
        raise NativeActivationV2RunnerError(
            "MIOpen node-local tmp root identity changed"
        )
    expected_local_roles = initial.get("local_tmp_role_directory_identities")
    if (
        not isinstance(expected_local_roles, Mapping)
        or set(expected_local_roles)
        != _expected_miopen_local_tmp_directory_names()
    ):
        raise NativeActivationV2RunnerError(
            "MIOpen node-local tmp role receipt differs"
        )
    for name, expected_role in expected_local_roles.items():
        observed_role = _private_directory_identity(
            local_root / name, label=f"live MIOpen node-local tmp {name}"
        )
        if observed_role != expected_role:
            raise NativeActivationV2RunnerError(
                f"MIOpen node-local tmp {name} identity changed"
            )
    launcher_proof = _launcher_local_tmp_empty_proof(
        local_root_receipt, expected_local_roles
    )
    if (
        launcher_proof != initial.get("launcher_local_tmp_fresh_empty_proof")
        or os.environ.get(MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV)
        != launcher_proof["sha256"]
    ):
        raise NativeActivationV2RunnerError(
            "launcher node-local tmp empty proof changed"
        )


def _expected_miopen_lock_basenames(user_db_path: Path) -> set[str]:
    try:
        parent_md5 = hashlib.md5(
            str(user_db_path).encode("utf-8"), usedforsecurity=False
        ).hexdigest()
    except TypeError:  # pragma: no cover - older Python compatibility
        parent_md5 = hashlib.md5(str(user_db_path).encode("utf-8")).hexdigest()
    return {
        f"{parent_md5}_{name}.lock" for name in MIOPEN_USER_DB_MAIN_BASENAMES
    }


def _snapshot_private_cache_tree(
    path: Path, *, role: str, user_db_path: Path, label: str
) -> Mapping[str, Any]:
    """Hash a quiescent private cache tree without following symlinks."""

    root_identity = _private_directory_identity(path, label=label)
    rows: list[Mapping[str, Any]] = []

    def visit(directory: Path, prefix: str) -> None:
        before_names = sorted(entry.name for entry in os.scandir(directory))
        for name in before_names:
            child = directory / name
            try:
                info = child.lstat()
            except OSError as error:
                raise NativeActivationV2RunnerError(
                    f"{label} entry disappeared"
                ) from error
            relative = f"{prefix}/{name}" if prefix else name
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode) or int(info.st_uid) != os.geteuid():
                raise NativeActivationV2RunnerError(
                    f"{label} entry ownership/type differs"
                )
            if stat.S_ISDIR(info.st_mode):
                expected_directory = (
                    role == "tmp"
                    and relative == "miopen-lockfiles"
                    and mode == 0o777
                )
                if not expected_directory:
                    raise NativeActivationV2RunnerError(
                        f"{label} nested directory name/mode differs"
                    )
                rows.append(
                    {
                        "path": relative,
                        "type": "directory",
                        "mode": mode,
                        "uid": int(info.st_uid),
                        "gid": int(info.st_gid),
                    }
                )
                visit(child, relative)
            elif stat.S_ISREG(info.st_mode):
                flat = "/" not in relative
                if role == "user-db":
                    valid_role_member = flat and (
                        (
                            name in MIOPEN_USER_DB_MAIN_BASENAMES
                            and mode == 0o777
                        )
                        or (
                            name in MIOPEN_USER_DB_TIME_BASENAMES
                            and mode & 0o7000 == 0
                        )
                    )
                elif role == "kernel-cache":
                    valid_role_member = (
                        flat
                        and name in MIOPEN_KERNEL_CACHE_BASENAMES
                        and mode == 0o600
                    )
                elif role == "tmp":
                    valid_role_member = (
                        relative.startswith("miopen-lockfiles/")
                        and relative.count("/") == 1
                        and name in _expected_miopen_lock_basenames(user_db_path)
                        and mode == 0o777
                    )
                else:
                    valid_role_member = False
                if info.st_nlink != 1 or not valid_role_member:
                    raise NativeActivationV2RunnerError(
                        f"{label} cache file name/mode differs"
                    )
                identity = _owned_file_identity(child, label=f"{label}/{relative}")
                rows.append(
                    {
                        "path": relative,
                        "type": "file",
                        "sha256": identity["sha256"],
                        "size": identity["size"],
                        "mode": identity["mode"],
                        "nlink": identity["nlink"],
                        "uid": int(info.st_uid),
                        "gid": int(info.st_gid),
                    }
                )
            else:
                raise NativeActivationV2RunnerError(
                    f"{label} contains a non-regular entry"
                )
        after_names = sorted(entry.name for entry in os.scandir(directory))
        if after_names != before_names:
            raise NativeActivationV2RunnerError(f"{label} changed during tree scan")

    visit(path, "")
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "root": dict(root_identity),
        "entries": rows,
        "entry_count": len(rows),
        "regular_file_count": sum(row["type"] == "file" for row in rows),
        "regular_file_bytes": sum(
            int(row.get("size", 0)) for row in rows if row["type"] == "file"
        ),
        "tree_sha256": _canonical_sha256(rows),
    }


def _stable_local_miopen_cache_snapshot(
    initial: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    _certify_active_miopen_environment(initial)
    rank = int(initial["rank"])
    paths = (
        Path(str(initial["user_db_path"])),
        Path(str(initial["kernel_cache_path"])),
    )
    roles = MIOPEN_RANK_ROLES
    snapshots = [
        _snapshot_private_cache_tree(
            path,
            role=role,
            user_db_path=paths[0],
            label=f"{label}/{path.name}",
        )
        for role, path in zip(roles, paths)
    ]
    replay = [
        _snapshot_private_cache_tree(
            path,
            role=role,
            user_db_path=paths[0],
            label=f"{label} replay/{path.name}",
        )
        for role, path in zip(roles, paths)
    ]
    if replay != snapshots:
        raise NativeActivationV2RunnerError(f"{label} cache tree is not quiescent")
    return {
        "rank": rank,
        "local_rank": int(initial["local_rank"]),
        "user_db_path": str(paths[0]),
        "kernel_cache_path": str(paths[1]),
        "trees": snapshots,
        "snapshot_sha256": _canonical_sha256(snapshots),
    }


def _gather_miopen_cache_snapshot(
    initial: Mapping[str, Any], *, torch: Any, dist: Any, label: str
) -> list[Mapping[str, Any]]:
    local_status: Mapping[str, Any]
    try:
        torch.cuda.synchronize()
        local_status = {
            "ok": True,
            "snapshot": _stable_local_miopen_cache_snapshot(initial, label=label),
        }
    except Exception as error:
        local_status = {
            "ok": False,
            "rank": initial.get("rank"),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    statuses: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(statuses, local_status)
    if any(
        not isinstance(status, Mapping) or status.get("ok") is not True
        for status in statuses
    ):
        raise NativeActivationV2RunnerError(
            f"{label} rank-local cache certification failed: {statuses}"
        )
    rows = [status["snapshot"] for status in statuses]
    if (
        any(not isinstance(row, Mapping) for row in rows)
        or [row.get("rank") for row in rows] != list(range(WORLD_SIZE))
        or [row.get("local_rank") for row in rows] != list(range(WORLD_SIZE))
        or len({row.get("user_db_path") for row in rows}) != WORLD_SIZE
        or len({row.get("kernel_cache_path") for row in rows}) != WORLD_SIZE
        or any(
            Path(str(row.get("user_db_path"))).parent
            != Path(str(initial["root"]["path"]))
            or Path(str(row.get("kernel_cache_path"))).parent
            != Path(str(initial["root"]["path"]))
            for row in rows
        )
    ):
        raise NativeActivationV2RunnerError(f"{label} WORLD4 cache topology differs")
    return [dict(row) for row in rows]


def _snapshot_node_local_tmp_tree(
    path: Path, *, user_db_path: Path, label: str
) -> Mapping[str, Any]:
    """Hash a quiescent rank-private local temp tree without following links."""

    root_identity = _private_directory_identity(path, label=label)
    root_device = int(root_identity["device"])
    rows: list[Mapping[str, Any]] = []
    entry_limit = 4096

    def visit(directory: Path, prefix: str) -> None:
        before_names = sorted(entry.name for entry in os.scandir(directory))
        for name in before_names:
            child = directory / name
            info = child.lstat()
            relative = f"{prefix}/{name}" if prefix else name
            mode = stat.S_IMODE(info.st_mode)
            if (
                stat.S_ISLNK(info.st_mode)
                or int(info.st_uid) != os.geteuid()
                or int(info.st_dev) != root_device
                or len(rows) >= entry_limit
            ):
                raise NativeActivationV2RunnerError(
                    f"{label} entry ownership/type/device/count differs"
                )
            is_miopen_lock_directory = (
                relative == "miopen-lockfiles"
                and stat.S_ISDIR(info.st_mode)
                and mode == 0o777
            )
            if stat.S_ISDIR(info.st_mode):
                if not is_miopen_lock_directory and (
                    mode & 0o077 != 0 or mode & 0o7000 != 0
                ):
                    raise NativeActivationV2RunnerError(
                        f"{label} local directory mode differs"
                    )
                rows.append(
                    {
                        "path": relative,
                        "type": "directory",
                        "mode": mode,
                        "uid": int(info.st_uid),
                        "gid": int(info.st_gid),
                        "device": int(info.st_dev),
                        "inode": int(info.st_ino),
                    }
                )
                visit(child, relative)
            elif stat.S_ISREG(info.st_mode):
                is_miopen_lock = (
                    relative.startswith("miopen-lockfiles/")
                    and relative.count("/") == 1
                    and name in _expected_miopen_lock_basenames(user_db_path)
                    and mode == 0o777
                )
                if (
                    int(info.st_nlink) != 1
                    or (
                        not is_miopen_lock
                        and (mode & 0o077 != 0 or mode & 0o7000 != 0)
                    )
                ):
                    raise NativeActivationV2RunnerError(
                        f"{label} local file mode/link differs"
                    )
                identity = _owned_file_identity(
                    child, label=f"{label}/{relative}"
                )
                if (
                    identity["device"] != root_device
                    or identity["inode"] != int(info.st_ino)
                ):
                    raise NativeActivationV2RunnerError(
                        f"{label} local file identity differs"
                    )
                rows.append(
                    {
                        "path": relative,
                        "type": "file",
                        **dict(identity),
                        "uid": int(info.st_uid),
                        "gid": int(info.st_gid),
                    }
                )
            else:
                raise NativeActivationV2RunnerError(
                    f"{label} contains a non-regular local entry"
                )
        after_names = sorted(entry.name for entry in os.scandir(directory))
        if after_names != before_names:
            raise NativeActivationV2RunnerError(
                f"{label} changed during local tree scan"
            )

    visit(path, "")
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "root": dict(root_identity),
        "entries": rows,
        "entry_count": len(rows),
        "regular_file_count": sum(row["type"] == "file" for row in rows),
        "regular_file_bytes": sum(
            int(row.get("size", 0)) for row in rows if row["type"] == "file"
        ),
        "tree_sha256": _canonical_sha256(rows),
    }


def _stable_local_tmp_snapshot(
    initial: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    _certify_active_miopen_environment(initial)
    rank = int(initial["rank"])
    path = Path(str(initial["tmp_path"]))
    user_db_path = Path(str(initial["user_db_path"]))
    snapshot = _snapshot_node_local_tmp_tree(
        path, user_db_path=user_db_path, label=f"{label}/{path.name}"
    )
    replay = _snapshot_node_local_tmp_tree(
        path, user_db_path=user_db_path, label=f"{label} replay/{path.name}"
    )
    if replay != snapshot:
        raise NativeActivationV2RunnerError(
            f"{label} node-local tmp tree is not quiescent"
        )
    return {
        "rank": rank,
        "local_rank": int(initial["local_rank"]),
        "tmp_path": str(path),
        "tree": snapshot,
        "snapshot_sha256": _canonical_sha256(snapshot),
    }


def _validate_local_tmp_world4_snapshot_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    initial: Mapping[str, Any],
    label: str,
) -> tuple[list[Mapping[str, Any]], str]:
    """Validate the full historical WORLD4 local-tmp snapshot ABI."""

    if len(rows) != WORLD_SIZE:
        raise NativeActivationV2RunnerError(f"{label} WORLD4 row count differs")
    local_root = Path(str(initial["local_tmp_root"]["path"]))
    expected_roles = initial.get("local_tmp_role_directory_identities")
    expected_names = _expected_miopen_local_tmp_directory_names()
    if not isinstance(expected_roles, Mapping) or set(expected_roles) != expected_names:
        raise NativeActivationV2RunnerError(f"{label} role identity receipt differs")
    copied: list[Mapping[str, Any]] = []
    row_keys = {"rank", "local_rank", "tmp_path", "tree", "snapshot_sha256"}
    tree_keys = {
        "root",
        "entries",
        "entry_count",
        "regular_file_count",
        "regular_file_bytes",
        "tree_sha256",
    }
    directory_keys = {
        "path",
        "type",
        "mode",
        "uid",
        "gid",
        "device",
        "inode",
    }
    file_keys = directory_keys | {"sha256", "size", "nlink"}
    for rank, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != row_keys:
            raise NativeActivationV2RunnerError(f"{label} row schema differs")
        row = dict(raw_row)
        tmp_path = local_root / f"rank-{rank}"
        tree = row.get("tree")
        if (
            row.get("rank") != rank
            or row.get("local_rank") != rank
            or row.get("tmp_path") != str(tmp_path)
            or not isinstance(tree, Mapping)
            or set(tree) != tree_keys
            or tree.get("root") != expected_roles[f"rank-{rank}"]
        ):
            raise NativeActivationV2RunnerError(f"{label} row topology differs")
        entries = tree.get("entries")
        if not isinstance(entries, list) or len(entries) > 4096:
            raise NativeActivationV2RunnerError(f"{label} entry list differs")
        observed_paths: list[str] = []
        observed_types: dict[str, str] = {}
        regular_file_count = 0
        regular_file_bytes = 0
        root_device = int(expected_roles[f"rank-{rank}"]["device"])
        expected_lock_basenames = _expected_miopen_lock_basenames(
            Path(str(initial["root"]["path"])) / f"rank-{rank}-user-db"
        )
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise NativeActivationV2RunnerError(f"{label} entry schema differs")
            entry_type = entry.get("type")
            expected_keys = directory_keys if entry_type == "directory" else file_keys
            relative = entry.get("path")
            numeric_keys = ("mode", "uid", "gid", "device", "inode")
            if (
                entry_type not in {"directory", "file"}
                or set(entry) != expected_keys
                or not isinstance(relative, str)
                or not relative
                or relative.startswith("/")
                or any(part in {"", ".", ".."} for part in relative.split("/"))
                or any(type(entry.get(key)) is not int for key in numeric_keys)
                or entry.get("uid") != os.geteuid()
                or entry.get("device") != root_device
                or int(entry.get("inode", 0)) <= 0
                or int(entry.get("mode", -1)) not in range(0o10000)
            ):
                raise NativeActivationV2RunnerError(f"{label} entry identity differs")
            parent = relative.rpartition("/")[0]
            if parent and observed_types.get(parent) != "directory":
                raise NativeActivationV2RunnerError(
                    f"{label} nested entry parent differs"
                )
            mode = int(entry["mode"])
            if entry_type == "directory":
                is_miopen_lock_directory = (
                    relative == "miopen-lockfiles" and mode == 0o777
                )
                if not is_miopen_lock_directory and (
                    mode & 0o077 != 0 or mode & 0o7000 != 0
                ):
                    raise NativeActivationV2RunnerError(
                        f"{label} local directory mode differs"
                    )
            else:
                is_miopen_lock = (
                    relative.startswith("miopen-lockfiles/")
                    and relative.count("/") == 1
                    and relative.rsplit("/", 1)[1] in expected_lock_basenames
                    and mode == 0o777
                )
                if (
                    type(entry.get("size")) is not int
                    or int(entry["size"]) < 0
                    or entry.get("nlink") != 1
                    or not isinstance(entry.get("sha256"), str)
                    or _SHA256.fullmatch(str(entry["sha256"])) is None
                    or (
                        not is_miopen_lock
                        and (mode & 0o077 != 0 or mode & 0o7000 != 0)
                    )
                ):
                    raise NativeActivationV2RunnerError(
                        f"{label} regular file identity differs"
                    )
                regular_file_count += 1
                regular_file_bytes += int(entry["size"])
            observed_paths.append(relative)
            observed_types[relative] = str(entry_type)
        if (
            observed_paths != sorted(set(observed_paths))
            or tree.get("entry_count") != len(entries)
            or tree.get("regular_file_count") != regular_file_count
            or tree.get("regular_file_bytes") != regular_file_bytes
            or tree.get("tree_sha256") != _canonical_sha256(entries)
            or row.get("snapshot_sha256") != _canonical_sha256(tree)
        ):
            raise NativeActivationV2RunnerError(f"{label} snapshot digest differs")
        copied.append(row)
    if len({row["tmp_path"] for row in copied}) != WORLD_SIZE:
        raise NativeActivationV2RunnerError(f"{label} tmp paths are not unique")
    return copied, _canonical_sha256(copied)


def _gather_local_tmp_snapshot(
    initial: Mapping[str, Any], *, torch: Any, dist: Any, label: str
) -> list[Mapping[str, Any]]:
    try:
        torch.cuda.synchronize()
        local_status: Mapping[str, Any] = {
            "ok": True,
            "snapshot": _stable_local_tmp_snapshot(initial, label=label),
        }
    except Exception as error:
        local_status = {
            "ok": False,
            "rank": initial.get("rank"),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    statuses: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(statuses, local_status)
    if any(
        not isinstance(status, Mapping) or status.get("ok") is not True
        for status in statuses
    ):
        raise NativeActivationV2RunnerError(
            f"{label} rank-local tmp certification failed: {statuses}"
        )
    rows, _ = _validate_local_tmp_world4_snapshot_rows(
        [status["snapshot"] for status in statuses],
        initial=initial,
        label=label,
    )
    return rows


def _gather_miopen_bootstrap_snapshot(
    initial: Mapping[str, Any], *, dist: Any, label: str
) -> Mapping[str, Any]:
    bootstrap = initial.get("launcher_bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise NativeActivationV2RunnerError("MIOpen bootstrap receipt differs")
    paths = (
        Path(str(bootstrap.get("user_db_path"))),
        Path(str(bootstrap.get("kernel_cache_path"))),
    )
    try:
        snapshots = [
            _snapshot_private_cache_tree(
                path,
                role=role,
                user_db_path=paths[0],
                label=f"{label}/{path.name}",
            )
            for role, path in zip(MIOPEN_BOOTSTRAP_ROLES, paths)
        ]
        local_status: Mapping[str, Any] = {
            "ok": True,
            "snapshot": {
                "paths": [str(path) for path in paths],
                "trees": snapshots,
                "snapshot_sha256": _canonical_sha256(snapshots),
            },
        }
    except Exception as error:
        local_status = {
            "ok": False,
            "rank": initial.get("rank"),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    statuses: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(statuses, local_status)
    if any(
        not isinstance(status, Mapping) or status.get("ok") is not True
        for status in statuses
    ):
        raise NativeActivationV2RunnerError(
            f"{label} bootstrap certification failed: {statuses}"
        )
    snapshots = [status["snapshot"] for status in statuses]
    if any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
        raise NativeActivationV2RunnerError(
            f"{label} bootstrap snapshot differs across WORLD4"
        )
    return dict(snapshots[0])


def _certify_loaded_miopen_library(*, maps_path: Path = Path("/proc/self/maps")) -> Mapping[str, Any]:
    identity = _certify_miopen_library_file()
    try:
        lines = maps_path.read_text(encoding="utf-8", errors="strict").splitlines()
    except OSError as error:
        raise NativeActivationV2RunnerError("cannot inspect loaded MIOpen library") from error
    observed = {
        line.split(maxsplit=5)[-1]
        for line in lines
        if len(line.split(maxsplit=5)) == 6
        and "libMIOpen.so" in line.split(maxsplit=5)[-1]
    }
    if observed != {str(MIOPEN_LIBRARY_PATH)}:
        raise NativeActivationV2RunnerError("loaded MIOpen library origin differs")
    return {**dict(identity), "loaded_from_proc_maps": True}


def _certify_serialized_host_load_lock() -> Mapping[str, Any]:
    """Require the one-node frozen flock used by every renderer load."""

    if os.environ.get("NATIVE_SERIALIZED_HOST_LOAD_REQUIRED") != "1":
        raise NativeActivationV2RunnerError(
            "serialized host-load requirement is not enabled"
        )
    value = os.environ.get("NATIVE_V_AXIS_LOAD_LOCK")
    if not isinstance(value, str) or not value:
        raise NativeActivationV2RunnerError("serialized host-load lock is absent")
    requested = Path(value)
    if not requested.is_absolute() or requested.is_symlink():
        raise NativeActivationV2RunnerError(
            "serialized host-load lock path differs"
        )
    path = requested.resolve(strict=True)
    if path != requested or path.is_symlink() or not path.is_file():
        raise NativeActivationV2RunnerError(
            "serialized host-load lock identity differs"
        )
    identity = _owned_file_identity(path, label="serialized host-load lock")
    if identity["size"] != 0 or identity["mode"] != 0o444 or identity["nlink"] != 1:
        raise NativeActivationV2RunnerError(
            "serialized host-load lock must be empty/frozen/one-link"
        )
    return {
        "required": True,
        "path": str(path),
        **dict(identity),
    }


def _callable_token(value: Any) -> Any:
    owner = getattr(value, "__self__", None)
    function = getattr(value, "__func__", None)
    return (owner, function) if function is not None else value


def _callable_row(owner: Any, name: str) -> tuple[Any, Mapping[str, Any]]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise NativeActivationV2RunnerError(f"runtime callable {name} is absent")
    try:
        instance_override = name in vars(owner)
    except TypeError as error:
        raise NativeActivationV2RunnerError(f"cannot inspect {name} owner") from error
    function = getattr(value, "__func__", value)
    return _callable_token(value), {
        "name": name,
        "module": str(getattr(function, "__module__", "")),
        "qualname": str(getattr(function, "__qualname__", "")),
        "instance_override": instance_override,
    }


def _capture_call_surface(diffusion: Any) -> _CallSurfaceSnapshot:
    transformer = getattr(diffusion, "transformer", None)
    scheduler = getattr(diffusion, "scheduler", None)
    owners = (
        (diffusion, "sample"),
        (diffusion, "shared_step"),
        (transformer, "patch_vae_latent"),
        (scheduler, "step"),
    )
    rows = tuple(_callable_row(owner, name) for owner, name in owners)
    return _CallSurfaceSnapshot(
        tokens=tuple(row[0] for row in rows),
        receipt={"all_instance_overrides_absent": not any(row[1]["instance_override"] for row in rows),
                 "callables": [dict(row[1]) for row in rows]},
    )


def _certify_call_surface(
    diffusion: Any, before: _CallSurfaceSnapshot, *, label: str
) -> Mapping[str, Any]:
    after = _capture_call_surface(diffusion)
    if before.tokens != after.tokens or before.receipt != after.receipt:
        raise NativeActivationV2RunnerError(f"{label} callable surface changed")
    if before.receipt.get("all_instance_overrides_absent") is not True:
        raise NativeActivationV2RunnerError(f"{label} entered with instance override")
    return dict(after.receipt)


def _all_rank_object(value: Any, *, dist: Any, label: str) -> list[Any]:
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, value)
    if any(row != rows[0] for row in rows[1:]):
        raise NativeActivationV2RunnerError(f"{label} differs across WORLD4")
    return rows


def _collective_owned_file_identity(
    path: Path, *, dist: Any, label: str
) -> Mapping[str, Any]:
    """Turn a rank-local owned-read failure into one synchronized WORLD4 error."""

    try:
        local_status: Mapping[str, Any] = {
            "ok": True,
            "identity": _owned_file_identity(path, label=label),
        }
    except Exception as error:
        local_status = {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    statuses: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(statuses, local_status)
    if any(
        not isinstance(status, Mapping) or status.get("ok") is not True
        for status in statuses
    ):
        raise NativeActivationV2RunnerError(
            f"{label} rank-local owned read failed: {statuses}"
        )
    identities = [status["identity"] for status in statuses]
    if any(identity != identities[0] for identity in identities[1:]):
        raise NativeActivationV2RunnerError(f"{label} differs across WORLD4")
    return dict(identities[0])


def _collective_exact_directory_names(
    path: Path, *, expected: set[str], dist: Any, label: str
) -> None:
    try:
        names = sorted(entry.name for entry in path.iterdir())
        local_status: Mapping[str, Any] = {"ok": True, "names": names}
    except Exception as error:
        local_status = {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    statuses: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(statuses, local_status)
    if any(
        not isinstance(status, Mapping)
        or status.get("ok") is not True
        or status.get("names") != sorted(expected)
        for status in statuses
    ):
        raise NativeActivationV2RunnerError(
            f"{label} exact directory entries differ: {statuses}"
        )


def _certify_world4_sp4(
    *, distributed: Any, torch: Any, dist: Any, parallel_state: Any
) -> Mapping[str, Any]:
    """Certify one-node rank/local-device order and the live Ulysses group."""

    group = getattr(parallel_state, "ulysses_group", None)
    sp_rank = getattr(parallel_state, "ulysses_rank", None)
    if (
        getattr(parallel_state, "ulysses_enabled", None) is not True
        or getattr(parallel_state, "ulysses_size", None) != WORLD_SIZE
        or type(sp_rank) is not int
        or not 0 <= sp_rank < WORLD_SIZE
        or dist.get_world_size(group) != WORLD_SIZE
        or dist.get_rank(group) != sp_rank
        or str(dist.get_backend(group)).lower() != "nccl"
        or torch.cuda.device_count() != WORLD_SIZE
        or torch.cuda.current_device() != distributed.local_rank
        or os.environ.get("ROCR_VISIBLE_DEVICES") != "0,1,2,3"
        or os.environ.get("LOCAL_WORLD_SIZE") != str(WORLD_SIZE)
        or os.environ.get("SLURM_JOB_ID") != AUTHORIZED_SLURM_JOB_ID
        or socket.gethostname().split(".", 1)[0] != AUTHORIZED_HOSTNAME
        or any(
            os.environ.get(name) is not None
            for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
        )
    ):
        raise NativeActivationV2RunnerError("live WORLD4/SP4 device contract differs")
    sp_members: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(sp_members, int(distributed.rank), group=group)
    if sp_members != list(range(WORLD_SIZE)):
        raise NativeActivationV2RunnerError("SP4 global-rank order differs")
    local = {
        "rank": int(distributed.rank),
        "local_rank": int(distributed.local_rank),
        "current_device": int(torch.cuda.current_device()),
        "device_count": int(torch.cuda.device_count()),
        "hostname": socket.gethostname(),
        "sp_rank": int(sp_rank),
    }
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, local)
    if (
        any(not isinstance(row, Mapping) for row in rows)
        or [row["rank"] for row in rows] != list(range(WORLD_SIZE))
        or [row["local_rank"] for row in rows] != list(range(WORLD_SIZE))
        or [row["current_device"] for row in rows] != list(range(WORLD_SIZE))
        or [row["sp_rank"] for row in rows] != list(range(WORLD_SIZE))
        or any(row["device_count"] != WORLD_SIZE for row in rows)
        or len({row["hostname"] for row in rows}) != 1
    ):
        raise NativeActivationV2RunnerError("WORLD4 rank/device topology differs")
    return {
        "authorized_slurm_job_id": AUTHORIZED_SLURM_JOB_ID,
        "authorized_hostname": AUTHORIZED_HOSTNAME,
        "world_size": WORLD_SIZE,
        "sequence_parallel_size": WORLD_SIZE,
        "backend": "nccl",
        "sp_ordered_global_ranks": sp_members,
        "rank_device_rows": rows,
        "rocr_visible_devices": "0,1,2,3",
        "reserved_gpu_indices": [0, 1, 2, 3],
        "intentionally_idle_gpu_indices": [4, 5, 6, 7],
        "arms_execute_strictly_serial": True,
        "one_node": True,
    }


def _rank_zero_only_load_rows(
    *, dist: Any, rank: int, role: str, label: str
) -> list[Mapping[str, Any]]:
    local = {"rank": rank, role: rank == 0}
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, local)
    expected = [
        {"rank": expected_rank, role: expected_rank == 0}
        for expected_rank in range(WORLD_SIZE)
    ]
    if rows != expected:
        raise NativeActivationV2RunnerError(f"{label} load roles differ")
    return [dict(row) for row in rows]


@contextmanager
def _all_rank_t5_constructor_bypass(
    *,
    t5_encoder_class: Any,
    checkpoint: Path,
    dtype: Any,
    placeholder_factory: Callable[[], Any],
) -> Any:
    """Skip only the unused T5 constructor after rank0 prompt broadcast.

    The class method is restored in ``finally`` and the exact expected ABI is
    accepted once.  A/B use this identical construction seam; neither arm
    changes the official denoiser or scheduler because of this memory guard.
    """

    own = vars(t5_encoder_class).get("from_pretrained")
    had_own = "from_pretrained" in vars(t5_encoder_class)
    audit = {"call_count": 0, "placeholder": None}

    def bypassed(cls: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            cls is not t5_encoder_class
            or len(args) != 1
            or str(args[0]) != str(checkpoint)
            or kwargs != {"subfolder": "text_encoder", "torch_dtype": dtype}
        ):
            raise NativeActivationV2RunnerError("arm T5 constructor ABI differs")
        audit["call_count"] += 1
        if audit["call_count"] != 1:
            raise NativeActivationV2RunnerError("arm renderer requested T5 repeatedly")
        audit["placeholder"] = placeholder_factory()
        return audit["placeholder"]

    setattr(t5_encoder_class, "from_pretrained", classmethod(bypassed))
    try:
        yield audit
    finally:
        if had_own:
            setattr(t5_encoder_class, "from_pretrained", own)
        else:
            delattr(t5_encoder_class, "from_pretrained")
    if audit["call_count"] != 1 or audit["placeholder"] is None:
        raise NativeActivationV2RunnerError("arm T5 bypass was not exercised")


def _sampling_contract(native: Any, *, seed: int) -> Mapping[str, Any]:
    value = native.native_sampling_contract("rv2v", steps=40, seed=seed)
    value.update(
        {
            "guidance_mode": "v2v_apg",
            "omega_img": 4.5,
            "omega_txt": 4.0,
            "flow_shift": 5.0,
            "eta": 0.5,
            "norm_threshold": (50.0, 50.0),
            "momentum": 0.0,
        }
    )
    expected = {
        "num_frames": 81,
        "num_inference_steps": 40,
        "guidance_mode": "v2v_apg",
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
    if value != expected:
        raise NativeActivationV2RunnerError("native exact40 sampling contract differs")
    return value


def _materialize_source_references(
    case: Any,
    *,
    authority: Any,
    activation: Any,
    native: Any,
    source_audit: Any,
    materialize_vae: Any,
    autoencoder_class: Any,
    vae_encode: Callable[..., Any],
    checkpoint: Path,
    torch: Any,
    dist: Any,
    device: Any,
) -> tuple[Any, tuple[Any, ...], Mapping[str, Any]]:
    """Encode the full source and four RGB frames on rank zero only."""

    distributed = native.legacy.inference_distributed_contract()
    source_seal, source_bytes = activation._seal_plain_file_v2(
        case.source_video_path,
        label=f"{case.case_id} live source before VAE",
        retain_bytes=True,
    )
    if source_seal.sha256 != case.source_sha256 or source_bytes is None:
        raise NativeActivationV2RunnerError(f"{case.case_id} source bytes differ")
    latent_shape = tuple(case.full_source_latent_geometry)
    reference_shape = tuple(case.reference_latent_geometry)
    bucket_hw = (latent_shape[-2] * 8, latent_shape[-1] * 8)
    status: list[Any] = [None]
    if distributed.rank == 0:
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"activation-v2-{case.case_id}-source-"
            ) as temporary:
                snapshot = Path(temporary) / "source.mp4"
                snapshot.write_bytes(source_bytes)
                if _sha256_file(snapshot) != case.source_sha256:
                    raise NativeActivationV2RunnerError(
                        f"{case.case_id} private source snapshot differs"
                    )
                raw_frames, reported_fps, input_hw = (
                    materialize_vae._decode_exact_video(snapshot)
                )
                source_pixels, metadata, source_sha = (
                    source_audit.prepare_hashed_source_snapshot(snapshot)
                )
            if (
                source_sha != case.source_sha256
                or tuple(source_pixels.shape) != (1, 3, 81, *bucket_hw)
                or metadata.get("frame_count") != 81
                or float(metadata.get("fps", -1.0)) != 25.0
                or tuple(metadata.get("source_derived_bucket_hw", ())) != bucket_hw
                or len(raw_frames) != 81
                or float(reported_fps) != 25.0
                or tuple(input_hw) != activation.EXPECTED_SOURCE_INPUT_HW[case.case_id]
            ):
                raise NativeActivationV2RunnerError(
                    f"{case.case_id} live source geometry differs"
                )
            raw_rgb_sha = tuple(
                _raw_rgb_sha256(raw_frames[index], expected_hw=tuple(input_hw))
                for index in activation.REFERENCE_RGB_INDICES
            )
            full_preprocessed_sha = activation.safe_core.tensor_content_sha256_v1(
                source_pixels
            )
            preprocessed_sha = tuple(
                activation.safe_core.tensor_content_sha256_v1(
                    source_pixels[:, :, index : index + 1].contiguous()
                )
                for index in activation.REFERENCE_RGB_INDICES
            )
            vae = autoencoder_class.from_pretrained(
                str(checkpoint),
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            vae.eval().requires_grad_(False).to(device)
            pixels_device = source_pixels.to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                source_latent = vae_encode(vae, pixels_device).float().contiguous()
                references = tuple(
                    vae_encode(
                        vae,
                        pixels_device[:, :, index : index + 1].contiguous(),
                    )
                    .float()
                    .contiguous()
                    for index in activation.REFERENCE_RGB_INDICES
                )
            del pixels_device, source_pixels, vae
            torch.cuda.empty_cache()
            status[0] = {
                "ok": True,
                "raw_rgb_sha256": list(raw_rgb_sha),
                "preprocessed_sha256": list(preprocessed_sha),
                "full_preprocessed_sha256": full_preprocessed_sha,
                "input_hw": list(input_hw),
            }
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    else:
        source_latent = torch.zeros(latent_shape, device=device, dtype=torch.float32)
        references = tuple(
            torch.zeros(reference_shape, device=device, dtype=torch.float32)
            for _ in range(4)
        )
    dist.broadcast_object_list(status, src=0)
    if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
        raise NativeActivationV2RunnerError(
            f"{case.case_id} rank-zero source/VAE failed: {status[0]}"
        )
    broadcasts = {
        "source": native._broadcast_condition_from_rank_zero(
            source_latent,
            label=f"{case.case_id}_source",
            world_size=WORLD_SIZE,
        ),
        "references": [
            native._broadcast_condition_from_rank_zero(
                value,
                label=f"{case.case_id}_reference_{index}",
                world_size=WORLD_SIZE,
            )
            for index, value in zip(activation.REFERENCE_RGB_INDICES, references)
        ],
    }
    source_latent = source_latent.detach().contiguous()
    references = tuple(value.detach().contiguous() for value in references)
    reference_receipt = activation.validate_reference_receipt_v2(
        authority,
        case_id=case.case_id,
        source_video_latent=source_latent,
        source_reference_latents=references,
    )
    if (
        tuple(status[0]["raw_rgb_sha256"])
        != tuple(reference_receipt.reference_raw_rgb_sha256)
        or tuple(status[0]["preprocessed_sha256"])
        != tuple(reference_receipt.reference_preprocessed_sha256)
        or status[0]["full_preprocessed_sha256"]
        != reference_receipt.source_preprocessed_sha256
    ):
        raise NativeActivationV2RunnerError(
            f"{case.case_id} live RGB/reference receipt differs"
        )
    source_after, _ = activation._seal_plain_file_v2(
        case.source_video_path,
        label=f"{case.case_id} live source after VAE",
        retain_bytes=False,
    )
    if source_after != source_seal:
        raise NativeActivationV2RunnerError(f"{case.case_id} source changed")
    identities = {
        "source": native._all_rank_tensor_identity(
            source_latent,
            label=f"{case.case_id}_source_latent",
            world_size=WORLD_SIZE,
        ),
        "references": [
            native._all_rank_tensor_identity(
                value,
                label=f"{case.case_id}_reference_{index}",
                world_size=WORLD_SIZE,
            )
            for index, value in zip(activation.REFERENCE_RGB_INDICES, references)
        ],
        "rank_zero_broadcasts": broadcasts,
        "four_independent_source_rgb_frame_vae_calls": True,
        "full_source_latent_slicing_used_for_references": False,
        "all_rank_vae_load_roles": _rank_zero_only_load_rows(
            dist=dist,
            rank=distributed.rank,
            role="vae_loaded",
            label=f"{case.case_id} VAE load role",
        ),
    }
    return source_latent, references, identities


def _materialize_prompts(
    case: Any,
    *,
    authority: Any,
    activation: Any,
    native: Any,
    prompt_builder: Any,
    prompt_clean: Callable[[str], str],
    tokenizer_class: Any,
    renderer_config_class: Any,
    renderer_model_class: Any,
    checkpoint: Path,
    bernini_root: Path,
    torch: Any,
    dist: Any,
    device: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Encode low/high/negative prompts on rank zero, then broadcast."""

    distributed = native.legacy.inference_distributed_contract()
    low_text = prompt_builder.build_mode_native_prompt(
        "low-vr2v", case.action_caption, prompt_cleaner=prompt_clean
    )
    high_text = prompt_builder.build_mode_native_prompt(
        "high-r2v4", case.action_caption, prompt_cleaner=prompt_clean
    )
    rendered = {
        "low_action": low_text,
        "high_action": high_text,
        "negative": native.legacy.DEFAULT_NEGATIVE_PROMPT,
    }
    status: list[Any] = [None]
    if distributed.rank == 0:
        try:
            tokenizer = tokenizer_class.from_pretrained(
                str(checkpoint),
                subfolder="tokenizer",
                **native.legacy.tokenizer_load_kwargs(),
            )
            tokenized = {
                "low_action": native.legacy._tokenize_training_prompt(
                    tokenizer, low_text
                ),
                "high_action": native.legacy._tokenize_training_prompt(
                    tokenizer, high_text
                ),
                "negative": native.legacy._tokenize_renderer_negative(
                    tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
                ),
            }
            token_rows = {
                name: {
                    "token_ids_sha256": activation.safe_core.tensor_content_sha256_v1(
                        pair[0]
                    ),
                    "attention_mask_sha256": activation.safe_core.tensor_content_sha256_v1(
                        pair[1]
                    ),
                }
                for name, pair in tokenized.items()
            }
            config = renderer_config_class.from_pretrained(
                str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
                local_files_only=True,
                **native.legacy.inference_renderer_config_overrides(checkpoint),
            )
            config.dtype = torch.bfloat16
            native.legacy.trainer.validate_renderer_config_mapping(
                config.to_dict(), checkpoint
            )
            model = renderer_model_class(config)
            model.eval().requires_grad_(False)
            model.t5_text_encoder.to(device)
            prompt_bank: dict[str, Any] = {}
            with torch.inference_mode():
                for name, (ids, mask) in tokenized.items():
                    prompt_bank[name] = (
                        model.encode_prompt(ids.to(device), mask.to(device))
                        .detach()
                        .contiguous()
                    )
            model.t5_text_encoder.to("cpu")
            del tokenizer, tokenized, model
            gc.collect()
            torch.cuda.empty_cache()
            status[0] = {"ok": True, "token_rows": token_rows}
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    else:
        prompt_bank = {
            name: torch.zeros(
                (1, 512, 4096), device=device, dtype=torch.bfloat16
            )
            for name in ("low_action", "high_action", "negative")
        }
    dist.broadcast_object_list(status, src=0)
    if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
        raise NativeActivationV2RunnerError(
            f"{case.case_id} rank-zero prompt encoding failed: {status[0]}"
        )
    for name in ("low_action", "high_action", "negative"):
        dist.broadcast(prompt_bank[name], src=0)
        prompt_bank[name] = prompt_bank[name].detach().contiguous()
    receipt = activation.validate_prompt_receipt_v2(
        authority,
        case_id=case.case_id,
        low_action_prompt_embeds=prompt_bank["low_action"],
        high_action_prompt_embeds=prompt_bank["high_action"],
        negative_prompt_embeds=prompt_bank["negative"],
    )
    expected_rendered = tuple(
        hashlib.sha256(rendered[name].encode("utf-8")).hexdigest()
        for name in ("low_action", "high_action", "negative")
    )
    token_rows = status[0]["token_rows"]
    expected_tokens = tuple(
        token_rows[name]["token_ids_sha256"]
        for name in ("low_action", "high_action", "negative")
    )
    expected_masks = tuple(
        token_rows[name]["attention_mask_sha256"]
        for name in ("low_action", "high_action", "negative")
    )
    if (
        tuple(receipt.rendered_text_sha256) != expected_rendered
        or tuple(receipt.token_ids_sha256) != expected_tokens
        or tuple(receipt.attention_mask_sha256) != expected_masks
    ):
        raise NativeActivationV2RunnerError(
            f"{case.case_id} live prompt/token receipt differs"
        )
    identities = {
        name: native._all_rank_tensor_identity(
            prompt_bank[name],
            label=f"{case.case_id}_{name}_prompt",
            world_size=WORLD_SIZE,
        )
        for name in ("low_action", "high_action", "negative")
    }
    identities["rendered_text_sha256"] = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name, value in rendered.items()
    }
    identities["rank0_only_text_encoder_load_and_encode"] = True
    identities["all_rank_prompt_byte_identity"] = True
    identities["all_rank_text_encoder_load_roles"] = _rank_zero_only_load_rows(
        dist=dist,
        rank=distributed.rank,
        role="real_text_encoder_loaded",
        label=f"{case.case_id} text encoder",
    )
    return prompt_bank, identities


def _load_fresh_arm_renderer(
    *,
    native: Any,
    renderer_config_class: Any,
    renderer_model_class: Any,
    t5_encoder_class: Any,
    bernini_root: Path,
    checkpoint: Path,
    torch: Any,
    dist: Any,
    device: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    config = renderer_config_class.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    if float(config.shift) != 5.0 or config.use_unipc is not True:
        raise NativeActivationV2RunnerError("fresh arm renderer config differs")
    bypass_audit: dict[str, Any] = {}

    def factory(value: Any) -> Any:
        nonlocal bypass_audit
        with _all_rank_t5_constructor_bypass(
            t5_encoder_class=t5_encoder_class,
            checkpoint=checkpoint,
            dtype=torch.bfloat16,
            placeholder_factory=torch.nn.Identity,
        ) as audit:
            model = renderer_model_class(value)
        bypass_audit = dict(audit)
        if model.t5_text_encoder is not audit["placeholder"]:
            raise NativeActivationV2RunnerError("arm renderer retained unexpected T5")
        model.t5_text_encoder = None
        return model

    model = native._load_frozen_renderer_gpu_resident_serialized(
        factory, config, device
    )
    dist.barrier()
    if (
        bypass_audit.get("call_count") != 1
        or bypass_audit.get("placeholder") is None
        or model.t5_text_encoder is not None
        or model.training
        or any(parameter.requires_grad for parameter in model.parameters())
    ):
        raise NativeActivationV2RunnerError("fresh arm model freeze/T5 closure differs")
    return model, config, {
        "fresh_model_and_scheduler_instance": True,
        "unused_t5_deserialization_bypassed_all_ranks": True,
        "constructor_bypass_scope": "UMT5EncoderModel.from_pretrained only",
        "bypass_call_count": 1,
        "provided_prompt_embeddings_materialized_rank0_and_broadcast_world4": True,
        "official_denoiser_scheduler_surface_changed_by_bypass": False,
    }


def _rng_receipt(torch: Any, *, device: Any) -> Mapping[str, Any]:
    cpu = torch.get_rng_state().detach().cpu().contiguous()
    cuda = torch.cuda.get_rng_state(device).detach().cpu().contiguous()
    return {
        "cpu_rng_sha256": hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest(),
        "cuda_rng_sha256": hashlib.sha256(cuda.numpy().tobytes(order="C")).hexdigest(),
    }


def _run_one_arm(
    *,
    case: Any,
    arm: str,
    authority: Any,
    source_latent: Any,
    references: Sequence[Any],
    prompts: Mapping[str, Any],
    activation: Any,
    native: Any,
    sampler_contract: Any,
    strong_freeze: Callable[[Any], Mapping[str, Any]],
    renderer_config_class: Any,
    renderer_model_class: Any,
    t5_encoder_class: Any,
    wan_diffusion: Any,
    bernini_root: Path,
    checkpoint: Path,
    bernini_revision: str,
    torch: Any,
    dist: Any,
    device: Any,
) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any]]:
    if arm not in ARM_ORDER_BY_CASE[case.case_id]:
        raise NativeActivationV2RunnerError("arm is outside case preregistration")
    model, config, load_receipt = _load_fresh_arm_renderer(
        native=native,
        renderer_config_class=renderer_config_class,
        renderer_model_class=renderer_model_class,
        t5_encoder_class=t5_encoder_class,
        bernini_root=bernini_root,
        checkpoint=checkpoint,
        torch=torch,
        dist=dist,
        device=device,
    )
    diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
    wan_sha = sampler_contract.validate_runtime_source_identity(
        bernini_commit=bernini_revision,
        wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
    )
    sampler_contract._validate_scheduler_contract(
        diffusion.scheduler, expected_flow_shift=5.0
    )
    if diffusion.transformer_2 is not None:
        raise NativeActivationV2RunnerError("runner requires single DiT transformer_1")
    surface_before = _capture_call_surface(diffusion)
    if surface_before.receipt.get("all_instance_overrides_absent") is not True:
        raise NativeActivationV2RunnerError("arm entered with stacked override")
    freeze_before = strong_freeze(model)
    torch.manual_seed(case.seed)
    torch.cuda.manual_seed_all(case.seed)
    rng_before = _rng_receipt(torch, device=device)
    sample_kwargs = {
        "prompt_embeds": prompts["low_action"],
        "uncond_prompt_embeds": prompts["negative"],
        "multi_video_vae_latents": [source_latent],
        "multi_image_vae_latents": list(references),
        "image_vae_latents": None,
        "width": int(case.full_source_latent_geometry[-1] * 8),
        "height": int(case.full_source_latent_geometry[-2] * 8),
        "device": device,
        **_sampling_contract(native, seed=case.seed),
    }
    patch = None
    capability = None
    if arm == ARM_LOCAL:
        capability = activation.mint_native_local_execution_capability_v2(
            authority,
            case_id=case.case_id,
            source_video_latent=source_latent,
            source_reference_latents=references,
            low_action_prompt_embeds=prompts["low_action"],
            high_action_prompt_embeds=prompts["high_action"],
            negative_prompt_embeds=prompts["negative"],
        )
        patch = activation.LocalOracleNativeBranchRuntimePatchV2(
            diffusion,
            config=activation.native_runtime.NativeBranchHomotopyRuntimeConfig(
                target_latent_shape=tuple(case.full_source_latent_geometry),
                expected_steps=40,
                expected_flow_shift=5.0,
                omega_image=4.5,
                omega_text=4.0,
                eta=0.5,
                image_norm_threshold=50.0,
                text_norm_threshold=50.0,
                momentum=0.0,
            ),
            capability=capability,
            expected_bernini_commit=bernini_revision,
            observed_wan_diffusion_sha256=wan_sha,
        )
        patch.install()
    try:
        with torch.inference_mode():
            result, noise_capture = native._sample_with_native_initial_noise_observer(
                sample_fn=lambda: diffusion.sample(**sample_kwargs),
                wan_diffusion_module=wan_diffusion,
                expected_shape=tuple(case.full_source_latent_geometry),
                expected_device=device,
                expected_seed=case.seed,
            )
    finally:
        if patch is not None:
            patch.restore()
    surface_after = _certify_call_surface(
        diffusion, surface_before, label=f"{case.case_id}/{arm}"
    )
    runtime_trace = (
        dict(patch.finalize())
        if patch is not None
        else {
            "arm": ARM_OFFICIAL,
            "runtime_local_patch_installed": False,
            "model_shared_step_scheduler_patch_vae_latent_override": False,
            "official_initial_gaussian_observer_only": True,
            "vendor_wan_diffusion_sha256": wan_sha,
        }
    )
    if (
        not isinstance(result, torch.Tensor)
        or tuple(result.shape) != tuple(case.full_source_latent_geometry)
        or result.dtype != torch.float32
        or result.requires_grad
        or result.grad_fn is not None
        or not bool(torch.isfinite(result).all().item())
    ):
        raise NativeActivationV2RunnerError(f"{case.case_id}/{arm} result differs")
    freeze_after = strong_freeze(model)
    if freeze_after != freeze_before or any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in model.parameters()
    ):
        raise NativeActivationV2RunnerError(f"{case.case_id}/{arm} model changed")
    rng_after = _rng_receipt(torch, device=device)
    stored = result.detach().to(device="cpu", dtype=torch.float32).contiguous()
    result_identity = native._all_rank_tensor_identity(
        stored,
        label=f"{case.case_id}_{arm}_clean_latent",
        world_size=WORLD_SIZE,
    )
    noise_identity = native._all_rank_tensor_identity(
        noise_capture.tensor,
        label=f"{case.case_id}_{arm}_official_gaussian",
        world_size=WORLD_SIZE,
    )
    arm_receipt = {
        "case_id": case.case_id,
        "arm": arm,
        "sample_kwargs_without_tensors": {
            key: value
            for key, value in sample_kwargs.items()
            if key
            not in {
                "prompt_embeds",
                "uncond_prompt_embeds",
                "multi_video_vae_latents",
                "multi_image_vae_latents",
                "device",
            }
        },
        "sample_kwargs_digest": _canonical_sha256(
            {
                **{
                    key: value
                    for key, value in sample_kwargs.items()
                    if key
                    not in {
                        "prompt_embeds",
                        "uncond_prompt_embeds",
                        "multi_video_vae_latents",
                        "multi_image_vae_latents",
                        "device",
                    }
                },
                "source_sha256": activation.safe_core.tensor_content_sha256_v1(
                    source_latent
                ),
                "reference_sha256": [
                    activation.safe_core.tensor_content_sha256_v1(value)
                    for value in references
                ],
                "low_prompt_sha256": activation.safe_core.tensor_content_sha256_v1(
                    prompts["low_action"]
                ),
                "high_prompt_sha256": activation.safe_core.tensor_content_sha256_v1(
                    prompts["high_action"]
                ),
                "negative_prompt_sha256": activation.safe_core.tensor_content_sha256_v1(
                    prompts["negative"]
                ),
            }
        ),
        "official_gaussian_raw_sha256": noise_capture.raw_value_sha256,
        "official_gaussian_content_sha256": noise_capture.content_sha256,
        "official_gaussian_all_rank_identity": noise_identity,
        "clean_latent_all_rank_identity": result_identity,
        "call_surface_before_after_exact": surface_after,
        "fresh_model_load": load_receipt,
        "official_sampling_surface": {
            "diffusion_sample_override": arm == ARM_LOCAL,
            "shared_step_override": arm == ARM_LOCAL,
            "scheduler_step_override": arm == ARM_LOCAL,
            "patch_vae_latent_override": arm == ARM_LOCAL,
            "official_arm_observer_only_randn_wrapper": arm == ARM_OFFICIAL,
            "provided_materialized_prompt_embeddings": True,
            "unused_text_encoder_constructor_bypassed_before_sampling": True,
            "canonical_full_text_encoder_construction_claimed": False,
            "bypass_identical_for_official_and_local_arms": True,
            "denoiser_scheduler_sample_path_untouched_for_official_arm": (
                arm == ARM_OFFICIAL
            ),
        },
        "freeze_certificate": freeze_after,
        "rng_before": rng_before,
        "rng_after": rng_after,
        "runtime_trace": runtime_trace,
        "training": False,
        "optimizer": False,
        "backward": False,
        "self_generated_anchor_tensor_used": False,
        "target_video_or_latent_used": False,
    }
    model.to("cpu")
    del model, diffusion, result, patch, capability
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return stored, noise_capture, arm_receipt, runtime_trace


def _revalidate_live_case(
    *,
    activation: Any,
    authority: Any,
    case: Any,
    source_latent: Any,
    references: Sequence[Any],
    prompts: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Rebind every live condition used by either arm to frozen receipts."""

    activation.revalidate_compiled_activation_authority_v2(authority)
    reference = activation.validate_reference_receipt_v2(
        authority,
        case_id=case.case_id,
        source_video_latent=source_latent,
        source_reference_latents=tuple(references),
    )
    prompt = activation.validate_prompt_receipt_v2(
        authority,
        case_id=case.case_id,
        low_action_prompt_embeds=prompts["low_action"],
        high_action_prompt_embeds=prompts["high_action"],
        negative_prompt_embeds=prompts["negative"],
    )
    return {
        "case_id": case.case_id,
        "source_latent_sha256": reference.source_latent_sha256,
        "reference_latent_sha256": list(reference.reference_latent_sha256),
        "low_action_prompt_sha256": prompt.low_action_sha256,
        "high_action_prompt_sha256": prompt.high_action_sha256,
        "negative_prompt_sha256": prompt.negative_sha256,
        "authority_graph_revalidated": True,
    }


def _checkpoint_identity_rank_zero(
    *,
    activation: Any,
    source_audit: Any,
    checkpoint: Path,
    checkpoint_manifest: Path,
    dist: Any,
    rank: int,
    label: str,
) -> Mapping[str, Any]:
    status: list[Any] = [None]
    if rank == 0:
        try:
            status[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(status, src=0)
    row = status[0]
    if not isinstance(row, Mapping) or row.get("ok") is not True or not isinstance(
        row.get("identity"), Mapping
    ):
        raise NativeActivationV2RunnerError(
            f"{label} checkpoint validation failed: {row}"
        )
    identity = dict(row["identity"])
    _all_rank_object(
        _checkpoint_content_identity_sha256(identity, activation=activation),
        dist=dist,
        label=f"{label} checkpoint identity",
    )
    return identity


def _owned_file_identity(path: Path, *, label: str) -> Mapping[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise NativeActivationV2RunnerError(f"{label} open failed") from error
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named = path.lstat()
    except OSError as error:
        raise NativeActivationV2RunnerError(f"{label} named file disappeared") from error
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_size,
        row.st_mode,
        row.st_nlink,
    )
    if (
        identity(before) != identity(after)
        or identity(after) != identity(named)
        or not stat.S_ISREG(after.st_mode)
        or size != after.st_size
    ):
        raise NativeActivationV2RunnerError(f"{label} changed during owned read")
    return {
        "sha256": digest.hexdigest(),
        "size": int(after.st_size),
        "mode": stat.S_IMODE(after.st_mode),
        "nlink": int(after.st_nlink),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
    }


def _publish_staged_file_no_replace(
    *, staging_path: Path, destination: Path, label: str
) -> Mapping[str, Any]:
    """Stream a staged file into one final O_EXCL inode without replacement.

    The final-name inode exists from the first byte onward.  A failed partial
    publish is deliberately preserved, making the fresh output directory
    permanently fail closed.  No link-count transition or timestamp equality
    is used, which avoids the observed Lustre metadata-cache false failure.
    """

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", destination.name) is None:
        raise NativeActivationV2RunnerError(f"{label} destination basename differs")
    destination_parent = destination.parent.resolve(strict=True)
    if destination_parent != destination.parent or destination_parent.is_symlink():
        raise NativeActivationV2RunnerError(f"{label} destination parent differs")
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    source_fd: Optional[int] = None
    directory_fd: Optional[int] = None
    destination_fd: Optional[int] = None
    try:
        directory_fd = os.open(str(destination_parent), directory_flags)
        source_fd = os.open(str(staging_path), source_flags)
        source_before = os.fstat(source_fd)
        if not stat.S_ISREG(source_before.st_mode) or source_before.st_nlink != 1:
            raise NativeActivationV2RunnerError(f"{label} staged file differs")
        destination_fd = os.open(
            destination.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        created = os.fstat(destination_fd)
        created_identity = (int(created.st_dev), int(created.st_ino))
        digest = hashlib.sha256()
        copied = 0
        while True:
            try:
                chunk = os.read(source_fd, 1024 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            offset = 0
            while offset < len(chunk):
                try:
                    written = os.write(destination_fd, chunk[offset:])
                except InterruptedError:
                    continue
                if written <= 0:
                    raise NativeActivationV2RunnerError(
                        f"{label} destination write made no progress"
                    )
                offset += written
        source_after = os.fstat(source_fd)
        source_identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_size,
            row.st_mode,
            row.st_nlink,
        )
        if (
            source_identity(source_before) != source_identity(source_after)
            or copied != source_after.st_size
        ):
            raise NativeActivationV2RunnerError(f"{label} staged file changed")
        os.fchmod(destination_fd, 0o444)
        os.fsync(destination_fd)
        os.lseek(destination_fd, 0, os.SEEK_SET)
        restored_digest = hashlib.sha256()
        restored_size = 0
        while True:
            try:
                chunk = os.read(destination_fd, 1024 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            restored_digest.update(chunk)
            restored_size += len(chunk)
        same_fd = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(same_fd.st_mode)
            or same_fd.st_nlink != 1
            or stat.S_IMODE(same_fd.st_mode) != 0o444
            or same_fd.st_size != copied
            or restored_size != copied
            or restored_digest.digest() != digest.digest()
            or (same_fd.st_dev, same_fd.st_ino) != created_identity
        ):
            raise NativeActivationV2RunnerError(
                f"{label} created destination identity differs"
            )
        os.close(destination_fd)
        destination_fd = None
        os.fsync(directory_fd)
        check_fd = os.open(
            destination.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            final_digest = hashlib.sha256()
            final_size = 0
            while True:
                try:
                    chunk = os.read(check_fd, 1024 * 1024)
                except InterruptedError:
                    continue
                if not chunk:
                    break
                final_digest.update(chunk)
                final_size += len(chunk)
            final = os.fstat(check_fd)
            named = os.stat(
                destination.name, dir_fd=directory_fd, follow_symlinks=False
            )
        finally:
            os.close(check_fd)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or stat.S_IMODE(final.st_mode) != 0o444
            or final_size != copied
            or final_digest.digest() != digest.digest()
            or (final.st_dev, final.st_ino) != created_identity
            or (named.st_dev, named.st_ino, named.st_size, named.st_mode, named.st_nlink)
            != (final.st_dev, final.st_ino, final.st_size, final.st_mode, final.st_nlink)
        ):
            raise NativeActivationV2RunnerError(f"{label} final identity differs")
        return {
            "sha256": final_digest.hexdigest(),
            "size": int(final_size),
            "mode": stat.S_IMODE(final.st_mode),
            "nlink": int(final.st_nlink),
        }
    except FileExistsError as error:
        raise NativeActivationV2RunnerError(
            f"{label} destination appeared; refusing overwrite"
        ) from error
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _write_json_no_replace(path: Path, value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Write canonical JSON directly into one final-name O_EXCL inode."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", path.name) is None:
        raise NativeActivationV2RunnerError("receipt basename differs")
    directory = path.parent.resolve(strict=True)
    if directory != path.parent or directory.is_symlink():
        raise NativeActivationV2RunnerError("receipt directory differs")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(str(directory), directory_flags)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        created_identity = (int(opened.st_dev), int(opened.st_ino))
        offset = 0
        while offset < len(payload):
            try:
                written = os.write(descriptor, payload[offset:])
            except InterruptedError:
                continue
            if written <= 0:
                raise NativeActivationV2RunnerError(
                    "receipt write made no progress"
                )
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        restored = b""
        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            restored += chunk
        same_fd = os.fstat(descriptor)
        if (
            restored != payload
            or not stat.S_ISREG(same_fd.st_mode)
            or same_fd.st_nlink != 1
            or stat.S_IMODE(same_fd.st_mode) != 0o444
            or same_fd.st_size != len(payload)
            or (same_fd.st_dev, same_fd.st_ino) != created_identity
        ):
            raise NativeActivationV2RunnerError("created receipt bytes differ")
        os.close(descriptor)
        descriptor = None
        os.fsync(directory_fd)
        final = _owned_file_identity(path, label="run receipt")
        if (
            final["sha256"] != hashlib.sha256(payload).hexdigest()
            or final["size"] != len(payload)
            or final["mode"] != 0o444
            or final["nlink"] != 1
            or (final["device"], final["inode"]) != created_identity
        ):
            raise NativeActivationV2RunnerError("published receipt bytes differ")
        return {key: final[key] for key in ("sha256", "size", "mode", "nlink")}
    except FileExistsError as error:
        raise NativeActivationV2RunnerError(
            "receipt destination appeared; refusing overwrite"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _seal_miopen_cache_receipt_world4(
    *,
    initial: Mapping[str, Any],
    activation_rows: Sequence[Mapping[str, Any]],
    initial_world4: Sequence[Mapping[str, Any]],
    bootstrap_initial: Mapping[str, Any],
    arm_snapshots: Mapping[str, Mapping[str, Mapping[str, Any]]],
    loaded_library: Mapping[str, Any],
    output_dir: Path,
    torch: Any,
    dist: Any,
    rank: int,
) -> Mapping[str, Any]:
    """Seal and all-rank reopen the retained engineering-only cache receipt."""

    cache_root = Path(str(initial["root"]["path"]))
    if cache_root != Path(f"{output_dir}{MIOPEN_CACHE_SUFFIX}"):
        raise NativeActivationV2RunnerError("final MIOpen cache/output binding differs")
    scheduler_tmpdir_normalization = initial.get(
        "scheduler_tmpdir_normalization"
    )
    if (
        not isinstance(scheduler_tmpdir_normalization, Mapping)
        or _certify_scheduler_tmpdir_normalization()
        != scheduler_tmpdir_normalization
    ):
        raise NativeActivationV2RunnerError(
            "final scheduler TMPDIR normalization differs"
        )
    expected_initial_names = _expected_miopen_cache_directory_names()
    _collective_exact_directory_names(
        cache_root,
        expected=expected_initial_names,
        dist=dist,
        label="pre-receipt MIOpen cache",
    )
    final_world4 = _gather_miopen_cache_snapshot(
        initial,
        torch=torch,
        dist=dist,
        label="final MIOpen cache",
    )
    replay_world4 = _gather_miopen_cache_snapshot(
        initial,
        torch=torch,
        dist=dist,
        label="final MIOpen cache quiescence replay",
    )
    if replay_world4 != final_world4:
        raise NativeActivationV2RunnerError("final WORLD4 MIOpen cache is not quiescent")
    bootstrap_final = _gather_miopen_bootstrap_snapshot(
        initial,
        dist=dist,
        label="final launcher-bootstrap MIOpen cache",
    )
    receipt: dict[str, Any] = {
        "schema_version": "bernini-miopen-rank-private-cache-receipt-r9",
        "scope": "persistent engineering MIOpen DB/cache evidence only",
        "root_path": str(cache_root),
        "output_sibling_path": str(output_dir),
        "outside_frozen_release": True,
        "outside_scientific_output_artifact_set": True,
        "retained_after_run": True,
        "deleted_by_runner": False,
        "world_size": WORLD_SIZE,
        "per_rank_private_user_db_and_kernel_cache": True,
        "node_local_tmp_is_bound_by_separate_receipt": True,
        "same_rank_namespace_reused_by_serial_official_and_local_arms": True,
        "official_environment_variables": [
            MIOPEN_USER_DB_ENV,
            MIOPEN_KERNEL_CACHE_ENV,
        ],
        "cache_disable_or_system_db_override_used": False,
        "explicit_solver_control_environment_override_used": False,
        "matched_arms_share_cache_lookup_state": True,
        "scheduler_tmpdir_normalization": dict(
            scheduler_tmpdir_normalization
        ),
        "library": dict(loaded_library),
        "activation_world4": [dict(row) for row in activation_rows],
        "initial_world4": [dict(row) for row in initial_world4],
        "launcher_bootstrap_initial": dict(bootstrap_initial),
        "launcher_bootstrap_final": dict(bootstrap_final),
        "launcher_bootstrap_not_consumed_by_workers": True,
        "arm_snapshots": {
            case_id: {
                arm: dict(row) for arm, row in arms.items()
            }
            for case_id, arms in arm_snapshots.items()
        },
        "final_world4": final_world4,
        "final_snapshot_replayed_byte_exact_before_receipt_commit": True,
        "training_authority": False,
        "scientific_authority": False,
    }
    receipt["receipt_digest"] = _canonical_sha256(receipt)
    status: list[Any] = [None]
    receipt_path = cache_root / MIOPEN_CACHE_RECEIPT_NAME
    if rank == 0:
        try:
            status[0] = {
                "ok": True,
                "identity": _write_json_no_replace(receipt_path, receipt),
            }
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(status, src=0)
    if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
        raise NativeActivationV2RunnerError(
            f"MIOpen cache receipt publish failed: {status[0]}"
        )
    bound = status[0].get("identity")
    observed = _collective_owned_file_identity(
        receipt_path,
        dist=dist,
        label="MIOpen cache receipt",
    )
    if (
        not isinstance(bound, Mapping)
        or any(
            observed.get(key) != bound.get(key)
            for key in ("sha256", "size", "mode", "nlink")
        )
        or observed["mode"] != 0o444
        or observed["nlink"] != 1
    ):
        raise NativeActivationV2RunnerError("MIOpen cache receipt reopen differs")
    _collective_exact_directory_names(
        cache_root,
        expected=expected_initial_names | {MIOPEN_CACHE_RECEIPT_NAME},
        dist=dist,
        label="post-receipt MIOpen cache",
    )
    reopened_rows = [
        {key: observed[key] for key in ("sha256", "size", "mode", "nlink")}
    ] * WORLD_SIZE
    final_reopened_world4 = _gather_miopen_cache_snapshot(
        initial,
        torch=torch,
        dist=dist,
        label="post-receipt final MIOpen cache",
    )
    if final_reopened_world4 != final_world4:
        raise NativeActivationV2RunnerError(
            "MIOpen cache changed while sealing its receipt"
        )
    return {
        "path": str(receipt_path),
        "file_identity": {
            key: observed[key] for key in ("sha256", "size", "mode", "nlink")
        },
        "receipt_digest": receipt["receipt_digest"],
        "all_rank_reopen_exact": True,
        "all_rank_reopen_rows_sha256": _canonical_sha256(reopened_rows),
        "final_world4": final_world4,
        "launcher_bootstrap_final": bootstrap_final,
        "scheduler_tmpdir_normalization": dict(
            scheduler_tmpdir_normalization
        ),
        "retained_engineering_evidence": True,
        "scientific_output_artifact": False,
    }


def _seal_local_tmp_receipt_world4(
    *,
    initial: Mapping[str, Any],
    activation_rows: Sequence[Mapping[str, Any]],
    cpu_preflight_local_tmp_empty_proof: Mapping[str, Any],
    post_runtime_init_baseline_world4: Sequence[Mapping[str, Any]],
    post_runtime_init_baseline_world4_sha256: str,
    arm_snapshots: Mapping[str, Mapping[str, Mapping[str, Any]]],
    persistent_cache_receipt: Mapping[str, Any],
    output_dir: Path,
    torch: Any,
    dist: Any,
    rank: int,
) -> Mapping[str, Any]:
    """Seal the ephemeral node-local engineering temp evidence."""

    pre_torch_activation_world4 = [dict(row) for row in activation_rows]
    if (
        len(pre_torch_activation_world4) != WORLD_SIZE
        or [row.get("rank") for row in pre_torch_activation_world4]
        != list(range(WORLD_SIZE))
        or [row.get("local_rank") for row in pre_torch_activation_world4]
        != list(range(WORLD_SIZE))
        or any(
            row.get("initial_rank_directories_empty") is not True
            for row in pre_torch_activation_world4
        )
    ):
        raise NativeActivationV2RunnerError(
            "pre-Torch WORLD4 node-local tmp empty proof differs"
        )
    worker_pre_torch_empty_proof = initial.get(
        "worker_pre_torch_local_tmp_empty_proof"
    )
    launcher_local_tmp_empty_proof = initial.get(
        "launcher_local_tmp_fresh_empty_proof"
    )
    if (
        not isinstance(cpu_preflight_local_tmp_empty_proof, Mapping)
        or not isinstance(worker_pre_torch_empty_proof, Mapping)
        or not isinstance(launcher_local_tmp_empty_proof, Mapping)
        or cpu_preflight_local_tmp_empty_proof != worker_pre_torch_empty_proof
        or launcher_local_tmp_empty_proof
        != _launcher_local_tmp_empty_proof(
            initial["local_tmp_root"],
            initial["local_tmp_role_directory_identities"],
        )
        or any(
            row.get("launcher_local_tmp_fresh_empty_proof")
            != launcher_local_tmp_empty_proof
            or row.get("worker_pre_torch_local_tmp_empty_proof")
            != worker_pre_torch_empty_proof
            for row in pre_torch_activation_world4
        )
    ):
        raise NativeActivationV2RunnerError(
            "three-layer pre-Torch node-local tmp empty proof differs"
        )
    validated_post_runtime_baseline, baseline_sha256 = (
        _validate_local_tmp_world4_snapshot_rows(
            post_runtime_init_baseline_world4,
            initial=initial,
            label="post-runtime-init node-local tmp receipt baseline",
        )
    )
    if (
        _SHA256.fullmatch(post_runtime_init_baseline_world4_sha256) is None
        or baseline_sha256 != post_runtime_init_baseline_world4_sha256
    ):
        raise NativeActivationV2RunnerError(
            "post-runtime-init node-local tmp baseline digest differs"
        )
    if set(arm_snapshots) != set(EXECUTION_CASES):
        raise NativeActivationV2RunnerError(
            "node-local tmp arm snapshot cases differ"
        )
    validated_arm_snapshots: dict[str, dict[str, Mapping[str, Any]]] = {}
    for case_id in EXECUTION_CASES:
        case_snapshots = arm_snapshots.get(case_id)
        if (
            not isinstance(case_snapshots, Mapping)
            or tuple(case_snapshots) != ARM_ORDER_BY_CASE[case_id]
        ):
            raise NativeActivationV2RunnerError(
                "node-local tmp arm snapshot order differs"
            )
        validated_arm_snapshots[case_id] = {}
        for arm in ARM_ORDER_BY_CASE[case_id]:
            boundaries = case_snapshots.get(arm)
            if not isinstance(boundaries, Mapping) or set(boundaries) != {
                "before_world4",
                "after_world4",
            }:
                raise NativeActivationV2RunnerError(
                    "node-local tmp arm boundary schema differs"
                )
            before_rows, before_sha = _validate_local_tmp_world4_snapshot_rows(
                boundaries["before_world4"],
                initial=initial,
                label=f"{case_id}/{arm} receipt pre-arm node-local tmp",
            )
            after_rows, after_sha = _validate_local_tmp_world4_snapshot_rows(
                boundaries["after_world4"],
                initial=initial,
                label=f"{case_id}/{arm} receipt post-arm node-local tmp",
            )
            validated_arm_snapshots[case_id][arm] = {
                "before_world4": before_rows,
                "before_world4_sha256": before_sha,
                "after_world4": after_rows,
                "after_world4_sha256": after_sha,
            }

    scheduler = initial.get("scheduler_tmpdir_normalization")
    local_root_receipt = initial.get("local_tmp_root")
    local_parent_receipt = initial.get("local_tmp_parent_identity")
    persistent_root_receipt = initial.get("root")
    if (
        not isinstance(scheduler, Mapping)
        or not isinstance(local_root_receipt, Mapping)
        or not isinstance(local_parent_receipt, Mapping)
        or not isinstance(persistent_root_receipt, Mapping)
        or _certify_scheduler_tmpdir_normalization() != scheduler
        or _node_local_tmp_parent_identity() != local_parent_receipt
    ):
        raise NativeActivationV2RunnerError(
            "final node-local tmp binding differs"
        )
    local_root = Path(str(local_root_receipt.get("path")))
    if (
        local_root
        != _expected_miopen_local_tmp_root(output_dir, scheduler)
        or local_root_receipt.get("device")
        == persistent_root_receipt.get("device")
    ):
        raise NativeActivationV2RunnerError(
            "final node-local tmp domain/filesystem differs"
        )
    expected_names = _expected_miopen_local_tmp_directory_names()
    _collective_exact_directory_names(
        local_root,
        expected=expected_names,
        dist=dist,
        label="pre-receipt node-local tmp",
    )
    final_world4 = _gather_local_tmp_snapshot(
        initial, torch=torch, dist=dist, label="final node-local tmp"
    )
    replay_world4 = _gather_local_tmp_snapshot(
        initial,
        torch=torch,
        dist=dist,
        label="final node-local tmp quiescence replay",
    )
    if replay_world4 != final_world4:
        raise NativeActivationV2RunnerError(
            "final WORLD4 node-local tmp is not quiescent"
        )
    receipt: dict[str, Any] = {
        "schema_version": "bernini-node-local-rank-tmp-receipt-r11",
        "scope": "ephemeral node-local engineering temp evidence only",
        "root_path": str(local_root),
        "output_domain_path": str(output_dir),
        "exact_parent": str(MIOPEN_LOCAL_TMP_PARENT),
        "parent_identity": dict(local_parent_receipt),
        "root_identity": dict(local_root_receipt),
        "persistent_cache_root_identity": dict(persistent_root_receipt),
        "persistent_and_local_devices_are_distinct": True,
        "scheduler_tmpdir_normalization": dict(scheduler),
        "world_size": WORLD_SIZE,
        "rank_private_tmp_directories": True,
        "same_rank_tmp_reused_by_serial_official_and_local_arms": True,
        "pre_torch_empty_activation_world4": pre_torch_activation_world4,
        "pre_torch_rank_directories_empty_all_ranks": True,
        "launcher_fresh_empty_proof": dict(launcher_local_tmp_empty_proof),
        "cpu_preflight_empty_proof": dict(
            cpu_preflight_local_tmp_empty_proof
        ),
        "worker_pre_torch_empty_proof": dict(worker_pre_torch_empty_proof),
        "launcher_cpu_preflight_worker_empty_proofs_exact": True,
        "post_runtime_init_baseline_world4": validated_post_runtime_baseline,
        "post_runtime_init_baseline_world4_sha256": baseline_sha256,
        "post_runtime_init_baseline_may_be_nonempty": True,
        "post_runtime_init_baseline_strictly_scanned_and_quiescent": True,
        "post_runtime_init_baseline_is_observation_not_allowlist": True,
        "post_runtime_init_baseline_is_not_immutable_or_monotonic_claim": True,
        "differences_present_at_observation_boundaries_are_recorded_not_forbidden": True,
        "continuous_monitoring_claimed": False,
        "transients_created_and_removed_between_observations_may_be_unrecorded": True,
        "every_observed_stage_is_fully_hashed": True,
        "final_seal_and_post_output_replays_require_exact_quiescence": True,
        "observation_order": [
            "launcher/preflight/worker-pre-Torch-empty",
            "post-runtime-init-baseline",
            "per-arm-pre/post",
            "final-and-replay-before-receipt",
        ],
        "arm_snapshots": validated_arm_snapshots,
        "final_world4": final_world4,
        "final_snapshot_replayed_byte_exact_before_receipt_commit": True,
        "persistent_cache_receipt": {
            "path": str(persistent_cache_receipt["path"]),
            "file_identity": dict(persistent_cache_receipt["file_identity"]),
            "receipt_digest": str(persistent_cache_receipt["receipt_digest"]),
        },
        "runner_cleanup_performed": False,
        "durability_guaranteed": False,
        "node_lifetime_only": True,
        "observed_and_replayed_before_WORLD4_step_exit": True,
        "existence_after_process_or_step_exit_guaranteed": False,
        "engineering_evidence_only": True,
        "scientific_output_artifact": False,
        "training_authority": False,
        "scientific_authority": False,
    }
    receipt["receipt_digest"] = _canonical_sha256(receipt)
    receipt_path = local_root / MIOPEN_LOCAL_TMP_RECEIPT_NAME
    status: list[Any] = [None]
    if rank == 0:
        try:
            status[0] = {
                "ok": True,
                "identity": _write_json_no_replace(receipt_path, receipt),
            }
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(status, src=0)
    if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
        raise NativeActivationV2RunnerError(
            f"node-local tmp receipt publish failed: {status[0]}"
        )
    bound = status[0].get("identity")
    observed = _collective_owned_file_identity(
        receipt_path, dist=dist, label="node-local tmp receipt"
    )
    if (
        not isinstance(bound, Mapping)
        or any(
            observed.get(key) != bound.get(key)
            for key in ("sha256", "size", "mode", "nlink")
        )
        or observed["mode"] != 0o444
        or observed["nlink"] != 1
    ):
        raise NativeActivationV2RunnerError(
            "node-local tmp receipt reopen differs"
        )
    _collective_exact_directory_names(
        local_root,
        expected=expected_names | {MIOPEN_LOCAL_TMP_RECEIPT_NAME},
        dist=dist,
        label="post-receipt node-local tmp",
    )
    post_receipt_world4 = _gather_local_tmp_snapshot(
        initial, torch=torch, dist=dist, label="post-receipt node-local tmp"
    )
    if post_receipt_world4 != final_world4:
        raise NativeActivationV2RunnerError(
            "node-local tmp changed while sealing receipt"
        )
    return {
        "path": str(receipt_path),
        "file_identity": {
            key: observed[key] for key in ("sha256", "size", "mode", "nlink")
        },
        "receipt_digest": receipt["receipt_digest"],
        # The node-local file can disappear with the Slurm step.  Preserve its
        # exact semantic content inside the durable scientific run receipt;
        # the file identity above still proves the step-local committed bytes.
        "receipt_content": receipt,
        "full_receipt_embedded_in_durable_output_receipt": True,
        "root_identity": dict(local_root_receipt),
        "parent_identity": dict(local_parent_receipt),
        "final_world4": final_world4,
        "scheduler_tmpdir_normalization": dict(scheduler),
        "runner_cleanup_performed": False,
        "durability_guaranteed": False,
        "node_lifetime_only": True,
        "observed_and_replayed_before_WORLD4_step_exit": True,
        "existence_after_process_or_step_exit_guaranteed": False,
        "scientific_output_artifact": False,
    }


def _stage_rank_zero_outputs(
    *,
    staging_dir: Path,
    output_dir: Path,
    generated: Mapping[str, Mapping[str, Any]],
    noises: Mapping[str, Mapping[str, Any]],
    arm_receipts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    authority: Any,
    activation: Any,
    native: Any,
    autoencoder_class: Any,
    vae_decode: Callable[..., Any],
    save_output_fn: Callable[..., Any],
    materialize_vae: Any,
    checkpoint: Path,
    torch: Any,
    device: Any,
) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    """Stage every latent/noise/video before the release directory exists."""

    if staging_dir.is_symlink() or stat.S_IMODE(staging_dir.stat().st_mode) != 0o700:
        raise NativeActivationV2RunnerError("private output staging directory differs")
    vae = autoencoder_class.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False).to(device)
    staged: dict[str, dict[str, Mapping[str, Any]]] = {}
    try:
        for case_id in EXECUTION_CASES:
            case = authority.cases[case_id]
            expected_hw = (
                int(case.full_source_latent_geometry[-2] * 8),
                int(case.full_source_latent_geometry[-1] * 8),
            )
            staged[case_id] = {}
            for arm in ARM_ORDER_BY_CASE[case_id]:
                prefix = f"{case_id}.{arm}"
                noise_path = staging_dir / f"{prefix}.official-gaussian.safetensors"
                noise_row = native._save_initial_noise_atomically(
                    noise_path,
                    noises[case_id][arm],
                    all_rank_identity=arm_receipts[case_id][arm][
                        "official_gaussian_all_rank_identity"
                    ],
                )
                latent_path = staging_dir / f"{prefix}.clean-latent.safetensors"
                latent_row = native._save_normalized_clean_latent_atomically(
                    latent_path,
                    generated[case_id][arm],
                )
                latent_device = generated[case_id][arm].to(
                    device=device, dtype=torch.float32
                ).contiguous()
                with torch.inference_mode():
                    decoded = vae_decode(vae, latent_device)
                del latent_device
                if tuple(int(item) for item in decoded.shape) != (
                    FRAME_COUNT,
                    expected_hw[0],
                    expected_hw[1],
                    3,
                ):
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} decoded video geometry differs"
                    )
                video_path = staging_dir / f"{prefix}.mp4"
                save_output_fn(decoded, str(video_path), fps=FPS)
                del decoded
                if video_path.is_symlink() or not video_path.is_file():
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} encoder output differs"
                    )
                raw_frames, encoded_fps, encoded_hw = (
                    materialize_vae._decode_exact_video(video_path)
                )
                if (
                    len(raw_frames) != FRAME_COUNT
                    or float(encoded_fps) != float(FPS)
                    or tuple(int(item) for item in encoded_hw) != expected_hw
                ):
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} encoded video metadata differs"
                    )
                video_identity = _owned_file_identity(
                    video_path, label=f"{case_id}/{arm} staged video"
                )
                staged[case_id][arm] = {
                    "official_gaussian": {
                        "staging_path": str(noise_path),
                        "destination_path": str(output_dir / noise_path.name),
                        "metadata": {
                            **dict(noise_row),
                            "path": str(output_dir / noise_path.name),
                        },
                    },
                    "clean_latent": {
                        "staging_path": str(latent_path),
                        "destination_path": str(output_dir / latent_path.name),
                        "metadata": {
                            **dict(latent_row),
                            "path": str(output_dir / latent_path.name),
                        },
                    },
                    "video": {
                        "staging_path": str(video_path),
                        "destination_path": str(output_dir / video_path.name),
                        "metadata": {
                            "path": str(output_dir / video_path.name),
                            "sha256": video_identity["sha256"],
                            "frame_count": FRAME_COUNT,
                            "fps": FPS,
                            "height": expected_hw[0],
                            "width": expected_hw[1],
                        },
                    },
                }
    finally:
        vae.to("cpu")
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    return staged


def _publish_staged_outputs(
    *, output_dir: Path, staged: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    """Create the final directory once, then O_EXCL-copy every artifact once."""

    try:
        output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as error:
        raise NativeActivationV2RunnerError(
            "output directory appeared; refusing overwrite"
        ) from error
    published: dict[str, dict[str, Mapping[str, Any]]] = {}
    for case_id in EXECUTION_CASES:
        published[case_id] = {}
        for arm in ARM_ORDER_BY_CASE[case_id]:
            role_rows: dict[str, Any] = {}
            for role in ("official_gaussian", "clean_latent", "video"):
                row = staged[case_id][arm][role]
                staging_path = Path(str(row["staging_path"]))
                destination = Path(str(row["destination_path"]))
                if destination.parent != output_dir:
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm}/{role} output parent differs"
                    )
                identity = _publish_staged_file_no_replace(
                    staging_path=staging_path,
                    destination=destination,
                    label=f"{case_id}/{arm}/{role}",
                )
                metadata = dict(row["metadata"])
                if metadata.get("sha256") != identity["sha256"]:
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm}/{role} staged digest differs"
                    )
                metadata["file_identity"] = dict(identity)
                role_rows[role] = metadata
            published[case_id][arm] = role_rows
    return published


def _validate_output_release_files(
    *,
    output_dir: Path,
    receipt: Mapping[str, Any],
    receipt_identity: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Rebind every named final artifact to the hashes recorded in receipt."""

    cases = receipt.get("cases")
    if not isinstance(cases, Mapping) or tuple(cases) != CASE_ORDER:
        raise NativeActivationV2RunnerError("output receipt cases/order differ")
    e03 = cases.get("e03")
    if (
        not isinstance(e03, Mapping)
        or e03.get("decision") != "ABSTAIN_KEEP_BASE"
        or e03.get("executed") is not False
        or e03.get("arms") != []
        or e03.get("condition_receipts") is not None
        or e03.get("arm_receipts") != {}
        or e03.get("runtime_traces") != {}
        or e03.get("outputs") != {}
        or e03.get("selection") != "ABSTAIN_KEEP_BASE"
        or e03.get("kept_frozen_base")
        != {
            "path": str(E03_FROZEN_BASE_PATH),
            "sha256": E03_FROZEN_BASE_SHA256,
        }
    ):
        raise NativeActivationV2RunnerError("e03 policy-only output receipt differs")
    expected: dict[str, Mapping[str, Any]] = {
        "receipt.json": dict(receipt_identity)
    }
    for case_id in EXECUTION_CASES:
        case_row = cases.get(case_id)
        if not isinstance(case_row, Mapping):
            raise NativeActivationV2RunnerError(
                f"output receipt {case_id} row differs"
            )
        outputs = case_row.get("outputs")
        if not isinstance(outputs, Mapping) or tuple(outputs) != ARM_ORDER_BY_CASE[
            case_id
        ]:
            raise NativeActivationV2RunnerError(
                f"output receipt {case_id} arms/order differ"
            )
        for arm in ARM_ORDER_BY_CASE[case_id]:
            roles = outputs.get(arm)
            if not isinstance(roles, Mapping) or set(roles) != {
                "official_gaussian",
                "clean_latent",
                "video",
            }:
                raise NativeActivationV2RunnerError(
                    f"output receipt {case_id}/{arm} roles differ"
                )
            filenames = {
                "official_gaussian": f"{case_id}.{arm}.official-gaussian.safetensors",
                "clean_latent": f"{case_id}.{arm}.clean-latent.safetensors",
                "video": f"{case_id}.{arm}.mp4",
            }
            for role, filename in filenames.items():
                metadata = roles.get(role)
                if not isinstance(metadata, Mapping):
                    raise NativeActivationV2RunnerError(
                        f"output receipt {case_id}/{arm}/{role} metadata differs"
                    )
                file_identity = metadata.get("file_identity")
                final_path = output_dir / filename
                if (
                    not isinstance(file_identity, Mapping)
                    or set(file_identity) != {"sha256", "size", "mode", "nlink"}
                    or metadata.get("path") != str(final_path)
                    or metadata.get("sha256") != file_identity.get("sha256")
                    or filename in expected
                ):
                    raise NativeActivationV2RunnerError(
                        f"output receipt {case_id}/{arm}/{role} binding differs"
                    )
                expected[filename] = dict(file_identity)

    observed_names = {path.name for path in output_dir.iterdir()}
    if observed_names != set(expected):
        raise NativeActivationV2RunnerError("output release file set differs")
    observed: dict[str, Mapping[str, Any]] = {}
    for name, bound in expected.items():
        identity = _owned_file_identity(output_dir / name, label=f"output {name}")
        if any(
            identity.get(key) != bound.get(key)
            for key in ("sha256", "size", "mode", "nlink")
        ) or identity["mode"] != 0o444 or identity["nlink"] != 1:
            raise NativeActivationV2RunnerError(
                f"output {name} differs from receipt"
            )
        observed[name] = {
            key: identity[key] for key in ("sha256", "size", "mode", "nlink")
        }
    return observed


def _freeze_output_release(
    *, output_dir: Path, receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    receipt_digest = receipt.get("receipt_digest")
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_digest", None)
    if (
        not isinstance(receipt_digest, str)
        or _SHA256.fullmatch(receipt_digest) is None
        or receipt_digest != _canonical_sha256(unsigned_receipt)
    ):
        raise NativeActivationV2RunnerError("run receipt digest differs")
    receipt_identity = _write_json_no_replace(output_dir / "receipt.json", receipt)
    receipt_payload = json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if receipt_identity != {
        "sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "size": len(receipt_payload),
        "mode": 0o444,
        "nlink": 1,
    }:
        raise NativeActivationV2RunnerError("published receipt identity differs")
    before_freeze = _validate_output_release_files(
        output_dir=output_dir,
        receipt=receipt,
        receipt_identity=receipt_identity,
    )
    output_dir.chmod(0o555)
    if output_dir.is_symlink() or stat.S_IMODE(output_dir.stat().st_mode) != 0o555:
        raise NativeActivationV2RunnerError("output release directory is not frozen")
    after_freeze = _validate_output_release_files(
        output_dir=output_dir,
        receipt=receipt,
        receipt_identity=receipt_identity,
    )
    if after_freeze != before_freeze:
        raise NativeActivationV2RunnerError("output release changed while freezing")
    return {
        "receipt": dict(receipt_identity),
        "artifact_identities": after_freeze,
        "directory_mode": "0555",
        "all_artifacts_mode": "0444",
        "all_artifacts_nlink": 1,
        "no_overwrite_publish": True,
    }


def _certify_local_release_import_closure(
    *,
    preflight_receipt: Mapping[str, Any],
    activation: Any,
    native: Any,
    prompt_builder: Any,
    source_audit: Any,
    freeze_provider: Any,
    sampler_contract: Any,
    native_schedule_contract: Any,
    tools_package: Any,
    materialize_vae: Any,
) -> Mapping[str, Any]:
    """Rehash the full local release and bind every executed provider object."""

    identities = preflight_receipt.get("component_identities")
    if not isinstance(identities, Mapping) or set(identities) != set(
        _LOCAL_RELEASE_PATHS
    ):
        raise NativeActivationV2RunnerError("preflight local closure map differs")
    observed: dict[str, str] = {}
    for label, relative in _LOCAL_RELEASE_PATHS.items():
        row = identities.get(label)
        path = METHOD_ROOT / relative
        resolved = path.resolve(strict=True)
        current = _owned_file_identity(resolved, label=f"local release {label}")
        if (
            not isinstance(row, Mapping)
            or resolved != path
            or resolved.is_symlink()
            or not resolved.is_file()
            or any(
                row.get(key) != current[key]
                for key in ("sha256", "size", "mode", "nlink")
            )
        ):
            raise NativeActivationV2RunnerError(
                f"local release closure differs: {label}"
            )
        observed[label] = str(row["sha256"])
    expected_tools = (METHOD_ROOT / "tools").resolve(strict=True)
    object_paths = {
        "activation_runtime": Path(activation.__file__).resolve(strict=True),
        "native": Path(native.__file__).resolve(strict=True),
        "prompt_builder": Path(prompt_builder.__file__).resolve(strict=True),
        "source_audit": Path(source_audit.__file__).resolve(strict=True),
        "freeze_provider": Path(freeze_provider.__file__).resolve(strict=True),
        "sampler_contract": Path(sampler_contract.__file__).resolve(strict=True),
        "native_schedule_contract": Path(
            native_schedule_contract.__file__
        ).resolve(strict=True),
        "safe_core": Path(activation.safe_core.__file__).resolve(strict=True),
        "native_runtime": Path(activation.native_runtime.__file__).resolve(strict=True),
        "native_homotopy": Path(activation.native_homotopy.__file__).resolve(
            strict=True
        ),
        "sgaf": Path(activation.sgaf.__file__).resolve(strict=True),
        "native_legacy": Path(native.legacy.__file__).resolve(strict=True),
        "tools_init": Path(tools_package.__file__).resolve(strict=True),
        "materialize_vae": Path(materialize_vae.__file__).resolve(strict=True),
        "build_renderer_dataset": Path(
            materialize_vae.raw_builder.__file__
        ).resolve(strict=True),
        "source_value_oracle": Path(
            sys.modules["infer_source_value_residual_oracle"].__file__
        ).resolve(strict=True),
        "train_lora": Path(sys.modules["train_lora"].__file__).resolve(strict=True),
        "model_authority": Path(
            sys.modules[
                "action_preservation_decoded_eval_model_authority_v2"
            ].__file__
        ).resolve(strict=True),
        "source_kv_replay": Path(
            sys.modules["source_kv_replay"].__file__
        ).resolve(strict=True),
        "source_kv_route_batches": Path(
            sys.modules["source_kv_route_batches"].__file__
        ).resolve(strict=True),
        "source_value_residual": Path(
            sys.modules["source_value_residual"].__file__
        ).resolve(strict=True),
    }
    expected_object_paths = {
        "activation_runtime": METHOD_ROOT
        / "oracle_regeneration_native_runtime_activation_v2.py",
        "native": METHOD_ROOT / "infer_native_identity_generation_canary.py",
        "prompt_builder": METHOD_ROOT / "infer_native_branch_homotopy_canary.py",
        "source_audit": METHOD_ROOT / "infer_source_kv_carrier_oracle.py",
        "freeze_provider": METHOD_ROOT
        / "infer_native_self_guided_action_field_canary.py",
        "sampler_contract": METHOD_ROOT / "tri_branch_unipc.py",
        "native_schedule_contract": METHOD_ROOT
        / "source_self_native_ref_contrastive_v3.py",
        "safe_core": METHOD_ROOT / "oracle_regeneration_canary_v1.py",
        "native_runtime": METHOD_ROOT / "native_branch_homotopy_runtime_v1.py",
        "native_homotopy": METHOD_ROOT / "native_branch_homotopy_v1.py",
        "sgaf": METHOD_ROOT / "self_guided_action_field_v1.py",
        "native_legacy": METHOD_ROOT / "infer_lora.py",
        "tools_init": expected_tools / "__init__.py",
        "materialize_vae": expected_tools / "materialize_vae.py",
        "build_renderer_dataset": expected_tools / "build_renderer_dataset.py",
        "source_value_oracle": METHOD_ROOT / "infer_source_value_residual_oracle.py",
        "train_lora": METHOD_ROOT / "train_lora.py",
        "model_authority": METHOD_ROOT
        / "action_preservation_decoded_eval_model_authority_v2.py",
        "source_kv_replay": METHOD_ROOT / "source_kv_replay.py",
        "source_kv_route_batches": METHOD_ROOT / "source_kv_route_batches.py",
        "source_value_residual": METHOD_ROOT / "source_value_residual.py",
    }
    value_audit = sys.modules["infer_source_value_residual_oracle"]
    trainer = sys.modules["train_lora"]
    model_authority = sys.modules[
        "action_preservation_decoded_eval_model_authority_v2"
    ]
    replay_core = sys.modules["source_kv_replay"]
    route_batches = sys.modules["source_kv_route_batches"]
    value_core = sys.modules["source_value_residual"]
    object_graph_exact = (
        native.source_audit is source_audit
        and native.value_audit is value_audit
        and prompt_builder.source_audit is source_audit
        and prompt_builder.value_audit is value_audit
        and freeze_provider.source_audit is source_audit
        and freeze_provider.value_audit is value_audit
        and native.legacy is source_audit.legacy
        and native.legacy is value_audit.legacy
        and native.legacy.trainer is trainer
        and native.legacy.model_authority is model_authority
        and source_audit.replay_core is replay_core
        and value_audit.replay_core is replay_core
        and source_audit.route_batches is route_batches
        and value_audit.route_batches is route_batches
        and value_audit.value_core is value_core
        and value_core.replay is replay_core
        and activation.sampler_contract is sampler_contract
        and activation.native_runtime.sampler_contract is sampler_contract
        and activation.native_runtime.homotopy is activation.native_homotopy
        and activation.native_runtime.sgaf is activation.sgaf
        and activation._load_native_schedule_contract_v2()
        is native_schedule_contract
        and sys.modules.get("source_self_native_ref_contrastive_v3")
        is native_schedule_contract
        and prompt_builder.sampler_contract is sampler_contract
        and freeze_provider.sampler_contract is sampler_contract
    )
    callable_origins = {
        "activation_loader": Path(
            activation.load_compiled_activation_authority_v2.__code__.co_filename
        ).resolve(strict=True),
        "noise_observer": Path(
            native._sample_with_native_initial_noise_observer.__code__.co_filename
        ).resolve(strict=True),
        "prompt_builder": Path(
            prompt_builder.build_mode_native_prompt.__code__.co_filename
        ).resolve(strict=True),
        "checkpoint_validator": Path(
            source_audit.validate_checkpoint_content.__code__.co_filename
        ).resolve(strict=True),
        "freeze_certificate": Path(
            freeze_provider._strong_model_freeze_certificate.__code__.co_filename
        ).resolve(strict=True),
        "diffusion_core_resolver": Path(
            sampler_contract.resolve_diffusion_core.__code__.co_filename
        ).resolve(strict=True),
        "native_schedule_receipt": Path(
            native_schedule_contract.native_unipc40_schedule_receipt.__code__.co_filename
        ).resolve(strict=True),
        "video_decoder": Path(
            materialize_vae._decode_exact_video.__code__.co_filename
        ).resolve(strict=True),
        "source_bucket": Path(
            materialize_vae.raw_builder.canonical_json_bytes.__code__.co_filename
        ).resolve(strict=True),
    }
    expected_callable_origins = {
        "activation_loader": expected_object_paths["activation_runtime"],
        "noise_observer": expected_object_paths["native"],
        "prompt_builder": expected_object_paths["prompt_builder"],
        "checkpoint_validator": expected_object_paths["source_audit"],
        "freeze_certificate": expected_object_paths["freeze_provider"],
        "diffusion_core_resolver": expected_object_paths["sampler_contract"],
        "native_schedule_receipt": expected_object_paths[
            "native_schedule_contract"
        ],
        "video_decoder": expected_object_paths["materialize_vae"],
        "source_bucket": expected_object_paths["build_renderer_dataset"],
    }
    if (
        object_paths != expected_object_paths
        or not object_graph_exact
        or callable_origins != expected_callable_origins
        or [Path(item).resolve(strict=True) for item in tools_package.__path__]
        != [expected_tools]
        or materialize_vae.raw_builder
        is not sys.modules.get("tools.build_renderer_dataset")
        or "_omnivideo2_strict_action_preview_materializer" in sys.modules
        or Path(
            freeze_provider._strong_model_freeze_certificate.__code__.co_filename
        ).resolve(strict=True)
        != expected_object_paths["freeze_provider"]
    ):
        raise NativeActivationV2RunnerError("local runtime import objects differ")
    for label, path in object_paths.items():
        if _sha256_file(path) != observed[
            {
                "activation_runtime": "runtime",
                "native": "dependency:infer_native_identity_generation_canary.py",
                "prompt_builder": "dependency:infer_native_branch_homotopy_canary.py",
                "source_audit": "dependency:infer_source_kv_carrier_oracle.py",
                "freeze_provider": "dependency:infer_native_self_guided_action_field_canary.py",
                "sampler_contract": "dependency:tri_branch_unipc.py",
                "native_schedule_contract": "dependency:source_self_native_ref_contrastive_v3.py",
                "safe_core": "dependency:oracle_regeneration_canary_v1.py",
                "native_runtime": "dependency:native_branch_homotopy_runtime_v1.py",
                "native_homotopy": "dependency:native_branch_homotopy_v1.py",
                "sgaf": "dependency:self_guided_action_field_v1.py",
                "native_legacy": "dependency:infer_lora.py",
                "tools_init": "dependency:tools/__init__.py",
                "materialize_vae": "dependency:tools/materialize_vae.py",
                "build_renderer_dataset": "dependency:tools/build_renderer_dataset.py",
                "source_value_oracle": "dependency:infer_source_value_residual_oracle.py",
                "train_lora": "dependency:train_lora.py",
                "model_authority": "dependency:action_preservation_decoded_eval_model_authority_v2.py",
                "source_kv_replay": "dependency:source_kv_replay.py",
                "source_kv_route_batches": "dependency:source_kv_route_batches.py",
                "source_value_residual": "dependency:source_value_residual.py",
            }[label]
        ]:
            raise NativeActivationV2RunnerError(
                f"local runtime provider bytes differ: {label}"
            )
    preflight_path = Path(release_preflight.__file__).resolve(strict=True)
    if preflight_path != PREFLIGHT_PATH:
        raise NativeActivationV2RunnerError("preflight provider origin differs")
    return {
        "release_sha256": observed,
        "provider_paths": {key: str(value) for key, value in object_paths.items()},
        "callable_origins": {
            key: str(value) for key, value in callable_origins.items()
        },
        "provider_object_graph_exact": True,
        "preflight_path": str(preflight_path),
        "preflight_sha256": _sha256_file(preflight_path),
        "lazy_omnivideo_materializer_loaded": False,
    }


def _certify_copied_local_prompt_role_v2(
    *,
    activation: Any,
    role: str,
    receipt_path: Any,
    receipt_sha256: Any,
    observed_current_path: Path,
    expected_current_pin_sha256: Any,
) -> str:
    """Bind one explicitly allowlisted copied-local prompt provider twice.

    The materializer receipt names the immutable authoring-bundle copy, while
    the sampler imports an independently frozen copy from this release.  Only
    the three fixed roles in ``_COPIED_LOCAL_PROMPT_ROLE_BINDINGS`` may
    use this dual-origin rule; every other implementation role retains exact
    receipt-path equality in ``_certify_runtime_import_closure``.
    """

    binding = _COPIED_LOCAL_PROMPT_ROLE_BINDINGS.get(role)
    if binding is None:
        raise NativeActivationV2RunnerError(
            f"runtime copied-local prompt role is not allowlisted: {role}"
        )
    relative = binding["release_relative_path"]
    if (
        _LOCAL_RELEASE_PATHS.get(f"dependency:{relative}") != relative
        or not isinstance(receipt_path, str)
        or not receipt_path
        or not isinstance(receipt_sha256, str)
        or _SHA256.fullmatch(receipt_sha256) is None
        or not isinstance(expected_current_pin_sha256, str)
        or _SHA256.fullmatch(expected_current_pin_sha256) is None
        or receipt_path != binding["receipt_path"]
        or receipt_sha256 != binding["sha256"]
        or expected_current_pin_sha256 != binding["sha256"]
    ):
        raise NativeActivationV2RunnerError(
            f"runtime copied-local {role} receipt binding differs"
        )

    requested_receipt_path = Path(receipt_path)
    expected_current_requested = METHOD_ROOT / relative
    try:
        resolved_receipt_path = requested_receipt_path.resolve(strict=True)
        expected_current_path = expected_current_requested.resolve(strict=True)
        observed_path = Path(observed_current_path).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise NativeActivationV2RunnerError(
            f"runtime copied-local {role} path is absent"
        ) from error
    if (
        not requested_receipt_path.is_absolute()
        or requested_receipt_path.is_symlink()
        or resolved_receipt_path != requested_receipt_path
        or expected_current_requested.is_symlink()
        or expected_current_path != expected_current_requested
        or observed_path != expected_current_path
        or resolved_receipt_path == expected_current_path
    ):
        raise NativeActivationV2RunnerError(
            f"runtime copied-local {role} origin differs"
        )
    try:
        receipt_seal, _ = activation._seal_plain_file_v2(
            resolved_receipt_path,
            label=f"runtime copied-local {role} materializer origin",
            retain_bytes=False,
            require_frozen=True,
        )
        current_seal, _ = activation._seal_plain_file_v2(
            expected_current_path,
            label=f"runtime copied-local {role} release origin",
            retain_bytes=False,
            require_frozen=True,
        )
    except Exception as error:
        raise NativeActivationV2RunnerError(
            f"runtime copied-local {role} owned origin differs"
        ) from error
    if (
        receipt_seal.sha256 != receipt_sha256
        or current_seal.sha256 != receipt_sha256
        or receipt_seal.sha256 != current_seal.sha256
    ):
        raise NativeActivationV2RunnerError(
            f"runtime copied-local {role} bytes differ"
        )
    return receipt_sha256


def _certify_runtime_import_closure(
    *,
    activation: Any,
    authority: Any,
    local_release_import_closure: Mapping[str, Any],
    native: Any,
    prompt_builder: Any,
    autoencoder_class: Any,
    auto_tokenizer_class: Any,
    text_encoder_class: Any,
    renderer_model_class: Any,
    vae_encode: Callable[..., Any],
    prompt_clean: Callable[[str], str],
) -> Mapping[str, str]:
    """Bind runtime objects to the exact implementation files in receipts."""

    local_release_sha256 = local_release_import_closure.get("release_sha256")
    if not isinstance(local_release_sha256, Mapping):
        raise NativeActivationV2RunnerError(
            "runtime local release digest closure differs"
        )

    observed_objects = {
        "vae_code": Path(vae_encode.__code__.co_filename).resolve(strict=True),
        "autoencoder_class_module": Path(
            inspect.getfile(autoencoder_class)
        ).resolve(strict=True),
        "tokenizer_code": Path(native.legacy.__file__).resolve(strict=True),
        "renderer_code": Path(inspect.getfile(renderer_model_class)).resolve(
            strict=True
        ),
        "prompt_builder_code": Path(prompt_builder.__file__).resolve(strict=True),
        "native_prompt_code": Path(native.__file__).resolve(strict=True),
        "prompt_cleaner_code": Path(prompt_clean.__code__.co_filename).resolve(
            strict=True
        ),
        "auto_tokenizer_module": Path(
            inspect.getfile(auto_tokenizer_class)
        ).resolve(strict=True),
        "text_encoder_class_module": Path(
            inspect.getfile(text_encoder_class)
        ).resolve(strict=True),
        "python_executable": Path(sys.executable).resolve(strict=True),
    }
    digests: dict[str, str] = {}
    for case_id in EXECUTION_CASES:
        case = authority.cases[case_id]
        reference = activation._bound_authority_json_v2(
            case,
            artifact_key="vae_reference_receipt",
            path=case.reference_receipt_path,
            expected_sha256=case.reference_receipt_sha256,
            label=f"{case_id} runtime VAE receipt",
        )
        prompt = activation._bound_authority_json_v2(
            case,
            artifact_key="prompt_receipt",
            path=case.prompt_receipt_path,
            expected_sha256=case.prompt_receipt_sha256,
            label=f"{case_id} runtime prompt receipt",
        )
        vae_contract = reference.get("vae_contract")
        prompt_contract = prompt.get("prompt_contract")
        if not isinstance(vae_contract, Mapping) or not isinstance(
            prompt_contract, Mapping
        ):
            raise NativeActivationV2RunnerError(
                f"{case_id} runtime implementation receipt differs"
            )
        expected_rows = {
            "vae_code": (
                vae_contract.get("vae_code_path"),
                vae_contract.get("vae_code_sha256"),
            ),
            "autoencoder_class_module": (
                vae_contract.get("autoencoder_class_module_path"),
                vae_contract.get("autoencoder_class_module_sha256"),
            ),
            **{
                key: (
                    prompt_contract.get(f"{key}_path"),
                    prompt_contract.get(f"{key}_sha256"),
                )
                for key in (
                    "tokenizer_code",
                    "renderer_code",
                    "prompt_builder_code",
                    "native_prompt_code",
                    "prompt_cleaner_code",
                    "auto_tokenizer_module",
                    "text_encoder_class_module",
                    "python_executable",
                )
            },
        }
        for key, path in observed_objects.items():
            expected_path, expected_sha = expected_rows[key]
            if key in _COPIED_LOCAL_PROMPT_ROLE_BINDINGS:
                certified_sha = _certify_copied_local_prompt_role_v2(
                    activation=activation,
                    role=key,
                    receipt_path=expected_path,
                    receipt_sha256=expected_sha,
                    observed_current_path=path,
                    expected_current_pin_sha256=local_release_sha256.get(
                        "dependency:"
                        + _COPIED_LOCAL_PROMPT_ROLE_BINDINGS[key][
                            "release_relative_path"
                        ]
                    ),
                )
            else:
                if (
                    not isinstance(expected_path, str)
                    or not isinstance(expected_sha, str)
                    or path != Path(expected_path)
                    or _sha256_file(path) != expected_sha
                ):
                    raise NativeActivationV2RunnerError(
                        f"{case_id} runtime import {key} differs"
                    )
                certified_sha = expected_sha
            previous = digests.setdefault(key, certified_sha)
            if previous != certified_sha:
                raise NativeActivationV2RunnerError(
                    f"{case_id} runtime import receipt disagrees"
                )
    return digests


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    packet_requested = Path(args.authority_packet).expanduser()
    ledger_requested = Path(args.external_ledger).expanduser()
    if (
        not packet_requested.is_absolute()
        or packet_requested.is_symlink()
        or not ledger_requested.is_absolute()
        or ledger_requested.is_symlink()
    ):
        raise NativeActivationV2RunnerError("authority paths must be plain absolute files")
    packet_path = packet_requested.resolve(strict=True)
    ledger_path = ledger_requested.resolve(strict=True)
    if packet_path != packet_requested or ledger_path != ledger_requested:
        raise NativeActivationV2RunnerError("authority path identity differs")

    # This is the only route to the code below.  In the moving candidate the
    # compiled roots are None, so execution stops here before Torch/model/dist.
    preflight_receipt = cpu_preflight(
        authority_packet=packet_path,
        external_ledger=ledger_path,
    )
    output_dir = _fresh_output_dir(args.output_dir)
    load_lock_initial = _certify_serialized_host_load_lock()
    miopen_cache_initial = _activate_miopen_cache_pre_torch(
        output_dir=output_dir,
        cache_root_value=args.miopen_cache_root,
        local_tmp_root_value=args.miopen_local_tmp_root,
    )
    cache_preflight = preflight_receipt.get("miopen_cache_preflight")
    scheduler_tmpdir_normalization = preflight_receipt.get(
        "scheduler_tmpdir_normalization"
    )
    cpu_preflight_local_tmp_empty_proof = (
        cache_preflight.get("cpu_preflight_local_tmp_empty_proof")
        if isinstance(cache_preflight, Mapping)
        else None
    )
    if (
        not isinstance(cache_preflight, Mapping)
        or not isinstance(scheduler_tmpdir_normalization, Mapping)
        or scheduler_tmpdir_normalization
        != miopen_cache_initial["scheduler_tmpdir_normalization"]
        or cache_preflight.get("scheduler_tmpdir_normalization")
        != scheduler_tmpdir_normalization
        or cache_preflight.get("root_path")
        != miopen_cache_initial["root"]["path"]
        or cache_preflight.get("root_identity") != miopen_cache_initial["root"]
        or cache_preflight.get("directory_identities")
        != miopen_cache_initial["role_directory_identities"]
        or cache_preflight.get("local_tmp_root_identity")
        != miopen_cache_initial["local_tmp_root"]
        or cache_preflight.get("local_tmp_parent_identity")
        != miopen_cache_initial["local_tmp_parent_identity"]
        or cache_preflight.get("local_tmp_directory_identities")
        != miopen_cache_initial["local_tmp_role_directory_identities"]
        or cache_preflight.get("launcher_local_tmp_fresh_empty_proof")
        != miopen_cache_initial["launcher_local_tmp_fresh_empty_proof"]
        or cpu_preflight_local_tmp_empty_proof
        != miopen_cache_initial["worker_pre_torch_local_tmp_empty_proof"]
        or cache_preflight.get("library") != miopen_cache_initial["library"]
        or cache_preflight.get(
            "official_variables_bound_to_launcher_bootstrap"
        )
        is not True
    ):
        raise NativeActivationV2RunnerError("MIOpen CPU preflight receipt differs")

    import oracle_regeneration_native_runtime_activation_v2 as activation
    import infer_native_branch_homotopy_canary as prompt_builder
    import infer_native_identity_generation_canary as native
    import infer_source_kv_carrier_oracle as source_audit
    import infer_native_self_guided_action_field_canary as freeze_provider
    import source_self_native_ref_contrastive_v3 as native_schedule_contract
    import tri_branch_unipc as sampler_contract
    import tools as tools_package
    from tools import materialize_vae

    _strong_model_freeze_certificate = (
        freeze_provider._strong_model_freeze_certificate
    )
    local_import_closure = _certify_local_release_import_closure(
        preflight_receipt=preflight_receipt,
        activation=activation,
        native=native,
        prompt_builder=prompt_builder,
        source_audit=source_audit,
        freeze_provider=freeze_provider,
        sampler_contract=sampler_contract,
        native_schedule_contract=native_schedule_contract,
        tools_package=tools_package,
        materialize_vae=materialize_vae,
    )

    authority = activation.load_compiled_activation_authority_v2(
        packet_path, ledger_path
    )
    if (
        tuple(authority.cases) != CASE_ORDER
        or tuple(preflight_receipt.get("cases", ())) != CASE_ORDER
        or tuple(activation.EXPECTED_ARMS_E02) != ARM_ORDER_BY_CASE["e02"]
        or tuple(activation.EXPECTED_ARMS_E03) != ARM_ORDER_BY_CASE["e03"]
        or activation.EXPECTED_E03_FROZEN_BASE_PATH != E03_FROZEN_BASE_PATH
        or activation.EXPECTED_E03_FROZEN_BASE_SHA256 != E03_FROZEN_BASE_SHA256
    ):
        raise NativeActivationV2RunnerError("activation cases/order differ")
    material_preflight = {
        case_id: activation.preflight_case_material_receipts_v2(
            authority, case_id=case_id
        )
        for case_id in CASE_ORDER
    }
    checkpoint_manifest = activation._plain_absolute_file(
        args.checkpoint_content_manifest,
        label="runtime checkpoint content manifest",
    )
    if any(
        Path(str(row["checkpoint_content_manifest_path"])) != checkpoint_manifest
        for case_id, row in material_preflight.items()
        if case_id in EXECUTION_CASES
    ):
        raise NativeActivationV2RunnerError(
            "runtime checkpoint manifest differs from material receipts"
        )
    if (
        _SHA1.fullmatch(args.expected_bernini_commit) is None
        or _SHA1.fullmatch(args.expected_veomni_commit) is None
        or _SHA256.fullmatch(args.expected_checkpoint_tree_sha256) is None
        or args.expected_bernini_commit
        != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT
        or args.expected_veomni_commit
        != native.legacy.trainer.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256
        != native.legacy.trainer.CHECKPOINT_TREE_SHA256
    ):
        raise NativeActivationV2RunnerError("source/checkpoint CLI identity differs")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
        inference_source_hashes = native.legacy.validate_inference_source_files(
            bernini_root
        )
        checkpoint_identity_initial = source_audit.validate_checkpoint_content(
            checkpoint, checkpoint_manifest
        )
    except Exception as error:
        raise NativeActivationV2RunnerError(str(error)) from error
    checkpoint_identity_sha256 = _checkpoint_content_identity_sha256(
        checkpoint_identity_initial, activation=activation
    )
    if (
        int(transformer_config["num_attention_heads"]) % WORLD_SIZE
        or any(
            row["checkpoint_content_identity_sha256"]
            != checkpoint_identity_sha256
            for case_id, row in material_preflight.items()
            if case_id in EXECUTION_CASES
        )
    ):
        raise NativeActivationV2RunnerError("checkpoint/material receipt identity differs")
    activation.verify_frozen_dependency_pins_v2()
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import get_parallel_state, init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, UMT5EncoderModel
    from transformers import __version__ as transformers_version

    if (
        SYSTEM_PROMPTS.get("r2v") != native.TASK_SYSTEM_PROMPTS["r2v"]
        or SYSTEM_PROMPTS.get("vr2v") != native.TASK_SYSTEM_PROMPTS["vr2v"]
        or DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT
    ):
        raise NativeActivationV2RunnerError("native prompt constants differ")
    import_closure = _certify_runtime_import_closure(
        activation=activation,
        authority=authority,
        local_release_import_closure=local_import_closure,
        native=native,
        prompt_builder=prompt_builder,
        autoencoder_class=AutoencoderKLWan,
        auto_tokenizer_class=AutoTokenizer,
        text_encoder_class=UMT5EncoderModel,
        renderer_model_class=BerniniRendererModel,
        vae_encode=_vae_encode,
        prompt_clean=prompt_clean,
    )
    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != WORLD_SIZE
        or distributed.rank not in range(WORLD_SIZE)
        or distributed.local_rank not in range(WORLD_SIZE)
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
        or dist.is_initialized()
    ):
        raise NativeActivationV2RunnerError("runner requires fresh AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    initialized = True
    try:
        init_parallel_state(ulysses_size=WORLD_SIZE)
        device = torch.device("cuda", distributed.local_rank)
        topology_receipt = _certify_world4_sp4(
            distributed=distributed,
            torch=torch,
            dist=dist,
            parallel_state=get_parallel_state(),
        )
        load_lock_world4_rows = _all_rank_object(
            load_lock_initial,
            dist=dist,
            label="serialized host-load lock identity",
        )
        miopen_activation_rows: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(miopen_activation_rows, miopen_cache_initial)
        if (
            any(not isinstance(row, Mapping) for row in miopen_activation_rows)
            or [row.get("rank") for row in miopen_activation_rows]
            != list(range(WORLD_SIZE))
            or [row.get("local_rank") for row in miopen_activation_rows]
            != list(range(WORLD_SIZE))
            or len({row.get("user_db_path") for row in miopen_activation_rows})
            != WORLD_SIZE
            or len(
                {row.get("kernel_cache_path") for row in miopen_activation_rows}
            )
            != WORLD_SIZE
            or len({row.get("tmp_path") for row in miopen_activation_rows})
            != WORLD_SIZE
            or any(
                row.get("initial_rank_directories_empty") is not True
                for row in miopen_activation_rows
            )
            or any(
                row.get("launcher_local_tmp_fresh_empty_proof")
                != miopen_activation_rows[0].get(
                    "launcher_local_tmp_fresh_empty_proof"
                )
                or row.get("worker_pre_torch_local_tmp_empty_proof")
                != miopen_activation_rows[0].get(
                    "worker_pre_torch_local_tmp_empty_proof"
                )
                for row in miopen_activation_rows[1:]
            )
            or not isinstance(
                miopen_activation_rows[0].get("local_tmp_root"), Mapping
            )
            or any(
                Path(str(row.get("tmp_path"))).parent
                != Path(
                    str(miopen_activation_rows[0]["local_tmp_root"].get("path"))
                )
                for row in miopen_activation_rows
            )
            or any(
                row.get("root") != miopen_activation_rows[0].get("root")
                or row.get("local_tmp_root")
                != miopen_activation_rows[0].get("local_tmp_root")
                or row.get("library") != miopen_activation_rows[0].get("library")
                or row.get("scheduler_tmpdir_normalization")
                != miopen_activation_rows[0].get(
                    "scheduler_tmpdir_normalization"
                )
                for row in miopen_activation_rows[1:]
            )
        ):
            raise NativeActivationV2RunnerError(
                "WORLD4 MIOpen cache activation differs"
            )
        miopen_initial_world = _gather_miopen_cache_snapshot(
            miopen_cache_initial,
            torch=torch,
            dist=dist,
            label="initial MIOpen cache",
        )
        if any(
            tree.get("entry_count") != 0
            for row in miopen_initial_world
            for tree in row.get("trees", ())
        ):
            raise NativeActivationV2RunnerError(
                "initial WORLD4 MIOpen cache is not empty"
            )
        local_tmp_post_runtime_init_baseline_world = _gather_local_tmp_snapshot(
            miopen_cache_initial,
            torch=torch,
            dist=dist,
            label="post-runtime-init node-local tmp baseline",
        )
        _, local_tmp_post_runtime_init_baseline_world_sha256 = (
            _validate_local_tmp_world4_snapshot_rows(
                local_tmp_post_runtime_init_baseline_world,
                initial=miopen_cache_initial,
                label="post-runtime-init node-local tmp baseline replay",
            )
        )
        _all_rank_object(
            local_tmp_post_runtime_init_baseline_world_sha256,
            dist=dist,
            label="post-runtime-init node-local tmp baseline digest",
        )
        miopen_bootstrap_initial = _gather_miopen_bootstrap_snapshot(
            miopen_cache_initial,
            dist=dist,
            label="initial launcher-bootstrap MIOpen cache",
        )
        checkpoint_identity = _checkpoint_identity_rank_zero(
            activation=activation,
            source_audit=source_audit,
            checkpoint=checkpoint,
            checkpoint_manifest=checkpoint_manifest,
            dist=dist,
            rank=distributed.rank,
            label="pre-condition",
        )
        if (
            _checkpoint_content_identity_sha256(
                checkpoint_identity, activation=activation
            )
            != checkpoint_identity_sha256
        ):
            raise NativeActivationV2RunnerError("checkpoint changed before conditions")

        condition_tensors: dict[str, Mapping[str, Any]] = {}
        condition_receipts: dict[str, Mapping[str, Any]] = {}
        for case_id in EXECUTION_CASES:
            case = authority.cases[case_id]
            source_latent, references, reference_identity = (
                _materialize_source_references(
                    case,
                    authority=authority,
                    activation=activation,
                    native=native,
                    source_audit=source_audit,
                    materialize_vae=materialize_vae,
                    autoencoder_class=AutoencoderKLWan,
                    vae_encode=_vae_encode,
                    checkpoint=checkpoint,
                    torch=torch,
                    dist=dist,
                    device=device,
                )
            )
            prompts, prompt_identity = _materialize_prompts(
                case,
                authority=authority,
                activation=activation,
                native=native,
                prompt_builder=prompt_builder,
                prompt_clean=prompt_clean,
                tokenizer_class=AutoTokenizer,
                renderer_config_class=BerniniRendererConfig,
                renderer_model_class=BerniniRendererModel,
                checkpoint=checkpoint,
                bernini_root=bernini_root,
                torch=torch,
                dist=dist,
                device=device,
            )
            condition_tensors[case_id] = {
                "source": source_latent,
                "references": references,
                "prompts": prompts,
            }
            condition_receipts[case_id] = {
                "source_references": reference_identity,
                "prompts": prompt_identity,
                "live_binding": _revalidate_live_case(
                    activation=activation,
                    authority=authority,
                    case=case,
                    source_latent=source_latent,
                    references=references,
                    prompts=prompts,
                ),
            }

        miopen_loaded_library = _certify_loaded_miopen_library()
        _all_rank_object(
            miopen_loaded_library,
            dist=dist,
            label="loaded Torch-bundled MIOpen library",
        )

        generated: dict[str, dict[str, Any]] = {
            case_id: {} for case_id in EXECUTION_CASES
        }
        noises: dict[str, dict[str, Any]] = {
            case_id: {} for case_id in EXECUTION_CASES
        }
        arm_receipts: dict[str, dict[str, Mapping[str, Any]]] = {
            case_id: {} for case_id in EXECUTION_CASES
        }
        runtime_traces: dict[str, dict[str, Mapping[str, Any]]] = {
            case_id: {} for case_id in EXECUTION_CASES
        }
        miopen_arm_snapshots: dict[str, dict[str, Mapping[str, Any]]] = {
            case_id: {} for case_id in EXECUTION_CASES
        }
        local_tmp_arm_snapshots: dict[str, dict[str, Mapping[str, Any]]] = {
            case_id: {} for case_id in EXECUTION_CASES
        }
        model_state_sha256: Optional[str] = None
        for case_id in EXECUTION_CASES:
            case = authority.cases[case_id]
            tensors = condition_tensors[case_id]
            for arm in ARM_ORDER_BY_CASE[case_id]:
                cache_before_arm = _gather_miopen_cache_snapshot(
                    miopen_cache_initial,
                    torch=torch,
                    dist=dist,
                    label=f"{case_id}/{arm} pre-arm MIOpen cache",
                )
                local_tmp_before_arm = _gather_local_tmp_snapshot(
                    miopen_cache_initial,
                    torch=torch,
                    dist=dist,
                    label=f"{case_id}/{arm} pre-arm node-local tmp",
                )
                if _certify_serialized_host_load_lock() != load_lock_initial:
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} serialized host-load lock changed"
                    )
                binding_before = _revalidate_live_case(
                    activation=activation,
                    authority=authority,
                    case=case,
                    source_latent=tensors["source"],
                    references=tensors["references"],
                    prompts=tensors["prompts"],
                )
                checkpoint_before_arm = _checkpoint_identity_rank_zero(
                    activation=activation,
                    source_audit=source_audit,
                    checkpoint=checkpoint,
                    checkpoint_manifest=checkpoint_manifest,
                    dist=dist,
                    rank=distributed.rank,
                    label=f"{case_id}/{arm} pre-arm",
                )
                if (
                    _checkpoint_content_identity_sha256(
                        checkpoint_before_arm, activation=activation
                    )
                    != checkpoint_identity_sha256
                ):
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} checkpoint changed before arm"
                    )
                stored, noise, arm_receipt, trace = _run_one_arm(
                    case=case,
                    arm=arm,
                    authority=authority,
                    source_latent=tensors["source"],
                    references=tensors["references"],
                    prompts=tensors["prompts"],
                    activation=activation,
                    native=native,
                    sampler_contract=sampler_contract,
                    strong_freeze=_strong_model_freeze_certificate,
                    renderer_config_class=BerniniRendererConfig,
                    renderer_model_class=BerniniRendererModel,
                    t5_encoder_class=UMT5EncoderModel,
                    wan_diffusion=wan_diffusion,
                    bernini_root=bernini_root,
                    checkpoint=checkpoint,
                    bernini_revision=bernini_revision,
                    torch=torch,
                    dist=dist,
                    device=device,
                )
                if _certify_serialized_host_load_lock() != load_lock_initial:
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} serialized host-load lock changed"
                    )
                cache_after_arm = _gather_miopen_cache_snapshot(
                    miopen_cache_initial,
                    torch=torch,
                    dist=dist,
                    label=f"{case_id}/{arm} post-arm MIOpen cache",
                )
                local_tmp_after_arm = _gather_local_tmp_snapshot(
                    miopen_cache_initial,
                    torch=torch,
                    dist=dist,
                    label=f"{case_id}/{arm} post-arm node-local tmp",
                )
                binding_after = _revalidate_live_case(
                    activation=activation,
                    authority=authority,
                    case=case,
                    source_latent=tensors["source"],
                    references=tensors["references"],
                    prompts=tensors["prompts"],
                )
                if binding_after != binding_before:
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} live binding changed"
                    )
                checkpoint_after_arm = _checkpoint_identity_rank_zero(
                    activation=activation,
                    source_audit=source_audit,
                    checkpoint=checkpoint,
                    checkpoint_manifest=checkpoint_manifest,
                    dist=dist,
                    rank=distributed.rank,
                    label=f"{case_id}/{arm} post-arm",
                )
                if (
                    _checkpoint_content_identity_sha256(
                        checkpoint_after_arm, activation=activation
                    )
                    != checkpoint_identity_sha256
                ):
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} checkpoint changed during arm"
                    )
                state_sha = str(arm_receipt["freeze_certificate"]["state_content_sha256"])
                if model_state_sha256 is None:
                    model_state_sha256 = state_sha
                elif state_sha != model_state_sha256:
                    raise NativeActivationV2RunnerError(
                        "fresh arm model state differs across matched arms"
                    )
                trace_digest = _canonical_sha256(trace)
                _all_rank_object(
                    trace_digest,
                    dist=dist,
                    label=f"{case_id}/{arm} runtime trace",
                )
                full_arm_receipt = {
                    **dict(arm_receipt),
                    "live_binding_before": binding_before,
                    "live_binding_after": binding_after,
                    "checkpoint_identity_sha256_before_after": checkpoint_identity_sha256,
                    "runtime_trace_sha256": trace_digest,
                    "miopen_cache": {
                        "same_rank_namespace_reused": True,
                        "before_world4": cache_before_arm,
                        "after_world4": cache_after_arm,
                    },
                    "node_local_tmp": {
                        "same_rank_namespace_reused": True,
                        "before_world4": local_tmp_before_arm,
                        "after_world4": local_tmp_after_arm,
                    },
                }
                _all_rank_object(
                    _canonical_sha256(full_arm_receipt),
                    dist=dist,
                    label=f"{case_id}/{arm} arm receipt",
                )
                generated[case_id][arm] = stored
                noises[case_id][arm] = noise
                arm_receipts[case_id][arm] = full_arm_receipt
                runtime_traces[case_id][arm] = trace
                miopen_arm_snapshots[case_id][arm] = {
                    "before_world4": cache_before_arm,
                    "after_world4": cache_after_arm,
                }
                local_tmp_arm_snapshots[case_id][arm] = {
                    "before_world4": local_tmp_before_arm,
                    "after_world4": local_tmp_after_arm,
                }

        e02_base = arm_receipts["e02"][ARM_OFFICIAL]
        e02_local = arm_receipts["e02"][ARM_LOCAL]
        if (
            e02_base["sample_kwargs_digest"] != e02_local["sample_kwargs_digest"]
            or e02_base["official_gaussian_raw_sha256"]
            != e02_local["official_gaussian_raw_sha256"]
            or e02_base["official_gaussian_content_sha256"]
            != e02_local["official_gaussian_content_sha256"]
            or noises["e02"][ARM_OFFICIAL].generator_initial_seed
            != noises["e02"][ARM_LOCAL].generator_initial_seed
            or runtime_traces["e02"][ARM_LOCAL].get(
                "outside_G_official_bytes_exact_all_steps"
            )
            is not True
        ):
            raise NativeActivationV2RunnerError(
                "e02 matched-arm Gaussian/input/local trace differs"
            )

        staging_context: Optional[tempfile.TemporaryDirectory[str]] = None
        staged_rows: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None
        stage_status: list[Any] = [None]
        if distributed.rank == 0:
            try:
                staging_context = tempfile.TemporaryDirectory(
                    prefix=f".{output_dir.name}.staging-",
                    dir=str(output_dir.parent),
                )
                staging_dir = Path(staging_context.name)
                staging_dir.chmod(0o700)
                staged_rows = _stage_rank_zero_outputs(
                    staging_dir=staging_dir,
                    output_dir=output_dir,
                    generated=generated,
                    noises=noises,
                    arm_receipts=arm_receipts,
                    authority=authority,
                    activation=activation,
                    native=native,
                    autoencoder_class=AutoencoderKLWan,
                    vae_decode=_vae_decode,
                    save_output_fn=save_output,
                    materialize_vae=materialize_vae,
                    checkpoint=checkpoint,
                    torch=torch,
                    device=device,
                )
                stage_status[0] = {"ok": True}
            except Exception as error:
                stage_status[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(stage_status, src=0)
        if not isinstance(stage_status[0], Mapping) or stage_status[0].get("ok") is not True:
            if staging_context is not None:
                staging_context.cleanup()
            raise NativeActivationV2RunnerError(
                f"rank-zero output staging failed: {stage_status[0]}"
            )

        checkpoint_identity_final = _checkpoint_identity_rank_zero(
            activation=activation,
            source_audit=source_audit,
            checkpoint=checkpoint,
            checkpoint_manifest=checkpoint_manifest,
            dist=dist,
            rank=distributed.rank,
            label="post-decode",
        )
        if (
            _checkpoint_content_identity_sha256(
                checkpoint_identity_final, activation=activation
            )
            != checkpoint_identity_sha256
        ):
            raise NativeActivationV2RunnerError("checkpoint changed during run/decode")
        activation.revalidate_compiled_activation_authority_v2(authority)
        for case_id in EXECUTION_CASES:
            tensors = condition_tensors[case_id]
            final_binding = _revalidate_live_case(
                activation=activation,
                authority=authority,
                case=authority.cases[case_id],
                source_latent=tensors["source"],
                references=tensors["references"],
                prompts=tensors["prompts"],
            )
            if final_binding != condition_receipts[case_id]["live_binding"]:
                raise NativeActivationV2RunnerError(
                    f"{case_id} final live binding changed"
                )
        activation.verify_frozen_dependency_pins_v2()
        final_local_import_closure = _certify_local_release_import_closure(
            preflight_receipt=preflight_receipt,
            activation=activation,
            native=native,
            prompt_builder=prompt_builder,
            source_audit=source_audit,
            freeze_provider=freeze_provider,
            sampler_contract=sampler_contract,
            native_schedule_contract=native_schedule_contract,
            tools_package=tools_package,
            materialize_vae=materialize_vae,
        )
        final_import_closure = _certify_runtime_import_closure(
            activation=activation,
            authority=authority,
            local_release_import_closure=final_local_import_closure,
            native=native,
            prompt_builder=prompt_builder,
            autoencoder_class=AutoencoderKLWan,
            auto_tokenizer_class=AutoTokenizer,
            text_encoder_class=UMT5EncoderModel,
            renderer_model_class=BerniniRendererModel,
            vae_encode=_vae_encode,
            prompt_clean=prompt_clean,
        )
        if (
            final_local_import_closure != local_import_closure
            or final_import_closure != import_closure
            or _certify_serialized_host_load_lock() != load_lock_initial
        ):
            raise NativeActivationV2RunnerError("runtime import closure changed")
        final_miopen_library = _certify_loaded_miopen_library()
        if final_miopen_library != miopen_loaded_library:
            raise NativeActivationV2RunnerError("MIOpen library changed during run")
        miopen_cache_receipt = _seal_miopen_cache_receipt_world4(
            initial=miopen_cache_initial,
            activation_rows=miopen_activation_rows,
            initial_world4=miopen_initial_world,
            bootstrap_initial=miopen_bootstrap_initial,
            arm_snapshots=miopen_arm_snapshots,
            loaded_library=final_miopen_library,
            output_dir=output_dir,
            torch=torch,
            dist=dist,
            rank=distributed.rank,
        )
        local_tmp_receipt = _seal_local_tmp_receipt_world4(
            initial=miopen_cache_initial,
            activation_rows=miopen_activation_rows,
            cpu_preflight_local_tmp_empty_proof=(
                cpu_preflight_local_tmp_empty_proof
            ),
            post_runtime_init_baseline_world4=(
                local_tmp_post_runtime_init_baseline_world
            ),
            post_runtime_init_baseline_world4_sha256=(
                local_tmp_post_runtime_init_baseline_world_sha256
            ),
            arm_snapshots=local_tmp_arm_snapshots,
            persistent_cache_receipt=miopen_cache_receipt,
            output_dir=output_dir,
            torch=torch,
            dist=dist,
            rank=distributed.rank,
        )

        publish_status: list[Any] = [None]
        if distributed.rank == 0:
            try:
                if staged_rows is None or staging_context is None:
                    raise NativeActivationV2RunnerError(
                        "rank-zero staged artifact state is absent"
                    )
                outputs = _publish_staged_outputs(
                    output_dir=output_dir, staged=staged_rows
                )
                e02 = authority.cases["e02"]
                e03 = authority.cases["e03"]
                case_receipts = {
                    "e02": {
                        "decision": e02.decision,
                        "executed": True,
                        "source_iid": e02.source_iid,
                        "source_video_path": str(e02.source_video_path),
                        "source_video_sha256": e02.source_sha256,
                        "action_caption": e02.action_caption,
                        "action_caption_sha256": e02.action_caption_sha256,
                        "structured_action_program_sha256": (
                            e02.structured_action_program_sha256
                        ),
                        "seed": e02.seed,
                        "arms": list(ARM_ORDER_BY_CASE["e02"]),
                        "condition_receipts": condition_receipts["e02"],
                        "arm_receipts": arm_receipts["e02"],
                        "runtime_traces": runtime_traces["e02"],
                        "outputs": outputs["e02"],
                        "selection": "SIDE_BY_SIDE_DIAGNOSTIC_NO_SELECTION",
                    },
                    "e03": {
                        "decision": "ABSTAIN_KEEP_BASE",
                        "executed": False,
                        "source_iid": e03.source_iid,
                        "source_video_path": str(e03.source_video_path),
                        "source_video_sha256": e03.source_sha256,
                        "action_caption": e03.action_caption,
                        "action_caption_sha256": e03.action_caption_sha256,
                        "structured_action_program_sha256": (
                            e03.structured_action_program_sha256
                        ),
                        "seed": e03.seed,
                        "arms": [],
                        "condition_receipts": None,
                        "arm_receipts": {},
                        "runtime_traces": {},
                        "outputs": {},
                        "kept_frozen_base": {
                            "path": str(e03.kept_frozen_base_path),
                            "sha256": e03.kept_frozen_base_sha256,
                        },
                        "selection": "ABSTAIN_KEEP_BASE",
                    },
                }
                receipt: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "method": METHOD,
                    "scope": "experimental diagnostic canary only",
                    "authority": {
                        "kind": "diagnostic_exact_packet_and_code_review_trust_root",
                        "packet_id": authority.packet_id,
                        "packet_sha256": authority.packet_sha256,
                        "external_ledger_sha256": authority.ledger_sha256,
                        "formal_authority": False,
                        "training_authority": False,
                    },
                    "cpu_preflight": dict(preflight_receipt),
                    "material_preflight": material_preflight,
                    "cases": case_receipts,
                    "matched_e02_contract": {
                        "same_seed": True,
                        "same_official_gaussian_raw_bytes": True,
                        "same_sample_kwargs_and_live_conditions": True,
                        "fresh_model_and_scheduler_per_arm": True,
                        "only_experimental_difference": "scheduled source-reference R2V4 velocity inside exact G",
                        "outside_G_claim": "within local arm each scheduler model_output is byte-exact to that same-step official V2V output outside exact G",
                        "cross_arm_final_outside_pixel_identity_claimed": False,
                    },
                    "source_revisions": {
                        "bernini_root": str(bernini_root),
                        "veomni_root": str(veomni_root),
                        "bernini_revision": bernini_revision,
                        "veomni_revision": veomni_revision,
                        "inference_source_hashes": inference_source_hashes,
                        "local_release_import_closure": local_import_closure,
                        "runtime_import_closure": import_closure,
                    },
                    "checkpoint": {
                        "content_identity": checkpoint_identity_final,
                        "content_identity_sha256": checkpoint_identity_sha256,
                        "validated_before_conditions_before_after_each_arm_and_after_decode": True,
                        "fresh_model_state_sha256": model_state_sha256,
                    },
                    "world": {
                        **dict(topology_receipt),
                        "serialized_host_load_lock": {
                            **dict(load_lock_initial),
                            "all_rank_exact": True,
                            "world4_rows_sha256": _canonical_sha256(
                                load_lock_world4_rows
                            ),
                        },
                        "miopen_rank_private_runtime_cache": dict(
                            miopen_cache_receipt
                        ),
                        "node_local_rank_private_tmp": dict(
                            local_tmp_receipt
                        ),
                    },
                    "runtime_versions": {
                        "python": sys.version,
                        "torch": str(torch.__version__),
                        "torch_hip": str(torch.version.hip),
                        "diffusers": str(diffusers_version),
                        "transformers": str(transformers_version),
                        "miopen_library": dict(final_miopen_library),
                    },
                    "scientific_boundary": {
                        "source_reference_r2v4_regeneration_expert": True,
                        "self_generated_anchor_tensor_used": False,
                        "anchor_used_only_as_review_context": True,
                        "anchor_reference_or_quotient_arm_deferred": True,
                        "global_source_reference_r2v4_upper_bound_arm_deferred": True,
                        "local_G_step0_domain_separated_gaussian_arm_deferred": True,
                        "this_run_tests_self_generated_anchor_action_representation": False,
                    },
                    "output_contract": {
                        "side_by_side_only": True,
                        "automatic_selection": False,
                        "background_cosine_selection": False,
                        "e03_keep_base": True,
                        "post_run_output_bound_independent_review_required_for_any_selection": True,
                        "no_overwrite_publish": True,
                        "miopen_cache_is_retained_output_sibling_engineering_evidence": True,
                        "miopen_cache_is_not_a_scientific_output_artifact": True,
                        "node_local_tmp_is_ephemeral_engineering_evidence": True,
                        "node_local_tmp_durability_guaranteed": False,
                        "node_local_tmp_runner_cleanup_performed": False,
                        "node_local_tmp_node_lifetime_only": True,
                        "node_local_tmp_observed_and_replayed_before_WORLD4_step_exit": True,
                        "node_local_tmp_existence_after_process_or_step_exit_guaranteed": False,
                        "node_local_tmp_baseline_is_observation_not_allowlist": True,
                        "node_local_tmp_continuous_monitoring_claimed": False,
                        "node_local_tmp_between_observation_transients_may_be_unrecorded": True,
                        "node_local_tmp_full_receipt_embedded_in_durable_output_receipt": True,
                    },
                    "training": False,
                    "optimizer": False,
                    "backward": False,
                    "parameter_update": False,
                    "flowedit": False,
                    "connected_route": False,
                    "learned_gate": False,
                    "automatic_replacement": False,
                    "selection_authority": None,
                }
                receipt["receipt_digest"] = _canonical_sha256(receipt)
                freeze_receipt = _freeze_output_release(
                    output_dir=output_dir, receipt=receipt
                )
                cache_receipt_now = _owned_file_identity(
                    Path(str(miopen_cache_receipt["path"])),
                    label="post-output MIOpen cache receipt",
                )
                local_tmp_receipt_now = _owned_file_identity(
                    Path(str(local_tmp_receipt["path"])),
                    label="post-output node-local tmp receipt",
                )
                if any(
                    cache_receipt_now.get(key)
                    != miopen_cache_receipt["file_identity"].get(key)
                    for key in ("sha256", "size", "mode", "nlink")
                ) or any(
                    local_tmp_receipt_now.get(key)
                    != local_tmp_receipt["file_identity"].get(key)
                    for key in ("sha256", "size", "mode", "nlink")
                ):
                    raise NativeActivationV2RunnerError(
                        "MIOpen cache/local tmp receipt changed during output freeze"
                    )
                publish_status[0] = {
                    "ok": True,
                    "receipt": receipt,
                    "release": freeze_receipt,
                }
            except Exception as error:
                publish_status[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            finally:
                if staging_context is not None:
                    staging_context.cleanup()
        dist.broadcast_object_list(publish_status, src=0)
        if (
            not isinstance(publish_status[0], Mapping)
            or publish_status[0].get("ok") is not True
        ):
            raise NativeActivationV2RunnerError(
                f"rank-zero output publish failed: {publish_status[0]}"
            )
        post_output_cache_world4 = _gather_miopen_cache_snapshot(
            miopen_cache_initial,
            torch=torch,
            dist=dist,
            label="post-output final MIOpen cache",
        )
        post_output_bootstrap = _gather_miopen_bootstrap_snapshot(
            miopen_cache_initial,
            dist=dist,
            label="post-output launcher-bootstrap MIOpen cache",
        )
        post_output_cache_receipt = _collective_owned_file_identity(
            Path(str(miopen_cache_receipt["path"])),
            dist=dist,
            label="post-output all-rank MIOpen cache receipt",
        )
        _collective_exact_directory_names(
            Path(str(local_tmp_receipt["root_identity"]["path"])),
            expected=_expected_miopen_local_tmp_directory_names()
            | {MIOPEN_LOCAL_TMP_RECEIPT_NAME},
            dist=dist,
            label="post-output node-local tmp",
        )
        post_output_local_tmp_world4 = _gather_local_tmp_snapshot(
            miopen_cache_initial,
            torch=torch,
            dist=dist,
            label="post-output final node-local tmp",
        )
        post_output_local_tmp_receipt = _collective_owned_file_identity(
            Path(str(local_tmp_receipt["path"])),
            dist=dist,
            label="post-output all-rank node-local tmp receipt",
        )
        if (
            post_output_cache_world4 != miopen_cache_receipt["final_world4"]
            or post_output_bootstrap
            != miopen_cache_receipt["launcher_bootstrap_final"]
            or _certify_scheduler_tmpdir_normalization()
            != miopen_cache_receipt["scheduler_tmpdir_normalization"]
            or post_output_local_tmp_world4
            != local_tmp_receipt["final_world4"]
            or _private_directory_identity(
                Path(str(local_tmp_receipt["root_identity"]["path"])),
                label="post-output node-local tmp root",
            )
            != local_tmp_receipt["root_identity"]
            or _node_local_tmp_parent_identity()
            != local_tmp_receipt["parent_identity"]
            or any(
                post_output_cache_receipt.get(key)
                != miopen_cache_receipt["file_identity"].get(key)
                for key in ("sha256", "size", "mode", "nlink")
            )
            or any(
                post_output_local_tmp_receipt.get(key)
                != local_tmp_receipt["file_identity"].get(key)
                for key in ("sha256", "size", "mode", "nlink")
            )
        ):
            raise NativeActivationV2RunnerError(
                "MIOpen cache/node-local tmp changed after output freeze"
            )
        dist.barrier()
        if distributed.rank == 0:
            print(
                json.dumps(
                    publish_status[0],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                flush=True,
            )
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_LOCAL",
    "ARM_OFFICIAL",
    "ARM_ORDER_BY_CASE",
    "AUTHORIZED_HOSTNAME",
    "AUTHORIZED_SLURM_JOB_ID",
    "CASE_ORDER",
    "EXECUTION_CASES",
    "METHOD",
    "NativeActivationV2RunnerError",
    "SCHEMA_VERSION",
    "build_parser",
    "cpu_preflight",
    "main",
]

#!/usr/bin/env python3
"""Build the deterministic ELAL-3 C2 staged-training source release.

The exact16 tensor payload remains external because it is large.  Its sealed
materializer receipt and RUN_COMPLETE receipt are archived with the runtime,
while the manifest binds the external tensor file by SHA-256, size, mode and
link count.  All archive sources are mode 0444; the compute launcher extracts
them into a fresh, independent 0444 training tree and never reuses the
materializer's 0644 execution tree.

Publication is intentionally disabled until both the C2 trainer and exact16
bundle literals have passed their independent final audits.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-elal3-c2-role-binding-training-release-v1"
ARCHIVE_FORMAT = "fixed-ustar-ascii-sorted-owner0-mtime0-record10240-v1"

TRAINER_SHA256: Optional[str] = (
    "63f35b39e60dbf2c1dd1dcecb29393c04d9f00fd0833054e7d81d40790dfe4ce"
)
TRAINER_SIZE: Optional[int] = 447_559
GATE_CONTROLLER_SHA256: Optional[str] = (
    "f4e931b1f50473a9391aa7e7e68464213aaf43e85cc5a8bee792c380c2035af1"
)
GATE_CONTROLLER_SIZE: Optional[int] = 28_107
ORIGIN_VERIFIER_SHA256: Optional[str] = (
    "07122fd71e8f170b5a50761255a664ac17fc2c66b7b8970a1c113bc8d5e605c1"
)
ORIGIN_VERIFIER_SIZE: Optional[int] = 24_717
LATENT_BUNDLE_SHA256: Optional[str] = (
    "b31d5e1594a112f965a3cebd527d5189a561e2cc2d83cfe94014872ffb94d1b8"
)
LATENT_BUNDLE_SIZE: Optional[int] = 78_277_976
LATENT_RECEIPT_SHA256: Optional[str] = (
    "a1ca0d3c015a54d61c8a71d00bc78688dab20d6592ba30ddf73b0ea18e7d70ee"
)
LATENT_RECEIPT_SIZE: Optional[int] = 52_752
LATENT_RECEIPT_DIGEST: Optional[str] = (
    "225255f5ada73848686b240c4a53001c9dd65b1373da2b293c2da8c2ec14f35d"
)
MATERIALIZER_RUN_COMPLETE_SHA256: Optional[str] = (
    "c6eee4766943c7959a2c1ad9b8b6b4e823dec054b31d2fdfb5d03aacd9f7e1ac"
)
MATERIALIZER_RUN_COMPLETE_SIZE: Optional[int] = 2_666
MATERIALIZER_RUN_COMPLETE_DIGEST: Optional[str] = (
    "186d10a0635a826ebb9bd34dcbc9af7cd23ae45881877c2d252981290edf6d6d"
)

RUNTIME_PINS: Mapping[str, tuple[Optional[str], Optional[int]]] = {
    "methods/bernini_action_editing/elal3_c2_staged_gate_controller_v1.py": (
        GATE_CONTROLLER_SHA256,
        GATE_CONTROLLER_SIZE,
    ),
    "methods/bernini_action_editing/elal3_c2_origin_receipt_verifier_v1.py": (
        ORIGIN_VERIFIER_SHA256,
        ORIGIN_VERIFIER_SIZE,
    ),
    "methods/bernini_action_editing/train_elal3_c2_simulator_role_pair_v1.py": (
        TRAINER_SHA256,
        TRAINER_SIZE,
    ),
    "methods/bernini_action_editing/train_elal3_c1_simulator_overfit_v1.py": (
        "521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3",
        90_600,
    ),
    "methods/bernini_action_editing/elal3_c0_v1.py": (
        "70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862",
        31_330,
    ),
    "methods/bernini_action_editing/elal3_simulator_c2_label_v1.py": (
        "1f09670a3dd2eae09cd27dbb5fe28c913f618d096a04a64fb0cf1dc9b6e1ec11",
        76_939,
    ),
    "methods/bernini_action_editing/materialize_elal3_simulator_c2_vae_v1.py": (
        "b9142b3da63499163623248d902c51d01bf5c5295ff171125e3c614dea788c0f",
        50_334,
    ),
    "methods/bernini_action_editing/tools/materialize_vae.py": (
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
        32_195,
    ),
    "methods/bernini_action_editing/tools/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        0,
    ),
    "methods/bernini_action_editing/tools/build_renderer_dataset.py": (
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
        31_012,
    ),
    "methods/bernini_action_editing/train_lora.py": (
        "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
        66_931,
    ),
    "methods/bernini_action_editing/packed_preservation_lora_v2.py": (
        "61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6",
        30_419,
    ),
    "methods/bernini_action_editing/source_self_runtime.py": (
        "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f",
        36_607,
    ),
    "methods/bernini_action_editing/inference_sigma_strata.py": (
        "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
        17_956,
    ),
}

DERIVATIVE_RELATIVE = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_simulator_optimizer_diagnostic_authority_v1.json"
)
MODEL_RELATIVE = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_real_model_authority_v1.json"
)
CONTRACT_RELATIVE = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_role_binding_experiment_contract_v1.json"
)
LATENT_RECEIPT_MEMBER = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_exact16_latent_bundle_receipt_v1.json"
)
MATERIALIZER_RUN_MEMBER = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_exact16_materializer_run_complete_v1.json"
)
CHECKPOINT_EXACT23_RELATIVE = (
    "methods/bernini_action_editing/audits/"
    "bernini_r13_ff4c5d4_checkpoint.sha256"
)
CHECKPOINT_EXACT23_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
CHECKPOINT_EXACT23_SIZE = 2_350

DERIVATIVE_SHA256 = "543aedd714c7a48c48b4dcc19d1dd6a8bba37d1edda9b1fa195083659380c64a"
DERIVATIVE_SIZE = 1_900
DERIVATIVE_DIGEST = "936e91cf3d1d39dd7f45d5f7a4d510dadcbcb4c2f89a8d22581638fccdefd599"
MODEL_SHA256 = "312d74a830ebec675af39b74e31c696ca188068a0e0ac9058745a537961c260d"
MODEL_SIZE = 3_292
MODEL_DIGEST = "c2c0c9037dea2fd56aa13ac56416bf38c6167686c75b69f0b4b568c82e670c1f"
CONTRACT_SHA256 = "92d700bde0ff9c644f998344d3fecb48bc7c0361f6e948a93c42b924245b25f8"
CONTRACT_SIZE = 8_553
CONTRACT_DIGEST = "18462dcfbeb017e48a7ed6816559667fa8de1911081261cdc103bc6dd9a229d6"
PACKET_MANIFEST_SHA256 = "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"

RUN_ASSIGNMENTS = (
    {
        "holder_job_id": "141620",
        "node": "auh7-1b-gpu-226",
        "arm_id": "A_duplicate_control",
        "recipe": "target_duplicate_exact2",
        "seed": 20260821,
    },
    {
        "holder_job_id": "141618",
        "node": "auh7-1b-gpu-249",
        "arm_id": "B_paired_role",
        "recipe": "target_and_role_swap_exact2",
        "seed": 20260821,
    },
    {
        "holder_job_id": "141619",
        "node": "auh7-1b-gpu-257",
        "arm_id": "B_paired_role_replica",
        "recipe": "target_and_role_swap_exact2",
        "seed": 20260822,
    },
)

_SHA = re.compile(r"^[0-9a-f]{64}$")


class ELAL3C2TrainingReleaseError(RuntimeError):
    """The requested release is not the exact audited C2 closure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ELAL3C2TrainingReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ELAL3C2TrainingReleaseError(
            "value is not finite canonical ASCII JSON"
        ) from error


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_pins(
    *,
    runtime_pins: Mapping[str, tuple[Optional[str], Optional[int]]],
    bundle_sha256: Optional[str],
    bundle_size: Optional[int],
    receipt_sha256: Optional[str],
    receipt_size: Optional[int],
    receipt_digest: Optional[str],
    run_sha256: Optional[str],
    run_size: Optional[int],
    run_digest: Optional[str],
) -> None:
    for relative, (sha, size) in runtime_pins.items():
        require(_SHA.fullmatch(str(sha)) is not None, f"runtime SHA is PENDING: {relative}")
        empty_package_marker = (
            relative == "methods/bernini_action_editing/tools/__init__.py"
            and size == 0
        )
        require(
            type(size) is int and (size > 0 or empty_package_marker),
            f"runtime size is PENDING: {relative}",
        )
    for value, label in (
        (bundle_sha256, "exact16 bundle SHA"),
        (receipt_sha256, "exact16 receipt SHA"),
        (receipt_digest, "exact16 receipt digest"),
        (run_sha256, "materializer RUN_COMPLETE SHA"),
        (run_digest, "materializer RUN_COMPLETE digest"),
    ):
        require(_SHA.fullmatch(str(value)) is not None, f"{label} is PENDING")
    for value, label in (
        (bundle_size, "exact16 bundle size"),
        (receipt_size, "exact16 receipt size"),
        (run_size, "materializer RUN_COMPLETE size"),
    ):
        require(type(value) is int and value > 0, f"{label} is PENDING")


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_rdev,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_rdev,
    )


def stable_file(path: Path, *, maximum_bytes: int) -> bytes:
    require(
        path.is_absolute()
        and path != Path(path.anchor)
        and all(part not in {"", ".", ".."} for part in path.parts[1:]),
        f"non-canonical file: {path}",
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_path = Path(path.anchor)
    root_named = root_path.lstat()
    root_descriptor = os.open(root_path, directory_flags)
    held: list[int] = [root_descriptor]
    parents: list[tuple[Path, os.stat_result, int]] = [
        (root_path, root_named, root_descriptor)
    ]
    try:
        require(
            stat.S_ISDIR(root_named.st_mode)
            and _directory_identity(root_named)
            == _directory_identity(os.fstat(root_descriptor)),
            f"filesystem root identity differs: {path}",
        )
        parent_descriptor = root_descriptor
        absolute_parent = root_path
        for component in path.parts[1:-1]:
            named = os.stat(
                component, dir_fd=parent_descriptor, follow_symlinks=False
            )
            child_descriptor = os.open(
                component, directory_flags, dir_fd=parent_descriptor
            )
            held.append(child_descriptor)
            child = os.fstat(child_descriptor)
            absolute_parent = absolute_parent / component
            require(
                stat.S_ISDIR(named.st_mode)
                and _directory_identity(named) == _directory_identity(child)
                and _directory_identity(absolute_parent.lstat())
                == _directory_identity(child),
                f"held-openat parent differs: {path}",
            )
            parents.append((absolute_parent, named, child_descriptor))
            parent_descriptor = child_descriptor
        basename = path.parts[-1]
        before_name = os.stat(
            basename, dir_fd=parent_descriptor, follow_symlinks=False
        )
        require(stat.S_ISREG(before_name.st_mode), f"not a regular file: {path}")
        require(before_name.st_nlink == 1, f"multi-link file: {path}")
        require(before_name.st_size <= maximum_bytes, f"file too large: {path}")
        descriptor = os.open(basename, file_flags, dir_fd=parent_descriptor)
        held.append(descriptor)
        before = os.fstat(descriptor)

        def read_pass() -> bytes:
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    return b"".join(chunks)
                chunks.append(block)

        first = read_pass()
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = read_pass()
        after = os.fstat(descriptor)
        after_name = os.stat(
            basename, dir_fd=parent_descriptor, follow_symlinks=False
        )
        absolute_after = path.lstat()
        for absolute, parent_before, parent_fd in parents:
            require(
                _directory_identity(parent_before)
                == _directory_identity(os.fstat(parent_fd))
                and _directory_identity(absolute.lstat())
                == _directory_identity(os.fstat(parent_fd)),
                f"held-openat parent final replay differs: {path}",
            )
        require(
            first == second
            and _identity(before_name) == _identity(before)
            and _identity(before) == _identity(after)
            and _identity(after) == _identity(after_name)
            and _identity(after_name) == _identity(absolute_after),
            f"held-FD replay differs: {path}",
        )
    finally:
        for held_descriptor in reversed(held):
            os.close(held_descriptor)
    return first


def strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3C2TrainingReleaseError(f"{label} is not UTF-8 JSON") from error
    require(isinstance(value, Mapping), f"{label} must be one JSON object")
    return value


def _validate_self_digest(
    value: Mapping[str, Any], *, key: str, expected: str, label: str
) -> None:
    unsigned = dict(value)
    stored = unsigned.pop(key, None)
    require(stored == expected == object_digest(unsigned), f"{label} self digest differs")


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o444
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


def build_payload(
    repo_root: Path,
    *,
    latent_receipt_path: Path,
    materializer_run_complete_path: Path,
    train_lora_source_path: Path,
    runtime_pins: Mapping[str, tuple[Optional[str], Optional[int]]] = RUNTIME_PINS,
    bundle_sha256: Optional[str] = LATENT_BUNDLE_SHA256,
    bundle_size: Optional[int] = LATENT_BUNDLE_SIZE,
    receipt_sha256: Optional[str] = LATENT_RECEIPT_SHA256,
    receipt_size: Optional[int] = LATENT_RECEIPT_SIZE,
    receipt_digest: Optional[str] = LATENT_RECEIPT_DIGEST,
    run_sha256: Optional[str] = MATERIALIZER_RUN_COMPLETE_SHA256,
    run_size: Optional[int] = MATERIALIZER_RUN_COMPLETE_SIZE,
    run_digest: Optional[str] = MATERIALIZER_RUN_COMPLETE_DIGEST,
) -> tuple[bytes, Mapping[str, Any]]:
    _require_pins(
        runtime_pins=runtime_pins,
        bundle_sha256=bundle_sha256,
        bundle_size=bundle_size,
        receipt_sha256=receipt_sha256,
        receipt_size=receipt_size,
        receipt_digest=receipt_digest,
        run_sha256=run_sha256,
        run_size=run_size,
        run_digest=run_digest,
    )
    require(set(runtime_pins) == set(RUNTIME_PINS), "runtime source closure differs")
    root = repo_root.resolve(strict=True)
    require(root.is_dir() and not repo_root.is_symlink(), "repo root is not canonical")
    payloads: dict[str, bytes] = {}
    for relative in sorted(runtime_pins, key=lambda item: item.encode("ascii")):
        sha, size = runtime_pins[relative]
        path = (
            train_lora_source_path.resolve(strict=True)
            if relative == "methods/bernini_action_editing/train_lora.py"
            else root.joinpath(*PurePosixPath(relative).parts)
        )
        require(path.is_absolute(), f"runtime source path is not absolute: {relative}")
        raw = stable_file(path, maximum_bytes=8 << 20)
        require(len(raw) == size and hashlib.sha256(raw).hexdigest() == sha, f"runtime pin differs: {relative}")
        compile(raw, relative, "exec")
        payloads[relative] = raw
    evidence_rows = (
        (DERIVATIVE_RELATIVE, root / DERIVATIVE_RELATIVE, DERIVATIVE_SHA256, DERIVATIVE_SIZE, "authority_digest", DERIVATIVE_DIGEST),
        (MODEL_RELATIVE, root / MODEL_RELATIVE, MODEL_SHA256, MODEL_SIZE, "authority_digest", MODEL_DIGEST),
        (CONTRACT_RELATIVE, root / CONTRACT_RELATIVE, CONTRACT_SHA256, CONTRACT_SIZE, "contract_digest", CONTRACT_DIGEST),
        (LATENT_RECEIPT_MEMBER, latent_receipt_path, str(receipt_sha256), int(receipt_size), "receipt_digest", str(receipt_digest)),
        (MATERIALIZER_RUN_MEMBER, materializer_run_complete_path, str(run_sha256), int(run_size), "run_digest", str(run_digest)),
    )
    evidence: dict[str, Mapping[str, Any]] = {}
    for member, path, sha, size, digest_key, digest in evidence_rows:
        path = path.resolve(strict=True)
        require(path.is_absolute(), f"evidence path is not absolute: {member}")
        raw = stable_file(path, maximum_bytes=2 << 20)
        require(len(raw) == size and hashlib.sha256(raw).hexdigest() == sha, f"evidence pin differs: {member}")
        value = strict_json(raw, label=member)
        _validate_self_digest(value, key=digest_key, expected=digest, label=member)
        evidence[member] = value
        payloads[member] = raw
    checkpoint_manifest_path = root / CHECKPOINT_EXACT23_RELATIVE
    checkpoint_manifest_raw = stable_file(
        checkpoint_manifest_path, maximum_bytes=1 << 20
    )
    require(
        len(checkpoint_manifest_raw) == CHECKPOINT_EXACT23_SIZE
        and hashlib.sha256(checkpoint_manifest_raw).hexdigest()
        == CHECKPOINT_EXACT23_SHA256,
        "checkpoint exact23 manifest binding differs",
    )
    payloads[CHECKPOINT_EXACT23_RELATIVE] = checkpoint_manifest_raw
    latent = evidence[LATENT_RECEIPT_MEMBER]
    run = evidence[MATERIALIZER_RUN_MEMBER]
    require(
        payloads[LATENT_RECEIPT_MEMBER]
        == canonical_json_bytes(latent) + b"\n"
        and payloads[MATERIALIZER_RUN_MEMBER]
        == canonical_json_bytes(run) + b"\n",
        "materializer receipt/RUN_COMPLETE is not canonical JSON+newline",
    )
    require(
        latent.get("schema_version")
        == "bernini-elal3-simulator-c2-exact16-latent-bundle-receipt-v1"
        and latent.get("status") == "ELAL3_SIMULATOR_C2_EXACT16_VAE_GO"
        and latent.get("bundle", {}).get("sha256") == bundle_sha256
        and latent.get("bundle", {}).get("size") == bundle_size
        and latent.get("bundle", {}).get("mode") == 0o444
        and latent.get("bundle", {}).get("nlink") == 1
        and len(latent.get("tensor_rows", ())) == 16,
        "exact16 latent receipt envelope differs",
    )
    materialized = run.get("materialized", {})
    require(
        run.get("schema_version")
        == "bernini-elal3-c2-exact16-materializer-run-complete-v1"
        and run.get("status") == "COMPLETE_SIMULATOR_C2_EXACT16_ONLY"
        and run.get("holder_job_id") == "141620"
        and run.get("node") == "auh7-1b-gpu-226"
        and materialized.get("bundle_sha256") == bundle_sha256
        and materialized.get("bundle_size") == bundle_size
        and materialized.get("receipt_sha256") == receipt_sha256
        and materialized.get("receipt_size") == receipt_size
        and materialized.get("receipt_digest") == receipt_digest
        and materialized.get("tensor_count") == 16,
        "materializer RUN_COMPLETE binding differs",
    )
    names = sorted(payloads, key=lambda item: item.encode("ascii"))
    rows = [
        {
            "path": name,
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "size": len(payloads[name]),
            "mode": "0444",
        }
        for name in names
    ]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in names:
            archive.addfile(_tar_info(name, len(payloads[name])), io.BytesIO(payloads[name]))
    archive_raw = stream.getvalue()
    require(len(archive_raw) % 10240 == 0, "archive record size differs")
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "archive_size": len(archive_raw),
        "archive_member_mode": "0444",
        "fresh_training_runtime_file_mode": "0444",
        "fresh_training_runtime_root_mode": "0555",
        "materializer_runtime_tree_reuse_forbidden": True,
        "files": rows,
        "file_count": len(rows),
        "runtime_pins": {
            name: {"sha256": sha, "size": size}
            for name, (sha, size) in runtime_pins.items()
        },
        "external_latent_bundle": {
            "sha256": bundle_sha256,
            "size": bundle_size,
            "mode": "0444",
            "nlink": 1,
            "node_local_stdin_stage_required": True,
            "login_compute_shared_view_assumed": False,
        },
        "latent_receipt_sha256": receipt_sha256,
        "latent_receipt_size": receipt_size,
        "latent_receipt_digest": receipt_digest,
        "materializer_run_complete_sha256": run_sha256,
        "materializer_run_complete_size": run_size,
        "materializer_run_complete_digest": run_digest,
        "authority_bindings": {
            "derivative_sha256": DERIVATIVE_SHA256,
            "derivative_digest": DERIVATIVE_DIGEST,
            "model_sha256": MODEL_SHA256,
            "model_digest": MODEL_DIGEST,
            "experiment_contract_sha256": CONTRACT_SHA256,
            "experiment_contract_digest": CONTRACT_DIGEST,
            "packet_manifest_sha256": PACKET_MANIFEST_SHA256,
            "checkpoint_exact23_manifest_sha256": CHECKPOINT_EXACT23_SHA256,
            "checkpoint_exact23_manifest_size": CHECKPOINT_EXACT23_SIZE,
            "checkpoint_exact23_file_count": 23,
        },
        "run_assignments": list(RUN_ASSIGNMENTS),
        "distributed_topology": {
            "world_size": 8,
            "data_parallel_size": 2,
            "sequence_parallel_size": 4,
            "one_independent_world8_run_per_arm": True,
            "cross_arm_collective_forbidden": True,
        },
        "stage_sequence": [
            "exact3_preflight_no_update",
            "cross_arm_preflight_gate",
            "exact3_fresh1",
            "fresh1_acceptance_gate",
            "exact3_fresh_exact10",
        ],
        "gate_failure_stops_later_stages": True,
        "exact10_resume_from_fresh1_forbidden": True,
        "formal_c2_authorized": False,
        "exact160_authorized": False,
        "source_instruction_inference_authorized": False,
        "real_video_generalization_authorized": False,
        "scientific_claim_authorized": False,
    }
    return archive_raw, {**unsigned, "manifest_digest": object_digest(unsigned)}


def _write_create_only(path: Path, raw: bytes, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(raw)
        while view:
            count = os.write(descriptor, view)
            require(count > 0, f"write stalled: {path}")
            view = view[count:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(
    repo_root: Path,
    *,
    latent_receipt_path: Path,
    materializer_run_complete_path: Path,
    train_lora_source_path: Path,
    output: Path,
) -> Mapping[str, Any]:
    require(output.is_absolute() and not output.exists() and not output.is_symlink(), "output must be a fresh absolute path")
    archive, manifest = build_payload(
        repo_root,
        latent_receipt_path=latent_receipt_path,
        materializer_run_complete_path=materializer_run_complete_path,
        train_lora_source_path=train_lora_source_path,
    )
    os.mkdir(output, 0o700)
    _write_create_only(output / "source.tar", archive, 0o444)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(output / "source.manifest.json", manifest_raw, 0o444)
    os.chmod(output, 0o555)
    return {
        "output": str(output),
        "archive_sha256": manifest["archive_sha256"],
        "archive_size": manifest["archive_size"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_size": len(manifest_raw),
        "manifest_digest": manifest["manifest_digest"],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", type=Path, required=True)
    value.add_argument("--latent-receipt", type=Path, required=True)
    value.add_argument("--materializer-run-complete", type=Path, required=True)
    value.add_argument("--train-lora-source", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = publish(
            args.repo_root,
            latent_receipt_path=args.latent_receipt,
            materializer_run_complete_path=args.materializer_run_complete,
            train_lora_source_path=args.train_lora_source,
            output=args.output,
        )
    except (ELAL3C2TrainingReleaseError, OSError, SyntaxError) as error:
        print(f"[elal3-c2-training-release] ERROR: {error}", file=os.sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

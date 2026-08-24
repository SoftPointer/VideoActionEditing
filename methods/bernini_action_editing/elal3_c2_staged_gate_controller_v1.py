#!/usr/bin/env python3
"""Create the two fail-closed ELAL-3 C2 staged-training gates.

This controller never launches a model and never authorizes an optimizer
update.  It consumes only sealed trainer receipts, replays the frozen C2
trainer's own validators through held/no-follow paths, and publishes either:

* the exact A/B no-update preflight gate; or
* the exact-three-arm fresh-one-update acceptance gate.

The output is create-only canonical JSON, mode 0444.  The controller is meant
to run on node226 after the node-local transport controller has materialized
the same canonical control paths on every holder node.  Gate rows therefore
contain paths which are valid, byte-identical and mode-identical on nodes 226,
249 and 257.  It does not assume that the login node and compute nodes share a
``/vast`` view.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-elal3-c2-staged-gate-controller-v1"

# Mechanically frozen only after the trainer and exact16 materializer are
# final.  The executable entry point refuses to read a receipt while any one
# of these literals is absent.
TRAINER_SHA256: Optional[str] = (
    "63f35b39e60dbf2c1dd1dcecb29393c04d9f00fd0833054e7d81d40790dfe4ce"
)
TRAINER_SIZE: Optional[int] = 447_559
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

STATIC_SOURCE_PINS: Mapping[str, str] = {
    "c1_trainer": "521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3",
    "elal3_core": "70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862",
    "c2_label": "1f09670a3dd2eae09cd27dbb5fe28c913f618d096a04a64fb0cf1dc9b6e1ec11",
    "c2_materializer": "b9142b3da63499163623248d902c51d01bf5c5295ff171125e3c614dea788c0f",
    "train_lora": "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
    "packed_lora": "61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6",
    "world8_runtime": "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f",
    "sigma_strata": "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
}
STATIC_SOURCE_ROWS: Mapping[str, tuple[str, str, int]] = {
    "c1_trainer": (
        "train_elal3_c1_simulator_overfit_v1.py",
        STATIC_SOURCE_PINS["c1_trainer"],
        90_600,
    ),
    "elal3_core": ("elal3_c0_v1.py", STATIC_SOURCE_PINS["elal3_core"], 31_330),
    "c2_label": (
        "elal3_simulator_c2_label_v1.py",
        STATIC_SOURCE_PINS["c2_label"],
        76_939,
    ),
    "c2_materializer": (
        "materialize_elal3_simulator_c2_vae_v1.py",
        STATIC_SOURCE_PINS["c2_materializer"],
        50_334,
    ),
    "train_lora": ("train_lora.py", STATIC_SOURCE_PINS["train_lora"], 66_931),
    "packed_lora": (
        "packed_preservation_lora_v2.py",
        STATIC_SOURCE_PINS["packed_lora"],
        30_419,
    ),
    "world8_runtime": (
        "source_self_runtime.py",
        STATIC_SOURCE_PINS["world8_runtime"],
        36_607,
    ),
    "sigma_strata": (
        "inference_sigma_strata.py",
        STATIC_SOURCE_PINS["sigma_strata"],
        17_956,
    ),
}

_SHA256_LENGTH = 64


class ELAL3C2GateControllerError(RuntimeError):
    """The staged receipt closure is not the preregistered C2 experiment."""


def fail(message: str) -> NoReturn:
    raise ELAL3C2GateControllerError(message)


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
        raise ELAL3C2GateControllerError(
            "value is not finite canonical ASCII JSON"
        ) from error


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{label} is not lowercase SHA-256")
    return value


def require_release_literals() -> None:
    rows = (
        (TRAINER_SHA256, "trainer SHA-256"),
        (ORIGIN_VERIFIER_SHA256, "origin verifier SHA-256"),
        (LATENT_BUNDLE_SHA256, "exact16 bundle SHA-256"),
        (LATENT_RECEIPT_SHA256, "exact16 receipt SHA-256"),
        (LATENT_RECEIPT_DIGEST, "exact16 receipt digest"),
    )
    for value, label in rows:
        require_sha(value, label=label)
    for value, label in (
        (TRAINER_SIZE, "trainer size"),
        (ORIGIN_VERIFIER_SIZE, "origin verifier size"),
        (LATENT_BUNDLE_SIZE, "exact16 bundle size"),
        (LATENT_RECEIPT_SIZE, "exact16 receipt size"),
    ):
        if type(value) is not int or value <= 0:
            fail(f"{label} is PENDING")


def source_pins() -> Mapping[str, Any]:
    require_release_literals()
    rows = {
        "c2_trainer": (
            "train_elal3_c2_simulator_role_pair_v1.py",
            str(TRAINER_SHA256),
            int(TRAINER_SIZE),
        ),
        **dict(STATIC_SOURCE_ROWS),
    }
    sources = {
        name: {
            "relative_path": relative,
            "sha256": sha,
            "size": size,
            "mode": 0o444,
            "nlink": 1,
            "held_fd_double_hash_verified": True,
            "held_openat_parent_chain_replayed": True,
            "actual_imported_module_file_verified": True,
        }
        for name, (relative, sha, size) in rows.items()
    }
    unsigned = {
        "source_count": 9,
        "sources": sources,
        "all_modes": "0444",
        "all_nlink1_no_follow_held_openat_double_hash": True,
        "actual_imported_module_files_verified": True,
        "callable_ownership_verified": True,
        "runtime_absolute_paths_devices_inodes_excluded": True,
    }
    return {**unsigned, "release_pin_digest": object_digest(unsigned)}


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


def read_sealed_bytes(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: Optional[int],
    label: str,
) -> bytes:
    """Read one sealed file twice through a held-openat parent chain."""

    expected_sha256 = require_sha(expected_sha256, label=f"{label} expected SHA")
    if (
        not path.is_absolute()
        or path == Path(path.anchor)
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        fail(f"{label} path is not canonical absolute/no-follow")
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
    root_named_before = root_path.lstat()
    if not stat.S_ISDIR(root_named_before.st_mode):
        fail(f"{label} filesystem root differs")
    root_descriptor = os.open(root_path, directory_flags)
    held: list[int] = [root_descriptor]
    parents: list[tuple[Path, os.stat_result, int]] = [
        (root_path, root_named_before, root_descriptor)
    ]
    try:
        if _directory_identity(root_named_before) != _directory_identity(
            os.fstat(root_descriptor)
        ):
            fail(f"{label} filesystem root identity differs")
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
            if (
                not stat.S_ISDIR(named.st_mode)
                or _directory_identity(named) != _directory_identity(child)
                or _directory_identity(absolute_parent.lstat())
                != _directory_identity(child)
            ):
                fail(f"{label} held-openat parent chain differs")
            parents.append((absolute_parent, named, child_descriptor))
            parent_descriptor = child_descriptor
        basename = path.parts[-1]
        before_name = os.stat(
            basename, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(before_name.st_mode)
            or stat.S_IMODE(before_name.st_mode) != 0o444
            or before_name.st_nlink != 1
        ):
            fail(f"{label} is not one sealed 0444/nlink1 regular file")
        if expected_size is not None and before_name.st_size != expected_size:
            fail(f"{label} size differs")
        descriptor = os.open(basename, file_flags, dir_fd=parent_descriptor)
        held.append(descriptor)
        before = os.fstat(descriptor)

        def one_pass() -> bytes:
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    return b"".join(chunks)
                chunks.append(block)

        first = one_pass()
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = one_pass()
        after = os.fstat(descriptor)
        after_name = os.stat(
            basename, dir_fd=parent_descriptor, follow_symlinks=False
        )
        absolute_after = path.lstat()
        for absolute, parent_before, parent_fd in parents:
            if (
                _directory_identity(parent_before)
                != _directory_identity(os.fstat(parent_fd))
                or _directory_identity(absolute.lstat())
                != _directory_identity(os.fstat(parent_fd))
            ):
                fail(f"{label} held-openat parent final replay differs")
        if (
            first != second
            or _identity(before_name) != _identity(before)
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(after_name)
            or _identity(after_name) != _identity(absolute_after)
            or hashlib.sha256(first).hexdigest() != expected_sha256
        ):
            fail(f"{label} held-FD identity/hash replay differs")
    finally:
        for held_descriptor in reversed(held):
            os.close(held_descriptor)
    return first


def read_sealed_json(
    path: Path, *, expected_sha256: str, label: str
) -> Mapping[str, Any]:
    raw = read_sealed_bytes(
        path,
        expected_sha256=expected_sha256,
        expected_size=None,
        label=label,
    )

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: fail(
                f"{label} contains non-finite token {token}"
            ),
        )
    except ELAL3C2GateControllerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3C2GateControllerError(f"{label} is not strict JSON") from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value) + b"\n":
        fail(f"{label} is not canonical JSON plus one newline")
    return value


def portable_tool_binding(
    path: Path, *, expected_sha256: str, expected_size: int, label: str
) -> Mapping[str, Any]:
    read_sealed_bytes(
        path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        label=label,
    )
    info = path.lstat()
    return {
        "name": path.name,
        "sha256": expected_sha256,
        "size": expected_size,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
    }


def load_frozen_trainer(method_root: Path) -> ModuleType:
    require_release_literals()
    root = method_root.resolve(strict=True)
    trainer_path = root / "train_elal3_c2_simulator_role_pair_v1.py"
    read_sealed_bytes(
        trainer_path,
        expected_sha256=str(TRAINER_SHA256),
        expected_size=TRAINER_SIZE,
        label="frozen C2 trainer",
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "_elal3_c2_frozen_gate_trainer_v1", trainer_path
    )
    if spec is None or spec.loader is None:
        fail("cannot construct frozen trainer import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    imported = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if imported != trainer_path.resolve(strict=True):
        fail("frozen trainer actual imported module path differs")
    read_sealed_bytes(
        imported,
        expected_sha256=str(TRAINER_SHA256),
        expected_size=TRAINER_SIZE,
        label="frozen C2 trainer post-import",
    )
    return module


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail("gate output must be one fresh absolute path")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        fail("gate output parent is not canonical")
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                fail("gate output write made no progress")
            view = view[count:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(payload).hexdigest()
    read_sealed_bytes(
        path,
        expected_sha256=digest,
        expected_size=len(payload),
        label="published staged gate",
    )
    return digest


def _comparable_preflight(value: Mapping[str, Any]) -> Mapping[str, Any]:
    # This is intentionally byte-for-byte the trainer validator's whitelist.
    result = json.loads(canonical_json_bytes(value).decode("ascii"))
    for key in (
        "arm_id",
        "branch_recipe",
        "holder_job_id",
        "node",
        "second_branch_descriptor",
        "receipt_digest",
    ):
        result.pop(key, None)
    for parent in ("actual_shape_preflight", "step0_gain_safety"):
        nested = result.get(parent)
        if isinstance(nested, dict):
            nested.pop("runtime_telemetry", None)
    closure = result.get("pre_publish_closure_replays")
    if isinstance(closure, dict):
        closure.pop("runtime_telemetry", None)
    return result


def build_cross_arm_gate(
    trainer: ModuleType,
    *,
    a_receipt_path: Path,
    a_receipt_sha256: str,
    b_receipt_path: Path,
    b_receipt_sha256: str,
    output: Path,
) -> Mapping[str, Any]:
    pins = source_pins()
    expected = (
        (trainer.ARM_DUPLICATE, a_receipt_path, a_receipt_sha256),
        (trainer.ARM_ROLE_PAIR, b_receipt_path, b_receipt_sha256),
    )
    receipts: list[Mapping[str, Any]] = []
    rows: list[Mapping[str, Any]] = []
    for arm_id, path, sha in expected:
        require_sha(sha, label=f"{arm_id} preflight SHA")
        receipt = trainer.validate_own_preflight_receipt_v1(
            path,
            expected_sha256=sha,
            arm_id=arm_id,
            expected_runner_sha256=str(TRAINER_SHA256),
            expected_bundle_sha256=str(LATENT_BUNDLE_SHA256),
            expected_source_pins=pins,
        )
        job_id, node, seed = trainer.ARM_PLACEMENT[arm_id]
        rows.append(
            {
                "arm_id": arm_id,
                "holder_job_id": job_id,
                "node": node,
                "seed": seed,
                "path": str(path),
                "sha256": sha,
                "receipt_digest": require_sha(
                    receipt.get("receipt_digest"),
                    label=f"{arm_id} receipt digest",
                ),
            }
        )
        receipts.append(receipt)
    if canonical_json_bytes(_comparable_preflight(receipts[0])) != canonical_json_bytes(
        _comparable_preflight(receipts[1])
    ):
        fail("A/B preflight differs outside the final trainer whitelist")
    initial = {row.get("initial_trainable_sha256") for row in receipts}
    schedule = {row.get("row_input_noise_schedule_digest") for row in receipts}
    common = {row.get("common_comparison_payload_digest") for row in receipts}
    recipe = {row.get("recipe_version_digest") for row in receipts}
    if any(len(values) != 1 for values in (initial, schedule, common, recipe)):
        fail("A/B common step-zero closure is not bit-identical")
    unsigned: dict[str, Any] = {
        "schema_version": trainer.CROSS_ARM_GATE_SCHEMA,
        "status": "CROSS_ARM_PREFLIGHT_GATE_PASS",
        "experiment_contract_sha256": trainer.EXPERIMENT_CONTRACT_SHA256,
        "external_authority_sha256": trainer.EXTERNAL_AUTHORITY_SHA256,
        "model_authority_sha256": trainer.MODEL_AUTHORITY_SHA256,
        "latent_bundle_sha256": str(LATENT_BUNDLE_SHA256),
        "runner_source_sha256": str(TRAINER_SHA256),
        "source_pins": dict(pins),
        "recipe_version_digest": next(iter(recipe)),
        "allowed_preflight_receipt_differences": list(
            trainer.CROSS_ARM_ALLOWED_DIFFERENCES
        ),
        "preflight_receipts": rows,
        "common_initial_trainable_sha256": next(iter(initial)),
        "common_row_input_noise_schedule_digest": next(iter(schedule)),
        "common_comparison_payload_digest": next(iter(common)),
        "updates_executed_before_gate": 0,
    }
    gate = {**unsigned, "gate_digest": trainer.object_sha256(unsigned)}
    file_sha = _write_create_only_json(output, gate)
    validated = trainer.validate_cross_arm_preflight_gate_v1(
        output,
        expected_sha256=file_sha,
        expected_runner_sha256=str(TRAINER_SHA256),
        expected_bundle_sha256=str(LATENT_BUNDLE_SHA256),
        expected_source_pins=pins,
    )
    if validated.get("gate_digest") != gate["gate_digest"]:
        fail("published cross-arm gate did not replay through frozen trainer")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "CROSS_ARM_GATE_PUBLISHED",
        "path": str(output),
        "sha256": file_sha,
        "gate_digest": gate["gate_digest"],
    }


def build_fresh1_gate(
    trainer: ModuleType,
    *,
    cross_gate_path: Path,
    cross_gate_sha256: str,
    attestation_rows: Sequence[tuple[Path, str]],
    origin_verifier_path: Path,
    gate_controller_path: Path,
    gate_controller_sha256: str,
    gate_controller_size: int,
    output: Path,
) -> Mapping[str, Any]:
    pins = source_pins()
    if len(attestation_rows) != 3:
        fail("fresh1 gate requires exact-three origin attestations")
    cross = trainer.validate_cross_arm_preflight_gate_v1(
        cross_gate_path,
        expected_sha256=require_sha(
            cross_gate_sha256, label="cross-arm gate SHA"
        ),
        expected_runner_sha256=str(TRAINER_SHA256),
        expected_bundle_sha256=str(LATENT_BUNDLE_SHA256),
        expected_source_pins=pins,
    )
    origin_binding = portable_tool_binding(
        origin_verifier_path,
        expected_sha256=str(ORIGIN_VERIFIER_SHA256),
        expected_size=int(ORIGIN_VERIFIER_SIZE),
        label="fresh1 origin verifier release",
    )
    controller_binding = portable_tool_binding(
        gate_controller_path,
        expected_sha256=require_sha(
            gate_controller_sha256, label="fresh1 gate controller self SHA"
        ),
        expected_size=gate_controller_size,
        label="fresh1 gate controller release",
    )
    rows: list[Mapping[str, Any]] = []
    for arm_id, (path, sha) in zip(trainer.ARM_IDS, attestation_rows):
        sha = require_sha(sha, label=f"{arm_id} fresh1 origin attestation SHA")
        attestation = trainer.validate_fresh1_origin_attestation_v1(
            path,
            expected_sha256=sha,
            arm_id=arm_id,
            expected_runner_sha256=str(TRAINER_SHA256),
            expected_bundle_sha256=str(LATENT_BUNDLE_SHA256),
            expected_source_pins=pins,
            cross_gate=cross,
            expected_origin_verifier_binding=origin_binding,
            expected_gate_controller_binding=controller_binding,
        )
        rows.append(
            {
                "arm_id": arm_id,
                "attestation_sha256": sha,
                "attestation_digest": require_sha(
                    attestation.get("attestation_digest"),
                    label=f"{arm_id} fresh1 origin attestation digest",
                ),
                "attestation": dict(attestation),
            }
        )
    unsigned: dict[str, Any] = {
        "schema_version": trainer.FRESH1_ACCEPTANCE_GATE_SCHEMA,
        "status": "FRESH1_ACCEPTANCE_GATE_PASS",
        "experiment_contract_sha256": trainer.EXPERIMENT_CONTRACT_SHA256,
        "external_authority_sha256": trainer.EXTERNAL_AUTHORITY_SHA256,
        "model_authority_sha256": trainer.MODEL_AUTHORITY_SHA256,
        "latent_bundle_sha256": str(LATENT_BUNDLE_SHA256),
        "runner_source_sha256": str(TRAINER_SHA256),
        "source_pins": dict(pins),
        "cross_arm_gate_sha256": cross["gate_sha256"],
        "cross_arm_gate_digest": cross["gate_digest"],
        "cross_arm_recipe_version_digest": cross["recipe_version_digest"],
        "origin_verifier_binding": dict(origin_binding),
        "gate_controller_binding": dict(controller_binding),
        "fresh1_origin_attestations": rows,
        "exact_fresh1_attestation_count": 3,
        "all_three_origin_physical_replays_passed": True,
        "exact10_resume_from_fresh1_forbidden": True,
    }
    gate = {**unsigned, "gate_digest": trainer.object_sha256(unsigned)}
    file_sha = _write_create_only_json(output, gate)
    validated = trainer.validate_fresh1_acceptance_gate_v1(
        output,
        expected_sha256=file_sha,
        expected_runner_sha256=str(TRAINER_SHA256),
        expected_bundle_sha256=str(LATENT_BUNDLE_SHA256),
        expected_source_pins=pins,
        cross_gate=cross,
        expected_origin_verifier_binding=origin_binding,
        expected_gate_controller_binding=controller_binding,
    )
    if validated.get("gate_digest") != gate["gate_digest"]:
        fail("published fresh1 gate did not replay through frozen trainer")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FRESH1_GATE_PUBLISHED",
        "path": str(output),
        "sha256": file_sha,
        "gate_digest": gate["gate_digest"],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--method-root", type=Path, required=True)
    value.add_argument("--expected-controller-source-sha256", required=True)
    value.add_argument("--expected-controller-source-size", type=int, required=True)
    subparsers = value.add_subparsers(dest="command", required=True)
    cross = subparsers.add_parser("cross-arm")
    cross.add_argument("--a-receipt", type=Path, required=True)
    cross.add_argument("--a-receipt-sha256", required=True)
    cross.add_argument("--b-receipt", type=Path, required=True)
    cross.add_argument("--b-receipt-sha256", required=True)
    cross.add_argument("--output", type=Path, required=True)
    fresh = subparsers.add_parser("fresh1")
    fresh.add_argument("--cross-gate", type=Path, required=True)
    fresh.add_argument("--cross-gate-sha256", required=True)
    for prefix in ("a", "b", "replica"):
        fresh.add_argument(
            f"--{prefix}-origin-attestation", type=Path, required=True
        )
        fresh.add_argument(f"--{prefix}-origin-attestation-sha256", required=True)
    fresh.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    require_release_literals()
    controller_sha = require_sha(
        args.expected_controller_source_sha256,
        label="gate controller external self SHA",
    )
    if args.expected_controller_source_size <= 0:
        fail("gate controller external self size is invalid")
    read_sealed_bytes(
        Path(__file__).resolve(strict=True),
        expected_sha256=controller_sha,
        expected_size=args.expected_controller_source_size,
        label="gate controller self source",
    )
    trainer = load_frozen_trainer(args.method_root)
    if args.command == "cross-arm":
        result = build_cross_arm_gate(
            trainer,
            a_receipt_path=args.a_receipt,
            a_receipt_sha256=args.a_receipt_sha256,
            b_receipt_path=args.b_receipt,
            b_receipt_sha256=args.b_receipt_sha256,
            output=args.output,
        )
    else:
        result = build_fresh1_gate(
            trainer,
            cross_gate_path=args.cross_gate,
            cross_gate_sha256=args.cross_gate_sha256,
            attestation_rows=(
                (
                    args.a_origin_attestation,
                    args.a_origin_attestation_sha256,
                ),
                (
                    args.b_origin_attestation,
                    args.b_origin_attestation_sha256,
                ),
                (
                    args.replica_origin_attestation,
                    args.replica_origin_attestation_sha256,
                ),
            ),
            origin_verifier_path=(
                args.method_root.resolve(strict=True)
                / "elal3_c2_origin_receipt_verifier_v1.py"
            ),
            gate_controller_path=Path(__file__).resolve(strict=True),
            gate_controller_sha256=controller_sha,
            gate_controller_size=args.expected_controller_source_size,
            output=args.output,
        )
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ELAL3C2GateControllerError, OSError) as error:
        print(f"ELAL3_C2_GATE_CONTROLLER_ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

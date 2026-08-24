#!/usr/bin/env python3
"""Origin-node physical verifier for portable ELAL-3 C2 attestations.

This executable runs only inside the holder allocation which created a
fresh1 or exact10 receipt.  It imports the frozen trainer, held-reads the
sealed receipt and all checkpoint records on that same node, invokes the
trainer's closed physical validator, and emits one canonical portable
attestation on stdout.  The central controller may transport that small
attestation; it never transports or dereferences a foreign checkpoint tree.

The final portable ABI is linked only after the frozen trainer exposes its
attestation builders/validators.  Until then this file fails before reading a
receipt.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-elal3-c2-origin-receipt-verifier-v1"
TRAINER_SHA256: Optional[str] = (
    "63f35b39e60dbf2c1dd1dcecb29393c04d9f00fd0833054e7d81d40790dfe4ce"
)
TRAINER_SIZE: Optional[int] = 447_559
LATENT_BUNDLE_SHA256 = "b31d5e1594a112f965a3cebd527d5189a561e2cc2d83cfe94014872ffb94d1b8"
LATENT_BUNDLE_SIZE = 78_277_976
ARM_PLACEMENT: Mapping[str, tuple[str, str, int]] = {
    "A_duplicate_control": ("141620", "auh7-1b-gpu-226", 20260821),
    "B_paired_role": ("141618", "auh7-1b-gpu-249", 20260821),
    "B_paired_role_replica": ("141619", "auh7-1b-gpu-257", 20260822),
}


class ELAL3C2OriginVerifierError(RuntimeError):
    """The origin receipt cannot be promoted to portable evidence."""


def fail(message: str) -> NoReturn:
    raise ELAL3C2OriginVerifierError(message)


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
        raise ELAL3C2OriginVerifierError(
            "value is not finite canonical ASCII JSON"
        ) from error


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{label} is not lowercase SHA-256")
    return value


def require_release_literals() -> None:
    require_sha(TRAINER_SHA256, label="origin verifier trainer SHA")
    if type(TRAINER_SIZE) is not int or TRAINER_SIZE <= 0:
        fail("origin verifier trainer size is PENDING")


def require_origin_placement(arm_id: str) -> Mapping[str, Any]:
    expected = ARM_PLACEMENT.get(arm_id)
    if expected is None:
        fail("origin verifier arm differs")
    expected_job, expected_node, seed = expected
    actual_job = os.environ.get("SLURM_JOB_ID")
    actual_node = os.environ.get("HOSTNAME", "").split(".", 1)[0]
    if (actual_job, actual_node) != (expected_job, expected_node):
        fail("origin verifier holder job/node differs")
    return {
        "arm_id": arm_id,
        "holder_job_id": expected_job,
        "node": expected_node,
        "seed": seed,
    }


def held_sealed_bytes(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: Optional[int],
    expected_mode: int = 0o444,
    label: str,
) -> bytes:
    """Held-openat parent-chain/no-follow replay of one sealed nlink1 file."""

    expected_sha256 = require_sha(expected_sha256, label=f"{label} expected SHA")
    if (
        not path.is_absolute()
        or path == Path(path.anchor)
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        fail(f"{label} path differs")
    identity = lambda value: (
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
    directory_identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_rdev,
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
    root_named_before = root_path.lstat()
    if not stat.S_ISDIR(root_named_before.st_mode):
        fail(f"{label} filesystem root differs")
    root_descriptor = os.open(root_path, directory_flags)
    held: list[int] = [root_descriptor]
    parents: list[tuple[Path, os.stat_result, int]] = [
        (root_path, root_named_before, root_descriptor)
    ]
    try:
        if directory_identity(root_named_before) != directory_identity(
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
                or directory_identity(named) != directory_identity(child)
                or directory_identity(absolute_parent.lstat())
                != directory_identity(child)
            ):
                fail(f"{label} held-openat parent chain differs")
            parents.append((absolute_parent, named, child_descriptor))
            parent_descriptor = child_descriptor
        basename = path.parts[-1]
        named_before = os.stat(
            basename, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(named_before.st_mode)
            or stat.S_IMODE(named_before.st_mode) != expected_mode
            or named_before.st_nlink != 1
            or (expected_size is not None and named_before.st_size != expected_size)
        ):
            fail(f"{label} type/mode/link/size differs")
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
        named_after = os.stat(
            basename, dir_fd=parent_descriptor, follow_symlinks=False
        )
        absolute_after = path.lstat()
        for absolute, parent_before, parent_fd in parents:
            if (
                directory_identity(parent_before)
                != directory_identity(os.fstat(parent_fd))
                or directory_identity(absolute.lstat())
                != directory_identity(os.fstat(parent_fd))
            ):
                fail(f"{label} held-openat parent final replay differs")
        if (
            first != second
            or identity(named_before) != identity(before)
            or identity(before) != identity(after)
            or identity(after) != identity(named_after)
            or identity(named_after) != identity(absolute_after)
            or hashlib.sha256(first).hexdigest() != expected_sha256
        ):
            fail(f"{label} held-FD replay differs")
    finally:
        for held_descriptor in reversed(held):
            os.close(held_descriptor)
    return first


def held_sealed_json(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_self_digest: str,
    self_digest_key: str,
    label: str,
) -> Mapping[str, Any]:
    raw = held_sealed_bytes(
        path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
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
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: fail(
                f"{label} contains non-finite token {token}"
            ),
        )
    except ELAL3C2OriginVerifierError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3C2OriginVerifierError(f"{label} is not strict JSON") from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value) + b"\n":
        fail(f"{label} is not canonical ASCII JSON plus one newline")
    expected_self_digest = require_sha(
        expected_self_digest, label=f"{label} expected self digest"
    )
    unsigned = dict(value)
    stored = unsigned.pop(self_digest_key, None)
    if stored != expected_self_digest or stored != object_digest(unsigned):
        fail(f"{label} self digest differs")
    return value


def load_trainer(method_root: Path) -> ModuleType:
    require_release_literals()
    root = method_root.resolve(strict=True)
    path = root / "train_elal3_c2_simulator_role_pair_v1.py"
    held_sealed_bytes(
        path,
        expected_sha256=str(TRAINER_SHA256),
        expected_size=TRAINER_SIZE,
        label="origin verifier trainer source",
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("_elal3_c2_origin_frozen_trainer_v1", path)
    if spec is None or spec.loader is None:
        fail("trainer import spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    if Path(str(getattr(module, "__file__", ""))).resolve(strict=True) != path:
        fail("trainer actual imported module path differs")
    held_sealed_bytes(
        path,
        expected_sha256=str(TRAINER_SHA256),
        expected_size=TRAINER_SIZE,
        label="origin verifier trainer source post-import",
    )
    return module


def load_source_pin_builder(
    method_root: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> ModuleType:
    root = method_root.resolve(strict=True)
    path = root / "elal3_c2_staged_gate_controller_v1.py"
    held_sealed_bytes(
        path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        label="origin verifier source-pin builder",
    )
    spec = importlib.util.spec_from_file_location(
        "_elal3_c2_origin_source_pin_builder_v1", path
    )
    if spec is None or spec.loader is None:
        fail("source-pin builder import spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    if Path(str(getattr(module, "__file__", ""))).resolve(strict=True) != path:
        fail("source-pin builder actual imported path differs")
    held_sealed_bytes(
        path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        label="origin verifier source-pin builder post-import",
    )
    pins = module.source_pins()
    if not isinstance(pins, Mapping) or pins.get("source_count") != 9:
        fail("source-pin builder returned a different release envelope")
    return module


def portable_tool_binding(
    path: Path, *, expected_sha256: str, expected_size: int, label: str
) -> Mapping[str, Any]:
    held_sealed_bytes(
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


def _require_predecessor_binding(
    receipt: Mapping[str, Any],
    *,
    key: str,
    path: Path,
    sha256: str,
    digest_key: str,
    digest: str,
    label: str,
) -> None:
    binding = receipt.get(key)
    if (
        not isinstance(binding, Mapping)
        or binding.get("path") != str(path)
        or binding.get("sha256") != sha256
        or binding.get(digest_key) != digest
    ):
        fail(f"{label} predecessor binding differs")


def build_origin_attestation(
    args: argparse.Namespace,
    *,
    verifier_sha: str,
) -> Mapping[str, Any]:
    method_root = args.method_root.resolve(strict=True)
    gate_controller_path = method_root / "elal3_c2_staged_gate_controller_v1.py"
    gate_sha = require_sha(
        args.expected_gate_controller_source_sha256,
        label="origin verifier gate-controller SHA",
    )
    if args.expected_gate_controller_source_size <= 0:
        fail("origin verifier gate-controller size differs")
    gate_module = load_source_pin_builder(
        method_root,
        expected_sha256=gate_sha,
        expected_size=args.expected_gate_controller_source_size,
    )
    trainer = load_trainer(method_root)
    pins = gate_module.source_pins()
    receipt = held_sealed_json(
        args.receipt,
        expected_sha256=args.expected_receipt_sha256,
        expected_size=args.expected_receipt_size,
        expected_self_digest=args.expected_receipt_digest,
        self_digest_key="receipt_digest",
        label=f"{args.arm_id} {args.stage} origin receipt",
    )
    if (
        receipt.get("arm_id") != args.arm_id
        or receipt.get("runner_source_sha256") != TRAINER_SHA256
        or receipt.get("latent_bundle_sha256") != LATENT_BUNDLE_SHA256
    ):
        fail("origin receipt arm/release binding differs")
    own_sha = require_sha(
        args.expected_own_preflight_receipt_sha256,
        label="origin own-preflight SHA",
    )
    own = trainer.validate_own_preflight_receipt_v1(
        args.own_preflight_receipt,
        expected_sha256=own_sha,
        arm_id=args.arm_id,
        expected_runner_sha256=str(TRAINER_SHA256),
        expected_bundle_sha256=LATENT_BUNDLE_SHA256,
        expected_source_pins=pins,
    )
    _require_predecessor_binding(
        receipt,
        key="own_preflight_binding",
        path=args.own_preflight_receipt,
        sha256=own_sha,
        digest_key="receipt_digest",
        digest=str(own.get("receipt_digest")),
        label="origin own-preflight",
    )
    cross_sha = require_sha(
        args.expected_cross_arm_gate_sha256, label="origin cross-arm gate SHA"
    )
    cross = trainer.validate_cross_arm_preflight_gate_v1(
        args.cross_arm_gate,
        expected_sha256=cross_sha,
        expected_runner_sha256=str(TRAINER_SHA256),
        expected_bundle_sha256=LATENT_BUNDLE_SHA256,
        expected_source_pins=pins,
    )
    _require_predecessor_binding(
        receipt,
        key="cross_arm_gate_binding",
        path=args.cross_arm_gate,
        sha256=cross_sha,
        digest_key="gate_digest",
        digest=str(cross.get("gate_digest")),
        label="origin cross-arm gate",
    )
    verifier_path = Path(__file__).resolve(strict=True)
    origin_binding = portable_tool_binding(
        verifier_path,
        expected_sha256=verifier_sha,
        expected_size=args.expected_verifier_source_size,
        label="origin verifier release self",
    )
    gate_binding = portable_tool_binding(
        gate_controller_path,
        expected_sha256=gate_sha,
        expected_size=args.expected_gate_controller_source_size,
        label="origin verifier gate-controller release",
    )
    if args.stage == "fresh1":
        if (
            args.fresh1_acceptance_gate is not None
            or args.expected_fresh1_acceptance_gate_sha256 is not None
        ):
            fail("fresh1 origin verifier received a future-stage gate")
        attestation = trainer.build_fresh1_origin_attestation_v1(
            args.receipt,
            expected_receipt_sha256=args.expected_receipt_sha256,
            arm_id=args.arm_id,
            expected_runner_sha256=str(TRAINER_SHA256),
            expected_bundle_sha256=LATENT_BUNDLE_SHA256,
            expected_source_pins=pins,
            cross_gate=cross,
            origin_verifier_path=verifier_path,
            expected_origin_verifier_sha256=verifier_sha,
            gate_controller_path=gate_controller_path,
            expected_gate_controller_sha256=gate_sha,
        )
        trainer._validate_fresh1_origin_attestation_value_v1(
            attestation,
            arm_id=args.arm_id,
            expected_runner_sha256=str(TRAINER_SHA256),
            expected_bundle_sha256=LATENT_BUNDLE_SHA256,
            expected_source_pins=pins,
            cross_gate=cross,
            expected_origin_verifier_binding=origin_binding,
            expected_gate_controller_binding=gate_binding,
        )
    else:
        if (
            args.fresh1_acceptance_gate is None
            or args.expected_fresh1_acceptance_gate_sha256 is None
        ):
            fail("exact10 origin verifier lacks the fresh1 acceptance gate")
        fresh_sha = require_sha(
            args.expected_fresh1_acceptance_gate_sha256,
            label="origin fresh1 acceptance gate SHA",
        )
        fresh = trainer.validate_fresh1_acceptance_gate_v1(
            args.fresh1_acceptance_gate,
            expected_sha256=fresh_sha,
            expected_runner_sha256=str(TRAINER_SHA256),
            expected_bundle_sha256=LATENT_BUNDLE_SHA256,
            expected_source_pins=pins,
            cross_gate=cross,
            expected_origin_verifier_binding=origin_binding,
            expected_gate_controller_binding=gate_binding,
        )
        _require_predecessor_binding(
            receipt,
            key="fresh1_acceptance_gate_binding",
            path=args.fresh1_acceptance_gate,
            sha256=fresh_sha,
            digest_key="gate_digest",
            digest=str(fresh.get("gate_digest")),
            label="origin fresh1 acceptance gate",
        )
        expected_cross_binding = {
            "gate_sha256": cross["gate_sha256"],
            "gate_digest": cross["gate_digest"],
            "recipe_version_digest": cross["recipe_version_digest"],
        }
        expected_fresh_binding = {
            key: fresh[key]
            for key in (
                "gate_sha256",
                "gate_digest",
                "cross_arm_gate_sha256",
                "cross_arm_gate_digest",
                "cross_arm_recipe_version_digest",
            )
        }
        attestation = trainer.build_exact10_origin_attestation_v1(
            args.receipt,
            expected_receipt_sha256=args.expected_receipt_sha256,
            arm_id=args.arm_id,
            expected_runner_sha256=str(TRAINER_SHA256),
            expected_bundle_sha256=LATENT_BUNDLE_SHA256,
            expected_source_pins=pins,
            origin_verifier_path=verifier_path,
            expected_origin_verifier_sha256=verifier_sha,
            gate_controller_path=gate_controller_path,
            expected_gate_controller_sha256=gate_sha,
            expected_cross_gate_binding=expected_cross_binding,
            expected_fresh1_gate_binding=expected_fresh_binding,
        )
        trainer._validate_exact10_origin_attestation_value_v1(
            attestation,
            arm_id=args.arm_id,
            expected_runner_sha256=str(TRAINER_SHA256),
            expected_bundle_sha256=LATENT_BUNDLE_SHA256,
            expected_source_pins=pins,
            expected_origin_verifier_binding=origin_binding,
            expected_gate_controller_binding=gate_binding,
            expected_cross_gate_binding=expected_cross_binding,
            expected_fresh1_gate_binding=expected_fresh_binding,
        )
        if (
            attestation.get("latent_hard_gates_pass") is not True
            or attestation.get("receipt_status")
            != "EXACT10_LATENT_GATES_PASS_DECODED_REVIEW_PENDING"
        ):
            fail("exact10 origin attestation is a latent NO-GO")
    if (
        attestation.get("receipt_sha256") != args.expected_receipt_sha256
        or attestation.get("receipt_size") != args.expected_receipt_size
        or attestation.get("receipt_digest") != args.expected_receipt_digest
        or attestation.get("origin_verifier_binding") != origin_binding
        or attestation.get("gate_controller_binding") != gate_binding
        or attestation.get("physical_origin_replay_passed") is not True
        or attestation.get("closed_validator_passed") is not True
    ):
        fail("origin portable attestation final join differs")
    return attestation


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--method-root", type=Path, required=True)
    value.add_argument("--expected-verifier-source-sha256", required=True)
    value.add_argument("--expected-verifier-source-size", type=int, required=True)
    value.add_argument("--expected-gate-controller-source-sha256", required=True)
    value.add_argument("--expected-gate-controller-source-size", type=int, required=True)
    value.add_argument("--arm-id", required=True)
    value.add_argument("--receipt", type=Path, required=True)
    value.add_argument("--expected-receipt-sha256", required=True)
    value.add_argument("--expected-receipt-size", type=int, required=True)
    value.add_argument("--expected-receipt-digest", required=True)
    value.add_argument("--own-preflight-receipt", type=Path, required=True)
    value.add_argument("--expected-own-preflight-receipt-sha256", required=True)
    value.add_argument("--cross-arm-gate", type=Path, required=True)
    value.add_argument("--expected-cross-arm-gate-sha256", required=True)
    value.add_argument("--fresh1-acceptance-gate", type=Path)
    value.add_argument("--expected-fresh1-acceptance-gate-sha256")
    value.add_argument("--stage", choices=("fresh1", "exact10"), required=True)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    require_release_literals()
    verifier_sha = require_sha(
        args.expected_verifier_source_sha256,
        label="origin verifier external self SHA",
    )
    if args.expected_verifier_source_size <= 0:
        fail("origin verifier external self size differs")
    held_sealed_bytes(
        Path(__file__).resolve(strict=True),
        expected_sha256=verifier_sha,
        expected_size=args.expected_verifier_source_size,
        label="origin verifier self source",
    )
    require_origin_placement(args.arm_id)
    require_sha(args.expected_receipt_sha256, label="origin receipt SHA")
    require_sha(args.expected_receipt_digest, label="origin receipt digest")
    if args.expected_receipt_size <= 0:
        fail("origin receipt size differs")
    # Reserve stdout for the one portable canonical envelope.  A dependency
    # that writes diagnostics while importing/replaying is redirected to
    # stderr so the login controller cannot accidentally accept mixed bytes.
    with redirect_stdout(sys.stderr):
        attestation = build_origin_attestation(args, verifier_sha=verifier_sha)
    raw = canonical_json_bytes(attestation) + b"\n"
    unsigned = dict(attestation)
    stored = unsigned.pop("attestation_digest", None)
    if stored != object_digest(unsigned):
        fail("origin portable attestation self digest differs before emission")
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ELAL3C2OriginVerifierError, OSError) as error:
        print(f"ELAL3_C2_ORIGIN_VERIFIER_ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

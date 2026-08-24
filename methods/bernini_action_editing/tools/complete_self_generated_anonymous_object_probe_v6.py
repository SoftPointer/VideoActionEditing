#!/usr/bin/env python3
"""External completion authority for the immutable V6 anonymous-object probe.

This controller is deliberately outside the probe runtime/source closure.  A
probe JSON is only a candidate.  The controller may create a completion seal
after its caller has observed the whole srun/torchrun/all-wrapper command exit
zero, and only after byte-verifying the immutable snapshot and every receipt
digest/claim boundary below.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-anonymous-object-v6-external-completion-seal-v1"
METHOD = "bernini-self-generated-anonymous-object-probe-v6"
SNAPSHOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/self_generated_anonymous_object_probe_v6_20260823/"
    "snapshot_83aa45bfe4bd_r2"
)
SNAPSHOT_SHA256SUMS_SHA256 = (
    "349b793e436ad08219775348770b363ed2a74e96606d506a70dfd3c75d0234e7"
)
CONTRACT_DIGEST = (
    "83aa45bfe4bd782a4ae55206758dae2aa151317cef45b198b8264858b9aa73aa"
)
RUNTIME_MANIFEST_DIGEST = (
    "0226358316c061bd813d4bfd8aaad98ad09f5c29d31bc223ff7085a5b1f2a0a9"
)
TEST_MANIFEST_DIGEST = (
    "5695fb948aed99367f52c174f86c4615fdbcf5484012c5998c622763070a99b8"
)
LAUNCH_TEMPLATE_DIGEST = (
    "eba7fe5f0ad7605c95d4877fcfbc186401142e4f15eb199e8b33b081a6c2a74a"
)
EXPECTED_SNAPSHOT_FILE_COUNT_EXCLUDING_SUMS = 32
EXPECTED_RUNTIME_SOURCE_COUNT = 26
EXPECTED_TEST_SOURCE_COUNT = 4
EXPECTED_PRELAUNCH_CPU_TEST_COUNT = 48
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class CompletionControllerV6Error(RuntimeError):
    """The candidate or its external completion authority is not valid."""


@dataclass(frozen=True)
class CompletionAuthorityV6:
    snapshot: Path
    snapshot_sums_sha256: str
    contract_digest: str
    runtime_manifest_digest: str
    test_manifest_digest: str
    launch_template_digest: str
    snapshot_file_count_excluding_sums: int = (
        EXPECTED_SNAPSHOT_FILE_COUNT_EXCLUDING_SUMS
    )

    def validate(self) -> None:
        if any(
            _SHA256.fullmatch(value) is None
            for value in (
                self.snapshot_sums_sha256,
                self.contract_digest,
                self.runtime_manifest_digest,
                self.test_manifest_digest,
                self.launch_template_digest,
            )
        ) or self.snapshot_file_count_excluding_sums <= 0:
            raise CompletionControllerV6Error("completion authority differs")


FROZEN_AUTHORITY = CompletionAuthorityV6(
    SNAPSHOT,
    SNAPSHOT_SHA256SUMS_SHA256,
    CONTRACT_DIGEST,
    RUNTIME_MANIFEST_DIGEST,
    TEST_MANIFEST_DIGEST,
    LAUNCH_TEMPLATE_DIGEST,
)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise CompletionControllerV6Error("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise CompletionControllerV6Error(f"cannot hash {path}") from error
    return digest.hexdigest()


def _plain_canonical_file(path: Path, *, label: str) -> Path:
    original = Path(path).absolute()
    if not original.is_file() or original.is_symlink():
        raise CompletionControllerV6Error(f"{label} is not a plain file")
    try:
        canonical = original.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CompletionControllerV6Error(f"{label} cannot resolve") from error
    if original != canonical:
        raise CompletionControllerV6Error(f"{label} path is not canonical")
    return canonical


def _plain_canonical_directory(path: Path, *, label: str) -> Path:
    original = Path(path).absolute()
    if not original.is_dir() or original.is_symlink():
        raise CompletionControllerV6Error(f"{label} is not a plain directory")
    try:
        canonical = original.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CompletionControllerV6Error(f"{label} cannot resolve") from error
    if original != canonical:
        raise CompletionControllerV6Error(f"{label} path is not canonical")
    return canonical


def _reject_group_world_writable(path: Path, *, label: str) -> None:
    try:
        mode = stat.S_IMODE(path.lstat().st_mode)
    except OSError as error:
        raise CompletionControllerV6Error(f"{label} mode cannot be read") from error
    if mode & 0o022:
        raise CompletionControllerV6Error(f"{label} is group/world writable")


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompletionControllerV6Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_one_json(path: Path, *, label: str) -> Mapping[str, Any]:
    source = _plain_canonical_file(path, label=label)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CompletionControllerV6Error(f"non-finite JSON token: {token}")
            ),
        )
    except CompletionControllerV6Error:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionControllerV6Error(f"{label} is not one JSON object") from error
    if not isinstance(value, Mapping):
        raise CompletionControllerV6Error(f"{label} is not a JSON object")
    return value


def verify_embedded_digest(value: Mapping[str, Any], *, label: str) -> str:
    body = dict(value)
    claimed = body.pop("digest", None)
    computed = object_sha256(body)
    if claimed != computed:
        raise CompletionControllerV6Error(f"{label} self digest differs")
    return computed


def verify_snapshot(
    authority: CompletionAuthorityV6 = FROZEN_AUTHORITY,
) -> Mapping[str, Any]:
    authority.validate()
    snapshot = _plain_canonical_directory(authority.snapshot, label="snapshot")
    _reject_group_world_writable(snapshot, label="snapshot")
    sums_path = _plain_canonical_file(snapshot / "SHA256SUMS", label="SHA256SUMS")
    _reject_group_world_writable(sums_path, label="SHA256SUMS")
    if file_sha256(sums_path) != authority.snapshot_sums_sha256:
        raise CompletionControllerV6Error("SHA256SUMS authority differs")
    rows: dict[str, str] = {}
    try:
        lines = sums_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise CompletionControllerV6Error("SHA256SUMS cannot be read") from error
    for line in lines:
        if "  " not in line:
            raise CompletionControllerV6Error("SHA256SUMS row differs")
        digest, relative = line.split("  ", 1)
        if _SHA256.fullmatch(digest) is None or not relative.startswith("./"):
            raise CompletionControllerV6Error("SHA256SUMS row authority differs")
        relative_path = Path(relative[2:])
        if (
            not relative_path.parts
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or str(relative_path) in rows
            or relative_path.name == "SHA256SUMS"
        ):
            raise CompletionControllerV6Error("SHA256SUMS path differs")
        rows[str(relative_path)] = digest
    if len(rows) != authority.snapshot_file_count_excluding_sums:
        raise CompletionControllerV6Error("snapshot file count differs")
    actual: set[str] = set()
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            raise CompletionControllerV6Error("snapshot contains a symlink")
        _reject_group_world_writable(path, label="snapshot entry")
        if path.is_file() and path != sums_path:
            relative = str(path.relative_to(snapshot))
            actual.add(relative)
            if relative not in rows or file_sha256(path) != rows[relative]:
                raise CompletionControllerV6Error(
                    f"snapshot byte authority differs: {relative}"
                )
    if actual != set(rows):
        raise CompletionControllerV6Error("snapshot closure has missing/extra files")

    contract_packet = load_one_json(
        snapshot / "CONTRACT_AND_LAUNCH_TEMPLATE.json",
        label="snapshot contract packet",
    )
    contract = contract_packet.get("contract")
    template = contract_packet.get("remote_launch_template")
    if not isinstance(contract, Mapping) or not isinstance(template, Mapping):
        raise CompletionControllerV6Error("snapshot contract packet differs")
    if (
        verify_embedded_digest(contract, label="snapshot contract")
        != authority.contract_digest
        or verify_embedded_digest(template, label="snapshot launch template")
        != authority.launch_template_digest
    ):
        raise CompletionControllerV6Error("snapshot reviewed contract differs")
    runtime = contract.get("source_manifest")
    tests = contract.get("test_source_manifest")
    if not isinstance(runtime, Mapping) or not isinstance(tests, Mapping):
        raise CompletionControllerV6Error("snapshot closure manifests are absent")
    if (
        verify_embedded_digest(runtime, label="runtime manifest")
        != authority.runtime_manifest_digest
        or verify_embedded_digest(tests, label="test manifest")
        != authority.test_manifest_digest
        or runtime.get("file_count") != EXPECTED_RUNTIME_SOURCE_COUNT
        or tests.get("file_count") != EXPECTED_TEST_SOURCE_COUNT
        or tests.get("expected_unittest_case_count")
        != EXPECTED_PRELAUNCH_CPU_TEST_COUNT
    ):
        raise CompletionControllerV6Error("snapshot source closure differs")
    return {
        "snapshot": str(snapshot),
        "snapshot_sha256s_sha256": authority.snapshot_sums_sha256,
        "contract": dict(contract),
        "launch_template": dict(template),
        "runtime_manifest": dict(runtime),
        "test_manifest": dict(tests),
    }


def verify_candidate(
    candidate_path: Path,
    snapshot_receipt: Mapping[str, Any],
    authority: CompletionAuthorityV6 = FROZEN_AUTHORITY,
) -> Mapping[str, Any]:
    candidate_path = _plain_canonical_file(candidate_path, label="candidate receipt")
    candidate = load_one_json(candidate_path, label="candidate receipt")
    candidate_digest = verify_embedded_digest(candidate, label="candidate receipt")
    contract = candidate.get("contract")
    runtime = candidate.get("source_manifest")
    tests = candidate.get("test_source_manifest")
    result = candidate.get("anonymous_object_result")
    if not all(isinstance(value, Mapping) for value in (contract, runtime, tests, result)):
        raise CompletionControllerV6Error("candidate authority sections are absent")
    verify_embedded_digest(contract, label="candidate contract")
    verify_embedded_digest(runtime, label="candidate runtime manifest")
    verify_embedded_digest(tests, label="candidate test manifest")
    verify_embedded_digest(result, label="candidate anonymous result")
    if (
        contract != snapshot_receipt["contract"]
        or runtime != snapshot_receipt["runtime_manifest"]
        or tests != snapshot_receipt["test_manifest"]
        or contract.get("digest") != authority.contract_digest
        or runtime.get("digest") != authority.runtime_manifest_digest
        or tests.get("digest") != authority.test_manifest_digest
    ):
        raise CompletionControllerV6Error("candidate reviewed closure differs")
    if (
        candidate.get("schema_version")
        != "bernini-auh-self-generated-anonymous-object-probe-v6"
        or candidate.get("method") != METHOD
        or contract.get("gpu_launch_authorized") is not True
        or contract.get("launch_blocked_pending_independent_audit") is not False
        or contract.get("representation_admission_hard_false") is not True
        or contract.get("scientific_claim_authorized") is not False
        or candidate.get("representation_admitted") is not False
        or candidate.get("stable_transferable_action_representation_claimed") is not False
        or candidate.get("scientific_claim_authorized") is not False
        or candidate.get("prompt_shuffle_control_executed") is not False
        or candidate.get("heldout_transfer_control_executed") is not False
        or candidate.get("decoder_called") is not False
        or candidate.get("renderer_called") is not False
        or candidate.get("optimizer_created") is not False
        or candidate.get("route_or_injection_called") is not False
        or candidate.get("parameter_updates") != 0
        or candidate.get("all_controls_executed") is not True
        or candidate.get("all_nine_B0_action_outputs_observer_bit_exact") is not True
        or result.get("representation_admitted") is not False
        or result.get("stable_transferable_action_representation_claimed") is not False
        or result.get("scientific_claim_authorized") is not False
    ):
        raise CompletionControllerV6Error("candidate claim boundary differs")
    expected_lengths = {
        "trajectory_step_registry": 120,
        "frozen_base_cells": 9,
        "anonymous_same_state_authorities": 9,
        "projected_capture_receipts": 72,
        "reduced_cell_receipts": 9,
        "rank_summaries": 4,
    }
    if any(
        not isinstance(candidate.get(name), list)
        or len(candidate[name]) != expected
        for name, expected in expected_lengths.items()
    ) or (
        candidate.get("trajectory_model_forward_count") != 240
        or candidate.get("trajectory_unipc_step_count") != 120
        or candidate.get("frozen_base_probe_forward_count") != 9
        or candidate.get("observer_probe_forward_count") != 72
        or candidate.get("total_frozen_transformer_forward_count") != 321
        or result.get("cell_count") != 9
    ):
        raise CompletionControllerV6Error("candidate execution matrix differs")
    return {
        "candidate_path": str(candidate_path),
        "candidate_file_sha256": file_sha256(candidate_path),
        "candidate_receipt_digest": candidate_digest,
    }


def _create_only_fsync_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).absolute()
    parent = _plain_canonical_directory(path.parent, label="seal parent")
    if path.parent != parent or path.is_symlink():
        raise CompletionControllerV6Error("completion seal path must be canonical absent")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    created = False
    try:
        descriptor = os.open(path, flags, 0o444)
        created = True
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise CompletionControllerV6Error("completion seal write stopped")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def complete_candidate(
    *,
    candidate: Path,
    completion_seal: Path,
    launcher_log: Path,
    srun_exit_code: int,
    caller_attests_all_rank_wrappers_exit_zero: bool,
    slurm_job_id: str,
    slurm_step_id: str,
    srun_command_sha256: str,
    authority: CompletionAuthorityV6 = FROZEN_AUTHORITY,
) -> Mapping[str, Any]:
    if (
        srun_exit_code != 0
        or caller_attests_all_rank_wrappers_exit_zero is not True
    ):
        raise CompletionControllerV6Error(
            "external controller did not attest whole-srun zero exit"
        )
    if (
        _SAFE_ID.fullmatch(slurm_job_id) is None
        or _SAFE_ID.fullmatch(slurm_step_id) is None
        or _SHA256.fullmatch(srun_command_sha256) is None
    ):
        raise CompletionControllerV6Error("Slurm completion identity differs")
    launcher_log = _plain_canonical_file(launcher_log, label="launcher log")
    snapshot_receipt = verify_snapshot(authority)
    candidate_receipt = verify_candidate(candidate, snapshot_receipt, authority)
    controller_path = _plain_canonical_file(Path(__file__), label="completion controller")
    body = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **candidate_receipt,
        "contract_digest": authority.contract_digest,
        "runtime_manifest_digest": authority.runtime_manifest_digest,
        "test_manifest_digest": authority.test_manifest_digest,
        "launch_template_digest": authority.launch_template_digest,
        "snapshot_path": snapshot_receipt["snapshot"],
        "snapshot_sha256s_sha256": authority.snapshot_sums_sha256,
        "snapshot_full_hash_check_passed": True,
        "launcher_log_path": str(launcher_log),
        "launcher_log_sha256": file_sha256(launcher_log),
        "srun_command_sha256": srun_command_sha256,
        "slurm_job_id": slurm_job_id,
        "slurm_step_id": slurm_step_id,
        "srun_exit_code": 0,
        "caller_attests_torchrun_and_all_rank_wrappers_exit_zero": True,
        "candidate_presence_alone_was_not_completion_authority": True,
        "controller_path": str(controller_path),
        "controller_sha256": file_sha256(controller_path),
        "completion_authority": "EXTERNAL_SEAL_CREATED_AFTER_ALL_GATES",
        "representation_admitted": False,
        "scientific_claim_authorized": False,
    }
    seal = {**body, "digest": object_sha256(body)}
    _create_only_fsync_json(completion_seal, seal)
    return seal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--completion-seal", type=Path, required=True)
    parser.add_argument("--launcher-log", type=Path, required=True)
    parser.add_argument("--srun-exit-code", type=int, required=True)
    parser.add_argument(
        "--caller-attests-all-rank-wrappers-exit-zero",
        action="store_true",
        required=True,
    )
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--slurm-step-id", required=True)
    parser.add_argument("--srun-command-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seal = complete_candidate(
        candidate=args.candidate,
        completion_seal=args.completion_seal,
        launcher_log=args.launcher_log,
        srun_exit_code=args.srun_exit_code,
        caller_attests_all_rank_wrappers_exit_zero=(
            args.caller_attests_all_rank_wrappers_exit_zero
        ),
        slurm_job_id=args.slurm_job_id,
        slurm_step_id=args.slurm_step_id,
        srun_command_sha256=args.srun_command_sha256,
    )
    print(json.dumps(seal, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

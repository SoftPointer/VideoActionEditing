#!/usr/bin/env python3
"""Fail-closed two-arm controller and formal pair-admission contract.

The structural command launches clean/noised WORLD8 preflights concurrently,
then admits a shared initialization only if their complete receipts have an
identical pair-invariant projection.  The formal command conjuncts that
preflight receipt with a verified decoded Stage-A admission before launching
either optimizer.  The intermediate backward-feasibility command executes one
real four-microbatch accumulation window per arm, but constructs no optimizer,
changes no parameters, and writes no checkpoint.  Parent Slurm allocations are
never cancelled or released.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, NoReturn, Optional, Sequence


# This controller executes from the immutable method release.  Set this before
# importing any release-local module so Python cannot add __pycache__ entries
# and invalidate the exact executed-tree closure checked by both child arms.
sys.dont_write_bytecode = True

METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_stage_b_contract_v1 as stage_b  # noqa: E402


PREFLIGHT_PAIR_SCHEMA = "bernini-clean-source-visual-context-preflight-pair-v1"
FORMAL_PAIR_ADMISSION_SCHEMA = (
    "bernini-clean-source-visual-context-formal-pair-admission-v1"
)
FORMAL_PAIR_RESULT_SCHEMA = "bernini-clean-source-visual-context-formal-pair-result-v1"
BACKWARD_PAIR_RESULT_SCHEMA = (
    "bernini-clean-source-visual-context-backward-feasibility-pair-v1"
)
PREFLIGHT_RECEIPT_SCHEMA = (
    "bernini-clean-source-visual-context-stage-b-structural-preflight-v1"
)
TRAINING_RECEIPT_SCHEMA = "bernini-clean-source-visual-context-stage-b-training-v1"
BACKWARD_RECEIPT_SCHEMA = (
    "bernini-clean-source-visual-context-stage-b-backward-feasibility-v1"
)
ARMS = (
    {
        "memory_input_kind": "clean_source",
        "holder_job": 135980,
        "holder_node": "auh7-1b-gpu-239",
        "wrapper": "scripts/auh_preflight_clean_source_visual_context_main_holder_v1.sh",
        "formal_wrapper": "scripts/auh_train_clean_source_visual_context_main_holder_v1.sh",
    },
    {
        "memory_input_kind": "same_noise_forward_noised_source",
        "holder_job": 135981,
        "holder_node": "auh7-1b-gpu-234",
        "wrapper": "scripts/auh_preflight_clean_source_visual_context_noised_holder_v1.sh",
        "formal_wrapper": "scripts/auh_train_clean_source_visual_context_noised_holder_v1.sh",
    },
)
ALLOWED_ARM_DIFFERENCES = (
    "memory_input_kind",
    "holder_job",
    "holder_node",
    "run_root",
    "master_port",
)
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CleanSourceVisualPairError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise CleanSourceVisualPairError(message)


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
        raise CleanSourceVisualPairError("pair value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if type(value) is not str or pattern.fullmatch(value) is None:
        fail(f"{label} differs")
    return value


def _plain(path_value: str | Path, *, label: str) -> Path:
    requested = Path(path_value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be absolute/non-symlink")
    path = requested.resolve(strict=True)
    if path != requested or not path.is_file() or path.is_symlink():
        fail(f"{label} must be one canonical file")
    return path


def load_canonical_receipt(
    path_value: str | Path,
    *,
    expected_file_sha256: str,
    expected_schema: str,
) -> Mapping[str, Any]:
    path = _plain(path_value, label=expected_schema)
    expected_file_sha256 = _digest(
        expected_file_sha256, length=64, label="receipt file SHA"
    )
    raw = path.read_bytes()
    if file_sha256(path) != expected_file_sha256:
        fail(f"{expected_schema} file SHA differs")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CleanSourceVisualPairError(f"cannot decode {expected_schema}") from error
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != expected_schema
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        fail(f"{expected_schema} canonical schema differs")
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if declared != object_sha256(unsigned):
        fail(f"{expected_schema} embedded digest differs")
    return dict(value)


def write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail("pair output must be a fresh absolute file")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _validate_shared_cli(args: argparse.Namespace) -> None:
    for name in (
        "source_only_manifest_sha256",
        "method_source_archive_sha256",
        "method_source_manifest_sha256",
    ):
        _digest(getattr(args, name), length=64, label=name)
    _digest(args.method_source_revision, length=40, label="method_source_revision")
    for path_name, sha_name in (
        ("source_only_manifest", "source_only_manifest_sha256"),
        ("method_source_archive", "method_source_archive_sha256"),
        ("method_source_manifest", "method_source_manifest_sha256"),
    ):
        path = _plain(getattr(args, path_name), label=path_name)
        if file_sha256(path) != getattr(args, sha_name):
            fail(f"{path_name} SHA differs")
    root = Path(args.method_root).expanduser()
    if (
        not root.is_absolute()
        or root.resolve(strict=True) != root
        or root.is_symlink()
        or not root.is_dir()
    ):
        fail("method_root differs")
    if args.clean_run_root == args.noised_run_root:
        fail("pair arm run roots must differ")
    if args.clean_master_port == args.noised_master_port:
        fail("pair arm ports must differ")
    for run_root in (args.clean_run_root, args.noised_run_root):
        path = Path(run_root).expanduser()
        if (
            not path.is_absolute()
            or path.exists()
            or path.is_symlink()
            or path.parent.resolve(strict=True) != path.parent
        ):
            fail("pair run root must be fresh/canonical")


def _base_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CSVC_METHOD_ROOT": str(Path(args.method_root)),
            "CSVC_SOURCE_ONLY_MANIFEST": str(Path(args.source_only_manifest)),
            "CSVC_SOURCE_ONLY_MANIFEST_SHA256": args.source_only_manifest_sha256,
            "CSVC_METHOD_REVISION": args.method_source_revision,
            "CSVC_METHOD_ARCHIVE": str(Path(args.method_source_archive)),
            "CSVC_METHOD_ARCHIVE_SHA256": args.method_source_archive_sha256,
            "CSVC_METHOD_MANIFEST": str(Path(args.method_source_manifest)),
            "CSVC_METHOD_MANIFEST_SHA256": args.method_source_manifest_sha256,
        }
    )
    return environment


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise CleanSourceVisualPairError(
            f"cannot inspect pair child process group {process_group_id}"
        ) from error
    return True


def _terminate_process_groups(
    processes: Sequence[subprocess.Popen[Any]], *, timeout_seconds: float = 30.0
) -> None:
    """Stop both child sessions without cancelling either parent allocation."""

    process_group_ids = [process.pid for process in processes]
    for process_group_id in process_group_ids:
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not any(_process_group_exists(group) for group in process_group_ids):
            break
        time.sleep(0.2)
    for process_group_id in process_group_ids:
        if _process_group_exists(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise CleanSourceVisualPairError(
                f"pair wrapper process {process.pid} survived SIGKILL"
            ) from error


def _run_two_wrappers(
    args: argparse.Namespace,
    *,
    formal_pair_admission: Optional[Path] = None,
    formal_pair_admission_sha256: Optional[str] = None,
    expected_initial_parameter_digest: Optional[str] = None,
    backward_pair_receipt: Optional[Path] = None,
    backward_pair_receipt_sha256: Optional[str] = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    base = _base_environment(args)
    processes: list[subprocess.Popen[Any]] = []
    logs: list[Any] = []
    try:
        for index, arm in enumerate(ARMS):
            environment = dict(base)
            run_root = args.clean_run_root if index == 0 else args.noised_run_root
            port = args.clean_master_port if index == 0 else args.noised_master_port
            environment["CSVC_RUN_ROOT"] = str(run_root)
            environment["CSVC_MASTER_PORT"] = str(port)
            if formal_pair_admission is not None:
                environment.pop("CSVC_PREFLIGHT_PAIR_RECEIPT", None)
                environment.pop("CSVC_PREFLIGHT_PAIR_RECEIPT_SHA256", None)
                expected_initial_parameter_digest = _digest(
                    expected_initial_parameter_digest,
                    length=64,
                    label="expected initial parameter digest",
                )
                environment["CSVC_STAGE_A_ADMISSION"] = str(args.stage_a_admission)
                environment["CSVC_STAGE_A_ADMISSION_SHA256"] = (
                    args.stage_a_admission_sha256
                )
                environment["CSVC_FORMAL_PAIR_ADMISSION"] = str(
                    formal_pair_admission
                )
                environment["CSVC_FORMAL_PAIR_ADMISSION_SHA256"] = str(
                    formal_pair_admission_sha256
                )
                environment["CSVC_EXPECTED_INITIAL_PARAMETER_DIGEST"] = (
                    expected_initial_parameter_digest
                )
                wrapper = arm["formal_wrapper"]
            elif backward_pair_receipt is not None:
                environment.pop("CSVC_STAGE_A_ADMISSION", None)
                environment.pop("CSVC_STAGE_A_ADMISSION_SHA256", None)
                environment.pop("CSVC_FORMAL_PAIR_ADMISSION", None)
                environment.pop("CSVC_FORMAL_PAIR_ADMISSION_SHA256", None)
                environment.pop("CSVC_EXPECTED_INITIAL_PARAMETER_DIGEST", None)
                environment["CSVC_PREFLIGHT_PAIR_RECEIPT"] = str(
                    backward_pair_receipt
                )
                environment["CSVC_PREFLIGHT_PAIR_RECEIPT_SHA256"] = _digest(
                    backward_pair_receipt_sha256,
                    length=64,
                    label="backward upstream pair receipt SHA",
                )
                environment["CSVC_HOLDER_JOB"] = str(arm["holder_job"])
                environment["CSVC_HOLDER_NODE"] = str(arm["holder_node"])
                environment["CSVC_MEMORY_INPUT_KIND"] = str(
                    arm["memory_input_kind"]
                )
                environment["CSVC_EXECUTION_SCOPE"] = (
                    "backward-feasibility-preflight"
                )
                wrapper = (
                    "scripts/auh_train_clean_source_visual_context_stage_b_holder_v1.sh"
                )
            else:
                environment.pop("CSVC_STAGE_A_ADMISSION", None)
                environment.pop("CSVC_STAGE_A_ADMISSION_SHA256", None)
                environment.pop("CSVC_FORMAL_PAIR_ADMISSION", None)
                environment.pop("CSVC_FORMAL_PAIR_ADMISSION_SHA256", None)
                environment.pop("CSVC_EXPECTED_INITIAL_PARAMETER_DIGEST", None)
                environment.pop("CSVC_PREFLIGHT_PAIR_RECEIPT", None)
                environment.pop("CSVC_PREFLIGHT_PAIR_RECEIPT_SHA256", None)
                wrapper = arm["wrapper"]
            log = (
                Path(args.pair_output_root) / f"arm_{index}_controller.log"
            ).open("xb", buffering=0)
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    ["bash", str(Path(args.method_root) / str(wrapper))],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
            )
        while True:
            statuses = [process.poll() for process in processes]
            if any(status not in (None, 0) for status in statuses):
                fail(f"pair child failed: {statuses}")
            if all(status == 0 for status in statuses):
                break
            time.sleep(5)
    except BaseException:
        _terminate_process_groups(processes)
        raise
    finally:
        for log in logs:
            log.close()
    if formal_pair_admission is not None:
        expected_schema = TRAINING_RECEIPT_SCHEMA
    elif backward_pair_receipt is not None:
        expected_schema = BACKWARD_RECEIPT_SCHEMA
    else:
        expected_schema = PREFLIGHT_RECEIPT_SCHEMA
    receipts = []
    for run_root in (args.clean_run_root, args.noised_run_root):
        path = Path(run_root) / "training" / "receipt.json"
        receipts.append(
            load_canonical_receipt(
                path,
                expected_file_sha256=file_sha256(path),
                expected_schema=expected_schema,
            )
        )
    return receipts[0], receipts[1]


def _assert_pair_receipts(
    clean: Mapping[str, Any],
    noised: Mapping[str, Any],
    *,
    formal: bool,
) -> Mapping[str, Any]:
    if (
        clean.get("memory_input_kind") != "clean_source"
        or noised.get("memory_input_kind")
        != "same_noise_forward_noised_source"
        or clean.get("pair_invariants") != noised.get("pair_invariants")
        or not isinstance(clean.get("pair_invariants"), Mapping)
    ):
        fail("clean/noised pair invariants differ")
    invariants = dict(clean["pair_invariants"])
    if formal:
        if (
            clean.get("complete") is not True
            or noised.get("complete") is not True
            or clean.get("optimizer_steps") != 80
            or noised.get("optimizer_steps") != 80
        ):
            fail("formal pair did not complete exact80")
    else:
        for receipt in (clean, noised):
            authority = receipt.get("authority")
            if (
                not isinstance(authority, Mapping)
                or authority.get("optimizer_constructed") is not False
                or authority.get("backward_executed") is not False
                or authority.get("checkpoint_written") is not False
            ):
                fail("structural pair receipt contains training authority")
        if invariants.get("stage_a_admission_digest") is not None:
            fail("structural pair unexpectedly consumed Stage-A")
    return invariants


def _assert_backward_pair_receipts(
    clean: Mapping[str, Any], noised: Mapping[str, Any]
) -> Mapping[str, Any]:
    if (
        clean.get("memory_input_kind") != "clean_source"
        or noised.get("memory_input_kind")
        != "same_noise_forward_noised_source"
        or clean.get("pair_invariants") != noised.get("pair_invariants")
        or not isinstance(clean.get("pair_invariants"), Mapping)
    ):
        fail("backward clean/noised pair invariants differ")
    upstreams = [item.get("upstream_structural_pair") for item in (clean, noised)]
    if (
        any(not isinstance(item, Mapping) for item in upstreams)
        or upstreams[0] != upstreams[1]
    ):
        fail("backward pair upstream structural binding differs")
    for receipt in (clean, noised):
        authority = receipt.get("authority")
        backward = receipt.get("backward_feasibility")
        parameters = backward.get("parameters") if isinstance(backward, Mapping) else None
        sync = backward.get("gradient_sync") if isinstance(backward, Mapping) else None
        if (
            receipt.get("complete") is not True
            or not isinstance(authority, Mapping)
            or authority.get("four_microbatch_forward_executed") is not True
            or authority.get("four_microbatch_backward_executed") is not True
            or authority.get("dp2_sp4_gradient_sync_executed") is not True
            or authority.get("optimizer_constructed") is not False
            or authority.get("optimizer_step_count") != 0
            or authority.get("parameters_changed") is not False
            or authority.get("checkpoint_written") is not False
            or not isinstance(backward, Mapping)
            or backward.get("microbatches_per_dp_arm") != 4
            or backward.get("logical_records") != 8
            or not isinstance(parameters, Mapping)
            or parameters.get("unchanged") is not True
            or parameters.get("sha256_before") != parameters.get("sha256_after")
            or not isinstance(sync, Mapping)
            or sync.get("finite_all_parameters_world8") is not True
            or sync.get("identical_full_gradient_digest_world8") is not True
        ):
            fail("backward pair authority/gradient closure differs")
    return dict(clean["pair_invariants"])


def validate_preflight_pair_receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the pair decision and re-open both bound WORLD8 receipts."""

    unsigned = dict(value)
    declared_digest = unsigned.pop("receipt_digest", None)
    arms = value.get("arms")
    shared = value.get("shared_pair_invariants")
    if (
        value.get("schema_version") != PREFLIGHT_PAIR_SCHEMA
        or declared_digest != object_sha256(unsigned)
        or value.get("complete") is not True
        or value.get("decision")
        != "shared_initialization_verified_formal_still_requires_stage_a"
        or value.get("optimizer_authorized") is not False
        or value.get("allowed_arm_differences") != list(ALLOWED_ARM_DIFFERENCES)
        or value.get("parent_allocations_released") is not False
        or not isinstance(arms, list)
        or len(arms) != 2
        or not isinstance(shared, Mapping)
        or shared.get("stage_a_admission_digest") is not None
    ):
        fail("preflight pair decision/shared schema differs")
    _digest(
        shared.get("initial_parameter_digest"),
        length=64,
        label="preflight shared initial parameter digest",
    )
    shared_unsigned = dict(shared)
    shared_declared_digest = shared_unsigned.pop("digest", None)
    if shared_declared_digest != object_sha256(shared_unsigned):
        fail("preflight shared-invariant digest differs")
    reopened = []
    seen_run_roots: set[str] = set()
    seen_ports: set[int] = set()
    for expected, arm in zip(ARMS, arms):
        run_root_value = arm.get("run_root") if isinstance(arm, Mapping) else None
        receipt_path_value = (
            arm.get("receipt_path") if isinstance(arm, Mapping) else None
        )
        try:
            run_root = Path(run_root_value).expanduser()
            receipt_path = Path(receipt_path_value).expanduser()
        except TypeError:
            fail("preflight pair arm path binding differs")
        if (
            not isinstance(arm, Mapping)
            or set(arm)
            != {
                "memory_input_kind",
                "holder_job",
                "holder_node",
                "run_root",
                "master_port",
                "receipt_path",
                "receipt_file_sha256",
                "receipt_digest",
            }
            or arm.get("memory_input_kind") != expected["memory_input_kind"]
            or arm.get("holder_job") != expected["holder_job"]
            or arm.get("holder_node") != expected["holder_node"]
            or type(arm.get("master_port")) is not int
            or not 1024 <= arm["master_port"] <= 65535
            or not run_root.is_absolute()
            or run_root.is_symlink()
            or run_root.resolve(strict=True) != run_root
            or receipt_path != run_root / "training" / "receipt.json"
            or str(run_root) in seen_run_roots
            or arm["master_port"] in seen_ports
        ):
            fail("preflight pair arm binding differs")
        seen_run_roots.add(str(run_root))
        seen_ports.add(arm["master_port"])
        receipt = load_canonical_receipt(
            receipt_path,
            expected_file_sha256=arm.get("receipt_file_sha256"),
            expected_schema=PREFLIGHT_RECEIPT_SCHEMA,
        )
        if (
            receipt.get("receipt_digest") != arm.get("receipt_digest")
            or receipt.get("memory_input_kind") != arm.get("memory_input_kind")
        ):
            fail("preflight pair bound arm receipt differs")
        reopened.append(receipt)
    if _assert_pair_receipts(reopened[0], reopened[1], formal=False) != shared:
        fail("preflight pair reopened shared invariants differ")
    return dict(shared)


def load_formal_pair_admission(
    path_value: str | Path,
    *,
    expected_file_sha256: str,
    memory_input_kind: str,
    expected_shared_invariants_without_initial: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = load_canonical_receipt(
        path_value,
        expected_file_sha256=expected_file_sha256,
        expected_schema=FORMAL_PAIR_ADMISSION_SCHEMA,
    )
    arms = value.get("arms")
    if (
        value.get("complete") is not True
        or value.get("decision") != "admit_both_or_neither"
        or value.get("optimizer_authorized") is not True
        or value.get("optimizer_authorized_by_pair_alone") is not False
        or value.get("allowed_arm_differences") != list(ALLOWED_ARM_DIFFERENCES)
        or value.get("synthetic_target_accessed") is not False
        or value.get("reward_used") is not False
        or not isinstance(arms, list)
        or len(arms) != 2
    ):
        fail("formal pair admission decision/arms differ")
    for expected, arm in zip(ARMS, arms):
        if (
            not isinstance(arm, Mapping)
            or set(arm)
            != {
                "memory_input_kind",
                "holder_job",
                "holder_node",
                "run_root",
                "master_port",
            }
            or arm.get("memory_input_kind") != expected["memory_input_kind"]
            or arm.get("holder_job") != expected["holder_job"]
            or arm.get("holder_node") != expected["holder_node"]
            or type(arm.get("run_root")) is not str
            or not Path(arm["run_root"]).is_absolute()
            or type(arm.get("master_port")) is not int
            or not 1024 <= arm["master_port"] <= 65535
        ):
            fail("formal pair arm binding differs")
    if memory_input_kind not in [arm["memory_input_kind"] for arm in arms]:
        fail("formal pair admission does not contain this arm")
    preflight_pair = load_canonical_receipt(
        value.get("preflight_pair_receipt_path"),
        expected_file_sha256=value.get("preflight_pair_receipt_file_sha256"),
        expected_schema=PREFLIGHT_PAIR_SCHEMA,
    )
    preflight_shared = validate_preflight_pair_receipt(preflight_pair)
    if preflight_pair.get("receipt_digest") != value.get(
        "preflight_pair_receipt_digest"
    ):
        fail("formal admission preflight-pair binding differs")
    shared = value.get("shared_pair_invariants")
    if not isinstance(shared, Mapping):
        fail("formal pair shared invariants are absent")
    stage_a_receipt = value.get("stage_a_admission")
    if not isinstance(stage_a_receipt, Mapping):
        fail("formal pair Stage-A admission is absent")
    stage_a_digest = _digest(
        stage_a_receipt.get("receipt_digest"),
        length=64,
        label="formal pair Stage-A receipt digest",
    )
    _digest(
        value.get("stage_a_admission_file_sha256"),
        length=64,
        label="formal pair Stage-A file SHA",
    )
    reconstructed = dict(preflight_shared)
    reconstructed.pop("digest", None)
    reconstructed["stage_a_admission_digest"] = stage_a_digest
    reconstructed = {**reconstructed, "digest": object_sha256(reconstructed)}
    if reconstructed != shared:
        fail("formal shared invariants are not derived from bound preflight pair")
    expected_initial = _digest(
        shared.get("initial_parameter_digest"),
        length=64,
        label="pair expected initial parameter digest",
    )
    projection = dict(shared)
    projection.pop("initial_parameter_digest", None)
    projection.pop("digest", None)
    expected = dict(expected_shared_invariants_without_initial)
    expected.pop("initial_parameter_digest", None)
    expected.pop("digest", None)
    if projection != expected:
        fail("formal pair shared invariants differ from runner inputs")
    return {**dict(value), "expected_initial_parameter_digest": expected_initial}


def _build_formal_admission(
    args: argparse.Namespace,
    preflight_pair: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_preflight_pair_receipt(preflight_pair)
    stage_a = stage_b.load_stage_a_admission(
        args.stage_a_admission,
        expected_sha256=args.stage_a_admission_sha256,
    )
    shared = dict(preflight_pair["shared_pair_invariants"])
    shared.pop("digest", None)
    shared["stage_a_admission_digest"] = stage_a.receipt_digest
    shared = {**shared, "digest": object_sha256(shared)}
    unsigned = {
        "schema_version": FORMAL_PAIR_ADMISSION_SCHEMA,
        "complete": True,
        "decision": "admit_both_or_neither",
        "optimizer_authorized": True,
        "optimizer_authorized_by_pair_alone": False,
        "preflight_pair_receipt_digest": preflight_pair["receipt_digest"],
        "preflight_pair_receipt_path": str(
            _plain(args.preflight_pair_receipt, label="preflight pair receipt")
        ),
        "preflight_pair_receipt_file_sha256": args.preflight_pair_receipt_sha256,
        "stage_a_admission": stage_a.receipt(),
        "stage_a_admission_file_sha256": args.stage_a_admission_sha256,
        "shared_pair_invariants": shared,
        "allowed_arm_differences": list(ALLOWED_ARM_DIFFERENCES),
        "arms": [
            {
                **{key: arm[key] for key in ("memory_input_kind", "holder_job", "holder_node")},
                "run_root": str(
                    args.clean_run_root if index == 0 else args.noised_run_root
                ),
                "master_port": (
                    args.clean_master_port if index == 0 else args.noised_master_port
                ),
            }
            for index, arm in enumerate(ARMS)
        ],
        "synthetic_target_accessed": False,
        "reward_used": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def _create_pair_root(args: argparse.Namespace) -> Path:
    root = Path(args.pair_output_root).expanduser()
    if (
        not root.is_absolute()
        or root.exists()
        or root.is_symlink()
        or root.parent.resolve(strict=True) != root.parent
    ):
        fail("pair output root must be fresh/canonical")
    root.mkdir(mode=0o700)
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_shared_cli(args)
    pair_root = _create_pair_root(args)
    if args.command == "run-preflight":
        clean, noised = _run_two_wrappers(args)
        shared = _assert_pair_receipts(clean, noised, formal=False)
        arm_rows = []
        for index, (arm, receipt, run_root, port) in enumerate(
            zip(
                ARMS,
                (clean, noised),
                (args.clean_run_root, args.noised_run_root),
                (args.clean_master_port, args.noised_master_port),
            )
        ):
            receipt_path = Path(run_root) / "training" / "receipt.json"
            arm_rows.append(
                {
                    **{key: arm[key] for key in ("memory_input_kind", "holder_job", "holder_node")},
                    "run_root": str(run_root),
                    "master_port": port,
                    "receipt_path": str(receipt_path),
                    "receipt_file_sha256": file_sha256(receipt_path),
                    "receipt_digest": receipt["receipt_digest"],
                }
            )
        unsigned = {
            "schema_version": PREFLIGHT_PAIR_SCHEMA,
            "complete": True,
            "decision": "shared_initialization_verified_formal_still_requires_stage_a",
            "optimizer_authorized": False,
            "shared_pair_invariants": shared,
            "allowed_arm_differences": list(ALLOWED_ARM_DIFFERENCES),
            "arms": arm_rows,
            "parent_allocations_released": False,
        }
        value = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        output = pair_root / "preflight_pair_receipt.json"
        write_create_only_json(output, value)
    elif args.command == "run-backward-preflight":
        upstream_path = _plain(
            args.preflight_pair_receipt, label="backward upstream pair receipt"
        )
        upstream = load_canonical_receipt(
            upstream_path,
            expected_file_sha256=args.preflight_pair_receipt_sha256,
            expected_schema=PREFLIGHT_PAIR_SCHEMA,
        )
        upstream_shared = validate_preflight_pair_receipt(upstream)
        clean, noised = _run_two_wrappers(
            args,
            backward_pair_receipt=upstream_path,
            backward_pair_receipt_sha256=args.preflight_pair_receipt_sha256,
        )
        shared = _assert_backward_pair_receipts(clean, noised)
        for field in (
            "source_only_manifest_file_sha256",
            "source_only_manifest_digest",
            "initial_parameter_digest",
            "bernini_commit",
            "veomni_commit",
            "checkpoint_tree_sha256",
            "checkpoint_content_manifest_sha256",
            "exact40_schedule_sha256",
            "objective",
            "topology",
        ):
            if shared.get(field) != upstream_shared.get(field):
                fail(f"backward pair changed upstream invariant: {field}")
        arm_rows = []
        for arm, receipt, run_root in zip(
            ARMS, (clean, noised), (args.clean_run_root, args.noised_run_root)
        ):
            receipt_path = Path(run_root) / "training" / "receipt.json"
            arm_rows.append(
                {
                    **{
                        key: arm[key]
                        for key in ("memory_input_kind", "holder_job", "holder_node")
                    },
                    "run_root": str(run_root),
                    "receipt_path": str(receipt_path),
                    "receipt_file_sha256": file_sha256(receipt_path),
                    "receipt_digest": receipt["receipt_digest"],
                    "total_gradient_norm": receipt["backward_feasibility"][
                        "gradient_sync"
                    ]["total_norm"],
                    "component_gradient_norms": receipt["backward_feasibility"][
                        "gradient_sync"
                    ]["component_norms"],
                    "resources_world8": receipt["backward_feasibility"][
                        "resources_world8"
                    ],
                }
            )
        unsigned = {
            "schema_version": BACKWARD_PAIR_RESULT_SCHEMA,
            "complete": True,
            "decision": "both_arms_backward_feasible_without_update",
            "upstream_preflight_pair_receipt_path": str(upstream_path),
            "upstream_preflight_pair_receipt_file_sha256": (
                args.preflight_pair_receipt_sha256
            ),
            "upstream_preflight_pair_receipt_digest": upstream["receipt_digest"],
            "shared_pair_invariants": shared,
            "arms": arm_rows,
            "microbatches_per_dp_arm": 4,
            "logical_records_per_arm": 8,
            "optimizer_constructed": False,
            "optimizer_step_count": 0,
            "parameters_changed": False,
            "checkpoint_written": False,
            "parent_allocations_released": False,
            "formal_optimizer_authorized": False,
        }
        value = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        output = pair_root / "backward_feasibility_pair_receipt.json"
        write_create_only_json(output, value)
    else:
        preflight_pair = load_canonical_receipt(
            args.preflight_pair_receipt,
            expected_file_sha256=args.preflight_pair_receipt_sha256,
            expected_schema=PREFLIGHT_PAIR_SCHEMA,
        )
        formal_admission = _build_formal_admission(args, preflight_pair)
        admission_path = pair_root / "formal_pair_admission.json"
        write_create_only_json(admission_path, formal_admission)
        admission_sha = file_sha256(admission_path)
        clean, noised = _run_two_wrappers(
            args,
            formal_pair_admission=admission_path,
            formal_pair_admission_sha256=admission_sha,
            expected_initial_parameter_digest=formal_admission[
                "shared_pair_invariants"
            ]["initial_parameter_digest"],
        )
        shared = _assert_pair_receipts(clean, noised, formal=True)
        if shared != formal_admission["shared_pair_invariants"]:
            fail("formal pair runtime invariants differ from pre-optimizer admission")
        unsigned = {
            "schema_version": FORMAL_PAIR_RESULT_SCHEMA,
            "complete": True,
            "formal_pair_admission_digest": formal_admission["receipt_digest"],
            "formal_pair_admission_file_sha256": admission_sha,
            "shared_pair_invariants": shared,
            "arms": [
                {
                    "memory_input_kind": receipt["memory_input_kind"],
                    "receipt_digest": receipt["receipt_digest"],
                }
                for receipt in (clean, noised)
            ],
            "checkpoint_steps": [0, 20, 40, 60, 80],
            "parent_allocations_released": False,
            "decoded_review_complete": False,
        }
        value = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        output = pair_root / "formal_pair_result.json"
        write_create_only_json(output, value)
    print(
        json.dumps(
            {"output": str(output), "sha256": file_sha256(output)},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run-preflight", "run-backward-preflight", "run-formal"):
        command = sub.add_parser(name)
        command.add_argument("--pair-output-root", required=True)
        command.add_argument("--method-root", required=True)
        command.add_argument("--source-only-manifest", required=True)
        command.add_argument("--source-only-manifest-sha256", required=True)
        command.add_argument("--method-source-revision", required=True)
        command.add_argument("--method-source-archive", required=True)
        command.add_argument("--method-source-archive-sha256", required=True)
        command.add_argument("--method-source-manifest", required=True)
        command.add_argument("--method-source-manifest-sha256", required=True)
        command.add_argument("--clean-run-root", required=True)
        command.add_argument("--noised-run-root", required=True)
        command.add_argument("--clean-master-port", type=int, required=True)
        command.add_argument("--noised-master-port", type=int, required=True)
        if name in ("run-backward-preflight", "run-formal"):
            command.add_argument("--preflight-pair-receipt", required=True)
            command.add_argument("--preflight-pair-receipt-sha256", required=True)
        if name == "run-formal":
            command.add_argument("--stage-a-admission", required=True)
            command.add_argument("--stage-a-admission-sha256", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_ARM_DIFFERENCES",
    "BACKWARD_PAIR_RESULT_SCHEMA",
    "FORMAL_PAIR_ADMISSION_SCHEMA",
    "PREFLIGHT_PAIR_SCHEMA",
    "build_parser",
    "load_formal_pair_admission",
    "main",
]

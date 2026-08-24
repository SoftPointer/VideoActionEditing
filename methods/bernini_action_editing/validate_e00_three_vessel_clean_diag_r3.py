#!/usr/bin/env python3
"""Fail-closed R3 protocol, capability, gate, and final-artifact validator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping, Optional, Sequence

import validate_e00_three_vessel_fresh_keyed_legacy_diagnostic_v1 as legacy


SCHEMA = "bernini-e00-clean-diagnostic-r3-protocol-v3"
REVISION_TAG = "E00_DFIX2_CLEAN_DIAG_R3_CAPABILITY_FINALCLOSURE_20260821"
RNG_SCHEMA = "bernini-e00-legacy-fixed-rng-r3-receipt-v3"
ARM_AUDIT_SCHEMA = "bernini-e00-clean-diagnostic-r3-arm-audit-v3"
AB_GATE_SCHEMA = "bernini-e00-clean-diagnostic-r3-ab-current-bit-exact-gate-v3"
FINAL_SCHEMA = "bernini-e00-clean-diagnostic-r3-final-current-artifact-audit-v3"
CAPABILITY_SCHEMA = "bernini-e00-clean-diagnostic-r3-bridge-capability-v3"
PHASE_A_MARKER_SCHEMA = "bernini-e00-clean-diagnostic-r3-phase-a-stopped-marker-v3"
METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
DEFAULT_PROTOCOL = METHOD_ROOT / "assets/e00_three_vessel_clean_diag_r3_protocol_20260821.json"
ARM_ROLES = legacy.ARM_ROLES
EXPECTED_SEEDS = [
    {"rank": rank, "cpu_seed": 83002700 + rank, "cuda_seed": 83003700 + rank}
    for rank in range(4)
]
REQUIRED_EQUALITIES = [
    "predecode_latent_raw_storage_sha256",
    "mp4_sha256",
    "per_rank_fixed_cpu_initial_rng_sha256",
    "per_rank_fixed_cuda_initial_rng_sha256",
    "raw_keyed_noise_bank_sha256",
    "outer_schedule_digest",
    "frozen_certificate_sha256",
]
REQUIRED_CURRENT_BINDINGS = [
    "native_receipt_path_and_bytes",
    "four_rank_rng_receipt_paths_and_bytes",
    "video_path_and_bytes",
    "arm_audit_path_and_bytes",
]
HEX = set("0123456789abcdef")


class E00R3Error(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise E00R3Error(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} is not an object")
    return value


def _get(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = _mapping(current, path).get(part)
    return current


def _eq(value: Mapping[str, Any], path: str, expected: Any) -> None:
    actual = _get(value, path)
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        _fail(f"{path} differs: {actual!r} != {expected!r}")


def _plain_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        _fail(f"{label} is not a plain file: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        _fail(f"{label} is not a lowercase SHA-256")
    return value


def _load(path: Path, label: str) -> Mapping[str, Any]:
    _plain_file(path, label)
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise E00R3Error(f"{label} is unreadable") from error


def _safe_repo_file(relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        _fail(f"{label} package path is unsafe")
    path = REPO_ROOT / relative
    _plain_file(path, label)
    return path


def protocol_identity(protocol: Mapping[str, Any], path: Path) -> dict[str, Any]:
    _plain_file(path, "R3 protocol")
    return {
        "revision_tag": REVISION_TAG,
        "path": str(path),
        "file_sha256": file_sha256(path),
        "canonical_sha256": canonical_sha256(protocol),
    }


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    for path, expected in (
        ("schema_version", SCHEMA),
        ("revision_tag", REVISION_TAG),
        ("status.draft_only", True),
        ("status.execution_authorized", False),
        ("status.gpu_run_started", False),
        ("status.training_performed", False),
        ("status.independent_package_audit_passed", False),
        ("fixed_initial_rng.scheme", "explicit_rank_owned_cpu_cuda_manual_seed_r3"),
        ("fixed_initial_rng.scope", "inside_fork_rng_immediately_before_entire_legacy_inference_entrypoint"),
        ("fixed_initial_rng.same_rank_state_must_be_bit_exact_across_arms", True),
        ("fixed_initial_rng.caller_state_must_be_restored", True),
        ("arm_order", list(ARM_ROLES)),
        ("ab_gate_contract.required_equalities", REQUIRED_EQUALITIES),
        ("ab_gate_contract.current_artifact_revalidation_required", REQUIRED_CURRENT_BINDINGS),
        ("ab_gate_contract.on_failure", "STOP_WITHOUT_C"),
        ("bridge_capability_contract.fixed_parent_job_id", "143808"),
        ("bridge_capability_contract.fixed_compute_node", "auh7-1b-gpu-292"),
        ("bridge_capability_contract.phase_a_only_token_required", True),
        ("bridge_capability_contract.phase_bc_token_required", True),
        ("bridge_capability_contract.phase_a_and_bc_tokens_must_be_distinct", True),
        ("bridge_capability_contract.phase_a_token_may_admit_only", [ARM_ROLES[0]]),
        ("bridge_capability_contract.phase_bc_token_may_admit_b_before_gate", [ARM_ROLES[1]]),
        ("bridge_capability_contract.phase_bc_token_may_admit_c_only_after_current_gate_revalidation", [ARM_ROLES[2]]),
        ("bridge_capability_contract.bridge_must_validate_authorization_bytes", True),
        ("bridge_capability_contract.bridge_must_validate_phase_a_marker_bytes_for_b_and_c", True),
        ("bridge_capability_contract.bridge_must_validate_current_ab_gate_bytes_for_c", True),
        ("bridge_capability_contract.direct_bridge_without_capability_forbidden", True),
        ("slurm_contract.parent_job_id", "143808"),
        ("slurm_contract.compute_node", "auh7-1b-gpu-292"),
        ("slurm_contract.gpu_count", 4),
        ("slurm_contract.preserve_slurm_visible_device_namespace", True),
        ("slurm_contract.submit_new_job", False),
        ("slurm_contract.cancel_parent_job", False),
        ("slurm_contract.serial_exclusive_steps_only", True),
        ("claims.diagnostic_only", True),
        ("claims.zero_update_frozen_inference", True),
        ("claims.training_authorized", False),
        ("claims.promotion_forbidden", True),
    ):
        _eq(protocol, path, expected)
    if protocol.get("fixed_initial_rng", {}).get("per_rank") != EXPECTED_SEEDS:
        _fail("R3 fixed per-rank seed table differs")
    expected_states = [
        "PACKAGE_REVIEW_REQUIRED",
        "EXTERNAL_A_ONLY_AUTHORIZATION_AND_TOKEN",
        "RUN_A",
        "A_STOPPED_REVIEW_REQUIRED",
        "EXTERNAL_BC_AUTHORIZATION_AND_TOKEN_BOUND_TO_A_BYTES",
        "RUN_B",
        "FRESH_AB_CURRENT_ARTIFACT_AND_BIT_EXACT_GATE",
        "RUN_C_ONLY_AFTER_BRIDGE_REVALIDATES_CURRENT_GATE_MARKER_AUTH",
        "FRESH_ABC_FINAL_CURRENT_ARTIFACT_CLOSURE",
    ]
    if protocol.get("two_phase_state_machine") != expected_states:
        _fail("R3 state machine differs")
    base = _mapping(protocol.get("base_diagnostic_spec"), "base spec binding")
    base_path = _safe_repo_file(base.get("package_relative_path"), "base diagnostic spec")
    if file_sha256(base_path) != base.get("file_sha256"):
        _fail("base diagnostic spec file SHA-256 differs")
    base_spec = legacy.load_spec(base_path)
    if canonical_sha256(base_spec) != base.get("canonical_sha256"):
        _fail("base diagnostic spec canonical SHA-256 differs")
    dependencies = _mapping(protocol.get("sealed_runtime_dependencies"), "sealed dependencies")
    expected_dependencies = {
        "methods/bernini_action_editing/e00_legacy_infer_fork_rng_wrapper_v1.py",
        "methods/bernini_action_editing/validate_e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.py",
        "methods/bernini_action_editing/assets/e00_source_frame0_static81_25fps_704x1056_v1.mp4",
    }
    if set(dependencies) != expected_dependencies:
        _fail("R3 sealed dependency closure differs")
    for relative, expected_sha in dependencies.items():
        if file_sha256(_safe_repo_file(relative, f"dependency {relative}")) != expected_sha:
            _fail(f"R3 dependency SHA-256 differs: {relative}")
    templates = _mapping(protocol.get("authorization_templates"), "authorization templates")
    for phase, schema in (
        ("phase_a", "bernini-e00-clean-diagnostic-r3-phase-a-authorization-v3"),
        ("phase_bc", "bernini-e00-clean-diagnostic-r3-phase-bc-authorization-v3"),
    ):
        template = _load(_safe_repo_file(templates.get(phase), f"{phase} template"), f"{phase} template")
        if template.get("schema_version") != schema or template.get("execution_authorized") is not False or template.get("authorized_by") != "":
            _fail(f"{phase} authorization template is not inert")
    return {
        "schema_version": SCHEMA,
        "revision_tag": REVISION_TAG,
        "canonical_sha256": canonical_sha256(protocol),
        "base_spec_canonical_sha256": base["canonical_sha256"],
    }


def load_protocol(path: Path | str = DEFAULT_PROTOCOL) -> Mapping[str, Any]:
    value = _load(Path(path), "R3 protocol")
    validate_protocol(value)
    return value


def load_base_spec(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    return legacy.load_spec(_safe_repo_file(protocol["base_diagnostic_spec"]["package_relative_path"], "bound base spec"))


def validate_rng_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    arm_role: str,
    expected_output_path: str,
    native_receipt_sha256: str,
) -> dict[str, Any]:
    if len(receipts) != 4:
        _fail("exactly four R3 RNG receipts are required")
    projected = []
    for receipt in receipts:
        item = copy.deepcopy(dict(receipt))
        item["schema_version"] = legacy.RNG_SCHEMA
        projected.append(item)
    try:
        base = legacy.validate_rng_receipts(
            projected,
            arm_role=arm_role,
            expected_output_path=expected_output_path,
            native_receipt_sha256=native_receipt_sha256,
        )
    except legacy.E00LegacyDiagnosticError as error:
        raise E00R3Error(str(error)) from error
    protocol_id = protocol_identity(protocol, protocol_path)
    ordered = sorted(receipts, key=lambda row: row.get("rank", -1))
    fixed = []
    for rank, row in enumerate(ordered):
        for path, expected in (
            ("schema_version", RNG_SCHEMA),
            ("revision_tag", REVISION_TAG),
            ("rank", rank),
            ("protocol.file_sha256", protocol_id["file_sha256"]),
            ("protocol.canonical_sha256", protocol_id["canonical_sha256"]),
            ("fixed_initial_rng.enabled", True),
            ("fixed_initial_rng.scheme", "explicit_rank_owned_cpu_cuda_manual_seed_r3"),
            ("fixed_initial_rng.scope", "inside_fork_rng_immediately_before_entire_legacy_inference_entrypoint"),
            ("fixed_initial_rng.cpu_seed", EXPECTED_SEEDS[rank]["cpu_seed"]),
            ("fixed_initial_rng.cuda_seed", EXPECTED_SEEDS[rank]["cuda_seed"]),
        ):
            _eq(row, path, expected)
        initial = _mapping(_get(row, "fixed_initial_rng.seeded_initial"), "initial RNG")
        terminal = _mapping(_get(row, "fixed_initial_rng.terminal_before_restore"), "terminal RNG")
        for name in ("cpu_sha256", "cuda_sha256"):
            _hex(initial.get(name), f"rank {rank} initial {name}")
            _hex(terminal.get(name), f"rank {rank} terminal {name}")
        fixed.append({
            "rank": rank,
            "cpu_seed": EXPECTED_SEEDS[rank]["cpu_seed"],
            "cuda_seed": EXPECTED_SEEDS[rank]["cuda_seed"],
            "cpu_initial_sha256": initial["cpu_sha256"],
            "cuda_initial_sha256": initial["cuda_sha256"],
        })
    return {
        **base,
        "explicit_initial_rng": True,
        "per_rank_fixed_initial_rng": fixed,
        "per_rank_fixed_initial_rng_digest": canonical_sha256(fixed),
    }


def build_arm_audit(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    arm_role: str,
    video_path: Path,
    native_receipt_path: Path,
    rng_receipt_paths: Sequence[Path],
) -> dict[str, Any]:
    if len(rng_receipt_paths) != 4:
        _fail("exactly four RNG receipt paths are required")
    video = video_path
    native_receipt = _load(native_receipt_path, "native receipt")
    rng_receipts = [_load(path, f"rank {rank} RNG receipt") for rank, path in enumerate(rng_receipt_paths)]
    spec = load_base_spec(protocol)
    try:
        native = legacy.validate_native_receipt(native_receipt, spec=spec, arm_role=arm_role, video=video)
    except legacy.E00LegacyDiagnosticError as error:
        raise E00R3Error(str(error)) from error
    output_path = _get(native_receipt, "output.path")
    if output_path != str(video):
        _fail("native receipt output path differs from current video path")
    rng = validate_rng_receipts(
        rng_receipts,
        protocol=protocol,
        protocol_path=protocol_path,
        arm_role=arm_role,
        expected_output_path=output_path,
        native_receipt_sha256=file_sha256(native_receipt_path),
    )
    arm = next(row for row in spec["arms"] if row["arm_role"] == arm_role)
    ordered_rng_artifacts = []
    for rank, path in enumerate(rng_receipt_paths):
        if rng_receipts[rank].get("rank") != rank:
            _fail("RNG receipt path order does not match rank order")
        ordered_rng_artifacts.append({"rank": rank, "path": str(path), "sha256": file_sha256(path)})
    return {
        "schema_version": ARM_AUDIT_SCHEMA,
        "revision_tag": REVISION_TAG,
        "complete": True,
        "arm_role": arm_role,
        "label": arm["label"],
        "protocol": protocol_identity(protocol, protocol_path),
        "spec_canonical_sha256": canonical_sha256(spec),
        "honest_scope": arm["honest_name"],
        "anchor_free": False,
        "training_performed": False,
        "optimization_steps": 0,
        "artifacts": {
            "video": {"path": str(video), "sha256": file_sha256(video)},
            "native_receipt": {"path": str(native_receipt_path), "sha256": file_sha256(native_receipt_path)},
            "rng_receipts_rank_order": ordered_rng_artifacts,
        },
        "native": native,
        "rng_and_noise": rng,
    }


def validate_current_artifact_bytes(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    """Reload every path in one arm closure and bind its current file bytes."""

    video = _mapping(artifacts.get("video"), "video artifact")
    native = _mapping(artifacts.get("native_receipt"), "native receipt artifact")
    rng_rows = artifacts.get("rng_receipts_rank_order")
    if not isinstance(rng_rows, list) or len(rng_rows) != 4:
        _fail("current RNG artifact closure must contain exactly four ranks")
    normalized = {"video": {}, "native_receipt": {}, "rng_receipts_rank_order": []}
    for row, label, target in (
        (video, "current video", normalized["video"]),
        (native, "current native receipt", normalized["native_receipt"]),
    ):
        if not isinstance(row.get("path"), str):
            _fail(f"{label} path is absent")
        path = Path(row["path"])
        _plain_file(path, label)
        observed = file_sha256(path)
        if row.get("sha256") != observed:
            _fail(f"{label} bytes changed after arm audit")
        target.update({"path": str(path), "sha256": observed})
    for rank, row in enumerate(rng_rows):
        if not isinstance(row, Mapping) or row.get("rank") != rank or not isinstance(row.get("path"), str):
            _fail("current RNG receipt rank/path closure differs")
        path = Path(row["path"])
        _plain_file(path, f"current rank {rank} RNG receipt")
        observed = file_sha256(path)
        if row.get("sha256") != observed:
            _fail(f"current rank {rank} RNG receipt bytes changed after arm audit")
        normalized["rng_receipts_rank_order"].append(
            {"rank": rank, "path": str(path), "sha256": observed}
        )
    return {"artifacts": normalized, "closure_sha256": canonical_sha256(normalized)}


def revalidate_arm_current(
    audit_path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    expected_role: str,
) -> Mapping[str, Any]:
    stored = _load(audit_path, f"{expected_role} arm audit")
    if stored.get("schema_version") != ARM_AUDIT_SCHEMA or stored.get("arm_role") != expected_role:
        _fail("stored R3 arm audit schema/role differs")
    validate_current_artifact_bytes(_mapping(stored.get("artifacts"), "stored artifacts"))
    video_path = Path(_get(stored, "artifacts.video.path"))
    native_path = Path(_get(stored, "artifacts.native_receipt.path"))
    rng_rows = _get(stored, "artifacts.rng_receipts_rank_order")
    if not isinstance(rng_rows, list) or len(rng_rows) != 4:
        _fail("stored R3 RNG artifact closure differs")
    rng_paths = []
    for rank, row in enumerate(rng_rows):
        if not isinstance(row, Mapping) or row.get("rank") != rank or not isinstance(row.get("path"), str):
            _fail("stored R3 RNG path/rank closure differs")
        rng_paths.append(Path(row["path"]))
    rebuilt = build_arm_audit(
        protocol=protocol,
        protocol_path=protocol_path,
        arm_role=expected_role,
        video_path=video_path,
        native_receipt_path=native_path,
        rng_receipt_paths=rng_paths,
    )
    if rebuilt != stored:
        _fail(f"{expected_role} current native/RNG/video bytes or paths differ from arm audit")
    return stored


def _cross_equal(audits: Sequence[Mapping[str, Any]], path: str) -> Any:
    values = [_get(audit, path) for audit in audits]
    if any(value != values[0] for value in values[1:]):
        _fail(f"cross-arm R3 equality differs at {path}")
    return values[0]


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    output[field] = canonical_sha256(output)
    return output


def _verify_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = copy.deepcopy(dict(value))
    claimed = unsigned.pop(field, None)
    if claimed != canonical_sha256(unsigned):
        _fail(f"{label} embedded digest differs")


def build_ab_gate(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    a_audit_path: Path,
    b_audit_path: Path,
) -> dict[str, Any]:
    a = revalidate_arm_current(a_audit_path, protocol=protocol, protocol_path=protocol_path, expected_role=ARM_ROLES[0])
    b = revalidate_arm_current(b_audit_path, protocol=protocol, protocol_path=protocol_path, expected_role=ARM_ROLES[1])
    audits = (a, b)
    for path in (
        "protocol.file_sha256",
        "protocol.canonical_sha256",
        "spec_canonical_sha256",
        "rng_and_noise.per_rank_fixed_initial_rng",
        "rng_and_noise.per_rank_fixed_initial_rng_digest",
        "rng_and_noise.raw_noise_rows",
        "rng_and_noise.raw_noise_bank_sha256",
        "native.outer_schedule_digest",
        "native.frozen_certificate_sha256",
    ):
        _cross_equal(audits, path)
    latent = _cross_equal(audits, "rng_and_noise.predecode_latent_sha256")
    video_sha = _cross_equal(audits, "artifacts.video.sha256")
    if _get(a, "native.native_output_sha256") != video_sha or _get(b, "native.native_output_sha256") != video_sha:
        _fail("A/B current MP4 bytes differ from native output receipt")
    value = {
        "schema_version": AB_GATE_SCHEMA,
        "revision_tag": REVISION_TAG,
        "complete": True,
        "c_execution_gate_passed": True,
        "protocol": protocol_identity(protocol, protocol_path),
        "required_equalities": list(REQUIRED_EQUALITIES),
        "current_artifact_revalidation": list(REQUIRED_CURRENT_BINDINGS),
        "a_arm_audit": {"path": str(a_audit_path), "sha256": file_sha256(a_audit_path)},
        "b_arm_audit": {"path": str(b_audit_path), "sha256": file_sha256(b_audit_path)},
        "predecode_latent_raw_storage_sha256": latent,
        "mp4_sha256": video_sha,
        "per_rank_fixed_initial_rng_digest": _get(a, "rng_and_noise.per_rank_fixed_initial_rng_digest"),
        "raw_keyed_noise_bank_sha256": _get(a, "rng_and_noise.raw_noise_bank_sha256"),
        "outer_schedule_digest": _get(a, "native.outer_schedule_digest"),
        "frozen_certificate_sha256": _get(a, "native.frozen_certificate_sha256"),
        "a_current_artifact_closure_sha256": canonical_sha256(a["artifacts"]),
        "b_current_artifact_closure_sha256": canonical_sha256(b["artifacts"]),
        "only_admitted_next_arm": ARM_ROLES[2],
        "training_performed": False,
    }
    return _seal(value, "gate_digest")


def verify_ab_gate_current(
    gate_path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
) -> Mapping[str, Any]:
    stored = _load(gate_path, "R3 A/B gate")
    _verify_seal(stored, "gate_digest", "R3 A/B gate")
    if stored.get("required_equalities") != REQUIRED_EQUALITIES:
        _fail("stored R3 gate required_equalities differs")
    a_path = Path(_get(stored, "a_arm_audit.path"))
    b_path = Path(_get(stored, "b_arm_audit.path"))
    rebuilt = build_ab_gate(protocol=protocol, protocol_path=protocol_path, a_audit_path=a_path, b_audit_path=b_path)
    if rebuilt != stored:
        _fail("stored R3 gate differs from fresh current-artifact recomputation")
    return stored


def _authorization_common(
    authorization: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    package_manifest_path: Path,
    capability_token_sha256: str,
) -> None:
    protocol_id = protocol_identity(protocol, protocol_path)
    _hex(capability_token_sha256, "bridge capability token SHA-256")
    for path, expected in (
        ("execution_authorized", True),
        ("package_manifest_sha256", file_sha256(package_manifest_path)),
        ("protocol_file_sha256", protocol_id["file_sha256"]),
        ("protocol_canonical_sha256", protocol_id["canonical_sha256"]),
        ("bridge_capability_token_sha256", capability_token_sha256),
        ("parent_job_id", "143808"),
        ("compute_node", "auh7-1b-gpu-292"),
        ("sp4_observer_released_node292", True),
    ):
        _eq(authorization, path, expected)
    if not isinstance(authorization.get("authorized_by"), str) or not authorization["authorized_by"]:
        _fail("bridge authorization has no reviewer identity")


def validate_phase_a_marker_current(
    marker_path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    package_manifest_path: Path,
) -> Mapping[str, Any]:
    marker = _load(marker_path, "phase A stopped marker")
    protocol_id = protocol_identity(protocol, protocol_path)
    for path, expected in (
        ("schema_version", PHASE_A_MARKER_SCHEMA),
        ("revision_tag", REVISION_TAG),
        ("state", "A_STOPPED_REVIEW_REQUIRED"),
        ("complete", True),
        ("training_performed", False),
        ("optimization_steps", 0),
        ("parent_job_id", "143808"),
        ("compute_node", "auh7-1b-gpu-292"),
        ("package_manifest_sha256", file_sha256(package_manifest_path)),
        ("protocol_file_sha256", protocol_id["file_sha256"]),
        ("protocol_canonical_sha256", protocol_id["canonical_sha256"]),
        ("observed_arm_order", [ARM_ROLES[0]]),
        ("must_stop_after_a", True),
        ("phase_bc_execution_authorized", False),
    ):
        _eq(marker, path, expected)
    auth_path = Path(_get(marker, "phase_a_authorization.path"))
    audit_path = Path(_get(marker, "phase_a_arm_audit.path"))
    video_path = Path(_get(marker, "phase_a_video.path"))
    for path, expected_sha, label in (
        (auth_path, _get(marker, "phase_a_authorization.sha256"), "phase A authorization"),
        (audit_path, _get(marker, "phase_a_arm_audit.sha256"), "phase A arm audit"),
        (video_path, _get(marker, "phase_a_video.sha256"), "phase A MP4"),
    ):
        _plain_file(path, label)
        if file_sha256(path) != expected_sha:
            _fail(f"current {label} bytes differ from phase A marker")
    phase_a_token_sha = _hex(
        marker.get("phase_a_capability_token_sha256"),
        "phase A marker capability token SHA-256",
    )
    phase_a_auth = _load(auth_path, "current phase A authorization")
    if (
        phase_a_auth.get("schema_version")
        != "bernini-e00-clean-diagnostic-r3-phase-a-authorization-v3"
        or phase_a_auth.get("execution_authorized") is not True
        or phase_a_auth.get("bridge_capability_token_sha256") != phase_a_token_sha
    ):
        _fail("current phase A authorization/token differs from stopped marker")
    a = revalidate_arm_current(audit_path, protocol=protocol, protocol_path=protocol_path, expected_role=ARM_ROLES[0])
    if _get(a, "artifacts.video.path") != str(video_path) or _get(a, "artifacts.video.sha256") != file_sha256(video_path):
        _fail("phase A marker video differs from current arm closure")
    return marker


def validate_bridge_capability(
    *,
    phase: str,
    arm_role: str,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    package_manifest_path: Path,
    authorization_path: Path,
    capability_token_sha256: str,
    phase_a_marker_path: Optional[Path] = None,
    ab_gate_path: Optional[Path] = None,
) -> dict[str, Any]:
    if phase not in ("A", "B", "C"):
        _fail("bridge phase differs")
    expected_role = {"A": ARM_ROLES[0], "B": ARM_ROLES[1], "C": ARM_ROLES[2]}[phase]
    if arm_role != expected_role:
        _fail("bridge phase cannot admit the requested arm")
    authorization = _load(authorization_path, f"phase {phase} authorization")
    _authorization_common(
        authorization,
        protocol=protocol,
        protocol_path=protocol_path,
        package_manifest_path=package_manifest_path,
        capability_token_sha256=capability_token_sha256,
    )
    if phase == "A":
        expected_keys = {
            "schema_version", "execution_authorized", "authorized_phase", "package_manifest_sha256",
            "protocol_file_sha256", "protocol_canonical_sha256", "bridge_capability_token_sha256",
            "parent_job_id", "compute_node", "only_authorized_arm", "must_stop_after_a",
            "bc_execution_authorized", "sp4_observer_released_node292", "authorized_by",
        }
        if set(authorization) != expected_keys:
            _fail("phase A authorization field closure differs")
        for path, expected in (
            ("schema_version", "bernini-e00-clean-diagnostic-r3-phase-a-authorization-v3"),
            ("authorized_phase", "A_ONLY_THEN_STOP"),
            ("only_authorized_arm", ARM_ROLES[0]),
            ("must_stop_after_a", True),
            ("bc_execution_authorized", False),
        ):
            _eq(authorization, path, expected)
        if phase_a_marker_path is not None or ab_gate_path is not None:
            _fail("phase A bridge must not accept later-phase evidence")
    else:
        expected_keys = {
            "schema_version", "execution_authorized", "authorized_phase", "package_manifest_sha256",
            "protocol_file_sha256", "protocol_canonical_sha256", "bridge_capability_token_sha256",
            "phase_a_stopped_marker_sha256", "phase_a_arm_audit_sha256", "phase_a_mp4_sha256",
            "parent_job_id", "compute_node", "authorized_arm_order",
            "c_requires_bridge_revalidated_current_ab_gate", "stop_without_c_on_gate_failure",
            "sp4_observer_released_node292", "authorized_by",
        }
        if set(authorization) != expected_keys:
            _fail("phase BC authorization field closure differs")
        for path, expected in (
            ("schema_version", "bernini-e00-clean-diagnostic-r3-phase-bc-authorization-v3"),
            ("authorized_phase", "B_THEN_CURRENT_AB_GATE_THEN_C"),
            ("authorized_arm_order", [ARM_ROLES[1], ARM_ROLES[2]]),
            ("c_requires_bridge_revalidated_current_ab_gate", True),
            ("stop_without_c_on_gate_failure", True),
        ):
            _eq(authorization, path, expected)
        if phase_a_marker_path is None:
            _fail("phase B/C bridge requires the current phase A marker")
        marker = validate_phase_a_marker_current(
            phase_a_marker_path,
            protocol=protocol,
            protocol_path=protocol_path,
            package_manifest_path=package_manifest_path,
        )
        if file_sha256(phase_a_marker_path) != authorization.get("phase_a_stopped_marker_sha256"):
            _fail("current phase A marker bytes differ from BC authorization")
        if _get(marker, "phase_a_arm_audit.sha256") != authorization.get("phase_a_arm_audit_sha256"):
            _fail("current phase A audit binding differs from BC authorization")
        if _get(marker, "phase_a_video.sha256") != authorization.get("phase_a_mp4_sha256"):
            _fail("current phase A MP4 binding differs from BC authorization")
        if capability_token_sha256 == marker.get("phase_a_capability_token_sha256"):
            _fail("phase BC capability token must be distinct from phase A token")
        if phase == "B":
            if ab_gate_path is not None:
                _fail("phase B bridge cannot consume or infer a C gate")
        else:
            if ab_gate_path is None:
                _fail("phase C bridge requires a current A/B gate")
            gate = verify_ab_gate_current(ab_gate_path, protocol=protocol, protocol_path=protocol_path)
            if gate.get("only_admitted_next_arm") != ARM_ROLES[2] or gate.get("c_execution_gate_passed") is not True:
                _fail("current A/B gate does not admit C")
    value = {
        "schema_version": CAPABILITY_SCHEMA,
        "revision_tag": REVISION_TAG,
        "complete": True,
        "phase": phase,
        "arm_role": arm_role,
        "authorization": {"path": str(authorization_path), "sha256": file_sha256(authorization_path)},
        "package_manifest": {"path": str(package_manifest_path), "sha256": file_sha256(package_manifest_path)},
        "capability_token_sha256": capability_token_sha256,
        "phase_a_marker": (
            {"path": str(phase_a_marker_path), "sha256": file_sha256(phase_a_marker_path)}
            if phase_a_marker_path is not None else None
        ),
        "ab_gate": (
            {"path": str(ab_gate_path), "sha256": file_sha256(ab_gate_path)}
            if ab_gate_path is not None else None
        ),
        "only_this_arm_admitted": True,
        "training_performed": False,
    }
    return _seal(value, "capability_digest")


def build_final_audit(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    a_audit_path: Path,
    b_audit_path: Path,
    c_audit_path: Path,
    ab_gate_path: Path,
) -> dict[str, Any]:
    gate = verify_ab_gate_current(ab_gate_path, protocol=protocol, protocol_path=protocol_path)
    audits = [
        revalidate_arm_current(path, protocol=protocol, protocol_path=protocol_path, expected_role=role)
        for path, role in zip((a_audit_path, b_audit_path, c_audit_path), ARM_ROLES)
    ]
    if str(a_audit_path) != _get(gate, "a_arm_audit.path") or str(b_audit_path) != _get(gate, "b_arm_audit.path"):
        _fail("final audit A/B paths differ from current gate")
    for path in (
        "protocol.file_sha256", "protocol.canonical_sha256", "spec_canonical_sha256",
        "rng_and_noise.per_rank_fixed_initial_rng", "rng_and_noise.per_rank_fixed_initial_rng_digest",
        "rng_and_noise.raw_noise_rows", "rng_and_noise.raw_noise_bank_sha256",
        "native.outer_schedule_digest", "native.frozen_certificate_sha256",
    ):
        _cross_equal(audits, path)
    if [_get(audit, "native.target_route_replay_steps") for audit in audits] != [0, 0, 40]:
        _fail("final A/B/C intervention order differs")
    artifact_rows = []
    for role, audit_path, audit in zip(ARM_ROLES, (a_audit_path, b_audit_path, c_audit_path), audits):
        artifact_rows.append({
            "role": role,
            "arm_audit": {"path": str(audit_path), "sha256": file_sha256(audit_path)},
            "current_artifacts": copy.deepcopy(audit["artifacts"]),
            "current_artifact_closure_sha256": canonical_sha256(audit["artifacts"]),
        })
    value = {
        "schema_version": FINAL_SCHEMA,
        "revision_tag": REVISION_TAG,
        "complete": True,
        "diagnostic_only": True,
        "training_performed": False,
        "arm_order": list(ARM_ROLES),
        "ab_gate": {"path": str(ab_gate_path), "sha256": file_sha256(ab_gate_path)},
        "arms_revalidated_from_current_native_and_four_rng_receipt_bytes": artifact_rows,
        "all_current_artifact_paths_and_bytes_bound": True,
        "same_fixed_initial_rng_all_arms": True,
        "same_raw_keyed_noise_all_arms": True,
        "same_outer_schedule_all_arms": True,
        "same_frozen_model_all_arms": True,
        "ab_predecode_latent_bit_exact": True,
        "ab_mp4_bit_exact": True,
    }
    return _seal(value, "final_audit_digest")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite R3 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        _fail(f"R3 temporary output already exists: {temporary}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    arm = sub.add_parser("arm")
    arm.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    arm.add_argument("--arm-role", required=True, choices=ARM_ROLES)
    arm.add_argument("--native-receipt", required=True)
    arm.add_argument("--rng-receipt", action="append", required=True)
    arm.add_argument("--video", required=True)
    arm.add_argument("--audit-output", required=True)
    gate = sub.add_parser("ab-gate")
    gate.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    gate.add_argument("--a-audit", required=True)
    gate.add_argument("--b-audit", required=True)
    gate.add_argument("--gate-output", required=True)
    verify_gate = sub.add_parser("verify-ab-gate")
    verify_gate.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    verify_gate.add_argument("--ab-gate", required=True)
    capability = sub.add_parser("bridge-capability")
    capability.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    capability.add_argument("--phase", required=True, choices=("A", "B", "C"))
    capability.add_argument("--arm-role", required=True, choices=ARM_ROLES)
    capability.add_argument("--package-manifest", required=True)
    capability.add_argument("--authorization", required=True)
    capability.add_argument("--capability-token-sha256", required=True)
    capability.add_argument("--phase-a-marker")
    capability.add_argument("--ab-gate")
    final = sub.add_parser("abc-final")
    final.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    final.add_argument("--a-audit", required=True)
    final.add_argument("--b-audit", required=True)
    final.add_argument("--c-audit", required=True)
    final.add_argument("--ab-gate", required=True)
    final.add_argument("--audit-output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = Path(args.protocol)
    protocol = load_protocol(protocol_path)
    if args.command == "protocol":
        value = {**validate_protocol(protocol), **protocol_identity(protocol, protocol_path)}
    elif args.command == "arm":
        rng_paths = [Path(path) for path in args.rng_receipt]
        value = build_arm_audit(
            protocol=protocol,
            protocol_path=protocol_path,
            arm_role=args.arm_role,
            video_path=Path(args.video),
            native_receipt_path=Path(args.native_receipt),
            rng_receipt_paths=rng_paths,
        )
        legacy._probe_video(Path(args.video))
        _write_new_json(Path(args.audit_output), value)
    elif args.command == "ab-gate":
        value = build_ab_gate(
            protocol=protocol,
            protocol_path=protocol_path,
            a_audit_path=Path(args.a_audit),
            b_audit_path=Path(args.b_audit),
        )
        _write_new_json(Path(args.gate_output), value)
    elif args.command == "verify-ab-gate":
        value = verify_ab_gate_current(Path(args.ab_gate), protocol=protocol, protocol_path=protocol_path)
    elif args.command == "bridge-capability":
        value = validate_bridge_capability(
            phase=args.phase,
            arm_role=args.arm_role,
            protocol=protocol,
            protocol_path=protocol_path,
            package_manifest_path=Path(args.package_manifest),
            authorization_path=Path(args.authorization),
            capability_token_sha256=args.capability_token_sha256,
            phase_a_marker_path=Path(args.phase_a_marker) if args.phase_a_marker else None,
            ab_gate_path=Path(args.ab_gate) if args.ab_gate else None,
        )
    else:
        value = build_final_audit(
            protocol=protocol,
            protocol_path=protocol_path,
            a_audit_path=Path(args.a_audit),
            b_audit_path=Path(args.b_audit),
            c_audit_path=Path(args.c_audit),
            ab_gate_path=Path(args.ab_gate),
        )
        _write_new_json(Path(args.audit_output), value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

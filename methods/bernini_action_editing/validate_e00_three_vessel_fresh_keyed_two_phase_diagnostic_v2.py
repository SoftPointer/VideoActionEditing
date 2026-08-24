#!/usr/bin/env python3
"""Fail-closed validator for the E00 clean diagnostic R2 state machine.

The validator keeps A, B, and C serial.  It refuses to construct the gate
that admits C unless A/B have identical predecode latent bytes, identical
MP4 bytes, identical keyed-noise/schedule/frozen-model evidence, and the same
explicit per-rank CPU/CUDA initial RNG state bytes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping, Optional, Sequence

import validate_e00_three_vessel_fresh_keyed_legacy_diagnostic_v1 as legacy


SCHEMA = "bernini-e00-three-vessel-fresh-keyed-two-phase-protocol-v2"
REVISION_TAG = "E00_DFIX2_CLEAN_DIAG_R2_FIXED_RNG_TWO_PHASE_20260821"
RNG_SCHEMA = "bernini-e00-legacy-infer-fixed-initial-rng-audit-v2"
ARM_AUDIT_SCHEMA = "bernini-e00-clean-diagnostic-r2-arm-audit-v2"
AB_GATE_SCHEMA = "bernini-e00-clean-diagnostic-r2-ab-bit-exact-gate-v2"
ABC_FINAL_SCHEMA = "bernini-e00-clean-diagnostic-r2-abc-final-audit-v2"
METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
DEFAULT_PROTOCOL = (
    METHOD_ROOT
    / "assets/e00_three_vessel_fresh_keyed_two_phase_diagnostic_v2.json"
)
ARM_ROLES = legacy.ARM_ROLES
EXPECTED_SEEDS = [
    {"rank": rank, "cpu_seed": 82002700 + rank, "cuda_seed": 82003700 + rank}
    for rank in range(4)
]
HEX = set("0123456789abcdef")


class E00TwoPhaseDiagnosticError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise E00TwoPhaseDiagnosticError(message)


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
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hex(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        _fail(f"{label} is not a lowercase SHA-256")
    return value


def _load(path: Path, label: str) -> Mapping[str, Any]:
    _plain_file(path, label)
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise E00TwoPhaseDiagnosticError(f"{label} is unreadable") from error


def _safe_repo_file(relative: Any, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        _fail(f"{label} package path is unsafe")
    path = REPO_ROOT / relative
    _plain_file(path, label)
    return path


def protocol_identity(protocol: Mapping[str, Any], path: Path) -> dict[str, Any]:
    _plain_file(path, "protocol")
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
        ("fixed_initial_rng.scheme", "explicit_rank_owned_cpu_cuda_manual_seed_v2"),
        (
            "fixed_initial_rng.scope",
            "inside_fork_rng_immediately_before_entire_legacy_inference_entrypoint",
        ),
        ("fixed_initial_rng.same_rank_state_must_be_bit_exact_across_arms", True),
        ("fixed_initial_rng.caller_state_must_be_restored", True),
        ("arm_order", list(ARM_ROLES)),
        ("review_marker_contract.must_name_full_arm_order", True),
        ("review_marker_contract.must_name_both_authorized_launchers", True),
        ("review_marker_contract.must_require_separate_a_and_bc_authorizations", True),
        ("review_marker_contract.must_require_a_stop", True),
        ("review_marker_contract.must_require_ab_bit_exact_before_c", True),
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
    if _get(protocol, "fixed_initial_rng.per_rank") != EXPECTED_SEEDS:
        _fail("fixed per-rank seed table differs")

    states = protocol.get("two_phase_state_machine")
    if not isinstance(states, list) or [row.get("state") for row in states] != [
        "PACKAGE_REVIEW_REQUIRED",
        "EXTERNAL_A_ONLY_AUTHORIZATION",
        "A_STOPPED_REVIEW_REQUIRED",
        "EXTERNAL_BC_AUTHORIZATION_BOUND_TO_A_BYTES",
        "A_B_BIT_EXACT_GATE",
        "C_MAY_RUN_UNDER_EXISTING_BC_AUTHORIZATION",
    ]:
        _fail("two-phase state-machine order differs")
    if (
        states[1].get("may_run") != [ARM_ROLES[0]]
        or states[1].get("must_stop_after") != ARM_ROLES[0]
        or states[1].get("next_phase_authorized") is not False
        or states[3].get("may_run_first") != [ARM_ROLES[1]]
        or states[4].get("on_failure") != "STOP_WITHOUT_C"
        or states[5].get("may_run") != [ARM_ROLES[2]]
    ):
        _fail("A-only stop or B-before-C gate contract differs")

    base_row = _mapping(protocol.get("base_diagnostic_spec"), "base spec binding")
    base_path = _safe_repo_file(base_row.get("package_relative_path"), "base spec")
    if file_sha256(base_path) != base_row.get("file_sha256"):
        _fail("base diagnostic spec file SHA-256 differs")
    base_spec = legacy.load_spec(base_path)
    if canonical_sha256(base_spec) != base_row.get("canonical_sha256"):
        _fail("base diagnostic spec canonical SHA-256 differs")

    dependencies = _mapping(
        protocol.get("sealed_runtime_dependencies"), "sealed runtime dependencies"
    )
    expected_dependency_paths = {
        "methods/bernini_action_editing/e00_legacy_infer_fork_rng_wrapper_v1.py",
        "methods/bernini_action_editing/validate_e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.py",
        "methods/bernini_action_editing/assets/e00_source_frame0_static81_25fps_704x1056_v1.mp4",
    }
    if set(dependencies) != expected_dependency_paths:
        _fail("sealed runtime dependency path closure differs")
    for relative, expected_sha in dependencies.items():
        path = _safe_repo_file(relative, f"sealed dependency {relative}")
        if file_sha256(path) != expected_sha:
            _fail(f"sealed runtime dependency SHA-256 differs: {relative}")

    templates = _mapping(protocol.get("authorization_templates"), "templates")
    for phase, schema in (
        ("phase_a", "bernini-e00-clean-diagnostic-r2-phase-a-execution-authorization-v2"),
        ("phase_bc", "bernini-e00-clean-diagnostic-r2-phase-bc-execution-authorization-v2"),
    ):
        template_path = _safe_repo_file(templates.get(phase), f"{phase} template")
        template = _load(template_path, f"{phase} template")
        if (
            template.get("schema_version") != schema
            or template.get("execution_authorized") is not False
            or template.get("authorized_by") != ""
        ):
            _fail(f"{phase} authorization template is not inert")
    return {
        "schema_version": SCHEMA,
        "revision_tag": REVISION_TAG,
        "canonical_sha256": canonical_sha256(protocol),
        "base_spec_canonical_sha256": base_row["canonical_sha256"],
    }


def load_protocol(path: Path | str = DEFAULT_PROTOCOL) -> Mapping[str, Any]:
    value = _load(Path(path), "two-phase protocol")
    validate_protocol(value)
    return value


def load_bound_base_spec(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    relative = protocol["base_diagnostic_spec"]["package_relative_path"]
    return legacy.load_spec(_safe_repo_file(relative, "bound base spec"))


def validate_fixed_rng_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    arm_role: str,
    expected_output_path: str,
    native_receipt_sha256: str,
) -> dict[str, Any]:
    if len(receipts) != 4:
        _fail("exactly four fixed-RNG receipts are required")
    protocol_id = protocol_identity(protocol, protocol_path)
    projected = []
    for row in receipts:
        item = copy.deepcopy(dict(row))
        item["schema_version"] = legacy.RNG_SCHEMA
        projected.append(item)
    try:
        base_result = legacy.validate_rng_receipts(
            projected,
            arm_role=arm_role,
            expected_output_path=expected_output_path,
            native_receipt_sha256=native_receipt_sha256,
        )
    except legacy.E00LegacyDiagnosticError as error:
        raise E00TwoPhaseDiagnosticError(str(error)) from error

    ordered = sorted(receipts, key=lambda row: row.get("rank", -1))
    fixed_rows = []
    for expected_rank, row in enumerate(ordered):
        for path, expected in (
            ("schema_version", RNG_SCHEMA),
            ("revision_tag", REVISION_TAG),
            ("rank", expected_rank),
            ("protocol.revision_tag", protocol_id["revision_tag"]),
            ("protocol.file_sha256", protocol_id["file_sha256"]),
            ("protocol.canonical_sha256", protocol_id["canonical_sha256"]),
            ("fixed_initial_rng.enabled", True),
            (
                "fixed_initial_rng.scheme",
                "explicit_rank_owned_cpu_cuda_manual_seed_v2",
            ),
            (
                "fixed_initial_rng.scope",
                "inside_fork_rng_immediately_before_entire_legacy_inference_entrypoint",
            ),
            (
                "fixed_initial_rng.cpu_seed",
                EXPECTED_SEEDS[expected_rank]["cpu_seed"],
            ),
            (
                "fixed_initial_rng.cuda_seed",
                EXPECTED_SEEDS[expected_rank]["cuda_seed"],
            ),
        ):
            _eq(row, path, expected)
        seeded = _mapping(
            _get(row, "fixed_initial_rng.seeded_initial"), "seeded initial RNG"
        )
        terminal = _mapping(
            _get(row, "fixed_initial_rng.terminal_before_restore"),
            "terminal RNG",
        )
        for name in ("cpu_sha256", "cuda_sha256"):
            _hex(seeded.get(name), f"rank {expected_rank} initial {name}")
            _hex(terminal.get(name), f"rank {expected_rank} terminal {name}")
        fixed_rows.append(
            {
                "rank": expected_rank,
                "cpu_seed": EXPECTED_SEEDS[expected_rank]["cpu_seed"],
                "cuda_seed": EXPECTED_SEEDS[expected_rank]["cuda_seed"],
                "cpu_initial_sha256": seeded["cpu_sha256"],
                "cuda_initial_sha256": seeded["cuda_sha256"],
            }
        )
    return {
        **base_result,
        "explicit_initial_rng": True,
        "per_rank_fixed_initial_rng": fixed_rows,
        "per_rank_fixed_initial_rng_digest": canonical_sha256(fixed_rows),
    }


def build_arm_audit(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    native_receipt: Mapping[str, Any],
    native_receipt_sha256: str,
    rng_receipts: Sequence[Mapping[str, Any]],
    rng_receipt_sha256s: Sequence[str],
    arm_role: str,
    video: Path,
) -> dict[str, Any]:
    spec = load_bound_base_spec(protocol)
    try:
        native = legacy.validate_native_receipt(
            native_receipt, spec=spec, arm_role=arm_role, video=video
        )
    except legacy.E00LegacyDiagnosticError as error:
        raise E00TwoPhaseDiagnosticError(str(error)) from error
    output_path = _get(native_receipt, "output.path")
    if not isinstance(output_path, str) or not output_path:
        _fail("native output path is absent")
    rng = validate_fixed_rng_receipts(
        rng_receipts,
        protocol=protocol,
        protocol_path=protocol_path,
        arm_role=arm_role,
        expected_output_path=output_path,
        native_receipt_sha256=native_receipt_sha256,
    )
    if len(rng_receipt_sha256s) != 4:
        _fail("exactly four RNG receipt hashes are required")
    for index, digest in enumerate(rng_receipt_sha256s):
        _hex(digest, f"rank {index} RNG receipt SHA-256")
    arm = next(row for row in spec["arms"] if row["arm_role"] == arm_role)
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
        "offline_anchor_graph": False,
        "training_performed": False,
        "optimization_steps": 0,
        "artifacts": {
            "video_path": str(video),
            "video_sha256": file_sha256(video),
            "native_receipt_sha256": native_receipt_sha256,
            "rng_receipt_sha256s_rank_order": list(rng_receipt_sha256s),
        },
        "native": native,
        "rng_and_noise": rng,
    }


def _validate_arm_shape(audit: Mapping[str, Any], role: str) -> None:
    for path, expected in (
        ("schema_version", ARM_AUDIT_SCHEMA),
        ("revision_tag", REVISION_TAG),
        ("complete", True),
        ("arm_role", role),
        ("anchor_free", False),
        ("training_performed", False),
        ("optimization_steps", 0),
        ("rng_and_noise.explicit_initial_rng", True),
    ):
        _eq(audit, path, expected)
    _hex(_get(audit, "artifacts.video_sha256"), f"{role} video SHA-256")
    _hex(
        _get(audit, "rng_and_noise.per_rank_fixed_initial_rng_digest"),
        f"{role} fixed RNG digest",
    )


def _cross_arm_equal(audits: Sequence[Mapping[str, Any]], path: str) -> Any:
    values = [_get(audit, path) for audit in audits]
    if any(value != values[0] for value in values[1:]):
        _fail(f"cross-arm diagnostic differs at {path}")
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
    a_audit: Mapping[str, Any],
    b_audit: Mapping[str, Any],
    a_audit_path: Path,
    b_audit_path: Path,
    a_video: Path,
    b_video: Path,
) -> dict[str, Any]:
    _validate_arm_shape(a_audit, ARM_ROLES[0])
    _validate_arm_shape(b_audit, ARM_ROLES[1])
    audits = (a_audit, b_audit)
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
        _cross_arm_equal(audits, path)
    latent = _cross_arm_equal(audits, "rng_and_noise.predecode_latent_sha256")
    _plain_file(a_video, "A MP4")
    _plain_file(b_video, "B MP4")
    a_video_sha = file_sha256(a_video)
    b_video_sha = file_sha256(b_video)
    if (
        str(a_video) != _get(a_audit, "artifacts.video_path")
        or str(b_video) != _get(b_audit, "artifacts.video_path")
        or a_video_sha != _get(a_audit, "artifacts.video_sha256")
        or b_video_sha != _get(b_audit, "artifacts.video_sha256")
    ):
        _fail("A/B MP4 path or audit binding differs")
    if a_video_sha != b_video_sha:
        _fail("A/B MP4 bytes differ; C remains forbidden")
    if _get(a_audit, "native.native_output_sha256") != a_video_sha or _get(
        b_audit, "native.native_output_sha256"
    ) != b_video_sha:
        _fail("A/B native output digest does not bind MP4 bytes")
    value = {
        "schema_version": AB_GATE_SCHEMA,
        "revision_tag": REVISION_TAG,
        "complete": True,
        "c_execution_gate_passed": True,
        "a_arm_audit": {
            "path": str(a_audit_path),
            "sha256": file_sha256(a_audit_path),
        },
        "b_arm_audit": {
            "path": str(b_audit_path),
            "sha256": file_sha256(b_audit_path),
        },
        "a_video": {"path": str(a_video), "sha256": a_video_sha},
        "b_video": {"path": str(b_video), "sha256": b_video_sha},
        "predecode_latent_raw_storage_sha256": latent,
        "mp4_sha256": a_video_sha,
        "per_rank_fixed_initial_rng_digest": _get(
            a_audit, "rng_and_noise.per_rank_fixed_initial_rng_digest"
        ),
        "raw_keyed_noise_bank_sha256": _get(
            a_audit, "rng_and_noise.raw_noise_bank_sha256"
        ),
        "outer_schedule_digest": _get(a_audit, "native.outer_schedule_digest"),
        "frozen_certificate_sha256": _get(
            a_audit, "native.frozen_certificate_sha256"
        ),
        "only_admitted_next_arm": ARM_ROLES[2],
        "training_performed": False,
    }
    return _seal(value, "gate_digest")


def validate_existing_ab_gate(
    gate: Mapping[str, Any],
    *,
    a_audit: Mapping[str, Any],
    b_audit: Mapping[str, Any],
    a_audit_path: Path,
    b_audit_path: Path,
    a_video: Path,
    b_video: Path,
) -> None:
    _verify_seal(gate, "gate_digest", "A/B gate")
    expected = build_ab_gate(
        a_audit=a_audit,
        b_audit=b_audit,
        a_audit_path=a_audit_path,
        b_audit_path=b_audit_path,
        a_video=a_video,
        b_video=b_video,
    )
    if gate != expected:
        _fail("stored A/B gate differs from a fresh byte-level recomputation")


def build_abc_final(
    *,
    a_audit: Mapping[str, Any],
    b_audit: Mapping[str, Any],
    c_audit: Mapping[str, Any],
    a_audit_path: Path,
    b_audit_path: Path,
    c_audit_path: Path,
    ab_gate: Mapping[str, Any],
    ab_gate_path: Path,
    a_video: Path,
    b_video: Path,
    c_video: Path,
) -> dict[str, Any]:
    validate_existing_ab_gate(
        ab_gate,
        a_audit=a_audit,
        b_audit=b_audit,
        a_audit_path=a_audit_path,
        b_audit_path=b_audit_path,
        a_video=a_video,
        b_video=b_video,
    )
    _validate_arm_shape(c_audit, ARM_ROLES[2])
    audits = (a_audit, b_audit, c_audit)
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
        _cross_arm_equal(audits, path)
    if [_get(audit, "native.target_route_replay_steps") for audit in audits] != [
        0,
        0,
        40,
    ]:
        _fail("A/B/C target-route intervention order differs")
    _plain_file(c_video, "C MP4")
    if (
        str(c_video) != _get(c_audit, "artifacts.video_path")
        or file_sha256(c_video) != _get(c_audit, "artifacts.video_sha256")
        or file_sha256(c_video) != _get(c_audit, "native.native_output_sha256")
    ):
        _fail("C MP4 bytes do not match the C audit")
    value = {
        "schema_version": ABC_FINAL_SCHEMA,
        "revision_tag": REVISION_TAG,
        "complete": True,
        "diagnostic_only": True,
        "training_performed": False,
        "arm_order": list(ARM_ROLES),
        "ab_gate": {"path": str(ab_gate_path), "sha256": file_sha256(ab_gate_path)},
        "arm_audits": [
            {"role": role, "path": str(path), "sha256": file_sha256(path)}
            for role, path in zip(ARM_ROLES, (a_audit_path, b_audit_path, c_audit_path))
        ],
        "videos": [
            {"role": role, "path": str(path), "sha256": file_sha256(path)}
            for role, path in zip(ARM_ROLES, (a_video, b_video, c_video))
        ],
        "same_fixed_initial_rng_all_arms": True,
        "same_raw_keyed_noise_all_arms": True,
        "same_outer_schedule_all_arms": True,
        "same_frozen_model_all_arms": True,
        "ab_predecode_latent_bit_exact": True,
        "ab_mp4_bit_exact": True,
        "only_causal_claim": (
            "B/C isolate the old pure-QK route only after A/B prove the "
            "observer-matched route-off arm is byte-identical to pure no-observer A"
        ),
    }
    return _seal(value, "final_audit_digest")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        _fail(f"temporary audit already exists: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
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
    ab = sub.add_parser("ab-gate")
    ab.add_argument("--a-audit", required=True)
    ab.add_argument("--b-audit", required=True)
    ab.add_argument("--a-video", required=True)
    ab.add_argument("--b-video", required=True)
    ab.add_argument("--gate-output", required=True)
    final = sub.add_parser("abc-final")
    final.add_argument("--a-audit", required=True)
    final.add_argument("--b-audit", required=True)
    final.add_argument("--c-audit", required=True)
    final.add_argument("--ab-gate", required=True)
    final.add_argument("--a-video", required=True)
    final.add_argument("--b-video", required=True)
    final.add_argument("--c-video", required=True)
    final.add_argument("--audit-output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "protocol":
        path = Path(args.protocol)
        protocol = load_protocol(path)
        value = {**validate_protocol(protocol), **protocol_identity(protocol, path)}
    elif args.command == "arm":
        protocol_path = Path(args.protocol)
        protocol = load_protocol(protocol_path)
        native_path = Path(args.native_receipt)
        rng_paths = [Path(path) for path in args.rng_receipt]
        video = Path(args.video)
        value = build_arm_audit(
            protocol=protocol,
            protocol_path=protocol_path,
            native_receipt=_load(native_path, "native receipt"),
            native_receipt_sha256=file_sha256(native_path),
            rng_receipts=[_load(path, "fixed RNG receipt") for path in rng_paths],
            rng_receipt_sha256s=[file_sha256(path) for path in rng_paths],
            arm_role=args.arm_role,
            video=video,
        )
        legacy._probe_video(video)
        _write_new_json(Path(args.audit_output), value)
    elif args.command == "ab-gate":
        a_path = Path(args.a_audit)
        b_path = Path(args.b_audit)
        value = build_ab_gate(
            a_audit=_load(a_path, "A arm audit"),
            b_audit=_load(b_path, "B arm audit"),
            a_audit_path=a_path,
            b_audit_path=b_path,
            a_video=Path(args.a_video),
            b_video=Path(args.b_video),
        )
        _write_new_json(Path(args.gate_output), value)
    else:
        a_path = Path(args.a_audit)
        b_path = Path(args.b_audit)
        c_path = Path(args.c_audit)
        gate_path = Path(args.ab_gate)
        value = build_abc_final(
            a_audit=_load(a_path, "A arm audit"),
            b_audit=_load(b_path, "B arm audit"),
            c_audit=_load(c_path, "C arm audit"),
            a_audit_path=a_path,
            b_audit_path=b_path,
            c_audit_path=c_path,
            ab_gate=_load(gate_path, "A/B gate"),
            ab_gate_path=gate_path,
            a_video=Path(args.a_video),
            b_video=Path(args.b_video),
            c_video=Path(args.c_video),
        )
        _write_new_json(Path(args.audit_output), value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""R6 old-QK leakage diagnostic protocol, capability, gate, and final validator."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Iterator, Mapping, Optional, Sequence

import validate_e00_three_vessel_clean_diag_r5 as r5


SCHEMA = "bernini-e00-clean-diagnostic-r6-protocol-v6"
REVISION_TAG = "E00_DFIX2_CLEAN_DIAG_R6_EXTERNAL_BOOTSTRAP_20260821"
RNG_SCHEMA = "bernini-e00-legacy-fixed-rng-r6-receipt-v6"
ARM_AUDIT_SCHEMA = "bernini-e00-clean-diagnostic-r6-arm-audit-v6"
AB_GATE_SCHEMA = "bernini-e00-clean-diagnostic-r6-ab-current-bit-exact-gate-v6"
FINAL_SCHEMA = "bernini-e00-clean-diagnostic-r6-final-current-artifact-audit-v6"
CAPABILITY_SCHEMA = "bernini-e00-clean-diagnostic-r6-bridge-capability-v6"
PHASE_A_MARKER_SCHEMA = "bernini-e00-clean-diagnostic-r6-phase-a-stopped-marker-v6"
METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
DEFAULT_PROTOCOL = METHOD_ROOT / "assets/e00_three_vessel_clean_diag_r6_protocol_20260821.json"
ARM_ROLES = r5.ARM_ROLES
EXPECTED_SEEDS = copy.deepcopy(r5.EXPECTED_SEEDS)
REQUIRED_EQUALITIES = list(r5.REQUIRED_EQUALITIES)
REQUIRED_CURRENT_BINDINGS = list(r5.REQUIRED_CURRENT_BINDINGS)
HEX = set("0123456789abcdef")


class E00R6Error(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise E00R6Error(message)


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
        raise E00R6Error(f"{label} is unreadable") from error


def _repo_file(relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        _fail(f"{label} path is unsafe")
    path = REPO_ROOT / relative
    _plain_file(path, label)
    return path


def protocol_identity(protocol: Mapping[str, Any], path: Path) -> dict[str, Any]:
    _plain_file(path, "R6 protocol")
    return {"revision_tag": REVISION_TAG, "path": str(path), "file_sha256": file_sha256(path), "canonical_sha256": canonical_sha256(protocol)}


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    pre_route = [
        "per_rank_initial_cpu_cuda_rng", "raw_keyed_noise_and_sampler_initial_latent",
        "outer_schedule", "effective_native_caption_hash_map", "source_video_sha256",
        "self_generated_anchor_video_sha256", "anchor_observer_configuration_and_schedule",
    ]
    final_retain = [
        "three_object_editing_instruction_utf8", "three_object_editing_instruction_utf8_sha256",
        "source_video_path_and_sha256", "self_generated_anchor_video_path_and_sha256",
    ]
    for path, expected in (
        ("schema_version", SCHEMA), ("revision_tag", REVISION_TAG),
        ("status.draft_only", True), ("status.execution_authorized", False),
        ("status.gpu_run_started", False), ("status.training_performed", False),
        ("status.independent_package_audit_passed", False),
        ("diagnostic_scope.old_qk_route_diagnostic_only", True),
        ("diagnostic_scope.property_preservation_fix_claimed", False),
        ("diagnostic_scope.method_promotion_allowed", False),
        ("diagnostic_scope.a_b_bit_exact_or_stop_without_c", True),
        ("diagnostic_scope.b_c_only_intended_difference", "final_pure_qk_route_application"),
        ("diagnostic_scope.b_c_pre_route_byte_exact_inputs", pre_route),
        ("diagnostic_scope.final_audit_must_retain", final_retain),
        ("external_bootstrap.stdlib_only_no_local_imports", True),
        ("external_bootstrap.root_one_way_pins_all_runtime_consumers_and_dependencies", True),
        ("external_bootstrap.consumers_pin_root", False),
        ("external_bootstrap.reject_cache_bytecode_before_consumer", True),
        ("external_bootstrap.consumer_exec_from_verified_open_fd", True),
        ("external_bootstrap.final_launch_template_checks_verifier_sha256_before_execution", True),
        ("phase_a_placeholder_contract.classification", "content_static_lossy_decode_reencode_placeholder"),
        ("phase_a_placeholder_contract.decoded_bit_exact_source_video_claimed", False),
        ("phase_a_placeholder_contract.encoded_bytes_equal_source_video_claimed", False),
        ("fixed_initial_rng.scheme", "explicit_rank_owned_cpu_cuda_manual_seed_r3"),
        ("fixed_initial_rng.per_rank", EXPECTED_SEEDS), ("arm_order", list(ARM_ROLES)),
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
        ("bridge_capability_contract.direct_bridge_without_capability_forbidden", True),
        ("package_integrity_contract.root_and_all_pins_required_before_local_import", True),
        ("package_integrity_contract.builder_legacy_helper_sha_before_lazy_import", True),
        ("package_integrity_contract.package_pycache_directories_forbidden", True),
        ("package_integrity_contract.package_pyc_files_forbidden", True),
        ("package_integrity_contract.optimized_python_must_preserve_all_integrity_failures", True),
        ("package_integrity_contract.terminal_markers_require_exclusive_temp_fsync_atomic_rename_reread", True),
        ("slurm_contract.parent_job_id", "143808"), ("slurm_contract.compute_node", "auh7-1b-gpu-292"),
        ("slurm_contract.gpu_count", 4), ("slurm_contract.preserve_slurm_visible_device_namespace", True),
        ("slurm_contract.submit_new_job", False), ("slurm_contract.cancel_parent_job", False),
        ("claims.diagnostic_only", True), ("claims.zero_update_frozen_inference", True),
        ("claims.training_authorized", False), ("claims.promotion_forbidden", True),
    ):
        _eq(protocol, path, expected)
    expected_states = [
        "PACKAGE_REVIEW_REQUIRED", "EXTERNAL_A_ONLY_AUTHORIZATION_AND_TOKEN", "RUN_A",
        "A_STOPPED_REVIEW_REQUIRED", "EXTERNAL_BC_AUTHORIZATION_AND_TOKEN_BOUND_TO_A_BYTES",
        "RUN_B", "FRESH_AB_CURRENT_ARTIFACT_AND_BIT_EXACT_GATE",
        "RUN_C_ONLY_AFTER_BRIDGE_REVALIDATES_CURRENT_GATE_MARKER_AUTH",
        "FRESH_ABC_FINAL_CURRENT_ARTIFACT_CLOSURE",
    ]
    if protocol.get("two_phase_state_machine") != expected_states:
        _fail("R6 phase state machine differs")
    base = _mapping(protocol.get("base_diagnostic_spec"), "base spec")
    base_path = _repo_file(base.get("package_relative_path"), "base diagnostic spec")
    if file_sha256(base_path) != base.get("file_sha256"):
        _fail("R6 base diagnostic spec bytes differ")
    base_value = r5.r4.r3.legacy.load_spec(base_path)
    if canonical_sha256(base_value) != base.get("canonical_sha256"):
        _fail("R6 base diagnostic spec canonical bytes differ")
    templates = _mapping(protocol.get("authorization_templates"), "authorization templates")
    for phase, schema in (
        ("phase_a", "bernini-e00-clean-diagnostic-r6-phase-a-authorization-v6"),
        ("phase_bc", "bernini-e00-clean-diagnostic-r6-phase-bc-authorization-v6"),
    ):
        template = _load(_repo_file(templates.get(phase), f"{phase} template"), f"{phase} template")
        if template.get("schema_version") != schema or template.get("execution_authorized") is not False or template.get("authorized_by") != "":
            _fail(f"R6 {phase} authorization template is not inert")
    return {"schema_version": SCHEMA, "revision_tag": REVISION_TAG, "canonical_sha256": canonical_sha256(protocol)}


def load_protocol(path: Path | str = DEFAULT_PROTOCOL) -> Mapping[str, Any]:
    protocol = _load(Path(path), "R6 protocol")
    validate_protocol(protocol)
    return protocol


@contextmanager
def _r6_semantic_core() -> Iterator[None]:
    replacements = {
        "REVISION_TAG": REVISION_TAG, "RNG_SCHEMA": RNG_SCHEMA,
        "ARM_AUDIT_SCHEMA": ARM_AUDIT_SCHEMA, "AB_GATE_SCHEMA": AB_GATE_SCHEMA,
        "FINAL_SCHEMA": FINAL_SCHEMA, "PHASE_A_MARKER_SCHEMA": PHASE_A_MARKER_SCHEMA,
        "EXPECTED_SEEDS": EXPECTED_SEEDS, "REQUIRED_EQUALITIES": REQUIRED_EQUALITIES,
        "REQUIRED_CURRENT_BINDINGS": REQUIRED_CURRENT_BINDINGS,
    }
    before = {name: getattr(r5, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(r5, name, value)
        yield
    finally:
        for name, value in before.items():
            setattr(r5, name, value)


def build_arm_audit(**kwargs: Any) -> dict[str, Any]:
    with _r6_semantic_core():
        return r5.build_arm_audit(**kwargs)


def revalidate_arm_current(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    with _r6_semantic_core():
        return r5.revalidate_arm_current(*args, **kwargs)


def validate_current_artifact_bytes(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    return r5.validate_current_artifact_bytes(artifacts)


def build_ab_gate(**kwargs: Any) -> dict[str, Any]:
    with _r6_semantic_core():
        return r5.build_ab_gate(**kwargs)


def verify_ab_gate_current(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    with _r6_semantic_core():
        return r5.verify_ab_gate_current(*args, **kwargs)


def _bc_input_binding(audit: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = _mapping(native.get("mechanism"), "native mechanism")
    trace = _mapping(mechanism.get("trace"), "native trace")
    cache = _mapping(trace.get("attention_cache"), "attention cache")
    rng_artifacts = _get(audit, "artifacts.rng_receipts_rank_order")
    if not isinstance(rng_artifacts, list) or len(rng_artifacts) != 4:
        _fail("R6 B/C RNG receipt path closure differs")
    initial_latents = []
    for rank, artifact in enumerate(rng_artifacts):
        artifact = _mapping(artifact, f"rank {rank} RNG artifact")
        if artifact.get("rank") != rank:
            _fail("R6 B/C RNG receipt rank order differs")
        receipt = _load(Path(artifact.get("path")), f"rank {rank} current RNG receipt")
        latent = _mapping(receipt.get("sampler_initial_latent"), "sampler initial latent")
        if latent.get("definition") != "edit_packed_is_initialized_from_source_latent_before_step0_keyed_noise":
            _fail("R6 sampler initial latent definition differs")
        identity = _mapping(latent.get("source_owned_initial_edit_latent"), "sampler initial latent identity")
        _hex(identity.get("raw_storage_sha256"), "sampler initial latent raw bytes")
        initial_latents.append(copy.deepcopy(latent))
    if any(value != initial_latents[0] for value in initial_latents[1:]):
        _fail("R6 per-rank sampler initial latent bytes differ")
    return {
        "per_rank_initial_rng": copy.deepcopy(_get(audit, "rng_and_noise.per_rank_fixed_initial_rng")),
        "raw_keyed_noise_bank_sha256": _get(audit, "rng_and_noise.raw_noise_bank_sha256"),
        "sampler_initial_latent": initial_latents[0],
        "outer_schedule_digest": _get(audit, "native.outer_schedule_digest"),
        "effective_native_caption_hash_map": copy.deepcopy(_mapping(native.get("prompts"), "native prompts")),
        "source_video_sha256": _get(native, "source.sha256"),
        "self_generated_anchor_video_sha256": _get(native, "pure_t2v_anchor.sha256"),
        "anchor_observer_configuration_and_schedule": {
            "configuration": {
                key: copy.deepcopy(mechanism.get(key))
                for key in (
                    "transport", "transport_strength", "transport_steps", "field_guidance",
                    "field_model", "anchor_state_mode", "anchor_cfg_scope",
                    "anchor_contrast_mode", "anchor_sigma_cap", "anchor_candidate_mode",
                    "anchor_bank_size", "selected_blocks",
                )
            },
            "active_schedule": copy.deepcopy(trace.get("anchor_active_schedule")),
            "anchor_model_forwards": trace.get("anchor_model_forwards"),
            "anchor_candidate_cells": trace.get("anchor_candidate_cells"),
            "attention_cache": {
                "capture_count": cache.get("capture_count"),
                "qk_only_capture_count": cache.get("qk_only_capture_count"),
                "selected_block_indices": cache.get("selected_block_indices"),
            },
            "anchor_donor_cached_fields": trace.get("anchor_donor_cached_fields"),
        },
    }


def build_final_audit(**kwargs: Any) -> dict[str, Any]:
    with _r6_semantic_core():
        value = r5.build_final_audit(**kwargs)
    protocol = kwargs["protocol"]
    protocol_path = kwargs["protocol_path"]
    audit_paths = [kwargs["a_audit_path"], kwargs["b_audit_path"], kwargs["c_audit_path"]]
    audits = [
        revalidate_arm_current(path, protocol=protocol, protocol_path=protocol_path, expected_role=role)
        for path, role in zip(audit_paths, ARM_ROLES)
    ]
    native_receipts = [
        _load(Path(_get(audit, "artifacts.native_receipt.path")), f"{role} native receipt")
        for audit, role in zip(audits, ARM_ROLES)
    ]
    b_binding = _bc_input_binding(audits[1], native_receipts[1])
    c_binding = _bc_input_binding(audits[2], native_receipts[2])
    if b_binding != c_binding:
        _fail("R6 B/C pre-route noise/latent/schedule/caption/anchor observer bytes differ")
    spec_path = _repo_file(_get(protocol, "base_diagnostic_spec.package_relative_path"), "R6 base spec")
    spec = r5.r4.r3.legacy.load_spec(spec_path)
    prompt = _mapping(spec.get("data_and_prompt_contract"), "R6 prompt contract")
    retained = {
        "three_object_editing_instruction_utf8": prompt["editing_instruction"],
        "three_object_editing_instruction_utf8_sha256": prompt["editing_instruction_utf8_sha256"],
        "source_video": {"path": prompt["source_video"], "sha256": prompt["source_video_sha256"]},
        "self_generated_anchor_video": {"path": prompt["anchor_video"], "sha256": prompt["anchor_video_sha256"]},
    }
    if hashlib.sha256(retained["three_object_editing_instruction_utf8"].encode("utf-8")).hexdigest() != retained["three_object_editing_instruction_utf8_sha256"]:
        _fail("R6 retained three-object instruction hash differs")
    value = dict(value)
    value.pop("final_audit_digest", None)
    value.update({
        "old_qk_route_white_leakage_diagnostic_only": True,
        "property_preservation_fix_claimed": False,
        "a_b_bit_exact_or_c_forbidden": True,
        "b_c_only_intended_difference": "final_pure_qk_route_application",
        "b_c_pre_route_byte_exact_binding": b_binding,
        "b_c_pre_route_byte_exact_binding_sha256": canonical_sha256(b_binding),
        "retained_three_object_instruction_and_source_anchor": retained,
    })
    value["final_audit_digest"] = canonical_sha256(value)
    return value


def _authorization_common(
    authorization: Mapping[str, Any], *, protocol: Mapping[str, Any], protocol_path: Path,
    package_manifest_path: Path, capability_token_sha256: str,
) -> None:
    protocol_id = protocol_identity(protocol, protocol_path)
    _hex(capability_token_sha256, "capability token SHA-256")
    _plain_file(package_manifest_path, "R6 package manifest")
    for path, expected in (
        ("execution_authorized", True), ("package_manifest_sha256", file_sha256(package_manifest_path)),
        ("protocol_file_sha256", protocol_id["file_sha256"]),
        ("protocol_canonical_sha256", protocol_id["canonical_sha256"]),
        ("bridge_capability_token_sha256", capability_token_sha256),
        ("parent_job_id", "143808"), ("compute_node", "auh7-1b-gpu-292"),
        ("sp4_observer_released_node292", True),
    ):
        _eq(authorization, path, expected)
    if not isinstance(authorization.get("authorized_by"), str) or not authorization["authorized_by"]:
        _fail("R6 authorization reviewer identity is absent")


def validate_phase_a_marker_current(
    marker_path: Path, *, protocol: Mapping[str, Any], protocol_path: Path, package_manifest_path: Path,
) -> Mapping[str, Any]:
    marker = _load(marker_path, "R6 phase A marker")
    protocol_id = protocol_identity(protocol, protocol_path)
    for path, expected in (
        ("schema_version", PHASE_A_MARKER_SCHEMA), ("revision_tag", REVISION_TAG),
        ("state", "A_STOPPED_REVIEW_REQUIRED"), ("complete", True),
        ("training_performed", False), ("optimization_steps", 0),
        ("parent_job_id", "143808"), ("compute_node", "auh7-1b-gpu-292"),
        ("package_manifest_sha256", file_sha256(package_manifest_path)),
        ("protocol_file_sha256", protocol_id["file_sha256"]),
        ("protocol_canonical_sha256", protocol_id["canonical_sha256"]),
        ("observed_arm_order", [ARM_ROLES[0]]), ("must_stop_after_a", True),
        ("phase_bc_execution_authorized", False),
        ("old_qk_route_white_leakage_diagnostic_only", True),
        ("property_preservation_fix_claimed", False),
        ("terminal_marker_atomic_write_verified", True),
    ):
        _eq(marker, path, expected)
    auth_path = Path(_get(marker, "phase_a_authorization.path"))
    audit_path = Path(_get(marker, "phase_a_arm_audit.path"))
    video_path = Path(_get(marker, "phase_a_video.path"))
    for current, expected_sha, label in (
        (auth_path, _get(marker, "phase_a_authorization.sha256"), "phase A authorization"),
        (audit_path, _get(marker, "phase_a_arm_audit.sha256"), "phase A audit"),
        (video_path, _get(marker, "phase_a_video.sha256"), "phase A video"),
    ):
        _plain_file(current, label)
        if file_sha256(current) != expected_sha:
            _fail(f"current {label} bytes differ")
    token_sha = _hex(marker.get("phase_a_capability_token_sha256"), "phase A token SHA-256")
    phase_a_auth = _load(auth_path, "phase A authorization")
    if phase_a_auth.get("schema_version") != "bernini-e00-clean-diagnostic-r6-phase-a-authorization-v6" or phase_a_auth.get("bridge_capability_token_sha256") != token_sha:
        _fail("phase A authorization/token binding differs")
    audit = revalidate_arm_current(audit_path, protocol=protocol, protocol_path=protocol_path, expected_role=ARM_ROLES[0])
    if _get(audit, "artifacts.video.path") != str(video_path) or _get(audit, "artifacts.video.sha256") != file_sha256(video_path):
        _fail("phase A video/audit binding differs")
    return marker


def validate_bridge_capability(
    *, phase: str, arm_role: str, protocol: Mapping[str, Any], protocol_path: Path,
    package_manifest_path: Path, authorization_path: Path, capability_token_sha256: str,
    phase_a_marker_path: Optional[Path] = None, ab_gate_path: Optional[Path] = None,
) -> dict[str, Any]:
    expected_role = {"A": ARM_ROLES[0], "B": ARM_ROLES[1], "C": ARM_ROLES[2]}.get(phase)
    if expected_role is None or arm_role != expected_role:
        _fail("R6 phase does not admit the requested arm")
    authorization = _load(authorization_path, f"phase {phase} authorization")
    _authorization_common(
        authorization, protocol=protocol, protocol_path=protocol_path,
        package_manifest_path=package_manifest_path, capability_token_sha256=capability_token_sha256,
    )
    if phase == "A":
        for path, expected in (
            ("schema_version", "bernini-e00-clean-diagnostic-r6-phase-a-authorization-v6"),
            ("authorized_phase", "A_ONLY_THEN_STOP"), ("only_authorized_arm", ARM_ROLES[0]),
            ("must_stop_after_a", True), ("bc_execution_authorized", False),
        ):
            _eq(authorization, path, expected)
        if phase_a_marker_path is not None or ab_gate_path is not None:
            _fail("R6 phase A does not accept later-phase artifacts")
    else:
        for path, expected in (
            ("schema_version", "bernini-e00-clean-diagnostic-r6-phase-bc-authorization-v6"),
            ("authorized_phase", "B_THEN_CURRENT_AB_GATE_THEN_C"),
            ("authorized_arm_order", [ARM_ROLES[1], ARM_ROLES[2]]),
            ("c_requires_bridge_revalidated_current_ab_gate", True),
            ("stop_without_c_on_gate_failure", True),
        ):
            _eq(authorization, path, expected)
        if phase_a_marker_path is None:
            _fail("R6 phase B/C requires the current phase A marker")
        marker = validate_phase_a_marker_current(
            phase_a_marker_path, protocol=protocol, protocol_path=protocol_path,
            package_manifest_path=package_manifest_path,
        )
        if file_sha256(phase_a_marker_path) != authorization.get("phase_a_stopped_marker_sha256"):
            _fail("R6 phase A marker differs from BC authorization")
        if _get(marker, "phase_a_arm_audit.sha256") != authorization.get("phase_a_arm_audit_sha256") or _get(marker, "phase_a_video.sha256") != authorization.get("phase_a_mp4_sha256"):
            _fail("R6 phase A audit/video differs from BC authorization")
        if capability_token_sha256 == marker.get("phase_a_capability_token_sha256"):
            _fail("R6 A and BC capability tokens must differ")
        if phase == "B":
            if ab_gate_path is not None:
                _fail("R6 phase B does not consume a C gate")
        else:
            if ab_gate_path is None:
                _fail("R6 phase C requires the current A/B gate")
            gate = verify_ab_gate_current(ab_gate_path, protocol=protocol, protocol_path=protocol_path)
            if gate.get("c_execution_gate_passed") is not True or gate.get("only_admitted_next_arm") != ARM_ROLES[2]:
                _fail("R6 current A/B gate does not admit C")
    value = {
        "schema_version": CAPABILITY_SCHEMA, "revision_tag": REVISION_TAG, "complete": True,
        "phase": phase, "arm_role": arm_role,
        "authorization": {"path": str(authorization_path), "sha256": file_sha256(authorization_path)},
        "package_manifest": {"path": str(package_manifest_path), "sha256": file_sha256(package_manifest_path)},
        "capability_token_sha256": capability_token_sha256,
        "phase_a_marker": ({"path": str(phase_a_marker_path), "sha256": file_sha256(phase_a_marker_path)} if phase_a_marker_path else None),
        "ab_gate": ({"path": str(ab_gate_path), "sha256": file_sha256(ab_gate_path)} if ab_gate_path else None),
        "only_this_arm_admitted": True, "training_performed": False,
        "old_qk_route_white_leakage_diagnostic_only": True,
        "property_preservation_fix_claimed": False,
    }
    value["capability_digest"] = canonical_sha256(value)
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    r5._write_new_json(path, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol"); protocol.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    arm = sub.add_parser("arm")
    arm.add_argument("--protocol", default=str(DEFAULT_PROTOCOL)); arm.add_argument("--arm-role", required=True, choices=ARM_ROLES)
    arm.add_argument("--native-receipt", required=True); arm.add_argument("--rng-receipt", action="append", required=True)
    arm.add_argument("--video", required=True); arm.add_argument("--audit-output", required=True)
    gate = sub.add_parser("ab-gate")
    gate.add_argument("--protocol", default=str(DEFAULT_PROTOCOL)); gate.add_argument("--a-audit", required=True)
    gate.add_argument("--b-audit", required=True); gate.add_argument("--gate-output", required=True)
    check = sub.add_parser("verify-ab-gate")
    check.add_argument("--protocol", default=str(DEFAULT_PROTOCOL)); check.add_argument("--ab-gate", required=True)
    capability = sub.add_parser("bridge-capability")
    capability.add_argument("--protocol", default=str(DEFAULT_PROTOCOL)); capability.add_argument("--phase", required=True, choices=("A", "B", "C"))
    capability.add_argument("--arm-role", required=True, choices=ARM_ROLES); capability.add_argument("--package-manifest", required=True)
    capability.add_argument("--authorization", required=True); capability.add_argument("--capability-token-sha256", required=True)
    capability.add_argument("--phase-a-marker"); capability.add_argument("--ab-gate")
    final = sub.add_parser("abc-final")
    final.add_argument("--protocol", default=str(DEFAULT_PROTOCOL)); final.add_argument("--a-audit", required=True)
    final.add_argument("--b-audit", required=True); final.add_argument("--c-audit", required=True)
    final.add_argument("--ab-gate", required=True); final.add_argument("--audit-output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = Path(args.protocol); protocol = load_protocol(protocol_path)
    if args.command == "protocol":
        value = {**validate_protocol(protocol), **protocol_identity(protocol, protocol_path)}
    elif args.command == "arm":
        value = build_arm_audit(
            protocol=protocol, protocol_path=protocol_path, arm_role=args.arm_role,
            video_path=Path(args.video), native_receipt_path=Path(args.native_receipt),
            rng_receipt_paths=[Path(path) for path in args.rng_receipt],
        )
        r5.r4.r3.legacy._probe_video(Path(args.video)); _write_new_json(Path(args.audit_output), value)
    elif args.command == "ab-gate":
        value = build_ab_gate(
            protocol=protocol, protocol_path=protocol_path,
            a_audit_path=Path(args.a_audit), b_audit_path=Path(args.b_audit),
        ); _write_new_json(Path(args.gate_output), value)
    elif args.command == "verify-ab-gate":
        value = verify_ab_gate_current(Path(args.ab_gate), protocol=protocol, protocol_path=protocol_path)
    elif args.command == "bridge-capability":
        value = validate_bridge_capability(
            phase=args.phase, arm_role=args.arm_role, protocol=protocol, protocol_path=protocol_path,
            package_manifest_path=Path(args.package_manifest), authorization_path=Path(args.authorization),
            capability_token_sha256=args.capability_token_sha256,
            phase_a_marker_path=Path(args.phase_a_marker) if args.phase_a_marker else None,
            ab_gate_path=Path(args.ab_gate) if args.ab_gate else None,
        )
    else:
        value = build_final_audit(
            protocol=protocol, protocol_path=protocol_path,
            a_audit_path=Path(args.a_audit), b_audit_path=Path(args.b_audit),
            c_audit_path=Path(args.c_audit), ab_gate_path=Path(args.ab_gate),
        ); _write_new_json(Path(args.audit_output), value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

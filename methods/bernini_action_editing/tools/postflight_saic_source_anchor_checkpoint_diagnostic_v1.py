#!/usr/bin/env python3
"""Externally admit one terminal source-anchor diagnostic as diagnostic only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence


SUBMISSION_SCHEMA = "saic-source-anchor-diagnostic-submission-v2"
RUNTIME_SCHEMA = "bernini-saic-source-anchor-checkpoint-diagnostic-v1"
DIAGNOSTIC_SCHEMA = "bernini-saic-source-anchor-checkpoint-decoded-diagnostics-v1"
POSTFLIGHT_SCHEMA = "saic-source-anchor-diagnostic-postflight-v2"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
JOB_ID = re.compile(r"[1-9][0-9]*\Z")
CELL_ORDER = (
    "base_correct_noop",
    "anchor_correct_noop",
    "anchor_wrong_noop",
    "anchor_route_drop_noop",
    "anchor_zero_condition_noop",
    "base_correct_action",
    "anchor_correct_action",
)
EXPECTED_NAMES = frozenset(
    {
        "shared.official-initial-gaussian.safetensors",
        "correct-source.normalized-clean-latent.safetensors",
        "wrong-source.normalized-clean-latent.safetensors",
        "decoded-diagnostics.json",
        "receipt.json",
    }
    | {f"{cell}.mp4" for cell in CELL_ORDER}
    | {f"{cell}.normalized-clean-latent.safetensors" for cell in CELL_ORDER}
)
SUBMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "submission_success",
        "job_success",
        "submitted_job",
        "request",
        "single_attempt_boundary",
        "inputs",
        "exports",
        "outputs",
        "authority",
        "receipt_digest",
    }
)
CHECKPOINT_RELEASE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "complete",
        "release_path",
        "formal_output_namespace",
        "job_id",
        "source_release_manifest_sha256",
        "submission_receipt_sha256",
        "postflight_source_sha256",
        "trainer_source_sha256",
        "history_digest",
        "heldout_gate",
        "authority",
        "artifacts",
        "payload_files_digest",
        "receipt_digest",
    }
)
CHECKPOINT_RELEASE_AUTHORITY = {
    "stage_a_checkpoint_release": True,
    "stage_b": False,
    "semantic_action": False,
    "identity": False,
    "candidate_selection": False,
    "production": False,
}
RUNTIME_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "classification",
        "complete",
        "first_real_runtime_status",
        "execution",
        "cell_order",
        "input",
        "stage_a",
        "conditions",
        "prompts",
        "sampling",
        "runtime_by_cell",
        "cell_inputs",
        "generated_identities",
        "route_drop_byte_identical_to_base_correct_noop",
        "outputs",
        "artifacts",
        "checkpoint",
        "vendor_restore",
        "source_revisions",
        "runtime_versions",
        "authority",
        "receipt_digest",
    }
)
DIAGNOSTIC_FIELDS = frozenset(
    {
        "schema_version",
        "cell_order",
        "correct_source",
        "wrong_source",
        "visual_model",
        "source_visual_evidence",
        "cells",
        "comparisons",
        "availability",
        "authority",
        "diagnostic_digest",
    }
)
RUNTIME_AUTHORITY = {
    "stage_a_decoded_runtime_qualified": False,
    "stage_b_runtime_available": False,
    "semantic_action_nonregression_available": False,
    "identity_authority": False,
    "training_allowed": False,
    "optimizer_step_allowed": False,
    "checkpoint_allowed": False,
    "selection_allowed": False,
    "publication_allowed": False,
    "production_allowed": False,
    "scientific_success_claimed": False,
}
DIAGNOSTIC_AUTHORITY = {
    "measurement_runtime_qualified_for_scientific_gate": False,
    "identity_authority": False,
    "semantic_action_authority": False,
    "candidate_selection_allowed": False,
    "training_allowed": False,
    "optimizer_step_allowed": False,
    "stage_b_allowed": False,
    "checkpoint_allowed": False,
    "publication_allowed": False,
    "production_allowed": False,
    "scientific_success_claimed": False,
}
POSTFLIGHT_AUTHORITY = {
    "operational_completion": True,
    "diagnostic_runtime_completed": True,
    "stage_a_decoded_runtime_qualified": False,
    "stage_b_runtime_available": False,
    "training": False,
    "optimizer_step": False,
    "checkpoint": False,
    "candidate_selection": False,
    "identity": False,
    "semantic_action": False,
    "publication": False,
    "production": False,
    "scientific_success": False,
}
SUBMISSION_AUTHORITY = {
    "diagnostic_canary_submission": True,
    "training": False,
    "optimizer_step": False,
    "checkpoint": False,
    "candidate_selection": False,
    "identity": False,
    "semantic_action": False,
    "publication": False,
    "production": False,
    "scientific_success": False,
}
RENDEZVOUS_AUTHORITY = {
    "scientific": False,
    "training": False,
    "checkpoint": False,
    "selection": False,
    "publication": False,
    "production": False,
}
RENDEZVOUS_RANK_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "slurm_job_id",
        "rdzv_backend",
        "rdzv_endpoint_request",
        "rdzv_id",
        "actual_master_addr",
        "actual_master_port",
        "rank",
        "local_rank",
        "world_size",
        "local_world_size",
        "gpu_visibility_source",
        "gpu_visibility",
        "physical_gpu_token",
        "logical_cuda_device",
        "torch_cuda_device_count",
        "torch_cuda_current_device",
        "model_loaded",
        "checkpoint_loaded",
        "generation_entered",
        "rendezvous_guard_sha256",
        "authority",
        "receipt_digest",
    }
)
RENDEZVOUS_ADMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "slurm_job_id",
        "rdzv_backend",
        "rdzv_endpoint_request",
        "rdzv_id",
        "actual_master_addr",
        "actual_master_port",
        "world_size",
        "rank_order",
        "rank_packet_digests",
        "gpu_visibility_source",
        "gpu_visibility",
        "physical_gpu_tokens",
        "logical_cuda_devices",
        "torch_cuda_device_count",
        "torch_cuda_current_devices",
        "all_four_ranks_admitted",
        "all_four_gpu_mappings_distinct",
        "kernel_selected_port",
        "numeric_port_preregistered",
        "model_load_authorized",
        "scientific_authority",
        "rendezvous_guard_sha256",
        "authority",
        "receipt_digest",
    }
)
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def die(message: str) -> None:
    raise SystemExit(f"postflight-saic-source-anchor-diagnostic-v1: {message}")


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        die(f"value is not canonical finite ASCII JSON: {error}")
    raise AssertionError("unreachable")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: str, *, label: str) -> str:
    if type(value) is not str or SHA256.fullmatch(value) is None:
        die(f"{label} SHA-256 differs")
    return value


def closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        die(f"{label} schema differs")
    return value


def exact_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_mode: int = 0o444,
    executable: bool = False,
) -> Path:
    if not path.is_absolute() or path == Path("/"):
        die(f"{label} path differs")
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        die(f"cannot resolve {label}: {error}")
    if (
        resolved != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != expected_mode
        or (executable and not os.access(path, os.X_OK))
    ):
        die(f"{label} identity/mode differs")
    if expected_sha256 is not None:
        require_sha(expected_sha256, label=label)
        if sha_file(path) != expected_sha256:
            die(f"{label} bytes differ")
    return path


def validate_file_snapshot_row(value: Any, *, label: str) -> Path:
    fields = {
        "path",
        "sha256",
        "byte_size",
        "device",
        "inode",
        "mode",
        "uid",
        "nlink",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        die(f"{label} snapshot schema differs")
    path = exact_file(
        Path(str(value.get("path", ""))),
        label=label,
        expected_sha256=require_sha(value.get("sha256"), label=label),
    )
    info = path.lstat()
    if value != {
        "path": str(path),
        "sha256": sha_file(path),
        "byte_size": int(info.st_size),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "uid": int(info.st_uid),
        "nlink": int(info.st_nlink),
    }:
        die(f"{label} current snapshot differs")
    return path


def strict_json(path: Path, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    def reject_constant(value: str) -> None:
        die(f"{label} contains {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                die(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        die(f"cannot decode {label}: {error}")
    if not isinstance(value, Mapping) or raw != canonical(value) + b"\n":
        die(f"{label} is not canonical one-object JSON")
    return value, raw


def validate_seal(
    value: Mapping[str, Any], *, digest_field: str, label: str
) -> str:
    unsigned = dict(value)
    declared = require_sha(unsigned.pop(digest_field, None), label=f"{label} digest")
    if sha_bytes(canonical(unsigned)) != declared:
        die(f"{label} digest differs")
    return declared


def exact_child(
    root: Path, declared: Any, name: str, sha256: Any, *, label: str
) -> Path:
    require_sha(sha256, label=label)
    if type(declared) is not str:
        die(f"{label} path is not text")
    path = Path(declared)
    expected = root / name
    if path != expected:
        die(f"{label} path differs")
    return exact_file(path, label=label, expected_sha256=sha256)


def tensor_raw_identity(value: Any, *, label: str) -> tuple[str, str, tuple[int, ...]]:
    if not isinstance(value, Mapping) or value.get("all_rank_exact") is not True:
        die(f"{label} lacks exact all-rank identity")
    identity = value.get("identity")
    if not isinstance(identity, Mapping):
        die(f"{label} tensor identity differs")
    raw = require_sha(identity.get("raw_storage_sha256"), label=f"{label} raw tensor")
    dtype = identity.get("dtype")
    shape = identity.get("shape")
    if (
        type(dtype) is not str
        or not isinstance(shape, list)
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
        or identity.get("finite") is not True
    ):
        die(f"{label} tensor metadata differs")
    return raw, dtype, tuple(shape)


def condition_raw_identity(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        die(f"{label} condition identity differs")
    source = tensor_raw_identity(value.get("source_video"), label=f"{label} source")
    references = value.get("references")
    if not isinstance(references, Mapping) or set(references) != {"0", "27", "53", "80"}:
        die(f"{label} reference identity closure differs")
    return {
        "source": source,
        "references": {
            key: tensor_raw_identity(references[key], label=f"{label} reference {key}")
            for key in ("0", "27", "53", "80")
        },
    }


def validate_submission(
    path: Path, *, expected_sha256: str
) -> tuple[Mapping[str, Any], str]:
    exact_file(path, label="submission receipt", expected_sha256=expected_sha256)
    value, raw = strict_json(path, label="submission receipt")
    closed(value, SUBMISSION_FIELDS, label="submission receipt")
    digest = validate_seal(value, digest_field="receipt_digest", label="submission")
    boundary = value.get("single_attempt_boundary")
    request = value.get("request")
    job = value.get("submitted_job")
    authority = value.get("authority")
    inputs = value.get("inputs")
    exports = value.get("exports")
    outputs = value.get("outputs")
    release_manifest = inputs.get("release_manifest") if isinstance(inputs, Mapping) else None
    release_value = inputs.get("resolved_release") if isinstance(inputs, Mapping) else None
    release_digest = inputs.get("release_manifest_digest") if isinstance(inputs, Mapping) else None
    export_names = (
        boundary.get("exact_export_names") if isinstance(boundary, Mapping) else None
    )
    if (
        value.get("schema_version") != SUBMISSION_SCHEMA
        or value.get("status") != "submitted_single_attempt_no_retry"
        or value.get("submission_success") is not True
        or value.get("job_success") is not None
        or not isinstance(boundary, Mapping)
        or boundary.get("reservation_created_before_sbatch") is not True
        or boundary.get("automatic_retry_allowed") is not False
        or boundary.get("launcher_submitted_from_retained_fd") is not True
        or boundary.get("environment_replaced") is not True
        or boundary.get("export_all") is not False
        or boundary.get("job_fails_before_model_load_if_receipt_is_not_terminal")
        is not True
        or not isinstance(request, Mapping)
        or request.get("world_size") != 4
        or request.get("ulysses_size") != 4
        or request.get("cell_count") != 7
        or request.get("job_name") != "saic-anchor-diag-v2"
        or request.get("partition") != "faculty"
        or type(request.get("qos")) is not str
        or request.get("first_run") != "single_world4_canary"
        or request.get("exact81") is not True
        or request.get("exact40") is not True
        or request.get("gpu_resource_requested") != "gpu:mi210:4"
        or request.get("nodes") != 1
        or request.get("ntasks") != 1
        or request.get("cpus_per_task") != 32
        or request.get("memory") != "256G"
        or request.get("walltime") != "24:00:00"
        or request.get("hold") is not False
        or request.get("dependency") is not None
        or not isinstance(job, Mapping)
        or JOB_ID.fullmatch(str(job.get("job_id", ""))) is None
        or not isinstance(inputs, Mapping)
        or set(inputs)
        != {
            "materialized_submitter",
            "release_manifest",
            "release_manifest_digest",
            "resolved_release",
        }
        or not isinstance(release_manifest, Mapping)
        or not isinstance(release_value, Mapping)
        or require_sha(release_digest, label="release manifest digest")
        != release_value.get("receipt_digest")
        or not isinstance(exports, Mapping)
        or not isinstance(export_names, list)
        or set(exports) != set(export_names)
        or len(exports) != len(export_names)
        or len(export_names) != len(set(export_names))
        or any(
            type(name) is not str
            or not name.startswith("SAIC_ANCHOR_DIAG_")
            or type(exports.get(name)) is not str
            for name in export_names
        )
        or not isinstance(outputs, Mapping)
        or authority != SUBMISSION_AUTHORITY
        or sha_bytes(raw) != expected_sha256
    ):
        die("submission receipt contract differs")
    release_path = Path(str(release_manifest.get("path", "")))
    release_sha = require_sha(
        release_manifest.get("sha256"), label="release manifest"
    )
    if validate_file_snapshot_row(
        release_manifest, label="release manifest"
    ) != release_path:
        die("release manifest path snapshot differs")
    release, release_raw = strict_json(release_path, label="release manifest")
    if (
        release != release_value
        or release.get("schema_version")
        != "saic-source-anchor-diagnostic-release-v2"
        or release.get("status") != "sealed_before_first_diagnostic_canary"
        or validate_seal(
            release, digest_field="receipt_digest", label="release manifest"
        )
        != release_digest
        or sha_bytes(release_raw) != release_sha
    ):
        die("submission release-manifest binding differs")
    release_launcher = release.get("code", {}).get("launcher")
    if (
        not isinstance(release_launcher, Mapping)
        or boundary.get("retained_wrapper_device")
        != release_launcher.get("device")
        or boundary.get("retained_wrapper_inode")
        != release_launcher.get("inode")
        or boundary.get("launcher_submitted_from_retained_fd") is not True
    ):
        die("retained wrapper full device/inode binding differs")
    validate_file_snapshot_row(release_launcher, label="retained release launcher")
    submitter_self = inputs.get("materialized_submitter")
    if (
        not isinstance(submitter_self, Mapping)
        or set(submitter_self)
        != {
            "path",
            "sha256",
            "byte_size",
            "device",
            "inode",
            "mode",
            "uid",
            "nlink",
            "archived_template_path",
            "archived_template_sha256",
            "exact_release_pin_substitution",
        }
        or submitter_self.get("exact_release_pin_substitution") is not True
        or require_sha(
            submitter_self.get("sha256"), label="materialized submitter"
        )
        != submitter_self.get("sha256")
        or require_sha(
            submitter_self.get("archived_template_sha256"),
            label="archived submitter template",
        )
        != submitter_self.get("archived_template_sha256")
    ):
        die("materialized submitter evidence differs")
    submitter_snapshot = {
        key: submitter_self[key]
        for key in (
            "path",
            "sha256",
            "byte_size",
            "device",
            "inode",
            "mode",
            "uid",
            "nlink",
        )
    }
    validate_file_snapshot_row(
        submitter_snapshot,
        label="materialized submitter",
    )
    return value, digest


def validate_runtime_receipts(
    output: Path,
    *,
    submission: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]:
    info = output.lstat()
    if (
        output.resolve(strict=True) != output
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o555
        or info.st_uid != os.getuid()
    ):
        die("diagnostic output directory identity/mode differs")
    entries = list(output.iterdir())
    if {path.name for path in entries} != EXPECTED_NAMES:
        die("diagnostic output file closure differs")
    for path in entries:
        exact_file(path, label=f"diagnostic artifact {path.name}")

    receipt_path = output / "receipt.json"
    receipt, _ = strict_json(receipt_path, label="runtime receipt")
    closed(receipt, RUNTIME_FIELDS, label="runtime receipt")
    validate_seal(receipt, digest_field="receipt_digest", label="runtime receipt")
    sampling = receipt.get("sampling")
    conditions = receipt.get("conditions")
    stage_a = receipt.get("stage_a")
    vendor_restore = receipt.get("vendor_restore")
    execution = receipt.get("execution")
    runtime_by_cell = receipt.get("runtime_by_cell")
    cell_inputs = receipt.get("cell_inputs")
    generated_identities = receipt.get("generated_identities")
    if (
        receipt.get("schema_version") != RUNTIME_SCHEMA
        or receipt.get("method")
        != "saic-source-anchor-formal32-first-decoded-canary"
        or receipt.get("classification")
        != "stage_a_checkpoint_diagnostic/no_stage_b_or_scientific_authority"
        or receipt.get("complete") is not True
        or receipt.get("first_real_runtime_status") != "canary_only"
        or not isinstance(execution, Mapping)
        or JOB_ID.fullmatch(str(execution.get("slurm_job_id", ""))) is None
        or execution.get("world_size") != 4
        or execution.get("ulysses_size") != 4
        or execution.get("single_node") is not True
        or execution.get("first_real_exact40_is_canary") is not True
        or receipt.get("cell_order") != list(CELL_ORDER)
        or receipt.get("route_drop_byte_identical_to_base_correct_noop") is not True
        or not isinstance(runtime_by_cell, Mapping)
        or set(runtime_by_cell) != set(CELL_ORDER)
        or not isinstance(cell_inputs, Mapping)
        or set(cell_inputs) != set(CELL_ORDER)
        or not isinstance(generated_identities, Mapping)
        or set(generated_identities) != set(CELL_ORDER)
        or receipt.get("authority") != RUNTIME_AUTHORITY
        or not isinstance(sampling, Mapping)
        or sampling.get("guidance_mode") != "v2v_apg"
        or sampling.get("num_frames") != 81
        or sampling.get("num_inference_steps") != 40
        or sampling.get("ulysses_size") != 4
        or sampling.get("same_official_gaussian_all_cells") is not True
        or sampling.get("external_initial_noise_injection") is not False
        or not isinstance(conditions, Mapping)
        or conditions.get("task_prompt") != "vr2v"
        or conditions.get("full_source_video_count") != 1
        or conditions.get("source_reference_count") != 4
        or conditions.get("wrong_source_replaces_video_and_all_four_references")
        is not True
        or conditions.get("route_drop_retains_complete_correct_source_condition")
        is not True
        or conditions.get("zero_condition_is_synthetic_ood") is not True
        or conditions.get("zero_condition_is_not_a_real_dropped_source_claim")
        is not True
        or not isinstance(stage_a, Mapping)
        or stage_a.get("source_anchor_pretext_only") is not True
        or stage_a.get("stage_b_authorized") is not False
        or not isinstance(vendor_restore, Mapping)
        or vendor_restore.get("byte_identical") is not True
    ):
        die("runtime diagnostic contract differs")
    rendezvous_runtime = execution.get("dynamic_rendezvous")
    rank_to_gpu = execution.get("rank_to_physical_gpu")
    gpu_visibility = execution.get("gpu_visibility")
    gpu_tokens = gpu_visibility.split(",") if isinstance(gpu_visibility, str) else []
    if (
        not isinstance(rendezvous_runtime, Mapping)
        or rendezvous_runtime.get("admitted_before_model_load") is not True
        or rendezvous_runtime.get("actual_master_addr") != "127.0.0.1"
        or not isinstance(rendezvous_runtime.get("actual_master_port"), int)
        or not 1024 <= rendezvous_runtime["actual_master_port"] <= 65535
        or rendezvous_runtime.get("gpu_visibility") != gpu_visibility
        or rendezvous_runtime.get("gpu_visibility_source")
        != execution.get("gpu_visibility_source")
        or rendezvous_runtime.get("physical_gpu_tokens") != gpu_tokens
        or len(gpu_tokens) != 4
        or len(set(gpu_tokens)) != 4
        or any(re.fullmatch(r"[0-9]+", token) is None for token in gpu_tokens)
        or rank_to_gpu
        != {str(index): token for index, token in enumerate(gpu_tokens)}
    ):
        die("runtime dynamic rendezvous/GPU mapping differs")
    condition_identities = conditions.get("condition_identities")
    if not isinstance(condition_identities, Mapping) or set(condition_identities) != {
        "correct",
        "wrong",
        "synthetic_zero",
    }:
        die("runtime condition-identity closure differs")
    canonical_conditions = {
        name: condition_raw_identity(value, label=name)
        for name, value in condition_identities.items()
    }
    distinctness = conditions.get("correct_wrong_distinctness")
    if (
        not isinstance(distinctness, Mapping)
        or distinctness.get("source_file_sha256_distinct") is not True
        or distinctness.get("encoded_full_source_raw_sha256_distinct") is not True
        or distinctness.get("encoded_reference_raw_sha256_distinct_indices")
        != [0, 27, 53, 80]
        or distinctness.get("all_four_encoded_references_distinct") is not True
        or canonical_conditions["correct"] == canonical_conditions["wrong"]
        or canonical_conditions["correct"]["source"][0]
        == canonical_conditions["wrong"]["source"][0]
        or any(
            canonical_conditions["correct"]["references"][str(index)][0]
            == canonical_conditions["wrong"]["references"][str(index)][0]
            for index in (0, 27, 53, 80)
        )
    ):
        die("correct/wrong source and encoded-condition distinctness differs")
    expected_cell_condition = {
        "base_correct_noop": "correct",
        "anchor_correct_noop": "correct",
        "anchor_wrong_noop": "wrong",
        "anchor_route_drop_noop": "correct",
        "anchor_zero_condition_noop": "synthetic_zero",
        "base_correct_action": "correct",
        "anchor_correct_action": "correct",
    }
    noise_identities: set[tuple[str, str, tuple[int, ...]]] = set()
    for cell in CELL_ORDER:
        runtime_row = runtime_by_cell[cell]
        input_row = cell_inputs[cell]
        if not isinstance(runtime_row, Mapping) or not isinstance(input_row, Mapping):
            die(f"runtime cell receipt differs: {cell}")
        if cell == "anchor_route_drop_noop":
            if (
                runtime_row.get("schema_version") != RUNTIME_SCHEMA
                or runtime_row.get("source_anchor_runtime_route_installed") is not False
                or runtime_row.get("complete_correct_source_vi_condition_retained")
                is not True
                or runtime_row.get("trained_adapter_parameters_present_but_inactive_without_route")
                is not True
                or runtime_row.get("exact40_certificate_inherited_only_after_byte_identity_to_base")
                is not True
                or runtime_row.get("semantic_or_scientific_authority") is not False
            ):
                die("route-drop runtime receipt differs")
        elif (
            runtime_row.get("schema_version")
            != "bernini-saic-source-anchor-native-runtime-v1"
            or runtime_row.get("official_sample_calls") != 1
            or runtime_row.get("exact81") is not True
            or runtime_row.get("exact40") is not True
            or runtime_row.get("branch_name") != "VI"
            or runtime_row.get("official_full_source_forwards") != 80
            or runtime_row.get("official_scheduler_steps") != 40
            or runtime_row.get("active_schedule_indices") != [35, 36, 37, 38, 39]
            or runtime_row.get("action_and_noop_share_anchor_route") is not True
            or runtime_row.get("target_only_teacher_has_anchor_route") is not False
            or runtime_row.get("optimizer_created") is not False
            or runtime_row.get("semantic_action_editing_claim") is not False
            or runtime_row.get("appearance_preservation_claim") is not False
        ):
            die(f"routed runtime receipt differs: {cell}")
        if (
            condition_raw_identity(input_row.get("condition"), label=f"{cell} input")
            != canonical_conditions[expected_cell_condition[cell]]
        ):
            die(f"runtime cell condition binding differs: {cell}")
        noise_identities.add(
            tensor_raw_identity(input_row.get("noise"), label=f"{cell} noise")
        )
        tensor_raw_identity(generated_identities[cell], label=f"{cell} generated")
    if len(noise_identities) != 1 or next(iter(noise_identities))[0] != sampling.get(
        "official_gaussian_raw_sha256"
    ):
        die("seven-cell official Gaussian identity differs")
    if tensor_raw_identity(
        generated_identities["anchor_route_drop_noop"],
        label="route-drop generated",
    ) != tensor_raw_identity(
        generated_identities["base_correct_noop"],
        label="base no-op generated",
    ):
        die("route-drop/base generated tensor identity differs")
    zero_certificate = stage_a.get("zero_initial_adapter_certificate")
    strict_load = stage_a.get("strict_load_receipt")
    if (
        not isinstance(zero_certificate, Mapping)
        or zero_certificate.get("every_output_up_element_exact_zero") is not True
        or zero_certificate.get("residual_function_exact_zero_before_checkpoint_load")
        is not True
        or zero_certificate.get("frozen_base_label_valid_before_load_only") is not True
        or not isinstance(strict_load, Mapping)
        or strict_load.get("optimizer_updates") != 32
        or strict_load.get("metadata_exact_registration_match") is not True
        or strict_load.get("semantic_action_success_claim") is not False
    ):
        die("Stage-A zero-base/load certificate differs")
    formal = stage_a.get("formal_postflight")
    stage_history = stage_a.get("history")
    formal_history = formal.get("history") if isinstance(formal, Mapping) else None
    if (
        not isinstance(formal, Mapping)
        or not isinstance(stage_history, Mapping)
        or not isinstance(formal_history, Mapping)
        or formal.get("terminal_postflight_required") is not True
        or formal.get("exact32_history_required") is not True
        or require_sha(
            formal.get("postflight_digest"), label="formal postflight digest"
        )
        != formal.get("postflight_digest")
        or require_sha(
            formal.get("history_digest"), label="formal history digest"
        )
        != formal.get("history_digest")
        or formal_history.get("sha256") != stage_history.get("sha256")
        or strict_load.get("optimizer_updates") != 32
    ):
        die("formal Stage-A postflight/history runtime binding differs")
    prompts = receipt.get("prompts")
    prompt_distinctness = (
        prompts.get("distinctness") if isinstance(prompts, Mapping) else None
    )
    if not isinstance(prompts, Mapping) or not isinstance(
        prompt_distinctness, Mapping
    ):
        die("prompt distinctness receipt is absent")
    noop_tokens = tensor_raw_identity(
        prompt_distinctness.get("noop_token_ids"), label="no-op token IDs"
    )
    action_tokens = tensor_raw_identity(
        prompt_distinctness.get("action_token_ids"), label="action token IDs"
    )
    noop_embedding = tensor_raw_identity(
        prompt_distinctness.get("noop_embedding"), label="no-op embedding"
    )
    action_embedding = tensor_raw_identity(
        prompt_distinctness.get("action_embedding"), label="action embedding"
    )
    noop_mask = tensor_raw_identity(
        prompt_distinctness.get("noop_attention_mask"),
        label="no-op attention mask",
    )
    action_mask = tensor_raw_identity(
        prompt_distinctness.get("action_attention_mask"),
        label="action attention mask",
    )
    if (
        prompt_distinctness.get("raw_instruction_distinct") is not True
        or prompt_distinctness.get("clean_body_distinct") is not True
        or prompt_distinctness.get("full_prompt_distinct") is not True
        or prompt_distinctness.get("token_ids_distinct") is not True
        or prompt_distinctness.get("attention_mask_distinct") is not True
        or prompt_distinctness.get("embedding_distinct") is not True
        or prompts.get("action_caption_raw_sha256")
        == prompts.get("noop_instruction_sha256")
        or prompt_distinctness.get("action_clean_body_sha256")
        == prompt_distinctness.get("noop_clean_body_sha256")
        or prompts.get("action_full_prompt_sha256")
        == prompts.get("noop_full_prompt_sha256")
        or action_tokens[0] == noop_tokens[0]
        or action_mask[0] == noop_mask[0]
        or action_embedding[0] == noop_embedding[0]
    ):
        die("action raw/clean/full/token/embedding collapsed to no-op")

    closure: list[Mapping[str, Any]] = []
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(CELL_ORDER):
        die("runtime output cell closure differs")
    for cell in CELL_ORDER:
        row = outputs[cell]
        if not isinstance(row, Mapping):
            die(f"runtime output row differs: {cell}")
        video = exact_child(
            output, row.get("path"), f"{cell}.mp4", row.get("sha256"), label=f"{cell} MP4"
        )
        latent = row.get("normalized_clean_latent")
        if (
            row.get("frame_count") != 81
            or row.get("fps") != 25
            or not isinstance(latent, Mapping)
            or latent.get("roundtrip_byte_exact_fp32") is not True
            or latent.get("native_sampler_before_vae_decode") is not True
            or latent.get("mp4_decode_reencode_used") is not False
        ):
            die(f"runtime output media/latent contract differs: {cell}")
        latent_path = exact_child(
            output,
            latent.get("path"),
            f"{cell}.normalized-clean-latent.safetensors",
            latent.get("sha256"),
            label=f"{cell} normalized latent",
        )
        closure.extend(
            (
                {"name": video.name, "sha256": row["sha256"]},
                {"name": latent_path.name, "sha256": latent["sha256"]},
            )
        )
    base_latent_sha = outputs["base_correct_noop"]["normalized_clean_latent"]["sha256"]
    route_drop_latent_sha = outputs["anchor_route_drop_noop"][
        "normalized_clean_latent"
    ]["sha256"]
    if route_drop_latent_sha != base_latent_sha:
        die("route-drop/base persisted latent bytes differ")

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        die("runtime artifact receipt differs")
    for key, name in (
        ("shared_official_gaussian", "shared.official-initial-gaussian.safetensors"),
        ("correct_source_latent", "correct-source.normalized-clean-latent.safetensors"),
        ("wrong_source_latent", "wrong-source.normalized-clean-latent.safetensors"),
    ):
        row = artifacts.get(key)
        if not isinstance(row, Mapping):
            die(f"runtime artifact row differs: {key}")
        path = exact_child(
            output, row.get("path"), name, row.get("sha256"), label=key
        )
        expected_roundtrip_key = (
            "roundtrip_raw_value_exact"
            if key == "shared_official_gaussian"
            else "roundtrip_byte_exact_fp32"
        )
        if row.get(expected_roundtrip_key) is not True:
            die(f"runtime artifact round trip differs: {key}")
        closure.append({"name": path.name, "sha256": row["sha256"]})

    decoded_row = artifacts.get("decoded_diagnostics")
    if not isinstance(decoded_row, Mapping):
        die("decoded diagnostic artifact receipt differs")
    decoded_path = exact_child(
        output,
        decoded_row.get("path"),
        "decoded-diagnostics.json",
        decoded_row.get("sha256"),
        label="decoded diagnostics",
    )
    decoded, _ = strict_json(decoded_path, label="decoded diagnostics")
    closed(decoded, DIAGNOSTIC_FIELDS, label="decoded diagnostics")
    decoded_digest = validate_seal(
        decoded, digest_field="diagnostic_digest", label="decoded diagnostics"
    )
    availability = decoded.get("availability")
    comparisons = decoded.get("comparisons")
    visual_model = decoded.get("visual_model")
    cells = decoded.get("cells")
    if (
        decoded.get("schema_version") != DIAGNOSTIC_SCHEMA
        or decoded.get("cell_order") != list(CELL_ORDER)
        or decoded.get("authority") != DIAGNOSTIC_AUTHORITY
        or decoded_row.get("diagnostic_digest") != decoded_digest
        or not isinstance(availability, Mapping)
        or availability.get("exact81_decode") != "available"
        or availability.get("full80_camera") != "diagnostic_only"
        or availability.get("dinov2_identity_appearance") != "proxy_only"
        or availability.get("semantic_action_event") != "unavailable"
        or not isinstance(comparisons, Mapping)
        or comparisons.get("route_drop_is_a_mechanism_control_not_a_dropped_source_claim")
        is not True
        or comparisons.get("zero_condition_is_synthetic_ood_not_a_real_source_claim")
        is not True
        or not isinstance(visual_model, Mapping)
        or visual_model.get("identity_authority") is not False
        or visual_model.get("every_non_cache_file_sha256_verified") is not True
        or visual_model.get("exact_all_file_set_no_cache_exclusion") is not True
        or visual_model.get("golden_preprocessor_exact") is not True
        or not isinstance(visual_model.get("visual_release_manifest"), Mapping)
        or not isinstance(visual_model.get("evaluator_spec"), Mapping)
        or not isinstance(visual_model.get("evaluator_sources"), list)
        or len(visual_model.get("evaluator_sources")) != 2
        or require_sha(
            visual_model.get("visual_release_digest"),
            label="visual release digest",
        )
        != visual_model.get("visual_release_digest")
        or require_sha(
            visual_model.get("evaluator_spec_digest"),
            label="visual evaluator spec digest",
        )
        != visual_model.get("evaluator_spec_digest")
        or require_sha(
            visual_model.get("evaluator_sources_digest"),
            label="visual evaluator sources digest",
        )
        != visual_model.get("evaluator_sources_digest")
        or not isinstance(cells, Mapping)
        or set(cells) != set(CELL_ORDER)
    ):
        die("decoded diagnostics contract differs")
    action_comparison = comparisons.get("action_anchor_minus_base")
    if (
        not isinstance(action_comparison, Mapping)
        or action_comparison.get("semantic_action_observer_available") is not False
        or action_comparison.get("semantic_action_nonregression_verdict") != "unavailable"
    ):
        die("semantic action limitation differs")
    for comparison_name in (
        "noop_anchor_minus_base",
        "action_anchor_minus_base",
    ):
        row = comparisons.get(comparison_name)
        if not isinstance(row, Mapping) or any(
            type(row.get(name)) not in (int, float)
            or isinstance(row.get(name), bool)
            or not math.isfinite(float(row[name]))
            for name in (
                "identity_appearance_proxy",
                "quality_proxy",
                "camera_error_delta_lower_is_better",
            )
        ):
            die(f"decoded comparison differs: {comparison_name}")
    for name in (
        "correct_vs_wrong_noop_identity_proxy",
        "correct_vs_zero_condition_noop_identity_proxy",
    ):
        if (
            type(comparisons.get(name)) not in (int, float)
            or isinstance(comparisons.get(name), bool)
            or not math.isfinite(float(comparisons[name]))
        ):
            die(f"decoded source-control proxy differs: {name}")
    for cell in CELL_ORDER:
        row = cells[cell]
        if not isinstance(row, Mapping):
            die(f"decoded cell row differs: {cell}")
        proxy = row.get("frozen_visual_proxy")
        media = row.get("full81_full80_media_diagnostic")
        candidate = row.get("candidate")
        media_authority = media.get("authority") if isinstance(media, Mapping) else None
        if (
            not isinstance(proxy, Mapping)
            or proxy.get("identity_or_appearance_authority") is not False
            or proxy.get("absolute_thresholds_calibrated") is not False
            or not isinstance(media, Mapping)
            or not isinstance(media_authority, Mapping)
            or any(value is not False for value in media_authority.values())
            or not isinstance(candidate, Mapping)
            or candidate.get("path") != outputs[cell].get("path")
            or candidate.get("sha256") != outputs[cell].get("sha256")
        ):
            die(f"decoded cell authority/binding differs: {cell}")
        validate_seal(media, digest_field="diagnostic_digest", label=f"{cell} media")
    closure.append({"name": decoded_path.name, "sha256": decoded_row["sha256"]})
    closure.append({"name": receipt_path.name, "sha256": sha_file(receipt_path)})
    if len(closure) != 19 or {row["name"] for row in closure} != EXPECTED_NAMES:
        die("postflight artifact closure differs")
    if submission is not None:
        submission_inputs = submission.get("inputs")
        submission_job = submission.get("submitted_job")
        runtime_input = receipt.get("input")
        source_revisions = receipt.get("source_revisions")
        checkpoint = receipt.get("checkpoint")
        release = (
            submission_inputs.get("resolved_release")
            if isinstance(submission_inputs, Mapping)
            else None
        )
        if not all(
            isinstance(value, Mapping)
            for value in (
                submission_inputs,
                submission_job,
                release,
                runtime_input,
                source_revisions,
                checkpoint,
                stage_a,
            )
        ):
            die("submission/runtime nested binding differs")
        source_release = release.get("source_release")
        base_release = release.get("base_model")
        scientific_release = release.get("scientific_input")
        stage_release = release.get("stage_a_release")
        visual_release = release.get("visual_release")
        run_policy = release.get("run_policy")
        source_manifest = runtime_input.get("source_manifest")
        action_caption = runtime_input.get("action_caption_file")
        runtime_adapter = stage_a.get("adapter")
        runtime_stage_a_receipt = stage_a.get("receipt")
        runtime_stage_a_history = stage_a.get("history")
        runtime_checkpoint_release = stage_a.get("checkpoint_release")
        runtime_formal = stage_a.get("formal_postflight")
        runtime_checkpoint_manifest = checkpoint.get("content_manifest")
        visual_checkpoint_manifest = visual_model.get("checkpoint_manifest")
        runtime_visual_release = visual_model.get("visual_release_manifest")
        runtime_evaluator_spec = visual_model.get("evaluator_spec")
        runtime_source_release = source_revisions.get(
            "method_source_release_manifest"
        )
        if not all(
            isinstance(value, Mapping)
            for value in (
                source_release,
                base_release,
                scientific_release,
                stage_release,
                visual_release,
                run_policy,
                source_manifest,
                action_caption,
                runtime_adapter,
                runtime_stage_a_receipt,
                runtime_stage_a_history,
                runtime_checkpoint_release,
                runtime_formal,
                runtime_checkpoint_manifest,
                visual_checkpoint_manifest,
                runtime_visual_release,
                runtime_evaluator_spec,
                runtime_source_release,
            )
        ):
            die("submission/runtime artifact binding differs")
        if (
            execution.get("slurm_job_id") != submission_job.get("job_id")
            or source_revisions.get("method")
            != source_release.get("revision")
            or source_revisions.get("method_source_archive_sha256")
            != source_release.get("archive", {}).get("sha256")
            or runtime_source_release.get("sha256")
            != submission_inputs.get("release_manifest", {}).get("sha256")
            or source_revisions.get("method_source_member_manifest_sha256")
            != source_release.get("member_manifest", {}).get("sha256")
            or source_revisions.get("method_source_member_manifest_digest")
            != source_release.get("member_manifest_digest")
            or source_revisions.get("method_source_member_count")
            != source_release.get("member_count")
            or source_revisions.get("method_source_origin_manifest_sha256")
            != source_release.get("origin_manifest", {}).get("sha256")
            or source_revisions.get("method_source_origin_manifest_digest")
            != source_release.get("origin_manifest_digest")
            or source_revisions.get("method_source_origin_count")
            != source_release.get("origin_count")
            or source_revisions.get(
                "archive_recursive_closure_preflighted_before_import"
            )
            is not True
            or source_revisions.get("all_import_origins_matched_extracted_tree")
            is not True
            or source_manifest.get("sha256")
            != scientific_release.get("source_manifest", {}).get("sha256")
            or runtime_adapter.get("sha256")
            != stage_release.get("adapter", {}).get("sha256")
            or runtime_stage_a_receipt.get("sha256")
            != stage_release.get("training_receipt", {}).get("sha256")
            or runtime_stage_a_history.get("sha256")
            != stage_release.get("training_history", {}).get("sha256")
            or runtime_checkpoint_release.get("sha256")
            != stage_release.get("checkpoint_release", {}).get("sha256")
            or runtime_formal.get("checkpoint_release", {}).get("sha256")
            != stage_release.get("checkpoint_release", {}).get("sha256")
            or runtime_formal.get("postflight", {}).get("sha256")
            != stage_release.get("formal_postflight", {}).get("sha256")
            or runtime_formal.get("postflight_digest")
            != stage_release.get("formal_postflight_digest")
            or action_caption.get("sha256")
            != scientific_release.get("action_caption", {}).get("sha256")
            or runtime_checkpoint_manifest.get("sha256")
            != base_release.get("checkpoint_manifest", {}).get("sha256")
            or visual_checkpoint_manifest.get("sha256")
            != visual_release.get("content_manifest", {}).get("sha256")
            or runtime_visual_release.get("sha256")
            != visual_release.get("release_manifest", {}).get("sha256")
            or runtime_evaluator_spec.get("sha256")
            != visual_release.get("evaluator_spec", {}).get("sha256")
            or runtime_input.get("heldout_row_index")
            != scientific_release.get("heldout_row_index")
            or sampling.get("seed") != run_policy.get("seed")
        ):
            die("submission/runtime/job cross-binding differs")
        formal_path = Path(str(stage_release["formal_postflight"]["path"]))
        exact_file(
            formal_path,
            label="formal Stage-A postflight",
            expected_sha256=stage_release["formal_postflight"]["sha256"],
        )
        formal_value, _ = strict_json(
            formal_path, label="formal Stage-A postflight"
        )
        if (
            formal_value.get("schema_version")
            != "saic-source-anchor-formal32-terminal-admission-v1"
            or formal_value.get("status")
            != "FORMAL_GATE_PASS_CHECKPOINT_RELEASED"
            or formal_value.get("complete") is not True
            or validate_seal(
                formal_value,
                digest_field="receipt_digest",
                label="formal Stage-A postflight",
            )
            != stage_release.get("formal_postflight_digest")
        ):
            die("formal Stage-A external postflight changed")
        history_path = Path(str(stage_release["training_history"]["path"]))
        exact_file(
            history_path,
            label="formal Stage-A history",
            expected_sha256=stage_release["training_history"]["sha256"],
        )
        history_value, _ = strict_json(
            history_path, label="formal Stage-A history"
        )
        if (
            history_value.get("schema_version")
            != "bernini-saic-source-anchor-history-v2"
            or history_value.get("complete") is not True
            or history_value.get("optimizer_updates") != 32
            or history_value.get("update_indices") != list(range(32))
            or not isinstance(history_value.get("rows"), list)
            or len(history_value["rows"]) != 32
            or validate_seal(
                history_value,
                digest_field="history_digest",
                label="formal Stage-A history",
            )
            != runtime_formal.get("history_digest")
        ):
            die("formal Stage-A exact32 history changed")
        checkpoint_release_path = Path(
            str(stage_release["checkpoint_release"]["path"])
        )
        exact_file(
            checkpoint_release_path,
            label="formal Stage-A checkpoint release",
            expected_sha256=stage_release["checkpoint_release"]["sha256"],
        )
        checkpoint_release_value, _ = strict_json(
            checkpoint_release_path,
            label="formal Stage-A checkpoint release",
        )
        closed(
            checkpoint_release_value,
            CHECKPOINT_RELEASE_FIELDS,
            label="formal Stage-A checkpoint release",
        )
        checkpoint_release_digest = validate_seal(
            checkpoint_release_value,
            digest_field="receipt_digest",
            label="formal Stage-A checkpoint release",
        )
        for field in (
            "source_release_manifest_sha256",
            "submission_receipt_sha256",
            "postflight_source_sha256",
            "trainer_source_sha256",
        ):
            require_sha(
                checkpoint_release_value.get(field),
                label=f"formal checkpoint release {field}",
            )
        artifact_inputs = {
            "adapter": stage_release["adapter"],
            "training_receipt": stage_release["training_receipt"],
            "training_history": stage_release["training_history"],
            "source_manifest": scientific_release["source_manifest"],
            "checkpoint_manifest": base_release["checkpoint_manifest"],
        }
        artifact_rows = checkpoint_release_value.get("artifacts")
        expected_artifact_rows = []
        if not isinstance(artifact_rows, Mapping) or set(artifact_rows) != set(
            artifact_inputs
        ):
            die("formal checkpoint release artifact closure differs")
        for name, source_row in artifact_inputs.items():
            row = artifact_rows[name]
            if not isinstance(row, Mapping) or set(row) != {
                "path",
                "sha256",
                "byte_size",
                "device",
                "inode",
            }:
                die(f"formal checkpoint release artifact schema differs: {name}")
            expected = {
                key: source_row[key]
                for key in ("path", "sha256", "byte_size", "device", "inode")
            }
            if row != expected:
                die(f"formal checkpoint release artifact binding differs: {name}")
            expected_artifact_rows.append({"name": name, **dict(row)})
        expected_artifact_rows.sort(key=lambda row: row["name"])
        if (
            checkpoint_release_value.get("schema_version")
            != "saic-source-anchor-formal32-checkpoint-release-v1"
            or checkpoint_release_value.get("status")
            != "FORMAL_GATE_PASS_CHECKPOINT_RELEASED"
            or checkpoint_release_value.get("complete") is not True
            or checkpoint_release_value.get("release_path")
            != str(checkpoint_release_path.parent)
            or checkpoint_release_value.get("formal_output_namespace")
            != str(checkpoint_release_path.parent)
            or str(checkpoint_release_value.get("job_id"))
            != str(runtime_formal.get("formal_slurm_job_id"))
            or checkpoint_release_value.get("history_digest")
            != runtime_formal.get("history_digest")
            or checkpoint_release_value.get("heldout_gate")
            != formal_value.get("heldout_gate")
            or checkpoint_release_value.get("trainer_source_sha256")
            != stage_a.get("trainer_source_sha256")
            or checkpoint_release_value.get("source_release_manifest_sha256")
            != stage_a.get("training_release_manifest_sha256")
            or checkpoint_release_value.get("submission_receipt_sha256")
            != stage_a.get("training_submission_receipt_sha256")
            or checkpoint_release_value.get("authority")
            != CHECKPOINT_RELEASE_AUTHORITY
            or checkpoint_release_value.get("payload_files_digest")
            != sha_bytes(canonical(expected_artifact_rows))
            or checkpoint_release_digest
            != runtime_formal.get("checkpoint_release_digest")
        ):
            die("formal Stage-A checkpoint release changed")
    return receipt, decoded, sorted(closure, key=lambda row: row["name"])


def observe_sacct(
    executable: Path, *, job_id: str, submission: Mapping[str, Any]
) -> Mapping[str, Any]:
    field_specs = (
        "JobIDRaw",
        "JobName",
        "Partition",
        "QOS",
        "State",
        "ExitCode",
        "AllocTRES%512",
        "NodeList",
        "Start",
        "End",
        "Elapsed",
        "SubmitLine%8192",
    )
    field_keys = tuple(value.split("%", 1)[0] for value in field_specs)
    command = [
        str(executable),
        "-X",
        "--noheader",
        "--parsable2",
        "--jobs",
        job_id,
        "--format=" + ",".join(field_specs),
    ]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        text = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        die("sacct output is not ASCII")
    rows = []
    for line in text.splitlines():
        values = line.split("|")
        if len(values) == len(field_keys) and values[0] == job_id:
            rows.append(dict(zip(field_keys, values)))
    if completed.returncode != 0 or completed.stderr or len(rows) != 1:
        die(
            "authoritative sacct row unavailable; "
            f"exit={completed.returncode} stderr_sha256={sha_bytes(completed.stderr)}"
        )
    row = rows[0]
    terminal_state = row["State"].split()[0].split("+")[0]
    tres: dict[str, str] = {}
    for token in row["AllocTRES"].split(","):
        if token.count("=") != 1:
            die("terminal AllocTRES token differs")
        key, item = token.split("=", 1)
        if not key or key in tres:
            die("terminal AllocTRES token closure differs")
        tres[key] = item
    request = submission.get("request")
    exports = submission.get("exports")
    outputs = submission.get("outputs")
    boundary = submission.get("single_attempt_boundary")
    export_names = (
        boundary.get("exact_export_names")
        if isinstance(boundary, Mapping)
        else None
    )
    if (
        not all(isinstance(item, Mapping) for item in (request, exports, outputs))
        or not isinstance(export_names, list)
        or len(export_names) != len(set(export_names))
        or set(export_names) != set(exports)
    ):
        die("submission accounting inputs differ")
    log_dir = Path(str(exports.get("SAIC_ANCHOR_DIAG_SLURM_LOG_DIR", "")))
    fixed_submit_line = " ".join(
        [
            "/usr/bin/sbatch",
            "--parsable",
            f"--qos={request.get('qos')}",
            f"--output={log_dir}/saic-anchor-diag-v2-%j.out",
            f"--error={log_dir}/saic-anchor-diag-v2-%j.err",
            "--export=NONE,"
            + ",".join(f"{name}={exports[name]}" for name in export_names),
        ]
    )
    retained_match = re.fullmatch(
        re.escape(fixed_submit_line + " /proc/self/fd/") + r"([0-9]+)",
        row["SubmitLine"],
    )
    if (
        terminal_state != "COMPLETED"
        or row["ExitCode"] != "0:0"
        or row["JobName"] != "saic-anchor-diag-v2"
        or row["Partition"] != "faculty"
        or row["QOS"] != request.get("qos")
        or tres
        != {
            "billing": "32",
            "cpu": "32",
            "gres/gpu:mi210": "4",
            "gres/gpu": "4",
            "mem": "256G",
            "node": "1",
        }
        or not row["NodeList"]
        or row["NodeList"] in {"Unknown", "None assigned"}
        or not row["Start"]
        or row["Start"] == "Unknown"
        or not row["End"]
        or row["End"] == "Unknown"
        or not row["Elapsed"]
        or re.fullmatch(
            r"(?:[0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}", row["Elapsed"]
        )
        is None
        or retained_match is None
        or str(int(retained_match.group(1))) != retained_match.group(1)
        or int(retained_match.group(1)) < 3
    ):
        die("Slurm terminal success/topology/SubmitLine differs")
    return {
        "command": command,
        "query_fields": list(field_specs),
        "stdout_sha256": sha_bytes(completed.stdout),
        "stderr_sha256": sha_bytes(completed.stderr),
        "row": row,
        "terminal_state": terminal_state,
        "alloc_tres": tres,
        "allocated_gpu_count": 4,
        "allocated_cpu_count": 32,
        "allocated_memory": "256G",
        "allocated_node_count": 1,
        "submit_line_sha256": sha_bytes(row["SubmitLine"].encode("ascii")),
        "retained_wrapper_fd": int(retained_match.group(1)),
        "exact_submit_line": True,
        "terminal_success": True,
    }


def validate_rendezvous_evidence(
    root: Path,
    *,
    runtime: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> Mapping[str, Any]:
    info = root.lstat()
    if (
        not root.is_absolute()
        or root.resolve(strict=True) != root
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o555
    ):
        die("rendezvous evidence directory identity/mode differs")
    names = {"admission.json"} | {f"rank-{rank}.json" for rank in range(4)}
    if {path.name for path in root.iterdir()} != names:
        die("rendezvous evidence file closure differs")
    execution = runtime.get("execution")
    submission_job = submission.get("submitted_job")
    submission_inputs = submission.get("inputs")
    release = (
        submission_inputs.get("resolved_release")
        if isinstance(submission_inputs, Mapping)
        else None
    )
    guard_row = release.get("code", {}).get("rendezvous_guard") if isinstance(release, Mapping) else None
    runtime_row = (
        execution.get("dynamic_rendezvous")
        if isinstance(execution, Mapping)
        else None
    )
    if not all(
        isinstance(item, Mapping)
        for item in (execution, submission_job, guard_row, runtime_row)
    ):
        die("rendezvous cross-binding inputs differ")
    admission_path = exact_file(root / "admission.json", label="rendezvous admission")
    admission, admission_raw = strict_json(
        admission_path, label="rendezvous admission"
    )
    closed(
        admission,
        RENDEZVOUS_ADMISSION_FIELDS,
        label="rendezvous admission",
    )
    admission_digest = validate_seal(
        admission, digest_field="receipt_digest", label="rendezvous admission"
    )
    tokens = admission.get("physical_gpu_tokens")
    rank_digests = admission.get("rank_packet_digests")
    if (
        admission.get("schema_version")
        != "saic-source-anchor-diagnostic-world4-admission-v1"
        or admission.get("status")
        != "exact_world4_dynamic_rendezvous_admitted_before_model_load"
        or admission.get("slurm_job_id") != submission_job.get("job_id")
        or admission.get("rdzv_backend") != "c10d"
        or admission.get("rdzv_endpoint_request") != "127.0.0.1:0"
        or admission.get("actual_master_addr") != "127.0.0.1"
        or not isinstance(admission.get("actual_master_port"), int)
        or not 1024 <= admission["actual_master_port"] <= 65535
        or admission.get("world_size") != 4
        or admission.get("rank_order") != [0, 1, 2, 3]
        or admission.get("logical_cuda_devices") != [0, 1, 2, 3]
        or admission.get("torch_cuda_device_count") != 4
        or admission.get("torch_cuda_current_devices") != [0, 1, 2, 3]
        or not isinstance(tokens, list)
        or len(tokens) != 4
        or len(set(tokens)) != 4
        or not isinstance(rank_digests, list)
        or len(rank_digests) != 4
        or admission.get("all_four_ranks_admitted") is not True
        or admission.get("all_four_gpu_mappings_distinct") is not True
        or admission.get("kernel_selected_port") is not True
        or admission.get("numeric_port_preregistered") is not False
        or admission.get("model_load_authorized") is not True
        or admission.get("scientific_authority") is not False
        or admission.get("rendezvous_guard_sha256") != guard_row.get("sha256")
        or admission.get("authority") != RENDEZVOUS_AUTHORITY
        or runtime_row.get("root") != str(root)
        or runtime_row.get("decision_path") != str(admission_path)
        or runtime_row.get("decision_sha256") != sha_bytes(admission_raw)
        or runtime_row.get("decision_digest") != admission_digest
        or runtime_row.get("rank_packet_digests") != rank_digests
        or runtime_row.get("rdzv_id") != admission.get("rdzv_id")
        or runtime_row.get("actual_master_port")
        != admission.get("actual_master_port")
        or runtime_row.get("physical_gpu_tokens") != tokens
        or runtime_row.get("admitted_before_model_load") is not True
    ):
        die("dynamic WORLD4 rendezvous admission differs")
    rows = []
    for rank in range(4):
        path = exact_file(root / f"rank-{rank}.json", label=f"rendezvous rank {rank}")
        packet, raw = strict_json(path, label=f"rendezvous rank {rank}")
        closed(
            packet,
            RENDEZVOUS_RANK_FIELDS,
            label=f"rendezvous rank {rank}",
        )
        digest = validate_seal(
            packet, digest_field="receipt_digest", label=f"rendezvous rank {rank}"
        )
        if (
            packet.get("schema_version")
            != "saic-source-anchor-diagnostic-rank-admission-v1"
            or packet.get("status") != "rank_admitted_before_model_load"
            or packet.get("slurm_job_id") != submission_job.get("job_id")
            or packet.get("rdzv_id") != admission.get("rdzv_id")
            or packet.get("actual_master_addr") != "127.0.0.1"
            or packet.get("actual_master_port")
            != admission.get("actual_master_port")
            or packet.get("rank") != rank
            or packet.get("local_rank") != rank
            or packet.get("world_size") != 4
            or packet.get("local_world_size") != 4
            or packet.get("gpu_visibility_source")
            != admission.get("gpu_visibility_source")
            or packet.get("gpu_visibility") != admission.get("gpu_visibility")
            or packet.get("physical_gpu_token") != tokens[rank]
            or packet.get("logical_cuda_device") != rank
            or packet.get("torch_cuda_device_count") != 4
            or packet.get("torch_cuda_current_device") != rank
            or packet.get("model_loaded") is not False
            or packet.get("checkpoint_loaded") is not False
            or packet.get("generation_entered") is not False
            or packet.get("rendezvous_guard_sha256") != guard_row.get("sha256")
            or packet.get("authority") != RENDEZVOUS_AUTHORITY
            or digest != rank_digests[rank]
        ):
            die(f"dynamic rendezvous rank {rank} evidence differs")
        rows.append(
            {
                "name": path.name,
                "sha256": sha_bytes(raw),
                "receipt_digest": digest,
            }
        )
    return {
        "root": str(root),
        "directory_mode": "0555",
        "file_count": 5,
        "admission": {
            "path": str(admission_path),
            "sha256": sha_bytes(admission_raw),
            "receipt_digest": admission_digest,
        },
        "rank_packets": rows,
        "rank_packets_digest": sha_bytes(canonical(rows)),
        "kernel_selected_port": True,
        "exact_world4_gpu_mapping": True,
        "scientific_authority": False,
    }


def _open_stable_log(path: Path, *, parent: Path, label: str) -> tuple[int, bytes, tuple[int, int]]:
    if path.parent != parent or not path.is_absolute():
        die(f"{label} path escaped the exact log directory")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        leaf = path.lstat()
        identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) not in {0o600, 0o640, 0o644, 0o444}
            or not stat.S_ISREG(leaf.st_mode)
            or stat.S_ISLNK(leaf.st_mode)
            or leaf.st_nlink != 1
            or (leaf.st_dev, leaf.st_ino) != identity
        ):
            die(f"{label} identity/mode differs")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        ):
            die(f"{label} changed during retained read")
        return descriptor, b"".join(chunks), identity
    except BaseException:
        os.close(descriptor)
        raise


def seal_terminal_logs(submission: Mapping[str, Any]) -> Mapping[str, Any]:
    outputs = submission.get("outputs")
    exports = submission.get("exports")
    job = submission.get("submitted_job")
    if not all(isinstance(item, Mapping) for item in (outputs, exports, job)):
        die("terminal log binding inputs differ")
    log_dir = Path(str(exports.get("SAIC_ANCHOR_DIAG_SLURM_LOG_DIR", "")))
    log_info = log_dir.lstat()
    boundary = submission.get("single_attempt_boundary")
    if (
        not isinstance(boundary, Mapping)
        or log_dir.resolve(strict=True) != log_dir
        or not stat.S_ISDIR(log_info.st_mode)
        or stat.S_ISLNK(log_info.st_mode)
        or stat.S_IMODE(log_info.st_mode) != 0o700
        or (log_info.st_dev, log_info.st_ino)
        != (
            boundary.get("slurm_log_dir_device"),
            boundary.get("slurm_log_dir_inode"),
        )
    ):
        die("terminal log directory identity changed")
    job_id = str(job.get("job_id"))
    stdout_path = Path(str(outputs.get("slurm_stdout", "")))
    stderr_path = Path(str(outputs.get("slurm_stderr", "")))
    if (
        stdout_path != log_dir / f"saic-anchor-diag-v2-{job_id}.out"
        or stderr_path != log_dir / f"saic-anchor-diag-v2-{job_id}.err"
    ):
        die("terminal log derived paths differ")
    stdout_fd, stdout_raw, stdout_identity = _open_stable_log(
        stdout_path, parent=log_dir, label="Slurm stdout"
    )
    try:
        stderr_fd, stderr_raw, stderr_identity = _open_stable_log(
            stderr_path, parent=log_dir, label="Slurm stderr"
        )
    except BaseException:
        os.close(stdout_fd)
        raise
    sentinel = str(outputs.get("terminal_stdout_sentinel", "")).encode("ascii")
    try:
        if (
            not sentinel
            or stdout_raw.count(sentinel) != 1
            or not stdout_raw.endswith(sentinel + b"\n")
            or stderr_raw
            or sha_bytes(stderr_raw) != EMPTY_SHA256
        ):
            die("terminal Slurm stdout/stderr/sentinel closure differs")
        os.fchmod(stdout_fd, 0o444)
        os.fchmod(stderr_fd, 0o444)
        if (
            stat.S_IMODE(os.fstat(stdout_fd).st_mode) != 0o444
            or stat.S_IMODE(os.fstat(stderr_fd).st_mode) != 0o444
            or (stdout_path.lstat().st_dev, stdout_path.lstat().st_ino)
            != stdout_identity
            or (stderr_path.lstat().st_dev, stderr_path.lstat().st_ino)
            != stderr_identity
        ):
            die("terminal log immutable transition differs")
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)
    return {
        "stdout": {
            "path": str(stdout_path),
            "sha256": sha_bytes(stdout_raw),
            "byte_size": len(stdout_raw),
            "device": stdout_identity[0],
            "inode": stdout_identity[1],
            "mode": "0444",
        },
        "stderr": {
            "path": str(stderr_path),
            "sha256": EMPTY_SHA256,
            "byte_size": 0,
            "device": stderr_identity[0],
            "inode": stderr_identity[1],
            "mode": "0444",
        },
        "sentinel": sentinel.decode("ascii"),
        "sentinel_exact_once_and_final_line": True,
        "logs_sealed_after_terminal_accounting": True,
    }


def write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        die("postflight admission is not create-only")
    payload = canonical(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    reserved = os.fstat(descriptor)
    try:
        leaf = path.lstat()
        if (
            not stat.S_ISREG(reserved.st_mode)
            or reserved.st_nlink != 1
            or stat.S_IMODE(reserved.st_mode) != 0o600
            or not stat.S_ISREG(leaf.st_mode)
            or stat.S_ISLNK(leaf.st_mode)
            or leaf.st_nlink != 1
            or (leaf.st_dev, leaf.st_ino) != (reserved.st_dev, reserved.st_ino)
        ):
            die("postflight reservation inode differs")
        offset = 0
        while offset < len(payload):
            wrote = os.write(descriptor, payload[offset:])
            if wrote <= 0:
                die("postflight write stalled")
            offset += wrote
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(payload) + 1) != payload:
            die("postflight retained-FD reread differs")
        staged = os.fstat(descriptor)
        public = path.lstat()
        if (
            (staged.st_dev, staged.st_ino) != (reserved.st_dev, reserved.st_ino)
            or staged.st_size != len(payload)
            or staged.st_nlink != 1
            or stat.S_IMODE(staged.st_mode) != 0o600
            or (public.st_dev, public.st_ino) != (reserved.st_dev, reserved.st_ino)
            or public.st_nlink != 1
            or stat.S_ISLNK(public.st_mode)
        ):
            die("postflight staged public identity differs")
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        os.fchmod(descriptor, 0o444)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    return sha_bytes(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-receipt", required=True)
    parser.add_argument("--expected-submission-receipt-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sacct", default="/usr/bin/sacct")
    parser.add_argument("--expected-sacct-sha256", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--ack-operational-only-no-scientific-authority", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ack_operational_only_no_scientific_authority is not True:
        die("operational-only acknowledgement is required")
    submission_path = Path(args.submission_receipt)
    submission, submission_digest = validate_submission(
        submission_path,
        expected_sha256=require_sha(
            args.expected_submission_receipt_sha256, label="submission receipt"
        ),
    )
    output = Path(args.output)
    if (
        not output.is_absolute()
        or submission.get("outputs", {}).get("diagnostic_output") != str(output)
    ):
        die("submission/output binding differs")
    output_parent = output.parent
    try:
        output_parent_info = output_parent.lstat()
    except OSError as error:
        die(f"cannot resolve output parent: {error}")
    if (
        output_parent.resolve(strict=True) != output_parent
        or not stat.S_ISDIR(output_parent_info.st_mode)
        or stat.S_ISLNK(output_parent_info.st_mode)
        or stat.S_IMODE(output_parent_info.st_mode) != 0o700
        or output_parent_info.st_uid != os.getuid()
    ):
        die("output parent identity/mode differs")
    boundary = submission.get("single_attempt_boundary")
    submission_outputs = submission.get("outputs")
    submission_info = submission_path.lstat()
    if (
        not isinstance(boundary, Mapping)
        or not isinstance(submission_outputs, Mapping)
        or submission_outputs.get("submission_receipt") != str(submission_path)
        or boundary.get("reservation_device") != int(submission_info.st_dev)
        or boundary.get("reservation_inode") != int(submission_info.st_ino)
        or boundary.get("output_parent_device") != int(output_parent_info.st_dev)
        or boundary.get("output_parent_inode") != int(output_parent_info.st_ino)
    ):
        die("submission namespace identity changed")
    resolved_release = submission["inputs"]["resolved_release"]
    release_postflight = resolved_release.get("code", {}).get("postflight")
    if not isinstance(release_postflight, Mapping):
        die("release postflight source snapshot is absent")
    postflight_source = validate_file_snapshot_row(
        release_postflight,
        label="postflight source",
    )
    if Path(__file__).resolve(strict=True) != postflight_source:
        die("executing postflight source differs from the external release")
    runtime, decoded, artifact_closure = validate_runtime_receipts(
        output, submission=submission
    )
    release_sacct = resolved_release.get("executables", {}).get("sacct")
    if (
        not isinstance(release_sacct, Mapping)
        or args.expected_sacct_sha256 != release_sacct.get("sha256")
        or args.sacct != release_sacct.get("path")
    ):
        die("sacct CLI does not match the external release manifest")
    sacct = exact_file(
        Path(args.sacct),
        label="sacct",
        expected_sha256=require_sha(args.expected_sacct_sha256, label="sacct"),
        expected_mode=0o755,
        executable=True,
    )
    job_id = str(submission["submitted_job"]["job_id"])
    accounting = observe_sacct(sacct, job_id=job_id, submission=submission)
    rendezvous_path = Path(str(submission_outputs.get("rendezvous_evidence", "")))
    if rendezvous_path != output.parent / f"{output.name}.rendezvous":
        die("rendezvous evidence path is not the derived sibling")
    rendezvous = validate_rendezvous_evidence(
        rendezvous_path,
        runtime=runtime,
        submission=submission,
    )
    terminal_logs = seal_terminal_logs(submission)
    admission = Path(args.admission)
    expected_admission = output.parent / f"{output.name}.postflight.json"
    if admission != expected_admission:
        die("postflight admission path is not the derived sibling")
    body = {
        "schema_version": POSTFLIGHT_SCHEMA,
        "status": "operationally_completed_stage_a_diagnostic_only",
        "slurm_job_id": job_id,
        "slurm_terminal_verified": True,
        "job_success": True,
        "submission": {
            "path": str(submission_path),
            "sha256": args.expected_submission_receipt_sha256,
            "receipt_digest": submission_digest,
            "release_manifest_sha256": submission["inputs"]["release_manifest"][
                "sha256"
            ],
            "release_manifest_digest": submission["inputs"][
                "release_manifest_digest"
            ],
            "exact_export_count": len(submission["exports"]),
            "exact_exports_digest": sha_bytes(canonical(submission["exports"])),
        },
        "runtime": {
            "path": str(output / "receipt.json"),
            "sha256": sha_file(output / "receipt.json"),
            "receipt_digest": runtime["receipt_digest"],
            "first_real_runtime_status": "canary_only",
        },
        "decoded_diagnostics": {
            "path": str(output / "decoded-diagnostics.json"),
            "sha256": sha_file(output / "decoded-diagnostics.json"),
            "diagnostic_digest": decoded["diagnostic_digest"],
            "semantic_action_nonregression_verdict": "unavailable",
            "dinov2_role": "identity_appearance_proxy_only",
        },
        "artifact_closure": {
            "file_count": 19,
            "files": artifact_closure,
            "digest": sha_bytes(canonical(artifact_closure)),
            "every_file_plain_mode_0444": True,
            "output_directory_mode_0555": True,
        },
        "rendezvous": rendezvous,
        "accounting": accounting,
        "terminal_logs": terminal_logs,
        "remaining_blockers": [
            "no_qualified_semantic_action_event_observer",
            "no_identity_or_appearance_authority",
            "no_stage_b_runtime",
            "no_calibrated_absolute_camera_or_quality_thresholds",
            "first_real_exact40_is_canary_not_formal_evaluation",
        ],
        "authority": POSTFLIGHT_AUTHORITY,
    }
    sealed = {**body, "receipt_digest": sha_bytes(canonical(body))}
    write_create_only(admission, sealed)
    os._exit(0)


__all__ = [
    "CELL_ORDER",
    "DIAGNOSTIC_AUTHORITY",
    "EXPECTED_NAMES",
    "POSTFLIGHT_AUTHORITY",
    "RUNTIME_AUTHORITY",
    "build_parser",
    "canonical",
    "main",
    "observe_sacct",
    "seal_terminal_logs",
    "validate_rendezvous_evidence",
    "validate_runtime_receipts",
    "validate_submission",
]


if __name__ == "__main__":
    raise SystemExit(main())

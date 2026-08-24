#!/usr/bin/env python3
"""Build/audit the exact-member 136309 reserve4 confirmation-media release.

This release can only render the sealed confirmation split: four seed cells and ten
branches per cell.  One all-eight-GPU Slurm child first runs one disposable
full-native40 WORLD4 compile smoke, then four sealed ``run-sp4`` shard runners
strictly serially, alternating the two registered XGMI4 islands.  Those formal
runners make forty WORLD4 DP1xSP4 candidate-model invocations.  Every rank uses
private node-local caches and rejects NFS COMGR temporary storage.  Each WORLD4
invocation serializes host checkpoint deserialization through one authenticated
node-local flock until the rank model is GPU-resident and host arenas are
trimmed.  A WORLD4 load-completion barrier then blocks every rank before any
source/tokenizer setup or native sampling, with dynamic ordering evidence in
the native receipt and mandatory compile-smoke replay.  The formal outputs are
confirmation-only pending-review media; the disposable smoke is never retained
or counted as confirmation evidence.  Review certification, Phi extraction, manifest
materialization, optimization, and training are forbidden by the only two
allowed entrypoints.  ``train_lora.py`` is shipped solely because the frozen
inference runtime imports it; its presence is not an OS-level impossibility
claim.  Outputs remain pending-review authoring media.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-generic-action-confirmation-data-prep-release-v3"
RELEASE_GENERATION = "r3"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-exact-modes-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
R10_PARITY_EVIDENCE: Mapping[str, Any] = {
    "schema_version": "bernini-generic-action-fit40-r10-parity-evidence-v1",
    "compile_smoke_receipt": {
        "path": (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "bernini_generic_source_anchored_action_v1_20260814/data_prep/"
            "fit40-generation-136141-r10-f5551895-r1/logs/"
            "compile-smoke-receipt.json"
        ),
        "file_sha256": (
            "e1b23a75258fac7dfcae0528c0a62c789365f683c1f096f5c2ba36ca7b1f34a3"
        ),
    },
    "generation_log": {
        "path": (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "bernini_generic_source_anchored_action_v1_20260814/data_prep/"
            "fit40-generation-136141-r10-f5551895-r1/logs/"
            "generation-fit-all8-serial4.log"
        ),
        "file_sha256": (
            "2998413eb2e37821b55dfdbfa43486063d0d97188de84680aceb9bf339a3d8dc"
        ),
    },
    "authority_derivation": (
        "unique-canonical-native-receipt-line-bound-by-r10-compile-receipt-"
        "file-sha256-and-receipt-digest"
    ),
    "tensor_values_handwritten": False,
}

_RUNTIME_FILES = (
    "infer_lora.py",
    "infer_native_identity_generation_canary.py",
    "infer_pair_v5_t2v_calibration_bank.py",
    "infer_source_kv_carrier_oracle.py",
    "infer_source_value_residual_oracle.py",
    "pair_v5_t2v_calibration_bank_spec.py",
    "source_kv_replay.py",
    "source_kv_route_batches.py",
    "source_value_residual.py",
    # Imported by the frozen inference runtime.  It is not an allowed release
    # entry point and the controller never creates an optimizer.
    "train_lora.py",
)
_TOOLS = (
    "tools/build_generic_action_confirmation_release_v1.py",
    "tools/build_pair_v5_t2v_seed2_bank.py",
    "tools/build_renderer_dataset.py",
    "tools/materialize_vae.py",
    "tools/reserve4_confirmation_generation_sp4_v1.py",
    "tools/reserve4_fixed_generation_sp4_v1.py",
)
FILES_AND_MODES: Mapping[str, int] = {
    **{path: 0o444 for path in _RUNTIME_FILES + _TOOLS},
    "generic_action_confirmation_data_prep_controller_v1.py": 0o444,
    "scripts/auh_generic_action_confirmation_data_prep_136309_world4_v1.sh": 0o555,
    "scripts/auh_generic_action_confirmation_data_prep_rank_exec_v1.sh": 0o555,
}
COMPONENT_FILES: Mapping[str, str] = {
    "controller_sha256": "generic_action_confirmation_data_prep_controller_v1.py",
    "launcher_sha256": "scripts/auh_generic_action_confirmation_data_prep_136309_world4_v1.sh",
    "generator_sha256": "tools/reserve4_confirmation_generation_sp4_v1.py",
    "resource_contract_sha256": "tools/reserve4_fixed_generation_sp4_v1.py",
    "rank_cache_wrapper_sha256": (
        "scripts/auh_generic_action_confirmation_data_prep_rank_exec_v1.sh"
    ),
}
ENTRYPOINTS = (
    "generic_action_confirmation_data_prep_controller_v1.py",
    "scripts/auh_generic_action_confirmation_data_prep_136309_world4_v1.sh",
)


class GenericActionDataReleaseError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise GenericActionDataReleaseError(message)


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
        raise GenericActionDataReleaseError(
            "release is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_plain_bytes(path: Path) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        fail("release input must be absolute and non-symlink")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if resolved != path or not stat.S_ISREG(before.st_mode):
        fail("release input must be one canonical plain file")
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
        or not raw
    ):
        fail("release input changed while reading or is empty")
    return raw


def build_manifest(
    method_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    root = method_root.resolve(strict=True)
    if root != method_root or root.is_symlink() or not root.is_dir():
        fail("method root must be one canonical directory")
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for relative in sorted(FILES_AND_MODES):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            fail("release member path differs")
        raw = _stable_plain_bytes(root / relative)
        payloads[relative] = raw
        rows.append(
            {
                "path": relative,
                "mode": FILES_AND_MODES[relative],
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    row_by_path = {str(row["path"]): row for row in rows}
    component_pins = {
        label: row_by_path[relative]["sha256"]
        for label, relative in COMPONENT_FILES.items()
    }
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "release_generation": RELEASE_GENERATION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "file_count": len(rows),
        "files": rows,
        "component_pins": component_pins,
        "allowed_entrypoints": list(ENTRYPOINTS),
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(
            canonical_json_bytes(closure)
        ).hexdigest(),
        "exact_member_closure": True,
        "release_scope": "reserve4-confirmation40-media-only-pending-external-blind-review",
        "external_evidence": R10_PARITY_EVIDENCE,
        "topology": {
            "holder": {"job_id": 136309, "node": "auh7-1b-gpu-280"},
            "fresh_run_root_required": True,
            "consecutive_master_port_count": 4,
            "master_port_inclusive_range": [1024, 65532],
            "compute_world_size": 4,
            "parallelism": "dp1_sp4_one_model_replica_at_a_time",
            "slurm_child_gpu_count": 8,
            "numbered_slurm_children": 1,
            "run_sp4_shard_process_count": 4,
            "world4_model_invocation_count": 40,
            "compile_smoke_world4_model_invocation_count": 1,
            "compile_smoke_full_native_sampling_steps": 40,
            "total_native_model_invocation_count": 41,
            "disposable_single_candidate_compile_smoke_before_formal40": True,
            "compile_smoke_candidate_not_counted_in_formal40": True,
            "formal_generation_requires_exact_compile_smoke_receipt": True,
            "per_rank_node_local_cache_wrapper": True,
            "nfs_comgr_tmp_rejected": True,
            "serialized_world4_host_checkpoint_load": True,
            "model_load_lock_node_local": True,
            "model_load_lock_held_through_gpu_move_and_malloc_trim": True,
            "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup": True,
            "resource_lifecycle_receipt_proves_all_four_load_complete_before_first_sampling": True,
            "compile_smoke_asserts_world4_load_completion_ordering": True,
            "t2v_text_encoder_rank_gpu_residency_required": True,
            "t2v_text_encoder_exact_cpu_offload_suppressed_once_per_rank": True,
            "t2v_text_encoder_retired_only_with_renderer": True,
            "t2v_rank_gpu_memory_limit_gib": 52,
            "compile_smoke_per_rank_gpu_peak_allocated_reserved_required": True,
            "compile_smoke_all_rank_gpu_peak_reserved_strictly_below_limit": True,
            "host_cgroup_sample_monitor_started_before_compile_smoke": True,
            "host_cgroup_current_pid_leaf_identity_bound": True,
            "host_cgroup_leaf_memory_max_inherited": True,
            "host_cgroup_governing_ancestor_nearest_finite": True,
            "host_cgroup_governing_scope_exact_slurm_step_user": True,
            "host_cgroup_sample_interval_ns": 10_000_000,
            "host_cgroup_max_sample_gap_ns": 100_000_000,
            "host_live_tail_max_age_ns": 100_000_000,
            "host_cgroup_memory_max_exactly_60_gib": True,
            "host_sampled_current_safe_ceiling_gib": 56,
            "compile_smoke_host_sampled_peak_strictly_below_56_gib": True,
            "compile_smoke_host_monitor_alive_before_formal40": True,
            "compile_smoke_zero_oom_and_oom_kill_before_formal40": True,
            "formal_candidate_boundary_host_monitor_checks_required": True,
            "formal_candidate_boundary_checks_run_inside_rank_wrapper": True,
            "terminal_host_sampled_current_receipt_required": True,
            "terminal_host_sampled_peak_strictly_below_56_gib": True,
            "terminal_host_monitor_wait_exit_status_zero": True,
            "terminal_gate_created_after_bound_supervisor_wait": True,
            "terminal_host_monitor_clean_exit": True,
            "terminal_zero_oom_and_oom_kill": True,
            "r10_smoke_authority_derived_from_pinned_receipt_and_log": True,
            "r10_smoke_mp4_gaussian_latent_byte_parity_required": True,
            "r10_smoke_mp4_whole_file_sha256_exact_required": True,
            "r10_smoke_gaussian_tensor_identity_exact_required": True,
            "r10_smoke_clean_latent_generated_identity_exact_required": True,
            "current_smoke_physical_safetensors_safe_open_required": True,
            "current_smoke_exact_single_tensor_key_required": True,
            "current_smoke_exact_safetensors_metadata_required": True,
            "current_smoke_tensor_identity_recomputed_from_physical_values": True,
            "current_smoke_physical_identity_bound_to_receipt_all_rank_generated_and_r10": True,
            "confirmation_smoke_supplemental_physical_receipt_create_only": True,
            "confirmation_smoke_supplemental_receipt_bound_before_disposable_delete": True,
            "safetensors_container_sha256_cross_process_equivalence_required": False,
            "t2v_vae_load_deferred_until_rank0_post_sampling": True,
            "world4_renderer_retirement_barrier_before_rank_zero_vae_load": True,
            "sealed_shard_order": [
                "seed1-sp4-a-confirmation",
                "seed1-sp4-b-confirmation",
                "seed2-sp4-a-confirmation",
                "seed2-sp4-b-confirmation",
            ],
            "physical_island_order": [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ],
            "all_model_invocations_strictly_serial": True,
            "per_shard_observed_uuid_pci_bus_join_before_model_forward": True,
            "physical_index_pci_unique_id_join_replayed": True,
            "pci_bus_is_authoritative_join_key": True,
            "hip_logical_order_is_observation_only": True,
            "logical_to_physical_mapping_uses_pci_and_unique_id": True,
            "per_shard_exact_physical_set_verified": True,
            "within_island_logical_permutation_allowed": True,
            "cross_island_shard_visibility_rejected": True,
            "xgmi_adjacency_exact": True,
            "generation_directory_exact_member_closure": True,
            "concurrent_model_replicas": 1,
            "rank_or_gpu_action_family_partition": False,
            "all_rows_share_one_generic_representation_contract": True,
            "host_memory_request_gib": 60,
            "non_authoritative_dynamic_probe_observation": {
                "probe_receipt_pinned": False,
                "is_release_authority": False,
                "four_gpu_mask_0xf0_was_usable": False,
                "all8_allocation_with_rocr_4_7_reported_torch_device_count": 4,
                "formal_launcher_revalidates_before_model_forward": True,
            },
        },
        "authority": {
            "analysis_split": "confirmation",
            "candidate_count": 40,
            "seed_cell_count": 4,
            "confirmation_iids": ["0c6915018a5f4d9b", "33322eb8ec1e4703"],
            "confirmation_seed_cells": [
                ["0c6915018a5f4d9b", 2026080822, "sp4-a"],
                ["33322eb8ec1e4703", 2026080823, "sp4-b"],
                ["0c6915018a5f4d9b", 2026080922, "sp4-a"],
                ["33322eb8ec1e4703", 2026080923, "sp4-b"],
            ],
            "branch_order": [
                "action", "noop", "incomplete", "reverse", "shuffle",
                "wrong_actor", "wrong_object", "camera_only",
                "appearance_only", "generic_wrong_motion",
            ],
            "authoring_registry_raw_sha256": "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c",
            "reserve4_selection_raw_sha256": "a4baa1aea27f6497ca2dd615cc09b2b90eee37173f506e60ae7d630c41886be6",
            "seed1_spec_raw_sha256": "2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab",
            "seed2_spec_raw_sha256": "0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e",
            "existing_core4_confirmation_media_included": False,
            "future_external_blind_review_population_candidate_count": 80,
            "independent_full81_blind_review_present": False,
            "same_runner_may_self_certify_visual_review": False,
            "generation_runner_has_review_authority": False,
            "pending_external_blind_review": True,
            "generated_media_role": "pending-external-blind-review-confirmation-authoring-media-only",
            "generated_rgb_latent_gaussian_is_editor_input_or_target": False,
            "confirmation_generation_authorized": True,
            "confirmation_scope_only": True,
            "disposable_smoke_is_confirmation_evidence": False,
            "dynamic_probe_receipt_pinned": False,
            "dynamic_probe_is_release_authority": False,
            "phi_materializer_present": False,
            "phi_v1_extraction_authorized": False,
            "optimizer_created": False,
            "optimizer_authorized": False,
            "p_or_o_manifest_materialization_authorized": False,
            "training_authorized": False,
        },
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}, payloads


def build_archive(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for row in manifest["files"]:
            relative = str(row["path"])
            raw = payloads[relative]
            member = tarfile.TarInfo(f"{MEMBER_ROOT}/{relative}")
            member.size = len(raw)
            member.mode = int(row["mode"])
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            member.type = tarfile.REGTYPE
            tar.addfile(member, io.BytesIO(raw))
    return output.getvalue()


def verify_archive(raw: bytes, manifest: Mapping[str, Any]) -> None:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        fail("release manifest files differ")
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in rows]
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            members = tar.getmembers()
            if [member.name for member in members] != expected:
                fail("release archive exact member order or set differs")
            for member, row in zip(members, rows):
                handle = tar.extractfile(member)
                payload = b"" if handle is None else handle.read()
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.mode != row["mode"]
                    or member.size != row["size"]
                    or member.pax_headers
                    or hashlib.sha256(payload).hexdigest() != row["sha256"]
                ):
                    fail(f"release archive member differs: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise GenericActionDataReleaseError(
            f"cannot verify release archive: {error}"
        ) from error


def validate_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    unsigned = dict(value)
    declared = unsigned.pop("manifest_digest", None)
    rows = value.get("files")
    expected_paths = sorted(FILES_AND_MODES)
    authority = value.get("authority")
    topology = value.get("topology")
    if (
        set(value)
        != {
            "schema_version", "release_generation", "archive_format",
            "member_root", "file_count", "files", "component_pins",
            "allowed_entrypoints", "revision_kind", "content_closure_sha1",
            "exact_member_closure", "release_scope", "external_evidence",
            "topology", "authority",
            "manifest_digest",
        }
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("release_generation") != RELEASE_GENERATION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("member_root") != MEMBER_ROOT
        or value.get("file_count") != len(FILES_AND_MODES)
        or value.get("exact_member_closure") is not True
        or value.get("release_scope")
        != "reserve4-confirmation40-media-only-pending-external-blind-review"
        or value.get("allowed_entrypoints") != list(ENTRYPOINTS)
        or value.get("external_evidence") != R10_PARITY_EVIDENCE
        or declared != object_sha256(unsigned)
        or not isinstance(rows, list)
        or [row.get("path") if isinstance(row, Mapping) else None for row in rows]
        != expected_paths
    ):
        fail("release manifest schema/digest/closure differs")
    if topology != {
        "holder": {"job_id": 136309, "node": "auh7-1b-gpu-280"},
        "fresh_run_root_required": True,
        "consecutive_master_port_count": 4,
        "master_port_inclusive_range": [1024, 65532],
        "compute_world_size": 4,
        "parallelism": "dp1_sp4_one_model_replica_at_a_time",
        "slurm_child_gpu_count": 8,
        "numbered_slurm_children": 1,
        "run_sp4_shard_process_count": 4,
        "world4_model_invocation_count": 40,
        "compile_smoke_world4_model_invocation_count": 1,
        "compile_smoke_full_native_sampling_steps": 40,
        "total_native_model_invocation_count": 41,
        "disposable_single_candidate_compile_smoke_before_formal40": True,
        "compile_smoke_candidate_not_counted_in_formal40": True,
        "formal_generation_requires_exact_compile_smoke_receipt": True,
        "per_rank_node_local_cache_wrapper": True,
        "nfs_comgr_tmp_rejected": True,
        "serialized_world4_host_checkpoint_load": True,
        "model_load_lock_node_local": True,
        "model_load_lock_held_through_gpu_move_and_malloc_trim": True,
        "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup": True,
        "resource_lifecycle_receipt_proves_all_four_load_complete_before_first_sampling": True,
        "compile_smoke_asserts_world4_load_completion_ordering": True,
        "t2v_text_encoder_rank_gpu_residency_required": True,
        "t2v_text_encoder_exact_cpu_offload_suppressed_once_per_rank": True,
        "t2v_text_encoder_retired_only_with_renderer": True,
        "t2v_rank_gpu_memory_limit_gib": 52,
        "compile_smoke_per_rank_gpu_peak_allocated_reserved_required": True,
        "compile_smoke_all_rank_gpu_peak_reserved_strictly_below_limit": True,
        "host_cgroup_sample_monitor_started_before_compile_smoke": True,
        "host_cgroup_current_pid_leaf_identity_bound": True,
        "host_cgroup_leaf_memory_max_inherited": True,
        "host_cgroup_governing_ancestor_nearest_finite": True,
        "host_cgroup_governing_scope_exact_slurm_step_user": True,
        "host_cgroup_sample_interval_ns": 10_000_000,
        "host_cgroup_max_sample_gap_ns": 100_000_000,
        "host_live_tail_max_age_ns": 100_000_000,
        "host_cgroup_memory_max_exactly_60_gib": True,
        "host_sampled_current_safe_ceiling_gib": 56,
        "compile_smoke_host_sampled_peak_strictly_below_56_gib": True,
        "compile_smoke_host_monitor_alive_before_formal40": True,
        "compile_smoke_zero_oom_and_oom_kill_before_formal40": True,
        "formal_candidate_boundary_host_monitor_checks_required": True,
        "formal_candidate_boundary_checks_run_inside_rank_wrapper": True,
        "terminal_host_sampled_current_receipt_required": True,
        "terminal_host_sampled_peak_strictly_below_56_gib": True,
        "terminal_host_monitor_wait_exit_status_zero": True,
        "terminal_gate_created_after_bound_supervisor_wait": True,
        "terminal_host_monitor_clean_exit": True,
        "terminal_zero_oom_and_oom_kill": True,
        "r10_smoke_authority_derived_from_pinned_receipt_and_log": True,
        "r10_smoke_mp4_gaussian_latent_byte_parity_required": True,
        "r10_smoke_mp4_whole_file_sha256_exact_required": True,
        "r10_smoke_gaussian_tensor_identity_exact_required": True,
        "r10_smoke_clean_latent_generated_identity_exact_required": True,
        "current_smoke_physical_safetensors_safe_open_required": True,
        "current_smoke_exact_single_tensor_key_required": True,
        "current_smoke_exact_safetensors_metadata_required": True,
        "current_smoke_tensor_identity_recomputed_from_physical_values": True,
        "current_smoke_physical_identity_bound_to_receipt_all_rank_generated_and_r10": True,
        "confirmation_smoke_supplemental_physical_receipt_create_only": True,
        "confirmation_smoke_supplemental_receipt_bound_before_disposable_delete": True,
        "safetensors_container_sha256_cross_process_equivalence_required": False,
        "t2v_vae_load_deferred_until_rank0_post_sampling": True,
        "world4_renderer_retirement_barrier_before_rank_zero_vae_load": True,
        "sealed_shard_order": [
            "seed1-sp4-a-confirmation",
            "seed1-sp4-b-confirmation",
            "seed2-sp4-a-confirmation",
            "seed2-sp4-b-confirmation",
        ],
        "physical_island_order": [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        "all_model_invocations_strictly_serial": True,
        "per_shard_observed_uuid_pci_bus_join_before_model_forward": True,
        "physical_index_pci_unique_id_join_replayed": True,
        "pci_bus_is_authoritative_join_key": True,
        "hip_logical_order_is_observation_only": True,
        "logical_to_physical_mapping_uses_pci_and_unique_id": True,
        "per_shard_exact_physical_set_verified": True,
        "within_island_logical_permutation_allowed": True,
        "cross_island_shard_visibility_rejected": True,
        "xgmi_adjacency_exact": True,
        "generation_directory_exact_member_closure": True,
        "concurrent_model_replicas": 1,
        "rank_or_gpu_action_family_partition": False,
        "all_rows_share_one_generic_representation_contract": True,
        "host_memory_request_gib": 60,
        "non_authoritative_dynamic_probe_observation": {
            "probe_receipt_pinned": False,
            "is_release_authority": False,
            "four_gpu_mask_0xf0_was_usable": False,
            "all8_allocation_with_rocr_4_7_reported_torch_device_count": 4,
            "formal_launcher_revalidates_before_model_forward": True,
        },
    }:
        fail("release topology differs")
    if authority != {
        "analysis_split": "confirmation",
        "candidate_count": 40,
        "seed_cell_count": 4,
        "confirmation_iids": ["0c6915018a5f4d9b", "33322eb8ec1e4703"],
        "confirmation_seed_cells": [
            ["0c6915018a5f4d9b", 2026080822, "sp4-a"],
            ["33322eb8ec1e4703", 2026080823, "sp4-b"],
            ["0c6915018a5f4d9b", 2026080922, "sp4-a"],
            ["33322eb8ec1e4703", 2026080923, "sp4-b"],
        ],
        "branch_order": [
            "action", "noop", "incomplete", "reverse", "shuffle",
            "wrong_actor", "wrong_object", "camera_only",
            "appearance_only", "generic_wrong_motion",
        ],
        "authoring_registry_raw_sha256": "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c",
        "reserve4_selection_raw_sha256": "a4baa1aea27f6497ca2dd615cc09b2b90eee37173f506e60ae7d630c41886be6",
        "seed1_spec_raw_sha256": "2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab",
        "seed2_spec_raw_sha256": "0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e",
        "existing_core4_confirmation_media_included": False,
        "future_external_blind_review_population_candidate_count": 80,
        "independent_full81_blind_review_present": False,
        "same_runner_may_self_certify_visual_review": False,
        "generation_runner_has_review_authority": False,
        "pending_external_blind_review": True,
        "generated_media_role": "pending-external-blind-review-confirmation-authoring-media-only",
        "generated_rgb_latent_gaussian_is_editor_input_or_target": False,
        "confirmation_generation_authorized": True,
        "confirmation_scope_only": True,
        "disposable_smoke_is_confirmation_evidence": False,
        "dynamic_probe_receipt_pinned": False,
        "dynamic_probe_is_release_authority": False,
        "phi_materializer_present": False,
        "phi_v1_extraction_authorized": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "p_or_o_manifest_materialization_authorized": False,
        "training_authorized": False,
    }:
        fail("release authority differs")
    row_by_path = {str(row["path"]): row for row in rows}
    for relative, expected_mode in FILES_AND_MODES.items():
        row = row_by_path[relative]
        if (
            set(row) != {"path", "mode", "size", "sha256"}
            or row["mode"] != expected_mode
            or type(row["size"]) is not int
            or row["size"] <= 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            fail(f"release member row differs: {relative}")
    expected_pins = {
        label: row_by_path[relative]["sha256"]
        for label, relative in COMPONENT_FILES.items()
    }
    if value.get("component_pins") != expected_pins:
        fail("release component pins differ")
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    if (
        value.get("revision_kind") != "content-closure-sha1"
        or value.get("content_closure_sha1")
        != hashlib.sha1(canonical_json_bytes(closure)).hexdigest()
    ):
        fail("release content closure differs")
    return value


def _write_create_only(path: Path, raw: bytes, *, mode: int) -> None:
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        fail("release output must be one fresh absolute file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(method_root: Path, archive: Path, manifest_path: Path) -> Mapping[str, Any]:
    manifest, payloads = build_manifest(method_root)
    validate_manifest(manifest)
    archive_raw = build_archive(manifest, payloads)
    verify_archive(archive_raw, manifest)
    if build_archive(manifest, payloads) != archive_raw:
        fail("release archive rebuild is not byte-identical")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(archive, archive_raw, mode=0o444)
    _write_create_only(manifest_path, manifest_raw, mode=0o444)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "component_pins": manifest["component_pins"],
        "file_count": len(FILES_AND_MODES),
        "optimizer_authorized": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def audit(
    archive: Path,
    manifest_path: Path,
    *,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    archive_raw = _stable_plain_bytes(archive)
    manifest_raw = _stable_plain_bytes(manifest_path)
    if hashlib.sha256(archive_raw).hexdigest() != expected_archive_sha256:
        fail("release archive SHA-256 differs")
    if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha256:
        fail("release manifest SHA-256 differs")
    try:
        manifest = json.loads(manifest_raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GenericActionDataReleaseError(
            f"cannot decode release manifest: {error}"
        ) from error
    if (
        not isinstance(manifest, Mapping)
        or manifest_raw != canonical_json_bytes(manifest) + b"\n"
    ):
        fail("release manifest bytes are not canonical JSON")
    validate_manifest(manifest)
    verify_archive(archive_raw, manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--method-root", required=True)
    create.add_argument("--archive", required=True)
    create.add_argument("--manifest", required=True)
    check = commands.add_parser("audit")
    check.add_argument("--archive", required=True)
    check.add_argument("--manifest", required=True)
    check.add_argument("--expected-archive-sha256", required=True)
    check.add_argument("--expected-manifest-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        receipt = build(
            Path(args.method_root), Path(args.archive), Path(args.manifest)
        )
        print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
    else:
        manifest = audit(
            Path(args.archive),
            Path(args.manifest),
            expected_archive_sha256=args.expected_archive_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
        print(canonical_json_bytes(manifest).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

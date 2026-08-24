#!/usr/bin/env python3
"""Fail-closed 136141 controller for sealed reserve4 fit media only.

The only runnable profile renders the frozen fit split: two registered seed
specs and four complete ten-branch cells.  One all-eight-GPU Slurm child runs
one disposable full-native40 compile smoke followed by four sealed ``run-sp4``
shard runners in order, one at a time, on the two physical XGMI4 islands.
The formal shard runners make forty serial WORLD4 DP1xSP4 candidate-model
invocations.  Per-rank caches are private node-local directories and WORLD4
checkpoint deserialization is rank-serialized through an authenticated
node-local flock until each model is GPU-resident and host arenas are trimmed.
A mandatory WORLD4 load-completion barrier then holds every rank before any
source/tokenizer setup or native sampling begins.
Outputs are pending-external-review
authoring media.  The shipped ``train_lora.py`` is an inference import
dependency; only the controller and launcher are allowed entrypoints, and
those entrypoints forbid confirmation, review certification, Phi extraction,
P/O manifests, optimizers, and training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


sys.dont_write_bytecode = True
METHOD_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_generic_action_data_prep_release_v1 as release  # noqa: E402
import reserve4_fixed_generation_sp4_v1 as reserve4_runner  # noqa: E402


PLAN_SCHEMA = "bernini-generic-action-fit40-generation-136141-plan-v13"
COMPLETION_SCHEMA = "bernini-generic-action-fit40-generation-136141-completion-v13"
PHYSICAL_INVENTORY_SCHEMA = "bernini-generic-action-rocm-physical-inventory-v1"
RUNTIME_MAPPING_SCHEMA = "bernini-generic-action-rocm-runtime-mapping-v3"
BINDING_RECEIPT_SCHEMA = (
    "bernini-generic-action-fit40-shard-physical-binding-v4"
)
GPU_ADMISSION_RECEIPT_SCHEMA = (
    "bernini-generic-action-fit40-gpu-host-admission-receipt-v10"
)
GENERATION_CLOSURE_SCHEMA = (
    "bernini-generic-action-fit40-generation-closure-receipt-v6"
)
HOLDER_JOB = 136141
HOLDER_NODE = "auh7-1b-gpu-299"
WORLD_SIZE = 4
HOST_MEMORY_LIMIT_GIB = reserve4_runner.HOST_MEMORY_LIMIT_GIB
HOST_MEMORY_LIMIT_BYTES = reserve4_runner.HOST_MEMORY_LIMIT_BYTES
HOST_MEMORY_SAFE_CEILING_GIB = reserve4_runner.HOST_MEMORY_SAFE_CEILING_GIB
HOST_MEMORY_SAFE_CEILING_BYTES = reserve4_runner.HOST_MEMORY_SAFE_CEILING_BYTES
HOST_MEMORY_SAMPLE_INTERVAL_NS = reserve4_runner.HOST_MEMORY_SAMPLE_INTERVAL_NS
HOST_MEMORY_MAX_SAMPLE_GAP_NS = reserve4_runner.HOST_MEMORY_MAX_SAMPLE_GAP_NS
T2V_GPU_MEMORY_LIMIT_GIB = 52
T2V_GPU_MEMORY_LIMIT_BYTES = T2V_GPU_MEMORY_LIMIT_GIB * 1024**3
HOST_CGROUP_MEMORY_GATE_SCHEMA = reserve4_runner.HOST_CGROUP_MEMORY_GATE_SCHEMA
LAUNCH_CONFIRMATION = "launch-approved-generic-action-fit40-generation-136141"
R10_PARITY_EVIDENCE = release.R10_PARITY_EVIDENCE
SEED1_SPEC_SHA256 = "2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab"
SEED2_SPEC_SHA256 = "0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e"
BERNINI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/"
    "phase_a_native_gpu_canary_dual4_all8_v1/releases/"
    "source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591"
)
VEOMNI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/"
    "phase_a_native_gpu_canary_dual4_all8_v1/releases/"
    "source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6"
)
CHECKPOINT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
CHECKPOINT_MANIFEST = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_"
    "20260808_74ed30c/runtime/methods/bernini_action_editing/audits/"
    "bernini_r13_ff4c5d4_checkpoint.sha256"
)
CHECKPOINT_MANIFEST_SHA256 = "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
UUID_HEX_RE = re.compile(r"[0-9a-f]{32}")
PCI_BUS_RE = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
ROCM_UNIQUE_ID_RE = re.compile(r"[0-9a-f]{8,64}")
SHARD_ORDER = [
    "seed1-sp4-a-fit",
    "seed1-sp4-b-fit",
    "seed2-sp4-a-fit",
    "seed2-sp4-b-fit",
]
PHYSICAL_ISLAND_ORDER = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 2, 3],
    [4, 5, 6, 7],
]
NON_AUTHORITATIVE_PROBE_OBSERVATION = {
    "probe_receipt_pinned": False,
    "is_release_authority": False,
    "four_gpu_mask_0xf0_was_usable": False,
    "all8_allocation_with_rocr_4_7_reported_torch_device_count": 4,
    "formal_launcher_revalidates_before_model_forward": True,
}
TOPOLOGY_CONTRACT = {
    "compute_world_size": 4,
    "parallelism": "dp1_sp4",
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
    "t2v_rank_gpu_memory_limit_gib": T2V_GPU_MEMORY_LIMIT_GIB,
    "compile_smoke_per_rank_gpu_peak_allocated_reserved_required": True,
    "compile_smoke_all_rank_gpu_peak_reserved_strictly_below_limit": True,
    "host_cgroup_sample_monitor_started_before_compile_smoke": True,
    "host_cgroup_current_pid_leaf_identity_bound": True,
    "host_cgroup_leaf_memory_max_inherited": True,
    "host_cgroup_governing_ancestor_nearest_finite": True,
    "host_cgroup_governing_scope_exact_slurm_step_user": True,
    "host_cgroup_sample_interval_ns": HOST_MEMORY_SAMPLE_INTERVAL_NS,
    "host_cgroup_max_sample_gap_ns": HOST_MEMORY_MAX_SAMPLE_GAP_NS,
    "host_live_tail_max_age_ns": HOST_MEMORY_MAX_SAMPLE_GAP_NS,
    "host_cgroup_memory_max_exactly_60_gib": True,
    "host_sampled_current_safe_ceiling_gib": HOST_MEMORY_SAFE_CEILING_GIB,
    "compile_smoke_host_sampled_peak_strictly_below_56_gib": True,
    "compile_smoke_host_monitor_alive_before_formal40": True,
    "compile_smoke_zero_oom_and_oom_kill_before_formal40": True,
    "formal_candidate_boundary_host_monitor_checks_required": True,
    "terminal_host_sampled_current_receipt_required": True,
    "terminal_host_sampled_peak_strictly_below_56_gib": True,
    "terminal_host_monitor_wait_exit_status_zero": True,
    "terminal_gate_created_after_bound_supervisor_wait": True,
    "terminal_host_monitor_clean_exit": True,
    "terminal_zero_oom_and_oom_kill": True,
    "r10_smoke_authority_derived_from_pinned_receipt_and_log": True,
    "r10_smoke_mp4_whole_file_sha256_exact_required": True,
    "r10_smoke_gaussian_tensor_identity_exact_required": True,
    "r10_smoke_clean_latent_generated_identity_exact_required": True,
    "current_smoke_physical_safetensors_safe_open_required": True,
    "current_smoke_exact_single_tensor_key_required": True,
    "current_smoke_exact_safetensors_metadata_required": True,
    "current_smoke_tensor_identity_recomputed_from_physical_values": True,
    "current_smoke_physical_identity_bound_to_receipt_all_rank_generated_and_r10": True,
    "safetensors_container_sha256_cross_process_equivalence_required": False,
    "t2v_vae_load_deferred_until_rank0_post_sampling": True,
    "world4_renderer_retirement_barrier_before_rank_zero_vae_load": True,
    "sealed_shard_order": SHARD_ORDER,
    "physical_island_order": PHYSICAL_ISLAND_ORDER,
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
    "host_memory_request_gib": HOST_MEMORY_LIMIT_GIB,
}
OUTPUT_CONTRACT = {
    "candidate_count": 40,
    "seed_cell_count": 4,
    "numbered_slurm_children": 1,
    "run_sp4_shard_process_count": 4,
    "world4_model_invocation_count": 40,
    "compile_smoke_candidate_count": 1,
    "compile_smoke_candidate_retained": False,
    "total_native_model_invocation_count": 41,
    "shard_order": SHARD_ORDER,
}
AUTHORITY_CONTRACT = {
    "independent_full81_blind_review_present": False,
    "generation_runner_may_self_certify_visual_review": False,
    "generated_media_role": "pending-external-review-authoring-media-only",
    "generated_rgb_latent_gaussian_is_editor_input_or_target": False,
    "confirmation_generation_authorized": False,
    "dynamic_probe_receipt_pinned": False,
    "dynamic_probe_is_release_authority": False,
    "phi_v1_extraction_authorized": False,
    "optimizer_created": False,
    "optimizer_authorized": False,
    "p_or_o_manifest_materialization_authorized": False,
    "training_authorized": False,
}


class GenericActionDataPrepError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise GenericActionDataPrepError(message)


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
        raise GenericActionDataPrepError(
            "controller value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        fail(f"input changed while hashing: {path}")
    return digest.hexdigest()


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be one lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value)
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise GenericActionDataPrepError(f"{label} is unavailable") from error
    if resolved != requested or not stat.S_ISREG(resolved.lstat().st_mode):
        fail(f"{label} must be one canonical plain file")
    return resolved


def _plain_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value)
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise GenericActionDataPrepError(f"{label} is unavailable") from error
    if resolved != requested or not stat.S_ISDIR(resolved.lstat().st_mode):
        fail(f"{label} must be one canonical directory")
    return resolved


def _fresh_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value)
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or requested.exists()
        or requested.is_symlink()
        or requested.parent.resolve(strict=True) != requested.parent
    ):
        fail(f"{label} must be one fresh canonical path")
    return requested


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    source = _plain_file(path, label=label)
    raw = source.read_bytes()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GenericActionDataPrepError(f"non-finite JSON: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GenericActionDataPrepError(f"cannot decode {label}") from error
    if not isinstance(value, Mapping):
        fail(f"{label} must be a JSON object")
    return value


def _load_canonical_receipt(
    path: Path,
    *,
    label: str,
    digest_field: str = "receipt_digest",
) -> Mapping[str, Any]:
    """Load a receipt only if its bytes, keys, and self digest replay exactly."""

    source = _plain_file(path, label=label)
    raw = source.read_bytes()
    value = _load_json(source, label=label)
    if raw != canonical_json_bytes(value) + b"\n":
        fail(f"{label} bytes are not canonical JSON")
    unsigned = dict(value)
    declared = unsigned.pop(digest_field, None)
    if declared != object_sha256(unsigned):
        fail(f"{label} self digest differs")
    return value


def normalize_pci_bus_id(value: Any) -> str:
    if not isinstance(value, str):
        fail("PCI bus identifier must be text")
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", normalized):
        normalized = f"0000:{normalized}"
    if PCI_BUS_RE.fullmatch(normalized) is None:
        fail(f"PCI bus identifier differs: {value!r}")
    return normalized


def normalize_rocm_unique_id(value: Any) -> str:
    if not isinstance(value, str):
        fail("ROCm unique ID must be text")
    normalized = value.strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    normalized = normalized.replace("-", "")
    if ROCM_UNIQUE_ID_RE.fullmatch(normalized) is None or set(normalized) == {"0"}:
        fail(f"ROCm unique ID differs: {value!r}")
    return normalized


def _write_create_only(path: Path, value: Mapping[str, Any], mode: int = 0o400) -> str:
    if (
        not path.is_absolute()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.exists()
        or path.is_symlink()
    ):
        fail(f"refusing non-fresh output: {path}")
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


def _parse_rocm_physical_inventory(
    identity_text: str, topology_text: str
) -> list[dict[str, Any]]:
    """Parse physical-indexed ROCm-SMI identity and XGMI topology output."""

    identity_rows: dict[int, dict[str, str]] = {
        index: {} for index in range(8)
    }
    identity_pattern = re.compile(
        r"GPU\[(?P<index>[0-7])\].*?"
        r"(?P<label>Unique\s+ID|PCI\s+Bus)\s*:\s*(?P<value>\S+)",
        re.IGNORECASE,
    )
    for line in identity_text.splitlines():
        match = identity_pattern.search(line)
        if match is None:
            continue
        index = int(match.group("index"))
        label = re.sub(r"\s+", " ", match.group("label").lower())
        field = "unique_id" if label == "unique id" else "pci_bus_id"
        if field in identity_rows[index]:
            fail(f"duplicate ROCm-SMI {field} row for physical GPU {index}")
        identity_rows[index][field] = match.group("value")

    xgmi_neighbors: dict[int, list[int]] = {}
    matrix_pattern = re.compile(r"^\s*GPU\[?([0-7])\]?\s+(.*)$", re.IGNORECASE)
    allowed_cells = {"0", "XGMI", "PCIE"}
    for line in topology_text.splitlines():
        match = matrix_pattern.match(line)
        if match is None:
            continue
        cells = match.group(2).split()
        if len(cells) < 8 or any(cell.upper() not in allowed_cells for cell in cells[:8]):
            continue
        index = int(match.group(1))
        if index in xgmi_neighbors:
            fail(f"duplicate ROCm-SMI topology matrix row for GPU {index}")
        xgmi_neighbors[index] = [
            peer for peer, cell in enumerate(cells[:8]) if cell.upper() == "XGMI"
        ]

    numa_nodes: dict[int, int] = {}
    numa_pattern = re.compile(
        r"GPU\[([0-7])\].*?Numa\s+Node\s*:\s*([0-9]+)", re.IGNORECASE
    )
    for line in topology_text.splitlines():
        match = numa_pattern.search(line)
        if match is None:
            continue
        index, node = int(match.group(1)), int(match.group(2))
        if index in numa_nodes:
            fail(f"duplicate ROCm-SMI NUMA row for GPU {index}")
        numa_nodes[index] = node

    if (
        set(identity_rows) != set(range(8))
        or any(set(row) != {"unique_id", "pci_bus_id"} for row in identity_rows.values())
        or set(xgmi_neighbors) != set(range(8))
        or set(numa_nodes) != set(range(8))
    ):
        fail("ROCm-SMI physical identity/topology inventory is not exact8")

    expected_neighbors = {
        index: [peer for peer in range(4) if peer != index]
        for index in range(4)
    }
    expected_neighbors.update(
        {
            index: [peer for peer in range(4, 8) if peer != index]
            for index in range(4, 8)
        }
    )
    if xgmi_neighbors != expected_neighbors:
        fail("ROCm-SMI XGMI adjacency is not the two exact physical XGMI4 islands")
    if [numa_nodes[index] for index in range(8)] != [0] * 4 + [1] * 4:
        fail("ROCm-SMI NUMA membership is not the two exact physical islands")

    rows = []
    for physical_index in range(8):
        rows.append(
            {
                "physical_index": physical_index,
                "pci_bus_id": normalize_pci_bus_id(
                    identity_rows[physical_index]["pci_bus_id"]
                ),
                "rocm_unique_id": normalize_rocm_unique_id(
                    identity_rows[physical_index]["unique_id"]
                ),
                "xgmi_neighbors": xgmi_neighbors[physical_index],
                "numa_node": numa_nodes[physical_index],
            }
        )
    if (
        len({row["pci_bus_id"] for row in rows}) != 8
        or len({row["rocm_unique_id"] for row in rows}) != 8
    ):
        fail("ROCm-SMI physical PCI buses or unique IDs are not distinct exact8")
    return rows


def seal_rocm_physical_inventory(
    identity_input: Path, topology_input: Path, output: Path
) -> Mapping[str, Any]:
    identity_path = _plain_file(identity_input, label="ROCm-SMI identity source")
    topology_path = _plain_file(topology_input, label="ROCm-SMI topology source")
    try:
        identity_text = identity_path.read_text(encoding="ascii")
        topology_text = topology_path.read_text(encoding="ascii")
    except UnicodeError as error:
        raise GenericActionDataPrepError("ROCm-SMI inventory is not ASCII") from error
    rows = _parse_rocm_physical_inventory(identity_text, topology_text)
    unsigned = {
        "schema_version": PHYSICAL_INVENTORY_SCHEMA,
        "identity_source": {
            "path": str(identity_path),
            "file_sha256": file_sha256(identity_path),
        },
        "topology_source": {
            "path": str(topology_path),
            "file_sha256": file_sha256(topology_path),
        },
        "physical_gpu_count": 8,
        "physical_gpus": rows,
        "xgmi_islands": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "pci_bus_is_authoritative_join_key": True,
    }
    value = {**unsigned, "inventory_digest": object_sha256(unsigned)}
    _write_create_only(output, value)
    return value


def load_rocm_physical_inventory(
    path: Path, expected_sha256: str
) -> Mapping[str, Any]:
    source = _plain_file(path, label="ROCm physical inventory")
    if file_sha256(source) != _digest(
        expected_sha256, label="ROCm physical inventory"
    ):
        fail("ROCm physical inventory SHA-256 differs")
    value = _load_canonical_receipt(
        source, label="ROCm physical inventory", digest_field="inventory_digest"
    )
    if set(value) != {
        "schema_version",
        "identity_source",
        "topology_source",
        "physical_gpu_count",
        "physical_gpus",
        "xgmi_islands",
        "pci_bus_is_authoritative_join_key",
        "inventory_digest",
    } or (
        value.get("schema_version") != PHYSICAL_INVENTORY_SCHEMA
        or value.get("physical_gpu_count") != 8
        or value.get("xgmi_islands") != [[0, 1, 2, 3], [4, 5, 6, 7]]
        or value.get("pci_bus_is_authoritative_join_key") is not True
    ):
        fail("ROCm physical inventory schema differs")
    for label, field in (
        ("ROCm-SMI identity source", "identity_source"),
        ("ROCm-SMI topology source", "topology_source"),
    ):
        reference = value.get(field)
        if not isinstance(reference, Mapping) or set(reference) != {
            "path",
            "file_sha256",
        }:
            fail(f"{label} binding differs")
        referenced_path = _plain_file(reference.get("path"), label=label)
        if file_sha256(referenced_path) != reference.get("file_sha256"):
            fail(f"{label} SHA-256 differs")
    identity_path = Path(str(value["identity_source"]["path"]))
    topology_path = Path(str(value["topology_source"]["path"]))
    try:
        replayed_rows = _parse_rocm_physical_inventory(
            identity_path.read_text(encoding="ascii"),
            topology_path.read_text(encoding="ascii"),
        )
    except UnicodeError as error:
        raise GenericActionDataPrepError("ROCm-SMI inventory is not ASCII") from error
    rows = value.get("physical_gpus")
    if rows != replayed_rows:
        fail("ROCm physical inventory does not replay from sealed source bytes")
    return value


def observe_rocm_runtime_mapping(
    output: Path,
    *,
    expected_count: int,
    expected_rocr: Optional[str],
    physical_inventory_path: Path,
    expected_physical_inventory_sha256: str,
) -> Mapping[str, Any]:
    """Observe the HIP UUID and PCI bus behind each torch-visible ordinal."""

    if expected_count not in {4, 8}:
        fail("runtime mapping expected count differs")
    if expected_rocr not in {None, "0,1,2,3", "4,5,6,7"}:
        fail("runtime mapping expected ROCR value differs")
    if os.environ.get("ROCR_VISIBLE_DEVICES") != expected_rocr:
        fail("runtime mapping ROCR environment differs")
    physical_inventory = load_rocm_physical_inventory(
        physical_inventory_path, expected_physical_inventory_sha256
    )
    physical_by_bus = {
        row["pci_bus_id"]: row for row in physical_inventory["physical_gpus"]
    }
    try:
        import ctypes
        import torch
    except ImportError as error:
        raise GenericActionDataPrepError(
            "torch/ctypes runtime mapping dependencies are unavailable"
        ) from error
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        fail("runtime mapping requires the HIP torch backend")
    torch_count = int(torch.cuda.device_count())
    if torch_count != expected_count:
        fail("torch-visible device count differs")

    class HipUuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    try:
        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        hip.hipGetDeviceCount.restype = ctypes.c_int
        hip.hipDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        hip.hipDeviceGet.restype = ctypes.c_int
        hip.hipDeviceGetUuid.argtypes = [ctypes.POINTER(HipUuid), ctypes.c_int]
        hip.hipDeviceGetUuid.restype = ctypes.c_int
        hip.hipDeviceGetPCIBusId.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    except (AttributeError, OSError) as error:
        raise GenericActionDataPrepError(
            "HIP UUID/PCI-bus observation API is unavailable"
        ) from error

    def hip_success(code: int, label: str) -> None:
        if code != 0:
            fail(f"{label} failed with HIP status {code}")

    hip_count = ctypes.c_int(-1)
    hip_success(hip.hipGetDeviceCount(ctypes.byref(hip_count)), "hipGetDeviceCount")
    if hip_count.value != expected_count:
        fail("HIP-visible device count differs from torch")
    devices = []
    for logical_index in range(expected_count):
        device = ctypes.c_int(-1)
        hip_success(
            hip.hipDeviceGet(ctypes.byref(device), logical_index),
            "hipDeviceGet",
        )
        uuid = HipUuid()
        hip_success(
            hip.hipDeviceGetUuid(ctypes.byref(uuid), device.value),
            "hipDeviceGetUuid",
        )
        uuid_hex = bytes(uuid.bytes).hex()
        if UUID_HEX_RE.fullmatch(uuid_hex) is None or uuid_hex == "0" * 32:
            fail("HIP device UUID differs")
        bus_buffer = ctypes.create_string_buffer(32)
        hip_success(
            hip.hipDeviceGetPCIBusId(bus_buffer, len(bus_buffer), device.value),
            "hipDeviceGetPCIBusId",
        )
        try:
            pci_bus_id = normalize_pci_bus_id(bus_buffer.value.decode("ascii"))
        except UnicodeError as error:
            raise GenericActionDataPrepError("HIP PCI bus is not ASCII") from error
        physical = physical_by_bus.get(pci_bus_id)
        if physical is None:
            fail("HIP PCI bus does not join one sealed ROCm-SMI physical row")
        properties = torch.cuda.get_device_properties(logical_index)
        property_uuid = getattr(properties, "uuid", None)
        if property_uuid is not None:
            property_uuid = str(property_uuid)
            if not property_uuid:
                fail("torch device-property UUID differs")
        device_name = str(properties.name)
        if not device_name:
            fail("torch device name differs")
        devices.append(
            {
                "logical_index": logical_index,
                "hip_device_handle": device.value,
                "hip_uuid_hex": uuid_hex,
                "pci_bus_id": pci_bus_id,
                "physical_index": physical["physical_index"],
                "physical_rocm_unique_id": physical["rocm_unique_id"],
                "torch_device_name": device_name,
                "torch_property_uuid": property_uuid,
            }
        )
    if (
        len({row["hip_uuid_hex"] for row in devices}) != expected_count
        or len({row["pci_bus_id"] for row in devices}) != expected_count
        or len({row["physical_index"] for row in devices}) != expected_count
        or len({row["physical_rocm_unique_id"] for row in devices})
        != expected_count
    ):
        fail("torch/HIP visible devices are not unique")
    expected_physical_indices = (
        list(range(8))
        if expected_count == 8
        else [int(token) for token in str(expected_rocr).split(",")]
    )
    if sorted(row["physical_index"] for row in devices) != expected_physical_indices:
        fail("torch/HIP mapping does not join the requested exact physical set")
    unsigned = {
        "schema_version": RUNTIME_MAPPING_SCHEMA,
        "rocr_visible_devices": expected_rocr,
        "torch_version": str(torch.__version__),
        "torch_hip_version": str(torch.version.hip),
        "torch_device_count": torch_count,
        "hip_runtime_device_count": hip_count.value,
        "physical_inventory": {
            "path": str(physical_inventory_path),
            "file_sha256": expected_physical_inventory_sha256,
            "inventory_digest": physical_inventory["inventory_digest"],
        },
        "pci_bus_is_authoritative_join_key": True,
        "physical_index_derived_from_pci_bus_join": True,
        "physical_rocm_unique_id_replayed": True,
        "hip_logical_order_is_observation_only": True,
        "devices": devices,
    }
    value = {**unsigned, "observation_digest": object_sha256(unsigned)}
    _write_create_only(output, value)
    return value


def load_rocm_runtime_mapping(
    path: Path,
    expected_sha256: str,
    *,
    expected_count: int,
    expected_rocr: Optional[str],
) -> Mapping[str, Any]:
    source = _plain_file(path, label="ROCm runtime mapping")
    expected = _digest(expected_sha256, label="ROCm runtime mapping")
    if file_sha256(source) != expected:
        fail("ROCm runtime mapping SHA-256 differs")
    value = _load_json(source, label="ROCm runtime mapping")
    unsigned = dict(value)
    declared = unsigned.pop("observation_digest", None)
    devices = value.get("devices")
    inventory_reference = value.get("physical_inventory")
    if (
        set(value)
        != {
            "schema_version",
            "rocr_visible_devices",
            "torch_version",
            "torch_hip_version",
            "torch_device_count",
            "hip_runtime_device_count",
            "physical_inventory",
            "pci_bus_is_authoritative_join_key",
            "physical_index_derived_from_pci_bus_join",
            "physical_rocm_unique_id_replayed",
            "hip_logical_order_is_observation_only",
            "devices",
            "observation_digest",
        }
        or value.get("schema_version") != RUNTIME_MAPPING_SCHEMA
        or value.get("rocr_visible_devices") != expected_rocr
        or value.get("torch_device_count") != expected_count
        or value.get("hip_runtime_device_count") != expected_count
        or value.get("pci_bus_is_authoritative_join_key") is not True
        or value.get("physical_index_derived_from_pci_bus_join") is not True
        or value.get("physical_rocm_unique_id_replayed") is not True
        or value.get("hip_logical_order_is_observation_only") is not True
        or not isinstance(value.get("torch_version"), str)
        or not value.get("torch_version")
        or not isinstance(value.get("torch_hip_version"), str)
        or not value.get("torch_hip_version")
        or declared != object_sha256(unsigned)
        or not isinstance(devices, list)
        or len(devices) != expected_count
    ):
        fail("ROCm runtime mapping contract differs")
    if not isinstance(inventory_reference, Mapping) or set(inventory_reference) != {
        "path",
        "file_sha256",
        "inventory_digest",
    }:
        fail("ROCm runtime mapping physical inventory reference differs")
    physical_inventory = load_rocm_physical_inventory(
        Path(str(inventory_reference.get("path"))),
        str(inventory_reference.get("file_sha256")),
    )
    if inventory_reference.get("inventory_digest") != physical_inventory.get(
        "inventory_digest"
    ):
        fail("ROCm runtime mapping physical inventory digest differs")
    physical_by_index = {
        row["physical_index"]: row for row in physical_inventory["physical_gpus"]
    }
    for logical_index, row in enumerate(devices):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "logical_index",
                "hip_device_handle",
                "hip_uuid_hex",
                "pci_bus_id",
                "physical_index",
                "physical_rocm_unique_id",
                "torch_device_name",
                "torch_property_uuid",
            }
            or row.get("logical_index") != logical_index
            or type(row.get("hip_device_handle")) is not int
            or UUID_HEX_RE.fullmatch(str(row.get("hip_uuid_hex"))) is None
            or row.get("hip_uuid_hex") == "0" * 32
            or PCI_BUS_RE.fullmatch(str(row.get("pci_bus_id"))) is None
            or type(row.get("physical_index")) is not int
            or row.get("physical_index") not in range(8)
            or ROCM_UNIQUE_ID_RE.fullmatch(
                str(row.get("physical_rocm_unique_id"))
            )
            is None
            or not isinstance(row.get("torch_device_name"), str)
            or not row.get("torch_device_name")
            or (
                row.get("torch_property_uuid") is not None
                and (
                    not isinstance(row.get("torch_property_uuid"), str)
                    or not row.get("torch_property_uuid")
                )
            )
        ):
            fail("ROCm runtime device observation differs")
        physical = physical_by_index[row["physical_index"]]
        if (
            row["pci_bus_id"] != physical["pci_bus_id"]
            or row["physical_rocm_unique_id"] != physical["rocm_unique_id"]
        ):
            fail("ROCm runtime device does not replay its physical PCI/unique-ID join")
    if (
        len({row["hip_uuid_hex"] for row in devices}) != expected_count
        or len({row["pci_bus_id"] for row in devices}) != expected_count
        or len({row["physical_index"] for row in devices}) != expected_count
        or len({row["physical_rocm_unique_id"] for row in devices})
        != expected_count
    ):
        fail("ROCm runtime mapping contains duplicate physical devices")
    return value


def validate_rocm_runtime_mapping_join(
    all8_path: Path,
    all8_sha256: str,
    observed_path: Path,
    observed_sha256: str,
    physical_indices: Sequence[int],
) -> Mapping[str, Any]:
    expected_indices = list(physical_indices)
    if expected_indices not in ([0, 1, 2, 3], [4, 5, 6, 7]):
        fail("expected physical ROCm island differs")
    expected_rocr = ",".join(str(index) for index in expected_indices)
    all8 = load_rocm_runtime_mapping(
        all8_path, all8_sha256, expected_count=8, expected_rocr=None
    )
    observed = load_rocm_runtime_mapping(
        observed_path,
        observed_sha256,
        expected_count=4,
        expected_rocr=expected_rocr,
    )

    # HIP logical ordinals are a runtime enumeration, not physical GPU
    # indices.  Join each observation to the sealed ROCm-SMI inventory through
    # PCI bus (with the physical UniqueID replayed by the loader), then compare
    # physical sets and per-device identities independent of logical order.
    # The original logical order remains in the receipt as evidence only.
    all8_by_physical = {
        row["physical_index"]: row for row in all8["devices"]
    }
    observed_by_physical = {
        row["physical_index"]: row for row in observed["devices"]
    }
    if (
        set(all8_by_physical) != set(range(8))
        or set(observed_by_physical) != set(expected_indices)
    ):
        fail("torch/HIP observed devices are not the requested exact physical island")

    def identity(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "physical_index": row["physical_index"],
            "pci_bus_id": row["pci_bus_id"],
            "physical_rocm_unique_id": row["physical_rocm_unique_id"],
            "hip_uuid_hex": row["hip_uuid_hex"],
        }

    canonical_identity_rows = [
        dict(identity(observed_by_physical[index])) for index in expected_indices
    ]
    observed_logical_rows = [
        {
            "logical_index": row["logical_index"],
            **dict(identity(row)),
        }
        for row in observed["devices"]
    ]
    all8_logical_to_physical = [
        row["physical_index"] for row in all8["devices"]
    ]
    observed_logical_to_physical = [
        row["physical_index"] for row in observed["devices"]
    ]
    if (
        observed.get("torch_version") != all8.get("torch_version")
        or observed.get("torch_hip_version") != all8.get("torch_hip_version")
        or observed.get("physical_inventory") != all8.get("physical_inventory")
        or any(
            identity(observed_by_physical[index])
            != identity(all8_by_physical[index])
            for index in expected_indices
        )
    ):
        fail(
            "torch/HIP observed physical-index/PCI/unique-ID/UUID identity "
            "differs from the sealed all8 mapping"
        )
    return {
        "expected_physical_indices": expected_indices,
        "expected_rocr_visible_devices": expected_rocr,
        "all8_observation_digest": all8["observation_digest"],
        "observed_observation_digest": observed["observation_digest"],
        "physical_inventory_digest": all8["physical_inventory"][
            "inventory_digest"
        ],
        "all8_logical_to_physical_order": all8_logical_to_physical,
        "all8_logical_to_physical_order_digest": object_sha256(
            all8_logical_to_physical
        ),
        "observed_logical_to_physical_order": observed_logical_to_physical,
        "observed_logical_to_physical_order_digest": object_sha256(
            observed_logical_to_physical
        ),
        "observed_logical_identity_rows": observed_logical_rows,
        "observed_logical_identity_rows_digest": object_sha256(
            observed_logical_rows
        ),
        "canonical_physical_identity_rows": canonical_identity_rows,
        "canonical_physical_identity_rows_digest": object_sha256(
            canonical_identity_rows
        ),
        "pci_bus_physical_join_replayed": True,
        "xgmi_island_exact": True,
        "hip_logical_order_is_observation_only": True,
        "mapping_exact_physical_set_verified": True,
        "logical_order_permutation_allowed": True,
        "cross_island_visibility_rejected": True,
    }


def validate_release_tree(args: argparse.Namespace) -> Mapping[str, Any]:
    method_root = _plain_directory(args.method_root, label="method root")
    archive = _plain_file(args.method_archive, label="method archive")
    manifest_path = _plain_file(args.method_manifest, label="method manifest")
    archive_sha = _digest(args.expected_method_archive_sha256, label="method archive")
    manifest_sha = _digest(args.expected_method_manifest_sha256, label="method manifest")
    try:
        manifest = release.audit(
            archive,
            manifest_path,
            expected_archive_sha256=archive_sha,
            expected_manifest_sha256=manifest_sha,
        )
    except release.GenericActionDataReleaseError as error:
        raise GenericActionDataPrepError(str(error)) from error
    if manifest.get("external_evidence") != R10_PARITY_EVIDENCE:
        fail("release r10 tensor-parity evidence binding differs")
    actual_files: set[str] = set()
    for path in method_root.rglob("*"):
        relative = path.relative_to(method_root).as_posix()
        if path.is_symlink():
            fail(f"runtime tree contains a symlink: {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif not path.is_dir():
            fail(f"runtime tree contains a non-file entry: {relative}")
    expected_files = {row["path"] for row in manifest["files"]}
    if actual_files != expected_files:
        fail("runtime tree is not the exact release member closure")
    for row in manifest["files"]:
        path = _plain_file(method_root / row["path"], label=f"runtime {row['path']}")
        if (
            file_sha256(path) != row["sha256"]
            or stat.S_IMODE(path.stat().st_mode) != row["mode"]
        ):
            fail(f"runtime member differs from archive: {row['path']}")
    python_bin = _plain_file(args.python_bin, label="Python executable")
    if not os.access(python_bin, os.X_OK):
        fail("Python executable is not executable")
    python_sha = _digest(args.expected_python_sha256, label="Python executable")
    if file_sha256(python_bin) != python_sha:
        fail("Python executable SHA-256 differs")
    return {
        "method_root": str(method_root),
        "method_archive": str(archive),
        "method_archive_sha256": archive_sha,
        "method_manifest": str(manifest_path),
        "method_manifest_sha256": manifest_sha,
        "method_manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "component_pins": dict(manifest["component_pins"]),
        "external_evidence": dict(manifest["external_evidence"]),
        "python_bin": str(python_bin),
        "python_sha256": python_sha,
    }


def _run_checked(command: Sequence[str], *, label: str) -> str:
    result = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )
    if result.returncode != 0:
        fail(f"{label} failed closed: {result.stdout[-4000:]}")
    return result.stdout


def build_upstream_plan(
    args: argparse.Namespace, closure: Mapping[str, Any], run_root: Path
) -> tuple[Path, str, Mapping[str, Any]]:
    seed1 = _plain_file(args.seed1_spec, label="reserve4 seed1 spec")
    seed2 = _plain_file(args.seed2_spec, label="reserve4 seed2 spec")
    if file_sha256(seed1) != SEED1_SPEC_SHA256:
        fail("reserve4 seed1 spec SHA-256 differs")
    if file_sha256(seed2) != SEED2_SPEC_SHA256:
        fail("reserve4 seed2 spec SHA-256 differs")
    method_root = Path(closure["method_root"])
    output = run_root / "generation-plan"
    _run_checked(
        [
            closure["python_bin"],
            "-B",
            str(method_root / "tools/reserve4_fixed_generation_sp4_v1.py"),
            "build-plan",
            "--seed1-spec",
            str(seed1),
            "--seed2-spec",
            str(seed2),
            "--split",
            "fit",
            "--output-dir",
            str(output),
        ],
        label="reserve4 fixed fit generation plan",
    )
    path = output / "reserve4-fixed-generation-plan-v1.json"
    observed = file_sha256(path)
    value = validate_upstream_fit_plan(path, observed)
    return path, observed, value


def validate_upstream_fit_plan(
    path: Path, expected_sha256: str
) -> Mapping[str, Any]:
    """Replay the generic seal, then require the exact four fit shards."""

    source = _plain_file(path, label="reserve4 fit generation plan")
    expected = _digest(expected_sha256, label="reserve4 fit generation plan")
    if file_sha256(source) != expected:
        fail("reserve4 fit generation plan SHA-256 differs")
    try:
        value, replayed_path, observed = reserve4_runner.load_plan(source, expected)
    except reserve4_runner.Reserve4GenerationError as error:
        raise GenericActionDataPrepError(str(error)) from error
    if replayed_path != source or observed != expected:
        fail("reserve4 fit generation plan replay binding differs")
    tasks = value.get("tasks")
    if (
        value.get("schema_version")
        != "bernini-reserve4-fixed-generation-sp4-plan-v1"
        or value.get("analysis_split") != "fit"
        or value.get("generation_invocation_count") != 40
        or value.get("seed_cell_count") != 4
        or not isinstance(tasks, list)
        or len(tasks) != 40
        or value.get("execution_contract", {}).get("optimizer_authorized")
        is not False
    ):
        fail("reserve4 fit generation plan authority differs")
    if any(
        row.get("analysis_split") != "fit"
        or row.get("group_id") not in {"sp4-a", "sp4-b"}
        or row.get("visible_gpus")
        != (
            [0, 1, 2, 3]
            if row.get("group_id") == "sp4-a"
            else [4, 5, 6, 7]
        )
        for row in tasks
        if isinstance(row, Mapping)
    ) or any(not isinstance(row, Mapping) for row in tasks):
        fail("fit rows are not on their sealed XGMI4 groups")
    expected_shards = []
    for seed_slot in ("seed1", "seed2"):
        for group_id, visible_gpus in (
            ("sp4-a", [0, 1, 2, 3]),
            ("sp4-b", [4, 5, 6, 7]),
        ):
            rows = [
                row
                for row in tasks
                if row["seed_slot"] == seed_slot and row["group_id"] == group_id
            ]
            if (
                len(rows) != 10
                or [row.get("semantic_branch") for row in rows]
                != list(reserve4_runner.bank_contract.MACE_BRANCH_ORDER)
            ):
                fail("fit shard is not one complete ordered ten-branch cell")
            candidate_ids = [row["candidate_id"] for row in rows]
            expected_shards.append(
                {
                    "shard_id": f"{seed_slot}-{group_id}-fit",
                    "seed_slot": seed_slot,
                    "group_id": group_id,
                    "visible_gpus": visible_gpus,
                    "candidate_ids": candidate_ids,
                    "candidate_count": 10,
                }
            )
    if value.get("shards") != expected_shards:
        fail("fit plan must close four exact serial shards")
    if [row["candidate_id"] for row in tasks] != [
        candidate_id
        for shard in expected_shards
        for candidate_id in shard["candidate_ids"]
    ]:
        fail("fit task order differs from sealed shard order")
    cell_proofs = value.get("cell_proofs")
    if (
        not isinstance(cell_proofs, list)
        or len(cell_proofs) != 4
        or any(
            row.get("analysis_split") != "fit"
            or row.get("group_id") not in {"sp4-a", "sp4-b"}
            or row.get("branch_order")
            != list(reserve4_runner.bank_contract.MACE_BRANCH_ORDER)
            for row in cell_proofs
            if isinstance(row, Mapping)
        )
        or any(not isinstance(row, Mapping) for row in cell_proofs)
    ):
        fail("fit plan cell semantics differ")
    return value


def build_controller_plan(
    args: argparse.Namespace,
    closure: Mapping[str, Any],
    run_root: Path,
    upstream_path: Path,
    upstream_sha: str,
    upstream: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Path, str]:
    unsigned = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "generic-action-fit40-generation-136141-r13",
        "phase": "generation",
        "analysis_split": "fit",
        "holder": {"job_id": HOLDER_JOB, "node": HOLDER_NODE},
        "release": dict(closure),
        "upstream_plan": {
            "path": str(upstream_path),
            "file_sha256": upstream_sha,
            "digest": upstream.get("plan_digest"),
        },
        "r10_parity_evidence": R10_PARITY_EVIDENCE,
        "master_port": args.master_port,
        "runtime": {
            "bernini_root": str(BERNINI_ROOT),
            "veomni_root": str(VEOMNI_ROOT),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_manifest": str(CHECKPOINT_MANIFEST),
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        },
        "topology": TOPOLOGY_CONTRACT,
        "expected_outputs": OUTPUT_CONTRACT,
        "non_authoritative_dynamic_probe_observation": (
            NON_AUTHORITATIVE_PROBE_OBSERVATION
        ),
        "authority": AUTHORITY_CONTRACT,
        "run_root": str(run_root),
    }
    plan = {**unsigned, "plan_digest": object_sha256(unsigned)}
    path = run_root / "controller-plan.json"
    sha = _write_create_only(path, plan)
    return plan, path, sha


def validate_controller_plan(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    source = _plain_file(path, label="controller plan")
    if file_sha256(source) != _digest(expected_sha256, label="controller plan"):
        fail("controller plan SHA-256 differs")
    value = _load_json(source, label="controller plan")
    unsigned = dict(value)
    declared = unsigned.pop("plan_digest", None)
    if (
        value.get("schema_version") != PLAN_SCHEMA
        or declared != object_sha256(unsigned)
        or value.get("phase") != "generation"
        or value.get("analysis_split") != "fit"
        or value.get("holder") != {"job_id": HOLDER_JOB, "node": HOLDER_NODE}
        or value.get("topology") != TOPOLOGY_CONTRACT
        or value.get("expected_outputs") != OUTPUT_CONTRACT
        or value.get("non_authoritative_dynamic_probe_observation")
        != NON_AUTHORITATIVE_PROBE_OBSERVATION
        or value.get("authority") != AUTHORITY_CONTRACT
        or value.get("r10_parity_evidence") != R10_PARITY_EVIDENCE
        or not isinstance(value.get("release"), Mapping)
        or value["release"].get("external_evidence") != R10_PARITY_EVIDENCE
    ):
        fail("controller plan contract differs")
    upstream = value.get("upstream_plan", {})
    upstream_path = _plain_file(upstream.get("path"), label="upstream plan")
    if file_sha256(upstream_path) != upstream.get("file_sha256"):
        fail("controller upstream plan binding differs")
    replayed = validate_upstream_fit_plan(
        upstream_path, str(upstream.get("file_sha256"))
    )
    if replayed.get("plan_digest") != upstream.get("digest"):
        fail("controller upstream plan digest/semantics differ")
    return value


def validate_launch_environment(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    plan = validate_controller_plan(path, expected_sha256)
    closure = plan["release"]
    pins = closure["component_pins"]
    r10 = plan["r10_parity_evidence"]
    expected = {
        "GADP_CONFIRM": LAUNCH_CONFIRMATION,
        "GADP_PHASE": "generation",
        "GADP_SPLIT": "fit",
        "GADP_RUN_ROOT": plan["run_root"],
        "GADP_MASTER_PORT": str(plan["master_port"]),
        "GADP_CONTROLLER_PLAN": str(path),
        "GADP_CONTROLLER_PLAN_SHA256": expected_sha256,
        "GADP_UPSTREAM_PLAN": plan["upstream_plan"]["path"],
        "GADP_UPSTREAM_PLAN_SHA256": plan["upstream_plan"]["file_sha256"],
        "GADP_METHOD_ROOT": closure["method_root"],
        "GADP_METHOD_ARCHIVE": closure["method_archive"],
        "GADP_METHOD_ARCHIVE_SHA256": closure["method_archive_sha256"],
        "GADP_METHOD_MANIFEST": closure["method_manifest"],
        "GADP_METHOD_MANIFEST_SHA256": closure["method_manifest_sha256"],
        "GADP_METHOD_REVISION": closure["content_closure_sha1"],
        "GADP_PYTHON_BIN": closure["python_bin"],
        "GADP_PYTHON_SHA256": closure["python_sha256"],
        "GADP_CONTROLLER_SHA256": pins["controller_sha256"],
        "GADP_LAUNCHER_SHA256": pins["launcher_sha256"],
        "GADP_GENERATOR_SHA256": pins["generator_sha256"],
        "GADP_R10_COMPILE_SMOKE_RECEIPT": r10[
            "compile_smoke_receipt"
        ]["path"],
        "GADP_R10_COMPILE_SMOKE_RECEIPT_SHA256": r10[
            "compile_smoke_receipt"
        ]["file_sha256"],
        "GADP_R10_GENERATION_LOG": r10["generation_log"]["path"],
        "GADP_R10_GENERATION_LOG_SHA256": r10["generation_log"][
            "file_sha256"
        ],
    }
    mismatches = [name for name, value in expected.items() if os.environ.get(name) != str(value)]
    if mismatches:
        fail(f"launcher environment is not cross-bound to plan: {sorted(mismatches)}")
    return plan


def _launch(
    args: argparse.Namespace,
    closure: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_sha: str,
) -> None:
    if args.confirm != LAUNCH_CONFIRMATION:
        fail("launch confirmation differs")
    method_root = Path(closure["method_root"])
    launcher = _plain_file(
        method_root / "scripts/auh_generic_action_data_prep_136141_world4_v1.sh",
        label="fit generation launcher",
    )
    pins = closure["component_pins"]
    r10 = plan["r10_parity_evidence"]
    environment = {
        **os.environ,
        "GADP_CONFIRM": LAUNCH_CONFIRMATION,
        "GADP_PHASE": "generation",
        "GADP_SPLIT": "fit",
        "GADP_RUN_ROOT": plan["run_root"],
        "GADP_MASTER_PORT": str(args.master_port),
        "GADP_CONTROLLER_PLAN": str(plan_path),
        "GADP_CONTROLLER_PLAN_SHA256": plan_sha,
        "GADP_UPSTREAM_PLAN": plan["upstream_plan"]["path"],
        "GADP_UPSTREAM_PLAN_SHA256": plan["upstream_plan"]["file_sha256"],
        "GADP_METHOD_ROOT": closure["method_root"],
        "GADP_METHOD_ARCHIVE": closure["method_archive"],
        "GADP_METHOD_ARCHIVE_SHA256": closure["method_archive_sha256"],
        "GADP_METHOD_MANIFEST": closure["method_manifest"],
        "GADP_METHOD_MANIFEST_SHA256": closure["method_manifest_sha256"],
        "GADP_METHOD_REVISION": closure["content_closure_sha1"],
        "GADP_PYTHON_BIN": closure["python_bin"],
        "GADP_PYTHON_SHA256": closure["python_sha256"],
        "GADP_CONTROLLER_SHA256": pins["controller_sha256"],
        "GADP_LAUNCHER_SHA256": pins["launcher_sha256"],
        "GADP_GENERATOR_SHA256": pins["generator_sha256"],
        "GADP_R10_COMPILE_SMOKE_RECEIPT": r10[
            "compile_smoke_receipt"
        ]["path"],
        "GADP_R10_COMPILE_SMOKE_RECEIPT_SHA256": r10[
            "compile_smoke_receipt"
        ]["file_sha256"],
        "GADP_R10_GENERATION_LOG": r10["generation_log"]["path"],
        "GADP_R10_GENERATION_LOG_SHA256": r10["generation_log"][
            "file_sha256"
        ],
    }
    process = subprocess.Popen(
        ["bash", str(launcher)],
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        env=environment,
        start_new_session=True,
    )
    process_group = process.pid
    try:
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    try:
                        os.killpg(process_group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
        raise
    if return_code != 0:
        fail(f"fit generation launcher exited {return_code}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _local_artifact_references(
    value: Any, candidate_root: Path
) -> dict[Path, str]:
    """Collect every SHA-bound artifact path local to one candidate root."""

    references: dict[Path, str] = {}

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if "path" in node and "sha256" in node:
                path_value, sha_value = node.get("path"), node.get("sha256")
                if isinstance(path_value, str) and isinstance(sha_value, str):
                    candidate_path = Path(path_value)
                    if candidate_path.is_absolute() and _is_within(
                        candidate_path, candidate_root
                    ):
                        digest = _digest(sha_value, label="candidate artifact")
                        previous = references.get(candidate_path)
                        if previous is not None and previous != digest:
                            fail("candidate artifact has conflicting SHA bindings")
                        references[candidate_path] = digest
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return references


def _validate_candidate_output_closure(
    task: Mapping[str, Any], candidate_root: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]:
    root = _plain_directory(candidate_root, label="candidate generation root")
    if root.name != task.get("candidate_id"):
        fail("candidate output directory identity differs")
    pair_path = root / "pair-v5-t2v-calibration-receipt.json"
    pair = _load_canonical_receipt(pair_path, label="candidate generation receipt")
    try:
        replayed, _ = reserve4_runner._validate_candidate_receipt(  # type: ignore[attr-defined]
            task, pair_path
        )
    except reserve4_runner.Reserve4GenerationError as error:
        raise GenericActionDataPrepError(str(error)) from error
    if replayed != pair:
        fail("candidate receipt differs from semantic replay")

    native_path = _plain_file(
        pair.get("native_receipt_path"), label="candidate native receipt"
    )
    if native_path != root / "receipt.json" or file_sha256(native_path) != pair.get(
        "native_receipt_sha256"
    ):
        fail("candidate native receipt path/SHA binding differs")
    native = _load_canonical_receipt(native_path, label="candidate native receipt")
    if native.get("receipt_digest") != pair.get("native_receipt_digest"):
        fail("candidate native receipt digest binding differs")

    references = _local_artifact_references(pair, root)
    for path, digest in _local_artifact_references(native, root).items():
        previous = references.get(path)
        if previous is not None and previous != digest:
            fail("pair/native artifact SHA bindings disagree")
        references[path] = digest
    references[pair_path] = file_sha256(pair_path)
    references[native_path] = file_sha256(native_path)

    actual_files: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            fail(f"candidate output contains symlink: {path}")
        if path.is_file():
            actual_files.add(path)
        elif path.is_dir():
            fail(f"candidate output contains an extra directory: {path}")
        else:
            fail(f"candidate output contains non-file entry: {path}")
    if actual_files != set(references):
        missing = sorted(str(path) for path in set(references) - actual_files)
        extra = sorted(str(path) for path in actual_files - set(references))
        fail(f"candidate artifact closure differs; missing={missing}, extra={extra}")
    artifact_rows = []
    for path in sorted(actual_files, key=lambda item: item.relative_to(root).as_posix()):
        expected = references[path]
        observed = file_sha256(_plain_file(path, label="candidate artifact"))
        if observed != expected:
            fail(f"candidate artifact SHA-256 differs: {path}")
        artifact_rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "file_sha256": observed,
                "size": path.stat().st_size,
            }
        )
    return pair, native, artifact_rows


def _strict_generation_audit(
    audit_path: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_sha: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    gaussian_proofs: Sequence[Mapping[str, Any]],
) -> tuple[Path, str, Mapping[str, Any]]:
    source = _plain_file(audit_path, label="generation audit receipt")
    value = _load_canonical_receipt(source, label="generation audit receipt")
    expected = {
        "schema_version": "bernini-reserve4-fixed-generation-audit-receipt-v1",
        "plan_path": str(plan_path),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "analysis_split": "fit",
        "candidate_count": 40,
        "seed_cell_count": 4,
        "candidate_receipts": list(candidate_rows),
        "same_cell_gaussian_proofs": list(gaussian_proofs),
        "generation_complete": True,
        "independent_full81_review_performed": False,
        "visual_review_required_before_phi_v1_extraction": True,
        "phi_v1_extraction_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "generated_media_is_editor_input_or_target": False,
    }
    if value != {**expected, "receipt_digest": object_sha256(expected)}:
        fail("generation audit receipt exact schema/ordered replay differs")
    return source, file_sha256(source), value


def validate_generation_output_closure(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    generation_roots: Sequence[Path],
    generation_audit_path: Path,
) -> Mapping[str, Any]:
    plan = validate_upstream_fit_plan(plan_path, expected_plan_sha256)
    resolved_plan = _plain_file(plan_path, label="reserve4 fit generation plan")
    plan_sha = file_sha256(resolved_plan)
    if len(generation_roots) != 4:
        fail("generation closure requires four ordered shard roots")
    expected_shards = list(plan["shards"])
    validated_for_gaussian: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    candidate_receipt_rows: list[Mapping[str, Any]] = []
    shard_rows: list[Mapping[str, Any]] = []
    candidate_artifact_closures: list[Mapping[str, Any]] = []

    for ordinal, (root_value, shard) in enumerate(
        zip(generation_roots, expected_shards), start=1
    ):
        root = _plain_directory(root_value, label="generation shard root")
        expected_root_name = f"{shard['seed_slot']}-{shard['group_id']}"
        if root.name != expected_root_name or shard.get("shard_id") != SHARD_ORDER[
            ordinal - 1
        ]:
            fail("generation shard root/order differs")
        matching = [
            task
            for task in plan["tasks"]
            if task["seed_slot"] == shard["seed_slot"]
            and task["group_id"] == shard["group_id"]
        ]
        if [task["candidate_id"] for task in matching] != shard["candidate_ids"]:
            fail("generation shard candidate order differs from plan")
        expected_children = set(shard["candidate_ids"]) | {
            "reserve4-generation-shard-receipt-v1.json"
        }
        actual_children: set[str] = set()
        for child in root.iterdir():
            if child.is_symlink():
                fail(f"generation shard contains symlink: {child}")
            actual_children.add(child.name)
            if child.name in shard["candidate_ids"]:
                if not child.is_dir():
                    fail("candidate output is not a directory")
            elif child.name == "reserve4-generation-shard-receipt-v1.json":
                if not child.is_file():
                    fail("shard receipt is not a plain file")
            else:
                fail(f"generation shard contains extra entry: {child}")
        if actual_children != expected_children:
            fail("generation shard root has missing or extra entries")

        shard_candidate_rows = []
        for task in matching:
            pair, _, artifacts = _validate_candidate_output_closure(
                task, root / task["candidate_id"]
            )
            pair_path = root / task["candidate_id"] / (
                "pair-v5-t2v-calibration-receipt.json"
            )
            pair_row = {
                "candidate_id": task["candidate_id"],
                "path": str(pair_path),
                "file_sha256": file_sha256(pair_path),
                "receipt_digest": pair["receipt_digest"],
            }
            shard_candidate_rows.append(pair_row)
            candidate_receipt_rows.append(pair_row)
            validated_for_gaussian.append((task, pair))
            candidate_artifact_closures.append(
                {
                    "candidate_id": task["candidate_id"],
                    "artifact_count": len(artifacts),
                    "artifacts": artifacts,
                    "artifact_closure_digest": object_sha256(artifacts),
                }
            )

        try:
            shard_gaussian = reserve4_runner._gaussian_cell_proofs(  # type: ignore[attr-defined]
                validated_for_gaussian[-10:]
            )
        except reserve4_runner.Reserve4GenerationError as error:
            raise GenericActionDataPrepError(str(error)) from error
        shard_receipt_path = root / "reserve4-generation-shard-receipt-v1.json"
        shard_receipt = _load_canonical_receipt(
            shard_receipt_path, label="generation shard receipt"
        )
        shard_unsigned = {
            "schema_version": "bernini-reserve4-fixed-generation-shard-receipt-v1",
            "plan_path": str(resolved_plan),
            "plan_file_sha256": plan_sha,
            "plan_digest": plan["plan_digest"],
            "analysis_split": "fit",
            "seed_slot": shard["seed_slot"],
            "group_id": shard["group_id"],
            "visible_gpus": shard["visible_gpus"],
            "candidate_count": 10,
            "candidate_receipts": shard_candidate_rows,
            "same_cell_gaussian_proofs": shard_gaussian,
            "independent_full81_review_performed": False,
            "phi_v1_extraction_authorized": False,
            "training_performed": False,
            "optimizer_created": False,
            "optimizer_authorized": False,
            "generated_media_is_editor_input_or_target": False,
        }
        if shard_receipt != {
            **shard_unsigned,
            "receipt_digest": object_sha256(shard_unsigned),
        }:
            fail("generation shard receipt exact schema/ordered replay differs")
        shard_rows.append(
            {
                "ordinal": ordinal,
                "shard_id": shard["shard_id"],
                "path": str(shard_receipt_path),
                "file_sha256": file_sha256(shard_receipt_path),
                "receipt_digest": shard_receipt["receipt_digest"],
            }
        )

    if len(candidate_receipt_rows) != 40:
        fail("generation candidate closure is not exact40")
    try:
        gaussian_proofs = reserve4_runner._gaussian_cell_proofs(  # type: ignore[attr-defined]
            validated_for_gaussian
        )
    except reserve4_runner.Reserve4GenerationError as error:
        raise GenericActionDataPrepError(str(error)) from error
    audit_path, audit_sha, audit = _strict_generation_audit(
        generation_audit_path,
        plan,
        resolved_plan,
        plan_sha,
        candidate_receipt_rows,
        gaussian_proofs,
    )
    unsigned = {
        "schema_version": GENERATION_CLOSURE_SCHEMA,
        "plan_path": str(resolved_plan),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "analysis_split": "fit",
        "sealed_shard_order": SHARD_ORDER,
        "shard_receipts": shard_rows,
        "candidate_count": 40,
        "candidate_receipts": candidate_receipt_rows,
        "candidate_artifact_closures": candidate_artifact_closures,
        "generation_audit": {
            "path": str(audit_path),
            "file_sha256": audit_sha,
            "receipt_digest": audit["receipt_digest"],
        },
        "generation_roots_exact_member_closure": True,
        "symlinks_rejected": True,
        "serialized_world4_host_checkpoint_load": True,
        "model_load_lock_node_local": True,
        "model_load_lock_held_through_gpu_move_and_malloc_trim": True,
        "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup": True,
        "resource_lifecycle_receipt_proves_all_four_load_complete_before_first_sampling": True,
        "t2v_text_encoder_rank_gpu_residency_required": True,
        "t2v_text_encoder_exact_cpu_offload_suppressed_once_per_rank": True,
        "t2v_text_encoder_retired_only_with_renderer": True,
        "t2v_rank_gpu_memory_limit_gib": T2V_GPU_MEMORY_LIMIT_GIB,
        "all_formal_candidate_world4_gpu_peak_allocated_reserved_receipts_present": True,
        "all_formal_candidate_rank_gpu_peak_reserved_strictly_below_limit": True,
        "r10_smoke_authority_derived_from_pinned_receipt_and_log": True,
        "r10_smoke_mp4_whole_file_sha256_exact_required": True,
        "r10_smoke_gaussian_tensor_identity_exact_required": True,
        "r10_smoke_clean_latent_generated_identity_exact_required": True,
        "safetensors_container_sha256_cross_process_equivalence_required": False,
        "t2v_vae_load_deferred_until_rank0_post_sampling": True,
        "world4_renderer_retirement_barrier_before_rank_zero_vae_load": True,
        "independent_full81_review_performed": False,
        "phi_v1_extraction_authorized": False,
        "optimizer_authorized": False,
        "training_authorized": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def write_generation_closure_receipt(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    generation_roots: Sequence[Path],
    generation_audit_path: Path,
    output: Path,
) -> Mapping[str, Any]:
    value = validate_generation_output_closure(
        plan_path=plan_path,
        expected_plan_sha256=expected_plan_sha256,
        generation_roots=generation_roots,
        generation_audit_path=generation_audit_path,
    )
    _write_create_only(output, value)
    return value


GPU_SHARD_IDENTITIES = [
    (1, "seed1", "sp4-a", "0,1,2,3", [0, 1, 2, 3]),
    (2, "seed1", "sp4-b", "4,5,6,7", [4, 5, 6, 7]),
    (3, "seed2", "sp4-a", "0,1,2,3", [0, 1, 2, 3]),
    (4, "seed2", "sp4-b", "4,5,6,7", [4, 5, 6, 7]),
]


def _expected_gpu_binding(
    run_root: Path,
    *,
    ordinal: int,
    seed_slot: str,
    group_id: str,
    visible: str,
    physical_indices: Sequence[int],
) -> Mapping[str, Any]:
    physical_path = _plain_file(
        run_root / "logs/gpu-physical-inventory.json",
        label="ROCm physical inventory",
    )
    physical_sha = file_sha256(physical_path)
    physical = load_rocm_physical_inventory(physical_path, physical_sha)
    all8_path = _plain_file(
        run_root / "logs/all8-rocm-runtime-mapping.json",
        label="all8 ROCm runtime mapping",
    )
    all8_sha = file_sha256(all8_path)
    all8 = load_rocm_runtime_mapping(
        all8_path, all8_sha, expected_count=8, expected_rocr=None
    )
    observed_path = _plain_file(
        run_root
        / f"logs/{ordinal}-{seed_slot}-{group_id}-rocm-runtime-mapping.json",
        label="shard ROCm runtime mapping",
    )
    observed_sha = file_sha256(observed_path)
    observed = load_rocm_runtime_mapping(
        observed_path,
        observed_sha,
        expected_count=4,
        expected_rocr=visible,
    )
    join = validate_rocm_runtime_mapping_join(
        all8_path, all8_sha, observed_path, observed_sha, physical_indices
    )
    unsigned = {
        "schema_version": BINDING_RECEIPT_SCHEMA,
        "ordinal": ordinal,
        "seed_slot": seed_slot,
        "group_id": group_id,
        "logical_visible_gpus": visible,
        "rocr_visible_devices": visible,
        "expected_physical_indices": list(physical_indices),
        "slurm_step_gpus": "0,1,2,3,4,5,6,7",
        "torch_device_count": 4,
        "world_size": 4,
        "previous_shards_completed": ordinal - 1,
        "model_subprocess_exit": 0,
        "model_process_backgrounded": False,
        "concurrent_model_replicas": 1,
        "physical_inventory": {
            "path": str(physical_path),
            "file_sha256": physical_sha,
            "inventory_digest": physical["inventory_digest"],
        },
        "all8_rocm_runtime_mapping": {
            "path": str(all8_path),
            "file_sha256": all8_sha,
            "observation_digest": all8["observation_digest"],
        },
        "observed_rocm_runtime_mapping": {
            "path": str(observed_path),
            "file_sha256": observed_sha,
            "observation_digest": observed["observation_digest"],
        },
        "all8_logical_to_physical_order": join[
            "all8_logical_to_physical_order"
        ],
        "all8_logical_to_physical_order_digest": join[
            "all8_logical_to_physical_order_digest"
        ],
        "observed_logical_to_physical_order": join[
            "observed_logical_to_physical_order"
        ],
        "observed_logical_to_physical_order_digest": join[
            "observed_logical_to_physical_order_digest"
        ],
        "observed_logical_identity_rows_digest": join[
            "observed_logical_identity_rows_digest"
        ],
        "canonical_physical_identity_rows_digest": join[
            "canonical_physical_identity_rows_digest"
        ],
        "physical_inventory_digest": join["physical_inventory_digest"],
        "pci_bus_is_authoritative_join_key": True,
        "pci_bus_physical_join_replayed": True,
        "xgmi_island_exact": True,
        "hip_logical_order_is_observation_only": True,
        "mapping_exact_physical_set_verified": True,
        "logical_order_permutation_allowed": True,
        "cross_island_visibility_rejected": True,
        "probe_receipt_pinned": False,
        "dynamic_probe_is_release_authority": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def write_gpu_binding_receipt(
    run_root: Path,
    *,
    ordinal: int,
    seed_slot: str,
    group_id: str,
    output: Path,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in GPU_SHARD_IDENTITIES
        if row[:3] == (ordinal, seed_slot, group_id)
    ]
    if len(matches) != 1:
        fail("GPU shard identity/order differs")
    _, _, _, visible, physical_indices = matches[0]
    value = _expected_gpu_binding(
        run_root,
        ordinal=ordinal,
        seed_slot=seed_slot,
        group_id=group_id,
        visible=visible,
        physical_indices=physical_indices,
    )
    _write_create_only(output, value)
    return value


def _load_gpu_binding(
    run_root: Path,
    identity: tuple[int, str, str, str, list[int]],
) -> tuple[Path, str, Mapping[str, Any]]:
    ordinal, seed_slot, group_id, visible, physical_indices = identity
    path = _plain_file(
        run_root
        / f"logs/{ordinal}-{seed_slot}-{group_id}-physical-binding.json",
        label="GPU shard binding receipt",
    )
    value = _load_canonical_receipt(path, label="GPU shard binding receipt")
    expected = _expected_gpu_binding(
        run_root,
        ordinal=ordinal,
        seed_slot=seed_slot,
        group_id=group_id,
        visible=visible,
        physical_indices=physical_indices,
    )
    if value != expected:
        fail("GPU shard physical/logical binding exact schema/replay differs")
    return path, file_sha256(path), value


def _expected_gpu_admission(
    run_root: Path, *, require_host_monitor_alive_now: bool
) -> Mapping[str, Any]:
    physical_path = _plain_file(
        run_root / "logs/gpu-physical-inventory.json",
        label="ROCm physical inventory",
    )
    physical_sha = file_sha256(physical_path)
    physical = load_rocm_physical_inventory(physical_path, physical_sha)
    all8_path = _plain_file(
        run_root / "logs/all8-rocm-runtime-mapping.json",
        label="all8 ROCm runtime mapping",
    )
    all8_sha = file_sha256(all8_path)
    all8 = load_rocm_runtime_mapping(
        all8_path, all8_sha, expected_count=8, expected_rocr=None
    )
    smoke_mapping_path = _plain_file(
        run_root / "logs/compile-smoke-rocm-runtime-mapping.json",
        label="compile-smoke ROCm runtime mapping",
    )
    smoke_mapping_sha = file_sha256(smoke_mapping_path)
    smoke_mapping = load_rocm_runtime_mapping(
        smoke_mapping_path,
        smoke_mapping_sha,
        expected_count=4,
        expected_rocr="0,1,2,3",
    )
    smoke_join = validate_rocm_runtime_mapping_join(
        all8_path,
        all8_sha,
        smoke_mapping_path,
        smoke_mapping_sha,
        [0, 1, 2, 3],
    )
    smoke_receipt_path = _plain_file(
        run_root / "logs/compile-smoke-receipt.json",
        label="full native40 compile-smoke receipt",
    )
    smoke_receipt_sha = file_sha256(smoke_receipt_path)
    try:
        smoke_receipt, replayed_smoke_path, replayed_smoke_sha = (
            reserve4_runner.load_compile_smoke_receipt(
                smoke_receipt_path, smoke_receipt_sha
            )
        )
    except reserve4_runner.Reserve4GenerationError as error:
        raise GenericActionDataPrepError(str(error)) from error
    if (
        replayed_smoke_path != smoke_receipt_path
        or replayed_smoke_sha != smoke_receipt_sha
    ):
        fail("compile-smoke receipt path/SHA replay differs")
    smoke_host_reference = smoke_receipt["host_cgroup_memory_gate"]
    try:
        smoke_host, _, _ = reserve4_runner.load_host_cgroup_memory_gate(
            smoke_host_reference["path"],
            smoke_host_reference["file_sha256"],
            expected_phase="compile_smoke_before_formal40",
            require_monitor_alive_now=require_host_monitor_alive_now,
        )
    except reserve4_runner.Reserve4GenerationError as error:
        raise GenericActionDataPrepError(str(error)) from error
    if smoke_host["receipt_digest"] != smoke_host_reference["receipt_digest"]:
        fail("compile-smoke host memory gate reference differs")
    unsigned = {
        "schema_version": GPU_ADMISSION_RECEIPT_SCHEMA,
        "all8_allocation_verified": True,
        "slurm_step_gpus": "0,1,2,3,4,5,6,7",
        "physical_gpu_count": 8,
        "gpu_unique_id_count": 8,
        "pci_bus_count": 8,
        "xgmi_islands": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "physical_inventory": {
            "path": str(physical_path),
            "file_sha256": physical_sha,
            "inventory_digest": physical["inventory_digest"],
        },
        "all8_rocm_runtime_mapping": {
            "path": str(all8_path),
            "file_sha256": all8_sha,
            "observation_digest": all8["observation_digest"],
        },
        "compile_smoke": {
            "candidate_id": smoke_receipt["smoke_task"]["candidate_id"],
            "mapping": {
                "path": str(smoke_mapping_path),
                "file_sha256": smoke_mapping_sha,
                "observation_digest": smoke_mapping["observation_digest"],
                "observed_logical_to_physical_order_digest": smoke_join[
                    "observed_logical_to_physical_order_digest"
                ],
            },
            "receipt": {
                "path": str(smoke_receipt_path),
                "file_sha256": smoke_receipt_sha,
                "receipt_digest": smoke_receipt["receipt_digest"],
            },
            "resource_lifecycle": smoke_receipt["candidate_evidence"][
                "resource_lifecycle"
            ],
            "host_cgroup_memory_gate_reference": smoke_host_reference,
            "host_cgroup_memory_gate": smoke_host,
            "world4_load_completion_ordering_asserted": True,
            "t2v_text_encoder_rank_gpu_residency_asserted": True,
            "per_rank_gpu_peak_allocated_reserved_asserted": True,
            "all_rank_gpu_peak_reserved_strictly_below_52_gib": True,
            "host_cgroup_memory_max_exactly_60_gib_at_formal_count_zero": True,
            "host_sampled_current_peak_strictly_below_56_gib_at_formal_count_zero": True,
            "host_monitor_alive_at_formal_count_zero": True,
            "host_sample_interval_ns": HOST_MEMORY_SAMPLE_INTERVAL_NS,
            "host_maximum_sample_gap_ns": HOST_MEMORY_MAX_SAMPLE_GAP_NS,
            "host_observed_tail_age_ns": smoke_host["observed_tail_age_ns"],
            "zero_oom_and_oom_kill_at_formal_count_zero": True,
            "r10_authority_replayed_from_pinned_receipt_and_log": True,
            "r10_mp4_whole_file_sha256_parity_asserted": True,
            "r10_gaussian_tensor_identity_parity_asserted": True,
            "r10_clean_latent_generated_identity_parity_asserted": True,
            "safetensors_container_sha256_equivalence_asserted": False,
            "world_size": 4,
            "full_native_sampling_steps": 40,
            "formal_candidate_count_at_gate": 0,
            "disposable_output_deleted": True,
            "exact_physical_island_verified_before_model_forward": True,
            "passed_before_formal40": True,
        },
        "formal_candidate_count_at_admission": 0,
        "run_sp4_shard_process_count": 4,
        "world4_model_invocation_count": 40,
        "compile_smoke_world4_model_invocation_count": 1,
        "total_native_model_invocation_count": 41,
        "all_model_invocations_strictly_serial": True,
        "formal_generation_requires_exact_compile_smoke_receipt": True,
        "compile_smoke_candidate_not_counted_in_formal40": True,
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
        "t2v_rank_gpu_memory_limit_gib": T2V_GPU_MEMORY_LIMIT_GIB,
        "compile_smoke_per_rank_gpu_peak_allocated_reserved_required": True,
        "compile_smoke_all_rank_gpu_peak_reserved_strictly_below_limit": True,
        "host_cgroup_sample_monitor_started_before_compile_smoke": True,
        "host_cgroup_current_pid_leaf_identity_bound": True,
        "host_cgroup_leaf_memory_max_inherited": True,
        "host_cgroup_governing_ancestor_nearest_finite": True,
        "host_cgroup_governing_scope_exact_slurm_step_user": True,
        "host_cgroup_sample_interval_ns": HOST_MEMORY_SAMPLE_INTERVAL_NS,
        "host_cgroup_max_sample_gap_ns": HOST_MEMORY_MAX_SAMPLE_GAP_NS,
        "host_live_tail_max_age_ns": HOST_MEMORY_MAX_SAMPLE_GAP_NS,
        "host_cgroup_memory_max_exactly_60_gib": True,
        "host_sampled_current_safe_ceiling_gib": HOST_MEMORY_SAFE_CEILING_GIB,
        "compile_smoke_host_sampled_peak_strictly_below_56_gib": True,
        "compile_smoke_host_monitor_alive_before_formal40": True,
        "compile_smoke_zero_oom_and_oom_kill_before_formal40": True,
        "formal_candidate_boundary_host_monitor_checks_required": True,
        "r10_smoke_authority_derived_from_pinned_receipt_and_log": True,
        "r10_smoke_mp4_whole_file_sha256_exact_required": True,
        "r10_smoke_gaussian_tensor_identity_exact_required": True,
        "r10_smoke_clean_latent_generated_identity_exact_required": True,
        "safetensors_container_sha256_cross_process_equivalence_required": False,
        "t2v_vae_load_deferred_until_rank0_post_sampling": True,
        "world4_renderer_retirement_barrier_before_rank_zero_vae_load": True,
        "per_shard_observed_uuid_pci_bus_join_before_model_forward": True,
        "pci_bus_is_authoritative_join_key": True,
        "physical_index_pci_unique_id_join_replayed": True,
        "hip_logical_order_is_observation_only": True,
        "per_shard_exact_physical_set_verified": True,
        "logical_order_permutation_allowed": True,
        "cross_island_shard_visibility_rejected": True,
        "concurrent_model_replicas": 1,
        "rank_or_gpu_action_family_partition": False,
        "sealed_shard_order": SHARD_ORDER,
        "probe_receipt_pinned": False,
        "dynamic_probe_is_release_authority": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def write_gpu_admission_receipt(
    run_root: Path,
    output: Path,
    *,
    require_host_monitor_alive_now: bool = True,
) -> Mapping[str, Any]:
    value = _expected_gpu_admission(
        run_root,
        require_host_monitor_alive_now=require_host_monitor_alive_now,
    )
    _write_create_only(output, value)
    return value


def validate_gpu_admission_receipt(run_root: Path) -> tuple[Path, str]:
    receipt_path = _plain_file(
        run_root / "gpu-admission-receipt.json", label="GPU admission receipt"
    )
    receipt = _load_canonical_receipt(
        receipt_path, label="GPU admission receipt"
    )
    if receipt != _expected_gpu_admission(
        run_root, require_host_monitor_alive_now=False
    ):
        fail("GPU admission receipt exact schema/ordered replay differs")
    return receipt_path, file_sha256(receipt_path)


def validate_terminal_host_cgroup_memory_receipt(
    run_root: Path,
    *,
    require_bound_cgroup_now: bool = False,
) -> tuple[Path, str, Mapping[str, Any]]:
    path = _plain_file(
        run_root / "logs/host-cgroup-memory-terminal-receipt.json",
        label="terminal host cgroup memory receipt",
    )
    try:
        validated, replayed_path, replayed_sha = (
            reserve4_runner.load_host_cgroup_memory_gate(
                path,
                file_sha256(path),
                expected_phase=(
                    "terminal_after_formal40_before_slurm_child_exit"
                ),
                require_monitor_alive_now=False,
                require_bound_cgroup_now=require_bound_cgroup_now,
            )
        )
    except reserve4_runner.Reserve4GenerationError as error:
        raise GenericActionDataPrepError(str(error)) from error
    if (
        replayed_path != path
        or replayed_sha != file_sha256(path)
        or validated["monitor_clean_terminal_stop"] is not True
        or validated["monitor_exit_status"] != 0
        or validated["monitor_identity_dead_at_gate"] is not True
        or validated["terminal_gate_created_after_bound_supervisor_wait"]
        is not True
        or validated["monitor_covered_compile_smoke_through_formal40"] is not True
        or validated["measurement_phase"]
        != "terminal_after_formal40_before_slurm_child_exit"
        or validated["formal_candidate_count_at_gate"] != 40
    ):
        fail("terminal host cgroup memory receipt phase differs")
    return path, file_sha256(path), validated


def validate_completion(
    plan: Mapping[str, Any], plan_path: Path, plan_sha: str
) -> Mapping[str, Any]:
    run_root = Path(plan["run_root"])
    admission_path, admission_sha = validate_gpu_admission_receipt(run_root)
    admission_receipt = _load_canonical_receipt(
        admission_path, label="GPU admission receipt"
    )
    compile_smoke = admission_receipt["compile_smoke"]
    terminal_host_path, terminal_host_sha, terminal_host = (
        validate_terminal_host_cgroup_memory_receipt(run_root)
    )
    smoke_host = compile_smoke["host_cgroup_memory_gate"]
    if (
        smoke_host["measurement_phase"] != "compile_smoke_before_formal40"
        or smoke_host["formal_candidate_count_at_gate"] != 0
        or smoke_host["monitor_start_receipt"]
        != terminal_host["monitor_start_receipt"]
        or smoke_host["monitor_pid"] != terminal_host["monitor_pid"]
        or smoke_host["monitor_proc_start_ticks"]
        != terminal_host["monitor_proc_start_ticks"]
        or smoke_host["supervisor_pid"] != terminal_host["supervisor_pid"]
        or smoke_host["supervisor_proc_start_ticks"]
        != terminal_host["supervisor_proc_start_ticks"]
        or smoke_host["monitor_stop_token_path"]
        != terminal_host["monitor_stop_token_path"]
        or smoke_host["leaf_cgroup"] != terminal_host["leaf_cgroup"]
        or smoke_host["governing_cgroup"]
        != terminal_host["governing_cgroup"]
        or smoke_host["cgroup_memory_max_bytes"]
        != terminal_host["cgroup_memory_max_bytes"]
        or {
            key: smoke_host["sample_journal"][key]
            for key in ("path", "device", "inode", "record_size", "record_encoding")
        }
        != {
            key: terminal_host["sample_journal"][key]
            for key in ("path", "device", "inode", "record_size", "record_encoding")
        }
        or int(terminal_host["sample_journal"]["prefix_byte_count"])
        <= int(smoke_host["sample_journal"]["prefix_byte_count"])
        or terminal_host["sampling"]["start_monotonic_time_ns"]
        != smoke_host["sampling"]["start_monotonic_time_ns"]
        or terminal_host["sampling"]["start_wall_time_ns"]
        != smoke_host["sampling"]["start_wall_time_ns"]
        or int(terminal_host["sampling"]["sample_count"])
        <= int(smoke_host["sampling"]["sample_count"])
        or int(terminal_host["sampling"]["end_monotonic_time_ns"])
        <= int(smoke_host["sampling"]["end_monotonic_time_ns"])
        or int(terminal_host["sampled_peak_memory_current_bytes"])
        < int(smoke_host["sampled_peak_memory_current_bytes"])
        or terminal_host["memory_events_at_gate"] != {"oom": 0, "oom_kill": 0}
        or terminal_host["sampled_peak_strictly_below_56_gib"] is not True
        or terminal_host["cgroup_memory_max_exactly_60_gib"] is not True
        or terminal_host["monitor_clean_terminal_stop"] is not True
        or terminal_host["monitor_exit_status"] != 0
        or terminal_host["monitor_identity_dead_at_gate"] is not True
        or terminal_host["terminal_gate_created_after_bound_supervisor_wait"]
        is not True
    ):
        fail("smoke/terminal host cgroup memory closure differs")
    binding_rows = []
    for identity in GPU_SHARD_IDENTITIES:
        path, sha, binding = _load_gpu_binding(run_root, identity)
        binding_rows.append(
            {
                "ordinal": identity[0],
                "shard_id": f"{identity[1]}-{identity[2]}-fit",
                "path": str(path),
                "file_sha256": sha,
                "receipt_digest": binding["receipt_digest"],
            }
        )
    result_path = _plain_file(
        run_root / "generation-audit.json", label="fit generation audit"
    )
    generation_roots = [
        run_root / "generation/seed1-sp4-a",
        run_root / "generation/seed1-sp4-b",
        run_root / "generation/seed2-sp4-a",
        run_root / "generation/seed2-sp4-b",
    ]
    closure_path = _plain_file(
        run_root / "generation-closure-receipt.json",
        label="generation closure receipt",
    )
    closure_receipt = _load_canonical_receipt(
        closure_path, label="generation closure receipt"
    )
    replayed_closure = validate_generation_output_closure(
        plan_path=Path(plan["upstream_plan"]["path"]),
        expected_plan_sha256=plan["upstream_plan"]["file_sha256"],
        generation_roots=generation_roots,
        generation_audit_path=result_path,
    )
    if closure_receipt != replayed_closure:
        fail("generation closure receipt exact schema/replay differs")
    unsigned = {
        "schema_version": COMPLETION_SCHEMA,
        "controller_plan_path": str(plan_path),
        "controller_plan_file_sha256": plan_sha,
        "controller_plan_digest": plan["plan_digest"],
        "phase": "generation",
        "analysis_split": "fit",
        "result_path": str(result_path),
        "result_file_sha256": file_sha256(result_path),
        "generation_closure_receipt_path": str(closure_path),
        "generation_closure_receipt_file_sha256": file_sha256(closure_path),
        "generation_closure_receipt_digest": closure_receipt[
            "receipt_digest"
        ],
        "candidate_count": 40,
        "seed_cell_count": 4,
        "world_size": 4,
        "slurm_child_gpu_count": 8,
        "numbered_slurm_children": 1,
        "run_sp4_shard_process_count": 4,
        "world4_model_invocation_count": 40,
        "compile_smoke_world4_model_invocation_count": 1,
        "compile_smoke_full_native_sampling_steps": 40,
        "total_native_model_invocation_count": 41,
        "compile_smoke_receipt_path": compile_smoke["receipt"]["path"],
        "compile_smoke_receipt_file_sha256": compile_smoke["receipt"][
            "file_sha256"
        ],
        "compile_smoke_receipt_digest": compile_smoke["receipt"][
            "receipt_digest"
        ],
        "compile_smoke_resource_lifecycle_digest": object_sha256(
            compile_smoke["resource_lifecycle"]
        ),
        "compile_smoke_host_cgroup_memory_gate_path": compile_smoke[
            "host_cgroup_memory_gate_reference"
        ]["path"],
        "compile_smoke_host_cgroup_memory_gate_file_sha256": compile_smoke[
            "host_cgroup_memory_gate_reference"
        ]["file_sha256"],
        "compile_smoke_host_cgroup_memory_gate_digest": smoke_host[
            "receipt_digest"
        ],
        "compile_smoke_host_sampled_peak_memory_current_bytes": smoke_host[
            "sampled_peak_memory_current_bytes"
        ],
        "compile_smoke_host_sample_count": smoke_host["sampling"][
            "sample_count"
        ],
        "compile_smoke_host_observed_maximum_gap_ns": smoke_host["sampling"][
            "observed_maximum_gap_ns"
        ],
        "compile_smoke_host_observed_tail_age_ns": smoke_host[
            "observed_tail_age_ns"
        ],
        "compile_smoke_host_memory_events": smoke_host[
            "memory_events_at_gate"
        ],
        "compile_smoke_host_leaf_cgroup": smoke_host["leaf_cgroup"],
        "compile_smoke_host_governing_cgroup": smoke_host[
            "governing_cgroup"
        ],
        "terminal_host_cgroup_memory_receipt_path": str(terminal_host_path),
        "terminal_host_cgroup_memory_receipt_file_sha256": terminal_host_sha,
        "terminal_host_cgroup_memory_receipt_digest": terminal_host[
            "receipt_digest"
        ],
        "terminal_host_sampled_peak_memory_current_bytes": terminal_host[
            "sampled_peak_memory_current_bytes"
        ],
        "terminal_host_sample_count": terminal_host["sampling"]["sample_count"],
        "terminal_host_start_monotonic_time_ns": terminal_host["sampling"][
            "start_monotonic_time_ns"
        ],
        "terminal_host_end_monotonic_time_ns": terminal_host["sampling"][
            "end_monotonic_time_ns"
        ],
        "terminal_host_observed_maximum_gap_ns": terminal_host["sampling"][
            "observed_maximum_gap_ns"
        ],
        "terminal_host_sample_journal_byte_count": terminal_host[
            "sample_journal"
        ]["prefix_byte_count"],
        "terminal_host_sample_journal_prefix_sha256": terminal_host[
            "sample_journal"
        ]["prefix_sha256"],
        "terminal_host_monitor_pid": terminal_host["monitor_pid"],
        "terminal_host_monitor_proc_start_ticks": terminal_host[
            "monitor_proc_start_ticks"
        ],
        "terminal_host_monitor_stop_token_path": terminal_host[
            "monitor_stop_token_path"
        ],
        "terminal_host_leaf_cgroup": terminal_host["leaf_cgroup"],
        "terminal_host_governing_cgroup": terminal_host[
            "governing_cgroup"
        ],
        "terminal_host_monitor_exit_status": terminal_host[
            "monitor_exit_status"
        ],
        "terminal_host_monitor_identity_dead_at_gate": terminal_host[
            "monitor_identity_dead_at_gate"
        ],
        "terminal_host_stop_observed_before_final_sample": terminal_host[
            "terminal_stop_observed_before_final_sample"
        ],
        "terminal_host_stop_final_sample_sequence": terminal_host[
            "terminal_stop_final_sample_sequence"
        ],
        "terminal_host_stop_final_sample_monotonic_time_ns": terminal_host[
            "terminal_stop_final_sample_monotonic_time_ns"
        ],
        "terminal_host_memory_events": terminal_host["memory_events_at_gate"],
        "compile_smoke_candidate_not_counted_in_formal40": True,
        "compile_smoke_disposable_output_deleted": True,
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
        "t2v_rank_gpu_memory_limit_gib": T2V_GPU_MEMORY_LIMIT_GIB,
        "compile_smoke_per_rank_gpu_peak_allocated_reserved_required": True,
        "compile_smoke_all_rank_gpu_peak_reserved_strictly_below_limit": True,
        "host_cgroup_sample_monitor_started_before_compile_smoke": True,
        "host_cgroup_current_pid_leaf_identity_bound": True,
        "host_cgroup_leaf_memory_max_inherited": True,
        "host_cgroup_governing_ancestor_nearest_finite": True,
        "host_cgroup_governing_scope_exact_slurm_step_user": True,
        "host_cgroup_sample_interval_ns": HOST_MEMORY_SAMPLE_INTERVAL_NS,
        "host_cgroup_max_sample_gap_ns": HOST_MEMORY_MAX_SAMPLE_GAP_NS,
        "host_live_tail_max_age_ns": HOST_MEMORY_MAX_SAMPLE_GAP_NS,
        "host_cgroup_memory_max_exactly_60_gib": True,
        "host_sampled_current_safe_ceiling_gib": HOST_MEMORY_SAFE_CEILING_GIB,
        "compile_smoke_host_sampled_peak_strictly_below_56_gib": True,
        "compile_smoke_host_monitor_alive_before_formal40": True,
        "compile_smoke_zero_oom_and_oom_kill_before_formal40": True,
        "formal_candidate_boundary_host_monitor_checks_required": True,
        "terminal_host_sampled_current_receipt_required": True,
        "terminal_host_sampled_peak_strictly_below_56_gib": True,
        "terminal_host_monitor_wait_exit_status_zero": True,
        "terminal_gate_created_after_bound_supervisor_wait": True,
        "terminal_host_monitor_clean_exit": True,
        "terminal_zero_oom_and_oom_kill": True,
        "r10_smoke_authority_derived_from_pinned_receipt_and_log": True,
        "r10_smoke_mp4_whole_file_sha256_exact_required": True,
        "r10_smoke_gaussian_tensor_identity_exact_required": True,
        "r10_smoke_clean_latent_generated_identity_exact_required": True,
        "safetensors_container_sha256_cross_process_equivalence_required": False,
        "t2v_vae_load_deferred_until_rank0_post_sampling": True,
        "world4_renderer_retirement_barrier_before_rank_zero_vae_load": True,
        "sealed_shard_order": SHARD_ORDER,
        "physical_island_order": PHYSICAL_ISLAND_ORDER,
        "all_model_invocations_strictly_serial": True,
        "per_shard_observed_uuid_pci_bus_join_before_model_forward": True,
        "hip_logical_order_is_observation_only": True,
        "per_shard_exact_physical_set_verified": True,
        "logical_order_permutation_allowed": True,
        "cross_island_shard_visibility_rejected": True,
        "concurrent_model_replicas": 1,
        "gpu_admission_receipt_path": str(admission_path),
        "gpu_admission_receipt_file_sha256": admission_sha,
        "formal_shard_binding_receipts": binding_rows,
        "probe_receipt_pinned": False,
        "dynamic_probe_is_release_authority": False,
        "rank_or_gpu_action_family_partition": False,
        "independent_full81_blind_review_present": False,
        "confirmation_generation_authorized": False,
        "phi_v1_extraction_authorized": False,
        "generated_media_is_editor_input_or_target": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "p_or_o_manifest_materialization_authorized": False,
        "training_authorized": False,
        "parent_allocation_released_or_cancelled": False,
    }
    completion = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_create_only(run_root / "controller-completion.json", completion)
    return completion


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--method-archive", required=True)
    parser.add_argument("--expected-method-archive-sha256", required=True)
    parser.add_argument("--method-manifest", required=True)
    parser.add_argument("--expected-method-manifest-sha256", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--seed1-spec", required=True)
    parser.add_argument("--seed2-spec", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    add_common_args(plan)
    launch = commands.add_parser("launch")
    add_common_args(launch)
    launch.add_argument("--confirm", required=True)
    validate = commands.add_parser("validate-plan")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--expected-plan-sha256", required=True)
    launch_environment = commands.add_parser("validate-launch-environment")
    launch_environment.add_argument("--plan", required=True)
    launch_environment.add_argument("--expected-plan-sha256", required=True)
    runtime = commands.add_parser("validate-runtime")
    runtime.add_argument("--method-root", required=True)
    runtime.add_argument("--method-archive", required=True)
    runtime.add_argument("--expected-method-archive-sha256", required=True)
    runtime.add_argument("--method-manifest", required=True)
    runtime.add_argument("--expected-method-manifest-sha256", required=True)
    runtime.add_argument("--python-bin", required=True)
    runtime.add_argument("--expected-python-sha256", required=True)
    physical = commands.add_parser("seal-physical-gpu-inventory")
    physical.add_argument("--identity-input", required=True)
    physical.add_argument("--topology-input", required=True)
    physical.add_argument("--output", required=True)
    observe = commands.add_parser("observe-gpu-mapping")
    observe.add_argument("--output", required=True)
    observe.add_argument("--expected-count", type=int, required=True)
    observe.add_argument("--expected-rocr", required=True)
    observe.add_argument("--physical-inventory", required=True)
    observe.add_argument("--expected-physical-inventory-sha256", required=True)
    mapping = commands.add_parser("validate-gpu-mapping")
    mapping.add_argument("--all8-mapping", required=True)
    mapping.add_argument("--expected-all8-mapping-sha256", required=True)
    mapping.add_argument("--observed-mapping", required=True)
    mapping.add_argument("--expected-observed-mapping-sha256", required=True)
    mapping.add_argument("--expected-physical-indices", required=True)
    binding = commands.add_parser("seal-gpu-binding-receipt")
    binding.add_argument("--run-root", required=True)
    binding.add_argument("--ordinal", type=int, required=True)
    binding.add_argument("--seed-slot", required=True)
    binding.add_argument("--group-id", required=True)
    binding.add_argument("--output", required=True)
    admission = commands.add_parser("seal-gpu-admission-receipt")
    admission.add_argument("--run-root", required=True)
    admission.add_argument("--output", required=True)
    host_terminal = commands.add_parser(
        "validate-terminal-host-cgroup-memory-receipt"
    )
    host_terminal.add_argument("--run-root", required=True)
    host_terminal.add_argument(
        "--require-live-child-cgroup", action="store_true"
    )
    generation = commands.add_parser("seal-generation-closure")
    generation.add_argument("--plan", required=True)
    generation.add_argument("--expected-plan-sha256", required=True)
    generation.add_argument("--generation-root", action="append", required=True)
    generation.add_argument("--generation-audit", required=True)
    generation.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "seal-physical-gpu-inventory":
        value = seal_rocm_physical_inventory(
            Path(args.identity_input), Path(args.topology_input), Path(args.output)
        )
        print(canonical_json_bytes(value).decode("ascii"), flush=True)
        return 0
    if args.command == "observe-gpu-mapping":
        expected_rocr = None if args.expected_rocr == "unset" else args.expected_rocr
        value = observe_rocm_runtime_mapping(
            Path(args.output),
            expected_count=args.expected_count,
            expected_rocr=expected_rocr,
            physical_inventory_path=Path(args.physical_inventory),
            expected_physical_inventory_sha256=(
                args.expected_physical_inventory_sha256
            ),
        )
        print(canonical_json_bytes(value).decode("ascii"), flush=True)
        return 0
    if args.command == "seal-gpu-binding-receipt":
        value = write_gpu_binding_receipt(
            Path(args.run_root),
            ordinal=args.ordinal,
            seed_slot=args.seed_slot,
            group_id=args.group_id,
            output=Path(args.output),
        )
        print(canonical_json_bytes(value).decode("ascii"), flush=True)
        return 0
    if args.command == "seal-gpu-admission-receipt":
        value = write_gpu_admission_receipt(
            Path(args.run_root), Path(args.output)
        )
        print(canonical_json_bytes(value).decode("ascii"), flush=True)
        return 0
    if args.command == "validate-terminal-host-cgroup-memory-receipt":
        _, _, value = validate_terminal_host_cgroup_memory_receipt(
            Path(args.run_root),
            require_bound_cgroup_now=args.require_live_child_cgroup,
        )
        print(canonical_json_bytes(value).decode("ascii"), flush=True)
        return 0
    if args.command == "seal-generation-closure":
        value = write_generation_closure_receipt(
            plan_path=Path(args.plan),
            expected_plan_sha256=args.expected_plan_sha256,
            generation_roots=[Path(path) for path in args.generation_root],
            generation_audit_path=Path(args.generation_audit),
            output=Path(args.output),
        )
        print(canonical_json_bytes(value).decode("ascii"), flush=True)
        return 0
    if args.command == "validate-gpu-mapping":
        try:
            physical_indices = [
                int(token) for token in args.expected_physical_indices.split(",")
            ]
        except ValueError as error:
            raise GenericActionDataPrepError(
                "expected physical ROCm indices differ"
            ) from error
        value = validate_rocm_runtime_mapping_join(
            Path(args.all8_mapping),
            args.expected_all8_mapping_sha256,
            Path(args.observed_mapping),
            args.expected_observed_mapping_sha256,
            physical_indices,
        )
        print(canonical_json_bytes(value).decode("ascii"), flush=True)
        return 0
    if args.command == "validate-plan":
        validate_controller_plan(Path(args.plan), args.expected_plan_sha256)
        return 0
    if args.command == "validate-launch-environment":
        validate_launch_environment(Path(args.plan), args.expected_plan_sha256)
        return 0
    if args.command == "validate-runtime":
        validate_release_tree(args)
        return 0
    if not 1024 <= args.master_port <= 65532:
        fail("four-port range differs")
    closure = validate_release_tree(args)
    run_root = _fresh_directory(args.run_root, label="run root")
    run_root.mkdir(mode=0o700)
    upstream_path, upstream_sha, upstream = build_upstream_plan(
        args, closure, run_root
    )
    plan, plan_path, plan_sha = build_controller_plan(
        args, closure, run_root, upstream_path, upstream_sha, upstream
    )
    validate_controller_plan(plan_path, plan_sha)
    if args.command == "plan":
        print(
            canonical_json_bytes(
                {"plan": str(plan_path), "plan_sha256": plan_sha, "launch_performed": False}
            ).decode("ascii"),
            flush=True,
        )
        return 0
    _launch(args, closure, plan, plan_path, plan_sha)
    completion = validate_completion(plan, plan_path, plan_sha)
    print(canonical_json_bytes(completion).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

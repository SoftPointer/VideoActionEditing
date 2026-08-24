from __future__ import annotations

from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


PORT_METHOD_ROOT_ENV = "GADP_GPU280_PORT_METHOD_ROOT"
R14_RUNTIME_TEST_PREIMAGE_SHA256 = (
    "1273ce6a83dc2aade967d22bf5530c582788a72d23bf52a8d75f86e43421db6a"
)
_method_root_value = os.environ.get(PORT_METHOD_ROOT_ENV)
if not _method_root_value:
    raise RuntimeError(
        f"{PORT_METHOD_ROOT_ENV} must name one extracted gpu280 exact18 method root"
    )
METHOD_ROOT = Path(_method_root_value).resolve(strict=True)
if METHOD_ROOT.is_symlink() or not METHOD_ROOT.is_dir():
    raise RuntimeError(f"{PORT_METHOD_ROOT_ENV} must resolve to a plain directory")
PACKAGE_ROOT = METHOD_ROOT.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from methods.bernini_action_editing.tools import (
    build_generic_action_data_prep_release_v1 as release,
)
from methods.bernini_action_editing import (
    generic_action_data_prep_controller_v1 as controller,
)

LAUNCHER = METHOD_ROOT / "scripts/auh_generic_action_data_prep_136309_world4_v1.sh"
LINK_MATRIX_AWK = (
    r'$1~/^GPU[0-7]$/ && NF==9 {ok=1; '
    r'for(i=2;i<=9;i++)if($i!~/^(0|XGMI|PCIE)$/)ok=0; if(ok)print}'
)
LIVE_GPU280_ROCM_SMI_TOPOLOGY = """\

============================ ROCm System Management Interface ============================
================================ Weight between two GPUs =================================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7
GPU0   0            15           15           15           72           72           72           72
GPU1   15           0            15           15           72           72           72           72
GPU2   15           15           0            15           72           72           72           72
GPU3   15           15           15           0            72           72           72           72
GPU4   72           72           72           72           0            15           15           15
GPU5   72           72           72           72           15           0            15           15
GPU6   72           72           72           72           15           15           0            15
GPU7   72           72           72           72           15           15           15           0

================================= Hops between two GPUs ==================================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7
GPU0   0            1            1            1            3            3            3            3
GPU1   1            0            1            1            3            3            3            3
GPU2   1            1            0            1            3            3            3            3
GPU3   1            1            1            0            3            3            3            3
GPU4   3            3            3            3            0            1            1            1
GPU5   3            3            3            3            1            0            1            1
GPU6   3            3            3            3            1            1            0            1
GPU7   3            3            3            3            1            1            1            0

=============================== Link Type between two GPUs ===============================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7
GPU0   0            XGMI         XGMI         XGMI         PCIE         PCIE         PCIE         PCIE
GPU1   XGMI         0            XGMI         XGMI         PCIE         PCIE         PCIE         PCIE
GPU2   XGMI         XGMI         0            XGMI         PCIE         PCIE         PCIE         PCIE
GPU3   XGMI         XGMI         XGMI         0            PCIE         PCIE         PCIE         PCIE
GPU4   PCIE         PCIE         PCIE         PCIE         0            XGMI         XGMI         XGMI
GPU5   PCIE         PCIE         PCIE         PCIE         XGMI         0            XGMI         XGMI
GPU6   PCIE         PCIE         PCIE         PCIE         XGMI         XGMI         0            XGMI
GPU7   PCIE         PCIE         PCIE         PCIE         XGMI         XGMI         XGMI         0

======================================= Numa Nodes =======================================
GPU[0]\t\t: (Topology) Numa Node: 0
GPU[0]\t\t: (Topology) Numa Affinity: 0
GPU[1]\t\t: (Topology) Numa Node: 0
GPU[1]\t\t: (Topology) Numa Affinity: 0
GPU[2]\t\t: (Topology) Numa Node: 0
GPU[2]\t\t: (Topology) Numa Affinity: 0
GPU[3]\t\t: (Topology) Numa Node: 0
GPU[3]\t\t: (Topology) Numa Affinity: 0
GPU[4]\t\t: (Topology) Numa Node: 1
GPU[4]\t\t: (Topology) Numa Affinity: 1
GPU[5]\t\t: (Topology) Numa Node: 1
GPU[5]\t\t: (Topology) Numa Affinity: 1
GPU[6]\t\t: (Topology) Numa Node: 1
GPU[6]\t\t: (Topology) Numa Affinity: 1
GPU[7]\t\t: (Topology) Numa Node: 1
GPU[7]\t\t: (Topology) Numa Affinity: 1
================================== End of ROCm SMI Log ===================================
"""

LIVE_V6_ALL8_MAPPING_FILE_SHA256 = (
    "a2b994290eefdd27608f274c797b983aea7ee0a477c0690823e1b5138c9d73bc"
)
LIVE_V6_SHARD_A_MAPPING_FILE_SHA256 = (
    "ff15d6293897791161543dd805240692164f59bbcd70eff1e93570fda43c7764"
)
LIVE_V6_PHYSICAL_INVENTORY_FILE_SHA256 = (
    "915ea6e0567891e594a472e1fb4a1c1d618a4808e774df4ddefa372922789e8e"
)
LIVE_V6_GPU_IDENTITIES = {
    0: ("0000:05:00.0", "5012f9bd2c780b", "30303530313266396264326337383062"),
    1: ("0000:08:00.0", "62d249081f1385ab", "36326432343930383166313338356162"),
    2: ("0000:47:00.0", "9f35e52888515de", "30396633356535323838383531356465"),
    3: ("0000:4a:00.0", "5f805c167d4d6149", "35663830356331363764346436313439"),
    4: ("0000:85:00.0", "4e1081af6407e1ee", "34653130383161663634303765316565"),
    5: ("0000:88:00.0", "63e61a6f7f7337a4", "36336536316136663766373333376134"),
    6: ("0000:c5:00.0", "e68afc16d72b889d", "65363861666331366437326238383964"),
    7: ("0000:c8:00.0", "f1e60a7ec19ef4c6", "66316536306137656331396566346336"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resign(value: dict, field: str = "receipt_digest") -> dict:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return {**unsigned, field: controller.object_sha256(unsigned)}


def _write_test_f32_safetensors(
    path: Path,
    *,
    key: str,
    metadata: dict[str, str],
    values: tuple[float, ...],
) -> dict[str, object]:
    payload = struct.pack(f"<{len(values)}f", *values)
    header_value = {
        "__metadata__": metadata,
        key: {
            "dtype": "F32",
            "shape": [len(values)],
            "data_offsets": [0, len(payload)],
        },
    }
    header = json.dumps(
        header_value,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return {"path": str(path), "sha256": _sha(path)}


def _valid_resource_lifecycle() -> dict:
    import infer_native_identity_generation_canary as native

    residency_rows = [
        {
            "rank": rank,
            "local_rank": rank,
            "hostname": "test-node",
            "guard_required": True,
            "guard_active": True,
            "module_path": "model.t5_text_encoder",
            "exact_positional_cpu_offload_request_only": True,
            "cpu_offload_requests_observed": 1,
            "cpu_offload_requests_suppressed": 1,
            "successful_cpu_materializations": 0,
            "delegated_to_requests": 1,
            "parameter_device_before": f"cuda:{rank}",
            "parameter_device_after": f"cuda:{rank}",
            "storage_fingerprint_before": f"{rank + 1:064x}",
            "storage_fingerprint_after": f"{rank + 1:064x}",
            "guard_method_restored": True,
            "vmrss_kib": 1000 + rank,
            "vmhwm_kib": 2000 + rank,
            "gpu_memory_limit_gib": native.T2V_GPU_MEMORY_LIMIT_GIB,
            "gpu_memory_limit_bytes": native.T2V_GPU_MEMORY_LIMIT_BYTES,
            "gpu_total_memory_bytes": 64 * 1024**3,
            "gpu_peak_allocated_bytes": 31 * 1024**3 + rank,
            "gpu_peak_reserved_bytes": 32 * 1024**3 + rank,
            "gpu_peak_reserved_within_limit": True,
        }
        for rank in range(4)
    ]
    return {
        **native.T2V_RESOURCE_LIFECYCLE_CONTRACT,
        "world4_load_completion_gate": {
            "schema_version": native.WORLD4_LOAD_COMPLETION_GATE_SCHEMA,
            "world_size": 4,
            "hostname": "test-node",
            "ranks": [0, 1, 2, 3],
            "renderer_gpu_resident_trimmed_monotonic_ns_by_rank": [
                101,
                102,
                103,
                104,
            ],
            "load_completion_barrier_returned_monotonic_ns_by_rank": [
                201,
                202,
                203,
                204,
            ],
            "source_tokenizer_setup_entered_monotonic_ns_by_rank": [
                301,
                302,
                303,
                304,
            ],
            "native_sampling_entered_monotonic_ns_by_rank": [401, 402, 403, 404],
            "world4_barrier_completed_before_source_tokenizer_setup": True,
            "all_four_renderer_loads_complete_before_any_source_tokenizer_setup": True,
            "all_four_renderer_loads_complete_before_first_native_sampling": True,
        },
        "world4_t2v_text_encoder_gpu_residency_gate": {
            "schema_version": native.T2V_TEXT_ENCODER_GPU_RESIDENCY_GATE_SCHEMA,
            "world_size": 4,
            "hostname": "test-node",
            "ranks": [0, 1, 2, 3],
            "module_path": "model.t5_text_encoder",
            "rank_evidence": residency_rows,
            "all_rank_exactly_one_cpu_offload_request_suppressed": True,
            "all_rank_zero_successful_cpu_materializations": True,
            "all_rank_gpu_resident_before_and_after_sampling": True,
            "all_rank_storage_fingerprint_unchanged": True,
            "all_rank_guard_method_restored": True,
            "all_rank_peak_reserved_within_52_gib": True,
        },
    }


def _write_auh_cgroup_hierarchy_fixture(
    root: Path,
    *,
    step_id: str = "1",
    job_max: str | int = 64 * 1024**3,
    step_max: str | int = "max",
    user_max: str | int = 60 * 1024**3,
    leaf_max: str | int = "max",
    selected_current: int = 8 * 1024**3,
    selected_oom: int = 0,
    selected_oom_kill: int = 0,
    holder_job_historical_oom: int = 5,
    holder_job_historical_oom_kill: int = 2,
    root_controller_files_present: bool = False,
    root_max: str | int = "max",
) -> tuple[dict, Path, Path, dict[str, Path]]:
    """Materialize the AUH hierarchy, including its file-less cgroup2 root."""

    runner = controller.reserve4_runner
    mount = (root / "cgroup-v2").resolve()
    relative_components = [
        "system.slice",
        "slurmstepd.scope",
        "job_136309",
        f"step_{step_id}",
        "user",
        "task_0",
    ]
    paths: dict[str, Path] = {"/": mount}
    current = mount
    relative = ""
    mount.mkdir(parents=True)
    for component in relative_components:
        current = current / component
        current.mkdir()
        relative += "/" + component
        paths[relative] = current
    job_relative = "/system.slice/slurmstepd.scope/job_136309"
    step_relative = f"{job_relative}/step_{step_id}"
    user_relative = f"{step_relative}/user"
    leaf_relative = f"{user_relative}/task_0"
    maximums: dict[str, str | int] = {
        "/": root_max,
        "/system.slice": "max",
        "/system.slice/slurmstepd.scope": "max",
        job_relative: job_max,
        step_relative: step_max,
        user_relative: user_max,
        leaf_relative: leaf_max,
    }
    selected_relative = (
        user_relative
        if user_max == runner.HOST_MEMORY_LIMIT_BYTES
        else step_relative
    )
    for row_relative, path in paths.items():
        if row_relative == "/" and not root_controller_files_present:
            continue
        (path / "memory.max").write_text(str(maximums[row_relative]) + "\n", encoding="ascii")
        (path / "memory.current").write_text(
            str(selected_current if row_relative == selected_relative else 1024) + "\n",
            encoding="ascii",
        )
        (path / "memory.events").write_text(
            "low 0\nhigh 0\nmax 0\noom %d\noom_kill %d\n"
            % (
                (
                    selected_oom
                    if row_relative == selected_relative
                    else holder_job_historical_oom
                    if row_relative == job_relative
                    else 0
                ),
                (
                    selected_oom_kill
                    if row_relative == selected_relative
                    else holder_job_historical_oom_kill
                    if row_relative == job_relative
                    else 0
                ),
            ),
            encoding="ascii",
        )
    proc_cgroup = (root / "proc-self-cgroup").resolve()
    proc_cgroup.write_text(f"0::{leaf_relative}\n", encoding="ascii")
    binding = runner._discover_live_cgroup_v2(
        proc_cgroup_path=proc_cgroup,
        cgroup_root=mount,
        slurm_job_id="136309",
        slurm_step_id=step_id,
    )
    return binding, proc_cgroup, mount, paths


def _write_host_memory_gate_fixture(
    root: Path,
    *,
    compile_peak_bytes: int = 40 * 1024**3,
    terminal_peak_bytes: int = 45 * 1024**3,
    wall_time_offsets_ns: tuple[int, int, int, int] | None = None,
) -> tuple[dict, Path, dict, Path, dict]:
    runner = controller.reserve4_runner
    root.mkdir(parents=True, exist_ok=True)
    journal = root / "host-cgroup-memory-current-samples.bin"
    start_path = root / "host-cgroup-memory-monitor-start-receipt.json"
    compile_path = root / "host-cgroup-memory-compile-smoke-gate.json"
    terminal_path = root / "host-cgroup-memory-terminal-receipt.json"
    wall0 = 2_000_000_000_000
    mono0 = 1_000_000_000_000
    wall_offsets = wall_time_offsets_ns or tuple(
        sequence * runner.HOST_MEMORY_SAMPLE_INTERVAL_NS for sequence in range(4)
    )
    currents = [8 * 1024**3, compile_peak_bytes, terminal_peak_bytes, 9 * 1024**3]
    rows = [
        (
            sequence,
            wall0 + wall_offsets[sequence],
            mono0 + sequence * runner.HOST_MEMORY_SAMPLE_INTERVAL_NS,
            current,
            runner.HOST_MEMORY_LIMIT_BYTES,
            0,
            0,
            1 if sequence == 3 else 0,
        )
        for sequence, current in enumerate(currents)
    ]
    raw = b"".join(runner.HOST_MEMORY_SAMPLE_STRUCT.pack(*row) for row in rows)
    journal.write_bytes(raw)
    journal.chmod(0o400)
    metadata = journal.stat()
    cgroup_binding, _, _, _ = _write_auh_cgroup_hierarchy_fixture(
        root / "auh-cgroup"
    )
    monitor_pid = os.getpid() + 100_000
    monitor_ticks = 123456
    supervisor_pid = os.getpid()
    supervisor_ticks = 234567
    unsigned_start = {
        "schema_version": runner.HOST_CGROUP_MEMORY_MONITOR_START_SCHEMA,
        "monitor_pid": monitor_pid,
        "monitor_proc_start_ticks": monitor_ticks,
        "supervisor_pid": supervisor_pid,
        "supervisor_proc_start_ticks": supervisor_ticks,
        "slurm_job_id": "136309",
        "slurm_step_id": "1",
        "monitor_started_before_compile_smoke_and_formal40": True,
        "cgroup_binding": cgroup_binding,
        "sampling_source": runner.HOST_MEMORY_SAMPLING_SOURCE,
        "sample_journal": {
            "path": str(journal),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "record_size": runner.HOST_MEMORY_SAMPLE_STRUCT.size,
            "record_encoding": runner.HOST_MEMORY_SAMPLE_ENCODING,
        },
        "sample_interval_ns": runner.HOST_MEMORY_SAMPLE_INTERVAL_NS,
        "maximum_sample_gap_ns": runner.HOST_MEMORY_MAX_SAMPLE_GAP_NS,
        "strict_host_memory_limit_gib": runner.HOST_MEMORY_LIMIT_GIB,
        "strict_host_memory_limit_bytes": runner.HOST_MEMORY_LIMIT_BYTES,
        "host_memory_safe_ceiling_gib": runner.HOST_MEMORY_SAFE_CEILING_GIB,
        "host_memory_safe_ceiling_bytes": runner.HOST_MEMORY_SAFE_CEILING_BYTES,
        "initial_sample": runner._sample_row(rows[0]),
    }
    start = {**unsigned_start, "receipt_digest": runner.object_sha256(unsigned_start)}
    start_path.write_bytes(runner.canonical_json_bytes(start) + b"\n")
    start_sha = _sha(start_path)
    compile_raw = raw[: 2 * runner.HOST_MEMORY_SAMPLE_STRUCT.size]
    compile_gate = runner._derive_host_cgroup_memory_gate(
        start_receipt=start,
        start_receipt_path=start_path,
        start_receipt_sha256=start_sha,
        raw_prefix=compile_raw,
        rows=rows[:2],
        measurement_phase="compile_smoke_before_formal40",
        formal_candidate_count_at_gate=0,
        live_tail_observed_monotonic_time_ns=(
            rows[1][2] + 1_000_000
        ),
    )
    compile_path.write_bytes(runner.canonical_json_bytes(compile_gate) + b"\n")
    terminal_gate = runner._derive_host_cgroup_memory_gate(
        start_receipt=start,
        start_receipt_path=start_path,
        start_receipt_sha256=start_sha,
        raw_prefix=raw,
        rows=rows,
        measurement_phase="terminal_after_formal40_before_slurm_child_exit",
        formal_candidate_count_at_gate=40,
    )
    terminal_path.write_bytes(runner.canonical_json_bytes(terminal_gate) + b"\n")
    return start, compile_path, compile_gate, terminal_path, terminal_gate


class GenericActionDataPrepReleaseTests(unittest.TestCase):
    @staticmethod
    def _dummy_root(root: Path) -> Path:
        method_root = root / "methods/bernini_action_editing"
        for index, (relative, mode) in enumerate(release.FILES_AND_MODES.items()):
            path = method_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"member-{index}-{relative}\n".encode("ascii"))
            path.chmod(mode)
        return method_root.resolve()

    def test_release_is_deterministic_exact_fit40_and_data_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            method_root = self._dummy_root(root)
            archive1, manifest1 = root / "one.tar", root / "one.json"
            archive2, manifest2 = root / "two.tar", root / "two.json"
            receipt1 = release.build(method_root, archive1, manifest1)
            receipt2 = release.build(method_root, archive2, manifest2)
            self.assertEqual(archive1.read_bytes(), archive2.read_bytes())
            self.assertEqual(manifest1.read_bytes(), manifest2.read_bytes())
            value = release.audit(
                archive1,
                manifest1,
                expected_archive_sha256=receipt1["archive_sha256"],
                expected_manifest_sha256=receipt1["manifest_sha256"],
            )
            self.assertEqual(
                value["release_scope"],
                "reserve4-fit40-media-only-pending-external-blind-review",
            )
            self.assertEqual(value["release_generation"], "r14")
            self.assertEqual(value["authority"]["analysis_split"], "fit")
            self.assertEqual(value["authority"]["candidate_count"], 40)
            self.assertEqual(value["authority"]["seed_cell_count"], 4)
            self.assertFalse(
                value["authority"]["independent_full81_blind_review_present"]
            )
            self.assertFalse(value["authority"]["confirmation_generation_authorized"])
            self.assertFalse(value["authority"]["phi_v1_extraction_authorized"])
            self.assertFalse(value["authority"]["optimizer_authorized"])
            self.assertFalse(value["authority"]["training_authorized"])
            self.assertEqual(value["topology"]["slurm_child_gpu_count"], 8)
            self.assertEqual(value["topology"]["numbered_slurm_children"], 1)
            self.assertEqual(value["topology"]["run_sp4_shard_process_count"], 4)
            self.assertEqual(value["topology"]["world4_model_invocation_count"], 40)
            self.assertEqual(
                value["topology"]["compile_smoke_world4_model_invocation_count"], 1
            )
            self.assertEqual(value["topology"]["total_native_model_invocation_count"], 41)
            self.assertTrue(value["topology"]["per_rank_node_local_cache_wrapper"])
            self.assertTrue(value["topology"]["nfs_comgr_tmp_rejected"])
            self.assertTrue(
                value["topology"]["serialized_world4_host_checkpoint_load"]
            )
            self.assertTrue(value["topology"]["model_load_lock_node_local"])
            self.assertTrue(
                value["topology"][
                    "model_load_lock_held_through_gpu_move_and_malloc_trim"
                ]
            )
            self.assertTrue(
                value["topology"][
                    "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup"
                ]
            )
            self.assertTrue(
                value["topology"][
                    "resource_lifecycle_receipt_proves_all_four_load_complete_before_first_sampling"
                ]
            )
            self.assertTrue(
                value["topology"][
                    "compile_smoke_asserts_world4_load_completion_ordering"
                ]
            )
            self.assertEqual(
                value["topology"]["t2v_rank_gpu_memory_limit_gib"], 52
            )
            self.assertTrue(
                value["topology"][
                    "compile_smoke_all_rank_gpu_peak_reserved_strictly_below_limit"
                ]
            )
            self.assertTrue(
                value["topology"][
                    "host_cgroup_sample_monitor_started_before_compile_smoke"
                ]
            )
            self.assertEqual(
                value["topology"]["host_cgroup_sample_interval_ns"], 10_000_000
            )
            self.assertEqual(
                value["topology"]["host_cgroup_max_sample_gap_ns"], 100_000_000
            )
            self.assertEqual(
                value["topology"]["host_sampled_current_safe_ceiling_gib"], 56
            )
            self.assertTrue(
                value["topology"][
                    "compile_smoke_zero_oom_and_oom_kill_before_formal40"
                ]
            )
            self.assertTrue(
                value["topology"][
                    "terminal_host_sampled_peak_strictly_below_56_gib"
                ]
            )
            self.assertTrue(
                value["topology"]["terminal_zero_oom_and_oom_kill"]
            )
            self.assertTrue(
                value["topology"]["t2v_vae_load_deferred_until_rank0_post_sampling"]
            )
            self.assertTrue(
                value["topology"][
                    "world4_renderer_retirement_barrier_before_rank_zero_vae_load"
                ]
            )
            self.assertEqual(
                value["topology"]["sealed_shard_order"], controller.SHARD_ORDER
            )
            self.assertEqual(
                value["topology"]["physical_island_order"],
                controller.PHYSICAL_ISLAND_ORDER,
            )
            self.assertTrue(
                value["topology"]["all_model_invocations_strictly_serial"]
            )
            self.assertTrue(
                value["topology"][
                    "per_shard_observed_uuid_pci_bus_join_before_model_forward"
                ]
            )
            self.assertTrue(value["topology"]["pci_bus_is_authoritative_join_key"])
            self.assertTrue(
                value["topology"][
                    "logical_to_physical_mapping_uses_pci_and_unique_id"
                ]
            )
            self.assertTrue(
                value["topology"]["per_shard_exact_physical_set_verified"]
            )
            self.assertTrue(
                value["topology"]["within_island_logical_permutation_allowed"]
            )
            self.assertTrue(
                value["topology"]["cross_island_shard_visibility_rejected"]
            )
            self.assertEqual(value["topology"]["concurrent_model_replicas"], 1)
            self.assertFalse(value["topology"]["rank_or_gpu_action_family_partition"])
            self.assertTrue(
                value["topology"][
                    "all_rows_share_one_generic_representation_contract"
                ]
            )
            self.assertEqual(value["topology"]["host_memory_request_gib"], 60)
            probe = value["topology"][
                "non_authoritative_dynamic_probe_observation"
            ]
            self.assertFalse(probe["probe_receipt_pinned"])
            self.assertFalse(probe["is_release_authority"])

    def test_release_excludes_phi_review_manifest_and_action_capabilities(self) -> None:
        forbidden = {
            "train_generic_source_anchored_action_v1.py",
            "generic_source_anchored_action_v1.py",
            "tools/generic_action_manifest_v1.py",
            "tools/materialize_phi_v1_sidecars_sp4.py",
            "assets/representation_train_manifest_v1.json",
            "assets/action_source_pair_manifest_v1.json",
        }
        self.assertTrue(forbidden.isdisjoint(release.FILES_AND_MODES))
        self.assertTrue(forbidden.isdisjoint(release.ENTRYPOINTS))
        self.assertNotIn("train_lora.py", release.ENTRYPOINTS)
        self.assertEqual(
            set(release.ENTRYPOINTS),
            {
                "generic_action_data_prep_controller_v1.py",
                "scripts/auh_generic_action_data_prep_136309_world4_v1.sh",
            },
        )
        self.assertIn(
            "tools/reserve4_fixed_generation_sp4_v1.py",
            release.FILES_AND_MODES,
        )
        self.assertIn("tools/build_renderer_dataset.py", release.FILES_AND_MODES)
        self.assertIn("tools/materialize_vae.py", release.FILES_AND_MODES)
        self.assertIn(
            "scripts/auh_generic_action_data_prep_rank_exec_v1.sh",
            release.FILES_AND_MODES,
        )
        self.assertNotIn(
            "tools/reserve4_fixed_generation_sp4_v1.py",
            release.ENTRYPOINTS,
        )

    def test_release_audit_rejects_tampered_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            method_root = self._dummy_root(root)
            archive, manifest = root / "source.tar", root / "source.json"
            receipt = release.build(method_root, archive, manifest)
            raw = bytearray(archive.read_bytes())
            raw[0] ^= 1
            archive.chmod(0o600)
            archive.write_bytes(bytes(raw))
            with self.assertRaisesRegex(
                release.GenericActionDataReleaseError, "archive SHA-256 differs"
            ):
                release.audit(
                    archive,
                    manifest,
                    expected_archive_sha256=receipt["archive_sha256"],
                    expected_manifest_sha256=receipt["manifest_sha256"],
                )

    def test_extracted_release_binds_exact_lazy_preprocessing_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive, manifest = root / "source.tar", root / "manifest.json"
            release.build(METHOD_ROOT, archive, manifest)
            extracted = root / "extracted"
            extracted.mkdir()
            with tarfile.open(archive, "r:") as handle:
                handle.extractall(extracted)
            method_root = extracted / "methods/bernini_action_editing"
            environment = {
                **os.environ,
                "PYTHONPATH": str(method_root),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
            normal = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        "import json, pathlib; "
                        "import infer_pair_v5_t2v_calibration_bank as pair; "
                        "import tools.build_renderer_dataset as raw; "
                        "import tools.materialize_vae as vae; "
                        "print(json.dumps({'raw': str(pathlib.Path(raw.__file__).resolve()), "
                        "'vae': str(pathlib.Path(vae.__file__).resolve()), "
                        "'ids': dict(pair.RELEASE_PREPROCESSING_TOOL_IDENTITIES)}, sort_keys=True))"
                    ),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
            observed = json.loads(normal.stdout.splitlines()[-1])
            self.assertEqual(
                observed["raw"], str((method_root / "tools/build_renderer_dataset.py").resolve())
            )
            self.assertEqual(
                observed["vae"], str((method_root / "tools/materialize_vae.py").resolve())
            )
            self.assertEqual(
                observed["ids"],
                {
                    "tools.build_renderer_dataset": controller.reserve4_runner.PREPROCESSING_TOOL_SHA256[
                        "tools/build_renderer_dataset.py"
                    ],
                    "tools.materialize_vae": controller.reserve4_runner.PREPROCESSING_TOOL_SHA256[
                        "tools/materialize_vae.py"
                    ],
                },
            )
            hostile = root / "hostile"
            hostile.mkdir()
            hostile_result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        "import pathlib, sys, types; "
                        f"pkg=types.ModuleType('tools'); pkg.__path__={[str(hostile)]!r}; "
                        "sys.modules['tools']=pkg; "
                        "import infer_pair_v5_t2v_calibration_bank"
                    ),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
            self.assertNotEqual(hostile_result.returncode, 0)
            self.assertIn(
                "preloaded tools package is outside this release",
                hostile_result.stderr,
            )


class GenericActionDataPrepControllerTests(unittest.TestCase):
    def test_live_cgroup_current_sampler_needs_no_kernel_peak_file(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binding, _, _, paths = _write_auh_cgroup_hierarchy_fixture(root)
            sample = runner._sample_row(
                runner._sample_live_cgroup_memory(binding, sequence=0)
            )
            self.assertEqual(sample["memory_current_bytes"], 8 * 1024**3)
            self.assertEqual(sample["memory_max_bytes"], 60 * 1024**3)
            self.assertEqual(sample["memory_events"], {"oom": 0, "oom_kill": 0})
            self.assertEqual(binding["self_leaf"]["memory_max_raw"], "max")
            self.assertEqual(
                binding["sampled_limit_cgroup"]["relative_path"],
                "/system.slice/slurmstepd.scope/job_136309/step_1/user",
            )
            self.assertTrue(
                binding["sampled_limit_cgroup"][
                    "covers_all_rank_and_child_cgroups"
                ]
            )
            self.assertEqual(
                binding["ancestor_chain_leaf_to_root"][1]["memory_max_bytes"],
                60 * 1024**3,
            )
            self.assertEqual(
                binding["ancestor_chain_leaf_to_root"][2]["memory_max_raw"],
                "max",
            )
            self.assertEqual(
                binding["ancestor_chain_leaf_to_root"][3]["memory_max_bytes"],
                64 * 1024**3,
            )
            hierarchy_root = binding["ancestor_chain_leaf_to_root"][-1]
            self.assertEqual(hierarchy_root["relative_path"], "/")
            self.assertFalse(hierarchy_root["memory_controller_file_present"])
            self.assertIsNone(hierarchy_root["memory_max_file_identity"])
            self.assertIsNone(hierarchy_root["memory_max_raw"])
            self.assertIsNone(hierarchy_root["memory_max_bytes"])
            self.assertEqual(
                hierarchy_root["directory_identity"],
                {
                    "device": binding["cgroup_mount"]["device"],
                    "inode": binding["cgroup_mount"]["inode"],
                },
            )
            self.assertIn(
                "oom 5\noom_kill 2",
                (
                    paths[
                        "/system.slice/slurmstepd.scope/job_136309"
                    ]
                    / "memory.events"
                ).read_text(encoding="ascii"),
            )
            group = paths[binding["sampled_limit_cgroup"]["relative_path"]]
            for case in ("wrong-max", "at-safe-ceiling", "oom", "oom-kill"):
                with self.subTest(case=case):
                    (group / "memory.current").write_text(
                        str(
                            runner.HOST_MEMORY_SAFE_CEILING_BYTES
                            if case == "at-safe-ceiling"
                            else 8 * 1024**3
                        ),
                        encoding="ascii",
                    )
                    (group / "memory.max").write_text(
                        str(
                            59 * 1024**3
                            if case == "wrong-max"
                            else runner.HOST_MEMORY_LIMIT_BYTES
                        ),
                        encoding="ascii",
                    )
                    (group / "memory.events").write_text(
                        "low 0\nhigh 0\nmax 0\noom %d\noom_kill %d\n"
                        % (case == "oom", case == "oom-kill"),
                        encoding="ascii",
                    )
                    with self.assertRaises(runner.Reserve4GenerationError):
                        runner._sample_live_cgroup_memory(binding, sequence=1)

    def test_cgroup_root_memory_files_are_optional_identity_only_evidence(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            absent, _, _, _ = _write_auh_cgroup_hierarchy_fixture(
                root / "real-root-absent"
            )
            present, _, _, _ = _write_auh_cgroup_hierarchy_fixture(
                root / "synthetic-root-present",
                root_controller_files_present=True,
            )
            absent_root = absent["ancestor_chain_leaf_to_root"][-1]
            present_root = present["ancestor_chain_leaf_to_root"][-1]
            self.assertFalse(absent_root["memory_controller_file_present"])
            self.assertTrue(present_root["memory_controller_file_present"])
            self.assertEqual(present_root["memory_max_raw"], "max")
            self.assertIsNone(present_root["memory_max_bytes"])
            for field in ("relative_path", "distance_from_leaf", "scope"):
                self.assertEqual(
                    absent["sampled_limit_cgroup"][field],
                    present["sampled_limit_cgroup"][field],
                )
            absent_resources = [
                (row["relative_path"], row["memory_max_raw"], row["memory_max_bytes"])
                for row in absent["ancestor_chain_leaf_to_root"]
                if row["relative_path"] != "/"
            ]
            present_resources = [
                (row["relative_path"], row["memory_max_raw"], row["memory_max_bytes"])
                for row in present["ancestor_chain_leaf_to_root"]
                if row["relative_path"] != "/"
            ]
            self.assertEqual(absent_resources, present_resources)
            for name, root_maximum in (
                ("undercut-59g", 59 * 1024**3),
                ("ambiguous-60g", runner.HOST_MEMORY_LIMIT_BYTES),
            ):
                with self.subTest(case=name), self.assertRaises(
                    runner.Reserve4GenerationError
                ):
                    _write_auh_cgroup_hierarchy_fixture(
                        root / name,
                        root_controller_files_present=True,
                        root_max=root_maximum,
                    )
            root64, _, _, _ = _write_auh_cgroup_hierarchy_fixture(
                root / "finite-root-64g",
                root_controller_files_present=True,
                root_max=runner.HOLDER_JOB_MEMORY_LIMIT_BYTES,
            )
            root64_row = root64["ancestor_chain_leaf_to_root"][-1]
            self.assertTrue(root64_row["memory_controller_file_present"])
            self.assertEqual(
                root64_row["memory_max_bytes"],
                runner.HOLDER_JOB_MEMORY_LIMIT_BYTES,
            )

    def test_cgroup_nonroot_memory_max_and_slurm_scope_directories_are_mandatory(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            baseline, proc_cgroup, mount, paths = _write_auh_cgroup_hierarchy_fixture(
                root / "baseline"
            )
            nonroot_relatives = [
                row["relative_path"]
                for row in baseline["ancestor_chain_leaf_to_root"]
                if row["relative_path"] != "/"
            ]
            for index, relative in enumerate(nonroot_relatives):
                with self.subTest(case=f"missing-memory-max-{relative}"):
                    case_root = root / f"missing-file-{index}"
                    _, case_proc, case_mount, case_paths = (
                        _write_auh_cgroup_hierarchy_fixture(case_root)
                    )
                    (case_paths[relative] / "memory.max").unlink()
                    with self.assertRaises(runner.Reserve4GenerationError):
                        runner._discover_live_cgroup_v2(
                            proc_cgroup_path=case_proc,
                            cgroup_root=case_mount,
                            slurm_job_id="136309",
                            slurm_step_id="1",
                        )
            for scope_relative in (
                "/system.slice/slurmstepd.scope/job_136309",
                "/system.slice/slurmstepd.scope/job_136309/step_1",
                "/system.slice/slurmstepd.scope/job_136309/step_1/user",
            ):
                with self.subTest(case=f"missing-directory-{scope_relative}"):
                    suffix = scope_relative.rsplit("/", 1)[-1]
                    case_root = root / f"missing-directory-{suffix}"
                    _, case_proc, case_mount, case_paths = (
                        _write_auh_cgroup_hierarchy_fixture(case_root)
                    )
                    target = case_paths[scope_relative]
                    target.rename(target.with_name(target.name + "-removed"))
                    with self.assertRaises(runner.Reserve4GenerationError):
                        runner._discover_live_cgroup_v2(
                            proc_cgroup_path=case_proc,
                            cgroup_root=case_mount,
                            slurm_job_id="136309",
                            slurm_step_id="1",
                        )

    def test_cgroup_root_identity_namespace_root_and_path_escape_fail_closed(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binding, proc_cgroup, mount, paths = _write_auh_cgroup_hierarchy_fixture(
                root / "base"
            )
            hostile = json.loads(json.dumps(binding))
            hostile["ancestor_chain_leaf_to_root"][-1]["directory_identity"][
                "inode"
            ] += 1
            hostile["ancestor_chain_digest"] = runner.object_sha256(
                hostile["ancestor_chain_leaf_to_root"]
            )
            unsigned = dict(hostile)
            unsigned.pop("binding_digest", None)
            hostile["binding_digest"] = runner.object_sha256(unsigned)
            with self.assertRaises(runner.Reserve4GenerationError):
                runner._validate_cgroup_binding(hostile)

            for name, membership in (
                ("namespace-root", "0::/\n"),
                (
                    "path-escape",
                    "0::/system.slice/slurmstepd.scope/job_136309/../"
                    "job_136309/step_1/user/task_0\n",
                ),
                (
                    "ambiguous-unified-membership",
                    "0::/system.slice/slurmstepd.scope/job_136309/step_1/user/task_0\n"
                    "0::/system.slice/slurmstepd.scope/job_136309/step_1/user/task_1\n",
                ),
            ):
                with self.subTest(case=name):
                    proc_cgroup.write_text(membership, encoding="ascii")
                    with self.assertRaises(runner.Reserve4GenerationError):
                        runner._discover_live_cgroup_v2(
                            proc_cgroup_path=proc_cgroup,
                            cgroup_root=mount,
                            slurm_job_id="136309",
                            slurm_step_id="1",
                        )
            proc_cgroup.write_text(
                "0::/system.slice/slurmstepd.scope/job_136309/step_1/user/task_0\n",
                encoding="ascii",
            )
            with self.subTest(case="mount-root-mismatch"):
                with self.assertRaises(runner.Reserve4GenerationError):
                    runner._discover_live_cgroup_v2(
                        proc_cgroup_path=proc_cgroup,
                        cgroup_root=paths["/system.slice"],
                        slurm_job_id="136309",
                        slurm_step_id="1",
                    )
            with self.subTest(case="root-memory-max-symlink"):
                (mount / "memory.max").symlink_to(
                    paths[
                        "/system.slice/slurmstepd.scope/job_136309"
                    ] / "memory.max"
                )
                with self.assertRaises(runner.Reserve4GenerationError):
                    runner._discover_live_cgroup_v2(
                        proc_cgroup_path=proc_cgroup,
                        cgroup_root=mount,
                        slurm_job_id="136309",
                        slurm_step_id="1",
                    )

    def test_cgroup_discovery_selects_only_unique_nearest_step_wide_60g_ancestor(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name, parameters, selected_suffix in (
                ("auh-user-limit", {}, "/step_1/user"),
                (
                    "step-limit",
                    {"step_max": runner.HOST_MEMORY_LIMIT_BYTES, "user_max": "max"},
                    "/step_1",
                ),
            ):
                with self.subTest(case=name):
                    binding, _, _, _ = _write_auh_cgroup_hierarchy_fixture(
                        root / name, **parameters
                    )
                    self.assertTrue(
                        binding["sampled_limit_cgroup"]["relative_path"].endswith(
                            selected_suffix
                        )
                    )
                    self.assertNotEqual(
                        binding["sampled_limit_cgroup"]["relative_path"],
                        binding["self_leaf"]["relative_path"],
                    )
                    self.assertEqual(
                        runner._validate_cgroup_binding(binding), binding
                    )
            hostile = (
                ("missing-60", {"user_max": "max"}),
                (
                    "ambiguous-two-60",
                    {"step_max": runner.HOST_MEMORY_LIMIT_BYTES},
                ),
                ("wrong-job-64", {"job_max": 65 * 1024**3}),
                ("finite-59-undercut", {"step_max": 59 * 1024**3}),
                (
                    "task-leaf-60",
                    {"user_max": "max", "leaf_max": runner.HOST_MEMORY_LIMIT_BYTES},
                ),
            )
            for name, parameters in hostile:
                with self.subTest(case=name), self.assertRaises(
                    runner.Reserve4GenerationError
                ):
                    _write_auh_cgroup_hierarchy_fixture(root / name, **parameters)

    def test_cgroup_discovery_rejects_cross_hierarchy_symlinks_and_membership_drift(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binding, proc_cgroup, mount, paths = _write_auh_cgroup_hierarchy_fixture(
                root / "base"
            )
            with self.subTest(case="wrong-step"):
                with self.assertRaises(runner.Reserve4GenerationError):
                    runner._discover_live_cgroup_v2(
                        proc_cgroup_path=proc_cgroup,
                        cgroup_root=mount,
                        slurm_job_id="136309",
                        slurm_step_id="2",
                    )
            with self.subTest(case="membership-drift"):
                proc_cgroup.write_text(
                    "0::/system.slice/slurmstepd.scope/job_136310/step_1/user/task_0\n",
                    encoding="ascii",
                )
                with self.assertRaisesRegex(
                    runner.Reserve4GenerationError, "left the bound"
                ):
                    runner._sample_live_cgroup_memory(binding, sequence=1)
                proc_cgroup.write_text(
                    "0::/system.slice/slurmstepd.scope/job_136309/step_1/user/task_0\n",
                    encoding="ascii",
                )
            with self.subTest(case="intermediate-symlink"):
                user = paths[
                    "/system.slice/slurmstepd.scope/job_136309/step_1/user"
                ]
                real_user = user.with_name("real-user")
                user.rename(real_user)
                user.symlink_to(real_user, target_is_directory=True)
                with self.assertRaises(runner.Reserve4GenerationError):
                    runner._discover_live_cgroup_v2(
                        proc_cgroup_path=proc_cgroup,
                        cgroup_root=mount,
                        slurm_job_id="136309",
                        slurm_step_id="1",
                    )
                user.unlink()
                real_user.rename(user)
            with self.subTest(case="control-file-symlink"):
                current = paths[
                    "/system.slice/slurmstepd.scope/job_136309/step_1/user"
                ] / "memory.current"
                leaf_current = paths[
                    "/system.slice/slurmstepd.scope/job_136309/step_1/user/task_0"
                ] / "memory.current"
                current.unlink()
                current.symlink_to(leaf_current)
                with self.assertRaises(runner.Reserve4GenerationError):
                    runner._discover_live_cgroup_v2(
                        proc_cgroup_path=proc_cgroup,
                        cgroup_root=mount,
                        slurm_job_id="136309",
                        slurm_step_id="1",
                    )
            with self.subTest(case="cgroup-root-symlink"):
                link = root / "cgroup-link"
                link.symlink_to(mount, target_is_directory=True)
                with self.assertRaises(runner.Reserve4GenerationError):
                    runner._discover_live_cgroup_v2(
                        proc_cgroup_path=proc_cgroup,
                        cgroup_root=link,
                        slurm_job_id="136309",
                        slurm_step_id="1",
                    )

    def test_cgroup_pinned_fds_do_not_drift_and_fresh_rewalk_rejects_replacement(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binding, proc_cgroup, mount, paths = _write_auh_cgroup_hierarchy_fixture(root)
            opened, descriptors = runner._open_live_cgroup_v2(
                proc_cgroup_path=proc_cgroup,
                cgroup_root=mount,
                slurm_job_id="136309",
                slurm_step_id="1",
            )
            self.assertEqual(opened, binding)
            user_relative = opened["sampled_limit_cgroup"]["relative_path"]
            user = paths[user_relative]
            old_user = user.with_name("user-old")
            try:
                user.rename(old_user)
                user.mkdir()
                (user / "memory.max").write_text(
                    str(runner.HOST_MEMORY_LIMIT_BYTES) + "\n", encoding="ascii"
                )
                (user / "memory.current").write_text(
                    str(9 * 1024**3) + "\n", encoding="ascii"
                )
                (user / "memory.events").write_text(
                    "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n", encoding="ascii"
                )
                task = user / "task_0"
                task.mkdir()
                (task / "memory.max").write_text("max\n", encoding="ascii")
                (task / "memory.current").write_text("1024\n", encoding="ascii")
                (task / "memory.events").write_text(
                    "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n", encoding="ascii"
                )
                pinned = runner._sample_row(
                    runner._sample_live_cgroup_memory(
                        opened,
                        sequence=1,
                        measurement_descriptors=descriptors,
                    )
                )
                self.assertEqual(pinned["memory_current_bytes"], 8 * 1024**3)
                fresh = runner._discover_live_cgroup_v2(
                    proc_cgroup_path=proc_cgroup,
                    cgroup_root=mount,
                    slurm_job_id="136309",
                    slurm_step_id="1",
                )
                self.assertNotEqual(fresh["binding_digest"], opened["binding_digest"])
            finally:
                for descriptor in descriptors.values():
                    os.close(descriptor)

    def test_cgroup_binding_rejects_fully_resigned_ancestor_forgery(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            binding, _, _, _ = _write_auh_cgroup_hierarchy_fixture(
                Path(directory).resolve()
            )
            for case in (
                "drop-chain-row",
                "selected-leaf",
                "job-pretends-60",
                "measurement-path",
            ):
                with self.subTest(case=case):
                    hostile = json.loads(json.dumps(binding))
                    if case == "drop-chain-row":
                        del hostile["ancestor_chain_leaf_to_root"][1]
                    elif case == "selected-leaf":
                        leaf = hostile["ancestor_chain_leaf_to_root"][0]
                        hostile["sampled_limit_cgroup"].update(
                            {
                                "relative_path": leaf["relative_path"],
                                "path": leaf["path"],
                                "directory_identity": leaf["directory_identity"],
                                "distance_from_leaf": 0,
                                "memory_max_bytes": runner.HOST_MEMORY_LIMIT_BYTES,
                            }
                        )
                    elif case == "job-pretends-60":
                        job = next(
                            row
                            for row in hostile["ancestor_chain_leaf_to_root"]
                            if row["relative_path"].endswith("/job_136309")
                        )
                        job["memory_max_raw"] = str(runner.HOST_MEMORY_LIMIT_BYTES)
                        job["memory_max_bytes"] = runner.HOST_MEMORY_LIMIT_BYTES
                    else:
                        hostile["measurement_files"]["memory_current"]["path"] += ".evil"
                    hostile["ancestor_chain_digest"] = runner.object_sha256(
                        hostile["ancestor_chain_leaf_to_root"]
                    )
                    unsigned = dict(hostile)
                    unsigned.pop("binding_digest", None)
                    hostile["binding_digest"] = runner.object_sha256(unsigned)
                    with self.assertRaises(runner.Reserve4GenerationError):
                        runner._validate_cgroup_binding(hostile)

    def test_cgroup_sampler_uses_selected_ancestor_events_not_clean_leaf_events(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            binding, _, _, _ = _write_auh_cgroup_hierarchy_fixture(
                Path(directory).resolve(), selected_oom=1
            )
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError, "recorded OOM activity"
            ):
                runner._sample_live_cgroup_memory(binding, sequence=0)

    def test_host_cgroup_sampled_gate_rejects_old_missing_resigned_and_unsafe(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, compile_path, valid, _, _ = _write_host_memory_gate_fixture(root)
            original_compile = compile_path.read_bytes()
            self.assertEqual(runner.validate_host_cgroup_memory_gate(valid), valid)
            runner.load_host_cgroup_memory_gate(
                compile_path,
                _sha(compile_path),
                expected_phase="compile_smoke_before_formal40",
                require_monitor_alive_now=False,
            )
            for case in (
                "old-schema", "missing", "resigned-peak", "wrong-max",
                "at-safe-ceiling", "oom", "oom-kill", "extra", "bad-digest",
            ):
                with self.subTest(case=case):
                    hostile = json.loads(json.dumps(valid))
                    if case == "old-schema":
                        hostile["schema_version"] = "old-host-gate"
                    elif case == "missing":
                        del hostile["sampling"]
                    elif case == "resigned-peak":
                        hostile["sampled_peak_memory_current_bytes"] += 1
                    elif case == "wrong-max":
                        hostile["cgroup_memory_max_bytes"] = 59 * 1024**3
                    elif case == "at-safe-ceiling":
                        hostile["sampled_peak_memory_current_bytes"] = (
                            runner.HOST_MEMORY_SAFE_CEILING_BYTES
                        )
                    elif case == "oom":
                        hostile["memory_events_at_gate"]["oom"] = 1
                    elif case == "oom-kill":
                        hostile["memory_events_at_gate"]["oom_kill"] = 1
                    elif case == "extra":
                        hostile["unsealed"] = True
                    else:
                        hostile["receipt_digest"] = "0" * 64
                    if case != "bad-digest":
                        hostile = _resign(hostile)
                    if case == "resigned-peak":
                        compile_path.chmod(0o600)
                        compile_path.write_bytes(
                            runner.canonical_json_bytes(hostile) + b"\n"
                        )
                        with self.assertRaisesRegex(
                            runner.Reserve4GenerationError,
                            "resigned/replayed fields differ",
                        ):
                            runner.load_host_cgroup_memory_gate(
                                compile_path,
                                _sha(compile_path),
                                expected_phase="compile_smoke_before_formal40",
                                require_monitor_alive_now=False,
                            )
                        compile_path.write_bytes(original_compile)
                    else:
                        with self.assertRaises(runner.Reserve4GenerationError):
                            runner.validate_host_cgroup_memory_gate(hostile)

    def test_host_memory_cadence_uses_monotonic_clock_not_wall_clock(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            _, compile_path, compile_gate, terminal_path, terminal_gate = (
                _write_host_memory_gate_fixture(
                    Path(directory).resolve(),
                    wall_time_offsets_ns=(0, 10_000_000, -50_000_000, 20_000_000),
                )
            )
            for path, gate, phase in (
                (
                    compile_path,
                    compile_gate,
                    "compile_smoke_before_formal40",
                ),
                (
                    terminal_path,
                    terminal_gate,
                    "terminal_after_formal40_before_slurm_child_exit",
                ),
            ):
                replayed, _, _ = runner.load_host_cgroup_memory_gate(
                    path,
                    _sha(path),
                    expected_phase=phase,
                    require_monitor_alive_now=False,
                )
                self.assertTrue(
                    replayed["sampling"][
                        "monotonic_timestamps_strictly_increasing"
                    ]
                )
                self.assertTrue(
                    replayed["sampling"]["wall_clock_timestamps_informational"]
                )

    def test_host_gate_requires_explicit_live_tail_and_terminal_stop_marker(
        self,
    ) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            start, _, compile_gate, _, _ = _write_host_memory_gate_fixture(root)
            self.assertEqual(compile_gate["observed_tail_age_ns"], 1_000_000)
            start_path = (
                root / "host-cgroup-memory-monitor-start-receipt.json"
            )
            start_sha = _sha(start_path)
            journal = Path(start["sample_journal"]["path"])
            raw = journal.read_bytes()
            rows = [
                list(runner.HOST_MEMORY_SAMPLE_STRUCT.unpack_from(raw, offset))
                for offset in range(
                    0, len(raw), runner.HOST_MEMORY_SAMPLE_STRUCT.size
                )
            ]
            compile_raw = raw[: 2 * runner.HOST_MEMORY_SAMPLE_STRUCT.size]
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "requires an explicit live-tail observation",
            ):
                runner._derive_host_cgroup_memory_gate(
                    start_receipt=start,
                    start_receipt_path=start_path,
                    start_receipt_sha256=start_sha,
                    raw_prefix=compile_raw,
                    rows=rows[:2],
                    measurement_phase="compile_smoke_before_formal40",
                    formal_candidate_count_at_gate=0,
                )

            early_stop = [list(row) for row in rows[:2]]
            early_stop[-1][7] = 1
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "stop/final sample marker differs",
            ):
                runner._derive_host_cgroup_memory_gate(
                    start_receipt=start,
                    start_receipt_path=start_path,
                    start_receipt_sha256=start_sha,
                    raw_prefix=b"".join(
                        runner.HOST_MEMORY_SAMPLE_STRUCT.pack(*row)
                        for row in early_stop
                    ),
                    rows=early_stop,
                    measurement_phase="compile_smoke_before_formal40",
                    formal_candidate_count_at_gate=0,
                    live_tail_observed_monotonic_time_ns=(
                        early_stop[-1][2] + 1_000_000
                    ),
                )

            missing_terminal_stop = [list(row) for row in rows]
            missing_terminal_stop[-1][7] = 0
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "stop/final sample marker differs",
            ):
                runner._derive_host_cgroup_memory_gate(
                    start_receipt=start,
                    start_receipt_path=start_path,
                    start_receipt_sha256=start_sha,
                    raw_prefix=b"".join(
                        runner.HOST_MEMORY_SAMPLE_STRUCT.pack(*row)
                        for row in missing_terminal_stop
                    ),
                    rows=missing_terminal_stop,
                    measurement_phase=(
                        "terminal_after_formal40_before_slurm_child_exit"
                    ),
                    formal_candidate_count_at_gate=40,
                )

    def test_terminal_gate_requires_bound_supervisor_wait0_and_dead_monitor(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            start, _, _, _, _ = _write_host_memory_gate_fixture(root)
            environment = {
                "GADP_HOST_MEMORY_SAMPLE_JOURNAL": start["sample_journal"]["path"],
                "GADP_HOST_MEMORY_MONITOR_START_RECEIPT": str(
                    root / "host-cgroup-memory-monitor-start-receipt.json"
                ),
                "GADP_HOST_MEMORY_MONITOR_PID": str(start["monitor_pid"]),
                "GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID": str(
                    start["supervisor_pid"]
                ),
            }

            def live_identity(pid: int, _: int) -> bool:
                return pid == start["supervisor_pid"]

            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                runner.os, "getppid", return_value=start["supervisor_pid"]
            ), mock.patch.object(
                runner, "_process_identity_is_live", side_effect=live_identity
            ):
                with self.assertRaisesRegex(
                    runner.Reserve4GenerationError, "wait0 and dead monitor"
                ):
                    runner.write_host_cgroup_memory_gate_receipt(
                        root / "wrong-exit.json",
                        measurement_phase=(
                            "terminal_after_formal40_before_slurm_child_exit"
                        ),
                        formal_candidate_count_at_gate=40,
                        monitor_exit_status=1,
                    )
                output = root / "supervisor-terminal-gate.json"
                value, observed = runner.write_host_cgroup_memory_gate_receipt(
                    output,
                    measurement_phase=(
                        "terminal_after_formal40_before_slurm_child_exit"
                    ),
                    formal_candidate_count_at_gate=40,
                    monitor_exit_status=0,
                )
                self.assertEqual(observed, _sha(output))
                self.assertEqual(value["monitor_exit_status"], 0)
                self.assertTrue(value["monitor_identity_dead_at_gate"])
                self.assertTrue(
                    value["terminal_gate_created_after_bound_supervisor_wait"]
                )
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                runner.os, "getppid", return_value=start["supervisor_pid"]
            ), mock.patch.object(
                runner, "_process_identity_is_live", return_value=True
            ):
                with self.assertRaisesRegex(
                    runner.Reserve4GenerationError, "wait0 and dead monitor"
                ):
                    runner.write_host_cgroup_memory_gate_receipt(
                        root / "live-monitor.json",
                        measurement_phase=(
                            "terminal_after_formal40_before_slurm_child_exit"
                        ),
                        formal_candidate_count_at_gate=40,
                        monitor_exit_status=0,
                    )

    def test_terminal_host_cgroup_receipt_rejects_resigned_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory).resolve()
            logs = run_root / "logs"
            _, _, _, path, valid = _write_host_memory_gate_fixture(logs)
            _, _, replayed = (
                controller.validate_terminal_host_cgroup_memory_receipt(run_root)
            )
            self.assertEqual(replayed, valid)
            original = path.read_bytes()
            for case in ("old-schema", "resigned-peak", "oom-kill", "journal-tamper"):
                with self.subTest(case=case):
                    path.write_bytes(original)
                    hostile = json.loads(json.dumps(valid))
                    if case == "old-schema":
                        hostile["schema_version"] = "old-host-gate"
                    elif case == "resigned-peak":
                        hostile["sampled_peak_memory_current_bytes"] -= 1
                    elif case == "oom-kill":
                        hostile["memory_events_at_gate"]["oom_kill"] = 1
                    else:
                        hostile["sample_journal"]["prefix_sha256"] = "f" * 64
                    hostile = _resign(hostile)
                    path.write_bytes(
                        controller.canonical_json_bytes(hostile) + b"\n"
                    )
                    with self.assertRaises(controller.GenericActionDataPrepError):
                        controller.validate_terminal_host_cgroup_memory_receipt(
                            run_root
                        )

    def test_host_monitor_rejects_old_start_cadence_and_journal_regressions(
        self,
    ) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            start, compile_path, _, terminal_path, _ = (
                _write_host_memory_gate_fixture(root)
            )
            start_path = root / "host-cgroup-memory-monitor-start-receipt.json"
            journal_path = Path(start["sample_journal"]["path"])
            original_start = start_path.read_bytes()
            old_start = dict(start)
            old_start["schema_version"] = (
                "bernini-generic-action-fit40-host-cgroup-memory-monitor-start-v0"
            )
            old_start = _resign(old_start)
            start_path.chmod(0o600)
            start_path.write_bytes(runner.canonical_json_bytes(old_start) + b"\n")
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError, "monitor start schema/digest differs"
            ):
                runner.load_host_cgroup_memory_monitor_start(
                    start_path, _sha(start_path)
                )
            start_path.write_bytes(original_start)

            raw = journal_path.read_bytes()
            rows = [
                list(runner.HOST_MEMORY_SAMPLE_STRUCT.unpack_from(raw, offset))
                for offset in range(
                    0, len(raw), runner.HOST_MEMORY_SAMPLE_STRUCT.size
                )
            ]
            rows[1][1] = rows[0][1] + runner.HOST_MEMORY_MAX_SAMPLE_GAP_NS + 1
            rows[1][2] = rows[0][2] + runner.HOST_MEMORY_MAX_SAMPLE_GAP_NS + 1
            bad_prefix = b"".join(
                runner.HOST_MEMORY_SAMPLE_STRUCT.pack(*row) for row in rows[:2]
            )
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError, "cadence gap exceeded"
            ):
                runner._derive_host_cgroup_memory_gate(
                    start_receipt=start,
                    start_receipt_path=start_path,
                    start_receipt_sha256=_sha(start_path),
                    raw_prefix=bad_prefix,
                    rows=rows[:2],
                    measurement_phase="compile_smoke_before_formal40",
                    formal_candidate_count_at_gate=0,
                    live_tail_observed_monotonic_time_ns=(
                        rows[1][2] + 1_000_000
                    ),
                )

            journal_path.chmod(0o600)
            journal_path.write_bytes(raw[:-1])
            with self.assertRaises(runner.Reserve4GenerationError):
                runner.load_host_cgroup_memory_gate(
                    terminal_path,
                    _sha(terminal_path),
                    expected_phase=(
                        "terminal_after_formal40_before_slurm_child_exit"
                    ),
                    require_monitor_alive_now=False,
                )
            journal_path.write_bytes(raw)
            journal_path.chmod(0o400)
            mutated = bytearray(raw)
            mutated[runner.HOST_MEMORY_SAMPLE_STRUCT.size + 8] ^= 1
            journal_path.chmod(0o600)
            journal_path.write_bytes(bytes(mutated))
            journal_path.chmod(0o400)
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError, "journal prefix replay differs"
            ):
                runner.load_host_cgroup_memory_gate(
                    compile_path,
                    _sha(compile_path),
                    expected_phase="compile_smoke_before_formal40",
                    require_monitor_alive_now=False,
                )

    def test_live_monitor_rejects_alive_but_hung_stale_tail(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            start, _, _, _, _ = _write_host_memory_gate_fixture(root)
            journal = Path(start["sample_journal"]["path"])
            raw = journal.read_bytes()
            journal.chmod(0o600)
            journal.write_bytes(raw[: 2 * runner.HOST_MEMORY_SAMPLE_STRUCT.size])
            environment = {
                "GADP_HOST_MEMORY_SAMPLE_JOURNAL": str(journal),
                "GADP_HOST_MEMORY_MONITOR_START_RECEIPT": str(
                    root / "host-cgroup-memory-monitor-start-receipt.json"
                ),
                "GADP_HOST_MEMORY_MONITOR_PID": str(start["monitor_pid"]),
                "GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID": str(
                    start["supervisor_pid"]
                ),
            }
            live_cgroup = start["cgroup_binding"]
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                runner, "_process_identity_is_live", return_value=True
            ), mock.patch.object(
                runner, "_discover_live_cgroup_v2", return_value=live_cgroup
            ):
                with self.assertRaisesRegex(
                    runner.Reserve4GenerationError, "live-tail age exceeded"
                ):
                    runner.assert_live_host_cgroup_memory_monitor()

    def _release_fixture(self, root: Path) -> tuple[Path, Path, Path, dict]:
        method_root = GenericActionDataPrepReleaseTests._dummy_root(root)
        archive, manifest = root / "source.tar", root / "source.json"
        receipt = release.build(method_root, archive, manifest)
        return method_root, archive, manifest, dict(receipt)

    def _runtime_args(
        self, method_root: Path, archive: Path, manifest: Path, receipt: dict
    ) -> Namespace:
        python = Path(sys.executable).resolve(strict=True)
        return Namespace(
            method_root=str(method_root),
            method_archive=str(archive),
            expected_method_archive_sha256=receipt["archive_sha256"],
            method_manifest=str(manifest),
            expected_method_manifest_sha256=receipt["manifest_sha256"],
            python_bin=str(python),
            expected_python_sha256=_sha(python),
        )

    @staticmethod
    def _valid_fit_upstream() -> dict:
        branches = list(controller.reserve4_runner.bank_contract.MACE_BRANCH_ORDER)
        tasks = []
        cell_proofs = []
        for seed_slot in ("seed1", "seed2"):
            for group_id, visible_gpus in (
                ("sp4-a", [0, 1, 2, 3]),
                ("sp4-b", [4, 5, 6, 7]),
            ):
                candidate_ids = []
                for branch in branches:
                    candidate_id = f"{seed_slot}-{group_id}-fit-{branch}"
                    candidate_ids.append(candidate_id)
                    tasks.append(
                        {
                            "candidate_id": candidate_id,
                            "seed_slot": seed_slot,
                            "group_id": group_id,
                            "visible_gpus": visible_gpus,
                            "analysis_split": "fit",
                            "semantic_branch": branch,
                        }
                    )
                cell_proofs.append(
                    {
                        "seed_slot": seed_slot,
                        "group_id": group_id,
                        "analysis_split": "fit",
                        "branch_order": branches,
                        "candidate_ids": candidate_ids,
                    }
                )
        shards = []
        for seed_slot in ("seed1", "seed2"):
            for group_id, visible_gpus in (
                ("sp4-a", [0, 1, 2, 3]),
                ("sp4-b", [4, 5, 6, 7]),
            ):
                candidate_ids = [
                    row["candidate_id"]
                    for row in tasks
                    if row["seed_slot"] == seed_slot
                    and row["group_id"] == group_id
                ]
                shards.append(
                    {
                        "shard_id": f"{seed_slot}-{group_id}-fit",
                        "seed_slot": seed_slot,
                        "group_id": group_id,
                        "visible_gpus": visible_gpus,
                        "candidate_ids": candidate_ids,
                        "candidate_count": 10,
                    }
                )
        return {
            "schema_version": "bernini-reserve4-fixed-generation-sp4-plan-v1",
            "analysis_split": "fit",
            "generation_invocation_count": 40,
            "seed_cell_count": 4,
            "tasks": tasks,
            "cell_proofs": cell_proofs,
            "shards": shards,
            "execution_contract": {"optimizer_authorized": False},
            "plan_digest": "a" * 64,
        }

    def _plan_fixture(self, root: Path) -> tuple[dict, Path, str, dict]:
        method_root, archive, manifest, receipt = self._release_fixture(root)
        closure = controller.validate_release_tree(
            self._runtime_args(method_root, archive, manifest, receipt)
        )
        run_root = root / "run"
        run_root.mkdir()
        upstream_path = run_root / "upstream.json"
        upstream = self._valid_fit_upstream()
        upstream_path.write_bytes(controller.canonical_json_bytes(upstream) + b"\n")
        args = Namespace(master_port=29571)
        plan, path, digest = controller.build_controller_plan(
            args, closure, run_root, upstream_path, _sha(upstream_path), upstream
        )
        return dict(plan), path, digest, upstream

    def _generation_closure_fixture(
        self, root: Path
    ) -> tuple[dict, Path, str, list[Path], Path]:
        plan = self._valid_fit_upstream()
        plan_path = root / "upstream.json"
        plan_path.write_bytes(controller.canonical_json_bytes(plan) + b"\n")
        plan_sha = _sha(plan_path)
        generation = root / "generation"
        generation.mkdir()
        roots: list[Path] = []
        all_candidate_rows = []
        all_gaussian_proofs = []
        for shard in plan["shards"]:
            shard_root = generation / f"{shard['seed_slot']}-{shard['group_id']}"
            shard_root.mkdir()
            roots.append(shard_root)
            matching = [
                task
                for task in plan["tasks"]
                if task["seed_slot"] == shard["seed_slot"]
                and task["group_id"] == shard["group_id"]
            ]
            shard_rows = []
            for task in matching:
                candidate_root = shard_root / task["candidate_id"]
                candidate_root.mkdir()
                artifact_path = candidate_root / "t2v.mp4"
                artifact_path.write_bytes(task["candidate_id"].encode("ascii"))
                native_unsigned = {
                    "schema_version": "test-native-receipt-v1",
                    "outputs": {
                        "t2v": {
                            "path": str(artifact_path),
                            "sha256": _sha(artifact_path),
                        }
                    },
                }
                native = _resign(native_unsigned)
                native_path = candidate_root / "receipt.json"
                native_path.write_bytes(
                    controller.canonical_json_bytes(native) + b"\n"
                )
                pair_unsigned = {
                    "schema_version": "test-pair-receipt-v1",
                    "candidate_id": task["candidate_id"],
                    "native_receipt_path": str(native_path),
                    "native_receipt_sha256": _sha(native_path),
                    "native_receipt_digest": native["receipt_digest"],
                    "artifacts": {
                        "mp4": {
                            "path": str(artifact_path),
                            "sha256": _sha(artifact_path),
                        }
                    },
                }
                pair = _resign(pair_unsigned)
                pair_path = candidate_root / "pair-v5-t2v-calibration-receipt.json"
                pair_path.write_bytes(
                    controller.canonical_json_bytes(pair) + b"\n"
                )
                row = {
                    "candidate_id": task["candidate_id"],
                    "path": str(pair_path),
                    "file_sha256": _sha(pair_path),
                    "receipt_digest": pair["receipt_digest"],
                }
                shard_rows.append(row)
                all_candidate_rows.append(row)
            shard_gaussian = [
                {
                    "seed_slot": shard["seed_slot"],
                    "group_id": shard["group_id"],
                    "candidate_count": 10,
                }
            ]
            all_gaussian_proofs.extend(shard_gaussian)
            shard_unsigned = {
                "schema_version": "bernini-reserve4-fixed-generation-shard-receipt-v1",
                "plan_path": str(plan_path),
                "plan_file_sha256": plan_sha,
                "plan_digest": plan["plan_digest"],
                "analysis_split": "fit",
                "seed_slot": shard["seed_slot"],
                "group_id": shard["group_id"],
                "visible_gpus": shard["visible_gpus"],
                "candidate_count": 10,
                "candidate_receipts": shard_rows,
                "same_cell_gaussian_proofs": shard_gaussian,
                "independent_full81_review_performed": False,
                "phi_v1_extraction_authorized": False,
                "training_performed": False,
                "optimizer_created": False,
                "optimizer_authorized": False,
                "generated_media_is_editor_input_or_target": False,
            }
            shard_receipt = _resign(shard_unsigned)
            (shard_root / "reserve4-generation-shard-receipt-v1.json").write_bytes(
                controller.canonical_json_bytes(shard_receipt) + b"\n"
            )
        audit_unsigned = {
            "schema_version": "bernini-reserve4-fixed-generation-audit-receipt-v1",
            "plan_path": str(plan_path),
            "plan_file_sha256": plan_sha,
            "plan_digest": plan["plan_digest"],
            "analysis_split": "fit",
            "candidate_count": 40,
            "seed_cell_count": 4,
            "candidate_receipts": all_candidate_rows,
            "same_cell_gaussian_proofs": all_gaussian_proofs,
            "generation_complete": True,
            "independent_full81_review_performed": False,
            "visual_review_required_before_phi_v1_extraction": True,
            "phi_v1_extraction_authorized": False,
            "training_performed": False,
            "optimizer_created": False,
            "optimizer_authorized": False,
            "generated_media_is_editor_input_or_target": False,
        }
        audit = _resign(audit_unsigned)
        audit_path = root / "generation-audit.json"
        audit_path.write_bytes(controller.canonical_json_bytes(audit) + b"\n")
        return plan, plan_path, plan_sha, roots, audit_path

    @staticmethod
    def _fixture_gaussian_proofs(
        validated: list[tuple[dict, dict]],
    ) -> list[dict]:
        proofs = []
        seen = set()
        for task, _ in validated:
            key = (task["seed_slot"], task["group_id"])
            if key not in seen:
                seen.add(key)
                proofs.append(
                    {
                        "seed_slot": key[0],
                        "group_id": key[1],
                        "candidate_count": 10,
                    }
                )
        return proofs

    def _validate_generation_fixture(
        self,
        plan: dict,
        plan_path: Path,
        plan_sha: str,
        roots: list[Path],
        audit_path: Path,
    ) -> dict:
        def validate_candidate(task: dict, receipt_path: Path) -> tuple[dict, dict]:
            value = json.loads(receipt_path.read_text(encoding="ascii"))
            return value, {"candidate_id": task["candidate_id"]}

        with mock.patch.object(
            controller,
            "validate_upstream_fit_plan",
            return_value=plan,
        ), mock.patch.object(
            controller.reserve4_runner,
            "_validate_candidate_receipt",
            side_effect=validate_candidate,
        ), mock.patch.object(
            controller.reserve4_runner,
            "_gaussian_cell_proofs",
            side_effect=self._fixture_gaussian_proofs,
        ):
            return dict(
                controller.validate_generation_output_closure(
                    plan_path=plan_path,
                    expected_plan_sha256=plan_sha,
                    generation_roots=roots,
                    generation_audit_path=audit_path,
                )
            )

    def test_runtime_tree_is_bound_to_every_archive_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            method_root, archive, manifest, receipt = self._release_fixture(root)
            args = self._runtime_args(method_root, archive, manifest, receipt)
            closure = controller.validate_release_tree(args)
            self.assertEqual(closure["component_pins"], receipt["component_pins"])
            extra = method_root / "unsealed.py"
            extra.write_text("unsealed\n", encoding="ascii")
            with self.assertRaisesRegex(
                controller.GenericActionDataPrepError, "exact release member closure"
            ):
                controller.validate_release_tree(args)

    def test_controller_plan_is_fixed_fit40_all8_serial_four_world4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan, path, digest, upstream = self._plan_fixture(Path(directory).resolve())
            with mock.patch.object(
                controller, "validate_upstream_fit_plan", return_value=upstream
            ):
                self.assertEqual(controller.validate_controller_plan(path, digest), plan)
            self.assertEqual(plan["phase"], "generation")
            self.assertEqual(plan["analysis_split"], "fit")
            self.assertEqual(plan["topology"]["slurm_child_gpu_count"], 8)
            self.assertEqual(plan["topology"]["numbered_slurm_children"], 1)
            self.assertEqual(plan["topology"]["run_sp4_shard_process_count"], 4)
            self.assertEqual(plan["topology"]["world4_model_invocation_count"], 40)
            self.assertEqual(plan["topology"]["sealed_shard_order"], controller.SHARD_ORDER)
            self.assertEqual(
                plan["topology"]["physical_island_order"],
                controller.PHYSICAL_ISLAND_ORDER,
            )
            self.assertTrue(plan["topology"]["all_model_invocations_strictly_serial"])
            self.assertTrue(plan["topology"]["serialized_world4_host_checkpoint_load"])
            self.assertTrue(
                plan["topology"][
                    "model_load_lock_held_through_gpu_move_and_malloc_trim"
                ]
            )
            self.assertTrue(
                plan["topology"]["t2v_vae_load_deferred_until_rank0_post_sampling"]
            )
            self.assertTrue(
                plan["topology"][
                    "world4_renderer_retirement_barrier_before_rank_zero_vae_load"
                ]
            )
            self.assertTrue(
                plan["topology"][
                    "host_cgroup_sample_monitor_started_before_compile_smoke"
                ]
            )
            self.assertEqual(
                plan["topology"]["host_cgroup_sample_interval_ns"],
                controller.HOST_MEMORY_SAMPLE_INTERVAL_NS,
            )
            self.assertEqual(
                plan["topology"]["host_cgroup_max_sample_gap_ns"],
                controller.HOST_MEMORY_MAX_SAMPLE_GAP_NS,
            )
            self.assertEqual(
                plan["topology"]["host_sampled_current_safe_ceiling_gib"],
                controller.HOST_MEMORY_SAFE_CEILING_GIB,
            )
            self.assertEqual(plan["topology"]["host_memory_request_gib"], 60)
            self.assertEqual(plan["topology"]["concurrent_model_replicas"], 1)
            self.assertFalse(plan["topology"]["rank_or_gpu_action_family_partition"])
            self.assertEqual(plan["expected_outputs"], controller.OUTPUT_CONTRACT)
            self.assertFalse(
                plan["non_authoritative_dynamic_probe_observation"][
                    "probe_receipt_pinned"
                ]
            )
            self.assertFalse(
                plan["non_authoritative_dynamic_probe_observation"][
                    "is_release_authority"
                ]
            )
            self.assertFalse(plan["authority"]["confirmation_generation_authorized"])
            self.assertFalse(plan["authority"]["phi_v1_extraction_authorized"])
            self.assertFalse(plan["authority"]["optimizer_authorized"])
            old_plan = dict(plan)
            old_plan["schema_version"] = (
                "bernini-generic-action-fit40-generation-136309-plan-v10"
            )
            old_plan = _resign(old_plan, field="plan_digest")
            path.chmod(0o600)
            path.write_bytes(controller.canonical_json_bytes(old_plan) + b"\n")
            with mock.patch.object(
                controller, "validate_upstream_fit_plan", return_value=upstream
            ):
                with self.assertRaisesRegex(
                    controller.GenericActionDataPrepError,
                    "controller plan contract differs",
                ):
                    controller.validate_controller_plan(path, _sha(path))

    def test_launcher_environment_is_exactly_cross_bound_to_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan, path, digest, upstream = self._plan_fixture(Path(directory).resolve())
            closure = plan["release"]
            pins = closure["component_pins"]
            environment = {
                "GADP_CONFIRM": controller.LAUNCH_CONFIRMATION,
                "GADP_PHASE": "generation",
                "GADP_SPLIT": "fit",
                "GADP_RUN_ROOT": plan["run_root"],
                "GADP_MASTER_PORT": str(plan["master_port"]),
                "GADP_CONTROLLER_PLAN": str(path),
                "GADP_CONTROLLER_PLAN_SHA256": digest,
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
            }
            with mock.patch.object(
                controller, "validate_upstream_fit_plan", return_value=upstream
            ), mock.patch.dict("os.environ", environment, clear=True):
                self.assertEqual(controller.validate_launch_environment(path, digest), plan)
            environment["GADP_METHOD_REVISION"] = "0" * 40
            with mock.patch.object(
                controller, "validate_upstream_fit_plan", return_value=upstream
            ), mock.patch.dict("os.environ", environment, clear=True):
                with self.assertRaisesRegex(
                    controller.GenericActionDataPrepError,
                    "not cross-bound",
                ):
                    controller.validate_launch_environment(path, digest)

    def test_upstream_semantic_replay_rejects_split_mapping_or_order_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "upstream.json"
            valid = self._valid_fit_upstream()
            path.write_bytes(controller.canonical_json_bytes(valid) + b"\n")
            observed = _sha(path)
            with mock.patch.object(
                controller.reserve4_runner,
                "load_plan",
                return_value=(valid, path, observed),
            ):
                self.assertEqual(
                    controller.validate_upstream_fit_plan(path, observed), valid
                )
            for case in (
                "confirmation",
                "wrong-physical-island",
                "wrong-group-count",
                "wrong-shard-order",
                "wrong-task-order",
            ):
                tampered = self._valid_fit_upstream()
                if case == "confirmation":
                    tampered["analysis_split"] = "confirmation"
                elif case == "wrong-physical-island":
                    tampered["tasks"][10]["visible_gpus"] = [0, 1, 2, 3]
                elif case == "wrong-group-count":
                    tampered["tasks"][10]["group_id"] = "sp4-a"
                elif case == "wrong-shard-order":
                    tampered["shards"][0], tampered["shards"][1] = (
                        tampered["shards"][1],
                        tampered["shards"][0],
                    )
                else:
                    tampered["tasks"][0], tampered["tasks"][1] = (
                        tampered["tasks"][1],
                        tampered["tasks"][0],
                    )
                with self.subTest(case=case), mock.patch.object(
                    controller.reserve4_runner,
                    "load_plan",
                    return_value=(tampered, path, observed),
                ):
                    with self.assertRaises(controller.GenericActionDataPrepError):
                        controller.validate_upstream_fit_plan(path, observed)

    def test_controller_cli_has_no_confirmation_phi_or_po_selector(self) -> None:
        parser = controller.build_parser()
        option_strings = set()
        command_names = set()
        for action in parser._actions:
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                command_names.update(choices)
                for child in choices.values():
                    for child_action in child._actions:
                        option_strings.update(child_action.option_strings)
        self.assertEqual(
            command_names,
            {
                "plan",
                "launch",
                "validate-plan",
                "validate-launch-environment",
                "validate-runtime",
                "observe-gpu-mapping",
                "validate-gpu-mapping",
                "seal-physical-gpu-inventory",
                "seal-gpu-binding-receipt",
                "seal-gpu-admission-receipt",
                "validate-terminal-host-cgroup-memory-receipt",
                "seal-generation-closure",
            },
        )
        self.assertTrue(
            {
                "--phase",
                "--split",
                "--review-root",
                "--representation-manifest",
                "--source-pair-manifest",
            }.isdisjoint(option_strings)
        )

    @staticmethod
    def _write_physical_inventory(root: Path) -> tuple[Path, dict, str]:
        root.mkdir(parents=True, exist_ok=True)
        identity_path = root / "gpu-identity-bus.raw.txt"
        topology_path = root / "gpu-topology.raw.txt"
        identity_lines = []
        for index in range(8):
            identity_lines.extend(
                [
                    f"GPU[{index}] : Unique ID: 0x{index + 1:016X}",
                    f"GPU[{index}] : PCI Bus: {0x40 + index:02X}:00.0",
                ]
            )
        identity_path.write_text("\n".join(identity_lines) + "\n", encoding="ascii")
        topology_lines = [
            "        GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7"
        ]
        for index in range(8):
            island = range(4) if index < 4 else range(4, 8)
            cells = [
                "0" if peer == index else "XGMI" if peer in island else "PCIE"
                for peer in range(8)
            ]
            topology_lines.append(f"GPU{index} " + " ".join(cells))
        topology_lines.extend(
            f"GPU[{index}] : (Topology) Numa Node: {0 if index < 4 else 1}"
            for index in range(8)
        )
        topology_path.write_text("\n".join(topology_lines) + "\n", encoding="ascii")
        output = root / "gpu-physical-inventory.json"
        value = controller.seal_rocm_physical_inventory(
            identity_path.resolve(), topology_path.resolve(), output.resolve()
        )
        return output, dict(value), _sha(output)

    @staticmethod
    def _write_live_v6_physical_inventory(root: Path) -> tuple[Path, dict, str]:
        root.mkdir(parents=True, exist_ok=True)
        identity_path = root / "gpu-identity-bus.raw.txt"
        topology_path = root / "gpu-topology.raw.txt"
        identity_lines = []
        for index in range(8):
            bus, unique_id, _ = LIVE_V6_GPU_IDENTITIES[index]
            identity_lines.extend(
                [
                    f"GPU[{index}] : Unique ID: 0x{unique_id}",
                    f"GPU[{index}] : PCI Bus: {bus}",
                ]
            )
        identity_path.write_text("\n".join(identity_lines) + "\n", encoding="ascii")
        topology_lines = ["        GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7"]
        for index in range(8):
            island = range(4) if index < 4 else range(4, 8)
            cells = [
                "0" if peer == index else "XGMI" if peer in island else "PCIE"
                for peer in range(8)
            ]
            topology_lines.append(f"GPU{index} " + " ".join(cells))
        topology_lines.extend(
            f"GPU[{index}] : (Topology) Numa Node: {0 if index < 4 else 1}"
            for index in range(8)
        )
        topology_path.write_text("\n".join(topology_lines) + "\n", encoding="ascii")
        output = root / "gpu-physical-inventory.json"
        value = controller.seal_rocm_physical_inventory(
            identity_path.resolve(), topology_path.resolve(), output.resolve()
        )
        return output, dict(value), _sha(output)

    @staticmethod
    def _write_runtime_mapping(
        path: Path,
        physical_indices: list[int],
        rocr: str | None,
        inventory_path: Path,
        inventory: dict,
        inventory_sha: str,
        device_identities: dict[int, tuple[str, str, str]] | None = None,
    ) -> tuple[dict, str]:
        devices = []
        for logical_index, physical_index in enumerate(physical_indices):
            if device_identities is None:
                pci_bus_id = f"0000:{0x40 + physical_index:02x}:00.0"
                unique_id = f"{physical_index + 1:016x}"
                hip_uuid = f"{physical_index + 1:032x}"
            else:
                pci_bus_id, unique_id, hip_uuid = device_identities[physical_index]
            devices.append(
                {
                    "logical_index": logical_index,
                    "hip_device_handle": logical_index,
                    "hip_uuid_hex": hip_uuid,
                    "pci_bus_id": pci_bus_id,
                    "physical_index": physical_index,
                    "physical_rocm_unique_id": unique_id,
                    "torch_device_name": "AMD Instinct MI210",
                    "torch_property_uuid": None,
                }
            )
        unsigned = {
            "schema_version": controller.RUNTIME_MAPPING_SCHEMA,
            "rocr_visible_devices": rocr,
            "torch_version": "test-rocm-torch",
            "torch_hip_version": "test-hip",
            "torch_device_count": len(physical_indices),
            "hip_runtime_device_count": len(physical_indices),
            "physical_inventory": {
                "path": str(inventory_path),
                "file_sha256": inventory_sha,
                "inventory_digest": inventory["inventory_digest"],
            },
            "pci_bus_is_authoritative_join_key": True,
            "physical_index_derived_from_pci_bus_join": True,
            "physical_rocm_unique_id_replayed": True,
            "hip_logical_order_is_observation_only": True,
            "devices": devices,
        }
        value = {**unsigned, "observation_digest": controller.object_sha256(unsigned)}
        path.write_bytes(controller.canonical_json_bytes(value) + b"\n")
        return value, _sha(path)

    @staticmethod
    def _write_gpu_admission(run_root: Path) -> None:
        logs = run_root / "logs"
        logs.mkdir()
        inventory_path, inventory, inventory_sha = (
            GenericActionDataPrepControllerTests._write_physical_inventory(logs)
        )
        all8_path = logs / "all8-rocm-runtime-mapping.json"
        all8_mapping, all8_sha = GenericActionDataPrepControllerTests._write_runtime_mapping(
            all8_path,
            [2, 3, 0, 1, 6, 7, 4, 5],
            None,
            inventory_path,
            inventory,
            inventory_sha,
        )
        smoke_mapping_path = logs / "compile-smoke-rocm-runtime-mapping.json"
        GenericActionDataPrepControllerTests._write_runtime_mapping(
            smoke_mapping_path,
            [2, 3, 0, 1],
            "0,1,2,3",
            inventory_path,
            inventory,
            inventory_sha,
        )
        host_start, host_compile_path, host_compile, _, _ = (
            _write_host_memory_gate_fixture(logs)
        )
        host_reference = {
            "path": str(host_compile_path),
            "file_sha256": _sha(host_compile_path),
            "receipt_digest": host_compile["receipt_digest"],
        }
        smoke_unsigned = {
            "schema_version": controller.reserve4_runner.COMPILE_SMOKE_SCHEMA,
            "plan": {
                "path": str(run_root / "generation-plan/plan.json"),
                "file_sha256": "1" * 64,
                "plan_digest": "2" * 64,
            },
            "smoke_task": {
                "candidate_id": "seed1-sp4-a-fit-smoke",
                "seed_slot": "seed1",
                "group_id": "sp4-a",
                "visible_gpus": [0, 1, 2, 3],
                "analysis_split": "fit",
                "ordinal": 0,
                "candidate_spec_path": str(run_root / "generation-plan/candidate.json"),
                "candidate_spec_sha256": "3" * 64,
                "root_spec_sha256": "4" * 64,
            },
            "runtime": {
                "method_root": str(run_root / "release/methods/bernini_action_editing"),
                "python": {"path": str(run_root / "python"), "sha256": "5" * 64},
                "bernini_root": str(run_root / "vendor/Bernini"),
                "veomni_root": str(run_root / "vendor/VeOmni"),
                "checkpoint": str(run_root / "checkpoint"),
                "checkpoint_content_manifest": {
                    "path": str(run_root / "checkpoint.sha256"),
                    "sha256": "6" * 64,
                },
                "method_source_revision": "7" * 40,
                "method_source_archive_sha256": "8" * 64,
                "generation_worker": {
                    "path": str(run_root / "release/worker.py"),
                    "sha256": "9" * 64,
                },
                "rank_cache_wrapper": {
                    "path": str(run_root / "release/rank-exec.sh"),
                    "sha256": "a" * 64,
                },
                "preprocessing_tools": dict(
                    controller.reserve4_runner.PREPROCESSING_TOOL_SHA256
                ),
                "node_local_scratch": {
                    "path": "/tmp/generic-action-fit40-test",
                    "filesystem_type": "ext2/ext3",
                },
                "serialized_host_checkpoint_load": {
                    "required": True,
                    "environment_variable": "NATIVE_V_AXIS_LOAD_LOCK",
                    "path": "/tmp/generic-action-fit40-test/renderer-load.lock",
                    "sha256": controller.reserve4_runner.EMPTY_FILE_SHA256,
                    "mode": "0400",
                    "parent_is_authenticated_node_local_scratch": True,
                    "lock_held_through_model_to_rank_gpu_and_malloc_trim": True,
                },
                "t2v_text_encoder_rank_gpu_residency": {
                    "required": True,
                    "environment_variable": "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED",
                    "official_model_sample_preserved": True,
                    "exact_positional_cpu_offload_request_suppressed_once_per_rank": True,
                    "all_other_to_requests_delegated": True,
                    "text_encoder_retired_only_with_renderer": True,
                    "gpu_memory_limit_gib": (
                        controller.reserve4_runner.T2V_GPU_MEMORY_LIMIT_GIB
                    ),
                    "gpu_memory_limit_bytes": (
                        controller.reserve4_runner.T2V_GPU_MEMORY_LIMIT_BYTES
                    ),
                    "per_rank_max_allocated_and_reserved_receipt_required": True,
                    "all_rank_peak_reserved_strictly_below_limit_required": True,
                },
                "host_cgroup_sampled_memory_monitor": {
                    "required": True,
                    "start_receipt": host_compile["monitor_start_receipt"],
                    "sample_journal": host_start["sample_journal"],
                    "monitor_pid": host_start["monitor_pid"],
                    "monitor_proc_start_ticks": host_start[
                        "monitor_proc_start_ticks"
                    ],
                    "supervisor_pid": host_start["supervisor_pid"],
                    "supervisor_proc_start_ticks": host_start[
                        "supervisor_proc_start_ticks"
                    ],
                    "slurm_job_id": "136309",
                    "slurm_step_id": "1",
                    "cgroup_binding_digest": host_start["cgroup_binding"][
                        "binding_digest"
                    ],
                    "self_task_leaf_relative_path": host_start["cgroup_binding"][
                        "self_leaf"
                    ]["relative_path"],
                    "sampled_limit_cgroup_relative_path": host_start[
                        "cgroup_binding"
                    ]["sampled_limit_cgroup"]["relative_path"],
                    "memory_max_exact_gib": 60,
                    "sampled_current_safe_ceiling_gib": 56,
                    "sample_interval_ns": 10_000_000,
                    "maximum_sample_gap_ns": 100_000_000,
                    "sampling_source": controller.reserve4_runner.HOST_MEMORY_SAMPLING_SOURCE,
                    "zero_oom_and_oom_kill_required": True,
                    "coverage": (
                        "before_compile_smoke_through_terminal_after_formal40"
                    ),
                },
            },
            "candidate_evidence": {
                "candidate_receipt_file_sha256": "b" * 64,
                "candidate_receipt_digest": "c" * 64,
                "native_receipt_file_sha256": "d" * 64,
                "native_receipt_digest": "e" * 64,
                "resource_lifecycle": _valid_resource_lifecycle(),
                "artifact_identities": (
                    controller.reserve4_runner._r10_smoke_semantic_authority_rows()
                ),
            },
            "host_cgroup_memory_gate": host_reference,
            "world_size": 4,
            "full_native_sampling_steps": 40,
            "formal_candidate_count_at_gate": 0,
            "per_rank_gpu_peak_memory_receipt_required": True,
            "gpu_peak_reserved_limit_gib": (
                controller.reserve4_runner.T2V_GPU_MEMORY_LIMIT_GIB
            ),
            "all_rank_gpu_peak_reserved_strictly_below_limit": True,
            "disposable_output_deleted": True,
            "compile_smoke_passed": True,
            "training_performed": False,
            "optimizer_authorized": False,
        }
        smoke_receipt = {
            **smoke_unsigned,
            "receipt_digest": controller.object_sha256(smoke_unsigned),
        }
        (logs / "compile-smoke-receipt.json").write_bytes(
            controller.canonical_json_bytes(smoke_receipt) + b"\n"
        )
        identities = (
            (1, "seed1", "sp4-a", "0,1,2,3", [0, 1, 2, 3]),
            (2, "seed1", "sp4-b", "4,5,6,7", [4, 5, 6, 7]),
            (3, "seed2", "sp4-a", "0,1,2,3", [0, 1, 2, 3]),
            (4, "seed2", "sp4-b", "4,5,6,7", [4, 5, 6, 7]),
        )
        for ordinal, seed_slot, group_id, visible, physical_indices in identities:
            mapping_path = logs / (
                f"{ordinal}-{seed_slot}-{group_id}-rocm-runtime-mapping.json"
            )
            mapping, mapping_sha = (
                GenericActionDataPrepControllerTests._write_runtime_mapping(
                    mapping_path,
                    ([2, 3, 0, 1] if group_id == "sp4-a" else [6, 7, 4, 5]),
                    visible,
                    inventory_path,
                    inventory,
                    inventory_sha,
                )
            )
            binding_path = logs / (
                f"{ordinal}-{seed_slot}-{group_id}-physical-binding.json"
            )
            controller.write_gpu_binding_receipt(
                run_root,
                ordinal=ordinal,
                seed_slot=seed_slot,
                group_id=group_id,
                output=binding_path,
            )
        controller.write_gpu_admission_receipt(
            run_root,
            run_root / "gpu-admission-receipt.json",
            require_host_monitor_alive_now=False,
        )

    def test_completion_never_authorizes_confirmation_phi_po_or_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory).resolve()
            plan_path = run_root / "controller-plan.json"
            plan_path.write_text("{}\n", encoding="ascii")
            plan = {
                "run_root": str(run_root),
                "plan_digest": "c" * 64,
                "upstream_plan": {
                    "path": str(run_root / "upstream.json"),
                    "file_sha256": "d" * 64,
                },
            }
            result = {
                "schema_version": "bernini-reserve4-fixed-generation-audit-receipt-v1",
                "analysis_split": "fit",
                "candidate_count": 40,
                "seed_cell_count": 4,
                "generation_complete": True,
                "independent_full81_review_performed": False,
                "phi_v1_extraction_authorized": False,
                "optimizer_created": False,
                "optimizer_authorized": False,
                "generated_media_is_editor_input_or_target": False,
            }
            result_path = run_root / "generation-audit.json"
            result_path.write_bytes(controller.canonical_json_bytes(result) + b"\n")
            closure_unsigned = {
                "schema_version": controller.GENERATION_CLOSURE_SCHEMA,
                "candidate_count": 40,
                "training_authorized": False,
            }
            closure = {
                **closure_unsigned,
                "receipt_digest": controller.object_sha256(closure_unsigned),
            }
            closure_path = run_root / "generation-closure-receipt.json"
            closure_path.write_bytes(
                controller.canonical_json_bytes(closure) + b"\n"
            )
            self._write_gpu_admission(run_root)
            with mock.patch.object(
                controller,
                "validate_generation_output_closure",
                return_value=closure,
            ):
                completion = controller.validate_completion(
                    plan, plan_path, _sha(plan_path)
                )
            self.assertEqual(completion["slurm_child_gpu_count"], 8)
            self.assertEqual(completion["run_sp4_shard_process_count"], 4)
            self.assertEqual(completion["world4_model_invocation_count"], 40)
            self.assertEqual(completion["sealed_shard_order"], controller.SHARD_ORDER)
            self.assertTrue(completion["all_model_invocations_strictly_serial"])
            self.assertTrue(
                completion[
                    "per_shard_observed_uuid_pci_bus_join_before_model_forward"
                ]
            )
            self.assertTrue(completion["hip_logical_order_is_observation_only"])
            self.assertTrue(completion["per_shard_exact_physical_set_verified"])
            self.assertTrue(completion["logical_order_permutation_allowed"])
            self.assertTrue(completion["cross_island_shard_visibility_rejected"])
            self.assertTrue(completion["serialized_world4_host_checkpoint_load"])
            self.assertTrue(completion["model_load_lock_node_local"])
            self.assertTrue(
                completion[
                    "model_load_lock_held_through_gpu_move_and_malloc_trim"
                ]
            )
            self.assertTrue(
                completion["t2v_vae_load_deferred_until_rank0_post_sampling"]
            )
            self.assertTrue(
                completion[
                    "world4_renderer_retirement_barrier_before_rank_zero_vae_load"
                ]
            )
            self.assertEqual(
                completion["t2v_rank_gpu_memory_limit_gib"], 52
            )
            self.assertTrue(
                completion[
                    "compile_smoke_all_rank_gpu_peak_reserved_strictly_below_limit"
                ]
            )
            self.assertLess(
                completion[
                    "compile_smoke_host_sampled_peak_memory_current_bytes"
                ],
                controller.HOST_MEMORY_SAFE_CEILING_BYTES,
            )
            self.assertLess(
                completion[
                    "terminal_host_sampled_peak_memory_current_bytes"
                ],
                controller.HOST_MEMORY_SAFE_CEILING_BYTES,
            )
            self.assertGreater(
                completion["terminal_host_sample_count"],
                completion["compile_smoke_host_sample_count"],
            )
            self.assertLessEqual(
                completion["terminal_host_observed_maximum_gap_ns"],
                controller.HOST_MEMORY_MAX_SAMPLE_GAP_NS,
            )
            self.assertEqual(
                completion["terminal_host_memory_events"],
                {"oom": 0, "oom_kill": 0},
            )
            self.assertFalse(completion["probe_receipt_pinned"])
            self.assertFalse(completion["dynamic_probe_is_release_authority"])
            self.assertFalse(completion["confirmation_generation_authorized"])
            self.assertFalse(completion["phi_v1_extraction_authorized"])
            self.assertFalse(completion["p_or_o_manifest_materialization_authorized"])
            self.assertFalse(completion["training_authorized"])
            closure_path.chmod(0o600)
            old_closure = dict(closure)
            old_closure["schema_version"] = (
                "bernini-generic-action-fit40-generation-closure-receipt-v4"
            )
            old_closure = _resign(old_closure)
            closure_path.write_bytes(
                controller.canonical_json_bytes(old_closure) + b"\n"
            )
            with mock.patch.object(
                controller,
                "validate_generation_output_closure",
                return_value=closure,
            ):
                with self.assertRaisesRegex(
                    controller.GenericActionDataPrepError,
                    "exact schema/replay differs",
                ):
                    controller.validate_completion(plan, plan_path, _sha(plan_path))

    def test_gpu_admission_rejects_probe_host_gate_order_or_serial_tamper(self) -> None:
        for case in (
            "probe-authority",
            "host-gate-reference",
            "host-gate-inline",
            "host-monitor-state",
            "serial-state",
            "logical-order-claim",
            "physical-set-claim",
            "extra-field",
            "old-schema",
            "bogus-digest",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                run_root = Path(directory).resolve()
                self._write_gpu_admission(run_root)
                admission_path = run_root / "gpu-admission-receipt.json"
                admission_path.chmod(0o600)
                admission = json.loads(admission_path.read_text(encoding="ascii"))
                if case == "probe-authority":
                    admission["probe_receipt_pinned"] = True
                    admission = _resign(admission)
                    admission_path.write_bytes(
                        controller.canonical_json_bytes(admission) + b"\n"
                    )
                elif case == "host-gate-reference":
                    admission["compile_smoke"][
                        "host_cgroup_memory_gate_reference"
                    ]["file_sha256"] = "f" * 64
                    admission = _resign(admission)
                    admission_path.write_bytes(
                        controller.canonical_json_bytes(admission) + b"\n"
                    )
                elif case == "host-gate-inline":
                    admission["compile_smoke"]["host_cgroup_memory_gate"][
                        "sampled_peak_memory_current_bytes"
                    ] += 1
                    admission = _resign(admission)
                    admission_path.write_bytes(
                        controller.canonical_json_bytes(admission) + b"\n"
                    )
                elif case == "host-monitor-state":
                    admission["compile_smoke"]["host_cgroup_memory_gate"][
                        "monitor_alive_at_gate"
                    ] = False
                    admission = _resign(admission)
                    admission_path.write_bytes(
                        controller.canonical_json_bytes(admission) + b"\n"
                    )
                elif case == "serial-state":
                    admission["all_model_invocations_strictly_serial"] = False
                    admission = _resign(admission)
                    admission_path.write_bytes(
                        controller.canonical_json_bytes(admission) + b"\n"
                    )
                elif case in {"logical-order-claim", "physical-set-claim"}:
                    if case == "logical-order-claim":
                        admission["hip_logical_order_is_observation_only"] = False
                    else:
                        admission["per_shard_exact_physical_set_verified"] = False
                    admission = _resign(admission)
                    admission_path.write_bytes(
                        controller.canonical_json_bytes(admission) + b"\n"
                    )
                elif case == "extra-field":
                    admission["unsealed_claim"] = True
                    admission = _resign(admission)
                    admission_path.write_bytes(
                        controller.canonical_json_bytes(admission) + b"\n"
                    )
                elif case == "old-schema":
                    admission["schema_version"] = (
                        "bernini-generic-action-fit40-gpu-admission-receipt-v8"
                    )
                    admission = _resign(admission)
                    admission_path.write_bytes(
                        controller.canonical_json_bytes(admission) + b"\n"
                    )
                else:
                    admission["receipt_digest"] = "0" * 64
                    admission_path.write_bytes(
                        controller.canonical_json_bytes(admission) + b"\n"
                    )
                with self.assertRaises(controller.GenericActionDataPrepError):
                    controller.validate_gpu_admission_receipt(run_root)

    def test_live_v6_logical_permutation_maps_to_exact_physical_islands(self) -> None:
        # Read-only live evidence from failed run v6:
        # all8 file SHA a2b99429..., observation ff175f75...;
        # shard-a file SHA ff15d629..., observation 65c192b3....
        # HIP logical ordinals were permuted within each physical XGMI island.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inventory_path, inventory, inventory_sha = (
                self._write_live_v6_physical_inventory(root / "inventory")
            )
            all8_path = root / "all8.json"
            all8_order = [2, 3, 0, 1, 6, 7, 4, 5]
            _, all8_sha = self._write_runtime_mapping(
                all8_path,
                all8_order,
                None,
                inventory_path,
                inventory,
                inventory_sha,
                LIVE_V6_GPU_IDENTITIES,
            )
            for label, observed_order, expected in (
                ("sp4-a", [2, 3, 0, 1], [0, 1, 2, 3]),
                ("sp4-b", [6, 7, 4, 5], [4, 5, 6, 7]),
            ):
                observed_path = root / f"{label}.json"
                _, observed_sha = self._write_runtime_mapping(
                    observed_path,
                    observed_order,
                    ",".join(str(index) for index in expected),
                    inventory_path,
                    inventory,
                    inventory_sha,
                    LIVE_V6_GPU_IDENTITIES,
                )
                joined = controller.validate_rocm_runtime_mapping_join(
                    all8_path,
                    all8_sha,
                    observed_path,
                    observed_sha,
                    expected,
                )
                self.assertEqual(joined["all8_logical_to_physical_order"], all8_order)
                self.assertEqual(
                    joined["observed_logical_to_physical_order"], observed_order
                )
                self.assertEqual(
                    [row["physical_index"] for row in joined["canonical_physical_identity_rows"]],
                    expected,
                )
                self.assertTrue(joined["mapping_exact_physical_set_verified"])
                self.assertTrue(joined["logical_order_permutation_allowed"])
                self.assertTrue(joined["cross_island_visibility_rejected"])

    def test_all8_cross_island_logical_permutation_is_not_physical_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inventory_path, inventory, inventory_sha = (
                self._write_physical_inventory(root / "inventory")
            )
            all8_path = root / "all8.json"
            _, all8_sha = self._write_runtime_mapping(
                all8_path,
                [0, 4, 1, 5, 2, 6, 3, 7],
                None,
                inventory_path,
                inventory,
                inventory_sha,
            )
            observed_path = root / "observed.json"
            _, observed_sha = self._write_runtime_mapping(
                observed_path,
                [3, 1, 0, 2],
                "0,1,2,3",
                inventory_path,
                inventory,
                inventory_sha,
            )
            joined = controller.validate_rocm_runtime_mapping_join(
                all8_path,
                all8_sha,
                observed_path,
                observed_sha,
                [0, 1, 2, 3],
            )
            self.assertEqual(
                joined["observed_logical_to_physical_order"], [3, 1, 0, 2]
            )

    def test_runtime_mapping_rejects_wrong_or_cross_island_and_identity_tamper(self) -> None:
        for case in (
            "ignored-mask",
            "wrong-island",
            "duplicate",
            "cross-island-visible-set",
            "all8-identity-mismatch",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                inventory_path, inventory, inventory_sha = (
                    self._write_physical_inventory(root / "inventory")
                )
                all8_path = root / "all8.json"
                _, all8_sha = self._write_runtime_mapping(
                    all8_path,
                    [2, 3, 0, 1, 6, 7, 4, 5],
                    None,
                    inventory_path,
                    inventory,
                    inventory_sha,
                )
                observed_path = root / "observed.json"
                expected = [0, 1, 2, 3]
                if case == "ignored-mask":
                    physical = list(range(8))
                elif case == "wrong-island":
                    physical = [6, 7, 4, 5]
                elif case == "duplicate":
                    physical = [0, 0, 2, 3]
                else:
                    physical = [0, 4, 1, 5] if case == "cross-island-visible-set" else [2, 3, 0, 1]
                observed, observed_sha = self._write_runtime_mapping(
                    observed_path,
                    physical,
                    "0,1,2,3",
                    inventory_path,
                    inventory,
                    inventory_sha,
                )
                if case == "all8-identity-mismatch":
                    observed["devices"][0]["hip_uuid_hex"] = "f" * 32
                    observed = _resign(observed, field="observation_digest")
                    observed_path.write_bytes(
                        controller.canonical_json_bytes(observed) + b"\n"
                    )
                    observed_sha = _sha(observed_path)
                with self.assertRaises(controller.GenericActionDataPrepError):
                    controller.validate_rocm_runtime_mapping_join(
                        all8_path,
                        all8_sha,
                        observed_path,
                        observed_sha,
                        expected,
                    )

    def test_generation_closure_replays_four_shards_and_exact40_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture = self._generation_closure_fixture(root)
            closure = self._validate_generation_fixture(*fixture)
            self.assertEqual(closure["schema_version"], controller.GENERATION_CLOSURE_SCHEMA)
            self.assertEqual(closure["sealed_shard_order"], controller.SHARD_ORDER)
            self.assertEqual(len(closure["shard_receipts"]), 4)
            self.assertEqual(len(closure["candidate_receipts"]), 40)
            self.assertEqual(len(closure["candidate_artifact_closures"]), 40)
            self.assertTrue(closure["generation_roots_exact_member_closure"])
            self.assertTrue(closure["serialized_world4_host_checkpoint_load"])
            self.assertTrue(closure["model_load_lock_node_local"])
            self.assertTrue(
                closure[
                    "model_load_lock_held_through_gpu_move_and_malloc_trim"
                ]
            )
            self.assertTrue(
                closure["t2v_vae_load_deferred_until_rank0_post_sampling"]
            )
            self.assertTrue(
                closure[
                    "world4_renderer_retirement_barrier_before_rank_zero_vae_load"
                ]
            )
            self.assertFalse(closure["training_authorized"])

    def test_generation_closure_rejects_hostile_receipts_and_tree_tamper(self) -> None:
        cases = (
            "generation-audit-extra-field",
            "generation-audit-bogus-digest",
            "extra-file",
            "extra-directory",
            "missing-shard-receipt",
            "swapped-shard-receipt",
            "symlink",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                plan, plan_path, plan_sha, roots, audit_path = (
                    self._generation_closure_fixture(root)
                )
                if case == "generation-audit-extra-field":
                    audit = json.loads(audit_path.read_text(encoding="ascii"))
                    audit["unsealed_claim"] = True
                    audit = _resign(audit)
                    audit_path.write_bytes(
                        controller.canonical_json_bytes(audit) + b"\n"
                    )
                elif case == "generation-audit-bogus-digest":
                    audit = json.loads(audit_path.read_text(encoding="ascii"))
                    audit["receipt_digest"] = "0" * 64
                    audit_path.write_bytes(
                        controller.canonical_json_bytes(audit) + b"\n"
                    )
                elif case == "extra-file":
                    candidate_id = plan["shards"][0]["candidate_ids"][0]
                    (roots[0] / candidate_id / "extra.bin").write_bytes(b"extra")
                elif case == "extra-directory":
                    candidate_id = plan["shards"][0]["candidate_ids"][0]
                    (roots[0] / candidate_id / "extra-dir").mkdir()
                elif case == "missing-shard-receipt":
                    (roots[2] / "reserve4-generation-shard-receipt-v1.json").unlink()
                elif case == "swapped-shard-receipt":
                    first = roots[0] / "reserve4-generation-shard-receipt-v1.json"
                    second = roots[1] / "reserve4-generation-shard-receipt-v1.json"
                    first_raw, second_raw = first.read_bytes(), second.read_bytes()
                    first.write_bytes(second_raw)
                    second.write_bytes(first_raw)
                else:
                    candidate_id = plan["shards"][0]["candidate_ids"][0]
                    target = roots[0] / candidate_id / "t2v.mp4"
                    (roots[0] / candidate_id / "linked.mp4").symlink_to(target)
                with self.assertRaises(controller.GenericActionDataPrepError):
                    self._validate_generation_fixture(
                        plan, plan_path, plan_sha, roots, audit_path
                    )

    def test_r14_actual_safetensors_container_replay_and_hostile_payloads(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            gaussian_metadata = dict(
                runner._SMOKE_SAFETENSORS_METADATA[
                    "official_initial_gaussian"
                ]
            )
            reversed_metadata = dict(reversed(list(gaussian_metadata.items())))
            first_artifact = _write_test_f32_safetensors(
                root / "gaussian-a.safetensors",
                key="official_initial_gaussian",
                metadata=gaussian_metadata,
                values=(1.0, -2.0, 3.5, 4.25),
            )
            second_artifact = _write_test_f32_safetensors(
                root / "gaussian-b.safetensors",
                key="official_initial_gaussian",
                metadata=reversed_metadata,
                values=(1.0, -2.0, 3.5, 4.25),
            )
            first = runner._parse_exact_f32_safetensors(
                first_artifact, artifact_name="official_initial_gaussian"
            )
            second = runner._parse_exact_f32_safetensors(
                second_artifact, artifact_name="official_initial_gaussian"
            )
            self.assertNotEqual(
                first["container_file_sha256"], second["container_file_sha256"]
            )
            self.assertEqual(first["semantic_evidence"], second["semantic_evidence"])
            self.assertEqual(
                first["semantic_evidence"]["tensor_identity"][
                    "raw_storage_sha256"
                ],
                hashlib.sha256(
                    struct.pack("<4f", 1.0, -2.0, 3.5, 4.25)
                ).hexdigest(),
            )

            changed_artifact = _write_test_f32_safetensors(
                root / "gaussian-changed.safetensors",
                key="official_initial_gaussian",
                metadata=gaussian_metadata,
                values=(1.0, -2.0, 3.5, 4.5),
            )
            changed = runner._parse_exact_f32_safetensors(
                changed_artifact, artifact_name="official_initial_gaussian"
            )
            self.assertNotEqual(
                first["semantic_evidence"]["tensor_identity"],
                changed["semantic_evidence"]["tensor_identity"],
            )

            hostile_metadata = dict(gaussian_metadata)
            hostile_metadata["source"] = "hostile-origin"
            hostile_metadata_artifact = _write_test_f32_safetensors(
                root / "gaussian-hostile-metadata.safetensors",
                key="official_initial_gaussian",
                metadata=hostile_metadata,
                values=(1.0, -2.0, 3.5, 4.25),
            )
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError, "safetensors metadata differs"
            ):
                runner._parse_exact_f32_safetensors(
                    hostile_metadata_artifact,
                    artifact_name="official_initial_gaussian",
                )

            nonfinite_artifact = _write_test_f32_safetensors(
                root / "gaussian-nonfinite.safetensors",
                key="official_initial_gaussian",
                metadata=gaussian_metadata,
                values=(1.0, float("inf")),
            )
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError, "tensor is non-finite"
            ):
                runner._parse_exact_f32_safetensors(
                    nonfinite_artifact, artifact_name="official_initial_gaussian"
                )

            clean_artifact = _write_test_f32_safetensors(
                root / "clean.safetensors",
                key="normalized_clean_latent",
                metadata=dict(
                    runner._SMOKE_SAFETENSORS_METADATA[
                        "predecode_clean_latent"
                    ]
                ),
                values=(-0.5, 0.25),
            )
            clean = runner._parse_exact_f32_safetensors(
                clean_artifact, artifact_name="predecode_clean_latent"
            )
            self.assertEqual(
                clean["semantic_evidence"]["tensor_identity"]["label"],
                "generated_t2v",
            )

    def test_r14_full_recursive_semantic_parity_hostile_matrix(self) -> None:
        runner = controller.reserve4_runner
        authority = runner._r10_smoke_semantic_authority_rows()
        runner._validate_r10_smoke_semantic_parity(authority)

        r13_containers = dict(runner._R10_SMOKE_OBSERVED_CONTAINER_FILE_SHA256)
        r13_containers["official_initial_gaussian"] = (
            "1269d8542e6e0d128c46d65328df56e47937f671e0e343678c12c3e8ba543958"
        )
        r13_containers["predecode_clean_latent"] = (
            "cea162214266659740d235dd6425d06263e226e7f3f3860fa94473a6eb481be9"
        )
        changed_containers = runner._r10_smoke_semantic_authority_rows(
            r13_containers
        )
        runner._validate_r10_smoke_semantic_parity(changed_containers)

        fresh_a = dict(
            authority[2]["semantic_identity"]["artifact"],
            path="/tmp/fresh-a/tensor.safetensors",
            sha256="1" * 64,
        )
        fresh_b = dict(
            authority[2]["semantic_identity"]["artifact"],
            path="/tmp/fresh-b/tensor.safetensors",
            sha256="2" * 64,
        )
        self.assertEqual(
            runner._artifact_semantics(
                fresh_a, artifact_name="predecode_clean_latent"
            ),
            runner._artifact_semantics(
                fresh_b, artifact_name="predecode_clean_latent"
            ),
        )
        for volatile_key in ("path", "sha256"):
            injected = dict(fresh_a)
            injected["nested"] = {volatile_key: "3" * 64}
            with self.subTest(volatile_key=volatile_key), self.assertRaisesRegex(
                runner.Reserve4GenerationError, "unexpected volatile key"
            ):
                runner._artifact_semantics(
                    injected, artifact_name="predecode_clean_latent"
                )

        hostile_edits = (
            ("requested_device", "cpu"),
            ("origin", "hostile-origin"),
        )
        for field, value in hostile_edits:
            hostile = json.loads(
                runner.canonical_json_bytes(authority).decode("ascii")
            )
            hostile[1]["semantic_identity"]["artifact"][field] = value
            hostile[1]["semantic_identity_digest"] = runner.object_sha256(
                hostile[1]["semantic_identity"]
            )
            with self.subTest(field=field), self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "r10 semantic artifact authority",
            ):
                runner._validate_r10_smoke_semantic_parity(hostile)

        explicit_semantic_mutations = (
            ("mp4-frame-count", 0, ("artifact", "frame_count"), 80),
            ("mp4-fps", 0, ("artifact", "fps"), 24),
            (
                "gaussian-raw",
                1,
                ("artifact", "raw_value_sha256"),
                "1" * 64,
            ),
            (
                "gaussian-content",
                1,
                ("artifact", "content_sha256"),
                "2" * 64,
            ),
            ("gaussian-shape", 1, ("artifact", "shape"), [1, 16, 20, 74, 50]),
            ("gaussian-dtype", 1, ("artifact", "dtype"), "torch.float16"),
            ("gaussian-seed", 1, ("artifact", "generator_initial_seed"), 7),
            (
                "gaussian-coordinate",
                1,
                ("artifact", "coordinate"),
                "hostile-coordinate",
            ),
            (
                "latent-raw",
                2,
                ("loaded_safetensors", "tensor_identity", "raw_storage_sha256"),
                "3" * 64,
            ),
            (
                "latent-content",
                2,
                ("loaded_safetensors", "tensor_identity", "content_sha256"),
                "4" * 64,
            ),
            (
                "latent-shape",
                2,
                ("loaded_safetensors", "tensor_identity", "shape"),
                [1, 16, 20, 74, 50],
            ),
            (
                "latent-dtype",
                2,
                ("loaded_safetensors", "tensor_identity", "dtype"),
                "torch.float16",
            ),
            (
                "latent-coordinate",
                2,
                ("artifact", "coordinate"),
                "hostile-coordinate",
            ),
        )
        for label, row_index, path, value in explicit_semantic_mutations:
            hostile = json.loads(
                runner.canonical_json_bytes(authority).decode("ascii")
            )
            cursor = hostile[row_index]["semantic_identity"]
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            hostile[row_index]["semantic_identity_digest"] = runner.object_sha256(
                hostile[row_index]["semantic_identity"]
            )
            with self.subTest(label=label), self.assertRaisesRegex(
                runner.Reserve4GenerationError, "r10 semantic artifact authority"
            ):
                runner._validate_r10_smoke_semantic_parity(hostile)

        omitted = json.loads(runner.canonical_json_bytes(authority).decode("ascii"))
        omitted[1]["semantic_identity"]["artifact"].pop("observer_only")
        omitted[1]["semantic_identity_digest"] = runner.object_sha256(
            omitted[1]["semantic_identity"]
        )
        with self.assertRaisesRegex(
            runner.Reserve4GenerationError, "r10 semantic artifact authority"
        ):
            runner._validate_r10_smoke_semantic_parity(omitted)

        tensor_tamper = json.loads(
            runner.canonical_json_bytes(authority).decode("ascii")
        )
        tensor_tamper[2]["semantic_identity"]["loaded_safetensors"][
            "tensor_identity"
        ]["raw_storage_sha256"] = "f" * 64
        tensor_tamper[2]["semantic_identity_digest"] = runner.object_sha256(
            tensor_tamper[2]["semantic_identity"]
        )
        with self.assertRaisesRegex(
            runner.Reserve4GenerationError, "r10 semantic artifact authority"
        ):
            runner._validate_r10_smoke_semantic_parity(tensor_tamper)

        mp4_tamper = json.loads(runner.canonical_json_bytes(authority).decode("ascii"))
        mp4_tamper[0]["observed_container_file_sha256"] = "e" * 64
        with self.assertRaisesRegex(
            runner.Reserve4GenerationError, "MP4 container differs"
        ):
            runner._validate_r10_smoke_semantic_parity(mp4_tamper)

    def test_r14_end_to_end_fresh_roots_accept_same_loaded_tensors(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            roots = [root / "fresh-a", root / "fresh-b"]
            for fresh_root in roots:
                fresh_root.mkdir()

            def write_pair(fresh_root: Path, reverse_metadata: bool) -> tuple[dict, dict]:
                gaussian_metadata = dict(
                    runner._SMOKE_SAFETENSORS_METADATA[
                        "official_initial_gaussian"
                    ]
                )
                clean_metadata = dict(
                    runner._SMOKE_SAFETENSORS_METADATA[
                        "predecode_clean_latent"
                    ]
                )
                if reverse_metadata:
                    gaussian_metadata = dict(
                        reversed(list(gaussian_metadata.items()))
                    )
                    clean_metadata = dict(reversed(list(clean_metadata.items())))
                gaussian_artifact = _write_test_f32_safetensors(
                    fresh_root / "gaussian.safetensors",
                    key="official_initial_gaussian",
                    metadata=gaussian_metadata,
                    values=(0.5, -1.25, 2.0, 4.0),
                )
                clean_artifact = _write_test_f32_safetensors(
                    fresh_root / "clean.safetensors",
                    key="normalized_clean_latent",
                    metadata=clean_metadata,
                    values=(-0.75, 0.125, 1.5, 3.0),
                )
                return gaussian_artifact, clean_artifact

            file_pairs = [
                write_pair(roots[0], False),
                write_pair(roots[1], True),
            ]

            def semantic_rows(
                fresh_root: Path, gaussian_file: dict, clean_file: dict
            ) -> list[dict]:
                gaussian_loaded = runner._parse_exact_f32_safetensors(
                    gaussian_file, artifact_name="official_initial_gaussian"
                )["semantic_evidence"]["tensor_identity"]
                clean_loaded = runner._parse_exact_f32_safetensors(
                    clean_file, artifact_name="predecode_clean_latent"
                )["semantic_evidence"]["tensor_identity"]
                gaussian = {
                    **gaussian_file,
                    "tensor_key": "official_initial_gaussian",
                    "tensor_value_sha256": gaussian_loaded["raw_storage_sha256"],
                    "raw_value_sha256": gaussian_loaded["raw_storage_sha256"],
                    "content_sha256": gaussian_loaded["content_sha256"],
                    "shape": gaussian_loaded["shape"],
                    "dtype": gaussian_loaded["dtype"],
                    "stored_dtype": gaussian_loaded["dtype"],
                    "requested_device": "cuda:0",
                    "generator_initial_seed": 123,
                    "origin": "observed-native",
                    "all_rank_identity": {
                        "all_rank_exact": True,
                        "identity": gaussian_loaded,
                    },
                }
                clean = {
                    **clean_file,
                    "tensor_key": "normalized_clean_latent",
                    "shape": clean_loaded["shape"],
                    "stored_dtype": clean_loaded["dtype"],
                    "sampler_return_dtype": clean_loaded["dtype"],
                    "coordinate": "bernini_normalized_clean_vae_latent",
                }
                mp4 = {
                    "path": str(fresh_root / "t2v.mp4"),
                    "sha256": "a" * 64,
                    "frame_count": 81,
                    "fps": 25,
                    "height": 16,
                    "width": 16,
                    "normalized_clean_latent": clean,
                }
                receipt = {
                    "artifacts": {
                        "mp4": mp4,
                        "official_initial_gaussian": gaussian,
                        "predecode_clean_latent": clean,
                    }
                }
                native = {
                    "outputs": {"t2v": mp4},
                    "initial_noise_artifacts": {"t2v": gaussian},
                    "generated_identities": {
                        "t2v": {
                            "all_rank_exact": True,
                            "identity": clean_loaded,
                        }
                    },
                }
                return runner._smoke_semantic_artifact_rows(receipt, native)

            rows_a = semantic_rows(roots[0], *file_pairs[0])
            rows_b = semantic_rows(roots[1], *file_pairs[1])
            self.assertEqual(
                [row["semantic_identity"] for row in rows_a],
                [row["semantic_identity"] for row in rows_b],
            )
            self.assertNotEqual(
                rows_a[1]["observed_container_file_sha256"],
                rows_b[1]["observed_container_file_sha256"],
            )
            self.assertNotEqual(
                rows_a[2]["observed_container_file_sha256"],
                rows_b[2]["observed_container_file_sha256"],
            )
            synthetic_authority = {
                row["name"]: row["semantic_identity"] for row in rows_a
            }
            synthetic_digests = {
                name: runner.object_sha256(value)
                for name, value in synthetic_authority.items()
            }
            with mock.patch.object(
                runner,
                "_R10_SMOKE_SEMANTIC_AUTHORITY",
                synthetic_authority,
            ), mock.patch.object(
                runner,
                "_R10_SMOKE_SEMANTIC_AUTHORITY_DIGESTS",
                synthetic_digests,
            ):
                runner._validate_r10_smoke_semantic_parity(rows_a)
                runner._validate_r10_smoke_semantic_parity(rows_b)

    def test_compile_smoke_runs_one_full_native40_candidate_and_deletes_output(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            scratch = root / "scratch"
            scratch.mkdir()
            monitor_start, monitor_compile_path, monitor_compile, _, _ = (
                _write_host_memory_gate_fixture(root / "monitor")
            )
            host_gate_output = root / "compile-smoke-host-memory-gate.json"
            plan_path = root / "plan.json"
            plan_path.write_text("{}\n", encoding="ascii")
            task = {
                "candidate_id": "seed1-sp4-a-fit-first",
                "seed_slot": "seed1",
                "group_id": "sp4-a",
                "visible_gpus": [0, 1, 2, 3],
                "analysis_split": "fit",
                "ordinal": 0,
                "candidate_spec_path": str(root / "candidate.json"),
                "candidate_spec_sha256": "1" * 64,
                "root_spec_sha256": "2" * 64,
            }
            plan = {
                "tasks": [task],
                "analysis_split": "fit",
                "plan_digest": "3" * 64,
            }
            runtime = {
                "method_root": str(METHOD_ROOT),
                "python": {"path": str(Path(sys.executable).resolve()), "sha256": "4" * 64},
                "bernini_root": str(root / "Bernini"),
                "veomni_root": str(root / "VeOmni"),
                "checkpoint": str(root / "checkpoint"),
                "checkpoint_content_manifest": {
                    "path": str(root / "checkpoint.sha256"),
                    "sha256": "5" * 64,
                },
                "method_source_revision": "6" * 40,
                "method_source_archive_sha256": "7" * 64,
                "generation_worker": {
                    "path": str(METHOD_ROOT / "infer_pair_v5_t2v_calibration_bank.py"),
                    "sha256": "8" * 64,
                },
                "rank_cache_wrapper": {
                    "path": str(runner.RANK_EXEC),
                    "sha256": "9" * 64,
                },
                "preprocessing_tools": dict(runner.PREPROCESSING_TOOL_SHA256),
                "node_local_scratch": {
                    "path": str(scratch),
                    "filesystem_type": "ext2/ext3",
                },
                "serialized_host_checkpoint_load": {
                    "required": True,
                    "environment_variable": "NATIVE_V_AXIS_LOAD_LOCK",
                    "path": str(scratch / "renderer-load.lock"),
                    "sha256": runner.EMPTY_FILE_SHA256,
                    "mode": "0400",
                    "parent_is_authenticated_node_local_scratch": True,
                    "lock_held_through_model_to_rank_gpu_and_malloc_trim": True,
                },
                "t2v_text_encoder_rank_gpu_residency": {
                    "required": True,
                    "environment_variable": "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED",
                    "official_model_sample_preserved": True,
                    "exact_positional_cpu_offload_request_suppressed_once_per_rank": True,
                    "all_other_to_requests_delegated": True,
                    "text_encoder_retired_only_with_renderer": True,
                    "gpu_memory_limit_gib": runner.T2V_GPU_MEMORY_LIMIT_GIB,
                    "gpu_memory_limit_bytes": runner.T2V_GPU_MEMORY_LIMIT_BYTES,
                    "per_rank_max_allocated_and_reserved_receipt_required": True,
                    "all_rank_peak_reserved_strictly_below_limit_required": True,
                },
                "host_cgroup_sampled_memory_monitor": {
                    "required": True,
                    "start_receipt": monitor_compile[
                        "monitor_start_receipt"
                    ],
                    "sample_journal": monitor_start["sample_journal"],
                    "monitor_pid": monitor_start["monitor_pid"],
                    "monitor_proc_start_ticks": monitor_start[
                        "monitor_proc_start_ticks"
                    ],
                    "supervisor_pid": monitor_start["supervisor_pid"],
                    "supervisor_proc_start_ticks": monitor_start[
                        "supervisor_proc_start_ticks"
                    ],
                    "slurm_job_id": "136309",
                    "slurm_step_id": "1",
                    "cgroup_binding_digest": monitor_start["cgroup_binding"][
                        "binding_digest"
                    ],
                    "self_task_leaf_relative_path": monitor_start[
                        "cgroup_binding"
                    ]["self_leaf"]["relative_path"],
                    "sampled_limit_cgroup_relative_path": monitor_start[
                        "cgroup_binding"
                    ]["sampled_limit_cgroup"]["relative_path"],
                    "memory_max_exact_gib": runner.HOST_MEMORY_LIMIT_GIB,
                    "sampled_current_safe_ceiling_gib": (
                        runner.HOST_MEMORY_SAFE_CEILING_GIB
                    ),
                    "sample_interval_ns": runner.HOST_MEMORY_SAMPLE_INTERVAL_NS,
                    "maximum_sample_gap_ns": (
                        runner.HOST_MEMORY_MAX_SAMPLE_GAP_NS
                    ),
                    "sampling_source": runner.HOST_MEMORY_SAMPLING_SOURCE,
                    "zero_oom_and_oom_kill_required": True,
                    "coverage": (
                        "before_compile_smoke_through_terminal_after_formal40"
                    ),
                },
            }
            pair_receipt = {
                "receipt_digest": "a" * 64,
                "native_receipt_sha256": "b" * 64,
                "native_receipt_digest": "c" * 64,
                "native_receipt_path": str(root / "native-receipt.json"),
                "artifacts": {
                    name: {"path": str(root / name), "sha256": "d" * 64}
                    for name in runner._SMOKE_ARTIFACT_NAMES
                },
            }
            (root / "native-receipt.json").write_bytes(
                runner.canonical_json_bytes(
                    {"resource_lifecycle": _valid_resource_lifecycle()}
                )
                + b"\n"
            )
            commands: list[list[str]] = []
            environments: list[dict[str, str]] = []

            def fake_run(command: list[str], *, check: bool, env: dict[str, str]) -> None:
                self.assertTrue(check)
                commands.append(command)
                environments.append(env)
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True)
                (output / "pair-v5-t2v-calibration-receipt.json").write_text(
                    "synthetic-pair-receipt\n", encoding="ascii"
                )

            def fake_write_host_gate(
                output: Path,
                *,
                measurement_phase: str,
                formal_candidate_count_at_gate: int,
            ) -> tuple[dict, str]:
                self.assertEqual(
                    (measurement_phase, formal_candidate_count_at_gate),
                    ("compile_smoke_before_formal40", 0),
                )
                self.assertEqual(output, host_gate_output)
                output.write_bytes(monitor_compile_path.read_bytes())
                return monitor_compile, _sha(output)

            args = Namespace(
                plan=str(plan_path),
                expected_plan_sha256=_sha(plan_path),
                python=str(Path(sys.executable).resolve()),
                bernini_root=str(root / "Bernini"),
                veomni_root=str(root / "VeOmni"),
                checkpoint=str(root / "checkpoint"),
                checkpoint_content_manifest=str(root / "checkpoint.sha256"),
                method_source_revision="6" * 40,
                method_source_archive_sha256="7" * 64,
                master_port=29571,
                receipt_output=str(root / "compile-smoke.json"),
                host_memory_gate_output=str(host_gate_output),
            )
            with mock.patch.dict(
                os.environ,
                {
                    "ROCR_VISIBLE_DEVICES": "0,1,2,3",
                    "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
                    "NATIVE_V_AXIS_LOAD_LOCK": str(scratch / "renderer-load.lock"),
                    "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED": "1",
                    "GADP_HOST_MEMORY_SAMPLE_JOURNAL": monitor_start[
                        "sample_journal"
                    ]["path"],
                    "GADP_HOST_MEMORY_MONITOR_START_RECEIPT": monitor_compile[
                        "monitor_start_receipt"
                    ]["path"],
                    "GADP_HOST_MEMORY_MONITOR_PID": str(
                        monitor_start["monitor_pid"]
                    ),
                    "GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID": str(
                        monitor_start["supervisor_pid"]
                    ),
                },
                clear=False,
            ), mock.patch.object(
                runner,
                "load_plan",
                return_value=(plan, plan_path, _sha(plan_path)),
            ), mock.patch.object(
                runner,
                "_runtime_binding",
                return_value=(
                    runtime,
                    Path(sys.executable).resolve(),
                    METHOD_ROOT / "infer_pair_v5_t2v_calibration_bank.py",
                    runner.RANK_EXEC,
                    scratch,
                ),
            ), mock.patch.object(
                runner.subprocess, "run", side_effect=fake_run
            ), mock.patch.object(
                runner,
                "_validate_candidate_receipt",
                return_value=(pair_receipt, {"candidate_id": task["candidate_id"]}),
            ), mock.patch.object(
                runner,
                "_smoke_artifact_identities",
                return_value=runner._r10_smoke_semantic_authority_rows(),
            ), mock.patch.object(
                runner,
                "write_host_cgroup_memory_gate_receipt",
                side_effect=fake_write_host_gate,
            ):
                self.assertEqual(runner.run_compile_smoke_sp4(args), 0)
            self.assertEqual(len(commands), 1)
            self.assertIn("--no_python", commands[0])
            self.assertLess(commands[0].index("--no_python"), commands[0].index(str(runner.RANK_EXEC)))
            self.assertEqual(environments[0]["TMPDIR"], str(scratch))
            self.assertEqual(environments[0]["GADP_RANK_PYTHON_BIN"], str(Path(sys.executable).resolve()))
            self.assertEqual(
                environments[0]["NATIVE_V_AXIS_LOAD_LOCK"],
                str(scratch / "renderer-load.lock"),
            )
            self.assertEqual(
                environments[0]["NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED"],
                "1",
            )
            self.assertEqual(list(scratch.iterdir()), [])
            receipt_path = root / "compile-smoke.json"
            receipt, _, _ = runner.load_compile_smoke_receipt(
                receipt_path, _sha(receipt_path)
            )
            self.assertEqual(receipt["smoke_task"], runner._smoke_task_binding(task))
            self.assertEqual(receipt["full_native_sampling_steps"], 40)
            self.assertEqual(receipt["formal_candidate_count_at_gate"], 0)
            self.assertTrue(receipt["per_rank_gpu_peak_memory_receipt_required"])
            self.assertEqual(
                receipt["gpu_peak_reserved_limit_gib"],
                runner.T2V_GPU_MEMORY_LIMIT_GIB,
            )
            self.assertTrue(
                receipt["all_rank_gpu_peak_reserved_strictly_below_limit"]
            )
            host_reference = receipt["host_cgroup_memory_gate"]
            self.assertEqual(host_reference["path"], str(host_gate_output))
            replayed_host, _, _ = runner.load_host_cgroup_memory_gate(
                host_reference["path"],
                host_reference["file_sha256"],
                expected_phase="compile_smoke_before_formal40",
                require_monitor_alive_now=False,
            )
            self.assertEqual(
                replayed_host["memory_events_at_gate"],
                {"oom": 0, "oom_kill": 0},
            )
            self.assertLess(
                replayed_host["sampled_peak_memory_current_bytes"],
                runner.HOST_MEMORY_SAFE_CEILING_BYTES,
            )
            self.assertTrue(receipt["disposable_output_deleted"])
            self.assertEqual(
                receipt["schema_version"],
                "bernini-generic-action-fit40-compile-smoke-v6",
            )
            replay_args = Namespace(
                compile_smoke_receipt=str(receipt_path),
                expected_compile_smoke_receipt_sha256=_sha(receipt_path),
            )
            mismatched_runtime = dict(runtime)
            mismatched_runtime["method_source_archive_sha256"] = "e" * 64
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError, "runtime replay differs"
            ):
                runner._validate_compile_smoke_for_runtime(
                    replay_args,
                    plan=plan,
                    plan_path=plan_path,
                    plan_sha=_sha(plan_path),
                    runtime=mismatched_runtime,
                )
            receipt_path.chmod(0o600)
            original_receipt_bytes = receipt_path.read_bytes()
            old_schema = json.loads(original_receipt_bytes.decode("ascii"))
            old_schema["schema_version"] = (
                "bernini-generic-action-fit40-compile-smoke-v3"
            )
            old_schema = _resign(old_schema)
            receipt_path.write_bytes(runner.canonical_json_bytes(old_schema) + b"\n")
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError, "schema/digest/authority differs"
            ):
                runner.load_compile_smoke_receipt(receipt_path, _sha(receipt_path))
            receipt_path.write_bytes(original_receipt_bytes)
            stale_lifecycle = json.loads(receipt_path.read_text(encoding="ascii"))
            stale_lifecycle["candidate_evidence"]["resource_lifecycle"][
                "schema_version"
            ] = "bernini-native-t2v-resource-lifecycle-v2"
            stale_lifecycle = _resign(stale_lifecycle)
            receipt_path.write_bytes(
                runner.canonical_json_bytes(stale_lifecycle) + b"\n"
            )
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "did not prove WORLD4 load completion",
            ):
                runner.load_compile_smoke_receipt(receipt_path, _sha(receipt_path))
            receipt_path.write_bytes(original_receipt_bytes)
            gpu_at_limit = json.loads(
                receipt_path.read_text(encoding="ascii")
            )
            gpu_at_limit["candidate_evidence"]["resource_lifecycle"][
                "world4_t2v_text_encoder_gpu_residency_gate"
            ]["rank_evidence"][3]["gpu_peak_reserved_bytes"] = (
                runner.T2V_GPU_MEMORY_LIMIT_BYTES
            )
            gpu_at_limit = _resign(gpu_at_limit)
            receipt_path.write_bytes(
                runner.canonical_json_bytes(gpu_at_limit) + b"\n"
            )
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "did not prove WORLD4 load completion",
            ):
                runner.load_compile_smoke_receipt(receipt_path, _sha(receipt_path))
            receipt_path.write_bytes(original_receipt_bytes)
            host_at_limit = json.loads(
                receipt_path.read_text(encoding="ascii")
            )
            original_host_gate_bytes = host_gate_output.read_bytes()
            hostile_host_gate = json.loads(
                original_host_gate_bytes.decode("ascii")
            )
            hostile_host_gate["sampled_peak_memory_current_bytes"] = (
                runner.HOST_MEMORY_SAFE_CEILING_BYTES
            )
            hostile_host_gate = _resign(hostile_host_gate)
            host_gate_output.write_bytes(
                runner.canonical_json_bytes(hostile_host_gate) + b"\n"
            )
            host_at_limit["host_cgroup_memory_gate"] = {
                "path": str(host_gate_output),
                "file_sha256": _sha(host_gate_output),
                "receipt_digest": hostile_host_gate["receipt_digest"],
            }
            host_at_limit = _resign(host_at_limit)
            receipt_path.write_bytes(
                runner.canonical_json_bytes(host_at_limit) + b"\n"
            )
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "sampled-current safety gate failed",
            ):
                runner.load_compile_smoke_receipt(receipt_path, _sha(receipt_path))
            host_gate_output.write_bytes(original_host_gate_bytes)
            receipt_path.write_bytes(original_receipt_bytes)
            parity_tamper = json.loads(receipt_path.read_text(encoding="ascii"))
            parity_tamper["candidate_evidence"]["artifact_identities"][0][
                "observed_container_file_sha256"
            ] = "f" * 64
            parity_tamper = _resign(parity_tamper)
            receipt_path.write_bytes(
                runner.canonical_json_bytes(parity_tamper) + b"\n"
            )
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "r10 semantic artifact authority|MP4 container differs",
            ):
                runner.load_compile_smoke_receipt(receipt_path, _sha(receipt_path))
            receipt_path.write_bytes(original_receipt_bytes)
            unsigned_semantic_tamper = json.loads(
                receipt_path.read_text(encoding="ascii")
            )
            unsigned_semantic_tamper["candidate_evidence"]["artifact_identities"][
                1
            ]["semantic_identity"]["artifact"]["requested_device"] = "cpu"
            receipt_path.write_bytes(
                runner.canonical_json_bytes(unsigned_semantic_tamper) + b"\n"
            )
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError, "schema/digest/authority differs"
            ):
                runner.load_compile_smoke_receipt(receipt_path, _sha(receipt_path))
            receipt_path.write_bytes(original_receipt_bytes)
            resigned_semantic_tamper = json.loads(
                receipt_path.read_text(encoding="ascii")
            )
            gaussian_row = resigned_semantic_tamper["candidate_evidence"][
                "artifact_identities"
            ][1]
            gaussian_row["semantic_identity"]["artifact"]["origin"] = (
                "hostile-resigned-origin"
            )
            gaussian_row["semantic_identity_digest"] = runner.object_sha256(
                gaussian_row["semantic_identity"]
            )
            resigned_semantic_tamper = _resign(resigned_semantic_tamper)
            receipt_path.write_bytes(
                runner.canonical_json_bytes(resigned_semantic_tamper) + b"\n"
            )
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "r10 semantic artifact authority",
            ):
                runner.load_compile_smoke_receipt(receipt_path, _sha(receipt_path))
            receipt_path.write_bytes(original_receipt_bytes)
            hostile = json.loads(receipt_path.read_text(encoding="ascii"))
            hostile["runtime"]["node_local_scratch"]["filesystem_type"] = "nfs"
            hostile = _resign(hostile)
            receipt_path.write_bytes(runner.canonical_json_bytes(hostile) + b"\n")
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError, "runtime binding differs"
            ):
                runner.load_compile_smoke_receipt(receipt_path, _sha(receipt_path))

    def test_formal_sp4_refuses_missing_compile_smoke_before_output_creation(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            scratch = root / "scratch"
            scratch.mkdir()
            plan_path = root / "plan.json"
            plan_path.write_text("{}\n", encoding="ascii")
            task = {
                "candidate_id": "seed1-sp4-a-fit-first",
                "seed_slot": "seed1",
                "group_id": "sp4-a",
                "visible_gpus": [0, 1, 2, 3],
                "analysis_split": "fit",
                "ordinal": 0,
                "candidate_spec_path": str(root / "candidate.json"),
                "candidate_spec_sha256": "1" * 64,
                "root_spec_sha256": "2" * 64,
            }
            plan = {
                "tasks": [task],
                "analysis_split": "fit",
                "plan_digest": "3" * 64,
            }
            runtime = {
                "preprocessing_tools": dict(runner.PREPROCESSING_TOOL_SHA256)
            }
            output = root / "formal-output"
            args = Namespace(
                plan=str(plan_path),
                expected_plan_sha256=_sha(plan_path),
                seed_slot="seed1",
                group_id="sp4-a",
                compile_smoke_receipt=str(root / "absent-smoke.json"),
                expected_compile_smoke_receipt_sha256="4" * 64,
                output_dir=str(output),
            )
            with mock.patch.dict(
                os.environ, {"ROCR_VISIBLE_DEVICES": "0,1,2,3"}, clear=False
            ), mock.patch.object(
                runner,
                "load_plan",
                return_value=(plan, plan_path, _sha(plan_path)),
            ), mock.patch.object(
                runner,
                "_runtime_binding",
                return_value=(
                    runtime,
                    Path(sys.executable).resolve(),
                    METHOD_ROOT / "infer_pair_v5_t2v_calibration_bank.py",
                    runner.RANK_EXEC,
                    scratch,
                ),
            ):
                with self.assertRaises(runner.Reserve4GenerationError):
                    runner.run_sp4(args)
            self.assertFalse(output.exists())

    def test_rank_wrapper_uses_private_node_local_caches_and_rejects_nfs(self) -> None:
        wrapper = METHOD_ROOT / "scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
        source = wrapper.read_text(encoding="utf-8")
        self.assertIn('stat -f -c \'%T\'', source)
        self.assertIn("ext2/ext3|xfs|tmpfs", source)
        self.assertNotIn("nfs)", source)
        self.assertIn('rank_root="$(mktemp -d --', source)
        for variable in (
            "TMPDIR",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "MIOPEN_USER_DB_PATH",
            "MIOPEN_CUSTOM_CACHE_DIR",
        ):
            self.assertIn(f"export {variable}=", source)

    def test_runtime_binding_authenticates_empty_node_local_model_load_lock(self) -> None:
        runner = controller.reserve4_runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            scratch = root / "scratch"
            bernini = root / "Bernini"
            veomni = root / "VeOmni"
            checkpoint = root / "checkpoint"
            for path in (scratch, bernini, veomni, checkpoint):
                path.mkdir()
            checkpoint_manifest = root / "checkpoint.sha256"
            checkpoint_manifest.write_text("sealed\n", encoding="ascii")
            load_lock = scratch / "renderer-load.lock"
            load_lock.write_bytes(b"")
            load_lock.chmod(0o400)
            monitor_root = root / "monitor"
            monitor_start, _, _, _, _ = _write_host_memory_gate_fixture(
                monitor_root
            )
            monitor_start_path = (
                monitor_root / "host-cgroup-memory-monitor-start-receipt.json"
            )
            args = Namespace(
                python=str(Path(sys.executable).resolve()),
                bernini_root=str(bernini),
                veomni_root=str(veomni),
                checkpoint=str(checkpoint),
                checkpoint_content_manifest=str(checkpoint_manifest),
                method_source_revision="1" * 40,
                method_source_archive_sha256="2" * 64,
                master_port=29571,
            )
            environment = {
                "GADP_NODE_LOCAL_SCRATCH": str(scratch),
                "GADP_NODE_LOCAL_SCRATCH_FSTYPE": "ext2/ext3",
                "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
                "NATIVE_V_AXIS_LOAD_LOCK": str(load_lock),
                "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED": "1",
                "GADP_HOST_MEMORY_SAMPLE_JOURNAL": monitor_start[
                    "sample_journal"
                ]["path"],
                "GADP_HOST_MEMORY_MONITOR_START_RECEIPT": str(
                    monitor_start_path
                ),
                "GADP_HOST_MEMORY_MONITOR_PID": str(
                    monitor_start["monitor_pid"]
                ),
                "GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID": str(
                    monitor_start["supervisor_pid"]
                ),
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                runner, "_filesystem_type", return_value="ext2/ext3"
            ), mock.patch.object(
                runner,
                "assert_live_host_cgroup_memory_monitor",
                return_value=monitor_start,
            ):
                runtime, *_ = runner._runtime_binding(args)
                self.assertEqual(
                    runtime["serialized_host_checkpoint_load"]["sha256"],
                    runner.EMPTY_FILE_SHA256,
                )
                self.assertTrue(
                    runtime["serialized_host_checkpoint_load"][
                        "lock_held_through_model_to_rank_gpu_and_malloc_trim"
                    ]
                )
                self.assertTrue(
                    runtime["t2v_text_encoder_rank_gpu_residency"]["required"]
                )
                load_lock.chmod(0o600)
                with self.assertRaisesRegex(
                    runner.Reserve4GenerationError, "lock identity differs"
                ):
                    runner._runtime_binding(args)

    @staticmethod
    def _live_topology_counts(topology: str) -> tuple[int, int, int, int, int]:
        selected = subprocess.run(
            ["awk", LINK_MATRIX_AWK],
            input=topology,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout.splitlines()
        xgmi = sum(token == "XGMI" for line in selected for token in line.split())
        pcie = sum(token == "PCIE" for line in selected for token in line.split())
        numa0 = len(
            re.findall(r"GPU\[[0-3]\].*Topology.*Numa Node:\s+0$", topology, re.M)
        )
        numa1 = len(
            re.findall(r"GPU\[[4-7]\].*Topology.*Numa Node:\s+1$", topology, re.M)
        )
        return len(selected), xgmi, pcie, numa0, numa1

    def test_live_gpu280_topology_fixture_passes_and_malformed_hostile_fails(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(f"awk '{LINK_MATRIX_AWK}'", source)
        old_rows = subprocess.run(
            ["awk", r'$1~/^GPU[0-7]$/ {print}'],
            input=LIVE_GPU280_ROCM_SMI_TOPOLOGY,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout.splitlines()
        self.assertEqual(len(old_rows), 27)
        self.assertEqual(
            self._live_topology_counts(LIVE_GPU280_ROCM_SMI_TOPOLOGY),
            (8, 24, 32, 4, 4),
        )
        malformed = LIVE_GPU280_ROCM_SMI_TOPOLOGY.replace(
            "GPU0   0            XGMI         XGMI         XGMI",
            "GPU0   0            PCIE         XGMI         XGMI",
            1,
        )
        self.assertNotEqual(
            self._live_topology_counts(malformed),
            (8, 24, 32, 4, 4),
        )

    def test_monitor_failure_signal_cannot_reproduce_r11_false_exit_zero(self) -> None:
        old = subprocess.run(
            [
                "bash",
                "-c",
                (
                    "set -Eeuo pipefail; "
                    "old_cleanup(){ local status=$?; trap - EXIT TERM; exit \"${status}\"; }; "
                    "trap old_cleanup EXIT TERM; kill -TERM $$; exit 99"
                ),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(old.returncode, 0)
        new = subprocess.run(
            [
                "bash",
                "-c",
                (
                    "set -Eeuo pipefail; monitor_pid=''; "
                    "cleanup(){ local status=\"${1:?}\"; local mf=\"${2:?}\"; "
                    "trap - EXIT INT TERM HUP USR1; "
                    "if [[ \"${mf}\" == true ]]; then "
                    "if wait \"${monitor_pid}\"; then status=70; else ms=$?; fi; fi; "
                    "exit \"${status}\"; }; "
                    "on_exit(){ local status=$?; trap - EXIT; cleanup \"${status}\" false; }; "
                    "trap on_exit EXIT; trap 'cleanup 138 true' USR1; "
                    "(exit 7) & monitor_pid=$!; kill -USR1 $$; exit 99"
                ),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(new.returncode, 138)
        ordinary = subprocess.run(
            [
                "bash",
                "-c",
                (
                    "set -Eeuo pipefail; cleanup(){ local status=\"${1:?}\"; "
                    "trap - EXIT; exit \"${status}\"; }; "
                    "on_exit(){ local status=$?; trap - EXIT; cleanup \"${status}\"; }; "
                    "trap on_exit EXIT; exit 2"
                ),
            ],
            check=False,
        )
        self.assertEqual(ordinary.returncode, 2)

    def test_launcher_uses_all8_for_one_serial_four_shard_child_and_is_parent_safe(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        lowered = source.lower()
        self.assertIn("--gpus-per-task=8", source)
        self.assertNotIn("--gpus-per-task=4", source)
        self.assertIn("--gpu-bind=none", source)
        self.assertIn("0,1,2,3,4,5,6,7|0-7", source)
        self.assertNotIn("0xf0", source)
        self.assertEqual(source.count("srun --jobid="), 1)
        self.assertIn("--mem=60G", source)
        calls = (
            "run_sealed_shard 1 seed1 sp4-a 0,1,2,3",
            "run_sealed_shard 2 seed1 sp4-b 4,5,6,7",
            "run_sealed_shard 3 seed2 sp4-a 0,1,2,3",
            "run_sealed_shard 4 seed2 sp4-b 4,5,6,7",
        )
        call_offsets = [source.index(call) for call in calls]
        self.assertEqual(call_offsets, sorted(call_offsets))
        self.assertIn('export ROCR_VISIBLE_DEVICES="${visible}"', source)
        self.assertEqual(source.count("observe-gpu-mapping"), 3)
        self.assertEqual(source.count("validate-gpu-mapping"), 2)
        self.assertLess(
            source.index("observe-gpu-mapping"),
            source.index('"${generator}" run-sp4'),
        )
        self.assertIn("seal-physical-gpu-inventory", source)
        self.assertIn("seal-gpu-binding-receipt", source)
        self.assertIn("seal-gpu-admission-receipt", source)
        self.assertIn("seal-generation-closure", source)
        controller_source = (
            METHOD_ROOT / "generic_action_data_prep_controller_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("hipDeviceGetUuid", controller_source)
        self.assertIn("hipDeviceGetPCIBusId", controller_source)
        self.assertIn("physical_index", controller_source)
        self.assertIn("pci_bus_is_authoritative_join_key", controller_source)
        self.assertIn("generation_roots_exact_member_closure", controller_source)
        self.assertIn("rocm-smi --showuniqueid --showbus", source)
        self.assertIn("jobs -pr", source)
        self.assertEqual(source.count("run-sp4 \\\n"), 1)
        self.assertEqual(source.count("smoke-sp4 \\\n"), 1)
        self.assertLess(
            source.index('"${generator}" smoke-sp4'),
            source.index('"${generator}" run-sp4'),
        )
        self.assertIn("--expected-compile-smoke-receipt-sha256", source)
        self.assertIn("GADP_NODE_LOCAL_SCRATCH", source)
        self.assertIn("nfs_comgr_tmp_rejected=true", source)
        self.assertIn('export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1', source)
        self.assertIn('export NATIVE_V_AXIS_LOAD_LOCK="${model_load_lock}"', source)
        self.assertIn('readonly model_load_lock="${task_scratch}/renderer-load.lock"', source)
        self.assertIn('chmod 0400 "${model_load_lock}"', source)
        self.assertIn("serialized_world4_host_checkpoint_load=true", source)
        self.assertIn("model_load_lock_held_through_gpu_move_and_malloc_trim=true", source)
        self.assertIn("t2v_vae_load_deferred_until_rank0_post_sampling=true", source)
        self.assertIn(
            "world4_renderer_retirement_barrier_before_rank_zero_vae_load=true",
            source,
        )
        self.assertIn("GENERIC_ACTION_FIT40_GENERATION_V14_COMPLETE", source)
        self.assertNotIn("GENERIC_ACTION_FIT40_GENERATION_V12_COMPLETE", source)
        self.assertNotIn("GENERIC_ACTION_FIT40_GENERATION_V11_COMPLETE", source)
        self.assertNotIn("GENERIC_ACTION_FIT40_GENERATION_V10_COMPLETE", source)
        self.assertNotIn("GENERIC_ACTION_FIT40_GENERATION_V9_COMPLETE", source)
        self.assertNotIn("GENERIC_ACTION_FIT40_GENERATION_V8_COMPLETE", source)
        self.assertIn(
            "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup=true",
            source,
        )
        self.assertIn(
            "resource_lifecycle_receipt_proves_all_four_load_complete_before_first_sampling=true",
            source,
        )
        self.assertIn(
            "compile_smoke_asserts_world4_load_completion_ordering=true",
            source,
        )
        self.assertIn("t2v_rank_gpu_memory_limit_gib=52", source)
        self.assertIn(
            "compile_smoke_all_rank_gpu_peak_reserved_strictly_below_limit=true",
            source,
        )
        self.assertIn(
            "host_cgroup_sample_monitor_started_before_compile_smoke=true",
            source,
        )
        self.assertIn("host_cgroup_sample_interval_ns=10000000", source)
        self.assertIn("host_cgroup_max_sample_gap_ns=100000000", source)
        self.assertIn("host_live_tail_max_age_ns=100000000", source)
        self.assertIn("host_cgroup_memory_max_exactly_60_gib=true", source)
        for claim in (
            "host_cgroup_task_leaf_memory_max_unlimited=true",
            "host_cgroup_unique_60g_limit_ancestor_selected=true",
            "host_cgroup_job_64g_ancestor_verified=true",
            "host_cgroup_full_ancestor_chain_bound=true",
            "host_cgroup_hierarchy_root_memory_max_optional_and_explicit=true",
            "host_cgroup_openat_nofollow_fd_pinned=true",
            "host_cgroup_binding_revalidated_at_gates=true",
            "host_monitor_failure_propagates_nonzero=true",
        ):
            self.assertIn(claim, source)
        self.assertIn("trap 'cleanup_task_scratch 138 true' USR1", source)
        self.assertIn("signal.SIGUSR1", (
            METHOD_ROOT / "tools/reserve4_fixed_generation_sp4_v1.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("host_sampled_current_safe_ceiling_gib=56", source)
        self.assertIn(
            "compile_smoke_host_sampled_peak_strictly_below_56_gib=true",
            source,
        )
        self.assertIn(
            "compile_smoke_zero_oom_and_oom_kill_before_formal40=true",
            source,
        )
        self.assertIn(
            "terminal_host_sampled_peak_strictly_below_56_gib=true", source
        )
        self.assertIn("terminal_host_monitor_clean_exit=true", source)
        self.assertIn("terminal_host_monitor_wait_exit_status_zero=true", source)
        self.assertIn(
            "terminal_gate_created_after_bound_supervisor_wait=true", source
        )
        self.assertIn("terminal_zero_oom_and_oom_kill=true", source)
        monitor_start = source.index(
            '"${generator}" host-memory-monitor'
        )
        smoke_start = source.index('"${generator}" smoke-sp4')
        terminal_stop = source.index(
            'mkdir -m 0700 "${host_memory_stop_path}"'
        )
        terminal_wait = source.index(
            'wait "${host_memory_monitor_pid}"', terminal_stop
        )
        terminal_seal = source.index("seal-terminal-host-memory-gate")
        terminal_validate = source.index(
            "validate-terminal-host-cgroup-memory-receipt"
        )
        last_formal = source.index(
            "run_sealed_shard 4 seed2 sp4-b 4,5,6,7"
        )
        child_exit = source.index("  exit 0\nfi", terminal_validate)
        self.assertLess(monitor_start, smoke_start)
        self.assertLess(smoke_start, last_formal)
        self.assertLess(last_formal, terminal_stop)
        self.assertLess(terminal_stop, terminal_wait)
        self.assertLess(terminal_wait, terminal_seal)
        self.assertLess(terminal_seal, terminal_validate)
        self.assertLess(terminal_validate, child_exit)
        self.assertIn("assert_only_host_memory_monitor", source)
        self.assertIn("GADP_HOST_MEMORY_MONITOR_START_RECEIPT", source)
        self.assertIn("--host-memory-gate-output", source)
        self.assertNotIn("--terminal-receipt-output", source)
        forbidden_kernel_counter = "memory." + "peak"
        owned_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                METHOD_ROOT / "tools/reserve4_fixed_generation_sp4_v1.py",
                METHOD_ROOT / "generic_action_data_prep_controller_v1.py",
                METHOD_ROOT / "tools/build_generic_action_data_prep_release_v1.py",
                LAUNCHER,
                Path(__file__).resolve(),
            )
        )
        self.assertNotIn(forbidden_kernel_counter, owned_sources)
        self.assertNotIn('cache_root="${run_root}', source)
        self.assertEqual(source.count("2>&1 &"), 1)
        self.assertIn("run_sp4_shard_process_count=4", source)
        self.assertIn("world4_model_invocation_count=40", source)
        self.assertIn("all_model_invocations_strictly_serial=true", source)
        self.assertIn(
            "per_shard_observed_uuid_pci_bus_join_before_model_forward=true",
            source,
        )
        self.assertIn("validate-launch-environment", source)
        self.assertIn("rank_action_family_partition=false", source)
        self.assertIn("probe_receipt_pinned=false", source)
        self.assertIn("dynamic_probe_is_release_authority=false", source)
        self.assertIn("independent_blind_review_present=false", source)
        self.assertIn("confirmation_authorized=false", source)
        self.assertIn("phi_authorized=false", source)
        self.assertNotIn("materialize_phi", source)
        self.assertNotIn("generic_action_manifest", source)
        first_admission = source.index("assert_idle_twice\nassert_topology")
        output_creation = source.index('mkdir -m 0700 "${run_root}/logs"')
        self.assertLess(first_admission, output_creation)
        self.assertNotIn("scancel", lowered)
        self.assertNotIn("scontrol release", lowered)
        self.assertNotIn("scontrol requeue", lowered)
        self.assertNotRegex(lowered, r"kill[^\n]*136309")


if __name__ == "__main__":
    unittest.main()

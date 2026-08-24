from __future__ import annotations

import ast
import copy
from contextlib import ExitStack
import errno
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for root in (METHOD_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import pair_v5_t2v_calibration_bank_spec as bank  # noqa: E402
import full30_action_arms_incomplete_repair_exact2_plan_v1 as plan  # noqa: E402
import full30_action_arms_incomplete_repair_exact2_generator_v1 as generator  # noqa: E402
import full30_action_arms_incomplete_repair_exact2_controller_v1 as controller  # noqa: E402
import build_full30_action_arms_incomplete_repair_exact2_release_v1 as release  # noqa: E402


EXTERNAL_ROOT = Path(
    "/private/tmp/full30_fit_repair_arms4_603865eb_sealed_"
    "473d776896869b6bcfea29684099f827"
)
EXTERNAL_KEY = EXTERNAL_ROOT / "sealed_key.json"
EXTERNAL_REVIEW = Path(
    "/private/tmp/blind_packet_473d776896869b6bcfea29684099f827/"
    "reviewer_receipt.json"
)
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_full30_action_arms_incomplete_repair_exact2_136140_world4_v1.sh"
)
RESOURCE_FIXTURE_RELEASE = (
    METHOD_ROOT
    / "releases/full30_action_arms_incomplete_repair_exact2_r3_terminal_physical_detached_v2"
)
RELEASE_R4 = (
    METHOD_ROOT
    / "releases/full30_action_arms_incomplete_repair_exact2_r4_resource_reuse_cross_step"
)
RELEASE_R5 = (
    Path(os.environ["F13_TEST_RELEASE_R5"]).resolve(strict=True)
    if "F13_TEST_RELEASE_R5" in os.environ
    else METHOD_ROOT
    / "releases/full30_action_arms_incomplete_repair_exact2_r5_canonical_python_physical_pin"
)


def _caption(slot: str, group_id: str, branch: str, seed: int) -> str:
    return (
        f"A continuous realistic medium shot for {slot} {group_id} seed {seed} "
        f"performs the uniquely specified {branch} branch while identity, scene, "
        "illumination, framing, and the locked camera remain coherent."
    )


def _candidate(
    *, slot: str, split: str, group_id: str, branch: str, seed: int,
) -> dict:
    fit = split == "fit"
    if group_id == "sp4-a":
        iid = "00435ad621c44fac" if fit else f"confirm-arms-{slot}"
        prefix = (
            f"pair5-t2v-reserve4-{'v1' if slot == 'seed1' else 'seed2'}-"
            f"00435ad621c44fac"
            if fit
            else f"fixture-{slot}-confirm-arms"
        )
        actor = "woman-raised-arms-fit" if fit else f"woman-arms-{slot}"
        scene = "portrait-interior-arms-fit" if fit else f"scene-arms-{slot}"
        action = "arms-down-hands-hips-fit" if fit else f"action-arms-{slot}"
        geometry = f"/private/tmp/fixture-{slot}-arms.mp4"
    else:
        iid = "71ba57892bd043df" if fit else f"confirm-reach-{slot}"
        prefix = (
            f"pair5-t2v-reserve4-{'v1' if slot == 'seed1' else 'seed2'}-"
            f"71ba57892bd043df"
            if fit
            else f"fixture-{slot}-confirm-reach"
        )
        actor = "left-arm-performer-fit" if fit else f"performer-{slot}"
        scene = "portrait-left-arm-fit" if fit else f"scene-reach-{slot}"
        action = "fist-to-palm-down-fit" if fit else f"action-reach-{slot}"
        geometry = f"/private/tmp/fixture-{slot}-reach.mp4"
    text = _caption(slot, group_id, branch, seed)
    return {
        "candidate_id": f"{prefix}-{branch}",
        "analysis_split": split,
        "action_family_id": "articulated-pose-transition",
        "calibration_group_id": f"cell-{iid}-s{seed}",
        "prompt_group_id": f"{actor}--{scene}",
        "action_family_group_id": action,
        "actor_group_id": actor,
        "scene_group_id": scene,
        "action_group_id": action,
        "geometry_source_video": geometry,
        "geometry_source_video_sha256": hashlib.sha256(geometry.encode()).hexdigest(),
        "geometry_contract": bank.GEOMETRY_CONTRACT,
        "semantic_branch": branch,
        "full_t2v_caption": text,
        "full_t2v_caption_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "caption_contract": bank.CAPTION_CONTRACT,
        "seed": seed,
    }


def _root_spec(slot: str) -> dict:
    base = 2026080800 if slot == "seed1" else 2026080900
    groups = []
    for group_id, visible in bank.GROUP_LAYOUT:
        fit_seed = base + (21 if group_id == "sp4-a" else 24)
        confirmation_seed = base + (31 if group_id == "sp4-a" else 34)
        rows = [
            _candidate(
                slot=slot,
                split=split,
                group_id=group_id,
                branch=branch,
                seed=fit_seed if split == "fit" else confirmation_seed,
            )
            for split in ("fit", "confirmation")
            for branch in bank.MACE_BRANCH_ORDER
        ]
        groups.append(
            {"group_id": group_id, "visible_gpus": visible, "candidates": rows}
        )
    return bank.validate_root_spec(
        {
            "schema_version": bank.SCHEMA_VERSION,
            "sampling_contract": bank.SAMPLING_CONTRACT,
            "semantic_input_closure": bank.SEMANTIC_INPUT_CLOSURE,
            "artifact_use_contract": bank.ARTIFACT_USE_CONTRACT,
            "split_contract": bank.SPLIT_CONTRACT,
            "groups": groups,
        }
    )


def _write_spec(parent: Path, slot: str) -> tuple[Path, str]:
    path = parent / f"{slot}.json"
    raw = bank.canonical_json_bytes(_root_spec(slot)) + b"\n"
    path.write_bytes(raw)
    return path.resolve(strict=True), hashlib.sha256(raw).hexdigest()


def _build_plan(parent: Path) -> dict:
    seed1, sha1 = _write_spec(parent, "seed1")
    seed2, sha2 = _write_spec(parent, "seed2")
    with mock.patch.object(plan, "SEED1_SPEC_SHA256", sha1), mock.patch.object(
        plan, "SEED2_SPEC_SHA256", sha2
    ):
        return plan.build_plan(
            seed1_spec=seed1,
            expected_seed1_spec_sha256=sha1,
            seed2_spec=seed2,
            expected_seed2_spec_sha256=sha2,
            external_key=EXTERNAL_KEY,
            external_review_receipt=EXTERNAL_REVIEW,
            output_dir=parent / "exact2-plan",
        )


def _sign(value: dict) -> dict:
    return {**value, "receipt_digest": controller.object_sha256(value)}


def _resign(value: dict) -> dict:
    unsigned = copy.deepcopy(value)
    unsigned.pop("receipt_digest", None)
    return _sign(unsigned)


def _write_json(path: Path, value: dict) -> tuple[Path, str]:
    raw = controller.canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return path.resolve(strict=True), hashlib.sha256(raw).hexdigest()


def _compute_preflight_args_fixture(
    parent: Path, media: list[str]
) -> tuple[SimpleNamespace, dict, dict, Path, str]:
    """Minimal signed-reference fixture for preflight ordering hostiles.

    The two preflight tests deliberately stop before full plan replay.  Keeping
    this helper to the already-validated prepare/plan reference boundary makes
    those tests sensitive to the v2 CLI without pretending local macOS /tmp is
    the pinned gpu215 mount.
    """

    controller_plan = {
        "plan_digest": "1" * 64,
        "release": {
            "manifest_path": str((parent / "source.manifest.json").resolve()),
            "manifest_file_sha256": "2" * 64,
            "manifest_digest": "3" * 64,
        },
    }
    controller_plan_path = (parent / "controller-plan.json").resolve()
    controller_plan_sha = "4" * 64
    prepare = _sign(
        {
            "controller_plan": {
                "path": str(controller_plan_path),
                "file_sha256": controller_plan_sha,
                "plan_digest": controller_plan["plan_digest"],
            },
            "release_manifest": {
                "path": controller_plan["release"]["manifest_path"],
                "file_sha256": controller_plan["release"][
                    "manifest_file_sha256"
                ],
                "manifest_digest": controller_plan["release"]["manifest_digest"],
            },
            "scratch_root": {
                "path": str((parent / "prepared-scratch").resolve()),
                "device": 1,
                "inode": 2,
            },
            "filesystem": {"raw_filesystem_type": "xfs"},
        }
    )
    prepare_path, prepare_sha = _write_json(parent / "scratch-prepare.json", prepare)
    args = SimpleNamespace(
        controller_plan=str(controller_plan_path),
        expected_controller_plan_sha256=controller_plan_sha,
        scratch_prepare=str(prepare_path),
        expected_scratch_prepare_sha256=prepare_sha,
        ffprobe_bin="/portable/ffprobe",
        expected_ffprobe_sha256=controller.PORTABLE_FFPROBE_SHA256,
        external_action_mp4_seed1=media[0],
        external_action_mp4_seed2=media[1],
        output=str((parent / "compute-preflight.json").resolve()),
    )
    return args, prepare, controller_plan, controller_plan_path, controller_plan_sha


def _materialize_current_specialized_resource(parent: Path) -> tuple[Path, Path]:
    method_root = (parent / "methods" / "bernini_action_editing").resolve()
    tools = method_root / "tools"
    tools.mkdir(parents=True)
    base = (
        METHOD_ROOT / "tools" / "reserve4_fixed_generation_sp4_v1.py"
    ).resolve(strict=True)
    preimage = base.read_bytes()
    # Mirror the release specialization mechanically while still asserting the
    # production generator/controller exact postimage pin below.  This keeps
    # the unit fixture usable while the r6 builder is being sealed in parallel.
    assert preimage.count(b"136141") == 7 and preimage.count(b"136140") == 0
    raw = preimage.replace(b"136141", b"136140")
    resource_path = tools / generator.RESOURCE_SPECIALIZED_BASENAME
    resource_path.write_bytes(raw)
    resource_path.chmod(0o444)
    assert controller.file_sha256(resource_path) == controller.TERMINAL_RESOURCE_CONTRACT_SHA256
    return method_root, resource_path.resolve(strict=True)


def _prepare_shape_fixture() -> dict:
    step = "77"
    outer = f"/tmp/BOX-EXP-013-r6-136140-{step}"
    # Linux's st_dev encoding for major 253, minor 0.  Darwin's os.makedev
    # encodes this as a negative integer, so shape-only tests patch makedev to
    # this pinned gpu215 value rather than weakening the production contract.
    device = 0xFD00
    probe_sha = hashlib.sha256(controller.CHILD_SCRATCH_PROBE_BYTES).hexdigest()
    return _sign(
        {
            "schema_version": controller.CHILD_SCRATCH_PREPARE_SCHEMA,
            "authority": controller.CHILD_SCRATCH_AUTHORITY,
            "controller_plan": {
                "path": "/shared/controller-plan.json",
                "file_sha256": "1" * 64,
                "plan_digest": "2" * 64,
            },
            "release_manifest": {
                "path": "/shared/source.manifest.json",
                "file_sha256": "3" * 64,
                "manifest_digest": "4" * 64,
            },
            "runtime": {
                "slurm_job_id": "136140",
                "slurm_step_id": step,
                "hostname": "auh7-1b-gpu-215",
                "sole_numbered_compute_child_required": True,
            },
            "pre_environment": {
                "SLURM_TMPDIR": {"present": False, "used_as_authority": False},
                "GADP_NODE_LOCAL_SCRATCH": {"present": False},
                "GADP_NODE_LOCAL_SCRATCH_FSTYPE": {"present": False},
                "TMPDIR": {
                    "present": False,
                    "value": None,
                    "used_as_authority": False,
                },
            },
            "delivery_boundary": {
                "parent_env_u_scrub_required": [
                    "SLURM_TMPDIR",
                    "TMPDIR",
                    "GADP_NODE_LOCAL_SCRATCH",
                    "GADP_NODE_LOCAL_SCRATCH_FSTYPE",
                ],
                "delivered_to_child_after_parent_scrub_absent": True,
                "preexisting_caller_value_not_used_as_authority": True,
                "slurm_provided_tmpdir_claimed": False,
            },
            "scratch_parent": {
                "path": "/tmp",
                "canonical_non_symlink": True,
                "device": device,
                "device_major_minor": "253:0",
                "inode": 10,
                "uid": 0,
                "gid": 0,
                "mode_octal": "1777",
                "link_count": 100,
            },
            "filesystem": {
                "magic_hex": "ef53",
                "raw_filesystem_type": "ext2/ext3",
                "mountinfo": {
                    "mount_id": "31",
                    "parent_mount_id": "1",
                    "major_minor": "253:0",
                    "mount_root": "/",
                    "mount_point": "/",
                    "mount_options": "rw,relatime",
                    "filesystem_type": "ext4",
                    "mount_source": "/dev/mapper/vgroot-lvroot",
                    "super_options": "rw",
                    "mount_namespace": "mnt:[4026531840]",
                },
                "scratch_and_parent_same_device": True,
                "local_block_device_required": True,
            },
            "scratch_root": {
                "path": outer,
                "basename": Path(outer).name,
                "canonical_non_symlink": True,
                "device": device,
                "device_major_minor": "253:0",
                "inode": 11,
                "uid": 2012,
                "gid": 2000,
                "mode_octal": "0700",
                "link_count": 2,
            },
            "retained_probe_file": {
                "path": f"{outer}/{controller.CHILD_SCRATCH_PROBE_BASENAME}",
                "basename": controller.CHILD_SCRATCH_PROBE_BASENAME,
                "device": device,
                "device_major_minor": "253:0",
                "inode": 12,
                "uid": 2012,
                "gid": 2000,
                "mode_octal": "0600",
                "link_count": 1,
                "size_bytes": len(controller.CHILD_SCRATCH_PROBE_BYTES),
                "file_sha256": probe_sha,
            },
            "creation": {
                "fresh_absent_before_mkdirat": True,
                "mkdirat_create_only": True,
                "parent_directory_fsync_after_mkdir": True,
                "o_excl_no_follow_probe": True,
                "probe_file_fsync": True,
                "probe_retained_as_authorized_forensic_member": True,
                "scratch_directory_fsync_after_probe_retention": True,
                "probe_bytes_sha256": probe_sha,
            },
            "formal_candidate_count_at_gate": 0,
            "diagnostic_task_count": 0,
            "optimizer_authorized": False,
        }
    )


def _task_bind_shape_fixture(prepare: dict) -> dict:
    outer = prepare["scratch_root"]
    inner_path = f"{outer['path']}/arms-incomplete-exact2-136140-77.deadbeef"
    inner = {
        "path": inner_path,
        "basename": Path(inner_path).name,
        "device": outer["device"],
        "device_major_minor": "253:0",
        "inode": 13,
        "uid": 2012,
        "gid": 2000,
        "mode_octal": "0700",
        "link_count": 2,
    }
    reference = lambda path, digit: {
        "path": path,
        "file_sha256": digit * 64,
        "receipt_digest": digit.upper().lower() * 64,
    }
    # Use only hexadecimal lower-case digits in signed references.
    prepare_ref = reference("/shared/scratch-prepare.json", "5")
    preflight_ref = reference("/shared/compute-preflight.json", "6")
    return _sign(
        {
            "schema_version": controller.CHILD_TASK_SCRATCH_BIND_SCHEMA,
            "authority": controller.CHILD_SCRATCH_AUTHORITY,
            "runtime": prepare["runtime"],
            "scratch_prepare": prepare_ref,
            "compute_preflight": preflight_ref,
            "scratch_outer": outer,
            "scratch_inner": inner,
            "renderer_load_lock": {
                **inner,
                "path": f"{inner_path}/{controller.CHILD_RENDERER_LOAD_LOCK_BASENAME}",
                "basename": controller.CHILD_RENDERER_LOAD_LOCK_BASENAME,
                "inode": 14,
                "mode_octal": "0400",
                "link_count": 1,
                "size_bytes": 0,
                "empty_file_sha256": hashlib.sha256(b"").hexdigest(),
            },
            "retained_probe_file": {
                key: value
                for key, value in prepare["retained_probe_file"].items()
            },
            "creation": {
                "nonce_hex": "deadbeef",
                "controller_generated_nonce": True,
                "caller_path_or_nonce_allowed": False,
                "outer_inventory_exact_retained_probe_before_mkdirat": True,
                "mkdirat_create_only": True,
                "outer_directory_fsync_after_mkdir": True,
                "outer_inventory_exact_retained_probe_and_inner_after_mkdir": True,
                "renderer_lock_openat_o_excl_no_follow": True,
                "renderer_lock_fchmod_0400": True,
                "renderer_lock_file_fsync": True,
                "inner_directory_fsync_after_renderer_lock": True,
            },
            "formal_candidate_count_at_gate": 0,
            "diagnostic_task_count": 0,
            "optimizer_authorized": False,
        }
    )


def _retained_shape_fixture() -> dict:
    reference = lambda name, digit: {
        "path": f"/shared/{name}.json",
        "file_sha256": digit * 64,
        "receipt_digest": digit * 64,
    }
    references = {
        "scratch_prepare": reference("prepare", "1"),
        "compute_preflight": reference("preflight", "2"),
        "task_scratch_bind": reference("bind", "3"),
        "generation_audit": reference("generation", "4"),
        "terminal_host_gate": reference("terminal", "5"),
        "physical_attestation": reference("attestation", "6"),
    }
    prepare = _prepare_shape_fixture()
    bind = _task_bind_shape_fixture(prepare)
    tree: dict = {}
    tree_sha = controller.object_sha256(tree)
    inventory = {
        "direct_entry_count": 15,
        "direct_entry_identities": {},
        "outer_entry_count": 2,
        "outer_probe": {},
        "terminal_outer_identity": {},
        "terminal_inner_identity": {},
        "rank_root_count": 12,
        "rank_root_count_by_rank": {"0": 3, "1": 3, "2": 3, "3": 3},
        "rank_root_basenames": [],
        "retained_smoke_root": {},
        "monitor_stop_root": {},
        "renderer_load_lock": bind["renderer_load_lock"],
        "recursive_regular_file_count": 0,
        "recursive_directory_count": 0,
        "tree_inventory": tree,
        "tree_inventory_sha256": tree_sha,
        "tree_regular_file_count": 0,
        "tree_directory_count_below_outer": 0,
        "tree_logical_regular_file_bytes": 0,
        "tree_allocated_bytes_from_st_blocks_512": 0,
        "tree_maximum_mtime_ns": 1,
        "regular_file_double_sha256_same_fd_recomputed": True,
        "regular_file_ctime_stable_across_both_hash_passes": True,
        "recursive_tree_full_second_replay_equal": True,
        "same_device_no_symlink_single_link_regular_enforced": True,
        "no_unexpected_direct_entries": True,
    }
    cgroup_rows = [
        {
            "pid": pid,
            "state": "S",
            "parent_pid": 2,
            "start_ticks": pid * 10,
            "cgroup_v2_path": "/slurm/job_136140/step_77",
        }
        for pid in (100, 101)
    ]
    return _sign(
        {
            "schema_version": controller.CHILD_SCRATCH_RETAINED_TERMINAL_SCHEMA,
            "authority": controller.CHILD_SCRATCH_AUTHORITY,
            "runtime": prepare["runtime"],
            "controller_plan": {
                "path": "/shared/controller-plan.json",
                "file_sha256": "7" * 64,
                "plan_digest": "8" * 64,
            },
            **references,
            "scratch_outer_creation_identity": bind["scratch_outer"],
            "scratch_inner_creation_identity": bind["scratch_inner"],
            "renderer_load_lock_creation_identity": bind["renderer_load_lock"],
            "retained_inventory": inventory,
            "mount_snapshot": {
                "mount_namespace": "mnt:[4026531840]",
                "mountinfo_file_sha256": "9" * 64,
                "owning_mount": {
                    "mount_id": "31",
                    "parent_mount_id": "1",
                    "major_minor": "253:0",
                    "mount_root": "/",
                    "mount_point": "/",
                    "mount_options": "rw,relatime",
                    "filesystem_type": "ext4",
                    "mount_source": "/dev/mapper/vgroot-lvroot",
                    "super_options": "rw",
                    "mount_namespace": "mnt:[4026531840]",
                },
                "scratch_or_descendant_mount_points": [],
            },
            "second_terminal_cgroup_census": {
                "supervisor": cgroup_rows[0],
                "attestation_process": cgroup_rows[1],
                "same_cgroup_process_count": 2,
                "same_cgroup_processes": cgroup_rows,
                "cgroup_procs_path": "/sys/fs/cgroup/slurm/job_136140/step_77/cgroup.procs",
                "stable_membership_before": [100, 101],
                "stable_membership_after": [100, 101],
                "identities_and_start_ticks_replayed_stably": True,
                "unexpected_same_cgroup_process_count": 0,
                "numbered_steps": ["136140.77"],
                "sole_numbered_step": "136140.77",
                "cgroup_census_before_terminal_retention": True,
            },
            "host_capacity_observation": {
                "observation_node": "auh7-1b-gpu-215",
                "observation_scope": "host_wide_ext4_statvfs_at_child_terminal_seal",
                "filesystem_total_bytes": 470_343_073_792,
                "filesystem_available_bytes_at_terminal_seal": 1,
                "filesystem_total_inodes": 29_237_248,
                "filesystem_used_inodes_at_terminal_seal": 1,
                "pre_r6_read_only_observation_available_bytes": 297_197_441_024,
                "pre_r6_read_only_observation_used_inodes": 1_389_837,
                "pre_r6_read_only_observation_node": "auh7-1b-gpu-215",
                "pre_r6_read_only_observation_source": (
                    "gpu215 statvfs read-only audit before r6 candidate"
                ),
                "host_wide_values_are_not_tree_usage": True,
                "future_capacity_guaranteed": False,
            },
            "retention_semantics": {
                "retained_at_child_terminal_seal": True,
                "deletion_attempted": False,
                "cleanup_authorized": False,
                "manual_cleanup_authorized_by_release": False,
                "retained_nonreusable": True,
                "parent_physical_replay_allowed": False,
                "future_availability_guaranteed": False,
                "future_content_immutability_guaranteed": False,
                "persistence_after_step_or_reboot_guaranteed": False,
                "cluster_or_admin_cleanup_controlled_by_release": False,
                "second_point_in_time_inventory_equal_to_attestation_observation": True,
                "continuous_immutability_between_observations_guaranteed": False,
                "atomic_filesystem_snapshot_performed": False,
            },
            "formal_candidate_count": 2,
            "diagnostic_task_count": 0,
            "optimizer_authorized": False,
        }
    )


def _terminal_marker_chain_fixture() -> dict:
    reference = lambda name, digit: {
        "path": f"/shared/{name}.json",
        "file_sha256": digit * 64,
        "receipt_digest": digit * 64,
    }
    return {
        "controller_plan": {
            "path": "/shared/controller-plan.json",
            "file_sha256": "1" * 64,
            "plan_digest": "2" * 64,
        },
        "generation_audit": reference("generation", "3"),
        "terminal_host_gate": reference("terminal", "4"),
        "physical_attestation": reference("attestation", "5"),
        "scratch_retained_terminal": reference("retained", "6"),
        "blind_review_manifest": reference("manifest", "7"),
        "blind_review_key": reference("key", "8"),
        "runtime": {
            "slurm_job_id": "136140",
            "slurm_step_id": "77",
            "hostname": "auh7-1b-gpu-215",
            "sole_numbered_compute_child_required": True,
        },
        "packet_id": "9" * 32,
    }


def _terminal_physical_fixture(parent: Path) -> tuple[SimpleNamespace, SimpleNamespace]:
    sample_struct = struct.Struct(">QQQQQQQQ")
    limit = 60 * 1024**3
    packed = [
        (0, 1_000_000_000, 2_000_000_000, 1024, limit, 0, 0, 0),
        (1, 1_010_000_000, 2_010_000_000, 2048, limit, 0, 0, 0),
        (2, 1_020_000_000, 2_020_000_000, 4096, limit, 0, 0, 1),
    ]

    def sample_row(values):
        return {
            "sequence": int(values[0]),
            "wall_time_ns": int(values[1]),
            "monotonic_time_ns": int(values[2]),
            "memory_current_bytes": int(values[3]),
            "memory_max_bytes": int(values[4]),
            "memory_events": {"oom": int(values[5]), "oom_kill": int(values[6])},
            "sample_kind": "stop_final" if int(values[7]) == 1 else "periodic",
        }

    journal = parent / "host-memory-journal.bin"
    journal.write_bytes(b"".join(sample_struct.pack(*row) for row in packed))
    journal.chmod(0o400)
    journal_meta = journal.stat()
    initial = sample_row(packed[0])
    start = _sign(
        {
            "slurm_job_id": "136140",
            "slurm_step_id": "77",
            "monitor_pid": 4242,
            "monitor_proc_start_ticks": 999,
            "sample_journal": {
                "path": str(journal.resolve(strict=True)),
                "device": journal_meta.st_dev,
                "inode": journal_meta.st_ino,
                "record_size": sample_struct.size,
                "record_encoding": "fixture-u64x8",
            },
            "initial_sample": initial,
        }
    )
    start_path, start_sha = _write_json(parent / "monitor-start.json", start)
    preflight = _sign(
        {
            "schema_version": controller.COMPUTE_PREFLIGHT_SCHEMA,
            "runtime": {
                "slurm_job_id": "136140",
                "slurm_step_id": "77",
            },
        }
    )
    preflight_path, preflight_sha = _write_json(
        parent / "compute-preflight.json", preflight
    )
    resource_path = parent / "resource.py"
    resource_path.write_bytes(b"# terminal resource fixture\n")
    resource_path = resource_path.resolve(strict=True)

    def load_start(path, expected_sha):
        resolved = Path(path).resolve(strict=True)
        raw = resolved.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected_sha:
            raise RuntimeError("monitor start SHA differs")
        return json.loads(raw), resolved, observed

    def journal_prefix(start_value, *, exact_terminal_size):
        path = Path(start_value["sample_journal"]["path"])
        metadata = path.stat()
        if (
            not exact_terminal_size
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o777 != 0o400
            or metadata.st_dev != start_value["sample_journal"]["device"]
            or metadata.st_ino != start_value["sample_journal"]["inode"]
        ):
            raise RuntimeError("terminal journal identity differs")
        raw = path.read_bytes()
        if not raw or len(raw) % sample_struct.size:
            raise RuntimeError("terminal journal size differs")
        rows = [
            sample_struct.unpack_from(raw, offset)
            for offset in range(0, len(raw), sample_struct.size)
        ]
        return raw, rows, metadata, 0

    resource = SimpleNamespace(
        load_host_cgroup_memory_monitor_start=load_start,
        _journal_prefix=journal_prefix,
        _sample_row=sample_row,
        _process_identity_is_live=lambda _pid, _ticks: False,
    )
    args = SimpleNamespace(
        compute_preflight=str(preflight_path),
        expected_compute_preflight_sha256=preflight_sha,
        resource_contract=str(resource_path),
        expected_resource_contract_sha256=controller.file_sha256(resource_path),
        monitor_start_receipt=str(start_path),
        expected_monitor_start_receipt_sha256=start_sha,
        monitor_exit_status=0,
        output=str(parent / "terminal-host-gate.json"),
    )
    return resource, args


def _real_frozen_terminal_args(
    parent: Path,
    *,
    preflight_path: Path,
    preflight_sha: str,
    resource_path: Path,
) -> SimpleNamespace:
    sample_struct = struct.Struct(">QQQQQQQQ")
    limit = 60 * 1024**3
    packed = [
        (0, 1_000_000_000, 2_000_000_000, 1024, limit, 0, 0, 0),
        (1, 1_010_000_000, 2_010_000_000, 2048, limit, 0, 0, 0),
        (2, 1_020_000_000, 2_020_000_000, 4096, limit, 0, 0, 1),
    ]
    journal = parent / "real-frozen-host-memory-journal.bin"
    journal.write_bytes(b"".join(sample_struct.pack(*row) for row in packed))
    journal.chmod(0o400)
    journal_meta = journal.stat()
    governing_relative = "/slurm/uid_1/job_136140/step_77/user"
    leaf_relative = f"{governing_relative}/task_0"
    cgroup_root = Path("/sys/fs/cgroup")
    governing_path = cgroup_root / governing_relative.lstrip("/")
    leaf_path = cgroup_root / leaf_relative.lstrip("/")
    initial = {
        "sequence": 0,
        "wall_time_ns": packed[0][1],
        "monotonic_time_ns": packed[0][2],
        "memory_current_bytes": packed[0][3],
        "memory_max_bytes": packed[0][4],
        "memory_events": {"oom": 0, "oom_kill": 0},
        "sample_kind": "periodic",
    }
    unsigned = {
        "schema_version": (
            "bernini-generic-action-fit40-host-cgroup-memory-monitor-start-v2"
        ),
        "monitor_pid": 999_999_999,
        "monitor_proc_start_ticks": 1,
        "supervisor_pid": 999_999_998,
        "supervisor_proc_start_ticks": 1,
        "slurm_job_id": "136140",
        "slurm_step_id": "77",
        "monitor_started_before_compile_smoke_and_formal40": True,
        "cgroup_version": 2,
        "leaf_cgroup": {
            "relative_path": leaf_relative,
            "path": str(leaf_path),
            "device": 7,
            "inode": 11,
            "memory_max": "max",
        },
        "governing_cgroup": {
            "relative_path": governing_relative,
            "path": str(governing_path),
            "device": 7,
            "inode": 10,
            "memory_max_bytes": limit,
            "slurm_job_id": "136140",
            "slurm_step_id": "77",
            "scope": "user",
        },
        "measurement_files": {
            "memory_current": str(governing_path / "memory.current"),
            "memory_max": str(governing_path / "memory.max"),
            "memory_events": str(governing_path / "memory.events"),
        },
        "sampling_source": "cgroup_v2_memory.current_fixed_10ms",
        "sample_journal": {
            "path": str(journal.resolve(strict=True)),
            "device": journal_meta.st_dev,
            "inode": journal_meta.st_ino,
            "record_size": sample_struct.size,
            "record_encoding": (
                "big-endian-u64-sequence-wall_ns-monotonic_ns-current-max-"
                "oom-oom_kill-kind-v2"
            ),
        },
        "stop_token_path": str((parent / "real-monitor-stop").resolve()),
        "sample_interval_ns": 10_000_000,
        "maximum_sample_gap_ns": 100_000_000,
        "strict_host_memory_limit_gib": 60,
        "strict_host_memory_limit_bytes": limit,
        "host_memory_safe_ceiling_gib": 56,
        "host_memory_safe_ceiling_bytes": 56 * 1024**3,
        "initial_sample": initial,
    }
    start = {**unsigned, "receipt_digest": controller.object_sha256(unsigned)}
    start_path, start_sha = _write_json(
        parent / "real-frozen-monitor-start.json", start
    )
    return SimpleNamespace(
        compute_preflight=str(preflight_path),
        expected_compute_preflight_sha256=preflight_sha,
        resource_contract=str(resource_path),
        expected_resource_contract_sha256=(
            controller.TERMINAL_RESOURCE_CONTRACT_SHA256
        ),
        monitor_start_receipt=str(start_path),
        expected_monitor_start_receipt_sha256=start_sha,
        monitor_exit_status=0,
        output=str(parent / "real-terminal-host-gate.json"),
    )


def _fake_receipt(task: dict, parent: Path) -> tuple[dict, Path]:
    candidate = parent / task["candidate_id"]
    candidate.mkdir()
    mp4 = candidate / "t2v.mp4"
    mp4.write_bytes(f"mp4-{task['candidate_id']}".encode("ascii"))
    value = _sign(
        {
            "candidate": {
                "candidate_id": task["candidate_id"],
                "semantic_branch": "incomplete",
                "seed": task["seed"],
                "calibration_group_id": task["calibration_group_id"],
            },
            "artifacts": {
                "mp4": {
                    "path": str(mp4.resolve(strict=True)),
                    "sha256": controller.file_sha256(mp4),
                    "frame_count": 81,
                    "fps": 25,
                }
            },
        }
    )
    path = candidate / "pair-v5-t2v-calibration-receipt.json"
    path.write_bytes(controller.canonical_json_bytes(value) + b"\n")
    return value, path.resolve(strict=True)


def _proof(seed: int, task: dict, action: dict) -> dict:
    return {
        "seed": seed,
        "calibration_group_id": task["calibration_group_id"],
        "branch_order": ["action", "incomplete"],
        "candidate_ids": [action["candidate_id"], task["candidate_id"]],
        "official_gaussian_identity": {"generator_initial_seed": seed},
        "same_seed": True,
        "cross_run": True,
        "action_incomplete_official_gaussian_tensor_values_byte_equal": True,
        "physical_artifacts_reopened": True,
        "physical_safetensors_safe_open_recomputed": True,
    }


def _audit_fixture(parent: Path, exact2: dict) -> tuple[dict, list[dict], list[Path]]:
    receipt_root = parent / "receipts"
    receipt_root.mkdir()
    receipts: list[dict] = []
    paths: list[Path] = []
    rows = []
    proofs = []
    actions = exact2["external_action_cells"]
    for task, action in zip(exact2["admission_tasks"], actions):
        receipt, path = _fake_receipt(task, receipt_root)
        receipts.append(receipt)
        paths.append(path)
        rows.append(
            {
                "candidate_id": task["candidate_id"],
                "calibration_group_id": task["calibration_group_id"],
                "semantic_branch": "incomplete",
                "candidate_receipt_path": str(path),
                "candidate_receipt_file_sha256": controller.file_sha256(path),
                "candidate_receipt_digest": receipt["receipt_digest"],
                "generated_mp4_path": receipt["artifacts"]["mp4"]["path"],
                "generated_mp4_file_sha256": receipt["artifacts"]["mp4"][
                    "sha256"
                ],
                "generated_mp4_frame_count": 81,
                "generated_mp4_fps": 25,
            }
        )
        proofs.append(_proof(task["seed"], task, action))
    unsigned = {
        "schema_version": generator.AUDIT_SCHEMA,
        "plan_path": exact2["_path"],
        "plan_file_sha256": exact2["_file_sha256"],
        "plan_digest": exact2["plan_digest"],
        "dataset": plan.DATASET,
        "candidate_count": 2,
        "comparator_cell_count": 2,
        "new_branch_order": ["incomplete", "incomplete"],
        "compute_preflight": {
            "path": str((parent / "compute-preflight.json").resolve()),
            "file_sha256": "8" * 64,
            "receipt_digest": "9" * 64,
        },
        "task_scratch_bind": {
            "path": str((parent / "child-task-scratch-bind.json").resolve()),
            "file_sha256": "c" * 64,
            "receipt_digest": "d" * 64,
        },
        "rank_resource_scratch_binding": {
            "preflight_scratch_parent_path": str(
                (parent / "slurm-tmpdir").resolve()
            ),
            "rank_task_scratch_path": str(
                (parent / "slurm-tmpdir" / "rank-task").resolve()
            ),
            "filesystem_type": "xfs",
            "source_environment_variable": "GADP_NODE_LOCAL_SCRATCH_FSTYPE",
            "preflight_stat_f_matches_rank_resource_receipt": True,
            "compile_smoke_runtime_matches_rank_resource_receipt": True,
        },
        "shard_receipt": {
            "path": str((parent / "generation" / "shard-receipt.json").resolve()),
            "file_sha256": "a" * 64,
            "receipt_digest": "b" * 64,
        },
        "candidate_receipts": rows,
        "cross_run_same_gaussian_pair_proofs": proofs,
        "all_candidates_exact81": True,
        "independent_full81_review_performed": False,
        "review_admission_authorized": False,
        "q_input_authorized": False,
        "a_min_input_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "diagnostic_task_count": 0,
        "diagnostic_generation_observed_or_required": False,
        "action_generation_observed_or_required": False,
    }
    return _sign(unsigned), receipts, paths


def _validate_audit_with_physical_recompute(
    audit: dict, exact2: dict, receipts: list[dict]
) -> None:
    action_receipts = [
        {"artifacts": {"official_initial_gaussian": {}}},
        {"artifacts": {"official_initial_gaussian": {}}},
    ]
    recomputed = [
        _proof(task["seed"], task, action)
        for task, action in zip(
            exact2["admission_tasks"], exact2["external_action_cells"]
        )
    ]
    compute_ref = audit["compute_preflight"]
    task_bind_ref = audit["task_scratch_bind"]
    rank_binding = audit["rank_resource_scratch_binding"]
    shard_ref = audit["shard_receipt"]
    preflight = {
        "scratch_parent": {
            "path": rank_binding["preflight_scratch_parent_path"],
            "filesystem_type": rank_binding["filesystem_type"],
        },
        "receipt_digest": compute_ref["receipt_digest"],
    }
    task_bind = {
        "compute_preflight": compute_ref,
        "scratch_inner": {"path": rank_binding["rank_task_scratch_path"]},
        "receipt_digest": task_bind_ref["receipt_digest"],
    }
    with mock.patch.object(
        plan, "load_plan",
        return_value=(exact2, Path(exact2["_path"]), exact2["_file_sha256"]),
    ), mock.patch.object(
        controller, "_load_or_reuse_resource_contract", return_value=object()
    ), mock.patch.object(
        controller,
        "load_json",
        return_value=(
            preflight,
            Path(compute_ref["path"]),
            compute_ref["file_sha256"],
        ),
    ), mock.patch.object(
        controller, "validate_compute_preflight", return_value=preflight
    ), mock.patch.object(
        generator,
        "load_task_scratch_bind",
        return_value=(
            task_bind,
            Path(task_bind_ref["path"]),
            task_bind_ref["file_sha256"],
        ),
    ), mock.patch.object(
        generator, "replay_task_scratch_bind_physical", return_value=Path(
            rank_binding["rank_task_scratch_path"]
        )
    ), mock.patch.object(
        generator,
        "validate_shard_scratch_binding",
        return_value=(compute_ref, rank_binding, shard_ref),
    ), mock.patch.object(
        generator, "_validate_candidate_receipt", side_effect=receipts
    ), mock.patch.object(
        generator,
        "_validate_external_action_artifacts",
        side_effect=[(value, {}) for value in action_receipts],
    ), mock.patch.object(
        generator,
        "cross_run_same_gaussian_proof",
        side_effect=recomputed,
    ), mock.patch.object(
        controller,
        "probe_exact81_25fps",
        return_value={"frame_count": 81, "fps": 25},
    ), mock.patch.object(
        controller, "validate_ffprobe", return_value=Path("/bin/true")
    ):
        controller.validate_exact2_audit(audit, exact2, Path("/bin/true"))


def _review_fixture(parent: Path, exact2: dict, audit: dict) -> dict:
    generation_path, generation_sha = _write_json(
        parent / "generation-audit.json", audit
    )
    packet = parent / "blind-packet"
    reviewer = packet / "reviewer"
    reviewer.mkdir(parents=True)
    packet_id = "a" * 32
    samples = []
    mappings = []
    review_rows = []
    for index, audit_row in enumerate(audit["candidate_receipts"], start=1):
        sample_id = f"{index:024x}"
        source = Path(audit_row["generated_mp4_path"])
        opaque_filename = f"sample_{sample_id}.mp4"
        review_mp4 = reviewer / opaque_filename
        shutil.copyfile(source, review_mp4)
        review_sha = controller.file_sha256(review_mp4)
        samples.append(
            {
                "sample_id": sample_id,
                "opaque_filename": opaque_filename,
                "reviewer_mp4_path": str(review_mp4.resolve(strict=True)),
                "reviewer_mp4_file_sha256": review_sha,
                "frame_count": 81,
                "fps": 25,
            }
        )
        mappings.append(
            {
                "sample_id": sample_id,
                "opaque_filename": opaque_filename,
                "candidate_id": audit_row["candidate_id"],
                "candidate_receipt_path": audit_row["candidate_receipt_path"],
                "candidate_receipt_file_sha256": audit_row[
                    "candidate_receipt_file_sha256"
                ],
                "candidate_receipt_digest": audit_row["candidate_receipt_digest"],
                "generated_mp4_path": audit_row["generated_mp4_path"],
                "generated_mp4_file_sha256": audit_row[
                    "generated_mp4_file_sha256"
                ],
                "reviewer_mp4_path": str(review_mp4.resolve(strict=True)),
                "reviewer_mp4_file_sha256": review_sha,
            }
        )
        review_rows.append(
            {
                "sample_id": sample_id,
                "opaque_filename": opaque_filename,
                "candidate_id": audit_row["candidate_id"],
                "semantic_branch": "incomplete",
                "generated_mp4_path": audit_row["generated_mp4_path"],
                "generated_mp4_file_sha256": audit_row[
                    "generated_mp4_file_sha256"
                ],
                "reviewer_mp4_path": str(review_mp4.resolve(strict=True)),
                "reviewer_mp4_file_sha256": review_sha,
                "frame_count": 81,
                "fps": 25,
                "reviewed_frame_count": 81,
                "reviewed_frame_indices_sha256": controller.FULL81_INDEX_SHA256,
                "all_81_frames_reviewed": True,
                "classification": "correct_before_terminal_and_hold",
                "verdict": "pass",
                "terminal_action_state_absent": True,
            }
        )
    manifest = _sign(
        {
            "schema_version": controller.BLIND_REVIEW_MANIFEST_SCHEMA,
            "packet_id": packet_id,
            "plan_digest": exact2["plan_digest"],
            "generation_audit_digest": audit["receipt_digest"],
            "required_ffprobe": {
                "path": controller.PORTABLE_FFPROBE_PATH,
                "file_sha256": controller.PORTABLE_FFPROBE_SHA256,
            },
            "candidate_count": 2,
            "blinded": True,
            "review_decision_present": False,
            "sample_order": [row["sample_id"] for row in samples],
            "samples": samples,
        }
    )
    manifest_path, manifest_sha = _write_json(
        reviewer / "review-manifest.json", manifest
    )
    generation_ref = {
        "path": str(generation_path),
        "file_sha256": generation_sha,
        "receipt_digest": audit["receipt_digest"],
    }
    key = _sign(
        {
            "schema_version": controller.BLIND_REVIEW_KEY_SCHEMA,
            "packet_id": packet_id,
            "plan_digest": exact2["plan_digest"],
            "generation_audit": generation_ref,
            "review_manifest": {
                "path": str(manifest_path),
                "file_sha256": manifest_sha,
                "receipt_digest": manifest["receipt_digest"],
            },
            "required_ffprobe": {
                "path": controller.PORTABLE_FFPROBE_PATH,
                "file_sha256": controller.PORTABLE_FFPROBE_SHA256,
            },
            "candidate_count": 2,
            "mappings": mappings,
            "review_decision_present": False,
        }
    )
    key_path, key_sha = _write_json(packet / "sealed-key.json", key)
    return _sign(
        {
            "schema_version": controller.REVIEW_SCHEMA,
            "packet_id": packet_id,
            "reviewer_receipt_id": "b" * 32,
            "plan_digest": exact2["plan_digest"],
            "generation_audit": generation_ref,
            "blind_review_manifest": {
                "path": str(manifest_path),
                "file_sha256": manifest_sha,
                "receipt_digest": manifest["receipt_digest"],
            },
            "sealed_key": {
                "path": str(key_path),
                "file_sha256": key_sha,
                "receipt_digest": key["receipt_digest"],
            },
            "required_ffprobe": {
                "path": controller.PORTABLE_FFPROBE_PATH,
                "file_sha256": controller.PORTABLE_FFPROBE_SHA256,
            },
            "review_population": plan.DATASET,
            "reviewer_independent_of_generator": True,
            "reviewer_independent_of_materializer": True,
            "reviewer_did_not_read_sealed_key_before_decisions": True,
            "candidate_count": 2,
            "external_action_count": 2,
            "external_actions_reused_from_locked_blind_review": True,
            "diagnostic_candidate_count": 0,
            "candidate_reviews": review_rows,
        }
    )


def _validate_review_with_probe(review: dict, exact2: dict, audit: dict) -> None:
    with mock.patch.object(
        controller,
        "probe_exact81_25fps",
        return_value={"frame_count": 81, "fps": 25},
    ), mock.patch.object(
        controller, "validate_ffprobe", return_value=Path("/bin/true")
    ):
        controller.validate_review_admission(
            review, exact2, audit, Path("/bin/true")
        )


def _fully_resign_review_chain(
    review: dict,
    *,
    manifest_mutator=None,
    key_mutator=None,
    review_mutator=None,
) -> dict:
    result = copy.deepcopy(review)
    manifest_path = Path(result["blind_review_manifest"]["path"])
    key_path = Path(result["sealed_key"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    key = json.loads(key_path.read_text(encoding="ascii"))
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    manifest = _resign(manifest)
    _, manifest_sha = _write_json(manifest_path, manifest)
    key["review_manifest"] = {
        "path": str(manifest_path),
        "file_sha256": manifest_sha,
        "receipt_digest": manifest["receipt_digest"],
    }
    if key_mutator is not None:
        key_mutator(key)
    key = _resign(key)
    _, key_sha = _write_json(key_path, key)
    result["blind_review_manifest"] = dict(key["review_manifest"])
    result["sealed_key"] = {
        "path": str(key_path),
        "file_sha256": key_sha,
        "receipt_digest": key["receipt_digest"],
    }
    if review_mutator is not None:
        review_mutator(result)
    return _resign(result)


class Exact2PromptAndPlanTests(unittest.TestCase):
    def test_prompt_is_positive_frozen_and_has_zero_forbidden_tokens(self) -> None:
        plan._assert_prompt_freeze()
        self.assertEqual(
            plan.INCOMPLETE_PROMPT_SHA256,
            "225d66cf0ad29fa7b7b51bf6177843629f2f8710d60b3278008495cbb049cde4",
        )
        self.assertEqual(plan.FORBIDDEN_PROMPT_TOKEN_RE.findall(plan.INCOMPLETE_PROMPT), [])
        for phrase in (
            "symmetric lower-chest to mid-torso pose",
            "elbows bent at about ninety degrees",
            "forearms held horizontal",
            "hands visibly separated above the waist",
            "settles into this before-terminal pose early",
            "holds it steadily through the final frame",
        ):
            self.assertIn(phrase, plan.INCOMPLETE_PROMPT)

    def test_hostile_forbidden_prompt_token_fails(self) -> None:
        with mock.patch.object(
            plan, "INCOMPLETE_PROMPT", plan.INCOMPLETE_PROMPT + " hip"
        ):
            with self.assertRaisesRegex(
                plan.ArmsIncompleteExact2PlanError, "forbidden terminal token"
            ):
                plan._assert_prompt_freeze()

    @unittest.skipUnless(EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(), "sealed external evidence unavailable")
    def test_plan_is_exact2_incomplete_only_and_binds_two_external_actions(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            value = _build_plan(Path(temporary))
        self.assertEqual(value["formal_candidate_count"], 2)
        self.assertEqual(value["comparator_cell_count"], 2)
        self.assertEqual([row["semantic_branch"] for row in value["admission_tasks"]], ["incomplete", "incomplete"])
        self.assertEqual([row["seed"] for row in value["admission_tasks"]], [2026080821, 2026080921])
        self.assertEqual(value["shards"][0]["candidate_count"], 2)
        self.assertNotIn("diagnostic_tasks", value)
        self.assertFalse(value["execution_contract"]["action_generation_allowed"])
        self.assertEqual(value["execution_contract"]["num_inference_steps"], 40)
        self.assertEqual(value["execution_contract"]["num_frames"], 81)
        self.assertEqual(
            [row["candidate_id"] for row in value["external_action_cells"]],
            [
                "pair5-t2v-fit-repair-v1-seed1-00435ad621c44fac-action",
                "pair5-t2v-fit-repair-v1-seed2-00435ad621c44fac-action",
            ],
        )


class ExternalAuthorityTests(unittest.TestCase):
    @unittest.skipUnless(EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(), "sealed external evidence unavailable")
    def test_external_action_mp4_native_calibration_and_blind_review_are_physical(self) -> None:
        provenance, rows = plan._validate_external_action_authority(
            EXTERNAL_KEY, EXTERNAL_REVIEW
        )
        self.assertEqual(provenance["sealed_key"]["file_sha256"], plan.EXTERNAL_KEY_SHA256)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["seed"] for row in rows], [2026080821, 2026080921])
        self.assertTrue(all(row["blind_review"]["classification"] == "complete_and_hold" for row in rows))
        self.assertTrue(all(row["official_initial_gaussian"]["physically_reopen_at_completion"] for row in rows))

    @unittest.skipUnless(EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(), "sealed external evidence unavailable")
    def test_hostile_external_action_receipt_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            (root / "source_media").mkdir()
            (root / "source_receipts").mkdir()
            for label in ("formal_00", "formal_02"):
                shutil.copyfile(
                    EXTERNAL_ROOT / "source_media" / f"{label}.mp4",
                    root / "source_media" / f"{label}.mp4",
                )
                (root / "source_receipts" / label).mkdir()
                for name in ("receipt.json", "pair-v5-t2v-calibration-receipt.json"):
                    shutil.copyfile(
                        EXTERNAL_ROOT / "source_receipts" / label / name,
                        root / "source_receipts" / label / name,
                    )
            hostile = root / "source_receipts/formal_00/pair-v5-t2v-calibration-receipt.json"
            hostile.write_bytes(hostile.read_bytes() + b" ")
            with self.assertRaisesRegex(
                plan.ArmsIncompleteExact2PlanError,
                "external calibration receipt seed 2026080821 SHA-256 differs",
            ):
                plan._validate_external_action_authority(
                    EXTERNAL_KEY, EXTERNAL_REVIEW, root.resolve(strict=True)
                )


class CompletionProofHostileTests(unittest.TestCase):
    @unittest.skipUnless(EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(), "sealed external evidence unavailable")
    def test_proof_omission_tamper_and_candidate_count_fail(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            exact2 = _build_plan(root)
            audit, receipts, _ = _audit_fixture(root, exact2)
            _validate_audit_with_physical_recompute(audit, exact2, receipts)

            omitted = copy.deepcopy(audit)
            omitted.pop("receipt_digest")
            omitted.pop("cross_run_same_gaussian_pair_proofs")
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError, "field closure"
            ):
                _validate_audit_with_physical_recompute(
                    _sign(omitted), exact2, receipts
                )

            tampered = copy.deepcopy(audit)
            tampered.pop("receipt_digest")
            tampered["cross_run_same_gaussian_pair_proofs"][0][
                "action_incomplete_official_gaussian_tensor_values_byte_equal"
            ] = False
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "proofs do not replay",
            ):
                _validate_audit_with_physical_recompute(
                    _sign(tampered), exact2, receipts
                )

            widened = copy.deepcopy(audit)
            widened.pop("receipt_digest")
            widened["candidate_count"] = 3
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "authority differs",
            ):
                _validate_audit_with_physical_recompute(
                    _sign(widened), exact2, receipts
                )

    def test_cross_run_proof_requires_exact_same_gaussian_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            mp4 = root / "new.mp4"
            gaussian_file = root / "new.safetensors"
            mp4.write_bytes(b"mp4-fixture")
            gaussian_file.write_bytes(b"gaussian-fixture")
            identity = {
                "raw_value_sha256": "1" * 64,
                "content_sha256": "2" * 64,
                "shape": [1, 16, 21, 74, 50],
                "dtype": "torch.float32",
                "stored_dtype": "torch.float32",
                "generator_initial_seed": 2026080821,
            }
            task = {
                "candidate_id": "new-incomplete",
                "calibration_group_id": "cell-00435ad621c44fac-s2026080821",
                "seed": 2026080821,
            }
            action = {
                "candidate_id": "external-action",
                "calibration_group_id": task["calibration_group_id"],
                "seed": task["seed"],
                "mp4": {"runtime_path": "/external/action.mp4", "file_sha256": "3" * 64},
                "native_receipt": {"runtime_path": "/external/receipt.json", "file_sha256": "4" * 64},
                "calibration_receipt": {"runtime_path": "/external/calibration.json", "file_sha256": "5" * 64},
                "official_initial_gaussian": {
                    "runtime_path": "/external/action.safetensors",
                    "file_sha256": "6" * 64,
                    "identity": identity,
                },
            }
            incomplete = {
                "artifacts": {
                    "mp4": {
                        "path": str(mp4.resolve()),
                        "sha256": generator.file_sha256(mp4),
                        "frame_count": 81,
                    },
                    "official_initial_gaussian": {
                        **identity,
                        "path": str(gaussian_file.resolve()),
                        "sha256": generator.file_sha256(gaussian_file),
                    },
                }
            }
            action_receipt = {
                "artifacts": {"official_initial_gaussian": dict(identity)}
            }
            proof = generator.cross_run_same_gaussian_proof(
                task=task,
                action=action,
                incomplete_receipt=incomplete,
                action_receipt=action_receipt,
            )
            self.assertTrue(
                proof["action_incomplete_official_gaussian_tensor_values_byte_equal"]
            )
            incomplete["artifacts"]["official_initial_gaussian"]["content_sha256"] = "7" * 64
            with self.assertRaisesRegex(
                generator.ArmsIncompleteExact2GenerationError,
                "cross-run action/incomplete Gaussian differs",
            ):
                generator.cross_run_same_gaussian_proof(
                    task=task,
                    action=action,
                    incomplete_receipt=incomplete,
                    action_receipt=action_receipt,
                )


class ReviewGeneratedMP4AuthorityTests(unittest.TestCase):
    @unittest.skipUnless(
        EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(),
        "sealed external evidence unavailable",
    )
    def test_review_rows_exactly_bind_generation_receipts_and_physical_mp4s(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            exact2 = _build_plan(root)
            audit, receipts, _ = _audit_fixture(root, exact2)
            _validate_audit_with_physical_recompute(audit, exact2, receipts)
            review = _review_fixture(root, exact2, audit)
            _validate_review_with_probe(review, exact2, audit)

    @unittest.skipUnless(
        EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(),
        "sealed external evidence unavailable",
    )
    def test_post_generation_blind_packet_is_create_only_and_has_no_decision(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            exact2 = _build_plan(root)
            audit, _, _ = _audit_fixture(root, exact2)
            generation_path, generation_sha = _write_json(
                root / "generation-audit-for-packet.json", audit
            )
            output = root / "packet-output"
            args = SimpleNamespace(
                controller_plan=str(root / "controller-plan.json"),
                expected_controller_plan_sha256="1" * 64,
                generation_audit=str(generation_path),
                expected_generation_audit_sha256=generation_sha,
                ffprobe_bin="/bin/true",
                expected_ffprobe_sha256="2" * 64,
                output_dir=str(output),
            )
            with mock.patch.object(
                controller,
                "load_controller_plan",
                return_value=({}, Path(args.controller_plan), "1" * 64, exact2),
            ), mock.patch.object(
                controller, "validate_ffprobe", return_value=Path("/bin/true")
            ), mock.patch.object(
                controller, "validate_exact2_audit", return_value=audit
            ), mock.patch.object(
                controller,
                "probe_exact81_25fps",
                return_value={"frame_count": 81, "fps": 25},
            ), mock.patch.object(
                controller.os, "getegid", return_value=os.stat(root).st_gid
            ):
                result = controller.seal_blind_review_input(args)
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "fresh absolute directory",
                ):
                    controller.seal_blind_review_input(args)
            manifest = json.loads(
                Path(result["review_manifest"]["path"]).read_text(encoding="ascii")
            )
            self.assertFalse(manifest["review_decision_present"])
            self.assertNotIn("classification", json.dumps(manifest))
            self.assertNotIn("verdict", json.dumps(manifest))
            self.assertFalse(result["review_decision_present"])

    @unittest.skipUnless(
        EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(),
        "sealed external evidence unavailable",
    )
    def test_hostile_wrong_path_sha_cross_candidate_and_extra_fields_fail(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            exact2 = _build_plan(root)
            audit, receipts, _ = _audit_fixture(root, exact2)
            _validate_audit_with_physical_recompute(audit, exact2, receipts)
            review = _review_fixture(root, exact2, audit)

            wrong_path = copy.deepcopy(review)
            wrong_path["candidate_reviews"][0]["generated_mp4_path"] = wrong_path[
                "candidate_reviews"
            ][0]["reviewer_mp4_path"]
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError, "binding failed"
            ):
                _validate_review_with_probe(_resign(wrong_path), exact2, audit)

            wrong_sha = copy.deepcopy(review)
            wrong_sha["candidate_reviews"][0]["generated_mp4_file_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError, "binding failed"
            ):
                _validate_review_with_probe(_resign(wrong_sha), exact2, audit)

            swapped = copy.deepcopy(review)
            for field in ("generated_mp4_path", "generated_mp4_file_sha256"):
                swapped["candidate_reviews"][0][field], swapped["candidate_reviews"][1][
                    field
                ] = (
                    swapped["candidate_reviews"][1][field],
                    swapped["candidate_reviews"][0][field],
                )
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError, "binding failed"
            ):
                _validate_review_with_probe(_resign(swapped), exact2, audit)

            extra_top = copy.deepcopy(review)
            extra_top["unexpected"] = False
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "top-level/population closure",
            ):
                _validate_review_with_probe(_resign(extra_top), exact2, audit)

            extra_row = copy.deepcopy(review)
            extra_row["candidate_reviews"][0]["unexpected"] = False
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError, "row field closure"
            ):
                _validate_review_with_probe(_resign(extra_row), exact2, audit)

    @unittest.skipUnless(
        EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(),
        "sealed external evidence unavailable",
    )
    def test_hostile_missing_mp4_and_fully_resigned_review_receipt_fail(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            exact2 = _build_plan(root)
            audit, receipts, _ = _audit_fixture(root, exact2)
            _validate_audit_with_physical_recompute(audit, exact2, receipts)
            review = _review_fixture(root, exact2, audit)

            resigned = copy.deepcopy(review)
            resigned["candidate_reviews"][0]["generated_mp4_file_sha256"] = "f" * 64
            resigned = _resign(resigned)
            self.assertEqual(
                resigned["receipt_digest"],
                controller.object_sha256(
                    {key: value for key, value in resigned.items() if key != "receipt_digest"}
                ),
            )
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError, "binding failed"
            ):
                _validate_review_with_probe(resigned, exact2, audit)

            Path(audit["candidate_receipts"][0]["generated_mp4_path"]).unlink()
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError, "is unavailable"
            ):
                _validate_review_with_probe(review, exact2, audit)

    @unittest.skipUnless(
        EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(),
        "sealed external evidence unavailable",
    )
    def test_fully_resigned_manifest_key_review_repoint_to_generated_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            exact2 = _build_plan(root)
            audit, receipts, _ = _audit_fixture(root, exact2)
            _validate_audit_with_physical_recompute(audit, exact2, receipts)
            review = _review_fixture(root, exact2, audit)
            generated = audit["candidate_receipts"][0]["generated_mp4_path"]
            generated_sha = audit["candidate_receipts"][0][
                "generated_mp4_file_sha256"
            ]

            def mutate_manifest(value):
                value["samples"][0]["opaque_filename"] = Path(generated).name
                value["samples"][0]["reviewer_mp4_path"] = generated
                value["samples"][0]["reviewer_mp4_file_sha256"] = generated_sha

            def mutate_key(value):
                value["mappings"][0]["opaque_filename"] = Path(generated).name
                value["mappings"][0]["reviewer_mp4_path"] = generated
                value["mappings"][0]["reviewer_mp4_file_sha256"] = generated_sha

            def mutate_review(value):
                value["candidate_reviews"][0]["opaque_filename"] = Path(
                    generated
                ).name
                value["candidate_reviews"][0]["reviewer_mp4_path"] = generated
                value["candidate_reviews"][0][
                    "reviewer_mp4_file_sha256"
                ] = generated_sha

            hostile = _fully_resign_review_chain(
                review,
                manifest_mutator=mutate_manifest,
                key_mutator=mutate_key,
                review_mutator=mutate_review,
            )
            with self.assertRaises(controller.ArmsIncompleteExact2ControllerError):
                _validate_review_with_probe(hostile, exact2, audit)

    @unittest.skipUnless(
        EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(),
        "sealed external evidence unavailable",
    )
    def test_hostile_same_inode_and_symlink_blind_copies_fail(self) -> None:
        for mode in ("hardlink", "symlink"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                dir="/private/tmp"
            ) as temporary:
                root = Path(temporary)
                exact2 = _build_plan(root)
                audit, receipts, _ = _audit_fixture(root, exact2)
                _validate_audit_with_physical_recompute(audit, exact2, receipts)
                review = _review_fixture(root, exact2, audit)
                source = Path(
                    review["candidate_reviews"][0]["generated_mp4_path"]
                )
                blind_copy = Path(
                    review["candidate_reviews"][0]["reviewer_mp4_path"]
                )
                blind_copy.unlink()
                if mode == "hardlink":
                    os.link(source, blind_copy)
                    expected = "opaque-copy path/bytes/inode topology"
                else:
                    blind_copy.symlink_to(source)
                    expected = "canonical plain file"
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError, expected
                ):
                    _validate_review_with_probe(review, exact2, audit)

    @unittest.skipUnless(
        EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(),
        "sealed external evidence unavailable",
    )
    def test_fully_resigned_wrong_opaque_basename_or_parent_fails(self) -> None:
        for mode in ("basename", "parent"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                dir="/private/tmp"
            ) as temporary:
                root = Path(temporary)
                exact2 = _build_plan(root)
                audit, receipts, _ = _audit_fixture(root, exact2)
                _validate_audit_with_physical_recompute(audit, exact2, receipts)
                review = _review_fixture(root, exact2, audit)
                row = review["candidate_reviews"][0]
                source = Path(row["generated_mp4_path"])
                current_copy = Path(row["reviewer_mp4_path"])
                if mode == "basename":
                    opaque_filename = f"wrong_{row['sample_id']}.mp4"
                    hostile_copy = current_copy.parent / opaque_filename
                else:
                    opaque_filename = row["opaque_filename"]
                    hostile_copy = current_copy.parent.parent / opaque_filename
                shutil.copyfile(source, hostile_copy)
                hostile_path = str(hostile_copy.resolve(strict=True))
                hostile_sha = controller.file_sha256(hostile_copy)

                def mutate_manifest(value):
                    value["samples"][0]["opaque_filename"] = opaque_filename
                    value["samples"][0]["reviewer_mp4_path"] = hostile_path
                    value["samples"][0][
                        "reviewer_mp4_file_sha256"
                    ] = hostile_sha

                def mutate_key(value):
                    value["mappings"][0]["opaque_filename"] = opaque_filename
                    value["mappings"][0]["reviewer_mp4_path"] = hostile_path
                    value["mappings"][0][
                        "reviewer_mp4_file_sha256"
                    ] = hostile_sha

                def mutate_review(value):
                    value["candidate_reviews"][0][
                        "opaque_filename"
                    ] = opaque_filename
                    value["candidate_reviews"][0][
                        "reviewer_mp4_path"
                    ] = hostile_path
                    value["candidate_reviews"][0][
                        "reviewer_mp4_file_sha256"
                    ] = hostile_sha

                hostile = _fully_resign_review_chain(
                    review,
                    manifest_mutator=mutate_manifest,
                    key_mutator=mutate_key,
                    review_mutator=mutate_review,
                )
                with self.assertRaises(
                    controller.ArmsIncompleteExact2ControllerError
                ):
                    _validate_review_with_probe(hostile, exact2, audit)


class PortableComputePreflightTests(unittest.TestCase):
    def test_portable_ffprobe_rejects_wrong_sha_and_nonexecutable(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            binary = Path(temporary) / "ffprobe"
            binary.write_bytes(b"portable-ffprobe-fixture")
            observed = controller.file_sha256(binary)
            binary.chmod(0o700)
            with mock.patch.object(
                controller, "PORTABLE_FFPROBE_PATH", str(binary)
            ), mock.patch.object(
                controller, "PORTABLE_FFPROBE_SHA256", observed
            ):
                self.assertEqual(
                    controller.validate_ffprobe(binary, observed), binary
                )
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "portable ffprobe",
                ):
                    controller.validate_ffprobe(binary, "0" * 64)
                binary.chmod(0o600)
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "portable ffprobe",
                ):
                    controller.validate_ffprobe(binary, observed)

    def test_scratch_stat_failure_and_nonlocal_filesystem_fail(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            scratch = Path(temporary).resolve(strict=True)
            with mock.patch.object(
                controller.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, ["stat"]),
            ):
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "cannot identify",
                ):
                    controller.filesystem_type(scratch)
            hostile = subprocess.CompletedProcess(
                ["stat"], 0, stdout="nfs\n", stderr=""
            )
            with mock.patch.object(
                controller.subprocess, "run", return_value=hostile
            ):
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "not allowed",
                ):
                    controller.filesystem_type(scratch)

    def test_compute_preflight_fails_when_external_action_probe_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            media_root = root / "source_media"
            media_root.mkdir()
            media = []
            specs = []
            for seed, basename in (
                (2026080821, "formal_00.mp4"),
                (2026080921, "formal_02.mp4"),
            ):
                path = media_root / basename
                path.write_bytes(f"media-{seed}".encode("ascii"))
                media.append(str(path.resolve(strict=True)))
                specs.append(
                    {
                        "seed": seed,
                        "basename": basename,
                        "file_sha256": controller.file_sha256(path),
                    }
                )
            args, prepare, controller_plan, controller_plan_path, controller_plan_sha = (
                _compute_preflight_args_fixture(root, media)
            )
            with mock.patch.object(
                controller, "EXTERNAL_ACTION_PREFLIGHT", tuple(specs)
            ), mock.patch.object(
                controller,
                "validate_child_scratch_prepare",
                return_value=prepare,
            ), mock.patch.object(
                controller,
                "_controller_plan_reference_without_source_replay",
                return_value=(
                    controller_plan,
                    controller_plan_path,
                    controller_plan_sha,
                ),
            ), mock.patch.object(
                controller, "validate_ffprobe", return_value=Path("/bin/true")
            ), mock.patch.object(
                controller,
                "probe_exact81_25fps",
                side_effect=controller.ArmsIncompleteExact2ControllerError(
                    "compute probe failed"
                ),
            ), mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "compute probe failed",
                ):
                    controller.seal_compute_preflight(args)

    def test_compute_preflight_rejects_caller_scratch_declaration(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            media_root = root / "source_media"
            media_root.mkdir()
            paths = []
            specs = []
            for seed, basename in (
                (2026080821, "formal_00.mp4"),
                (2026080921, "formal_02.mp4"),
            ):
                path = media_root / basename
                path.write_bytes(f"media-{seed}".encode("ascii"))
                paths.append(str(path.resolve(strict=True)))
                specs.append(
                    {
                        "seed": seed,
                        "basename": basename,
                        "file_sha256": controller.file_sha256(path),
                    }
                )
            args, prepare, controller_plan, controller_plan_path, controller_plan_sha = (
                _compute_preflight_args_fixture(root, paths)
            )
            with mock.patch.object(
                controller, "EXTERNAL_ACTION_PREFLIGHT", tuple(specs)
            ), mock.patch.object(
                controller,
                "validate_child_scratch_prepare",
                return_value=prepare,
            ), mock.patch.object(
                controller,
                "_controller_plan_reference_without_source_replay",
                return_value=(
                    controller_plan,
                    controller_plan_path,
                    controller_plan_sha,
                ),
            ), mock.patch.object(
                controller, "validate_ffprobe", return_value=Path("/bin/true")
            ), mock.patch.object(
                controller,
                "probe_exact81_25fps",
                return_value={"frame_count": 81, "fps": 25, "width": 8, "height": 8},
            ), mock.patch.dict(
                os.environ,
                {"GADP_NODE_LOCAL_SCRATCH_FSTYPE": "xfs"},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "scratch-prepare/environment binding differs",
                ):
                    controller.seal_compute_preflight(args)

    @unittest.skipUnless(
        EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(),
        "sealed external evidence unavailable",
    )
    def test_hostile_rank_resource_scratch_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            exact2 = _build_plan(root)
            audit, receipts, _ = _audit_fixture(root, exact2)
            hostile = copy.deepcopy(audit)
            hostile.pop("receipt_digest")
            hostile["rank_resource_scratch_binding"]["filesystem_type"] = "nfs"
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "stat-f/rank resource",
            ):
                _validate_audit_with_physical_recompute(
                    _sign(hostile), exact2, receipts
                )


class ResourceModuleReuseTests(unittest.TestCase):
    def test_v9_runtime_snapshot_and_candidate_environment_are_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            method_root, resource_path = _materialize_current_specialized_resource(
                Path(temporary)
            )
            module_name = generator.RESOURCE_SPECIALIZED_MODULE_NAME
            previous_module = sys.modules.pop(module_name, None)
            previous_cache = generator._RESOURCE_MODULE_CACHE
            previous_primitives = generator._RESOURCE_MODULE_PRIMITIVES
            generator._RESOURCE_MODULE_CACHE = None
            generator._RESOURCE_MODULE_PRIMITIVES = None
            try:
                with mock.patch.object(generator, "METHOD_ROOT", method_root):
                    loaded = generator.load_resource_contract(resource_path)
                self.assertEqual(
                    loaded.COMPILE_SMOKE_SCHEMA,
                    "bernini-generic-action-fit40-compile-smoke-v10",
                )
                scratch = Path(temporary) / "rank-scratch"
                scratch.mkdir()
                required_environment = {
                    "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
                    "NATIVE_V_AXIS_LOAD_LOCK": str(scratch / "renderer.lock"),
                    "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED": "1",
                    "GADP_HOST_MEMORY_SAMPLE_JOURNAL": str(
                        scratch / "monitor.bin"
                    ),
                    "GADP_HOST_MEMORY_MONITOR_START_RECEIPT": str(
                        scratch / "monitor-start.json"
                    ),
                    "GADP_HOST_MEMORY_MONITOR_PID": "101",
                    "GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID": "100",
                    "UNLISTED_CALLER_SECRET": "must-not-propagate",
                }
                runtime = {
                    "host_cgroup_sampled_memory_monitor": {
                        "slurm_job_id": "136140",
                        "slurm_step_id": "77",
                    },
                    "node_local_scratch": {
                        "path": str(scratch),
                        "filesystem_type": "xfs",
                    },
                    "verified_release_execution": {
                        "held_fd_frozen_python_execution": {
                            "root_owned_bootstrap": {
                                "path": "/usr/bin/python3.10",
                                "sha256": (
                                    "11dde438e1a636073e79c81d4c2543708"
                                    "cc0a2922e7c42c38b1b588e17545f96"
                                ),
                            },
                            "frozen_target": {
                                "path": "/frozen/python3.12",
                                "sha256": loaded.FROZEN_PYTHON_SHA256,
                            },
                        },
                        "manifest": {
                            "path": "/shared/source.manifest.json",
                            "sha256": "1" * 64,
                        },
                        "runner": {
                            "path": "/shared/release-builder.py",
                            "sha256": "2" * 64,
                        },
                        "torchrun_runtime_snapshot": {
                            "path": "/frozen/site-packages/torch/distributed/run.py",
                            "sha256": "3" * 64,
                            "site_packages": "/frozen/site-packages",
                        },
                    },
                    "rank_cache_wrapper": {"sha256": "4" * 64},
                }
                with mock.patch.dict(
                    loaded.os.environ, required_environment, clear=True
                ):
                    environment = loaded._candidate_environment(
                        expected_visible="0,1,2,3",
                        python=Path("/frozen/python3.12"),
                        scratch=scratch,
                        cache_token="candidate-cache-token",
                        runtime=runtime,
                    )
                expected_names = {
                    "PATH",
                    "LC_ALL",
                    "LANG",
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONNOUSERSITE",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                    "TOKENIZERS_PARALLELISM",
                    "MODELING_BACKEND",
                    "ROCR_VISIBLE_DEVICES",
                    "TMPDIR",
                    "SLURM_JOB_ID",
                    "SLURM_STEP_ID",
                    "GADP_NODE_LOCAL_SCRATCH",
                    "GADP_NODE_LOCAL_SCRATCH_FSTYPE",
                    "GADP_RANK_CACHE_TOKEN",
                    "GADP_RANK_PYTHON_BIN",
                    "GADP_METHOD_ROOT",
                    "F13_METHOD_MANIFEST",
                    "F13_METHOD_MANIFEST_SHA256",
                    "F13_VERIFIED_RUNNER_PATH",
                    "F13_VERIFIED_RUNNER_SHA256",
                    "F13_RANK_WRAPPER_SHA256",
                    "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED",
                    "NATIVE_V_AXIS_LOAD_LOCK",
                    "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED",
                    "GADP_HOST_MEMORY_SAMPLE_JOURNAL",
                    "GADP_HOST_MEMORY_MONITOR_START_RECEIPT",
                    "GADP_HOST_MEMORY_MONITOR_PID",
                    "GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID",
                }
                self.assertEqual(set(environment), expected_names)
                self.assertNotIn("UNLISTED_CALLER_SECRET", environment)
                self.assertEqual(environment["PATH"], loaded.SAFE_RUNTIME_PATH)
                self.assertEqual(environment["ROCR_VISIBLE_DEVICES"], "0,1,2,3")
                for hostile_name in ("BASH_ENV", "BASH_FUNC_hostile%%"):
                    hostile_environment = {
                        **required_environment,
                        hostile_name: "hostile",
                    }
                    with self.subTest(hostile_name=hostile_name), mock.patch.dict(
                        loaded.os.environ, hostile_environment, clear=True
                    ), self.assertRaisesRegex(
                        loaded.Reserve4GenerationError,
                        "shell-startup environment presence is forbidden",
                    ):
                        loaded._candidate_environment(
                            expected_visible="0,1,2,3",
                            python=Path("/frozen/python3.12"),
                            scratch=scratch,
                            cache_token="candidate-cache-token",
                            runtime=runtime,
                        )
            finally:
                sys.modules.pop(module_name, None)
                generator._RESOURCE_MODULE_CACHE = previous_cache
                generator._RESOURCE_MODULE_PRIMITIVES = previous_primitives
                if previous_module is not None:
                    sys.modules[module_name] = previous_module

    def test_resource_and_rank_use_root_bootstrap_not_same_uid_python_exec(
        self,
    ) -> None:
        resource_path = (
            METHOD_ROOT / "tools/reserve4_fixed_generation_sp4_v1.py"
        ).resolve(strict=True)
        module_name = "_box_exp_013_r6_resource_bootstrap_contract_test"
        specification = importlib.util.spec_from_file_location(
            module_name, resource_path
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        loaded = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = loaded
        try:
            specification.loader.exec_module(loaded)  # type: ignore[union-attr]
            self.assertEqual(
                loaded.ROOT_BOOTSTRAP_PYTHON_PATH,
                Path("/usr/bin/python3.10"),
            )
            self.assertEqual(
                loaded.ROOT_BOOTSTRAP_PYTHON_SHA256,
                "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96",
            )
            self.assertEqual(loaded.ROOT_BOOTSTRAP_PYTHON_SIZE, 5_937_800)
            self.assertEqual(
                loaded.HOST_CGROUP_MEMORY_MONITOR_START_SCHEMA,
                "bernini-generic-action-fit40-host-cgroup-memory-monitor-start-v3",
            )
            self.assertEqual(
                loaded.HOST_CGROUP_MEMORY_GATE_SCHEMA,
                "bernini-generic-action-fit40-host-cgroup-sampled-memory-gate-v4",
            )
            frozen_python = loaded.FROZEN_PYTHON_PATH
            runner_path = Path("/released/release-builder.py")
            runtime = {
                "verified_release_execution": {
                    "torchrun_runtime_snapshot": {
                        "path": "/frozen/torch/distributed/run.py",
                        "sha256": "3" * 64,
                        "site_packages": "/frozen/site-packages",
                    },
                    "runner": {
                        "path": str(runner_path),
                        "sha256": "2" * 64,
                    },
                    "held_fd_frozen_python_execution": {
                        "root_owned_bootstrap": {
                            "path": str(loaded.ROOT_BOOTSTRAP_PYTHON_PATH),
                            "sha256": loaded.ROOT_BOOTSTRAP_PYTHON_SHA256,
                        },
                        "frozen_target": {
                            "path": str(frozen_python),
                            "sha256": loaded.FROZEN_PYTHON_SHA256,
                        },
                    },
                }
            }
            command = loaded._candidate_command(
                SimpleNamespace(
                    master_port=38142,
                    bernini_root="/released/Bernini",
                    veomni_root="/released/VeOmni",
                    checkpoint="/released/checkpoint",
                    checkpoint_content_manifest="/released/checkpoint.json",
                    method_source_revision="1" * 40,
                    method_source_archive_sha256="4" * 64,
                ),
                task={
                    "candidate_spec_path": "/released/candidate.json",
                    "root_spec_sha256": "5" * 64,
                },
                candidate_output=Path("/scratch/candidate"),
                python=frozen_python,
                worker=Path("/released/worker.py"),
                rank_exec=Path("/released/rank.sh"),
                rank_exec_source="verified rank wrapper bytes",
                torchrun_source="verified torchrun bytes",
                runtime=runtime,
            )
            self.assertEqual(
                command[:12],
                [
                    str(loaded.ROOT_BOOTSTRAP_PYTHON_PATH),
                    "-I",
                    "-S",
                    "-s",
                    "-B",
                    "-c",
                    loaded._VERIFIED_RUNNER_BOOTSTRAP,
                    str(runner_path),
                    "2" * 64,
                    "held-fd-exec-frozen-python",
                    "--start-gate-stdin",
                    "--",
                ],
            )
            self.assertNotIn(str(frozen_python), command)
            self.assertEqual(command.count("--checkpoint"), 1)
            self.assertEqual(command.count("--checkpoint-content-manifest"), 1)
            self.assertIn("/bin/bash", command)
            self.assertIn("--no_python", command)
            for caller in (
                loaded.run_compile_smoke_sp4,
                loaded.run_sp4,
            ):
                caller_source = inspect.getsource(caller)
                self.assertIn(
                    "_run_candidate_under_live_monitor(command, environment)",
                    caller_source,
                )
                self.assertNotIn("subprocess.run([str(python)", caller_source)
            monitor_runner_source = inspect.getsource(
                loaded._run_candidate_under_live_monitor
            )
            self.assertIn("process = subprocess.Popen(\n        list(command)", monitor_runner_source)
            self.assertIn("stdin=subprocess.PIPE", monitor_runner_source)
            self.assertIn("start_new_session=True", monitor_runner_source)
            resource_parser = loaded.build_parser()
            resource_commands = next(
                action
                for action in resource_parser._actions
                if "host-memory-monitor"
                in (getattr(action, "choices", {}) or {})
            )
            monitor_parser = resource_commands.choices["host-memory-monitor"]
            monitor_options = {
                option: action
                for action in monitor_parser._actions
                for option in action.option_strings
            }
            self.assertIn("--stop-fd", monitor_options)
            self.assertTrue(monitor_options["--stop-fd"].required)
            self.assertNotIn("--stop-path", monitor_options)
            stop_validator_source = inspect.getsource(
                loaded._validate_monitor_stop_capability
            )
            self.assertIn('"anonymous-pipe-read-end"', stop_validator_source)
            self.assertIn(
                'value.get("named_filesystem_stop_token") is False',
                stop_validator_source,
            )
        finally:
            sys.modules.pop(module_name, None)

        rank_path = (
            METHOD_ROOT / "scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
        ).resolve(strict=True)
        rank_source = rank_path.read_text(encoding="utf-8")
        self.assertIn(
            "readonly root_bootstrap_python=/usr/bin/python3.10",
            rank_source,
        )
        self.assertNotIn('exec "${python_bin}"', rank_source)
        self.assertEqual(
            rank_source.count('exec "${root_bootstrap_python}"'), 1
        )
        self.assertIn("held-fd-exec-frozen-python --", rank_source)
        self.assertIn("coproc RANK_WORKER", rank_source)
        self.assertNotIn("worker-start.gate", rank_source)
        self.assertIn("IFS= read -r gate_value", rank_source)
        rank_bootstrap = rank_source[
            rank_source.index("readonly verified_runner_bootstrap='")
            + len("readonly verified_runner_bootstrap='") : rank_source.index(
                "'\nspawn_identity_in_progress=true"
            )
        ]
        for required in (
            'getattr(os,"O_NOFOLLOW",0)',
            "fields(a)==fields(b)==fields(c)==fields(n)",
            "x==y",
            "hashlib.sha256(x).hexdigest()==e",
            "exec(compile(x,p,\"exec\",dont_inherit=True),g)",
        ):
            self.assertIn(required, rank_bootstrap)

        held_exec_source = inspect.getsource(release.held_fd_exec_frozen_python)
        stable_fd_source = inspect.getsource(release._stable_executable_fd)
        self.assertIn('os.readlink("/proc/self/exe")', held_exec_source)
        self.assertIn("ROOT_BOOTSTRAP_PYTHON_SHA256", held_exec_source)
        self.assertIn("expected_uid=0", held_exec_source)
        self.assertIn("expected_gid=0", held_exec_source)
        self.assertIn("target_descriptor = _stable_executable_fd(", held_exec_source)
        self.assertIn("expected_uid=2012", held_exec_source)
        self.assertIn("expected_gid=2000", held_exec_source)
        self.assertIn("os.execve(\n            target_descriptor", held_exec_source)
        self.assertNotIn("os.execve(\n            FROZEN_PYTHON", held_exec_source)
        self.assertEqual(stable_fd_source.count("= hash_pass()"), 2)
        self.assertIn("first_size != expected_size", stable_fd_source)
        self.assertIn("second_size != expected_size", stable_fd_source)
        for metadata_field in (
            "st_dev",
            "st_ino",
            "st_uid",
            "st_gid",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_blocks",
            "st_mtime_ns",
            "st_ctime_ns",
        ):
            self.assertIn(metadata_field, stable_fd_source)
        parser = release.build_parser()
        commands = next(
            action
            for action in parser._actions
            if "held-fd-exec-frozen-python"
            in (getattr(action, "choices", {}) or {})
        )
        held_parser = commands.choices["held-fd-exec-frozen-python"]
        self.assertEqual(
            sum(
                "--start-gate-stdin" in action.option_strings
                for action in held_parser._actions
            ),
            1,
        )

    def test_resource_and_rank_captured_runner_bytes_ignore_late_path_replacement(
        self,
    ) -> None:
        resource_path = (
            METHOD_ROOT / "tools/reserve4_fixed_generation_sp4_v1.py"
        ).resolve(strict=True)
        resource_source = resource_path.read_text(encoding="utf-8")
        tree = ast.parse(resource_source, filename=str(resource_path))
        bootstrap_value = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name)
                and target.id == "_VERIFIED_RUNNER_BOOTSTRAP"
                for target in node.targets
            ):
                continue
            expression = node.value
            if (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Attribute)
                and expression.func.attr == "strip"
            ):
                expression = expression.func.value
            bootstrap_value = ast.literal_eval(expression).strip()
            break
        self.assertIsInstance(bootstrap_value, str)

        rank_source = (
            METHOD_ROOT / "scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
        ).read_text(encoding="utf-8")
        rank_start = rank_source.index("readonly verified_runner_bootstrap='") + len(
            "readonly verified_runner_bootstrap='"
        )
        rank_bootstrap = rank_source[
            rank_start : rank_source.index(
                "'\nspawn_identity_in_progress=true", rank_start
            )
        ]

        safe_source = (
            "from pathlib import Path\n"
            "def main(arguments):\n"
            "    Path(arguments[0]).write_text('captured-safe-runner')\n"
            "    return 0\n"
        ).encode("utf-8")
        real_compile = compile
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary).resolve(strict=True)
            for label, bootstrap in (
                ("resource", bootstrap_value),
                ("rank", rank_bootstrap),
            ):
                with self.subTest(label=label):
                    runner = root / f"{label}-verified-runner.py"
                    held_original = root / f"{label}-held-original.py"
                    safe_output = root / f"{label}-safe-output"
                    malicious_sentinel = root / f"{label}-malicious-sentinel"
                    malicious_source = (
                        "from pathlib import Path\n"
                        f"Path({str(malicious_sentinel)!r}).write_text('executed')\n"
                        "def main(arguments):\n"
                        "    return 99\n"
                    ).encode("utf-8")
                    runner.write_bytes(safe_source)
                    runner.chmod(0o444)
                    expected_sha = hashlib.sha256(safe_source).hexdigest()
                    compiled_bootstrap = real_compile(
                        bootstrap,
                        f"<{label}-verified-runner-bootstrap>",
                        "exec",
                    )
                    replacement_count = 0

                    def compile_captured(
                        source_value, filename, mode, *args, **kwargs
                    ):
                        nonlocal replacement_count
                        self.assertEqual(filename, str(runner))
                        self.assertEqual(source_value, safe_source)
                        replacement_count += 1
                        runner.rename(held_original)
                        runner.write_bytes(malicious_source)
                        runner.chmod(0o444)
                        self.assertNotEqual(
                            runner.stat().st_ino, held_original.stat().st_ino
                        )
                        return real_compile(
                            source_value, filename, mode, *args, **kwargs
                        )

                    with mock.patch.object(
                        sys,
                        "argv",
                        [
                            "-c",
                            str(runner),
                            expected_sha,
                            str(safe_output),
                        ],
                    ), mock.patch(
                        "builtins.compile", side_effect=compile_captured
                    ), self.assertRaises(SystemExit) as raised:
                        exec(
                            compiled_bootstrap,
                            {"__builtins__": __builtins__},
                        )
                    self.assertEqual(raised.exception.code, 0)
                    self.assertEqual(replacement_count, 1)
                    self.assertEqual(runner.read_bytes(), malicious_source)
                    self.assertEqual(
                        safe_output.read_text(encoding="utf-8"),
                        "captured-safe-runner",
                    )
                    self.assertFalse(malicious_sentinel.exists())

            frozen_target = root / "same-uid-frozen-python"
            held_target = root / "held-frozen-python-inode"
            frozen_safe = b"validated executable bytes"
            frozen_sentinel = b"replacement executable sentinel"
            frozen_target.write_bytes(frozen_safe)
            frozen_target.chmod(0o755)
            target_metadata = frozen_target.stat()
            descriptor = release._stable_executable_fd(
                frozen_target,
                expected_sha256=hashlib.sha256(frozen_safe).hexdigest(),
                expected_size=len(frozen_safe),
                expected_uid=target_metadata.st_uid,
                expected_gid=target_metadata.st_gid,
            )
            try:
                frozen_target.rename(held_target)
                frozen_target.write_bytes(frozen_sentinel)
                frozen_target.chmod(0o755)
                self.assertNotEqual(
                    frozen_target.stat().st_ino,
                    held_target.stat().st_ino,
                )
                os.lseek(descriptor, 0, os.SEEK_SET)
                self.assertEqual(os.read(descriptor, 1024), frozen_safe)
                self.assertEqual(frozen_target.read_bytes(), frozen_sentinel)
            finally:
                os.close(descriptor)

    def test_candidate_anonymous_gate_monitor_failure_and_bounded_kill(self) -> None:
        resource_path = (
            METHOD_ROOT / "tools/reserve4_fixed_generation_sp4_v1.py"
        ).resolve(strict=True)
        module_name = "_box_exp_013_r6_candidate_monitor_contract_test"
        specification = importlib.util.spec_from_file_location(
            module_name, resource_path
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        loaded = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = loaded
        try:
            specification.loader.exec_module(loaded)  # type: ignore[union-attr]
            candidate_stdin = mock.Mock()
            process = mock.Mock()
            process.pid = 4242
            process.stdin = candidate_stdin
            process.poll.return_value = None
            monitor_start = {
                "monitor_pid": 501,
                "monitor_proc_start_ticks": 601,
                "supervisor_pid": 502,
                "supervisor_proc_start_ticks": 602,
            }
            terminate = mock.Mock()
            with mock.patch.object(
                loaded.subprocess, "Popen", return_value=process
            ) as popen, mock.patch.object(
                loaded,
                "assert_live_host_cgroup_memory_monitor",
                return_value=monitor_start,
            ), mock.patch.object(
                loaded, "_process_start_ticks", return_value=700
            ), mock.patch.object(
                loaded.os, "getpgid", return_value=4242
            ), mock.patch.object(
                loaded,
                "_owned_candidate_process_observation",
                return_value="live",
            ), mock.patch.object(
                loaded,
                "_process_identity_is_live",
                side_effect=[True, True, False, True],
            ), mock.patch.object(
                loaded, "_assert_fresh_live_journal_tail"
            ), mock.patch.object(
                loaded,
                "_terminate_owned_candidate_process_group",
                terminate,
            ), self.assertRaisesRegex(
                loaded.Reserve4GenerationError,
                "monitor/supervisor died",
            ):
                loaded._run_candidate_under_live_monitor(
                    ["/usr/bin/python3.10", "held-runner"],
                    {"PATH": "/usr/bin:/bin"},
                )
            popen.assert_called_once_with(
                ["/usr/bin/python3.10", "held-runner"],
                env={"PATH": "/usr/bin:/bin"},
                stdin=loaded.subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
            )
            candidate_stdin.write.assert_called_once_with(b"go\n")
            candidate_stdin.flush.assert_called_once_with()
            candidate_stdin.close.assert_called_once_with()
            terminate.assert_called_once_with(process, 700)

            process = mock.Mock()
            process.pid = 4242
            process.poll.return_value = None
            process.wait.side_effect = [
                loaded.subprocess.TimeoutExpired("candidate", 0.1)
                for _ in range(50)
            ] + [0]
            killpg = mock.Mock()
            with mock.patch.object(
                loaded,
                "_owned_candidate_process_observation",
                return_value="live",
            ), mock.patch.object(
                loaded.os, "getpgid", return_value=4242
            ), mock.patch.object(loaded.os, "killpg", killpg):
                loaded._terminate_owned_candidate_process_group(process, 700)
            self.assertEqual(
                killpg.call_args_list,
                [
                    mock.call(4242, loaded.signal.SIGTERM),
                    mock.call(4242, loaded.signal.SIGKILL),
                ],
            )
            self.assertEqual(process.wait.call_count, 51)

            killpg.reset_mock()
            process.poll.return_value = None
            with mock.patch.object(
                loaded,
                "_owned_candidate_process_observation",
                return_value="unknown",
            ), mock.patch.object(
                loaded.os, "getpgid", return_value=4242
            ), mock.patch.object(
                loaded.os, "killpg", killpg
            ), self.assertRaisesRegex(
                loaded.Reserve4GenerationError,
                "identity became ambiguous before termination",
            ):
                loaded._terminate_owned_candidate_process_group(process, 700)
            killpg.assert_not_called()
        finally:
            sys.modules.pop(module_name, None)

    def test_real_sealed_resource_loads_once_and_safe_cache_reuses(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            method_root, resource_path = _materialize_current_specialized_resource(
                Path(temporary)
            )
            module_name = controller.TERMINAL_RESOURCE_MODULE_NAME
            previous_module = sys.modules.pop(module_name, None)
            previous_cache = controller._RESOURCE_MODULE_CACHE
            previous_primitives = controller._RESOURCE_MODULE_PRIMITIVES
            previous_generator_cache = generator._RESOURCE_MODULE_CACHE
            previous_generator_primitives = generator._RESOURCE_MODULE_PRIMITIVES
            controller._RESOURCE_MODULE_CACHE = None
            controller._RESOURCE_MODULE_PRIMITIVES = None
            generator._RESOURCE_MODULE_CACHE = None
            generator._RESOURCE_MODULE_PRIMITIVES = None
            try:
                with mock.patch.object(generator, "METHOD_ROOT", method_root):
                    forged = ModuleType(module_name)
                    forged.__file__ = str(resource_path)
                    forged.__spec__ = importlib.util.spec_from_file_location(
                        module_name, resource_path
                    )
                    sys.modules[module_name] = forged
                    with self.assertRaisesRegex(
                        controller.ArmsIncompleteExact2ControllerError,
                        "untrusted.*preloaded",
                    ):
                        controller._load_or_reuse_resource_contract(
                            resource_path,
                            controller.TERMINAL_RESOURCE_CONTRACT_SHA256,
                        )
                    sys.modules.pop(module_name, None)

                    with self.assertRaisesRegex(
                        controller.ArmsIncompleteExact2ControllerError,
                        "path/SHA-256",
                    ):
                        controller._load_or_reuse_resource_contract(
                            resource_path, "0" * 64
                        )

                    wrong_root = Path(temporary) / "wrong-source"
                    wrong_root.mkdir()
                    wrong_source = wrong_root / resource_path.name
                    shutil.copyfile(resource_path, wrong_source)
                    with self.assertRaisesRegex(
                        controller.ArmsIncompleteExact2ControllerError,
                        "path/SHA-256",
                    ):
                        controller._load_or_reuse_resource_contract(
                            wrong_source,
                            controller.TERMINAL_RESOURCE_CONTRACT_SHA256,
                        )

                    first = controller._load_or_reuse_resource_contract(
                        resource_path,
                        controller.TERMINAL_RESOURCE_CONTRACT_SHA256,
                    )
                    second = controller._load_or_reuse_resource_contract(
                        resource_path,
                        controller.TERMINAL_RESOURCE_CONTRACT_SHA256,
                        first,
                    )
                    self.assertIs(first, second)
                    self.assertIs(sys.modules[module_name], first)

                    replacement = ModuleType(module_name)
                    replacement.__file__ = str(resource_path)
                    replacement.__spec__ = first.__spec__
                    sys.modules[module_name] = replacement
                    with self.assertRaisesRegex(
                        controller.ArmsIncompleteExact2ControllerError,
                        "registration drifted",
                    ):
                        controller._load_or_reuse_resource_contract(
                            resource_path,
                            controller.TERMINAL_RESOURCE_CONTRACT_SHA256,
                        )
                    sys.modules[module_name] = first

                    original_sample_row = first._sample_row
                    first._sample_row = lambda _row: {}
                    try:
                        with self.assertRaisesRegex(
                            controller.ArmsIncompleteExact2ControllerError,
                            "primitives drifted",
                        ):
                            controller._load_or_reuse_resource_contract(
                                resource_path,
                                controller.TERMINAL_RESOURCE_CONTRACT_SHA256,
                                first,
                            )
                    finally:
                        first._sample_row = original_sample_row
            finally:
                sys.modules.pop(module_name, None)
                controller._RESOURCE_MODULE_CACHE = previous_cache
                controller._RESOURCE_MODULE_PRIMITIVES = previous_primitives
                generator._RESOURCE_MODULE_CACHE = previous_generator_cache
                generator._RESOURCE_MODULE_PRIMITIVES = previous_generator_primitives
                if previous_module is not None:
                    sys.modules[module_name] = previous_module

    @unittest.skipUnless(
        EXTERNAL_KEY.is_file() and EXTERNAL_REVIEW.is_file(),
        "sealed external evidence unavailable",
    )
    def test_real_exact_audit_terminal_completion_reuse_one_module(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            method_root, resource_path = _materialize_current_specialized_resource(
                root
            )

            exact2 = _build_plan(root)
            audit, receipts, _ = _audit_fixture(root, exact2)
            rank_binding = audit["rank_resource_scratch_binding"]
            preflight = _sign(
                {
                    "schema_version": controller.COMPUTE_PREFLIGHT_SCHEMA,
                    "runtime": {
                        "slurm_job_id": "136140",
                        "slurm_step_id": "77",
                    },
                    "scratch_parent": {
                        "path": rank_binding[
                            "preflight_scratch_parent_path"
                        ],
                        "filesystem_type": rank_binding["filesystem_type"],
                    },
                }
            )
            preflight_path, preflight_sha = _write_json(
                root / "real-compute-preflight.json", preflight
            )
            audit_unsigned = dict(audit)
            audit_unsigned.pop("receipt_digest")
            audit_unsigned["compute_preflight"] = {
                "path": str(preflight_path),
                "file_sha256": preflight_sha,
                "receipt_digest": preflight["receipt_digest"],
            }
            audit = _sign(audit_unsigned)
            compute_ref = audit["compute_preflight"]
            task_bind_ref = audit["task_scratch_bind"]
            task_bind = {
                "compute_preflight": compute_ref,
                "scratch_inner": {
                    "path": audit["rank_resource_scratch_binding"][
                        "rank_task_scratch_path"
                    ]
                },
                "receipt_digest": task_bind_ref["receipt_digest"],
            }
            shard_ref = audit["shard_receipt"]
            proofs = list(audit["cross_run_same_gaussian_pair_proofs"])
            receipt_by_candidate = {
                task["candidate_id"]: receipt
                for task, receipt in zip(exact2["admission_tasks"], receipts)
            }
            proof_by_seed = {row["seed"]: row for row in proofs}
            terminal_args = _real_frozen_terminal_args(
                root,
                preflight_path=preflight_path,
                preflight_sha=preflight_sha,
                resource_path=resource_path,
            )

            module_name = controller.TERMINAL_RESOURCE_MODULE_NAME
            previous_module = sys.modules.pop(module_name, None)
            previous_cache = controller._RESOURCE_MODULE_CACHE
            previous_primitives = controller._RESOURCE_MODULE_PRIMITIVES
            previous_generator_cache = generator._RESOURCE_MODULE_CACHE
            previous_generator_primitives = generator._RESOURCE_MODULE_PRIMITIVES
            controller._RESOURCE_MODULE_CACHE = None
            controller._RESOURCE_MODULE_PRIMITIVES = None
            generator._RESOURCE_MODULE_CACHE = None
            generator._RESOURCE_MODULE_PRIMITIVES = None
            try:
                with mock.patch.object(
                    generator, "METHOD_ROOT", method_root
                ), mock.patch.object(
                    plan,
                    "load_plan",
                    return_value=(
                        exact2,
                        Path(exact2["_path"]),
                        exact2["_file_sha256"],
                    ),
                ), mock.patch.object(
                    controller,
                    "validate_compute_preflight",
                    side_effect=lambda value, **_: value,
                ), mock.patch.object(
                    generator,
                    "load_task_scratch_bind",
                    return_value=(
                        task_bind,
                        Path(task_bind_ref["path"]),
                        task_bind_ref["file_sha256"],
                    ),
                ), mock.patch.object(
                    generator,
                    "replay_task_scratch_bind_physical",
                    return_value=Path(task_bind["scratch_inner"]["path"]),
                ), mock.patch.object(
                    generator,
                    "validate_shard_scratch_binding",
                    return_value=(compute_ref, rank_binding, shard_ref),
                ), mock.patch.object(
                    generator,
                    "_validate_candidate_receipt",
                    side_effect=lambda _resource, task, _path: (
                        receipt_by_candidate[task["candidate_id"]]
                    ),
                ), mock.patch.object(
                    generator,
                    "_validate_external_action_artifacts",
                    side_effect=lambda _action, _resource: ({}, {}),
                ), mock.patch.object(
                    generator,
                    "cross_run_same_gaussian_proof",
                    side_effect=lambda **kwargs: proof_by_seed[
                        kwargs["task"]["seed"]
                    ],
                ), mock.patch.object(
                    controller,
                    "probe_exact81_25fps",
                    return_value={"frame_count": 81, "fps": 25},
                ), mock.patch.object(
                    controller,
                    "validate_ffprobe",
                    return_value=Path("/bin/true"),
                ), mock.patch.object(
                    controller.os, "getegid", return_value=os.stat(root).st_gid
                ):
                    controller.validate_exact2_audit(
                        audit, exact2, Path("/bin/true")
                    )
                    loaded = controller._RESOURCE_MODULE_CACHE
                    self.assertIsNotNone(loaded)
                    self.assertIs(sys.modules[module_name], loaded)

                    terminal = controller.seal_terminal_host_gate(terminal_args)
                    self.assertIs(controller._RESOURCE_MODULE_CACHE, loaded)
                    terminal_path = Path(terminal_args.output)
                    terminal_sha = controller.file_sha256(terminal_path)

                    generation_path, generation_sha = _write_json(
                        root / "real-generation-audit.json", audit
                    )
                    review = _sign(
                        {
                            "reviewer_receipt_id": "4" * 32,
                            "packet_id": "5" * 32,
                            "blind_review_manifest": {
                                "path": str(root / "blind-review-manifest.json"),
                                "file_sha256": "b" * 64,
                                "receipt_digest": "c" * 64,
                            },
                            "sealed_key": {
                                "path": str(root / "blind-review-key.json"),
                                "file_sha256": "d" * 64,
                                "receipt_digest": "e" * 64,
                            },
                        }
                    )
                    review_path, review_sha = _write_json(
                        root / "real-review.json", review
                    )
                    controller_plan = {"plan_digest": "6" * 64}
                    controller_plan_path = root / "controller-plan.json"
                    controller_plan_path.write_bytes(b"{}\n")
                    generation_ref = {
                        "path": str(generation_path),
                        "file_sha256": generation_sha,
                        "receipt_digest": audit["receipt_digest"],
                    }
                    terminal_ref = {
                        "path": str(terminal_path),
                        "file_sha256": terminal_sha,
                        "receipt_digest": terminal["receipt_digest"],
                    }
                    runtime = {
                        "slurm_job_id": "136140",
                        "slurm_step_id": "77",
                        "hostname": "auh7-1b-gpu-215",
                        "sole_numbered_compute_child_required": True,
                    }
                    attestation = _sign(
                        {
                            "runtime": runtime,
                            "scratch_prepare": {
                                "path": str(root / "scratch-prepare.json"),
                                "file_sha256": "9" * 64,
                                "receipt_digest": "a" * 64,
                            },
                            "compute_preflight": audit["compute_preflight"],
                            "task_scratch_bind": audit["task_scratch_bind"],
                            "generation_audit": generation_ref,
                            "terminal_host_gate": terminal_ref,
                        }
                    )
                    attestation_path, attestation_sha = _write_json(
                        root / "child-terminal-physical-attestation.json",
                        attestation,
                    )
                    retained = _sign(
                        {
                            "physical_attestation": {
                                "path": str(attestation_path),
                                "file_sha256": attestation_sha,
                                "receipt_digest": attestation["receipt_digest"],
                            }
                        }
                    )
                    retained_path, retained_sha = _write_json(
                        root / "child-scratch-retained-terminal.json", retained
                    )
                    parent_status = _sign(
                        {
                            "controller_plan": {
                                "path": str(controller_plan_path),
                                "file_sha256": "7" * 64,
                                "plan_digest": controller_plan["plan_digest"],
                            },
                            "generation_audit": generation_ref,
                            "terminal_host_gate": terminal_ref,
                            "physical_attestation": {
                                "path": str(attestation_path),
                                "file_sha256": attestation_sha,
                                "receipt_digest": attestation["receipt_digest"],
                            },
                            "scratch_retained_terminal": {
                                "path": str(retained_path),
                                "file_sha256": retained_sha,
                                "receipt_digest": retained["receipt_digest"],
                            },
                            "blind_review_manifest": review[
                                "blind_review_manifest"
                            ],
                            "blind_review_key": review["sealed_key"],
                            "runtime": runtime,
                        }
                    )
                    parent_status_path, parent_status_sha = _write_json(
                        root / "parent-generation.status", parent_status
                    )
                    completion_args = SimpleNamespace(
                        controller_plan=str(controller_plan_path),
                        expected_controller_plan_sha256="7" * 64,
                        generation_audit=str(generation_path),
                        expected_generation_audit_sha256=generation_sha,
                        review_admission=str(review_path),
                        expected_review_admission_sha256=review_sha,
                        terminal_host_gate=str(terminal_path),
                        expected_terminal_host_gate_sha256=terminal_sha,
                        child_terminal_physical_attestation=str(attestation_path),
                        expected_child_terminal_physical_attestation_sha256=(
                            attestation_sha
                        ),
                        child_scratch_retained_terminal=str(retained_path),
                        expected_child_scratch_retained_terminal_sha256=retained_sha,
                        parent_generation_status=str(parent_status_path),
                        expected_parent_generation_status_sha256=(
                            parent_status_sha
                        ),
                        ffprobe_bin="/bin/true",
                        expected_ffprobe_sha256="8" * 64,
                        output=str(root / "real-completion.json"),
                    )
                    with ExitStack() as stack:
                        stack.enter_context(
                            mock.patch.object(
                                controller,
                                "load_controller_plan",
                                return_value=(
                                    controller_plan,
                                    controller_plan_path,
                                    "7" * 64,
                                    exact2,
                                ),
                            )
                        )
                        for name in (
                            "_validate_child_terminal_attestation_shape",
                            "_validate_child_scratch_retained_terminal_shape",
                            "validate_child_scratch_retained_terminal",
                        ):
                            stack.enter_context(mock.patch.object(controller, name))
                        stack.enter_context(
                            mock.patch.object(
                                controller,
                                "_validate_exact2_audit_postretention_attested",
                                return_value=audit,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                controller,
                                "validate_review_admission",
                                return_value=review,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                controller,
                                "_validate_terminal_host_gate_postretention_attested",
                                return_value=terminal,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                controller,
                                "_load_parent_generation_status",
                                return_value=(
                                    parent_status,
                                    parent_status_path,
                                    parent_status_sha,
                                    {},
                                    {},
                                ),
                            )
                        )
                        completed = controller.seal_completion(completion_args)
                    self.assertIs(controller._RESOURCE_MODULE_CACHE, loaded)
                    self.assertIs(sys.modules[module_name], loaded)
                    self.assertEqual(
                        completed["terminal_host_gate"]["receipt_digest"],
                        terminal["receipt_digest"],
                    )
            finally:
                sys.modules.pop(module_name, None)
                controller._RESOURCE_MODULE_CACHE = previous_cache
                controller._RESOURCE_MODULE_PRIMITIVES = previous_primitives
                generator._RESOURCE_MODULE_CACHE = previous_generator_cache
                generator._RESOURCE_MODULE_PRIMITIVES = previous_generator_primitives
                if previous_module is not None:
                    sys.modules[module_name] = previous_module


class TerminalPhysicalDerivationTests(unittest.TestCase):
    def test_terminal_seal_validate_and_hostile_exact_closure(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            resource, args = _terminal_physical_fixture(root)
            with mock.patch.object(
                controller, "_load_or_reuse_resource_contract", return_value=resource
            ), mock.patch.object(
                controller, "validate_compute_preflight", side_effect=lambda value, **_: value
            ), mock.patch.object(
                controller,
                "TERMINAL_RESOURCE_CONTRACT_SHA256",
                args.expected_resource_contract_sha256,
            ), mock.patch.object(
                controller.os, "getegid", return_value=os.stat(root).st_gid
            ):
                value = controller.seal_terminal_host_gate(args)
                self.assertEqual(
                    controller.validate_terminal_host_gate(value), value
                )
                self.assertEqual(value["sample_journal"]["nlink"], 1)
                self.assertEqual(value["sample_journal"]["mode_octal"], "0400")
                self.assertEqual(value["sample_journal"]["sample_count"], 3)
                self.assertEqual(
                    value["sampling"]["maximum_observed_gap_ns"], 10_000_000
                )

                hostiles = []
                extra = copy.deepcopy(value)
                extra["extra"] = True
                hostiles.append(_resign(extra))
                missing = copy.deepcopy(value)
                missing.pop("memory")
                hostiles.append(_resign(missing))
                nested_extra = copy.deepcopy(value)
                nested_extra["compute_preflight"]["extra"] = True
                hostiles.append(_resign(nested_extra))
                nested_missing = copy.deepcopy(value)
                nested_missing["sample_journal"].pop("file_sha256")
                hostiles.append(_resign(nested_missing))
                negative = copy.deepcopy(value)
                negative["sample_journal"]["sample_count"] = -1
                hostiles.append(_resign(negative))
                resigned = copy.deepcopy(value)
                resigned["sampling"]["maximum_observed_gap_ns"] = 1
                hostiles.append(_resign(resigned))
                for hostile in hostiles:
                    with self.assertRaises(
                        controller.ArmsIncompleteExact2ControllerError
                    ):
                        controller.validate_terminal_host_gate(hostile)

                journal = Path(value["sample_journal"]["path"])
                journal.chmod(0o600)
                raw = bytearray(journal.read_bytes())
                raw[0] ^= 1
                journal.write_bytes(raw)
                journal.chmod(0o400)
                with self.assertRaises(
                    controller.ArmsIncompleteExact2ControllerError
                ):
                    controller.validate_terminal_host_gate(value)

    def test_terminal_rejects_nonzero_wait_or_live_monitor(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            resource, args = _terminal_physical_fixture(root)
            with mock.patch.object(
                controller, "_load_or_reuse_resource_contract", return_value=resource
            ), mock.patch.object(
                controller, "validate_compute_preflight", side_effect=lambda value, **_: value
            ), mock.patch.object(
                controller,
                "TERMINAL_RESOURCE_CONTRACT_SHA256",
                args.expected_resource_contract_sha256,
            ):
                args.monitor_exit_status = 1
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "exact status zero",
                ):
                    controller.seal_terminal_host_gate(args)
                args.monitor_exit_status = 0
                resource._process_identity_is_live = lambda _pid, _ticks: True
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "monitor-dead",
                ):
                    controller.seal_terminal_host_gate(args)

    def test_terminal_rejects_resigned_cross_step_physical_splice(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            resource, args = _terminal_physical_fixture(root)
            start = json.loads(Path(args.monitor_start_receipt).read_bytes())
            start["slurm_step_id"] = "78"
            start = _resign(start)
            cross_path, cross_sha = _write_json(
                root / "monitor-start-cross-step.json", start
            )
            args.monitor_start_receipt = str(cross_path)
            args.expected_monitor_start_receipt_sha256 = cross_sha
            args.output = str(root / "cross-step-terminal.json")
            with mock.patch.object(
                controller,
                "_load_or_reuse_resource_contract",
                return_value=resource,
            ), mock.patch.object(
                controller,
                "validate_compute_preflight",
                side_effect=lambda value, **_: value,
            ), mock.patch.object(
                controller,
                "TERMINAL_RESOURCE_CONTRACT_SHA256",
                args.expected_resource_contract_sha256,
            ):
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "job-step binding",
                ):
                    controller.seal_terminal_host_gate(args)

    def test_completion_rederives_terminal_and_rejects_resigned_tamper(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            resource, terminal_args = _terminal_physical_fixture(root)
            exact2 = {"plan_digest": "1" * 64}
            controller_plan = {"plan_digest": "2" * 64}
            plan_path = root / "controller-plan.json"
            plan_path.write_bytes(b"{}\n")
            proof_rows = [
                {
                    "seed": seed,
                    "action_incomplete_official_gaussian_tensor_values_byte_equal": True,
                    "physical_artifacts_reopened": True,
                    "physical_safetensors_safe_open_recomputed": True,
                }
                for seed in (2026080821, 2026080921)
            ]
            common_patches = (
                mock.patch.object(
                    controller,
                    "_load_or_reuse_resource_contract",
                    return_value=resource,
                ),
                mock.patch.object(
                    controller,
                    "validate_compute_preflight",
                    side_effect=lambda value, **_: value,
                ),
                mock.patch.object(
                    controller,
                    "load_controller_plan",
                    return_value=(controller_plan, plan_path, "3" * 64, exact2),
                ),
                mock.patch.object(
                    controller, "validate_ffprobe", return_value=Path("/bin/true")
                ),
                mock.patch.object(controller, "validate_exact2_audit"),
                mock.patch.object(controller, "validate_review_admission"),
                mock.patch.object(
                    controller, "_validate_exact2_audit_postretention_attested"
                ),
                mock.patch.object(
                    controller,
                    "_validate_compute_preflight_postretention",
                    side_effect=lambda value, **_: value,
                ),
                mock.patch.object(
                    controller, "_validate_child_terminal_attestation_shape"
                ),
                mock.patch.object(
                    controller, "_validate_child_scratch_retained_terminal_shape"
                ),
                mock.patch.object(
                    controller, "validate_child_scratch_retained_terminal"
                ),
                mock.patch.object(
                    controller,
                    "TERMINAL_RESOURCE_CONTRACT_SHA256",
                    terminal_args.expected_resource_contract_sha256,
                ),
                mock.patch.object(
                    controller.os, "getegid", return_value=os.stat(root).st_gid
                ),
            )
            with ExitStack() as stack:
                for patcher in common_patches:
                    stack.enter_context(patcher)
                terminal = controller.seal_terminal_host_gate(terminal_args)
                terminal_path = Path(terminal_args.output)
                terminal_sha = controller.file_sha256(terminal_path)
                generation = _sign(
                    {
                        "compute_preflight": terminal["compute_preflight"],
                        "rank_resource_scratch_binding": {
                            "preflight_scratch_parent_path": str(
                                (root / "slurm-tmpdir").resolve()
                            ),
                            "rank_task_scratch_path": str(
                                (root / "slurm-tmpdir" / "rank-task").resolve()
                            ),
                            "filesystem_type": "xfs",
                            "source_environment_variable": (
                                "GADP_NODE_LOCAL_SCRATCH_FSTYPE"
                            ),
                            "preflight_stat_f_matches_rank_resource_receipt": True,
                            "compile_smoke_runtime_matches_rank_resource_receipt": True,
                        },
                        "cross_run_same_gaussian_pair_proofs": proof_rows,
                    }
                )
                generation_path, generation_sha = _write_json(
                    root / "generation.json", generation
                )
                review = _sign(
                    {
                        "reviewer_receipt_id": "4" * 32,
                        "packet_id": "5" * 32,
                        "blind_review_manifest": {
                            "path": str(root / "blind-review-manifest.json"),
                            "file_sha256": "7" * 64,
                            "receipt_digest": "8" * 64,
                        },
                        "sealed_key": {
                            "path": str(root / "blind-review-key.json"),
                            "file_sha256": "9" * 64,
                            "receipt_digest": "a" * 64,
                        },
                    }
                )
                review_path, review_sha = _write_json(root / "review.json", review)
                generation_ref = {
                    "path": str(generation_path),
                    "file_sha256": generation_sha,
                    "receipt_digest": generation["receipt_digest"],
                }
                terminal_ref = {
                    "path": str(terminal_path),
                    "file_sha256": terminal_sha,
                    "receipt_digest": terminal["receipt_digest"],
                }
                runtime = {
                    "slurm_job_id": "136140",
                    "slurm_step_id": "77",
                    "hostname": "auh7-1b-gpu-215",
                    "sole_numbered_compute_child_required": True,
                }
                attestation = _sign(
                    {
                        "runtime": runtime,
                        "scratch_prepare": {
                            "path": str(root / "scratch-prepare.json"),
                            "file_sha256": "b" * 64,
                            "receipt_digest": "c" * 64,
                        },
                        "compute_preflight": terminal["compute_preflight"],
                        "task_scratch_bind": {
                            "path": str(root / "task-bind.json"),
                            "file_sha256": "d" * 64,
                            "receipt_digest": "e" * 64,
                        },
                        "generation_audit": generation_ref,
                        "terminal_host_gate": terminal_ref,
                    }
                )
                attestation_path, attestation_sha = _write_json(
                    root / "child-terminal-physical-attestation.json", attestation
                )
                retained = _sign(
                    {
                        "physical_attestation": {
                            "path": str(attestation_path),
                            "file_sha256": attestation_sha,
                            "receipt_digest": attestation["receipt_digest"],
                        }
                    }
                )
                retained_path, retained_sha = _write_json(
                    root / "child-scratch-retained-terminal.json", retained
                )
                parent_status = _sign(
                    {
                        "controller_plan": {
                            "path": str(plan_path),
                            "file_sha256": "3" * 64,
                            "plan_digest": controller_plan["plan_digest"],
                        },
                        "generation_audit": generation_ref,
                        "terminal_host_gate": terminal_ref,
                        "physical_attestation": {
                            "path": str(attestation_path),
                            "file_sha256": attestation_sha,
                            "receipt_digest": attestation["receipt_digest"],
                        },
                        "scratch_retained_terminal": {
                            "path": str(retained_path),
                            "file_sha256": retained_sha,
                            "receipt_digest": retained["receipt_digest"],
                        },
                        "blind_review_manifest": review[
                            "blind_review_manifest"
                        ],
                        "blind_review_key": review["sealed_key"],
                        "runtime": runtime,
                    }
                )
                parent_status_path, parent_status_sha = _write_json(
                    root / "parent-generation.status", parent_status
                )
                args = SimpleNamespace(
                    controller_plan=str(plan_path),
                    expected_controller_plan_sha256="3" * 64,
                    generation_audit=str(generation_path),
                    expected_generation_audit_sha256=generation_sha,
                    review_admission=str(review_path),
                    expected_review_admission_sha256=review_sha,
                    terminal_host_gate=str(terminal_path),
                    expected_terminal_host_gate_sha256=terminal_sha,
                    child_terminal_physical_attestation=str(attestation_path),
                    expected_child_terminal_physical_attestation_sha256=(
                        attestation_sha
                    ),
                    child_scratch_retained_terminal=str(retained_path),
                    expected_child_scratch_retained_terminal_sha256=retained_sha,
                    parent_generation_status=str(parent_status_path),
                    expected_parent_generation_status_sha256=parent_status_sha,
                    ffprobe_bin="/bin/true",
                    expected_ffprobe_sha256="6" * 64,
                    output=str(root / "completion.json"),
                )
                status_loader = stack.enter_context(
                    mock.patch.object(
                        controller,
                        "_load_parent_generation_status",
                        return_value=(
                            parent_status,
                            parent_status_path,
                            parent_status_sha,
                            {},
                            {},
                        ),
                    )
                )
                completed = controller.seal_completion(args)
                self.assertEqual(
                    completed["terminal_host_gate"]["receipt_digest"],
                    terminal["receipt_digest"],
                )

                forged_status = copy.deepcopy(parent_status)
                forged_status["generation_audit"] = {
                    **forged_status["generation_audit"],
                    "receipt_digest": "f" * 64,
                }
                forged_status = _resign(forged_status)
                status_loader.return_value = (
                    forged_status,
                    parent_status_path,
                    parent_status_sha,
                    {},
                    {},
                )
                args.output = str(root / "forged-status-completion.json")
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    "completion attestation/terminal-retention chain differs",
                ):
                    controller.seal_completion(args)
                self.assertFalse(Path(args.output).exists())
                status_loader.return_value = (
                    parent_status,
                    parent_status_path,
                    parent_status_sha,
                    {},
                    {},
                )

                hostile = copy.deepcopy(terminal)
                hostile["sample_journal"]["byte_count"] += 64
                hostile = _resign(hostile)
                terminal_path.chmod(0o600)
                terminal_path.write_bytes(
                    controller.canonical_json_bytes(hostile) + b"\n"
                )
                args.expected_terminal_host_gate_sha256 = controller.file_sha256(
                    terminal_path
                )
                args.output = str(root / "hostile-completion.json")
                with self.assertRaises(
                    controller.ArmsIncompleteExact2ControllerError
                ):
                    controller.seal_completion(args)


class R6ScratchReceiptHostileTests(unittest.TestCase):
    def test_prepare_runtime_mount_nested_omission_extra_and_tamper_fail(self) -> None:
        baseline = _prepare_shape_fixture()
        with mock.patch.object(controller.os, "makedev", return_value=0xFD00):
            self.assertIs(
                controller._validate_child_scratch_prepare_shape(baseline), baseline
            )
            hostiles = []
            runtime_extra = copy.deepcopy(baseline)
            runtime_extra["runtime"]["unexpected"] = False
            hostiles.append(_resign(runtime_extra))
            mount_missing = copy.deepcopy(baseline)
            mount_missing["filesystem"]["mountinfo"].pop("mount_source")
            hostiles.append(_resign(mount_missing))
            mount_extra = copy.deepcopy(baseline)
            mount_extra["filesystem"]["mountinfo"]["unexpected"] = False
            hostiles.append(_resign(mount_extra))
            mount_tamper = copy.deepcopy(baseline)
            mount_tamper["filesystem"]["mountinfo"]["major_minor"] = "0:1"
            hostiles.append(_resign(mount_tamper))
            for hostile in hostiles:
                with self.subTest(keys=sorted(hostile)):
                    with self.assertRaisesRegex(
                        controller.ArmsIncompleteExact2ControllerError,
                        "prepare field/authority closure",
                    ):
                        controller._validate_child_scratch_prepare_shape(hostile)

    def test_controller_created_inner_and_renderer_lock_signed_bind_closes(self) -> None:
        prepare = _prepare_shape_fixture()
        baseline = _task_bind_shape_fixture(prepare)
        with mock.patch.object(controller.os, "makedev", return_value=0xFD00):
            self.assertIs(
                controller._validate_child_task_scratch_bind_shape(baseline),
                baseline,
            )
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            baseline_path, baseline_sha = _write_json(root / "bind.json", baseline)
            with mock.patch.object(generator.os, "makedev", return_value=0xFD00):
                loaded, path, observed = generator.load_task_scratch_bind(
                    baseline_path, baseline_sha
                )
            self.assertEqual((loaded, path, observed), (baseline, baseline_path, baseline_sha))
            for label, mutate in (
                ("lock-extra", lambda value: value["renderer_load_lock"].update({"extra": 0})),
                ("lock-device", lambda value: value["renderer_load_lock"].update({"device": 0})),
                ("old-schema", lambda value: value.update({"schema_version": "old-cleanup-v1"})),
            ):
                hostile = copy.deepcopy(baseline)
                mutate(hostile)
                hostile = _resign(hostile)
                hostile_path, hostile_sha = _write_json(root / f"{label}.json", hostile)
                with self.subTest(label=label), mock.patch.object(
                    generator.os, "makedev", return_value=0xFD00
                ):
                    with self.assertRaises(
                        generator.ArmsIncompleteExact2GenerationError
                    ):
                        generator.load_task_scratch_bind(hostile_path, hostile_sha)

    def test_fully_resigned_fake_plan_prepare_bind_stops_before_resource_loader(self) -> None:
        exact2_path = Path("/shared/exact2-plan.json")
        exact2_sha = "a" * 64
        exact2 = {
            "plan_digest": "b" * 64,
            "admission_tasks": [
                {
                    "candidate_id": f"candidate-{index}",
                    "group_id": "sp4-a",
                    "semantic_branch": "incomplete",
                    "visible_gpus": [0, 1, 2, 3],
                }
                for index in range(2)
            ],
        }
        plan_path = Path("/shared/controller-plan.json")
        plan_sha = "c" * 64
        controller_plan = {
            "plan_digest": "d" * 64,
            "exact2_plan": {
                "path": str(exact2_path),
                "file_sha256": exact2_sha,
                "plan_digest": exact2["plan_digest"],
            },
        }
        correct_plan_ref = {
            "path": str(plan_path),
            "file_sha256": plan_sha,
            "plan_digest": controller_plan["plan_digest"],
        }
        prepare_path = Path("/shared/prepare.json")
        preflight_path = Path("/shared/preflight.json")
        bind_path = Path("/shared/bind.json")
        prepare = _sign(
            {
                # Fully signed, but deliberately spliced to a different plan.
                "controller_plan": {**correct_plan_ref, "plan_digest": "e" * 64}
            }
        )
        prepare_ref = {
            "path": str(prepare_path),
            "file_sha256": "1" * 64,
            "receipt_digest": prepare["receipt_digest"],
        }
        preflight = _sign(
            {
                "controller_plan": correct_plan_ref,
                "scratch_prepare": prepare_ref,
                "runtime": {"slurm_step_id": "77"},
                "scratch_parent": {"path": "/tmp/outer", "device": 1, "inode": 2},
            }
        )
        preflight_ref = {
            "path": str(preflight_path),
            "file_sha256": "2" * 64,
            "receipt_digest": preflight["receipt_digest"],
        }
        bind = _sign(
            {
                "scratch_prepare": prepare_ref,
                "compute_preflight": preflight_ref,
                "runtime": preflight["runtime"],
                "scratch_outer": preflight["scratch_parent"],
            }
        )
        args = SimpleNamespace(
            plan=str(exact2_path),
            expected_plan_sha256=exact2_sha,
            group_id="sp4-a",
            controller_plan=str(plan_path),
            expected_controller_plan_sha256=plan_sha,
            scratch_prepare=str(prepare_path),
            expected_scratch_prepare_sha256=prepare_ref["file_sha256"],
            compute_preflight=str(preflight_path),
            expected_compute_preflight_sha256=preflight_ref["file_sha256"],
            task_scratch_bind=str(bind_path),
            expected_task_scratch_bind_sha256="3" * 64,
        )
        resource_loader = mock.Mock()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "0,1,2,3"}, clear=True))
            stack.enter_context(
                mock.patch.object(
                    generator.plan_contract,
                    "load_plan",
                    return_value=(exact2, exact2_path, exact2_sha),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    controller,
                    "load_controller_plan",
                    return_value=(controller_plan, plan_path, plan_sha, exact2),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    controller,
                    "load_json",
                    side_effect=[
                        (prepare, prepare_path, prepare_ref["file_sha256"]),
                        (preflight, preflight_path, preflight_ref["file_sha256"]),
                        (bind, bind_path, "3" * 64),
                    ],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    controller, "_replay_child_scratch_prepare_physical"
                )
            )
            stack.enter_context(mock.patch.object(controller, "validate_compute_preflight"))
            stack.enter_context(
                mock.patch.object(controller, "_replay_child_task_scratch_bind_physical")
            )
            primitives = (
                controller.load_controller_plan,
                controller.load_json,
                controller._replay_child_scratch_prepare_physical,
                controller.validate_compute_preflight,
                controller._replay_child_task_scratch_bind_physical,
            )
            stack.enter_context(
                mock.patch.object(
                    generator, "_STRICT_CONTROLLER_CACHE", (controller, primitives)
                )
            )
            stack.enter_context(
                mock.patch.object(
                    generator,
                    "STRICT_CONTROLLER_SHA256",
                    generator.file_sha256(Path(controller.__file__)),
                )
            )
            stack.enter_context(
                mock.patch.object(generator, "load_resource_contract", resource_loader)
            )
            with self.assertRaisesRegex(
                generator.ArmsIncompleteExact2GenerationError,
                "controller/plan/prepare/compute/task chain differs",
            ):
                generator.run_shard(args)
        resource_loader.assert_not_called()

    def test_child_terminal_attestation_executes_reference_gate_and_writes(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary).resolve(strict=True)
            controller_plan_path = root / "controller-plan.json"
            controller_plan_path.write_bytes(b"{}\n")
            controller_plan_sha = "1" * 64
            controller_plan = {"plan_digest": "2" * 64}
            controller_ref = {
                "path": str(controller_plan_path),
                "file_sha256": controller_plan_sha,
                "plan_digest": controller_plan["plan_digest"],
            }
            runtime = {
                "slurm_job_id": "136140",
                "slurm_step_id": "77",
                "hostname": "auh7-1b-gpu-215",
                "sole_numbered_compute_child_required": True,
            }

            def write(name: str, value: dict) -> tuple[Path, str, dict]:
                path, sha = _write_json(root / f"{name}.json", value)
                return path, sha, {
                    "path": str(path),
                    "file_sha256": sha,
                    "receipt_digest": value["receipt_digest"],
                }

            prepare = _sign(
                {
                    "controller_plan": controller_ref,
                    "runtime": runtime,
                    "scratch_root": {
                        "path": "/tmp/BOX-EXP-013-r6-136140-77"
                    },
                }
            )
            prepare_path, prepare_sha, prepare_ref = write("prepare", prepare)
            preflight = _sign(
                {
                    "runtime": runtime,
                    "scratch_prepare": prepare_ref,
                }
            )
            preflight_path, preflight_sha, preflight_ref = write(
                "preflight", preflight
            )
            scratch_inner = {
                "path": (
                    "/tmp/BOX-EXP-013-r6-136140-77/"
                    "arms-incomplete-exact2-136140-77.fixture"
                ),
                "inode": 23,
            }
            renderer_lock = {
                "path": f"{scratch_inner['path']}/renderer-load.lock"
            }
            task_bind = _sign(
                {
                    "runtime": runtime,
                    "scratch_prepare": prepare_ref,
                    "compute_preflight": preflight_ref,
                    "scratch_inner": scratch_inner,
                    "renderer_load_lock": renderer_lock,
                }
            )
            task_bind_path, task_bind_sha, task_bind_ref = write(
                "task-bind", task_bind
            )
            generation = _sign(
                {
                    "compute_preflight": preflight_ref,
                    "task_scratch_bind": task_bind_ref,
                }
            )
            generation_path, generation_sha, _ = write(
                "generation", generation
            )
            terminal = _sign({"compute_preflight": preflight_ref})
            terminal_path, terminal_sha, _ = write("terminal", terminal)
            manifest = _sign({"packet_id": "3" * 32})
            manifest_path, manifest_sha, _ = write("manifest", manifest)
            key = _sign({"packet_id": manifest["packet_id"]})
            key_path, key_sha, _ = write("key", key)
            output = root / "physical-attestation.json"
            args = SimpleNamespace(
                controller_plan=str(controller_plan_path),
                expected_controller_plan_sha256=controller_plan_sha,
                scratch_prepare=str(prepare_path),
                expected_scratch_prepare_sha256=prepare_sha,
                compute_preflight=str(preflight_path),
                expected_compute_preflight_sha256=preflight_sha,
                task_scratch_bind=str(task_bind_path),
                expected_task_scratch_bind_sha256=task_bind_sha,
                generation_audit=str(generation_path),
                expected_generation_audit_sha256=generation_sha,
                terminal_host_gate=str(terminal_path),
                expected_terminal_host_gate_sha256=terminal_sha,
                blind_review_manifest=str(manifest_path),
                expected_blind_review_manifest_sha256=manifest_sha,
                blind_review_key=str(key_path),
                expected_blind_review_key_sha256=key_sha,
                ffprobe_bin="/bin/true",
                expected_ffprobe_sha256="4" * 64,
                supervisor_pid=4242,
                output=str(output),
            )
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "load_controller_plan",
                        return_value=(
                            controller_plan,
                            controller_plan_path,
                            controller_plan_sha,
                            {},
                        ),
                    )
                )
                for name in (
                    "_replay_child_scratch_prepare_physical",
                    "validate_compute_preflight",
                    "_replay_child_task_scratch_bind_physical",
                    "validate_exact2_audit",
                    "validate_terminal_host_gate",
                ):
                    stack.enter_context(mock.patch.object(controller, name))
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "validate_ffprobe",
                        return_value=Path("/bin/true"),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "_load_or_reuse_resource_contract",
                        return_value=object(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "_validate_blind_packet",
                        return_value=(manifest, key),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "_generation_resource_smoke_receipt",
                        return_value={},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "_attest_authorized_scratch_inventory",
                        return_value={"outer_probe": {}},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "_task_scratch_physical_value",
                        return_value={"inode": scratch_inner["inode"]},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        "_slurm_child_cgroup_census",
                        return_value={},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        controller.os,
                        "getegid",
                        return_value=root.stat().st_gid,
                    )
                )
                value = controller.seal_child_terminal_physical_attestation(
                    args
                )
            self.assertEqual(
                value["schema_version"],
                controller.CHILD_TERMINAL_PHYSICAL_ATTESTATION_SCHEMA,
            )
            self.assertEqual(value["runtime"], runtime)
            self.assertEqual(
                json.loads(output.read_bytes()),
                value,
            )


class R6ControllerBytesExecHostileTests(unittest.TestCase):
    def test_same_path_malicious_controller_is_rejected_before_top_level_exec(self) -> None:
        module_name = "full30_action_arms_incomplete_repair_exact2_controller_v1"
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            method_root = Path(temporary).resolve(strict=True)
            sentinel = method_root / "controller-top-level-sentinel"
            controller_path = method_root / f"{module_name}.py"
            controller_path.write_text(
                "from pathlib import Path\n"
                f"Path({str(sentinel)!r}).write_text('executed')\n",
                encoding="utf-8",
            )
            previous_module = sys.modules.pop(module_name, None)
            previous_cache = generator._STRICT_CONTROLLER_CACHE
            generator._STRICT_CONTROLLER_CACHE = None
            resource_loader = mock.Mock()
            try:
                with mock.patch.object(
                    generator, "METHOD_ROOT", method_root
                ), mock.patch.object(
                    generator, "load_resource_contract", resource_loader
                ):
                    with self.assertRaisesRegex(
                        generator.ArmsIncompleteExact2GenerationError,
                        "stable source identity/SHA differs",
                    ):
                        generator._strict_public_entry_scratch_chain(
                            controller_plan="/shared/controller-plan.json",
                            expected_controller_plan_sha256="1" * 64,
                            scratch_prepare="/shared/prepare.json",
                            expected_scratch_prepare_sha256="2" * 64,
                            compute_preflight="/shared/preflight.json",
                            expected_compute_preflight_sha256="3" * 64,
                            task_scratch_bind="/shared/bind.json",
                            expected_task_scratch_bind_sha256="4" * 64,
                            exact2_plan={},
                            exact2_plan_path=Path("/shared/plan.json"),
                            exact2_plan_sha256="5" * 64,
                        )
                self.assertFalse(sentinel.exists())
                resource_loader.assert_not_called()
                self.assertNotIn(module_name, sys.modules)
            finally:
                generator._STRICT_CONTROLLER_CACHE = previous_cache
                if previous_module is not None:
                    sys.modules[module_name] = previous_module

    def test_verified_controller_bytes_execute_into_fresh_module(self) -> None:
        module_name = "full30_action_arms_incomplete_repair_exact2_controller_v1"
        source_path = (
            METHOD_ROOT / f"{module_name}.py"
        ).resolve(strict=True)
        self.assertEqual(
            generator.file_sha256(source_path),
            generator.STRICT_CONTROLLER_SHA256,
        )
        verified_path, source = generator._stable_verified_file_bytes(
            source_path,
            generator.STRICT_CONTROLLER_SHA256,
            "test strict controller",
        )
        previous_module = sys.modules.pop(module_name, None)
        try:
            loaded = ModuleType(module_name)
            loaded.__file__ = str(verified_path)
            loaded.__package__ = ""
            loaded.__loader__ = None
            loaded.__spec__ = importlib.util.spec_from_loader(
                module_name, loader=None, origin=str(verified_path)
            )
            sys.modules[module_name] = loaded
            exec(
                compile(
                    source,
                    str(verified_path),
                    "exec",
                    dont_inherit=True,
                ),
                loaded.__dict__,
            )
            self.assertTrue(callable(loaded.load_controller_plan))
            self.assertTrue(callable(loaded.validate_compute_preflight))
            self.assertEqual(
                loaded.TERMINAL_RESOURCE_CONTRACT_SHA256,
                generator.RESOURCE_SPECIALIZED_SHA256,
            )
        finally:
            sys.modules.pop(module_name, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module


class R6RetentionShapeHostileTests(unittest.TestCase):
    @staticmethod
    def _validate(value: dict) -> dict:
        with mock.patch.object(
            controller,
            "_validate_tree_inventory_manifest",
            return_value={"regular_count": 0, "directory_count": 0, "logical_bytes": 0},
        ), mock.patch.object(
            controller, "_validate_renderer_load_lock_shape"
        ), mock.patch.object(
            controller,
            "_scratch_inventory_creation_identity_closes",
            return_value=True,
        ), mock.patch.object(controller.os, "makedev", return_value=0xFD00):
            return controller._validate_child_scratch_retained_terminal_shape(value)

    def test_retention_is_signed_point_in_time_and_never_cleanup_authority(self) -> None:
        baseline = _retained_shape_fixture()
        self.assertIs(self._validate(baseline), baseline)
        for field, hostile_value in (
            ("retained_at_child_terminal_seal", False),
            ("cleanup_authorized", True),
            ("manual_cleanup_authorized_by_release", True),
            ("future_availability_guaranteed", True),
        ):
            hostile = copy.deepcopy(baseline)
            hostile["retention_semantics"][field] = hostile_value
            with self.subTest(field=field), self.assertRaises(
                controller.ArmsIncompleteExact2ControllerError
            ):
                self._validate(_resign(hostile))
        incomplete_replay = copy.deepcopy(baseline)
        incomplete_replay["retained_inventory"][
            "recursive_tree_full_second_replay_equal"
        ] = False
        with self.assertRaises(controller.ArmsIncompleteExact2ControllerError):
            self._validate(_resign(incomplete_replay))

    def test_retained_cgroup_capacity_nested_omission_extra_and_tamper_fail(self) -> None:
        baseline = _retained_shape_fixture()
        hostiles = []
        for parent, field in (
            ("second_terminal_cgroup_census", "unexpected_same_cgroup_process_count"),
            ("host_capacity_observation", "filesystem_total_bytes"),
        ):
            missing = copy.deepcopy(baseline)
            missing[parent].pop(field)
            hostiles.append(_resign(missing))
            extra = copy.deepcopy(baseline)
            extra[parent]["unexpected"] = False
            hostiles.append(_resign(extra))
        tampered = copy.deepcopy(baseline)
        tampered["host_capacity_observation"]["filesystem_total_inodes"] += 1
        hostiles.append(_resign(tampered))
        for hostile in hostiles:
            with self.assertRaises(
                controller.ArmsIncompleteExact2ControllerError
            ):
                self._validate(hostile)

    def test_old_cleanup_cli_and_schema_are_rejected_and_scratch_sealers_do_not_delete(self) -> None:
        with mock.patch.object(sys, "stderr"):
            with self.assertRaises(SystemExit):
                controller.build_parser().parse_args(
                    ["seal-child-scratch-cleanup"]
                )
        old = _retained_shape_fixture()
        old["schema_version"] = "bernini-full30-action-arms-incomplete-exact2-cleanup-v1"
        with self.assertRaises(controller.ArmsIncompleteExact2ControllerError):
            self._validate(_resign(old))
        for function in (
            controller.seal_child_scratch_retained_terminal,
            controller.seal_child_scratch_failure,
        ):
            source = Path(controller.__file__).read_text(encoding="utf-8")
            start = source.index(f"def {function.__name__}")
            tail = source[start:]
            next_def = tail.find("\ndef ", 1)
            body = tail if next_def < 0 else tail[:next_def]
            self.assertNotIn("rmtree", body)
            self.assertNotIn("os.unlink", body)
            self.assertNotIn("Path.unlink", body)

    def test_parent_postretention_validator_never_touches_child_tmp_proc_cgroup_or_capacity(self) -> None:
        controller_plan_path = Path("/shared/controller-plan.json")
        controller_plan_sha = "1" * 64
        controller_plan = {"plan_digest": "2" * 64}
        controller_ref = {
            "path": str(controller_plan_path),
            "file_sha256": controller_plan_sha,
            "plan_digest": controller_plan["plan_digest"],
        }

        def ref(path: Path, sha: str, value: dict) -> dict:
            return {
                "path": str(path),
                "file_sha256": sha,
                "receipt_digest": value["receipt_digest"],
            }

        paths = {
            name: Path(f"/shared/{name}.json")
            for name in (
                "prepare",
                "preflight",
                "bind",
                "generation",
                "terminal",
                "attestation",
                "retained",
                "manifest",
                "key",
            )
        }
        shas = {name: str(index) * 64 for index, name in enumerate(paths, start=1)}
        runtime = {
            "slurm_job_id": "136140",
            "slurm_step_id": "77",
            "hostname": "auh7-1b-gpu-215",
            "sole_numbered_compute_child_required": True,
        }
        prepare = _sign(
            {
                "controller_plan": controller_ref,
                "runtime": runtime,
                "scratch_root": {"path": "/tmp/BOX-EXP-013-r6-136140-77"},
                "retained_probe_file": {"path": "/tmp/BOX-EXP-013-r6-136140-77/probe"},
            }
        )
        prepare_ref = ref(paths["prepare"], shas["prepare"], prepare)
        preflight = _sign({"runtime": runtime, "scratch_prepare": prepare_ref})
        preflight_ref = ref(paths["preflight"], shas["preflight"], preflight)
        task_bind = _sign(
            {
                "scratch_prepare": prepare_ref,
                "compute_preflight": preflight_ref,
                "runtime": runtime,
                "scratch_inner": {
                    "path": "/tmp/BOX-EXP-013-r6-136140-77/inner"
                },
                "renderer_load_lock": {
                    "path": "/tmp/BOX-EXP-013-r6-136140-77/inner/renderer-load.lock"
                },
                "retained_probe_file": prepare["retained_probe_file"],
            }
        )
        bind_ref = ref(paths["bind"], shas["bind"], task_bind)
        generation = _sign(
            {"compute_preflight": preflight_ref, "task_scratch_bind": bind_ref}
        )
        generation_ref = ref(
            paths["generation"], shas["generation"], generation
        )
        terminal = _sign({"compute_preflight": preflight_ref})
        terminal_ref = ref(paths["terminal"], shas["terminal"], terminal)
        manifest = _sign({"packet_id": "a" * 32})
        manifest_ref = ref(paths["manifest"], shas["manifest"], manifest)
        key = _sign({"packet_id": manifest["packet_id"]})
        key_ref = ref(paths["key"], shas["key"], key)
        scratch_inventory = {
            "tree_inventory_sha256": "b" * 64,
            "outer_probe": prepare["retained_probe_file"],
        }
        attestation = _sign(
            {
                "controller_plan": controller_ref,
                "runtime": runtime,
                "scratch_prepare": prepare_ref,
                "compute_preflight": preflight_ref,
                "task_scratch_bind": bind_ref,
                "generation_audit": generation_ref,
                "terminal_host_gate": terminal_ref,
                "blind_review_manifest": manifest_ref,
                "blind_review_key": key_ref,
                "scratch_outer": prepare["scratch_root"],
                "scratch_inner": task_bind["scratch_inner"],
                "renderer_load_lock": task_bind["renderer_load_lock"],
                "scratch_inventory": scratch_inventory,
            }
        )
        attestation_ref = ref(
            paths["attestation"], shas["attestation"], attestation
        )
        retained = _sign(
            {
                "controller_plan": controller_ref,
                "runtime": runtime,
                "scratch_prepare": prepare_ref,
                "compute_preflight": preflight_ref,
                "task_scratch_bind": bind_ref,
                "generation_audit": generation_ref,
                "terminal_host_gate": terminal_ref,
                "physical_attestation": attestation_ref,
                "retained_inventory": scratch_inventory,
                "scratch_outer_creation_identity": prepare["scratch_root"],
                "scratch_inner_creation_identity": task_bind["scratch_inner"],
                "renderer_load_lock_creation_identity": task_bind[
                    "renderer_load_lock"
                ],
            }
        )
        values = {
            paths["prepare"]: prepare,
            paths["preflight"]: preflight,
            paths["bind"]: task_bind,
            paths["generation"]: generation,
            paths["terminal"]: terminal,
            paths["attestation"]: attestation,
            paths["retained"]: retained,
            paths["manifest"]: manifest,
            paths["key"]: key,
        }

        def fake_load(path, _label, _expected):
            resolved = Path(path)
            return values[resolved], resolved, shas[next(
                name for name, candidate in paths.items() if candidate == resolved
            )]

        args = SimpleNamespace(
            controller_plan=str(controller_plan_path),
            expected_controller_plan_sha256=controller_plan_sha,
            scratch_prepare=str(paths["prepare"]),
            expected_scratch_prepare_sha256=shas["prepare"],
            compute_preflight=str(paths["preflight"]),
            expected_compute_preflight_sha256=shas["preflight"],
            task_scratch_bind=str(paths["bind"]),
            expected_task_scratch_bind_sha256=shas["bind"],
            generation_audit=str(paths["generation"]),
            expected_generation_audit_sha256=shas["generation"],
            terminal_host_gate=str(paths["terminal"]),
            expected_terminal_host_gate_sha256=shas["terminal"],
            physical_attestation=str(paths["attestation"]),
            expected_physical_attestation_sha256=shas["attestation"],
            scratch_retained_terminal=str(paths["retained"]),
            expected_scratch_retained_terminal_sha256=shas["retained"],
            blind_review_manifest=str(paths["manifest"]),
            expected_blind_review_manifest_sha256=shas["manifest"],
            blind_review_key=str(paths["key"]),
            expected_blind_review_key_sha256=shas["key"],
        )
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    controller,
                    "load_controller_plan",
                    return_value=(
                        controller_plan,
                        controller_plan_path,
                        controller_plan_sha,
                        {},
                    ),
                )
            )
            stack.enter_context(mock.patch.object(controller, "load_json", side_effect=fake_load))
            for name in (
                "_validate_child_scratch_prepare_postretention",
                "_validate_compute_preflight_postretention",
                "_validate_child_task_scratch_bind_shape",
                "_validate_terminal_host_gate_postretention_attested",
                "_validate_child_terminal_attestation_shape",
                "_validate_child_scratch_retained_terminal_shape",
            ):
                stack.enter_context(mock.patch.object(controller, name))
            stack.enter_context(
                mock.patch.object(controller, "validate_ffprobe", return_value=Path("/bin/true"))
            )
            stack.enter_context(
                mock.patch.object(controller, "_load_or_reuse_resource_contract", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(controller, "_validate_exact2_audit_postretention_attested")
            )
            stack.enter_context(
                mock.patch.object(controller, "_validate_blind_packet", return_value=(manifest, key))
            )
            for name in (
                "_proc_identity",
                "_stable_cgroup_membership",
                "_retention_mount_snapshot",
                "_attest_authorized_scratch_inventory",
            ):
                stack.enter_context(
                    mock.patch.object(
                        controller,
                        name,
                        side_effect=AssertionError(f"forbidden parent physical call: {name}"),
                    )
                )
            stack.enter_context(
                mock.patch.object(
                    controller.os,
                    "statvfs",
                    side_effect=AssertionError("forbidden parent capacity call"),
                )
            )
            for method_name in (
                "lstat",
                "stat",
                "iterdir",
                "resolve",
                "read_text",
                "read_bytes",
            ):
                stack.enter_context(
                    mock.patch.object(
                        Path,
                        method_name,
                        side_effect=AssertionError(
                            f"forbidden parent path access: {method_name}"
                        ),
                    )
                )
            result = controller.validate_child_scratch_retained_terminal(args)
        self.assertEqual(result["receipt_digest"], retained["receipt_digest"])


class R6DurableMarkerHostileTests(unittest.TestCase):
    @staticmethod
    def _write_marker_fixture(path: Path, value: dict) -> str:
        raw = controller.canonical_json_bytes(value) + b"\n"
        path.write_bytes(raw)
        path.chmod(0o400)
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _marker_layout(shared: Path, suffix: str) -> tuple[Path, Path]:
        run = (
            shared
            / (
                "full30-action-arms-incomplete-exact2-r6-deadbeef-"
                f"j136140-r{suffix}"
            )
        )
        logs = run / "logs"
        logs.mkdir(parents=True)
        run.chmod(0o700)
        logs.chmod(0o700)
        return run, logs

    @staticmethod
    def _resident_publish(
        args: SimpleNamespace, token: bytes, after_ready=None, printer=None
    ) -> tuple[dict, mock.Mock, mock.Mock, mock.Mock]:
        def read_commit(_limit: int) -> bytes:
            if after_ready is not None:
                after_ready()
            return token

        reader = mock.Mock(side_effect=read_commit)
        if printer is None:
            printer = mock.Mock()
        signal_mask = mock.Mock()
        stdin = SimpleNamespace(buffer=SimpleNamespace(readline=reader))
        output_gid = Path(args.output).parent.stat().st_gid
        directory_identity = (
            controller._PreparedResidentSharedMarker._directory_identity
        )

        def linux_directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
            # APFS changes a directory's st_nlink when regular files are
            # created; Linux/NFS does not.  Preserve every other held-fd field.
            observed = directory_identity(metadata)
            return (*observed[:-1], 2)

        with mock.patch.object(
            controller.sys, "stdin", stdin
        ), mock.patch(
            "builtins.print", printer
        ), mock.patch.object(
            controller, "_require_shared_nfs_fd"
        ), mock.patch.object(
            controller.os, "getegid", return_value=output_gid
        ), mock.patch.object(
            controller._PreparedResidentSharedMarker,
            "_directory_identity",
            side_effect=linux_directory_identity,
        ), mock.patch.object(
            controller.signal,
            "pthread_sigmask",
            signal_mask,
            create=True,
        ):
            value = controller.resident_publish_parent_generation_status(args)
        return value, printer, reader, signal_mask

    def _ready(self, chain: dict) -> dict:
        return _sign(
            {
                "schema_version": controller.CHILD_TERMINAL_READY_SCHEMA,
                **chain,
                "retained_at_child_terminal_seal": True,
                "srun_exit_observed_by_child": False,
                "parent_step_gone_observed_by_child": False,
                "formal_candidate_count": 2,
                "diagnostic_task_count": 0,
                "optimizer_authorized": False,
            }
        )

    def test_completion_cli_requires_parent_generation_status_pair(self) -> None:
        parser = controller.build_parser()
        commands = next(
            action
            for action in parser._actions
            if "seal-completion" in (getattr(action, "choices", {}) or {})
        )
        complete = commands.choices["seal-completion"]
        options = {
            option: action
            for action in complete._actions
            for option in action.option_strings
        }
        for option in (
            "--parent-generation-status",
            "--expected-parent-generation-status-sha256",
        ):
            self.assertIn(option, options)
            self.assertTrue(options[option].required)
        self.assertNotIn("publish-parent-generation-status", commands.choices)
        self.assertIn(
            "resident-publish-parent-generation-status", commands.choices
        )
        resident = commands.choices[
            "resident-publish-parent-generation-status"
        ]
        resident_options = {
            option: action
            for action in resident._actions
            for option in action.option_strings
        }
        for option in (
            "--parent-generation-precommit",
            "--expected-parent-generation-precommit-sha256",
            "--output",
        ):
            self.assertIn(option, resident_options)
            self.assertTrue(resident_options[option].required)

    def test_child_ready_and_parent_generation_status_are_signed_chain_bound(self) -> None:
        chain = _terminal_marker_chain_fixture()
        ready = self._ready(chain)
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            shared = Path(temporary).resolve(strict=True)
            run, logs = self._marker_layout(shared, "1")
            ready_path = logs / controller.CHILD_TERMINAL_READY_BASENAME
            ready_sha = self._write_marker_fixture(ready_path, ready)
            precommit_path = (
                logs / controller.PARENT_GENERATION_PRECOMMIT_BASENAME
            )
            status_path = logs / controller.PARENT_GENERATION_STATUS_BASENAME
            prepare_args = SimpleNamespace(
                child_terminal_ready=str(ready_path),
                expected_child_terminal_ready_sha256=ready_sha,
                srun_exit_status=0,
                output=str(precommit_path),
            )
            publish_args = SimpleNamespace(
                parent_generation_precommit=str(precommit_path),
                expected_parent_generation_precommit_sha256="",
                output=str(status_path),
            )
            holder = {
                "job_id": "136140",
                "job_state": "RUNNING",
                "node": "auh7-1b-gpu-215",
                "owner_uid": 2012,
                "numbered_steps": [],
                "parent_job_untouched": True,
            }
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller, "_terminal_marker_chain", return_value=chain
            ), mock.patch.object(
                controller,
                "_parent_holder_and_step_observation",
                return_value=holder,
            ), mock.patch.object(
                controller,
                "_write_shared_terminal_marker",
                side_effect=self._write_marker_fixture,
            ):
                precommit = controller.prepare_parent_generation_status(
                    prepare_args
                )
                publish_args.expected_parent_generation_precommit_sha256 = (
                    controller.file_sha256(precommit_path)
                )
                status, printer, reader, signal_mask = self._resident_publish(
                    publish_args,
                    controller.PARENT_GENERATION_PUBLISH_COMMIT_TOKEN,
                )
                loaded = controller.validate_parent_generation_status(
                    SimpleNamespace(
                        parent_generation_status=str(status_path),
                        expected_parent_generation_status_sha256=(
                            controller.file_sha256(status_path)
                        ),
                    )
                )
            expected_status_sha = hashlib.sha256(
                controller.canonical_json_bytes(status) + b"\n"
            ).hexdigest()
            self.assertEqual(
                printer.call_args_list,
                [
                    mock.call(
                        controller.PARENT_GENERATION_PUBLISH_READY_PREFIX
                        + expected_status_sha,
                        flush=True,
                    ),
                    mock.call(
                        controller.PARENT_GENERATION_PUBLISH_ACK_PREFIX
                        + expected_status_sha,
                        flush=True,
                    ),
                ],
            )
            reader.assert_called_once_with(
                len(controller.PARENT_GENERATION_PUBLISH_COMMIT_TOKEN) + 1
            )
            signal_mask.assert_called_once_with(
                controller.signal.SIG_BLOCK,
                {
                    controller.signal.SIGINT,
                    controller.signal.SIGTERM,
                    controller.signal.SIGHUP,
                },
            )
            self.assertEqual(loaded, status)
            self.assertTrue(status["generation_success"])
            self.assertTrue(status["review_pending"])
            self.assertFalse(status["experiment_completion"])
            self.assertTrue(status["publication_from_precommit_only"])
            self.assertFalse(
                status["parent_child_tmp_or_proc_physical_replay_performed"]
            )
            self.assertEqual(
                Path(status["parent_generation_precommit"]["path"]),
                precommit_path,
            )
            self.assertEqual(run, logs.parent)

            prepare_args.srun_exit_status = 1
            rejected_output = logs / "rejected-parent-generation.precommit.json"
            prepare_args.output = str(rejected_output)
            writer = mock.Mock()
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller, "_terminal_marker_chain", return_value=chain
            ), mock.patch.object(
                controller, "_write_shared_terminal_marker", writer
            ), self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError, "not zero"
            ):
                controller.prepare_parent_generation_status(prepare_args)
            writer.assert_not_called()

    def test_resident_parent_status_wrong_commit_or_eof_writes_zero(self) -> None:
        chain = _terminal_marker_chain_fixture()
        ready = self._ready(chain)
        holder = {
            "job_id": "136140",
            "job_state": "RUNNING",
            "node": "auh7-1b-gpu-215",
            "owner_uid": 2012,
            "numbered_steps": [],
            "parent_job_untouched": True,
        }
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            shared = Path(temporary).resolve(strict=True)
            for suffix, token in (
                ("1", b""),
                ("2", b"BOX-EXP-013-r6-PARENT-PUBLISH-COMMIT\nextra"),
                ("3", b"BOX-EXP-013-r6-PARENT-PUBLISH-COMMIU\n"),
            ):
                with self.subTest(token=token):
                    _, logs = self._marker_layout(shared, suffix)
                    ready_path = (
                        logs / controller.CHILD_TERMINAL_READY_BASENAME
                    )
                    ready_sha = self._write_marker_fixture(ready_path, ready)
                    precommit_path = (
                        logs
                        / controller.PARENT_GENERATION_PRECOMMIT_BASENAME
                    )
                    status_path = (
                        logs / controller.PARENT_GENERATION_STATUS_BASENAME
                    )
                    with mock.patch.object(
                        controller, "SHARED_DATA_PREP_ROOT", shared
                    ), mock.patch.object(
                        controller,
                        "_terminal_marker_chain",
                        return_value=chain,
                    ), mock.patch.object(
                        controller,
                        "_parent_holder_and_step_observation",
                        return_value=holder,
                    ), mock.patch.object(
                        controller,
                        "_write_shared_terminal_marker",
                        side_effect=self._write_marker_fixture,
                    ):
                        controller.prepare_parent_generation_status(
                            SimpleNamespace(
                                child_terminal_ready=str(ready_path),
                                expected_child_terminal_ready_sha256=ready_sha,
                                srun_exit_status=0,
                                output=str(precommit_path),
                            )
                        )
                        with self.assertRaisesRegex(
                            controller.ArmsIncompleteExact2ControllerError,
                            "commit token differs",
                        ):
                            self._resident_publish(
                                SimpleNamespace(
                                    parent_generation_precommit=str(
                                        precommit_path
                                    ),
                                    expected_parent_generation_precommit_sha256=(
                                        controller.file_sha256(precommit_path)
                                    ),
                                    output=str(status_path),
                                ),
                                token,
                            )
                    self.assertFalse(status_path.exists())

    def test_resident_publish_timeout_has_no_ack_and_rolls_back_only_created_inode(
        self,
    ) -> None:
        resident_source = inspect.getsource(
            controller.resident_publish_parent_generation_status
        )
        after_token = resident_source[resident_source.index("token = ") :]
        for forbidden in (
            "subprocess.",
            ".resolve(",
            "fstatfs",
            "canonical_json",
            "json.",
        ):
            self.assertNotIn(forbidden, after_token)
        self.assertEqual(resident_source.count("print("), 2)
        self.assertIn("observed = prepared.publish()", after_token)

        main_source = inspect.getsource(controller.main)
        resident_branch = main_source[
            main_source.index(
                'elif args.command == "resident-publish-parent-generation-status"'
            ) : main_source.index(
                'elif args.command == "validate-parent-generation-status"'
            )
        ]
        self.assertIn("return 0", resident_branch)
        self.assertNotIn("canonical_json", resident_branch)

        value = {"schema_version": "resident-timeout-fixture"}
        raw = controller.canonical_json_bytes(value) + b"\n"
        expected_sha = hashlib.sha256(raw).hexdigest()
        real_unlink = os.unlink

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            shared = Path(temporary).resolve(strict=True)
            for suffix, replace_created_inode in (
                ("1", False),
                ("2", True),
            ):
                with self.subTest(replace_created_inode=replace_created_inode):
                    _, logs = self._marker_layout(shared, suffix)
                    status_path = (
                        logs / controller.PARENT_GENERATION_STATUS_BASENAME
                    )
                    precommit = {
                        "schema_version": "resident-timeout-precommit-fixture"
                    }
                    precommit_raw = (
                        controller.canonical_json_bytes(precommit) + b"\n"
                    )
                    precommit_path = (
                        logs
                        / controller.PARENT_GENERATION_PRECOMMIT_BASENAME
                    )
                    precommit_path.write_bytes(precommit_raw)
                    precommit_path.chmod(0o400)
                    precommit_sha = hashlib.sha256(precommit_raw).hexdigest()
                    args = SimpleNamespace(output=str(status_path))
                    events: list[str] = []
                    output_lines: list[str] = []
                    installed: dict[str, object] = {}
                    fsync_calls = 0
                    shared_unlinks: list[tuple[object, object]] = []
                    poison = b"replacement-inode-must-remain-poisoned\n"
                    directory_identity = (
                        controller._PreparedResidentSharedMarker._directory_identity
                    )

                    def linux_directory_identity(
                        metadata: os.stat_result,
                    ) -> tuple[int, ...]:
                        observed = directory_identity(metadata)
                        return (*observed[:-1], 2)

                    def signal_handler_install(
                        signal_number: int, handler: object
                    ) -> object:
                        self.assertEqual(signal_number, controller.signal.SIGALRM)
                        if callable(handler):
                            installed["handler"] = handler
                            events.append("install-timeout-handler")
                        else:
                            events.append("restore-timeout-handler")
                        return controller.signal.SIG_DFL

                    def arm_alarm(seconds: int) -> int:
                        events.append(f"alarm-{seconds}")
                        return 0

                    def emit(line: str, *, flush: bool) -> None:
                        self.assertTrue(flush)
                        output_lines.append(line)
                        events.append(
                            "ready"
                            if line.startswith(
                                controller.PARENT_GENERATION_PUBLISH_READY_PREFIX
                            )
                            else "ack"
                        )

                    def read_commit(limit: int) -> bytes:
                        self.assertEqual(
                            limit,
                            len(
                                controller.PARENT_GENERATION_PUBLISH_COMMIT_TOKEN
                            )
                            + 1,
                        )
                        events.append("commit-token")
                        return controller.PARENT_GENERATION_PUBLISH_COMMIT_TOKEN

                    def fail_first_fsync(_descriptor: int) -> None:
                        nonlocal fsync_calls
                        fsync_calls += 1
                        events.append(f"fsync-{fsync_calls}")
                        if fsync_calls != 1:
                            return
                        if replace_created_inode:
                            real_unlink(status_path)
                            status_path.write_bytes(poison)
                        timeout_handler = installed.get("handler")
                        self.assertTrue(callable(timeout_handler))
                        timeout_handler(controller.signal.SIGALRM, None)  # type: ignore[operator]

                    def record_shared_unlink(
                        path: object, *, dir_fd: object = None
                    ) -> None:
                        shared_unlinks.append((path, dir_fd))
                        real_unlink(path, dir_fd=dir_fd)

                    stdin = SimpleNamespace(
                        buffer=SimpleNamespace(readline=read_commit)
                    )
                    with ExitStack() as stack:
                        stack.enter_context(
                            mock.patch.object(
                                controller,
                                "SHARED_DATA_PREP_ROOT",
                                shared,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                controller,
                                "_prepare_parent_generation_status_value",
                                return_value=(
                                    value,
                                    status_path,
                                    precommit,
                                    precommit_path,
                                    precommit_sha,
                                ),
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(controller, "_require_shared_nfs_fd")
                        )
                        stack.enter_context(
                            mock.patch.object(
                                controller.os,
                                "getegid",
                                return_value=logs.stat().st_gid,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                controller._PreparedResidentSharedMarker,
                                "_directory_identity",
                                side_effect=linux_directory_identity,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(controller.sys, "stdin", stdin)
                        )
                        stack.enter_context(
                            mock.patch("builtins.print", side_effect=emit)
                        )
                        stack.enter_context(
                            mock.patch.object(
                                controller.signal,
                                "signal",
                                side_effect=signal_handler_install,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                controller.signal,
                                "pthread_sigmask",
                                create=True,
                            )
                        )
                        alarm = stack.enter_context(
                            mock.patch.object(
                                controller.signal,
                                "alarm",
                                side_effect=arm_alarm,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                controller.os,
                                "fsync",
                                side_effect=fail_first_fsync,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                controller.os,
                                "unlink",
                                side_effect=record_shared_unlink,
                            )
                        )
                        scratch_remove = stack.enter_context(
                            mock.patch.object(controller.os, "remove")
                        )
                        scratch_rmdir = stack.enter_context(
                            mock.patch.object(controller.os, "rmdir")
                        )
                        scratch_path_unlink = stack.enter_context(
                            mock.patch.object(Path, "unlink")
                        )
                        with self.assertRaisesRegex(
                            controller.ArmsIncompleteExact2ControllerError,
                            "held-fd publication timed out",
                        ):
                            controller.resident_publish_parent_generation_status(
                                args
                            )

                    self.assertLess(
                        events.index("install-timeout-handler"),
                        events.index("ready"),
                    )
                    self.assertLess(
                        events.index("commit-token"), events.index("alarm-30")
                    )
                    self.assertLess(events.index("alarm-30"), events.index("fsync-1"))
                    self.assertIn("alarm-5", events)
                    self.assertEqual(output_lines, [
                        controller.PARENT_GENERATION_PUBLISH_READY_PREFIX
                        + expected_sha
                    ])
                    self.assertFalse(
                        any(
                            line.startswith(
                                controller.PARENT_GENERATION_PUBLISH_ACK_PREFIX
                            )
                            for line in output_lines
                        )
                    )
                    alarm.assert_any_call(
                        controller.PARENT_GENERATION_PUBLISH_TIMEOUT_SECONDS
                    )
                    alarm.assert_any_call(
                        controller.PARENT_GENERATION_ROLLBACK_TIMEOUT_SECONDS
                    )
                    scratch_remove.assert_not_called()
                    scratch_rmdir.assert_not_called()
                    scratch_path_unlink.assert_not_called()
                    if replace_created_inode:
                        self.assertEqual(shared_unlinks, [])
                        self.assertEqual(status_path.read_bytes(), poison)
                    else:
                        self.assertEqual(len(shared_unlinks), 1)
                        self.assertEqual(
                            shared_unlinks[0][0],
                            controller.PARENT_GENERATION_STATUS_BASENAME,
                        )
                        self.assertIsNotNone(shared_unlinks[0][1])
                        self.assertFalse(status_path.exists())

    def test_resident_publish_holds_and_replays_precommit_and_status_across_ready(
        self,
    ) -> None:
        resident_source = inspect.getsource(
            controller.resident_publish_parent_generation_status
        )
        held_init = resident_source.index("held_precommit = _HeldResidentReceipt(")
        ready = resident_source.index("print(", held_init)
        token = resident_source.index("token = ", ready)
        precommit_before = resident_source.index(
            "held_precommit.replay()", token
        )
        publish = resident_source.index("observed = prepared.publish()", token)
        precommit_after = resident_source.index(
            "held_precommit.replay()", precommit_before + 1
        )
        status_after = resident_source.index(
            "prepared.replay_published_marker()", publish
        )
        ack = resident_source.index(
            "PARENT_GENERATION_PUBLISH_ACK_PREFIX", status_after
        )
        self.assertLess(
            held_init,
            ready,
        )
        self.assertLess(
            token,
            precommit_before,
        )
        self.assertLess(
            precommit_before,
            publish,
        )
        self.assertLess(
            publish,
            precommit_after,
        )
        self.assertLess(
            precommit_after,
            status_after,
        )
        self.assertLess(status_after, ack)

        held_init_source = inspect.getsource(controller._HeldResidentReceipt.__init__)
        held_replay_source = inspect.getsource(
            controller._HeldResidentReceipt.replay
        )
        held_identity_source = inspect.getsource(
            controller._HeldResidentReceipt._identity
        )
        self.assertIn('getattr(os, "O_NOFOLLOW", 0)', held_init_source)
        self.assertIn("dir_fd=logs_descriptor", held_init_source)
        self.assertEqual(held_replay_source.count("os.read(self.descriptor"), 2)
        for metadata_field in (
            "st_dev",
            "st_ino",
            "st_uid",
            "st_gid",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_blocks",
            "st_mtime_ns",
            "st_ctime_ns",
        ):
            self.assertIn(metadata_field, held_identity_source)
            self.assertIn(
                metadata_field,
                inspect.getsource(
                    controller._PreparedResidentSharedMarker.publish
                ),
            )
            self.assertIn(
                metadata_field,
                inspect.getsource(
                    controller._PreparedResidentSharedMarker.replay_published_marker
                ),
            )

        chain = _terminal_marker_chain_fixture()
        child_ready = self._ready(chain)
        holder = {
            "job_id": "136140",
            "job_state": "RUNNING",
            "node": "auh7-1b-gpu-215",
            "owner_uid": 2012,
            "numbered_steps": [],
            "parent_job_untouched": True,
        }

        def rewrite_same_inode(path: Path) -> tuple[tuple[int, int], bytes]:
            before = path.stat()
            original = path.read_bytes()
            poison = original[:-1] + (b" " if original[-1:] != b" " else b"\n")
            path.chmod(0o600)
            with path.open("r+b") as writer:
                writer.seek(0)
                writer.write(poison)
                writer.truncate()
                writer.flush()
                os.fsync(writer.fileno())
            path.chmod(0o400)
            after = path.stat()
            self.assertEqual(
                (before.st_dev, before.st_ino),
                (after.st_dev, after.st_ino),
            )
            return (before.st_dev, before.st_ino), poison

        def assert_ready_without_ack(printer: mock.Mock) -> None:
            self.assertEqual(len(printer.call_args_list), 1)
            line = printer.call_args_list[0].args[0]
            self.assertTrue(
                line.startswith(controller.PARENT_GENERATION_PUBLISH_READY_PREFIX)
            )
            self.assertFalse(
                line.startswith(controller.PARENT_GENERATION_PUBLISH_ACK_PREFIX)
            )
            self.assertEqual(
                printer.call_args_list[0].kwargs,
                {"flush": True},
            )

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            shared = Path(temporary).resolve(strict=True)

            def prepare(suffix: str) -> tuple[Path, Path, SimpleNamespace]:
                _, logs = self._marker_layout(shared, suffix)
                ready_path = logs / controller.CHILD_TERMINAL_READY_BASENAME
                ready_sha = self._write_marker_fixture(ready_path, child_ready)
                precommit_path = (
                    logs / controller.PARENT_GENERATION_PRECOMMIT_BASENAME
                )
                status_path = logs / controller.PARENT_GENERATION_STATUS_BASENAME
                with mock.patch.object(
                    controller, "SHARED_DATA_PREP_ROOT", shared
                ), mock.patch.object(
                    controller, "_terminal_marker_chain", return_value=chain
                ), mock.patch.object(
                    controller,
                    "_parent_holder_and_step_observation",
                    return_value=holder,
                ), mock.patch.object(
                    controller,
                    "_write_shared_terminal_marker",
                    side_effect=self._write_marker_fixture,
                ):
                    controller.prepare_parent_generation_status(
                        SimpleNamespace(
                            child_terminal_ready=str(ready_path),
                            expected_child_terminal_ready_sha256=ready_sha,
                            srun_exit_status=0,
                            output=str(precommit_path),
                        )
                    )
                return (
                    precommit_path,
                    status_path,
                    SimpleNamespace(
                        parent_generation_precommit=str(precommit_path),
                        expected_parent_generation_precommit_sha256=(
                            controller.file_sha256(precommit_path)
                        ),
                        output=str(status_path),
                    ),
                )

            for suffix, attack in (
                ("1", "unlink"),
                ("2", "replace-inode"),
                ("3", "rewrite-same-inode"),
            ):
                with self.subTest(attack=attack):
                    precommit_path, status_path, args = prepare(suffix)
                    original = precommit_path.read_bytes()
                    original_identity = (
                        precommit_path.stat().st_dev,
                        precommit_path.stat().st_ino,
                    )

                    def attack_after_ready() -> None:
                        if attack == "unlink":
                            os.unlink(precommit_path)
                        elif attack == "replace-inode":
                            os.unlink(precommit_path)
                            precommit_path.write_bytes(original)
                            precommit_path.chmod(0o400)
                            replacement = precommit_path.stat()
                            self.assertNotEqual(
                                original_identity,
                                (replacement.st_dev, replacement.st_ino),
                            )
                        else:
                            observed_identity, _ = rewrite_same_inode(
                                precommit_path
                            )
                            self.assertEqual(observed_identity, original_identity)

                    printer = mock.Mock()
                    with mock.patch.object(
                        controller, "SHARED_DATA_PREP_ROOT", shared
                    ), self.assertRaises(Exception):
                        self._resident_publish(
                            args,
                            controller.PARENT_GENERATION_PUBLISH_COMMIT_TOKEN,
                            after_ready=attack_after_ready,
                            printer=printer,
                        )
                    assert_ready_without_ack(printer)
                    self.assertFalse(status_path.exists())

            precommit_path, status_path, args = prepare("4")
            real_publish = controller._PreparedResidentSharedMarker.publish

            def publish_then_tamper_precommit(
                prepared: controller._PreparedResidentSharedMarker,
            ) -> str:
                observed = real_publish(prepared)
                rewrite_same_inode(precommit_path)
                return observed

            printer = mock.Mock()
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller._PreparedResidentSharedMarker,
                "publish",
                new=publish_then_tamper_precommit,
            ), self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "resident held receipt bytes/identity drifted",
            ):
                self._resident_publish(
                    args,
                    controller.PARENT_GENERATION_PUBLISH_COMMIT_TOKEN,
                    printer=printer,
                )
            assert_ready_without_ack(printer)
            self.assertFalse(status_path.exists())

            precommit_path, status_path, args = prepare("5")
            real_status_replay = (
                controller._PreparedResidentSharedMarker.replay_published_marker
            )
            status_identity: list[tuple[int, int]] = []
            status_poison: list[bytes] = []

            def tamper_then_replay_status(
                prepared: controller._PreparedResidentSharedMarker,
            ) -> None:
                identity, poison = rewrite_same_inode(status_path)
                status_identity.append(identity)
                status_poison.append(poison)
                real_status_replay(prepared)

            printer = mock.Mock()
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller._PreparedResidentSharedMarker,
                "replay_published_marker",
                new=tamper_then_replay_status,
            ), self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "resident marker final stable replay differs",
            ):
                self._resident_publish(
                    args,
                    controller.PARENT_GENERATION_PUBLISH_COMMIT_TOKEN,
                    printer=printer,
                )
            assert_ready_without_ack(printer)
            self.assertTrue(status_path.exists())
            self.assertEqual(status_path.read_bytes(), status_poison[0])
            status_after = status_path.stat()
            self.assertEqual(
                status_identity[0],
                (status_after.st_dev, status_after.st_ino),
            )

    def test_blind_child_exit_replacement_and_fully_resigned_chain_tamper_write_zero(self) -> None:
        chain = _terminal_marker_chain_fixture()
        ready = self._ready(chain)
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            shared = Path(temporary).resolve(strict=True)
            run, logs = self._marker_layout(shared, "1")
            _, alternate_logs = self._marker_layout(shared, "2")
            path = logs / controller.CHILD_TERMINAL_READY_BASENAME
            original_sha = self._write_marker_fixture(path, ready)
            replaced = copy.deepcopy(ready)
            replaced["packet_id"] = "a" * 32
            replaced = _resign(replaced)
            path.chmod(0o600)
            path.write_bytes(controller.canonical_json_bytes(replaced) + b"\n")
            writer = mock.Mock()
            args = SimpleNamespace(
                child_terminal_ready=str(path),
                expected_child_terminal_ready_sha256=original_sha,
                srun_exit_status=0,
                output=str(
                    logs / controller.PARENT_GENERATION_PRECOMMIT_BASENAME
                ),
            )
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller, "_terminal_marker_chain", return_value=chain
            ), mock.patch.object(controller, "_write_shared_terminal_marker", writer):
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError, "SHA-256 differs"
                ):
                    controller.prepare_parent_generation_status(args)
            writer.assert_not_called()

            args.expected_child_terminal_ready_sha256 = controller.file_sha256(path)
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller, "_terminal_marker_chain", return_value=chain
            ), mock.patch.object(controller, "_write_shared_terminal_marker", writer):
                with self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError, "chain differs"
                ):
                    controller.prepare_parent_generation_status(args)
            writer.assert_not_called()

            original_sha = self._write_marker_fixture(path, ready)
            args.expected_child_terminal_ready_sha256 = original_sha
            root_ready = run / controller.CHILD_TERMINAL_READY_BASENAME
            root_ready_sha = self._write_marker_fixture(root_ready, ready)
            args.child_terminal_ready = str(root_ready)
            args.expected_child_terminal_ready_sha256 = root_ready_sha
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller, "_terminal_marker_chain", return_value=chain
            ), mock.patch.object(
                controller, "_write_shared_terminal_marker", writer
            ), self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "path/topology differs",
            ):
                controller.prepare_parent_generation_status(args)
            writer.assert_not_called()

            renamed_ready = logs / "renamed-child-terminal-ready.status"
            renamed_ready_sha = self._write_marker_fixture(renamed_ready, ready)
            args.child_terminal_ready = str(renamed_ready)
            args.expected_child_terminal_ready_sha256 = renamed_ready_sha
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller, "_terminal_marker_chain", return_value=chain
            ), mock.patch.object(
                controller, "_write_shared_terminal_marker", writer
            ), self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "path/topology differs",
            ):
                controller.prepare_parent_generation_status(args)
            writer.assert_not_called()

            args.child_terminal_ready = str(path)
            args.expected_child_terminal_ready_sha256 = original_sha
            args.output = str(
                alternate_logs
                / controller.PARENT_GENERATION_PRECOMMIT_BASENAME
            )
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller, "_terminal_marker_chain", return_value=chain
            ), mock.patch.object(
                controller, "_write_shared_terminal_marker", writer
            ), self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "run roots differ",
            ):
                controller.prepare_parent_generation_status(args)
            writer.assert_not_called()

            args.output = str(
                run / controller.PARENT_GENERATION_PRECOMMIT_BASENAME
            )
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller, "_terminal_marker_chain", return_value=chain
            ), mock.patch.object(
                controller, "_write_shared_terminal_marker", writer
            ), self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "path/topology differs",
            ):
                controller.prepare_parent_generation_status(args)
            writer.assert_not_called()

            valid_precommit_path = (
                logs / controller.PARENT_GENERATION_PRECOMMIT_BASENAME
            )
            args.output = str(valid_precommit_path)
            holder = {
                "job_id": "136140",
                "job_state": "RUNNING",
                "node": "auh7-1b-gpu-215",
                "owner_uid": 2012,
                "numbered_steps": [],
                "parent_job_untouched": True,
            }
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller, "_terminal_marker_chain", return_value=chain
            ), mock.patch.object(
                controller,
                "_parent_holder_and_step_observation",
                return_value=holder,
            ), mock.patch.object(
                controller,
                "_write_shared_terminal_marker",
                side_effect=self._write_marker_fixture,
            ):
                precommit = controller.prepare_parent_generation_status(args)

            for hostile_output, expected_message in (
                (
                    alternate_logs
                    / controller.PARENT_GENERATION_STATUS_BASENAME,
                    "run roots differ",
                ),
                (
                    run / controller.PARENT_GENERATION_STATUS_BASENAME,
                    "path/topology differs",
                ),
            ):
                with self.subTest(status_output=str(hostile_output)), mock.patch.object(
                    controller, "SHARED_DATA_PREP_ROOT", shared
                ), self.assertRaisesRegex(
                    controller.ArmsIncompleteExact2ControllerError,
                    expected_message,
                ):
                    self._resident_publish(
                        SimpleNamespace(
                            parent_generation_precommit=str(valid_precommit_path),
                            expected_parent_generation_precommit_sha256=(
                                controller.file_sha256(valid_precommit_path)
                            ),
                            output=str(hostile_output),
                        ),
                        controller.PARENT_GENERATION_PUBLISH_COMMIT_TOKEN,
                    )
                self.assertFalse(hostile_output.exists())

            forged = copy.deepcopy(precommit)
            forged["child_terminal_ready"]["receipt_digest"] = "f" * 64
            forged = _resign(forged)
            valid_precommit_path.chmod(0o600)
            valid_precommit_path.write_bytes(
                controller.canonical_json_bytes(forged) + b"\n"
            )
            valid_precommit_path.chmod(0o400)
            status_path = logs / controller.PARENT_GENERATION_STATUS_BASENAME
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ):
                self._resident_publish(
                    SimpleNamespace(
                        parent_generation_precommit=str(valid_precommit_path),
                        expected_parent_generation_precommit_sha256=(
                            controller.file_sha256(valid_precommit_path)
                        ),
                        output=str(status_path),
                    ),
                    controller.PARENT_GENERATION_PUBLISH_COMMIT_TOKEN,
                )
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "chain differs",
            ):
                controller.validate_parent_generation_status(
                    SimpleNamespace(
                        parent_generation_status=str(status_path),
                        expected_parent_generation_status_sha256=(
                            controller.file_sha256(status_path)
                        ),
                    )
                )

    def test_shared_marker_shortwrite_enospc_fsync_symlink_and_rename_fail_closed(self) -> None:
        value = {"schema_version": "fixture", "receipt_digest": "0" * 64}
        real_require = controller.require

        def allow_host_directory_metadata(condition: bool, message: str) -> None:
            if message == "terminal marker run/log fd-to-name topology differs":
                return
            real_require(condition, message)

        def layout(parent: Path, suffix: str) -> tuple[Path, Path]:
            run = parent / f"full30-action-arms-incomplete-exact2-r6-deadbeef-j136140-r{suffix}"
            logs = run / "logs"
            logs.mkdir(parents=True)
            run.chmod(0o700)
            logs.chmod(0o700)
            return logs / "marker.json", logs

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            shared = Path(temporary).resolve(strict=True)

            class NfsFstatfs:
                argtypes = None
                restype = None

                def __call__(self, _descriptor, filesystem_pointer) -> int:
                    filesystem_pointer._obj.f_type = 0x6969
                    return 0

            nfs_library = SimpleNamespace(fstatfs=NfsFstatfs())
            for label, fault in (
                ("shortwrite", "shortwrite"),
                ("enospc", "enospc"),
                ("fsync", "fsync"),
            ):
                marker, logs = layout(shared, str(len(list(shared.iterdir())) + 1))
                with self.subTest(label=label), ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(controller, "SHARED_DATA_PREP_ROOT", shared)
                    )
                    stack.enter_context(
                        mock.patch.object(
                            controller,
                            "require",
                            side_effect=allow_host_directory_metadata,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            controller.ctypes, "CDLL", return_value=nfs_library
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            controller.os, "getegid", return_value=logs.stat().st_gid
                        )
                    )
                    if fault == "shortwrite":
                        stack.enter_context(
                            mock.patch.object(controller.os, "write", side_effect=[1, 0])
                        )
                    elif fault == "enospc":
                        stack.enter_context(
                            mock.patch.object(
                                controller.os,
                                "write",
                                side_effect=OSError(errno.ENOSPC, "full"),
                            )
                        )
                    else:
                        stack.enter_context(
                            mock.patch.object(
                                controller.os,
                                "fsync",
                                side_effect=OSError(errno.EIO, "fsync"),
                            )
                        )
                    with self.assertRaises(Exception):
                        controller._write_shared_terminal_marker(marker, value)
                self.assertFalse(marker.exists())

            symlink, logs = layout(shared, str(len(list(shared.iterdir())) + 1))
            target = logs / "target"
            target.write_bytes(b"target")
            symlink.symlink_to(target.name)
            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller,
                "require",
                side_effect=allow_host_directory_metadata,
            ), self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "output name already exists",
            ):
                controller._write_shared_terminal_marker(symlink, value)
            self.assertTrue(symlink.is_symlink())

            marker, logs = layout(shared, str(len(list(shared.iterdir())) + 1))
            renamed = logs / "renamed-invalid-marker.json"

            real_open = controller.os.open
            open_calls = []

            def rename_before_replay(path_value, flags, *args, **kwargs):
                open_calls.append((str(path_value), flags))
                if (
                    path_value == marker.name
                    and not flags & os.O_WRONLY
                    and marker.exists()
                ):
                    marker.rename(renamed)
                return real_open(path_value, flags, *args, **kwargs)

            with mock.patch.object(
                controller, "SHARED_DATA_PREP_ROOT", shared
            ), mock.patch.object(
                controller,
                "require",
                side_effect=allow_host_directory_metadata,
            ), mock.patch.object(
                controller.ctypes, "CDLL", return_value=nfs_library
            ), mock.patch.object(
                controller.os, "getegid", return_value=logs.stat().st_gid
            ), mock.patch.object(
                controller.os, "open", side_effect=rename_before_replay
            ):
                with self.assertRaises(Exception) as raised:
                    controller._write_shared_terminal_marker(marker, value)
            self.assertFalse(marker.exists())
            self.assertTrue(renamed.exists(), (open_calls, str(raised.exception)))


class R6InventoryAndPidHostileTests(unittest.TestCase):
    def test_same_inode_same_size_byte_tamper_during_inventory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary).resolve(strict=True)
            payload = root / "payload.bin"
            payload.write_bytes(b"A" * 4096)
            before = payload.stat()
            descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            real_read = controller.os.read
            mutated = False

            def hostile_read(fd: int, size: int) -> bytes:
                nonlocal mutated
                chunk = real_read(fd, size)
                if chunk and not mutated:
                    mutated = True
                    payload.write_bytes(b"B" * 4096)
                    os.utime(
                        payload,
                        ns=(before.st_atime_ns, before.st_mtime_ns),
                    )
                return chunk

            try:
                with mock.patch.object(
                    controller, "CHILD_SCRATCH_OWNER_UID", os.geteuid()
                ), mock.patch.object(
                    controller, "CHILD_SCRATCH_OWNER_GID", payload.stat().st_gid
                ), mock.patch.object(controller.os, "read", side_effect=hostile_read):
                    with self.assertRaisesRegex(
                        controller.ArmsIncompleteExact2ControllerError,
                        "changed during double stable SHA read",
                    ):
                        controller._scan_directory_contents_no_follow(
                            descriptor, expected_device=payload.stat().st_dev
                        )
            finally:
                os.close(descriptor)
            self.assertTrue(mutated)

    def test_pid_reuse_start_ticks_drift_is_rejected(self) -> None:
        identities = [
            {
                "pid": 42,
                "state": "S",
                "parent_pid": 2,
                "start_ticks": ticks,
                "cgroup_v2_path": "/fixture",
            }
            for ticks in (100, 101)
        ]
        with mock.patch.object(
            Path, "read_text", return_value="42\n"
        ), mock.patch.object(
            controller, "_proc_identity", side_effect=identities
        ):
            with self.assertRaisesRegex(
                controller.ArmsIncompleteExact2ControllerError,
                "did not replay stably",
            ):
                controller._stable_cgroup_membership("/fixture")


class R6ResourceIdentityHostileTests(unittest.TestCase):
    def test_current_r6_resource_preimage_specialization_and_module_close(self) -> None:
        base_path = (
            METHOD_ROOT / "tools/reserve4_fixed_generation_sp4_v1.py"
        ).resolve(strict=True)
        base = base_path.read_bytes()
        self.assertEqual(base.count(b"136141"), 7)
        self.assertEqual(base.count(b"136140"), 0)
        specialized = base.replace(b"136141", b"136140")
        relative = "tools/reserve4_fixed_generation_sp4_136140_specialized_v1.py"
        rank_relative = "scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
        self.assertEqual(
            hashlib.sha256(base).hexdigest(),
            "721c2bed86b3d547ea97649a914b40e4d26b744799c3c83b010d3d81cc342102",
        )
        self.assertEqual(
            hashlib.sha256(base).hexdigest(), generator.RESOURCE_PREIMAGE_SHA256
        )
        self.assertEqual(len(specialized), 188_050)
        self.assertEqual(len(specialized), generator.RESOURCE_SPECIALIZED_SIZE)
        self.assertEqual(
            hashlib.sha256(specialized).hexdigest(),
            "052b2ce7cc468d37f37bfc21df8d9a9cffd2c66eebac779e70f9186738f62ad3",
        )
        self.assertEqual(
            hashlib.sha256(specialized).hexdigest(),
            generator.RESOURCE_SPECIALIZED_SHA256,
        )
        self.assertEqual(release._specialize_r6_resource(base), specialized)
        self.assertEqual(
            generator.RESOURCE_SPECIALIZED_MODULE_NAME,
            "_bernini_full30_fit_repair_resource_r6_052b2ce7",
        )
        self.assertEqual(
            release.R6_REPLACEMENT_PINS[relative],
            (0o444, generator.RESOURCE_SPECIALIZED_SHA256),
        )
        self.assertEqual(
            release.R6_REPLACEMENT_PINS[rank_relative][1],
            controller.file_sha256(
                METHOD_ROOT / rank_relative
            ),
        )

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            method_root = Path(temporary).resolve(strict=True)
            resource = method_root / relative
            resource.parent.mkdir(parents=True)
            resource.write_bytes(specialized)
            resource.chmod(0o444)
            module_name = generator.RESOURCE_SPECIALIZED_MODULE_NAME
            previous_module = sys.modules.pop(module_name, None)
            previous_cache = generator._RESOURCE_MODULE_CACHE
            previous_primitives = generator._RESOURCE_MODULE_PRIMITIVES
            generator._RESOURCE_MODULE_CACHE = None
            generator._RESOURCE_MODULE_PRIMITIVES = None
            try:
                with mock.patch.object(generator, "METHOD_ROOT", method_root):
                    loaded = generator.load_resource_contract(resource)
                self.assertIs(loaded, sys.modules[module_name])
                self.assertEqual(loaded.__name__, module_name)
                self.assertEqual(loaded.__file__, str(resource))
            finally:
                sys.modules.pop(module_name, None)
                generator._RESOURCE_MODULE_CACHE = previous_cache
                generator._RESOURCE_MODULE_PRIMITIVES = previous_primitives
                if previous_module is not None:
                    sys.modules[module_name] = previous_module

    def test_revoked_aa2f_resource_postimage_is_rejected_before_import(self) -> None:
        relative = "tools/reserve4_fixed_generation_sp4_136140_specialized_v1.py"
        revoked = release._frozen_base_payloads(METHOD_ROOT.resolve(strict=True))[relative]
        self.assertEqual(
            hashlib.sha256(revoked).hexdigest(),
            generator.REVOKED_R5_RESOURCE_SPECIALIZED_SHA256,
        )
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            method_root = Path(temporary) / "methods/bernini_action_editing"
            resource = method_root / relative
            resource.parent.mkdir(parents=True)
            resource.write_bytes(revoked)
            resource.chmod(0o444)
            previous_module = sys.modules.pop(
                generator.RESOURCE_SPECIALIZED_MODULE_NAME, None
            )
            previous_cache = generator._RESOURCE_MODULE_CACHE
            previous_primitives = generator._RESOURCE_MODULE_PRIMITIVES
            generator._RESOURCE_MODULE_CACHE = None
            generator._RESOURCE_MODULE_PRIMITIVES = None
            try:
                with mock.patch.object(generator, "METHOD_ROOT", method_root):
                    with self.assertRaisesRegex(
                        generator.ArmsIncompleteExact2GenerationError,
                        "stable source identity/SHA differs",
                    ):
                        generator.load_resource_contract(resource)
            finally:
                generator._RESOURCE_MODULE_CACHE = previous_cache
                generator._RESOURCE_MODULE_PRIMITIVES = previous_primitives
                if previous_module is not None:
                    sys.modules[generator.RESOURCE_SPECIALIZED_MODULE_NAME] = previous_module


class R6RankStartTicksHostileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rank_path = (
            METHOD_ROOT / "scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
        ).resolve(strict=True)
        cls.rank_source = cls.rank_path.read_text(encoding="utf-8")
        start = cls.rank_source.index("proc_start_ticks() {")
        end = cls.rank_source.index("\nbounded_reap_owned_child() {", start)
        cls.signal_functions = cls.rank_source[start:end]
        record_start = cls.rank_source.index("record_signal() {")
        record_end = cls.rank_source.index("\non_int() {", record_start)
        cls.record_function = cls.rank_source[record_start:record_end]

    @staticmethod
    def _bash(source: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-s"],
            input=source,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_unknown_malformed_reused_and_zombie_never_signal(self) -> None:
        cases = {
            "proc-unreadable": ("proc_start_ticks() { return 1; }", 70),
            "malformed": ("proc_start_ticks() { printf 'malformed\\n'; }", 70),
            "pid-reused": ("proc_start_ticks() { printf '101:S\\n'; }", 70),
            "instant-exit-zombie": (
                "proc_start_ticks() { printf '100:Z\\n'; }",
                1,
            ),
        }
        for label, (observation, expected_status) in cases.items():
            with self.subTest(label=label):
                completed = self._bash(
                    self.signal_functions
                    + "\n"
                    + observation
                    + "\n"
                    + "kill() { printf 'KILL:%s:%s\\n' \"$1\" \"$2\"; }\n"
                    + "child_pid=4242\nchild_start_ticks=100\n"
                    + "signal_owned_child_exact TERM\nstatus=$?\n"
                    + "printf 'STATUS:%s\\n' \"${status}\"\n"
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertNotIn("KILL:", completed.stdout)
                self.assertEqual(
                    completed.stdout.strip(), f"STATUS:{expected_status}"
                )

    def test_exact_same_read_identity_authorizes_one_signal(self) -> None:
        completed = self._bash(
            self.signal_functions
            + "\nproc_start_ticks() { printf '100:S\\n'; }\n"
            + "kill() { printf 'KILL:%s:%s\\n' \"$1\" \"$2\"; }\n"
            + "child_pid=4242\nchild_start_ticks=100\n"
            + "signal_owned_child_exact TERM\nstatus=$?\n"
            + "printf 'STATUS:%s\\n' \"${status}\"\n"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.splitlines(),
            ["KILL:-TERM:4242", "STATUS:0"],
        )

    def test_record_signal_only_latches_and_never_calls_kill(self) -> None:
        completed = self._bash(
            self.record_function
            + "\npending_signal_status=0\n"
            + "kill() { printf 'KILL:%s:%s\\n' \"$1\" \"$2\"; }\n"
            + "record_signal 143\nrecord_signal 130\n"
            + "printf 'PENDING:%s\\n' \"${pending_signal_status}\"\n"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "PENDING:143")
        self.assertNotIn("KILL:", completed.stdout)


class ReleaseAndLauncherTests(unittest.TestCase):
    def test_core_release_is_deterministic_and_contains_byte_exact_frozen_stack(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT.resolve(strict=True))
        first = release.build_archive(manifest, payloads)
        second = release.build_archive(manifest, payloads)
        self.assertEqual(first, second)
        self.assertEqual(manifest["file_count"], 25)
        self.assertEqual(manifest["frozen_base"]["source_member_count"], 21)
        self.assertEqual(
            manifest["frozen_base"]["byte_exact_carried_member_count"], 18
        )
        self.assertEqual(
            manifest["frozen_base"]["r6_owned_replacement_member_count"], 3
        )
        self.assertEqual(
            manifest["frozen_base"]["r6_owned_replacement_members"],
            sorted(release.R6_REPLACEMENT_PINS),
        )
        self.assertFalse(manifest["frozen_base"]["all_members_byte_exact"])
        self.assertTrue(manifest["frozen_base"]["modified"])
        self.assertEqual(manifest["authority"]["formal_candidate_count"], 2)
        self.assertEqual(manifest["authority"]["diagnostic_task_count"], 0)
        self.assertFalse(manifest["authority"]["optimizer_authorized"])
        self.assertEqual(manifest["release_generation"], release.RELEASE_GENERATION)
        self.assertEqual(manifest["release_generation"], "r6")
        self.assertNotIn(release.LAUNCHER_MEMBER, payloads)
        self.assertNotIn(
            release.LAUNCHER_MEMBER,
            [row["path"] for row in manifest["files"]],
        )
        self.assertEqual(manifest["topology"]["core_archive_member_count"], 25)
        self.assertEqual(
            manifest["topology"]["frozen_base_byte_exact_member_count"], 18
        )
        self.assertEqual(
            manifest["topology"]["r6_owned_replacement_member_count"], 3
        )
        self.assertTrue(
            manifest["topology"]["detached_launcher_excluded_from_core"]
        )
        portable = manifest["authority"]["portable_ffprobe"]
        self.assertEqual(portable["path"], controller.PORTABLE_FFPROBE_PATH)
        self.assertEqual(
            portable["file_sha256"], controller.PORTABLE_FFPROBE_SHA256
        )
        self.assertFalse(portable["caller_override_allowed"])
        frozen_python = manifest["authority"]["frozen_python"]
        self.assertEqual(frozen_python["path"], release.FROZEN_PYTHON_PATH)
        self.assertEqual(
            frozen_python["realpath"], release.FROZEN_PYTHON_REALPATH
        )
        self.assertEqual(
            frozen_python["file_type"], release.FROZEN_PYTHON_FILE_TYPE
        )
        self.assertEqual(
            frozen_python["mode_octal"], release.FROZEN_PYTHON_MODE_OCTAL
        )
        self.assertEqual(frozen_python["uid"], release.FROZEN_PYTHON_UID)
        self.assertEqual(frozen_python["size"], release.FROZEN_PYTHON_SIZE)
        self.assertEqual(
            frozen_python["link_count"], release.FROZEN_PYTHON_LINK_COUNT
        )
        self.assertEqual(
            frozen_python["file_sha256"], release.FROZEN_PYTHON_SHA256
        )
        self.assertFalse(frozen_python["caller_override_allowed"])
        self.assertFalse(frozen_python["symlink_allowed"])
        self.assertTrue(
            frozen_python[
                "physical_validation_before_run_root_or_srun_required"
            ]
        )
        revoked_r4 = manifest["authority"]["revoked_resource_reuse_r4"]
        self.assertEqual(revoked_r4["launcher_invocation_count"], 1)
        self.assertEqual(revoked_r4["numbered_child_count"], 0)
        self.assertFalse(revoked_r4["run_root_created"])
        self.assertEqual(revoked_r4["gpu_or_model_invocation_count"], 0)
        self.assertTrue(revoked_r4["permanent_no_go"])

    def test_current_r6_deployment_audits_and_revoked_r5_is_rejected(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT.resolve(strict=True))
        archive_raw = release.build_archive(manifest, payloads)
        manifest_raw = release.canonical_json_bytes(manifest) + b"\n"
        launcher_raw = LAUNCHER.read_bytes()
        expected_archive = hashlib.sha256(archive_raw).hexdigest()
        expected_manifest = hashlib.sha256(manifest_raw).hexdigest()
        expected_launcher = hashlib.sha256(launcher_raw).hexdigest()
        envelope = release.deployment_envelope_value(
            manifest=manifest,
            archive_sha256=expected_archive,
            manifest_sha256=expected_manifest,
            detached_launcher_sha256=expected_launcher,
        )
        envelope_raw = release.canonical_json_bytes(envelope) + b"\n"
        expected_envelope = hashlib.sha256(envelope_raw).hexdigest()

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary).resolve(strict=True)
            archive = root / "source.tar"
            manifest_path = root / "source.manifest.json"
            detached = root / LAUNCHER.name
            envelope_path = root / "deployment-envelope.json"
            archive.write_bytes(archive_raw)
            manifest_path.write_bytes(manifest_raw)
            detached.write_bytes(launcher_raw)
            envelope_path.write_bytes(envelope_raw)
            archive.chmod(0o444)
            manifest_path.chmod(0o444)
            detached.chmod(0o555)
            envelope_path.chmod(0o444)
            audited_manifest, audited_envelope = release.audit_deployment(
                archive=archive,
                expected_archive_sha256=expected_archive,
                manifest_path=manifest_path,
                expected_manifest_sha256=expected_manifest,
                detached_launcher=detached,
                expected_detached_launcher_sha256=expected_launcher,
                deployment_envelope_path=envelope_path,
                expected_deployment_envelope_sha256=expected_envelope,
            )
            self.assertEqual(audited_manifest, manifest)
            self.assertEqual(audited_envelope, envelope)

        self.assertEqual(envelope["remote_release_exact_entry_count"], 4)
        self.assertNotIn(
            release.LAUNCHER_MEMBER,
            {row["path"] for row in manifest["files"]},
        )
        launcher_source = launcher_raw.decode("utf-8")
        required_literals = {
            "core archive": expected_archive,
            "core manifest": expected_manifest,
            "manifest digest": manifest["manifest_digest"],
            "content closure": manifest["content_closure_sha1"],
            **{
                label: manifest["component_pins"][label]
                for label in (
                    "prompt_plan_sha256",
                    "generator_sha256",
                    "controller_sha256",
                    "release_builder_sha256",
                )
            },
        }
        missing_literals = {
            label: literal
            for label, literal in required_literals.items()
            if literal not in launcher_source
        }
        self.assertEqual(
            missing_literals,
            {},
            "launcher current-r6 literal pins differ",
        )
        self.assertNotIn("__EXP013", launcher_source)

        with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
            release.audit_deployment(
                archive=RELEASE_R5 / "source.tar",
                expected_archive_sha256=release.REVOKED_TERMINAL_R5_ARCHIVE_SHA256,
                manifest_path=RELEASE_R5 / "source.manifest.json",
                expected_manifest_sha256=release.REVOKED_TERMINAL_R5_MANIFEST_SHA256,
                detached_launcher=RELEASE_R5 / LAUNCHER.name,
                expected_detached_launcher_sha256=(
                    release.REVOKED_TERMINAL_R5_LAUNCHER_SHA256
                ),
                deployment_envelope_path=(
                    RELEASE_R5 / "deployment-envelope.json"
                ),
                expected_deployment_envelope_sha256=(
                    release.REVOKED_TERMINAL_R5_ENVELOPE_SHA256
                ),
            )

    def test_launcher_root_bootstrap_holds_frozen_python_fd_across_replacement(
        self,
    ) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        bootstrap_path = "/usr/bin/python3.10"
        bootstrap_sha = (
            "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
        )
        bootstrap_size = 5_937_800
        self.assertIn(
            f"readonly root_bootstrap_python={bootstrap_path}", source
        )
        self.assertIn(
            f"readonly root_bootstrap_python_sha={bootstrap_sha}", source
        )
        self.assertIn(
            f"readonly root_bootstrap_python_size={bootstrap_size}", source
        )
        self.assertIn("regular file|755|0|0|${root_bootstrap_python_size}|1", source)
        self.assertIn('os.readlink("/proc/self/exe")', source)
        self.assertIn('os.stat("/proc/self/exe")', source)
        self.assertIn("os.execve(target_fd", source)
        self.assertIn("os.execve not in os.supports_fd", source)
        self.assertIn("readonly -a frozen_python_exec_prefix=(", source)
        self.assertIn(
            '"${root_bootstrap_python}" -I -S -s -B -c '
            '"${frozen_python_fd_exec_bootstrap_py}"',
            source,
        )
        self.assertIn(
            '"${python_bin}" "${frozen_python_sha}" '
            '"${frozen_python_size}" --',
            source,
        )
        self.assertIn(
            'run_frozen_python() {\n  "${frozen_python_exec_prefix[@]}" "$@"\n}',
            source,
        )
        self.assertNotIn('exec "${python_bin}"', source)
        allowed_python_path_uses = (
            'readonly python_bin="${F13_PYTHON_BIN:',
            '"${python_bin}" == "${frozen_python_path}"',
            '"${python_bin}" && ! -L "${python_bin}"',
            'readlink -f -- "${python_bin}"',
            '"${observed_realpath}" == "${python_bin}"',
            'stat -c \'%F|%a|%u|%s|%h\' -- "${python_bin}"',
            'sha256_file "${python_bin}"',
            '"${python_bin}" "${frozen_python_sha}" "${frozen_python_size}"',
            '"${external_review}" "${python_bin}" "${ffprobe_bin}"',
            '[[ -x "${python_bin}" ]]',
            '--python "${python_bin}"',
            'F13_PYTHON_BIN="${python_bin}"',
            'run_parent_owned_command "${python_bin}"',
        )
        for line in source.splitlines():
            if "${python_bin}" in line:
                self.assertTrue(
                    any(allowed in line for allowed in allowed_python_path_uses),
                    f"ungated frozen Python path use: {line}",
                )

        marker = "read -r -d '' frozen_python_fd_exec_bootstrap_py <<'PY' || true\n"
        start = source.index(marker) + len(marker)
        end = source.index("\nPY\nreadonly frozen_python_fd_exec_bootstrap_py", start)
        bootstrap_source = source[start:end]
        for metadata_field in (
            "st_dev",
            "st_ino",
            "st_uid",
            "st_gid",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ):
            self.assertIn(metadata_field, bootstrap_source)
        self.assertEqual(bootstrap_source.count("os.read(descriptor"), 2)
        self.assertIn('getattr(os, "O_NOFOLLOW", 0)', bootstrap_source)

        target_path = release.FROZEN_PYTHON_PATH
        safe_bootstrap = b"root-owned-bootstrap-bytes"
        safe_target = b"validated-frozen-python-bytes"
        sentinel = b"REPLACEMENT_SENTINEL_MUST_NOT_EXECUTE"

        def metadata(
            inode: int, content: bytes, *, uid: int, gid: int
        ) -> SimpleNamespace:
            return SimpleNamespace(
                st_dev=7,
                st_ino=inode,
                st_uid=uid,
                st_gid=gid,
                st_mode=stat.S_IFREG | 0o755,
                st_nlink=1,
                st_size=len(content),
                st_mtime_ns=1000 + inode,
                st_ctime_ns=2000 + inode,
            )

        bootstrap_file = SimpleNamespace(
            content=safe_bootstrap,
            metadata=metadata(11, safe_bootstrap, uid=0, gid=0),
        )
        target_file = SimpleNamespace(
            content=safe_target,
            metadata=metadata(12, safe_target, uid=2012, gid=2000),
        )
        sentinel_file = SimpleNamespace(
            content=sentinel,
            metadata=metadata(13, sentinel, uid=2012, gid=2000),
        )
        named = {
            bootstrap_path: bootstrap_file,
            target_path: target_file,
        }
        descriptors: dict[int, SimpleNamespace] = {}
        next_descriptor = 40
        opened_flags: list[int] = []
        captured_exec: dict[str, object] = {}
        sentinel_exec_count = 0

        def fake_open(path: str, flags: int) -> int:
            nonlocal next_descriptor
            opened_flags.append(flags)
            descriptor = next_descriptor
            next_descriptor += 1
            descriptors[descriptor] = SimpleNamespace(
                file=named[path], offset=0, closed=False
            )
            return descriptor

        def fake_fstat(descriptor: int) -> SimpleNamespace:
            return descriptors[descriptor].file.metadata

        def fake_read(descriptor: int, size: int) -> bytes:
            held = descriptors[descriptor]
            chunk = held.file.content[held.offset : held.offset + size]
            held.offset += len(chunk)
            return chunk

        def fake_lseek(descriptor: int, offset: int, whence: int) -> int:
            self.assertEqual((offset, whence), (0, os.SEEK_SET))
            descriptors[descriptor].offset = 0
            return 0

        def fake_stat(path: str, *, follow_symlinks: bool = True) -> SimpleNamespace:
            if path == "/proc/self/exe":
                return bootstrap_file.metadata
            self.assertFalse(follow_symlinks)
            return named[path].metadata

        def fake_close(descriptor: int) -> None:
            descriptors[descriptor].closed = True

        def fake_execve(
            executable: object, arguments: list[str], environment: dict[str, str]
        ) -> None:
            nonlocal sentinel_exec_count
            if isinstance(executable, str):
                if named[executable] is sentinel_file:
                    sentinel_exec_count += 1
                executed_bytes = named[executable].content
            else:
                executed_bytes = descriptors[int(executable)].file.content
            captured_exec.update(
                {
                    "executable": executable,
                    "arguments": list(arguments),
                    "environment": dict(environment),
                    "bytes": executed_bytes,
                }
            )

        fake_os = SimpleNamespace(
            O_RDONLY=os.O_RDONLY,
            O_CLOEXEC=getattr(os, "O_CLOEXEC", 0),
            O_NOFOLLOW=getattr(os, "O_NOFOLLOW", 0),
            SEEK_SET=os.SEEK_SET,
            path=SimpleNamespace(realpath=lambda path: path),
            environ={"PATH": "/usr/bin:/bin"},
            open=fake_open,
            fstat=fake_fstat,
            read=fake_read,
            lseek=fake_lseek,
            stat=fake_stat,
            readlink=lambda path: (
                bootstrap_path
                if path == "/proc/self/exe"
                else self.fail(f"unexpected readlink: {path}")
            ),
            close=fake_close,
            execve=fake_execve,
            supports_fd={fake_execve},
        )
        fake_sys = SimpleNamespace(
            flags=SimpleNamespace(
                isolated=1,
                ignore_environment=1,
                no_site=1,
                no_user_site=1,
                dont_write_bytecode=1,
            ),
            argv=[
                "-c",
                bootstrap_path,
                hashlib.sha256(safe_bootstrap).hexdigest(),
                str(len(safe_bootstrap)),
                target_path,
                hashlib.sha256(safe_target).hexdigest(),
                str(len(safe_target)),
                "--",
                "-c",
                "replacement-sentinel-probe",
            ],
        )

        def fake_alarm(seconds: int) -> int:
            if seconds == 0:
                named[target_path] = sentinel_file
            return 0

        with mock.patch.dict(
            sys.modules, {"os": fake_os, "sys": fake_sys}
        ), mock.patch.object(
            controller.signal, "alarm", side_effect=fake_alarm
        ):
            exec(
                compile(
                    bootstrap_source,
                    "<frozen-python-fd-exec-bootstrap>",
                    "exec",
                ),
                {"__builtins__": __builtins__},
            )

        self.assertIs(named[target_path], sentinel_file)
        self.assertIsInstance(captured_exec["executable"], int)
        self.assertEqual(captured_exec["bytes"], safe_target)
        self.assertEqual(captured_exec["arguments"][0], target_path)  # type: ignore[index]
        self.assertEqual(sentinel_exec_count, 0)
        self.assertTrue(all(flags & fake_os.O_NOFOLLOW for flags in opened_flags))

    def test_deployment_envelope_exact4_and_cross_version_hostiles(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT.resolve(strict=True))
        archive_sha = hashlib.sha256(
            release.build_archive(manifest, payloads)
        ).hexdigest()
        manifest_sha = hashlib.sha256(
            release.canonical_json_bytes(manifest) + b"\n"
        ).hexdigest()
        launcher_sha = hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()
        envelope = release.deployment_envelope_value(
            manifest=manifest,
            archive_sha256=archive_sha,
            manifest_sha256=manifest_sha,
            detached_launcher_sha256=launcher_sha,
        )
        self.assertEqual(envelope["remote_release_exact_entry_count"], 4)
        self.assertEqual(
            envelope["remote_release_exact_entries"],
            [
                "source.tar",
                "source.manifest.json",
                LAUNCHER.name,
                "deployment-envelope.json",
            ],
        )
        self.assertFalse(envelope["remote_release_extra_entries_allowed"])
        self.assertEqual(
            envelope["detached_launcher"]["expected_mode_octal"], "0555"
        )
        self.assertTrue(
            envelope["detached_launcher"][
                "excluded_from_core_archive_and_manifest"
            ]
        )
        self.assertEqual(
            release.validate_deployment_envelope(
                envelope,
                manifest=manifest,
                archive_sha256=archive_sha,
                manifest_sha256=manifest_sha,
                detached_launcher_sha256=launcher_sha,
            ),
            envelope,
        )

        hostile_extra = copy.deepcopy(envelope)
        hostile_extra.pop("envelope_digest")
        hostile_extra["extra"] = True
        hostile_extra["envelope_digest"] = release.object_sha256(hostile_extra)
        hostile_cases = (
            (hostile_extra, archive_sha, manifest_sha, launcher_sha),
            (
                envelope,
                release.REVOKED_PORTABLE_R2_ARCHIVE_SHA256,
                manifest_sha,
                launcher_sha,
            ),
            (
                envelope,
                archive_sha,
                release.REVOKED_PORTABLE_R2_MANIFEST_SHA256,
                launcher_sha,
            ),
            (
                envelope,
                archive_sha,
                manifest_sha,
                release.REVOKED_PORTABLE_R2_LAUNCHER_SHA256,
            ),
            (
                envelope,
                release.REVOKED_TERMINAL_R3_ARCHIVE_SHA256,
                manifest_sha,
                launcher_sha,
            ),
            (
                envelope,
                archive_sha,
                release.REVOKED_TERMINAL_R3_MANIFEST_SHA256,
                launcher_sha,
            ),
            (
                envelope,
                archive_sha,
                manifest_sha,
                release.REVOKED_TERMINAL_R3_LAUNCHER_SHA256,
            ),
            (
                envelope,
                release.REVOKED_RESOURCE_REUSE_R4_ARCHIVE_SHA256,
                manifest_sha,
                launcher_sha,
            ),
            (
                envelope,
                archive_sha,
                release.REVOKED_RESOURCE_REUSE_R4_MANIFEST_SHA256,
                launcher_sha,
            ),
            (
                envelope,
                archive_sha,
                manifest_sha,
                release.REVOKED_RESOURCE_REUSE_R4_LAUNCHER_SHA256,
            ),
        )
        for value, candidate_archive, candidate_manifest, candidate_launcher in hostile_cases:
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                release.validate_deployment_envelope(
                    value,
                    manifest=manifest,
                    archive_sha256=candidate_archive,
                    manifest_sha256=candidate_manifest,
                    detached_launcher_sha256=candidate_launcher,
                )

    def test_deployment_audit_rejects_renamed_old_pairs_and_topology(self) -> None:
        current_manifest, current_payloads = release.build_manifest(
            METHOD_ROOT.resolve(strict=True)
        )
        current_archive_raw = release.build_archive(
            current_manifest, current_payloads
        )
        current_manifest_raw = (
            release.canonical_json_bytes(current_manifest) + b"\n"
        )
        current_launcher_raw = LAUNCHER.read_bytes()
        expected_archive = hashlib.sha256(current_archive_raw).hexdigest()
        expected_manifest = hashlib.sha256(current_manifest_raw).hexdigest()
        expected_launcher = hashlib.sha256(current_launcher_raw).hexdigest()
        current_envelope = release.deployment_envelope_value(
            manifest=current_manifest,
            archive_sha256=expected_archive,
            manifest_sha256=expected_manifest,
            detached_launcher_sha256=expected_launcher,
        )
        current_envelope_raw = (
            release.canonical_json_bytes(current_envelope) + b"\n"
        )
        expected_envelope = hashlib.sha256(current_envelope_raw).hexdigest()
        r2 = METHOD_ROOT / "releases/full30_action_arms_incomplete_repair_exact2_r2"
        with tarfile.open(r2 / "source.tar", "r:") as archive:
            member = archive.getmember(
                f"{release.MEMBER_ROOT}/{release.LAUNCHER_MEMBER}"
            )
            handle = archive.extractfile(member)
            self.assertIsNotNone(handle)
            old_launcher_raw = handle.read()

        def stage(
            root: Path,
            *,
            core_release: Path | None = None,
            launcher_raw: bytes | None = None,
            envelope_release: Path | None = None,
        ) -> tuple[Path, Path, Path, Path]:
            archive = root / "source.tar"
            manifest = root / "source.manifest.json"
            launcher = root / LAUNCHER.name
            envelope = root / "deployment-envelope.json"
            if core_release is None:
                archive.write_bytes(current_archive_raw)
                manifest.write_bytes(current_manifest_raw)
            else:
                shutil.copyfile(core_release / "source.tar", archive)
                shutil.copyfile(
                    core_release / "source.manifest.json", manifest
                )
            launcher.write_bytes(
                current_launcher_raw if launcher_raw is None else launcher_raw
            )
            if envelope_release is None:
                envelope.write_bytes(current_envelope_raw)
            else:
                shutil.copyfile(
                    envelope_release / "deployment-envelope.json", envelope
                )
            archive.chmod(0o444)
            manifest.chmod(0o444)
            launcher.chmod(0o555)
            envelope.chmod(0o444)
            return archive, manifest, launcher, envelope

        def audit_staged(paths: tuple[Path, Path, Path, Path]) -> None:
            archive, manifest, launcher, envelope = paths
            release.audit_deployment(
                archive=archive,
                expected_archive_sha256=expected_archive,
                manifest_path=manifest,
                expected_manifest_sha256=expected_manifest,
                detached_launcher=launcher,
                expected_detached_launcher_sha256=expected_launcher,
                deployment_envelope_path=envelope,
                expected_deployment_envelope_sha256=expected_envelope,
            )

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            audit_staged(stage(Path(temporary).resolve(strict=True)))
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(stage(root, core_release=r2))
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(stage(root, launcher_raw=old_launcher_raw))
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(
                    stage(root, core_release=RESOURCE_FIXTURE_RELEASE)
                )
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(
                    stage(
                        root,
                        launcher_raw=(
                            RESOURCE_FIXTURE_RELEASE / LAUNCHER.name
                        ).read_bytes(),
                    )
                )
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(
                    stage(root, envelope_release=RESOURCE_FIXTURE_RELEASE)
                )
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(stage(root, core_release=RELEASE_R4))
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(
                    stage(
                        root,
                        launcher_raw=(RELEASE_R4 / LAUNCHER.name).read_bytes(),
                    )
                )
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(stage(root, envelope_release=RELEASE_R4))
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(stage(root, core_release=RELEASE_R5))
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(
                    stage(
                        root,
                        launcher_raw=(RELEASE_R5 / LAUNCHER.name).read_bytes(),
                    )
                )
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(stage(root, envelope_release=RELEASE_R5))
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            paths = stage(root)
            (root / "extra").write_bytes(b"forbidden")
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(paths)
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            paths = stage(root)
            paths[2].chmod(0o755)
            with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                audit_staged(paths)
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            paths = stage(root)
            target = root.parent / f"{root.name}-launcher-target"
            target.write_bytes(paths[2].read_bytes())
            target.chmod(0o555)
            paths[2].unlink()
            paths[2].symlink_to(target)
            try:
                with self.assertRaises(release.ArmsIncompleteExact2ReleaseError):
                    audit_staged(paths)
            finally:
                target.unlink()

    def test_launcher_is_exact2_parent_safe_and_syntax_valid(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        controller_source = Path(controller.__file__).read_text(encoding="utf-8")
        generator_source = Path(generator.__file__).read_text(encoding="utf-8")
        self.assertIn("readonly holder_job=136140", source)
        self.assertIn("readonly holder_node=auh7-1b-gpu-215", source)
        self.assertIn("--gpus-per-task=8", source)
        self.assertIn("--cpus-per-task=32 --mem=60G", source)
        self.assertEqual(source.count('"${generator}" run-sp4'), 1)
        self.assertIn("seal-blind-review-input", source)
        self.assertEqual(source.count("seal-child-terminal-ready"), 1)
        self.assertEqual(source.count("prepare-parent-generation-status"), 1)
        self.assertEqual(
            source.count("resident-publish-parent-generation-status"), 1
        )
        self.assertIn(controller.PARENT_GENERATION_PUBLISH_READY_PREFIX, source)
        self.assertIn(controller.PARENT_GENERATION_PUBLISH_ACK_PREFIX, source)
        self.assertIn(
            controller.PARENT_GENERATION_PUBLISH_COMMIT_TOKEN.decode("ascii").strip(),
            source,
        )
        self.assertIn(controller.CHILD_TERMINAL_READY_SCHEMA, controller_source)
        self.assertIn(controller.PARENT_GENERATION_STATUS_SCHEMA, controller_source)
        self.assertIn(f"readonly ffprobe_bin={controller.PORTABLE_FFPROBE_PATH}", source)
        self.assertIn(f"readonly ffprobe_sha={controller.PORTABLE_FFPROBE_SHA256}", source)
        self.assertNotIn("F13_FFPROBE", source)
        self.assertNotIn("/usr/bin/ffprobe", source)
        self.assertIn("prepare-child-scratch", source)
        self.assertIn("seal-compute-preflight", source)
        self.assertIn("create-and-bind-child-task-scratch", source)
        self.assertIn("formal_00.mp4", source)
        self.assertIn("formal_02.mp4", source)
        self.assertIn("stat -f -c '%T'", source)
        self.assertNotIn('${SLURM_TMPDIR:?Slurm node-local scratch required}', source)
        self.assertIn(
            "env -u SLURM_TMPDIR -u TMPDIR -u GADP_NODE_LOCAL_SCRATCH "
            "-u GADP_NODE_LOCAL_SCRATCH_FSTYPE",
            source,
        )
        self.assertIn(
            'GADP_NODE_LOCAL_SCRATCH_FSTYPE="${receipt_scratch_fstype}"',
            source,
        )
        self.assertIn("rank_resource_scratch_binding", controller_source)
        self.assertIn("rank_resource_scratch_binding", generator_source)
        self.assertIn(generator.RESOURCE_SPECIALIZED_SHA256, source)
        self.assertNotIn(generator.REVOKED_R5_RESOURCE_SPECIALIZED_SHA256, source)
        child_source = source[source.rindex('if [[ "${role}" == child ]]'):]
        portable_sha_index = child_source.index(
            'sha256_file "${ffprobe_bin}"'
        )
        prepare_index = child_source.index("prepare-child-scratch")
        preflight_index = child_source.index("seal-compute-preflight")
        task_bind_index = child_source.index("create-and-bind-child-task-scratch")
        sealed_source_index = child_source.index(
            "validate_sealed_inputs_after_compute_preflight"
        )
        self.assertLess(portable_sha_index, prepare_index)
        self.assertLess(prepare_index, preflight_index)
        self.assertLess(preflight_index, sealed_source_index)
        self.assertLess(preflight_index, task_bind_index)
        self.assertLess(task_bind_index, child_source.index("export GADP_NODE_LOCAL_SCRATCH"))
        self.assertLess(preflight_index, child_source.index("host-memory-monitor"))
        self.assertLess(preflight_index, child_source.index("smoke-sp4"))
        self.assertLess(
            preflight_index, child_source.index('"${generator}" run-sp4')
        )
        preflight_function = controller_source[
            controller_source.index("def seal_compute_preflight"):
            controller_source.index("def validate_compute_preflight")
        ]
        self.assertLess(
            preflight_function.index("validate_ffprobe"),
            preflight_function.index("media_paths"),
        )
        self.assertLess(
            preflight_function.index("media_paths"),
            preflight_function.index("load_controller_plan"),
        )
        self.assertLess(
            child_source.index("seal-child-scratch-retained-terminal"),
            child_source.index("seal-child-terminal-ready"),
        )
        parent_source = source[: source.rindex('if [[ "${role}" == child ]]')]
        self.assertLess(
            source.index("validate-child-scratch-retained-terminal"),
            source.index("prepare-parent-generation-status"),
        )
        self.assertLess(
            source.index("prepare-parent-generation-status"),
            source.index("resident-publish-parent-generation-status"),
        )
        self.assertIn(
            "revoked live run root or descendant is permanent NO-GO", source
        )
        self.assertIn("revoked live release is permanent NO-GO", source)
        self.assertTrue(
            controller.REVOKED_LIVE_LOG_SHA256 in source,
            "launcher omits controller's current revoked-live-log identity",
        )
        self.assertIn("readlink -m -- \"${run_root}\"", source)
        self.assertIn(
            'case "${run_root_canonical}" in "${revoked_run_root}"|"${revoked_run_root}"/*)',
            source,
        )
        self.assertIn(release.REVOKED_PORTABLE_R2_ARCHIVE_SHA256, source)
        self.assertIn(release.REVOKED_PORTABLE_R2_MANIFEST_SHA256, source)
        for revoked_r3 in (
            release.REVOKED_TERMINAL_R3_ARCHIVE_SHA256,
            release.REVOKED_TERMINAL_R3_MANIFEST_SHA256,
            release.REVOKED_TERMINAL_R3_LAUNCHER_SHA256,
            release.REVOKED_TERMINAL_R3_ENVELOPE_SHA256,
        ):
            self.assertIn(revoked_r3, source)
        for revoked_r4 in (
            release.REVOKED_RESOURCE_REUSE_R4_ARCHIVE_SHA256,
            release.REVOKED_RESOURCE_REUSE_R4_MANIFEST_SHA256,
            release.REVOKED_RESOURCE_REUSE_R4_LAUNCHER_SHA256,
            release.REVOKED_RESOURCE_REUSE_R4_ENVELOPE_SHA256,
            release.REVOKED_RESOURCE_REUSE_R4_RELEASE_ROOT,
            release.REVOKED_RESOURCE_REUSE_R4_MATERIALIZATION_ROOT,
            release.REVOKED_RESOURCE_REUSE_R4_RUN_ROOT,
        ):
            self.assertIn(revoked_r4, source)
        for revoked_r5 in (
            release.REVOKED_TERMINAL_R5_ARCHIVE_SHA256,
            release.REVOKED_TERMINAL_R5_MANIFEST_SHA256,
            release.REVOKED_TERMINAL_R5_LAUNCHER_SHA256,
            release.REVOKED_TERMINAL_R5_ENVELOPE_SHA256,
        ):
            self.assertIn(revoked_r5, source)
        self.assertIn("terminal-physical r3 core is permanent NO-GO", source)
        self.assertIn(
            "terminal-physical r3 launcher/envelope is permanent NO-GO",
            source,
        )
        self.assertIn(
            "resource-reuse r4 launcher/envelope is permanent NO-GO", source
        )
        self.assertIn(
            "resource-reuse r4 release/materialization/run subtree is permanent NO-GO",
            source,
        )
        self.assertIn(
            "resource-reuse r4 core is permanent NO-GO even if renamed", source
        )
        for python_literal in (
            f"readonly frozen_python_path={release.FROZEN_PYTHON_PATH}",
            f"readonly frozen_python_realpath={release.FROZEN_PYTHON_REALPATH}",
            f"readonly frozen_python_file_type='{release.FROZEN_PYTHON_FILE_TYPE}'",
            "readonly frozen_python_mode=755",
            f"readonly frozen_python_uid={release.FROZEN_PYTHON_UID}",
            f"readonly frozen_python_size={release.FROZEN_PYTHON_SIZE}",
            f"readonly frozen_python_nlink={release.FROZEN_PYTHON_LINK_COUNT}",
            f"readonly frozen_python_sha={release.FROZEN_PYTHON_SHA256}",
            f"readonly rejected_python_symlink={release.REJECTED_PYTHON_SYMLINK_PATH}",
        ):
            self.assertIn(python_literal, source)
        self.assertIn(
            '[[ "${python_bin}" == "${frozen_python_path}" ]]', source
        )
        self.assertIn(
            '[[ -f "${python_bin}" && ! -L "${python_bin}" ]]', source
        )
        self.assertIn("stat -c '%F|%a|%u|%s|%h'", source)
        self.assertIn('sha256_file "${python_bin}"', source)
        self.assertIn(
            'F13_PARENT_DETACHED_LAUNCHER_PATH="${detached_launcher}"', source
        )
        self.assertIn(
            'F13_PARENT_DETACHED_LAUNCHER_SHA256="${detached_launcher_sha}"',
            source,
        )
        self.assertIn('bash "${launcher}" __child', source)
        self.assertEqual(source.count('readonly launcher="${detached_launcher}"'), 1)
        parent_preflight = source.index(
            'if [[ "${role}" == parent ]]; then\n  validate_sealed_inputs_after_compute_preflight'
        )
        python_preflight = source.index(
            "validate_frozen_python_before_run_root_or_srun\n"
        )
        r4_path_preflight = source.index(
            "for name in run_root release_root method_root method_archive method_manifest deployment_envelope; do"
        )
        run_freshness = source.index(
            '[[ ! -e "${run_root}" && ! -L "${run_root}"'
        )
        run_mkdir = source.index('mkdir -m 0700 "${run_root}"')
        srun_index = source.index("srun --jobid=")
        self.assertLess(python_preflight, run_freshness)
        self.assertLess(python_preflight, run_mkdir)
        self.assertLess(python_preflight, srun_index)
        self.assertLess(r4_path_preflight, run_freshness)
        self.assertLess(r4_path_preflight, run_mkdir)
        self.assertLess(r4_path_preflight, srun_index)
        self.assertLess(parent_preflight, run_freshness)
        self.assertLess(parent_preflight, run_mkdir)
        self.assertGreaterEqual(source.count("assert_retained_parent_unchanged"), 3)
        for forbidden_parent_mutation in (
            "scancel",
            "scontrol release",
            "scontrol requeue",
            'kill "${holder_job}"',
            'kill -- "${holder_job}"',
        ):
            self.assertNotIn(forbidden_parent_mutation, source)
        self.assertNotIn(controller.CHILD_TERMINAL_READY_SCHEMA, source)
        self.assertNotIn(controller.PARENT_GENERATION_STATUS_SCHEMA, source)
        for forbidden in ("--lane diagnostic", "optimizer.step"):
            self.assertNotIn(forbidden, source)
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_hostile_python_aliases_fail_before_run_root_or_srun(self) -> None:
        original = LAUNCHER.read_text(encoding="utf-8")
        required_paths = {
            "F13_RELEASE_ROOT": "/private/tmp/f13-release",
            "F13_METHOD_ROOT": "/private/tmp/f13-method",
            "F13_METHOD_ARCHIVE": "/private/tmp/f13-release/source.tar",
            "F13_METHOD_MANIFEST": "/private/tmp/f13-release/source.manifest.json",
            "F13_DEPLOYMENT_ENVELOPE": "/private/tmp/f13-release/deployment-envelope.json",
            "F13_SEED1_SPEC": "/private/tmp/f13-seed1.json",
            "F13_SEED2_SPEC": "/private/tmp/f13-seed2.json",
            "F13_EXTERNAL_EVIDENCE_ROOT": "/private/tmp/f13-external",
            "F13_EXTERNAL_KEY": "/private/tmp/f13-key.json",
            "F13_EXTERNAL_REVIEW_RECEIPT": "/private/tmp/f13-review.json",
            "F13_BERNINI_ROOT": "/private/tmp/f13-bernini",
            "F13_VEOMNI_ROOT": "/private/tmp/f13-veomni",
            "F13_CHECKPOINT": "/private/tmp/f13-checkpoint",
            "F13_CHECKPOINT_MANIFEST": "/private/tmp/f13-checkpoint.json",
            "F13_R10_COMPILE_SMOKE_RECEIPT": "/private/tmp/f13-r10.json",
            "F13_R10_GENERATION_LOG": "/private/tmp/f13-r10.log",
        }

        def run_hostile(
            root: Path, *, candidate: Path, replace_expected: bool
        ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
            run_root = root / "must-remain-absent"
            srun_marker = root / "srun-called"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_srun = fake_bin / "srun"
            fake_srun.write_text(
                f"#!/bin/sh\n: > {srun_marker}\nexit 99\n", encoding="utf-8"
            )
            fake_srun.chmod(0o755)
            launcher_source = (
                original.replace(release.FROZEN_PYTHON_PATH, str(candidate))
                if replace_expected
                else original
            )
            staged = root / "launcher.sh"
            staged.write_text(launcher_source, encoding="utf-8")
            staged.chmod(0o555)
            env = os.environ.copy()
            env.update(required_paths)
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "HOME": "/vast/users/guangyi.chen",
                    "USER": "guangyi.chen",
                    "LOGNAME": "guangyi.chen",
                    "F13_CONFIRM": (
                        "launch-approved-BOX-EXP-013-arms-incomplete-"
                        "repair-exact2-r6-136140"
                    ),
                    "F13_ENTRY_ENVIRONMENT": "clean-env-i-bash-p-v1",
                    "F13_RUN_ROOT": str(run_root),
                    "F13_DEPLOYMENT_ENVELOPE_SHA256": "1" * 64,
                    "F13_DETACHED_LAUNCHER_SHA256": "2" * 64,
                    "F13_PYTHON_BIN": str(candidate),
                    "F13_MASTER_PORT": "38142",
                    "F13_R10_COMPILE_SMOKE_RECEIPT_SHA256": "3" * 64,
                    "F13_R10_GENERATION_LOG_SHA256": "4" * 64,
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    "--noprofile",
                    "--norc",
                    "-p",
                    str(staged),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            return completed, run_root, srun_marker

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            completed, run_root, marker = run_hostile(
                root, candidate=Path("/bin/true"), replace_expected=False
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("alternate Python interpreter is forbidden", completed.stderr)
            self.assertFalse(run_root.exists())
            self.assertFalse(marker.exists())

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            target = root / "python3.12-target"
            target.write_bytes(b"not the frozen interpreter")
            target.chmod(0o755)
            alias = root / "python"
            alias.symlink_to(target.name)
            completed, run_root, marker = run_hostile(
                root, candidate=alias, replace_expected=True
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("non-symlink regular file", completed.stderr)
            self.assertFalse(run_root.exists())
            self.assertFalse(marker.exists())

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            drifted = root / "python3.12"
            drifted.write_bytes(b"drifted executable bytes")
            drifted.chmod(0o755)
            completed, run_root, marker = run_hostile(
                root, candidate=drifted, replace_expected=True
            )
            self.assertNotEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(run_root.exists())
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()

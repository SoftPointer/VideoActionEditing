#!/usr/bin/env python3
"""CPU-only fail-closed preflight for the Round37 native activation canary.

This process never imports Torch, initializes distributed state, loads model
weights, or writes output.  A caller cannot provide an expected trust hash.
Every release pin is compiled below from the independent packet/ledger and the
final runtime, runner, materializer, and spec bytes.  No caller or environment
value can replace these roots.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-oracle-regeneration-native-activation-v2-preflight-r11"
SPEC_SCHEMA_VERSION = "bernini-oracle-regeneration-native-activation-v2-spec-r11"
METHOD_ROOT = Path(__file__).resolve().parent
SPEC_PATH = METHOD_ROOT / "assets/oracle_regeneration_native_activation_v2_r2_spec.json"
RUNTIME_PATH = METHOD_ROOT / "oracle_regeneration_native_runtime_activation_v2.py"
RUNNER_PATH = METHOD_ROOT / "infer_oracle_regeneration_native_activation_v2_r2.py"
VAE_TOOL_PATH = (
    METHOD_ROOT / "tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py"
)
PROMPT_TOOL_PATH = (
    METHOD_ROOT / "tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py"
)
_COMPONENT_RELATIVE_PATHS = {
    "activation_runtime": "oracle_regeneration_native_runtime_activation_v2.py",
    "native_runner": "infer_oracle_regeneration_native_activation_v2_r2.py",
    "vae_reference_materializer": (
        "tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py"
    ),
    "prompt_materializer": (
        "tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py"
    ),
}
_RUNTIME_DEPENDENCY_RELATIVE_PATHS = {
    "oracle_regeneration_canary_v1.py": "oracle_regeneration_canary_v1.py",
    "native_branch_homotopy_runtime_v1.py": "native_branch_homotopy_runtime_v1.py",
    "native_branch_homotopy_v1.py": "native_branch_homotopy_v1.py",
    "self_guided_action_field_v1.py": "self_guided_action_field_v1.py",
    "tri_branch_unipc.py": "tri_branch_unipc.py",
    "infer_native_identity_generation_canary.py": (
        "infer_native_identity_generation_canary.py"
    ),
    "infer_native_branch_homotopy_canary.py": (
        "infer_native_branch_homotopy_canary.py"
    ),
    "infer_source_kv_carrier_oracle.py": "infer_source_kv_carrier_oracle.py",
    "infer_source_value_residual_oracle.py": (
        "infer_source_value_residual_oracle.py"
    ),
    "infer_native_self_guided_action_field_canary.py": (
        "infer_native_self_guided_action_field_canary.py"
    ),
    "infer_lora.py": "infer_lora.py",
    "tools/materialize_vae.py": "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py": "tools/build_renderer_dataset.py",
    "tools/__init__.py": "tools/__init__.py",
    "train_lora.py": "train_lora.py",
    "action_preservation_decoded_eval_model_authority_v2.py": (
        "action_preservation_decoded_eval_model_authority_v2.py"
    ),
    "source_kv_replay.py": "source_kv_replay.py",
    "source_kv_route_batches.py": "source_kv_route_batches.py",
    "source_self_native_ref_contrastive_v3.py": (
        "source_self_native_ref_contrastive_v3.py"
    ),
    "source_value_residual.py": "source_value_residual.py",
}
_MIOPEN_CACHE_SUFFIX = ".miopen-cache-r9"
_MIOPEN_CACHE_ROOT_ENV = "ORACLE_ACTIVATION_V2_MIOPEN_CACHE_ROOT"
_MIOPEN_LOCAL_TMP_ROOT_ENV = "ORACLE_ACTIVATION_V2_MIOPEN_LOCAL_TMP_ROOT"
_MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV = (
    "ORACLE_ACTIVATION_V2_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_SHA256"
)
_MIOPEN_OUTPUT_ENV = "ORACLE_ACTIVATION_V2_OUTPUT_DIR"
_MIOPEN_USER_DB_ENV = "MIOPEN_USER_DB_PATH"
_MIOPEN_KERNEL_CACHE_ENV = "MIOPEN_CUSTOM_CACHE_DIR"
_MIOPEN_TEMP_ENV_NAMES = ("TMPDIR", "TMP", "TEMP", "TEMPDIR")
_MIOPEN_LOCAL_TMP_PARENT = Path("/tmp")
_MIOPEN_LOCAL_TMP_DOMAIN = (
    "bernini-oracle-regeneration-native-activation-v2-local-tmp-r9"
)
_SCHEDULER_TMPDIR_NORMALIZATION_ENV = {
    "job_id": "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_JOB_ID",
    "step_id": "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_STEP_ID",
    "hostname": "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_HOSTNAME",
    "observed_tmpdir": "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_OBSERVED",
    "action": "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_ACTION",
}
_SCHEDULER_TMPDIR_NORMALIZATION_ACTION = "UNSET_BEFORE_ANY_PYTHON"
_MIOPEN_LIBRARY_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/torch/lib/libMIOpen.so"
)
_MIOPEN_LIBRARY_SHA256 = (
    "1e6cc33ca21951dce12795e6c5d99578e8f2f1754b84a703508df44426b44b52"
)
_MIOPEN_LIBRARY_SIZE = 690355265
_MIOPEN_CACHE_DIRECTORY_NAMES = {
    *(f"launcher-bootstrap-{role}" for role in ("user-db", "kernel-cache")),
    *(
        f"rank-{rank}-{role}"
        for rank in range(4)
        for role in ("user-db", "kernel-cache")
    ),
}
_MIOPEN_LOCAL_TMP_DIRECTORY_NAMES = {f"rank-{rank}" for rank in range(4)}


def _launcher_local_tmp_empty_proof(
    root_identity: Mapping[str, Any],
    directory_identities: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Recompute the frozen launcher's fresh/empty layout digest."""

    identity_keys = {"path", "mode", "uid", "gid", "device", "inode"}
    if (
        not isinstance(root_identity, Mapping)
        or set(root_identity) != identity_keys
        or not isinstance(directory_identities, Mapping)
        or set(directory_identities) != _MIOPEN_LOCAL_TMP_DIRECTORY_NAMES
        or any(
            not isinstance(directory_identities[name], Mapping)
            or set(directory_identities[name]) != identity_keys
            for name in _MIOPEN_LOCAL_TMP_DIRECTORY_NAMES
        )
    ):
        raise ActivationPreflightError(
            "launcher node-local tmp empty proof identity differs"
        )

    def line(kind: str, name: str, row: Mapping[str, Any], empty: str) -> str:
        return "\t".join(
            (
                kind,
                name,
                str(row["path"]),
                format(int(row["mode"]), "o"),
                str(int(row["uid"])),
                str(int(row["gid"])),
                str(int(row["device"])),
                str(int(row["inode"])),
                empty,
            )
        )

    payload = "\n".join(
        [
            "bernini-launcher-node-local-tmp-fresh-empty-proof-r11",
            line("root", "root", root_identity, "roles-only"),
            *(
                line("role", name, directory_identities[name], "empty")
                for name in sorted(_MIOPEN_LOCAL_TMP_DIRECTORY_NAMES)
            ),
        ]
    ).encode("utf-8")
    return {
        "schema_version": (
            "bernini-launcher-node-local-tmp-fresh-empty-proof-r11"
        ),
        "environment_key": _MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "fresh_root_created_by_frozen_launcher": True,
        "exact_rank_directory_set_observed_by_frozen_launcher": True,
        "all_rank_directories_observed_empty_by_frozen_launcher": True,
        "not_an_independent_signature": True,
    }
_SPEC_KEYS = {
    "schema_version",
    "status",
    "launch_ready",
    "scope",
    "authority_kind",
    "formal_authority",
    "training_authority",
    "training",
    "optimizer",
    "automatic_replacement",
    "selection_authority",
    "native_only",
    "flowedit",
    "connected_route",
    "learned_gate",
    "world_size",
    "sequence_parallel_size",
    "one_node",
    "candidate_count_per_arm",
    "host_load_contract",
    "allocation_contract",
    "miopen_cache_contract",
    "runner_allowlist",
    "components",
    "frozen_runtime_dependencies",
    "compiled_authority_packet_sha256",
    "compiled_external_ledger_receipt_sha256",
    "cases",
    "scientific_boundary",
    "post_run_contract",
    "mandatory_blockers",
}

_EXPECTED_HOST_LOAD_CONTRACT = {
    "serialized_host_load_required": True,
    "shared_lock_absolute_empty_frozen_one_link": True,
    "world4_lock_identity_exact": True,
    "unused_t5_constructor_bypassed_all_ranks": True,
    "prompt_embeddings_rank0_materialized_and_world4_broadcast": True,
    "same_bypass_for_e02_official_and_local_arms": True,
}
_EXPECTED_ALLOCATION_CONTRACT = {
    "slurm_job_id": "141620",
    "hostname": "auh7-1b-gpu-226",
    "numbered_slurm_step_required": True,
    "scheduler_injected_tmpdir": "/tmp",
    "scheduler_tmpdir_unset_before_any_python": True,
    "scheduler_normalization_receipt_environment_keys": sorted(
        _SCHEDULER_TMPDIR_NORMALIZATION_ENV.values()
    ),
    "allocated_gpu_count": 8,
    "world4_visible_gpu_indices": [0, 1, 2, 3],
    "intentionally_idle_gpu_indices": [4, 5, 6, 7],
    "e02_arms_execute_strictly_serial": True,
}
_EXPECTED_MIOPEN_CACHE_CONTRACT = {
    "cache_root": "exact fresh output sibling with suffix .miopen-cache-r9",
    "launcher_bootstrap_roles": ["user-db", "kernel-cache"],
    "rank_private_persistent_roles": ["user-db", "kernel-cache"],
    "rank_private_node_local_tmp_role": True,
    "node_local_tmp_parent": "/tmp",
    "node_local_tmp_domain_includes": [
        "fixed-domain",
        "effective-uid",
        "slurm-job-id",
        "slurm-step-id",
        "canonical-output-path",
    ],
    "persistent_and_node_local_filesystems_must_differ": True,
    "rank_count": 4,
    "root_and_role_directory_mode": "0700",
    "effective_user_owned": True,
    "symlinks_allowed": False,
    "caller_miopen_environment_must_initially_be_empty": True,
    "launcher_mints_exact_miopen_environment_keys": [
        "MIOPEN_CUSTOM_CACHE_DIR",
        "MIOPEN_USER_DB_PATH",
    ],
    "launcher_sets_bootstrap_before_any_torch_import": True,
    "workers_switch_to_rank_private_paths_before_their_torch_import": True,
    "same_rank_namespace_reused_by_serial_arms": True,
    "torchrun_parent_temp_environment_unset": ["TEMP", "TEMPDIR", "TMP", "TMPDIR"],
    "rank_private_tmpdir_isolates_miopen_lockfiles": True,
    "miopen_disable_cache_used": False,
    "miopen_system_db_override_used": False,
    "library": {
        "path": str(_MIOPEN_LIBRARY_PATH),
        "sha256": _MIOPEN_LIBRARY_SHA256,
        "size": _MIOPEN_LIBRARY_SIZE,
        "mode": "0755",
        "nlink": 1,
    },
    "retained_engineering_evidence": True,
    "node_local_tmp_runner_cleanup_performed": False,
    "node_local_tmp_durability_guaranteed": False,
    "node_local_tmp_node_lifetime_only": True,
    "node_local_tmp_pre_torch_empty_gate_layers": [
        "launcher",
        "cpu-preflight",
        "worker-activation",
    ],
    "node_local_tmp_pre_torch_rank_directories_empty_all_ranks": True,
    "node_local_tmp_post_runtime_init_baseline_may_be_nonempty": True,
    "node_local_tmp_post_runtime_init_baseline_strictly_scanned_and_quiescent": True,
    "node_local_tmp_post_runtime_init_baseline_bound_in_receipt": True,
    "node_local_tmp_baseline_is_observation_not_allowlist": True,
    "node_local_tmp_differences_at_observation_boundaries_recorded_not_forbidden": True,
    "node_local_tmp_continuous_monitoring_claimed": False,
    "node_local_tmp_between_observation_transients_may_be_unrecorded": True,
    "node_local_tmp_full_receipt_embedded_in_durable_output_receipt": True,
    "node_local_tmp_observed_and_replayed_before_WORLD4_step_exit": True,
    "node_local_tmp_existence_after_process_or_step_exit_guaranteed": False,
    "scientific_output_artifact": False,
    "explicit_solver_control_environment_override_used": False,
    "matched_arms_share_cache_lookup_state": True,
}
_EXPECTED_SPEC_CASES = {
    "e02": {
        "decision": "ACTIVE_DIAGNOSTIC",
        "executed": True,
        "seed": 0,
        "arms": [
            "official-v2v-base",
            "local-source-reference-r2v4-in-manual-G",
        ],
        "gate_channels": ["D", "C", "K", "G"],
        "hard_support": "G=D|C|K",
        "expected_nonempty_phase_windows": {
            "D": [1, 20],
            "C": [5, 20],
            "K": [4, 20],
            "G": [1, 20],
        },
        "maximum_per_phase_G_fraction": 0.3,
        "gate_author_kind": "AI_AGENT",
        "gate_reviewer_kind": "AI_AGENT",
        "gate_manifest_sha256": (
            "f5c0e20f478ff11a63fa5c1ecb1d2bfb4a37e655e14f0e2c9739975a3e63dae1"
        ),
        "gate_review_receipt_sha256": (
            "4e060227512687060a32acc0543462be4dfe0364c46e5fe6e7e02416dac93962"
        ),
        "vae_reference_receipt_sha256": (
            "b89840c3f87a0950e5d0634c3697eed7641ae06e840d1804ea30d2ebaf74611f"
        ),
        "vae_authoring_run_receipt_sha256": (
            "65b3acc8e2581bcc7c3311475c95bada1fd25067328627542572933fd930bbb7"
        ),
        "prompt_receipt_sha256": (
            "ed6e11b50651b6d30f067c4dc9d285a3b790658edfd82ac9e0e0df7c3d290924"
        ),
        "prompt_authoring_run_receipt_sha256": (
            "2fd4a232db890dde19cf26fa4596be1279c8756dfa60a943837705aece69da9a"
        ),
        "materialization_review_receipt_sha256": (
            "788cf7b83851b79a662c0040e7470fd32a8aea2456ce47e6b53baff0b2a73c6e"
        ),
    },
    "e03": {
        "decision": "ABSTAIN_KEEP_BASE",
        "executed": False,
        "seed": 0,
        "arms": [],
        "active_gate": None,
        "automatic_replacement": False,
        "strict_non_regression_required_for_any_later_selection": True,
        "vae_reference_receipt": None,
        "prompt_receipt": None,
        "kept_frozen_base": {
            "path": (
                "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
                "VideoEditing/VideoEdit_experiments/"
                "action_flow_noise_stage0_job140846_v1/stage1/"
                "interaction_complex8_rv2v_candidates_v1/"
                "complex8-e03-rv2v-s0/rv2v.mp4"
            ),
            "sha256": (
                "d75bbafbbc225ea3935c2d149be8b3969fffd6d8b645c5ec9edb5968bf25f654"
            ),
        },
    },
}
_EXPECTED_SCIENTIFIC_BOUNDARY = {
    "source_reference_r2v4_regeneration_expert": True,
    "self_generated_anchor_tensor_used": False,
    "anchor_used_only_as_review_context": True,
    "anchor_reference_or_quotient_arm_deferred": True,
    "global_source_reference_r2v4_upper_bound_arm_deferred": True,
    "local_G_step0_domain_separated_gaussian_arm_deferred": True,
    "outside_G_claim": (
        "within local arm, scheduler model_output bytes equal that same step "
        "official V2V bytes outside exact G"
    ),
    "cross_arm_final_outside_pixel_identity_claimed": False,
}
_EXPECTED_POST_RUN_CONTRACT = {
    "outputs_side_by_side_only": True,
    "base_and_regen_video_hashes_required": True,
    "automatic_selection": False,
    "background_cosine_selection": False,
    "e03_keep_base": True,
    "later_output_bound_independent_review_required_for_any_selection": True,
}

# There is deliberately no CLI/env override for any release root.
COMPILED_SPEC_SHA256: Optional[str] = (
    "edadfa5be1758aaed8b8c4c5f72354bd6becf9a3b999ad116814886e08487d7e"
)
COMPILED_RUNTIME_SHA256: Optional[str] = (
    "b8e0018893c9582d97d20446956c2bea0506fbc48c7d333a021f70f467edc0d0"
)
COMPILED_RUNNER_SHA256: Optional[str] = (
    "ee7fe068096231f222fe9cb153e754b35be3f9078dc00f3924bb7889142aa2f9"
)
COMPILED_VAE_TOOL_SHA256: Optional[str] = (
    "07b40cdd67771d257ce546ca4166301980c6768269acc5f097fc08973656bbde"
)
COMPILED_PROMPT_TOOL_SHA256: Optional[str] = (
    "5bba9f977fa40e5044053baaaf73eba779b3816ef6137457a06ceac82a3463af"
)
COMPILED_AUTHORITY_PACKET_SHA256: Optional[str] = (
    "6ae5602350d54696e0ddcd716a311f96a3569c6f062622840ad130fcbba0baeb"
)
COMPILED_EXTERNAL_LEDGER_SHA256: Optional[str] = (
    "5a9efae443bc8d3cb0886dee7f950204377f653f7dbc474f820d7abbbe437e51"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ActivationPreflightError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _owned_bytes(path: Path, *, label: str) -> tuple[bytes, Mapping[str, Any]]:
    if not path.is_absolute() or path.is_symlink():
        raise ActivationPreflightError(f"{label} must be an absolute non-symlink file")
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise ActivationPreflightError(f"{label} open failed") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o222
        ):
            raise ActivationPreflightError(
                f"{label} must be a frozen one-link regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = path.lstat()
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_size,
        row.st_mode,
        row.st_nlink,
    )
    if identity(before) != identity(after) or identity(after) != identity(named):
        raise ActivationPreflightError(f"{label} changed during owned read")
    return b"".join(chunks), {
        "sha256": hashlib.sha256(b"".join(chunks)).hexdigest(),
        "size": int(after.st_size),
        "mode": stat.S_IMODE(after.st_mode),
        "nlink": int(after.st_nlink),
    }


def _owned_stream_identity(path: Path, *, label: str) -> Mapping[str, Any]:
    """Stream a large frozen file without retaining its payload in RAM."""

    if not path.is_absolute() or path.is_symlink():
        raise ActivationPreflightError(f"{label} must be an absolute non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise ActivationPreflightError(f"{label} open failed") from error
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = path.lstat()
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_size,
        row.st_mode,
        row.st_nlink,
    )
    if (
        identity(before) != identity(after)
        or identity(after) != identity(named)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or size != after.st_size
    ):
        raise ActivationPreflightError(f"{label} changed during owned stream")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size": int(after.st_size),
        "mode": stat.S_IMODE(after.st_mode),
        "nlink": int(after.st_nlink),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
    }


def _private_directory_identity(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_absolute() or path == Path("/") or path.is_symlink():
        raise ActivationPreflightError(f"{label} path differs")
    try:
        resolved = path.resolve(strict=True)
        row = path.lstat()
    except OSError as error:
        raise ActivationPreflightError(f"{label} stat failed") from error
    if (
        resolved != path
        or not stat.S_ISDIR(row.st_mode)
        or stat.S_IMODE(row.st_mode) != 0o700
        or int(row.st_uid) != os.geteuid()
    ):
        raise ActivationPreflightError(
            f"{label} must be canonical/private/effective-user-owned"
        )
    return {
        "path": str(path),
        "mode": stat.S_IMODE(row.st_mode),
        "uid": int(row.st_uid),
        "gid": int(row.st_gid),
        "device": int(row.st_dev),
        "inode": int(row.st_ino),
    }


def _node_local_tmp_parent_identity() -> Mapping[str, Any]:
    """Validate and capture the live public ``/tmp`` mount-point identity."""

    path = _MIOPEN_LOCAL_TMP_PARENT
    if not path.is_absolute() or path.is_symlink():
        raise ActivationPreflightError("node-local tmp parent path differs")
    try:
        resolved = path.resolve(strict=True)
        row = path.lstat()
    except OSError as error:
        raise ActivationPreflightError(
            "node-local tmp parent stat failed"
        ) from error
    mode = stat.S_IMODE(row.st_mode)
    if (
        resolved != path
        or not stat.S_ISDIR(row.st_mode)
        or mode != 0o1777
        or int(row.st_uid) != 0
    ):
        raise ActivationPreflightError(
            "node-local tmp parent must be canonical root-owned mode 1777"
        )
    return {
        "path": str(path),
        "mode": mode,
        "uid": int(row.st_uid),
        "gid": int(row.st_gid),
        "device": int(row.st_dev),
        "inode": int(row.st_ino),
    }


def _strict_json_bytes(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise ActivationPreflightError(f"{label} duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except ActivationPreflightError:
        raise
    except Exception as error:
        raise ActivationPreflightError(f"{label} JSON differs") from error
    if not isinstance(value, Mapping):
        raise ActivationPreflightError(f"{label} root must be an object")
    return value


def _all_compiled() -> bool:
    values = (
        COMPILED_SPEC_SHA256,
        COMPILED_RUNTIME_SHA256,
        COMPILED_RUNNER_SHA256,
        COMPILED_VAE_TOOL_SHA256,
        COMPILED_PROMPT_TOOL_SHA256,
        COMPILED_AUTHORITY_PACKET_SHA256,
        COMPILED_EXTERNAL_LEDGER_SHA256,
    )
    return all(isinstance(value, str) and _SHA256.fullmatch(value) for value in values)


def _binding(
    value: Any, *, relative_path: str, sha256: str, label: str
) -> None:
    expected = {"path": relative_path, "sha256": sha256}
    if value != expected:
        raise ActivationPreflightError(f"{label} binding differs")


def _validate_scheduler_tmpdir_normalization() -> Mapping[str, Any]:
    """Bind the exact scheduler-injected ``/tmp`` normalization receipt."""

    prefix = "ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_"
    observed_environment = {
        key: value for key, value in os.environ.items() if key.startswith(prefix)
    }
    expected_keys = set(_SCHEDULER_TMPDIR_NORMALIZATION_ENV.values())
    if set(observed_environment) != expected_keys:
        raise ActivationPreflightError(
            "scheduler TMPDIR normalization environment differs"
        )
    values = {
        role: observed_environment[name]
        for role, name in _SCHEDULER_TMPDIR_NORMALIZATION_ENV.items()
    }
    step_id = values["step_id"]
    if (
        values["job_id"] != "141620"
        or os.environ.get("SLURM_JOB_ID") != values["job_id"]
        or not isinstance(step_id, str)
        or re.fullmatch(r"0|[1-9][0-9]*", step_id) is None
        or os.environ.get("SLURM_STEP_ID") != step_id
        or values["hostname"] != "auh7-1b-gpu-226"
        or socket.gethostname().split(".", 1)[0] != values["hostname"]
        or values["observed_tmpdir"] != "/tmp"
        or values["action"] != _SCHEDULER_TMPDIR_NORMALIZATION_ACTION
        or any(os.environ.get(name) is not None for name in _MIOPEN_TEMP_ENV_NAMES)
    ):
        raise ActivationPreflightError(
            "scheduler TMPDIR normalization binding differs"
        )
    return {
        "schema_version": "bernini-slurm-tmpdir-normalization-r9",
        "slurm_job_id": values["job_id"],
        "slurm_step_id": step_id,
        "hostname": values["hostname"],
        "scheduler_observed_tmpdir": values["observed_tmpdir"],
        "normalization_action": values["action"],
        "normalized_before_any_python_or_torch_import": True,
        "launcher_receipt_environment_keys": sorted(expected_keys),
    }


def _validate_host_load_environment() -> Mapping[str, Any]:
    """Bind the cross-rank renderer load lock before model/distributed startup."""

    if (
        os.environ.get("NATIVE_SERIALIZED_HOST_LOAD_REQUIRED") != "1"
        or os.environ.get("SLURM_JOB_ID") != "141620"
        or socket.gethostname().split(".", 1)[0] != "auh7-1b-gpu-226"
        or os.environ.get("ROCR_VISIBLE_DEVICES") != "0,1,2,3"
    ):
        raise ActivationPreflightError("serialized host-load requirement differs")
    raw_path = os.environ.get("NATIVE_V_AXIS_LOAD_LOCK")
    if not isinstance(raw_path, str) or not raw_path:
        raise ActivationPreflightError("serialized host-load lock is absent")
    requested = Path(raw_path)
    if not requested.is_absolute() or requested.is_symlink():
        raise ActivationPreflightError("serialized host-load lock path differs")
    path = requested.resolve(strict=True)
    if path != requested or path.is_symlink():
        raise ActivationPreflightError("serialized host-load lock identity differs")
    raw, identity = _owned_bytes(path, label="serialized host-load lock")
    if raw != b"" or identity != {
        "sha256": hashlib.sha256(b"").hexdigest(),
        "size": 0,
        "mode": 0o444,
        "nlink": 1,
    }:
        raise ActivationPreflightError(
            "serialized host-load lock must be empty/frozen/one-link"
        )
    return {
        "required": True,
        "path": str(path),
        "authorized_slurm_job_id": "141620",
        "authorized_hostname": "auh7-1b-gpu-226",
        "rocr_visible_devices": "0,1,2,3",
        **dict(identity),
    }


def _expected_miopen_local_tmp_root(
    output: Path, scheduler: Mapping[str, Any]
) -> Path:
    job_id = str(scheduler.get("slurm_job_id", ""))
    step_id = str(scheduler.get("slurm_step_id", ""))
    uid = str(os.geteuid())
    if (
        job_id != "141620"
        or re.fullmatch(r"0|[1-9][0-9]*", step_id) is None
        or not output.is_absolute()
    ):
        raise ActivationPreflightError(
            "MIOpen node-local tmp domain binding differs"
        )
    payload = b"\x00".join(
        (
            _MIOPEN_LOCAL_TMP_DOMAIN.encode("utf-8"),
            job_id.encode("ascii"),
            step_id.encode("ascii"),
            uid.encode("ascii"),
            str(output).encode("utf-8"),
        )
    )
    digest = hashlib.sha256(payload).hexdigest()
    return _MIOPEN_LOCAL_TMP_PARENT / (
        f"oracle-regeneration-native-v2-r9-u{uid}-j{job_id}-s{step_id}-o{digest}"
    )


def _validate_miopen_cache_environment() -> Mapping[str, Any]:
    """Validate the launcher-minted writable cache before any Torch import."""

    if "torch" in sys.modules:
        raise ActivationPreflightError("Torch loaded before MIOpen cache preflight")
    scheduler_tmpdir_normalization = _validate_scheduler_tmpdir_normalization()
    output_raw = os.environ.get(_MIOPEN_OUTPUT_ENV)
    root_raw = os.environ.get(_MIOPEN_CACHE_ROOT_ENV)
    local_tmp_raw = os.environ.get(_MIOPEN_LOCAL_TMP_ROOT_ENV)
    if (
        not isinstance(output_raw, str)
        or not isinstance(root_raw, str)
        or not isinstance(local_tmp_raw, str)
    ):
        raise ActivationPreflightError("MIOpen cache/output environment is absent")
    output = Path(output_raw)
    root = Path(root_raw)
    local_tmp_root = Path(local_tmp_raw)
    if (
        not output.is_absolute()
        or output == Path("/")
        or output.exists()
        or output.is_symlink()
        or output.parent.resolve(strict=True) != output.parent
        or root != Path(f"{output}{_MIOPEN_CACHE_SUFFIX}")
    ):
        raise ActivationPreflightError("MIOpen cache/output sibling binding differs")
    expected_local_tmp_root = _expected_miopen_local_tmp_root(
        output, scheduler_tmpdir_normalization
    )
    if (
        local_tmp_root != expected_local_tmp_root
    ):
        raise ActivationPreflightError(
            "MIOpen node-local tmp root binding differs"
        )
    try:
        output.relative_to(METHOD_ROOT.parents[1])
    except ValueError:
        pass
    else:
        raise ActivationPreflightError(
            "MIOpen cache/output must be outside the frozen release tree"
        )
    root_identity = _private_directory_identity(root, label="MIOpen cache root")
    local_tmp_root_identity = _private_directory_identity(
        local_tmp_root, label="MIOpen node-local tmp root"
    )
    local_tmp_parent_identity = _node_local_tmp_parent_identity()
    if (
        int(local_tmp_parent_identity["device"])
        != local_tmp_root_identity["device"]
        or local_tmp_root_identity["device"] == root_identity["device"]
    ):
        raise ActivationPreflightError(
            "MIOpen node-local tmp filesystem/mode/owner differs"
        )
    names = {entry.name for entry in root.iterdir()}
    if names != _MIOPEN_CACHE_DIRECTORY_NAMES:
        raise ActivationPreflightError("MIOpen cache initial directory set differs")
    directories: dict[str, Mapping[str, Any]] = {}
    for name in sorted(names):
        path = root / name
        directories[name] = _private_directory_identity(
            path, label=f"MIOpen cache {name}"
        )
        if any(path.iterdir()):
            raise ActivationPreflightError(
                f"MIOpen cache {name} is not initially empty"
            )
    local_tmp_names = {entry.name for entry in local_tmp_root.iterdir()}
    if local_tmp_names != _MIOPEN_LOCAL_TMP_DIRECTORY_NAMES:
        raise ActivationPreflightError(
            "MIOpen node-local tmp initial directory set differs"
        )
    local_tmp_directories: dict[str, Mapping[str, Any]] = {}
    for name in sorted(local_tmp_names):
        path = local_tmp_root / name
        local_tmp_directories[name] = _private_directory_identity(
            path, label=f"MIOpen node-local tmp {name}"
        )
        if (
            local_tmp_directories[name]["device"]
            != local_tmp_root_identity["device"]
            or any(path.iterdir())
        ):
            raise ActivationPreflightError(
                f"MIOpen node-local tmp {name} identity/emptiness differs"
            )
    launcher_local_tmp_empty_proof = _launcher_local_tmp_empty_proof(
        local_tmp_root_identity, local_tmp_directories
    )
    if os.environ.get(_MIOPEN_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_ENV) != str(
        launcher_local_tmp_empty_proof["sha256"]
    ):
        raise ActivationPreflightError(
            "launcher node-local tmp empty proof differs"
        )
    bootstrap_user_db = root / "launcher-bootstrap-user-db"
    bootstrap_kernel_cache = root / "launcher-bootstrap-kernel-cache"
    miopen_environment = {
        key: value for key, value in os.environ.items() if key.startswith("MIOPEN_")
    }
    expected_miopen_environment = {
        _MIOPEN_USER_DB_ENV: str(bootstrap_user_db),
        _MIOPEN_KERNEL_CACHE_ENV: str(bootstrap_kernel_cache),
    }
    if (
        miopen_environment != expected_miopen_environment
        or any(os.environ.get(name) is not None for name in _MIOPEN_TEMP_ENV_NAMES)
    ):
        raise ActivationPreflightError(
            "launcher MIOpen bootstrap environment differs"
        )
    library = _owned_stream_identity(
        _MIOPEN_LIBRARY_PATH, label="Torch-bundled MIOpen library"
    )
    if (
        library["sha256"] != _MIOPEN_LIBRARY_SHA256
        or library["size"] != _MIOPEN_LIBRARY_SIZE
        or library["mode"] != 0o755
        or library["nlink"] != 1
    ):
        raise ActivationPreflightError("Torch-bundled MIOpen library differs")
    local_rank_raw = os.environ.get("LOCAL_RANK")
    if local_rank_raw is not None:
        try:
            local_rank = int(local_rank_raw)
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        except (KeyError, TypeError, ValueError) as error:
            raise ActivationPreflightError("MIOpen worker rank env differs") from error
        if (
            local_rank not in range(4)
            or rank != local_rank
            or world_size != 4
            or local_world_size != 4
        ):
            raise ActivationPreflightError("MIOpen worker WORLD4 env differs")
    else:
        local_rank = None
    return {
        "schema_version": "bernini-miopen-cache-cpu-preflight-r11",
        "root_path": str(root),
        "output_sibling_path": str(output),
        "root_identity": dict(root_identity),
        "directory_identities": directories,
        "local_tmp_root_path": str(local_tmp_root),
        "local_tmp_root_identity": dict(local_tmp_root_identity),
        "local_tmp_parent_identity": dict(local_tmp_parent_identity),
        "local_tmp_directory_identities": local_tmp_directories,
        "launcher_local_tmp_fresh_empty_proof": dict(
            launcher_local_tmp_empty_proof
        ),
        "cpu_preflight_local_tmp_empty_proof": {
            "root_identity": dict(local_tmp_root_identity),
            "directory_identities": local_tmp_directories,
            "all_rank_directories_empty": True,
        },
        "persistent_and_local_tmp_devices_are_distinct": True,
        "initial_directory_count": len(directories),
        "all_directories_initially_empty": True,
        "official_variables_bound_to_launcher_bootstrap": True,
        "exact_miopen_environment_key_allowlist": sorted(
            expected_miopen_environment
        ),
        "torchrun_parent_tmpdir": None,
        "scheduler_tmpdir_normalization": dict(
            scheduler_tmpdir_normalization
        ),
        "worker_local_rank": local_rank,
        "library": dict(library),
        "torch_imported": False,
        "engineering_evidence_only": True,
        "local_tmp_runner_cleanup_performed": False,
        "local_tmp_durability_guaranteed": False,
        "local_tmp_node_lifetime_only": True,
    }


def _validate_runner_ast(raw: bytes) -> Mapping[str, Any]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(RUNNER_PATH))
    except Exception as error:
        raise ActivationPreflightError("native runner syntax differs") from error
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "main" not in functions or "cpu_preflight" not in functions:
        raise ActivationPreflightError("native runner entry ABI differs")
    source = raw.decode("utf-8")
    required_markers = (
        "LocalOracleNativeBranchRuntimePatchV2",
        "_sample_with_native_initial_noise_observer",
        "official-v2v-base",
        "local-source-reference-r2v4-in-manual-G",
        "ABSTAIN_KEEP_BASE",
        "self_generated_anchor_tensor_used",
        "MIOPEN_USER_DB_PATH",
        "MIOPEN_CUSTOM_CACHE_DIR",
        "_activate_miopen_cache_pre_torch",
        "_seal_miopen_cache_receipt_world4",
        "_seal_local_tmp_receipt_world4",
        "launcher_local_tmp_fresh_empty_proof",
        "post_runtime_init_baseline_world4",
        "continuous_monitoring_claimed",
        "--miopen-local-tmp-root",
        "node_lifetime_only",
    )
    if any(marker not in source for marker in required_markers):
        raise ActivationPreflightError("native runner required ABI marker is absent")
    return {"top_level_functions": sorted(functions), "required_markers": list(required_markers)}


def validate_release(
    *,
    packet_path: Path,
    ledger_path: Path,
) -> Mapping[str, Any]:
    """Return a ready receipt only for the later exact compiled release."""

    if not _all_compiled():
        raise ActivationPreflightError(
            "activation release pins are intentionally not compiled in this candidate"
        )
    if "source_self_native_ref_contrastive_v3" in sys.modules:
        raise ActivationPreflightError(
            "native UniPC40 schedule module was loaded before the CPU gate"
        )

    files = {
        "spec": SPEC_PATH,
        "runtime": RUNTIME_PATH,
        "runner": RUNNER_PATH,
        "vae_tool": VAE_TOOL_PATH,
        "prompt_tool": PROMPT_TOOL_PATH,
    }
    observed: dict[str, Mapping[str, Any]] = {}
    raw: dict[str, bytes] = {}
    for label, path in files.items():
        if path.resolve(strict=True) != path or path.is_symlink():
            raise ActivationPreflightError(f"{label} path identity differs")
        value, identity = _owned_bytes(path, label=label)
        raw[label] = value
        observed[label] = identity
    spec = _strict_json_bytes(raw["spec"], label="activation spec")
    expected_hashes = {
        "spec": COMPILED_SPEC_SHA256,
        "runtime": COMPILED_RUNTIME_SHA256,
        "runner": COMPILED_RUNNER_SHA256,
        "vae_tool": COMPILED_VAE_TOOL_SHA256,
        "prompt_tool": COMPILED_PROMPT_TOOL_SHA256,
    }
    if any(observed[label]["sha256"] != digest for label, digest in expected_hashes.items()):
        raise ActivationPreflightError("compiled component bytes differ")
    if (
        set(spec) != _SPEC_KEYS
        or spec.get("schema_version") != SPEC_SCHEMA_VERSION
        or spec.get("status") != "ACTIVATED_INDEPENDENT_MODEL_REVIEWED_DIAGNOSTIC_CANARY"
        or spec.get("launch_ready") is not True
        or spec.get("scope") != "experimental diagnostic canary only"
        or spec.get("authority_kind")
        != "diagnostic_exact_packet_and_code_review_trust_root"
        or spec.get("formal_authority") is not False
        or spec.get("training_authority") is not False
        or spec.get("training") is not False
        or spec.get("optimizer") is not False
        or spec.get("automatic_replacement") is not False
        or spec.get("selection_authority") is not None
        or spec.get("native_only") is not True
        or spec.get("flowedit") is not False
        or spec.get("connected_route") is not False
        or spec.get("learned_gate") is not False
        or spec.get("world_size") != 4
        or spec.get("sequence_parallel_size") != 4
        or spec.get("one_node") is not True
        or spec.get("candidate_count_per_arm") != 1
        or spec.get("host_load_contract") != _EXPECTED_HOST_LOAD_CONTRACT
        or spec.get("allocation_contract") != _EXPECTED_ALLOCATION_CONTRACT
        or spec.get("miopen_cache_contract") != _EXPECTED_MIOPEN_CACHE_CONTRACT
        or spec.get("runner_allowlist") != [RUNNER_PATH.name]
        or spec.get("cases") != _EXPECTED_SPEC_CASES
        or spec.get("scientific_boundary") != _EXPECTED_SCIENTIFIC_BOUNDARY
        or spec.get("post_run_contract") != _EXPECTED_POST_RUN_CONTRACT
        or spec.get("mandatory_blockers") != []
        or spec.get("compiled_authority_packet_sha256")
        != COMPILED_AUTHORITY_PACKET_SHA256
        or spec.get("compiled_external_ledger_receipt_sha256")
        != COMPILED_EXTERNAL_LEDGER_SHA256
    ):
        raise ActivationPreflightError("activation spec is not exact launch-ready policy")
    host_load_lock = _validate_host_load_environment()
    miopen_cache = _validate_miopen_cache_environment()
    components = spec.get("components")
    if (
        not isinstance(components, Mapping)
        or set(components) != set(_COMPONENT_RELATIVE_PATHS)
    ):
        raise ActivationPreflightError("activation component map differs")
    for key, label in (
        ("activation_runtime", "runtime"),
        ("native_runner", "runner"),
        ("vae_reference_materializer", "vae_tool"),
        ("prompt_materializer", "prompt_tool"),
    ):
        _binding(
            components.get(key),
            relative_path=_COMPONENT_RELATIVE_PATHS[key],
            sha256=str(expected_hashes[label]),
            label=key,
        )
    runtime_dependencies = spec.get("frozen_runtime_dependencies")
    if (
        not isinstance(runtime_dependencies, Mapping)
        or set(runtime_dependencies) != set(_RUNTIME_DEPENDENCY_RELATIVE_PATHS)
    ):
        raise ActivationPreflightError("runtime dependency allowlist differs")
    for declared_name, relative_path in _RUNTIME_DEPENDENCY_RELATIVE_PATHS.items():
        expected_digest = runtime_dependencies.get(declared_name)
        if not isinstance(expected_digest, str) or _SHA256.fullmatch(expected_digest) is None:
            raise ActivationPreflightError(
                f"runtime dependency {declared_name} digest differs"
            )
        dependency_path = (METHOD_ROOT / relative_path).resolve(strict=True)
        if dependency_path != METHOD_ROOT / relative_path:
            raise ActivationPreflightError(
                f"runtime dependency {declared_name} path identity differs"
            )
        dependency_raw, dependency_identity = _owned_bytes(
            dependency_path, label=f"runtime dependency {declared_name}"
        )
        del dependency_raw
        if dependency_identity["sha256"] != expected_digest:
            raise ActivationPreflightError(
                f"runtime dependency {declared_name} bytes differ"
            )
        observed[f"dependency:{declared_name}"] = dependency_identity
    runner_abi = _validate_runner_ast(raw["runner"])

    # The runtime has already passed an exact byte check.  Reject a preloaded or
    # shadowed module before executing it.
    while str(METHOD_ROOT) in sys.path:
        sys.path.remove(str(METHOD_ROOT))
    sys.path.insert(0, str(METHOD_ROOT))
    runtime_name = "oracle_regeneration_native_runtime_activation_v2"
    preloaded = sys.modules.get(runtime_name)
    if preloaded is not None and Path(
        str(getattr(preloaded, "__file__", ""))
    ).resolve(strict=True) != RUNTIME_PATH.resolve(strict=True):
        raise ActivationPreflightError("preloaded activation runtime origin differs")
    runtime_spec = importlib.util.find_spec(runtime_name)
    if (
        runtime_spec is None
        or not isinstance(runtime_spec.origin, str)
        or Path(runtime_spec.origin).resolve(strict=True)
        != RUNTIME_PATH.resolve(strict=True)
    ):
        raise ActivationPreflightError("activation runtime import origin differs")
    import oracle_regeneration_native_runtime_activation_v2 as activation

    if (
        activation.COMPILED_AUTHORITY_PACKET_SHA256
        != COMPILED_AUTHORITY_PACKET_SHA256
        or activation.COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256
        != COMPILED_EXTERNAL_LEDGER_SHA256
    ):
        raise ActivationPreflightError("preflight/runtime trust anchors differ")
    try:
        authority = activation.load_compiled_activation_authority_v2(
            packet_path, ledger_path
        )
    except Exception as error:
        raise ActivationPreflightError(
            "compiled authority graph failed validation"
        ) from error
    if (
        authority.packet_sha256 != COMPILED_AUTHORITY_PACKET_SHA256
        or authority.ledger_sha256 != COMPILED_EXTERNAL_LEDGER_SHA256
        or tuple(authority.cases) != ("e02", "e03")
        or authority.cases["e02"].decision != "ACTIVE_DIAGNOSTIC"
        or authority.cases["e02"].seed != 0
        or list(authority.cases["e02"].run_arms)
        != _EXPECTED_SPEC_CASES["e02"]["arms"]
        or authority.cases["e03"].decision != "ABSTAIN_KEEP_BASE"
        or authority.cases["e03"].seed != 0
        or authority.cases["e03"].run_arms
        or authority.cases["e03"].reference_receipt_path is not None
        or authority.cases["e03"].prompt_receipt_path is not None
        or str(authority.cases["e03"].kept_frozen_base_path)
        != _EXPECTED_SPEC_CASES["e03"]["kept_frozen_base"]["path"]
        or authority.cases["e03"].kept_frozen_base_sha256
        != _EXPECTED_SPEC_CASES["e03"]["kept_frozen_base"]["sha256"]
    ):
        raise ActivationPreflightError("loaded authority roots differ")
    return {
        "schema_version": SCHEMA_VERSION,
        "ready": True,
        "cpu_only": True,
        "torch_imported": "torch" in sys.modules,
        "distributed_initialized": False,
        "model_loaded": False,
        "spec_sha256": observed["spec"]["sha256"],
        "component_identities": observed,
        "authority_packet_sha256": authority.packet_sha256,
        "external_ledger_sha256": authority.ledger_sha256,
        "packet_id": authority.packet_id,
        "cases": list(authority.cases),
        "runner_abi": runner_abi,
        "serialized_host_load_lock": host_load_lock,
        "scheduler_tmpdir_normalization": dict(
            miopen_cache["scheduler_tmpdir_normalization"]
        ),
        "miopen_cache_preflight": miopen_cache,
        "training": False,
        "optimizer": False,
        "flowedit": False,
        "connected_route": False,
        "automatic_replacement": False,
        "selection_authority": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-packet", required=True)
    parser.add_argument("--external-ledger", required=True)
    parser.add_argument("--require-ready", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_release(
            packet_path=Path(args.authority_packet),
            ledger_path=Path(args.external_ledger),
        )
    except Exception as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "ready": False,
            "reason": str(error),
            "cpu_only": True,
            "torch_imported": "torch" in sys.modules,
            "distributed_initialized": False,
            "model_loaded": False,
            "training": False,
            "optimizer": False,
            "flowedit": False,
            "connected_route": False,
            "automatic_replacement": False,
            "selection_authority": None,
        }
        print(_canonical(result).decode("utf-8"))
        return 3
    print(_canonical(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed plan and receipt contract for the case01 trajectory exact-five.

This is an engineering-oracle contract, not evidence that object-centric
learning works.  It deliberately builds a non-launchable HOLD plan unless the
new inference wrapper and every external condition authority are fully pinned.
Unlike the historical evaluator, this module accepts and verifies external
object masks, tracks, a typed trajectory scaffold, and a bone-removed auxiliary
source.  It never routes custom receipts through the legacy verifier whose ABI
requires all of those conditions to be absent.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "case01-object-trajectory-exact5-plan-v1"
REPORT_SCHEMA = "case01-object-trajectory-exact5-report-v1"
INFERENCE_RECEIPT_SCHEMA = (
    "bernini-r-1p3b-case01-object-trajectory-oracle-inference-receipt-v3"
)
LEGACY_INFERENCE_RECEIPT_SCHEMA = (
    "bernini-r-1p3b-action-lora-inference-receipt-v5"
)
OBJECT_ORACLE_RUNTIME_SCHEMA = (
    "bernini-case01-object-trajectory-oracle-runtime-v3"
)
PROJECTION_TRACE_SCHEMA = "bernini-object-trajectory-unipc-projection-v1"
FILE_AUTHORITY_SCHEMA = "case01-object-trajectory-file-authority-v1"
EXPERIMENT_ID = "case01-288545b9c031491a-object-trajectory-exact5-v1"
CAMPAIGN = "case01-object-trajectory-exact5-r64-engineering-oracle"
IID = "288545b9c031491a"
INSTRUCTION = "Make the dog pick up the bone and hold it in its mouth."
INSTRUCTION_SHA256 = (
    "84df12ede824d239a4c7c3d21dccdf22663535d1e504e7b280544c8a9be0fd5d"
)
SEED = 2027
NUM_INFERENCE_STEPS = 40
AUX_COLLECTIVE_STAGES = (
    "aux_readiness",
    "aux_post_broadcast",
)
PROJECTION_COLLECTIVE_STAGES = (
    "projection_runtime_readiness",
    "projection_row_build",
    "projection_contract_build",
    "projection_projector_lookup",
    "projection_lazy_bootstrap_install",
    "projection_projector_install",
    "projection_final_validation",
)
EXPECTED_SUCCESSFUL_ALL_GATHER_OBJECT_CALLS = 10
ARM_ORDER = (
    "null_before",
    "route_off",
    "trajectory_bone_only",
    "trajectory_dog_bone",
    "null_after",
)
VARIANT_ORDER = ARM_ORDER
TASK_IDS = tuple(f"case01-object-trajectory-{arm}-full644" for arm in ARM_ORDER)
EXTERNAL_AUTHORITY_KEYS = (
    "stage0_masks",
    "g0_mouth_track",
    "trajectory_scaffold",
    "aux_bone_removed_source",
)
ADMISSION_AUTHORITY_KEYS = ("scaffold_independent_audit",)
EXTERNAL_CONDITION_NAMES = (
    "stage0_object_masks",
    "g0_mouth_track",
    "object_trajectory_scaffold",
    "aux_bone_removed_source",
)
BASE_CONDITION_NAMES = ("source_video", "edit_instruction")
EXPECTED_SOURCE_SHA256 = (
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
)
EXPECTED_SOURCE_SIZE = 10_887_043
EXPECTED_STAGE0_SHA256 = (
    "c9dc8567d4f147f943171d2d7abb55e52aadc685fbfc2f75fff4e837e2ab0b50"
)
EXPECTED_STAGE0_SIZE = 22_160
EXPECTED_G0_SHA256 = (
    "e5185a1edd72fa8a1f2ece15e98c67d66e3fa65a2a9eb724bf06031c4d0e2020"
)
EXPECTED_G0_SIZE = 6_882
EXPECTED_AUX_REMOVED_SHA256 = (
    "8c525385832586fa7b7fd7ae6e5701c599694d26ee27b502dbf0bb582e55e1c9"
)
EXPECTED_AUX_REMOVED_SIZE = 5_424_975
EXPECTED_LEGACY_INFER_LORA_SHA256 = (
    "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
)
EXPECTED_LEGACY_INFER_LORA_SIZE = 177_300
EXPECTED_INFERENCE_WRAPPER_SHA256 = (
    "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9"
)
EXPECTED_INFERENCE_WRAPPER_SIZE = 74_281
EXPECTED_TRAJECTORY_PROJECTION_MODULE_SHA256 = (
    "a851afd220d770e6c8082ff8e7f0a0c2b5a5be894bcbf9feeaf8bca4cc6c9e7e"
)
EXPECTED_TRAJECTORY_PROJECTION_MODULE_SIZE = 47_588
EXPECTED_TRAJECTORY_SCAFFOLD_MODULE_SHA256 = (
    "a7d4e008e78d373370b832c0754e5d7420b504fa8b2022eef9a9bb899added8a"
)
EXPECTED_TRAJECTORY_SCAFFOLD_MODULE_SIZE = 35_803
EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_SHA256 = (
    "7b1bec6e9764a1297bb0029f8fea01ebe4b2deab0acc2c7f07fdee96bc0a098a"
)
EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_SIZE = 54_801
EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST = (
    "5e6156909d8261a23c3add3134059bec20505b682ca0eb13dc88fa8512eeace1"
)
EXPECTED_TRAJECTORY_SCAFFOLD_SCHEMA = (
    "case01-oracle-object-trajectory-scaffold-v1"
)
EXPECTED_STAGE0_RECEIPT_DIGEST = (
    "36d9b072febab782647f4cda4e63df9d78656b392d33ce4e02777af11697b8fa"
)
EXPECTED_SCAFFOLD_AUDIT_SHA256 = (
    "acbe4a6e635e3429605a8aac4d655816fd6187ea7aec77d5a8b1e08a56a47e0e"
)
EXPECTED_SCAFFOLD_AUDIT_SIZE = 2_493
EXPECTED_SCAFFOLD_AUDIT_DIGEST = (
    "c142e1a18abab58784ad2fc5fef8588eb4e21657ed922b6910e120540752a3fe"
)
EXPECTED_SCAFFOLD_AUDIT_SCHEMA = (
    "case01-object-trajectory-scaffold-independent-audit-v1"
)
EXPECTED_CHECKPOINT = {
    "sha256": "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2",
    "manifest_digest": "7bae23da51a3c5a67adb41ee85dd026c374d2581bd3409e868e18b2f6f4dffc4",
    "global_step": 644,
    "receipt_digest": "aaf348a7daa6c5ca2fe721771857287125ee02eb2c9a499f45b11a2e113d15d7",
    "file_count": 5,
    "adapter_config_sha256": "94bfaf73d714d7e77095ff68ce57e24932e0c05bde324263f5fe321660b95f62",
    "adapter_model_sha256": "44efdc5a0501238250b1d32ae2859abe248ffc37b152cd8db86ff84b378d6b22",
    "training_receipt_sha256": "3402c8c93c092bfc4490bf86790ab6429b4cbaad38358956cb0beeb5df7d4c4c",
    "optimizer_sha256": "77b7b22db4da92f28f23b4ae91c7271f55ab6a92353bfc8b0bbeb30529a7af63",
}
FULL644_PROFILE = "full644-r64-reference-dpo-preservation-one-pass-v1"
EXPECTED_TARGET_MODULES_SHA256 = (
    "d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a"
)
EXPECTED_SYSTEM_PROMPT_SHA256 = (
    "12ce75b4360bf5f6d2fdb1e22619438fad6363fd5356634fa698fcb28a83e0ba"
)
EXPECTED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
EXPECTED_VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
EXPECTED_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
EXPECTED_BERNINI_INFERENCE_FILES = {
    "bernini/pipeline.py": (
        "c6acf05c01a637d9bce69e8160eb6eb4260ff4ec798fd990de8e5aa73999ab40"
    ),
    "bernini/cli.py": (
        "26949fbf246003403ed0cca1ec1bbb62c2099fc9740bb17ba5a1e7c86fbc0edf"
    ),
    "bernini/io_utils.py": (
        "233541373746f5d97e1cb3680d3c2a41d5d212b797eefb97693afa6e3ab5f30a"
    ),
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER_PREFIX = "__INCOMPLETE_"
EXPECTED_PRODUCER_BASENAMES = {
    "infer_lora_path": "infer_lora_full644_r5_frozen_acc46.py",
    "inference_wrapper_path": "infer_case01_object_trajectory_oracle_v1.py",
    "trajectory_projection_module_path": "object_trajectory_projection_v1.py",
    "trajectory_scaffold_module_path": "case01_oracle_object_trajectory_v1.py",
}


class ObjectTrajectoryEvalError(RuntimeError):
    """The trajectory plan, receipt, or retained result closure differs."""


# Compatibility name consumed by the source-loaded exact5 runner.
Exact5EvalError = ObjectTrajectoryEvalError


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ObjectTrajectoryEvalError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_digest(value: Mapping[str, Any], field: str, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise ObjectTrajectoryEvalError(f"{label} is not an object")
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    observed = object_sha256(unsigned)
    if not isinstance(claimed, str) or claimed != observed:
        raise ObjectTrajectoryEvalError(f"{label} digest differs")
    return claimed


def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _stable_file(
    path_value: str | Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
    return_bytes: bool = False,
) -> tuple[bytes | None, str, int]:
    path = Path(path_value).expanduser()
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise ObjectTrajectoryEvalError("pinned file path is not canonical")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if return_bytes:
                chunks.append(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    identity = lambda info: (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    observed = digest.hexdigest()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or identity(before) != identity(after)
        or identity(before) != identity(named)
        or observed != expected_sha256
        or (expected_size is not None and before.st_size != expected_size)
    ):
        raise ObjectTrajectoryEvalError("pinned file identity/SHA differs")
    return (b"".join(chunks) if return_bytes else None, observed, before.st_size)


def build_file_authority(
    path_value: str | Path, *, role: str, payload_digest: str | None = None
) -> dict[str, Any]:
    path = Path(path_value).resolve(strict=True)
    _, sha256, size = _stable_file(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    row: dict[str, Any] = {
        "schema_version": FILE_AUTHORITY_SCHEMA,
        "role": role,
        "complete": True,
        "path": str(path),
        "sha256": sha256,
        "size": size,
        "payload_digest": sha256 if payload_digest is None else payload_digest,
    }
    row["authority_digest"] = object_sha256(row)
    return validate_file_authority(row, expected_role=role, reopen=True)


def incomplete_file_authority(role: str) -> dict[str, Any]:
    token = re.sub(r"[^A-Z0-9]+", "_", role.upper()).strip("_")
    row: dict[str, Any] = {
        "schema_version": FILE_AUTHORITY_SCHEMA,
        "role": role,
        "complete": False,
        "path": f"{PLACEHOLDER_PREFIX}{token}_PATH__",
        "sha256": f"{PLACEHOLDER_PREFIX}{token}_SHA256__",
        "size": 0,
        "payload_digest": f"{PLACEHOLDER_PREFIX}{token}_PAYLOAD_DIGEST__",
    }
    row["authority_digest"] = object_sha256(row)
    return validate_file_authority(row, expected_role=role, reopen=False)


def validate_file_authority(
    value: Mapping[str, Any], *, expected_role: str, reopen: bool
) -> dict[str, Any]:
    fields = {
        "schema_version", "role", "complete", "path", "sha256", "size",
        "payload_digest",
        "authority_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ObjectTrajectoryEvalError(f"{expected_role} authority schema differs")
    row = dict(value)
    _strict_digest(row, "authority_digest", label=f"{expected_role} authority")
    complete = row.get("complete")
    path = row.get("path")
    sha256 = row.get("sha256")
    size = row.get("size")
    payload_digest = row.get("payload_digest")
    if (
        row.get("schema_version") != FILE_AUTHORITY_SCHEMA
        or row.get("role") != expected_role
        or type(complete) is not bool
        or not isinstance(path, str)
        or not isinstance(sha256, str)
        or not isinstance(payload_digest, str)
        or type(size) is not int
    ):
        raise ObjectTrajectoryEvalError(f"{expected_role} authority value differs")
    if complete:
        authority_path = Path(path)
        if (
            SHA256_RE.fullmatch(sha256) is None
            or SHA256_RE.fullmatch(payload_digest) is None
            or size <= 0
            or not authority_path.is_absolute()
            or os.path.normpath(path) != path
            or PLACEHOLDER_PREFIX in path
        ):
            raise ObjectTrajectoryEvalError(f"{expected_role} complete pin differs")
        if reopen:
            _stable_file(
                authority_path, expected_sha256=sha256, expected_size=size
            )
    elif (
        not path.startswith(PLACEHOLDER_PREFIX)
        or not sha256.startswith(PLACEHOLDER_PREFIX)
        or not payload_digest.startswith(PLACEHOLDER_PREFIX)
        or size != 0
    ):
        raise ObjectTrajectoryEvalError(f"{expected_role} incomplete pin differs")
    return row


def _require_exact_complete_authority(
    row: Mapping[str, Any], *, expected_sha256: str, expected_size: int,
    expected_payload_digest: str, label: str,
) -> None:
    if (
        row.get("complete") is not True
        or row.get("sha256") != expected_sha256
        or row.get("size") != expected_size
        or row.get("payload_digest") != expected_payload_digest
    ):
        raise ObjectTrajectoryEvalError(f"{label} authority pin differs")


def _load_canonical_json_authority(
    row: Mapping[str, Any], *, label: str,
) -> dict[str, Any]:
    raw, _, _ = _stable_file(
        row["path"], expected_sha256=row["sha256"],
        expected_size=row["size"], return_bytes=True,
    )
    if raw is None:
        raise ObjectTrajectoryEvalError(f"{label} replay returned no bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise ObjectTrajectoryEvalError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ObjectTrajectoryEvalError(f"{label} is not canonical JSON plus LF")
    return value


def _validate_scaffold_payload(
    payload: Mapping[str, Any], *, source: Mapping[str, Any],
    externals: Mapping[str, Mapping[str, Any]],
) -> None:
    fields = {
        "schema_version", "status", "case_id", "iid", "instruction",
        "authority", "geometry", "typed_action_program", "dog_identity_policy",
        "latent_layout", "frames", "latent_phases", "invariants", "claim_limits",
        "artifact_digest",
    }
    unsigned = dict(payload)
    artifact_digest = unsigned.pop("artifact_digest", None)
    expected_authority = {
        "source_video": {
            "sha256": source["sha256"], "size": source["size"],
        },
        "bone_removed_auxiliary_video": {
            "sha256": externals["aux_bone_removed_source"]["sha256"],
            "size": externals["aux_bone_removed_source"]["size"],
        },
        "stage0_receipt": {
            "sha256": externals["stage0_masks"]["sha256"],
            "size": externals["stage0_masks"]["size"],
            "receipt_digest": externals["stage0_masks"]["payload_digest"],
            "mask_count": 162,
        },
        "g0_sparse_annotations": {
            "sha256": externals["g0_mouth_track"]["sha256"],
            "size": externals["g0_mouth_track"]["size"],
        },
    }
    if (
        set(payload) != fields
        or artifact_digest != EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST
        or object_sha256(unsigned) != artifact_digest
        or payload.get("schema_version") != EXPECTED_TRAJECTORY_SCAFFOLD_SCHEMA
        or payload.get("status") != "ORACLE_SCAFFOLD_READY_NOT_RENDERER_RESULT"
        or payload.get("case_id") != "case01"
        or payload.get("iid") != IID
        or payload.get("instruction") != INSTRUCTION
        or payload.get("authority") != expected_authority
        or payload.get("latent_layout") != {
            "attention_source_half_offset": 0,
            "attention_target_half_offset": 19_530,
            "causal_phase_policy": "phase0=f0;phase_p=union_frames_4p-3_through_4p",
            "latent_phases": 21,
            "packed_token_count": 19_530,
            "patch_cols": 30,
            "patch_rows": 31,
            "scheduler_target_packed_token": "phase*930+side_local_token",
            "side_local_spatial_token": "patch_y*30+patch_x",
            "tokens_per_phase": 930,
        }
        or payload.get("invariants") != {
            "all_21_latent_phases_bound": True,
            "all_81_frames_bound": True,
            "bone_correspondence_bijective_every_phase": True,
            "bone_trajectory_in_bounds": True,
            "dog_identity_core_nonempty_every_phase": True,
            "dog_patient_projection_disjoint_every_phase": True,
            "hold_at_least_10_frames": True,
            "source_target_bone_token_count_equal_every_phase": True,
        }
        or payload.get("claim_limits") != {
            "hand_authored_oracle": True,
            "learned_representation": False,
            "purpose": "frozen-renderer-feasibility-and-condition-consumption-canary",
            "renderer_execution": False,
            "scientific_claim_authorized": False,
            "target_motion_ground_truth": False,
        }
        or not isinstance(payload.get("frames"), list)
        or len(payload["frames"]) != 81
        or not isinstance(payload.get("latent_phases"), list)
        or len(payload["latent_phases"]) != 21
    ):
        raise ObjectTrajectoryEvalError("trajectory scaffold identity/authority differs")


def _validate_scaffold_audit_payload(
    payload: Mapping[str, Any], *, source: Mapping[str, Any],
    externals: Mapping[str, Mapping[str, Any]],
) -> None:
    fields = {
        "schema_version", "status", "auditor_role", "artifact", "module",
        "upstream_authorities", "frame_rows", "latent_rows", "claim_replay",
        "tests", "launch_or_package_performed", "audit_digest",
    }
    unsigned = dict(payload)
    audit_digest = unsigned.pop("audit_digest", None)
    scaffold = externals["trajectory_scaffold"]
    if (
        set(payload) != fields
        or audit_digest != EXPECTED_SCAFFOLD_AUDIT_DIGEST
        or object_sha256(unsigned) != audit_digest
        or payload.get("schema_version") != EXPECTED_SCAFFOLD_AUDIT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("auditor_role")
        != "independent_contract_replay_without_importing_scaffold_module"
        or payload.get("launch_or_package_performed") is not False
        or payload.get("artifact") != {
            "path": "artifacts/case01_oracle_object_trajectory_v1/scaffold.json",
            "sha256": scaffold["sha256"],
            "size": scaffold["size"],
            "schema_version": EXPECTED_TRAJECTORY_SCAFFOLD_SCHEMA,
            "artifact_digest": scaffold["payload_digest"],
            "artifact_digest_recomputed_excluding_artifact_digest": True,
            "canonical_json_plus_lf": True,
        }
        or payload.get("module") != {
            "path": "methods/bernini_action_editing/case01_oracle_object_trajectory_v1.py",
            "sha256": EXPECTED_TRAJECTORY_SCAFFOLD_MODULE_SHA256,
            "size": EXPECTED_TRAJECTORY_SCAFFOLD_MODULE_SIZE,
        }
        or payload.get("upstream_authorities") != {
            "exact_original_source": {
                "sha256": source["sha256"], "size": source["size"],
            },
            "bone_removed_auxiliary": {
                "sha256": externals["aux_bone_removed_source"]["sha256"],
                "size": externals["aux_bone_removed_source"]["size"],
            },
            "stage0_receipt": {
                "sha256": externals["stage0_masks"]["sha256"],
                "size": externals["stage0_masks"]["size"],
                "receipt_digest": externals["stage0_masks"]["payload_digest"],
                "mask_count": 162,
                "visible_frame_counts": {"bone": 81, "dog": 81},
            },
            "g0_sparse_annotations": {
                "sha256": externals["g0_mouth_track"]["sha256"],
                "size": externals["g0_mouth_track"]["size"],
                "frame_indices": [0, 10, 20, 30, 40, 50, 60, 70, 80],
            },
        }
        or payload.get("frame_rows") != {
            "count": 81,
            "ordered_0_through_80": True,
            "pre_lift_shift_zero_frames_0_through_36": True,
            "hold_shift_nonzero_frames_61_through_80": True,
            "source_target_bone_count_equal_all_frames": True,
            "terminal_hold_frame_count": 20,
        }
        or payload.get("latent_rows") != {
            "count": 21,
            "layout": "21x31x30",
            "packed_token_count": 19_530,
            "phase_windows_exact": True,
            "packed_target_indices_in_bounds": True,
            "bone_correspondence_bijective_all_phases": True,
            "origin_clear_exact_for_nonzero_shift": True,
            "target_bone_within_responsibility_all_phases": True,
            "dog_patient_projection_disjoint_all_phases": True,
            "first_nonzero_shift_phase": 10,
            "hold_phase_count": 5,
            "source_bone_token_count_minmax": [17, 20],
            "dog_identity_token_count_minmax": [53, 112],
        }
        or payload.get("claim_replay") != {
            "hand_authored_oracle": True,
            "learned_representation": False,
            "renderer_execution": False,
            "scientific_claim_authorized": False,
            "target_motion_ground_truth": False,
            "zero_training": True,
        }
        or payload.get("tests") != {
            "normal": "6/6 PASS", "optimized": "6/6 PASS",
        }
    ):
        raise ObjectTrajectoryEvalError("scaffold independent audit differs")


def _routing_for_arm(arm: str) -> dict[str, Any]:
    if arm not in ARM_ORDER:
        raise ObjectTrajectoryEvalError("arm differs")
    external = arm not in {"null_before", "null_after"}
    route_enabled = arm in {"trajectory_bone_only", "trajectory_dog_bone"}
    dog = arm == "trajectory_dog_bone"
    bone = route_enabled
    return {
        "oracle_assets_validated": external,
        "direct_runtime_conditions": (
            list(BASE_CONDITION_NAMES + (
                "object_trajectory_scaffold", "aux_bone_removed_source"
            ))
            if external else list(BASE_CONDITION_NAMES)
        ),
        "derived_scaffold_authorities": (
            ["stage0_object_masks", "g0_mouth_track"] if external else []
        ),
        "renderer_conditions_consumed": (
            list(BASE_CONDITION_NAMES + (
                "object_trajectory_scaffold", "aux_bone_removed_source"
            ))
            if route_enabled else list(BASE_CONDITION_NAMES)
        ),
        "oracle_runtime_conditions_consumed": (
            ["object_trajectory_scaffold", "aux_bone_removed_source"]
            if route_enabled else []
        ),
        "raw_stage0_masks_accessed_at_runtime": False,
        "raw_g0_annotations_accessed_at_runtime": False,
        "route_enabled": route_enabled,
        "route_off_after_condition_validation": arm == "route_off",
        "dog_identity_projection_enabled": dog,
        "source_bone_trajectory_projection_enabled": bone,
        "bone_origin_clear_enabled": bone,
        "single_source_bone_instance_required": bone,
    }


def _expected_condition_names(arm: str) -> list[str]:
    names = list(BASE_CONDITION_NAMES)
    if arm not in {"null_before", "null_after"}:
        names.extend(EXTERNAL_CONDITION_NAMES)
    return names


def _validate_checkpoint(value: Mapping[str, Any], *, require_complete: bool) -> dict[str, Any]:
    fields = set(EXPECTED_CHECKPOINT) | {"path", "pin_complete"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ObjectTrajectoryEvalError("checkpoint schema differs")
    row = dict(value)
    path = row.get("path")
    if (
        any(row.get(key) != expected for key, expected in EXPECTED_CHECKPOINT.items())
        or type(row.get("pin_complete")) is not bool
        or not isinstance(path, str)
    ):
        raise ObjectTrajectoryEvalError("checkpoint identity differs")
    if row["pin_complete"]:
        if (
            not Path(path).is_absolute()
            or os.path.normpath(path) != path
            or PLACEHOLDER_PREFIX in path
        ):
            raise ObjectTrajectoryEvalError("complete checkpoint path differs")
    elif not path.startswith(PLACEHOLDER_PREFIX):
        raise ObjectTrajectoryEvalError("incomplete checkpoint path differs")
    if require_complete and not row["pin_complete"]:
        raise ObjectTrajectoryEvalError("checkpoint pin is incomplete")
    return row


def _validate_producer(value: Mapping[str, Any], *, require_complete: bool, reopen: bool) -> dict[str, Any]:
    fields = {
        "inference_receipt_schemas",
        "infer_lora_path",
        "infer_lora_sha256",
        "infer_lora_size",
        "infer_lora_role",
        "inference_wrapper_path",
        "inference_wrapper_sha256",
        "inference_wrapper_size",
        "trajectory_projection_module_path",
        "trajectory_projection_module_sha256",
        "trajectory_projection_module_size",
        "trajectory_scaffold_module_path",
        "trajectory_scaffold_module_sha256",
        "trajectory_scaffold_module_size",
        "ffprobe_path",
        "ffprobe_sha256",
        "ffprobe_size",
        "method_source_revision",
        "method_source_archive_sha256",
        "pins_complete",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ObjectTrajectoryEvalError("producer schema differs")
    row = dict(value)
    integer_fields = {
        "infer_lora_size", "inference_wrapper_size",
        "trajectory_projection_module_size", "trajectory_scaffold_module_size",
        "ffprobe_size",
    }
    string_fields = fields - {"pins_complete", "inference_receipt_schemas"} - integer_fields
    receipt_schemas = row.get("inference_receipt_schemas")
    if (
        receipt_schemas
        != {
            "off": LEGACY_INFERENCE_RECEIPT_SCHEMA,
            "route_or_active": INFERENCE_RECEIPT_SCHEMA,
        }
        or type(row.get("pins_complete")) is not bool
        or any(not isinstance(row.get(key), str) for key in string_fields)
        or any(type(row.get(key)) is not int for key in integer_fields)
        or row.get("infer_lora_role")
        != "frozen_legacy_exact5_infer_lora_not_workspace_head"
        or row.get("infer_lora_sha256") != EXPECTED_LEGACY_INFER_LORA_SHA256
        or row.get("infer_lora_size") != EXPECTED_LEGACY_INFER_LORA_SIZE
    ):
        raise ObjectTrajectoryEvalError("producer value differs")
    pin_triples = (
        ("infer_lora_path", "infer_lora_sha256", "infer_lora_size"),
        (
            "inference_wrapper_path", "inference_wrapper_sha256",
            "inference_wrapper_size",
        ),
        (
            "trajectory_projection_module_path",
            "trajectory_projection_module_sha256",
            "trajectory_projection_module_size",
        ),
        (
            "trajectory_scaffold_module_path",
            "trajectory_scaffold_module_sha256",
            "trajectory_scaffold_module_size",
        ),
        ("ffprobe_path", "ffprobe_sha256", "ffprobe_size"),
    )
    frozen_source_pins = {
        "infer_lora_path": (
            EXPECTED_LEGACY_INFER_LORA_SHA256,
            EXPECTED_LEGACY_INFER_LORA_SIZE,
        ),
        "inference_wrapper_path": (
            EXPECTED_INFERENCE_WRAPPER_SHA256,
            EXPECTED_INFERENCE_WRAPPER_SIZE,
        ),
        "trajectory_projection_module_path": (
            EXPECTED_TRAJECTORY_PROJECTION_MODULE_SHA256,
            EXPECTED_TRAJECTORY_PROJECTION_MODULE_SIZE,
        ),
        "trajectory_scaffold_module_path": (
            EXPECTED_TRAJECTORY_SCAFFOLD_MODULE_SHA256,
            EXPECTED_TRAJECTORY_SCAFFOLD_MODULE_SIZE,
        ),
    }
    incomplete_count = 0
    for path_key, sha_key, size_key in pin_triples:
        path = row[path_key]
        frozen_pin = frozen_source_pins.get(path_key)
        if frozen_pin is not None and (
            row[sha_key] != frozen_pin[0] or row[size_key] != frozen_pin[1]
        ):
            raise ObjectTrajectoryEvalError("frozen producer source pin differs")
        if path.startswith(PLACEHOLDER_PREFIX):
            incomplete_count += 1
            if frozen_pin is not None:
                pass
            elif (
                not row[sha_key].startswith(PLACEHOLDER_PREFIX)
                or row[size_key] != 0
            ):
                raise ObjectTrajectoryEvalError("incomplete producer source pin differs")
        elif (
            not Path(path).is_absolute()
            or os.path.normpath(path) != path
            or SHA256_RE.fullmatch(row[sha_key]) is None
            or row[size_key] <= 0
            or (
                path_key in EXPECTED_PRODUCER_BASENAMES
                and Path(path).name != EXPECTED_PRODUCER_BASENAMES[path_key]
            )
        ):
            raise ObjectTrajectoryEvalError("producer source pin differs")
        elif reopen:
            _stable_file(
                path, expected_sha256=row[sha_key], expected_size=row[size_key]
            )
    if row["pins_complete"]:
        if incomplete_count:
            raise ObjectTrajectoryEvalError("complete producer retains placeholders")
        if (
            not row["method_source_revision"]
            or SHA256_RE.fullmatch(row["method_source_archive_sha256"]) is None
        ):
            raise ObjectTrajectoryEvalError("producer revision pin differs")
    else:
        if incomplete_count == 0:
            raise ObjectTrajectoryEvalError("incomplete producer lacks placeholders")
    if require_complete and not row["pins_complete"]:
        raise ObjectTrajectoryEvalError("producer pins are incomplete")
    return row


def _incomplete_reasons(
    checkpoint: Mapping[str, Any],
    producer: Mapping[str, Any],
    source: Mapping[str, Any],
    externals: Mapping[str, Mapping[str, Any]],
    admissions: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if checkpoint.get("pin_complete") is not True:
        reasons.append("checkpoint_manifest_path_incomplete")
    if producer.get("pins_complete") is not True:
        reasons.append("custom_inference_producer_pins_incomplete")
    if source.get("complete") is not True:
        reasons.append("exact_original_source_pin_incomplete")
    for key in EXTERNAL_AUTHORITY_KEYS:
        if externals[key].get("complete") is not True:
            reasons.append(f"{key}_authority_incomplete")
    for key in ADMISSION_AUTHORITY_KEYS:
        if admissions[key].get("complete") is not True:
            reasons.append(f"{key}_admission_authority_incomplete")
    return reasons


def incomplete_checkpoint_manifest() -> dict[str, Any]:
    """Return the fixed R64 identity with only its physical path unresolved."""

    return {
        "path": PLACEHOLDER_PREFIX + "R64_CHECKPOINT_MANIFEST_PATH__",
        "pin_complete": False,
        **EXPECTED_CHECKPOINT,
    }


def incomplete_producer() -> dict[str, Any]:
    """Return the default non-launchable producer placeholder closure."""

    return {
        "inference_receipt_schemas": {
            "off": LEGACY_INFERENCE_RECEIPT_SCHEMA,
            "route_or_active": INFERENCE_RECEIPT_SCHEMA,
        },
        "infer_lora_path": PLACEHOLDER_PREFIX + "BASE_INFER_LORA_PATH__",
        "infer_lora_sha256": EXPECTED_LEGACY_INFER_LORA_SHA256,
        "infer_lora_size": EXPECTED_LEGACY_INFER_LORA_SIZE,
        "infer_lora_role": "frozen_legacy_exact5_infer_lora_not_workspace_head",
        "inference_wrapper_path": PLACEHOLDER_PREFIX + "OBJECT_TRAJECTORY_WRAPPER_PATH__",
        "inference_wrapper_sha256": EXPECTED_INFERENCE_WRAPPER_SHA256,
        "inference_wrapper_size": EXPECTED_INFERENCE_WRAPPER_SIZE,
        "trajectory_projection_module_path": PLACEHOLDER_PREFIX + "TRAJECTORY_PROJECTION_MODULE_PATH__",
        "trajectory_projection_module_sha256": EXPECTED_TRAJECTORY_PROJECTION_MODULE_SHA256,
        "trajectory_projection_module_size": EXPECTED_TRAJECTORY_PROJECTION_MODULE_SIZE,
        "trajectory_scaffold_module_path": PLACEHOLDER_PREFIX + "TRAJECTORY_SCAFFOLD_MODULE_PATH__",
        "trajectory_scaffold_module_sha256": EXPECTED_TRAJECTORY_SCAFFOLD_MODULE_SHA256,
        "trajectory_scaffold_module_size": EXPECTED_TRAJECTORY_SCAFFOLD_MODULE_SIZE,
        "ffprobe_path": PLACEHOLDER_PREFIX + "FFPROBE_PATH__",
        "ffprobe_sha256": PLACEHOLDER_PREFIX + "FFPROBE_SHA256__",
        "ffprobe_size": 0,
        "method_source_revision": PLACEHOLDER_PREFIX + "METHOD_SOURCE_REVISION__",
        "method_source_archive_sha256": PLACEHOLDER_PREFIX + "METHOD_SOURCE_ARCHIVE_SHA256__",
        "pins_complete": False,
    }


def validate_plan(
    plan: Mapping[str, Any], *, reopen_sources: bool = False,
    require_fresh_outputs: bool = True, require_launchable: bool = False,
) -> dict[str, Any]:
    fields = {
        "schema_version", "experiment_id", "status", "production_ready",
        "launch_allowed", "hold_reasons", "source_authority",
        "condition_authorities", "admission_authorities", "checkpoint_manifest", "producer",
        "condition_contract", "arms", "task_count", "tasks", "claim_limits",
        "plan_digest",
    }
    if not isinstance(plan, Mapping) or set(plan) != fields:
        raise ObjectTrajectoryEvalError("plan root schema differs")
    _strict_digest(plan, "plan_digest", label="trajectory plan")
    launch_allowed = plan.get("launch_allowed")
    if type(launch_allowed) is not bool:
        raise ObjectTrajectoryEvalError("launch_allowed is not boolean")
    checkpoint = _validate_checkpoint(
        plan.get("checkpoint_manifest", {}), require_complete=require_launchable or launch_allowed
    )
    producer = _validate_producer(
        plan.get("producer", {}),
        require_complete=require_launchable or launch_allowed,
        reopen=reopen_sources and (require_launchable or launch_allowed),
    )
    source = validate_file_authority(
        plan.get("source_authority", {}),
        expected_role="exact_original_source",
        reopen=reopen_sources,
    )
    externals_value = plan.get("condition_authorities")
    if not isinstance(externals_value, Mapping) or set(externals_value) != set(EXTERNAL_AUTHORITY_KEYS):
        raise ObjectTrajectoryEvalError("external condition authority set differs")
    externals = {
        key: validate_file_authority(
            externals_value[key], expected_role=key, reopen=reopen_sources
        )
        for key in EXTERNAL_AUTHORITY_KEYS
    }
    admissions_value = plan.get("admission_authorities")
    if (
        not isinstance(admissions_value, Mapping)
        or set(admissions_value) != set(ADMISSION_AUTHORITY_KEYS)
    ):
        raise ObjectTrajectoryEvalError("admission authority set differs")
    admissions = {
        key: validate_file_authority(
            admissions_value[key], expected_role=key, reopen=reopen_sources
        )
        for key in ADMISSION_AUTHORITY_KEYS
    }
    _require_exact_complete_authority(
        source, expected_sha256=EXPECTED_SOURCE_SHA256,
        expected_size=EXPECTED_SOURCE_SIZE,
        expected_payload_digest=EXPECTED_SOURCE_SHA256,
        label="exact_original source",
    )
    _require_exact_complete_authority(
        externals["stage0_masks"], expected_sha256=EXPECTED_STAGE0_SHA256,
        expected_size=EXPECTED_STAGE0_SIZE,
        expected_payload_digest=EXPECTED_STAGE0_RECEIPT_DIGEST,
        label="Stage0 mask receipt",
    )
    _require_exact_complete_authority(
        externals["g0_mouth_track"], expected_sha256=EXPECTED_G0_SHA256,
        expected_size=EXPECTED_G0_SIZE, expected_payload_digest=EXPECTED_G0_SHA256,
        label="G0 mouth track",
    )
    aux = externals["aux_bone_removed_source"]
    _require_exact_complete_authority(
        aux, expected_sha256=EXPECTED_AUX_REMOVED_SHA256,
        expected_size=EXPECTED_AUX_REMOVED_SIZE,
        expected_payload_digest=EXPECTED_AUX_REMOVED_SHA256,
        label="bone-removed auxiliary source",
    )
    scaffold = externals["trajectory_scaffold"]
    _require_exact_complete_authority(
        scaffold, expected_sha256=EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_SHA256,
        expected_size=EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_SIZE,
        expected_payload_digest=EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST,
        label="trajectory scaffold",
    )
    scaffold_audit = admissions["scaffold_independent_audit"]
    _require_exact_complete_authority(
        scaffold_audit, expected_sha256=EXPECTED_SCAFFOLD_AUDIT_SHA256,
        expected_size=EXPECTED_SCAFFOLD_AUDIT_SIZE,
        expected_payload_digest=EXPECTED_SCAFFOLD_AUDIT_DIGEST,
        label="scaffold independent audit",
    )
    if reopen_sources:
        scaffold_payload = _load_canonical_json_authority(
            scaffold, label="trajectory scaffold"
        )
        _validate_scaffold_payload(
            scaffold_payload, source=source, externals=externals
        )
        audit_payload = _load_canonical_json_authority(
            scaffold_audit, label="scaffold independent audit"
        )
        _validate_scaffold_audit_payload(
            audit_payload, source=source, externals=externals
        )
    reasons = _incomplete_reasons(
        checkpoint, producer, source, externals, admissions
    )
    if not launch_allowed:
        reasons.append("explicit_launch_release_not_granted")
    expected_status = (
        "READY_FOR_EXPLICIT_LOCAL_LAUNCH" if launch_allowed
        else "HOLD_INCOMPLETE_PRODUCER_OR_AUTHORITY"
    )
    condition = plan.get("condition_contract")
    tasks = plan.get("tasks")
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("experiment_id") != EXPERIMENT_ID
        or plan.get("status") != expected_status
        or plan.get("production_ready") is not False
        or plan.get("hold_reasons") != ([] if launch_allowed else reasons)
        or (launch_allowed and reasons)
        or plan.get("arms") != list(ARM_ORDER)
        or plan.get("task_count") != 5
        or not isinstance(condition, Mapping)
        or condition != {
            "iid": IID,
            "instruction": INSTRUCTION,
            "instruction_sha256": INSTRUCTION_SHA256,
            "seed": SEED,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "num_frames": 81,
            "source_variant": "exact_original",
            "source_onset_policy": "hard1_every_step",
            "checkpoint_profile": FULL644_PROFILE,
            "null_envelope_coordinates_equal_required": True,
            "null_envelope_output_byte_equality_required": False,
            "external_authorities_required_for_route_off_and_active_arms": True,
        }
        or not isinstance(tasks, list)
        or [task.get("task_id") for task in tasks] != list(TASK_IDS)
        or plan.get("claim_limits") != {
            "engineering_oracle_only": True,
            "learned_object_centric_method_claim_authorized": False,
            "causal_claim_authorized": False,
            "scientific_claim_authorized": False,
            "formal_claim_authorized": False,
            "manual_review_required": True,
        }
    ):
        raise ObjectTrajectoryEvalError("plan identity/condition closure differs")
    output_roots: set[Path] = set()
    publication_paths: set[Path] = set()
    for index, (task, arm) in enumerate(zip(tasks, ARM_ORDER)):
        output = task.get("output") if isinstance(task, Mapping) else None
        adapter = task.get("adapter") if isinstance(task, Mapping) else None
        external = task.get("external_conditions") if isinstance(task, Mapping) else None
        video_raw = output.get("video_path") if isinstance(output, Mapping) else None
        receipt_raw = output.get("receipt_path") if isinstance(output, Mapping) else None
        video = Path(video_raw) if isinstance(video_raw, str) else Path("")
        receipt = Path(receipt_raw) if isinstance(receipt_raw, str) else Path("")
        expected_external = (
            {} if arm in {"null_before", "null_after"}
            else {key: externals[key] for key in EXTERNAL_AUTHORITY_KEYS}
        )
        if (
            not isinstance(task, Mapping)
            or set(task) != {
                "task_id", "case_index", "iid", "oracle_arm", "source_video",
                "source_video_sha256", "instruction", "instruction_sha256", "seed",
                "num_inference_steps", "source_onset_policy", "arm", "adapter",
                "accepted_model_conditions", "external_conditions", "routing", "output",
            }
            or task.get("task_id") != TASK_IDS[index]
            or task.get("case_index") != 1
            or task.get("iid") != IID
            or task.get("oracle_arm") != arm
            or task.get("source_video") != source["path"]
            or task.get("source_video_sha256") != EXPECTED_SOURCE_SHA256
            or task.get("instruction") != INSTRUCTION
            or task.get("instruction_sha256") != INSTRUCTION_SHA256
            or task.get("seed") != SEED
            or task.get("num_inference_steps") != NUM_INFERENCE_STEPS
            or task.get("source_onset_policy") != "hard1_every_step"
            or task.get("arm") != "full644"
            or task.get("accepted_model_conditions") != _expected_condition_names(arm)
            or external != expected_external
            or task.get("routing") != _routing_for_arm(arm)
            or not isinstance(adapter, Mapping)
            or adapter != {
                "checkpoint_root": (
                    str(Path(checkpoint["path"]).parent)
                    if checkpoint["pin_complete"] else PLACEHOLDER_PREFIX + "CHECKPOINT_ROOT__"
                ),
                "checkpoint_manifest": checkpoint,
                "adapter_model_sha256": EXPECTED_CHECKPOINT["adapter_model_sha256"],
                "profile": FULL644_PROFILE,
            }
            or not isinstance(output, Mapping)
            or set(output) != {"video_path", "receipt_path", "create_only"}
            or output.get("create_only") is not True
            or not video.is_absolute()
            or not receipt.is_absolute()
            or os.path.normpath(video_raw) != video_raw
            or os.path.normpath(receipt_raw) != receipt_raw
            or video.name != f"{TASK_IDS[index]}.mp4"
            or receipt != video.with_name(video.name + ".receipt.json")
        ):
            raise ObjectTrajectoryEvalError(f"task closure differs: {arm}")
        output_roots.add(video.parent)
        publication_paths.update((video, receipt))
        if require_fresh_outputs and (
            video.exists() or video.is_symlink() or receipt.exists() or receipt.is_symlink()
        ):
            raise ObjectTrajectoryEvalError(f"planned output is not fresh: {arm}")
    if len(output_roots) != 1 or len(publication_paths) != 10:
        raise ObjectTrajectoryEvalError("five-arm publication closure differs")
    output_root = next(iter(output_roots))
    if (
        not output_root.is_absolute()
        or os.path.normpath(str(output_root)) != str(output_root)
        or not output_root.is_dir()
        or output_root.is_symlink()
        or output_root.resolve(strict=True) != output_root
    ):
        raise ObjectTrajectoryEvalError("publication root differs")
    if require_launchable and launch_allowed is not True:
        raise ObjectTrajectoryEvalError("HOLD plan is not launchable")
    return dict(plan)


def build_plan(
    *,
    source_authority: Mapping[str, Any],
    condition_authorities: Mapping[str, Mapping[str, Any]],
    admission_authorities: Mapping[str, Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    producer: Mapping[str, Any],
    output_root: str | Path,
    launch_allowed: bool = False,
) -> dict[str, Any]:
    source = validate_file_authority(
        source_authority, expected_role="exact_original_source", reopen=False
    )
    if set(condition_authorities) != set(EXTERNAL_AUTHORITY_KEYS):
        raise ObjectTrajectoryEvalError("condition authority keys differ")
    externals = {
        key: validate_file_authority(
            condition_authorities[key], expected_role=key, reopen=False
        )
        for key in EXTERNAL_AUTHORITY_KEYS
    }
    if set(admission_authorities) != set(ADMISSION_AUTHORITY_KEYS):
        raise ObjectTrajectoryEvalError("admission authority keys differ")
    admissions = {
        key: validate_file_authority(
            admission_authorities[key], expected_role=key, reopen=False
        )
        for key in ADMISSION_AUTHORITY_KEYS
    }
    checkpoint = _validate_checkpoint(checkpoint_manifest, require_complete=False)
    producer_value = _validate_producer(producer, require_complete=False, reopen=False)
    root = Path(output_root)
    if (
        not root.is_absolute()
        or os.path.normpath(str(root)) != str(root)
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
    ):
        raise ObjectTrajectoryEvalError("output root differs")
    reasons = _incomplete_reasons(
        checkpoint, producer_value, source, externals, admissions
    )
    if launch_allowed and reasons:
        raise ObjectTrajectoryEvalError("cannot mark an incomplete plan launchable")
    if not launch_allowed:
        reasons.append("explicit_launch_release_not_granted")
    tasks: list[dict[str, Any]] = []
    for arm, task_id in zip(ARM_ORDER, TASK_IDS):
        video = root / f"{task_id}.mp4"
        tasks.append(
            {
                "task_id": task_id,
                "case_index": 1,
                "iid": IID,
                "oracle_arm": arm,
                "source_video": source["path"],
                "source_video_sha256": source["sha256"],
                "instruction": INSTRUCTION,
                "instruction_sha256": INSTRUCTION_SHA256,
                "seed": SEED,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "source_onset_policy": "hard1_every_step",
                "arm": "full644",
                "adapter": {
                    "checkpoint_root": (
                        str(Path(checkpoint["path"]).parent)
                        if checkpoint["pin_complete"] else PLACEHOLDER_PREFIX + "CHECKPOINT_ROOT__"
                    ),
                    "checkpoint_manifest": checkpoint,
                    "adapter_model_sha256": EXPECTED_CHECKPOINT["adapter_model_sha256"],
                    "profile": FULL644_PROFILE,
                },
                "accepted_model_conditions": _expected_condition_names(arm),
                "external_conditions": (
                    {} if arm in {"null_before", "null_after"}
                    else {key: externals[key] for key in EXTERNAL_AUTHORITY_KEYS}
                ),
                "routing": _routing_for_arm(arm),
                "output": {
                    "video_path": str(video),
                    "receipt_path": str(video.with_name(video.name + ".receipt.json")),
                    "create_only": True,
                },
            }
        )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "READY_FOR_EXPLICIT_LOCAL_LAUNCH" if launch_allowed
            else "HOLD_INCOMPLETE_PRODUCER_OR_AUTHORITY"
        ),
        "production_ready": False,
        "launch_allowed": launch_allowed,
        "hold_reasons": [] if launch_allowed else reasons,
        "source_authority": source,
        "condition_authorities": externals,
        "admission_authorities": admissions,
        "checkpoint_manifest": checkpoint,
        "producer": producer_value,
        "condition_contract": {
            "iid": IID,
            "instruction": INSTRUCTION,
            "instruction_sha256": INSTRUCTION_SHA256,
            "seed": SEED,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "num_frames": 81,
            "source_variant": "exact_original",
            "source_onset_policy": "hard1_every_step",
            "checkpoint_profile": FULL644_PROFILE,
            "null_envelope_coordinates_equal_required": True,
            "null_envelope_output_byte_equality_required": False,
            "external_authorities_required_for_route_off_and_active_arms": True,
        },
        "arms": list(ARM_ORDER),
        "task_count": 5,
        "tasks": tasks,
        "claim_limits": {
            "engineering_oracle_only": True,
            "learned_object_centric_method_claim_authorized": False,
            "causal_claim_authorized": False,
            "scientific_claim_authorized": False,
            "formal_claim_authorized": False,
            "manual_review_required": True,
        },
    }
    plan["plan_digest"] = object_sha256(plan)
    return validate_plan(plan, reopen_sources=False)


def load_plan(path_value: str | Path, expected_sha256: str) -> dict[str, Any]:
    raw, _, _ = _stable_file(
        path_value, expected_sha256=expected_sha256, return_bytes=True
    )
    if raw is None:
        raise ObjectTrajectoryEvalError("stable plan reader returned no bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise ObjectTrajectoryEvalError("plan is not strict JSON") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ObjectTrajectoryEvalError("plan is not canonical JSON plus LF")
    return validate_plan(
        value, reopen_sources=True, require_fresh_outputs=True,
        require_launchable=True,
    )


def _is_json_int(value: Any, expected: int | None = None) -> bool:
    return type(value) is int and (expected is None or value == expected)


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float("-inf") < float(value) < float("inf")
    )


def _validate_legacy_source_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the real v5 ``hard1_every_step`` UniPC trace locally.

    The frozen v1 evaluator hard-codes ``source_onset_policy=none`` and cannot
    truthfully validate this coordinate, so no legacy receipt verifier is used
    here.
    """

    fields = {
        "schema_version", "policy", "integrator", "prediction_type", "phase",
        "latent_phases", "initial_packed_noise_captured", "step_count",
        "expected_steps", "steps", "target_video_accessed",
        "identity_or_background_claim",
    }
    steps = trace.get("steps") if isinstance(trace, Mapping) else None
    if (
        not isinstance(trace, Mapping)
        or set(trace) != fields
        or trace.get("schema_version") != "bernini-source-phase0-unipc-clamp-v1"
        or trace.get("policy") != "hard1_every_step"
        or trace.get("integrator") != "original_unipc_scheduler_step"
        or trace.get("prediction_type") != "flow_prediction"
        or not _is_json_int(trace.get("phase"), 0)
        or not _is_json_int(trace.get("latent_phases"), 21)
        or trace.get("initial_packed_noise_captured") is not True
        or not _is_json_int(trace.get("step_count"), NUM_INFERENCE_STEPS)
        or not _is_json_int(trace.get("expected_steps"), NUM_INFERENCE_STEPS)
        or trace.get("target_video_accessed") is not False
        or trace.get("identity_or_background_claim") is not False
        or not isinstance(steps, list)
        or len(steps) != NUM_INFERENCE_STEPS
    ):
        raise ObjectTrajectoryEvalError("legacy hard1 sampling trace differs")
    previous_next: float | None = None
    for index, row in enumerate(steps):
        if not isinstance(row, Mapping) or set(row) != {
            "step_index", "timestep", "sigma", "next_sigma", "phase0_velocity",
            "phase0_post_step", "other_phases_projected",
            "original_scheduler_step_calls",
        }:
            raise ObjectTrajectoryEvalError("legacy hard1 trace row schema differs")
        sigma = row.get("sigma")
        next_sigma = row.get("next_sigma")
        if (
            not _is_json_int(row.get("step_index"), index)
            or not _is_finite_number(row.get("timestep"))
            or not _is_finite_number(sigma)
            or not _is_finite_number(next_sigma)
            or not (0.0 <= float(next_sigma) < float(sigma) <= 1.0)
            or (previous_next is not None and float(sigma) != previous_next)
            or row.get("phase0_velocity")
            != "captured_epsilon_minus_clean_source"
            or row.get("phase0_post_step")
            != "source_noise_flow_trajectory_projection"
            or row.get("other_phases_projected") is not False
            or not _is_json_int(row.get("original_scheduler_step_calls"), 1)
        ):
            raise ObjectTrajectoryEvalError("legacy hard1 trace row differs")
        previous_next = float(next_sigma)
    if previous_next != 0.0:
        raise ObjectTrajectoryEvalError("legacy hard1 trace lacks terminal zero")
    return dict(trace)


def _validate_common_v5_receipt_root(
    receipt: Mapping[str, Any], task: Mapping[str, Any],
    producer: Mapping[str, Any], *, custom_object_oracle: bool,
) -> dict[str, Mapping[str, Any]]:
    """Validate the unchanged legacy-v5 closure without its false/none ABI."""

    top_fields = {
        "schema_version", "infer_lora_source_sha256", "method_source_revision",
        "method_source_archive_sha256", "bernini_commit", "veomni_commit",
        "bernini_inference_files", "checkpoint_tree_sha256", "adapter", "input",
        "preprocessing", "prompt_contract", "sampling", "output",
        "runtime_versions", "experimental_inference", "production_claim_forbidden",
        "scientific_claim_authorized", "consumption_input_digest",
        "task_input_digest", "model_consumption", "receipt_digest",
    }
    if custom_object_oracle:
        top_fields.add("object_oracle")
    if not isinstance(receipt, Mapping) or set(receipt) != top_fields:
        raise ObjectTrajectoryEvalError("inference receipt root schema differs")
    _strict_digest(receipt, "receipt_digest", label="inference receipt")
    names = (
        "input", "preprocessing", "prompt_contract", "sampling", "adapter",
        "output", "model_consumption", "runtime_versions",
    )
    values = {name: receipt.get(name) for name in names}
    if any(not isinstance(value, Mapping) for value in values.values()):
        raise ObjectTrajectoryEvalError("inference receipt core object differs")
    input_value = values["input"]
    preprocessing = values["preprocessing"]
    prompt = values["prompt_contract"]
    sampling = values["sampling"]
    adapter = values["adapter"]
    output = values["output"]
    consumption = values["model_consumption"]
    runtime_versions = values["runtime_versions"]
    input_fields = {
        "source_video_path", "source_video_sha256", "instruction_utf8_sha256",
        "instruction_utf8_bytes", "accepted_model_conditions",
        "target_video_argument", "target_accessed_by_inference",
        "external_mask_or_swept_tube", "external_tracking_pose_or_trajectory",
        "reference_image_or_video", "external_shared_i0",
        "source_video_physical_authority",
        "source_video_physical_authority_digest", "retained_source_fd_consumed",
        "source_video_pre_and_post_decode_rehashed",
    }
    if custom_object_oracle:
        input_fields.update(
            {
                "direct_runtime_conditions", "derived_scaffold_authorities",
                "raw_stage0_masks_accessed_at_runtime",
                "raw_g0_annotations_accessed_at_runtime",
            }
        )
    source_authority = input_value.get("source_video_physical_authority")
    source_fields = {
        "path", "sha256", "size", "mode", "device", "inode", "uid", "gid",
        "nlink", "rdev", "blocks", "mtime_ns", "ctime_ns",
    }
    expected_conditions = (
        task["accepted_model_conditions"]
        if custom_object_oracle else list(BASE_CONDITION_NAMES)
    )
    external_truth = custom_object_oracle
    if (
        set(input_value) != input_fields
        or input_value.get("source_video_path") != task["source_video"]
        or input_value.get("source_video_sha256") != task["source_video_sha256"]
        or input_value.get("instruction_utf8_sha256") != task["instruction_sha256"]
        or not _is_json_int(
            input_value.get("instruction_utf8_bytes"),
            len(task["instruction"].encode("utf-8")),
        )
        or input_value.get("accepted_model_conditions") != expected_conditions
        or (
            custom_object_oracle
            and input_value.get("direct_runtime_conditions")
            != task["routing"]["direct_runtime_conditions"]
        )
        or (
            custom_object_oracle
            and input_value.get("derived_scaffold_authorities")
            != task["routing"]["derived_scaffold_authorities"]
        )
        or (
            custom_object_oracle
            and input_value.get("raw_stage0_masks_accessed_at_runtime")
            is not task["routing"]["raw_stage0_masks_accessed_at_runtime"]
        )
        or (
            custom_object_oracle
            and input_value.get("raw_g0_annotations_accessed_at_runtime")
            is not task["routing"]["raw_g0_annotations_accessed_at_runtime"]
        )
        or input_value.get("target_video_argument") is not False
        or input_value.get("target_accessed_by_inference") is not False
        or input_value.get("external_mask_or_swept_tube") is not external_truth
        or input_value.get("external_tracking_pose_or_trajectory") is not external_truth
        or input_value.get("reference_image_or_video") is not external_truth
        or input_value.get("external_shared_i0") is not False
        or input_value.get("retained_source_fd_consumed") is not True
        or input_value.get("source_video_pre_and_post_decode_rehashed") is not True
        or not isinstance(source_authority, Mapping)
        or set(source_authority) != source_fields
        or source_authority.get("path") != task["source_video"]
        or source_authority.get("sha256") != task["source_video_sha256"]
        or not _is_json_int(source_authority.get("size"), EXPECTED_SOURCE_SIZE)
        or not _is_json_int(source_authority.get("mode"))
        or not 0 <= source_authority["mode"] <= 0o7777
        or any(
            not _is_json_int(source_authority.get(field))
            for field in source_fields - {"path", "sha256", "size", "mode"}
        )
        or not _is_json_int(source_authority.get("nlink"), 1)
        or input_value.get("source_video_physical_authority_digest")
        != object_sha256(source_authority)
    ):
        raise ObjectTrajectoryEvalError("inference receipt input differs")
    if (
        set(preprocessing) != {
            "frame_count", "fps", "reported_fps", "source_input_hw",
            "source_derived_bucket_hw", "max_pixels", "stride", "temporal_policy",
            "spatial_policy", "resize", "external_shared_i0",
        }
        or not _is_json_int(preprocessing.get("frame_count"), 81)
        or preprocessing.get("fps") != 25.0
        or preprocessing.get("reported_fps") != 25.0
        or preprocessing.get("source_input_hw") != [736, 704]
        or preprocessing.get("source_derived_bucket_hw") != [496, 480]
        or not _is_json_int(preprocessing.get("max_pixels"), 245_760)
        or not _is_json_int(preprocessing.get("stride"), 16)
        or preprocessing.get("temporal_policy")
        != "all_integer_frames_0_through_80_no_subsampling"
        or preprocessing.get("spatial_policy")
        != "sqrt_max_pixels_then_floor_each_dimension_to_stride"
        or preprocessing.get("resize") != "torchvision_bicubic_antialias_true"
        or preprocessing.get("external_shared_i0") is not False
        or prompt != {
            "task": "mv2v",
            "system_prompt_sha256": EXPECTED_SYSTEM_PROMPT_SHA256,
            "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenizer_fix_mistral_regex": True,
            "tokenizer_padding_side": "right",
            "max_sequence_length": 512,
            "prompt_enhancer": False,
        }
    ):
        raise ObjectTrajectoryEvalError("inference preprocessing/prompt differs")
    sampling_fields = {
        "num_frames", "num_inference_steps", "guidance_mode", "omega_vid",
        "omega_img", "omega_txt", "omega_scale", "flow_shift", "seed", "eta",
        "norm_threshold", "momentum", "single_expert", "ulysses_size",
        "rank0_decode_and_save_only", "source_onset_policy",
        "source_onset_solver_trace",
    }
    active = task["oracle_arm"] in {
        "trajectory_bone_only", "trajectory_dog_bone"
    }
    if active:
        sampling_fields.add("legacy_dispatch_source_onset_policy")
    expected_policy = (
        "case01_object_trajectory_oracle_v3" if active else "hard1_every_step"
    )
    if (
        set(sampling) != sampling_fields
        or not _is_json_int(sampling.get("num_frames"), 81)
        or not _is_json_int(
            sampling.get("num_inference_steps"), NUM_INFERENCE_STEPS
        )
        or sampling.get("guidance_mode") != "v2v_apg"
        or sampling.get("omega_vid") != 1.25
        or sampling.get("omega_img") != 0.0
        or sampling.get("omega_txt") != 4.0
        or sampling.get("omega_scale") != 0.8
        or sampling.get("flow_shift") != 5.0
        or not _is_json_int(sampling.get("seed"), SEED)
        or sampling.get("eta") != 0.5
        or sampling.get("norm_threshold") != [50.0, 50.0]
        or sampling.get("momentum") != 0.0
        or sampling.get("single_expert") != "transformer_1"
        or not _is_json_int(sampling.get("ulysses_size"), 4)
        or sampling.get("rank0_decode_and_save_only") is not True
        or sampling.get("source_onset_policy") != expected_policy
        or (
            active
            and sampling.get("legacy_dispatch_source_onset_policy")
            != "hard1_every_step"
        )
    ):
        raise ObjectTrajectoryEvalError("inference sampling coordinates differ")
    if not active:
        _validate_legacy_source_trace(sampling["source_onset_solver_trace"])
    output_fields = {
        "path", "sha256", "frame_count", "fps", "height", "width",
        "audio_preserved", "size", "publication_identity",
        "prepublication_identity", "anonymous_creation_method",
        "anonymous_seal_mask", "sealed_source_sha256", "sealed_source_size",
        "anonymous_inode_encoded_and_decoded_before_publication",
        "create_only_copy_publication_after_decode",
        "sealed_source_and_publication_bytes_equal",
        "retained_inode_encoded_and_replayed", "named_output_never_replaced",
    }
    identity_fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev", "size",
        "blocks", "mtime_ns", "ctime_ns",
    }
    published = output.get("publication_identity")
    prepublished = output.get("prepublication_identity")
    if (
        set(output) != output_fields
        or output.get("path") != task["output"]["video_path"]
        or SHA256_RE.fullmatch(str(output.get("sha256"))) is None
        or not _is_json_int(output.get("size")) or output["size"] <= 0
        or not _is_json_int(output.get("frame_count"), 81)
        or output.get("fps") != 25.0
        or not _is_json_int(output.get("height"), 496)
        or not _is_json_int(output.get("width"), 480)
        or output.get("audio_preserved") is not False
        or not isinstance(published, Mapping) or set(published) != identity_fields
        or not isinstance(prepublished, Mapping) or set(prepublished) != identity_fields
        or any(not _is_json_int(value) for value in published.values())
        or any(not _is_json_int(value) for value in prepublished.values())
        or not stat.S_ISREG(published["mode"])
        or stat.S_IMODE(published["mode"]) != 0o444
        or not _is_json_int(published.get("nlink"), 1)
        or not _is_json_int(published.get("size"), output["size"])
        or not stat.S_ISREG(prepublished["mode"])
        or stat.S_IMODE(prepublished["mode"]) != 0o600
        or not _is_json_int(prepublished.get("nlink"), 0)
        or not _is_json_int(prepublished.get("size"), output["size"])
        or output.get("anonymous_creation_method") != "linux-sealed-memfd-v1"
        or not _is_json_int(output.get("anonymous_seal_mask"), 15)
        or output.get("sealed_source_sha256") != output["sha256"]
        or not _is_json_int(output.get("sealed_source_size"), output["size"])
        or output.get("anonymous_inode_encoded_and_decoded_before_publication")
        is not True
        or output.get("create_only_copy_publication_after_decode") is not True
        or output.get("sealed_source_and_publication_bytes_equal") is not True
        or output.get("retained_inode_encoded_and_replayed") is not True
        or output.get("named_output_never_replaced") is not True
    ):
        raise ObjectTrajectoryEvalError("inference output declaration differs")
    adapter_fields = {
        "enabled", "mode", "checkpoint_root", "adapter_model_path",
        "adapter_model_sha256", "training_receipt_path",
        "training_receipt_digest", "training_global_step", "strictly_reloaded",
        "safe_merged_for_inference", "tensor_count", "target_modules_sha256",
        "profile", "lora_rank", "lora_alpha", "target_module_count",
        "checkpoint_manifest",
    }
    expected_manifest = dict(task["adapter"]["checkpoint_manifest"])
    expected_manifest.pop("pin_complete", None)
    if (
        set(adapter) != adapter_fields
        or adapter.get("enabled") is not True
        or adapter.get("mode") != "lora_safe_merge"
        or adapter.get("strictly_reloaded") is not True
        or adapter.get("safe_merged_for_inference") is not True
        or not _is_json_int(adapter.get("training_global_step"), 644)
        or adapter.get("profile") != FULL644_PROFILE
        or not _is_json_int(adapter.get("lora_rank"), 64)
        or not _is_json_int(adapter.get("lora_alpha"), 64)
        or not _is_json_int(adapter.get("tensor_count"), 480)
        or not _is_json_int(adapter.get("target_module_count"), 240)
        or adapter.get("target_modules_sha256") != EXPECTED_TARGET_MODULES_SHA256
        or adapter.get("adapter_model_sha256")
        != EXPECTED_CHECKPOINT["adapter_model_sha256"]
        or adapter.get("training_receipt_digest")
        != EXPECTED_CHECKPOINT["receipt_digest"]
        or adapter.get("checkpoint_manifest") != expected_manifest
        or any(
            not isinstance(adapter.get(key), str) or not adapter[key]
            for key in (
                "checkpoint_root", "adapter_model_path", "training_receipt_path"
            )
        )
    ):
        raise ObjectTrajectoryEvalError("inference R64 adapter differs")
    consumption_fields = {
        "consumption_input_digest", "task_input_digest", "model_capture_digest",
        "model_view_root", "adapter_capture_digest", "adapter_view_root",
        "fd_view_files_authorized", "inherited_fd_binding_digest",
        "inherited_fd_count", "ptrace_authorization_used", "source_video_sha256",
        "source_video_physical_authority_digest", "all_ranks_use_retained_source_fd",
        "four_rank_attestation",
    }
    attestation = consumption.get("four_rank_attestation")
    if (
        set(consumption) != consumption_fields
        or receipt.get("consumption_input_digest")
        != consumption.get("consumption_input_digest")
        or receipt.get("task_input_digest") != consumption.get("task_input_digest")
        or any(
            SHA256_RE.fullmatch(str(consumption.get(key))) is None
            for key in (
                "consumption_input_digest", "task_input_digest",
                "model_capture_digest", "inherited_fd_binding_digest",
                "adapter_capture_digest",
            )
        )
        or any(
            not isinstance(consumption.get(key), str) or not consumption[key]
            for key in ("model_view_root", "adapter_view_root")
        )
        or not _is_json_int(consumption.get("fd_view_files_authorized"))
        or consumption["fd_view_files_authorized"] <= 0
        or not _is_json_int(consumption.get("inherited_fd_count"))
        or consumption["inherited_fd_count"] <= 0
        or consumption.get("ptrace_authorization_used") is not False
        or consumption.get("source_video_sha256") != task["source_video_sha256"]
        or consumption.get("source_video_physical_authority_digest")
        != input_value["source_video_physical_authority_digest"]
        or consumption.get("all_ranks_use_retained_source_fd") is not True
        or not isinstance(attestation, Mapping)
        or set(attestation) != {
            "world_size", "all_ranks_replayed_exact_fd_views",
            "rank_evidence_digest", "ordered_rank_evidence_digests",
        }
        or not _is_json_int(attestation.get("world_size"), 4)
        or attestation.get("all_ranks_replayed_exact_fd_views") is not True
        or SHA256_RE.fullmatch(str(attestation.get("rank_evidence_digest"))) is None
        or attestation.get("ordered_rank_evidence_digests")
        != [attestation["rank_evidence_digest"]] * 4
    ):
        raise ObjectTrajectoryEvalError("inference model consumption differs")
    expected_schema = (
        INFERENCE_RECEIPT_SCHEMA
        if custom_object_oracle else LEGACY_INFERENCE_RECEIPT_SCHEMA
    )
    if (
        receipt.get("schema_version") != expected_schema
        or receipt.get("infer_lora_source_sha256")
        != EXPECTED_LEGACY_INFER_LORA_SHA256
        or receipt.get("infer_lora_source_sha256") != producer["infer_lora_sha256"]
        or receipt.get("method_source_revision") != producer["method_source_revision"]
        or receipt.get("method_source_archive_sha256")
        != producer["method_source_archive_sha256"]
        or receipt.get("bernini_commit") != EXPECTED_BERNINI_COMMIT
        or receipt.get("veomni_commit") != EXPECTED_VEOMNI_COMMIT
        or receipt.get("bernini_inference_files") != EXPECTED_BERNINI_INFERENCE_FILES
        or receipt.get("checkpoint_tree_sha256") != EXPECTED_CHECKPOINT_TREE_SHA256
        or receipt.get("experimental_inference") is not True
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
        or set(runtime_versions) != {
            "torch", "torch_hip", "transformers", "diffusers", "peft"
        }
        or any(
            not isinstance(value, str) or not value
            for value in runtime_versions.values()
        )
        or runtime_versions.get("peft") != "0.19.1"
    ):
        raise ObjectTrajectoryEvalError("inference producer/runtime differs")
    return values


def _expected_embedded_authorities(
    task: Mapping[str, Any],
) -> dict[str, Any]:
    external = task.get("external_conditions")
    if not isinstance(external, Mapping) or set(external) != set(EXTERNAL_AUTHORITY_KEYS):
        raise ObjectTrajectoryEvalError("custom task external authority set differs")
    return {
        "source_video": {
            "sha256": task["source_video_sha256"],
            "size": EXPECTED_SOURCE_SIZE,
        },
        "bone_removed_auxiliary_video": {
            "sha256": external["aux_bone_removed_source"]["sha256"],
            "size": external["aux_bone_removed_source"]["size"],
        },
        "stage0_receipt": {
            "sha256": external["stage0_masks"]["sha256"],
            "size": external["stage0_masks"]["size"],
            "receipt_digest": external["stage0_masks"]["payload_digest"],
            "mask_count": 162,
        },
        "g0_sparse_annotations": {
            "sha256": external["g0_mouth_track"]["sha256"],
            "size": external["g0_mouth_track"]["size"],
        },
    }


def _validate_oracle_file_receipt(
    value: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    label: str,
    consumed: bool,
    artifact_digest: str | None = None,
) -> dict[str, Any]:
    fields = {"path", "sha256", "identity", "authority_digest"}
    if artifact_digest is not None:
        fields.add("artifact_digest")
    else:
        fields.add("consumed_via_retained_fd")
    identity = value.get("identity") if isinstance(value, Mapping) else None
    identity_fields = {
        "device", "inode", "mode", "nlink", "uid", "gid", "size",
        "mtime_ns", "ctime_ns",
    }
    digest_payload = {
        "path": value.get("path") if isinstance(value, Mapping) else None,
        "sha256": value.get("sha256") if isinstance(value, Mapping) else None,
        "identity": identity,
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("path") != authority["path"]
        or value.get("sha256") != authority["sha256"]
        or not isinstance(identity, Mapping)
        or set(identity) != identity_fields
        or any(not _is_json_int(identity.get(key)) for key in identity_fields)
        or not stat.S_ISREG(identity["mode"])
        or stat.S_IMODE(identity["mode"]) not in {0o444, 0o644}
        or not _is_json_int(identity.get("nlink"), 1)
        or not _is_json_int(identity.get("size"), authority["size"])
        or value.get("authority_digest") != object_sha256(digest_payload)
        or (
            artifact_digest is not None
            and value.get("artifact_digest") != artifact_digest
        )
        or (
            artifact_digest is None
            and value.get("consumed_via_retained_fd") is not consumed
        )
    ):
        raise ObjectTrajectoryEvalError(f"{label} retained authority differs")
    return dict(value)


def _expected_oracle_producer_hashes(
    producer: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "wrapper_source_sha256": producer["inference_wrapper_sha256"],
        "legacy_infer_lora_source_sha256": producer["infer_lora_sha256"],
        "projection_source_sha256": producer[
            "trajectory_projection_module_sha256"
        ],
        "scaffold_source_sha256": producer["trajectory_scaffold_module_sha256"],
    }


def _validate_collective_gate(
    value: Mapping[str, Any], *, stage: str,
) -> dict[str, Any]:
    success_status = {
        "stage": stage,
        "ok": True,
        "error_type": None,
        "error_text_sha256": None,
    }
    expected_digest = object_sha256([success_status] * 4)
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "stage", "world_size", "all_ranks_reported_ok",
            "ordered_status_digest",
        }
        or value.get("stage") != stage
        or not _is_json_int(value.get("world_size"), 4)
        or value.get("all_ranks_reported_ok") is not True
        or value.get("ordered_status_digest") != expected_digest
    ):
        raise ObjectTrajectoryEvalError(f"{stage} four-rank gate differs")
    return dict(value)


def _validate_tensor_authority(
    value: Mapping[str, Any], *, row_names: list[str],
) -> dict[str, Any]:
    tensor_shapes = {
        "source_packed_full": ("source_packed_full", [1, 19_530, 64]),
        "aux_packed_full": ("aux_packed_full", [1, 19_530, 64]),
        "legacy_phase0_selected_clean": (
            "legacy_phase0_selected_clean", [1, 930, 64]
        ),
        "source_bone_correspondence_values": (
            "source_bone_correspondence_values", [1, 377, 64]
        ),
        "source_effective_origin_values": (
            "source_effective_origin_values", [1, 187, 64]
        ),
        "aux_effective_origin_values": (
            "aux_effective_origin_values", [1, 187, 64]
        ),
        "constructed_bone_selected_clean": (
            "constructed_bone_selected_clean", [1, 564, 64]
        ),
        "constructed_dog_identity_clean": (
            "constructed_dog_identity_clean", [1, 1_548, 64]
        ),
    }
    tensors = value.get("tensors") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "tensors", "effective_origin_element_count",
            "source_aux_effective_origin_differing_element_count",
            "source_aux_effective_origin_differ", "local_device",
            "content_contract_digest",
        }
        or not isinstance(tensors, Mapping)
        or set(tensors) != set(tensor_shapes)
        or not _is_json_int(value.get("effective_origin_element_count"), 187 * 64)
        or not _is_json_int(
            value.get("source_aux_effective_origin_differing_element_count")
        )
        or not (
            0
            < value["source_aux_effective_origin_differing_element_count"]
            <= value["effective_origin_element_count"]
        )
        or value.get("source_aux_effective_origin_differ") is not True
        or not isinstance(value.get("local_device"), str)
        or not value["local_device"].startswith("cuda")
    ):
        raise ObjectTrajectoryEvalError("projection tensor authority differs")
    observed_dtype: str | None = None
    element_sizes = {
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.float32": 4,
        "torch.float64": 8,
    }
    for name, (label, shape) in tensor_shapes.items():
        row = tensors[name]
        dtype = row.get("dtype") if isinstance(row, Mapping) else None
        expected_bytes = (
            shape[0] * shape[1] * shape[2] * element_sizes.get(dtype, 0)
        )
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "label", "shape", "dtype", "device_type",
                "contiguous_before_snapshot", "byte_count", "sha256",
            }
            or row.get("label") != label
            or row.get("shape") != shape
            or dtype not in element_sizes
            or row.get("device_type") != "cuda"
            or row.get("contiguous_before_snapshot") is not True
            or not _is_json_int(row.get("byte_count"), expected_bytes)
            or SHA256_RE.fullmatch(str(row.get("sha256"))) is None
        ):
            raise ObjectTrajectoryEvalError(
                f"projection tensor byte authority differs: {name}"
            )
        if observed_dtype is None:
            observed_dtype = dtype
        elif dtype != observed_dtype:
            raise ObjectTrajectoryEvalError("projection tensor dtypes differ")
    content_payload = {
        key: value[key]
        for key in (
            "tensors", "effective_origin_element_count",
            "source_aux_effective_origin_differing_element_count",
            "source_aux_effective_origin_differ",
        )
    }
    if value.get("content_contract_digest") != object_sha256(content_payload):
        raise ObjectTrajectoryEvalError("projection tensor content digest differs")
    if (
        tensors["source_packed_full"]["sha256"]
        == tensors["aux_packed_full"]["sha256"]
        or tensors["source_effective_origin_values"]["sha256"]
        == tensors["aux_effective_origin_values"]["sha256"]
    ):
        raise ObjectTrajectoryEvalError(
            "source/aux projection tensor authority does not differ"
        )
    if row_names == [
        "legacy_phase0_hard1_every_step", "bone_conservation_all_sigma"
    ] and "constructed_dog_identity_clean" not in tensors:
        raise ObjectTrajectoryEvalError("bone arm lacks audited dog exclusion authority")
    return dict(value)


def _expected_row_specs(arm: str) -> list[dict[str, Any]]:
    rows = [
        {
            "name": "legacy_phase0_hard1_every_step",
            "selected_token_count": 930,
            "weight_shape": [1, 19_530, 1],
            "active_next_sigma_min": None,
            "active_next_sigma_max": None,
            "step_gates": None,
            "gate_policy": "all_steps_intersect_sigma_bounds",
        },
        {
            "name": "bone_conservation_all_sigma",
            "selected_token_count": 564,
            "weight_shape": [1, 19_530, 1],
            "active_next_sigma_min": None,
            "active_next_sigma_max": None,
            "step_gates": None,
            "gate_policy": "all_steps_intersect_sigma_bounds",
        },
    ]
    if arm == "trajectory_dog_bone":
        rows.append(
            {
                "name": "dog_core_low_mid",
                "selected_token_count": 1_548,
                "weight_shape": [1, 19_530, 1],
                "active_next_sigma_min": None,
                "active_next_sigma_max": 0.5,
                "step_gates": None,
                "gate_policy": "all_steps_intersect_sigma_bounds",
            }
        )
    return rows


def _validate_row_construction(
    value: Mapping[str, Any], *, arm: str,
) -> tuple[list[str], dict[str, Any]]:
    expected_specs = _expected_row_specs(arm)
    row_names = [row["name"] for row in expected_specs]
    fixed = {
        "bone_origin_clear_token_count": 187,
        "bone_scaffold_origin_support_token_count": 198,
        "bone_target_tube_token_count": 377,
        "bone_correspondence_count": 377,
        "bone_correspondence_sha256": (
            "3ddd38d6ab846b121eaa3629f121e14cb51e26d23afa0def4b8a1012c982ea7e"
        ),
        "dog_core_token_count": 1_548,
        "responsibility_tube_token_count": 2_760,
        "overlapping_origin_target_policy": "target_source_bone_detail_wins",
        "plan_digest": (
            "7eaef1dbd09e91afb9df109b358f0166757df5ddc2ac59fa09831bfeec955103"
        ),
    }
    expected_fields = {
        "row_names", "row_specs", *fixed,
        "dog_row_consumed", "origin_authority",
        "target_bone_detail_authority", "dog_detail_authority",
        "single_instance_conservation_constructed",
        "matched_legacy_phase0_baseline",
        "legacy_phase0_selected_token_count", "legacy_phase0_sigma_gate",
        "tensor_authority", "pre_projection_build_gate",
        "pre_projection_contract_gate", "projector_lookup_gate",
        "lazy_bootstrap_install_gate", "projector_install_gate",
        "final_validation_gate",
        "projection_contract",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("row_names") != row_names
        or value.get("row_specs") != expected_specs
        or any(value.get(key) != expected for key, expected in fixed.items())
        or value.get("dog_row_consumed") is not (arm == "trajectory_dog_bone")
        or value.get("origin_authority")
        != "aux_bone_removed_source_packed"
        or value.get("target_bone_detail_authority")
        != "same_source_bone_correspondence_scatter"
        or value.get("dog_detail_authority")
        != "same_source_packed_dog_core"
        or value.get("single_instance_conservation_constructed") is not True
        or value.get("matched_legacy_phase0_baseline") is not True
        or not _is_json_int(value.get("legacy_phase0_selected_token_count"), 930)
        or value.get("legacy_phase0_sigma_gate") != "all_steps_all_sigma"
    ):
        raise ObjectTrajectoryEvalError("projection row construction differs")
    tensor_authority = _validate_tensor_authority(
        value["tensor_authority"], row_names=row_names
    )
    _validate_collective_gate(
        value["pre_projection_build_gate"], stage="projection_row_build"
    )
    _validate_collective_gate(
        value["pre_projection_contract_gate"],
        stage="projection_contract_build",
    )
    _validate_collective_gate(
        value["projector_lookup_gate"],
        stage="projection_projector_lookup",
    )
    _validate_collective_gate(
        value["lazy_bootstrap_install_gate"],
        stage="projection_lazy_bootstrap_install",
    )
    _validate_collective_gate(
        value["projector_install_gate"],
        stage="projection_projector_install",
    )
    _validate_collective_gate(
        value["final_validation_gate"],
        stage="projection_final_validation",
    )
    contract = value.get("projection_contract")
    contract_base = {
        "arm": arm,
        "expected_steps": NUM_INFERENCE_STEPS,
        "row_names": row_names,
        "row_specs": expected_specs,
        "token_plan_digest": fixed["plan_digest"],
        "tensor_content_contract_digest": tensor_authority[
            "content_contract_digest"
        ],
    }
    expected_contract_digest = object_sha256(contract_base)
    consensus = contract.get("four_rank_consensus") if isinstance(contract, Mapping) else None
    if (
        not isinstance(contract, Mapping)
        or set(contract) != {
            *contract_base, "projection_contract_digest",
            "four_rank_consensus",
        }
        or any(contract.get(key) != expected for key, expected in contract_base.items())
        or contract.get("projection_contract_digest") != expected_contract_digest
        or not isinstance(consensus, Mapping)
        or set(consensus) != {
            "world_size", "all_ranks_exact_projection_contract_equal",
            "ordered_projection_contract_digests",
        }
        or not _is_json_int(consensus.get("world_size"), 4)
        or consensus.get("all_ranks_exact_projection_contract_equal") is not True
        or consensus.get("ordered_projection_contract_digests")
        != [expected_contract_digest] * 4
    ):
        raise ObjectTrajectoryEvalError("projection four-rank contract differs")
    return row_names, tensor_authority


def _expected_tensor_core_contract() -> dict[str, Any]:
    return {
        "schema_version": PROJECTION_TRACE_SCHEMA,
        "scope": "zero_training_oracle_tensor_core",
        "production_runner_integration": False,
        "renderer_abi_integration": False,
        "integrator": "original_unipc_scheduler_step",
        "prediction_type": "flow_prediction",
        "packed_layout": "B,N,64",
        "supported_packed_dtypes": [
            "torch.float16", "torch.bfloat16", "torch.float32", "torch.float64",
        ],
        "weight_policy": "strict_binary_0_or_1_v1",
        "fractional_weights_supported": False,
        "step_gate_coordinate": "next_sigma_after_native_step",
        "velocity_on_selected_elements": "initial_noise_minus_clean_authority",
        "post_step_on_selected_elements": (
            "(1-next_sigma)*clean_authority+next_sigma*initial_noise"
        ),
        "inactive_step_policy": (
            "exact_native_delegate_no_argument_clone_or_replacement"
        ),
        "all_zero_policy": "do_not_install_wrapper",
        "initial_noise_policy": (
            "explicit_exact_first_sample_or_lazy_clone_of_first_native_sample_no_rng"
        ),
        "row_overlap_policy": "allow_only_if_clean_values_are_exactly_equal",
        "terminal_sigma_policy": "positive_zero_required",
        "error_policy": "fail_closed_and_restore_step_wrapper",
    }


def _validate_tensor_core(
    core: Mapping[str, Any], *, arm: str, row_names: list[str],
    tensor_authority: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version", "contract", "zero_training_oracle",
        "production_runner_integration", "dimensions", "clean_dtype",
        "clean_device", "initial_noise_dtype", "initial_noise_device",
        "initial_noise_registration", "rows",
        "globally_selected_token_count", "globally_selected_element_count",
        "globally_enabled", "wrapper_installed", "wrapper_restored",
        "initial_noise_verified",
        "initial_noise_captured_from_first_native_sample", "step_count",
        "expected_steps", "steps", "finalized",
    }
    expected_specs = _expected_row_specs(arm)
    core_rows = [
        {
            "name": row["name"],
            "clean_shape": [1, 19_530, 64],
            "weight_shape": row["weight_shape"],
            "selected_token_count": row["selected_token_count"],
            "selected_element_count": row["selected_token_count"] * 64,
            "active_next_sigma_min": row["active_next_sigma_min"],
            "active_next_sigma_max": row["active_next_sigma_max"],
            "step_gates": row["step_gates"],
        }
        for row in expected_specs
    ]
    global_tokens = 1_477 if arm == "trajectory_bone_only" else 2_913
    steps = core.get("steps") if isinstance(core, Mapping) else None
    dtype = tensor_authority["tensors"]["source_packed_full"]["dtype"]
    device = tensor_authority["local_device"]
    if (
        not isinstance(core, Mapping)
        or set(core) != fields
        or core.get("schema_version") != PROJECTION_TRACE_SCHEMA
        or core.get("contract") != _expected_tensor_core_contract()
        or core.get("zero_training_oracle") is not True
        or core.get("production_runner_integration") is not False
        or core.get("dimensions") != {
            "source_reference": [1, 19_530, 64],
            "target_sampler": [1, 19_530, 64],
        }
        or core.get("clean_dtype") != dtype
        or core.get("clean_device") != device
        or core.get("initial_noise_dtype") != dtype
        or core.get("initial_noise_device") != device
        or core.get("initial_noise_registration")
        != "lazy_capture_first_native_sample"
        or core.get("rows") != core_rows
        or not _is_json_int(core.get("globally_selected_token_count"), global_tokens)
        or not _is_json_int(
            core.get("globally_selected_element_count"), global_tokens * 64
        )
        or core.get("globally_enabled") is not True
        or core.get("wrapper_installed") is not True
        or core.get("wrapper_restored") is not True
        or core.get("initial_noise_verified") is not True
        or core.get("initial_noise_captured_from_first_native_sample") is not True
        or not _is_json_int(core.get("step_count"), NUM_INFERENCE_STEPS)
        or not _is_json_int(core.get("expected_steps"), NUM_INFERENCE_STEPS)
        or not isinstance(steps, list)
        or len(steps) != NUM_INFERENCE_STEPS
        or core.get("finalized") is not True
    ):
        raise ObjectTrajectoryEvalError("projection tensor core differs")
    step_fields = {
        "step_index", "timestep", "sigma", "next_sigma", "cursor_before",
        "cursor_after", "projection_applied", "active_rows", "inactive_rows",
        "selected_token_count", "selected_element_count",
        "total_element_count", "original_scheduler_step_calls",
        "exact_native_delegate_no_argument_clone",
        "initial_noise_snapshot_created_this_step",
        "initial_sample_matches_registered_noise", "selected_velocity_exact",
        "unselected_velocity_exact", "selected_post_step_exact",
        "unselected_post_step_exact",
    }
    previous_next: float | None = None
    dog_seen = False
    for index, row in enumerate(steps):
        sigma = row.get("sigma") if isinstance(row, Mapping) else None
        next_sigma = row.get("next_sigma") if isinstance(row, Mapping) else None
        dog_active = arm == "trajectory_dog_bone" and (
            _is_finite_number(next_sigma) and float(next_sigma) <= 0.5
        )
        active_rows = row_names if dog_active or arm == "trajectory_bone_only" else row_names[:2]
        inactive_rows = [] if dog_active or arm == "trajectory_bone_only" else row_names[2:]
        selected_tokens = (
            2_913 if dog_active else 1_477
        )
        if dog_seen and not dog_active:
            raise ObjectTrajectoryEvalError("dog sigma gate is not a suffix")
        dog_seen = dog_seen or dog_active
        if (
            not isinstance(row, Mapping)
            or set(row) != step_fields
            or not _is_json_int(row.get("step_index"), index)
            or not _is_finite_number(row.get("timestep"))
            or not _is_finite_number(sigma)
            or not _is_finite_number(next_sigma)
            or not 0.0 <= float(next_sigma) < float(sigma) <= 1.0
            or (previous_next is not None and float(sigma) != previous_next)
            or row.get("cursor_before") != (None if index == 0 else index)
            or not _is_json_int(row.get("cursor_after"), index + 1)
            or row.get("projection_applied") is not True
            or row.get("active_rows") != active_rows
            or row.get("inactive_rows") != inactive_rows
            or not _is_json_int(row.get("selected_token_count"), selected_tokens)
            or not _is_json_int(
                row.get("selected_element_count"), selected_tokens * 64
            )
            or not _is_json_int(
                row.get("total_element_count"), 19_530 * 64
            )
            or not _is_json_int(row.get("original_scheduler_step_calls"), 1)
            or row.get("exact_native_delegate_no_argument_clone") is not False
            or row.get("initial_noise_snapshot_created_this_step") is not (index == 0)
            or row.get("initial_sample_matches_registered_noise") is not (index == 0)
            or row.get("selected_velocity_exact") is not True
            or row.get("unselected_velocity_exact") is not True
            or row.get("selected_post_step_exact") is not True
            or row.get("unselected_post_step_exact") is not True
        ):
            raise ObjectTrajectoryEvalError("projection tensor core step differs")
        previous_next = float(next_sigma)
    if previous_next != 0.0 or (
        arm == "trajectory_dog_bone" and dog_seen is not True
    ):
        raise ObjectTrajectoryEvalError("projection terminal/sigma gate differs")
    return dict(core)


def _validate_active_projection_trace(
    trace: Mapping[str, Any], task: Mapping[str, Any],
    producer: Mapping[str, Any], assets: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version", "arm", "manual_oracle", "zero_training",
        "renderer_abi_integration", "legacy_clamp_replaced",
        "projection_installation", "aux_latent_broadcast_from_rank0",
        "aux_latent_broadcast_calls", "vae_encode", "aux_collective_gates",
        "projection_collective_gates",
        "row_construction", "typed_action_program_scope",
        "approach_contact_dynamics_directly_enforced",
        "new_action_signal_for_unprojected_dynamics", "authority",
        "tensor_core", "target_video_accessed", "learned_method_claim",
        "trace_digest",
    }
    if not isinstance(trace, Mapping) or set(trace) != fields:
        raise ObjectTrajectoryEvalError("active projection trace schema differs")
    _strict_digest(trace, "trace_digest", label="active projection trace")
    arm = task["oracle_arm"]
    row_names, tensor_authority = _validate_row_construction(
        trace["row_construction"], arm=arm
    )
    _validate_tensor_core(
        trace["tensor_core"], arm=arm, row_names=row_names,
        tensor_authority=tensor_authority,
    )
    vae = trace.get("vae_encode")
    gates = trace.get("aux_collective_gates")
    projection_gates = trace.get("projection_collective_gates")
    authority = trace.get("authority")
    expected_hashes = _expected_oracle_producer_hashes(producer)
    if (
        trace.get("schema_version") != OBJECT_ORACLE_RUNTIME_SCHEMA
        or trace.get("arm") != arm
        or trace.get("manual_oracle") is not True
        or trace.get("zero_training") is not True
        or trace.get("renderer_abi_integration") is not True
        or trace.get("legacy_clamp_replaced") is not True
        or trace.get("projection_installation")
        != "lazy_at_first_native_step_after_runtime_schedule"
        or trace.get("aux_latent_broadcast_from_rank0") is not True
        or not _is_json_int(trace.get("aux_latent_broadcast_calls"), 1)
        or vae != {
            "rank0_source_original_calls": 1,
            "rank0_aux_attempts": 1,
            "rank0_aux_original_calls": 1,
        }
        or not isinstance(gates, list)
        or len(gates) != len(AUX_COLLECTIVE_STAGES)
        or not isinstance(projection_gates, list)
        or len(projection_gates) != len(PROJECTION_COLLECTIVE_STAGES)
        or (
            len(gates) + len(projection_gates) + 1
            != EXPECTED_SUCCESSFUL_ALL_GATHER_OBJECT_CALLS
        )
        or trace.get("typed_action_program_scope")
        != "patient_support_trajectory_and_dog_identity_exclusion_only"
        or trace.get("approach_contact_dynamics_directly_enforced") is not False
        or trace.get("new_action_signal_for_unprojected_dynamics")
        != "legacy_edit_instruction_prompt"
        or not isinstance(authority, Mapping)
        or set(authority) != {
            "scaffold", "aux_bone_removed_source",
            "embedded_authorities_digest", "direct_runtime_authorities",
            "derived_scaffold_authorities",
            "raw_stage0_or_g0_runtime_accessed", "producer_hashes",
        }
        or authority.get("scaffold") != assets["scaffold"]
        or authority.get("aux_bone_removed_source")
        != assets["aux_bone_removed_source"]
        or authority.get("embedded_authorities_digest")
        != assets["embedded_authorities_digest"]
        or authority.get("direct_runtime_authorities")
        != ["object_trajectory_scaffold", "aux_bone_removed_source"]
        or authority.get("derived_scaffold_authorities")
        != ["stage0_object_masks", "g0_mouth_track"]
        or authority.get("raw_stage0_or_g0_runtime_accessed") is not False
        or authority.get("producer_hashes") != expected_hashes
        or trace.get("target_video_accessed") is not False
        or trace.get("learned_method_claim") is not False
    ):
        raise ObjectTrajectoryEvalError("active projection trace differs")
    for gate, stage in zip(gates, AUX_COLLECTIVE_STAGES):
        _validate_collective_gate(gate, stage=stage)
    for gate, stage in zip(projection_gates, PROJECTION_COLLECTIVE_STAGES):
        _validate_collective_gate(gate, stage=stage)
    row_construction = trace["row_construction"]
    row_gate_links = (
        ("pre_projection_build_gate", 1),
        ("pre_projection_contract_gate", 2),
        ("projector_lookup_gate", 3),
        ("lazy_bootstrap_install_gate", 4),
        ("projector_install_gate", 5),
        ("final_validation_gate", 6),
    )
    if any(
        row_construction[key] != projection_gates[index]
        for key, index in row_gate_links
    ):
        raise ObjectTrajectoryEvalError(
            "projection row/collective gate cross-link differs"
        )
    return dict(trace)


def _validate_object_oracle(
    oracle: Mapping[str, Any], task: Mapping[str, Any],
    producer: Mapping[str, Any], sampling: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version", "arm", "status", "manual_oracle", "zero_training",
        "production_method_claim", "assets", "producer_hashes", "runtime",
    }
    if not isinstance(oracle, Mapping) or set(oracle) != fields:
        raise ObjectTrajectoryEvalError("object oracle receipt schema differs")
    arm = task["oracle_arm"]
    if arm not in {"route_off", "trajectory_bone_only", "trajectory_dog_bone"}:
        raise ObjectTrajectoryEvalError("custom receipt arm differs")
    active = arm in {"trajectory_bone_only", "trajectory_dog_bone"}
    expected_hashes = _expected_oracle_producer_hashes(producer)
    assets = oracle.get("assets")
    external = task["external_conditions"]
    embedded = _expected_embedded_authorities(task)
    if not isinstance(assets, Mapping) or set(assets) != {
        "scaffold", "aux_bone_removed_source", "embedded_authorities",
        "embedded_authorities_digest", "direct_runtime_authorities",
        "derived_scaffold_authorities",
    }:
        raise ObjectTrajectoryEvalError("object oracle asset schema differs")
    _validate_oracle_file_receipt(
        assets["scaffold"], external["trajectory_scaffold"],
        label="scaffold", consumed=False,
        artifact_digest=EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST,
    )
    _validate_oracle_file_receipt(
        assets["aux_bone_removed_source"],
        external["aux_bone_removed_source"], label="aux bone-removed source",
        consumed=active,
    )
    if (
        assets.get("embedded_authorities") != embedded
        or assets.get("embedded_authorities_digest") != object_sha256(embedded)
        or assets.get("direct_runtime_authorities") != {
            "object_trajectory_scaffold": {
                "sha256": EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_SHA256,
                "artifact_digest": EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST,
            },
            "aux_bone_removed_source": {
                "sha256": EXPECTED_AUX_REMOVED_SHA256,
                "consumed_by_renderer": active,
            },
        }
        or assets.get("derived_scaffold_authorities") != {
            "stage0_object_masks": embedded["stage0_receipt"],
            "g0_mouth_track": embedded["g0_sparse_annotations"],
            "raw_files_opened_at_runtime": False,
        }
        or oracle.get("schema_version") != OBJECT_ORACLE_RUNTIME_SCHEMA
        or oracle.get("arm") != arm
        or oracle.get("status")
        != ("consumed_projection" if active else "validated_not_consumed")
        or oracle.get("manual_oracle") is not True
        or oracle.get("zero_training") is not True
        or oracle.get("production_method_claim") is not False
        or oracle.get("producer_hashes") != expected_hashes
    ):
        raise ObjectTrajectoryEvalError("object oracle asset/producer closure differs")
    runtime = oracle.get("runtime")
    runtime_fields = {
        "object_oracle_renderer_or_scheduler_patched",
        "receipt_builder_augmented", "aux_bytes_consumed_by_renderer",
        "legacy_dispatch_source_onset_policy",
        "legacy_source_onset_solver_trace_present", "vae_encode",
        "aux_collective_gates", "projection_collective_gates",
        "aux_latent_broadcast_calls",
        "projection_trace", "direct_runtime_conditions_consumed",
        "oracle_runtime_conditions_consumed",
        "derived_scaffold_authorities_consumed_directly",
    }
    expected_vae = {
        "rank0_source_original_calls": int(active),
        "rank0_aux_attempts": int(active),
        "rank0_aux_original_calls": int(active),
    }
    gates = runtime.get("aux_collective_gates") if isinstance(runtime, Mapping) else None
    projection_gates = (
        runtime.get("projection_collective_gates")
        if isinstance(runtime, Mapping) else None
    )
    projection_trace = (
        sampling.get("source_onset_solver_trace") if active else None
    )
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != runtime_fields
        or runtime.get("object_oracle_renderer_or_scheduler_patched") is not active
        or runtime.get("receipt_builder_augmented") is not True
        or runtime.get("aux_bytes_consumed_by_renderer") is not active
        or runtime.get("legacy_dispatch_source_onset_policy") != "hard1_every_step"
        or runtime.get("legacy_source_onset_solver_trace_present") is not True
        or runtime.get("vae_encode") != expected_vae
        or not isinstance(gates, list)
        or len(gates) != (len(AUX_COLLECTIVE_STAGES) if active else 0)
        or not isinstance(projection_gates, list)
        or len(projection_gates)
        != (len(PROJECTION_COLLECTIVE_STAGES) if active else 0)
        or not _is_json_int(runtime.get("aux_latent_broadcast_calls"), int(active))
        or runtime.get("projection_trace") != projection_trace
        or runtime.get("direct_runtime_conditions_consumed")
        != task["routing"]["renderer_conditions_consumed"]
        or runtime.get("oracle_runtime_conditions_consumed")
        != task["routing"]["oracle_runtime_conditions_consumed"]
        or runtime.get("derived_scaffold_authorities_consumed_directly") != []
    ):
        raise ObjectTrajectoryEvalError("object oracle runtime consumption differs")
    if active:
        for gate, stage in zip(gates, AUX_COLLECTIVE_STAGES):
            _validate_collective_gate(gate, stage=stage)
        for gate, stage in zip(projection_gates, PROJECTION_COLLECTIVE_STAGES):
            _validate_collective_gate(gate, stage=stage)
        validated_trace = _validate_active_projection_trace(
            projection_trace, task, producer, assets
        )
        if (
            gates != validated_trace["aux_collective_gates"]
            or projection_gates
            != validated_trace["projection_collective_gates"]
        ):
            raise ObjectTrajectoryEvalError(
                "object oracle runtime/trace collective gates differ"
            )
    return dict(oracle)


def validate_off_inference_receipt(
    receipt: Mapping[str, Any], task: Mapping[str, Any],
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    if task.get("oracle_arm") not in {"null_before", "null_after"}:
        raise ObjectTrajectoryEvalError("off receipt task arm differs")
    _validate_common_v5_receipt_root(
        receipt, task, producer, custom_object_oracle=False
    )
    return dict(receipt)


def validate_custom_inference_receipt(
    receipt: Mapping[str, Any], task: Mapping[str, Any], producer: Mapping[str, Any]
) -> dict[str, Any]:
    if task.get("oracle_arm") not in {
        "route_off", "trajectory_bone_only", "trajectory_dog_bone"
    }:
        raise ObjectTrajectoryEvalError("custom inference task arm differs")
    values = _validate_common_v5_receipt_root(
        receipt, task, producer, custom_object_oracle=True
    )
    _validate_object_oracle(
        receipt["object_oracle"], task, producer, values["sampling"]
    )
    return dict(receipt)


def _load_retained_receipt(
    task: Mapping[str, Any], frozen_v2: Any,
    publication_authority: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    authority = frozen_v2.validate_retained_publication_authority(
        publication_authority, task
    )
    raw = frozen_v2._pread_exact(authority["receipt_fd"], authority["receipt_size"])
    try:
        receipt = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise ObjectTrajectoryEvalError("retained custom receipt is not strict JSON") from error
    if not isinstance(receipt, dict) or raw != canonical_json_bytes(receipt) + b"\n":
        raise ObjectTrajectoryEvalError("retained custom receipt is not canonical JSON plus LF")
    if hashlib.sha256(raw).hexdigest() != authority["receipt_sha256"]:
        raise ObjectTrajectoryEvalError("retained custom receipt SHA differs")
    return receipt, authority["receipt_sha256"], authority


def _verified_result_coordinates(task: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the frozen publication ABI while exposing the oracle variant."""

    if (
        task.get("task_id") not in TASK_IDS
        or task.get("arm") != "full644"
        or task.get("oracle_arm") not in ARM_ORDER
        or not isinstance(task.get("output"), Mapping)
    ):
        raise ObjectTrajectoryEvalError("verified result task coordinates differ")
    return {
        "task_id": task["task_id"],
        # Consumed by frozen replay_task_authority_artifacts.
        "arm": task["arm"],
        # Kept separately for the five-arm engineering-oracle report.
        "oracle_arm": task["oracle_arm"],
        "receipt_path": task["output"]["receipt_path"],
        "output_path": task["output"]["video_path"],
    }


def _verify_one(
    task: Mapping[str, Any], producer: Mapping[str, Any], *, frozen_v2: Any,
    publication_root: Path, publication_root_fd: int,
    ffprobe_authority: Mapping[str, Any], publication_authority: Mapping[str, Any],
) -> dict[str, Any]:
    receipt, receipt_sha, authority = _load_retained_receipt(
        task, frozen_v2, publication_authority
    )
    if task["oracle_arm"] in {"null_before", "null_after"}:
        validate_off_inference_receipt(receipt, task, producer)
    else:
        validate_custom_inference_receipt(receipt, task, producer)
    frozen_v2.validate_real_source_authority(task, receipt)
    logical_output = Path(task["output"]["video_path"])
    context = frozen_v2._v1_output_fd_compatibility(
        logical_output, publication_root, publication_root_fd, producer,
        ffprobe_authority, authority, task,
    )
    with context:
        _, output_sha, output_size = frozen_v2.v1._stable_file(
            logical_output, expected_sha256=receipt["output"]["sha256"],
            return_bytes=False,
        )
        publication_identity = frozen_v2.v1._publication_identity(logical_output)
        media_probe = frozen_v2.v1._probe_mp4(logical_output, producer)
    if (
        output_sha != authority["output_sha256"]
        or output_size != authority["output_size"]
        or receipt["output"]["size"] != output_size
        or receipt["output"]["publication_identity"] != publication_identity
        or authority["output_identity"] != publication_identity
        or media_probe.get("frame_count") != 81
        or media_probe.get("fps_num") != 25
        or media_probe.get("fps_den") != 1
        or media_probe.get("width") != receipt["output"]["width"]
        or media_probe.get("height") != receipt["output"]["height"]
    ):
        raise ObjectTrajectoryEvalError("retained custom output/media differs")
    return {
        **_verified_result_coordinates(task),
        "receipt_file_sha256": receipt_sha,
        "receipt_digest": receipt["receipt_digest"],
        "output_sha256": output_sha,
        "output_size": output_size,
        "media_probe": media_probe,
        "receipt": receipt,
    }


def validate_null_envelope_receipts(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate matched coordinates without demanding byte-identical outputs."""

    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ObjectTrajectoryEvalError("null envelope receipt root differs")
    for key in ("preprocessing", "prompt_contract"):
        if before.get(key) != after.get(key):
            raise ObjectTrajectoryEvalError(f"null envelope coordinate differs: {key}")
    before_sampling = before.get("sampling")
    after_sampling = after.get("sampling")
    if not isinstance(before_sampling, Mapping) or not isinstance(after_sampling, Mapping):
        raise ObjectTrajectoryEvalError("null envelope sampling differs")
    sampling_before = dict(before_sampling)
    sampling_after = dict(after_sampling)
    sampling_before.pop("custom_trace", None)
    sampling_after.pop("custom_trace", None)
    if sampling_before != sampling_after:
        raise ObjectTrajectoryEvalError("null envelope sampler coordinates differ")
    before_output = before.get("output")
    after_output = after.get("output")
    if not isinstance(before_output, Mapping) or not isinstance(after_output, Mapping):
        raise ObjectTrajectoryEvalError("null envelope output declaration differs")
    return {
        "same_source_prompt_seed_steps_sampler_coordinates": True,
        "output_byte_equality_required": False,
        "observed_output_sha256_equal": (
            before_output.get("sha256") == after_output.get("sha256")
        ),
        "historical_exact_sha_gate_applied": False,
    }


def verify_results(
    plan: Mapping[str, Any], *, frozen_v2: Any, publication_root_fd: int,
    ffprobe_authority: Mapping[str, Any],
    publication_authorities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validated = validate_plan(
        plan, reopen_sources=True, require_fresh_outputs=False,
        require_launchable=True,
    )
    if set(publication_authorities) != set(TASK_IDS):
        raise ObjectTrajectoryEvalError("retained publication task set differs")
    checkpoint = validated["checkpoint_manifest"]
    if frozen_v2.validate_terminal_checkpoint_manifest(
        checkpoint["path"], checkpoint["sha256"]
    ) != {key: value for key, value in checkpoint.items() if key != "pin_complete"}:
        raise ObjectTrajectoryEvalError("terminal R64 checkpoint changed")
    publication_root = Path(validated["tasks"][0]["output"]["video_path"]).parent
    frozen_v2._validate_publication_root_fd(publication_root, publication_root_fd)
    frozen_v2.validate_retained_ffprobe_authority(ffprobe_authority, validated["producer"])
    rows = [
        _verify_one(
            task, validated["producer"], frozen_v2=frozen_v2,
            publication_root=publication_root, publication_root_fd=publication_root_fd,
            ffprobe_authority=ffprobe_authority,
            publication_authority=publication_authorities[task["task_id"]],
        )
        for task in validated["tasks"]
    ]
    if tuple(row["task_id"] for row in rows) != TASK_IDS:
        raise ObjectTrajectoryEvalError("verified result order differs")
    before = rows[0]["receipt"]
    after = rows[4]["receipt"]
    null_envelope = validate_null_envelope_receipts(before, after)
    model_captures = {
        row["receipt"]["model_consumption"]["model_capture_digest"] for row in rows
    }
    if len(model_captures) != 1:
        raise ObjectTrajectoryEvalError("five arms do not share one model capture")
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        clean = dict(row)
        clean.pop("receipt")
        clean_rows.append(clean)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "ENGINEERING_ORACLE_COMPLETE_AWAITING_MANUAL_REVIEW",
        "campaign_mode": CAMPAIGN,
        "plan_schema_version": validated["schema_version"],
        "plan_digest": validated["plan_digest"],
        "task_count": 5,
        "task_ids": list(TASK_IDS),
        "variant_order": list(ARM_ORDER),
        "all_exact5_tasks_verified_no_cherry_pick": True,
        "same_model_capture_all_tasks": True,
        "null_envelope": null_envelope,
        "retained_publication_root_fd_replayed": True,
        "retained_ffprobe_executable_fd_replayed": True,
        "retained_publication_leaf_fds_replayed": True,
        "manual_blind_review_required": True,
        "formal_full16_report": False,
        "results": clean_rows,
        "claim_limits": dict(validated["claim_limits"]),
    }
    report["report_digest"] = object_sha256(report)
    return report

#!/usr/bin/env python3
"""Fail-closed receipt orchestration for the BRAID Stage-0 all8 canary.

This file is intentionally a control-plane component.  It never imports
PyTorch or Bernini and cannot execute a model.  A separate WORLD4 runner must
instantiate ``BraidDualNativeAPGRuntimePatch`` in a fresh process for every
arm and emit one sealed WORLD4 receipt.  This module preregisters those arms,
validates their closed receipts, and aggregates dog+human evidence into one
create-only all8 manifest.

The manifest is only forward-path engineering evidence.  In particular it
does not authorize Stage A: decode, backward/recompute, optimizer creation,
parameter updates, checkpoint writes, and semantic action-editing claims are
all forbidden by the plan and rechecked in every receipt.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence


METHOD = "bernini-braid-stage0-forward-only-all8-v1"
PLAN_SCHEMA = "bernini-braid-stage0-forward-only-plan-v1"
WORLD4_SCHEMA = "bernini-braid-stage0-forward-only-world4-v1"
ALL8_SCHEMA = "bernini-braid-stage0-forward-only-all8-v1"
REFERENCE4F_A_ONLY_ALL8_SCHEMA = (
    "bernini-braid-stage0-reference-4f-a-only-all8-audit-v1"
)
DUAL_RUNTIME_SCHEMA = "bernini-braid-dual-native-apg-stage0-runtime-v1"

WORLD_SIZE = 4
SP_SIZE = 4
ALL8_GPU_COUNT = 8
FRAME_COUNT = 81
FPS = 25
SCHEDULER_STEPS = 40
BLOCK_INDEX = 15
VISUAL_PACK_MODE = "V_source_video_only_runtime_infrastructure_canary"
NATIVE_UNIPC40_TIMESTEPS = (
    999, 994, 989, 984, 978, 972, 965, 959, 952, 945,
    937, 929, 921, 912, 902, 893, 882, 871, 859, 847,
    833, 819, 803, 787, 769, 750, 729, 707, 682, 655,
    625, 593, 556, 516, 470, 418, 359, 291, 211, 117,
)
NATIVE_UNIPC40_SIGMAS = (
    0.9999989867210388, 0.9949031472206116, 0.9895941615104675,
    0.9840595126152039, 0.978284478187561, 0.9722530841827393,
    0.9659478068351746, 0.9593496322631836, 0.9524376392364502,
    0.9451888799667358, 0.9375780820846558, 0.9295775294303894,
    0.9211564660072327, 0.912280797958374, 0.9029127359390259,
    0.893010139465332, 0.8825258612632751, 0.871407151222229,
    0.8595945835113525, 0.8470211625099182, 0.8336109519004822,
    0.8192774057388306, 0.8039219379425049, 0.7874310612678528,
    0.7696741223335266, 0.7504994869232178, 0.7297303080558777,
    0.7071589827537537, 0.6825404167175293, 0.6555827856063843,
    0.6259360909461975, 0.5931769013404846, 0.55678790807724,
    0.5161304473876953, 0.4704066216945648, 0.41860657930374146,
    0.3594328761100769, 0.2911904454231262, 0.21162153780460358,
    0.11765105277299881,
)

# Only this arm has a production WORLD4 adapter in v1.  The remaining arms
# stay preregistered so their eventual cross-arm coordinates cannot drift, but
# a receipt for them is rejected until its mechanism-specific evidence is
# implemented.  In particular, old-motion/action-capacity may never be
# promoted from caller-reported booleans.
IMPLEMENTED_WORLD4_ARM_IDS = frozenset(
    {"parity-reset-off-reference-4f-a"}
)

PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
PINNED_BERNINI_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
PINNED_VEOMNI_REVISION = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
PINNED_NATIVE_SCHEDULE_DIGEST = (
    "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2"
)
PINNED_QUERY_REGISTRY_SHA256 = (
    "01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96"
)
PINNED_BRAID_ARM_REGISTRY_SHA256 = (
    "8a771b459f3a70a1ee618b154527e67cecfaf88aae7575db8ce47a4b7ebd84c3"
)
PINNED_EDITOR_PUBLIC_KEY_SHA256 = (
    "b1357fcf5d3b30e51d686a2f1170bc139a7d8c5ea3ef99dc7cc9b2b008d3052d"
)
EXECUTION_SIGNATURE_SCHEME = "ed25519-canonical-json-v1"
EXECUTION_PUBLIC_KEY_FILENAME = "stage0-execution-ed25519-public.pem"
DEVICE_ENVIRONMENT_SCHEMA = "bernini-braid-stage0-live-device-environment-v1"

CELL_SPECS = (
    {
        "cell_id": "dog",
        "query_seed": 2026081502,
        "source_iid": "7b88a1ca1f804f41",
        "visible_devices": [0, 1, 2, 3],
    },
    {
        "cell_id": "human",
        "query_seed": 2026081505,
        "source_iid": "a35b590961d24694",
        "visible_devices": [4, 5, 6, 7],
    },
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class BraidStage0OrchestrationError(RuntimeError):
    """A preregistration, receipt, artifact, or authority boundary differed."""


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    canary: str
    forward_mode: str
    reset_source_costate: bool
    allow_shared_negative_diagnostic: bool
    noop_prompt_role: str
    action_prompt_role: str
    source_bias_mode: str
    repeatability_slot: str
    expected_steps: int = SCHEDULER_STEPS
    expected_num_frames: int = FRAME_COUNT
    block_index: int = BLOCK_INDEX


ARM_SPECS = (
    ArmSpec(
        arm_id="parity-reset-off-reference-4f-a",
        canary="two_branch_native_apg_parity",
        forward_mode="reference_4f",
        reset_source_costate=False,
        allow_shared_negative_diagnostic=False,
        noop_prompt_role="c0",
        action_prompt_role="c0",
        source_bias_mode="none",
        repeatability_slot="a",
    ),
    ArmSpec(
        arm_id="parity-reset-off-reference-4f-b",
        canary="two_branch_native_apg_parity",
        forward_mode="reference_4f",
        reset_source_costate=False,
        allow_shared_negative_diagnostic=False,
        noop_prompt_role="c0",
        action_prompt_role="c0",
        source_bias_mode="none",
        repeatability_slot="b",
    ),
    ArmSpec(
        arm_id="parity-reset-off-shared-negative-3f",
        canary="two_branch_native_apg_parity",
        forward_mode="shared_negative_3f_diagnostic",
        reset_source_costate=False,
        allow_shared_negative_diagnostic=True,
        noop_prompt_role="c0",
        action_prompt_role="c0",
        source_bias_mode="none",
        repeatability_slot="shared_negative",
    ),
    ArmSpec(
        arm_id="reset-on-reference-4f",
        canary="co_state_reset_world4_sp4_oracle",
        forward_mode="reference_4f",
        reset_source_costate=True,
        allow_shared_negative_diagnostic=False,
        noop_prompt_role="c0",
        action_prompt_role="c0",
        source_bias_mode="none",
        repeatability_slot="reset_on",
    ),
    ArmSpec(
        arm_id="capacity-source-bias-off-reference-4f",
        canary="old_motion_action_capacity_oracle",
        forward_mode="reference_4f",
        reset_source_costate=False,
        allow_shared_negative_diagnostic=False,
        noop_prompt_role="c0",
        action_prompt_role="ca",
        source_bias_mode="off",
        repeatability_slot="capacity_baseline",
    ),
    ArmSpec(
        arm_id="capacity-source-bias-on-reference-4f",
        canary="old_motion_action_capacity_oracle",
        forward_mode="reference_4f",
        reset_source_costate=False,
        allow_shared_negative_diagnostic=False,
        noop_prompt_role="c0",
        action_prompt_role="ca",
        source_bias_mode="read_only_simulated_stage_a_bias",
        repeatability_slot="capacity_source_bias",
    ),
)

ARM_BY_ID = {item.arm_id: item for item in ARM_SPECS}
CELL_BY_ID = {str(item["cell_id"]): item for item in CELL_SPECS}

PROHIBITIONS = {
    "decode_allowed": False,
    "backward_allowed": False,
    "optimizer_allowed": False,
    "parameter_update_allowed": False,
    "checkpoint_write_allowed": False,
    "semantic_action_editing_claim_allowed": False,
}

EXECUTION_AUTHORITY = {
    "forward_only": True,
    "decode_executed": False,
    "backward_executed": False,
    "optimizer_created": False,
    "parameter_update_authorized": False,
    "parameter_update_performed": False,
    "checkpoint_write_performed": False,
    "stage_a_shadow_updates_authorized": 0,
    "scientific_action_editing_success_claim": False,
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return one finite, type-closed, ASCII JSON representation."""

    def validate(item: Any, *, path: str) -> None:
        if item is None or type(item) in (bool, int, str):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise BraidStage0OrchestrationError(
                    f"{path} contains a non-finite float"
                )
            return
        if type(item) is list:
            for index, child in enumerate(item):
                validate(child, path=f"{path}[{index}]")
            return
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise BraidStage0OrchestrationError(
                        f"{path} contains a non-string key"
                    )
                validate(child, path=f"{path}.{key}")
            return
        raise BraidStage0OrchestrationError(
            f"{path} contains a non-JSON type {type(item).__name__}"
        )

    validate(value, path="receipt")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise BraidStage0OrchestrationError(
            "receipt is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def apg_state_identity_sha256(
    *, process_start_identity_sha256: str, binding: Mapping[str, Any]
) -> str:
    """Bind one process-local vendor APG object to its runtime receipt."""

    return object_sha256(
        {
            "schema_version": "bernini-braid-process-apg-object-v1",
            "process_start_identity_sha256": _sha256(
                process_start_identity_sha256, label="process start identity"
            ),
            "branch": binding.get("branch"),
            "vendor_type": binding.get("vendor_type"),
            "buffer_object_id": binding.get("buffer_object_id"),
        }
    )


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise BraidStage0OrchestrationError("hashed artifact is not an absolute plain file")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha1(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA1.fullmatch(value) is None:
        raise BraidStage0OrchestrationError(f"{label} must be full lowercase SHA-1")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise BraidStage0OrchestrationError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise BraidStage0OrchestrationError(f"{label} is not a closed safe identifier")
    return value


def seal_receipt(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if type(unsigned) is not dict or "receipt_digest" in unsigned:
        raise BraidStage0OrchestrationError("only an unsealed exact dict may be sealed")
    value = dict(unsigned)
    return {**value, "receipt_digest": object_sha256(value)}


def validate_sealed_receipt(
    value: Any, *, schema: str, required_keys: set[str], label: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != required_keys | {"receipt_digest"}:
        raise BraidStage0OrchestrationError(f"{label} has a non-closed schema")
    row = dict(value)
    digest = row.pop("receipt_digest")
    if row.get("schema_version") != schema:
        raise BraidStage0OrchestrationError(f"{label} schema version differs")
    if _sha256(digest, label=f"{label} receipt") != object_sha256(row):
        raise BraidStage0OrchestrationError(f"{label} seal differs")
    return dict(value)


def _absolute_output_root(value: Any) -> Path:
    if type(value) is not str:
        raise BraidStage0OrchestrationError("output root must be one absolute string")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or path.name in {"", ".", ".."}:
        raise BraidStage0OrchestrationError("output root must be absolute non-root")
    if str(path) != str(path.resolve(strict=False)):
        raise BraidStage0OrchestrationError("output root must be canonical")
    return path


def build_plan(
    *,
    slurm_job_id: int,
    output_root: str,
    method_source_revision: str,
    source_archive_sha256: str,
    runtime_source_sha256: str,
    runner_source_sha256: str,
    dog_editor_receipt_file_sha256: str,
    human_editor_receipt_file_sha256: str,
    execution_public_key_file_sha256: str,
) -> dict[str, Any]:
    if isinstance(slurm_job_id, bool) or not isinstance(slurm_job_id, int) or slurm_job_id <= 0:
        raise BraidStage0OrchestrationError("Slurm job ID must be a positive integer")
    root = _absolute_output_root(output_root)
    provenance = {
        "method_source_revision": _sha1(
            method_source_revision, label="method source revision"
        ),
        "source_archive_sha256": _sha256(
            source_archive_sha256, label="source archive"
        ),
        "runtime_source_sha256": _sha256(
            runtime_source_sha256, label="dual runtime source"
        ),
        "runner_source_sha256": _sha256(
            runner_source_sha256, label="WORLD4 runner source"
        ),
        "checkpoint_content_manifest_sha256": PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "bernini_revision": PINNED_BERNINI_REVISION,
        "veomni_revision": PINNED_VEOMNI_REVISION,
        "native_schedule_digest": PINNED_NATIVE_SCHEDULE_DIGEST,
        "query_registry_sha256": PINNED_QUERY_REGISTRY_SHA256,
        "braid_arm_registry_sha256": PINNED_BRAID_ARM_REGISTRY_SHA256,
        "editor_public_key_file_sha256": PINNED_EDITOR_PUBLIC_KEY_SHA256,
    }
    cells = []
    receipt_shas = {
        "dog": _sha256(
            dog_editor_receipt_file_sha256,
            label="dog signed editor receipt file",
        ),
        "human": _sha256(
            human_editor_receipt_file_sha256,
            label="human signed editor receipt file",
        ),
    }
    for cell in CELL_SPECS:
        cells.append(
            {
                **dict(cell),
                "editor_receipt_file_sha256": receipt_shas[str(cell["cell_id"])],
            }
        )
    unsigned = {
        "schema_version": PLAN_SCHEMA,
        "method": METHOD,
        "classification": "ENGINEERING_FORWARD_CANARY_ONLY",
        "slurm_job_id": slurm_job_id,
        "output_root": str(root),
        "provenance": provenance,
        "topology": {
            "node_count": 1,
            "visible_gpu_count": ALL8_GPU_COUNT,
            "independent_world4_group_count": 2,
            "world4_size": WORLD_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "dog_visible_devices": [0, 1, 2, 3],
            "human_visible_devices": [4, 5, 6, 7],
            "fresh_process_per_cell_arm": True,
            "shared_world8_process_group": False,
        },
        "cells": cells,
        "arms": [asdict(item) for item in ARM_SPECS],
        "execution_authentication": {
            "signature_scheme": EXECUTION_SIGNATURE_SCHEME,
            "public_key_filename": EXECUTION_PUBLIC_KEY_FILENAME,
            "public_key_file_sha256": _sha256(
                execution_public_key_file_sha256,
                label="ephemeral execution public key file",
            ),
            "authority_scope": (
                "this_job_world4_runner_lineage_only_no_training_or_scientific_authority"
            ),
        },
        "prohibitions": dict(PROHIBITIONS),
        "publication_contract": {
            "all_expected_receipts_required": True,
            "failed_or_missing_arm_authorizes_nothing": True,
            "all8_manifest_is_only_complete_boundary": True,
            "stage0_training_authorization_emitted": False,
            "stage_a_shadow_updates_authorized": 0,
            "semantic_or_decoded_video_assessment_in_scope": False,
        },
    }
    return seal_receipt(unsigned)


_PLAN_KEYS = {
    "schema_version",
    "method",
    "classification",
    "slurm_job_id",
    "output_root",
    "provenance",
    "topology",
    "cells",
    "arms",
    "execution_authentication",
    "prohibitions",
    "publication_contract",
}


def validate_plan(value: Any) -> dict[str, Any]:
    row = validate_sealed_receipt(
        value, schema=PLAN_SCHEMA, required_keys=_PLAN_KEYS, label="Stage-0 plan"
    )
    if (
        row["method"] != METHOD
        or row["classification"] != "ENGINEERING_FORWARD_CANARY_ONLY"
        or row["arms"] != [asdict(item) for item in ARM_SPECS]
        or row["prohibitions"] != PROHIBITIONS
        or row["topology"]
        != {
            "node_count": 1,
            "visible_gpu_count": ALL8_GPU_COUNT,
            "independent_world4_group_count": 2,
            "world4_size": WORLD_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "dog_visible_devices": [0, 1, 2, 3],
            "human_visible_devices": [4, 5, 6, 7],
            "fresh_process_per_cell_arm": True,
            "shared_world8_process_group": False,
        }
        or row["publication_contract"]
        != {
            "all_expected_receipts_required": True,
            "failed_or_missing_arm_authorizes_nothing": True,
            "all8_manifest_is_only_complete_boundary": True,
            "stage0_training_authorization_emitted": False,
            "stage_a_shadow_updates_authorized": 0,
            "semantic_or_decoded_video_assessment_in_scope": False,
        }
    ):
        raise BraidStage0OrchestrationError("Stage-0 plan contract differs")
    cells = row["cells"]
    if type(cells) is not list or len(cells) != len(CELL_SPECS):
        raise BraidStage0OrchestrationError("Stage-0 editor cell binding differs")
    for observed, fixed in zip(cells, CELL_SPECS):
        expected_keys = set(fixed) | {"editor_receipt_file_sha256"}
        if (
            type(observed) is not dict
            or set(observed) != expected_keys
            or any(observed[name] != value for name, value in fixed.items())
        ):
            raise BraidStage0OrchestrationError("Stage-0 editor cell binding differs")
        _sha256(
            observed["editor_receipt_file_sha256"],
            label=f"{fixed['cell_id']} editor receipt file",
        )
    execution = row["execution_authentication"]
    if execution != {
        "signature_scheme": EXECUTION_SIGNATURE_SCHEME,
        "public_key_filename": EXECUTION_PUBLIC_KEY_FILENAME,
        "public_key_file_sha256": execution.get("public_key_file_sha256")
        if type(execution) is dict
        else None,
        "authority_scope": (
            "this_job_world4_runner_lineage_only_no_training_or_scientific_authority"
        ),
    }:
        raise BraidStage0OrchestrationError(
            "Stage-0 execution authentication contract differs"
        )
    _sha256(
        execution["public_key_file_sha256"],
        label="ephemeral execution public key file",
    )
    _absolute_output_root(row["output_root"])
    provenance = row["provenance"]
    expected_provenance_keys = {
        "method_source_revision",
        "source_archive_sha256",
        "runtime_source_sha256",
        "runner_source_sha256",
        "checkpoint_content_manifest_sha256",
        "bernini_revision",
        "veomni_revision",
        "native_schedule_digest",
        "query_registry_sha256",
        "braid_arm_registry_sha256",
        "editor_public_key_file_sha256",
    }
    if type(provenance) is not dict or set(provenance) != expected_provenance_keys:
        raise BraidStage0OrchestrationError("Stage-0 provenance schema differs")
    _sha1(provenance["method_source_revision"], label="method source revision")
    for name in expected_provenance_keys - {"method_source_revision", "bernini_revision", "veomni_revision"}:
        _sha256(provenance[name], label=name)
    if (
        provenance["checkpoint_content_manifest_sha256"]
        != PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or provenance["bernini_revision"] != PINNED_BERNINI_REVISION
        or provenance["veomni_revision"] != PINNED_VEOMNI_REVISION
        or provenance["native_schedule_digest"] != PINNED_NATIVE_SCHEDULE_DIGEST
        or provenance["query_registry_sha256"] != PINNED_QUERY_REGISTRY_SHA256
        or provenance["braid_arm_registry_sha256"]
        != PINNED_BRAID_ARM_REGISTRY_SHA256
        or provenance["editor_public_key_file_sha256"]
        != PINNED_EDITOR_PUBLIC_KEY_SHA256
    ):
        raise BraidStage0OrchestrationError("Stage-0 pinned provenance differs")
    return dict(value)


def write_create_only_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if (
        not target.is_absolute()
        or target == Path("/")
        or target.name in {"", ".", ".."}
        or not target.parent.is_dir()
        or target.parent.is_symlink()
        or target.exists()
        or target.is_symlink()
    ):
        raise BraidStage0OrchestrationError(
            "JSON output must be create-only under a plain existing directory"
        )
    payload = canonical_json_bytes(dict(value)) + b"\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise


def load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise BraidStage0OrchestrationError(f"{label} must be an absolute plain file")
    try:
        value = json.loads(source.read_bytes().decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BraidStage0OrchestrationError(f"{label} is not canonical ASCII JSON") from error
    if canonical_json_bytes(value) != canonical_json_bytes(value):  # defensive type walk
        raise AssertionError("unreachable canonicalization mismatch")
    if type(value) is not dict:
        raise BraidStage0OrchestrationError(f"{label} must contain one JSON object")
    return value


_WORLD4_KEYS = {
    "schema_version",
    "method",
    "plan_receipt_digest",
    "cell_id",
    "query_seed",
    "source_iid",
    "arm_id",
    "arm_contract",
    "topology",
    "provenance",
    "coordinate_evidence",
    "mechanism_evidence",
    "runtime_receipts",
    "fresh_process_evidence",
    "device_environment_evidence",
    "measurements",
    "execution_authority",
    "result",
}

_WORLD4_SIGNATURE_KEYS = {
    "execution_signature_scheme",
    "execution_public_key_file_sha256",
    "execution_signature_ed25519_base64",
}


def plan_cell(plan: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    """Return the plan-bound cell, including its exact signed editor file."""

    matches = [cell for cell in plan["cells"] if cell.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise BraidStage0OrchestrationError("plan cell binding differs")
    return dict(matches[0])


def execution_public_key_path(plan: Mapping[str, Any]) -> Path:
    return (
        Path(plan["output_root"])
        / plan["execution_authentication"]["public_key_filename"]
    )


def _verify_world4_execution_signature(
    value: Any, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the job-ephemeral Ed25519 signature before trusting a receipt."""

    expected_keys = _WORLD4_KEYS | {"receipt_digest"} | _WORLD4_SIGNATURE_KEYS
    if type(value) is not dict or set(value) != expected_keys:
        raise BraidStage0OrchestrationError(
            "WORLD4 receipt lacks its closed execution signature"
        )
    row = dict(value)
    encoded = row.pop("execution_signature_ed25519_base64")
    scheme = row.pop("execution_signature_scheme")
    declared_key_sha = row.pop("execution_public_key_file_sha256")
    execution = plan["execution_authentication"]
    key_path = execution_public_key_path(plan)
    if (
        scheme != EXECUTION_SIGNATURE_SCHEME
        or scheme != execution["signature_scheme"]
        or declared_key_sha != execution["public_key_file_sha256"]
        or not key_path.is_absolute()
        or not key_path.is_file()
        or key_path.is_symlink()
        or key_path.resolve(strict=True) != key_path
        or file_sha256(key_path) != declared_key_sha
    ):
        raise BraidStage0OrchestrationError(
            "WORLD4 execution public-key binding differs"
        )
    if type(encoded) is not str:
        raise BraidStage0OrchestrationError("WORLD4 execution signature encoding differs")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as error:  # pragma: no cover
        raise BraidStage0OrchestrationError(
            "cryptography Ed25519 support is required"
        ) from error
    try:
        signature = base64.b64decode(encoded.encode("ascii"), validate=True)
        public = serialization.load_pem_public_key(key_path.read_bytes())
        if not isinstance(public, Ed25519PublicKey) or len(signature) != 64:
            raise BraidStage0OrchestrationError(
                "WORLD4 execution key/signature type differs"
            )
        public.verify(signature, canonical_json_bytes(row))
    except BraidStage0OrchestrationError:
        raise
    except (
        binascii.Error,
        InvalidSignature,
        OSError,
        ValueError,
        TypeError,
        UnicodeEncodeError,
    ) as error:
        raise BraidStage0OrchestrationError(
            "WORLD4 execution Ed25519 verification failed"
        ) from error
    return validate_sealed_receipt(
        row,
        schema=WORLD4_SCHEMA,
        required_keys=_WORLD4_KEYS,
        label="WORLD4 arm signed payload",
    )


_RUNTIME_KEYS = {
    "schema_version",
    "method",
    "pinned_bernini_commit",
    "pinned_wan_diffusion_sha256",
    "forward_mode",
    "forward_mode_authority",
    "per_step_forward_order",
    "steps",
    "transformer_forwards",
    "base_forwards",
    "action_forwards",
    "vendor_base_apg_calls",
    "vendor_action_apg_calls",
    "original_scheduler_calls",
    "scheduler_execution",
    "vendor_apg_function",
    "base_apg_binding",
    "action_apg_binding",
    "layout",
    "block15",
    "trace",
    "parameter_and_buffer_versions_unchanged",
    "optimizer_created",
    "backward_executed",
    "video_decoded",
    "checkpoint_read_or_written_by_runtime",
    "semantic_action_editing_claim",
    "training_authorized",
    "runtime_source_identity_enforcement",
    "runtime_digest",
}


def _validate_runtime_receipt(
    value: Any, *, rank: int, arm: ArmSpec
) -> dict[str, Any]:
    if type(value) is not dict:
        raise BraidStage0OrchestrationError("dual runtime receipt is not one exact dict")
    if value.get("schema_version") != DUAL_RUNTIME_SCHEMA:
        raise BraidStage0OrchestrationError("dual runtime receipt schema differs")
    if set(value) != _RUNTIME_KEYS:
        raise BraidStage0OrchestrationError("dual runtime receipt field closure differs")
    unsigned = dict(value)
    digest = unsigned.pop("runtime_digest")
    if _sha256(digest, label="dual runtime receipt") != object_sha256(unsigned):
        raise BraidStage0OrchestrationError("dual runtime receipt seal differs")
    layout = value["layout"]
    layout_keys = {
        "schema_version",
        "sp_rank",
        "sp_size",
        "total_tokens",
        "condition_tokens",
        "target_tokens",
        "local_length_ceil",
        "shard_global_start",
        "shard_global_stop_padded",
        "source_rows",
        "target_rows",
        "padding_rows",
        "global_index_formula",
        "cross_rank_hidden_gather_or_reinjection",
    }
    if (
        type(layout) is not dict
        or set(layout) != layout_keys
        or layout["schema_version"] != "bernini-braid-sp4-role-layout-v1"
        or layout["sp_rank"] != rank
        or layout["sp_size"] != SP_SIZE
        or layout["target_tokens"] != layout["total_tokens"] - layout["condition_tokens"]
        or layout["cross_rank_hidden_gather_or_reinjection"] is not False
    ):
        raise BraidStage0OrchestrationError("dual runtime SP4 layout differs")
    local_length = math.ceil(layout["total_tokens"] / SP_SIZE)
    start = rank * local_length
    global_rows = range(start, start + local_length)
    expected_source_rows = sum(
        index < layout["condition_tokens"] for index in global_rows
    )
    expected_target_rows = sum(
        layout["condition_tokens"] <= index < layout["total_tokens"]
        for index in global_rows
    )
    expected_padding_rows = local_length - expected_source_rows - expected_target_rows
    if (
        layout["local_length_ceil"] != local_length
        or layout["shard_global_start"] != start
        or layout["shard_global_stop_padded"] != start + local_length
        or layout["source_rows"] != expected_source_rows
        or layout["target_rows"] != expected_target_rows
        or layout["padding_rows"] != expected_padding_rows
    ):
        raise BraidStage0OrchestrationError("dual runtime SP4 row formula differs")
    expected_order = (
        ["base_negative", "base_positive", "action_negative", "action_positive"]
        if arm.forward_mode == "reference_4f"
        else ["base_negative", "base_positive", "action_positive"]
    )
    expected_forwards = len(expected_order)
    if (
        value["method"] != "BRAID Stage-0 dual-native APG structural canary"
        or value["pinned_bernini_commit"] != PINNED_BERNINI_REVISION
        or _SHA256.fullmatch(str(value["pinned_wan_diffusion_sha256"])) is None
        or value["forward_mode"] != arm.forward_mode
        or value["per_step_forward_order"] != expected_order
        or value["steps"] != SCHEDULER_STEPS
        or value["transformer_forwards"] != expected_forwards * SCHEDULER_STEPS
        or value["base_forwards"] != 2 * SCHEDULER_STEPS
        or value["action_forwards"] != (expected_forwards - 2) * SCHEDULER_STEPS
        or value["vendor_base_apg_calls"] != SCHEDULER_STEPS
        or value["vendor_action_apg_calls"] != SCHEDULER_STEPS
        or value["original_scheduler_calls"] != SCHEDULER_STEPS
        or value["scheduler_execution"] != "stock_base_V0_exact_object_only"
        or value["vendor_apg_function"]
        != "bernini.models.wan_diffusion.normalized_guidance"
        or value["parameter_and_buffer_versions_unchanged"] is not True
        or value["optimizer_created"] is not False
        or value["backward_executed"] is not False
        or value["video_decoded"] is not False
        or value["checkpoint_read_or_written_by_runtime"] is not False
        or value["semantic_action_editing_claim"] is not False
        or value["training_authorized"] is not False
        or value["runtime_source_identity_enforcement"]
        != "external_canary_required"
    ):
        raise BraidStage0OrchestrationError("dual runtime call/authority closure differs")
    if value["forward_mode_authority"] != (
        "four_forward_reference"
        if arm.forward_mode == "reference_4f"
        else "shared_negative_diagnostic_only"
    ):
        raise BraidStage0OrchestrationError("dual runtime forward-mode authority differs")

    binding_keys = {
        "branch",
        "vendor_type",
        "buffer_object_id",
        "momentum",
        "initial_integer_zero_authenticated",
        "normalized_guidance_calls",
    }
    base_binding = value["base_apg_binding"]
    action_binding = value["action_apg_binding"]
    if (
        type(base_binding) is not dict
        or type(action_binding) is not dict
        or set(base_binding) != binding_keys
        or set(action_binding) != binding_keys
        or base_binding["branch"] != "base"
        or action_binding["branch"] != "action"
        or base_binding["vendor_type"]
        != "bernini.models.wan_diffusion.MomentumBuffer"
        or action_binding["vendor_type"] != base_binding["vendor_type"]
        or base_binding["buffer_object_id"] == action_binding["buffer_object_id"]
        or base_binding["momentum"] != 0.0
        or action_binding["momentum"] != 0.0
        or base_binding["initial_integer_zero_authenticated"] is not True
        or action_binding["initial_integer_zero_authenticated"] is not True
        or base_binding["normalized_guidance_calls"] != SCHEDULER_STEPS
        or action_binding["normalized_guidance_calls"] != SCHEDULER_STEPS
    ):
        raise BraidStage0OrchestrationError("dual runtime APG state binding differs")

    block = value["block15"]
    block_keys = {
        "schema_version",
        "block_index",
        "selection_authority",
        "reset_enabled",
        "rank_local_only",
        "hidden_collective_or_reinjection",
        "records",
        "semantic_action_editing_claim",
        "training_authorized",
    }
    if (
        type(block) is not dict
        or set(block) != block_keys
        or block["schema_version"]
        != "bernini-braid-block15-source-costate-canary-v1"
        or block["block_index"] != BLOCK_INDEX
        or block["selection_authority"]
        != "infrastructure_canary_only_not_an_authorized_braid_reset_boundary"
        or block["reset_enabled"] is not arm.reset_source_costate
        or block["rank_local_only"] is not True
        or block["hidden_collective_or_reinjection"] is not False
        or block["semantic_action_editing_claim"] is not False
        or block["training_authorized"] is not False
        or type(block["records"]) is not list
        or len(block["records"]) != SCHEDULER_STEPS
    ):
        raise BraidStage0OrchestrationError("dual runtime block15 receipt differs")
    block_record_keys = {
        "step_index",
        "block_index",
        "forward_order",
        "reset_enabled",
        "source_rows",
        "target_rows",
        "padding_rows",
        "source_pre_reset_mismatch_bytes",
        "source_post_reset_mismatch_bytes",
        "target_post_reset_mismatch_bytes",
        "padding_post_reset_mismatch_bytes",
        "reset_returned_new_object",
        "reset_off_returned_original_object",
        "cache_created_once",
        "cache_consumed_once",
    }
    for step, record in enumerate(block["records"]):
        if (
            type(record) is not dict
            or set(record) != block_record_keys
            or record["step_index"] != step
            or record["block_index"] != BLOCK_INDEX
            or record["forward_order"] != expected_order
            or record["reset_enabled"] is not arm.reset_source_costate
            or record["source_rows"] != layout["source_rows"]
            or record["target_rows"] != layout["target_rows"]
            or record["padding_rows"] != layout["padding_rows"]
            or record["source_post_reset_mismatch_bytes"] != 0
            or record["target_post_reset_mismatch_bytes"] != 0
            or record["padding_post_reset_mismatch_bytes"] != 0
            or record["reset_returned_new_object"] is not arm.reset_source_costate
            or record["reset_off_returned_original_object"]
            is not (not arm.reset_source_costate)
            or record["cache_created_once"] is not True
            or record["cache_consumed_once"] is not True
        ):
            raise BraidStage0OrchestrationError("dual runtime block15 step differs")

    trace = value["trace"]
    trace_keys = {
        "schema_version",
        "step_index",
        "timestep",
        "sigma",
        "forward_mode",
        "forward_order",
        "transformer_forwards",
        "base_forwards",
        "action_forwards",
        "shared_negative",
        "independent_complete_native_apg_pairs",
        "vendor_base_apg_calls",
        "vendor_action_apg_calls",
        "base_action_buffers_distinct",
        "base_stock_apg_exact_parity",
        "base_stock_apg_parity_max_abs",
        "base_stock_apg_parity_rms",
        "negative_repeat_exact_parity",
        "negative_repeat_mismatch_bytes",
        "action_base_velocity_delta_rms",
        "original_scheduler_calls",
        "scheduler_received_stock_base_object",
        "block15",
    }
    if type(trace) is not list or len(trace) != SCHEDULER_STEPS:
        raise BraidStage0OrchestrationError("dual runtime trace length differs")
    for step, item in enumerate(trace):
        if (
            type(item) is not dict
            or set(item) != trace_keys
            or item["schema_version"] != DUAL_RUNTIME_SCHEMA
            or item["step_index"] != step
            or item["timestep"] != NATIVE_UNIPC40_TIMESTEPS[step]
            or item["sigma"] != NATIVE_UNIPC40_SIGMAS[step]
            or item["forward_mode"] != arm.forward_mode
            or item["forward_order"] != expected_order
            or item["transformer_forwards"] != expected_forwards
            or item["base_forwards"] != 2
            or item["action_forwards"] != expected_forwards - 2
            or item["shared_negative"]
            is not (arm.forward_mode == "shared_negative_3f_diagnostic")
            or item["independent_complete_native_apg_pairs"]
            is not (arm.forward_mode == "reference_4f")
            or item["vendor_base_apg_calls"] != 1
            or item["vendor_action_apg_calls"] != 1
            or item["base_action_buffers_distinct"] is not True
            or item["base_stock_apg_exact_parity"] is not True
            or item["base_stock_apg_parity_max_abs"] != 0.0
            or item["base_stock_apg_parity_rms"] != 0.0
            or item["negative_repeat_exact_parity"] is not True
            or item["negative_repeat_mismatch_bytes"] != 0
            or (
                arm.noop_prompt_role == arm.action_prompt_role == "c0"
                and item["action_base_velocity_delta_rms"] != 0.0
            )
            or item["original_scheduler_calls"] != 1
            or item["scheduler_received_stock_base_object"] is not True
            or item["block15"] != block["records"][step]
        ):
            raise BraidStage0OrchestrationError("dual runtime per-step trace differs")
    return dict(value)


def validate_world4_receipt(
    value: Any, *, plan: Mapping[str, Any], expected_cell: str, expected_arm: str
) -> dict[str, Any]:
    plan_row = validate_plan(dict(plan))
    row = _verify_world4_execution_signature(value, plan=plan_row)
    fixed_cell = CELL_BY_ID.get(expected_cell)
    arm = ARM_BY_ID.get(expected_arm)
    if fixed_cell is None or arm is None:
        raise BraidStage0OrchestrationError("expected cell/arm is not preregistered")
    cell = plan_cell(plan_row, expected_cell)
    if expected_arm not in IMPLEMENTED_WORLD4_ARM_IDS:
        raise BraidStage0OrchestrationError(
            "WORLD4 arm is preregistered but its evidence implementation is unavailable"
        )
    if (
        row["method"] != METHOD
        or row["plan_receipt_digest"] != plan_row["receipt_digest"]
        or row["cell_id"] != expected_cell
        or row["query_seed"] != cell["query_seed"]
        or row["source_iid"] != cell["source_iid"]
        or row["arm_id"] != expected_arm
        or row["arm_contract"] != asdict(arm)
        or row["provenance"] != plan_row["provenance"]
        or row["topology"]
        != {
            "world_size": WORLD_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "rank_order": [0, 1, 2, 3],
            "visible_devices": cell["visible_devices"],
        }
        or row["execution_authority"] != EXECUTION_AUTHORITY
    ):
        raise BraidStage0OrchestrationError("WORLD4 coordinate or authority differs")

    coordinate = row["coordinate_evidence"]
    coordinate_keys = {
        "editor_runtime_input_receipt_digest",
        "editor_runtime_input_receipt_file_sha256",
        "editor_public_key_file_sha256",
        "editor_method_source_revision",
        "editor_method_source_archive_sha256",
        "source_latent_sha256",
        "official_initial_noise_sha256",
        "endpoint_latent_sha256",
        "noop_prompt_tensor_sha256",
        "action_prompt_tensor_sha256",
        "negative_prompt_tensor_sha256",
        "exact40_timestep_sigma_digest",
        "source_and_noise_byte_identity_revalidated",
        "prompt_byte_identity_revalidated",
        "all_rank_coordinate_consensus",
    }
    if type(coordinate) is not dict or set(coordinate) != coordinate_keys:
        raise BraidStage0OrchestrationError("WORLD4 coordinate evidence schema differs")
    for name in coordinate_keys - {
        "editor_method_source_revision",
        "source_and_noise_byte_identity_revalidated",
        "prompt_byte_identity_revalidated",
        "all_rank_coordinate_consensus",
    }:
        _sha256(coordinate[name], label=name)
    _sha1(
        coordinate["editor_method_source_revision"],
        label="editor method source revision",
    )
    if (
        coordinate["exact40_timestep_sigma_digest"] != PINNED_NATIVE_SCHEDULE_DIGEST
        or coordinate["editor_runtime_input_receipt_file_sha256"]
        != cell["editor_receipt_file_sha256"]
        or coordinate["editor_public_key_file_sha256"]
        != plan_row["provenance"]["editor_public_key_file_sha256"]
        or coordinate["source_and_noise_byte_identity_revalidated"] is not True
        or coordinate["prompt_byte_identity_revalidated"] is not True
        or coordinate["all_rank_coordinate_consensus"] is not True
    ):
        raise BraidStage0OrchestrationError("WORLD4 coordinate evidence failed")
    if arm.action_prompt_role == "c0":
        if coordinate["action_prompt_tensor_sha256"] != coordinate["noop_prompt_tensor_sha256"]:
            raise BraidStage0OrchestrationError("c0/c0 arm prompt bytes differ")
    elif coordinate["action_prompt_tensor_sha256"] == coordinate["noop_prompt_tensor_sha256"]:
        raise BraidStage0OrchestrationError("capacity ca prompt equals c0 bytes")

    mechanism = row["mechanism_evidence"]
    mechanism_keys = {
        "visual_pack_mode",
        "sp4_collective_receipt_digest",
        "source_bias_mode",
        "source_bias_operator_digest",
        "source_bias_read_only",
        "source_bias_parameter_mutation",
        "comparison_evaluator_source_sha256",
        "comparison_threshold_registry_sha256",
        "all_rank_metric_packet_digest",
        "all_rank_mechanism_consensus",
    }
    if type(mechanism) is not dict or set(mechanism) != mechanism_keys:
        raise BraidStage0OrchestrationError("WORLD4 mechanism evidence schema differs")
    if (
        mechanism["visual_pack_mode"] != VISUAL_PACK_MODE
        or mechanism["source_bias_mode"] != arm.source_bias_mode
        or mechanism["source_bias_read_only"] is not True
        or mechanism["source_bias_parameter_mutation"] is not False
        or mechanism["comparison_evaluator_source_sha256"]
        != plan_row["provenance"]["runner_source_sha256"]
        or mechanism["all_rank_mechanism_consensus"] is not True
    ):
        raise BraidStage0OrchestrationError("WORLD4 mechanism binding differs")
    for name in (
        "sp4_collective_receipt_digest",
        "comparison_evaluator_source_sha256",
        "comparison_threshold_registry_sha256",
        "all_rank_metric_packet_digest",
    ):
        _sha256(mechanism[name], label=name)
    if arm.source_bias_mode == "read_only_simulated_stage_a_bias":
        _sha256(mechanism["source_bias_operator_digest"], label="source bias operator")
    elif mechanism["source_bias_operator_digest"] is not None:
        raise BraidStage0OrchestrationError("source-bias-off arm carries an operator")

    runtime_receipts = row["runtime_receipts"]
    if type(runtime_receipts) is not list or len(runtime_receipts) != WORLD_SIZE:
        raise BraidStage0OrchestrationError("WORLD4 lacks four dual runtime receipts")
    for rank, runtime_receipt in enumerate(runtime_receipts):
        _validate_runtime_receipt(runtime_receipt, rank=rank, arm=arm)

    device_rows = row["device_environment_evidence"]
    device_keys = {
        "schema_version",
        "sp_rank",
        "rank",
        "local_rank",
        "world_size",
        "rocr_visible_devices",
        "physical_visible_devices",
        "hip_visible_devices_unset",
        "cuda_visible_devices_unset",
        "gpu_device_ordinal_unset",
        "observed_before_torch_import",
        "environment_digest",
    }
    expected_rocr = ",".join(str(item) for item in cell["visible_devices"])
    if type(device_rows) is not list or len(device_rows) != WORLD_SIZE:
        raise BraidStage0OrchestrationError(
            "WORLD4 live device environment lacks four ranks"
        )
    for rank, device_row in enumerate(device_rows):
        if type(device_row) is not dict or set(device_row) != device_keys:
            raise BraidStage0OrchestrationError(
                "WORLD4 live device environment schema differs"
            )
        unsigned_device = dict(device_row)
        digest = unsigned_device.pop("environment_digest")
        if (
            _sha256(digest, label="live device environment")
            != object_sha256(unsigned_device)
            or device_row["schema_version"] != DEVICE_ENVIRONMENT_SCHEMA
            or device_row["sp_rank"] != rank
            or device_row["rank"] != rank
            or device_row["local_rank"] != rank
            or device_row["world_size"] != WORLD_SIZE
            or device_row["rocr_visible_devices"] != expected_rocr
            or device_row["physical_visible_devices"] != cell["visible_devices"]
            or device_row["hip_visible_devices_unset"] is not True
            or device_row["cuda_visible_devices_unset"] is not True
            or device_row["gpu_device_ordinal_unset"] is not True
            or device_row["observed_before_torch_import"] is not True
        ):
            raise BraidStage0OrchestrationError(
                "WORLD4 live device environment binding differs"
            )

    process_rows = row["fresh_process_evidence"]
    process_keys = {
        "sp_rank",
        "process_start_identity_sha256",
        "model_object_identity_sha256",
        "scheduler_object_identity_sha256",
        "noop_apg_state_identity_sha256",
        "action_apg_state_identity_sha256",
        "model_construct_count",
        "scheduler_construct_count",
        "sample_call_count",
    }
    if type(process_rows) is not list or len(process_rows) != WORLD_SIZE:
        raise BraidStage0OrchestrationError("fresh-process evidence lacks four ranks")
    process_ids: list[str] = []
    for rank, process in enumerate(process_rows):
        if type(process) is not dict or set(process) != process_keys:
            raise BraidStage0OrchestrationError("fresh-process evidence schema differs")
        if (
            process["sp_rank"] != rank
            or process["model_construct_count"] != 1
            or process["scheduler_construct_count"] != 1
            or process["sample_call_count"] != 1
        ):
            raise BraidStage0OrchestrationError("fresh process/model/sample count differs")
        for name in process_keys - {
            "sp_rank",
            "model_construct_count",
            "scheduler_construct_count",
            "sample_call_count",
        }:
            _sha256(process[name], label=name)
        if process["noop_apg_state_identity_sha256"] == process["action_apg_state_identity_sha256"]:
            raise BraidStage0OrchestrationError("noop/action APG states share an identity")
        runtime = runtime_receipts[rank]
        if (
            process["noop_apg_state_identity_sha256"]
            != apg_state_identity_sha256(
                process_start_identity_sha256=process[
                    "process_start_identity_sha256"
                ],
                binding=runtime["base_apg_binding"],
            )
            or process["action_apg_state_identity_sha256"]
            != apg_state_identity_sha256(
                process_start_identity_sha256=process[
                    "process_start_identity_sha256"
                ],
                binding=runtime["action_apg_binding"],
            )
        ):
            raise BraidStage0OrchestrationError(
                "fresh-process APG object binding differs"
            )
        process_ids.append(process["process_start_identity_sha256"])
    if len(set(process_ids)) != WORLD_SIZE:
        raise BraidStage0OrchestrationError("WORLD4 reused one process identity")

    measurement_keys = {
        "runtime_finalize_passed",
        "projection_local_zero_residual_exact",
        "off_off_path_structural_pass",
        "reset_on_off_path_structural_pass",
        "old_motion_axis_observed",
        "desired_action_capacity_axis_observed",
        "old_motion_action_capacity_non_regression_pass",
        "scheduler_steps_observed",
        "scheduler_advances_per_step",
        "exact81_latent_rollout_observed",
        "decoded_video_observed",
    }
    measurements = row["measurements"]
    if type(measurements) is not dict or set(measurements) != measurement_keys:
        raise BraidStage0OrchestrationError("WORLD4 measurements schema differs")
    if (
        measurements["runtime_finalize_passed"] is not True
        or measurements["projection_local_zero_residual_exact"] is not True
        or measurements["off_off_path_structural_pass"] is not True
        or measurements["scheduler_steps_observed"] != SCHEDULER_STEPS
        or measurements["scheduler_advances_per_step"] != 1
        or measurements["exact81_latent_rollout_observed"] is not True
        or measurements["decoded_video_observed"] is not False
    ):
        raise BraidStage0OrchestrationError("common forward-only measurements failed")
    if arm.canary == "co_state_reset_world4_sp4_oracle":
        if (
            measurements["reset_on_off_path_structural_pass"] is not True
            or measurements["old_motion_axis_observed"] is not None
            or measurements["desired_action_capacity_axis_observed"] is not None
            or measurements["old_motion_action_capacity_non_regression_pass"] is not None
        ):
            raise BraidStage0OrchestrationError("reset oracle measurements differ")
    elif arm.canary == "old_motion_action_capacity_oracle":
        if (
            measurements["reset_on_off_path_structural_pass"] is not None
            or measurements["old_motion_axis_observed"] is not True
            or measurements["desired_action_capacity_axis_observed"] is not True
            or measurements["old_motion_action_capacity_non_regression_pass"] is not True
        ):
            raise BraidStage0OrchestrationError("old-motion/action-capacity measurements failed")
    else:
        if any(
            measurements[name] is not None
            for name in (
                "reset_on_off_path_structural_pass",
                "old_motion_axis_observed",
                "desired_action_capacity_axis_observed",
                "old_motion_action_capacity_non_regression_pass",
            )
        ):
            raise BraidStage0OrchestrationError("parity arm contains foreign oracle claims")
    if row["result"] != {
        "status": "PASS",
        "classification": "ENGINEERING_FORWARD_PATH_ONLY",
        "semantic_authority": False,
        "decoded_quality_authority": False,
        "stage0_training_authority": False,
    }:
        raise BraidStage0OrchestrationError("WORLD4 result authority differs")
    return dict(value)


def _load_plan(path: str | Path) -> dict[str, Any]:
    return validate_plan(load_json(path, label="Stage-0 plan"))


def _load_world4(path: Path, *, plan: Mapping[str, Any], cell: str, arm: str) -> dict[str, Any]:
    if path.name != "world4.receipt.json":
        raise BraidStage0OrchestrationError("WORLD4 receipt basename differs")
    return validate_world4_receipt(
        load_json(path, label="WORLD4 arm receipt"),
        plan=plan,
        expected_cell=cell,
        expected_arm=arm,
    )


_REFERENCE4F_A_ONLY_KEYS = {
    "schema_version",
    "method",
    "decision",
    "partial_stage0",
    "full_stage0_complete",
    "plan_path",
    "plan_file_sha256",
    "plan_receipt_digest",
    "completed_arm_ids",
    "missing_arm_ids",
    "topology",
    "provenance",
    "execution_authentication",
    "visual_pack_mode",
    "world4_artifacts",
    "world4_receipt_count",
    "fresh_rank_process_count",
    "execution_authority",
    "stage0_training_authority",
    "stage_a_authorized",
    "stage_a_shadow_updates_authorized",
    "scientific_authority",
    "semantic_action_editing_success_claim",
}


def validate_reference4f_a_only_all8_receipt(
    value: Any, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    plan_row = validate_plan(dict(plan))
    row = validate_sealed_receipt(
        value,
        schema=REFERENCE4F_A_ONLY_ALL8_SCHEMA,
        required_keys=_REFERENCE4F_A_ONLY_KEYS,
        label="reference-4f-a-only all8 audit receipt",
    )
    completed = "parity-reset-off-reference-4f-a"
    missing = [arm.arm_id for arm in ARM_SPECS if arm.arm_id != completed]
    artifacts = row["world4_artifacts"]
    artifact_keys = {
        "cell_id",
        "arm_id",
        "world4_receipt_path",
        "world4_receipt_file_sha256",
        "world4_receipt_digest",
        "editor_receipt_file_sha256",
        "editor_public_key_file_sha256",
        "device_environment_evidence_digest",
        "execution_public_key_file_sha256",
        "world4_execution_signature_sha256",
    }
    if (
        row["method"] != METHOD
        or row["decision"] != "PARTIAL_STAGE0_REFERENCE_4F_A_ENGINEERING_AUDIT_ONLY"
        or row["partial_stage0"] is not True
        or row["full_stage0_complete"] is not False
        or row["plan_receipt_digest"] != plan_row["receipt_digest"]
        or row["completed_arm_ids"] != [completed]
        or row["missing_arm_ids"] != missing
        or row["topology"] != plan_row["topology"]
        or row["provenance"] != plan_row["provenance"]
        or row["execution_authentication"]
        != plan_row["execution_authentication"]
        or row["visual_pack_mode"] != VISUAL_PACK_MODE
        or type(artifacts) is not list
        or len(artifacts) != 2
        or [artifact.get("cell_id") for artifact in artifacts] != ["dog", "human"]
        or any(
            type(artifact) is not dict
            or set(artifact) != artifact_keys
            or artifact["arm_id"] != completed
            for artifact in artifacts
        )
        or row["world4_receipt_count"] != 2
        or row["fresh_rank_process_count"] != 8
        or row["execution_authority"] != EXECUTION_AUTHORITY
        or row["stage0_training_authority"] is not False
        or row["stage_a_authorized"] is not False
        or row["stage_a_shadow_updates_authorized"] != 0
        or row["scientific_authority"] is not False
        or row["semantic_action_editing_success_claim"] is not False
    ):
        raise BraidStage0OrchestrationError(
            "reference-4f-a-only partial authority closure differs"
        )
    expected_plan_path = Path(plan_row["output_root"]) / "stage0.plan.json"
    if (
        row["plan_path"] != str(expected_plan_path)
        or not expected_plan_path.is_file()
        or expected_plan_path.is_symlink()
        or row["plan_file_sha256"] != file_sha256(expected_plan_path)
    ):
        raise BraidStage0OrchestrationError("partial plan artifact binding differs")
    _sha256(row["plan_file_sha256"], label="partial plan file")
    for artifact in artifacts:
        _sha256(artifact["world4_receipt_file_sha256"], label="partial WORLD4 file")
        _sha256(artifact["world4_receipt_digest"], label="partial WORLD4 receipt")
        for field in (
            "editor_receipt_file_sha256",
            "editor_public_key_file_sha256",
            "device_environment_evidence_digest",
            "execution_public_key_file_sha256",
            "world4_execution_signature_sha256",
        ):
            _sha256(artifact[field], label=f"partial {field}")
        expected_world4 = (
            Path(plan_row["output_root"])
            / "evidence"
            / artifact["cell_id"]
            / completed
            / "world4.receipt.json"
        )
        reopened = _load_world4(
            expected_world4,
            plan=plan_row,
            cell=artifact["cell_id"],
            arm=completed,
        )
        plan_cell_row = plan_cell(plan_row, artifact["cell_id"])
        try:
            signature_bytes = base64.b64decode(
                reopened["execution_signature_ed25519_base64"].encode("ascii"),
                validate=True,
            )
        except (binascii.Error, ValueError, UnicodeEncodeError) as error:
            raise BraidStage0OrchestrationError(
                "partial WORLD4 execution signature encoding differs"
            ) from error
        if (
            artifact["world4_receipt_path"] != str(expected_world4)
            or artifact["world4_receipt_file_sha256"] != file_sha256(expected_world4)
            or reopened["receipt_digest"] != artifact["world4_receipt_digest"]
            or artifact["editor_receipt_file_sha256"]
            != plan_cell_row["editor_receipt_file_sha256"]
            or artifact["editor_public_key_file_sha256"]
            != plan_row["provenance"]["editor_public_key_file_sha256"]
            or artifact["device_environment_evidence_digest"]
            != object_sha256(reopened["device_environment_evidence"])
            or artifact["execution_public_key_file_sha256"]
            != plan_row["execution_authentication"]["public_key_file_sha256"]
            or artifact["world4_execution_signature_sha256"]
            != hashlib.sha256(signature_bytes).hexdigest()
        ):
            raise BraidStage0OrchestrationError(
                "partial WORLD4 artifact binding differs"
            )
    return row


def aggregate_reference4f_a_only_all8(
    *, plan_path: str | Path, evidence_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """Publish two real WORLD4 parity receipts without claiming full Stage-0."""

    plan_source = Path(plan_path)
    plan = _load_plan(plan_source)
    root = Path(evidence_root)
    output_path = Path(output)
    output_root = Path(plan["output_root"])
    arm_id = "parity-reset-off-reference-4f-a"
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != (output_root / "evidence").resolve(strict=True)
        or output_path
        != output_root / "reference-4f-a-only-all8.receipt.json"
    ):
        raise BraidStage0OrchestrationError(
            "reference-4f-a-only evidence/output root differs from plan"
        )
    expected_files = {
        f"{cell['cell_id']}/{arm_id}/world4.receipt.json" for cell in CELL_SPECS
    }
    expected_dirs = {str(cell["cell_id"]) for cell in CELL_SPECS} | {
        f"{cell['cell_id']}/{arm_id}" for cell in CELL_SPECS
    }
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise BraidStage0OrchestrationError("partial evidence contains a symlink")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_dirs.add(relative)
        else:
            raise BraidStage0OrchestrationError("partial evidence contains a non-file")
    if observed_files != expected_files or observed_dirs != expected_dirs:
        raise BraidStage0OrchestrationError(
            "reference-4f-a-only evidence closure differs"
        )

    artifacts: list[dict[str, Any]] = []
    process_ids: list[str] = []
    for fixed_cell in CELL_SPECS:
        cell_id = str(fixed_cell["cell_id"])
        cell = plan_cell(plan, cell_id)
        source = root / cell_id / arm_id / "world4.receipt.json"
        receipt = _load_world4(source, plan=plan, cell=cell_id, arm=arm_id)
        process_ids.extend(
            item["process_start_identity_sha256"]
            for item in receipt["fresh_process_evidence"]
        )
        artifacts.append(
            {
                "cell_id": cell_id,
                "arm_id": arm_id,
                "world4_receipt_path": str(source),
                "world4_receipt_file_sha256": file_sha256(source),
                "world4_receipt_digest": receipt["receipt_digest"],
                "editor_receipt_file_sha256": cell[
                    "editor_receipt_file_sha256"
                ],
                "editor_public_key_file_sha256": plan["provenance"][
                    "editor_public_key_file_sha256"
                ],
                "device_environment_evidence_digest": object_sha256(
                    receipt["device_environment_evidence"]
                ),
                "execution_public_key_file_sha256": plan[
                    "execution_authentication"
                ]["public_key_file_sha256"],
                "world4_execution_signature_sha256": hashlib.sha256(
                    base64.b64decode(
                        receipt["execution_signature_ed25519_base64"].encode(
                            "ascii"
                        ),
                        validate=True,
                    )
                ).hexdigest(),
            }
        )
    if len(process_ids) != 8 or len(set(process_ids)) != 8:
        raise BraidStage0OrchestrationError(
            "partial all8 reused a WORLD4 process identity"
        )
    missing = [arm.arm_id for arm in ARM_SPECS if arm.arm_id != arm_id]
    unsigned = {
        "schema_version": REFERENCE4F_A_ONLY_ALL8_SCHEMA,
        "method": METHOD,
        "decision": "PARTIAL_STAGE0_REFERENCE_4F_A_ENGINEERING_AUDIT_ONLY",
        "partial_stage0": True,
        "full_stage0_complete": False,
        "plan_path": str(plan_source),
        "plan_file_sha256": file_sha256(plan_source),
        "plan_receipt_digest": plan["receipt_digest"],
        "completed_arm_ids": [arm_id],
        "missing_arm_ids": missing,
        "topology": plan["topology"],
        "provenance": plan["provenance"],
        "execution_authentication": plan["execution_authentication"],
        "visual_pack_mode": VISUAL_PACK_MODE,
        "world4_artifacts": artifacts,
        "world4_receipt_count": 2,
        "fresh_rank_process_count": 8,
        "execution_authority": dict(EXECUTION_AUTHORITY),
        "stage0_training_authority": False,
        "stage_a_authorized": False,
        "stage_a_shadow_updates_authorized": 0,
        "scientific_authority": False,
        "semantic_action_editing_success_claim": False,
    }
    receipt = validate_reference4f_a_only_all8_receipt(
        seal_receipt(unsigned), plan=plan
    )
    write_create_only_json(output_path, receipt)
    reopened = load_json(output_path, label="published reference-4f-a-only receipt")
    return validate_reference4f_a_only_all8_receipt(reopened, plan=plan)


def aggregate_all8(
    *, plan_path: str | Path, evidence_root: str | Path, output: str | Path
) -> dict[str, Any]:
    plan_source = Path(plan_path)
    plan = _load_plan(plan_source)
    root = Path(evidence_root)
    output_path = Path(output)
    expected_root = Path(plan["output_root"]) / "evidence"
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != expected_root.resolve(strict=True)
        or output_path != Path(plan["output_root"]) / "all8.manifest.json"
    ):
        raise BraidStage0OrchestrationError("all8 evidence/output root differs from plan")

    expected_files = {
        f"{cell['cell_id']}/{arm.arm_id}/world4.receipt.json"
        for cell in CELL_SPECS
        for arm in ARM_SPECS
    }
    expected_dirs = {
        str(cell["cell_id"]) for cell in CELL_SPECS
    } | {
        f"{cell['cell_id']}/{arm.arm_id}"
        for cell in CELL_SPECS
        for arm in ARM_SPECS
    }
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise BraidStage0OrchestrationError("all8 evidence contains a symlink")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_dirs.add(relative)
        else:
            raise BraidStage0OrchestrationError("all8 evidence contains a non-file")
    if observed_files != expected_files or observed_dirs != expected_dirs:
        raise BraidStage0OrchestrationError("all8 evidence closure differs")

    artifacts: list[dict[str, Any]] = []
    global_process_ids: list[str] = []
    canary_counts: dict[str, int] = {
        "two_branch_native_apg_parity": 0,
        "co_state_reset_world4_sp4_oracle": 0,
        "old_motion_action_capacity_oracle": 0,
    }
    cell_coordinates: dict[str, list[tuple[ArmSpec, Mapping[str, Any]]]] = {
        "dog": [],
        "human": [],
    }
    for fixed_cell in CELL_SPECS:
        cell_id = str(fixed_cell["cell_id"])
        cell = plan_cell(plan, cell_id)
        for arm in ARM_SPECS:
            source = root / cell_id / arm.arm_id / "world4.receipt.json"
            receipt = _load_world4(
                source, plan=plan, cell=cell_id, arm=arm.arm_id
            )
            canary_counts[arm.canary] += 1
            cell_coordinates[cell_id].append((arm, receipt["coordinate_evidence"]))
            global_process_ids.extend(
                item["process_start_identity_sha256"]
                for item in receipt["fresh_process_evidence"]
            )
            artifacts.append(
                {
                    "cell_id": cell_id,
                    "arm_id": arm.arm_id,
                    "canary": arm.canary,
                    "world4_receipt_path": str(source),
                    "world4_receipt_file_sha256": file_sha256(source),
                    "world4_receipt_digest": receipt["receipt_digest"],
                    "editor_receipt_file_sha256": cell[
                        "editor_receipt_file_sha256"
                    ],
                    "editor_public_key_file_sha256": plan["provenance"][
                        "editor_public_key_file_sha256"
                    ],
                    "device_environment_evidence_digest": object_sha256(
                        receipt["device_environment_evidence"]
                    ),
                    "execution_public_key_file_sha256": plan[
                        "execution_authentication"
                    ]["public_key_file_sha256"],
                    "world4_execution_signature_sha256": hashlib.sha256(
                        base64.b64decode(
                            receipt[
                                "execution_signature_ed25519_base64"
                            ].encode("ascii"),
                            validate=True,
                        )
                    ).hexdigest(),
                }
            )
    expected_processes = len(CELL_SPECS) * len(ARM_SPECS) * WORLD_SIZE
    if len(global_process_ids) != expected_processes or len(set(global_process_ids)) != expected_processes:
        raise BraidStage0OrchestrationError("one process identity was reused across fresh arms")
    if canary_counts != {
        "two_branch_native_apg_parity": 6,
        "co_state_reset_world4_sp4_oracle": 2,
        "old_motion_action_capacity_oracle": 4,
    }:
        raise BraidStage0OrchestrationError("all8 canary arm counts differ")
    for cell_id, rows in cell_coordinates.items():
        if len(rows) != len(ARM_SPECS):
            raise BraidStage0OrchestrationError(f"{cell_id} coordinate arm count differs")
        for field in (
            "editor_runtime_input_receipt_digest",
            "editor_runtime_input_receipt_file_sha256",
            "editor_public_key_file_sha256",
            "source_latent_sha256",
            "official_initial_noise_sha256",
            "noop_prompt_tensor_sha256",
            "negative_prompt_tensor_sha256",
            "exact40_timestep_sigma_digest",
        ):
            if len({coordinate[field] for _, coordinate in rows}) != 1:
                raise BraidStage0OrchestrationError(
                    f"{cell_id} changed {field} across fresh arms"
                )
        c0 = {
            coordinate["action_prompt_tensor_sha256"]
            for arm, coordinate in rows
            if arm.action_prompt_role == "c0"
        }
        ca = {
            coordinate["action_prompt_tensor_sha256"]
            for arm, coordinate in rows
            if arm.action_prompt_role == "ca"
        }
        noop = {coordinate["noop_prompt_tensor_sha256"] for _, coordinate in rows}
        if len(c0) != 1 or len(ca) != 1 or len(noop) != 1 or c0 != noop or c0 == ca:
            raise BraidStage0OrchestrationError(
                f"{cell_id} c0/ca prompt coordinate closure differs"
            )

    unsigned = {
        "schema_version": ALL8_SCHEMA,
        "method": METHOD,
        "decision": "FORWARD_ONLY_ENGINEERING_CANARIES_COMPLETE",
        "plan_path": str(plan_source),
        "plan_file_sha256": file_sha256(plan_source),
        "plan_receipt_digest": plan["receipt_digest"],
        "topology": plan["topology"],
        "provenance": plan["provenance"],
        "execution_authentication": plan["execution_authentication"],
        "world4_artifacts": artifacts,
        "world4_receipt_count": len(artifacts),
        "fresh_rank_process_count": expected_processes,
        "canary_arm_counts": canary_counts,
        "execution_authority": dict(EXECUTION_AUTHORITY),
        "remaining_required_before_stage_a": [
            "decoded_exact81_semantic_and_preservation_audit",
            "single_step_world4_sp4_backward_recompute_reset_oracle",
            "signed_stage0_authorization_receipt",
        ],
        "stage0_training_authority": False,
        "stage_a_shadow_updates_authorized": 0,
        "scientific_authority": False,
    }
    receipt = seal_receipt(unsigned)
    write_create_only_json(output_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("write-plan", help="create the immutable all8 plan")
    plan.add_argument("--output", required=True)
    plan.add_argument("--slurm-job-id", type=int, required=True)
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--method-source-revision", required=True)
    plan.add_argument("--source-archive-sha256", required=True)
    plan.add_argument("--runtime-source-sha256", required=True)
    plan.add_argument("--runner-source-sha256", required=True)
    plan.add_argument("--dog-editor-receipt-file-sha256", required=True)
    plan.add_argument("--human-editor-receipt-file-sha256", required=True)
    plan.add_argument("--execution-public-key-file-sha256", required=True)

    validate = commands.add_parser("validate-world4", help="reopen one arm receipt")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--receipt", required=True)
    validate.add_argument("--cell-id", choices=tuple(CELL_BY_ID), required=True)
    validate.add_argument("--arm-id", choices=tuple(ARM_BY_ID), required=True)

    aggregate = commands.add_parser("aggregate-all8", help="seal all 12 WORLD4 arms")
    aggregate.add_argument("--plan", required=True)
    aggregate.add_argument("--evidence-root", required=True)
    aggregate.add_argument("--output", required=True)
    partial = commands.add_parser(
        "aggregate-reference4f-a-only-all8",
        help="seal dog+human reference-4f-a as an explicitly partial audit",
    )
    partial.add_argument("--plan", required=True)
    partial.add_argument("--evidence-root", required=True)
    partial.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "write-plan":
        receipt = build_plan(
            slurm_job_id=args.slurm_job_id,
            output_root=args.output_root,
            method_source_revision=args.method_source_revision,
            source_archive_sha256=args.source_archive_sha256,
            runtime_source_sha256=args.runtime_source_sha256,
            runner_source_sha256=args.runner_source_sha256,
            dog_editor_receipt_file_sha256=(
                args.dog_editor_receipt_file_sha256
            ),
            human_editor_receipt_file_sha256=(
                args.human_editor_receipt_file_sha256
            ),
            execution_public_key_file_sha256=(
                args.execution_public_key_file_sha256
            ),
        )
        write_create_only_json(args.output, receipt)
    elif args.command == "validate-world4":
        plan = _load_plan(args.plan)
        receipt = _load_world4(
            Path(args.receipt), plan=plan, cell=args.cell_id, arm=args.arm_id
        )
    elif args.command == "aggregate-all8":
        receipt = aggregate_all8(
            plan_path=args.plan, evidence_root=args.evidence_root, output=args.output
        )
    else:
        receipt = aggregate_reference4f_a_only_all8(
            plan_path=args.plan, evidence_root=args.evidence_root, output=args.output
        )
    print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL8_SCHEMA",
    "ARM_BY_ID",
    "ARM_SPECS",
    "BraidStage0OrchestrationError",
    "CELL_BY_ID",
    "CELL_SPECS",
    "DUAL_RUNTIME_SCHEMA",
    "EXECUTION_AUTHORITY",
    "METHOD",
    "PLAN_SCHEMA",
    "PROHIBITIONS",
    "REFERENCE4F_A_ONLY_ALL8_SCHEMA",
    "IMPLEMENTED_WORLD4_ARM_IDS",
    "VISUAL_PACK_MODE",
    "WORLD4_SCHEMA",
    "aggregate_reference4f_a_only_all8",
    "aggregate_all8",
    "apg_state_identity_sha256",
    "build_plan",
    "canonical_json_bytes",
    "file_sha256",
    "main",
    "object_sha256",
    "seal_receipt",
    "validate_plan",
    "validate_reference4f_a_only_all8_receipt",
    "validate_sealed_receipt",
    "validate_world4_receipt",
    "write_create_only_json",
]

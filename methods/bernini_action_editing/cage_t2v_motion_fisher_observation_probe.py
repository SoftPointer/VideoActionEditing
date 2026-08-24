#!/usr/bin/env python3
"""Frozen exact81 T2V parameter-gradient observations for CAGE Motion-Fisher.

The executable observes gradients of temporary zero-output LoRA probes.  It
never trains Bernini and never publishes a T2V pixel, latent, noise, velocity,
or hidden state.  The only tensor artifact is a detached CPU FP32 gradient in
the registered LoRA-B coordinates.

The built-in scalar is intentionally a *negative control*.  It is the raw
gradient of a frozen flow-energy log ratio and cannot authorize a Motion-
Fisher fit or an optimizer.  A separately frozen learned temporal-event
critic can be plugged in through :class:`ExternalEventCriticBackend`; its
score is differentiated through Bernini's predicted-clean latent, while the
critic checkpoint and nuisance-removal contract are hash-bound.  The runtime
never changes a score sign.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

import torch
from torch import nn
import torch.nn.functional as F


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import cage_self_generated_motion_fisher as motion_fisher  # noqa: E402
import infer_native_identity_generation_canary as native_generation  # noqa: E402
import mace_candidate_action_energy as mace  # noqa: E402
import pair_v5_t2v_calibration_bank_spec as bank_contract  # noqa: E402
import pair_v5_t2v_energy_calibration_v3 as event_contract  # noqa: E402
import score_pair_v5_t2v_energy_bank_v3 as frozen_runtime  # noqa: E402
import source_self_native_ref_contrastive_v3 as native_schedule  # noqa: E402


SCHEMA_VERSION = "bernini-cage-t2v-motion-fisher-observation-probe-v1"
SAMPLE_RECEIPT_SCHEMA = "bernini-cage-t2v-motion-fisher-sample-observation-v1"
GROUP_RECEIPT_SCHEMA = "bernini-cage-t2v-motion-fisher-group-observation-v1"
MASTER_RECEIPT_SCHEMA = "bernini-cage-t2v-motion-fisher-observation-master-v1"
EVENT_INDEX_SCHEMA = "bernini-cage-t2v-motion-fisher-event-index-v1"
EXTERNAL_BACKEND_SCHEMA = (
    "bernini-cage-frozen-text-temporal-event-critic-backend-v1"
)

REGISTRATION_PATH = (
    METHOD_ROOT / "assets/cage_t2v_motion_fisher_observation_probe_v1.json"
)
REGISTRATION_SHA256 = (
    "2f21c3036f1320ce87067ae969be3df8aa4d1fd03abc8201903bc20a1b7b8fa7"
)
TOPUP_SPEC_PATH = METHOD_ROOT / "assets/cage_t2v_motion_fisher_action_topup_v1.json"
TOPUP_SPEC_SHA256 = (
    "be6134679fe2d791fd32e8680eb20370ee13e75f764118bd51d9c0f65b615916"
)
TOPUP_SPEC_SCHEMA = "bernini-cage-t2v-motion-fisher-action-topup-spec-v1"
TOPUP_PLAN_SCHEMA = "bernini-cage-t2v-motion-fisher-action-topup-plan-v1"
WORLD_SIZE = 4
SP_SIZE = 4
LATENT_PHASES = 21
TRANSFORM_ORDER = tuple(motion_fisher.REQUIRED_TRANSFORMS)
BUILTIN_BACKEND_ID = "frozen_flow_log_ratio_to_same_state_noop_negative_control_v1"
EXTERNAL_BACKEND_ID = "frozen_text_conditioned_temporal_event_critic_raw_score_vjp_v1"
GRADIENT_FILENAME = "motion-fisher-lora-b-gradients.safetensors"
SAMPLE_RECEIPT_FILENAME = "motion-fisher-observation-receipt.json"
GROUP_RECEIPT_FILENAME = "motion-fisher-group-receipt.json"
MASTER_RECEIPT_FILENAME = "motion-fisher-observation-master.json"
GRADIENT_KEY_TEMPLATE = "block_{block:02d}__{transform}"
ENERGY_EPSILON = 1.0e-8

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")

_INDEX_MAPS: Mapping[str, tuple[int, ...]] = {
    "action": tuple(range(LATENT_PHASES)),
    "reverse": tuple(range(LATENT_PHASES - 1, -1, -1)),
    "freeze": (0,) * LATENT_PHASES,
    "shuffle": tuple((8 * index) % LATENT_PHASES for index in range(LATENT_PHASES)),
    "camera": tuple(range(LATENT_PHASES)),
    "appearance": tuple(range(LATENT_PHASES)),
}
_CONDITION_BRANCH: Mapping[str, str] = {
    "action": "action",
    "reverse": "action",
    "freeze": "action",
    "shuffle": "action",
    "camera": "camera_only",
    "appearance": "appearance_only",
}


class MotionFisherObservationProbeError(RuntimeError):
    """An input, runtime, gradient, or receipt escaped the closed probe."""


@dataclass(frozen=True)
class World4SP4:
    world_size: int
    rank: int
    local_rank: int
    local_world_size: int


@dataclass(frozen=True)
class PopulationSelection:
    candidate_ids: tuple[str, ...]
    group_candidate_ids: Mapping[str, tuple[str, ...]]
    action_families: tuple[str, ...]
    sample_metadata: Mapping[str, Mapping[str, Any]]
    population_digest: str


@dataclass(frozen=True)
class ViewScalarResult:
    scalar: float
    component_scalars: Mapping[str, float]
    block_gradients: tuple[torch.Tensor, ...]
    receipt: Mapping[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise MotionFisherObservationProbeError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    value = Path(path)
    digest = hashlib.sha256()
    before = value.stat()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = value.stat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
    if identity(before) != identity(after):
        raise MotionFisherObservationProbeError(f"file changed while hashing: {value}")
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    owned = value.detach().to(device="cpu").contiguous().clone()
    return hashlib.sha256(bytes(owned.untyped_storage())).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MotionFisherObservationProbeError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise MotionFisherObservationProbeError(f"{label} must be lowercase SHA-1")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise MotionFisherObservationProbeError(f"{label} is not path-safe")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise MotionFisherObservationProbeError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise MotionFisherObservationProbeError(
            f"{label} must be an absolute plain directory"
        )
    return path.resolve(strict=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = path.read_bytes()

    def reject_constant(token: str) -> None:
        raise MotionFisherObservationProbeError(
            f"{label} contains non-finite constant {token}"
        )

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MotionFisherObservationProbeError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MotionFisherObservationProbeError(f"{label} is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise MotionFisherObservationProbeError(f"{label} root must be an object")
    return value


def _verify_embedded(value: Mapping[str, Any], *, field: str, label: str) -> str:
    digest = _sha256(value.get(field), label=f"{label} {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if object_sha256(unsigned) != digest:
        raise MotionFisherObservationProbeError(f"{label} embedded digest differs")
    return digest


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise MotionFisherObservationProbeError("receipt path must be fresh")
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if path.exists() or path.is_symlink():
            path.unlink()
        raise
    return hashlib.sha256(raw).hexdigest()


def load_registration(
    path: str | Path = REGISTRATION_PATH,
    expected_sha256: str = REGISTRATION_SHA256,
) -> dict[str, Any]:
    expected = _sha256(expected_sha256, label="registration SHA-256")
    source = _plain_file(path, label="motion-fisher registration")
    if file_sha256(source) != expected:
        raise MotionFisherObservationProbeError("motion-fisher registration hash differs")
    row = _read_json(source, label="motion-fisher registration")
    expected_keys = {
        "schema_version",
        "purpose",
        "required_action_families",
        "minimum_identity_count_per_family",
        "minimum_generation_seeds_per_identity",
        "exact_selected_action_clip_count",
        "frame_contract",
        "candidate_block_indices",
        "projection_suffix",
        "probe_rank",
        "probe_scale",
        "fixed_a_construction",
        "zero_b_required",
        "native_schedule",
        "transform_registry",
        "built_in_scalar_backend",
        "external_scalar_backend_contract",
        "same_action_clip_and_official_gaussian_shared_across_six_views",
        "official_gaussian_temporal_transform_applied",
        "event_qualified_action_receipt_required_before_gpu_compute",
        "sp4_gradient_reduction",
        "output_payload",
        "training_performed",
        "optimizer_created",
        "optimizer_update_authorized",
    }
    coordinate = row.get("native_schedule")
    if (
        set(row) != expected_keys
        or row["schema_version"]
        != "bernini-cage-t2v-motion-fisher-observation-registration-v1"
        or row["purpose"] != "frozen_parameter_gradient_observation_only"
        or row["required_action_families"]
        != ["dog-sit-facing-camera", "human-rise-to-stand"]
        or row["minimum_identity_count_per_family"] != 2
        or row["minimum_generation_seeds_per_identity"] != 2
        or row["exact_selected_action_clip_count"] != 8
        or row["frame_contract"] != "exact81_latent21"
        or row["candidate_block_indices"] != list(range(30))
        or row["projection_suffix"] != "attn2.to_out.0"
        or row["probe_rank"] != 8
        or float(row["probe_scale"]).hex() != float(1.0 / math.sqrt(8.0)).hex()
        or row["zero_b_required"] is not True
        or not isinstance(coordinate, Mapping)
        or coordinate.get("schedule_digest")
        != native_schedule.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST
        or coordinate.get("schedule_index") != 33
        or float(coordinate.get("sigma", -1.0)).hex()
        != float(native_schedule.NATIVE_UNIPC40_SIGMAS[33]).hex()
        or coordinate.get("native_timestep")
        != int(native_schedule.NATIVE_UNIPC40_TIMESTEPS[33])
        or set(row["transform_registry"]) != set(TRANSFORM_ORDER)
        or row["built_in_scalar_backend"].get("backend_id") != BUILTIN_BACKEND_ID
        or row["built_in_scalar_backend"].get("scientific_role")
        != "negative_control_only"
        or row["built_in_scalar_backend"].get("motion_fisher_fit_authorized")
        is not False
        or row["external_scalar_backend_contract"].get("backend_id")
        != EXTERNAL_BACKEND_ID
        or row["event_qualified_action_receipt_required_before_gpu_compute"]
        is not True
        or row["training_performed"] is not False
        or row["optimizer_created"] is not False
        or row["optimizer_update_authorized"] is not False
    ):
        raise MotionFisherObservationProbeError("registration semantic closure differs")
    for transform in TRANSFORM_ORDER:
        if (
            row["transform_registry"][transform].get("condition_branch")
            != _CONDITION_BRANCH[transform]
        ):
            raise MotionFisherObservationProbeError(
                f"registration condition branch differs for {transform}"
            )
    return {**row, "file_sha256": expected, "path": str(source)}


def load_topup_spec(
    path: str | Path,
    expected_sha256: str,
    *,
    base_spec: Mapping[str, Any],
    expected_base_spec_sha256: str,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    """Expand four action-only second seeds into standard PAIR envelopes."""

    source = _plain_file(path, label="action-only top-up spec")
    digest = _sha256(expected_sha256, label="top-up spec SHA-256")
    if file_sha256(source) != digest:
        raise MotionFisherObservationProbeError("top-up spec file hash differs")
    value = _read_json(source, label="action-only top-up spec")
    if set(value) != {
        "schema_version",
        "base_core4_v2_spec_sha256",
        "sampling_contract",
        "semantic_contract",
        "groups",
        "training_performed",
        "optimizer_created",
        "optimizer_update_authorized",
    }:
        raise MotionFisherObservationProbeError("top-up spec field closure differs")
    semantic = {
        "action_branch_only": True,
        "base_action_caption_and_geometry_reused_exactly": True,
        "new_generation_seed_required": True,
        "official_initial_gaussian_required": True,
        "event_audit_not_in_generation_condition": True,
        "negative_branch_generation_required": False,
        "rv2v_condition_target_donor_or_noise": False,
    }
    if (
        value["schema_version"] != TOPUP_SPEC_SCHEMA
        or value["base_core4_v2_spec_sha256"] != expected_base_spec_sha256
        or value["sampling_contract"]
        != "inherit_pair_v5_native_exact81_unipc40_t2v"
        or value["semantic_contract"] != semantic
        or value["training_performed"] is not False
        or value["optimizer_created"] is not False
        or value["optimizer_update_authorized"] is not False
        or not isinstance(value["groups"], list)
        or len(value["groups"]) != 2
    ):
        raise MotionFisherObservationProbeError("top-up spec semantic closure differs")
    try:
        checked_base = bank_contract.validate_root_spec(base_spec)
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise MotionFisherObservationProbeError(str(error)) from error
    base_actions: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for group in checked_base["groups"]:
        for candidate in group["candidates"]:
            if candidate["semantic_branch"] == "action":
                base_actions[candidate["candidate_id"]] = (group["group_id"], candidate)
    if len(base_actions) != 4:
        raise MotionFisherObservationProbeError("base core4 action population differs")
    layout = {
        "sp4-a": [0, 1, 2, 3],
        "sp4-b": [4, 5, 6, 7],
    }
    rows: list[Mapping[str, Any]] = []
    used_base: set[str] = set()
    used_ids: set[str] = set()
    for group in value["groups"]:
        if not isinstance(group, Mapping) or set(group) != {
            "group_id",
            "visible_gpus",
            "candidates",
        }:
            raise MotionFisherObservationProbeError("top-up group closure differs")
        group_id = group["group_id"]
        if layout.get(group_id) != group["visible_gpus"] or not isinstance(
            group["candidates"], list
        ) or len(group["candidates"]) != 2:
            raise MotionFisherObservationProbeError("top-up SP4 layout differs")
        for ordinal, topup in enumerate(group["candidates"]):
            if not isinstance(topup, Mapping) or set(topup) != {
                "candidate_id",
                "base_action_candidate_id",
                "seed",
            }:
                raise MotionFisherObservationProbeError("top-up candidate closure differs")
            candidate_id = _safe_id(topup["candidate_id"], label="top-up candidate ID")
            base_id = _safe_id(
                topup["base_action_candidate_id"], label="base action candidate ID"
            )
            base_binding = base_actions.get(base_id)
            seed = topup["seed"]
            if (
                base_binding is None
                or base_binding[0] != group_id
                or base_id in used_base
                or candidate_id in used_ids
                or type(seed) is not int
                or not 0 <= seed < 2**63
                or seed == base_binding[1]["seed"]
            ):
                raise MotionFisherObservationProbeError("top-up identity/seed binding differs")
            used_base.add(base_id)
            used_ids.add(candidate_id)
            candidate = dict(base_binding[1])
            candidate.update(
                {
                    "candidate_id": candidate_id,
                    "calibration_group_id": (
                        f"mf-topup-{base_binding[1]['calibration_group_id']}-s{seed}"
                    ),
                    "seed": seed,
                }
            )
            try:
                checked_candidate = bank_contract.validate_candidate(candidate)
            except bank_contract.PairT2VCalibrationSpecError as error:
                raise MotionFisherObservationProbeError(str(error)) from error
            rows.append(
                {
                    "group_id": group_id,
                    "visible_gpus": list(group["visible_gpus"]),
                    "ordinal": ordinal,
                    "base_action_candidate_id": base_id,
                    "candidate": checked_candidate,
                }
            )
    if used_base != set(base_actions) or len(rows) != 4:
        raise MotionFisherObservationProbeError(
            "top-up must contain one second seed for every base action identity"
        )
    normalized = {
        **value,
        "path": str(source),
        "file_sha256": digest,
        "expanded_candidate_digest": object_sha256(rows),
    }
    return normalized, tuple(rows)


def materialize_topup_plan(
    *,
    topup_spec_path: str | Path,
    expected_topup_spec_sha256: str,
    base_spec_path: str | Path,
    expected_base_spec_sha256: str,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    try:
        base_spec, observed = bank_contract.load_sealed_spec(
            base_spec_path, expected_base_spec_sha256
        )
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise MotionFisherObservationProbeError(str(error)) from error
    if observed != expected_base_spec_sha256:
        raise MotionFisherObservationProbeError("base spec digest differs")
    topup, rows = load_topup_spec(
        topup_spec_path,
        expected_topup_spec_sha256,
        base_spec=base_spec,
        expected_base_spec_sha256=expected_base_spec_sha256,
    )
    output = _fresh_directory(output_dir)
    output.mkdir(parents=False, exist_ok=False)
    records = []
    for row in rows:
        group_dir = output / row["group_id"]
        group_dir.mkdir(exist_ok=True)
        envelope = {
            "schema_version": bank_contract.CANDIDATE_SCHEMA_VERSION,
            "root_spec_raw_sha256": expected_topup_spec_sha256,
            "group_id": row["group_id"],
            "visible_gpus": row["visible_gpus"],
            "ordinal": row["ordinal"],
            "sampling_contract": bank_contract.SAMPLING_CONTRACT,
            "semantic_input_closure": bank_contract.SEMANTIC_INPUT_CLOSURE,
            "artifact_use_contract": bank_contract.ARTIFACT_USE_CONTRACT,
            "split_contract": bank_contract.SPLIT_CONTRACT,
            "candidate": row["candidate"],
        }
        raw = canonical_json_bytes(envelope) + b"\n"
        path = group_dir / f"{row['ordinal']:04d}-{row['candidate']['candidate_id']}.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        records.append(
            {
                "group_id": row["group_id"],
                "visible_gpus": row["visible_gpus"],
                "candidate_id": row["candidate"]["candidate_id"],
                "base_action_candidate_id": row["base_action_candidate_id"],
                "path": str(path),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    unsigned = {
        "schema_version": TOPUP_PLAN_SCHEMA,
        "topup_spec_sha256": topup["file_sha256"],
        "base_spec_sha256": expected_base_spec_sha256,
        "candidate_count": 4,
        "candidate_records": records,
        "render_command": (
            "torchrun --nproc_per_node=4 infer_pair_v5_t2v_calibration_bank.py "
            "--candidate-spec <record.path> --expected-root-spec-sha256 "
            "<topup_spec_sha256> --output-dir <fresh_candidate_dir> ..."
        ),
        "negative_branch_generation_required": False,
        "event_audit_required_after_generation": True,
        "training_performed": False,
        "optimizer_created": False,
    }
    plan = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_create_only_json(output / "motion-fisher-action-topup-plan.json", plan)
    return plan


def audit_topup_outputs(
    *,
    topup_spec_path: str | Path,
    expected_topup_spec_sha256: str,
    base_spec_path: str | Path,
    expected_base_spec_sha256: str,
    topup_output_dir: str | Path,
) -> Mapping[str, Any]:
    try:
        base_spec, _ = bank_contract.load_sealed_spec(
            base_spec_path, expected_base_spec_sha256
        )
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise MotionFisherObservationProbeError(str(error)) from error
    topup, rows = load_topup_spec(
        topup_spec_path,
        expected_topup_spec_sha256,
        base_spec=base_spec,
        expected_base_spec_sha256=expected_base_spec_sha256,
    )
    bound = [
        *load_topup_group(
            rows=rows,
            topup_spec_sha256=expected_topup_spec_sha256,
            topup_output_dir=topup_output_dir,
            group_id="sp4-a",
        ),
        *load_topup_group(
            rows=rows,
            topup_spec_sha256=expected_topup_spec_sha256,
            topup_output_dir=topup_output_dir,
            group_id="sp4-b",
        ),
    ]
    unsigned = {
        "schema_version": TOPUP_SPEC_SCHEMA,
        "topup_spec_sha256": topup["file_sha256"],
        "base_spec_sha256": expected_base_spec_sha256,
        "candidate_count": len(bound),
        "candidate_ids": sorted(row["candidate"]["candidate_id"] for row in bound),
        "generation_receipt_digests": {
            row["candidate"]["candidate_id"]: row["generation_receipt_digest"]
            for row in bound
        },
        "exact81_unipc40_action_only": True,
        "official_gaussian_authenticated": True,
        "event_audit_performed": False,
        "event_audit_required_before_observation_probe": True,
        "training_performed": False,
        "optimizer_created": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def load_topup_group(
    *,
    rows: Sequence[Mapping[str, Any]],
    topup_spec_sha256: str,
    topup_output_dir: str | Path,
    group_id: str,
) -> list[dict[str, Any]]:
    """Authenticate four action-only outputs made by the standard T2V renderer."""

    bank_runner = frozen_runtime.bank_runner
    root = _plain_directory(topup_output_dir, label="action-only top-up output")
    selected = [row for row in rows if row["group_id"] == group_id]
    if len(selected) != 2:
        raise MotionFisherObservationProbeError("top-up group must contain two actions")
    bound: list[dict[str, Any]] = []
    for row in selected:
        candidate = row["candidate"]
        candidate_dir = root / candidate["candidate_id"]
        if not candidate_dir.is_dir() or candidate_dir.is_symlink():
            raise MotionFisherObservationProbeError("top-up candidate directory differs")
        receipt_path = _plain_file(
            candidate_dir / "pair-v5-t2v-calibration-receipt.json",
            label="top-up PAIR generation receipt",
        )
        try:
            receipt = bank_runner._load_pair_receipt(receipt_path)
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise MotionFisherObservationProbeError(str(error)) from error
        if (
            receipt["root_spec_raw_sha256"] != topup_spec_sha256
            or receipt["candidate"] != candidate
            or receipt["group_id"] != group_id
            or receipt["visible_gpus"] != row["visible_gpus"]
            or receipt["ordinal"] != row["ordinal"]
            or receipt["sampling_contract"] != bank_contract.SAMPLING_CONTRACT
            or receipt["semantic_input_closure"] != bank_contract.SEMANTIC_INPUT_CLOSURE
            or receipt["artifact_use_contract"] != bank_contract.ARTIFACT_USE_CONTRACT
            or receipt["split_contract"] != bank_contract.SPLIT_CONTRACT
            or receipt["interpretation"].get("training_performed") is not False
            or receipt["interpretation"].get("optimizer_authorized") is not False
        ):
            raise MotionFisherObservationProbeError("top-up generation receipt differs")
        native_path = _plain_file(
            receipt["native_receipt_path"], label="top-up native receipt"
        )
        if native_path.parent != candidate_dir or file_sha256(native_path) != receipt[
            "native_receipt_sha256"
        ]:
            raise MotionFisherObservationProbeError("top-up native receipt path/hash differs")
        try:
            native_raw = bank_runner._load_json(native_path, "top-up native receipt")
            native_artifacts = bank_runner._verify_native_receipt(native_raw, candidate)
            artifacts = {
                name: bank_runner._verify_file_artifact(
                    receipt["artifacts"][name], f"top-up {candidate['candidate_id']} {name}"
                )
                for name in (
                    "mp4",
                    "predecode_clean_latent",
                    "official_initial_gaussian",
                )
            }
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise MotionFisherObservationProbeError(str(error)) from error
        if (
            artifacts != {
                name: native_artifacts[name]
                for name in (
                    "mp4",
                    "predecode_clean_latent",
                    "official_initial_gaussian",
                )
            }
            or native_artifacts["native_receipt_digest"]
            != receipt["native_receipt_digest"]
        ):
            raise MotionFisherObservationProbeError("top-up artifact binding differs")
        bound.append(
            {
                "candidate": candidate,
                "base_action_candidate_id": row["base_action_candidate_id"],
                "candidate_envelope_sha256": receipt["candidate_envelope_sha256"],
                "generation_receipt_digest": receipt["receipt_digest"],
                "generation_receipt_file_sha256": file_sha256(receipt_path),
                "native_rollout_receipt_digest": receipt["native_receipt_digest"],
                "native_rollout_receipt_file_sha256": file_sha256(native_path),
                "artifacts": artifacts,
            }
        )
    return bound


def world4_sp4_contract(environment: Mapping[str, str] = os.environ) -> World4SP4:
    values: dict[str, int] = {}
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        try:
            values[name] = int(environment.get(name, ""))
        except (TypeError, ValueError) as error:
            raise MotionFisherObservationProbeError(
                f"invalid torchrun environment field {name}"
            ) from error
    if values["WORLD_SIZE"] != WORLD_SIZE or values["LOCAL_WORLD_SIZE"] != WORLD_SIZE:
        raise MotionFisherObservationProbeError("probe requires exact WORLD4")
    if values["RANK"] != values["LOCAL_RANK"] or not 0 <= values["RANK"] < WORLD_SIZE:
        raise MotionFisherObservationProbeError("rank/local-rank topology differs")
    return World4SP4(
        world_size=WORLD_SIZE,
        rank=values["RANK"],
        local_rank=values["LOCAL_RANK"],
        local_world_size=WORLD_SIZE,
    )


def validate_population_coverage(
    spec: Mapping[str, Any],
    registration: Mapping[str, Any],
    *,
    topup_rows: Sequence[Mapping[str, Any]] = (),
    topup_spec_sha256: Optional[str] = None,
) -> PopulationSelection:
    try:
        checked = bank_contract.validate_root_spec(spec)
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise MotionFisherObservationProbeError(str(error)) from error
    actions: list[tuple[str, Mapping[str, Any], Optional[str]]] = []
    for group in checked["groups"]:
        for candidate in group["candidates"]:
            if candidate["semantic_branch"] == "action":
                actions.append((group["group_id"], candidate, None))
    base_actions = {
        candidate["candidate_id"]: (group_id, candidate)
        for group_id, candidate, _ in actions
    }
    for raw in topup_rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "group_id",
            "visible_gpus",
            "ordinal",
            "base_action_candidate_id",
            "candidate",
        }:
            raise MotionFisherObservationProbeError("top-up row field closure differs")
        candidate = raw["candidate"]
        try:
            candidate = bank_contract.validate_candidate(candidate)
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise MotionFisherObservationProbeError(str(error)) from error
        base_id = _safe_id(
            raw["base_action_candidate_id"], label="top-up base action candidate ID"
        )
        base = base_actions.get(base_id)
        if (
            base is None
            or raw["group_id"] != base[0]
            or candidate["semantic_branch"] != "action"
            or candidate["seed"] == base[1]["seed"]
            or any(
                candidate[field] != base[1][field]
                for field in (
                    "analysis_split",
                    "action_family_id",
                    "actor_group_id",
                    "scene_group_id",
                    "action_group_id",
                    "prompt_group_id",
                    "action_family_group_id",
                    "full_t2v_caption",
                    "full_t2v_caption_utf8_sha256",
                    "geometry_source_video",
                    "geometry_source_video_sha256",
                    "caption_contract",
                    "geometry_contract",
                )
            )
        ):
            raise MotionFisherObservationProbeError(
                "top-up must change only candidate/calibration IDs and generation seed"
            )
        actions.append((raw["group_id"], candidate, base_id))
    required_families = tuple(registration["required_action_families"])
    if (
        len(actions) != registration["exact_selected_action_clip_count"]
        or {candidate["action_family_id"] for _, candidate, _ in actions}
        != set(required_families)
    ):
        raise MotionFisherObservationProbeError(
            "bank must expose exactly eight selected action clips in two families"
        )
    keys: set[tuple[str, str, int]] = set()
    metadata: dict[str, Mapping[str, Any]] = {}
    by_group: dict[str, list[str]] = {"sp4-a": [], "sp4-b": []}
    for group_id, candidate, base_action_id in actions:
        key = (
            candidate["action_family_id"],
            candidate["actor_group_id"],
            candidate["seed"],
        )
        if key in keys:
            raise MotionFisherObservationProbeError(
                "action family/identity/generation-seed tuple repeats"
            )
        keys.add(key)
        candidate_id = candidate["candidate_id"]
        metadata[candidate_id] = {
            "candidate_id": candidate_id,
            "action_family": candidate["action_family_id"],
            "identity_key": candidate["actor_group_id"],
            "seed_key": f"generation-seed-{candidate['seed']}",
            "seed": candidate["seed"],
            "calibration_group_id": candidate["calibration_group_id"],
            "analysis_split": candidate["analysis_split"],
            "group_id": group_id,
            "prompt_action_candidate_id": base_action_id or candidate_id,
        }
        by_group[group_id].append(candidate_id)
    for family in required_families:
        family_rows = [key for key in keys if key[0] == family]
        identities = {key[1] for key in family_rows}
        if len(identities) < registration["minimum_identity_count_per_family"]:
            raise MotionFisherObservationProbeError(
                f"family {family} lacks two independent identities"
            )
        for identity in identities:
            seeds = {key[2] for key in family_rows if key[1] == identity}
            if len(seeds) < registration["minimum_generation_seeds_per_identity"]:
                raise MotionFisherObservationProbeError(
                    f"family {family}, identity {identity} lacks two generation seeds"
                )
    if any(len(by_group[group]) != 4 for group in ("sp4-a", "sp4-b")):
        raise MotionFisherObservationProbeError(
            "each SP4 group must own exactly four selected action clips"
        )
    population_payload = {
        "normalized_base_spec_digest": object_sha256(checked),
        "topup_spec_sha256": topup_spec_sha256,
        "candidate_ids": sorted(metadata),
        "sample_metadata": {key: metadata[key] for key in sorted(metadata)},
    }
    return PopulationSelection(
        candidate_ids=tuple(sorted(metadata)),
        group_candidate_ids={key: tuple(value) for key, value in by_group.items()},
        action_families=required_families,
        sample_metadata=metadata,
        population_digest=object_sha256(population_payload),
    )


def _bank_generation_digest_by_candidate(bank: Mapping[str, Any]) -> dict[str, str]:
    rows = bank.get("candidate_receipts")
    if not isinstance(rows, list):
        raise MotionFisherObservationProbeError("bank candidate receipt index differs")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise MotionFisherObservationProbeError("bank candidate receipt row differs")
        candidate_id = _safe_id(row.get("candidate_id"), label="bank candidate ID")
        digest = _sha256(row.get("receipt_digest"), label="generation receipt digest")
        if candidate_id in result:
            raise MotionFisherObservationProbeError("bank candidate IDs repeat")
        result[candidate_id] = digest
    return result


def seal_event_index(
    *,
    root_spec_sha256: str,
    bank_receipt_digest: str,
    rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    normalized = []
    for raw in rows:
        if set(raw) != {
            "candidate_id",
            "event_receipt_path",
            "event_receipt_file_sha256",
            "event_receipt_digest",
        }:
            raise MotionFisherObservationProbeError("event index row field closure differs")
        normalized.append(
            {
                "candidate_id": _safe_id(raw["candidate_id"], label="event candidate ID"),
                "event_receipt_path": str(raw["event_receipt_path"]),
                "event_receipt_file_sha256": _sha256(
                    raw["event_receipt_file_sha256"], label="event receipt file SHA-256"
                ),
                "event_receipt_digest": _sha256(
                    raw["event_receipt_digest"], label="event receipt digest"
                ),
            }
        )
    normalized.sort(key=lambda item: item["candidate_id"])
    if len({item["candidate_id"] for item in normalized}) != len(normalized):
        raise MotionFisherObservationProbeError("event index candidate IDs repeat")
    unsigned = {
        "schema_version": EVENT_INDEX_SCHEMA,
        "root_spec_sha256": _sha256(root_spec_sha256, label="root spec SHA-256"),
        "bank_receipt_digest": _sha256(
            bank_receipt_digest, label="bank receipt digest"
        ),
        "rows": normalized,
        "external_labels_are_detached_booleans": True,
        "labels_used_as_model_condition": False,
        "event_receipts_required_before_gpu_compute": True,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def validate_event_index(
    value: Mapping[str, Any],
    *,
    selection: PopulationSelection,
    population_digest: str,
    bank: Mapping[str, Any],
    topup_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if set(value) != {
        "schema_version",
        "root_spec_sha256",
        "bank_receipt_digest",
        "rows",
        "external_labels_are_detached_booleans",
        "labels_used_as_model_condition",
        "event_receipts_required_before_gpu_compute",
        "receipt_digest",
    }:
        raise MotionFisherObservationProbeError("event index field closure differs")
    _verify_embedded(value, field="receipt_digest", label="event index")
    if (
        value["schema_version"] != EVENT_INDEX_SCHEMA
        or value["root_spec_sha256"] != population_digest
        or value["bank_receipt_digest"] != bank.get("receipt_digest")
        or value["external_labels_are_detached_booleans"] is not True
        or value["labels_used_as_model_condition"] is not False
        or value["event_receipts_required_before_gpu_compute"] is not True
        or not isinstance(value["rows"], list)
    ):
        raise MotionFisherObservationProbeError("event index semantic closure differs")
    generation = _bank_generation_digest_by_candidate(bank)
    generation.update(
        {
            row["candidate"]["candidate_id"]: _sha256(
                row["generation_receipt_digest"],
                label="top-up generation receipt digest",
            )
            for row in topup_rows
        }
    )
    indexed: dict[str, Mapping[str, Any]] = {}
    for binding in value["rows"]:
        if not isinstance(binding, Mapping) or set(binding) != {
            "candidate_id",
            "event_receipt_path",
            "event_receipt_file_sha256",
            "event_receipt_digest",
        }:
            raise MotionFisherObservationProbeError("event binding differs")
        candidate_id = _safe_id(binding["candidate_id"], label="event candidate ID")
        path = _plain_file(binding["event_receipt_path"], label="event audit receipt")
        expected_file = _sha256(
            binding["event_receipt_file_sha256"], label="event receipt file SHA-256"
        )
        if file_sha256(path) != expected_file:
            raise MotionFisherObservationProbeError("event receipt file hash differs")
        raw = _read_json(path, label="event audit receipt")
        try:
            receipt = event_contract.validate_event_audit_receipt(raw)
        except event_contract.PairV5EnergyCalibrationV3Error as error:
            raise MotionFisherObservationProbeError(str(error)) from error
        meta = selection.sample_metadata.get(candidate_id)
        if (
            meta is None
            or candidate_id in indexed
            or receipt["candidate_id"] != candidate_id
            or receipt["receipt_digest"] != binding["event_receipt_digest"]
            or receipt["generation_receipt_digest"] != generation.get(candidate_id)
            or receipt["semantic_branch"] != "action"
            or receipt["action_family_id"] != meta["action_family"]
            or receipt["calibration_group_id"] != meta["calibration_group_id"]
            or receipt["actor_group_id"] != meta["identity_key"]
            or receipt["analysis_split"] != meta["analysis_split"]
            or receipt["event_qualified_action_positive"] is not True
            or receipt["complete_target_transition_observed"] is not True
            or receipt["terminal_hold_observed"] is not True
            or receipt["full_target_action_observed"] is not True
        ):
            raise MotionFisherObservationProbeError(
                f"event receipt does not qualify selected action clip {candidate_id}"
            )
        indexed[candidate_id] = {
            "receipt": receipt,
            "path": str(path),
            "file_sha256": expected_file,
        }
    if set(indexed) != set(selection.candidate_ids):
        raise MotionFisherObservationProbeError(
            "event index must qualify exactly all eight selected action clips"
        )
    return indexed


def materialize_event_index(
    *,
    registration_path: str | Path,
    expected_registration_sha256: str,
    root_spec_path: str | Path,
    expected_root_spec_sha256: str,
    bank_output_dir: str | Path,
    bank_receipt_path: str | Path,
    expected_bank_receipt_sha256: str,
    topup_spec_path: str | Path,
    expected_topup_spec_sha256: str,
    topup_output_dir: str | Path,
    event_receipt_paths: Sequence[str | Path],
    output_path: str | Path,
) -> Mapping[str, Any]:
    """Authenticate eight detached event audits and seal their population index.

    Both base SP4 groups are loaded independently so this command authenticates
    every selected base action artifact, not merely the global bank receipt.
    The output is create-only and is immediately round-tripped through the same
    validator used by GPU preflight.
    """

    registration = load_registration(
        registration_path, expected_registration_sha256
    )
    loaded = []
    for group_id in ("sp4-a", "sp4-b"):
        try:
            loaded.append(
                frozen_runtime.load_group_bank(
                    root_spec=root_spec_path,
                    root_spec_sha256=expected_root_spec_sha256,
                    bank_output_dir=bank_output_dir,
                    bank_receipt=bank_receipt_path,
                    bank_receipt_sha256=expected_bank_receipt_sha256,
                    group_id=group_id,
                )
            )
        except frozen_runtime.PairV5T2VEnergyScoringError as error:
            raise MotionFisherObservationProbeError(str(error)) from error
    spec, bank, _ = loaded[0]
    if loaded[1][0] != spec or loaded[1][1] != bank:
        raise MotionFisherObservationProbeError(
            "base bank authentication differs across SP4 groups"
        )
    _, topup_spec_rows = load_topup_spec(
        topup_spec_path,
        expected_topup_spec_sha256,
        base_spec=spec,
        expected_base_spec_sha256=expected_root_spec_sha256,
    )
    topup_rows = [
        *load_topup_group(
            rows=topup_spec_rows,
            topup_spec_sha256=expected_topup_spec_sha256,
            topup_output_dir=topup_output_dir,
            group_id="sp4-a",
        ),
        *load_topup_group(
            rows=topup_spec_rows,
            topup_spec_sha256=expected_topup_spec_sha256,
            topup_output_dir=topup_output_dir,
            group_id="sp4-b",
        ),
    ]
    selection = validate_population_coverage(
        spec,
        registration,
        topup_rows=topup_spec_rows,
        topup_spec_sha256=expected_topup_spec_sha256,
    )
    if len(event_receipt_paths) != len(selection.candidate_ids):
        raise MotionFisherObservationProbeError(
            "exactly eight detached event receipt paths are required"
        )
    bindings = []
    for raw_path in event_receipt_paths:
        path = _plain_file(raw_path, label="detached event audit receipt")
        raw = _read_json(path, label="detached event audit receipt")
        try:
            receipt = event_contract.validate_event_audit_receipt(raw)
        except event_contract.PairV5EnergyCalibrationV3Error as error:
            raise MotionFisherObservationProbeError(str(error)) from error
        bindings.append(
            {
                "candidate_id": receipt["candidate_id"],
                "event_receipt_path": str(path),
                "event_receipt_file_sha256": file_sha256(path),
                "event_receipt_digest": receipt["receipt_digest"],
            }
        )
    index = seal_event_index(
        root_spec_sha256=selection.population_digest,
        bank_receipt_digest=bank["receipt_digest"],
        rows=bindings,
    )
    validate_event_index(
        index,
        selection=selection,
        population_digest=selection.population_digest,
        bank=bank,
        topup_rows=topup_rows,
    )
    destination = Path(output_path)
    if not destination.is_absolute() or destination == Path("/"):
        raise MotionFisherObservationProbeError(
            "event index output must be an absolute non-root path"
        )
    destination = destination.parent.resolve(strict=True) / destination.name
    file_digest = _write_create_only_json(destination, index)
    return {
        "event_index_path": str(destination),
        "event_index_file_sha256": file_digest,
        "event_index_receipt_digest": index["receipt_digest"],
        "population_digest": selection.population_digest,
        "event_qualified_candidate_ids": list(selection.candidate_ids),
        "gpu_compute_performed": False,
        "training_performed": False,
        "optimizer_created": False,
    }


def _walsh_sign(row: int, column: int) -> float:
    return -1.0 if (bin(row & column).count("1") & 1) else 1.0


def make_fixed_orthogonal_a(
    *, rank: int = 8, in_features: int = 1536, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    if rank != 8 or in_features != 1536 or in_features != 3 * 512:
        raise MotionFisherObservationProbeError("fixed-A geometry differs")
    base = torch.tensor(
        [[_walsh_sign(row, column) for column in range(512)] for row in range(rank)],
        dtype=torch.float64,
    )
    value = torch.cat((base, base, base), dim=1) / math.sqrt(float(in_features))
    gram = value @ value.T
    if not torch.allclose(gram, torch.eye(rank, dtype=torch.float64), atol=1e-12, rtol=0.0):
        raise MotionFisherObservationProbeError("fixed-A Walsh rows are not orthonormal")
    result = value.to(dtype=dtype).contiguous()
    cast_gram = result.float() @ result.float().T
    if not torch.allclose(cast_gram, torch.eye(rank), atol=8e-3, rtol=0.0):
        raise MotionFisherObservationProbeError("fixed-A model-dtype rows lost orthogonality")
    return result


class FixedAZeroBProbeBank(nn.Module):
    """Hook-only probes; base Bernini module names and parameters never change."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        block_indices: Sequence[int] = tuple(range(30)),
        rank: int = 8,
    ) -> None:
        super().__init__()
        blocks = getattr(transformer, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != 30:
            raise MotionFisherObservationProbeError("Bernini 1.3B must expose 30 blocks")
        self.block_indices = tuple(int(item) for item in block_indices)
        if self.block_indices != tuple(range(30)) or rank != 8:
            raise MotionFisherObservationProbeError("probe block/rank registry differs")
        dtype = getattr(transformer, "dtype", None)
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise MotionFisherObservationProbeError("transformer dtype differs")
        fixed_a = make_fixed_orthogonal_a(rank=rank, dtype=dtype)
        self.register_buffer("fixed_a", fixed_a, persistent=True)
        self.probe_b = nn.ParameterList()
        targets: list[nn.Linear] = []
        for index in self.block_indices:
            try:
                target = blocks[index].attn2.to_out[0]
            except (AttributeError, IndexError, TypeError) as error:
                raise MotionFisherObservationProbeError(
                    f"block {index} cross-attention output projection differs"
                ) from error
            if (
                not isinstance(target, nn.Linear)
                or target.in_features != 1536
                or target.out_features != 1536
                or target.weight.requires_grad
                or target.weight.grad is not None
            ):
                raise MotionFisherObservationProbeError(
                    f"block {index} projection is not frozen 1536x1536 Linear"
                )
            targets.append(target)
            self.probe_b.append(
                nn.Parameter(
                    torch.zeros((target.out_features, rank), dtype=dtype),
                    requires_grad=True,
                )
            )
        self.targets = tuple(targets)
        self.scale = float(1.0 / math.sqrt(rank))
        self._handles: list[Any] = []

    def coordinate_receipt(self, block_index: int) -> dict[str, Any]:
        if block_index not in self.block_indices:
            raise MotionFisherObservationProbeError("block is outside probe registry")
        parameter = self.probe_b[self.block_indices.index(block_index)]
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "block_index": block_index,
            "projection_name": f"transformer_1.blocks.{block_index}.attn2.to_out.0",
            "base_in_features": 1536,
            "base_out_features": 1536,
            "probe_rank": 8,
            "probe_scale_float64_hex": self.scale.hex(),
            "fixed_a_shape": list(self.fixed_a.shape),
            "fixed_a_dtype": str(self.fixed_a.dtype).removeprefix("torch."),
            "fixed_a_sha256": tensor_sha256(self.fixed_a),
            "lora_b_shape": list(parameter.shape),
            "lora_b_dtype": str(parameter.dtype).removeprefix("torch."),
            "flatten_order": "C_contiguous_out_features_then_rank",
            "zero_b_at_every_forward": True,
            "base_projection_replaced": False,
            "hook_only_additive_probe": True,
        }
        return {**unsigned, "coordinate_digest": object_sha256(unsigned)}

    def assert_zero_clean(self) -> None:
        for index, parameter in zip(self.block_indices, self.probe_b):
            if parameter.grad is not None or bool(torch.count_nonzero(parameter).item()):
                raise MotionFisherObservationProbeError(
                    f"block {index} probe-B is nonzero or retains a gradient"
                )

    def _hook(self, position: int) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def add_probe(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            if (
                module is not self.targets[position]
                or len(inputs) != 1
                or not isinstance(inputs[0], torch.Tensor)
                or not isinstance(output, torch.Tensor)
                or inputs[0].shape[:-1] != output.shape[:-1]
                or inputs[0].shape[-1] != 1536
                or output.shape[-1] != 1536
            ):
                raise MotionFisherObservationProbeError("LoRA probe hook tensor closure differs")
            a = self.fixed_a.to(device=inputs[0].device)
            b = self.probe_b[position]
            projected = F.linear(inputs[0], a)
            delta = F.linear(projected, b) * self.scale
            return output + delta.to(dtype=output.dtype)

        return add_probe

    @contextmanager
    def installed(self) -> Iterable["FixedAZeroBProbeBank"]:
        if self._handles:
            raise MotionFisherObservationProbeError("probe hooks are already installed")
        self.assert_zero_clean()
        self._handles = [
            target.register_forward_hook(self._hook(position))
            for position, target in enumerate(self.targets)
        ]
        try:
            yield self
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            for parameter in self.probe_b:
                parameter.grad = None
            self.assert_zero_clean()


def apply_registered_transform(value: torch.Tensor, transform: str) -> torch.Tensor:
    if (
        transform not in TRANSFORM_ORDER
        or not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3]) != (1, 16, 21)
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise MotionFisherObservationProbeError("exact81 transform input differs")
    index = torch.tensor(_INDEX_MAPS[transform], dtype=torch.long, device=value.device)
    result = value.index_select(2, index).contiguous()
    if result.shape != value.shape or result.requires_grad or result.grad_fn is not None:
        raise MotionFisherObservationProbeError("exact81 transform output differs")
    return result


def _unpack_prediction(packed: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = map(int, reference.shape)
    expected = (batch, frames * (height // 2) * (width // 2), channels * 4)
    if tuple(packed.shape) != expected:
        raise MotionFisherObservationProbeError(
            f"packed prediction shape differs: expected={expected}, actual={tuple(packed.shape)}"
        )
    return (
        packed.reshape(batch, frames, height // 2, width // 2, 1, 2, 2, channels)
        .permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(reference.shape)
        .contiguous()
    )


def forward_predicted_clean(
    *,
    diffusion: nn.Module,
    transformer: nn.Module,
    clean_view: torch.Tensor,
    official_gaussian: torch.Tensor,
    condition: torch.Tensor,
    sigma: float,
    native_timestep: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if clean_view.shape != official_gaussian.shape:
        raise MotionFisherObservationProbeError("clean/noise geometry differs")
    dtype = getattr(transformer, "dtype", None)
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise MotionFisherObservationProbeError("transformer dtype differs")
    if (
        tuple(condition.shape) != (1, 512, 4096)
        or condition.device != clean_view.device
        or condition.requires_grad
        or condition.grad_fn is not None
    ):
        raise MotionFisherObservationProbeError("prompt condition differs")
    x_sigma = ((1.0 - sigma) * clean_view + sigma * official_gaussian).detach().contiguous()
    velocity_target = (official_gaussian - clean_view).detach().contiguous()
    patched = transformer.patch_vae_latent(x_sigma.to(dtype=dtype), source_id=0)
    if not isinstance(patched, (tuple, list)) or len(patched) != 2:
        raise MotionFisherObservationProbeError("patch_vae_latent output differs")
    tokens, rotary = patched
    timestep = torch.tensor(
        [float(native_timestep)], dtype=torch.float32, device=clean_view.device
    )
    prediction = diffusion.shared_step(
        model_id="transformer_1",
        noisy_latents=tokens,
        timesteps=timestep,
        cond_embeds=condition,
        rotary_embs=rotary,
        batch_vae_seqlen=[int(tokens.shape[1])],
        batch_text_seqlen=[512],
    )
    spatial = _unpack_prediction(prediction, clean_view)
    predicted_clean = x_sigma - float(sigma) * spatial.float()
    if (
        spatial.grad_fn is None
        or not spatial.requires_grad
        or predicted_clean.grad_fn is None
        or not predicted_clean.requires_grad
        or not bool(torch.isfinite(spatial).all().item())
        or not bool(torch.isfinite(predicted_clean).all().item())
    ):
        raise MotionFisherObservationProbeError("prediction graph closure differs")
    return spatial, predicted_clean, velocity_target


def _parameter_gradients(
    scalar: torch.Tensor, probe: FixedAZeroBProbeBank
) -> tuple[torch.Tensor, ...]:
    if (
        not isinstance(scalar, torch.Tensor)
        or scalar.shape != torch.Size([])
        or scalar.dtype != torch.float32
        or not scalar.requires_grad
        or scalar.grad_fn is None
        or not bool(torch.isfinite(scalar).item())
    ):
        raise MotionFisherObservationProbeError("probe scalar must be finite FP32 scalar")
    gradients = torch.autograd.grad(
        scalar,
        tuple(probe.probe_b),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    result = tuple(value.detach().float().contiguous() for value in gradients)
    if len(result) != 30 or any(
        value.shape != (1536, 8) or not bool(torch.isfinite(value).all().item())
        for value in result
    ):
        raise MotionFisherObservationProbeError("LoRA-B gradient closure differs")
    for parameter in probe.probe_b:
        if parameter.grad is not None or bool(torch.count_nonzero(parameter).item()):
            raise MotionFisherObservationProbeError("autograd mutated zero-B probe state")
    return result


def _sp4_reduce_gradients(
    gradients: Sequence[torch.Tensor], distributed_module: Any
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any]]:
    if len(gradients) != 30:
        raise MotionFisherObservationProbeError("SP4 gradient registry differs")
    flat = torch.cat([item.reshape(-1) for item in gradients], dim=0).float().contiguous()
    if flat.numel() != 30 * 1536 * 8 or not bool(torch.isfinite(flat).all().item()):
        raise MotionFisherObservationProbeError("SP4 flat gradient differs")
    local_identity = {
        "sha256": tensor_sha256(flat),
        "norm": float(torch.linalg.vector_norm(flat).item()),
        "numel": int(flat.numel()),
    }
    local_rows: list[Any] = [None] * WORLD_SIZE
    distributed_module.all_gather_object(local_rows, local_identity)
    distributed_module.all_reduce(flat, op=distributed_module.ReduceOp.SUM)
    flat.div_(float(WORLD_SIZE))
    reduced_identity = {
        "sha256": tensor_sha256(flat),
        "norm": float(torch.linalg.vector_norm(flat).item()),
        "numel": int(flat.numel()),
    }
    reduced_rows: list[Any] = [None] * WORLD_SIZE
    distributed_module.all_gather_object(reduced_rows, reduced_identity)
    if any(row != reduced_rows[0] for row in reduced_rows[1:]):
        raise MotionFisherObservationProbeError("SP4 reduced gradients differ by rank")
    chunks = flat.split(1536 * 8)
    output = tuple(chunk.reshape(1536, 8).detach().cpu().contiguous() for chunk in chunks)
    receipt = {
        "world_size": WORLD_SIZE,
        "reduction": "all_reduce_sum_fp32_then_divide_world4",
        "local_gradient_identities": local_rows,
        "reduced_gradient_identity": reduced_identity,
        "reduced_all_rank_exact_consensus": True,
    }
    return output, {**receipt, "digest": object_sha256(receipt)}


def _require_scalar_consensus(
    values: Mapping[str, float], distributed_module: Any
) -> Mapping[str, Any]:
    encoded = {
        key: struct.pack("!f", float(value)).hex() for key, value in sorted(values.items())
    }
    rows: list[Any] = [None] * WORLD_SIZE
    distributed_module.all_gather_object(rows, encoded)
    if any(row != rows[0] for row in rows[1:]):
        raise MotionFisherObservationProbeError("view scalars differ across SP4 ranks")
    result = {"float32_be_hex": encoded, "all_rank_exact_consensus": True}
    return {**result, "digest": object_sha256(result)}


def builtin_negative_control_vjp(
    *,
    transform: str,
    diffusion: nn.Module,
    transformer: nn.Module,
    probe: FixedAZeroBProbeBank,
    clean_view: torch.Tensor,
    official_gaussian: torch.Tensor,
    condition_by_branch: Mapping[str, torch.Tensor],
    sigma: float,
    native_timestep: int,
    distributed_module: Any,
) -> ViewScalarResult:
    branch = _CONDITION_BRANCH[transform]

    def component(prompt_branch: str) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        prediction, _, target = forward_predicted_clean(
            diffusion=diffusion,
            transformer=transformer,
            clean_view=clean_view,
            official_gaussian=official_gaussian,
            condition=condition_by_branch[prompt_branch],
            sigma=sigma,
            native_timestep=native_timestep,
        )
        energy = torch.mean((prediction.float() - target.float()).square())
        gradients = _parameter_gradients(energy, probe)
        return energy.detach(), gradients

    condition_energy, condition_gradients = component(branch)
    noop_energy, noop_gradients = component("noop")
    condition_value = float(condition_energy.item())
    noop_value = float(noop_energy.item())
    if min(condition_value, noop_value) < 0.0:
        raise MotionFisherObservationProbeError("flow energy is negative")
    scalar_value = math.log(condition_value + ENERGY_EPSILON) - math.log(
        noop_value + ENERGY_EPSILON
    )
    raw = tuple(
        left / (condition_value + ENERGY_EPSILON)
        - right / (noop_value + ENERGY_EPSILON)
        for left, right in zip(condition_gradients, noop_gradients)
    )
    reduced, reduction_receipt = _sp4_reduce_gradients(raw, distributed_module)
    scalar_receipt = _require_scalar_consensus(
        {
            "condition_energy": condition_value,
            "noop_energy": noop_value,
            "log_energy_ratio": scalar_value,
        },
        distributed_module,
    )
    unsigned = {
        "backend_id": BUILTIN_BACKEND_ID,
        "scientific_role": "negative_control_only",
        "transform": transform,
        "condition_branch": branch,
        "noop_branch": "noop",
        "formula": "log(E_condition+1e-8)-log(E_same_state_noop+1e-8)",
        "condition_energy": condition_value,
        "noop_energy": noop_value,
        "scalar": scalar_value,
        "raw_parameter_gradient_no_manual_sign_flip": True,
        "model_forwards": 2,
        "model_backwards": 2,
        "scalar_consensus": scalar_receipt,
        "gradient_reduction": reduction_receipt,
        "motion_fisher_fit_authorized": False,
        "optimizer_update_authorized": False,
    }
    return ViewScalarResult(
        scalar=scalar_value,
        component_scalars={
            "condition_energy": condition_value,
            "noop_energy": noop_value,
        },
        block_gradients=reduced,
        receipt={**unsigned, "digest": object_sha256(unsigned)},
    )


class EventCriticPlugin(Protocol):
    backend_id: str

    def contract_receipt(self) -> Mapping[str, Any]: ...

    def score_predicted_clean(
        self,
        *,
        transform: str,
        predicted_clean: torch.Tensor,
        action_caption: str,
        condition_caption: str,
        metadata: Mapping[str, Any],
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class ExternalEventCriticBackend:
    plugin: EventCriticPlugin
    implementation_path: Path
    implementation_sha256: str
    contract: Mapping[str, Any]


def load_external_event_critic_backend(
    implementation_path: str | Path,
    expected_sha256: str,
    *,
    device: torch.device,
) -> ExternalEventCriticBackend:
    path = _plain_file(implementation_path, label="external event critic implementation")
    digest = _sha256(expected_sha256, label="external critic implementation SHA-256")
    if file_sha256(path) != digest:
        raise MotionFisherObservationProbeError("external critic implementation hash differs")
    spec = importlib.util.spec_from_file_location(
        f"cage_external_event_critic_{digest[:16]}", path
    )
    if spec is None or spec.loader is None:
        raise MotionFisherObservationProbeError("cannot load external critic module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "build_cage_motion_fisher_event_critic_v1", None)
    if not callable(builder):
        raise MotionFisherObservationProbeError("external critic builder is absent")
    plugin = builder(device=device)
    if getattr(plugin, "backend_id", None) != EXTERNAL_BACKEND_ID:
        raise MotionFisherObservationProbeError("external critic backend ID differs")
    if not callable(getattr(plugin, "contract_receipt", None)) or not callable(
        getattr(plugin, "score_predicted_clean", None)
    ):
        raise MotionFisherObservationProbeError("external critic protocol differs")
    contract = dict(plugin.contract_receipt())
    _verify_embedded(contract, field="receipt_digest", label="external critic contract")
    required_true = {
        "critic_frozen",
        "checkpoint_content_receipt_verified",
        "same_clip_six_view_input_required",
        "identity_adversarial_nuisance_removal_trained",
        "camera_adversarial_nuisance_removal_trained",
        "raw_score_vjp_no_runtime_sign_flip",
        "generated_media_as_target_forbidden",
    }
    if (
        contract.get("schema_version") != EXTERNAL_BACKEND_SCHEMA
        or contract.get("backend_id") != EXTERNAL_BACKEND_ID
        or any(contract.get(name) is not True for name in required_true)
        or contract.get("optimizer_update_authorized") is not False
        or not _SHA256_RE.fullmatch(str(contract.get("critic_checkpoint_receipt_digest")))
    ):
        raise MotionFisherObservationProbeError("external critic contract differs")
    return ExternalEventCriticBackend(
        plugin=plugin,
        implementation_path=path,
        implementation_sha256=digest,
        contract=contract,
    )


def external_event_critic_vjp(
    *,
    backend: ExternalEventCriticBackend,
    transform: str,
    diffusion: nn.Module,
    transformer: nn.Module,
    probe: FixedAZeroBProbeBank,
    clean_view: torch.Tensor,
    official_gaussian: torch.Tensor,
    condition_by_branch: Mapping[str, torch.Tensor],
    caption_by_branch: Mapping[str, str],
    metadata: Mapping[str, Any],
    sigma: float,
    native_timestep: int,
    distributed_module: Any,
) -> ViewScalarResult:
    branch = _CONDITION_BRANCH[transform]
    _, predicted_clean, _ = forward_predicted_clean(
        diffusion=diffusion,
        transformer=transformer,
        clean_view=clean_view,
        official_gaussian=official_gaussian,
        condition=condition_by_branch[branch],
        sigma=sigma,
        native_timestep=native_timestep,
    )
    score = backend.plugin.score_predicted_clean(
        transform=transform,
        predicted_clean=predicted_clean,
        action_caption=caption_by_branch["action"],
        condition_caption=caption_by_branch[branch],
        metadata=dict(metadata),
    )
    gradients = _parameter_gradients(score, probe)
    reduced, reduction_receipt = _sp4_reduce_gradients(gradients, distributed_module)
    scalar_value = float(score.detach().item())
    scalar_receipt = _require_scalar_consensus(
        {"external_event_score": scalar_value}, distributed_module
    )
    unsigned = {
        "backend_id": EXTERNAL_BACKEND_ID,
        "transform": transform,
        "condition_branch": branch,
        "scalar": scalar_value,
        "raw_parameter_gradient_no_manual_sign_flip": True,
        "model_forwards": 1,
        "model_backwards": 1,
        "external_critic_contract_digest": backend.contract["receipt_digest"],
        "external_critic_implementation_sha256": backend.implementation_sha256,
        "score_on": "bernini_predicted_clean_exact81_not_generated_media_target",
        "scalar_consensus": scalar_receipt,
        "gradient_reduction": reduction_receipt,
        "optimizer_update_authorized": False,
    }
    return ViewScalarResult(
        scalar=scalar_value,
        component_scalars={"external_event_score": scalar_value},
        block_gradients=reduced,
        receipt={**unsigned, "digest": object_sha256(unsigned)},
    )


def save_sample_gradients(
    path: Path,
    *,
    gradients: Mapping[str, Sequence[torch.Tensor]],
    probe: FixedAZeroBProbeBank,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise MotionFisherObservationProbeError("gradient artifact path must be fresh")
    if set(gradients) != set(TRANSFORM_ORDER):
        raise MotionFisherObservationProbeError("gradient transform closure differs")
    tensors: dict[str, torch.Tensor] = {}
    rows: dict[str, Mapping[str, Any]] = {}
    for transform in TRANSFORM_ORDER:
        values = gradients[transform]
        if len(values) != 30:
            raise MotionFisherObservationProbeError("gradient block closure differs")
        for block, value in zip(range(30), values):
            if (
                value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.shape != (1536, 8)
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
            ):
                raise MotionFisherObservationProbeError("stored gradient differs")
            key = GRADIENT_KEY_TEMPLATE.format(block=block, transform=transform)
            tensor = value.contiguous()
            tensors[key] = tensor
            coordinate = probe.coordinate_receipt(block)
            rows[key] = {
                "tensor_key": key,
                "block_index": block,
                "transform": transform,
                "shape": [1536, 8],
                "dtype": "float32",
                "tensor_sha256": tensor_sha256(tensor),
                "coordinate_digest": coordinate["coordinate_digest"],
            }
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".safetensors", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(
            tensors,
            str(temporary),
            metadata={
                "schema_version": SAMPLE_RECEIPT_SCHEMA,
                "payload": "detached_cpu_fp32_lora_b_gradients_only",
                "contains_t2v_pixel_latent_noise_velocity_or_hidden_state": "false",
                "optimizer_update_authorized": "false",
            },
        )
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if set(opened.keys()) != set(tensors):
                raise MotionFisherObservationProbeError("gradient safetensors keys differ")
            metadata = dict(opened.metadata() or {})
            for key, expected in tensors.items():
                actual = opened.get_tensor(key).contiguous()
                if actual.dtype != torch.float32 or not torch.equal(actual, expected):
                    raise MotionFisherObservationProbeError("gradient round trip differs")
        if metadata.get("contains_t2v_pixel_latent_noise_velocity_or_hidden_state") != "false":
            raise MotionFisherObservationProbeError("gradient metadata closure differs")
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    artifact = {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "tensor_count": len(tensors),
        "payload": "detached_cpu_fp32_lora_b_gradients_only",
        "contains_t2v_pixel_latent_noise_velocity_or_hidden_state": False,
        "optimizer_update_authorized": False,
    }
    return {**artifact, "artifact_digest": object_sha256(artifact)}, rows


def load_motion_fisher_observations(
    master_receipt_path: str | Path,
    *,
    allow_negative_control: bool = False,
) -> tuple[motion_fisher.MotionGradientObservation, ...]:
    """Load only detached gradients from a completed eight-clip master receipt."""

    from safetensors import safe_open

    master_path = _plain_file(master_receipt_path, label="motion-fisher master receipt")
    master = _read_json(master_path, label="motion-fisher master receipt")
    _verify_embedded(master, field="receipt_digest", label="motion-fisher master")
    if (
        master.get("schema_version") != MASTER_RECEIPT_SCHEMA
        or master.get("sample_count") != 8
        or master.get("optimizer_update_authorized") is not False
        or not isinstance(master.get("sample_receipts"), list)
    ):
        raise MotionFisherObservationProbeError("master receipt closure differs")
    if master.get("scalar_backend_id") == BUILTIN_BACKEND_ID and not allow_negative_control:
        raise MotionFisherObservationProbeError(
            "built-in flow observation is a negative control; explicit diagnostic opt-in required"
        )
    observations: list[motion_fisher.MotionGradientObservation] = []
    for binding in master["sample_receipts"]:
        sample_path = _plain_file(binding["path"], label="sample receipt")
        if file_sha256(sample_path) != binding["file_sha256"]:
            raise MotionFisherObservationProbeError("sample receipt file hash differs")
        sample = _read_json(sample_path, label="sample receipt")
        _verify_embedded(sample, field="receipt_digest", label="sample receipt")
        if (
            sample.get("schema_version") != SAMPLE_RECEIPT_SCHEMA
            or sample.get("receipt_digest") != binding["receipt_digest"]
            or sample.get("scalar_backend_id") != master.get("scalar_backend_id")
            or sample.get("event_qualified") is not True
            or sample.get("optimizer_update_authorized") is not False
        ):
            raise MotionFisherObservationProbeError("sample receipt closure differs")
        artifact = sample["gradient_artifact"]
        tensor_path = _plain_file(artifact["path"], label="gradient artifact")
        if file_sha256(tensor_path) != artifact["file_sha256"]:
            raise MotionFisherObservationProbeError("gradient artifact hash differs")
        registry = sample["gradient_registry"]
        with safe_open(str(tensor_path), framework="pt", device="cpu") as opened:
            if set(opened.keys()) != set(registry):
                raise MotionFisherObservationProbeError("gradient key registry differs")
            for key in sorted(registry):
                row = registry[key]
                value = opened.get_tensor(key).float().contiguous()
                if (
                    tensor_sha256(value) != row["tensor_sha256"]
                    or list(value.shape) != row["shape"]
                ):
                    raise MotionFisherObservationProbeError("gradient tensor identity differs")
                observations.append(
                    motion_fisher.MotionGradientObservation(
                        block_index=row["block_index"],
                        action_family=sample["action_family"],
                        identity_key=sample["identity_key"],
                        seed_key=sample["seed_key"],
                        transform=row["transform"],
                        coordinate_digest=row["coordinate_digest"],
                        event_receipt_digest=sample["event_receipt_digest"],
                        event_qualified=True,
                        origin=motion_fisher.OBSERVATION_ORIGIN,
                        gradient=value.reshape(-1).detach().contiguous(),
                    )
                )
    expected = 8 * 30 * len(TRANSFORM_ORDER)
    if len(observations) != expected:
        raise MotionFisherObservationProbeError("observation registry is incomplete")
    return tuple(observations)


def _fresh_directory(value: str | Path) -> Path:
    requested = Path(value)
    if not requested.is_absolute() or requested == Path("/"):
        raise MotionFisherObservationProbeError("output directory must be absolute non-root")
    parent = requested.parent.resolve(strict=True)
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise MotionFisherObservationProbeError("output directory must be fresh")
    return output


def _load_authenticated_inputs(args: argparse.Namespace) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    list[dict[str, Any]],
    PopulationSelection,
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    registration = load_registration(args.registration, args.expected_registration_sha256)
    try:
        spec, bank, rows = frozen_runtime.load_group_bank(
            root_spec=args.root_spec,
            root_spec_sha256=args.expected_root_spec_sha256,
            bank_output_dir=args.bank_output_dir,
            bank_receipt=args.bank_receipt,
            bank_receipt_sha256=args.expected_bank_receipt_sha256,
            group_id=args.group_id,
        )
    except frozen_runtime.PairV5T2VEnergyScoringError as error:
        raise MotionFisherObservationProbeError(str(error)) from error
    topup, topup_spec_rows = load_topup_spec(
        args.topup_spec,
        args.expected_topup_spec_sha256,
        base_spec=spec,
        expected_base_spec_sha256=args.expected_root_spec_sha256,
    )
    topup_rows = [
        *load_topup_group(
            rows=topup_spec_rows,
            topup_spec_sha256=args.expected_topup_spec_sha256,
            topup_output_dir=args.topup_output_dir,
            group_id="sp4-a",
        ),
        *load_topup_group(
            rows=topup_spec_rows,
            topup_spec_sha256=args.expected_topup_spec_sha256,
            topup_output_dir=args.topup_output_dir,
            group_id="sp4-b",
        ),
    ]
    selection = validate_population_coverage(
        spec,
        registration,
        topup_rows=topup_spec_rows,
        topup_spec_sha256=args.expected_topup_spec_sha256,
    )
    event_path = _plain_file(args.event_index, label="event index")
    expected_event_sha = _sha256(
        args.expected_event_index_sha256, label="event index file SHA-256"
    )
    if file_sha256(event_path) != expected_event_sha:
        raise MotionFisherObservationProbeError("event index file SHA-256 differs")
    event_index = _read_json(event_path, label="event index")
    events = validate_event_index(
        event_index,
        selection=selection,
        population_digest=selection.population_digest,
        bank=bank,
        topup_rows=topup_rows,
    )
    group_topups = [
        row
        for row in topup_rows
        if selection.sample_metadata[row["candidate"]["candidate_id"]]["group_id"]
        == args.group_id
    ]
    return (
        spec,
        bank,
        [*rows, *group_topups],
        selection,
        events,
        registration,
        topup,
    )


def preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    spec, bank, rows, selection, events, registration, topup = (
        _load_authenticated_inputs(args)
    )
    action_rows = [row for row in rows if row["candidate"]["semantic_branch"] == "action"]
    expected = set(selection.group_candidate_ids[args.group_id])
    if {row["candidate"]["candidate_id"] for row in action_rows} != expected:
        raise MotionFisherObservationProbeError("group action population differs")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "mode": "preflight",
        "group_id": args.group_id,
        "registration_sha256": registration["file_sha256"],
        "root_spec_sha256": args.expected_root_spec_sha256,
        "topup_spec_sha256": topup["file_sha256"],
        "population_digest": selection.population_digest,
        "bank_receipt_digest": bank["receipt_digest"],
        "bank_receipt_file_sha256": bank["file_sha256"],
        "event_index_receipt_digest": _sha256(
            _read_json(_plain_file(args.event_index, label="event index"), label="event index")[
                "receipt_digest"
            ],
            label="event index receipt digest",
        ),
        "selected_action_candidate_ids": list(selection.candidate_ids),
        "group_action_candidate_ids": list(selection.group_candidate_ids[args.group_id]),
        "event_qualified_candidate_ids": sorted(events),
        "two_families_two_identities_two_generation_seeds": True,
        "all_event_receipts_verified_before_gpu_compute": True,
        "gpu_compute_performed": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_update_authorized": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def _normal_prompt_conditions(
    renderer: nn.Module,
    tokenizer: Any,
    prompt_by_branch: Mapping[str, str],
    *,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    encoded = frozen_runtime._encode_prompt_bank(
        renderer, tokenizer, prompt_by_branch, device=device
    )
    result = {}
    for branch, value in encoded.items():
        normal = value.clone() if torch.is_inference(value) else value
        if normal.requires_grad or normal.grad_fn is not None or torch.is_inference(normal):
            raise MotionFisherObservationProbeError("prompt tensor cannot enter backward")
        result[branch] = normal
    return result


def _base_freeze_certificate(renderer: nn.Module) -> Mapping[str, Any]:
    certificate = native_generation.source_audit.model_freeze_certificate(renderer)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in renderer.parameters()):
        raise MotionFisherObservationProbeError("base Bernini is not frozen")
    return certificate


def run_group_probe(args: argparse.Namespace) -> int:
    if args.ack_observation_only is not True:
        raise MotionFisherObservationProbeError("--ack-observation-only is mandatory")
    if args.scalar_backend == BUILTIN_BACKEND_ID and args.ack_negative_control is not True:
        raise MotionFisherObservationProbeError("negative-control acknowledgement is mandatory")
    topology = world4_sp4_contract()
    output = _fresh_directory(args.output_dir)
    spec, bank, rows, selection, events, registration, topup = (
        _load_authenticated_inputs(args)
    )
    action_by_id = {
        row["candidate"]["candidate_id"]: row
        for row in rows
        if row["candidate"]["semantic_branch"] == "action"
    }
    expected_group_ids = set(selection.group_candidate_ids[args.group_id])
    if set(action_by_id) != expected_group_ids:
        raise MotionFisherObservationProbeError("group action rows differ")
    rows_by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_cell.setdefault(row["candidate"]["calibration_group_id"], []).append(row)
    base_action_by_id = {
        row["candidate"]["candidate_id"]: row
        for row in rows
        if row["candidate"]["semantic_branch"] == "action"
        and row.get("base_action_candidate_id") is None
    }

    legacy = native_generation.legacy
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(args.checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise MotionFisherObservationProbeError(str(error)) from error
    if int(transformer_config.get("num_layers", -1)) != 30:
        raise MotionFisherObservationProbeError("checkpoint transformer layer count differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch.distributed as dist
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise MotionFisherObservationProbeError("AUH ROCm GPU is required")
    if torch.cuda.device_count() != WORLD_SIZE:
        raise MotionFisherObservationProbeError("visible accelerator count differs")
    torch.cuda.set_device(topology.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=topology.rank,
        world_size=topology.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", topology.local_rank)

    checkpoint_rows: list[Any] = [None]
    if topology.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": native_generation.source_audit.validate_checkpoint_content(
                    checkpoint, Path(args.checkpoint_content_manifest)
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
        raise MotionFisherObservationProbeError(
            f"checkpoint audit failed: {checkpoint_rows[0]}"
        )
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise MotionFisherObservationProbeError("probe requires transformer_1 only")
    transformer.gradient_checkpointing = True
    freeze_before = _base_freeze_certificate(renderer)
    checkpoint_binding = frozen_runtime.checkpoint_content_binding(
        checkpoint_identity, freeze_before
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    probe = FixedAZeroBProbeBank(transformer).to(device)
    probe.assert_zero_clean()

    external_backend: Optional[ExternalEventCriticBackend] = None
    if args.scalar_backend == EXTERNAL_BACKEND_ID:
        external_backend = load_external_event_critic_backend(
            args.external_backend_implementation,
            args.expected_external_backend_sha256,
            device=device,
        )
    elif args.scalar_backend != BUILTIN_BACKEND_ID:
        raise MotionFisherObservationProbeError("scalar backend differs")

    sample_bindings: list[Mapping[str, Any]] = []
    output.mkdir(parents=False, exist_ok=False) if topology.rank == 0 else None
    dist.barrier()
    with probe.installed():
        for candidate_id in selection.group_candidate_ids[args.group_id]:
            row = action_by_id[candidate_id]
            candidate = row["candidate"]
            prompt_action_id = selection.sample_metadata[candidate_id][
                "prompt_action_candidate_id"
            ]
            prompt_action_row = base_action_by_id.get(prompt_action_id)
            if prompt_action_row is None:
                raise MotionFisherObservationProbeError(
                    "top-up sample lost its base prompt-cell action binding"
                )
            cell_rows = rows_by_cell[
                prompt_action_row["candidate"]["calibration_group_id"]
            ]
            if [item["candidate"]["semantic_branch"] for item in cell_rows] != list(
                mace.BRANCH_ORDER
            ):
                raise MotionFisherObservationProbeError("prompt cell branch order differs")
            prompt_by_branch = frozen_runtime.prompt_bank_from_cell(
                cell_rows,
                task_prompt_builder=lambda caption: native_generation.build_task_prompt(
                    "t2v", caption, prompt_cleaner=prompt_clean
                ),
            )
            caption_by_branch = {
                item["candidate"]["semantic_branch"]: item["candidate"]["full_t2v_caption"]
                for item in cell_rows
            }
            conditions = _normal_prompt_conditions(
                renderer, tokenizer, prompt_by_branch, device=device
            )
            clean_cpu = frozen_runtime._load_exact81_tensor(
                row["artifacts"]["predecode_clean_latent"],
                key="normalized_clean_latent",
                label=f"{candidate_id} action clean latent",
            )
            gaussian_cpu = frozen_runtime._load_exact81_tensor(
                row["artifacts"]["official_initial_gaussian"],
                key="official_initial_gaussian",
                label=f"{candidate_id} official Gaussian",
            )
            frozen_runtime.verify_native_tensor_value_identity(
                clean_cpu,
                row["artifacts"]["predecode_clean_latent"],
                label=f"{candidate_id} action clean latent",
            )
            frozen_runtime.verify_native_tensor_value_identity(
                gaussian_cpu,
                row["artifacts"]["official_initial_gaussian"],
                label=f"{candidate_id} official Gaussian",
            )
            clean = clean_cpu.to(device=device).contiguous()
            gaussian = gaussian_cpu.to(device=device).contiguous()
            input_identity = {
                "clean": frozen_runtime.native_tensor_value_identity(clean_cpu),
                "official_gaussian": frozen_runtime.native_tensor_value_identity(gaussian_cpu),
            }
            del clean_cpu, gaussian_cpu

            gradients: dict[str, Sequence[torch.Tensor]] = {}
            scalar_receipts: dict[str, Mapping[str, Any]] = {}
            scalar_values: dict[str, float] = {}
            for transform in TRANSFORM_ORDER:
                clean_view = apply_registered_transform(clean, transform)
                if external_backend is None:
                    result = builtin_negative_control_vjp(
                        transform=transform,
                        diffusion=diffusion,
                        transformer=transformer,
                        probe=probe,
                        clean_view=clean_view,
                        official_gaussian=gaussian,
                        condition_by_branch=conditions,
                        sigma=float(registration["native_schedule"]["sigma"]),
                        native_timestep=int(
                            registration["native_schedule"]["native_timestep"]
                        ),
                        distributed_module=dist,
                    )
                else:
                    result = external_event_critic_vjp(
                        backend=external_backend,
                        transform=transform,
                        diffusion=diffusion,
                        transformer=transformer,
                        probe=probe,
                        clean_view=clean_view,
                        official_gaussian=gaussian,
                        condition_by_branch=conditions,
                        caption_by_branch=caption_by_branch,
                        metadata=selection.sample_metadata[candidate_id],
                        sigma=float(registration["native_schedule"]["sigma"]),
                        native_timestep=int(
                            registration["native_schedule"]["native_timestep"]
                        ),
                        distributed_module=dist,
                    )
                gradients[transform] = result.block_gradients
                scalar_receipts[transform] = result.receipt
                scalar_values[transform] = result.scalar
                del clean_view, result

            if topology.rank == 0:
                sample_dir = output / candidate_id
                sample_dir.mkdir(parents=False, exist_ok=False)
                artifact, gradient_registry = save_sample_gradients(
                    sample_dir / GRADIENT_FILENAME,
                    gradients=gradients,
                    probe=probe,
                )
                event = events[candidate_id]
                meta = selection.sample_metadata[candidate_id]
                unsigned_sample = {
                    "schema_version": SAMPLE_RECEIPT_SCHEMA,
                    "candidate_id": candidate_id,
                    "action_family": meta["action_family"],
                    "identity_key": meta["identity_key"],
                    "seed_key": meta["seed_key"],
                    "generation_seed": meta["seed"],
                    "group_id": args.group_id,
                    "event_qualified": True,
                    "event_receipt_path": event["path"],
                    "event_receipt_file_sha256": event["file_sha256"],
                    "event_receipt_digest": event["receipt"]["receipt_digest"],
                    "generation_receipt_digest": row["generation_receipt_digest"],
                    "registration_sha256": registration["file_sha256"],
                    "root_spec_sha256": args.expected_root_spec_sha256,
                    "topup_spec_sha256": topup["file_sha256"],
                    "population_digest": selection.population_digest,
                    "bank_receipt_digest": bank["receipt_digest"],
                    "checkpoint_binding_digest": checkpoint_binding["binding_digest"],
                    "scalar_backend_id": args.scalar_backend,
                    "scalar_values": scalar_values,
                    "scalar_receipts": scalar_receipts,
                    "coordinate_receipts": [
                        probe.coordinate_receipt(block) for block in range(30)
                    ],
                    "gradient_artifact": artifact,
                    "gradient_registry": gradient_registry,
                    "authenticated_input_identity": input_identity,
                    "input_use_boundary": {
                        "t2v_action_latent_and_gaussian_consumed_only_inside_frozen_probe": True,
                        "t2v_pixel_latent_noise_velocity_or_hidden_state_published": False,
                        "rv2v_condition_target_donor_or_noise": False,
                        "generated_media_as_training_target": False,
                    },
                    "raw_parameter_gradient_no_manual_sign_flip": True,
                    "base_parameter_update_performed": False,
                    "probe_parameter_update_performed": False,
                    "training_performed": False,
                    "optimizer_created": False,
                    "optimizer_update_authorized": False,
                }
                sample_receipt = {
                    **unsigned_sample,
                    "receipt_digest": object_sha256(unsigned_sample),
                }
                sample_path = sample_dir / SAMPLE_RECEIPT_FILENAME
                sample_file_sha = _write_create_only_json(sample_path, sample_receipt)
                sample_bindings.append(
                    {
                        "candidate_id": candidate_id,
                        "path": str(sample_path),
                        "file_sha256": sample_file_sha,
                        "receipt_digest": sample_receipt["receipt_digest"],
                    }
                )
            del gradients, scalar_receipts, scalar_values, conditions, clean, gaussian
            torch.cuda.empty_cache()

    freeze_after = _base_freeze_certificate(renderer)
    if freeze_before != freeze_after:
        raise MotionFisherObservationProbeError("base freeze certificate changed")
    probe.assert_zero_clean()
    if topology.rank == 0:
        unsigned_group = {
            "schema_version": GROUP_RECEIPT_SCHEMA,
            "group_id": args.group_id,
            "world_size": WORLD_SIZE,
            "ulysses_size": SP_SIZE,
            "sample_count": len(sample_bindings),
            "sample_receipts": sample_bindings,
            "scalar_backend_id": args.scalar_backend,
            "registration_sha256": registration["file_sha256"],
            "root_spec_sha256": args.expected_root_spec_sha256,
            "topup_spec_sha256": topup["file_sha256"],
            "population_digest": selection.population_digest,
            "bank_receipt_digest": bank["receipt_digest"],
            "checkpoint_binding_digest": checkpoint_binding["binding_digest"],
            "freeze_certificate_unchanged": True,
            "gradient_checkpointing_observation_only": True,
            "base_parameter_update_performed": False,
            "probe_parameter_update_performed": False,
            "training_performed": False,
            "optimizer_created": False,
            "optimizer_update_authorized": False,
            "bernini_revision": _sha1(bernini_revision, label="Bernini revision"),
            "veomni_revision": _sha1(veomni_revision, label="VeOmni revision"),
        }
        group_receipt = {**unsigned_group, "receipt_digest": object_sha256(unsigned_group)}
        _write_create_only_json(output / GROUP_RECEIPT_FILENAME, group_receipt)
    dist.barrier()
    dist.destroy_process_group()
    return 0


def aggregate(args: argparse.Namespace) -> Mapping[str, Any]:
    output = _fresh_directory(args.output_dir)
    groups = []
    sample_bindings: list[Mapping[str, Any]] = []
    backend_ids: set[str] = set()
    for group_id, raw_dir in (("sp4-a", args.sp4_a_dir), ("sp4-b", args.sp4_b_dir)):
        directory = _plain_directory(raw_dir, label=f"{group_id} output")
        path = _plain_file(directory / GROUP_RECEIPT_FILENAME, label=f"{group_id} receipt")
        receipt = _read_json(path, label=f"{group_id} receipt")
        _verify_embedded(receipt, field="receipt_digest", label=f"{group_id} receipt")
        if (
            receipt.get("schema_version") != GROUP_RECEIPT_SCHEMA
            or receipt.get("group_id") != group_id
            or receipt.get("sample_count") != 4
            or receipt.get("optimizer_update_authorized") is not False
            or not isinstance(receipt.get("sample_receipts"), list)
        ):
            raise MotionFisherObservationProbeError(f"{group_id} receipt closure differs")
        groups.append(
            {
                "group_id": group_id,
                "path": str(path),
                "file_sha256": file_sha256(path),
                "receipt_digest": receipt["receipt_digest"],
            }
        )
        sample_bindings.extend(receipt["sample_receipts"])
        backend_ids.add(receipt["scalar_backend_id"])
    if len(backend_ids) != 1 or len(sample_bindings) != 8 or len(
        {item["candidate_id"] for item in sample_bindings}
    ) != 8:
        raise MotionFisherObservationProbeError("aggregate sample/backend closure differs")
    backend_id = next(iter(backend_ids))
    output.mkdir(parents=False, exist_ok=False)
    unsigned = {
        "schema_version": MASTER_RECEIPT_SCHEMA,
        "group_receipts": groups,
        "sample_count": 8,
        "sample_receipts": sorted(sample_bindings, key=lambda item: item["candidate_id"]),
        "scalar_backend_id": backend_id,
        "built_in_flow_backend_is_negative_control": backend_id == BUILTIN_BACKEND_ID,
        "payload": "detached_cpu_fp32_lora_b_gradients_only",
        "t2v_pixel_latent_noise_velocity_or_hidden_state_published": False,
        "raw_parameter_gradient_no_manual_sign_flip": True,
        "motion_fisher_fit_authorized_by_probe": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_update_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_create_only_json(output / MASTER_RECEIPT_FILENAME, receipt)
    return receipt


def _add_common_bank_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registration", default=str(REGISTRATION_PATH))
    parser.add_argument(
        "--expected-registration-sha256", default=REGISTRATION_SHA256
    )
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--bank-output-dir", required=True)
    parser.add_argument("--bank-receipt", required=True)
    parser.add_argument("--expected-bank-receipt-sha256", required=True)
    parser.add_argument("--topup-spec", default=str(TOPUP_SPEC_PATH))
    parser.add_argument(
        "--expected-topup-spec-sha256", default=TOPUP_SPEC_SHA256
    )
    parser.add_argument("--topup-output-dir", required=True)
    parser.add_argument("--event-index", required=True)
    parser.add_argument("--expected-event-index-sha256", required=True)
    parser.add_argument("--group-id", choices=("sp4-a", "sp4-b"), required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    topup_parser = subparsers.add_parser("materialize-topup-plan")
    topup_parser.add_argument("--topup-spec", default=str(TOPUP_SPEC_PATH))
    topup_parser.add_argument(
        "--expected-topup-spec-sha256", default=TOPUP_SPEC_SHA256
    )
    topup_parser.add_argument("--root-spec", required=True)
    topup_parser.add_argument("--expected-root-spec-sha256", required=True)
    topup_parser.add_argument("--output-dir", required=True)
    audit_topup_parser = subparsers.add_parser("audit-topup")
    audit_topup_parser.add_argument("--topup-spec", default=str(TOPUP_SPEC_PATH))
    audit_topup_parser.add_argument(
        "--expected-topup-spec-sha256", default=TOPUP_SPEC_SHA256
    )
    audit_topup_parser.add_argument("--root-spec", required=True)
    audit_topup_parser.add_argument("--expected-root-spec-sha256", required=True)
    audit_topup_parser.add_argument("--topup-output-dir", required=True)
    event_parser = subparsers.add_parser("materialize-event-index")
    event_parser.add_argument("--registration", default=str(REGISTRATION_PATH))
    event_parser.add_argument(
        "--expected-registration-sha256", default=REGISTRATION_SHA256
    )
    event_parser.add_argument("--root-spec", required=True)
    event_parser.add_argument("--expected-root-spec-sha256", required=True)
    event_parser.add_argument("--bank-output-dir", required=True)
    event_parser.add_argument("--bank-receipt", required=True)
    event_parser.add_argument("--expected-bank-receipt-sha256", required=True)
    event_parser.add_argument("--topup-spec", default=str(TOPUP_SPEC_PATH))
    event_parser.add_argument(
        "--expected-topup-spec-sha256", default=TOPUP_SPEC_SHA256
    )
    event_parser.add_argument("--topup-output-dir", required=True)
    event_parser.add_argument(
        "--event-receipt", action="append", required=True,
        help="Repeat exactly eight times; order is irrelevant.",
    )
    event_parser.add_argument("--output", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _add_common_bank_arguments(preflight_parser)

    probe_parser = subparsers.add_parser("probe")
    _add_common_bank_arguments(probe_parser)
    probe_parser.add_argument("--bernini-root", required=True)
    probe_parser.add_argument("--veomni-root", required=True)
    probe_parser.add_argument("--checkpoint", required=True)
    probe_parser.add_argument("--checkpoint-content-manifest", required=True)
    probe_parser.add_argument("--output-dir", required=True)
    probe_parser.add_argument(
        "--expected-bernini-commit",
        default=native_generation.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    probe_parser.add_argument(
        "--expected-veomni-commit",
        default=native_generation.legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    probe_parser.add_argument(
        "--scalar-backend",
        choices=(BUILTIN_BACKEND_ID, EXTERNAL_BACKEND_ID),
        default=BUILTIN_BACKEND_ID,
    )
    probe_parser.add_argument("--external-backend-implementation")
    probe_parser.add_argument("--expected-external-backend-sha256")
    probe_parser.add_argument("--ack-observation-only", action="store_true")
    probe_parser.add_argument("--ack-negative-control", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--sp4-a-dir", required=True)
    aggregate_parser.add_argument("--sp4-b-dir", required=True)
    aggregate_parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize-topup-plan":
        plan = materialize_topup_plan(
            topup_spec_path=args.topup_spec,
            expected_topup_spec_sha256=args.expected_topup_spec_sha256,
            base_spec_path=args.root_spec,
            expected_base_spec_sha256=args.expected_root_spec_sha256,
            output_dir=args.output_dir,
        )
        print(canonical_json_bytes(plan).decode("ascii"), flush=True)
        return 0
    if args.command == "audit-topup":
        receipt = audit_topup_outputs(
            topup_spec_path=args.topup_spec,
            expected_topup_spec_sha256=args.expected_topup_spec_sha256,
            base_spec_path=args.root_spec,
            expected_base_spec_sha256=args.expected_root_spec_sha256,
            topup_output_dir=args.topup_output_dir,
        )
        print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
        return 0
    if args.command == "materialize-event-index":
        result = materialize_event_index(
            registration_path=args.registration,
            expected_registration_sha256=args.expected_registration_sha256,
            root_spec_path=args.root_spec,
            expected_root_spec_sha256=args.expected_root_spec_sha256,
            bank_output_dir=args.bank_output_dir,
            bank_receipt_path=args.bank_receipt,
            expected_bank_receipt_sha256=args.expected_bank_receipt_sha256,
            topup_spec_path=args.topup_spec,
            expected_topup_spec_sha256=args.expected_topup_spec_sha256,
            topup_output_dir=args.topup_output_dir,
            event_receipt_paths=args.event_receipt,
            output_path=args.output,
        )
        print(canonical_json_bytes(result).decode("ascii"), flush=True)
        return 0
    if args.command == "preflight":
        print(canonical_json_bytes(preflight(args)).decode("ascii"), flush=True)
        return 0
    if args.command == "probe":
        if args.scalar_backend == EXTERNAL_BACKEND_ID and (
            not args.external_backend_implementation
            or not args.expected_external_backend_sha256
        ):
            raise MotionFisherObservationProbeError(
                "external backend requires a hash-pinned implementation"
            )
        return run_group_probe(args)
    if args.command == "aggregate":
        print(canonical_json_bytes(aggregate(args)).decode("ascii"), flush=True)
        return 0
    raise MotionFisherObservationProbeError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUILTIN_BACKEND_ID",
    "EVENT_INDEX_SCHEMA",
    "EXTERNAL_BACKEND_ID",
    "FixedAZeroBProbeBank",
    "MASTER_RECEIPT_FILENAME",
    "MotionFisherObservationProbeError",
    "PopulationSelection",
    "REGISTRATION_PATH",
    "REGISTRATION_SHA256",
    "TOPUP_SPEC_PATH",
    "TOPUP_SPEC_SHA256",
    "TRANSFORM_ORDER",
    "aggregate",
    "audit_topup_outputs",
    "apply_registered_transform",
    "builtin_negative_control_vjp",
    "canonical_json_bytes",
    "file_sha256",
    "load_external_event_critic_backend",
    "load_motion_fisher_observations",
    "load_registration",
    "load_topup_group",
    "load_topup_spec",
    "make_fixed_orthogonal_a",
    "materialize_event_index",
    "materialize_topup_plan",
    "object_sha256",
    "seal_event_index",
    "tensor_sha256",
    "validate_event_index",
    "validate_population_coverage",
    "world4_sp4_contract",
]

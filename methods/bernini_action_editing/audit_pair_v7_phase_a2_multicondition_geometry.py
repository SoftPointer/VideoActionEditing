#!/usr/bin/env python3
"""WORLD8 read-only PAIR-v7 Phase-A2 multicondition geometry runtime.

The prospective multicondition preregistration intentionally carries no
runtime authority because it does not inspect CAST-v4.  This executable does
not reinterpret that boundary.  Its preflight first revalidates the sealed
preregistration and the complete two-group / forty-child CAST-v4 bank, then
binds the four fixed action receipts to the exact source media, clean latent,
official Gaussian, prompt bank, split, family, and checkpoint.  That combined
receipt authorizes read-only gradient measurement only.

One WORLD8 job uses DP2 x Ulysses-SP4.  The fit and confirmation source pairs
are visited in fixed order and each is measured at exact40 indices 16 and 35.
After each SP4 average and DP2 exchange, every rank owns the same eight action
gradients and sixty-four unprojected identity probes.  Their byte-bound input
manifest must reach WORLD consensus before global rank zero copies the bank to
CPU and calls ``pair_v7_multicondition_nullspace_transport`` exactly once.
Only sealed receipts are broadcast or persisted; raw gradients and the safe
direction remain ephemeral.

There is no optimizer, parameter update, decode, mask, flow, pose, track,
trajectory, candidate selection, or action-editing success claim in this
program.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Callable, Mapping, Optional, Sequence

import pair_v7_multicondition_geometry_authority as preregistration


METHOD_NAME = "bernini-pair-v7-phase-a2-multicondition-read-only-geometry"
RUN_RECEIPT_SCHEMA = "bernini-pair-v7-phase-a2-multicondition-audit-v1"
BANK_BINDING_SCHEMA = "bernini-pair-v7-phase-a2-live-cast-bank-binding-v1"
WORLD_SOLVER_AUTHORITY_SCHEMA = (
    "bernini-pair-v7-phase-a2-world-root-cpu-solver-authority-v1"
)
WORLD_INPUT_SCHEMA = "bernini-pair-v7-phase-a2-world-gradient-bank-input-v1"
ACTION_QUERY_SCHEMA = "bernini-pair-v7-phase-a2-action-query-v1"
ACTION_OBJECTIVE_SCHEMA = "bernini-pair-v7-phase-a2-measurement-objective-v1"
ACTION_GRADIENT_SCHEMA = "bernini-pair-v7-phase-a2-action-gradient-v1"
IDENTITY_PROTOCOL_SCHEMA = (
    "bernini-pair-v7-phase-a2-identity-deployment-protocol-v1"
)

FRAME_COUNT = 81
FPS = 25.0
DP_SIZE = 2
SP_SIZE = 4
WORLD_SIZE = 8
PAIR_IDS = ("fit", "confirmation")
SCHEDULE_INDICES = (16, 35)
IDENTITY_FAMILIES = ("deploy_camera_delta", "deploy_noop_identity")
SKETCH_COUNT = 4
EXPECTED_ACTION_COUNT = 8
EXPECTED_IDENTITY_COUNT = 64
SOURCE_NOISE_MASTER_SEED = 20260808
DEPLOYMENT_FLOW_SHIFT = 5.0

PHASE_A2_RUNTIME_ARCHIVE_REQUIRED = frozenset(
    {
        "methods/bernini_action_editing/audit_pair_v7_phase_a2_multicondition_geometry.py",
        "methods/bernini_action_editing/pair_v7_multicondition_geometry_authority.py",
        "methods/bernini_action_editing/pair_v7_multicondition_nullspace_transport.py",
    }
)

BRANCH_ORDER = (
    "action",
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


class PairV7PhaseA2Error(RuntimeError):
    """Raised before an ambiguous measurement, solve, or publication."""


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
        raise PairV7PhaseA2Error("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _same_json_value(left: Any, right: Any) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except PairV7PhaseA2Error:
        return False


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise PairV7PhaseA2Error("receipt is already sealed")
    value = dict(unsigned)
    for field in (
        "global_population_go",
        "optimizer_authorized",
        "parameter_update_authorized",
        "action_success_claimed",
    ):
        if field in value and value[field] is not False:
            raise PairV7PhaseA2Error(f"{field} must remain false")
        value[field] = False
    return {**value, "receipt_digest": object_sha256(value)}


def _check_seal(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(declared, str)
        or _SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
    ):
        raise PairV7PhaseA2Error(f"{label} seal differs")
    return declared


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV7PhaseA2Error(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise PairV7PhaseA2Error(f"{label} must be lowercase SHA-1")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise PairV7PhaseA2Error(f"{label} is unsafe")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_absolute_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise PairV7PhaseA2Error(f"{label} path differs")
    raw = Path(value)
    if not raw.is_absolute():
        raise PairV7PhaseA2Error(f"{label} must be absolute")
    try:
        path = raw.resolve(strict=True)
    except OSError as error:
        raise PairV7PhaseA2Error(f"{label} is absent") from error
    if path != raw or not path.is_file() or path.is_symlink():
        raise PairV7PhaseA2Error(f"{label} must be a canonical plain file")
    return path


def _closed_prompt_bank(value: Any, *, label: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(BRANCH_ORDER)
        or any(
            not isinstance(value[name], str)
            or not value[name].strip()
            or "\x00" in value[name]
            for name in BRANCH_ORDER
        )
    ):
        raise PairV7PhaseA2Error(f"{label} branch closure differs")
    result = {name: value[name] for name in BRANCH_ORDER}
    if len(set(result.values())) != len(result):
        raise PairV7PhaseA2Error(f"{label} contains duplicate branches")
    return result


def _candidate_projection_from_child(
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Rebind one validator projection to the score-receipt bytes themselves."""

    path = _plain_absolute_file(candidate.get("path"), label="CAST child receipt")
    file_sha = _sha256(candidate.get("file_sha256"), label="CAST child file")
    if _file_sha256(path) != file_sha:
        raise PairV7PhaseA2Error("CAST child bytes differ after validation")
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes.decode("ascii", errors="strict"))
    except Exception as error:
        raise PairV7PhaseA2Error("CAST child must remain strict ASCII JSON") from error
    if not isinstance(raw, Mapping):
        raise PairV7PhaseA2Error("CAST child receipt is not an object")
    unsigned = dict(raw)
    declared = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(declared, str)
        or _SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
        or candidate.get("receipt_digest") != declared
        or raw.get("scientific_action_editing_claim") is not False
    ):
        raise PairV7PhaseA2Error("CAST child embedded receipt binding differs")
    packet = raw.get("frozen_t2v_packet_binding")
    child_projection = {
        "candidate_id": raw.get("candidate_id"),
        "receipt_digest": declared,
        "analysis_split": raw.get("analysis_split"),
        "action_family_id": raw.get("action_family_id"),
        "semantic_branch": raw.get("semantic_branch"),
        "root_spec_raw_sha256": raw.get("root_spec_raw_sha256"),
        "frozen_checkpoint_receipt_digest": raw.get(
            "frozen_checkpoint_receipt_digest"
        ),
        "checkpoint_content_binding": raw.get("checkpoint_content_binding"),
        "geometry_source_video_sha256": raw.get(
            "geometry_source_video_sha256"
        ),
        "full_t2v_caption_by_branch": raw.get(
            "full_t2v_caption_by_branch"
        ),
        "clean_latent_tensor_sha256": raw.get("clean_latent_tensor_sha256"),
        "official_gaussian_tensor_sha256": raw.get(
            "official_gaussian_tensor_sha256"
        ),
        "prompt_by_branch": raw.get("prompt_by_branch"),
        "candidate_shape": (
            packet.get("candidate_shape") if isinstance(packet, Mapping) else None
        ),
        "raw_global_action_energy_score": raw.get(
            "raw_global_action_energy_score"
        ),
    }
    for field, expected in child_projection.items():
        if not _same_json_value(candidate.get(field), expected):
            raise PairV7PhaseA2Error(
                f"CAST child validator projection differs: {field}"
            )
    return dict(candidate)


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str

    def assert_unchanged(self) -> None:
        if (
            not self.path.is_file()
            or self.path.is_symlink()
            or _file_sha256(self.path) != self.sha256
        ):
            raise PairV7PhaseA2Error(f"bound file changed: {self.path}")


@dataclass(frozen=True)
class PhaseA2EventSpec:
    event_id: str
    pair_id: str
    dp_arm: int
    action_family: str
    prompt_by_branch: Mapping[str, str]
    prompt_bank_sha256: str
    raw_caption_by_branch: Mapping[str, str]
    raw_caption_bank_sha256: str
    source_sample_id: str
    generation_seed: int
    source_noise_key_sha256: str
    source_video: FileBinding
    clean_latent: FileBinding
    clean_latent_tensor_key: str
    clean_latent_tensor_sha256: str
    official_gaussian: FileBinding
    official_gaussian_tensor_key: str
    official_gaussian_tensor_sha256: str
    latent_shape: tuple[int, ...]
    event_digest: str
    cast_score_receipt: FileBinding
    cast_score_receipt_digest: str
    raw_global_action_energy_score: float

    def assert_unchanged(self) -> None:
        self.source_video.assert_unchanged()
        self.clean_latent.assert_unchanged()
        self.official_gaussian.assert_unchanged()
        self.cast_score_receipt.assert_unchanged()


@dataclass(frozen=True)
class PhaseA2RuntimeEvent:
    spec: PhaseA2EventSpec
    event_latent_cpu: Any
    official_epsilon_cpu: Any


@dataclass(frozen=True)
class PhaseA2Manifest:
    path: Path
    raw_sha256: str
    preregistration_digest: str
    checkpoint_tree_sha256: str
    action_adapter_schema_sha256: str
    events: tuple[PhaseA2EventSpec, ...]
    manifest_digest: str

    def assert_unchanged(self) -> None:
        if _file_sha256(self.path) != self.raw_sha256:
            raise PairV7PhaseA2Error("multicondition preregistration changed")
        for event in self.events:
            event.assert_unchanged()


@dataclass(frozen=True)
class PhaseA2BankBinding:
    receipt: Mapping[str, Any]
    external_files: tuple[FileBinding, ...]

    @property
    def measurement_authority_digest(self) -> str:
        return str(self.receipt["receipt_digest"])

    def assert_unchanged(self) -> None:
        for binding in self.external_files:
            binding.assert_unchanged()


@dataclass(frozen=True)
class PhaseA2Preflight:
    manifest: PhaseA2Manifest
    runtime_events: tuple[PhaseA2RuntimeEvent, ...]
    bank_binding: PhaseA2BankBinding
    checkpoint_identity: Mapping[str, Any]
    runtime_archive: FileBinding
    runtime_revision: str
    cast_archive: FileBinding
    cast_revision: str


@dataclass(frozen=True)
class PhaseA2WorldSolve:
    primary_replication_go: bool
    transport_receipt: Mapping[str, Any]
    authority_receipt: Mapping[str, Any]


def bind_plan_events_to_cast_candidates(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_file_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
    cast_group_receipt_digests: Sequence[str],
    cast_method_binding: Mapping[str, Any],
    cast_root_binding: Mapping[str, Any],
    prompt_rebuilder: Callable[[Mapping[str, str]], Mapping[str, str]],
) -> tuple[PhaseA2Manifest, Mapping[str, Any]]:
    """Bind the prospective four-event plan to all forty validated CAST rows."""

    plan_file = _sha256(plan_file_sha256, label="preregistration file")
    canonical_plan_path = _plain_absolute_file(
        plan_path, label="multicondition preregistration"
    )
    if canonical_plan_path != plan_path or _file_sha256(canonical_plan_path) != plan_file:
        raise PairV7PhaseA2Error("preregistration path/bytes differ")
    if (
        plan.get("schema_version") != preregistration.PLAN_SCHEMA
        or plan.get("geometry_measurement_authorized") is not False
        or plan.get("optimizer_authorized") is not False
        or plan.get("parameter_update_authorized") is not False
        or plan.get("primary_schedule_indices") != list(SCHEDULE_INDICES)
        or plan.get("event_count") != 4
        or plan.get("primary_condition_count") != 4
        or plan.get("global_common_direction_spec")
        != preregistration._global_common_direction_spec()
        or plan.get("primary_gate_definition")
        != preregistration._primary_gate_definition()
    ):
        raise PairV7PhaseA2Error("prospective preregistration closure differs")
    prereg_digest = plan.get("preregistration_digest")
    unsigned_plan = dict(plan)
    unsigned_plan.pop("preregistration_digest", None)
    if (
        not isinstance(prereg_digest, str)
        or _SHA256_RE.fullmatch(prereg_digest) is None
        or preregistration.object_sha256(unsigned_plan) != prereg_digest
    ):
        raise PairV7PhaseA2Error("preregistration seal differs")
    supplied_rows = tuple(candidates)
    if (
        len(supplied_rows) != 40
        or any(not isinstance(row, Mapping) for row in supplied_rows)
        or len({row.get("candidate_id") for row in supplied_rows}) != 40
        or len({row.get("receipt_digest") for row in supplied_rows}) != 40
    ):
        raise PairV7PhaseA2Error("CAST bank must contain forty unique child receipts")
    rows = tuple(_candidate_projection_from_child(row) for row in supplied_rows)
    indexed = {str(row["candidate_id"]): row for row in rows}
    event_rows = plan.get("events")
    if not isinstance(event_rows, list) or len(event_rows) != 4:
        raise PairV7PhaseA2Error("preregistration event closure differs")
    specs: list[PhaseA2EventSpec] = []
    selected_receipts: list[Mapping[str, Any]] = []
    for ordinal, event in enumerate(event_rows):
        if not isinstance(event, Mapping):
            raise PairV7PhaseA2Error(f"event[{ordinal}] differs")
        event_id = _safe_id(event.get("event_id"), label="event ID")
        candidate = indexed.get(event_id)
        if candidate is None:
            raise PairV7PhaseA2Error(f"event {event_id} lacks one CAST child")
        pair_id = _safe_id(event.get("pair_wave"), label="pair wave")
        source_id = _safe_id(
            event.get("source_sample_id"), label="source sample ID"
        )
        family = _safe_id(event.get("action_family"), label="action family")
        dp_arm = event.get("dp_arm")
        shape = event.get("latent_shape")
        prompts = _closed_prompt_bank(
            candidate.get("prompt_by_branch"), label="CAST prompt bank"
        )
        captions = _closed_prompt_bank(
            candidate.get("full_t2v_caption_by_branch"),
            label="CAST raw-caption bank",
        )
        rebuilt_prompts = _closed_prompt_bank(
            prompt_rebuilder(captions), label="rebuilt T2V prompt bank"
        )
        if rebuilt_prompts != prompts:
            raise PairV7PhaseA2Error(
                f"CAST prompt/raw-caption semantics differ for event {event_id}"
            )
        # The plan fixes the event ID plus source/artifact bytes; the validated
        # CAST child fixes the two semantic banks.  Keep their values (not only
        # caller-supplied hashes) inside the live authority receipt and later
        # reconstruct prompt_by_branch from the raw captions with the deployed
        # prompt cleaner before any VJP is allowed.
        score = candidate.get("raw_global_action_energy_score")
        generation_seed = event.get("generation_seed")
        source_noise_key = event.get("source_noise_key_sha256")
        if (
            pair_id not in PAIR_IDS
            or event.get("analysis_split") != pair_id
            or type(dp_arm) is not int
            or dp_arm not in (0, 1)
            or type(generation_seed) is not int
            or generation_seed < 0
            or source_noise_key != preregistration._source_noise_key(source_id)
            or not isinstance(shape, list)
            or len(shape) != 5
            or shape[:3] != [1, 16, 21]
            or shape[3] % 2
            or shape[4] % 2
            or candidate.get("analysis_split") != pair_id
            or candidate.get("semantic_branch") != "action"
            or candidate.get("action_family_id") != family
            or candidate.get("clean_latent_tensor_sha256")
            != event.get("clean_latent_tensor_sha256")
            or candidate.get("official_gaussian_tensor_sha256")
            != event.get("official_gaussian_tensor_sha256")
            or candidate.get("geometry_source_video_sha256")
            != event.get("source_video_file_sha256")
            or candidate.get("candidate_shape") != shape
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise PairV7PhaseA2Error(f"CAST binding differs for event {event_id}")
        score_path = _plain_absolute_file(
            candidate.get("path"), label="CAST child receipt"
        )
        score_file_sha = _sha256(
            candidate.get("file_sha256"), label="CAST child file"
        )
        score_digest = _sha256(
            candidate.get("receipt_digest"), label="CAST child receipt"
        )
        if _file_sha256(score_path) != score_file_sha:
            raise PairV7PhaseA2Error("CAST child bytes differ after validation")
        source_path = _plain_absolute_file(
            event.get("source_video_path"), label="source video"
        )
        clean_path = _plain_absolute_file(
            event.get("clean_latent_path"), label="clean latent"
        )
        gaussian_path = _plain_absolute_file(
            event.get("official_gaussian_path"), label="official Gaussian"
        )
        file_rows = (
            (source_path, "source_video_file_sha256"),
            (clean_path, "clean_latent_file_sha256"),
            (gaussian_path, "official_gaussian_file_sha256"),
        )
        for path, field in file_rows:
            declared = _sha256(event.get(field), label=field)
            if _file_sha256(path) != declared:
                raise PairV7PhaseA2Error(f"event {event_id} artifact bytes changed")
        spec = PhaseA2EventSpec(
            event_id=event_id,
            pair_id=pair_id,
            dp_arm=dp_arm,
            action_family=family,
            prompt_by_branch=prompts,
            prompt_bank_sha256=object_sha256(prompts),
            raw_caption_by_branch=captions,
            raw_caption_bank_sha256=object_sha256(captions),
            source_sample_id=source_id,
            generation_seed=generation_seed,
            source_noise_key_sha256=_sha256(
                source_noise_key, label="source-noise key"
            ),
            source_video=FileBinding(
                source_path, str(event["source_video_file_sha256"])
            ),
            clean_latent=FileBinding(
                clean_path, str(event["clean_latent_file_sha256"])
            ),
            clean_latent_tensor_key=str(event["clean_latent_tensor_key"]),
            clean_latent_tensor_sha256=_sha256(
                event.get("clean_latent_tensor_sha256"),
                label="clean latent tensor",
            ),
            official_gaussian=FileBinding(
                gaussian_path, str(event["official_gaussian_file_sha256"])
            ),
            official_gaussian_tensor_key=str(
                event["official_gaussian_tensor_key"]
            ),
            official_gaussian_tensor_sha256=_sha256(
                event.get("official_gaussian_tensor_sha256"),
                label="official Gaussian tensor",
            ),
            latent_shape=tuple(int(item) for item in shape),
            event_digest=_sha256(event.get("event_digest"), label="event digest"),
            cast_score_receipt=FileBinding(score_path, score_file_sha),
            cast_score_receipt_digest=score_digest,
            raw_global_action_energy_score=float(score),
        )
        specs.append(spec)
        selected_receipts.append(
            {
                "event_id": event_id,
                "pair_id": pair_id,
                "dp_arm": dp_arm,
                "source_sample_id": source_id,
                "generation_seed": spec.generation_seed,
                "source_noise_key_sha256": spec.source_noise_key_sha256,
                "action_family": family,
                "event_digest": spec.event_digest,
                "cast_score_receipt_path": str(score_path),
                "cast_score_receipt_file_sha256": score_file_sha,
                "cast_score_receipt_digest": score_digest,
                "raw_global_action_energy_score": float(score),
                "source_video_path": str(spec.source_video.path),
                "source_video_file_sha256": spec.source_video.sha256,
                "clean_latent_path": str(spec.clean_latent.path),
                "clean_latent_file_sha256": spec.clean_latent.sha256,
                "clean_latent_tensor_key": spec.clean_latent_tensor_key,
                "geometry_source_video_sha256": spec.source_video.sha256,
                "clean_latent_tensor_sha256": (
                    spec.clean_latent_tensor_sha256
                ),
                "official_gaussian_tensor_sha256": (
                    spec.official_gaussian_tensor_sha256
                ),
                "official_gaussian_path": str(spec.official_gaussian.path),
                "official_gaussian_file_sha256": spec.official_gaussian.sha256,
                "official_gaussian_tensor_key": (
                    spec.official_gaussian_tensor_key
                ),
                "candidate_shape": list(spec.latent_shape),
                "prompt_by_branch": dict(spec.prompt_by_branch),
                "full_t2v_caption_by_branch": dict(
                    spec.raw_caption_by_branch
                ),
                "prompt_bank_sha256": spec.prompt_bank_sha256,
                "raw_caption_bank_sha256": spec.raw_caption_bank_sha256,
            }
        )
    if (
        [(row.pair_id, row.dp_arm) for row in specs]
        != [("fit", 0), ("fit", 1), ("confirmation", 0), ("confirmation", 1)]
        or len({row.source_sample_id for row in specs}) != 4
        or any(
            len({row.action_family for row in specs if row.pair_id == pair}) != 2
            for pair in PAIR_IDS
        )
    ):
        raise PairV7PhaseA2Error("fixed two-pair DP2 factorial differs")
    cell_keys = {
        (str(cell.get("pair_wave")), int(cell.get("schedule", {}).get("schedule_index", -1)))
        for cell in plan.get("primary_cells", ())
        if isinstance(cell, Mapping) and isinstance(cell.get("schedule"), Mapping)
    }
    if cell_keys != {(pair, schedule) for pair in PAIR_IDS for schedule in SCHEDULE_INDICES}:
        raise PairV7PhaseA2Error("four preregistered primary cells differ")
    group_digests = [
        _sha256(value, label="CAST group receipt")
        for value in cast_group_receipt_digests
    ]
    if len(group_digests) != 2 or len(set(group_digests)) != 2:
        raise PairV7PhaseA2Error("CAST requires two distinct group receipts")
    binding = _seal(
        {
            "schema_version": BANK_BINDING_SCHEMA,
            "method_name": METHOD_NAME,
            "authority_composition": (
                "prospective_preregistration_plus_live_complete_CAST_v4_bank_validation"
            ),
            "preregistration_alone_geometry_measurement_authorized": False,
            "combined_read_only_geometry_measurement_authorized": True,
            "preregistration_path": str(plan_path),
            "preregistration_file_sha256": plan_file,
            "preregistration_digest": prereg_digest,
            "checkpoint_tree_sha256": plan.get("checkpoint_tree_sha256"),
            "action_adapter_schema_sha256": plan.get(
                "action_adapter_schema_sha256"
            ),
            "cast_method_binding": dict(cast_method_binding),
            "cast_root_binding": dict(cast_root_binding),
            "cast_group_receipt_digests": group_digests,
            "cast_candidate_receipt_count": len(rows),
            "all_forty_cast_children_semantically_validated": True,
            "selected_event_count": 4,
            "selected_events": selected_receipts,
            "primary_pair_ids": list(PAIR_IDS),
            "primary_schedule_indices": list(SCHEDULE_INDICES),
            "pilot_schedule_index_33_reused": False,
            "candidate_or_seed_selection_performed": False,
            "confirmation_used_as_preregistered_measurement_not_optimizer_gate": True,
            "population_go_consumed": False,
            "raw_gradient_persistence_authorized": False,
            "optimizer_constructed": False,
            "parameter_mutation_performed": False,
            "mask_flow_pose_track_or_trajectory_used": False,
        }
    )
    manifest = PhaseA2Manifest(
        path=plan_path,
        raw_sha256=plan_file,
        preregistration_digest=str(prereg_digest),
        checkpoint_tree_sha256=str(plan["checkpoint_tree_sha256"]),
        action_adapter_schema_sha256=str(plan["action_adapter_schema_sha256"]),
        events=tuple(specs),
        manifest_digest=binding["receipt_digest"],
    )
    return manifest, binding


def _validate_complete_cast_bank(
    *,
    fit_authority: Any,
    checkpoint_identity: Mapping[str, Any],
    cast_archive_path: str | Path,
    expected_cast_archive_sha256: str,
    expected_cast_revision: str,
    cast_root_path: str | Path,
    expected_cast_root_sha256: str,
    group_paths: Sequence[str | Path],
    expected_group_sha256: Sequence[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    method = fit_authority._validate_cast_method_archive(
        cast_archive_path,
        expected_sha256=expected_cast_archive_sha256,
        expected_revision=expected_cast_revision,
    )
    root = fit_authority._validate_cast_root_spec(
        cast_root_path, expected_sha256=expected_cast_root_sha256
    )
    if len(group_paths) != 2 or len(expected_group_sha256) != 2:
        raise PairV7PhaseA2Error("exactly two CAST group receipts are required")
    groups = [
        fit_authority._validate_cast_group(
            path,
            expected_file_sha256=digest,
            root_spec_sha256=root["file_sha256"],
            method_archive_sha256=method["file_sha256"],
            method_revision=method["git_archive_revision"],
            checkpoint_content_identity=checkpoint_identity,
        )
        for path, digest in zip(group_paths, expected_group_sha256)
    ]
    if (
        len({row["group_id"] for row in groups}) != 2
        or len({row["receipt_digest"] for row in groups}) != 2
        or len({row["frozen_checkpoint_receipt_digest"] for row in groups}) != 1
    ):
        raise PairV7PhaseA2Error("CAST A/B group closure differs")
    candidates = [row for group in groups for row in group["candidate_receipts"]]
    if len(candidates) != 40:
        raise PairV7PhaseA2Error("CAST A/B bank is not forty children")
    return method, root, groups, candidates


def _deduplicate_bindings(rows: Sequence[FileBinding]) -> tuple[FileBinding, ...]:
    indexed: dict[Path, str] = {}
    for row in rows:
        previous = indexed.setdefault(row.path, row.sha256)
        if previous != row.sha256:
            raise PairV7PhaseA2Error("one external path has two declared digests")
    return tuple(FileBinding(path, indexed[path]) for path in sorted(indexed, key=str))


def preflight(args: argparse.Namespace) -> PhaseA2Preflight:
    if args.ack_root_reviewed_phase_a2_launch is not True:
        raise PairV7PhaseA2Error("root-reviewed Phase-A2 acknowledgement is required")
    if args.ack_no_parameter_mutation_no_success_claim is not True:
        raise PairV7PhaseA2Error("no-mutation/no-success acknowledgement is required")
    for field in ("runtime_source_revision", "cast_method_revision"):
        _sha1(getattr(args, field), label=field)
    for field in (
        "expected_checkpoint_content_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_preregistration_sha256",
        "runtime_source_archive_sha256",
        "cast_method_archive_sha256",
        "expected_cast_root_spec_sha256",
    ):
        _sha256(getattr(args, field), label=field)
    if len(args.scorer_group_receipt) != 2 or len(
        args.expected_scorer_group_receipt_sha256
    ) != 2:
        raise PairV7PhaseA2Error("two CAST group receipts are required")
    for digest in args.expected_scorer_group_receipt_sha256:
        _sha256(digest, label="CAST group receipt")

    import audit_pair_v7_phase_a_geometry as phase_a
    import infer_source_kv_carrier_oracle as checkpoint_audit
    import pair_v7_fit_only_geometry_authority as fit_authority
    import pair_v5_t2v_guidance_distill as cagd

    runtime_receipt = phase_a._validate_git_archive(
        args.runtime_source_archive,
        expected_sha256=args.runtime_source_archive_sha256,
        expected_revision=args.runtime_source_revision,
        label="Phase-A2 runtime source archive",
        required_members=tuple(
            phase_a._RUNTIME_ARCHIVE_REQUIRED
            | PHASE_A2_RUNTIME_ARCHIVE_REQUIRED
        ),
    )
    try:
        checkpoint_identity = checkpoint_audit.validate_checkpoint_content(
            Path(args.checkpoint),
            Path(args.checkpoint_content_manifest),
            expected_manifest_sha256=(
                args.expected_checkpoint_content_manifest_sha256
            ),
        )
    except Exception as error:
        raise PairV7PhaseA2Error(f"checkpoint content audit failed: {error}") from error
    plan_path = _plain_absolute_file(
        args.multicondition_preregistration,
        label="multicondition preregistration",
    )
    plan = preregistration.validate_preregistration(
        plan_path=plan_path,
        expected_plan_file_sha256=args.expected_preregistration_sha256,
    )
    if (
        plan.get("checkpoint_tree_sha256") != args.expected_checkpoint_tree_sha256
        or plan.get("action_adapter_schema_sha256")
        != cagd.ACTION_ADAPTER_SCHEMA_SHA256
    ):
        raise PairV7PhaseA2Error("plan checkpoint/Action-LoRA schema differs")
    method, root, groups, candidates = _validate_complete_cast_bank(
        fit_authority=fit_authority,
        checkpoint_identity=checkpoint_identity,
        cast_archive_path=args.cast_method_archive,
        expected_cast_archive_sha256=args.cast_method_archive_sha256,
        expected_cast_revision=args.cast_method_revision,
        cast_root_path=args.cast_root_spec,
        expected_cast_root_sha256=args.expected_cast_root_spec_sha256,
        group_paths=args.scorer_group_receipt,
        expected_group_sha256=args.expected_scorer_group_receipt_sha256,
    )
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    def rebuild_t2v_prompts(
        captions: Mapping[str, str],
    ) -> Mapping[str, str]:
        prompts, _deployment, _receipt = phase_a.build_task_prompt_registry(
            captions, prompt_cleaner=prompt_clean
        )
        return prompts

    manifest, binding_receipt = bind_plan_events_to_cast_candidates(
        plan=plan,
        plan_path=plan_path,
        plan_file_sha256=args.expected_preregistration_sha256,
        candidates=candidates,
        cast_group_receipt_digests=[row["receipt_digest"] for row in groups],
        cast_method_binding=method,
        cast_root_binding=root,
        prompt_rebuilder=rebuild_t2v_prompts,
    )

    runtime_events: list[PhaseA2RuntimeEvent] = []
    for spec in manifest.events:
        media = fit_authority._inspect_source_media(spec.source_video.path)
        if media != {"frame_count": FRAME_COUNT, "fps": FPS}:
            raise PairV7PhaseA2Error("source media is not exact81 at 25 fps")
        t2v_prompts, _deployment_prompts, _receipt = (
            phase_a.build_task_prompt_registry(
                spec.raw_caption_by_branch, prompt_cleaner=prompt_clean
            )
        )
        if t2v_prompts != dict(spec.prompt_by_branch):
            raise PairV7PhaseA2Error(
                f"CAST prompt/caption reconstruction differs: {spec.event_id}"
            )
        latent = fit_authority._load_tensor_artifact(
            spec.clean_latent.path, spec.clean_latent_tensor_key
        )
        epsilon = fit_authority._load_tensor_artifact(
            spec.official_gaussian.path, spec.official_gaussian_tensor_key
        )
        if (
            tuple(int(item) for item in latent.shape) != spec.latent_shape
            or tuple(int(item) for item in epsilon.shape) != spec.latent_shape
            or cagd.tensor_sha256(latent) != spec.clean_latent_tensor_sha256
            or cagd.tensor_sha256(epsilon)
            != spec.official_gaussian_tensor_sha256
        ):
            raise PairV7PhaseA2Error("loaded event tensor binding differs")
        runtime_events.append(PhaseA2RuntimeEvent(spec, latent, epsilon))

    external: list[FileBinding] = [
        FileBinding(plan_path, args.expected_preregistration_sha256),
        FileBinding(Path(runtime_receipt["path"]), runtime_receipt["file_sha256"]),
        FileBinding(Path(method["path"]), method["file_sha256"]),
        FileBinding(Path(root["path"]), root["file_sha256"]),
    ]
    checkpoint_manifest_path = _plain_absolute_file(
        checkpoint_identity["manifest_path"], label="checkpoint manifest"
    )
    external.append(
        FileBinding(
            checkpoint_manifest_path,
            checkpoint_identity["manifest_sha256_computed"],
        )
    )
    for group in groups:
        external.append(FileBinding(Path(group["path"]), group["file_sha256"]))
        external.extend(
            FileBinding(Path(row["path"]), row["file_sha256"])
            for row in group["candidate_receipts"]
        )
    for spec in manifest.events:
        external.extend(
            (
                spec.source_video,
                spec.clean_latent,
                spec.official_gaussian,
                spec.cast_score_receipt,
            )
        )
    files = _deduplicate_bindings(external)
    binding = PhaseA2BankBinding(
        receipt={
            **dict(binding_receipt),
            "external_file_count": len(files),
            "external_file_manifest_digest": object_sha256(
                [
                    {"path": str(row.path), "sha256": row.sha256}
                    for row in files
                ]
            ),
        },
        external_files=files,
    )
    # The extra two fields above are a runtime TOCTOU envelope, so reseal it.
    unsigned_binding = dict(binding.receipt)
    unsigned_binding.pop("receipt_digest")
    binding = PhaseA2BankBinding(
        receipt=_seal(unsigned_binding), external_files=files
    )
    # Action-query and WORLD-input receipts must name the final live authority,
    # including its complete TOCTOU file envelope, rather than the intermediate
    # pure semantic binding returned by the helper above.
    manifest = replace(
        manifest, manifest_digest=binding.measurement_authority_digest
    )
    return PhaseA2Preflight(
        manifest=manifest,
        runtime_events=tuple(runtime_events),
        bank_binding=binding,
        checkpoint_identity=checkpoint_identity,
        runtime_archive=FileBinding(
            Path(runtime_receipt["path"]), runtime_receipt["file_sha256"]
        ),
        runtime_revision=runtime_receipt["git_archive_revision"],
        cast_archive=FileBinding(Path(method["path"]), method["file_sha256"]),
        cast_revision=method["git_archive_revision"],
    )


def _phase_a2_schedule_policy(schedule_index: int) -> Mapping[str, Any]:
    if type(schedule_index) is not int or schedule_index not in SCHEDULE_INDICES:
        raise PairV7PhaseA2Error("Phase-A2 schedule is outside {16,35}")
    import pair_v5_action_adapter as action_adapter
    import inference_sigma_strata as sigma_strata

    gate_name, gate_weight = action_adapter.sigma_gate(schedule_index)
    expected = preregistration.SCHEDULES[schedule_index]
    if (
        gate_name != expected["gate_name"]
        or float(gate_weight) != float(expected["gate_weight"])
        or sigma_strata.PINNED_TIMESTEPS[schedule_index] != expected["timestep"]
        or sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[schedule_index]
        != expected["sigma_float32_be_hex"]
        or gate_weight <= 0.0
    ):
        raise PairV7PhaseA2Error("Phase-A2 schedule registration differs")
    return {
        **dict(expected),
        "model_callbacks_authorized": True,
        "gradient_audit_authorized": True,
        "parameter_update_authorized": False,
    }


def _fresh_source_epsilon(
    shape: Sequence[int],
    *,
    source_sample_id: str,
    source_noise_contract: Mapping[str, Any],
    device: Any,
) -> tuple[Any, Mapping[str, Any]]:
    import torch

    keys = source_noise_contract.get("source_key_sha256_by_sample")
    declared = keys.get(source_sample_id) if isinstance(keys, Mapping) else None
    expected = preregistration._source_noise_key(source_sample_id)
    if (
        source_noise_contract.get("master_seed") != SOURCE_NOISE_MASTER_SEED
        or declared != expected
    ):
        raise PairV7PhaseA2Error("source-noise plan binding differs")
    seed = int.from_bytes(bytes.fromhex(expected)[:8], "big") % 2**63
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    epsilon = torch.randn(
        tuple(int(item) for item in shape),
        generator=generator,
        dtype=torch.float32,
    ).to(device).detach()
    return epsilon, _seal(
        {
            "schema_version": "bernini-pair-v7-phase-a2-source-noise-v1",
            "source_sample_id": source_sample_id,
            "master_seed": SOURCE_NOISE_MASTER_SEED,
            "source_noise_key_sha256": expected,
            "derived_torch_seed": seed,
            "same_epsilon_reused_for_s16_and_s35": True,
        }
    )


def build_phase_a2_source_coordinate(
    source_clean_latent: Any,
    source_native_epsilon: Any,
    *,
    schedule_index: int,
    sample_id: str,
    bank_binding_digest: str,
) -> Any:
    import torch
    import audit_pair_v7_phase_a_geometry as phase_a
    import source_self_native_ref_contrastive_v3 as native
    import source_self_runtime as distributed_runtime

    policy = _phase_a2_schedule_policy(schedule_index)
    for label, value in (
        ("source latent", source_clean_latent),
        ("source epsilon", source_native_epsilon),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.ndim != 5
            or tuple(value.shape[:3]) != (1, 16, 21)
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise PairV7PhaseA2Error(f"{label} must be detached exact81 FP32")
    if source_clean_latent.shape != source_native_epsilon.shape:
        raise PairV7PhaseA2Error("source latent/epsilon geometry differs")
    states = native.build_multi_sigma_states(
        source_clean_latent,
        source_native_epsilon,
        indices=[schedule_index],
        device=source_clean_latent.device,
    )
    x_sigma = states.noisy[0].detach().float().contiguous()
    timestep = states.timesteps[0:1].detach().float().contiguous()
    sigma = float(states.sigmas[0].item())
    receipt = _seal(
        {
            "schema_version": "bernini-pair-v7-phase-a2-source-coordinate-v1",
            "sample_id": sample_id,
            "schedule_index": schedule_index,
            "schedule_policy": policy,
            "source_clean_latent_sha256": distributed_runtime.tensor_sha256(
                source_clean_latent
            ),
            "source_native_epsilon_sha256": distributed_runtime.tensor_sha256(
                source_native_epsilon
            ),
            "x_sigma_sha256": distributed_runtime.tensor_sha256(x_sigma),
            "sigma_float64_hex": sigma.hex(),
            "timestep_float32_be_hex": struct.pack(
                "!f", float(timestep.item())
            ).hex(),
            "bank_binding_digest": _sha256(
                bank_binding_digest, label="bank binding"
            ),
            "construction": "(1-sigma)*source_clean_latent+sigma*source_native_epsilon",
            "pure_t2v_official_gaussian_used": False,
        }
    )
    return phase_a.PhaseASourceCoordinate(
        x_sigma=x_sigma,
        timestep=timestep,
        sigma=sigma,
        schedule_index=schedule_index,
        receipt=receipt,
    )


def build_phase_a2_action_query(
    event_latent: Any,
    official_epsilon: Any,
    *,
    event_spec: PhaseA2EventSpec,
    manifest: PhaseA2Manifest,
    schedule_index: int,
    bank_binding_digest: str,
) -> tuple[Any, Mapping[str, Any]]:
    import torch
    import pair_v5_action_adapter as action_adapter
    import pair_v5_t2v_guidance_distill as cagd
    import source_self_native_ref_contrastive_v3 as native

    policy = _phase_a2_schedule_policy(schedule_index)
    if event_spec not in manifest.events:
        raise PairV7PhaseA2Error("event is absent from multicondition manifest")
    for label, value, digest in (
        ("event latent", event_latent, event_spec.clean_latent_tensor_sha256),
        (
            "official Gaussian",
            official_epsilon,
            event_spec.official_gaussian_tensor_sha256,
        ),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(int(item) for item in value.shape) != event_spec.latent_shape
            or value.device.type == "meta"
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
            or cagd.tensor_sha256(value) != digest
        ):
            raise PairV7PhaseA2Error(f"sealed {label} binding differs")
    gate_name, gate_weight = action_adapter.sigma_gate(schedule_index)
    sigma = torch.tensor(
        [native.NATIVE_UNIPC40_SIGMAS[schedule_index]],
        dtype=torch.float32,
        device=event_latent.device,
    )
    timestep = torch.tensor(
        [native.NATIVE_UNIPC40_TIMESTEPS[schedule_index]],
        dtype=torch.float32,
        device=event_latent.device,
    )
    sigma_view = sigma.reshape(1, 1, 1, 1, 1)
    x_sigma = (
        (1.0 - sigma_view) * event_latent + sigma_view * official_epsilon
    ).detach().contiguous()
    coordinate = {
        "authority_scope": "multicondition_read_only_gradient_geometry",
        "event_id": event_spec.event_id,
        "event_digest": event_spec.event_digest,
        "pair_id": event_spec.pair_id,
        "source_sample_id": event_spec.source_sample_id,
        "manifest_digest": manifest.manifest_digest,
        "bank_binding_digest": _sha256(
            bank_binding_digest, label="bank binding"
        ),
        "clean_t2v_latent_tensor_sha256": (
            event_spec.clean_latent_tensor_sha256
        ),
        "official_gaussian_tensor_sha256": (
            event_spec.official_gaussian_tensor_sha256
        ),
        "x_sigma_tensor_sha256": cagd.tensor_sha256(x_sigma),
        "schedule_index": schedule_index,
        "sigma_float32_be_hex": policy["sigma_float32_be_hex"],
        "timestep_float32_be_hex": struct.pack(
            "!f", float(timestep.item())
        ).hex(),
        "construction": "(1-sigma)*preregistered_event_y0+sigma*its_official_epsilon",
    }
    query = cagd.SameStateQuery(
        sample_id=event_spec.event_id,
        x_sigma=x_sigma,
        sigma=sigma,
        timestep=timestep,
        schedule_index=schedule_index,
        gate_name=gate_name,
        gate_weight=float(gate_weight),
        coordinate_digest=object_sha256(coordinate),
        x_sigma_object_id=id(x_sigma),
        sigma_object_id=id(sigma),
        timestep_object_id=id(timestep),
        x_sigma_version=int(x_sigma._version),
        sigma_version=int(sigma._version),
        timestep_version=int(timestep._version),
    )
    receipt = _seal(
        {
            **coordinate,
            "schema_version": ACTION_QUERY_SCHEMA,
            "prompt_bank_sha256": event_spec.prompt_bank_sha256,
            "checkpoint_tree_sha256": manifest.checkpoint_tree_sha256,
            "combined_bank_geometry_measurement_authorized": True,
            "population_go_consumed": False,
            "parameter_update_authorized": False,
        }
    )
    return query, receipt


def build_phase_a2_measurement_objective(packet: Any) -> tuple[Any, Mapping[str, Any]]:
    import torch
    import pair_v5_t2v_guidance_distill as cagd

    if not isinstance(packet, cagd.PredictionPacket):
        raise PairV7PhaseA2Error("measurement packet type differs")
    packet.query.assert_unchanged()
    if (
        packet.query.schedule_index not in SCHEDULE_INDICES
        or packet.query.gate_name == "low_base_only"
        or packet.query.gate_weight <= 0.0
    ):
        raise PairV7PhaseA2Error("measurement coordinate differs")
    config = cagd.DistillConfig()
    config.validate()
    teacher = cagd.build_bounded_teacher(packet.base_by_branch, config=config)
    gated_teacher = teacher.vector * packet.query.gate_weight
    correction = (
        packet.student_by_branch["action"].float()
        - packet.base_by_branch["action"].float()
    )
    action_match = torch.nn.functional.mse_loss(correction, gated_teacher)
    parity = {
        branch: torch.nn.functional.mse_loss(
            packet.student_by_branch[branch].float(),
            packet.base_by_branch[branch].float(),
        )
        for branch in cagd.NEGATIVE_BRANCHES
    }
    negative_parity = torch.stack(tuple(parity.values())).mean()
    student_rms = (
        correction.float().square().mean().add(config.epsilon**2).sqrt()
        - config.epsilon
    )
    trust_cap = correction.new_tensor(
        max(
            teacher.bounded_rms
            * packet.query.gate_weight
            * config.student_teacher_rms_ratio,
            config.minimum_teacher_rms,
        )
    )
    trust_penalty = torch.relu(student_rms - trust_cap).square()
    loss = (
        action_match
        + config.negative_parity_weight * negative_parity
        + config.trust_penalty_weight * trust_penalty
    )
    components = (loss, action_match, negative_parity, trust_penalty)
    if any(
        value.dtype != torch.float32
        or value.ndim != 0
        or not bool(torch.isfinite(value).item())
        for value in components
    ) or not loss.requires_grad:
        raise PairV7PhaseA2Error("measurement objective differs")
    receipt = _seal(
        {
            "schema_version": ACTION_OBJECTIVE_SCHEMA,
            "authority_scope": "backward_and_vjp_measurement_only",
            "coordinate_digest": packet.query.coordinate_digest,
            "prompt_bank_sha256": packet.prompt_bank_digest,
            "schedule_index": packet.query.schedule_index,
            "sigma_gate": packet.query.gate_name,
            "sigma_gate_weight": packet.query.gate_weight,
            "leaf_vjp_mode": packet.leaf_vjp_mode,
            "branch_order": list(cagd.BRANCH_ORDER),
            "call_order": list(packet.call_order),
            "loss_value": float(loss.detach().item()),
            "teacher_vector_sha256": cagd.tensor_sha256(teacher.vector),
            "optimizer_capable_receipt_constructed": False,
            "backward_measurement_authorized": True,
            "vjp_replay_measurement_authorized": True,
            "parameter_mutation_performed": False,
        }
    )
    return cagd.DistillObjective(
        loss=loss,
        action_match_loss=action_match,
        negative_parity_loss=negative_parity,
        trust_penalty=trust_penalty,
        parity_by_branch=parity,
        teacher=teacher,
        receipt=receipt,
    ), receipt


def extract_phase_a2_action_gradient(
    *,
    runtime_event: PhaseA2RuntimeEvent,
    manifest: PhaseA2Manifest,
    bank_binding_digest: str,
    diffusion: Any,
    transformer: Any,
    action_handle: Any,
    conditions: Mapping[str, Any],
    gauge: Any,
    parallel: Any,
    sp_rank: int,
    schedule_index: int,
    device: Any,
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    import torch
    import audit_pair_v7_phase_a_geometry as phase_a
    import pair_v5_t2v_guidance_distill as cagd

    state_before = phase_a._assert_fixed_gauge_state(
        gauge, parameter_state_sha256, label="Phase-A2 action before VJP"
    )
    phase_a._clear_gauge_gradients(gauge)
    latent = runtime_event.event_latent_cpu.to(device=device).contiguous()
    epsilon = runtime_event.official_epsilon_cpu.to(device=device).contiguous()
    phase_a._broadcast_sp(latent, parallel=parallel)
    phase_a._broadcast_sp(epsilon, parallel=parallel)
    callback = phase_a.NativeT2VGeometryCallback(
        diffusion=diffusion,
        transformer=transformer,
        action_handle=action_handle,
        condition_by_branch=conditions,
        prompt_by_branch=runtime_event.spec.prompt_by_branch,
        sp_rank=sp_rank,
    )
    query, query_receipt = build_phase_a2_action_query(
        latent,
        epsilon,
        event_spec=runtime_event.spec,
        manifest=manifest,
        schedule_index=schedule_index,
        bank_binding_digest=bank_binding_digest,
    )
    packet = cagd.collect_same_state_predictions(
        query,
        prompt_by_branch=runtime_event.spec.prompt_by_branch,
        denoise_callback=callback,
        leaf_vjp_mode=True,
    )
    objective, objective_receipt = build_phase_a2_measurement_objective(packet)
    objective.loss.backward()
    replay = cagd.replay_student_vjp(
        packet,
        runtime_event.spec.prompt_by_branch,
        callback,
        rtol=phase_a.VJP_RTOL,
        atol=phase_a.VJP_ATOL,
    )
    phase_a._validate_frozen_a_gradients(gauge)
    gradient = phase_a._average_b_gradients_over_sp4(
        gauge, parallel, label="Phase-A2 pure-T2V action"
    )
    state_after = phase_a._assert_fixed_gauge_state(
        gauge, parameter_state_sha256, label="Phase-A2 action after VJP"
    )
    digest = phase_a._named_gradient_sha256(gradient)
    rank_receipt = _seal(
        {
            "schema_version": ACTION_GRADIENT_SCHEMA,
            "condition_id": (
                f"{runtime_event.spec.pair_id}."
                f"{runtime_event.spec.source_sample_id}.s{schedule_index}"
            ),
            "pair_id": runtime_event.spec.pair_id,
            "source_sample_id": runtime_event.spec.source_sample_id,
            "schedule_index": schedule_index,
            "candidate_id": runtime_event.spec.event_id,
            "action_family": runtime_event.spec.action_family,
            "sp_rank": sp_rank,
            "event_digest": runtime_event.spec.event_digest,
            "cast_score_receipt_digest": (
                runtime_event.spec.cast_score_receipt_digest
            ),
            "action_query_receipt_digest": query_receipt["receipt_digest"],
            "measurement_objective_receipt_digest": objective_receipt[
                "receipt_digest"
            ],
            "bank_binding_digest": bank_binding_digest,
            "gradient_sha256": digest,
            "vjp_replay_max_abs": max(replay.values()),
            "checkpoint_content_receipt_digest": (
                checkpoint_content_receipt_digest
            ),
            "parameter_state_sha256": parameter_state_sha256,
            "parameter_state_before_vjp_sha256": state_before,
            "parameter_state_after_vjp_sha256": state_after,
            "sp4_averaged": True,
            "dp_averaged_before_global_solve": False,
            "pure_t2v_visual_used_as_rv2v_target_or_donor": False,
            "parameter_mutation_performed": False,
        }
    )
    bundle = phase_a._bundle_sp4_vjp_receipts(
        local_receipt=rank_receipt,
        averaged_gradient=gradient,
        parallel=parallel,
        label=f"phase-a2-action:{rank_receipt['condition_id']}",
        common_fields=(
            "schema_version",
            "condition_id",
            "pair_id",
            "source_sample_id",
            "schedule_index",
            "candidate_id",
            "action_family",
            "event_digest",
            "cast_score_receipt_digest",
            "action_query_receipt_digest",
            "measurement_objective_receipt_digest",
            "bank_binding_digest",
            "gradient_sha256",
            "checkpoint_content_receipt_digest",
            "parameter_state_sha256",
            "parameter_state_before_vjp_sha256",
            "parameter_state_after_vjp_sha256",
        ),
    )
    return gradient, {"rank_receipt": rank_receipt, "sp4_bundle": bundle}


def _prepare_identity_protocol(
    diffusion: Any,
    coordinate: Any,
    *,
    bank_binding_digest: str,
) -> tuple[Any, Mapping[str, Any]]:
    import torch
    import audit_pair_v7_phase_a_geometry as phase_a
    import source_self_native_ref_contrastive_v3 as native

    policy = _phase_a2_schedule_policy(coordinate.schedule_index)
    scheduler = getattr(diffusion, "scheduler", None)
    setter = getattr(scheduler, "set_timesteps", None)
    if not callable(setter):
        raise PairV7PhaseA2Error("diffusion lacks UniPC scheduler")
    config = getattr(scheduler, "config", None)
    flow_shift = config.get("flow_shift") if isinstance(config, Mapping) else getattr(
        config, "flow_shift", None
    )
    if float(flow_shift) != DEPLOYMENT_FLOW_SHIFT:
        raise PairV7PhaseA2Error("identity scheduler flow shift differs")
    setter(40)
    sigmas = getattr(scheduler, "sigmas", None)
    timesteps = getattr(scheduler, "timesteps", None)
    index = coordinate.schedule_index
    expected_sigma = torch.tensor(native.NATIVE_UNIPC40_SIGMAS[index], dtype=torch.float32)
    expected_timestep = torch.tensor(
        native.NATIVE_UNIPC40_TIMESTEPS[index], dtype=torch.float32
    )
    if (
        not isinstance(sigmas, torch.Tensor)
        or sigmas.device.type != "cpu"
        or sigmas.dtype != torch.float32
        or sigmas.ndim != 1
        or int(sigmas.numel()) != 41
        or not isinstance(timesteps, torch.Tensor)
        or int(timesteps.numel()) != 40
        or not torch.equal(sigmas[index], expected_sigma)
        or not torch.equal(timesteps[index].float().cpu(), expected_timestep)
        or struct.pack("!f", float(sigmas[index].item()))
        != bytes.fromhex(policy["sigma_float32_be_hex"])
    ):
        raise PairV7PhaseA2Error("official scheduler cell differs")
    receipt = _seal(
        {
            "schema_version": IDENTITY_PROTOCOL_SCHEMA,
            "authority": "VideoEdit_infer_lora_frozen_deployment_contract",
            "bank_binding_digest": bank_binding_digest,
            "guidance_mode": phase_a.APG_GUIDANCE_MODE,
            "visual_condition": "source_video_only_V",
            "image_reference_count": 0,
            "forward_order_per_field": ["V_negative", "V_positive"],
            "omega_txt": phase_a.APG_GUIDANCE_SCALE,
            "eta": phase_a.APG_ETA,
            "norm_threshold": phase_a.APG_NORM_THRESHOLD,
            "momentum": phase_a.APG_MOMENTUM,
            "flow_shift": DEPLOYMENT_FLOW_SHIFT,
            "num_inference_steps": 40,
            "schedule_index": index,
            "timestep": int(expected_timestep.item()),
            "sigma_source": f"scheduler.sigmas[{index}]_cpu_fp32",
            "sigma_float32_be_hex": policy["sigma_float32_be_hex"],
            "fresh_zero_momentum_history_equivalent": True,
            "old_diff_vjp_coefficient": 0.0,
            "full_sampler_trajectory_equivalent": False,
            "parameter_mutation_performed": False,
        }
    )
    return sigmas[index], receipt


def build_phase_a2_feature_runtime(
    *,
    diffusion: Any,
    transformer: Any,
    action_handle: Any,
    correct_source: Any,
    coordinate: Any,
    condition_by_branch: Mapping[str, Any],
    unconditional: Any,
    sp_rank: int,
    bank_binding_digest: str,
) -> Any:
    import audit_pair_v7_phase_a_geometry as phase_a

    class PhaseA2FeatureRuntime(phase_a.NativeFeatureVJPRuntime):
        def __init__(self) -> None:
            self.diffusion = diffusion
            self.transformer = transformer
            self.action_handle = action_handle
            self.correct_source = correct_source
            self.coordinate = coordinate
            self.condition_by_branch = dict(condition_by_branch)
            self.unconditional = unconditional
            self.sp_rank = sp_rank
            self.pack = None
            self.sigma, self.deployment_protocol = _prepare_identity_protocol(
                diffusion,
                coordinate,
                bank_binding_digest=bank_binding_digest,
            )
            self.measurement_cache = {}

    return PhaseA2FeatureRuntime()


def validate_cross_family_identity_coordinate_closure(
    identity_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Require both identity families to probe one identical source cell."""

    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for raw in identity_rows:
        if not isinstance(raw, Mapping):
            raise PairV7PhaseA2Error("identity coordinate row type differs")
        key = (
            str(raw.get("pair_id")),
            str(raw.get("source_sample_id")),
            int(raw.get("schedule_index", -1)),
        )
        groups.setdefault(key, []).append(raw)
    if len(groups) != 8:
        raise PairV7PhaseA2Error("identity source-coordinate cell count differs")
    result: list[Mapping[str, Any]] = []
    for (pair_id, source_id, schedule_index), rows in sorted(groups.items()):
        observed = {
            (str(row.get("family")), int(row.get("sketch_index", -1)))
            for row in rows
        }
        coordinates = {
            row.get("source_coordinate_receipt_digest") for row in rows
        }
        if (
            len(rows) != len(IDENTITY_FAMILIES) * SKETCH_COUNT
            or observed
            != {
                (family, sketch)
                for family in IDENTITY_FAMILIES
                for sketch in range(SKETCH_COUNT)
            }
            or len(coordinates) != 1
        ):
            raise PairV7PhaseA2Error(
                "two identity families do not share one source coordinate"
            )
        coordinate = next(iter(coordinates))
        if not isinstance(coordinate, str) or _SHA256_RE.fullmatch(coordinate) is None:
            raise PairV7PhaseA2Error("identity source-coordinate digest differs")
        result.append(
            {
                "pair_id": pair_id,
                "source_sample_id": source_id,
                "schedule_index": schedule_index,
                "identity_family_count": len(IDENTITY_FAMILIES),
                "identity_probe_count": len(rows),
                "source_coordinate_receipt_digest": coordinate,
                "cross_family_coordinate_consensus": True,
            }
        )
    return tuple(result)


def validate_measured_bank_against_binding(
    *,
    action_conditions: Sequence[Any],
    identity_conditions: Sequence[Any],
    manifest: PhaseA2Manifest,
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
) -> Mapping[str, Any]:
    import pair_v7_multicondition_nullspace_transport as multi
    import pair_v7_dual_coordinate_nullspace_transport as core

    checkpoint = _sha256(
        checkpoint_content_receipt_digest, label="checkpoint content receipt"
    )
    state = _sha256(parameter_state_sha256, label="parameter state")
    actions = tuple(action_conditions)
    identities = tuple(identity_conditions)
    if len(actions) != EXPECTED_ACTION_COUNT or len(identities) != EXPECTED_IDENTITY_COUNT:
        raise PairV7PhaseA2Error("measured 8x64 bank count differs")
    spec_by_key = {
        (row.pair_id, row.source_sample_id): row for row in manifest.events
    }
    expected_action = {
        (row.pair_id, row.source_sample_id, schedule)
        for row in manifest.events
        for schedule in SCHEDULE_INDICES
    }
    observed_action = set()
    action_rows: list[Mapping[str, Any]] = []
    for row in actions:
        if not isinstance(row, multi.ActionConditionGradient):
            raise PairV7PhaseA2Error("action bank row type differs")
        key = (row.pair_id, row.source_sample_id)
        spec = spec_by_key.get(key)
        if (
            spec is None
            or row.schedule_index not in SCHEDULE_INDICES
            or row.candidate_id != spec.event_id
            or row.action_family != spec.action_family
            or row.event_digest != spec.event_digest
            or row.checkpoint_content_receipt_digest != checkpoint
            or row.parameter_state_sha256 != state
        ):
            raise PairV7PhaseA2Error("action row/authority binding differs")
        layout = core.GradientLayout.from_named_gradients(row.gradient_by_parameter)
        gradient_sha = core._tensor_sha256(
            layout.flatten(row.gradient_by_parameter, label=row.condition_id).float()
        )
        observed_action.add((row.pair_id, row.source_sample_id, row.schedule_index))
        action_rows.append(
            {
                "condition_id": row.condition_id,
                "pair_id": row.pair_id,
                "source_sample_id": row.source_sample_id,
                "schedule_index": row.schedule_index,
                "candidate_id": row.candidate_id,
                "action_family": row.action_family,
                "event_digest": row.event_digest,
                "gradient_computation_receipt_digest": (
                    row.gradient_computation_receipt_digest
                ),
                "gradient_sha256": gradient_sha,
            }
        )
    if observed_action != expected_action:
        raise PairV7PhaseA2Error("action bank factorial differs")
    expected_identity = {
        (row.pair_id, row.source_sample_id, schedule, family, sketch)
        for row in manifest.events
        for schedule in SCHEDULE_INDICES
        for family in IDENTITY_FAMILIES
        for sketch in range(SKETCH_COUNT)
    }
    observed_identity = set()
    identity_rows: list[Mapping[str, Any]] = []
    for row in identities:
        if not isinstance(row, multi.IdentityConditionProbe):
            raise PairV7PhaseA2Error("identity bank row type differs")
        row.probe.validate_metadata()
        key = (
            row.pair_id,
            row.source_sample_id,
            row.schedule_index,
            row.probe.family,
            row.sketch_index,
        )
        if (
            (row.pair_id, row.source_sample_id) not in spec_by_key
            or row.schedule_index not in SCHEDULE_INDICES
            or row.probe.checkpoint_content_receipt_digest != checkpoint
            or row.probe.parameter_state_sha256 != state
        ):
            raise PairV7PhaseA2Error("identity row/authority binding differs")
        layout = core.GradientLayout.from_named_gradients(
            row.probe.gradient_by_parameter
        )
        gradient_sha = core._tensor_sha256(
            layout.flatten(
                row.probe.gradient_by_parameter, label=row.probe.probe_id
            ).float()
        )
        observed_identity.add(key)
        identity_rows.append(
            {
                "probe_id": row.probe.probe_id,
                "pair_id": row.pair_id,
                "source_sample_id": row.source_sample_id,
                "schedule_index": row.schedule_index,
                "family": row.probe.family,
                "sketch_index": row.sketch_index,
                "feature_sketch_sha256": row.probe.feature_sketch_sha256,
                "source_coordinate_receipt_digest": (
                    row.probe.source_coordinate_receipt_digest
                ),
                "gradient_computation_receipt_digest": (
                    row.probe.gradient_computation_receipt_digest
                ),
                "gradient_sha256": gradient_sha,
            }
        )
    if observed_identity != expected_identity:
        raise PairV7PhaseA2Error("identity bank factorial differs")
    identity_coordinate_cells = validate_cross_family_identity_coordinate_closure(
        identity_rows
    )
    unsigned = {
        "schema_version": WORLD_INPUT_SCHEMA,
        "manifest_digest": manifest.manifest_digest,
        "checkpoint_content_receipt_digest": checkpoint,
        "parameter_state_sha256": state,
        "action_condition_count": len(actions),
        "identity_probe_count": len(identities),
        "action_rows": sorted(action_rows, key=lambda row: row["condition_id"]),
        "identity_rows": sorted(identity_rows, key=lambda row: row["probe_id"]),
        "identity_cross_family_coordinate_cells": list(
            identity_coordinate_cells
        ),
        "raw_gradient_values_persisted": False,
    }
    return {**unsigned, "input_digest": object_sha256(unsigned)}


def _cpu_action(row: Any) -> Any:
    import audit_pair_v7_phase_a_geometry as phase_a
    import pair_v7_multicondition_nullspace_transport as multi

    return multi.ActionConditionGradient(
        **{
            **row.__dict__,
            "gradient_by_parameter": phase_a._cpu_named_gradient_mapping(
                row.gradient_by_parameter, label=f"CPU action {row.condition_id}"
            ),
        }
    )


def _cpu_identity(row: Any) -> Any:
    import audit_pair_v7_phase_a_geometry as phase_a
    import pair_v7_multicondition_nullspace_transport as multi

    return multi.IdentityConditionProbe(
        pair_id=row.pair_id,
        source_sample_id=row.source_sample_id,
        schedule_index=row.schedule_index,
        sketch_index=row.sketch_index,
        probe=phase_a._cpu_identity_probe(row.probe),
    )


def validate_world_input_consensus(
    manifests: Sequence[Mapping[str, Any]], *, expected_count: int = WORLD_SIZE
) -> str:
    """Pure validator shared by the real collective path and model-free tests."""

    rows = tuple(manifests)
    if len(rows) != expected_count or any(not isinstance(row, Mapping) for row in rows):
        raise PairV7PhaseA2Error("WORLD input consensus count/type differs")
    digests = [row.get("input_digest") for row in rows]
    if (
        any(not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None for value in digests)
        or len(set(digests)) != 1
        or any(dict(row) != dict(rows[0]) for row in rows[1:])
    ):
        raise PairV7PhaseA2Error("WORLD gradient-bank input consensus differs")
    return str(digests[0])


def validate_root_solver_result(
    result: Mapping[str, Any],
    *,
    bank_binding_digest: str,
    expected_world_input_digest: str,
) -> tuple[Mapping[str, Any], str, bool]:
    """Validate the rank-zero wire envelope, including a legitimate NO-GO."""

    authority_digest = _sha256(bank_binding_digest, label="bank binding")
    world_input_digest = _sha256(
        expected_world_input_digest, label="WORLD input"
    )
    if (
        not isinstance(result, Mapping)
        or set(result)
        != {
            "ok",
            "transport_receipt",
            "transport_receipt_digest",
            "primary_replication_go",
        }
        or result.get("ok") is not True
    ):
        raise PairV7PhaseA2Error("rank-zero solver success envelope differs")
    transport = result.get("transport_receipt")
    go = result.get("primary_replication_go")
    if not isinstance(transport, Mapping) or type(go) is not bool:
        raise PairV7PhaseA2Error("rank-zero solver result type differs")
    unsigned = dict(transport)
    declared = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(declared, str)
        or _SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
        or declared != result.get("transport_receipt_digest")
        or transport.get("action_condition_count") != EXPECTED_ACTION_COUNT
        or transport.get("identity_probe_count") != EXPECTED_IDENTITY_COUNT
        or transport.get("multicondition_authority_digest")
        != authority_digest
        or transport.get("validated_world_input_digest")
        != world_input_digest
        or transport.get("primary_replication_go") is not go
        or transport.get("geometry_audit_passed") is not go
        or transport.get("parameter_mutation_performed") is not False
        or transport.get("gradient_or_adapter_artifact_written") is not False
    ):
        raise PairV7PhaseA2Error("rank-zero transport receipt closure differs")
    return dict(transport), declared, go


def world_rank0_cpu_multicondition_solve(
    *,
    action_conditions: Sequence[Any],
    identity_conditions: Sequence[Any],
    manifest: PhaseA2Manifest,
    bank_binding_digest: str,
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
    runtime_source_archive_sha256: str,
    runtime_source_revision: str,
    parallel: Any,
) -> PhaseA2WorldSolve:
    import torch
    import torch.distributed as dist
    import pair_v7_multicondition_nullspace_transport as multi
    import source_self_runtime as distributed_runtime

    authority_digest = _sha256(bank_binding_digest, label="bank binding")
    runtime_archive_digest = _sha256(
        runtime_source_archive_sha256, label="runtime source archive"
    )
    runtime_revision = _sha1(runtime_source_revision, label="runtime source revision")
    if manifest.manifest_digest != authority_digest:
        raise PairV7PhaseA2Error("manifest does not bind final TOCTOU authority")
    local_status: Mapping[str, Any]
    try:
        local_manifest = validate_measured_bank_against_binding(
            action_conditions=action_conditions,
            identity_conditions=identity_conditions,
            manifest=manifest,
            checkpoint_content_receipt_digest=checkpoint_content_receipt_digest,
            parameter_state_sha256=parameter_state_sha256,
        )
        local_status = {"ok": True, "manifest": local_manifest}
    except Exception as error:
        local_status = {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    statuses: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(statuses, dict(local_status), group=parallel.world_group)
    failures = [row for row in statuses if not isinstance(row, Mapping) or row.get("ok") is not True]
    if failures:
        raise PairV7PhaseA2Error(f"WORLD local bank validation failed: {failures[0]}")
    manifests = [row["manifest"] for row in statuses]
    input_digest = validate_world_input_consensus(manifests)
    rank = int(parallel.contract.rank)
    envelope: list[Any] = [None]
    if rank == 0:
        try:
            cpu_actions = tuple(_cpu_action(row) for row in action_conditions)
            cpu_identities = tuple(_cpu_identity(row) for row in identity_conditions)
            if any(
                tensor.device.type != "cpu" or tensor.dtype != torch.float32
                for row in cpu_actions
                for tensor in row.gradient_by_parameter.values()
            ) or any(
                tensor.device.type != "cpu" or tensor.dtype != torch.float32
                for row in cpu_identities
                for tensor in row.probe.gradient_by_parameter.values()
            ):
                raise PairV7PhaseA2Error("root solver inputs are not CPU FP32")
            solved = multi.solve_multicondition_common_direction(
                action_conditions=cpu_actions,
                identity_conditions=cpu_identities,
                multicondition_authority_digest=authority_digest,
                validated_measurement_input_receipt=manifests[0],
            )
            transport_receipt = dict(solved.receipt)
            envelope[0] = {
                "ok": True,
                "transport_receipt": transport_receipt,
                "transport_receipt_digest": transport_receipt["receipt_digest"],
                "primary_replication_go": solved.primary_replication_go,
            }
            del solved, cpu_actions, cpu_identities
        except Exception as error:
            envelope[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(envelope, src=0, group=parallel.world_group)
    result = envelope[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        raise PairV7PhaseA2Error(f"rank-zero CPU solve failed: {result}")
    transport_receipt, declared_transport, replication_go = (
        validate_root_solver_result(
            result,
            bank_binding_digest=authority_digest,
            expected_world_input_digest=input_digest,
        )
    )
    distributed_runtime.digest_consensus(
        declared_transport,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="Phase-A2 rank-zero transport result",
    )
    authority = _seal(
        {
            "schema_version": WORLD_SOLVER_AUTHORITY_SCHEMA,
            "world_size": WORLD_SIZE,
            "topology": "WORLD8-DP2xUlysses-SP4",
            "input_consensus": True,
            "input_digest": input_digest,
            "input_consensus_rank_count": WORLD_SIZE,
            "final_toctou_bank_binding_digest": authority_digest,
            "manifest_digest": manifest.manifest_digest,
            "runtime_source_archive_sha256": runtime_archive_digest,
            "runtime_source_revision": runtime_revision,
            "solver_execution_rank": 0,
            "solver_execution_device": "cpu",
            "solver_input_dtype": "torch.float32",
            "solver_internal_geometry_dtype": "torch.float64",
            "solver_execution_count": 1,
            "single_global_direction_solve": True,
            "local_project_then_average": False,
            "transport_receipt_digest": declared_transport,
            "result_consensus": True,
            "raw_gradient_artifact_written": False,
            "safe_direction_artifact_written": False,
            "phase_b_requires_independent_remeasurement": True,
            "phase_b_must_apply_remeasured_direction_in_memory": True,
            "receipt_can_reconstruct_safe_direction": False,
            "parameter_mutation_performed": False,
        }
    )
    distributed_runtime.digest_consensus(
        authority["receipt_digest"],
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="Phase-A2 root CPU solver authority",
    )
    return PhaseA2WorldSolve(
        primary_replication_go=replication_go,
        transport_receipt=dict(transport_receipt),
        authority_receipt=authority,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--multicondition-preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--cast-method-archive", required=True)
    parser.add_argument("--cast-method-archive-sha256", required=True)
    parser.add_argument("--cast-method-revision", required=True)
    parser.add_argument("--cast-root-spec", required=True)
    parser.add_argument("--expected-cast-root-spec-sha256", required=True)
    parser.add_argument("--scorer-group-receipt", action="append", required=True)
    parser.add_argument(
        "--expected-scorer-group-receipt-sha256", action="append", required=True
    )
    parser.add_argument("--runtime-source-archive", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--ack-root-reviewed-phase-a2-launch", action="store_true")
    parser.add_argument(
        "--ack-no-parameter-mutation-no-success-claim", action="store_true"
    )
    return parser


def _events_for_pair(pre: PhaseA2Preflight, pair_id: str) -> tuple[PhaseA2RuntimeEvent, ...]:
    rows = tuple(row for row in pre.runtime_events if row.spec.pair_id == pair_id)
    rows = tuple(sorted(rows, key=lambda row: row.spec.dp_arm))
    if len(rows) != 2 or [row.spec.dp_arm for row in rows] != [0, 1]:
        raise PairV7PhaseA2Error(f"{pair_id} DP2 runtime event closure differs")
    return rows


def _source_receipt_consensus(
    *, pair_id: str, local: Mapping[str, Any], parallel: Any
) -> list[Mapping[str, Any]]:
    import torch.distributed as dist

    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, dict(local), group=parallel.world_group)
    result: list[Mapping[str, Any]] = []
    for arm in range(DP_SIZE):
        arm_rows = sorted(
            (row for row in rows if isinstance(row, Mapping) and row.get("arm_index") == arm),
            key=lambda row: int(row.get("sp_rank", -1)),
        )
        if (
            len(arm_rows) != SP_SIZE
            or [row.get("sp_rank") for row in arm_rows] != list(range(SP_SIZE))
            or len({object_sha256(row.get("source_receipt")) for row in arm_rows}) != 1
            or any(row.get("pair_id") != pair_id for row in arm_rows)
        ):
            raise PairV7PhaseA2Error(f"{pair_id} WORLD source consensus differs")
        result.append(
            {
                "pair_id": pair_id,
                "arm_index": arm,
                "source_sample_id": arm_rows[0]["source_sample_id"],
                "sp4_receipt_consensus": True,
                "source_receipt": arm_rows[0]["source_receipt"],
            }
        )
    return result


def _publish_receipt_only(stage: Path, output: Path, receipt: Mapping[str, Any]) -> None:
    import source_self_runtime as distributed_runtime

    distributed_runtime.atomic_json(stage / "receipt.json", receipt)
    entries = list(stage.iterdir())
    if len(entries) != 1 or entries[0].name != "receipt.json" or entries[0].is_symlink():
        raise PairV7PhaseA2Error("Phase-A2 output artifact closure differs")
    os.replace(stage, output)
    distributed_runtime.fsync_directory(output.parent)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    pre = preflight(args)
    import audit_pair_v7_phase_a_geometry as phase_a
    import infer_source_kv_carrier_oracle as checkpoint_audit

    bernini_root, veomni_root, source_tree_pre = phase_a._validate_and_bind_source_trees(args)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_only": True,
                    "phase_a2_read_only_measurement_authorized": True,
                    "preregistration_alone_authorized": False,
                    "live_cast_bank_binding_digest": (
                        pre.bank_binding.measurement_authority_digest
                    ),
                    "topology": "WORLD8-DP2xSP4",
                    "primary_pair_ids": list(PAIR_IDS),
                    "primary_schedule_indices": list(SCHEDULE_INDICES),
                    "action_condition_count": EXPECTED_ACTION_COUNT,
                    "identity_probe_count": EXPECTED_IDENTITY_COUNT,
                    "parameter_update_authorized": False,
                    "scientific_action_editing_success_claim": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    import train_lora as legacy

    legacy.activate_source_trees(bernini_root, veomni_root)
    import torch
    import torch.distributed as dist
    from diffusers.models import AutoencoderKLWan
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    import source_self_runtime as distributed_runtime
    import infer_lora as deployment_infer
    import pair_v5_action_adapter as action_adapter
    import pair_v5_t2v_guidance_distill as cagd
    import pair_v7_dual_coordinate_nullspace_transport as core
    import pair_v7_multicondition_nullspace_transport as multi

    # This program is a receipt-only observation pass.  Its SP4 groups remain
    # node-local while the two DP arms may occupy two nodes with four visible
    # devices each; source-self training callers retain the stricter 1x8
    # default in the shared runtime.
    contract = distributed_runtime.distributed_contract(
        allow_multinode_dp2_sp4=True
    )
    device = distributed_runtime.initialise_distributed(contract)
    parallel = distributed_runtime.validate_parallel_state(
        contract, init_parallel_state(ulysses_size=SP_SIZE)
    )
    output, stage = distributed_runtime.prepare_output_transaction(
        args.output, contract.rank, parallel.world_group
    )
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    renderer_config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **deployment_infer.inference_renderer_config_overrides(checkpoint),
    )
    renderer_config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(renderer_config.to_dict(), checkpoint)
    if float(renderer_config.shift) != DEPLOYMENT_FLOW_SHIFT or renderer_config.use_unipc is not True:
        raise PairV7PhaseA2Error("renderer is not deployment UniPC shift 5")
    renderer = BerniniRendererModel(renderer_config).requires_grad_(False).eval()
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise PairV7PhaseA2Error("Phase-A2 requires one Bernini-R 1.3B expert")
    phase_a._disable_gradient_checkpointing(renderer, transformer)
    if next(transformer.parameters()).device.type != "cpu":
        raise PairV7PhaseA2Error("Action-LoRA must initialize deterministically on CPU")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(phase_a.FIXED_ACTION_LORA_INIT_SEED)
        action_handle = action_adapter.install_pair_v5_action_adapter(transformer)
    gauge = phase_a.configure_fixed_a_b_only_gauge(action_handle)
    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    # Keep the VAE and 1.3B renderer out of GPU memory at the same time.  Each
    # rank needs only its DP arm's two source latents; exact81 latents are small
    # enough to park losslessly as detached CPU FP32 between the two pair waves.
    encoded_sources: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    for pair_id in PAIR_IDS:
        local_spec = _events_for_pair(pre, pair_id)[contract.arm_index].spec
        encoded, encoded_receipt = phase_a._encode_source_video(
            local_spec.source_video.path,
            local_spec.source_video.sha256,
            vae=vae,
            device=device,
            parallel=parallel,
        )
        encoded_sources[pair_id] = (
            encoded.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            dict(encoded_receipt),
        )
        del encoded
        torch.cuda.empty_cache()
    del vae
    torch.cuda.empty_cache()
    renderer.to(device).eval()
    gauge = phase_a._validate_fixed_gauge_consensus(gauge, parallel=parallel)
    parameter_state = gauge.initial_full_state_sha256
    checkpoint_receipt = object_sha256(pre.checkpoint_identity)
    source_noise_contract = preregistration.validate_preregistration(
        plan_path=pre.manifest.path,
        expected_plan_file_sha256=pre.manifest.raw_sha256,
    )["source_noise_contract"]
    measured_actions: list[Any] = []
    measured_identities: list[Any] = []
    action_metadata: list[Mapping[str, Any]] = []
    identity_metadata: list[Mapping[str, Any]] = []
    source_receipts: list[Mapping[str, Any]] = []
    local_provenance: list[Mapping[str, Any]] = []

    for pair_id in PAIR_IDS:
        pair_events = _events_for_pair(pre, pair_id)
        local_event = pair_events[contract.arm_index]
        spec = local_event.spec
        source_cpu, source_receipt = encoded_sources[pair_id]
        source = source_cpu.to(device=device, dtype=torch.float32).contiguous()
        (
            native_conditions,
            unconditional,
            rebuilt_prompts,
            _deployment_prompts,
            prompt_receipt,
        ) = phase_a._encode_native_conditions(
            renderer,
            tokenizer,
            spec.raw_caption_by_branch,
            device=device,
            parallel=parallel,
        )
        if rebuilt_prompts != dict(spec.prompt_by_branch):
            raise PairV7PhaseA2Error("runtime prompt reconstruction changed")
        action_embeddings = phase_a._encode_action_prompt_bank(
            renderer=renderer,
            tokenizer=tokenizer,
            prompt_by_branch=spec.prompt_by_branch,
            device=device,
            parallel=parallel,
        )
        source_epsilon, source_noise_receipt = _fresh_source_epsilon(
            source.shape,
            source_sample_id=spec.source_sample_id,
            source_noise_contract=source_noise_contract,
            device=device,
        )
        phase_a._broadcast_sp(source_epsilon, parallel=parallel)
        source_receipts.extend(
            _source_receipt_consensus(
                pair_id=pair_id,
                local={
                    "pair_id": pair_id,
                    "arm_index": contract.arm_index,
                    "sp_rank": contract.sp_rank,
                    "source_sample_id": spec.source_sample_id,
                    "source_receipt": dict(source_receipt),
                },
                parallel=parallel,
            )
        )
        for schedule_index in SCHEDULE_INDICES:
            local_action, action_bundle = extract_phase_a2_action_gradient(
                runtime_event=local_event,
                manifest=pre.manifest,
                bank_binding_digest=pre.bank_binding.measurement_authority_digest,
                diffusion=diffusion,
                transformer=transformer,
                action_handle=action_handle,
                conditions=action_embeddings,
                gauge=gauge,
                parallel=parallel,
                sp_rank=contract.sp_rank,
                schedule_index=schedule_index,
                device=device,
                checkpoint_content_receipt_digest=checkpoint_receipt,
                parameter_state_sha256=parameter_state,
            )
            phase_a._clear_gauge_gradients(gauge)
            torch.cuda.empty_cache()
            action_pair = phase_a._exchange_named_mapping_dp2(
                local_action, parallel=parallel, label="Phase-A2 action"
            )
            local_action_meta = {
                "arm_index": contract.arm_index,
                "condition_id": f"{pair_id}.{spec.source_sample_id}.s{schedule_index}",
                "pair_id": pair_id,
                "source_sample_id": spec.source_sample_id,
                "schedule_index": schedule_index,
                "candidate_id": spec.event_id,
                "action_family": spec.action_family,
                "event_digest": spec.event_digest,
                "gradient_sha256": action_bundle["sp4_bundle"]["averaged_gradient_sha256"],
                "gradient_computation_receipt_digest": action_bundle["sp4_bundle"]["receipt_digest"],
                "sp4_bundle": action_bundle["sp4_bundle"],
                "checkpoint_content_receipt_digest": checkpoint_receipt,
                "parameter_state_sha256": parameter_state,
            }
            action_meta_pair = phase_a._exchange_metadata_dp2(
                local_action_meta, parallel=parallel, label="Phase-A2 action"
            )
            for arm in range(DP_SIZE):
                meta = action_meta_pair[arm]
                if phase_a._named_gradient_sha256(action_pair[arm]) != meta["gradient_sha256"]:
                    raise PairV7PhaseA2Error("exchanged action gradient digest differs")
                measured_actions.append(
                    multi.ActionConditionGradient(
                        condition_id=meta["condition_id"],
                        pair_id=meta["pair_id"],
                        source_sample_id=meta["source_sample_id"],
                        schedule_index=meta["schedule_index"],
                        candidate_id=meta["candidate_id"],
                        action_family=meta["action_family"],
                        event_digest=meta["event_digest"],
                        gradient_computation_receipt_digest=meta[
                            "gradient_computation_receipt_digest"
                        ],
                        gradient_by_parameter=action_pair[arm],
                        checkpoint_content_receipt_digest=checkpoint_receipt,
                        parameter_state_sha256=parameter_state,
                    )
                )
                action_metadata.append(meta)

            coordinate = build_phase_a2_source_coordinate(
                source,
                source_epsilon,
                schedule_index=schedule_index,
                sample_id=spec.source_sample_id,
                bank_binding_digest=pre.bank_binding.measurement_authority_digest,
            )
            feature_runtime = build_phase_a2_feature_runtime(
                diffusion=diffusion,
                transformer=transformer,
                action_handle=action_handle,
                correct_source=source,
                coordinate=coordinate,
                condition_by_branch=native_conditions,
                unconditional=unconditional,
                sp_rank=contract.sp_rank,
                bank_binding_digest=pre.bank_binding.measurement_authority_digest,
            )
            for family in IDENTITY_FAMILIES:
                for sketch_index in range(SKETCH_COUNT):
                    local_gradient, local_bundle = phase_a.extract_identity_probe_gradient(
                        family=family,
                        sketch_index=sketch_index,
                        sample_id=spec.source_sample_id,
                        runtime=feature_runtime,
                        gauge=gauge,
                        parallel=parallel,
                        checkpoint_content_receipt_digest=checkpoint_receipt,
                        parameter_state_sha256=parameter_state,
                    )
                    phase_a._clear_gauge_gradients(gauge)
                    torch.cuda.empty_cache()
                    gradient_pair = phase_a._exchange_named_mapping_dp2(
                        local_gradient,
                        parallel=parallel,
                        label=f"Phase-A2 identity {family} k{sketch_index}",
                    )
                    rank_receipt = local_bundle["rank_receipt"]
                    bundle = local_bundle["sp4_bundle"]
                    local_meta = {
                        "arm_index": contract.arm_index,
                        "pair_id": pair_id,
                        "source_sample_id": spec.source_sample_id,
                        "schedule_index": schedule_index,
                        "probe_id": (
                            f"{pair_id}.{spec.source_sample_id}.s{schedule_index}."
                            f"{family}.k{sketch_index}"
                        ),
                        "family": family,
                        "sketch_index": sketch_index,
                        "gradient_sha256": bundle["averaged_gradient_sha256"],
                        "feature_sketch_sha256": rank_receipt[
                            "feature_sketch_sha256"
                        ],
                        "source_coordinate_receipt_digest": rank_receipt[
                            "source_coordinate_receipt_digest"
                        ],
                        "gradient_computation_receipt_digest": bundle[
                            "receipt_digest"
                        ],
                        "sp4_bundle": bundle,
                        "checkpoint_content_receipt_digest": checkpoint_receipt,
                        "parameter_state_sha256": parameter_state,
                    }
                    meta_pair = phase_a._exchange_metadata_dp2(
                        local_meta,
                        parallel=parallel,
                        label=f"Phase-A2 identity {family} k{sketch_index}",
                    )
                    for arm in range(DP_SIZE):
                        meta = meta_pair[arm]
                        if phase_a._named_gradient_sha256(gradient_pair[arm]) != meta["gradient_sha256"]:
                            raise PairV7PhaseA2Error("exchanged identity gradient digest differs")
                        probe = core.IdentityGradientProbe(
                            probe_id=meta["probe_id"],
                            family=meta["family"],
                            gradient_by_parameter=gradient_pair[arm],
                            feature_sketch_sha256=meta["feature_sketch_sha256"],
                            source_coordinate_receipt_digest=meta[
                                "source_coordinate_receipt_digest"
                            ],
                            gradient_computation_receipt_digest=meta[
                                "gradient_computation_receipt_digest"
                            ],
                            checkpoint_content_receipt_digest=checkpoint_receipt,
                            parameter_state_sha256=parameter_state,
                        )
                        measured_identities.append(
                            multi.IdentityConditionProbe(
                                pair_id=meta["pair_id"],
                                source_sample_id=meta["source_sample_id"],
                                schedule_index=meta["schedule_index"],
                                sketch_index=meta["sketch_index"],
                                probe=probe,
                            )
                        )
                        identity_metadata.append(meta)
            local_provenance.append(
                {
                    "rank": contract.rank,
                    "arm_index": contract.arm_index,
                    "sp_rank": contract.sp_rank,
                    "pair_id": pair_id,
                    "source_sample_id": spec.source_sample_id,
                    "schedule_index": schedule_index,
                    "source_noise_receipt_digest": source_noise_receipt[
                        "receipt_digest"
                    ],
                    "source_coordinate_receipt_digest": coordinate.receipt[
                        "receipt_digest"
                    ],
                    "identity_deployment_protocol_digest": feature_runtime.deployment_protocol[
                        "receipt_digest"
                    ],
                    "prompt_receipt_digest": prompt_receipt["receipt_digest"],
                }
            )
        del source, source_epsilon, native_conditions, unconditional, action_embeddings
        torch.cuda.empty_cache()

    del tokenizer, encoded_sources
    world_solve = world_rank0_cpu_multicondition_solve(
        action_conditions=measured_actions,
        identity_conditions=measured_identities,
        manifest=pre.manifest,
        bank_binding_digest=pre.bank_binding.measurement_authority_digest,
        checkpoint_content_receipt_digest=checkpoint_receipt,
        parameter_state_sha256=parameter_state,
        runtime_source_archive_sha256=pre.runtime_archive.sha256,
        runtime_source_revision=pre.runtime_revision,
        parallel=parallel,
    )
    phase_a._clear_gauge_gradients(gauge)
    final_parameter_state = core.named_parameter_state_sha256(
        gauge.full_state_mapping()
    )
    if final_parameter_state != parameter_state:
        raise PairV7PhaseA2Error("Phase-A2 mutated Action-LoRA parameters")
    final_checkpoint = checkpoint_audit.validate_checkpoint_content(
        Path(args.checkpoint),
        Path(args.checkpoint_content_manifest),
        expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
    )
    if object_sha256(final_checkpoint) != checkpoint_receipt:
        raise PairV7PhaseA2Error("checkpoint content changed during Phase-A2")
    pre.manifest.assert_unchanged()
    pre.bank_binding.assert_unchanged()
    pre.runtime_archive.assert_unchanged()
    pre.cast_archive.assert_unchanged()
    post_bernini, post_veomni, source_tree_post = phase_a._validate_and_bind_source_trees(args)
    if (
        post_bernini != bernini_root
        or post_veomni != veomni_root
        or source_tree_post != source_tree_pre
    ):
        raise PairV7PhaseA2Error("model source trees changed during Phase-A2")
    gathered_provenance: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_provenance, local_provenance, group=parallel.world_group
    )
    if (
        len(gathered_provenance) != WORLD_SIZE
        or any(not isinstance(rows, list) or len(rows) != 4 for rows in gathered_provenance)
    ):
        raise PairV7PhaseA2Error("WORLD runtime provenance closure differs")
    receipt_unsigned = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "method_name": METHOD_NAME,
        "audit_complete": True,
        "geometry_audit_performed": True,
        "geometry_audit_passed": world_solve.primary_replication_go,
        "primary_replication_go": world_solve.primary_replication_go,
        "optimizer_constructed": False,
        "optimizer_step_called": False,
        "candidate_delta_constructed": False,
        "parameter_add_called": False,
        "parameter_mutation_performed": False,
        "parameter_update_authorized": False,
        "scientific_action_editing_success_claim": False,
        "topology": "WORLD8-DP2xUlysses-SP4",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "primary_pair_ids": list(PAIR_IDS),
        "primary_schedule_indices": list(SCHEDULE_INDICES),
        "action_condition_count": len(measured_actions),
        "identity_probe_count": len(measured_identities),
        "preregistration": {
            "path": str(pre.manifest.path),
            "file_sha256": pre.manifest.raw_sha256,
            "preregistration_digest": pre.manifest.preregistration_digest,
            "preregistration_alone_geometry_measurement_authorized": False,
        },
        "live_cast_bank_binding": dict(pre.bank_binding.receipt),
        "checkpoint": {
            "tree_sha256": args.expected_checkpoint_tree_sha256,
            "content_manifest_sha256": args.expected_checkpoint_content_manifest_sha256,
            "content_receipt_digest": checkpoint_receipt,
            "post_audit_unchanged": True,
        },
        "parameter_state_sha256": parameter_state,
        "post_audit_parameter_state_sha256": final_parameter_state,
        "correct_source_coordinates": {
            "source_pair_count": 2,
            "source_count": 4,
            "source_receipts": source_receipts,
            "deployment_visual_condition": "source_video_only_V",
            "image_reference_count": 0,
            "same_source_epsilon_reused_across_schedules": True,
        },
        "gradient_information_flow": {
            "pure_t2v_action_gradient_count": EXPECTED_ACTION_COUNT,
            "deployment_identity_probe_count": EXPECTED_IDENTITY_COUNT,
            "unprojected_rows_preserved_until_root_solver": True,
            "local_project_then_average": False,
            "mask_flow_pose_track_or_trajectory_used": False,
            "raw_gradient_artifact_written": False,
        },
        "action_gradient_metadata": action_metadata,
        "identity_probe_metadata": identity_metadata,
        "world_solver_authority": world_solve.authority_receipt,
        "multicondition_transport_receipt": world_solve.transport_receipt,
        "phase_b_handoff": {
            "phase_a2_safe_direction_persisted": False,
            "receipt_can_reconstruct_safe_direction": False,
            "if_primary_replication_go_then_next_job_must_remeasure": True,
            "phase_b_must_apply_remeasured_direction_in_memory": True,
            "phase_b_is_separate_root_authorized_job": True,
        },
        "rank_runtime_provenance": gathered_provenance,
        "runtime_source": {
            "revision": pre.runtime_revision,
            "archive_sha256": pre.runtime_archive.sha256,
            "post_audit_unchanged": True,
        },
        "cast_method_source": {
            "revision": pre.cast_revision,
            "archive_sha256": pre.cast_archive.sha256,
            "post_audit_unchanged": True,
        },
        "model_source_trees": {
            "preflight_receipt": source_tree_pre,
            "postflight_receipt": source_tree_post,
            "post_audit_unchanged": True,
        },
    }
    phase_a._assert_world_receipt_field_consensus(
        receipt_unsigned, parallel=parallel
    )
    receipt = _seal(receipt_unsigned)
    distributed_runtime.digest_consensus(
        receipt["receipt_digest"],
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="Phase-A2 final receipt",
    )
    if contract.rank == 0:
        _publish_receipt_only(stage, output, receipt)
        print(
            json.dumps(
                {
                    "audit_complete": True,
                    "primary_replication_go": world_solve.primary_replication_go,
                    "parameter_mutation_performed": False,
                    "receipt": str(output / "receipt.json"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=parallel.world_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_GRADIENT_SCHEMA",
    "BANK_BINDING_SCHEMA",
    "EXPECTED_ACTION_COUNT",
    "EXPECTED_IDENTITY_COUNT",
    "PAIR_IDS",
    "PHASE_A2_RUNTIME_ARCHIVE_REQUIRED",
    "PairV7PhaseA2Error",
    "PhaseA2BankBinding",
    "PhaseA2EventSpec",
    "PhaseA2Manifest",
    "PhaseA2Preflight",
    "PhaseA2WorldSolve",
    "RUN_RECEIPT_SCHEMA",
    "SCHEDULE_INDICES",
    "WORLD_INPUT_SCHEMA",
    "WORLD_SOLVER_AUTHORITY_SCHEMA",
    "bind_plan_events_to_cast_candidates",
    "build_parser",
    "build_phase_a2_action_query",
    "build_phase_a2_feature_runtime",
    "build_phase_a2_measurement_objective",
    "build_phase_a2_source_coordinate",
    "extract_phase_a2_action_gradient",
    "main",
    "preflight",
    "validate_measured_bank_against_binding",
    "validate_cross_family_identity_coordinate_closure",
    "validate_root_solver_result",
    "validate_world_input_consensus",
    "world_rank0_cpu_multicondition_solve",
]

#!/usr/bin/env python3
"""Target-free full644 source and rollout-preference training boundary.

This module deliberately does *not* consume the legacy full644 VAE rows.  Those
rows contain ``[source, synthetic_preview]`` and merely taking element zero in
the optimizer process is too easy to regress.  The only admitted catalogue is
an exact-644 source-only JSON manifest.  Each row contains a source video and
an edit instruction; there is no edited video, clean edited latent, anchor, or
teacher field.

An action update would additionally require a same-source, same-instruction
pair of current-policy rollout trajectories.  A separately qualified verifier
must mark the chosen rollout as passing every event and preservation axis and
the rejected rollout as failing at least one named axis.  Rollout RGB/latents
are trajectory provenance, never flow/epsilon/velocity truth.

The repository's current Bernini/UniPC sampler is deterministic and does not
expose a normalized, differentiable transition log probability.  Production
updates are therefore deliberately fail-closed for every non-empty preference
set.  The loss and a tiny dependency-injected optimizer helper remain available
only as mathematical unit-test utilities; they are not a training seam and do
not emit target-free provenance claims.  No pair has a genuine zero-update
path.  This distinction prevents a caller-supplied tensor expression from
being mistaken for evidence that the real renderer consumed a bound rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, Callable, Iterable, Mapping, NoReturn, Optional, Sequence, Tuple


SOURCE_SCHEMA = "bernini-full644-source-only-catalog-v1"
SOURCE_ROW_SCHEMA = "bernini-full644-source-only-row-v1"
PREFERENCE_SCHEMA = "bernini-full644-target-free-preference-set-v1"
PREFERENCE_PAIR_SCHEMA = "bernini-full644-target-free-preference-pair-v1"
ROLLOUT_SCHEMA = "bernini-full644-policy-rollout-v1"
UPDATE_RECEIPT_SCHEMA = "bernini-full644-target-free-update-receipt-v1"
OBJECTIVE_TEST_RECEIPT_SCHEMA = (
    "bernini-full644-target-free-objective-unit-test-receipt-v1"
)
TRAINING_MODE = "TARGET_FREE_ON_POLICY_PREFERENCE"
PRODUCTION_RUNTIME_READY = False
PRODUCTION_RUNTIME_STATUS = (
    "BLOCKED_STOCHASTIC_BERNINI_TRAJECTORY_RECORDER_AND_REPLAY_NOT_IMPLEMENTED"
)

SOURCE_COUNT = 644
ACTION_FAMILY_COUNT = 28
FRAME_COUNT = 81
FPS = 25.0
TRAJECTORY_STEPS = 40

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")

# These names may not occur anywhere inside a source row or preference pair.
# Root-level negative claims use separate, literal field names and are checked
# against exact values below.
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "target",
        "target_video",
        "target_video_path",
        "target_video_sha256",
        "target_latent",
        "target_posterior",
        "edited_target",
        "synthetic_target",
        "preview_target",
        "paired_target",
        "positive_video",
        "positive_latent",
        "video_vae_latents",
        "teacher_unit",
        "teacher_cache",
        "frozen_velocity",
        "flow_target",
        "velocity_target",
        "epsilon_target",
        "oracle_q",
    }
)

HARD_AXES = (
    "event",
    "participant",
    "ordered_transition",
    "terminal_hold",
    "identity",
    "camera",
    "background",
    "non_target_motion",
)

SOURCE_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "row_id",
        "group_id",
        "action_family",
        "source_video_path",
        "source_video_sha256",
        "source_frame_count",
        "source_fps",
        "instruction",
        "instruction_sha256",
        "upstream_preview_row_digest",
        "row_digest",
    }
)
SOURCE_INPUT_ROW_FIELDS = SOURCE_ROW_FIELDS - {"row_digest"}
SOURCE_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "training_mode",
        "source_count",
        "action_family_count",
        "rows",
        "source_authority",
        "row_order",
        "input_closure",
        "manifest_digest",
    }
)
SOURCE_INPUT_CLOSURE = {
    "source_video_only": True,
    "legacy_pair_shard_allowed": False,
    "legacy_target_index1_allowed": False,
    "paired_edited_target_present": False,
    "preview_target_authorized": False,
    "action_signal": "separately_qualified_current_policy_rollout_preference",
}
PINNED_FULL644_SOURCE_AUTHORITY = {
    "upstream_row_schema": "omnivideo2-action-preview-row-v1",
    "preview_manifest_sha256": (
        "49506e003f86f319ebe8a5e843d19c88cef75e84cd4250968da283bb19252e47"
    ),
    "natural_manifest_sha256": (
        "29036f2232af0dfad92e2ae740477285cc7471bc69c54e53c1c8b254ecf3da76"
    ),
    "raw_parquet_sha256": (
        "706d835a8cdf924776000d69b229c272fd434a91abc8942c67dc6fd7732b7d1b"
    ),
    "sorted_iid_set_sha256": (
        "4909037868c6f2c3f9697595995fba770025cae7a215669f36e9e7ae00b67685"
    ),
    "source_role_join": "iid_to_upstream_row_digest_to_source_video_sha256",
    "preview_target_bytes_authorized": False,
    "source_bytes_scope_only": True,
}

ROLLOUT_FIELDS = frozenset(
    {
        "schema_version",
        "rollout_id",
        "policy_sha256",
        "round_index",
        "seed",
        "source_row_id",
        "source_video_sha256",
        "instruction_sha256",
        "trajectory_receipt_path",
        "trajectory_receipt_sha256",
        "output_media_path",
        "output_media_sha256",
        "verifier_receipt_path",
        "verifier_receipt_sha256",
        "axis_pass",
        "failure_tags",
        "rollout_digest",
    }
)
PAIR_FIELDS = frozenset(
    {
        "schema_version",
        "pair_id",
        "source_row_id",
        "source_video_sha256",
        "instruction_sha256",
        "chosen_rollout",
        "rejected_rollout",
        "pair_digest",
    }
)
PREFERENCE_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "training_mode",
        "behavior_policy_sha256",
        "source_manifest_sha256",
        "source_manifest_digest",
        "round_index",
        "pair_count",
        "pairs",
        "verifier_qualification",
        "input_closure",
        "preference_set_digest",
    }
)
PREFERENCE_INPUT_CLOSURE = {
    "paired_edited_target_present": False,
    "old_target_index1_runtime_access_count": 0,
    "rollout_role": "current_policy_trajectory_only",
    "rollout_clean_latent_as_flow_truth": False,
    "pseudo_output_used_as_target": False,
    "action_reference_pixels_or_latents_consumed": False,
    "frozen_model_optimizer_forward_count": 0,
    "frozen_velocity_or_teacher_cache_read_count": 0,
}
VERIFIER_QUALIFICATION_FIELDS = frozenset(
    {
        "schema_version",
        "verifier_release_sha256",
        "verifier_model_sha256",
        "qualification_set_sha256",
        "independent_from_student",
        "hard_axis_conjunction",
        "scalar_compensation_allowed",
    }
)
VERIFIER_QUALIFICATION_SCHEMA = (
    "bernini-full644-hard-axis-verifier-qualification-v1"
)


class TargetFreeTrainingError(RuntimeError):
    """Raised before ambiguous data or an unauthorized update is admitted."""


def fail(message: str) -> NoReturn:
    raise TargetFreeTrainingError(message)


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
        raise TargetFreeTrainingError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} must be an object")
    actual = set(value)
    if not all(type(key) is str for key in actual) or actual != set(fields):
        fail(
            f"{label} fields differ; missing={sorted(set(fields) - actual)}, "
            f"extra={sorted(actual - set(fields))}"
        )
    return value


def _exact_json_equal(value: Any, expected: Any) -> bool:
    """JSON equality that does not allow ``False == 0`` or ``True == 1``."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(value) == set(expected) and all(
            type(key) is str and _exact_json_equal(value[key], expected[key])
            for key in expected
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact_json_equal(item, reference)
            for item, reference in zip(value, expected)
        )
    return value == expected


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be one lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        fail(f"{label} must be a safe identifier")
    return value


def _natural_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or not 8 <= len(value) <= 4096
        or "\x00" in value
        or not any(character.isalpha() for character in value)
    ):
        fail(f"{label} must be one canonical natural-language string")
    return value


def _absolute_path(value: Any, *, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        fail(f"{label} must be a path string")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or any(
        part in ("", ".", "..") for part in path.parts[1:]
    ):
        fail(f"{label} must be an absolute lexical path")
    return path


def _reject_forbidden_keys(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                fail(f"{label} contains a non-string key")
            normalized = key.lower().replace("-", "_")
            if normalized in FORBIDDEN_PAYLOAD_KEYS:
                fail(f"{label} exposes forbidden field {key!r}")
            _reject_forbidden_keys(item, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, label=f"{label}[{index}]")


def _embedded_digest(value: Mapping[str, Any], field: str, *, label: str) -> str:
    declared = _sha256(value.get(field), label=f"{label} {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if object_sha256(unsigned) != declared:
        fail(f"{label} embedded digest differs")
    return declared


def _strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        fail(f"{label} contains non-finite constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TargetFreeTrainingError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        fail(f"{label} root must be an object")
    return value


def _read_stable_file(path: Path, *, expected_sha256: str, label: str) -> bytes:
    expected = _sha256(expected_sha256, label=f"{label} SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TargetFreeTrainingError(f"{label} cannot be opened safely: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            fail(f"{label} must be one regular nlink1 file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_nlink,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_nlink,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(raw) != int(before.st_size):
            fail(f"{label} changed while held open")
        if hashlib.sha256(raw).hexdigest() != expected:
            fail(f"{label} bytes differ")
        return raw
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class SourceRecordV1:
    row_id: str
    group_id: str
    action_family: str
    source_video_path: Path
    source_video_sha256: str
    instruction: str
    instruction_sha256: str
    upstream_preview_row_digest: str
    row_digest: str

    def receipt(self) -> Mapping[str, Any]:
        return {
            "row_id": self.row_id,
            "group_id": self.group_id,
            "action_family": self.action_family,
            "source_video_path": str(self.source_video_path),
            "source_video_sha256": self.source_video_sha256,
            "instruction_sha256": self.instruction_sha256,
            "upstream_preview_row_digest": self.upstream_preview_row_digest,
            "row_digest": self.row_digest,
        }


@dataclass(frozen=True)
class SourceCatalogV1:
    rows: tuple[SourceRecordV1, ...]
    manifest_digest: str
    manifest_sha256: str
    _row_by_id: Mapping[str, SourceRecordV1] = field(repr=False, compare=False)

    def row(self, row_id: str) -> SourceRecordV1:
        try:
            return self._row_by_id[row_id]
        except KeyError as error:
            raise TargetFreeTrainingError(f"unknown source row {row_id!r}") from error


def _validate_source_row(value: Any) -> SourceRecordV1:
    row = _closed(value, SOURCE_ROW_FIELDS, label="source row")
    _reject_forbidden_keys(row, label="source row")
    if row["schema_version"] != SOURCE_ROW_SCHEMA:
        fail("source row schema differs")
    row_id = _safe_id(row["row_id"], label="source row_id")
    group_id = _safe_id(row["group_id"], label="source group_id")
    family = _safe_id(row["action_family"], label="source action_family")
    source_path = _absolute_path(row["source_video_path"], label="source video")
    source_sha = _sha256(row["source_video_sha256"], label="source video SHA")
    if (
        type(row["source_frame_count"]) is not int
        or row["source_frame_count"] != FRAME_COUNT
        or type(row["source_fps"]) is not float
        or row["source_fps"] != FPS
    ):
        fail("source exact81/25fps contract differs")
    instruction = _natural_text(row["instruction"], label="edit instruction")
    instruction_sha = _sha256(row["instruction_sha256"], label="instruction SHA")
    if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != instruction_sha:
        fail("instruction bytes differ from instruction SHA")
    row_digest = _embedded_digest(row, "row_digest", label=f"source row {row_id}")
    upstream_digest = _sha256(
        row["upstream_preview_row_digest"], label="upstream preview row digest"
    )
    return SourceRecordV1(
        row_id=row_id,
        group_id=group_id,
        action_family=family,
        source_video_path=source_path,
        source_video_sha256=source_sha,
        instruction=instruction,
        instruction_sha256=instruction_sha,
        upstream_preview_row_digest=upstream_digest,
        row_digest=row_digest,
    )


def validate_source_catalog_value(
    value: Any,
    *,
    manifest_sha256: str,
    require_source_files: bool = False,
) -> SourceCatalogV1:
    root = _closed(value, SOURCE_ROOT_FIELDS, label="source catalogue")
    if (
        root["schema_version"] != SOURCE_SCHEMA
        or root["training_mode"] != TRAINING_MODE
        or type(root["source_count"]) is not int
        or root["source_count"] != SOURCE_COUNT
        or type(root["action_family_count"]) is not int
        or root["action_family_count"] != ACTION_FAMILY_COUNT
        or not _exact_json_equal(
            root["source_authority"], PINNED_FULL644_SOURCE_AUTHORITY
        )
        or root["row_order"] != "ascii_ascending_row_id"
        or not _exact_json_equal(root["input_closure"], SOURCE_INPUT_CLOSURE)
    ):
        fail("source catalogue contract differs")
    raw_rows = root["rows"]
    if not isinstance(raw_rows, list) or len(raw_rows) != SOURCE_COUNT:
        fail("source catalogue must contain exact644 rows")
    rows = tuple(_validate_source_row(row) for row in raw_rows)
    if (
        [row.row_id for row in rows] != sorted(row.row_id for row in rows)
        or len({row.upstream_preview_row_digest for row in rows}) != SOURCE_COUNT
        or len({row.row_id for row in rows}) != SOURCE_COUNT
        or len({row.group_id for row in rows}) != SOURCE_COUNT
        or len({row.source_video_sha256 for row in rows}) != SOURCE_COUNT
        or len({row.action_family for row in rows}) != ACTION_FAMILY_COUNT
    ):
        fail("source row/source/family cardinality differs")
    manifest_digest = _embedded_digest(
        root, "manifest_digest", label="source catalogue"
    )
    if require_source_files:
        for row in rows:
            _read_stable_file(
                row.source_video_path,
                expected_sha256=row.source_video_sha256,
                label=f"source video {row.row_id}",
            )
    return SourceCatalogV1(
        rows=rows,
        manifest_digest=manifest_digest,
        manifest_sha256=_sha256(manifest_sha256, label="source manifest SHA"),
        _row_by_id={row.row_id: row for row in rows},
    )


def load_source_catalog(
    path: str | Path,
    *,
    expected_sha256: str,
    require_source_files: bool = False,
) -> SourceCatalogV1:
    requested = _absolute_path(str(path), label="source manifest")
    raw = _read_stable_file(
        requested, expected_sha256=expected_sha256, label="source manifest"
    )
    return validate_source_catalog_value(
        _strict_json(raw, label="source manifest"),
        manifest_sha256=expected_sha256,
        require_source_files=require_source_files,
    )


@dataclass(frozen=True)
class RolloutV1:
    rollout_id: str
    policy_sha256: str
    round_index: int
    seed: int
    source_row_id: str
    source_video_sha256: str
    instruction_sha256: str
    trajectory_receipt_path: Path
    trajectory_receipt_sha256: str
    output_media_path: Path
    output_media_sha256: str
    verifier_receipt_path: Path
    verifier_receipt_sha256: str
    axis_pass: Mapping[str, bool]
    failure_tags: tuple[str, ...]
    rollout_digest: str

    @property
    def passes_all_axes(self) -> bool:
        return all(self.axis_pass[axis] is True for axis in HARD_AXES)


@dataclass(frozen=True)
class PreferencePairV1:
    pair_id: str
    source: SourceRecordV1
    chosen: RolloutV1
    rejected: RolloutV1
    pair_digest: str


@dataclass(frozen=True)
class PreferenceSetV1:
    behavior_policy_sha256: str
    round_index: int
    pairs: tuple[PreferencePairV1, ...]
    preference_set_digest: str
    preference_set_sha256: str
    verifier_qualification: Mapping[str, Any]


def _validate_axis_pass(value: Any, *, label: str) -> Mapping[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(HARD_AXES):
        fail(f"{label} axis closure differs")
    if any(type(value[axis]) is not bool for axis in HARD_AXES):
        fail(f"{label} axis verdicts must be built-in bool")
    return {axis: value[axis] for axis in HARD_AXES}


def _validate_rollout(
    value: Any,
    *,
    behavior_policy_sha256: str,
    round_index: int,
    source: SourceRecordV1,
    label: str,
) -> RolloutV1:
    row = _closed(value, ROLLOUT_FIELDS, label=label)
    _reject_forbidden_keys(row, label=label)
    if row["schema_version"] != ROLLOUT_SCHEMA:
        fail(f"{label} schema differs")
    rollout_id = _safe_id(row["rollout_id"], label=f"{label} rollout_id")
    policy_sha = _sha256(row["policy_sha256"], label=f"{label} policy SHA")
    if (
        policy_sha != behavior_policy_sha256
        or type(row["round_index"]) is not int
        or row["round_index"] != round_index
    ):
        fail(f"{label} is not from the admitted current behavior policy")
    if type(row["seed"]) is not int or row["seed"] < 0 or row["seed"] >= 2**63:
        fail(f"{label} seed differs")
    source_row_id = _safe_id(row["source_row_id"], label=f"{label} source row")
    source_video_sha = _sha256(
        row["source_video_sha256"], label=f"{label} source video SHA"
    )
    instruction_sha = _sha256(
        row["instruction_sha256"], label=f"{label} instruction SHA"
    )
    if (
        source_row_id != source.row_id
        or source_video_sha != source.source_video_sha256
        or instruction_sha != source.instruction_sha256
    ):
        fail(f"{label} source/instruction binding differs")
    failure_tags = row["failure_tags"]
    if (
        not isinstance(failure_tags, list)
        or any(type(item) is not str or _SAFE_ID.fullmatch(item) is None for item in failure_tags)
        or len(set(failure_tags)) != len(failure_tags)
    ):
        fail(f"{label} failure tags differ")
    axis_pass = _validate_axis_pass(row["axis_pass"], label=label)
    failed_axes = [axis for axis in HARD_AXES if axis_pass[axis] is False]
    expected_failure_tags = [f"{axis}_failed" for axis in failed_axes]
    if failure_tags != expected_failure_tags:
        fail(f"{label} failure tags do not exactly name failed axes")
    return RolloutV1(
        rollout_id=rollout_id,
        policy_sha256=policy_sha,
        round_index=round_index,
        seed=row["seed"],
        source_row_id=source_row_id,
        source_video_sha256=source_video_sha,
        instruction_sha256=instruction_sha,
        trajectory_receipt_path=_absolute_path(
            row["trajectory_receipt_path"], label=f"{label} trajectory receipt"
        ),
        trajectory_receipt_sha256=_sha256(
            row["trajectory_receipt_sha256"], label=f"{label} trajectory receipt SHA"
        ),
        output_media_path=_absolute_path(
            row["output_media_path"], label=f"{label} output media"
        ),
        output_media_sha256=_sha256(
            row["output_media_sha256"], label=f"{label} output media SHA"
        ),
        verifier_receipt_path=_absolute_path(
            row["verifier_receipt_path"], label=f"{label} verifier receipt"
        ),
        verifier_receipt_sha256=_sha256(
            row["verifier_receipt_sha256"], label=f"{label} verifier receipt SHA"
        ),
        axis_pass=axis_pass,
        failure_tags=tuple(failure_tags),
        rollout_digest=_embedded_digest(row, "rollout_digest", label=label),
    )


def validate_preference_set_value(
    value: Any,
    *,
    source_catalog: SourceCatalogV1,
    preference_set_sha256: str,
) -> PreferenceSetV1:
    root = _closed(value, PREFERENCE_ROOT_FIELDS, label="preference set")
    if (
        root["schema_version"] != PREFERENCE_SCHEMA
        or root["training_mode"] != TRAINING_MODE
        or not _exact_json_equal(root["input_closure"], PREFERENCE_INPUT_CLOSURE)
    ):
        fail("preference-set target-free closure differs")
    behavior_sha = _sha256(
        root["behavior_policy_sha256"], label="behavior policy SHA"
    )
    source_manifest_sha = _sha256(
        root["source_manifest_sha256"], label="preference source manifest SHA"
    )
    source_manifest_digest = _sha256(
        root["source_manifest_digest"], label="preference source manifest digest"
    )
    if (
        source_manifest_sha != source_catalog.manifest_sha256
        or source_manifest_digest != source_catalog.manifest_digest
    ):
        fail("preference-set source catalogue binding differs")
    round_index = root["round_index"]
    if type(round_index) is not int or round_index < 0:
        fail("preference-set round_index differs")
    qualification = _closed(
        root["verifier_qualification"],
        VERIFIER_QUALIFICATION_FIELDS,
        label="verifier qualification",
    )
    _reject_forbidden_keys(qualification, label="verifier qualification")
    if (
        qualification["schema_version"] != VERIFIER_QUALIFICATION_SCHEMA
        or _sha256(
            qualification["verifier_release_sha256"],
            label="verifier release SHA",
        )
        != qualification["verifier_release_sha256"]
        or _sha256(
            qualification["verifier_model_sha256"],
            label="verifier model SHA",
        )
        != qualification["verifier_model_sha256"]
        or _sha256(
            qualification["qualification_set_sha256"],
            label="verifier qualification-set SHA",
        )
        != qualification["qualification_set_sha256"]
        or qualification["independent_from_student"] is not True
        or not _exact_json_equal(
            qualification["hard_axis_conjunction"], list(HARD_AXES)
        )
        or qualification["scalar_compensation_allowed"] is not False
    ):
        fail("preference verifier qualification differs")
    raw_pairs = root["pairs"]
    if (
        not isinstance(raw_pairs, list)
        or type(root["pair_count"]) is not int
        or root["pair_count"] != len(raw_pairs)
        or len(raw_pairs) > SOURCE_COUNT * 4
    ):
        fail("preference-set pair count differs")
    pairs: list[PreferencePairV1] = []
    for index, raw_pair in enumerate(raw_pairs):
        pair = _closed(raw_pair, PAIR_FIELDS, label=f"preference pair {index}")
        _reject_forbidden_keys(pair, label=f"preference pair {index}")
        if pair["schema_version"] != PREFERENCE_PAIR_SCHEMA:
            fail("preference pair schema differs")
        pair_id = _safe_id(pair["pair_id"], label="preference pair_id")
        source = source_catalog.row(_safe_id(pair["source_row_id"], label="pair source row"))
        pair_source_sha = _sha256(
            pair["source_video_sha256"], label="pair source video SHA"
        )
        pair_instruction_sha = _sha256(
            pair["instruction_sha256"], label="pair instruction SHA"
        )
        if (
            pair_source_sha != source.source_video_sha256
            or pair_instruction_sha != source.instruction_sha256
        ):
            fail(f"preference pair {pair_id} source/instruction binding differs")
        chosen = _validate_rollout(
            pair["chosen_rollout"],
            behavior_policy_sha256=behavior_sha,
            round_index=round_index,
            source=source,
            label=f"{pair_id} chosen rollout",
        )
        rejected = _validate_rollout(
            pair["rejected_rollout"],
            behavior_policy_sha256=behavior_sha,
            round_index=round_index,
            source=source,
            label=f"{pair_id} rejected rollout",
        )
        if (
            chosen.rollout_id == rejected.rollout_id
            or chosen.output_media_sha256 == rejected.output_media_sha256
        ):
            fail(f"preference pair {pair_id} endpoints are identical")
        if not chosen.passes_all_axes or chosen.failure_tags:
            fail(f"preference pair {pair_id} chosen rollout did not pass every hard axis")
        if rejected.passes_all_axes or not rejected.failure_tags:
            fail(f"preference pair {pair_id} rejected rollout has no named hard failure")
        pair_digest = _embedded_digest(pair, "pair_digest", label=f"pair {pair_id}")
        pairs.append(
            PreferencePairV1(
                pair_id=pair_id,
                source=source,
                chosen=chosen,
                rejected=rejected,
                pair_digest=pair_digest,
            )
        )
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        fail("preference pair IDs are not unique")
    if len({(pair.chosen.rollout_id, pair.rejected.rollout_id) for pair in pairs}) != len(pairs):
        fail("preference endpoint pairs are duplicated")
    all_rollouts = [
        rollout
        for pair in pairs
        for rollout in (pair.chosen, pair.rejected)
    ]
    if (
        len({rollout.rollout_id for rollout in all_rollouts}) != len(all_rollouts)
        or len({rollout.rollout_digest for rollout in all_rollouts})
        != len(all_rollouts)
        or len({rollout.output_media_sha256 for rollout in all_rollouts})
        != len(all_rollouts)
    ):
        fail("preference rollout identity/media inventory is not unique")
    digest = _embedded_digest(root, "preference_set_digest", label="preference set")
    return PreferenceSetV1(
        behavior_policy_sha256=behavior_sha,
        round_index=round_index,
        pairs=tuple(pairs),
        preference_set_digest=digest,
        preference_set_sha256=_sha256(
            preference_set_sha256, label="preference-set file SHA"
        ),
        verifier_qualification=dict(qualification),
    )


def load_preference_set(
    path: str | Path,
    *,
    expected_sha256: str,
    source_catalog: SourceCatalogV1,
    require_rollout_files: bool = True,
) -> PreferenceSetV1:
    requested = _absolute_path(str(path), label="preference-set manifest")
    raw = _read_stable_file(
        requested, expected_sha256=expected_sha256, label="preference-set manifest"
    )
    preference_set = validate_preference_set_value(
        _strict_json(raw, label="preference-set manifest"),
        source_catalog=source_catalog,
        preference_set_sha256=expected_sha256,
    )
    if require_rollout_files:
        for pair in preference_set.pairs:
            for role, rollout in (("chosen", pair.chosen), ("rejected", pair.rejected)):
                _read_stable_file(
                    rollout.trajectory_receipt_path,
                    expected_sha256=rollout.trajectory_receipt_sha256,
                    label=f"{pair.pair_id} {role} trajectory receipt",
                )
                _read_stable_file(
                    rollout.output_media_path,
                    expected_sha256=rollout.output_media_sha256,
                    label=f"{pair.pair_id} {role} output media",
                )
                _read_stable_file(
                    rollout.verifier_receipt_path,
                    expected_sha256=rollout.verifier_receipt_sha256,
                    label=f"{pair.pair_id} {role} verifier receipt",
                )
    return preference_set


def _pairwise_preference_objective_math_unit_v1(
    chosen_step_log_probs: Any,
    rejected_step_log_probs: Any,
    *,
    beta: float = 1.0,
) -> Any:
    """Bradley-Terry algebra check over synthetic step log probabilities.

    This function has no clean-video or velocity target argument.  Both inputs
    are intentionally not treated as proof of a bound Bernini rollout.
    """

    import torch
    import torch.nn.functional as functional

    chosen = chosen_step_log_probs
    rejected = rejected_step_log_probs
    if (
        type(chosen) is not torch.Tensor
        or type(rejected) is not torch.Tensor
        or chosen.ndim != 2
        or tuple(chosen.shape) != tuple(rejected.shape)
        or int(chosen.shape[0]) < 1
        or int(chosen.shape[1]) != TRAJECTORY_STEPS
        or chosen.dtype != torch.float32
        or rejected.dtype != torch.float32
        or chosen.device != rejected.device
        or not chosen.requires_grad
        or not rejected.requires_grad
        or chosen.grad_fn is None
        or rejected.grad_fn is None
        or not bool(torch.isfinite(chosen).all().item())
        or not bool(torch.isfinite(rejected).all().item())
    ):
        fail("trajectory log probabilities must be finite trainable FP32 [B,40]")
    if type(beta) is not float or not math.isfinite(beta) or beta <= 0.0:
        fail("preference beta must be one positive finite built-in float")
    margin = chosen.sum(dim=1) - rejected.sum(dim=1)
    loss = functional.softplus(-beta * margin).mean()
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        fail("target-free preference loss is non-finite")
    return loss


def _tensor_bytes(value: Any) -> bytes:
    import torch

    if not isinstance(value, torch.Tensor):
        fail("trainable inventory contains a non-tensor")
    cpu = value.detach().to(device="cpu").contiguous()
    storage = cpu.untyped_storage()
    expected = int(cpu.numel()) * int(cpu.element_size())
    if int(storage.nbytes()) != expected or int(cpu.storage_offset()) != 0:
        cpu = cpu.clone(memory_format=torch.contiguous_format)
        storage = cpu.untyped_storage()
    raw = bytes(storage)
    if len(raw) != expected:
        fail("trainable tensor byte count differs")
    return raw


def parameter_digest(parameters: Sequence[Any]) -> str:
    digest = hashlib.sha256(b"full644-target-free-parameters-v1\x00")
    for index, parameter in enumerate(parameters):
        metadata = canonical_json_bytes(
            {
                "index": index,
                "dtype": str(parameter.dtype),
                "shape": [int(item) for item in parameter.shape],
            }
        )
        payload = _tensor_bytes(parameter)
        digest.update(struct.pack(">Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


LogprobProvider = Callable[[PreferencePairV1], Tuple[Any, Any]]
OptimizerFactory = Callable[[Sequence[Any]], Any]


def run_target_free_preference_update(
    *,
    preference_set: PreferenceSetV1,
    student_before_sha256: str,
    trainable_parameters: Iterable[Any],
) -> Mapping[str, Any]:
    """Return a genuine zero-update or reject until Bernini replay is closed.

    In particular this public production boundary never accepts an optimizer
    factory or a log-probability callback.  Such callables cannot prove that
    their tensors came from the held rollout, current student, source, or
    instruction.  A non-empty preference set therefore fails before optimizer
    construction.  The eventual implementation must own stochastic rollout
    recording, exact40 streaming replay, SP/DP reduction, and checkpointing.
    """

    import torch

    if type(preference_set) is not PreferenceSetV1:
        fail("production update requires one closed PreferenceSetV1 loader result")
    student_sha = _sha256(student_before_sha256, label="student-before SHA")
    if student_sha != preference_set.behavior_policy_sha256:
        fail("behavior policy is not the exact pre-update student")
    if preference_set.pairs:
        fail(PRODUCTION_RUNTIME_STATUS)
    parameters = tuple(trainable_parameters)
    if not parameters or any(
        not isinstance(parameter, torch.Tensor) or not parameter.requires_grad
        for parameter in parameters
    ):
        fail("trainable parameter inventory differs")
    before = parameter_digest(parameters)
    receipt = {
        "schema_version": UPDATE_RECEIPT_SCHEMA,
        "status": "ZERO_UPDATE_NO_QUALIFIED_PAIR",
        "training_mode": TRAINING_MODE,
        "authoritative_training_receipt": False,
        "optimizer_authorized": False,
        "scientific_result_claimed": False,
        "production_runtime_ready": False,
        "production_runtime_status": PRODUCTION_RUNTIME_STATUS,
        "preference_set_sha256": preference_set.preference_set_sha256,
        "preference_set_digest": preference_set.preference_set_digest,
        "behavior_policy_sha256": preference_set.behavior_policy_sha256,
        "round_index": preference_set.round_index,
        "pair_count": 0,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "parameter_digest_before": before,
        "parameter_digest_after": before,
        "function_owned_data_file_open_count": 0,
    }
    return {**receipt, "receipt_digest": object_sha256(receipt)}


def _run_objective_unit_test_update_v1(
    *,
    preference_set: PreferenceSetV1,
    trainable_parameters: Iterable[Any],
    optimizer_factory: OptimizerFactory,
    logprob_provider: LogprobProvider,
    beta: float = 1.0,
) -> Mapping[str, Any]:
    """Exercise only the preference-loss sign in a tiny synthetic unit test.

    This helper intentionally emits no claims about target access, current
    policy identity, rollout provenance, distributed gradients, or production
    readiness.  It is private so it cannot be confused with the public
    fail-closed runtime boundary above.
    """

    import torch

    if type(preference_set) is not PreferenceSetV1 or not preference_set.pairs:
        fail("objective unit test requires one loaded non-empty preference set")
    parameters = tuple(trainable_parameters)
    if not parameters or any(
        not isinstance(parameter, torch.Tensor) or not parameter.requires_grad
        for parameter in parameters
    ):
        fail("objective-test trainable parameter inventory differs")
    before = parameter_digest(parameters)
    optimizer = optimizer_factory(parameters)
    if optimizer is None or not callable(getattr(optimizer, "zero_grad", None)):
        fail("objective-test optimizer factory did not return an optimizer")
    optimizer.zero_grad(set_to_none=True)
    chosen_rows: list[Any] = []
    rejected_rows: list[Any] = []
    for pair in preference_set.pairs:
        result = logprob_provider(pair)
        if not isinstance(result, tuple) or len(result) != 2:
            fail("objective-test provider must return chosen/rejected tensors")
        chosen, rejected = result
        if type(chosen) is not torch.Tensor or type(rejected) is not torch.Tensor:
            fail("objective-test provider returned a non-tensor")
        if tuple(chosen.shape) != (TRAJECTORY_STEPS,) or tuple(rejected.shape) != (
            TRAJECTORY_STEPS,
        ):
            fail("each trajectory must provide exact40 step log probabilities")
        chosen_rows.append(chosen)
        rejected_rows.append(rejected)
    chosen_batch = torch.stack(chosen_rows, dim=0)
    rejected_batch = torch.stack(rejected_rows, dim=0)
    loss = _pairwise_preference_objective_math_unit_v1(
        chosen_batch, rejected_batch, beta=beta
    )
    loss.backward()
    gradient_count = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all().item()):
            fail("target-free preference gradient is non-finite")
        if bool((gradient != 0).any().item()):
            gradient_count += 1
    if gradient_count == 0:
        fail("objective unit test produced no nonzero trainable gradient")
    optimizer.step()
    after = parameter_digest(parameters)
    if after == before:
        fail("objective unit-test optimizer did not change parameter bytes")
    receipt = {
        "schema_version": OBJECTIVE_TEST_RECEIPT_SCHEMA,
        "status": "OBJECTIVE_UNIT_TEST_UPDATE_COMPLETE_NOT_PRODUCTION",
        "production_runtime_ready": False,
        "pair_count": len(preference_set.pairs),
        "optimizer_constructed": True,
        "optimizer_step_executed": True,
        "loss": float(loss.detach().item()),
        "nonzero_gradient_tensor_count": gradient_count,
        "parameter_digest_before": before,
        "parameter_digest_after": after,
    }
    return {**receipt, "receipt_digest": object_sha256(receipt)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--preference-set", required=True)
    parser.add_argument("--preference-set-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    source = load_source_catalog(
        args.source_manifest,
        expected_sha256=args.source_manifest_sha256,
        require_source_files=True,
    )
    preferences = load_preference_set(
        args.preference_set,
        expected_sha256=args.preference_set_sha256,
        source_catalog=source,
        require_rollout_files=True,
    )
    result = {
        "status": (
            "TARGET_FREE_NONEMPTY_PREFERENCE_RUNTIME_BLOCKED"
            if preferences.pairs
            else "TARGET_FREE_ZERO_UPDATE_INPUTS_VALID"
        ),
        "training_mode": TRAINING_MODE,
        "source_count": len(source.rows),
        "action_family_count": len({row.action_family for row in source.rows}),
        "candidate_pair_count": len(preferences.pairs),
        "optimizer_authorized": False,
        "production_runtime_ready": PRODUCTION_RUNTIME_READY,
        "production_runtime_status": PRODUCTION_RUNTIME_STATUS,
        "paired_target_fields_admitted_by_schema": False,
        "target_byte_nonaccess_claim_authorized": False,
        "source_manifest_digest": source.manifest_digest,
        "preference_set_digest": preferences.preference_set_digest,
    }
    print(canonical_json_bytes(result).decode("ascii"))
    return 2 if preferences.pairs else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed Stage-A admission, exact80 plan, and checkpoint loader.

This module deliberately cannot create a Stage-A admission receipt.  It only
accepts a separately produced, SHA-pinned decoded-intervention runtime receipt
and its explicit manual admission.  The training entry point calls this
validator before constructing an optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, NoReturn, Optional, Sequence

import inference_sigma_strata as exact40


SCHEMA_VERSION = "bernini-clean-source-visual-context-stage-b-contract-v1"
STAGE_A_RUNTIME_SCHEMA = "bernini-schedule-block-source-edge-decoded-runtime-v2"
STAGE_A_ADMISSION_SCHEMA = (
    "bernini-clean-source-visual-context-stage-a-admission-v1"
)
CHECKPOINT_SCHEMA = "bernini-clean-source-visual-context-checkpoint-v1"
CHECKPOINT_STEPS = (0, 20, 40, 60, 80)
OPTIMIZER_STEPS = 80
GRADIENT_ACCUMULATION_STEPS = 4
PHYSICAL_DP_SIZE = 2
GLOBAL_BATCH = GRADIENT_ACCUMULATION_STEPS * PHYSICAL_DP_SIZE
MICROBATCHES_PER_DP_ARM = OPTIMIZER_STEPS * GRADIENT_ACCUMULATION_STEPS
LOGICAL_RECORDS = OPTIMIZER_STEPS * GLOBAL_BATCH
TRAIN_ROWS = 64
EXPECTED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
EXPECTED_MODEL_REVISION = "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
EXPECTED_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
EXPECTED_SCHEDULE_SHA256 = (
    "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
)
PREREGISTERED_SPARSE_BLOCK_INDICES = (8, 12, 16, 20)
STAGE_A_SCHEDULE_INDICES = (16, 29, 35, 38)
STAGE_A_REQUIRED_BLOCK_BANDS = (
    "early_middle",
    "late_middle",
)
STAGE_A_BLOCKS_BY_BAND = {
    "early": tuple(range(0, 8)),
    "early_middle": tuple(range(8, 16)),
    "late_middle": tuple(range(16, 23)),
    "late": tuple(range(23, 30)),
}
PREREGISTERED_SPARSE_BLOCKS_BY_BAND = {
    "early_middle": (8, 12),
    "late_middle": (16, 20),
}
STAGE_A_REQUIRED_FAMILIES = ("dog", "human")
STAGE_A_TEXT_BRANCHES = (
    "forward",
    "noop",
    "reverse",
    "incomplete",
    "camera_only",
    "appearance_only",
)
STAGE_A_OUTPUTS_PER_REQUIRED_FAMILY = 56
EXPECTED_STAGE_A_METHOD = "frozen-target-query-source-kv-edge-causal-localization-v2"
EXPECTED_STAGE_A_POLICY_DIGEST = (
    "dfac73238ad8d560bb31178d5cd0775e1a5924377a8799e7768024c2ea8a7c51"
)
EXPECTED_STAGE_A_INTERVENTION_DIGEST = (
    "a88f478f4b10e1cbf6f31b9fa2dfdd3ff0341c024437e4d3c7fb163f3dce7715"
)
EXPECTED_STAGE_A_FULL_GRID_DIGEST = (
    "af3ba9615b737d8a8f506bf532649e27c200b4927ec90dd0b173472997eb658a"
)
EXPECTED_VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
MEMORY_INPUT_KINDS = (
    "clean_source",
    "same_noise_forward_noised_source",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CleanSourceVisualStageBContractError(RuntimeError):
    """Raised before unadmitted training/checkpoint state can be consumed."""


def fail(message: str) -> NoReturn:
    raise CleanSourceVisualStageBContractError(message)


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
        raise CleanSourceVisualStageBContractError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    named = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not stat.S_ISREG(resolved.lstat().st_mode):
        fail(f"{label} must be one canonical plain file")
    return resolved


def _strict_json_bytes(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        fail(f"{label} contains non-finite constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CleanSourceVisualStageBContractError(
            f"cannot decode {label}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        fail(f"{label} must contain one object")
    return value


def _read_strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        fail(f"{label} changed while reading")
    return _strict_json_bytes(raw, label=label)


def _embedded_digest(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    unsigned = dict(value)
    declared = _sha(unsigned.pop(field, None), label=f"{label} {field}")
    if object_sha256(unsigned) != declared:
        fail(f"{label} embedded digest differs")
    return declared


def _expected_stage_a_plan() -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for branch in STAGE_A_TEXT_BRANCHES:
        rows.append(
            {
                "key": f"native-correct-{branch}",
                "role": "native_correct_prompt_baseline",
                "owner": "correct_owner",
                "text_branch": branch,
                "hook": "native-unhooked",
                "schedule_index": None,
                "band_name": None,
            }
        )
    rows.extend(
        (
            {
                "key": "native-wrong-owner-forward",
                "role": "compatible_wrong_owner_forward_baseline",
                "owner": "wrong_owner",
                "text_branch": "forward",
                "hook": "native-unhooked",
                "schedule_index": None,
                "band_name": None,
            },
            {
                "key": "parity-source-on-s16-early-forward",
                "role": "hooked_source_on_native_parity",
                "owner": "correct_owner",
                "text_branch": "forward",
                "hook": "source-on",
                "schedule_index": 16,
                "band_name": "early",
            },
        )
    )
    for schedule in STAGE_A_SCHEDULE_INDICES:
        for band in STAGE_A_REQUIRED_BLOCK_BANDS:
            for branch in STAGE_A_TEXT_BRANCHES:
                rows.append(
                    {
                        "key": f"off-s{schedule:02d}-{band}-{branch}",
                        "role": "source_edge_off_cell",
                        "owner": "correct_owner",
                        "text_branch": branch,
                        "hook": "source-off",
                        "schedule_index": schedule,
                        "band_name": band,
                    }
                )
    if len(rows) != STAGE_A_OUTPUTS_PER_REQUIRED_FAMILY:
        fail("Stage-A required family plan count differs")
    return tuple(rows)


def validate_stage_a_runtime_receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate one real A1 v2 dog/human decoded family receipt.

    This consumes the existing Stage-A runtime schema.  It accepts the minimal
    B-v1 localization shard: all four schedules crossed with the two middle
    bands, plus the v2 native/wrong-owner/source-on controls (56 outputs).
    An early-only 14-output pilot cannot pass this validator.
    """

    required_top = {
        "schema_version",
        "method",
        "stage",
        "registered_schedule_block_policy",
        "intervention_contract",
        "full_grid_contract",
        "shard",
        "authority",
        "runtime_source",
        "pinned_sources",
        "checkpoint",
        "source",
        "prompts",
        "sampling",
        "candidates",
        "traces",
        "generated_identities",
        "outputs",
        "frozen_model",
        "resource_lifetime",
        "runtime_versions",
        "interpretation",
        "receipt_digest",
    }
    if not isinstance(value, Mapping) or not required_top.issubset(value):
        fail("Stage-A A1 runtime receipt fields differ")
    _embedded_digest(value, field="receipt_digest", label="Stage-A A1 runtime")
    policy = value.get("registered_schedule_block_policy")
    intervention = value.get("intervention_contract")
    full_grid = value.get("full_grid_contract")
    shard = value.get("shard")
    pinned = value.get("pinned_sources")
    checkpoint = value.get("checkpoint")
    source = value.get("source")
    sampling = value.get("sampling")
    frozen = value.get("frozen_model")
    interpretation = value.get("interpretation")
    if (
        value.get("schema_version") != STAGE_A_RUNTIME_SCHEMA
        or value.get("method") != EXPECTED_STAGE_A_METHOD
        or value.get("stage") != "preservation_stage_A_decoded_causal_localization"
        or not isinstance(policy, Mapping)
        or policy.get("receipt_digest") != EXPECTED_STAGE_A_POLICY_DIGEST
        or policy.get("schedule_indices") != list(STAGE_A_SCHEDULE_INDICES)
        or policy.get("block_bands")
        != {name: list(blocks) for name, blocks in STAGE_A_BLOCKS_BY_BAND.items()}
        or policy.get("optimizer_authorized") is not False
        or policy.get("parameter_update_authorized") is not False
        or not isinstance(intervention, Mapping)
        or intervention.get("digest") != EXPECTED_STAGE_A_INTERVENTION_DIGEST
        or intervention.get("optimizer") is not False
        or intervention.get("parameter_update") is not False
        or intervention.get("reward") is not False
        or intervention.get("feature_scalar") is not False
        or intervention.get("ranking") is not False
        or intervention.get("selection") is not False
        or not isinstance(full_grid, Mapping)
        or full_grid.get("digest") != EXPECTED_STAGE_A_FULL_GRID_DIGEST
        or not isinstance(shard, Mapping)
        or shard.get("family") not in STAGE_A_REQUIRED_FAMILIES
        or shard.get("schedule_indices") != list(STAGE_A_SCHEDULE_INDICES)
        or shard.get("block_bands") != list(STAGE_A_REQUIRED_BLOCK_BANDS)
        or shard.get("full_registered_grid") is not False
        or shard.get("candidate_count") != STAGE_A_OUTPUTS_PER_REQUIRED_FAMILY
        or not isinstance(pinned, Mapping)
        or pinned.get("bernini_commit") != EXPECTED_BERNINI_COMMIT
        or pinned.get("veomni_commit") != EXPECTED_VEOMNI_COMMIT
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("tree_sha256") != EXPECTED_CHECKPOINT_TREE_SHA256
        or checkpoint.get("opened_read_only") is not True
        or not isinstance(source, Mapping)
        or source.get("wrong_owner_same_action_family") is not True
        or not isinstance(sampling, Mapping)
        or sampling.get("exact40") is not True
        or sampling.get("exact81") is not True
        or sampling.get("same_initial_gaussian_all_candidates") is not True
        or sampling.get("source_on_native_parity_bit_exact") is not True
        or not isinstance(frozen, Mapping)
        or frozen.get("unchanged") is not True
        or not isinstance(interpretation, Mapping)
        or interpretation.get("decoded_complete_video_required") is not True
        or interpretation.get("score_computed") is not False
        or interpretation.get("reward_computed") is not False
        or interpretation.get("ranking_performed") is not False
        or interpretation.get("selection_performed") is not False
        or interpretation.get("training_performed") is not False
        or interpretation.get("optimizer_present") is not False
        or interpretation.get("backward_performed") is not False
        or interpretation.get("parameter_update") is not False
        or interpretation.get("stage_B_authorized_by_runtime_alone") is not False
    ):
        fail("Stage-A A1 decoded source-edge runtime evidence differs")
    content_identity = checkpoint.get("content_identity")
    if (
        not isinstance(content_identity, Mapping)
        or content_identity.get("manifest_sha256_computed")
        != EXPECTED_CHECKPOINT_MANIFEST_SHA256
        or content_identity.get("manifest_sha256_expected")
        != EXPECTED_CHECKPOINT_MANIFEST_SHA256
        or content_identity.get("every_file_sha256_verified") is not True
    ):
        fail("Stage-A A1 checkpoint content identity differs")
    expected_plan = _expected_stage_a_plan()
    plan = shard.get("plan")
    candidates = value.get("candidates")
    outputs = value.get("outputs")
    expected_keys = [str(row["key"]) for row in expected_plan]
    if (
        plan != list(expected_plan)
        or not isinstance(candidates, list)
        or len(candidates) != len(expected_plan)
        or not isinstance(outputs, Mapping)
        or set(outputs) != set(expected_keys)
    ):
        fail("Stage-A A1 56-output plan closure differs")
    seed = sampling.get("seed")
    for expected_row, candidate in zip(expected_plan, candidates):
        if not isinstance(candidate, Mapping):
            fail("Stage-A A1 candidate row differs")
        _embedded_digest(
            candidate,
            field="candidate_digest",
            label=f"Stage-A candidate {expected_row['key']}",
        )
        trace_gate = candidate.get("trace_gate")
        if (
            any(candidate.get(key) != expected for key, expected in expected_row.items())
            or candidate.get("seed") != seed
            or candidate.get("score") is not None
            or candidate.get("rank") is not None
            or candidate.get("selected") is not False
            or not isinstance(trace_gate, Mapping)
            or trace_gate.get("passed") is not True
            or trace_gate.get("hook") != expected_row["hook"]
        ):
            fail(f"Stage-A A1 candidate evidence differs: {expected_row['key']}")
        output = outputs[expected_row["key"]]
        if (
            not isinstance(output, Mapping)
            or output.get("frame_count") != 81
            or float(output.get("fps", -1.0)) != 25.0
            or _SHA256.fullmatch(str(output.get("sha256"))) is None
        ):
            fail(f"Stage-A A1 decoded output differs: {expected_row['key']}")
        media_path = _plain_file(
            output.get("path"), label=f"Stage-A decoded {expected_row['key']}"
        )
        if file_sha256(media_path) != output["sha256"]:
            fail(f"Stage-A decoded media SHA differs: {expected_row['key']}")
    return value


@dataclass(frozen=True)
class StageAAdmission:
    path: Path
    file_sha256: str
    receipt_digest: str
    runtime_paths: tuple[Path, ...]
    runtime_file_sha256s: tuple[str, ...]
    runtime_receipt_digests: tuple[str, ...]
    runtime_families: tuple[str, ...]
    passed_block_bands: tuple[str, ...]
    installed_sparse_block_indices: tuple[int, ...]

    def receipt(self) -> Mapping[str, Any]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "receipt_digest": self.receipt_digest,
            "runtime_receipts": [
                {
                    "family": family,
                    "path": str(path),
                    "file_sha256": file_sha,
                    "receipt_digest": digest,
                }
                for family, path, file_sha, digest in zip(
                    self.runtime_families,
                    self.runtime_paths,
                    self.runtime_file_sha256s,
                    self.runtime_receipt_digests,
                )
            ],
            "passed_block_bands": list(self.passed_block_bands),
            "preregistered_sparse_representatives_by_band": {
                name: list(indices)
                for name, indices in PREREGISTERED_SPARSE_BLOCKS_BY_BAND.items()
            },
            "installed_sparse_block_indices": list(
                self.installed_sparse_block_indices
            ),
            "per_block_causal_localization_claimed": False,
            "optimizer_authorized": True,
        }


def load_stage_a_admission(
    path_value: str | Path, *, expected_sha256: str
) -> StageAAdmission:
    """Load a separately pinned manual admission; never infer one from scores."""

    path = _plain_file(path_value, label="Stage-A admission")
    expected_sha = _sha(expected_sha256, label="Stage-A admission expected SHA")
    observed_sha = file_sha256(path)
    if observed_sha != expected_sha:
        fail("Stage-A admission file SHA-256 differs")
    value = _read_strict_json(path, label="Stage-A admission")
    expected_fields = {
        "schema_version",
        "complete",
        "decision",
        "optimizer_authorized",
        "passed_block_bands",
        "preregistered_sparse_representatives_by_band",
        "installed_sparse_block_indices",
        "per_block_causal_localization_claimed",
        "runtime_receipts",
        "manual_conjunctive_review",
        "training_constraints",
        "receipt_digest",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        fail("Stage-A admission receipt fields differ")
    receipt_digest = _embedded_digest(
        value, field="receipt_digest", label="Stage-A admission"
    )
    runtime_refs = value.get("runtime_receipts")
    review = value.get("manual_conjunctive_review")
    constraints = value.get("training_constraints")
    if (
        value.get("schema_version") != STAGE_A_ADMISSION_SCHEMA
        or value.get("complete") is not True
        or value.get("decision") != "admit"
        or value.get("optimizer_authorized") is not True
        or value.get("passed_block_bands")
        != list(STAGE_A_REQUIRED_BLOCK_BANDS)
        or value.get("preregistered_sparse_representatives_by_band")
        != {
            name: list(indices)
            for name, indices in PREREGISTERED_SPARSE_BLOCKS_BY_BAND.items()
        }
        or value.get("installed_sparse_block_indices")
        != list(PREREGISTERED_SPARSE_BLOCK_INDICES)
        or value.get("per_block_causal_localization_claimed") is not False
        or not isinstance(runtime_refs, list)
        or len(runtime_refs) != len(STAGE_A_REQUIRED_FAMILIES)
        or not isinstance(review, Mapping)
        or review.get("decoded_media_reviewed_families")
        != list(STAGE_A_REQUIRED_FAMILIES)
        or review.get("reviewed_schedule_indices")
        != list(STAGE_A_SCHEDULE_INDICES)
        or review.get("reviewed_block_bands")
        != list(STAGE_A_REQUIRED_BLOCK_BANDS)
        or review.get("per_band_manual_pass")
        != {name: True for name in STAGE_A_REQUIRED_BLOCK_BANDS}
        or review.get("forward_action_prior_acceptable") is not True
        or review.get("forward_differs_from_source_off") is not True
        or review.get("reverse_and_incomplete_differ_from_forward") is not True
        or review.get("owner_specific_difference_interpretable") is not True
        or review.get("preservation_not_worse_than_adapter_off") is not True
        or review.get("scalar_threshold_used") is not False
        or review.get("feature_reward_used") is not False
        or review.get("vlm_used") is not False
        or not isinstance(constraints, Mapping)
        or constraints.get("source_only_train_rows") != TRAIN_ROWS
        or constraints.get("optimizer_steps") != OPTIMIZER_STEPS
        or constraints.get("gradient_accumulation_steps")
        != GRADIENT_ACCUMULATION_STEPS
        or constraints.get("effective_global_batch") != GLOBAL_BATCH
        or constraints.get("logical_training_records") != LOGICAL_RECORDS
        or constraints.get("checkpoint_steps") != list(CHECKPOINT_STEPS)
        or constraints.get("synthetic_target_accessed") is not False
        or constraints.get("objective") != "standard_target_only_noop_flow_matching"
        or constraints.get("arms") != list(MEMORY_INPUT_KINDS)
    ):
        fail("Stage-A admission decision/constraints differ")
    runtime_paths: list[Path] = []
    runtime_shas: list[str] = []
    runtime_digests: list[str] = []
    runtime_families: list[str] = []
    for expected_family, runtime_ref in zip(STAGE_A_REQUIRED_FAMILIES, runtime_refs):
        if (
            not isinstance(runtime_ref, Mapping)
            or set(runtime_ref) != {"family", "path", "file_sha256", "receipt_digest"}
            or runtime_ref.get("family") != expected_family
        ):
            fail("Stage-A admission runtime family references differ")
        runtime_path = _plain_file(
            runtime_ref["path"], label=f"Stage-A {expected_family} runtime receipt"
        )
        runtime_sha = _sha(
            runtime_ref["file_sha256"],
            label=f"Stage-A {expected_family} runtime receipt SHA",
        )
        if file_sha256(runtime_path) != runtime_sha:
            fail(f"Stage-A {expected_family} runtime receipt SHA-256 differs")
        runtime = validate_stage_a_runtime_receipt(
            _read_strict_json(
                runtime_path, label=f"Stage-A {expected_family} runtime receipt"
            )
        )
        runtime_digest = _embedded_digest(
            runtime, field="receipt_digest", label=f"Stage-A {expected_family} runtime"
        )
        if (
            runtime_digest != runtime_ref.get("receipt_digest")
            or runtime["shard"].get("family") != expected_family
        ):
            fail(f"Stage-A admission does not bind {expected_family} runtime evidence")
        runtime_paths.append(runtime_path)
        runtime_shas.append(runtime_sha)
        runtime_digests.append(runtime_digest)
        runtime_families.append(expected_family)
    return StageAAdmission(
        path=path,
        file_sha256=observed_sha,
        receipt_digest=receipt_digest,
        runtime_paths=tuple(runtime_paths),
        runtime_file_sha256s=tuple(runtime_shas),
        runtime_receipt_digests=tuple(runtime_digests),
        runtime_families=tuple(runtime_families),
        passed_block_bands=STAGE_A_REQUIRED_BLOCK_BANDS,
        installed_sparse_block_indices=PREREGISTERED_SPARSE_BLOCK_INDICES,
    )


@dataclass(frozen=True)
class Exact80Coordinate:
    optimizer_step: int
    checkpoint_interval: int
    step_in_checkpoint_interval: int
    microbatch_index: int
    interval_micro_ordinal: int
    interval_schedule_cycle: int
    schedule_index: int
    timestep: int
    sigma: float
    sigma_float32_be_hex: str

    def receipt(self) -> Mapping[str, Any]:
        return {
            "optimizer_step": self.optimizer_step,
            "checkpoint_interval": self.checkpoint_interval,
            "step_in_checkpoint_interval": self.step_in_checkpoint_interval,
            "microbatch_index": self.microbatch_index,
            "interval_micro_ordinal": self.interval_micro_ordinal,
            "interval_schedule_cycle": self.interval_schedule_cycle,
            "schedule_index": self.schedule_index,
            "timestep_int64": self.timestep,
            "sigma": self.sigma,
            "sigma_float32_be_hex": self.sigma_float32_be_hex,
        }


def exact80_coordinates() -> tuple[Exact80Coordinate, ...]:
    if exact40.SCHEDULE_SHA256 != EXPECTED_SCHEDULE_SHA256:
        fail("pinned exact40 schedule differs")
    values = tuple(
        Exact80Coordinate(
            optimizer_step=step + 1,
            checkpoint_interval=step // 20,
            step_in_checkpoint_interval=step % 20,
            microbatch_index=microbatch,
            interval_micro_ordinal=(step % 20) * GRADIENT_ACCUMULATION_STEPS
            + microbatch,
            interval_schedule_cycle=(
                ((step % 20) * GRADIENT_ACCUMULATION_STEPS + microbatch) // 40
            ),
            schedule_index=(
                ((step % 20) * GRADIENT_ACCUMULATION_STEPS + microbatch) % 40
            ),
            timestep=exact40.PINNED_TIMESTEPS[
                ((step % 20) * GRADIENT_ACCUMULATION_STEPS + microbatch) % 40
            ],
            sigma=exact40.PINNED_POSITIVE_SIGMAS[
                ((step % 20) * GRADIENT_ACCUMULATION_STEPS + microbatch) % 40
            ],
            sigma_float32_be_hex=exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                ((step % 20) * GRADIENT_ACCUMULATION_STEPS + microbatch) % 40
            ],
        )
        for step in range(OPTIMIZER_STEPS)
        for microbatch in range(GRADIENT_ACCUMULATION_STEPS)
    )
    if (
        len(values) != MICROBATCHES_PER_DP_ARM
        or {item.checkpoint_interval for item in values} != {0, 1, 2, 3}
        or any(
            sum(
                item.checkpoint_interval == interval
                and item.schedule_index == index
                for item in values
            )
            != 2
            for interval in range(4)
            for index in range(40)
        )
    ):
        fail("exact80 interval-local two-cycle schedule coverage differs")
    return values


def coordinates_for_optimizer_step(
    optimizer_step_zero_based: int,
) -> tuple[Exact80Coordinate, ...]:
    if (
        type(optimizer_step_zero_based) is not int
        or not 0 <= optimizer_step_zero_based < OPTIMIZER_STEPS
    ):
        fail("optimizer step coordinate differs")
    start = optimizer_step_zero_based * GRADIENT_ACCUMULATION_STEPS
    values = exact80_coordinates()[start : start + GRADIENT_ACCUMULATION_STEPS]
    if (
        len(values) != GRADIENT_ACCUMULATION_STEPS
        or [item.microbatch_index for item in values]
        != list(range(GRADIENT_ACCUMULATION_STEPS))
        or any(item.optimizer_step != optimizer_step_zero_based + 1 for item in values)
    ):
        fail("optimizer-step microbatch schedule differs")
    return values


def train_row_position(
    *, optimizer_step_zero_based: int, microbatch_index: int, dp_arm: int
) -> int:
    if (
        type(optimizer_step_zero_based) is not int
        or not 0 <= optimizer_step_zero_based < OPTIMIZER_STEPS
        or type(microbatch_index) is not int
        or not 0 <= microbatch_index < GRADIENT_ACCUMULATION_STEPS
        or type(dp_arm) is not int
        or not 0 <= dp_arm < PHYSICAL_DP_SIZE
    ):
        fail("exact80 sample coordinate differs")
    ordinal = (
        optimizer_step_zero_based * GLOBAL_BATCH
        + microbatch_index * PHYSICAL_DP_SIZE
        + dp_arm
    )
    epoch = ordinal // TRAIN_ROWS
    return (ordinal + 17 * epoch) % TRAIN_ROWS


def sample_coverage_receipt() -> Mapping[str, Any]:
    positions = [
        train_row_position(
            optimizer_step_zero_based=step,
            microbatch_index=microbatch,
            dp_arm=arm,
        )
        for step in range(OPTIMIZER_STEPS)
        for microbatch in range(GRADIENT_ACCUMULATION_STEPS)
        for arm in range(PHYSICAL_DP_SIZE)
    ]
    counts = [positions.count(index) for index in range(TRAIN_ROWS)]
    value = {
        "physical_dp_size": PHYSICAL_DP_SIZE,
        "ulysses_sp_size": 4,
        "per_rank_microbatch": 1,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": GLOBAL_BATCH,
        "optimizer_steps": OPTIMIZER_STEPS,
        "sample_exposures": len(positions),
        "train_rows": TRAIN_ROWS,
        "all_train_rows_seen": len(set(positions)) == TRAIN_ROWS,
        "minimum_exposures_per_row": min(counts),
        "maximum_exposures_per_row": max(counts),
        "checkpoint_intervals": 4,
        "microbatches_per_checkpoint_interval_per_dp_arm": 80,
        "schedule_cycles_per_checkpoint_interval": 2,
        "microbatch_uses_per_schedule_coordinate_per_interval_per_dp_arm": 2,
        "logical_records_at_checkpoint_steps": [
            step * GLOBAL_BATCH for step in CHECKPOINT_STEPS
        ],
        "row_positions_by_step_microbatch_dp_arm": [
            [
                [
                    train_row_position(
                        optimizer_step_zero_based=step,
                        microbatch_index=microbatch,
                        dp_arm=arm,
                    )
                    for arm in range(PHYSICAL_DP_SIZE)
                ]
                for microbatch in range(GRADIENT_ACCUMULATION_STEPS)
            ]
            for step in range(OPTIMIZER_STEPS)
        ],
    }
    if (
        value["sample_exposures"] != LOGICAL_RECORDS
        or value["all_train_rows_seen"] is not True
        or value["minimum_exposures_per_row"] != 10
        or value["maximum_exposures_per_row"] != 10
        or value["logical_records_at_checkpoint_steps"]
        != [0, 160, 320, 480, 640]
    ):
        fail("exact80 source-row coverage differs")
    return {**value, "digest": object_sha256(value)}


def validate_memory_input_kind(value: Any) -> str:
    if value not in MEMORY_INPUT_KINDS:
        fail("visual-context input kind differs")
    return str(value)


def load_visual_context_checkpoint(
    path_value: str | Path,
    *,
    expected_file_sha256: str,
    expected_step: int,
    expected_manifest_digest: str,
    expected_admission_digest: str,
    expected_memory_input_kind: str,
    handle: Any,
) -> Mapping[str, Any]:
    """Strictly validate, then load one immutable adapter-only checkpoint."""

    import torch
    import clean_source_visual_context_training_v1 as training

    path = _plain_file(path_value, label="visual-context checkpoint")
    if file_sha256(path) != _sha(
        expected_file_sha256, label="visual-context checkpoint expected SHA"
    ):
        fail("visual-context checkpoint file SHA-256 differs")
    if type(expected_step) is not int or expected_step not in CHECKPOINT_STEPS:
        fail("visual-context expected checkpoint step differs")
    _sha(expected_manifest_digest, label="source-only manifest digest")
    _sha(expected_admission_digest, label="Stage-A admission digest")
    kind = validate_memory_input_kind(expected_memory_input_kind)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older AUH torch
        payload = torch.load(path, map_location="cpu")
    except Exception as error:
        raise CleanSourceVisualStageBContractError(
            f"cannot load visual-context checkpoint: {error}"
        ) from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "metadata",
        "adapter_state_dict",
        "optimizer_state_dict",
    }:
        fail("visual-context checkpoint payload fields differ")
    metadata = payload.get("metadata")
    state = payload.get("adapter_state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        fail("visual-context checkpoint metadata/state differs")
    authorization = metadata.get("authorization")
    if (
        metadata.get("schema_version") != CHECKPOINT_SCHEMA
        or metadata.get("global_step") != expected_step
        or metadata.get("logical_records_seen") != expected_step * GLOBAL_BATCH
        or metadata.get("gradient_accumulation_steps")
        != GRADIENT_ACCUMULATION_STEPS
        or metadata.get("effective_global_batch") != GLOBAL_BATCH
        or metadata.get("checkpoint_cadence") != list(CHECKPOINT_STEPS)
        or metadata.get("manifest_digest") != expected_manifest_digest
        or metadata.get("base_frozen") is not True
        or metadata.get("native_kv_untouched") is not True
        or metadata.get("source_posterior_only") is not True
        or metadata.get("synthetic_target_posterior_accessed") is not False
        or metadata.get("objective") != "same_real_source_noop_flow_matching"
        or metadata.get("feature_or_vlm_reward") is not False
        or not isinstance(authorization, Mapping)
        or authorization.get("stage_a_admission_digest")
        != expected_admission_digest
        or authorization.get("memory_input_kind") != kind
        or authorization.get("optimizer_authorized") is not True
    ):
        fail("visual-context checkpoint scientific metadata differs")
    expected_named = dict(handle.trainable_named_parameters())
    if set(state) != set(expected_named):
        fail("visual-context checkpoint parameter names differ")
    normalized: dict[str, Any] = {}
    for name, parameter in expected_named.items():
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != tuple(parameter.shape)
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
        ):
            fail(f"visual-context checkpoint tensor differs: {name}")
        normalized[name] = tensor
    if training._state_digest(normalized) != metadata.get(
        "adapter_parameter_digest"
    ):
        fail("visual-context checkpoint parameter digest differs")
    incompatible = handle.components.load_state_dict(normalized, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        fail("visual-context checkpoint strict state load differs")
    return dict(metadata)


def checkpoint_decode_chain(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_digest: str,
    admission_digest: str,
    memory_input_kind: str,
) -> Mapping[str, Any]:
    """Build the exact ordered inputs a later inference holder must consume."""

    _sha(manifest_digest, label="decode-chain manifest digest")
    _sha(admission_digest, label="decode-chain admission digest")
    kind = validate_memory_input_kind(memory_input_kind)
    if not isinstance(records, Sequence) or len(records) != len(CHECKPOINT_STEPS):
        fail("decode chain requires exact checkpoint 0/20/40/60/80")
    normalized = []
    for expected_step, raw in zip(CHECKPOINT_STEPS, records):
        if (
            not isinstance(raw, Mapping)
            or raw.get("step") != expected_step
            or raw.get("logical_records_seen") != expected_step * GLOBAL_BATCH
            or type(raw.get("path")) is not str
            or not Path(str(raw["path"])).is_absolute()
            or _SHA256.fullmatch(str(raw.get("file_sha256"))) is None
            or _SHA256.fullmatch(str(raw.get("adapter_parameter_digest"))) is None
        ):
            fail(f"decode-chain checkpoint {expected_step} differs")
        normalized.append(
            {
                "step": expected_step,
                "logical_records_seen": expected_step * GLOBAL_BATCH,
                "checkpoint": str(raw["path"]),
                "checkpoint_sha256": str(raw["file_sha256"]),
                "adapter_parameter_digest": str(
                    raw["adapter_parameter_digest"]
                ),
            }
        )
    value = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "ordered_checkpoints": normalized,
        "source_only_manifest_digest": manifest_digest,
        "stage_a_admission_digest": admission_digest,
        "memory_input_kind": kind,
        "strict_loader": "load_visual_context_checkpoint",
        "one_fixed_inference_manifest_for_all_checkpoints_required": True,
        "same_source_instruction_seed_cells_required": True,
        "decoded_full_video_required": True,
        "scalar_selection_or_ranking_forbidden": True,
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "PREREGISTERED_SPARSE_BLOCK_INDICES",
    "CHECKPOINT_STEPS",
    "CleanSourceVisualStageBContractError",
    "Exact80Coordinate",
    "MEMORY_INPUT_KINDS",
    "OPTIMIZER_STEPS",
    "STAGE_A_ADMISSION_SCHEMA",
    "STAGE_A_RUNTIME_SCHEMA",
    "StageAAdmission",
    "checkpoint_decode_chain",
    "exact80_coordinates",
    "load_stage_a_admission",
    "load_visual_context_checkpoint",
    "sample_coverage_receipt",
    "train_row_position",
    "validate_memory_input_kind",
    "validate_stage_a_runtime_receipt",
]

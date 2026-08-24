#!/usr/bin/env python3
"""WORLD8 Phase-A field14: frozen short bootstrap followed by exact40 fields.

The process reconstructs the already versioned two-update short experiment
(updates 29 then 38, confirmations 29/38, adapter-off parity 0/25) from the
same authenticated source release.  Only after that protocol succeeds, it
executes a torch.no_grad six-field sweep at official UniPC indices 0..39.

For every coordinate the confirmation source, epsilon, x_sigma, native full
source V-pack, rotary, timestep and negative condition are byte-identical
across model fields.  Wrong identity uses only the same-family fit atlas;
drop disables only IdentityRebinder; action/no-op changes only positive text.
Indices 0..25 additionally reproduce their pre-install BF16 raw outputs at an
exact-zero route gate.  Indices 26..39 require the pinned finite nonzero gate.
Each six-field tensor set is canonically hashed and released before the next
coordinate.  Metrics are diagnostic only.  No checkpoint/media/publication is
written and an afterok dependency is only a queue gate, never weight lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Mapping, Optional, Sequence

import torch

import graft_phase_a_field14_exact40_v1 as field14
import identity_rebinder_v1 as rebinder
import infer_lora as legacy
import infer_native_identity_generation_canary as native_generation
import infer_source_kv_carrier_oracle as source_audit
import run_graft_phase_a_a_lite_short_gpu_v1 as short_runner
import train_graft_phase_a_a_lite_short_v1 as short_trainer
import tri_branch_unipc as sampler_contract


SCHEMA_VERSION = "bernini-graft-phase-a-field14-exact40-gpu-runner-v1"
BASELINE_SCHEMA_VERSION = "bernini-graft-phase-a-field14-preinstall-baseline-v1"
WORLD8_SCHEMA_VERSION = "bernini-graft-phase-a-field14-world8-result-v1"
PLAN_SCHEMA_VERSION = "bernini-graft-phase-a-field14-exact40-world8-plan-v1"

PINNED_SHORT_RUNNER_SHA256 = (
    "4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4"
)
PINNED_SHORT_SOURCE_COMMIT = "a884d357a6c0742f751be48d226ba72c952bae76"
# Filled from the frozen sibling source in the release build and also supplied
# independently by the sealed CLI/plan.  This constant deliberately binds the
# source dependency, not the live runner file itself (which would be circular).
PINNED_FIELD14_CORE_SHA256 = (
    "dbec11e9adc80171adf071bf2d79fc8ba7498b76e5740e19b077b90a0f0a5280"
)
MAX_WORLD8_PACKET_BYTES = 32 * 1024 * 1024


class Field14GPUError(RuntimeError):
    """Fail closed without a checkpoint or filesystem result path."""


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Field14GPUError(f"{label} must be lowercase SHA256")
    return value


def _require_git_commit(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Field14GPUError(f"{label} must be a full lowercase Git commit")
    return value


def _false_authority() -> dict[str, bool]:
    return {name: False for name in field14.AUTHORITY_FIELDS}


def assert_pinned_dependencies() -> None:
    short_runner.assert_pinned_dependencies()
    if (
        file_sha256(short_runner.__file__) != PINNED_SHORT_RUNNER_SHA256
        or file_sha256(field14.__file__) != PINNED_FIELD14_CORE_SHA256
        or tuple(field14.EXACT40_INDICES) != tuple(range(40))
        or tuple(field14.INACTIVE_INDICES) != tuple(range(26))
        or tuple(field14.ACTIVE_INDICES) != tuple(range(26, 40))
        or tuple(field14.FIELD_ROLES)
        != tuple(short_trainer.CONFIRMATION_FIELD_ROLES)
    ):
        raise Field14GPUError("field14 pinned dependency source/protocol differs")


def build_parser() -> argparse.ArgumentParser:
    parser = short_runner.build_parser()
    parser.description = __doc__
    parser.add_argument("--plan-path", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-field14-core-sha256", required=True)
    parser.add_argument("--expected-field14-runner-sha256", required=True)
    parser.add_argument("--expected-field14-source-commit", required=True)
    parser.add_argument(
        "--ack-exact40-no-grad-diagnostic-no-checkpoint-no-scientific-claim",
        action="store_true",
    )
    return parser


def _read_plan(args: argparse.Namespace) -> Mapping[str, Any]:
    raw = short_runner._read_exact_0444_file(  # noqa: SLF001
        args.plan_path, expected_sha256=args.expected_plan_sha256
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Field14GPUError("field14 plan is not ASCII JSON") from error
    expected_keys = {
        "action_instructions",
        "active_indices",
        "adapter_off_parity_indices",
        "afterok_dependency",
        "afterok_is_queue_gate_only",
        "authority",
        "checkpoint",
        "claim_scope",
        "confirmation_indices",
        "exact40_indices",
        "families",
        "field_roles",
        "inactive_indices",
        "inherits_weights_from_dependency",
        "measurement_contract",
        "no_checkpoint",
        "noop_instruction",
        "release",
        "resources",
        "runtime",
        "schema_version",
        "topology",
        "update_indices",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or raw != field14.canonical_json_bytes(value) + b"\n"
        or value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("exact40_indices") != list(range(40))
        or value.get("inactive_indices") != list(range(26))
        or value.get("active_indices") != list(range(26, 40))
        or value.get("update_indices") != [29, 38]
        or value.get("confirmation_indices") != [29, 38]
        or value.get("adapter_off_parity_indices") != [0, 25]
        or value.get("no_checkpoint") is not True
        or value.get("afterok_is_queue_gate_only") is not True
        or value.get("inherits_weights_from_dependency") is not False
        or value.get("field_roles") != list(field14.FIELD_ROLES)
        or value.get("noop_instruction") != short_runner.source_consumer.NOOP_INSTRUCTION
        or value.get("action_instructions")
        != dict(
            zip(
                short_runner.FAMILY_BY_DP_ARM,
                short_runner.ACTION_INSTRUCTION_BY_DP_ARM,
            )
        )
        or value.get("afterok_dependency")
        != {
            "job_id": "133524",
            "kind": "afterok",
            "purpose": "queue-gate-only",
        }
        or value.get("claim_scope")
        != "same-process-two-update-short-bootstrap-then-no-grad-exact40-six-field-diagnostic-no-checkpoint-no-scientific-claim"
    ):
        raise Field14GPUError("field14 plan protocol differs")
    runtime = value.get("runtime")
    topology = value.get("topology")
    resources = value.get("resources")
    authority = value.get("authority")
    families = value.get("families")
    measurement = value.get("measurement_contract")
    checkpoint = value.get("checkpoint")
    release = value.get("release")
    expected_families = [
        {
            "confirmation_iid": short_runner.CONFIRMATION_IID_BY_DP_ARM[arm],
            "dp_arm": arm,
            "family": short_runner.FAMILY_BY_DP_ARM[arm],
            "fit_iid": short_runner.FIT_IID_BY_DP_ARM[arm],
            "same_family_wrong_iid": short_runner.FIT_IID_BY_DP_ARM[arm],
        }
        for arm in range(2)
    ]
    expected_measurement = {
        "action_noop_change": "positive-text-only",
        "active_gate": "finite-nonzero-pinned-mid-low-sigma-gate",
        "cross_index_compensation": False,
        "cross_index_selection": False,
        "inactive_gate": "exact-zero-and-same-condition-raw-equals-preinstall",
        "per_index": "canonical-hash-then-release-before-next",
        "same_state": [
            "confirmation-source-zs",
            "epsilon",
            "x-sigma",
            "native-full-source-v-pack",
            "rotary",
            "timestep",
            "negative-condition",
        ],
        "semantic_metrics": "diagnostic-only",
        "sweep_grad_mode": "torch.no_grad",
        "wrong_identity": "same-family-fit-atlas-only",
    }
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("short_runner_sha256") != PINNED_SHORT_RUNNER_SHA256
        or runtime.get("short_source_commit") != PINNED_SHORT_SOURCE_COMMIT
        or runtime.get("field14_core_sha256")
        != args.expected_field14_core_sha256
        or runtime.get("field14_runner_sha256")
        != args.expected_field14_runner_sha256
        or runtime.get("field14_source_commit")
        != args.expected_field14_source_commit
        or runtime.get("bernini_commit") != args.expected_bernini_commit
        or runtime.get("veomni_commit") != args.expected_veomni_commit
        or runtime.get("consumer_sha256")
        != short_runner.PINNED_CONSUMER_SOURCE_SHA256
        or runtime.get("native_v2_sha256")
        != short_runner.PINNED_NATIVE_V2_SOURCE_SHA256
        or runtime.get("short_trainer_sha256")
        != short_runner.PINNED_SHORT_TRAINER_SOURCE_SHA256
        or runtime.get("short_trainer_execution_runtime_sha256")
        != short_runner.PINNED_SHORT_TRAINER_EXECUTION_RUNTIME_SHA256
        or runtime.get("identity_rebinder_sha256")
        != args.expected_identity_rebinder_sha256
        or set(runtime)
        != {
            "bernini_commit",
            "consumer_sha256",
            "field14_core_sha256",
            "field14_runner_sha256",
            "field14_source_commit",
            "identity_rebinder_sha256",
            "native_v2_sha256",
            "short_runner_sha256",
            "short_source_commit",
            "short_trainer_execution_runtime_sha256",
            "short_trainer_sha256",
            "veomni_commit",
        }
        or topology
        != {
            "allocation": "single-node-8xMI210",
            "dp_size": 2,
            "sp_size": 4,
            "world_size": 8,
        }
        or resources
        != {
            "cpus_per_task": 64,
            "gpus": 8,
            "memory_gib": 256,
            "nodes": 1,
            "ntasks": 1,
            "time_limit_hours": 48,
        }
        or not isinstance(authority, Mapping)
        or set(authority) != set(field14.AUTHORITY_FIELDS)
        or any(authority.get(name) is not False for name in field14.AUTHORITY_FIELDS)
        or families != expected_families
        or measurement != expected_measurement
        or checkpoint
        != {
            "content_manifest_sha256": (
                args.expected_checkpoint_content_manifest_sha256
            ),
            "tree_sha256": args.expected_checkpoint_tree_sha256,
        }
        or not isinstance(release, Mapping)
        or release.get("job_id") != "132549"
        or release.get("terminal_admission", {}).get("path")
        != args.terminal_admission_path
        or release.get("terminal_admission", {}).get("slurm_state") != "COMPLETED"
        or release.get("terminal_admission", {}).get("slurm_exit_code") != "0:0"
    ):
        raise Field14GPUError("field14 plan runtime/topology/authority differs")
    artifacts = release.get("artifacts")
    expected_artifact_paths = {
        "manifest": args.manifest_path,
        "producer": args.producer_receipt_path,
        "execution": args.execution_receipt_path,
        "submission": args.submission_receipt_path,
    }
    expected_artifact_hashes = {
        "manifest": args.manifest_sha256,
        "producer": args.producer_receipt_sha256,
        "execution": args.execution_receipt_sha256,
        "submission": args.submission_receipt_sha256,
    }
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != set(expected_artifact_paths)
        or any(
            artifacts.get(name)
            != {
                "path": expected_artifact_paths[name],
                "sha256": expected_artifact_hashes[name],
            }
            for name in expected_artifact_paths
        )
    ):
        raise Field14GPUError("field14 plan source-release artifacts differ")
    return value


def validate_cli(args: argparse.Namespace) -> argparse.Namespace:
    # The inherited option pins the old short runner because this process calls
    # that exact implementation for the bootstrap protocol.
    short_runner.validate_cli(args)
    assert_pinned_dependencies()
    for name in (
        "expected_plan_sha256",
        "expected_field14_core_sha256",
        "expected_field14_runner_sha256",
    ):
        _require_sha256(getattr(args, name), label=name)
    _require_git_commit(
        args.expected_field14_source_commit,
        label="expected_field14_source_commit",
    )
    if (
        args.expected_runner_sha256 != PINNED_SHORT_RUNNER_SHA256
        or args.expected_field14_core_sha256 != PINNED_FIELD14_CORE_SHA256
        or file_sha256(Path(__file__)) != args.expected_field14_runner_sha256
        or args.ack_exact40_no_grad_diagnostic_no_checkpoint_no_scientific_claim
        is not True
        or not Path(args.plan_path).is_absolute()
    ):
        raise Field14GPUError("field14 CLI source/acknowledgement differs")
    _read_plan(args)
    return args


def _capture_full_preinstall_baseline(
    *,
    diffusion: torch.nn.Module,
    transformer: torch.nn.Module,
    schedule: short_runner.Exact40CoordinateRegistry,
    confirmation: short_runner.SourceState,
    negative_condition: torch.Tensor,
    noop_condition: torch.Tensor,
    action_condition: torch.Tensor,
) -> Mapping[str, Any]:
    conditions = {
        "negative": negative_condition,
        "noop_positive": noop_condition,
        "action_positive": action_condition,
    }
    rows = []
    for index in field14.INACTIVE_INDICES:
        coordinate = schedule.coordinate(index)
        if field14.expected_route_gate(index) != 0.0:
            raise Field14GPUError("preinstall baseline coordinate is active")
        noise = short_runner.keyed_fresh_gaussian(
            shape=confirmation.source_latent.shape,
            device=confirmation.source_latent.device,
            source_sha256=confirmation.row.source_sha256,
            purpose="adapter-off-parity",
            schedule_index=index,
        )
        noisy, noisy_receipt = short_runner.native_runner_v1.build_noisy_target(
            confirmation.source_latent, noise.epsilon, sigma=coordinate.sigma
        )
        pack = short_runner._build_native_pack(  # noqa: SLF001
            transformer=transformer,
            source_latent=confirmation.source_latent,
            noisy_target=noisy,
        )
        for role, condition in conditions.items():
            raw = short_runner._native_raw_forward(  # noqa: SLF001
                diffusion=diffusion,
                pack=pack,
                coordinate=coordinate,
                condition=condition,
                route_context=short_runner.nullcontext(),
            )
            raw_identity = short_runner.tensor_identity(raw)
            rows.append(
                {
                    "schedule_index": index,
                    "branch_role": role,
                    "coordinate_digest": coordinate.receipt["digest"],
                    "epsilon": short_runner.tensor_identity(noise.epsilon),
                    "epsilon_receipt_digest": noise.receipt["digest"],
                    "noisy_target": short_runner.tensor_identity(noisy),
                    "noisy_target_receipt_digest": noisy_receipt["digest"],
                    "visual_pack": pack.visual_identity,
                    "rotary_pack": pack.rotary_identity,
                    "condition": short_runner.tensor_identity(condition),
                    "adapter_off_raw": raw_identity,
                    "adapter_off_raw_sha256": raw_identity["content_sha256"],
                }
            )
            del raw
        del pack, noisy, noise
        torch.cuda.empty_cache()
    schedule.assert_unchanged()
    return field14.seal_mapping(
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "confirmation_iid": confirmation.row.iid,
            "schedule_indices": list(field14.INACTIVE_INDICES),
            "branch_roles": list(short_runner.PARITY_BRANCH_ROLES),
            "captured_before_adapter_install": True,
            "raw_dtype": "torch.bfloat16",
            "rows": rows,
            "one_index_released_before_next": True,
            "target_video_used": False,
            **_false_authority(),
        }
    )


def _short_baseline_projection(value: Mapping[str, Any]) -> Mapping[str, Any]:
    baseline = field14.validate_sealed_mapping(value, label="field14 preinstall baseline")
    rows = [
        row
        for row in baseline.get("rows", [])
        if row.get("schedule_index") in short_runner.ADAPTER_OFF_PARITY_INDICES
    ]
    expected_order = [
        (index, role)
        for index in short_runner.ADAPTER_OFF_PARITY_INDICES
        for role in short_runner.PARITY_BRANCH_ROLES
    ]
    if [(row.get("schedule_index"), row.get("branch_role")) for row in rows] != expected_order:
        raise Field14GPUError("short baseline projection order differs")
    return short_runner.seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-short-preinstall-baseline-v1",
            "confirmation_iid": baseline["confirmation_iid"],
            "schedule_indices": list(short_runner.ADAPTER_OFF_PARITY_INDICES),
            "branch_roles": list(short_runner.PARITY_BRANCH_ROLES),
            "captured_before_adapter_install": True,
            "raw_dtype": "torch.bfloat16",
            "rows": rows,
            "target_video_used": False,
            **_false_authority(),
        }
    )


class OfficialField14Runtime:
    """One post-short, no-grad exact40 field producer with no tensor cache."""

    def __init__(
        self,
        *,
        short_runtime: short_runner.OfficialShortRuntime,
        preinstall_baseline: Mapping[str, Any],
    ) -> None:
        if (
            type(short_runtime) is not short_runner.OfficialShortRuntime
            or short_runtime._confirmation_seen  # noqa: SLF001
            != list(short_runner.CONFIRMATION_INDICES)
        ):
            raise Field14GPUError("field14 requires one completed official short runtime")
        baseline = field14.validate_sealed_mapping(
            preinstall_baseline, label="field14 preinstall baseline"
        )
        expected_rows = [
            (index, role)
            for index in field14.INACTIVE_INDICES
            for role in short_runner.PARITY_BRANCH_ROLES
        ]
        rows = baseline.get("rows")
        if (
            baseline.get("schema_version") != BASELINE_SCHEMA_VERSION
            or baseline.get("schedule_indices") != list(field14.INACTIVE_INDICES)
            or baseline.get("branch_roles") != list(short_runner.PARITY_BRANCH_ROLES)
            or baseline.get("captured_before_adapter_install") is not True
            or baseline.get("raw_dtype") != "torch.bfloat16"
            or not isinstance(rows, list)
            or [(row.get("schedule_index"), row.get("branch_role")) for row in rows]
            != expected_rows
        ):
            raise Field14GPUError("field14 preinstall baseline inventory differs")
        self.short_runtime = short_runtime
        self.baseline = baseline
        self._baseline_by_key = {
            (row["schedule_index"], row["branch_role"]): row for row in rows
        }
        self._next_index = 0
        self._released: list[int] = []

    @staticmethod
    def _without_atlas(value: Mapping[str, Any]) -> dict[str, Any]:
        plain = short_runner.validate_sealed_mapping(
            value, label="field14 identity route"
        )
        plain.pop("digest")
        plain.pop("atlas_receipt_digest")
        return plain

    def measure_index(self, schedule_index: int) -> field14.Field14TensorSet:
        runtime = self.short_runtime
        if (
            torch.is_grad_enabled()
            or schedule_index != self._next_index
            or self._released != list(range(schedule_index))
        ):
            raise Field14GPUError("field14 index order/release transaction differs")
        coordinate = runtime.schedule.coordinate(schedule_index)
        expected_gate = field14.expected_route_gate(schedule_index)
        purpose = (
            "adapter-off-parity"
            if schedule_index in field14.INACTIVE_INDICES
            else "field14-active-six-field"
        )
        noise = short_runner.keyed_fresh_gaussian(
            shape=runtime.confirmation.source_latent.shape,
            device=runtime.confirmation.source_latent.device,
            source_sha256=runtime.confirmation.row.source_sha256,
            purpose=purpose,
            schedule_index=schedule_index,
        )
        noisy, noisy_receipt = short_runner.native_runner_v1.build_noisy_target(
            runtime.confirmation.source_latent,
            noise.epsilon,
            sigma=coordinate.sigma,
        )
        pack = short_runner._build_native_pack(  # noqa: SLF001
            transformer=runtime.transformer,
            source_latent=runtime.confirmation.source_latent,
            noisy_target=noisy,
        )
        correct_atlas = runtime._atlas(runtime.confirmation)  # noqa: SLF001
        wrong_atlas = runtime._atlas(runtime.fit)  # noqa: SLF001

        def same_state_identities() -> dict[str, Any]:
            return {
                "confirmation_source_zs": short_runner.tensor_identity(
                    runtime.confirmation.source_latent
                ),
                "epsilon": short_runner.tensor_identity(noise.epsilon),
                "noisy_target_x_sigma": short_runner.tensor_identity(noisy),
                "native_visual_pack": short_runner.tensor_identity(pack.visual),
                "native_rotary_pack": short_runner.tensor_identity(pack.rotary),
                "sigma": short_runner.tensor_identity(coordinate.sigma),
                "timestep": short_runner.tensor_identity(coordinate.timestep),
                "negative_condition": short_runner.tensor_identity(
                    runtime.negative_condition
                ),
                "noop_positive_condition": short_runner.tensor_identity(
                    runtime.noop_condition
                ),
                "action_positive_condition": short_runner.tensor_identity(
                    runtime.action_condition
                ),
            }

        state_before = same_state_identities()
        atlas_before = {
            "correct_confirmation_atlas": short_runner.tensor_identity(
                correct_atlas.tokens
            ),
            "wrong_same_family_fit_atlas": short_runner.tensor_identity(
                wrong_atlas.tokens
            ),
        }

        def raw(
            condition: torch.Tensor,
            atlas: Optional[rebinder.IdentityAtlas],
            mode: str,
        ) -> tuple[torch.Tensor, Mapping[str, Any]]:
            return runtime._raw_for_mode(  # noqa: SLF001
                pack=pack,
                coordinate=coordinate,
                condition=condition,
                atlas=atlas,
                mode=mode,
            )

        correct_negative, correct_negative_route = raw(
            runtime.negative_condition, correct_atlas, "atlas"
        )
        correct_noop, correct_noop_route = raw(
            runtime.noop_condition, correct_atlas, "atlas"
        )
        correct_action, correct_action_route = raw(
            runtime.action_condition, correct_atlas, "atlas"
        )
        wrong_negative, wrong_negative_route = raw(
            runtime.negative_condition, wrong_atlas, "atlas"
        )
        wrong_noop, wrong_noop_route = raw(
            runtime.noop_condition, wrong_atlas, "atlas"
        )
        drop_negative, drop_negative_route = raw(
            runtime.negative_condition, None, "drop"
        )
        drop_noop, drop_noop_route = raw(runtime.noop_condition, None, "drop")
        drop_action, drop_action_route = raw(runtime.action_condition, None, "drop")

        raw_tensors = {
            "correct_negative": correct_negative,
            "correct_noop": correct_noop,
            "correct_action": correct_action,
            "wrong_negative": wrong_negative,
            "wrong_noop": wrong_noop,
            "drop_negative": drop_negative,
            "drop_noop": drop_noop,
            "drop_action": drop_action,
        }
        routes = {
            "correct_negative": correct_negative_route,
            "correct_noop": correct_noop_route,
            "correct_action": correct_action_route,
            "wrong_negative": wrong_negative_route,
            "wrong_noop": wrong_noop_route,
            "drop_negative": drop_negative_route,
            "drop_noop": drop_noop_route,
            "drop_action": drop_action_route,
        }
        if tuple(raw_tensors) != field14.RAW_ROLES or tuple(routes) != field14.RAW_ROLES:
            raise Field14GPUError("field14 raw call inventory differs")

        enabled_names = (
            "correct_negative",
            "correct_noop",
            "correct_action",
            "wrong_negative",
            "wrong_noop",
        )
        drop_names = ("drop_negative", "drop_noop", "drop_action")
        if any(
            routes[name].get("gate_hex") != expected_gate.hex()
            or routes[name].get("enabled") is not True
            or routes[name].get("branch_name") != "V"
            for name in enabled_names
        ) or any(
            routes[name].get("gate_hex") != 0.0.hex()
            or routes[name].get("enabled") is not False
            or routes[name].get("branch_name") != "V"
            or routes[name].get("atlas_receipt_digest") is not None
            for name in drop_names
        ):
            raise Field14GPUError("field14 route gate/provenance differs")
        if schedule_index in field14.ACTIVE_INDICES and (
            not math.isfinite(expected_gate) or expected_gate <= 0.0
        ):
            raise Field14GPUError("field14 active route is not finite nonzero")

        correct_routes_equal = (
            correct_negative_route == correct_noop_route == correct_action_route
        )
        wrong_routes_equal = wrong_negative_route == wrong_noop_route
        drop_routes_equal = drop_negative_route == drop_noop_route == drop_action_route
        wrong_only_atlas = (
            self._without_atlas(correct_negative_route)
            == self._without_atlas(wrong_negative_route)
            and correct_negative_route["atlas_receipt_digest"]
            == correct_atlas.receipt()["digest"]
            and wrong_negative_route["atlas_receipt_digest"]
            == wrong_atlas.receipt()["digest"]
            and correct_negative_route["atlas_receipt_digest"]
            != wrong_negative_route["atlas_receipt_digest"]
        )
        correct_without_adapter = self._without_atlas(correct_negative_route)
        drop_without_adapter = self._without_atlas(drop_negative_route)
        for key in ("enabled", "gate_hex"):
            correct_without_adapter.pop(key)
            drop_without_adapter.pop(key)
        drop_only_rebinder = correct_without_adapter == drop_without_adapter

        state_after = same_state_identities()
        atlas_after = {
            "correct_confirmation_atlas": short_runner.tensor_identity(
                correct_atlas.tokens
            ),
            "wrong_same_family_fit_atlas": short_runner.tensor_identity(
                wrong_atlas.tokens
            ),
        }
        if (
            state_before != state_after
            or atlas_before != atlas_after
            or not correct_routes_equal
            or not wrong_routes_equal
            or not drop_routes_equal
            or not wrong_only_atlas
            or not drop_only_rebinder
            or torch.equal(runtime.noop_condition, runtime.action_condition)
        ):
            raise Field14GPUError("field14 same-state/only-intervention proof differs")

        inactive_raw_parity: Optional[Mapping[str, Any]] = None
        if schedule_index in field14.INACTIVE_INDICES:
            raw_identities = {
                name: short_runner.tensor_identity(value)
                for name, value in raw_tensors.items()
            }
            baseline_negative = self._baseline_by_key[(schedule_index, "negative")]
            baseline_noop = self._baseline_by_key[(schedule_index, "noop_positive")]
            baseline_action = self._baseline_by_key[(schedule_index, "action_positive")]
            negative_equal = all(
                raw_identities[name] == baseline_negative["adapter_off_raw"]
                for name in ("correct_negative", "wrong_negative", "drop_negative")
            )
            noop_equal = all(
                raw_identities[name] == baseline_noop["adapter_off_raw"]
                for name in ("correct_noop", "wrong_noop", "drop_noop")
            )
            action_equal = all(
                raw_identities[name] == baseline_action["adapter_off_raw"]
                for name in ("correct_action", "drop_action")
            )
            shared_inputs_equal = all(
                baseline["coordinate_digest"] == coordinate.receipt["digest"]
                and baseline["epsilon"] == short_runner.tensor_identity(noise.epsilon)
                and baseline["noisy_target"] == short_runner.tensor_identity(noisy)
                and baseline["visual_pack"] == pack.visual_identity
                and baseline["rotary_pack"] == pack.rotary_identity
                and baseline["condition"] == short_runner.tensor_identity(condition)
                for baseline, condition in (
                    (baseline_negative, runtime.negative_condition),
                    (baseline_noop, runtime.noop_condition),
                    (baseline_action, runtime.action_condition),
                )
            )
            if not (negative_equal and noop_equal and action_equal and shared_inputs_equal):
                raise Field14GPUError(
                    f"field14 inactive raw/preinstall parity failed at {schedule_index}"
                )
            inactive_raw_parity = field14.seal_mapping(
                {
                    "schedule_index": schedule_index,
                    "route_gate_float64_hex": expected_gate.hex(),
                    "correct_wrong_drop_negative_raw_byte_exact": True,
                    "correct_wrong_drop_noop_raw_byte_exact": True,
                    "correct_drop_action_raw_byte_exact": True,
                    "all_same_condition_raw_equal_preinstall": True,
                    "source_noise_xsigma_vpack_rotary_timestep_conditions_equal_preinstall": True,
                    "preinstall_row_sha256": {
                        role: self._baseline_by_key[(schedule_index, role)][
                            "adapter_off_raw_sha256"
                        ]
                        for role in short_runner.PARITY_BRANCH_ROLES
                    },
                }
            )

        target = (
            (noisy - runtime.confirmation.source_latent) / coordinate.sigma
        ).float()
        fields = {
            "source_noop_target_velocity": target.detach().clone().contiguous(),
            "correct_atlas_noop_velocity": short_runner._guided_velocity_from_raw(  # noqa: SLF001
                bindings=runtime.bindings,
                noisy_target=noisy,
                coordinate=coordinate,
                pack=pack,
                negative_raw=correct_negative,
                positive_raw=correct_noop,
            ),
            "wrong_atlas_noop_velocity": short_runner._guided_velocity_from_raw(  # noqa: SLF001
                bindings=runtime.bindings,
                noisy_target=noisy,
                coordinate=coordinate,
                pack=pack,
                negative_raw=wrong_negative,
                positive_raw=wrong_noop,
            ),
            "dropped_atlas_noop_velocity": short_runner._guided_velocity_from_raw(  # noqa: SLF001
                bindings=runtime.bindings,
                noisy_target=noisy,
                coordinate=coordinate,
                pack=pack,
                negative_raw=drop_negative,
                positive_raw=drop_noop,
            ),
            "correct_atlas_action_velocity": short_runner._guided_velocity_from_raw(  # noqa: SLF001
                bindings=runtime.bindings,
                noisy_target=noisy,
                coordinate=coordinate,
                pack=pack,
                negative_raw=correct_negative,
                positive_raw=correct_action,
            ),
            "dropped_atlas_action_velocity": short_runner._guided_velocity_from_raw(  # noqa: SLF001
                bindings=runtime.bindings,
                noisy_target=noisy,
                coordinate=coordinate,
                pack=pack,
                negative_raw=drop_negative,
                positive_raw=drop_action,
            ),
        }
        raw_identities = {
            name: short_runner.tensor_identity(value) for name, value in raw_tensors.items()
        }
        runtime_evidence = {
            "coordinate": dict(coordinate.receipt),
            "confirmation_source_state_receipt_digest": runtime.confirmation.receipt[
                "digest"
            ],
            "wrong_fit_source_state_receipt_digest": runtime.fit.receipt["digest"],
            "epsilon_receipt_digest": noise.receipt["digest"],
            "noisy_target_receipt_digest": noisy_receipt["digest"],
            "same_state_identities_before_model_fields": state_before,
            "same_state_identities_after_all_model_fields": state_after,
            "atlas_identities_before_model_fields": atlas_before,
            "atlas_identities_after_all_model_fields": atlas_after,
            "same_state_tensor_identities_recomputed_byte_equal": True,
            "wrong_route_receipts_differ_only_in_atlas_memory": True,
            "drop_route_receipts_retain_v_branch_disable_only_rebinder": True,
            "action_noop_route_receipts_equal_with_negative_raw_reuse": True,
            "raw_call_order": list(field14.RAW_ROLES),
            "raw_tensor_identities": raw_identities,
            "route_receipts": {name: dict(route) for name, route in routes.items()},
            "expected_enabled_route_gate_float64_hex_recomputed": expected_gate.hex(),
            "inactive_raw_parity": (
                dict(inactive_raw_parity) if inactive_raw_parity is not None else None
            ),
            "ambient_torch_no_grad": True,
        }
        provenance = field14.build_field14_provenance(
            schedule_index=schedule_index,
            family=runtime.local.family,
            confirmation_iid=runtime.local.confirmation_iid,
            confirmation_source_sha256=runtime.confirmation.row.source_sha256,
            wrong_owner_iid=runtime.local.fit_iid,
            wrong_owner_source_sha256=runtime.fit.row.source_sha256,
            fields=fields,
            runtime_evidence=runtime_evidence,
        )
        self._next_index += 1
        return field14.Field14TensorSet(**fields, provenance=provenance)

    def release_index(self, schedule_index: int) -> Mapping[str, Any]:
        if (
            schedule_index != len(self._released)
            or self._next_index != schedule_index + 1
        ):
            raise Field14GPUError("field14 release order differs")
        torch.cuda.empty_cache()
        self._released.append(schedule_index)
        return field14.build_release_receipt(
            schedule_index, cuda_cache_requested=True
        )

    def assert_complete(self) -> None:
        if (
            self._next_index != 40
            or self._released != list(range(40))
            or rebinder.active_route() is not None
        ):
            raise Field14GPUError("field14 runtime did not complete/release exact40")


def _assemble_world8_packets(packets: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(packets, (list, tuple)) or len(packets) != short_runner.WORLD_SIZE:
        raise Field14GPUError("field14 WORLD8 packet coverage differs")
    short_packets = []
    admitted = []
    admitted_sweeps = []
    packet_keys = {
        "global_rank",
        "short_result_digest",
        "short_result",
        "field14_result_digest",
        "field14_result",
        "trainable_sha256_before_sweep",
        "trainable_sha256_after_sweep",
        "base_sha256_before",
        "base_sha256_after",
        "checkpoint_written",
        "checkpoint_payload_returned",
        "publication_performed",
    } | set(field14.AUTHORITY_FIELDS)
    for rank, packet in enumerate(packets):
        if (
            not isinstance(packet, Mapping)
            or set(packet) != packet_keys
            or packet.get("global_rank") != rank
            or len(field14.canonical_json_bytes(packet)) >= MAX_WORLD8_PACKET_BYTES
        ):
            raise Field14GPUError("field14 WORLD8 packet rank/size differs")
        short_result = short_runner.validate_sealed_mapping(
            packet.get("short_result"), label=f"field14 rank{rank} short result"
        )
        sweep = field14.validate_sealed_mapping(
            packet.get("field14_result"), label=f"field14 rank{rank} exact40 result"
        )
        arm = rank // short_runner.SP_SIZE
        sweep_keys = {
            "schema_version",
            "status",
            "family",
            "confirmation_iid",
            "confirmation_source_sha256",
            "wrong_owner_iid",
            "wrong_owner_source_sha256",
            "short_result_digest",
            "preinstall_baseline_digest",
            "schedule_indices",
            "inactive_indices",
            "active_indices",
            "field_roles",
            "rows",
            "exact40_official_order",
            "ambient_torch_no_grad",
            "one_index_admitted_hashed_and_released_before_next",
            "cross_index_tensor_retention",
            "cross_index_compensation_used",
            "cross_index_selection_used",
            "semantic_metrics_are_diagnostic_only",
            "checkpoint_written",
            "checkpoint_payload_returned",
            "publication_performed",
            "digest",
        } | set(field14.AUTHORITY_FIELDS)
        if (
            set(sweep) != sweep_keys
            or
            packet.get("short_result_digest") != short_result["digest"]
            or packet.get("field14_result_digest") != sweep["digest"]
            or sweep.get("schema_version") != field14.SCHEMA_VERSION
            or sweep.get("status")
            != "completed_in_memory_exact40_no_grad_no_checkpoint"
            or sweep.get("family") != short_runner.FAMILY_BY_DP_ARM[arm]
            or sweep.get("confirmation_iid")
            != short_runner.CONFIRMATION_IID_BY_DP_ARM[arm]
            or sweep.get("wrong_owner_iid") != short_runner.FIT_IID_BY_DP_ARM[arm]
            or sweep.get("short_result_digest") != short_result["digest"]
            or _require_sha256(
                sweep.get("preinstall_baseline_digest"),
                label="WORLD8 preinstall baseline digest",
            )
            != sweep.get("preinstall_baseline_digest")
            or _require_sha256(
                sweep.get("confirmation_source_sha256"),
                label="WORLD8 confirmation source SHA256",
            )
            != sweep.get("confirmation_source_sha256")
            or _require_sha256(
                sweep.get("wrong_owner_source_sha256"),
                label="WORLD8 wrong-owner source SHA256",
            )
            != sweep.get("wrong_owner_source_sha256")
            or sweep.get("confirmation_source_sha256")
            == sweep.get("wrong_owner_source_sha256")
            or sweep.get("schedule_indices") != list(range(40))
            or sweep.get("inactive_indices") != list(range(26))
            or sweep.get("active_indices") != list(range(26, 40))
            or sweep.get("field_roles") != list(field14.FIELD_ROLES)
            or len(sweep.get("rows", [])) != 40
            or sweep.get("exact40_official_order") is not True
            or sweep.get("ambient_torch_no_grad") is not True
            or sweep.get("one_index_admitted_hashed_and_released_before_next")
            is not True
            or sweep.get("cross_index_tensor_retention") is not False
            or sweep.get("cross_index_compensation_used") is not False
            or sweep.get("cross_index_selection_used") is not False
            or sweep.get("semantic_metrics_are_diagnostic_only") is not True
            or sweep.get("checkpoint_written") is not False
            or sweep.get("checkpoint_payload_returned") is not False
            or sweep.get("publication_performed") is not False
            or any(sweep.get(name) is not False for name in field14.AUTHORITY_FIELDS)
            or packet.get("trainable_sha256_before_sweep")
            != packet.get("trainable_sha256_after_sweep")
            or packet.get("base_sha256_before") != packet.get("base_sha256_after")
            or any(
                _require_sha256(packet.get(name), label=f"WORLD8 packet {name}")
                != packet.get(name)
                for name in (
                    "short_result_digest",
                    "field14_result_digest",
                    "trainable_sha256_before_sweep",
                    "base_sha256_before",
                )
            )
            or packet.get("checkpoint_written") is not False
            or packet.get("checkpoint_payload_returned") is not False
            or packet.get("publication_performed") is not False
            or any(packet.get(name) is not False for name in field14.AUTHORITY_FIELDS)
        ):
            raise Field14GPUError("field14 WORLD8 local result differs")
        sweep_rows = sweep["rows"]
        row_keys = {
            "schedule_index",
            "admission_digest",
            "field_tensor_sha256",
            "semantic_metrics_digest",
            "provenance_digest",
            "release_digest",
            "all_field_tensor_objects_released",
        }
        if any(
            not isinstance(row, Mapping)
            or set(row) != row_keys
            or row.get("schedule_index") != index
            or set(row.get("field_tensor_sha256", {})) != set(field14.FIELD_ROLES)
            or any(
                _require_sha256(value, label="WORLD8 field tensor SHA256") != value
                for value in row["field_tensor_sha256"].values()
            )
            or any(
                _require_sha256(row.get(name), label=f"WORLD8 {name}")
                != row.get(name)
                for name in (
                    "admission_digest",
                    "semantic_metrics_digest",
                    "provenance_digest",
                    "release_digest",
                )
            )
            or row.get("all_field_tensor_objects_released") is not True
            for index, row in enumerate(sweep_rows)
        ):
            raise Field14GPUError("field14 WORLD8 per-index coverage differs")
        short_runner._assert_no_elevated_authority_or_checkpoint(packet)  # noqa: SLF001
        short_packets.append(
            {
                "global_rank": rank,
                "result_digest": short_result["digest"],
                "local_result": short_result,
            }
        )
        admitted.append(
            {
                "global_rank": rank,
                "dp_arm": arm,
                "sp_rank": rank % short_runner.SP_SIZE,
                "family": sweep["family"],
                "short_result_digest": short_result["digest"],
                "field14_result_digest": sweep["digest"],
                "preinstall_baseline_digest": sweep["preinstall_baseline_digest"],
                "trainable_sha256_after_short_before_sweep": packet[
                    "trainable_sha256_before_sweep"
                ],
                "trainable_sha256_after_sweep": packet[
                    "trainable_sha256_after_sweep"
                ],
                "base_sha256_before": packet["base_sha256_before"],
                "base_sha256_after": packet["base_sha256_after"],
            }
        )
        admitted_sweeps.append(sweep)
    short_world8 = short_runner.assemble_world8_local_results(short_packets)
    arm_representatives = []
    for arm in range(2):
        rows = admitted_sweeps[
            arm * short_runner.SP_SIZE : (arm + 1) * short_runner.SP_SIZE
        ]
        digests = [row["digest"] for row in rows]
        consensus_records = [
            [
                {
                    "schedule_index": item["schedule_index"],
                    "field_tensor_sha256": item["field_tensor_sha256"],
                    "semantic_metrics_digest": item["semantic_metrics_digest"],
                }
                for item in row["rows"]
            ]
            for row in rows
        ]
        consensus_digest = field14.object_sha256(consensus_records[0])
        if any(
            field14.canonical_json_bytes(record)
            != field14.canonical_json_bytes(consensus_records[0])
            for record in consensus_records[1:]
        ):
            raise Field14GPUError(f"field14 arm{arm} SP4 sweep consensus differs")
        arm_representatives.append(
            {
                "dp_arm": arm,
                "family": short_runner.FAMILY_BY_DP_ARM[arm],
                "global_ranks": list(
                    range(
                        arm * short_runner.SP_SIZE,
                        (arm + 1) * short_runner.SP_SIZE,
                    )
                ),
                "representative_global_rank": arm * short_runner.SP_SIZE,
                "representative_field14_result": rows[0],
                "per_rank_field14_result_digests": digests,
                "field_hash_and_metric_consensus_digest": consensus_digest,
                "sp4_exact_field_hash_and_metric_consensus": True,
            }
        )
    return field14.seal_mapping(
        {
            "schema_version": WORLD8_SCHEMA_VERSION,
            "rank_order": list(range(8)),
            "dp2_family_order": list(short_runner.FAMILY_BY_DP_ARM),
            "rows": admitted,
            "all_eight_field14_results": admitted_sweeps,
            "arm_representatives": arm_representatives,
            "short_world8_full_results": dict(short_world8),
            "all_eight_exact40_completed": True,
            "all_eight_trainable_bytes_unchanged_during_sweep": True,
            "all_eight_base_bytes_unchanged_entire_process": True,
            "both_sp4_arms_exact_field_hash_and_metric_consensus": True,
            "checkpoint_written": False,
            "checkpoint_payload_returned": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


def _run_official_gpu(
    args: argparse.Namespace, routing: short_runner.source_consumer.TrainerRouting
) -> Mapping[str, Any]:
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise Field14GPUError(str(error)) from error
    if int(transformer_config.get("num_attention_heads", -1)) != 12:
        raise Field14GPUError("field14 checkpoint head count differs")
    manifest_path = Path(args.checkpoint_content_manifest).resolve(strict=True)
    if file_sha256(manifest_path) != args.expected_checkpoint_content_manifest_sha256:
        raise Field14GPUError("field14 checkpoint content manifest differs")
    inference_hashes = legacy.validate_inference_source_files(bernini_root)
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    import bernini.models.transformer_wan as transformer_wan
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if (
        SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT
        or DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT
    ):
        raise Field14GPUError("field14 official RV2V prompt constants differ")
    try:
        topology = short_runner._initialize_world8_dp2sp4(  # noqa: SLF001
            init_parallel_state
        )
    except Exception:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    local = short_runner.route_local_family(routing, dp_arm=topology.dp_arm)
    device = torch.device("cuda", topology.local_rank)
    handle: Optional[rebinder.IdentityRebinderHandle] = None
    try:
        plan = _read_plan(args)
        source_binding = field14.seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-field14-source-binding-v1",
                "field14_runner_sha256": args.expected_field14_runner_sha256,
                "field14_core_sha256": args.expected_field14_core_sha256,
                "field14_source_commit": args.expected_field14_source_commit,
                "short_runner_sha256": PINNED_SHORT_RUNNER_SHA256,
                "short_source_commit": PINNED_SHORT_SOURCE_COMMIT,
                "consumer_sha256": short_runner.PINNED_CONSUMER_SOURCE_SHA256,
                "native_v2_sha256": short_runner.PINNED_NATIVE_V2_SOURCE_SHA256,
                "short_trainer_sha256": short_runner.PINNED_SHORT_TRAINER_SOURCE_SHA256,
                "identity_rebinder_sha256": args.expected_identity_rebinder_sha256,
                "plan_sha256": args.expected_plan_sha256,
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "bernini_inference_files": inference_hashes,
            }
        )
        short_runner._gather_equal(  # noqa: SLF001
            source_binding,
            group=topology.world_group,
            count=short_runner.WORLD_SIZE,
            label="field14 source binding",
        )

        checkpoint_rows: list[Any] = [None]
        if topology.global_rank == 0:
            try:
                checkpoint_rows[0] = {
                    "ok": True,
                    "identity": source_audit.validate_checkpoint_content(
                        checkpoint,
                        manifest_path,
                        expected_manifest_sha256=(
                            args.expected_checkpoint_content_manifest_sha256
                        ),
                    ),
                }
            except Exception as error:
                checkpoint_rows[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(checkpoint_rows, src=0)
        checkpoint_result = checkpoint_rows[0]
        if (
            not isinstance(checkpoint_result, Mapping)
            or checkpoint_result.get("ok") is not True
        ):
            raise Field14GPUError(
                f"field14 checkpoint content validation failed: {checkpoint_result}"
            )
        checkpoint_identity = field14.seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-field14-checkpoint-v1",
                "identity": dict(checkpoint_result["identity"]),
            }
        )
        short_runner._gather_equal(  # noqa: SLF001
            checkpoint_identity,
            group=topology.world_group,
            count=short_runner.WORLD_SIZE,
            label="field14 checkpoint content",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint),
            subfolder="tokenizer",
            **legacy.tokenizer_load_kwargs(),
        )
        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        if float(config.shift) != native_generation.FLOW_SHIFT or config.use_unipc is not True:
            raise Field14GPUError("field14 renderer is not pinned UniPC shift5")
        renderer = BerniniRendererModel(config)
        renderer.eval().requires_grad_(False)
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval().requires_grad_(False).to(device)
        fit = short_runner._encode_source_state(  # noqa: SLF001
            row=local.fit_row,
            vae=vae,
            vae_encode=_vae_encode,
            device=device,
            sp_group=topology.sp_group,
        )
        confirmation = short_runner._encode_source_state(  # noqa: SLF001
            row=local.confirmation_row,
            vae=vae,
            vae_encode=_vae_encode,
            device=device,
            sp_group=topology.sp_group,
        )
        vae.to("cpu")
        del vae
        torch.cuda.empty_cache()

        renderer.to(device)
        diffusion = source_audit.resolve_diffusion_core(renderer)
        transformer = diffusion.transformer
        if transformer is None or getattr(diffusion, "transformer_2", None) is not None:
            raise Field14GPUError("field14 runner requires one pinned transformer_1")
        renderer.eval().requires_grad_(False)
        wan_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
        )
        schedule = short_runner.Exact40CoordinateRegistry(
            diffusion.scheduler, device=device
        )
        short_runner._gather_equal(  # noqa: SLF001
            schedule.receipt,
            group=topology.world_group,
            count=short_runner.WORLD_SIZE,
            label="field14 exact40 schedule registry",
        )
        negative, noop, action, condition_receipt = short_runner._encode_conditions(  # noqa: SLF001
            tokenizer=tokenizer,
            renderer=renderer,
            prompt_cleaner=prompt_clean,
            device=device,
            local=local,
        )
        short_runner._gather_equal(  # noqa: SLF001
            condition_receipt,
            group=topology.sp_group,
            count=short_runner.SP_SIZE,
            label="field14 family RV2V conditions",
        )
        renderer.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()

        preinstall_baseline = _capture_full_preinstall_baseline(
            diffusion=diffusion,
            transformer=transformer,
            schedule=schedule,
            confirmation=confirmation,
            negative_condition=negative,
            noop_condition=noop,
            action_condition=action,
        )
        short_runner._gather_equal(  # noqa: SLF001
            preinstall_baseline,
            group=topology.sp_group,
            count=short_runner.SP_SIZE,
            label="field14 full preinstall baseline",
        )
        short_baseline = _short_baseline_projection(preinstall_baseline)
        base_rows = short_runner.native_runner_v1._base_parameter_rows(  # noqa: SLF001
            transformer
        )
        base_before = short_runner.short_chunked_parameter_registry_digest(base_rows)
        handle = rebinder.install_identity_rebinder_v1(
            transformer,
            runtime_source_commit=bernini_revision,
            model_revision=rebinder.PINNED_BERNINI_MODEL_REVISION,
            checkpoint_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        )
        transformer.eval()
        handle.atlas_encoder.eval()
        trainable_rows = handle.trainable_named_parameters()
        initialization = short_runner._broadcast_initial_trainables(  # noqa: SLF001
            trainable_rows, world_group=topology.world_group
        )
        route_factory = short_runner.ShortTrainingAtlasRouteFactory(
            handle=handle,
            sp_rank=topology.sp_rank,
            sp_group=topology.sp_group,
        )
        route_receipt = short_runner._official_forward_route_receipt()  # noqa: SLF001
        bindings = short_runner.native_v2.authenticate_pinned_native_bindings(
            diffusion=diffusion,
            transformer=transformer,
            named_trainable_parameters=trainable_rows,
            external_trainable_owner_modules={"atlas_encoder": handle.atlas_encoder},
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
            transformer_wan_path=Path(transformer_wan.__file__).resolve(strict=True),
            bernini_commit=bernini_revision,
            forward_context_factory=route_factory,
            forward_route_receipt=route_receipt,
        )
        backend = short_trainer.authenticate_torch_distributed_world8_dp2sp4(
            world_group=topology.world_group,
            sp_group=topology.sp_group,
            dp_group=topology.dp_group,
        )
        short_runtime = short_runner.OfficialShortRuntime(
            local=local,
            bindings=bindings,
            diffusion=diffusion,
            transformer=transformer,
            handle=handle,
            route_factory=route_factory,
            schedule=schedule,
            fit=fit,
            confirmation=confirmation,
            negative_condition=negative,
            noop_condition=noop,
            action_condition=action,
            sp_rank=topology.sp_rank,
            sp_group=topology.sp_group,
            adapter_off_baseline=short_baseline,
        )
        services = short_runner.authenticate_official_services(short_runtime)
        short_result = short_runner.execute_authenticated_short_run(
            routing=routing,
            bindings=bindings,
            collectives=backend,
            services=services,
        )
        trainable_before_sweep = short_runner.short_chunked_parameter_registry_digest(
            trainable_rows
        )
        field_runtime = OfficialField14Runtime(
            short_runtime=short_runtime,
            preinstall_baseline=preinstall_baseline,
        )
        with torch.no_grad():
            exact40_result = field14.execute_exact40_sweep(
                family=local.family,
                confirmation_iid=local.confirmation_iid,
                confirmation_source_sha256=confirmation.row.source_sha256,
                wrong_owner_iid=local.fit_iid,
                wrong_owner_source_sha256=fit.row.source_sha256,
                short_result_digest=short_result.receipt["digest"],
                preinstall_baseline_digest=preinstall_baseline["digest"],
                measure_index=field_runtime.measure_index,
                release_index=field_runtime.release_index,
            )
        field_runtime.assert_complete()
        trainable_after_sweep = short_runner.short_chunked_parameter_registry_digest(
            trainable_rows
        )
        if trainable_before_sweep != trainable_after_sweep:
            raise Field14GPUError("field14 no-grad sweep changed trainable bytes")
        schedule.assert_unchanged()
        if any(parameter.grad is not None for _, parameter in base_rows):
            raise Field14GPUError("field14 frozen base acquired a gradient")
        base_after = short_runner.short_chunked_parameter_registry_digest(base_rows)
        if base_before != base_after:
            raise Field14GPUError("field14 frozen base bytes changed")

        # Own the full nested receipt trees before pickle-based object collectives;
        # ``dict(mappingproxy)`` is intentionally only a shallow conversion.
        short_result_plain = json.loads(
            field14.canonical_json_bytes(short_result.receipt)
        )
        exact40_result_plain = json.loads(
            field14.canonical_json_bytes(exact40_result)
        )
        if not isinstance(short_result_plain, dict) or not isinstance(
            exact40_result_plain, dict
        ):
            raise Field14GPUError("field14 collective receipt ownership differs")
        local_packet = {
            "global_rank": topology.global_rank,
            "short_result_digest": short_result.receipt["digest"],
            "short_result": short_result_plain,
            "field14_result_digest": exact40_result["digest"],
            "field14_result": exact40_result_plain,
            "trainable_sha256_before_sweep": trainable_before_sweep,
            "trainable_sha256_after_sweep": trainable_after_sweep,
            "base_sha256_before": base_before,
            "base_sha256_after": base_after,
            "checkpoint_written": False,
            "checkpoint_payload_returned": False,
            "publication_performed": False,
            **_false_authority(),
        }
        if (
            len(field14.canonical_json_bytes(local_packet)) >= MAX_WORLD8_PACKET_BYTES
            or pickle.loads(pickle.dumps(local_packet, protocol=5)) != local_packet
        ):
            raise Field14GPUError("field14 local packet is not bounded pickle-safe")
        packets: list[Any] = [None] * short_runner.WORLD_SIZE
        dist.all_gather_object(packets, local_packet)
        world8 = _assemble_world8_packets(packets)
        assembled = field14.seal_mapping(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed_in_memory_short_then_exact40_no_checkpoint",
                "world8": dict(world8),
                "topology_receipt": dict(topology.receipt),
                "source_binding": dict(source_binding),
                "plan": dict(plan),
                "plan_sha256": args.expected_plan_sha256,
                "checkpoint_identity": dict(checkpoint_identity),
                "initialization": dict(initialization),
                "local_short_result": dict(short_result.receipt),
                "local_field14_result": dict(exact40_result),
                "base_sha256_before": base_before,
                "base_sha256_after": base_after,
                "base_bytes_unchanged": True,
                "base_gradients_all_none": True,
                "trainable_sha256_after_short_before_sweep": trainable_before_sweep,
                "trainable_sha256_after_sweep": trainable_after_sweep,
                "trainable_bytes_unchanged_during_sweep": True,
                "wan_diffusion_sha256": wan_sha,
                "transformer_wan_sha256": file_sha256(
                    Path(transformer_wan.__file__).resolve(strict=True)
                ),
                "runtime_versions": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "diffusers": diffusers_version,
                    "transformers": transformers_version,
                },
                "dependency_afterok_is_queue_gate_only": True,
                "weights_inherited_from_dependency_job": False,
                "checkpoint_written": False,
                "checkpoint_payload_returned": False,
                "publication_performed": False,
                **_false_authority(),
            }
        )
        short_runner._assert_no_elevated_authority_or_checkpoint(assembled)  # noqa: SLF001
        dist.barrier()
        if topology.global_rank == 0:
            print(
                field14.canonical_json_bytes(dict(assembled)).decode("ascii"),
                flush=True,
            )
        dist.barrier()
        return assembled
    finally:
        if (
            handle is not None
            and not handle.restored
            and rebinder.active_route() is None
        ):
            handle.restore()
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    routing = short_runner.consume_authenticated_source_routing(args)
    _run_official_gpu(args, routing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "Field14GPUError",
    "OfficialField14Runtime",
    "PINNED_FIELD14_CORE_SHA256",
    "PINNED_SHORT_RUNNER_SHA256",
    "PINNED_SHORT_SOURCE_COMMIT",
    "PLAN_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "WORLD8_SCHEMA_VERSION",
    "_assemble_world8_packets",
    "_capture_full_preinstall_baseline",
    "_short_baseline_projection",
    "assert_pinned_dependencies",
    "build_parser",
    "file_sha256",
    "main",
    "validate_cli",
]

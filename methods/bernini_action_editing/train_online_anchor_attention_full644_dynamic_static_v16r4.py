#!/usr/bin/env python3
"""v16r4 exact644 training with required source-descent preservation.

This is a minimal corrective wrapper around the fresh-from-base v16 data and
the v16r3 exact-zero-RMS route implementation.  It deliberately changes only
the optimizer geometry and learning rate:

* the same sealed 644-row manifest and same-IID self-generated donors are used;
* ``source_halfspace_001`` replaces action-priority PCGrad;
* a global-RMS normalized, zero-momentum projected-SGD update uses a fixed
  1e-6 RMS step per active coordinate and no coordinate-wise preconditioner;
* both the formal merged gradient and the actual optimizer displacement must be
  descent directions for action and source replay;
* no action-only fallback, retry, optimizer reset, or per-sample gate exists.

Decoded Heldout8 canaries are an automatic checkpoint-promotion gate, not an
optimizer-row admission gate.  Training receipts bind their immutable input
manifest and cadence, while a separate controller owns decode execution and
publishes the result sidecars.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r3 as v16r3


v16 = v16r3.v16
r2 = v16.r2
v15 = v16.v15
base = v16r3.base
qk = v16r3.qk

METHOD = "bernini-online-anchor-full644-dynamic-static-routed-teacher-v16r4"
RECEIPT_SCHEMA = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-receipt-v16r4"
)
REPLAY_COMBINE_MODE = "source_halfspace_001"
LEARNING_RATE = 1.0e-6
MAX_GRAD_NORM = 10.0
SEED = 2026082302
OPTIMIZER_FAILURE_POLICY = (
    "fail_closed_no_retry_no_action_only_fallback_no_optimizer_state_reset_v16r4"
)
OPTIMIZER = "global_rms_normalized_source_halfspace_sgd_v1"
V16R3_S1_DIRECTION_EVIDENCE = {
    "source_receipt_sha256": (
        "f4405fb4d5a4d01e2de286ca63abdc727530205c325c52964ec3f5a32c5a6048"
    ),
    "source_receipt_schema": v16r3.RECEIPT_SCHEMA,
    "global_step": 1,
    "action_gradient_l2_norm_fp64": 0.12306103935871118,
    "raw_replay_gradient_l2_norm_fp64": 0.47813936082602443,
    "action_replay_cosine": -0.9423967803410712,
    "replay_combine_mode": "action_priority_pcgrad_010",
    "formal_raw_replay_dot_combined_fp64": -0.053482743553625776,
    "formal_first_order_source_fm_preserved": False,
    "adamw_delta_theta_l2_norm_fp64": 0.09121778237701504,
    "adamw_action_descent_fp64": 0.002251319709668499,
    "adamw_source_descent_fp64": -0.005455651851443389,
    "adamw_source_descent_passed": False,
    "parameter_snapshot_dtype": "float32",
    "changed_element_count": 94371840,
}
DECODED_CANARY_SCHEMA = "bernini-v16r4-heldout8-checkpoint-canary-contract-v1"
DECODED_CANARY_INPUT_SCHEMA = "action-editing-shared8-input-v1"
DECODED_CANARY_CASE_COUNT = 8
DECODED_CANARY_STEPS = (1, 8, 32, 128, 359, 644)
DECODED_CANARY_ARMS = (
    "adapter_only_route_off",
    "trained_editor_route_on",
)


_V16_BUILD_PARSER = v16.build_parser
_V16_VALIDATE_ARGS = v16.validate_args
_V16_CHECKPOINT_RECEIPT = v16.checkpoint_receipt
_V15_CHECKPOINT_RECEIPT = r2._V15_CHECKPOINT_RECEIPT
_BASE_MERGE_COMPONENT_GRADIENTS = base.merge_component_gradients
_BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE = base.actual_optimizer_update_probe
_V16R3_BUILD_REAL_SOURCE = (
    v16r3.build_real_source_paired_records_full644_v16r3
)


def _empty_runtime_audit() -> dict[str, Any]:
    return {
        "formal_merge_count": 0,
        "actual_update_steps": [],
        "failed_actual_probe_steps": [],
        "last_formal_merge": None,
        "last_actual_update": None,
        "optimizer_step_count": 0,
        "last_optimizer_step": None,
    }


_RUNTIME_AUDIT = _empty_runtime_audit()
_CANARY_BINDING: Optional[dict[str, Any]] = None
_ACTIVE_OPTIMIZER: Any = None


def fail(message: str) -> None:
    base.fail(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _V16_BUILD_PARSER()
    parser.add_argument(
        "--decoded-canary-manifest",
        required=True,
        help="Pinned eight-case held-out JSONL used by checkpoint canaries",
    )
    parser.add_argument(
        "--decoded-canary-manifest-sha256",
        required=True,
        help="Exact byte SHA-256 of --decoded-canary-manifest",
    )
    return parser


def _load_decoded_canary_binding(path_value: Any, sha_value: Any) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    expected_sha = str(sha_value)
    if v16.SHA256.fullmatch(expected_sha) is None:
        fail("v16r4 decoded-canary manifest SHA-256 syntax differs")
    if not path.is_file():
        fail("v16r4 decoded-canary manifest is not a regular file")
    actual_sha = v16.file_sha256(path)
    if actual_sha != expected_sha:
        fail("v16r4 decoded-canary manifest SHA-256 differs")

    rows: list[Mapping[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    fail("v16r4 decoded-canary row is not an object")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"v16r4 decoded-canary manifest cannot be parsed: {error}")

    if len(rows) != DECODED_CANARY_CASE_COUNT:
        fail("v16r4 decoded-canary manifest must contain exactly eight rows")
    iids: list[str] = []
    seeds: list[int] = []
    for index, row in enumerate(rows):
        iid = row.get("iid")
        instruction = row.get("instruction")
        source_video = row.get("source_video")
        seed = row.get("seed")
        if (
            row.get("schema_version") != DECODED_CANARY_INPUT_SCHEMA
            or row.get("index") != index
            or type(row.get("index")) is not int
            or row.get("split") not in ("test", "validation")
            or not isinstance(iid, str)
            or not iid
            or not isinstance(instruction, str)
            or not instruction.strip()
            or not isinstance(source_video, str)
            or not Path(source_video).is_absolute()
            or type(seed) is not int
        ):
            fail(f"v16r4 decoded-canary row contract differs at index {index}")
        iids.append(iid)
        seeds.append(seed)
    if len(set(iids)) != len(iids) or len(set(seeds)) != len(seeds):
        fail("v16r4 decoded-canary IID or seed uniqueness differs")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "case_count": len(rows),
        "iids": tuple(iids),
        "seeds": tuple(seeds),
        "iids_sha256": base.legacy.object_sha256(iids),
    }


def validate_args(args: argparse.Namespace) -> None:
    """Retain v16 gates while changing only merge geometry and learning rate."""

    global _CANARY_BINDING

    # The frozen v16 validator is the authority for every unchanged argument.
    # Validate a shadow carrying the old two values, then bind the v16r4 values
    # against the real namespace below.
    shadow = argparse.Namespace(**vars(args))
    shadow.replay_combine_mode = v15.REPLAY_COMBINE_MODE
    shadow.learning_rate = 1.0e-5
    _V16_VALIDATE_ARGS(shadow)

    if getattr(args, "replay_combine_mode", None) != REPLAY_COMBINE_MODE:
        fail(f"v16r4 requires --replay-combine-mode={REPLAY_COMBINE_MODE}")
    if float(getattr(args, "learning_rate", float("nan"))) != LEARNING_RATE:
        fail("v16r4 requires --learning-rate=1e-6")
    if float(getattr(args, "max_grad_norm", float("nan"))) != MAX_GRAD_NORM:
        fail("v16r4 requires the inherited max gradient norm 10")
    if int(getattr(args, "max_steps", -1)) != v16.FULL644_ROWS:
        fail("v16r4 requires one continuous exact644 optimizer run")
    if int(getattr(args, "seed", -1)) != SEED:
        fail(f"v16r4 requires the unchanged v16r3 seed {SEED}")
    if bool(getattr(args, "gradient_diagnostic_only", False)):
        fail("v16r4 is an optimizer run, not gradient-diagnostic-only")
    if "v16r4" not in str(Path(args.output)).lower():
        fail("v16r4 output path must carry an explicit v16r4 namespace")

    v16r3._validate_zero_rms_operator()
    _CANARY_BINDING = _load_decoded_canary_binding(
        args.decoded_canary_manifest,
        args.decoded_canary_manifest_sha256,
    )


def _make_global_rms_projected_sgd(
    parameters: Sequence[Any], *, lr: float
) -> Any:
    """Build SGD whose global direction is unchanged by optimizer geometry.

    The L2 update norm is ``lr * sqrt(nonzero_gradient_elements)``.  Thus the
    RMS update over active coordinates is exactly ``lr``, while every
    coordinate shares one positive scalar multiplier.  Unlike Adam/AdamW,
    this cannot rotate a source-halfspace gradient out of either descent
    half-space before finite-precision parameter storage.
    """

    import torch

    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        fail("v16r4 projected optimizer has no parameter")
    if float(lr) != LEARNING_RATE:
        fail("v16r4 projected optimizer learning rate differs")

    class GlobalRMSProjectedSGD(torch.optim.Optimizer):
        def __init__(self) -> None:
            super().__init__(parameter_tuple, {"lr": float(lr)})
            self._v16r4_step_count = 0
            self._v16r4_last_step: Optional[dict[str, Any]] = None

        @torch.no_grad()
        def step(self, closure: Any = None) -> Any:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            if len(self.param_groups) != 1:
                fail("v16r4 projected optimizer parameter-group count differs")
            group = self.param_groups[0]
            if float(group.get("lr", float("nan"))) != LEARNING_RATE:
                fail("v16r4 projected optimizer live learning rate differs")

            gradients: list[tuple[Any, Any]] = []
            first_device = None
            gradient_sq = None
            active_elements = None
            total_elements = 0
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    fail("v16r4 projected optimizer gradient is absent")
                if gradient.is_sparse:
                    fail("v16r4 projected optimizer forbids sparse gradients")
                if not bool(torch.isfinite(gradient).all().item()):
                    fail("v16r4 projected optimizer gradient is non-finite")
                if first_device is None:
                    first_device = gradient.device
                    gradient_sq = torch.zeros(
                        (), dtype=torch.float64, device=first_device
                    )
                    active_elements = torch.zeros(
                        (), dtype=torch.int64, device=first_device
                    )
                if gradient.device != first_device:
                    fail("v16r4 projected optimizer gradient device differs")
                gradient_sq += gradient.detach().double().square().sum()
                active_elements += gradient.detach().count_nonzero().to(
                    dtype=torch.int64
                )
                total_elements += int(gradient.numel())
                gradients.append((parameter, gradient))

            if gradient_sq is None or active_elements is None:
                fail("v16r4 projected optimizer gradient closure is empty")
            gradient_norm = math.sqrt(float(gradient_sq.item()))
            active_count = int(active_elements.item())
            if gradient_norm <= 0.0 or active_count <= 0:
                fail("v16r4 projected optimizer requires a nonzero gradient")
            planned_delta_norm = LEARNING_RATE * math.sqrt(active_count)
            global_scale = planned_delta_norm / gradient_norm
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (gradient_norm, planned_delta_norm, global_scale)
            ):
                fail("v16r4 projected optimizer scale is invalid")

            for parameter, gradient in gradients:
                parameter.add_(gradient, alpha=-global_scale)
            self._v16r4_step_count += 1
            self._v16r4_last_step = {
                "schema_version": "bernini-global-rms-projected-sgd-step-v1",
                "step": int(self._v16r4_step_count),
                "optimizer": OPTIMIZER,
                "learning_rate_active_coordinate_rms": LEARNING_RATE,
                "gradient_l2_norm_fp64": gradient_norm,
                "active_gradient_element_count": active_count,
                "total_parameter_element_count": total_elements,
                "planned_delta_theta_l2_norm_fp64": planned_delta_norm,
                "global_positive_direction_scale": global_scale,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "coordinatewise_preconditioner": False,
                "global_gradient_direction_preserved_before_storage_rounding": True,
            }
            return loss

    return GlobalRMSProjectedSGD()


def _projected_optimizer_factory() -> Any:
    def factory(parameters: Any, *args: Any, **kwargs: Any) -> Any:
        global _ACTIVE_OPTIMIZER

        if _ACTIVE_OPTIMIZER is not None:
            fail("v16r4 expected exactly one optimizer construction")
        if args:
            fail("v16r4 projected optimizer forbids positional AdamW options")
        options = dict(kwargs)
        lr = float(options.pop("lr", float("nan")))
        weight_decay = float(options.pop("weight_decay", float("nan")))
        if options:
            fail("v16r4 projected optimizer received unsupported AdamW options")
        if lr != LEARNING_RATE or weight_decay != 0.0:
            fail("v16r4 projected optimizer construction differs")
        optimizer = _make_global_rms_projected_sgd(tuple(parameters), lr=lr)
        _ACTIVE_OPTIMIZER = optimizer
        return optimizer

    return factory


def _collective_pass_or_failure(
    local_passed: bool, *, device: Any, phase: str
) -> bool:
    """Make every distributed rank take the same fail-closed branch."""

    import torch
    import torch.distributed as dist

    counts = torch.tensor(
        [int(local_passed), int(not local_passed)],
        dtype=torch.int32,
        device=device,
    )
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = int(dist.get_world_size())
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    passed, failed = (int(value) for value in counts.cpu().tolist())
    if (passed, failed) == (world_size, 0):
        return True
    if (passed, failed) == (0, world_size):
        return False
    fail(
        f"v16r4 {phase} result differs across ranks: "
        f"passed={passed}, failed={failed}, world_size={world_size}"
    )


def merge_component_gradients(
    named: Sequence[tuple[str, Any]],
    action_gradients: Sequence[Any],
    *,
    replay_combine_mode: str,
    base_replay_scale: float,
    diagnostic_only: bool = False,
) -> Mapping[str, Any]:
    """Use only the source-halfspace merge and fail on infeasible geometry."""

    if replay_combine_mode != REPLAY_COMBINE_MODE:
        fail("v16r4 merge is restricted to source_halfspace_001")
    if not named:
        fail("v16r4 merge has no trainable parameter")
    local_error: Optional[base.OnlineAnchorTrainingError] = None
    values: Optional[dict[str, Any]] = None
    try:
        values = dict(
            _BASE_MERGE_COMPONENT_GRADIENTS(
                named,
                action_gradients,
                replay_combine_mode=replay_combine_mode,
                base_replay_scale=base_replay_scale,
                diagnostic_only=diagnostic_only,
            )
        )
        if (
            values.get("first_order_source_fm_preserved") is not True
            or float(
                values.get(
                    "raw_replay_gradient_dot_combined_gradient_fp64", -1.0
                )
            )
            < -1.0e-8
            or float(
                values.get("action_gradient_dot_combined_gradient_fp64", 0.0)
            )
            <= 0.0
            or float(values.get("action_alignment_ratio", 0.0)) < 0.1
        ):
            fail("v16r4 formal action/source descent closure differs")
    except base.OnlineAnchorTrainingError as error:
        local_error = error
    passed = _collective_pass_or_failure(
        local_error is None,
        device=named[0][1].device,
        phase="formal source-halfspace merge",
    )
    if not passed:
        if local_error is None:
            fail("v16r4 formal merge failed without a local error")
        raise local_error
    if values is None:
        fail("v16r4 formal merge passed without values")
    values.update(
        {
            "v16r4_source_descent_required": True,
            "v16r4_action_descent_required": True,
            "v16r4_action_only_fallback_allowed": False,
            "v16r4_optimizer_state_reset_allowed": False,
        }
    )
    _RUNTIME_AUDIT["formal_merge_count"] += 1
    _RUNTIME_AUDIT["last_formal_merge"] = dict(values)
    return values


def actual_optimizer_update_probe(
    named: Sequence[tuple[str, Any]],
    parameter_values_before_step: Sequence[Any],
    action_gradients: Sequence[Any],
    raw_replay_gradients: Sequence[Any],
    *,
    replay_combine_mode: str,
    step: int,
) -> Mapping[str, Any]:
    """Admit only actual projected-SGD displacements descending both objectives."""

    if replay_combine_mode != REPLAY_COMBINE_MODE:
        fail("v16r4 actual-update probe is restricted to source_halfspace_001")
    if not named:
        fail("v16r4 actual-update probe has no trainable parameter")
    local_error: Optional[base.OnlineAnchorTrainingError] = None
    values: Optional[dict[str, Any]] = None
    expected_step = len(_RUNTIME_AUDIT["actual_update_steps"]) + 1
    try:
        values = dict(
            _BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE(
                named,
                parameter_values_before_step,
                action_gradients,
                raw_replay_gradients,
                replay_combine_mode=replay_combine_mode,
                step=step,
            )
        )
        if int(step) != expected_step:
            fail("v16r4 actual-update probe step sequence differs")
        if (
            values.get("optimizer_semantics_observed_not_modified") is not True
            or values.get("action_descent_required") is not True
            or values.get("action_descent_passed") is not True
            or values.get("source_descent_required") is not True
            or values.get("source_descent_passed") is not True
        ):
            fail("v16r4 actual optimizer action/source descent closure differs")
        optimizer = _ACTIVE_OPTIMIZER
        optimizer_step = getattr(optimizer, "_v16r4_last_step", None)
        if (
            optimizer is None
            or not isinstance(optimizer_step, Mapping)
            or int(getattr(optimizer, "_v16r4_step_count", -1)) != int(step)
            or int(optimizer_step.get("step", -1)) != int(step)
            or optimizer_step.get("optimizer") != OPTIMIZER
            or len(optimizer.state) != 0
        ):
            fail("v16r4 projected optimizer runtime closure differs")
        planned_delta = float(
            optimizer_step.get("planned_delta_theta_l2_norm_fp64", float("nan"))
        )
        actual_delta = float(values.get("delta_theta_l2_norm_fp64", float("nan")))
        relative_delta_error = abs(actual_delta - planned_delta) / planned_delta
        if (
            not math.isfinite(relative_delta_error)
            or relative_delta_error > 1.0e-3
        ):
            fail(
                "v16r4 stored parameter displacement differs from the planned "
                f"global-RMS step: actual={actual_delta!r}, "
                f"planned={planned_delta!r}, relative_error={relative_delta_error!r}"
            )
        values.update(
            {
                "v16r4_optimizer": OPTIMIZER,
                "v16r4_optimizer_step": dict(optimizer_step),
                "v16r4_actual_to_planned_delta_l2_ratio": (
                    actual_delta / planned_delta
                ),
                "v16r4_actual_vs_planned_delta_l2_relative_error": (
                    relative_delta_error
                ),
            }
        )
    except base.OnlineAnchorTrainingError as error:
        local_error = error
    passed = _collective_pass_or_failure(
        local_error is None,
        device=named[0][1].device,
        phase="actual projected-SGD action/source descent probe",
    )
    if not passed:
        _RUNTIME_AUDIT["failed_actual_probe_steps"].append(int(step))
        if local_error is None:
            fail("v16r4 actual-update probe failed without a local error")
        raise local_error
    if values is None:
        fail("v16r4 actual-update probe passed without values")
    values.update(
        {
            "v16r4_optimizer_failure_policy": OPTIMIZER_FAILURE_POLICY,
            "v16r4_probe_retry_count": 0,
            "v16r4_action_only_fallback_applied": False,
            "v16r4_optimizer_state_reset": False,
            "v16r4_failed_candidate_checkpoint_publication_allowed": False,
        }
    )
    _RUNTIME_AUDIT["actual_update_steps"].append(int(step))
    _RUNTIME_AUDIT["last_actual_update"] = dict(values)
    _RUNTIME_AUDIT["optimizer_step_count"] = int(
        getattr(_ACTIVE_OPTIMIZER, "_v16r4_step_count")
    )
    _RUNTIME_AUDIT["last_optimizer_step"] = dict(
        getattr(_ACTIVE_OPTIMIZER, "_v16r4_last_step")
    )
    return values


def build_real_source_paired_records_full644_dynamic_static_v16r4(
    **kwargs: Any,
) -> Any:
    return _V16R3_BUILD_REAL_SOURCE(**kwargs)


def _zero_rms_summary(step: int) -> dict[str, Any]:
    calls = list(v16r3._RUNTIME_AUDIT["s279_builder_calls"])
    covered = step >= v16r3.S279_STEP
    if covered:
        if calls != list(v16r3.S279_EXPECTED_CALLS):
            fail("v16r4 S279 endpoint-canary runtime closure differs")
    elif calls:
        fail("v16r4 S279 endpoint canary appeared before its sealed step")
    return {
        "policy": v16r3.ZERO_RMS_POLICY,
        "scope": list(v16r3.ZERO_RMS_SCOPE),
        "finite_nonnegative_forward_values_bit_exact": True,
        "zero_forward_value": 0.0,
        "zero_backward_subgradient": 0.0,
        "positive_backward_matches_standard_sqrt": True,
        "negative_or_nonfinite_values_masked": False,
        "component_preallreduce_finite_gate_relaxed": False,
        "nonfinite_gradient_committed": False,
        "policy_fixed_from_step_one": True,
        "s279_endpoint_canary": {
            "step": v16r3.S279_STEP,
            "target_iid": v16r3.S279_TARGET_IID,
            "target_family": v16r3.S279_TARGET_FAMILY,
            "expected_calls": [dict(item) for item in v16r3.S279_EXPECTED_CALLS],
            "observed_calls": calls,
            "covered_by_checkpoint": covered,
        },
    }


def _s1_projected_update_forecast() -> dict[str, Any]:
    evidence = V16R3_S1_DIRECTION_EVIDENCE
    action_norm = float(evidence["action_gradient_l2_norm_fp64"])
    replay_norm = float(evidence["raw_replay_gradient_l2_norm_fp64"])
    cosine = float(evidence["action_replay_cosine"])
    correction_q = max(0.01, -cosine + 0.01)
    action_inner = action_norm * action_norm * (1.0 + correction_q * cosine)
    source_inner = action_norm * replay_norm * (cosine + correction_q)
    combined_norm = action_norm * math.sqrt(
        1.0 + 2.0 * correction_q * cosine + correction_q * correction_q
    )
    active_count = int(evidence["changed_element_count"])
    planned_delta_norm = LEARNING_RATE * math.sqrt(active_count)
    global_scale = planned_delta_norm / combined_norm
    return {
        "schema_version": "bernini-v16r4-s1-projected-update-forecast-v1",
        "basis": "observed_v16r3_s1_global_action_replay_geometry",
        "v16r3_observation": dict(evidence),
        "source_halfspace_correction_ratio_q": correction_q,
        "source_halfspace_action_alignment_ratio": 1.0 + correction_q * cosine,
        "source_halfspace_combined_gradient_l2_forecast": combined_norm,
        "source_halfspace_action_inner_product_forecast": action_inner,
        "source_halfspace_source_inner_product_forecast": source_inner,
        "active_element_count_proxy_from_v16r3_s1": active_count,
        "projected_sgd_planned_delta_l2_forecast": planned_delta_norm,
        "projected_sgd_global_scale_forecast": global_scale,
        "projected_sgd_action_descent_forecast": global_scale * action_inner,
        "projected_sgd_source_descent_forecast": global_scale * source_inner,
        "v16r3_adamw_to_projected_delta_l2_ratio": (
            float(evidence["adamw_delta_theta_l2_norm_fp64"])
            / planned_delta_norm
        ),
        "forecast_is_design_evidence_not_execution_receipt": True,
    }


def checkpoint_receipt(**kwargs: Any) -> dict[str, Any]:
    receipt = _V16_CHECKPOINT_RECEIPT(**kwargs)
    step = int(receipt.get("global_step", 0))
    contract = receipt.get("training_contract")
    if not isinstance(contract, MutableMapping) or step <= 0:
        fail("v16r4 inherited receipt closure differs")
    if _CANARY_BINDING is None:
        fail("v16r4 decoded-canary binding is absent")
    expected_steps = list(range(1, step + 1))
    optimizer_step = _RUNTIME_AUDIT["last_optimizer_step"]
    if (
        int(_RUNTIME_AUDIT["formal_merge_count"]) != step
        or list(_RUNTIME_AUDIT["actual_update_steps"]) != expected_steps
        or _RUNTIME_AUDIT["failed_actual_probe_steps"]
        or int(_RUNTIME_AUDIT["optimizer_step_count"]) != step
        or not isinstance(optimizer_step, Mapping)
        or int(optimizer_step.get("step", -1)) != step
        or optimizer_step.get("optimizer") != OPTIMIZER
        or _ACTIVE_OPTIMIZER is None
        or int(getattr(_ACTIVE_OPTIMIZER, "_v16r4_step_count", -1)) != step
        or len(_ACTIVE_OPTIMIZER.state) != 0
    ):
        fail("v16r4 successful optimizer-update accounting differs")

    last_interaction = receipt.get("component_gradient_probes", {}).get(
        "interaction", {}
    )
    last_actual = receipt.get("actual_optimizer_update_probe", {})
    if (
        last_interaction.get("replay_combine_mode") != REPLAY_COMBINE_MODE
        or last_interaction.get("first_order_source_fm_preserved") is not True
        or last_actual.get("replay_combine_mode") != REPLAY_COMBINE_MODE
        or last_actual.get("action_descent_passed") is not True
        or last_actual.get("source_descent_required") is not True
        or last_actual.get("source_descent_passed") is not True
        or last_actual.get("optimizer_semantics_observed_not_modified") is not True
        or last_actual.get("v16r4_optimizer") != OPTIMIZER
    ):
        fail("v16r4 last-update receipt geometry differs")

    training_iids = set(v16._RUNTIME_AUDIT.get("manifest_iids", ()))
    canary_iids = set(_CANARY_BINDING["iids"])
    if len(training_iids) != v16.FULL644_ROWS or training_iids & canary_iids:
        fail("v16r4 training/decoded-canary IID separation differs")

    zero_rms = _zero_rms_summary(step)
    anchor_cache = receipt.get("anchor_cache")
    if (
        not isinstance(anchor_cache, Mapping)
        or anchor_cache.get("qk_only_zero_rms_backward_policy")
        != v16r3.ZERO_RMS_POLICY
    ):
        fail("v16r4 anchor-cache zero-RMS policy receipt differs")

    # These keys can only arise if an inherited fallback receipt is accidentally
    # reintroduced.  Delete none silently: their presence is a wiring failure.
    if "v15r2_collinear_fallback_summary" in receipt:
        fail("v16r4 inherited an action-only gradient fallback receipt")
    forbidden_contract_keys = (
        "near_collinear_fallback_drops_auxiliary_replay_for_that_update",
        "actual_action_descent_fallback_uses_primary_action_only",
    )
    if any(key in contract for key in forbidden_contract_keys):
        fail("v16r4 inherited an action-only optimizer fallback contract")

    trigger = step in DECODED_CANARY_STEPS
    source_summary = {
        "replay_combine_mode": REPLAY_COMBINE_MODE,
        "selection_reason": (
            "dynamic source-halfspace correction preserves a positive source "
            "margin; fixed q=0.25 is infeasible when cosine<-0.25"
        ),
        "optimizer": OPTIMIZER,
        "optimizer_scalar_learning_rate": LEARNING_RATE,
        "learning_rate_semantics": "rms_delta_per_active_gradient_coordinate",
        "numeric_learning_rate_ratio_to_v16r3_adamw": LEARNING_RATE / 1.0e-5,
        "learning_rate_units_directly_comparable_to_v16r3_adamw": False,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "coordinatewise_preconditioner": False,
        "global_positive_direction_scale_only": True,
        "global_positive_scale_preserves_both_formal_halfspaces": True,
        "coordinatewise_adaptive_updates_can_rotate_a_formally_safe_direction": True,
        "v16r3_s1_does_not_isolate_adamw_distortion_from_old_pcgrad_geometry": True,
        "seed": SEED,
        "training_data_changed_from_v16r3": False,
        "self_generated_donor_policy_changed_from_v16r3": False,
        "action_and_source_objectives_changed_from_v16r3": False,
        "formal_source_descent_required": True,
        "actual_optimizer_source_descent_required": True,
        "actual_optimizer_action_descent_required": True,
        "successful_update_count": step,
        "successful_update_steps_sha256": base.legacy.object_sha256(
            expected_steps
        ),
        "optimizer_failure_policy": OPTIMIZER_FAILURE_POLICY,
        "action_only_fallback_allowed": False,
        "optimizer_retry_allowed": False,
        "optimizer_state_reset_allowed": False,
        "optimizer_state_entry_count": 0,
        "optimizer_has_no_momentum_or_history_by_design": True,
        "last_optimizer_step": dict(optimizer_step),
        "distributed_probe_agreement_required": True,
        "failed_candidate_checkpoint_publication_allowed": False,
        "per_sample_manual_or_visual_optimizer_gate": False,
        "decoded_source_preservation_claimed": False,
        "scientific_claim_authorized": False,
        "v16r3_s1_failure_and_projected_update_design_evidence": (
            _s1_projected_update_forecast()
        ),
    }
    canary_contract = {
        "schema_version": DECODED_CANARY_SCHEMA,
        "input_manifest": _CANARY_BINDING["path"],
        "input_manifest_sha256": _CANARY_BINDING["sha256"],
        "input_manifest_schema": DECODED_CANARY_INPUT_SCHEMA,
        "case_count": DECODED_CANARY_CASE_COUNT,
        "case_iids_sha256": _CANARY_BINDING["iids_sha256"],
        "training_iid_overlap_count": 0,
        "checkpoint_save_steps": list(v16.SAVE_STEPS),
        "decoded_canary_trigger_steps": list(DECODED_CANARY_STEPS),
        "current_checkpoint_step": step,
        "current_checkpoint_requires_decoded_canary": trigger,
        "arms": list(DECODED_CANARY_ARMS),
        "cases_per_arm": DECODED_CANARY_CASE_COUNT,
        "automatic_metrics_required": [
            "decode_complete_81_frames_25fps",
            "high_frequency_collapse_ratio_vs_frozen_base",
            "source_structure_similarity_vs_frozen_base",
            "temporal_flicker_ratio_vs_frozen_base",
        ],
        "training_process_executes_decode": False,
        "external_automatic_controller_executes_decode": True,
        "checkpoint_promotion_requires_decoded_canary_sidecar": trigger,
        "checkpoint_promotion_eligible_from_training_receipt_alone": False,
        "decoded_canary_controls_optimizer_row_admission": False,
        "per_sample_manual_review_required": False,
        "scientific_claim_authorized": False,
    }

    receipt["schema_version"] = RECEIPT_SCHEMA
    receipt["scientific_claim_authorized"] = False
    receipt["claim_scope"] = (
        "engineering_training_run_only_non_scientific_until_automatic_held_out_evaluation"
    )
    receipt["v16r3_zero_rms_backward_summary"] = zero_rms
    receipt["v16r4_source_descent_summary"] = source_summary
    receipt["v16r4_decoded_canary_contract"] = canary_contract
    contract.update(
        {
            "method": METHOD,
            "replay_combine_mode": REPLAY_COMBINE_MODE,
            "optimizer": OPTIMIZER,
            "optimizer_scalar_learning_rate": LEARNING_RATE,
            "learning_rate_semantics": "rms_delta_per_active_gradient_coordinate",
            "numeric_learning_rate_ratio_to_v16r3_adamw": (
                LEARNING_RATE / 1.0e-5
            ),
            "learning_rate_units_directly_comparable_to_v16r3_adamw": False,
            "optimizer_momentum": 0.0,
            "optimizer_weight_decay": 0.0,
            "optimizer_coordinatewise_preconditioner": False,
            "optimizer_global_positive_direction_scale_only": True,
            "optimizer_global_positive_scale_preserves_both_formal_halfspaces": True,
            "coordinatewise_adaptive_preconditioner_forbidden": True,
            "seed": SEED,
            "training_data_changed_from_v16r3": False,
            "self_generated_donor_policy_changed_from_v16r3": False,
            "action_and_source_objectives_changed_from_v16r3": False,
            "source_gradient_preservation_enforced": True,
            "formal_source_descent_required": True,
            "actual_optimizer_source_descent_required": True,
            "actual_optimizer_action_descent_required": True,
            "optimizer_failure_policy": OPTIMIZER_FAILURE_POLICY,
            "action_only_fallback_allowed": False,
            "optimizer_retry_allowed": False,
            "optimizer_state_reset_allowed": False,
            "optimizer_state_entry_count": 0,
            "optimizer_has_no_momentum_or_history_by_design": True,
            "last_optimizer_step_schema": (
                optimizer_step["schema_version"]
            ),
            "distributed_probe_agreement_required": True,
            "failed_candidate_checkpoint_publication_allowed": False,
            "manual_or_visual_review_controls_optimizer_admission": False,
            "all_rows_admitted_from_sealed_manifest_without_per_sample_review": True,
            "decoded_canary_manifest_sha256": _CANARY_BINDING["sha256"],
            "decoded_canary_trigger_steps": list(DECODED_CANARY_STEPS),
            "current_checkpoint_requires_decoded_canary": trigger,
            "current_checkpoint_promotion_requires_decoded_canary_sidecar": trigger,
            "decoded_source_preservation_claimed": False,
            "qk_only_zero_rms_backward_policy": v16r3.ZERO_RMS_POLICY,
            "s279_endpoint_canary_covered": step >= v16r3.S279_STEP,
            "component_preallreduce_finite_gate_relaxed": False,
            "nonfinite_gradient_committed": False,
            "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
            "source_preservation_claimed": False,
            "scientific_claim_authorized": False,
        }
    )
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _RUNTIME_AUDIT, _CANARY_BINDING, _ACTIVE_OPTIMIZER

    _RUNTIME_AUDIT = _empty_runtime_audit()
    _CANARY_BINDING = None
    _ACTIVE_OPTIMIZER = None
    v16r3._RUNTIME_AUDIT = v16r3._empty_runtime_audit()

    import torch

    original_parser = v16.build_parser
    original_validate = v16.validate_args
    original_receipt = v16.checkpoint_receipt
    original_real_builder = v16.build_real_source_paired_records_full644_v16
    original_r2_receipt_parent = v16._R2_CHECKPOINT_RECEIPT
    original_r2_merge = r2.merge_component_gradients
    original_actual_probe = base.actual_optimizer_update_probe
    original_adamw = torch.optim.AdamW

    v16.build_parser = build_parser
    v16.validate_args = validate_args
    v16.checkpoint_receipt = checkpoint_receipt
    v16.build_real_source_paired_records_full644_v16 = (
        build_real_source_paired_records_full644_dynamic_static_v16r4
    )
    # Bypass v15r2's action-only collinearity fallback receipt and runtime
    # implementation.  v15's receipt plus v16's full644 extension remains the
    # inherited authority for all unchanged data/route semantics.
    v16._R2_CHECKPOINT_RECEIPT = _V15_CHECKPOINT_RECEIPT
    r2.merge_component_gradients = merge_component_gradients
    base.actual_optimizer_update_probe = actual_optimizer_update_probe
    torch.optim.AdamW = _projected_optimizer_factory()
    try:
        result = v16.main(argv)
        if (
            _ACTIVE_OPTIMIZER is None
            or int(getattr(_ACTIVE_OPTIMIZER, "_v16r4_step_count", -1))
            != v16.FULL644_ROWS
        ):
            fail("v16r4 completed without the exact projected-optimizer closure")
        return result
    finally:
        torch.optim.AdamW = original_adamw
        base.actual_optimizer_update_probe = original_actual_probe
        r2.merge_component_gradients = original_r2_merge
        v16._R2_CHECKPOINT_RECEIPT = original_r2_receipt_parent
        v16.build_real_source_paired_records_full644_v16 = original_real_builder
        v16.checkpoint_receipt = original_receipt
        v16.validate_args = original_validate
        v16.build_parser = original_parser
        _CANARY_BINDING = None
        _ACTIVE_OPTIMIZER = None


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the Q-MOSAIC clean-latent editor direction on real WORLD4/SP4.

This is a read-only direction experiment.  It authenticates the frozen owner,
the signed editor runtime packet, the checkpoint content, and Bernini's live
Ulysses-SP4 group; replays the owner cotangent to the current clean latent;
and decodes the fixed ``base/plus/minus`` exact81 arms at relative L2 dose
0.01.  The Action-LoRA gauge is installed only so the native route is the same
one that a later Phase-B experiment would use.  Its B tensors remain exact
zero and this program has no LoRA-VJP call or parameter-update path.

There is currently no method-owned decoded action/identity evaluator in this
runner.  Accordingly every semantic field is emitted as ``UNASSESSED`` and
LoRA authority remains false even when all numerical/media checks pass.
Caller booleans, callbacks, or self-reported semantic scores are not accepted.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as legacy  # noqa: E402
import infer_source_value_residual_oracle as value_audit  # noqa: E402
import self_imagined_native_rv2v_hidden_vjp_v1 as qmosaic  # noqa: E402


METHOD_NAME = "bernini-qmosaic-editor-clean-direction-sp4"
RUN_RECEIPT_SCHEMA = "bernini-qmosaic-editor-direction-sp4-run-v3"
WORLD4_ZERO_ROUTE_PROOF_SCHEMA = (
    "bernini-qmosaic-core16-zero-route-world4-proof-v1"
)
FIXED_REGISTRY_SHA256 = (
    "01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96"
)
FIXED_QUERY_SEEDS = {
    "dog": (2026081502, 2026081503),
    "human": (2026081505, 2026081506),
}
ARM_ORDER = ("base", "plus", "minus")
RELATIVE_L2_DOSE = 0.01
NATIVE_SCHEDULE_INDEX = 33
NATIVE_TIMESTEP = 516
EXPECTED_FRAMES = 81
EXPECTED_FPS = 25
EXACT81_25FPS_PROBE_FIELDS = frozenset(
    {*qmosaic.EXACT81_MEDIA_PROBE_FIELDS, "fps_exact_integer"}
)
WORLD_SIZE = 4
SP_SIZE = 4
CHECKPOINT_CONTENT_FILE_COUNT = 23
SEMANTIC_UNASSESSED = "UNASSESSED_NO_METHOD_OWNED_EVALUATOR"
RUN_RECEIPT_FILENAME = "run.receipt.json"
TERMINAL_FULL_SEAL_SCHEMA = "bernini-qmosaic-terminal-full-runtime-seal-v1"
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class QMosaicEditorDirectionError(RuntimeError):
    """A fixed input, native replay, direction, decode, or seal differed."""


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise QMosaicEditorDirectionError("hashed artifact must be an absolute plain file")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise QMosaicEditorDirectionError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise QMosaicEditorDirectionError(f"{label} must be full lowercase SHA-1")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise QMosaicEditorDirectionError("receipt is not finite canonical ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _zero_route_forward_binding(parity: Mapping[str, Any]) -> str:
    """Bind the structural proof to this exact source/noise/prompt state."""

    required_sha_fields = (
        "source_latent_tensor_sha256",
        "official_initial_noise_tensor_sha256",
        "action_prompt_sha256",
        "noop_prompt_sha256",
        "prompt_condition_binding_digest",
        "checkpoint_content_receipt_digest",
    )
    if (
        parity.get("native_schedule_index") != NATIVE_SCHEDULE_INDEX
        or parity.get("native_timestep") != NATIVE_TIMESTEP
        or any(
            _SHA256_RE.fullmatch(str(parity.get(name))) is None
            for name in required_sha_fields
        )
    ):
        raise QMosaicEditorDirectionError(
            "zero-route forward binding coordinate differs"
        )
    return object_sha256(
        {
            "schema_version": "bernini-qmosaic-zero-route-forward-binding-v1",
            **{name: parity[name] for name in required_sha_fields},
            "native_schedule_index": NATIVE_SCHEDULE_INDEX,
            "native_timestep": NATIVE_TIMESTEP,
            "sp_size": SP_SIZE,
            "branch_name": "VI",
        }
    )


def _validate_local_zero_route_proof(
    proof: Mapping[str, Any], *, role: str, sp_rank: int
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "role",
        "sp_rank",
        "sp_size",
        "branch_name",
        "native_schedule_index",
        "native_timestep",
        "sigma_gate",
        "sigma_gate_weight",
        "grad_enabled",
        "inference_mode_enabled",
        "wrapper_count",
        "canonical_wrapper_order_sha256",
        "call_evidence",
        "call_evidence_sha256",
        "b_state_before_sha256",
        "b_state_after_sha256",
        "total_local_row_count",
        "total_selected_row_count",
        "missing_wrapper_count",
        "repeated_wrapper_count",
        "all_selected_deltas_numerically_exact_zero",
        "all_base_result_raw_bytes_equal",
        "b_unchanged",
        "digest",
    }
    if not isinstance(proof, Mapping) or set(proof) != expected_fields:
        raise QMosaicEditorDirectionError(
            "local zero-route structural proof field closure differs"
        )
    unsigned = dict(proof)
    digest = unsigned.pop("digest", None)
    wrapper_count = proof.get("wrapper_count")
    total_local = proof.get("total_local_row_count")
    total_selected = proof.get("total_selected_row_count")
    call_evidence = proof.get("call_evidence")
    call_fields = {
        "canonical_b_name",
        "local_row_count",
        "selected_row_count",
        "selector_sha256",
        "selector_exact_expected",
        "b_raw_nonzero_byte_count",
        "selected_delta_nonzero_element_count",
        "base_result_raw_byte_mismatch_count",
        "output_dtype",
        "output_shape",
        "selected_delta_numerically_exact_zero",
        "base_result_raw_bytes_equal",
        "autograd_enabled",
        "inference_mode_enabled",
    }
    sha_fields = (
        "canonical_wrapper_order_sha256",
        "call_evidence_sha256",
        "b_state_before_sha256",
        "b_state_after_sha256",
    )
    if (
        proof.get("schema_version") != qmosaic.ZERO_ROUTE_PROOF_SCHEMA_VERSION
        or proof.get("role") != role
        or proof.get("sp_rank") != sp_rank
        or proof.get("sp_size") != SP_SIZE
        or proof.get("branch_name") != "VI"
        or proof.get("native_schedule_index") != NATIVE_SCHEDULE_INDEX
        or proof.get("native_timestep") != NATIVE_TIMESTEP
        or proof.get("sigma_gate") != "mid"
        or proof.get("sigma_gate_weight") != 0.5
        or proof.get("grad_enabled") is not True
        or proof.get("inference_mode_enabled") is not False
        or wrapper_count != len(qmosaic.CANONICAL_B_PARAMETER_NAMES)
        or not isinstance(call_evidence, list)
        or len(call_evidence) != wrapper_count
        or [
            row.get("canonical_b_name")
            if isinstance(row, Mapping)
            else None
            for row in call_evidence
        ]
        != list(qmosaic.CANONICAL_B_PARAMETER_NAMES)
        or any(
            not isinstance(row, Mapping)
            or set(row) != call_fields
            or type(row.get("local_row_count")) is not int
            or row.get("local_row_count") <= 0
            or type(row.get("selected_row_count")) is not int
            or not 0 <= row.get("selected_row_count") <= row.get("local_row_count")
            or _SHA256_RE.fullmatch(str(row.get("selector_sha256"))) is None
            or row.get("selector_exact_expected") is not True
            or row.get("b_raw_nonzero_byte_count") != 0
            or row.get("selected_delta_nonzero_element_count") != 0
            or row.get("base_result_raw_byte_mismatch_count") != 0
            or row.get("output_dtype") not in ("torch.float32", "torch.bfloat16")
            or row.get("output_shape")
            != [1, row.get("local_row_count"), qmosaic.HIDDEN_SIZE]
            or row.get("selected_delta_numerically_exact_zero") is not True
            or row.get("base_result_raw_bytes_equal") is not True
            or row.get("autograd_enabled") is not True
            or row.get("inference_mode_enabled") is not False
            for row in call_evidence
        )
        or len({row["local_row_count"] for row in call_evidence}) != 1
        or len({row["selected_row_count"] for row in call_evidence}) != 1
        or len({row["selector_sha256"] for row in call_evidence}) != 1
        or sum(row["local_row_count"] for row in call_evidence) != total_local
        or sum(row["selected_row_count"] for row in call_evidence)
        != total_selected
        or proof.get("call_evidence_sha256")
        != object_sha256(call_evidence)
        or type(total_local) is not int
        or total_local <= 0
        or total_local % wrapper_count != 0
        or type(total_selected) is not int
        or not 0 <= total_selected <= total_local
        or total_selected % wrapper_count != 0
        or proof.get("missing_wrapper_count") != 0
        or proof.get("repeated_wrapper_count") != 0
        or proof.get("all_selected_deltas_numerically_exact_zero") is not True
        or proof.get("all_base_result_raw_bytes_equal") is not True
        or proof.get("b_unchanged") is not True
        or proof.get("b_state_before_sha256")
        != proof.get("b_state_after_sha256")
        or proof.get("canonical_wrapper_order_sha256")
        != qmosaic.object_sha256(list(qmosaic.CANONICAL_B_PARAMETER_NAMES))
        or any(_SHA256_RE.fullmatch(str(proof.get(name))) is None for name in sha_fields)
        or digest != object_sha256(unsigned)
    ):
        raise QMosaicEditorDirectionError(
            "local zero-route structural proof differs"
        )
    return dict(proof)


def build_world4_zero_route_proof(
    *,
    action_rows: Sequence[Mapping[str, Any]],
    noop_rows: Sequence[Mapping[str, Any]],
    parity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Close the two role-specific single-forward proofs across WORLD4."""

    if len(action_rows) != SP_SIZE or len(noop_rows) != SP_SIZE:
        raise QMosaicEditorDirectionError(
            "WORLD4 zero-route structural proof rank count differs"
        )
    validated = {
        role: [
            _validate_local_zero_route_proof(row, role=role, sp_rank=rank)
            for rank, row in enumerate(rows)
        ]
        for role, rows in (("action", action_rows), ("noop", noop_rows))
    }
    action = validated["action"]
    noop = validated["noop"]
    if (
        any(
            left["total_local_row_count"] != right["total_local_row_count"]
            or left["total_selected_row_count"]
            != right["total_selected_row_count"]
            for left, right in zip(action, noop)
        )
        or len(
            {
                row["b_state_before_sha256"]
                for row in (*action, *noop)
            }
        )
        != 1
        or {row["digest"] for row in action}
        & {row["digest"] for row in noop}
    ):
        raise QMosaicEditorDirectionError(
            "WORLD4 action/no-op zero-route structural closure differs"
        )
    wrapper_count = len(qmosaic.CANONICAL_B_PARAMETER_NAMES)
    world_selected = sum(
        int(row["total_selected_row_count"]) for row in action
    )
    world_local = sum(int(row["total_local_row_count"]) for row in action)
    if (
        world_selected <= 0
        or world_selected >= world_local
        or world_selected % wrapper_count != 0
        or world_local % wrapper_count != 0
    ):
        raise QMosaicEditorDirectionError(
            "WORLD4 zero-route selected/local row closure differs"
        )
    unsigned = {
        "schema_version": WORLD4_ZERO_ROUTE_PROOF_SCHEMA,
        "roles": ["action", "noop"],
        "world_size": WORLD_SIZE,
        "sp_size": SP_SIZE,
        "rank_order": list(range(SP_SIZE)),
        "forward_binding_digest": _zero_route_forward_binding(parity),
        "action_prompt_sha256": parity["action_prompt_sha256"],
        "noop_prompt_sha256": parity["noop_prompt_sha256"],
        "per_rank_wrapper_count": wrapper_count,
        "per_role_world_wrapper_call_count": SP_SIZE * wrapper_count,
        "total_world_wrapper_call_count": 2 * SP_SIZE * wrapper_count,
        "world_local_row_count_per_wrapper": world_local // wrapper_count,
        "world_selected_target_row_count_per_wrapper": (
            world_selected // wrapper_count
        ),
        "action_local_proofs": action,
        "noop_local_proofs": noop,
        "all_rank_b_state_digest_consensus": True,
        "all_rank_selected_delta_exact_zero": True,
        "all_rank_output_byte_equal_base": True,
        "action_noop_role_closure_passed": True,
        "full_model_off_enabled_sketch_comparison_used_for_authority": False,
        "allclose_or_tolerance_used": False,
        "semantic_authority": False,
        "lora_vjp_or_update_authority": False,
        "structural_zero_route_authority_passed": True,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def validate_world4_zero_route_proof(
    proof: Mapping[str, Any], *, parity: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not isinstance(proof, Mapping):
        raise QMosaicEditorDirectionError(
            "WORLD4 zero-route structural proof must be a mapping"
        )
    try:
        rebuilt = build_world4_zero_route_proof(
            action_rows=proof["action_local_proofs"],
            noop_rows=proof["noop_local_proofs"],
            parity=parity,
        )
    except (KeyError, TypeError) as error:
        raise QMosaicEditorDirectionError(
            "WORLD4 zero-route structural proof cannot be rebuilt"
        ) from error
    if dict(proof) != dict(rebuilt):
        raise QMosaicEditorDirectionError(
            "WORLD4 zero-route structural proof digest/closure differs"
        )
    return rebuilt


def build_nonauthoritative_sketch_diagnostic(
    *,
    sp_rank: int,
    role: str,
    adapter_off: Any,
    enabled_zero_b: Any,
) -> Mapping[str, Any]:
    """Describe separate-forward sketch drift without using it as a gate."""

    import torch

    if (
        type(sp_rank) is not int
        or not 0 <= sp_rank < SP_SIZE
        or role not in ("action", "noop")
        or not isinstance(adapter_off, torch.Tensor)
        or not isinstance(enabled_zero_b, torch.Tensor)
        or adapter_off.dtype != torch.float32
        or enabled_zero_b.dtype != torch.float32
        or adapter_off.shape != enabled_zero_b.shape
        or adapter_off.requires_grad
        or enabled_zero_b.requires_grad
        or not bool(torch.isfinite(adapter_off).all().item())
        or not bool(torch.isfinite(enabled_zero_b).all().item())
    ):
        raise QMosaicEditorDirectionError(
            "non-authoritative sketch diagnostic coordinate differs"
        )
    difference = (enabled_zero_b - adapter_off).detach()
    left_sha = qmosaic.tensor_sha256(
        adapter_off, label=f"rank{sp_rank} {role} adapter-off sketch"
    )
    right_sha = qmosaic.tensor_sha256(
        enabled_zero_b, label=f"rank{sp_rank} {role} enabled-zero-B sketch"
    )
    return {
        "sp_rank": sp_rank,
        "role": role,
        "shape": list(map(int, adapter_off.shape)),
        "dtype": str(adapter_off.dtype),
        "adapter_off_tensor_sha256": left_sha,
        "enabled_zero_b_tensor_sha256": right_sha,
        "numeric_exact_equal": bool(torch.equal(adapter_off, enabled_zero_b)),
        "raw_byte_exact_equal": left_sha == right_sha,
        "numeric_mismatch_element_count": int(
            torch.count_nonzero(difference).item()
        ),
        "max_absolute_difference": float(difference.abs().max().item()),
        "confounded_by_separate_forward_rocm_reduction_and_route_mode": True,
        "authoritative_for_zero_route_identity": False,
        "allclose_or_tolerance_used": False,
    }


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise QMosaicEditorDirectionError("receipt is already sealed")
    value = dict(unsigned)
    return {**value, "receipt_digest": object_sha256(value)}


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise QMosaicEditorDirectionError("receipt output must be a fresh absolute path")
    payload = _canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _strict_registry_cell(
    registry_path: str | Path, *, cell_id: str, query_seed: int
) -> Mapping[str, Any]:
    path = Path(registry_path)
    if _file_sha256(path) != FIXED_REGISTRY_SHA256:
        raise QMosaicEditorDirectionError("fixed Q-MOSAIC registry bytes changed")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QMosaicEditorDirectionError("fixed registry JSON differs") from error
    cells = value.get("cells") if isinstance(value, Mapping) else None
    contract = value.get("contract") if isinstance(value, Mapping) else None
    if (
        value.get("schema_version")
        != "bernini-self-imagined-motion-cotangent-core2-registry-v1"
        or not isinstance(cells, list)
        or len(cells) != 2
        or not isinstance(contract, Mapping)
        or contract.get("native_schedule_index") != NATIVE_SCHEDULE_INDEX
        or contract.get("native_timestep") != NATIVE_TIMESTEP
        or contract.get("frame_count") != EXPECTED_FRAMES
        or contract.get("relative_l2_dose") != RELATIVE_L2_DOSE
        or contract.get("parameter_update") is not False
        or contract.get("optimizer") is not False
    ):
        raise QMosaicEditorDirectionError("fixed registry contract differs")
    by_id = {row.get("cell_id"): row for row in cells if isinstance(row, Mapping)}
    if set(by_id) != set(FIXED_QUERY_SEEDS) or cell_id not in by_id:
        raise QMosaicEditorDirectionError("cell is outside fixed dog/human registry")
    for expected_id, seeds in FIXED_QUERY_SEEDS.items():
        if by_id[expected_id].get("query_seeds") != list(seeds):
            raise QMosaicEditorDirectionError("fixed dog/human query seeds changed")
    if query_seed not in FIXED_QUERY_SEEDS[cell_id]:
        raise QMosaicEditorDirectionError("query seed is not fixed for selected cell")
    cell = by_id[cell_id]
    if (
        cell.get("selected_before_generation") is not True
        or cell.get("latent_shape")[:3] != [1, 16, 21]
        or _SHA256_RE.fullmatch(str(cell.get("source_video_sha256"))) is None
        or _SHA256_RE.fullmatch(str(cell.get("action_caption_utf8_sha256"))) is None
        or _SHA256_RE.fullmatch(str(cell.get("noop_caption_utf8_sha256"))) is None
    ):
        raise QMosaicEditorDirectionError("fixed registry cell binding differs")
    return dict(cell)


def construct_symmetric_latents(
    *, base_clean_latent: Any, clean_vjp: Any
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    """Construct the only allowed FP32 ``base +/- 0.01 ||base|| q`` arms."""

    import torch

    if (
        not isinstance(base_clean_latent, torch.Tensor)
        or not isinstance(clean_vjp, torch.Tensor)
        or base_clean_latent.dtype != torch.float32
        or clean_vjp.dtype != torch.float32
        or base_clean_latent.shape != clean_vjp.shape
        or base_clean_latent.ndim != 5
        or tuple(map(int, base_clean_latent.shape[:3])) != (1, 16, 21)
        or base_clean_latent.requires_grad
        or clean_vjp.requires_grad
        or not bool(torch.isfinite(base_clean_latent).all().item())
        or not bool(torch.isfinite(clean_vjp).all().item())
    ):
        raise QMosaicEditorDirectionError("clean latent/VJP coordinate differs")
    base = base_clean_latent.detach().float().cpu().contiguous().clone()
    gradient = clean_vjp.detach().float().cpu().contiguous()
    base_norm = torch.linalg.vector_norm(base)
    gradient_norm = torch.linalg.vector_norm(gradient)
    if (
        not bool(torch.isfinite(base_norm).item())
        or not bool(torch.isfinite(gradient_norm).item())
        or float(base_norm.item()) <= 0.0
        or float(gradient_norm.item()) <= 0.0
    ):
        raise QMosaicEditorDirectionError("clean latent/VJP norm is zero or non-finite")
    direction = (gradient / gradient_norm).contiguous()
    scale = torch.tensor(RELATIVE_L2_DOSE, dtype=torch.float32) * base_norm
    plus = (base + scale * direction).contiguous()
    minus = (base - scale * direction).contiguous()
    expected_plus = (base + scale * direction).contiguous()
    expected_minus = (base - scale * direction).contiguous()
    if not torch.equal(plus, expected_plus) or not torch.equal(minus, expected_minus):
        raise QMosaicEditorDirectionError("signed latent construction is not exact FP32")
    plus_delta = plus.double() - base.double()
    minus_delta = minus.double() - base.double()
    midpoint_error = float(((plus.double() + minus.double()) * 0.5 - base.double()).abs().max().item())
    antisymmetry_error = float((plus_delta + minus_delta).abs().max().item())
    scale_value = float(scale.item())
    tolerance = max(2.0e-6 * max(scale_value, 1.0), 2.0e-7)
    plus_norm = float(torch.linalg.vector_norm(plus_delta).item())
    minus_norm = float(torch.linalg.vector_norm(minus_delta).item())
    norm_error = abs(plus_norm - minus_norm)
    if (
        midpoint_error > tolerance
        or antisymmetry_error > 2.0 * tolerance
        or norm_error > 2.0e-6 * max(scale_value, 1.0)
    ):
        raise QMosaicEditorDirectionError("P+/P- latent symmetry failed")
    evidence = {
        "formula": "q=g/l2(g);scale=0.01*l2(base);plus=base+scale*q;minus=base-scale*q",
        "relative_l2_dose": RELATIVE_L2_DOSE,
        "base_tensor_sha256": qmosaic.tensor_sha256(base, label="direction base"),
        "clean_vjp_tensor_sha256": qmosaic.tensor_sha256(gradient, label="direction VJP"),
        "direction_tensor_sha256": qmosaic.tensor_sha256(direction, label="direction unit"),
        "plus_tensor_sha256": qmosaic.tensor_sha256(plus, label="direction plus"),
        "minus_tensor_sha256": qmosaic.tensor_sha256(minus, label="direction minus"),
        "base_l2_norm": float(base_norm.item()),
        "clean_vjp_l2_norm": float(gradient_norm.item()),
        "direction_l2_norm": float(torch.linalg.vector_norm(direction.double()).item()),
        "absolute_dose_l2": scale_value,
        "plus_delta_l2": plus_norm,
        "minus_delta_l2": minus_norm,
        "delta_norm_symmetry_absolute_error": norm_error,
        "midpoint_max_abs_error": midpoint_error,
        "delta_antisymmetry_max_abs_error": antisymmetry_error,
        "symmetry_tolerance": tolerance,
        "formula_recomputed_exact_fp32": True,
        "latent_symmetry_passed": True,
    }
    return base, plus, minus, evidence


def _probe_exact81_25fps(path: str | Path) -> Mapping[str, Any]:
    """Decode a published MP4 and prove one 25-fps, 81-frame stream."""

    source = Path(path)
    try:
        probe = dict(
            qmosaic._probe_decode_exact81(source)  # noqa: SLF001 - pinned core proof
        )
    except Exception as error:
        diagnostic = str(error).replace(str(source), "<redacted-media-path>")
        if len(diagnostic) > 4096:
            diagnostic = diagnostic[:4096] + "...[truncated]"
        raise QMosaicEditorDirectionError(
            "published arm portable media probe failed "
            f"[{type(error).__name__}: {diagnostic}]"
        ) from error
    if (
        set(probe) != qmosaic.EXACT81_MEDIA_PROBE_FIELDS
        or probe.get("schema_version")
        != qmosaic.EXACT81_MEDIA_PROBE_SCHEMA_VERSION
        or probe.get("video_stream_count") != 1
        or probe.get("container_stream_count") != 1
        or probe.get("pyav_decoded_frame_count") != EXPECTED_FRAMES
        or probe.get("bundled_ffmpeg_framemd5_frame_count")
        != EXPECTED_FRAMES
        or probe.get("pyav_pts_cadence_rational") != "1/25"
        or probe.get("pyav_exact_25fps_pts_cadence") is not True
    ):
        raise QMosaicEditorDirectionError(
            "published arm portable media probe closure differs"
        )
    rate = str(probe.get("avg_frame_rate"))
    try:
        numerator, denominator = (int(item) for item in rate.split("/", 1))
    except (ValueError, TypeError) as error:
        raise QMosaicEditorDirectionError("published arm frame rate differs") from error
    if denominator <= 0 or numerator != EXPECTED_FPS * denominator:
        raise QMosaicEditorDirectionError("published arm is not exactly 25 fps")
    value = {**probe, "fps_exact_integer": EXPECTED_FPS}
    if set(value) != EXACT81_25FPS_PROBE_FIELDS:
        raise QMosaicEditorDirectionError(
            "published arm exact81@25 probe field closure differs"
        )
    return value


def _all_rank_equal(value: Any, *, dist: Any, label: str) -> tuple[Any, ...]:
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, value)
    if rows != [value] * WORLD_SIZE:
        raise QMosaicEditorDirectionError(f"{label} differs across WORLD4")
    return tuple(rows)


def validate_terminal_full_seal_receipt(
    value: Any, *, sp_rank: int
) -> Mapping[str, Any]:
    """Validate the public terminal deep assertion before any publication."""

    if not isinstance(value, Mapping):
        raise QMosaicEditorDirectionError("terminal full seal receipt is absent")
    unsigned = dict(value)
    digest = unsigned.pop("digest", None)
    expected_fields = {
        "schema_version",
        "complete_model_runtime_seal_digest",
        "checkpoint_content_receipt_digest",
        "authenticated_runtime_input_receipt_digest",
        "deep_full_byte_revalidated",
        "every_model_parameter_and_buffer_byte_revalidated",
        "checkpoint_tree_revalidated",
        "signed_runtime_input_revalidated",
        "publication_authority",
    }
    if (
        set(unsigned) != expected_fields
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or object_sha256(unsigned) != digest
        or unsigned.get("schema_version") != TERMINAL_FULL_SEAL_SCHEMA
        or any(
            _SHA256_RE.fullmatch(str(unsigned.get(name))) is None
            for name in (
                "complete_model_runtime_seal_digest",
                "checkpoint_content_receipt_digest",
                "authenticated_runtime_input_receipt_digest",
            )
        )
        or value.get("deep_full_byte_revalidated") is not True
        or value.get("every_model_parameter_and_buffer_byte_revalidated") is not True
        or value.get("checkpoint_tree_revalidated") is not True
        or value.get("signed_runtime_input_revalidated") is not True
        or value.get("publication_authority")
        != "integrity_only_no_semantic_or_update_authority"
    ):
        raise QMosaicEditorDirectionError("terminal full seal receipt differs")
    return {
        "sp_rank": sp_rank,
        "terminal_full_seal_receipt_digest": digest,
        "deep_full_byte_revalidated": True,
    }


def assert_terminal_full_seal_before_publish(
    native_runner: Any, *, sp_rank: int
) -> Mapping[str, Any]:
    """Invoke the core's non-optional terminal assertion with no callback path."""

    terminal_method = getattr(native_runner, "assert_terminal_runtime_live", None)
    if not callable(terminal_method):
        raise QMosaicEditorDirectionError(
            "native runner lacks the terminal full-seal assertion"
        )
    return validate_terminal_full_seal_receipt(terminal_method(), sp_rank=sp_rank)


def build_run_receipt(
    *,
    cell: Mapping[str, Any],
    query_seed: int,
    owner_receipt: Mapping[str, Any],
    editor_receipt: Mapping[str, Any],
    score_receipt: Mapping[str, Any],
    clean_vjp_receipt: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
    collective_receipt: Mapping[str, Any],
    runner_contract: Mapping[str, Any],
    parity_evidence: Mapping[str, Any],
    direction_evidence: Mapping[str, Any],
    terminal_full_seal_evidence: Sequence[Mapping[str, Any]],
    arm_artifacts: Sequence[Mapping[str, Any]],
    parameter_invariance: Mapping[str, Any],
    method_source_revision: str,
    method_source_archive_sha256: str,
    _p_qmosaic: bool = False,
) -> Mapping[str, Any]:
    """Build the closed receipt; semantic status is method-owned and fixed."""

    if [row.get("role") for row in arm_artifacts] != list(ARM_ORDER):
        raise QMosaicEditorDirectionError("all fixed base/plus/minus arms are required")
    if (
        editor_receipt.get("method_source_revision")
        != method_source_revision
        or editor_receipt.get("method_source_archive_sha256")
        != method_source_archive_sha256
        or _SHA256_RE.fullmatch(
            str(editor_receipt.get("materialization_receipt_digest"))
        )
        is None
        or _SHA256_RE.fullmatch(
            str(editor_receipt.get("materialization_receipt_file_sha256"))
        )
        is None
        or not isinstance(
            editor_receipt.get("materialization_receipt_path"), str
        )
    ):
        raise QMosaicEditorDirectionError(
            "editor materialization/source authority differs from run source"
        )
    if (
        len(terminal_full_seal_evidence) != SP_SIZE
        or [
            row.get("sp_rank") if isinstance(row, Mapping) else None
            for row in terminal_full_seal_evidence
        ]
        != list(range(SP_SIZE))
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "sp_rank",
                "terminal_full_seal_receipt_digest",
                "deep_full_byte_revalidated",
            }
            or row.get("deep_full_byte_revalidated") is not True
            or _SHA256_RE.fullmatch(
                str(row.get("terminal_full_seal_receipt_digest"))
            )
            is None
            for row in terminal_full_seal_evidence
        )
    ):
        raise QMosaicEditorDirectionError("WORLD4 terminal full seal evidence differs")
    if len({row.get("mp4_path") for row in arm_artifacts}) != 3:
        raise QMosaicEditorDirectionError("published arm paths alias")
    if (
        parity_evidence.get("b0_z0_predecode_exact_parity") is not True
        or parity_evidence.get(
            "native_zero_lora_structural_forward_identity_proven"
        )
        is not True
        or parity_evidence.get(
            "separate_off_enabled_sketch_comparison_used_for_authority"
        )
        is not False
        or parity_evidence.get(
            "b0_z0_and_all_direction_arms_share_source_noise_prompt_scheduler"
        )
        is not True
        or (
            not _p_qmosaic
            and direction_evidence.get("latent_symmetry_passed") is not True
        )
        or parameter_invariance.get("parameter_bytes_unchanged") is not True
        or parameter_invariance.get("lora_b_exact_zero_after") is not True
    ):
        raise QMosaicEditorDirectionError("method-owned numerical gate is incomplete")
    p_profile: Any = None
    if _p_qmosaic:
        import p_qmosaic_direction_envelope_v1 as p_profile

        by_role = {row["role"]: row for row in arm_artifacts}
        try:
            p_profile.validate_envelope(
                direction_evidence,
                cell_id=str(cell["cell_id"]),
                query_seed=query_seed,
                clean_vjp_receipt_digest=str(clean_vjp_receipt.get("digest")),
                clean_vjp_value_sha256=str(clean_vjp_receipt.get("value_sha256")),
                base_tensor_sha256=str(parity_evidence.get("b0_tensor_sha256")),
                plus_tensor_sha256=str(by_role["plus"]["latent_tensor_sha256"]),
                minus_tensor_sha256=str(by_role["minus"]["latent_tensor_sha256"]),
            )
        except p_profile.PQMosaicDirectionEnvelopeError as error:
            raise QMosaicEditorDirectionError(str(error)) from error
        if by_role["base"]["latent_tensor_sha256"] != parity_evidence.get(
            "b0_tensor_sha256"
        ):
            raise QMosaicEditorDirectionError("P-Q base tensor hash differs")
    validate_world4_zero_route_proof(
        parity_evidence.get("world4_zero_lora_structural_proof"),
        parity=parity_evidence,
    )
    semantic = {
        "action": SEMANTIC_UNASSESSED,
        "identity": SEMANTIC_UNASSESSED,
        "camera": SEMANTIC_UNASSESSED,
        "background": SEMANTIC_UNASSESSED,
        "quality": SEMANTIC_UNASSESSED,
        "method_owned_decoded_evaluator_available": False,
        "caller_boolean_or_callback_consumed": False,
        "self_reported_semantic_score_consumed": False,
        "decoded_semantic_gate_passed": False,
    }
    unsigned = {
        "schema_version": p_profile.RUN_RECEIPT_SCHEMA if _p_qmosaic else RUN_RECEIPT_SCHEMA,
        "method_name": p_profile.METHOD_NAME if _p_qmosaic else METHOD_NAME,
        "experiment_scope": {
            "classification": "ENGINEERING_ONLY" if _p_qmosaic else "ENGINEERING_SMOKE_ONLY",
            "scientific_evidence_authority": False,
            "semantic_authority": False,
            "lora_or_parameter_update_authority": False,
        },
        "registry": {
            "file_sha256": FIXED_REGISTRY_SHA256,
            "cell_id": cell["cell_id"],
            "source_iid": cell["source_iid"],
            "source_video_sha256": cell["source_video_sha256"],
            "action_family_id": cell["action_family_id"],
            "query_seed": query_seed,
            "fixed_query_seeds_for_cell": list(FIXED_QUERY_SEEDS[cell["cell_id"]]),
        },
        "native_coordinate": {
            "world_size": WORLD_SIZE,
            "ulysses_size": SP_SIZE,
            "schedule_index": NATIVE_SCHEDULE_INDEX,
            "timestep": NATIVE_TIMESTEP,
            "frame_count": EXPECTED_FRAMES,
            "fps": EXPECTED_FPS,
            "relative_l2_dose": RELATIVE_L2_DOSE,
            "owner_packet_receipt_digest": owner_receipt["digest"],
            "editor_runtime_input_receipt_digest": editor_receipt["digest"],
            "editor_materialization_receipt_path": editor_receipt[
                "materialization_receipt_path"
            ],
            "editor_materialization_receipt_file_sha256": editor_receipt[
                "materialization_receipt_file_sha256"
            ],
            "editor_materialization_receipt_digest": editor_receipt[
                "materialization_receipt_digest"
            ],
            "editor_method_source_revision": editor_receipt[
                "method_source_revision"
            ],
            "editor_method_source_archive_sha256": editor_receipt[
                "method_source_archive_sha256"
            ],
            "score_cotangent_receipt_digest": score_receipt["digest"],
            "sp4_clean_vjp_receipt_digest": clean_vjp_receipt["digest"],
            "checkpoint_content_receipt_digest": checkpoint_receipt["digest"],
            "sp4_collective_receipt_digest": collective_receipt["digest"],
            "runner_contract_receipt_digest": runner_contract["digest"],
        },
        "predecode_parity": dict(parity_evidence),
        "symmetric_direction": dict(direction_evidence),
        "terminal_full_seal": {
            "called_before_any_mp4_or_receipt_publication": True,
            "deep_full_byte_revalidated": True,
            "rank_receipts": [dict(row) for row in terminal_full_seal_evidence],
        },
        "published_arms": [dict(row) for row in arm_artifacts],
        "all_fixed_arms_published": True,
        "parameter_invariance": dict(parameter_invariance),
        "semantic_assessment": semantic,
        "authorization": {
            "cli_no_lora_vjp_required": True,
            "clean_latent_vjp_executed": True,
            "lora_vjp_requested": False,
            "lora_vjp_executed": False,
            "lora_vjp_authorized": False,
            "optimizer_created": False,
            "parameter_update_authorized": False,
            "parameter_update_performed": False,
            "adapter_checkpoint_written": False,
            "scientific_action_editing_success_claim": False,
        },
        "output_contract": {
            "durable_artifacts": ["run_receipt", "base_mp4", "plus_mp4", "minus_mp4"],
            "latent_or_gradient_tensor_artifact_written": False,
            "receipt_and_video_only": True,
        },
        "method_source": {
            "revision": _sha1(method_source_revision, label="method source revision"),
            "archive_sha256": _sha256(
                method_source_archive_sha256, label="method source archive SHA-256"
            ),
        },
    }
    if _p_qmosaic:
        unsigned["direction_variant"] = dict(p_profile.variant_lock())
    return _seal(unsigned)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", default=FIXED_REGISTRY_SHA256)
    parser.add_argument("--cell-id", choices=tuple(FIXED_QUERY_SEEDS), required=True)
    parser.add_argument("--query-seed", type=int, required=True)
    parser.add_argument("--owner-root", required=True)
    parser.add_argument("--owner-master-receipt", required=True)
    parser.add_argument("--expected-owner-master-receipt-sha256", required=True)
    parser.add_argument("--owner-audit-sidecar", required=True)
    parser.add_argument("--expected-owner-audit-sidecar-sha256", required=True)
    parser.add_argument("--owner-audit-evidence", required=True)
    parser.add_argument("--owner-audit-public-key", required=True)
    parser.add_argument("--expected-owner-audit-public-key-sha256", required=True)
    parser.add_argument("--owner-cell-root", required=True)
    parser.add_argument("--owner-cell-receipt", required=True)
    parser.add_argument("--expected-owner-cell-receipt-sha256", required=True)
    parser.add_argument("--editor-receipt", required=True)
    parser.add_argument("--expected-editor-receipt-sha256", required=True)
    parser.add_argument("--editor-public-key", required=True)
    parser.add_argument("--expected-editor-public-key-sha256", required=True)
    parser.add_argument("--editor-artifact-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument("--no-lora-vjp", action="store_true")
    return parser


def validate_cli(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.expected_registry_sha256 != FIXED_REGISTRY_SHA256:
        raise QMosaicEditorDirectionError("registry SHA is fixed, not caller-selectable")
    if args.no_lora_vjp is not True:
        raise QMosaicEditorDirectionError("this phase requires explicit --no-lora-vjp")
    if args.query_seed not in FIXED_QUERY_SEEDS.get(args.cell_id, ()):
        raise QMosaicEditorDirectionError("cell/query seed is outside preregistration")
    _sha1(args.expected_bernini_commit, label="expected Bernini commit")
    _sha1(args.expected_veomni_commit, label="expected VeOmni commit")
    _sha1(args.method_source_revision, label="method source revision")
    for name in (
        "expected_checkpoint_content_manifest_sha256",
        "expected_owner_master_receipt_sha256",
        "expected_owner_audit_sidecar_sha256",
        "expected_owner_audit_public_key_sha256",
        "expected_owner_cell_receipt_sha256",
        "expected_editor_receipt_sha256",
        "expected_editor_public_key_sha256",
        "method_source_archive_sha256",
    ):
        _sha256(getattr(args, name), label=name)
    output = Path(args.output_dir)
    if (
        not output.is_absolute()
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
    ):
        raise QMosaicEditorDirectionError("output directory must be a fresh absolute path")
    return _strict_registry_cell(
        args.registry, cell_id=args.cell_id, query_seed=args.query_seed
    )


def _validate_vae_decoded_clip(
    decoded: Any,
    *,
    role: str,
    expected_height: int,
    expected_width: int,
) -> Mapping[str, Any]:
    """Validate pinned Bernini's numpy ``_vae_decode``/``save_output`` contract."""

    import numpy as np

    expected_shape = (
        EXPECTED_FRAMES,
        int(expected_height),
        int(expected_width),
        3,
    )
    if (
        role not in ARM_ORDER
        or expected_height <= 0
        or expected_width <= 0
        or not isinstance(decoded, np.ndarray)
        or decoded.ndim != 4
        or tuple(map(int, decoded.shape)) != expected_shape
    ):
        raise QMosaicEditorDirectionError(
            f"{role} VAE decode differs from numpy [81,H,W,3] contract"
        )
    # ``bernini.pipeline._vae_decode`` documents a normalized numpy clip and
    # ``bernini.io_utils.save_output`` consumes non-uint8 arrays as [0, 1]
    # pixels.  Do not silently accept tensors, object/string arrays, integer
    # encodings, complex values, or a decoder whose numerical contract drifted.
    if decoded.dtype.kind != "f" or decoded.dtype.itemsize not in (2, 4, 8):
        raise QMosaicEditorDirectionError(
            f"{role} VAE decode dtype differs from normalized floating numpy contract"
        )
    if not bool(np.isfinite(decoded).all()):
        raise QMosaicEditorDirectionError(f"{role} VAE decode contains non-finite values")
    value_min = float(decoded.min())
    value_max = float(decoded.max())
    if value_min < 0.0 or value_max > 1.0:
        raise QMosaicEditorDirectionError(
            f"{role} VAE decode differs from normalized [0,1] save_output contract"
        )
    return {
        "array_type": "numpy.ndarray",
        "shape": list(expected_shape),
        "dtype": str(decoded.dtype),
        "finite": True,
        "normalized_zero_one": True,
        "value_min": value_min,
        "value_max": value_max,
    }


def _decode_arms_rank_zero(
    *,
    checkpoint: Path,
    output_dir: Path,
    published_output_dir: Optional[Path] = None,
    arms: Mapping[str, Any],
    device: Any,
) -> list[Mapping[str, Any]]:
    import torch
    from diffusers.models import AutoencoderKLWan
    from bernini.io_utils import save_output
    from bernini.pipeline import _vae_decode

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False).to(device)
    artifacts: list[Mapping[str, Any]] = []
    try:
        for role in ARM_ORDER:
            latent = arms[role].to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                decoded = _vae_decode(vae, latent)
            _validate_vae_decoded_clip(
                decoded,
                role=role,
                expected_height=int(latent.shape[-2]) * 8,
                expected_width=int(latent.shape[-1]) * 8,
            )
            path = output_dir / f"{role}.mp4"
            published_path = (
                published_output_dir / f"{role}.mp4"
                if published_output_dir is not None
                else path
            )
            value_audit.save_video_atomically(
                decoded,
                path,
                fps=EXPECTED_FPS,
                save_output_fn=save_output,
            )
            probe = _probe_exact81_25fps(path)
            artifacts.append(
                {
                    "role": role,
                    "mp4_path": str(published_path),
                    "mp4_file_sha256": _file_sha256(path),
                    "latent_tensor_sha256": qmosaic.tensor_sha256(
                        arms[role], label=f"published {role} latent"
                    ),
                    "decode_seed": int(arms["query_seed"]),
                    "frame_count": EXPECTED_FRAMES,
                    "fps": EXPECTED_FPS,
                    "decode_probe": dict(probe),
                }
            )
    finally:
        vae.to("cpu")
    if (
        [row["role"] for row in artifacts] != list(ARM_ORDER)
        or len({row["mp4_file_sha256"] for row in artifacts}) != len(ARM_ORDER)
        or len({row["latent_tensor_sha256"] for row in artifacts}) != len(ARM_ORDER)
    ):
        raise QMosaicEditorDirectionError("fixed decoded arms alias or are incomplete")
    return artifacts


def run(
    args: argparse.Namespace, *, _p_qmosaic: bool = False
) -> Mapping[str, Any]:
    cell = validate_cli(args)
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
        raise QMosaicEditorDirectionError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise QMosaicEditorDirectionError("pinned Bernini attention geometry differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import get_parallel_state, init_parallel_state

    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise QMosaicEditorDirectionError("runner requires one AUH WORLD4/SP4 ROCm group")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    output = Path(args.output_dir)
    observer: Any = None
    action_handle: Any = None
    try:
        checkpoint_packet = qmosaic.load_validated_checkpoint_content_manifest(
            checkpoint_root=checkpoint,
            content_manifest_path=args.checkpoint_content_manifest,
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
            expected_file_count=CHECKPOINT_CONTENT_FILE_COUNT,
        )
        owner = qmosaic.load_authenticated_owner_quotient_packet(
            registry=args.registry,
            expected_registry_sha256=FIXED_REGISTRY_SHA256,
            owner_root=args.owner_root,
            owner_master_receipt=args.owner_master_receipt,
            expected_owner_master_receipt_sha256=args.expected_owner_master_receipt_sha256,
            audit_sidecar=args.owner_audit_sidecar,
            expected_audit_sidecar_sha256=args.expected_owner_audit_sidecar_sha256,
            audit_evidence=args.owner_audit_evidence,
            audit_public_key=args.owner_audit_public_key,
            expected_audit_public_key_sha256=args.expected_owner_audit_public_key_sha256,
            cell_root=args.owner_cell_root,
            receipt_path=args.owner_cell_receipt,
            expected_receipt_file_sha256=args.expected_owner_cell_receipt_sha256,
            query_seed=args.query_seed,
        )
        if (
            owner.cell_id != args.cell_id
            or owner.query_seed != args.query_seed
            or owner.source_iid != cell["source_iid"]
            or owner.source_video_sha256 != cell["source_video_sha256"]
        ):
            raise QMosaicEditorDirectionError("owner differs from fixed registry cell")
        runtime_inputs = qmosaic.load_authenticated_editor_runtime_input_packet(
            receipt_path=args.editor_receipt,
            expected_receipt_file_sha256=args.expected_editor_receipt_sha256,
            public_key_path=args.editor_public_key,
            expected_public_key_file_sha256=args.expected_editor_public_key_sha256,
            artifact_root=args.editor_artifact_root,
            owner=owner,
            checkpoint=checkpoint_packet,
        )
        editor_source_binding = runtime_inputs.receipt()
        if (
            editor_source_binding.get("method_source_revision")
            != args.method_source_revision
            or editor_source_binding.get("method_source_archive_sha256")
            != args.method_source_archive_sha256
        ):
            raise QMosaicEditorDirectionError(
                "editor packet source archive differs from direction source"
            )

        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
        # Runtime prompt conditions are authenticated packet tensors; the T5
        # encoder is not used by the direction graph and need not occupy HBM.
        renderer.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()
        diffusion = renderer.diff_dec
        transformer = diffusion.transformer
        if transformer is None or diffusion.transformer_2 is not None:
            raise QMosaicEditorDirectionError("renderer is not frozen transformer_1-only")
        action_handle = qmosaic.install_core16_fixed_a_b_only_action_lora(transformer)
        action_handle.assert_fixed_gauge()
        parameter_before = action_handle.state_digest()
        b_before = action_handle.b_parameter_state_sha256()
        snapshot = action_handle.adapter_state_snapshot()

        clean_shape = runtime_inputs.tensors["clean_latent"].shape
        patch_positions = (int(clean_shape[3]) // 2) * (int(clean_shape[4]) // 2)
        observer = qmosaic.Block15TargetSuffixObserver(
            transformer,
            spatial_sketch=qmosaic.make_fixed_spatial_sketch(patch_positions),
        )
        observer.install()
        collective = qmosaic.authenticate_live_bernini_sp4_collective(
            parallel_state=get_parallel_state()
        )
        runner = qmosaic.NativeSharedStepSP4ReplayRunner(
            diffusion=diffusion,
            transformer=transformer,
            owner=owner,
            runtime_inputs=runtime_inputs,
            action_handle=action_handle,
            observer=observer,
            sp4_collective=collective,
            sp_rank=collective.sp_rank,
            checkpoint_content=checkpoint_packet,
        )
        b0 = runtime_inputs.tensors["clean_latent"].detach().float().cpu().contiguous()
        z0 = runner.clean_latent.detach().float().cpu().contiguous()
        if not torch.equal(b0, z0):
            raise QMosaicEditorDirectionError("B0/Z0 predecode clean-latent parity failed")
        parity = {
            "coordinate": "signed_editor_runtime_clean_latent_before_any_decode",
            "b0_tensor_sha256": qmosaic.tensor_sha256(b0, label="B0 predecode"),
            "z0_tensor_sha256": qmosaic.tensor_sha256(z0, label="Z0 predecode"),
            "b0_z0_predecode_exact_parity": True,
            "source_latent_tensor_sha256": qmosaic.tensor_sha256(
                runner.source_latent, label="shared direction source latent"
            ),
            "official_initial_noise_tensor_sha256": qmosaic.tensor_sha256(
                runner.initial_noise, label="shared direction official noise"
            ),
            "action_prompt_sha256": owner.action_prompt_sha256,
            "noop_prompt_sha256": owner.noop_prompt_sha256,
            "prompt_condition_binding_digest": qmosaic.object_sha256(
                dict(runner.prompt_condition_binding)
            ),
            "native_schedule_index": NATIVE_SCHEDULE_INDEX,
            "native_timestep": NATIVE_TIMESTEP,
            "checkpoint_content_receipt_digest": (
                runner.checkpoint_content_receipt_digest
            ),
            "b0_z0_and_all_direction_arms_share_source_noise_prompt_scheduler": True,
        }

        editor, replay_session = runner.seal_editor_packet(owner)
        # A separate ROCm forward cannot be a byte-level oracle for another
        # forward: the block15 sketch uses repeated-index GPU reductions and
        # may legitimately differ in accumulation order.  Instead, prove the
        # zero adapter locally inside each of the 32 wrappers during exactly
        # one enabled action forward and one enabled no-op forward per rank.
        # The custom zero-point autograd edge returns base bytes unchanged but
        # retains the real derivative into LoRA-B.
        with action_handle.capture_zero_route_proof(
            role="action", sp_rank=collective.sp_rank
        ) as action_proof_holder:
            connected_zero_action = replay_session.replay(
                owner=owner, role="action", adapter_enabled=True
            )
        action_proof = action_proof_holder.require_receipt()
        zero_route_action = connected_zero_action.detach().float().contiguous()
        del connected_zero_action

        with action_handle.capture_zero_route_proof(
            role="noop", sp_rank=collective.sp_rank
        ) as noop_proof_holder:
            connected_zero_noop = replay_session.replay(
                owner=owner, role="noop", adapter_enabled=True
            )
        noop_proof = noop_proof_holder.require_receipt()
        zero_route_noop = connected_zero_noop.detach().float().contiguous()
        del connected_zero_noop

        local_route_evidence = {
            "sp_rank": collective.sp_rank,
            "action_structural_proof": action_proof,
            "noop_structural_proof": noop_proof,
            "separate_forward_sketch_diagnostics": [
                build_nonauthoritative_sketch_diagnostic(
                    sp_rank=collective.sp_rank,
                    role="action",
                    adapter_off=editor.local_action_measurement,
                    enabled_zero_b=zero_route_action,
                ),
                build_nonauthoritative_sketch_diagnostic(
                    sp_rank=collective.sp_rank,
                    role="noop",
                    adapter_off=editor.local_noop_measurement,
                    enabled_zero_b=zero_route_noop,
                ),
            ],
        }
        route_evidence_rows: list[Any] = [None] * SP_SIZE
        dist.all_gather_object(route_evidence_rows, local_route_evidence)
        if [
            row.get("sp_rank") if isinstance(row, Mapping) else None
            for row in route_evidence_rows
        ] != list(range(SP_SIZE)):
            raise QMosaicEditorDirectionError(
                "WORLD4 zero-route evidence rank order differs"
            )
        world4_zero_proof = build_world4_zero_route_proof(
            action_rows=[row["action_structural_proof"] for row in route_evidence_rows],
            noop_rows=[row["noop_structural_proof"] for row in route_evidence_rows],
            parity=parity,
        )
        parity["world4_zero_lora_structural_proof"] = world4_zero_proof
        parity["native_zero_lora_structural_forward_identity_proven"] = True
        parity["separate_off_enabled_sketch_diagnostic_by_sp_rank"] = [
            {
                "sp_rank": row["sp_rank"],
                "roles": row["separate_forward_sketch_diagnostics"],
            }
            for row in route_evidence_rows
        ]
        parity["separate_off_enabled_sketch_comparison_used_for_authority"] = False
        if collective.sp_rank == 0:
            print(
                _canonical_json_bytes(
                    {
                        "schema_version": (
                            "bernini-qmosaic-separate-forward-sketch-diagnostic-v1"
                        ),
                        "authoritative_for_zero_route_identity": False,
                        "rows": parity[
                            "separate_off_enabled_sketch_diagnostic_by_sp_rank"
                        ],
                    }
                ).decode("ascii"),
                flush=True,
            )
        score_packet = qmosaic.score_cotangent_from_authenticated_packets(owner, editor)
        local_clean = qmosaic.replay_score_cotangent(
            score_packet,
            owner=owner,
            replay_session=replay_session,
            vjp_target="clean_latent",
            sp_rank=collective.sp_rank,
            action_handle=action_handle,
            clean_latent=runner.clean_latent,
        )
        clean_vjp = runner.sum_rank_local_vjp(local_clean)
        clean_vjp.assert_live()
        _all_rank_equal(
            clean_vjp.receipt()["digest"], dist=dist, label="SP4 clean VJP receipt"
        )
        if _p_qmosaic:
            import p_qmosaic_direction_envelope_v1 as p_profile

            try:
                base, plus, minus, direction_evidence = p_profile.construct(
                    cell_id=args.cell_id,
                    base_clean_latent=z0,
                    clean_vjp_row=clean_vjp,
                )
            except p_profile.PQMosaicDirectionEnvelopeError as error:
                raise QMosaicEditorDirectionError(str(error)) from error
            _all_rank_equal(
                direction_evidence["receipt_digest"],
                dist=dist,
                label="P-Q direction evidence",
            )
        else:
            base, plus, minus, direction_evidence = construct_symmetric_latents(
                base_clean_latent=z0,
                clean_vjp=clean_vjp.values.detach().float().cpu().contiguous(),
            )
        latent_identity = {
            role: qmosaic.tensor_sha256(value, label=f"WORLD4 {role} latent")
            for role, value in {"base": base, "plus": plus, "minus": minus}.items()
        }
        _all_rank_equal(latent_identity, dist=dist, label="symmetric latent identity")
        if not action_handle.adapter_state_matches(snapshot):
            raise QMosaicEditorDirectionError("clean direction replay changed Action-LoRA bytes")
        parameter_after = action_handle.state_digest()
        b_after = action_handle.b_parameter_state_sha256()
        if parameter_after != parameter_before or b_after != b_before:
            raise QMosaicEditorDirectionError("parameter bytes changed during direction run")
        action_handle.assert_fixed_gauge()
        parameter_invariance = {
            "action_lora_state_sha256_before": parameter_before,
            "action_lora_state_sha256_after": parameter_after,
            "lora_b_state_sha256_before": b_before,
            "lora_b_state_sha256_after": b_after,
            "parameter_bytes_unchanged": True,
            "lora_b_exact_zero_before": True,
            "lora_b_exact_zero_after": True,
            "optimizer_created": False,
            "parameter_update_performed": False,
        }

        # Capture the immutable bindings first.  Some of these public receipt
        # accessors perform their own live checks; the terminal assertion below
        # must remain the final deep check before publication.
        owner_receipt = owner.receipt()
        editor_input_receipt = editor_source_binding
        score_receipt = score_packet.receipt()
        clean_receipt = clean_vjp.receipt()
        checkpoint_receipt = checkpoint_packet.receipt()
        collective_receipt = collective.receipt()
        # The constructor already established the start-of-session full-byte
        # seal.  This intermediate receipt deliberately uses the cheap
        # metadata/version/storage-pointer check; the mandatory terminal call
        # immediately below performs the only end-of-session full-byte pass.
        runner_contract = runner.contract_receipt(deep=False)

        # This public core assertion deliberately performs the expensive full
        # model-byte, checkpoint-tree and signed-runtime-input revalidation
        # exactly once at the terminal boundary.  No output directory, MP4,
        # run receipt, rename, or semantic decision exists before all four
        # ranks have supplied a valid terminal receipt.
        terminal_local = assert_terminal_full_seal_before_publish(
            runner, sp_rank=collective.sp_rank
        )
        terminal_rows: list[Any] = [None] * SP_SIZE
        dist.all_gather_object(terminal_rows, terminal_local)
        if (
            [row.get("sp_rank") if isinstance(row, Mapping) else None for row in terminal_rows]
            != list(range(SP_SIZE))
            or any(
                row.get("deep_full_byte_revalidated") is not True
                or _SHA256_RE.fullmatch(
                    str(row.get("terminal_full_seal_receipt_digest"))
                )
                is None
                for row in terminal_rows
            )
        ):
            raise QMosaicEditorDirectionError(
                "WORLD4 terminal full-seal consensus differs"
            )

        # The transformer, observer and fixed-gauge adapter are no longer
        # needed after the terminal seal.  Free their HBM on every rank before
        # rank zero loads the FP32 VAE for decoding.
        if observer is not None:
            observer.remove()
            observer = None
        if action_handle is not None and not action_handle.restored:
            action_handle.restore()
        renderer.to("cpu")
        action_handle = None
        del (
            editor,
            replay_session,
            score_packet,
            local_clean,
            clean_vjp,
            zero_route_action,
            zero_route_noop,
            runner,
            runtime_inputs,
            diffusion,
            transformer,
        )
        torch.cuda.empty_cache()
        dist.barrier()
        decode_result: list[Any] = [None]
        if distributed.rank == 0:
            staging_output: Optional[Path] = None
            try:
                staging_output = Path(
                    tempfile.mkdtemp(
                        prefix=f".{output.name}.stage-", dir=output.parent
                    )
                )
                artifacts = _decode_arms_rank_zero(
                    checkpoint=checkpoint,
                    output_dir=staging_output,
                    published_output_dir=output,
                    arms={
                        "base": base,
                        "plus": plus,
                        "minus": minus,
                        "query_seed": args.query_seed,
                    },
                    device=device,
                )
                receipt = build_run_receipt(
                    cell=cell,
                    query_seed=args.query_seed,
                    owner_receipt=owner_receipt,
                    editor_receipt=editor_input_receipt,
                    score_receipt=score_receipt,
                    clean_vjp_receipt=clean_receipt,
                    checkpoint_receipt=checkpoint_receipt,
                    collective_receipt=collective_receipt,
                    runner_contract=runner_contract,
                    parity_evidence=parity,
                    direction_evidence=direction_evidence,
                    terminal_full_seal_evidence=terminal_rows,
                    arm_artifacts=artifacts,
                    parameter_invariance=parameter_invariance,
                    method_source_revision=args.method_source_revision,
                    method_source_archive_sha256=args.method_source_archive_sha256,
                    _p_qmosaic=_p_qmosaic,
                )
                receipt_path = staging_output / RUN_RECEIPT_FILENAME
                _write_create_only_json(receipt_path, receipt)
                receipt_file_sha = _file_sha256(receipt_path)
                os.replace(staging_output, output)
                staging_output = None
                decode_result[0] = {
                    "ok": True,
                    "receipt": receipt,
                    "receipt_file_sha256": receipt_file_sha,
                }
            except Exception as error:
                if (
                    staging_output is not None
                    and staging_output.exists()
                    and not staging_output.is_symlink()
                ):
                    shutil.rmtree(staging_output)
                decode_result[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(decode_result, src=0)
        if not isinstance(decode_result[0], Mapping) or decode_result[0].get("ok") is not True:
            raise QMosaicEditorDirectionError(f"rank-zero decode failed: {decode_result[0]}")
        _all_rank_equal(
            decode_result[0]["receipt"]["receipt_digest"],
            dist=dist,
            label="run receipt",
        )
        return dict(decode_result[0]["receipt"])
    finally:
        if observer is not None:
            observer.remove()
        if action_handle is not None and not action_handle.restored:
            action_handle.restore()
        if dist.is_initialized():
            dist.destroy_process_group()


def run_p_qmosaic(args: argparse.Namespace) -> Mapping[str, Any]:
    return run(args, _p_qmosaic=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run(args)
    if receipt["authorization"]["lora_vjp_authorized"] is not False:
        raise QMosaicEditorDirectionError("LoRA authority unexpectedly changed")
    if os.environ.get("RANK", "0") == "0":
        print(_canonical_json_bytes(receipt).decode("ascii"), flush=True)
    return 0


def main_p_qmosaic(argv: Optional[Sequence[str]] = None) -> int:
    import p_qmosaic_direction_envelope_v1 as p_profile

    args = build_parser().parse_args(argv)
    receipt = run_p_qmosaic(args)
    if (
        receipt.get("schema_version") != p_profile.RUN_RECEIPT_SCHEMA
        or receipt.get("method_name") != p_profile.METHOD_NAME
        or receipt.get("direction_variant") != p_profile.variant_lock()
        or receipt["authorization"]["lora_vjp_authorized"] is not False
    ):
        raise QMosaicEditorDirectionError("P-Q entrypoint lock differs")
    if os.environ.get("RANK", "0") == "0":
        print(_canonical_json_bytes(receipt).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "EXPECTED_FPS",
    "EXPECTED_FRAMES",
    "FIXED_QUERY_SEEDS",
    "FIXED_REGISTRY_SHA256",
    "METHOD_NAME",
    "QMosaicEditorDirectionError",
    "RELATIVE_L2_DOSE",
    "RUN_RECEIPT_SCHEMA",
    "SEMANTIC_UNASSESSED",
    "TERMINAL_FULL_SEAL_SCHEMA",
    "assert_terminal_full_seal_before_publish",
    "build_parser",
    "build_run_receipt",
    "construct_symmetric_latents",
    "object_sha256",
    "run",
    "run_p_qmosaic",
    "validate_cli",
    "validate_terminal_full_seal_receipt",
]

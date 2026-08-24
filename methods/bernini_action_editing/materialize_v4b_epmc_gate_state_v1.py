#!/usr/bin/env python3
"""Materialize one fail-closed v4-B decoded-residual temporal gate.

This is a diagnostic bridge, not a renderer and not an action predictor.  It
is permitted to run only after the frozen v4-B exact-five receipt reports its
aggregate decoded-temporal-codec development gate as true.  For the one
preregistered OOF row, it loads the selected fold-1 codec and computes

    R(x) = C(D(E(C(x)))) - C(D(0)).

The scalar 32-step profile is the feature RMS of ``R`` at each step.  Its only
scale is the p95 of the same quantity over fold-1 model-fit originals.  The
clipped 32-step profile is linearly interpolated to EPMC's 20 nonzero target
phases.  Phase zero and every block/head gate are byte-exact positive zero,
so the downstream intervention is a conservative temporal-only diagnostic at
the existing first-16-block real post-attention-head hook.

No RGB target or anchor path is accepted.  The exact644 frozen feature
authority is used only here to create a sealed gate-state artifact; Bernini
inference later consumes that state plus source video and instruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Optional, Sequence

import torch
from torch.nn import functional as F

from methods.bernini_action_editing import fewshot_privileged_motion_code as epmc
from methods.bernini_action_editing import semantic_anchor_linear_frontier_v4_fast as v4a
from methods.bernini_action_editing import semantic_anchor_temporal_convae_v4b_fast as v4b


SCHEMA = "semantic-anchor-v4b-epmc-temporal-gate-state-v1"
CHECKPOINT_SCHEMA = (
    "semantic-anchor-temporal-convae-selected-fold-checkpoint-v4b-fast"
)
EXPECTED_IID = "7b88a1ca1f804f41"
EXPECTED_OUTER_FOLD = 1
EXPECTED_SOURCE_VIDEO_SHA256 = (
    "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
)
EXPECTED_ANCHOR_VIDEO_SHA256 = (
    "8234f5f35f7001134cf074263c481e3a8079c10f799370090d30e054aef02015"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "105ee8052a0f65d700736a8a25fdf02eb56f1b60d581403c328a8db3d500558c"
)
EXPECTED_V4B_IMPLEMENTATION_SHA256 = (
    "6afa9fc39f993cedcb7ef672ca1297412ab95f5fdacbaf33a431fb49ef586ac4"
)
EXPECTED_FEATURE_RECEIPT_SHA256 = (
    "8ff8f5fd5be36cb67ce40d5558a4406bdf70cbe9b72b0c43c71fa3abe8f6ad9c"
)
PROFILE_SOURCE_STEPS = 32
PROFILE_TARGET_STEPS = 20
P95_QUANTILE = 0.95
ARM_ORDER = ("zero", "correct", "reverse", "shuffle")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_RECEIPT_BYTES = 32 << 20
_MAX_CHECKPOINT_BYTES = 16 << 20


class V4BEPMCGateStateError(RuntimeError):
    """A v4-B authority or derived gate violated the frozen bridge ABI."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise V4BEPMCGateStateError("value is not canonical JSON") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise V4BEPMCGateStateError("digest input must be a non-meta tensor")
    tensor = value.detach().reshape(-1).repeat(1).cpu()
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {"dtype": str(value.dtype), "shape": [int(x) for x in value.shape]}
        )
    )
    digest.update(b"\0")
    digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()


def _required_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise V4BEPMCGateStateError(f"{label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _plain_absolute_file(
    value: str | Path, *, label: str, maximum_bytes: int
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise V4BEPMCGateStateError(f"{label} must be an absolute path")
    try:
        info = path.lstat()
    except OSError as error:
        raise V4BEPMCGateStateError(f"cannot stat {label}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise V4BEPMCGateStateError(f"{label} must be a plain regular file")
    if not 0 < info.st_size <= maximum_bytes:
        raise V4BEPMCGateStateError(f"{label} size is outside its frozen bound")
    if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444:
        raise V4BEPMCGateStateError(f"{label} must be mode0444/nlink1")
    return path.resolve(strict=True)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V4BEPMCGateStateError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise V4BEPMCGateStateError(f"non-finite JSON number: {value}")


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except V4BEPMCGateStateError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4BEPMCGateStateError(f"{label} is not strict ASCII JSON") from error
    if type(value) is not dict:
        raise V4BEPMCGateStateError(f"{label} must contain one JSON object")
    return value


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    digest = _required_sha256(value.get("receipt_digest"), label=f"{label} digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    if _object_sha256(unsigned) != digest:
        raise V4BEPMCGateStateError(f"{label} self-digest differs")
    return digest


def _current_source_binding() -> dict[str, str]:
    paths = {
        "implementation": Path(v4b.__file__).resolve(strict=True),
        "v4a_implementation": Path(v4a.__file__).resolve(strict=True),
        "v2_split_authority": Path(v4b.v2.__file__).resolve(strict=True),
        "feature_authority": Path(v4b.authority.__file__).resolve(strict=True),
        "gate_materializer": Path(__file__).resolve(strict=True),
    }
    return {
        f"{name}_sha256": _file_sha256(path) for name, path in paths.items()
    }


def validate_v4b_receipt_gate(
    receipt: Mapping[str, Any],
    *,
    expected_feature_receipt_sha256: str,
) -> Mapping[str, Any]:
    """Return the fold-1 artifact only after every aggregate gate joins."""

    _verify_self_digest(receipt, label="v4-B receipt")
    metrics = receipt.get("metrics")
    scope = receipt.get("qualification_scope")
    authority = receipt.get("feature_authority")
    closure = receipt.get("oof_closure")
    selected = receipt.get("selected_fold_checkpoint_artifacts")
    implementation = receipt.get("implementation")
    current_binding = _current_source_binding()
    if (
        receipt.get("schema_version") != v4b.SCHEMA
        or receipt.get("status")
        != "V4B_FAST_EXACT5_TEMPORAL_CONVAE_COMPLETE_BURNED_DEVELOPMENT"
        or not isinstance(implementation, Mapping)
        or implementation.get("implementation_sha256")
        != EXPECTED_V4B_IMPLEMENTATION_SHA256
        or implementation.get("implementation_sha256")
        != current_binding["implementation_sha256"]
        or implementation.get("v4a_implementation_sha256")
        != current_binding["v4a_implementation_sha256"]
        or implementation.get("v2_split_authority_sha256")
        != current_binding["v2_split_authority_sha256"]
        or implementation.get("feature_authority_sha256")
        != current_binding["feature_authority_sha256"]
        or not isinstance(metrics, Mapping)
        or metrics.get("decoded_temporal_codec_development_gate") is not True
        or not isinstance(scope, Mapping)
        or scope.get("temporal_codec_development_gate") is not True
        or not isinstance(authority, Mapping)
        or authority.get("feature_receipt_sha256")
        != expected_feature_receipt_sha256
        or not isinstance(closure, Mapping)
        or closure.get("unique_original_iids") != 644
        or closure.get("each_original_evaluated_exactly_once") is not True
        or not isinstance(selected, Mapping)
        or selected.get("count") != 5
        or selected.get("fold_selected_step_join_verified") is not True
        or selected.get("all_create_only_mode0444_nlink1") is not True
    ):
        raise V4BEPMCGateStateError(
            "v4-B aggregate decoded-temporal-codec gate is not a closed true gate"
        )
    evidence = closure.get("embedded_per_iid_evidence")
    if (
        not isinstance(evidence, list)
        or len(evidence) != 644
        or closure.get("embedded_per_iid_evidence_count") != 644
        or closure.get("embedded_per_iid_evidence_sha256") != _object_sha256(evidence)
        or closure.get("evidence_sufficient_to_recompute_all_gates") is not True
    ):
        raise V4BEPMCGateStateError("v4-B OOF evidence is not exact644")
    matches = [row for row in evidence if row.get("iid") == EXPECTED_IID]
    if len(matches) != 1 or int(matches[0].get("outer_fold", -1)) != EXPECTED_OUTER_FOLD:
        raise V4BEPMCGateStateError("preregistered IID is not fold-1 OOF")
    artifacts = selected.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != 5
        or selected.get("artifacts_manifest_sha256") != _object_sha256(artifacts)
        or selected.get("artifacts_reverified_immediately_before_receipt_write")
        is not True
        or selected.get("artifacts_reverified_after_receipt_write_by_command_before_success_return")
        is not True
    ):
        raise V4BEPMCGateStateError("v4-B selected checkpoint manifest differs")
    for artifact in artifacts:
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("mode_octal") != "0444"
            or artifact.get("nlink") != 1
            or artifact.get("selected_training_audit_state_join_verified") is not True
            or artifact.get("fresh_reload_strict_state_verified") is not True
            or artifact.get("fresh_reload_output_bit_exact") is not True
        ):
            raise V4BEPMCGateStateError("v4-B checkpoint artifact audit differs")
    fold1 = [row for row in artifacts if row.get("outer_fold") == EXPECTED_OUTER_FOLD]
    if len(fold1) != 1:
        raise V4BEPMCGateStateError("v4-B fold-1 checkpoint join is not unique")
    return dict(fold1[0])


def _load_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    artifact: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, torch.Tensor]]:
    actual_sha = _file_sha256(path)
    if actual_sha != expected_sha256 or artifact.get("file_sha256") != actual_sha:
        raise V4BEPMCGateStateError("fold-1 checkpoint SHA join differs")
    if artifact.get("size_bytes") != path.stat().st_size:
        raise V4BEPMCGateStateError("fold-1 checkpoint size join differs")
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise V4BEPMCGateStateError("could not load pinned fold-1 checkpoint") from error
    if type(loaded) is not dict or set(loaded) != {"metadata", "state_dict"}:
        raise V4BEPMCGateStateError("fold-1 checkpoint envelope differs")
    metadata, state = loaded["metadata"], loaded["state_dict"]
    if not isinstance(metadata, Mapping) or type(state) is not dict:
        raise V4BEPMCGateStateError("fold-1 checkpoint payload types differ")
    declared_digest = _required_sha256(
        metadata.get("metadata_digest"), label="checkpoint metadata digest"
    )
    unsigned = dict(metadata)
    unsigned.pop("metadata_digest", None)
    if (
        _object_sha256(unsigned) != declared_digest
        or metadata.get("schema_version") != CHECKPOINT_SCHEMA
        or metadata.get("outer_fold") != EXPECTED_OUTER_FOLD
        or metadata.get("artifact_scope")
        != "selected burned-development fold codec checkpoint; not refit or authorized inference"
        or metadata.get("refit_artifact") is not False
        or metadata.get("inference_authorized") is not False
        or metadata.get("model_state_sha256") != v4b._state_sha(state)
        or artifact.get("metadata_digest") != declared_digest
        or artifact.get("model_state_sha256") != metadata.get("model_state_sha256")
        or artifact.get("selected_step") != metadata.get("selected_step")
    ):
        raise V4BEPMCGateStateError("fold-1 checkpoint metadata/state join differs")
    return metadata, state


def _model_from_state(
    metadata: Mapping[str, Any], state: Mapping[str, torch.Tensor]
) -> v4b.TuckerInitializedTemporalConvAE:
    required = {"frame_mean", "temporal_basis", "content_basis", "fit_only_rms"}
    if not required.issubset(state):
        raise V4BEPMCGateStateError("fold-1 state lacks frozen analytic buffers")
    basis = metadata.get("basis")
    if not isinstance(basis, Mapping) or (
        basis.get("frame_mean_sha256") != v4b._tensor_sha(state["frame_mean"])
        or basis.get("temporal_basis_sha256")
        != v4b._tensor_sha(state["temporal_basis"])
        or basis.get("content_basis_first96_sha256")
        != v4b._tensor_sha(state["content_basis"])
        or basis.get("fit_only_global_rms_sha256")
        != v4b._tensor_sha(state["fit_only_rms"])
    ):
        raise V4BEPMCGateStateError("checkpoint analytic-basis hashes differ")
    fitted = v4a.FrontierFit(
        frame_mean=state["frame_mean"].detach().clone(),
        frame_basis=torch.empty(v4b.FEATURE_DIM, 0, dtype=torch.float32),
        clip_mean=torch.empty(1, 0, dtype=torch.float32),
        clip_basis=torch.empty(0, 0, dtype=torch.float32),
        temporal_basis=state["temporal_basis"].detach().clone(),
        content_basis=state["content_basis"].detach().clone(),
        fit_iid_digest=str(metadata.get("model_fit_iid_digest")),
        fit_input_sha256=str(basis.get("fixed_tucker_fit_input_sha256")),
        diagnostics={},
    )
    model = v4b.TuckerInitializedTemporalConvAE(
        fitted, state["fit_only_rms"].detach().clone()
    )
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if v4b._state_sha(model.state_dict()) != metadata.get("model_state_sha256"):
        raise V4BEPMCGateStateError("strictly reloaded model state digest differs")
    return model


def decoded_residual(
    model: v4b.TuckerInitializedTemporalConvAE,
    centered_values: torch.Tensor,
    zero_decode: torch.Tensor,
) -> torch.Tensor:
    """Return ``C(D(E(C(x)))) - C(D(0))`` in original feature coordinates."""

    if (
        centered_values.ndim != 3
        or tuple(centered_values.shape[1:]) != (v4b.TIME_STEPS, v4b.FEATURE_DIM)
        or tuple(zero_decode.shape) != (1, v4b.TIME_STEPS, v4b.FEATURE_DIM)
    ):
        raise V4BEPMCGateStateError("decoded-residual input geometry differs")
    with torch.no_grad():
        decoded = model(centered_values)
    residual = (decoded - zero_decode).contiguous()
    if not bool(torch.isfinite(residual).all()):
        raise V4BEPMCGateStateError("decoded residual contains NaN or infinity")
    return residual


def residual_rms_profile(residual: torch.Tensor) -> torch.Tensor:
    if residual.ndim != 3 or tuple(residual.shape[1:]) != (
        PROFILE_SOURCE_STEPS,
        v4b.FEATURE_DIM,
    ):
        raise V4BEPMCGateStateError("decoded residual must be [N,32,768]")
    profile = residual.square().mean(dim=2).sqrt().contiguous()
    if not bool(torch.isfinite(profile).all()):
        raise V4BEPMCGateStateError("decoded-residual RMS is non-finite")
    return profile


def scaled_profile_32_to_20(
    held_residual: torch.Tensor, fit_residuals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the sole fit-only p95 scale and frozen 32->20 interpolation."""

    held_rms = residual_rms_profile(held_residual)
    fit_rms = residual_rms_profile(fit_residuals)
    if held_rms.shape[0] != 1 or fit_rms.shape[0] < 1:
        raise V4BEPMCGateStateError("held/fit residual population differs")
    p95 = torch.quantile(
        fit_rms.reshape(-1).to(torch.float64),
        P95_QUANTILE,
        interpolation="linear",
    ).to(torch.float32).reshape(1)
    if not bool(torch.isfinite(p95).all()) or float(p95) <= 0.0:
        raise V4BEPMCGateStateError("fit-only decoded-residual p95 is non-positive")
    profile32 = (held_rms / p95).clamp_(0.0, 1.0).to(torch.float32).contiguous()
    profile20 = F.interpolate(
        profile32[:, None, :],
        size=PROFILE_TARGET_STEPS,
        mode="linear",
        align_corners=True,
    )[:, 0].contiguous()
    if (
        tuple(profile32.shape) != (1, PROFILE_SOURCE_STEPS)
        or tuple(profile20.shape) != (1, PROFILE_TARGET_STEPS)
        or bool((profile20 < 0.0).any())
        or bool((profile20 > 1.0).any())
    ):
        raise V4BEPMCGateStateError("scaled temporal gate geometry/range differs")
    return p95, profile32, profile20


def build_motion_codes(profile20: torch.Tensor) -> dict[str, epmc.MotionCode]:
    if profile20.dtype != torch.float32 or tuple(profile20.shape) != (1, 20):
        raise V4BEPMCGateStateError("correct temporal profile must be FP32 [1,20]")
    positive_zero = torch.zeros(1, 1, dtype=torch.float32)
    phase = torch.cat((positive_zero, profile20.detach().cpu()), dim=1).contiguous()
    block_head = torch.zeros(1, 16, 12, dtype=torch.float32)
    correct = epmc.MotionCode(phase, block_head)
    if int(torch.count_nonzero(correct.phase_gates[:, 1:]).item()) == 0:
        raise V4BEPMCGateStateError("correct temporal gate degenerated to all zero")
    codes = {
        "zero": epmc.canonical_noop_motion_code(1, device="cpu"),
        "correct": correct,
        "reverse": epmc.permute_motion_code_phases(
            correct, epmc.REVERSE_PHASE_INDICES
        ),
        "shuffle": epmc.permute_motion_code_phases(
            correct, epmc.SHUFFLE_PHASE_INDICES
        ),
    }
    if tuple(codes) != ARM_ORDER:
        raise V4BEPMCGateStateError("derived arm order differs")
    reference = torch.sort(correct.phase_gates[:, 1:], dim=1).values
    for name in ("reverse", "shuffle"):
        actual = torch.sort(codes[name].phase_gates[:, 1:], dim=1).values
        if not torch.equal(reference, actual):
            raise V4BEPMCGateStateError(f"{name} changed the temporal-gate multiset")
        if torch.equal(codes[name].phase_gates, correct.phase_gates):
            raise V4BEPMCGateStateError(
                f"{name} is byte-identical to correct; causal control degenerated"
            )
    for code in codes.values():
        if int(torch.count_nonzero(code.block_head_gates)) != 0:
            raise V4BEPMCGateStateError("block/head gates must remain exact zero")
    return codes


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise V4BEPMCGateStateError("output must be a fresh absolute JSON child")
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    info = path.stat()
    digest = hashlib.sha256(payload).hexdigest()
    if (
        stat.S_IMODE(info.st_mode) != 0o444
        or info.st_nlink != 1
        or _file_sha256(path) != digest
    ):
        raise V4BEPMCGateStateError("gate-state seal/readback differs")
    return digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4b-receipt", required=True)
    parser.add_argument("--expected-v4b-receipt-sha256", required=True)
    parser.add_argument("--fold1-checkpoint", required=True)
    parser.add_argument("--expected-fold1-checkpoint-sha256", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument(
        "--expected-feature-receipt-sha256",
        default=EXPECTED_FEATURE_RECEIPT_SHA256,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_binding = _current_source_binding()
    for name in (
        "expected_v4b_receipt_sha256",
        "expected_fold1_checkpoint_sha256",
        "expected_feature_receipt_sha256",
    ):
        _required_sha256(getattr(args, name), label=name)
    if args.expected_feature_receipt_sha256 != EXPECTED_FEATURE_RECEIPT_SHA256:
        raise V4BEPMCGateStateError("feature receipt pin differs")
    if type(args.batch_size) is not int or not 1 <= args.batch_size <= 128:
        raise V4BEPMCGateStateError("batch-size must be an integer in [1,128]")
    receipt_path = _plain_absolute_file(
        args.v4b_receipt,
        label="v4-B receipt",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    if _file_sha256(receipt_path) != args.expected_v4b_receipt_sha256:
        raise V4BEPMCGateStateError("v4-B receipt file SHA differs")
    receipt = _strict_json(receipt_path, label="v4-B receipt")
    artifact = validate_v4b_receipt_gate(
        receipt,
        expected_feature_receipt_sha256=args.expected_feature_receipt_sha256,
    )
    checkpoint_path = _plain_absolute_file(
        args.fold1_checkpoint,
        label="fold-1 checkpoint",
        maximum_bytes=_MAX_CHECKPOINT_BYTES,
    )
    metadata, state = _load_checkpoint(
        checkpoint_path,
        expected_sha256=args.expected_fold1_checkpoint_sha256,
        artifact=artifact,
    )
    model = _model_from_state(metadata, state)

    feature_root = Path(args.feature_root).expanduser()
    if not feature_root.is_absolute() or feature_root.is_symlink():
        raise V4BEPMCGateStateError("feature-root must be an absolute real directory")
    pairs, feature_receipt = v4b.authority.load_exact644_pairs(
        feature_root, args.expected_feature_receipt_sha256
    )
    groups, split = v4b.v2._split_fold(pairs, EXPECTED_OUTER_FOLD, v4b.SEED)
    if (
        split.get("iid_digest") != v4a.V2_FOLD_IID_DIGESTS[EXPECTED_OUTER_FOLD]
        or split.get("model_fit_iid_digest") != metadata.get("model_fit_iid_digest")
        or len(groups["exploratory_oof"]) != v4b.FROZEN_OOF_COUNTS[EXPECTED_OUTER_FOLD]
    ):
        raise V4BEPMCGateStateError("fold-1 feature/split/checkpoint join differs")
    held = [row for row in groups["exploratory_oof"] if row.iid == EXPECTED_IID]
    if (
        len(held) != 1
        or held[0].instruction_sha256 != EXPECTED_INSTRUCTION_SHA256
    ):
        raise V4BEPMCGateStateError("preregistered IID is not uniquely fold-1 OOF")

    zero_code = torch.zeros(1, v4b.CODE_TIME, v4b.CODE_CHANNELS, dtype=torch.float32)
    with torch.no_grad():
        zero_decode = model.decode(zero_code).detach().cpu().contiguous()
    if tuple(zero_decode.shape) != (1, v4b.TIME_STEPS, v4b.FEATURE_DIM):
        raise V4BEPMCGateStateError("C(D(0)) geometry differs")

    fit_residual_batches: list[torch.Tensor] = []
    fit_rows = groups["model_fit"]
    for start in range(0, len(fit_rows), args.batch_size):
        values = torch.stack(
            [
                v4a.canonical_action(row.anchor_sequence)
                for row in fit_rows[start : start + args.batch_size]
            ]
        )
        fit_residual_batches.append(decoded_residual(model, values, zero_decode))
    fit_residuals = torch.cat(fit_residual_batches, dim=0).contiguous()
    held_value = v4a.canonical_action(held[0].anchor_sequence)[None]
    held_residual = decoded_residual(model, held_value, zero_decode)
    p95, profile32, profile20 = scaled_profile_32_to_20(
        held_residual, fit_residuals
    )
    codes = build_motion_codes(profile20)

    arm_payload = {
        name: {
            "phase_gates": [float(x) for x in codes[name].phase_gates[0].tolist()],
            "block_head_gates": [
                [float(x) for x in row]
                for row in codes[name].block_head_gates[0].tolist()
            ],
            "phase_gates_sha256": _tensor_sha256(codes[name].phase_gates),
            "block_head_gates_sha256": _tensor_sha256(
                codes[name].block_head_gates
            ),
        }
        for name in ARM_ORDER
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "V4B_EPMC_TEMPORAL_GATE_STATE_COMPLETE_DIAGNOSTIC_ONLY",
        "iid": EXPECTED_IID,
        "outer_fold": EXPECTED_OUTER_FOLD,
        "detached_media_authority": {
            "source_video_sha256": EXPECTED_SOURCE_VIDEO_SHA256,
            "anchor_video_sha256": EXPECTED_ANCHOR_VIDEO_SHA256,
            "instruction_sha256": EXPECTED_INSTRUCTION_SHA256,
            "source_or_anchor_rgb_opened_by_materializer": False,
            "source_rgb_role": "future Bernini model input and HTML reference",
            "anchor_rgb_role": "detached HTML reference only; never Bernini input",
        },
        "v4b_aggregate_gate_verified_true": True,
        "v4b_receipt": {
            "path": str(receipt_path),
            "file_sha256": args.expected_v4b_receipt_sha256,
            "receipt_digest": receipt["receipt_digest"],
            "decoded_temporal_codec_development_gate": True,
        },
        "implementation_binding": source_binding,
        "fold_checkpoint": {
            "path": str(checkpoint_path),
            "file_sha256": args.expected_fold1_checkpoint_sha256,
            "metadata_digest": metadata["metadata_digest"],
            "model_state_sha256": metadata["model_state_sha256"],
            "selected_step": metadata["selected_step"],
            "outer_fold": EXPECTED_OUTER_FOLD,
        },
        "feature_authority": {
            "root": str(feature_root.resolve(strict=True)),
            "receipt_sha256": args.expected_feature_receipt_sha256,
            "receipt_digest": feature_receipt["receipt_digest"],
            "exact644_loaded": True,
        },
        "fit_only_calibration": {
            "outer_fold": EXPECTED_OUTER_FOLD,
            "model_fit_count": len(fit_rows),
            "model_fit_iid_digest": split["model_fit_iid_digest"],
            "oof_iid_excluded": EXPECTED_IID,
            "held_or_oof_values_used_for_scale": False,
            "statistic": "p95 over model-fit IID x 32 of sqrt(mean_d(R**2))",
            "quantile": P95_QUANTILE,
            "p95_value": float(p95.item()),
            "p95_tensor_sha256": _tensor_sha256(p95),
            "fit_decoded_residuals_sha256": _tensor_sha256(fit_residuals),
            "fit_rms_profiles_sha256": _tensor_sha256(
                residual_rms_profile(fit_residuals)
            ),
        },
        "decoded_residual_contract": {
            "definition": "R=C(D(E(C(anchor))))-C(D(0))",
            "zero_code_shape": [1, v4b.CODE_TIME, v4b.CODE_CHANNELS],
            "zero_code_sha256": _tensor_sha256(zero_code),
            "c_d_zero_shape": [1, v4b.TIME_STEPS, v4b.FEATURE_DIM],
            "c_d_zero_sha256": _tensor_sha256(zero_decode),
            "held_residual_shape": [1, v4b.TIME_STEPS, v4b.FEATURE_DIM],
            "held_residual_sha256": _tensor_sha256(held_residual),
            "full_decoded_output_used_as_gate": False,
            "latent_code_used_as_gate": False,
        },
        "temporal_mapping": {
            "profile32": [float(x) for x in profile32[0].tolist()],
            "profile32_sha256": _tensor_sha256(profile32),
            "profile20": [float(x) for x in profile20[0].tolist()],
            "profile20_sha256": _tensor_sha256(profile20),
            "scale": "divide by fit-only p95 then clamp [0,1]",
            "interpolation": "torch linear size=20 align_corners=True",
            "phase0_exact_positive_zero": True,
            "all_16x12_block_head_gates_exact_positive_zero": True,
            "epmc_effective_head_gate_nonzero_phase": "0.5*(profile20+0)=0.5*profile20",
            "downstream_outer_cpmr_gate": 0.10,
            "total_projected_motion_residual_coefficient": "0.10*0.5*profile20=0.05*profile20",
            "total_coefficient_scale": 0.05,
            "source_and_phase0_total_coefficient": 0.0,
        },
        "arms": {
            "order": list(ARM_ORDER),
            "reverse_and_shuffle_preserve_correct_phase_multiset": True,
            "values": arm_payload,
        },
        "scope": {
            "temporal_gating_diagnostic_only": True,
            "action_representation_qualified": False,
            "renderer_qualified": False,
            "video_editing_qualified": False,
            "video_quality_claim": False,
            "heldout_action_anchor_feature_consumed": True,
            "heldout_action_anchor_rgb_consumed": False,
            "target_rgb_consumed": False,
            "source_plus_instruction_only_end_to_end_claim": False,
            "gate_state_is_derived_from_heldout_action_anchor_feature": True,
            "bernini_model_execution_performed": False,
        },
    }
    payload["receipt_digest"] = _object_sha256(payload)
    if _current_source_binding() != source_binding:
        raise V4BEPMCGateStateError("consumer source binding changed during execution")
    output = Path(args.output).expanduser()
    file_sha = _write_json_create_only(output, payload)
    return {
        "gate_state": str(output.resolve(strict=True)),
        "gate_state_sha256": file_sha,
        "receipt_digest": payload["receipt_digest"],
        "v4b_aggregate_gate_verified_true": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(_canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "EXPECTED_IID",
    "EXPECTED_OUTER_FOLD",
    "SCHEMA",
    "V4BEPMCGateStateError",
    "build_motion_codes",
    "build_parser",
    "decoded_residual",
    "main",
    "residual_rms_profile",
    "run",
    "scaled_profile_32_to_20",
    "validate_v4b_receipt_gate",
]

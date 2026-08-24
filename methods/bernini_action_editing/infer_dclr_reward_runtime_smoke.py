#!/usr/bin/env python3
"""Frozen Bernini dual-conditional reward runtime smoke.

This executable asks a deliberately narrow engineering question: does the
*frozen* Bernini-R 1.3B renderer expose usable conditional denoising-energy
ordering for one exact-81-frame proposal?

For every explicitly requested rectified-flow sigma it evaluates four raw,
positive-conditional ``GEN_Wanx22.shared_step`` cells:

* T2V target action versus a T2V hard-negative action; and
* MV2V correct source versus a manually matched wrong source.

All four cells reuse one candidate clean latent and one epsilon realization.
The query coordinate is shared exactly across modes: ``t = 1000 * sigma``.
Bernini's shift-3 (T2V) and shift-5 (MV2V) settings affect *sampling density*,
not this explicit flow coordinate, and are therefore never used to snap the
query to two different grids.  T2V is a direct storage view of the correct
MV2V target tail.  The wrong-source cell changes only the source prefix.

The model call is intentionally below ``renderer.forward``: text embeddings
are obtained with ``renderer.get_t5_text_embeddings`` and the prediction is
obtained directly from ``renderer.diff_dec.shared_step``.  There is no CFG,
APG, unconditional branch, scalar renderer loss, adapter, reward reduction,
or trainable parameter.  Under SP4 the upstream call already returns the full
sequence on every rank; ranks only ``all_gather_object`` identical evidence.

The default proposal input is a content-bound native/student sampler latent
captured before VAE decode, paired with separately captured correct- and
wrong-source video condition latents.  Both source conditions are source-only
pre-decode safetensors artifacts with content-bound provenance.  In particular,
the wrong source is supplied by a source-only VAE materializer receipt rather
than by a native sampler arm, and the runtime never loads a wrong-source parquet
target posterior.  Access to the candidate
parquet target is possible only behind the explicit
``--positive-control-paired-target`` flag and is then marked ineligible for
source-reward calibration or training.  The reported quantities are
denoising-error log-ratio *proxies*, not likelihoods and not evidence that an
editing method works.  This file writes a receipt only; it never updates a
checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dual_conditional_ratio_core as ratio_core  # noqa: E402
import dclr_runtime_contract as runtime_contract  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import motion_residual as motion  # noqa: E402
import train_lora as legacy  # noqa: E402


METHOD_NAME = "bernini-frozen-dclr-runtime-smoke-v1"
RECEIPT_SCHEMA = "bernini-frozen-dclr-runtime-smoke-receipt-v1"
SOURCE_ONLY_VAE_RECEIPT_SCHEMA = "bernini-source-only-vae-materialization-v1"
WRONG_SOURCE_MATCH_SCHEMA = "bernini-dclr-wrong-source-match-v2"
WRONG_SOURCE_MATCH_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_iid",
        "candidate_source_video_sha256",
        "wrong_source_iid",
        "wrong_source_video_sha256",
        "criteria",
        "declared_use",
        "reviewer",
    }
)
NUM_FRAMES = 81
LATENT_PHASES = 21
PATCH_SIZE = (1, 2, 2)
ROPE_COMPLEX_DIM = runtime_contract.PINNED_ROPE_DIM
FLOW_TRAIN_TIMESTEPS = runtime_contract.NUM_TRAIN_TIMESTEPS
T2V_SHIFT = 3.0
MV2V_SHIFT = 5.0
FORWARD_IMPLEMENTATION = (
    "renderer.get_t5_text_embeddings+renderer.diff_dec.shared_step"
)
BRANCH_ORDER = (
    "t2v_target_action",
    "t2v_hard_negative",
    "mv2v_correct_source",
    "mv2v_matched_wrong_source",
)
MATCH_CRITERIA = (
    "distinct_identity",
    "same_actor_class",
    "same_actor_count",
    "same_spatial_bucket",
    "same_camera_class",
    "same_composition_class",
    "same_length",
    "manual_reviewed",
)
RUNTIME_REQUIRED_MATCH_CRITERIA = (
    "distinct_identity",
    "same_actor_class",
    "same_actor_count",
    "same_spatial_bucket",
    "same_length",
    "manual_reviewed",
)
PROPOSAL_ORIGINS = (
    "native_rollout_predecode_latent",
    "paired_target_positive_control",
)


class DCLRRuntimeSmokeError(RuntimeError):
    """Raised before an ambiguous reward query or receipt can be emitted."""


@dataclass(frozen=True)
class FlowQueryPoint:
    """One explicit, mode-shared rectified-flow query coordinate."""

    sigma: float
    timestep: float
    sigma_float32_bits_hex: str
    timestep_float32_bits_hex: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryBundle:
    """Patch-embedded T2V/MV2V queries sharing one target tail."""

    point: FlowQueryPoint
    target_tokens: int
    t2v_noisy_latents: Any
    t2v_rotary_embs: Any
    t2v_target_mask: Any
    mv2v_correct_noisy_latents: Any
    mv2v_correct_rotary_embs: Any
    mv2v_correct_target_mask: Any
    mv2v_wrong_noisy_latents: Any
    mv2v_wrong_rotary_embs: Any
    mv2v_wrong_target_mask: Any
    true_velocity_packed: Any
    noisy_target_spatial: Any
    correct_source_spatial: Any
    wrong_source_spatial: Any
    student_clean_spatial: Any
    epsilon_spatial: Any


@dataclass(frozen=True)
class TextCondition:
    """One cached, official-prefix positive text condition."""

    text_lens: Any
    text_embs: Any
    prompt_sha256: str
    instruction_sha256: str
    task_name: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise DCLRRuntimeSmokeError(f"{label} must be a lowercase SHA-256")
    return text


def _require_sha1(value: Any, *, label: str) -> str:
    text = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise DCLRRuntimeSmokeError(f"{label} must be a full SHA-1")
    return text


def flow_query_point(sigma: Any) -> FlowQueryPoint:
    """Map explicit RF sigma to Bernini/Wan's universal ``1000*sigma`` t.

    Shift-3 and shift-5 are intentionally absent: they change how sigma is
    sampled, not the transformer's flow coordinate once sigma is selected.
    """

    if isinstance(sigma, bool):
        raise DCLRRuntimeSmokeError("sigma must be a finite float in (0, 1)")
    try:
        value = float(sigma)
    except (TypeError, ValueError, OverflowError) as error:
        raise DCLRRuntimeSmokeError(
            "sigma must be a finite float in (0, 1)"
        ) from error
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise DCLRRuntimeSmokeError("sigma must be a finite float in (0, 1)")
    import torch

    # Canonicalize once to one explicit FP32 tensor, then use the shared pinned
    # scheduler contract.  Returning Python floats does not change those stored
    # bits; both GPU tensors below are reconstructed from the same values.
    sigma_tensor = torch.tensor([value], dtype=torch.float32)
    try:
        timestep_tensor = runtime_contract.fp32_sigma_to_timestep(sigma_tensor)
    except runtime_contract.DCLRRuntimeContractError as error:
        raise DCLRRuntimeSmokeError(str(error)) from error
    value = float(sigma_tensor.item())
    timestep = float(timestep_tensor.item())
    if not math.isfinite(timestep) or not 0.0 < timestep < FLOW_TRAIN_TIMESTEPS:
        raise DCLRRuntimeSmokeError("derived flow timestep is invalid")
    return FlowQueryPoint(
        sigma=value,
        timestep=timestep,
        sigma_float32_bits_hex=struct.pack("!f", value).hex(),
        timestep_float32_bits_hex=struct.pack("!f", timestep).hex(),
    )


def validate_sigma_request(
    sigmas: Sequence[Any], sigma_weights: Optional[Sequence[Any]]
) -> tuple[tuple[FlowQueryPoint, ...], tuple[float, ...]]:
    """Validate at least two distinct queries and one shared weight vector."""

    if isinstance(sigmas, (str, bytes)) or len(sigmas) < 2:
        raise DCLRRuntimeSmokeError("at least two sigma values are required")
    points = tuple(flow_query_point(value) for value in sigmas)
    if len({point.sigma_float32_bits_hex for point in points}) != len(points):
        raise DCLRRuntimeSmokeError("sigma values must be distinct")
    if sigma_weights is None:
        weights = tuple(1.0 for _ in points)
    else:
        if isinstance(sigma_weights, (str, bytes)) or len(sigma_weights) != len(points):
            raise DCLRRuntimeSmokeError(
                "sigma_weights must have exact shared shape [S]"
            )
        try:
            weights = tuple(float(value) for value in sigma_weights)
        except (TypeError, ValueError, OverflowError) as error:
            raise DCLRRuntimeSmokeError("sigma weights must be finite") from error
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise DCLRRuntimeSmokeError(
            "sigma weights must be finite, nonnegative, and have positive mass"
        )
    if not sum(weights) > 0.0:
        raise DCLRRuntimeSmokeError(
            "sigma weights must be finite, nonnegative, and have positive mass"
        )
    return points, weights


def validate_wrong_source_match_manifest(
    value: Mapping[str, Any],
    *,
    candidate_iid: str,
    candidate_source_video_sha256: str,
    wrong_source_iid: str,
    wrong_source_video_sha256: str,
) -> dict[str, Any]:
    """Separate a legal runtime donor from a reward-calibration donor.

    A camera/composition-confounded donor is useful for proving that the raw
    MV2V branch runs, but it is not evidence for an identity/source reward.
    The manifest must state which use it requests; calibration fails closed
    unless every criterion is true.
    """

    if not isinstance(value, Mapping):
        raise DCLRRuntimeSmokeError("wrong-source match manifest must be an object")
    if set(value) != WRONG_SOURCE_MATCH_FIELDS:
        raise DCLRRuntimeSmokeError("wrong-source match manifest fields differ")
    if value.get("schema_version") != WRONG_SOURCE_MATCH_SCHEMA:
        raise DCLRRuntimeSmokeError("wrong-source match manifest schema differs")
    if not candidate_iid or not wrong_source_iid or candidate_iid == wrong_source_iid:
        raise DCLRRuntimeSmokeError("candidate and wrong-source IIDs must be distinct")
    if value.get("candidate_iid") != candidate_iid:
        raise DCLRRuntimeSmokeError("match manifest candidate IID differs")
    if value.get("wrong_source_iid") != wrong_source_iid:
        raise DCLRRuntimeSmokeError("match manifest wrong-source IID differs")
    correct_sha = _require_sha256(
        candidate_source_video_sha256, label="candidate source video SHA-256"
    )
    wrong_sha = _require_sha256(
        wrong_source_video_sha256, label="wrong source video SHA-256"
    )
    if value.get("candidate_source_video_sha256") != correct_sha:
        raise DCLRRuntimeSmokeError("match manifest candidate source video SHA differs")
    if value.get("wrong_source_video_sha256") != wrong_sha:
        raise DCLRRuntimeSmokeError("match manifest wrong source video SHA differs")
    if correct_sha == wrong_sha:
        raise DCLRRuntimeSmokeError("correct and wrong source videos must differ")
    criteria = value.get("criteria")
    if not isinstance(criteria, Mapping) or set(criteria) != set(MATCH_CRITERIA):
        raise DCLRRuntimeSmokeError(
            "match manifest must contain the exact required criteria"
        )
    if any(type(criteria.get(name)) is not bool for name in MATCH_CRITERIA):
        raise DCLRRuntimeSmokeError(
            "wrong-source match criteria must all be plain booleans"
        )
    failed_runtime = [
        name
        for name in RUNTIME_REQUIRED_MATCH_CRITERIA
        if criteria.get(name) is not True
    ]
    if failed_runtime:
        raise DCLRRuntimeSmokeError(
            f"wrong-source runtime criteria are not certified: {failed_runtime}"
        )
    declared_use = value.get("declared_use")
    if declared_use not in ("runtime_plumbing_only", "reward_calibration"):
        raise DCLRRuntimeSmokeError(
            "wrong-source match declared_use must be runtime_plumbing_only or reward_calibration"
        )
    scientific_eligibility = all(criteria[name] for name in MATCH_CRITERIA)
    if declared_use == "reward_calibration" and not scientific_eligibility:
        raise DCLRRuntimeSmokeError(
            "reward-calibration donor must satisfy every match criterion"
        )
    reviewer = value.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip() or "\x00" in reviewer:
        raise DCLRRuntimeSmokeError("match manifest requires a non-empty reviewer")
    return {
        "schema_version": WRONG_SOURCE_MATCH_SCHEMA,
        "candidate_iid": candidate_iid,
        "candidate_source_video_sha256": correct_sha,
        "wrong_source_iid": wrong_source_iid,
        "wrong_source_video_sha256": wrong_sha,
        "criteria": {name: bool(criteria[name]) for name in MATCH_CRITERIA},
        "declared_use": declared_use,
        "scientific_eligibility": scientific_eligibility,
        "source_reward_calibration_authorized": (
            declared_use == "reward_calibration" and scientific_eligibility
        ),
        "reviewer": reviewer.strip(),
        "manifest_digest": _object_sha256(dict(value)),
    }


def load_wrong_source_match_manifest(
    path_value: str | Path,
    *,
    expected_sha256: str,
    candidate_iid: str,
    candidate_source_video_sha256: str,
    wrong_source_iid: str,
    wrong_source_video_sha256: str,
) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise DCLRRuntimeSmokeError("wrong-source match manifest must be absolute")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise DCLRRuntimeSmokeError(
            f"wrong-source match manifest is unavailable: {error}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise DCLRRuntimeSmokeError("wrong-source match manifest must be a plain file")
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise DCLRRuntimeSmokeError("wrong-source match manifest SHA-256 differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DCLRRuntimeSmokeError(
            f"cannot decode wrong-source match manifest: {error}"
        ) from error
    result = validate_wrong_source_match_manifest(
        value,
        candidate_iid=candidate_iid,
        candidate_source_video_sha256=candidate_source_video_sha256,
        wrong_source_iid=wrong_source_iid,
        wrong_source_video_sha256=wrong_source_video_sha256,
    )
    result.update({"path": str(path), "file_sha256": actual_sha256})
    return result


def load_normalized_clean_latent_artifact(
    path_value: str | Path,
    *,
    expected_sha256: str,
    expected_role: str,
) -> tuple[Any, dict[str, Any]]:
    """Load one content-bound, pre-decode Bernini latent without MP4 recovery."""

    import torch
    from safetensors import safe_open

    if expected_role not in ("native_sampler_proposal", "source_video_condition"):
        raise DCLRRuntimeSmokeError("normalized latent role is unsupported")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise DCLRRuntimeSmokeError("normalized latent artifact must be absolute")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise DCLRRuntimeSmokeError(
            f"normalized latent artifact is unavailable: {error}"
        ) from error
    if not path.is_file() or path.is_symlink() or path.suffix != ".safetensors":
        raise DCLRRuntimeSmokeError(
            "normalized latent artifact must be a plain safetensors file"
        )
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != _require_sha256(
        expected_sha256, label="normalized latent artifact SHA-256"
    ):
        raise DCLRRuntimeSmokeError("normalized latent artifact SHA-256 differs")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != ["normalized_clean_latent"]:
                raise DCLRRuntimeSmokeError(
                    "normalized latent artifact tensor key differs"
                )
            tensor = opened.get_tensor("normalized_clean_latent").contiguous()
            metadata = dict(opened.metadata() or {})
    except DCLRRuntimeSmokeError:
        raise
    except Exception as error:
        raise DCLRRuntimeSmokeError(
            f"cannot load normalized latent artifact: {error}"
        ) from error
    if metadata != {
        "coordinate": "bernini_normalized_clean_vae_latent",
        "frame_contract": "exact81_latent21",
        "artifact_role": expected_role,
        "source": (
            "native_sampler_before_vae_decode"
            if expected_role == "native_sampler_proposal"
            else "source_video_vae_encode_before_any_decode"
        ),
    }:
        raise DCLRRuntimeSmokeError("normalized latent artifact metadata differs")
    if (
        tensor.dtype != torch.float32
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or tensor.ndim != 5
        or tuple(int(item) for item in tensor.shape[:3]) != (1, 16, LATENT_PHASES)
        or int(tensor.shape[3]) <= 0
        or int(tensor.shape[4]) <= 0
        or int(tensor.shape[3]) % 2
        or int(tensor.shape[4]) % 2
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise DCLRRuntimeSmokeError(
            "normalized latent artifact must contain finite FP32 [1,16,21,H,W]"
        )
    identity = _tensor_identity(tensor, label=expected_role)
    return tensor, {
        "path": str(path),
        "file_sha256": actual_sha256,
        "tensor_key": "normalized_clean_latent",
        "metadata": metadata,
        "tensor_identity": identity,
        "mp4_decode_reencode_used": False,
    }


def _load_native_canary_receipt(
    path_value: str | Path, *, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise DCLRRuntimeSmokeError("native canary receipt must be absolute")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise DCLRRuntimeSmokeError(
            f"native canary receipt is unavailable: {error}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise DCLRRuntimeSmokeError("native canary receipt must be a plain file")
    file_sha256 = _file_sha256(path)
    if file_sha256 != _require_sha256(
        expected_sha256, label="native canary receipt SHA-256"
    ):
        raise DCLRRuntimeSmokeError("native canary receipt file SHA-256 differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DCLRRuntimeSmokeError(
            f"cannot decode native canary receipt: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DCLRRuntimeSmokeError("native canary receipt must be an object")
    declared_digest = value.pop("receipt_digest", None)
    if legacy.object_sha256(value) != _require_sha256(
        declared_digest, label="native canary receipt digest"
    ):
        raise DCLRRuntimeSmokeError("native canary receipt digest differs")
    value["receipt_digest"] = declared_digest
    if (
        value.get("schema_version")
        != "bernini-native-identity-generation-canary-v1"
        or value.get("experimental_canary") is not True
        or value.get("scientific_claim_authorized") is not False
    ):
        raise DCLRRuntimeSmokeError("native canary receipt schema/claim state differs")
    return value, {
        "path": str(path),
        "file_sha256": file_sha256,
        "receipt_digest": declared_digest,
    }


def _load_source_only_vae_receipt(
    path_value: str | Path, *, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a dedicated source-only VAE materialization receipt.

    This schema is deliberately independent of the native-generation canary:
    producing a wrong-source condition does not require running or declaring a
    sampler arm.  The receipt must instead close over the exact source IID and
    media, the VAE checkpoint content, and an explicit no-target access audit.
    """

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise DCLRRuntimeSmokeError(
            "source-only VAE provenance receipt must be absolute"
        )
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise DCLRRuntimeSmokeError(
            f"source-only VAE provenance receipt is unavailable: {error}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise DCLRRuntimeSmokeError(
            "source-only VAE provenance receipt must be a plain file"
        )
    file_sha256 = _file_sha256(path)
    if file_sha256 != _require_sha256(
        expected_sha256, label="source-only VAE provenance receipt SHA-256"
    ):
        raise DCLRRuntimeSmokeError(
            "source-only VAE provenance receipt file SHA-256 differs"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DCLRRuntimeSmokeError(
            f"cannot decode source-only VAE provenance receipt: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DCLRRuntimeSmokeError(
            "source-only VAE provenance receipt must be an object"
        )
    declared_digest = value.pop("receipt_digest", None)
    if legacy.object_sha256(value) != _require_sha256(
        declared_digest, label="source-only VAE provenance receipt digest"
    ):
        raise DCLRRuntimeSmokeError(
            "source-only VAE provenance receipt digest differs"
        )
    value["receipt_digest"] = declared_digest
    access = value.get("access_audit")
    if (
        value.get("schema_version") != SOURCE_ONLY_VAE_RECEIPT_SCHEMA
        or value.get("source_only") is not True
        or value.get("scientific_claim_authorized") is not False
        or not isinstance(access, Mapping)
        or access.get("source_columns_accessed")
        != ["iid", "source_video", "source_video_sha256"]
        or access.get("target_columns_accessed") != []
        or access.get("target_media_accessed") is not False
        or access.get("paired_target_accessed") is not False
    ):
        raise DCLRRuntimeSmokeError(
            "source-only VAE provenance schema/access closure differs"
        )
    return value, {
        "path": str(path),
        "file_sha256": file_sha256,
        "receipt_digest": declared_digest,
        "schema_version": SOURCE_ONLY_VAE_RECEIPT_SCHEMA,
    }


def validate_native_rollout_provenance(
    *,
    candidate_receipt_path: str | Path,
    expected_candidate_receipt_sha256: str,
    source_receipt_path: str | Path,
    expected_source_receipt_sha256: str,
    candidate_arm: str,
    candidate_artifact_path: str | Path,
    candidate_artifact_sha256: str,
    source_artifact_path: str | Path,
    source_artifact_sha256: str,
    expected_source_video_sha256: str,
    expected_action_prompt_sha256: str,
    expected_checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    """Bind proposal and source latent files to two audited native receipts."""

    candidate, candidate_file = _load_native_canary_receipt(
        candidate_receipt_path,
        expected_sha256=expected_candidate_receipt_sha256,
    )
    source, source_file = _load_native_canary_receipt(
        source_receipt_path,
        expected_sha256=expected_source_receipt_sha256,
    )
    if candidate_arm not in ("t2v", "r2v", "rv2v"):
        raise DCLRRuntimeSmokeError("native candidate arm is unsupported")
    expected_source_sha = _require_sha256(
        expected_source_video_sha256, label="expected source video SHA-256"
    )
    expected_prompt_sha = _require_sha256(
        expected_action_prompt_sha256, label="expected action prompt SHA-256"
    )
    for label, receipt in (("candidate", candidate), ("source", source)):
        inputs = receipt.get("input")
        checkpoint = receipt.get("checkpoint")
        content = checkpoint.get("content") if isinstance(checkpoint, Mapping) else None
        if (
            not isinstance(inputs, Mapping)
            or inputs.get("source_video_sha256") != expected_source_sha
            or inputs.get("action_prompt_utf8_sha256") != expected_prompt_sha
            or inputs.get("target_video") is not False
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("tree_sha256") != expected_checkpoint_tree_sha256
            or not isinstance(content, Mapping)
            or content.get("every_file_sha256_verified") is not True
        ):
            raise DCLRRuntimeSmokeError(
                f"{label} native receipt source/prompt/checkpoint binding differs"
            )
    if candidate.get("checkpoint") != source.get("checkpoint"):
        raise DCLRRuntimeSmokeError("candidate/source native checkpoint identities differ")
    if candidate_arm not in candidate.get("arms", []):
        raise DCLRRuntimeSmokeError("candidate arm is absent from its native receipt")
    outputs = candidate.get("outputs")
    output = outputs.get(candidate_arm) if isinstance(outputs, Mapping) else None
    proposal = (
        output.get("normalized_clean_latent")
        if isinstance(output, Mapping)
        else None
    )
    if not isinstance(proposal, Mapping):
        raise DCLRRuntimeSmokeError("native candidate receipt lacks proposal latent")
    candidate_path = Path(candidate_artifact_path).expanduser().resolve(strict=True)
    if (
        proposal.get("sha256") != candidate_artifact_sha256
        or Path(str(proposal.get("path", ""))).resolve(strict=True) != candidate_path
        or proposal.get("artifact_role") != "native_sampler_proposal"
        or proposal.get("native_sampler_before_vae_decode") is not True
        or proposal.get("mp4_decode_reencode_used") is not False
    ):
        raise DCLRRuntimeSmokeError("native candidate latent provenance differs")
    source_proposal = source.get("source_condition_artifact")
    if not isinstance(source_proposal, Mapping):
        raise DCLRRuntimeSmokeError("native source receipt lacks source-condition latent")
    source_path = Path(source_artifact_path).expanduser().resolve(strict=True)
    if (
        source_proposal.get("sha256") != source_artifact_sha256
        or Path(str(source_proposal.get("path", ""))).resolve(strict=True) != source_path
        or source_proposal.get("artifact_role") != "source_video_condition"
        or source_proposal.get("source_video_vae_encode_before_any_decode") is not True
        or source_proposal.get("mp4_decode_reencode_used") is not False
    ):
        raise DCLRRuntimeSmokeError("native source latent provenance differs")
    return {
        "candidate_receipt": candidate_file,
        "source_receipt": source_file,
        "candidate_arm": candidate_arm,
        "source_video_sha256": expected_source_sha,
        "action_prompt_sha256": expected_prompt_sha,
        "checkpoint_tree_sha256": expected_checkpoint_tree_sha256,
        "checkpoint_content_identity": dict(candidate["checkpoint"]["content"]),
        "proposal_latent_sha256": candidate_artifact_sha256,
        "source_condition_latent_sha256": source_artifact_sha256,
        "paired_target_accessed": False,
        "mp4_decode_reencode_used": False,
    }


def validate_source_condition_provenance(
    *,
    source_receipt_path: str | Path,
    expected_source_receipt_sha256: str,
    source_iid: str,
    source_artifact_path: str | Path,
    source_artifact_sha256: str,
    expected_source_video_sha256: str,
    expected_checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    """Bind one source-only condition latent to IID, media, and checkpoint.

    The IID is bound through the caller's message-only parquet row and its
    source-video SHA; a dedicated source-only VAE receipt independently binds
    both values to the pre-decode source artifact and audited checkpoint.  No
    native sampler arm or target posterior field is accepted by this function.
    """

    if not isinstance(source_iid, str) or not source_iid.strip() or "\x00" in source_iid:
        raise DCLRRuntimeSmokeError("source-condition IID must be non-empty text")
    receipt, receipt_file = _load_source_only_vae_receipt(
        source_receipt_path,
        expected_sha256=expected_source_receipt_sha256,
    )
    source_sha = _require_sha256(
        expected_source_video_sha256,
        label="source-condition source video SHA-256",
    )
    tree_sha = _require_sha256(
        expected_checkpoint_tree_sha256,
        label="source-condition checkpoint tree SHA-256",
    )
    inputs = receipt.get("input")
    access = receipt.get("access_audit")
    checkpoint = receipt.get("checkpoint")
    content = checkpoint.get("content") if isinstance(checkpoint, Mapping) else None
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("source_iid") != source_iid.strip()
        or inputs.get("source_video_sha256") != source_sha
        or not isinstance(access, Mapping)
        or access.get("target_columns_accessed") != []
        or access.get("target_media_accessed") is not False
        or access.get("paired_target_accessed") is not False
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("tree_sha256") != tree_sha
        or not isinstance(content, Mapping)
        or content.get("every_file_sha256_verified") is not True
    ):
        raise DCLRRuntimeSmokeError(
            "source-condition receipt media/checkpoint binding differs"
        )
    artifact = receipt.get("source_condition_artifact")
    if not isinstance(artifact, Mapping):
        raise DCLRRuntimeSmokeError(
            "source-condition receipt lacks a source-only latent artifact"
        )
    artifact_path = Path(source_artifact_path).expanduser().resolve(strict=True)
    artifact_sha = _require_sha256(
        source_artifact_sha256,
        label="source-condition latent SHA-256",
    )
    try:
        receipt_artifact_path = Path(
            str(artifact.get("path", ""))
        ).resolve(strict=True)
    except OSError as error:
        raise DCLRRuntimeSmokeError(
            f"source-condition receipt artifact is unavailable: {error}"
        ) from error
    artifact_shape = artifact.get("shape")
    if (
        artifact.get("sha256") != artifact_sha
        or receipt_artifact_path != artifact_path
        or artifact.get("artifact_role") != "source_video_condition"
        or artifact.get("tensor_key") != "normalized_clean_latent"
        or artifact.get("coordinate") != "bernini_normalized_clean_vae_latent"
        or artifact.get("frame_contract") != "exact81_latent21"
        or artifact.get("stored_dtype") != "torch.float32"
        or not isinstance(artifact_shape, list)
        or artifact_shape[:3] != [1, 16, LATENT_PHASES]
        or artifact.get("source_video_vae_encode_before_any_decode") is not True
        or artifact.get("mp4_decode_reencode_used") is not False
    ):
        raise DCLRRuntimeSmokeError(
            "source-condition latent provenance differs"
        )
    return {
        "source_receipt": receipt_file,
        "source_iid": source_iid.strip(),
        "source_video_sha256": source_sha,
        "checkpoint_tree_sha256": tree_sha,
        "checkpoint_content_identity": dict(content),
        "source_condition_latent_sha256": artifact_sha,
        "source_columns_accessed": list(access["source_columns_accessed"]),
        "target_columns_accessed": [],
        "target_media_accessed": False,
        "paired_target_accessed": False,
        "mp4_decode_reencode_used": False,
    }


def _require_floating_tensor(value: Any, *, label: str, ndim: int) -> Any:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
        or value.numel() <= 0
        or not bool(torch.isfinite(value).all().item())
    ):
        raise DCLRRuntimeSmokeError(
            f"{label} must be one finite floating {ndim}D tensor"
        )
    return value


def _require_rotary_tensor(value: Any, *, label: str) -> Any:
    """Require Bernini/Wan's real runtime complex rotary representation."""

    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 4
        or tuple(int(item) for item in value.shape[:2]) != (1, 1)
        or int(value.shape[-1]) != ROPE_COMPLEX_DIM
        or value.dtype != torch.complex128
        or value.requires_grad
        or value.grad_fn is not None
        or value.numel() <= 0
        or not bool(torch.isfinite(value).all().item())
    ):
        raise DCLRRuntimeSmokeError(
            f"{label} must be pinned complex128 [1,1,N,{ROPE_COMPLEX_DIM}]"
        )
    return value


def pack_spatial_velocity(value: Any) -> Any:
    """Pack ``[B,C,T,H,W]`` in official ``(pt ph pw c)`` output order."""

    tensor = _require_floating_tensor(value, label="spatial velocity", ndim=5)
    batch, channels, phases, height, width = (int(x) for x in tensor.shape)
    pt, ph, pw = PATCH_SIZE
    if phases % pt or height % ph or width % pw:
        raise DCLRRuntimeSmokeError("spatial velocity is not patch divisible")
    return (
        tensor.reshape(
            batch,
            channels,
            phases // pt,
            pt,
            height // ph,
            ph,
            width // pw,
            pw,
        )
        .permute(0, 2, 4, 6, 3, 5, 7, 1)
        .reshape(
            batch,
            (phases // pt) * (height // ph) * (width // pw),
            pt * ph * pw * channels,
        )
    )


def _same_tensor(left: Any, right: Any) -> bool:
    import torch

    return bool(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and tuple(left.shape) == tuple(right.shape)
        and left.dtype == right.dtype
        and left.device == right.device
        and left.layout == right.layout
        and torch.equal(left, right)
    )


def _storage_ptr(value: Any) -> int:
    storage_getter = getattr(value, "untyped_storage", None)
    storage = storage_getter() if storage_getter is not None else value.storage()
    return int(storage.data_ptr())


def build_same_state_query_bundle(
    transformer: Any,
    *,
    correct_source_spatial: Any,
    wrong_source_spatial: Any,
    student_clean_spatial: Any,
    epsilon_spatial: Any,
    point: FlowQueryPoint,
) -> QueryBundle:
    """Patch one shared target and assemble T2V/correct/wrong-source queries."""

    import torch

    tensors = {
        "correct source": _require_floating_tensor(
            correct_source_spatial, label="correct source", ndim=5
        ),
        "wrong source": _require_floating_tensor(
            wrong_source_spatial, label="wrong source", ndim=5
        ),
        "student clean": _require_floating_tensor(
            student_clean_spatial, label="student clean", ndim=5
        ),
        "epsilon": _require_floating_tensor(
            epsilon_spatial, label="epsilon", ndim=5
        ),
    }
    shapes = {tuple(value.shape) for value in tensors.values()}
    devices = {value.device for value in tensors.values()}
    if len(shapes) != 1 or len(devices) != 1:
        raise DCLRRuntimeSmokeError(
            "correct/wrong/student/noise tensors must share exact geometry and device"
        )
    shape = tuple(student_clean_spatial.shape)
    if (
        int(shape[0]) != 1
        or int(shape[1]) != 16
        or int(shape[2]) != LATENT_PHASES
        or int(shape[3]) <= 0
        or int(shape[4]) <= 0
        or int(shape[3]) % 2
        or int(shape[4]) % 2
    ):
        raise DCLRRuntimeSmokeError(
            "same-state inputs must be normalized exact81 [1,16,21,H,W] with even H/W"
        )
    if torch.equal(correct_source_spatial, wrong_source_spatial):
        raise DCLRRuntimeSmokeError("matched wrong source is tensor-identical to correct source")

    transformer_dtype = getattr(transformer, "dtype", None)
    if transformer_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise DCLRRuntimeSmokeError("active transformer exposes no supported floating dtype")
    sigma = torch.tensor(
        point.sigma,
        device=student_clean_spatial.device,
        dtype=torch.float32,
    )
    student_fp32 = student_clean_spatial.float()
    epsilon_fp32 = epsilon_spatial.float()
    noisy_target_fp32 = (1.0 - sigma) * student_fp32 + sigma * epsilon_fp32
    # Only the model input follows the transformer's compute dtype.  The RF
    # supervision remains the exact FP32 epsilon-y target used by training;
    # dividing BF16 input-rounding error by sigma would corrupt low-sigma
    # reward cells.
    noisy_target = noisy_target_fp32.to(dtype=transformer_dtype)
    true_velocity = epsilon_fp32 - student_fp32
    true_velocity_packed = pack_spatial_velocity(true_velocity)

    correct_tokens, correct_rope = transformer.patch_vae_latent(
        correct_source_spatial.to(dtype=transformer_dtype), source_id=1
    )
    wrong_tokens, wrong_rope = transformer.patch_vae_latent(
        wrong_source_spatial.to(dtype=transformer_dtype), source_id=1
    )
    target_tokens, target_rope = transformer.patch_vae_latent(
        noisy_target, source_id=0
    )
    for label, value in (
        ("correct source tokens", correct_tokens),
        ("wrong source tokens", wrong_tokens),
        ("target tokens", target_tokens),
    ):
        _require_floating_tensor(value, label=label, ndim=3)
    if not (
        tuple(correct_tokens.shape) == tuple(wrong_tokens.shape) == tuple(target_tokens.shape)
        and int(target_tokens.shape[0]) == 1
    ):
        raise DCLRRuntimeSmokeError("patched source/target token geometry differs")
    target_count = int(target_tokens.shape[1])
    if target_count <= 0 or target_count % LATENT_PHASES:
        raise DCLRRuntimeSmokeError("patched target does not expose exact81 phase geometry")
    if tuple(true_velocity_packed.shape[:2]) != (1, target_count):
        raise DCLRRuntimeSmokeError("packed velocity token geometry differs from target")
    for label, rope in (
        ("correct source rope", correct_rope),
        ("wrong source rope", wrong_rope),
        ("target rope", target_rope),
    ):
        _require_rotary_tensor(rope, label=label)
    if not (
        int(correct_rope.shape[2])
        == int(wrong_rope.shape[2])
        == int(target_rope.shape[2])
        == target_count
    ):
        raise DCLRRuntimeSmokeError("rotary token geometry differs")
    if not _same_tensor(correct_rope, wrong_rope):
        raise DCLRRuntimeSmokeError(
            "correct/wrong source-prefix rotary must be exactly identical"
        )

    mv_correct = torch.cat((correct_tokens, target_tokens), dim=1)
    mv_correct_rope = torch.cat((correct_rope, target_rope), dim=2)
    mv_wrong = torch.cat((wrong_tokens, target_tokens), dim=1)
    mv_wrong_rope = torch.cat((wrong_rope, target_rope), dim=2)
    # The generator branch is literally the correct editor branch's target
    # tail, including storage identity, offset, and stride.
    t2v = mv_correct[:, target_count:, :]
    t2v_rope = mv_correct_rope[:, :, target_count:, :]
    t2v_mask = torch.ones(target_count, dtype=torch.bool, device=t2v.device)
    mv2v_mask = torch.cat(
        (
            torch.zeros(target_count, dtype=torch.bool, device=t2v.device),
            t2v_mask,
        ),
        dim=0,
    )
    bundle = QueryBundle(
        point=point,
        target_tokens=target_count,
        t2v_noisy_latents=t2v,
        t2v_rotary_embs=t2v_rope,
        t2v_target_mask=t2v_mask,
        mv2v_correct_noisy_latents=mv_correct,
        mv2v_correct_rotary_embs=mv_correct_rope,
        mv2v_correct_target_mask=mv2v_mask,
        mv2v_wrong_noisy_latents=mv_wrong,
        mv2v_wrong_rotary_embs=mv_wrong_rope,
        mv2v_wrong_target_mask=mv2v_mask.clone(),
        true_velocity_packed=true_velocity_packed,
        noisy_target_spatial=noisy_target,
        correct_source_spatial=correct_source_spatial,
        wrong_source_spatial=wrong_source_spatial,
        student_clean_spatial=student_clean_spatial,
        epsilon_spatial=epsilon_spatial,
    )
    validate_query_bundle(bundle)
    return bundle


def validate_query_bundle(bundle: QueryBundle) -> dict[str, Any]:
    """Prove T2V=N, MV2V=2N, direct tail, and wrong-prefix-only delta."""

    import torch

    if not isinstance(bundle, QueryBundle):
        raise DCLRRuntimeSmokeError("query bundle type differs")
    count = bundle.target_tokens
    if type(count) is not int or count <= 0 or count % LATENT_PHASES:
        raise DCLRRuntimeSmokeError("query target token count is invalid")
    t2v = bundle.t2v_noisy_latents
    correct = bundle.mv2v_correct_noisy_latents
    wrong = bundle.mv2v_wrong_noisy_latents
    if tuple(t2v.shape[:2]) != (1, count):
        raise DCLRRuntimeSmokeError("T2V query must contain exactly target N")
    if tuple(correct.shape[:2]) != (1, 2 * count) or tuple(wrong.shape) != tuple(correct.shape):
        raise DCLRRuntimeSmokeError("MV2V queries must contain source N + target N")
    expected_tail = correct[:, count:, :]
    if not _same_tensor(t2v, expected_tail):
        raise DCLRRuntimeSmokeError("T2V query differs from correct MV2V target tail")
    if (
        _storage_ptr(t2v) != _storage_ptr(correct)
        or int(t2v.storage_offset()) != int(expected_tail.storage_offset())
        or tuple(t2v.stride()) != tuple(expected_tail.stride())
    ):
        raise DCLRRuntimeSmokeError("T2V query is not a direct MV2V target-tail view")
    if not _same_tensor(wrong[:, count:, :], expected_tail):
        raise DCLRRuntimeSmokeError("wrong-source query changed the shared target tail")
    if torch.equal(wrong[:, :count, :], correct[:, :count, :]):
        raise DCLRRuntimeSmokeError("wrong-source query did not change the source prefix")

    t2v_mask = bundle.t2v_target_mask
    correct_mask = bundle.mv2v_correct_target_mask
    wrong_mask = bundle.mv2v_wrong_target_mask
    if (
        not isinstance(t2v_mask, torch.Tensor)
        or t2v_mask.dtype != torch.bool
        or tuple(t2v_mask.shape) != (count,)
        or not bool(t2v_mask.all().item())
    ):
        raise DCLRRuntimeSmokeError("T2V target mask must be all-True [N]")
    expected_mv2v_mask = torch.cat(
        (
            torch.zeros(count, dtype=torch.bool, device=t2v_mask.device),
            t2v_mask,
        ),
        dim=0,
    )
    if not _same_tensor(correct_mask, expected_mv2v_mask) or not _same_tensor(
        wrong_mask, expected_mv2v_mask
    ):
        raise DCLRRuntimeSmokeError(
            "MV2V masks must select one contiguous target tail after source N"
        )

    t2v_rope = bundle.t2v_rotary_embs
    correct_rope = bundle.mv2v_correct_rotary_embs
    wrong_rope = bundle.mv2v_wrong_rotary_embs
    if int(t2v_rope.shape[2]) != count or int(correct_rope.shape[2]) != 2 * count:
        raise DCLRRuntimeSmokeError("T2V/MV2V rotary geometry differs")
    expected_rope_tail = correct_rope[:, :, count:, :]
    if not _same_tensor(t2v_rope, expected_rope_tail):
        raise DCLRRuntimeSmokeError("T2V rotary differs from correct MV2V target tail")
    if (
        _storage_ptr(t2v_rope) != _storage_ptr(correct_rope)
        or int(t2v_rope.storage_offset()) != int(expected_rope_tail.storage_offset())
    ):
        raise DCLRRuntimeSmokeError("T2V rotary is not a direct target-tail view")
    if not _same_tensor(wrong_rope[:, :, count:, :], expected_rope_tail):
        raise DCLRRuntimeSmokeError("wrong-source query changed target rotary")
    if not _same_tensor(wrong_rope, correct_rope):
        raise DCLRRuntimeSmokeError(
            "wrong-source query changed full source-prefix/target rotary"
        )
    if tuple(bundle.true_velocity_packed.shape[:2]) != (1, count):
        raise DCLRRuntimeSmokeError("velocity target must contain exactly target N")
    sigma_tensor = torch.tensor(
        bundle.point.sigma,
        device=bundle.student_clean_spatial.device,
        dtype=torch.float32,
    )
    expected_noisy = (
        (1.0 - sigma_tensor) * bundle.student_clean_spatial.float()
        + sigma_tensor * bundle.epsilon_spatial.float()
    ).to(dtype=bundle.noisy_target_spatial.dtype)
    if not _same_tensor(bundle.noisy_target_spatial, expected_noisy):
        raise DCLRRuntimeSmokeError("noisy target is not shared FP32 RF state cast once")
    expected_velocity = pack_spatial_velocity(
        bundle.epsilon_spatial.float() - bundle.student_clean_spatial.float()
    )
    if not _same_tensor(bundle.true_velocity_packed, expected_velocity):
        raise DCLRRuntimeSmokeError("velocity target must equal exact FP32 epsilon minus clean")
    return {
        "verified": True,
        "target_tokens": count,
        "t2v_total_tokens": count,
        "mv2v_total_tokens": 2 * count,
        "t2v_is_correct_mv2v_target_tail_view": True,
        "wrong_source_changes_prefix_only": True,
        "correct_wrong_full_rotary_exact_equal": True,
        "target_source_id": 0,
        "source_source_id": 1,
        "rotary_dtype": str(t2v_rope.dtype),
        "target_tail_mask": "T2V all N; MV2V final N of 2N",
    }


def _active_transformer(renderer: Any) -> tuple[str, Any]:
    decoder = getattr(renderer, "diff_dec", None)
    if decoder is None:
        raise DCLRRuntimeSmokeError("renderer.diff_dec is unavailable")
    first = getattr(decoder, "transformer", None)
    second = getattr(decoder, "transformer_2", None)
    if (first is None) == (second is None):
        raise DCLRRuntimeSmokeError("runtime requires exactly one active Wan expert")
    return ("transformer_1", first) if first is not None else ("transformer_2", second)


def shared_step_target_prediction(
    renderer: Any,
    *,
    model_id: str,
    noisy_latents: Any,
    rotary_embs: Any,
    target_tokens: int,
    target_mask: Any,
    timestep: Any,
    condition: TextCondition,
) -> Any:
    """Call the real shared_step and select target tokens, with no guidance."""

    import torch

    latents = _require_floating_tensor(noisy_latents, label="noisy tokens", ndim=3)
    rope = _require_rotary_tensor(rotary_embs, label="rotary embeddings")
    if int(latents.shape[0]) != 1 or int(rope.shape[2]) != int(latents.shape[1]):
        raise DCLRRuntimeSmokeError("shared_step latent/rotary geometry differs")
    total = int(latents.shape[1])
    if type(target_tokens) is not int or target_tokens <= 0 or target_tokens > total:
        raise DCLRRuntimeSmokeError("shared_step target-tail length is invalid")
    if (
        not isinstance(target_mask, torch.Tensor)
        or target_mask.dtype != torch.bool
        or target_mask.device != latents.device
        or tuple(target_mask.shape) != (total,)
        or int(target_mask.sum().item()) != target_tokens
        or bool(target_mask[: total - target_tokens].any().item())
        or not bool(target_mask[total - target_tokens :].all().item())
    ):
        raise DCLRRuntimeSmokeError(
            "shared_step target mask must select one contiguous target tail"
        )
    if (
        not isinstance(timestep, torch.Tensor)
        or tuple(timestep.shape) != (1,)
        or timestep.dtype != torch.float32
        or timestep.device != latents.device
        or not bool(torch.isfinite(timestep).all().item())
    ):
        raise DCLRRuntimeSmokeError("shared timestep must be finite GPU FP32 [1]")
    prediction = renderer.diff_dec.shared_step(
        model_id=model_id,
        noisy_latents=latents,
        timesteps=timestep,
        cond_embeds=condition.text_embs,
        rotary_embs=rope,
        batch_vae_seqlen=[total],
        batch_text_seqlen=condition.text_lens,
    )
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.ndim != 3
        or tuple(prediction.shape[:2]) != (1, total)
        or not prediction.is_floating_point()
        or not bool(torch.isfinite(prediction).all().item())
    ):
        raise DCLRRuntimeSmokeError(
            "shared_step must return one finite full-sequence prediction"
        )
    # SP/Ulysses shared_step already all-gathers the complete sequence.
    return prediction[:, target_mask, :]


def assemble_sp4_receipt(
    local_evidence: Mapping[str, Any], rank_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Fail closed unless four ranks independently report identical evidence."""

    if not isinstance(local_evidence, Mapping):
        raise DCLRRuntimeSmokeError("local evidence must be an object")
    if local_evidence.get("forward_implementation") != FORWARD_IMPLEMENTATION:
        raise DCLRRuntimeSmokeError("receipt did not use direct shared_step")
    if local_evidence.get("branch_order") != list(BRANCH_ORDER):
        raise DCLRRuntimeSmokeError("receipt branch order differs")
    if local_evidence.get("num_frames") != NUM_FRAMES or local_evidence.get(
        "latent_phases"
    ) != LATENT_PHASES:
        raise DCLRRuntimeSmokeError("receipt is not exact81")
    sigma_records = local_evidence.get("sigma_records")
    if not isinstance(sigma_records, list) or len(sigma_records) < 2:
        raise DCLRRuntimeSmokeError("receipt requires at least two sigma records")
    if local_evidence.get("forwards_per_rank") != 4 * len(sigma_records):
        raise DCLRRuntimeSmokeError("receipt forward count differs")
    if local_evidence.get("adapter_state") != "absent_frozen_base":
        raise DCLRRuntimeSmokeError("receipt must certify adapter-free frozen base")
    geometry = local_evidence.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("verified") is not True:
        raise DCLRRuntimeSmokeError("receipt target-tail geometry is unverified")
    if local_evidence.get("reward_reduction") != "none":
        raise DCLRRuntimeSmokeError("reward reduction is forbidden under SP4")
    candidate = local_evidence.get("candidate")
    if not isinstance(candidate, Mapping):
        raise DCLRRuntimeSmokeError("receipt candidate evidence is absent")
    if candidate.get("message_template_columns_loaded") != [
        "iid",
        "inputs",
        "source_video_sha256",
    ]:
        raise DCLRRuntimeSmokeError(
            "receipt message template loaded privileged parquet columns"
        )
    proposal_origin = candidate.get("proposal_origin")
    if proposal_origin not in PROPOSAL_ORIGINS:
        raise DCLRRuntimeSmokeError("receipt proposal origin differs")
    paired = proposal_origin == "paired_target_positive_control"
    if (
        candidate.get("paired_target_accessed") is not paired
        or candidate.get("positive_control_only") is not paired
    ):
        raise DCLRRuntimeSmokeError("paired-target positive-control flags differ")
    if paired:
        if candidate.get("proposal_artifact") is not None:
            raise DCLRRuntimeSmokeError(
                "paired-target positive-control cannot claim a rollout artifact"
            )
    else:
        if not isinstance(candidate.get("proposal_artifact"), Mapping) or not isinstance(
            local_evidence.get("correct_source_artifact"), Mapping
        ) or not isinstance(candidate.get("native_provenance"), Mapping):
            raise DCLRRuntimeSmokeError(
                "native rollout receipt lacks candidate/source artifacts or provenance"
            )
    wrong_source = local_evidence.get("wrong_source")
    match_manifest = (
        wrong_source.get("match_manifest")
        if isinstance(wrong_source, Mapping)
        else None
    )
    if (
        not isinstance(wrong_source, Mapping)
        or wrong_source.get("paired_target_accessed") is not False
        or not isinstance(wrong_source.get("source_artifact"), Mapping)
        or not isinstance(wrong_source.get("source_provenance"), Mapping)
        or wrong_source["source_provenance"].get("target_columns_accessed")
        != []
        or wrong_source["source_provenance"].get("target_media_accessed")
        is not False
        or wrong_source["source_provenance"].get("paired_target_accessed")
        is not False
        or not isinstance(match_manifest, Mapping)
        or type(match_manifest.get("source_reward_calibration_authorized"))
        is not bool
    ):
        raise DCLRRuntimeSmokeError("wrong-source eligibility evidence is absent")
    source_reward_calibration_authorized = bool(
        match_manifest["source_reward_calibration_authorized"] and not paired
    )

    if len(rank_records) != 4:
        raise DCLRRuntimeSmokeError("SP4 receipt requires exactly four rank records")
    ranks = []
    digests = []
    for record in rank_records:
        if not isinstance(record, Mapping):
            raise DCLRRuntimeSmokeError("rank record must be an object")
        if record.get("world_size") != 4 or record.get("ulysses_size") != 4:
            raise DCLRRuntimeSmokeError("rank record is not world4/Ulysses4")
        rank = record.get("rank")
        if type(rank) is not int:
            raise DCLRRuntimeSmokeError("rank must be an integer")
        ranks.append(rank)
        digests.append(
            _require_sha256(record.get("local_evidence_digest"), label="rank digest")
        )
    if sorted(ranks) != [0, 1, 2, 3] or len(set(digests)) != 1:
        raise DCLRRuntimeSmokeError("SP4 ranks did not report identical evidence")
    local_digest = _object_sha256(dict(local_evidence))
    if digests[0] != local_digest:
        raise DCLRRuntimeSmokeError("rank evidence digest differs from receipt payload")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "local_evidence": dict(local_evidence),
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "upstream_shared_step_returns_full_sequence_per_rank": True,
            "reward_reduction": "none",
            "rank_records": [dict(value) for value in sorted(rank_records, key=lambda x: x["rank"])],
            "all_gather_evidence_only": True,
        },
        "engineering_smoke_only": True,
        "proposal_origin": proposal_origin,
        "paired_target_positive_control": paired,
        "wrong_source_paired_target_accessed": False,
        "source_reward_calibration_authorized": source_reward_calibration_authorized,
        "training_pair_authorized": False,
        "denoising_error_proxy_not_likelihood": True,
        "scientific_claim_authorized": False,
        "production_claim_forbidden": True,
    }
    receipt["receipt_digest"] = _object_sha256(receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen exact81 Bernini DCLR reward smoke on SP4"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--candidate-row-index", required=True, type=int)
    parser.add_argument("--wrong-source-row-index", required=True, type=int)
    parser.add_argument("--expected-candidate-iid", required=True)
    parser.add_argument("--proposal-source-iid", required=True)
    parser.add_argument("--expected-wrong-source-iid", required=True)
    parser.add_argument(
        "--wrong-source-clean-latent",
        required=True,
        help="source-only wrong-source FP32 pre-decode safetensors condition",
    )
    parser.add_argument(
        "--expected-wrong-source-clean-latent-sha256", required=True
    )
    parser.add_argument("--wrong-source-provenance-receipt", required=True)
    parser.add_argument(
        "--expected-wrong-source-provenance-receipt-sha256", required=True
    )
    parser.add_argument("--expected-wrong-source-video-sha256", required=True)
    parser.add_argument("--wrong-source-match-json", required=True)
    parser.add_argument("--expected-wrong-source-match-sha256", required=True)
    parser.add_argument("--hard-negative-instruction", required=True)
    parser.add_argument("--expected-hard-negative-instruction-sha256", required=True)
    parser.add_argument("--action-instruction", required=True)
    parser.add_argument("--expected-action-instruction-sha256", required=True)
    proposal = parser.add_mutually_exclusive_group(required=True)
    proposal.add_argument(
        "--candidate-clean-latent",
        help="native/student sampler FP32 pre-decode safetensors proposal",
    )
    proposal.add_argument(
        "--positive-control-paired-target",
        action="store_true",
        help="privileged paired target; API/geometry positive-control only",
    )
    parser.add_argument("--expected-candidate-clean-latent-sha256")
    parser.add_argument("--correct-source-clean-latent")
    parser.add_argument("--expected-correct-source-clean-latent-sha256")
    parser.add_argument("--candidate-arm", choices=("t2v", "r2v", "rv2v"))
    parser.add_argument("--candidate-provenance-receipt")
    parser.add_argument("--expected-candidate-provenance-receipt-sha256")
    parser.add_argument("--source-provenance-receipt")
    parser.add_argument("--expected-source-provenance-receipt-sha256")
    parser.add_argument("--expected-proposal-source-video-sha256")
    parser.add_argument("--sigmas", nargs="+", type=float, default=(0.80, 0.35))
    parser.add_argument("--sigma-weights", nargs="+", type=float)
    parser.add_argument("--noise-seed", type=int, default=20260808)
    parser.add_argument("--output-receipt", required=True)
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    parser.add_argument("--num-frames", type=int, choices=(81,), default=81)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> tuple[tuple[FlowQueryPoint, ...], tuple[float, ...]]:
    if args.num_frames != NUM_FRAMES:
        raise DCLRRuntimeSmokeError("runtime smoke requires exact81")
    points, weights = validate_sigma_request(args.sigmas, args.sigma_weights)
    if type(args.noise_seed) is not int or not 0 <= args.noise_seed < 2**31:
        raise DCLRRuntimeSmokeError("noise_seed must lie in [0,2^31)")
    if (
        type(args.candidate_row_index) is not int
        or type(args.wrong_source_row_index) is not int
        or args.candidate_row_index < 0
        or args.wrong_source_row_index < 0
        or args.candidate_row_index == args.wrong_source_row_index
    ):
        raise DCLRRuntimeSmokeError("candidate and wrong-source row indices must be distinct nonnegative integers")
    for name in (
        "expected_candidate_iid",
        "proposal_source_iid",
        "expected_wrong_source_iid",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise DCLRRuntimeSmokeError(f"{name} must be non-empty text")
    if args.proposal_source_iid == args.expected_wrong_source_iid:
        raise DCLRRuntimeSmokeError("proposal source and wrong-source IIDs must differ")
    instruction = args.hard_negative_instruction
    action_instruction = args.action_instruction
    if not isinstance(instruction, str) or not instruction.strip() or "\x00" in instruction:
        raise DCLRRuntimeSmokeError("hard-negative instruction must be non-empty text")
    if (
        not isinstance(action_instruction, str)
        or not action_instruction.strip()
        or "\x00" in action_instruction
    ):
        raise DCLRRuntimeSmokeError("action instruction must be non-empty text")
    if instruction.strip() == action_instruction.strip():
        raise DCLRRuntimeSmokeError("action and hard-negative instructions must differ")
    hard_sha = hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest()
    if hard_sha != _require_sha256(
        args.expected_hard_negative_instruction_sha256,
        label="expected_hard_negative_instruction_sha256",
    ):
        raise DCLRRuntimeSmokeError("hard-negative instruction SHA-256 differs")
    action_sha = hashlib.sha256(
        action_instruction.strip().encode("utf-8")
    ).hexdigest()
    if action_sha != _require_sha256(
        args.expected_action_instruction_sha256,
        label="expected_action_instruction_sha256",
    ):
        raise DCLRRuntimeSmokeError("action instruction SHA-256 differs")
    _require_sha256(
        args.expected_wrong_source_match_sha256,
        label="expected_wrong_source_match_sha256",
    )
    _require_sha256(
        args.expected_checkpoint_tree_sha256,
        label="expected_checkpoint_tree_sha256",
    )
    _require_sha256(
        args.method_source_archive_sha256,
        label="method_source_archive_sha256",
    )
    _require_sha1(args.expected_bernini_commit, label="expected_bernini_commit")
    _require_sha1(args.expected_veomni_commit, label="expected_veomni_commit")
    _require_sha1(args.method_source_revision, label="method_source_revision")
    if args.expected_bernini_commit.lower() != legacy.BERNINI_OFFICIAL_COMMIT:
        raise DCLRRuntimeSmokeError("Bernini revision differs from pinned release")
    if args.expected_veomni_commit.lower() != legacy.VEOMNI_TESTED_COMMIT:
        raise DCLRRuntimeSmokeError("VeOmni revision differs from pinned release")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise DCLRRuntimeSmokeError("checkpoint tree identity differs")
    manifest = Path(args.checkpoint_content_manifest).expanduser()
    if not manifest.is_absolute():
        raise DCLRRuntimeSmokeError("checkpoint-content-manifest must be absolute")
    for label, path_value in (
        ("wrong-source-clean-latent", args.wrong_source_clean_latent),
        (
            "wrong-source-provenance-receipt",
            args.wrong_source_provenance_receipt,
        ),
    ):
        if not isinstance(path_value, str) or not Path(
            path_value
        ).expanduser().is_absolute():
            raise DCLRRuntimeSmokeError(f"{label} must be absolute")
    for label, value in (
        (
            "expected_wrong_source_clean_latent_sha256",
            args.expected_wrong_source_clean_latent_sha256,
        ),
        (
            "expected_wrong_source_provenance_receipt_sha256",
            args.expected_wrong_source_provenance_receipt_sha256,
        ),
        (
            "expected_wrong_source_video_sha256",
            args.expected_wrong_source_video_sha256,
        ),
    ):
        _require_sha256(value, label=label)
    paired_control = bool(args.positive_control_paired_target)
    candidate_path = args.candidate_clean_latent
    external_values = (
        candidate_path,
        args.expected_candidate_clean_latent_sha256,
        args.correct_source_clean_latent,
        args.expected_correct_source_clean_latent_sha256,
        args.candidate_arm,
        args.candidate_provenance_receipt,
        args.expected_candidate_provenance_receipt_sha256,
        args.source_provenance_receipt,
        args.expected_source_provenance_receipt_sha256,
        args.expected_proposal_source_video_sha256,
    )
    if paired_control:
        if any(value is not None for value in external_values):
            raise DCLRRuntimeSmokeError(
                "paired positive-control cannot accept external clean latents"
            )
    else:
        if any(value is None for value in external_values):
            raise DCLRRuntimeSmokeError(
                "native rollout mode requires candidate and correct-source latent artifacts"
            )
        for label, path_value in (
            ("candidate-clean-latent", candidate_path),
            ("correct-source-clean-latent", args.correct_source_clean_latent),
            ("candidate-provenance-receipt", args.candidate_provenance_receipt),
            ("source-provenance-receipt", args.source_provenance_receipt),
        ):
            if not Path(str(path_value)).expanduser().is_absolute():
                raise DCLRRuntimeSmokeError(f"{label} must be absolute")
        _require_sha256(
            args.expected_candidate_clean_latent_sha256,
            label="expected_candidate_clean_latent_sha256",
        )
        _require_sha256(
            args.expected_correct_source_clean_latent_sha256,
            label="expected_correct_source_clean_latent_sha256",
        )
        for label, value in (
            (
                "expected_candidate_provenance_receipt_sha256",
                args.expected_candidate_provenance_receipt_sha256,
            ),
            (
                "expected_source_provenance_receipt_sha256",
                args.expected_source_provenance_receipt_sha256,
            ),
            (
                "expected_proposal_source_video_sha256",
                args.expected_proposal_source_video_sha256,
            ),
        ):
            _require_sha256(value, label=label)
    if paired_control and args.proposal_source_iid != args.expected_candidate_iid:
        raise DCLRRuntimeSmokeError(
            "paired positive-control proposal source must equal its candidate row IID"
        )
    return points, weights


def _row_iid(row: Mapping[str, Any], *, label: str) -> str:
    value = row.get("iid")
    if not isinstance(value, str) or not value or "\x00" in value:
        raise DCLRRuntimeSmokeError(f"{label} row lacks a valid IID")
    return value


def _load_message_only_row(dataset: Any, index: int) -> dict[str, Any]:
    """Read only IID/text columns; native proposals must not load paired targets."""

    import bisect
    import pyarrow.parquet as pq

    length = len(dataset)
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise DCLRRuntimeSmokeError("message-only parquet row index is out of range")
    groups = getattr(dataset, "_groups", None)
    ends = getattr(dataset, "_ends", None)
    if not isinstance(groups, list) or not isinstance(ends, list):
        raise DCLRRuntimeSmokeError("parquet row-store geometry is unavailable")
    group_index = bisect.bisect_right(ends, index)
    start, _, path, row_group = groups[group_index]
    try:
        rows = pq.ParquetFile(path).read_row_group(
            row_group, columns=["iid", "inputs", "source_video_sha256"]
        ).to_pylist()
    except Exception as error:
        raise DCLRRuntimeSmokeError(
            f"cannot read message-only parquet columns: {error}"
        ) from error
    row = rows[index - start]
    if set(row) != {"iid", "inputs", "source_video_sha256"}:
        raise DCLRRuntimeSmokeError(
            "message-only parquet read exposed unexpected/privileged columns"
        )
    _row_iid(row, label="message-only")
    _instruction(row)
    _require_sha256(
        row["source_video_sha256"], label="message-only source video SHA-256"
    )
    return dict(row)


def _instruction(sample: Mapping[str, Any]) -> str:
    try:
        messages = json.loads(str(sample["inputs"]))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise DCLRRuntimeSmokeError("cannot decode renderer instruction") from error
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or not isinstance(messages[1], Mapping)
        or messages[1].get("type") != "text"
    ):
        raise DCLRRuntimeSmokeError("renderer row lacks one edit instruction")
    text = messages[1].get("text")
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise DCLRRuntimeSmokeError("renderer instruction is invalid")
    return text.strip()


def _normalized_mode_spatial(blob: Any, vae_mean: Any, vae_std: Any) -> Any:
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    distribution = DiagonalGaussianDistribution(legacy._load_tensor_blob(blob))
    clean = distribution.mode()
    if (
        not isinstance(clean, torch.Tensor)
        or tuple(clean.shape[:3]) != (1, 16, LATENT_PHASES)
        or clean.ndim != 5
    ):
        raise DCLRRuntimeSmokeError("posterior mode is not exact81 [1,16,21,H,W]")
    normalized = (clean.squeeze(0) - vae_mean) / vae_std
    normalized = normalized.unsqueeze(0).contiguous().float()
    _require_floating_tensor(normalized, label="normalized posterior mode", ndim=5)
    return normalized


def _tokenize_positive_condition(
    *,
    renderer: Any,
    tokenizer: Any,
    encode_renderer_messages: Any,
    sample: Mapping[str, Any],
    task_name: str,
    device: Any,
) -> TextCondition:
    import torch

    if task_name not in ("t2v", "mv2v"):
        raise DCLRRuntimeSmokeError("text task must be t2v or mv2v")
    instruction = _instruction(sample)
    messages = json.loads(str(sample["inputs"]))
    encoded = encode_renderer_messages(
        messages,
        tokenizer,
        task_name=task_name,
        drop_text=False,
        drop_video=False,
        drop_img=False,
    )
    ids = encoded.get("input_ids")
    attention = encoded.get("attention_mask")
    lens = encoded.get("t5_input_lens")
    if (
        not isinstance(ids, torch.Tensor)
        or not isinstance(attention, torch.Tensor)
        or not isinstance(lens, torch.Tensor)
        or ids.ndim != 1
        or tuple(attention.shape) != tuple(ids.shape)
        or lens.numel() != 1
        or int(lens.item()) != int(ids.numel())
        or not 0 < int(ids.numel()) <= 512
    ):
        raise DCLRRuntimeSmokeError("official positive tokenization geometry differs")
    ids = ids.unsqueeze(0).to(device)
    attention = attention.unsqueeze(0).to(device)
    lens = lens.reshape(1, 1).to(device)
    text_lens, text_embs = renderer.get_t5_text_embeddings(ids, attention, lens)
    # The renderer consumes the actual lens above, then returns one padded
    # max-sequence length for Wan's packed cross-attention metadata.
    max_text_length = int(getattr(renderer, "max_sequence_length", 0))
    if max_text_length != 512 or text_lens != [max_text_length]:
        raise DCLRRuntimeSmokeError(
            "runtime T5 packed length differs from pinned renderer max=512"
        )
    prompt_identity = {
        "task_name": task_name,
        "input_ids": ids.detach().cpu().tolist(),
        "attention_mask": attention.detach().cpu().tolist(),
    }
    return TextCondition(
        text_lens=text_lens,
        text_embs=text_embs,
        prompt_sha256=_object_sha256(prompt_identity),
        instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        task_name=task_name,
    )


def _tensor_identity(value: Any, *, label: str) -> dict[str, Any]:
    import torch

    tensor = _require_floating_tensor(value, label=label, ndim=value.ndim)
    # Clone to an offset-zero, exactly sized CPU storage, then hash its raw
    # bytes directly.  Do not make receipt integrity depend on NumPy being
    # importable in the ROCm/runtime environment.
    cpu = tensor.detach().to(device="cpu").contiguous().clone()
    raw = bytes(cpu.untyped_storage())
    if len(raw) != int(cpu.numel() * cpu.element_size()):
        raise DCLRRuntimeSmokeError(f"{label} storage byte count differs")
    metadata = {
        "label": label,
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "byte_count": len(raw),
    }
    payload = _canonical_json_bytes(metadata) + b"\0" + raw
    metadata["content_sha256"] = hashlib.sha256(payload).hexdigest()
    metadata["raw_storage_sha256"] = hashlib.sha256(raw).hexdigest()
    metadata["finite"] = True
    return metadata


def _diagnostics_to_dict(value: Any) -> dict[str, Any]:
    def numbers(tensor: Any) -> Any:
        data = tensor.detach().double().cpu()
        return data.tolist()

    return {
        "proxy": numbers(value.proxy),
        "preferred_error": numbers(value.preferred_error),
        "contrast_error": numbers(value.contrast_error),
        "preferred_error_by_sigma": numbers(value.preferred_error_by_sigma),
        "contrast_error_by_sigma": numbers(value.contrast_error_by_sigma),
        "per_sigma_proxy": numbers(value.per_sigma_proxy),
        "normalized_sigma_weights": numbers(value.normalized_sigma_weights),
        "sign_convention": "positive means preferred condition has lower denoising error",
    }


def _validate_output_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise DCLRRuntimeSmokeError("output receipt path must be absolute")
    parent = path.parent.resolve(strict=True)
    result = parent / path.name
    if result.exists() or result.is_symlink():
        raise DCLRRuntimeSmokeError("output receipt already exists")
    if not parent.is_dir() or parent.is_symlink():
        raise DCLRRuntimeSmokeError("output receipt parent must be a plain directory")
    return result


def _write_receipt_atomically(path: Path, value: Mapping[str, Any]) -> None:
    token = f"pid-{os.getpid()}"
    temporary = path.with_name(f".{path.name}.tmp-{token}")
    if temporary.exists() or temporary.is_symlink():
        raise DCLRRuntimeSmokeError("stale temporary receipt exists")
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(_canonical_json_bytes(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    points, weights = validate_cli(args)
    output_receipt = _validate_output_path(args.output_receipt)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise DCLRRuntimeSmokeError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % 4:
        raise DCLRRuntimeSmokeError("1.3B attention heads must divide Ulysses=4")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.training.data import encode_renderer_messages

    distributed = legacy.distributed_contract()
    if distributed.world_size != 4 or distributed.ulysses_size != 4:
        raise DCLRRuntimeSmokeError("runtime smoke requires exact world4/Ulysses4")
    device, backend = legacy.initialise_distributed(distributed)
    from bernini.parallel import init_parallel_state

    init_parallel_state(ulysses_size=4)
    legacy.seed_same_sample(args.noise_seed)

    manifest_path = Path(args.checkpoint_content_manifest).expanduser()
    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, manifest_path
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_result = checkpoint_rows[0]
    if (
        not isinstance(checkpoint_result, Mapping)
        or checkpoint_result.get("ok") is not True
    ):
        raise DCLRRuntimeSmokeError(
            f"rank-zero checkpoint content validation failed: {checkpoint_result}"
        )
    checkpoint_identity = dict(checkpoint_result["identity"])

    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = legacy.validate_preprocessed_dataset_summary(
        args.dataset_summary,
        dataset,
        allow_incomplete=bool(args.allow_incomplete_dataset),
    )
    if args.candidate_row_index >= len(dataset) or args.wrong_source_row_index >= len(dataset):
        raise DCLRRuntimeSmokeError("candidate or wrong-source row index is out of range")
    candidate_message_row = _load_message_only_row(
        dataset, args.candidate_row_index
    )
    wrong_message_row = _load_message_only_row(
        dataset, args.wrong_source_row_index
    )
    candidate_iid = _row_iid(candidate_message_row, label="candidate")
    wrong_iid = _row_iid(wrong_message_row, label="wrong source")
    if candidate_iid != args.expected_candidate_iid or wrong_iid != args.expected_wrong_source_iid:
        raise DCLRRuntimeSmokeError("runtime row IID differs from CLI binding")
    proposal_source_video_sha256 = (
        str(candidate_message_row["source_video_sha256"])
        if args.positive_control_paired_target
        else args.expected_proposal_source_video_sha256
    )
    wrong_source_video_sha256 = str(
        wrong_message_row["source_video_sha256"]
    )
    if wrong_source_video_sha256 != args.expected_wrong_source_video_sha256:
        raise DCLRRuntimeSmokeError(
            "wrong-source message-only row video SHA-256 differs"
        )
    match_manifest = load_wrong_source_match_manifest(
        args.wrong_source_match_json,
        expected_sha256=args.expected_wrong_source_match_sha256,
        candidate_iid=args.proposal_source_iid,
        candidate_source_video_sha256=proposal_source_video_sha256,
        wrong_source_iid=wrong_iid,
        wrong_source_video_sha256=wrong_source_video_sha256,
    )
    candidate: Mapping[str, Any] = candidate_message_row
    paired_candidate: Optional[Mapping[str, Any]] = None
    if args.positive_control_paired_target:
        try:
            paired_candidate = legacy.sanitize_preprocessed_row(
                dataset[args.candidate_row_index]
            )
            legacy.validate_81_frame_latents(paired_candidate)
        except legacy.TrainingContractError as error:
            raise DCLRRuntimeSmokeError(str(error)) from error
    candidate = motion.replace_edit_instruction(
        candidate, args.action_instruction.strip()
    )
    action_instruction = _instruction(candidate)
    action_instruction_sha = hashlib.sha256(action_instruction.encode("utf-8")).hexdigest()
    if action_instruction_sha != args.expected_action_instruction_sha256:
        raise DCLRRuntimeSmokeError("candidate action instruction SHA-256 differs")
    hard_negative_sample = motion.replace_edit_instruction(
        candidate, args.hard_negative_instruction.strip()
    )
    if _instruction(hard_negative_sample) == action_instruction:
        raise DCLRRuntimeSmokeError("hard-negative instruction equals target action")

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except legacy.TrainingContractError as error:
        raise DCLRRuntimeSmokeError(str(error)) from error
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False)
    renderer.eval()
    renderer.to(device)
    if any(parameter.requires_grad for parameter in renderer.parameters()):
        raise DCLRRuntimeSmokeError("frozen renderer retains trainable parameters")
    if any("lora" in name.lower() for name, _ in renderer.named_modules()):
        raise DCLRRuntimeSmokeError("runtime renderer unexpectedly contains LoRA modules")
    model_id, transformer = _active_transformer(renderer)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    vae_mean, vae_std, _ = legacy._vae_statistics(checkpoint)
    proposal_artifact: Optional[dict[str, Any]] = None
    source_artifact: Optional[dict[str, Any]] = None
    native_provenance: Optional[dict[str, Any]] = None
    if args.positive_control_paired_target:
        proposal_origin = "paired_target_positive_control"
        assert paired_candidate is not None
        correct_source_cpu = _normalized_mode_spatial(
            paired_candidate["video_vae_latents"][0], vae_mean, vae_std
        )
        student_clean_cpu = _normalized_mode_spatial(
            paired_candidate["video_vae_latents"][1], vae_mean, vae_std
        )
    else:
        proposal_origin = "native_rollout_predecode_latent"
        student_clean_cpu, proposal_artifact = load_normalized_clean_latent_artifact(
            args.candidate_clean_latent,
            expected_sha256=args.expected_candidate_clean_latent_sha256,
            expected_role="native_sampler_proposal",
        )
        correct_source_cpu, source_artifact = load_normalized_clean_latent_artifact(
            args.correct_source_clean_latent,
            expected_sha256=args.expected_correct_source_clean_latent_sha256,
            expected_role="source_video_condition",
        )
        native_provenance = validate_native_rollout_provenance(
            candidate_receipt_path=args.candidate_provenance_receipt,
            expected_candidate_receipt_sha256=(
                args.expected_candidate_provenance_receipt_sha256
            ),
            source_receipt_path=args.source_provenance_receipt,
            expected_source_receipt_sha256=(
                args.expected_source_provenance_receipt_sha256
            ),
            candidate_arm=args.candidate_arm,
            candidate_artifact_path=args.candidate_clean_latent,
            candidate_artifact_sha256=args.expected_candidate_clean_latent_sha256,
            source_artifact_path=args.correct_source_clean_latent,
            source_artifact_sha256=(
                args.expected_correct_source_clean_latent_sha256
            ),
            expected_source_video_sha256=(
                args.expected_proposal_source_video_sha256
            ),
            expected_action_prompt_sha256=action_instruction_sha,
            expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        )
        if native_provenance["checkpoint_content_identity"] != checkpoint_identity:
            raise DCLRRuntimeSmokeError(
                "native rollout and active checkpoint content identities differ"
            )
    wrong_source_cpu, wrong_source_artifact = load_normalized_clean_latent_artifact(
        args.wrong_source_clean_latent,
        expected_sha256=args.expected_wrong_source_clean_latent_sha256,
        expected_role="source_video_condition",
    )
    wrong_source_provenance = validate_source_condition_provenance(
        source_receipt_path=args.wrong_source_provenance_receipt,
        expected_source_receipt_sha256=(
            args.expected_wrong_source_provenance_receipt_sha256
        ),
        source_iid=wrong_iid,
        source_artifact_path=args.wrong_source_clean_latent,
        source_artifact_sha256=args.expected_wrong_source_clean_latent_sha256,
        expected_source_video_sha256=wrong_source_video_sha256,
        expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
    )
    if wrong_source_provenance["checkpoint_content_identity"] != checkpoint_identity:
        raise DCLRRuntimeSmokeError(
            "wrong-source artifact and active checkpoint content identities differ"
        )
    if not (
        tuple(student_clean_cpu.shape)
        == tuple(correct_source_cpu.shape)
        == tuple(wrong_source_cpu.shape)
    ):
        raise DCLRRuntimeSmokeError(
            "proposal/correct-source/wrong-source latent geometry differs"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.noise_seed)
    epsilon_cpu = torch.randn(
        tuple(student_clean_cpu.shape), generator=generator, dtype=torch.float32
    )
    correct_source = correct_source_cpu.to(device)
    wrong_source = wrong_source_cpu.to(device)
    student_clean = student_clean_cpu.to(device)
    epsilon = epsilon_cpu.to(device)

    with torch.inference_mode():
        t2v_action_condition = _tokenize_positive_condition(
            renderer=renderer,
            tokenizer=tokenizer,
            encode_renderer_messages=encode_renderer_messages,
            sample=candidate,
            task_name="t2v",
            device=device,
        )
        t2v_hard_condition = _tokenize_positive_condition(
            renderer=renderer,
            tokenizer=tokenizer,
            encode_renderer_messages=encode_renderer_messages,
            sample=hard_negative_sample,
            task_name="t2v",
            device=device,
        )
        mv2v_action_condition = _tokenize_positive_condition(
            renderer=renderer,
            tokenizer=tokenizer,
            encode_renderer_messages=encode_renderer_messages,
            sample=candidate,
            task_name="mv2v",
            device=device,
        )
        if t2v_action_condition.prompt_sha256 == t2v_hard_condition.prompt_sha256:
            raise DCLRRuntimeSmokeError("T2V target and hard-negative tokenization are identical")

        action_errors = []
        hard_errors = []
        correct_errors = []
        wrong_errors = []
        sigma_records = []
        geometry: Optional[dict[str, Any]] = None
        for point in points:
            bundle = build_same_state_query_bundle(
                transformer,
                correct_source_spatial=correct_source,
                wrong_source_spatial=wrong_source,
                student_clean_spatial=student_clean,
                epsilon_spatial=epsilon,
                point=point,
            )
            current_geometry = validate_query_bundle(bundle)
            if geometry is None:
                geometry = current_geometry
            elif geometry != current_geometry:
                raise DCLRRuntimeSmokeError("query geometry changed across sigma")
            shared_timestep = torch.tensor(
                [point.timestep], device=device, dtype=torch.float32
            )
            predictions = (
                shared_step_target_prediction(
                    renderer,
                    model_id=model_id,
                    noisy_latents=bundle.t2v_noisy_latents,
                    rotary_embs=bundle.t2v_rotary_embs,
                    target_tokens=bundle.target_tokens,
                    target_mask=bundle.t2v_target_mask,
                    timestep=shared_timestep,
                    condition=t2v_action_condition,
                ),
                shared_step_target_prediction(
                    renderer,
                    model_id=model_id,
                    noisy_latents=bundle.t2v_noisy_latents,
                    rotary_embs=bundle.t2v_rotary_embs,
                    target_tokens=bundle.target_tokens,
                    target_mask=bundle.t2v_target_mask,
                    timestep=shared_timestep,
                    condition=t2v_hard_condition,
                ),
                shared_step_target_prediction(
                    renderer,
                    model_id=model_id,
                    noisy_latents=bundle.mv2v_correct_noisy_latents,
                    rotary_embs=bundle.mv2v_correct_rotary_embs,
                    target_tokens=bundle.target_tokens,
                    target_mask=bundle.mv2v_correct_target_mask,
                    timestep=shared_timestep,
                    condition=mv2v_action_condition,
                ),
                shared_step_target_prediction(
                    renderer,
                    model_id=model_id,
                    noisy_latents=bundle.mv2v_wrong_noisy_latents,
                    rotary_embs=bundle.mv2v_wrong_rotary_embs,
                    target_tokens=bundle.target_tokens,
                    target_mask=bundle.mv2v_wrong_target_mask,
                    timestep=shared_timestep,
                    condition=mv2v_action_condition,
                ),
            )
            target = bundle.true_velocity_packed
            target_mask = torch.ones(
                (1, bundle.target_tokens, 1), dtype=torch.bool, device=device
            )
            energies = tuple(
                ratio_core.masked_per_sample_mse(prediction, target, target_mask)
                for prediction in predictions
            )
            action_errors.append(energies[0])
            hard_errors.append(energies[1])
            correct_errors.append(energies[2])
            wrong_errors.append(energies[3])
            sigma_records.append(
                {
                    "flow_query": point.as_dict(),
                    "mode_shared_sigma_and_timestep": True,
                    "t2v_shift_sampling_density_only": T2V_SHIFT,
                    "mv2v_shift_sampling_density_only": MV2V_SHIFT,
                    "target_query": _tensor_identity(
                        bundle.noisy_target_spatial, label=f"target_query_sigma_{point.sigma_float32_bits_hex}"
                    ),
                    "errors": {
                        name: float(value.detach().double().cpu().item())
                        for name, value in zip(BRANCH_ORDER, energies)
                    },
                    "raw_positive_conditional_only": True,
                    "cfg": False,
                    "apg": False,
                }
            )

        weight_tensor = torch.tensor(weights, dtype=torch.float64, device=device)
        action_diagnostics = ratio_core.multi_sigma_denoising_error_ratio_proxy(
            torch.stack(action_errors, dim=0),
            torch.stack(hard_errors, dim=0),
            weight_tensor,
        )
        source_diagnostics = ratio_core.multi_sigma_denoising_error_ratio_proxy(
            torch.stack(correct_errors, dim=0),
            torch.stack(wrong_errors, dim=0),
            weight_tensor,
        )
    assert geometry is not None
    local_evidence = {
        "method_source_revision": args.method_source_revision.lower(),
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_content_identity": checkpoint_identity,
        "dataset_signature": dataset.signature,
        "dataset_summary_sha256": dataset_summary["sha256"],
        "candidate": {
            "row_index": args.candidate_row_index,
            "message_template_iid": candidate_iid,
            "message_template_columns_loaded": [
                "iid",
                "inputs",
                "source_video_sha256",
            ],
            "proposal_source_iid": args.proposal_source_iid,
            "action_instruction_sha256": action_instruction_sha,
            "proposal_origin": proposal_origin,
            "proposal_artifact": proposal_artifact,
            "native_provenance": native_provenance,
            "paired_target_accessed": bool(args.positive_control_paired_target),
            "positive_control_only": bool(args.positive_control_paired_target),
        },
        "wrong_source": {
            "row_index": args.wrong_source_row_index,
            "iid": wrong_iid,
            "source_video_sha256": wrong_source_video_sha256,
            "message_template_columns_loaded": [
                "iid",
                "inputs",
                "source_video_sha256",
            ],
            "source_artifact": wrong_source_artifact,
            "source_provenance": wrong_source_provenance,
            "paired_target_accessed": False,
            "match_manifest": match_manifest,
        },
        "text_conditions": {
            "t2v_action_prompt_sha256": t2v_action_condition.prompt_sha256,
            "t2v_hard_negative_prompt_sha256": t2v_hard_condition.prompt_sha256,
            "mv2v_action_prompt_sha256": mv2v_action_condition.prompt_sha256,
            "hard_negative_instruction_sha256": t2v_hard_condition.instruction_sha256,
        },
        "num_frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "patch_size": list(PATCH_SIZE),
        "noise_seed": args.noise_seed,
        "shared_student_clean": _tensor_identity(student_clean_cpu, label="student_clean"),
        "shared_epsilon": _tensor_identity(epsilon_cpu, label="epsilon"),
        "correct_source": _tensor_identity(correct_source_cpu, label="correct_source"),
        "correct_source_artifact": source_artifact,
        "wrong_source_tensor": _tensor_identity(wrong_source_cpu, label="wrong_source"),
        "geometry": geometry,
        "sigma_weights": list(weights),
        "sigma_records": sigma_records,
        "action_target_vs_hard_negative": _diagnostics_to_dict(action_diagnostics),
        "source_correct_vs_matched_wrong": _diagnostics_to_dict(source_diagnostics),
        "forward_implementation": FORWARD_IMPLEMENTATION,
        "active_model_id": model_id,
        "branch_order": list(BRANCH_ORDER),
        "forwards_per_rank": 4 * len(points),
        "adapter_state": "absent_frozen_base",
        "model_mode": "eval_inference_mode",
        "raw_positive_conditional_only": True,
        "cfg": False,
        "apg": False,
        "renderer_forward_scalar_loss": False,
        "flow_coordinate": "explicit shared t=1000*sigma; shift affects sampling density only",
        "reward_reduction": "none",
        "backend": backend,
    }
    local_digest = _object_sha256(local_evidence)
    local_rank_record = {
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "ulysses_size": distributed.ulysses_size,
        "local_evidence_digest": local_digest,
    }
    rank_records: list[Optional[dict[str, Any]]] = [None] * 4
    # No all_reduce: shared_step already reconstructed the full SP sequence.
    dist.all_gather_object(rank_records, local_rank_record)
    if any(value is None for value in rank_records):
        raise DCLRRuntimeSmokeError("SP4 evidence gather returned an empty rank")
    receipt = assemble_sp4_receipt(local_evidence, rank_records)  # type: ignore[arg-type]
    dist.barrier()
    if distributed.rank == 0:
        _write_receipt_atomically(output_receipt, receipt)
        print(_canonical_json_bytes({
            "receipt": str(output_receipt),
            "receipt_digest": receipt["receipt_digest"],
            "action_proxy": receipt["local_evidence"]["action_target_vs_hard_negative"]["proxy"],
            "source_proxy": receipt["local_evidence"]["source_correct_vs_matched_wrong"]["proxy"],
        }).decode("ascii"))
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline latent dataset contract for full-target action editing."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from pact.dataset import (
    UMT5_EMBEDDING_DIM,
    VLM_EMBEDDING_DIM,
    WAN21_VAE_CHANNEL_MEAN,
    encoder_contract_sha256,
    validate_encoder_contract,
)

from .config import ACTION_TASK_TYPES, MOTION_TOKEN_DIM, DataConfig


ACTION_PAYLOAD_FORMAT = "omnivideo2-action-latents-v1"
ACTION_MANIFEST_FORMAT = "omnivideo2-action-manifest-v1"
ACTION_PROVENANCE_FORMAT = "omnivideo2-action-materialization-provenance-v1"
ACTION_TRAINING_RELEASE_FORMAT = "omnivideo2-action-training-release-binding-v1"
ACTION_TRAINING_RELEASE_VERIFICATION_FORMAT = (
    "omnivideo2-action-training-release-verification-v1"
)
ACTION_PAYLOAD_FIELDS = frozenset(
    {
        "format",
        "sample_id",
        "encoder_contract",
        "source_latent",
        "target_latent",
        "text_context",
        "source_vlm_context",
        "target_motion_tokens",
        "task_type",
        "preview_only",
    }
)
ACTION_MANIFEST_ROW_FIELDS = frozenset(
    {
        "format",
        "sample_id",
        "payload_path",
        "payload_sha256",
        "provenance_path",
        "provenance_sha256",
        "task_type",
        "preview_only",
    }
)
ACTION_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "parent_id",
        "split_group",
        "direction",
        "task_type",
        "preview_only",
        "training_authorized",
        "training_use_forbidden",
        "production_eligible",
        "post_video_acceptance",
        "preview_join",
        "media",
        "conditioning",
        "encoder",
        "tensor_sha256",
        "payload",
    }
)
ACTION_PROVENANCE_PRODUCTION_FIELDS = ACTION_PROVENANCE_FIELDS | frozenset(
    {"training_release"}
)
ACTION_TRAINING_RELEASE_FIELDS = frozenset(
    {
        "schema_version",
        "release_path",
        "release_sha256",
        "verification_receipt_path",
        "verification_receipt_sha256",
        "sample_row_sha256",
        "verification_status",
    }
)
ACTION_TRAINING_RELEASE_VERIFICATION_FIELDS = frozenset(
    {"schema_version", "status", "sample_id", "release_sha256", "release_row"}
)
ACTION_TRAINING_RELEASE_ROW_FIELDS = frozenset(
    {
        "sample_id",
        "training_authorized",
        "training_use_forbidden",
        "production_eligible",
        "post_video_acceptance",
    }
)
ACTION_PROVENANCE_PREVIEW_JOIN_FIELDS = frozenset(
    {
        "manifest_path",
        "manifest_sha256",
        "row_digest",
        "row_file_sha256",
        "upstream_provenance_sha256",
    }
)
ACTION_PROVENANCE_MEDIA_FIELDS = frozenset(
    {
        "source_video_path",
        "source_video_sha256",
        "target_video_path",
        "target_video_sha256",
        "shared_i0_path",
        "shared_i0_sha256",
        "preprocessing",
    }
)
ACTION_PROVENANCE_CONDITIONING_FIELDS = frozenset(
    {
        "instruction",
        "instruction_sha256",
        "instruction_source",
        "generation_instruction_sha256",
        "target_caption",
        "target_caption_sha256",
        "target_caption_origin",
        "motion_text",
        "motion_text_sha256",
        "motion_teacher_visual_input",
        "motion_teacher_feature_input",
        "motion_pool",
        "target_motion_tokens_usage",
    }
)
ACTION_PROVENANCE_ENCODER_FIELDS = frozenset(
    {"contract", "contract_sha256", "checkpoint_identities"}
)
ACTION_PROVENANCE_TENSOR_FIELDS = frozenset(
    {
        "source_latent",
        "target_latent",
        "text_context",
        "source_vlm_context",
        "target_motion_tokens",
    }
)
ACTION_PROVENANCE_PAYLOAD_FIELDS = frozenset({"path", "sha256"})


class ActionDatasetError(ValueError):
    """Raised when action training data violates the closed contract."""


def _validate_materialization_sampling(
    preprocessing: Any,
    *,
    expected: DataConfig,
    sample_id: str,
) -> None:
    """Bind latent payloads to the configured raw temporal/spatial semantics."""

    if not expected.require_materialization_metadata:
        return
    if not isinstance(preprocessing, Mapping):
        raise ActionDatasetError(
            f"sample {sample_id} preprocessing metadata must be an object"
        )
    if expected.temporal_mode == "full_81_25fps":
        indices = list(range(81))
        sampling_policy = "all_frames_in_order_no_temporal_subsampling"
        temporal_subsampled = False
    elif expected.temporal_mode == "smoke_41_12p5fps":
        indices = list(range(0, 81, 2))
        sampling_policy = "explicit_stride_2_smoke_ablation_only"
        temporal_subsampled = True
    else:
        raise ActionDatasetError(
            f"sample {sample_id} uses unsupported real temporal mode "
            f"{expected.temporal_mode!r}"
        )
    exact = {
        "temporal_mode": expected.temporal_mode,
        "spatial_profile": expected.spatial_profile,
        "frame_indices": indices,
        "source_frame_count": expected.expected_raw_num_frames,
        "target_frame_count": expected.expected_raw_num_frames,
        "materialized_frame_count": expected.video_num_frames,
        "sampling_policy": sampling_policy,
        "temporal_subsampled": temporal_subsampled,
    }
    for field, expected_value in exact.items():
        actual = preprocessing.get(field)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ActionDatasetError(
                f"sample {sample_id} preprocessing {field} differs: "
                f"expected={expected_value!r}, actual={actual!r}"
            )
    for field, expected_value in (
        ("source_fps", expected.expected_raw_fps),
        ("target_fps", expected.expected_raw_fps),
        ("materialized_fps", expected.video_fps),
    ):
        actual = preprocessing.get(field)
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected_value, rel_tol=0.0, abs_tol=1e-3)
        ):
            raise ActionDatasetError(
                f"sample {sample_id} preprocessing {field} differs: "
                f"expected={expected_value}, actual={actual!r}"
            )
    allowed_buckets = {
        (expected.video_height, expected.video_width),
    }
    if expected.allow_transpose:
        allowed_buckets.add((expected.video_width, expected.video_height))
    bucket = preprocessing.get("bucket_hw")
    if (
        type(bucket) is not list
        or len(bucket) != 2
        or any(type(size) is not int for size in bucket)
        or tuple(bucket) not in allowed_buckets
    ):
        raise ActionDatasetError(
            f"sample {sample_id} preprocessing bucket_hw differs: "
            f"expected one of {sorted(allowed_buckets)}, actual={bucket!r}"
        )


def _closed_mapping(
    value: Any, *, name: str, fields: frozenset[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionDatasetError(f"{name} must be an object")
    result = dict(value)
    actual = set(result)
    if actual != fields:
        raise ActionDatasetError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return result


def _digest(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActionDatasetError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ActionDatasetError(f"{name} must be a non-empty string")
    return value


def _finite_cpu_tensor(
    value: Any,
    *,
    name: str,
    ndim: int,
    final_dim: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ActionDatasetError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ActionDatasetError(f"{name} must have {ndim} dimensions")
    if not value.is_floating_point():
        raise ActionDatasetError(f"{name} must have a floating dtype")
    if min(value.shape) <= 0:
        raise ActionDatasetError(f"{name} cannot have an empty dimension")
    if final_dim is not None and value.shape[-1] != final_dim:
        raise ActionDatasetError(f"{name} last dimension must be {final_dim}")
    if value.device.type != "cpu":
        raise ActionDatasetError(f"{name} must be stored on CPU")
    if not bool(torch.isfinite(value).all()):
        raise ActionDatasetError(f"{name} contains NaN or Inf")
    return value


def action_tensor_sha256(value: Tensor) -> str:
    """Hash dtype, shape, and contiguous bytes using the materializer contract."""

    if not isinstance(value, Tensor):
        raise ActionDatasetError("tensor digest input must be a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    try:
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    except (RuntimeError, TypeError):
        raw = bytes(tensor.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ActionDatasetError(f"value is not canonical JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _task_type(value: Any, *, allowed_task_types: Sequence[str]) -> str:
    if type(value) is not str or value not in allowed_task_types:
        raise ActionDatasetError(
            f"task_type must be one of {sorted(allowed_task_types)}"
        )
    return value


def validate_action_payload(
    value: Mapping[str, Any],
    *,
    expected_motion_tokens: int | None = None,
    allowed_task_types: Sequence[str] = ACTION_TASK_TYPES,
) -> dict[str, Any]:
    """Validate one closed, CPU-only precomputed training payload."""

    payload = _closed_mapping(
        value, name="action payload", fields=ACTION_PAYLOAD_FIELDS
    )
    if payload["format"] != ACTION_PAYLOAD_FORMAT:
        raise ActionDatasetError(
            f"payload format must be {ACTION_PAYLOAD_FORMAT!r}"
        )
    payload["sample_id"] = _nonempty_string(
        payload["sample_id"], name="sample_id"
    )
    payload["encoder_contract"] = validate_encoder_contract(
        payload["encoder_contract"]
    )
    source = _finite_cpu_tensor(
        payload["source_latent"], name="source_latent", ndim=4
    )
    target = _finite_cpu_tensor(
        payload["target_latent"], name="target_latent", ndim=4
    )
    if source.shape != target.shape:
        raise ActionDatasetError("source_latent and target_latent shapes differ")
    expected_channels = len(WAN21_VAE_CHANNEL_MEAN)
    if source.shape[0] != expected_channels:
        raise ActionDatasetError(
            f"source and target latents must have {expected_channels} channels"
        )
    _finite_cpu_tensor(
        payload["text_context"],
        name="text_context",
        ndim=2,
        final_dim=UMT5_EMBEDDING_DIM,
    )
    _finite_cpu_tensor(
        payload["source_vlm_context"],
        name="source_vlm_context",
        ndim=2,
        final_dim=VLM_EMBEDDING_DIM,
    )
    target_motion_tokens = _finite_cpu_tensor(
        payload["target_motion_tokens"],
        name="target_motion_tokens",
        ndim=2,
        final_dim=MOTION_TOKEN_DIM,
    )
    if expected_motion_tokens is not None:
        if type(expected_motion_tokens) is not int or expected_motion_tokens <= 0:
            raise ActionDatasetError(
                "expected_motion_tokens must be a positive integer"
            )
        if target_motion_tokens.shape[0] != expected_motion_tokens:
            raise ActionDatasetError(
                "target_motion_tokens first dimension must be "
                f"{expected_motion_tokens}"
            )
    payload["task_type"] = _task_type(
        payload["task_type"], allowed_task_types=allowed_task_types
    )
    if type(payload["preview_only"]) is not bool:
        raise ActionDatasetError("preview_only must be bool")
    return payload


def _safe_load(data: bytes, *, path: Path) -> Mapping[str, Any]:
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise ActionDatasetError(
            "this PyTorch lacks safe weights_only loading; upgrade PyTorch"
        )
    value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ActionDatasetError(f"payload at {path} is not an object")
    return value


def _resolve_bound_path(root: Path, raw_path: Any, *, name: str) -> Path:
    if type(raw_path) is not str or not raw_path:
        raise ActionDatasetError(f"{name} must be a non-empty relative path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ActionDatasetError(
            f"{name} must be relative without parent-directory components"
        )
    candidate = root / relative
    current = candidate
    while current != root:
        if current.is_symlink():
            raise ActionDatasetError(f"{name} contains a symlink: {current}")
        parent = current.parent
        if parent == current:
            raise ActionDatasetError(f"{name} escaped its root")
        current = parent
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ActionDatasetError(f"{name} escaped its root") from error
    return resolved


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActionDatasetError(f"duplicate JSON key in provenance: {key!r}")
        result[key] = value
    return result


def _validate_action_provenance(
    value: Any,
    *,
    row: Mapping[str, Any],
    payload_path: Path,
    manifest_root: Path,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionDatasetError("action provenance must be an object")
    raw = dict(value)
    if type(raw.get("preview_only")) is not bool:
        raise ActionDatasetError("provenance preview_only must be bool")
    expected_fields = (
        ACTION_PROVENANCE_FIELDS
        if raw["preview_only"]
        else ACTION_PROVENANCE_PRODUCTION_FIELDS
    )
    provenance = _closed_mapping(
        raw, name="action provenance", fields=expected_fields
    )
    if provenance["schema_version"] != ACTION_PROVENANCE_FORMAT:
        raise ActionDatasetError(
            f"provenance schema_version must be {ACTION_PROVENANCE_FORMAT!r}"
        )
    for field in ("sample_id", "task_type", "preview_only"):
        if provenance[field] != row[field]:
            raise ActionDatasetError(
                f"provenance {field} differs from manifest for sample "
                f"{row['sample_id']}"
            )
    for field in ("parent_id", "split_group", "direction", "post_video_acceptance"):
        _nonempty_string(provenance[field], name=f"provenance {field}")
    for field in (
        "preview_only",
        "training_authorized",
        "training_use_forbidden",
        "production_eligible",
    ):
        if type(provenance[field]) is not bool:
            raise ActionDatasetError(f"provenance {field} must be bool")
    if provenance["preview_only"]:
        if (
            provenance["training_authorized"]
            or not provenance["training_use_forbidden"]
            or provenance["production_eligible"]
        ):
            raise ActionDatasetError(
                "preview provenance must remain training-forbidden and non-production"
            )
    else:
        if (
            not provenance["training_authorized"]
            or provenance["training_use_forbidden"]
            or not provenance["production_eligible"]
            or provenance["post_video_acceptance"] != "accepted"
        ):
            raise ActionDatasetError(
                "non-preview provenance must be training-authorized, "
                "training-allowed, production-eligible, and post-video accepted"
            )
        release = _closed_mapping(
            provenance["training_release"],
            name="provenance training_release",
            fields=ACTION_TRAINING_RELEASE_FIELDS,
        )
        if release["schema_version"] != ACTION_TRAINING_RELEASE_FORMAT:
            raise ActionDatasetError(
                "provenance training_release schema_version differs"
            )
        if release["verification_status"] != "verified":
            raise ActionDatasetError(
                "provenance training_release must have verified status"
            )
        expected_row_digest = _digest(
            release["sample_row_sha256"],
            name="provenance training_release sample_row_sha256",
        )
        bound_file_bytes: dict[str, bytes] = {}
        for path_field, digest_field in (
            ("release_path", "release_sha256"),
            ("verification_receipt_path", "verification_receipt_sha256"),
        ):
            bound_path = _resolve_bound_path(
                manifest_root,
                release[path_field],
                name=f"provenance training_release {path_field}",
            )
            if not bound_path.is_file():
                raise ActionDatasetError(
                    f"provenance training_release file does not exist: {bound_path}"
                )
            expected_digest = _digest(
                release[digest_field],
                name=f"provenance training_release {digest_field}",
            )
            bound_bytes = bound_path.read_bytes()
            actual_digest = hashlib.sha256(bound_bytes).hexdigest()
            if actual_digest != expected_digest:
                raise ActionDatasetError(
                    f"provenance training_release {digest_field} differs"
                )
            bound_file_bytes[path_field] = bound_bytes

        try:
            receipt_value = json.loads(
                bound_file_bytes["verification_receipt_path"].decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ActionDatasetError(
                f"invalid training-release verification receipt: {error}"
            ) from error
        receipt = _closed_mapping(
            receipt_value,
            name="training-release verification receipt",
            fields=ACTION_TRAINING_RELEASE_VERIFICATION_FIELDS,
        )
        if (
            receipt["schema_version"]
            != ACTION_TRAINING_RELEASE_VERIFICATION_FORMAT
        ):
            raise ActionDatasetError(
                "training-release verification receipt schema_version differs"
            )
        if receipt["status"] != release["verification_status"]:
            raise ActionDatasetError(
                "training-release verification status differs from provenance"
            )
        if receipt["status"] != "verified":
            raise ActionDatasetError(
                "training-release verification receipt must be verified"
            )
        if receipt["sample_id"] != provenance["sample_id"]:
            raise ActionDatasetError(
                "training-release verification receipt sample_id differs"
            )
        receipt_release_digest = _digest(
            receipt["release_sha256"],
            name="training-release verification receipt release_sha256",
        )
        if receipt_release_digest != release["release_sha256"]:
            raise ActionDatasetError(
                "training-release verification receipt release_sha256 differs"
            )
        release_row = _closed_mapping(
            receipt["release_row"],
            name="training-release verification receipt release_row",
            fields=ACTION_TRAINING_RELEASE_ROW_FIELDS,
        )
        if (
            release_row["sample_id"] != provenance["sample_id"]
            or release_row["training_authorized"] is not True
            or release_row["training_use_forbidden"] is not False
            or release_row["production_eligible"] is not True
            or release_row["post_video_acceptance"] != "accepted"
        ):
            raise ActionDatasetError(
                "training-release verification receipt does not authorize sample"
            )
        if _canonical_json_sha256(release_row) != expected_row_digest:
            raise ActionDatasetError(
                "training-release sample_row_sha256 differs from verified row"
            )

    preview_join = _closed_mapping(
        provenance["preview_join"],
        name="provenance preview_join",
        fields=ACTION_PROVENANCE_PREVIEW_JOIN_FIELDS,
    )
    _nonempty_string(
        preview_join["manifest_path"], name="provenance preview manifest_path"
    )
    for field in (
        "manifest_sha256",
        "row_digest",
        "row_file_sha256",
        "upstream_provenance_sha256",
    ):
        _digest(preview_join[field], name=f"provenance preview_join {field}")

    media = _closed_mapping(
        provenance["media"],
        name="provenance media",
        fields=ACTION_PROVENANCE_MEDIA_FIELDS,
    )
    for field in ("source_video_path", "target_video_path", "shared_i0_path"):
        _nonempty_string(media[field], name=f"provenance media {field}")
    for field in (
        "source_video_sha256",
        "target_video_sha256",
        "shared_i0_sha256",
    ):
        _digest(media[field], name=f"provenance media {field}")
    if not isinstance(media["preprocessing"], Mapping):
        raise ActionDatasetError("provenance media preprocessing must be an object")

    conditioning = _closed_mapping(
        provenance["conditioning"],
        name="provenance conditioning",
        fields=ACTION_PROVENANCE_CONDITIONING_FIELDS,
    )
    for field in (
        "instruction",
        "instruction_source",
        "target_caption",
        "target_caption_origin",
        "motion_text",
        "motion_teacher_visual_input",
        "motion_teacher_feature_input",
        "motion_pool",
        "target_motion_tokens_usage",
    ):
        _nonempty_string(conditioning[field], name=f"provenance conditioning {field}")
    for field in (
        "instruction_sha256",
        "generation_instruction_sha256",
        "target_caption_sha256",
        "motion_text_sha256",
    ):
        _digest(conditioning[field], name=f"provenance conditioning {field}")
    for text_field, digest_field in (
        ("instruction", "instruction_sha256"),
        ("target_caption", "target_caption_sha256"),
        ("motion_text", "motion_text_sha256"),
    ):
        actual_text_digest = hashlib.sha256(
            conditioning[text_field].encode("utf-8")
        ).hexdigest()
        if actual_text_digest != conditioning[digest_field]:
            raise ActionDatasetError(
                f"provenance conditioning {digest_field} differs from text"
            )
    if conditioning["motion_teacher_visual_input"] != "target_video_only":
        raise ActionDatasetError(
            "motion teacher visual input must be target_video_only"
        )
    if conditioning["motion_teacher_feature_input"] != "canonical_motion_text_only":
        raise ActionDatasetError(
            "motion teacher feature input must be canonical_motion_text_only"
        )
    if conditioning["target_motion_tokens_usage"] != "planner_loss_only":
        raise ActionDatasetError(
            "target motion tokens must be marked planner_loss_only"
        )

    encoder = _closed_mapping(
        provenance["encoder"],
        name="provenance encoder",
        fields=ACTION_PROVENANCE_ENCODER_FIELDS,
    )
    contract = validate_encoder_contract(encoder["contract"])
    contract_digest = _digest(
        encoder["contract_sha256"], name="provenance encoder contract_sha256"
    )
    if encoder_contract_sha256(contract) != contract_digest:
        raise ActionDatasetError("provenance encoder contract digest differs")
    if not isinstance(encoder["checkpoint_identities"], Mapping):
        raise ActionDatasetError(
            "provenance encoder checkpoint_identities must be an object"
        )

    tensor_sha256 = _closed_mapping(
        provenance["tensor_sha256"],
        name="provenance tensor_sha256",
        fields=ACTION_PROVENANCE_TENSOR_FIELDS,
    )
    for field in ACTION_PROVENANCE_TENSOR_FIELDS:
        _digest(tensor_sha256[field], name=f"provenance tensor_sha256 {field}")

    bound_payload = _closed_mapping(
        provenance["payload"],
        name="provenance payload",
        fields=ACTION_PROVENANCE_PAYLOAD_FIELDS,
    )
    bound_payload_path = _resolve_bound_path(
        manifest_root, bound_payload["path"], name="provenance payload path"
    )
    if bound_payload_path != payload_path:
        raise ActionDatasetError("provenance payload path differs from manifest")
    if _digest(
        bound_payload["sha256"], name="provenance payload sha256"
    ) != row["payload_sha256"]:
        raise ActionDatasetError("provenance payload digest differs from manifest")

    if payload is not None:
        if contract != payload["encoder_contract"]:
            raise ActionDatasetError("provenance encoder contract differs from payload")
        for field in ACTION_PROVENANCE_TENSOR_FIELDS:
            actual_tensor_digest = action_tensor_sha256(payload[field])
            if actual_tensor_digest != tensor_sha256[field]:
                raise ActionDatasetError(
                    f"provenance tensor digest differs for {field}"
                )
    return provenance


def _load_provenance_bytes(
    data: bytes,
    *,
    path: Path,
    row: Mapping[str, Any],
    payload_path: Path,
    manifest_root: Path,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActionDatasetError(f"invalid provenance JSON at {path}: {error}") from error
    return _validate_action_provenance(
        value,
        row=row,
        payload_path=payload_path,
        manifest_root=manifest_root,
        payload=payload,
    )


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ActionDatasetError(f"cannot read manifest at {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ActionDatasetError(
                f"invalid JSON on manifest line {line_number}: {error}"
            ) from error
        row = _closed_mapping(
            raw,
            name=f"manifest line {line_number}",
            fields=ACTION_MANIFEST_ROW_FIELDS,
        )
        if row["format"] != ACTION_MANIFEST_FORMAT:
            raise ActionDatasetError(
                f"manifest row format must be {ACTION_MANIFEST_FORMAT!r}"
            )
        row["sample_id"] = _nonempty_string(
            row["sample_id"], name=f"manifest line {line_number} sample_id"
        )
        row["payload_sha256"] = _digest(
            row["payload_sha256"],
            name=f"manifest line {line_number} payload_sha256",
        )
        row["provenance_sha256"] = _digest(
            row["provenance_sha256"],
            name=f"manifest line {line_number} provenance_sha256",
        )
        if type(row["preview_only"]) is not bool:
            raise ActionDatasetError(
                f"manifest line {line_number} preview_only must be bool"
            )
        rows.append(row)
    if not rows:
        raise ActionDatasetError("action manifest is empty")
    return rows


class ActionLatentDataset(Dataset[dict[str, Any]]):
    """Digest-bound action payloads with preview data rejected by default."""

    def __init__(
        self,
        manifest_path: os.PathLike[str] | str,
        *,
        payload_root: os.PathLike[str] | str | None = None,
        expected_motion_tokens: int | None = None,
        expected_latent_shapes: Sequence[Sequence[int]] | None = None,
        expected_data_config: DataConfig | None = None,
        allowed_task_types: Sequence[str] = ACTION_TASK_TYPES,
        allow_preview: bool = False,
        verify_payload_digest: bool = True,
    ) -> None:
        if type(allow_preview) is not bool:
            raise ActionDatasetError("allow_preview must be bool")
        if type(verify_payload_digest) is not bool:
            raise ActionDatasetError("verify_payload_digest must be bool")
        allowed = tuple(allowed_task_types)
        if not allowed or any(
            type(value) is not str or value not in ACTION_TASK_TYPES
            for value in allowed
        ):
            raise ActionDatasetError("allowed_task_types is invalid")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_root = self.manifest_path.parent
        self.payload_root = Path(
            self.manifest_root if payload_root is None else payload_root
        ).resolve()
        if not self.payload_root.is_dir():
            raise ActionDatasetError(
                f"payload_root is not a directory: {self.payload_root}"
            )
        self.expected_motion_tokens = expected_motion_tokens
        if expected_data_config is not None and not isinstance(
            expected_data_config, DataConfig
        ):
            raise ActionDatasetError("expected_data_config must be a DataConfig")
        self.expected_data_config = expected_data_config
        configured_shapes = (
            expected_data_config.expected_latent_shapes
            if expected_data_config is not None
            else None
        )
        if expected_latent_shapes is None and configured_shapes is not None:
            expected_latent_shapes = configured_shapes
        elif (
            expected_latent_shapes is not None
            and configured_shapes is not None
            and tuple(tuple(shape) for shape in expected_latent_shapes)
            != configured_shapes
        ):
            raise ActionDatasetError(
                "expected_latent_shapes differs from expected_data_config"
            )
        if expected_latent_shapes is None:
            self.expected_latent_shapes: tuple[tuple[int, int, int, int], ...] | None = None
        else:
            shapes: list[tuple[int, int, int, int]] = []
            for index, raw_shape in enumerate(expected_latent_shapes):
                shape = tuple(raw_shape)
                if (
                    len(shape) != 4
                    or any(type(size) is not int or size <= 0 for size in shape)
                ):
                    raise ActionDatasetError(
                        f"expected_latent_shapes[{index}] must be four positive integers"
                    )
                shapes.append(shape)
            if not shapes:
                raise ActionDatasetError("expected_latent_shapes cannot be empty")
            if len(set(shapes)) != len(shapes):
                raise ActionDatasetError("expected_latent_shapes contains duplicates")
            self.expected_latent_shapes = tuple(shapes)
        self.allowed_task_types = allowed
        self.allow_preview = allow_preview
        self.verify_payload_digest = verify_payload_digest
        self.rows: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        for row in _load_manifest(self.manifest_path):
            _task_type(row["task_type"], allowed_task_types=allowed)
            if row["preview_only"] and not allow_preview:
                raise ActionDatasetError(
                    f"sample {row['sample_id']} is preview-only and cannot train"
                )
            if row["sample_id"] in sample_ids:
                raise ActionDatasetError(f"duplicate sample_id: {row['sample_id']}")
            sample_ids.add(row["sample_id"])
            payload_path = _resolve_bound_path(
                self.payload_root, row["payload_path"], name="payload_path"
            )
            if not payload_path.is_file():
                raise ActionDatasetError(f"payload does not exist: {payload_path}")
            if verify_payload_digest:
                actual_digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
                if actual_digest != row["payload_sha256"]:
                    raise ActionDatasetError(
                        f"payload digest differs for sample {row['sample_id']}"
                    )
            provenance_path = _resolve_bound_path(
                self.manifest_root,
                row["provenance_path"],
                name="provenance_path",
            )
            if not provenance_path.is_file():
                raise ActionDatasetError(
                    f"provenance does not exist: {provenance_path}"
                )
            provenance_bytes = provenance_path.read_bytes()
            actual_provenance_digest = hashlib.sha256(provenance_bytes).hexdigest()
            if actual_provenance_digest != row["provenance_sha256"]:
                raise ActionDatasetError(
                    f"provenance digest differs for sample {row['sample_id']}"
                )
            provenance = _load_provenance_bytes(
                provenance_bytes,
                path=provenance_path,
                row=row,
                payload_path=payload_path,
                manifest_root=self.manifest_root,
            )
            if self.expected_data_config is not None:
                _validate_materialization_sampling(
                    provenance["media"]["preprocessing"],
                    expected=self.expected_data_config,
                    sample_id=row["sample_id"],
                )
            row["_payload_path"] = str(payload_path)
            row["_provenance_path"] = str(provenance_path)
            self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        payload_path = Path(row["_payload_path"])
        payload_bytes = payload_path.read_bytes()
        if self.verify_payload_digest:
            actual_digest = hashlib.sha256(payload_bytes).hexdigest()
            if actual_digest != row["payload_sha256"]:
                raise ActionDatasetError(
                    f"payload digest differs for sample {row['sample_id']} at load time"
                )
        payload = validate_action_payload(
            _safe_load(payload_bytes, path=payload_path),
            expected_motion_tokens=self.expected_motion_tokens,
            allowed_task_types=self.allowed_task_types,
        )
        for field in ("sample_id", "task_type", "preview_only"):
            if payload[field] != row[field]:
                raise ActionDatasetError(
                    f"payload {field} differs from manifest for sample "
                    f"{row['sample_id']}"
                )
        if payload["preview_only"] and not self.allow_preview:
            raise ActionDatasetError(
                f"sample {row['sample_id']} became preview-only at load time"
            )
        provenance_path = Path(row["_provenance_path"])
        provenance_bytes = provenance_path.read_bytes()
        actual_provenance_digest = hashlib.sha256(provenance_bytes).hexdigest()
        if actual_provenance_digest != row["provenance_sha256"]:
            raise ActionDatasetError(
                f"provenance digest differs for sample {row['sample_id']} at load time"
            )
        provenance = _load_provenance_bytes(
            provenance_bytes,
            path=provenance_path,
            row=row,
            payload_path=payload_path,
            manifest_root=self.manifest_root,
            payload=payload,
        )
        if self.expected_data_config is not None:
            _validate_materialization_sampling(
                provenance["media"]["preprocessing"],
                expected=self.expected_data_config,
                sample_id=row["sample_id"],
            )
        latent_shape = tuple(payload["source_latent"].shape)
        if (
            self.expected_latent_shapes is not None
            and latent_shape not in self.expected_latent_shapes
        ):
            raise ActionDatasetError(
                f"sample {row['sample_id']} latent shape {latent_shape} differs from "
                f"configured shapes {self.expected_latent_shapes}"
            )
        return {
            "sample_id": payload["sample_id"],
            "encoder_contract": payload["encoder_contract"],
            "encoder_contract_sha256": encoder_contract_sha256(
                payload["encoder_contract"]
            ),
            "source_latent": payload["source_latent"],
            "target_latent": payload["target_latent"],
            "text_context": payload["text_context"],
            "source_vlm_context": payload["source_vlm_context"],
            "target_motion_tokens": payload["target_motion_tokens"],
            "task_type": payload["task_type"],
            "preview_only": payload["preview_only"],
            "provenance_sha256": row["provenance_sha256"],
        }


def collate_action_latents(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ActionDatasetError("cannot collate an empty batch")
    contracts: list[dict[str, Any]] = []
    digests: list[str] = []
    for index, sample in enumerate(samples):
        try:
            contract = validate_encoder_contract(sample["encoder_contract"])
            supplied_digest = _digest(
                sample["encoder_contract_sha256"],
                name=f"sample {index} encoder_contract_sha256",
            )
        except KeyError as error:
            raise ActionDatasetError(
                f"sample {index} lacks encoder contract provenance"
            ) from error
        actual_digest = encoder_contract_sha256(contract)
        if supplied_digest != actual_digest:
            raise ActionDatasetError(
                f"sample {index} encoder_contract_sha256 differs from contract"
            )
        contracts.append(contract)
        digests.append(actual_digest)
    if any(
        contract != contracts[0] or digest != digests[0]
        for contract, digest in zip(contracts[1:], digests[1:])
    ):
        raise ActionDatasetError("cannot collate mixed encoder contracts")

    result: dict[str, Any] = {
        "encoder_contract": contracts[0],
        "encoder_contract_sha256": digests[0],
    }
    for field in ("source_latent", "target_latent", "target_motion_tokens"):
        try:
            result[field] = torch.stack([sample[field] for sample in samples])
        except (KeyError, RuntimeError) as error:
            raise ActionDatasetError(
                f"batch has missing or incompatible tensors for {field}"
            ) from error
    result["text_context"] = [sample["text_context"] for sample in samples]
    result["source_vlm_context"] = [
        sample["source_vlm_context"] for sample in samples
    ]
    result["sample_id"] = [sample["sample_id"] for sample in samples]
    result["task_type"] = [sample["task_type"] for sample in samples]
    result["provenance_sha256"] = [
        _digest(
            sample["provenance_sha256"],
            name=f"sample {index} provenance_sha256",
        )
        for index, sample in enumerate(samples)
    ]
    result["preview_only"] = torch.tensor(
        [sample["preview_only"] for sample in samples], dtype=torch.bool
    )
    return result


__all__ = [
    "ACTION_MANIFEST_FORMAT",
    "ACTION_MANIFEST_ROW_FIELDS",
    "ACTION_PAYLOAD_FIELDS",
    "ACTION_PAYLOAD_FORMAT",
    "ACTION_PROVENANCE_FIELDS",
    "ACTION_PROVENANCE_FORMAT",
    "ACTION_PROVENANCE_PRODUCTION_FIELDS",
    "ACTION_TRAINING_RELEASE_FIELDS",
    "ACTION_TRAINING_RELEASE_FORMAT",
    "ACTION_TRAINING_RELEASE_ROW_FIELDS",
    "ACTION_TRAINING_RELEASE_VERIFICATION_FIELDS",
    "ACTION_TRAINING_RELEASE_VERIFICATION_FORMAT",
    "ActionDatasetError",
    "ActionLatentDataset",
    "action_tensor_sha256",
    "collate_action_latents",
    "validate_action_payload",
]

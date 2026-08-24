#!/usr/bin/env python3
"""Native RV2V block-15 VJP primitives for query-wise action editing.

This module implements the *read-only* part of Q-MOSAIC.  A frozen Bernini
RV2V query is measured at ``block.15.output`` on the native noisy-target
suffix, after the condition prefix and official contiguous Ulysses-SP4
sharding have been accounted for.  A detached action/no-op measurement is
turned into one score cotangent, then the graph is replayed serially towards
either:

* the current clean latent (all Action-LoRA routes frozen/off), for the
  symmetric ``+q/-q`` exact81 direction gate; or
* the zero-init Action-LoRA-B coordinate, only after that direction gate.

The Action-LoRA gauge is deliberately narrow: block 0..15 cross-attention
Q/O, rank 8, alpha 8, deterministic FP32 A frozen forever, and exact-zero
FP32 B as the only differentiable parameter.  Its canonical B ordering is
Q/O interleaved per block and therefore matches the downstream 32 x
``(1536, 8)`` QP coordinate (393216 scalars).  This differs from the legacy
PAIR-v5 handle enumeration, which lists all Q parameters before all O
parameters; callers must use the canonical accessors below.

Every SP rank differentiates ``score / 4``.  Rank-local VJPs are then reduced
with SUM exactly once.  Dividing after that SUM is forbidden.  There is no
optimizer, parameter mutation, decoded success claim, or AUH validation in
this file.  A failed two-seed construction returns ``None`` at the explicit
fail-closed boundary; lower-level contract violations raise before a partial
row can escape.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import base64
import csv
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn

import pair_v5_action_adapter as pair_adapter
import pair_v5_native_bridge as native_bridge
import self_imagined_motion_cotangent_v1 as motion_cotangent
import materialize_self_imagined_owner_core2_v1 as owner_materializer
import source_self_native_ref_contrastive_v3 as native


SCHEMA_VERSION = "bernini-self-imagined-native-rv2v-hidden-vjp-v1"
LAYOUT_SCHEMA_VERSION = "bernini-native-rv2v-target-suffix-sp4-layout-v1"
GAUGE_SCHEMA_VERSION = "bernini-qmosaic-core16-fixed-a-b-only-gauge-v1"
ZERO_ROUTE_PROOF_SCHEMA_VERSION = (
    "bernini-qmosaic-core16-zero-route-structural-proof-v1"
)
COTANGENT_SCHEMA_VERSION = "bernini-qmosaic-detached-score-cotangent-v2"
VJP_ROW_SCHEMA_VERSION = "bernini-qmosaic-rank-local-vjp-row-v2"
SP4_ROW_SCHEMA_VERSION = "bernini-qmosaic-sp4-summed-vjp-row-v2"
OWNER_PACKET_SCHEMA_VERSION = "bernini-qmosaic-authenticated-owner-quotient-v1"
EDITOR_PACKET_SCHEMA_VERSION = "bernini-qmosaic-runtime-owned-editor-same-state-v1"
REPLAY_SESSION_SCHEMA_VERSION = "bernini-qmosaic-runtime-owned-replay-session-v1"
DIRECTION_GATE_SCHEMA_VERSION = "bernini-qmosaic-exact81-direction-gate-v3"
DIRECTION_GATE_SIGNATURE_SCHEME = "ed25519-canonical-json-v1"
EXACT81_MEDIA_PROBE_SCHEMA_VERSION = "bernini-portable-exact81-media-probe-v1"
PINNED_PYAV_VERSION = "13.1.0"
PINNED_PYAV_LIBRARY_VERSIONS = MappingProxyType(
    {
        "libavcodec": (61, 3, 100),
        "libavdevice": (61, 1, 100),
        "libavfilter": (10, 1, 100),
        "libavformat": (61, 1, 100),
        "libavutil": (59, 8, 100),
        "libswresample": (5, 1, 100),
        "libswscale": (8, 1, 100),
    }
)
PINNED_PYAV_MODULE_SHA256 = (
    "ee1cfd64a1e7449f27fc97f0cd65ffc6bfd13b1da3b7478510fb473602fd6ae3"
)
PINNED_PYAV_RECORD_SHA256 = (
    "103b49b8cdf3ae2049eed81b4a1f76e48bb8759563ff7b891b361634eb4d3233"
)
PINNED_PYAV_DISTRIBUTION_HASHED_TREE_SHA256 = (
    "58cdbcca117dab1e1e5db309fd8f6baa5fa1afc95f2c41b95dcbc42fab9c2043"
)
PINNED_PYAV_DISTRIBUTION_HASHED_FILE_COUNT = 228
PINNED_IMAGEIO_FFMPEG_VERSION = "0.6.0"
PINNED_IMAGEIO_FFMPEG_MODULE_SHA256 = (
    "41afc231dfeca422c692f9a219b51778b7560c5c4dffa097bf92f80872d1fd8a"
)
PINNED_IMAGEIO_FFMPEG_RECORD_SHA256 = (
    "a2aec39e8a934b5e0202a55e9c34a63456dd6566f27a14a358ffae0f5453eeca"
)
PINNED_IMAGEIO_FFMPEG_DISTRIBUTION_HASHED_TREE_SHA256 = (
    "69f5f4004c57d3fc28fa2003e11049755653f5061250dcd2c36372bbef809101"
)
PINNED_IMAGEIO_FFMPEG_DISTRIBUTION_HASHED_FILE_COUNT = 8
PINNED_BUNDLED_FFMPEG_BASENAME = "ffmpeg-linux-x86_64-v7.0.2"
PINNED_BUNDLED_FFMPEG_SHA256 = (
    "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
)
PINNED_BUNDLED_FFMPEG_VERSION_LINE = (
    "ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/  "
    "Copyright (c) 2000-2024 the FFmpeg developers"
)
CLOSED_NATIVE_ADAPTER_REGISTRY_ID = "qmosaic-core16-fixed-a-b-only-v1"
FUNCTIONAL_PRESERVATION_SCHEMA_VERSION = (
    "bernini-qmosaic-paired-functional-preservation-cone-v1"
)

TOTAL_BLOCKS_1P3B = 30
ACTION_BLOCK_INDICES = tuple(range(16))
HOOK_BLOCK_INDEX = 15
HIDDEN_SIZE = 1536
PACKED_PREDICTION_DIM = 64
LORA_RANK = 8
LORA_ALPHA = 8.0
LORA_SCALE = LORA_ALPHA / float(LORA_RANK)
FIXED_LORA_A_SEED = 720260809
SP_SIZE = 4
LATENT_PHASES = 21
SPATIAL_SKETCH_COORDINATES = 16
SPATIAL_SKETCH_SEED = 620260809
QUERY_SEED_COUNT = 2
NATIVE_SCHEDULE_INDEX = 33
NATIVE_TIMESTEP = 516
NATIVE_SIGMA = 0.5161304473876953
BRANCH_NAMES = ("none", "V", "I", "VI")
CONDITION_PHASES_BY_BRANCH = MappingProxyType(
    {"none": 0, "V": 21, "I": 4, "VI": 25}
)
REPLAY_RTOL = 2.0e-5
REPLAY_ATOL = 2.0e-5

EXACT81_MEDIA_PROBE_FIELDS = frozenset(
    {
        "schema_version",
        "format_name",
        "video_stream_count",
        "container_stream_count",
        "codec_name",
        "width",
        "height",
        "pix_fmt",
        "avg_frame_rate",
        "pyav_backend_name",
        "pyav_version",
        "pyav_linked_library_versions",
        "pyav_module_file_sha256",
        "pyav_distribution_record_sha256",
        "pyav_distribution_hashed_tree_digest_sha256",
        "pyav_distribution_hashed_file_count",
        "pyav_decoded_frame_count",
        "pyav_first_pts",
        "pyav_last_pts",
        "pyav_time_base",
        "pyav_pts_cadence_rational",
        "pyav_exact_25fps_pts_cadence",
        "pyav_rgb24_frame_transcript_sha256",
        "imageio_ffmpeg_version",
        "imageio_ffmpeg_module_file_sha256",
        "imageio_ffmpeg_distribution_record_sha256",
        "imageio_ffmpeg_distribution_hashed_tree_digest_sha256",
        "imageio_ffmpeg_distribution_hashed_file_count",
        "bundled_ffmpeg_executable_realpath",
        "bundled_ffmpeg_executable_sha256",
        "bundled_ffmpeg_version_line",
        "bundled_ffmpeg_framemd5_frame_count",
        "bundled_ffmpeg_framemd5_transcript_sha256",
        "decoded_frame_transcript_sha256",
    }
)

PRESERVATION_RADEMACHER_SEEDS = (810260809, 910260809)
FUNCTIONAL_PRESERVATION_SPECS = MappingProxyType(
    {
        "noop_predicted_clean_invariance": ("noop", "VI_noop"),
        "action_noop_temporal_dc_static_appearance": ("identity", "VI_action-minus-noop"),
        "action_noop_spatial_dc": ("camera", "VI_action-minus-noop"),
        "action_noop_camera_ramp_x": ("camera", "VI_action-minus-noop"),
        "action_noop_camera_ramp_y": ("camera", "VI_action-minus-noop"),
        "action_noop_high_pass_detail": ("sharpness", "VI_action-minus-noop"),
        "action_noop_temporal_lag1": ("flicker", "VI_action-minus-noop"),
        "action_noop_temporal_lag2": ("flicker", "VI_action-minus-noop"),
        "action_noop_temporal_lag4": ("flicker", "VI_action-minus-noop"),
        "source_video_v_none_sensitivity": ("background", "V-minus-none"),
    }
)
WEAK_I_AXIS_FUNCTIONAL_ID = "weak_i_axis_identity"
_FUNCTIONAL_ROW_TOKEN = object()
_FUNCTIONAL_CONE_TOKEN = object()


def _predicted_clean_grid(
    value: Any, *, patch_height: int, patch_width: int, label: str
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 3
        or tuple(map(int, value.shape[:1])) != (1,)
        or int(value.shape[1]) != LATENT_PHASES * patch_height * patch_width
        or int(value.shape[2]) != PACKED_PREDICTION_DIM
        or patch_height <= 0
        or patch_width <= 0
        or not value.requires_grad
        or value.grad_fn is None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise NativeRV2VHiddenVJPError(
            f"{label} must be graph-connected native predicted-clean target tokens"
        )
    return value.reshape(
        1, LATENT_PHASES, patch_height, patch_width, PACKED_PREDICTION_DIM
    )


def paired_functional_preservation_feature(
    functional_id: str,
    *,
    patch_height: int,
    patch_width: int,
    noop_predicted_clean: torch.Tensor,
    action_predicted_clean: Optional[torch.Tensor] = None,
    video_predicted_clean: Optional[torch.Tensor] = None,
    image_predicted_clean: Optional[torch.Tensor] = None,
    none_predicted_clean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact functional whose independent Rademacher VJP becomes one slab.

    This function defines representation geometry only.  It does not seal a
    row and therefore cannot cross the authoritative VJP/QP boundary by
    itself.
    """

    if (
        functional_id not in FUNCTIONAL_PRESERVATION_SPECS
        and functional_id != WEAK_I_AXIS_FUNCTIONAL_ID
    ):
        raise NativeRV2VHiddenVJPError("paired preservation functional differs")
    noop = _predicted_clean_grid(
        noop_predicted_clean,
        patch_height=patch_height,
        patch_width=patch_width,
        label="no-op predicted clean",
    )
    if functional_id == "noop_predicted_clean_invariance":
        feature = noop
    elif functional_id in (
        "source_video_v_none_sensitivity",
        WEAK_I_AXIS_FUNCTIONAL_ID,
    ):
        coordinate = _predicted_clean_grid(
            (
                video_predicted_clean
                if functional_id == "source_video_v_none_sensitivity"
                else image_predicted_clean
            ),
            patch_height=patch_height,
            patch_width=patch_width,
            label=(
                "V predicted clean"
                if functional_id == "source_video_v_none_sensitivity"
                else "I predicted clean"
            ),
        )
        none = _predicted_clean_grid(
            none_predicted_clean,
            patch_height=patch_height,
            patch_width=patch_width,
            label="none predicted clean",
        )
        feature = coordinate - none
    else:
        action = _predicted_clean_grid(
            action_predicted_clean,
            patch_height=patch_height,
            patch_width=patch_width,
            label="action predicted clean",
        )
        residual = action - noop
        if functional_id == "action_noop_temporal_dc_static_appearance":
            feature = residual.mean(dim=1)
        elif functional_id == "action_noop_spatial_dc":
            feature = residual.mean(dim=(2, 3))
        elif functional_id in (
            "action_noop_camera_ramp_x",
            "action_noop_camera_ramp_y",
        ):
            length = patch_width if functional_id.endswith("_x") else patch_height
            ramp = torch.linspace(
                -1.0, 1.0, length, dtype=torch.float32, device=residual.device
            )
            ramp = ramp - ramp.mean()
            ramp = ramp / torch.linalg.vector_norm(ramp).clamp_min(1.0e-12)
            if functional_id.endswith("_x"):
                feature = torch.einsum("bthwd,w->bthd", residual, ramp)
            else:
                feature = torch.einsum("bthwd,h->btwd", residual, ramp)
        elif functional_id == "action_noop_high_pass_detail":
            horizontal = residual[:, :, :, 1:, :] - residual[:, :, :, :-1, :]
            vertical = residual[:, :, 1:, :, :] - residual[:, :, :-1, :, :]
            feature = torch.cat(
                (horizontal.reshape(1, -1), vertical.reshape(1, -1)), dim=1
            )
        else:
            lag = int(functional_id.rsplit("lag", 1)[1])
            feature = residual[:, lag:] - residual[:, :-lag]
    result = feature.float().reshape(1, -1).contiguous()
    if (
        not result.requires_grad
        or result.grad_fn is None
        or not bool(torch.isfinite(result).all().item())
        or result.numel() == 0
    ):
        raise NativeRV2VHiddenVJPError(
            "paired preservation functional graph/value differs"
        )
    return result


def fixed_rademacher_functional_scalar(
    feature: torch.Tensor, *, rademacher_seed: int
) -> torch.Tensor:
    """Apply one fixed row; callers must never average seeds before a QP."""

    if (
        not isinstance(feature, torch.Tensor)
        or feature.dtype != torch.float32
        or feature.ndim != 2
        or int(feature.shape[0]) != 1
        or not feature.requires_grad
        or feature.grad_fn is None
        or rademacher_seed not in PRESERVATION_RADEMACHER_SEEDS
    ):
        raise NativeRV2VHiddenVJPError("functional Rademacher input differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rademacher_seed)
    signs = (
        torch.randint(
            0,
            2,
            tuple(feature.shape),
            generator=generator,
            dtype=torch.int64,
            device="cpu",
        ).float()
        * 2.0
        - 1.0
    ).to(device=feature.device)
    scalar = (feature * signs).sum() / math.sqrt(float(feature.numel()))
    if scalar.numel() != 1 or not scalar.requires_grad or scalar.grad_fn is None:
        raise NativeRV2VHiddenVJPError("functional Rademacher scalar differs")
    return scalar.reshape(())

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OWNER_PACKET_TOKEN = object()
_EDITOR_PACKET_TOKEN = object()
_REPLAY_SESSION_TOKEN = object()
_DIRECTION_GATE_TOKEN = object()
_COTANGENT_PACKET_TOKEN = object()
_CHECKPOINT_CONTENT_TOKEN = object()
_SP4_COLLECTIVE_TOKEN = object()
_RANK_VJP_ROW_TOKEN = object()
_SP4_VJP_ROW_TOKEN = object()
_EDITOR_RUNTIME_INPUT_TOKEN = object()


def _canonical_b_parameter_names() -> tuple[str, ...]:
    rows: list[str] = []
    for block in ACTION_BLOCK_INDICES:
        rows.append(f"blocks.{block}.attn2.to_q.action_lora_b.weight")
        rows.append(f"blocks.{block}.attn2.to_out.0.action_lora_b.weight")
    return tuple(rows)


CANONICAL_B_PARAMETER_NAMES = _canonical_b_parameter_names()
CANONICAL_A_PARAMETER_NAMES = tuple(
    name.replace("action_lora_b.weight", "action_lora_a.weight")
    for name in CANONICAL_B_PARAMETER_NAMES
)
CANONICAL_B_SHAPE = (HIDDEN_SIZE, LORA_RANK)
CANONICAL_A_SHAPE = (LORA_RANK, HIDDEN_SIZE)
CANONICAL_B_PARAMETER_COUNT = (
    len(CANONICAL_B_PARAMETER_NAMES) * HIDDEN_SIZE * LORA_RANK
)


class NativeRV2VHiddenVJPError(RuntimeError):
    """A layout, gauge, sketch, replay, or aggregation contract failed."""


class NativeRuntimeSealChangedError(NativeRV2VHiddenVJPError):
    """The immutable model graph/state changed; this process must be discarded."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise NativeRV2VHiddenVJPError(
            "receipt is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NativeRV2VHiddenVJPError(f"{label} must be lowercase SHA-256")
    return value


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise NativeRV2VHiddenVJPError("hashed artifact must be an absolute plain file")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_file(
    path: str | Path, *, expected_sha256: str, label: str
) -> tuple[dict[str, Any], Path, str]:
    source = Path(path)
    expected = _sha256(expected_sha256, label=f"{label} file SHA-256")
    observed = file_sha256(source)
    if observed != expected:
        raise NativeRV2VHiddenVJPError(f"{label} bytes changed")
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeRV2VHiddenVJPError(f"{label} JSON differs") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise NativeRV2VHiddenVJPError(f"{label} is not canonical create-only JSON")
    return value, source.resolve(strict=True), observed


def _tensor_runtime_binding(value: torch.Tensor, *, label: str) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "object_id": id(value),
            "shape": list(map(int, value.shape)),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "tensor_sha256": tensor_sha256(value, label=label),
        }
    )


def _tensor_runtime_metadata_binding(
    value: torch.Tensor, *, label: str
) -> Mapping[str, Any]:
    """Bind tensor identity/storage without copying its payload to the CPU.

    Full byte hashing remains mandatory at session construction and terminal
    publication.  This lighter binding is used only between model calls, where
    repeatedly cloning every Bernini parameter and runtime tensor to the CPU is
    prohibitively expensive.  Tensor version and storage-pointer checks catch
    ordinary in-place mutation immediately; the terminal full seal catches any
    mutation that evades a version counter before an artifact can be published.
    """

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise NativeRV2VHiddenVJPError(f"{label} must be a live real tensor")
    return MappingProxyType(
        {
            "object_id": id(value),
            "shape": list(map(int, value.shape)),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "layout": str(value.layout),
            "version": int(value._version),
            "storage_data_ptr": _parameter_storage_pointer(value),
        }
    )


def _assert_tensor_runtime_metadata_binding(
    value: Any, binding: Mapping[str, Any], *, label: str
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or id(value) != binding.get("object_id")
        or list(map(int, value.shape)) != binding.get("shape")
        or str(value.dtype) != binding.get("dtype")
        or str(value.device) != binding.get("device")
        or str(value.layout) != binding.get("layout")
        or int(value._version) != binding.get("version")
        or _parameter_storage_pointer(value) != binding.get("storage_data_ptr")
    ):
        raise NativeRuntimeSealChangedError(
            f"{label} runtime metadata changed; process is poisoned"
        )


def _assert_tensor_runtime_binding(
    value: Any, binding: Mapping[str, Any], *, label: str, require_same_object: bool
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or (require_same_object and id(value) != binding.get("object_id"))
        or list(map(int, value.shape)) != binding.get("shape")
        or str(value.dtype) != binding.get("dtype")
        or str(value.device) != binding.get("device")
        or tensor_sha256(value, label=label) != binding.get("tensor_sha256")
    ):
        raise NativeRV2VHiddenVJPError(f"{label} live binding changed")


def tensor_sha256(value: torch.Tensor, *, label: str) -> str:
    """Hash an owned contiguous tensor without NumPy."""

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or not bool(torch.isfinite(value).all().item())
    ):
        raise NativeRV2VHiddenVJPError(f"{label} must be a finite real tensor")
    owned = value.detach().to(device="cpu").contiguous().clone()
    # ``Tensor.untyped_storage`` is absent in Bernini's pinned Torch 1.12 and
    # iterating a TypedStorage materializes each byte in Python.  The pinned
    # storage writer copies the owned storage straight into a BytesIO in C++.
    # This stays deterministic and NumPy-free while binding every exact byte;
    # the original dtype/shape remain in the explicit header.
    payload = io.BytesIO()
    storage = owned.storage()
    untyped = storage._untyped() if callable(getattr(storage, "_untyped", None)) else owned.untyped_storage()
    untyped._write_file(payload, False, False, 1)
    raw = payload.getvalue()
    expected = int(owned.numel()) * int(owned.element_size())
    if len(raw) != expected:
        raise NativeRV2VHiddenVJPError(f"{label} owned storage closure differs")
    header = canonical_json_bytes(
        {
            "dtype": str(owned.dtype),
            "shape": list(map(int, owned.shape)),
            "numel": int(owned.numel()),
        }
    )
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


def _named_tensor_sha256(
    ordered: Sequence[tuple[str, torch.Tensor]], *, label: str
) -> str:
    if isinstance(ordered, Mapping) or not isinstance(ordered, Sequence) or not ordered:
        raise NativeRV2VHiddenVJPError(f"{label} must be a nonempty ordered sequence")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, value in ordered:
        if not isinstance(name, str) or name in seen:
            raise NativeRV2VHiddenVJPError(f"{label} names are invalid or repeated")
        seen.add(name)
        rows.append(
            {
                "name": name,
                "shape": list(map(int, value.shape)),
                "dtype": str(value.dtype),
                "tensor_sha256": tensor_sha256(value, label=f"{label} {name}"),
            }
        )
    return object_sha256(rows)


def _validate_checkpoint_content_tree(
    *,
    checkpoint_root: Path,
    content_manifest: Path,
    expected_manifest_sha256: str,
    expected_file_count: int,
) -> Mapping[str, Any]:
    root = checkpoint_root.resolve(strict=True)
    manifest = content_manifest.resolve(strict=True)
    if (
        not root.is_dir()
        or root.is_symlink()
        or not manifest.is_file()
        or manifest.is_symlink()
        or type(expected_file_count) is not int
        or expected_file_count <= 0
    ):
        raise NativeRV2VHiddenVJPError("checkpoint content authority differs")
    expected_manifest = _sha256(
        expected_manifest_sha256, label="checkpoint content manifest SHA-256"
    )
    if file_sha256(manifest) != expected_manifest:
        raise NativeRV2VHiddenVJPError(
            "checkpoint content manifest bytes changed"
        )
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise NativeRV2VHiddenVJPError(
            "checkpoint content manifest cannot be read"
        ) from error
    if len(lines) != expected_file_count:
        raise NativeRV2VHiddenVJPError(
            "checkpoint content manifest file count differs"
        )
    pattern = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
    expected: dict[str, str] = {}
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            raise NativeRV2VHiddenVJPError(
                "checkpoint manifest is not canonical sha256sum syntax"
            )
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise NativeRV2VHiddenVJPError(
                "checkpoint manifest contains an unsafe path"
            )
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in expected:
            raise NativeRV2VHiddenVJPError(
                "checkpoint manifest path closure differs"
            )
        expected[normalized] = digest
    actual: set[str] = set()
    try:
        descendants = tuple(root.rglob("*"))
    except OSError as error:
        raise NativeRV2VHiddenVJPError(
            "checkpoint content cannot be enumerated"
        ) from error
    for path in descendants:
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise NativeRV2VHiddenVJPError(
                "checkpoint content cannot be inspected"
            ) from error
        if stat.S_ISLNK(mode):
            raise NativeRV2VHiddenVJPError(
                "checkpoint content contains a non-cache symlink"
            )
        if stat.S_ISREG(mode):
            actual.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise NativeRV2VHiddenVJPError(
                "checkpoint content contains a non-regular entry"
            )
    if actual != set(expected):
        raise NativeRV2VHiddenVJPError(
            "checkpoint file closure differs from the signed manifest"
        )
    verified = []
    for relative in sorted(expected):
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise NativeRV2VHiddenVJPError(
                "checkpoint manifest path escaped its root"
            ) from error
        observed = file_sha256(path)
        if observed != expected[relative]:
            raise NativeRV2VHiddenVJPError(
                f"checkpoint content hash differs: {relative}"
            )
        verified.append({"path": relative, "sha256": observed})
    value = {
        "schema_version": "bernini-qmosaic-checkpoint-content-v1",
        "checkpoint_root": str(root),
        "content_manifest_path": str(manifest),
        "content_manifest_file_sha256": expected_manifest,
        "verified_file_count": len(verified),
        "verified_entries_digest": object_sha256(verified),
        "every_non_cache_file_rehashed": True,
    }
    return MappingProxyType({**value, "digest": object_sha256(value)})


@dataclass(frozen=True)
class ValidatedCheckpointContentManifest:
    checkpoint_root: Path
    content_manifest_path: Path
    expected_manifest_sha256: str
    expected_file_count: int
    content_receipt: Mapping[str, Any]
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def assert_live(self) -> None:
        if self._token is not _CHECKPOINT_CONTENT_TOKEN:
            raise NativeRV2VHiddenVJPError(
                "checkpoint content was not manifest-authenticated"
            )
        observed = _validate_checkpoint_content_tree(
            checkpoint_root=self.checkpoint_root,
            content_manifest=self.content_manifest_path,
            expected_manifest_sha256=self.expected_manifest_sha256,
            expected_file_count=self.expected_file_count,
        )
        if dict(observed) != dict(self.content_receipt):
            raise NativeRV2VHiddenVJPError(
                "live checkpoint content receipt changed"
            )

    def receipt(self) -> Mapping[str, Any]:
        self.assert_live()
        return self.content_receipt


def load_validated_checkpoint_content_manifest(
    *,
    checkpoint_root: str | Path,
    content_manifest_path: str | Path,
    expected_manifest_sha256: str,
    expected_file_count: int = 23,
) -> ValidatedCheckpointContentManifest:
    receipt = _validate_checkpoint_content_tree(
        checkpoint_root=Path(checkpoint_root),
        content_manifest=Path(content_manifest_path),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_file_count=expected_file_count,
    )
    packet = ValidatedCheckpointContentManifest(
        checkpoint_root=Path(receipt["checkpoint_root"]),
        content_manifest_path=Path(receipt["content_manifest_path"]),
        expected_manifest_sha256=receipt["content_manifest_file_sha256"],
        expected_file_count=expected_file_count,
        content_receipt=receipt,
    )
    object.__setattr__(packet, "_token", _CHECKPOINT_CONTENT_TOKEN)
    packet.assert_live()
    return packet


@dataclass(frozen=True)
class AuthenticatedSP4Collective:
    sp_rank: int
    global_ranks: tuple[int, int, int, int]
    group_contract_digest: str
    _parallel_state: Any = field(repr=False, compare=False)
    _group: Any = field(repr=False, compare=False)
    _dist: Any = field(repr=False, compare=False)
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def _validate_live(self, *, perform_consensus: bool) -> None:
        import torch.distributed as dist

        group_type = getattr(dist, "ProcessGroup", None)
        if group_type is None:
            group_type = getattr(
                getattr(dist, "distributed_c10d", None), "ProcessGroup", None
            )
        try:
            from bernini import parallel as bernini_parallel

            live_state = bernini_parallel.get_parallel_state()
        except Exception as error:
            raise NativeRV2VHiddenVJPError(
                "Bernini live parallel state is unavailable"
            ) from error
        if (
            self._token is not _SP4_COLLECTIVE_TOKEN
            or self._dist is not dist
            or not dist.is_available()
            or not dist.is_initialized()
            or live_state is not self._parallel_state
            or getattr(live_state, "ulysses_group", None) is not self._group
            or group_type is None
            or not isinstance(self._group, group_type)
            or getattr(live_state, "ulysses_size", None) != SP_SIZE
            or getattr(live_state, "ulysses_rank", None) != self.sp_rank
            or dist.get_world_size(self._group) != SP_SIZE
            or dist.get_rank(self._group) != self.sp_rank
            or str(dist.get_backend(self._group)).lower() != "nccl"
        ):
            raise NativeRV2VHiddenVJPError(
                "authenticated Ulysses-SP4 collective changed"
            )
        if perform_consensus:
            ranks: list[Any] = [None] * SP_SIZE
            dist.all_gather_object(ranks, int(dist.get_rank()), group=self._group)
            if tuple(ranks) != self.global_ranks:
                raise NativeRV2VHiddenVJPError(
                    "Ulysses-SP4 ordered global-rank membership changed"
                )
            peers: list[Any] = [None] * SP_SIZE
            dist.all_gather_object(
                peers, self.group_contract_digest, group=self._group
            )
            if peers != [self.group_contract_digest] * SP_SIZE:
                raise NativeRV2VHiddenVJPError(
                    "Ulysses-SP4 collective contract lacks peer consensus"
                )

    def assert_live(self) -> None:
        self._validate_live(perform_consensus=True)

    def receipt(self) -> Mapping[str, Any]:
        self.assert_live()
        value = {
            "schema_version": "bernini-qmosaic-authenticated-sp4-v1",
            "sp_rank": self.sp_rank,
            "sp_size": SP_SIZE,
            "ordered_global_ranks": list(self.global_ranks),
            "backend": "nccl",
            "ulysses_group_is_bernini_live_group": True,
            "group_contract_digest": self.group_contract_digest,
        }
        return {**value, "digest": object_sha256(value)}

    def all_reduce_sum(self, value: torch.Tensor) -> None:
        self.assert_live()
        self._dist.all_reduce(
            value, op=self._dist.ReduceOp.SUM, group=self._group
        )
        self.assert_live()

    def all_gather_object(self, value: Any) -> tuple[Any, Any, Any, Any]:
        self.assert_live()
        gathered: list[Any] = [None] * SP_SIZE
        self._dist.all_gather_object(gathered, value, group=self._group)
        self.assert_live()
        return tuple(gathered)  # type: ignore[return-value]


def authenticate_live_bernini_sp4_collective(
    *, parallel_state: Any
) -> AuthenticatedSP4Collective:
    """Mint a token only for the currently installed real Bernini group."""

    import torch.distributed as dist

    try:
        from bernini import parallel as bernini_parallel

        live_state = bernini_parallel.get_parallel_state()
    except Exception as error:
        raise NativeRV2VHiddenVJPError(
            "Bernini live parallel state is unavailable"
        ) from error
    group = getattr(live_state, "ulysses_group", None)
    group_type = getattr(dist, "ProcessGroup", None)
    if group_type is None:
        group_type = getattr(
            getattr(dist, "distributed_c10d", None), "ProcessGroup", None
        )
    if (
        parallel_state is not live_state
        or not dist.is_available()
        or not dist.is_initialized()
        or group_type is None
        or not isinstance(group, group_type)
        or getattr(live_state, "ulysses_size", None) != SP_SIZE
        or type(getattr(live_state, "ulysses_rank", None)) is not int
        or dist.get_world_size(group) != SP_SIZE
        or dist.get_rank(group) != live_state.ulysses_rank
        or str(dist.get_backend(group)).lower() != "nccl"
    ):
        raise NativeRV2VHiddenVJPError(
            "native runtime is not real Bernini Ulysses-SP4/NCCL"
        )
    ranks: list[Any] = [None] * SP_SIZE
    dist.all_gather_object(ranks, int(dist.get_rank()), group=group)
    if (
        any(type(rank) is not int for rank in ranks)
        or len(set(ranks)) != SP_SIZE
        or tuple(ranks) != tuple(range(ranks[0], ranks[0] + SP_SIZE))
        or ranks[0] % SP_SIZE != 0
    ):
        raise NativeRV2VHiddenVJPError(
            "native Ulysses group rank membership differs"
        )
    state_type = type(live_state)
    if not state_type.__module__.startswith("bernini.parallel"):
        raise NativeRV2VHiddenVJPError(
            "Bernini parallel state exact type differs"
        )
    group_digest = object_sha256(
        {
            "schema_version": "bernini-qmosaic-sp4-group-contract-v1",
            "parallel_state_type": (
                f"{state_type.__module__}.{state_type.__qualname__}"
            ),
            "ordered_global_ranks": ranks,
            "backend": "nccl",
            "sp_size": SP_SIZE,
        }
    )
    peers: list[Any] = [None] * SP_SIZE
    dist.all_gather_object(peers, group_digest, group=group)
    if peers != [group_digest] * SP_SIZE:
        raise NativeRV2VHiddenVJPError(
            "native Ulysses group contract differs by peer"
        )
    token = AuthenticatedSP4Collective(
        sp_rank=int(live_state.ulysses_rank),
        global_ranks=tuple(ranks),  # type: ignore[arg-type]
        group_contract_digest=group_digest,
        _parallel_state=live_state,
        _group=group,
        _dist=dist,
    )
    object.__setattr__(token, "_token", _SP4_COLLECTIVE_TOKEN)
    token.assert_live()
    return token


def _validate_cross_module_constants() -> None:
    if (
        native.LATENT_PHASES != LATENT_PHASES
        or native.REFERENCE_COUNT != 4
        or tuple(native.BRANCH_CONCAT_ORDER) != BRANCH_NAMES
        or native_bridge.LATENT_PHASES != LATENT_PHASES
        or native_bridge.PATCH_SIZE != (1, 2, 2)
        or pair_adapter.TOTAL_BLOCKS_1P3B != TOTAL_BLOCKS_1P3B
        or pair_adapter.ACTION_LORA_RANK != LORA_RANK
        or pair_adapter.ACTION_LORA_ALPHA != LORA_ALPHA
        or tuple(pair_adapter.NATIVE_BRANCHES) != BRANCH_NAMES
        or CANONICAL_B_PARAMETER_COUNT != 393216
        or len(CANONICAL_B_PARAMETER_NAMES) != 32
    ):
        raise RuntimeError("native Bernini/Q-MOSAIC pinned geometry differs")


_validate_cross_module_constants()


@dataclass(frozen=True)
class NativeTargetSuffixLayout:
    """One native condition-prefix/target-suffix contiguous SP shard.

    For local index ``j`` on rank ``r`` the official padded global token is

    ``g = r * ceil(total_tokens / sp_size) + j``.

    A real target token satisfies ``condition_tokens <= g < total_tokens``.
    Its target-flat index is ``u = g - condition_tokens``; exact81 phase and
    spatial-patch indices are ``u // patch_positions`` and
    ``u % patch_positions`` respectively.
    """

    branch_name: str
    patch_height: int
    patch_width: int
    patch_positions: int
    condition_tokens: int
    target_tokens: int
    total_tokens: int
    sp_rank: int
    sp_size: int
    local_length: int
    shard_global_start: int
    shard_global_stop_padded: int
    local_target_indices: torch.Tensor = field(repr=False)
    global_target_indices: torch.Tensor = field(repr=False)
    target_flat_indices: torch.Tensor = field(repr=False)
    target_phase_indices: torch.Tensor = field(repr=False)
    target_patch_indices: torch.Tensor = field(repr=False)
    phase_patch_count: torch.Tensor = field(repr=False)
    condition_rows_excluded: int
    padding_rows_excluded: int

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "branch_name": self.branch_name,
            "native_concat_order": list(native.BRANCH_CONCAT_ORDER[self.branch_name]),
            "patch_grid_height_width": [self.patch_height, self.patch_width],
            "patch_positions": self.patch_positions,
            "condition_phases": CONDITION_PHASES_BY_BRANCH[self.branch_name],
            "condition_tokens": self.condition_tokens,
            "target_phases": LATENT_PHASES,
            "target_tokens": self.target_tokens,
            "total_tokens": self.total_tokens,
            "sp_rank": self.sp_rank,
            "sp_size": self.sp_size,
            "local_length_ceil": self.local_length,
            "shard_global_start": self.shard_global_start,
            "shard_global_stop_padded": self.shard_global_stop_padded,
            "selected_target_tokens": int(self.local_target_indices.numel()),
            "condition_rows_excluded": self.condition_rows_excluded,
            "padding_rows_excluded": self.padding_rows_excluded,
            "phase_patch_count": [int(row) for row in self.phase_patch_count.tolist()],
            "global_index_formula": (
                "g=sp_rank*ceil(total_tokens/sp_size)+local_index;"
                "target iff condition_tokens<=g<total_tokens;"
                "u=g-condition_tokens;phase=u//patch_positions;patch=u%patch_positions"
            ),
            "global_order": "condition_prefix_then_phase-major_target_patch-y-x",
            "padding_policy": "append_then_contiguous-rank-chunk",
        }
        return {**value, "digest": object_sha256(value)}


def build_native_target_suffix_layout(
    *,
    branch_name: str,
    patch_height: int,
    patch_width: int,
    sp_rank: int,
    sp_size: int = SP_SIZE,
    observed_condition_tokens: Optional[int] = None,
    observed_total_tokens: Optional[int] = None,
) -> NativeTargetSuffixLayout:
    """Build and, when supplied, authenticate one native branch geometry."""

    if branch_name not in BRANCH_NAMES:
        raise NativeRV2VHiddenVJPError("branch is not native none/V/I/VI")
    if (
        type(patch_height) is not int
        or type(patch_width) is not int
        or patch_height <= 0
        or patch_width <= 0
    ):
        raise NativeRV2VHiddenVJPError("patch grid must contain positive integers")
    if type(sp_size) is not int or sp_size not in (1, SP_SIZE):
        raise NativeRV2VHiddenVJPError("only SP1 tests and production SP4 are supported")
    if type(sp_rank) is not int or not 0 <= sp_rank < sp_size:
        raise NativeRV2VHiddenVJPError("SP rank lies outside its group")
    patch_positions = patch_height * patch_width
    condition_tokens = CONDITION_PHASES_BY_BRANCH[branch_name] * patch_positions
    target_tokens = LATENT_PHASES * patch_positions
    total_tokens = condition_tokens + target_tokens
    if (
        observed_condition_tokens is not None
        and observed_condition_tokens != condition_tokens
    ):
        raise NativeRV2VHiddenVJPError("native condition-prefix token count differs")
    if observed_total_tokens is not None and observed_total_tokens != total_tokens:
        raise NativeRV2VHiddenVJPError("native total token count differs")

    local_length = math.ceil(total_tokens / sp_size)
    start = sp_rank * local_length
    stop = start + local_length
    local = torch.arange(local_length, dtype=torch.int64)
    global_index = start + local
    real = global_index < total_tokens
    target = real & (global_index >= condition_tokens)
    selected_local = local[target].contiguous()
    selected_global = global_index[target].contiguous()
    target_flat = (selected_global - condition_tokens).contiguous()
    phases = torch.div(target_flat, patch_positions, rounding_mode="floor").contiguous()
    patches = torch.remainder(target_flat, patch_positions).contiguous()
    counts = torch.bincount(phases, minlength=LATENT_PHASES).to(torch.int64)
    condition_excluded = int((real & (global_index < condition_tokens)).sum().item())
    padding_excluded = int((~real).sum().item())
    if (
        int(counts.sum().item()) != int(selected_local.numel())
        or (target_flat.numel() and int(target_flat.min().item()) < 0)
        or (
            target_flat.numel()
            and int(target_flat.max().item()) >= target_tokens
        )
        or (phases.numel() and int(phases.max().item()) >= LATENT_PHASES)
        or (patches.numel() and int(patches.max().item()) >= patch_positions)
        or condition_excluded + int(selected_local.numel()) + padding_excluded
        != local_length
    ):
        raise NativeRV2VHiddenVJPError("target-suffix global-index closure differs")
    return NativeTargetSuffixLayout(
        branch_name=branch_name,
        patch_height=patch_height,
        patch_width=patch_width,
        patch_positions=patch_positions,
        condition_tokens=condition_tokens,
        target_tokens=target_tokens,
        total_tokens=total_tokens,
        sp_rank=sp_rank,
        sp_size=sp_size,
        local_length=local_length,
        shard_global_start=start,
        shard_global_stop_padded=stop,
        local_target_indices=selected_local,
        global_target_indices=selected_global,
        target_flat_indices=target_flat,
        target_phase_indices=phases,
        target_patch_indices=patches,
        phase_patch_count=counts,
        condition_rows_excluded=condition_excluded,
        padding_rows_excluded=padding_excluded,
    )


def layout_from_native_branch(
    branch: native.NativeRV2VBranch,
    *,
    patch_height: int,
    patch_width: int,
    sp_rank: int,
    sp_size: int = SP_SIZE,
) -> NativeTargetSuffixLayout:
    if not isinstance(branch, native.NativeRV2VBranch):
        raise NativeRV2VHiddenVJPError("layout requires a native RV2V branch")
    layout = build_native_target_suffix_layout(
        branch_name=branch.name,
        patch_height=patch_height,
        patch_width=patch_width,
        sp_rank=sp_rank,
        sp_size=sp_size,
        observed_condition_tokens=branch.condition_tokens,
        observed_total_tokens=branch.total_tokens,
    )
    expected_mask = torch.zeros(
        branch.total_tokens, dtype=torch.bool, device=branch.target_mask.device
    )
    expected_mask[branch.condition_tokens :] = True
    if not torch.equal(branch.target_mask, expected_mask):
        raise NativeRV2VHiddenVJPError("native branch target mask is not the suffix")
    return layout


def make_fixed_spatial_sketch(
    patch_positions: int,
    *,
    coordinates: int = SPATIAL_SKETCH_COORDINATES,
    seed: int = SPATIAL_SKETCH_SEED,
    device: Any = None,
) -> torch.Tensor:
    """Construct a content-independent FP32 Rademacher spatial sketch."""

    if (
        type(patch_positions) is not int
        or patch_positions < coordinates
        or type(coordinates) is not int
        or coordinates <= 1
        or type(seed) is not int
        or seed < 0
    ):
        raise NativeRV2VHiddenVJPError("fixed spatial sketch dimensions/seed differ")
    scale = 1.0 / math.sqrt(float(patch_positions))
    result = torch.empty(coordinates, patch_positions, dtype=torch.float32)
    for row in range(coordinates):
        for column in range(patch_positions):
            token = f"{seed}:{row}:{column}".encode("ascii")
            result[row, column] = (
                scale if hashlib.sha256(token).digest()[0] & 1 else -scale
            )
    if (
        int(torch.linalg.matrix_rank(result).item()) != coordinates
        or not bool(torch.isfinite(result).all().item())
    ):
        raise NativeRV2VHiddenVJPError("fixed spatial sketch is rank deficient")
    return result.to(device=device) if device is not None else result


@dataclass(frozen=True)
class LocalBlock15TargetSketch:
    role: str
    layout: NativeTargetSuffixLayout
    tensor: torch.Tensor = field(repr=False)
    graph_connected: bool
    tensor_digest: str

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "role": self.role,
            "branch_name": self.layout.branch_name,
            "layout_digest": self.layout.receipt()["digest"],
            "shape": list(map(int, self.tensor.shape)),
            "dtype": str(self.tensor.dtype),
            "graph_connected": self.graph_connected,
            "tensor_sha256": self.tensor_digest,
            "condition_and_padding_rows_excluded": True,
        }
        return {**value, "digest": object_sha256(value)}


def sketch_local_block15_target_suffix(
    hidden: torch.Tensor,
    *,
    layout: NativeTargetSuffixLayout,
    spatial_sketch: torch.Tensor,
    role: str,
    detach: bool,
) -> LocalBlock15TargetSketch:
    """Select native target rows and sketch them without repacking the prefix."""

    if not isinstance(layout, NativeTargetSuffixLayout):
        raise NativeRV2VHiddenVJPError("block15 sketch layout differs")
    if not isinstance(role, str) or not role:
        raise NativeRV2VHiddenVJPError("block15 sketch role must be nonempty text")
    if type(detach) is not bool:
        raise NativeRV2VHiddenVJPError("block15 detach flag must be boolean")
    if (
        not isinstance(hidden, torch.Tensor)
        or tuple(map(int, hidden.shape))
        != (1, layout.local_length, HIDDEN_SIZE)
        or not hidden.is_floating_point()
        or hidden.device.type == "meta"
        or not bool(torch.isfinite(hidden).all().item())
    ):
        raise NativeRV2VHiddenVJPError(
            "block15 hidden must be finite [1,local_length,1536]"
        )
    if (
        not isinstance(spatial_sketch, torch.Tensor)
        or spatial_sketch.dtype != torch.float32
        or spatial_sketch.ndim != 2
        or tuple(map(int, spatial_sketch.shape))
        != (SPATIAL_SKETCH_COORDINATES, layout.patch_positions)
        or spatial_sketch.requires_grad
        or spatial_sketch.grad_fn is not None
        or not bool(torch.isfinite(spatial_sketch).all().item())
    ):
        raise NativeRV2VHiddenVJPError("fixed spatial sketch tensor differs")

    source = hidden.detach() if detach else hidden
    local_indices = layout.local_target_indices.to(device=hidden.device)
    phases = layout.target_phase_indices.to(device=hidden.device)
    patches = layout.target_patch_indices.to(device=hidden.device)
    values = source[0].index_select(0, local_indices).float()
    weights = spatial_sketch.to(device=hidden.device)
    result = torch.zeros(
        LATENT_PHASES,
        SPATIAL_SKETCH_COORDINATES,
        HIDDEN_SIZE,
        dtype=torch.float32,
        device=hidden.device,
    )
    # Match the established Bernini materializer's deterministic FP32
    # index_add order: sketch-coordinate outer loop, phase accumulation inner.
    for coordinate in range(SPATIAL_SKETCH_COORDINATES):
        scaled = values * weights[coordinate].index_select(0, patches).unsqueeze(1)
        result[:, coordinate, :].index_add_(0, phases, scaled)
    result = result.unsqueeze(0).contiguous()
    if detach:
        result = result.detach().contiguous()
    graph_connected = bool(result.requires_grad and result.grad_fn is not None)
    if (
        tuple(map(int, result.shape))
        != (1, LATENT_PHASES, SPATIAL_SKETCH_COORDINATES, HIDDEN_SIZE)
        or result.dtype != torch.float32
        or not bool(torch.isfinite(result).all().item())
        or graph_connected == detach
    ):
        raise NativeRV2VHiddenVJPError("block15 target sketch graph/value differs")
    return LocalBlock15TargetSketch(
        role=role,
        layout=layout,
        tensor=result,
        graph_connected=graph_connected,
        tensor_digest=tensor_sha256(result, label=f"{role} local block15 sketch"),
    )


class Block15TargetSuffixObserver:
    """Narrow one-call hook for fake or real Bernini block 15."""

    def __init__(self, transformer: nn.Module, *, spatial_sketch: torch.Tensor):
        blocks = tuple(getattr(transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B or not callable(
            getattr(blocks[HOOK_BLOCK_INDEX], "register_forward_hook", None)
        ):
            raise NativeRV2VHiddenVJPError("Bernini block15 hook structure differs")
        self.transformer = transformer
        self.block = blocks[HOOK_BLOCK_INDEX]
        self.spatial_sketch = spatial_sketch
        self._handle: Any = None
        self._pending: Optional[tuple[str, NativeTargetSuffixLayout, bool]] = None
        self._capture: Optional[LocalBlock15TargetSketch] = None
        self._calls = 0

    def install(self) -> None:
        if self._handle is not None:
            raise NativeRV2VHiddenVJPError("block15 observer is already installed")
        self._handle = self.block.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._pending is not None:
            raise NativeRV2VHiddenVJPError("cannot remove an active block15 capture")
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        if self._pending is None or self._capture is not None:
            raise NativeRV2VHiddenVJPError("unexpected or repeated block15 hook call")
        role, layout, detach = self._pending
        self._capture = sketch_local_block15_target_suffix(
            output,
            layout=layout,
            spatial_sketch=self.spatial_sketch,
            role=role,
            detach=detach,
        )
        self._calls += 1

    @contextmanager
    def capture(
        self, *, role: str, layout: NativeTargetSuffixLayout, detach: bool
    ) -> Iterator[list[LocalBlock15TargetSketch]]:
        if self._handle is None:
            raise NativeRV2VHiddenVJPError("install the block15 observer first")
        if self._pending is not None:
            raise NativeRV2VHiddenVJPError("nested block15 captures are forbidden")
        self._pending = (role, layout, detach)
        self._capture = None
        before = self._calls
        holder: list[LocalBlock15TargetSketch] = []
        try:
            yield holder
            if self._calls != before + 1 or self._capture is None:
                raise NativeRV2VHiddenVJPError("block15 capture did not fire exactly once")
            holder.append(self._capture)
        finally:
            self._pending = None
            self._capture = None


def sum_detached_sp4_sketches(
    captures: Sequence[LocalBlock15TargetSketch],
) -> torch.Tensor:
    """Assemble one detached global target sketch by exact SP4 SUM."""

    if len(captures) != SP_SIZE:
        raise NativeRV2VHiddenVJPError("detached measurement requires four SP captures")
    ordered = sorted(captures, key=lambda row: row.layout.sp_rank)
    first = ordered[0]
    if [row.layout.sp_rank for row in ordered] != list(range(SP_SIZE)):
        raise NativeRV2VHiddenVJPError("detached SP4 capture rank order differs")
    if any(
        row.graph_connected
        or row.role != first.role
        or row.layout.branch_name != first.layout.branch_name
        or row.layout.patch_positions != first.layout.patch_positions
        or row.layout.sp_size != SP_SIZE
        or row.tensor.shape != first.tensor.shape
        or row.tensor.dtype != torch.float32
        or row.tensor.device != first.tensor.device
        for row in ordered
    ):
        raise NativeRV2VHiddenVJPError("detached SP4 sketch closure differs")
    counts = torch.stack([row.layout.phase_patch_count for row in ordered]).sum(dim=0)
    if not torch.equal(
        counts,
        torch.full((LATENT_PHASES,), first.layout.patch_positions, dtype=torch.int64),
    ):
        raise NativeRV2VHiddenVJPError("SP4 target suffix does not cover every patch once")
    result = torch.stack([row.tensor for row in ordered], dim=0).sum(dim=0).detach()
    if result.requires_grad or result.grad_fn is not None:
        raise NativeRV2VHiddenVJPError("detached global measurement retained a graph")
    return result.contiguous()


def _fixed_a_cpu(parameter_name: str, shape: tuple[int, int]) -> torch.Tensor:
    material = f"{FIXED_LORA_A_SEED}\0{parameter_name}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bound = 1.0 / math.sqrt(float(shape[1]))
    value = torch.empty(shape, dtype=torch.float32).uniform_(
        -bound, bound, generator=generator
    )
    if int(torch.linalg.matrix_rank(value).item()) != shape[0]:
        raise NativeRV2VHiddenVJPError("fixed LoRA-A is not full row rank")
    return value


class _ExactZeroSelectedResidual(torch.autograd.Function):
    """Return exact base bytes while retaining the zero-point B Jacobian.

    Evaluating ``base + delta`` is not byte preserving when both operands are
    numerical zero: IEEE-754 signed ``-0`` base values may become ``+0``.  At
    the Q-MOSAIC fixed gauge, ``delta`` is exactly zero but its derivative with
    respect to LoRA-B is the coordinate we need.  This autograd edge therefore
    copies the base value in forward and routes the selected output cotangent
    to the already-built ``delta`` graph in backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        context: Any,
        base: torch.Tensor,
        selected_delta: torch.Tensor,
        selector: torch.Tensor,
    ) -> torch.Tensor:
        context.save_for_backward(selector)
        return base.clone()

    @staticmethod
    def backward(  # type: ignore[override]
        context: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        (selector,) = context.saved_tensors
        return gradient, gradient[:, selector, :].contiguous(), None


def _raw_nonzero_byte_count(value: torch.Tensor, *, label: str) -> int:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or not value.is_contiguous()
    ):
        raise NativeRV2VHiddenVJPError(
            f"{label} must be a contiguous live tensor for raw-byte audit"
        )
    raw = value.detach().view(torch.uint8).reshape(-1)
    return int(torch.count_nonzero(raw).item())


def _raw_byte_mismatch_count(
    left: torch.Tensor, right: torch.Tensor, *, label: str
) -> int:
    """Compare exact bytes on-device in bounded chunks without a host copy."""

    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.device.type == "meta"
        or right.device != left.device
        or right.dtype != left.dtype
        or tuple(right.shape) != tuple(left.shape)
        or not left.is_contiguous()
        or not right.is_contiguous()
    ):
        raise NativeRV2VHiddenVJPError(
            f"{label} tensors differ before raw-byte comparison"
        )
    left_bytes = left.detach().view(torch.uint8).reshape(-1)
    right_bytes = right.detach().view(torch.uint8).reshape(-1)
    chunk_bytes = 4 * 1024 * 1024
    mismatches = 0
    for start in range(0, int(left_bytes.numel()), chunk_bytes):
        stop = min(start + chunk_bytes, int(left_bytes.numel()))
        mismatches += int(
            torch.count_nonzero(
                left_bytes[start:stop] != right_bytes[start:stop]
            ).item()
        )
    return mismatches


@dataclass
class Core16ZeroRouteProofHolder:
    """One-shot holder populated after a complete zero-route proof scope."""

    receipt: Optional[dict[str, Any]] = None

    def require_receipt(self) -> dict[str, Any]:
        if self.receipt is None:
            raise NativeRV2VHiddenVJPError(
                "core16 zero-route proof receipt is not complete"
            )
        return dict(self.receipt)


@dataclass
class _Core16ZeroRouteProofRecorder:
    handle: Any = field(repr=False)
    role: str
    sp_rank: int
    b_state_before_sha256: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    seen_names: set[str] = field(default_factory=set)

    def record(
        self,
        *,
        wrapper: "Core16ExactZeroTargetRowActionLoRA",
        route: pair_adapter.PairV5ActionRoute,
        selector: torch.Tensor,
        selected_delta: torch.Tensor,
        base: torch.Tensor,
        result: torch.Tensor,
    ) -> None:
        name = wrapper.canonical_b_name
        if name in self.seen_names:
            raise NativeRV2VHiddenVJPError(
                f"core16 zero-route wrapper repeated: {name}"
            )
        position = len(self.rows)
        if (
            position >= len(CANONICAL_B_PARAMETER_NAMES)
            or name != CANONICAL_B_PARAMETER_NAMES[position]
            or self.handle.wrappers_by_b_name.get(name) is not wrapper
        ):
            raise NativeRV2VHiddenVJPError(
                "core16 zero-route wrapper order/identity differs"
            )
        if (
            not route.adapter_active
            or route.branch_name != "VI"
            or route.sequence_parallel_size != SP_SIZE
            or route.sequence_parallel_rank != self.sp_rank
            or route.sigma_schedule_index != NATIVE_SCHEDULE_INDEX
            or route.gate_name != "mid"
            or route.gate_weight != 0.5
            or not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or not selected_delta.requires_grad
            or selected_delta.grad_fn is None
            or not result.requires_grad
            or result.grad_fn is None
        ):
            raise NativeRV2VHiddenVJPError(
                "core16 zero-route proof route differs"
            )
        expected_selector = route.local_target_selector(device=selector.device)
        if (
            selector.dtype != torch.bool
            or selector.ndim != 1
            or not torch.equal(selector, expected_selector)
            or int(base.shape[1]) != int(selector.numel())
            or tuple(result.shape) != tuple(base.shape)
            or result.dtype != base.dtype
            or result.device != base.device
            or tuple(selected_delta.shape)
            != (
                int(base.shape[0]),
                int(selector.sum().item()),
                int(base.shape[2]),
            )
            or selected_delta.dtype != base.dtype
            or selected_delta.device != base.device
        ):
            raise NativeRV2VHiddenVJPError(
                "core16 zero-route selector/tensor closure differs"
            )
        b_raw_nonzero_byte_count = _raw_nonzero_byte_count(
            wrapper.action_lora_b.weight, label=f"zero-route B {name}"
        )
        selected_delta_nonzero_element_count = int(
            torch.count_nonzero(selected_delta.detach()).item()
        )
        if (
            b_raw_nonzero_byte_count
            or not bool(torch.isfinite(selected_delta).all().item())
            or selected_delta_nonzero_element_count
        ):
            raise NativeRV2VHiddenVJPError(
                "core16 zero-route selected delta is not exact zero"
            )
        mismatch_count = _raw_byte_mismatch_count(
            base, result, label=f"zero-route base/result {name}"
        )
        if mismatch_count:
            raise NativeRV2VHiddenVJPError(
                "core16 zero-route forward changed base bytes"
            )
        selector_sha256 = tensor_sha256(
            selector.to(dtype=torch.uint8), label=f"zero-route selector {name}"
        )
        self.seen_names.add(name)
        self.rows.append(
            {
                "canonical_b_name": name,
                "local_row_count": int(selector.numel()),
                "selected_row_count": int(selector.sum().item()),
                "selector_sha256": selector_sha256,
                "selector_exact_expected": True,
                "b_raw_nonzero_byte_count": b_raw_nonzero_byte_count,
                "selected_delta_nonzero_element_count": (
                    selected_delta_nonzero_element_count
                ),
                "base_result_raw_byte_mismatch_count": mismatch_count,
                "output_dtype": str(result.dtype),
                "output_shape": list(map(int, result.shape)),
                "selected_delta_numerically_exact_zero": True,
                "base_result_raw_bytes_equal": True,
                "autograd_enabled": True,
                "inference_mode_enabled": False,
            }
        )

    def finalize(self) -> dict[str, Any]:
        if (
            tuple(row["canonical_b_name"] for row in self.rows)
            != CANONICAL_B_PARAMETER_NAMES
            or self.seen_names != set(CANONICAL_B_PARAMETER_NAMES)
        ):
            raise NativeRV2VHiddenVJPError(
                "core16 zero-route proof has missing or reordered wrappers"
            )
        if pair_adapter.active_route() is not None:
            raise NativeRV2VHiddenVJPError(
                "core16 zero-route proof ended inside an active route"
            )
        self.handle.assert_fixed_gauge()
        after = self.handle.b_parameter_state_sha256()
        if after != self.b_state_before_sha256:
            raise NativeRV2VHiddenVJPError(
                "core16 zero-route proof changed LoRA-B bytes"
            )
        unsigned = {
            "schema_version": ZERO_ROUTE_PROOF_SCHEMA_VERSION,
            "role": self.role,
            "sp_rank": self.sp_rank,
            "sp_size": SP_SIZE,
            "branch_name": "VI",
            "native_schedule_index": NATIVE_SCHEDULE_INDEX,
            "native_timestep": NATIVE_TIMESTEP,
            "sigma_gate": "mid",
            "sigma_gate_weight": 0.5,
            "grad_enabled": True,
            "inference_mode_enabled": False,
            "wrapper_count": len(self.rows),
            "canonical_wrapper_order_sha256": object_sha256(
                list(CANONICAL_B_PARAMETER_NAMES)
            ),
            "call_evidence": [dict(row) for row in self.rows],
            "call_evidence_sha256": object_sha256(self.rows),
            "b_state_before_sha256": self.b_state_before_sha256,
            "b_state_after_sha256": after,
            "total_local_row_count": sum(
                int(row["local_row_count"]) for row in self.rows
            ),
            "total_selected_row_count": sum(
                int(row["selected_row_count"]) for row in self.rows
            ),
            "missing_wrapper_count": 0,
            "repeated_wrapper_count": 0,
            "all_selected_deltas_numerically_exact_zero": True,
            "all_base_result_raw_bytes_equal": True,
            "b_unchanged": True,
        }
        return {**unsigned, "digest": object_sha256(unsigned)}


_ACTIVE_CORE16_ZERO_ROUTE_PROOF: ContextVar[
    Optional[_Core16ZeroRouteProofRecorder]
] = ContextVar("bernini_qmosaic_core16_zero_route_proof", default=None)


class Core16ExactZeroTargetRowActionLoRA(
    pair_adapter.PairV5TargetRowActionLoRA
):
    """PAIR-v5 row routing with a byte-exact fixed-gauge zero point."""

    def __init__(
        self, base: nn.Module, *, projection: str, canonical_b_name: str
    ) -> None:
        if canonical_b_name not in CANONICAL_B_PARAMETER_NAMES:
            raise NativeRV2VHiddenVJPError(
                "core16 wrapper canonical B name differs"
            )
        super().__init__(base, projection=projection)
        self.canonical_b_name = canonical_b_name

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden_states)
        route = pair_adapter.active_route()
        if route is None:
            return base
        selector = self._selector(hidden_states, route)
        if not route.adapter_active:
            return base
        selected_delta = self._selected_delta(
            hidden_states, selector, route.gate_weight
        ).to(base.dtype)
        b_is_raw_positive_zero = not _raw_nonzero_byte_count(
            self.action_lora_b.weight,
            label=f"core16 wrapper B {self.canonical_b_name}",
        )
        recorder = _ACTIVE_CORE16_ZERO_ROUTE_PROOF.get()
        if not b_is_raw_positive_zero:
            if recorder is not None:
                raise NativeRV2VHiddenVJPError(
                    "core16 zero-route proof observed nonzero LoRA-B"
                )
            # Outside the Q-MOSAIC fixed-zero gauge, preserve the established
            # PAIR-v5 residual semantics instead of silently discarding it.
            result = base.clone()
            result[:, selector, :] = base[:, selector, :] + selected_delta
            return result
        if (
            not bool(torch.isfinite(selected_delta).all().item())
            or bool(torch.count_nonzero(selected_delta.detach()).item())
        ):
            raise NativeRV2VHiddenVJPError(
                "zero LoRA-B did not produce an exact-zero selected delta"
            )
        if tuple(selected_delta.shape) != (
            int(base.shape[0]),
            int(selector.sum().item()),
            int(base.shape[2]),
        ):
            raise NativeRV2VHiddenVJPError(
                "zero LoRA-B selected delta shape differs"
            )
        result = _ExactZeroSelectedResidual.apply(base, selected_delta, selector)
        if recorder is not None:
            recorder.record(
                wrapper=self,
                route=route,
                selector=selector,
                selected_delta=selected_delta,
                base=base,
                result=result,
            )
        return result


@dataclass(frozen=True)
class _AdapterRuntimeSnapshot:
    rows: tuple[
        tuple[
            str,
            nn.Parameter,
            int,
            torch.Tensor,
            bool,
            Optional[torch.Tensor],
        ],
        ...,
    ]
    active_route: Any = field(repr=False)


@dataclass
class Core16ActionLoRAHandle:
    transformer: nn.Module
    wrappers_by_b_name: Mapping[str, pair_adapter.PairV5TargetRowActionLoRA]
    original_q: tuple[tuple[int, nn.Module], ...]
    original_o: tuple[tuple[int, nn.Module], ...]
    original_late_qo_ids: tuple[tuple[int, int, int], ...]
    original_patch_embedding_id: int
    gauge_receipt: Mapping[str, Any]
    restored: bool = False

    def canonical_b_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise NativeRV2VHiddenVJPError("core16 Action-LoRA has been restored")
        return tuple(
            (name, self.wrappers_by_b_name[name].action_lora_b.weight)
            for name in CANONICAL_B_PARAMETER_NAMES
        )

    def canonical_a_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise NativeRV2VHiddenVJPError("core16 Action-LoRA has been restored")
        return tuple(
            (
                b_name.replace("action_lora_b.weight", "action_lora_a.weight"),
                self.wrappers_by_b_name[b_name].action_lora_a.weight,
            )
            for b_name in CANONICAL_B_PARAMETER_NAMES
        )

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """Compatibility view for the Phase-A gauge audit.

        The returned order is explicit per-block Q-A/Q-B/O-A/O-B, never the
        legacy all-Q/all-O order.  A is present for gauge authentication even
        though it is frozen; B is the only trainable subset.
        """

        a = dict(self.canonical_a_named_parameters())
        b = dict(self.canonical_b_named_parameters())
        rows: list[tuple[str, nn.Parameter]] = []
        for b_name in CANONICAL_B_PARAMETER_NAMES:
            a_name = b_name.replace("action_lora_b.weight", "action_lora_a.weight")
            rows.extend(((a_name, a[a_name]), (b_name, b[b_name])))
        return tuple(rows)

    def state_digest(self) -> str:
        return _named_tensor_sha256(
            self.trainable_named_parameters(), label="core16 Action-LoRA state"
        )

    def b_parameter_state_sha256(self) -> str:
        """QP-compatible hash of the canonical current B coordinate only."""

        return _named_tensor_sha256(
            self.canonical_b_named_parameters(), label="core16 LoRA-B state"
        )

    @contextmanager
    def capture_zero_route_proof(
        self, *, role: str, sp_rank: int
    ) -> Iterator[Core16ZeroRouteProofHolder]:
        """Prove one enabled zero-B forward closes all 32 wrappers exactly once.

        The scope must surround a complete native transformer forward.  Its
        receipt is populated only after canonical call order, selector closure,
        exact base/output bytes, and unchanged B bytes all pass.
        """

        if (
            self.restored
            or role not in ("action", "noop")
            or type(sp_rank) is not int
            or not 0 <= sp_rank < SP_SIZE
            or pair_adapter.active_route() is not None
            or _ACTIVE_CORE16_ZERO_ROUTE_PROOF.get() is not None
            or not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
        ):
            raise NativeRV2VHiddenVJPError(
                "core16 zero-route proof scope differs"
            )
        self.assert_fixed_gauge()
        recorder = _Core16ZeroRouteProofRecorder(
            handle=self,
            role=role,
            sp_rank=sp_rank,
            b_state_before_sha256=self.b_parameter_state_sha256(),
        )
        holder = Core16ZeroRouteProofHolder()
        token = _ACTIVE_CORE16_ZERO_ROUTE_PROOF.set(recorder)
        try:
            yield holder
        finally:
            _ACTIVE_CORE16_ZERO_ROUTE_PROOF.reset(token)
        holder.receipt = recorder.finalize()

    def adapter_state_snapshot(self) -> _AdapterRuntimeSnapshot:
        if pair_adapter.active_route() is not None:
            raise NativeRV2VHiddenVJPError(
                "cannot snapshot adapters inside an active route"
            )
        return _AdapterRuntimeSnapshot(
            rows=tuple(
                (
                    name,
                    parameter,
                    id(parameter),
                    parameter.detach().float().cpu().contiguous().clone(),
                    bool(parameter.requires_grad),
                    None
                    if parameter.grad is None
                    else parameter.grad.detach().cpu().contiguous().clone(),
                )
                for name, parameter in self.trainable_named_parameters()
            ),
            active_route=pair_adapter.active_route(),
        )

    def adapter_state_matches(
        self,
        snapshot: _AdapterRuntimeSnapshot,
        *,
        allow_b_temporarily_frozen: bool = False,
    ) -> bool:
        if not isinstance(snapshot, _AdapterRuntimeSnapshot):
            return False
        current = self.trainable_named_parameters()
        if len(current) != len(snapshot.rows):
            return False
        for (name, parameter), row in zip(current, snapshot.rows):
            (
                snapshot_name,
                snapshot_parameter,
                parameter_id,
                value,
                requires_grad,
                gradient,
            ) = row
            permitted = requires_grad
            if (
                allow_b_temporarily_frozen
                and name in CANONICAL_B_PARAMETER_NAMES
                and requires_grad
            ):
                permitted = False
            if (
                name != snapshot_name
                or parameter is not snapshot_parameter
                or id(parameter) != parameter_id
                or bool(parameter.requires_grad) != permitted
                or not torch.equal(parameter.detach().float().cpu(), value)
                or (parameter.grad is None) != (gradient is None)
                or (
                    gradient is not None
                    and not torch.equal(parameter.grad.detach().cpu(), gradient)
                )
            ):
                return False
        return pair_adapter.active_route() is snapshot.active_route

    def restore_adapter_state(self, snapshot: _AdapterRuntimeSnapshot) -> None:
        """Restore A/B only; this function never rewrites the base model graph."""

        if not isinstance(snapshot, _AdapterRuntimeSnapshot):
            raise NativeRV2VHiddenVJPError("adapter rollback snapshot differs")
        current = self.trainable_named_parameters()
        if len(current) != len(snapshot.rows):
            raise NativeRuntimeSealChangedError(
                "adapter parameter topology changed; process is poisoned"
            )
        for (name, parameter), row in zip(current, snapshot.rows):
            snapshot_name, snapshot_parameter, parameter_id, value, _, _ = row
            if (
                name != snapshot_name
                or parameter is not snapshot_parameter
                or id(parameter) != parameter_id
            ):
                raise NativeRuntimeSealChangedError(
                    "adapter parameter identity changed; process is poisoned"
                )
            if not torch.equal(parameter.detach().float().cpu(), value):
                with torch.no_grad():
                    parameter.copy_(value.to(device=parameter.device))
        for (name, parameter), row in zip(current, snapshot.rows):
            _, _, _, _, requires_grad, gradient = row
            parameter.requires_grad_(requires_grad)
            parameter.grad = (
                None
                if gradient is None
                else gradient.to(
                    device=parameter.device, dtype=parameter.dtype
                ).clone()
            )
        if pair_adapter.active_route() is not snapshot.active_route:
            route_context = getattr(pair_adapter, "_ACTIVE_ROUTE", None)
            setter = getattr(route_context, "set", None)
            if not callable(setter):
                raise NativeRV2VHiddenVJPError(
                    "adapter route cannot be restored"
                )
            setter(snapshot.active_route)
        if not self.adapter_state_matches(snapshot):
            raise NativeRV2VHiddenVJPError("adapter-only rollback audit failed")

    def assert_fixed_gauge(self) -> None:
        a = self.canonical_a_named_parameters()
        b = self.canonical_b_named_parameters()
        trainable_ids = {id(parameter) for _, parameter in b}
        observed_ids = {
            id(parameter)
            for parameter in self.transformer.parameters()
            if parameter.requires_grad
        }
        if (
            tuple(name for name, _ in a) != CANONICAL_A_PARAMETER_NAMES
            or tuple(name for name, _ in b) != CANONICAL_B_PARAMETER_NAMES
            or len({id(parameter) for _, parameter in (*a, *b)}) != 64
            or any(
                type(self.wrappers_by_b_name.get(name))
                is not Core16ExactZeroTargetRowActionLoRA
                or self.wrappers_by_b_name[name].canonical_b_name != name
                for name in CANONICAL_B_PARAMETER_NAMES
            )
            or any(
                parameter.dtype != torch.float32
                or tuple(map(int, parameter.shape)) != CANONICAL_A_SHAPE
                or parameter.requires_grad
                or parameter.grad is not None
                or not bool(torch.isfinite(parameter).all().item())
                for _, parameter in a
            )
            or any(
                parameter.dtype != torch.float32
                or tuple(map(int, parameter.shape)) != CANONICAL_B_SHAPE
                or not parameter.requires_grad
                or parameter.grad is not None
                or _raw_nonzero_byte_count(
                    parameter, label="core16 fixed-gauge LoRA-B"
                )
                for _, parameter in b
            )
            or observed_ids != trainable_ids
            or sum(parameter.numel() for _, parameter in b)
            != CANONICAL_B_PARAMETER_COUNT
        ):
            raise NativeRV2VHiddenVJPError("core16 fixed-A/B-only gauge differs")
        for name, parameter in a:
            expected = _fixed_a_cpu(name, CANONICAL_A_SHAPE).to(parameter.device)
            if not torch.equal(parameter.detach(), expected):
                raise NativeRV2VHiddenVJPError("fixed LoRA-A seed/value differs")

    @contextmanager
    def route(
        self,
        branch: native.NativeRV2VBranch,
        *,
        sp_rank: int,
        adapter_enabled: bool,
    ) -> Iterator[None]:
        if self.restored:
            raise NativeRV2VHiddenVJPError("cannot route a restored Action-LoRA")
        if not isinstance(branch, native.NativeRV2VBranch):
            raise NativeRV2VHiddenVJPError("Action-LoRA route requires a native branch")
        route = pair_adapter.PairV5ActionRoute(
            total_tokens=branch.total_tokens,
            condition_tokens=branch.condition_tokens,
            sequence_parallel_rank=sp_rank,
            sequence_parallel_size=SP_SIZE,
            branch_name=branch.name,
            sigma_schedule_index=NATIVE_SCHEDULE_INDEX,
            enabled=adapter_enabled,
        )
        with pair_adapter.activate_route(route):
            yield

    @contextmanager
    def frozen_b_for_clean_vjp(
        self, *, poison_check: Optional[Callable[[], None]] = None
    ) -> Iterator[None]:
        """Audit that an adapter-off clean-latent VJP never reaches LoRA-B.

        The native replay is executed with ``adapter_enabled=False``, so the
        Action-LoRA branch is absent from the autograd graph.  B must therefore
        stay in its canonical trainable gauge throughout this scope.  Earlier
        versions temporarily toggled ``B.requires_grad`` off; that legal local
        toggle nevertheless changed the authenticated model-runtime metadata
        and made every live runner fail its seal before the first replay.

        ``torch.autograd.grad`` targets only the clean latent, and the exact
        zero/no-gradient B state is checked below.  Keeping the gauge unchanged
        lets the same start/end runtime seal distinguish a real mutation from
        the intended adapter-off graph routing.
        """

        self.assert_fixed_gauge()
        snapshot = self.adapter_state_snapshot()
        b = self.canonical_b_named_parameters()
        failure: Optional[BaseException] = None
        fatal_runtime_tamper = False
        try:
            yield
        except NativeRuntimeSealChangedError:
            fatal_runtime_tamper = True
            raise
        except BaseException as error:
            failure = error
        finally:
            if not fatal_runtime_tamper:
                if poison_check is not None:
                    try:
                        poison_check()
                    except NativeRuntimeSealChangedError:
                        # A model-seal failure outranks the original error and
                        # forbids even adapter-only cleanup in this process.
                        raise
                if any(parameter.grad is not None for _, parameter in b):
                    failure = NativeRV2VHiddenVJPError(
                        "clean-latent VJP reached LoRA-B"
                    )
                legal_frozen_state = self.adapter_state_matches(snapshot)
                self.restore_adapter_state(snapshot)
                if not legal_frozen_state and failure is None:
                    failure = NativeRV2VHiddenVJPError(
                        "clean-latent VJP changed adapter state and was rolled back"
                    )
                if not self.adapter_state_matches(snapshot):
                    raise NativeRV2VHiddenVJPError(
                        "clean-latent adapter rollback audit failed"
                    )
                if failure is not None:
                    raise failure
        self.assert_fixed_gauge()

    def late_blocks_untouched(self) -> bool:
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B:
            return False
        rows = []
        for index in range(16, TOTAL_BLOCKS_1P3B):
            attention = getattr(blocks[index], "attn2", None)
            output = getattr(attention, "to_out", None)
            if output is None or len(output) < 1:
                return False
            rows.append((index, id(attention.to_q), id(output[0])))
        return tuple(rows) == self.original_late_qo_ids

    def restore(self) -> None:
        if self.restored or pair_adapter.active_route() is not None:
            raise NativeRV2VHiddenVJPError("core16 Action-LoRA cannot be restored now")
        if not self.late_blocks_untouched():
            raise NativeRV2VHiddenVJPError("late cross-attention changed during gauge")
        blocks = tuple(self.transformer.blocks)
        for index, module in self.original_q:
            blocks[index].attn2.to_q = module
        for index, module in self.original_o:
            blocks[index].attn2.to_out[0] = module
        self.restored = True


def _native_bridge_core16_route_factory(
    *,
    adapter: Core16ActionLoRAHandle,
    branch: native.NativeRV2VBranch,
    sequence_parallel_rank: int,
    sequence_parallel_size: int,
    sigma_schedule_index: int,
    enabled: bool,
) -> Any:
    if (
        type(adapter) is not Core16ActionLoRAHandle
        or sequence_parallel_size != SP_SIZE
        or sigma_schedule_index != NATIVE_SCHEDULE_INDEX
    ):
        raise native_bridge.PairV5NativeBridgeError(
            "Q-MOSAIC core16 native route coordinate differs"
        )
    if enabled:
        adapter.assert_fixed_gauge()
    else:
        a = adapter.canonical_a_named_parameters()
        b = adapter.canonical_b_named_parameters()
        adapter_ids = {id(parameter) for _, parameter in (*a, *b)}
        base_trainable = {
            id(parameter)
            for parameter in adapter.transformer.parameters()
            if parameter.requires_grad and id(parameter) not in adapter_ids
        }
        if (
            base_trainable
            or any(
                parameter.requires_grad
                or parameter.grad is not None
                or not torch.equal(
                    parameter.detach(),
                    _fixed_a_cpu(name, CANONICAL_A_SHAPE).to(parameter.device),
                )
                for name, parameter in a
            )
            or any(
                parameter.grad is not None
                or bool(torch.count_nonzero(parameter.detach()).item())
                for _, parameter in b
            )
        ):
            raise native_bridge.PairV5NativeBridgeError(
                "Q-MOSAIC disabled route gauge differs"
            )
    return adapter.route(
        branch,
        sp_rank=sequence_parallel_rank,
        adapter_enabled=enabled,
    )


def _native_bridge_core16_gate_factory(
    *, adapter: Core16ActionLoRAHandle, sigma_schedule_index: int
) -> tuple[str, float]:
    if type(adapter) is not Core16ActionLoRAHandle:
        raise native_bridge.PairV5NativeBridgeError(
            "Q-MOSAIC core16 gate handle differs"
        )
    if sigma_schedule_index != NATIVE_SCHEDULE_INDEX:
        raise native_bridge.PairV5NativeBridgeError(
            "Q-MOSAIC core16 gate is pinned to native schedule index 33"
        )
    gate_name, gate_weight = pair_adapter.sigma_gate(sigma_schedule_index)
    return str(gate_name), float(gate_weight)


native_bridge.register_closed_action_adapter_type(
    registry_id=CLOSED_NATIVE_ADAPTER_REGISTRY_ID,
    adapter_type=Core16ActionLoRAHandle,
    route_factory=_native_bridge_core16_route_factory,
    gate_factory=_native_bridge_core16_gate_factory,
)


def install_core16_fixed_a_b_only_action_lora(
    transformer: nn.Module,
) -> Core16ActionLoRAHandle:
    """Install the deterministic 32-tensor B coordinate without an optimizer."""

    if not isinstance(transformer, nn.Module) or any(
        parameter.requires_grad for parameter in transformer.parameters()
    ):
        raise NativeRV2VHiddenVJPError("freeze the complete Bernini base first")
    blocks = tuple(getattr(transformer, "blocks", ()))
    patch = getattr(transformer, "patch_embedding", None)
    if (
        len(blocks) != TOTAL_BLOCKS_1P3B
        or not isinstance(patch, nn.Conv3d)
        or int(patch.out_channels) != HIDDEN_SIZE
        or not callable(getattr(transformer, "patch_vae_latent", None))
    ):
        raise NativeRV2VHiddenVJPError("Bernini-R 1.3B native structure differs")

    originals_q: list[tuple[int, nn.Module]] = []
    originals_o: list[tuple[int, nn.Module]] = []
    late_ids: list[tuple[int, int, int]] = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn2", None)
        query = getattr(attention, "to_q", None)
        output = getattr(attention, "to_out", None)
        if (
            not isinstance(query, nn.Linear)
            or not isinstance(output, nn.ModuleList)
            or len(output) != 2
            or not isinstance(output[0], nn.Linear)
            or (query.in_features, query.out_features) != (HIDDEN_SIZE, HIDDEN_SIZE)
            or (output[0].in_features, output[0].out_features)
            != (HIDDEN_SIZE, HIDDEN_SIZE)
        ):
            raise NativeRV2VHiddenVJPError(f"block {index} attn2 Q/O differs")
        if index in ACTION_BLOCK_INDICES:
            originals_q.append((index, query))
            originals_o.append((index, output[0]))
        else:
            late_ids.append((index, id(query), id(output[0])))

    wrappers: dict[str, pair_adapter.PairV5TargetRowActionLoRA] = {}
    try:
        for index in ACTION_BLOCK_INDICES:
            query = dict(originals_q)[index]
            output = dict(originals_o)[index]
            q_name = f"blocks.{index}.attn2.to_q.action_lora_b.weight"
            o_name = f"blocks.{index}.attn2.to_out.0.action_lora_b.weight"
            q_wrapper = Core16ExactZeroTargetRowActionLoRA(
                query, projection="to_q", canonical_b_name=q_name
            ).to(device=query.weight.device)
            o_wrapper = Core16ExactZeroTargetRowActionLoRA(
                output, projection="to_out.0", canonical_b_name=o_name
            ).to(device=output.weight.device)
            blocks[index].attn2.to_q = q_wrapper
            blocks[index].attn2.to_out[0] = o_wrapper
            wrappers[q_name] = q_wrapper
            wrappers[o_name] = o_wrapper

        with torch.no_grad():
            for b_name in CANONICAL_B_PARAMETER_NAMES:
                wrapper = wrappers[b_name]
                a_name = b_name.replace(
                    "action_lora_b.weight", "action_lora_a.weight"
                )
                wrapper.action_lora_a.weight.copy_(
                    _fixed_a_cpu(a_name, CANONICAL_A_SHAPE).to(
                        device=wrapper.action_lora_a.weight.device
                    )
                )
                wrapper.action_lora_b.weight.zero_()
                wrapper.action_lora_a.weight.requires_grad_(False)
                wrapper.action_lora_b.weight.requires_grad_(True)
                wrapper.action_lora_a.weight.grad = None
                wrapper.action_lora_b.weight.grad = None
    except Exception:
        for index, module in originals_q:
            blocks[index].attn2.to_q = module
        for index, module in originals_o:
            blocks[index].attn2.to_out[0] = module
        raise

    a_rows = []
    b_rows = []
    for b_name in CANONICAL_B_PARAMETER_NAMES:
        wrapper = wrappers[b_name]
        a_rows.append(
            (
                b_name.replace("action_lora_b.weight", "action_lora_a.weight"),
                wrapper.action_lora_a.weight,
            )
        )
        b_rows.append((b_name, wrapper.action_lora_b.weight))
    unsigned = {
        "schema_version": GAUGE_SCHEMA_VERSION,
        "blocks": list(ACTION_BLOCK_INDICES),
        "projections_interleaved_per_block": ["attn2.to_q", "attn2.to_out.0"],
        "canonical_b_parameter_names": list(CANONICAL_B_PARAMETER_NAMES),
        "b_tensor_count": len(b_rows),
        "b_tensor_shape": list(CANONICAL_B_SHAPE),
        "b_parameter_count": CANONICAL_B_PARAMETER_COUNT,
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "scale": LORA_SCALE,
        "fixed_a_seed": FIXED_LORA_A_SEED,
        "a_frozen_fp32": True,
        "b_only_trainable_fp32": True,
        "b_exact_zero": True,
        "zero_b_forward_is_base_byte_exact_with_custom_vjp": True,
        "zero_route_proof_schema_version": ZERO_ROUTE_PROOF_SCHEMA_VERSION,
        "a_state_sha256": _named_tensor_sha256(a_rows, label="fixed LoRA-A"),
        "b_state_sha256": _named_tensor_sha256(b_rows, label="zero LoRA-B"),
        "legacy_all_q_then_all_o_enumeration_used": False,
        "optimizer_or_parameter_update": False,
        "real_auh_runtime_validated": False,
    }
    handle = Core16ActionLoRAHandle(
        transformer=transformer,
        wrappers_by_b_name=MappingProxyType(wrappers),
        original_q=tuple(originals_q),
        original_o=tuple(originals_o),
        original_late_qo_ids=tuple(late_ids),
        original_patch_embedding_id=id(patch),
        gauge_receipt=MappingProxyType({}),
    )
    try:
        handle.assert_fixed_gauge()
        if not handle.late_blocks_untouched():
            raise NativeRV2VHiddenVJPError("blocks 16..29 changed during install")
        # Reuse the established Phase-A gauge validator after imposing the
        # stricter core16/canonical-order contract.  We deliberately do not
        # call the legacy adapter's NumPy-backed save-state digest.
        import audit_pair_v7_phase_a_geometry as phase_a_geometry

        phase_a_gauge = phase_a_geometry.configure_fixed_a_b_only_gauge(handle)
        phase_b_names = tuple(name for name, _ in phase_a_gauge.trainable_b_named)
        if phase_b_names != CANONICAL_B_PARAMETER_NAMES:
            raise NativeRV2VHiddenVJPError(
                "Phase-A fixed-gauge B order differs from canonical Q/O interleave"
            )
        unsigned["phase_a_fixed_gauge_validator_reused"] = True
        unsigned["phase_a_fixed_gauge_receipt_digest"] = phase_a_gauge.receipt[
            "receipt_digest"
        ]
        handle.gauge_receipt = {
            **unsigned,
            "digest": object_sha256(unsigned),
        }
    except Exception:
        handle.restore()
        raise
    return handle


def _validate_detached_global_sketch(value: Any, *, label: str) -> torch.Tensor:
    wanted = (1, LATENT_PHASES, SPATIAL_SKETCH_COORDINATES, HIDDEN_SIZE)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(map(int, value.shape)) != wanted
        or value.dtype != torch.float32
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise NativeRV2VHiddenVJPError(
            f"{label} must be detached finite FP32 {wanted}"
        )
    return value.contiguous()


def _read_authenticated_owner_packet(
    *,
    authority: owner_materializer.AuthorizedOwnerInputs,
    cell_root: str | Path,
    receipt_path: str | Path,
    expected_receipt_file_sha256: str,
    query_seed: int,
) -> tuple[dict[str, Any], Any, torch.Tensor, Mapping[str, Any]]:
    if type(authority) is not owner_materializer.AuthorizedOwnerInputs:
        raise NativeRV2VHiddenVJPError("owner authority type differs")
    root = Path(cell_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise NativeRV2VHiddenVJPError("owner packet root must be an absolute plain directory")
    expected_receipt_path = root / owner_materializer.CELL_RECEIPT_FILENAME
    value, resolved_receipt, receipt_file_sha = _strict_json_file(
        receipt_path,
        expected_sha256=expected_receipt_file_sha256,
        label="owner quotient receipt",
    )
    if resolved_receipt != expected_receipt_path.resolve(strict=True):
        raise NativeRV2VHiddenVJPError("owner quotient receipt escaped its cell root")
    try:
        checked = owner_materializer.validate_published_cell_packet(
            value, cell_root=root, authority=authority
        )
    except owner_materializer.OwnerTemplateMaterializationError as error:
        raise NativeRV2VHiddenVJPError(str(error)) from error
    if type(query_seed) is not int or query_seed < 0:
        raise NativeRV2VHiddenVJPError("owner query seed differs")
    cell_id = checked.get("cell_id")
    cell = authority.registry.cell(cell_id)
    if query_seed not in cell.query_seeds:
        raise NativeRV2VHiddenVJPError("query seed is outside the authenticated cell")
    index = tuple(cell.query_seeds).index(query_seed)
    query_row = checked["query_rows"][index]
    key = f"{owner_materializer.TENSOR_KEY_PREFIX}{query_seed}"
    quotient_path = root / owner_materializer.QUOTIENT_FILENAME
    try:
        from safetensors import safe_open

        with safe_open(str(quotient_path), framework="pt", device="cpu") as opened:
            if key not in opened.keys():
                raise NativeRV2VHiddenVJPError("authenticated quotient key is missing")
            unit = opened.get_tensor(key).float().contiguous().detach()
    except NativeRV2VHiddenVJPError:
        raise
    except Exception as error:
        raise NativeRV2VHiddenVJPError("authenticated quotient could not be reopened") from error
    template = query_row.get("template")
    if (
        not isinstance(template, Mapping)
        or template.get("query_seed") != query_seed
        or owner_materializer.tensor_sha256(unit, label="authenticated owner quotient")
        != template.get("unit_feature_digest")
        or unit.ndim != 2
        or int(unit.shape[0]) != 1
        or unit.dtype != torch.float32
        or unit.requires_grad
        or unit.grad_fn is not None
        or not bool(torch.isfinite(unit).all().item())
    ):
        raise NativeRV2VHiddenVJPError("authenticated owner quotient value differs")
    artifact = checked["quotient_artifact"]
    bindings = MappingProxyType(
        {
            "receipt_path": str(resolved_receipt),
            "receipt_file_sha256": receipt_file_sha,
            "receipt_digest": checked["receipt_digest"],
            "quotient_path": str(quotient_path.resolve(strict=True)),
            "quotient_file_sha256": artifact["file_sha256"],
            "unit_feature_digest": template["unit_feature_digest"],
            "model_binding_digest": object_sha256(checked["model_binding"]),
            "owner_child_receipt_digest": checked["owner_child_receipt_digest"],
            "owner_audit_receipt_digest": checked[
                "external_full81_audit_sidecar_receipt_digest"
            ],
        }
    )
    return checked, cell, unit, bindings


@dataclass(frozen=True)
class _OwnerAuthorityLiveSpec:
    registry: Path
    expected_registry_sha256: str
    owner_root: Path
    owner_master_receipt: Path
    expected_owner_master_receipt_sha256: str
    audit_sidecar: Path
    expected_audit_sidecar_sha256: str
    audit_evidence: Path
    audit_public_key: Path
    expected_audit_public_key_sha256: str

    def reload(self) -> owner_materializer.AuthorizedOwnerInputs:
        try:
            return owner_materializer.load_authorized_owner_inputs(
                registry=self.registry,
                expected_registry_sha256=self.expected_registry_sha256,
                owner_root=self.owner_root,
                owner_master_receipt=self.owner_master_receipt,
                expected_owner_master_receipt_sha256=(
                    self.expected_owner_master_receipt_sha256
                ),
                audit_sidecar=self.audit_sidecar,
                expected_audit_sidecar_sha256=self.expected_audit_sidecar_sha256,
                audit_evidence=self.audit_evidence,
                audit_public_key=self.audit_public_key,
                expected_audit_public_key_sha256=(
                    self.expected_audit_public_key_sha256
                ),
            )
        except owner_materializer.OwnerTemplateMaterializationError as error:
            raise NativeRV2VHiddenVJPError(str(error)) from error


@dataclass(frozen=True)
class AuthenticatedOwnerQuotientPacket:
    cell_id: str
    query_seed: int
    source_iid: str
    source_video_sha256: str
    action_prompt_sha256: str
    noop_prompt_sha256: str
    action_family_id: str
    unit_feature: torch.Tensor = field(repr=False)
    unit_feature_runtime_sha256: str
    authority_bindings: Mapping[str, Any]
    _authority_live_spec: _OwnerAuthorityLiveSpec = field(
        repr=False, compare=False
    )
    _cell_root: Path = field(repr=False, compare=False)
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        self.assert_live()
        value = {
            "schema_version": OWNER_PACKET_SCHEMA_VERSION,
            "cell_id": self.cell_id,
            "query_seed": self.query_seed,
            "source_iid": self.source_iid,
            "source_video_sha256": self.source_video_sha256,
            "action_prompt_sha256": self.action_prompt_sha256,
            "noop_prompt_sha256": self.noop_prompt_sha256,
            "action_family_id": self.action_family_id,
            "unit_feature_runtime_sha256": self.unit_feature_runtime_sha256,
            "authority_bindings": dict(self.authority_bindings),
            "only_authenticated_detached_normalized_quotient_consumed": True,
            "raw_scorer_or_owner_tensor_callback_consumed": False,
        }
        return {**value, "digest": object_sha256(value)}

    def assert_live(self) -> None:
        if self._token is not _OWNER_PACKET_TOKEN:
            raise NativeRV2VHiddenVJPError("owner quotient packet was not authority-loaded")
        # Reload the registry, pending master, every child and every bound
        # artifact, then reverify the external Ed25519 audit and evidence on
        # every use.  An already-materialized AuthorizedOwnerInputs object is
        # deliberately not an authority cache.
        live_authority = self._authority_live_spec.reload()
        checked, cell, unit, bindings = _read_authenticated_owner_packet(
            authority=live_authority,
            cell_root=self._cell_root,
            receipt_path=self.authority_bindings["receipt_path"],
            expected_receipt_file_sha256=self.authority_bindings[
                "receipt_file_sha256"
            ],
            query_seed=self.query_seed,
        )
        del checked
        if (
            cell.cell_id != self.cell_id
            or cell.source_iid != self.source_iid
            or cell.source_video_sha256 != self.source_video_sha256
            or cell.action_caption_utf8_sha256 != self.action_prompt_sha256
            or cell.noop_caption_utf8_sha256 != self.noop_prompt_sha256
            or cell.action_family_id != self.action_family_id
            or dict(bindings) != dict(self.authority_bindings)
            or not torch.equal(unit, self.unit_feature)
            or tensor_sha256(unit, label="live owner quotient")
            != self.unit_feature_runtime_sha256
        ):
            raise NativeRV2VHiddenVJPError("live owner quotient packet changed")


def load_authenticated_owner_quotient_packet(
    *,
    registry: str | Path,
    expected_registry_sha256: str,
    owner_root: str | Path,
    owner_master_receipt: str | Path,
    expected_owner_master_receipt_sha256: str,
    audit_sidecar: str | Path,
    expected_audit_sidecar_sha256: str,
    audit_evidence: str | Path,
    audit_public_key: str | Path,
    expected_audit_public_key_sha256: str,
    cell_root: str | Path,
    receipt_path: str | Path,
    expected_receipt_file_sha256: str,
    query_seed: int,
) -> AuthenticatedOwnerQuotientPacket:
    """Load the sole authoritative owner-to-editor tensor channel."""

    live_spec = _OwnerAuthorityLiveSpec(
        registry=Path(registry).resolve(strict=True),
        expected_registry_sha256=_sha256(
            expected_registry_sha256, label="owner registry SHA-256"
        ),
        owner_root=Path(owner_root).resolve(strict=True),
        owner_master_receipt=Path(owner_master_receipt).resolve(strict=True),
        expected_owner_master_receipt_sha256=_sha256(
            expected_owner_master_receipt_sha256,
            label="owner master receipt SHA-256",
        ),
        audit_sidecar=Path(audit_sidecar).resolve(strict=True),
        expected_audit_sidecar_sha256=_sha256(
            expected_audit_sidecar_sha256,
            label="owner audit sidecar SHA-256",
        ),
        audit_evidence=Path(audit_evidence).resolve(strict=True),
        audit_public_key=Path(audit_public_key).resolve(strict=True),
        expected_audit_public_key_sha256=_sha256(
            expected_audit_public_key_sha256,
            label="owner audit public key SHA-256",
        ),
    )
    authority = live_spec.reload()
    _checked, cell, unit, bindings = _read_authenticated_owner_packet(
        authority=authority,
        cell_root=cell_root,
        receipt_path=receipt_path,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        query_seed=query_seed,
    )
    packet = AuthenticatedOwnerQuotientPacket(
        cell_id=cell.cell_id,
        query_seed=query_seed,
        source_iid=cell.source_iid,
        source_video_sha256=cell.source_video_sha256,
        action_prompt_sha256=cell.action_caption_utf8_sha256,
        noop_prompt_sha256=cell.noop_caption_utf8_sha256,
        action_family_id=cell.action_family_id,
        unit_feature=unit.clone(),
        unit_feature_runtime_sha256=tensor_sha256(
            unit, label="loaded owner quotient"
        ),
        authority_bindings=bindings,
        _authority_live_spec=live_spec,
        _cell_root=Path(cell_root).resolve(strict=True),
    )
    object.__setattr__(packet, "_token", _OWNER_PACKET_TOKEN)
    packet.assert_live()
    return packet


@dataclass(frozen=True)
class EditorSameStatePromptPacket:
    cell_id: str
    query_seed: int
    sp_rank: int
    branch_name: str
    source_iid: str
    source_video_sha256: str
    action_prompt_sha256: str
    noop_prompt_sha256: str
    action_measurement: torch.Tensor = field(repr=False)
    noop_measurement: torch.Tensor = field(repr=False)
    local_action_measurement: torch.Tensor = field(repr=False)
    local_noop_measurement: torch.Tensor = field(repr=False)
    shared_state_binding_digest: str
    bindings: Mapping[str, Any]
    _runtime_tensors: Mapping[str, torch.Tensor] = field(repr=False, compare=False)
    _runtime_tensor_bindings: Mapping[str, Mapping[str, Any]] = field(
        repr=False, compare=False
    )
    _runtime_owner_digest: str = field(repr=False, compare=False)
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": EDITOR_PACKET_SCHEMA_VERSION,
            "cell_id": self.cell_id,
            "query_seed": self.query_seed,
            "sp_rank": self.sp_rank,
            "sp_size": SP_SIZE,
            "branch_name": self.branch_name,
            "source_iid": self.source_iid,
            "source_video_sha256": self.source_video_sha256,
            "action_prompt_sha256": self.action_prompt_sha256,
            "noop_prompt_sha256": self.noop_prompt_sha256,
            "action_measurement_sha256": tensor_sha256(
                self.action_measurement, label="editor action measurement"
            ),
            "noop_measurement_sha256": tensor_sha256(
                self.noop_measurement, label="editor no-op measurement"
            ),
            "local_action_measurement_sha256": tensor_sha256(
                self.local_action_measurement, label="editor local action measurement"
            ),
            "local_noop_measurement_sha256": tensor_sha256(
                self.local_noop_measurement, label="editor local no-op measurement"
            ),
            "shared_state_binding_digest": self.shared_state_binding_digest,
            "bindings": dict(self.bindings),
            "same_x_sigma_action_noop": True,
            "native_shared_step_only": True,
            "runtime_owned": True,
        }
        return {**value, "digest": object_sha256(value)}

    def assert_live(self, owner: AuthenticatedOwnerQuotientPacket) -> None:
        if self._token is not _EDITOR_PACKET_TOKEN:
            raise NativeRV2VHiddenVJPError("editor packet is not runtime-owned")
        owner.assert_live()
        if (
            self._runtime_owner_digest != owner.receipt()["digest"]
            or self.cell_id != owner.cell_id
            or self.query_seed != owner.query_seed
            or self.source_iid != owner.source_iid
            or self.source_video_sha256 != owner.source_video_sha256
            or self.action_prompt_sha256 != owner.action_prompt_sha256
            or self.noop_prompt_sha256 != owner.noop_prompt_sha256
            or self.branch_name != "VI"
            or self.sp_rank not in range(SP_SIZE)
            or self.bindings.get("native_schedule_index") != NATIVE_SCHEDULE_INDEX
            or self.bindings.get("native_timestep") != NATIVE_TIMESTEP
            or float(self.bindings.get("native_sigma", -1.0)).hex()
            != float(NATIVE_SIGMA).hex()
            or self.bindings.get("sp_size") != SP_SIZE
            or self.bindings.get("sp_rank") != self.sp_rank
            or self.bindings.get("shared_state_binding_digest")
            != self.shared_state_binding_digest
            or _SHA256_RE.fullmatch(self.shared_state_binding_digest) is None
            or self.bindings.get("model_proof_runtime_owned") is not True
            or self.bindings.get("owner_query_seed") != self.query_seed
            or self.bindings.get("editor_noise_seed")
            != editor_noise_seed_from_owner_query_seed(self.query_seed)
            or self.bindings.get("editor_noise_seed") == self.query_seed
            or self.bindings.get("official_cpu_generator_gaussian") is not True
            or _SHA256_RE.fullmatch(
                str(self.bindings.get("prompt_condition_binding_digest"))
            )
            is None
        ):
            raise NativeRV2VHiddenVJPError("editor/owner same-state binding differs")
        for name, value in self._runtime_tensors.items():
            _assert_tensor_runtime_binding(
                value,
                self._runtime_tensor_bindings[name],
                label=f"editor runtime {name}",
                require_same_object=True,
            )
        for label, measurement in (
            ("action", self.action_measurement),
            ("noop", self.noop_measurement),
            ("local action", self.local_action_measurement),
            ("local noop", self.local_noop_measurement),
        ):
            _validate_detached_global_sketch(measurement, label=label)
        sealed_measurements = {
            "sealed_action_measurement_sha256": tensor_sha256(
                self.action_measurement, label="sealed action measurement"
            ),
            "sealed_noop_measurement_sha256": tensor_sha256(
                self.noop_measurement, label="sealed no-op measurement"
            ),
            "sealed_local_action_measurement_sha256": tensor_sha256(
                self.local_action_measurement,
                label="sealed local action measurement",
            ),
            "sealed_local_noop_measurement_sha256": tensor_sha256(
                self.local_noop_measurement,
                label="sealed local no-op measurement",
            ),
        }
        if any(
            self.bindings.get(name) != digest
            for name, digest in sealed_measurements.items()
        ):
            raise NativeRV2VHiddenVJPError("editor measurement bytes changed after sealing")


@dataclass(frozen=True)
class RuntimeOwnedReplaySession:
    editor_packet: EditorSameStatePromptPacket
    session_digest: str
    _runner: Any = field(repr=False, compare=False)
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def assert_live(self, owner: AuthenticatedOwnerQuotientPacket) -> None:
        if self._token is not _REPLAY_SESSION_TOKEN:
            raise NativeRV2VHiddenVJPError("replay session is not runtime-owned")
        self.editor_packet.assert_live(owner)
        if isinstance(self._runner, NativeSharedStepSP4ReplayRunner):
            runner_contract = self._runner.contract_receipt(deep=False)
        else:
            # Tiny test fixtures predate the production runner's explicit
            # boundary-seal mode and intentionally expose only the old API.
            runner_contract = self._runner.contract_receipt()
        expected = object_sha256(
            {
                "schema_version": REPLAY_SESSION_SCHEMA_VERSION,
                "editor_packet_digest": self.editor_packet.receipt()["digest"],
                "runner_contract_digest": runner_contract["digest"],
            }
        )
        if expected != self.session_digest:
            raise NativeRV2VHiddenVJPError("runtime replay session binding changed")

    def replay(
        self,
        *,
        owner: AuthenticatedOwnerQuotientPacket,
        role: str,
        adapter_enabled: bool,
    ) -> torch.Tensor:
        self.assert_live(owner)
        value = self._runner._replay_runtime_owned(  # noqa: SLF001 - closed runner/session
            role=role, adapter_enabled=adapter_enabled
        )
        # Rehash all runtime objects immediately after the model call too.  A
        # callback cannot swap a prompt/source/timestep during replay and then
        # restore it before the row escapes.
        self.assert_live(owner)
        return value


_EDITOR_RUNTIME_INPUT_SCHEMA = "bernini-qmosaic-signed-editor-runtime-input-v2"
_EDITOR_NOISE_SEED_DOMAIN_OFFSET = 1000


def editor_noise_seed_from_owner_query_seed(owner_query_seed: int) -> int:
    """Return the fixed, preregistered editor-noise domain separation.

    The owner query seed selects a detached owner quotient and must never also
    select the editor's Gaussian.  A fixed +1000 domain tag keeps the mapping
    transparent and deterministic while making byte-identical owner/editor
    noise impossible for equal latent shapes.
    """

    if type(owner_query_seed) is not int or not 0 <= owner_query_seed < 2**63 - 1000:
        raise NativeRV2VHiddenVJPError("owner query seed is outside editor domain")
    return owner_query_seed + _EDITOR_NOISE_SEED_DOMAIN_OFFSET


_EDITOR_RUNTIME_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "cell_id",
        "owner_query_seed",
        "editor_noise_seed",
        "method_source_revision",
        "method_source_archive_sha256",
        "source_iid",
        "source_video_sha256",
        "action_prompt_utf8",
        "noop_prompt_utf8",
        "action_prompt_sha256",
        "noop_prompt_sha256",
        "owner_packet_receipt_digest",
        "checkpoint_content_receipt_digest",
        "tokenizer_receipt_digest",
        "text_encoder_receipt_digest",
        "source_video_artifact",
        "materialization_receipt_artifact",
        "runtime_tensor_artifact",
        "authority_public_key_sha256",
        "authority_signature_scheme",
        "receipt_digest",
        "authority_signature_ed25519_base64",
    }
)
_EDITOR_RUNTIME_TENSOR_KEYS = (
    "source_latent",
    "image_reference_0",
    "image_reference_1",
    "image_reference_2",
    "image_reference_3",
    "clean_latent",
    "official_initial_noise",
    "action_condition",
    "noop_condition",
    "timestep",
)
_EDITOR_RUNTIME_CODE_ROLES = (
    "qmosaic_materializer",
    "native_rv2v_loader_encoder_entrypoint",
    "native_rv2v_legacy_contract",
    "source_snapshot_and_freeze_audit",
    "owner_registry_loader",
    "editor_runtime_core_loader",
    "native_sampler_contract",
    "native_exact40_schedule_contract",
    "official_bernini_pipeline",
    "official_bernini_renderer",
    "official_bernini_parallel",
    "official_bernini_wan_diffusion",
    "diffusers_wan_vae_implementation",
    "diffusers_unipc_scheduler_implementation",
    "transformers_tokenizer_implementation",
    "transformers_t5_encoder_implementation",
)
_EDITOR_METHOD_OWNED_RUNTIME_CODE_ROLES = _EDITOR_RUNTIME_CODE_ROLES[:8]
_EDITOR_MATERIALIZER_ARCHIVE_MEMBER = (
    "methods/bernini_action_editing/materialize_qmosaic_editor_runtime_v1.py"
)


def _validate_editor_material_runtime_code(
    receipt: Mapping[str, Any], *, artifact_root: Path
) -> str:
    """Rehash the complete code closure, including durable packet-owned code."""

    if not isinstance(receipt, Mapping):
        raise NativeRV2VHiddenVJPError(
            "editor materialization runtime code receipt differs"
        )
    unsigned = dict(receipt)
    declared = unsigned.pop("digest", None)
    rows = receipt.get("files")
    if (
        set(receipt)
        != {
            "schema_version",
            "role_order",
            "file_count",
            "files",
            "all_files_hashed_from_canonical_plain_paths",
            "digest",
        }
        or receipt.get("schema_version") != "qmosaic-runtime-code-closure-v1"
        or receipt.get("role_order") != list(_EDITOR_RUNTIME_CODE_ROLES)
        or receipt.get("file_count") != len(_EDITOR_RUNTIME_CODE_ROLES)
        or receipt.get("all_files_hashed_from_canonical_plain_paths") is not True
        or not isinstance(rows, list)
        or len(rows) != len(_EDITOR_RUNTIME_CODE_ROLES)
        or declared != object_sha256(unsigned)
    ):
        raise NativeRV2VHiddenVJPError(
            "editor materialization runtime code seal/closure differs"
        )
    for index, (role, row) in enumerate(
        zip(_EDITOR_RUNTIME_CODE_ROLES, rows)
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"role", "path", "file_sha256"}
            or row.get("role") != role
            or not isinstance(row.get("path"), str)
            or _SHA256_RE.fullmatch(str(row.get("file_sha256"))) is None
        ):
            raise NativeRV2VHiddenVJPError(
                f"editor materialization runtime code row differs for {role}"
            )
        path = Path(row["path"])
        if role in _EDITOR_METHOD_OWNED_RUNTIME_CODE_ROLES:
            expected = artifact_root / "runtime-code" / f"{index:02d}-{role}.py"
            if path != expected:
                raise NativeRV2VHiddenVJPError(
                    f"packet-owned runtime code path differs for {role}"
                )
        if file_sha256(path) != row["file_sha256"]:
            raise NativeRV2VHiddenVJPError(
                f"editor materialization runtime code bytes changed for {role}"
            )
    return _sha256(declared, label="editor runtime code closure digest")


def _validate_editor_materializer_archive_member(
    archive: Path, *, expected_file_sha256: str
) -> None:
    expected = _sha256(
        expected_file_sha256,
        label="editor materializer archive member SHA-256",
    )
    try:
        with tarfile.open(archive, "r:*") as handle:
            rows = [
                member
                for member in handle.getmembers()
                if member.name == _EDITOR_MATERIALIZER_ARCHIVE_MEMBER
            ]
            if len(rows) != 1 or not rows[0].isfile():
                raise NativeRV2VHiddenVJPError(
                    "editor materializer archive member closure differs"
                )
            opened = handle.extractfile(rows[0])
            if opened is None or hashlib.sha256(opened.read()).hexdigest() != expected:
                raise NativeRV2VHiddenVJPError(
                    "editor materializer archive member bytes differ"
                )
    except NativeRV2VHiddenVJPError:
        raise
    except (tarfile.TarError, OSError) as error:
        raise NativeRV2VHiddenVJPError(
            "editor materializer archive cannot be reopened"
        ) from error


def _verify_editor_runtime_input_payload(
    *,
    receipt_path: str | Path,
    expected_receipt_file_sha256: str,
    public_key_path: str | Path,
    expected_public_key_file_sha256: str,
    artifact_root: str | Path,
) -> tuple[dict[str, Any], Path, Path, Path, Mapping[str, torch.Tensor]]:
    value, resolved_receipt, _ = _strict_json_file(
        receipt_path,
        expected_sha256=expected_receipt_file_sha256,
        label="signed editor runtime input",
    )
    if set(value) != _EDITOR_RUNTIME_INPUT_FIELDS:
        raise NativeRV2VHiddenVJPError(
            "editor runtime input field closure differs"
        )
    if (
        type(value.get("owner_query_seed")) is not int
        or value["owner_query_seed"] < 0
        or type(value.get("editor_noise_seed")) is not int
        or value["editor_noise_seed"]
        != editor_noise_seed_from_owner_query_seed(value["owner_query_seed"])
        or value["editor_noise_seed"] == value["owner_query_seed"]
        or _SHA1_RE.fullmatch(str(value.get("method_source_revision"))) is None
        or _SHA256_RE.fullmatch(
            str(value.get("method_source_archive_sha256"))
        )
        is None
        or not isinstance(value.get("cell_id"), str)
        or not value["cell_id"]
        or not isinstance(value.get("source_iid"), str)
        or not value["source_iid"]
        or any(
            _SHA256_RE.fullmatch(str(value.get(name))) is None
            for name in (
                "source_video_sha256",
                "action_prompt_sha256",
                "noop_prompt_sha256",
                "owner_packet_receipt_digest",
                "checkpoint_content_receipt_digest",
                "tokenizer_receipt_digest",
                "text_encoder_receipt_digest",
            )
        )
    ):
        raise NativeRV2VHiddenVJPError(
            "editor runtime scalar/hash closure differs"
        )
    key = Path(public_key_path)
    observed_key_sha = file_sha256(key)
    if (
        observed_key_sha
        != _sha256(
            expected_public_key_file_sha256,
            label="editor runtime public key SHA-256",
        )
        or value.get("schema_version") != _EDITOR_RUNTIME_INPUT_SCHEMA
        or value.get("authority_signature_scheme")
        != DIRECTION_GATE_SIGNATURE_SCHEME
        or value.get("authority_public_key_sha256") != observed_key_sha
    ):
        raise NativeRV2VHiddenVJPError(
            "editor runtime signing authority differs"
        )
    signed = dict(value)
    encoded = signed.pop("authority_signature_ed25519_base64", None)
    unsigned = dict(signed)
    declared_digest = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(encoded, str)
        or declared_digest != object_sha256(unsigned)
    ):
        raise NativeRV2VHiddenVJPError("editor runtime receipt seal differs")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        signature = base64.b64decode(encoded.encode("ascii"), validate=True)
        public_key = serialization.load_pem_public_key(key.read_bytes())
        if not isinstance(public_key, Ed25519PublicKey) or len(signature) != 64:
            raise NativeRV2VHiddenVJPError(
                "editor runtime Ed25519 key/signature differs"
            )
        public_key.verify(signature, canonical_json_bytes(signed))
    except NativeRV2VHiddenVJPError:
        raise
    except (InvalidSignature, ValueError, UnicodeError) as error:
        raise NativeRV2VHiddenVJPError(
            "editor runtime Ed25519 verification failed"
        ) from error
    root = Path(artifact_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise NativeRV2VHiddenVJPError("editor runtime artifact root differs")
    resolved_root = root.resolve(strict=True)
    materialization = value.get("materialization_receipt_artifact")
    if not isinstance(materialization, Mapping) or set(materialization) != {
        "path", "file_sha256", "receipt_digest",
    }:
        raise NativeRV2VHiddenVJPError(
            "editor materialization receipt binding differs"
        )
    material_path = Path(materialization["path"])
    material_value, resolved_material, material_sha = _strict_json_file(
        material_path,
        expected_sha256=materialization["file_sha256"],
        label="editor materialization receipt",
    )
    try:
        resolved_material.relative_to(resolved_root)
    except ValueError as error:
        raise NativeRV2VHiddenVJPError(
            "editor materialization receipt escaped its artifact root"
        ) from error
    material_unsigned = dict(material_value)
    material_digest = material_unsigned.pop("receipt_digest", None)
    if (
        material_sha != materialization.get("file_sha256")
        or material_digest != object_sha256(material_unsigned)
        or material_digest != materialization.get("receipt_digest")
    ):
        raise NativeRV2VHiddenVJPError(
            "editor materialization receipt seal differs"
        )
    material_source = material_value.get("method_source")
    material_runtime_code = material_value.get("runtime_code")
    material_endpoint = material_value.get("native_base_endpoint")
    material_interpretation = material_value.get("interpretation")
    if (
        material_value.get("schema_version")
        != "bernini-qmosaic-editor-runtime-materialization-v1"
        or material_value.get("method")
        != "qmosaic-method-owned-native-rv2v-editor-runtime-v1"
        or not isinstance(material_source, Mapping)
        or not isinstance(material_runtime_code, Mapping)
        or not isinstance(material_endpoint, Mapping)
        or not isinstance(material_interpretation, Mapping)
        or set(material_source)
        != {
            "revision",
            "archive_path",
            "archive_file_sha256",
            "archive_member",
            "archive_member_file_sha256",
            "fresh_archive_extraction_required",
            "path",
            "file_sha256",
        }
        or not isinstance(material_source.get("archive_path"), str)
        or not isinstance(material_source.get("path"), str)
        or material_source.get("fresh_archive_extraction_required") is not True
        or material_source.get("archive_member")
        != _EDITOR_MATERIALIZER_ARCHIVE_MEMBER
        or material_source.get("archive_member_file_sha256")
        != material_source.get("file_sha256")
    ):
        raise NativeRV2VHiddenVJPError(
            "editor materialization scientific evidence differs"
        )
    durable_archive = Path(material_source["archive_path"])
    durable_materializer = Path(material_source["path"])
    runtime_code_digest = _validate_editor_material_runtime_code(
        material_runtime_code, artifact_root=resolved_root
    )
    _validate_editor_materializer_archive_member(
        durable_archive,
        expected_file_sha256=material_source[
            "archive_member_file_sha256"
        ],
    )
    if (
        durable_archive != resolved_root / "method-source-archive.tar"
        or durable_materializer
        != resolved_root / "runtime-code/00-qmosaic_materializer.py"
        or file_sha256(durable_archive)
        != material_source.get("archive_file_sha256")
        or material_source.get("revision")
        != value.get("method_source_revision")
        or material_source.get("archive_file_sha256")
        != value.get("method_source_archive_sha256")
        or file_sha256(durable_materializer)
        != material_source.get("file_sha256")
        or runtime_code_digest
        != material_value.get("runtime_code_receipt_digest")
        or material_endpoint.get("owner_query_seed")
        != value.get("owner_query_seed")
        or material_endpoint.get("editor_noise_seed")
        != value.get("editor_noise_seed")
        or material_endpoint.get("owner_editor_noise_seed_shared") is not False
        or material_interpretation.get("training_performed") is not False
        or material_interpretation.get("parameter_update") is not False
    ):
        raise NativeRV2VHiddenVJPError(
            "editor materialization scientific evidence differs"
        )
    source = value.get("source_video_artifact")
    if not isinstance(source, Mapping) or set(source) != {"path", "file_sha256"}:
        raise NativeRV2VHiddenVJPError(
            "editor runtime source-video binding differs"
        )
    source_path = Path(source["path"])
    source_sha = file_sha256(source_path)
    resolved_source = source_path.resolve(strict=True)
    try:
        resolved_source.relative_to(resolved_root)
    except ValueError as error:
        raise NativeRV2VHiddenVJPError(
            "editor runtime source video escaped its artifact root"
        ) from error
    if (
        source_sha != source.get("file_sha256")
        or source_sha != value.get("source_video_sha256")
    ):
        raise NativeRV2VHiddenVJPError(
            "editor runtime source-video bytes differ"
        )
    artifact = value.get("runtime_tensor_artifact")
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "path",
        "file_sha256",
        "tensor_keys",
        "tensor_sha256_by_key",
        "create_only",
        "materializer_receipt_digest",
    }:
        raise NativeRV2VHiddenVJPError(
            "editor runtime tensor artifact binding differs"
        )
    tensor_path = Path(artifact["path"])
    tensor_file_sha = file_sha256(tensor_path)
    resolved_tensor = tensor_path.resolve(strict=True)
    try:
        resolved_tensor.relative_to(resolved_root)
    except ValueError as error:
        raise NativeRV2VHiddenVJPError(
            "editor runtime tensor artifact escaped its root"
        ) from error
    if (
        tensor_file_sha != artifact.get("file_sha256")
        or artifact.get("tensor_keys") != list(_EDITOR_RUNTIME_TENSOR_KEYS)
        or not isinstance(artifact.get("tensor_sha256_by_key"), Mapping)
        or set(artifact["tensor_sha256_by_key"])
        != set(_EDITOR_RUNTIME_TENSOR_KEYS)
        or artifact.get("create_only") is not True
        or artifact.get("materializer_receipt_digest") != material_digest
        or _SHA256_RE.fullmatch(
            str(artifact.get("materializer_receipt_digest"))
        )
        is None
    ):
        raise NativeRV2VHiddenVJPError(
            "editor runtime tensor file authority differs"
        )
    try:
        from safetensors import safe_open

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(resolved_tensor), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != sorted(_EDITOR_RUNTIME_TENSOR_KEYS):
                raise NativeRV2VHiddenVJPError(
                    "editor runtime safetensors key closure differs"
                )
            for name in _EDITOR_RUNTIME_TENSOR_KEYS:
                tensor = opened.get_tensor(name).detach().contiguous()
                if (
                    tensor.dtype != torch.float32
                    or tensor.requires_grad
                    or tensor.grad_fn is not None
                    or not bool(torch.isfinite(tensor).all().item())
                    or tensor_sha256(tensor, label=f"editor runtime {name}")
                    != artifact["tensor_sha256_by_key"].get(name)
                ):
                    raise NativeRV2VHiddenVJPError(
                        f"editor runtime tensor {name} differs"
                    )
                tensors[name] = tensor
    except NativeRV2VHiddenVJPError:
        raise
    except Exception as error:
        raise NativeRV2VHiddenVJPError(
            "editor runtime safetensors could not be reopened"
        ) from error
    clean = tensors["clean_latent"]
    noise = tensors["official_initial_noise"]
    source_latent = tensors["source_latent"]
    references = tuple(tensors[f"image_reference_{index}"] for index in range(4))
    if (
        tuple(map(int, clean.shape[:3])) != (1, 16, LATENT_PHASES)
        or clean.ndim != 5
        or noise.shape != clean.shape
        or source_latent.shape != clean.shape
        or any(
            reference.ndim != 5
            or tuple(map(int, reference.shape[:3])) != (1, 16, 1)
            or tuple(reference.shape[3:]) != tuple(clean.shape[3:])
            for reference in references
        )
        or tensors["action_condition"].ndim != 3
        or tensors["action_condition"].shape
        != tensors["noop_condition"].shape
        or tensors["timestep"].numel() != 1
        or float(tensors["timestep"].item()) != float(NATIVE_TIMESTEP)
    ):
        raise NativeRV2VHiddenVJPError(
            "editor runtime tensor geometry differs"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(value.get("editor_noise_seed"))
    official = torch.randn(
        tuple(map(int, noise.shape)), generator=generator, dtype=torch.float32
    )
    if not torch.equal(noise, official):
        raise NativeRV2VHiddenVJPError(
            "editor runtime noise is not the domain-separated editor Gaussian"
        )
    return (
        value,
        resolved_receipt,
        key.resolve(strict=True),
        resolved_root,
        MappingProxyType(tensors),
    )


@dataclass(frozen=True)
class AuthenticatedEditorRuntimeInputPacket:
    payload: Mapping[str, Any]
    tensors: Mapping[str, torch.Tensor] = field(repr=False)
    receipt_path: Path
    receipt_file_sha256: str
    public_key_path: Path
    public_key_file_sha256: str
    artifact_root: Path
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def assert_live(
        self,
        owner: AuthenticatedOwnerQuotientPacket,
        checkpoint: ValidatedCheckpointContentManifest,
    ) -> None:
        if self._token is not _EDITOR_RUNTIME_INPUT_TOKEN:
            raise NativeRV2VHiddenVJPError(
                "editor runtime input was not signature-loaded"
            )
        owner.assert_live()
        checkpoint.assert_live()
        value, receipt, key, root, tensors = _verify_editor_runtime_input_payload(
            receipt_path=self.receipt_path,
            expected_receipt_file_sha256=self.receipt_file_sha256,
            public_key_path=self.public_key_path,
            expected_public_key_file_sha256=self.public_key_file_sha256,
            artifact_root=self.artifact_root,
        )
        action_prompt = value.get("action_prompt_utf8")
        noop_prompt = value.get("noop_prompt_utf8")
        if (
            value != dict(self.payload)
            or receipt != self.receipt_path
            or key != self.public_key_path
            or root != self.artifact_root
            or value.get("cell_id") != owner.cell_id
            or value.get("owner_query_seed") != owner.query_seed
            or value.get("editor_noise_seed")
            != editor_noise_seed_from_owner_query_seed(owner.query_seed)
            or value.get("source_iid") != owner.source_iid
            or value.get("source_video_sha256") != owner.source_video_sha256
            or not isinstance(action_prompt, str)
            or not isinstance(noop_prompt, str)
            or hashlib.sha256(action_prompt.encode("utf-8")).hexdigest()
            != owner.action_prompt_sha256
            or hashlib.sha256(noop_prompt.encode("utf-8")).hexdigest()
            != owner.noop_prompt_sha256
            or value.get("action_prompt_sha256") != owner.action_prompt_sha256
            or value.get("noop_prompt_sha256") != owner.noop_prompt_sha256
            or value.get("owner_packet_receipt_digest")
            != owner.receipt()["digest"]
            or value.get("checkpoint_content_receipt_digest")
            != checkpoint.receipt()["digest"]
            or any(
                _SHA256_RE.fullmatch(str(value.get(name))) is None
                for name in (
                    "tokenizer_receipt_digest",
                    "text_encoder_receipt_digest",
                )
            )
            or any(
                not torch.equal(tensors[name], self.tensors[name])
                for name in _EDITOR_RUNTIME_TENSOR_KEYS
            )
        ):
            raise NativeRV2VHiddenVJPError(
                "editor runtime input live owner/model binding differs"
            )

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": _EDITOR_RUNTIME_INPUT_SCHEMA,
            "signed_receipt_digest": self.payload["receipt_digest"],
            "signed_receipt_file_sha256": self.receipt_file_sha256,
            "owner_query_seed": self.payload["owner_query_seed"],
            "editor_noise_seed": self.payload["editor_noise_seed"],
            "method_source_revision": self.payload[
                "method_source_revision"
            ],
            "method_source_archive_sha256": self.payload[
                "method_source_archive_sha256"
            ],
            "owner_packet_receipt_digest": self.payload[
                "owner_packet_receipt_digest"
            ],
            "checkpoint_content_receipt_digest": self.payload[
                "checkpoint_content_receipt_digest"
            ],
            "materialization_receipt_digest": self.payload[
                "materialization_receipt_artifact"
            ]["receipt_digest"],
            "materialization_receipt_file_sha256": self.payload[
                "materialization_receipt_artifact"
            ]["file_sha256"],
            "materialization_receipt_path": self.payload[
                "materialization_receipt_artifact"
            ]["path"],
            "runtime_tensor_artifact_file_sha256": self.payload[
                "runtime_tensor_artifact"
            ]["file_sha256"],
        }
        return {**value, "digest": object_sha256(value)}


def load_authenticated_editor_runtime_input_packet(
    *,
    receipt_path: str | Path,
    expected_receipt_file_sha256: str,
    public_key_path: str | Path,
    expected_public_key_file_sha256: str,
    artifact_root: str | Path,
    owner: AuthenticatedOwnerQuotientPacket,
    checkpoint: ValidatedCheckpointContentManifest,
) -> AuthenticatedEditorRuntimeInputPacket:
    value, receipt, key, root, tensors = _verify_editor_runtime_input_payload(
        receipt_path=receipt_path,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        public_key_path=public_key_path,
        expected_public_key_file_sha256=expected_public_key_file_sha256,
        artifact_root=artifact_root,
    )
    packet = AuthenticatedEditorRuntimeInputPacket(
        payload=MappingProxyType(value),
        tensors=MappingProxyType(
            {name: tensor.clone() for name, tensor in tensors.items()}
        ),
        receipt_path=receipt,
        receipt_file_sha256=_sha256(
            expected_receipt_file_sha256,
            label="editor runtime receipt file SHA-256",
        ),
        public_key_path=key,
        public_key_file_sha256=_sha256(
            expected_public_key_file_sha256,
            label="editor runtime public key file SHA-256",
        ),
        artifact_root=root,
    )
    object.__setattr__(packet, "_token", _EDITOR_RUNTIME_INPUT_TOKEN)
    packet.assert_live(owner, checkpoint)
    return packet


def _parameter_storage_pointer(value: torch.Tensor) -> int:
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:  # Torch 1.12
        return int(value.storage().data_ptr())


def _callable_runtime_binding(value: Any) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    target = getattr(value, "__func__", value)
    code = getattr(target, "__code__", None)
    return {
        "object_id": id(target),
        "exact_type": f"{type(target).__module__}.{type(target).__qualname__}",
        "code_object_id": None if code is None else id(code),
    }


def _module_behavior_binding(module: nn.Module) -> Mapping[str, Any]:
    hook_names = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
    )
    hooks: dict[str, Any] = {}
    for registry_name in hook_names:
        registry = getattr(module, registry_name, None)
        if registry is None:
            hooks[registry_name] = None
            continue
        if not isinstance(registry, Mapping):
            raise NativeRV2VHiddenVJPError(
                f"model hook registry {registry_name} differs"
            )
        hooks[registry_name] = [
            {
                "key": key if isinstance(key, (str, int)) else id(key),
                "key_type": f"{type(key).__module__}.{type(key).__qualname__}",
                "callable": _callable_runtime_binding(callback),
            }
            for key, callback in registry.items()
        ]
    module_type = type(module)
    return MappingProxyType(
        {
            "class_forward": _callable_runtime_binding(
                getattr(module_type, "forward", None)
            ),
            "instance_forward_override": _callable_runtime_binding(
                module.__dict__.get("forward")
            ),
            "class_call_impl": _callable_runtime_binding(
                getattr(module_type, "_call_impl", None)
            ),
            "compiled_call_impl": _callable_runtime_binding(
                getattr(module, "_compiled_call_impl", None)
            ),
            "hook_registries": hooks,
            "non_persistent_buffer_names": sorted(
                module._non_persistent_buffers_set
            ),
            "full_backward_hook_mode": getattr(
                module, "_is_full_backward_hook", None
            ),
        }
    )


def _complete_model_runtime_receipt(
    *,
    diffusion: nn.Module,
    transformer: nn.Module,
    adapter_b_ids: frozenset[int],
    _hash_tensor_bytes: bool = True,
) -> Mapping[str, Any]:
    """Bind the complete module graph, optionally including every tensor byte.

    ``_hash_tensor_bytes=False`` is an internal between-forward integrity seal.
    It retains object identity, storage pointer, tensor version, module edges,
    hooks, train/eval state and gradient metadata, but performs no device-to-CPU
    copies.  Authoritative start/end receipts always use the default full-byte
    mode.
    """

    if (
        not isinstance(diffusion, nn.Module)
        or not isinstance(transformer, nn.Module)
        or type(_hash_tensor_bytes) is not bool
    ):
        raise NativeRV2VHiddenVJPError("model runtime seal roots differ")
    tensor_hash_cache: dict[int, str] = {}

    def tensor_row(
        value: torch.Tensor, *, path: str, parameter: bool
    ) -> Mapping[str, Any]:
        key = id(value)
        digest = tensor_hash_cache.get(key)
        if _hash_tensor_bytes and digest is None:
            digest = tensor_sha256(value, label=f"sealed model tensor {path}")
            tensor_hash_cache[key] = digest
        gradient = value.grad if parameter else None
        gradient_row: Any = None
        if gradient is not None:
            gradient_row = {
                "object_id": id(gradient),
                "shape": list(map(int, gradient.shape)),
                "dtype": str(gradient.dtype),
                "device": str(gradient.device),
                "layout": str(gradient.layout),
                "version": int(gradient._version),
                "storage_data_ptr": _parameter_storage_pointer(gradient),
            }
            if _hash_tensor_bytes:
                gradient_row["tensor_sha256"] = tensor_sha256(
                    gradient, label=f"sealed parameter gradient {path}"
                )
        row = {
            "path": path,
            "object_id": key,
            "shape": list(map(int, value.shape)),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "layout": str(value.layout),
            "requires_grad": (
                "adapter_b_runtime_managed"
                if parameter and key in adapter_b_ids
                else bool(value.requires_grad)
            ),
            "version": int(value._version),
            "storage_data_ptr": _parameter_storage_pointer(value),
            "gradient": gradient_row,
        }
        if _hash_tensor_bytes:
            row["tensor_sha256"] = digest
        return row

    def root_rows(label: str, root: nn.Module) -> Mapping[str, Any]:
        modules: list[Mapping[str, Any]] = []
        edges: list[Mapping[str, Any]] = []
        parameters: list[Mapping[str, Any]] = []
        buffers: list[Mapping[str, Any]] = []
        queue: list[tuple[str, nn.Module]] = [(label, root)]
        expanded: set[int] = set()
        while queue:
            path, module = queue.pop(0)
            if id(module) in expanded:
                continue
            expanded.add(id(module))
            module_type = type(module)
            modules.append(
                {
                    "path": path,
                    "object_id": id(module),
                    "exact_type": (
                        f"{module_type.__module__}.{module_type.__qualname__}"
                    ),
                    "training": bool(module.training),
                    "behavior": dict(_module_behavior_binding(module)),
                }
            )
            for name, child in module._modules.items():
                child_path = f"{path}.{name}"
                edges.append(
                    {
                        "parent_path": path,
                        "name": name,
                        "child_path": child_path,
                        "child_object_id": None if child is None else id(child),
                        "child_exact_type": (
                            None
                            if child is None
                            else (
                                f"{type(child).__module__}."
                                f"{type(child).__qualname__}"
                            )
                        ),
                    }
                )
                if child is not None:
                    queue.append((child_path, child))
            for name, parameter_value in module._parameters.items():
                path_value = f"{path}.{name}"
                if parameter_value is None:
                    parameters.append({"path": path_value, "object_id": None})
                else:
                    parameters.append(
                        tensor_row(
                            parameter_value, path=path_value, parameter=True
                        )
                    )
            for name, buffer_value in module._buffers.items():
                path_value = f"{path}.{name}"
                if buffer_value is None:
                    buffers.append({"path": path_value, "object_id": None})
                else:
                    row = dict(
                        tensor_row(
                            buffer_value, path=path_value, parameter=False
                        )
                    )
                    row["persistent"] = name not in module._non_persistent_buffers_set
                    buffers.append(row)
        return {
            "root_object_id": id(root),
            "modules": modules,
            "module_edges": edges,
            "parameters": parameters,
            "buffers": buffers,
        }

    shared_step = getattr(diffusion, "shared_step", None)
    if not callable(shared_step):
        raise NativeRV2VHiddenVJPError("diffusion shared_step identity differs")
    function = getattr(shared_step, "__func__", shared_step)
    owner = getattr(shared_step, "__self__", None)
    value = {
        "schema_version": (
            "bernini-qmosaic-complete-model-runtime-seal-v1"
            if _hash_tensor_bytes
            else "bernini-qmosaic-complete-model-runtime-metadata-seal-v1"
        ),
        "diffusion_object_id": id(diffusion),
        "transformer_object_id": id(transformer),
        "shared_step_function_object_id": id(function),
        "shared_step_owner_object_id": None if owner is None else id(owner),
        "diffusion": root_rows("diffusion", diffusion),
        "transformer": root_rows("transformer", transformer),
        "every_registered_parameter_and_buffer_byte_hashed": _hash_tensor_bytes,
        "every_registered_module_edge_and_training_flag_bound": True,
    }
    if not _hash_tensor_bytes:
        value["every_registered_tensor_identity_storage_version_bound"] = True
        value["authoritative_publication_seal"] = False
    return MappingProxyType({**value, "digest": object_sha256(value)})


def _base_transformer_runtime_receipt(
    *, transformer: nn.Module, adapter_parameter_ids: frozenset[int]
) -> Mapping[str, Any]:
    """Seal the full transformer while treating A/B as rollback-managed state.

    This is used only by the private algebra fixture.  Every module edge,
    module training flag, base parameter byte and buffer byte remains bound;
    only the registered adapter values/gradients/requires-grad flags are
    masked so a legal adapter-only rollback can be distinguished from base
    model tampering.
    """

    if not isinstance(transformer, nn.Module):
        raise NativeRV2VHiddenVJPError("base transformer seal root differs")
    modules: list[Mapping[str, Any]] = []
    edges: list[Mapping[str, Any]] = []
    parameters: list[Mapping[str, Any]] = []
    buffers: list[Mapping[str, Any]] = []
    queue: list[tuple[str, nn.Module]] = [("transformer", transformer)]
    expanded: set[int] = set()
    while queue:
        path, module = queue.pop(0)
        if id(module) in expanded:
            continue
        expanded.add(id(module))
        modules.append(
            {
                "path": path,
                "object_id": id(module),
                "exact_type": (
                    f"{type(module).__module__}.{type(module).__qualname__}"
                ),
                "training": bool(module.training),
                "behavior": dict(_module_behavior_binding(module)),
            }
        )
        for name, child in module._modules.items():
            child_path = f"{path}.{name}"
            edges.append(
                {
                    "parent_path": path,
                    "name": name,
                    "child_path": child_path,
                    "child_object_id": None if child is None else id(child),
                    "child_exact_type": (
                        None
                        if child is None
                        else f"{type(child).__module__}.{type(child).__qualname__}"
                    ),
                }
            )
            if child is not None:
                queue.append((child_path, child))
        for name, parameter in module._parameters.items():
            parameter_path = f"{path}.{name}"
            if parameter is None:
                parameters.append({"path": parameter_path, "object_id": None})
            elif id(parameter) in adapter_parameter_ids:
                parameters.append(
                    {
                        "path": parameter_path,
                        "object_id": id(parameter),
                        "shape": list(map(int, parameter.shape)),
                        "dtype": str(parameter.dtype),
                        "device": str(parameter.device),
                        "layout": str(parameter.layout),
                        "storage_data_ptr": _parameter_storage_pointer(parameter),
                        "runtime_managed_adapter_value": True,
                    }
                )
            else:
                gradient = parameter.grad
                parameters.append(
                    {
                        "path": parameter_path,
                        "object_id": id(parameter),
                        "shape": list(map(int, parameter.shape)),
                        "dtype": str(parameter.dtype),
                        "device": str(parameter.device),
                        "layout": str(parameter.layout),
                        "requires_grad": bool(parameter.requires_grad),
                        "version": int(parameter._version),
                        "storage_data_ptr": _parameter_storage_pointer(parameter),
                        "tensor_sha256": tensor_sha256(
                            parameter, label=f"base transformer {parameter_path}"
                        ),
                        "gradient": (
                            None
                            if gradient is None
                            else {
                                "object_id": id(gradient),
                                "tensor_sha256": tensor_sha256(
                                    gradient,
                                    label=(
                                        "base transformer gradient "
                                        f"{parameter_path}"
                                    ),
                                ),
                            }
                        ),
                    }
                )
        for name, buffer in module._buffers.items():
            buffer_path = f"{path}.{name}"
            if buffer is None:
                buffers.append({"path": buffer_path, "object_id": None})
            else:
                buffers.append(
                    {
                        "path": buffer_path,
                        "object_id": id(buffer),
                        "shape": list(map(int, buffer.shape)),
                        "dtype": str(buffer.dtype),
                        "device": str(buffer.device),
                        "layout": str(buffer.layout),
                        "version": int(buffer._version),
                        "storage_data_ptr": _parameter_storage_pointer(buffer),
                        "tensor_sha256": tensor_sha256(
                            buffer, label=f"base transformer {buffer_path}"
                        ),
                        "persistent": name not in module._non_persistent_buffers_set,
                    }
                )
    observed_adapter_ids = {
        row["object_id"]
        for row in parameters
        if row.get("runtime_managed_adapter_value") is True
    }
    if observed_adapter_ids != set(adapter_parameter_ids):
        raise NativeRV2VHiddenVJPError(
            "base transformer adapter parameter closure differs"
        )
    value = {
        "schema_version": "bernini-qmosaic-base-transformer-runtime-seal-v1",
        "root_object_id": id(transformer),
        "modules": modules,
        "module_edges": edges,
        "base_parameters": parameters,
        "buffers": buffers,
        "adapter_values_masked_for_adapter_only_rollback": True,
        "base_model_restoration_authorized": False,
    }
    return MappingProxyType({**value, "digest": object_sha256(value)})


class NativeSharedStepSP4ReplayRunner:
    """Closed Bernini ``shared_step`` measurement/replay implementation.

    Unlike the old callback surface, this runner owns the source, four native
    image references, exact noisy target, two prompt embeddings, timestep,
    block-15 observer and SP4 process group for the complete session.
    """

    def __init__(
        self,
        *,
        diffusion: nn.Module,
        transformer: nn.Module,
        owner: AuthenticatedOwnerQuotientPacket,
        runtime_inputs: AuthenticatedEditorRuntimeInputPacket,
        action_handle: Core16ActionLoRAHandle,
        observer: Block15TargetSuffixObserver,
        sp4_collective: AuthenticatedSP4Collective,
        sp_rank: int,
        checkpoint_content: ValidatedCheckpointContentManifest,
    ) -> None:
        if (
            not isinstance(diffusion, nn.Module)
            or not isinstance(transformer, nn.Module)
            or action_handle.transformer is not transformer
            or observer.transformer is not transformer
            or type(sp_rank) is not int
            or sp_rank not in range(SP_SIZE)
            or not callable(getattr(diffusion, "shared_step", None))
            or type(owner) is not AuthenticatedOwnerQuotientPacket
            or type(runtime_inputs) is not AuthenticatedEditorRuntimeInputPacket
            or runtime_inputs._token is not _EDITOR_RUNTIME_INPUT_TOKEN
            or type(sp4_collective) is not AuthenticatedSP4Collective
            or sp4_collective._token is not _SP4_COLLECTIVE_TOKEN
            or sp4_collective.sp_rank != sp_rank
            or type(checkpoint_content) is not ValidatedCheckpointContentManifest
            or checkpoint_content._token is not _CHECKPOINT_CONTENT_TOKEN
        ):
            raise NativeRV2VHiddenVJPError("native shared_step/SP4 runtime differs")
        sp4_collective.assert_live()
        checkpoint_content.assert_live()
        runtime_inputs.assert_live(owner, checkpoint_content)
        device = next(
            parameter.device
            for _, parameter in action_handle.canonical_b_named_parameters()
        )
        signed = runtime_inputs.tensors
        source_latent = signed["source_latent"].to(device=device).clone().detach()
        image_references = tuple(
            signed[f"image_reference_{index}"].to(device=device).clone().detach()
            for index in range(native.REFERENCE_COUNT)
        )
        clean_latent = (
            signed["clean_latent"].to(device=device).clone().detach().requires_grad_(True)
        )
        initial_noise = (
            signed["official_initial_noise"].to(device=device).clone().detach()
        )
        x_sigma = (
            (1.0 - float(NATIVE_SIGMA)) * clean_latent
            + float(NATIVE_SIGMA) * initial_noise
        ).contiguous()
        action_condition = (
            signed["action_condition"].to(device=device).clone().detach()
        )
        noop_condition = signed["noop_condition"].to(device=device).clone().detach()
        timestep = signed["timestep"].to(device=device).clone().detach()
        prompt_condition_binding = {
            "action_prompt_sha256": owner.action_prompt_sha256,
            "noop_prompt_sha256": owner.noop_prompt_sha256,
            "action_condition_tensor_sha256": tensor_sha256(
                action_condition, label="signed action condition"
            ),
            "noop_condition_tensor_sha256": tensor_sha256(
                noop_condition, label="signed no-op condition"
            ),
            "tokenizer_receipt_digest": runtime_inputs.payload[
                "tokenizer_receipt_digest"
            ],
            "text_encoder_receipt_digest": runtime_inputs.payload[
                "text_encoder_receipt_digest"
            ],
        }
        refs = tuple(image_references)
        if len(refs) != native.REFERENCE_COUNT:
            raise NativeRV2VHiddenVJPError("native replay requires four image references")
        tensors = {
            "source_latent": source_latent,
            "clean_latent": clean_latent,
            "initial_noise": initial_noise,
            "x_sigma": x_sigma,
            "action_condition": action_condition,
            "noop_condition": noop_condition,
            "timestep": timestep,
            **{f"image_reference_{index}": value for index, value in enumerate(refs)},
        }
        if any(
            not isinstance(value, torch.Tensor)
            or value.device.type == "meta"
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
            for value in tensors.values()
        ):
            raise NativeRV2VHiddenVJPError("native replay runtime tensor differs")
        if (
            tuple(map(int, clean_latent.shape[:3])) != (1, 16, LATENT_PHASES)
            or clean_latent.shape != initial_noise.shape
            or clean_latent.shape != x_sigma.shape
            or clean_latent.dtype != torch.float32
            or initial_noise.dtype != torch.float32
            or x_sigma.dtype != torch.float32
            or not clean_latent.requires_grad
            or clean_latent.grad_fn is not None
            or initial_noise.requires_grad
            or initial_noise.grad_fn is not None
            or not x_sigma.requires_grad
            or x_sigma.grad_fn is None
            or timestep.dtype != torch.float32
            or timestep.numel() != 1
            or float(timestep.item()) != float(NATIVE_TIMESTEP)
            or action_condition.shape != noop_condition.shape
            or action_condition.ndim != 3
            or action_condition.requires_grad
            or noop_condition.requires_grad
            or action_condition.device != x_sigma.device
            or noop_condition.device != x_sigma.device
        ):
            raise NativeRV2VHiddenVJPError("native same-state prompt coordinate differs")
        expected_x = (
            (1.0 - float(NATIVE_SIGMA)) * clean_latent
            + float(NATIVE_SIGMA) * initial_noise
        )
        if not torch.allclose(
            x_sigma.detach(), expected_x.detach(), rtol=0.0, atol=2.0e-6
        ):
            raise NativeRV2VHiddenVJPError("x_sigma is not bound to clean/noise at sigma33")
        checkpoint_content_receipt_digest = checkpoint_content.receipt()["digest"]
        if not isinstance(prompt_condition_binding, Mapping) or set(
            prompt_condition_binding
        ) != {
            "action_prompt_sha256",
            "noop_prompt_sha256",
            "action_condition_tensor_sha256",
            "noop_condition_tensor_sha256",
            "tokenizer_receipt_digest",
            "text_encoder_receipt_digest",
        }:
            raise NativeRV2VHiddenVJPError("prompt condition binding fields differ")
        prompt_binding = dict(prompt_condition_binding)
        for name in (
            "action_prompt_sha256",
            "noop_prompt_sha256",
            "tokenizer_receipt_digest",
            "text_encoder_receipt_digest",
        ):
            _sha256(prompt_binding[name], label=f"prompt binding {name}")
        if (
            prompt_binding["action_condition_tensor_sha256"]
            != tensor_sha256(action_condition, label="bound action condition")
            or prompt_binding["noop_condition_tensor_sha256"]
            != tensor_sha256(noop_condition, label="bound no-op condition")
            or prompt_binding["action_prompt_sha256"]
            == prompt_binding["noop_prompt_sha256"]
        ):
            raise NativeRV2VHiddenVJPError("prompt condition tensor binding differs")
        if observer._handle is None or observer._pending is not None:  # noqa: SLF001
            raise NativeRV2VHiddenVJPError("block15 observer is not idle and installed")
        action_handle.assert_fixed_gauge()
        self.diffusion = diffusion
        self.transformer = transformer
        self.owner = owner
        self.runtime_inputs = runtime_inputs
        self.action_handle = action_handle
        self.observer = observer
        self.collective = sp4_collective
        self.sp_rank = sp_rank
        self.source_latent = source_latent
        self.image_references = refs
        self.clean_latent = clean_latent
        self.initial_noise = initial_noise
        self.x_sigma = x_sigma
        self.action_condition = action_condition
        self.noop_condition = noop_condition
        self.timestep = timestep
        self.checkpoint_content_receipt_digest = checkpoint_content_receipt_digest
        self.checkpoint_content = checkpoint_content
        self.prompt_condition_binding = MappingProxyType(prompt_binding)
        self._runtime_tensors = MappingProxyType(tensors)
        self._runtime_bindings = MappingProxyType(
            {
                name: _tensor_runtime_binding(value, label=f"runner {name}")
                for name, value in tensors.items()
            }
        )
        self._runtime_metadata_bindings = MappingProxyType(
            {
                name: _tensor_runtime_metadata_binding(
                    value, label=f"runner {name} metadata"
                )
                for name, value in tensors.items()
            }
        )
        self._diffusion_object = diffusion
        self._transformer_object = transformer
        self._action_handle_object = action_handle
        self._observer_object = observer
        self._collective_object = sp4_collective
        self._checkpoint_content_object = checkpoint_content
        self._owner_object = owner
        self._runtime_inputs_object = runtime_inputs
        self._observer_runtime_binding = MappingProxyType(
            {
                "observer_object_id": id(observer),
                "observer_transformer_object_id": id(observer.transformer),
                "observer_block_object_id": id(observer.block),
                "observer_handle_object_id": id(observer._handle),  # noqa: SLF001
                "observer_spatial_sketch": dict(
                    _tensor_runtime_binding(
                        observer.spatial_sketch,
                        label="runner observer spatial sketch",
                    )
                ),
                "observer_spatial_sketch_metadata": dict(
                    _tensor_runtime_metadata_binding(
                        observer.spatial_sketch,
                        label="runner observer spatial sketch metadata",
                    )
                ),
            }
        )
        adapter_b_ids = frozenset(
            id(parameter)
            for _, parameter in action_handle.canonical_b_named_parameters()
        )
        self._model_runtime_seal = _complete_model_runtime_receipt(
            diffusion=diffusion,
            transformer=transformer,
            adapter_b_ids=adapter_b_ids,
        )
        self._model_runtime_metadata_seal = _complete_model_runtime_receipt(
            diffusion=diffusion,
            transformer=transformer,
            adapter_b_ids=adapter_b_ids,
            _hash_tensor_bytes=False,
        )
        self._adapter_b_ids = adapter_b_ids
        self._assert_runtime_live()

    def contract_receipt(self, *, deep: bool = True) -> Mapping[str, Any]:
        if type(deep) is not bool:
            raise NativeRV2VHiddenVJPError("runtime contract seal mode differs")
        self._assert_runtime_live(deep=deep)
        value = {
            "schema_version": REPLAY_SESSION_SCHEMA_VERSION,
            "native_forward": "source_self_native_ref_contrastive_v3.forward_native_target_branch",
            "vendor_call": "WanDiffusion.shared_step(model_id=transformer_1)",
            "branch": "VI",
            "native_schedule_index": NATIVE_SCHEDULE_INDEX,
            "native_timestep": NATIVE_TIMESTEP,
            "native_sigma_float64_hex": float(NATIVE_SIGMA).hex(),
            "sp_rank": self.sp_rank,
            "sp_size": SP_SIZE,
            "hook_coordinate": "transformer_1.blocks.15.output.target_suffix",
            "checkpoint_content_receipt_digest": self.checkpoint_content_receipt_digest,
            "authenticated_runtime_input_receipt_digest": (
                self.runtime_inputs.receipt()["digest"]
            ),
            "prompt_condition_binding_digest": object_sha256(
                dict(self.prompt_condition_binding)
            ),
            "transformer_object_id": id(self.transformer),
            "diffusion_object_id": id(self.diffusion),
            "complete_model_runtime_seal_digest": self._model_runtime_seal[
                "digest"
            ],
            "authenticated_sp4_collective_receipt_digest": (
                self.collective.receipt()["digest"]
            ),
            "action_handle_gauge_digest": self.action_handle.gauge_receipt["digest"],
            "runtime_tensor_bindings": {
                name: dict(binding) for name, binding in self._runtime_bindings.items()
            },
            "generic_replay_callback_exposed": False,
            "real_auh_runtime_validated": False,
        }
        return {**value, "digest": object_sha256(value)}

    def assert_terminal_runtime_live(self) -> Mapping[str, Any]:
        """Run the mandatory full-byte pre-publication integrity seal."""

        self._assert_runtime_live(deep=True)
        value = {
            "schema_version": "bernini-qmosaic-terminal-full-runtime-seal-v1",
            "complete_model_runtime_seal_digest": self._model_runtime_seal[
                "digest"
            ],
            "checkpoint_content_receipt_digest": (
                self.checkpoint_content_receipt_digest
            ),
            "authenticated_runtime_input_receipt_digest": (
                self.runtime_inputs.receipt()["digest"]
            ),
            "deep_full_byte_revalidated": True,
            "every_model_parameter_and_buffer_byte_revalidated": True,
            "checkpoint_tree_revalidated": True,
            "signed_runtime_input_revalidated": True,
            "publication_authority": "integrity_only_no_semantic_or_update_authority",
        }
        return {**value, "digest": object_sha256(value)}

    @staticmethod
    def canary_contract() -> Mapping[str, Any]:
        value = {
            "schema_version": "bernini-qmosaic-native-shared-step-sp4-canary-v1",
            "required_world_size": 4,
            "required_ulysses_size": 4,
            "required_branch": "VI",
            "required_schedule_index": 33,
            "required_timestep": 516,
            "required_frame_count": 81,
            "required_latent_phases": 21,
            "shared_step_model_id": "transformer_1",
            "block15_target_suffix_hook": True,
            "measurement_all_reduce": "SUM_detached",
            "replay_all_reduce": False,
            "rank_score_divisor_before_vjp": 4,
            "lora_b_requires_signed_exact81_gate": True,
            "real_auh_runtime_validated": False,
        }
        return {**value, "digest": object_sha256(value)}

    @staticmethod
    def functional_preservation_contract() -> Mapping[str, Any]:
        value = {
            "schema_version": FUNCTIONAL_PRESERVATION_SCHEMA_VERSION,
            "identity_primary_constraint": "paired_source_native_functional_cone",
            "i_axis_role": "weak_slab_only",
            "rademacher_seeds": list(PRESERVATION_RADEMACHER_SEEDS),
            "functional_rows": [
                {
                    "functional_id": WEAK_I_AXIS_FUNCTIONAL_ID,
                    "qp_family": "identity",
                    "native_contrast": "I-minus-none",
                    "weak_i_axis_slab": True,
                }
            ]
            + [
                {
                    "functional_id": functional_id,
                    "qp_family": family,
                    "native_contrast": contrast,
                }
                for functional_id, (family, contrast) in FUNCTIONAL_PRESERVATION_SPECS.items()
            ],
            "each_rademacher_row_is_an_independent_qp_slab": True,
            "row_averaging": False,
            "same_source_x_sigma_timestep_query_seed_model": True,
            "same_native_state_as_action_clean_vjp": True,
            "qp_infeasible_policy": "byte_exact_zero_no_update",
            "real_auh_runtime_validated": False,
        }
        return {**value, "digest": object_sha256(value)}

    def _assert_runtime_live(self, *, deep: bool = True) -> None:
        if type(deep) is not bool:
            raise NativeRV2VHiddenVJPError("runtime seal mode differs")
        if (
            self.diffusion is not self._diffusion_object
            or self.transformer is not self._transformer_object
            or self.action_handle is not self._action_handle_object
            or self.observer is not self._observer_object
            or self.collective is not self._collective_object
            or self.checkpoint_content is not self._checkpoint_content_object
            or self.owner is not self._owner_object
            or self.runtime_inputs is not self._runtime_inputs_object
            or id(self.observer) != self._observer_runtime_binding["observer_object_id"]
            or self.observer.transformer is not self.transformer
            or id(self.observer.block)
            != self._observer_runtime_binding["observer_block_object_id"]
            or self.observer._handle is None  # noqa: SLF001
            or id(self.observer._handle)  # noqa: SLF001
            != self._observer_runtime_binding["observer_handle_object_id"]
            or self.observer._pending is not None  # noqa: SLF001
            or self.action_handle.transformer is not self.transformer
            or self.collective.sp_rank != self.sp_rank
        ):
            raise NativeRuntimeSealChangedError(
                "native shared_step/model object identity changed; process is poisoned"
            )
        try:
            observed_metadata = _complete_model_runtime_receipt(
                diffusion=self.diffusion,
                transformer=self.transformer,
                adapter_b_ids=self._adapter_b_ids,
                _hash_tensor_bytes=False,
            )
        except NativeRV2VHiddenVJPError as error:
            raise NativeRuntimeSealChangedError(
                "diffusion/transformer runtime seal cannot be re-read; process is poisoned"
            ) from error
        if dict(observed_metadata) != dict(self._model_runtime_metadata_seal):
            raise NativeRuntimeSealChangedError(
                "diffusion/transformer runtime metadata changed; process is poisoned"
            )
        self.collective.assert_live()
        _assert_tensor_runtime_metadata_binding(
            self.observer.spatial_sketch,
            self._observer_runtime_binding["observer_spatial_sketch_metadata"],
            label="runner live observer spatial sketch metadata",
        )
        for name, value in self._runtime_tensors.items():
            _assert_tensor_runtime_metadata_binding(
                value,
                self._runtime_metadata_bindings[name],
                label=f"runner live {name} metadata",
            )
        if not deep:
            return
        try:
            observed_model = _complete_model_runtime_receipt(
                diffusion=self.diffusion,
                transformer=self.transformer,
                adapter_b_ids=self._adapter_b_ids,
            )
        except NativeRV2VHiddenVJPError as error:
            raise NativeRuntimeSealChangedError(
                "diffusion/transformer full runtime seal cannot be re-read; "
                "process is poisoned"
            ) from error
        if dict(observed_model) != dict(self._model_runtime_seal):
            raise NativeRuntimeSealChangedError(
                "diffusion/transformer full runtime seal changed; process is poisoned"
            )
        # Full artifact and tensor bytes are revalidated only at authoritative
        # session boundaries.  A terminal failure prevents publication.
        self.checkpoint_content.assert_live()
        self.runtime_inputs.assert_live(self.owner, self.checkpoint_content)
        _assert_tensor_runtime_binding(
            self.observer.spatial_sketch,
            self._observer_runtime_binding["observer_spatial_sketch"],
            label="runner live observer spatial sketch",
            require_same_object=True,
        )
        for name, value in self._runtime_tensors.items():
            _assert_tensor_runtime_binding(
                value,
                self._runtime_bindings[name],
                label=f"runner live {name}",
                require_same_object=True,
            )

    def _forward_local(self, *, role: str, adapter_enabled: bool, detach: bool) -> torch.Tensor:
        if role not in ("action", "noop") or type(adapter_enabled) is not bool:
            raise NativeRV2VHiddenVJPError("native replay role/route differs")
        self._assert_runtime_live(deep=False)
        pack = native.build_native_rv2v_pack(
            self.transformer,
            donor_video=self.source_latent,
            image_references=self.image_references,
            noisy_target=self.x_sigma,
        )
        branch = pack.video_image
        patch_height = int(self.x_sigma.shape[3]) // 2
        patch_width = int(self.x_sigma.shape[4]) // 2
        layout = layout_from_native_branch(
            branch,
            patch_height=patch_height,
            patch_width=patch_width,
            sp_rank=self.sp_rank,
        )
        condition = self._vendor_condition(role)
        with native_bridge._route_context(  # noqa: SLF001 - pinned bridge integration
            self.action_handle,
            transformer=self.transformer,
            branch=branch,
            sequence_parallel_rank=self.sp_rank,
            sequence_parallel_size=SP_SIZE,
            sigma_schedule_index=NATIVE_SCHEDULE_INDEX,
            enabled=adapter_enabled,
        ):
            with self.observer.capture(
                role=f"{role}-{'measure' if detach else 'replay'}",
                layout=layout,
                detach=detach,
            ) as holder:
                native.forward_native_target_branch(
                    self.diffusion,
                    branch,
                    timestep=self.timestep,
                    cond_embeds=condition,
                )
        if len(holder) != 1:
            raise NativeRV2VHiddenVJPError("native block15 capture count differs")
        self._assert_runtime_live(deep=False)
        return holder[0].tensor

    def _vendor_condition(self, role: str) -> torch.Tensor:
        """Cast the signed FP32 prompt packet only at the Bernini call edge."""

        if role not in ("action", "noop"):
            raise NativeRV2VHiddenVJPError("native condition role differs")
        signed = self.action_condition if role == "action" else self.noop_condition
        patch_embedding = getattr(self.transformer, "patch_embedding", None)
        weight = getattr(patch_embedding, "weight", None)
        if (
            not isinstance(signed, torch.Tensor)
            or signed.dtype != torch.float32
            or signed.requires_grad
            or signed.grad_fn is not None
            or not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16
        ):
            raise NativeRV2VHiddenVJPError(
                "signed FP32/Bernini BF16 condition boundary differs"
            )
        vendor = signed.to(device=self.x_sigma.device, dtype=weight.dtype)
        if (
            vendor.dtype != torch.bfloat16
            or vendor.device != self.x_sigma.device
            or vendor.requires_grad
            or vendor.grad_fn is not None
            or not bool(torch.isfinite(vendor).all().item())
        ):
            raise NativeRV2VHiddenVJPError("Bernini vendor condition cast differs")
        return vendor.contiguous()

    def _measure_role(self, role: str) -> tuple[torch.Tensor, torch.Tensor]:
        # The detached measurement and the later VJP replay must traverse the
        # same vendor grad-mode branch.  Bernini/SDPA may select a different
        # implementation under ``no_grad``; comparing or differentiating a
        # value sampled from that other path would confound the action score.
        # Keep the native graph only for this call, detach the observed sketch,
        # and release the graph before the collective or the next role.
        with torch.enable_grad():
            connected = self._forward_local(
                role=role, adapter_enabled=False, detach=False
            )
        local = connected.detach().contiguous()
        del connected
        global_value = local.clone().contiguous()
        self.collective.all_reduce_sum(global_value)
        global_value = global_value.detach().contiguous()
        return local, global_value

    def seal_editor_packet(
        self, owner: AuthenticatedOwnerQuotientPacket
    ) -> tuple[EditorSameStatePromptPacket, RuntimeOwnedReplaySession]:
        owner.assert_live()
        if (
            owner is not self.owner
            or owner.source_iid == ""
            or owner.action_prompt_sha256 == owner.noop_prompt_sha256
            or self.prompt_condition_binding["action_prompt_sha256"]
            != owner.action_prompt_sha256
            or self.prompt_condition_binding["noop_prompt_sha256"]
            != owner.noop_prompt_sha256
        ):
            raise NativeRV2VHiddenVJPError("owner/editor prompt identity differs")
        generator = torch.Generator(device="cpu")
        owner_query_seed = self.runtime_inputs.payload["owner_query_seed"]
        editor_noise_seed = self.runtime_inputs.payload["editor_noise_seed"]
        if (
            owner_query_seed != owner.query_seed
            or editor_noise_seed
            != editor_noise_seed_from_owner_query_seed(owner.query_seed)
            or editor_noise_seed == owner.query_seed
        ):
            raise NativeRV2VHiddenVJPError(
                "owner/editor Gaussian seed domains are not separated"
            )
        generator.manual_seed(editor_noise_seed)
        official_noise = torch.randn(
            tuple(map(int, self.initial_noise.shape)),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device=self.initial_noise.device)
        if not torch.equal(official_noise, self.initial_noise.detach()):
            raise NativeRV2VHiddenVJPError(
                "editor noise is not the official CPU-generator query seed"
            )
        local_action, action = self._measure_role("action")
        local_noop, noop = self._measure_role("noop")
        # This is the rank-independent consensus identity.  In particular it
        # excludes the local shard, CUDA ordinal, Python object IDs and SP
        # rank.  Those belong in the rank-local editor receipt below.
        shared_state_binding_digest = object_sha256(
            {
                "schema_version": "bernini-qmosaic-editor-shared-state-v1",
                "owner_packet_receipt_digest": owner.receipt()["digest"],
                "cell_id": owner.cell_id,
                "owner_query_seed": owner.query_seed,
                "editor_noise_seed": editor_noise_seed,
                "owner_editor_noise_seed_shared": False,
                "source_iid": owner.source_iid,
                "source_video_sha256": owner.source_video_sha256,
                "action_prompt_sha256": owner.action_prompt_sha256,
                "noop_prompt_sha256": owner.noop_prompt_sha256,
                "native_schedule_index": NATIVE_SCHEDULE_INDEX,
                "native_timestep": NATIVE_TIMESTEP,
                "native_sigma_float64_hex": float(NATIVE_SIGMA).hex(),
                "checkpoint_content_receipt_digest": (
                    self.checkpoint_content_receipt_digest
                ),
                "authenticated_runtime_input_receipt_digest": (
                    self.runtime_inputs.receipt()["digest"]
                ),
                "prompt_condition_binding_digest": object_sha256(
                    dict(self.prompt_condition_binding)
                ),
                "runtime_tensor_values": {
                    name: {
                        "shape": binding["shape"],
                        "dtype": binding["dtype"],
                        "tensor_sha256": binding["tensor_sha256"],
                    }
                    for name, binding in self._runtime_bindings.items()
                },
                "global_action_measurement_sha256": tensor_sha256(
                    action, label="shared global action measurement"
                ),
                "global_noop_measurement_sha256": tensor_sha256(
                    noop, label="shared global no-op measurement"
                ),
            }
        )
        bindings = MappingProxyType(
            {
                "native_schedule_index": NATIVE_SCHEDULE_INDEX,
                "native_timestep": NATIVE_TIMESTEP,
                "native_sigma": NATIVE_SIGMA,
                "sp_rank": self.sp_rank,
                "sp_size": SP_SIZE,
                "shared_state_binding_digest": shared_state_binding_digest,
                "runner_contract_digest": self.contract_receipt(deep=False)["digest"],
                "checkpoint_content_receipt_digest": self.checkpoint_content_receipt_digest,
                "model_proof_runtime_owned": True,
                "owner_query_seed": owner.query_seed,
                "editor_noise_seed": editor_noise_seed,
                "owner_editor_noise_seed_shared": False,
                "official_cpu_generator_gaussian": True,
                "prompt_condition_binding_digest": object_sha256(
                    dict(self.prompt_condition_binding)
                ),
                "action_condition_tensor_sha256": self.prompt_condition_binding[
                    "action_condition_tensor_sha256"
                ],
                "noop_condition_tensor_sha256": self.prompt_condition_binding[
                    "noop_condition_tensor_sha256"
                ],
                "same_x_sigma_object_id": id(self.x_sigma),
                "same_timestep_object_id": id(self.timestep),
                "sealed_action_measurement_sha256": tensor_sha256(
                    action, label="sealed action measurement"
                ),
                "sealed_noop_measurement_sha256": tensor_sha256(
                    noop, label="sealed no-op measurement"
                ),
                "sealed_local_action_measurement_sha256": tensor_sha256(
                    local_action, label="sealed local action measurement"
                ),
                "sealed_local_noop_measurement_sha256": tensor_sha256(
                    local_noop, label="sealed local no-op measurement"
                ),
            }
        )
        packet = EditorSameStatePromptPacket(
            cell_id=owner.cell_id,
            query_seed=owner.query_seed,
            sp_rank=self.sp_rank,
            branch_name="VI",
            source_iid=owner.source_iid,
            source_video_sha256=owner.source_video_sha256,
            action_prompt_sha256=owner.action_prompt_sha256,
            noop_prompt_sha256=owner.noop_prompt_sha256,
            action_measurement=action,
            noop_measurement=noop,
            local_action_measurement=local_action,
            local_noop_measurement=local_noop,
            shared_state_binding_digest=shared_state_binding_digest,
            bindings=bindings,
            _runtime_tensors=self._runtime_tensors,
            _runtime_tensor_bindings=self._runtime_bindings,
            _runtime_owner_digest=owner.receipt()["digest"],
        )
        object.__setattr__(packet, "_token", _EDITOR_PACKET_TOKEN)
        packet.assert_live(owner)
        session_unsigned = {
            "schema_version": REPLAY_SESSION_SCHEMA_VERSION,
            "editor_packet_digest": packet.receipt()["digest"],
            "runner_contract_digest": self.contract_receipt(deep=False)["digest"],
        }
        session = RuntimeOwnedReplaySession(
            editor_packet=packet,
            session_digest=object_sha256(session_unsigned),
            _runner=self,
        )
        object.__setattr__(session, "_token", _REPLAY_SESSION_TOKEN)
        session.assert_live(owner)
        return packet, session

    def _replay_runtime_owned(self, *, role: str, adapter_enabled: bool) -> torch.Tensor:
        return self._forward_local(
            role=role, adapter_enabled=adapter_enabled, detach=False
        )

    def sum_rank_local_vjp(
        self, row: "RankLocalVJPRow"
    ) -> "SP4SummedVJPRow":
        """Gather four authenticated rank rows on the live Ulysses group.

        The wire format is deliberately plain and reconstructed through the
        rank-row validator on every peer.  Python capability tokens are never
        trusted across pickle boundaries.
        """

        self._assert_runtime_live(deep=False)
        if type(row) is not RankLocalVJPRow:
            raise NativeRV2VHiddenVJPError(
                "native SP4 SUM requires one sealed rank-local VJP"
            )
        row.assert_live()
        if row.sp_rank != self.sp_rank or row.query_seed != self.owner.query_seed:
            raise NativeRV2VHiddenVJPError(
                "native SP4 SUM local rank/query provenance differs"
            )
        if row.vjp_target == "clean_latent":
            wire_values: Any = row.values.detach().float().cpu().contiguous()
        else:
            wire_values = {
                name: row.values[name].detach().float().cpu().contiguous()
                for name in CANONICAL_B_PARAMETER_NAMES
            }
        wire = {
            "query_seed": row.query_seed,
            "sp_rank": row.sp_rank,
            "vjp_target": row.vjp_target,
            "values": wire_values,
            "score_cotangent_receipt_digest": (
                row.score_cotangent_receipt_digest
            ),
            "editor_packet_receipt_digest": row.editor_packet_receipt_digest,
            "global_cotangent_identity_digest": (
                row.global_cotangent_identity_digest
            ),
            "value_sha256": row.value_sha256,
            "value_norm": row.value_norm,
            "replay_max_abs": row.replay_max_abs,
            "parameter_state_sha256": row.parameter_state_sha256,
            "rank_row_receipt_digest": row.receipt()["digest"],
        }
        gathered = self.collective.all_gather_object(wire)
        required_fields = set(wire)
        rows: list[RankLocalVJPRow] = []
        for expected_rank, raw in enumerate(gathered):
            if (
                not isinstance(raw, Mapping)
                or set(raw) != required_fields
                or raw.get("sp_rank") != expected_rank
            ):
                raise NativeRV2VHiddenVJPError(
                    "native SP4 SUM gathered rank-row wire differs"
                )
            candidate = _seal_rank_local_vjp_row(
                RankLocalVJPRow(
                    query_seed=raw["query_seed"],
                    sp_rank=raw["sp_rank"],
                    vjp_target=raw["vjp_target"],
                    values=raw["values"],
                    score_cotangent_receipt_digest=raw[
                        "score_cotangent_receipt_digest"
                    ],
                    editor_packet_receipt_digest=raw[
                        "editor_packet_receipt_digest"
                    ],
                    global_cotangent_identity_digest=raw[
                        "global_cotangent_identity_digest"
                    ],
                    value_sha256=raw["value_sha256"],
                    value_norm=raw["value_norm"],
                    replay_max_abs=raw["replay_max_abs"],
                    parameter_state_sha256=raw["parameter_state_sha256"],
                )
            )
            if candidate.receipt()["digest"] != raw["rank_row_receipt_digest"]:
                raise NativeRV2VHiddenVJPError(
                    "native SP4 SUM rank-row receipt changed in transit"
                )
            rows.append(candidate)
        result = _sum_rank_local_vjp_rows_unsafe_for_test(rows)
        peer_results = self.collective.all_gather_object(
            result.receipt()["digest"]
        )
        if peer_results != (result.receipt()["digest"],) * SP_SIZE:
            raise NativeRV2VHiddenVJPError(
                "native SP4 SUM result lacks four-rank consensus"
            )
        self._assert_runtime_live(deep=False)
        return result

    def _packed_x_sigma_target(self) -> torch.Tensor:
        value = self.x_sigma
        batch, channels, phases, height, width = map(int, value.shape)
        if (
            (batch, channels, phases) != (1, 16, LATENT_PHASES)
            or height % 2
            or width % 2
        ):
            raise NativeRV2VHiddenVJPError(
                "predicted-clean physical latent geometry differs"
            )
        return (
            value.reshape(batch, channels, phases, height // 2, 2, width // 2, 2)
            .permute(0, 2, 3, 5, 4, 6, 1)
            .reshape(
                batch,
                phases * (height // 2) * (width // 2),
                PACKED_PREDICTION_DIM,
            )
            .float()
            .contiguous()
        )

    def _forward_functional_predicted_clean(
        self, *, role: str, branch_name: str
    ) -> torch.Tensor:
        if role not in ("action", "noop") or branch_name not in BRANCH_NAMES:
            raise NativeRV2VHiddenVJPError(
                "functional native role/branch differs"
            )
        self._assert_runtime_live(deep=False)
        pack = native.build_native_rv2v_pack(
            self.transformer,
            donor_video=self.source_latent,
            image_references=self.image_references,
            noisy_target=self.x_sigma,
        )
        branch = {
            "none": pack.none,
            "V": pack.video,
            "I": pack.image,
            "VI": pack.video_image,
        }[branch_name]
        patch_height = int(self.x_sigma.shape[3]) // 2
        patch_width = int(self.x_sigma.shape[4]) // 2
        layout = layout_from_native_branch(
            branch,
            patch_height=patch_height,
            patch_width=patch_width,
            sp_rank=self.sp_rank,
        )
        condition = self._vendor_condition(role)
        with native_bridge._route_context(  # noqa: SLF001 - pinned integration
            self.action_handle,
            transformer=self.transformer,
            branch=branch,
            sequence_parallel_rank=self.sp_rank,
            sequence_parallel_size=SP_SIZE,
            sigma_schedule_index=NATIVE_SCHEDULE_INDEX,
            enabled=True,
        ):
            with self.observer.capture(
                role=f"functional:{branch_name}:{role}",
                layout=layout,
                detach=False,
            ) as holder:
                prediction = native.forward_native_target_branch(
                    self.diffusion,
                    branch,
                    timestep=self.timestep,
                    cond_embeds=condition,
                )
        if len(holder) != 1:
            raise NativeRV2VHiddenVJPError(
                "functional block15 observer capture differs"
            )
        noisy = self._packed_x_sigma_target()
        predicted_clean = (
            noisy - torch.tensor(
                float(NATIVE_SIGMA),
                dtype=torch.float32,
                device=prediction.device,
            )
            * prediction.float()
        ).contiguous()
        if (
            predicted_clean.shape != noisy.shape
            or not predicted_clean.requires_grad
            or predicted_clean.grad_fn is None
            or not bool(torch.isfinite(predicted_clean).all().item())
        ):
            raise NativeRV2VHiddenVJPError(
                "native predicted-clean packed velocity coordinate differs"
            )
        self._assert_runtime_live(deep=False)
        return predicted_clean

    def collect_functional_preservation_cone(
        self,
        *,
        owner: AuthenticatedOwnerQuotientPacket,
        editor: EditorSameStatePromptPacket,
        score_packet: DetachedScoreCotangent,
        clean_vjp_row: SP4SummedVJPRow,
        direction_gate: ValidatedExact81DirectionGate,
        maximum_absolute_dot_by_functional: Mapping[str, float],
    ) -> PairedFunctionalPreservationCone:
        """Collect every preservation row live; no caller tensor row is accepted."""

        required_ids = {
            WEAK_I_AXIS_FUNCTIONAL_ID,
            *FUNCTIONAL_PRESERVATION_SPECS.keys(),
        }
        if (
            not isinstance(maximum_absolute_dot_by_functional, Mapping)
            or set(maximum_absolute_dot_by_functional) != required_ids
        ):
            raise NativeRV2VHiddenVJPError(
                "functional bound policy must cover the exact closed row registry"
            )
        bounds = {
            functional_id: float(maximum_absolute_dot_by_functional[functional_id])
            for functional_id in required_ids
        }
        if any(
            isinstance(maximum_absolute_dot_by_functional[name], bool)
            or not math.isfinite(value)
            or value < 0.0
            for name, value in bounds.items()
        ):
            raise NativeRV2VHiddenVJPError(
                "functional preservation bounds differ"
            )
        nonweak = [
            value
            for name, value in bounds.items()
            if name != WEAK_I_AXIS_FUNCTIONAL_ID
        ]
        if bounds[WEAK_I_AXIS_FUNCTIONAL_ID] < 4.0 * max(nonweak):
            raise NativeRV2VHiddenVJPError(
                "weak I-axis slab must be at least four times looser"
            )
        self._assert_runtime_live(deep=False)
        owner.assert_live()
        editor.assert_live(owner)
        score_packet.assert_live(owner, editor)
        clean_vjp_row.assert_live()
        direction_gate.assert_live(
            owner=owner,
            editor=editor,
            score_packet=score_packet,
            clean_vjp_row=clean_vjp_row,
        )
        local_editor_digest = editor.receipt()["digest"]
        local_score_digest = score_packet.receipt()["digest"]
        editor_digests = self.collective.all_gather_object(local_editor_digest)
        score_digests = self.collective.all_gather_object(local_score_digest)
        if (
            editor_digests != clean_vjp_row.rank_editor_packet_receipt_digests
            or score_digests
            != clean_vjp_row.rank_score_cotangent_receipt_digests
            or clean_vjp_row.global_cotangent_identity_digest
            != score_packet.global_cotangent_identity_digest
            or clean_vjp_row.query_seed != owner.query_seed
            or clean_vjp_row.parameter_state_sha256
            != self.action_handle.b_parameter_state_sha256()
        ):
            raise NativeRV2VHiddenVJPError(
                "functional collector differs from clean SP4 provenance"
            )
        parameters = self.action_handle.canonical_b_named_parameters()
        rows: list[FunctionalPreservationVJPRow] = []
        ordered_functionals = (
            WEAK_I_AXIS_FUNCTIONAL_ID,
            *FUNCTIONAL_PRESERVATION_SPECS.keys(),
        )
        patch_height = int(self.x_sigma.shape[3]) // 2
        patch_width = int(self.x_sigma.shape[4]) // 2
        for functional_id in ordered_functionals:
            for seed in PRESERVATION_RADEMACHER_SEEDS:
                snapshot = self.action_handle.adapter_state_snapshot()
                fatal_runtime_tamper = False
                try:
                    kwargs: dict[str, Any] = {
                        "noop_predicted_clean": self._forward_functional_predicted_clean(
                            role="noop", branch_name="VI"
                        )
                    }
                    if functional_id == WEAK_I_AXIS_FUNCTIONAL_ID:
                        kwargs["image_predicted_clean"] = (
                            self._forward_functional_predicted_clean(
                                role="action", branch_name="I"
                            )
                        )
                        kwargs["none_predicted_clean"] = (
                            self._forward_functional_predicted_clean(
                                role="action", branch_name="none"
                            )
                        )
                    elif functional_id == "source_video_v_none_sensitivity":
                        kwargs["video_predicted_clean"] = (
                            self._forward_functional_predicted_clean(
                                role="action", branch_name="V"
                            )
                        )
                        kwargs["none_predicted_clean"] = (
                            self._forward_functional_predicted_clean(
                                role="action", branch_name="none"
                            )
                        )
                    elif functional_id != "noop_predicted_clean_invariance":
                        kwargs["action_predicted_clean"] = (
                            self._forward_functional_predicted_clean(
                                role="action", branch_name="VI"
                            )
                        )
                    feature = paired_functional_preservation_feature(
                        functional_id,
                        patch_height=patch_height,
                        patch_width=patch_width,
                        **kwargs,
                    )
                    scalar = fixed_rademacher_functional_scalar(
                        feature, rademacher_seed=seed
                    ) / float(SP_SIZE)
                    gradients = torch.autograd.grad(
                        scalar,
                        tuple(parameter for _, parameter in parameters),
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=False,
                    )
                    local_values = tuple(
                        gradient.detach().float().contiguous()
                        for gradient in gradients
                    )
                    local_digest = _named_tensor_sha256(
                        tuple(
                            (name, value)
                            for (name, _), value in zip(parameters, local_values)
                        ),
                        label=(
                            f"functional local {functional_id} seed {seed}"
                        ),
                    )
                    local_rank_receipt = object_sha256(
                        {
                            "schema_version": (
                                "bernini-qmosaic-functional-rank-vjp-v1"
                            ),
                            "functional_id": functional_id,
                            "rademacher_seed": seed,
                            "sp_rank": self.sp_rank,
                            "editor_packet_receipt_digest": local_editor_digest,
                            "score_cotangent_receipt_digest": local_score_digest,
                            "global_cotangent_identity_digest": (
                                score_packet.global_cotangent_identity_digest
                            ),
                            "local_value_sha256": local_digest,
                        }
                    )
                    rank_receipts = self.collective.all_gather_object(
                        local_rank_receipt
                    )
                    summed_values = []
                    for value in local_values:
                        summed = value.clone()
                        self.collective.all_reduce_sum(summed)
                        summed_values.append(
                            summed.detach().float().cpu().contiguous()
                        )
                    mapping = MappingProxyType(
                        {
                            name: value
                            for (name, _), value in zip(
                                parameters, summed_values
                            )
                        }
                    )
                except NativeRuntimeSealChangedError:
                    fatal_runtime_tamper = True
                    raise
                finally:
                    if not fatal_runtime_tamper:
                        # Detect a forward that mutated the base and then
                        # raised before its normal post-forward assertion.
                        # This check must precede every adapter rollback.
                        self._assert_runtime_live(deep=False)
                        changed = not self.action_handle.adapter_state_matches(
                            snapshot
                        )
                        if changed:
                            self.action_handle.restore_adapter_state(snapshot)
                        if not self.action_handle.adapter_state_matches(snapshot):
                            raise NativeRV2VHiddenVJPError(
                                "functional adapter rollback audit failed"
                            )
                        self._assert_runtime_live(deep=False)
                        if changed:
                            raise NativeRV2VHiddenVJPError(
                                "functional VJP mutated adapter state"
                            )
                rows.append(
                    _seal_functional_preservation_row_from_runtime_sp4(
                        owner=owner,
                        editor=editor,
                        score_packet=score_packet,
                        clean_vjp_row=clean_vjp_row,
                        functional_id=functional_id,
                        rademacher_seed=seed,
                        values=mapping,
                        maximum_absolute_dot=bounds[functional_id],
                        sp4_editor_packet_receipt_digests=editor_digests,
                        sp4_rank_vjp_receipt_digests=rank_receipts,
                        parameter_state_sha256=(
                            self.action_handle.b_parameter_state_sha256()
                        ),
                    )
                )
        cone = _seal_paired_functional_preservation_cone_from_runtime(
            owner=owner,
            editor=editor,
            score_packet=score_packet,
            clean_vjp_row=clean_vjp_row,
            rows=rows,
        )
        self._assert_runtime_live(deep=False)
        return cone


@dataclass(frozen=True)
class DetachedScoreCotangent:
    query_seed: int
    score: float
    score_divisor: int
    action_measurement: torch.Tensor = field(repr=False)
    noop_measurement: torch.Tensor = field(repr=False)
    action_cotangent: torch.Tensor = field(repr=False)
    noop_cotangent: torch.Tensor = field(repr=False)
    scorer_id: str
    action_measurement_sha256: str
    noop_measurement_sha256: str
    action_cotangent_sha256: str
    noop_cotangent_sha256: str
    owner_packet_receipt_digest: str
    editor_packet_receipt_digest: str
    global_cotangent_identity_digest: str
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": COTANGENT_SCHEMA_VERSION,
            "query_seed": self.query_seed,
            "score": self.score,
            "score_divisor": self.score_divisor,
            "normalization": "each_rank_differentiates_score_divided_by_sp4",
            "post_sum_divisor": None,
            "scorer_id": self.scorer_id,
            "action_measurement_sha256": self.action_measurement_sha256,
            "noop_measurement_sha256": self.noop_measurement_sha256,
            "action_cotangent_sha256": self.action_cotangent_sha256,
            "noop_cotangent_sha256": self.noop_cotangent_sha256,
            "owner_packet_receipt_digest": self.owner_packet_receipt_digest,
            "editor_packet_receipt_digest": self.editor_packet_receipt_digest,
            "global_cotangent_identity_digest": (
                self.global_cotangent_identity_digest
            ),
            "action_is_exact_negative_noop": torch.equal(
                self.action_cotangent, -self.noop_cotangent
            ),
            "detached_measurement_then_serial_graph_replay": True,
            "authenticated_owner_quotient_only": self.scorer_id.startswith(
                "authenticated-owner:"
            ),
            "generic_scorer_callback_consumed": False,
        }
        return {**value, "digest": object_sha256(value)}

    def assert_live(
        self,
        owner: AuthenticatedOwnerQuotientPacket,
        editor: EditorSameStatePromptPacket,
    ) -> None:
        if self._token is not _COTANGENT_PACKET_TOKEN:
            raise NativeRV2VHiddenVJPError(
                "score cotangent was not built from authenticated packets"
            )
        editor.assert_live(owner)
        if (
            self.query_seed != owner.query_seed
            or self.score_divisor != SP_SIZE
            or self.owner_packet_receipt_digest != owner.receipt()["digest"]
            or self.editor_packet_receipt_digest != editor.receipt()["digest"]
            or self.global_cotangent_identity_digest
            != _global_cotangent_identity_digest(
                query_seed=self.query_seed,
                score=self.score,
                scorer_id=self.scorer_id,
                owner_packet_receipt_digest=self.owner_packet_receipt_digest,
                shared_state_binding_digest=editor.shared_state_binding_digest,
                action_measurement_sha256=self.action_measurement_sha256,
                noop_measurement_sha256=self.noop_measurement_sha256,
                action_cotangent_sha256=self.action_cotangent_sha256,
                noop_cotangent_sha256=self.noop_cotangent_sha256,
            )
            or self.scorer_id
            != f"authenticated-owner:{self.owner_packet_receipt_digest}"
            or tensor_sha256(
                self.action_measurement, label="live score action measurement"
            )
            != self.action_measurement_sha256
            or tensor_sha256(
                self.noop_measurement, label="live score no-op measurement"
            )
            != self.noop_measurement_sha256
            or tensor_sha256(
                self.action_cotangent, label="live score action cotangent"
            )
            != self.action_cotangent_sha256
            or tensor_sha256(
                self.noop_cotangent, label="live score no-op cotangent"
            )
            != self.noop_cotangent_sha256
            or not torch.equal(self.action_cotangent, -self.noop_cotangent)
        ):
            raise NativeRV2VHiddenVJPError("score cotangent live binding changed")


def _global_cotangent_identity_digest(
    *,
    query_seed: int,
    score: float,
    scorer_id: str,
    owner_packet_receipt_digest: str,
    shared_state_binding_digest: str,
    action_measurement_sha256: str,
    noop_measurement_sha256: str,
    action_cotangent_sha256: str,
    noop_cotangent_sha256: str,
) -> str:
    """Rank-independent identity for one globally reduced score cotangent."""

    if not math.isfinite(float(score)):
        raise NativeRV2VHiddenVJPError("global cotangent score differs")
    hashes = (
        owner_packet_receipt_digest,
        shared_state_binding_digest,
        action_measurement_sha256,
        noop_measurement_sha256,
        action_cotangent_sha256,
        noop_cotangent_sha256,
    )
    if any(_SHA256_RE.fullmatch(str(value)) is None for value in hashes):
        raise NativeRV2VHiddenVJPError("global cotangent hash closure differs")
    return object_sha256(
        {
            "schema_version": "bernini-qmosaic-global-cotangent-identity-v1",
            "query_seed": query_seed,
            "score_float64_hex": float(score).hex(),
            "score_divisor": SP_SIZE,
            "scorer_id": scorer_id,
            "owner_packet_receipt_digest": owner_packet_receipt_digest,
            "shared_state_binding_digest": shared_state_binding_digest,
            "action_measurement_sha256": action_measurement_sha256,
            "noop_measurement_sha256": noop_measurement_sha256,
            "action_cotangent_sha256": action_cotangent_sha256,
            "noop_cotangent_sha256": noop_cotangent_sha256,
        }
    )


def _score_cotangent_from_detached_measurement_unsafe_for_test(
    *,
    query_seed: int,
    action_measurement: torch.Tensor,
    noop_measurement: torch.Tensor,
    scorer: Any,
) -> DetachedScoreCotangent:
    """Unsafe tensor helper for algebra tests, never an authority boundary."""

    if type(query_seed) is not int or query_seed < 0:
        raise NativeRV2VHiddenVJPError("query seed must be a nonnegative integer")
    action = _validate_detached_global_sketch(
        action_measurement, label="action measurement"
    )
    noop = _validate_detached_global_sketch(noop_measurement, label="no-op measurement")
    if action.shape != noop.shape or action.device != noop.device:
        raise NativeRV2VHiddenVJPError("action/no-op measurement geometry differs")
    forward = getattr(scorer, "forward_sketched_residual", None)
    if not callable(forward):
        raise NativeRV2VHiddenVJPError("scorer lacks forward_sketched_residual")
    scorer_seed = getattr(scorer, "query_seed", query_seed)
    if scorer_seed != query_seed:
        raise NativeRV2VHiddenVJPError("scorer/query seed binding differs")
    scorer_id = str(getattr(scorer, "template_digest", type(scorer).__name__))

    action_leaf = action.detach().clone().requires_grad_(True)
    noop_leaf = noop.detach().clone().requires_grad_(True)
    output = forward(
        (action_leaf - noop_leaf).float().contiguous(), require_input_grad=True
    )
    score = getattr(output, "score", None)
    if (
        not isinstance(score, torch.Tensor)
        or score.numel() != 1
        or not score.requires_grad
        or score.grad_fn is None
        or not bool(torch.isfinite(score).all().item())
    ):
        raise NativeRV2VHiddenVJPError("per-query target-suffix score graph differs")
    action_q, noop_q = torch.autograd.grad(
        score.reshape(()) / float(SP_SIZE),
        (action_leaf, noop_leaf),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    action_q = action_q.detach().float().contiguous()
    noop_q = noop_q.detach().float().contiguous()
    norm = torch.linalg.vector_norm(action_q)
    if (
        not torch.equal(action_q, -noop_q)
        or not bool(torch.isfinite(norm).item())
        or float(norm.item()) <= 0.0
    ):
        raise NativeRV2VHiddenVJPError("detached score cotangent is zero or asymmetric")
    return DetachedScoreCotangent(
        query_seed=query_seed,
        score=float(score.detach().item()),
        score_divisor=SP_SIZE,
        action_measurement=action,
        noop_measurement=noop,
        action_cotangent=action_q,
        noop_cotangent=noop_q,
        scorer_id=scorer_id,
        action_measurement_sha256=tensor_sha256(action, label="action measurement"),
        noop_measurement_sha256=tensor_sha256(noop, label="no-op measurement"),
        action_cotangent_sha256=tensor_sha256(action_q, label="action cotangent"),
        noop_cotangent_sha256=tensor_sha256(noop_q, label="no-op cotangent"),
        owner_packet_receipt_digest="0" * 64,
        editor_packet_receipt_digest="0" * 64,
        global_cotangent_identity_digest=_global_cotangent_identity_digest(
            query_seed=query_seed,
            score=float(score.detach().item()),
            scorer_id=scorer_id,
            owner_packet_receipt_digest="0" * 64,
            shared_state_binding_digest="0" * 64,
            action_measurement_sha256=tensor_sha256(
                action, label="unsafe global action measurement"
            ),
            noop_measurement_sha256=tensor_sha256(
                noop, label="unsafe global no-op measurement"
            ),
            action_cotangent_sha256=tensor_sha256(
                action_q, label="unsafe global action cotangent"
            ),
            noop_cotangent_sha256=tensor_sha256(
                noop_q, label="unsafe global no-op cotangent"
            ),
        ),
    )


def score_cotangent_from_authenticated_packets(
    owner: AuthenticatedOwnerQuotientPacket,
    editor: EditorSameStatePromptPacket,
) -> DetachedScoreCotangent:
    """Differentiate the authenticated owner cosine score divided by SP4."""

    if type(owner) is not AuthenticatedOwnerQuotientPacket or type(
        editor
    ) is not EditorSameStatePromptPacket:
        raise NativeRV2VHiddenVJPError(
            "authoritative cotangent requires sealed owner/editor packets"
        )
    editor.assert_live(owner)
    action = _validate_detached_global_sketch(
        editor.action_measurement, label="authenticated action measurement"
    )
    noop = _validate_detached_global_sketch(
        editor.noop_measurement, label="authenticated no-op measurement"
    )
    if action.shape != noop.shape or action.device != noop.device:
        raise NativeRV2VHiddenVJPError("authenticated action/no-op geometry differs")
    action_leaf = action.detach().clone().requires_grad_(True)
    noop_leaf = noop.detach().clone().requires_grad_(True)
    try:
        feature = motion_cotangent.temporal_motion_quotient(
            (action_leaf - noop_leaf).float().contiguous(),
            require_input_grad=True,
        )
    except motion_cotangent.SelfImaginedCotangentContractError as error:
        raise NativeRV2VHiddenVJPError(str(error)) from error
    norm = torch.linalg.vector_norm(feature, dim=1, keepdim=True)
    owner_unit = owner.unit_feature.to(device=feature.device, dtype=feature.dtype)
    if (
        owner_unit.shape != feature.shape
        or not bool(torch.isfinite(norm).all().item())
        or float(norm.min().item())
        < motion_cotangent.MotionQuotientConfig().minimum_feature_norm
    ):
        raise NativeRV2VHiddenVJPError("authenticated editor quotient is degenerate")
    score = ((feature / norm) * owner_unit).sum().reshape(())
    action_q, noop_q = torch.autograd.grad(
        score / float(SP_SIZE),
        (action_leaf, noop_leaf),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    action_q = action_q.detach().float().contiguous()
    noop_q = noop_q.detach().float().contiguous()
    if (
        not torch.equal(action_q, -noop_q)
        or not bool(torch.isfinite(action_q).all().item())
        or float(torch.linalg.vector_norm(action_q).item()) <= 0.0
    ):
        raise NativeRV2VHiddenVJPError("authenticated score cotangent differs")
    editor.assert_live(owner)
    owner_digest = owner.receipt()["digest"]
    editor_digest = editor.receipt()["digest"]
    packet = DetachedScoreCotangent(
        query_seed=owner.query_seed,
        score=float(score.detach().item()),
        score_divisor=SP_SIZE,
        action_measurement=action,
        noop_measurement=noop,
        action_cotangent=action_q,
        noop_cotangent=noop_q,
        scorer_id=f"authenticated-owner:{owner_digest}",
        action_measurement_sha256=tensor_sha256(action, label="action measurement"),
        noop_measurement_sha256=tensor_sha256(noop, label="noop measurement"),
        action_cotangent_sha256=tensor_sha256(action_q, label="action cotangent"),
        noop_cotangent_sha256=tensor_sha256(noop_q, label="noop cotangent"),
        owner_packet_receipt_digest=owner_digest,
        editor_packet_receipt_digest=editor_digest,
        global_cotangent_identity_digest=_global_cotangent_identity_digest(
            query_seed=owner.query_seed,
            score=float(score.detach().item()),
            scorer_id=f"authenticated-owner:{owner_digest}",
            owner_packet_receipt_digest=owner_digest,
            shared_state_binding_digest=editor.shared_state_binding_digest,
            action_measurement_sha256=tensor_sha256(
                action, label="global action measurement"
            ),
            noop_measurement_sha256=tensor_sha256(
                noop, label="global no-op measurement"
            ),
            action_cotangent_sha256=tensor_sha256(
                action_q, label="global action cotangent"
            ),
            noop_cotangent_sha256=tensor_sha256(
                noop_q, label="global no-op cotangent"
            ),
        ),
    )
    object.__setattr__(packet, "_token", _COTANGENT_PACKET_TOKEN)
    packet.assert_live(owner, editor)
    return packet


def _validate_replay(
    value: Any, expected: torch.Tensor, *, role: str
) -> tuple[torch.Tensor, float]:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != expected.shape
        or value.dtype != torch.float32
        or value.device != expected.device
        or not value.requires_grad
        or value.grad_fn is None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise NativeRV2VHiddenVJPError(f"{role} graph replay sketch differs")
    error = float((value.detach() - expected).abs().max().item())
    scale = float(expected.abs().max().item())
    if error > REPLAY_ATOL + REPLAY_RTOL * scale:
        raise NativeRV2VHiddenVJPError(f"{role} graph replay changed measurement")
    return value, error


_DIRECTION_GATE_FIELDS = frozenset(
    {
        "schema_version",
        "cell_id",
        "query_seed",
        "source_iid",
        "source_video_sha256",
        "action_prompt_sha256",
        "noop_prompt_sha256",
        "owner_packet_receipt_digest",
        "global_cotangent_identity_digest",
        "clean_vjp_binding",
        "direction_artifact",
        "latent_artifacts",
        "symmetric_pair_contract",
        "decode_configuration",
        "decode_arms",
        "decision",
        "authority_public_key_sha256",
        "authority_signature_scheme",
        "receipt_digest",
        "authority_signature_ed25519_base64",
    }
)


def _load_gate_tensor_artifact(
    raw: Any,
    *,
    root: Path,
    expected_role: str,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    fields = {
        "role",
        "path",
        "file_sha256",
        "tensor_key",
        "dtype",
        "shape",
        "tensor_sha256",
        "materializer_receipt_digest",
        "create_only",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise NativeRV2VHiddenVJPError("direction-gate tensor artifact differs")
    if (
        raw.get("role") != expected_role
        or raw.get("dtype") != "torch.float32"
        or not isinstance(raw.get("shape"), list)
        or not raw["shape"]
        or any(type(item) is not int or item <= 0 for item in raw["shape"])
        or not isinstance(raw.get("tensor_key"), str)
        or not raw["tensor_key"]
        or not isinstance(raw.get("path"), str)
        or not Path(raw["path"]).is_absolute()
        or raw.get("create_only") is not True
        or _SHA256_RE.fullmatch(str(raw.get("file_sha256"))) is None
        or _SHA256_RE.fullmatch(str(raw.get("tensor_sha256"))) is None
        or _SHA256_RE.fullmatch(
            str(raw.get("materializer_receipt_digest"))
        )
        is None
    ):
        raise NativeRV2VHiddenVJPError(
            "direction-gate tensor artifact binding differs"
        )
    path = Path(raw["path"])
    if not path.is_file() or path.is_symlink():
        raise NativeRV2VHiddenVJPError(
            "direction-gate tensor artifact must be a plain file"
        )
    observed_file_sha = file_sha256(path)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise NativeRV2VHiddenVJPError(
            "direction-gate tensor escaped its artifact root"
        ) from error
    if observed_file_sha != raw["file_sha256"]:
        raise NativeRV2VHiddenVJPError(
            "direction-gate tensor artifact bytes changed"
        )
    try:
        from safetensors import safe_open

        with safe_open(str(resolved), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [raw["tensor_key"]]:
                raise NativeRV2VHiddenVJPError(
                    "direction-gate safetensors key closure differs"
                )
            value = opened.get_tensor(raw["tensor_key"]).detach().contiguous()
    except NativeRV2VHiddenVJPError:
        raise
    except Exception as error:
        raise NativeRV2VHiddenVJPError(
            "direction-gate tensor artifact could not be reopened"
        ) from error
    if (
        value.dtype != torch.float32
        or list(map(int, value.shape)) != raw["shape"]
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
        or tensor_sha256(value, label=f"live {expected_role}")
        != raw["tensor_sha256"]
    ):
        raise NativeRV2VHiddenVJPError(
            "direction-gate tensor artifact value differs"
        )
    return value, MappingProxyType(dict(raw))


def _redacted_media_probe_error(
    error: BaseException, *, sensitive_paths: Sequence[str]
) -> str:
    """Return bounded diagnostics without publishing runtime artifact paths."""

    def clean(value: Any) -> str:
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value)
        for sensitive in sorted(
            {item for item in sensitive_paths if item}, key=len, reverse=True
        ):
            text = text.replace(sensitive, "<redacted-path>")
        text = "".join(
            character
            if character in "\n\r\t" or 32 <= ord(character) <= 126
            else "?"
            for character in text
        )
        if len(text) > 4096:
            text = text[:4096] + "...[truncated]"
        return text

    stderr = getattr(error, "stderr", None)
    stderr_text = "<none>" if stderr in (None, b"", "") else clean(stderr)
    return f"{type(error).__name__}: {clean(error)}; stderr={stderr_text}"


@lru_cache(maxsize=4)
def _verify_pinned_distribution_tree(
    *,
    record_path: str,
    site_packages_root: str,
    expected_record_sha256: str,
    expected_tree_sha256: str,
    expected_file_count: int,
    path_prefixes: tuple[str, ...],
) -> tuple[str, int]:
    """Verify every hashed RECORD member under the runtime package prefixes."""

    record = Path(record_path)
    root = Path(site_packages_root)
    if (
        not record.is_absolute()
        or not record.is_file()
        or record.is_symlink()
        or record.resolve(strict=True) != record
        or not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
        or file_sha256(record) != expected_record_sha256
        or _SHA256_RE.fullmatch(expected_tree_sha256) is None
        or type(expected_file_count) is not int
        or expected_file_count <= 0
        or not path_prefixes
        or any(
            not isinstance(prefix, str)
            or not prefix
            or prefix.startswith("/")
            or not prefix.endswith("/")
            for prefix in path_prefixes
        )
    ):
        raise NativeRV2VHiddenVJPError("distribution RECORD authority differs")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with record.open("r", encoding="utf-8", newline="") as handle:
        for record_row in csv.reader(handle):
            if len(record_row) != 3:
                raise NativeRV2VHiddenVJPError(
                    "distribution RECORD row closure differs"
                )
            relative_text, encoded_digest, encoded_size = record_row
            if not any(relative_text.startswith(prefix) for prefix in path_prefixes):
                continue
            relative = PurePosixPath(relative_text)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in ("", ".", "..") for part in relative.parts)
                or "\\" in relative_text
                or ((not encoded_digest) != (not encoded_size))
                or (
                    bool(encoded_digest)
                    and not encoded_digest.startswith("sha256=")
                )
                or (
                    bool(encoded_size)
                    and re.fullmatch(r"[0-9]+", encoded_size) is None
                )
                or (
                    not encoded_digest
                    and (
                        "__pycache__" not in relative.parts
                        or not relative_text.endswith(".pyc")
                    )
                )
            ):
                raise NativeRV2VHiddenVJPError(
                    "distribution RECORD member binding differs"
                )
            if not encoded_digest:
                # Wheel RECORD intentionally leaves generated bytecode
                # unhashed.  It is runtime cache, never package authority.
                continue
            if relative_text in seen:
                raise NativeRV2VHiddenVJPError(
                    "distribution RECORD member binding differs"
                )
            seen.add(relative_text)
            digest_text = encoded_digest.split("=", 1)[1]
            try:
                expected_digest = base64.urlsafe_b64decode(
                    digest_text + "=" * (-len(digest_text) % 4)
                )
            except (ValueError, UnicodeError) as error:
                raise NativeRV2VHiddenVJPError(
                    "distribution RECORD SHA-256 encoding differs"
                ) from error
            target = root.joinpath(*relative.parts)
            if (
                len(expected_digest) != 32
                or not target.is_file()
                or target.is_symlink()
                or not stat.S_ISREG(target.stat().st_mode)
                or target.resolve(strict=True) != target
            ):
                raise NativeRV2VHiddenVJPError(
                    "installed distribution member differs from pinned RECORD"
                )
            observed_size = int(target.stat().st_size)
            observed_sha = file_sha256(target)
            if (
                observed_size != int(encoded_size)
                or observed_sha != expected_digest.hex()
            ):
                raise NativeRV2VHiddenVJPError(
                    "installed distribution member differs from pinned RECORD"
                )
            rows.append(
                {
                    "path": relative_text,
                    "sha256": observed_sha,
                    "size": observed_size,
                }
            )
    observed_paths: set[str] = set()
    for prefix in path_prefixes:
        prefix_root = root.joinpath(*PurePosixPath(prefix[:-1]).parts)
        if not prefix_root.is_dir() or prefix_root.is_symlink():
            raise NativeRV2VHiddenVJPError(
                "installed distribution prefix root differs"
            )
        for candidate in prefix_root.rglob("*"):
            relative_candidate = candidate.relative_to(root)
            if "__pycache__" in relative_candidate.parts:
                continue
            if candidate.is_symlink():
                raise NativeRV2VHiddenVJPError(
                    "installed distribution tree contains a symlink"
                )
            if candidate.is_file():
                if (
                    not stat.S_ISREG(candidate.stat().st_mode)
                    or candidate.resolve(strict=True) != candidate
                ):
                    raise NativeRV2VHiddenVJPError(
                        "installed distribution tree contains a non-plain file"
                    )
                observed_paths.add(relative_candidate.as_posix())
    if (
        not rows
        or len(rows) != expected_file_count
        or seen != observed_paths
    ):
        raise NativeRV2VHiddenVJPError("distribution RECORD prefix closure is empty")
    rows.sort(key=lambda row: row["path"])
    transcript = hashlib.sha256()
    for row in rows:
        transcript.update(canonical_json_bytes(row) + b"\n")
    if transcript.hexdigest() != expected_tree_sha256:
        raise NativeRV2VHiddenVJPError(
            "installed distribution tree digest differs"
        )
    return transcript.hexdigest(), len(rows)


def _resolve_imageio_bundled_ffmpeg() -> tuple[
    Path, str, str, str, str, str, str, int
]:
    """Resolve and bind imageio-ffmpeg's packaged binary, never PATH/env input."""

    if "IMAGEIO_FFMPEG_EXE" in os.environ:
        raise NativeRV2VHiddenVJPError(
            "IMAGEIO_FFMPEG_EXE caller injection is forbidden"
        )
    try:
        import imageio_ffmpeg
    except Exception as error:
        raise NativeRV2VHiddenVJPError(
            f"imageio_ffmpeg import failed ({type(error).__name__}: {error})"
        ) from error
    package_version = getattr(imageio_ffmpeg, "__version__", None)
    module_file_raw = getattr(imageio_ffmpeg, "__file__", None)
    if (
        package_version != PINNED_IMAGEIO_FFMPEG_VERSION
        or not isinstance(module_file_raw, str)
    ):
        raise NativeRV2VHiddenVJPError("imageio_ffmpeg package binding differs")
    module_file = Path(module_file_raw)
    distribution_record = (
        module_file.parent.parent / "imageio_ffmpeg-0.6.0.dist-info" / "RECORD"
    )
    if (
        not module_file.is_absolute()
        or not module_file.is_file()
        or module_file.is_symlink()
        or module_file.resolve(strict=True) != module_file
        or module_file.name != "__init__.py"
        or module_file.parent.name != "imageio_ffmpeg"
        or not distribution_record.is_file()
        or distribution_record.is_symlink()
        or distribution_record.resolve(strict=True) != distribution_record
    ):
        raise NativeRV2VHiddenVJPError("imageio_ffmpeg module is not a plain file")
    module_sha = file_sha256(module_file)
    record_sha = file_sha256(distribution_record)
    if (
        module_sha != PINNED_IMAGEIO_FFMPEG_MODULE_SHA256
        or record_sha != PINNED_IMAGEIO_FFMPEG_RECORD_SHA256
    ):
        raise NativeRV2VHiddenVJPError(
            "imageio_ffmpeg installed package bytes differ"
        )
    distribution_tree_digest, distribution_file_count = (
        _verify_pinned_distribution_tree(
            record_path=str(distribution_record),
            site_packages_root=str(module_file.parent.parent),
            expected_record_sha256=PINNED_IMAGEIO_FFMPEG_RECORD_SHA256,
            expected_tree_sha256=PINNED_IMAGEIO_FFMPEG_DISTRIBUTION_HASHED_TREE_SHA256,
            expected_file_count=PINNED_IMAGEIO_FFMPEG_DISTRIBUTION_HASHED_FILE_COUNT,
            path_prefixes=("imageio_ffmpeg/",),
        )
    )
    bundle_root = (module_file.resolve(strict=True).parent / "binaries").resolve(
        strict=True
    )
    try:
        executable_raw = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        raise NativeRV2VHiddenVJPError(
            f"imageio_ffmpeg executable resolution failed ({type(error).__name__}: {error})"
        ) from error
    if not isinstance(executable_raw, str):
        raise NativeRV2VHiddenVJPError("bundled ffmpeg path binding differs")
    executable = Path(executable_raw)
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or executable.is_symlink()
        or not stat.S_ISREG(executable.stat().st_mode)
        or executable.stat().st_mode & 0o111 == 0
        or not os.access(executable, os.X_OK)
    ):
        raise NativeRV2VHiddenVJPError(
            "bundled ffmpeg must be an absolute executable regular file"
        )
    resolved = executable.resolve(strict=True)
    expected_executable = bundle_root / PINNED_BUNDLED_FFMPEG_BASENAME
    if resolved != executable or resolved != expected_executable:
        raise NativeRV2VHiddenVJPError(
            "ffmpeg must be the non-symlink imageio_ffmpeg packaged binary"
        )
    executable_sha = file_sha256(resolved)
    if executable_sha != PINNED_BUNDLED_FFMPEG_SHA256:
        raise NativeRV2VHiddenVJPError(
            "bundled ffmpeg executable SHA-256 differs from the pinned vace binary"
        )
    version = subprocess.run(
        (str(resolved), "-version"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    version_lines = version.stdout.decode("utf-8").splitlines()
    if (
        not version_lines
        or version_lines[0] != PINNED_BUNDLED_FFMPEG_VERSION_LINE
    ):
        raise NativeRV2VHiddenVJPError("bundled ffmpeg version binding differs")
    return (
        resolved,
        executable_sha,
        version_lines[0],
        package_version,
        module_sha,
        record_sha,
        distribution_tree_digest,
        distribution_file_count,
    )


def _probe_decode_exact81_impl(
    path: Path, *, sensitive_paths: list[str]
) -> Mapping[str, Any]:
    try:
        import av
    except Exception as error:
        raise NativeRV2VHiddenVJPError(
            f"PyAV import failed ({type(error).__name__}: {error})"
        ) from error
    pyav_version = getattr(av, "__version__", None)
    if pyav_version != PINNED_PYAV_VERSION:
        raise NativeRV2VHiddenVJPError("PyAV backend version binding differs")
    pyav_module_raw = getattr(av, "__file__", None)
    if not isinstance(pyav_module_raw, str):
        raise NativeRV2VHiddenVJPError("PyAV module file binding differs")
    pyav_module = Path(pyav_module_raw)
    pyav_record = (
        pyav_module.parent.parent / "av-13.1.0.dist-info" / "RECORD"
    )
    if (
        not pyav_module.is_absolute()
        or not pyav_module.is_file()
        or pyav_module.is_symlink()
        or pyav_module.resolve(strict=True) != pyav_module
        or pyav_module.name != "__init__.py"
        or pyav_module.parent.name != "av"
        or not pyav_record.is_file()
        or pyav_record.is_symlink()
        or pyav_record.resolve(strict=True) != pyav_record
    ):
        raise NativeRV2VHiddenVJPError("PyAV package-root containment differs")
    pyav_module_sha = file_sha256(pyav_module)
    pyav_record_sha = file_sha256(pyav_record)
    if (
        pyav_module_sha != PINNED_PYAV_MODULE_SHA256
        or pyav_record_sha != PINNED_PYAV_RECORD_SHA256
    ):
        raise NativeRV2VHiddenVJPError("PyAV installed package bytes differ")
    pyav_tree_digest, pyav_file_count = _verify_pinned_distribution_tree(
        record_path=str(pyav_record),
        site_packages_root=str(pyav_module.parent.parent),
        expected_record_sha256=PINNED_PYAV_RECORD_SHA256,
        expected_tree_sha256=PINNED_PYAV_DISTRIBUTION_HASHED_TREE_SHA256,
        expected_file_count=PINNED_PYAV_DISTRIBUTION_HASHED_FILE_COUNT,
        path_prefixes=("av/", "av.libs/"),
    )
    raw_library_versions = getattr(av, "library_versions", None)
    if not isinstance(raw_library_versions, Mapping) or not raw_library_versions:
        raise NativeRV2VHiddenVJPError("PyAV linked-library binding differs")
    pyav_library_versions: dict[str, list[int]] = {}
    for library_name, library_version in sorted(raw_library_versions.items()):
        if (
            not isinstance(library_name, str)
            or not library_name
            or not isinstance(library_version, tuple)
            or len(library_version) != 3
            or any(type(component) is not int or component < 0 for component in library_version)
        ):
            raise NativeRV2VHiddenVJPError("PyAV linked-library binding differs")
        pyav_library_versions[library_name] = list(library_version)
    if pyav_library_versions != {
        name: list(version)
        for name, version in PINNED_PYAV_LIBRARY_VERSIONS.items()
    }:
        raise NativeRV2VHiddenVJPError("PyAV linked-library versions differ")

    pyav_transcript = hashlib.sha256()
    with av.open(str(path), mode="r") as container:
        format_name = getattr(getattr(container, "format", None), "name", None)
        streams = list(container.streams)
        video_streams = list(container.streams.video)
        if (
            not isinstance(format_name, str)
            or "mp4" not in format_name.split(",")
            or len(streams) != 1
            or len(video_streams) != 1
        ):
            raise NativeRV2VHiddenVJPError(
                "PyAV requires one video-only MP4 stream"
            )
        stream = video_streams[0]
        codec_context = stream.codec_context
        codec_name = getattr(codec_context, "name", None)
        width = getattr(codec_context, "width", None)
        height = getattr(codec_context, "height", None)
        pixel_format = getattr(codec_context, "pix_fmt", None)
        average_rate = getattr(stream, "average_rate", None)
        numerator = getattr(average_rate, "numerator", None)
        denominator = getattr(average_rate, "denominator", None)
        if (
            not isinstance(codec_name, str)
            or not codec_name
            or type(width) is not int
            or width <= 0
            or type(height) is not int
            or height <= 0
            or not isinstance(pixel_format, str)
            or not pixel_format
            or type(numerator) is not int
            or numerator <= 0
            or type(denominator) is not int
            or denominator <= 0
        ):
            raise NativeRV2VHiddenVJPError("PyAV video metadata binding differs")
        avg_frame_rate = f"{numerator}/{denominator}"
        pyav_frame_count = 0
        decoded_pts: list[int] = []
        decoded_time_base: Optional[tuple[int, int]] = None
        for frame in container.decode(video=0):
            pyav_frame_count += 1
            if pyav_frame_count > 81:
                raise NativeRV2VHiddenVJPError("PyAV decoded more than 81 frames")
            frame_pts = getattr(frame, "pts", None)
            time_base = getattr(frame, "time_base", None)
            time_numerator = getattr(time_base, "numerator", None)
            time_denominator = getattr(time_base, "denominator", None)
            source_format = getattr(getattr(frame, "format", None), "name", None)
            rgb24 = frame.to_ndarray(format="rgb24")
            if (
                type(frame_pts) is not int
                or type(time_numerator) is not int
                or time_numerator <= 0
                or type(time_denominator) is not int
                or time_denominator <= 0
                or getattr(frame, "width", None) != width
                or getattr(frame, "height", None) != height
                or not isinstance(source_format, str)
                or tuple(map(int, getattr(rgb24, "shape", ())))
                != (height, width, 3)
                or str(getattr(rgb24, "dtype", "")) != "uint8"
            ):
                raise NativeRV2VHiddenVJPError(
                    "PyAV decoded RGB24 frame contract differs"
                )
            current_time_base = (time_numerator, time_denominator)
            if decoded_time_base is None:
                decoded_time_base = current_time_base
            elif current_time_base != decoded_time_base:
                raise NativeRV2VHiddenVJPError(
                    "PyAV decoded frame time-base changed within the stream"
                )
            decoded_pts.append(frame_pts)
            frame_record = {
                "frame_index": pyav_frame_count - 1,
                "pts": frame_pts,
                "time_base": f"{time_numerator}/{time_denominator}",
                "width": width,
                "height": height,
                "source_pixel_format": source_format,
                "rgb24_sha256": hashlib.sha256(
                    rgb24.tobytes(order="C")
                ).hexdigest(),
            }
            pyav_transcript.update(canonical_json_bytes(frame_record) + b"\n")
    if pyav_frame_count != 81:
        raise NativeRV2VHiddenVJPError(
            f"PyAV decoded {pyav_frame_count} frames instead of 81"
        )
    if (
        numerator != 25 * denominator
        or decoded_time_base is None
        or len(decoded_pts) != 81
        or decoded_pts[0] != 0
        or any(
            (right - left) * decoded_time_base[0] * 25
            != decoded_time_base[1]
            for left, right in zip(decoded_pts, decoded_pts[1:])
        )
    ):
        raise NativeRV2VHiddenVJPError(
            "PyAV decoded timestamps are not an exact constant 25-fps cadence"
        )

    (
        ffmpeg,
        ffmpeg_sha,
        ffmpeg_version,
        imageio_version,
        imageio_module_sha,
        imageio_record_sha,
        imageio_tree_digest,
        imageio_file_count,
    ) = (
        _resolve_imageio_bundled_ffmpeg()
    )
    sensitive_paths.append(str(ffmpeg))
    decoded = subprocess.run(
        (
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "framemd5",
            "-",
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    transcript_lines = [
        line.strip()
        for line in decoded.stdout.decode("ascii").splitlines()
        if line.strip()
    ]
    frame_lines = [line for line in transcript_lines if not line.startswith("#")]
    header_lines = {line for line in transcript_lines if line.startswith("#")}
    frame_records = [
        tuple(part.strip() for part in line.split(",")) for line in frame_lines
    ]
    if (
        not {
            "#format: frame checksums",
            "#version: 2",
            "#hash: MD5",
            "#software: Lavf61.1.100",
            "#tb 0: 1/25",
            f"#dimensions 0: {width}x{height}",
        }.issubset(header_lines)
        or len(frame_lines) != 81
        or any(
            len(record) != 6
            or record[0] != "0"
            or record[1] != str(index)
            or record[2] != str(index)
            or record[3] != "1"
            or re.fullmatch(r"[1-9][0-9]*", record[4]) is None
            or re.fullmatch(r"[0-9a-f]{32}", record[5]) is None
            for index, record in enumerate(frame_records)
        )
    ):
        raise NativeRV2VHiddenVJPError(
            "bundled ffmpeg framemd5 did not decode exact81"
        )
    ffmpeg_transcript = ("\n".join(transcript_lines) + "\n").encode("ascii")
    pyav_transcript_sha = pyav_transcript.hexdigest()
    ffmpeg_transcript_sha = hashlib.sha256(ffmpeg_transcript).hexdigest()
    transcript_binding = {
        "pyav_rgb24_frame_transcript_sha256": pyav_transcript_sha,
        "bundled_ffmpeg_framemd5_transcript_sha256": ffmpeg_transcript_sha,
        "width": width,
        "height": height,
        "avg_frame_rate": avg_frame_rate,
        "frame_count": 81,
    }
    value = {
        "schema_version": EXACT81_MEDIA_PROBE_SCHEMA_VERSION,
        "format_name": format_name,
        "video_stream_count": 1,
        "container_stream_count": 1,
        "codec_name": codec_name,
        "width": width,
        "height": height,
        "pix_fmt": pixel_format,
        "avg_frame_rate": avg_frame_rate,
        "pyav_backend_name": "PyAV",
        "pyav_version": pyav_version,
        "pyav_linked_library_versions": pyav_library_versions,
        "pyav_module_file_sha256": pyav_module_sha,
        "pyav_distribution_record_sha256": pyav_record_sha,
        "pyav_distribution_hashed_tree_digest_sha256": pyav_tree_digest,
        "pyav_distribution_hashed_file_count": pyav_file_count,
        "pyav_decoded_frame_count": 81,
        "pyav_first_pts": decoded_pts[0],
        "pyav_last_pts": decoded_pts[-1],
        "pyav_time_base": f"{decoded_time_base[0]}/{decoded_time_base[1]}",
        "pyav_pts_cadence_rational": "1/25",
        "pyav_exact_25fps_pts_cadence": True,
        "pyav_rgb24_frame_transcript_sha256": pyav_transcript_sha,
        "imageio_ffmpeg_version": imageio_version,
        "imageio_ffmpeg_module_file_sha256": imageio_module_sha,
        "imageio_ffmpeg_distribution_record_sha256": imageio_record_sha,
        "imageio_ffmpeg_distribution_hashed_tree_digest_sha256": imageio_tree_digest,
        "imageio_ffmpeg_distribution_hashed_file_count": imageio_file_count,
        "bundled_ffmpeg_executable_realpath": str(ffmpeg),
        "bundled_ffmpeg_executable_sha256": ffmpeg_sha,
        "bundled_ffmpeg_version_line": ffmpeg_version,
        "bundled_ffmpeg_framemd5_frame_count": 81,
        "bundled_ffmpeg_framemd5_transcript_sha256": ffmpeg_transcript_sha,
        "decoded_frame_transcript_sha256": object_sha256(transcript_binding),
    }
    if set(value) != EXACT81_MEDIA_PROBE_FIELDS or any(
        _SHA256_RE.fullmatch(str(value[name])) is None
        for name in (
            "pyav_rgb24_frame_transcript_sha256",
            "pyav_module_file_sha256",
            "pyav_distribution_record_sha256",
            "pyav_distribution_hashed_tree_digest_sha256",
            "imageio_ffmpeg_module_file_sha256",
            "imageio_ffmpeg_distribution_record_sha256",
            "imageio_ffmpeg_distribution_hashed_tree_digest_sha256",
            "bundled_ffmpeg_executable_sha256",
            "bundled_ffmpeg_framemd5_transcript_sha256",
            "decoded_frame_transcript_sha256",
        )
    ):
        raise NativeRV2VHiddenVJPError("portable media probe receipt closure differs")
    return MappingProxyType(value)


def _probe_decode_exact81(path: Path) -> Mapping[str, Any]:
    """Decode exact81 through PyAV and a hash-bound packaged ffmpeg binary."""

    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
    ):
        raise NativeRV2VHiddenVJPError(
            "direction-gate MP4 must be an absolute plain file"
        )
    sensitive_paths = [str(path)]
    try:
        return _probe_decode_exact81_impl(path, sensitive_paths=sensitive_paths)
    except Exception as error:
        raise NativeRV2VHiddenVJPError(
            "direction-gate portable media probe failed ["
            + _redacted_media_probe_error(
                error, sensitive_paths=sensitive_paths
            )
            + "]"
        ) from error


def _verify_direction_gate_payload(
    *,
    gate_path: str | Path,
    expected_gate_file_sha256: str,
    public_key_path: str | Path,
    expected_public_key_file_sha256: str,
    artifact_root: str | Path,
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    Mapping[str, torch.Tensor],
    tuple[Mapping[str, Any], ...],
]:
    value, resolved_gate, _ = _strict_json_file(
        gate_path,
        expected_sha256=expected_gate_file_sha256,
        label="exact81 direction gate",
    )
    if set(value) != _DIRECTION_GATE_FIELDS:
        raise NativeRV2VHiddenVJPError("exact81 direction gate field closure differs")
    key_path = Path(public_key_path)
    expected_key_sha = _sha256(
        expected_public_key_file_sha256,
        label="direction gate public key SHA-256",
    )
    observed_key_sha = file_sha256(key_path)
    if observed_key_sha != expected_key_sha:
        raise NativeRV2VHiddenVJPError("direction gate public key bytes changed")
    if (
        value["schema_version"] != DIRECTION_GATE_SCHEMA_VERSION
        or value["authority_signature_scheme"]
        != DIRECTION_GATE_SIGNATURE_SCHEME
        or value["authority_public_key_sha256"] != observed_key_sha
    ):
        raise NativeRV2VHiddenVJPError("direction gate signing authority differs")
    signed = dict(value)
    encoded = signed.pop("authority_signature_ed25519_base64")
    if not isinstance(encoded, str):
        raise NativeRV2VHiddenVJPError("direction gate signature encoding differs")
    unsigned = dict(signed)
    declared_digest = unsigned.pop("receipt_digest", None)
    if declared_digest != object_sha256(unsigned):
        raise NativeRV2VHiddenVJPError("direction gate receipt seal differs")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        signature = base64.b64decode(encoded.encode("ascii"), validate=True)
        public_key = serialization.load_pem_public_key(key_path.read_bytes())
        if not isinstance(public_key, Ed25519PublicKey) or len(signature) != 64:
            raise NativeRV2VHiddenVJPError("direction gate Ed25519 key/signature differs")
        public_key.verify(signature, canonical_json_bytes(signed))
    except NativeRV2VHiddenVJPError:
        raise
    except (InvalidSignature, ValueError, UnicodeError) as error:
        raise NativeRV2VHiddenVJPError(
            "direction gate Ed25519 signature verification failed"
        ) from error
    root = Path(artifact_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise NativeRV2VHiddenVJPError("direction gate artifact root differs")
    resolved_root = root.resolve(strict=True)
    direction, _ = _load_gate_tensor_artifact(
        value.get("direction_artifact"),
        root=resolved_root,
        expected_role="normalized_clean_vjp_direction",
    )
    raw_latents = value.get("latent_artifacts")
    if not isinstance(raw_latents, list) or [
        row.get("role") if isinstance(row, Mapping) else None
        for row in raw_latents
    ] != ["base_clean_latent", "plus_clean_latent", "minus_clean_latent"]:
        raise NativeRV2VHiddenVJPError(
            "direction-gate latent artifact role closure differs"
        )
    tensors: dict[str, torch.Tensor] = {"direction": direction}
    for role, raw in zip(("base", "plus", "minus"), raw_latents):
        tensor, _binding = _load_gate_tensor_artifact(
            raw,
            root=resolved_root,
            expected_role=f"{role}_clean_latent",
        )
        tensors[role] = tensor
    raw_arms = value.get("decode_arms")
    required_arm_fields = {
        "role",
        "latent_tensor_sha256",
        "mp4_path",
        "mp4_file_sha256",
        "decode_seed",
        "decode_configuration_digest",
        "portable_decode_receipt",
        "decoded_frame_transcript_sha256",
    }
    if not isinstance(raw_arms, list) or [
        row.get("role") if isinstance(row, Mapping) else None for row in raw_arms
    ] != ["base", "plus", "minus"]:
        raise NativeRV2VHiddenVJPError(
            "direction-gate decode arm role closure differs"
        )
    arms: list[Mapping[str, Any]] = []
    arm_paths: list[Path] = []
    for role, raw in zip(("base", "plus", "minus"), raw_arms):
        if not isinstance(raw, Mapping) or set(raw) != required_arm_fields:
            raise NativeRV2VHiddenVJPError(
                "direction-gate decode arm binding differs"
            )
        if (
            not isinstance(raw.get("mp4_path"), str)
            or not Path(raw["mp4_path"]).is_absolute()
            or type(raw.get("decode_seed")) is not int
            or raw["decode_seed"] < 0
            or any(
                _SHA256_RE.fullmatch(str(raw.get(name))) is None
                for name in (
                    "latent_tensor_sha256",
                    "mp4_file_sha256",
                    "decode_configuration_digest",
                    "decoded_frame_transcript_sha256",
                )
            )
        ):
            raise NativeRV2VHiddenVJPError(
                "direction-gate decode arm scalar binding differs"
            )
        path = Path(raw["mp4_path"])
        if not path.is_file() or path.is_symlink():
            raise NativeRV2VHiddenVJPError(
                "direction-gate MP4 must be a plain file"
            )
        observed_file_sha = file_sha256(path)
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise NativeRV2VHiddenVJPError(
                "direction-gate MP4 escaped its artifact root"
            ) from error
        live_probe = _probe_decode_exact81(resolved)
        if (
            raw["latent_tensor_sha256"]
            != tensor_sha256(tensors[role], label=f"gate {role} latent")
            or raw["mp4_file_sha256"] != observed_file_sha
            or raw["portable_decode_receipt"] != dict(live_probe)
            or raw["decoded_frame_transcript_sha256"]
            != live_probe["decoded_frame_transcript_sha256"]
        ):
            raise NativeRV2VHiddenVJPError(
                "direction-gate decode arm live proof differs"
            )
        arm_paths.append(resolved)
        arms.append(MappingProxyType(dict(raw)))
    if (
        len(set(arm_paths)) != 3
        or len({arm["mp4_file_sha256"] for arm in arms}) != 3
        or len({arm["decoded_frame_transcript_sha256"] for arm in arms}) != 3
    ):
        raise NativeRV2VHiddenVJPError(
            "direction-gate decode arms alias one rendered result"
        )
    return (
        value,
        resolved_gate,
        key_path.resolve(strict=True),
        MappingProxyType(tensors),
        tuple(arms),
    )


def _validate_symmetric_gate_latents(
    *,
    clean_vjp: torch.Tensor,
    runtime_clean_latent: torch.Tensor,
    direction: torch.Tensor,
    base: torch.Tensor,
    plus: torch.Tensor,
    minus: torch.Tensor,
    relative_l2_dose: float,
) -> None:
    """Recompute ``q`` and both latent arms in the signed FP32 coordinate."""

    values = {
        "clean_vjp": clean_vjp,
        "runtime_clean_latent": runtime_clean_latent,
        "direction": direction,
        "base": base,
        "plus": plus,
        "minus": minus,
    }
    normalized: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise NativeRV2VHiddenVJPError(
                f"direction gate {name} tensor differs"
            )
        tensor = value.detach().float().cpu().contiguous()
        if tensor.requires_grad or tensor.grad_fn is not None or not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise NativeRV2VHiddenVJPError(
                f"direction gate {name} tensor differs"
            )
        normalized[name] = tensor
    clean = normalized["clean_vjp"]
    runtime_base = normalized["runtime_clean_latent"]
    signed_base = normalized["base"]
    if (
        isinstance(relative_l2_dose, bool)
        or not isinstance(relative_l2_dose, (int, float))
        or not math.isfinite(float(relative_l2_dose))
        or not 0.0 < float(relative_l2_dose) < 1.0
        or any(value.shape != clean.shape for value in normalized.values())
        or not torch.equal(signed_base, runtime_base)
    ):
        raise NativeRV2VHiddenVJPError(
            "direction gate clean latent/VJP coordinate differs"
        )
    clean_norm = torch.linalg.vector_norm(clean)
    if (
        not bool(torch.isfinite(clean_norm).item())
        or float(clean_norm.item()) <= 0.0
    ):
        raise NativeRV2VHiddenVJPError(
            "direction gate clean latent/VJP coordinate differs"
        )
    expected_direction = (clean / clean_norm).contiguous()
    dose = torch.tensor(float(relative_l2_dose), dtype=torch.float32)
    scale = dose * torch.linalg.vector_norm(signed_base)
    expected_plus = (signed_base + scale * expected_direction).contiguous()
    expected_minus = (signed_base - scale * expected_direction).contiguous()
    if (
        not torch.equal(normalized["direction"], expected_direction)
        or not torch.equal(normalized["plus"], expected_plus)
        or not torch.equal(normalized["minus"], expected_minus)
    ):
        raise NativeRV2VHiddenVJPError(
            "signed direction/plus/minus latent numeric proof differs"
        )


@dataclass(frozen=True)
class ValidatedExact81DirectionGate:
    payload: Mapping[str, Any]
    gate_path: Path
    gate_file_sha256: str
    public_key_path: Path
    public_key_file_sha256: str
    artifact_root: Path
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": DIRECTION_GATE_SCHEMA_VERSION,
            "signed_gate_receipt_digest": self.payload["receipt_digest"],
            "signed_gate_file_sha256": self.gate_file_sha256,
            "public_key_file_sha256": self.public_key_file_sha256,
            "clean_vjp_receipt_digest": self.payload["clean_vjp_binding"][
                "receipt_digest"
            ],
            "query_seed": self.payload["query_seed"],
            "decode_seed": self.payload["decode_configuration"]["decode_seed"],
            "relative_l2_dose": self.payload["symmetric_pair_contract"][
                "relative_l2_dose"
            ],
            "all_mp4_arms_pyav_and_bundled_ffmpeg_decoded_exact81": True,
            "direction_and_symmetric_latents_numerically_recomputed": True,
            "naked_boolean_gate": False,
        }
        return {**value, "digest": object_sha256(value)}

    def assert_live(
        self,
        *,
        owner: AuthenticatedOwnerQuotientPacket,
        editor: EditorSameStatePromptPacket,
        score_packet: DetachedScoreCotangent,
        clean_vjp_row: SP4SummedVJPRow,
    ) -> None:
        if self._token is not _DIRECTION_GATE_TOKEN:
            raise NativeRV2VHiddenVJPError("direction gate was not signature-loaded")
        owner.assert_live()
        editor.assert_live(owner)
        score_packet.assert_live(owner, editor)
        if type(clean_vjp_row) is not SP4SummedVJPRow:
            raise NativeRV2VHiddenVJPError("direction gate clean SP4 row differs")
        clean_vjp_row.assert_live()
        value, gate_path, key_path, tensors, arms = _verify_direction_gate_payload(
            gate_path=self.gate_path,
            expected_gate_file_sha256=self.gate_file_sha256,
            public_key_path=self.public_key_path,
            expected_public_key_file_sha256=self.public_key_file_sha256,
            artifact_root=self.artifact_root,
        )
        clean = value.get("clean_vjp_binding")
        pair = value.get("symmetric_pair_contract")
        decode = value.get("decode_configuration")
        decision = value.get("decision")
        required_top_hashes = (
            "source_video_sha256",
            "action_prompt_sha256",
            "noop_prompt_sha256",
            "owner_packet_receipt_digest",
            "global_cotangent_identity_digest",
            "authority_public_key_sha256",
            "receipt_digest",
        )
        if (
            value != dict(self.payload)
            or gate_path != self.gate_path
            or key_path != self.public_key_path
            or not isinstance(clean, Mapping)
            or set(clean)
            != {
                "receipt_digest",
                "value_sha256",
                "global_cotangent_identity_digest",
                "rank_editor_packet_receipt_digests",
                "rank_score_cotangent_receipt_digests",
            }
            or type(clean_vjp_row) is not SP4SummedVJPRow
            or clean_vjp_row.vjp_target != "clean_latent"
            or clean.get("receipt_digest") != clean_vjp_row.receipt()["digest"]
            or clean.get("value_sha256") != clean_vjp_row.value_sha256
            or clean.get("global_cotangent_identity_digest")
            != clean_vjp_row.global_cotangent_identity_digest
            or clean.get("rank_editor_packet_receipt_digests")
            != list(clean_vjp_row.rank_editor_packet_receipt_digests)
            or clean.get("rank_score_cotangent_receipt_digests")
            != list(clean_vjp_row.rank_score_cotangent_receipt_digests)
            or clean_vjp_row.rank_editor_packet_receipt_digests[editor.sp_rank]
            != editor.receipt()["digest"]
            or clean_vjp_row.rank_score_cotangent_receipt_digests[editor.sp_rank]
            != score_packet.receipt()["digest"]
            or any(
                not isinstance(clean.get(name), str)
                or _SHA256_RE.fullmatch(clean[name]) is None
                for name in (
                    "receipt_digest",
                    "value_sha256",
                    "global_cotangent_identity_digest",
                )
            )
            or value.get("cell_id") != owner.cell_id
            or value.get("query_seed") != owner.query_seed
            or any(
                not isinstance(value.get(name), str)
                or _SHA256_RE.fullmatch(value[name]) is None
                for name in required_top_hashes
            )
            or value.get("source_iid") != owner.source_iid
            or value.get("source_video_sha256") != owner.source_video_sha256
            or value.get("action_prompt_sha256") != owner.action_prompt_sha256
            or value.get("noop_prompt_sha256") != owner.noop_prompt_sha256
            or value.get("owner_packet_receipt_digest")
            != owner.receipt()["digest"]
            or value.get("global_cotangent_identity_digest")
            != score_packet.global_cotangent_identity_digest
            or not isinstance(pair, Mapping)
            or set(pair)
            != {
                "formula",
                "relative_l2_dose",
                "fp32_operation_order",
            }
            or pair.get("formula")
            != "q=clean_vjp/l2(clean_vjp); scale=dose*l2(base); plus=base+scale*q; minus=base-scale*q"
            or pair.get("fp32_operation_order")
            != "torch.float32_cpu_vector_norm_then_mul_then_add_or_sub"
            or isinstance(pair.get("relative_l2_dose"), bool)
            or not isinstance(pair.get("relative_l2_dose"), (int, float))
            or not 0.0 < float(pair["relative_l2_dose"]) < 1.0
            or not isinstance(decode, Mapping)
            or set(decode)
            != {
                "decoder_id",
                "decoder_checkpoint_receipt_digest",
                "decode_seed",
                "fps_rational",
                "width",
                "height",
                "pixel_format",
                "frame_count",
                "deterministic",
            }
            or not isinstance(decode.get("decoder_id"), str)
            or not decode["decoder_id"]
            or _SHA256_RE.fullmatch(
                str(decode.get("decoder_checkpoint_receipt_digest"))
            )
            is None
            or type(decode.get("decode_seed")) is not int
            or decode["decode_seed"] < 0
            or not isinstance(decode.get("fps_rational"), str)
            or re.fullmatch(
                r"[1-9][0-9]*/[1-9][0-9]*", decode["fps_rational"]
            )
            is None
            or type(decode.get("width")) is not int
            or decode["width"] <= 0
            or type(decode.get("height")) is not int
            or decode["height"] <= 0
            or not isinstance(decode.get("pixel_format"), str)
            or not decode["pixel_format"]
            or decode.get("frame_count") != 81
            or decode.get("deterministic") is not True
            or any(
                arm["decode_seed"] != decode["decode_seed"]
                or arm["decode_configuration_digest"]
                != object_sha256(dict(decode))
                or arm["portable_decode_receipt"]["width"] != decode["width"]
                or arm["portable_decode_receipt"]["height"] != decode["height"]
                or arm["portable_decode_receipt"]["pix_fmt"]
                != decode["pixel_format"]
                or arm["portable_decode_receipt"]["avg_frame_rate"]
                != decode["fps_rational"]
                for arm in arms
            )
            or not isinstance(decision, Mapping)
            or set(decision)
            != {
                "decoded_direction_gate_passed",
                "plus_improves_requested_action",
                "minus_does_not_match_or_exceed_plus",
                "identity_and_unedited_content_acceptable",
            }
            or any(decision.get(name) is not True for name in decision)
        ):
            raise NativeRV2VHiddenVJPError("signed exact81 direction gate binding differs")
        _validate_symmetric_gate_latents(
            clean_vjp=clean_vjp_row.values,
            runtime_clean_latent=editor._runtime_tensors["clean_latent"],
            direction=tensors["direction"],
            base=tensors["base"],
            plus=tensors["plus"],
            minus=tensors["minus"],
            relative_l2_dose=float(pair["relative_l2_dose"]),
        )


def load_validated_exact81_direction_gate(
    *,
    gate_path: str | Path,
    expected_gate_file_sha256: str,
    public_key_path: str | Path,
    expected_public_key_file_sha256: str,
    artifact_root: str | Path,
    owner: AuthenticatedOwnerQuotientPacket,
    editor: EditorSameStatePromptPacket,
    score_packet: DetachedScoreCotangent,
    clean_vjp_row: SP4SummedVJPRow,
) -> ValidatedExact81DirectionGate:
    value, resolved_gate, resolved_key, _tensors, _arms = _verify_direction_gate_payload(
        gate_path=gate_path,
        expected_gate_file_sha256=expected_gate_file_sha256,
        public_key_path=public_key_path,
        expected_public_key_file_sha256=expected_public_key_file_sha256,
        artifact_root=artifact_root,
    )
    gate = ValidatedExact81DirectionGate(
        payload=MappingProxyType(value),
        gate_path=resolved_gate,
        gate_file_sha256=_sha256(
            expected_gate_file_sha256, label="direction gate file SHA-256"
        ),
        public_key_path=resolved_key,
        public_key_file_sha256=_sha256(
            expected_public_key_file_sha256,
            label="direction gate public key SHA-256",
        ),
        artifact_root=Path(artifact_root).resolve(strict=True),
    )
    object.__setattr__(gate, "_token", _DIRECTION_GATE_TOKEN)
    gate.assert_live(
        owner=owner,
        editor=editor,
        score_packet=score_packet,
        clean_vjp_row=clean_vjp_row,
    )
    return gate


@dataclass(frozen=True)
class RankLocalVJPRow:
    query_seed: int
    sp_rank: int
    vjp_target: str
    values: Any = field(repr=False)
    score_cotangent_receipt_digest: str
    editor_packet_receipt_digest: str
    global_cotangent_identity_digest: str
    value_sha256: str
    value_norm: float
    replay_max_abs: float
    parameter_state_sha256: Optional[str]
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        self.assert_live()
        value = {
            "schema_version": VJP_ROW_SCHEMA_VERSION,
            "query_seed": self.query_seed,
            "sp_rank": self.sp_rank,
            "sp_size": SP_SIZE,
            "vjp_target": self.vjp_target,
            "action_lora_route": (
                "off_and_b_gauge_unchanged"
                if self.vjp_target == "clean_latent"
                else "native_VI_cond_target_suffix_only"
            ),
            "score_cotangent_receipt_digest": self.score_cotangent_receipt_digest,
            "editor_packet_receipt_digest": self.editor_packet_receipt_digest,
            "global_cotangent_identity_digest": (
                self.global_cotangent_identity_digest
            ),
            "value_sha256": self.value_sha256,
            "value_norm": self.value_norm,
            "replay_max_abs": self.replay_max_abs,
            "parameter_state_sha256": self.parameter_state_sha256,
            "score_divided_by_sp4_before_replay": True,
            "rank_gradient_normalized_after_replay": False,
            "optimizer_or_parameter_update": False,
        }
        return {**value, "digest": object_sha256(value)}

    def assert_live(self) -> None:
        if self._token is not _RANK_VJP_ROW_TOKEN:
            raise NativeRV2VHiddenVJPError(
                "rank-local VJP was not sealed by a replay path"
            )
        if (
            type(self.query_seed) is not int
            or self.query_seed < 0
            or self.sp_rank not in range(SP_SIZE)
            or self.vjp_target not in ("clean_latent", "lora_b")
            or not math.isfinite(float(self.value_norm))
            or self.value_norm <= 0.0
            or not math.isfinite(float(self.replay_max_abs))
            or self.replay_max_abs < 0.0
            or any(
                _SHA256_RE.fullmatch(str(value)) is None
                for value in (
                    self.score_cotangent_receipt_digest,
                    self.editor_packet_receipt_digest,
                    self.global_cotangent_identity_digest,
                    self.value_sha256,
                    self.parameter_state_sha256,
                )
            )
        ):
            raise NativeRV2VHiddenVJPError(
                "rank-local VJP scalar/provenance binding differs"
            )
        if self.vjp_target == "clean_latent":
            if (
                not isinstance(self.values, torch.Tensor)
                or self.values.dtype != torch.float32
                or self.values.requires_grad
                or self.values.grad_fn is not None
                or tensor_sha256(self.values, label="live rank clean VJP")
                != self.value_sha256
                or not math.isclose(
                    float(torch.linalg.vector_norm(self.values.double()).item()),
                    self.value_norm,
                    rel_tol=2.0e-6,
                    abs_tol=1.0e-8,
                )
            ):
                raise NativeRV2VHiddenVJPError(
                    "rank-local clean VJP bytes changed"
                )
        else:
            mapping = _validate_functional_b_mapping(
                self.values, label="live rank LoRA-B VJP"
            )
            norm = float(
                torch.sqrt(
                    sum(value.double().square().sum() for value in mapping.values())
                ).item()
            )
            if (
                _named_tensor_sha256(
                    tuple(mapping.items()), label="live rank LoRA-B VJP"
                )
                != self.value_sha256
                or not math.isclose(
                    norm, self.value_norm, rel_tol=2.0e-6, abs_tol=1.0e-8
                )
            ):
                raise NativeRV2VHiddenVJPError(
                    "rank-local LoRA-B VJP bytes changed"
                )


def _seal_rank_local_vjp_row(row: RankLocalVJPRow) -> RankLocalVJPRow:
    object.__setattr__(row, "_token", _RANK_VJP_ROW_TOKEN)
    row.assert_live()
    return row


def replay_score_cotangent_to_clean_latent(
    packet: DetachedScoreCotangent,
    *,
    owner: AuthenticatedOwnerQuotientPacket,
    replay_session: RuntimeOwnedReplaySession,
    sp_rank: int,
    clean_latent: torch.Tensor,
    action_handle: Core16ActionLoRAHandle,
) -> RankLocalVJPRow:
    """Serially replay one packet to current clean latent with LoRA frozen/off."""

    if type(packet) is not DetachedScoreCotangent:
        raise NativeRV2VHiddenVJPError("clean VJP packet differs")
    if type(sp_rank) is not int or not 0 <= sp_rank < SP_SIZE:
        raise NativeRV2VHiddenVJPError("clean VJP SP rank differs")
    if (
        not isinstance(clean_latent, torch.Tensor)
        or clean_latent.ndim != 5
        or tuple(map(int, clean_latent.shape[:3])) != (1, 16, LATENT_PHASES)
        or int(clean_latent.shape[3]) <= 0
        or int(clean_latent.shape[4]) <= 0
        or clean_latent.dtype != torch.float32
        or not clean_latent.requires_grad
        or clean_latent.grad_fn is not None
        or clean_latent.device.type == "meta"
        or not bool(torch.isfinite(clean_latent).all().item())
    ):
        raise NativeRV2VHiddenVJPError(
            "current clean latent must be a finite leaf FP32 tensor"
        )
    if (
        type(owner) is not AuthenticatedOwnerQuotientPacket
        or type(replay_session) is not RuntimeOwnedReplaySession
        or type(action_handle) is not Core16ActionLoRAHandle
        or replay_session.editor_packet.sp_rank != sp_rank
        or replay_session.editor_packet.receipt()["digest"]
        != packet.editor_packet_receipt_digest
        or owner.receipt()["digest"] != packet.owner_packet_receipt_digest
        or packet.query_seed != owner.query_seed
    ):
        raise NativeRV2VHiddenVJPError("clean VJP packet/session binding differs")
    replay_session.assert_live(owner)
    packet.assert_live(owner, replay_session.editor_packet)
    action_handle.assert_fixed_gauge()

    total = torch.zeros_like(clean_latent, dtype=torch.float32)
    maxima: list[float] = []
    runtime_snapshot = action_handle.adapter_state_snapshot()
    fatal_runtime_tamper = False
    try:
        with action_handle.frozen_b_for_clean_vjp(
            poison_check=lambda: replay_session.assert_live(owner)
        ):
            replay_rows = (
                (
                    "action",
                    replay_session.editor_packet.local_action_measurement,
                    packet.action_cotangent,
                ),
                (
                    "noop",
                    replay_session.editor_packet.local_noop_measurement,
                    packet.noop_cotangent,
                ),
            )
            for replay_index, (role, expected, cotangent) in enumerate(replay_rows):
                replay_session.assert_live(owner)
                packet.assert_live(owner, replay_session.editor_packet)
                graph, maximum = _validate_replay(
                    replay_session.replay(
                        owner=owner, role=role, adapter_enabled=False
                    ),
                    expected,
                    role=role,
                )
                gradient = torch.autograd.grad(
                    graph,
                    clean_latent,
                    grad_outputs=cotangent.to(graph.device),
                    create_graph=False,
                    # Both native forwards consume the same authenticated
                    # ``x_sigma`` object, whose clean->x_sigma autograd prefix
                    # was constructed exactly once by the runner.  Preserve
                    # that shared prefix after the action branch and release
                    # it immediately after the terminal no-op branch.
                    retain_graph=replay_index + 1 < len(replay_rows),
                    allow_unused=False,
                )[0]
                total.add_(gradient.detach().float())
                maxima.append(maximum)
                del graph, gradient
    except NativeRuntimeSealChangedError:
        fatal_runtime_tamper = True
        raise
    finally:
        if not fatal_runtime_tamper:
            if not action_handle.adapter_state_matches(runtime_snapshot):
                action_handle.restore_adapter_state(runtime_snapshot)
            if not action_handle.adapter_state_matches(runtime_snapshot):
                raise NativeRV2VHiddenVJPError(
                    "clean VJP final adapter rollback audit failed"
                )
            replay_session.assert_live(owner)
            packet.assert_live(owner, replay_session.editor_packet)
    total = total.contiguous()
    norm = torch.linalg.vector_norm(total)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
        raise NativeRV2VHiddenVJPError("clean-latent target-suffix VJP is zero/non-finite")
    packet_digest = packet.receipt()["digest"]
    return _seal_rank_local_vjp_row(RankLocalVJPRow(
        query_seed=packet.query_seed,
        sp_rank=sp_rank,
        vjp_target="clean_latent",
        values=total,
        score_cotangent_receipt_digest=packet_digest,
        editor_packet_receipt_digest=packet.editor_packet_receipt_digest,
        global_cotangent_identity_digest=(
            packet.global_cotangent_identity_digest
        ),
        value_sha256=tensor_sha256(total, label="rank-local clean VJP"),
        value_norm=float(norm.item()),
        replay_max_abs=max(maxima),
        parameter_state_sha256=action_handle.b_parameter_state_sha256(),
    ))


def replay_score_cotangent_to_lora_b(
    packet: DetachedScoreCotangent,
    *,
    owner: AuthenticatedOwnerQuotientPacket,
    replay_session: RuntimeOwnedReplaySession,
    sp_rank: int,
    action_handle: Core16ActionLoRAHandle,
    direction_gate: ValidatedExact81DirectionGate,
    clean_vjp_row: SP4SummedVJPRow,
) -> RankLocalVJPRow:
    """Serially replay to canonical LoRA-B only after an external direction gate."""

    if type(packet) is not DetachedScoreCotangent:
        raise NativeRV2VHiddenVJPError("LoRA-B VJP packet differs")
    if type(sp_rank) is not int or not 0 <= sp_rank < SP_SIZE:
        raise NativeRV2VHiddenVJPError("LoRA-B VJP SP rank differs")
    if (
        type(owner) is not AuthenticatedOwnerQuotientPacket
        or type(replay_session) is not RuntimeOwnedReplaySession
        or type(action_handle) is not Core16ActionLoRAHandle
        or type(direction_gate) is not ValidatedExact81DirectionGate
        or type(clean_vjp_row) is not SP4SummedVJPRow
        or replay_session.editor_packet.sp_rank != sp_rank
        or replay_session.editor_packet.receipt()["digest"]
        != packet.editor_packet_receipt_digest
        or owner.receipt()["digest"] != packet.owner_packet_receipt_digest
    ):
        raise NativeRV2VHiddenVJPError("LoRA-B replay authority binding differs")
    direction_gate.assert_live(
        owner=owner,
        editor=replay_session.editor_packet,
        score_packet=packet,
        clean_vjp_row=clean_vjp_row,
    )
    packet.assert_live(owner, replay_session.editor_packet)
    action_handle.assert_fixed_gauge()
    before = action_handle.b_parameter_state_sha256()
    if clean_vjp_row.parameter_state_sha256 != before:
        raise NativeRV2VHiddenVJPError(
            "LoRA-B replay clean direction was measured at a different B state"
        )
    state_snapshot = action_handle.adapter_state_snapshot()
    parameters = action_handle.canonical_b_named_parameters()
    totals = [torch.zeros_like(parameter, dtype=torch.float32) for _, parameter in parameters]
    maxima: list[float] = []
    caught: Optional[BaseException] = None
    try:
        for role, expected, cotangent in (
            (
                "action",
                replay_session.editor_packet.local_action_measurement,
                packet.action_cotangent,
            ),
            (
                "noop",
                replay_session.editor_packet.local_noop_measurement,
                packet.noop_cotangent,
            ),
        ):
            direction_gate.assert_live(
                owner=owner,
                editor=replay_session.editor_packet,
                score_packet=packet,
                clean_vjp_row=clean_vjp_row,
            )
            packet.assert_live(owner, replay_session.editor_packet)
            graph, maximum = _validate_replay(
                replay_session.replay(
                    owner=owner, role=role, adapter_enabled=True
                ),
                expected,
                role=role,
            )
            gradients = torch.autograd.grad(
                graph,
                tuple(parameter for _, parameter in parameters),
                grad_outputs=cotangent.to(graph.device),
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
            for total, gradient in zip(totals, gradients):
                total.add_(gradient.detach().float())
            maxima.append(maximum)
    except BaseException as error:
        caught = error
    finally:
        fatal_runtime_tamper = isinstance(caught, NativeRuntimeSealChangedError)
        if not fatal_runtime_tamper:
            try:
                # A failing shared_step may bypass the runner's normal
                # post-forward assertion.  Recheck the complete runtime seal
                # before any adapter-only rollback is attempted.
                replay_session.assert_live(owner)
            except NativeRuntimeSealChangedError as error:
                caught = error
                fatal_runtime_tamper = True
        if not fatal_runtime_tamper:
            state_changed = not action_handle.adapter_state_matches(state_snapshot)
            if state_changed:
                action_handle.restore_adapter_state(state_snapshot)
            if not action_handle.adapter_state_matches(state_snapshot):
                raise NativeRV2VHiddenVJPError(
                    "LoRA-B final adapter rollback audit failed"
                )
            replay_session.assert_live(owner)
            direction_gate.assert_live(
                owner=owner,
                editor=replay_session.editor_packet,
                score_packet=packet,
                clean_vjp_row=clean_vjp_row,
            )
            if caught is not None:
                raise caught
            if state_changed:
                raise NativeRV2VHiddenVJPError(
                    "LoRA-B VJP mutated adapter state and was rolled back"
                )
        elif caught is not None:
            raise caught
    if any(parameter.grad is not None for _, parameter in action_handle.canonical_a_named_parameters()):
        raise NativeRV2VHiddenVJPError("frozen LoRA-A received a gradient")
    mapping = MappingProxyType(
        {
            name: value.detach().float().cpu().contiguous()
            for (name, _), value in zip(parameters, totals)
        }
    )
    norm = torch.sqrt(
        sum(value.double().square().sum() for value in mapping.values())
    )
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
        raise NativeRV2VHiddenVJPError("LoRA-B target-suffix VJP is zero/non-finite")
    return _seal_rank_local_vjp_row(RankLocalVJPRow(
        query_seed=packet.query_seed,
        sp_rank=sp_rank,
        vjp_target="lora_b",
        values=mapping,
        score_cotangent_receipt_digest=packet.receipt()["digest"],
        editor_packet_receipt_digest=packet.editor_packet_receipt_digest,
        global_cotangent_identity_digest=(
            packet.global_cotangent_identity_digest
        ),
        value_sha256=_named_tensor_sha256(
            tuple(mapping.items()), label="rank-local LoRA-B VJP"
        ),
        value_norm=float(norm.item()),
        replay_max_abs=max(maxima),
        parameter_state_sha256=before,
    ))


def replay_score_cotangent(
    packet: DetachedScoreCotangent,
    *,
    owner: AuthenticatedOwnerQuotientPacket,
    replay_session: RuntimeOwnedReplaySession,
    vjp_target: str,
    sp_rank: int,
    action_handle: Core16ActionLoRAHandle,
    clean_latent: Optional[torch.Tensor] = None,
    direction_gate: Optional[ValidatedExact81DirectionGate] = None,
    clean_vjp_row: Optional[SP4SummedVJPRow] = None,
) -> RankLocalVJPRow:
    """Select the only two authorized VJP targets through one shared API.

    The runtime-owned session replays the fixed order ``action, noop``.  No
    generic graph callback or naked boolean direction gate is accepted.
    """

    if vjp_target == "clean_latent":
        if (
            clean_latent is None
            or direction_gate is not None
            or clean_vjp_row is not None
        ):
            raise NativeRV2VHiddenVJPError(
                "clean-latent VJP precedes and cannot consume a direction gate"
            )
        return replay_score_cotangent_to_clean_latent(
            packet,
            owner=owner,
            replay_session=replay_session,
            sp_rank=sp_rank,
            clean_latent=clean_latent,
            action_handle=action_handle,
        )
    if vjp_target == "lora_b":
        if (
            clean_latent is not None
            or direction_gate is None
            or clean_vjp_row is None
        ):
            raise NativeRV2VHiddenVJPError(
                "LoRA-B replay requires a signed exact81 gate and bound clean VJP"
            )
        return replay_score_cotangent_to_lora_b(
            packet,
            owner=owner,
            replay_session=replay_session,
            sp_rank=sp_rank,
            action_handle=action_handle,
            direction_gate=direction_gate,
            clean_vjp_row=clean_vjp_row,
        )
    raise NativeRV2VHiddenVJPError(
        "vjp_target must be exactly 'clean_latent' or 'lora_b'"
    )


def _replay_score_cotangent_unsafe_for_test(
    packet: DetachedScoreCotangent,
    *,
    vjp_target: str,
    sp_rank: int,
    replay_graph: Any,
    action_handle: Core16ActionLoRAHandle,
    clean_latent: Optional[torch.Tensor] = None,
    decoded_direction_gate_passed: bool = False,
) -> RankLocalVJPRow:
    """Private algebra fixture with the same no-base-rollback poison rule."""

    if (
        not callable(replay_graph)
        or type(action_handle) is not Core16ActionLoRAHandle
    ):
        raise NativeRV2VHiddenVJPError("unsafe test replay callback differs")
    action_handle.assert_fixed_gauge()
    adapter_parameter_ids = frozenset(
        id(parameter)
        for _, parameter in action_handle.trainable_named_parameters()
    )

    def base_receipt() -> Mapping[str, Any]:
        return _base_transformer_runtime_receipt(
            transformer=action_handle.transformer,
            adapter_parameter_ids=adapter_parameter_ids,
        )

    if vjp_target == "clean_latent":
        if clean_latent is None or decoded_direction_gate_passed:
            raise NativeRV2VHiddenVJPError("unsafe clean replay target differs")
        snapshot = action_handle.adapter_state_snapshot()
        sealed_base = base_receipt()
        total = torch.zeros_like(clean_latent)
        maxima: list[float] = []
        failure: Optional[BaseException] = None
        try:
            replay_rows = (
                ("action", packet.action_measurement, packet.action_cotangent),
                ("noop", packet.noop_measurement, packet.noop_cotangent),
            )
            for replay_index, (role, expected, cotangent) in enumerate(replay_rows):
                graph, maximum = _validate_replay(
                    replay_graph(role=role, adapter_enabled=False),
                    expected,
                    role=role,
                )
                total.add_(
                    torch.autograd.grad(
                        graph,
                        clean_latent,
                        grad_outputs=cotangent,
                        retain_graph=replay_index + 1 < len(replay_rows),
                        allow_unused=False,
                    )[0].detach()
                )
                maxima.append(maximum)
                del graph
        except BaseException as error:
            failure = error
        finally:
            try:
                observed_base = base_receipt()
            except BaseException as error:
                raise NativeRuntimeSealChangedError(
                    "unsafe clean replay changed the base model; no restoration attempted"
                ) from error
            if dict(observed_base) != dict(sealed_base):
                raise NativeRuntimeSealChangedError(
                    "unsafe clean replay changed the base model; no restoration attempted"
                ) from failure
            legal_frozen = action_handle.adapter_state_matches(snapshot)
            action_handle.restore_adapter_state(snapshot)
            if not action_handle.adapter_state_matches(snapshot):
                raise NativeRV2VHiddenVJPError(
                    "unsafe clean adapter-only rollback audit failed"
                )
            if failure is not None:
                raise failure
            if not legal_frozen:
                raise NativeRV2VHiddenVJPError(
                    "unsafe clean replay changed adapter state and was rolled back"
                )
        norm = torch.linalg.vector_norm(total)
        return _seal_rank_local_vjp_row(RankLocalVJPRow(
            query_seed=packet.query_seed,
            sp_rank=sp_rank,
            vjp_target="clean_latent",
            values=total.contiguous(),
            score_cotangent_receipt_digest=packet.receipt()["digest"],
            editor_packet_receipt_digest=packet.editor_packet_receipt_digest,
            global_cotangent_identity_digest=(
                packet.global_cotangent_identity_digest
            ),
            value_sha256=tensor_sha256(total, label="unsafe clean VJP"),
            value_norm=float(norm.item()),
            replay_max_abs=max(maxima),
            parameter_state_sha256=action_handle.b_parameter_state_sha256(),
        ))
    if vjp_target == "lora_b":
        if clean_latent is not None or decoded_direction_gate_passed is not True:
            raise NativeRV2VHiddenVJPError("unsafe test direction gate differs")
        snapshot = action_handle.adapter_state_snapshot()
        sealed_base = base_receipt()
        parameters = action_handle.canonical_b_named_parameters()
        totals = [torch.zeros_like(parameter) for _, parameter in parameters]
        maxima: list[float] = []
        failure = None
        try:
            for role, expected, cotangent in (
                ("action", packet.action_measurement, packet.action_cotangent),
                ("noop", packet.noop_measurement, packet.noop_cotangent),
            ):
                graph, maximum = _validate_replay(
                    replay_graph(role=role, adapter_enabled=True), expected, role=role
                )
                gradients = torch.autograd.grad(
                    graph,
                    tuple(parameter for _, parameter in parameters),
                    grad_outputs=cotangent,
                    allow_unused=False,
                )
                for total, gradient in zip(totals, gradients):
                    total.add_(gradient.detach())
                maxima.append(maximum)
        except BaseException as error:
            failure = error
        finally:
            try:
                observed_base = base_receipt()
            except BaseException as error:
                raise NativeRuntimeSealChangedError(
                    "unsafe LoRA-B replay changed the base model; no restoration attempted"
                ) from error
            if dict(observed_base) != dict(sealed_base):
                raise NativeRuntimeSealChangedError(
                    "unsafe LoRA-B replay changed the base model; no restoration attempted"
                ) from failure
            changed = not action_handle.adapter_state_matches(snapshot)
            if changed:
                action_handle.restore_adapter_state(snapshot)
            if not action_handle.adapter_state_matches(snapshot):
                raise NativeRV2VHiddenVJPError(
                    "unsafe LoRA-B adapter-only rollback audit failed"
                )
            if failure is not None:
                raise failure
            if changed:
                raise NativeRV2VHiddenVJPError(
                    "unsafe LoRA-B replay changed state and was rolled back"
                )
        mapping = MappingProxyType(
            {
                name: total.contiguous()
                .detach()
                .float()
                .cpu()
                for (name, _), total in zip(parameters, totals)
            }
        )
        norm = torch.sqrt(
            sum(value.double().square().sum() for value in mapping.values())
        )
        return _seal_rank_local_vjp_row(RankLocalVJPRow(
            query_seed=packet.query_seed,
            sp_rank=sp_rank,
            vjp_target="lora_b",
            values=mapping,
            score_cotangent_receipt_digest=packet.receipt()["digest"],
            editor_packet_receipt_digest=packet.editor_packet_receipt_digest,
            global_cotangent_identity_digest=(
                packet.global_cotangent_identity_digest
            ),
            value_sha256=_named_tensor_sha256(
                tuple(mapping.items()), label="unsafe LoRA-B VJP"
            ),
            value_norm=float(norm.item()),
            replay_max_abs=max(maxima),
            parameter_state_sha256=action_handle.b_parameter_state_sha256(),
        ))
    raise NativeRV2VHiddenVJPError(
        "vjp_target must be exactly 'clean_latent' or 'lora_b'"
    )


@dataclass(frozen=True)
class SP4SummedVJPRow:
    query_seed: int
    vjp_target: str
    values: Any = field(repr=False)
    value_sha256: str
    value_norm: float
    rank_row_receipt_digests: tuple[str, str, str, str]
    rank_editor_packet_receipt_digests: tuple[str, str, str, str]
    rank_score_cotangent_receipt_digests: tuple[str, str, str, str]
    global_cotangent_identity_digest: str
    parameter_state_sha256: Optional[str]
    _rank_rows: tuple[RankLocalVJPRow, ...] = field(
        default=(), repr=False, compare=False
    )
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        self.assert_live()
        value = {
            "schema_version": SP4_ROW_SCHEMA_VERSION,
            "query_seed": self.query_seed,
            "vjp_target": self.vjp_target,
            "sp_size": SP_SIZE,
            "aggregation": "SUM",
            "score_divided_by_sp4_before_rank_replay": True,
            "divide_after_sum": False,
            "normalization_count": 1,
            "value_sha256": self.value_sha256,
            "value_norm": self.value_norm,
            "rank_row_receipt_digests": list(self.rank_row_receipt_digests),
            "rank_editor_packet_receipt_digests": list(
                self.rank_editor_packet_receipt_digests
            ),
            "rank_score_cotangent_receipt_digests": list(
                self.rank_score_cotangent_receipt_digests
            ),
            "global_cotangent_identity_digest": (
                self.global_cotangent_identity_digest
            ),
            "parameter_state_sha256": self.parameter_state_sha256,
            "optimizer_or_parameter_update": False,
        }
        return {**value, "digest": object_sha256(value)}

    def assert_live(self) -> None:
        if self._token is not _SP4_VJP_ROW_TOKEN:
            raise NativeRV2VHiddenVJPError(
                "SP4 VJP was not sealed by the four-rank SUM"
            )
        digest_sets = (
            self.rank_row_receipt_digests,
            self.rank_editor_packet_receipt_digests,
            self.rank_score_cotangent_receipt_digests,
        )
        if (
            self.vjp_target not in ("clean_latent", "lora_b")
            or type(self.query_seed) is not int
            or self.query_seed < 0
            or not math.isfinite(float(self.value_norm))
            or self.value_norm <= 0.0
            or any(
                len(rows) != SP_SIZE
                or len(set(rows)) != SP_SIZE
                or any(_SHA256_RE.fullmatch(value) is None for value in rows)
                for rows in digest_sets
            )
            or _SHA256_RE.fullmatch(self.global_cotangent_identity_digest) is None
            or _SHA256_RE.fullmatch(str(self.parameter_state_sha256)) is None
        ):
            raise NativeRV2VHiddenVJPError("SP4 VJP live provenance differs")
        if len(self._rank_rows) != SP_SIZE or any(
            type(row) is not RankLocalVJPRow for row in self._rank_rows
        ):
            raise NativeRV2VHiddenVJPError(
                "SP4 VJP retained rank-row provenance differs"
            )
        ordered = tuple(sorted(self._rank_rows, key=lambda row: row.sp_rank))
        for row in ordered:
            row.assert_live()
        if (
            len(ordered) != SP_SIZE
            or tuple(row.sp_rank for row in ordered) != tuple(range(SP_SIZE))
            or any(
                row.query_seed != self.query_seed
                or row.vjp_target != self.vjp_target
                or row.global_cotangent_identity_digest
                != self.global_cotangent_identity_digest
                or row.parameter_state_sha256 != self.parameter_state_sha256
                for row in ordered
            )
            or tuple(row.receipt()["digest"] for row in ordered)
            != self.rank_row_receipt_digests
            or tuple(row.editor_packet_receipt_digest for row in ordered)
            != self.rank_editor_packet_receipt_digests
            or tuple(row.score_cotangent_receipt_digest for row in ordered)
            != self.rank_score_cotangent_receipt_digests
        ):
            raise NativeRV2VHiddenVJPError(
                "SP4 VJP retained rank-row provenance differs"
            )
        if self.vjp_target == "clean_latent":
            expected_sum = torch.stack(
                [row.values for row in ordered], dim=0
            ).sum(dim=0).contiguous()
            if (
                not isinstance(self.values, torch.Tensor)
                or self.values.dtype != torch.float32
                or self.values.requires_grad
                or self.values.grad_fn is not None
                or not torch.equal(self.values, expected_sum)
                or tensor_sha256(self.values, label="live SP4 clean VJP")
                != self.value_sha256
                or not math.isclose(
                    float(torch.linalg.vector_norm(self.values.double()).item()),
                    self.value_norm,
                    rel_tol=2.0e-6,
                    abs_tol=1.0e-8,
                )
            ):
                raise NativeRV2VHiddenVJPError("SP4 clean VJP bytes changed")
        else:
            mapping = _validate_functional_b_mapping(
                self.values, label="live SP4 LoRA-B VJP"
            )
            norm = float(
                torch.sqrt(
                    sum(value.double().square().sum() for value in mapping.values())
                ).item()
            )
            if (
                any(
                    not torch.equal(
                        mapping[name],
                        torch.stack(
                            [row.values[name] for row in ordered], dim=0
                        ).sum(dim=0).contiguous(),
                    )
                    for name in CANONICAL_B_PARAMETER_NAMES
                )
                or
                _named_tensor_sha256(
                    tuple(mapping.items()), label="live SP4 LoRA-B VJP"
                )
                != self.value_sha256
                or not math.isclose(
                    norm, self.value_norm, rel_tol=2.0e-6, abs_tol=1.0e-8
                )
            ):
                raise NativeRV2VHiddenVJPError("SP4 LoRA-B VJP bytes changed")


def _sum_rank_local_vjp_rows_unsafe_for_test(
    rows: Sequence[RankLocalVJPRow],
) -> SP4SummedVJPRow:
    """Local algebra fixture; production uses the runner's live collective."""

    if len(rows) != SP_SIZE:
        raise NativeRV2VHiddenVJPError("SP4 VJP aggregation requires four rows")
    if any(
        not isinstance(row, RankLocalVJPRow)
        or row._token is not _RANK_VJP_ROW_TOKEN
        for row in rows
    ):
        raise NativeRV2VHiddenVJPError("SP4 VJP contains an invalid row type")
    for row in rows:
        row.assert_live()
    ordered = sorted(rows, key=lambda row: row.sp_rank)
    if [row.sp_rank for row in ordered] != list(range(SP_SIZE)):
        raise NativeRV2VHiddenVJPError("SP4 VJP rank order differs")
    first = ordered[0]
    if any(
        not isinstance(row, RankLocalVJPRow)
        or row.query_seed != first.query_seed
        or row.vjp_target != first.vjp_target
        or row.global_cotangent_identity_digest
        != first.global_cotangent_identity_digest
        or row.parameter_state_sha256 != first.parameter_state_sha256
        for row in ordered
    ):
        raise NativeRV2VHiddenVJPError("SP4 VJP row provenance differs")
    editor_receipts = tuple(row.editor_packet_receipt_digest for row in ordered)
    cotangent_receipts = tuple(
        row.score_cotangent_receipt_digest for row in ordered
    )
    if (
        len(set(editor_receipts)) != SP_SIZE
        or len(set(cotangent_receipts)) != SP_SIZE
        or any(
            _SHA256_RE.fullmatch(value) is None
            for value in (
                *editor_receipts,
                *cotangent_receipts,
                first.global_cotangent_identity_digest,
            )
        )
    ):
        raise NativeRV2VHiddenVJPError(
            "SP4 requires four distinct rank-local editor/cotangent receipts"
        )
    if first.vjp_target == "clean_latent":
        tensors = [row.values for row in ordered]
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != tensors[0].shape
            or value.dtype != torch.float32
            or value.device != tensors[0].device
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
            for value in tensors
        ):
            raise NativeRV2VHiddenVJPError("clean SP4 VJP tensor closure differs")
        for row, value in zip(ordered, tensors):
            norm = float(torch.linalg.vector_norm(value.double()).item())
            if (
                tensor_sha256(value, label="rank-local clean VJP audit")
                != row.value_sha256
                or not math.isclose(
                    norm, row.value_norm, rel_tol=2.0e-6, abs_tol=1.0e-8
                )
            ):
                raise NativeRV2VHiddenVJPError(
                    "rank-local clean VJP value binding differs"
                )
        summed: Any = torch.stack(tensors, dim=0).sum(dim=0).contiguous()
        digest = tensor_sha256(summed, label="SP4-summed clean VJP")
        norm = torch.linalg.vector_norm(summed.double())
    elif first.vjp_target == "lora_b":
        mappings = [row.values for row in ordered]
        if any(
            not isinstance(mapping, Mapping)
            or tuple(mapping) != CANONICAL_B_PARAMETER_NAMES
            for mapping in mappings
        ):
            raise NativeRV2VHiddenVJPError("LoRA-B SP4 mapping order differs")
        for row, mapping in zip(ordered, mappings):
            norm = float(
                torch.sqrt(
                    sum(value.double().square().sum() for value in mapping.values())
                ).item()
            )
            if (
                _named_tensor_sha256(
                    tuple(mapping.items()), label="rank-local LoRA-B VJP audit"
                )
                != row.value_sha256
                or not math.isclose(
                    norm, row.value_norm, rel_tol=2.0e-6, abs_tol=1.0e-8
                )
            ):
                raise NativeRV2VHiddenVJPError(
                    "rank-local LoRA-B VJP value binding differs"
                )
        summed_mapping: dict[str, torch.Tensor] = {}
        for name in CANONICAL_B_PARAMETER_NAMES:
            values = [mapping[name] for mapping in mappings]
            if any(
                not isinstance(value, torch.Tensor)
                or tuple(map(int, value.shape)) != CANONICAL_B_SHAPE
                or value.dtype != torch.float32
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
                for value in values
            ):
                raise NativeRV2VHiddenVJPError(f"SP4 LoRA-B row {name} differs")
            summed_mapping[name] = torch.stack(values, dim=0).sum(dim=0).contiguous()
        summed = MappingProxyType(summed_mapping)
        digest = _named_tensor_sha256(
            tuple(summed.items()), label="SP4-summed LoRA-B VJP"
        )
        norm = torch.sqrt(
            sum(value.double().square().sum() for value in summed.values())
        )
    else:
        raise NativeRV2VHiddenVJPError("unknown VJP target")
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
        raise NativeRV2VHiddenVJPError("SP4-summed VJP is zero/non-finite")
    result = SP4SummedVJPRow(
        query_seed=first.query_seed,
        vjp_target=first.vjp_target,
        values=summed,
        value_sha256=digest,
        value_norm=float(norm.item()),
        rank_row_receipt_digests=tuple(
            row.receipt()["digest"] for row in ordered
        ),  # type: ignore[arg-type]
        rank_editor_packet_receipt_digests=editor_receipts,  # type: ignore[arg-type]
        rank_score_cotangent_receipt_digests=cotangent_receipts,  # type: ignore[arg-type]
        global_cotangent_identity_digest=(
            first.global_cotangent_identity_digest
        ),
        parameter_state_sha256=first.parameter_state_sha256,
        _rank_rows=tuple(ordered),
    )
    object.__setattr__(result, "_token", _SP4_VJP_ROW_TOKEN)
    result.assert_live()
    return result


def _validate_functional_b_mapping(
    values: Any, *, label: str
) -> Mapping[str, torch.Tensor]:
    if not isinstance(values, Mapping) or tuple(values) != CANONICAL_B_PARAMETER_NAMES:
        raise NativeRV2VHiddenVJPError(f"{label} canonical B ordering differs")
    result: dict[str, torch.Tensor] = {}
    for name in CANONICAL_B_PARAMETER_NAMES:
        value = values[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or tuple(map(int, value.shape)) != CANONICAL_B_SHAPE
            or value.requires_grad
            or value.grad_fn is not None
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise NativeRV2VHiddenVJPError(
                f"{label} {name} must be detached contiguous CPU FP32"
            )
        result[name] = value
    return MappingProxyType(result)


@dataclass(frozen=True)
class FunctionalPreservationVJPRow:
    row_id: str
    functional_id: str
    qp_family: str
    native_contrast: str
    rademacher_seed: int
    weak_i_axis_slab: bool
    values: Mapping[str, torch.Tensor] = field(repr=False)
    maximum_absolute_dot: float
    same_state_binding_digest: str
    clean_vjp_receipt_digest: str
    checkpoint_content_receipt_digest: str
    parameter_state_sha256: str
    sp4_editor_packet_receipt_digests: tuple[str, str, str, str]
    sp4_rank_vjp_receipt_digests: tuple[str, str, str, str]
    value_sha256: str
    value_norm: float
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        self.assert_live()
        value = {
            "schema_version": FUNCTIONAL_PRESERVATION_SCHEMA_VERSION,
            "row_id": self.row_id,
            "functional_id": self.functional_id,
            "qp_family": self.qp_family,
            "native_contrast": self.native_contrast,
            "rademacher_seed": self.rademacher_seed,
            "weak_i_axis_slab": self.weak_i_axis_slab,
            "maximum_absolute_dot": self.maximum_absolute_dot,
            "same_state_binding_digest": self.same_state_binding_digest,
            "clean_vjp_receipt_digest": self.clean_vjp_receipt_digest,
            "checkpoint_content_receipt_digest": self.checkpoint_content_receipt_digest,
            "parameter_state_sha256": self.parameter_state_sha256,
            "sp4_editor_packet_receipt_digests": list(
                self.sp4_editor_packet_receipt_digests
            ),
            "sp4_rank_vjp_receipt_digests": list(
                self.sp4_rank_vjp_receipt_digests
            ),
            "value_sha256": self.value_sha256,
            "value_norm": self.value_norm,
            "each_row_is_an_independent_qp_slab": True,
            "averaged_with_another_functional_or_seed": False,
        }
        return {**value, "digest": object_sha256(value)}

    def assert_live(self) -> None:
        if self._token is not _FUNCTIONAL_ROW_TOKEN:
            raise NativeRV2VHiddenVJPError(
                "functional preservation row is not runtime-sealed"
            )
        mapping = _validate_functional_b_mapping(
            self.values, label=f"functional row {self.row_id}"
        )
        norm = float(
            torch.sqrt(
                sum(value.double().square().sum() for value in mapping.values())
            ).item()
        )
        if self.functional_id == WEAK_I_AXIS_FUNCTIONAL_ID:
            expected_semantics = ("identity", "I-minus-none", True)
        else:
            spec = FUNCTIONAL_PRESERVATION_SPECS.get(self.functional_id)
            if spec is None:
                raise NativeRV2VHiddenVJPError(
                    "functional row ID is outside the closed registry"
                )
            expected_semantics = (spec[0], spec[1], False)
        if (
            (
                self.qp_family,
                self.native_contrast,
                self.weak_i_axis_slab,
            )
            != expected_semantics
            or not self.row_id.endswith(
                f":{self.functional_id}:r{self.rademacher_seed}"
            )
            or not self.row_id[: -len(
                f":{self.functional_id}:r{self.rademacher_seed}"
            )]
            or
            _named_tensor_sha256(
                tuple(mapping.items()), label=f"functional row {self.row_id} live"
            )
            != self.value_sha256
            or not math.isclose(norm, self.value_norm, rel_tol=2.0e-6, abs_tol=1.0e-10)
            or not math.isfinite(self.value_norm)
            or self.value_norm <= 0.0
            or not math.isfinite(self.maximum_absolute_dot)
            or self.maximum_absolute_dot < 0.0
            or self.rademacher_seed not in PRESERVATION_RADEMACHER_SEEDS
            or len(self.sp4_editor_packet_receipt_digests) != SP_SIZE
            or len(set(self.sp4_editor_packet_receipt_digests)) != SP_SIZE
            or len(self.sp4_rank_vjp_receipt_digests) != SP_SIZE
            or len(set(self.sp4_rank_vjp_receipt_digests)) != SP_SIZE
            or any(_SHA256_RE.fullmatch(value) is None for value in (
                self.same_state_binding_digest,
                self.clean_vjp_receipt_digest,
                self.checkpoint_content_receipt_digest,
                self.parameter_state_sha256,
                *self.sp4_editor_packet_receipt_digests,
                *self.sp4_rank_vjp_receipt_digests,
            ))
        ):
            raise NativeRV2VHiddenVJPError(
                "functional preservation row live binding changed"
            )


def _seal_functional_preservation_row_from_runtime_sp4(
    *,
    owner: AuthenticatedOwnerQuotientPacket,
    editor: EditorSameStatePromptPacket,
    score_packet: DetachedScoreCotangent,
    clean_vjp_row: SP4SummedVJPRow,
    functional_id: str,
    rademacher_seed: int,
    values: Mapping[str, torch.Tensor],
    maximum_absolute_dot: float,
    sp4_editor_packet_receipt_digests: Sequence[str],
    sp4_rank_vjp_receipt_digests: Sequence[str],
    parameter_state_sha256: str,
) -> FunctionalPreservationVJPRow:
    """Private post-all-reduce seal used by the closed native collector."""

    score_packet.assert_live(owner, editor)
    if type(clean_vjp_row) is not SP4SummedVJPRow:
        raise NativeRV2VHiddenVJPError("functional clean VJP receipt differs")
    clean_vjp_row.assert_live()
    clean_digest = clean_vjp_row.receipt()["digest"]
    if (
        clean_vjp_row.vjp_target != "clean_latent"
        or clean_vjp_row.query_seed != owner.query_seed
        or clean_vjp_row.global_cotangent_identity_digest
        != score_packet.global_cotangent_identity_digest
        or clean_vjp_row.rank_editor_packet_receipt_digests[editor.sp_rank]
        != editor.receipt()["digest"]
        or clean_vjp_row.rank_score_cotangent_receipt_digests[editor.sp_rank]
        != score_packet.receipt()["digest"]
    ):
        raise NativeRV2VHiddenVJPError(
            "functional rows do not share the action clean-VJP state"
        )
    weak = functional_id == WEAK_I_AXIS_FUNCTIONAL_ID
    if weak:
        qp_family, contrast = "identity", "I-minus-none"
    else:
        spec = FUNCTIONAL_PRESERVATION_SPECS.get(functional_id)
        if spec is None:
            raise NativeRV2VHiddenVJPError("functional preservation ID differs")
        qp_family, contrast = spec
    if rademacher_seed not in PRESERVATION_RADEMACHER_SEEDS:
        raise NativeRV2VHiddenVJPError("functional Rademacher seed differs")
    bound = float(maximum_absolute_dot)
    if (
        isinstance(maximum_absolute_dot, bool)
        or not math.isfinite(bound)
        or bound < 0.0
    ):
        raise NativeRV2VHiddenVJPError("functional preservation slab differs")
    mapping = _validate_functional_b_mapping(values, label="SP4 functional VJP")
    norm = float(
        torch.sqrt(sum(value.double().square().sum() for value in mapping.values())).item()
    )
    if not math.isfinite(norm) or norm <= 0.0:
        raise NativeRV2VHiddenVJPError("functional preservation VJP is zero/non-finite")
    editor_digests = tuple(sp4_editor_packet_receipt_digests)
    rank_digests = tuple(sp4_rank_vjp_receipt_digests)
    if (
        len(editor_digests) != SP_SIZE
        or len(rank_digests) != SP_SIZE
        or len(set(editor_digests)) != SP_SIZE
        or len(set(rank_digests)) != SP_SIZE
        or editor_digests[editor.sp_rank] != editor.receipt()["digest"]
        or editor_digests != clean_vjp_row.rank_editor_packet_receipt_digests
        or any(_SHA256_RE.fullmatch(str(value)) is None for value in (*editor_digests, *rank_digests))
    ):
        raise NativeRV2VHiddenVJPError("functional SP4 provenance differs")
    checkpoint_digest = _sha256(
        editor.bindings.get("checkpoint_content_receipt_digest"),
        label="functional checkpoint content receipt",
    )
    state_digest = _sha256(
        parameter_state_sha256, label="functional parameter state"
    )
    runtime_hashes = {
        name: binding.get("tensor_sha256")
        for name, binding in editor._runtime_tensor_bindings.items()
    }
    required_runtime = {
        "source_latent",
        "clean_latent",
        "initial_noise",
        "x_sigma",
        "action_condition",
        "noop_condition",
        "timestep",
        "image_reference_0",
        "image_reference_1",
        "image_reference_2",
        "image_reference_3",
    }
    if set(runtime_hashes) != required_runtime or any(
        _SHA256_RE.fullmatch(str(value)) is None for value in runtime_hashes.values()
    ):
        raise NativeRV2VHiddenVJPError(
            "functional source/prompt/noise runtime closure differs"
        )
    state_binding = {
        "cell_id": owner.cell_id,
        "query_seed": owner.query_seed,
        "source_iid": owner.source_iid,
        "source_video_sha256": owner.source_video_sha256,
        "action_prompt_sha256": owner.action_prompt_sha256,
        "noop_prompt_sha256": owner.noop_prompt_sha256,
        "owner_packet_receipt_digest": owner.receipt()["digest"],
        "clean_vjp_receipt_digest": clean_digest,
        "native_schedule_index": NATIVE_SCHEDULE_INDEX,
        "native_timestep": NATIVE_TIMESTEP,
        "native_sigma_float64_hex": float(NATIVE_SIGMA).hex(),
        "native_branch_contract": "same_source_native_none_V_VI_state",
        "checkpoint_content_receipt_digest": checkpoint_digest,
        "runtime_tensor_value_sha256": runtime_hashes,
    }
    row = FunctionalPreservationVJPRow(
        row_id=f"{owner.cell_id}:{functional_id}:r{rademacher_seed}",
        functional_id=functional_id,
        qp_family=qp_family,
        native_contrast=contrast,
        rademacher_seed=rademacher_seed,
        weak_i_axis_slab=weak,
        values=mapping,
        maximum_absolute_dot=bound,
        same_state_binding_digest=object_sha256(state_binding),
        clean_vjp_receipt_digest=clean_digest,
        checkpoint_content_receipt_digest=checkpoint_digest,
        parameter_state_sha256=state_digest,
        sp4_editor_packet_receipt_digests=editor_digests,  # type: ignore[arg-type]
        sp4_rank_vjp_receipt_digests=rank_digests,  # type: ignore[arg-type]
        value_sha256=_named_tensor_sha256(
            tuple(mapping.items()), label="sealed SP4 functional VJP"
        ),
        value_norm=norm,
    )
    object.__setattr__(row, "_token", _FUNCTIONAL_ROW_TOKEN)
    row.assert_live()
    return row


@dataclass(frozen=True)
class PairedFunctionalPreservationCone:
    rows: tuple[FunctionalPreservationVJPRow, ...]
    same_state_binding_digest: str
    clean_vjp_receipt_digest: str
    owner_packet_receipt_digest: str
    functional_contract_digest: str
    _owner: AuthenticatedOwnerQuotientPacket = field(repr=False, compare=False)
    _editor: EditorSameStatePromptPacket = field(repr=False, compare=False)
    _score_packet: DetachedScoreCotangent = field(repr=False, compare=False)
    _clean_vjp_row: SP4SummedVJPRow = field(repr=False, compare=False)
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        self.assert_live()
        value = {
            "schema_version": FUNCTIONAL_PRESERVATION_SCHEMA_VERSION,
            "same_state_binding_digest": self.same_state_binding_digest,
            "clean_vjp_receipt_digest": self.clean_vjp_receipt_digest,
            "owner_packet_receipt_digest": self.owner_packet_receipt_digest,
            "functional_contract_digest": self.functional_contract_digest,
            "row_receipt_digests": [row.receipt()["digest"] for row in self.rows],
            "row_count": len(self.rows),
            "rademacher_seeds": list(PRESERVATION_RADEMACHER_SEEDS),
            "i_axis_role": "weak_slab_only",
            "paired_functional_cone_is_primary_identity_constraint": True,
            "each_row_constrained_independently": True,
            "row_averaging": False,
            "qp_infeasible_returns_exact_zero": True,
        }
        return {**value, "digest": object_sha256(value)}

    def assert_live(self) -> None:
        if self._token is not _FUNCTIONAL_CONE_TOKEN:
            raise NativeRV2VHiddenVJPError(
                "functional preservation cone is not runtime-sealed"
            )
        self._score_packet.assert_live(self._owner, self._editor)
        self._clean_vjp_row.assert_live()
        required = {
            (functional_id, seed)
            for functional_id in (
                WEAK_I_AXIS_FUNCTIONAL_ID,
                *FUNCTIONAL_PRESERVATION_SPECS.keys(),
            )
            for seed in PRESERVATION_RADEMACHER_SEEDS
        }
        observed = {
            (row.functional_id, row.rademacher_seed) for row in self.rows
        }
        if (
            len(self.rows) != len(required)
            or len(observed) != len(self.rows)
            or observed != required
            or
            self._clean_vjp_row.receipt()["digest"]
            != self.clean_vjp_receipt_digest
            or self.owner_packet_receipt_digest != self._owner.receipt()["digest"]
            or self.functional_contract_digest
            != NativeSharedStepSP4ReplayRunner.functional_preservation_contract()[
                "digest"
            ]
            or any(row.same_state_binding_digest != self.same_state_binding_digest for row in self.rows)
            or len({row.row_id for row in self.rows}) != len(self.rows)
            or len({row.value_sha256 for row in self.rows}) != len(self.rows)
        ):
            raise NativeRV2VHiddenVJPError(
                "functional preservation cone live binding changed"
            )
        for row in self.rows:
            row.assert_live()

    def to_qp_rows(self, layout: Any) -> tuple[Any, ...]:
        """Convert every sealed row one-for-one; never average rows or seeds."""

        self.assert_live()
        import mosaic_starc_stateless_jacobian_qp as qp

        if not isinstance(layout, qp.FixedParameterLayout):
            raise NativeRV2VHiddenVJPError("functional cone QP layout differs")
        if layout.names != CANONICAL_B_PARAMETER_NAMES:
            raise NativeRV2VHiddenVJPError("functional cone canonical layout differs")
        if any(
            row.parameter_state_sha256 != layout.parameter_state_sha256
            for row in self.rows
        ):
            raise NativeRV2VHiddenVJPError(
                "functional cone is bound to a different LoRA-B state"
            )
        result = []
        for row in self.rows:
            flat = torch.cat(
                [row.values[name].reshape(-1) for name in CANONICAL_B_PARAMETER_NAMES]
            ).contiguous()
            layout.validate_row(flat, label=f"functional QP row {row.row_id}")
            result.append(
                qp.PreservationConstraintRow(
                    row_id=row.row_id,
                    family=row.qp_family,
                    values=flat,
                    maximum_absolute_dot=row.maximum_absolute_dot,
                    layout_digest=layout.layout_digest,
                    checkpoint_content_receipt_digest=(
                        row.checkpoint_content_receipt_digest
                    ),
                    parameter_state_sha256=row.parameter_state_sha256,
                    gradient_computation_receipt_digest=row.receipt()["digest"],
                )
            )
        return tuple(result)


def _seal_paired_functional_preservation_cone_from_runtime(
    *,
    owner: AuthenticatedOwnerQuotientPacket,
    editor: EditorSameStatePromptPacket,
    score_packet: DetachedScoreCotangent,
    clean_vjp_row: SP4SummedVJPRow,
    rows: Sequence[FunctionalPreservationVJPRow],
) -> PairedFunctionalPreservationCone:
    score_packet.assert_live(owner, editor)
    if type(clean_vjp_row) is not SP4SummedVJPRow:
        raise NativeRV2VHiddenVJPError("functional cone clean SP4 row differs")
    clean_vjp_row.assert_live()
    required = {
        (functional_id, seed)
        for functional_id in (
            WEAK_I_AXIS_FUNCTIONAL_ID,
            *FUNCTIONAL_PRESERVATION_SPECS.keys(),
        )
        for seed in PRESERVATION_RADEMACHER_SEEDS
    }
    observed = {(row.functional_id, row.rademacher_seed) for row in rows}
    if (
        len(rows) != len(required)
        or observed != required
        or len(observed) != len(rows)
        or any(type(row) is not FunctionalPreservationVJPRow for row in rows)
    ):
        raise NativeRV2VHiddenVJPError(
            "functional cone requires every fixed Rademacher row independently"
        )
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                0 if row.functional_id == WEAK_I_AXIS_FUNCTIONAL_ID else 1,
                row.functional_id,
                row.rademacher_seed,
            ),
        )
    )
    for row in ordered:
        row.assert_live()
    state_digests = {row.same_state_binding_digest for row in ordered}
    clean_digests = {row.clean_vjp_receipt_digest for row in ordered}
    weak_bounds = [row.maximum_absolute_dot for row in ordered if row.weak_i_axis_slab]
    functional_bounds = [
        row.maximum_absolute_dot for row in ordered if not row.weak_i_axis_slab
    ]
    if (
        len(state_digests) != 1
        or clean_digests != {clean_vjp_row.receipt()["digest"]}
        or len({row.parameter_state_sha256 for row in ordered}) != 1
        or len({row.checkpoint_content_receipt_digest for row in ordered}) != 1
        or len({row.sp4_editor_packet_receipt_digests for row in ordered}) != 1
        or len(weak_bounds) != len(PRESERVATION_RADEMACHER_SEEDS)
        or min(weak_bounds) < 4.0 * max(functional_bounds)
        or len({row.value_sha256 for row in ordered}) != len(ordered)
        or {row.qp_family for row in ordered}
        != {"identity", "camera", "background", "sharpness", "flicker", "noop"}
    ):
        raise NativeRV2VHiddenVJPError(
            "functional cone state/family/weak-I policy differs"
        )
    cone = PairedFunctionalPreservationCone(
        rows=ordered,
        same_state_binding_digest=next(iter(state_digests)),
        clean_vjp_receipt_digest=next(iter(clean_digests)),
        owner_packet_receipt_digest=owner.receipt()["digest"],
        functional_contract_digest=(
            NativeSharedStepSP4ReplayRunner.functional_preservation_contract()[
                "digest"
            ]
        ),
        _owner=owner,
        _editor=editor,
        _score_packet=score_packet,
        _clean_vjp_row=clean_vjp_row,
    )
    object.__setattr__(cone, "_token", _FUNCTIONAL_CONE_TOKEN)
    cone.assert_live()
    return cone


@dataclass(frozen=True)
class TwoQuerySeedActionRows:
    ordered_query_seeds: tuple[int, int]
    rows: tuple[SP4SummedVJPRow, SP4SummedVJPRow]

    def receipt(self) -> Mapping[str, Any]:
        for row in self.rows:
            row.assert_live()
        value = {
            "schema_version": SCHEMA_VERSION,
            "ordered_query_seeds": list(self.ordered_query_seeds),
            "vjp_target": "lora_b",
            "row_receipt_digests": [row.receipt()["digest"] for row in self.rows],
            "rows_kept_independent": True,
            "seed_averaging": False,
            "seed_ranking_or_selection": False,
            "failure_policy": "null_no_partial_row_no_update",
            "optimizer_or_parameter_update": False,
            "real_auh_runtime_validated": False,
        }
        return {**value, "digest": object_sha256(value)}


def build_two_query_seed_action_rows(
    rows: Sequence[SP4SummedVJPRow],
    *,
    ordered_query_seeds: Sequence[int],
) -> TwoQuerySeedActionRows:
    seeds = tuple(ordered_query_seeds)
    if (
        len(seeds) != QUERY_SEED_COUNT
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != QUERY_SEED_COUNT
        or len(rows) != QUERY_SEED_COUNT
    ):
        raise NativeRV2VHiddenVJPError("exactly two distinct fixed query seeds are required")
    by_seed = {row.query_seed: row for row in rows}
    if (
        set(by_seed) != set(seeds)
        or len(by_seed) != QUERY_SEED_COUNT
        or any(row.vjp_target != "lora_b" for row in by_seed.values())
    ):
        raise NativeRV2VHiddenVJPError("two-seed LoRA-B row closure differs")
    ordered = (by_seed[seeds[0]], by_seed[seeds[1]])
    if ordered[0].values is ordered[1].values:
        raise NativeRV2VHiddenVJPError("two query seeds alias one action row object")
    for row in ordered:
        row.assert_live()
        mapping = row.values
        if (
            not isinstance(mapping, Mapping)
            or tuple(mapping) != CANONICAL_B_PARAMETER_NAMES
            or any(
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != CANONICAL_B_SHAPE
                or value.dtype != torch.float32
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
                for value in mapping.values()
            )
            or _named_tensor_sha256(
                tuple(mapping.items()), label="two-seed LoRA-B row audit"
            )
            != row.value_sha256
        ):
            raise NativeRV2VHiddenVJPError(
                "two-seed LoRA-B row value binding differs"
            )
    return TwoQuerySeedActionRows(
        ordered_query_seeds=seeds,  # type: ignore[arg-type]
        rows=ordered,
    )


def try_build_two_query_seed_action_rows(
    rows: Sequence[SP4SummedVJPRow],
    *,
    ordered_query_seeds: Sequence[int],
) -> Optional[TwoQuerySeedActionRows]:
    """The sole fail-to-null boundary; never return one surviving seed row."""

    try:
        return build_two_query_seed_action_rows(
            rows, ordered_query_seeds=ordered_query_seeds
        )
    except (NativeRV2VHiddenVJPError, TypeError, AttributeError):
        return None


__all__ = [
    "ACTION_BLOCK_INDICES",
    "AuthenticatedSP4Collective",
    "AuthenticatedOwnerQuotientPacket",
    "AuthenticatedEditorRuntimeInputPacket",
    "Block15TargetSuffixObserver",
    "CANONICAL_A_PARAMETER_NAMES",
    "CANONICAL_A_SHAPE",
    "CANONICAL_B_PARAMETER_COUNT",
    "CANONICAL_B_PARAMETER_NAMES",
    "CANONICAL_B_SHAPE",
    "CONDITION_PHASES_BY_BRANCH",
    "Core16ActionLoRAHandle",
    "Core16ExactZeroTargetRowActionLoRA",
    "Core16ZeroRouteProofHolder",
    "DetachedScoreCotangent",
    "EditorSameStatePromptPacket",
    "FIXED_LORA_A_SEED",
    "FUNCTIONAL_PRESERVATION_SPECS",
    "FunctionalPreservationVJPRow",
    "HIDDEN_SIZE",
    "HOOK_BLOCK_INDEX",
    "LATENT_PHASES",
    "LORA_ALPHA",
    "LORA_RANK",
    "NATIVE_SCHEDULE_INDEX",
    "NativeRuntimeSealChangedError",
    "NativeSharedStepSP4ReplayRunner",
    "NativeRV2VHiddenVJPError",
    "NativeTargetSuffixLayout",
    "PRESERVATION_RADEMACHER_SEEDS",
    "PairedFunctionalPreservationCone",
    "PACKED_PREDICTION_DIM",
    "QUERY_SEED_COUNT",
    "RankLocalVJPRow",
    "RuntimeOwnedReplaySession",
    "SP4SummedVJPRow",
    "SP_SIZE",
    "SPATIAL_SKETCH_COORDINATES",
    "SPATIAL_SKETCH_SEED",
    "TwoQuerySeedActionRows",
    "ValidatedCheckpointContentManifest",
    "ValidatedExact81DirectionGate",
    "build_native_target_suffix_layout",
    "build_two_query_seed_action_rows",
    "authenticate_live_bernini_sp4_collective",
    "canonical_json_bytes",
    "editor_noise_seed_from_owner_query_seed",
    "install_core16_fixed_a_b_only_action_lora",
    "layout_from_native_branch",
    "load_authenticated_owner_quotient_packet",
    "load_authenticated_editor_runtime_input_packet",
    "load_validated_checkpoint_content_manifest",
    "load_validated_exact81_direction_gate",
    "make_fixed_spatial_sketch",
    "object_sha256",
    "paired_functional_preservation_feature",
    "replay_score_cotangent",
    "replay_score_cotangent_to_clean_latent",
    "replay_score_cotangent_to_lora_b",
    "score_cotangent_from_authenticated_packets",
    "sketch_local_block15_target_suffix",
    "sum_detached_sp4_sketches",
    "tensor_sha256",
    "fixed_rademacher_functional_scalar",
    "try_build_two_query_seed_action_rows",
    "WEAK_I_AXIS_FUNCTIONAL_ID",
    "ZERO_ROUTE_PROOF_SCHEMA_VERSION",
]

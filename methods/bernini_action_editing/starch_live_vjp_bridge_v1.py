#!/usr/bin/env python3
"""Independent STARC live input-VJP bridge for frozen Bernini block 15.

This module proves one deliberately narrow mechanism:

    current clean latent -> one shared noisy state -> frozen Bernini
    block.15 action/no-op hidden residual -> frozen STARC critic scalar
    -> finite non-zero VJP with respect to that same current clean latent.

It is not an editor trainer and contains no optimizer path.  Public inference
still has only ``source_video`` and ``instruction``.  The no-op text embedding,
native noise and critic query are internal runtime state; no mask, detector,
track, pose or flow is accepted.

Bernini's block output is sequence-parallel.  Each SP4 rank exposes one
contiguous shard of the phase-major patch sequence.  We discard only explicit
tail padding, place the valid shard in its exact global interval, and use
``torch.distributed.nn.functional.all_reduce`` so the global hidden residual
remains in autograd.  The replicated critic scalar is divided by four before
backward, and the four replicated-input VJP contributions are summed afterward.

No result from this prototype authorizes a scientific/action-editing claim or
an editor update.  Missing authenticated candidate metadata, a hash-bound real
critic checkpoint, a real SP4 differentiable collective, or a live hook graph
is recorded as a blocker; it is never converted into ``passed=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tarfile
from typing import Any, Callable, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-starc-live-vjp-bridge-v1"
COMPOSITE_SCHEMA_VERSION = (
    "bernini-starc-current-rv2v-live-vjp-composite-binding-v2"
)
CANDIDATE_BINDING_SCHEMA = "bernini-starc-current-candidate-vjp-binding-v1"
CRITIC_CHECKPOINT_SCHEMA = "bernini-starc-core4-critic-checkpoint-v1"
CRITIC_CONFIG_SCHEMA = "bernini-starc-core4-critic-pilot-config-v1"
HOOK_COORDINATE = "block.15.output"
MODEL_ID = "transformer_1"
SP_SIZE = 4
BLOCK_COUNT = 30
BLOCK_INDEX = 15
HIDDEN_SIZE = 1536
TEXT_TOKENS = 512
TEXT_WIDTH = 4096
ROTARY_WIDTH = 64
LATENT_CHANNELS = 16
LATENT_PHASES = 21
PATCH_SIZE = (1, 2, 2)
SPATIAL_SKETCH_COORDINATES = 16
SPATIAL_SKETCH_SEED = 20260808017
SPATIAL_SKETCH_FAMILY_ID = "starc-counter-rademacher-s20260808017-v1"
SPATIAL_SKETCH_CONSTRUCTION_ID = "sha256-counter-rademacher-f32le-v1"
SPATIAL_SKETCH_VALUE_DIGEST_SCHEME = "fitq-canonical-fp32-little-endian-v1"
SPATIAL_SKETCH_CRITIC_DIGEST_SCHEME = "bernini-ltec-f32le-v1"
NATIVE_SIGMA = 0.5161304473876953
NATIVE_TIMESTEP = 516
NATIVE_SCHEDULE_INDEX = 33
LIVE_VJP_BACKEND_ID = (
    "frozen_text_conditioned_temporal_event_critic_raw_score_vjp_v1"
)
LIVE_VJP_SP4_IMPLEMENTATION = (
    "torch.distributed.nn.functional.all_reduce_autograd"
)
LIVE_VJP_BRIDGE_ARCHIVE_MEMBER = (
    "methods/bernini_action_editing/starch_live_vjp_bridge_v1.py"
)
BERNINI_OFFICIAL_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_TESTED_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
BERNINI_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
BERNINI_CHECKPOINT_CONTENT_FILE_COUNT = 23
NON_HEAD_CRITIC_STATE_KEYS = ("spatial_sketch", "nuisance_basis")
GEOMETRY_NEUTRAL_CRITIC_CONFIG = {
    "hidden_size": 1536,
    "patch_positions": 16,
    "spatial_coordinates": 16,
    "spatial_sketch_seed": SPATIAL_SKETCH_SEED,
    "projected_size": 48,
    "model_size": 96,
    "attention_heads": 4,
    "transformer_layers": 1,
    "softmin_temperature": 0.25,
    "dropout": 0.0,
    "require_nuisance_basis": False,
    "production_geometry": False,
}

EXTERNAL_INFERENCE_INPUTS = ("source_video", "instruction")
FORBIDDEN_AUXILIARY_INPUTS = (
    "mask",
    "track",
    "pose",
    "flow",
    "detector_box",
    "swept_tube",
)

# The three authenticated full644 candidate geometries currently produced by
# the AUH runtime.  Patch geometry is derived; it is not globally pinned to 930.
SUPPORTED_FULL644_LATENT_SHAPES = (
    (1, 16, 21, 60, 62),
    (1, 16, 21, 64, 58),
    (1, 16, 21, 68, 54),
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class STARCLiveVJPContractError(RuntimeError):
    """A candidate, model, hook, collective, critic or VJP failed closed."""


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
        raise STARCLiveVJPContractError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise STARCLiveVJPContractError(f"{label} must be lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    try:
        before = path.stat()
    except OSError as error:
        raise STARCLiveVJPContractError(f"cannot stat file while hashing: {path}") from error
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as error:
        raise STARCLiveVJPContractError(f"cannot hash file: {path}") from error
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
        raise STARCLiveVJPContractError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise STARCLiveVJPContractError(f"{label} must be an absolute non-root file")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise STARCLiveVJPContractError(f"{label} contains a symlink component")
    if not path.is_file() or path.resolve(strict=True) != path:
        raise STARCLiveVJPContractError(f"{label} must be a normalized plain file")
    return path


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise STARCLiveVJPContractError(
            f"{label} must be an absolute non-root directory"
        )
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise STARCLiveVJPContractError(f"{label} contains a symlink component")
    try:
        mode = path.stat().st_mode
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise STARCLiveVJPContractError(f"{label} is unavailable") from error
    if resolved != path or not stat.S_ISDIR(mode):
        raise STARCLiveVJPContractError(
            f"{label} must be a normalized plain directory"
        )
    return path


def _strict_json_file(
    value: str | Path, *, label: str, expected_sha256: str
) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(value, label=label)
    expected = _sha256(expected_sha256, label=f"expected {label} SHA-256")
    observed = _file_sha256(path)
    if observed != expected:
        raise STARCLiveVJPContractError(f"{label} file SHA-256 differs")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise STARCLiveVJPContractError(f"{label} contains duplicate key {key}")
            result[key] = item
        return result

    try:
        raw = path.read_bytes()
        decoded = json.loads(raw.decode("ascii"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise STARCLiveVJPContractError(f"{label} is not strict ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise STARCLiveVJPContractError(f"{label} root must be an object")
    return decoded, path, observed


def _validate_sealed_receipt(
    value: Mapping[str, Any], *, schema: str, label: str
) -> dict[str, Any]:
    row = dict(value)
    digest = _sha256(row.pop("receipt_digest", None), label=f"{label} receipt digest")
    if row.get("schema_version") != schema or object_sha256(row) != digest:
        raise STARCLiveVJPContractError(f"{label} schema or receipt digest differs")
    return {**row, "receipt_digest": digest}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency-light hosts
        raise STARCLiveVJPContractError("PyTorch is required for live VJP execution") from error
    return torch


@dataclass(frozen=True)
class LatentPatchGeometry:
    """Derived phase-major patch geometry for one current candidate."""

    latent_shape: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        shape = self.latent_shape
        if (
            not isinstance(shape, tuple)
            or len(shape) != 5
            or any(type(item) is not int or item <= 0 for item in shape)
            or shape[3] % PATCH_SIZE[1]
            or shape[4] % PATCH_SIZE[2]
        ):
            raise STARCLiveVJPContractError(
                "latent shape must be positive [B,C,T,H,W] with even H/W"
            )
        if shape[0] != 1:
            raise STARCLiveVJPContractError("live VJP supports batch size one only")

    @property
    def channels(self) -> int:
        return self.latent_shape[1]

    @property
    def phases(self) -> int:
        return self.latent_shape[2]

    @property
    def patch_rows(self) -> int:
        return self.latent_shape[3] // PATCH_SIZE[1]

    @property
    def patch_columns(self) -> int:
        return self.latent_shape[4] // PATCH_SIZE[2]

    @property
    def patch_positions(self) -> int:
        return self.patch_rows * self.patch_columns

    @property
    def global_tokens(self) -> int:
        return self.phases * self.patch_positions

    @property
    def local_tokens(self) -> int:
        return math.ceil(self.global_tokens / SP_SIZE)

    @property
    def padded_global_tokens(self) -> int:
        return self.local_tokens * SP_SIZE

    @property
    def padding_tokens(self) -> int:
        return self.padded_global_tokens - self.global_tokens

    @property
    def patch_dimension(self) -> int:
        return self.channels * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]

    @property
    def is_supported_full644(self) -> bool:
        return self.latent_shape in SUPPORTED_FULL644_LATENT_SHAPES

    def token_coordinate(self, global_index: int) -> tuple[int, int, int]:
        if type(global_index) is not int or not 0 <= global_index < self.global_tokens:
            raise STARCLiveVJPContractError("global token index lies outside candidate")
        phase, spatial = divmod(global_index, self.patch_positions)
        row, column = divmod(spatial, self.patch_columns)
        return phase, row, column

    def receipt(self) -> dict[str, Any]:
        return {
            "latent_shape": list(self.latent_shape),
            "patch_size": list(PATCH_SIZE),
            "patch_grid": [self.patch_rows, self.patch_columns],
            "patch_positions": self.patch_positions,
            "global_tokens": self.global_tokens,
            "sp_size": SP_SIZE,
            "local_tokens_ceil": self.local_tokens,
            "padded_global_tokens": self.padded_global_tokens,
            "tail_padding_tokens": self.padding_tokens,
            "global_order": "phase_major_then_patch_row_major",
            "supported_full644_geometry": self.is_supported_full644,
        }


@dataclass(frozen=True)
class SP4ContiguousShard:
    rank: int
    global_start: int
    global_valid_stop: int
    padded_stop: int
    local_tokens: int
    valid_tokens: int
    padding_tokens: int

    def receipt(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "global_start": self.global_start,
            "global_valid_stop": self.global_valid_stop,
            "padded_stop": self.padded_stop,
            "local_tokens": self.local_tokens,
            "valid_tokens": self.valid_tokens,
            "padding_tokens_excluded": self.padding_tokens,
        }


def make_sp4_contiguous_shard(
    geometry: LatentPatchGeometry, rank: int
) -> SP4ContiguousShard:
    if not isinstance(geometry, LatentPatchGeometry):
        raise STARCLiveVJPContractError("geometry type differs")
    if type(rank) is not int or not 0 <= rank < SP_SIZE:
        raise STARCLiveVJPContractError("SP rank must be one of 0,1,2,3")
    start = rank * geometry.local_tokens
    padded_stop = start + geometry.local_tokens
    valid_stop = min(padded_stop, geometry.global_tokens)
    valid = max(valid_stop - start, 0)
    return SP4ContiguousShard(
        rank=rank,
        global_start=start,
        global_valid_stop=valid_stop,
        padded_stop=padded_stop,
        local_tokens=geometry.local_tokens,
        valid_tokens=valid,
        padding_tokens=geometry.local_tokens - valid,
    )


def fixed_spatial_sketch_digest(
    patch_positions: int,
    *,
    coordinates: int = SPATIAL_SKETCH_COORDINATES,
    seed: int = SPATIAL_SKETCH_SEED,
) -> str:
    """Digest the exact float32 sketch family without importing Torch."""

    if (
        type(patch_positions) is not int
        or patch_positions < 2
        or type(coordinates) is not int
        or not 2 <= coordinates <= patch_positions
        or type(seed) is not int
        or seed < 0
    ):
        raise STARCLiveVJPContractError("spatial sketch dimensions/seed differ")
    scale = 1.0 / math.sqrt(float(patch_positions))
    raw = bytearray()
    for row in range(coordinates):
        for column in range(patch_positions):
            token = f"{seed}:{row}:{column}".encode("ascii")
            value = scale if hashlib.sha256(token).digest()[0] & 1 else -scale
            raw.extend(struct.pack("<f", value))
    header = (
        "bernini-ltec-f32le-v1|shape="
        f"{coordinates},{patch_positions}|"
    ).encode("ascii")
    return hashlib.sha256(header + raw).hexdigest()


def geometry_spatial_sketch_binding(
    geometry: LatentPatchGeometry,
) -> dict[str, Any]:
    """Reconstruct the runner's exact geometry-specific sketch receipt."""

    if not isinstance(geometry, LatentPatchGeometry) or not geometry.is_supported_full644:
        raise STARCLiveVJPContractError(
            "composite sketch binding requires one registered full644 geometry"
        )
    positions = geometry.patch_positions
    scale = struct.unpack("<f", struct.pack("<f", 1.0 / math.sqrt(positions)))[0]
    raw = bytearray()
    for row in range(SPATIAL_SKETCH_COORDINATES):
        for column in range(positions):
            token = f"{SPATIAL_SKETCH_SEED}:{row}:{column}".encode("ascii")
            sign = 1.0 if hashlib.sha256(token).digest()[0] & 1 else -1.0
            raw.extend(struct.pack("<f", sign * scale))
    owned = bytes(raw)
    critic_digest = hashlib.sha256(
        (
            f"{SPATIAL_SKETCH_CRITIC_DIGEST_SCHEME}|"
            f"shape={SPATIAL_SKETCH_COORDINATES},{positions}|"
        ).encode("ascii")
        + owned
    ).hexdigest()
    if critic_digest != fixed_spatial_sketch_digest(positions):
        raise STARCLiveVJPContractError("dynamic spatial sketch digest implementation drift")
    return {
        "sketch_family_id": SPATIAL_SKETCH_FAMILY_ID,
        "sketch_id": (
            f"starc-patch{geometry.patch_rows}x{geometry.patch_columns}-"
            f"counter-rademacher-s{SPATIAL_SKETCH_SEED}-v1"
        ),
        "construction_id": SPATIAL_SKETCH_CONSTRUCTION_ID,
        "seed": SPATIAL_SKETCH_SEED,
        "patch_positions": positions,
        "matrix_shape": [SPATIAL_SKETCH_COORDINATES, positions],
        "patch_grid_height_width": [geometry.patch_rows, geometry.patch_columns],
        "flatten_order": "patch-y-x",
        "normalization": f"per-row-rademacher-1-over-sqrt-{positions}",
        "matrix_dtype": "torch.float32",
        "matrix_raw_bytes_sha256": hashlib.sha256(owned).hexdigest(),
        "matrix_value_digest_scheme": SPATIAL_SKETCH_VALUE_DIGEST_SCHEME,
        "matrix_value_sha256": hashlib.sha256(
            (
                f"{SPATIAL_SKETCH_VALUE_DIGEST_SCHEME}|"
                f"shape={SPATIAL_SKETCH_COORDINATES},{positions}|"
            ).encode("ascii")
            + owned
        ).hexdigest(),
        "critic_tensor_digest_scheme": SPATIAL_SKETCH_CRITIC_DIGEST_SCHEME,
        "critic_tensor_sha256": critic_digest,
        "full_support_no_mask_or_localizer": True,
        "data_dependent": False,
    }


def make_fixed_spatial_sketch(
    patch_positions: int,
    *,
    coordinates: int = SPATIAL_SKETCH_COORDINATES,
    seed: int = SPATIAL_SKETCH_SEED,
    device: Any = None,
) -> Any:
    """Build the content-independent KxP sketch for the actual candidate P."""

    torch = _require_torch()
    expected_digest = fixed_spatial_sketch_digest(
        patch_positions, coordinates=coordinates, seed=seed
    )
    scale = 1.0 / math.sqrt(float(patch_positions))
    matrix = torch.empty(coordinates, patch_positions, dtype=torch.float32)
    for row in range(coordinates):
        for column in range(patch_positions):
            token = f"{seed}:{row}:{column}".encode("ascii")
            matrix[row, column] = scale if hashlib.sha256(token).digest()[0] & 1 else -scale
    observed = _tensor_f32le_digest(matrix, label="fixed spatial sketch")
    if observed != expected_digest:
        raise STARCLiveVJPContractError("fixed spatial sketch byte digest differs")
    if int(torch.linalg.matrix_rank(matrix).item()) != coordinates:
        raise STARCLiveVJPContractError("fixed spatial sketch is rank deficient")
    return matrix.to(device=device) if device is not None else matrix


def _tensor_f32le_digest(value: Any, *, label: str) -> str:
    torch = _require_torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise STARCLiveVJPContractError(f"{label} must be detached finite FP32")
    owned = value.detach().cpu().contiguous().clone()
    header = (
        "bernini-ltec-f32le-v1|shape="
        f"{','.join(str(int(item)) for item in owned.shape)}|"
    ).encode("ascii")
    return hashlib.sha256(header + bytes(owned.untyped_storage())).hexdigest()


def _tensor_value_digest(value: Any, *, label: str) -> str:
    torch = _require_torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or not bool(torch.isfinite(value).all().item())
    ):
        raise STARCLiveVJPContractError(f"{label} must be finite tensor")
    owned = value.detach().cpu().contiguous().clone()
    metadata = {"dtype": str(owned.dtype), "shape": list(map(int, owned.shape))}
    return hashlib.sha256(
        canonical_json_bytes(metadata) + b"|" + bytes(owned.untyped_storage())
    ).hexdigest()


@dataclass(frozen=True)
class CurrentCandidateBinding:
    candidate_id: str
    geometry: LatentPatchGeometry
    source_video_sha256: Optional[str]
    instruction_sha256: str
    clean_latent_tensor_sha256: Optional[str]
    manifest_path: Optional[str]
    manifest_file_sha256: Optional[str]
    manifest_receipt_digest: Optional[str]
    authenticated: bool

    def blockers(self) -> tuple[str, ...]:
        result = []
        if not self.authenticated:
            result.append("authenticated_current_candidate_manifest_missing")
        if not self.geometry.is_supported_full644:
            result.append("current_candidate_is_not_supported_full644_geometry")
        return tuple(result)


def authenticate_current_candidate_manifest(
    path_value: str | Path,
    *,
    expected_manifest_sha256: str,
    instruction: str,
) -> CurrentCandidateBinding:
    """Authenticate geometry and public source/instruction identity."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise STARCLiveVJPContractError("instruction must be nonempty text")
    raw, path, observed = _strict_json_file(
        path_value,
        label="current candidate manifest",
        expected_sha256=expected_manifest_sha256,
    )
    manifest = _validate_sealed_receipt(
        raw, schema=CANDIDATE_BINDING_SCHEMA, label="current candidate manifest"
    )
    required = {
        "schema_version",
        "candidate_id",
        "source_video_sha256",
        "instruction_sha256",
        "current_clean_latent_tensor_sha256",
        "latent_shape",
        "patch_order",
        "external_inference_inputs",
        "auxiliary_spatial_inputs",
        "receipt_digest",
    }
    if set(manifest) != required:
        raise STARCLiveVJPContractError("current candidate manifest field closure differs")
    candidate_id = manifest["candidate_id"]
    if not isinstance(candidate_id, str) or _SAFE_ID_RE.fullmatch(candidate_id) is None:
        raise STARCLiveVJPContractError("candidate ID is not path-safe")
    source_digest = _sha256(
        manifest["source_video_sha256"], label="source video SHA-256"
    )
    instruction_digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    if _sha256(
        manifest["instruction_sha256"], label="instruction SHA-256"
    ) != instruction_digest:
        raise STARCLiveVJPContractError("instruction does not match candidate manifest")
    clean_digest = _sha256(
        manifest["current_clean_latent_tensor_sha256"],
        label="current clean latent tensor SHA-256",
    )
    shape_value = manifest["latent_shape"]
    if not isinstance(shape_value, list) or any(type(item) is not int for item in shape_value):
        raise STARCLiveVJPContractError("candidate latent shape manifest differs")
    geometry = LatentPatchGeometry(tuple(shape_value))
    if not geometry.is_supported_full644:
        raise STARCLiveVJPContractError("authenticated candidate geometry is not full644")
    if (
        manifest["patch_order"] != "phase_major_then_patch_row_major"
        or manifest["external_inference_inputs"] != list(EXTERNAL_INFERENCE_INPUTS)
        or manifest["auxiliary_spatial_inputs"] != []
    ):
        raise STARCLiveVJPContractError("candidate public-input or patch-order contract differs")
    return CurrentCandidateBinding(
        candidate_id=candidate_id,
        geometry=geometry,
        source_video_sha256=source_digest,
        instruction_sha256=instruction_digest,
        clean_latent_tensor_sha256=clean_digest,
        manifest_path=str(path),
        manifest_file_sha256=observed,
        manifest_receipt_digest=manifest["receipt_digest"],
        authenticated=True,
    )


def mechanism_only_candidate_binding(
    latent_shape: Sequence[int], *, instruction: str, candidate_id: str = "toy"
) -> CurrentCandidateBinding:
    """Explicitly unauthenticated binding for unit mechanism tests only."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise STARCLiveVJPContractError("instruction must be nonempty text")
    if not isinstance(candidate_id, str) or _SAFE_ID_RE.fullmatch(candidate_id) is None:
        raise STARCLiveVJPContractError("candidate ID is not path-safe")
    return CurrentCandidateBinding(
        candidate_id=candidate_id,
        geometry=LatentPatchGeometry(tuple(latent_shape)),
        source_video_sha256=None,
        instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        clean_latent_tensor_sha256=None,
        manifest_path=None,
        manifest_file_sha256=None,
        manifest_receipt_digest=None,
        authenticated=False,
    )


@dataclass(frozen=True)
class BridgeSourceArchiveBinding:
    source_path: str
    source_file_sha256: str
    source_archive_path: str
    source_archive_file_sha256: str
    source_archive_bridge_member_path: str
    source_archive_bridge_member_sha256: str
    source_git_revision: str

    def receipt(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_file_sha256": self.source_file_sha256,
            "source_archive_path": self.source_archive_path,
            "source_archive_file_sha256": self.source_archive_file_sha256,
            "source_archive_bridge_member_path": (
                self.source_archive_bridge_member_path
            ),
            "source_archive_bridge_member_sha256": (
                self.source_archive_bridge_member_sha256
            ),
            "source_git_revision": self.source_git_revision,
        }


def authenticate_bridge_source_archive(
    source_archive: str | Path,
    *,
    expected_source_archive_sha256: str,
    source_git_revision: str,
) -> BridgeSourceArchiveBinding:
    """Bind the executing bridge bytes to one revision-bearing git archive."""

    revision = source_git_revision
    if not isinstance(revision, str) or _SHA1_RE.fullmatch(revision) is None:
        raise STARCLiveVJPContractError("bridge source revision must be lowercase 40-hex")
    source = _plain_file(Path(__file__).resolve(), label="executing live VJP bridge")
    source_sha = _file_sha256(source)
    archive = _plain_file(source_archive, label="live VJP bridge source archive")
    archive_sha = _sha256(
        expected_source_archive_sha256,
        label="expected live VJP bridge source archive SHA-256",
    )
    if _file_sha256(archive) != archive_sha:
        raise STARCLiveVJPContractError("live VJP bridge source archive SHA-256 differs")
    matches = []
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            archive_revision = handle.pax_headers.get("comment")
            for member in handle.getmembers():
                pure = PurePosixPath(member.name)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                ):
                    raise STARCLiveVJPContractError(
                        "live VJP bridge source archive contains an unsafe member"
                    )
                if pure.as_posix() == LIVE_VJP_BRIDGE_ARCHIVE_MEMBER:
                    matches.append(member)
            if archive_revision != revision:
                raise STARCLiveVJPContractError(
                    "live VJP bridge source archive revision differs"
                )
            if len(matches) != 1 or not matches[0].isfile():
                raise STARCLiveVJPContractError(
                    "source archive lacks exactly one plain live VJP bridge member"
                )
            stream = handle.extractfile(matches[0])
            member_sha = None if stream is None else hashlib.sha256(stream.read()).hexdigest()
            if member_sha != source_sha:
                raise STARCLiveVJPContractError(
                    "executing live VJP bridge differs from archived bridge member"
                )
    except STARCLiveVJPContractError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise STARCLiveVJPContractError(
            "live VJP bridge source archive cannot be authenticated"
        ) from error
    return BridgeSourceArchiveBinding(
        source_path=str(source),
        source_file_sha256=source_sha,
        source_archive_path=str(archive),
        source_archive_file_sha256=archive_sha,
        source_archive_bridge_member_path=LIVE_VJP_BRIDGE_ARCHIVE_MEMBER,
        source_archive_bridge_member_sha256=source_sha,
        source_git_revision=revision,
    )


@dataclass(frozen=True)
class BerniniCheckpointContentBinding:
    checkpoint_root: str
    checkpoint_tree_sha256: str
    checkpoint_content_manifest_path: str
    checkpoint_content_manifest_file_sha256: str
    checkpoint_content_verified_file_count: int
    checkpoint_content_verified_entries_digest: str

    def receipt(self) -> dict[str, Any]:
        return {
            "checkpoint_root": self.checkpoint_root,
            "checkpoint_tree_sha256": self.checkpoint_tree_sha256,
            "checkpoint_content_manifest_path": (
                self.checkpoint_content_manifest_path
            ),
            "checkpoint_content_manifest_file_sha256": (
                self.checkpoint_content_manifest_file_sha256
            ),
            "checkpoint_content_verified_file_count": (
                self.checkpoint_content_verified_file_count
            ),
            "checkpoint_content_verified_entries_digest": (
                self.checkpoint_content_verified_entries_digest
            ),
        }


def authenticate_frozen_bernini_checkpoint_content(
    checkpoint_root: str | Path,
    checkpoint_content_manifest: str | Path,
    *,
    expected_checkpoint_tree_sha256: Optional[str] = None,
    expected_checkpoint_content_manifest_sha256: Optional[str] = None,
) -> BerniniCheckpointContentBinding:
    """Verify all 23 pinned non-cache Bernini checkpoint files."""

    tree_sha = _sha256(
        BERNINI_CHECKPOINT_TREE_SHA256
        if expected_checkpoint_tree_sha256 is None
        else expected_checkpoint_tree_sha256,
        label="expected Bernini checkpoint tree SHA-256",
    )
    manifest_sha = _sha256(
        BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
        if expected_checkpoint_content_manifest_sha256 is None
        else expected_checkpoint_content_manifest_sha256,
        label="expected Bernini checkpoint content manifest SHA-256",
    )
    if (
        tree_sha != BERNINI_CHECKPOINT_TREE_SHA256
        or manifest_sha != BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        raise STARCLiveVJPContractError("Bernini checkpoint identity is not pinned")
    root = _plain_directory(checkpoint_root, label="Bernini checkpoint root")
    manifest = _plain_file(
        checkpoint_content_manifest, label="Bernini checkpoint content manifest"
    )
    if _file_sha256(manifest) != manifest_sha:
        raise STARCLiveVJPContractError(
            "Bernini checkpoint content manifest SHA-256 differs"
        )
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise STARCLiveVJPContractError(
            "Bernini checkpoint content manifest cannot be read"
        ) from error
    if len(lines) != BERNINI_CHECKPOINT_CONTENT_FILE_COUNT:
        raise STARCLiveVJPContractError(
            "Bernini checkpoint content manifest file count differs"
        )
    expected: dict[str, str] = {}
    pattern = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            raise STARCLiveVJPContractError(
                "checkpoint manifest line is not canonical sha256sum syntax"
            )
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise STARCLiveVJPContractError(
                "checkpoint manifest contains an unsafe path"
            )
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in expected:
            raise STARCLiveVJPContractError(
                "checkpoint manifest contains an empty or duplicate path"
            )
        expected[normalized] = digest
    actual_paths: set[str] = set()
    try:
        descendants = tuple(root.rglob("*"))
    except OSError as error:
        raise STARCLiveVJPContractError(
            "Bernini checkpoint content cannot be enumerated"
        ) from error
    for path in descendants:
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise STARCLiveVJPContractError(
                "Bernini checkpoint content cannot be inspected"
            ) from error
        if stat.S_ISLNK(mode):
            raise STARCLiveVJPContractError(
                "Bernini checkpoint contains a non-cache symlink"
            )
        if stat.S_ISREG(mode):
            actual_paths.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise STARCLiveVJPContractError(
                "Bernini checkpoint contains a non-regular entry"
            )
    if actual_paths != set(expected):
        raise STARCLiveVJPContractError(
            "Bernini checkpoint file closure differs from content manifest"
        )
    verified_entries = []
    for relative in sorted(expected):
        path = _plain_file(root / relative, label=f"checkpoint file {relative}")
        actual = _file_sha256(path)
        if actual != expected[relative]:
            raise STARCLiveVJPContractError(
                f"Bernini checkpoint content hash differs: {relative}"
            )
        verified_entries.append({"path": relative, "sha256": actual})
    return BerniniCheckpointContentBinding(
        checkpoint_root=str(root),
        checkpoint_tree_sha256=tree_sha,
        checkpoint_content_manifest_path=str(manifest),
        checkpoint_content_manifest_file_sha256=manifest_sha,
        checkpoint_content_verified_file_count=len(verified_entries),
        checkpoint_content_verified_entries_digest=object_sha256(verified_entries),
    )


def _tensor_state_digest(value: Any, *, label: str) -> str:
    torch = _require_torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise STARCLiveVJPContractError(f"{label} must be detached finite tensor")
    owned = value.detach().cpu().contiguous().clone()
    header = canonical_json_bytes(
        {"dtype": str(owned.dtype), "shape": list(map(int, owned.shape))}
    )
    return hashlib.sha256(header + b"|" + bytes(owned.untyped_storage())).hexdigest()


def _critic_state_content_digest(
    critic: Any, *, excluded_keys: Sequence[str] = ()
) -> tuple[str, int]:
    excluded = tuple(excluded_keys)
    if len(set(excluded)) != len(excluded) or any(
        not isinstance(name, str) or not name for name in excluded
    ):
        raise STARCLiveVJPContractError("critic excluded state key closure differs")
    state = {
        name: tensor
        for name, tensor in critic.state_dict().items()
        if name not in excluded
    }
    if not isinstance(state, Mapping) or not state:
        raise STARCLiveVJPContractError("critic state must be a nonempty mapping")
    rows = []
    for name in sorted(state):
        if not isinstance(name, str) or not name:
            raise STARCLiveVJPContractError("critic state name differs")
        tensor = state[name]
        rows.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(map(int, tensor.shape)),
                "tensor_digest": _tensor_state_digest(
                    tensor.detach(), label=f"critic state tensor {name}"
                ),
            }
        )
    return object_sha256(rows), len(rows)


@dataclass(frozen=True)
class VerifiedCriticArtifact:
    checkpoint_path: str
    checkpoint_file_sha256: str
    manifest_path: str
    manifest_file_sha256: str
    manifest_receipt_digest: str
    config_manifest_path: str
    config_manifest_file_sha256: str
    config_manifest_receipt_digest: str
    checkpoint_state_content_digest: str
    checkpoint_tensor_count: int
    excluded_state_keys: tuple[str, ...]
    runtime_class: str
    verified: bool


def verify_frozen_starc_critic_artifact(
    critic: Any,
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    config_manifest_path: str | Path,
    expected_config_manifest_sha256: str,
) -> VerifiedCriticArtifact:
    """Bind a frozen runtime critic to the pilot checkpoint and receipt."""

    torch = _require_torch()
    if not isinstance(critic, torch.nn.Module):
        raise STARCLiveVJPContractError("critic must be a torch module")
    runtime_class = f"{type(critic).__module__}.{type(critic).__name__}"
    if (
        type(critic).__name__ != "FrozenHiddenTemporalEventCritic"
        or not type(critic).__module__.endswith("latent_temporal_event_critic")
    ):
        raise STARCLiveVJPContractError("runtime is not the STARC critic class")
    if critic.training or any(parameter.requires_grad for parameter in critic.parameters()):
        raise STARCLiveVJPContractError("runtime critic is not eval-mode frozen")
    config = getattr(critic, "config", None)
    runtime_config = {
        name: getattr(config, name, None) for name in GEOMETRY_NEUTRAL_CRITIC_CONFIG
    }
    if config is None or runtime_config != GEOMETRY_NEUTRAL_CRITIC_CONFIG:
        raise STARCLiveVJPContractError("runtime critic geometry-neutral config differs")
    spatial_sentinel = getattr(critic, "spatial_sketch", None)
    nuisance_sentinel = getattr(critic, "nuisance_basis", None)
    if (
        not isinstance(spatial_sentinel, torch.Tensor)
        or spatial_sentinel.dtype != torch.float32
        or tuple(spatial_sentinel.shape) != (16, 16)
        or spatial_sentinel.requires_grad
        or not torch.equal(
            spatial_sentinel.detach().cpu(), torch.eye(16, dtype=torch.float32)
        )
        or not isinstance(nuisance_sentinel, torch.Tensor)
        or nuisance_sentinel.dtype != torch.float32
        or tuple(nuisance_sentinel.shape) != (HIDDEN_SIZE, 0)
        or nuisance_sentinel.requires_grad
    ):
        raise STARCLiveVJPContractError("geometry-neutral critic constructor sentinels differ")

    checkpoint = _plain_file(checkpoint_path, label="critic checkpoint")
    if checkpoint.suffix != ".safetensors":
        raise STARCLiveVJPContractError("critic checkpoint must be safetensors")
    expected_checkpoint = _sha256(
        expected_checkpoint_sha256, label="expected critic checkpoint SHA-256"
    )
    observed_checkpoint = _file_sha256(checkpoint)
    if observed_checkpoint != expected_checkpoint:
        raise STARCLiveVJPContractError("critic checkpoint file SHA-256 differs")
    raw, receipt_path, observed_manifest = _strict_json_file(
        manifest_path,
        label="critic checkpoint manifest",
        expected_sha256=expected_manifest_sha256,
    )
    manifest = _validate_sealed_receipt(
        raw, schema=CRITIC_CHECKPOINT_SCHEMA, label="critic checkpoint manifest"
    )
    config_raw, config_path, observed_config_manifest = _strict_json_file(
        config_manifest_path,
        label="critic config manifest",
        expected_sha256=expected_config_manifest_sha256,
    )
    config_manifest = _validate_sealed_receipt(
        config_raw, schema=CRITIC_CONFIG_SCHEMA, label="critic config manifest"
    )
    expected_config_digest = object_sha256(GEOMETRY_NEUTRAL_CRITIC_CONFIG)
    head_contract = config_manifest.get("pre_sketched_head_contract")
    if (
        config_manifest.get("critic_config") != GEOMETRY_NEUTRAL_CRITIC_CONFIG
        or config_manifest.get("critic_config_content_digest")
        != expected_config_digest
        or not isinstance(head_contract, Mapping)
        or head_contract.get("entrypoint") != "forward_sketched_residual_only"
        or head_contract.get("geometry_neutral_after_fixed_sketch") is not True
        or head_contract.get("constructor_spatial_buffer")
        != "inert_16x16_identity_never_consumed"
        or head_contract.get("constructor_spatial_buffer_checkpointed") is not False
        or head_contract.get("full_hidden_forward_authorized") is not False
        or head_contract.get("geometry_specific_sketches_authenticated_by_materializer")
        is not True
        or config_manifest.get("nuisance_basis_used") is not False
        or config_manifest.get("core4_scientific_claim_authorized") is not False
        or config_manifest.get("editor_optimizer_present_or_authorized") is not False
    ):
        raise STARCLiveVJPContractError("critic geometry-neutral config manifest differs")
    required_true = (
        "state_tensor_byte_parity_after_fresh_load",
        "fit_score_parity_after_fresh_load",
        "critic_frozen_after_reload",
        "only_final_checkpoint_saved",
    )
    if any(manifest.get(name) is not True for name in required_true):
        raise STARCLiveVJPContractError("critic checkpoint verification evidence differs")
    if (
        manifest.get("editor_checkpoint_or_parameter_present") is not False
        or manifest.get("editor_optimizer_authorized") is not False
        or manifest.get("checkpoint_path") != str(checkpoint)
        or manifest.get("checkpoint_file_sha256") != observed_checkpoint
        or manifest.get("checkpoint_scope")
        != "geometry_neutral_pre_sketched_critic_head_only"
        or manifest.get("excluded_constructor_buffer_keys")
        != list(NON_HEAD_CRITIC_STATE_KEYS)
        or manifest.get("config_receipt_digest")
        != config_manifest["receipt_digest"]
        or manifest.get("optimizer_step") != 200
        or manifest.get("best_checkpoint_saved") is not False
        or manifest.get("confirmation_sample_seen_before_checkpoint_save") is not False
        or type(manifest.get("checkpoint_tensor_count")) is not int
        or manifest.get("checkpoint_tensor_count") <= 0
    ):
        raise STARCLiveVJPContractError("critic checkpoint manifest binding differs")
    state_digest, tensor_count = _critic_state_content_digest(
        critic, excluded_keys=NON_HEAD_CRITIC_STATE_KEYS
    )
    if (
        _sha256(
            manifest.get("checkpoint_state_content_digest"),
            label="critic checkpoint state digest",
        )
        != state_digest
        or manifest.get("checkpoint_tensor_count") != tensor_count
    ):
        raise STARCLiveVJPContractError("runtime critic state differs from checkpoint manifest")
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as error:
        raise STARCLiveVJPContractError(
            "safetensors is required to authenticate the critic checkpoint"
        ) from error
    loaded_state = load_file(str(checkpoint), device="cpu")
    runtime_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in critic.state_dict().items()
        if name not in NON_HEAD_CRITIC_STATE_KEYS
    }
    if set(loaded_state) != set(runtime_state):
        raise STARCLiveVJPContractError("checkpoint/runtime critic state key closure differs")
    for name in runtime_state:
        if (
            loaded_state[name].dtype != runtime_state[name].dtype
            or loaded_state[name].shape != runtime_state[name].shape
            or not torch.equal(loaded_state[name], runtime_state[name])
        ):
            raise STARCLiveVJPContractError(
                f"checkpoint/runtime critic tensor {name} differs"
            )
    with safe_open(str(checkpoint), framework="pt", device="cpu") as opened:
        metadata = opened.metadata()
    expected_metadata = {
        "schema_version": CRITIC_CHECKPOINT_SCHEMA,
        "config_receipt_digest": manifest.get("config_receipt_digest"),
        "checkpoint_state_content_digest": state_digest,
        "optimizer_step": str(manifest.get("optimizer_step")),
        "selection": "final_step_200_only",
    }
    if metadata != expected_metadata:
        raise STARCLiveVJPContractError("critic safetensors metadata differs")
    return VerifiedCriticArtifact(
        checkpoint_path=str(checkpoint),
        checkpoint_file_sha256=observed_checkpoint,
        manifest_path=str(receipt_path),
        manifest_file_sha256=observed_manifest,
        manifest_receipt_digest=manifest["receipt_digest"],
        config_manifest_path=str(config_path),
        config_manifest_file_sha256=observed_config_manifest,
        config_manifest_receipt_digest=config_manifest["receipt_digest"],
        checkpoint_state_content_digest=state_digest,
        checkpoint_tensor_count=tensor_count,
        excluded_state_keys=NON_HEAD_CRITIC_STATE_KEYS,
        runtime_class=runtime_class,
        verified=True,
    )


def scatter_local_shard_to_global(
    local_hidden: Any,
    *,
    geometry: LatentPatchGeometry,
    shard: SP4ContiguousShard,
) -> Any:
    """Place one valid contiguous shard into a differentiable global buffer."""

    torch = _require_torch()
    if (
        not isinstance(local_hidden, torch.Tensor)
        or local_hidden.ndim != 3
        or int(local_hidden.shape[0]) != 1
        or int(local_hidden.shape[1]) != shard.local_tokens
        or not local_hidden.is_floating_point()
        or not bool(torch.isfinite(local_hidden).all().item())
    ):
        raise STARCLiveVJPContractError("local block hidden geometry/value differs")
    valid = local_hidden[:, : shard.valid_tokens, :]
    # F.pad is differentiable and gives explicit zeros outside this rank's
    # contiguous interval; rank-3 padding is never allowed into the critic.
    return torch.nn.functional.pad(
        valid,
        (
            0,
            0,
            shard.global_start,
            geometry.global_tokens - shard.global_valid_stop,
        ),
    )


def assemble_sp4_shards_for_test(
    local_shards: Sequence[Any], *, geometry: LatentPatchGeometry
) -> Any:
    """Single-process differentiable oracle used only for mapping/parity tests."""

    torch = _require_torch()
    if not isinstance(local_shards, Sequence) or len(local_shards) != SP_SIZE:
        raise STARCLiveVJPContractError("SP4 test oracle requires four ordered shards")
    valid = []
    batch = None
    width = None
    for rank, tensor in enumerate(local_shards):
        shard = make_sp4_contiguous_shard(geometry, rank)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 3
            or int(tensor.shape[1]) != shard.local_tokens
        ):
            raise STARCLiveVJPContractError("SP4 test shard geometry differs")
        if batch is None:
            batch, width = int(tensor.shape[0]), int(tensor.shape[2])
        elif (int(tensor.shape[0]), int(tensor.shape[2])) != (batch, width):
            raise STARCLiveVJPContractError("SP4 test shard batch/width differs")
        valid.append(tensor[:, : shard.valid_tokens, :])
    result = torch.cat(valid, dim=1)
    if int(result.shape[1]) != geometry.global_tokens:
        raise STARCLiveVJPContractError("SP4 test oracle global coverage differs")
    return result


def sketch_rank_local_hidden_exact(
    local_hidden: Any,
    *,
    geometry: LatentPatchGeometry,
    shard: SP4ContiguousShard,
    spatial_sketch: Any,
) -> Any:
    """Match the materializer's FP32 local index-add sketch, with autograd.

    Branch hidden is cast to FP32 *before* multiplication/accumulation.  Action
    and no-op call this separately; only their globally reduced FP32 sketches
    may be subtracted.  That operation order is part of the critic's training
    distribution, not an algebraic implementation detail.
    """

    torch = _require_torch()
    if (
        not isinstance(local_hidden, torch.Tensor)
        or tuple(local_hidden.shape[:2]) != (1, shard.local_tokens)
        or local_hidden.ndim != 3
        or not local_hidden.is_floating_point()
        or not local_hidden.requires_grad
        or local_hidden.grad_fn is None
        or not bool(torch.isfinite(local_hidden).all().item())
        or not isinstance(spatial_sketch, torch.Tensor)
        or spatial_sketch.dtype != torch.float32
        or tuple(spatial_sketch.shape[1:]) != (geometry.patch_positions,)
        or spatial_sketch.device != local_hidden.device
        or spatial_sketch.requires_grad
        or spatial_sketch.grad_fn is not None
    ):
        raise STARCLiveVJPContractError("rank-local hidden/sketch contract differs")
    local_indices = torch.arange(
        shard.valid_tokens, dtype=torch.long, device=local_hidden.device
    )
    global_indices = local_indices + shard.global_start
    phases = torch.div(
        global_indices, geometry.patch_positions, rounding_mode="floor"
    )
    patches = torch.remainder(global_indices, geometry.patch_positions)
    values = local_hidden[0].index_select(0, local_indices).float()
    rows = []
    for coordinate in range(int(spatial_sketch.shape[0])):
        weights = spatial_sketch[coordinate].index_select(0, patches).unsqueeze(1)
        row = torch.zeros(
            geometry.phases,
            int(local_hidden.shape[2]),
            dtype=torch.float32,
            device=local_hidden.device,
        ).index_add(0, phases, values * weights)
        rows.append(row)
    result = torch.stack(rows, dim=1).unsqueeze(0).contiguous()
    if (
        tuple(result.shape)
        != (
            1,
            geometry.phases,
            int(spatial_sketch.shape[0]),
            int(local_hidden.shape[2]),
        )
        or result.dtype != torch.float32
        or not result.requires_grad
        or result.grad_fn is None
        or not bool(torch.isfinite(result).all().item())
    ):
        raise STARCLiveVJPContractError("rank-local differentiable FP32 sketch differs")
    return result


@dataclass(frozen=True)
class GlobalSketchAssembly:
    tensor: Any
    backend: str
    real_sp4_autograd_collective: bool
    rank: int
    role: str


@dataclass(frozen=True)
class AllRankHiddenBackwardEvidence:
    ordered_rank_evidence: tuple[Mapping[str, Any], ...]
    evidence_digest: str

    @property
    def rank_gradient_tensor_digests(self) -> tuple[str, ...]:
        return tuple(str(row["action_digest"]) for row in self.ordered_rank_evidence)

    def receipt_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.ordered_rank_evidence]


class TorchDistributedNNFunctionalSP4:
    """The only production collective backend accepted by this prototype."""

    backend = LIVE_VJP_SP4_IMPLEMENTATION
    real_sp4_autograd_collective = True

    def __init__(self, *, group: Any = None) -> None:
        self.group = group

    def _runtime(self, expected_rank: int) -> tuple[Any, Any, int]:
        torch = _require_torch()
        import torch.distributed as distributed
        from torch.distributed.nn import functional as distributed_nn_functional

        if not distributed.is_available() or not distributed.is_initialized():
            raise STARCLiveVJPContractError("real differentiable SP4 requires initialized dist")
        world_size = distributed.get_world_size(self.group)
        rank = distributed.get_rank(self.group)
        if world_size != SP_SIZE or rank != expected_rank:
            raise STARCLiveVJPContractError("real differentiable collective topology differs")
        return torch, (distributed, distributed_nn_functional), rank

    def globalize_sketch(
        self,
        local_sketch: Any,
        *,
        geometry: LatentPatchGeometry,
        shard: SP4ContiguousShard,
        role: str,
    ) -> GlobalSketchAssembly:
        torch, modules, rank = self._runtime(shard.rank)
        distributed, distributed_nn_functional = modules
        if (
            role not in ("action", "noop")
            or not isinstance(local_sketch, torch.Tensor)
            or tuple(local_sketch.shape[:2]) != (1, geometry.phases)
            or local_sketch.ndim != 4
            or local_sketch.dtype != torch.float32
            or not local_sketch.requires_grad
            or local_sketch.grad_fn is None
        ):
            raise STARCLiveVJPContractError("rank-local live STARC sketch differs")
        global_sketch = distributed_nn_functional.all_reduce(
            local_sketch, op=distributed.ReduceOp.SUM, group=self.group
        )
        if (
            not isinstance(global_sketch, torch.Tensor)
            or global_sketch.shape != local_sketch.shape
            or global_sketch.dtype != torch.float32
            or not global_sketch.requires_grad
            or global_sketch.grad_fn is None
            or not bool(torch.isfinite(global_sketch).all().item())
        ):
            raise STARCLiveVJPContractError("differentiable global STARC sketch failed")
        return GlobalSketchAssembly(
            tensor=global_sketch,
            backend=self.backend,
            real_sp4_autograd_collective=True,
            rank=rank,
            role=role,
        )

    def assert_replica_consensus(self, digest: str, *, rank: int) -> bool:
        """Agree on graph inputs before any rank enters a model collective."""

        _torch, modules, observed_rank = self._runtime(rank)
        distributed, _distributed_nn_functional = modules
        value = _sha256(digest, label="SP4 preflight contract digest")
        rows: list[Any] = [None] * SP_SIZE
        distributed.all_gather_object(
            rows, {"rank": observed_rank, "digest": value}, group=self.group
        )
        if (
            [row.get("rank") if isinstance(row, Mapping) else None for row in rows]
            != list(range(SP_SIZE))
            or any(
                not isinstance(row, Mapping) or row.get("digest") != value
                for row in rows
            )
        ):
            raise STARCLiveVJPContractError("SP4 preflight graph-input consensus differs")
        return True

    def assert_replicated_score_consensus(self, score: Any, *, rank: int) -> str:
        """Require the frozen critic scalar to be byte-identical on all ranks."""

        torch, modules, observed_rank = self._runtime(rank)
        distributed, _distributed_nn_functional = modules
        if (
            not isinstance(score, torch.Tensor)
            or score.numel() != 1
            or not bool(torch.isfinite(score).all().item())
        ):
            raise STARCLiveVJPContractError("replicated critic score differs")
        digest = _tensor_value_digest(score, label="replicated critic score")
        rows: list[Any] = [None] * SP_SIZE
        distributed.all_gather_object(
            rows, {"rank": observed_rank, "digest": digest}, group=self.group
        )
        if (
            [row.get("rank") if isinstance(row, Mapping) else None for row in rows]
            != list(range(SP_SIZE))
            or any(
                not isinstance(row, Mapping) or row.get("digest") != digest
                for row in rows
            )
        ):
            raise STARCLiveVJPContractError("replicated critic score consensus differs")
        return digest

    def assert_all_rank_hidden_backward(
        self, action_gradient: Any, noop_gradient: Any, *, rank: int
    ) -> AllRankHiddenBackwardEvidence:
        """Prove every local action/no-op FP32 sketch received critic cotangent."""

        torch, modules, observed_rank = self._runtime(rank)
        distributed, _distributed_nn_functional = modules
        if (
            not isinstance(action_gradient, torch.Tensor)
            or not isinstance(noop_gradient, torch.Tensor)
            or action_gradient.shape != noop_gradient.shape
            or action_gradient.dtype != torch.float32
            or noop_gradient.dtype != torch.float32
            or not bool(torch.isfinite(action_gradient).all().item())
            or not bool(torch.isfinite(noop_gradient).all().item())
            or not torch.equal(action_gradient, -noop_gradient)
        ):
            raise STARCLiveVJPContractError("local hidden action/no-op cotangent differs")
        norm = torch.linalg.vector_norm(action_gradient)
        if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
            raise STARCLiveVJPContractError("local hidden critic cotangent is zero")
        local = {
            "rank": observed_rank,
            "shape": list(map(int, action_gradient.shape)),
            "action_digest": _tensor_value_digest(
                action_gradient, label="action hidden cotangent"
            ),
            "noop_digest": _tensor_value_digest(
                noop_gradient, label="no-op hidden cotangent"
            ),
            "norm": float(norm.item()),
            "finite_nonzero": True,
            "action_is_exact_negative_noop": True,
        }
        rows: list[Any] = [None] * SP_SIZE
        distributed.all_gather_object(rows, local, group=self.group)
        if (
            [row.get("rank") if isinstance(row, Mapping) else None for row in rows]
            != list(range(SP_SIZE))
            or any(
                not isinstance(row, Mapping)
                or row.get("shape") != local["shape"]
                or row.get("finite_nonzero") is not True
                or row.get("action_is_exact_negative_noop") is not True
                for row in rows
            )
        ):
            raise STARCLiveVJPContractError("all-rank hidden backward evidence differs")
        sealed = {
            "schema_version": "bernini-starc-all-rank-hidden-vjp-v2",
            "ordered_rank_evidence": rows,
        }
        return AllRankHiddenBackwardEvidence(
            ordered_rank_evidence=tuple(dict(row) for row in rows),
            evidence_digest=object_sha256(sealed),
        )

    def reduce_replicated_input_vjp(self, local_vjp: Any, *, rank: int) -> Any:
        torch, modules, observed_rank = self._runtime(rank)
        distributed, distributed_nn_functional = modules
        reduced = distributed_nn_functional.all_reduce(
            local_vjp, op=distributed.ReduceOp.SUM, group=self.group
        )
        if (
            observed_rank != rank
            or not isinstance(reduced, torch.Tensor)
            or reduced.shape != local_vjp.shape
            or not bool(torch.isfinite(reduced).all().item())
        ):
            raise STARCLiveVJPContractError("replicated clean-latent VJP reduction failed")
        return reduced


@dataclass(frozen=True)
class BerniniRuntimeDimensions:
    block_count: int = BLOCK_COUNT
    block_index: int = BLOCK_INDEX
    hidden_size: int = HIDDEN_SIZE
    text_tokens: int = TEXT_TOKENS
    text_width: int = TEXT_WIDTH
    rotary_width: int = ROTARY_WIDTH
    spatial_sketch_coordinates: int = SPATIAL_SKETCH_COORDINATES

    @property
    def is_production(self) -> bool:
        return self == BerniniRuntimeDimensions()


class _Block15LivePairCapture:
    """Read-only hook retaining exactly two graph-connected local tensors."""

    def __init__(
        self,
        transformer: Any,
        *,
        shard: SP4ContiguousShard,
        dimensions: BerniniRuntimeDimensions,
    ) -> None:
        blocks = getattr(transformer, "blocks", None)
        if blocks is None or len(blocks) != dimensions.block_count:
            raise STARCLiveVJPContractError("Bernini transformer block closure differs")
        block = blocks[dimensions.block_index]
        if not callable(getattr(block, "register_forward_hook", None)):
            raise STARCLiveVJPContractError("block.15 is not hookable")
        self.block = block
        self.shard = shard
        self.dimensions = dimensions
        self.handle: Any = None
        self.active_role: Optional[str] = None
        self.captures: dict[str, Any] = {}
        self.call_order: list[str] = []

    def install(self) -> None:
        registry = getattr(self.block, "_forward_hooks", None)
        if self.handle is not None or registry is None or len(registry) != 0:
            raise STARCLiveVJPContractError("block.15 hook registry is not exclusively available")
        self.handle = self.block.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.active_role = None

    def begin(self, role: str) -> None:
        if (
            self.handle is None
            or self.active_role is not None
            or role not in ("action", "noop")
            or role in self.captures
        ):
            raise STARCLiveVJPContractError("block.15 pair hook state differs")
        self.active_role = role

    def end(self, role: str) -> None:
        if self.active_role != role or role not in self.captures:
            raise STARCLiveVJPContractError(f"block.15 {role} hook did not fire exactly once")
        self.active_role = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        torch = _require_torch()
        role = self.active_role
        if role is None or role in self.captures:
            raise STARCLiveVJPContractError("block.15 hook fired outside pair protocol")
        wanted = (1, self.shard.local_tokens, self.dimensions.hidden_size)
        if (
            not isinstance(output, torch.Tensor)
            or tuple(output.shape) != wanted
            or not output.is_floating_point()
            or not output.requires_grad
            or output.grad_fn is None
            or not bool(torch.isfinite(output).all().item())
        ):
            raise STARCLiveVJPContractError(
                f"block.15 live hidden must be graph-connected {wanted}"
            )
        self.captures[role] = output
        self.call_order.append(role)
        return None

    def pair_hidden(self) -> tuple[Any, Any]:
        if self.active_role is not None or self.call_order != ["action", "noop"]:
            raise STARCLiveVJPContractError("block.15 action/no-op hook closure differs")
        return self.captures["action"], self.captures["noop"]


@dataclass(frozen=True)
class STARCLiveVJPProof:
    gradient: Any
    critic_score: float
    gradient_norm: float
    minimum_norm: float
    geometry: LatentPatchGeometry
    shard: SP4ContiguousShard
    candidate: CurrentCandidateBinding
    critic_artifact: Optional[VerifiedCriticArtifact]
    sketch_digest: str
    sketch_coordinates: int
    sketch_seed: int
    clean_latent_value_digest: str
    collective_backend: str
    real_sp4_autograd_collective: bool
    replica_contract_digest: str
    replica_consensus_observed: bool
    replicated_score_consensus_digest: Optional[str]
    all_rank_hidden_backward_digest: Optional[str]
    production_runtime_dimensions: bool
    hook_call_order: tuple[str, str]
    x_sigma_value_digest: Optional[str] = None
    action_condition_value_digest: Optional[str] = None
    noop_condition_value_digest: Optional[str] = None
    all_rank_hidden_backward_evidence: Optional[
        AllRankHiddenBackwardEvidence
    ] = None
    instruction_text: Optional[str] = None

    @property
    def mechanism_vjp_nonzero_finite(self) -> bool:
        return self.gradient_norm >= self.minimum_norm and math.isfinite(self.gradient_norm)

    @property
    def scientific_claim_authorized(self) -> bool:
        # This file is a mechanism probe, not an outcome evaluator or trainer.
        return False

    def blockers(self) -> tuple[str, ...]:
        rows = list(self.candidate.blockers())
        if self.critic_artifact is None or not self.critic_artifact.verified:
            rows.append("hash_bound_real_critic_checkpoint_manifest_missing")
        if not self.real_sp4_autograd_collective:
            rows.append("real_sp4_differentiable_collective_not_observed")
        if not self.replica_consensus_observed:
            rows.append("sp4_graph_input_consensus_not_observed")
        if self.replicated_score_consensus_digest is None:
            rows.append("sp4_replicated_critic_score_consensus_not_observed")
        if (
            self.all_rank_hidden_backward_digest is None
            or self.all_rank_hidden_backward_evidence is None
        ):
            rows.append("critic_backward_not_proven_on_all_sp4_hidden_shards")
        if not self.production_runtime_dimensions:
            rows.append("official_bernini_block15_runtime_not_observed")
        rows.append("frozen_bernini_checkpoint_runtime_not_hash_authenticated_here")
        rows.append("existing_starc_composite_gate_receipt_not_closed")
        rows.append("prototype_has_no_action_editing_outcome_evaluation")
        return tuple(dict.fromkeys(rows))

    def receipt(self) -> dict[str, Any]:
        artifact = self.critic_artifact
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate.candidate_id,
            "candidate_authenticated": self.candidate.authenticated,
            "candidate_manifest_path": self.candidate.manifest_path,
            "candidate_manifest_file_sha256": self.candidate.manifest_file_sha256,
            "candidate_manifest_receipt_digest": self.candidate.manifest_receipt_digest,
            "source_video_sha256": self.candidate.source_video_sha256,
            "instruction_sha256": self.candidate.instruction_sha256,
            "candidate_bound_clean_latent_tensor_sha256": (
                self.candidate.clean_latent_tensor_sha256
            ),
            "geometry": self.geometry.receipt(),
            "local_shard": self.shard.receipt(),
            "current_clean_latent_value_digest": self.clean_latent_value_digest,
            "spatial_sketch": {
                "coordinates": self.sketch_coordinates,
                "seed": self.sketch_seed,
                "patch_positions": self.geometry.patch_positions,
                "matrix_f32le_digest": self.sketch_digest,
                "content_dependent": False,
                "mask_or_track_derived": False,
            },
            "hook_coordinate": HOOK_COORDINATE,
            "hook_call_order": list(self.hook_call_order),
            "live_differentiable_forward_hook_observed": True,
            "production_runtime_dimensions": self.production_runtime_dimensions,
            "same_x_sigma_object_for_action_noop": True,
            "patch_vae_latent_call_count": 1,
            "shared_step_call_count": 2,
            "collective_backend": self.collective_backend,
            "real_sp4_autograd_collective_observed": self.real_sp4_autograd_collective,
            "sp4_preflight_replica_contract_digest": self.replica_contract_digest,
            "sp4_preflight_replica_consensus_observed": self.replica_consensus_observed,
            "sp4_replicated_critic_score_consensus_digest": (
                self.replicated_score_consensus_digest
            ),
            "sp4_all_rank_hidden_backward_digest": (
                self.all_rank_hidden_backward_digest
            ),
            "replicated_critic_scalar_divisor": SP_SIZE,
            "replicated_clean_vjp_sum_after_autograd": True,
            "critic_score": self.critic_score,
            "gradient_shape": list(self.gradient.shape),
            "gradient_norm": self.gradient_norm,
            "minimum_norm": self.minimum_norm,
            "gradient_finite": True,
            "gradient_nonzero": self.mechanism_vjp_nonzero_finite,
            "critic_artifact_verified": artifact is not None and artifact.verified,
            "critic_checkpoint_file_sha256": (
                None if artifact is None else artifact.checkpoint_file_sha256
            ),
            "critic_manifest_file_sha256": (
                None if artifact is None else artifact.manifest_file_sha256
            ),
            "critic_config_manifest_file_sha256": (
                None if artifact is None else artifact.config_manifest_file_sha256
            ),
            "mechanism_vjp_nonzero_finite": self.mechanism_vjp_nonzero_finite,
            "mechanism_probe_only": True,
            "existing_starc_composite_gate_compatible": False,
            "external_inference_inputs": list(EXTERNAL_INFERENCE_INPUTS),
            "auxiliary_spatial_inputs": [],
            "forbidden_auxiliary_inputs": list(FORBIDDEN_AUXILIARY_INPUTS),
            "generated_t2v_target_consumed": False,
            "editor_optimizer_present": False,
            "editor_optimizer_authorized": False,
            "scientific_claim_authorized": False,
            "action_editing_success_claim_authorized": False,
            "claim_blockers": list(self.blockers()),
        }


def _revalidate_composite_critic_artifact(
    artifact: VerifiedCriticArtifact,
) -> None:
    if not isinstance(artifact, VerifiedCriticArtifact) or not artifact.verified:
        raise STARCLiveVJPContractError(
            "authenticated composite requires a verified critic artifact"
        )
    checkpoint = _plain_file(artifact.checkpoint_path, label="critic checkpoint")
    checkpoint_receipt = _plain_file(
        artifact.manifest_path, label="critic checkpoint receipt"
    )
    config_receipt = _plain_file(
        artifact.config_manifest_path, label="critic config receipt"
    )
    if (
        _file_sha256(checkpoint) != artifact.checkpoint_file_sha256
        or _file_sha256(checkpoint_receipt) != artifact.manifest_file_sha256
        or _file_sha256(config_receipt) != artifact.config_manifest_file_sha256
    ):
        raise STARCLiveVJPContractError(
            "critic checkpoint or receipt changed after live VJP"
        )
    checkpoint_raw, _, _ = _strict_json_file(
        checkpoint_receipt,
        label="critic checkpoint receipt",
        expected_sha256=artifact.manifest_file_sha256,
    )
    checkpoint_manifest = _validate_sealed_receipt(
        checkpoint_raw,
        schema=CRITIC_CHECKPOINT_SCHEMA,
        label="critic checkpoint receipt",
    )
    config_raw, _, _ = _strict_json_file(
        config_receipt,
        label="critic config receipt",
        expected_sha256=artifact.config_manifest_file_sha256,
    )
    config_manifest = _validate_sealed_receipt(
        config_raw,
        schema=CRITIC_CONFIG_SCHEMA,
        label="critic config receipt",
    )
    if (
        checkpoint_manifest.get("receipt_digest")
        != artifact.manifest_receipt_digest
        or checkpoint_manifest.get("checkpoint_path") != str(checkpoint)
        or checkpoint_manifest.get("checkpoint_file_sha256")
        != artifact.checkpoint_file_sha256
        or checkpoint_manifest.get("checkpoint_state_content_digest")
        != artifact.checkpoint_state_content_digest
        or checkpoint_manifest.get("config_receipt_digest")
        != artifact.config_manifest_receipt_digest
        or config_manifest.get("receipt_digest")
        != artifact.config_manifest_receipt_digest
    ):
        raise STARCLiveVJPContractError(
            "critic checkpoint/config receipt binding changed after live VJP"
        )


def _validate_composite_live_proof(proof: STARCLiveVJPProof) -> None:
    if not isinstance(proof, STARCLiveVJPProof):
        raise STARCLiveVJPContractError("composite input is not a live VJP proof")
    candidate = proof.candidate
    artifact = proof.critic_artifact
    evidence = proof.all_rank_hidden_backward_evidence
    if (
        not candidate.authenticated
        or not candidate.geometry.is_supported_full644
        or candidate.clean_latent_tensor_sha256 is None
        or candidate.manifest_path is None
        or candidate.manifest_file_sha256 is None
        or candidate.manifest_receipt_digest is None
        or candidate.source_video_sha256 is None
        or proof.clean_latent_value_digest != candidate.clean_latent_tensor_sha256
        or artifact is None
        or not artifact.verified
        or proof.collective_backend != LIVE_VJP_SP4_IMPLEMENTATION
        or not proof.real_sp4_autograd_collective
        or not proof.replica_consensus_observed
        or proof.replicated_score_consensus_digest is None
        or proof.all_rank_hidden_backward_digest is None
        or not proof.production_runtime_dimensions
        or proof.hook_call_order != ("action", "noop")
        or proof.x_sigma_value_digest is None
        or proof.action_condition_value_digest is None
        or proof.noop_condition_value_digest is None
        or proof.action_condition_value_digest == proof.noop_condition_value_digest
        or proof.instruction_text is None
        or not proof.mechanism_vjp_nonzero_finite
        or evidence is None
    ):
        raise STARCLiveVJPContractError(
            "mechanism-only or incomplete proof cannot become a composite receipt"
        )
    for label, digest in (
        ("clean latent", proof.clean_latent_value_digest),
        ("x_sigma", proof.x_sigma_value_digest),
        ("action condition", proof.action_condition_value_digest),
        ("no-op condition", proof.noop_condition_value_digest),
        ("SP4 replica contract", proof.replica_contract_digest),
        ("replicated critic score", proof.replicated_score_consensus_digest),
        ("all-rank hidden backward", proof.all_rank_hidden_backward_digest),
    ):
        _sha256(digest, label=f"{label} SHA-256")
    if hashlib.sha256(proof.instruction_text.encode("utf-8")).hexdigest() != (
        candidate.instruction_sha256
    ):
        raise STARCLiveVJPContractError("live proof instruction/candidate binding differs")
    rebound = authenticate_current_candidate_manifest(
        candidate.manifest_path,
        expected_manifest_sha256=candidate.manifest_file_sha256,
        instruction=proof.instruction_text,
    )
    if rebound != candidate:
        raise STARCLiveVJPContractError("candidate manifest changed after live VJP")
    _revalidate_composite_critic_artifact(artifact)

    rows = evidence.receipt_rows()
    expected_fields = {
        "rank",
        "shape",
        "action_digest",
        "noop_digest",
        "norm",
        "finite_nonzero",
        "action_is_exact_negative_noop",
    }
    if len(rows) != SP_SIZE:
        raise STARCLiveVJPContractError("SP4 hidden backward rank closure differs")
    for index, row in enumerate(rows):
        norm = row.get("norm")
        if (
            set(row) != expected_fields
            or row.get("rank") != index
            or row.get("shape")
            != [1, LATENT_PHASES, SPATIAL_SKETCH_COORDINATES, HIDDEN_SIZE]
            or isinstance(norm, bool)
            or not isinstance(norm, (int, float))
            or not math.isfinite(float(norm))
            or float(norm) <= 0.0
            or row.get("finite_nonzero") is not True
            or row.get("action_is_exact_negative_noop") is not True
        ):
            raise STARCLiveVJPContractError(
                "SP4 hidden backward per-rank evidence differs"
            )
        _sha256(row.get("action_digest"), label=f"rank {index} action gradient")
        _sha256(row.get("noop_digest"), label=f"rank {index} no-op gradient")
    expected_evidence_digest = object_sha256(
        {
            "schema_version": "bernini-starc-all-rank-hidden-vjp-v2",
            "ordered_rank_evidence": rows,
        }
    )
    if (
        evidence.evidence_digest != expected_evidence_digest
        or proof.all_rank_hidden_backward_digest != expected_evidence_digest
    ):
        raise STARCLiveVJPContractError("SP4 hidden backward evidence digest differs")


def build_authenticated_composite_receipt(
    proof: STARCLiveVJPProof,
    *,
    materializer_master: str | Path,
    expected_materializer_master_sha256: str,
    bridge_source_archive: str | Path,
    expected_bridge_source_archive_sha256: str,
    bridge_source_git_revision: str,
    checkpoint_root: str | Path,
    checkpoint_content_manifest: str | Path,
    expected_checkpoint_tree_sha256: Optional[str] = None,
    expected_checkpoint_content_manifest_sha256: Optional[str] = None,
    bernini_commit: str = BERNINI_OFFICIAL_COMMIT,
    veomni_commit: str = VEOMNI_TESTED_COMMIT,
) -> dict[str, Any]:
    """Build the runner's exact v2 receipt from one in-process live proof."""

    _validate_composite_live_proof(proof)
    if bernini_commit != BERNINI_OFFICIAL_COMMIT or veomni_commit != VEOMNI_TESTED_COMMIT:
        raise STARCLiveVJPContractError("Bernini or VeOmni runtime revision is not pinned")
    source_binding = authenticate_bridge_source_archive(
        bridge_source_archive,
        expected_source_archive_sha256=expected_bridge_source_archive_sha256,
        source_git_revision=bridge_source_git_revision,
    )
    checkpoint_binding = authenticate_frozen_bernini_checkpoint_content(
        checkpoint_root,
        checkpoint_content_manifest,
        expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
        expected_checkpoint_content_manifest_sha256=(
            expected_checkpoint_content_manifest_sha256
        ),
    )
    try:
        import run_starc_core4_critic_pilot_v1 as pilot_runner
    except ImportError as error:
        raise STARCLiveVJPContractError(
            "STARC pilot runner is required to bind the materializer graph"
        ) from error
    try:
        graph = pilot_runner.StarcMaterializerAdapter.load(
            materializer_master,
            expected_master_sha256=_sha256(
                expected_materializer_master_sha256,
                label="expected materializer master SHA-256",
            ),
        )
        runner_sketch = pilot_runner.reconstruct_geometry_spatial_sketch_binding(
            proof.geometry.patch_rows, proof.geometry.patch_columns
        )
    except Exception as error:
        raise STARCLiveVJPContractError(
            "materializer graph or dynamic sketch cannot be authenticated"
        ) from error
    spatial_sketch = geometry_spatial_sketch_binding(proof.geometry)
    if canonical_json_bytes(runner_sketch) != canonical_json_bytes(spatial_sketch):
        raise STARCLiveVJPContractError("bridge/runner dynamic sketch binding differs")
    if proof.sketch_digest != spatial_sketch["critic_tensor_sha256"]:
        raise STARCLiveVJPContractError("live proof used another dynamic sketch")

    artifact = proof.critic_artifact
    candidate = proof.candidate
    evidence = proof.all_rank_hidden_backward_evidence
    assert artifact is not None
    assert evidence is not None
    gradient_digest = _tensor_value_digest(
        proof.gradient, label="current clean latent global VJP"
    )
    gradient_dtype = str(getattr(proof.gradient, "dtype", ""))
    gradient_shape = list(map(int, proof.gradient.shape))
    if gradient_dtype != "torch.float32" or gradient_shape != list(proof.geometry.latent_shape):
        raise STARCLiveVJPContractError("current clean latent global VJP tensor differs")
    rows = evidence.receipt_rows()
    collective_unsigned = {
        "world_size": SP_SIZE,
        "implementation": LIVE_VJP_SP4_IMPLEMENTATION,
        "rank_local_hidden_global_shape": [
            1,
            proof.geometry.phases,
            proof.geometry.patch_positions,
            HIDDEN_SIZE,
        ],
        "autograd_collective_tensor_shape": [
            1,
            proof.geometry.phases,
            SPATIAL_SKETCH_COORDINATES,
            HIDDEN_SIZE,
        ],
        "dynamic_spatial_sketch_critic_tensor_sha256": spatial_sketch[
            "critic_tensor_sha256"
        ],
        "preflight_replica_contract_digest": proof.replica_contract_digest,
        "replica_graph_input_consensus_observed": True,
        "replicated_score_consensus_digest": (
            proof.replicated_score_consensus_digest
        ),
        "all_rank_hidden_backward_evidence_digest": evidence.evidence_digest,
        "forward_autograd_connected": True,
        "backward_reached_all_rank_local_hidden_shards": True,
        "detached_or_object_collective_used": False,
        "ordered_rank_hidden_backward_evidence": rows,
        "rank_gradient_tensor_digests": list(
            evidence.rank_gradient_tensor_digests
        ),
    }
    collective = {
        **collective_unsigned,
        "proof_digest": object_sha256(collective_unsigned),
    }
    unsigned = {
        "schema_version": COMPOSITE_SCHEMA_VERSION,
        "critic_binding": {
            "checkpoint_path": artifact.checkpoint_path,
            "checkpoint_file_sha256": artifact.checkpoint_file_sha256,
            "checkpoint_state_content_digest": (
                artifact.checkpoint_state_content_digest
            ),
            "checkpoint_receipt_path": artifact.manifest_path,
            "checkpoint_receipt_file_sha256": artifact.manifest_file_sha256,
            "checkpoint_receipt_digest": artifact.manifest_receipt_digest,
            "config_receipt_path": artifact.config_manifest_path,
            "config_receipt_file_sha256": artifact.config_manifest_file_sha256,
            "config_receipt_digest": artifact.config_manifest_receipt_digest,
        },
        "materializer_binding": {
            "master_path": str(graph.master_path),
            "master_file_sha256": graph.master_file_sha256,
            "master_receipt_digest": graph.master_receipt_digest,
            "population_content_digest": graph.content_digest,
        },
        "live_bridge_binding": {
            **source_binding.receipt(),
            "backend_id": LIVE_VJP_BACKEND_ID,
            "bernini_commit": bernini_commit,
            "veomni_commit": veomni_commit,
            **checkpoint_binding.receipt(),
            "adapter_enabled": False,
            "frozen_bernini_and_critic": True,
        },
        "current_rv2v_clean_latent": {
            "candidate_id": candidate.candidate_id,
            "candidate_manifest_path": candidate.manifest_path,
            "candidate_manifest_file_sha256": candidate.manifest_file_sha256,
            "candidate_manifest_receipt_digest": (
                candidate.manifest_receipt_digest
            ),
            "source_video_sha256": candidate.source_video_sha256,
            "instruction_sha256": candidate.instruction_sha256,
            "tensor_sha256": proof.clean_latent_value_digest,
            "tensor_shape": list(proof.geometry.latent_shape),
            "tensor_dtype": "torch.float32",
            "requires_grad": True,
            "generated_t2v_owner_or_target": False,
            "patch_grid_height_width": [
                proof.geometry.patch_rows,
                proof.geometry.patch_columns,
            ],
            "patch_positions": proof.geometry.patch_positions,
            "spatial_sketch_binding": spatial_sketch,
        },
        "same_state_hidden_query": {
            "native_schedule_index": NATIVE_SCHEDULE_INDEX,
            "physical_sigma": NATIVE_SIGMA,
            "native_timestep": NATIVE_TIMESTEP,
            "hook_coordinate": HOOK_COORDINATE,
            "action_text_tensor_sha256": proof.action_condition_value_digest,
            "noop_text_tensor_sha256": proof.noop_condition_value_digest,
            "action_x_sigma_tensor_sha256": proof.x_sigma_value_digest,
            "noop_x_sigma_tensor_sha256": proof.x_sigma_value_digest,
            "action_and_noop_received_same_python_x_sigma_object": True,
            "action_and_noop_x_sigma_value_equal": True,
            "source_condition_consumed": False,
        },
        "sp4_differentiable_collective_proof": collective,
        "gradient_audit": {
            "tensor_sha256": gradient_digest,
            "tensor_shape": gradient_shape,
            "tensor_dtype": gradient_dtype,
            "gradient_norm": proof.gradient_norm,
            "minimum_norm": proof.minimum_norm,
            "finite": True,
            "nonzero": True,
            "reached_current_rv2v_clean_latent": True,
        },
        "generated_t2v_target_consumed": False,
        "editor_parameter_or_optimizer_present": False,
        "editor_optimizer_authorized": False,
        "scientific_critic_claim_authorized": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def write_authenticated_composite_receipt(
    output_path: str | Path,
    proof: STARCLiveVJPProof,
    **binding_arguments: Any,
) -> dict[str, Any]:
    """Rank-0 create-only emitter; it never accepts a mechanism receipt JSON."""

    if not isinstance(proof, STARCLiveVJPProof) or proof.shard.rank != 0:
        raise STARCLiveVJPContractError(
            "only SP rank 0 may emit the authenticated composite receipt"
        )
    receipt = build_authenticated_composite_receipt(
        proof, **binding_arguments
    )
    path = Path(output_path)
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        raise STARCLiveVJPContractError(
            "composite output must be a fresh absolute non-root file"
        )
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise STARCLiveVJPContractError("composite output parent is unavailable") from error
    if not parent.is_dir() or path != parent / path.name:
        raise STARCLiveVJPContractError("composite output path is not canonical")
    payload = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    try:
        with path.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(payload)
        os.chmod(path, 0o400)
    except OSError as error:
        raise STARCLiveVJPContractError("cannot create composite receipt") from error
    return {
        "schema_version": COMPOSITE_SCHEMA_VERSION,
        "receipt_path": str(path),
        "receipt_file_sha256": _file_sha256(path),
        "receipt_digest": receipt["receipt_digest"],
    }


class STARCLiveVJPBridgeV1:
    """Frozen Bernini patch/shared_step + block.15 + critic input-VJP probe."""

    def __init__(
        self,
        *,
        diffusion: Any,
        transformer: Any,
        critic: Any,
        candidate: CurrentCandidateBinding,
        instruction: str,
        action_condition: Any,
        noop_condition: Any,
        sp_rank: int,
        critic_artifact: Optional[VerifiedCriticArtifact] = None,
        collective: Optional[Any] = None,
        dimensions: BerniniRuntimeDimensions = BerniniRuntimeDimensions(),
        model_id: str = MODEL_ID,
    ) -> None:
        torch = _require_torch()
        if not isinstance(candidate, CurrentCandidateBinding):
            raise STARCLiveVJPContractError("current candidate binding type differs")
        if not isinstance(instruction, str) or not instruction.strip():
            raise STARCLiveVJPContractError("instruction must be nonempty text")
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != candidate.instruction_sha256:
            raise STARCLiveVJPContractError("instruction differs from current candidate")
        if candidate.authenticated:
            if candidate.manifest_path is None or candidate.manifest_file_sha256 is None:
                raise STARCLiveVJPContractError(
                    "authenticated candidate has no manifest file binding"
                )
            reverified_candidate = authenticate_current_candidate_manifest(
                candidate.manifest_path,
                expected_manifest_sha256=candidate.manifest_file_sha256,
                instruction=instruction,
            )
            if reverified_candidate != candidate:
                raise STARCLiveVJPContractError(
                    "current candidate binding changed after authentication"
                )
        if not isinstance(dimensions, BerniniRuntimeDimensions):
            raise STARCLiveVJPContractError("Bernini runtime dimensions type differs")
        if candidate.authenticated and not dimensions.is_production:
            raise STARCLiveVJPContractError("authenticated candidates require production dimensions")
        if model_id != MODEL_ID:
            raise STARCLiveVJPContractError("only frozen transformer_1 is supported")
        if (
            not isinstance(diffusion, torch.nn.Module)
            or not isinstance(transformer, torch.nn.Module)
            or not isinstance(critic, torch.nn.Module)
            or not callable(getattr(diffusion, "shared_step", None))
            or not callable(getattr(transformer, "patch_vae_latent", None))
            or not callable(getattr(critic, "forward_sketched_residual", None))
        ):
            raise STARCLiveVJPContractError("frozen Bernini/critic module interface differs")
        if getattr(
            getattr(diffusion, "shared_step"), "_bernini_tri_branch_unipc", None
        ) is not None:
            raise STARCLiveVJPContractError(
                "restore the full644 tri-branch sampler hook before live VJP replay"
            )
        if (
            getattr(diffusion, "transformer", None) is not transformer
            or getattr(diffusion, "transformer_2", None) is not None
        ):
            raise STARCLiveVJPContractError("diffusion must own exactly this transformer_1")
        for label, module in (
            ("diffusion", diffusion),
            ("transformer", transformer),
            ("critic", critic),
        ):
            if module.training or any(parameter.requires_grad for parameter in module.parameters()):
                raise STARCLiveVJPContractError(f"{label} must be eval-mode frozen")
        blocks = getattr(transformer, "blocks", None)
        if blocks is None or len(blocks) != dimensions.block_count:
            raise STARCLiveVJPContractError("transformer block closure differs")
        config = getattr(transformer, "config", None)
        if tuple(getattr(config, "patch_size", ())) != PATCH_SIZE:
            raise STARCLiveVJPContractError("transformer patch size must be (1,2,2)")
        dtype = getattr(transformer, "dtype", None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise STARCLiveVJPContractError("transformer dtype differs")
        if critic_artifact is not None and (
            not isinstance(critic_artifact, VerifiedCriticArtifact)
            or not critic_artifact.verified
        ):
            raise STARCLiveVJPContractError("critic artifact seal differs")
        if critic_artifact is not None and not candidate.geometry.is_supported_full644:
            raise STARCLiveVJPContractError("real critic artifact requires full644 geometry")
        if critic_artifact is not None:
            reverified_artifact = verify_frozen_starc_critic_artifact(
                critic,
                checkpoint_path=critic_artifact.checkpoint_path,
                expected_checkpoint_sha256=critic_artifact.checkpoint_file_sha256,
                manifest_path=critic_artifact.manifest_path,
                expected_manifest_sha256=critic_artifact.manifest_file_sha256,
                config_manifest_path=critic_artifact.config_manifest_path,
                expected_config_manifest_sha256=(
                    critic_artifact.config_manifest_file_sha256
                ),
            )
            if reverified_artifact != critic_artifact:
                raise STARCLiveVJPContractError(
                    "critic changed after checkpoint artifact verification"
                )

        self.diffusion = diffusion
        self.transformer = transformer
        self.critic = critic
        self.candidate = candidate
        self.instruction = instruction
        self.dimensions = dimensions
        self.model_id = model_id
        self.shard = make_sp4_contiguous_shard(candidate.geometry, sp_rank)
        self.critic_artifact = critic_artifact
        self.collective = collective or TorchDistributedNNFunctionalSP4()
        if (
            (candidate.authenticated or critic_artifact is not None)
            and type(self.collective) is not TorchDistributedNNFunctionalSP4
        ):
            raise STARCLiveVJPContractError(
                "authenticated live VJP requires the concrete differentiable SP4 backend"
            )
        if (
            not callable(getattr(self.collective, "assert_replica_consensus", None))
            or not callable(
                getattr(self.collective, "assert_replicated_score_consensus", None)
            )
            or not callable(
                getattr(self.collective, "assert_all_rank_hidden_backward", None)
            )
            or not callable(getattr(self.collective, "globalize_sketch", None))
            or not callable(getattr(self.collective, "reduce_replicated_input_vjp", None))
        ):
            raise STARCLiveVJPContractError("collective backend interface differs")
        self.action_condition = self._normalize_condition(
            action_condition, label="action condition"
        )
        self.noop_condition = self._normalize_condition(
            noop_condition, label="no-op condition"
        )
        self.action_condition_digest = _tensor_value_digest(
            self.action_condition, label="action condition"
        )
        self.noop_condition_digest = _tensor_value_digest(
            self.noop_condition, label="no-op condition"
        )
        if self.action_condition_digest == self.noop_condition_digest:
            raise STARCLiveVJPContractError("action and no-op conditions are byte-identical")

    @property
    def trainable_parameters(self) -> tuple[Any, ...]:
        return ()

    def _normalize_condition(self, value: Any, *, label: str) -> Any:
        torch = _require_torch()
        wanted = (1, self.dimensions.text_tokens, self.dimensions.text_width)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != wanted
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise STARCLiveVJPContractError(f"{label} must be frozen finite {wanted}")
        normal = value.clone() if torch.is_inference(value) else value
        if torch.is_inference(normal) or normal.requires_grad or normal.grad_fn is not None:
            raise STARCLiveVJPContractError(f"{label} could not enter input-VJP graph safely")
        return normal

    def _validate_current_state(self, clean: Any, epsilon: Any) -> tuple[Any, Any]:
        torch = _require_torch()
        wanted = self.candidate.geometry.latent_shape
        if (
            not isinstance(clean, torch.Tensor)
            or tuple(clean.shape) != wanted
            or clean.dtype != torch.float32
            or clean.device.type == "meta"
            or not clean.requires_grad
            or torch.is_inference(clean)
            or not bool(torch.isfinite(clean).all().item())
        ):
            raise STARCLiveVJPContractError(
                f"current clean latent must be live finite FP32 {wanted}"
            )
        if (
            not isinstance(epsilon, torch.Tensor)
            or tuple(epsilon.shape) != wanted
            or epsilon.dtype != torch.float32
            or epsilon.device != clean.device
            or epsilon.requires_grad
            or epsilon.grad_fn is not None
            or torch.is_inference(epsilon)
            or not bool(torch.isfinite(epsilon).all().item())
        ):
            raise STARCLiveVJPContractError(
                f"native noise must be detached finite FP32 {wanted}"
            )
        return clean, epsilon

    def _patch(self, x_sigma: Any) -> tuple[Any, Any]:
        torch = _require_torch()
        geometry = self.candidate.geometry
        result = self.transformer.patch_vae_latent(
            x_sigma.to(dtype=self.transformer.dtype), source_id=0
        )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise STARCLiveVJPContractError("patch_vae_latent return closure differs")
        tokens, rotary = result
        if (
            not isinstance(tokens, torch.Tensor)
            or tuple(tokens.shape)
            != (1, geometry.global_tokens, self.dimensions.hidden_size)
            or tokens.device != x_sigma.device
            or not tokens.requires_grad
            or tokens.grad_fn is None
            or not bool(torch.isfinite(tokens).all().item())
            or not isinstance(rotary, torch.Tensor)
            or tuple(rotary.shape)
            != (1, 1, geometry.global_tokens, self.dimensions.rotary_width)
            or rotary.dtype != torch.complex128
            or rotary.device != x_sigma.device
            or not bool(torch.isfinite(rotary).all().item())
        ):
            raise STARCLiveVJPContractError("native patch token/rotary graph contract differs")
        return tokens, rotary

    def _shared_step(
        self, *, tokens: Any, rotary: Any, timestep: Any, condition: Any
    ) -> Any:
        torch = _require_torch()
        geometry = self.candidate.geometry
        if condition.device != tokens.device:
            raise STARCLiveVJPContractError("condition and current candidate device differ")
        if (
            not isinstance(timestep, torch.Tensor)
            or tuple(timestep.shape) != (1,)
            or timestep.dtype != torch.float32
            or timestep.device != tokens.device
            or timestep.requires_grad
            or float(timestep.item()) != float(NATIVE_TIMESTEP)
        ):
            raise STARCLiveVJPContractError("shared native timestep object differs")
        prediction = self.diffusion.shared_step(
            model_id=self.model_id,
            noisy_latents=tokens,
            timesteps=timestep,
            cond_embeds=condition,
            rotary_embs=rotary,
            batch_vae_seqlen=[geometry.global_tokens],
            batch_text_seqlen=[self.dimensions.text_tokens],
        )
        if (
            not isinstance(prediction, torch.Tensor)
            or tuple(prediction.shape)
            != (1, geometry.global_tokens, geometry.patch_dimension)
            or prediction.device != tokens.device
            or not prediction.is_floating_point()
            or not prediction.requires_grad
            or prediction.grad_fn is None
            or not bool(torch.isfinite(prediction).all().item())
        ):
            raise STARCLiveVJPContractError("native shared_step graph/output contract differs")
        return prediction

    def prove_current_clean_latent_vjp(
        self,
        current_clean_latent: Any,
        epsilon: Any,
        *,
        minimum_norm: float = 1.0e-12,
    ) -> STARCLiveVJPProof:
        """Execute exactly two frozen forwards and consume the graph in one VJP."""

        torch = _require_torch()
        if torch.is_inference_mode_enabled():
            raise STARCLiveVJPContractError("live VJP cannot execute inside inference_mode")
        threshold = float(minimum_norm)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise STARCLiveVJPContractError("minimum VJP norm must be positive finite")
        clean, noise = self._validate_current_state(current_clean_latent, epsilon)
        clean_digest = _tensor_value_digest(clean, label="current clean latent")
        if (
            self.candidate.authenticated
            and clean_digest != self.candidate.clean_latent_tensor_sha256
        ):
            raise STARCLiveVJPContractError(
                "current clean latent differs from authenticated candidate snapshot"
            )
        geometry = self.candidate.geometry
        sketch_digest = fixed_spatial_sketch_digest(
            geometry.patch_positions,
            coordinates=self.dimensions.spatial_sketch_coordinates,
            seed=SPATIAL_SKETCH_SEED,
        )
        replica_contract_digest = object_sha256(
            {
                "schema_version": "bernini-starc-live-vjp-sp4-preflight-v1",
                "candidate_id": self.candidate.candidate_id,
                "candidate_manifest_receipt_digest": (
                    self.candidate.manifest_receipt_digest
                ),
                "geometry": geometry.receipt(),
                "clean_latent_value_digest": clean_digest,
                "epsilon_value_digest": _tensor_value_digest(
                    noise, label="native noise"
                ),
                "action_condition_value_digest": self.action_condition_digest,
                "noop_condition_value_digest": self.noop_condition_digest,
                "critic_manifest_receipt_digest": (
                    None
                    if self.critic_artifact is None
                    else self.critic_artifact.manifest_receipt_digest
                ),
                "spatial_sketch_digest": sketch_digest,
                "native_sigma_float32_be_hex": struct.pack(
                    "!f", float(NATIVE_SIGMA)
                ).hex(),
                "native_timestep": NATIVE_TIMESTEP,
                "hook_coordinate": HOOK_COORDINATE,
            }
        )
        replica_consensus = self.collective.assert_replica_consensus(
            replica_contract_digest, rank=self.shard.rank
        )
        if type(replica_consensus) is not bool:
            raise STARCLiveVJPContractError("SP4 preflight consensus evidence differs")

        with torch.enable_grad():
            x_sigma = (1.0 - NATIVE_SIGMA) * clean + NATIVE_SIGMA * noise
            if not x_sigma.requires_grad or x_sigma.grad_fn is None:
                raise STARCLiveVJPContractError("shared noisy state detached from current clean")
            before = x_sigma.detach().clone()
            tokens, rotary = self._patch(x_sigma)
            timestep = torch.tensor(
                [float(NATIVE_TIMESTEP)], dtype=torch.float32, device=tokens.device
            )
            protected = {
                "x_sigma": (id(x_sigma), int(x_sigma._version), _tensor_value_digest(x_sigma, label="x_sigma")),
                "tokens": (id(tokens), int(tokens._version), _tensor_value_digest(tokens, label="patched tokens")),
                "rotary": (id(rotary), int(rotary._version), _tensor_value_digest(rotary, label="rotary")),
                "timestep": (id(timestep), int(timestep._version), _tensor_value_digest(timestep, label="native timestep")),
            }

            def assert_shared_state_unchanged(stage: str) -> None:
                for name, tensor in (
                    ("x_sigma", x_sigma),
                    ("tokens", tokens),
                    ("rotary", rotary),
                    ("timestep", timestep),
                ):
                    expected_id, expected_version, expected_digest = protected[name]
                    if (
                        id(tensor) != expected_id
                        or int(tensor._version) != expected_version
                        or _tensor_value_digest(tensor, label=name) != expected_digest
                    ):
                        raise STARCLiveVJPContractError(
                            f"{stage} mutated shared {name} state"
                        )

            observer = _Block15LivePairCapture(
                self.transformer, shard=self.shard, dimensions=self.dimensions
            )
            observer.install()
            try:
                observer.begin("action")
                self._shared_step(
                    tokens=tokens,
                    rotary=rotary,
                    timestep=timestep,
                    condition=self.action_condition,
                )
                observer.end("action")
                assert_shared_state_unchanged("action forward")
                observer.begin("noop")
                self._shared_step(
                    tokens=tokens,
                    rotary=rotary,
                    timestep=timestep,
                    condition=self.noop_condition,
                )
                observer.end("noop")
                assert_shared_state_unchanged("no-op forward")
                action_hidden, noop_hidden = observer.pair_hidden()
                call_order = tuple(observer.call_order)
            finally:
                observer.remove()
            if not torch.equal(x_sigma.detach(), before):
                raise STARCLiveVJPContractError("frozen branch forward mutated shared x_sigma")

            sketch = make_fixed_spatial_sketch(
                geometry.patch_positions,
                coordinates=self.dimensions.spatial_sketch_coordinates,
                seed=SPATIAL_SKETCH_SEED,
                device=action_hidden.device,
            )
            if _tensor_f32le_digest(
                sketch.detach().cpu(), label="runtime spatial sketch"
            ) != sketch_digest:
                raise STARCLiveVJPContractError("runtime sketch does not match geometry digest")
            local_action_sketch = sketch_rank_local_hidden_exact(
                action_hidden,
                geometry=geometry,
                shard=self.shard,
                spatial_sketch=sketch,
            )
            local_noop_sketch = sketch_rank_local_hidden_exact(
                noop_hidden,
                geometry=geometry,
                shard=self.shard,
                spatial_sketch=sketch,
            )
            action_assembly = self.collective.globalize_sketch(
                local_action_sketch,
                geometry=geometry,
                shard=self.shard,
                role="action",
            )
            noop_assembly = self.collective.globalize_sketch(
                local_noop_sketch,
                geometry=geometry,
                shard=self.shard,
                role="noop",
            )
            wanted_sketch_shape = (
                1,
                geometry.phases,
                self.dimensions.spatial_sketch_coordinates,
                self.dimensions.hidden_size,
            )
            for role, assembly in (
                ("action", action_assembly),
                ("noop", noop_assembly),
            ):
                if (
                    not isinstance(assembly, GlobalSketchAssembly)
                    or tuple(assembly.tensor.shape) != wanted_sketch_shape
                    or assembly.rank != self.shard.rank
                    or assembly.role != role
                    or assembly.tensor.dtype != torch.float32
                    or not assembly.tensor.requires_grad
                    or assembly.tensor.grad_fn is None
                    or not bool(torch.isfinite(assembly.tensor).all().item())
                ):
                    raise STARCLiveVJPContractError(
                        f"global {role} FP32 sketch assembly evidence differs"
                    )
            if (
                action_assembly.backend != noop_assembly.backend
                or action_assembly.real_sp4_autograd_collective
                != noop_assembly.real_sp4_autograd_collective
            ):
                raise STARCLiveVJPContractError("action/no-op collective backend differs")
            sketched_residual = (
                action_assembly.tensor - noop_assembly.tensor
            ).float().contiguous()
            if (
                tuple(sketched_residual.shape) != wanted_sketch_shape
                or sketched_residual.dtype != torch.float32
                or not sketched_residual.requires_grad
                or sketched_residual.grad_fn is None
            ):
                raise STARCLiveVJPContractError(
                    "global FP32 action-sketch minus no-op-sketch differs"
                )
            critic_output = self.critic.forward_sketched_residual(
                sketched_residual, require_input_grad=True
            )
            score = getattr(critic_output, "score", None)
            if (
                not isinstance(score, torch.Tensor)
                or score.numel() != 1
                or not score.requires_grad
                or score.grad_fn is None
                or not bool(torch.isfinite(score).all().item())
            ):
                raise STARCLiveVJPContractError("frozen critic scalar graph differs")
            score_consensus_digest = self.collective.assert_replicated_score_consensus(
                score, rank=self.shard.rank
            )
            if score_consensus_digest is not None:
                _sha256(
                    score_consensus_digest,
                    label="replicated critic score consensus digest",
                )

            # Every SP rank computes the same replicated critic scalar.  The
            # autograd all-reduce backward sums four scalar cotangents, so each
            # rank contributes score/4.  The resulting clean-input shard VJPs
            # are then summed once to recover the full replicated-input VJP.
            local_vjp, action_hidden_cotangent, noop_hidden_cotangent = torch.autograd.grad(
                score.sum() / float(SP_SIZE),
                (clean, local_action_sketch, local_noop_sketch),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            all_rank_hidden_backward_evidence = (
                self.collective.assert_all_rank_hidden_backward(
                    action_hidden_cotangent,
                    noop_hidden_cotangent,
                    rank=self.shard.rank,
                )
            )
            all_rank_hidden_backward_digest = None
            if all_rank_hidden_backward_evidence is not None:
                if not isinstance(
                    all_rank_hidden_backward_evidence,
                    AllRankHiddenBackwardEvidence,
                ):
                    raise STARCLiveVJPContractError(
                        "all-rank hidden backward structured evidence differs"
                    )
                all_rank_hidden_backward_digest = (
                    all_rank_hidden_backward_evidence.evidence_digest
                )
                _sha256(
                    all_rank_hidden_backward_digest,
                    label="all-rank hidden backward digest",
                )
            global_vjp = self.collective.reduce_replicated_input_vjp(
                local_vjp, rank=self.shard.rank
            )
        if (
            not isinstance(global_vjp, torch.Tensor)
            or tuple(global_vjp.shape) != geometry.latent_shape
            or not bool(torch.isfinite(global_vjp).all().item())
        ):
            raise STARCLiveVJPContractError("current clean-latent global VJP differs")
        norm = torch.linalg.vector_norm(global_vjp.float())
        if not bool(torch.isfinite(norm).item()) or float(norm.item()) < threshold:
            raise STARCLiveVJPContractError("current clean-latent VJP is zero or non-finite")
        return STARCLiveVJPProof(
            gradient=global_vjp.detach(),
            critic_score=float(score.detach().item()),
            gradient_norm=float(norm.item()),
            minimum_norm=threshold,
            geometry=geometry,
            shard=self.shard,
            candidate=self.candidate,
            critic_artifact=self.critic_artifact,
            sketch_digest=sketch_digest,
            sketch_coordinates=self.dimensions.spatial_sketch_coordinates,
            sketch_seed=SPATIAL_SKETCH_SEED,
            clean_latent_value_digest=clean_digest,
            collective_backend=str(action_assembly.backend),
            real_sp4_autograd_collective=bool(
                type(self.collective) is TorchDistributedNNFunctionalSP4
                and action_assembly.real_sp4_autograd_collective
                and noop_assembly.real_sp4_autograd_collective
            ),
            replica_contract_digest=replica_contract_digest,
            replica_consensus_observed=replica_consensus,
            replicated_score_consensus_digest=score_consensus_digest,
            all_rank_hidden_backward_digest=all_rank_hidden_backward_digest,
            production_runtime_dimensions=self.dimensions.is_production,
            hook_call_order=call_order,  # type: ignore[arg-type]
            x_sigma_value_digest=protected["x_sigma"][2],
            action_condition_value_digest=self.action_condition_digest,
            noop_condition_value_digest=self.noop_condition_digest,
            all_rank_hidden_backward_evidence=all_rank_hidden_backward_evidence,
            instruction_text=self.instruction,
        )


__all__ = [
    "AllRankHiddenBackwardEvidence",
    "BERNINI_CHECKPOINT_CONTENT_FILE_COUNT",
    "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
    "BERNINI_CHECKPOINT_TREE_SHA256",
    "BERNINI_OFFICIAL_COMMIT",
    "BLOCK_INDEX",
    "BerniniCheckpointContentBinding",
    "BerniniRuntimeDimensions",
    "BridgeSourceArchiveBinding",
    "CANDIDATE_BINDING_SCHEMA",
    "COMPOSITE_SCHEMA_VERSION",
    "CurrentCandidateBinding",
    "EXTERNAL_INFERENCE_INPUTS",
    "FORBIDDEN_AUXILIARY_INPUTS",
    "GEOMETRY_NEUTRAL_CRITIC_CONFIG",
    "GlobalSketchAssembly",
    "HOOK_COORDINATE",
    "LatentPatchGeometry",
    "LIVE_VJP_BACKEND_ID",
    "LIVE_VJP_BRIDGE_ARCHIVE_MEMBER",
    "LIVE_VJP_SP4_IMPLEMENTATION",
    "NATIVE_SCHEDULE_INDEX",
    "NATIVE_SIGMA",
    "NATIVE_TIMESTEP",
    "NON_HEAD_CRITIC_STATE_KEYS",
    "PATCH_SIZE",
    "SCHEMA_VERSION",
    "SP4ContiguousShard",
    "SPATIAL_SKETCH_COORDINATES",
    "SPATIAL_SKETCH_SEED",
    "STARCLiveVJPBridgeV1",
    "STARCLiveVJPContractError",
    "STARCLiveVJPProof",
    "SUPPORTED_FULL644_LATENT_SHAPES",
    "TorchDistributedNNFunctionalSP4",
    "VerifiedCriticArtifact",
    "VEOMNI_TESTED_COMMIT",
    "assemble_sp4_shards_for_test",
    "authenticate_current_candidate_manifest",
    "authenticate_bridge_source_archive",
    "authenticate_frozen_bernini_checkpoint_content",
    "build_authenticated_composite_receipt",
    "canonical_json_bytes",
    "fixed_spatial_sketch_digest",
    "geometry_spatial_sketch_binding",
    "make_fixed_spatial_sketch",
    "make_sp4_contiguous_shard",
    "mechanism_only_candidate_binding",
    "object_sha256",
    "scatter_local_shard_to_global",
    "sketch_rank_local_hidden_exact",
    "verify_frozen_starc_critic_artifact",
    "write_authenticated_composite_receipt",
]

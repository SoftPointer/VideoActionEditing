#!/usr/bin/env python3
"""Materialize the frozen Bernini STARC core4 block-15 hidden residuals.

One ``materialize-group`` invocation owns one four-rank Ulysses group and two
sealed core4 cells.  For every cell it reconstructs the exact 13-arm episode,
queries the frozen, adapter-off Bernini transformer under the cell action and
scene-matched no-op prompts at one *shared* noisy state, and stores only the
fixed spatial sketch of ``block.15.output(action) - block.15.output(noop)``.

The generated RGB, clean latent, Gaussian, full hidden state, velocity, and
labels are never editor inputs or targets.  The only persistent tensor is a
detached finite FP32 ``[1,21,16,1536]`` critic feature.  This program has no
optimizer and cannot authorize an editor update or a scientific critic claim.

Core4 contains three authenticated native geometries: ``60x62``, ``64x58``,
and ``68x54``, giving patch grids ``30x31`` (P=930), ``32x29`` (P=928), and
``34x27`` (P=918).  Each episode derives its grid from its authenticated clean
latent and uses the same counter-hash sketch family scaled by ``1/sqrt(P)``.
The verified FITQ contiguous Ulysses global-index/phase rule is generalized to
``N=21P``; no historical fixed 31-by-30 orientation is treated as ground truth.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
for _search_root in (METHOD_ROOT, TOOLS_ROOT):
    if str(_search_root) not in sys.path:
        sys.path.insert(0, str(_search_root))

import internal_temporal_quotient_observer as fitq_observer  # noqa: E402
import latent_temporal_event_critic_dataset as dataset_contract  # noqa: E402
import temporal_counterfactual_action_scorer_v1 as temporal_scorer  # noqa: E402
import temporal_counterfactual_contract_v1 as temporal_contract  # noqa: E402
import author_pair_v5_core4_event_labels_d541801_v3 as label_author  # noqa: E402


ARM_SCHEMA = "bernini-starc-core4-same-state-hidden-arm-v1"
GROUP_SCHEMA = "bernini-starc-core4-same-state-hidden-group-v1"
MASTER_SCHEMA = "bernini-starc-core4-same-state-hidden-master-v1"

ARM_RECEIPT_FILENAME = "starc-block15-hidden-arm-receipt-v1.json"
TENSOR_FILENAME = "starc-block15-hidden-residual.safetensors"
GROUP_FILENAME = "starc-core4-hidden-group-{group_id}-v1.json"
MASTER_FILENAME = "starc-core4-hidden-master-v1.json"
TENSOR_KEY = "sketched_action_minus_noop_hidden_residual"

GROUP_ORDER = ("sp4-a", "sp4-b")
ARM_ORDER = tuple(dataset_contract.ARM_ROLES)
PROMPT_ORDER = ("target_action", "noop")
CELLS_PER_GROUP = 2
ARMS_PER_CELL = 13
ARMS_PER_GROUP = CELLS_PER_GROUP * ARMS_PER_CELL
CORE4_CELL_COUNT = 4
CORE4_CANDIDATE_COUNT = 40
CORE4_ARM_COUNT = CORE4_CELL_COUNT * ARMS_PER_CELL
MODEL_FORWARDS_PER_ARM = 2
MODEL_FORWARDS_PER_GROUP = ARMS_PER_GROUP * MODEL_FORWARDS_PER_ARM
MODEL_FORWARDS_TOTAL = CORE4_ARM_COUNT * MODEL_FORWARDS_PER_ARM

LATENT_PREFIX = (1, 16, 21)
LATENT_PHASES = 21
CORE4_LATENT_SHAPES = (
    (1, 16, 21, 60, 62),
    (1, 16, 21, 64, 58),
    (1, 16, 21, 68, 54),
)
CORE4_PATCH_GRIDS = ((30, 31), (32, 29), (34, 27))
HIDDEN_SIZE = 1536
SKETCH_COORDINATES = 16
SKETCH_SEED = 20260808017
SKETCH_FAMILY_ID = "starc-counter-rademacher-s20260808017-v1"
SKETCH_CONSTRUCTION_ID = "sha256-counter-rademacher-f32le-v1"
SKETCH_VALUE_DIGEST_SCHEME = "fitq-canonical-fp32-little-endian-v1"
P930_SKETCH_RAW_BYTES_SHA256 = (
    "5a75404b60cadddb29ac7473fc4596d7ebfcd306acfb3fa1a6bc6575a228a246"
)
P930_SKETCH_VALUE_SHA256 = (
    "be43863f6a000fb00083798610e3993200c24e5fd94dcb2ef7d4e3858618dde7"
)
P930_SKETCH_CRITIC_TENSOR_SHA256 = (
    "4a8330c77079671f6515bda07acc21f0d060176c4c07d2609ad2553acf657561"
)
SKETCH_DIGESTS_BY_PATCH_POSITIONS = {
    930: (
        P930_SKETCH_RAW_BYTES_SHA256,
        P930_SKETCH_VALUE_SHA256,
        P930_SKETCH_CRITIC_TENSOR_SHA256,
    ),
    928: (
        "260d47275c7d407512ff4fca9fa20d2223eaa29b6e4d151b7495e51721980df4",
        "9fdee154009d0d4283716a4e93abe4df2dde5241065040eaf05bd2c9a9f2fa64",
        "be52cac4d90f0a5a70368d25fef2fb1edb4d346fb10598329f5bb7e8e7285ede",
    ),
    918: (
        "f48f9577ec829cc67bd5f9da09721bebccec7e6c92b18f5322e25ab76f19192a",
        "d05582d93963ae8de876171526f00671b7fbe0ca27841b1ab4c32b196afbc911",
        "9cc6e96d5909542189ca43ea2ff54efda6a44b302483890629b82d2ecad7f7ba",
    ),
}

HOOK_COORDINATE = "block.15.output"
SCHEDULE_INDEX = 33
SIGMA = 0.5161304473876953
NATIVE_TIMESTEP = 516
EXPECTED_SP_WORLD = 4

REQUIRED_ROOT_SPEC_RAW_SHA256 = (
    "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95"
)
REQUIRED_BANK_RECEIPT_FILE_SHA256 = (
    "8c4f77bdd24fa14786f3dff28a4044d819f444c0338484a2fa6df9588100cb59"
)
REQUIRED_BANK_RECEIPT_DIGEST = (
    "79276ad5f499fe37775a23bc5a789b7eb6dd83170517f750daecf07bb782cdb2"
)
REQUIRED_DETACHED_LABEL_FILE_SHA256 = (
    "9246504e97e1ee46c2cdcf7dfac0f41364dca40f26e5c26f28f0968d0443808d"
)
REQUIRED_CRITIC_USE_EVIDENCE_SHA256 = (
    "c24e0193b29c7a8fa05cf9a25035ac01816fe54f3b80820b8e8de47418b90457"
)
REQUIRED_CRITIC_USE_SOURCE = (
    "md/action_editing/bernini_starc_core4_critic_use_authority_20260808.md"
)
REQUIRED_D541801_SOURCE_REVISION = "d541801a162796aacde34c2bfc2b1f0472d954d2"
REQUIRED_D541801_SOURCE_ARCHIVE_SHA256 = (
    "535aba9b5445e2a9b06cf3da267325c49d247ab2ef9f4a9dd129a51fdbb008c7"
)
REQUIRED_D541801_SCORER_SOURCE_SHA256 = (
    "3d7ce459ddb9a014873acd6384c7c4030b4e3aca9004c1b8486ebbc1f0f5d32e"
)

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class STARCMaterializationError(RuntimeError):
    """A bank, label, frozen forward, hook, artifact, or receipt failed closed."""


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
        raise STARCMaterializationError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
        raise STARCMaterializationError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise STARCMaterializationError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise STARCMaterializationError(f"{label} must be lowercase SHA-1")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise STARCMaterializationError(f"{label} must be a path-safe identifier")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise STARCMaterializationError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise STARCMaterializationError(
            f"{label} must be an absolute plain directory"
        )
    return path.resolve(strict=True)


def _fresh_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise STARCMaterializationError(
            f"{label} must be a fresh absolute file under a plain parent"
        )
    return path


def _reject_constant(token: str) -> None:
    raise STARCMaterializationError(f"non-finite JSON constant is forbidden: {token}")


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise STARCMaterializationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json_file(
    value: str | Path, *, expected_sha256: str, label: str
) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(value, label=label)
    observed = file_sha256(path)
    if observed != _sha256(expected_sha256, label=f"{label} expected SHA-256"):
        raise STARCMaterializationError(f"{label} file SHA-256 differs")
    try:
        decoded = json.loads(
            path.read_text(encoding="ascii"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise STARCMaterializationError(f"{label} is invalid ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise STARCMaterializationError(f"{label} root must be an object")
    return decoded, path, observed


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(unsigned)
    return {**row, "receipt_digest": object_sha256(row)}


def _verify_seal(value: Mapping[str, Any], *, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise STARCMaterializationError(f"{label} must be an object")
    row = dict(value)
    declared = _sha256(row.pop("receipt_digest", None), label=f"{label} digest")
    if row.get("schema_version") != schema or object_sha256(row) != declared:
        raise STARCMaterializationError(f"{label} schema or digest differs")
    return {**row, "receipt_digest": declared}


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise STARCMaterializationError(f"refusing to overwrite {path}")
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    os.chmod(path, 0o400)
    return hashlib.sha256(raw).hexdigest()


def tensor_sha256(value: Any) -> str:
    """Hash exact tensor values with dtype/shape/layout provenance."""

    import torch

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise STARCMaterializationError("tensor hash requires a real tensor")
    owned = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in owned.shape],
        "dtype": str(owned.dtype),
        "layout": str(owned.layout),
    }
    raw = owned.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
    return hashlib.sha256(canonical_json_bytes(metadata) + b"\x00" + raw).hexdigest()


def verify_authenticated_native_clean_tensor_identity(
    value: Any,
    artifact: Mapping[str, Any],
    *,
    label: str,
    frozen: Any,
    allowed_latent_shapes: Sequence[tuple[int, ...]] = CORE4_LATENT_SHAPES,
    allowed_patch_grids: Sequence[tuple[int, int]] = CORE4_PATCH_GRIDS,
    geometry_label: str = "sealed core4",
) -> dict[str, Any]:
    """Authenticate an old or new native clean-latent receipt fail-closed.

    Historical core4 clean receipts bind a complete single-tensor container,
    key/shape/FP32 storage, native coordinate/role, and byte-exact round trip,
    but do not claim producer-time raw/content value digests.  Both absent is
    therefore a registered compatibility path; exactly one present is always
    rejected.  Current raw/content hashes are sealed as materializer-time
    observations after reopening the authenticated file.
    """

    import torch
    from safetensors import safe_open

    historical_fields = {
        "artifact_role",
        "coordinate",
        "mp4_decode_reencode_used",
        "native_sampler_before_vae_decode",
        "origin",
        "path",
        "roundtrip_byte_exact_fp32",
        "sampler_return_dtype",
        "sha256",
        "shape",
        "source_video_vae_encode_before_any_decode",
        "stored_dtype",
        "tensor_key",
    }
    value_fields = {"raw_value_sha256", "content_sha256"}
    if not isinstance(artifact, Mapping):
        raise STARCMaterializationError(f"{label} native artifact must be an object")
    declared = set(artifact) & value_fields
    if declared not in (set(), value_fields):
        raise STARCMaterializationError(
            f"{label} declares a partial native value identity"
        )
    if set(artifact) != historical_fields | declared:
        raise STARCMaterializationError(
            f"{label} historical native artifact field closure differs"
        )
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3]) != LATENT_PREFIX
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise STARCMaterializationError(
            f"{label} must be detached contiguous CPU FP32 exact81"
        )
    latent_geometry(
        value,
        allowed_latent_shapes=allowed_latent_shapes,
        allowed_patch_grids=allowed_patch_grids,
        geometry_label=geometry_label,
    )

    raw_path = artifact.get("path")
    if not isinstance(raw_path, (str, Path)):
        raise STARCMaterializationError(f"{label} artifact path differs")
    requested = Path(raw_path)
    path = _plain_file(requested, label=f"{label} artifact")
    if requested != path:
        raise STARCMaterializationError(
            f"{label} artifact path is not canonical plain path text"
        )
    container_sha256 = _sha256(
        artifact.get("sha256"), label=f"{label} container SHA-256"
    )
    if file_sha256(path) != container_sha256:
        raise STARCMaterializationError(
            f"{label} authenticated container SHA-256 differs"
        )
    tensor_key = "normalized_clean_latent"
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [tensor_key]:
            raise STARCMaterializationError(
                f"{label} authenticated container key closure differs"
            )
        reopened = opened.get_tensor(tensor_key).contiguous()
        metadata = dict(opened.metadata() or {})
    if file_sha256(path) != container_sha256:
        raise STARCMaterializationError(
            f"{label} authenticated container changed while reopening"
        )
    if (
        reopened.dtype != torch.float32
        or reopened.shape != value.shape
        or not torch.equal(reopened, value)
    ):
        raise STARCMaterializationError(
            f"{label} loaded value differs from authenticated container"
        )

    expected = {
        "tensor_key": tensor_key,
        "shape": [int(item) for item in value.shape],
        "stored_dtype": "torch.float32",
        "sampler_return_dtype": "torch.float32",
        "coordinate": "bernini_normalized_clean_vae_latent",
        "artifact_role": "native_sampler_proposal",
        "origin": "native_sampler_before_vae_decode",
        "native_sampler_before_vae_decode": True,
        "source_video_vae_encode_before_any_decode": False,
        "mp4_decode_reencode_used": False,
        "roundtrip_byte_exact_fp32": True,
    }
    for field, expected_value in expected.items():
        if artifact.get(field) != expected_value:
            raise STARCMaterializationError(
                f"{label} historical artifact field {field} differs"
            )
    expected_metadata = {
        "coordinate": "bernini_normalized_clean_vae_latent",
        "frame_contract": "exact81_latent21",
        "artifact_role": "native_sampler_proposal",
        "source": "native_sampler_before_vae_decode",
    }
    if metadata != expected_metadata:
        raise STARCMaterializationError(f"{label} safetensors metadata differs")

    try:
        actual = frozen.native_tensor_value_identity(value)
        reopened_identity = frozen.native_tensor_value_identity(reopened)
        if declared:
            strict = frozen.verify_native_tensor_value_identity(
                value, artifact, label=label
            )
            if strict != actual:
                raise STARCMaterializationError(
                    f"{label} strict/current native identity differs"
                )
    except frozen.PairV5T2VEnergyScoringError as error:
        raise STARCMaterializationError(str(error)) from error
    if (
        reopened_identity != actual
        or actual.get("shape") != expected["shape"]
        or actual.get("dtype") != "torch.float32"
        or set(actual)
        != {
            "shape",
            "dtype",
            "numel",
            "byte_count",
            "raw_value_sha256",
            "content_sha256",
        }
    ):
        raise STARCMaterializationError(
            f"{label} current tensor/container value identity differs"
        )
    recorded = bool(declared)
    unsigned = {
        **actual,
        "authenticated_container_path": str(path),
        "authenticated_container_sha256": container_sha256,
        "single_tensor_container_reopened_byte_exact": True,
        "safetensors_metadata": metadata,
        "historical_native_coordinate_role_roundtrip_verified": True,
        "recorded_value_hashes_present": recorded,
        "historical_native_receipt_value_hashes_absent": not recorded,
        "strict_recorded_value_identity_verified": recorded,
        "native_receipt_value_hashes_synthesized": False,
        "producer_time_value_digest_claimed_by_materializer": False,
        "observed_value_hashes_recomputed_after_authenticated_reopen": True,
        "value_identity_observation_time": "materializer_authenticated_reopen",
        "identity_authority": (
            "recorded_native_value_digests_and_authenticated_container"
            if recorded
            else "authenticated_single_tensor_container_sha256_and_native_fp32_roundtrip"
        ),
    }
    return {**unsigned, "binding_digest": object_sha256(unsigned)}


def validate_clean_latent_authentication_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise STARCMaterializationError(
            "clean latent authentication binding must be an object"
        )
    row = dict(value)
    declared = _sha256(
        row.pop("binding_digest", None), label="clean authentication binding digest"
    )
    if object_sha256(row) != declared:
        raise STARCMaterializationError("clean authentication binding digest differs")
    for field in (
        "raw_value_sha256",
        "content_sha256",
        "authenticated_container_sha256",
    ):
        _sha256(row.get(field), label=f"clean authentication {field}")
    if (
        row.get("dtype") != "torch.float32"
        or type(row.get("recorded_value_hashes_present")) is not bool
        or type(row.get("historical_native_receipt_value_hashes_absent")) is not bool
        or row.get("single_tensor_container_reopened_byte_exact") is not True
        or row.get("historical_native_coordinate_role_roundtrip_verified") is not True
        or row.get("strict_recorded_value_identity_verified")
        is not row.get("recorded_value_hashes_present")
        or row.get("historical_native_receipt_value_hashes_absent")
        != (not row.get("recorded_value_hashes_present"))
        or row.get("native_receipt_value_hashes_synthesized") is not False
        or row.get("producer_time_value_digest_claimed_by_materializer") is not False
        or row.get("observed_value_hashes_recomputed_after_authenticated_reopen")
        is not True
        or row.get("value_identity_observation_time")
        != "materializer_authenticated_reopen"
    ):
        raise STARCMaterializationError(
            "clean authentication semantic closure differs"
        )
    return {**row, "binding_digest": declared}


def _validate_patch_grid(
    patch_height: Any,
    patch_width: Any,
    *,
    allowed_patch_grids: Sequence[tuple[int, int]] = CORE4_PATCH_GRIDS,
    geometry_label: str = "sealed core4",
) -> tuple[int, int, int]:
    if (
        type(patch_height) is not int
        or type(patch_width) is not int
        or (patch_height, patch_width) not in allowed_patch_grids
    ):
        raise STARCMaterializationError(
            f"patch grid is not a {geometry_label} geometry"
        )
    return patch_height, patch_width, patch_height * patch_width


def latent_geometry(
    value: Any,
    *,
    allowed_latent_shapes: Sequence[tuple[int, ...]] = CORE4_LATENT_SHAPES,
    allowed_patch_grids: Sequence[tuple[int, int]] = CORE4_PATCH_GRIDS,
    geometry_label: str = "sealed core4",
) -> tuple[tuple[int, ...], int, int, int]:
    """Return exact native shape, patch H/W, and P for one registered tensor."""

    shape = tuple(int(item) for item in getattr(value, "shape", value))
    if shape not in allowed_latent_shapes:
        raise STARCMaterializationError(
            f"latent is not one of the {geometry_label} geometries"
        )
    patch_height = shape[3] // 2
    patch_width = shape[4] // 2
    _validate_patch_grid(
        patch_height,
        patch_width,
        allowed_patch_grids=allowed_patch_grids,
        geometry_label=geometry_label,
    )
    return shape, patch_height, patch_width, patch_height * patch_width


def fixed_spatial_sketch(
    *,
    patch_height: int,
    patch_width: int,
    device: Any = "cpu",
    allowed_patch_grids: Sequence[tuple[int, int]] = CORE4_PATCH_GRIDS,
    geometry_label: str = "sealed core4",
) -> Any:
    """Reconstruct one geometry-specific full-support FP32 sketch."""

    import torch

    patch_height, patch_width, patch_positions = _validate_patch_grid(
        patch_height,
        patch_width,
        allowed_patch_grids=allowed_patch_grids,
        geometry_label=geometry_label,
    )
    scale = torch.tensor(1.0 / math.sqrt(patch_positions), dtype=torch.float32)
    matrix = torch.empty(SKETCH_COORDINATES, patch_positions, dtype=torch.float32)
    for row in range(SKETCH_COORDINATES):
        for column in range(patch_positions):
            token = f"{SKETCH_SEED}:{row}:{column}".encode("ascii")
            matrix[row, column] = scale if hashlib.sha256(token).digest()[0] & 1 else -scale
    if int(torch.linalg.matrix_rank(matrix).item()) != SKETCH_COORDINATES or not bool(
        (matrix != 0).all().item()
    ):
        raise STARCMaterializationError("fixed spatial sketch reconstruction differs")
    return matrix.detach().to(device=device)


def reconstruct_spatial_sketch_bytes(
    *,
    patch_height: int,
    patch_width: int,
    allowed_patch_grids: Sequence[tuple[int, int]] = CORE4_PATCH_GRIDS,
    geometry_label: str = "sealed core4",
) -> bytes:
    """Pure-Python reference reconstruction for receipts and CPU audits."""

    _patch_height, _patch_width, patch_positions = _validate_patch_grid(
        patch_height,
        patch_width,
        allowed_patch_grids=allowed_patch_grids,
        geometry_label=geometry_label,
    )
    scale = struct.unpack(
        "<f", struct.pack("<f", 1.0 / math.sqrt(float(patch_positions)))
    )[0]
    output = bytearray(SKETCH_COORDINATES * patch_positions * 4)
    offset = 0
    for row in range(SKETCH_COORDINATES):
        for column in range(patch_positions):
            positive = hashlib.sha256(
                f"{SKETCH_SEED}:{row}:{column}".encode("ascii")
            ).digest()[0] & 1
            struct.pack_into("<f", output, offset, scale if positive else -scale)
            offset += 4
    return bytes(output)


def spatial_sketch_digests(
    *,
    patch_height: int,
    patch_width: int,
    allowed_patch_grids: Sequence[tuple[int, int]] = CORE4_PATCH_GRIDS,
    geometry_label: str = "sealed core4",
) -> tuple[str, str, str]:
    """Return raw, FITQ-value, and critic-canonical digests for one grid."""

    _patch_height, _patch_width, patch_positions = _validate_patch_grid(
        patch_height,
        patch_width,
        allowed_patch_grids=allowed_patch_grids,
        geometry_label=geometry_label,
    )
    raw = reconstruct_spatial_sketch_bytes(
        patch_height=patch_height,
        patch_width=patch_width,
        allowed_patch_grids=allowed_patch_grids,
        geometry_label=geometry_label,
    )
    return (
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(
            f"fitq-canonical-fp32-little-endian-v1|shape=16,{patch_positions}|".encode(
                "ascii"
            )
            + raw
        ).hexdigest(),
        hashlib.sha256(
            f"bernini-ltec-f32le-v1|shape=16,{patch_positions}|".encode("ascii")
            + raw
        ).hexdigest(),
    )


def sketch_binding(
    matrix: Any,
    *,
    patch_height: int,
    patch_width: int,
    allowed_patch_grids: Sequence[tuple[int, int]] = CORE4_PATCH_GRIDS,
    expected_digests_by_patch_positions: Mapping[
        int, tuple[str, str, str]
    ] = SKETCH_DIGESTS_BY_PATCH_POSITIONS,
    geometry_label: str = "sealed core4",
) -> dict[str, Any]:
    import torch

    patch_height, patch_width, patch_positions = _validate_patch_grid(
        patch_height,
        patch_width,
        allowed_patch_grids=allowed_patch_grids,
        geometry_label=geometry_label,
    )
    if (
        not isinstance(matrix, torch.Tensor)
        or tuple(int(item) for item in matrix.shape)
        != (SKETCH_COORDINATES, patch_positions)
        or matrix.dtype != torch.float32
        or matrix.requires_grad
        or matrix.grad_fn is not None
        or not bool(torch.isfinite(matrix).all().item())
    ):
        raise STARCMaterializationError("spatial sketch tensor closure differs")
    owned = matrix.detach().cpu().contiguous().clone()
    raw = owned.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
    raw_digest = hashlib.sha256(raw).hexdigest()
    value_digest = hashlib.sha256(
        f"fitq-canonical-fp32-little-endian-v1|shape=16,{patch_positions}|".encode(
            "ascii"
        )
        + raw
    ).hexdigest()
    critic_digest = hashlib.sha256(
        f"bernini-ltec-f32le-v1|shape=16,{patch_positions}|".encode("ascii")
        + raw
    ).hexdigest()
    expected_digests = expected_digests_by_patch_positions.get(patch_positions)
    if expected_digests != (raw_digest, value_digest, critic_digest):
        raise STARCMaterializationError("geometry-specific spatial sketch digest differs")
    if expected_digests != spatial_sketch_digests(
        patch_height=patch_height,
        patch_width=patch_width,
        allowed_patch_grids=allowed_patch_grids,
        geometry_label=geometry_label,
    ):
        raise STARCMaterializationError("Torch/Python sketch reconstruction differs")
    return {
        "sketch_family_id": SKETCH_FAMILY_ID,
        "sketch_id": (
            f"starc-patch{patch_height}x{patch_width}-"
            f"counter-rademacher-s{SKETCH_SEED}-v1"
        ),
        "construction_id": SKETCH_CONSTRUCTION_ID,
        "seed": SKETCH_SEED,
        "matrix_shape": [SKETCH_COORDINATES, patch_positions],
        "patch_positions": patch_positions,
        "patch_grid_height_width": [patch_height, patch_width],
        "flatten_order": "patch-y-x",
        "normalization": f"per-row-rademacher-1-over-sqrt-{patch_positions}",
        "matrix_dtype": "torch.float32",
        "matrix_raw_bytes_sha256": raw_digest,
        "matrix_value_digest_scheme": SKETCH_VALUE_DIGEST_SCHEME,
        "matrix_value_sha256": value_digest,
        "critic_tensor_digest_scheme": "bernini-ltec-f32le-v1",
        "critic_tensor_sha256": critic_digest,
        "full_support_no_mask_or_localizer": True,
        "data_dependent": False,
    }


def apply_temporal_transform(clean: Any, transform: str) -> Any:
    """Apply one registered transform before noising; never transform epsilon."""

    import torch

    if (
        not isinstance(clean, torch.Tensor)
        or clean.dtype != torch.float32
        or clean.requires_grad
        or clean.grad_fn is not None
        or not bool(torch.isfinite(clean).all().item())
    ):
        raise STARCMaterializationError(
            "clean latent must be detached finite FP32"
        )
    source_shape, _patch_height, _patch_width, _patch_positions = latent_geometry(clean)
    maps = {
        "chronological": tuple(range(LATENT_PHASES)),
        "reverse": tuple(range(LATENT_PHASES - 1, -1, -1)),
        "freeze_first": (0,) * LATENT_PHASES,
        "phase_shuffle": tuple((8 * index) % LATENT_PHASES for index in range(LATENT_PHASES)),
    }
    if transform not in maps:
        raise STARCMaterializationError("temporal transform is not registered")
    index = torch.tensor(maps[transform], dtype=torch.long, device=clean.device)
    result = clean.index_select(2, index).float().contiguous().detach()
    if tuple(int(item) for item in result.shape) != source_shape:
        raise STARCMaterializationError("transformed clean geometry differs")
    return result


@dataclass(frozen=True)
class STARCLocalLayout:
    """Target-token layout for one contiguous official Ulysses rank shard."""

    sp_rank: int
    patch_height: int
    patch_width: int
    patch_positions: int
    shard_start: int
    shard_stop: int
    local_sequence_length: int
    valid_sequence_tokens: int
    padding_tokens_excluded: int
    target_tokens_selected: int
    target_local_indices: Any = field(repr=False)
    target_phase_indices: Any = field(repr=False)
    target_patch_indices: Any = field(repr=False)
    phase_token_count: Any = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sp_rank": self.sp_rank,
            "sp_world": EXPECTED_SP_WORLD,
            "shard_start": self.shard_start,
            "shard_stop": self.shard_stop,
            "local_sequence_length": self.local_sequence_length,
            "valid_sequence_tokens": self.valid_sequence_tokens,
            "padding_tokens_excluded": self.padding_tokens_excluded,
            "target_tokens_selected": self.target_tokens_selected,
            "phase_token_count": [int(item) for item in self.phase_token_count.tolist()],
            "patch_positions": self.patch_positions,
            "patch_grid_height_width": [self.patch_height, self.patch_width],
            "patch_flatten_order": "patch-y-x",
        }


def build_starc_local_layout(
    sp_rank: int,
    *,
    patch_height: int,
    patch_width: int,
    allowed_patch_grids: Sequence[tuple[int, int]] = CORE4_PATCH_GRIDS,
    geometry_label: str = "sealed core4",
) -> STARCLocalLayout:
    """Generalize FITQ's contiguous Ulysses mapping to the episode's exact P."""

    import torch

    if type(sp_rank) is not int or not 0 <= sp_rank < EXPECTED_SP_WORLD:
        raise STARCMaterializationError("SP rank is outside the pinned world")
    patch_height, patch_width, patch_positions = _validate_patch_grid(
        patch_height,
        patch_width,
        allowed_patch_grids=allowed_patch_grids,
        geometry_label=geometry_label,
    )
    global_tokens = LATENT_PHASES * patch_positions
    local_length = math.ceil(global_tokens / EXPECTED_SP_WORLD)
    shard_start = sp_rank * local_length
    shard_stop = shard_start + local_length
    local_index = torch.arange(local_length, dtype=torch.int64)
    global_index = local_index + shard_start
    valid = global_index < global_tokens
    selected_local = local_index[valid]
    selected_global = global_index[valid]
    phase = torch.div(selected_global, patch_positions, rounding_mode="floor")
    patch = torch.remainder(selected_global, patch_positions)
    counts = torch.bincount(phase, minlength=LATENT_PHASES).to(torch.int64)
    if (
        patch_height * patch_width != patch_positions
        or int(counts.sum().item()) != int(selected_local.numel())
        or (phase.numel() and (int(phase.min()) < 0 or int(phase.max()) >= LATENT_PHASES))
        or (patch.numel() and (int(patch.min()) < 0 or int(patch.max()) >= patch_positions))
    ):
        raise STARCMaterializationError("STARC Ulysses target layout differs")
    return STARCLocalLayout(
        sp_rank=sp_rank,
        patch_height=patch_height,
        patch_width=patch_width,
        patch_positions=patch_positions,
        shard_start=shard_start,
        shard_stop=shard_stop,
        local_sequence_length=local_length,
        valid_sequence_tokens=int(valid.sum().item()),
        padding_tokens_excluded=local_length - int(valid.sum().item()),
        target_tokens_selected=int(valid.sum().item()),
        target_local_indices=selected_local,
        target_phase_indices=phase,
        target_patch_indices=patch,
        phase_token_count=counts,
    )


@dataclass(frozen=True)
class LocalBlock15Sketch:
    branch: str
    layout: STARCLocalLayout
    sketch: Any = field(repr=False)
    target_hidden_shape: tuple[int, ...]
    target_hidden_value_sha256: str
    block0_input_value_sha256: str
    block0_attn1_value_sha256: str
    hook_call_counts: Mapping[str, int]


class Block15SpatialSketchObserver:
    """Read-only narrow hook: block-0 parity plus block-15 output sketch."""

    HOOK_ORDER = ("block.00.input", "block.00.attn1", HOOK_COORDINATE)

    def __init__(
        self,
        model: Any,
        *,
        sp_rank: int,
        patch_height: int,
        patch_width: int,
        spatial_sketch: Any,
        allowed_latent_shapes: Sequence[tuple[int, ...]] = CORE4_LATENT_SHAPES,
        allowed_patch_grids: Sequence[tuple[int, int]] = CORE4_PATCH_GRIDS,
        geometry_label: str = "sealed core4",
    ) -> None:
        import torch

        try:
            self.transformer = fitq_observer.resolve_pinned_wan_transformer(model)
        except fitq_observer.InternalTemporalQuotientObserverError as error:
            raise STARCMaterializationError(str(error)) from error
        self.layout = build_starc_local_layout(
            sp_rank,
            patch_height=patch_height,
            patch_width=patch_width,
            allowed_patch_grids=allowed_patch_grids,
            geometry_label=geometry_label,
        )
        self.allowed_latent_shapes = tuple(allowed_latent_shapes)
        self.allowed_patch_grids = tuple(allowed_patch_grids)
        self.geometry_label = geometry_label
        if (
            not isinstance(spatial_sketch, torch.Tensor)
            or tuple(int(item) for item in spatial_sketch.shape)
            != (SKETCH_COORDINATES, self.layout.patch_positions)
            or spatial_sketch.dtype != torch.float32
            or spatial_sketch.requires_grad
            or spatial_sketch.grad_fn is not None
        ):
            raise STARCMaterializationError("observer spatial sketch differs")
        self.spatial_sketch = spatial_sketch.detach()
        self._handles: list[Any] = []
        self._installed = False
        self._active_branch: Optional[str] = None
        self._order: list[str] = []
        self._hashes: dict[str, str] = {}
        self._sketch: Any = None
        self._hidden_shape: Optional[tuple[int, ...]] = None
        self._poisoned: Optional[str] = None

    @property
    def trainable_parameters(self) -> tuple[Any, ...]:
        return ()

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def active(self) -> bool:
        return self._active_branch is not None

    def _fail(self, message: str) -> STARCMaterializationError:
        self._poisoned = message
        return STARCMaterializationError(message)

    def install(self) -> "Block15SpatialSketchObserver":
        if self._installed or self._active_branch is not None:
            raise STARCMaterializationError("observer installation state differs")
        handles: list[Any] = []
        try:
            handles.append(
                self.transformer.blocks[0].register_forward_pre_hook(
                    self._make_pre_hook("block.00.input")
                )
            )
            handles.append(
                self.transformer.blocks[0].attn1.to_out[0].register_forward_pre_hook(
                    self._make_pre_hook("block.00.attn1")
                )
            )
            handles.append(
                self.transformer.blocks[15].register_forward_hook(
                    self._block15_hook
                )
            )
        except Exception:
            for handle in reversed(handles):
                handle.remove()
            raise
        self._handles = handles
        self._installed = True
        return self

    def remove(self) -> None:
        if not self._installed or self._active_branch is not None:
            raise STARCMaterializationError("observer removal state differs")
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._installed = False

    def __enter__(self) -> "Block15SpatialSketchObserver":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._active_branch is not None:
            self.abort()
        if self._installed:
            self.remove()

    def begin(self, branch: str) -> None:
        if (
            not self._installed
            or self._active_branch is not None
            or self._poisoned is not None
            or not isinstance(branch, str)
            or not branch
        ):
            raise STARCMaterializationError("observer begin state differs")
        self._active_branch = branch
        self._order = []
        self._hashes = {}
        self._sketch = None
        self._hidden_shape = None

    def abort(self) -> None:
        if self._active_branch is None:
            raise STARCMaterializationError("no active observer capture")
        self._active_branch = None
        self._order = []
        self._hashes = {}
        self._sketch = None
        self._hidden_shape = None

    @contextmanager
    def capture(self, branch: str) -> Iterator[list[LocalBlock15Sketch]]:
        holder: list[LocalBlock15Sketch] = []
        self.begin(branch)
        try:
            yield holder
        except BaseException:
            if self._active_branch is not None:
                self.abort()
            raise
        else:
            holder.append(self.finish())

    def _check_order(self, site: str) -> None:
        wanted = self.HOOK_ORDER[len(self._order)] if len(self._order) < 3 else None
        if site != wanted:
            raise self._fail(f"unexpected hook order: got {site}, expected {wanted}")
        self._order.append(site)

    def _validate_local_hidden(self, tensor: Any, *, site: str) -> Any:
        import torch

        wanted = (1, self.layout.local_sequence_length, HIDDEN_SIZE)
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(int(item) for item in tensor.shape) != wanted
            or not tensor.is_floating_point()
            or tensor.device.type == "meta"
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise self._fail(f"{site} local hidden closure differs")
        return tensor

    def _make_pre_hook(self, site: str) -> Callable[..., None]:
        def hook(module: Any, inputs: Sequence[Any]) -> None:
            del module
            if self._active_branch is None or not isinstance(inputs, tuple) or not inputs:
                raise self._fail(f"{site} fired outside an active tensor forward")
            self._check_order(site)
            value = self._validate_local_hidden(inputs[0], site=site)
            self._hashes[site] = tensor_sha256(value)
            return None

        return hook

    def _block15_hook(self, module: Any, inputs: Sequence[Any], output: Any) -> None:
        del module, inputs
        import torch

        if self._active_branch is None:
            raise self._fail("block.15.output fired outside an active forward")
        self._check_order(HOOK_COORDINATE)
        hidden = self._validate_local_hidden(output, site=HOOK_COORDINATE)
        local_index = self.layout.target_local_indices.to(device=hidden.device)
        phase = self.layout.target_phase_indices.to(device=hidden.device)
        patch = self.layout.target_patch_indices.to(device=hidden.device)
        values = hidden.detach()[0].index_select(0, local_index).float()
        if tuple(values.shape) != (self.layout.target_tokens_selected, HIDDEN_SIZE):
            raise self._fail("block-15 target-token selection differs")
        sketch_matrix = self.spatial_sketch.to(device=hidden.device)
        result = torch.zeros(
            LATENT_PHASES,
            SKETCH_COORDINATES,
            HIDDEN_SIZE,
            dtype=torch.float32,
            device=hidden.device,
        )
        for coordinate in range(SKETCH_COORDINATES):
            weights = sketch_matrix[coordinate].index_select(0, patch).unsqueeze(1)
            result[:, coordinate, :].index_add_(0, phase, values * weights)
        result = result.unsqueeze(0).contiguous().detach()
        if (
            tuple(int(item) for item in result.shape)
            != (1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE)
            or result.requires_grad
            or result.grad_fn is not None
            or not bool(torch.isfinite(result).all().item())
        ):
            raise self._fail("rank-local block-15 sketch differs")
        self._hidden_shape = tuple(int(item) for item in values.shape)
        self._hashes[HOOK_COORDINATE] = tensor_sha256(values)
        self._sketch = result
        return None

    def finish(self) -> LocalBlock15Sketch:
        branch = self._active_branch
        try:
            if (
                branch is None
                or tuple(self._order) != self.HOOK_ORDER
                or tuple(self._hashes) != self.HOOK_ORDER
                or self._sketch is None
                or self._hidden_shape is None
            ):
                raise self._fail("narrow hook capture is incomplete")
            return LocalBlock15Sketch(
                branch=branch,
                layout=self.layout,
                sketch=self._sketch,
                target_hidden_shape=self._hidden_shape,
                target_hidden_value_sha256=self._hashes[HOOK_COORDINATE],
                block0_input_value_sha256=self._hashes["block.00.input"],
                block0_attn1_value_sha256=self._hashes["block.00.attn1"],
                hook_call_counts=MappingProxyType({site: 1 for site in self.HOOK_ORDER}),
            )
        finally:
            self._active_branch = None
            self._order = []
            self._hashes = {}
            self._sketch = None
            self._hidden_shape = None


@dataclass(frozen=True)
class GlobalBlock15Sketch:
    sketch: Any = field(repr=False)
    sketch_tensor_sha256: str
    local_hidden_value_sha256_by_rank: tuple[str, ...]
    global_hidden_composite_sha256: str
    local_layouts: tuple[Mapping[str, Any], ...]
    phase_patch_count: tuple[int, ...]


def all_reduce_block15_sketch(
    local: LocalBlock15Sketch, *, dist_module: Any = None, group: Any = None
) -> GlobalBlock15Sketch:
    """All-reduce exactly one rank-local ``[1,21,16,1536]`` sketch."""

    import torch

    dist = torch.distributed if dist_module is None else dist_module
    try:
        world = int(dist.get_world_size(group=group))
    except TypeError:
        world = int(dist.get_world_size(group))
    if world != EXPECTED_SP_WORLD:
        raise STARCMaterializationError("block-15 sketch reduction requires SP world 4")
    value = local.sketch.clone()
    count = local.layout.phase_token_count.to(device=value.device, dtype=torch.int64).clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
    if (
        tuple(int(item) for item in value.shape)
        != (1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE)
        or value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
        or not torch.equal(
            count.cpu(),
            torch.full(
                (LATENT_PHASES,), local.layout.patch_positions, dtype=torch.int64
            ),
        )
    ):
        raise STARCMaterializationError("global block-15 sketch/coverage differs")
    local_rows: list[Any] = [None] * world
    dist.all_gather_object(
        local_rows,
        {
            "rank": local.layout.sp_rank,
            "hidden_value_sha256": local.target_hidden_value_sha256,
            "layout": local.layout.as_dict(),
        },
        group=group,
    )
    if (
        any(not isinstance(row, Mapping) for row in local_rows)
        or [row["rank"] for row in local_rows] != list(range(world))
        or any(
            _SHA256_RE.fullmatch(str(row.get("hidden_value_sha256"))) is None
            for row in local_rows
        )
    ):
        raise STARCMaterializationError("rank-local hidden hash gathering differs")
    hashes = tuple(str(row["hidden_value_sha256"]) for row in local_rows)
    layouts = tuple(dict(row["layout"]) for row in local_rows)
    composite = object_sha256(
        {
            "scheme": "ordered-rank-local-target-hidden-value-sha256-v1",
            "hook_coordinate": HOOK_COORDINATE,
            "rank_order": list(range(world)),
            "hashes": list(hashes),
            "layouts": list(layouts),
        }
    )
    return GlobalBlock15Sketch(
        sketch=value.detach(),
        sketch_tensor_sha256=tensor_sha256(value),
        local_hidden_value_sha256_by_rank=hashes,
        global_hidden_composite_sha256=composite,
        local_layouts=layouts,
        phase_patch_count=tuple(int(item) for item in count.cpu().tolist()),
    )


_ARTIFACT_FIELDS = frozenset(
    {
        "path",
        "file_sha256",
        "tensor_key",
        "tensor_shape",
        "tensor_dtype",
        "tensor_sha256",
        "detached_finite_fp32",
    }
)
_SKETCH_FIELDS = frozenset(
    {
        "sketch_family_id",
        "sketch_id",
        "construction_id",
        "seed",
        "matrix_shape",
        "patch_positions",
        "patch_grid_height_width",
        "flatten_order",
        "normalization",
        "matrix_dtype",
        "matrix_raw_bytes_sha256",
        "matrix_value_digest_scheme",
        "matrix_value_sha256",
        "critic_tensor_digest_scheme",
        "critic_tensor_sha256",
        "full_support_no_mask_or_localizer",
        "data_dependent",
    }
)
_ARM_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "episode_id",
        "split",
        "role",
        "label",
        "action_family_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "seed",
        "source_candidate_binding",
        "event_label_binding",
        "critic_use_binding",
        "latent_binding",
        "official_gaussian_binding",
        "prompt_binding",
        "same_state_query_binding",
        "hidden_binding",
        "spatial_sketch_binding",
        "artifact",
        "model_binding",
        "runtime_binding",
        "model_forward_count",
        "labels_entered_model_condition",
        "training_performed",
        "optimizer_authorized",
        "editor_optimizer_authorized",
        "scientific_critic_claim_authorized",
        "generated_media_editor_use_authorized",
        "receipt_digest",
    }
)
_ARM_BINDING_FIELDS = frozenset(
    {
        "episode_id",
        "split",
        "role",
        "label",
        "receipt_path",
        "receipt_file_sha256",
        "receipt_digest",
        "artifact_path",
        "artifact_file_sha256",
        "artifact_tensor_sha256",
    }
)
_GROUP_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "root_spec_binding",
        "bank_binding",
        "detached_event_label_binding",
        "critic_use_binding",
        "model_binding",
        "runtime_binding",
        "spatial_sketch_bindings_by_episode",
        "episode_order",
        "episode_splits",
        "arm_order",
        "arm_bindings",
        "candidate_count",
        "episode_count",
        "arm_count",
        "tensor_artifact_count",
        "model_forward_count",
        "training_performed",
        "optimizer_authorized",
        "editor_optimizer_authorized",
        "scientific_critic_claim_authorized",
        "generated_media_editor_use_authorized",
        "receipt_digest",
    }
)
_GROUP_BINDING_FIELDS = frozenset(
    {
        "group_id",
        "manifest_path",
        "manifest_file_sha256",
        "receipt_digest",
        "episode_order",
        "episode_splits",
        "arm_count",
        "model_forward_count",
    }
)
_MASTER_FIELDS = frozenset(
    {
        "schema_version",
        "root_spec_binding",
        "bank_binding",
        "detached_event_label_binding",
        "critic_use_binding",
        "model_binding",
        "runtime_binding",
        "spatial_sketch_bindings_by_episode",
        "group_order",
        "group_bindings",
        "episode_order",
        "episode_splits",
        "arm_order",
        "candidate_count",
        "episode_count",
        "arm_count",
        "tensor_artifact_count",
        "model_forward_count",
        "fit_episode_count",
        "confirmation_episode_count",
        "confirmation_consumed_by_optimizer",
        "training_performed",
        "optimizer_authorized",
        "editor_optimizer_authorized",
        "scientific_critic_claim_authorized",
        "generated_media_editor_use_authorized",
        "receipt_digest",
    }
)


def _exact_fields(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise STARCMaterializationError(
            f"{label} fields differ; expected={sorted(fields)!r}, actual={actual!r}"
        )
    return dict(value)


def _absolute_path_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise STARCMaterializationError(f"{label} must be absolute path text")
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise STARCMaterializationError(f"{label} must be absolute and non-root")
    return str(path)


def _validate_denials(row: Mapping[str, Any], *, label: str) -> None:
    for name in (
        "training_performed",
        "optimizer_authorized",
        "editor_optimizer_authorized",
        "scientific_critic_claim_authorized",
        "generated_media_editor_use_authorized",
    ):
        if row.get(name) is not False:
            raise STARCMaterializationError(f"{label} {name} must remain false")


def validate_spatial_sketch_binding(value: Any) -> dict[str, Any]:
    row = _exact_fields(value, _SKETCH_FIELDS, label="spatial sketch binding")
    grid = row["patch_grid_height_width"]
    if not isinstance(grid, list) or len(grid) != 2:
        raise STARCMaterializationError("spatial sketch patch grid differs")
    patch_height, patch_width, patch_positions = _validate_patch_grid(grid[0], grid[1])
    for name in (
        "matrix_raw_bytes_sha256",
        "matrix_value_sha256",
        "critic_tensor_sha256",
    ):
        _sha256(row[name], label=f"spatial sketch {name}")
    if (
        row["sketch_family_id"] != SKETCH_FAMILY_ID
        or row["construction_id"] != SKETCH_CONSTRUCTION_ID
        or row["seed"] != SKETCH_SEED
        or row["matrix_shape"] != [SKETCH_COORDINATES, patch_positions]
        or row["patch_positions"] != patch_positions
        or row["flatten_order"] != "patch-y-x"
        or row["normalization"]
        != f"per-row-rademacher-1-over-sqrt-{patch_positions}"
        or row["matrix_dtype"] != "torch.float32"
        or row["matrix_value_digest_scheme"] != SKETCH_VALUE_DIGEST_SCHEME
        or row["critic_tensor_digest_scheme"] != "bernini-ltec-f32le-v1"
        or row["full_support_no_mask_or_localizer"] is not True
        or row["data_dependent"] is not False
    ):
        raise STARCMaterializationError("spatial sketch contract differs")
    matrix = fixed_spatial_sketch(
        patch_height=patch_height, patch_width=patch_width
    )
    if row != sketch_binding(
        matrix, patch_height=patch_height, patch_width=patch_width
    ):
        raise STARCMaterializationError("spatial sketch reconstructed bytes differ")
    return row


def _validate_artifact(value: Any, *, verify_file: bool) -> dict[str, Any]:
    import torch

    row = _exact_fields(value, _ARTIFACT_FIELDS, label="residual artifact")
    path_text = _absolute_path_text(row["path"], label="residual artifact path")
    _sha256(row["file_sha256"], label="residual artifact file SHA-256")
    _sha256(row["tensor_sha256"], label="residual artifact tensor SHA-256")
    if (
        row["tensor_key"] != TENSOR_KEY
        or row["tensor_shape"] != [1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE]
        or row["tensor_dtype"] != "torch.float32"
        or row["detached_finite_fp32"] is not True
    ):
        raise STARCMaterializationError("residual artifact tensor contract differs")
    if verify_file:
        from safetensors import safe_open

        path = _plain_file(path_text, label="residual artifact")
        if file_sha256(path) != row["file_sha256"]:
            raise STARCMaterializationError("residual artifact file hash differs")
        with safe_open(str(path), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [TENSOR_KEY]:
                raise STARCMaterializationError("residual artifact key closure differs")
            tensor = opened.get_tensor(TENSOR_KEY)
        if (
            tuple(int(item) for item in tensor.shape)
            != (1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE)
            or tensor.dtype != torch.float32
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not bool(torch.isfinite(tensor).all().item())
            or tensor_sha256(tensor) != row["tensor_sha256"]
        ):
            raise STARCMaterializationError("residual artifact tensor bytes differ")
    return row


def save_residual_artifact(path: Path, residual: Any) -> dict[str, Any]:
    """Create and immediately reopen one detached safetensors artifact."""

    import torch
    from safetensors.torch import save_file

    path = _fresh_file(path, label="residual artifact output")
    if (
        not isinstance(residual, torch.Tensor)
        or tuple(int(item) for item in residual.shape)
        != (1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE)
        or residual.dtype != torch.float32
        or residual.requires_grad
        or residual.grad_fn is not None
        or not bool(torch.isfinite(residual).all().item())
    ):
        raise STARCMaterializationError("residual tensor must be detached finite FP32")
    owned = residual.detach().cpu().contiguous().clone()
    tensor_digest = tensor_sha256(owned)
    save_file({TENSOR_KEY: owned}, str(path))
    os.chmod(path, 0o400)
    row = {
        "path": str(path.resolve(strict=True)),
        "file_sha256": file_sha256(path),
        "tensor_key": TENSOR_KEY,
        "tensor_shape": [1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE],
        "tensor_dtype": "torch.float32",
        "tensor_sha256": tensor_digest,
        "detached_finite_fp32": True,
    }
    return _validate_artifact(row, verify_file=True)


def make_arm_receipt(
    *,
    group_id: str,
    episode_id: str,
    split: str,
    role: str,
    label: int,
    action_family_id: str,
    actor_group_id: str,
    scene_group_id: str,
    action_group_id: str,
    seed: int,
    source_candidate_binding: Mapping[str, Any],
    event_label_binding: Mapping[str, Any],
    critic_use_binding: Mapping[str, Any],
    latent_binding: Mapping[str, Any],
    official_gaussian_binding: Mapping[str, Any],
    prompt_binding: Mapping[str, Any],
    same_state_query_binding: Mapping[str, Any],
    hidden_binding: Mapping[str, Any],
    spatial_sketch_binding: Mapping[str, Any],
    artifact: Mapping[str, Any],
    model_binding: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": ARM_SCHEMA,
        "group_id": group_id,
        "episode_id": episode_id,
        "split": split,
        "role": role,
        "label": label,
        "action_family_id": action_family_id,
        "actor_group_id": actor_group_id,
        "scene_group_id": scene_group_id,
        "action_group_id": action_group_id,
        "seed": seed,
        "source_candidate_binding": dict(source_candidate_binding),
        "event_label_binding": dict(event_label_binding),
        "critic_use_binding": dict(critic_use_binding),
        "latent_binding": dict(latent_binding),
        "official_gaussian_binding": dict(official_gaussian_binding),
        "prompt_binding": dict(prompt_binding),
        "same_state_query_binding": dict(same_state_query_binding),
        "hidden_binding": dict(hidden_binding),
        "spatial_sketch_binding": dict(spatial_sketch_binding),
        "artifact": dict(artifact),
        "model_binding": dict(model_binding),
        "runtime_binding": dict(runtime_binding),
        "model_forward_count": MODEL_FORWARDS_PER_ARM,
        "labels_entered_model_condition": False,
        "training_performed": False,
        "optimizer_authorized": False,
        "editor_optimizer_authorized": False,
        "scientific_critic_claim_authorized": False,
        "generated_media_editor_use_authorized": False,
    }
    result = _seal(unsigned)
    validate_arm_receipt(result, verify_artifact=False)
    return result


def validate_arm_receipt(value: Any, *, verify_artifact: bool = True) -> dict[str, Any]:
    row = _exact_fields(value, _ARM_FIELDS, label="STARC arm receipt")
    row = _verify_seal(row, schema=ARM_SCHEMA, label="STARC arm receipt")
    _safe_id(row["group_id"], label="group_id")
    _safe_id(row["episode_id"], label="episode_id")
    for name in ("action_family_id", "actor_group_id", "scene_group_id", "action_group_id"):
        _safe_id(row[name], label=name)
    if (
        row["group_id"] not in GROUP_ORDER
        or row["split"] not in dataset_contract.PILOT_SPLITS
        or row["role"] not in ARM_ORDER
        or row["label"] != (1 if row["role"] == "positive" else 0)
        or type(row["seed"]) is not int
        or not 0 <= row["seed"] < 2**63
        or row["model_forward_count"] != MODEL_FORWARDS_PER_ARM
        or row["labels_entered_model_condition"] is not False
    ):
        raise STARCMaterializationError("STARC arm identity/label closure differs")
    _validate_denials(row, label="STARC arm")
    _validate_artifact(row["artifact"], verify_file=verify_artifact)
    query = row["same_state_query_binding"]
    if (
        not isinstance(query, Mapping)
        or query.get("native_schedule_index") != SCHEDULE_INDEX
        or float(query.get("sigma", -1.0)).hex() != float(SIGMA).hex()
        or query.get("native_timestep") != NATIVE_TIMESTEP
        or query.get("action_and_noop_share_exact_x_sigma_object") is not True
        or query.get("action_and_noop_share_exact_rotary_object") is not True
        or query.get("action_and_noop_share_exact_timestep_object") is not True
        or query.get("shared_tensor_bytes_unchanged") is not True
        or query.get("block0_input_and_attn1_exact_parity") is not True
    ):
        raise STARCMaterializationError("STARC same-state query proof differs")
    latent = row["latent_binding"]
    clean_authentication = validate_clean_latent_authentication_binding(
        latent.get("clean_latent_authentication")
        if isinstance(latent, Mapping)
        else None
    )
    expected_transform = dataset_contract.TEMPORAL_TRANSFORM_BY_ROLE[row["role"]]
    try:
        source_shape, patch_height, patch_width, patch_positions = latent_geometry(
            latent.get("source_shape") if isinstance(latent, Mapping) else ()
        )
    except STARCMaterializationError as error:
        raise STARCMaterializationError("STARC latent geometry binding differs") from error
    if (
        not isinstance(latent, Mapping)
        or latent.get("source_shape") != list(source_shape)
        or latent.get("transformed_shape") != list(source_shape)
        or latent.get("temporal_transform") != expected_transform
        or latent.get("transform_applied_before_noising") is not True
        or clean_authentication.get("shape") != list(source_shape)
        or clean_authentication.get("authenticated_container_sha256")
        != latent.get("file_sha256")
        or clean_authentication.get("raw_value_sha256")
        != latent.get("raw_value_sha256")
        or clean_authentication.get("content_sha256")
        != latent.get("content_sha256")
    ):
        raise STARCMaterializationError("STARC latent/transform binding differs")
    hidden = row["hidden_binding"]
    if (
        not isinstance(hidden, Mapping)
        or hidden.get("hook_coordinate") != HOOK_COORDINATE
        or hidden.get("action_global_sketch_shape")
        != [1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE]
        or hidden.get("noop_global_sketch_shape")
        != [1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE]
        or hidden.get("residual_shape")
        != [1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE]
        or hidden.get("full_hidden_persisted") is not False
    ):
        raise STARCMaterializationError("STARC hidden binding differs")
    checked_sketch = validate_spatial_sketch_binding(row["spatial_sketch_binding"])
    if checked_sketch["patch_grid_height_width"] != [patch_height, patch_width]:
        raise STARCMaterializationError("STARC arm uses another spatial sketch")
    if hidden.get("patch_positions") != patch_positions or hidden.get(
        "patch_grid_height_width"
    ) != [patch_height, patch_width]:
        raise STARCMaterializationError("STARC hidden/sketch geometry differs")
    event = row["event_label_binding"]
    if (
        not isinstance(event, Mapping)
        or event.get("labels_are_external_and_detached") is not True
        or event.get("labels_may_enter_model_condition") is not False
    ):
        raise STARCMaterializationError("STARC detached-label binding differs")
    use = row["critic_use_binding"]
    if (
        not isinstance(use, Mapping)
        or use.get("bank_receipt_digest") != REQUIRED_BANK_RECEIPT_DIGEST
        or use.get("authorization_evidence_sha256")
        != REQUIRED_CRITIC_USE_EVIDENCE_SHA256
        or use.get("authorized_use") != dataset_contract.CRITIC_ONLY_USE
    ):
        raise STARCMaterializationError("STARC critic-only authority differs")
    return row


def arm_binding_from_receipt(
    receipt: Mapping[str, Any], *, receipt_path: str | Path, receipt_file_sha256: str
) -> dict[str, Any]:
    row = validate_arm_receipt(receipt, verify_artifact=True)
    path = _plain_file(receipt_path, label="arm receipt")
    observed = file_sha256(path)
    if observed != _sha256(receipt_file_sha256, label="arm receipt file SHA-256"):
        raise STARCMaterializationError("arm receipt file hash differs")
    artifact = row["artifact"]
    return {
        "episode_id": row["episode_id"],
        "split": row["split"],
        "role": row["role"],
        "label": row["label"],
        "receipt_path": str(path),
        "receipt_file_sha256": observed,
        "receipt_digest": row["receipt_digest"],
        "artifact_path": artifact["path"],
        "artifact_file_sha256": artifact["file_sha256"],
        "artifact_tensor_sha256": artifact["tensor_sha256"],
    }


def make_group_receipt(
    *,
    group_id: str,
    root_spec_binding: Mapping[str, Any],
    bank_binding: Mapping[str, Any],
    detached_event_label_binding: Mapping[str, Any],
    critic_use_binding: Mapping[str, Any],
    model_binding: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    spatial_sketch_bindings_by_episode: Mapping[str, Mapping[str, Any]],
    episode_order: Sequence[str],
    episode_splits: Mapping[str, str],
    arm_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": GROUP_SCHEMA,
        "group_id": group_id,
        "root_spec_binding": dict(root_spec_binding),
        "bank_binding": dict(bank_binding),
        "detached_event_label_binding": dict(detached_event_label_binding),
        "critic_use_binding": dict(critic_use_binding),
        "model_binding": dict(model_binding),
        "runtime_binding": dict(runtime_binding),
        "spatial_sketch_bindings_by_episode": {
            episode: dict(binding)
            for episode, binding in spatial_sketch_bindings_by_episode.items()
        },
        "episode_order": list(episode_order),
        "episode_splits": dict(episode_splits),
        "arm_order": list(ARM_ORDER),
        "arm_bindings": [dict(row) for row in arm_bindings],
        "candidate_count": 20,
        "episode_count": CELLS_PER_GROUP,
        "arm_count": ARMS_PER_GROUP,
        "tensor_artifact_count": ARMS_PER_GROUP,
        "model_forward_count": MODEL_FORWARDS_PER_GROUP,
        "training_performed": False,
        "optimizer_authorized": False,
        "editor_optimizer_authorized": False,
        "scientific_critic_claim_authorized": False,
        "generated_media_editor_use_authorized": False,
    }
    result = _seal(unsigned)
    validate_group_receipt(result, verify_children=False)
    return result


def validate_group_receipt(value: Any, *, verify_children: bool = True) -> dict[str, Any]:
    row = _exact_fields(value, _GROUP_FIELDS, label="STARC group receipt")
    row = _verify_seal(row, schema=GROUP_SCHEMA, label="STARC group receipt")
    episode_order = row["episode_order"]
    splits = row["episode_splits"]
    bindings = row["arm_bindings"]
    sketches = row["spatial_sketch_bindings_by_episode"]
    if (
        row["group_id"] not in GROUP_ORDER
        or not isinstance(episode_order, list)
        or len(episode_order) != CELLS_PER_GROUP
        or len(set(episode_order)) != CELLS_PER_GROUP
        or not isinstance(splits, Mapping)
        or set(splits) != set(episode_order)
        or sorted(splits.values()) != ["confirmation", "fit"]
        or row["arm_order"] != list(ARM_ORDER)
        or not isinstance(bindings, list)
        or len(bindings) != ARMS_PER_GROUP
        or row["candidate_count"] != 20
        or row["episode_count"] != CELLS_PER_GROUP
        or row["arm_count"] != ARMS_PER_GROUP
        or row["tensor_artifact_count"] != ARMS_PER_GROUP
        or row["model_forward_count"] != MODEL_FORWARDS_PER_GROUP
        or not isinstance(sketches, Mapping)
        or set(sketches) != set(episode_order)
    ):
        raise STARCMaterializationError("STARC group topology differs")
    _validate_denials(row, label="STARC group")
    checked_sketches = {
        episode: validate_spatial_sketch_binding(sketches[episode])
        for episode in episode_order
    }
    expected_pairs = [(episode, role) for episode in episode_order for role in ARM_ORDER]
    observed_pairs: list[tuple[str, str]] = []
    receipt_paths: set[str] = set()
    artifact_paths: set[str] = set()
    for raw in bindings:
        binding = _exact_fields(raw, _ARM_BINDING_FIELDS, label="group arm binding")
        observed_pairs.append((binding["episode_id"], binding["role"]))
        if (
            binding["split"] != splits.get(binding["episode_id"])
            or binding["label"] != (1 if binding["role"] == "positive" else 0)
        ):
            raise STARCMaterializationError("group arm label/split differs")
        for name in (
            "receipt_file_sha256",
            "receipt_digest",
            "artifact_file_sha256",
            "artifact_tensor_sha256",
        ):
            _sha256(binding[name], label=f"group arm {name}")
        receipt_path = _absolute_path_text(binding["receipt_path"], label="arm receipt path")
        artifact_path = _absolute_path_text(binding["artifact_path"], label="arm artifact path")
        if receipt_path in receipt_paths or artifact_path in artifact_paths:
            raise STARCMaterializationError("group child paths repeat")
        receipt_paths.add(receipt_path)
        artifact_paths.add(artifact_path)
        if verify_children:
            child, child_path, child_sha = _strict_json_file(
                receipt_path,
                expected_sha256=binding["receipt_file_sha256"],
                label="bound STARC arm receipt",
            )
            checked = validate_arm_receipt(child, verify_artifact=True)
            if (
                child_path != Path(receipt_path)
                or child_sha != binding["receipt_file_sha256"]
                or checked["receipt_digest"] != binding["receipt_digest"]
                or checked["group_id"] != row["group_id"]
                or checked["episode_id"] != binding["episode_id"]
                or checked["split"] != binding["split"]
                or checked["role"] != binding["role"]
                or checked["label"] != binding["label"]
                or checked["artifact"]["path"] != binding["artifact_path"]
                or checked["artifact"]["file_sha256"]
                != binding["artifact_file_sha256"]
                or checked["artifact"]["tensor_sha256"]
                != binding["artifact_tensor_sha256"]
                or checked["spatial_sketch_binding"]
                != checked_sketches[binding["episode_id"]]
            ):
                raise STARCMaterializationError("group-to-arm child binding differs")
    if observed_pairs != expected_pairs:
        raise STARCMaterializationError("group arm order differs")
    return row


def make_master_receipt(group_manifests: Sequence[tuple[Mapping[str, Any], Path, str]]) -> dict[str, Any]:
    """Aggregate exactly two independently materialized SP4 group receipts."""

    if len(group_manifests) != len(GROUP_ORDER):
        raise STARCMaterializationError("master requires exactly two group manifests")
    checked_groups: list[tuple[dict[str, Any], Path, str]] = []
    for expected_group, (raw, path, file_digest) in zip(GROUP_ORDER, group_manifests):
        checked = validate_group_receipt(raw, verify_children=True)
        resolved = _plain_file(path, label=f"{expected_group} group manifest")
        observed = file_sha256(resolved)
        if (
            checked["group_id"] != expected_group
            or observed != _sha256(file_digest, label="group manifest file SHA-256")
        ):
            raise STARCMaterializationError("master group binding differs")
        checked_groups.append((checked, resolved, observed))
    common_fields = (
        "root_spec_binding",
        "bank_binding",
        "detached_event_label_binding",
        "critic_use_binding",
        "model_binding",
        "runtime_binding",
    )
    for name in common_fields:
        if checked_groups[0][0][name] != checked_groups[1][0][name]:
            raise STARCMaterializationError(f"two groups disagree on {name}")
    episode_order = [
        episode
        for group, _path, _sha in checked_groups
        for episode in group["episode_order"]
    ]
    splits = {
        episode: split
        for group, _path, _sha in checked_groups
        for episode, split in group["episode_splits"].items()
    }
    sketches = {
        episode: dict(binding)
        for group, _path, _sha in checked_groups
        for episode, binding in group[
            "spatial_sketch_bindings_by_episode"
        ].items()
    }
    if (
        len(episode_order) != CORE4_CELL_COUNT
        or len(set(episode_order)) != CORE4_CELL_COUNT
        or set(splits) != set(episode_order)
        or set(sketches) != set(episode_order)
    ):
        raise STARCMaterializationError("master episode closure differs")
    group_bindings = [
        {
            "group_id": group["group_id"],
            "manifest_path": str(path),
            "manifest_file_sha256": digest,
            "receipt_digest": group["receipt_digest"],
            "episode_order": list(group["episode_order"]),
            "episode_splits": dict(group["episode_splits"]),
            "arm_count": group["arm_count"],
            "model_forward_count": group["model_forward_count"],
        }
        for group, path, digest in checked_groups
    ]
    first = checked_groups[0][0]
    unsigned = {
        "schema_version": MASTER_SCHEMA,
        **{name: first[name] for name in common_fields},
        "spatial_sketch_bindings_by_episode": sketches,
        "group_order": list(GROUP_ORDER),
        "group_bindings": group_bindings,
        "episode_order": episode_order,
        "episode_splits": splits,
        "arm_order": list(ARM_ORDER),
        "candidate_count": CORE4_CANDIDATE_COUNT,
        "episode_count": CORE4_CELL_COUNT,
        "arm_count": CORE4_ARM_COUNT,
        "tensor_artifact_count": CORE4_ARM_COUNT,
        "model_forward_count": MODEL_FORWARDS_TOTAL,
        "fit_episode_count": 2,
        "confirmation_episode_count": 2,
        "confirmation_consumed_by_optimizer": False,
        "training_performed": False,
        "optimizer_authorized": False,
        "editor_optimizer_authorized": False,
        "scientific_critic_claim_authorized": False,
        "generated_media_editor_use_authorized": False,
    }
    result = _seal(unsigned)
    validate_master_receipt(result, verify_groups=False)
    return result


def validate_master_receipt(value: Any, *, verify_groups: bool = True) -> dict[str, Any]:
    row = _exact_fields(value, _MASTER_FIELDS, label="STARC master receipt")
    row = _verify_seal(row, schema=MASTER_SCHEMA, label="STARC master receipt")
    groups = row["group_bindings"]
    episodes = row["episode_order"]
    splits = row["episode_splits"]
    sketches = row["spatial_sketch_bindings_by_episode"]
    if (
        row["group_order"] != list(GROUP_ORDER)
        or not isinstance(groups, list)
        or len(groups) != 2
        or not isinstance(episodes, list)
        or len(episodes) != CORE4_CELL_COUNT
        or len(set(episodes)) != CORE4_CELL_COUNT
        or not isinstance(splits, Mapping)
        or set(splits) != set(episodes)
        or not isinstance(sketches, Mapping)
        or set(sketches) != set(episodes)
        or sorted(splits.values()) != ["confirmation", "confirmation", "fit", "fit"]
        or row["arm_order"] != list(ARM_ORDER)
        or row["candidate_count"] != CORE4_CANDIDATE_COUNT
        or row["episode_count"] != CORE4_CELL_COUNT
        or row["arm_count"] != CORE4_ARM_COUNT
        or row["tensor_artifact_count"] != CORE4_ARM_COUNT
        or row["model_forward_count"] != MODEL_FORWARDS_TOTAL
        or row["fit_episode_count"] != 2
        or row["confirmation_episode_count"] != 2
        or row["confirmation_consumed_by_optimizer"] is not False
    ):
        raise STARCMaterializationError("STARC master topology differs")
    _validate_denials(row, label="STARC master")
    checked_sketches = {
        episode: validate_spatial_sketch_binding(sketches[episode])
        for episode in episodes
    }
    flattened_episodes: list[str] = []
    for expected_group, raw in zip(GROUP_ORDER, groups):
        binding = _exact_fields(raw, _GROUP_BINDING_FIELDS, label="master group binding")
        if (
            binding["group_id"] != expected_group
            or binding["arm_count"] != ARMS_PER_GROUP
            or binding["model_forward_count"] != MODEL_FORWARDS_PER_GROUP
            or not isinstance(binding["episode_order"], list)
            or not isinstance(binding["episode_splits"], Mapping)
            or set(binding["episode_splits"]) != set(binding["episode_order"])
        ):
            raise STARCMaterializationError("master group topology differs")
        flattened_episodes.extend(binding["episode_order"])
        path = _absolute_path_text(binding["manifest_path"], label="group manifest path")
        for name in ("manifest_file_sha256", "receipt_digest"):
            _sha256(binding[name], label=f"master group {name}")
        if verify_groups:
            child, child_path, child_sha = _strict_json_file(
                path,
                expected_sha256=binding["manifest_file_sha256"],
                label="bound STARC group manifest",
            )
            checked = validate_group_receipt(child, verify_children=True)
            if (
                str(child_path) != path
                or child_sha != binding["manifest_file_sha256"]
                or checked["receipt_digest"] != binding["receipt_digest"]
                or checked["episode_order"] != binding["episode_order"]
                or checked["episode_splits"] != binding["episode_splits"]
                or checked["spatial_sketch_bindings_by_episode"]
                != {
                    episode: checked_sketches[episode]
                    for episode in binding["episode_order"]
                }
            ):
                raise STARCMaterializationError("master-to-group child binding differs")
    if flattened_episodes != episodes:
        raise STARCMaterializationError("master episode order differs from groups")
    return row


def make_required_critic_use_sidecar(*, authority_evidence: str | Path) -> dict[str, Any]:
    evidence = _plain_file(authority_evidence, label="critic-use authority evidence")
    if file_sha256(evidence) != REQUIRED_CRITIC_USE_EVIDENCE_SHA256:
        raise STARCMaterializationError("critic-use authority evidence SHA-256 differs")
    value = dataset_contract.make_critic_usage_authority(
        bank_receipt_digest=REQUIRED_BANK_RECEIPT_DIGEST,
        authorization_source=REQUIRED_CRITIC_USE_SOURCE,
        authorization_evidence_sha256=REQUIRED_CRITIC_USE_EVIDENCE_SHA256,
    )
    return dataset_contract.validate_critic_usage_authority(value)


def load_required_critic_use_sidecar(
    value: str | Path, *, expected_sha256: str
) -> tuple[dict[str, Any], Path, str]:
    decoded, path, observed = _strict_json_file(
        value, expected_sha256=expected_sha256, label="critic-use sidecar"
    )
    try:
        checked = dataset_contract.validate_critic_usage_authority(decoded)
    except dataset_contract.LatentTemporalEventDatasetError as error:
        raise STARCMaterializationError(str(error)) from error
    if (
        checked["bank_receipt_digest"] != REQUIRED_BANK_RECEIPT_DIGEST
        or checked["authorization_source"] != REQUIRED_CRITIC_USE_SOURCE
        or checked["authorization_evidence_sha256"]
        != REQUIRED_CRITIC_USE_EVIDENCE_SHA256
        or checked["authorized_use"] != dataset_contract.CRITIC_ONLY_USE
    ):
        raise STARCMaterializationError("critic-use sidecar is not the core4 authority")
    return checked, path, observed


def _snapshot_tensors(values: Mapping[str, Any]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in values.items()}


def forward_same_state_hidden_pair(
    *,
    diffusion: Any,
    transformer: Any,
    observer: Block15SpatialSketchObserver,
    x_sigma: Any,
    action_condition: Any,
    noop_condition: Any,
    arm_key: str,
    dist_module: Any = None,
    group: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run two frozen prompts over one patched state/timestep/rotary object set."""

    import torch
    import dclr_runtime_contract as runtime_contract
    import pair_v5_native_bridge as native_bridge

    dist = torch.distributed if dist_module is None else dist_module
    try:
        native_bridge._validate_exact81_spatial(
            x_sigma, label="STARC x_sigma", detached_fp32=True
        )
    except native_bridge.PairV5NativeBridgeError as error:
        raise STARCMaterializationError(str(error)) from error
    x_shape, patch_height, patch_width, patch_positions = latent_geometry(
        x_sigma,
        allowed_latent_shapes=observer.allowed_latent_shapes,
        allowed_patch_grids=observer.allowed_patch_grids,
        geometry_label=observer.geometry_label,
    )
    if (
        observer.layout.patch_height != patch_height
        or observer.layout.patch_width != patch_width
        or observer.layout.patch_positions != patch_positions
    ):
        raise STARCMaterializationError("observer/query geometry differs")
    if (
        not callable(getattr(diffusion, "shared_step", None))
        or not callable(getattr(transformer, "patch_vae_latent", None))
        or any(parameter.requires_grad for parameter in diffusion.parameters())
        or any(parameter.requires_grad for parameter in transformer.parameters())
    ):
        raise STARCMaterializationError("STARC Bernini runtime is not frozen")
    dtype = getattr(transformer, "dtype", None)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise STARCMaterializationError("STARC transformer dtype differs")
    for name, condition in (
        ("action", action_condition),
        ("noop", noop_condition),
    ):
        if (
            not isinstance(condition, torch.Tensor)
            or tuple(int(item) for item in condition.shape) != (1, 512, 4096)
            or condition.device != x_sigma.device
            or condition.requires_grad
            or condition.grad_fn is not None
            or not bool(torch.isfinite(condition).all().item())
        ):
            raise STARCMaterializationError(f"{name} condition closure differs")
    if torch.equal(action_condition, noop_condition):
        raise STARCMaterializationError("action/no-op prompt conditions alias")

    # Patch once.  Both text queries consume this exact branch's tensor and
    # rotary objects; merely recreating equal tensors would not satisfy STARC.
    with torch.inference_mode():
        patched = transformer.patch_vae_latent(
            x_sigma.to(dtype=dtype), source_id=native_bridge.T2V_TARGET_SOURCE_ID
        )
    if not isinstance(patched, (tuple, list)) or len(patched) != 2:
        raise STARCMaterializationError("patch_vae_latent output differs")
    try:
        branch = runtime_contract.build_t2v_target_branch(
            patched[0], patched[1], target_source_id=native_bridge.T2V_TARGET_SOURCE_ID
        )
    except runtime_contract.DCLRRuntimeContractError as error:
        raise STARCMaterializationError(str(error)) from error
    if branch.target_token_count != LATENT_PHASES * patch_positions:
        raise STARCMaterializationError("patched target token geometry differs")
    timestep = torch.tensor(
        [float(NATIVE_TIMESTEP)], dtype=torch.float32, device=x_sigma.device
    )
    tracked = {
        "x_sigma": x_sigma,
        "noisy_latents": branch.noisy_latents,
        "rotary_embs": branch.rotary_embs,
        "native_timestep": timestep,
    }
    object_ids = {name: id(value) for name, value in tracked.items()}
    hashes_by_stage = {"before_action": _snapshot_tensors(tracked)}
    captures: dict[str, tuple[LocalBlock15Sketch, GlobalBlock15Sketch]] = {}
    prediction_hashes: dict[str, str] = {}
    for prompt_name, condition in zip(PROMPT_ORDER, (action_condition, noop_condition)):
        with observer.capture(f"{arm_key}:{prompt_name}") as holder:
            with torch.inference_mode():
                prediction = diffusion.shared_step(
                    model_id="transformer_1",
                    noisy_latents=branch.noisy_latents,
                    timesteps=timestep,
                    cond_embeds=condition,
                    rotary_embs=branch.rotary_embs,
                    batch_vae_seqlen=list(branch.batch_vae_seqlen),
                    batch_text_seqlen=[runtime_contract.PINNED_TEXT_TOKENS],
                )
        if len(holder) != 1:
            raise STARCMaterializationError("block-15 hook result closure differs")
        total = branch.total_token_count
        if (
            not isinstance(prediction, torch.Tensor)
            or tuple(int(item) for item in prediction.shape)
            != (1, total, runtime_contract.PINNED_PATCH_DIM)
            or prediction.device != x_sigma.device
            or prediction.requires_grad
            or prediction.grad_fn is not None
            or not bool(torch.isfinite(prediction).all().item())
        ):
            raise STARCMaterializationError("frozen shared_step output closure differs")
        local = holder[0]
        global_capture = all_reduce_block15_sketch(
            local, dist_module=dist, group=group
        )
        captures[prompt_name] = (local, global_capture)
        prediction_hashes[prompt_name] = tensor_sha256(prediction)
        stage = "after_action" if prompt_name == "target_action" else "after_noop"
        hashes_by_stage[stage] = _snapshot_tensors(tracked)
        if any(id(tracked[name]) != object_ids[name] for name in tracked):
            raise STARCMaterializationError(
                "same-state x/noisy/rotary/timestep object identity changed"
            )
        del prediction
    if any(
        len({hashes_by_stage[stage][name] for stage in hashes_by_stage}) != 1
        for name in tracked
    ):
        raise STARCMaterializationError("same-state tensor bytes changed across prompts")

    action_local, action_global = captures["target_action"]
    noop_local, noop_global = captures["noop"]
    local_parity = {
        "block.00.input": (
            action_local.block0_input_value_sha256
            == noop_local.block0_input_value_sha256
        ),
        "block.00.attn1": (
            action_local.block0_attn1_value_sha256
            == noop_local.block0_attn1_value_sha256
        ),
    }
    parity_rows: list[Any] = [None] * EXPECTED_SP_WORLD
    dist.all_gather_object(
        parity_rows,
        {
            "rank": action_local.layout.sp_rank,
            "block0_input_action": action_local.block0_input_value_sha256,
            "block0_input_noop": noop_local.block0_input_value_sha256,
            "block0_attn1_action": action_local.block0_attn1_value_sha256,
            "block0_attn1_noop": noop_local.block0_attn1_value_sha256,
            "parity": local_parity,
        },
        group=group,
    )
    if (
        [row.get("rank") for row in parity_rows] != list(range(EXPECTED_SP_WORLD))
        or not all(row.get("parity") == {"block.00.input": True, "block.00.attn1": True} for row in parity_rows)
    ):
        raise STARCMaterializationError("same-state block-0 exact parity failed")

    residual = (action_global.sketch - noop_global.sketch).float().contiguous().detach()
    if (
        tuple(int(item) for item in residual.shape)
        != (1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE)
        or residual.requires_grad
        or residual.grad_fn is not None
        or not bool(torch.isfinite(residual).all().item())
    ):
        raise STARCMaterializationError("STARC hidden residual closure differs")
    residual_digest = tensor_sha256(residual)
    digest_rows: list[Any] = [None] * EXPECTED_SP_WORLD
    dist.all_gather_object(digest_rows, residual_digest, group=group)
    if len(set(digest_rows)) != 1:
        raise STARCMaterializationError("SP4 residual tensor values differ by rank")

    proof = {
        "native_schedule_index": SCHEDULE_INDEX,
        "sigma": SIGMA,
        "native_timestep": NATIVE_TIMESTEP,
        "x_sigma_tensor_sha256": hashes_by_stage["before_action"]["x_sigma"],
        "noisy_latents_tensor_sha256": hashes_by_stage["before_action"]["noisy_latents"],
        "rotary_embs_tensor_sha256": hashes_by_stage["before_action"]["rotary_embs"],
        "native_timestep_tensor_sha256": hashes_by_stage["before_action"]["native_timestep"],
        "tensor_sha256_by_stage": hashes_by_stage,
        "action_and_noop_share_exact_x_sigma_object": True,
        "action_and_noop_share_exact_noisy_latents_object": True,
        "action_and_noop_share_exact_rotary_object": True,
        "action_and_noop_share_exact_timestep_object": True,
        "shared_tensor_bytes_unchanged": True,
        "block0_input_and_attn1_exact_parity": True,
        "block0_parity_by_rank": parity_rows,
        "source_condition_consumed": False,
        "mask_flow_pose_track_or_trajectory_consumed": False,
        "event_labels_consumed": False,
    }
    hidden = {
        "hook_coordinate": HOOK_COORDINATE,
        "ulysses_world": EXPECTED_SP_WORLD,
        "latent_shape": list(x_shape),
        "patch_positions": patch_positions,
        "patch_grid_height_width": [patch_height, patch_width],
        "patch_flatten_order": "patch-y-x",
        "phase_patch_count": list(action_global.phase_patch_count),
        "action_local_hidden_value_sha256_by_rank": list(
            action_global.local_hidden_value_sha256_by_rank
        ),
        "noop_local_hidden_value_sha256_by_rank": list(
            noop_global.local_hidden_value_sha256_by_rank
        ),
        "action_global_hidden_composite_sha256": action_global.global_hidden_composite_sha256,
        "noop_global_hidden_composite_sha256": noop_global.global_hidden_composite_sha256,
        "action_global_sketch_shape": [1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE],
        "noop_global_sketch_shape": [1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE],
        "action_global_sketch_tensor_sha256": action_global.sketch_tensor_sha256,
        "noop_global_sketch_tensor_sha256": noop_global.sketch_tensor_sha256,
        "action_prediction_tensor_sha256": prediction_hashes["target_action"],
        "noop_prediction_tensor_sha256": prediction_hashes["noop"],
        "action_noop_hidden_composites_distinct": (
            action_global.global_hidden_composite_sha256
            != noop_global.global_hidden_composite_sha256
        ),
        "residual_shape": [1, LATENT_PHASES, SKETCH_COORDINATES, HIDDEN_SIZE],
        "residual_dtype": "torch.float32",
        "residual_tensor_sha256": residual_digest,
        "residual_l2_norm": float(torch.linalg.vector_norm(residual).item()),
        "rank_local_sketch_then_all_reduce": True,
        "full_hidden_persisted": False,
    }
    return residual, proof, hidden


def _encode_prompt_pair(
    renderer: Any,
    tokenizer: Any,
    *,
    action_caption: str,
    noop_caption: str,
    device: Any,
    frozen: Any,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    native_generation = frozen.native_generation
    action_prompt = native_generation.build_task_prompt(
        "t2v", action_caption, prompt_cleaner=prompt_clean
    )
    noop_prompt = native_generation.build_task_prompt(
        "t2v", noop_caption, prompt_cleaner=prompt_clean
    )
    conditions, hashes = temporal_scorer._encode_prompt_pair(
        renderer,
        tokenizer,
        action_prompt=action_prompt,
        noop_prompt=noop_prompt,
        device=device,
        frozen=frozen,
    )
    return conditions, hashes, {
        "action_prompt": action_prompt,
        "noop_prompt": noop_prompt,
    }


def _candidate_evidence(
    *,
    bound: Mapping[str, Any],
    label: Mapping[str, Any],
    clean: Any,
    gaussian_tensor_sha256: str,
    bank_receipt_digest: str,
) -> dict[str, Any]:
    candidate = bound["candidate"]
    artifact = bound["artifacts"]["predecode_clean_latent"]
    clean_shape, _patch_height, _patch_width, _patch_positions = latent_geometry(clean)
    return {
        "candidate_id": candidate["candidate_id"],
        "bank_receipt_digest": bank_receipt_digest,
        "cell_id": candidate["calibration_group_id"],
        "analysis_split": candidate["analysis_split"],
        "action_family_id": candidate["action_family_id"],
        "actor_group_id": candidate["actor_group_id"],
        "scene_group_id": candidate["scene_group_id"],
        "action_group_id": candidate["action_group_id"],
        "seed": candidate["seed"],
        "official_gaussian_tensor_sha256": gaussian_tensor_sha256,
        "semantic_branch": candidate["semantic_branch"],
        "full_t2v_caption": candidate["full_t2v_caption"],
        "full_t2v_caption_utf8_sha256": candidate["full_t2v_caption_utf8_sha256"],
        "clean_latent_artifact_path": artifact["path"],
        "clean_latent_artifact_sha256": artifact["sha256"],
        "clean_latent_tensor_sha256": tensor_sha256(clean),
        "clean_latent_shape": list(clean_shape),
        "generation_receipt_digest": bound["generation_receipt_digest"],
        "event_audit_artifact_sha256": label["external_audit_artifact_sha256"],
        "complete_target_transition_observed": label[
            "complete_target_transition_observed"
        ],
        "terminal_hold_observed": label["terminal_hold_observed"],
        "full_target_action_observed": label["full_target_action_observed"],
        "full_target_action_false_confirmed": label[
            "full_target_action_false_confirmed"
        ],
    }


def _rank0_action(
    *, dist: Any, rank: int, action: Callable[[], Any], label: str
) -> Any:
    """Run filesystem mutation on rank zero and broadcast success/failure."""

    payload: list[Any] = [None]
    if rank == 0:
        try:
            payload[0] = {"ok": True, "value": action()}
        except BaseException as error:  # propagate instead of hanging at next collective
            payload[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        raise STARCMaterializationError(f"rank-zero {label} failed: {result}")
    return result.get("value")


def _validate_source_hash(path: Path, expected: str, *, label: str) -> str:
    observed = file_sha256(_plain_file(path, label=label))
    if observed != _sha256(expected, label=f"{label} expected SHA-256"):
        raise STARCMaterializationError(f"{label} source SHA-256 differs")
    return observed


def _validate_materialize_cli(args: argparse.Namespace) -> None:
    fixed_hashes = {
        "expected_root_spec_sha256": REQUIRED_ROOT_SPEC_RAW_SHA256,
        "expected_bank_receipt_sha256": REQUIRED_BANK_RECEIPT_FILE_SHA256,
        "expected_detached_label_manifest_sha256": REQUIRED_DETACHED_LABEL_FILE_SHA256,
    }
    for name, required in fixed_hashes.items():
        observed = _sha256(getattr(args, name), label=name)
        if observed != required:
            raise STARCMaterializationError(f"{name} is not the sealed core4 authority")
    for name in (
        "expected_critic_use_sidecar_sha256",
        "method_source_archive_sha256",
        "formal_d541801_source_archive_sha256",
        "expected_materializer_source_sha256",
        "expected_formal_d541801_scorer_source_sha256",
        "expected_temporal_scorer_source_sha256",
        "expected_temporal_contract_source_sha256",
        "expected_fitq_observer_source_sha256",
        "expected_dataset_contract_source_sha256",
        "expected_label_author_source_sha256",
    ):
        _sha256(getattr(args, name), label=name)
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
        "formal_d541801_source_revision",
    ):
        _sha1(getattr(args, name), label=name)
    if (
        args.expected_bernini_commit != temporal_contract.REQUIRED_BERNINI_REVISION
        or args.expected_veomni_commit != temporal_contract.REQUIRED_VEOMNI_REVISION
    ):
        raise STARCMaterializationError("Bernini/VeOmni revisions differ")
    if (
        args.formal_d541801_source_revision != REQUIRED_D541801_SOURCE_REVISION
        or args.formal_d541801_source_archive_sha256
        != REQUIRED_D541801_SOURCE_ARCHIVE_SHA256
        or args.expected_formal_d541801_scorer_source_sha256
        != REQUIRED_D541801_SCORER_SOURCE_SHA256
    ):
        raise STARCMaterializationError("formal d541801 label-verifier authority differs")
    for name in (
        "ack_generated_t2v_hidden_critic_only",
        "ack_no_generated_media_editor_use",
        "ack_no_optimizer_or_editor_update",
    ):
        if getattr(args, name) is not True:
            raise STARCMaterializationError(f"mandatory acknowledgement missing: {name}")
    source_hashes = (
        (Path(__file__).resolve(), args.expected_materializer_source_sha256, "materializer"),
        (
            METHOD_ROOT / "score_pair_v5_t2v_energy_bank_v3.py",
            args.expected_formal_d541801_scorer_source_sha256,
            "formal d541801 scorer",
        ),
        (
            METHOD_ROOT / "temporal_counterfactual_action_scorer_v1.py",
            args.expected_temporal_scorer_source_sha256,
            "temporal scorer",
        ),
        (
            METHOD_ROOT / "temporal_counterfactual_contract_v1.py",
            args.expected_temporal_contract_source_sha256,
            "temporal contract",
        ),
        (
            METHOD_ROOT / "internal_temporal_quotient_observer.py",
            args.expected_fitq_observer_source_sha256,
            "FITQ observer",
        ),
        (
            METHOD_ROOT / "latent_temporal_event_critic_dataset.py",
            args.expected_dataset_contract_source_sha256,
            "STARC dataset contract",
        ),
        (
            TOOLS_ROOT / "author_pair_v5_core4_event_labels_d541801_v3.py",
            args.expected_label_author_source_sha256,
            "detached label author",
        ),
    )
    for path, expected, label in source_hashes:
        _validate_source_hash(path, expected, label=label)
    evidence = _plain_file(
        args.critic_use_authority_evidence, label="critic-use authority evidence"
    )
    if file_sha256(evidence) != REQUIRED_CRITIC_USE_EVIDENCE_SHA256:
        raise STARCMaterializationError("critic-use authority evidence differs")
    output_root = _plain_directory(args.output_root, label="output root")
    group_output = output_root / args.group_id
    if group_output.exists() or group_output.is_symlink():
        raise STARCMaterializationError("group output must not already exist")


def _source_runtime_binding(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import diffusers
    import transformers

    return {
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "formal_d541801_source_revision": args.formal_d541801_source_revision,
        "formal_d541801_source_archive_sha256": args.formal_d541801_source_archive_sha256,
        "formal_d541801_scorer_source_sha256": args.expected_formal_d541801_scorer_source_sha256,
        "materializer_source_sha256": args.expected_materializer_source_sha256,
        "temporal_scorer_source_sha256": args.expected_temporal_scorer_source_sha256,
        "temporal_contract_source_sha256": args.expected_temporal_contract_source_sha256,
        "fitq_observer_source_sha256": args.expected_fitq_observer_source_sha256,
        "dataset_contract_source_sha256": args.expected_dataset_contract_source_sha256,
        "label_author_source_sha256": args.expected_label_author_source_sha256,
        "torch": str(torch.__version__),
        "torch_hip": str(getattr(torch.version, "hip", None)),
        "transformers": str(transformers.__version__),
        "diffusers": str(diffusers.__version__),
        "ulysses_world": EXPECTED_SP_WORLD,
        "materialization_only": True,
        "optimizer_constructed": False,
        "editor_forward_performed": False,
    }


def _artifact_native_binding(
    artifact: Mapping[str, Any],
    *,
    tensor_digest: str,
    authenticated_identity: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "path": artifact["path"],
        "file_sha256": artifact["sha256"],
        "tensor_key": artifact.get("tensor_key"),
        "stored_dtype": artifact.get("stored_dtype"),
        "shape": artifact.get("shape"),
        "raw_value_sha256": (
            authenticated_identity.get("raw_value_sha256")
            if authenticated_identity is not None
            else artifact.get("raw_value_sha256")
        ),
        "content_sha256": (
            authenticated_identity.get("content_sha256")
            if authenticated_identity is not None
            else artifact.get("content_sha256")
        ),
        "tensor_sha256": tensor_digest,
    }


def materialize_group(args: argparse.Namespace) -> int:
    _validate_materialize_cli(args)
    frozen = temporal_scorer._frozen_d541801_runtime()
    temporal_scorer.validate_native_coordinate_runtime(frozen)
    coordinate = next(
        (
            row
            for row in temporal_contract.NATIVE_SIGMA_COORDINATES
            if row[0] == SCHEDULE_INDEX
        ),
        None,
    )
    if coordinate is None or float(coordinate[1]).hex() != float(SIGMA).hex() or coordinate[2] != NATIVE_TIMESTEP:
        raise STARCMaterializationError("registered schedule-33 coordinate differs")

    native_generation = frozen.native_generation
    legacy = native_generation.legacy
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
        raise STARCMaterializationError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise STARCMaterializationError("pinned Bernini head count differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != EXPECTED_SP_WORLD
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise STARCMaterializationError("materializer requires one AUH ROCm SP4 group")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=EXPECTED_SP_WORLD)
    device = torch.device("cuda", distributed.local_rank)
    observer: Optional[Block15SpatialSketchObserver] = None
    try:
        # Authenticate every candidate and detached label row before any model
        # forward.  The label author delegates tensor/media hashes to the same
        # native bank verifier as the frozen temporal scorer.
        try:
            spec, bank, bound_all = label_author.load_core4_bound_bank(
                root_spec=args.root_spec,
                root_spec_sha256=args.expected_root_spec_sha256,
                bank_output_dir=args.bank_output_dir,
                bank_receipt=args.bank_receipt,
                bank_receipt_sha256=args.expected_bank_receipt_sha256,
            )
        except label_author.PairV5Core4LabelAuthoringError as error:
            raise STARCMaterializationError(str(error)) from error
        if (
            bank.get("receipt_digest") != REQUIRED_BANK_RECEIPT_DIGEST
            or bank.get("file_sha256") != REQUIRED_BANK_RECEIPT_FILE_SHA256
            or len(bound_all) != CORE4_CANDIDATE_COUNT
        ):
            raise STARCMaterializationError("core4 bank authority differs")
        try:
            label_manifest, label_path, label_file_sha = label_author.load_label_manifest(
                args.detached_label_manifest,
                expected_sha256=args.expected_detached_label_manifest_sha256,
                root_spec_raw_sha256=args.expected_root_spec_sha256,
                bank_receipt_digest=bank["receipt_digest"],
                bound_rows=bound_all,
            )
        except label_author.PairV5Core4LabelAuthoringError as error:
            raise STARCMaterializationError(str(error)) from error
        if label_file_sha != REQUIRED_DETACHED_LABEL_FILE_SHA256:
            raise STARCMaterializationError("detached label file authority differs")
        sidecar, sidecar_path, sidecar_file_sha = load_required_critic_use_sidecar(
            args.critic_use_sidecar,
            expected_sha256=args.expected_critic_use_sidecar_sha256,
        )
        if _plain_file(args.critic_use_authority_evidence, label="authority evidence") is None:
            raise AssertionError("unreachable")

        checkpoint_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                identity = native_generation.source_audit.validate_checkpoint_content(
                    checkpoint, Path(args.checkpoint_content_manifest)
                )
                checkpoint_rows[0] = {"ok": True, "identity": identity}
            except Exception as error:
                checkpoint_rows[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(checkpoint_rows, src=0)
        checkpoint_result = checkpoint_rows[0]
        if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get("ok") is not True:
            raise STARCMaterializationError(
                f"rank-zero checkpoint audit failed: {checkpoint_result}"
            )
        checkpoint_identity = dict(checkpoint_result["identity"])
        checkpoint_receipt_digest = frozen.object_sha256(checkpoint_identity)

        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
        try:
            freeze_before = native_generation.source_audit.model_freeze_certificate(renderer)
            checkpoint_binding = frozen.checkpoint_content_binding(
                checkpoint_identity, freeze_before
            )
        except Exception as error:
            raise STARCMaterializationError(str(error)) from error
        diffusion = renderer.diff_dec
        transformer = diffusion.transformer
        if (
            transformer is None
            or diffusion.transformer_2 is not None
            or any(parameter.requires_grad for parameter in renderer.parameters())
        ):
            raise STARCMaterializationError("frozen transformer_1 closure differs")
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
        )
        builder_contract = frozen.prompt_builder_contract()

        runtime_binding = _source_runtime_binding(args)
        model_binding = {
            "frozen_checkpoint_receipt_digest": checkpoint_receipt_digest,
            "checkpoint_content_binding": checkpoint_binding,
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
            "native_schedule_digest": temporal_contract.NATIVE_SCHEDULE_DIGEST,
            "native_schedule_index": SCHEDULE_INDEX,
            "native_timestep": NATIVE_TIMESTEP,
            "sigma": SIGMA,
            "hook_coordinate": HOOK_COORDINATE,
            "transformer_1_only": True,
            "adapter_loaded": False,
            "all_parameters_frozen": True,
        }
        root_spec_binding = {
            "path": str(_plain_file(args.root_spec, label="root spec")),
            "file_sha256": REQUIRED_ROOT_SPEC_RAW_SHA256,
            "schema_version": spec["schema_version"],
        }
        bank_binding = {
            "path": str(_plain_file(args.bank_receipt, label="bank receipt")),
            "file_sha256": REQUIRED_BANK_RECEIPT_FILE_SHA256,
            "receipt_digest": REQUIRED_BANK_RECEIPT_DIGEST,
            "candidate_count": CORE4_CANDIDATE_COUNT,
        }
        detached_label_binding = {
            "path": str(label_path),
            "file_sha256": label_file_sha,
            "manifest_digest": label_manifest["manifest_digest"],
            "candidate_count": label_manifest["candidate_count"],
            "labels_are_external_and_detached": True,
            "labels_may_enter_model_condition": False,
        }
        critic_use_binding = {
            "path": str(sidecar_path),
            "file_sha256": sidecar_file_sha,
            "receipt_digest": sidecar["receipt_digest"],
            "bank_receipt_digest": sidecar["bank_receipt_digest"],
            "authorized_use": sidecar["authorized_use"],
            "authorization_source": sidecar["authorization_source"],
            "authorization_evidence_sha256": sidecar[
                "authorization_evidence_sha256"
            ],
        }
        bound_group = [row for row in bound_all if row["group_id"] == args.group_id]
        if len(bound_group) != 20:
            raise STARCMaterializationError("group candidate count differs")
        labels_by_id = {row["candidate_id"]: row for row in label_manifest["rows"]}
        by_cell: dict[str, list[Mapping[str, Any]]] = {}
        for row in bound_group:
            by_cell.setdefault(row["candidate"]["calibration_group_id"], []).append(row)
        if len(by_cell) != CELLS_PER_GROUP:
            raise STARCMaterializationError("group cell count differs")
        episode_order = list(by_cell)
        episode_splits = {
            cell: rows[0]["candidate"]["analysis_split"] for cell, rows in by_cell.items()
        }
        group_root = Path(args.output_root) / args.group_id
        _rank0_action(
            dist=dist,
            rank=distributed.rank,
            label="create group output",
            action=lambda: (group_root.mkdir(), str(group_root.resolve(strict=True)))[1],
        )
        arm_bindings: list[dict[str, Any]] = []
        spatial_bindings_by_episode: dict[str, dict[str, Any]] = {}

        for cell_id, cell_rows in by_cell.items():
            if [row["candidate"]["semantic_branch"] for row in cell_rows] != list(
                dataset_contract.SEMANTIC_BRANCHES
            ):
                raise STARCMaterializationError("cell semantic branch order differs")
            cached: dict[str, dict[str, Any]] = {}
            first_gaussian: Any = None
            first_gaussian_sha: Optional[str] = None
            cell_shape: Optional[tuple[int, ...]] = None
            patch_height: Optional[int] = None
            patch_width: Optional[int] = None
            patch_positions: Optional[int] = None
            evidence_rows: list[dict[str, Any]] = []
            for row_index, bound in enumerate(cell_rows):
                candidate = bound["candidate"]
                candidate_id = candidate["candidate_id"]
                clean_artifact = bound["artifacts"]["predecode_clean_latent"]
                gaussian_artifact = bound["artifacts"]["official_initial_gaussian"]
                clean = frozen._load_exact81_tensor(
                    clean_artifact,
                    key="normalized_clean_latent",
                    label=f"{candidate_id} clean latent",
                )
                gaussian = frozen._load_exact81_tensor(
                    gaussian_artifact,
                    key="official_initial_gaussian",
                    label=f"{candidate_id} official Gaussian",
                )
                clean_geometry = latent_geometry(clean)
                gaussian_geometry = latent_geometry(gaussian)
                if clean_geometry != gaussian_geometry:
                    raise STARCMaterializationError("candidate clean/Gaussian geometry differs")
                if row_index == 0:
                    cell_shape, patch_height, patch_width, patch_positions = clean_geometry
                elif clean_geometry != (
                    cell_shape,
                    patch_height,
                    patch_width,
                    patch_positions,
                ):
                    raise STARCMaterializationError("same-cell native geometry differs")
                clean_identity = verify_authenticated_native_clean_tensor_identity(
                    clean,
                    clean_artifact,
                    label=f"{candidate_id} clean latent",
                    frozen=frozen,
                )
                frozen.verify_native_tensor_value_identity(
                    gaussian, gaussian_artifact, label=f"{candidate_id} official Gaussian"
                )
                gaussian_sha = tensor_sha256(gaussian)
                if row_index == 0:
                    first_gaussian = gaussian
                    first_gaussian_sha = gaussian_sha
                elif gaussian_sha != first_gaussian_sha or not torch.equal(
                    gaussian, first_gaussian
                ):
                    raise STARCMaterializationError(
                        "same-cell official Gaussian tensor values differ"
                    )
                label = labels_by_id.get(candidate_id)
                if not isinstance(label, Mapping):
                    raise STARCMaterializationError("candidate detached label is missing")
                cached[candidate_id] = {
                    "bound": bound,
                    "label": dict(label),
                    "clean": clean,
                    "gaussian": gaussian,
                    "clean_tensor_sha256": tensor_sha256(clean),
                    "clean_authentication": clean_identity,
                    "gaussian_tensor_sha256": gaussian_sha,
                }
                evidence_rows.append(
                    _candidate_evidence(
                        bound=bound,
                        label=label,
                        clean=clean,
                        gaussian_tensor_sha256=gaussian_sha,
                        bank_receipt_digest=bank["receipt_digest"],
                    )
                )
            assert (
                first_gaussian is not None
                and first_gaussian_sha is not None
                and cell_shape is not None
                and patch_height is not None
                and patch_width is not None
                and patch_positions is not None
            )
            spatial = fixed_spatial_sketch(
                patch_height=patch_height,
                patch_width=patch_width,
                device=device,
            )
            spatial_binding = sketch_binding(
                spatial, patch_height=patch_height, patch_width=patch_width
            )
            spatial_bindings_by_episode[cell_id] = spatial_binding
            if observer is not None:
                raise STARCMaterializationError("previous cell observer was not removed")
            observer = Block15SpatialSketchObserver(
                transformer,
                sp_rank=distributed.rank,
                patch_height=patch_height,
                patch_width=patch_width,
                spatial_sketch=spatial,
            ).install()
            try:
                episode = dataset_contract.build_episode_plan(
                    evidence_rows, usage_authority=sidecar
                )
            except dataset_contract.LatentTemporalEventDatasetError as error:
                raise STARCMaterializationError(str(error)) from error
            if (
                episode["episode_id"] != cell_id
                or episode["arm_order"] != list(ARM_ORDER)
                or len(episode["arms"]) != ARMS_PER_CELL
            ):
                raise STARCMaterializationError("episode plan closure differs")

            action_bound = cell_rows[0]
            noop_bound = cell_rows[1]
            action_candidate = action_bound["candidate"]
            noop_candidate = noop_bound["candidate"]
            conditions, condition_hashes, prompt_text = _encode_prompt_pair(
                renderer,
                tokenizer,
                action_caption=action_candidate["full_t2v_caption"],
                noop_caption=noop_candidate["full_t2v_caption"],
                device=device,
                frozen=frozen,
            )
            prompt_binding = temporal_scorer._prompt_binding(
                target_action_caption_sha256=action_candidate[
                    "full_t2v_caption_utf8_sha256"
                ],
                target_noop_caption_sha256=noop_candidate[
                    "full_t2v_caption_utf8_sha256"
                ],
                action_prompt=prompt_text["action_prompt"],
                noop_prompt=prompt_text["noop_prompt"],
                condition_hashes=condition_hashes,
                prompt_builder_contract_digest=builder_contract["contract_digest"],
            )
            prompt_binding.update(
                {
                    "target_action_candidate_id": action_candidate["candidate_id"],
                    "target_noop_candidate_id": noop_candidate["candidate_id"],
                    "all_13_arms_use_cell_fixed_prompt_pair": True,
                    "branch_caption_never_used_as_condition": True,
                    "detached_labels_never_used_as_condition": True,
                }
            )
            epsilon = first_gaussian.to(device=device).float().contiguous().detach()
            for arm in episode["arms"]:
                role = arm["role"]
                candidate_id = arm["source_candidate_id"]
                owner = cached[candidate_id]
                bound = owner["bound"]
                label = owner["label"]
                candidate = bound["candidate"]
                transformed = apply_temporal_transform(
                    owner["clean"], arm["temporal_transform"]
                ).to(device=device).contiguous().detach()
                transformed_sha = tensor_sha256(transformed)
                sigma = torch.tensor(SIGMA, dtype=torch.float32, device=device)
                x_sigma = (
                    transformed + sigma.reshape(1, 1, 1, 1, 1) * (epsilon - transformed)
                ).float().contiguous().detach()
                residual, same_state_proof, hidden_binding = forward_same_state_hidden_pair(
                    diffusion=diffusion,
                    transformer=transformer,
                    observer=observer,
                    x_sigma=x_sigma,
                    action_condition=conditions["target_action"],
                    noop_condition=conditions["noop"],
                    arm_key=f"{cell_id}:{role}",
                    dist_module=dist,
                )
                arm_dir = group_root / cell_id / role

                def write_artifact() -> dict[str, Any]:
                    arm_dir.parent.mkdir(exist_ok=True)
                    arm_dir.mkdir()
                    return save_residual_artifact(arm_dir / TENSOR_FILENAME, residual)

                artifact = _rank0_action(
                    dist=dist,
                    rank=distributed.rank,
                    label=f"write {cell_id}/{role} artifact",
                    action=write_artifact,
                )
                if artifact["tensor_sha256"] != hidden_binding["residual_tensor_sha256"]:
                    raise STARCMaterializationError("artifact/residual tensor digest differs")
                clean_artifact = bound["artifacts"]["predecode_clean_latent"]
                gaussian_artifact = bound["artifacts"]["official_initial_gaussian"]
                latent_binding = {
                    **_artifact_native_binding(
                        clean_artifact,
                        tensor_digest=owner["clean_tensor_sha256"],
                        authenticated_identity=owner["clean_authentication"],
                    ),
                    "clean_latent_authentication": dict(
                        owner["clean_authentication"]
                    ),
                    "source_shape": list(cell_shape),
                    "temporal_transform": arm["temporal_transform"],
                    "transform_applied_before_noising": True,
                    "transformed_shape": list(cell_shape),
                    "transformed_tensor_sha256": transformed_sha,
                    "generated_clean_latent_used_only_as_frozen_hidden_query": True,
                }
                gaussian_binding = {
                    **_artifact_native_binding(
                        gaussian_artifact,
                        tensor_digest=owner["gaussian_tensor_sha256"],
                    ),
                    "shape": list(cell_shape),
                    "same_cell_tensor_sha256": first_gaussian_sha,
                    "temporal_transform_applied": False,
                    "absolute_phase_official_gaussian": True,
                }
                source_binding = {
                    "candidate_id": candidate_id,
                    "semantic_branch": candidate["semantic_branch"],
                    "candidate_envelope_sha256": bound["candidate_envelope_sha256"],
                    "generation_receipt_digest": bound["generation_receipt_digest"],
                    "generation_receipt_file_sha256": bound[
                        "generation_receipt_file_sha256"
                    ],
                    "native_rollout_receipt_digest": bound[
                        "native_rollout_receipt_digest"
                    ],
                    "native_rollout_receipt_file_sha256": bound[
                        "native_rollout_receipt_file_sha256"
                    ],
                }
                event_binding = {
                    "manifest_path": str(label_path),
                    "manifest_file_sha256": label_file_sha,
                    "manifest_digest": label_manifest["manifest_digest"],
                    "audit_source_kind": label["audit_source_kind"],
                    "external_audit_artifact_path": label[
                        "external_audit_artifact_path"
                    ],
                    "external_audit_artifact_sha256": label[
                        "external_audit_artifact_sha256"
                    ],
                    "complete_target_transition_observed": label[
                        "complete_target_transition_observed"
                    ],
                    "terminal_hold_observed": label["terminal_hold_observed"],
                    "full_target_action_observed": label[
                        "full_target_action_observed"
                    ],
                    "full_target_action_false_confirmed": label[
                        "full_target_action_false_confirmed"
                    ],
                    "labels_are_external_and_detached": True,
                    "labels_may_enter_model_condition": False,
                }
                receipt = make_arm_receipt(
                    group_id=args.group_id,
                    episode_id=cell_id,
                    split=candidate["analysis_split"],
                    role=role,
                    label=arm["label"],
                    action_family_id=candidate["action_family_id"],
                    actor_group_id=candidate["actor_group_id"],
                    scene_group_id=candidate["scene_group_id"],
                    action_group_id=candidate["action_group_id"],
                    seed=candidate["seed"],
                    source_candidate_binding=source_binding,
                    event_label_binding=event_binding,
                    critic_use_binding=critic_use_binding,
                    latent_binding=latent_binding,
                    official_gaussian_binding=gaussian_binding,
                    prompt_binding=prompt_binding,
                    same_state_query_binding=same_state_proof,
                    hidden_binding=hidden_binding,
                    spatial_sketch_binding=spatial_binding,
                    artifact=artifact,
                    model_binding=model_binding,
                    runtime_binding=runtime_binding,
                )
                receipt_digests: list[Any] = [None] * EXPECTED_SP_WORLD
                dist.all_gather_object(receipt_digests, receipt["receipt_digest"])
                if len(set(receipt_digests)) != 1:
                    raise STARCMaterializationError("SP4 arm receipt digests differ")

                def write_receipt() -> dict[str, Any]:
                    path = arm_dir / ARM_RECEIPT_FILENAME
                    receipt_file_sha = _write_json_create_only(path, receipt)
                    return arm_binding_from_receipt(
                        receipt,
                        receipt_path=path.resolve(strict=True),
                        receipt_file_sha256=receipt_file_sha,
                    )

                binding = _rank0_action(
                    dist=dist,
                    rank=distributed.rank,
                    label=f"write {cell_id}/{role} receipt",
                    action=write_receipt,
                )
                arm_bindings.append(dict(binding))
                del transformed, x_sigma, residual, sigma
            del epsilon, conditions
            cached.clear()
            observer.remove()
            observer = None
            del spatial

        try:
            freeze_after = native_generation.source_audit.model_freeze_certificate(renderer)
        except Exception as error:
            raise STARCMaterializationError(str(error)) from error
        if freeze_after != freeze_before or any(
            parameter.requires_grad for parameter in renderer.parameters()
        ):
            raise STARCMaterializationError("frozen renderer changed during materialization")
        group_receipt = make_group_receipt(
            group_id=args.group_id,
            root_spec_binding=root_spec_binding,
            bank_binding=bank_binding,
            detached_event_label_binding=detached_label_binding,
            critic_use_binding=critic_use_binding,
            model_binding=model_binding,
            runtime_binding=runtime_binding,
            spatial_sketch_bindings_by_episode=spatial_bindings_by_episode,
            episode_order=episode_order,
            episode_splits=episode_splits,
            arm_bindings=arm_bindings,
        )
        group_digests: list[Any] = [None] * EXPECTED_SP_WORLD
        dist.all_gather_object(group_digests, group_receipt["receipt_digest"])
        if len(set(group_digests)) != 1:
            raise STARCMaterializationError("SP4 group receipt digests differ")

        def write_group() -> dict[str, Any]:
            validate_group_receipt(group_receipt, verify_children=True)
            path = group_root / GROUP_FILENAME.format(group_id=args.group_id)
            digest = _write_json_create_only(path, group_receipt)
            for cell_id in episode_order:
                for role in ARM_ORDER:
                    os.chmod(group_root / cell_id / role, 0o500)
                os.chmod(group_root / cell_id, 0o500)
            os.chmod(group_root, 0o500)
            return {"path": str(path.resolve(strict=True)), "file_sha256": digest}

        _rank0_action(
            dist=dist,
            rank=distributed.rank,
            label="write group receipt",
            action=write_group,
        )
        dist.barrier()
        return 0
    finally:
        if observer is not None and observer.active:
            observer.abort()
        if observer is not None and observer.installed:
            observer.remove()
        if dist.is_initialized():
            dist.destroy_process_group()


def aggregate_master(args: argparse.Namespace) -> int:
    output_root = _plain_directory(args.output_root, label="master output root")
    output = _fresh_file(output_root / MASTER_FILENAME, label="master receipt output")
    inputs = []
    for group_id, path_text, expected in (
        ("sp4-a", args.sp4_a_group_manifest, args.expected_sp4_a_group_manifest_sha256),
        ("sp4-b", args.sp4_b_group_manifest, args.expected_sp4_b_group_manifest_sha256),
    ):
        raw, path, observed = _strict_json_file(
            path_text,
            expected_sha256=_sha256(expected, label=f"{group_id} manifest expected SHA-256"),
            label=f"{group_id} group manifest",
        )
        inputs.append((raw, path, observed))
    receipt = make_master_receipt(inputs)
    validate_master_receipt(receipt, verify_groups=True)
    _write_json_create_only(output, receipt)
    return 0


def author_sidecar(args: argparse.Namespace) -> int:
    output = _fresh_file(args.output, label="critic-use sidecar output")
    sidecar = make_required_critic_use_sidecar(
        authority_evidence=args.critic_use_authority_evidence
    )
    _write_json_create_only(output, sidecar)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sidecar = commands.add_parser(
        "author-sidecar", description="Seal the exact core4 critic-only use sidecar."
    )
    sidecar.add_argument("--critic-use-authority-evidence", required=True)
    sidecar.add_argument("--output", required=True)

    materialize = commands.add_parser(
        "materialize-group", description="Materialize one 2-cell/26-arm SP4 group."
    )
    materialize.add_argument("--root-spec", required=True)
    materialize.add_argument("--expected-root-spec-sha256", required=True)
    materialize.add_argument("--bank-output-dir", required=True)
    materialize.add_argument("--bank-receipt", required=True)
    materialize.add_argument("--expected-bank-receipt-sha256", required=True)
    materialize.add_argument("--detached-label-manifest", required=True)
    materialize.add_argument("--expected-detached-label-manifest-sha256", required=True)
    materialize.add_argument("--critic-use-sidecar", required=True)
    materialize.add_argument("--expected-critic-use-sidecar-sha256", required=True)
    materialize.add_argument("--critic-use-authority-evidence", required=True)
    materialize.add_argument("--group-id", choices=GROUP_ORDER, required=True)
    materialize.add_argument("--bernini-root", required=True)
    materialize.add_argument("--veomni-root", required=True)
    materialize.add_argument("--checkpoint", required=True)
    materialize.add_argument("--checkpoint-content-manifest", required=True)
    materialize.add_argument("--output-root", required=True)
    materialize.add_argument("--expected-bernini-commit", required=True)
    materialize.add_argument("--expected-veomni-commit", required=True)
    materialize.add_argument("--method-source-revision", required=True)
    materialize.add_argument("--method-source-archive-sha256", required=True)
    materialize.add_argument("--formal-d541801-source-revision", required=True)
    materialize.add_argument("--formal-d541801-source-archive-sha256", required=True)
    materialize.add_argument("--expected-materializer-source-sha256", required=True)
    materialize.add_argument("--expected-formal-d541801-scorer-source-sha256", required=True)
    materialize.add_argument("--expected-temporal-scorer-source-sha256", required=True)
    materialize.add_argument("--expected-temporal-contract-source-sha256", required=True)
    materialize.add_argument("--expected-fitq-observer-source-sha256", required=True)
    materialize.add_argument("--expected-dataset-contract-source-sha256", required=True)
    materialize.add_argument("--expected-label-author-source-sha256", required=True)
    materialize.add_argument("--ack-generated-t2v-hidden-critic-only", action="store_true")
    materialize.add_argument("--ack-no-generated-media-editor-use", action="store_true")
    materialize.add_argument("--ack-no-optimizer-or-editor-update", action="store_true")

    aggregate = commands.add_parser(
        "aggregate-master", description="Authenticate both group trees and seal core4 master."
    )
    aggregate.add_argument("--output-root", required=True)
    aggregate.add_argument("--sp4-a-group-manifest", required=True)
    aggregate.add_argument("--expected-sp4-a-group-manifest-sha256", required=True)
    aggregate.add_argument("--sp4-b-group-manifest", required=True)
    aggregate.add_argument("--expected-sp4-b-group-manifest-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "author-sidecar":
        return author_sidecar(args)
    if args.command == "materialize-group":
        return materialize_group(args)
    if args.command == "aggregate-master":
        return aggregate_master(args)
    raise STARCMaterializationError("unknown materializer command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "ARM_RECEIPT_FILENAME",
    "ARM_SCHEMA",
    "ARMS_PER_CELL",
    "ARMS_PER_GROUP",
    "Block15SpatialSketchObserver",
    "CORE4_ARM_COUNT",
    "GROUP_FILENAME",
    "GROUP_ORDER",
    "GROUP_SCHEMA",
    "GlobalBlock15Sketch",
    "HOOK_COORDINATE",
    "CORE4_LATENT_SHAPES",
    "CORE4_PATCH_GRIDS",
    "LATENT_PREFIX",
    "LocalBlock15Sketch",
    "MASTER_FILENAME",
    "MASTER_SCHEMA",
    "MODEL_FORWARDS_TOTAL",
    "REQUIRED_CRITIC_USE_EVIDENCE_SHA256",
    "SCHEDULE_INDEX",
    "SIGMA",
    "SKETCH_SEED",
    "STARCMaterializationError",
    "TENSOR_FILENAME",
    "TENSOR_KEY",
    "all_reduce_block15_sketch",
    "apply_temporal_transform",
    "arm_binding_from_receipt",
    "build_parser",
    "build_starc_local_layout",
    "fixed_spatial_sketch",
    "forward_same_state_hidden_pair",
    "load_required_critic_use_sidecar",
    "latent_geometry",
    "make_arm_receipt",
    "make_group_receipt",
    "make_master_receipt",
    "make_required_critic_use_sidecar",
    "save_residual_artifact",
    "sketch_binding",
    "tensor_sha256",
    "validate_arm_receipt",
    "validate_clean_latent_authentication_binding",
    "validate_group_receipt",
    "validate_master_receipt",
    "validate_spatial_sketch_binding",
    "verify_authenticated_native_clean_tensor_identity",
]

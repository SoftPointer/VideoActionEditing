#!/usr/bin/env python3
"""Materialize the core4 pilot's frozen Bernini hidden-event queries.

This executable is deliberately narrower than a trainer.  For each of the four
sealed core4-v2 cells it materializes exactly thirteen critic arms: one
event-qualified action owner, three deterministic same-video temporal controls,
and all nine externally false-confirmed semantic controls.  Every arm is queried
at the preregistered native UniPC-40 coordinate 33 under the cell's target-action
and scene-matched no-op captions on the *same* ``x_sigma`` object.

Only ``block.15.output`` is observed.  A read-only hook forms a fixed,
content-independent Rademacher spatial sketch on each Ulysses shard; the SP4
collective is executed after the official forward returns.  The saved tensor is
``sketch(H_action) - sketch(H_noop)`` with shape ``[1,21,16,1536]``.  Full hidden
states are hashed by ordered shard identities and are not persisted.

Generated T2V media/latents remain critic-query owners only.  They are never an
RV2V target, condition, donor, reference, feature target, or initial noise.  The
program has no optimizer, performs no training, and every receipt explicitly
denies editor authority.  The all-eight-GPU launcher runs two independent SP4
instances and invokes this file's CPU-only ``merge`` command afterward.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
for _search_root in (METHOD_ROOT, TOOLS_ROOT):
    if str(_search_root) not in sys.path:
        sys.path.insert(0, str(_search_root))

import latent_temporal_event_critic_dataset as data_contract  # noqa: E402
import temporal_counterfactual_action_scorer_v1 as frozen_pair  # noqa: E402
import temporal_counterfactual_contract_v1 as temporal_contract  # noqa: E402


GROUP_SCHEMA = "bernini-core4-frozen-hidden-event-query-group-v1"
ARM_SCHEMA = "bernini-core4-frozen-hidden-event-query-arm-v1"
POPULATION_SCHEMA = "bernini-core4-frozen-hidden-event-query-population-v1"
GROUP_FILENAME = "latent-temporal-event-hidden-group-v1.json"
POPULATION_FILENAME = "latent-temporal-event-hidden-population-v1.json"
EPISODE_FILENAME = "episode-plan-v1.json"
ARM_RECEIPT_FILENAME = "hidden-query-arm-v1.json"
RESIDUAL_FILENAME = "sketched-action-minus-noop.safetensors"
RESIDUAL_KEY = "sketched_action_minus_noop"

GROUP_IDS = ("sp4-a", "sp4-b")
GROUP_SIZE = 20
CELLS_PER_GROUP = 2
ARMS_PER_CELL = len(data_contract.ARM_ROLES)
CORE4_CELLS = 4
CORE4_ARMS = CORE4_CELLS * ARMS_PER_CELL
PROMPTS_PER_ARM = 2
MODEL_FORWARDS_PER_GROUP = CELLS_PER_GROUP * ARMS_PER_CELL * PROMPTS_PER_ARM
# One action/no-op pair is repeated on each SP4 group with the hook disabled.
HOOK_PARITY_FORWARDS_PER_GROUP = PROMPTS_PER_ARM
TOTAL_MODEL_FORWARDS_PER_GROUP = (
    MODEL_FORWARDS_PER_GROUP + HOOK_PARITY_FORWARDS_PER_GROUP
)

HOOK_BLOCK_INDEX = 15
HOOK_COORDINATE = "block.15.output"
SPATIAL_SKETCH_COORDINATES = 16
SPATIAL_SKETCH_SEED = 20260808017
# The seed is preregistered; the actual matrix (and therefore its digest) is
# reconstructed from each episode's authenticated native P.  These pins are a
# startup self-test for the three geometries in the sealed core4 bank, not a
# license to substitute one matrix for another.
CORE4_SPATIAL_SKETCH_DIGESTS = {
    930: "4a8330c77079671f6515bda07acc21f0d060176c4c07d2609ad2553acf657561",
    928: "be52cac4d90f0a5a70368d25fef2fb1edb4d346fb10598329f5bb7e8e7285ede",
    918: "9cc6e96d5909542189ca43ea2ff54efda6a44b302483890629b82d2ecad7f7ba",
}
NATIVE_SCHEDULE_INDEX = 33
NATIVE_SIGMA = 0.5161304473876953
NATIVE_TIMESTEP = 516

REQUIRED_LABEL_MANIFEST_FILE_SHA256 = (
    "9246504e97e1ee46c2cdcf7dfac0f41364dca40f26e5c26f28f0968d0443808d"
)

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class Core4HiddenMaterializationError(RuntimeError):
    """A source, bank, label, model, hook, tensor, or receipt failed closed."""


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
        raise Core4HiddenMaterializationError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(value: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(value).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Core4HiddenMaterializationError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise Core4HiddenMaterializationError(f"{label} must be lowercase SHA-1")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise Core4HiddenMaterializationError(f"{label} must be a path-safe ID")
    return value


def _no_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise Core4HiddenMaterializationError(f"{label} must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise Core4HiddenMaterializationError(
                f"{label} contains a symlink component"
            )


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    _no_symlink_components(path, label=label)
    if not path.is_file() or path.resolve(strict=True) != path:
        raise Core4HiddenMaterializationError(
            f"{label} must be a normalized absolute plain file"
        )
    return path


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    _no_symlink_components(path, label=label)
    if not path.is_dir() or path.resolve(strict=True) != path:
        raise Core4HiddenMaterializationError(
            f"{label} must be a normalized absolute plain directory"
        )
    return path


def _fresh_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        raise Core4HiddenMaterializationError(
            f"{label} must be a fresh normalized absolute directory"
        )
    parent = _plain_directory(path.parent, label=f"{label} parent")
    if parent / path.name != path:
        raise Core4HiddenMaterializationError(f"{label} path is not normalized")
    return path


def _plain_descendant_file(
    value: str | Path, *, root: Path, label: str
) -> Path:
    path = _plain_file(value, label=label)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Core4HiddenMaterializationError(
            f"{label} is outside the materialization output root"
        ) from error
    return path


def _reject_constant(token: str) -> None:
    raise Core4HiddenMaterializationError(f"non-finite JSON is forbidden: {token}")


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Core4HiddenMaterializationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_strict_json(
    value: str | Path, *, label: str, expected_sha256: str
) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(value, label=label)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != _sha256(expected_sha256, label=f"{label} expected SHA-256"):
        raise Core4HiddenMaterializationError(f"{label} SHA-256 differs")
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Core4HiddenMaterializationError(f"{label} is invalid ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise Core4HiddenMaterializationError(f"{label} root must be an object")
    return decoded, path, observed


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(unsigned)
    return {**row, "receipt_digest": object_sha256(row)}


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise Core4HiddenMaterializationError(f"refusing to overwrite {path}")
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    os.chmod(path, 0o400)
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class NativeEpisodeGeometry:
    latent_shape: tuple[int, int, int, int, int]
    latent_height: int
    latent_width: int
    patch_height: int
    patch_width: int
    patch_positions: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "latent_shape": list(self.latent_shape),
            "latent_height_width": [self.latent_height, self.latent_width],
            "spatial_patch_size_height_width": list(
                data_contract.LATENT_SPATIAL_PATCH_SIZE
            ),
            "patch_grid_height_width": [self.patch_height, self.patch_width],
            "patch_positions": self.patch_positions,
            "patch_flatten_order": "patch-y-x",
            "resize_or_crop_applied": False,
        }


def derive_core4_native_geometry(value: Any) -> NativeEpisodeGeometry:
    """Derive, then core4-pin, one authenticated tensor/receipt geometry."""

    raw_shape = getattr(value, "shape", value)
    try:
        shape = tuple(int(item) for item in raw_shape)
    except (TypeError, ValueError) as error:
        raise Core4HiddenMaterializationError(
            "native latent shape is not an integer sequence"
        ) from error
    try:
        binding = data_contract.derive_native_geometry(shape)
    except data_contract.LatentTemporalEventDatasetError as error:
        raise Core4HiddenMaterializationError(str(error)) from error
    if shape not in data_contract.CORE4_NATIVE_LATENT_SHAPES:
        raise Core4HiddenMaterializationError(
            "native latent shape is not one of the authenticated core4 geometries"
        )
    patch_height, patch_width = binding["patch_grid_height_width"]
    patch_positions = binding["patch_positions"]
    if CORE4_SPATIAL_SKETCH_DIGESTS.get(patch_positions) is None:
        raise Core4HiddenMaterializationError(
            "native core4 patch count has no preregistered sketch-family pin"
        )
    return NativeEpisodeGeometry(
        latent_shape=shape,
        latent_height=shape[3],
        latent_width=shape[4],
        patch_height=patch_height,
        patch_width=patch_width,
        patch_positions=patch_positions,
    )


@dataclass(frozen=True)
class LocalTargetIndexPlan:
    """Pure-Python map from one contiguous Ulysses shard to global 21xP."""

    sp_rank: int
    patch_height: int
    patch_width: int
    patch_positions: int
    global_sequence_length: int
    shard_start: int
    shard_stop: int
    local_sequence_length: int
    local_indices: tuple[int, ...]
    global_indices: tuple[int, ...]
    phase_indices: tuple[int, ...]
    spatial_indices: tuple[int, ...]
    phase_counts: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sp_rank": self.sp_rank,
            "sp_world": 4,
            "patch_grid_height_width": [self.patch_height, self.patch_width],
            "patch_positions": self.patch_positions,
            "patch_flatten_order": "patch-y-x",
            "global_sequence_length": self.global_sequence_length,
            "shard_start": self.shard_start,
            "shard_stop": self.shard_stop,
            "local_sequence_length": self.local_sequence_length,
            "target_tokens_selected": len(self.local_indices),
            "padding_tokens_excluded": self.local_sequence_length
            - len(self.local_indices),
            "phase_counts": list(self.phase_counts),
        }


def make_local_target_index_plan(
    sp_rank: int, *, patch_height: int, patch_width: int
) -> LocalTargetIndexPlan:
    """Build one native-geometry T2V target-only contiguous SP4 layout."""

    if type(sp_rank) is not int or not 0 <= sp_rank < 4:
        raise Core4HiddenMaterializationError("SP rank must be one of 0,1,2,3")
    geometry = derive_core4_native_geometry(
        (
            1,
            data_contract.LATENT_CHANNELS,
            data_contract.LATENT_PHASES,
            patch_height * data_contract.LATENT_SPATIAL_PATCH_SIZE[0],
            patch_width * data_contract.LATENT_SPATIAL_PATCH_SIZE[1],
        )
    )
    global_tokens = data_contract.LATENT_PHASES * geometry.patch_positions
    local_length = math.ceil(global_tokens / 4)
    start = sp_rank * local_length
    stop = start + local_length
    local: list[int] = []
    global_values: list[int] = []
    phases: list[int] = []
    positions: list[int] = []
    counts = [0] * data_contract.LATENT_PHASES
    for local_index in range(local_length):
        global_index = start + local_index
        if global_index >= global_tokens:
            continue
        phase, position = divmod(global_index, geometry.patch_positions)
        local.append(local_index)
        global_values.append(global_index)
        phases.append(phase)
        positions.append(position)
        counts[phase] += 1
    if (
        len(local) != min(local_length, max(global_tokens - start, 0))
        or sum(counts) != len(local)
        or any(not 0 <= value < geometry.patch_positions for value in positions)
    ):
        raise Core4HiddenMaterializationError("local target layout differs")
    return LocalTargetIndexPlan(
        sp_rank=sp_rank,
        patch_height=geometry.patch_height,
        patch_width=geometry.patch_width,
        patch_positions=geometry.patch_positions,
        global_sequence_length=global_tokens,
        shard_start=start,
        shard_stop=stop,
        local_sequence_length=local_length,
        local_indices=tuple(local),
        global_indices=tuple(global_values),
        phase_indices=tuple(phases),
        spatial_indices=tuple(positions),
        phase_counts=tuple(counts),
    )


def validate_sp4_contiguous_layouts(
    layouts: Sequence[Mapping[str, Any]], *, geometry: NativeEpisodeGeometry
) -> tuple[dict[str, Any], ...]:
    """Rebuild every rank plan and reject gaps, overlap, or misplaced padding."""

    if not isinstance(layouts, Sequence) or isinstance(layouts, (str, bytes)):
        raise Core4HiddenMaterializationError("SP4 layout collection differs")
    expected = tuple(
        make_local_target_index_plan(
            rank,
            patch_height=geometry.patch_height,
            patch_width=geometry.patch_width,
        ).as_dict()
        for rank in range(4)
    )
    observed = tuple(dict(row) for row in layouts if isinstance(row, Mapping))
    if len(observed) != 4 or observed != expected:
        raise Core4HiddenMaterializationError(
            "SP4 contiguous shard layout/padding differs from native geometry"
        )
    if (
        sum(row["target_tokens_selected"] for row in observed)
        != geometry.patch_positions * data_contract.LATENT_PHASES
        or sum(row["padding_tokens_excluded"] for row in observed)
        != 4 * observed[0]["local_sequence_length"]
        - geometry.patch_positions * data_contract.LATENT_PHASES
        or any(row["padding_tokens_excluded"] for row in observed[:-1])
    ):
        raise Core4HiddenMaterializationError(
            "SP4 valid-token or terminal-padding closure differs"
        )
    return observed


def spatial_sketch_binding(
    spatial_sketch: Any, *, geometry: NativeEpisodeGeometry, critic_core: Any
) -> dict[str, Any]:
    """Bind the actual FP32 matrix used for one episode, including its P."""

    import torch

    if (
        not isinstance(spatial_sketch, torch.Tensor)
        or spatial_sketch.dtype != torch.float32
        or tuple(int(item) for item in spatial_sketch.shape)
        != (SPATIAL_SKETCH_COORDINATES, geometry.patch_positions)
        or spatial_sketch.requires_grad
        or spatial_sketch.grad_fn is not None
        or not bool(torch.isfinite(spatial_sketch).all().item())
    ):
        raise Core4HiddenMaterializationError(
            "episode spatial sketch tensor closure differs"
        )
    digest = critic_core._canonical_tensor_digest(
        spatial_sketch, label="episode fixed spatial sketch"
    )
    pure_digest = pure_python_spatial_sketch_digest(
        patch_positions=geometry.patch_positions
    )
    if (
        digest != pure_digest
        or digest != CORE4_SPATIAL_SKETCH_DIGESTS[geometry.patch_positions]
    ):
        raise Core4HiddenMaterializationError(
            "episode Torch/Python spatial sketch digest differs"
        )
    return {
        "family": "sha256-counter-rademacher-dynamic-native-p-v1",
        "coordinates": SPATIAL_SKETCH_COORDINATES,
        "patch_grid_height_width": [
            geometry.patch_height,
            geometry.patch_width,
        ],
        "patch_positions": geometry.patch_positions,
        "flatten_order": "patch-y-x",
        "seed": SPATIAL_SKETCH_SEED,
        "normalization": "one_over_sqrt_patch_positions",
        "tensor_dtype": "torch.float32",
        "tensor_shape": [
            SPATIAL_SKETCH_COORDINATES,
            geometry.patch_positions,
        ],
        "tensor_digest_scheme": "bernini-ltec-f32le-v1",
        "tensor_digest": digest,
        "content_dependent": False,
        "mask_or_localization_used": False,
    }


def validate_spatial_sketch_binding(value: Any) -> dict[str, Any]:
    """Pure-Python validation of a receipt's geometry-specific sketch binding."""

    if not isinstance(value, Mapping):
        raise Core4HiddenMaterializationError("spatial sketch binding must be an object")
    row = dict(value)
    expected_fields = {
        "family",
        "coordinates",
        "patch_grid_height_width",
        "patch_positions",
        "flatten_order",
        "seed",
        "normalization",
        "tensor_dtype",
        "tensor_shape",
        "tensor_digest_scheme",
        "tensor_digest",
        "content_dependent",
        "mask_or_localization_used",
    }
    if set(row) != expected_fields:
        raise Core4HiddenMaterializationError("spatial sketch binding field closure differs")
    grid = row.get("patch_grid_height_width")
    if not isinstance(grid, list) or len(grid) != 2:
        raise Core4HiddenMaterializationError("spatial sketch patch grid differs")
    geometry = derive_core4_native_geometry(
        (
            1,
            data_contract.LATENT_CHANNELS,
            data_contract.LATENT_PHASES,
            grid[0] * data_contract.LATENT_SPATIAL_PATCH_SIZE[0]
            if type(grid[0]) is int
            else 0,
            grid[1] * data_contract.LATENT_SPATIAL_PATCH_SIZE[1]
            if type(grid[1]) is int
            else 0,
        )
    )
    expected_digest = pure_python_spatial_sketch_digest(
        patch_positions=geometry.patch_positions
    )
    if row != {
        "family": "sha256-counter-rademacher-dynamic-native-p-v1",
        "coordinates": SPATIAL_SKETCH_COORDINATES,
        "patch_grid_height_width": [geometry.patch_height, geometry.patch_width],
        "patch_positions": geometry.patch_positions,
        "flatten_order": "patch-y-x",
        "seed": SPATIAL_SKETCH_SEED,
        "normalization": "one_over_sqrt_patch_positions",
        "tensor_dtype": "torch.float32",
        "tensor_shape": [SPATIAL_SKETCH_COORDINATES, geometry.patch_positions],
        "tensor_digest_scheme": "bernini-ltec-f32le-v1",
        "tensor_digest": expected_digest,
        "content_dependent": False,
        "mask_or_localization_used": False,
    }:
        raise Core4HiddenMaterializationError("spatial sketch binding values differ")
    return row


def pure_python_spatial_sketch_digest(*, patch_positions: int) -> str:
    """Recompute one dynamic-P preregistered FP32 sketch digest without Torch."""

    if patch_positions not in CORE4_SPATIAL_SKETCH_DIGESTS:
        raise Core4HiddenMaterializationError(
            "spatial sketch P is not an authenticated core4 geometry"
        )
    scale = 1.0 / math.sqrt(float(patch_positions))
    raw = bytearray()
    for row in range(SPATIAL_SKETCH_COORDINATES):
        for column in range(patch_positions):
            token = f"{SPATIAL_SKETCH_SEED}:{row}:{column}".encode("ascii")
            sign = 1.0 if hashlib.sha256(token).digest()[0] & 1 else -1.0
            raw.extend(struct.pack("<f", sign * scale))
    header = (
        "bernini-ltec-f32le-v1|shape="
        f"{SPATIAL_SKETCH_COORDINATES},{patch_positions}|"
    ).encode("ascii")
    return hashlib.sha256(header + raw).hexdigest()


def _tensor_local_identity(value: Any, *, label: str) -> dict[str, Any]:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.numel() <= 0
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Core4HiddenMaterializationError(
            f"{label} must be detached finite hidden values"
        )
    cpu = value.detach().to(device="cpu").contiguous().clone()
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "byte_count": len(raw),
    }
    return {
        **metadata,
        "raw_value_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": hashlib.sha256(
            canonical_json_bytes(metadata) + b"\x00" + raw
        ).hexdigest(),
    }


@dataclass(frozen=True)
class HiddenPairCapture:
    action_sketch: Any
    noop_sketch: Any
    action_distributed_hidden_digest: str
    noop_distributed_hidden_digest: str
    shard_layouts: tuple[dict[str, Any], ...]
    global_phase_counts: tuple[int, ...]


class Block15FixedSketchPairObserver:
    """Read-only one-site hook; collectives occur only in :meth:`finish_pair`."""

    def __init__(
        self,
        transformer: Any,
        sketch: Any,
        *,
        sp_rank: int,
        geometry: NativeEpisodeGeometry,
    ) -> None:
        import torch

        blocks = getattr(transformer, "blocks", None)
        if blocks is None or len(blocks) != 30:
            raise Core4HiddenMaterializationError(
                "frozen Bernini transformer must contain exactly 30 blocks"
            )
        if not callable(getattr(blocks[HOOK_BLOCK_INDEX], "register_forward_hook", None)):
            raise Core4HiddenMaterializationError("block.15 is not hookable")
        if (
            not isinstance(sketch, torch.Tensor)
            or sketch.dtype != torch.float32
            or tuple(int(item) for item in sketch.shape)
            != (SPATIAL_SKETCH_COORDINATES, geometry.patch_positions)
            or sketch.requires_grad
            or sketch.grad_fn is not None
            or not bool(torch.isfinite(sketch).all().item())
        ):
            raise Core4HiddenMaterializationError("fixed spatial sketch differs")
        self.transformer = transformer
        self.block = blocks[HOOK_BLOCK_INDEX]
        self.sketch = sketch
        self.geometry = geometry
        self.plan = make_local_target_index_plan(
            sp_rank,
            patch_height=geometry.patch_height,
            patch_width=geometry.patch_width,
        )
        self._handle: Any = None
        self._active = False
        self._poisoned: Optional[str] = None
        self._captures: list[tuple[Any, Any]] = []

    @property
    def installed(self) -> bool:
        return self._handle is not None

    def install(self) -> None:
        if self.installed or self._poisoned is not None:
            raise Core4HiddenMaterializationError("observer cannot be installed")
        self._handle = self.block.register_forward_hook(self._hook)

    def remove(self) -> None:
        if not self.installed or self._active:
            raise Core4HiddenMaterializationError("observer removal state differs")
        self._handle.remove()
        self._handle = None

    def begin_pair(self) -> None:
        if not self.installed or self._active or self._poisoned is not None:
            raise Core4HiddenMaterializationError("observer pair state differs")
        self._captures = []
        self._active = True

    def abort_pair(self) -> None:
        self._captures = []
        self._active = False

    def _fail(self, message: str) -> None:
        self._poisoned = message
        self._active = False
        self._captures = []
        raise Core4HiddenMaterializationError(message)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        # No collective, no returned replacement tensor, and no retained graph.
        import torch

        if not self._active:
            return None
        if len(self._captures) >= PROMPTS_PER_ARM:
            self._fail("block.15 hook was called more than twice for one prompt pair")
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 3
            or int(output.shape[0]) != 1
            or int(output.shape[1]) != self.plan.local_sequence_length
            or int(output.shape[2]) != data_contract.HIDDEN_SIZE
            or output.device != self.sketch.device
            or not bool(torch.isfinite(output).all().item())
        ):
            self._fail("block.15 local hidden geometry/value differs")
        local_indices = torch.tensor(
            self.plan.local_indices, dtype=torch.long, device=output.device
        )
        phases = torch.tensor(
            self.plan.phase_indices, dtype=torch.long, device=output.device
        )
        positions = torch.tensor(
            self.plan.spatial_indices, dtype=torch.long, device=output.device
        )
        selected_native = output[0].index_select(0, local_indices).detach().contiguous()
        selected = selected_native.float()
        local_sketch = torch.zeros(
            data_contract.LATENT_PHASES,
            SPATIAL_SKETCH_COORDINATES,
            data_contract.HIDDEN_SIZE,
            dtype=torch.float32,
            device=output.device,
        )
        for phase in range(data_contract.LATENT_PHASES):
            phase_mask = phases == phase
            values = selected[phase_mask]
            phase_positions = positions[phase_mask]
            if int(values.shape[0]) != self.plan.phase_counts[phase]:
                self._fail("block.15 local phase coverage differs")
            if values.numel():
                weights = self.sketch.index_select(1, phase_positions).transpose(0, 1)
                local_sketch[phase] = torch.einsum("nk,nd->kd", weights, values)
        self._captures.append((selected_native, local_sketch.detach()))
        return None

    def finish_pair(self, dist: Any) -> HiddenPairCapture:
        """Reduce additive sketches only after both official forwards returned."""

        import torch

        if not self._active or len(self._captures) != PROMPTS_PER_ARM:
            self._fail("block.15 action/no-op capture closure differs")
        if dist.get_world_size() != 4 or dist.get_rank() != self.plan.sp_rank:
            self._fail("observer distributed topology differs")
        local_hidden = [capture[0] for capture in self._captures]
        global_sketches = [capture[1].clone() for capture in self._captures]
        local_layout = self.plan.as_dict()
        layouts: list[Any] = [None] * 4
        dist.all_gather_object(layouts, local_layout)
        try:
            checked_layouts = validate_sp4_contiguous_layouts(
                layouts, geometry=self.geometry
            )
        except Core4HiddenMaterializationError as error:
            self._fail(str(error))
        phase_counts = torch.tensor(
            self.plan.phase_counts, dtype=torch.int64, device=global_sketches[0].device
        )
        dist.all_reduce(phase_counts, op=dist.ReduceOp.SUM)
        expected_counts = (self.geometry.patch_positions,) * data_contract.LATENT_PHASES
        observed_counts = tuple(int(value) for value in phase_counts.tolist())
        if observed_counts != expected_counts:
            self._fail(
                "SP4 target phase coverage is not exactly the episode's native 21xP"
            )
        distributed_digests: list[str] = []
        for hidden, sketch in zip(local_hidden, global_sketches):
            local_identity = {
                "sp_rank": self.plan.sp_rank,
                "layout": local_layout,
                "hidden": _tensor_local_identity(hidden, label="local block.15 hidden"),
            }
            identities: list[Any] = [None] * 4
            dist.all_gather_object(identities, local_identity)
            if (
                [row.get("sp_rank") for row in identities] != [0, 1, 2, 3]
                or [row.get("layout") for row in identities]
                != list(checked_layouts)
                or any(
                    row.get("hidden", {}).get("shape")
                    != [
                        checked_layouts[index]["target_tokens_selected"],
                        data_contract.HIDDEN_SIZE,
                    ]
                    for index, row in enumerate(identities)
                )
            ):
                self._fail("distributed hidden identity order differs")
            distributed_digests.append(
                object_sha256(
                    {
                        "schema_version": "bernini-block15-distributed-hidden-digest-v1",
                        "hook_coordinate": HOOK_COORDINATE,
                        "ordered_shards": identities,
                    }
                )
            )
            dist.all_reduce(sketch, op=dist.ReduceOp.SUM)
            if not bool(torch.isfinite(sketch).all().item()):
                self._fail("globally reduced hidden sketch is non-finite")
        self._captures = []
        self._active = False
        return HiddenPairCapture(
            action_sketch=global_sketches[0].unsqueeze(0).detach(),
            noop_sketch=global_sketches[1].unsqueeze(0).detach(),
            action_distributed_hidden_digest=distributed_digests[0],
            noop_distributed_hidden_digest=distributed_digests[1],
            shard_layouts=checked_layouts,
            global_phase_counts=observed_counts,
        )


def _load_usage_authority(
    path: str | Path, *, expected_sha256: str, bank_receipt_digest: str
) -> tuple[dict[str, Any], Path, str]:
    raw, resolved, observed = _read_strict_json(
        path, label="critic-only use authority", expected_sha256=expected_sha256
    )
    try:
        authority = data_contract.validate_critic_usage_authority(raw)
    except data_contract.LatentTemporalEventDatasetError as error:
        raise Core4HiddenMaterializationError(str(error)) from error
    if authority["bank_receipt_digest"] != bank_receipt_digest:
        raise Core4HiddenMaterializationError("critic-use authority belongs to another bank")
    return authority, resolved, observed


def verify_authenticated_native_clean_tensor_identity(
    value: Any,
    artifact: Mapping[str, Any],
    *,
    label: str,
    frozen: Any,
) -> dict[str, Any]:
    """Authenticate one historical predecode clean latent fail-closed.

    Core4's clean-latent receipt predates the native ``raw_value_sha256`` and
    ``content_sha256`` fields.  Its producer-time authority is instead the
    SHA-256 of a single-tensor safetensors container plus the exact native
    coordinate/role and an FP32 save/reopen round trip.  This path reopens that
    already authenticated container and derives a *current observation* of the
    value identity.  It does not synthesize a producer-time value-digest claim.

    Newer receipts may declare both value digests, in which case the standard
    strict native-value verifier is also mandatory.  Declaring only one digest
    is an ambiguous downgrade and is rejected.
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
    value_digest_fields = {"raw_value_sha256", "content_sha256"}
    if not isinstance(artifact, Mapping):
        raise Core4HiddenMaterializationError(
            f"{label} native artifact must be an object"
        )
    declared_value_fields = set(artifact) & value_digest_fields
    if declared_value_fields not in (set(), value_digest_fields):
        raise Core4HiddenMaterializationError(
            f"{label} declares a partial native value identity"
        )
    if set(artifact) != historical_fields | declared_value_fields:
        raise Core4HiddenMaterializationError(
            f"{label} historical native artifact field closure differs"
        )
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3]) != (1, 16, 21)
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Core4HiddenMaterializationError(
            f"{label} must be one detached contiguous CPU FP32 exact81 tensor"
        )

    raw_path = artifact.get("path")
    if not isinstance(raw_path, (str, Path)):
        raise Core4HiddenMaterializationError(f"{label} artifact path differs")
    path = _plain_file(raw_path, label=f"{label} artifact")
    container_sha256 = _sha256(
        artifact.get("sha256"), label=f"{label} container SHA-256"
    )
    if file_sha256(path) != container_sha256:
        raise Core4HiddenMaterializationError(
            f"{label} authenticated container SHA-256 differs"
        )

    tensor_key = "normalized_clean_latent"
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [tensor_key]:
            raise Core4HiddenMaterializationError(
                f"{label} authenticated container key closure differs"
            )
        stored = opened.get_tensor(tensor_key).contiguous()
        metadata = dict(opened.metadata() or {})
    if file_sha256(path) != container_sha256:
        raise Core4HiddenMaterializationError(
            f"{label} authenticated container changed while reopening"
        )
    if (
        stored.dtype != torch.float32
        or stored.shape != value.shape
        or not torch.equal(stored, value)
    ):
        raise Core4HiddenMaterializationError(
            f"{label} loaded value differs from authenticated container"
        )

    expected_historical_values = {
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
    for field, expected in expected_historical_values.items():
        if artifact.get(field) != expected:
            raise Core4HiddenMaterializationError(
                f"{label} historical artifact field {field} differs"
            )
    expected_metadata = {
        "coordinate": "bernini_normalized_clean_vae_latent",
        "frame_contract": "exact81_latent21",
        "artifact_role": "native_sampler_proposal",
        "source": "native_sampler_before_vae_decode",
    }
    if metadata != expected_metadata:
        raise Core4HiddenMaterializationError(
            f"{label} safetensors metadata differs"
        )

    try:
        actual = frozen.native_tensor_value_identity(value)
        reopened = frozen.native_tensor_value_identity(stored)
        if declared_value_fields:
            strict = frozen.verify_native_tensor_value_identity(
                value, artifact, label=label
            )
            if strict != actual:
                raise Core4HiddenMaterializationError(
                    f"{label} strict/current native identity differs"
                )
    except frozen.PairV5T2VEnergyScoringError as error:
        raise Core4HiddenMaterializationError(str(error)) from error
    expected_identity_keys = {
        "shape",
        "dtype",
        "numel",
        "byte_count",
        "raw_value_sha256",
        "content_sha256",
    }
    if (
        set(actual) != expected_identity_keys
        or reopened != actual
        or actual["shape"] != expected_historical_values["shape"]
        or actual["dtype"] != "torch.float32"
    ):
        raise Core4HiddenMaterializationError(
            f"{label} current tensor/container value identity differs"
        )
    recorded_value_identity = bool(declared_value_fields)
    binding = {
        **actual,
        "authenticated_container_path": str(path),
        "authenticated_container_sha256": container_sha256,
        "single_tensor_container_reopened_byte_exact": True,
        "safetensors_metadata": metadata,
        "historical_native_coordinate_role_roundtrip_verified": True,
        "recorded_value_hashes_present": recorded_value_identity,
        "historical_native_receipt_value_hashes_absent": not recorded_value_identity,
        "strict_recorded_value_identity_verified": recorded_value_identity,
        "native_receipt_value_hashes_synthesized": False,
        "producer_time_value_digest_claimed_by_materializer": False,
        "observed_value_hashes_recomputed_after_authenticated_reopen": True,
        "value_identity_observation_time": "materializer_authenticated_reopen",
        "identity_authority": (
            "recorded_native_value_digests_and_authenticated_container"
            if recorded_value_identity
            else "authenticated_single_tensor_container_sha256_and_native_fp32_roundtrip"
        ),
    }
    return {**binding, "binding_digest": object_sha256(binding)}


def validate_clean_latent_authentication_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Core4HiddenMaterializationError(
            "clean latent authentication binding must be an object"
        )
    row = dict(value)
    declared = _sha256(
        row.pop("binding_digest", None), label="clean latent authentication digest"
    )
    if object_sha256(row) != declared:
        raise Core4HiddenMaterializationError(
            "clean latent authentication binding digest differs"
        )
    for field in (
        "raw_value_sha256",
        "content_sha256",
        "authenticated_container_sha256",
    ):
        _sha256(row.get(field), label=f"clean latent authentication {field}")
    if (
        row.get("shape") is None
        or row.get("dtype") != "torch.float32"
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
        raise Core4HiddenMaterializationError(
            "clean latent authentication semantic closure differs"
        )
    return {**row, "binding_digest": declared}


def _candidate_evidence(
    bound: Mapping[str, Any],
    label: Mapping[str, Any],
    *,
    bank_receipt_digest: str,
    official_gaussian_tensor_sha256: str,
    clean_latent_tensor_sha256: str,
    authenticated_clean_latent_shape: Sequence[int],
) -> dict[str, Any]:
    candidate = bound["candidate"]
    clean = bound["artifacts"]["predecode_clean_latent"]
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
        "official_gaussian_tensor_sha256": official_gaussian_tensor_sha256,
        "semantic_branch": candidate["semantic_branch"],
        "full_t2v_caption": candidate["full_t2v_caption"],
        "full_t2v_caption_utf8_sha256": candidate["full_t2v_caption_utf8_sha256"],
        "clean_latent_artifact_path": clean["path"],
        "clean_latent_artifact_sha256": clean["sha256"],
        "clean_latent_tensor_sha256": clean_latent_tensor_sha256,
        "clean_latent_shape": [int(item) for item in authenticated_clean_latent_shape],
        "generation_receipt_digest": bound["generation_receipt_digest"],
        "event_audit_artifact_sha256": label["external_audit_artifact_sha256"],
        **{
            name: label[name]
            for name in (
                "complete_target_transition_observed",
                "terminal_hold_observed",
                "full_target_action_observed",
                "full_target_action_false_confirmed",
            )
        },
    }


def _source_binding(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "materializer_source_sha256": args.expected_materializer_source_sha256,
        "observer_source_sha256": args.expected_observer_source_sha256,
        "critic_source_sha256": args.expected_critic_source_sha256,
        "dataset_contract_source_sha256": args.expected_dataset_source_sha256,
        "prompt_pair_scorer_source_sha256": args.expected_prompt_pair_scorer_source_sha256,
        "temporal_contract_source_sha256": args.expected_temporal_contract_source_sha256,
        "label_author_source_sha256": args.expected_label_author_source_sha256,
    }


def _validate_source_closure(args: argparse.Namespace) -> None:
    entries = {
        "materializer": (Path(__file__).resolve(), args.expected_materializer_source_sha256),
        "observer": (
            METHOD_ROOT / "internal_temporal_quotient_observer.py",
            args.expected_observer_source_sha256,
        ),
        "critic": (
            METHOD_ROOT / "latent_temporal_event_critic.py",
            args.expected_critic_source_sha256,
        ),
        "dataset": (
            METHOD_ROOT / "latent_temporal_event_critic_dataset.py",
            args.expected_dataset_source_sha256,
        ),
        "prompt-pair scorer": (
            METHOD_ROOT / "temporal_counterfactual_action_scorer_v1.py",
            args.expected_prompt_pair_scorer_source_sha256,
        ),
        "temporal contract": (
            METHOD_ROOT / "temporal_counterfactual_contract_v1.py",
            args.expected_temporal_contract_source_sha256,
        ),
        "label author": (
            TOOLS_ROOT / "author_pair_v5_core4_event_labels_d541801_v3.py",
            args.expected_label_author_source_sha256,
        ),
    }
    for label, (path, expected) in entries.items():
        if file_sha256(path) != _sha256(expected, label=f"{label} source SHA-256"):
            raise Core4HiddenMaterializationError(f"{label} source bytes differ")
    observed_geometry_pins = {
        derive_core4_native_geometry(shape).patch_positions
        for shape in data_contract.CORE4_NATIVE_LATENT_SHAPES
    }
    if observed_geometry_pins != set(CORE4_SPATIAL_SKETCH_DIGESTS):
        raise Core4HiddenMaterializationError("core4 geometry/sketch pin closure differs")
    for patch_positions, expected_digest in CORE4_SPATIAL_SKETCH_DIGESTS.items():
        if (
            pure_python_spatial_sketch_digest(patch_positions=patch_positions)
            != expected_digest
        ):
            raise Core4HiddenMaterializationError(
                "geometry-specific fixed spatial sketch digest differs"
            )
    _sha1(args.method_source_revision, label="method source revision")
    _sha256(args.method_source_archive_sha256, label="method source archive SHA-256")


def _group_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("group", help="materialize one SP4 group")
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--bank-output-dir", required=True)
    parser.add_argument("--bank-receipt", required=True)
    parser.add_argument("--expected-bank-receipt-sha256", required=True)
    parser.add_argument("--detached-event-label-manifest", required=True)
    parser.add_argument("--expected-detached-event-label-manifest-sha256", required=True)
    parser.add_argument("--critic-use-authority", required=True)
    parser.add_argument("--expected-critic-use-authority-sha256", required=True)
    parser.add_argument("--group-id", choices=GROUP_IDS, required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    for name in (
        "materializer",
        "observer",
        "critic",
        "dataset",
        "prompt-pair-scorer",
        "temporal-contract",
        "label-author",
    ):
        parser.add_argument(f"--expected-{name}-source-sha256", required=True)
    parser.add_argument(
        "--ack-core4-pilot-only-never-editor-target-or-authority",
        action="store_true",
    )


def _merge_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("merge", help="merge two finished SP4 receipts")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--bank-receipt", required=True)
    parser.add_argument("--expected-bank-receipt-sha256", required=True)
    parser.add_argument("--detached-event-label-manifest", required=True)
    parser.add_argument("--expected-detached-event-label-manifest-sha256", required=True)
    parser.add_argument("--critic-use-authority", required=True)
    parser.add_argument("--expected-critic-use-authority-sha256", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    for name in (
        "materializer",
        "observer",
        "critic",
        "dataset",
        "prompt-pair-scorer",
        "temporal-contract",
        "label-author",
    ):
        parser.add_argument(f"--expected-{name}-source-sha256", required=True)
    parser.add_argument(
        "--ack-core4-pilot-only-never-editor-target-or-authority",
        action="store_true",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _group_parser(subparsers)
    _merge_parser(subparsers)
    return parser


def _validate_common_args(args: argparse.Namespace) -> None:
    _validate_source_closure(args)
    if args.expected_root_spec_sha256 != temporal_contract.REQUIRED_CORE4_V2_SPEC_SHA256:
        raise Core4HiddenMaterializationError("core4-v2 root spec authority differs")
    if (
        args.expected_bank_receipt_sha256
        != temporal_contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
    ):
        raise Core4HiddenMaterializationError("core4-v2 bank receipt authority differs")
    if (
        args.expected_detached_event_label_manifest_sha256
        != REQUIRED_LABEL_MANIFEST_FILE_SHA256
    ):
        raise Core4HiddenMaterializationError("reviewed event-label authority differs")
    _sha256(
        args.expected_critic_use_authority_sha256,
        label="critic-use authority SHA-256",
    )
    if args.ack_core4_pilot_only_never_editor_target_or_authority is not True:
        raise Core4HiddenMaterializationError(
            "core4 pilot-only/no-editor acknowledgement is mandatory"
        )


def _validate_group_args(args: argparse.Namespace) -> None:
    _validate_common_args(args)
    for name in ("expected_bernini_commit", "expected_veomni_commit"):
        _sha1(getattr(args, name), label=name)
    if (
        args.expected_bernini_commit != temporal_contract.REQUIRED_BERNINI_REVISION
        or args.expected_veomni_commit != temporal_contract.REQUIRED_VEOMNI_REVISION
    ):
        raise Core4HiddenMaterializationError("frozen source-tree revisions differ")


def _load_core4_authorities(args: argparse.Namespace) -> tuple[Any, ...]:
    try:
        import author_pair_v5_core4_event_labels_d541801_v3 as label_author
    except ImportError as error:
        raise Core4HiddenMaterializationError("label-author runtime is unavailable") from error
    try:
        spec, bank, bound_rows = label_author.load_core4_bound_bank(
            root_spec=args.root_spec,
            root_spec_sha256=args.expected_root_spec_sha256,
            bank_output_dir=args.bank_output_dir,
            bank_receipt=args.bank_receipt,
            bank_receipt_sha256=args.expected_bank_receipt_sha256,
        )
        labels, label_path, label_file_sha = label_author.load_label_manifest(
            args.detached_event_label_manifest,
            expected_sha256=args.expected_detached_event_label_manifest_sha256,
            root_spec_raw_sha256=args.expected_root_spec_sha256,
            bank_receipt_digest=bank["receipt_digest"],
            bound_rows=bound_rows,
        )
    except label_author.PairV5Core4LabelAuthoringError as error:
        raise Core4HiddenMaterializationError(str(error)) from error
    if (
        bank["receipt_digest"]
        != temporal_contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST
        or len(bound_rows) != 40
    ):
        raise Core4HiddenMaterializationError("authenticated bank population differs")
    authority, authority_path, authority_file_sha = _load_usage_authority(
        args.critic_use_authority,
        expected_sha256=args.expected_critic_use_authority_sha256,
        bank_receipt_digest=bank["receipt_digest"],
    )
    return (
        spec,
        bank,
        bound_rows,
        labels,
        label_path,
        label_file_sha,
        authority,
        authority_path,
        authority_file_sha,
    )


def _encode_cell_prompts(
    *,
    rows: Sequence[Mapping[str, Any]],
    renderer: Any,
    tokenizer: Any,
    device: Any,
    frozen: Any,
    prompt_clean: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_branch = {row["candidate"]["semantic_branch"]: row for row in rows}
    if tuple(by_branch) != data_contract.SEMANTIC_BRANCHES:
        raise Core4HiddenMaterializationError("cell semantic branch order differs")
    native_generation = frozen.native_generation
    action = by_branch["action"]["candidate"]
    noop = by_branch["noop"]["candidate"]
    action_prompt = native_generation.build_task_prompt(
        "t2v", action["full_t2v_caption"], prompt_cleaner=prompt_clean
    )
    noop_prompt = native_generation.build_task_prompt(
        "t2v", noop["full_t2v_caption"], prompt_cleaner=prompt_clean
    )
    conditions, hashes = frozen_pair._encode_prompt_pair(
        renderer,
        tokenizer,
        action_prompt=action_prompt,
        noop_prompt=noop_prompt,
        device=device,
        frozen=frozen,
    )
    binding = frozen_pair._prompt_binding(
        target_action_caption_sha256=action["full_t2v_caption_utf8_sha256"],
        target_noop_caption_sha256=noop["full_t2v_caption_utf8_sha256"],
        action_prompt=action_prompt,
        noop_prompt=noop_prompt,
        condition_hashes=hashes,
        prompt_builder_contract_digest=frozen.prompt_builder_contract()["contract_digest"],
    )
    return conditions, binding


def _save_residual(
    *, output: Path, residual: Any, dist: Any, critic_core: Any
) -> dict[str, Any]:
    import torch

    if (
        not isinstance(residual, torch.Tensor)
        or residual.dtype != torch.float32
        or tuple(int(item) for item in residual.shape)
        != (
            1,
            data_contract.LATENT_PHASES,
            SPATIAL_SKETCH_COORDINATES,
            data_contract.HIDDEN_SIZE,
        )
        or residual.requires_grad
        or residual.grad_fn is not None
        or not bool(torch.isfinite(residual).all().item())
    ):
        raise Core4HiddenMaterializationError("fixed-sketch residual closure differs")
    tensor_digest = critic_core._canonical_tensor_digest(
        residual, label="fixed-sketch residual"
    )
    digests: list[Any] = [None] * 4
    dist.all_gather_object(digests, tensor_digest)
    if len(set(digests)) != 1:
        raise Core4HiddenMaterializationError("SP4 residual tensor digests differ")
    binding: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            from safetensors.torch import save_file
        except ImportError as error:
            raise Core4HiddenMaterializationError("safetensors runtime is unavailable") from error
        output.mkdir()
        artifact = output / RESIDUAL_FILENAME
        save_file({RESIDUAL_KEY: residual.detach().cpu().contiguous()}, str(artifact))
        os.chmod(artifact, 0o400)
        binding[0] = {
            "path": str(artifact),
            "sha256": file_sha256(artifact),
            "tensor_key": RESIDUAL_KEY,
            "tensor_digest": tensor_digest,
            "shape": list(residual.shape),
            "dtype": "torch.float32",
        }
    dist.broadcast_object_list(binding, src=0)
    if not isinstance(binding[0], Mapping):
        raise Core4HiddenMaterializationError("rank-zero residual binding differs")
    return dict(binding[0])


def _make_arm_receipt(
    *,
    group_id: str,
    episode: Mapping[str, Any],
    arm: Mapping[str, Any],
    prompt_binding: Mapping[str, Any],
    x_sigma_sha256: str,
    transformed_clean_sha256: str,
    clean_latent_authentication: Mapping[str, Any],
    effective_gaussian_sha256: str,
    capture: HiddenPairCapture,
    action_sketch_digest: str,
    noop_sketch_digest: str,
    residual_binding: Mapping[str, Any],
    same_state_proof: Mapping[str, Any],
    native_geometry: NativeEpisodeGeometry,
    fixed_spatial_sketch: Mapping[str, Any],
    model_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    hook_parity: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": ARM_SCHEMA,
            "group_id": group_id,
            "episode_id": episode["episode_id"],
            "episode_receipt_digest": episode["receipt_digest"],
            "arm_role": arm["role"],
            "label": arm["label"],
            "source_candidate_id": arm["source_candidate_id"],
            "source_semantic_branch": arm["source_semantic_branch"],
            "temporal_transform": arm["temporal_transform"],
            "clean_latent_artifact_sha256": arm["clean_latent_artifact_sha256"],
            "clean_latent_tensor_sha256": arm["clean_latent_tensor_sha256"],
            "clean_latent_authentication": dict(clean_latent_authentication),
            "transformed_clean_tensor_sha256": transformed_clean_sha256,
            "official_gaussian_temporally_transformed": False,
            "effective_gaussian_tensor_sha256": effective_gaussian_sha256,
            "x_sigma_tensor_sha256": x_sigma_sha256,
            "native_schedule_index": NATIVE_SCHEDULE_INDEX,
            "physical_sigma": NATIVE_SIGMA,
            "native_timestep": NATIVE_TIMESTEP,
            "hook_coordinate": HOOK_COORDINATE,
            "native_geometry": native_geometry.as_dict(),
            "fixed_spatial_sketch": dict(fixed_spatial_sketch),
            "action_distributed_hidden_digest": capture.action_distributed_hidden_digest,
            "noop_distributed_hidden_digest": capture.noop_distributed_hidden_digest,
            "action_sketched_hidden_digest": action_sketch_digest,
            "noop_sketched_hidden_digest": noop_sketch_digest,
            "residual_artifact": dict(residual_binding),
            "same_state_execution_proof": dict(same_state_proof),
            "shard_layouts": list(capture.shard_layouts),
            "global_phase_counts": list(capture.global_phase_counts),
            "prompt_binding": dict(prompt_binding),
            "model_binding": dict(model_binding),
            "source_binding": dict(source_binding),
            "hook_on_off_byte_parity": None if hook_parity is None else dict(hook_parity),
            "generated_t2v_role": "frozen_hidden_query_owner_only",
            "generated_media_or_latent_is_editor_target_condition_donor_or_noise": False,
            "generated_hidden_is_editor_feature_target": False,
            "event_labels_entered_model_condition": False,
            "mask_flow_pose_track_or_trajectory_used": False,
            "training_performed": False,
            "critic_optimizer_authorized": False,
            "editor_optimizer_authorized": False,
            "scientific_action_editing_claim_authorized": False,
        }
    )


def validate_arm_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Core4HiddenMaterializationError("arm receipt must be an object")
    row = dict(value)
    declared = _sha256(row.pop("receipt_digest", None), label="arm receipt digest")
    if object_sha256(row) != declared or row.get("schema_version") != ARM_SCHEMA:
        raise Core4HiddenMaterializationError("arm receipt digest/schema differs")
    residual = row.get("residual_artifact")
    try:
        clean_authentication = validate_clean_latent_authentication_binding(
            row.get("clean_latent_authentication")
        )
        geometry = derive_core4_native_geometry(
            row.get("native_geometry", {}).get("latent_shape")
        )
        fixed_spatial_sketch = validate_spatial_sketch_binding(
            row.get("fixed_spatial_sketch")
        )
        checked_layouts = validate_sp4_contiguous_layouts(
            row.get("shard_layouts", []), geometry=geometry
        )
    except (AttributeError, Core4HiddenMaterializationError) as error:
        raise Core4HiddenMaterializationError(
            "arm receipt native geometry/sketch/layout differs"
        ) from error
    if (
        row.get("native_geometry") != geometry.as_dict()
        or clean_authentication["authenticated_container_sha256"]
        != row.get("clean_latent_artifact_sha256")
        or clean_authentication["shape"] != list(geometry.latent_shape)
        or fixed_spatial_sketch["patch_grid_height_width"]
        != [geometry.patch_height, geometry.patch_width]
        or fixed_spatial_sketch["patch_positions"] != geometry.patch_positions
        or list(checked_layouts) != row.get("shard_layouts")
        or row.get("global_phase_counts")
        != [geometry.patch_positions] * data_contract.LATENT_PHASES
    ):
        raise Core4HiddenMaterializationError(
            "arm receipt native geometry/sketch/layout differs"
        )
    if (
        row.get("arm_role") not in data_contract.ARM_ROLES
        or row.get("native_schedule_index") != NATIVE_SCHEDULE_INDEX
        or float(row.get("physical_sigma", -1.0)).hex() != NATIVE_SIGMA.hex()
        or row.get("native_timestep") != NATIVE_TIMESTEP
        or row.get("hook_coordinate") != HOOK_COORDINATE
        or row.get("official_gaussian_temporally_transformed") is not False
        or not isinstance(residual, Mapping)
        or residual.get("shape")
        != [1, 21, SPATIAL_SKETCH_COORDINATES, data_contract.HIDDEN_SIZE]
        or row.get("generated_t2v_role") != "frozen_hidden_query_owner_only"
        or row.get(
            "generated_media_or_latent_is_editor_target_condition_donor_or_noise"
        )
        is not False
        or row.get("generated_hidden_is_editor_feature_target") is not False
        or row.get("event_labels_entered_model_condition") is not False
        or row.get("mask_flow_pose_track_or_trajectory_used") is not False
        or row.get("training_performed") is not False
        or row.get("critic_optimizer_authorized") is not False
        or row.get("editor_optimizer_authorized") is not False
        or row.get("scientific_action_editing_claim_authorized") is not False
    ):
        raise Core4HiddenMaterializationError("arm receipt semantic closure differs")
    return {**row, "receipt_digest": declared}


def make_group_receipt(
    *,
    group_id: str,
    episodes: Sequence[Mapping[str, Any]],
    arm_receipts: Sequence[Mapping[str, Any]],
    episode_file_bindings: Sequence[Mapping[str, Any]],
    arm_file_bindings: Sequence[Mapping[str, Any]],
    population_audit: Mapping[str, Any],
    bank_receipt_digest: str,
    root_spec_raw_sha256: str,
    label_binding: Mapping[str, Any],
    usage_binding: Mapping[str, Any],
    model_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    checked_arms = [validate_arm_receipt(row) for row in arm_receipts]
    if (
        group_id not in GROUP_IDS
        or len(episodes) != CELLS_PER_GROUP
        or len(checked_arms) != CELLS_PER_GROUP * ARMS_PER_CELL
        or len(episode_file_bindings) != CELLS_PER_GROUP
        or len(arm_file_bindings) != CELLS_PER_GROUP * ARMS_PER_CELL
        or [row["arm_role"] for row in checked_arms]
        != list(data_contract.ARM_ROLES) * CELLS_PER_GROUP
    ):
        raise Core4HiddenMaterializationError("group episode/arm closure differs")
    episode_order = [row["episode_id"] for row in episodes]
    native_geometries_by_episode: dict[str, dict[str, Any]] = {}
    spatial_sketches_by_episode: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        episode_id = episode["episode_id"]
        episode_arms = [row for row in checked_arms if row["episode_id"] == episode_id]
        if len(episode_arms) != ARMS_PER_CELL:
            raise Core4HiddenMaterializationError("group episode arm geometry closure differs")
        geometries = {canonical_json_bytes(row["native_geometry"]) for row in episode_arms}
        sketches = {
            canonical_json_bytes(row["fixed_spatial_sketch"]) for row in episode_arms
        }
        if (
            len(geometries) != 1
            or len(sketches) != 1
            or episode.get("native_geometry") != episode_arms[0]["native_geometry"]
        ):
            raise Core4HiddenMaterializationError(
                "one episode reused or mixed another native geometry/sketch"
            )
        native_geometries_by_episode[episode_id] = dict(
            episode_arms[0]["native_geometry"]
        )
        spatial_sketches_by_episode[episode_id] = dict(
            episode_arms[0]["fixed_spatial_sketch"]
        )
    return _seal(
        {
            "schema_version": GROUP_SCHEMA,
            "group_id": group_id,
            "root_spec_raw_sha256": root_spec_raw_sha256,
            "bank_receipt_digest": bank_receipt_digest,
            "detached_event_label_binding": dict(label_binding),
            "critic_use_authority_binding": dict(usage_binding),
            "model_binding": dict(model_binding),
            "source_binding": dict(source_binding),
            "episode_count": len(episodes),
            "episode_order": episode_order,
            "native_geometries_by_episode": native_geometries_by_episode,
            "spatial_sketches_by_episode": spatial_sketches_by_episode,
            "episode_receipt_digests": [row["receipt_digest"] for row in episodes],
            "episode_file_bindings": [dict(row) for row in episode_file_bindings],
            "split_order": [row["split"] for row in episodes],
            "arm_count": len(checked_arms),
            "arm_receipt_digests": [row["receipt_digest"] for row in checked_arms],
            "arm_file_bindings": [dict(row) for row in arm_file_bindings],
            "population_audit": dict(population_audit),
            "model_forwards": MODEL_FORWARDS_PER_GROUP,
            "hook_parity_forwards": HOOK_PARITY_FORWARDS_PER_GROUP,
            "total_model_forwards": TOTAL_MODEL_FORWARDS_PER_GROUP,
            "training_performed": False,
            "critic_optimizer_authorized": False,
            "confirmation_samples_consumed_by_optimizer": False,
            "editor_optimizer_authorized": False,
            "scientific_action_editing_claim_authorized": False,
            "pilot_pass_can_only_authorize": "fixed_topup_generation",
        }
    )


def validate_group_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Core4HiddenMaterializationError("group receipt must be an object")
    row = dict(value)
    declared = _sha256(row.pop("receipt_digest", None), label="group receipt digest")
    if object_sha256(row) != declared or row.get("schema_version") != GROUP_SCHEMA:
        raise Core4HiddenMaterializationError("group receipt digest/schema differs")
    episode_order = row.get("episode_order", [])
    native_geometries = row.get("native_geometries_by_episode")
    sketches = row.get("spatial_sketches_by_episode")
    geometry_bindings_valid = (
        isinstance(native_geometries, Mapping)
        and isinstance(sketches, Mapping)
        and list(native_geometries) == episode_order
        and list(sketches) == episode_order
    )
    if geometry_bindings_valid:
        try:
            for episode_id in episode_order:
                geometry = derive_core4_native_geometry(
                    native_geometries[episode_id].get("latent_shape")
                )
                if (
                    native_geometries[episode_id] != geometry.as_dict()
                    or validate_spatial_sketch_binding(sketches[episode_id])[
                        "patch_positions"
                    ]
                    != geometry.patch_positions
                ):
                    geometry_bindings_valid = False
        except (AttributeError, Core4HiddenMaterializationError):
            geometry_bindings_valid = False
    if (
        row.get("group_id") not in GROUP_IDS
        or row.get("episode_count") != CELLS_PER_GROUP
        or row.get("arm_count") != CELLS_PER_GROUP * ARMS_PER_CELL
        or len(row.get("episode_order", [])) != CELLS_PER_GROUP
        or not geometry_bindings_valid
        or len(row.get("episode_file_bindings", [])) != CELLS_PER_GROUP
        or len(row.get("arm_receipt_digests", []))
        != CELLS_PER_GROUP * ARMS_PER_CELL
        or len(row.get("arm_file_bindings", []))
        != CELLS_PER_GROUP * ARMS_PER_CELL
        or [item.get("receipt_digest") for item in row["episode_file_bindings"]]
        != row.get("episode_receipt_digests")
        or [item.get("receipt_digest") for item in row["arm_file_bindings"]]
        != row.get("arm_receipt_digests")
        or row.get("model_forwards") != MODEL_FORWARDS_PER_GROUP
        or row.get("hook_parity_forwards") != HOOK_PARITY_FORWARDS_PER_GROUP
        or row.get("total_model_forwards") != TOTAL_MODEL_FORWARDS_PER_GROUP
        or row.get("training_performed") is not False
        or row.get("critic_optimizer_authorized") is not False
        or row.get("confirmation_samples_consumed_by_optimizer") is not False
        or row.get("editor_optimizer_authorized") is not False
        or row.get("scientific_action_editing_claim_authorized") is not False
        or row.get("pilot_pass_can_only_authorize") != "fixed_topup_generation"
    ):
        raise Core4HiddenMaterializationError("group receipt semantic closure differs")
    return {**row, "receipt_digest": declared}


def make_population_receipt(
    group_receipts: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    population_audit: Mapping[str, Any],
) -> dict[str, Any]:
    checked = [validate_group_receipt(row) for row in group_receipts]
    if [row["group_id"] for row in checked] != list(GROUP_IDS):
        raise Core4HiddenMaterializationError("population group order differs")
    common_fields = (
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "detached_event_label_binding",
        "critic_use_authority_binding",
        "model_binding",
        "source_binding",
    )
    for field in common_fields:
        if checked[0][field] != checked[1][field]:
            raise Core4HiddenMaterializationError(
                f"two SP4 groups disagree on {field}"
            )
    episode_order = [item for row in checked for item in row["episode_order"]]
    split_order = [item for row in checked for item in row["split_order"]]
    native_geometries_by_episode = {
        episode_id: dict(row["native_geometries_by_episode"][episode_id])
        for row in checked
        for episode_id in row["episode_order"]
    }
    spatial_sketches_by_episode = {
        episode_id: dict(row["spatial_sketches_by_episode"][episode_id])
        for row in checked
        for episode_id in row["episode_order"]
    }
    if (
        len(episode_order) != CORE4_CELLS
        or len(set(episode_order)) != CORE4_CELLS
        or sorted(split_order) != ["confirmation", "confirmation", "fit", "fit"]
        or population_audit.get("protocol") != "core4_pilot"
        or population_audit.get("episode_count") != CORE4_CELLS
        or population_audit.get("population_eligible") is not True
        or population_audit.get("critic_head_pilot_training_authorized") is not True
        or population_audit.get("scientific_critic_claim_authorized") is not False
        or population_audit.get("editor_optimizer_authorized") is not False
    ):
        raise Core4HiddenMaterializationError("core4 pilot cell/split closure differs")
    return _seal(
        {
            "schema_version": POPULATION_SCHEMA,
            **{field: checked[0][field] for field in common_fields},
            "output_root": str(output_root),
            "group_order": list(GROUP_IDS),
            "group_receipt_digests": [row["receipt_digest"] for row in checked],
            "episode_count": CORE4_CELLS,
            "episode_order": episode_order,
            "native_geometries_by_episode": native_geometries_by_episode,
            "spatial_sketches_by_episode": spatial_sketches_by_episode,
            "split_counts": {"fit": 2, "confirmation": 2},
            "arm_count": CORE4_ARMS,
            "population_audit": dict(population_audit),
            "main_frozen_model_forwards": CORE4_ARMS * PROMPTS_PER_ARM,
            "hook_parity_forwards": len(GROUP_IDS) * HOOK_PARITY_FORWARDS_PER_GROUP,
            "total_frozen_model_forwards": len(GROUP_IDS)
            * TOTAL_MODEL_FORWARDS_PER_GROUP,
            "fit_scope": "critic_head_only_two_complete_cells",
            "confirmation_scope": "heldout_scoring_only_two_complete_cells",
            "confirmation_samples_consumed_by_optimizer": False,
            "core4_can_authorize_scientific_claim": False,
            "core4_can_authorize_editor_optimizer": False,
            "core4_can_supply_editor_target_condition_donor_or_noise": False,
            "passing_core4_pilot_can_only_authorize": "fixed_topup_generation",
            "training_performed": False,
            "optimizer_step_performed": False,
        }
    )


def _run_group(args: argparse.Namespace) -> int:
    _validate_group_args(args)
    frozen = frozen_pair._frozen_d541801_runtime()
    frozen_pair.validate_native_coordinate_runtime(frozen)
    if data_contract.PILOT_HIDDEN_QUERY != {
        **data_contract.PILOT_HIDDEN_QUERY,
        "native_schedule_index": NATIVE_SCHEDULE_INDEX,
        "sigma": NATIVE_SIGMA,
        "native_timestep": NATIVE_TIMESTEP,
        "hook_coordinate": HOOK_COORDINATE,
    }:
        raise Core4HiddenMaterializationError("pilot hidden-query coordinate differs")
    (
        spec,
        bank,
        bound_rows,
        labels,
        label_path,
        label_file_sha,
        authority,
        authority_path,
        authority_file_sha,
    ) = _load_core4_authorities(args)
    group_rows = [row for row in bound_rows if row["group_id"] == args.group_id]
    if len(group_rows) != GROUP_SIZE:
        raise Core4HiddenMaterializationError("SP4 group must contain exactly 20 owners")
    output = _fresh_directory(args.output_dir, label="group output")

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
        raise Core4HiddenMaterializationError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise Core4HiddenMaterializationError("pinned Bernini attention heads differ")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    import latent_temporal_event_critic as critic_core

    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise Core4HiddenMaterializationError("materializer requires four AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=4)
    device = torch.device("cuda", distributed.local_rank)

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
        raise Core4HiddenMaterializationError(
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
        freeze_certificate = native_generation.source_audit.model_freeze_certificate(renderer)
        checkpoint_binding = frozen.checkpoint_content_binding(
            checkpoint_identity, freeze_certificate
        )
    except Exception as error:
        raise Core4HiddenMaterializationError(str(error)) from error
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if (
        transformer is None
        or diffusion.transformer_2 is not None
        or any(parameter.requires_grad for parameter in renderer.parameters())
    ):
        raise Core4HiddenMaterializationError("frozen transformer_1 closure differs")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    if distributed.rank == 0:
        output.mkdir()
    dist.barrier()
    source_binding = _source_binding(args)
    label_binding = {
        "path": str(label_path),
        "file_sha256": label_file_sha,
        "manifest_digest": labels["manifest_digest"],
    }
    usage_binding = {
        "path": str(authority_path),
        "file_sha256": authority_file_sha,
        "receipt_digest": authority["receipt_digest"],
        "authorized_use": authority["authorized_use"],
    }
    model_binding = {
        "frozen_checkpoint_receipt_digest": checkpoint_receipt_digest,
        "checkpoint_content_manifest_sha256": checkpoint_binding["manifest_sha256"],
        "checkpoint_content_binding_digest": checkpoint_binding["binding_digest"],
        "d541801_scorer_source_revision": temporal_contract.REQUIRED_D541801_SCORER_REVISION,
        "d541801_scorer_source_sha256": temporal_contract.REQUIRED_D541801_SCORER_SHA256,
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "native_schedule_digest": temporal_contract.NATIVE_SCHEDULE_DIGEST,
        "adapter_loaded": False,
        "frozen": True,
    }
    labels_by_id = {row["candidate_id"]: row for row in labels["rows"]}
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in group_rows:
        by_cell.setdefault(row["candidate"]["calibration_group_id"], []).append(row)
    episodes: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    episode_file_bindings: list[dict[str, Any]] = []
    arm_file_bindings: list[dict[str, Any]] = []
    parity_done = False
    for cell_id, rows in by_cell.items():
        if len(rows) != len(data_contract.SEMANTIC_BRANCHES):
            raise Core4HiddenMaterializationError("cell owner closure differs")
        first_gaussian_cpu = frozen._load_exact81_tensor(
            rows[0]["artifacts"]["official_initial_gaussian"],
            key="official_initial_gaussian",
            label=f"{cell_id} official Gaussian",
        )
        first_gaussian_identity = frozen.verify_native_tensor_value_identity(
            first_gaussian_cpu,
            rows[0]["artifacts"]["official_initial_gaussian"],
            label=f"{cell_id} official Gaussian",
        )
        first_gaussian_geometry = derive_core4_native_geometry(
            first_gaussian_identity["shape"]
        )
        if first_gaussian_geometry != derive_core4_native_geometry(
            rows[0]["artifacts"]["official_initial_gaussian"].get("shape")
        ):
            raise Core4HiddenMaterializationError(
                "official Gaussian tensor/receipt native geometry differs"
            )
        epsilon = first_gaussian_cpu.to(device=device).contiguous()
        epsilon_sha = frozen.tensor_sha256(epsilon)
        clean_by_id: dict[str, Any] = {}
        clean_authentication_by_id: dict[str, dict[str, Any]] = {}
        evidence: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            gaussian = (
                first_gaussian_cpu
                if row_index == 0
                else frozen._load_exact81_tensor(
                    row["artifacts"]["official_initial_gaussian"],
                    key="official_initial_gaussian",
                    label=f"{row['candidate']['candidate_id']} official Gaussian",
                )
            )
            identity = frozen.verify_native_tensor_value_identity(
                gaussian,
                row["artifacts"]["official_initial_gaussian"],
                label=f"{row['candidate']['candidate_id']} official Gaussian",
            )
            gaussian_geometry = derive_core4_native_geometry(identity["shape"])
            declared_gaussian_geometry = derive_core4_native_geometry(
                row["artifacts"]["official_initial_gaussian"].get("shape")
            )
            if identity != first_gaussian_identity or not torch.equal(
                gaussian, first_gaussian_cpu
            ) or gaussian_geometry != declared_gaussian_geometry or (
                gaussian_geometry != first_gaussian_geometry
            ):
                raise Core4HiddenMaterializationError(
                    "same-cell official Gaussian values/native geometry differ"
                )
            clean_cpu = frozen._load_exact81_tensor(
                row["artifacts"]["predecode_clean_latent"],
                key="normalized_clean_latent",
                label=f"{row['candidate']['candidate_id']} clean latent",
            )
            clean_identity = verify_authenticated_native_clean_tensor_identity(
                clean_cpu,
                row["artifacts"]["predecode_clean_latent"],
                label=f"{row['candidate']['candidate_id']} clean latent",
                frozen=frozen,
            )
            clean_geometry = derive_core4_native_geometry(clean_identity["shape"])
            declared_clean_geometry = derive_core4_native_geometry(
                row["artifacts"]["predecode_clean_latent"].get("shape")
            )
            if (
                clean_geometry != declared_clean_geometry
                or clean_geometry != first_gaussian_geometry
            ):
                raise Core4HiddenMaterializationError(
                    "clean/Gaussian tensor-and-receipt native geometry differs"
                )
            candidate_id = row["candidate"]["candidate_id"]
            clean_by_id[candidate_id] = clean_cpu
            clean_authentication_by_id[candidate_id] = clean_identity
            evidence.append(
                _candidate_evidence(
                    row,
                    labels_by_id[candidate_id],
                    bank_receipt_digest=bank["receipt_digest"],
                    official_gaussian_tensor_sha256=epsilon_sha,
                    clean_latent_tensor_sha256=frozen.tensor_sha256(clean_cpu),
                    authenticated_clean_latent_shape=clean_identity["shape"],
                )
            )
            if row_index != 0:
                del gaussian
        try:
            episode = data_contract.build_episode_plan(
                evidence, usage_authority=authority
            )
        except data_contract.LatentTemporalEventDatasetError as error:
            raise Core4HiddenMaterializationError(str(error)) from error
        if episode.get("native_geometry") != first_gaussian_geometry.as_dict():
            raise Core4HiddenMaterializationError(
                "episode plan differs from authenticated tensor/receipt geometry"
            )
        sketch = critic_core.make_fixed_spatial_sketch(
            patch_positions=first_gaussian_geometry.patch_positions,
            coordinates=SPATIAL_SKETCH_COORDINATES,
            seed=SPATIAL_SKETCH_SEED,
        ).to(device=device)
        sketch_receipt_binding = spatial_sketch_binding(
            sketch,
            geometry=first_gaussian_geometry,
            critic_core=critic_core,
        )
        observer = Block15FixedSketchPairObserver(
            transformer,
            sketch,
            sp_rank=distributed.rank,
            geometry=first_gaussian_geometry,
        )
        if parity_done:
            observer.install()
        episodes.append(episode)
        episode_dir = output / cell_id
        episode_binding: list[Any] = [None]
        if distributed.rank == 0:
            episode_dir.mkdir()
            episode_path = episode_dir / EPISODE_FILENAME
            episode_file_sha = _write_json_create_only(episode_path, episode)
            episode_binding[0] = {
                "path": str(episode_path),
                "file_sha256": episode_file_sha,
                "receipt_digest": episode["receipt_digest"],
                "episode_id": episode["episode_id"],
            }
        dist.broadcast_object_list(episode_binding, src=0)
        if not isinstance(episode_binding[0], Mapping):
            raise Core4HiddenMaterializationError("rank-zero episode binding differs")
        episode_file_bindings.append(dict(episode_binding[0]))
        conditions, prompt_binding = _encode_cell_prompts(
            rows=rows,
            renderer=renderer,
            tokenizer=tokenizer,
            device=device,
            frozen=frozen,
            prompt_clean=prompt_clean,
        )
        for ordinal, arm in enumerate(episode["arms"]):
            clean = clean_by_id[arm["source_candidate_id"]].to(device=device).contiguous()
            transformed = critic_core.apply_registered_temporal_transform(
                clean, arm["temporal_transform"]
            ).detach().contiguous()
            target = (epsilon - transformed).float().contiguous().detach()
            sigma = torch.tensor([NATIVE_SIGMA], dtype=torch.float32, device=device)
            x_sigma = (
                transformed + sigma.reshape(1, 1, 1, 1, 1) * target
            ).float().contiguous().detach()
            transformed_sha = frozen.tensor_sha256(transformed)
            x_sigma_sha = frozen.tensor_sha256(x_sigma)
            parity: Optional[dict[str, Any]] = None
            off_pair: Optional[tuple[Any, Any]] = None
            if not parity_done:
                off_action, off_noop, off_proof = frozen_pair.forward_native_prompt_pair(
                    diffusion=diffusion,
                    transformer=transformer,
                    x_sigma=x_sigma,
                    native_schedule_index=NATIVE_SCHEDULE_INDEX,
                    action_condition=conditions["target_action"],
                    noop_condition=conditions["noop"],
                )
                off_pair = (off_action, off_noop)
                observer.install()
            observer.begin_pair()
            try:
                action_velocity, noop_velocity, same_state_proof = (
                    frozen_pair.forward_native_prompt_pair(
                        diffusion=diffusion,
                        transformer=transformer,
                        x_sigma=x_sigma,
                        native_schedule_index=NATIVE_SCHEDULE_INDEX,
                        action_condition=conditions["target_action"],
                        noop_condition=conditions["noop"],
                    )
                )
                capture = observer.finish_pair(dist)
            except Exception:
                if observer._active:
                    observer.abort_pair()
                raise
            if off_pair is not None:
                parity = {
                    "episode_id": cell_id,
                    "arm_role": arm["role"],
                    "x_sigma_tensor_sha256": x_sigma_sha,
                    "hook_off_same_state_proof": off_proof,
                    "hook_on_same_state_proof": same_state_proof,
                    "action_velocity_hook_off_sha256": frozen.tensor_sha256(
                        off_pair[0].float()
                    ),
                    "action_velocity_hook_on_sha256": frozen.tensor_sha256(
                        action_velocity.float()
                    ),
                    "noop_velocity_hook_off_sha256": frozen.tensor_sha256(
                        off_pair[1].float()
                    ),
                    "noop_velocity_hook_on_sha256": frozen.tensor_sha256(
                        noop_velocity.float()
                    ),
                    "action_velocity_byte_equal": torch.equal(
                        off_pair[0], action_velocity
                    ),
                    "noop_velocity_byte_equal": torch.equal(off_pair[1], noop_velocity),
                    "hook_returned_replacement_tensor": False,
                }
                if (
                    parity["action_velocity_byte_equal"] is not True
                    or parity["noop_velocity_byte_equal"] is not True
                ):
                    raise Core4HiddenMaterializationError(
                        "block.15 hook-on/off prediction parity failed"
                    )
                parity_done = True
                del off_pair, off_action, off_noop
            residual = (capture.action_sketch - capture.noop_sketch).float().detach()
            arm_dir = output / cell_id / f"{ordinal:02d}-{arm['role']}"
            residual_binding = _save_residual(
                output=arm_dir,
                residual=residual,
                dist=dist,
                critic_core=critic_core,
            )
            action_sketch_digest = critic_core._canonical_tensor_digest(
                capture.action_sketch, label="action sketched hidden"
            )
            noop_sketch_digest = critic_core._canonical_tensor_digest(
                capture.noop_sketch, label="noop sketched hidden"
            )
            receipt = _make_arm_receipt(
                group_id=args.group_id,
                episode=episode,
                arm=arm,
                prompt_binding=prompt_binding,
                x_sigma_sha256=x_sigma_sha,
                transformed_clean_sha256=transformed_sha,
                clean_latent_authentication=clean_authentication_by_id[
                    arm["source_candidate_id"]
                ],
                effective_gaussian_sha256=epsilon_sha,
                capture=capture,
                action_sketch_digest=action_sketch_digest,
                noop_sketch_digest=noop_sketch_digest,
                residual_binding=residual_binding,
                same_state_proof=same_state_proof,
                native_geometry=first_gaussian_geometry,
                fixed_spatial_sketch=sketch_receipt_binding,
                model_binding=model_binding,
                source_binding=source_binding,
                hook_parity=parity,
            )
            validate_arm_receipt(receipt)
            receipt_digests: list[Any] = [None] * 4
            dist.all_gather_object(receipt_digests, receipt["receipt_digest"])
            if len(set(receipt_digests)) != 1:
                raise Core4HiddenMaterializationError("SP4 arm receipts differ")
            arm_binding: list[Any] = [None]
            if distributed.rank == 0:
                arm_receipt_path = arm_dir / ARM_RECEIPT_FILENAME
                arm_receipt_file_sha = _write_json_create_only(
                    arm_receipt_path, receipt
                )
                arm_binding[0] = {
                    "path": str(arm_receipt_path),
                    "file_sha256": arm_receipt_file_sha,
                    "receipt_digest": receipt["receipt_digest"],
                    "episode_id": episode["episode_id"],
                    "arm_role": arm["role"],
                    "residual_path": residual_binding["path"],
                    "residual_file_sha256": residual_binding["sha256"],
                }
            dist.broadcast_object_list(arm_binding, src=0)
            if not isinstance(arm_binding[0], Mapping):
                raise Core4HiddenMaterializationError(
                    "rank-zero arm receipt binding differs"
                )
            arm_file_bindings.append(dict(arm_binding[0]))
            receipts.append(receipt)
            del (
                clean,
                transformed,
                target,
                sigma,
                x_sigma,
                action_velocity,
                noop_velocity,
                capture,
                residual,
            )
        observer.remove()
        del (
            clean_by_id,
            evidence,
            epsilon,
            first_gaussian_cpu,
            conditions,
            observer,
            sketch,
        )
    try:
        population_audit = data_contract.audit_episode_population(
            episodes, protocol="core4_pilot"
        )
    except data_contract.LatentTemporalEventDatasetError as error:
        raise Core4HiddenMaterializationError(str(error)) from error
    # A single group has one fit and one confirmation cell per family subset;
    # only the merged four-cell population may become pilot-training eligible.
    if population_audit["editor_optimizer_authorized"] is not False:
        raise Core4HiddenMaterializationError("pilot audit exceeded editor authority")
    try:
        freeze_after = native_generation.source_audit.model_freeze_certificate(renderer)
    except Exception as error:
        raise Core4HiddenMaterializationError(str(error)) from error
    if freeze_after != freeze_certificate or any(
        parameter.requires_grad for parameter in renderer.parameters()
    ):
        raise Core4HiddenMaterializationError("frozen renderer changed during queries")
    group_receipt = make_group_receipt(
        group_id=args.group_id,
        episodes=episodes,
        arm_receipts=receipts,
        episode_file_bindings=episode_file_bindings,
        arm_file_bindings=arm_file_bindings,
        population_audit=population_audit,
        bank_receipt_digest=bank["receipt_digest"],
        root_spec_raw_sha256=args.expected_root_spec_sha256,
        label_binding=label_binding,
        usage_binding=usage_binding,
        model_binding=model_binding,
        source_binding=source_binding,
    )
    group_digests: list[Any] = [None] * 4
    dist.all_gather_object(group_digests, group_receipt["receipt_digest"])
    if len(set(group_digests)) != 1:
        raise Core4HiddenMaterializationError("SP4 group receipts differ")
    if distributed.rank == 0:
        _write_json_create_only(output / GROUP_FILENAME, group_receipt)
        os.chmod(output, 0o500)
    dist.barrier()
    dist.destroy_process_group()
    del spec
    return 0


def _run_merge(args: argparse.Namespace) -> int:
    _validate_common_args(args)
    output = _plain_directory(args.output_root, label="population output root")
    # Merge is intentionally CPU/JSON-only.  It authenticates the externally
    # supplied roots and byte bindings but never imports Bernini or Torch.
    root_spec = _plain_file(args.root_spec, label="root spec")
    bank_receipt = _plain_file(args.bank_receipt, label="bank receipt")
    labels = _plain_file(
        args.detached_event_label_manifest, label="detached event-label manifest"
    )
    authority = _plain_file(args.critic_use_authority, label="critic-use authority")
    for path, expected, label in (
        (root_spec, args.expected_root_spec_sha256, "root spec"),
        (bank_receipt, args.expected_bank_receipt_sha256, "bank receipt"),
        (
            labels,
            args.expected_detached_event_label_manifest_sha256,
            "detached event-label manifest",
        ),
        (authority, args.expected_critic_use_authority_sha256, "critic-use authority"),
    ):
        if file_sha256(path) != expected:
            raise Core4HiddenMaterializationError(f"merge {label} SHA-256 differs")
    receipts = []
    episodes: list[dict[str, Any]] = []
    for group_id in GROUP_IDS:
        path = output / group_id / GROUP_FILENAME
        raw, _resolved, _sha = _read_strict_json(
            path,
            label=f"{group_id} hidden-query group receipt",
            expected_sha256=file_sha256(path),
        )
        group = validate_group_receipt(raw)
        receipts.append(group)
        for binding in group["episode_file_bindings"]:
            episode_path = _plain_descendant_file(
                binding["path"], root=output, label="episode plan"
            )
            if file_sha256(episode_path) != binding["file_sha256"]:
                raise Core4HiddenMaterializationError("episode plan file changed")
            episode_raw, _episode_resolved, _episode_sha = _read_strict_json(
                episode_path,
                label="episode plan",
                expected_sha256=binding["file_sha256"],
            )
            try:
                episode = data_contract._validate_episode_surface(episode_raw)
            except data_contract.LatentTemporalEventDatasetError as error:
                raise Core4HiddenMaterializationError(str(error)) from error
            if (
                episode["receipt_digest"] != binding["receipt_digest"]
                or episode["episode_id"] != binding["episode_id"]
                or episode.get("native_geometry")
                != group["native_geometries_by_episode"].get(
                    episode["episode_id"]
                )
            ):
                raise Core4HiddenMaterializationError(
                    "episode plan binding differs"
                )
            episodes.append(episode)
        for binding in group["arm_file_bindings"]:
            arm_path = _plain_descendant_file(
                binding["path"], root=output, label="arm receipt"
            )
            if file_sha256(arm_path) != binding["file_sha256"]:
                raise Core4HiddenMaterializationError("arm receipt file changed")
            arm_raw, _arm_resolved, _arm_sha = _read_strict_json(
                arm_path,
                label="arm receipt",
                expected_sha256=binding["file_sha256"],
            )
            arm = validate_arm_receipt(arm_raw)
            residual = arm["residual_artifact"]
            residual_path = _plain_descendant_file(
                binding["residual_path"], root=output, label="residual artifact"
            )
            if (
                arm["receipt_digest"] != binding["receipt_digest"]
                or arm["episode_id"] != binding["episode_id"]
                or arm["arm_role"] != binding["arm_role"]
                or residual["path"] != str(residual_path)
                or binding["residual_path"] != str(residual_path)
                or residual["sha256"] != binding["residual_file_sha256"]
                or file_sha256(residual_path) != binding["residual_file_sha256"]
                or arm.get("native_geometry")
                != group["native_geometries_by_episode"].get(arm["episode_id"])
                or arm.get("fixed_spatial_sketch")
                != group["spatial_sketches_by_episode"].get(arm["episode_id"])
            ):
                raise Core4HiddenMaterializationError(
                    "arm/residual artifact binding differs"
                )
    try:
        population_audit = data_contract.audit_episode_population(
            episodes, protocol="core4_pilot"
        )
    except data_contract.LatentTemporalEventDatasetError as error:
        raise Core4HiddenMaterializationError(str(error)) from error
    population = make_population_receipt(
        receipts, output_root=output, population_audit=population_audit
    )
    if population["source_binding"] != _source_binding(args):
        raise Core4HiddenMaterializationError("population source binding differs")
    _write_json_create_only(output / POPULATION_FILENAME, population)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "group":
        return _run_group(args)
    if args.command == "merge":
        return _run_merge(args)
    raise Core4HiddenMaterializationError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_RECEIPT_FILENAME",
    "ARM_SCHEMA",
    "ARMS_PER_CELL",
    "Block15FixedSketchPairObserver",
    "CORE4_ARMS",
    "CORE4_SPATIAL_SKETCH_DIGESTS",
    "Core4HiddenMaterializationError",
    "GROUP_FILENAME",
    "GROUP_IDS",
    "GROUP_SCHEMA",
    "HOOK_COORDINATE",
    "MODEL_FORWARDS_PER_GROUP",
    "NativeEpisodeGeometry",
    "POPULATION_FILENAME",
    "POPULATION_SCHEMA",
    "RESIDUAL_FILENAME",
    "TOTAL_MODEL_FORWARDS_PER_GROUP",
    "build_parser",
    "canonical_json_bytes",
    "derive_core4_native_geometry",
    "make_group_receipt",
    "make_local_target_index_plan",
    "make_population_receipt",
    "object_sha256",
    "pure_python_spatial_sketch_digest",
    "spatial_sketch_binding",
    "validate_sp4_contiguous_layouts",
    "validate_spatial_sketch_binding",
    "validate_arm_receipt",
    "validate_group_receipt",
]

#!/usr/bin/env python3
"""Frozen V-JEPA2 ordered contextual features for exact644 action anchors.

This is a burned-development feature materializer, not a video editor.  Each
exact81 anchor is decoded in display/PTS order, sampled to one canonical
64-frame processor tensor, and transformed only along that tensor's temporal
axis.  The original, monotone warp, reverse, block shuffle, and phase swap are
then sent through five separate frozen-backbone calls.  Token outputs are
never permuted after the backbone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import torch


MANIFEST_SCHEMA = "semantic-moments-action-reward-audit-v1"
FEATURE_SCHEMA = "vjepa2-ordered-contextual-anchor-shard-v4c"
RECEIPT_SCHEMA = "vjepa2-ordered-contextual-exact644-receipt-v4c"
SOURCE_MANIFEST_DIGEST = (
    "96fe6188ad0f5ee72dcd89fbc018835f3f2995e45ff116f07449e863fa9b51d5"
)
SOURCE_MANIFEST_FILE_SHA256 = (
    "61da995eb680b9fba7ab3b7d3b6041c7b51c7e95253c74e607ddab6fdd6a61aa"
)
FEATURE_MANIFEST_SHA256 = (
    "963cc02f9875048120fbea042ecbeac9b59e5e40d23121c52a9d2488556ca4e5"
)
FEATURE_MANIFEST_DIGEST = (
    "51480da9c13f1dd060bead9309badf12c59a05810a83c1f6eab643f62418687a"
)
MODEL_REPO = "facebook/vjepa2-vitl-fpc64-256"
MODEL_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
MODEL_FILES = {
    "config.json": {
        "sha256": "3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9",
        "size_bytes": 785,
    },
    "model.safetensors": {
        "sha256": "25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b",
        "size_bytes": 1_303_947_864,
    },
    "video_preprocessor_config.json": {
        "sha256": "d2fab4418fc0390b62c4cd72ade56908a7929f80c62288adbe10dd8d23421227",
        "size_bytes": 1_298,
    },
}
TRANSFORMERS_VERSION = "5.5.4"
TRANSFORMERS_MODULES = {
    "transformers.models.vjepa2.modeling_vjepa2": (
        "08613fb28cf11f4c46f234d9681a954fd12bd93b2f8b67229e5721974e0632fc"
    ),
    "transformers.models.vjepa2.video_processing_vjepa2": (
        "1008567f52c0afc585e5cbbf40d1b7e166ba8039c33fffe3c8676b67e6d1a453"
    ),
}
BASE64_INDICES_SHA256 = (
    "28d03d93dc7a23114435e90af5a75a85ffae610ae2d69b88a7dc5e4117aebb2a"
)
WARP64_COORDINATES_SHA256 = (
    "667111bcba955ff76fb5a72adcc2d18deed6e3ab5f0ae86596a67fe67b5c4b81"
)
WARP32_COORDINATES = (
    0.000000000, 1.241935484, 2.467741935, 3.677419355,
    4.870967742, 6.048387097, 7.209677419, 8.354838710,
    9.483870968, 10.596774194, 11.693548387, 12.774193548,
    13.838709677, 14.887096774, 15.919354839, 16.935483871,
    17.935483871, 18.919354839, 19.887096774, 20.838709677,
    21.774193548, 22.693548387, 23.596774194, 24.483870968,
    25.354838710, 26.209677419, 27.048387097, 27.870967742,
    28.677419355, 29.467741935, 30.241935484, 31.000000000,
)
PHASE_BLOCK_PERMUTATION = (0, 1, 4, 5, 2, 3, 6, 7)
VIEW_NAMES = (
    "original", "monotone_warp", "reverse", "block_shuffle", "phase_swap"
)
SEED = 20260819
SHA_RE = re.compile(r"[0-9a-f]{64}")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_binding() -> dict[str, Any]:
    source = Path(__file__).resolve(strict=True)
    return {
        "path": str(source), "realpath": str(source.resolve(strict=True)),
        "sha256": file_sha256(source), "size_bytes": source.stat().st_size,
    }


def tensor_sha256(value: torch.Tensor) -> str:
    if type(value) is not torch.Tensor:
        raise TypeError("tensor digest requires an exact Tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(_canonical({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def base64_indices() -> torch.Tensor:
    indices = torch.tensor([(80 * k) // 63 for k in range(64)], dtype=torch.long)
    if (
        indices.tolist()[0] != 0 or indices.tolist()[-1] != 80
        or len(set(indices.tolist())) != 64
        or not bool((indices[1:] > indices[:-1]).all())
        or tensor_sha256(indices) != BASE64_INDICES_SHA256
    ):
        raise RuntimeError("exact81-to-64 sampling ABI differs")
    return indices


def warp64_coordinates() -> torch.Tensor:
    # The order is contractual: round WARP32 to FP32 first, then multiply/add.
    base = torch.tensor(WARP32_COORDINATES, dtype=torch.float32)
    coordinates = torch.stack((2.0 * base, 2.0 * base + 1.0), dim=1).reshape(64)
    if (
        float(coordinates[0]) != 0.0 or float(coordinates[-1]) != 63.0
        or not bool((coordinates[1:] > coordinates[:-1]).all())
        or tensor_sha256(coordinates) != WARP64_COORDINATES_SHA256
    ):
        raise RuntimeError("64-frame monotone-warp ABI differs")
    return coordinates


def block_permutation(iid: str) -> tuple[int, ...]:
    if type(iid) is not str or not iid:
        raise ValueError("block-shuffle IID differs")
    ordered = sorted(
        range(8),
        key=lambda block: hashlib.sha256(
            f"v4a-block-shuffle:{SEED}:{iid}:{block}".encode("utf-8")
        ).hexdigest(),
    )
    forbidden = {
        tuple(range(8)), tuple(reversed(range(8))), PHASE_BLOCK_PERMUTATION,
    }
    candidate = tuple(ordered)
    if candidate in forbidden or all(candidate[i] < candidate[i + 1] for i in range(7)):
        candidate = tuple(ordered[2:] + ordered[:2])
    if candidate in forbidden or set(candidate) != set(range(8)):
        candidate = (2, 0, 5, 1, 7, 3, 6, 4)
    if candidate in forbidden or set(candidate) != set(range(8)):
        raise RuntimeError("block-shuffle ABI differs")
    return candidate


def _lift32_indices(indices: Sequence[int]) -> torch.Tensor:
    if len(indices) != 32 or tuple(sorted(indices)) != tuple(range(32)):
        raise ValueError("32-frame permutation differs")
    return torch.tensor(
        [2 * index + parity for index in indices for parity in (0, 1)],
        dtype=torch.long,
    )


def phase64_indices() -> torch.Tensor:
    frame32 = [4 * block + offset for block in PHASE_BLOCK_PERMUTATION for offset in range(4)]
    return _lift32_indices(frame32)


def block64_indices(iid: str) -> torch.Tensor:
    frame32 = [4 * block + offset for block in block_permutation(iid) for offset in range(4)]
    return _lift32_indices(frame32)


def pixel_views(canonical_pixels: torch.Tensor, iid: str) -> dict[str, torch.Tensor]:
    """Derive five [1,64,3,256,256] FP32 inputs only on temporal dim 1."""

    if (
        type(canonical_pixels) is not torch.Tensor
        or tuple(canonical_pixels.shape) != (1, 64, 3, 256, 256)
        or canonical_pixels.dtype != torch.float32
        or not bool(torch.isfinite(canonical_pixels).all())
    ):
        raise ValueError("canonical processor pixel tensor differs")
    device = canonical_pixels.device
    coordinates = warp64_coordinates().to(device)
    lower = coordinates.floor().to(torch.long)
    upper = coordinates.ceil().to(torch.long)
    weight = (coordinates - lower.to(coordinates.dtype)).reshape(1, 64, 1, 1, 1)
    monotone = canonical_pixels.index_select(1, lower) * (1.0 - weight)
    monotone = monotone + canonical_pixels.index_select(1, upper) * weight
    views = {
        "original": canonical_pixels,
        "monotone_warp": monotone.contiguous(),
        "reverse": canonical_pixels.flip(1).contiguous(),
        "block_shuffle": canonical_pixels.index_select(
            1, block64_indices(iid).to(device)
        ).contiguous(),
        "phase_swap": canonical_pixels.index_select(
            1, phase64_indices().to(device)
        ).contiguous(),
    }
    if tuple(views) != VIEW_NAMES or any(
        tuple(value.shape) != (1, 64, 3, 256, 256)
        or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
        for value in views.values()
    ):
        raise RuntimeError("view pixel tensor closure differs")
    return views


@dataclass(frozen=True)
class AnchorItem:
    ordinal: int
    iid: str
    family: str
    group_id: str
    instruction_sha256: str
    strict: bool
    path: Path
    media_sha256: str


def load_anchor_manifest(path: Path, expected_sha256: str) -> tuple[list[AnchorItem], dict[str, Any]]:
    requested = path
    if not requested.is_absolute() or requested.is_symlink():
        raise ValueError("manifest path must be absolute and non-symlink")
    path = requested.resolve(strict=True)
    if str(requested) != str(path):
        raise ValueError("manifest logical/resolved path differs")
    before = requested.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
        or before.st_size <= 0
        or _sha(expected_sha256, "manifest SHA") != FEATURE_MANIFEST_SHA256
    ):
        raise ValueError("manifest file envelope differs")
    with requested.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        raw = handle.read()
        digest_before = hashlib.sha256(raw).hexdigest()
        handle.seek(0)
        raw_after = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = requested.lstat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_nlink,
        value.st_mode, value.st_mtime_ns, value.st_ctime_ns,
    )
    if (
        digest_before != FEATURE_MANIFEST_SHA256
        or raw_after != raw
        or hashlib.sha256(raw_after).hexdigest() != FEATURE_MANIFEST_SHA256
        or identity(before) != identity(opened_before)
        or identity(opened_before) != identity(opened_after)
        or identity(opened_after) != identity(after)
    ):
        raise RuntimeError("manifest changed during single-FD load/readback")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest JSON differs") from error
    if (
        type(value) is not dict or value.get("schema_version") != MANIFEST_SCHEMA
        or value.get("formal_training_authorized") is not False
        or value.get("paired_ground_truth_claimed") is not False
        or value.get("manifest_digest") != FEATURE_MANIFEST_DIGEST
        or value.get("source_release", {}).get("sha256") != SOURCE_MANIFEST_FILE_SHA256
        or value.get("source_release", {}).get("manifest_digest") != SOURCE_MANIFEST_DIGEST
        or value.get("source_release", {}).get("row_count") != 644
        or value.get("counts", {}).get("total") != 1288
        or value.get("counts", {}).get("unique_base_clips") != 644
        or value.get("counts", {}).get("by_group") != {
            "exact644_action_anchor": 644, "exact644_source": 644,
        }
    ):
        raise ValueError("exact644 manifest authority differs")
    items = value.get("items")
    if type(items) is not list or len(items) != 1288:
        raise ValueError("manifest item closure differs")
    anchors: list[AnchorItem] = []
    seen_iids: set[str] = set()
    for source_ordinal, row in enumerate(items):
        metadata = row.get("metadata") if type(row) is dict else None
        if type(metadata) is not dict or metadata.get("role") != "action_anchor":
            continue
        iid = metadata.get("iid")
        media_path = row.get("path")
        if (
            type(iid) is not str or not iid or iid in seen_iids
            or row.get("item_id") != f"exact644:{iid}:action_anchor"
            or row.get("group") != "exact644_action_anchor"
            or type(media_path) is not str or not media_path.startswith("/")
            or metadata.get("source_manifest_digest") != SOURCE_MANIFEST_DIGEST
            or metadata.get("paired_ground_truth_claimed") is not False
            or type(metadata.get("family")) is not str or not metadata["family"]
            or type(metadata.get("strict_selection_gates_all_true")) is not bool
        ):
            raise ValueError("anchor manifest row differs")
        seen_iids.add(iid)
        anchors.append(AnchorItem(
            ordinal=len(anchors), iid=iid, family=metadata["family"],
            group_id=_sha(metadata.get("group_id"), "group ID"),
            instruction_sha256=_sha(metadata.get("instruction_sha256"), "instruction SHA"),
            strict=metadata["strict_selection_gates_all_true"],
            path=Path(media_path), media_sha256=_sha(row.get("sha256"), "media SHA"),
        ))
    if (
        len(anchors) != 644 or len(seen_iids) != 644
        or len({row.group_id for row in anchors}) != 644
        or len({row.family for row in anchors}) != 28
        or sum(row.strict for row in anchors) != 359
        or sum(not row.strict for row in anchors) != 285
    ):
        raise ValueError("manifest does not contain exact644 unique anchors")
    return anchors, value


def sealed_model_closure(model_root: Path) -> dict[str, Any]:
    requested = model_root
    if not requested.is_absolute() or requested.is_symlink():
        raise ValueError("model root must be an absolute non-symlink directory")
    root = requested.resolve(strict=True)
    root_stat = root.stat()
    if not root.is_dir() or stat.S_IMODE(root_stat.st_mode) != 0o555:
        raise ValueError("model root must be a mode0555 directory")
    children = list(root.iterdir())
    if {path.name for path in children} != set(MODEL_FILES) or any(
        path.is_symlink() or not path.is_file() for path in children
    ):
        raise ValueError("sealed model root must contain exact3 regular files")
    records = []
    for name, expected in MODEL_FILES.items():
        path = root / name
        value_stat = path.stat()
        actual = {
            "relative_path": name,
            "logical_path": str(path),
            "realpath": str(path.resolve(strict=True)),
            "sha256": file_sha256(path),
            "size_bytes": value_stat.st_size,
            "mode": stat.S_IMODE(value_stat.st_mode),
            "nlink": value_stat.st_nlink,
            "device": value_stat.st_dev,
            "inode": value_stat.st_ino,
        }
        if (
            actual["realpath"] != actual["logical_path"]
            or actual["sha256"] != expected["sha256"]
            or actual["size_bytes"] != expected["size_bytes"]
            or actual["mode"] != 0o444 or actual["nlink"] != 1
        ):
            raise ValueError(f"sealed V-JEPA2 file differs: {name}")
        records.append(actual)
    if len({(row["device"], row["inode"]) for row in records}) != 3:
        raise ValueError("sealed V-JEPA2 exact3 files alias storage")
    return {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "root": str(root),
        "root_realpath": str(root.resolve(strict=True)),
        "root_mode": stat.S_IMODE(root_stat.st_mode),
        "exact_top_level_regular_file_count": 3,
        "files": records,
        "closure_sha256": object_sha256(records),
    }


def transformers_module_closure() -> dict[str, Any]:
    transformers = importlib.import_module("transformers")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError("transformers version differs from frozen v4-C runtime")
    rows = []
    for module_name, expected_sha in TRANSFORMERS_MODULES.items():
        module = importlib.import_module(module_name)
        source = Path(module.__file__).resolve(strict=True)
        row = {
            "module": module_name,
            "source_path": str(source),
            "source_realpath": str(source.resolve(strict=True)),
            "sha256": file_sha256(source),
            "size_bytes": source.stat().st_size,
        }
        if row["sha256"] != expected_sha:
            raise RuntimeError(f"frozen transformers module differs: {module_name}")
        rows.append(row)
    return {
        "transformers_version": transformers.__version__,
        "modules": rows,
        "closure_sha256": object_sha256(rows),
    }


def decode_exact81_rgb(item: AnchorItem) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode all frames through PyAV in display order and prove monotone PTS."""

    import av

    if av.__version__ != "13.1.0":
        raise RuntimeError("PyAV version differs from frozen v4-C runtime")
    if not item.path.is_absolute() or item.path.is_symlink():
        raise ValueError("anchor media path must be absolute and non-symlink")
    media_path = item.path.resolve(strict=True)
    if str(item.path) != str(media_path):
        raise ValueError("anchor manifest logical/resolved path differs")
    arrays = []
    pts: list[int] = []
    time_bases: list[str] = []
    path_before = media_path.stat()
    with media_path.open("rb") as media_handle:
        opened_before = os.fstat(media_handle.fileno())
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or (opened_before.st_dev, opened_before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
        ):
            raise ValueError("anchor media open/path identity differs")
        digest = hashlib.sha256()
        for chunk in iter(lambda: media_handle.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != item.media_sha256:
            raise ValueError("anchor media file SHA differs")
        media_handle.seek(0)
        with av.open(media_handle, mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise ValueError("anchor must have exactly one video stream")
            stream = streams[0]
            rate = stream.average_rate
            if (
                rate is None or Fraction(rate) != Fraction(25, 1)
                or int(stream.frames) != 81
            ):
                raise ValueError("anchor stream is not exact81 at 25 fps")
            for frame in container.decode(video=0):
                if type(frame.pts) is not int or frame.time_base is None:
                    raise ValueError("decoded frame lacks integral PTS/time base")
                array = frame.to_ndarray(format="rgb24")
                tensor = torch.from_numpy(array).clone()
                if (
                    tensor.ndim != 3 or tensor.shape[-1] != 3
                    or tensor.dtype != torch.uint8
                ):
                    raise ValueError("decoded RGB24 frame geometry/dtype differs")
                arrays.append(tensor)
                pts.append(frame.pts)
                time_bases.append(str(frame.time_base))
            stream_geometry = {
                "width": int(stream.width), "height": int(stream.height),
                "average_rate": f"{rate.numerator}/{rate.denominator}",
                "stream_frames_metadata": int(stream.frames),
            }
        opened_after = os.fstat(media_handle.fileno())
        media_handle.seek(0)
        digest_after = hashlib.sha256()
        for chunk in iter(lambda: media_handle.read(1024 * 1024), b""):
            digest_after.update(chunk)
    path_after = media_path.stat()
    if (
        len(arrays) != 81 or len(pts) != 81
        or any(right <= left for left, right in zip(pts, pts[1:]))
        or len(set(time_bases)) != 1
        or any(tuple(array.shape) != tuple(arrays[0].shape) for array in arrays)
        or any(
            Fraction(right - left) * Fraction(time_bases[0]) != Fraction(1, 25)
            for left, right in zip(pts, pts[1:])
        )
        or digest_after.hexdigest() != item.media_sha256
        or (
            opened_before.st_dev, opened_before.st_ino, opened_before.st_size,
            opened_before.st_mtime_ns, opened_before.st_ctime_ns,
        ) != (
            opened_after.st_dev, opened_after.st_ino, opened_after.st_size,
            opened_after.st_mtime_ns, opened_after.st_ctime_ns,
        )
        or (
            path_before.st_dev, path_before.st_ino, path_before.st_size,
            path_before.st_mtime_ns, path_before.st_ctime_ns,
        ) != (
            path_after.st_dev, path_after.st_ino, path_after.st_size,
            path_after.st_mtime_ns, path_after.st_ctime_ns,
        )
    ):
        raise ValueError("anchor decode is not exact81 monotone display/PTS order")
    video = torch.stack(arrays).contiguous()
    return video, {
        **stream_geometry,
        "manifest_logical_path": str(item.path),
        "resolved_path": str(media_path),
        "logical_equals_resolved_path": str(item.path) == str(media_path),
        "decoder": "PyAV",
        "pyav_version": av.__version__,
        "decoded_frame_count": 81,
        "decoded_display_order_is_iteration_order": True,
        "all_pts_integral": True,
        "pts_strictly_increasing": True,
        "every_pts_delta_is_exactly_one_over_25_seconds": True,
        "pts": pts,
        "pts_sha256": object_sha256(pts),
        "single_time_base": time_bases[0],
        "decoded_exact81_rgb24_sha256": tensor_sha256(video),
    }


def pool_time_major_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """Reshape frozen encoder tokens only; never permute/reindex token output."""

    if (
        type(hidden) is not torch.Tensor
        or tuple(hidden.shape) != (1, 8192, 1024)
        or hidden.dtype != torch.float16
        or not bool(torch.isfinite(hidden).all())
    ):
        raise ValueError("V-JEPA2 last_hidden_state geometry differs")
    # Conv3d flatten(2).transpose(1,2) is time-major for the pinned module.
    grid = hidden.detach().to(dtype=torch.float32).reshape(1, 32, 256, 1024)
    pooled = grid.mean(dim=2).to(device="cpu").contiguous()
    if tuple(pooled.shape) != (1, 32, 1024) or not bool(torch.isfinite(pooled).all()):
        raise RuntimeError("V-JEPA2 time-major spatial pooling differs")
    return pooled


class FrozenVJepa2:
    def __init__(self, model_root: Path, device_name: str):
        if (
            device_name != "cuda:0" or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != "AMD Instinct MI210"
        ):
            raise RuntimeError("v4-C extraction requires exactly one logical MI210 cuda:0")
        if torch.__version__ != "2.7.1+rocm6.3":
            raise RuntimeError("torch/ROCm runtime differs from frozen v4-C environment")
        if type(torch.version.hip) is not str or not torch.version.hip.startswith("6.3"):
            raise RuntimeError("ROCm HIP runtime differs from frozen v4-C environment")
        if {
            "SLURM_NTASKS": os.environ.get("SLURM_NTASKS"),
            "SLURM_NNODES": os.environ.get("SLURM_NNODES"),
            "SLURM_PROCID": os.environ.get("SLURM_PROCID"),
            "SLURM_LOCALID": os.environ.get("SLURM_LOCALID"),
        } != {
            "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
            "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
        }:
            raise RuntimeError("v4-C shard must run in one exact Slurm task/node")
        if os.environ.get("TRANSFORMERS_OFFLINE") != "1" or os.environ.get("HF_HUB_OFFLINE") != "1":
            raise RuntimeError("V-JEPA2 loading must be explicitly offline")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True)
        self.device = torch.device(device_name)
        self.device_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
        if not self.device_uuid:
            raise RuntimeError("MI210 logical cuda:0 UUID is unavailable")
        self.model_root = model_root.resolve(strict=True)
        self.model_closure_before = sealed_model_closure(model_root)
        self.module_closure_before = transformers_module_closure()
        from transformers import AutoModel, AutoVideoProcessor

        self.processor = AutoVideoProcessor.from_pretrained(
            str(self.model_root), local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            str(self.model_root), local_files_only=True, dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(self.device).requires_grad_(False).eval()
        parameters = list(self.model.parameters())
        if (
            not parameters or any(parameter.requires_grad for parameter in parameters)
            or any(
                parameter.is_floating_point() and parameter.dtype != torch.float16
                for parameter in parameters
            )
        ):
            raise RuntimeError("V-JEPA2 backbone is not frozen FP16")
        config = self.model.config
        if (
            getattr(config, "frames_per_clip", None) != 64
            or getattr(config, "hidden_size", None) != 1024
            or getattr(config, "image_size", None) != 256
            or getattr(config, "patch_size", None) != 16
            or getattr(config, "tubelet_size", None) != 2
        ):
            raise RuntimeError("V-JEPA2 model config geometry differs")
        if sealed_model_closure(model_root) != self.model_closure_before:
            raise RuntimeError("sealed model closure changed while loading")
        if transformers_module_closure() != self.module_closure_before:
            raise RuntimeError("transformers source closure changed while loading")
        self.processor_call_count = 0
        self.forward_call_count = 0

    def process_canonical_base64(self, sampled_rgb: torch.Tensor) -> torch.Tensor:
        if (
            type(sampled_rgb) is not torch.Tensor
            or sampled_rgb.dtype != torch.uint8
            or sampled_rgb.ndim != 4 or sampled_rgb.shape[0] != 64
            or sampled_rgb.shape[-1] != 3
        ):
            raise ValueError("canonical base64 decoded RGB tensor differs")
        from PIL import Image

        pil_frames = [Image.fromarray(frame.numpy(), mode="RGB") for frame in sampled_rgb]
        self.processor_call_count += 1
        batch = self.processor(videos=pil_frames, return_tensors="pt")
        if set(batch) != {"pixel_values_videos"}:
            raise RuntimeError("V-JEPA2 processor output key closure differs")
        pixels = batch["pixel_values_videos"]
        if (
            tuple(pixels.shape) != (1, 64, 3, 256, 256)
            or pixels.dtype != torch.float32
            or not bool(torch.isfinite(pixels).all())
        ):
            raise RuntimeError("V-JEPA2 canonical processor tensor differs")
        return pixels.contiguous()

    @torch.inference_mode()
    def forward_view(self, pixels: torch.Tensor) -> torch.Tensor:
        if tuple(pixels.shape) != (1, 64, 3, 256, 256) or pixels.dtype != torch.float32:
            raise ValueError("V-JEPA2 model view input differs")
        self.forward_call_count += 1
        outputs = self.model(
            pixel_values_videos=pixels.to(self.device), skip_predictor=True,
        )
        hidden = outputs.last_hidden_state
        return pool_time_major_hidden(hidden)

    def final_closure(self) -> dict[str, Any]:
        model_after = sealed_model_closure(self.model_root)
        module_after = transformers_module_closure()
        if model_after != self.model_closure_before or module_after != self.module_closure_before:
            raise RuntimeError("V-JEPA2 file/source closure changed during extraction")
        return {
            "model_files_before_and_after_exact": True,
            "transformers_modules_before_and_after_exact": True,
            "model": model_after,
            "transformers": module_after,
        }


def extract_one(item: AnchorItem, frozen: FrozenVJepa2) -> dict[str, Any]:
    processor_before = frozen.processor_call_count
    forward_before = frozen.forward_call_count
    decoded, media = decode_exact81_rgb(item)
    indices = base64_indices()
    selected = decoded.index_select(0, indices).contiguous()
    canonical_pixels = frozen.process_canonical_base64(selected)
    views = pixel_views(canonical_pixels, item.iid)
    sequences: dict[str, torch.Tensor] = {}
    view_receipts: dict[str, Any] = {}
    for forward_ordinal, name in enumerate(VIEW_NAMES, start=1):
        pixels = views[name]
        sequence = frozen.forward_view(pixels)[0]
        sequences[name] = sequence
        view_receipts[name] = {
            "forward_ordinal_within_anchor": forward_ordinal,
            "model_input_pixel_values_videos_sha256": tensor_sha256(pixels),
            "model_input_shape": list(pixels.shape),
            "model_input_dtype": str(pixels.dtype),
            "last_hidden_state_shape": [1, 8192, 1024],
            "token_output_permuted_or_reindexed": False,
            "time_major_reshape": [1, 32, 256, 1024],
            "spatial_mean_output_shape": [32, 1024],
            "ordered_contextual_sequence_sha256": tensor_sha256(sequence),
        }
    if (
        frozen.processor_call_count - processor_before != 1
        or frozen.forward_call_count - forward_before != 5
        or set(sequences) != set(VIEW_NAMES)
        or any(
            tuple(value.shape) != (32, 1024) or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
            for value in sequences.values()
        )
    ):
        raise RuntimeError("v4-C exact one-processor/five-forward contract differs")
    return {
        "ordinal": item.ordinal,
        "iid": item.iid,
        "family": item.family,
        "group_id": item.group_id,
        "instruction_sha256": item.instruction_sha256,
        "strict_selection_gates_all_true": item.strict,
        "role": "action_anchor",
        "media_sha256": item.media_sha256,
        "media": media,
        "exact81_to_base64_indices": base64_indices().tolist(),
        "exact81_to_base64_indices_sha256": tensor_sha256(base64_indices()),
        "selected_base64_rgb24_sha256": tensor_sha256(selected),
        "block64_indices": block64_indices(item.iid).tolist(),
        "block64_indices_sha256": tensor_sha256(block64_indices(item.iid)),
        "phase64_indices": phase64_indices().tolist(),
        "phase64_indices_sha256": tensor_sha256(phase64_indices()),
        "canonical_processor_call_count": 1,
        "independent_frozen_backbone_forward_count": 5,
        "model_forward_batching_across_views": False,
        "view_order": list(VIEW_NAMES),
        "view_receipts": view_receipts,
        "view_sequences": sequences,
        "post_backbone_token_permutation_used": False,
    }


def _shard_semantic_sha256(value: Mapping[str, Any]) -> str:
    if type(value) is not dict or type(value.get("records")) is not list:
        raise ValueError("shard semantic digest payload differs")
    top = dict(value)
    rows = []
    for record in top.pop("records"):
        if type(record) is not dict or type(record.get("view_sequences")) is not dict:
            raise ValueError("shard semantic digest record differs")
        row = dict(record)
        sequences = row.pop("view_sequences")
        row["view_sequence_tensor_sha256"] = {
            name: tensor_sha256(sequences[name]) for name in VIEW_NAMES
        }
        rows.append(row)
    top["records"] = rows
    return object_sha256(top)


def _save_torch_create_only(path: Path, value: Any) -> dict[str, Any]:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError("output must be a fresh absolute file")
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    value_stat = path.lstat()
    if (
        path.is_symlink() or not stat.S_ISREG(value_stat.st_mode)
        or stat.S_IMODE(value_stat.st_mode) != 0o444
        or value_stat.st_nlink != 1 or value_stat.st_size <= 0
    ):
        raise RuntimeError("created shard seal differs")
    expected_semantic = _shard_semantic_sha256(value)
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        handle.seek(0)
        reloaded = torch.load(handle, map_location="cpu", weights_only=True)
        after = os.fstat(handle.fileno())
    if (
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or _shard_semantic_sha256(reloaded) != expected_semantic
        or reloaded.get("schema_version") != FEATURE_SCHEMA
        or len(reloaded.get("records", [])) != value.get("record_count")
    ):
        raise RuntimeError("created shard readback differs")
    return {
        "path": str(path.resolve(strict=True)), "sha256": digest.hexdigest(),
        "size_bytes": value_stat.st_size, "mode": stat.S_IMODE(value_stat.st_mode),
        "nlink": value_stat.st_nlink, "semantic_sha256": expected_semantic,
        "fresh_torch_load_readback_exact": True,
    }


def extract_shard(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_shards != 6 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index/count differs")
    binding = implementation_binding()
    anchors, manifest = load_anchor_manifest(
        Path(args.manifest), args.expected_manifest_sha256
    )
    selected = [row for row in anchors if row.ordinal % args.num_shards == args.shard_index]
    if len(selected) != sum(
        ordinal % args.num_shards == args.shard_index for ordinal in range(644)
    ):
        raise RuntimeError("deterministic shard population differs")
    frozen = FrozenVJepa2(Path(args.model_root), args.device)
    torch.cuda.reset_peak_memory_stats(0)
    records = []
    for local_ordinal, item in enumerate(selected):
        print(json.dumps({
            "shard_index": args.shard_index, "local_ordinal": local_ordinal,
            "shard_count": len(selected), "iid": item.iid,
        }, sort_keys=True), flush=True)
        records.append(extract_one(item, frozen))
    closure = frozen.final_closure()
    expected_processor_calls = len(selected)
    expected_forward_calls = 5 * len(selected)
    if (
        frozen.processor_call_count != expected_processor_calls
        or frozen.forward_call_count != expected_forward_calls
    ):
        raise RuntimeError("shard processor/backbone invocation count differs")
    payload = {
        "schema_version": FEATURE_SCHEMA,
        "status": "V4C_VJEPA2_ORDERED_CONTEXTUAL_SHARD_COMPLETE_BURNED_DEVELOPMENT",
        "authority": "feature_mechanics_diagnostic_only",
        "formal_training_authorized": False,
        "paired_ground_truth_claimed": False,
        "implementation": binding,
        "manifest_path": str(Path(args.manifest).resolve(strict=True)),
        "manifest_sha256": FEATURE_MANIFEST_SHA256,
        "manifest_digest": FEATURE_MANIFEST_DIGEST,
        "source_manifest_sha256": SOURCE_MANIFEST_FILE_SHA256,
        "source_manifest_digest": SOURCE_MANIFEST_DIGEST,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "record_count": len(records),
        "global_anchor_ordinals": [row.ordinal for row in selected],
        "global_anchor_iid_digest": object_sha256([row.iid for row in selected]),
        "processor_call_count": frozen.processor_call_count,
        "frozen_backbone_forward_count": frozen.forward_call_count,
        "one_processor_then_exact5_separate_forwards_per_anchor": True,
        "model_forward_batching_across_views": False,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_dtype": "torch.float16",
        "skip_predictor": True,
        "model_and_source_closure": closure,
        "runtime": {
            "torch": str(torch.__version__), "torch_hip": str(torch.version.hip),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "device_uuid": frozen.device_uuid,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "manual_seed": SEED,
            "cuda_manual_seed_all": SEED,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        },
        "sampling_and_transform_abi": {
            "exact81_to_base64_formula": "floor(80*k/63), k=0..63",
            "exact81_to_base64_indices_sha256": BASE64_INDICES_SHA256,
            "warp64_formula": "coord[2*i+j]=2*float32(WARP32[i])+j",
            "warp64_coordinates_sha256": WARP64_COORDINATES_SHA256,
            "phase32_block_permutation": list(PHASE_BLOCK_PERMUTATION),
            "transform_axis": "pixel_values_videos temporal dim 1",
            "post_backbone_token_permutation_used": False,
        },
        "feature_geometry": {
            "processor_output": [1, 64, 3, 256, 256],
            "last_hidden_state": [1, 8192, 1024],
            "time_major_grid": [1, 32, 256, 1024],
            "stored_sequence_per_view": [32, 1024],
            "view_names": list(VIEW_NAMES),
        },
        "records": records,
    }
    output = Path(args.output)
    if implementation_binding() != binding:
        raise RuntimeError("extractor implementation changed during shard command")
    output_binding = _save_torch_create_only(output, payload)
    if implementation_binding() != binding:
        raise RuntimeError("extractor implementation changed after shard write")
    return {
        "shard": str(output.resolve(strict=True)),
        "sha256": output_binding["sha256"],
        "output_binding": output_binding,
        "record_count": len(records), "processor_calls": frozen.processor_call_count,
        "backbone_forwards": frozen.forward_call_count,
    }


def _load_sealed_shard(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("shard path must be absolute and non-symlink")
    resolved = path.resolve(strict=True)
    if str(path) != str(resolved):
        raise ValueError("shard logical/resolved path differs")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1 or before.st_size <= 0
    ):
        raise ValueError("shard must be a sealed 0444/nlink1 regular file")
    expected = _sha(expected_sha256, "expected shard SHA")
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        digest_before = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_before.update(chunk)
        if digest_before.hexdigest() != expected:
            raise ValueError("shard file SHA differs")
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
        opened_after_load = os.fstat(handle.fileno())
        handle.seek(0)
        digest_after = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_after.update(chunk)
    after = path.lstat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_nlink,
        value.st_mode, value.st_mtime_ns, value.st_ctime_ns,
    )
    if (
        digest_after.hexdigest() != expected
        or identity(before) != identity(opened_before)
        or identity(opened_before) != identity(opened_after_load)
        or identity(opened_after_load) != identity(after)
    ):
        raise RuntimeError("shard changed during single-FD load/readback")
    binding = {
        "path": str(resolved), "sha256": expected, "size_bytes": before.st_size,
        "mode": stat.S_IMODE(before.st_mode), "nlink": before.st_nlink,
        "semantic_sha256": _shard_semantic_sha256(payload),
        "single_fd_pre_post_sha256_exact": True,
    }
    return payload, binding


def _validate_record(record: Any, expected: AnchorItem) -> None:
    if type(record) is not dict or type(record.get("view_sequences")) is not dict:
        raise ValueError("V-JEPA2 shard record differs")
    if (
        record.get("ordinal") != expected.ordinal or record.get("iid") != expected.iid
        or record.get("family") != expected.family
        or record.get("group_id") != expected.group_id
        or record.get("instruction_sha256") != expected.instruction_sha256
        or record.get("strict_selection_gates_all_true") is not expected.strict
        or record.get("role") != "action_anchor"
        or record.get("media_sha256") != expected.media_sha256
        or record.get("exact81_to_base64_indices") != base64_indices().tolist()
        or record.get("exact81_to_base64_indices_sha256") != BASE64_INDICES_SHA256
        or record.get("block64_indices") != block64_indices(expected.iid).tolist()
        or record.get("block64_indices_sha256") != tensor_sha256(block64_indices(expected.iid))
        or record.get("phase64_indices") != phase64_indices().tolist()
        or record.get("phase64_indices_sha256") != tensor_sha256(phase64_indices())
        or record.get("canonical_processor_call_count") != 1
        or record.get("independent_frozen_backbone_forward_count") != 5
        or record.get("model_forward_batching_across_views") is not False
        or record.get("post_backbone_token_permutation_used") is not False
        or record.get("view_order") != list(VIEW_NAMES)
    ):
        raise ValueError("V-JEPA2 record authority/transform contract differs")
    sequences = record["view_sequences"]
    receipts = record.get("view_receipts")
    media = record.get("media")
    pts = media.get("pts") if type(media) is dict else None
    try:
        time_base = Fraction(media.get("single_time_base")) if type(media) is dict else None
    except (TypeError, ValueError, ZeroDivisionError):
        time_base = None
    if (
        type(media) is not dict or media.get("manifest_logical_path") != str(expected.path)
        or media.get("resolved_path") != str(expected.path)
        or media.get("logical_equals_resolved_path") is not True
        or media.get("decoder") != "PyAV"
        or media.get("pyav_version") != "13.1.0"
        or media.get("decoded_display_order_is_iteration_order") is not True
        or media.get("average_rate") != "25/1"
        or media.get("stream_frames_metadata") != 81
        or media.get("decoded_frame_count") != 81
        or media.get("all_pts_integral") is not True
        or media.get("pts_strictly_increasing") is not True
        or media.get("every_pts_delta_is_exactly_one_over_25_seconds") is not True
        or type(pts) is not list or len(pts) != 81
        or any(type(value) is not int for value in pts)
        or media.get("pts_sha256") != object_sha256(pts)
        or time_base is None or time_base <= 0
        or any(right <= left for left, right in zip(pts, pts[1:]))
        or any(
            Fraction(right - left) * time_base != Fraction(1, 25)
            for left, right in zip(pts, pts[1:])
        )
        or SHA_RE.fullmatch(str(media.get("decoded_exact81_rgb24_sha256"))) is None
        or SHA_RE.fullmatch(str(record.get("selected_base64_rgb24_sha256"))) is None
    ):
        raise ValueError("V-JEPA2 record media/decode proof differs")
    if set(sequences) != set(VIEW_NAMES) or type(receipts) is not dict or set(receipts) != set(VIEW_NAMES):
        raise ValueError("V-JEPA2 record view closure differs")
    for ordinal, name in enumerate(VIEW_NAMES, start=1):
        sequence = sequences[name]
        view = receipts[name]
        if (
            type(sequence) is not torch.Tensor or tuple(sequence.shape) != (32, 1024)
            or sequence.dtype != torch.float32 or not bool(torch.isfinite(sequence).all())
            or view.get("forward_ordinal_within_anchor") != ordinal
            or view.get("model_input_shape") != [1, 64, 3, 256, 256]
            or view.get("model_input_dtype") != "torch.float32"
            or view.get("last_hidden_state_shape") != [1, 8192, 1024]
            or view.get("token_output_permuted_or_reindexed") is not False
            or view.get("time_major_reshape") != [1, 32, 256, 1024]
            or view.get("spatial_mean_output_shape") != [32, 1024]
            or view.get("ordered_contextual_sequence_sha256") != tensor_sha256(sequence)
            or SHA_RE.fullmatch(str(view.get("model_input_pixel_values_videos_sha256"))) is None
        ):
            raise ValueError(f"V-JEPA2 record view differs: {name}")


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError("receipt output must be a fresh absolute file")
    raw = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
    ).encode("ascii") + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    value_stat = path.lstat()
    sha = hashlib.sha256(raw).hexdigest()
    if (
        path.is_symlink() or not stat.S_ISREG(value_stat.st_mode)
        or stat.S_IMODE(value_stat.st_mode) != 0o444 or value_stat.st_nlink != 1
        or value_stat.st_size != len(raw) or file_sha256(path) != sha
    ):
        raise RuntimeError("receipt write/readback seal differs")
    return {
        "path": str(path.resolve(strict=True)), "sha256": sha,
        "size_bytes": value_stat.st_size, "mode": 0o444, "nlink": 1,
    }


def aggregate_shards(args: argparse.Namespace) -> dict[str, Any]:
    binding = implementation_binding()
    if len(args.shard) != 6 or len(args.expected_shard_sha256) != 6:
        raise ValueError("postflight requires exact6 shard paths and SHAs")
    anchors, _ = load_anchor_manifest(Path(args.manifest), args.expected_manifest_sha256)
    loaded = [
        _load_sealed_shard(Path(path), sha)
        for path, sha in zip(args.shard, args.expected_shard_sha256)
    ]
    by_index = {payload.get("shard_index"): (payload, shard_binding) for payload, shard_binding in loaded}
    if set(by_index) != set(range(6)) or len(by_index) != 6:
        raise ValueError("postflight shard index closure differs")
    common_closure = by_index[0][0].get("model_and_source_closure")
    if (
        type(common_closure) is not dict
        or common_closure.get("model_files_before_and_after_exact") is not True
        or common_closure.get("transformers_modules_before_and_after_exact") is not True
        or type(common_closure.get("model")) is not dict
        or type(common_closure.get("transformers")) is not dict
        or sealed_model_closure(Path(common_closure["model"]["root"]))
        != common_closure["model"]
        or transformers_module_closure() != common_closure["transformers"]
    ):
        raise ValueError("postflight live model/source closure differs")
    records_by_ordinal: dict[int, dict[str, Any]] = {}
    shard_bindings = []
    for index in range(6):
        payload, shard_binding = by_index[index]
        records = payload.get("records")
        expected_ordinals = [ordinal for ordinal in range(644) if ordinal % 6 == index]
        if (
            payload.get("schema_version") != FEATURE_SCHEMA
            or payload.get("status") != "V4C_VJEPA2_ORDERED_CONTEXTUAL_SHARD_COMPLETE_BURNED_DEVELOPMENT"
            or payload.get("authority") != "feature_mechanics_diagnostic_only"
            or payload.get("formal_training_authorized") is not False
            or payload.get("paired_ground_truth_claimed") is not False
            or payload.get("implementation") != binding
            or payload.get("manifest_sha256") != FEATURE_MANIFEST_SHA256
            or payload.get("manifest_digest") != FEATURE_MANIFEST_DIGEST
            or payload.get("source_manifest_sha256") != SOURCE_MANIFEST_FILE_SHA256
            or payload.get("source_manifest_digest") != SOURCE_MANIFEST_DIGEST
            or payload.get("num_shards") != 6
            or payload.get("global_anchor_ordinals") != expected_ordinals
            or type(records) is not list or payload.get("record_count") != len(expected_ordinals)
            or len(records) != len(expected_ordinals)
            or payload.get("processor_call_count") != len(records)
            or payload.get("frozen_backbone_forward_count") != 5 * len(records)
            or payload.get("one_processor_then_exact5_separate_forwards_per_anchor") is not True
            or payload.get("model_forward_batching_across_views") is not False
            or payload.get("model_repo") != MODEL_REPO
            or payload.get("model_revision") != MODEL_REVISION
            or payload.get("model_dtype") != "torch.float16"
            or payload.get("skip_predictor") is not True
            or payload.get("model_and_source_closure") != common_closure
        ):
            raise ValueError(f"postflight shard authority differs: {index}")
        for record, ordinal in zip(records, expected_ordinals):
            _validate_record(record, anchors[ordinal])
            if ordinal in records_by_ordinal:
                raise ValueError("postflight duplicate anchor ordinal")
            records_by_ordinal[ordinal] = record
        shard_bindings.append({"index": index, **shard_binding, "record_count": len(records)})
    if set(records_by_ordinal) != set(range(644)):
        raise ValueError("postflight union is not exact644 once each")
    ordered = [records_by_ordinal[index] for index in range(644)]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "FEATURES_EXTRACTED_NOT_REPRESENTATION_QUALIFIED",
        "authority": "feature_mechanics_diagnostic_only",
        "formal_training_authorized": False,
        "paired_ground_truth_claimed": False,
        "burned_development_only": True,
        "implementation": binding,
        "manifest": {
            "path": str(Path(args.manifest).resolve(strict=True)),
            "sha256": FEATURE_MANIFEST_SHA256, "manifest_digest": FEATURE_MANIFEST_DIGEST,
            "source_manifest_sha256": SOURCE_MANIFEST_FILE_SHA256,
            "source_manifest_digest": SOURCE_MANIFEST_DIGEST,
        },
        "population": {
            "unique_base_clips": 644, "action_anchor_records": 644,
            "source_records": 0, "total_feature_records": 644,
            "view_evaluations_per_anchor": 5, "derived_views_are_independent_samples": False,
            "family_count": 28, "strict_true": 359, "strict_false": 285,
        },
        "exact644_ordered_iid_digest": object_sha256([row.iid for row in anchors]),
        "exact644_record_semantic_sha256": object_sha256([
            {
                "iid": row["iid"], "ordinal": row["ordinal"],
                "view_sequence_sha256": {
                    name: tensor_sha256(row["view_sequences"][name]) for name in VIEW_NAMES
                },
            } for row in ordered
        ]),
        "feature_geometry": {
            "views": list(VIEW_NAMES), "stored_sequence_per_view": [32, 1024],
            "teacher": "V-JEPA2 ViT-L fpc64 256 frozen FP16 skip_predictor",
            "post_backbone_token_permutation_used": False,
        },
        "sampling_and_transform_abi": by_index[0][0]["sampling_and_transform_abi"],
        "model_and_source_closure": common_closure,
        "shards": shard_bindings,
        "exact6_shards": True,
        "each_anchor_processor_call_count": 1,
        "each_anchor_independent_backbone_forward_count": 5,
        "action_representation_qualified": False,
        "scientific_confirmation_claimed": False,
        "identity_disentanglement_qualified": False,
        "identity_preservation_qualified": False,
        "prior_generation_qualified": False,
        "generation_qualified": False,
        "video_editing_qualified": False,
        "full644_refit_authorized": False,
        "renderer_authorized": False,
        "inference_authorized": False,
        "web_evaluation_authorized": False,
        "vae_necessary": None,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    if implementation_binding() != binding:
        raise RuntimeError("extractor implementation changed during postflight")
    output_binding = _write_json_create_only(Path(args.output), receipt)
    if implementation_binding() != binding:
        raise RuntimeError("extractor implementation changed after postflight")
    return {
        "receipt": output_binding["path"], "receipt_sha256": output_binding["sha256"],
        "receipt_digest": receipt["receipt_digest"], "record_count": 644,
        "shard_count": 6, "output_binding": output_binding,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("extract-shard")
    shard.add_argument("--manifest", required=True)
    shard.add_argument("--expected-manifest-sha256", required=True)
    shard.add_argument("--model-root", required=True)
    shard.add_argument("--shard-index", required=True, type=int)
    shard.add_argument("--num-shards", required=True, type=int)
    shard.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    shard.add_argument("--output", required=True)
    shard.set_defaults(handler=extract_shard)
    aggregate = commands.add_parser("aggregate-shards")
    aggregate.add_argument("--manifest", required=True)
    aggregate.add_argument("--expected-manifest-sha256", required=True)
    aggregate.add_argument("--shard", action="append", required=True)
    aggregate.add_argument("--expected-shard-sha256", action="append", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(handler=aggregate_shards)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

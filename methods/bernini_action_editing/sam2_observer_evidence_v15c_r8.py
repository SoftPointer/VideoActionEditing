#!/usr/bin/env python3
"""Strict, local-only SAM2 observer evidence replay for v15c-r8.

This module deliberately separates *byte replay* from *observer execution*.
It can prove that a published bundle contains complete, standards-compatible
safetensors artifacts and that every derived mask/order/AMG/freeze claim is a
deterministic function of those bytes.  It cannot, by itself, prove that a
remote SAM2 worker ran, localize an object, or authorize routing/training.

The public replay entry point never accepts caller supplied transcripts.  It
reopens raw AMG, prompt-logit and propagation-logit artifacts and constructs
the transcript itself.  This permanently rejects the r7 counterexample made
from PNG masks plus invented 64-hex logit/model hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
from typing import Any, Mapping, Optional, Sequence


LOCAL_SCHEMA = "bernini-source-sam2-observer-evidence-v15c-r8-local"
LOCAL_REPLAY_SCHEMA = "bernini-source-sam2-observer-evidence-v15c-r8-local-replay"
AMG_SCHEMA = "bernini-source-sam2-amg-artifact-v15c-r8"
LOGIT_FILE_SCHEMA = "bernini-source-sam2-logit-file-v15c-r8"
FREEZE_SCHEMA = "bernini-source-sam2-freeze-transcript-v15c-r8"
MODEL_MANIFEST_SCHEMA = "bernini-source-sam2-model-tensor-manifest-v15c-r8"
RUN_SCHEMA = "bernini-source-sam2-worker-run-v15c-r8"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

AMG_TENSOR_ORDER = (
    "area",
    "bbox_xywh",
    "predicted_iou",
    "stability_score",
    "masks",
)
MODEL_KINDS = ("image_model", "video_model")
FRAME_COUNT = 81
HEIGHT = 1056
WIDTH = 704


class SAM2ObserverEvidenceV15CR8Error(RuntimeError):
    """A byte, schema, provenance, order, or local-only gate differs."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise SAM2ObserverEvidenceV15CR8Error("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise SAM2ObserverEvidenceV15CR8Error(f"{label} is not lowercase SHA256")
    return value


def require_exact_keys(value: Any, keys: Sequence[str], label: str) -> None:
    if type(value) is not dict or list(value) != list(keys):
        raise SAM2ObserverEvidenceV15CR8Error(f"{label} exact key/order differs")


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise SAM2ObserverEvidenceV15CR8Error(f"{label} is absent") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SAM2ObserverEvidenceV15CR8Error(f"{label} is not one regular file")
    if info.st_nlink != 1:
        raise SAM2ObserverEvidenceV15CR8Error(f"{label} must have one hard link")
    return info


def read_stable_bytes(path: Path, label: str) -> bytes:
    _regular(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    current = _regular(path, label)
    if (
        identity_before != identity_after
        or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
    ):
        raise SAM2ObserverEvidenceV15CR8Error(f"{label} changed while reading")
    return b"".join(chunks)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(read_stable_bytes(path, str(path))).hexdigest()


def array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(int(item)) for item in array.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SAM2ObserverEvidenceV15CR8Error(f"{label} path differs")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise SAM2ObserverEvidenceV15CR8Error(f"{label} path escapes its root")
    return value


def _beneath(root: Path, relative: str, label: str) -> Path:
    root_absolute = root.absolute()
    if root_absolute.is_symlink() or not root_absolute.is_dir():
        raise SAM2ObserverEvidenceV15CR8Error("evidence root differs")
    root_resolved = root_absolute.resolve(strict=True)
    normalized = _relative(relative, label)
    candidate = root_resolved
    parts = PurePosixPath(normalized).parts
    for index, part in enumerate(parts):
        candidate = candidate / part
        try:
            info = candidate.lstat()
        except OSError as error:
            raise SAM2ObserverEvidenceV15CR8Error(f"{label} is absent") from error
        if stat.S_ISLNK(info.st_mode):
            raise SAM2ObserverEvidenceV15CR8Error(f"{label} contains a symlink")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise SAM2ObserverEvidenceV15CR8Error(f"{label} parent differs")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise SAM2ObserverEvidenceV15CR8Error(f"{label} is absent") from error
    if root_resolved not in resolved.parents:
        raise SAM2ObserverEvidenceV15CR8Error(f"{label} escapes its evidence root")
    _regular(resolved, label)
    return resolved


def _json_no_duplicates(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SAM2ObserverEvidenceV15CR8Error("duplicate safetensors header key")
        result[key] = value
    return result


def strict_safetensors(
    path: Path,
    *,
    expected_order: Sequence[str],
    expected_contract: Mapping[str, tuple[str, tuple[int, ...]]],
    expected_file_sha256: str,
    expected_array_sha256: Mapping[str, str],
    expected_metadata: Mapping[str, str],
) -> Mapping[str, Any]:
    """Reopen a standard safetensors file with exact registry and extents.

    Unlike ``safetensors.numpy.load_file``, this parser also checks header key
    order, duplicate keys, gaps/overlaps, trailing bytes and exact metadata.
    It intentionally supports only the dtypes emitted by the sealed worker.
    """

    import numpy as np

    if tuple(expected_order) != tuple(expected_contract):
        raise SAM2ObserverEvidenceV15CR8Error("tensor contract order differs")
    if list(expected_array_sha256) != list(expected_order):
        raise SAM2ObserverEvidenceV15CR8Error("tensor hash order differs")
    raw = read_stable_bytes(path, "safetensors artifact")
    expected_file = require_sha256(expected_file_sha256, "safetensors file hash")
    if hashlib.sha256(raw).hexdigest() != expected_file or len(raw) <= 8:
        raise SAM2ObserverEvidenceV15CR8Error("safetensors file bytes differ")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if header_length <= 1 or header_length > 64 * 1024 * 1024:
        raise SAM2ObserverEvidenceV15CR8Error("safetensors header length differs")
    data_start = 8 + header_length
    if data_start >= len(raw):
        raise SAM2ObserverEvidenceV15CR8Error("safetensors payload is empty")
    try:
        header = json.loads(
            raw[8:data_start].decode("utf-8"),
            object_pairs_hook=_json_no_duplicates,
        )
    except SAM2ObserverEvidenceV15CR8Error:
        raise
    except Exception as error:
        raise SAM2ObserverEvidenceV15CR8Error("safetensors header differs") from error
    if type(header) is not dict:
        raise SAM2ObserverEvidenceV15CR8Error("safetensors header is not one object")
    header_order = list(header)
    metadata = header.pop("__metadata__", None)
    tensor_order = [key for key in header_order if key != "__metadata__"]
    if metadata != dict(expected_metadata) or tensor_order != list(expected_order):
        raise SAM2ObserverEvidenceV15CR8Error("safetensors metadata/key order differs")
    dtype_map = {
        "F32": np.dtype("<f4"),
        "F64": np.dtype("<f8"),
        "I64": np.dtype("<i8"),
        "I32": np.dtype("<i4"),
        "U8": np.dtype("u1"),
        "I8": np.dtype("i1"),
        "BOOL": np.dtype("bool"),
    }
    arrays = {}
    transcript = {}
    intervals = []
    for key in expected_order:
        row = header.get(key)
        if type(row) is not dict or list(row) != ["dtype", "shape", "data_offsets"]:
            raise SAM2ObserverEvidenceV15CR8Error(f"tensor {key} descriptor differs")
        dtype_name, expected_shape = expected_contract[key]
        shape = row.get("shape")
        offsets = row.get("data_offsets")
        if (
            row.get("dtype") != dtype_name
            or dtype_name not in dtype_map
            or type(shape) is not list
            or tuple(shape) != tuple(expected_shape)
            or any(type(item) is not int or item <= 0 for item in shape)
            or type(offsets) is not list
            or len(offsets) != 2
            or any(type(item) is not int or item < 0 for item in offsets)
            or offsets[0] >= offsets[1]
        ):
            raise SAM2ObserverEvidenceV15CR8Error(f"tensor {key} contract differs")
        element_count = math.prod(shape)
        extent = element_count * dtype_map[dtype_name].itemsize
        if offsets[1] - offsets[0] != extent:
            raise SAM2ObserverEvidenceV15CR8Error(f"tensor {key} extent differs")
        begin = data_start + offsets[0]
        end = data_start + offsets[1]
        if end > len(raw):
            raise SAM2ObserverEvidenceV15CR8Error(f"tensor {key} exceeds its file")
        value = np.frombuffer(
            raw,
            dtype=dtype_map[dtype_name],
            count=element_count,
            offset=begin,
        ).reshape(tuple(shape))
        if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
            raise SAM2ObserverEvidenceV15CR8Error(f"tensor {key} is non-finite")
        observed_hash = array_sha256(value)
        if observed_hash != require_sha256(
            expected_array_sha256[key], f"tensor {key} array hash"
        ):
            raise SAM2ObserverEvidenceV15CR8Error(f"tensor {key} content differs")
        arrays[key] = value
        transcript[key] = {
            "dtype": dtype_name,
            "shape": list(shape),
            "array_sha256": observed_hash,
            "finite": True,
        }
        intervals.append((offsets[0], offsets[1], key))
    intervals.sort()
    cursor = 0
    for begin, end, _key in intervals:
        if begin != cursor:
            raise SAM2ObserverEvidenceV15CR8Error(
                "safetensors payload has a gap or overlap"
            )
        cursor = end
    if data_start + cursor != len(raw):
        raise SAM2ObserverEvidenceV15CR8Error(
            "safetensors has trailing/unregistered bytes"
        )
    return {
        "file_sha256": expected_file,
        "metadata": metadata,
        "tensor_order": list(expected_order),
        "tensors": transcript,
        "arrays": arrays,
    }


ARTIFACT_KEYS = (
    "schema_version",
    "relative_path",
    "file_sha256",
    "tensor_order",
    "tensor_array_sha256",
)


def _artifact_descriptor(value: Any, *, schema: str, expected_path: str) -> Mapping[str, Any]:
    require_exact_keys(value, ARTIFACT_KEYS, "tensor artifact")
    if (
        value["schema_version"] != schema
        or _relative(value["relative_path"], "tensor artifact") != expected_path
        or list(value["tensor_array_sha256"]) != value["tensor_order"]
    ):
        raise SAM2ObserverEvidenceV15CR8Error("tensor artifact descriptor differs")
    require_sha256(value["file_sha256"], "tensor artifact file hash")
    for key in value["tensor_order"]:
        require_sha256(value["tensor_array_sha256"][key], f"tensor artifact {key}")
    return value


def mask_bbox_xywh(mask: Any) -> list[float]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    return [
        float(x0),
        float(y0),
        float(int(xs.max()) + 1 - x0),
        float(int(ys.max()) + 1 - y0),
    ]


def _mask_iou(left: Any, right: Any) -> float:
    import numpy as np

    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return float(intersection / union) if union else 0.0


def _admit_amg_candidates(
    masks: Any,
    predicted_iou: Any,
    stability_score: Any,
    *,
    admission: Mapping[str, Any],
) -> list[int]:
    import numpy as np

    ranked = []
    image_area = int(masks.shape[1] * masks.shape[2])
    for index in range(int(masks.shape[0])):
        mask = masks[index] != 0
        area = int(mask.sum())
        if (
            area < int(admission["minimum_area_pixels"])
            or area / float(image_area) > float(admission["maximum_area_fraction"])
        ):
            continue
        digest = array_sha256(np.ascontiguousarray(mask, dtype=np.uint8))
        quality = 0.5 * float(predicted_iou[index]) + 0.5 * float(stability_score[index])
        ranked.append((-quality, digest, index))
    ranked.sort(key=lambda row: (row[0], row[1]))
    selected = []
    for _negative_quality, _digest, index in ranked:
        if all(
            _mask_iou(masks[index] != 0, masks[kept] != 0)
            < float(admission["near_duplicate_iou"])
            for kept in selected
        ):
            selected.append(index)
    if not 1 <= len(selected) <= int(admission["maximum_distinct_proposals"]):
        raise SAM2ObserverEvidenceV15CR8Error("AMG admission count differs")
    return sorted(
        selected,
        key=lambda index: array_sha256(
            np.ascontiguousarray(masks[index] != 0, dtype=np.uint8)
        ),
    )


def replay_amg_artifact(
    *,
    root: Path,
    run_ordinal: int,
    descriptor: Mapping[str, Any],
    admission: Mapping[str, Any],
    automatic_generator: Mapping[str, Any],
    height: int,
    width: int,
) -> Mapping[str, Any]:
    """Recompute proposal rows and admission from raw AMG tensor bytes."""

    import numpy as np

    expected_path = f"observer_evidence/run_{run_ordinal}/amg.safetensors"
    row = _artifact_descriptor(descriptor, schema=AMG_SCHEMA, expected_path=expected_path)
    order = tuple(row["tensor_order"])
    if order != AMG_TENSOR_ORDER:
        raise SAM2ObserverEvidenceV15CR8Error("AMG tensor order differs")
    hashes = row["tensor_array_sha256"]
    # Candidate count is encoded in every tensor shape and is therefore not a
    # separately trusted receipt field.
    path = _beneath(root, expected_path, "AMG safetensors")
    raw = read_stable_bytes(path, "AMG safetensors")
    if hashlib.sha256(raw).hexdigest() != row["file_sha256"]:
        raise SAM2ObserverEvidenceV15CR8Error("AMG file hash differs")
    if len(raw) <= 8:
        raise SAM2ObserverEvidenceV15CR8Error("AMG safetensors is empty")
    header_length = struct.unpack("<Q", raw[:8])[0]
    try:
        header = json.loads(
            raw[8 : 8 + header_length].decode("utf-8"),
            object_pairs_hook=_json_no_duplicates,
        )
    except Exception as error:
        raise SAM2ObserverEvidenceV15CR8Error("AMG header differs") from error
    masks_header = header.get("masks") if type(header) is dict else None
    if type(masks_header) is not dict or type(masks_header.get("shape")) is not list:
        raise SAM2ObserverEvidenceV15CR8Error("AMG masks descriptor differs")
    shape = masks_header["shape"]
    if len(shape) != 3 or shape[1:] != [height, width] or type(shape[0]) is not int:
        raise SAM2ObserverEvidenceV15CR8Error("AMG mask shape differs")
    count = shape[0]
    if not 1 <= count <= 4096:
        raise SAM2ObserverEvidenceV15CR8Error("AMG raw candidate count differs")
    metadata = {
        "schema_version": AMG_SCHEMA,
        "run_ordinal": str(run_ordinal),
        "source_frame_index": "0",
    }
    parsed = strict_safetensors(
        path,
        expected_order=AMG_TENSOR_ORDER,
        expected_contract={
            "area": ("I64", (count,)),
            "bbox_xywh": ("F32", (count, 4)),
            "predicted_iou": ("F32", (count,)),
            "stability_score": ("F32", (count,)),
            "masks": ("U8", (count, height, width)),
        },
        expected_file_sha256=row["file_sha256"],
        expected_array_sha256=hashes,
        expected_metadata=metadata,
    )
    arrays = parsed["arrays"]
    masks = arrays["masks"]
    areas = arrays["area"]
    boxes = arrays["bbox_xywh"]
    predicted = arrays["predicted_iou"]
    stability = arrays["stability_score"]
    if not bool(np.isin(masks, np.asarray([0, 1], dtype=np.uint8)).all()):
        raise SAM2ObserverEvidenceV15CR8Error("AMG masks are not binary")
    if (
        not bool(((predicted >= 0.0) & (predicted <= 1.0)).all())
        or not bool(((stability >= 0.0) & (stability <= 1.0)).all())
        or not bool(
            (predicted >= float(automatic_generator["pred_iou_thresh"])).all()
        )
        or not bool(
            (stability >= float(automatic_generator["stability_score_thresh"])).all()
        )
    ):
        raise SAM2ObserverEvidenceV15CR8Error("AMG score/range/admission differs")
    candidates = []
    for index in range(count):
        mask = masks[index] != 0
        area = int(mask.sum())
        bbox = mask_bbox_xywh(mask)
        if int(areas[index]) != area or [float(item) for item in boxes[index]] != bbox:
            raise SAM2ObserverEvidenceV15CR8Error("AMG area/bbox is not mask-derived")
        digest = array_sha256(np.ascontiguousarray(mask, dtype=np.uint8))
        candidates.append(
            {
                "candidate_index": index,
                "prompt_mask_sha256": digest,
                "area": area,
                "bbox_xywh": bbox,
                "predicted_iou": float(predicted[index]),
                "stability_score": float(stability[index]),
            }
        )
    selected_indices = _admit_amg_candidates(
        masks,
        predicted,
        stability,
        admission=admission,
    )
    selected = []
    for index in selected_indices:
        candidate = candidates[index]
        selected.append(
            {
                "proposal_id": "sam2-f000-" + candidate["prompt_mask_sha256"],
                "candidate_index": index,
                **{key: candidate[key] for key in candidate if key != "candidate_index"},
            }
        )
    return {
        "artifact": {key: value for key, value in parsed.items() if key != "arrays"},
        "candidate_count": count,
        "candidates": candidates,
        "selected_candidate_indices": selected_indices,
        "selected_proposals": selected,
    }


def _replay_one_logit_file(
    *,
    root: Path,
    descriptor: Mapping[str, Any],
    expected_path: str,
    expected_shape: tuple[int, ...],
    metadata: Mapping[str, str],
) -> Mapping[str, Any]:
    row = _artifact_descriptor(
        descriptor,
        schema=LOGIT_FILE_SCHEMA,
        expected_path=expected_path,
    )
    if row["tensor_order"] != ["logits"]:
        raise SAM2ObserverEvidenceV15CR8Error("logit tensor registry differs")
    parsed = strict_safetensors(
        _beneath(root, expected_path, "logit safetensors"),
        expected_order=("logits",),
        expected_contract={"logits": ("F32", expected_shape)},
        expected_file_sha256=row["file_sha256"],
        expected_array_sha256=row["tensor_array_sha256"],
        expected_metadata=metadata,
    )
    return parsed


BATCH_DESCRIPTOR_KEYS = (
    "schema_version",
    "batch_index",
    "batch_start",
    "batch_stop",
    "prompt_artifacts",
    "propagation_artifacts",
)


def replay_tracking_batch(
    *,
    root: Path,
    run_ordinal: int,
    descriptor: Mapping[str, Any],
    expected_batch_index: int,
    expected_batch_start: int,
    expected_batch_stop: int,
    frame_count: int,
    height: int,
    width: int,
) -> Mapping[str, Any]:
    """Derive prompt/propagation order and mask hashes from tensor files."""

    import numpy as np

    require_exact_keys(descriptor, BATCH_DESCRIPTOR_KEYS, "tracking batch descriptor")
    prompt_artifacts = descriptor["prompt_artifacts"]
    propagation_artifacts = descriptor["propagation_artifacts"]
    if (
        descriptor["schema_version"] != RUN_SCHEMA
        or descriptor["batch_index"] != expected_batch_index
        or descriptor["batch_start"] != expected_batch_start
        or descriptor["batch_stop"] != expected_batch_stop
        or type(prompt_artifacts) is not list
        or len(prompt_artifacts) != expected_batch_stop - expected_batch_start
        or type(propagation_artifacts) is not list
        or len(propagation_artifacts) != frame_count
    ):
        raise SAM2ObserverEvidenceV15CR8Error("tracking batch descriptor differs")
    object_ids = list(range(expected_batch_start, expected_batch_stop))
    prefix = f"observer_evidence/run_{run_ordinal}/batch_{expected_batch_index:03d}"
    prompt_transcript = []
    for call_index, artifact in enumerate(prompt_artifacts):
        out_ids = object_ids[: call_index + 1]
        expected_path = f"{prefix}/prompt_call_{call_index:03d}.safetensors"
        parsed = _replay_one_logit_file(
            root=root,
            descriptor=artifact,
            expected_path=expected_path,
            expected_shape=(len(out_ids), 1, height, width),
            metadata={
                "schema_version": LOGIT_FILE_SCHEMA,
                "kind": "prompt",
                "run_ordinal": str(run_ordinal),
                "batch_index": str(expected_batch_index),
                "call_index": str(call_index),
                "inserted_object_id": str(object_ids[call_index]),
                "out_ids": ",".join(str(item) for item in out_ids),
            },
        )
        prompt_transcript.append(
            {
                "call_index": call_index,
                "inserted_object_id": object_ids[call_index],
                "frame_index": 0,
                "out_ids": out_ids,
                "artifact_file_sha256": parsed["file_sha256"],
                "logits": parsed["tensors"]["logits"],
            }
        )
    mask_hashes = {object_id: [] for object_id in object_ids}
    propagation_transcript = []
    for frame_index, artifact in enumerate(propagation_artifacts):
        expected_path = f"{prefix}/propagation_frame_{frame_index:05d}.safetensors"
        parsed = _replay_one_logit_file(
            root=root,
            descriptor=artifact,
            expected_path=expected_path,
            expected_shape=(len(object_ids), 1, height, width),
            metadata={
                "schema_version": LOGIT_FILE_SCHEMA,
                "kind": "propagation",
                "run_ordinal": str(run_ordinal),
                "batch_index": str(expected_batch_index),
                "frame_index": str(frame_index),
                "out_ids": ",".join(str(item) for item in object_ids),
            },
        )
        logits = parsed["arrays"]["logits"]
        for position, object_id in enumerate(object_ids):
            mask = np.ascontiguousarray(logits[position, 0] > 0.0, dtype=np.uint8)
            mask_hashes[object_id].append(array_sha256(mask))
        propagation_transcript.append(
            {
                "frame_index": frame_index,
                "out_ids": object_ids,
                "artifact_file_sha256": parsed["file_sha256"],
                "logits": parsed["tensors"]["logits"],
            }
        )
    return {
        "batch_index": expected_batch_index,
        "batch_start": expected_batch_start,
        "batch_stop": expected_batch_stop,
        "object_ids": object_ids,
        "prompt_calls": prompt_transcript,
        "propagation_frames": propagation_transcript,
        "mask_array_sha256_by_object_id": [
            {"object_id": object_id, "mask_array_sha256_by_frame": mask_hashes[object_id]}
            for object_id in object_ids
        ],
    }


MODEL_ENTRY_KEYS = ("name", "dtype", "shape", "numel", "array_sha256")
MODEL_MANIFEST_KEYS = (
    "schema_version",
    "tensor_kind",
    "tensor_count",
    "element_count",
    "entries",
    "stream_sha256",
    "manifest_sha256",
)
MODEL_STATE_KEYS = (
    "eval_mode",
    "requires_grad_true_count",
    "non_none_grad_count",
    "parameters",
    "buffers",
    "state_sha256",
)
FREEZE_KEYS = (
    "schema_version",
    "run_ordinal",
    "model_kind",
    "evidence_mode",
    "construction_binding",
    "before",
    "after",
    "all_freeze_gates_pass",
    "transcript_sha256",
)
BINDING_KEYS = (
    "source_video_sha256",
    "source_frame0_array_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "sam2_tree_sha256",
    "key_module_sha256",
    "resolved_config_sha256",
    "worker_code_sha256",
    "model_class",
)


def _validate_model_manifest(value: Any, *, tensor_kind: str) -> Mapping[str, Any]:
    require_exact_keys(value, MODEL_MANIFEST_KEYS, f"{tensor_kind} manifest")
    entries = value["entries"]
    if (
        value["schema_version"] != MODEL_MANIFEST_SCHEMA
        or value["tensor_kind"] != tensor_kind
        or type(entries) is not list
        or not entries
    ):
        raise SAM2ObserverEvidenceV15CR8Error(f"{tensor_kind} manifest differs")
    names = []
    element_count = 0
    allowed_dtypes = {
        "torch.bool",
        "torch.uint8",
        "torch.int8",
        "torch.int16",
        "torch.int32",
        "torch.int64",
        "torch.float16",
        "torch.bfloat16",
        "torch.float32",
        "torch.float64",
    }
    for entry in entries:
        require_exact_keys(entry, MODEL_ENTRY_KEYS, f"{tensor_kind} entry")
        name = entry["name"]
        shape = entry["shape"]
        if (
            not isinstance(name, str)
            or not name
            or type(shape) is not list
            or any(type(item) is not int or item < 0 for item in shape)
            or type(entry["numel"]) is not int
            or entry["numel"] < 0
            or entry["numel"] != math.prod(shape)
            or entry["dtype"] not in allowed_dtypes
        ):
            raise SAM2ObserverEvidenceV15CR8Error(f"{tensor_kind} entry differs")
        require_sha256(entry["array_sha256"], f"{tensor_kind} tensor hash")
        names.append(name)
        element_count += entry["numel"]
    require_sha256(value["stream_sha256"], f"{tensor_kind} stream hash")
    payload = dict(value)
    claimed = payload.pop("manifest_sha256", None)
    if (
        names != sorted(names)
        or len(set(names)) != len(names)
        or value["tensor_count"] != len(entries)
        or value["element_count"] != element_count
        or require_sha256(claimed, f"{tensor_kind} manifest hash")
        != object_sha256(payload)
    ):
        raise SAM2ObserverEvidenceV15CR8Error(f"{tensor_kind} manifest replay differs")
    return value


def _validate_model_state(value: Any) -> Mapping[str, Any]:
    require_exact_keys(value, MODEL_STATE_KEYS, "model state")
    parameters = _validate_model_manifest(value["parameters"], tensor_kind="parameters")
    buffers = _validate_model_manifest(value["buffers"], tensor_kind="buffers")
    expected_state = object_sha256(
        {
            "parameters_manifest_sha256": parameters["manifest_sha256"],
            "buffers_manifest_sha256": buffers["manifest_sha256"],
        }
    )
    if (
        value["eval_mode"] is not True
        or value["requires_grad_true_count"] != 0
        or value["non_none_grad_count"] != 0
        or require_sha256(value["state_sha256"], "model state hash") != expected_state
    ):
        raise SAM2ObserverEvidenceV15CR8Error("model state/freeze gate differs")
    return value


def validate_freeze_transcript(
    value: Any,
    *,
    run_ordinal: int,
    model_kind: str,
    expected_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    require_exact_keys(value, FREEZE_KEYS, "freeze transcript")
    binding = value["construction_binding"]
    require_exact_keys(binding, BINDING_KEYS, "model construction binding")
    if dict(binding) != dict(expected_binding):
        raise SAM2ObserverEvidenceV15CR8Error("model construction bytes differ")
    for key in BINDING_KEYS[:-1]:
        require_sha256(binding[key], f"model construction binding {key}")
    if (
        not isinstance(binding["model_class"], str)
        or not binding["model_class"].startswith("sam2.")
    ):
        raise SAM2ObserverEvidenceV15CR8Error("model construction class differs")
    before = _validate_model_state(value["before"])
    after = _validate_model_state(value["after"])
    payload = dict(value)
    claimed = payload.pop("transcript_sha256", None)
    if (
        value["schema_version"] != FREEZE_SCHEMA
        or value["run_ordinal"] != run_ordinal
        or model_kind not in MODEL_KINDS
        or value["model_kind"] != model_kind
        or value["evidence_mode"] != "sealed_deterministic_worker_tensor_manifest"
        or before != after
        or value["all_freeze_gates_pass"] is not True
        or require_sha256(claimed, "freeze transcript hash") != object_sha256(payload)
    ):
        raise SAM2ObserverEvidenceV15CR8Error("freeze transcript replay differs")
    return value


RUN_KEYS = (
    "schema_version",
    "run_ordinal",
    "sam2_execution",
    "amg_artifact",
    "tracking_batches",
    "freeze_transcripts",
)
SAM2_EXECUTION_KEYS = (
    "sam2_python_module_loaded",
    "automatic_generator_class",
    "video_predictor_class",
    "automatic_generate_call_count",
    "add_new_mask_call_count",
    "propagate_in_video_call_count",
    "sam2__C_imported",
)
LOCAL_RECEIPT_KEYS = (
    "schema_version",
    "status",
    "binding",
    "proposal_count",
    "runs",
    "repeat_semantic_sha256",
    "local_schema_replay_only",
    "remote_worker_execution_verified",
    "observer_execution_authorized",
    "localization_semantically_certified",
    "scientific_claim_authorized",
    "route_authorized",
    "decode_authorized",
    "training_authorized",
    "receipt_sha256",
)
LOCAL_BINDING_KEYS = (
    "spec_raw_sha256",
    "spec_canonical_sha256",
    "source_video_sha256",
    "source_frame0_array_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "sam2_tree_sha256",
    "key_module_sha256",
    "image_resolved_config_sha256",
    "video_resolved_config_sha256",
    "worker_code_sha256",
)


def replay_worker_run(
    *,
    root: Path,
    run: Mapping[str, Any],
    run_ordinal: int,
    proposal_count: int,
    expected_binding: Mapping[str, Any],
    admission: Mapping[str, Any],
    automatic_generator: Mapping[str, Any],
    tracking_batch_size: int,
    frame_count: int,
    height: int,
    width: int,
) -> Mapping[str, Any]:
    """Reopen one run; no published batch/logit transcript is consumed."""

    require_exact_keys(run, RUN_KEYS, "worker run")
    execution = run["sam2_execution"]
    require_exact_keys(execution, SAM2_EXECUTION_KEYS, "SAM2 execution transcript")
    if (
        run["schema_version"] != RUN_SCHEMA
        or run["run_ordinal"] != run_ordinal
        or execution
        != {
            "sam2_python_module_loaded": True,
            "automatic_generator_class": "sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator",
            "video_predictor_class": "sam2.sam2_video_predictor.SAM2VideoPredictor",
            "automatic_generate_call_count": 1,
            "add_new_mask_call_count": proposal_count,
            "propagate_in_video_call_count": int(
                math.ceil(proposal_count / float(tracking_batch_size))
            ),
            "sam2__C_imported": False,
        }
    ):
        raise SAM2ObserverEvidenceV15CR8Error("sealed SAM2 call transcript differs")
    amg = replay_amg_artifact(
        root=root,
        run_ordinal=run_ordinal,
        descriptor=run["amg_artifact"],
        admission=admission,
        automatic_generator=automatic_generator,
        height=height,
        width=width,
    )
    if len(amg["selected_proposals"]) != proposal_count:
        raise SAM2ObserverEvidenceV15CR8Error("AMG/receipt proposal count differs")
    batches = run["tracking_batches"]
    expected_batch_count = int(math.ceil(proposal_count / float(tracking_batch_size)))
    if type(batches) is not list or len(batches) != expected_batch_count:
        raise SAM2ObserverEvidenceV15CR8Error("tracking batch count differs")
    batch_replays = []
    expected_start = 0
    for batch_index, batch in enumerate(batches):
        expected_stop = min(expected_start + tracking_batch_size, proposal_count)
        batch_replays.append(
            replay_tracking_batch(
                root=root,
                run_ordinal=run_ordinal,
                descriptor=batch,
                expected_batch_index=batch_index,
                expected_batch_start=expected_start,
                expected_batch_stop=expected_stop,
                frame_count=frame_count,
                height=height,
                width=width,
            )
        )
        expected_start = expected_stop
    freezes = run["freeze_transcripts"]
    if type(freezes) is not dict or list(freezes) != list(MODEL_KINDS):
        raise SAM2ObserverEvidenceV15CR8Error("freeze transcript registry differs")
    freeze_replays = {}
    for model_kind in MODEL_KINDS:
        resolved_key = (
            "image_resolved_config_sha256"
            if model_kind == "image_model"
            else "video_resolved_config_sha256"
        )
        binding = {
            "source_video_sha256": expected_binding["source_video_sha256"],
            "source_frame0_array_sha256": expected_binding[
                "source_frame0_array_sha256"
            ],
            "checkpoint_sha256": expected_binding["checkpoint_sha256"],
            "config_sha256": expected_binding["config_sha256"],
            "sam2_tree_sha256": expected_binding["sam2_tree_sha256"],
            "key_module_sha256": expected_binding["key_module_sha256"],
            "resolved_config_sha256": expected_binding[resolved_key],
            "worker_code_sha256": expected_binding["worker_code_sha256"],
            "model_class": (
                "sam2.modeling.sam2_base.SAM2Base"
                if model_kind == "image_model"
                else "sam2.sam2_video_predictor.SAM2VideoPredictor"
            ),
        }
        freeze_replays[model_kind] = validate_freeze_transcript(
            freezes[model_kind],
            run_ordinal=run_ordinal,
            model_kind=model_kind,
            expected_binding=binding,
        )
    return {
        "run_ordinal": run_ordinal,
        "sam2_execution": execution,
        "amg": amg,
        "tracking_batches": batch_replays,
        "freeze_transcripts": freeze_replays,
    }


def semantic_run_payload(replayed: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove run/path container identity while retaining every tensor value hash."""

    return {
        "sam2_execution": replayed["sam2_execution"],
        "amg_candidates": replayed["amg"]["candidates"],
        "selected_candidate_indices": replayed["amg"][
            "selected_candidate_indices"
        ],
        "selected_proposals": replayed["amg"]["selected_proposals"],
        "tracking_batches": [
            {
                "batch_start": batch["batch_start"],
                "batch_stop": batch["batch_stop"],
                "object_ids": batch["object_ids"],
                "prompt_calls": [
                    {
                        key: value
                        for key, value in prompt.items()
                        if key != "artifact_file_sha256"
                    }
                    for prompt in batch["prompt_calls"]
                ],
                "propagation_frames": [
                    {
                        key: value
                        for key, value in frame.items()
                        if key != "artifact_file_sha256"
                    }
                    for frame in batch["propagation_frames"]
                ],
                "mask_array_sha256_by_object_id": batch[
                    "mask_array_sha256_by_object_id"
                ],
            }
            for batch in replayed["tracking_batches"]
        ],
        "freeze_transcripts": {
            key: {
                field: value
                for field, value in transcript.items()
                if field not in {"run_ordinal", "transcript_sha256"}
            }
            for key, transcript in replayed["freeze_transcripts"].items()
        },
    }


def replay_local_evidence(
    *,
    root: Path,
    receipt: Mapping[str, Any],
    expected_binding: Mapping[str, Any],
    admission: Mapping[str, Any],
    automatic_generator: Mapping[str, Any],
    tracking_batch_size: int,
    frame_count: int = FRAME_COUNT,
    height: int = HEIGHT,
    width: int = WIDTH,
) -> Mapping[str, Any]:
    """Pure LOCAL_SCHEMA replay; never returns an observer/route authorization."""

    require_exact_keys(receipt, LOCAL_RECEIPT_KEYS, "local evidence receipt")
    require_exact_keys(receipt["binding"], LOCAL_BINDING_KEYS, "local evidence binding")
    if dict(receipt["binding"]) != dict(expected_binding):
        raise SAM2ObserverEvidenceV15CR8Error("local evidence input binding differs")
    for key, value in expected_binding.items():
        require_sha256(value, f"local evidence binding {key}")
    payload = dict(receipt)
    claimed = payload.pop("receipt_sha256", None)
    if require_sha256(claimed, "local evidence receipt hash") != object_sha256(payload):
        raise SAM2ObserverEvidenceV15CR8Error("local evidence self-hash differs")
    runs = receipt["runs"]
    proposal_count = receipt["proposal_count"]
    if (
        receipt["schema_version"] != LOCAL_SCHEMA
        or receipt["status"] != "LOCAL_SCHEMA_ARTIFACTS_PUBLISHED_REMOTE_UNVERIFIED"
        or type(proposal_count) is not int
        or not 1 <= proposal_count <= int(admission["maximum_distinct_proposals"])
        or type(runs) is not list
        or len(runs) != 2
        or receipt["local_schema_replay_only"] is not True
        or receipt["remote_worker_execution_verified"] is not False
        or receipt["observer_execution_authorized"] is not False
        or receipt["localization_semantically_certified"] is not False
        or receipt["scientific_claim_authorized"] is not False
        or receipt["route_authorized"] is not False
        or receipt["decode_authorized"] is not False
        or receipt["training_authorized"] is not False
    ):
        raise SAM2ObserverEvidenceV15CR8Error("local-only claim boundary differs")

    replayed_runs = []
    semantic_runs = []
    for run_index, run in enumerate(runs):
        run_ordinal = run_index + 1
        replayed = replay_worker_run(
            root=root,
            run=run,
            run_ordinal=run_ordinal,
            proposal_count=proposal_count,
            expected_binding=expected_binding,
            admission=admission,
            automatic_generator=automatic_generator,
            tracking_batch_size=tracking_batch_size,
            frame_count=frame_count,
            height=height,
            width=width,
        )
        replayed_runs.append(replayed)
        semantic_runs.append(semantic_run_payload(replayed))
    semantic_hashes = [object_sha256(row) for row in semantic_runs]
    if (
        semantic_hashes[0] != semantic_hashes[1]
        or receipt["repeat_semantic_sha256"] != semantic_hashes[0]
    ):
        raise SAM2ObserverEvidenceV15CR8Error("two-run SAM2 tensor replay differs")
    result = {
        "schema_version": LOCAL_REPLAY_SCHEMA,
        "status": "LOCAL_SCHEMA_REPLAY_PASS_REMOTE_OBSERVER_UNVERIFIED",
        "proposal_count": proposal_count,
        "repeat_semantic_sha256": semantic_hashes[0],
        "runs": replayed_runs,
        "local_schema_replay_only": True,
        "remote_worker_execution_verified": False,
        "observer_execution_authorized": False,
        "localization_semantically_certified": False,
        "scientific_claim_authorized": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    result["receipt_sha256"] = object_sha256(result)
    return result

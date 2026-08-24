#!/usr/bin/env python3
"""Sealed source-only SAM2 proposal/track materializer for v15c-r8 LOCAL evidence.

SAM2 proposes instances on source frame 0 and tracks them over all 81 source
frames.  No text detector, anchor, target, renderer, role score, ROI, training,
or route is available here.  The full pipeline is repeated under a restored RNG
scope; any byte/order/state mismatch aborts before a receipt is published.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import struct
import sys
from typing import Any, Iterator, Mapping, Sequence

try:
    from . import sam2_observer_evidence_v15c_r8 as observer_evidence
except ImportError:  # pragma: no cover - flat sealed deployment
    import sam2_observer_evidence_v15c_r8 as observer_evidence


SPEC_SCHEMA = "bernini-e00-source-sam2-proposal-role-probe-v15c-r3"
TRACK_SCHEMA = "bernini-source-sam2-proposal-tracks-v15c-r3"
ARTIFACT_MANIFEST_SCHEMA = "bernini-source-sam2-proposal-track-artifacts-v15c-r3"
OUTPUT_MANIFEST_SCHEMA = "bernini-source-sam2-proposal-track-output-v15c-r3"
TRANSCRIPT_SCHEMA = "bernini-source-sam2-repeat-transcript-v15c-r3"
FREEZE_SCHEMA = "bernini-frozen-sam2-model-state-v15c-r3"
TRACKING_BATCH_SCHEMA = "bernini-sam2-tracking-batch-transcript-v15c-r3"
PROMPT_LOGIT_SCHEMA = "bernini-sam2-prompt-logit-transcript-v15c-r3"
PROPAGATION_LOGIT_SCHEMA = "bernini-sam2-propagation-logit-transcript-v15c-r3"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REPEAT_GATE_KEYS = (
    "first_transcript_self_hash",
    "second_transcript_self_hash",
    "equivalence_signature_equal",
    "proposal_rows_equal",
    "prompt_signatures_equal",
    "mask_signatures_equal",
    "phase_coverage_equal",
    "tracking_batches_equal",
    "freeze_receipts_equal",
    "rng_state_unchanged_outside_fork_rng",
    "sam2__C_absent_both_runs",
)
TRACK_RECEIPT_KEYS = (
    "schema_version",
    "status",
    "spec",
    "source",
    "runtime",
    "sam2",
    "phase_frames",
    "phase_grid",
    "proposal_count",
    "proposals",
    "repeat_transcripts",
    "repeat",
    "sam2__C_imported_before_or_after",
    "phase_coverage_tensor_sha256",
    "phase_coverage_array_sha256",
    "artifact_manifest_file_sha256",
    "artifact_manifest_internal_sha256",
    "observer_evidence_file_sha256",
    "observer_evidence_internal_sha256",
    "claim_limits",
    "receipt_sha256",
)
REPEAT_RECEIPT_KEYS = (
    "seed",
    "first_transcript_sha256",
    "second_transcript_sha256",
    "first_equivalence_sha256",
    "second_equivalence_sha256",
    "gates",
    "rng_state_before_sha256",
    "rng_state_after_sha256",
)
EXPECTED_SPEC_RAW_SHA256 = (
    "d8932e965db30b5929e479527c46c27ad0395a9dc573a13b6e68c24720d3d1f8"
)
EXPECTED_SPEC_CANONICAL_SHA256 = (
    "d91c12566cc9e065e66ce6dfc466aaaf8a252d0653f3978b89844c1966b8bfb3"
)
EXPECTED_SOURCE_SHA256 = (
    "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
)
EXPECTED_CONFIG_SHA256 = (
    "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
)
PHASE_FRAMES = tuple(range(0, 81, 4))
GRID_HEIGHT = 37
GRID_WIDTH = 25
FRAME_COUNT = 81
MAXIMUM_PROPOSALS = 64


def require_exact_keys(value: Any, keys: Sequence[str], label: str) -> None:
    if type(value) is not dict or set(value) != set(keys) or len(value) != len(keys):
        raise SourceSAM2ProposalTracksV15CError(f"{label} exact keys differ")


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise SourceSAM2ProposalTracksV15CError(f"{label} is not lowercase SHA256")
    return value


class SourceSAM2ProposalTracksV15CError(RuntimeError):
    """A sealed source-only SAM2 materialization contract was violated."""


def canonical_bytes(value: Any) -> bytes:
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


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _tensor_artifact(
    *,
    root: Path,
    relative_path: str,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, str],
    schema: str,
) -> Mapping[str, Any]:
    """Publish one standard safetensors artifact and immediately reopen it."""

    import numpy as np
    from safetensors.numpy import save_file

    path = root / relative_path
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    ordered = {
        key: np.ascontiguousarray(arrays[key]) for key in sorted(arrays)
    }
    save_file(ordered, str(path), metadata=dict(metadata))
    raw = path.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length].decode("utf-8"))
    tensor_order = [key for key in header if key != "__metadata__"]
    if set(tensor_order) != set(ordered):
        raise SourceSAM2ProposalTracksV15CError(
            "safetensors writer tensor registry differs"
        )
    descriptor = {
        "schema_version": schema,
        "relative_path": relative_path,
        "file_sha256": file_sha256(path),
        "tensor_order": tensor_order,
        "tensor_array_sha256": {
            key: array_sha256(ordered[key]) for key in tensor_order
        },
    }
    # The worker cannot publish a receipt for bytes it has not itself reopened.
    contract = {
        key: (
            {
                "float32": "F32",
                "float64": "F64",
                "int64": "I64",
                "int32": "I32",
                "uint8": "U8",
                "int8": "I8",
                "bool": "BOOL",
            }[str(value.dtype)],
            tuple(int(item) for item in value.shape),
        )
        for key in tensor_order
        for value in (ordered[key],)
    }
    observer_evidence.strict_safetensors(
        path,
        expected_order=tuple(tensor_order),
        expected_contract=contract,
        expected_file_sha256=descriptor["file_sha256"],
        expected_array_sha256=descriptor["tensor_array_sha256"],
        expected_metadata=metadata,
    )
    return descriptor


def _torch_tensor_digest(value: Any) -> str:
    import torch

    tensor = value.detach().contiguous().to("cpu")
    metadata = {
        "dtype": str(tensor.dtype),
        "shape": [int(item) for item in tensor.shape],
    }
    digest = hashlib.sha256(canonical_bytes(metadata))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _model_tensor_manifest(model: Any, *, tensor_kind: str) -> Mapping[str, Any]:
    import torch

    if tensor_kind == "parameters":
        rows = model.named_parameters()
    elif tensor_kind == "buffers":
        rows = model.named_buffers()
    else:
        raise SourceSAM2ProposalTracksV15CError("model tensor kind differs")
    entries = []
    stream = hashlib.sha256()
    for name, tensor in sorted(rows, key=lambda item: item[0]):
        value = tensor.detach().contiguous().to("cpu")
        metadata = {
            "name": name,
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
        stream.update(canonical_bytes(metadata))
        stream.update(value.view(torch.uint8).numpy().tobytes(order="C"))
        entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": [int(item) for item in tensor.shape],
                "numel": int(tensor.numel()),
                "array_sha256": _torch_tensor_digest(tensor),
            }
        )
    if not entries:
        raise SourceSAM2ProposalTracksV15CError("model tensor manifest is empty")
    manifest = {
        "schema_version": observer_evidence.MODEL_MANIFEST_SCHEMA,
        "tensor_kind": tensor_kind,
        "tensor_count": len(entries),
        "element_count": sum(int(row["numel"]) for row in entries),
        "entries": entries,
        "stream_sha256": stream.hexdigest(),
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    return manifest


def _model_state_manifest(model: Any) -> Mapping[str, Any]:
    parameters = list(model.parameters())
    state = {
        "eval_mode": model.training is False,
        "requires_grad_true_count": sum(
            int(parameter.requires_grad) for parameter in parameters
        ),
        "non_none_grad_count": sum(
            int(parameter.grad is not None) for parameter in parameters
        ),
        "parameters": _model_tensor_manifest(model, tensor_kind="parameters"),
        "buffers": _model_tensor_manifest(model, tensor_kind="buffers"),
    }
    state["state_sha256"] = object_sha256(
        {
            "parameters_manifest_sha256": state["parameters"]["manifest_sha256"],
            "buffers_manifest_sha256": state["buffers"]["manifest_sha256"],
        }
    )
    return state


def _freeze_transcript_before(
    model: Any,
    *,
    run_ordinal: int,
    model_kind: str,
    construction_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    model.eval()
    model.requires_grad_(False)
    observed_class = f"{model.__class__.__module__}.{model.__class__.__name__}"
    if observed_class != construction_binding.get("model_class"):
        raise SourceSAM2ProposalTracksV15CError("constructed SAM2 model class differs")
    before = _model_state_manifest(model)
    if (
        before["eval_mode"] is not True
        or before["requires_grad_true_count"] != 0
        or before["non_none_grad_count"] != 0
    ):
        raise SourceSAM2ProposalTracksV15CError("model freeze precondition differs")
    return {
        "schema_version": observer_evidence.FREEZE_SCHEMA,
        "run_ordinal": run_ordinal,
        "model_kind": model_kind,
        "evidence_mode": "sealed_deterministic_worker_tensor_manifest",
        "construction_binding": dict(construction_binding),
        "before": before,
    }


def _freeze_transcript_after(model: Any, before: Mapping[str, Any]) -> Mapping[str, Any]:
    after = _model_state_manifest(model)
    result = dict(before)
    result["after"] = after
    result["all_freeze_gates_pass"] = bool(before["before"] == after)
    result["transcript_sha256"] = object_sha256(result)
    if result["all_freeze_gates_pass"] is not True:
        raise SourceSAM2ProposalTracksV15CError("model bytes changed during inference")
    try:
        observer_evidence.validate_freeze_transcript(
            result,
            run_ordinal=result["run_ordinal"],
            model_kind=result["model_kind"],
            expected_binding=result["construction_binding"],
        )
    except observer_evidence.SAM2ObserverEvidenceV15CR8Error as error:
        raise SourceSAM2ProposalTracksV15CError(str(error)) from error
    return result


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    claimed = payload.pop(field, None)
    observed = object_sha256(payload)
    if claimed is not None and claimed != observed:
        raise SourceSAM2ProposalTracksV15CError(f"{field} self-hash differs")
    return observed


def validate_spec(value: Any) -> None:
    """Validate every spec field through the external canonical pin."""

    if (
        type(value) is not dict
        or value.get("schema_version") != SPEC_SCHEMA
        or object_sha256(value) != EXPECTED_SPEC_CANONICAL_SHA256
    ):
        raise SourceSAM2ProposalTracksV15CError("sealed canonical spec differs")
    source = value["source"]
    sam2 = value["sam2"]
    runtime = sam2["runtime"]
    hydra = sam2["hydra"]
    generator = sam2["automatic_generator"]
    tracking = sam2["tracking"]
    determinism = sam2["determinism"]
    admission = sam2["proposal_admission"]
    r6 = value["r6"]
    roles = value["role_assignment"]
    observer = value["source_object_observer"]
    execution = value["execution"]
    claims = value["claim_limits"]
    if (
        value["event_id"] != "pour-liquid-into-cup"
        or source["sha256"] != EXPECTED_SOURCE_SHA256
        or [source[key] for key in ("frame_count", "fps", "width", "height")]
        != [81, 25.0, 704, 1056]
        or source["phase_frames"] != list(PHASE_FRAMES)
        or runtime["python_version"] != "3.12.13"
        or runtime["torch_version"] != "2.7.1+rocm6.3"
        or runtime["torch_hip_version"] != "6.3.42131-fa1d09cbd"
        or runtime["numpy_version"] != "1.26.4"
        or runtime["opencv_version"] != "4.11.0"
        or runtime["hydra_version"] != "1.3.2"
        or runtime["omegaconf_version"] != "2.3.0"
        or runtime["safetensors_version"] != "0.8.0-rc.0"
        or sam2["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256
        or hydra["actual_config_authority_sha256"] != EXPECTED_CONFIG_SHA256
        or hydra["image_apply_postprocessing"] is not False
        or hydra["video_apply_postprocessing"] is not False
        or generator["min_mask_region_area"] != 0
        or generator["point_grids"] is not None
        or tracking["fill_hole_area"] != 0
        or tracking["non_overlap_masks"] is not False
        or tracking["all_81_frames_required"] is not True
        or tracking["whole_object_gate_scope"]
        != "source_mask_geometry_only_for_every_proposal_no_material_or_transparency_label"
        or admission["maximum_distinct_proposals"] != MAXIMUM_PROPOSALS
        or admission["overflow_policy"]
        != "fail_closed_without_ranking_or_truncation"
        or determinism["full_automatic_proposal_and_81_frame_track_repeat_required"]
        is not True
        or determinism["sam2__C_imported_before_or_after"] is not False
        or determinism["sam2__C_required"] is not False
        or determinism["repeat_transcript_schema"] != TRANSCRIPT_SCHEMA
        or determinism["freeze_receipt_schema"] != FREEZE_SCHEMA
        or determinism["tracking_batch_schema"] != TRACKING_BATCH_SCHEMA
        or determinism["prompt_logit_schema"] != PROMPT_LOGIT_SCHEMA
        or determinism["propagation_logit_schema"] != PROPAGATION_LOGIT_SCHEMA
        or determinism["repeat_gate_keys"] != list(REPEAT_GATE_KEYS)
        or r6["null_span_count"] != 64
        or r6["raw_role_null_and_shuffled_affinity_only"] is not True
        or r6["calibration_masks_consumed"] is not False
        or roles["familywise_role_count"] != 3
        or roles["forced_assignment"] is not False
        or roles["roi_or_manual_box_consumed"] is not False
        or roles["family_overlap_nesting_policy"]
        != "all_members_unassigned_before_role_competition"
        or observer != {
            "source_pixels_and_sam2_masks_only": True,
            "anchor_consumed": False,
            "target_instruction_consumed": False,
            "material_or_transparency_classification": False,
            "transparent_or_reflective_fragment_policy": "fail_closed_unless_source_generic_whole_object_spatial_extent_and_temporal_continuity_gates_pass",
            "family_overlap_nesting_conflict_policy": "leave_all_conflicting_proposals_unassigned",
            "semantic_whole_object_certified": False,
        }
        or execution != {
            "parent_job_id": 143808,
            "required_node": "auh7-1b-gpu-292",
            "required_visible_gpu_count": 1,
            "release_core_member_count": 8,
            "sealed_code_snapshot_file_count": 9,
            "construction_code_snapshot_directory_mode": "0700",
            "sealed_code_snapshot_directory_mode": "0500",
            "sealed_code_snapshot_member_mode": "0400",
            "construction_mode_receipt_semantics": "historical_observation_before_sealing",
            "sealed_mode_receipt_semantics": "current_state_reverified_at_runtime",
            "python_flags": ["-E", "-s", "-B"],
            "python_environment_cleanup_required": True,
            "stage_and_final_code_hash_revalidation_required": True,
            "complete_manifest_required": True,
            "fresh_output_required": True,
            "parent_cancel_forbidden": True,
            "scancel_forbidden": True,
            "observer_only": True,
        }
        or claims["observer_only"] is not True
        or claims["renderer_forward_calls"] != 0
        or claims["optimizer_updates"] != 0
        or claims["training_authorized"] is not False
        or claims["decode_authorized"] is not False
        or claims["route_authorized"] is not False
    ):
        raise SourceSAM2ProposalTracksV15CError("sealed E00 v15c-r3 semantics differ")


def read_spec(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SourceSAM2ProposalTracksV15CError("spec must be one regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SPEC_RAW_SHA256:
        raise SourceSAM2ProposalTracksV15CError("sealed raw spec differs")
    try:
        value = json.loads(raw)
    except Exception as error:
        raise SourceSAM2ProposalTracksV15CError("spec JSON differs") from error
    validate_spec(value)
    return value


def mask_iou(left: Any, right: Any) -> float:
    import numpy as np

    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return float(intersection / union) if union else 0.0


def admit_automatic_proposals(
    annotations: Sequence[Mapping[str, Any]],
    *,
    image_area: int,
    minimum_area_pixels: int,
    maximum_area_fraction: float,
    near_duplicate_iou: float,
    maximum_distinct_proposals: int,
) -> list[Mapping[str, Any]]:
    """Apply source-generic geometry/NMS, then stable-sort by full mask SHA."""

    import numpy as np

    ranked: list[tuple[float, str, Mapping[str, Any]]] = []
    for annotation in annotations:
        mask = annotation.get("segmentation")
        area = annotation.get("area")
        predicted_iou = annotation.get("predicted_iou")
        stability = annotation.get("stability_score")
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.bool_
            or mask.ndim != 2
            or type(area) not in (int, np.integer)
            or int(area) != int(mask.sum())
            or int(area) < int(minimum_area_pixels)
            or int(area) / float(image_area) > float(maximum_area_fraction)
            or type(predicted_iou) not in (float, int, np.floating)
            or type(stability) not in (float, int, np.floating)
            or not math.isfinite(float(predicted_iou))
            or not math.isfinite(float(stability))
            or not 0.0 <= float(predicted_iou) <= 1.0
            or not 0.0 <= float(stability) <= 1.0
        ):
            continue
        digest = array_sha256(mask.astype(np.uint8))
        quality = 0.5 * float(predicted_iou) + 0.5 * float(stability)
        ranked.append((-quality, digest, annotation))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected: list[Mapping[str, Any]] = []
    for _negative_quality, _digest, annotation in ranked:
        if all(
            mask_iou(annotation["segmentation"], kept["segmentation"])
            < near_duplicate_iou
            for kept in selected
        ):
            selected.append(annotation)
    if not 1 <= len(selected) <= int(maximum_distinct_proposals):
        reason = "no automatic proposal survived" if not selected else (
            "automatic proposal overflow; fail closed without truncation"
        )
        raise SourceSAM2ProposalTracksV15CError(reason)
    return sorted(
        selected,
        key=lambda row: array_sha256(row["segmentation"].astype(np.uint8)),
    )


def mask_geometry(mask: Any) -> Mapping[str, Any]:
    import cv2
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {
            "visible": False,
            "area": 0,
            "centroid_xy": None,
            "bbox_xyxy": None,
            "bbox_fill_fraction": None,
            "bbox_diagonal_frame_fraction": None,
            "largest_component_fraction": None,
        }
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8
    )
    largest_component = (
        int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else 0
    )
    frame_height, frame_width = mask.shape
    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    return {
        "visible": True,
        "area": int(len(xs)),
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_fill_fraction": float(len(xs) / bbox_area),
        "bbox_diagonal_frame_fraction": float(
            math.hypot(x1 - x0, y1 - y0) / math.hypot(frame_width, frame_height)
        ),
        "largest_component_fraction": float(largest_component / len(xs)),
    }


def track_geometry_receipt(
    masks: Sequence[Any],
    prompt_mask: Any,
    *,
    width: int,
    height: int,
    tracking_spec: Mapping[str, Any],
) -> Mapping[str, Any]:
    import numpy as np

    if len(masks) != FRAME_COUNT:
        raise SourceSAM2ProposalTracksV15CError("track must contain exactly 81 masks")
    rows = [mask_geometry(mask) for mask in masks]
    areas = np.asarray([row["area"] for row in rows], dtype=np.float64)
    visible = areas > 0
    adjacent_iou = np.asarray(
        [mask_iou(masks[index], masks[index + 1]) for index in range(80)],
        dtype=np.float64,
    )
    diagonal = math.hypot(width, height)
    centroid_steps: list[float] = []
    for first, second in zip(rows[:-1], rows[1:]):
        if not first["visible"] or not second["visible"]:
            centroid_steps.append(float("inf"))
        else:
            x0, y0 = first["centroid_xy"]
            x1, y1 = second["centroid_xy"]
            centroid_steps.append(math.hypot(x1 - x0, y1 - y0) / diagonal)
    steps = np.asarray(centroid_steps, dtype=np.float64)
    area_ratio = (
        float(
            np.quantile(areas[visible], 0.95)
            / max(np.quantile(areas[visible], 0.05), 1.0)
        )
        if bool(visible.any())
        else float("inf")
    )
    seed_iou = mask_iou(masks[0], prompt_mask)
    p95_step = float(np.quantile(steps, 0.95))
    median_iou = float(np.median(adjacent_iou))
    visible_rows = [row for row in rows if row["visible"]]
    p10_area = float(np.quantile(areas[visible], 0.10)) if visible_rows else 0.0
    median_largest_component = (
        float(np.median([row["largest_component_fraction"] for row in visible_rows]))
        if visible_rows
        else 0.0
    )
    median_bbox_fill = (
        float(np.median([row["bbox_fill_fraction"] for row in visible_rows]))
        if visible_rows
        else 0.0
    )
    p10_bbox_diagonal = (
        float(
            np.quantile(
                [row["bbox_diagonal_frame_fraction"] for row in visible_rows], 0.10
            )
        )
        if visible_rows
        else 0.0
    )
    gates = {
        "all_81_frames_visible": bool(visible.all()),
        "seed_prompt_iou": bool(
            seed_iou >= float(tracking_spec["minimum_seed_prompt_iou"])
        ),
        "area_p95_to_p05_ratio": bool(
            area_ratio <= float(tracking_spec["maximum_area_p95_to_p05_ratio"])
        ),
        "median_adjacent_iou": bool(
            median_iou >= float(tracking_spec["minimum_median_adjacent_iou"])
        ),
        "p95_centroid_step": bool(
            p95_step
            <= float(
                tracking_spec[
                    "maximum_p95_centroid_step_frame_diagonal_fraction"
                ]
            )
        ),
        "whole_object_area_extent": bool(
            p10_area >= float(tracking_spec["minimum_p10_area_pixels"])
            and p10_bbox_diagonal
            >= float(tracking_spec["minimum_p10_bbox_diagonal_frame_fraction"])
        ),
        "whole_object_component_integrity": bool(
            median_largest_component
            >= float(tracking_spec["minimum_median_largest_component_fraction"])
        ),
        "whole_object_bbox_support": bool(
            median_bbox_fill
            >= float(tracking_spec["minimum_median_bbox_fill_fraction"])
        ),
    }
    if any(type(value) is not bool for value in gates.values()):
        raise SourceSAM2ProposalTracksV15CError("geometry gates are not strict bools")
    return {
        "frame_geometry": rows,
        "seed_prompt_iou": seed_iou,
        "area_p95_to_p05_ratio": area_ratio if math.isfinite(area_ratio) else None,
        "median_adjacent_iou": median_iou if math.isfinite(median_iou) else None,
        "p95_centroid_step_frame_diagonal_fraction": (
            p95_step if math.isfinite(p95_step) else None
        ),
        "p10_area_pixels": p10_area,
        "median_largest_component_fraction": median_largest_component,
        "median_bbox_fill_fraction": median_bbox_fill,
        "p10_bbox_diagonal_frame_fraction": p10_bbox_diagonal,
        "whole_object_observer_scope": (
            "source_mask_geometry_only_no_material_or_transparency_label"
        ),
        "automatic_track_geometry_gates": gates,
        "automatic_track_geometry_gate_pass": bool(all(gates.values())),
    }


def _sam2_tree_receipt(root: Path) -> Mapping[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in (".py", ".yaml", ".so")
        ):
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": file_sha256(path),
                    "size": path.stat().st_size,
                }
            )
    return {
        "file_count": len(rows),
        "tree_sha256": object_sha256(rows),
    }


def _resolved_config_sha256(model_cfg: str, overrides: Sequence[str]) -> str:
    from hydra import compose
    from omegaconf import OmegaConf

    cfg = compose(config_name=model_cfg, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return object_sha256(OmegaConf.to_container(cfg, resolve=True))


def _module_state_sha256(model: Any, *, buffers: bool) -> tuple[str, int, int]:
    import torch

    rows = model.named_buffers() if buffers else model.named_parameters()
    digest = hashlib.sha256()
    tensor_count = 0
    element_count = 0
    for name, tensor in sorted(rows, key=lambda item: item[0]):
        value = tensor.detach().contiguous().to("cpu")
        metadata = {
            "name": name,
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
        raw = value.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(canonical_bytes(metadata))
        digest.update(raw)
        tensor_count += 1
        element_count += int(value.numel())
    return digest.hexdigest(), tensor_count, element_count


FREEZE_RECEIPT_KEYS = (
    "schema_version",
    "model_kind",
    "eval_mode_before",
    "eval_mode_after",
    "requires_grad_true_count_before",
    "requires_grad_true_count_after",
    "non_none_grad_count_before",
    "non_none_grad_count_after",
    "parameter_sha256_before",
    "parameter_sha256_after",
    "buffer_sha256_before",
    "buffer_sha256_after",
    "parameter_tensor_count",
    "parameter_element_count",
    "buffer_tensor_count",
    "buffer_element_count",
    "parameter_and_buffer_bytes_unchanged",
    "all_freeze_gates_pass",
)


def _freeze_receipt_before(model: Any, *, model_kind: str) -> Mapping[str, Any]:
    model.eval()
    model.requires_grad_(False)
    parameters = list(model.parameters())
    parameter_sha, parameter_tensors, parameter_elements = _module_state_sha256(
        model, buffers=False
    )
    buffer_sha, buffer_tensors, buffer_elements = _module_state_sha256(
        model, buffers=True
    )
    receipt = {
        "schema_version": FREEZE_SCHEMA,
        "model_kind": model_kind,
        "eval_mode_before": model.training is False,
        "requires_grad_true_count_before": sum(
            int(parameter.requires_grad) for parameter in parameters
        ),
        "non_none_grad_count_before": sum(
            int(parameter.grad is not None) for parameter in parameters
        ),
        "parameter_sha256_before": parameter_sha,
        "buffer_sha256_before": buffer_sha,
        "parameter_tensor_count": parameter_tensors,
        "parameter_element_count": parameter_elements,
        "buffer_tensor_count": buffer_tensors,
        "buffer_element_count": buffer_elements,
    }
    if (
        receipt["eval_mode_before"] is not True
        or receipt["requires_grad_true_count_before"] != 0
        or receipt["non_none_grad_count_before"] != 0
        or receipt["parameter_tensor_count"] <= 0
        or receipt["parameter_element_count"] <= 0
        or receipt["buffer_tensor_count"] <= 0
        or receipt["buffer_element_count"] <= 0
    ):
        raise SourceSAM2ProposalTracksV15CError("SAM2 freeze precondition differs")
    return receipt


def _freeze_receipt_after(model: Any, before: Mapping[str, Any]) -> Mapping[str, Any]:
    parameters = list(model.parameters())
    parameter_sha, _, _ = _module_state_sha256(model, buffers=False)
    buffer_sha, _, _ = _module_state_sha256(model, buffers=True)
    result = {
        "schema_version": before["schema_version"],
        "model_kind": before["model_kind"],
        "eval_mode_before": before["eval_mode_before"],
        "eval_mode_after": model.training is False,
        "requires_grad_true_count_before": before[
            "requires_grad_true_count_before"
        ],
        "requires_grad_true_count_after": sum(
            int(parameter.requires_grad) for parameter in parameters
        ),
        "non_none_grad_count_before": before["non_none_grad_count_before"],
        "non_none_grad_count_after": sum(
            int(parameter.grad is not None) for parameter in parameters
        ),
        "parameter_sha256_before": before["parameter_sha256_before"],
        "parameter_sha256_after": parameter_sha,
        "buffer_sha256_before": before["buffer_sha256_before"],
        "buffer_sha256_after": buffer_sha,
        "parameter_tensor_count": before["parameter_tensor_count"],
        "parameter_element_count": before["parameter_element_count"],
        "buffer_tensor_count": before["buffer_tensor_count"],
        "buffer_element_count": before["buffer_element_count"],
    }
    result["parameter_and_buffer_bytes_unchanged"] = bool(
        result["parameter_sha256_before"] == result["parameter_sha256_after"]
        and result["buffer_sha256_before"] == result["buffer_sha256_after"]
    )
    result["all_freeze_gates_pass"] = bool(
        result["parameter_and_buffer_bytes_unchanged"]
        and result["eval_mode_before"] is True
        and result["eval_mode_after"] is True
        and result["requires_grad_true_count_before"] == 0
        and result["requires_grad_true_count_after"] == 0
        and result["non_none_grad_count_before"] == 0
        and result["non_none_grad_count_after"] == 0
        and result["parameter_tensor_count"] > 0
        and result["parameter_element_count"] > 0
        and result["buffer_tensor_count"] > 0
        and result["buffer_element_count"] > 0
    )
    require_exact_keys(result, FREEZE_RECEIPT_KEYS, "freeze receipt")
    for field in (
        "parameter_sha256_before",
        "parameter_sha256_after",
        "buffer_sha256_before",
        "buffer_sha256_after",
    ):
        require_sha256(result[field], f"freeze receipt {field}")
    if result["all_freeze_gates_pass"] is not True:
        raise SourceSAM2ProposalTracksV15CError("SAM2 freeze/state gate differs")
    return result


def _rng_sha256() -> str:
    import numpy as np
    import torch

    payload = {
        "python": pickle.dumps(random.getstate(), protocol=4).hex(),
        "numpy": pickle.dumps(np.random.get_state(), protocol=4).hex(),
        "torch_cpu": torch.get_rng_state().numpy().tobytes().hex(),
        "torch_cuda": [
            state.cpu().numpy().tobytes().hex()
            for state in torch.cuda.get_rng_state_all()
        ],
    }
    return object_sha256(payload)


@contextmanager
def _repeat_rng_scope(seed: int) -> Iterator[None]:
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0], enabled=True):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _write_or_hash_mask(path: Path, mask: Any, *, write: bool) -> Mapping[str, Any]:
    import cv2
    import numpy as np

    value = np.ascontiguousarray(mask, dtype=np.bool_)
    if write:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), value.astype(np.uint8) * 255):
            raise SourceSAM2ProposalTracksV15CError("mask PNG write failed")
    return {
        "array_sha256": array_sha256(value.astype(np.uint8)),
        "relative_path": str(path),
    }


def _pipeline_once(
    *,
    spec: Mapping[str, Any],
    checkpoint_path: Path,
    frames: Sequence[Any],
    frame_dir: Path,
    artifact_root: Path,
    write_artifacts: bool,
    run_ordinal: int,
    construction_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Any]:
    import cv2
    import numpy as np
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2, build_sam2_video_predictor

    hydra_spec = spec["sam2"]["hydra"]
    generator_spec = dict(spec["sam2"]["automatic_generator"])
    admission = spec["sam2"]["proposal_admission"]
    tracking_spec = spec["sam2"]["tracking"]
    autocast_dtype = torch.bfloat16

    image_model = build_sam2(
        hydra_spec["model_cfg"],
        str(checkpoint_path),
        device="cuda",
        hydra_overrides_extra=list(hydra_spec["image_overrides"]),
        apply_postprocessing=False,
    )
    image_before = _freeze_receipt_before(image_model, model_kind="image_model")
    image_evidence_before = _freeze_transcript_before(
        image_model,
        run_ordinal=run_ordinal,
        model_kind="image_model",
        construction_binding=construction_bindings["image_model"],
    )
    generator = SAM2AutomaticMaskGenerator(image_model, **generator_spec)
    automatic_generator_class = (
        f"{generator.__class__.__module__}.{generator.__class__.__name__}"
    )
    automatic_generate_call_count = 0
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=autocast_dtype
    ):
        annotations = generator.generate(
            cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        )
        automatic_generate_call_count += 1
    image_freeze = _freeze_receipt_after(image_model, image_before)
    image_evidence_freeze = _freeze_transcript_after(
        image_model, image_evidence_before
    )
    if type(annotations) is not list or not annotations:
        raise SourceSAM2ProposalTracksV15CError("AMG returned no raw candidates")
    amg_masks = []
    amg_area = []
    amg_bbox = []
    amg_predicted = []
    amg_stability = []
    for annotation in annotations:
        if type(annotation) is not dict:
            raise SourceSAM2ProposalTracksV15CError("AMG annotation differs")
        mask = np.ascontiguousarray(annotation.get("segmentation"), dtype=np.uint8)
        if mask.shape != (1056, 704) or not bool(np.isin(mask, [0, 1]).all()):
            raise SourceSAM2ProposalTracksV15CError("AMG raw mask differs")
        predicted = float(annotation.get("predicted_iou"))
        stability = float(annotation.get("stability_score"))
        if (
            not math.isfinite(predicted)
            or not math.isfinite(stability)
            or not 0.0 <= predicted <= 1.0
            or not 0.0 <= stability <= 1.0
        ):
            raise SourceSAM2ProposalTracksV15CError("AMG score is outside [0,1]")
        amg_masks.append(mask)
        amg_area.append(int(annotation.get("area")))
        amg_bbox.append([float(item) for item in annotation.get("bbox")])
        amg_predicted.append(predicted)
        amg_stability.append(stability)
    amg_relative = f"observer_evidence/run_{run_ordinal}/amg.safetensors"
    amg_artifact = _tensor_artifact(
        root=artifact_root,
        relative_path=amg_relative,
        arrays={
            "area": np.asarray(amg_area, dtype=np.int64),
            "bbox_xywh": np.asarray(amg_bbox, dtype=np.float32),
            "masks": np.stack(amg_masks, axis=0).astype(np.uint8, copy=False),
            "predicted_iou": np.asarray(amg_predicted, dtype=np.float32),
            "stability_score": np.asarray(amg_stability, dtype=np.float32),
        },
        metadata={
            "schema_version": observer_evidence.AMG_SCHEMA,
            "run_ordinal": str(run_ordinal),
            "source_frame_index": "0",
        },
        schema=observer_evidence.AMG_SCHEMA,
    )
    del generator
    del image_model
    torch.cuda.empty_cache()

    selected = admit_automatic_proposals(
        annotations,
        image_area=int(spec["source"]["width"] * spec["source"]["height"]),
        minimum_area_pixels=int(admission["minimum_area_pixels"]),
        maximum_area_fraction=float(admission["maximum_area_fraction"]),
        near_duplicate_iou=float(admission["near_duplicate_iou"]),
        maximum_distinct_proposals=int(admission["maximum_distinct_proposals"]),
    )
    prompt_masks = []
    proposal_rows = []
    prompt_signatures = []
    for annotation in selected:
        mask = np.ascontiguousarray(annotation["segmentation"], dtype=np.bool_)
        digest = array_sha256(mask.astype(np.uint8))
        proposal_id = f"sam2-f000-{digest}"
        prompt_masks.append(mask)
        prompt_path = artifact_root / "prompts" / f"{proposal_id}.png"
        signature = _write_or_hash_mask(prompt_path, mask, write=write_artifacts)
        signature["relative_path"] = str(prompt_path.relative_to(artifact_root))
        prompt_signatures.append({"proposal_id": proposal_id, **signature})
        proposal_rows.append(
            {
                "proposal_id": proposal_id,
                "prompt_mask_sha256": digest,
                "prompt_relative_path": str(prompt_path.relative_to(artifact_root)),
                "area": int(mask.sum()),
                "bbox_xywh": [float(item) for item in annotation["bbox"]],
                "predicted_iou": float(annotation["predicted_iou"]),
                "stability_score": float(annotation["stability_score"]),
            }
        )
    proposal_ids = [row["proposal_id"] for row in proposal_rows]
    if (
        not 1 <= len(proposal_ids) <= MAXIMUM_PROPOSALS
        or len(set(proposal_ids)) != len(proposal_ids)
        or proposal_ids != sorted(proposal_ids)
    ):
        raise SourceSAM2ProposalTracksV15CError("full-SHA proposal registry differs")

    predictor = build_sam2_video_predictor(
        hydra_spec["model_cfg"],
        str(checkpoint_path),
        device="cuda",
        hydra_overrides_extra=list(hydra_spec["video_extra_overrides"]),
        apply_postprocessing=False,
    )
    video_before = _freeze_receipt_before(predictor, model_kind="video_model")
    video_evidence_before = _freeze_transcript_before(
        predictor,
        run_ordinal=run_ordinal,
        model_kind="video_model",
        construction_binding=construction_bindings["video_model"],
    )
    video_predictor_class = (
        f"{predictor.__class__.__module__}.{predictor.__class__.__name__}"
    )
    add_new_mask_call_count = 0
    propagate_in_video_call_count = 0
    phase_coverage = np.zeros(
        (len(proposal_rows), len(PHASE_FRAMES), GRID_HEIGHT, GRID_WIDTH),
        dtype=np.float32,
    )
    track_rows: list[Mapping[str, Any] | None] = [None] * len(proposal_rows)
    mask_signatures: list[Mapping[str, Any] | None] = [None] * len(proposal_rows)
    batch_receipts = []
    batch_size = int(tracking_spec["batch_size"])
    for batch_start in range(0, len(proposal_rows), batch_size):
        batch_stop = min(batch_start + batch_size, len(proposal_rows))
        object_ids = list(range(batch_start, batch_stop))
        state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        prompt_call_receipts = []
        prompt_artifacts = []
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=autocast_dtype
        ):
            for global_index in object_ids:
                result = predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=global_index,
                    mask=prompt_masks[global_index],
                )
                add_new_mask_call_count += 1
                if not isinstance(result, tuple) or len(result) != 3:
                    raise SourceSAM2ProposalTracksV15CError(
                        "SAM2 prompt return contract differs"
                    )
                prompt_frame, prompt_ids, prompt_logits = result
                prompt_ids = [int(item) for item in prompt_ids]
                expected_prompt_ids = list(range(batch_start, global_index + 1))
                if (
                    int(prompt_frame) != 0
                    or prompt_ids != expected_prompt_ids
                    or not isinstance(prompt_logits, torch.Tensor)
                    or tuple(prompt_logits.shape)
                    != (len(expected_prompt_ids), 1, 1056, 704)
                    or str(prompt_logits.dtype) != tracking_spec["logits_dtype"]
                    or not bool(torch.isfinite(prompt_logits).all().item())
                ):
                    raise SourceSAM2ProposalTracksV15CError(
                        "SAM2 prompt logits/out_ids contract differs"
                    )
                prompt_call_receipts.append(
                    {
                        "schema_version": PROMPT_LOGIT_SCHEMA,
                        "inserted_object_id": global_index,
                        "frame_index": 0,
                        "out_ids": prompt_ids,
                        "shape": [int(item) for item in prompt_logits.shape],
                        "dtype": str(prompt_logits.dtype),
                        "finite": True,
                        "logits_sha256": array_sha256(
                            prompt_logits.detach().float().cpu().numpy()
                        ),
                    }
                )
                prompt_array = np.ascontiguousarray(
                    prompt_logits.detach().float().cpu().numpy(), dtype=np.float32
                )
                call_index = len(prompt_artifacts)
                prompt_relative = (
                    f"observer_evidence/run_{run_ordinal}/"
                    f"batch_{len(batch_receipts):03d}/"
                    f"prompt_call_{call_index:03d}.safetensors"
                )
                prompt_artifacts.append(
                    _tensor_artifact(
                        root=artifact_root,
                        relative_path=prompt_relative,
                        arrays={"logits": prompt_array},
                        metadata={
                            "schema_version": observer_evidence.LOGIT_FILE_SCHEMA,
                            "kind": "prompt",
                            "run_ordinal": str(run_ordinal),
                            "batch_index": str(len(batch_receipts)),
                            "call_index": str(call_index),
                            "inserted_object_id": str(global_index),
                            "out_ids": ",".join(str(item) for item in prompt_ids),
                        },
                        schema=observer_evidence.LOGIT_FILE_SCHEMA,
                    )
                )
            tracks_by_id = {
                object_id: [None] * FRAME_COUNT for object_id in object_ids
            }
            frame_receipts = []
            propagation_artifacts = []
            seen_frames = []
            propagation_iterator = predictor.propagate_in_video(state)
            propagate_in_video_call_count += 1
            for frame_index, out_ids, logits in propagation_iterator:
                frame_index = int(frame_index)
                ids = [int(item) for item in out_ids]
                seen_frames.append(frame_index)
                if (
                    ids != object_ids
                    or not isinstance(logits, torch.Tensor)
                    or tuple(logits.shape)
                    != (len(object_ids), 1, 1056, 704)
                    or str(logits.dtype) != tracking_spec["logits_dtype"]
                    or not bool(torch.isfinite(logits).all().item())
                ):
                    raise SourceSAM2ProposalTracksV15CError(
                        "SAM2 propagation logits/out_ids contract differs"
                    )
                logits_cpu = logits.detach().float().cpu().numpy()
                frame_receipts.append(
                    {
                        "schema_version": PROPAGATION_LOGIT_SCHEMA,
                        "frame_index": frame_index,
                        "out_ids": ids,
                        "shape": [int(item) for item in logits.shape],
                        "dtype": str(logits.dtype),
                        "finite": True,
                        "logits_sha256": array_sha256(logits_cpu),
                    }
                )
                propagation_relative = (
                    f"observer_evidence/run_{run_ordinal}/"
                    f"batch_{len(batch_receipts):03d}/"
                    f"propagation_frame_{frame_index:05d}.safetensors"
                )
                propagation_artifacts.append(
                    _tensor_artifact(
                        root=artifact_root,
                        relative_path=propagation_relative,
                        arrays={
                            "logits": np.ascontiguousarray(
                                logits_cpu, dtype=np.float32
                            )
                        },
                        metadata={
                            "schema_version": observer_evidence.LOGIT_FILE_SCHEMA,
                            "kind": "propagation",
                            "run_ordinal": str(run_ordinal),
                            "batch_index": str(len(batch_receipts)),
                            "frame_index": str(frame_index),
                            "out_ids": ",".join(str(item) for item in ids),
                        },
                        schema=observer_evidence.LOGIT_FILE_SCHEMA,
                    )
                )
                for position, object_id in enumerate(ids):
                    tracks_by_id[object_id][frame_index] = np.ascontiguousarray(
                        logits_cpu[position, 0] > 0.0, dtype=np.bool_
                    )
        if seen_frames != list(range(FRAME_COUNT)):
            raise SourceSAM2ProposalTracksV15CError(
                "SAM2 propagation frame order differs"
            )
        for global_index in object_ids:
            masks = tracks_by_id[global_index]
            if any(item is None for item in masks):
                raise SourceSAM2ProposalTracksV15CError("SAM2 track is incomplete")
            proposal_id = proposal_rows[global_index]["proposal_id"]
            digests = []
            for frame_index, mask in enumerate(masks):
                mask_path = (
                    artifact_root / "masks" / proposal_id / f"{frame_index:05d}.png"
                )
                signature = _write_or_hash_mask(
                    mask_path, mask, write=write_artifacts
                )
                digests.append(signature["array_sha256"])
            geometry = track_geometry_receipt(
                masks,
                prompt_masks[global_index],
                width=int(spec["source"]["width"]),
                height=int(spec["source"]["height"]),
                tracking_spec=tracking_spec,
            )
            proposal_rows[global_index].update(geometry)
            track_rows[global_index] = geometry
            mask_signatures[global_index] = {
                "proposal_id": proposal_id,
                "frame_count": FRAME_COUNT,
                "mask_array_sha256_by_frame": digests,
            }
            for phase, frame_index in enumerate(PHASE_FRAMES):
                phase_coverage[global_index, phase] = cv2.resize(
                    np.asarray(masks[frame_index], dtype=np.float32),
                    (GRID_WIDTH, GRID_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
        batch_receipts.append(
            {
                "schema_version": TRACKING_BATCH_SCHEMA,
                "batch_index": len(batch_receipts),
                "batch_start": batch_start,
                "batch_stop": batch_stop,
                "object_ids": object_ids,
                "prompt_calls": prompt_call_receipts,
                "propagation_frames": frame_receipts,
            }
        )
        if len(propagation_artifacts) != FRAME_COUNT:
            raise SourceSAM2ProposalTracksV15CError(
                "propagation safetensors registry differs"
            )
        # Evidence descriptors intentionally contain no caller-reported
        # out_ids/shape/dtype/finite values; replay derives all of them from
        # batch indices and the safetensors bytes.
        batch_receipts[-1]["evidence_artifacts"] = {
            "schema_version": observer_evidence.RUN_SCHEMA,
            "batch_index": len(batch_receipts) - 1,
            "batch_start": batch_start,
            "batch_stop": batch_stop,
            "prompt_artifacts": prompt_artifacts,
            "propagation_artifacts": propagation_artifacts,
        }
        predictor.reset_state(state)
        del state
        torch.cuda.empty_cache()
    video_freeze = _freeze_receipt_after(predictor, video_before)
    video_evidence_freeze = _freeze_transcript_after(
        predictor, video_evidence_before
    )
    del predictor
    torch.cuda.empty_cache()

    if any(row is None for row in track_rows) or any(
        row is None for row in mask_signatures
    ):
        raise SourceSAM2ProposalTracksV15CError("track registry is incomplete")
    signature = {
        "proposal_rows": proposal_rows,
        "prompt_signatures": prompt_signatures,
        "mask_signatures": mask_signatures,
        "phase_coverage_array_sha256": array_sha256(phase_coverage),
        "tracking_batch_receipts": batch_receipts,
        "freeze_receipts": {
            "image_model": image_freeze,
            "video_model": video_freeze,
        },
        "evidence_run": {
            "schema_version": observer_evidence.RUN_SCHEMA,
            "run_ordinal": run_ordinal,
            "sam2_execution": {
                "sam2_python_module_loaded": bool("sam2" in sys.modules),
                "automatic_generator_class": automatic_generator_class,
                "video_predictor_class": video_predictor_class,
                "automatic_generate_call_count": automatic_generate_call_count,
                "add_new_mask_call_count": add_new_mask_call_count,
                "propagate_in_video_call_count": propagate_in_video_call_count,
                "sam2__C_imported": False,
            },
            "amg_artifact": amg_artifact,
            "tracking_batches": [
                row.pop("evidence_artifacts") for row in batch_receipts
            ],
            "freeze_transcripts": {
                "image_model": image_evidence_freeze,
                "video_model": video_evidence_freeze,
            },
        },
    }
    return signature, np.ascontiguousarray(phase_coverage)


TRANSCRIPT_KEYS = (
    "schema_version",
    "run_ordinal",
    "proposal_rows",
    "prompt_signatures",
    "mask_signatures",
    "phase_coverage",
    "tracking_batches",
    "freeze_receipts",
    "sam2__C_imported",
    "transcript_sha256",
)


def build_repeat_transcript(
    *,
    run_ordinal: int,
    proposal_rows: Sequence[Mapping[str, Any]],
    prompt_signatures: Sequence[Mapping[str, Any]],
    mask_signatures: Sequence[Mapping[str, Any]],
    phase_coverage: Any,
    tracking_batches: Sequence[Mapping[str, Any]],
    freeze_receipts: Mapping[str, Any],
    sam2__C_imported: bool,
) -> Mapping[str, Any]:
    """Build the exact, self-hashed transcript replayed independently by runner."""

    import numpy as np

    coverage = np.ascontiguousarray(phase_coverage)
    if (
        run_ordinal not in (1, 2)
        or coverage.dtype != np.float32
        or coverage.shape
        != (len(proposal_rows), len(PHASE_FRAMES), GRID_HEIGHT, GRID_WIDTH)
        or not bool(np.isfinite(coverage).all())
        or type(sam2__C_imported) is not bool
    ):
        raise SourceSAM2ProposalTracksV15CError("repeat transcript input differs")
    transcript: dict[str, Any] = {
        "schema_version": TRANSCRIPT_SCHEMA,
        "run_ordinal": run_ordinal,
        "proposal_rows": list(proposal_rows),
        "prompt_signatures": list(prompt_signatures),
        "mask_signatures": list(mask_signatures),
        "phase_coverage": {
            "shape": [int(item) for item in coverage.shape],
            "dtype": str(coverage.dtype),
            "array_sha256": array_sha256(coverage),
        },
        "tracking_batches": list(tracking_batches),
        "freeze_receipts": dict(freeze_receipts),
        "sam2__C_imported": sam2__C_imported,
    }
    transcript["transcript_sha256"] = object_sha256(transcript)
    require_exact_keys(transcript, TRANSCRIPT_KEYS, "repeat transcript")
    require_sha256(transcript["transcript_sha256"], "repeat transcript self hash")
    return transcript


def transcript_equivalence_sha256(transcript: Mapping[str, Any]) -> str:
    require_exact_keys(transcript, TRANSCRIPT_KEYS, "repeat transcript")
    payload = dict(transcript)
    payload.pop("run_ordinal")
    payload.pop("transcript_sha256")
    return object_sha256(payload)


def validate_repeat_transcript(
    transcript: Mapping[str, Any], *, proposal_count: int, run_ordinal: int
) -> None:
    """Strict schema validation shared by materializer, runner and postflight."""

    require_exact_keys(transcript, TRANSCRIPT_KEYS, "repeat transcript")
    claimed = require_sha256(
        transcript["transcript_sha256"], "repeat transcript self hash"
    )
    if (
        transcript["schema_version"] != TRANSCRIPT_SCHEMA
        or transcript["run_ordinal"] != run_ordinal
        or claimed != _self_hash(transcript, "transcript_sha256")
        or type(transcript["proposal_rows"]) is not list
        or len(transcript["proposal_rows"]) != proposal_count
        or type(transcript["prompt_signatures"]) is not list
        or len(transcript["prompt_signatures"]) != proposal_count
        or type(transcript["mask_signatures"]) is not list
        or len(transcript["mask_signatures"]) != proposal_count
        or type(transcript["tracking_batches"]) is not list
        or not transcript["tracking_batches"]
        or type(transcript["freeze_receipts"]) is not dict
        or list(transcript["freeze_receipts"]) != ["image_model", "video_model"]
        or type(transcript["sam2__C_imported"]) is not bool
    ):
        raise SourceSAM2ProposalTracksV15CError("repeat transcript schema differs")
    coverage = transcript["phase_coverage"]
    require_exact_keys(coverage, ("shape", "dtype", "array_sha256"), "coverage")
    if coverage != {
        "shape": [proposal_count, len(PHASE_FRAMES), GRID_HEIGHT, GRID_WIDTH],
        "dtype": "float32",
        "array_sha256": require_sha256(
            coverage["array_sha256"], "coverage array hash"
        ),
    }:
        raise SourceSAM2ProposalTracksV15CError("coverage transcript differs")
    for model_kind in ("image_model", "video_model"):
        freeze = transcript["freeze_receipts"][model_kind]
        require_exact_keys(freeze, FREEZE_RECEIPT_KEYS, "freeze receipt")
        if (
            freeze["schema_version"] != FREEZE_SCHEMA
            or freeze["model_kind"] != model_kind
            or freeze["eval_mode_before"] is not True
            or freeze["eval_mode_after"] is not True
            or freeze["requires_grad_true_count_before"] != 0
            or freeze["requires_grad_true_count_after"] != 0
            or freeze["non_none_grad_count_before"] != 0
            or freeze["non_none_grad_count_after"] != 0
            or freeze["parameter_tensor_count"] <= 0
            or freeze["parameter_element_count"] <= 0
            or freeze["buffer_tensor_count"] <= 0
            or freeze["buffer_element_count"] <= 0
            or freeze["parameter_sha256_before"]
            != freeze["parameter_sha256_after"]
            or freeze["buffer_sha256_before"] != freeze["buffer_sha256_after"]
            or freeze["parameter_and_buffer_bytes_unchanged"] is not True
            or freeze["all_freeze_gates_pass"] is not True
        ):
            raise SourceSAM2ProposalTracksV15CError("freeze transcript differs")
        for key in (
            "parameter_sha256_before",
            "parameter_sha256_after",
            "buffer_sha256_before",
            "buffer_sha256_after",
        ):
            require_sha256(freeze[key], f"freeze {key}")
    expected_ids: list[int] = []
    for batch_index, batch in enumerate(transcript["tracking_batches"]):
        require_exact_keys(
            batch,
            (
                "schema_version",
                "batch_index",
                "batch_start",
                "batch_stop",
                "object_ids",
                "prompt_calls",
                "propagation_frames",
            ),
            "tracking batch",
        )
        object_ids = batch["object_ids"]
        if (
            batch["schema_version"] != TRACKING_BATCH_SCHEMA
            or batch["batch_index"] != batch_index
            or type(object_ids) is not list
            or not object_ids
            or object_ids != list(range(batch["batch_start"], batch["batch_stop"]))
            or type(batch["prompt_calls"]) is not list
            or len(batch["prompt_calls"]) != len(object_ids)
            or type(batch["propagation_frames"]) is not list
            or len(batch["propagation_frames"]) != FRAME_COUNT
        ):
            raise SourceSAM2ProposalTracksV15CError("tracking batch order differs")
        expected_ids.extend(object_ids)
        for position, row in enumerate(batch["prompt_calls"]):
            require_exact_keys(
                row,
                (
                    "schema_version",
                    "inserted_object_id",
                    "frame_index",
                    "out_ids",
                    "shape",
                    "dtype",
                    "finite",
                    "logits_sha256",
                ),
                "prompt logits",
            )
            expected_out_ids = object_ids[: position + 1]
            if (
                row["schema_version"] != PROMPT_LOGIT_SCHEMA
                or row["inserted_object_id"] != object_ids[position]
                or row["frame_index"] != 0
                or row["out_ids"] != expected_out_ids
                or row["shape"] != [len(expected_out_ids), 1, 1056, 704]
                or row["dtype"] != "torch.float32"
                or row["finite"] is not True
            ):
                raise SourceSAM2ProposalTracksV15CError("prompt logits differ")
            require_sha256(row["logits_sha256"], "prompt logits hash")
        for frame_index, row in enumerate(batch["propagation_frames"]):
            require_exact_keys(
                row,
                (
                    "schema_version",
                    "frame_index",
                    "out_ids",
                    "shape",
                    "dtype",
                    "finite",
                    "logits_sha256",
                ),
                "propagation logits",
            )
            if (
                row["schema_version"] != PROPAGATION_LOGIT_SCHEMA
                or row["frame_index"] != frame_index
                or row["out_ids"] != object_ids
                or row["shape"] != [len(object_ids), 1, 1056, 704]
                or row["dtype"] != "torch.float32"
                or row["finite"] is not True
            ):
                raise SourceSAM2ProposalTracksV15CError("propagation logits differ")
            require_sha256(row["logits_sha256"], "propagation logits hash")
    if expected_ids != list(range(proposal_count)):
        raise SourceSAM2ProposalTracksV15CError("tracking object registry differs")


def _runtime_receipt(
    spec: Mapping[str, Any], *, config_arg: Path, checkpoint_arg: Path
) -> Mapping[str, Any]:
    import cv2
    import hydra
    import importlib.metadata
    import numpy as np
    import omegaconf
    import safetensors
    import sam2
    import torch

    runtime = spec["sam2"]["runtime"]
    hydra_spec = spec["sam2"]["hydra"]
    checkpoint_spec = spec["sam2"]["checkpoint"]
    root = Path(sam2.__path__[0]).resolve(strict=True)
    actual_config = (root / hydra_spec["model_cfg"]).resolve(strict=True)
    authority = Path(hydra_spec["actual_config_authority_path"]).resolve(strict=True)
    legacy_copy = Path(
        hydra_spec["legacy_byte_equal_copy_not_authority"]
    ).resolve(strict=True)
    configured_checkpoint = Path(checkpoint_spec["path"]).resolve(strict=True)
    key_modules = runtime["key_module_sha256"]
    observed_key_modules = {
        relative: file_sha256(root / relative) for relative in key_modules
    }
    tree = _sam2_tree_receipt(root)
    image_resolved = _resolved_config_sha256(
        hydra_spec["model_cfg"], hydra_spec["image_overrides"]
    )
    video_resolved = _resolved_config_sha256(
        hydra_spec["model_cfg"], hydra_spec["video_effective_overrides"]
    )
    receipt = {
        "python_executable": sys.executable,
        "python_executable_samefile_as_authority": os.path.samefile(
            sys.executable, runtime["python_executable"]
        ),
        "python_executable_sha256": file_sha256(Path(sys.executable)),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "sam2_distribution": f"SAM-2=={importlib.metadata.version('SAM-2')}",
        "sam2_package_root": str(root),
        "sam2_tree": tree,
        "key_module_sha256": observed_key_modules,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "hydra_version": hydra.__version__,
        "omegaconf_version": omegaconf.__version__,
        "safetensors_version": safetensors.__version__,
        "visible_gpu_count": torch.cuda.device_count(),
        "visible_gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.device_count() == 1 else None
        ),
        "actual_hydra_config_path": str(actual_config),
        "actual_hydra_config_samefile_as_authority": os.path.samefile(
            actual_config, authority
        ),
        "config_argument_samefile_as_actual": os.path.samefile(
            config_arg, actual_config
        ),
        "legacy_copy_samefile_as_actual": os.path.samefile(
            legacy_copy, actual_config
        ),
        "actual_hydra_config_sha256": file_sha256(actual_config),
        "legacy_copy_sha256": file_sha256(legacy_copy),
        "image_resolved_config_canonical_sha256": image_resolved,
        "video_resolved_config_canonical_sha256": video_resolved,
        "checkpoint_argument_samefile_as_authority": os.path.samefile(
            checkpoint_arg, configured_checkpoint
        ),
        "checkpoint_sha256": file_sha256(checkpoint_arg),
        "checkpoint_size": checkpoint_arg.stat().st_size,
        "sam2__C_imported": "sam2._C" in sys.modules,
    }
    expected = {
        "python_executable": runtime["python_executable"],
        "python_executable_samefile_as_authority": True,
        "python_executable_sha256": runtime["python_executable_sha256"],
        "python_version": runtime["python_version"],
        "sam2_distribution": runtime["sam2_distribution"],
        "sam2_package_root": runtime["sam2_package_root"],
        "torch_version": runtime["torch_version"],
        "torch_hip_version": runtime["torch_hip_version"],
        "numpy_version": runtime["numpy_version"],
        "opencv_version": runtime["opencv_version"],
        "hydra_version": runtime["hydra_version"],
        "omegaconf_version": runtime["omegaconf_version"],
        "safetensors_version": runtime["safetensors_version"],
        "visible_gpu_count": spec["execution"]["required_visible_gpu_count"],
        "visible_gpu_name": runtime["required_device_name"],
        "actual_hydra_config_path": hydra_spec["actual_config_authority_path"],
        "actual_hydra_config_samefile_as_authority": True,
        "config_argument_samefile_as_actual": True,
        "legacy_copy_samefile_as_actual": False,
        "actual_hydra_config_sha256": hydra_spec[
            "actual_config_authority_sha256"
        ],
        "legacy_copy_sha256": hydra_spec["actual_config_authority_sha256"],
        "image_resolved_config_canonical_sha256": hydra_spec[
            "image_resolved_config_canonical_sha256"
        ],
        "video_resolved_config_canonical_sha256": hydra_spec[
            "video_resolved_config_canonical_sha256"
        ],
        "checkpoint_argument_samefile_as_authority": True,
        "checkpoint_sha256": checkpoint_spec["sha256"],
        "checkpoint_size": checkpoint_spec["size"],
        "sam2__C_imported": False,
    }
    for key, value in expected.items():
        if receipt[key] != value:
            raise SourceSAM2ProposalTracksV15CError(
                f"pinned SAM2 runtime field differs: {key}"
            )
    if (
        tree["file_count"] != runtime["sam2_python_yaml_tree_file_count"]
        or tree["tree_sha256"] != runtime["sam2_python_yaml_tree_sha256"]
        or observed_key_modules != key_modules
    ):
        raise SourceSAM2ProposalTracksV15CError("pinned SAM2 source tree differs")
    return receipt


def _file_manifest(root: Path, files: Sequence[Path]) -> Mapping[str, Any]:
    rows = {}
    for path in sorted(files):
        if not path.is_file() or path.is_symlink():
            raise SourceSAM2ProposalTracksV15CError("manifest member differs")
        rows[str(path.relative_to(root))] = {
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config-authority", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if sys.platform != "linux":
        raise SourceSAM2ProposalTracksV15CError("real SAM2 materialization requires Linux")
    spec_path = args.spec.resolve(strict=True)
    source_path = args.source_video.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    config_path = args.config_authority.resolve(strict=True)
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise SourceSAM2ProposalTracksV15CError("refusing to reuse an output directory")
    spec = read_spec(spec_path)
    if (
        not os.path.samefile(source_path, Path(spec["source"]["path"]))
        or file_sha256(source_path) != EXPECTED_SOURCE_SHA256
    ):
        raise SourceSAM2ProposalTracksV15CError("source file binding differs")

    import cv2
    import numpy as np
    import torch
    from safetensors.numpy import save_file

    if not torch.cuda.is_available():
        raise SourceSAM2ProposalTracksV15CError("one ROCm-visible GPU is required")
    runtime_receipt = _runtime_receipt(
        spec, config_arg=config_path, checkpoint_arg=checkpoint_path
    )
    if "sam2._C" in sys.modules:
        raise SourceSAM2ProposalTracksV15CError("sam2._C ambiguity is forbidden")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise SourceSAM2ProposalTracksV15CError("source video could not be opened")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if (len(frames), fps, width, height) != (81, 25.0, 704, 1056):
        raise SourceSAM2ProposalTracksV15CError("decoded source media differs")

    local_binding = {
        "spec_raw_sha256": file_sha256(spec_path),
        "spec_canonical_sha256": object_sha256(spec),
        "source_video_sha256": file_sha256(source_path),
        "source_frame0_array_sha256": array_sha256(
            np.ascontiguousarray(frames[0], dtype=np.uint8)
        ),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_sha256": file_sha256(config_path),
        "sam2_tree_sha256": runtime_receipt["sam2_tree"]["tree_sha256"],
        "key_module_sha256": object_sha256(runtime_receipt["key_module_sha256"]),
        "image_resolved_config_sha256": runtime_receipt[
            "image_resolved_config_canonical_sha256"
        ],
        "video_resolved_config_sha256": runtime_receipt[
            "video_resolved_config_canonical_sha256"
        ],
        "worker_code_sha256": file_sha256(Path(__file__).resolve(strict=True)),
    }
    observer_evidence.require_exact_keys(
        local_binding,
        observer_evidence.LOCAL_BINDING_KEYS,
        "local evidence binding",
    )
    construction_bindings = {}
    for model_kind in observer_evidence.MODEL_KINDS:
        construction_bindings[model_kind] = {
            "source_video_sha256": local_binding["source_video_sha256"],
            "source_frame0_array_sha256": local_binding[
                "source_frame0_array_sha256"
            ],
            "checkpoint_sha256": local_binding["checkpoint_sha256"],
            "config_sha256": local_binding["config_sha256"],
            "sam2_tree_sha256": local_binding["sam2_tree_sha256"],
            "key_module_sha256": local_binding["key_module_sha256"],
            "resolved_config_sha256": local_binding[
                "image_resolved_config_sha256"
                if model_kind == "image_model"
                else "video_resolved_config_sha256"
            ],
            "worker_code_sha256": local_binding["worker_code_sha256"],
            "model_class": (
                "sam2.modeling.sam2_base.SAM2Base"
                if model_kind == "image_model"
                else "sam2.sam2_video_predictor.SAM2VideoPredictor"
            ),
        }

    output.mkdir(mode=0o700, parents=True)
    frame_dir = output / "_decoded_jpeg_frames"
    frame_dir.mkdir(mode=0o700)
    for index, frame in enumerate(frames):
        if not cv2.imwrite(
            str(frame_dir / f"{index:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 98],
        ):
            raise SourceSAM2ProposalTracksV15CError("source JPEG materialization failed")

    rng_before = _rng_sha256()
    seed = int(spec["sam2"]["determinism"]["repeat_seed"])
    with _repeat_rng_scope(seed):
        first_signature, first_coverage = _pipeline_once(
            spec=spec,
            checkpoint_path=checkpoint_path,
            frames=frames,
            frame_dir=frame_dir,
            artifact_root=output,
            write_artifacts=True,
            run_ordinal=1,
            construction_bindings=construction_bindings,
        )
    first_c_imported = "sam2._C" in sys.modules
    with _repeat_rng_scope(seed):
        second_signature, second_coverage = _pipeline_once(
            spec=spec,
            checkpoint_path=checkpoint_path,
            frames=frames,
            frame_dir=frame_dir,
            artifact_root=output,
            write_artifacts=False,
            run_ordinal=2,
            construction_bindings=construction_bindings,
        )
    second_c_imported = "sam2._C" in sys.modules
    rng_after = _rng_sha256()
    first_transcript = build_repeat_transcript(
        run_ordinal=1,
        proposal_rows=first_signature["proposal_rows"],
        prompt_signatures=first_signature["prompt_signatures"],
        mask_signatures=first_signature["mask_signatures"],
        phase_coverage=first_coverage,
        tracking_batches=first_signature["tracking_batch_receipts"],
        freeze_receipts=first_signature["freeze_receipts"],
        sam2__C_imported=first_c_imported,
    )
    second_transcript = build_repeat_transcript(
        run_ordinal=2,
        proposal_rows=second_signature["proposal_rows"],
        prompt_signatures=second_signature["prompt_signatures"],
        mask_signatures=second_signature["mask_signatures"],
        phase_coverage=second_coverage,
        tracking_batches=second_signature["tracking_batch_receipts"],
        freeze_receipts=second_signature["freeze_receipts"],
        sam2__C_imported=second_c_imported,
    )
    validate_repeat_transcript(
        first_transcript,
        proposal_count=len(first_signature["proposal_rows"]),
        run_ordinal=1,
    )
    validate_repeat_transcript(
        second_transcript,
        proposal_count=len(second_signature["proposal_rows"]),
        run_ordinal=2,
    )
    first_equivalence = transcript_equivalence_sha256(first_transcript)
    second_equivalence = transcript_equivalence_sha256(second_transcript)
    repeat_gates = {
        "first_transcript_self_hash": bool(
            first_transcript["transcript_sha256"]
            == _self_hash(first_transcript, "transcript_sha256")
        ),
        "second_transcript_self_hash": bool(
            second_transcript["transcript_sha256"]
            == _self_hash(second_transcript, "transcript_sha256")
        ),
        "equivalence_signature_equal": bool(
            first_equivalence == second_equivalence
        ),
        "proposal_rows_equal": bool(
            first_transcript["proposal_rows"] == second_transcript["proposal_rows"]
        ),
        "prompt_signatures_equal": bool(
            first_transcript["prompt_signatures"]
            == second_transcript["prompt_signatures"]
        ),
        "mask_signatures_equal": bool(
            first_transcript["mask_signatures"]
            == second_transcript["mask_signatures"]
        ),
        "phase_coverage_equal": bool(
            np.array_equal(first_coverage, second_coverage)
            and first_transcript["phase_coverage"]
            == second_transcript["phase_coverage"]
        ),
        "tracking_batches_equal": bool(
            first_transcript["tracking_batches"]
            == second_transcript["tracking_batches"]
        ),
        "freeze_receipts_equal": bool(
            first_transcript["freeze_receipts"]
            == second_transcript["freeze_receipts"]
        ),
        "rng_state_unchanged_outside_fork_rng": bool(rng_before == rng_after),
        "sam2__C_absent_both_runs": bool(
            first_c_imported is False and second_c_imported is False
        ),
    }
    if (
        list(repeat_gates) != list(REPEAT_GATE_KEYS)
        or not repeat_gates
        or any(type(value) is not bool for value in repeat_gates.values())
        or not all(repeat_gates.values())
    ):
        raise SourceSAM2ProposalTracksV15CError(
            "full SAM2 repeat/RNG gate differs; strict NO-GO"
        )
    if first_c_imported or second_c_imported or "sam2._C" in sys.modules:
        raise SourceSAM2ProposalTracksV15CError(
            "sam2._C was imported despite both zero-area settings"
        )

    evidence_runs = [
        first_signature["evidence_run"],
        second_signature["evidence_run"],
    ]
    replayed_evidence_runs = []
    for ordinal, evidence_run in enumerate(evidence_runs, start=1):
        try:
            replayed_evidence_runs.append(
                observer_evidence.replay_worker_run(
                    root=output,
                    run=evidence_run,
                    run_ordinal=ordinal,
                    proposal_count=len(first_signature["proposal_rows"]),
                    expected_binding=local_binding,
                    admission=spec["sam2"]["proposal_admission"],
                    automatic_generator=spec["sam2"]["automatic_generator"],
                    tracking_batch_size=int(spec["sam2"]["tracking"]["batch_size"]),
                    frame_count=FRAME_COUNT,
                    height=int(spec["source"]["height"]),
                    width=int(spec["source"]["width"]),
                )
            )
        except observer_evidence.SAM2ObserverEvidenceV15CR8Error as error:
            raise SourceSAM2ProposalTracksV15CError(str(error)) from error
    evidence_semantic_hashes = [
        observer_evidence.object_sha256(
            observer_evidence.semantic_run_payload(value)
        )
        for value in replayed_evidence_runs
    ]
    if evidence_semantic_hashes[0] != evidence_semantic_hashes[1]:
        raise SourceSAM2ProposalTracksV15CError(
            "two complete SAM2 evidence runs differ"
        )
    local_evidence_receipt = {
        "schema_version": observer_evidence.LOCAL_SCHEMA,
        "status": "LOCAL_SCHEMA_ARTIFACTS_PUBLISHED_REMOTE_UNVERIFIED",
        "binding": local_binding,
        "proposal_count": len(first_signature["proposal_rows"]),
        "runs": evidence_runs,
        "repeat_semantic_sha256": evidence_semantic_hashes[0],
        "local_schema_replay_only": True,
        "remote_worker_execution_verified": False,
        "observer_execution_authorized": False,
        "localization_semantically_certified": False,
        "scientific_claim_authorized": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    local_evidence_receipt["receipt_sha256"] = observer_evidence.object_sha256(
        local_evidence_receipt
    )
    local_evidence_path = output / "observer_evidence/local_evidence.json"
    local_evidence_path.write_bytes(canonical_bytes(local_evidence_receipt))
    try:
        replayed_local_evidence = observer_evidence.replay_local_evidence(
            root=output,
            receipt=local_evidence_receipt,
            expected_binding=local_binding,
            admission=spec["sam2"]["proposal_admission"],
            automatic_generator=spec["sam2"]["automatic_generator"],
            tracking_batch_size=int(spec["sam2"]["tracking"]["batch_size"]),
            frame_count=FRAME_COUNT,
            height=int(spec["source"]["height"]),
            width=int(spec["source"]["width"]),
        )
    except observer_evidence.SAM2ObserverEvidenceV15CR8Error as error:
        raise SourceSAM2ProposalTracksV15CError(str(error)) from error
    if (
        replayed_local_evidence["observer_execution_authorized"] is not False
        or replayed_local_evidence["route_authorized"] is not False
    ):
        raise SourceSAM2ProposalTracksV15CError("LOCAL evidence claim boundary differs")

    coverage_path = output / "phase_coverage.safetensors"
    save_file(
        {"phase_coverage": np.ascontiguousarray(first_coverage, dtype=np.float32)},
        str(coverage_path),
    )
    shutil.rmtree(frame_dir)
    artifact_files = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json"}
    ]
    artifact_manifest = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "files": _file_manifest(output, artifact_files),
        "route_authorized": False,
        "training_authorized": False,
    }
    artifact_manifest["manifest_sha256"] = object_sha256(artifact_manifest)
    artifact_manifest_path = output / "artifact_manifest.json"
    artifact_manifest_path.write_bytes(canonical_bytes(artifact_manifest))

    proposal_rows = first_signature["proposal_rows"]
    receipt = {
        "schema_version": TRACK_SCHEMA,
        "status": "SOURCE_ONLY_SAM2_TRACK_CANDIDATES_REQUIRE_REJECT_ONLY_AUDIT",
        "spec": {
            "raw_sha256": file_sha256(spec_path),
            "canonical_sha256": object_sha256(spec),
        },
        "source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "frame_count": FRAME_COUNT,
            "fps": fps,
            "width": width,
            "height": height,
        },
        "runtime": runtime_receipt,
        "sam2": {
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "actual_config_authority_sha256": file_sha256(config_path),
            "hydra": spec["sam2"]["hydra"],
            "automatic_generator": spec["sam2"]["automatic_generator"],
            "proposal_admission": spec["sam2"]["proposal_admission"],
            "tracking": spec["sam2"]["tracking"],
        },
        "phase_frames": list(PHASE_FRAMES),
        "phase_grid": [GRID_HEIGHT, GRID_WIDTH],
        "proposal_count": len(proposal_rows),
        "proposals": proposal_rows,
        "repeat_transcripts": {
            "first": first_transcript,
            "second": second_transcript,
        },
        "repeat": {
            "seed": seed,
            "first_transcript_sha256": first_transcript["transcript_sha256"],
            "second_transcript_sha256": second_transcript["transcript_sha256"],
            "first_equivalence_sha256": first_equivalence,
            "second_equivalence_sha256": second_equivalence,
            "gates": repeat_gates,
            "rng_state_before_sha256": rng_before,
            "rng_state_after_sha256": rng_after,
        },
        "sam2__C_imported_before_or_after": False,
        "phase_coverage_tensor_sha256": file_sha256(coverage_path),
        "phase_coverage_array_sha256": array_sha256(first_coverage),
        "artifact_manifest_file_sha256": file_sha256(artifact_manifest_path),
        "artifact_manifest_internal_sha256": artifact_manifest["manifest_sha256"],
        "observer_evidence_file_sha256": file_sha256(local_evidence_path),
        "observer_evidence_internal_sha256": local_evidence_receipt[
            "receipt_sha256"
        ],
        "claim_limits": {
            "observer_only": True,
            "source_video_only": True,
            "text_detector_consumed": False,
            "manual_box_consumed": False,
            "r6_affinity_consumed": False,
            "anchor_consumed": False,
            "target_instruction_consumed": False,
            "material_or_transparency_classification_consumed": False,
            "semantic_whole_object_certified": False,
            "renderer_forward_calls": 0,
            "optimizer_updates": 0,
            "training_authorized": False,
            "decode_authorized": False,
            "route_authorized": False,
            "localization_semantically_certified": False,
        },
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    require_exact_keys(receipt, TRACK_RECEIPT_KEYS, "track receipt")
    require_exact_keys(receipt["repeat"], REPEAT_RECEIPT_KEYS, "repeat receipt")
    require_exact_keys(receipt["repeat"]["gates"], REPEAT_GATE_KEYS, "repeat gates")
    if not receipt["repeat"]["gates"] or any(
        type(value) is not bool or value is not True
        for value in receipt["repeat"]["gates"].values()
    ):
        raise SourceSAM2ProposalTracksV15CError("repeat gates differ")
    for key in (
        "receipt_sha256",
        "phase_coverage_tensor_sha256",
        "phase_coverage_array_sha256",
        "artifact_manifest_file_sha256",
        "artifact_manifest_internal_sha256",
        "observer_evidence_file_sha256",
        "observer_evidence_internal_sha256",
    ):
        require_sha256(receipt[key], f"track receipt {key}")
    for key in (
        "first_transcript_sha256",
        "second_transcript_sha256",
        "first_equivalence_sha256",
        "second_equivalence_sha256",
        "rng_state_before_sha256",
        "rng_state_after_sha256",
    ):
        require_sha256(receipt["repeat"][key], f"repeat receipt {key}")
    receipt_path = output / "track_receipt.json"
    receipt_path.write_bytes(canonical_bytes(receipt))

    final_files = [path for path in output.rglob("*") if path.is_file()]
    output_manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA,
        "files": _file_manifest(output, final_files),
        "route_authorized": False,
        "training_authorized": False,
    }
    output_manifest["manifest_sha256"] = _self_hash(
        output_manifest, "manifest_sha256"
    )
    output_manifest_path = output / "output_manifest.json"
    output_manifest_path.write_bytes(canonical_bytes(output_manifest))
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

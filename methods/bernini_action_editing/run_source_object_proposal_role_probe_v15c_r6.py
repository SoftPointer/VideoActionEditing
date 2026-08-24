#!/usr/bin/env python3
"""Replay sealed source tracks plus frozen r6 affinity through v15c-r3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from . import materialize_source_sam2_proposal_tracks_v15c_r6 as materializer
    from . import source_object_proposal_role_probe_v15c as core
except ImportError:  # pragma: no cover - flat AUH deployment
    import materialize_source_sam2_proposal_tracks_v15c_r6 as materializer
    import source_object_proposal_role_probe_v15c as core


class RunSourceProposalRoleProbeV15CError(RuntimeError):
    """The sealed r6/track/spec provenance closure differs."""


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RunSourceProposalRoleProbeV15CError("input must be one regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RunSourceProposalRoleProbeV15CError("input JSON differs") from error
    if type(value) is not dict:
        raise RunSourceProposalRoleProbeV15CError("input is not one JSON object")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    payload = dict(value)
    claimed = payload.pop(field, None)
    if (
        not isinstance(claimed, str)
        or materializer.SHA256_PATTERN.fullmatch(claimed) is None
        or claimed != core.object_sha256(payload)
    ):
        raise RunSourceProposalRoleProbeV15CError(f"{field} self-hash differs")


def require_sha256(value: Any, label: str) -> str:
    try:
        return materializer.require_sha256(value, label)
    except materializer.SourceSAM2ProposalTracksV15CError as error:
        raise RunSourceProposalRoleProbeV15CError(str(error)) from error


def require_exact_keys(value: Any, keys: tuple[str, ...], label: str) -> None:
    try:
        materializer.require_exact_keys(value, keys, label)
    except materializer.SourceSAM2ProposalTracksV15CError as error:
        raise RunSourceProposalRoleProbeV15CError(str(error)) from error


def thresholds_from_spec(spec: Mapping[str, Any]) -> core.ProbeThresholdsV15C:
    value = spec["role_assignment"]
    thresholds = core.ProbeThresholdsV15C(
        familywise_alpha=float(value["familywise_alpha"]),
        familywise_role_count=int(value["familywise_role_count"]),
        phase_proposal_max_null_percentile=float(
            value["phase_proposal_max_null_percentile"]
        ),
        minimum_consistent_phases=int(value["minimum_consistent_phases"]),
        minimum_longest_consistent_run=int(
            value["minimum_longest_consistent_run"]
        ),
        minimum_real_over_permutation_phases=int(
            value["minimum_real_over_permutation_phases"]
        ),
        minimum_proposal_dominance_phases=int(
            value["minimum_proposal_dominance_phases"]
        ),
        minimum_distinct_null_track_scores=int(
            value["minimum_distinct_null_track_scores"]
        ),
        null_track_score_epsilon=float(value["null_track_score_epsilon"]),
        duplicate_median_iou=float(value["duplicate_median_iou"]),
        duplicate_median_containment=float(value["duplicate_median_containment"]),
        family_overlap_median_iou=float(value["family_overlap_median_iou"]),
        family_nesting_median_containment=float(
            value["family_nesting_median_containment"]
        ),
        minimum_phase_coverage_mass=float(value["minimum_phase_coverage_mass"]),
    )
    if thresholds.required_track_null_percentile != float(
        value["required_track_null_percentile"]
    ):
        raise RunSourceProposalRoleProbeV15CError("FWER percentile differs")
    return thresholds


def validate_r6_receipt(
    receipt: Mapping[str, Any], spec: Mapping[str, Any], tensor_path: Path
) -> None:
    source = receipt.get("source_binding")
    diagnostics = receipt.get("diagnostics")
    gates = receipt.get("mechanical_gates")
    r6 = spec["r6"]
    if (
        receipt.get("schema_version")
        != "bernini-source-owned-instance-role-null64-sp4-probe-v15b-r6"
        or receipt.get("status")
        != "observer_only_sp4_null64_mechanical_pass_overlay_pending"
        or receipt.get("receipt_sha256") != r6["probe_receipt_internal_sha256"]
        or receipt.get("role_names") != list(core.FULL_R6_ROLE_NAMES)
        or receipt.get("selected_block_indices") != list(core.BLOCK_INDICES)
        or receipt.get("null_span_count") != core.NULL_COUNT
        or receipt.get("null_registry_sha256") != r6["null_registry_sha256"]
        or receipt.get("source_text_provenance_sha256")
        != r6["source_text_provenance_sha256"]
        or receipt.get("role_event_sha256") != r6["role_event_sha256"]
        or receipt.get("role_asset_sha256") != r6["role_asset_sha256"]
        or receipt.get("route_authorized") is not False
        or receipt.get("training_authorized") is not False
        or receipt.get("decode_authorized") is not False
        or receipt.get("localization_semantically_certified") is not False
        or not isinstance(gates, Mapping)
        or not gates
        or any(type(value) is not bool or value is not True for value in gates.values())
        or not isinstance(source, Mapping)
        or source.get("source_video_sha256") != spec["source"]["sha256"]
        or source.get("observer_only") is not True
        or source.get("route_authorized") is not False
        or source.get("training_authorized") is not False
        or source.get("anchor_consumed", False) is not False
        or not isinstance(diagnostics, Mapping)
        or diagnostics.get("file_sha256") != core.file_sha256(tensor_path)
        or diagnostics.get("null_span_count") != core.NULL_COUNT
        or diagnostics.get("role_names") != list(core.FULL_R6_ROLE_NAMES)
    ):
        raise RunSourceProposalRoleProbeV15CError("r6 observer receipt differs")


def _validate_manifest_file(
    *, root: Path, manifest_path: Path, schema: str, exclude_manifest: bool
) -> Mapping[str, Any]:
    manifest = read_json(manifest_path)
    require_exact_keys(
        manifest,
        (
            "schema_version",
            "files",
            "route_authorized",
            "training_authorized",
            "manifest_sha256",
        ),
        "manifest",
    )
    verify_self_hash(manifest, "manifest_sha256")
    require_sha256(manifest["manifest_sha256"], "manifest self hash")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != schema
        or type(files) is not dict
        or manifest.get("route_authorized") is not False
        or manifest.get("training_authorized") is not False
        or list(files) != sorted(files)
    ):
        raise RunSourceProposalRoleProbeV15CError("track manifest differs")
    expected_names = set(files)
    observed_names = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and (not exclude_manifest or path.resolve() != manifest_path.resolve())
    }
    if expected_names != observed_names:
        raise RunSourceProposalRoleProbeV15CError("track manifest file set differs")
    for relative, row in files.items():
        path = root / relative
        relative_path = Path(relative)
        if (
            type(row) is not dict
            or set(row) != {"sha256", "size"}
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not path.is_file()
            or path.is_symlink()
            or root.resolve() not in path.resolve().parents
            or require_sha256(row.get("sha256"), "manifest member hash")
            != row["sha256"]
            or core.file_sha256(path) != row["sha256"]
            or path.stat().st_size != row["size"]
        ):
            raise RunSourceProposalRoleProbeV15CError(
                "track manifest member differs"
            )
    return manifest


def validate_track_bundle(
    receipt: Mapping[str, Any],
    spec: Mapping[str, Any],
    receipt_path: Path,
    tensor_path: Path,
) -> None:
    import cv2
    import numpy as np
    from safetensors.numpy import load_file

    verify_self_hash(receipt, "receipt_sha256")
    root = receipt_path.parent
    output_manifest_path = root / "output_manifest.json"
    output_manifest = _validate_manifest_file(
        root=root,
        manifest_path=output_manifest_path,
        schema=materializer.OUTPUT_MANIFEST_SCHEMA,
        exclude_manifest=True,
    )
    artifact_manifest_path = root / "artifact_manifest.json"
    artifact_manifest = read_json(artifact_manifest_path)
    require_exact_keys(
        artifact_manifest,
        (
            "schema_version",
            "files",
            "route_authorized",
            "training_authorized",
            "manifest_sha256",
        ),
        "artifact manifest",
    )
    verify_self_hash(artifact_manifest, "manifest_sha256")
    require_sha256(artifact_manifest["manifest_sha256"], "artifact manifest hash")
    artifact_files = artifact_manifest.get("files")
    if (
        artifact_manifest.get("schema_version")
        != materializer.ARTIFACT_MANIFEST_SCHEMA
        or type(artifact_files) is not dict
        or list(artifact_files) != sorted(artifact_files)
        or artifact_manifest.get("route_authorized") is not False
        or artifact_manifest.get("training_authorized") is not False
        or core.file_sha256(artifact_manifest_path)
        != receipt.get("artifact_manifest_file_sha256")
        or artifact_manifest["manifest_sha256"]
        != receipt.get("artifact_manifest_internal_sha256")
        or output_manifest["files"].get("track_receipt.json", {}).get("sha256")
        != core.file_sha256(receipt_path)
    ):
        raise RunSourceProposalRoleProbeV15CError("artifact manifest binding differs")
    for relative, row in artifact_files.items():
        path = root / relative
        relative_path = Path(relative)
        if (
            type(row) is not dict
            or set(row) != {"sha256", "size"}
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not path.is_file()
            or path.is_symlink()
            or root.resolve() not in path.resolve().parents
            or require_sha256(row.get("sha256"), "artifact member hash")
            != row["sha256"]
            or core.file_sha256(path) != row["sha256"]
            or path.stat().st_size != row["size"]
        ):
            raise RunSourceProposalRoleProbeV15CError(
                "artifact manifest member differs"
            )

    source = receipt.get("source")
    runtime = receipt.get("runtime")
    sam2_receipt = receipt.get("sam2")
    claims = receipt.get("claim_limits")
    proposals = receipt.get("proposals")
    transcripts = receipt.get("repeat_transcripts")
    repeat = receipt.get("repeat")
    require_exact_keys(receipt, materializer.TRACK_RECEIPT_KEYS, "track receipt")
    require_exact_keys(repeat, materializer.REPEAT_RECEIPT_KEYS, "repeat receipt")
    require_exact_keys(repeat.get("gates"), materializer.REPEAT_GATE_KEYS, "repeat gates")
    if type(transcripts) is not dict or list(transcripts) != ["first", "second"]:
        raise RunSourceProposalRoleProbeV15CError("repeat transcript registry differs")
    first_transcript = transcripts["first"]
    second_transcript = transcripts["second"]
    masks_registry = first_transcript.get("mask_signatures")
    prompts_registry = first_transcript.get("prompt_signatures")
    require_exact_keys(receipt.get("spec"), ("raw_sha256", "canonical_sha256"), "spec binding")
    require_exact_keys(
        source,
        ("path", "sha256", "frame_count", "fps", "width", "height"),
        "source binding",
    )
    require_exact_keys(
        sam2_receipt,
        (
            "checkpoint_sha256",
            "actual_config_authority_sha256",
            "hydra",
            "automatic_generator",
            "proposal_admission",
            "tracking",
        ),
        "sam2 binding",
    )
    require_exact_keys(
        claims,
        (
            "observer_only",
            "source_video_only",
            "text_detector_consumed",
            "manual_box_consumed",
            "r6_affinity_consumed",
            "anchor_consumed",
            "target_instruction_consumed",
            "material_or_transparency_classification_consumed",
            "semantic_whole_object_certified",
            "renderer_forward_calls",
            "optimizer_updates",
            "training_authorized",
            "decode_authorized",
            "route_authorized",
            "localization_semantically_certified",
        ),
        "claim limits",
    )
    if (
        receipt.get("schema_version") != core.TRACK_SCHEMA_VERSION
        or receipt.get("spec")
        != {
            "raw_sha256": materializer.EXPECTED_SPEC_RAW_SHA256,
            "canonical_sha256": materializer.EXPECTED_SPEC_CANONICAL_SHA256,
        }
        or not isinstance(source, Mapping)
        or source.get("sha256") != spec["source"]["sha256"]
        or source.get("path") != spec["source"]["path"]
        or source.get("frame_count") != 81
        or source.get("fps") != 25.0
        or source.get("width") != 704
        or source.get("height") != 1056
        or receipt.get("phase_frames") != list(core.PHASE_FRAMES)
        or receipt.get("phase_grid") != [core.GRID_HEIGHT, core.GRID_WIDTH]
        or receipt.get("phase_coverage_tensor_sha256")
        != core.file_sha256(tensor_path)
        or not isinstance(proposals, list)
        or not 1 <= len(proposals) <= 64
        or receipt.get("proposal_count") != len(proposals)
        or not isinstance(masks_registry, list)
        or not isinstance(prompts_registry, list)
        or len(masks_registry) != len(proposals)
        or len(prompts_registry) != len(proposals)
        or receipt.get("sam2__C_imported_before_or_after") is not False
        or not isinstance(runtime, Mapping)
        or not isinstance(sam2_receipt, Mapping)
        or not isinstance(repeat, Mapping)
        or not isinstance(repeat.get("gates"), Mapping)
        or not repeat["gates"]
        or any(
            type(value) is not bool or value is not True
            for value in repeat["gates"].values()
        )
        or repeat.get("rng_state_before_sha256")
        != repeat.get("rng_state_after_sha256")
        or not isinstance(claims, Mapping)
        or claims.get("observer_only") is not True
        or claims.get("source_video_only") is not True
        or claims.get("text_detector_consumed") is not False
        or claims.get("manual_box_consumed") is not False
        or claims.get("r6_affinity_consumed") is not False
        or claims.get("anchor_consumed") is not False
        or claims.get("target_instruction_consumed") is not False
        or claims.get("material_or_transparency_classification_consumed") is not False
        or claims.get("semantic_whole_object_certified") is not False
        or claims.get("renderer_forward_calls") != 0
        or claims.get("optimizer_updates") != 0
        or claims.get("route_authorized") is not False
        or claims.get("training_authorized") is not False
        or claims.get("decode_authorized") is not False
    ):
        raise RunSourceProposalRoleProbeV15CError("SAM2 track receipt differs")
    for key in (
        "receipt_sha256",
        "phase_coverage_tensor_sha256",
        "phase_coverage_array_sha256",
        "artifact_manifest_file_sha256",
        "artifact_manifest_internal_sha256",
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
        require_sha256(repeat[key], f"repeat receipt {key}")
    try:
        materializer.validate_repeat_transcript(
            first_transcript, proposal_count=len(proposals), run_ordinal=1
        )
        materializer.validate_repeat_transcript(
            second_transcript, proposal_count=len(proposals), run_ordinal=2
        )
    except materializer.SourceSAM2ProposalTracksV15CError as error:
        raise RunSourceProposalRoleProbeV15CError(str(error)) from error

    runtime_spec = spec["sam2"]["runtime"]
    hydra_spec = spec["sam2"]["hydra"]
    checkpoint_spec = spec["sam2"]["checkpoint"]
    expected_runtime_fields = {
        "python_executable": runtime_spec["python_executable"],
        "python_executable_samefile_as_authority": True,
        "python_executable_sha256": runtime_spec["python_executable_sha256"],
        "python_version": runtime_spec["python_version"],
        "sam2_distribution": runtime_spec["sam2_distribution"],
        "sam2_package_root": runtime_spec["sam2_package_root"],
        "sam2_tree": {
            "file_count": runtime_spec["sam2_python_yaml_tree_file_count"],
            "tree_sha256": runtime_spec["sam2_python_yaml_tree_sha256"],
        },
        "key_module_sha256": runtime_spec["key_module_sha256"],
        "torch_version": runtime_spec["torch_version"],
        "torch_hip_version": runtime_spec["torch_hip_version"],
        "numpy_version": runtime_spec["numpy_version"],
        "opencv_version": runtime_spec["opencv_version"],
        "hydra_version": runtime_spec["hydra_version"],
        "omegaconf_version": runtime_spec["omegaconf_version"],
        "safetensors_version": runtime_spec["safetensors_version"],
        "visible_gpu_count": spec["execution"]["required_visible_gpu_count"],
        "visible_gpu_name": runtime_spec["required_device_name"],
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
    if runtime != expected_runtime_fields or sam2_receipt != {
        "checkpoint_sha256": checkpoint_spec["sha256"],
        "actual_config_authority_sha256": hydra_spec[
            "actual_config_authority_sha256"
        ],
        "hydra": hydra_spec,
        "automatic_generator": spec["sam2"]["automatic_generator"],
        "proposal_admission": spec["sam2"]["proposal_admission"],
        "tracking": spec["sam2"]["tracking"],
    }:
        raise RunSourceProposalRoleProbeV15CError("SAM2 runtime/config evidence differs")

    tensors = load_file(str(tensor_path))
    if set(tensors) != {"phase_coverage"}:
        raise RunSourceProposalRoleProbeV15CError("coverage tensor registry differs")
    stored_coverage = tensors["phase_coverage"]
    if (
        stored_coverage.dtype != np.float32
        or stored_coverage.shape
        != (len(proposals), len(core.PHASE_FRAMES), core.GRID_HEIGHT, core.GRID_WIDTH)
        or not bool(np.isfinite(stored_coverage).all())
        or materializer.array_sha256(stored_coverage)
        != receipt.get("phase_coverage_array_sha256")
    ):
        raise RunSourceProposalRoleProbeV15CError("coverage tensor differs")

    ids = [row.get("proposal_id") for row in proposals]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise RunSourceProposalRoleProbeV15CError("proposal order differs")
    prompt_by_id = {row.get("proposal_id"): row for row in prompts_registry}
    mask_by_id = {row.get("proposal_id"): row for row in masks_registry}
    if set(prompt_by_id) != set(ids) or set(mask_by_id) != set(ids):
        raise RunSourceProposalRoleProbeV15CError("mask/prompt registry differs")
    expected_artifact_names = {"phase_coverage.safetensors"}
    expected_artifact_names.update(f"prompts/{proposal_id}.png" for proposal_id in ids)
    expected_artifact_names.update(
        f"masks/{proposal_id}/{frame_index:05d}.png"
        for proposal_id in ids
        for frame_index in range(81)
    )
    if set(artifact_files) != expected_artifact_names:
        raise RunSourceProposalRoleProbeV15CError("artifact manifest membership differs")
    recomputed_coverage = np.zeros_like(stored_coverage)
    recomputed_prompt_signatures = []
    recomputed_mask_signatures = []
    geometry_keys = {
        "frame_geometry",
        "seed_prompt_iou",
        "area_p95_to_p05_ratio",
        "median_adjacent_iou",
        "p95_centroid_step_frame_diagonal_fraction",
        "p10_area_pixels",
        "median_largest_component_fraction",
        "median_bbox_fill_fraction",
        "p10_bbox_diagonal_frame_fraction",
        "whole_object_observer_scope",
        "automatic_track_geometry_gates",
        "automatic_track_geometry_gate_pass",
    }
    proposal_keys = {
        "proposal_id",
        "prompt_mask_sha256",
        "prompt_relative_path",
        "area",
        "bbox_xywh",
        "predicted_iou",
        "stability_score",
        *geometry_keys,
    }
    frame_geometry_keys = {
        "visible",
        "area",
        "centroid_xy",
        "bbox_xyxy",
        "bbox_fill_fraction",
        "bbox_diagonal_frame_fraction",
        "largest_component_fraction",
    }
    geometry_gate_keys = {
        "all_81_frames_visible",
        "seed_prompt_iou",
        "area_p95_to_p05_ratio",
        "median_adjacent_iou",
        "p95_centroid_step",
        "whole_object_area_extent",
        "whole_object_component_integrity",
        "whole_object_bbox_support",
    }
    for proposal_index, row in enumerate(proposals):
        proposal_id = ids[proposal_index]
        if (
            not isinstance(proposal_id, str)
            or core.PROPOSAL_ID_PATTERN.fullmatch(proposal_id) is None
            or set(row) != proposal_keys
            or require_sha256(row.get("prompt_mask_sha256"), "prompt mask hash")
            != row["prompt_mask_sha256"]
            or row.get("prompt_relative_path") != f"prompts/{proposal_id}.png"
            or type(row.get("automatic_track_geometry_gate_pass")) is not bool
            or not isinstance(row.get("automatic_track_geometry_gates"), Mapping)
            or set(row["automatic_track_geometry_gates"]) != geometry_gate_keys
            or any(
                type(value) is not bool
                for value in row["automatic_track_geometry_gates"].values()
            )
            or type(row.get("frame_geometry")) is not list
            or len(row["frame_geometry"]) != 81
            or any(
                type(frame_row) is not dict
                or set(frame_row) != frame_geometry_keys
                for frame_row in row["frame_geometry"]
            )
        ):
            raise RunSourceProposalRoleProbeV15CError("proposal strict gate differs")
        prompt_path = root / row["prompt_relative_path"]
        prompt_u8 = cv2.imread(str(prompt_path), cv2.IMREAD_GRAYSCALE)
        if (
            not isinstance(prompt_u8, np.ndarray)
            or prompt_u8.shape != (1056, 704)
            or not set(np.unique(prompt_u8)).issubset({0, 255})
        ):
            raise RunSourceProposalRoleProbeV15CError("prompt PNG differs")
        prompt = prompt_u8 > 0
        prompt_digest = materializer.array_sha256(prompt.astype(np.uint8))
        if (
            prompt_digest != row["prompt_mask_sha256"]
            or proposal_id != f"sam2-f000-{prompt_digest}"
            or prompt_by_id[proposal_id].get("array_sha256") != prompt_digest
            or prompt_by_id[proposal_id].get("relative_path")
            != row["prompt_relative_path"]
            or set(prompt_by_id[proposal_id])
            != {"proposal_id", "array_sha256", "relative_path"}
        ):
            raise RunSourceProposalRoleProbeV15CError("full-SHA prompt ID differs")
        recomputed_prompt_signatures.append(
            {
                "proposal_id": proposal_id,
                "array_sha256": prompt_digest,
                "relative_path": row["prompt_relative_path"],
            }
        )
        mask_row = mask_by_id[proposal_id]
        digests = mask_row.get("mask_array_sha256_by_frame")
        if mask_row.get("frame_count") != 81 or not isinstance(digests, list) or len(digests) != 81:
            raise RunSourceProposalRoleProbeV15CError("81-mask registry differs")
        if set(mask_row) != {
            "proposal_id",
            "frame_count",
            "mask_array_sha256_by_frame",
        }:
            raise RunSourceProposalRoleProbeV15CError("mask signature fields differ")
        masks = []
        for frame_index, expected_digest in enumerate(digests):
            mask_path = root / "masks" / proposal_id / f"{frame_index:05d}.png"
            mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if (
                not isinstance(mask_u8, np.ndarray)
                or mask_u8.shape != (1056, 704)
                or not set(np.unique(mask_u8)).issubset({0, 255})
                or materializer.array_sha256((mask_u8 > 0).astype(np.uint8))
                != expected_digest
            ):
                raise RunSourceProposalRoleProbeV15CError("published mask PNG differs")
            masks.append(mask_u8 > 0)
        recomputed_mask_signatures.append(
            {
                "proposal_id": proposal_id,
                "frame_count": 81,
                "mask_array_sha256_by_frame": list(digests),
            }
        )
        geometry = materializer.track_geometry_receipt(
            masks,
            prompt,
            width=704,
            height=1056,
            tracking_spec=spec["sam2"]["tracking"],
        )
        if any(row[key] != geometry[key] for key in geometry_keys):
            raise RunSourceProposalRoleProbeV15CError("track geometry replay differs")
        for phase, frame_index in enumerate(core.PHASE_FRAMES):
            recomputed_coverage[proposal_index, phase] = cv2.resize(
                masks[frame_index].astype(np.float32),
                (core.GRID_WIDTH, core.GRID_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
    if not np.array_equal(recomputed_coverage, stored_coverage):
        raise RunSourceProposalRoleProbeV15CError("phase coverage replay differs")

    # Rebuild both full transcripts from the independently reopened prompt,
    # every P x 81 mask, recomputed coverage, and each run's batch/freeze/logit
    # evidence.  The second transcript is not inferred from the first.
    rebuilt = []
    for ordinal, published in ((1, first_transcript), (2, second_transcript)):
        candidate = materializer.build_repeat_transcript(
            run_ordinal=ordinal,
            proposal_rows=proposals,
            prompt_signatures=recomputed_prompt_signatures,
            mask_signatures=recomputed_mask_signatures,
            phase_coverage=recomputed_coverage,
            tracking_batches=published["tracking_batches"],
            freeze_receipts=published["freeze_receipts"],
            sam2__C_imported=published["sam2__C_imported"],
        )
        if candidate != published:
            raise RunSourceProposalRoleProbeV15CError(
                f"repeat transcript {ordinal} disk replay differs"
            )
        rebuilt.append(candidate)
    first_equivalence = materializer.transcript_equivalence_sha256(rebuilt[0])
    second_equivalence = materializer.transcript_equivalence_sha256(rebuilt[1])
    expected_repeat_gates = {
        "first_transcript_self_hash": True,
        "second_transcript_self_hash": True,
        "equivalence_signature_equal": first_equivalence == second_equivalence,
        "proposal_rows_equal": (
            rebuilt[0]["proposal_rows"] == rebuilt[1]["proposal_rows"]
        ),
        "prompt_signatures_equal": (
            rebuilt[0]["prompt_signatures"] == rebuilt[1]["prompt_signatures"]
        ),
        "mask_signatures_equal": (
            rebuilt[0]["mask_signatures"] == rebuilt[1]["mask_signatures"]
        ),
        "phase_coverage_equal": (
            rebuilt[0]["phase_coverage"] == rebuilt[1]["phase_coverage"]
        ),
        "tracking_batches_equal": (
            rebuilt[0]["tracking_batches"] == rebuilt[1]["tracking_batches"]
        ),
        "freeze_receipts_equal": (
            rebuilt[0]["freeze_receipts"] == rebuilt[1]["freeze_receipts"]
        ),
        "rng_state_unchanged_outside_fork_rng": (
            repeat["rng_state_before_sha256"] == repeat["rng_state_after_sha256"]
        ),
        "sam2__C_absent_both_runs": (
            rebuilt[0]["sam2__C_imported"] is False
            and rebuilt[1]["sam2__C_imported"] is False
        ),
    }
    if (
        repeat["first_transcript_sha256"] != rebuilt[0]["transcript_sha256"]
        or repeat["second_transcript_sha256"] != rebuilt[1]["transcript_sha256"]
        or repeat["first_equivalence_sha256"] != first_equivalence
        or repeat["second_equivalence_sha256"] != second_equivalence
        or repeat["gates"] != expected_repeat_gates
        or not expected_repeat_gates
        or any(type(value) is not bool or value is not True for value in expected_repeat_gates.values())
    ):
        raise RunSourceProposalRoleProbeV15CError("exact repeat replay gates differ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--r6-receipt", required=True, type=Path)
    parser.add_argument("--r6-tensors", required=True, type=Path)
    parser.add_argument("--track-receipt", required=True, type=Path)
    parser.add_argument("--track-tensors", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    output = args.output_json.absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise RunSourceProposalRoleProbeV15CError("output path is not fresh")
    spec_path = args.spec.resolve(strict=True)
    spec = materializer.read_spec(spec_path)
    r6_receipt_path = args.r6_receipt.resolve(strict=True)
    r6_tensor_path = args.r6_tensors.resolve(strict=True)
    track_receipt_path = args.track_receipt.resolve(strict=True)
    track_tensor_path = args.track_tensors.resolve(strict=True)
    if (
        core.file_sha256(r6_receipt_path)
        != spec["r6"]["probe_receipt_file_sha256"]
        or core.file_sha256(r6_tensor_path)
        != spec["r6"]["affinity_tensor_file_sha256"]
    ):
        raise RunSourceProposalRoleProbeV15CError("sealed r6 files differ")
    r6_receipt = read_json(r6_receipt_path)
    track_receipt = read_json(track_receipt_path)
    validate_r6_receipt(r6_receipt, spec, r6_tensor_path)
    validate_track_bundle(
        track_receipt, spec, track_receipt_path, track_tensor_path
    )
    tracks, loaded_track_receipt = core.load_tracks_for_v15c(
        track_receipt_path, track_tensor_path
    )
    if loaded_track_receipt != track_receipt:
        raise RunSourceProposalRoleProbeV15CError("track receipt replay differs")
    affinity = core.load_r6_affinity_for_v15c(r6_tensor_path)
    result = dict(
        core.run_source_object_proposal_role_probe_v15c(
            tracks=tracks,
            affinity=affinity,
            thresholds=thresholds_from_spec(spec),
        )
    )
    core_receipt_sha256 = result.pop("receipt_sha256")
    result["provenance"] = {
        "spec_raw_sha256": core.file_sha256(spec_path),
        "spec_canonical_sha256": core.object_sha256(spec),
        "source_video_sha256": spec["source"]["sha256"],
        "source_text_provenance_sha256": spec["r6"][
            "source_text_provenance_sha256"
        ],
        "r6_receipt_file_sha256": core.file_sha256(r6_receipt_path),
        "r6_affinity_file_sha256": core.file_sha256(r6_tensor_path),
        "r6_internal_receipt_sha256": r6_receipt["receipt_sha256"],
        "track_receipt_file_sha256": core.file_sha256(track_receipt_path),
        "track_tensor_file_sha256": core.file_sha256(track_tensor_path),
        "track_internal_receipt_sha256": track_receipt["receipt_sha256"],
        "track_output_manifest_file_sha256": core.file_sha256(
            track_receipt_path.parent / "output_manifest.json"
        ),
        "assignment_core_receipt_sha256": core_receipt_sha256,
    }
    result["route_authorized"] = False
    result["training_authorized"] = False
    result["decode_authorized"] = False
    result["receipt_sha256"] = core.object_sha256(result)
    output.write_bytes(core.canonical_bytes(result))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

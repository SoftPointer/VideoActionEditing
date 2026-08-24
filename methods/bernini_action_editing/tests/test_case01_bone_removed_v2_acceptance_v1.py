#!/usr/bin/env python3
"""Hostile contract tests for the fresh case01 bone-removed-v2 gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import unittest
from unittest import mock
import zlib


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
PROGRAM = ROOT / "tools" / "case01_bone_removed_v2_acceptance_v1.py"
SUPPORT_PACKET_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "case01_bone_contact_support_review_v1_unsigned_20260822"
)
SPEC = importlib.util.spec_from_file_location("case01_bone_removed_v2_acceptance_v1", PROGRAM)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def file_row(tag: str, size: int = 100) -> dict:
    digit = format((sum(tag.encode("utf-8")) % 15) + 1, "x")
    return {"path": f"/authority/{tag}", "sha256": digit * 64, "size": size}


def actual_file_row(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def write_canonical_json(path: Path, value: dict) -> dict:
    path.write_bytes(GATE.canonical_json_bytes(value) + b"\n")
    return actual_file_row(path)


def refresh_digest(value: dict, field: str) -> None:
    value.pop(field, None)
    value[field] = GATE.object_sha256(value)


def external_support_review(
    root: Path,
    *,
    reviewer_slot: int,
    reviewer_identity: str,
    frame_update: tuple[int, dict] | None = None,
) -> dict:
    review_root = root / f"external-review-{reviewer_slot}"
    review_root.mkdir()
    evidence_path = review_root / "opaque-evidence.json"
    evidence = {
        "kind": "opaque_external_trust_root_evidence_fixture",
        "reviewer_slot": reviewer_slot,
        "opaque_reference": f"external-review-evidence-{reviewer_slot}",
    }
    evidence_row = write_canonical_json(evidence_path, evidence)
    independence = {
        name: True for name in GATE.EXTERNAL_INDEPENDENCE_KEYS
    }
    frames = [
        {
            "frame_index": frame_index,
            "bone_coverage": "PASS",
            "contact_shadow_coverage": "PASS",
            "halo_and_adjacent_ground_coverage": "PASS",
            "dog_and_guard_protection": "PASS",
            "boundary_edit_requested": False,
            "notes": (
                f"Reviewer {reviewer_slot} inspected native frame {frame_index}."
            ),
            "decision": "PASS",
        }
        for frame_index in range(GATE.FRAME_COUNT)
    ]
    if frame_update is not None:
        frame_index, update = frame_update
        frames[frame_index].update(update)
    reviewed_at_utc = f"2026-08-22T20:00:0{reviewer_slot}Z"
    receipt = {
        "schema_version": GATE.EXTERNAL_REVIEW_SCHEMA,
        "reviewer_slot": reviewer_slot,
        "reviewer_identity": reviewer_identity,
        "reviewer_affiliation_or_role": f"external-role-{reviewer_slot}",
        "candidate_manifest_sha256": GATE.SUPPORT_PACKET_MANIFEST_SHA256,
        "reviewed_at_utc": reviewed_at_utc,
        "independence_attestation": independence,
        "all_81_native_frames_reviewed": True,
        "instructions": list(GATE.EXTERNAL_REVIEW_INSTRUCTIONS),
        "frames": frames,
        "overall_decision": "PASS",
        "signature_or_external_receipt": None,
        "claim_limits_acknowledged": True,
    }
    signature = {
        "kind": GATE.EXTERNAL_SIGNATURE_KIND,
        "review_projection_sha256": GATE.object_sha256(receipt),
        "evidence_sha256": evidence_row["sha256"],
        "evidence_size": evidence_row["size"],
    }
    receipt["signature_or_external_receipt"] = signature
    receipt_row = write_canonical_json(review_root / "receipt.json", receipt)
    return {
        "reviewer_slot": reviewer_slot,
        "receipt": receipt_row,
        "reviewer_identity": reviewer_identity,
        "reviewer_affiliation_or_role": receipt["reviewer_affiliation_or_role"],
        "reviewed_at_utc": reviewed_at_utc,
        "independence_attestation": independence,
        "signature": signature,
        "evidence": evidence_row,
    }


def dual_support_review_fixture(
    root: Path,
    *,
    reviewer_identities: tuple[str, str] = (
        "external-support-reviewer-one",
        "external-support-reviewer-two",
    ),
    frame_updates: dict[int, tuple[int, dict]] | None = None,
) -> tuple[dict, dict, Path]:
    packet_manifest_path = SUPPORT_PACKET_ROOT / "manifest.json"
    packet_sums_path = SUPPORT_PACKET_ROOT / "SHA256SUMS"
    packet_manifest = json.loads(packet_manifest_path.read_text(encoding="utf-8"))
    promotion_root = root / "promotion-bundle"
    formal_packet_root = promotion_root / "formal-support-review"
    formal_packet_root.mkdir(parents=True)
    packet_manifest_copy = formal_packet_root / "candidate-manifest.json"
    packet_sums_copy = formal_packet_root / "candidate-SHA256SUMS"
    shutil.copyfile(packet_manifest_path, packet_manifest_copy)
    shutil.copyfile(packet_sums_path, packet_sums_copy)
    promoted_mask_root = promotion_root / "candidate_support"
    promoted_mask_root.mkdir(parents=True)
    producer_rows = []
    formal_rows = []
    for frame_index, frame in enumerate(packet_manifest["frames"]):
        packet_row = frame["outputs"]["candidate_support"]
        source_path = SUPPORT_PACKET_ROOT / packet_row["path"]
        promoted_path = promoted_mask_root / f"{frame_index:05d}.png"
        shutil.copyfile(source_path, promoted_path)
        row = {"frame_index": frame_index, **actual_file_row(promoted_path)}
        if (row["sha256"], row["size"]) != (
            packet_row["sha256"],
            packet_row["size"],
        ):
            raise AssertionError(f"fixture support row differs: {frame_index}")
        producer_rows.append(row)
        formal_rows.append(
            {
                **row,
                "bone_and_cast_shadow_covered": True,
                "native_resolution_reviewed": True,
            }
        )
    producer = producer_receipt()
    producer["support"]["frame_masks"] = producer_rows
    updates = frame_updates or {}
    external_reviews = [
        external_support_review(
            root,
            reviewer_slot=slot,
            reviewer_identity=reviewer_identities[slot - 1],
            frame_update=updates.get(slot),
        )
        for slot in (1, 2)
    ]
    review = {
        "schema_version": GATE.SUPPORT_REVIEW_SCHEMA,
        "status": GATE.SUPPORT_REVIEW_STATUS,
        "case_id": GATE.CASE_ID,
        "iid": GATE.IID,
        "candidate_packet": {
            "manifest": actual_file_row(packet_manifest_copy),
            "sha256sums": actual_file_row(packet_sums_copy),
            "premanifest_output_tree_digest": (
                GATE.SUPPORT_PACKET_PREMANIFEST_DIGEST
            ),
        },
        "source": copy.deepcopy(producer["source"]),
        "sam2_receipt": copy.deepcopy(producer["mask_authority"]["receipt"]),
        "external_reviews": external_reviews,
        "protocol": {
            "native_resolution_704x736": True,
            "all_81_frames_reviewed_by_each": True,
            "required_external_reviewers": 2,
            "bone_covered_all_frames_by_each": True,
            "cast_shadow_and_halo_covered_all_frames_by_each": True,
            "minimum_bone_dilation_pixels": GATE.MIN_SUPPORT_DILATION,
            "old_dilate3_reused": False,
            "dog_guard_excluded_all_frames": True,
        },
        "frame_masks": formal_rows,
        "claim_limits": copy.deepcopy(GATE.SUPPORT_REVIEW_CLAIM_LIMITS),
    }
    refresh_digest(review, "review_digest")
    return producer, review, promoted_mask_root


def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", crc)
    )


def grayscale_png() -> bytes:
    ihdr = struct.pack(
        ">IIBBBBB",
        GATE.WIDTH,
        GATE.HEIGHT,
        8,
        0,
        0,
        0,
        0,
    )
    scanlines = (b"\x00" + bytes(GATE.WIDTH)) * GATE.HEIGHT
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(scanlines))
        + png_chunk(b"IEND", b"")
    )


def authority_tree_manifest(role: str, tree_root: Path) -> dict:
    entries = []
    for path in sorted(row for row in tree_root.rglob("*") if row.is_file()):
        file_authority = actual_file_row(path)
        entries.append(
            {
                "relative_path": path.relative_to(tree_root).as_posix(),
                "sha256": file_authority["sha256"],
                "size": file_authority["size"],
            }
        )
    value = {
        "schema_version": GATE.TREE_MANIFEST_SCHEMA,
        "authority_role": role,
        "inventory_policy": "exact_recursive_regular_nonsymlink_nlink1",
        "tree_root": str(tree_root),
        "entries": entries,
        "file_count": len(entries),
        "total_bytes": sum(row["size"] for row in entries),
        "tree_digest": GATE.object_sha256(entries),
    }
    value["manifest_digest"] = GATE.object_sha256(value)
    return value


def producer_receipt() -> dict:
    model_authorities = [
        {"role": role, **file_row(f"{role}-manifest.json")}
        for role in GATE.MODEL_AUTHORITY_ROLES
    ]
    value = {
        "schema_version": GATE.RECEIPT_SCHEMA,
        "status": "COMPLETE_CANDIDATE_PENDING_INDEPENDENT_ACCEPTANCE",
        "case_id": GATE.CASE_ID,
        "iid": GATE.IID,
        "role": GATE.ROLE,
        "source": {
            "path": "/authority/source.mp4",
            "sha256": GATE.SOURCE_SHA256,
            "size": GATE.SOURCE_SIZE,
        },
        "mask_authority": {
            "receipt": {
                "path": "/authority/masks/receipt.json",
                "sha256": GATE.SAM2_RECEIPT_SHA256,
                "size": GATE.SAM2_RECEIPT_SIZE,
            },
            "bone_mask_count": 81,
            "dog_mask_count": 81,
            "all_81_masks_hash_bound": True,
        },
        "media_tools": {
            "ffmpeg": {
                "path": GATE.FFMPEG_PATH,
                "sha256": GATE.FFMPEG_SHA256,
                "size": GATE.FFMPEG_SIZE,
            },
            "ffprobe": {
                "path": GATE.FFPROBE_PATH,
                "sha256": GATE.FFPROBE_SHA256,
                "size": GATE.FFPROBE_SIZE,
            },
        },
        "acceptance_contract": file_row("case01_bone_removed_v2_acceptance_v1.py"),
        "generator": {
            "program": {
                "path": GATE.GENERATOR_PATH,
                "sha256": GATE.GENERATOR_SHA256,
                "size": GATE.GENERATOR_SIZE,
            },
            "model_authorities": model_authorities,
            "authority_replay": {
                "before_generation_digest": "b" * 64,
                "after_generation_digest": "b" * 64,
                "unchanged": True,
            },
            "raw_support_donor": {
                "video": file_row("raw-vace-support-donor.mp4"),
                "frame_count": 81,
                "index_mapping": "exact_frame_index_0_through_80_ignore_container_fps_timestamps",
                "normalization": {
                    "algorithm": "vace_precanvas_fitpad624x640_inverse_crop_lanczos_v1",
                    "source_width": 704,
                    "source_height": 736,
                    "precanvas_width": 624,
                    "precanvas_height": 640,
                    "fit_width": 612,
                    "fit_height": 640,
                    "pad_left": 6,
                    "pad_right": 6,
                    "pad_top": 0,
                    "pad_bottom": 0,
                    "source_fit_kernel": "lanczos",
                    "support_fit_kernel": "nearest",
                    "pad_value": 0,
                    "inverse_crop_xyxy": [6, 0, 618, 640],
                    "inverse_resize_kernel": "lanczos",
                    "python_hash_seed": 20260822,
                    "frame_indices": list(range(81)),
                    "prepare_source_trace": {
                        "frame_indices": list(range(81)),
                        "resize_crop_applied": False,
                        "digest_definition": (
                            "sha256(torch.float32 contiguous CPU little-endian "
                            "C-order bytes)"
                        ),
                        "source_tensor": {
                            "shape": [3, 81, 640, 624],
                            "dtype": "float32",
                            "pre_generate_sha256": "c" * 64,
                            "post_generate_sha256": "c" * 64,
                            "unchanged": True,
                        },
                        "mask_tensor": {
                            "shape": [1, 81, 640, 624],
                            "dtype": "float32",
                            "pre_generate_sha256": "d" * 64,
                            "post_generate_sha256": "d" * 64,
                            "unchanged": True,
                        },
                    },
                    "precanvas_authority_scope": "lossless_vace_input_authority",
                    "processed_cache_authority_scope": (
                        "nonauthoritative_codec_diagnostic_only"
                    ),
                    "precanvas_source_video": file_row("precanvas-source.mkv"),
                    "precanvas_mask_video": file_row("precanvas-mask.mkv"),
                    "processed_source_video": file_row("vace-src-video.mp4"),
                    "processed_mask_video": file_row("vace-src-mask.mp4"),
                },
                "used_only_inside_support": True,
                "source_or_identity_authority": False,
            },
            "algorithm_id": "masked_temporally_coherent_background_inpaint_v2",
            "deterministic": True,
            "seed": 20260822,
            "generative_inpainting_inside_support": True,
            "whole_frame_generation": False,
            "outside_support_hard_composite_source_rgb": True,
            "uses_bidirectional_boundary_interpolation": False,
            "uses_ffmpeg_removelogo": False,
            "training_performed": False,
            "optimizer_updates": 0,
        },
        "support": {
            "tube": file_row("bone_plus_shadow_v2.mkv"),
            "definition": "per_frame_bone_plus_cast_shadow_support_v2",
            "frame_count": 81,
            "contains_bone_and_cast_shadow_all_frames": True,
            "all_81_frames_manually_reviewed": True,
            "old_dilate3_tube_reused": False,
            "review_receipt": file_row("support-review.json"),
            "frame_masks": [
                {
                    "frame_index": frame_index,
                    **file_row(f"support-mask-{frame_index:05d}.png"),
                }
                for frame_index in range(81)
            ],
        },
        "canonical_candidate": {
            "video": file_row("bone_removed_v2_canonical.mkv"),
            "codec": "ffv1",
            "lossless": True,
            "stored_pixel_format": "bgr0",
            "decoded_pixel_format": "rgb24",
            "frame_count": 81,
        },
        "delivery_candidate": {
            "video": file_row("bone_removed_v2.mp4"),
            "codec": "h264",
            "pixel_format": "yuv420p",
            "frame_count": 81,
            "derived_only_from_canonical": True,
            "authority_scope": "human_playback_convenience_lossy_transport_only",
            "identity_authority": False,
            "canonical_is_identity_authority": True,
        },
        "construction_audit": {
            "frame_count": 81,
            "outside_support_changed_pixels": 0,
            "dog_guard_changed_pixels": 0,
            "support_pixels_not_equal_raw_donor": 0,
            "full_frame_pixel_scan": True,
            "source_bone_changed_fraction_minimum": 1.0,
        },
        "create_only_authority": {
            "controller_program": file_row("create-only-controller.py"),
            "attempt_receipt": file_row("attempt.json"),
            "publication_receipt": file_row("publication.json"),
            "controller_distinct_from_generator": True,
            "fresh_root": True,
            "existing_path_reused": False,
            "overwrite_performed": False,
            "atomic_publish": True,
            "staging_removed_after_publish": True,
        },
        "claim_limits": {
            "input_asset_authority_only": True,
            "renderer_inference_performed": False,
            "renderer_result_claim_authorized": False,
            "scientific_claim_authorized": False,
            "semantic_absence_requires_human_review": True,
            "downstream_identity_sensitive_consumption_requires_canonical": True,
            "actual_downstream_consumer_verified": False,
            "generation_execution_lineage_verified": False,
        },
    }
    value["receipt_digest"] = GATE.object_sha256(value)
    return value


def reviewer(reviewer_id: str) -> dict:
    result = {
        "reviewer_id": reviewer_id,
        "all_81_frames_reviewed": True,
    }
    for gate_name in (
        "support_bone_shadow_coverage",
        "bone_absence",
        "bone_shaped_scar_absence",
        "seam_absence",
        "texture_collapse_absence",
        "temporal_flicker_absence",
        "cast_shadow_absence",
        "dog_identity_preservation",
        "background_identity_preservation",
    ):
        result[gate_name] = {
            "status": "PASS",
            "failure_frames": [],
            "note": f"All 81 frames pass {gate_name}.",
        }
    return result


def observations(candidate_sha256: str, support_sha256: str) -> dict:
    value = {
        "schema_version": GATE.OBSERVATION_SCHEMA,
        "case_id": GATE.CASE_ID,
        "iid": GATE.IID,
        "candidate_sha256": candidate_sha256,
        "support_sha256": support_sha256,
        "blinding": {
            "candidate_id_randomized": True,
            "arm_name_hidden": True,
            "reviewers_independent": True,
        },
        "review_protocol": {
            "evidence_source": "direct_hash_bound_candidate_decode_not_pre_rendered_surfaces",
            "canonical_candidate_sha256": candidate_sha256,
            "delivery_candidate_sha256": file_row("bone_removed_v2.mp4")["sha256"],
            "support_sha256": support_sha256,
            "source_sha256": GATE.SOURCE_SHA256,
            "decoded_frame_indices": list(range(81)),
            "direct_canonical_decode_reviewed": True,
            "direct_delivery_playback_reviewed": True,
            "native_resolution_support_crop_reviewed_all_frames": True,
            "mask_outline_hidden_during_scar_ballot": True,
            "convenience_surfaces_used_as_authority": False,
        },
        "reviewers": [reviewer("reviewer-a"), reviewer("reviewer-b")],
        "claim_limits": {
            "input_asset_review_only": True,
            "renderer_result_reviewed": False,
            "scientific_claim_authorized": False,
        },
    }
    value["observation_digest"] = GATE.object_sha256(value)
    return value


def passing_metrics() -> dict:
    return {
        "frame_count": 81,
        "outside_support_changed_pixels": 0,
        "dog_guard_changed_pixels": 0,
        "support_pixels_not_equal_raw_donor": 0,
        "precanvas_source_mismatch_pixels": 0,
        "precanvas_mask_mismatch_pixels": 0,
        "precanvas_support_pad_active_pixels": 0,
        "processed_source_rgb_mad_mean": 2.2,
        "processed_source_rgb_mad_frame_maximum": 3.7,
        "processed_mask_threshold_mismatch_pixels": 0,
        "bone_changed_fraction_minimum": 1.0,
        "bone_source_residual_p10": 32.0,
        "texture_ratio_p10": 0.72,
        "texture_ratio_median": 0.96,
        "texture_ratio_maximum": 1.31,
        "low_texture_frame_count": 0,
        "seam_ratio_median": 1.04,
        "seam_ratio_maximum": 1.55,
        "delivery_rgb_mad_mean": 2.4,
        "delivery_rgb_mad_frame_maximum": 3.2,
        "delivery_outside_support_rgb_mad_mean": 2.3,
        "delivery_outside_support_rgb_mad_frame_maximum": 3.3,
        "delivery_dog_guard_rgb_mad_mean": 2.1,
        "delivery_dog_guard_rgb_mad_frame_maximum": 3.5,
    }


def create_only_receipts(value: dict) -> tuple[dict, dict]:
    final_root = "/fresh/case01-bone-removed-v2-r1"
    staging_root = "/fresh/.case01-bone-removed-v2-r1.staging"
    for key, filename in (
        ("support", "support.mkv"),
        ("canonical_candidate", "canonical.mkv"),
        ("delivery_candidate", "delivery.mp4"),
    ):
        if key == "support":
            row = value["support"]["tube"]
        else:
            row = value[key]["video"]
        row["path"] = f"{final_root}/{filename}"
    token = "7" * 64
    attempt = {
        "schema_version": GATE.ATTEMPT_SCHEMA,
        "status": "RESERVED_FRESH_BEFORE_GENERATION",
        "case_id": GATE.CASE_ID,
        "iid": GATE.IID,
        "attempt_token": token,
        "controller_program_sha256": value["create_only_authority"]["controller_program"]["sha256"],
        "generator_program_sha256": value["generator"]["program"]["sha256"],
        "model_authorities_digest": GATE.object_sha256(
            value["generator"]["model_authorities"]
        ),
        "final_root": final_root,
        "staging_root": staging_root,
        "preflight": {
            "performed_before_generation": True,
            "final_root_absent": True,
            "staging_root_absent": True,
            "all_target_paths_absent": True,
            "reservation_create_only": True,
        },
    }
    attempt["attempt_digest"] = GATE.object_sha256(attempt)
    publication = {
        "schema_version": GATE.PUBLICATION_SCHEMA,
        "status": "PUBLISHED_FRESH_NO_REPLACE",
        "case_id": GATE.CASE_ID,
        "iid": GATE.IID,
        "attempt_token": token,
        "controller_program_sha256": value["create_only_authority"]["controller_program"]["sha256"],
        "final_root": final_root,
        "staging_root": staging_root,
        "published_assets": {
            "support": copy.deepcopy(value["support"]["tube"]),
            "canonical_candidate": copy.deepcopy(value["canonical_candidate"]["video"]),
            "delivery_candidate": copy.deepcopy(value["delivery_candidate"]["video"]),
        },
        "publication": {
            "atomic_rename_noreplace": True,
            "final_root_absent_before_publish": True,
            "overwrite_performed": False,
            "staging_removed_after_publish": True,
            "published_tree_regular_nonsymlink_nlink1": True,
            "directory_fsync_performed": True,
        },
    }
    publication["publication_digest"] = GATE.object_sha256(publication)
    return attempt, publication


class PreparedTensorReplayTests(unittest.TestCase):
    # Torch 2.2.1 oracle fixture: uint8 bytes(range(36)) reshaped THWC as
    # [T=2,H=2,W=3,C=3], then the exact VACE operations.  These hashes are
    # from the complete contiguous tensors, not a per-value sample.
    TORCH_SOURCE_SHA256 = (
        "9e04c5d06bb06ad002dada6760dea75d4095a4bcc209e2482c6096a02bff4830"
    )
    TORCH_MASK_SHA256 = (
        "d3b0f54f432b6770d1b01142b1523a25862c3ff5ddcdb4b73900517accb1ec56"
    )

    @staticmethod
    def fixture() -> tuple[tuple[bytes, bytes, bytes], bytes]:
        thwc = bytes(range(36))
        return tuple(thwc[channel::3] for channel in range(3)), thwc[0::3]

    def test_float32_cthw_digest_matches_complete_torch_2_2_1_oracle(self) -> None:
        source_planes, mask_plane = self.fixture()
        replay = GATE._prepared_tensor_digests(source_planes, mask_plane)
        self.assertEqual(replay["source_tensor_sha256"], self.TORCH_SOURCE_SHA256)
        self.assertEqual(replay["mask_tensor_sha256"], self.TORCH_MASK_SHA256)

        # If torch is installed, independently regenerate the whole fixture.
        # Fixed oracle hashes keep the test active (not skipped) without torch.
        if importlib.util.find_spec("torch") is not None:
            import torch

            thwc = torch.tensor(list(range(36)), dtype=torch.uint8).reshape(
                2, 2, 3, 3
            )
            source = (
                thwc.permute(0, 3, 1, 2)
                .transpose(0, 1)
                .float()
                .div_(127.5)
                .sub_(1.0)
                .contiguous()
            )
            mask = torch.clamp(
                (source[:1, :, :, :] + 1.0) / 2.0,
                min=0.0,
                max=1.0,
            ).contiguous()
            self.assertEqual(
                hashlib.sha256(bytes(source.untyped_storage())).hexdigest(),
                self.TORCH_SOURCE_SHA256,
            )
            self.assertEqual(
                hashlib.sha256(bytes(mask.untyped_storage())).hexdigest(),
                self.TORCH_MASK_SHA256,
            )

    def test_float32_digest_locks_channel_frame_order_and_every_byte(self) -> None:
        source_planes, mask_plane = self.fixture()
        expected = GATE._prepared_tensor_digests(source_planes, mask_plane)

        reordered = GATE._prepared_tensor_digests(
            (source_planes[1], source_planes[0], source_planes[2]),
            mask_plane,
        )
        self.assertNotEqual(
            reordered["source_tensor_sha256"],
            expected["source_tensor_sha256"],
        )

        changed_mask = bytearray(mask_plane)
        changed_mask[-1] ^= 1
        changed = GATE._prepared_tensor_digests(source_planes, changed_mask)
        self.assertNotEqual(
            changed["mask_tensor_sha256"],
            expected["mask_tensor_sha256"],
        )

    def test_self_consistent_fake_pre_post_digest_cannot_pass_replay(self) -> None:
        source_planes, mask_plane = self.fixture()
        trace = {
            "digest_definition": (
                "sha256(torch.float32 contiguous CPU little-endian C-order bytes)"
            ),
            "source_tensor": {
                "shape": [3, 2, 2, 3],
                "pre_generate_sha256": "a" * 64,
                "post_generate_sha256": "a" * 64,
            },
            "mask_tensor": {
                "shape": [1, 2, 2, 3],
                "pre_generate_sha256": "b" * 64,
                "post_generate_sha256": "b" * 64,
            },
        }
        with self.assertRaisesRegex(
            GATE.BoneRemovedV2Error,
            "differs from lossless precanvas replay",
        ):
            GATE._verify_prepared_tensor_replay(trace, source_planes, mask_plane)

    def test_vace_keep_last_formula_is_only_consistency_not_decord_replay(self) -> None:
        timing = {
            "time_base_num": 1,
            "time_base_den": 25,
            "frame_pts": list(range(GATE.FRAME_COUNT)),
            "uniform_step_num": 1,
            "uniform_step_den": 25,
        }
        probe = {
            "width": GATE.RAW_DONOR_WIDTH,
            "height": GATE.RAW_DONOR_HEIGHT,
            "frame_timing": timing,
            "frame_timing_digest": GATE.object_sha256(timing),
        }
        trace = {
            "frame_indices": list(range(GATE.FRAME_COUNT)),
            "resize_crop_applied": False,
        }
        replay = GATE._verify_vace_frame_and_geometry_consistency(
            probe,
            copy.deepcopy(probe),
            trace,
        )
        self.assertTrue(replay["static_ffmpeg_timing_and_frozen_formula_consistent"])
        self.assertFalse(replay["frame_ids_trace_independently_replayed_with_decord"])

        trace["frame_indices"][40] = 39
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "inconsistent"):
            GATE._verify_vace_frame_and_geometry_consistency(
                probe,
                copy.deepcopy(probe),
                trace,
            )


class PngFramingHostileTests(unittest.TestCase):
    def test_single_grayscale_png_has_exact_ihdr_crc_and_iend_eof(self) -> None:
        payload = grayscale_png()
        GATE._validate_single_png_frame(
            payload,
            "oracle support PNG",
            require_grayscale_8bit=True,
        )

        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "after IEND"):
            GATE._validate_single_png_frame(
                payload + payload,
                "concatenated hostile PNG",
                require_grayscale_8bit=True,
            )

        bad_crc = bytearray(payload)
        bad_crc[-1] ^= 1
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "CRC"):
            GATE._validate_single_png_frame(
                bytes(bad_crc),
                "bad CRC hostile PNG",
                require_grayscale_8bit=True,
            )

    def test_support_png_wrong_geometry_or_color_type_fails(self) -> None:
        for width, color_type, message in (
            (GATE.WIDTH - 1, 0, "geometry"),
            (GATE.WIDTH, 2, "8-bit grayscale"),
        ):
            with self.subTest(width=width, color_type=color_type):
                ihdr = struct.pack(
                    ">IIBBBBB",
                    width,
                    GATE.HEIGHT,
                    8,
                    color_type,
                    0,
                    0,
                    0,
                )
                payload = (
                    b"\x89PNG\r\n\x1a\n"
                    + png_chunk(b"IHDR", ihdr)
                    + png_chunk(b"IDAT", zlib.compress(b"\x00"))
                    + png_chunk(b"IEND", b"")
                )
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, message):
                    GATE._validate_single_png_frame(
                        payload,
                        "hostile support PNG",
                        require_grayscale_8bit=True,
                    )


class ProducerReceiptHostileTests(unittest.TestCase):
    def test_valid_contract_is_reachable(self) -> None:
        frozen_generator = (
            ROOT
            / "generate_case01_bone_removed_v2_vace_v1.py"
        )
        self.assertEqual(
            actual_file_row(frozen_generator),
            {
                "path": str(frozen_generator),
                "sha256": GATE.GENERATOR_SHA256,
                "size": GATE.GENERATOR_SIZE,
            },
        )
        value = producer_receipt()
        self.assertEqual(
            value["generator"]["program"],
            {
                "path": GATE.GENERATOR_PATH,
                "sha256": GATE.GENERATOR_SHA256,
                "size": GATE.GENERATOR_SIZE,
            },
        )
        GATE.validate_producer_receipt(value)
        obs = observations(
            value["canonical_candidate"]["video"]["sha256"],
            value["support"]["tube"]["sha256"],
        )
        GATE.validate_observations(
            obs,
            candidate_sha256=value["canonical_candidate"]["video"]["sha256"],
            delivery_sha256=value["delivery_candidate"]["video"]["sha256"],
            support_sha256=value["support"]["tube"]["sha256"],
        )
        GATE.evaluate_metric_summary(passing_metrics())

    def test_support_frame_index_bool_cannot_alias_integer_zero(self) -> None:
        value = producer_receipt()
        value["support"]["frame_masks"][0]["frame_index"] = False
        refresh_digest(value, "receipt_digest")
        with self.assertRaisesRegex(
            GATE.BoneRemovedV2Error,
            "support frame-mask order differs",
        ):
            GATE.validate_producer_receipt(value)

    def test_frozen_generator_path_hash_and_size_each_reject_drift(self) -> None:
        for field, hostile in (
            ("path", GATE.GENERATOR_PATH + ".hostile"),
            ("sha256", "0" * 64),
            ("size", GATE.GENERATOR_SIZE + 1),
        ):
            with self.subTest(field=field):
                value = producer_receipt()
                value["generator"]["program"][field] = hostile
                value["receipt_digest"] = GATE.object_sha256(
                    {key: row for key, row in value.items() if key != "receipt_digest"}
                )
                with self.assertRaisesRegex(
                    GATE.BoneRemovedV2Error,
                    "frozen producer",
                ):
                    GATE.validate_producer_receipt(value)

    def test_old_interpolation_algorithm_cannot_be_resealed(self) -> None:
        for algorithm in sorted(GATE.FORBIDDEN_ALGORITHM_IDS):
            with self.subTest(algorithm=algorithm):
                value = producer_receipt()
                value["generator"]["algorithm_id"] = algorithm
                value["receipt_digest"] = GATE.object_sha256(
                    {key: row for key, row in value.items() if key != "receipt_digest"}
                )
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "forbidden old interpolation"):
                    GATE.validate_producer_receipt(value)

    def test_old_video_and_dilate3_support_are_independently_forbidden(self) -> None:
        value = producer_receipt()
        value["delivery_candidate"]["video"]["sha256"] = GATE.OLD_BONE_REMOVED_VIDEO_SHA256
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "old delivery video"):
            GATE.validate_producer_receipt(value)

        value = producer_receipt()
        value["support"]["tube"]["sha256"] = GATE.OLD_REMOVAL_SUPPORT_SHA256
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "old dilate3 support"):
            GATE.validate_producer_receipt(value)

    def test_actual_r4_files_match_forbidden_lineage_and_cannot_pass(self) -> None:
        old_root = REPO_ROOT / "artifacts/object_grounded_case01_0821_bone_interventions_r4"
        video_sha, _ = GATE.stable_file(old_root / "videos/bone_removed.mp4")
        support_sha, _ = GATE.stable_file(old_root / "tubes/removal_support_dilate3.mkv")
        self.assertEqual(video_sha, GATE.OLD_BONE_REMOVED_VIDEO_SHA256)
        self.assertEqual(support_sha, GATE.OLD_REMOVAL_SUPPORT_SHA256)

        value = producer_receipt()
        value["delivery_candidate"]["video"]["sha256"] = video_sha
        value["support"]["tube"]["sha256"] = support_sha
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "old dilate3 support"):
            GATE.validate_producer_receipt(value)

    def test_whole_frame_generation_and_nonhard_composite_fail(self) -> None:
        for field, bad_value, message in (
            ("whole_frame_generation", True, "whole-frame generation"),
            ("outside_support_hard_composite_source_rgb", False, "hard composite"),
            ("uses_bidirectional_boundary_interpolation", True, "boundary interpolation"),
            ("uses_ffmpeg_removelogo", True, "removelogo"),
        ):
            with self.subTest(field=field):
                value = producer_receipt()
                value["generator"][field] = bad_value
                value["receipt_digest"] = GATE.object_sha256(
                    {key: row for key, row in value.items() if key != "receipt_digest"}
                )
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, message):
                    GATE.validate_producer_receipt(value)

    def test_raw_support_donor_is_bound_and_cannot_claim_identity(self) -> None:
        value = producer_receipt()
        value["generator"]["raw_support_donor"]["source_or_identity_authority"] = True
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "improperly claims"):
            GATE.validate_producer_receipt(value)

        value = producer_receipt()
        value["generator"]["raw_support_donor"]["video"]["sha256"] = (
            GATE.OLD_BONE_REMOVED_VIDEO_SHA256
        )
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "old bone-removed"):
            GATE.validate_producer_receipt(value)

        value = producer_receipt()
        value["generator"]["raw_support_donor"]["normalization"]["pad_left"] = 5
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "normalization differs"):
            GATE.validate_producer_receipt(value)

        value = producer_receipt()
        value["generator"]["raw_support_donor"]["normalization"][
            "prepare_source_trace"
        ]["source_tensor"]["post_generate_sha256"] = "e" * 64
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "before/after"):
            GATE.validate_producer_receipt(value)

    def test_unpinned_media_tools_fail_before_any_decode(self) -> None:
        for name, expected_message in (
            ("ffmpeg", "ffmpeg authority"),
            ("ffprobe", "ffprobe authority"),
        ):
            with self.subTest(name=name):
                value = producer_receipt()
                value["media_tools"][name]["sha256"] = "a" * 64
                value["receipt_digest"] = GATE.object_sha256(
                    {key: row for key, row in value.items() if key != "receipt_digest"}
                )
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, expected_message):
                    GATE.validate_producer_receipt(value)

        value = producer_receipt()
        value["media_tools"]["ffprobe"]["path"] = "/usr/bin/ffprobe"
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "ffprobe authority"):
            GATE.validate_producer_receipt(value)

    def test_contract_duplicate_expected_keys_fail_closed(self) -> None:
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "duplicate expected"):
            GATE._exact_keys({"one": 1}, ("one", "one"), "hostile contract")

    def test_canonical_ffv1_requires_actual_8bit_rgb_bgr0_storage(self) -> None:
        for hostile_format in ("yuv420p", "gbrp"):
            with self.subTest(hostile_format=hostile_format):
                value = producer_receipt()
                value["canonical_candidate"]["stored_pixel_format"] = hostile_format
                value["receipt_digest"] = GATE.object_sha256(
                    {key: row for key, row in value.items() if key != "receipt_digest"}
                )
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "bgr0"):
                    GATE.validate_producer_receipt(value)

    def test_ffv1_bgr0_encode_to_rgb24_decode_is_byte_exact(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        self.assertIsNotNone(ffmpeg, "ffmpeg is required for the bgr0 roundtrip test")
        width, height, frame_count = 8, 6, 2
        source = bytes(
            (index * 37 + 19) % 256
            for index in range(width * height * 3 * frame_count)
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            raw_path = root / "source.rgb"
            encoded_path = root / "authority.mkv"
            raw_path.write_bytes(source)
            encode = GATE.subprocess.run(
                (
                    str(ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    "25",
                    "-i",
                    str(raw_path),
                    "-frames:v",
                    str(frame_count),
                    "-c:v",
                    "ffv1",
                    "-pix_fmt",
                    "bgr0",
                    str(encoded_path),
                ),
                stdout=GATE.subprocess.PIPE,
                stderr=GATE.subprocess.PIPE,
                check=False,
            )
            self.assertEqual(encode.returncode, 0, encode.stderr)
            inspect = GATE.subprocess.run(
                (
                    str(ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "info",
                    "-i",
                    str(encoded_path),
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ),
                stdout=GATE.subprocess.PIPE,
                stderr=GATE.subprocess.PIPE,
                check=False,
            )
            self.assertEqual(inspect.returncode, 0, inspect.stderr)
            self.assertRegex(
                inspect.stderr.decode("utf-8", errors="replace"),
                r"Video: ffv1[^\n]*, bgr0(?:\(|,)",
            )
            decode = GATE.subprocess.run(
                (
                    str(ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(encoded_path),
                    "-frames:v",
                    str(frame_count),
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ),
                stdout=GATE.subprocess.PIPE,
                stderr=GATE.subprocess.PIPE,
                check=False,
            )
            self.assertEqual(decode.returncode, 0, decode.stderr)
            self.assertEqual(decode.stdout, source)

        value = producer_receipt()
        value["claim_limits"][
            "downstream_identity_sensitive_consumption_requires_canonical"
        ] = False
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "canonical downstream"):
            GATE.validate_producer_receipt(value)

        for field, message in (
            ("actual_downstream_consumer_verified", "actual downstream"),
            ("generation_execution_lineage_verified", "execution lineage"),
        ):
            with self.subTest(field=field):
                value = producer_receipt()
                value["claim_limits"][field] = True
                value["receipt_digest"] = GATE.object_sha256(
                    {key: row for key, row in value.items() if key != "receipt_digest"}
                )
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, message):
                    GATE.validate_producer_receipt(value)

    def test_model_authority_is_closed_and_actual_checkpoint_is_replayed(self) -> None:
        value = producer_receipt()
        value["generator"]["model_authorities"] = []
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "exact VACE/runtime"):
            GATE.validate_producer_receipt(value)

        value = producer_receipt()
        value["generator"]["generative_inpainting_inside_support"] = False
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "non-generative"):
            GATE.validate_producer_receipt(value)

        approved = b"approved-checkpoint-v1"
        hostile = b"hostile--checkpoint-v1"
        self.assertEqual(len(approved), len(hostile))
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            checkpoint = Path(temporary).resolve() / "checkpoint.bin"
            checkpoint.write_bytes(hostile)
            generator = {
                "model_authorities": [
                    {
                        "role": "inpainting_checkpoint",
                        "path": str(checkpoint),
                        "sha256": hashlib.sha256(approved).hexdigest(),
                        "size": len(approved),
                    }
                ]
            }
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "SHA-256 does not match"):
                GATE._verify_model_authorities(generator)

    def test_authority_tree_manifest_replays_exact_actual_inventory(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            tree = root / "vace-source"
            nested = tree / "package"
            nested.mkdir(parents=True)
            (tree / "README").write_bytes(b"pinned source tree")
            (nested / "module.py").write_bytes(b"VALUE = 1\n")
            manifest = authority_tree_manifest("vace_source_tree", tree)
            manifest_path = root / "vace-source-manifest.json"
            manifest_path.write_bytes(GATE.canonical_json_bytes(manifest) + b"\n")
            replay = GATE._replay_authority_tree_manifest(
                manifest,
                manifest_path=manifest_path,
                expected_role="vace_source_tree",
            )
            self.assertEqual(replay["file_count"], 2)

            (tree / "unmanifested.py").write_bytes(b"hostile extra")
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "extra file"):
                GATE._replay_authority_tree_manifest(
                    manifest,
                    manifest_path=manifest_path,
                    expected_role="vace_source_tree",
                )

    def test_three_model_tree_replays_match_before_after_generation_digest(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            authorities = []
            replay_rows = []
            for role in GATE.MODEL_AUTHORITY_ROLES:
                tree = root / f"{role}-root"
                tree.mkdir()
                (tree / "authority.bin").write_bytes(role.encode("ascii"))
                manifest = authority_tree_manifest(role, tree)
                manifest_path = root / f"{role}-manifest.json"
                manifest_path.write_bytes(GATE.canonical_json_bytes(manifest) + b"\n")
                manifest_row = {"role": role, **actual_file_row(manifest_path)}
                authorities.append(manifest_row)
                replay_rows.append(
                    {
                        "role": role,
                        "manifest_sha256": manifest_row["sha256"],
                        "tree_digest": manifest["tree_digest"],
                    }
                )
            replay_rows.sort(key=lambda row: row["role"])
            replay_digest = GATE.object_sha256(replay_rows)
            generator = {
                "model_authorities": authorities,
                "authority_replay": {
                    "before_generation_digest": replay_digest,
                    "after_generation_digest": replay_digest,
                    "unchanged": True,
                },
            }
            self.assertEqual(
                len(GATE._replay_model_authority_manifests(generator)),
                len(GATE.MODEL_AUTHORITY_ROLES),
            )
            generator["authority_replay"]["after_generation_digest"] = "f" * 64
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "actual authority tree replay"):
                GATE._replay_model_authority_manifests(generator)

    def test_dual_support_review_replays_packet_receipts_and_81_masks(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            producer, review, mask_root = dual_support_review_fixture(root)
            payloads, replay = GATE._replay_support_review(review, producer=producer)
            self.assertEqual(len(payloads), GATE.FRAME_COUNT)
            self.assertEqual(replay["frame_count"], GATE.FRAME_COUNT)
            self.assertEqual(
                replay["reviewer_identities"],
                [
                    "external-support-reviewer-one",
                    "external-support-reviewer-two",
                ],
            )
            self.assertEqual(
                replay["external_review_receipt_sha256s"],
                [
                    row["receipt"]["sha256"]
                    for row in review["external_reviews"]
                ],
            )
            self.assertNotIn("reviewer_id", replay)
            self.assertEqual(replay["review_digest"], review["review_digest"])

            (mask_root / "00017.png").write_bytes(b"same-size-hostile")
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "support reviewed-mask file 17",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_dual_support_review_rejects_deleted_second_reviewer(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            producer, review, _ = dual_support_review_fixture(
                Path(temporary).resolve()
            )
            del review["external_reviews"][1]
            refresh_digest(review, "review_digest")
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "exactly two external reviewers",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_dual_support_review_rejects_producer_mask_binding_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            producer, review, _ = dual_support_review_fixture(
                Path(temporary).resolve()
            )
            producer["support"]["frame_masks"][29]["path"] = (
                "/promotion/bundle/drifted-00029.png"
            )
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "support reviewed-mask binding differs: 29",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_dual_support_review_rejects_duplicate_reviewer_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            producer, review, _ = dual_support_review_fixture(
                Path(temporary).resolve(),
                reviewer_identities=("duplicate-reviewer", "duplicate-reviewer"),
            )
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "external reviewer identity repeats",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_dual_support_review_rejects_receipt_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            producer, review, _ = dual_support_review_fixture(
                Path(temporary).resolve()
            )
            receipt_path = Path(review["external_reviews"][0]["receipt"]["path"])
            receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "external review receipt SHA-256",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_dual_support_review_rejects_evidence_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            producer, review, _ = dual_support_review_fixture(
                Path(temporary).resolve()
            )
            evidence_path = Path(review["external_reviews"][1]["evidence"]["path"])
            evidence_path.write_bytes(evidence_path.read_bytes() + b" ")
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "external review evidence SHA-256",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_dual_support_review_rejects_candidate_manifest_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            producer, review, _ = dual_support_review_fixture(root)
            drifted_manifest = root / "drifted-manifest.json"
            drifted_manifest.write_bytes(
                (SUPPORT_PACKET_ROOT / "manifest.json").read_bytes() + b" "
            )
            review["candidate_packet"]["manifest"] = actual_file_row(
                drifted_manifest
            )
            refresh_digest(review, "review_digest")
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "candidate packet manifest authority differs",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_dual_support_review_rejects_one_nonpass_frame(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            producer, review, _ = dual_support_review_fixture(
                Path(temporary).resolve(),
                frame_updates={2: (37, {"decision": "FAIL"})},
            )
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "external review frame ballot differs: 37",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_dual_support_review_rejects_boundary_edit_request(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            producer, review, _ = dual_support_review_fixture(
                Path(temporary).resolve(),
                frame_updates={1: (23, {"boundary_edit_requested": True})},
            )
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "external review frame ballot differs: 23",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_dual_support_review_rejects_packet_inventory_extra(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            producer, review, _ = dual_support_review_fixture(root)
            sums_path = root / "SHA256SUMS-with-extra"
            sums_path.write_bytes(
                (SUPPORT_PACKET_ROOT / "SHA256SUMS").read_bytes()
                + b"0000000000000000000000000000000000000000000000000000000000000000  extra.bin\n"
            )
            review["candidate_packet"]["sha256sums"] = actual_file_row(sums_path)
            refresh_digest(review, "review_digest")
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "candidate packet SHA256SUMS inventory differs",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_dual_support_review_rejects_claim_overstatement(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            producer, review, _ = dual_support_review_fixture(
                Path(temporary).resolve()
            )
            review["claim_limits"][
                "reviewer_identity_verified_by_generator"
            ] = True
            refresh_digest(review, "review_digest")
            with self.assertRaisesRegex(
                GATE.BoneRemovedV2Error,
                "support review overclaims external facts",
            ):
                GATE._replay_support_review(review, producer=producer)

    def test_digest_and_extra_key_tampering_fail_closed(self) -> None:
        value = producer_receipt()
        value["generator"]["seed"] += 1
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "digest mismatch"):
            GATE.validate_producer_receipt(value)

        value = producer_receipt()
        value["self_attested_clean"] = True
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "key closure"):
            GATE.validate_producer_receipt(value)

    def test_create_only_controller_must_be_independent_and_receipted(self) -> None:
        value = producer_receipt()
        value["create_only_authority"]["controller_program"]["sha256"] = value[
            "generator"
        ]["program"]["sha256"]
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "not independent"):
            GATE.validate_producer_receipt(value)

        value = producer_receipt()
        del value["create_only_authority"]["attempt_receipt"]
        value["receipt_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "key closure"):
            GATE.validate_producer_receipt(value)

    def test_create_only_receipts_cross_bind_attempt_and_published_assets(self) -> None:
        value = producer_receipt()
        attempt, publication = create_only_receipts(value)
        replay = GATE.validate_create_only_receipts(
            attempt,
            publication,
            producer=value,
        )
        self.assertEqual(replay["attempt_token"], "7" * 64)

        hostile = copy.deepcopy(publication)
        hostile["attempt_token"] = "8" * 64
        hostile["publication_digest"] = GATE.object_sha256(
            {key: row for key, row in hostile.items() if key != "publication_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "attempt token"):
            GATE.validate_create_only_receipts(attempt, hostile, producer=value)

        hostile = copy.deepcopy(publication)
        hostile["publication"]["overwrite_performed"] = True
        hostile["publication_digest"] = GATE.object_sha256(
            {key: row for key, row in hostile.items() if key != "publication_digest"}
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "overwrote"):
            GATE.validate_create_only_receipts(attempt, hostile, producer=value)

        with mock.patch.object(GATE.os.path, "lexists", return_value=True):
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "actual publication staging"):
                GATE.validate_create_only_receipts(attempt, publication, producer=value)

        hostile_attempt = copy.deepcopy(attempt)
        hostile_attempt["model_authorities_digest"] = "9" * 64
        hostile_attempt["attempt_digest"] = GATE.object_sha256(
            {
                key: row
                for key, row in hostile_attempt.items()
                if key != "attempt_digest"
            }
        )
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "model authority binding"):
            GATE.validate_create_only_receipts(
                hostile_attempt,
                publication,
                producer=value,
            )


class ObservationHostileTests(unittest.TestCase):
    def setUp(self) -> None:
        receipt = producer_receipt()
        self.candidate = receipt["canonical_candidate"]["video"]["sha256"]
        self.support = receipt["support"]["tube"]["sha256"]

    def validate(self, value: dict) -> None:
        value["observation_digest"] = GATE.object_sha256(
            {key: row for key, row in value.items() if key != "observation_digest"}
        )
        GATE.validate_observations(
            value,
            candidate_sha256=self.candidate,
            delivery_sha256=producer_receipt()["delivery_candidate"]["video"]["sha256"],
            support_sha256=self.support,
        )

    def test_one_reviewer_or_unblinded_review_fails(self) -> None:
        value = observations(self.candidate, self.support)
        value["reviewers"] = value["reviewers"][:1]
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "at least two"):
            self.validate(value)

        value = observations(self.candidate, self.support)
        value["blinding"]["arm_name_hidden"] = False
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "not blinded"):
            self.validate(value)

        value = observations(self.candidate, self.support)
        value["review_protocol"]["convenience_surfaces_used_as_authority"] = True
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "convenience surfaces"):
            self.validate(value)

    def test_any_bone_scar_or_identity_failure_fails(self) -> None:
        for field in (
            "support_bone_shadow_coverage",
            "bone_absence",
            "bone_shaped_scar_absence",
            "seam_absence",
            "texture_collapse_absence",
            "temporal_flicker_absence",
            "cast_shadow_absence",
            "dog_identity_preservation",
            "background_identity_preservation",
        ):
            with self.subTest(field=field):
                value = observations(self.candidate, self.support)
                value["reviewers"][1][field] = {
                    "status": "FAIL",
                    "failure_frames": [17],
                    "note": "Visible failure at frame 17.",
                }
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, field):
                    self.validate(value)

    def test_candidate_and_support_substitution_fail(self) -> None:
        value = observations("f" * 64, self.support)
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "candidate binding"):
            self.validate(value)
        value = observations(self.candidate, "e" * 64)
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "support binding"):
            self.validate(value)


class MetricAndPublicationHostileTests(unittest.TestCase):
    def test_known_smooth_scar_profile_fails(self) -> None:
        # The current r4 asset measured about 0.50 median core/context texture
        # energy and a much lower tail.  It cannot pass merely by changing its
        # receipt or filename.
        value = passing_metrics()
        value["texture_ratio_p10"] = 0.37
        value["texture_ratio_median"] = 0.50
        value["low_texture_frame_count"] = 35
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "low-texture scar tail"):
            GATE.evaluate_metric_summary(value)

    def test_outside_support_and_dog_guard_each_fail(self) -> None:
        for field, message in (
            ("outside_support_changed_pixels", "outside support"),
            ("dog_guard_changed_pixels", "dog-guard"),
        ):
            with self.subTest(field=field):
                value = passing_metrics()
                value[field] = 1
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, message):
                    GATE.evaluate_metric_summary(value)

        value = passing_metrics()
        value["support_pixels_not_equal_raw_donor"] = 1
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "differ from raw donor"):
            GATE.evaluate_metric_summary(value)

        value = passing_metrics()
        value["precanvas_support_pad_active_pixels"] = 1
        with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "donor-free precanvas"):
            GATE.evaluate_metric_summary(value)

    def test_delivery_background_and_dog_guard_distortion_each_fail(self) -> None:
        for field, message in (
            ("delivery_outside_support_rgb_mad_frame_maximum", "delivery background"),
            ("delivery_dog_guard_rgb_mad_frame_maximum", "delivery dog-guard"),
        ):
            with self.subTest(field=field):
                value = passing_metrics()
                value[field] = GATE.MAX_DELIVERY_RGB_MAD_FRAME + 0.01
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, message):
                    GATE.evaluate_metric_summary(value)

    def test_minimum_support_dilation_is_geometry_not_self_attestation(self) -> None:
        bone = {300 * GATE.WIDTH + 400}
        dilated = GATE.dilate(bone, GATE.MIN_SUPPORT_DILATION)
        self.assertGreater(len(dilated), 1)
        self.assertTrue(bone < dilated)
        self.assertFalse(dilated <= GATE.dilate(bone, 3))

    def test_report_write_is_create_only(self) -> None:
        report = {"schema_version": "test", "status": "PASS"}
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            path = Path(temporary).resolve() / "report.json"
            GATE.write_create_only(path, report)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "fresh report path"):
                GATE.write_create_only(path, report)

    def test_actual_published_tree_must_be_exact_and_single_link(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            temporary_root = Path(temporary).resolve()
            final_root = temporary_root / "fresh-v2"
            final_root.mkdir()
            assets = {
                "support": final_root / "support.mkv",
                "canonical_candidate": final_root / "canonical.mkv",
                "delivery_candidate": final_root / "delivery.mp4",
            }
            for index, path in enumerate(assets.values()):
                path.write_bytes(f"asset-{index}".encode("ascii"))
            self.assertEqual(
                set(GATE._verify_exact_published_tree(final_root, assets)),
                {"support.mkv", "canonical.mkv", "delivery.mp4"},
            )

            extra = final_root / "unpublished-clean-looking-surface.png"
            extra.write_bytes(b"not authority")
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "extra file"):
                GATE._verify_exact_published_tree(final_root, assets)
            extra.unlink()

            outside_hardlink = temporary_root / "support-alias.mkv"
            os.link(assets["support"], outside_hardlink)
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "single-link"):
                GATE._verify_exact_published_tree(final_root, assets)

    def test_held_authority_rejects_named_inode_substitution(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            authority = root / "candidate.mkv"
            original = b"authority-A"
            authority.write_bytes(original)
            row = {
                "path": str(authority),
                "sha256": hashlib.sha256(original).hexdigest(),
                "size": len(original),
            }
            held = GATE._open_held_file_row(row, "hostile held candidate")
            replacement = root / "replacement.mkv"
            replacement.write_bytes(b"authority-B")
            os.replace(replacement, authority)
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "named join changed"):
                held.close_verified()

    def test_static_ffmpeg_probe_consumes_held_executable_and_media_fds(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            executable_path = root / "ffmpeg"
            media_path = root / "candidate.mkv"
            executable_path.write_bytes(b"pinned executable")
            media_path.write_bytes(b"pinned media")
            executable = GATE._open_held_file_row(
                actual_file_row(executable_path),
                "held static ffmpeg",
            )
            media = GATE._open_held_file_row(
                actual_file_row(media_path),
                "held media",
            )
            showinfo = [
                "Input #0, matroska,webm, from held-fd:",
                (
                    "  Stream #0:0: Video: ffv1 (FFV1), "
                    f"bgr0(pc), {GATE.WIDTH}x{GATE.HEIGHT}, 25 fps"
                ),
                "Stream mapping:",
                (
                    "[Parsed_showinfo_0 @ fixture] config in time_base: "
                    "1/25, frame_rate: 25/1"
                ),
            ]
            showinfo.extend(
                (
                    "[Parsed_showinfo_0 @ fixture] "
                    f"n: {index} pts: {index} pts_time:{index / 25} "
                    f"fmt:bgr0 s:{GATE.WIDTH}x{GATE.HEIGHT}"
                )
                for index in range(GATE.FRAME_COUNT)
            )
            showinfo_text = "\n".join(showinfo)
            completed = mock.Mock(returncode=0, stdout="", stderr=showinfo_text)
            try:
                with mock.patch.object(GATE.sys, "platform", "linux"):
                    with mock.patch.object(
                        GATE.subprocess,
                        "run",
                        return_value=completed,
                    ) as run:
                        GATE._probe_video(executable, media)
                command = run.call_args.args[0]
                self.assertEqual(command[0], f"/proc/self/fd/{executable.descriptor}")
                self.assertEqual(
                    command[command.index("-i") + 1],
                    f"/proc/self/fd/{media.descriptor}",
                )
                self.assertEqual(
                    set(run.call_args.kwargs["pass_fds"]),
                    {executable.descriptor, media.descriptor},
                )

                extra_stream = showinfo_text.replace(
                    "Stream mapping:",
                    "  Stream #0:1: Audio: aac, 48000 Hz\nStream mapping:",
                    1,
                )
                with self.assertRaisesRegex(
                    GATE.BoneRemovedV2Error,
                    "stream closure",
                ):
                    GATE._parse_static_ffmpeg_probe(
                        extra_stream,
                        label="hostile extra stream",
                        enforce_fps=True,
                        expected_width=GATE.WIDTH,
                        expected_height=GATE.HEIGHT,
                    )

                bad_pts = showinfo_text.replace(
                    "n: 40 pts: 40 ",
                    "n: 40 pts: 41 ",
                    1,
                )
                with self.assertRaisesRegex(
                    GATE.BoneRemovedV2Error,
                    "not strictly uniform",
                ):
                    GATE._parse_static_ffmpeg_probe(
                        bad_pts,
                        label="hostile PTS",
                        enforce_fps=True,
                        expected_width=GATE.WIDTH,
                        expected_height=GATE.HEIGHT,
                    )
            finally:
                GATE._close_held_files((executable, media))

    def test_zero_write_and_write_exception_leave_no_report(self) -> None:
        report = {"schema_version": "test", "status": "PASS"}
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            zero_path = root / "zero.json"
            with mock.patch.object(GATE.os, "write", return_value=0):
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "zero-byte"):
                    GATE.write_create_only(zero_path, report)
            self.assertFalse(zero_path.exists())

            error_path = root / "error.json"
            with mock.patch.object(GATE.os, "write", side_effect=OSError("hostile write")):
                with self.assertRaisesRegex(OSError, "hostile write"):
                    GATE.write_create_only(error_path, report)
            self.assertFalse(error_path.exists())

    def test_report_reopen_rejects_same_bytes_inode_substitution(self) -> None:
        report = {"schema_version": "test", "status": "PASS"}
        payload = GATE.canonical_json_bytes(report) + b"\n"
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            path = root / "report.json"
            real_open = GATE.os.open
            substituted = False

            def hostile_open(path_value, flags, *args, **kwargs):
                nonlocal substituted
                if (
                    path_value == path.name
                    and kwargs.get("dir_fd") is not None
                    and not flags & os.O_CREAT
                    and not substituted
                ):
                    replacement = root / "replacement.json"
                    replacement.write_bytes(payload)
                    replacement.chmod(0o444)
                    os.replace(replacement, path)
                    substituted = True
                return real_open(path_value, flags, *args, **kwargs)

            with mock.patch.object(GATE.os, "open", side_effect=hostile_open):
                with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "reopened report identity"):
                    GATE.write_create_only(path, report)
            self.assertTrue(substituted)
            # Fail closed without deleting an attacker's replacement inode;
            # the helper's documented scope does not promise failure-path
            # name absence against a same-UID concurrent renamer.
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), payload)

    def test_json_loader_rejects_duplicate_and_noncanonical_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            duplicate = root / "duplicate.json"
            duplicate.write_bytes(b'{"a":1,"a":2}\n')
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "duplicate JSON key"):
                GATE._load_json(duplicate, "hostile duplicate")

            pretty = root / "pretty.json"
            pretty.write_bytes(b'{\n  "a": 1\n}\n')
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "not canonical"):
                GATE._load_json(pretty, "hostile pretty")

            missing_lf = root / "missing-lf.json"
            missing_lf.write_bytes(b'{"a":1}')
            with self.assertRaisesRegex(GATE.BoneRemovedV2Error, "not canonical"):
                GATE._load_json(missing_lf, "hostile missing LF")


if __name__ == "__main__":
    unittest.main()

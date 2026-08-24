#!/usr/bin/env python3
"""Strict plan and result contract for the case01 exact-five R64 canary.

This module is deliberately renderer-free.  It validates a separately
audited asset authority, an exact five-task plan, and delegates each native
receipt/media replay to the frozen r5f evaluator supplied by the caller.
No plan is launchable until the asset authority says both ``PASS`` and
``launch_allowed: true``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "case01-source-bone-exact5-r64-plan-v1"
ASSET_AUTHORITY_SCHEMA = "case01-source-bone-exact5-asset-authority-v1"
ASSET_AUTHORITY_STATUS = "APPROVED_FOR_EXACT5_R64_RENDERER_CANARY"
REPORT_SCHEMA = "case01-source-bone-exact5-r64-report-v1"
EXPERIMENT_ID = "case01-288545b9c031491a-source-bone-exact5-r64-v1"
CAMPAIGN = "case01-source-bone-exact5-r64-canary"
IID = "288545b9c031491a"
INSTRUCTION = "Make the dog pick up the bone and hold it in its mouth."
INSTRUCTION_SHA256 = (
    "84df12ede824d239a4c7c3d21dccdf22663535d1e504e7b280544c8a9be0fd5d"
)
SEED = 2027
VARIANT_ORDER = (
    "exact_original",
    "codec_only_present",
    "bone_removed",
    "bone_translated_up150",
    "sham_control_up150",
)
TASK_IDS = tuple(f"case01-{variant}-full644" for variant in VARIANT_ORDER)
EXPECTED_VIDEO = {
    "frame_count": 81,
    "fps_num": 25,
    "fps_den": 1,
    "width": 704,
    "height": 736,
}
EXPECTED_TREATMENTS = {
    "exact_original": "exact_original_bytes",
    "codec_only_present": "matched_libx264_crf17_codec_only_transcode",
    "bone_removed": (
        "per_frame_SAM2_bone_mask_dilate3_bidirectional_boundary_interpolation"
    ),
    "bone_translated_up150": (
        "same_original_support_interpolation_plus_exact_source_bone_pixels_up150"
    ),
    "sham_control_up150": (
        "same_boundary_interpolation_on_spatially_matched_up150_support"
    ),
}
ASSET_MANIFEST_SHA256 = (
    "0a62b74056f4be1ab17ed632d31068964aed27c607212f58c2a7d17b74becf5e"
)
ASSET_MANIFEST_SIZE = 249_082
ASSET_MANIFEST_DIGEST = (
    "879318860b7d96824ec2da4b10b657b320945285a1607faf8c89bb577a1cc538"
)
ASSET_MANIFEST_SCHEMA = "bernini-case01-matched-bone-interventions-v1"
INDEPENDENT_AUDIT_SCHEMA = "case01-source-bone-exact5-independent-audit-v1"
INDEPENDENT_AUDIT_SHA256 = (
    "040c53a3647ae957212a1d2d6da3ffa75b4207ace07e1c7ba6ce128033dce969"
)
INDEPENDENT_AUDIT_SIZE = 8_285
INDEPENDENT_AUDIT_DIGEST = (
    "13ea77d95e8529585f1bcda1ff5fc9b1f71a42062adfa2994c6dfbe51d22d7d1"
)
INDEPENDENT_AUDIT_FIELDS = frozenset(
    {
        "all81_sheets",
        "all_81_frames_reviewed",
        "audit_digest",
        "audit_digest_definition",
        "auditor_role",
        "claim_limits",
        "decision",
        "decision_scope",
        "dog_overlap_space_time_pixels",
        "findings",
        "full_frame_precodec_audit",
        "manifest_digest",
        "manifest_path",
        "manifest_sha256",
        "manifest_size",
        "matched_codec_audit",
        "materializer",
        "p0_count",
        "p1_count",
        "renderer_outputs_reviewed",
        "schema_version",
        "scope",
        "source_sha256",
        "source_size",
        "status",
        "test_program",
        "test_runs",
        "training_outputs_reviewed",
        "translated_symmetry",
        "variant_order",
        "videos",
        "visual_review",
    }
)
EXPECTED_SOURCE_SHA256 = {
    "exact_original": "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18",
    "codec_only_present": "7104ada43b9f8e0168f38dc9710e8dc76f47606c446e5b119155048929af403b",
    "bone_removed": "8c525385832586fa7b7fd7ae6e5701c599694d26ee27b502dbf0bb582e55e1c9",
    "bone_translated_up150": "5ba28a7efd6269872ba7924162eeba8df2fd29df06c6668557310984ef9ed5f6",
    "sham_control_up150": "ea380344b1d5c644beee9c886a67b413170d26517bab3315daed32a388ebfac3",
}
EXPECTED_SOURCE_SIZE = {
    "exact_original": 10_887_043,
    "codec_only_present": 5_432_063,
    "bone_removed": 5_424_975,
    "bone_translated_up150": 5_441_472,
    "sham_control_up150": 5_420_413,
}
EXPECTED_ALL81_SHEETS = {
    "exact_original": {
        "path": (
            "artifacts/object_grounded_case01_0821_bone_interventions_r4/"
            "qa/all81_exact_original_9x9.jpg"
        ),
        "sha256": "aa345fba578cd9008656c7992fbce1d90d27ee04bb02b5f870e5936cb4f60c0c",
        "size": 310_091,
    },
    "codec_only_present": {
        "path": (
            "artifacts/object_grounded_case01_0821_bone_interventions_r4/"
            "qa/all81_codec_only_present_9x9.jpg"
        ),
        "sha256": "494ac2d772c3ae68aee30118f196ba0b0dc28906b527cc030723fcda6dfaab27",
        "size": 318_573,
    },
    "bone_removed": {
        "path": (
            "artifacts/object_grounded_case01_0821_bone_interventions_r4/"
            "qa/all81_bone_removed_9x9.jpg"
        ),
        "sha256": "8fe8b3770ed5c9fed3aa3f98b5752253e8e3135e6e1c1a6fa2df34219434f4cd",
        "size": 304_020,
    },
    "bone_translated_up150": {
        "path": (
            "artifacts/object_grounded_case01_0821_bone_interventions_r4/"
            "qa/all81_bone_translated_up150_9x9.jpg"
        ),
        "sha256": "caf3143c76685bec64beba634b16bfcdd9012a8b3fb6cac22b191ce6dadd55c9",
        "size": 319_956,
    },
    "sham_control_up150": {
        "path": (
            "artifacts/object_grounded_case01_0821_bone_interventions_r4/"
            "qa/all81_sham_control_up150_9x9.jpg"
        ),
        "sha256": "98e4f256f090d3ed0d6205caeefbb1cfc15e9d1d479545597a630339f1fcdd69",
        "size": 317_701,
    },
}
REFERENCE_EXACT_ORIGINAL_R64_OUTPUT_SHA256 = (
    "e0d3c07d1d3e6ae4d45e59713d2af3f04786c305f8842c20d79172a9cae22403"
)
EXPECTED_CHECKPOINT = {
    "sha256": "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2",
    "manifest_digest": "7bae23da51a3c5a67adb41ee85dd026c374d2581bd3409e868e18b2f6f4dffc4",
    "global_step": 644,
    "receipt_digest": "aaf348a7daa6c5ca2fe721771857287125ee02eb2c9a499f45b11a2e113d15d7",
    "file_count": 5,
    "adapter_config_sha256": "94bfaf73d714d7e77095ff68ce57e24932e0c05bde324263f5fe321660b95f62",
    "adapter_model_sha256": "44efdc5a0501238250b1d32ae2859abe248ffc37b152cd8db86ff84b378d6b22",
    "training_receipt_sha256": "3402c8c93c092bfc4490bf86790ab6429b4cbaad38358956cb0beeb5df7d4c4c",
    "optimizer_sha256": "77b7b22db4da92f28f23b4ae91c7271f55ab6a92353bfc8b0bbeb30529a7af63",
}
EXPECTED_PRODUCER = {
    "inference_receipt_schema": "bernini-r-1p3b-action-lora-inference-receipt-v5",
    "infer_lora_sha256": "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
    "method_source_revision": "ce4cffc1e8a144448c92252d9fb63087f03bbd8c",
    "method_source_archive_sha256": "12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828",
    "ffprobe_sha256": "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5",
}
SHA256_RE = __import__("re").compile(r"[0-9a-f]{64}")


class Exact5EvalError(RuntimeError):
    """The exact-five asset, plan, or result closure differs."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Exact5EvalError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_rdev,
        info.st_size,
        getattr(info, "st_blocks", 0),
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def stable_file(
    path_value: str | Path,
    *,
    expected_sha256: str | None = None,
    return_bytes: bool = False,
) -> tuple[bytes | None, str, int]:
    path = Path(path_value).expanduser()
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise Exact5EvalError(f"file path is not one canonical inode: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
            if return_bytes:
                chunks.append(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    observed = digest.hexdigest()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or size != before.st_size
        or (expected_sha256 is not None and observed != expected_sha256)
    ):
        raise Exact5EvalError(f"stable file identity/SHA differs: {path}")
    return (b"".join(chunks) if return_bytes else None), observed, size


def _strict_digest(value: Mapping[str, Any], field: str, *, label: str) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    if (
        type(claimed) is not str
        or SHA256_RE.fullmatch(claimed) is None
        or claimed != object_sha256(unsigned)
    ):
        raise Exact5EvalError(f"{label} digest differs")
    return claimed


def validate_independent_audit_receipt(
    value: Mapping[str, Any], *, raw: bytes, sha256: str, size: int
) -> dict[str, Any]:
    expected_claim_limits = {
        "asset_input_qa_only": True,
        "bone_removed_is_clean_background_recovery": False,
        "exploratory_canary_launch_authorized": True,
        "fully_deconfounded_causal_estimate_authorized": False,
        "interpolation_scar_description": (
            "A moving bone-shaped interpolation scar remains in the removed and "
            "translated original supports, with a shape-matched support scar "
            "shifted up 150 pixels in the sham arm."
        ),
        "interpolation_scar_visible": True,
        "original_location_cast_shadow_guaranteed_removed": False,
        "renderer_result_claim_authorized_by_this_receipt": False,
        "scientific_causal_result_claim_authorized": False,
        "training_claim_authorized_by_this_receipt": False,
        "translated_target_shadow_description": (
            "The translated source bone has no synthesized target-location shadow."
        ),
        "translated_target_shadow_synthesized": False,
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != INDEPENDENT_AUDIT_FIELDS
        or raw != canonical_json_bytes(value) + b"\n"
        or sha256 != INDEPENDENT_AUDIT_SHA256
        or size != INDEPENDENT_AUDIT_SIZE
        or value.get("schema_version") != INDEPENDENT_AUDIT_SCHEMA
        or value.get("status") != "PASS_P0_0_P1_0"
        or value.get("decision") != "GO"
        or value.get("scope") != "exact5_input_assets_only"
        or value.get("decision_scope")
        != "exploratory_object_grounding_canary_input_authority_only"
        or value.get("auditor_role") != "independent_asset_auditor"
        or value.get("manifest_path")
        != "artifacts/object_grounded_case01_0821_bone_interventions_r4/manifest.json"
        or value.get("manifest_sha256") != ASSET_MANIFEST_SHA256
        or value.get("manifest_size") != ASSET_MANIFEST_SIZE
        or value.get("manifest_digest") != ASSET_MANIFEST_DIGEST
        or value.get("source_sha256") != EXPECTED_SOURCE_SHA256
        or value.get("source_size") != EXPECTED_SOURCE_SIZE
        or value.get("variant_order") != list(VARIANT_ORDER)
        or value.get("all_81_frames_reviewed") is not True
        or value.get("renderer_outputs_reviewed") is not False
        or value.get("training_outputs_reviewed") is not False
        or value.get("p0_count") != 0
        or value.get("p1_count") != 0
        or value.get("findings") != {"p0": [], "p1": []}
        or value.get("dog_overlap_space_time_pixels")
        != {"removal_support": 0, "sham_support": 0, "translated_bone": 0}
        or value.get("claim_limits") != expected_claim_limits
        or value.get("audit_digest_definition")
        != (
            "sha256(canonical UTF-8 JSON after removing audit_digest; "
            "sort_keys=true; separators=(',',':'))"
        )
    ):
        raise Exact5EvalError("independent audit identity/claim closure differs")
    if value.get("full_frame_precodec_audit") != {
        "arms": {
            "bone_removed": {
                "changed_space_time_pixels": 430101,
                "declared_support_space_time_pixels": 430234,
                "outside_declared_support_changed_pixels": 0,
            },
            "bone_translated_up150": {
                "changed_space_time_pixels": 859280,
                "declared_support_space_time_pixels": 860468,
                "outside_declared_support_changed_pixels": 0,
            },
            "codec_only_present": {
                "changed_space_time_pixels": 0,
                "declared_support_space_time_pixels": 0,
                "outside_declared_support_changed_pixels": 0,
            },
            "sham_control_up150": {
                "changed_space_time_pixels": 428600,
                "declared_support_space_time_pixels": 430234,
                "outside_declared_support_changed_pixels": 0,
            },
        },
        "every_frame_pixel_scanned": True,
        "frame_count": 81,
        "frame_hash_mismatches": 0,
        "frame_pixels": 518144,
        "sequence_digest_mismatches": 0,
    }:
        raise Exact5EvalError("independent full-frame pixel audit differs")
    if value.get("matched_codec_audit") != {
        "codec_headers_equal": True,
        "durations_equal": True,
        "frame_type_sequences_equal": True,
        "keyframe_indices": [0],
        "matched_variant_order": list(VARIANT_ORDER[1:]),
        "timestamps_equal": True,
    }:
        raise Exact5EvalError("independent matched-codec audit differs")
    if value.get("translated_symmetry") != {
        "failures": 0,
        "frame_count": 81,
        "original_and_target_supports_disjoint": True,
        "original_support_equals_removed_pixels": 430234,
        "target_bone_equals_shifted_source_pixels": 309530,
        "target_nonbone_support_equals_sham_pixels": 120704,
    }:
        raise Exact5EvalError("independent translated-symmetry audit differs")
    if value.get("visual_review") != {
        "bone_present_all_frames_in_codec_only_present": True,
        "bone_present_all_frames_in_exact_original": True,
        "bone_present_all_frames_in_sham_control_original_location": True,
        "bone_removed_all_frames_from_original_support_in_bone_removed": True,
        "bone_removed_all_frames_from_original_support_in_translated": True,
        "mask_id_switch_observed": False,
        "old_fixed_union_rectangular_blur_observed": False,
        "translated_bone_clipped_frames": 0,
        "translated_bone_continuous_all_frames": True,
    }:
        raise Exact5EvalError("independent visual review differs")
    if value.get("materializer") != {
        "path": "methods/bernini_action_editing/materialize_case01_bone_interventions_v1.py",
        "sha256": "271d71c9742d7c847cceb10291e2d1b8957c652096f10f86e1a3058d818e7d7a",
        "size": 40340,
    } or value.get("test_program") != {
        "path": "methods/bernini_action_editing/tests/test_materialize_case01_bone_interventions_v1.py",
        "sha256": "2bf7d8eb2938316ab467c2abe0414329bafb17261aea17e25de091f7d0556467",
        "size": 4816,
    } or value.get("test_runs") != {
        "normal": {"errors": 0, "failures": 0, "passed": 5, "ran": 5},
        "optimized_python": {
            "errors": 0,
            "failures": 0,
            "optimization_flag": "-O",
            "passed": 5,
            "ran": 5,
        },
    }:
        raise Exact5EvalError("independent producer/test audit differs")
    videos = value.get("videos")
    sheets = value.get("all81_sheets")
    if not isinstance(videos, Mapping) or set(videos) != set(VARIANT_ORDER):
        raise Exact5EvalError("independent video audit rows differ")
    if not isinstance(sheets, Mapping) or set(sheets) != set(VARIANT_ORDER):
        raise Exact5EvalError("independent all81 sheet rows differ")
    for variant in VARIANT_ORDER:
        video = videos[variant]
        sheet = sheets[variant]
        expected_media = {
            "audio": False,
            "avg_frame_rate": "25/1",
            "codec_name": "h264",
            "duration_seconds": 3.24,
            "encoding_role": (
                "byte_exact_source_provenance"
                if variant == "exact_original"
                else "matched_codec_arm"
            ),
            "frame_count": 81,
            "height": 736,
            "pixel_format": "yuv420p",
            "r_frame_rate": "25/1",
            "width": 704,
        }
        if variant != "exact_original":
            expected_media.update(
                {"crf": 17, "encoder": "libx264", "preset": "medium"}
            )
        if (
            video
            != {
                "path": (
                    "artifacts/object_grounded_case01_0821_bone_interventions_r4/"
                    f"videos/{variant}.mp4"
                ),
                "sha256": EXPECTED_SOURCE_SHA256[variant],
                "size": EXPECTED_SOURCE_SIZE[variant],
                "media_contract": expected_media,
            }
            or sheet != EXPECTED_ALL81_SHEETS[variant]
        ):
            raise Exact5EvalError(f"independent per-variant evidence differs: {variant}")
    _strict_digest(value, "audit_digest", label="independent audit receipt")
    if value["audit_digest"] != INDEPENDENT_AUDIT_DIGEST:
        raise Exact5EvalError("independent audit digest differs")
    return dict(value)


def validate_asset_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "status",
        "launch_allowed",
        "independent_visual_audit_status",
        "manifest_path",
        "manifest_sha256",
        "manifest_size",
        "manifest_digest",
        "independent_audit_receipt_path",
        "independent_audit_receipt_sha256",
        "independent_audit_receipt_size",
        "independent_audit_receipt_digest",
        "iid",
        "sources",
        "source_rows_digest",
        "authority_digest",
    }
    sources = value.get("sources") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema_version") != ASSET_AUTHORITY_SCHEMA
        or value.get("status") != ASSET_AUTHORITY_STATUS
        or value.get("launch_allowed") is not True
        or value.get("independent_visual_audit_status") != "PASS_P0_0_P1_0"
        or value.get("iid") != IID
        or not isinstance(value.get("manifest_path"), str)
        or not Path(value["manifest_path"]).is_absolute()
        or value.get("manifest_sha256") != ASSET_MANIFEST_SHA256
        or value.get("manifest_size") != ASSET_MANIFEST_SIZE
        or value.get("manifest_digest") != ASSET_MANIFEST_DIGEST
        or not isinstance(value.get("independent_audit_receipt_path"), str)
        or not Path(value["independent_audit_receipt_path"]).is_absolute()
        or value.get("independent_audit_receipt_sha256")
        != INDEPENDENT_AUDIT_SHA256
        or value.get("independent_audit_receipt_size") != INDEPENDENT_AUDIT_SIZE
        or value.get("independent_audit_receipt_digest")
        != INDEPENDENT_AUDIT_DIGEST
        or not isinstance(sources, list)
        or [row.get("variant") for row in sources] != list(VARIANT_ORDER)
        or value.get("source_rows_digest") != object_sha256(sources)
    ):
        raise Exact5EvalError("asset authority approval/closure differs")
    observed_paths: set[str] = set()
    observed_hashes: set[str] = set()
    for row, variant in zip(sources, VARIANT_ORDER):
        path = row.get("path") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "variant",
                "path",
                "sha256",
                "size",
                "geometry",
                "treatment",
                "bone_present",
                "bone_position",
                "visual_audit_status",
            }
            or path is None
            or not isinstance(path, str)
            or not Path(path).is_absolute()
            or os.path.normpath(path) != path
            or SHA256_RE.fullmatch(row.get("sha256", "")) is None
            or row.get("sha256") != EXPECTED_SOURCE_SHA256[variant]
            or type(row.get("size")) is not int
            or row["size"] != EXPECTED_SOURCE_SIZE[variant]
            or row.get("geometry") != EXPECTED_VIDEO
            or row.get("treatment") != EXPECTED_TREATMENTS[variant]
            or row.get("visual_audit_status") != "PASS"
        ):
            raise Exact5EvalError(f"asset source row differs: {variant}")
        observed_paths.add(path)
        observed_hashes.add(row["sha256"])
    if len(observed_paths) != 5 or len(observed_hashes) != 5:
        raise Exact5EvalError("exact-five source paths/hashes are not distinct")
    if (
        sources[0].get("bone_present") is not True
        or sources[1].get("bone_present") is not True
        or sources[2].get("bone_present") is not False
        or sources[3].get("bone_present") is not True
        or sources[4].get("bone_present") is not True
        or sources[0].get("bone_position") != "source_original"
        or sources[1].get("bone_position") != "source_original"
        or sources[2].get("bone_position") != "absent"
        or sources[3].get("bone_position") != "translated_up150"
        or sources[4].get("bone_position") != "source_original"
    ):
        raise Exact5EvalError("asset causal semantics differ")
    manifest_raw, manifest_sha, manifest_size = stable_file(
        value["manifest_path"],
        expected_sha256=ASSET_MANIFEST_SHA256,
        return_bytes=True,
    )
    if manifest_raw is None:
        raise Exact5EvalError("stable asset manifest reader returned no bytes")
    try:
        manifest = json.loads(
            manifest_raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise Exact5EvalError("asset manifest is not strict JSON") from error
    unsigned_manifest = dict(manifest) if isinstance(manifest, dict) else {}
    claimed_manifest_digest = unsigned_manifest.pop("artifact_digest", None)
    output_rows = {
        row.get("path"): row
        for row in manifest.get("outputs", [])
        if isinstance(row, Mapping)
    } if isinstance(manifest, Mapping) else {}
    probe_rows = manifest.get("media_probes") if isinstance(manifest, Mapping) else None
    if (
        manifest_sha != ASSET_MANIFEST_SHA256
        or manifest_size != ASSET_MANIFEST_SIZE
        or not isinstance(manifest, dict)
        or manifest.get("schema_version") != ASSET_MANIFEST_SCHEMA
        or claimed_manifest_digest != ASSET_MANIFEST_DIGEST
        or object_sha256(unsigned_manifest) != ASSET_MANIFEST_DIGEST
        or manifest.get("iid") != IID
        or manifest.get("instruction") != INSTRUCTION
        or manifest.get("manual_visual_audit", {}).get("status")
        != "PASS_INPUT_ASSET_QA_ONLY"
        or manifest.get("claim_limits", {}).get("renderer_inference_performed")
        is not False
        or manifest.get("claim_limits", {}).get("training_performed") is not False
        or not isinstance(probe_rows, Mapping)
    ):
        raise Exact5EvalError("frozen asset manifest identity/audit differs")
    for row, variant in zip(sources, VARIANT_ORDER):
        output = output_rows.get(f"videos/{variant}.mp4")
        probe = probe_rows.get(variant)
        streams = probe.get("streams") if isinstance(probe, Mapping) else None
        stream = streams[0] if isinstance(streams, list) and len(streams) == 1 else None
        if (
            output
            != {
                "path": f"videos/{variant}.mp4",
                "sha256": EXPECTED_SOURCE_SHA256[variant],
                "size": EXPECTED_SOURCE_SIZE[variant],
            }
            or not isinstance(stream, Mapping)
            or stream.get("codec_name") != "h264"
            or stream.get("pix_fmt") != "yuv420p"
            or stream.get("width") != 704
            or stream.get("height") != 736
            or stream.get("r_frame_rate") != "25/1"
            or stream.get("avg_frame_rate") != "25/1"
            or stream.get("nb_frames") != "81"
            or stream.get("nb_read_frames") != "81"
            or Path(row["path"]).name != f"{variant}.mp4"
        ):
            raise Exact5EvalError(f"manifest video authority differs: {variant}")
    audit_raw, audit_sha, audit_size = stable_file(
        value["independent_audit_receipt_path"],
        expected_sha256=INDEPENDENT_AUDIT_SHA256,
        return_bytes=True,
    )
    if audit_raw is None:
        raise Exact5EvalError("stable independent audit reader returned no bytes")
    try:
        audit = json.loads(
            audit_raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise Exact5EvalError("independent audit receipt is not strict JSON") from error
    validate_independent_audit_receipt(
        audit, raw=audit_raw, sha256=audit_sha, size=audit_size
    )
    if audit["audit_digest"] != value["independent_audit_receipt_digest"]:
        raise Exact5EvalError("independent audit receipt digest binding differs")
    _strict_digest(value, "authority_digest", label="asset authority")
    return dict(value)


def validate_plan(
    plan: Mapping[str, Any], *, reopen_sources: bool = True,
    require_fresh_outputs: bool = True,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "experiment_id",
        "production_ready",
        "launch_allowed",
        "asset_authority",
        "checkpoint_manifest",
        "producer",
        "condition_contract",
        "arms",
        "task_count",
        "tasks",
        "claim_limits",
        "plan_digest",
    }
    if not isinstance(plan, Mapping) or set(plan) != fields:
        raise Exact5EvalError("plan root schema differs")
    _strict_digest(plan, "plan_digest", label="plan")
    authority = validate_asset_authority(plan.get("asset_authority", {}))
    checkpoint = plan.get("checkpoint_manifest")
    producer = plan.get("producer")
    tasks = plan.get("tasks")
    condition = plan.get("condition_contract")
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("experiment_id") != EXPERIMENT_ID
        or plan.get("production_ready") is not True
        or plan.get("launch_allowed") is not True
        or plan.get("arms") != ["full644"]
        or plan.get("task_count") != 5
        or not isinstance(checkpoint, Mapping)
        or set(checkpoint) != set(EXPECTED_CHECKPOINT) | {"path"}
        or any(checkpoint.get(key) != item for key, item in EXPECTED_CHECKPOINT.items())
        or not isinstance(checkpoint.get("path"), str)
        or not Path(checkpoint["path"]).is_absolute()
        or not isinstance(producer, Mapping)
        or set(producer)
        != set(EXPECTED_PRODUCER) | {"infer_lora_path", "ffprobe_path"}
        or any(producer.get(key) != item for key, item in EXPECTED_PRODUCER.items())
        or not isinstance(producer.get("infer_lora_path"), str)
        or not Path(producer["infer_lora_path"]).is_absolute()
        or not isinstance(producer.get("ffprobe_path"), str)
        or not Path(producer["ffprobe_path"]).is_absolute()
        or not isinstance(condition, Mapping)
        or condition
        != {
            "iid": IID,
            "instruction": INSTRUCTION,
            "instruction_sha256": INSTRUCTION_SHA256,
            "seed": SEED,
            "num_inference_steps": 40,
            "source_onset_policy": "none",
            "same_sampler_all_tasks": True,
            "same_model_capture_all_tasks_required": True,
            "codec_only_present_control_required": True,
        }
        or not isinstance(tasks, list)
        or [task.get("task_id") for task in tasks] != list(TASK_IDS)
        or plan.get("claim_limits")
        != {
            "exploratory_only": True,
            "scientific_claim_authorized": False,
            "formal_claim_authorized": False,
            "manual_blind_review_required": True,
        }
    ):
        raise Exact5EvalError("plan identity/condition closure differs")
    source_rows = {
        row["variant"]: row for row in authority["sources"]
    }
    output_parents: set[Path] = set()
    publication_leaves: set[Path] = set()
    for index, (task, variant) in enumerate(zip(tasks, VARIANT_ORDER)):
        source = source_rows[variant]
        output = task.get("output") if isinstance(task, Mapping) else None
        adapter = task.get("adapter") if isinstance(task, Mapping) else None
        video_raw = output.get("video_path") if isinstance(output, Mapping) else None
        receipt_raw = (
            output.get("receipt_path") if isinstance(output, Mapping) else None
        )
        video_path = Path(video_raw) if isinstance(video_raw, str) else Path("")
        receipt_path = (
            Path(receipt_raw) if isinstance(receipt_raw, str) else Path("")
        )
        if (
            not isinstance(task, Mapping)
            or set(task)
            != {
                "task_id",
                "case_index",
                "iid",
                "intervention_variant",
                "source_video",
                "source_video_sha256",
                "instruction",
                "instruction_sha256",
                "seed",
                "num_inference_steps",
                "source_onset_policy",
                "arm",
                "adapter",
                "output",
            }
            or task.get("task_id") != TASK_IDS[index]
            or task.get("case_index") != 1
            or task.get("iid") != IID
            or task.get("intervention_variant") != variant
            or task.get("source_video") != source["path"]
            or task.get("source_video_sha256") != source["sha256"]
            or task.get("instruction") != INSTRUCTION
            or task.get("instruction_sha256") != INSTRUCTION_SHA256
            or task.get("seed") != SEED
            or task.get("num_inference_steps") != 40
            or task.get("source_onset_policy") != "none"
            or task.get("arm") != "full644"
            or not isinstance(adapter, Mapping)
            or set(adapter)
            != {
                "checkpoint_root",
                "checkpoint_manifest",
                "adapter_model_sha256",
                "profile",
            }
            or adapter.get("profile")
            != "full644-r64-reference-dpo-preservation-one-pass-v1"
            or adapter.get("adapter_model_sha256")
            != EXPECTED_CHECKPOINT["adapter_model_sha256"]
            or adapter.get("checkpoint_manifest") != checkpoint
            or adapter.get("checkpoint_root")
            != str(Path(checkpoint["path"]).parent)
            or not isinstance(output, Mapping)
            or set(output) != {"video_path", "receipt_path", "create_only"}
            or output.get("create_only") is not True
            or not isinstance(video_raw, str)
            or not isinstance(receipt_raw, str)
            or not video_path.is_absolute()
            or not receipt_path.is_absolute()
            or os.path.normpath(video_raw) != video_raw
            or os.path.normpath(receipt_raw) != receipt_raw
            or video_path.name != f"{TASK_IDS[index]}.mp4"
            or receipt_path.name != f"{TASK_IDS[index]}.mp4.receipt.json"
            or receipt_path != video_path.with_name(video_path.name + ".receipt.json")
        ):
            raise Exact5EvalError(f"task closure differs: {variant}")
        output_parents.add(video_path.parent)
        publication_leaves.update((video_path, receipt_path))
        if reopen_sources:
            _, observed, size = stable_file(
                source["path"], expected_sha256=source["sha256"]
            )
            if observed != source["sha256"] or size != source["size"]:
                raise Exact5EvalError(f"source changed: {variant}")
        if require_fresh_outputs and (
            Path(output["video_path"]).exists()
            or Path(output["video_path"]).is_symlink()
            or Path(output["receipt_path"]).exists()
            or Path(output["receipt_path"]).is_symlink()
        ):
            raise Exact5EvalError(f"planned output is not fresh: {variant}")
    if len(output_parents) != 1:
        raise Exact5EvalError("exact-five tasks do not share one output root")
    if len(publication_leaves) != 10:
        raise Exact5EvalError("exact-five publication leaves are not distinct")
    output_root = next(iter(output_parents))
    if (
        not output_root.is_absolute()
        or os.path.normpath(str(output_root)) != str(output_root)
        or not output_root.is_dir()
        or output_root.is_symlink()
        or output_root.resolve(strict=True) != output_root
    ):
        raise Exact5EvalError("exact-five output root is not canonical")
    base_suffixes = (
        "-model-capture.json",
        "-model-pre-use.json",
        "-consumption-input.json",
        "-model-post-use.json",
        "-eval-consumption-chain.json",
    )
    adapter_suffixes = (
        "-adapter-capture.json",
        "-adapter-pre-use.json",
        "-adapter-post-use.json",
        "-adapter-final.json",
    )
    internal_leaves = {
        output_root / (f".matched-v2-{index:02d}-{task_id}" + suffix)
        for index, task_id in enumerate(TASK_IDS)
        for suffix in (*base_suffixes, *adapter_suffixes, ".log", "-runner-task.json")
    }
    if publication_leaves & internal_leaves:
        raise Exact5EvalError("publication leaves overlap internal artifacts")
    return dict(plan)


def load_plan(path_value: str | Path, expected_sha256: str) -> dict[str, Any]:
    raw, _, _ = stable_file(
        path_value, expected_sha256=expected_sha256, return_bytes=True
    )
    if raw is None:
        raise Exact5EvalError("stable plan reader returned no bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise Exact5EvalError("plan is not strict JSON") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise Exact5EvalError("plan is not canonical JSON plus LF")
    return validate_plan(value, reopen_sources=True)


def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def build_asset_authority(
    manifest_path: str | Path,
    asset_root: str | Path,
    independent_audit_receipt_path: str | Path,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve(strict=True)
    root = Path(asset_root).resolve(strict=True)
    if manifest != root / "manifest.json":
        raise Exact5EvalError("asset manifest/root adjacency differs")
    _, manifest_sha, manifest_size = stable_file(
        manifest, expected_sha256=ASSET_MANIFEST_SHA256
    )
    audit_path = Path(independent_audit_receipt_path).resolve(strict=True)
    audit_raw, audit_sha, audit_size = stable_file(
        audit_path, expected_sha256=INDEPENDENT_AUDIT_SHA256, return_bytes=True
    )
    if audit_raw is None:
        raise Exact5EvalError("stable independent audit reader returned no bytes")
    try:
        audit_value = json.loads(audit_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Exact5EvalError("independent audit receipt is not JSON") from error
    if not isinstance(audit_value, dict):
        raise Exact5EvalError("independent audit receipt root differs")
    validate_independent_audit_receipt(
        audit_value, raw=audit_raw, sha256=audit_sha, size=audit_size
    )
    audit_digest = audit_value["audit_digest"]
    semantics = {
        "exact_original": (True, "source_original"),
        "codec_only_present": (True, "source_original"),
        "bone_removed": (False, "absent"),
        "bone_translated_up150": (True, "translated_up150"),
        "sham_control_up150": (True, "source_original"),
    }
    sources: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        path = root / "videos" / f"{variant}.mp4"
        _, sha256, size = stable_file(
            path, expected_sha256=EXPECTED_SOURCE_SHA256[variant]
        )
        if size != EXPECTED_SOURCE_SIZE[variant]:
            raise Exact5EvalError(f"asset source size differs: {variant}")
        bone_present, bone_position = semantics[variant]
        sources.append(
            {
                "variant": variant,
                "path": str(path),
                "sha256": sha256,
                "size": size,
                "geometry": dict(EXPECTED_VIDEO),
                "treatment": EXPECTED_TREATMENTS[variant],
                "bone_present": bone_present,
                "bone_position": bone_position,
                "visual_audit_status": "PASS",
            }
        )
    authority: dict[str, Any] = {
        "schema_version": ASSET_AUTHORITY_SCHEMA,
        "status": ASSET_AUTHORITY_STATUS,
        "launch_allowed": True,
        "independent_visual_audit_status": "PASS_P0_0_P1_0",
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "manifest_size": manifest_size,
        "manifest_digest": ASSET_MANIFEST_DIGEST,
        "independent_audit_receipt_path": str(audit_path),
        "independent_audit_receipt_sha256": audit_sha,
        "independent_audit_receipt_size": audit_size,
        "independent_audit_receipt_digest": audit_digest,
        "iid": IID,
        "sources": sources,
        "source_rows_digest": object_sha256(sources),
    }
    authority["authority_digest"] = object_sha256(authority)
    return validate_asset_authority(authority)


def build_plan(
    *,
    asset_authority: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    producer: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    authority = validate_asset_authority(asset_authority)
    root = Path(output_root)
    if (
        not root.is_absolute()
        or os.path.normpath(str(root)) != str(root)
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
    ):
        raise Exact5EvalError("plan output root differs")
    checkpoint = dict(checkpoint_manifest)
    producer_value = dict(producer)
    source_by_variant = {
        row["variant"]: row for row in authority["sources"]
    }
    tasks: list[dict[str, Any]] = []
    for variant, task_id in zip(VARIANT_ORDER, TASK_IDS):
        source = source_by_variant[variant]
        video = root / f"{task_id}.mp4"
        tasks.append(
            {
                "task_id": task_id,
                "case_index": 1,
                "iid": IID,
                "intervention_variant": variant,
                "source_video": source["path"],
                "source_video_sha256": source["sha256"],
                "instruction": INSTRUCTION,
                "instruction_sha256": INSTRUCTION_SHA256,
                "seed": SEED,
                "num_inference_steps": 40,
                "source_onset_policy": "none",
                "arm": "full644",
                "adapter": {
                    "checkpoint_root": str(Path(checkpoint["path"]).parent),
                    "checkpoint_manifest": checkpoint,
                    "adapter_model_sha256": checkpoint.get(
                        "adapter_model_sha256"
                    ),
                    "profile": (
                        "full644-r64-reference-dpo-preservation-one-pass-v1"
                    ),
                },
                "output": {
                    "video_path": str(video),
                    "receipt_path": str(video.with_name(video.name + ".receipt.json")),
                    "create_only": True,
                },
            }
        )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "production_ready": True,
        "launch_allowed": True,
        "asset_authority": authority,
        "checkpoint_manifest": checkpoint,
        "producer": producer_value,
        "condition_contract": {
            "iid": IID,
            "instruction": INSTRUCTION,
            "instruction_sha256": INSTRUCTION_SHA256,
            "seed": SEED,
            "num_inference_steps": 40,
            "source_onset_policy": "none",
            "same_sampler_all_tasks": True,
            "same_model_capture_all_tasks_required": True,
            "codec_only_present_control_required": True,
        },
        "arms": ["full644"],
        "task_count": 5,
        "tasks": tasks,
        "claim_limits": {
            "exploratory_only": True,
            "scientific_claim_authorized": False,
            "formal_claim_authorized": False,
            "manual_blind_review_required": True,
        },
    }
    plan["plan_digest"] = object_sha256(plan)
    return validate_plan(plan, reopen_sources=True)


def verify_results(
    plan: Mapping[str, Any],
    *,
    frozen_v2: Any,
    publication_root_fd: int,
    ffprobe_authority: Mapping[str, Any],
    publication_authorities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validated = validate_plan(
        plan, reopen_sources=True, require_fresh_outputs=False
    )
    tasks = validated["tasks"]
    publication_root = Path(tasks[0]["output"]["video_path"]).parent
    if (
        set(publication_authorities) != set(TASK_IDS)
        or frozen_v2.validate_terminal_checkpoint_manifest(
            validated["checkpoint_manifest"]["path"],
            validated["checkpoint_manifest"]["sha256"],
        )
        != validated["checkpoint_manifest"]
    ):
        raise Exact5EvalError("result retained/checkpoint authority differs")
    verified_with_receipts = [
        frozen_v2.verify_arm(
            task,
            validated["producer"],
            publication_root=publication_root,
            publication_root_fd=publication_root_fd,
            ffprobe_authority=ffprobe_authority,
            publication_authority=publication_authorities[task["task_id"]],
        )
        for task in tasks
    ]
    sampler_rows = {
        canonical_json_bytes(row["receipt"].get("sampling"))
        for row in verified_with_receipts
    }
    prompt_rows = {
        canonical_json_bytes(row["receipt"].get("prompt_contract"))
        for row in verified_with_receipts
    }
    model_captures = {
        row["receipt"].get("model_consumption", {}).get("model_capture_digest")
        for row in verified_with_receipts
    }
    if len(sampler_rows) != 1 or len(prompt_rows) != 1 or len(model_captures) != 1:
        raise Exact5EvalError("exact-five matched receipt coordinates differ")
    results: list[dict[str, Any]] = []
    for row in verified_with_receipts:
        clean = dict(row)
        clean.pop("receipt")
        results.append(clean)
    if results[0].get("output_sha256") != REFERENCE_EXACT_ORIGINAL_R64_OUTPUT_SHA256:
        raise Exact5EvalError(
            "exact_original deterministic parity failed against frozen case01 R64"
        )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "EXACT5_COMPLETE_AWAITING_BLIND_REVIEW",
        "campaign_mode": CAMPAIGN,
        "plan_schema_version": validated["schema_version"],
        "plan_digest": validated["plan_digest"],
        "task_count": 5,
        "task_ids": list(TASK_IDS),
        "variant_order": list(VARIANT_ORDER),
        "all_exact5_tasks_verified_no_cherry_pick": True,
        "same_sampler_all_tasks": True,
        "same_prompt_contract_all_tasks": True,
        "same_model_capture_all_tasks": True,
        "deterministic_reference_parity": {
            "policy": "HARD_FAIL",
            "variant": "exact_original",
            "reference_output_sha256": REFERENCE_EXACT_ORIGINAL_R64_OUTPUT_SHA256,
            "observed_output_sha256": results[0]["output_sha256"],
            "status": "PASS",
            "kept_separate_from_intervention_effect_interpretation": True,
        },
        "codec_only_control_interpretation": (
            "isolates matched transcode/container effects from source-object treatment"
        ),
        "retained_publication_root_fd_replayed": True,
        "retained_ffprobe_executable_fd_replayed": True,
        "retained_publication_leaf_fds_replayed": True,
        "manual_blind_review_required": True,
        "formal_full16_report": False,
        "results": results,
        "claim_limits": dict(validated["claim_limits"]),
    }
    report["report_digest"] = object_sha256(report)
    return report

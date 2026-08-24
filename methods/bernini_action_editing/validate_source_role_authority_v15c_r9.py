#!/usr/bin/env python3
"""Exact local authority validation for the v15c-r9 four-role observer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import source_role_authority_v15c_r9 as core
except ImportError:  # pragma: no cover
    import source_role_authority_v15c_r9 as core


AUTHORITY_SCHEMA = "bernini-e00-source-four-role-authority-v15c-r9-local"
AUTHORITY_RELATIVE_PATH = (
    "methods/bernini_action_editing/assets/"
    "e00_source_four_role_authority_v15c_r9.json"
)
# Updated only by the local release builder after the authority JSON changes.
EXPECTED_AUTHORITY_RAW_SHA256 = (
    "0fced569a60ae9ed82b4fd355b03a12faedfac6ff60aa74b6f0a6aa8c46da776"
)
EXPECTED_AUTHORITY_CANONICAL_SHA256 = (
    "15e5bf69c64b79b20ce2b06af96ba53883e6ebec00342a0bb231b272d2aff855"
)
ROLE_ASSET_RELATIVE_PATH = (
    "methods/bernini_action_editing/assets/"
    "interaction_e00_source_instance_role_token_spans_v15b.json"
)
BASE_SPEC_RELATIVE_PATH = (
    "methods/bernini_action_editing/assets/"
    "e00_source_sam2_proposal_role_probe_v15c_r6.json"
)
BASE_RELEASE_RELATIVE_PATH = (
    "methods/bernini_action_editing/assets/"
    "e00_source_sam2_proposal_role_probe_v15c_r8_release.json"
)
AUTHORITY_KEYS = (
    "schema_version",
    "tag",
    "status",
    "source",
    "r8_local_replay_base",
    "token_source_authority",
    "r6_affinity_authority",
    "role_assignment",
    "thresholds",
    "ownership_partition",
    "overlay_plan",
    "claim_limits",
)


class ValidateSourceRoleAuthorityV15CR9Error(RuntimeError):
    """The local authority or a byte-pinned dependency differs."""


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidateSourceRoleAuthorityV15CR9Error("duplicate JSON key")
        result[key] = value
    return result


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValidateSourceRoleAuthorityV15CR9Error("JSON authority is not regular")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValidateSourceRoleAuthorityV15CR9Error("non-finite JSON")
            ),
        )
    except ValidateSourceRoleAuthorityV15CR9Error:
        raise
    except Exception as error:
        raise ValidateSourceRoleAuthorityV15CR9Error("JSON authority differs") from error
    if type(value) is not dict:
        raise ValidateSourceRoleAuthorityV15CR9Error("JSON authority is not an object")
    return value


def require_exact_keys(value: Any, keys: Sequence[str], label: str) -> None:
    if type(value) is not dict or set(value) != set(keys):
        raise ValidateSourceRoleAuthorityV15CR9Error(f"{label} keys differ")


def _beneath(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValidateSourceRoleAuthorityV15CR9Error("relative authority path differs")
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / candidate).resolve(strict=True)
    if resolved_root not in resolved.parents or not resolved.is_file() or resolved.is_symlink():
        raise ValidateSourceRoleAuthorityV15CR9Error("authority member escapes root")
    return resolved


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(core.canonical_bytes(value)).hexdigest()


def thresholds_from_authority(authority: Mapping[str, Any]) -> core.RoleThresholdsV15CR9:
    values = authority.get("thresholds")
    expected = tuple(core.RoleThresholdsV15CR9.__dataclass_fields__)
    require_exact_keys(values, expected, "thresholds")
    try:
        result = core.RoleThresholdsV15CR9(**values)
    except (TypeError, ValueError, core.SourceRoleAuthorityV15CR9Error) as error:
        raise ValidateSourceRoleAuthorityV15CR9Error("threshold values differ") from error
    if {name: getattr(result, name) for name in expected} != values:
        raise ValidateSourceRoleAuthorityV15CR9Error("threshold round trip differs")
    return result


def validate_authority(
    *, root: Path, authority_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    root = root.resolve(strict=True)
    authority_path = authority_path.resolve(strict=True)
    expected_path = _beneath(root, AUTHORITY_RELATIVE_PATH)
    if authority_path != expected_path:
        raise ValidateSourceRoleAuthorityV15CR9Error("authority path differs")
    authority = read_json(authority_path)
    require_exact_keys(authority, AUTHORITY_KEYS, "authority")
    if (
        core.file_sha256(authority_path) != EXPECTED_AUTHORITY_RAW_SHA256
        or _canonical_sha(authority) != EXPECTED_AUTHORITY_CANONICAL_SHA256
        or authority.get("schema_version") != AUTHORITY_SCHEMA
        or authority.get("tag") != "v15c-r9-local-source-four-role-authority"
        or authority.get("status")
        != "COMMON_NULL_DIAGNOSTIC_ONLY_FOUR_ROLE_JOINT_FWER_UNCERTIFIED"
    ):
        raise ValidateSourceRoleAuthorityV15CR9Error("authority byte pin differs")

    source = authority["source"]
    require_exact_keys(
        source,
        ("iid", "video_sha256", "frame_count", "fps", "width", "height", "phase_frames"),
        "source authority",
    )
    if source != {
        "iid": "2f183dbf9e7a4d2e",
        "video_sha256": "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de",
        "frame_count": 81,
        "fps": 25.0,
        "width": 704,
        "height": 1056,
        "phase_frames": list(core.PHASE_FRAMES),
    }:
        raise ValidateSourceRoleAuthorityV15CR9Error("source authority differs")

    base = authority["r8_local_replay_base"]
    require_exact_keys(
        base,
        (
            "spec_path", "spec_raw_sha256", "spec_canonical_sha256",
            "release_path", "release_file_sha256", "evidence_semantics",
        ),
        "r8 base",
    )
    if base["spec_path"] != BASE_SPEC_RELATIVE_PATH or base["release_path"] != BASE_RELEASE_RELATIVE_PATH:
        raise ValidateSourceRoleAuthorityV15CR9Error("r8 base path differs")
    base_spec_path = _beneath(root, base["spec_path"])
    base_release_path = _beneath(root, base["release_path"])
    base_spec = read_json(base_spec_path)
    if (
        core.file_sha256(base_spec_path) != base["spec_raw_sha256"]
        or _canonical_sha(base_spec) != base["spec_canonical_sha256"]
        or core.file_sha256(base_release_path) != base["release_file_sha256"]
        or base["evidence_semantics"] != "r8_LOCAL_tensor_byte_replay_only_remote_worker_unverified"
        or base_spec.get("source", {}).get("sha256") != source["video_sha256"]
    ):
        raise ValidateSourceRoleAuthorityV15CR9Error("r8 base bytes differ")

    token = authority["token_source_authority"]
    require_exact_keys(
        token,
        (
            "asset_path", "asset_file_sha256", "asset_canonical_sha256",
            "asset_internal_sha256", "source_text_provenance_sha256",
            "role_event_sha256", "model_text_sha256", "tokenizer_tree_sha256",
            "role_channel_binding",
        ),
        "token source authority",
    )
    if token["asset_path"] != ROLE_ASSET_RELATIVE_PATH:
        raise ValidateSourceRoleAuthorityV15CR9Error("role asset path differs")
    role_asset_path = _beneath(root, token["asset_path"])
    role_asset = read_json(role_asset_path)
    event = role_asset.get("event")
    if (
        core.file_sha256(role_asset_path) != token["asset_file_sha256"]
        or _canonical_sha(role_asset) != token["asset_canonical_sha256"]
        or role_asset.get("asset_sha256") != token["asset_internal_sha256"]
        or role_asset.get("source_video_sha256") != source["video_sha256"]
        or type(event) is not dict
        or event.get("event_sha256") != token["role_event_sha256"]
        or event.get("model_text_sha256") != token["model_text_sha256"]
        or event.get("tokenizer_tree_sha256") != token["tokenizer_tree_sha256"]
        or base_spec.get("r6", {}).get("source_text_provenance_sha256") != token["source_text_provenance_sha256"]
        or base_spec.get("r6", {}).get("role_asset_sha256") != token["asset_internal_sha256"]
        or base_spec.get("r6", {}).get("role_event_sha256") != token["role_event_sha256"]
    ):
        raise ValidateSourceRoleAuthorityV15CR9Error("token asset authority differs")
    asset_rows = event.get("roles")
    bindings = token.get("role_channel_binding")
    if type(asset_rows) is not list or type(bindings) is not list or len(bindings) != len(core.ROLE_NAMES):
        raise ValidateSourceRoleAuthorityV15CR9Error("role binding registry differs")
    asset_by_role = {row.get("role"): row for row in asset_rows if type(row) is dict}
    expected_binding_keys = {
        "r9_role", "r6_role", "r6_channel_index", "kind", "substring",
        "char_start", "char_end", "token_start", "token_end", "token_ids",
        "token_ids_sha256", "span_sha256",
    }
    for index, binding in enumerate(bindings):
        if type(binding) is not dict or set(binding) != expected_binding_keys:
            raise ValidateSourceRoleAuthorityV15CR9Error("role binding fields differ")
        r9_role = core.ROLE_NAMES[index]
        r6_role = "agent" if r9_role == "human_agent" else r9_role
        asset_row = asset_by_role.get(r6_role)
        if (
            binding["r9_role"] != r9_role
            or binding["r6_role"] != r6_role
            or binding["r6_channel_index"] != core.R6_ROLE_INDEX[r9_role]
            or type(asset_row) is not dict
            or any(binding[key] != asset_row.get(key) for key in expected_binding_keys - {"r9_role", "r6_role", "r6_channel_index"})
        ):
            raise ValidateSourceRoleAuthorityV15CR9Error("role/token span binding differs")

    r6 = authority["r6_affinity_authority"]
    require_exact_keys(
        r6,
        (
            "probe_receipt_file_sha256", "affinity_tensor_file_sha256",
            "null_registry_sha256", "null_count", "null_index_alignment",
            "null_tensor_shape", "four_role_joint_null_axis_available",
            "common_null_broadcast_for_certification", "fwer_status",
            "blocks", "full_r6_role_names", "raw_role_null_and_shuffled_affinity_only",
            "calibration_masks_consumed",
        ),
        "r6 affinity authority",
    )
    if (
        r6["probe_receipt_file_sha256"] != base_spec["r6"]["probe_receipt_file_sha256"]
        or r6["affinity_tensor_file_sha256"] != base_spec["r6"]["affinity_tensor_file_sha256"]
        or r6["null_registry_sha256"] != base_spec["r6"]["null_registry_sha256"]
        or r6["null_count"] != core.NULL_COUNT
        or r6["blocks"] != list(core.BLOCK_INDICES)
        or r6["full_r6_role_names"] != list(core.FULL_R6_ROLE_NAMES)
        or r6["null_index_alignment"]
        != "same_preregistered_null_registry_index_j_across_blocks_phases_and_proposals_only"
        or r6["null_tensor_shape"]
        != [len(core.BLOCK_INDICES), core.NULL_COUNT, core.PHASE_COUNT,
            core.GRID_HEIGHT, core.GRID_WIDTH]
        or r6["four_role_joint_null_axis_available"] is not False
        or r6["common_null_broadcast_for_certification"] is not False
        or r6["fwer_status"] != "FOUR_ROLE_JOINT_FWER_UNCERTIFIED"
        or r6["raw_role_null_and_shuffled_affinity_only"] is not True
        or r6["calibration_masks_consumed"] is not False
    ):
        raise ValidateSourceRoleAuthorityV15CR9Error("r6/null alignment differs")

    roles = authority["role_assignment"]
    if (
        type(roles) is not dict
        or roles.get("role_names") != list(core.ROLE_NAMES)
        or roles.get("vessel_role_names") != list(core.VESSEL_ROLE_NAMES)
        or roles.get("global_multiple_comparison_control")
        != "requires_exact_role_indexed_null_tensor_[block,role,64,phase,height,width]_then_max_over_all_geometry_valid_proposals_and_four_roles_else_no_go"
        or roles.get("global_familywise_alpha") != 0.05
        or roles.get("future_joint_null_minimum_attainable_p") != 1.0 / 65.0
        or roles.get("current_r6_common_null_scope")
        != "COMMON_NULL_DIAGNOSTIC_ONLY"
        or roles.get("four_role_joint_fwer_certified") is not False
        or roles.get("mechanical_candidate_must_remain_false") is not True
        or roles.get("vessel_three_role_bonferroni_additional_gate") is not True
        or roles.get("forced_assignment") is not False
        or roles.get("roi_or_manual_box_consumed") is not False
    ):
        raise ValidateSourceRoleAuthorityV15CR9Error("four-role statistical authority differs")
    thresholds_from_authority(authority)

    ownership = authority["ownership_partition"]
    claims = authority["claim_limits"]
    if (
        type(ownership) is not dict
        or ownership.get("external_signature_or_tee_claimed") is not False
        or ownership.get("four_role_final_ownership_pairwise_exclusive") is not True
        or ownership.get("raw_overlap_evidence_preserved") is not True
        or ownership.get("independent_human_vessel_contact_relation_masks") is not True
        or ownership.get("morphological_fill_close_dilate_or_repair") is not False
        or type(claims) is not dict
        or claims != {
            "local_schema_replay_only": True,
            "mechanical_candidate_qualified": False,
            "remote_worker_execution_verified": False,
            "observer_execution_authorized": False,
            "localization_semantically_certified": False,
            "scientific_claim_authorized": False,
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
            "renderer_forward_calls": 0,
            "optimizer_updates": 0,
        }
    ):
        raise ValidateSourceRoleAuthorityV15CR9Error("ownership/claim boundary differs")
    return authority, base_spec, role_asset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    args = parser.parse_args()
    authority, _base, _roles = validate_authority(root=args.root, authority_path=args.authority)
    receipt = {
        "schema_version": "bernini-source-four-role-authority-validation-v15c-r9-local",
        "status": "COMMON_NULL_DIAGNOSTIC_ONLY_FOUR_ROLE_JOINT_FWER_UNCERTIFIED",
        "authority_raw_sha256": core.file_sha256(args.authority),
        "authority_canonical_sha256": core.object_sha256(authority),
        "observer_execution_authorized": False,
        "localization_semantically_certified": False,
        "scientific_claim_authorized": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    receipt["receipt_sha256"] = core.object_sha256(receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

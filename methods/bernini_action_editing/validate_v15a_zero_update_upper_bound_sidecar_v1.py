#!/usr/bin/env python3
"""Fail-closed audit for the v15a max-strength clean-noised route probe.

The inference program already writes its native receipt.  This validator binds
that runtime receipt to the immutable dfix2 controller and emits a second,
small audit sidecar.  The latter makes the otherwise implicit dynamic/static
data flow explicit: both anchor calls use the action caption, the same model
timestep and the exact same candidate noise; only the clean video term differs
(dynamic anchor versus its repeated phase-0 state).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "bernini-v15a-r3-max-strength-clean-noised-route-probe-audit-v1"
SOURCE_SCHEMA = "bernini-pure-t2v-anchor-sga-anc-event-canary-v47"
SOURCE_SHA256 = "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de"
ANCHOR_SHA256 = "e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa"
GAUSSIAN_FILE_SHA256 = "e4828f4594caf65b45a937a271847dcf621a75e4f6862269a28216759ff087b7"
GAUSSIAN_RAW_SHA256 = "38438c769db8a539ed84902e8493d66ab257144a97be9f4f57dfccb80880f832"
CONTROLLER_SHA256 = "1427a4908e0a4239e95a353d3406c41cb77fdb7f0be81727126a2cfd23f1f3ad"
QK_TRANSPORT_SHA256 = "37941e30853b16fa242a7c91940620069f87a1a975d2ecf610f3cde800557a99"
INFER_SHA256 = "dd3558a4c38c5541ba6b7ad455ac599f43eb48b1b56f207a07776c9e1819145f"
DECODE_SHA256 = "4ed2f22df876613ecfc720a662a48f8e028eb89fe9778e491bc962a4f8f68ab1"
BRIDGE_SHA256 = "0365aacb88d976fcdc1f9bf169384f5d336bd4abe2f3c899ab6bdd502a580034"
DEPLOYMENT_VALIDATOR_SHA256 = "443e8a4966485edfa5abd375e921d3bc1e314bb16d2299659a537b271f7530ba"
ARCHIVE_SHA256 = "88c47356a83368ccdb0649718c8c06fa8c7baf368e7c97df8a48d9b93ab55bd9"
REVISION_SHA256 = "be15562d0e75fa953a9464447d187265ef5a8e36b5541a1f73dc13b0f41a98a3"
CONTENT_SHA256 = "ebb8d86545c9b9f476d3ee3b1855ccd310cad321c8fd2f93a8540e807e13a1af"
MARKER_SHA256 = "03bad1ae839a1135a1cca08990cd2a68788f4b44485abf90271453cbf3f9b969"
REVISION_VALUE = "online-anchor-targetowned-qk-v14r3-gradient-geometry-decodefix2-20260820"
AUTHORING_MANIFEST_SHA256 = "767aa8e0502f247c3ab576db4c40a132344295e54aef4170e728bc3ff71cafc5"
EXPERIMENT_TAG = "v15a_zero_update_dynamic_static_e00_maxstrength_routeprobe_r3_20260820"
OUTER_SEED = 2027
OUTPUT_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/"
    "online_anchor_attention_training_v1/dynaedit_maxstrength_routeprobe_v15a_r3/"
    + EXPERIMENT_TAG
)
PROMPT_SHA256 = {
    "action_mv2v_sha256": "c535bd8ebf9b3ff2de08d15f1fbf0327f8610a42f4c6ff07754d5ccf4d747de2",
    "source_noop_mv2v_sha256": "67a74d3aafbbca1a42598a431bef3b66f9b24789c066da08d4097a7eafdc223b",
    "anchor_t2v_sha256": "8d33aa4b3bf0459cf7dc850f6a65c1d583e83614e51e9783d2754939a9d2a7f6",
    "anchor_noop_t2v_sha256": "033b947d7d83223ae61c84adb08b2d9f81001834e451a79dee79e9f4c49de981",
    "source_t2v_sha256": "c424fda6ec36b5c78a856f8abcba1db217c407f2920317832cc0e26095274451",
    "target_t2v_sha256": "e7e4e5dd2b0cae49950f6cf42829a49d0c3995cfc69f1b80ca949d74d9191388",
    "negative_sha256": "ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e",
}

TRANSPORTS = (
    "self_target_owned_temporal_kernel_attn_output_v14r2",
    "self_target_owned_activity_kernel25_attn_output_v14r2",
)
ARM_ROLES = (
    "route_on_temporal",
    "route_on_activity25",
    "route_off_plain_frozen",
)
COMPARISON_GROUP = "e00-v15a-r3-plain-frozen-max-strength-clean-noised-route-probe-v1"
ARM_LABELS = {
    "route_on_temporal": "E00_V15AR3_FROZEN_ZEROUPDATE_DYNSTATIC_TARGETOWNED_TEMPORAL_ROUTEON_K40_A100.mp4",
    "route_on_activity25": "E00_V15AR3_FROZEN_ZEROUPDATE_DYNSTATIC_TARGETOWNED_ACTIVITY25_ROUTEON_K40_A100.mp4",
    "route_off_plain_frozen": "E00_V15AR3_FROZEN_ZEROUPDATE_DYNSTATIC_MATCHED_PLAIN_FROZEN_ROUTEOFF_K0_A100.mp4",
}
BLOCKS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29]
FORBIDDEN_CACHE_FIELDS = [
    "value", "hidden_state", "attention_output", "rgb", "latent",
    "absolute_spatial_coordinate",
]
EXPECTED_CELLS = 3 * 5 + 37
EXPECTED_CAPTURES = 2 * EXPECTED_CELLS * len(BLOCKS)
EXPECTED_REPLAYS = 2 * EXPECTED_CAPTURES
HEX = set("0123456789abcdef")


class V15AValidationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise V15AValidationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} is not an object")
    return value


def _get(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = _mapping(current, path).get(part)
    return current


def _eq(value: Mapping[str, Any], path: str, expected: Any) -> None:
    actual = _get(value, path)
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        _fail(f"{path} differs: {actual!r} != {expected!r}")


def _plain_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        _fail(f"{label} is not a plain file: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        _fail(f"{label} is not a SHA-256")
    return value


def _load(path: Path, label: str) -> Mapping[str, Any]:
    _plain_file(path, label)
    try:
        return _mapping(json.loads(path.read_text(encoding="ascii")), label)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V15AValidationError(f"{label} is unreadable") from error


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    video: Path,
    transport: str,
    arm_role: str = "route_on_temporal",
    enforce_output_binding: bool = True,
) -> dict[str, Any]:
    if transport not in TRANSPORTS:
        _fail("unsupported v15a transport")
    if arm_role not in ARM_ROLES:
        _fail("unsupported v15a arm role")
    route_off = arm_role == "route_off_plain_frozen"
    if arm_role == "route_on_temporal" and transport != TRANSPORTS[0]:
        _fail("temporal route-on arm has the wrong transport")
    if arm_role == "route_on_activity25" and transport != TRANSPORTS[1]:
        _fail("activity25 route-on arm has the wrong transport")
    if route_off and transport != TRANSPORTS[0]:
        _fail("matched route-off must retain the temporal transport identity")
    transport_steps = 0 if route_off else 40
    for path, expected in (
        ("schema_version", SOURCE_SCHEMA), ("complete", True),
        ("training_performed", False), ("optimization_steps", 0),
        ("loaded_trained_attention_checkpoint", False),
        ("trained_attention_checkpoint", None),
        ("source.sha256", SOURCE_SHA256),
        ("source.role", "clean_edit_state_identity_appearance_scene_authority"),
        ("pure_t2v_anchor.sha256", ANCHOR_SHA256),
        ("pure_t2v_anchor.active_solver_steps", transport_steps),
        ("pure_t2v_anchor.model_forward_at_every_active_solver_step_and_candidate", not route_off),
        ("anchor_generation_initial_gaussian.file_sha256", GAUSSIAN_FILE_SHA256),
        ("anchor_generation_initial_gaussian.tensor_identity.raw_storage_sha256", GAUSSIAN_RAW_SHA256),
        ("anchor_generation_initial_gaussian.role", "dynaedit_step0_candidate0_native_generation_noise"),
    ):
        _eq(receipt, path, expected)

    freeze_before = _mapping(receipt.get("freeze_before"), "freeze_before")
    freeze_after = _mapping(receipt.get("freeze_after"), "freeze_after")
    if freeze_before != freeze_after:
        _fail("frozen-base certificate changed during inference")
    for path, expected in (
        ("base_frozen", True), ("trainable_parameter_tensors", 0),
        ("trainable_parameter_elements", 0), ("lora_module_count", 0),
        ("adapter_modules_absent", True),
    ):
        _eq(freeze_before, path, expected)

    causal = _mapping(receipt.get("causal_control"), "causal_control")
    for path, expected in (
        ("enabled", False), ("kind", None), ("explicit_opt_in", False),
        ("trained_adapter_loaded", False),
        ("adapter_enabled_for_target_source_calls", False),
        ("anchor_injection_enabled", not route_off),
        ("transport_steps", transport_steps),
    ):
        _eq(causal, path, expected)

    mechanism = _mapping(receipt.get("mechanism"), "mechanism")
    for path, expected in (
        ("arm", "AQK_SGA5"), ("transport", transport),
        ("transport_strength", 1.0), ("transport_steps", transport_steps),
        ("initial_phase_clamp", True), ("field_guidance", "raw_cfg"),
        ("field_model", "first_phase_caption_i2v"),
        ("source_cfg_scale", 4.5), ("target_cfg_scale", 4.5),
        ("early_candidate_count", 5),
        ("initial_noise_proposal_mode", "anchor_candidate0"),
        ("anchor_state_mode", "clean_noised"),
        ("anchor_cfg_scope", "shared"),
        ("anchor_contrast_mode", "dynamic_static_same_caption"),
        ("anchor_sigma_cap", 1.0), ("preservation_mode", "none"),
        ("preservation_residual_fraction", 0.0),
        ("preservation_object_identity_strength", 0.0),
        ("anchor_candidate_mode", "single_shared"), ("anchor_bank_size", 1),
        ("sga_score_mode", "global_source_cosine"),
        ("anchor_spatial_alignment", "none"), ("selected_blocks", BLOCKS),
        ("event01_forced_role_proposal_index", -1),
        ("pure_t2v_anchor_online_block_transport_enabled", not route_off),
        ("pure_t2v_anchor_online_velocity_transport_enabled", False),
        ("pure_t2v_anchor_values_or_pixels_copied_to_output", False),
    ):
        _eq(mechanism, path, expected)

    audit = _mapping(mechanism.get("decode_audit_contract"), "decode_audit_contract")
    for path, expected in (
        ("transport_steps", transport_steps), ("anchor_state_mode", "clean_noised"),
        ("anchor_cfg_scope", "shared"), ("source_cfg_scale", 4.5),
        ("target_cfg_scale", 4.5), ("source_and_target_cfg_equal", True),
        ("pure_t2v_teacher_adapter_policy", "plain_frozen_base"),
        ("target_source_editor_adapter_policy", "plain_frozen_base"),
        ("trained_route_off_control_explicitly_allowed", False),
        ("same_checkpoint_route_off_causal_control", False),
        ("anchor_injection_enabled", not route_off),
    ):
        _eq(audit, path, expected)

    prompts = _mapping(receipt.get("prompts"), "prompts")
    for name, expected in PROMPT_SHA256.items():
        if _hex(prompts.get(name), name) != expected:
            _fail(f"{name} differs from the pinned E00 prompt")
    action_prompt_sha = PROMPT_SHA256["anchor_t2v_sha256"]

    trace = _mapping(mechanism.get("trace"), "trace")
    expected_activity = transport == TRANSPORTS[1] and not route_off
    expected_cells = 0 if route_off else EXPECTED_CELLS
    expected_captures = 0 if route_off else EXPECTED_CAPTURES
    expected_replays = 0 if route_off else EXPECTED_REPLAYS
    for path, expected in (
        ("anchor_state_mode", "clean_noised"),
        ("anchor_contrast_mode", "dynamic_static_same_caption"),
        ("anchor_reference_is_static_phase0_video", True),
        ("anchor_initial_gaussian_used_at_step0_candidate0", True),
        ("anchor_model_forwards", 2 * expected_cells),
        ("anchor_candidate_cells", expected_cells),
        ("target_owned_qk_route_v14r2", not route_off),
        ("anchor_donor_cached_fields", None if route_off else ["query", "key"]),
        ("anchor_donor_value_hidden_output_or_coordinate_used", None if route_off else False),
        ("anchor_to_target_appearance_correspondence_used", None if route_off else False),
        ("anchor_temporal_attention_kernel_contrast", not route_off),
        ("anchor_temporal_kernel_applied_to_target_value_only", not route_off),
        ("anchor_target_activity_gated_hard_kernel", expected_activity),
        ("anchor_route_shared_by_target_negative_and_condition", True),
        ("anchor_route_target_conditional_only", False),
        ("initial_latent_phase_clamped_after_every_update", True),
        ("anchor_value_stream_copied", False),
        ("source_value_stream_retained", True),
        ("anchor_present_in_every_active_target_candidate", not route_off),
        ("anchor_present_after_active_interval", False),
        ("anchor_action_reward_used_for_sga", False),
        ("sga_weights_forced_to_anchor_candidate0", False),
        ("candidate_counts", [5, 5, 5] + [1] * 37),
    ):
        _eq(trace, path, expected)
    schedule = trace.get("anchor_active_schedule")
    if not isinstance(schedule, list) or len(schedule) != transport_steps:
        _fail("active anchor schedule length differs")
    for index, row_value in enumerate(schedule):
        row = _mapping(row_value, "anchor schedule row")
        if (
            row.get("step_index") != index
            or row.get("candidate_count") != (5 if index < 3 else 1)
            or row.get("anchor_timestep") != row.get("outer_timestep")
            or row.get("anchor_sigma") != row.get("outer_sigma")
            or row.get("cap_applied") is not False
        ):
            _fail(f"anchor schedule row {index} differs")

    cache = _mapping(trace.get("attention_cache"), "attention_cache")
    for path, expected in (
        ("method", "bernini-online-pure-t2v-anchor-qk-transport-v3"),
        ("capture_count", expected_captures),
        ("qk_only_capture_count", expected_captures),
        ("replay_count", expected_replays),
        ("qk_only_replay_count", expected_replays),
        ("pending_entries", 0), ("qk_only_cached_fields", ["query", "key"]),
        ("qk_only_forbidden_cached_fields", FORBIDDEN_CACHE_FIELDS),
        ("selected_block_indices", BLOCKS),
    ):
        _eq(cache, path, expected)

    if enforce_output_binding and video != OUTPUT_ROOT / ARM_LABELS[arm_role]:
        _fail("v15a output root or exact arm label differs")
    if str(video) != _get(receipt, "output.path"):
        _fail("output path differs")
    for path, expected in (("output.frames", 81), ("output.fps", 25)):
        _eq(receipt, path, expected)
    _plain_file(video, "video")
    video_sha = _sha256(video)
    if _get(receipt, "output.sha256") != video_sha:
        _fail("video SHA-256 differs")

    return {
        "action_prompt_sha256": action_prompt_sha,
        "static_prompt_sha256": action_prompt_sha,
        "same_caption": True if not route_off else None,
        "same_model_timestep": True if not route_off else None,
        "same_exact_candidate_noise": True if not route_off else None,
        "contrast_pair_executed": not route_off,
        "route_injection_off": route_off,
        "state_pair": (
            "route_off_no_dynamic_static_pair_executed"
            if route_off
            else "clean_noised_dynamic_anchor_vs_clean_noised_repeated_phase0"
        ),
        "same_noise_implementation": (
            None
            if route_off
            else "one_candidate_noise_tensor_reused_by_dynamic_and_static_clean_noised_states"
        ),
        "official_anchor_gaussian_bound_to_outer_step0_candidate0": True,
        "instruction_prompt_sha256": PROMPT_SHA256["action_mv2v_sha256"],
        "source_noop_prompt_sha256": PROMPT_SHA256["source_noop_mv2v_sha256"],
        "source_caption_prompt_sha256": PROMPT_SHA256["source_t2v_sha256"],
        "target_caption_prompt_sha256": PROMPT_SHA256["target_t2v_sha256"],
        "anchor_noop_prompt_sha256": PROMPT_SHA256["anchor_noop_t2v_sha256"],
        "negative_prompt_sha256": PROMPT_SHA256["negative_sha256"],
        "outer_seed": OUTER_SEED,
        "seed_proof": "immutable_controller_default_and_validation_bound_by_sha256",
        "proof_scope": "immutable_controller_dataflow_plus_runtime_trace",
        "video_sha256": video_sha,
    }


def _probe(video: Path) -> None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or streams[0].get("nb_read_frames") != "81" or streams[0].get("avg_frame_rate") != "25/1":
        _fail("ffprobe does not certify exact 81 frames at 25 fps")


def validate_authority(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "deployment_marker": (Path(args.deployment_marker), MARKER_SHA256),
        "archive": (Path(args.archive), ARCHIVE_SHA256),
        "revision": (Path(args.revision), REVISION_SHA256),
        "content_manifest": (Path(args.content_manifest), CONTENT_SHA256),
        "controller": (Path(args.controller), CONTROLLER_SHA256),
        "qk_transport": (Path(args.qk_transport), QK_TRANSPORT_SHA256),
        "infer": (Path(args.infer), INFER_SHA256),
        "decode": (Path(args.decode), DECODE_SHA256),
        "bridge": (Path(args.bridge), BRIDGE_SHA256),
        "deployment_validator": (
            Path(args.deployment_validator), DEPLOYMENT_VALIDATOR_SHA256
        ),
        "authoring_manifest": (
            Path(args.authoring_manifest), AUTHORING_MANIFEST_SHA256
        ),
    }
    result: dict[str, Any] = {}
    for label, (path, expected) in paths.items():
        _plain_file(path, label)
        actual = _sha256(path)
        if actual != expected:
            _fail(f"{label} SHA-256 differs")
        result[label] = {"path": str(path), "sha256": actual}
    marker = _load(paths["deployment_marker"][0], "deployment marker")
    if marker.get("complete") is not True or marker.get("role") != "decode" or _get(marker, "tests.total_passed") != 144:
        _fail("dfix2 marker completion/test count differs")
    if marker.get("source_tree") != args.source_tree:
        _fail("dfix2 source-tree binding differs")
    for relative, digest in (
        ("methods/bernini_action_editing/anchor_sga_anc_controller.py", CONTROLLER_SHA256),
        ("methods/bernini_action_editing/anchor_qk_transport.py", QK_TRANSPORT_SHA256),
        ("methods/bernini_action_editing/infer_anchor_sga_anc_event_v1.py", INFER_SHA256),
        ("methods/bernini_action_editing/infer_anchor_sga_anc_trained_editor_decode_v1.py", DECODE_SHA256),
        ("methods/bernini_action_editing/scripts/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh", BRIDGE_SHA256),
        ("methods/bernini_action_editing/validate_v14r2_deployment_marker.py", DEPLOYMENT_VALIDATOR_SHA256),
        ("methods/bernini_action_editing/assets/interaction_complex8_multianchor_authoring_v2.json", AUTHORING_MANIFEST_SHA256),
    ):
        if _get(marker, "required_files").get(relative) != digest:
            _fail(f"dfix2 marker omits exact {relative}")
    if paths["revision"][0].read_text(encoding="ascii").strip() != REVISION_VALUE:
        _fail("dfix2 revision value differs")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--transport", required=True, choices=TRANSPORTS)
    parser.add_argument("--arm-role", required=True, choices=ARM_ROLES)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--deployment-marker", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--content-manifest", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--qk-transport", required=True)
    parser.add_argument("--infer", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--deployment-validator", required=True)
    parser.add_argument("--authoring-manifest", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar, video, output = Path(args.sidecar), Path(args.video), Path(args.audit_output)
    if output.exists() or output.is_symlink():
        _fail("refusing to overwrite v15a audit sidecar")
    receipt = _load(sidecar, "native inference sidecar")
    proof = validate_receipt(
        receipt, video=video, transport=args.transport, arm_role=args.arm_role
    )
    _probe(video)
    authority = validate_authority(args)
    audit = {
        "schema_version": SCHEMA, "complete": True,
        "experiment_tag": EXPERIMENT_TAG,
        "probe_kind": "max-strength clean-noised route probe",
        "native_sidecar": {"path": str(sidecar), "sha256": _sha256(sidecar)},
        "output": {"path": str(video), "sha256": proof.pop("video_sha256"), "frames": 81, "fps": 25},
        "zero_update_frozen_base": {
            "training_performed": False, "optimization_steps": 0,
            "trained_checkpoint_loaded": False, "adapter_present": False,
            "base_frozen_before_and_after": True,
        },
        "contrast_proof": proof,
        "comparison": {
            "group_id": COMPARISON_GROUP,
            "arm_role": args.arm_role,
            "exact_label": ARM_LABELS[args.arm_role],
            "fixed_output_root": str(OUTPUT_ROOT),
            "outer_seed": OUTER_SEED,
            "matched_control_role": "route_off_plain_frozen",
            "pairable": True,
        },
        "qk_route_proof": {
            "transport": args.transport, "strength": 1.0,
            "active_steps": 0 if args.arm_role == "route_off_plain_frozen" else 40,
            "route_injection_enabled": args.arm_role != "route_off_plain_frozen",
            "selected_blocks": BLOCKS,
            "capture_count": 0 if args.arm_role == "route_off_plain_frozen" else EXPECTED_CAPTURES,
            "replay_count": 0 if args.arm_role == "route_off_plain_frozen" else EXPECTED_REPLAYS,
            "cached_fields": (
                []
                if args.arm_role == "route_off_plain_frozen"
                else ["query", "key"]
            ),
            "capture_schema_fields": ["query", "key"],
            "donor_value_or_pixels_used": False,
        },
        "dfix2_authority": authority,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as handle:
        json.dump(audit, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

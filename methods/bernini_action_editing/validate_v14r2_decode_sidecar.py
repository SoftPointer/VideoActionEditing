#!/usr/bin/env python3
"""Fail-closed validator for immutable v14r2 decode artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "bernini-pure-t2v-anchor-sga-anc-event-canary-v47"
TRAINING_SCHEMA = "bernini-online-anchor-attention-training-receipt-v3"
OBJECTIVE = "real_source_target_owned_routed_teacher_delta_v14r2"
EDITOR_FREEZE_SCHEMA = "bernini-frozen-peft-editor-certificate-v1"
EDITOR_LORA_LAYERS = 240
EDITOR_LORA_PARAMETER_TENSORS = 480
BLOCKS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29]
HEX = set("0123456789abcdef")


class V14R2DecodeValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Expected:
    video: str
    checkpoint: str
    step: int
    route: str
    transport: str
    transport_steps: int
    adapter_sha256: str
    adapter_config_sha256: str
    receipt_sha256: str
    source_sha256: str
    anchor_sha256: str
    preservation_mode: str
    sga_score_mode: str
    route_off: bool


def _fail(label: str) -> None:
    raise V14R2DecodeValidationError(label)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} is not an object")
    return value


def _get(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        current = _mapping(current, path).get(key)
    return current


def _eq(value: Mapping[str, Any], path: str, expected: Any) -> None:
    actual = _get(value, path)
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        _fail(f"{path} differs: {actual!r} != {expected!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_receipt(receipt: Mapping[str, Any], expected: Expected) -> None:
    """Validate binding, decode semantics, and actual pure-QK route activity."""

    if expected.route_off != (expected.transport_steps == 0):
        _fail("route-off flag and transport-step count differ")
    for label, digest in (
        ("adapter", expected.adapter_sha256),
        ("adapter config", expected.adapter_config_sha256),
        ("receipt", expected.receipt_sha256),
        ("source", expected.source_sha256),
        ("anchor", expected.anchor_sha256),
    ):
        if len(digest) != 64 or any(character not in HEX for character in digest):
            _fail(f"{label} SHA-256 is invalid")

    freeze_before = _mapping(receipt.get("freeze_before"), "freeze before")
    freeze_after = _mapping(receipt.get("freeze_after"), "freeze after")
    if freeze_before != freeze_after:
        _fail("trained editor freeze certificate changed during decode")
    for path, value in (
        ("schema_version", EDITOR_FREEZE_SCHEMA),
        ("base_and_adapter_frozen", True),
        ("base_frozen", True),
        ("trainable_parameter_tensors", 0),
        ("trainable_parameter_elements", 0),
        ("peft_model_authenticated", True),
        ("adapter_disable_context_available", True),
        ("adapter_disable_context_reversible", True),
        ("adapter_disable_context_refreezes_parameters", True),
        ("adapter_kept_unmerged", True),
        ("adapter_enabled_for_editor_calls", True),
        ("pure_t2v_teacher_policy", "temporary_disable_adapter_context"),
        ("adapter_config_names", ["default"]),
        ("lora_layer_count", EDITOR_LORA_LAYERS),
        ("lora_parameter_tensors", EDITOR_LORA_PARAMETER_TENSORS),
    ):
        _eq(freeze_before, path, value)
    elements = freeze_before.get("lora_parameter_elements")
    if isinstance(elements, bool) or not isinstance(elements, int) or elements <= 0:
        _fail("trained editor LoRA parameter element count is invalid")
    for key in (
        "lora_layer_inventory_sha256",
        "lora_parameter_inventory_sha256",
    ):
        digest = freeze_before.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in HEX for character in digest)
        ):
            _fail(f"trained editor {key} is invalid")

    _eq(receipt, "schema_version", SCHEMA)
    _eq(receipt, "complete", True)
    _eq(receipt, "loaded_trained_attention_checkpoint", True)
    trained = _mapping(receipt.get("trained_attention_checkpoint"), "checkpoint")
    for path, value in (
        ("path", expected.checkpoint),
        ("schema_version", TRAINING_SCHEMA),
        ("global_step", expected.step),
        ("training_objective", OBJECTIVE),
        ("route_operator", expected.route),
        ("required_decode_transport", expected.transport),
        ("adapter_model_sha256", expected.adapter_sha256),
        ("adapter_config_sha256", expected.adapter_config_sha256),
        ("receipt_sha256", expected.receipt_sha256),
        ("adapter_kept_unmerged", True),
        ("frozen_anchor_calls_use_disable_adapter", True),
        ("target_and_source_editor_calls_keep_adapter_enabled", True),
        ("adapter_enabled_for_target_source_calls", True),
        ("anchor_injection_enabled", not expected.route_off),
        ("same_checkpoint_route_off_causal_control", expected.route_off),
    ):
        _eq(trained, path, value)
    binding = _mapping(trained.get("checkpoint_binding"), "checkpoint binding")
    for path, value in (
        ("receipt_sha256", expected.receipt_sha256),
        ("adapter_config_sha256", expected.adapter_config_sha256),
        ("adapter_model_sha256", expected.adapter_sha256),
        ("global_step", expected.step),
        ("training_objective", OBJECTIVE),
        ("route_operator", expected.route),
        ("required_decode_transport", expected.transport),
    ):
        _eq(binding, path, value)
    _eq(trained, "checkpoint_binding_sha256", _canonical_sha256(binding))
    fail_closed = _mapping(
        trained.get("expectations_fail_closed"), "checkpoint expectations"
    )
    for path, value in (
        ("global_step", expected.step),
        ("training_objective", OBJECTIVE),
        ("route_operator", expected.route),
        ("adapter_model_sha256", expected.adapter_sha256),
        ("adapter_config_sha256", expected.adapter_config_sha256),
        ("receipt_sha256", expected.receipt_sha256),
        ("all_validated", True),
    ):
        _eq(fail_closed, path, value)

    _eq(receipt, "source.sha256", expected.source_sha256)
    _eq(receipt, "source.role", "clean_edit_state_identity_appearance_scene_authority")
    _eq(receipt, "pure_t2v_anchor.sha256", expected.anchor_sha256)
    anchor_bank = receipt.get("pure_t2v_anchor_bank")
    if not isinstance(anchor_bank, list) or len(anchor_bank) != 1:
        _fail("pure-T2V anchor bank is not exact singleton")
    _eq(_mapping(anchor_bank[0], "anchor bank row"), "sha256", expected.anchor_sha256)

    mechanism = _mapping(receipt.get("mechanism"), "mechanism")
    for path, value in (
        ("arm", "AQK_SGA5"),
        ("transport", expected.transport),
        ("transport_strength", 0.25),
        ("transport_steps", expected.transport_steps),
        ("initial_phase_clamp", True),
        ("field_guidance", "raw_cfg"),
        ("field_model", "first_phase_caption_i2v"),
        ("source_cfg_scale", 4.5),
        ("target_cfg_scale", 4.5),
        ("sga_temperature", 0.01),
        ("early_candidate_count", 5),
        ("initial_noise_proposal_mode", "keyed_only"),
        ("anchor_state_mode", "clean_noised"),
        ("anchor_cfg_scope", "shared"),
        ("anchor_contrast_mode", "caption_noop_same_video"),
        ("anchor_sigma_cap", 1.0),
        ("preservation_mode", expected.preservation_mode),
        ("preservation_keep_fraction", 0.2),
        ("preservation_outside_scale", 0.0),
        ("preservation_dilation", 1),
        ("preservation_residual_fraction", 0.0),
        ("preservation_object_identity_strength", 0.0),
        ("preservation_start_step", 0),
        ("preservation_ramp_steps", 1),
        ("sga_score_mode", expected.sga_score_mode),
        ("anchor_candidate_mode", "single_shared"),
        ("anchor_bank_size", 1),
        ("anchor_spatial_alignment", "none"),
        ("selected_blocks", BLOCKS),
        ("pure_t2v_anchor_online_block_transport_enabled", not expected.route_off),
        ("pure_t2v_anchor_online_velocity_transport_enabled", False),
        ("pure_t2v_anchor_values_or_pixels_copied_to_output", False),
    ):
        _eq(mechanism, path, value)
    audit = _mapping(mechanism.get("decode_audit_contract"), "decode audit")
    for path, value in (
        ("transport_steps", expected.transport_steps),
        ("anchor_state_mode", "clean_noised"),
        ("anchor_cfg_scope", "shared"),
        ("source_cfg_scale", 4.5),
        ("target_cfg_scale", 4.5),
        ("source_and_target_cfg_equal", True),
        ("pure_t2v_teacher_adapter_policy", "disable_loaded_editor_adapter"),
        ("target_source_editor_adapter_policy", "loaded_adapter_enabled"),
        ("trained_route_off_control_explicitly_allowed", expected.route_off),
        ("same_checkpoint_route_off_causal_control", expected.route_off),
        ("anchor_injection_enabled", not expected.route_off),
    ):
        _eq(audit, path, value)

    causal = _mapping(receipt.get("causal_control"), "causal control")
    for path, value in (
        ("enabled", expected.route_off),
        ("kind", "same_trained_checkpoint_route_off" if expected.route_off else None),
        ("explicit_opt_in", expected.route_off),
        ("trained_adapter_loaded", True),
        ("adapter_enabled_for_target_source_calls", True),
        ("anchor_injection_enabled", not expected.route_off),
        ("transport_steps", expected.transport_steps),
    ):
        _eq(causal, path, value)

    trace = _mapping(mechanism.get("trace"), "trace")
    cache = _mapping(trace.get("attention_cache"), "attention cache")
    if expected.route_off:
        for path, value in (
            ("anchor_model_forwards", 0),
            ("anchor_native_trajectory_model_forwards", 0),
            ("anchor_candidate_cells", 0),
            ("anchor_active_schedule", []),
            ("target_owned_qk_route_v14r2", False),
            ("anchor_donor_cached_fields", None),
            ("anchor_donor_value_hidden_output_or_coordinate_used", None),
        ):
            _eq(trace, path, value)
        for key in (
            "capture_count",
            "replay_count",
            "qk_only_capture_count",
            "qk_only_replay_count",
            "pending_entries",
        ):
            _eq(cache, key, 0)
        _eq(receipt, "pure_t2v_anchor.active_solver_steps", 0)
        _eq(
            receipt,
            "pure_t2v_anchor.model_forward_at_every_active_solver_step_and_candidate",
            False,
        )
    else:
        early_steps = min(expected.transport_steps, 3)
        expected_cells = early_steps * 5 + max(0, expected.transport_steps - early_steps)
        expected_capture = 2 * expected_cells * len(BLOCKS)
        expected_replay = 2 * expected_capture
        for path, value in (
            ("anchor_model_forwards", 2 * expected_cells),
            ("anchor_native_trajectory_model_forwards", 0),
            ("anchor_candidate_cells", expected_cells),
            ("target_owned_qk_route_v14r2", True),
            ("anchor_donor_cached_fields", ["query", "key"]),
            ("anchor_donor_value_hidden_output_or_coordinate_used", False),
            ("anchor_to_target_appearance_correspondence_used", False),
            ("anchor_temporal_attention_kernel_contrast", True),
            ("anchor_temporal_kernel_applied_to_target_value_only", True),
            ("anchor_route_shared_by_target_negative_and_condition", True),
            ("anchor_route_target_conditional_only", False),
            ("initial_latent_phase_clamped_after_every_update", True),
            ("anchor_value_stream_copied", False),
            ("source_value_stream_retained", True),
            ("anchor_present_in_every_active_target_candidate", True),
            ("anchor_present_after_active_interval", False),
        ):
            _eq(trace, path, value)
        if not isinstance(trace.get("anchor_active_schedule"), list) or len(
            trace["anchor_active_schedule"]
        ) != expected.transport_steps:
            _fail("anchor active schedule length differs")
        for path, value in (
            ("capture_count", expected_capture),
            ("qk_only_capture_count", expected_capture),
            ("replay_count", expected_replay),
            ("qk_only_replay_count", expected_replay),
            ("pending_entries", 0),
            ("qk_only_cached_fields", ["query", "key"]),
        ):
            _eq(cache, path, value)
        _eq(receipt, "pure_t2v_anchor.active_solver_steps", expected.transport_steps)
        _eq(
            receipt,
            "pure_t2v_anchor.model_forward_at_every_active_solver_step_and_candidate",
            True,
        )

    _eq(receipt, "output.path", expected.video)
    _eq(receipt, "output.frames", 81)
    _eq(receipt, "output.fps", 25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--route", required=True)
    parser.add_argument("--transport", required=True)
    parser.add_argument("--transport-steps", required=True, type=int)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--adapter-config-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--preservation-mode", required=True)
    parser.add_argument("--sga-score-mode", required=True)
    parser.add_argument("--route-off", required=True, choices=("0", "1"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    video = Path(args.video)
    sidecar = Path(args.sidecar)
    if (
        not video.is_file()
        or video.is_symlink()
        or not sidecar.is_file()
        or sidecar.is_symlink()
    ):
        _fail("video/sidecar artifact closure differs")
    try:
        receipt = json.loads(sidecar.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V14R2DecodeValidationError("sidecar is unreadable") from error
    receipt = _mapping(receipt, "sidecar")
    expected = Expected(
        video=str(video),
        checkpoint=args.checkpoint,
        step=args.step,
        route=args.route,
        transport=args.transport,
        transport_steps=args.transport_steps,
        adapter_sha256=args.adapter_sha256,
        adapter_config_sha256=args.adapter_config_sha256,
        receipt_sha256=args.receipt_sha256,
        source_sha256=args.source_sha256,
        anchor_sha256=args.anchor_sha256,
        preservation_mode=args.preservation_mode,
        sga_score_mode=args.sga_score_mode,
        route_off=args.route_off == "1",
    )
    validate_receipt(receipt, expected)
    _eq(receipt, "output.sha256", _sha256(video))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

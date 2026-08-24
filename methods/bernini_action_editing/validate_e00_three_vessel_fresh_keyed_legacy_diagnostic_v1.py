#!/usr/bin/env python3
"""Validate the honest dfix2 E00 corrected-caption/old-route diagnostic.

The legacy ABI cannot split anchor observation from target sampling.  The K0
arm therefore runs the audited exact-identity action/no-op observer and is
named ``observer_matched_output_routeoff``.  It is deliberately rejected if
described as anchor-free.  This validator also binds per-rank ``fork_rng``
restoration and the raw keyed-noise hashes captured by the runtime wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "bernini-e00-three-vessel-fresh-keyed-legacy-diagnostic-spec-v1"
NATIVE_SCHEMA = "bernini-pure-t2v-anchor-sga-anc-event-canary-v47"
RNG_SCHEMA = "bernini-e00-legacy-infer-fork-rng-audit-v1"
ARM_AUDIT_SCHEMA = "bernini-e00-fresh-keyed-legacy-arm-audit-v1"
PAIR_AUDIT_SCHEMA = "bernini-e00-fresh-keyed-legacy-pair-audit-v1"
METHOD_ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC = METHOD_ROOT / "assets/e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.json"
ARM_ROLES = (
    "pure_noobserver_output_routeoff",
    "observer_matched_output_routeoff",
    "old_pureqk_temporal_routeon",
)
PURE_QK_TRANSPORT = "self_target_owned_temporal_kernel_attn_output_v14r2"
EXPECTED_TRANSPORT = {role: PURE_QK_TRANSPORT for role in ARM_ROLES}
BLOCKS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29]
EXPECTED_CAPTURE = 2 * 40 * len(BLOCKS)
EXPECTED_REPLAY = 2 * EXPECTED_CAPTURE
MV2V_SYSTEM_PROMPT = (
    "You are a helpful assistant for editing. You might need to adjust the "
    "video's style, lighting, colors, textures, and the subject's pose or action."
)
T2V_SYSTEM_PROMPT = "You are a helpful assistant specialized in text-to-video generation."
LEGACY_NOOP = (
    "Keep the source video exactly unchanged, including every subject, appearance, "
    "action, camera motion, background, timing, and composition."
)
NEGATIVE_SHA256 = "ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e"
CHECKPOINT_CONTENT_FILE_COUNT = 23
HEX = set("0123456789abcdef")


class E00LegacyDiagnosticError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise E00LegacyDiagnosticError(message)


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail("prompt text must be non-empty UTF-8")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _keyed_noise_seed(master_seed: int, step: int, candidate: int) -> int:
    payload = (
        f"bernini-guided-sac-v2\0{master_seed}\0{step}\0{candidate}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        _fail(f"{label} is not a lowercase SHA-256")
    return value


def _load(path: Path, label: str) -> Mapping[str, Any]:
    _plain_file(path, label)
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise E00LegacyDiagnosticError(f"{label} is unreadable") from error


def validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    _eq(spec, "schema_version", SCHEMA)
    for path, expected in (
        ("status.draft_only", True),
        ("status.execution_authorized", False),
        ("status.gpu_run_started", False),
        ("status.training_performed", False),
        ("legacy_abi_audit.offline_anchor_graph_export_supported", False),
        ("legacy_abi_audit.offline_anchor_graph_import_supported", False),
        ("legacy_abi_audit.preferred_split_process_contract_satisfied", False),
        ("legacy_abi_audit.route_off_target_process_reads_anchor_video_path", True),
        ("legacy_abi_audit.route_off_target_process_decodes_anchor_video", True),
        ("legacy_abi_audit.pure_noobserver_arm_uses_source_video_as_required_anchor_path_placeholder", False),
        ("legacy_abi_audit.pure_noobserver_arm_uses_source_frame0_static_required_anchor_path_placeholder", True),
        ("common_runtime_contract.initial_noise_proposal_mode", "keyed_only"),
        ("common_runtime_contract.anchor_generation_gaussian_path_read", False),
        ("common_runtime_contract.arm", "AQK_IID1"),
        ("common_runtime_contract.sga_enabled", False),
        ("common_runtime_contract.anc_enabled", False),
        ("common_runtime_contract.adapter_present", False),
        ("common_runtime_contract.optimization_steps", 0),
        ("common_runtime_contract.observer_steps_for_matched_route_pair", 40),
        ("common_runtime_contract.initial_phase_clamp", True),
        ("comparison_limits.cache_abi_identical", True),
        ("comparison_limits.fully_matches_ideal_offline_graph_pair", False),
        ("comparison_limits.promotion_forbidden", True),
    ):
        _eq(spec, path, expected)
    forbidden = _get(spec, "legacy_abi_audit.forbidden_claims")
    if not isinstance(forbidden, list) or "anchor-free K0" not in forbidden:
        _fail("legacy fallback must explicitly forbid the anchor-free K0 claim")
    prompts = _mapping(spec.get("data_and_prompt_contract"), "data_and_prompt_contract")
    placeholder = _mapping(
        prompts.get("pure_noobserver_placeholder"), "pure_noobserver_placeholder"
    )
    placeholder_relative = placeholder.get("package_relative_path")
    if (
        not isinstance(placeholder_relative, str)
        or Path(placeholder_relative).is_absolute()
        or ".." in Path(placeholder_relative).parts
    ):
        _fail("source-static placeholder package path is unsafe")
    placeholder_path = METHOD_ROOT.parents[1] / placeholder_relative
    _plain_file(placeholder_path, "source-static placeholder")
    if file_sha256(placeholder_path) != placeholder.get("sha256"):
        _fail("source-static placeholder SHA-256 differs")
    for path, expected in (
        ("derived_only_from_source_sha256", prompts.get("source_video_sha256")),
        ("frame_count", 81),
        ("fps", 25),
        ("width", 704),
        ("height", 1056),
    ):
        if placeholder.get(path) != expected:
            _fail(f"source-static placeholder {path} differs")
    for stem in (
        "source_caption",
        "target_caption",
        "ideal_source_noop_caption",
        "source_noop_caption",
        "editing_instruction",
        "anchor_caption",
        "anchor_noop_caption",
    ):
        actual = _text_sha256(prompts.get(stem))
        if prompts.get(f"{stem}_utf8_sha256") != actual:
            _fail(f"{stem} hash differs")
    instruction = prompts["editing_instruction"]
    for fragment in (
        "Frame 0 must exactly retain the source state",
        "upper-left white ceramic pouring vessel (#1)",
        "transparent handled glass vessel (#2)",
        "small white ceramic teacup on its saucer (#3)",
        "continuous amber stream from #2 into #3",
        "do not turn #2 into a white or opaque vessel",
    ):
        if fragment not in instruction:
            _fail(f"corrected three-vessel instruction omits {fragment!r}")
    if prompts.get("source_noop_caption") != LEGACY_NOOP:
        _fail("legacy effective source no-op instruction differs")
    expected_native_prompts = {
        "action_mv2v_sha256": _text_sha256(MV2V_SYSTEM_PROMPT + instruction),
        "source_noop_mv2v_sha256": _text_sha256(MV2V_SYSTEM_PROMPT + LEGACY_NOOP),
        "anchor_t2v_sha256": _text_sha256(T2V_SYSTEM_PROMPT + prompts["anchor_caption"]),
        "anchor_noop_t2v_sha256": _text_sha256(
            T2V_SYSTEM_PROMPT + prompts["anchor_noop_caption"]
        ),
        "source_t2v_sha256": _text_sha256(T2V_SYSTEM_PROMPT + prompts["source_caption"]),
        "target_t2v_sha256": _text_sha256(T2V_SYSTEM_PROMPT + prompts["target_caption"]),
        "negative_sha256": NEGATIVE_SHA256,
    }
    if prompts.get("effective_native_prompt_sha256") != expected_native_prompts:
        _fail("effective native prompt hash closure differs")
    common = _mapping(spec.get("common_runtime_contract"), "common_runtime_contract")
    if common.get("candidate_count_by_step") != [1] * 40:
        _fail("fresh diagnostic must be IID1 at all 40 solver steps")
    if common.get("selected_blocks") != BLOCKS:
        _fail("selected blocks differ from dfix2 pure-QK audit")
    arms = spec.get("arms")
    if not isinstance(arms, list) or [row.get("arm_role") for row in arms] != list(ARM_ROLES):
        _fail("arm order/closure differs")
    for row, role in zip(arms, ARM_ROLES):
        if row.get("legacy_transport") != EXPECTED_TRANSPORT[role]:
            _fail(f"{role} transport differs")
        label = row.get("label")
        if not isinstance(label, str) or "DIAG" not in label or (
            role == ARM_ROLES[0] and "PURE_NOOBSERVER" not in label
        ) or (role == ARM_ROLES[1] and "NOT_ANCHORFREE" not in label
        ) or (role == ARM_ROLES[2] and "NOT_V15B" not in label):
            _fail(f"{role} label is not honest")
    if (
        arms[0].get("legacy_transport_steps") != 0
        or arms[0].get("observer_reads_anchor") is not False
        or arms[0].get("self_generated_anchor_video_read") is not False
        or arms[0].get("required_anchor_path_placeholder") != "source_frame0_static81_package_asset"
        or arms[0].get("target_route_replay_steps") != 0
    ):
        _fail("pure no-observer arm closure differs")
    if (
        arms[1].get("legacy_transport_steps") != 40
        or arms[1].get("observer_reads_anchor") is not True
        or arms[1].get("target_route_replay_steps") != 0
        or arms[1].get("target_output_reads_anchor_route") is not False
    ):
        _fail("observer-matched control is not output-route-off K0")
    if (
        arms[2].get("legacy_transport_steps") != 40
        or arms[2].get("observer_reads_anchor") is not True
        or arms[2].get("target_route_replay_steps") != 40
        or arms[2].get("target_output_reads_anchor_route") is not True
    ):
        _fail("old pure-QK arm is not target-route-on K40")
    return {
        "schema_version": SCHEMA,
        "canonical_sha256": canonical_sha256(spec),
        "editing_instruction_sha256": prompts["editing_instruction_utf8_sha256"],
    }


def load_spec(path: Path | str = DEFAULT_SPEC) -> Mapping[str, Any]:
    spec = _load(Path(path), "diagnostic spec")
    validate_spec(spec)
    return spec


def _arm_row(spec: Mapping[str, Any], arm_role: str) -> Mapping[str, Any]:
    rows = [row for row in spec["arms"] if row.get("arm_role") == arm_role]
    if len(rows) != 1:
        _fail(f"arm role lookup differs: {arm_role}")
    return _mapping(rows[0], "arm row")


def validate_native_receipt(
    receipt: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    arm_role: str,
    video: Optional[Path] = None,
) -> dict[str, Any]:
    if arm_role not in ARM_ROLES:
        _fail("unsupported arm role")
    prompts = _mapping(spec["data_and_prompt_contract"], "prompt contract")
    common = _mapping(spec["common_runtime_contract"], "runtime contract")
    transport = EXPECTED_TRANSPORT[arm_role]
    no_observer = arm_role == ARM_ROLES[0]
    route_on = arm_role == ARM_ROLES[2]
    legacy_steps = 0 if no_observer else 40
    expected_anchor_sha = (
        prompts["pure_noobserver_placeholder"]["sha256"]
        if no_observer
        else prompts["anchor_video_sha256"]
    )
    for path, expected in (
        ("schema_version", NATIVE_SCHEMA),
        ("complete", True),
        ("training_performed", False),
        ("optimization_steps", 0),
        ("loaded_trained_attention_checkpoint", False),
        ("trained_attention_checkpoint", None),
        ("source.sha256", prompts["source_video_sha256"]),
        ("pure_t2v_anchor.sha256", expected_anchor_sha),
        ("pure_t2v_anchor.active_solver_steps", legacy_steps),
        ("pure_t2v_anchor.model_forward_at_every_active_solver_step_and_candidate", not no_observer),
        ("anchor_generation_initial_gaussian", None),
        ("mechanism.arm", "AQK_IID1"),
        ("mechanism.transport", transport),
        ("mechanism.transport_strength", 1.0),
        ("mechanism.transport_steps", legacy_steps),
        ("mechanism.initial_phase_clamp", True),
        ("mechanism.field_guidance", "raw_cfg"),
        ("mechanism.field_model", "first_phase_caption_i2v"),
        ("mechanism.source_cfg_scale", 4.5),
        ("mechanism.target_cfg_scale", 4.5),
        ("mechanism.early_candidate_count", 5),
        ("mechanism.initial_noise_proposal_mode", "keyed_only"),
        ("mechanism.anchor_state_mode", "clean_noised"),
        ("mechanism.anchor_cfg_scope", "shared"),
        ("mechanism.anchor_contrast_mode", "dynamic_static_same_caption"),
        ("mechanism.preservation_mode", "none"),
        ("mechanism.anchor_candidate_mode", "single_shared"),
        ("mechanism.anchor_bank_size", 1),
        ("mechanism.selected_blocks", BLOCKS),
        ("mechanism.pure_t2v_anchor_values_or_pixels_copied_to_output", False),
    ):
        _eq(receipt, path, expected)
    before = _mapping(receipt.get("freeze_before"), "freeze_before")
    after = _mapping(receipt.get("freeze_after"), "freeze_after")
    if before != after:
        _fail("frozen certificate changed during inference")
    for path, expected in (
        ("base_frozen", True),
        ("trainable_parameter_tensors", 0),
        ("trainable_parameter_elements", 0),
        ("lora_module_count", 0),
        ("adapter_modules_absent", True),
    ):
        _eq(before, path, expected)
    checkpoint = _mapping(receipt.get("checkpoint_content"), "checkpoint_content")
    for path, expected in (
        ("manifest_sha256_computed", common["checkpoint_manifest_sha256"]),
        ("manifest_sha256_expected", common["checkpoint_manifest_sha256"]),
        ("verified_file_count", CHECKPOINT_CONTENT_FILE_COUNT),
        ("every_file_sha256_verified", True),
    ):
        _eq(checkpoint, path, expected)
    _hex(checkpoint.get("verified_entries_digest"), "checkpoint entries digest")
    native_prompts = _mapping(receipt.get("prompts"), "native prompts")
    expected_native_prompts = _mapping(
        prompts.get("effective_native_prompt_sha256"), "effective native prompt hashes"
    )
    for native_field, expected_hash in expected_native_prompts.items():
        if native_prompts.get(native_field) != expected_hash:
            _fail(f"native prompt hash differs: {native_field}")
    trace = _mapping(_get(receipt, "mechanism.trace"), "trace")
    if _get(receipt, "mechanism.trace_digest") != canonical_sha256(trace):
        _fail("native trace digest does not bind the trace bytes")
    for path, expected in (
        ("candidate_counts", [1] * 40),
        ("configured_early_candidate_count", 5),
        ("initial_noise_proposal_mode", "keyed_only"),
        ("anchor_initial_gaussian_used_at_step0_candidate0", False),
        ("anchor_action_reward_used_for_sga", False),
        ("sga_weights_forced_to_anchor_candidate0", False),
        ("anchor_model_forwards", 0 if no_observer else 80),
        ("anchor_candidate_cells", 0 if no_observer else 40),
        ("target_raw_cfg_forwards", 80),
        ("source_raw_cfg_forwards", 80),
        ("target_model_forwards", 80),
        ("source_model_forwards", 80),
        ("anchor_action_noop_attention_observed_without_transport", False),
        ("target_owned_qk_route_v14r2", not no_observer),
        ("anchor_temporal_attention_kernel_contrast", not no_observer),
        ("anchor_temporal_kernel_applied_to_target_value_only", not no_observer),
        ("anchor_donor_cached_fields", None if no_observer else ["query", "key"]),
        ("anchor_donor_value_hidden_output_or_coordinate_used", None if no_observer else False),
        ("anchor_to_target_appearance_correspondence_used", None if no_observer else False),
        ("initial_latent_phase_clamped_after_every_update", True),
        ("anchor_value_stream_copied", False),
        ("source_value_stream_retained", True),
    ):
        _eq(trace, path, expected)
    cache = _mapping(trace.get("attention_cache"), "attention cache")
    for path, expected in (
        ("capture_count", 0 if no_observer else EXPECTED_CAPTURE),
        ("replay_count", 0 if no_observer else EXPECTED_REPLAY),
        ("qk_only_capture_count", 0 if no_observer else EXPECTED_CAPTURE),
        ("qk_only_replay_count", 0 if no_observer else EXPECTED_REPLAY),
        ("pending_entries", 0),
        ("selected_block_indices", BLOCKS),
    ):
        _eq(cache, path, expected)
    if video is not None:
        _plain_file(video, "output video")
        if _get(receipt, "output.path") != str(video):
            _fail("native receipt output path differs")
        if _get(receipt, "output.sha256") != file_sha256(video):
            _fail("native receipt output SHA-256 differs")
    for path, expected in (("output.frames", 81), ("output.fps", 25)):
        _eq(receipt, path, expected)
    return {
        "arm_role": arm_role,
        "transport": transport,
        "observer_steps": legacy_steps,
        "target_route_replay_steps": 40 if route_on else 0,
        "outer_schedule_digest": trace.get("outer_schedule_digest"),
        "trace_digest": _get(receipt, "mechanism.trace_digest"),
        "native_output_sha256": _get(receipt, "output.sha256"),
        "frozen_certificate_sha256": canonical_sha256(before),
    }


def validate_rng_receipts(
    receipts: Sequence[Mapping[str, Any]], *, arm_role: str,
    expected_output_path: str, native_receipt_sha256: str,
) -> dict[str, Any]:
    if len(receipts) != 4:
        _fail("exactly four per-rank RNG receipts are required")
    ordered = sorted(receipts, key=lambda row: row.get("rank", -1))
    if [row.get("rank") for row in ordered] != [0, 1, 2, 3]:
        _fail("RNG receipt ranks differ")
    reference_rows: Optional[Any] = None
    for expected_rank, row in enumerate(ordered):
        for path, expected in (
            ("schema_version", RNG_SCHEMA),
            ("complete", True),
            ("arm_role", arm_role),
            ("rank", expected_rank),
            ("local_rank", expected_rank),
            ("world_size", 4),
            ("fork_rng.enabled", True),
            ("fork_rng.scope", "entire_legacy_inference_entrypoint_per_rank"),
            ("fork_rng.cpu_state_restored", True),
            ("fork_rng.owned_cuda_state_restored", True),
            ("legacy_abi.offline_anchor_graph_split_supported", False),
            ("legacy_abi.target_process_reads_anchor_video_path", True),
        ("legacy_abi.target_process_decodes_anchor_video", True),
        ("legacy_abi.self_generated_anchor_video_read", arm_role != ARM_ROLES[0]),
        ("legacy_abi.required_anchor_path_is_source_placeholder", False),
        ("legacy_abi.required_anchor_path_is_source_frame0_static_placeholder", arm_role == ARM_ROLES[0]),
        ("legacy_abi.output_path", expected_output_path),
        ("route_application.enabled", arm_role == ARM_ROLES[2]),
        ("route_application.exact_identity_gate", arm_role == ARM_ROLES[1]),
        ("route_application.call_count", 0 if arm_role == ARM_ROLES[0] else 2 * 40 * len(BLOCKS)),
            ("runtime_noise.scheme", "sha256_keyed_cpu_torch_generator_v1"),
            ("runtime_noise.master_seed", 2027),
        ):
            _eq(row, path, expected)
        _eq(row, "fork_rng.owned_cuda_device", expected_rank)
        before = _mapping(_get(row, "fork_rng.before"), "fork_rng before")
        after = _mapping(_get(row, "fork_rng.after"), "fork_rng after")
        if before != after:
            _fail(f"rank {expected_rank} fork_rng state bytes differ")
        _hex(before.get("cpu_sha256"), "CPU RNG SHA-256")
        _hex(before.get("cuda_sha256"), "CUDA RNG SHA-256")
        native_proof = row.get("native_adapter_off_proof_rank0")
        if expected_rank == 0:
            native_proof = _mapping(native_proof, "rank-zero native receipt proof")
            if native_proof.get("native_receipt_path") != expected_output_path + ".receipt.json":
                _fail("rank-zero RNG receipt is not bound to the native receipt path")
            if native_proof.get("native_receipt_sha256") != native_receipt_sha256:
                _fail("rank-zero RNG receipt is not bound to the native receipt bytes")
        elif native_proof is not None:
            _fail("only rank zero may bind the native receipt")
        rows = _get(row, "runtime_noise.rows")
        if not isinstance(rows, list) or len(rows) != 40:
            _fail(f"rank {expected_rank} raw noise closure differs")
        for step, noise in enumerate(rows):
            if (
                not isinstance(noise, Mapping)
                or noise.get("master_seed") != 2027
                or noise.get("step") != step
                or noise.get("candidate") != 0
                or noise.get("derived_seed") != _keyed_noise_seed(2027, step, 0)
            ):
                _fail(f"rank {expected_rank} noise coordinates differ at step {step}")
            _hex(noise.get("raw_noise_sha256"), "raw noise SHA-256")
        if reference_rows is None:
            reference_rows = rows
        elif rows != reference_rows:
            _fail("raw keyed-noise rows differ across distributed ranks")
        latent = _mapping(row.get("predecode_latent"), "predecode latent")
        _hex(latent.get("raw_storage_sha256"), "predecode latent SHA-256")
        if latent.get("dtype") != "torch.float32" or latent.get("finite") is not True:
            _fail(f"rank {expected_rank} predecode latent contract differs")
        shape = latent.get("shape")
        if not isinstance(shape, list) or len(shape) != 5 or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in shape
        ):
            _fail(f"rank {expected_rank} predecode latent shape differs")
    latent_hashes = {_get(row, "predecode_latent.raw_storage_sha256") for row in ordered}
    if len(latent_hashes) != 1:
        _fail("predecode latent hash differs across distributed ranks")
    return {
        "rank_count": 4,
        "all_rank_rng_state_restored": True,
        "raw_noise_rows": reference_rows,
        "raw_noise_bank_sha256": canonical_sha256(reference_rows),
        "route_application_enabled": arm_role == ARM_ROLES[2],
        "predecode_latent_sha256": latent_hashes.pop(),
    }


def _probe_video(video: Path) -> None:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or streams[0].get("nb_read_frames") != "81" or streams[0].get("avg_frame_rate") != "25/1":
        _fail("output video is not exactly 81 frames at 25 fps")


def build_arm_audit(
    *,
    spec: Mapping[str, Any],
    native_receipt: Mapping[str, Any],
    rng_receipts: Sequence[Mapping[str, Any]],
    arm_role: str,
    native_receipt_sha256: str,
    video: Optional[Path] = None,
) -> dict[str, Any]:
    native = validate_native_receipt(
        native_receipt, spec=spec, arm_role=arm_role, video=video
    )
    expected_output_path = _get(native_receipt, "output.path")
    if not isinstance(expected_output_path, str) or not expected_output_path:
        _fail("native output path is absent")
    rng = validate_rng_receipts(
        rng_receipts,
        arm_role=arm_role,
        expected_output_path=expected_output_path,
        native_receipt_sha256=native_receipt_sha256,
    )
    arm = _arm_row(spec, arm_role)
    return {
        "schema_version": ARM_AUDIT_SCHEMA,
        "complete": True,
        "arm_role": arm_role,
        "label": arm["label"],
        "spec_canonical_sha256": canonical_sha256(spec),
        "honest_scope": arm["honest_name"],
        "anchor_free": False,
        "offline_anchor_graph": False,
        "training_performed": False,
        "optimization_steps": 0,
        "native": native,
        "rng_and_noise": rng,
    }


def validate_pair_audits(
    pure_noobserver: Mapping[str, Any],
    observer_routeoff: Mapping[str, Any],
    observer_routeon: Mapping[str, Any],
) -> dict[str, Any]:
    audits = (pure_noobserver, observer_routeoff, observer_routeon)
    for audit, role in zip(audits, ARM_ROLES):
        _eq(audit, "schema_version", ARM_AUDIT_SCHEMA)
        _eq(audit, "complete", True)
        _eq(audit, "arm_role", role)
        _eq(audit, "anchor_free", False)
        _eq(audit, "training_performed", False)
    if len({audit.get("spec_canonical_sha256") for audit in audits}) != 1:
        _fail("arm specs differ")
    for path in (
        "rng_and_noise.raw_noise_rows",
        "rng_and_noise.raw_noise_bank_sha256",
        "native.outer_schedule_digest",
        "native.frozen_certificate_sha256",
    ):
        values = [_get(audit, path) for audit in audits]
        if any(value != values[0] for value in values[1:]):
            _fail(f"three-arm diagnostic differs at {path}")
    if [
        _get(audit, "native.target_route_replay_steps") for audit in audits
    ] != [0, 0, 40]:
        _fail("three-arm target-route intervention differs")
    if _get(pure_noobserver, "rng_and_noise.predecode_latent_sha256") != _get(
        observer_routeoff, "rng_and_noise.predecode_latent_sha256"
    ):
        _fail("observer changed the route-off predecode latent")
    if _get(pure_noobserver, "native.native_output_sha256") != _get(
        observer_routeoff, "native.native_output_sha256"
    ):
        _fail("observer changed the route-off MP4 bytes")
    return {
        "schema_version": PAIR_AUDIT_SCHEMA,
        "complete": True,
        "diagnostic_only": True,
        "not_v15b": True,
        "not_fully_matched_offline_graph_pair": True,
        "same_raw_keyed_noise": True,
        "same_outer_schedule": True,
        "same_frozen_model": True,
        "pure_noobserver_vs_observer_routeoff_predecode_latent_exact": True,
        "pure_noobserver_vs_observer_routeoff_video_sha256_exact": True,
        "pure_noobserver_role": ARM_ROLES[0],
        "observer_routeoff_role": ARM_ROLES[1],
        "observer_routeon_role": ARM_ROLES[2],
        "only_causal_claim": "B/C isolate the old pure-QK route application; A/B prove that the same-process observer has zero decoded-output side effect under the pinned legacy ABI",
    }


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    spec_parser = sub.add_parser("spec")
    spec_parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    arm = sub.add_parser("arm")
    arm.add_argument("--spec", default=str(DEFAULT_SPEC))
    arm.add_argument("--arm-role", required=True, choices=ARM_ROLES)
    arm.add_argument("--native-receipt", required=True)
    arm.add_argument("--rng-receipt", action="append", required=True)
    arm.add_argument("--video", required=True)
    arm.add_argument("--audit-output", required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--pure-noobserver-audit", required=True)
    pair.add_argument("--observer-routeoff-audit", required=True)
    pair.add_argument("--observer-routeon-audit", required=True)
    pair.add_argument("--audit-output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "spec":
        value = validate_spec(load_spec(args.spec))
    elif args.command == "arm":
        spec = load_spec(args.spec)
        video = Path(args.video)
        native_receipt_path = Path(args.native_receipt)
        value = build_arm_audit(
            spec=spec,
            native_receipt=_load(native_receipt_path, "native receipt"),
            rng_receipts=[_load(Path(path), "RNG receipt") for path in args.rng_receipt],
            arm_role=args.arm_role,
            native_receipt_sha256=file_sha256(native_receipt_path),
            video=video,
        )
        _probe_video(video)
        _write_new_json(Path(args.audit_output), value)
    else:
        value = validate_pair_audits(
            _load(Path(args.pure_noobserver_audit), "pure no-observer arm audit"),
            _load(Path(args.observer_routeoff_audit), "observer route-off arm audit"),
            _load(Path(args.observer_routeon_audit), "observer route-on arm audit"),
        )
        _write_new_json(Path(args.audit_output), value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

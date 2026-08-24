#!/usr/bin/env python3
"""Build the fail-closed Stage-A1 s16 x early source-edge pilot review.

Inputs are the completed dog and human output directories produced by
``infer_schedule_block_source_edge_localization_v2.py`` with exactly
``--schedule-indices 16 --block-bands early``.  Before publishing anything,
the builder verifies the signed runtime receipt, registered policy and edge
contract, exact40/exact81 closure, all decoded MP4 hashes, the source-on
pre-decode parity proof, and the absence of optimizer/reward/ranking authority.

The fresh output directory contains only relative media links, copied source
and decoded MP4s, the two exact runtime receipts, a compact manifest and one
``index.html``.  It is intended to be built next to the remote run and then
copied as a directory to the local review tree.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


INPUT_RECEIPT_SCHEMA = "bernini-schedule-block-source-edge-decoded-runtime-v2"
OUTPUT_MANIFEST_SCHEMA = "bernini-s16-early-source-edge-pilot-review-v2"
METHOD = "frozen-target-query-source-kv-edge-causal-localization-v2"
STAGE = "preservation_stage_A_decoded_causal_localization"
AUTHORING_SCHEMA = "pair-v5-pure-t2v-calibration-authoring-v1"
AUTHORING_SHA256 = "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"
REGISTERED_POLICY_DIGEST = "dfac73238ad8d560bb31178d5cd0775e1a5924377a8799e7768024c2ea8a7c51"
INTERVENTION_CONTRACT_DIGEST = "a88f478f4b10e1cbf6f31b9fa2dfdd3ff0341c024437e4d3c7fb163f3dce7715"
FULL_GRID_CONTRACT_DIGEST = "af3ba9615b737d8a8f506bf532649e27c200b4927ec90dd0b173472997eb658a"
REGISTERED_GRID_SHA256 = "992dc6e59399216f7556c8a0db7faa7e8bb98d81e6b6a37d8340284232267de8"
EXACT40_SCHEDULE_SHA256 = "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
FAMILY_ORDER = ("dog", "human")
TEXT_BRANCHES = (
    "forward",
    "noop",
    "reverse",
    "incomplete",
    "camera_only",
    "appearance_only",
)
BRANCH_LABELS = {
    "forward": "forward · 目标动作",
    "noop": "noop · 不执行动作",
    "reverse": "reverse · 反向动作",
    "incomplete": "incomplete · 未完成动作",
    "camera_only": "camera-only · 仅相机变化",
    "appearance_only": "appearance-only · 仅外观变化",
}
FAMILY_BINDINGS = {
    "dog": {
        "correct_iid": "7b88a1ca1f804f41",
        "wrong_iid": "841b5e0080a1441d",
        "correct_sha256": "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed",
        "wrong_sha256": "5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a",
        "seed": 2026080825,
    },
    "human": {
        "correct_iid": "a35b590961d24694",
        "wrong_iid": "a66e6818e4144928",
        "correct_sha256": "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed",
        "wrong_sha256": "0fdc54d89250f355d2170a4d6f6aac0867abf592afb849668a8e2879a6617147",
        "seed": 2026080827,
    },
}
PILOT_SCHEDULE = 16
PILOT_BAND = "early"
PILOT_BLOCKS = tuple(range(8))
FRAME_COUNT = 81
FPS = 25
NUM_STEPS = 40
WORLD_SIZE = 4
NATIVE_BRANCH_ORDER = ("none_uncond", "V_uncond", "VI_uncond", "VI_cond")
SOURCE_BEARING_BRANCHES = NATIVE_BRANCH_ORDER[1:]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")

AUTHORITY = {
    "optimizer_present": False,
    "training_present": False,
    "reward_present": False,
    "feature_score_present": False,
    "ranking_present": False,
    "selection_present": False,
    "automatic_success_judgment_present": False,
    "method_success_claimed": False,
}


class SourceEdgePilotReviewError(RuntimeError):
    """Raised before incomplete or semantically ambiguous evidence is copied."""


def fail(message: str) -> NoReturn:
    raise SourceEdgePilotReviewError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise SourceEdgePilotReviewError("object is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(raw: Any, *, label: str) -> str:
    if type(raw) is not str or _SHA256.fullmatch(raw) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return raw


def _text(raw: Any, *, label: str) -> str:
    if type(raw) is not str or not raw.strip() or "\x00" in raw:
        fail(f"{label} must be non-empty text")
    return raw


def _mapping(raw: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        fail(f"{label} must be an object")
    return raw


def _plain_dir(raw: str | Path, *, label: str) -> Path:
    requested = Path(raw).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise SourceEdgePilotReviewError(f"missing {label}: {requested}") from error
    if resolved != requested or not resolved.is_dir():
        fail(f"{label} directory differs")
    return resolved


def _plain_file(path: Path, *, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SourceEdgePilotReviewError(f"missing {label}: {path}") from error
    if (
        resolved != path
        or resolved.is_symlink()
        or not resolved.is_file()
        or root not in resolved.parents
    ):
        fail(f"{label} must be one plain file below its run directory")
    return resolved


def _receipt_file(raw: Any, *, root: Path, expected_name: str, label: str) -> Path:
    declared = Path(_text(raw, label=f"{label} declared path"))
    expected = root / expected_name
    if not declared.is_absolute() or declared != expected:
        fail(f"{label} does not bind {expected_name}")
    return _plain_file(expected, root=root, label=label)


def _validate_signed(
    raw: Any,
    *,
    digest_field: str,
    label: str,
    expected_digest: Optional[str] = None,
) -> tuple[dict[str, Any], str]:
    value = dict(_mapping(raw, label=label))
    unsigned = dict(value)
    digest = _sha(unsigned.pop(digest_field, None), label=f"{label} digest")
    if object_sha256(unsigned) != digest:
        fail(f"{label} embedded digest differs")
    if expected_digest is not None and digest != expected_digest:
        fail(f"{label} is not the pinned registered contract")
    return value, digest


def _load_receipt(root: Path) -> tuple[dict[str, Any], Path, str, str]:
    path = _plain_file(root / "receipt.json", root=root, label="runtime receipt")
    try:
        receipt = json.loads(path.read_bytes().decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceEdgePilotReviewError("runtime receipt is not ASCII JSON") from error
    receipt, digest = _validate_signed(
        receipt, digest_field="receipt_digest", label="runtime receipt"
    )
    return receipt, path, file_sha256(path), digest


def _expected_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in TEXT_BRANCHES:
        rows.append(
            {
                "key": f"native-correct-{branch}",
                "role": "native_correct_prompt_baseline",
                "owner": "correct_owner",
                "text_branch": branch,
                "hook": "native-unhooked",
                "schedule_index": None,
                "band_name": None,
            }
        )
    rows.extend(
        (
            {
                "key": "native-wrong-owner-forward",
                "role": "compatible_wrong_owner_forward_baseline",
                "owner": "wrong_owner",
                "text_branch": "forward",
                "hook": "native-unhooked",
                "schedule_index": None,
                "band_name": None,
            },
            {
                "key": "parity-source-on-s16-early-forward",
                "role": "hooked_source_on_native_parity",
                "owner": "correct_owner",
                "text_branch": "forward",
                "hook": "source-on",
                "schedule_index": PILOT_SCHEDULE,
                "band_name": PILOT_BAND,
            },
        )
    )
    for branch in TEXT_BRANCHES:
        rows.append(
            {
                "key": f"off-s16-early-{branch}",
                "role": "source_edge_off_cell",
                "owner": "correct_owner",
                "text_branch": branch,
                "hook": "source-off",
                "schedule_index": PILOT_SCHEDULE,
                "band_name": PILOT_BAND,
            }
        )
    return rows


def _branch_captions(correct: Mapping[str, Any]) -> dict[str, str]:
    descriptions = _mapping(correct.get("branch_descriptions"), label="branch descriptions")
    scene = _text(correct.get("scene_caption"), label="scene caption").strip()
    camera = _text(correct.get("camera_caption"), label="camera caption").strip()
    captions: dict[str, str] = {}
    for branch in TEXT_BRANCHES:
        key = "action" if branch == "forward" else branch
        description = _text(descriptions.get(key), label=f"{branch} description").strip()
        captions[branch] = " ".join((scene, description, camera))
    if len(set(captions.values())) != len(TEXT_BRANCHES):
        fail("typed full instructions are not distinct")
    return captions


def _validate_policy(receipt: Mapping[str, Any]) -> None:
    policy, _ = _validate_signed(
        receipt.get("registered_schedule_block_policy"),
        digest_field="receipt_digest",
        label="registered schedule/block policy",
        expected_digest=REGISTERED_POLICY_DIGEST,
    )
    if (
        policy.get("schema_version") != "bernini-schedule-block-causal-policy-v1"
        or policy.get("registered_grid_sha256") != REGISTERED_GRID_SHA256
        or policy.get("exact40_schedule_sha256") != EXACT40_SCHEDULE_SHA256
        or policy.get("schedule_indices") != [16, 29, 35, 38]
        or policy.get("block_bands", {}).get("early") != list(PILOT_BLOCKS)
        or policy.get("cell_count") != 16
        or policy.get("num_schedule_steps") != NUM_STEPS
        or policy.get("num_transformer_blocks") != 30
        or policy.get("optimizer_authorized") is not False
        or policy.get("parameter_update_authorized") is not False
        or policy.get("method_success_claimed") is not False
        or policy.get("decoded_intervention_required") is not True
    ):
        fail("registered schedule/block policy semantics differ")
    intervention, _ = _validate_signed(
        receipt.get("intervention_contract"),
        digest_field="digest",
        label="source-edge intervention contract",
        expected_digest=INTERVENTION_CONTRACT_DIGEST,
    )
    if (
        intervention.get("method") != METHOD
        or intervention.get("attention") != "attn1-self-attention"
        or intervention.get("frame_count") != FRAME_COUNT
        or intervention.get("source_definition") != "all-non-target-visual-prefix-tokens"
        or intervention.get("target_definition") != "native-noisy-target-visual-suffix"
        or intervention.get("source_off_operation")
        != "target_queries-attend-target-KV-only;source-query-native-output-retained"
        or intervention.get("source_on_operation")
        != "delegate-exact-official-attn1-processor-object"
        or intervention.get("block_bands", {}).get(PILOT_BAND) != list(PILOT_BLOCKS)
        or intervention.get("text_branches") != list(TEXT_BRANCHES)
        or any(
            intervention.get(field) is not False
            for field in (
                "optimizer",
                "parameter_update",
                "training",
                "reward",
                "feature_scalar",
                "ranking",
                "selection",
                "token_order_or_rope_phase_changed",
            )
        )
        or intervention.get("decoded_exact81_required") is not True
    ):
        fail("source-edge intervention semantics differ")
    grid, _ = _validate_signed(
        receipt.get("full_grid_contract"),
        digest_field="digest",
        label="decoded full-grid contract",
        expected_digest=FULL_GRID_CONTRACT_DIGEST,
    )
    if (
        grid.get("schedule_block_cell_count") != 16
        or grid.get("source_off_prompt_outputs_per_cell") != len(TEXT_BRANCHES)
        or grid.get("same_seed_scheduler_gaussian_decode_within_family") is not True
        or grid.get("scalar_score_or_reward") is not False
        or grid.get("automatic_ranking_or_selection") is not False
    ):
        fail("decoded full-grid contract semantics differ")


def _validate_authority(
    receipt: Mapping[str, Any], *, family: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, str]]:
    authority = _mapping(receipt.get("authority"), label="authoring authority")
    correct = _mapping(authority.get("correct_row"), label="correct authority row")
    wrong = _mapping(authority.get("wrong_owner_row"), label="wrong-owner authority row")
    binding = FAMILY_BINDINGS[family]
    if (
        authority.get("sha256") != AUTHORING_SHA256
        or authority.get("schema_version") != AUTHORING_SCHEMA
        or authority.get("bank_id") != "pair5-t2v-first8-v1"
        or correct.get("iid") != binding["correct_iid"]
        or wrong.get("iid") != binding["wrong_iid"]
        or correct.get("analysis_split") != "fit"
        or wrong.get("analysis_split") != "confirmation"
        or correct.get("action_family_id") != wrong.get("action_family_id")
        or correct.get("seed") != binding["seed"]
    ):
        fail(f"{family} authoring authority binding differs")
    return correct, wrong, _branch_captions(correct)


def _validate_trace(
    raw: Any,
    *,
    key: str,
    hook: str,
    trace_all_rank: Mapping[str, Any],
) -> tuple[str, Optional[str]]:
    trace = dict(_mapping(raw, label=f"{key} trace"))
    steps = trace.get("steps")
    if (
        not isinstance(steps, list)
        or len(steps) != NUM_STEPS
        or [row.get("step_index") for row in steps if isinstance(row, Mapping)]
        != list(range(NUM_STEPS))
        or any(
            not isinstance(row, Mapping)
            or row.get("transformer_forward_count") != 4
            or row.get("native_formula_exact_parity") is not True
            or row.get("original_scheduler_call_count") != 1
            for row in steps
        )
        or trace.get("step_count") != NUM_STEPS
        or trace.get("observed_transformer_forwards") != 4 * NUM_STEPS
    ):
        fail(f"{key} exact40 trace semantics differ")
    native_unsigned = dict(trace)
    native_digest = _sha(native_unsigned.pop("trace_digest", None), label=f"{key} native trace")
    native_unsigned.pop("source_edge", None)
    native_unsigned.pop("source_edge_trace_digest", None)
    if object_sha256(native_unsigned) != native_digest:
        fail(f"{key} native trace digest differs")
    edge_digest: Optional[str] = None
    expected_trace_identity = native_digest
    if hook == "native-unhooked":
        if trace.get("source_edge") is not None or trace.get("source_edge_trace_digest") is not None:
            fail(f"{key} unhooked trace contains source-edge state")
    else:
        edge_receipt, edge_digest = _validate_signed(
            trace.get("source_edge"),
            digest_field="digest",
            label=f"{key} source-edge trace",
        )
        contract, contract_digest = _validate_signed(
            edge_receipt.get("contract"),
            digest_field="digest",
            label=f"{key} source-edge trace contract",
            expected_digest=INTERVENTION_CONTRACT_DIGEST,
        )
        if contract.get("method") != METHOD or contract_digest != INTERVENTION_CONTRACT_DIGEST:
            fail(f"{key} source-edge trace contract differs")
        if (
            edge_receipt.get("edge_mode") != hook
            or edge_receipt.get("registered_schedule_index") != PILOT_SCHEDULE
            or edge_receipt.get("band_name") != PILOT_BAND
            or edge_receipt.get("selected_blocks") != list(PILOT_BLOCKS)
            or edge_receipt.get("source_bearing_branches") != list(SOURCE_BEARING_BRANCHES)
            or edge_receipt.get("expected_active_calls_per_selected_block") != 3
            or edge_receipt.get("native_trace_digest") != native_digest
        ):
            fail(f"{key} source-edge coordinate differs")
        per_block = edge_receipt.get("per_block")
        if not isinstance(per_block, list) or len(per_block) != 30:
            fail(f"{key} source-edge block trace differs")
        expected_branch_calls = {name: NUM_STEPS for name in NATIVE_BRANCH_ORDER}
        expected_schedule_calls = {str(index): 4 for index in range(NUM_STEPS)}
        for index, raw_block in enumerate(per_block):
            block = _mapping(raw_block, label=f"{key} block {index} trace")
            selected = index in PILOT_BLOCKS
            expected_deleted = 3 if hook == "source-off" and selected else 0
            expected_source_on = 3 if hook == "source-on" and selected else 0
            if (
                block.get("block_index") != index
                or block.get("branch_calls") != expected_branch_calls
                or block.get("schedule_calls") != expected_schedule_calls
                or block.get("active_edge_deletion_calls") != expected_deleted
                or block.get("active_source_on_calls") != expected_source_on
                or block.get("official_delegate_calls") != 160 - expected_deleted
            ):
                fail(f"{key} block {index} call closure differs")
            geometry = block.get("last_active_geometry")
            if hook == "source-off" and selected:
                geometry = _mapping(geometry, label=f"{key} block {index} active geometry")
                total = geometry.get("total_tokens")
                source = geometry.get("source_tokens")
                target = geometry.get("target_tokens")
                if (
                    geometry.get("schedule_index") != PILOT_SCHEDULE
                    or geometry.get("band_name") != PILOT_BAND
                    or geometry.get("branch_name") != "VI_cond"
                    or type(total) is not int
                    or type(source) is not int
                    or type(target) is not int
                    or source <= 0
                    or target <= 0
                    or total != source + target
                    or geometry.get("source_query_rows_from_native_full_attention") is not True
                    or geometry.get("target_query_rows_from_target_KV_only_attention") is not True
                    or geometry.get("post_rope_token_order_unchanged") is not True
                ):
                    fail(f"{key} block {index} deleted-edge geometry differs")
            elif geometry is not None:
                fail(f"{key} block {index} unexpectedly reports deleted-edge geometry")
        expected_trace_identity = object_sha256(
            {"native": native_digest, "edge": edge_receipt}
        )
        if trace.get("source_edge_trace_digest") != expected_trace_identity:
            fail(f"{key} combined source-edge trace digest differs")
    if (
        trace_all_rank.get("all_rank_exact") is not True
        or trace_all_rank.get("value") != expected_trace_identity
    ):
        fail(f"{key} WORLD4 trace identity differs")
    return native_digest, edge_digest


def _validate_claim_closure(receipt: Mapping[str, Any]) -> None:
    interpretation = _mapping(receipt.get("interpretation"), label="interpretation")
    expected_false = (
        "hidden_or_feature_metric_authorizes_route",
        "score_computed",
        "reward_computed",
        "ranking_performed",
        "selection_performed",
        "training_performed",
        "optimizer_present",
        "backward_performed",
        "parameter_update",
        "stage_B_authorized_by_runtime_alone",
    )
    if (
        interpretation.get("decoded_complete_video_required") is not True
        or interpretation.get("manual_joint_action_and_preservation_review_pending") is not True
        or any(interpretation.get(field) is not False for field in expected_false)
    ):
        fail("runtime receipt does not close optimizer/reward/ranking authority")
    frozen = _mapping(receipt.get("frozen_model"), label="frozen-model receipt")
    prompt_guard = _mapping(frozen.get("prompt_guard"), label="prompt mutation guard")
    sampling_before = _mapping(
        frozen.get("sampling_guard_before"), label="sampling mutation guard before"
    )
    sampling_after = _mapping(
        frozen.get("sampling_guard_after"), label="sampling mutation guard after"
    )
    if (
        frozen.get("unchanged") is not True
        or prompt_guard.get("schema_version") != "bernini-model-mutation-guard-v1"
        or sampling_before.get("schema_version") != "bernini-model-mutation-guard-v1"
        or sampling_before != sampling_after
    ):
        fail("frozen model mutation guard differs")


def _validate_cell(root: Path, *, family: str) -> dict[str, Any]:
    receipt, receipt_path, receipt_file_sha, receipt_digest = _load_receipt(root)
    if (
        receipt.get("schema_version") != INPUT_RECEIPT_SCHEMA
        or receipt.get("method") != METHOD
        or receipt.get("stage") != STAGE
    ):
        fail(f"{family} runtime method/stage differs")
    _validate_policy(receipt)
    _validate_claim_closure(receipt)
    binding = FAMILY_BINDINGS[family]
    shard = _mapping(receipt.get("shard"), label="pilot shard")
    expected_plan = _expected_plan()
    if (
        shard.get("family") != family
        or shard.get("schedule_indices") != [PILOT_SCHEDULE]
        or shard.get("block_bands") != [PILOT_BAND]
        or shard.get("full_registered_grid") is not False
        or shard.get("candidate_count") != len(expected_plan)
        or shard.get("plan") != expected_plan
    ):
        fail(f"{family} output is not exactly the s16 x early pilot shard")
    correct, wrong, captions = _validate_authority(receipt, family=family)

    prompts = _mapping(receipt.get("prompts"), label="typed prompts")
    if set(prompts) != set(TEXT_BRANCHES):
        fail("typed prompt key closure differs")
    for branch in TEXT_BRANCHES:
        prompt = _mapping(prompts[branch], label=f"{branch} prompt")
        caption = captions[branch]
        if (
            prompt.get("caption") != caption
            or prompt.get("caption_utf8_sha256")
            != hashlib.sha256(caption.encode("utf-8")).hexdigest()
            or _SHA256.fullmatch(str(prompt.get("native_prompt_utf8_sha256"))) is None
        ):
            fail(f"{branch} full instruction identity differs")

    checkpoint = _mapping(receipt.get("checkpoint"), label="checkpoint")
    checkpoint_tree = _sha(checkpoint.get("tree_sha256"), label="checkpoint tree SHA")
    if checkpoint.get("opened_read_only") is not True:
        fail("checkpoint was not opened read-only")
    checkpoint_content = _mapping(
        checkpoint.get("content_identity"), label="checkpoint content identity"
    )
    runtime_source = _mapping(receipt.get("runtime_source"), label="runtime source")
    revision = runtime_source.get("revision")
    if type(revision) is not str or _SHA1.fullmatch(revision) is None:
        fail("runtime source revision must be a full lowercase SHA-1")
    runtime_closure = _sha(runtime_source.get("closure_sha256"), label="runtime closure")
    launcher_sha = _sha(runtime_source.get("launcher_sha256"), label="launcher source")

    source = _mapping(receipt.get("source"), label="source receipt")
    correct_sha = _sha(source.get("correct_sha256"), label="correct source SHA")
    wrong_sha = _sha(source.get("wrong_owner_sha256"), label="wrong source SHA")
    if (
        correct_sha != binding["correct_sha256"]
        or wrong_sha != binding["wrong_sha256"]
        or source.get("wrong_owner_same_action_family") is not True
        or source.get("wrong_owner_identity_only_control") is not False
        or source.get("scene_and_geometry_confound_acknowledged") is not True
    ):
        fail(f"{family} source role/confound closure differs")
    correct_source = _receipt_file(
        source.get("correct_snapshot"),
        root=root,
        expected_name="source-correct.mp4",
        label="correct source snapshot",
    )
    wrong_source = _receipt_file(
        source.get("wrong_owner_snapshot"),
        root=root,
        expected_name="source-wrong-owner.mp4",
        label="wrong-owner source snapshot",
    )
    if file_sha256(correct_source) != correct_sha or file_sha256(wrong_source) != wrong_sha:
        fail("source snapshot MP4 SHA differs")

    sampling = _mapping(receipt.get("sampling"), label="sampling receipt")
    shared_gaussian = _sha(
        sampling.get("shared_initial_gaussian_raw_sha256"),
        label="shared official Gaussian",
    )
    if (
        sampling.get("seed") != binding["seed"]
        or sampling.get("exact40") is not True
        or sampling.get("exact81") is not True
        or sampling.get("scheduler") != "native-UniPC-flow-shift-5"
        or sampling.get("same_initial_gaussian_all_candidates") is not True
        or sampling.get("source_on_native_parity_bit_exact") is not True
    ):
        fail(f"{family} exact40/exact81 sampling closure differs")

    plan_by_key = {row["key"]: row for row in expected_plan}
    expected_keys = set(plan_by_key)
    candidates_raw = receipt.get("candidates")
    if not isinstance(candidates_raw, list) or len(candidates_raw) != len(expected_keys):
        fail("candidate list closure differs")
    candidates: dict[str, Mapping[str, Any]] = {}
    for raw_candidate in candidates_raw:
        candidate, _ = _validate_signed(
            raw_candidate,
            digest_field="candidate_digest",
            label="decoded candidate",
        )
        key = _text(candidate.get("key"), label="candidate key")
        if key in candidates or key not in expected_keys:
            fail("candidate keys repeat or leave the pilot plan")
        plan = plan_by_key[key]
        if any(candidate.get(field) != value for field, value in plan.items()):
            fail(f"{key} candidate plan binding differs")
        if (
            candidate.get("seed") != binding["seed"]
            or candidate.get("prompt_sha256")
            != prompts[plan["text_branch"]]["native_prompt_utf8_sha256"]
            or candidate.get("initial_gaussian_raw_sha256") != shared_gaussian
            or candidate.get("score") is not None
            or candidate.get("rank") is not None
            or candidate.get("selected") is not False
        ):
            fail(f"{key} candidate authority differs")
        candidates[key] = candidate
    if set(candidates) != expected_keys:
        fail("candidate key closure differs")

    generated_identities = _mapping(
        receipt.get("generated_identities"), label="generated identities"
    )
    traces = _mapping(receipt.get("traces"), label="exact40 traces")
    outputs = _mapping(receipt.get("outputs"), label="decoded outputs")
    if set(generated_identities) != expected_keys or set(traces) != expected_keys or set(outputs) != expected_keys:
        fail("generated identity/trace/output key closure differs")
    decoded: dict[str, Any] = {}
    latent_shas: dict[str, str] = {}
    for key in plan_by_key:
        candidate = candidates[key]
        identity = _mapping(generated_identities[key], label=f"{key} generated identity")
        inner_identity = _mapping(identity.get("identity"), label=f"{key} tensor identity")
        latent_sha = _sha(
            inner_identity.get("raw_storage_sha256"), label=f"{key} pre-decode latent"
        )
        if identity.get("all_rank_exact") is not True or candidate.get("generated_identity") != identity:
            fail(f"{key} WORLD4 generated identity differs")
        latent_shas[key] = latent_sha
        gate, gate_digest = _validate_signed(
            candidate.get("trace_gate"),
            digest_field="digest",
            label=f"{key} trace gate",
        )
        hook = plan_by_key[key]["hook"]
        _, edge_digest = _validate_trace(
            traces[key],
            key=key,
            hook=hook,
            trace_all_rank=_mapping(
                candidate.get("trace_all_rank"), label=f"{key} all-rank trace"
            ),
        )
        if (
            gate.get("passed") is not True
            or gate.get("hook") != hook
            or gate.get("step_count") != NUM_STEPS
            or gate.get("transformer_forward_count") != 4 * NUM_STEPS
            or gate.get("edge_receipt_digest") != edge_digest
            or _SHA256.fullmatch(gate_digest) is None
        ):
            fail(f"{key} exact40 trace gate differs")
        output = _mapping(outputs[key], label=f"{key} decoded output")
        video_sha = _sha(output.get("sha256"), label=f"{key} MP4 SHA")
        video_path = _receipt_file(
            output.get("path"), root=root, expected_name=f"{key}.mp4", label=f"{key} MP4"
        )
        if (
            output.get("frame_count") != FRAME_COUNT
            or output.get("fps") != FPS
            or type(output.get("height")) is not int
            or output.get("height") <= 0
            or type(output.get("width")) is not int
            or output.get("width") <= 0
            or file_sha256(video_path) != video_sha
        ):
            fail(f"{key} decoded exact81 MP4 differs")
        decoded[key] = {"input_path": video_path, "sha256": video_sha}

    native_forward = "native-correct-forward"
    source_on_forward = "parity-source-on-s16-early-forward"
    parity_sha = _sha(
        sampling.get("source_on_native_parity_raw_sha256"),
        label="source-on native parity latent",
    )
    native_identity = dict(
        _mapping(
            generated_identities[native_forward].get("identity"),
            label="native forward tensor identity",
        )
    )
    source_on_identity = dict(
        _mapping(
            generated_identities[source_on_forward].get("identity"),
            label="source-on forward tensor identity",
        )
    )
    # Runtime labels deliberately contain the candidate key.  They are not
    # tensor semantics and therefore differ even for the same exact bytes.
    native_identity.pop("label", None)
    source_on_identity.pop("label", None)
    if (
        latent_shas[native_forward] != latent_shas[source_on_forward]
        or latent_shas[native_forward] != parity_sha
        or native_identity != source_on_identity
    ):
        fail("source-on forward is not bit-exact with native forward before decode")

    return {
        "family": family,
        "seed": binding["seed"],
        "cell": "s16 x early (blocks 0-7)",
        "correct_iid": correct["iid"],
        "wrong_iid": wrong["iid"],
        "captions": captions,
        "correct_source_path": correct_source,
        "correct_source_sha256": correct_sha,
        "wrong_source_path": wrong_source,
        "wrong_source_sha256": wrong_sha,
        "decoded": decoded,
        "predecode_parity_sha256": parity_sha,
        "shared_gaussian_sha256": shared_gaussian,
        "checkpoint_tree_sha256": checkpoint_tree,
        "checkpoint_content_identity": dict(checkpoint_content),
        "runtime_source_revision": revision,
        "runtime_source_closure_sha256": runtime_closure,
        "launcher_source_sha256": launcher_sha,
        "receipt_path": receipt_path,
        "receipt_file_sha256": receipt_file_sha,
        "receipt_digest": receipt_digest,
    }


def _copy_verified(source: Path, target: Path, expected_sha: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if target.is_symlink() or file_sha256(target) != expected_sha:
        fail(f"published copy differs: {target}")


def _video_card(
    *,
    title: str,
    video: str,
    sha256: str,
    instruction: str,
    note: str,
    badge: str,
) -> str:
    return f"""
      <article class="video-card">
        <div class="card-head"><h4>{html.escape(title)}</h4><span class="badge">{html.escape(badge)}</span></div>
        <video controls playsinline preload="metadata" src="{html.escape(video, quote=True)}"></video>
        <p class="instruction"><b>完整 instruction</b><span>{html.escape(instruction)}</span></p>
        <p class="note">{html.escape(note)}</p>
        <dl><div><dt>81-frame MP4 SHA-256</dt><dd><code>{sha256}</code></dd></div></dl>
      </article>"""


def _source_card(cell: Mapping[str, Any], *, wrong: bool) -> str:
    if wrong:
        title = "Wrong-owner source · 完整视频"
        local = f"cells/{cell['family']}/source-wrong-owner.mp4"
        source_iid = cell["wrong_iid"]
        digest = cell["wrong_source_sha256"]
        note = (
            "只用于 native wrong-owner forward 背景对照；它同时改变完整视频、四帧 reference、"
            "人物/动物、场景与几何，明确不是纯 identity control。"
        )
        badge = "有场景/几何混杂"
    else:
        title = "Correct source · 编辑输入"
        local = f"cells/{cell['family']}/source-correct.mp4"
        source_iid = cell["correct_iid"]
        digest = cell["correct_source_sha256"]
        note = "native、source-on 与 source-off 使用同一个 correct 完整视频和同源四帧 reference。"
        badge = "固定输入"
    return f"""
      <article class="video-card source-card">
        <div class="card-head"><h4>{html.escape(title)}</h4><span class="badge">{html.escape(badge)}</span></div>
        <video controls playsinline preload="metadata" src="{html.escape(local, quote=True)}"></video>
        <p class="note">{html.escape(note)}</p>
        <dl>
          <div><dt>source IID</dt><dd>{html.escape(str(source_iid))}</dd></div>
          <div><dt>source MP4 SHA-256</dt><dd><code>{digest}</code></dd></div>
        </dl>
      </article>"""


def render_html(cells: Sequence[Mapping[str, Any]]) -> str:
    sections: list[str] = []
    for cell in cells:
        family = str(cell["family"])
        decoded = cell["decoded"]
        native_cards = "".join(
            _video_card(
                title=BRANCH_LABELS[branch],
                video=f"cells/{family}/native-correct-{branch}.mp4",
                sha256=decoded[f"native-correct-{branch}"]["sha256"],
                instruction=cell["captions"][branch],
                note="原生 frozen RV2V；correct source；未安装 source-edge 数值干预。",
                badge="native correct-owner",
            )
            for branch in TEXT_BRANCHES
        )
        off_cards = "".join(
            _video_card(
                title=BRANCH_LABELS[branch],
                video=f"cells/{family}/off-s16-early-{branch}.mp4",
                sha256=decoded[f"off-s16-early-{branch}"]["sha256"],
                instruction=cell["captions"][branch],
                note="仅 s16、blocks 0–7 的 target-query → visual-prefix K/V edge 关闭。",
                badge="source-off · s16×early",
            )
            for branch in TEXT_BRANCHES
        )
        source_on = _video_card(
            title="forward · source-on hook parity",
            video=f"cells/{family}/parity-source-on-s16-early-forward.mp4",
            sha256=decoded["parity-source-on-s16-early-forward"]["sha256"],
            instruction=cell["captions"]["forward"],
            note=(
                "hook 已安装，但每次 attn1 调用都直接委托官方 processor；"
                "pre-decode FP32 latent 与 native forward bit-exact。"
            ),
            badge="source-on parity",
        )
        wrong_output = _video_card(
            title="forward · native wrong-owner context",
            video=f"cells/{family}/native-wrong-owner-forward.mp4",
            sha256=decoded["native-wrong-owner-forward"]["sha256"],
            instruction=cell["captions"]["forward"],
            note=(
                "同 action family 的 compatible wrong owner；完整视频与四帧 reference 都来自 wrong source，"
                "因此仅作有混杂的可视背景对照。"
            ),
            badge="native wrong-owner",
        )
        sections.append(
            f"""
            <section class="family" id="{html.escape(family, quote=True)}">
              <header class="family-head">
                <div><p class="eyebrow">{html.escape(family)} family</p><h2>Cell s16 × early · blocks 0–7</h2></div>
                <span class="pill">seed {cell['seed']} · exact40 · exact81</span>
              </header>
              <div class="instruction-hero"><h3>Forward 的完整 editing instruction</h3><p>{html.escape(cell['captions']['forward'])}</p></div>
              <div class="identity-strip">
                <span><b>correct IID</b> {html.escape(str(cell['correct_iid']))}</span>
                <span><b>wrong-owner IID</b> {html.escape(str(cell['wrong_iid']))}</span>
                <span><b>official Gaussian SHA-256</b> <code>{cell['shared_gaussian_sha256']}</code></span>
                <span><b>checkpoint tree SHA-256</b> <code>{cell['checkpoint_tree_sha256']}</code></span>
                <span><b>source-on parity latent SHA-256</b> <code>{cell['predecode_parity_sha256']}</code></span>
                <span><b>exact receipt</b> <a href="cells/{html.escape(family, quote=True)}/receipt.json">receipt.json</a> · <code>{cell['receipt_file_sha256']}</code></span>
              </div>
              <h3 class="section-title">A · Source 输入</h3>
              <div class="source-grid">{_source_card(cell, wrong=False)}{_source_card(cell, wrong=True)}</div>
              <h3 class="section-title">B · Native typed controls（correct owner）</h3>
              <p class="section-note">六个分支分别解码；每张卡完整写出对应 instruction，没有共用缩写。</p>
              <div class="six-grid">{native_cards}</div>
              <h3 class="section-title">C · Source-on forward parity</h3>
              <p class="section-note">与 B 中 native forward 使用同 seed / Gaussian / scheduler / source / references / instruction；运行时先在 pre-decode latent 上验证 bit-exact。</p>
              <div class="single-grid">{source_on}</div>
              <h3 class="section-title">D · Source-off typed controls（s16 × early）</h3>
              <div class="edge-definition"><b>真正关闭的 edge：</b>只在 denoising step index 16 与 transformer blocks 0–7，对 V_uncond、VI_uncond、VI_cond 三个含 source-prefix 的原生分支，target noisy-suffix queries 不再访问 non-target visual-prefix K/V，而只访问 target-suffix K/V。source-query rows 仍采用原生全 attention 输出；target-to-target attention 保留；none_uncond 原生不含 source prefix；其他 step、block、token 顺序、RoPE phase、文本、correct source、四帧 references、Gaussian、scheduler 与模型参数全部不变。</div>
              <div class="six-grid">{off_cards}</div>
              <h3 class="section-title">E · Wrong-owner native forward（仅背景对照）</h3>
              <div class="single-grid">{wrong_output}</div>
            </section>"""
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage-A1 · s16 × early source-edge pilot</title>
<style>
:root{{--bg:#071018;--panel:#101d29;--soft:#172837;--line:#315067;--ink:#f1f6fb;--muted:#adc0cf;--cyan:#62d9dd;--amber:#ffc66d;--red:#ff827c;--green:#76d8a4}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0,#12324a 0,#071018 35%,#050a0f 100%);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{width:min(1860px,97vw);margin:auto;padding:40px 0 80px}}h1{{font-size:clamp(34px,5vw,68px);line-height:1.02;margin:.12em 0}}h2,h3,h4{{margin:.25em 0}}.eyebrow{{color:var(--cyan);text-transform:uppercase;letter-spacing:.14em;margin:0}}.lede{{max-width:1320px;font-size:18px;color:#d3e0ea}}.contract-grid{{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:12px;margin:24px 0}}.contract-grid article,.identity-strip,.instruction-hero,.edge-definition{{border:1px solid var(--line);background:var(--soft);border-radius:12px;padding:15px}}.contract-grid h2{{font-size:18px;color:var(--cyan)}}.notice{{border-left:5px solid var(--amber);background:#20202a;padding:15px 18px;border-radius:10px}}.family{{margin:30px 0;padding:22px;border:1px solid var(--line);border-radius:18px;background:rgba(16,29,41,.97)}}.family-head{{display:flex;justify-content:space-between;gap:18px;align-items:start}}.pill,.badge{{display:inline-block;border-radius:999px;font-weight:800;white-space:nowrap}}.pill{{padding:7px 11px;color:var(--green);background:rgba(118,216,164,.12)}}.badge{{padding:4px 8px;color:var(--cyan);background:rgba(98,217,221,.1);font-size:11px}}.instruction-hero{{margin:16px 0;border-left:5px solid var(--cyan)}}.instruction-hero p{{font-size:19px;white-space:pre-wrap}}.identity-strip{{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:9px;color:var(--muted)}}code{{color:var(--cyan);overflow-wrap:anywhere;font-size:11px}}a{{color:var(--cyan)}}.section-title{{margin:30px 0 6px}}.section-note{{margin:0 0 12px;color:var(--muted)}}.source-grid{{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:12px}}.six-grid{{display:grid;grid-template-columns:repeat(6,minmax(250px,1fr));gap:12px;overflow-x:auto}}.single-grid{{display:grid;grid-template-columns:minmax(280px,420px);gap:12px}}.video-card{{min-width:250px;padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--soft)}}.card-head{{display:flex;justify-content:space-between;gap:8px;align-items:start}}video{{display:block;width:100%;aspect-ratio:1/1;max-height:520px;object-fit:contain;background:#020507;border-radius:8px;margin:9px 0}}.video-card .instruction{{display:grid;gap:4px;min-height:136px;padding:10px;background:rgba(1,7,10,.35);border-radius:8px}}.video-card .instruction span{{white-space:pre-wrap}}.note{{color:#d3dfeb;min-height:72px}}dl{{margin:0}}dl div{{padding:8px;background:rgba(1,7,10,.35);border-radius:7px}}dt{{font-size:12px;color:var(--muted)}}dd{{margin:2px 0 0;font-weight:700}}.edge-definition{{margin:10px 0 14px;border-left:5px solid var(--amber);font-size:16px}}@media(max-width:1000px){{.contract-grid,.source-grid,.identity-strip{{grid-template-columns:1fr}}.family-head{{display:block}}}}
</style></head><body><main>
<p class="eyebrow">frozen-model decoded causal localization · Stage A1</p>
<h1>s16 × early source-edge pilot</h1>
<p class="lede">本页展示 dog / human 两个预注册 family 在同一个 Stage-A1 pilot cell 的完整 81 帧视频。目标是人工判断：删除一个明确的 target-query → source visual-prefix K/V 访问边后，forward、noop、reverse、incomplete、camera-only 与 appearance-only 分支怎样变化。</p>
<section class="contract-grid">
  <article><h2>固定项</h2><p>同 family 内固定 seed、official Gaussian、exact40 UniPC flow-shift-5、81 frames、checkpoint、target geometry；correct-owner arms 固定 source、四帧 references 与逐分支完整 instruction。</p></article>
  <article><h2>干预项</h2><p>只干预 attn1 self-attention；坐标是 denoising s16 × transformer early blocks 0–7。source-on 逐调用委托官方 processor，source-off 才关闭指定 edge。</p></article>
  <article><h2>审阅方式</h2><p>观看完整视频，同时检查动作、owner/外观、场景、camera 与对象 inventory。页面只报告有明确含义的实验坐标、seed 和加密身份。</p></article>
</section>
<p class="notice">这是无训练的 frozen causal pilot。本页不含 optimizer、reward、特征分数、自动排序、选择或成败判断；运行时 receipt 单独声明 Stage B 不会由该次运行自动授权。</p>
{''.join(sections)}
</main></body></html>"""


def build(
    *, dog_output: str | Path, human_output: str | Path, output_dir: str | Path
) -> Path:
    roots = {
        "dog": _plain_dir(dog_output, label="dog output"),
        "human": _plain_dir(human_output, label="human output"),
    }
    if roots["dog"] == roots["human"]:
        fail("dog and human inputs must be distinct run directories")
    cells = [_validate_cell(roots[family], family=family) for family in FAMILY_ORDER]
    shared = {
        (
            cell["checkpoint_tree_sha256"],
            object_sha256(cell["checkpoint_content_identity"]),
            cell["runtime_source_revision"],
            cell["runtime_source_closure_sha256"],
            cell["launcher_source_sha256"],
        )
        for cell in cells
    }
    if len(shared) != 1:
        fail("dog/human runs do not share one checkpoint/runtime/launcher closure")

    target = Path(output_dir).expanduser()
    if not target.is_absolute() or target.is_symlink() or target.exists():
        fail("output directory must be one absolute fresh non-symlink path")
    target = target.absolute()
    if not target.parent.is_dir() or target.parent.is_symlink():
        fail("output parent must be an existing non-symlink directory")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        manifest_cells: list[dict[str, Any]] = []
        rendered_cells: list[dict[str, Any]] = []
        for cell in cells:
            family = cell["family"]
            local_root = staging / "cells" / family
            _copy_verified(
                cell["receipt_path"], local_root / "receipt.json", cell["receipt_file_sha256"]
            )
            _copy_verified(
                cell["correct_source_path"],
                local_root / "source-correct.mp4",
                cell["correct_source_sha256"],
            )
            _copy_verified(
                cell["wrong_source_path"],
                local_root / "source-wrong-owner.mp4",
                cell["wrong_source_sha256"],
            )
            local_outputs: dict[str, Any] = {}
            for key, item in cell["decoded"].items():
                basename = f"{key}.mp4"
                _copy_verified(item["input_path"], local_root / basename, item["sha256"])
                local_outputs[key] = {
                    "video": f"cells/{family}/{basename}",
                    "mp4_sha256": item["sha256"],
                    "text_branch": next(row["text_branch"] for row in _expected_plan() if row["key"] == key),
                    "hook": next(row["hook"] for row in _expected_plan() if row["key"] == key),
                }
            rendered_cells.append(cell)
            manifest_cells.append(
                {
                    "family": family,
                    "seed": cell["seed"],
                    "pilot_cell": {
                        "schedule_index": PILOT_SCHEDULE,
                        "block_band": PILOT_BAND,
                        "block_indices": list(PILOT_BLOCKS),
                    },
                    "full_instructions": dict(cell["captions"]),
                    "correct_source": {
                        "iid": cell["correct_iid"],
                        "video": f"cells/{family}/source-correct.mp4",
                        "mp4_sha256": cell["correct_source_sha256"],
                    },
                    "wrong_owner_source": {
                        "iid": cell["wrong_iid"],
                        "video": f"cells/{family}/source-wrong-owner.mp4",
                        "mp4_sha256": cell["wrong_source_sha256"],
                        "pure_identity_control": False,
                        "scene_and_geometry_confound_acknowledged": True,
                    },
                    "source_on_predecode_bit_exact_with_native_forward": True,
                    "source_on_predecode_latent_sha256": cell["predecode_parity_sha256"],
                    "shared_official_gaussian_sha256": cell["shared_gaussian_sha256"],
                    "outputs": local_outputs,
                    "runtime_receipt": {
                        "path": f"cells/{family}/receipt.json",
                        "file_sha256": cell["receipt_file_sha256"],
                        "embedded_digest": cell["receipt_digest"],
                    },
                }
            )
        manifest = {
            "schema_version": OUTPUT_MANIFEST_SCHEMA,
            "authority": dict(AUTHORITY),
            "experiment": {
                "method": METHOD,
                "stage": STAGE,
                "schedule_index": PILOT_SCHEDULE,
                "block_band": PILOT_BAND,
                "block_indices": list(PILOT_BLOCKS),
                "exact40": True,
                "exact81": True,
                "frame_count": FRAME_COUNT,
                "fps": FPS,
                "training": False,
            },
            "edge_explanation": {
                "attention": "attn1 self-attention",
                "active_native_branches": list(SOURCE_BEARING_BRANCHES),
                "source_off": (
                    "target noisy-suffix queries attend target-suffix K/V only at s16 "
                    "and blocks 0-7; source-query rows retain native full-attention output"
                ),
                "source_on": "delegate exact official attn1 processor on every call",
                "token_order_or_rope_phase_changed": False,
            },
            "cells": manifest_cells,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        (staging / "index.html").write_text(render_html(rendered_cells), encoding="utf-8")
        for path in (staging / "manifest.json", staging / "index.html"):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target / "index.html"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dog-output", required=True)
    parser.add_argument("--human-output", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        build(
            dog_output=args.dog_output,
            human_output=args.human_output,
            output_dir=args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

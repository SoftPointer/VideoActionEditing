#!/usr/bin/env python3
"""Build the fail-closed Stage-A1 formal source-edge grid review.

The dog and human inputs must be complete strict-v2 decoded shards with
schedules 16,29,35,38 crossed with block bands early_middle and late_middle.
Each family therefore contributes eight global baselines plus 48 source-off
videos.  The builder verifies signed receipts, the pinned policy, exact40
traces, exact81 MP4 bytes, source-on parity and the frozen/no-training claim
before atomically publishing a fresh self-contained review directory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(TOOLS_ROOT))

import build_schedule_block_source_edge_pilot_review_html_v2 as common  # noqa: E402


OUTPUT_MANIFEST_SCHEMA = "bernini-formal-source-edge-grid-review-v2"
FORMAL_SCHEDULES = (16, 29, 35, 38)
FORMAL_BANDS = ("early_middle", "late_middle")
BAND_BLOCKS = {
    "early": tuple(range(0, 8)),
    "early_middle": tuple(range(8, 16)),
    "late_middle": tuple(range(16, 23)),
}
SCHEDULE_CELLS = {
    16: {
        "timestep": 882,
        "sigma_float32_be_hex": "3f61ed37",
        "sigma_decimal": "0.8825258612632751",
    },
    29: {
        "timestep": 655,
        "sigma_float32_be_hex": "3f27d446",
        "sigma_decimal": "0.6555827856063843",
    },
    35: {
        "timestep": 418,
        "sigma_float32_be_hex": "3ed6539a",
        "sigma_decimal": "0.41860657930374146",
    },
    38: {
        "timestep": 211,
        "sigma_float32_be_hex": "3e58b351",
        "sigma_decimal": "0.21162153780460358",
    },
}
EXPECTED_CANDIDATE_COUNT = 56
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


class FormalSourceEdgeReviewError(RuntimeError):
    """Raised before an incomplete formal grid can be published."""


def fail(message: str) -> NoReturn:
    raise FormalSourceEdgeReviewError(message)


def _adapt(error: Exception) -> FormalSourceEdgeReviewError:
    return FormalSourceEdgeReviewError(str(error))


def _expected_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in common.TEXT_BRANCHES:
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
                "schedule_index": 16,
                "band_name": "early",
            },
        )
    )
    for schedule in FORMAL_SCHEDULES:
        for band in FORMAL_BANDS:
            for branch in common.TEXT_BRANCHES:
                rows.append(
                    {
                        "key": f"off-s{schedule:02d}-{band}-{branch}",
                        "role": "source_edge_off_cell",
                        "owner": "correct_owner",
                        "text_branch": branch,
                        "hook": "source-off",
                        "schedule_index": schedule,
                        "band_name": band,
                    }
                )
    if len(rows) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("formal source-edge plan constant differs")
    return rows


def _coord_for_plan(plan: Mapping[str, Any]) -> tuple[int, str, tuple[int, ...]]:
    hook = plan["hook"]
    if hook == "source-on":
        return 16, "early", BAND_BLOCKS["early"]
    if hook == "source-off":
        schedule = int(plan["schedule_index"])
        band = str(plan["band_name"])
        return schedule, band, BAND_BLOCKS[band]
    fail("unhooked plan has no source-edge coordinate")


def _validate_trace(
    raw: Any,
    *,
    key: str,
    plan: Mapping[str, Any],
    trace_all_rank: Mapping[str, Any],
) -> tuple[str, Optional[str]]:
    try:
        trace = dict(common._mapping(raw, label=f"{key} trace"))
        steps = trace.get("steps")
        if (
            not isinstance(steps, list)
            or len(steps) != common.NUM_STEPS
            or [row.get("step_index") for row in steps if isinstance(row, Mapping)]
            != list(range(common.NUM_STEPS))
            or any(
                not isinstance(row, Mapping)
                or row.get("transformer_forward_count") != 4
                or row.get("native_formula_exact_parity") is not True
                or row.get("original_scheduler_call_count") != 1
                for row in steps
            )
            or trace.get("step_count") != common.NUM_STEPS
            or trace.get("observed_transformer_forwards") != 160
        ):
            fail(f"{key} exact40 trace semantics differ")
        native_unsigned = dict(trace)
        native_digest = common._sha(
            native_unsigned.pop("trace_digest", None), label=f"{key} native trace"
        )
        native_unsigned.pop("source_edge", None)
        native_unsigned.pop("source_edge_trace_digest", None)
        if common.object_sha256(native_unsigned) != native_digest:
            fail(f"{key} native trace digest differs")

        hook = str(plan["hook"])
        edge_digest: Optional[str] = None
        trace_identity = native_digest
        if hook == "native-unhooked":
            if trace.get("source_edge") is not None or trace.get("source_edge_trace_digest") is not None:
                fail(f"{key} native trace unexpectedly contains source-edge state")
        else:
            schedule, band, selected_blocks = _coord_for_plan(plan)
            edge_receipt, edge_digest = common._validate_signed(
                trace.get("source_edge"),
                digest_field="digest",
                label=f"{key} source-edge trace",
            )
            _, contract_digest = common._validate_signed(
                edge_receipt.get("contract"),
                digest_field="digest",
                label=f"{key} source-edge contract",
                expected_digest=common.INTERVENTION_CONTRACT_DIGEST,
            )
            if (
                contract_digest != common.INTERVENTION_CONTRACT_DIGEST
                or edge_receipt.get("edge_mode") != hook
                or edge_receipt.get("registered_schedule_index") != schedule
                or edge_receipt.get("band_name") != band
                or edge_receipt.get("selected_blocks") != list(selected_blocks)
                or edge_receipt.get("source_bearing_branches")
                != list(common.SOURCE_BEARING_BRANCHES)
                or edge_receipt.get("expected_active_calls_per_selected_block") != 3
                or edge_receipt.get("native_trace_digest") != native_digest
            ):
                fail(f"{key} source-edge coordinate differs")
            per_block = edge_receipt.get("per_block")
            if not isinstance(per_block, list) or len(per_block) != 30:
                fail(f"{key} per-block trace closure differs")
            branch_calls = {
                name: common.NUM_STEPS for name in common.NATIVE_BRANCH_ORDER
            }
            schedule_calls = {str(index): 4 for index in range(common.NUM_STEPS)}
            for index, raw_block in enumerate(per_block):
                block = common._mapping(raw_block, label=f"{key} block {index}")
                selected = index in selected_blocks
                deleted = 3 if hook == "source-off" and selected else 0
                delegated_on = 3 if hook == "source-on" and selected else 0
                if (
                    block.get("block_index") != index
                    or block.get("branch_calls") != branch_calls
                    or block.get("schedule_calls") != schedule_calls
                    or block.get("active_edge_deletion_calls") != deleted
                    or block.get("active_source_on_calls") != delegated_on
                    or block.get("official_delegate_calls") != 160 - deleted
                ):
                    fail(f"{key} block {index} call closure differs")
                geometry = block.get("last_active_geometry")
                if hook == "source-off" and selected:
                    geometry = common._mapping(
                        geometry, label=f"{key} block {index} active geometry"
                    )
                    total = geometry.get("total_tokens")
                    source = geometry.get("source_tokens")
                    target = geometry.get("target_tokens")
                    if (
                        geometry.get("schedule_index") != schedule
                        or geometry.get("band_name") != band
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
                        fail(f"{key} block {index} edge geometry differs")
                elif geometry is not None:
                    fail(f"{key} block {index} unexpectedly reports deleted-edge geometry")
            trace_identity = common.object_sha256(
                {"native": native_digest, "edge": edge_receipt}
            )
            if trace.get("source_edge_trace_digest") != trace_identity:
                fail(f"{key} combined source-edge trace digest differs")
        if (
            trace_all_rank.get("all_rank_exact") is not True
            or trace_all_rank.get("value") != trace_identity
        ):
            fail(f"{key} WORLD4 trace identity differs")
        return native_digest, edge_digest
    except common.SourceEdgePilotReviewError as error:
        raise _adapt(error) from error


def _validate_cell(root: Path, *, family: str) -> dict[str, Any]:
    try:
        receipt, receipt_path, receipt_file_sha, receipt_digest = common._load_receipt(root)
        if (
            receipt.get("schema_version") != common.INPUT_RECEIPT_SCHEMA
            or receipt.get("method") != common.METHOD
            or receipt.get("stage") != common.STAGE
        ):
            fail(f"{family} runtime method/stage differs")
        common._validate_policy(receipt)
        common._validate_claim_closure(receipt)
        expected_plan = _expected_plan()
        shard = common._mapping(receipt.get("shard"), label="formal shard")
        if (
            shard.get("family") != family
            or shard.get("schedule_indices") != list(FORMAL_SCHEDULES)
            or shard.get("block_bands") != list(FORMAL_BANDS)
            or shard.get("full_registered_grid") is not False
            or shard.get("candidate_count") != EXPECTED_CANDIDATE_COUNT
            or shard.get("plan") != expected_plan
        ):
            fail(f"{family} output is not the strict formal 4x2 grid")
        correct, wrong, captions = common._validate_authority(receipt, family=family)
        binding = common.FAMILY_BINDINGS[family]

        prompts = common._mapping(receipt.get("prompts"), label="typed prompts")
        if set(prompts) != set(common.TEXT_BRANCHES):
            fail("typed prompt key closure differs")
        for branch in common.TEXT_BRANCHES:
            prompt = common._mapping(prompts[branch], label=f"{branch} prompt")
            caption = captions[branch]
            if (
                prompt.get("caption") != caption
                or prompt.get("caption_utf8_sha256")
                != hashlib.sha256(caption.encode("utf-8")).hexdigest()
                or common._SHA256.fullmatch(str(prompt.get("native_prompt_utf8_sha256")))
                is None
            ):
                fail(f"{branch} full instruction identity differs")

        checkpoint = common._mapping(receipt.get("checkpoint"), label="checkpoint")
        checkpoint_tree = common._sha(
            checkpoint.get("tree_sha256"), label="checkpoint tree SHA"
        )
        if checkpoint.get("opened_read_only") is not True:
            fail("checkpoint was not opened read-only")
        checkpoint_content = dict(
            common._mapping(
                checkpoint.get("content_identity"), label="checkpoint content identity"
            )
        )
        runtime = common._mapping(receipt.get("runtime_source"), label="runtime source")
        revision = runtime.get("revision")
        if type(revision) is not str or common._SHA1.fullmatch(revision) is None:
            fail("runtime source revision differs")
        runtime_closure = common._sha(
            runtime.get("closure_sha256"), label="runtime closure"
        )
        launcher_sha = common._sha(
            runtime.get("launcher_sha256"), label="launcher source"
        )

        source = common._mapping(receipt.get("source"), label="source receipt")
        correct_sha = common._sha(
            source.get("correct_sha256"), label="correct source SHA"
        )
        wrong_sha = common._sha(
            source.get("wrong_owner_sha256"), label="wrong source SHA"
        )
        if (
            correct_sha != binding["correct_sha256"]
            or wrong_sha != binding["wrong_sha256"]
            or source.get("wrong_owner_same_action_family") is not True
            or source.get("wrong_owner_identity_only_control") is not False
            or source.get("scene_and_geometry_confound_acknowledged") is not True
        ):
            fail(f"{family} source role/confound closure differs")
        correct_source = common._receipt_file(
            source.get("correct_snapshot"),
            root=root,
            expected_name="source-correct.mp4",
            label="correct source snapshot",
        )
        wrong_source = common._receipt_file(
            source.get("wrong_owner_snapshot"),
            root=root,
            expected_name="source-wrong-owner.mp4",
            label="wrong-owner source snapshot",
        )
        if common.file_sha256(correct_source) != correct_sha or common.file_sha256(wrong_source) != wrong_sha:
            fail("source snapshot MP4 SHA differs")

        sampling = common._mapping(receipt.get("sampling"), label="sampling receipt")
        gaussian = common._sha(
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
        raw_candidates = receipt.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) != EXPECTED_CANDIDATE_COUNT:
            fail("candidate list closure differs")
        candidates: dict[str, Mapping[str, Any]] = {}
        for raw_candidate in raw_candidates:
            candidate, _ = common._validate_signed(
                raw_candidate,
                digest_field="candidate_digest",
                label="decoded candidate",
            )
            key = common._text(candidate.get("key"), label="candidate key")
            if key in candidates or key not in expected_keys:
                fail("candidate keys repeat or leave the formal plan")
            plan = plan_by_key[key]
            if any(candidate.get(field) != value for field, value in plan.items()):
                fail(f"{key} plan binding differs")
            if (
                candidate.get("seed") != binding["seed"]
                or candidate.get("prompt_sha256")
                != prompts[plan["text_branch"]]["native_prompt_utf8_sha256"]
                or candidate.get("initial_gaussian_raw_sha256") != gaussian
                or candidate.get("score") is not None
                or candidate.get("rank") is not None
                or candidate.get("selected") is not False
            ):
                fail(f"{key} candidate authority differs")
            candidates[key] = candidate
        if set(candidates) != expected_keys:
            fail("candidate key closure differs")

        identities = common._mapping(
            receipt.get("generated_identities"), label="generated identities"
        )
        traces = common._mapping(receipt.get("traces"), label="exact40 traces")
        outputs = common._mapping(receipt.get("outputs"), label="decoded outputs")
        if set(identities) != expected_keys or set(traces) != expected_keys or set(outputs) != expected_keys:
            fail("identity/trace/output key closure differs")
        decoded: dict[str, Any] = {}
        latent_shas: dict[str, str] = {}
        for key, plan in plan_by_key.items():
            candidate = candidates[key]
            identity = common._mapping(identities[key], label=f"{key} identity")
            tensor_identity = common._mapping(
                identity.get("identity"), label=f"{key} tensor identity"
            )
            latent_sha = common._sha(
                tensor_identity.get("raw_storage_sha256"),
                label=f"{key} predecode latent",
            )
            if identity.get("all_rank_exact") is not True or candidate.get("generated_identity") != identity:
                fail(f"{key} WORLD4 generated identity differs")
            latent_shas[key] = latent_sha
            gate, _ = common._validate_signed(
                candidate.get("trace_gate"),
                digest_field="digest",
                label=f"{key} trace gate",
            )
            _, edge_digest = _validate_trace(
                traces[key],
                key=key,
                plan=plan,
                trace_all_rank=common._mapping(
                    candidate.get("trace_all_rank"), label=f"{key} all-rank trace"
                ),
            )
            if (
                gate.get("passed") is not True
                or gate.get("hook") != plan["hook"]
                or gate.get("step_count") != 40
                or gate.get("transformer_forward_count") != 160
                or gate.get("edge_receipt_digest") != edge_digest
            ):
                fail(f"{key} trace gate differs")
            output = common._mapping(outputs[key], label=f"{key} decoded output")
            video_sha = common._sha(output.get("sha256"), label=f"{key} MP4 SHA")
            video_path = common._receipt_file(
                output.get("path"),
                root=root,
                expected_name=f"{key}.mp4",
                label=f"{key} MP4",
            )
            if (
                output.get("frame_count") != 81
                or output.get("fps") != 25
                or type(output.get("height")) is not int
                or output.get("height") <= 0
                or type(output.get("width")) is not int
                or output.get("width") <= 0
                or common.file_sha256(video_path) != video_sha
            ):
                fail(f"{key} decoded exact81 MP4 differs")
            decoded[key] = {"input_path": video_path, "sha256": video_sha}

        native_key = "native-correct-forward"
        source_on_key = "parity-source-on-s16-early-forward"
        parity_sha = common._sha(
            sampling.get("source_on_native_parity_raw_sha256"),
            label="source-on parity latent",
        )
        native_identity = dict(
            common._mapping(identities[native_key]["identity"], label="native identity")
        )
        source_on_identity = dict(
            common._mapping(
                identities[source_on_key]["identity"], label="source-on identity"
            )
        )
        native_identity.pop("label", None)
        source_on_identity.pop("label", None)
        if (
            latent_shas[native_key] != latent_shas[source_on_key]
            or parity_sha != latent_shas[native_key]
            or native_identity != source_on_identity
        ):
            fail("source-on forward lost native predecode bit parity")

        return {
            "family": family,
            "seed": binding["seed"],
            "correct_iid": correct["iid"],
            "wrong_iid": wrong["iid"],
            "captions": captions,
            "correct_source_path": correct_source,
            "correct_source_sha256": correct_sha,
            "wrong_source_path": wrong_source,
            "wrong_source_sha256": wrong_sha,
            "decoded": decoded,
            "predecode_parity_sha256": parity_sha,
            "shared_gaussian_sha256": gaussian,
            "checkpoint_tree_sha256": checkpoint_tree,
            "checkpoint_content_identity": checkpoint_content,
            "runtime_source_revision": revision,
            "runtime_source_closure_sha256": runtime_closure,
            "launcher_source_sha256": launcher_sha,
            "receipt_path": receipt_path,
            "receipt_file_sha256": receipt_file_sha,
            "receipt_digest": receipt_digest,
        }
    except common.SourceEdgePilotReviewError as error:
        raise _adapt(error) from error


def _copy_verified(source: Path, target: Path, sha256: str) -> None:
    try:
        common._copy_verified(source, target, sha256)
    except common.SourceEdgePilotReviewError as error:
        raise _adapt(error) from error


def _source_card(cell: Mapping[str, Any], *, wrong: bool) -> str:
    family = cell["family"]
    if wrong:
        title = "Wrong-owner source · 完整视频"
        video = f"cells/{family}/source-wrong-owner.mp4"
        iid = cell["wrong_iid"]
        digest = cell["wrong_source_sha256"]
        note = "仅供 native wrong-owner forward 背景对照；完整视频、四帧 references、场景与几何同时变化，不是纯 identity control。"
    else:
        title = "Correct source · 实际编辑输入"
        video = f"cells/{family}/source-correct.mp4"
        iid = cell["correct_iid"]
        digest = cell["correct_source_sha256"]
        note = "native correct、source-on 与全部 source-off 使用该完整视频和同源四帧 references。"
    return f"""
      <article class="card source-card"><h4>{html.escape(title)}</h4>
        <video controls playsinline preload="metadata" src="{html.escape(video, quote=True)}"></video>
        <p>{html.escape(note)}</p><dl><div><dt>source IID</dt><dd>{html.escape(str(iid))}</dd></div><div><dt>MP4 SHA-256</dt><dd><code>{digest}</code></dd></div></dl>
      </article>"""


def _video_card(
    *, title: str, video: str, digest: str, instruction: str, role: str, note: str
) -> str:
    return f"""
      <article class="card video-card"><div class="card-head"><h4>{html.escape(title)}</h4><span class="badge">{html.escape(role)}</span></div>
        <video controls playsinline preload="metadata" src="{html.escape(video, quote=True)}"></video>
        <p class="instruction"><b>完整 instruction</b><span>{html.escape(instruction)}</span></p>
        <p class="note">{html.escape(note)}</p><dl><div><dt>81-frame MP4 SHA-256</dt><dd><code>{digest}</code></dd></div></dl>
      </article>"""


def _schedule_legend() -> str:
    rows = "".join(
        f"<tr><td><b>s{index}</b></td><td>{cell['timestep']}</td><td><code>{cell['sigma_float32_be_hex']}</code></td><td>{cell['sigma_decimal']}</td></tr>"
        for index, cell in SCHEDULE_CELLS.items()
    )
    return f"""<table><thead><tr><th>schedule index</th><th>timestep</th><th>sigma FP32 BE hex</th><th>sigma decimal</th></tr></thead><tbody>{rows}</tbody></table>"""


def render_html(cells: Sequence[Mapping[str, Any]]) -> str:
    sections: list[str] = []
    for cell in cells:
        family = cell["family"]
        decoded = cell["decoded"]
        native = "".join(
            _video_card(
                title=common.BRANCH_LABELS[branch],
                video=f"cells/{family}/native-correct-{branch}.mp4",
                digest=decoded[f"native-correct-{branch}"]["sha256"],
                instruction=cell["captions"][branch],
                role="native · correct-owner",
                note="Frozen RV2V 原生输出；没有安装 source-edge 数值干预。",
            )
            for branch in common.TEXT_BRANCHES
        )
        source_on = _video_card(
            title="forward · source-on parity",
            video=f"cells/{family}/parity-source-on-s16-early-forward.mp4",
            digest=decoded["parity-source-on-s16-early-forward"]["sha256"],
            instruction=cell["captions"]["forward"],
            role="source-on · s16×early",
            note="Hook 安装后每次都委托官方 attn1 processor；predecode latent 与 native forward bit-exact。",
        )
        wrong = _video_card(
            title="forward · wrong-owner native context",
            video=f"cells/{family}/native-wrong-owner-forward.mp4",
            digest=decoded["native-wrong-owner-forward"]["sha256"],
            instruction=cell["captions"]["forward"],
            role="native · wrong-owner",
            note="同 action family，但完整 source 与四帧 references 都换成 wrong owner；包含场景/几何混杂。",
        )
        grid_sections: list[str] = []
        for schedule in FORMAL_SCHEDULES:
            schedule_row = SCHEDULE_CELLS[schedule]
            for band in FORMAL_BANDS:
                blocks = BAND_BLOCKS[band]
                cards = "".join(
                    _video_card(
                        title=common.BRANCH_LABELS[branch],
                        video=f"cells/{family}/off-s{schedule:02d}-{band}-{branch}.mp4",
                        digest=decoded[f"off-s{schedule:02d}-{band}-{branch}"]["sha256"],
                        instruction=cell["captions"][branch],
                        role=f"source-off · s{schedule}×{band}",
                        note=(
                            f"只在 schedule s{schedule}、blocks {blocks[0]}–{blocks[-1]} 关闭 target-query → source-prefix K/V edge。"
                        ),
                    )
                    for branch in common.TEXT_BRANCHES
                )
                grid_sections.append(
                    f"""
                    <section class="grid-cell">
                      <header><div><p class="eyebrow">formal cell</p><h3>s{schedule} × {html.escape(band)} · blocks {blocks[0]}–{blocks[-1]}</h3></div><div class="sigma"><b>timestep {schedule_row['timestep']}</b><span>sigma {schedule_row['sigma_decimal']}</span><code>{schedule_row['sigma_float32_be_hex']}</code></div></header>
                      <div class="six-grid">{cards}</div>
                    </section>"""
                )
        sections.append(
            f"""
            <section class="family" id="{html.escape(family, quote=True)}">
              <header class="family-head"><div><p class="eyebrow">{html.escape(family)} family</p><h2>Strict 4 schedules × 2 block bands</h2></div><span class="pill">seed {cell['seed']} · 56 outputs · exact40 / exact81</span></header>
              <div class="instruction-hero"><h3>Forward 的完整 editing instruction</h3><p>{html.escape(cell['captions']['forward'])}</p></div>
              <div class="identity-strip"><span><b>correct IID</b> {html.escape(str(cell['correct_iid']))}</span><span><b>wrong IID</b> {html.escape(str(cell['wrong_iid']))}</span><span><b>Gaussian SHA-256</b> <code>{cell['shared_gaussian_sha256']}</code></span><span><b>checkpoint tree SHA-256</b> <code>{cell['checkpoint_tree_sha256']}</code></span><span><b>source-on parity latent SHA-256</b> <code>{cell['predecode_parity_sha256']}</code></span><span><b>receipt</b> <a href="cells/{html.escape(family, quote=True)}/receipt.json">exact JSON</a> · <code>{cell['receipt_file_sha256']}</code></span></div>
              <h3 class="section-title">Source 输入</h3><div class="source-grid">{_source_card(cell, wrong=False)}{_source_card(cell, wrong=True)}</div>
              <h3 class="section-title">Global baselines · 每个 family 共 8 个</h3><p class="section-note">六个 native correct-owner typed controls，加一个 source-on forward 数值 parity，以及一个有混杂的 wrong-owner native forward。每个输出只展示一次。</p>
              <div class="six-grid">{native}</div><div class="two-grid">{source_on}{wrong}</div>
              <h3 class="section-title">Formal source-off grid · 8 cells × 6 typed controls</h3>
              {''.join(grid_sections)}
            </section>"""
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage-A1 · formal source-edge grid</title><style>
:root{{--bg:#061018;--panel:#101e2b;--soft:#172a39;--line:#31546b;--ink:#f2f7fb;--muted:#adc1cf;--cyan:#61d9df;--amber:#ffca70;--green:#78d9a5}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#12334d,#071018 38%,#050a0f);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{width:min(1900px,97vw);margin:auto;padding:38px 0 80px}}h1{{font-size:clamp(34px,5vw,68px);line-height:1.03;margin:.12em 0}}h2,h3,h4{{margin:.25em 0}}.eyebrow{{margin:0;color:var(--cyan);text-transform:uppercase;letter-spacing:.14em}}.lede{{max-width:1350px;color:#d4e1ea;font-size:18px}}.first-screen{{min-height:90vh}}.legend-grid{{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:11px;margin:22px 0}}.legend-grid article,.schedule-panel,.band-panel,.identity-strip,.instruction-hero,.edge-note{{border:1px solid var(--line);border-radius:12px;background:var(--soft);padding:14px}}.legend-grid h2{{font-size:17px;color:var(--cyan)}}.axis-grid{{display:grid;grid-template-columns:2fr 1fr;gap:12px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:var(--muted);font-size:12px}}code{{color:var(--cyan);overflow-wrap:anywhere;font-size:11px}}.edge-note{{margin-top:12px;border-left:5px solid var(--amber)}}.family{{margin:30px 0;padding:22px;border:1px solid var(--line);border-radius:18px;background:rgba(16,30,43,.97)}}.family-head,.grid-cell>header{{display:flex;justify-content:space-between;gap:16px;align-items:start}}.pill,.badge{{display:inline-block;border-radius:999px;white-space:nowrap;font-weight:800}}.pill{{padding:7px 11px;color:var(--green);background:rgba(120,217,165,.12)}}.badge{{padding:4px 8px;color:var(--cyan);background:rgba(97,217,223,.1);font-size:11px}}.instruction-hero{{margin:16px 0;border-left:5px solid var(--cyan)}}.instruction-hero p{{font-size:19px;white-space:pre-wrap}}.identity-strip{{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:9px;color:var(--muted)}}a{{color:var(--cyan)}}.section-title{{margin:30px 0 6px}}.section-note{{color:var(--muted)}}.source-grid,.two-grid{{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:12px}}.six-grid{{display:grid;grid-template-columns:repeat(6,minmax(250px,1fr));gap:12px;overflow-x:auto}}.card{{min-width:250px;padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--soft)}}.card-head{{display:flex;justify-content:space-between;gap:8px;align-items:start}}video{{display:block;width:100%;aspect-ratio:1/1;max-height:520px;object-fit:contain;background:#020507;border-radius:8px;margin:9px 0}}.instruction{{display:grid;gap:4px;min-height:136px;padding:10px;background:rgba(1,7,10,.35);border-radius:8px}}.instruction span{{white-space:pre-wrap}}.note{{min-height:72px;color:#d3dfe8}}dl{{margin:0}}dl div{{padding:8px;background:rgba(1,7,10,.35);border-radius:7px}}dt{{font-size:12px;color:var(--muted)}}dd{{margin:2px 0 0;font-weight:700}}.grid-cell{{margin:16px 0;padding:15px;border:1px solid var(--line);border-radius:14px;background:#0d1924}}.sigma{{display:grid;text-align:right;color:var(--muted)}}@media(max-width:1100px){{.legend-grid{{grid-template-columns:1fr 1fr}}.axis-grid,.source-grid,.two-grid,.identity-strip{{grid-template-columns:1fr}}.family-head,.grid-cell>header{{display:block}}}}@media(max-width:680px){{.legend-grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="first-screen"><p class="eyebrow">frozen decoded causal localization · Stage A1</p><h1>Formal source-edge grid</h1><p class="lede">Dog 与 human 各 56 个完整 81 帧输出：8 个全局 baseline，加上 4 个 schedule × 2 个 block band × 6 个 typed control。所有 cell 使用同一 frozen checkpoint；不存在训练或参数更新。</p>
<section class="legend-grid">
  <article><h2>Source 是什么</h2><p>Correct source 是完整编辑输入视频，并独立抽取 4 帧作为 image references。Wrong-owner source 只进入单独背景对照，且同时带来场景与几何混杂。</p></article>
  <article><h2>Full instruction 是什么</h2><p>完整 scene caption + 当前 typed action/control 描述 + camera caption。每张结果卡原样显示对应完整文本，不用缩写，也不只写动作标签。</p></article>
  <article><h2>Schedule index / sigma</h2><p>s16、s29、s35、s38 是 native exact40 去噪轨迹中的离散调用位置；sigma 是该位置的固定噪声尺度。它不是视频帧，也不是质量分数。</p></article>
  <article><h2>Block band 是什么</h2><p>early_middle = transformer blocks 8–15；late_middle = blocks 16–22。Schedule 轴与网络深度轴彼此独立。</p></article>
  <article><h2>Native</h2><p>原生 frozen RV2V，没有 source-edge 数值干预。六种 correct-owner typed prompts 各自解码一次。</p></article>
  <article><h2>Source-on</h2><p>安装 hook 但逐调用委托官方 attn1 processor；固定的 s16×early forward 只用于证明 hook 不改变 native predecode latent。</p></article>
  <article><h2>Source-off</h2><p>只在指定 schedule×band，让 noisy-target queries 不再访问 source visual-prefix K/V；target-to-target 与 source-query 原生输出保留。</p></article>
  <article><h2>Wrong-owner</h2><p>同 action family 的另一个 owner；完整视频与 references 一起替换，因此仅作有混杂的背景对照，不解释为纯 identity effect。</p></article>
</section>
<div class="axis-grid"><section class="schedule-panel"><h2>预注册 schedule / sigma</h2>{_schedule_legend()}</section><section class="band-panel"><h2>本次 formal bands</h2><p><b>early_middle</b><br>blocks 8–15</p><p><b>late_middle</b><br>blocks 16–22</p></section></div>
<p class="edge-note"><b>Source-off 真正删除的边：</b>在选定 schedule 与 blocks 内，仅对 V_uncond、VI_uncond、VI_cond 三个含 source-prefix 的 native branches，target noisy-suffix queries 改为只访问 target-suffix K/V。其他 schedule/block、none_uncond、文本、source、references、Gaussian、scheduler、token 顺序、RoPE phase 与模型参数保持不变。本页不展示特征总分、reward、排序、选择或自动成败判断。</p></section>
{''.join(sections)}</main></body></html>"""


def build(
    *, dog_output: str | Path, human_output: str | Path, output_dir: str | Path
) -> Path:
    try:
        roots = {
            "dog": common._plain_dir(dog_output, label="dog output"),
            "human": common._plain_dir(human_output, label="human output"),
        }
    except common.SourceEdgePilotReviewError as error:
        raise _adapt(error) from error
    if roots["dog"] == roots["human"]:
        fail("dog and human inputs must be distinct run directories")
    cells = [_validate_cell(roots[family], family=family) for family in common.FAMILY_ORDER]
    shared = {
        (
            cell["checkpoint_tree_sha256"],
            common.object_sha256(cell["checkpoint_content_identity"]),
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
        rendered: list[dict[str, Any]] = []
        manifest_cells: list[dict[str, Any]] = []
        plan = _expected_plan()
        plan_by_key = {row["key"]: row for row in plan}
        for cell in cells:
            family = cell["family"]
            local = staging / "cells" / family
            _copy_verified(
                cell["receipt_path"], local / "receipt.json", cell["receipt_file_sha256"]
            )
            _copy_verified(
                cell["correct_source_path"],
                local / "source-correct.mp4",
                cell["correct_source_sha256"],
            )
            _copy_verified(
                cell["wrong_source_path"],
                local / "source-wrong-owner.mp4",
                cell["wrong_source_sha256"],
            )
            local_outputs: dict[str, Any] = {}
            for key, item in cell["decoded"].items():
                _copy_verified(item["input_path"], local / f"{key}.mp4", item["sha256"])
                row = plan_by_key[key]
                local_outputs[key] = {
                    "video": f"cells/{family}/{key}.mp4",
                    "mp4_sha256": item["sha256"],
                    "text_branch": row["text_branch"],
                    "hook": row["hook"],
                    "schedule_index": row["schedule_index"],
                    "block_band": row["band_name"],
                }
            rendered.append(cell)
            manifest_cells.append(
                {
                    "family": family,
                    "seed": cell["seed"],
                    "candidate_count": EXPECTED_CANDIDATE_COUNT,
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
                    "source_on_predecode_latent_sha256": cell[
                        "predecode_parity_sha256"
                    ],
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
                "method": common.METHOD,
                "stage": common.STAGE,
                "schedule_indices": list(FORMAL_SCHEDULES),
                "schedule_cells": SCHEDULE_CELLS,
                "block_bands": {name: list(BAND_BLOCKS[name]) for name in FORMAL_BANDS},
                "baseline_count_per_family": 8,
                "source_off_cell_count_per_family": 8,
                "source_off_outputs_per_cell": 6,
                "candidate_count_per_family": EXPECTED_CANDIDATE_COUNT,
                "exact40": True,
                "exact81": True,
                "training": False,
            },
            "cells": manifest_cells,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        (staging / "index.html").write_text(render_html(rendered), encoding="utf-8")
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

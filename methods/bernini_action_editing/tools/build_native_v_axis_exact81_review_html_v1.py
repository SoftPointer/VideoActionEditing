#!/usr/bin/env python3
"""Build a self-contained, fail-closed Stage-A0 native V-axis review.

The two inputs are the completed dog and human output directories from
``infer_native_v_axis_exact81_probe_v1.py``.  The builder verifies their signed
receipts and required MP4 bytes, then publishes a fresh review directory with
only local media, exact receipts, a compact manifest and ``index.html``.

This is a causal visualization packet, not an evaluator.  It has no scalar,
reward, ranking, selection or automatic/manual success verdict.
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


INPUT_RECEIPT_SCHEMA = "bernini-native-v-axis-exact81-probe-receipt-v1"
OUTPUT_MANIFEST_SCHEMA = "bernini-native-v-axis-exact81-review-manifest-v1"
METHOD = "frozen-bernini-native-full-video-axis-causal-probe-v1"
STAGE = "stage_A0_native_full_video_axis_exact40_exact81"
CELL_ORDER = ("dog", "human")
ARM_ORDER = ("V-on", "V-off", "wrong-V")
NATIVE_FORMULA = "vN=v0+1.25*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
V_OFF_FORMULA = "vOff=v0+0.0*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
AUTHORITY = {
    "feature_scalar_present": False,
    "reward_present": False,
    "ranking_present": False,
    "selection_present": False,
    "automatic_verdict_present": False,
    "manual_verdict_present": False,
    "method_success_claimed": False,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class NativeVAxisReviewError(RuntimeError):
    """Raised before incomplete or overclaiming evidence is published."""


def fail(message: str) -> NoReturn:
    raise NativeVAxisReviewError(message)


def canonical_json_bytes(raw: Any) -> bytes:
    try:
        return json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise NativeVAxisReviewError("object is not canonical JSON") from error


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
    resolved = requested.resolve(strict=True)
    if resolved != requested or not resolved.is_dir():
        fail(f"{label} directory differs")
    return resolved


def _plain_file(path: Path, *, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise NativeVAxisReviewError(f"missing {label}: {path}") from error
    if (
        resolved != path
        or resolved.is_symlink()
        or not resolved.is_file()
        or root not in resolved.parents
    ):
        fail(f"{label} must be one plain file below its output directory")
    return resolved


def _receipt_file(raw: Any, *, root: Path, expected_name: str, label: str) -> Path:
    declared = Path(_text(raw, label=f"{label} receipt path"))
    expected = root / expected_name
    if not declared.is_absolute() or declared != expected:
        fail(f"{label} receipt path does not bind {expected_name}")
    return _plain_file(expected, root=root, label=label)


def _load_signed_receipt(root: Path) -> tuple[dict[str, Any], Path, str, str]:
    path = _plain_file(root / "receipt.json", root=root, label="receipt")
    try:
        raw = path.read_bytes()
        receipt = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeVAxisReviewError("cannot read receipt") from error
    if type(receipt) is not dict:
        fail("receipt must contain one JSON object")
    unsigned = dict(receipt)
    digest = _sha(unsigned.pop("receipt_digest", None), label="receipt digest")
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != digest:
        fail("receipt embedded digest differs")
    return receipt, path, file_sha256(path), digest


def _candidate_key(seed: int, arm: str) -> str:
    return f"seed-{seed}__{arm}"


def _validate_claim_closure(receipt: Mapping[str, Any]) -> None:
    interpretation = _mapping(
        receipt.get("interpretation"), label="receipt interpretation"
    )
    required_false = (
        "training_performed",
        "trainer_instantiated",
        "backward",
        "model_weights_written",
        "adapter_loaded",
        "target_video",
        "feature_scorer_consumed",
        "reward_computed",
        "score_computed",
        "ranking_performed",
        "best_arm_selected",
        "visual_selection_performed",
        "action_success_evaluated",
        "preservation_success_evaluated",
        "scientific_claim_authorized_before_blind_review",
    )
    if any(interpretation.get(field) is not False for field in required_false):
        fail("receipt does not close scorer/reward/ranking/verdict authority")
    if (
        interpretation.get("optimizer") is not None
        or interpretation.get("wrong_V_changes_only_full_video_condition") is not True
        or interpretation.get("V_off_zeros_only_standalone_vV_minus_v0_coefficient")
        is not True
        or interpretation.get("V_off_retains_vVIu_minus_vV_term") is not True
    ):
        fail("receipt intervention/optimizer closure differs")


def _validate_arm_contract(raw: Any, *, arm: str) -> dict[str, Any]:
    contract = dict(_mapping(raw, label=f"{arm} arm contract"))
    expected_video_role = "wrong" if arm == "wrong-V" else "correct"
    expected_omega = 0.0 if arm == "V-off" else 1.25
    expected_intervention = {
        "V-on": "native_no_numerical_intervention",
        "V-off": "zero_standalone_vV_minus_v0_coefficient_only",
        "wrong-V": "replace_full_video_condition_only",
    }[arm]
    if (
        contract.get("arm") != arm
        or contract.get("full_video_condition_role") != expected_video_role
        or contract.get("omega_video") != expected_omega
        or contract.get("omega_image") != 4.5
        or contract.get("omega_text") != 4.0
        or contract.get("correct_image_references") is not True
        or contract.get("same_instruction") is not True
        or contract.get("same_scheduler") is not True
        or contract.get("same_target_geometry") is not True
        or contract.get("intervention") != expected_intervention
        or contract.get("v_vi_u_minus_v_v_term_retained") is not True
    ):
        fail(f"{arm} intervention contract differs")
    return contract


def _validate_cell(root: Path, *, expected_cell_id: str) -> dict[str, Any]:
    receipt, receipt_path, receipt_sha, receipt_digest = _load_signed_receipt(root)
    if (
        receipt.get("schema_version") != INPUT_RECEIPT_SCHEMA
        or receipt.get("method") != METHOD
        or receipt.get("stage") != STAGE
    ):
        fail(f"{expected_cell_id} receipt method/stage differs")
    _validate_claim_closure(receipt)
    cell_spec = _mapping(receipt.get("cell_spec"), label="cell spec")
    contract = _mapping(cell_spec.get("contract"), label="cell spec contract")
    cell = _mapping(cell_spec.get("cell"), label="cell spec cell")
    if (
        cell.get("cell_id") != expected_cell_id
        or cell.get("actor_kind") != expected_cell_id
        or contract.get("method") != METHOD
        or contract.get("frame_count") != 81
        or contract.get("latent_phases") != 21
        or contract.get("fps") != 25
        or contract.get("num_inference_steps") != 40
        or contract.get("guidance_mode") != "rv2v"
        or contract.get("native_velocity_formula") != NATIVE_FORMULA
        or contract.get("v_off_velocity_formula") != V_OFF_FORMULA
        or contract.get("arm_order") != list(ARM_ORDER)
        or contract.get("wrong_v_replaces_full_video_condition_only") is not True
        or contract.get("wrong_v_keeps_correct_image_references_and_text") is not True
        or contract.get("training") is not False
        or contract.get("optimizer") is not False
        or contract.get("feature_scorer") is not False
        or contract.get("reward") is not False
        or contract.get("ranking") is not False
        or contract.get("selection") is not False
    ):
        fail(f"{expected_cell_id} registered cell contract differs")
    source_iid = _text(cell.get("source_iid"), label="correct source IID")
    wrong_source_iid = _text(cell.get("wrong_source_iid"), label="wrong source IID")
    if source_iid == wrong_source_iid:
        fail("correct and wrong source IID must differ")
    instruction = _text(cell.get("action_caption"), label="editing instruction")
    instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    if (
        cell.get("action_caption_utf8_sha256") != instruction_sha
        or cell.get("selected_before_generation") is not True
    ):
        fail("editing instruction identity differs")
    prompt = _mapping(receipt.get("prompt"), label="prompt receipt")
    if (
        prompt.get("action_caption") != instruction
        or prompt.get("action_caption_utf8_sha256") != instruction_sha
        or prompt.get("same_across_all_arms_and_seeds") is not True
    ):
        fail("receipt does not bind the full editing instruction")

    sampling = _mapping(receipt.get("sampling"), label="sampling receipt")
    seeds = cell.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 2
        or len(set(seeds)) != 2
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or sampling.get("seeds") != seeds
        or sampling.get("exact40") is not True
        or sampling.get("exact81") is not True
        or sampling.get("frame_count") != 81
        or sampling.get("latent_phases") != 21
        or sampling.get("fps") != 25
        or sampling.get("num_inference_steps") != 40
        or sampling.get("arm_order") != list(ARM_ORDER)
        or sampling.get("same_official_gaussian_within_seed") is not True
        or sampling.get("same_x_t_t_target_geometry_within_seed") is not True
    ):
        fail(f"{expected_cell_id} exact40/exact81 sampling differs")
    hook = _mapping(sampling.get("hook_contract"), label="V-axis hook contract")
    if (
        hook.get("native_formula")
        != "v0+1.25*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
        or hook.get("v_off_formula")
        != "v0+0.0*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
        or hook.get("transformer_forwards_per_step") != 4
        or hook.get("original_unipc_calls_per_step") != 1
        or hook.get("training") is not False
        or hook.get("optimizer") is not False
        or hook.get("feature_scorer") is not False
        or hook.get("selection") is not False
    ):
        fail("V-axis hook formula/execution closure differs")

    checkpoint = _mapping(receipt.get("checkpoint"), label="checkpoint receipt")
    checkpoint_sha = _sha(checkpoint.get("tree_sha256"), label="checkpoint tree SHA")
    if checkpoint.get("opened_read_only") is not True:
        fail("base checkpoint was not opened read-only")
    runtime_source = _mapping(receipt.get("runtime_source"), label="runtime source")
    revision = runtime_source.get("revision")
    if type(revision) is not str or _REVISION.fullmatch(revision) is None:
        fail("runtime source revision differs")

    correct = _mapping(receipt.get("correct_source"), label="correct source")
    wrong = _mapping(receipt.get("wrong_V_source"), label="wrong-V source")
    correct_sha = _sha(correct.get("sha256"), label="correct source SHA")
    wrong_sha = _sha(wrong.get("sha256"), label="wrong source SHA")
    correct_path = _receipt_file(
        correct.get("snapshot_mp4"),
        root=root,
        expected_name="source-correct.mp4",
        label="correct source snapshot",
    )
    wrong_path = _receipt_file(
        wrong.get("snapshot_mp4"),
        root=root,
        expected_name="source-wrong-V.mp4",
        label="wrong source snapshot",
    )
    if file_sha256(correct_path) != correct_sha or file_sha256(wrong_path) != wrong_sha:
        fail("source snapshot SHA differs")
    geometry_confound = cell.get("wrong_source_geometry_confound")
    if (
        type(geometry_confound) is not bool
        or wrong.get("geometry_confound_present") is not geometry_confound
        or wrong.get("used_only_as_full_video_condition_in_wrong_V") is not True
        or wrong.get("used_as_image_reference") is not False
        or wrong.get("pure_identity_control") is not False
        or cell.get("wrong_source_pure_identity_control") is not False
    ):
        fail("wrong-V source role/confound closure differs")

    outputs = _mapping(receipt.get("outputs"), label="decoded outputs")
    candidates = receipt.get("candidates")
    if not isinstance(candidates, list):
        fail("candidate rows are missing")
    candidate_index: dict[str, Mapping[str, Any]] = {}
    for raw_candidate in candidates:
        candidate = _mapping(raw_candidate, label="candidate")
        unsigned_candidate = dict(candidate)
        candidate_digest = _sha(
            unsigned_candidate.pop("candidate_receipt_digest", None),
            label="candidate receipt digest",
        )
        if hashlib.sha256(canonical_json_bytes(unsigned_candidate)).hexdigest() != candidate_digest:
            fail("candidate embedded digest differs")
        key = _text(candidate.get("candidate_key"), label="candidate key")
        if key in candidate_index:
            fail("candidate key repeats")
        candidate_index[key] = candidate
    expected_keys = {_candidate_key(seed, arm) for seed in seeds for arm in ARM_ORDER}
    if set(outputs) != expected_keys or set(candidate_index) != expected_keys:
        fail("decoded output/candidate closure differs")

    traces = _mapping(receipt.get("traces"), label="exact40 traces")
    if set(traces) != expected_keys:
        fail("exact40 trace key closure differs")
    seed_rows: list[dict[str, Any]] = []
    gaussian_by_seed: dict[int, set[str]] = {seed: set() for seed in seeds}
    for seed in seeds:
        arms: dict[str, Any] = {}
        for arm in ARM_ORDER:
            key = _candidate_key(seed, arm)
            candidate = candidate_index[key]
            arm_contract = _validate_arm_contract(
                candidate.get("arm_contract"), arm=arm
            )
            gate = _mapping(
                candidate.get("exact40_trace_gate"), label="exact40 trace gate"
            )
            trace = _mapping(traces[key], label="exact40 trace")
            if (
                candidate.get("seed") != seed
                or candidate.get("arm") != arm
                or candidate.get("score") is not None
                or candidate.get("rank") is not None
                or candidate.get("selected") is not False
                or gate.get("passed") is not True
                or gate.get("arm") != arm
                or gate.get("step_count") != 40
                or gate.get("four_native_branch_calls_per_step") is not True
                or gate.get("one_original_unipc_call_per_step") is not True
                or trace.get("step_count") != 40
                or trace.get("trace_digest") != candidate.get("trace_digest")
            ):
                fail(f"{expected_cell_id} seed {seed} {arm} exact40 closure differs")
            gaussian_by_seed[seed].add(
                _sha(
                    candidate.get("official_initial_gaussian_raw_value_sha256"),
                    label="official Gaussian SHA",
                )
            )
            output = _mapping(outputs[key], label=f"{key} decoded output")
            video_sha = _sha(output.get("sha256"), label=f"{key} MP4 SHA")
            video_path = _receipt_file(
                output.get("path"),
                root=root,
                expected_name=f"{key}.mp4",
                label=f"{key} MP4",
            )
            if (
                output.get("frame_count") != 81
                or output.get("fps") != 25
                or file_sha256(video_path) != video_sha
            ):
                fail(f"{key} decoded MP4 differs")
            arms[arm] = {
                "input_path": video_path,
                "sha256": video_sha,
                "formula": V_OFF_FORMULA if arm == "V-off" else NATIVE_FORMULA,
                "contract": arm_contract,
                "trace_digest": candidate["trace_digest"],
            }
        if len(gaussian_by_seed[seed]) != 1:
            fail(f"seed {seed} arms do not share one official Gaussian")
        seed_rows.append(
            {
                "seed": seed,
                "exact40": True,
                "official_gaussian_sha256": next(iter(gaussian_by_seed[seed])),
                "arms": arms,
            }
        )
    if len({next(iter(raw)) for raw in gaussian_by_seed.values()}) != len(seeds):
        fail("sealed seeds unexpectedly share one official Gaussian")
    return {
        "cell_id": expected_cell_id,
        "source_iid": source_iid,
        "wrong_source_iid": wrong_source_iid,
        "instruction": instruction,
        "instruction_sha256": instruction_sha,
        "checkpoint_tree_sha256": checkpoint_sha,
        "runtime_source_revision": revision,
        "cell_spec_file_sha256": _sha(
            cell_spec.get("file_sha256"), label="cell spec file SHA"
        ),
        "receipt_path": receipt_path,
        "receipt_file_sha256": receipt_sha,
        "receipt_digest": receipt_digest,
        "correct_source_path": correct_path,
        "correct_source_sha256": correct_sha,
        "wrong_source_path": wrong_path,
        "wrong_source_sha256": wrong_sha,
        "wrong_source_geometry_confound": geometry_confound,
        "wrong_source_pure_identity_control": False,
        "seeds": seed_rows,
    }


def _copy_verified(source: Path, target: Path, expected_sha: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if target.is_symlink() or file_sha256(target) != expected_sha:
        fail(f"published copy differs: {target}")


def _source_card(*, cell: Mapping[str, Any], wrong: bool) -> str:
    if wrong:
        title = "Wrong-V source · 完整视频"
        filename = f"cells/{cell['cell_id']}/source-wrong-V.mp4"
        source_iid = cell["wrong_source_iid"]
        source_sha = cell["wrong_source_sha256"]
        confound = cell["wrong_source_geometry_confound"]
        badge = (
            '<span class="badge danger">geometry confound: PRESENT</span>'
            if confound
            else '<span class="badge caution">geometry confound: absent</span>'
        )
        note = (
            "仅在 wrong-V arm 中替换完整视频 V condition；正确 source 的 4 个 image refs、"
            "完整 instruction、seed、scheduler 与 target geometry 不变。它明确不是 pure identity control。"
        )
    else:
        title = "Correct source · 完整编辑输入"
        filename = f"cells/{cell['cell_id']}/source-correct.mp4"
        source_iid = cell["source_iid"]
        source_sha = cell["correct_source_sha256"]
        badge = '<span class="badge stable">编辑前 source</span>'
        note = "V-on 与 V-off 使用此完整视频作为 V condition；三个 arms 的 4 个 image refs 都来自此 source。"
    return f"""
      <article class="video-card source-card">
        <h4>{html.escape(title)}</h4>{badge}
        <video controls playsinline preload="metadata" src="{html.escape(filename, quote=True)}"></video>
        <p>{html.escape(note)}</p>
        <dl><div><dt>source IID</dt><dd>{html.escape(str(source_iid))}</dd></div><div><dt>source MP4 SHA-256</dt><dd><code>{source_sha}</code></dd></div></dl>
      </article>"""


def _arm_card(*, arm: str, item: Mapping[str, Any], cell: Mapping[str, Any]) -> str:
    if arm == "V-on":
        title = "V-on / native"
        unique = "不做数值干预；使用 correct full-video V。"
        source_sha = cell["correct_source_sha256"]
    elif arm == "V-off":
        title = "V-off"
        unique = "唯一变化：standalone (vV−v0) 系数从 1.25 置为 0；其余两项保留。"
        source_sha = cell["correct_source_sha256"]
    else:
        title = "Wrong-V"
        unique = "唯一条件替换：full-video V 改为 wrong source；correct image refs 与 instruction 不变。"
        source_sha = cell["wrong_source_sha256"]
    return f"""
      <article class="video-card arm-card">
        <h4>{html.escape(title)}</h4>
        <video controls playsinline preload="metadata" src="{html.escape(item['local_video'], quote=True)}"></video>
        <p class="unique"><b>唯一变量：</b>{html.escape(unique)}</p>
        <dl>
          <div><dt>arm 公式</dt><dd><code>{html.escape(item['formula'])}</code></dd></div>
          <div><dt>本 arm full-video source SHA-256</dt><dd><code>{source_sha}</code></dd></div>
          <div><dt>完整 81f MP4 SHA-256</dt><dd><code>{item['sha256']}</code></dd></div>
        </dl>
      </article>"""


def render_html(cells: Sequence[Mapping[str, Any]]) -> str:
    sections: list[str] = []
    for cell in cells:
        seed_sections: list[str] = []
        for seed_row in cell["seeds"]:
            cards = [
                _source_card(cell=cell, wrong=False),
                _source_card(cell=cell, wrong=True),
            ]
            cards.extend(
                _arm_card(arm=arm, item=seed_row["arms"][arm], cell=cell)
                for arm in ARM_ORDER
            )
            seed_sections.append(
                f"""
                <section class="seed-block">
                  <header><h3>Seed {seed_row['seed']}</h3><p>exact40 · 40 UniPC steps · 81 frames · common official Gaussian <code>{seed_row['official_gaussian_sha256']}</code></p></header>
                  <div class="five-grid">{''.join(cards)}</div>
                </section>"""
            )
        confound_note = (
            "PRESENT：wrong source 原生 geometry 与 target cell bucket 不一致；wrong-V 结果同时含 source-identity/content 与 geometry confound，不能解释为纯 identity 因果效应。"
            if cell["wrong_source_geometry_confound"]
            else "absent：未观测到 bucket geometry confound；但该分支仍明确不是 pure identity control。"
        )
        sections.append(
            f"""
            <section class="cell" id="cell-{html.escape(cell['cell_id'])}">
              <header class="cell-head"><div><p class="eyebrow">cell {html.escape(cell['cell_id'])}</p><h2>Correct source {html.escape(cell['source_iid'])} → native V-axis probe</h2></div><span class="badge {'danger' if cell['wrong_source_geometry_confound'] else 'caution'}">wrong-source geometry confound: {'PRESENT' if cell['wrong_source_geometry_confound'] else 'absent'}</span></header>
              <div class="instruction"><h3>完整 editing instruction / action caption</h3><p>{html.escape(cell['instruction'])}</p><code>{cell['instruction_sha256']}</code></div>
              <div class="identity-strip">
                <span><b>exact schedule</b> 40 UniPC steps</span>
                <span><b>base checkpoint tree SHA-256</b> <code>{cell['checkpoint_tree_sha256']}</code></span>
                <span><b>correct source SHA-256</b> <code>{cell['correct_source_sha256']}</code></span>
                <span><b>wrong-V source SHA-256</b> <code>{cell['wrong_source_sha256']}</code></span>
                <span><b>receipt</b> <a href="cells/{html.escape(cell['cell_id'], quote=True)}/receipt.json">exact JSON</a> · <code>{cell['receipt_file_sha256']}</code></span>
              </div>
              <p class="confound {'danger-box' if cell['wrong_source_geometry_confound'] else ''}"><b>Wrong-source geometry confound：</b>{html.escape(confound_note)}</p>
              {''.join(seed_sections)}
            </section>"""
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage-A0 · native full-video V-axis causal review</title>
<style>
:root{{--bg:#071018;--panel:#101d29;--soft:#172837;--line:#315067;--ink:#f1f6fb;--muted:#a9bac8;--cyan:#63d7dd;--amber:#ffc66d;--red:#ff7b78;--green:#72d6a0}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(150deg,#050b11,#0a1c2d);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{width:min(1900px,97vw);margin:auto;padding:42px 0 80px}}h1{{font-size:clamp(34px,5vw,68px);line-height:1.02;margin:.12em 0}}h2,h3,h4{{margin:.25em 0}}.eyebrow{{color:var(--cyan);text-transform:uppercase;letter-spacing:.13em;margin:0}}.lede{{max-width:1250px;color:#d3dfeb;font-size:18px}}.definitions{{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:12px;margin:24px 0}}.definitions article,.identity-strip,.instruction,.confound{{border:1px solid var(--line);background:var(--soft);border-radius:12px;padding:15px}}.definitions h2{{font-size:18px;color:var(--cyan)}}.definitions p{{margin:.4em 0}}.notice{{border-left:5px solid var(--amber);background:#20202a;padding:15px 18px;border-radius:10px}}.cell{{margin:30px 0;padding:22px;border:1px solid var(--line);border-radius:18px;background:rgba(16,29,41,.97)}}.cell-head{{display:flex;justify-content:space-between;gap:16px;align-items:start}}.badge{{display:inline-block;padding:5px 9px;border-radius:999px;font-size:12px;font-weight:800;white-space:nowrap}}.stable{{color:var(--green);background:rgba(114,214,160,.12)}}.caution{{color:var(--amber);background:rgba(255,198,109,.12)}}.danger{{color:var(--red);background:rgba(255,123,120,.14)}}.instruction{{margin:18px 0}}.instruction p{{font-size:18px;white-space:pre-wrap}}code{{color:var(--cyan);overflow-wrap:anywhere;font-size:11px}}.identity-strip{{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:9px;color:var(--muted)}}.identity-strip span{{min-width:0}}.confound{{margin:14px 0}}.danger-box{{border-color:var(--red);background:rgba(255,123,120,.08)}}.seed-block{{margin:24px 0 8px;padding-top:18px;border-top:1px solid var(--line)}}.seed-block header p{{margin:.3em 0 12px;color:var(--muted)}}.five-grid{{display:grid;grid-template-columns:repeat(5,minmax(250px,1fr));gap:12px;overflow-x:auto}}.video-card{{min-width:250px;padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--soft)}}video{{display:block;width:100%;aspect-ratio:1/1;max-height:540px;object-fit:contain;background:#020507;border-radius:8px;margin:9px 0}}.video-card p{{color:#d3dfeb}}.unique{{min-height:72px}}dl{{margin:0;display:grid;gap:7px}}dl div{{background:rgba(2,7,10,.35);padding:8px;border-radius:7px}}dt{{font-size:12px;color:var(--muted)}}dd{{margin:2px 0 0;font-weight:700}}a{{color:var(--cyan)}}@media(max-width:1100px){{.definitions{{grid-template-columns:1fr}}.identity-strip{{grid-template-columns:1fr}}.cell-head{{display:block}}}}
</style></head><body><main>
<p class="eyebrow">frozen-model causal visualization · Stage A0</p>
<h1>Native full-video V-axis review</h1>
<p class="lede">本页回答一个窄问题：在 frozen RV2V 的原生四分支 guidance 中，完整 source-video V 条件是否对生成结果有可见因果作用。它不是训练结果，也不是 preservation 新方法的成功展示。</p>
<section class="definitions">
  <article><h2>每行保持什么</h2><p>同一 cell / seed 内固定完整 editing instruction、4 个 correct-source image references、official Gaussian、exact40 scheduler、target geometry、base checkpoint。</p></article>
  <article><h2>三列各改什么</h2><p>V-on 是 native；V-off 只把 standalone (vV−v0) 系数 1.25→0；wrong-V 只替换 full-video V source，仍保留 correct image refs 与 instruction。</p></article>
  <article><h2>如何阅读</h2><p>直接人工观看完整 81 帧视频。每张卡只列有明确语义的 seed、公式、source / checkpoint / MP4 SHA；不存在通用分数字段。</p></article>
</section>
<p class="notice">本页不计算 feature scalar、reward、ranking、selection 或自动/人工成功 verdict。Wrong-V 的 geometry confound 会逐 cell 明确标红；有 confound 时不得把差异归因成纯 identity effect。</p>
{''.join(sections)}
</main></body></html>"""


def build(
    *,
    dog_output: str | Path,
    human_output: str | Path,
    output_dir: str | Path,
) -> Path:
    roots = {
        "dog": _plain_dir(dog_output, label="dog output"),
        "human": _plain_dir(human_output, label="human output"),
    }
    if roots["dog"] == roots["human"]:
        fail("dog and human inputs must be distinct output directories")
    cells = [
        _validate_cell(roots[cell_id], expected_cell_id=cell_id)
        for cell_id in CELL_ORDER
    ]
    shared_identity = {
        (
            cell["checkpoint_tree_sha256"],
            cell["runtime_source_revision"],
            cell["cell_spec_file_sha256"],
        )
        for cell in cells
    }
    if len(shared_identity) != 1:
        fail("dog/human receipts do not share one checkpoint/runtime/spec")

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
            local_root = staging / "cells" / cell["cell_id"]
            _copy_verified(
                cell["receipt_path"],
                local_root / "receipt.json",
                cell["receipt_file_sha256"],
            )
            _copy_verified(
                cell["correct_source_path"],
                local_root / "source-correct.mp4",
                cell["correct_source_sha256"],
            )
            _copy_verified(
                cell["wrong_source_path"],
                local_root / "source-wrong-V.mp4",
                cell["wrong_source_sha256"],
            )
            rendered = dict(cell)
            rendered["seeds"] = []
            manifest_seed_rows: list[dict[str, Any]] = []
            for seed_row in cell["seeds"]:
                rendered_seed = dict(seed_row)
                rendered_seed["arms"] = {}
                manifest_arms: dict[str, Any] = {}
                for arm in ARM_ORDER:
                    item = seed_row["arms"][arm]
                    basename = f"seed-{seed_row['seed']}__{arm}.mp4"
                    _copy_verified(
                        item["input_path"], local_root / basename, item["sha256"]
                    )
                    rendered_seed["arms"][arm] = {
                        **item,
                        "local_video": f"cells/{cell['cell_id']}/{basename}",
                    }
                    manifest_arms[arm] = {
                        "video": f"cells/{cell['cell_id']}/{basename}",
                        "mp4_sha256": item["sha256"],
                        "formula": item["formula"],
                        "unique_intervention": item["contract"]["intervention"],
                        "full_video_condition_role": item["contract"][
                            "full_video_condition_role"
                        ],
                        "trace_digest": item["trace_digest"],
                    }
                rendered["seeds"].append(rendered_seed)
                manifest_seed_rows.append(
                    {
                        "seed": seed_row["seed"],
                        "exact40": True,
                        "official_gaussian_sha256": seed_row[
                            "official_gaussian_sha256"
                        ],
                        "arms": manifest_arms,
                    }
                )
            rendered_cells.append(rendered)
            manifest_cells.append(
                {
                    "cell_id": cell["cell_id"],
                    "source_iid": cell["source_iid"],
                    "wrong_source_iid": cell["wrong_source_iid"],
                    "full_editing_instruction": cell["instruction"],
                    "instruction_sha256": cell["instruction_sha256"],
                    "checkpoint_tree_sha256": cell["checkpoint_tree_sha256"],
                    "correct_source": {
                        "video": f"cells/{cell['cell_id']}/source-correct.mp4",
                        "sha256": cell["correct_source_sha256"],
                    },
                    "wrong_V_source": {
                        "video": f"cells/{cell['cell_id']}/source-wrong-V.mp4",
                        "sha256": cell["wrong_source_sha256"],
                        "geometry_confound_present": cell[
                            "wrong_source_geometry_confound"
                        ],
                        "pure_identity_control": False,
                    },
                    "receipt": {
                        "path": f"cells/{cell['cell_id']}/receipt.json",
                        "file_sha256": cell["receipt_file_sha256"],
                        "embedded_digest": cell["receipt_digest"],
                    },
                    "seeds": manifest_seed_rows,
                }
            )
        manifest = {
            "schema_version": OUTPUT_MANIFEST_SCHEMA,
            "authority": dict(AUTHORITY),
            "experiment": {
                "method": METHOD,
                "stage": STAGE,
                "training": False,
                "exact40": True,
                "exact81": True,
                "frame_count": 81,
                "fps": 25,
                "native_formula": NATIVE_FORMULA,
                "v_off_formula": V_OFF_FORMULA,
                "arm_order": list(ARM_ORDER),
            },
            "cells": manifest_cells,
        }
        (staging / "manifest.json").write_text(
            json.dumps(
                manifest,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "index.html").write_text(
            render_html(rendered_cells), encoding="utf-8"
        )
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

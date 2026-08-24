#!/usr/bin/env python3
"""Build the v3 synchronized review page with source-residue evidence."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit import MANIFEST_SCHEMA, assert_not_protected_write, file_sha256
from .corrected_eval import load_json
from .source_residue_eval import SUMMARY_SCHEMA as QWEN_SUMMARY_SCHEMA


REPRESENTATION_SUMMARY_SCHEMA = "mev-action-representation-summary-v3"
ROLE_ORDER = ("source", "real_target", "anchor", "frozen_base")
ROLE_LABELS = {
    "source": "Source（只供人工上下文）",
    "real_target": "MEV real target",
    "anchor": "Self-generated anchor",
    "frozen_base": "Frozen-base edit",
}
METRIC_TO_VIDEO = {
    "source": "source_noop",
    "real_target": "target_forward",
    "anchor": "anchor",
    "frozen_base": "frozen_base",
}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _index(rows: Sequence[Mapping[str, Any]], key: str = "pair_prefix") -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or value in indexed:
            raise ValueError(f"invalid or duplicate {key}")
        indexed[value] = row
    return indexed


def build_cases(
    manifest: Mapping[str, Any],
    qwen: Mapping[str, Any],
    vjepa: Mapping[str, Any],
    videoprism: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema differs")
    if qwen.get("schema_version") != QWEN_SUMMARY_SCHEMA:
        raise ValueError("Qwen v3 summary schema differs")
    for name, payload in (("V-JEPA2", vjepa), ("VideoPrism", videoprism)):
        if payload.get("schema_version") != REPRESENTATION_SUMMARY_SCHEMA:
            raise ValueError(f"{name} summary schema differs")
    samples = _index(manifest.get("samples", []))
    qwen_rows = _index(qwen.get("pairs", []))
    vjepa_rows = _index(vjepa.get("pairs", []))
    videoprism_rows = _index(videoprism.get("pairs", []))
    if not samples or not (set(samples) == set(qwen_rows) == set(vjepa_rows) == set(videoprism_rows)):
        raise ValueError("v3 review pair sets differ")
    cases = []
    for sample in sorted(samples.values(), key=lambda row: row["ordinal"]):
        prefix = sample["pair_prefix"]
        qrow = qwen_rows[prefix]
        cases.append({
            "pair_prefix": prefix,
            "ordinal": sample["ordinal"],
            "instruction": sample["instruction"],
            "source_caption": sample["source_action_caption"],
            "target_caption": sample["target_action_caption"],
            "target_action": qrow["target_action"],
            "manual_winner": qrow["manual_winner"],
            "qwen_winner": qrow["qwen_winner"],
            "agrees": qrow["agrees_with_manual"],
            "human_note": qrow["human_note"],
            "pass_winners": qrow["pass_winners"],
            "source_aware_gates": qrow["source_aware_gate_scores"],
            "target_only_gates": qrow["target_only_gate_scores"],
            "coverage": qrow["coverage_scores"],
            "residue_results": qrow["source_residue_results"],
            "residue_contract": qrow["source_residue_contract"],
            "qwen_roles": qrow["roles"],
            "vjepa_scores": vjepa_rows[prefix]["scores"],
            "videoprism_scores": videoprism_rows[prefix]["scores"]["text_margin"],
        })
    return cases


def _badge(value: str) -> str:
    return f'<span class="badge {_e(value)}">{_e(value)}</span>'


def _qwen_evidence(case: Mapping[str, Any], role: str) -> str:
    records = case["qwen_roles"][role]
    trace = records[0].get("neutral_trace") if records else None
    blocks = []
    if trace:
        checkpoints = "".join(
            f'<tr><td>{item["index"]}</td><td>{_e(item["phase"])}</td>'
            f'<td>{_e(item["actor_orientation"])}</td><td>{_e(item["continuity_from_previous"])}</td>'
            f'<td>{_e(item["body_pose"])}</td><td>{_e(item["hands_and_objects"])}</td>'
            f'<td>{_e(item["observation"])}</td></tr>'
            for item in trace["dense_temporal_observations"]
        )
        blocks.append(
            f'<div class="trace"><b>冻结的 instruction-free 12-checkpoint trace</b>'
            f'<p>{_e(trace["neutral_summary"])}</p><div class="scroll"><table>'
            f'<tr><th>#</th><th>Phase</th><th>Orientation</th><th>Continuity</th>'
            f'<th>Body</th><th>Hands / objects</th><th>Observation</th></tr>{checkpoints}</table></div></div>'
        )
    for record in records:
        observation = record.get("observation")
        if not observation:
            blocks.append(f'<p class="bad">Pass {record["pass_index"] + 1} parse error: {_e(record.get("parse_error"))}</p>')
            continue
        required = "".join(
            f'<li><code>{_e(item["id"])}</code> = <b>{_e(item["result"])}</b> — '
            f'{_e("; ".join(ev["phase"] + ": " + ev["observation"] for ev in item["evidence"]))}</li>'
            for item in observation["required_predicates"]
        )
        forbidden = "".join(
            f'<li><code>{_e(item["id"])}</code> = <b>{_e(item["result"])}</b> — '
            f'{_e("; ".join(ev["phase"] + ": " + ev["observation"] for ev in item["evidence"]))}</li>'
            for item in observation["forbidden_behaviors"]
        )
        residue = observation["source_action_residue"]
        source_only = record.get("source_only_observation") or {}
        residue_evidence = "; ".join(
            ev["phase"] + ": " + ev["observation"] for ev in residue["evidence"]
        )
        blocks.append(
            f'<div class="pass"><b>Pass {record["pass_index"] + 1} · source-aware gate '
            f'{case["source_aware_gates"][role][record["pass_index"]]} · target-only gate '
            f'{case["target_only_gates"][role][record["pass_index"]]}</b><p>{_e(observation["summary"])}</p>'
            f'<div class="egrid"><div><span>Target required</span><ul>{required}</ul></div>'
            f'<div><span>Target forbidden</span><ul>{forbidden}</ul></div></div>'
            f'<div class="residue"><b>Source-action residue: {_e(residue["result"])}</b> · '
            f'<code>{_e(residue["id"])}</code><p>{_e(residue_evidence)}</p>'
            f'<p><b>独立 source-only 时序规则：</b>{_e(source_only.get("temporal_rule_application", "unavailable"))}</p>'
            f'</div></div>'
        )
    return "".join(blocks)


def _metric_rows(case: Mapping[str, Any]) -> str:
    return "".join(
        f'<tr><th>{_e(ROLE_LABELS[video_role])}</th>'
        f'<td>{case["vjepa_scores"]["ordered_residual"][METRIC_TO_VIDEO[video_role]]:+.4f}</td>'
        f'<td>{case["vjepa_scores"]["global_mean"][METRIC_TO_VIDEO[video_role]]:+.4f}</td>'
        f'<td>{case["videoprism_scores"][METRIC_TO_VIDEO[video_role]]:+.4f}</td></tr>'
        for video_role in ROLE_ORDER
    )


def _case_html(case: Mapping[str, Any], media_prefix: str) -> str:
    videos = "".join(
        f'<div><h4>{_e(ROLE_LABELS[role])}</h4><video playsinline muted preload="metadata" '
        f'src="{_e(media_prefix)}/{_e(case["pair_prefix"])}/{role}.mp4"></video></div>'
        for role in ROLE_ORDER
    )
    qrows = "".join(
        f'<tr><th>{_e(ROLE_LABELS[role])}</th>'
        f'<td>{" / ".join(map(str, case["source_aware_gates"][METRIC_TO_VIDEO[role]]))}</td>'
        f'<td>{" / ".join(map(str, case["target_only_gates"][METRIC_TO_VIDEO[role]]))}</td>'
        f'<td>{" / ".join(map(str, case["residue_results"][METRIC_TO_VIDEO[role]]))}</td></tr>'
        for role in ROLE_ORDER
    )
    agreement = "yes" if case["agrees"] else "no"
    return f'''<section class="case" data-manual="{_e(case['manual_winner'])}" data-agreement="{agreement}">
<div class="head"><div><span class="mono">{case['ordinal']:02d} · {_e(case['pair_prefix'])}</span><h2>{_e(case['target_action'])}</h2></div>
<div class="verdict"><span>人工 {_badge(case['manual_winner'])}</span><span>Qwen-v3 {_badge(case['qwen_winner'])}</span><span class="agree-{agreement}">一致 {agreement}</span></div></div>
<p class="note">{_e(case['human_note'])}</p>
<div class="toolbar"><button data-act="play">同步播放</button><button data-act="pause">暂停</button><button data-act="reset">归零</button>
<label>时间 <input data-act="seek" type="range" min="0" max="1000" value="0"></label><label>速度 <select data-act="rate"><option>.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label><span data-clock>0.00s</span></div>
<div class="videos">{videos}</div>
<div class="metrics"><div><h3>Qwen-v3 非补偿式门控</h3><table><tr><th>Role</th><th>含 source residue</th><th>只看 target</th><th>residue 结果</th></tr>{qrows}</table><p>pass winners: {_e(' / '.join(case['pass_winners']))}</p></div>
<div><h3>表征 action margin</h3><table><tr><th>Role</th><th>V-JEPA ordered ✓</th><th>V-JEPA global ✗</th><th>VideoPrism ✗</th></tr>{_metric_rows(case)}</table><p>分数是相对 margin，可为负；只有同一列内的相对顺序有意义。</p></div></div>
<details><summary>展开 source-action residue 合同与 Qwen 证据</summary><div class="contract"><b>{_e(case['residue_contract']['id'])}</b><p>{_e(case['residue_contract']['description'])}</p></div><h3>Anchor</h3>{_qwen_evidence(case, 'anchor')}<h3>Frozen-base</h3>{_qwen_evidence(case, 'frozen_base')}</details>
<details><summary>MEV 原始文本元数据</summary><dl><dt>Instruction</dt><dd>{_e(case['instruction'])}</dd><dt>Source event</dt><dd>{_e(case['source_caption'])}</dd><dt>Target event</dt><dd>{_e(case['target_caption'])}</dd></dl></details>
</section>'''


def render_html(
    cases: Sequence[Mapping[str, Any]],
    qwen: Mapping[str, Any],
    vjepa: Mapping[str, Any],
    videoprism: Mapping[str, Any],
    media_prefix: str,
) -> str:
    qc = qwen["control_calibration"]
    vo = vjepa["metrics"]["ordered_residual"]["admission"]
    vg = vjepa["metrics"]["global_mean"]["admission"]
    vp = videoprism["metrics"]["text_margin"]["admission"]
    cards = "".join(_case_html(case, media_prefix) for case in cases)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MEV action anchor gap · source-residue v3</title><style>
:root{{--bg:#0b111b;--panel:#151e2b;--line:#2b3b51;--text:#edf3ff;--muted:#a6b5ca;--blue:#74b6ff;--green:#58d69c;--red:#ff818b;--amber:#f2c96f}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}}main{{max-width:1720px;margin:auto;padding:24px}}a{{color:var(--blue)}}h1{{margin:.2em 0}}h2{{font-size:18px}}h3{{font-size:16px}}h4{{color:var(--muted);margin:0 0 8px}}p,li,dd{{line-height:1.5}}.lede,.summary,.case{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}}.lede{{border-left:5px solid var(--amber)}}.summary-grid,.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.head{{display:flex;justify-content:space-between;gap:16px;align-items:start}}.verdict,.toolbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}.badge{{border:1px solid var(--line);padding:4px 8px;border-radius:999px}}.badge.anchor,.agree-yes{{color:var(--green)}}.badge.frozen_base,.agree-no,.bad{{color:var(--red)}}.badge.tie{{color:var(--amber)}}.badge.abstain{{color:var(--muted)}}.mono,code,td{{font-family:ui-monospace,SFMono-Regular,monospace}}.note{{color:var(--amber)}}button,select{{background:#223147;color:var(--text);border:1px solid #3a506d;border-radius:8px;padding:7px 10px}}input{{accent-color:var(--blue)}}input[type=range]{{width:260px}}.videos{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0}}video{{display:block;width:100%;aspect-ratio:16/9;background:#000;border-radius:10px;object-fit:contain}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}details{{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}}summary{{cursor:pointer;font-weight:700}}.trace,.pass,.contract{{background:#0d1521;padding:12px;border-radius:10px;margin:10px 0}}.residue{{border-left:4px solid var(--amber);padding-left:12px}}.scroll{{overflow:auto}}.trace table{{min-width:1200px;font-size:12px}}.egrid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;color:var(--muted)}}dl{{display:grid;grid-template-columns:150px 1fr;gap:8px}}dt{{color:var(--muted)}}dd{{margin:0}}@media(max-width:980px){{.videos,.summary-grid,.metrics,.egrid{{grid-template-columns:1fr 1fr}}}}@media(max-width:650px){{.videos,.summary-grid,.metrics,.egrid{{grid-template-columns:1fr}}.head{{display:block}}}}
</style></head><body><main><h1>MEV action anchor vs frozen-base · source-residue v3</h1>
<div class="lede"><b>本页修正“把 source 不变性误当编辑成功”的判别错误。</b>Qwen 看视频的阶段完全不知道 source/target；target-only 与 source-only 是两个互不披露、互不补偿的文本审计，source-only 只读冻结轨迹和 MEV source-event 行为负例。“任一时刻发生”会永久触发 violation，后续完成 target 不能洗掉；人物、场景、layout、服装、相机与初始 pose 均禁止当作 residue。原页面完整保留：<a href="../20260819_anchor_gap16_review/index.html">v1 failure evidence</a> · <a href="../20260819_anchor_gap16_review_v2/index.html">v2 corrected audit</a>。</div>
<div class="summary"><div class="summary-grid"><div><h3>Qwen-v3 · diagnostic only</h3><p class="bad">未通过 reverse/shuffle controls，不参与最终 winner 投票。</p><p>人工一致 {qwen['manual_agreement_count']}/{qwen['pair_count']} ({qwen['manual_agreement_rate']:.1%})<br>winners: {_e(qwen['winner_counts'])}</p></div>
<div><h3>Qwen controls</h3><p>forward strict {qc['target_forward_strict_pass_count']}/16 · source strict {qc['source_noop_strict_pass_count']}/16<br>forward&gt;reverse {qc['reverse_below_forward_count']}/16 · forward&gt;shuffle {qc['shuffle_below_forward_count']}/16<br>source residue detected {qc['source_noop_residue_yes_count']}/16 · target residue absent {qc['target_forward_residue_no_count']}/16</p></div>
<div><h3>V-JEPA2 ordered residual · admitted {str(vo['admitted_for_candidate_voting']).lower()}</h3><p>forward&gt;source {vo['counts']['forward_over_source_noop']}/16 · forward&gt;reverse {vo['counts']['forward_over_target_reverse']}/16 · forward&gt;shuffle {vo['counts']['forward_over_target_shuffle']}/16</p></div>
<div><h3>被拒绝的全局表征</h3><p>V-JEPA global admitted {str(vg['admitted_for_candidate_voting']).lower()} · shuffle {vg['counts']['forward_over_target_shuffle']}/16<br>VideoPrism admitted {str(vp['admitted_for_candidate_voting']).lower()} · reverse {vp['counts']['forward_over_target_reverse']}/16 · shuffle {vp['counts']['forward_over_target_shuffle']}/16</p></div></div></div>
<div class="toolbar"><button data-filter="all">全部</button><button data-filter="anchor">人工 anchor</button><button data-filter="tie">人工 tie</button><button data-filter="mismatch">Qwen 不一致</button></div>{cards}
<script>const groups=[...document.querySelectorAll('.case')];function vids(g){{return [...g.querySelectorAll('video')]}}function seek(g,t){{vids(g).forEach(v=>{{if(Number.isFinite(v.duration))v.currentTime=Math.min(t,v.duration)}})}}groups.forEach(g=>{{const vs=vids(g),range=g.querySelector('[data-act=seek]'),clock=g.querySelector('[data-clock]');g.querySelector('[data-act=play]').onclick=()=>{{seek(g,vs[0].currentTime);vs.forEach(v=>v.play())}};g.querySelector('[data-act=pause]').onclick=()=>vs.forEach(v=>v.pause());g.querySelector('[data-act=reset]').onclick=()=>{{vs.forEach(v=>v.pause());seek(g,0);range.value=0}};g.querySelector('[data-act=rate]').onchange=e=>vs.forEach(v=>v.playbackRate=+e.target.value);range.oninput=e=>{{const d=vs[0].duration||3.2;seek(g,d*(+e.target.value/1000))}};vs[0].ontimeupdate=()=>{{const d=vs[0].duration||3.2;range.value=1000*vs[0].currentTime/d;clock.textContent=vs[0].currentTime.toFixed(2)+'s'}};vs.forEach(v=>v.onclick=()=>vs.some(x=>!x.paused)?vs.forEach(x=>x.pause()):g.querySelector('[data-act=play]').click())}});document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>groups.forEach(g=>g.hidden=!(b.dataset.filter==='all'||g.dataset.manual===b.dataset.filter||(b.dataset.filter==='mismatch'&&g.dataset.agreement==='no'))));</script></main></body></html>'''


def run(args: argparse.Namespace) -> int:
    paths = {
        "manifest": Path(args.manifest).resolve(strict=True),
        "qwen": Path(args.qwen_summary).resolve(strict=True),
        "vjepa": Path(args.vjepa_summary).resolve(strict=True),
        "videoprism": Path(args.videoprism_summary).resolve(strict=True),
    }
    output = Path(args.output).resolve(strict=False)
    assert_not_protected_write(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    payloads = {name: load_json(path) for name, path in paths.items()}
    cases = build_cases(payloads["manifest"], payloads["qwen"], payloads["vjepa"], payloads["videoprism"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(
        cases, payloads["qwen"], payloads["vjepa"], payloads["videoprism"],
        args.media_prefix.rstrip("/"),
    ), encoding="utf-8")
    receipt = {
        "schema_version": "mev-action-anchor-source-residue-review-v3",
        "pair_count": len(cases),
        "html_sha256": file_sha256(output),
        "media_copied": False,
        "v1_and_v2_html_preserved": True,
        "inputs": {name: file_sha256(path) for name, path in paths.items()},
    }
    output.with_name("bundle_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qwen-summary", required=True)
    parser.add_argument("--vjepa-summary", required=True)
    parser.add_argument("--videoprism-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--media-prefix", default="../20260819_anchor_gap16_review/media")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

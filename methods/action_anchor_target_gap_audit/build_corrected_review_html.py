#!/usr/bin/env python3
"""Build the corrected v2 synchronized MEV action-gap review HTML."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit import MANIFEST_SCHEMA, assert_not_protected_write, file_sha256
from .corrected_eval import INTERNVIDEO_SCHEMA, SUMMARY_SCHEMA, load_json


ROLE_ORDER = ("source", "real_target", "anchor", "frozen_base")
ROLE_LABELS = {
    "source": "Source (context only)",
    "real_target": "MEV real target",
    "anchor": "Self-generated anchor",
    "frozen_base": "Frozen-base edit",
}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _index(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or value in result:
            raise ValueError(f"invalid or duplicate {key}")
        result[value] = row
    return result


def build_cases(manifest: Mapping[str, Any], qwen: Mapping[str, Any], intern: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema differs")
    if qwen.get("schema_version") != SUMMARY_SCHEMA:
        raise ValueError("Qwen v2 summary schema differs")
    if intern.get("schema_version") != INTERNVIDEO_SCHEMA:
        raise ValueError("InternVideo2 summary schema differs")
    samples = _index(manifest.get("samples", []), "pair_prefix")
    qwen_rows = _index(qwen.get("pairs", []), "pair_prefix")
    intern_rows = _index(intern.get("pairs", []), "pair_prefix")
    if not samples or set(samples) != set(qwen_rows) or set(samples) != set(intern_rows):
        raise ValueError("manifest and corrected evaluator pair sets differ")
    cases = []
    for prefix in sorted(samples):
        sample, qrow, irow = samples[prefix], qwen_rows[prefix], intern_rows[prefix]
        cases.append({
            "pair_prefix": prefix, "pair_id": sample["pair_id"],
            "instruction": sample["instruction"],
            "source_caption": sample["source_action_caption"],
            "target_caption": sample["target_action_caption"],
            "manual_winner": qrow["manual_winner"], "qwen_winner": qrow["qwen_winner"],
            "agrees": qrow["agrees_with_manual"], "human_note": qrow["human_note"],
            "target_action": qrow["target_action"], "pass_winners": qrow["pass_winners"],
            "gate_scores": qrow["gate_scores"], "coverage_scores": qrow["coverage_scores"],
            "qwen_roles": qrow["roles"],
            "intern_scores": irow["role_scores"],
        })
    return cases


def _badge(value: str) -> str:
    return f'<span class="badge {_e(value)}">{_e(value)}</span>'


def _evidence(case: Mapping[str, Any], role: str) -> str:
    blocks = []
    records = case["qwen_roles"][role]
    trace = records[0].get("neutral_trace") if records else None
    if trace:
        checkpoints = "".join(
            f'<tr><td>{item["index"]}</td><td>{_e(item["phase"])}</td><td>{_e(item["actor_orientation"])}</td>'
            f'<td>{_e(item["continuity_from_previous"])}</td><td>{_e(item["body_pose"])}</td>'
            f'<td>{_e(item["hands_and_objects"])}</td></tr>'
            for item in trace["dense_temporal_observations"]
        )
        blocks.append(
            f'<div class="trace"><b>Instruction-free 12-checkpoint trace</b>'
            f'<p>{_e(trace["neutral_summary"])}</p><div class="table-scroll"><table><tr><th>#</th><th>Phase</th><th>Orientation</th><th>Continuity</th><th>Body</th><th>Hands / objects</th></tr>{checkpoints}</table></div></div>'
        )
    elif records:
        blocks.append(f'<p class="bad">Neutral trace parse error: {_e(records[0].get("neutral_trace_parse_error"))}</p>')
    for record in records:
        observation = record.get("observation")
        if not observation:
            blocks.append(f'<div class="pass"><b>Pass {record["pass_index"] + 1}</b><p class="bad">Parse error: {_e(record.get("parse_error"))}</p></div>')
            continue
        required = []
        for item in observation["required_predicates"]:
            evidence = "; ".join(f'{ev["phase"]}: {ev["observation"]}' for ev in item["evidence"])
            required.append(f'<li><code>{_e(item["id"])}</code> = <b>{_e(item["result"])}</b> — {_e(evidence)}</li>')
        forbidden = []
        for item in observation["forbidden_behaviors"]:
            evidence = "; ".join(f'{ev["phase"]}: {ev["observation"]}' for ev in item["evidence"])
            forbidden.append(f'<li><code>{_e(item["id"])}</code> = <b>{_e(item["result"])}</b> — {_e(evidence)}</li>')
        blocks.append(
            f'<div class="pass"><b>Pass {record["pass_index"] + 1} · gate {case["gate_scores"][role][record["pass_index"]]}</b>'
            f'<p>{_e(observation["summary"])}</p><div class="egrid"><div><span>Required</span><ul>{"".join(required)}</ul></div>'
            f'<div><span>Forbidden (presence fails)</span><ul>{"".join(forbidden)}</ul></div></div></div>'
        )
    return "".join(blocks)


def _case_html(case: Mapping[str, Any], media_prefix: str, intern_admitted: bool) -> str:
    videos = "".join(
        f'<div class="video-cell"><h4>{_e(ROLE_LABELS[role])}</h4>'
        f'<video playsinline muted preload="metadata" src="{_e(media_prefix)}/{_e(case["pair_prefix"])}/{role}.mp4"></video></div>'
        for role in ROLE_ORDER
    )
    margins = case["intern_scores"]
    intern_label = "admitted auxiliary" if intern_admitted else "rejected diagnostic"
    metric_roles = (("target_forward", "real_target"), ("source_noop", "source"), ("anchor", "anchor"), ("frozen_base", "frozen_base"))
    intern_rows = "".join(
        f'<tr><th>{_e(ROLE_LABELS[label_role])}</th><td>{margins[metric_role]["target_action_text_cosine"]:+.4f}</td>'
        f'<td>{margins[metric_role]["source_action_text_cosine"]:+.4f}</td><td>{margins[metric_role]["target_minus_source_action_margin"]:+.4f}</td></tr>'
        for metric_role, label_role in metric_roles
    )
    agreement = "yes" if case["agrees"] else "no"
    return f'''<section class="case" data-manual="{_e(case['manual_winner'])}" data-agreement="{agreement}">
      <div class="case-head"><div><span class="mono">{_e(case['pair_prefix'])}</span><h2>{_e(case['target_action'])}</h2></div>
      <div class="verdict"><span>人工 {_badge(case['manual_winner'])}</span><span>Qwen v2 {_badge(case['qwen_winner'])}</span><span class="agree-{agreement}">一致: {agreement}</span></div></div>
      <p class="note">{_e(case['human_note'])}</p>
      <div class="toolbar"><button data-act="play">同步播放</button><button data-act="pause">暂停</button><button data-act="reset">归零</button>
      <label>时间 <input data-act="seek" type="range" min="0" max="1000" value="0"></label>
      <label>速度 <select data-act="rate"><option>.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label><span data-clock>0.00s</span></div>
      <div class="videos">{videos}</div>
      <div class="metrics"><div><h3>Qwen v2 source-hidden native-video</h3>
      <table><tr><th></th><th>Pass 1</th><th>Pass 2</th></tr>
      <tr><th>Anchor gate</th><td>{case['gate_scores']['anchor'][0]}</td><td>{case['gate_scores']['anchor'][1]}</td></tr>
      <tr><th>Anchor coverage (tie-break only)</th><td>{case['coverage_scores']['anchor'][0]:.2f}</td><td>{case['coverage_scores']['anchor'][1]:.2f}</td></tr>
      <tr><th>Frozen-base gate</th><td>{case['gate_scores']['frozen_base'][0]}</td><td>{case['gate_scores']['frozen_base'][1]}</td></tr>
      <tr><th>Frozen coverage (tie-break only)</th><td>{case['coverage_scores']['frozen_base'][0]:.2f}</td><td>{case['coverage_scores']['frozen_base'][1]:.2f}</td></tr></table>
      <p>pass winners: {_e(' / '.join(case['pass_winners']))}</p></div>
      <div><h3>InternVideo2 · {_e(intern_label)}</h3><table><tr><th>Role</th><th>target text</th><th>source text</th><th>margin</th></tr>{intern_rows}</table></div></div>
      <details><summary>展开 Qwen 原子判据证据</summary><h3>Anchor</h3>{_evidence(case, 'anchor')}<h3>Frozen-base</h3>{_evidence(case, 'frozen_base')}</details>
      <details><summary>MEV 原始文本元数据</summary><dl><dt>Instruction</dt><dd>{_e(case['instruction'])}</dd><dt>Source event</dt><dd>{_e(case['source_caption'])}</dd><dt>Target event</dt><dd>{_e(case['target_caption'])}</dd></dl></details>
    </section>'''


def render_html(cases: Sequence[Mapping[str, Any]], qwen: Mapping[str, Any], intern: Mapping[str, Any], media_prefix: str) -> str:
    calibration = qwen["control_calibration"]
    ic = intern["calibration"]
    admitted = bool(ic["admitted_for_candidate_ranking"])
    cards = "".join(_case_html(case, media_prefix, admitted) for case in cases)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MEV action anchor gap · corrected v2</title><style>
:root{{--bg:#0b111b;--panel:#151e2b;--line:#2a3a50;--text:#ecf2ff;--muted:#9dadc4;--blue:#6fb2ff;--green:#57d69b;--red:#ff7d87;--amber:#f3c86b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}}main{{max-width:1680px;margin:auto;padding:24px}}h1{{margin:.2em 0}}h2{{font-size:18px;margin:8px 0}}h3{{font-size:16px}}h4{{margin:0 0 8px;color:var(--muted)}}p,li,dd{{line-height:1.55}}a{{color:var(--blue)}}.lede,.case,.summary{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}}.warning{{border-left:5px solid var(--red)}}.summary-grid,.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.case-head{{display:flex;justify-content:space-between;gap:18px;align-items:start}}.verdict{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}.badge{{border:1px solid var(--line);padding:4px 8px;border-radius:999px}}.badge.anchor{{color:var(--green)}}.badge.frozen_base{{color:var(--red)}}.badge.tie{{color:var(--amber)}}.badge.abstain{{color:var(--muted)}}.agree-yes{{color:var(--green)}}.agree-no,.bad{{color:var(--red)}}.mono,code,td{{font-family:ui-monospace,SFMono-Regular,monospace}}.note{{color:var(--amber)}}.toolbar{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:14px 0}}button,select,input{{accent-color:var(--blue)}}button,select{{background:#223147;color:var(--text);border:1px solid #3a506d;border-radius:8px;padding:7px 10px}}input[type=range]{{width:260px}}.videos{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}video{{display:block;width:100%;aspect-ratio:16/9;background:#000;border-radius:10px;object-fit:contain}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}details{{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}}summary{{cursor:pointer;font-weight:700}}.pass,.trace{{background:#0d1521;padding:12px;border-radius:10px;margin:10px 0}}.table-scroll{{overflow:auto}}.trace table{{min-width:980px;font-size:12px}}.egrid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;color:var(--muted)}}dl{{display:grid;grid-template-columns:150px 1fr;gap:8px}}dt{{color:var(--muted)}}dd{{margin:0}}@media(max-width:980px){{.videos,.summary-grid,.metrics,.egrid{{grid-template-columns:1fr 1fr}}}}@media(max-width:640px){{.videos,.summary-grid,.metrics,.egrid{{grid-template-columns:1fr}}.case-head{{display:block}}}}
</style></head><body><main><h1>MEV action anchor vs frozen-base · corrected v2</h1>
<div class="lede warning"><b>v1 判别已撤回。</b>旧页面原样保留为失败分析证据：<a href="../20260819_anchor_gap16_review/index.html">打开旧 HTML</a>。v2 不把 source、real target 与候选放进同一个 Qwen 上下文；下面四路视频只供人工同步复核。<b>人工 14 anchor / 2 tie 是本页主结论；Qwen v2 是校准诊断，不是替代人工真值的投票器。</b></div>
<div class="summary"><div class="summary-grid"><div><h3>Qwen v2</h3><p>人工一致率 <b>{qwen['manual_agreement_count']}/{qwen['pair_count']} ({qwen['manual_agreement_rate']:.1%})</b></p><p>winners: {_e(qwen['winner_counts'])}</p></div>
<div><h3>时序控制</h3><p>forward strict {calibration['target_forward_strict_pass_count']}/16 · source no-op strict {calibration['source_noop_strict_pass_count']}/16<br>reverse below forward {calibration['reverse_below_forward_count']}/16 · shuffle below forward {calibration['shuffle_below_forward_count']}/16</p></div>
<div><h3>InternVideo2-CLIP-S</h3><p>admitted: <b>{str(admitted).lower()}</b><br>forward&gt;reverse {ic['forward_over_reverse_count']}/16 · forward&gt;shuffle {ic['forward_over_shuffle_count']}/16 · forward&gt;source {ic['forward_over_source_count']}/16</p></div>
<div><h3>解释边界</h3><p>这 16 个样本用于按人工标签校准，不能作为独立测试准确率；模型泛化必须另取未参与合同设计的 MEV held-out 样本。</p></div></div></div>
<div class="toolbar"><button data-filter="all">全部</button><button data-filter="anchor">人工 anchor</button><button data-filter="tie">人工 tie</button><button data-filter="mismatch">Qwen 不一致</button></div>{cards}
<script>
const groups=[...document.querySelectorAll('.case')];
function vids(g){{return [...g.querySelectorAll('video')]}}function seek(g,t){{vids(g).forEach(v=>{{if(Number.isFinite(v.duration))v.currentTime=Math.min(t,v.duration)}})}}
groups.forEach(g=>{{const vs=vids(g),range=g.querySelector('[data-act=seek]'),clock=g.querySelector('[data-clock]');
g.querySelector('[data-act=play]').onclick=()=>{{const t=vs[0].currentTime;seek(g,t);vs.forEach(v=>v.play())}};g.querySelector('[data-act=pause]').onclick=()=>vs.forEach(v=>v.pause());
g.querySelector('[data-act=reset]').onclick=()=>{{vs.forEach(v=>v.pause());seek(g,0);range.value=0}};g.querySelector('[data-act=rate]').onchange=e=>vs.forEach(v=>v.playbackRate=+e.target.value);
range.oninput=e=>{{const d=vs[0].duration||3.2;seek(g,d*(+e.target.value/1000))}};vs[0].ontimeupdate=()=>{{const d=vs[0].duration||3.2;range.value=1000*vs[0].currentTime/d;clock.textContent=vs[0].currentTime.toFixed(2)+'s'}};
vs.forEach(v=>v.onclick=()=>vs.some(x=>!x.paused)?vs.forEach(x=>x.pause()):g.querySelector('[data-act=play]').click())}});
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>groups.forEach(g=>g.hidden=!(b.dataset.filter==='all'||g.dataset.manual===b.dataset.filter||(b.dataset.filter==='mismatch'&&g.dataset.agreement==='no'))));
</script></main></body></html>'''


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve(strict=True)
    qwen_path = Path(args.qwen_summary).resolve(strict=True)
    intern_path = Path(args.internvideo_summary).resolve(strict=True)
    output = Path(args.output).resolve(strict=False)
    assert_not_protected_write(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    manifest, qwen, intern = load_json(manifest_path), load_json(qwen_path), load_json(intern_path)
    cases = build_cases(manifest, qwen, intern)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(cases, qwen, intern, args.media_prefix.rstrip("/")), encoding="utf-8")
    receipt = {
        "schema_version": "mev-action-anchor-corrected-review-v2", "pair_count": len(cases),
        "html_sha256": file_sha256(output), "media_copied": False,
        "old_html_preserved": True,
        "inputs": {"manifest": file_sha256(manifest_path), "qwen": file_sha256(qwen_path), "internvideo2": file_sha256(intern_path)},
    }
    receipt_path = output.with_name("bundle_manifest.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qwen-summary", required=True)
    parser.add_argument("--internvideo-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--media-prefix", default="../20260819_anchor_gap16_review/media")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a synchronized human-review page for frozen and trained reward arms."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = (
    ("frozen_base", "Frozen base（0 update）"),
    ("baseline", "No-reward SFT（40 updates）"),
    ("action_only", "Action-selected SFT（40 updates）"),
    ("preservation_only", "Preservation-selected SFT（40 updates）"),
    ("composite", "Composite-selected SFT（40 updates）"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value):.4f}"


def build(score_result: Path, media_root: Path, output: Path) -> dict[str, Any]:
    result = json.loads(score_result.read_text(encoding="utf-8"))
    if result.get("schema_version") != "action-editing-reward-ablation-result-v1":
        raise ValueError("score result schema differs")
    groups = result.get("groups")
    if not isinstance(groups, list) or len(groups) != 4:
        raise ValueError("review requires exactly four groups")
    cards = []
    receipt_groups = []
    for group in groups:
        iid = group["iid"]
        group_root = media_root / iid
        expected = ["source", "anchor", *(arm for arm, _ in ARMS)]
        media = {}
        for role in expected:
            path = group_root / f"{role}.mp4"
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"missing plain review media: {path}")
            media[role] = {
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(output.parent)),
                "sha256": file_sha256(path),
            }
        action_rows = {row["candidate_id"]: row for row in group["action_reward"]["candidates"]}
        preservation_rows = {
            row["candidate_id"]: row for row in group["preservation_reward"]["rows"]
        }
        video_cards = [
            f'<article class="video-card context"><h3>Source</h3><video controls muted loop playsinline preload="metadata" src="{html.escape(media["source"]["relative_path"])}"></video><p>输入视频，不参与投票</p></article>',
            f'<article class="video-card context"><h3>Self-generated action anchor</h3><video controls muted loop playsinline preload="metadata" src="{html.escape(media["anchor"]["relative_path"])}"></video><p>只提供动作语义/时序参照</p></article>',
        ]
        score_rows = []
        for arm, label in ARMS:
            candidate_id = f"reward-training-eval-{iid}-{arm}"
            action = action_rows[candidate_id]
            preservation = preservation_rows[candidate_id]
            raw_action = action["raw_scores"]
            raw_pres = preservation["raw_scores"]
            video_cards.append(
                f'<article class="video-card arm" data-arm="{arm}"><h3>{html.escape(label)}</h3>'
                f'<video controls muted loop playsinline preload="metadata" src="{html.escape(media[arm]["relative_path"])}"></video>'
                f'<div class="audit"><button data-v="action_ok">动作成功</button><button data-v="action_fail">动作失败</button>'
                f'<button data-v="pres_fail">主体/背景失败</button><button data-v="unclear">无法判断</button></div></article>'
            )
            score_rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td><td>{_fmt(action.get('event_score'))}</td>"
                f"<td>{_fmt(raw_action.get('motion_set'))}</td><td>{_fmt(raw_action.get('ordered_alignment'))}</td>"
                f"<td>{_fmt(raw_action.get('reverse_contrast'))}</td><td>{_fmt(raw_pres.get('source_appearance_set_proxy'))}</td>"
                f"<td>{_fmt(raw_pres.get('fixed_grid_background_dominant_proxy'))}</td>"
                f"<td>{_fmt(raw_pres.get('camera_translation_agreement_proxy'))}</td><td>{_fmt(raw_pres.get('decode_quality_proxy'))}</td>"
                "</tr>"
            )
        options = "".join(
            f'<option value="{arm}">{html.escape(label)}</option>' for arm, label in ARMS
        )
        cards.append(
            f'<section class="group" data-iid="{iid}"><header><h2>{iid}</h2>'
            f'<p class="instruction">{html.escape(group["instruction"])}</p>'
            '<div class="controls"><button class="sync-play">同步播放/暂停</button><button class="sync-zero">全部归零</button>'
            '<button class="sync-phase">按归一化进度对齐</button><label>速度 <select class="rate"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label></div></header>'
            f'<div class="video-grid">{"".join(video_cards)}</div>'
            f'<div class="verdict"><label>整体最好：<select class="winner"><option value="">未判断</option>{options}<option value="tie">近似相同</option><option value="none">都不可用</option></select></label>'
            '<textarea placeholder="备注：动作完成度/方向、主体、背景、camera、artifact…"></textarea></div>'
            '<details><summary>机器诊断分数（先看视频；不是正确答案，且 action/preservation 均未做人类绝对校准）</summary>'
            '<table><thead><tr><th>arm</th><th>组内 action event</th><th>motion set</th><th>ordered</th><th>reverse contrast</th><th>appearance*</th><th>fixed-grid*</th><th>camera*</th><th>quality*</th></tr></thead>'
            f'<tbody>{"".join(score_rows)}</tbody></table><p class="foot">* preservation 为弱 proxy，不等于实例身份或分割后的 safe background。</p></details></section>'
        )
        receipt_groups.append({"iid": iid, "media": media})
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reward training · Job 135096</title>
<style>:root{{--bg:#f6f4ee;--ink:#17201d;--line:#d6cdbd;--accent:#176b58}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,sans-serif}}main{{max-width:2200px;margin:auto;padding:24px}}.intro,.group{{background:#fff;border:1px solid var(--line);border-radius:18px;padding:20px;margin:0 0 24px}}h1,h2,h3{{margin:.2em 0}}.instruction{{max-width:1300px;color:#4b5752}}button,select,textarea{{font:inherit}}button,select{{border:1px solid #aa9f8c;background:#fbfaf6;border-radius:10px;padding:8px 12px}}button.active{{background:var(--accent);color:white;border-color:var(--accent)}}.controls{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:14px 0}}.sync-status{{color:#63706b;font-size:14px}}.sync-status.error{{color:#a52a2a}}.video-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}.video-card{{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#faf9f5}}.video-card h3,.video-card p,.audit{{padding:8px 12px;margin:0}}video{{display:block;width:100%;aspect-ratio:16/9;background:#050706;object-fit:contain}}.audit{{display:flex;gap:6px;flex-wrap:wrap}}.audit button{{font-size:13px}}.verdict{{display:grid;grid-template-columns:minmax(280px,420px) 1fr;gap:12px;margin-top:15px}}textarea{{min-height:74px;border:1px solid #aa9f8c;border-radius:10px;padding:10px}}details{{margin-top:14px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid var(--line);text-align:left}}.foot{{color:#63706b}}@media(max-width:700px){{main{{padding:10px}}.verdict{{grid-template-columns:1fr}}}}</style></head><body><main>
<section class="intro"><h1>Reward-guided training · 同 seed 人工审计</h1><p>每组展示 source、self-generated action anchor、frozen base、无 reward synthetic-SFT 与三个 reward-selected synthetic-SFT。所有输出使用相同 seed <code>2026081601</code>；不存在机器“正确答案”。先同步看完整视频，再展开诊断分数。</p><button class="export">导出人工审计 JSON</button></section>{''.join(cards)}</main>
<script>const state={{schema_version:'reward-training-human-review-v1',created_at:new Date().toISOString(),groups:{{}}}};
const waitForMetadata=v=>v.readyState>=1?Promise.resolve():new Promise(resolve=>{{const done=()=>{{v.removeEventListener('loadedmetadata',done);v.removeEventListener('error',done);resolve()}};v.addEventListener('loadedmetadata',done,{{once:true}});v.addEventListener('error',done,{{once:true}});v.load()}});
document.querySelectorAll('.group').forEach(g=>{{
  const iid=g.dataset.iid,vs=[...g.querySelectorAll('video')],playButton=g.querySelector('.sync-play'),rate=g.querySelector('.rate');
  const status=document.createElement('span');status.className='sync-status';status.textContent='等待播放';playButton.parentElement.appendChild(status);
  const align=()=>{{const ref=vs.find(v=>Number.isFinite(v.duration)&&v.duration>0);if(!ref)return;const q=ref.currentTime/ref.duration;vs.forEach(v=>{{if(Number.isFinite(v.duration)&&v.duration>0)v.currentTime=q*v.duration}})}};
  state.groups[iid]={{arms:{{}},winner:'',note:''}};
  playButton.onclick=async()=>{{
    const shouldPlay=vs.some(v=>v.paused||v.ended);
    if(!shouldPlay){{vs.forEach(v=>v.pause());playButton.textContent='同步播放';playButton.classList.remove('active');status.textContent='已暂停';return}}
    status.className='sync-status';status.textContent='正在载入…';
    await Promise.all(vs.map(waitForMetadata));align();
    vs.forEach(v=>{{v.muted=true;v.playbackRate=+rate.value}});
    const outcomes=await Promise.allSettled(vs.map(v=>v.play()));
    const failed=outcomes.filter(x=>x.status==='rejected').length;
    if(failed){{status.className='sync-status error';status.textContent=failed+'/'+vs.length+' 个视频播放失败';playButton.textContent='重试同步播放'}}else{{status.textContent='同步播放中';playButton.textContent='同步暂停';playButton.classList.add('active')}}
  }};
  g.querySelector('.sync-zero').onclick=()=>{{vs.forEach(v=>{{v.pause();v.currentTime=0}});playButton.textContent='同步播放';playButton.classList.remove('active');status.textContent='已归零'}};
  g.querySelector('.sync-phase').onclick=()=>{{align();status.textContent='已按进度对齐'}};
  rate.onchange=e=>vs.forEach(v=>v.playbackRate=+e.target.value);
  g.querySelectorAll('.arm').forEach(c=>{{const arm=c.dataset.arm;state.groups[iid].arms[arm]=[];c.querySelectorAll('button').forEach(b=>b.onclick=()=>{{b.classList.toggle('active');const a=state.groups[iid].arms[arm],v=b.dataset.v,i=a.indexOf(v);i<0?a.push(v):a.splice(i,1)}})}});
  g.querySelector('.winner').onchange=e=>state.groups[iid].winner=e.target.value;g.querySelector('textarea').oninput=e=>state.groups[iid].note=e.target.value;
}});
document.querySelector('.export').onclick=()=>{{state.exported_at=new Date().toISOString();const b=new Blob([JSON.stringify(state,null,2)+'\\n'],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='reward-training-human-review.json';a.click();URL.revokeObjectURL(a.href)}};</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    receipt = {
        "schema_version": "reward-training-review-html-receipt-v1",
        "groups": receipt_groups,
        "html_path": str(output.resolve()),
        "html_sha256": file_sha256(output),
        "score_result_path": str(score_result.resolve()),
        "score_result_sha256": file_sha256(score_result),
        "video_tags": len(groups) * (2 + len(ARMS)),
    }
    (output.parent / "html-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-result", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build(
        args.score_result.expanduser().resolve(strict=True),
        args.media_root.expanduser().resolve(strict=True),
        args.output.expanduser().absolute(),
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

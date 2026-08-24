#!/usr/bin/env python3
"""Build a compact synchronized human review for full-field V4 checkpoints."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    value = json.loads(Path(args.manifest).resolve(strict=True).read_text(encoding="utf-8"))
    cases = value.get("cases")
    arms = value.get("arm_definitions")
    steps = tuple(int(item) for item in value.get("steps", (5, 10, 20, 40)))
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("review manifest has no cases")
    if not isinstance(arms, list) or len(arms) != 4:
        raise RuntimeError("full-field review requires exactly four arm definitions")
    if not steps or len(set(steps)) != len(steps) or any(item <= 0 for item in steps):
        raise RuntimeError("review steps must be unique positive integers")
    initial = steps[0]

    sections = []
    for case in cases:
        fixed = (
            ("Source", "身份与初态权威", case["source"], "source"),
            ("Self-generated T2V", "仅示范目标动作", case["anchor"], "anchor"),
            ("Frozen base", "0 update · matched phase-0 clamp", case["base"], "base"),
        )
        cards = [
            f'<article class="card fixed {esc(kind)}">'
            f'<h3>{esc(title)}</h3><p>{esc(note)}</p>'
            f'<video controls muted loop playsinline preload="auto" src="{esc(path)}"></video>'
            "</article>"
            for title, note, path, kind in fixed
        ]
        for arm in arms:
            key = arm["key"]
            template = case["arms"][key]
            path = template.replace("{step}", str(initial))
            cards.append(
                f'<article class="card candidate" data-arm="{esc(key)}" '
                f'data-template="{esc(template)}"><h3>{esc(arm["title"])}</h3>'
                f'<p>{esc(arm["note"])} · <span class="step-label">{initial}</span> updates</p>'
                f'<video controls muted loop playsinline preload="auto" src="{esc(path)}"></video>'
                "</article>"
            )
        sections.append(
            f'<section class="case" data-iid="{esc(case["iid"])}"><header><div>'
            f'<h2>{esc(case["iid"])} · {esc(case.get("family", ""))}</h2>'
            f'<p>{esc(case["instruction"])}</p></div><div class="actions">'
            '<button class="sync">同步从头播放</button><button class="pause">全部暂停</button>'
            f'</div></header><div class="grid">{"".join(cards)}</div>'
            '<details><summary>可选：记录整例备注</summary>'
            '<textarea placeholder="动作起始/完成/保持，identity、毛色/服饰、背景、camera、物体交互…"></textarea>'
            "</details></section>"
        )

    step_buttons = "".join(
        f'<button class="step{" active" if step == initial else ""}" data-step="{step}">{step}</button>'
        for step in steps
    )
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full-field action training review · Job 140846</title>
<style>
:root{{--bg:#f3f0e8;--panel:#fffdf8;--ink:#17211e;--muted:#64706b;--line:#cec5b4;--green:#146c55;--brown:#8a5327}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.38 system-ui,-apple-system,sans-serif}}
.top{{position:sticky;top:0;z-index:5;padding:10px 16px;background:#f3f0e8f2;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
.top h1{{font-size:20px;margin:0 0 3px}}.top p{{margin:2px 0;color:var(--muted)}}
.toolbar{{display:flex;align-items:center;gap:6px;margin-top:7px;flex-wrap:wrap}}button{{font:inherit;border:1px solid #9e9584;background:#fffaf0;border-radius:7px;padding:6px 9px;cursor:pointer}}
.step.active{{background:var(--green);border-color:var(--green);color:white}}#sync-visible{{margin-left:8px}}
.case{{margin:12px;border:1px solid var(--line);border-radius:12px;padding:10px;background:var(--panel)}}
.case>header{{display:flex;align-items:start;justify-content:space-between;gap:14px}}h2{{font-size:17px;margin:0}}header p{{margin:3px 0 8px;color:var(--muted);max-width:1250px}}.actions{{display:flex;gap:6px;white-space:nowrap}}
.grid{{display:grid;grid-template-columns:repeat(7,minmax(150px,1fr));gap:7px;overflow-x:auto}}
.card{{min-width:150px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:white}}.card h3{{font-size:13px;margin:0;padding:6px 7px 0}}.card p{{font-size:11px;line-height:1.25;min-height:28px;margin:2px 7px 5px;color:var(--muted)}}.anchor h3{{color:var(--brown)}}.candidate h3{{color:var(--green)}}
video{{display:block;width:100%;height:clamp(132px,12.2vw,224px);object-fit:contain;background:#080a09}}
details{{margin-top:7px;color:var(--muted)}}summary{{cursor:pointer}}textarea{{width:100%;min-height:58px;margin-top:6px;border:1px solid var(--line);border-radius:7px;padding:7px;font:inherit}}
@media(max-width:1150px){{.grid{{grid-template-columns:repeat(7,180px)}}video{{height:180px}}}}@media(max-width:700px){{.case>header{{display:block}}.actions{{margin-bottom:7px}}}}
</style></head><body>
<div class="top"><h1>完整时空动作监督 × identity preservation</h1>
<p>没有机器 winner。Source 决定身份/初态；T2V 只示范动作；所有生成列使用同一 seed、40 denoise steps 和 hard1_every_step phase-0 clamp。</p>
<p>先检查是否真正完成动作，再看随 checkpoint 增长是否向 T2V 外观漂移。下方不再放逐列标注框。</p>
<div class="toolbar"><span>Checkpoint:</span>{step_buttons}<button id="sync-visible">同步播放当前可见样本</button><button id="pause-all">全部暂停</button></div></div>
{''.join(sections)}
<script>
let currentStep={initial};
function waitFor(video,event,test,timeout=25000){{return new Promise((resolve,reject)=>{{if(test())return resolve();let timer;const clear=()=>{{clearTimeout(timer);video.removeEventListener(event,done);video.removeEventListener('error',fail)}};const done=()=>{{clear();resolve()}};const fail=()=>{{clear();reject(new Error('video load failed: '+(video.currentSrc||video.src)))}};video.addEventListener(event,done,{{once:true}});video.addEventListener('error',fail,{{once:true}});timer=setTimeout(fail,timeout)}})}}
async function prepare(video){{video.pause();video.muted=true;video.loop=true;if(video.readyState<1)await waitFor(video,'loadedmetadata',()=>video.readyState>=1);video.currentTime=0;if(video.readyState<3)await waitFor(video,'canplay',()=>video.readyState>=3)}}
async function playCase(scope,button){{const videos=[...scope.querySelectorAll('video')],label=button.textContent;scope.dataset.sync='0';button.disabled=true;button.textContent='载入并对齐…';try{{await Promise.all(videos.map(prepare));const result=await Promise.allSettled(videos.map(video=>video.play()));const failed=result.find(item=>item.status==='rejected');if(failed)throw failed.reason;scope.dataset.sync='1';button.textContent='已同步播放'}}catch(error){{console.error(error);button.textContent='播放失败，点击重试'}}finally{{button.disabled=false;setTimeout(()=>{{if(button.textContent==='已同步播放')button.textContent=label}},1000)}}}}
function pauseCase(scope){{scope.dataset.sync='0';scope.querySelectorAll('video').forEach(video=>video.pause())}}
document.querySelectorAll('.case').forEach(scope=>{{scope.dataset.sync='0';scope.querySelector('.sync').onclick=event=>playCase(scope,event.currentTarget);scope.querySelector('.pause').onclick=()=>pauseCase(scope)}});
document.querySelectorAll('.step').forEach(button=>button.onclick=()=>{{currentStep=Number(button.dataset.step);document.querySelectorAll('.step').forEach(item=>item.classList.toggle('active',item===button));document.querySelectorAll('.case').forEach(pauseCase);document.querySelectorAll('.candidate').forEach(card=>{{const video=card.querySelector('video');video.src=card.dataset.template.replace('{{step}}',String(currentStep));video.load();card.querySelector('.step-label').textContent=String(currentStep)}})}});
document.querySelector('#sync-visible').onclick=async()=>{{const visible=[...document.querySelectorAll('.case')].filter(scope=>{{const box=scope.getBoundingClientRect();return box.bottom>0&&box.top<innerHeight}});for(const scope of visible)await playCase(scope,scope.querySelector('.sync'))}};
document.querySelector('#pause-all').onclick=()=>document.querySelectorAll('.case').forEach(pauseCase);
setInterval(()=>document.querySelectorAll('.case[data-sync="1"]').forEach(scope=>{{const videos=[...scope.querySelectorAll('video')],master=videos[0];if(master.paused||master.readyState<2)return;videos.slice(1).forEach(video=>{{if(!video.paused&&video.readyState>=2&&Math.abs(video.currentTime-master.currentTime)>.10)video.currentTime=master.currentTime}})}}),200);
</script></body></html>'''
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

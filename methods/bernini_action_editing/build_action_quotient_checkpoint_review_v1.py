#!/usr/bin/env python3
"""Build the human-authority synchronized checkpoint review for Job 140846."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence


ARMS = (
    ("action_only", "Action quotient only"),
    ("action_only_lowlr", "Action only · LR 5e-5"),
    ("action_noop", "+ no-op retain"),
    ("action_start", "+ phase-0 source"),
    ("action_nuisance", "+ camera/appearance suppression"),
    ("action_start_nuisance", "+ phase-0 + nuisance"),
    ("action_start_nuisance_noop", "+ phase-0 + nuisance + no-op"),
    ("action_start_nuisance_border", "+ phase-0 + nuisance + border"),
)
STEPS = (10, 20, 40, 80, 160)
DEFAULT_AUDIT_FIELDS = (
    ("action", "Action", ("pass", "partial", "noop", "reverse", "wrong", "unclear")),
    ("phase0", "Phase-0", ("source-correct", "already-edited", "unclear")),
    ("identity", "Identity", ("source-like", "anchor-drift", "other-drift", "unclear")),
)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest).resolve(strict=True)
    output = Path(args.output).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("review manifest has no cases")
    arm_definitions = value.get("arm_definitions")
    if arm_definitions is None:
        arms = ARMS
    else:
        if not isinstance(arm_definitions, list) or not arm_definitions:
            raise RuntimeError("arm_definitions must be a non-empty list")
        arms = tuple((item["key"], item["title"]) for item in arm_definitions)
    steps = tuple(int(step) for step in value.get("steps", STEPS))
    if not steps:
        raise RuntimeError("steps must be non-empty")
    initial_step = steps[0]
    audit_fields_value = value.get("audit_fields")
    if audit_fields_value is None:
        audit_fields = DEFAULT_AUDIT_FIELDS
    else:
        if not isinstance(audit_fields_value, list) or not audit_fields_value:
            raise RuntimeError("audit_fields must be a non-empty list")
        audit_fields = tuple(
            (item["key"], item["label"], tuple(item["options"]))
            for item in audit_fields_value
        )

    label_controls = "".join(
        '<label>'
        f'{esc(label)} <select data-label="{esc(key)}"><option value="">—</option>'
        + "".join(f"<option>{esc(option)}</option>" for option in options)
        + "</select></label>"
        for key, label, options in audit_fields
    )

    sections = []
    for case in cases:
        iid = esc(case["iid"])
        cards = [
            f'<article class="card fixed"><h3>Source · identity authority</h3><video controls muted loop preload="metadata" src="{esc(case["source"])}"></video></article>',
            f'<article class="card anchor"><h3>Self-generated T2V · action only</h3><video controls muted loop preload="metadata" src="{esc(case["anchor"])}"></video></article>',
            f'<article class="card fixed"><h3>Frozen base · 0 update</h3><video controls muted loop preload="metadata" src="{esc(case["base"])}"></video></article>',
        ]
        for arm, title in arms:
            template = case["arms"][arm]
            initial = template.replace("{step}", str(initial_step))
            cards.append(
                f'<article class="card candidate" data-arm="{esc(arm)}" data-template="{esc(template)}">'
                f'<h3>{esc(title)} · <span class="step-label">{initial_step}</span> updates</h3>'
                f'<video controls muted loop preload="metadata" src="{esc(initial)}"></video>'
                f'<div class="labels">{label_controls}</div></article>'
            )
        sections.append(
            f'<section class="case" data-iid="{iid}"><header><div><h2>{iid} · {esc(case.get("family", ""))}</h2>'
            f'<p>{esc(case["instruction"])}</p></div><div class="case-actions"><button class="sync">同步从头播放</button><button class="pause">全部暂停</button></div></header>'
            f'<div class="grid">{"".join(cards)}</div><textarea placeholder="整例备注：动作完成、首帧、人物/毛色/服饰、背景/camera、遮挡/交互…"></textarea></section>'
        )

    buttons = "".join(
        f'<button class="step{" active" if step == initial_step else ""}" data-step="{step}">{step}</button>'
        for step in steps
    )
    page_title = value.get("page_title", "Action quotient checkpoint review · Job 140846")
    headline = value.get("headline", "Self-generated action quotient × identity preservation")
    authority_note = value.get(
        "authority_note",
        "人工权威：网页没有“正确答案”或机器 winner。Source 是 identity/start-state authority；T2V anchor 只示范动作，不是外观 target。",
    )
    review_note = value.get(
        "review_note",
        "切换 checkpoint 后，比较动作方向/完成度与 identity 漂移随 step 的 Pareto 变化。每例按钮会把该例所有视频归零并同步播放。",
    )
    download_name = value.get("download_name", "action_quotient_review.json")
    schema_version = value.get("review_output_schema", "action-quotient-human-review-v1")
    download_name_js = json.dumps(download_name, ensure_ascii=False)
    schema_version_js = json.dumps(schema_version, ensure_ascii=False)
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(page_title)}</title>
<style>
:root{{--bg:#f4f0e7;--panel:#fffdf8;--ink:#18221f;--muted:#65716c;--line:#cfc6b4;--accent:#176b55;--anchor:#8b5427}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}}
.top{{position:sticky;top:0;z-index:5;background:#f4f0e7f2;border-bottom:1px solid var(--line);padding:14px 20px;backdrop-filter:blur(10px)}}
.top h1{{margin:0 0 6px;font-size:22px}} .top p{{margin:4px 0;color:var(--muted)}} button,select{{font:inherit;border:1px solid #9e9584;background:#fffaf0;border-radius:8px;padding:7px 10px;cursor:pointer}}
.steps{{display:flex;gap:7px;align-items:center;margin-top:10px}} .step.active{{background:var(--accent);color:white;border-color:var(--accent)}}
.case{{margin:18px;border:1px solid var(--line);border-radius:14px;background:var(--panel);padding:14px}} .case>header{{display:flex;justify-content:space-between;gap:18px;align-items:start}}
h2{{font-size:18px;margin:0}} header p{{margin:5px 0 12px;color:var(--muted);max-width:1050px}} .case-actions{{display:flex;gap:7px;white-space:nowrap}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:10px}} .card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:white}}
.card h3{{font-size:14px;margin:0;padding:9px 10px;background:#f2eee4}} .anchor h3{{color:var(--anchor)}} .candidate h3{{color:var(--accent)}} video{{width:100%;height:auto;background:#080a09;display:block}}
.labels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:6px;padding:8px}} .labels label{{font-size:12px;color:var(--muted)}} .labels select{{display:block;width:100%;margin-top:3px;padding:5px}}
textarea{{width:100%;min-height:72px;margin-top:10px;padding:9px;border:1px solid var(--line);border-radius:9px;font:inherit}} .save{{margin-left:auto;background:#17231f;color:white}}
@media(max-width:1050px){{.grid{{grid-template-columns:repeat(2,minmax(260px,1fr))}}}} @media(max-width:680px){{.grid{{grid-template-columns:1fr}}.case>header{{display:block}}}}
</style></head><body>
<div class="top"><h1>{esc(headline)}</h1>
<p>{esc(authority_note)}</p>
<p>{esc(review_note)}</p>
<div class="steps"><span>Checkpoint:</span>{buttons}<button id="sync-all">同步全部可见视频</button><button class="save" id="save">下载人工标注 JSON</button></div></div>
{''.join(sections)}
<script>
let currentStep={initial_step};
function waitForVideo(video,event,ready,timeout=20000){{return new Promise((resolve,reject)=>{{if(ready())return resolve();let timer;const done=()=>{{clearTimeout(timer);video.removeEventListener(event,done);video.removeEventListener('error',fail);resolve();}};const fail=()=>{{clearTimeout(timer);video.removeEventListener(event,done);video.removeEventListener('error',fail);reject(new Error('video load failed: '+video.currentSrc));}};video.addEventListener(event,done,{{once:true}});video.addEventListener('error',fail,{{once:true}});timer=setTimeout(()=>fail(),timeout);}})}}
async function prepareVideo(video){{video.pause();video.muted=true;video.loop=true;if(video.readyState<1)await waitForVideo(video,'loadedmetadata',()=>video.readyState>=1);if(Math.abs(video.currentTime)>0.01){{video.currentTime=0;await waitForVideo(video,'seeked',()=>Math.abs(video.currentTime)<=0.01);}}else video.currentTime=0;if(video.readyState<3)await waitForVideo(video,'canplay',()=>video.readyState>=3);}}
async function playCase(scope,button){{const videos=[...scope.querySelectorAll('video')];scope.dataset.syncActive='0';const original=button.textContent;button.disabled=true;button.textContent='载入并对齐…';try{{await Promise.all(videos.map(prepareVideo));const results=await Promise.allSettled(videos.map(v=>v.play()));const failures=results.filter(x=>x.status==='rejected');if(failures.length)throw failures[0].reason;scope.dataset.syncActive='1';button.textContent='已同步播放';}}catch(error){{button.textContent='播放失败，点击重试';console.error(error);}}finally{{button.disabled=false;setTimeout(()=>{{if(button.textContent==='已同步播放')button.textContent=original;}},1200);}}}}
function pauseCase(scope){{scope.dataset.syncActive='0';scope.querySelectorAll('video').forEach(v=>v.pause());}}
document.querySelectorAll('.step').forEach(b=>b.onclick=()=>{{currentStep=Number(b.dataset.step);document.querySelectorAll('.case').forEach(pauseCase);document.querySelectorAll('.step').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.candidate').forEach(c=>{{const v=c.querySelector('video');v.src=c.dataset.template.replace('{{step}}',String(currentStep));v.load();c.querySelector('.step-label').textContent=currentStep;}})}});
document.querySelectorAll('.case').forEach(c=>{{c.dataset.syncActive='0';c.querySelector('.sync').onclick=event=>playCase(c,event.currentTarget);c.querySelector('.pause').onclick=()=>pauseCase(c);}});
document.querySelector('#sync-all').onclick=async event=>{{const cases=[...document.querySelectorAll('.case')].filter(c=>{{const r=c.getBoundingClientRect();return r.bottom>0&&r.top<innerHeight;}});for(const c of cases)await playCase(c,c.querySelector('.sync'));}};
setInterval(()=>{{document.querySelectorAll('.case[data-sync-active="1"]').forEach(c=>{{const videos=[...c.querySelectorAll('video')];const master=videos[0];if(master.paused||master.readyState<2)return;for(const video of videos.slice(1)){{if(video.paused||video.readyState<2)continue;if(Math.abs(video.currentTime-master.currentTime)>0.12)video.currentTime=master.currentTime;}}}});}},250);
document.querySelector('#save').onclick=()=>{{const rows=[...document.querySelectorAll('.case')].map(c=>({{iid:c.dataset.iid,checkpoint:currentStep,note:c.querySelector('textarea').value,arms:Object.fromEntries([...c.querySelectorAll('.candidate')].map(card=>[card.dataset.arm,Object.fromEntries([...card.querySelectorAll('select')].map(s=>[s.dataset.label,s.value]))]))}}));const blob=new Blob([JSON.stringify({{schema_version:{schema_version_js},saved_at:new Date().toISOString(),rows}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download={download_name_js};a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}};
</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

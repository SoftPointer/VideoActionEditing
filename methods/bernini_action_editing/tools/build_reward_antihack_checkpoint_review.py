#!/usr/bin/env python3
"""Build a human-only synchronized review page for the anti-hacking sweep."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


ARMS = [
    {
        "key": "detached_rotate_w1",
        "label": "Detached margin · rotate",
        "description": "Rejected branch detached; noop/reverse/incomplete rotate; preference weight 1.0.",
    },
    {
        "key": "detached_hardmix_w1",
        "label": "Detached margin · noop+incomplete",
        "description": "Rejected branch detached; alternates the two hardest failures; preference weight 1.0.",
    },
    {
        "key": "detached_noop_w1",
        "label": "Detached margin · noop only",
        "description": "Rejected branch detached; trains only against no-action outputs.",
    },
    {
        "key": "detached_incomplete_w1",
        "label": "Detached margin · incomplete only",
        "description": "Rejected branch detached; trains only against incomplete actions.",
    },
    {
        "key": "detached_hardmix_pres005",
        "label": "Detached hard-mix + preservation 0.05",
        "description": "Hard-mix detached preference plus a weak source-identity reconstruction branch.",
    },
    {
        "key": "detached_hardmix_pres010",
        "label": "Detached hard-mix + preservation 0.10",
        "description": "Hard-mix detached preference plus a stronger source-identity reconstruction branch.",
    },
    {
        "key": "ref_hardmix_w025_b3",
        "label": "Reference-DPO · hard-mix · w0.25",
        "description": "Reference-relative DPO, beta 3, preference weight 0.25, noop+incomplete.",
    },
    {
        "key": "ref_hardmix_w05_b3",
        "label": "Reference-DPO · hard-mix · w0.50",
        "description": "Reference-relative DPO, beta 3, preference weight 0.50, noop+incomplete.",
    },
]
STEPS = [80, 160, 240, 320]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--context-media-prefix",
        default="../../model_gain_135096/training_review/media",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.eval_manifest.read_text())
    rows = manifest.get("rows", [])
    if len(rows) != 4:
        raise SystemExit(f"expected 4 manifest rows, found {len(rows)}")
    samples = []
    context_root = Path(args.context_media_prefix)
    if not context_root.is_absolute():
        context_root = (args.output.parent / context_root).resolve()
    for row in rows:
        iid = row["iid"]
        media_dir = args.media_root / iid
        candidates = {}
        for arm in ARMS:
            candidates[arm["key"]] = {}
            for step in STEPS:
                filename = f"{arm['key']}_u{step}.mp4"
                path = media_dir / filename
                if not path.is_file():
                    raise SystemExit(f"missing sweep video: {path}")
                candidates[arm["key"]][str(step)] = f"media/{iid}/{filename}"
        long_sft = media_dir / "long_sft_u320.mp4"
        if not long_sft.is_file():
            raise SystemExit(f"missing long-SFT baseline: {long_sft}")
        for filename in ("source.mp4", "anchor.mp4", "frozen_base.mp4", "baseline.mp4"):
            context_path = context_root / iid / filename
            if not context_path.is_file():
                raise SystemExit(f"missing linked context video: {context_path}")
        context = args.context_media_prefix.rstrip("/")
        samples.append(
            {
                "iid": iid,
                "instruction": row["instruction"],
                "context": {
                    "source": f"{context}/{iid}/source.mp4",
                    "anchor": f"{context}/{iid}/anchor.mp4",
                    "frozen": f"{context}/{iid}/frozen_base.mp4",
                    "sft40": f"{context}/{iid}/baseline.mp4",
                    "sft320": f"media/{iid}/long_sft_u320.mp4",
                },
                "candidates": candidates,
            }
        )
    return {
        "schema_version": "reward-antihack-human-review-v1",
        "seed": manifest["inference_seed"],
        "arms": ARMS,
        "steps": STEPS,
        "samples": samples,
    }


def render(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Anti-hacking reward checkpoint 人工审计</title>
<style>
:root{{--bg:#f4f2eb;--panel:#fffdf8;--ink:#17201d;--muted:#65706b;--line:#d6cdbd;--accent:#176b58;--warn:#9b4d1f}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}}
main{{max-width:1900px;margin:auto;padding:22px}} .intro,.workspace{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin-bottom:18px}}
h1,h2,h3,p{{margin-top:0}} .intro p{{max-width:1200px}} code{{background:#eeeae0;border-radius:5px;padding:2px 5px}}
.topbar,.tabs,.sync,.audit,.verdict{{display:flex;gap:9px;align-items:center;flex-wrap:wrap}} label{{color:var(--muted)}}
button,select,input,textarea{{font:inherit}} button,select{{border:1px solid #aea38f;background:#fbfaf5;border-radius:10px;padding:8px 11px;color:var(--ink)}}
button:hover{{border-color:var(--accent)}} button.active,.tabs button[aria-selected="true"]{{background:var(--accent);border-color:var(--accent);color:#fff}}
.tabs{{margin:14px 0}} .tabs button{{font-size:13px}} .arm-description{{color:var(--muted);min-height:22px}}
.sync{{padding:11px;background:#f6f3ea;border:1px solid var(--line);border-radius:12px;margin:12px 0}} .sync input[type=range]{{min-width:260px;flex:1;accent-color:var(--accent)}}
.sync-status{{color:var(--muted);min-width:180px}} .sync-status.error{{color:#a52a2a}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.card{{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#faf8f1}} .card h3,.card p,.audit{{padding:8px 11px;margin:0}}
.card.context{{border-style:dashed}} .card.baseline{{border-color:#9b8c70}} .card.candidate{{border-color:#75a898}}
video{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#050706}} .tag{{color:var(--muted);font-size:12px}}
.audit button{{padding:5px 8px;font-size:12px}} .audit button[data-v="identity_fail"].active,.audit button[data-v="artifact"].active{{background:var(--warn);border-color:var(--warn)}}
.verdict{{margin-top:15px;padding-top:14px;border-top:1px solid var(--line)}} textarea{{flex:1;min-width:320px;min-height:72px;border:1px solid #aea38f;border-radius:10px;padding:9px}}
.saved{{color:var(--muted)}} .human-only{{font-weight:650;color:var(--accent)}}
@media(max-width:720px){{main{{padding:9px}} .sync input[type=range],textarea{{min-width:100%;width:100%}}}}
</style></head><body><main>
<section class="intro"><h1>Anti-hacking reward · checkpoint sweep</h1>
<p class="human-only">这是纯人工审计页：没有机器胜者、没有隐藏正解、没有把相对排序当作动作成功。</p>
<p>每次只比较一个 objective：source、self-generated action anchor、三个 baseline，以及 update 80/160/240/320。所有模型输出使用同一 seed <code>{payload['seed']}</code>。这四个 IID 是训练 program，仅 seed 未参与训练；本页审计早停、动作路径和保持性，不代表 unseen-IID 泛化。请分别判断动作是否完整、是否退化成 noop/reverse、主体是否保持。</p>
<div class="topbar"><label>样本 <select id="iid-select"></select></label><button id="export">导出审计 JSON</button><span class="saved" id="saved">尚未标注</span></div></section>
<section class="workspace"><h2 id="iid-title"></h2><p id="instruction"></p><div class="tabs" id="tabs" role="tablist"></div><p class="arm-description" id="arm-description"></p>
<div class="sync"><button id="sync-play" data-testid="sync-play">同步播放</button><button id="sync-zero">全部归零</button><label>进度 <input id="progress" type="range" min="0" max="1000" value="0"></label><label>速度 <select id="rate"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label><label><input id="normalized" type="checkbox" checked>归一化相位</label><span class="sync-status" id="sync-status" role="status">等待播放</span></div>
<div class="grid" id="grid"></div><div class="verdict"><label>该 objective 最佳 checkpoint <select id="best-step"><option value="">未判断</option><option>80</option><option>160</option><option>240</option><option>320</option><option value="baseline">baseline 更好</option><option value="tie">近似相同</option><option value="none">均不可用</option></select></label><label>结论 <select id="go"><option value="">未判断</option><option value="go">GO</option><option value="conditional">Conditional GO</option><option value="nogo">NO GO</option></select></label><textarea id="note" placeholder="动作方向/完成度、主体与背景保持、artifact、最佳 checkpoint 的原因…"></textarea></div></section>
</main><script id="review-data" type="application/json">{data}</script>
<script>
const spec=JSON.parse(document.getElementById('review-data').textContent);
const state={{schema_version:'reward-antihack-human-verdict-v1',seed:spec.seed,judgments:{{}}}};
const $=s=>document.querySelector(s), iidSelect=$('#iid-select'), tabs=$('#tabs'), grid=$('#grid');
let sampleIndex=0,armIndex=0,animation=0,renderToken=0;
const sample=()=>spec.samples[sampleIndex],arm=()=>spec.arms[armIndex],key=()=>sample().iid+'::'+arm().key;
const judgment=()=>state.judgments[key()]||(state.judgments[key()]={{videos:{{}},best_step:'',go:'',note:''}});
const labels={{action_ok:'动作成功',noop:'No-op',reverse:'方向错误',incomplete:'动作不完整',identity_fail:'主体漂移',artifact:'明显伪影',unclear:'无法判断'}};
const contexts=[['source','Source','输入，不参与候选投票'],['anchor','Self-generated action anchor','仅作动作语义/时序参照'],['frozen','Frozen base · 0 update','相同 seed baseline'],['sft40','No-reward SFT · 40 updates','旧短训练 baseline'],['sft320','No-reward SFT · 320 updates','容量匹配 baseline']];
function card(id,title,src,kind,subtitle){{const audit=kind==='candidate'?`<div class="audit">${{Object.entries(labels).map(([v,l])=>`<button data-video="${{id}}" data-v="${{v}}">${{l}}</button>`).join('')}}</div>`:'';return `<article class="card ${{kind}}" data-video-card="${{id}}"><h3>${{title}}</h3><video controls muted loop playsinline preload="metadata" src="${{src}}"></video><p class="tag">${{subtitle}}</p>${{audit}}</article>`}}
function stop(){{cancelAnimationFrame(animation);animation=0;document.querySelectorAll('video').forEach(v=>v.pause());$('#sync-play').textContent='同步播放';$('#sync-play').classList.remove('active')}}
function videos(){{return [...document.querySelectorAll('#grid video')]}}
function waitMetadata(v){{if(v.readyState>=1)return Promise.resolve();return new Promise(resolve=>{{const done=()=>{{v.removeEventListener('loadedmetadata',done);v.removeEventListener('error',done);resolve()}};v.addEventListener('loadedmetadata',done,{{once:true}});v.addEventListener('error',done,{{once:true}});v.load()}})}}
function phase(v){{return Number.isFinite(v.duration)&&v.duration>0?v.currentTime/v.duration:0}}
function seek(q){{videos().forEach(v=>{{if(Number.isFinite(v.duration)&&v.duration>0)v.currentTime=Math.min(v.duration*.999,Math.max(0,q*v.duration))}});$('#progress').value=Math.round(q*1000)}}
function syncLoop(token){{if(token!==renderToken)return;const vs=videos(),leader=vs[0];if(!leader)return;const q=phase(leader);$('#progress').value=Math.round(q*1000);if($('#normalized').checked&&!leader.paused)vs.slice(1).forEach(v=>{{if(Math.abs(phase(v)-q)>.04)v.currentTime=q*v.duration}});animation=requestAnimationFrame(()=>syncLoop(token))}}
function bindAudit(){{const j=judgment();grid.querySelectorAll('.audit button').forEach(b=>{{const values=j.videos[b.dataset.video]||[];b.classList.toggle('active',values.includes(b.dataset.v));b.onclick=()=>{{const a=j.videos[b.dataset.video]||(j.videos[b.dataset.video]=[]),i=a.indexOf(b.dataset.v);i<0?a.push(b.dataset.v):a.splice(i,1);b.classList.toggle('active');updateSaved()}}}});$('#best-step').value=j.best_step;$('#go').value=j.go;$('#note').value=j.note}}
function render(){{stop();renderToken++;const s=sample(),a=arm();$('#iid-title').textContent=s.iid;$('#instruction').textContent=s.instruction;$('#arm-description').textContent=a.description;[...tabs.children].forEach((b,i)=>b.setAttribute('aria-selected',String(i===armIndex)));let out='';contexts.forEach(([id,title,sub])=>out+=card(id,title,s.context[id],id==='source'||id==='anchor'?'context':'baseline',sub));spec.steps.forEach(step=>{{const id='u'+step;out+=card(id,'Update '+step,s.candidates[a.key][String(step)],'candidate',a.label)}});grid.innerHTML=out;bindAudit();$('#sync-status').textContent='等待播放';$('#sync-status').className='sync-status'}}
function updateSaved(){{$('#saved').textContent='当前会话已记录 '+Object.keys(state.judgments).length+' 个 sample/objective 组合'}}
spec.samples.forEach((s,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=s.iid;iidSelect.appendChild(o)}});
spec.arms.forEach((a,i)=>{{const b=document.createElement('button');b.textContent=a.label;b.setAttribute('role','tab');b.onclick=()=>{{armIndex=i;render()}};tabs.appendChild(b)}});
iidSelect.onchange=e=>{{sampleIndex=+e.target.value;render()}};
$('#sync-play').onclick=async()=>{{const vs=videos(),play=vs.some(v=>v.paused||v.ended);if(!play){{stop();$('#sync-status').textContent='已暂停';return}};const token=renderToken;$('#sync-status').textContent='正在载入…';await Promise.all(vs.map(waitMetadata));if(token!==renderToken)return;const q=phase(vs[0]);seek(q);vs.forEach(v=>{{v.muted=true;v.playbackRate=+$('#rate').value}});const results=await Promise.allSettled(vs.map(v=>v.play()));const failed=results.filter(x=>x.status==='rejected').length;if(failed){{vs.forEach(v=>v.pause());$('#sync-play').textContent='同步播放';$('#sync-play').classList.remove('active');$('#sync-status').className='sync-status error';$('#sync-status').textContent=failed+'/'+vs.length+' 个视频播放失败'}}else{{$('#sync-play').textContent='同步暂停';$('#sync-play').classList.add('active');$('#sync-status').textContent='同步播放中';syncLoop(token)}}}};
$('#sync-zero').onclick=()=>{{stop();seek(0);$('#sync-status').textContent='已归零'}};$('#progress').oninput=e=>seek(+e.target.value/1000);$('#rate').onchange=e=>videos().forEach(v=>v.playbackRate=+e.target.value);
$('#best-step').onchange=e=>{{judgment().best_step=e.target.value;updateSaved()}};$('#go').onchange=e=>{{judgment().go=e.target.value;updateSaved()}};$('#note').oninput=e=>{{judgment().note=e.target.value;updateSaved()}};
$('#export').onclick=()=>{{state.exported_at=new Date().toISOString();const blob=new Blob([JSON.stringify(state,null,2)+'\\n'],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reward-antihack-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};
render();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    output = render(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    media_files = sorted(args.media_root.rglob("*.mp4"))
    expected_media = len(payload["samples"]) * (
        len(payload["arms"]) * len(payload["steps"]) + 1
    )
    if len(media_files) != expected_media:
        raise SystemExit(
            f"expected {expected_media} packaged videos, found {len(media_files)}"
        )
    checksum_path = args.media_root / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.media_root)}\n"
            for path in media_files
        )
    )
    receipt = {
        "schema_version": "reward-antihack-review-html-receipt-v1",
        "html": args.output.name,
        "html_sha256": sha256(args.output),
        "sample_count": len(payload["samples"]),
        "arm_count": len(payload["arms"]),
        "checkpoint_count": len(payload["steps"]),
        "candidate_video_count": len(payload["samples"]) * len(payload["arms"]) * len(payload["steps"]),
        "packaged_video_count": len(media_files),
        "media_sha256sums_sha256": sha256(checksum_path),
        "machine_winner_exposed": False,
        "synchronized_playback": True,
    }
    args.output.with_name("html-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the synchronized human audit for the unseen multi-seed reward sweep."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


CONDITIONS = [
    {"key": "frozen_base", "label": "Frozen base", "role": "zero-update baseline"},
    {"key": "action_sft_u80", "label": "Action-only SFT · u80", "role": "matched no-reward baseline"},
    {"key": "action_sft_u160", "label": "Action-only SFT · u160", "role": "matched no-reward baseline"},
    {"key": "action_sft_u320", "label": "Action-only SFT · u320", "role": "long-training baseline"},
    {"key": "detached_rotate_u80", "label": "Detached rotate · u80", "role": "reward candidate"},
    {"key": "detached_rotate_u160", "label": "Detached rotate · u160", "role": "reward candidate"},
    {"key": "detached_incomplete_u80", "label": "Detached incomplete · u80", "role": "reward candidate"},
    {"key": "detached_incomplete_u160", "label": "Detached incomplete · u160", "role": "reward candidate"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != "bernini-reward-unseen-multiseed-manifest-v1":
        raise SystemExit("unexpected manifest schema")
    rows = manifest.get("rows", [])
    seeds = manifest.get("inference_seeds", [])
    if len(rows) != 4 or len(seeds) != 3:
        raise SystemExit(f"expected four rows and three seeds, got {len(rows)} and {len(seeds)}")
    samples = []
    for row in rows:
        iid = row["iid"]
        iid_root = args.media_root / iid
        context = {}
        for name in ("source", "anchor"):
            path = iid_root / f"{name}.mp4"
            if not path.is_file():
                raise SystemExit(f"missing context media: {path}")
            context[name] = f"media/{iid}/{name}.mp4"
        candidates = {}
        for seed in seeds:
            candidates[str(seed)] = {}
            for condition in CONDITIONS:
                filename = f"{condition['key']}.mp4"
                path = iid_root / f"seed-{seed}" / filename
                if not path.is_file():
                    raise SystemExit(f"missing candidate media: {path}")
                candidates[str(seed)][condition["key"]] = (
                    f"media/{iid}/seed-{seed}/{filename}"
                )
        samples.append(
            {
                "iid": iid,
                "actor_family": row["actor_family"],
                "action_family_id": row["action_family_id"],
                "instruction": row["instruction"],
                "context": context,
                "candidates": candidates,
            }
        )
    return {
        "schema_version": "reward-unseen-multiseed-human-review-v1",
        "manifest_sha256": sha256(args.manifest),
        "seeds": seeds,
        "conditions": CONDITIONS,
        "samples": samples,
    }


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unseen multi-seed reward 人工审计</title>
<style>
:root{--bg:#f1f0e9;--panel:#fffdf8;--ink:#17201d;--muted:#66716c;--line:#d4cab9;--accent:#176b58;--warn:#a14d24;--base:#8b7250;--reward:#477f70}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}
main{max-width:1900px;margin:auto;padding:20px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:17px;padding:18px;margin-bottom:16px}
h1,h2,h3,p{margin-top:0}.human{font-weight:700;color:var(--accent)}code{background:#eee9df;border-radius:5px;padding:2px 5px}
.top,.tabs,.sync,.audit,.flags{display:flex;gap:8px;align-items:center;flex-wrap:wrap}label{color:var(--muted)}
button,select,textarea,input{font:inherit}button,select{border:1px solid #ab9f8b;background:#fbfaf5;color:var(--ink);border-radius:9px;padding:7px 10px}
button:hover{border-color:var(--accent)}button.active,.tabs button[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.sync{padding:10px;background:#f5f1e7;border:1px solid var(--line);border-radius:11px;margin:12px 0}.sync input[type=range]{min-width:250px;flex:1;accent-color:var(--accent)}
.status{min-width:180px;color:var(--muted)}.status.error{color:#b12929}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:11px}
.card{overflow:hidden;border:1px solid var(--line);border-radius:13px;background:#faf8f1}.card.context{border-style:dashed}.card.baseline{border-color:var(--base)}.card.reward{border-color:var(--reward)}
.card h3,.card p,.audit,.flags{padding:7px 10px;margin:0}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#050706}.tag{font-size:12px;color:var(--muted)}
.audit,.flags{border-top:1px solid #e2dacd}.audit button,.flags button{font-size:12px;padding:4px 7px}.flags button.active{background:var(--warn);border-color:var(--warn);color:#fff}
.dashboard table{width:100%;border-collapse:collapse}.dashboard th,.dashboard td{padding:7px;border-bottom:1px solid var(--line);text-align:left}.dashboard .pending{color:var(--muted)}
.note{width:100%;min-height:70px;border:1px solid #ab9f8b;border-radius:9px;padding:8px;margin-top:10px}.saved{color:var(--muted)}
@media(max-width:720px){main{padding:8px}.sync input[type=range]{min-width:100%;width:100%}}
</style></head><body><main>
<section class="panel">
<h1>Unseen-IID multi-seed reward challenge</h1>
<p class="human">纯人工绝对动作审计：没有机器胜者、没有隐藏正解、不允许删除 seed。</p>
<p>四个 sealed confirmation IID、三个预注册 seed、八个条件。每页比较同一 source 和同一 noise seed。先判断每个输出是否真正完成动作，再由页面按预注册规则计算 rescue/harm；identity/scene failure 是独立 veto。</p>
<div class="top"><label>IID <select id="iid"></select></label><button id="export">导出 JSON</button><button id="clear">清除全部标注</button><span class="saved" id="saved"></span></div>
</section>
<section class="panel">
<h2 id="title"></h2><p id="instruction"></p><div class="tabs" id="seeds" role="tablist"></div>
<div class="sync"><button id="sync-play">同步播放</button><button id="sync-zero">全部归零</button><label>进度 <input id="progress" type="range" min="0" max="1000" value="0"></label><label>速度 <select id="rate"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label><label><input id="normalized" type="checkbox" checked>归一化相位</label><span class="status" id="sync-status">等待播放</span></div>
<div class="grid" id="grid"></div><textarea class="note" id="note" placeholder="该 IID/seed 的动作路径、主体保持、难以判定之处…"></textarea>
</section>
<section class="panel dashboard"><h2>由人工标签计算的 paired rescue/harm</h2><p id="coverage"></p><div id="dashboard"></div></section>
</main><script id="review-data" type="application/json">__DATA__</script>
<script>
const spec=JSON.parse(document.getElementById('review-data').textContent);
const storageKey='reward-unseen-multiseed-human-review-v1-'+spec.manifest_sha256;
let state={schema_version:'reward-unseen-multiseed-human-verdict-v1',manifest_sha256:spec.manifest_sha256,judgments:{}};
try{const prior=JSON.parse(localStorage.getItem(storageKey));if(prior&&prior.manifest_sha256===spec.manifest_sha256)state=prior}catch(_){}
const $=s=>document.querySelector(s),iidSelect=$('#iid'),seedTabs=$('#seeds'),grid=$('#grid');
let sampleIndex=0,seedIndex=0,animation=0,renderToken=0;
const actionLabels={action_success:'动作成功',noop:'No-op',reverse:'方向错误',incomplete:'动作不完整',unclear:'无法判断'};
const flagLabels={identity_scene_failure:'主体/场景失败',artifact:'明显伪影'};
const sample=()=>spec.samples[sampleIndex],seed=()=>String(spec.seeds[seedIndex]),key=()=>sample().iid+'::'+seed();
function judgment(){return state.judgments[key()]||(state.judgments[key()]={videos:{},note:''})}
function videoJudgment(id){const j=judgment();return j.videos[id]||(j.videos[id]={action:'',flags:[]})}
function escapeText(value){const d=document.createElement('div');d.textContent=value;return d.innerHTML}
function card(id,title,src,kind,subtitle,audit){
  let controls='';
  if(audit){
    const actions=Object.entries(actionLabels).map(x=>'<button data-video="'+id+'" data-kind="action" data-v="'+x[0]+'">'+x[1]+'</button>').join('');
    const flags=Object.entries(flagLabels).map(x=>'<button data-video="'+id+'" data-kind="flag" data-v="'+x[0]+'">'+x[1]+'</button>').join('');
    controls='<div class="audit">'+actions+'</div><div class="flags">'+flags+'</div>';
  }
  return '<article class="card '+kind+'"><h3>'+escapeText(title)+'</h3><video controls muted loop playsinline preload="metadata" src="'+src+'"></video><p class="tag">'+escapeText(subtitle)+'</p>'+controls+'</article>';
}
function videos(){return [...document.querySelectorAll('#grid video')]}
function stop(){cancelAnimationFrame(animation);animation=0;videos().forEach(v=>v.pause());$('#sync-play').textContent='同步播放';$('#sync-play').classList.remove('active')}
function phase(v){return Number.isFinite(v.duration)&&v.duration>0?v.currentTime/v.duration:0}
function seek(q){videos().forEach(v=>{if(Number.isFinite(v.duration)&&v.duration>0)v.currentTime=Math.min(v.duration*.999,Math.max(0,q*v.duration))});$('#progress').value=Math.round(q*1000)}
function waitMetadata(v){if(v.readyState>=1)return Promise.resolve();return new Promise(resolve=>{const done=()=>{v.removeEventListener('loadedmetadata',done);v.removeEventListener('error',done);resolve()};v.addEventListener('loadedmetadata',done,{once:true});v.addEventListener('error',done,{once:true});v.load()})}
function syncLoop(token){if(token!==renderToken)return;const vs=videos(),leader=vs[0];if(!leader)return;const q=phase(leader);$('#progress').value=Math.round(q*1000);if($('#normalized').checked&&!leader.paused)vs.slice(1).forEach(v=>{if(Number.isFinite(v.duration)&&Math.abs(phase(v)-q)>.04)v.currentTime=q*v.duration});animation=requestAnimationFrame(()=>syncLoop(token))}
function save(){localStorage.setItem(storageKey,JSON.stringify(state));updateDashboard()}
function bindAudit(){
  grid.querySelectorAll('button[data-video]').forEach(b=>{
    const row=videoJudgment(b.dataset.video);
    b.classList.toggle('active',b.dataset.kind==='action'?row.action===b.dataset.v:row.flags.includes(b.dataset.v));
    b.onclick=()=>{
      if(b.dataset.kind==='action'){
        row.action=b.dataset.v;
        grid.querySelectorAll('button[data-video="'+b.dataset.video+'"][data-kind="action"]').forEach(x=>x.classList.toggle('active',x===b));
      }else{
        const i=row.flags.indexOf(b.dataset.v);i<0?row.flags.push(b.dataset.v):row.flags.splice(i,1);b.classList.toggle('active');
      }
      save();
    };
  });
}
function render(){
  stop();renderToken++;
  const s=sample(),sd=seed(),j=judgment();
  $('#title').textContent=s.iid+' · seed '+sd+' · '+s.action_family_id;
  $('#instruction').textContent=s.instruction;
  [...seedTabs.children].forEach((b,i)=>b.setAttribute('aria-selected',String(i===seedIndex)));
  let out=card('source','Source',s.context.source,'context','真实输入；不参与候选投票',false);
  out+=card('anchor','Self-generated action anchor',s.context.anchor,'context','只作为动作/时序参照',false);
  spec.conditions.forEach(c=>{const kind=c.role.includes('reward')?'reward':'baseline';out+=card(c.key,c.label,s.candidates[sd][c.key],kind,c.role,true)});
  grid.innerHTML=out;$('#note').value=j.note;bindAudit();$('#sync-status').textContent='等待播放';$('#sync-status').className='status';
}
const comparisons=[
  ['rotate u80','detached_rotate_u80','action_sft_u80'],
  ['rotate u160','detached_rotate_u160','action_sft_u160'],
  ['incomplete u80','detached_incomplete_u80','action_sft_u80'],
  ['incomplete u160','detached_incomplete_u160','action_sft_u160']
];
function updateDashboard(){
  let labeled=0;Object.values(state.judgments).forEach(j=>Object.values(j.videos||{}).forEach(v=>{if(v.action)labeled++}));
  $('#coverage').textContent='已标注 '+labeled+' / '+(spec.samples.length*spec.seeds.length*spec.conditions.length)+' 个模型输出。';
  let body='<table><thead><tr><th>比较</th><th>完整配对</th><th>rescue</th><th>harm</th><th>net</th><th>rescue IID</th><th>identity reward/base</th><th>结论</th></tr></thead><tbody>';
  comparisons.forEach(c=>{
    let complete=0,rescue=0,harm=0,rewardIdentity=0,baseIdentity=0;const rescueIids=new Set();
    spec.samples.forEach(s=>spec.seeds.forEach(sd=>{
      const j=state.judgments[s.iid+'::'+sd];if(!j)return;
      const r=j.videos&&j.videos[c[1]],b=j.videos&&j.videos[c[2]];
      if(!r||!b||!r.action||!b.action)return;complete++;
      const baseFail=['noop','reverse','incomplete'].includes(b.action);
      if(baseFail&&r.action==='action_success'){rescue++;rescueIids.add(s.iid)}
      else if(b.action==='action_success'&&r.action!=='action_success')harm++;
      if((r.flags||[]).includes('identity_scene_failure'))rewardIdentity++;
      if((b.flags||[]).includes('identity_scene_failure'))baseIdentity++;
    }));
    const net=rescue-harm,preserve=rewardIdentity<=baseIdentity+1;
    let decision='待完成';
    if(complete===12){if(net>=3&&rescueIids.size>=2&&preserve)decision='GO';else if(net>=1&&net<=2&&preserve)decision='Conditional GO';else decision='NO-GO'}
    body+='<tr><td>'+c[0]+'</td><td>'+complete+'/12</td><td>'+rescue+'</td><td>'+harm+'</td><td>'+net+'</td><td>'+rescueIids.size+'</td><td>'+rewardIdentity+'/'+baseIdentity+'</td><td class="'+(complete===12?'':'pending')+'">'+decision+'</td></tr>';
  });
  $('#dashboard').innerHTML=body+'</tbody></table>';$('#saved').textContent='本地已保存 '+Object.keys(state.judgments).length+' 个 IID/seed cell';
}
spec.samples.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.iid+' · '+s.actor_family;iidSelect.appendChild(o)});
spec.seeds.forEach((sd,i)=>{const b=document.createElement('button');b.textContent='seed '+sd;b.setAttribute('role','tab');b.onclick=()=>{seedIndex=i;render()};seedTabs.appendChild(b)});
iidSelect.onchange=e=>{sampleIndex=+e.target.value;render()};
$('#note').oninput=e=>{judgment().note=e.target.value;save()};
$('#sync-play').onclick=async()=>{
  const vs=videos(),play=vs.some(v=>v.paused||v.ended);if(!play){stop();$('#sync-status').textContent='已暂停';return}
  const token=renderToken;$('#sync-status').textContent='正在载入…';await Promise.all(vs.map(waitMetadata));if(token!==renderToken)return;
  const q=phase(vs[0]);seek(q);vs.forEach(v=>{v.muted=true;v.playbackRate=+$('#rate').value});
  const results=await Promise.allSettled(vs.map(v=>v.play())),failed=results.filter(x=>x.status==='rejected').length;
  if(failed){vs.forEach(v=>v.pause());$('#sync-status').className='status error';$('#sync-status').textContent=failed+'/'+vs.length+' 个视频播放失败'}
  else{$('#sync-play').textContent='同步暂停';$('#sync-play').classList.add('active');$('#sync-status').textContent='同步播放中';syncLoop(token)}
};
$('#sync-zero').onclick=()=>{stop();seek(0);$('#sync-status').textContent='已归零'};
$('#progress').oninput=e=>seek(+e.target.value/1000);$('#rate').onchange=e=>videos().forEach(v=>v.playbackRate=+e.target.value);
$('#export').onclick=()=>{state.exported_at=new Date().toISOString();const blob=new Blob([JSON.stringify(state,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reward-unseen-multiseed-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
$('#clear').onclick=()=>{if(confirm('清除本页保存的全部人工标注？')){localStorage.removeItem(storageKey);state={schema_version:'reward-unseen-multiseed-human-verdict-v1',manifest_sha256:spec.manifest_sha256,judgments:{}};render();updateDashboard()}};
render();updateDashboard();
</script></body></html>
"""


def render(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", data)


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(payload))
    media_files = sorted(args.media_root.rglob("*.mp4"))
    expected = len(payload["samples"]) * (
        2 + len(payload["seeds"]) * len(payload["conditions"])
    )
    if len(media_files) != expected:
        raise SystemExit(f"expected {expected} packaged videos, found {len(media_files)}")
    checksums = args.media_root / "SHA256SUMS"
    checksums.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.media_root)}\n"
            for path in media_files
        )
    )
    receipt = {
        "schema_version": "reward-unseen-multiseed-html-receipt-v1",
        "html": args.output.name,
        "html_sha256": sha256(args.output),
        "manifest_sha256": payload["manifest_sha256"],
        "sample_count": len(payload["samples"]),
        "seed_count": len(payload["seeds"]),
        "condition_count": len(payload["conditions"]),
        "candidate_video_count": (
            len(payload["samples"])
            * len(payload["seeds"])
            * len(payload["conditions"])
        ),
        "packaged_video_count": len(media_files),
        "media_sha256sums_sha256": sha256(checksums),
        "machine_winner_exposed": False,
        "human_derived_rescue_harm": True,
        "synchronized_playback": True,
    }
    args.output.with_name("html-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

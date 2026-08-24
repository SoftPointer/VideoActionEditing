#!/usr/bin/env python3
"""Build the pre-training review page for the complex8 multi-anchor bank."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


SCHEMA = "bernini-interaction-complex8-multianchor-authoring-v2"
RECEIPT_SCHEMA = "bernini-interaction-complex8-review-media-receipt-v2"
ROLES = ("action", "noop", "reverse", "incomplete")


class ReviewBuildError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ReviewBuildError(message)


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewBuildError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> Mapping[str, Any]:
    try:
        raw = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,avg_frame_rate,nb_read_frames,width,height",
                "-of", "json", str(path),
            ],
            stderr=subprocess.STDOUT,
        )
        stream = json.loads(raw)["streams"][0]
    except (OSError, subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError) as error:
        raise ReviewBuildError(f"ffprobe failed for {path}: {error}") from error
    value = {
        "codec_name": stream.get("codec_name"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_read_frames": int(stream.get("nb_read_frames", -1)),
        "width": int(stream.get("width", -1)),
        "height": int(stream.get("height", -1)),
    }
    if value["codec_name"] != "h264" or value["avg_frame_rate"] != "25/1" or value["nb_read_frames"] != 81:
        fail(f"media contract differs for {path}: {value}")
    if value["width"] <= 0 or value["height"] <= 0:
        fail(f"invalid video geometry for {path}")
    return value


def validate_authoring(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if value.get("schema_version") != SCHEMA:
        fail("authoring schema differs")
    if value.get("training_authorized_before_video_review") is not False:
        fail("authoring accidentally authorizes training")
    if value.get("qwen_used") is not False:
        fail("Qwen boundary differs")
    if value.get("event_count") != 8 or value.get("variants_per_event") != 4 or value.get("positive_count") != 32:
        fail("authoring count contract differs")
    if value.get("roles") != list(ROLES):
        fail("role closure differs")
    events = value.get("events")
    if not isinstance(events, list) or len(events) != 8:
        fail("event closure differs")
    if [event.get("ordinal") for event in events] != list(range(8)):
        fail("event order differs")
    if len({event.get("event_id") for event in events}) != 8:
        fail("event IDs are not unique")
    seeds: set[int] = set()
    for event in events:
        for field in (
            "source_iid", "event_id", "category", "review_requirement",
            "geometry_source_video", "action", "constraints",
        ):
            if not isinstance(event.get(field), str) or not event[field].strip():
                fail(f"event {event.get('ordinal')} field {field} differs")
        variants = event.get("variants")
        if not isinstance(variants, list) or len(variants) != 4:
            fail(f"event {event['event_id']} variant closure differs")
        if [variant.get("variant_id") for variant in variants] != ["v0", "v1", "v2", "v3"]:
            fail(f"event {event['event_id']} variant order differs")
        for variant in variants:
            seed = variant.get("seed")
            if not isinstance(seed, int) or seed in seeds:
                fail("variant seed closure differs")
            seeds.add(seed)
            if not isinstance(variant.get("setup"), str) or "81-frame" not in variant["setup"]:
                fail("variant setup differs")
    if len(seeds) != 32:
        fail("seed count differs")
    return events


def relative(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path, output)).as_posix()


def validate_media(
    *, authoring: Mapping[str, Any], media_root: Path, source_root: Path, output: Path
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    events = validate_authoring(authoring)
    rendered: list[Mapping[str, Any]] = []
    media_audit: list[Mapping[str, Any]] = []
    for event in events:
        source = (source_root / event["source_iid"] / "source.mp4").resolve(strict=True)
        source_probe = probe(source)
        row: dict[str, Any] = {
            "ordinal": event["ordinal"],
            "source_iid": event["source_iid"],
            "event_id": event["event_id"],
            "category": event["category"],
            "requirement": event["review_requirement"],
            "action": event["action"],
            "source": relative(source, output),
            "variants": [],
        }
        for variant in event["variants"]:
            root = media_root / f"e{event['ordinal']:02d}_{event['event_id']}" / variant["variant_id"]
            marker = root / "REVIEW_MEDIA_COMPLETE"
            if not marker.is_file():
                fail(f"missing review marker: {marker}")
            receipt_path = root / "review_receipt.json"
            receipt = read_json(receipt_path)
            expected = {
                "schema_version": RECEIPT_SCHEMA,
                "complete": True,
                "event_ordinal": event["ordinal"],
                "source_iid": event["source_iid"],
                "event_id": event["event_id"],
                "category": event["category"],
                "variant_id": variant["variant_id"],
                "seed": variant["seed"],
                "lineage": "source_free_pure_t2v",
                "training_performed": False,
                "qwen_used": False,
            }
            for key, value in expected.items():
                if receipt.get(key) != value:
                    fail(f"receipt {receipt_path} field {key} differs")
            prompt = " ".join((variant["setup"], event["action"], event["constraints"]))
            if receipt.get("prompt") != prompt or receipt.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
                fail(f"prompt binding differs: {receipt_path}")
            paths: dict[str, str] = {}
            files: dict[str, Any] = {}
            receipt_media = receipt.get("media")
            if not isinstance(receipt_media, dict) or set(receipt_media) != set(ROLES):
                fail(f"receipt media closure differs: {receipt_path}")
            geometry: tuple[int, int] | None = None
            for role in ROLES:
                file_name = "t2v.mp4" if role == "action" else f"{role}.mp4"
                path = (root / file_name).resolve(strict=True)
                actual_probe = probe(path)
                actual_sha = sha256(path)
                declared = receipt_media[role]
                if declared.get("file") != file_name or declared.get("sha256") != actual_sha:
                    fail(f"media SHA binding differs: {path}")
                if declared.get("probe") != actual_probe:
                    fail(f"media probe binding differs: {path}")
                current = (actual_probe["width"], actual_probe["height"])
                if geometry is not None and current != geometry:
                    fail(f"role geometry differs within {root}")
                geometry = current
                paths[role] = relative(path, output)
                files[role] = {"sha256": actual_sha, **actual_probe}
            variant_row = {
                "variant_id": variant["variant_id"],
                "seed": variant["seed"],
                "setup": variant["setup"],
                "prompt": prompt,
                "paths": paths,
            }
            row["variants"].append(variant_row)
            media_audit.append(
                {
                    "event_id": event["event_id"],
                    "variant_id": variant["variant_id"],
                    "seed": variant["seed"],
                    "receipt": relative(receipt_path.resolve(), output),
                    "media": files,
                }
            )
        rendered.append(row)
        media_audit.append(
            {
                "event_id": event["event_id"],
                "role": "real_source_display_only",
                "source_iid": event["source_iid"],
                "sha256": sha256(source),
                "probe": source_probe,
            }
        )
    audit = {
        "schema_version": "bernini-interaction-complex8-multianchor-review-audit-v2",
        "complete": True,
        "training_authorized": False,
        "training_performed": False,
        "qwen_used": False,
        "event_count": 8,
        "positive_count": 32,
        "derived_negative_count": 96,
        "display_video_count": 136,
        "roles": list(ROLES),
        "all_controls_share_exact_positive_appearance": True,
        "media": media_audit,
    }
    return rendered, audit


def page(events: Sequence[Mapping[str, Any]], audit_url: str, bank_id: str) -> str:
    payload = json.dumps(events, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    bank = html.escape(bank_id)
    audit_link = html.escape(audit_url)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Complex interaction T2V data review</title>
<style>
:root{{font-family:system-ui,-apple-system,sans-serif;color:#17211e;background:#f2efe7}}*{{box-sizing:border-box}}body{{margin:0;padding:8px}}header{{position:sticky;top:0;z-index:5;background:#f2efe7f4;padding:6px;border-bottom:1px solid #bcb29f}}h1{{font-size:18px;margin:0 0 5px}}button{{font:inherit;border:1px solid #9f9581;border-radius:7px;padding:5px 8px;background:#fffaf0;cursor:pointer}}button.active{{background:#176c57;color:white;border-color:#176c57}}.banner{{font-size:12px;line-height:1.4;padding:7px 9px;background:#fff3cf;border-left:4px solid #b26a00;margin:7px 0}}.toolbar{{display:flex;gap:5px;align-items:center;flex-wrap:wrap;font-size:12px}}.spacer{{flex:1}}.event{{background:#fffdf8;border:1px solid #c8bda8;border-radius:10px;margin:7px 0;padding:6px}}.event-head{{display:flex;gap:8px;align-items:center;font-size:12px;margin-bottom:5px}}.event-title{{font-weight:800;font-size:14px}}.requirement{{color:#6f3423;flex:1}}.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:4px}}.card{{border:1px solid #d6ccb9;border-radius:7px;overflow:hidden;background:#ece7dd}}.source{{background:#e4edf1}}.label{{font-size:10.5px;font-weight:750;padding:4px 5px;min-height:34px;line-height:1.25}}.sub{{font-size:9.5px;color:#61706a;font-weight:500}}video{{display:block;width:100%;height:142px;object-fit:contain;background:#121414}}.decisions{{display:flex;gap:3px;padding:4px;flex-wrap:wrap}}.decisions button{{font-size:9.5px;padding:3px 5px}}.decisions button.selected{{background:#1e6a55;color:white}}details{{font-size:10px;margin-top:4px;color:#596761}}.error{{display:none;color:#a32219;font-size:10px;padding:3px}}.bad .error{{display:block}}#progress{{font-weight:750}}@media(max-width:1100px){{.grid{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:700px){{.grid{{grid-template-columns:repeat(2,1fr)}}video{{height:170px}}}}
</style></head><body>
<header><h1>Complex interaction pure-T2V bank · pre-training review</h1><div class="toolbar"><strong>显示角色：</strong><button data-role="action" class="active">Action positive</button><button data-role="noop">Noop</button><button data-role="reverse">Reverse</button><button data-role="incomplete">Incomplete</button><span class="spacer"></span><span id="progress">已标记 0/32</span><button id="playAll">同步播放可见视频</button><button id="pauseAll">全部暂停</button><button id="export">导出审核 JSON</button><a href="{audit_link}">媒体审计</a></div></header>
<div class="banner"><strong>尚未训练。</strong>本页只用于决定哪些数据允许进入下一阶段。共 8 个复杂事件 × 4 个不同外观/场景的 source-free T2V anchor。Noop、Reverse、Incomplete 都由同一 positive 的帧确定性构造，因此服装、人物、背景与颜色完全一致；请先在 Action positive 检查动作和视频质量，再切换 negatives 检查对照是否符合定义。按钮标记只保存在本浏览器，点击“导出审核 JSON”后才形成可交付决定。</div>
<main id="root"></main>
<script>
const bankId={json.dumps(bank_id)};const events={payload};const roles=['action','noop','reverse','incomplete'];let role='action';const root=document.querySelector('#root');
const key=bankId+':review-v2';let decisions={{}};try{{decisions=JSON.parse(localStorage.getItem(key)||'{{}}')}}catch{{decisions={{}}}}
function visibleVideos(scope=document){{return [...scope.querySelectorAll('video:not([data-hidden="true"])')]}}
async function restart(list){{list.forEach(v=>{{v.pause();v.currentTime=0;v.muted=true}});await Promise.allSettled(list.map(v=>v.play()))}}
function save(){{localStorage.setItem(key,JSON.stringify(decisions));const n=Object.keys(decisions).length;document.querySelector('#progress').textContent=`已标记 ${{n}}/32`;for(const card of document.querySelectorAll('[data-candidate]')){{const id=card.dataset.candidate;for(const b of card.querySelectorAll('[data-decision]'))b.classList.toggle('selected',decisions[id]===b.dataset.decision)}}}}
for(const e of events){{const s=document.createElement('section');s.className='event';s.dataset.event='';s.innerHTML=`<div class="event-head"><span class="event-title">${{e.ordinal+1}}. ${{e.event_id}}</span><span class="requirement">${{e.requirement}}</span><button data-play>同步播放本事件</button><button data-pause>暂停</button></div><div class="grid"></div><details><summary>动作定义</summary>${{e.action}}</details>`;const grid=s.querySelector('.grid');const source=document.createElement('div');source.className='card source';source.innerHTML=`<div class="label">Real source · display only<br><span class="sub">${{e.source_iid}} · 不进入 T2V conditioning</span></div><video controls muted playsinline preload="metadata" src="${{e.source}}"></video><div class="error"></div>`;grid.appendChild(source);for(const v of e.variants){{const id=e.event_id+':'+v.variant_id;const c=document.createElement('div');c.className='card';c.dataset.candidate=id;c.innerHTML=`<div class="label"><span data-role-label>Action positive</span> · ${{v.variant_id}}<br><span class="sub">seed ${{v.seed}} · source-free T2V</span></div><video controls muted playsinline preload="metadata"></video><div class="decisions"><button data-decision="usable">可用</button><button data-decision="action_failed">动作失败</button><button data-decision="quality_failed">质量失败</button><button data-decision="uncertain">不确定</button></div><details><summary>外观/场景 prompt</summary>${{v.setup}}</details><div class="error"></div>`;const vid=c.querySelector('video');for(const r of roles)vid.dataset[r]=v.paths[r];vid.src=v.paths.action;for(const b of c.querySelectorAll('[data-decision]'))b.onclick=()=>{{decisions[id]=b.dataset.decision;save()}};grid.appendChild(c)}}s.querySelector('[data-play]').onclick=()=>restart(visibleVideos(s));s.querySelector('[data-pause]').onclick=()=>visibleVideos(s).forEach(v=>v.pause());root.appendChild(s)}}
function setRole(next){{role=next;for(const b of document.querySelectorAll('[data-role]'))b.classList.toggle('active',b.dataset.role===role);for(const c of document.querySelectorAll('[data-candidate]')){{const v=c.querySelector('video');v.pause();v.src=v.dataset[role];v.load();c.querySelector('[data-role-label]').textContent={{action:'Action positive',noop:'Noop · frame 0 hold',reverse:'Reverse · exact frame reversal',incomplete:'Incomplete · half action then hold'}}[role]}}}}
for(const b of document.querySelectorAll('[data-role]'))b.onclick=()=>setRole(b.dataset.role);document.querySelector('#playAll').onclick=()=>restart(visibleVideos());document.querySelector('#pauseAll').onclick=()=>visibleVideos().forEach(v=>v.pause());document.querySelector('#export').onclick=()=>{{const data={{schema_version:'bernini-interaction-complex8-human-review-v2',bank_id:bankId,exported_at:new Date().toISOString(),decisions}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+'\\n'],{{type:'application/json'}}));a.download='interaction_complex8_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};for(const v of document.querySelectorAll('video'))v.addEventListener('error',()=>{{const c=v.closest('.card');c.classList.add('bad');c.querySelector('.error').textContent='加载失败：'+(v.currentSrc||v.src)}});save();
</script></body></html>"""


def build(args: argparse.Namespace) -> None:
    authoring_path = Path(args.authoring).expanduser().resolve(strict=True)
    media_root = Path(args.media_root).expanduser().resolve(strict=True)
    source_root = Path(args.source_review_root).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    authoring = read_json(authoring_path)
    events, audit = validate_media(
        authoring=authoring, media_root=media_root, source_root=source_root, output=output
    )
    audit = {
        **audit,
        "bank_id": authoring["bank_id"],
        "authoring_sha256": sha256(authoring_path),
        "authoring": relative(authoring_path, output),
    }
    audit_path = output / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "index.html").write_text(
        page(events, "audit.json", authoring["bank_id"]), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "events": 8, "positives": 32, "videos": 136, "training_authorized": False}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--authoring", required=True)
    result.add_argument("--media-root", required=True)
    result.add_argument("--source-review-root", required=True)
    result.add_argument("--output", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        build(parser().parse_args(argv))
    except ReviewBuildError as error:
        print(f"ERROR: {error}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

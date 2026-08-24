#!/usr/bin/env python3
"""Build a compact, auditable video-comparison page from a JSON specification."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
from typing import Any


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain_video(value: Any, *, label: str) -> Path:
    if not isinstance(value, str):
        raise BuildError(f"{label} path must be a string")
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink() or path.suffix.lower() != ".mp4":
        raise BuildError(f"{label} is not a plain MP4: {path}")
    return path


def text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BuildError(f"{label} must be non-empty text")
    return value.strip()


def card(item: dict[str, Any], event_index: int, row_index: int, card_index: int,
         media: Path) -> dict[str, Any]:
    source = plain_video(item.get("path"), label="card")
    suffix = source.suffix.lower()
    name = f"e{event_index:02d}-r{row_index:02d}-c{card_index:02d}-{sha256(source)[:12]}{suffix}"
    destination = media / name
    if destination.exists():
        raise BuildError(f"duplicate output media: {destination}")
    shutil.copy2(source, destination)
    return {
        "label": text(item.get("label"), label="card label"),
        "detail": str(item.get("detail", "")).strip(),
        "path": f"media/{name}",
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
    }


def build(spec_path: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output == Path("/"):
        raise BuildError("output must be a fresh non-root path")
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("events"), list):
        raise BuildError("spec must contain an events array")
    output.mkdir(parents=True)
    media = output / "media"
    media.mkdir()
    events = []
    for event_index, raw_event in enumerate(raw["events"]):
        if not isinstance(raw_event, dict):
            raise BuildError("event must be an object")
        raw_rows = raw_event.get("rows")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise BuildError("event rows must be a non-empty array")
        rows = []
        for row_index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, dict):
                raise BuildError("row must be an object")
            raw_cards = raw_row.get("cards")
            if not isinstance(raw_cards, list) or not 1 <= len(raw_cards) <= 5:
                raise BuildError("every comparison row must contain one to five cards")
            rows.append({
                "title": text(raw_row.get("title"), label="row title"),
                "note": str(raw_row.get("note", "")).strip(),
                "cards": [
                    card(item, event_index, row_index, card_index, media)
                    for card_index, item in enumerate(raw_cards)
                ],
            })
        events.append({
            "name": text(raw_event.get("name"), label="event name"),
            "instruction": str(raw_event.get("instruction", "")).strip(),
            "rows": rows,
        })
    receipt = {
        "schema_version": "compact-video-comparison-v1",
        "title": text(raw.get("title"), label="title"),
        "subtitle": str(raw.get("subtitle", "")).strip(),
        "machine_correct_answer_shown": False,
        "human_annotation_controls_shown": False,
        "max_cards_per_row": 5,
        "events": events,
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(render(receipt), encoding="utf-8")
    (output / "COMPLETE").write_text("complete\n", encoding="ascii")
    return receipt


def render_card(item: dict[str, Any]) -> str:
    detail = (
        f'<div class="detail">{html.escape(str(item["detail"]))}</div>'
        if item["detail"] else ""
    )
    return f'''<article class="card"><div class="label">{html.escape(str(item["label"]))}</div>
<video controls muted playsinline preload="metadata" src="{html.escape(str(item["path"]), quote=True)}"></video>{detail}</article>'''


def render(receipt: dict[str, Any]) -> str:
    sections = []
    for event_index, event in enumerate(receipt["events"]):
        rows = []
        for row_index, row in enumerate(event["rows"]):
            selector = f"#event-{event_index}-row-{row_index} video"
            rows.append(f'''<div class="row" id="event-{event_index}-row-{row_index}">
<div class="rowhead"><div><h3>{html.escape(str(row["title"]))}</h3><p>{html.escape(str(row["note"]))}</p></div>
<button onclick="syncPlay('{selector}',this)">同步播放本行</button></div>
<div class="grid">{"".join(render_card(item) for item in row["cards"])}</div></div>''')
        sections.append(f'''<section class="event" id="event-{event_index}"><header><div><h2>{html.escape(str(event["name"]))}</h2>
<p>{html.escape(str(event["instruction"]))}</p></div><button onclick="syncPlay('#event-{event_index} video',this)">同步播放本事件</button></header>{"".join(rows)}</section>''')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(receipt["title"]))}</title><style>
:root{{--bg:#f3efe7;--panel:#fffdf8;--ink:#18211e;--muted:#65706c;--line:#d4c9b8;--accent:#176b57}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.35 system-ui,-apple-system,sans-serif}}
.top{{position:sticky;top:0;z-index:5;display:flex;gap:9px;align-items:center;padding:8px 12px;background:#f3efe7ee;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
.top h1{{font-size:17px;margin:0}}.top span{{color:var(--muted);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.top button{{margin-left:auto}}
button{{border:1px solid #9e927f;background:#fffaf1;border-radius:8px;padding:7px 10px;font-weight:750;cursor:pointer;white-space:nowrap}}button:disabled{{opacity:.5}}
main{{padding:8px}}.event{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:9px;margin-bottom:10px}}.event>header,.rowhead{{display:flex;align-items:start;gap:10px}}.event>header>div,.rowhead>div{{flex:1;min-width:0}}
h2{{font-size:18px;margin:0 0 2px}}h3{{font-size:14px;margin:0}}p{{margin:0;color:var(--muted);font-size:12px}}.event>header>div>p{{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
.row{{margin-top:8px;padding-top:7px;border-top:1px solid #e2d9cc}}.rowhead{{margin-bottom:5px}}.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px;align-items:start}}
.card{{min-width:0;border:1px solid #6f9286;border-radius:8px;overflow:hidden;background:#fff}}.label{{min-height:39px;padding:5px 7px;display:flex;align-items:center;font-size:12px;line-height:1.18;font-weight:800}}
video{{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#101110}}.detail{{min-height:30px;padding:5px 7px;color:var(--muted);font-size:11px}}
@media(max-width:1050px){{.grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}}}@media(max-width:650px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.top span{{display:none}}}}
</style></head><body><div class="top"><h1>{html.escape(str(receipt["title"]))}</h1><span>{html.escape(str(receipt["subtitle"]))}</span><button onclick="pauseAll()">全部暂停</button></div><main>{"".join(sections)}</main>
<script>function ready(v){{if(v.readyState>=1)return Promise.resolve();return new Promise((ok,bad)=>{{v.addEventListener('loadedmetadata',ok,{{once:true}});v.addEventListener('error',bad,{{once:true}});v.load()}})}}
async function syncPlay(selector,button){{const vs=[...document.querySelectorAll(selector)];const old=button.textContent;button.disabled=true;button.textContent='加载并对齐…';try{{vs.forEach(v=>{{v.pause();v.muted=true;v.currentTime=0}});await Promise.all(vs.map(ready));vs.forEach(v=>v.currentTime=0);const r=await Promise.allSettled(vs.map(v=>v.play()));if(r.some(x=>x.status==='rejected'))throw Error('浏览器拒绝部分视频播放')}}catch(e){{alert('同步播放失败：'+e.message+'。请使用 http://127.0.0.1 服务打开并检查 media。')}}finally{{button.disabled=false;button.textContent=old}}}}function pauseAll(){{document.querySelectorAll('video').forEach(v=>v.pause())}}</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = build(Path(args.spec).resolve(strict=True), Path(args.output).resolve())
    print(json.dumps({"events": len(receipt["events"]), "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

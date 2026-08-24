#!/usr/bin/env python3
"""Build the five-way visual causal review for online-anchor LoRA training."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any


EVENTS = {
    0: ("pour liquid into cup", "E00_pour-liquid-into-cup"),
    2: ("twist-pull mushroom", "E02_twist-pull-mushroom"),
    4: ("close door, then drawer", "E04_close-door-then-drawer"),
    7: ("players contact, then separate", "E07_players-contact-then-separate"),
}
ARMS = (
    ("frozen", "Frozen + online anchor + SGA/ANC"),
    ("no_anchor", "No-anchor trained + same solver"),
    ("action_noop", "Action−noop trained + same solver"),
    ("dynamic_static", "Dynamic−static trained + same solver"),
    ("hybrid", "Hybrid contrast trained + same solver"),
)


class ReviewError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReviewError(f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_video(source: Path, media: Path, name: str) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise ReviewError(f"missing plain video: {source}")
    destination = media / name
    if destination.exists():
        raise ReviewError(f"duplicate media name: {destination}")
    shutil.copy2(source, destination)
    return {
        "path": f"media/{name}",
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
    }


def anchor_directory(stage: Path, event: int) -> Path:
    matches = sorted(
        (stage / "interaction_complex8_multianchor_v2_r1").glob(f"e{event:02d}_*")
    )
    if len(matches) != 1 or not matches[0].is_dir():
        raise ReviewError(f"anchor directory differs for E{event:02d}: {matches}")
    return matches[0]


def output_path(training_root: Path, event: int, stem: str, arm: str) -> Path:
    return (
        training_root
        / "dynaedit_decode_v1"
        / f"e{event:02d}"
        / f"{stem}_{arm}_S8_ONLINE_ANCHOR_REAL_SGA_ANC.mp4"
    )


def build(stage: Path, training_root: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output == Path("/"):
        raise ReviewError("output must be a fresh non-root directory")
    output.mkdir(parents=True)
    media = output / "media"
    media.mkdir()
    events: list[dict[str, Any]] = []
    for event, (short_name, stem) in EVENTS.items():
        candidate = read_json(
            stage
            / "interaction_complex8_rv2v_candidates_v1"
            / f"complex8-e{event:02d}-rv2v-s0"
            / "pair-v5-rollout-receipt.json"
        )["candidate"]
        source = copy_video(
            Path(str(candidate["source_video"])), media, f"e{event:02d}-source.mp4"
        )
        anchor = copy_video(
            anchor_directory(stage, event) / "v0" / "t2v.mp4",
            media,
            f"e{event:02d}-pure-t2v-anchor-v0.mp4",
        )
        results = []
        for arm, label in ARMS:
            results.append(
                {
                    "arm": arm,
                    "label": label,
                    "media": copy_video(
                        output_path(training_root, event, stem, arm),
                        media,
                        f"e{event:02d}-{arm}.mp4",
                    ),
                }
            )
        events.append(
            {
                "event": event,
                "name": short_name,
                "instruction": str(candidate["complete_caption"]),
                "source": source,
                "anchor": anchor,
                "results": results,
            }
        )
    receipt = {
        "schema_version": "bernini-online-anchor-lora-dynaedit-review-v1",
        "events": events,
        "arms": [arm for arm, _ in ARMS],
        "machine_correct_answer_shown": False,
        "human_annotation_controls_shown": False,
        "same_source_anchor_prompt_seed_and_solver": True,
        "training_steps": 8,
        "probe_events_seen_during_training": False,
        "solver": {
            "steps": 40,
            "early_sga_candidates": 5,
            "annealed_anc": True,
            "online_pure_t2v_anchor": True,
            "transport_strength": 0.25,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(render(receipt), encoding="utf-8")
    (output / "COMPLETE").write_text("complete\n", encoding="ascii")
    return receipt


def card(media: dict[str, Any], label: str, classes: str = "") -> str:
    return f"""<article class="card {html.escape(classes)}">
    <div class="label">{html.escape(label)}</div>
    <video controls muted playsinline preload="metadata" src="{html.escape(str(media['path']), quote=True)}"></video>
  </article>"""


def render(receipt: dict[str, Any]) -> str:
    sections = []
    for item in receipt["events"]:
        event = int(item["event"])
        references = "".join(
            (
                card(item["source"], "Source authority", "source"),
                card(item["anchor"], "Pure-T2V action anchor (appearance irrelevant)", "anchor"),
            )
        )
        results = "".join(
            card(result["media"], str(result["label"]), "result")
            for result in item["results"]
        )
        sections.append(
            f"""<section class="event" id="event-{event}">
  <header><div><h2>Event {event:02d} · {html.escape(str(item['name']))}</h2>
  <p>{html.escape(str(item['instruction']))}</p></div>
  <button onclick="syncPlay('#event-{event} video',this)">同步播放本事件</button></header>
  <div class="reference-grid">{references}</div>
  <div class="divider">同 source / anchor / prompt / seed / 40-step SGA+ANC solver；仅模型权重不同</div>
  <div class="result-grid">{results}</div>
</section>"""
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Online-anchor LoRA × real SGA/ANC</title>
<style>
:root{{--bg:#f3efe7;--panel:#fffdf8;--ink:#18211e;--muted:#66716d;--line:#d4c9b8;--green:#176b57;--orange:#a96b2d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.35 system-ui,-apple-system,sans-serif}}
.top{{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:8px;padding:8px 12px;background:#f3efe7ee;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
.top h1{{font-size:17px;margin:0}}.top span{{margin-left:auto;color:var(--muted);font-size:12px}}button{{border:1px solid #9e927f;background:#fffaf1;border-radius:8px;padding:7px 10px;font-weight:750;cursor:pointer;white-space:nowrap}}button:disabled{{opacity:.5}}
main{{padding:8px}}.event{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:9px;margin-bottom:10px}}.event>header{{display:flex;gap:10px;align-items:start;margin-bottom:7px}}.event>header>div{{flex:1;min-width:0}}h2{{font-size:18px;margin:0 0 2px}}p{{margin:0;color:var(--muted);font-size:12px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
.reference-grid{{display:grid;grid-template-columns:repeat(2,minmax(220px,320px));gap:6px;justify-content:start}}.result-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px}}.divider{{margin:7px 0 5px;padding-top:6px;border-top:1px solid #e2d9cc;color:#46534e;font-size:11px;font-weight:750}}
.card{{min-width:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff}}.card.source{{border-color:#438d7b}}.card.anchor{{border-color:#bd7a38}}.card.result{{border-color:#6f9286}}.label{{height:38px;padding:5px 7px;display:flex;align-items:center;font-size:12px;line-height:1.18;font-weight:800}}video{{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#111}}
@media(max-width:1100px){{.result-grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}}}@media(max-width:700px){{.result-grid,.reference-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.top span{{display:none}}}}@media(max-width:480px){{.event>header{{display:block}}.event>header button{{margin-top:5px}}}}
</style></head><body><div class="top"><h1>Online-anchor LoRA × real SGA/ANC</h1><button onclick="pauseAll()">全部暂停</button><span>8-step canary · E02/E04 unseen-event probe · 无机器正确答案</span></div><main>{''.join(sections)}</main>
<script>
function ready(v){{if(v.readyState>=1)return Promise.resolve();return new Promise((ok,bad)=>{{v.addEventListener('loadedmetadata',ok,{{once:true}});v.addEventListener('error',bad,{{once:true}});v.load()}})}}
async function syncPlay(selector,button){{const vs=[...document.querySelectorAll(selector)];const old=button.textContent;button.disabled=true;button.textContent='加载并对齐…';try{{vs.forEach(v=>{{v.pause();v.muted=true;v.currentTime=0}});await Promise.all(vs.map(ready));vs.forEach(v=>v.currentTime=0);const r=await Promise.allSettled(vs.map(v=>v.play()));if(r.some(x=>x.status==='rejected'))throw Error('浏览器拒绝部分视频播放')}}catch(e){{alert('同步播放失败：'+e.message+'。请通过 http://127.0.0.1 服务打开并检查 media。')}}finally{{button.disabled=false;button.textContent=old}}}}
function pauseAll(){{document.querySelectorAll('video').forEach(v=>v.pause())}}
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = build(
        Path(args.stage).resolve(strict=True),
        Path(args.training_root).resolve(strict=True),
        Path(args.output).resolve(),
    )
    print(json.dumps({"events": len(receipt["events"]), "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

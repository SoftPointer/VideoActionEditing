#!/usr/bin/env python3
"""Build a compact human review for true Complex8 SGA/ANC training."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any


EVENTS = (0, 2, 4, 7)
STEPS = (1, 10, 32)
PROFILES = (
    ("sgaanc", "FM-score weighted consensus · ρ.25 · gain.10"),
    ("hard_sga", "Hard FM-score selection · ρ0 · gain.10"),
    ("uniform_anc", "Uniform anchor-gradient mean · ρ1 · gain.10"),
    ("no_gain", "FM-score weighted consensus · ρ.25 · no gain"),
)


class ReviewError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReviewError(f"JSON root differs: {path}")
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


def event_anchor_directory(stage: Path, event: int) -> Path:
    matches = sorted(
        (stage / "interaction_complex8_multianchor_v2_r1").glob(f"e{event:02d}_*")
    )
    if len(matches) != 1 or not matches[0].is_dir():
        raise ReviewError(f"event anchor directory differs: {event}")
    return matches[0]


def build(stage: Path, output: Path) -> dict[str, Any]:
    release = stage / "complex8_sga_anc_training_v1"
    candidate_root = stage / "interaction_complex8_rv2v_candidates_v1"
    if output.exists() or output == Path("/"):
        raise ReviewError("output must be a fresh non-root directory")
    output.mkdir(parents=True)
    media = output / "media"
    media.mkdir()
    rows: list[dict[str, Any]] = []
    for event in EVENTS:
        candidate_receipt = read_json(
            candidate_root
            / f"complex8-e{event:02d}-rv2v-s0"
            / "pair-v5-rollout-receipt.json"
        )
        candidate = candidate_receipt["candidate"]
        anchor_root = event_anchor_directory(stage, event)
        source = copy_video(
            Path(str(candidate["source_video"])), media, f"e{event:02d}-source.mp4"
        )
        anchors = [
            copy_video(
                anchor_root / f"v{variant}" / "t2v.mp4",
                media,
                f"e{event:02d}-anchor-v{variant}.mp4",
            )
            for variant in (1, 2, 3)
        ]
        frozen = copy_video(
            release
            / "decode_v1"
            / "frozen"
            / f"event_{event:02d}"
            / "step_0000"
            / "output.mp4",
            media,
            f"e{event:02d}-frozen.mp4",
        )
        checkpoints: dict[str, list[dict[str, Any]]] = {}
        for step in STEPS:
            cards: list[dict[str, Any]] = []
            for profile, label in PROFILES:
                cards.append(
                    {
                        "profile": profile,
                        "label": label,
                        "media": copy_video(
                            release
                            / "decode_v1"
                            / profile
                            / f"event_{event:02d}"
                            / f"step_{step:04d}"
                            / "output.mp4",
                            media,
                            f"e{event:02d}-{profile}-s{step:04d}.mp4",
                        ),
                    }
                )
            checkpoints[str(step)] = cards
        rows.append(
            {
                "event": event,
                "caption": str(candidate["complete_caption"]),
                "source": source,
                "anchors": anchors,
                "frozen": frozen,
                "checkpoints": checkpoints,
            }
        )
    receipt = {
        "schema_version": "bernini-complex8-cross-anchor-flow-training-review-v2",
        "events": rows,
        "steps": list(STEPS),
        "profiles": [profile for profile, _ in PROFILES],
        "machine_correct_answer_shown": False,
        "source_is_identity_and_initial_state_authority": True,
        "anchors_are_motion_demonstrations_not_appearance_targets": True,
        "dynaedit_sga_implemented": False,
        "dynaedit_anc_implemented": False,
        "correction": (
            "The historical profile directory names sgaanc/hard_sga/uniform_anc "
            "refer to FM-loss candidate weighting and gradient aggregation, not "
            "DynaEdit's sampling-time SGA or annealed inter-step noise correlation."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(render(receipt), encoding="utf-8")
    (output / "COMPLETE").write_text("complete\n", encoding="ascii")
    return receipt


def video(path: str, classes: str, label: str, note: str) -> str:
    return f"""
    <article class="card {html.escape(classes)}">
      <div class="label">{html.escape(label)}</div>
      <video controls muted playsinline preload="metadata" src="{html.escape(path, quote=True)}"></video>
      <div class="note">{html.escape(note)}</div>
    </article>"""


def render(receipt: dict[str, Any]) -> str:
    sections: list[str] = []
    for row in receipt["events"]:
        event = int(row["event"])
        refs = [
            video(row["source"]["path"], "source", "Source authority", "身份、物体、场景与初态权威"),
        ]
        for index, anchor in enumerate(row["anchors"], start=1):
            note = "本页推理使用该motion flow；外观不可学习" if index == 1 else "训练候选motion；外观不可学习"
            refs.append(video(anchor["path"], "anchor", f"Pure-T2V motion anchor v{index}", note))
        bands: list[str] = []
        for step in STEPS:
            group = f"event-{event}-step-{step}"
            cards = [
                video(
                    row["frozen"]["path"],
                    "frozen",
                    "Frozen base · 0 update",
                    "同source / instruction / seed；本行基准",
                )
            ]
            cards.extend(
                video(
                    card["media"]["path"],
                    "trained",
                    card["label"],
                    f"真实训练 checkpoint · update {step}",
                )
                for card in row["checkpoints"][str(step)]
            )
            bands.append(
                f"""<div class="band"><div class="band-head"><h3>Checkpoint update {step}</h3>
                <button onclick="syncPlay('.{group} video',this)">同步本行</button></div>
                <div class="result-grid {group}">{''.join(cards)}</div></div>"""
            )
        caption = html.escape(str(row["caption"]))
        sections.append(
            f"""<section class="event" id="event-{event}"><header><div><h2>Event {event:02d}</h2>
            <p>{caption}</p></div><button onclick="syncPlay('#event-{event} video',this)">同步本事件</button></header>
            <div class="ref-grid">{''.join(refs)}</div>{''.join(bands)}</section>"""
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Complex8 cross-anchor flow training diagnostic</title>
<style>
:root{{--bg:#f3efe7;--panel:#fffdf8;--ink:#18211e;--muted:#68726e;--line:#d7ccba;--green:#176b57;--orange:#a96b2d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.35 system-ui,-apple-system,sans-serif}}
.top{{position:sticky;top:0;z-index:5;display:flex;gap:8px;align-items:center;padding:8px 12px;background:#f3efe7ee;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
.top h1{{font-size:17px;margin:0}}.top span{{margin-left:auto;color:var(--muted);font-size:12px}}button{{border:1px solid #9c907c;background:#fffaf1;border-radius:8px;padding:7px 10px;font-weight:750;cursor:pointer;white-space:nowrap}}button:disabled{{opacity:.5}}
main{{padding:7px}}.event{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px;margin-bottom:9px}}.event>header{{display:flex;gap:10px;align-items:start;margin-bottom:7px}}.event>header>div{{flex:1;min-width:0}}h2{{font-size:18px;margin:0 0 2px}}p{{margin:0;color:var(--muted);font-size:12px;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;overflow:hidden}}
.ref-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin-bottom:8px}}.result-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px}}.band{{border-top:1px solid #e2dacd;padding-top:5px;margin-top:5px}}.band-head{{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}}h3{{font-size:13px;margin:0;color:#44514c}}
.card{{min-width:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff;display:grid;grid-template-rows:38px auto 35px}}.card.source{{border-color:#438d7b}}.card.anchor{{border-color:#bd7a38}}.card.trained{{border-color:#6f9286}}.label{{padding:5px 7px;font-weight:800;font-size:12px;line-height:1.2;display:flex;align-items:center}}video{{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#111}}.note{{padding:4px 7px;color:var(--muted);font-size:10.5px;line-height:1.25;overflow:hidden}}
@media(max-width:1000px){{.result-grid,.ref-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.top span{{display:none}}}}@media(max-width:600px){{.event>header{{display:block}}.event>header button{{margin-top:5px}}}}
</style></head><body><div class="top"><h1>Complex8 · cross-anchor flow training diagnostic</h1><button onclick="pauseAll()">全部暂停</button><span>不是DynaEdit SGA/ANC · 没有机器“正确答案” · Source管身份/初态</span></div><main>{''.join(sections)}</main>
<script>
function ready(v){{if(v.readyState>=1)return Promise.resolve();return new Promise((ok,bad)=>{{v.addEventListener('loadedmetadata',ok,{{once:true}});v.addEventListener('error',bad,{{once:true}});v.load();}})}}
async function syncPlay(selector,button){{const vs=[...document.querySelectorAll(selector)];button.disabled=true;const old=button.textContent;button.textContent='加载并对齐…';try{{vs.forEach(v=>{{v.pause();v.muted=true;v.currentTime=0}});await Promise.all(vs.map(ready));vs.forEach(v=>v.currentTime=0);const r=await Promise.allSettled(vs.map(v=>v.play()));if(r.some(x=>x.status==='rejected'))throw Error('浏览器拒绝部分视频播放')}}catch(e){{alert('同步播放失败：'+e.message+'。请通过 http://127.0.0.1 服务打开。')}}finally{{button.disabled=false;button.textContent=old}}}}
function pauseAll(){{document.querySelectorAll('video').forEach(v=>v.pause())}}
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(Path(args.stage).resolve(strict=True), Path(args.output).resolve())
    print(json.dumps({"events": len(result["events"]), "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

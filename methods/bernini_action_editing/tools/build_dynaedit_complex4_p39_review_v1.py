#!/usr/bin/env python3
"""Build a compact four-column review for the Round-89 complex interactions."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


EVENTS = (
    (0, "Pour pitcher into cup", "COMPLEX4_E0_pour-liquid-into-cup_OBSERVER_P39_R8.mp4", "Lift and align the pitcher, show a continuous stream and rising cup level, then return the same pitcher upright."),
    (2, "Twist and pull mushroom", "COMPLEX4_E2_twist-pull-mushroom_OBSERVER_P39_R8.mp4", "Grasp and twist the same rooted mushroom, detach and lift it, leave the original hole empty."),
    (4, "Close door, then drawer", "COMPLEX4_E4_close-door-then-drawer_OBSERVER_P39_R8.mp4", "Close the hinged lower door first, then push the separate upper drawer inward and hold both closed."),
    (7, "Players contact, then separate", "COMPLEX4_E7_players-contact-then-separate_OBSERVER_P39_R8.mp4", "Push off once, create a visible gap between the same two players, then continue on distinct paths."),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(root: Path) -> None:
    media = root / "media"
    if not media.is_dir():
        raise SystemExit(f"missing media directory: {media}")
    sections: list[str] = []
    manifest_events: list[dict[str, object]] = []
    for event, name, result_name, action in EVENTS:
        prefix = f"e{event:02d}"
        cards = (
            ("Source authority", f"{prefix}-source.mp4", "Identity, objects, scene, camera and frame 0 are judged only against this video."),
            ("Pure-T2V action anchor", f"{prefix}-t2v-anchor-v0.mp4", "Online action/no-op donor. Its person, clothing, objects and background are not target content."),
            ("Frozen RV2V", f"{prefix}-frozen-s0.mp4", "Matched source/instruction baseline. This is a failure control, not a target for small deviations."),
            ("SGA + ANC observer · P39", result_name, "Five early candidates, online action/no-op relation and ANC; one weak final source projection. Not trained and not labelled correct."),
        )
        rows: list[dict[str, object]] = []
        markup: list[str] = []
        for label, filename, note in cards:
            path = media / filename
            if not path.is_file() or path.stat().st_size == 0:
                continue
            rows.append({"label": label, "path": f"media/{filename}", "note": note, "bytes": path.stat().st_size, "sha256": sha256(path)})
            markup.append(f'''<article class="card">
<h3>{html.escape(label)}</h3>
<video controls muted loop playsinline preload="metadata" data-group="e{event}" src="media/{html.escape(filename)}"></video>
<p>{html.escape(note)}</p>
</article>''')
        manifest_events.append({"event": event, "name": name, "strict_action": action, "cards": rows})
        sections.append(f'''<section>
<div class="section-head"><div><h2>Event {event:02d} · {html.escape(name)}</h2><p class="action">{html.escape(action)}</p></div><button onclick="playGroup('e{event}')">同步播放本事件</button></div>
<div class="grid">{''.join(markup)}</div>
</section>''')
    manifest = {
        "schema_version": "dynaedit-complex4-p39-review-v1",
        "source_is_identity_authority": True,
        "anchor_appearance_is_target": False,
        "frozen_is_target": False,
        "p39_is_ground_truth": False,
        "events": manifest_events,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Round 89 · complex action transfer</title><style>
:root{{--bg:#f5f1e9;--panel:#fffdf8;--ink:#17221f;--muted:#62706c;--line:#cfc5b3;--accent:#176b57}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.4 system-ui,-apple-system,sans-serif}}
header{{position:sticky;top:0;z-index:4;display:flex;gap:10px;align-items:center;padding:12px 18px;background:rgba(245,241,233,.97);border-bottom:1px solid var(--line)}}
header h1{{font-size:20px;margin:0 auto 0 0}}button{{padding:9px 14px;border:1px solid #988c78;border-radius:10px;background:#fffaf1;font-weight:700;cursor:pointer}}
main{{padding:14px 18px 40px}}section{{margin:0 0 18px;padding:14px;border:1px solid var(--line);border-radius:16px;background:var(--panel)}}
.section-head{{display:flex;gap:14px;align-items:center;margin-bottom:10px}}.section-head>div{{margin-right:auto}}h2{{font-size:19px;margin:0}}.action{{margin:3px 0 0;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;align-items:stretch}}.card{{display:grid;grid-template-rows:52px auto 1fr;min-width:0;border:1px solid var(--line);border-radius:13px;overflow:hidden;background:#fff}}
h3{{margin:0;padding:10px 12px;font-size:16px;display:flex;align-items:center}}video{{display:block;width:100%;aspect-ratio:13/18;object-fit:contain;background:#070a09}}.card p{{margin:0;padding:9px 12px;color:var(--muted)}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:620px){{.grid{{grid-template-columns:1fr}}header{{flex-wrap:wrap}}}}
</style></head><body><header><h1>Round 89 · complex interaction transfer</h1><button onclick="playAll()">全部同步播放</button><button onclick="pauseAll()">全部暂停</button></header><main>{''.join(sections)}</main>
<script>function vids(s='video'){{return [...document.querySelectorAll(s)]}}function start(xs){{xs.forEach(v=>{{v.pause();v.currentTime=0}});Promise.all(xs.map(v=>v.play().catch(()=>null)))}}function playGroup(g){{start(vids(`video[data-group="${{g}}"]`))}}function playAll(){{start(vids())}}function pauseAll(){{vids().forEach(v=>v.pause())}}</script></body></html>'''
    (root / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    build(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

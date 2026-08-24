#!/usr/bin/env python3
"""Build a portable synchronized four-video review bundle for the MEV audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .audit import (
    MANIFEST_SCHEMA,
    QWEN_SUMMARY_SCHEMA,
    SM_SUMMARY_SCHEMA,
    assert_not_protected_write,
    file_sha256,
    load_json,
    write_json,
)
from .generation_controller import _probe_video as _opencv_probe_video


BUNDLE_SCHEMA = "mev-action-anchor-target-gap-review-bundle-v1"
ROLE_ORDER = ("source", "real_target", "anchor", "frozen_base")
ROLE_LABELS = {
    "source": "Source",
    "real_target": "Real target",
    "anchor": "T2V action anchor",
    "frozen_base": "Frozen-base RV2V",
}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _role_paths(sample: Mapping[str, Any]) -> dict[str, str]:
    generation = sample["generation"]
    return {
        "source": generation["normalized_source"]["path"],
        "real_target": sample["real_target"]["path"],
        "anchor": generation["anchor"]["path"],
        "frozen_base": generation["frozen_base"]["path"],
    }


def _index_by_pair(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in result:
            raise ValueError(f"{label} contains an invalid or duplicate pair_id")
        result[pair_id] = row
    return result


def build_cases(
    manifest: Mapping[str, Any],
    qwen: Mapping[str, Any],
    semantic_moments: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema differs")
    if qwen.get("schema_version") != QWEN_SUMMARY_SCHEMA:
        raise ValueError("Qwen summary schema differs")
    if semantic_moments.get("schema_version") != SM_SUMMARY_SCHEMA:
        raise ValueError("SemanticMoments summary schema differs")
    samples = _index_by_pair(manifest.get("samples", []), "manifest")
    qwen_rows = _index_by_pair(qwen.get("pairs", []), "Qwen summary")
    sm_rows = _index_by_pair(semantic_moments.get("pairs", []), "SemanticMoments summary")
    if not samples or set(samples) != set(qwen_rows) or set(samples) != set(sm_rows):
        raise ValueError("manifest, Qwen, and SemanticMoments pair sets differ")
    if semantic_moments.get("manifest_digest") != manifest.get("manifest_digest"):
        raise ValueError("SemanticMoments summary is not bound to this manifest")

    cases = []
    for pair_id, sample in sorted(samples.items(), key=lambda item: item[1]["pair_prefix"]):
        qwen_row = qwen_rows[pair_id]
        sm_row = sm_rows[pair_id]
        role_scores = qwen_row.get("role_scores")
        if not isinstance(role_scores, Mapping) or set(role_scores) != {"anchor", "frozen_base"}:
            raise ValueError(f"Qwen role scores absent for {sample['pair_prefix']}")
        similarities = sm_row.get("similarities")
        if not isinstance(similarities, Mapping) or not {"m3", "m23"}.issubset(similarities):
            raise ValueError(f"SemanticMoments similarities absent for {sample['pair_prefix']}")
        cases.append({
            "pair_id": pair_id,
            "pair_prefix": sample["pair_prefix"],
            "instruction": sample["instruction"],
            "source_action_caption": sample["source_action_caption"],
            "target_action_caption": sample["target_action_caption"],
            "winner": qwen_row["winner"],
            "winner_reason": qwen_row["reason"],
            "gate_pass_winners": qwen_row.get("gate_pass_winners", []),
            "direct_pairwise_winners": qwen_row.get("direct_pairwise_winners", []),
            "role_scores": role_scores,
            "qwen_passes": qwen_row.get("passes", []),
            "m3": similarities["m3"],
            "m23": similarities["m23"],
            "source_paths": _role_paths(sample),
        })
    return cases


def _score_table(case: Mapping[str, Any]) -> str:
    labels = (
        ("action_semantics", "Semantics"),
        ("temporal_order", "Order"),
        ("action_completion", "Completion"),
        ("reference_motion_match", "Ref motion"),
        ("gate_score", "Gate"),
    )
    rows = []
    for key, label in labels:
        anchor = case["role_scores"]["anchor"][key]
        base = case["role_scores"]["frozen_base"][key]
        rows.append(
            f"<tr><th>{_e(label)}</th><td>{anchor:.2f}</td><td>{base:.2f}</td>"
            f"<td class=\"delta\">{anchor - base:+.2f}</td></tr>"
        )
    return "".join(rows)


def _qwen_evidence(case: Mapping[str, Any]) -> str:
    blocks = []
    for pass_index, evidence in enumerate(case["role_scores"]["anchor"].get("evidence", [])):
        base_evidence = case["role_scores"]["frozen_base"].get("evidence", [])
        base_values = base_evidence[pass_index] if pass_index < len(base_evidence) else []
        anchor_items = "".join(f"<li>{_e(value)}</li>" for value in evidence)
        base_items = "".join(f"<li>{_e(value)}</li>" for value in base_values)
        blocks.append(
            f"<div class=\"pass-evidence\"><h5>Pass {pass_index + 1}</h5>"
            f"<div><strong>Anchor</strong><ul>{anchor_items}</ul></div>"
            f"<div><strong>Frozen-base</strong><ul>{base_items}</ul></div></div>"
        )
    return "".join(blocks)


def _case_html(case: Mapping[str, Any], ordinal: int) -> str:
    prefix = case["pair_prefix"]
    winner = case["winner"]
    winner_label = {
        "anchor": "Anchor wins",
        "frozen_base": "Frozen-base wins",
        "abstain": "Abstain",
        "tie": "Tie",
    }.get(winner, winner)
    media = "".join(
        f"""<div class="video-cell">
          <div class="video-label">{_e(ROLE_LABELS[role])}</div>
          <video muted playsinline preload="metadata" data-src="media/{_e(prefix)}/{role}.mp4"
                 aria-label="{_e(ROLE_LABELS[role])} for {prefix}"></video>
        </div>"""
        for role in ROLE_ORDER
    )
    m3 = case["m3"]["anchor_minus_frozen_base"]
    m23 = case["m23"]["anchor_minus_frozen_base"]
    conflict = (
        winner in {"anchor", "frozen_base"}
        and ((m3 > 0 and winner == "frozen_base") or (m3 < 0 and winner == "anchor"))
    )
    conflict_badge = '<span class="badge conflict">Qwen / M3 conflict</span>' if conflict else ""
    return f"""
    <article class="case-card" id="case-{_e(prefix)}" data-sync-group data-winner="{_e(winner)}"
             data-conflict="{'yes' if conflict else 'no'}">
      <header class="case-head">
        <div><span class="ordinal">{ordinal:02d}</span><code>{_e(prefix)}</code>
          <span class="badge winner-{_e(winner)}">{_e(winner_label)}</span>{conflict_badge}</div>
        <h2>{_e(case['target_action_caption'])}</h2>
        <p class="instruction">{_e(case['instruction'])}</p>
        <p class="source-caption"><strong>Source event:</strong> {_e(case['source_action_caption'])}</p>
      </header>
      <div class="videos-grid">{media}</div>
      <div class="sync-controls">
        <button type="button" data-action="restart">↺ 从头同步播放</button>
        <button type="button" data-action="play">▶ 同步播放</button>
        <button type="button" data-action="pause">Ⅱ 暂停</button>
        <button type="button" data-action="back" aria-label="previous frame">−1 帧</button>
        <button type="button" data-action="forward" aria-label="next frame">+1 帧</button>
        <input data-action="seek" type="range" min="0" max="80" step="1" value="0"
               aria-label="normalized frame index">
        <output data-time>F00 / 80 · 0.00s</output>
        <label>速度 <select data-action="speed">
          <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
          <option value="1" selected>1×</option><option value="2">2×</option>
        </select></label>
      </div>
      <div class="metrics-grid">
        <div class="metric-panel"><h3>Qwen double-pass mean</h3>
          <table><thead><tr><th>Axis</th><th>Anchor</th><th>Base</th><th>Δ</th></tr></thead>
          <tbody>{_score_table(case)}</tbody></table></div>
        <div class="metric-panel semantic"><h3>SemanticMoments diagnostic</h3>
          <dl><div><dt>M3 Δ</dt><dd>{m3:+.6f}</dd></div>
              <div><dt>M23 Δ</dt><dd>{m23:+.6f}</dd></div>
              <div><dt>Gate pass winners</dt><dd>{_e(' / '.join(case['gate_pass_winners']))}</dd></div>
              <div><dt>Strict reason</dt><dd>{_e(case['winner_reason'])}</dd></div></dl>
          <p>Δ = anchor − frozen-base；SemanticMoments 不判断方向与完成度。</p></div>
      </div>
      <details><summary>展开 Qwen 双 pass 可见证据</summary>{_qwen_evidence(case)}</details>
    </article>"""


def render_html(cases: Sequence[Mapping[str, Any]]) -> str:
    counts = {name: sum(case["winner"] == name for case in cases) for name in ("frozen_base", "anchor", "abstain", "tie")}
    m3_base = sum(case["m3"]["anchor_minus_frozen_base"] < 0 for case in cases)
    m3_anchor = sum(case["m3"]["anchor_minus_frozen_base"] > 0 for case in cases)
    cards = "".join(_case_html(case, ordinal) for ordinal, case in enumerate(cases, 1))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>MEV Action Anchor vs Frozen-base · {len(cases)}-way synchronized review</title>
  <style>
    :root {{ --bg:#080b10; --panel:#111720; --panel2:#171f2b; --line:#293445;
      --ink:#f4f7fb; --muted:#9ba9bc; --green:#69dda7; --blue:#78b8ff;
      --amber:#f7c66b; --red:#ff7d88; --violet:#c4a3ff; }}
    * {{ box-sizing:border-box }} html {{ scroll-behavior:smooth }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 85% 0,#17243c 0,transparent 34rem),var(--bg);
      font:15px/1.5 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif }}
    button,select,input {{ font:inherit }} code {{ color:#c9d5e7 }}
    .wrap {{ width:min(1560px,calc(100% - 32px)); margin:auto }}
    .top {{ position:sticky; top:0; z-index:30; padding:10px 0; border-bottom:1px solid var(--line);
      background:rgba(8,11,16,.88); backdrop-filter:blur(16px) }}
    .toolbar {{ display:flex; align-items:center; flex-wrap:wrap; gap:8px }}
    .brand {{ margin-right:auto; font-weight:800 }}
    button,.chip,select {{ border:1px solid var(--line); border-radius:9px; padding:7px 11px;
      color:var(--ink); background:#151d28; cursor:pointer }}
    button:hover,.chip:hover,.chip.active {{ border-color:var(--blue); background:#1b2b40 }}
    .hero {{ padding:60px 0 34px }} h1 {{ margin:0 0 12px; font-size:clamp(34px,6vw,72px); line-height:1; letter-spacing:-.05em }}
    .hero p {{ max-width:920px; color:#bfcbda; font-size:17px }}
    .stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:26px }}
    .stat {{ padding:18px; border:1px solid var(--line); border-radius:14px; background:rgba(17,23,32,.88) }}
    .stat b {{ display:block; font-size:30px }} .stat span {{ color:var(--muted) }}
    .note {{ margin:22px 0 0; padding:13px 15px; border-left:3px solid var(--blue); background:#101a28; color:#c7d2df }}
    .cases {{ display:grid; gap:28px; padding:24px 0 70px }}
    .case-card {{ border:1px solid var(--line); border-radius:18px; overflow:hidden; background:rgba(17,23,32,.96); box-shadow:0 22px 70px #0005 }}
    .case-card[hidden] {{ display:none }} .case-head {{ padding:20px 22px 17px }}
    .case-head h2 {{ margin:10px 0 4px; font-size:22px }} .case-head p {{ margin:5px 0; color:var(--muted) }}
    .ordinal {{ display:inline-grid; place-items:center; width:31px; height:25px; margin-right:8px; border-radius:7px; background:#273345 }}
    .badge {{ display:inline-flex; margin-left:8px; padding:3px 8px; border-radius:999px; font-size:11px; font-weight:800 }}
    .winner-frozen_base {{ background:var(--green); color:#06140d }} .winner-anchor {{ background:var(--blue); color:#07111d }}
    .winner-abstain,.winner-tie {{ background:var(--amber); color:#1a1002 }} .conflict {{ background:var(--violet); color:#13091e }}
    .videos-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--line) }}
    .video-cell {{ min-width:0; background:#05070a }} .video-label {{ padding:8px 11px; color:#d9e2ee; background:#101722; font-size:12px; font-weight:750 }}
    video {{ display:block; width:100%; aspect-ratio:16/10; object-fit:contain; background:#030405; cursor:pointer }}
    .sync-controls {{ display:grid; grid-template-columns:auto auto auto auto auto minmax(180px,1fr) auto auto; gap:8px; align-items:center; padding:13px 15px; border-top:1px solid var(--line); border-bottom:1px solid var(--line) }}
    .sync-controls input {{ width:100%; accent-color:var(--blue) }} .sync-controls output {{ min-width:118px; color:#c2cede; font:12px ui-monospace,monospace }}
    .sync-controls label {{ display:flex; gap:6px; align-items:center; color:var(--muted); white-space:nowrap }}
    .metrics-grid {{ display:grid; grid-template-columns:1.15fr .85fr; gap:12px; padding:16px }}
    .metric-panel {{ padding:15px; border:1px solid var(--line); border-radius:12px; background:var(--panel2) }}
    .metric-panel h3 {{ margin:0 0 10px; font-size:15px }} table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums }}
    th,td {{ padding:6px 8px; border-bottom:1px solid var(--line); text-align:right }} th:first-child {{ text-align:left }}
    td.delta {{ color:var(--blue) }} dl {{ margin:0 }} dl div {{ display:flex; justify-content:space-between; gap:12px; padding:6px 0; border-bottom:1px solid var(--line) }}
    dt {{ color:var(--muted) }} dd {{ margin:0; text-align:right; font-family:ui-monospace,monospace }} .semantic p {{ color:var(--muted); font-size:12px }}
    details {{ margin:0 16px 17px; padding:12px 14px; border:1px solid var(--line); border-radius:11px; background:#0d131c }}
    summary {{ cursor:pointer; font-weight:700 }} .pass-evidence {{ display:grid; grid-template-columns:110px 1fr 1fr; gap:12px; padding-top:12px }}
    .pass-evidence h5 {{ margin:0 }} .pass-evidence ul {{ margin:6px 0; padding-left:19px; color:var(--muted) }}
    @media(max-width:1050px) {{ .videos-grid {{ grid-template-columns:repeat(2,1fr) }} .sync-controls {{ grid-template-columns:repeat(5,auto); }} .sync-controls input {{ grid-column:1/-1 }} .metrics-grid {{ grid-template-columns:1fr }} }}
    @media(max-width:650px) {{ .wrap {{ width:min(100% - 18px,1560px) }} .stats,.videos-grid {{ grid-template-columns:1fr }} .sync-controls {{ display:flex; flex-wrap:wrap }} .sync-controls input {{ flex-basis:100% }} .pass-evidence {{ grid-template-columns:1fr }} }}
  </style>
</head>
<body>
  <nav class="top"><div class="wrap toolbar">
    <span class="brand">MEV · synchronized action review</span>
    <button class="chip active" data-filter="all">全部 {len(cases)}</button>
    <button class="chip" data-filter="frozen_base">Base 胜 {counts['frozen_base']}</button>
    <button class="chip" data-filter="anchor">Anchor 胜 {counts['anchor']}</button>
    <button class="chip" data-filter="abstain">Abstain {counts['abstain']}</button>
    <button class="chip" data-filter="conflict">Qwen/M3 冲突</button>
    <button type="button" id="pause-all">全部暂停</button>
  </div></nav>
  <header class="hero wrap">
    <h1>Anchor vs Frozen-base</h1>
    <p>每个样本按 Source、真实 target、T2V action anchor、source-conditioned frozen-base RV2V 四路并排。
       点击任意视频或本组按钮即可同步播放；拖动滑杆与逐帧按钮会同时定位四路。</p>
    <div class="stats">
      <div class="stat"><b>{counts['frozen_base']}</b><span>Qwen strict · frozen-base wins</span></div>
      <div class="stat"><b>{counts['anchor']}</b><span>Qwen strict · anchor wins</span></div>
      <div class="stat"><b>{counts['abstain']}</b><span>slot-swap unstable · abstain</span></div>
      <div class="stat"><b>{m3_base} / {m3_anchor}</b><span>SemanticMoments M3 · base / anchor</span></div>
    </div>
    <div class="note">同步口径：四路均为 review-only exact-81 / 25 fps H.264 派生文件，覆盖各自完整事件。
      它们只用于可视化，不改变评测输入、分数或 `/vast/users/guangyi.chen/dataset/MEV/MEV` 中的任何文件。</div>
  </header>
  <main class="wrap cases">{cards}</main>
  <script>
    const groups = [...document.querySelectorAll('[data-sync-group]')];
    const state = new WeakMap();
    const FRAME_RATE = 25;
    const LAST_FRAME = 80;
    const allVideos = group => [...group.querySelectorAll('video')];
    function groupState(group) {{
      if (!state.has(group)) state.set(group, {{raf:0, loading:null}});
      return state.get(group);
    }}
    function loadGroup(group) {{
      const value = groupState(group);
      if (value.loading) return value.loading;
      value.loading = Promise.all(allVideos(group).map(video => new Promise((resolve,reject) => {{
        if (!video.src) {{ video.src = video.dataset.src; video.load(); }}
        if (video.readyState >= 1) return resolve();
        video.addEventListener('loadedmetadata', resolve, {{once:true}});
        video.addEventListener('error', () => reject(new Error(`Cannot load ${{video.dataset.src}}`)), {{once:true}});
      }})));
      return value.loading;
    }}
    function pauseGroup(group) {{
      const value = groupState(group); cancelAnimationFrame(value.raf); value.raf = 0;
      allVideos(group).forEach(video => video.pause());
    }}
    function updateReadout(group, seconds) {{
      const frame = Math.max(0, Math.min(LAST_FRAME, Math.round(seconds * FRAME_RATE)));
      group.querySelector('[data-action=seek]').value = frame;
      group.querySelector('[data-time]').value = `F${{String(frame).padStart(2,'0')}} / 80 · ${{(frame / FRAME_RATE).toFixed(2)}}s`;
    }}
    function seekGroup(group, frame) {{
      const seconds = Math.max(0, Math.min(LAST_FRAME, frame)) / FRAME_RATE;
      allVideos(group).forEach(video => {{ video.currentTime = Math.min(seconds, Math.max(0, video.duration - .001)); }});
      updateReadout(group, seconds);
    }}
    function tick(group) {{
      const value = groupState(group), videos = allVideos(group), master = videos[0];
      if (master.paused || master.ended) {{ pauseGroup(group); updateReadout(group, master.currentTime); return; }}
      videos.slice(1).forEach(video => {{ if (Math.abs(video.currentTime - master.currentTime) > .035) video.currentTime = master.currentTime; }});
      updateReadout(group, master.currentTime);
      value.raf = requestAnimationFrame(() => tick(group));
    }}
    async function playGroup(group, restart=false) {{
      groups.filter(other => other !== group).forEach(pauseGroup);
      try {{
        await loadGroup(group);
        if (restart || allVideos(group)[0].ended) seekGroup(group, 0);
        const speed = Number(group.querySelector('[data-action=speed]').value);
        allVideos(group).forEach(video => video.playbackRate = speed);
        await Promise.all(allVideos(group).map(video => video.play()));
        cancelAnimationFrame(groupState(group).raf); tick(group);
      }} catch (error) {{ alert(error.message); }}
    }}
    groups.forEach(group => {{
      group.querySelector('[data-action=restart]').addEventListener('click', () => playGroup(group,true));
      group.querySelector('[data-action=play]').addEventListener('click', () => playGroup(group));
      group.querySelector('[data-action=pause]').addEventListener('click', () => pauseGroup(group));
      group.querySelector('[data-action=back]').addEventListener('click', async () => {{ await loadGroup(group); pauseGroup(group); seekGroup(group, Number(group.querySelector('[data-action=seek]').value)-1); }});
      group.querySelector('[data-action=forward]').addEventListener('click', async () => {{ await loadGroup(group); pauseGroup(group); seekGroup(group, Number(group.querySelector('[data-action=seek]').value)+1); }});
      group.querySelector('[data-action=seek]').addEventListener('input', async event => {{ await loadGroup(group); pauseGroup(group); seekGroup(group, Number(event.target.value)); }});
      group.querySelector('[data-action=speed]').addEventListener('change', event => allVideos(group).forEach(video => video.playbackRate=Number(event.target.value)));
      allVideos(group).forEach(video => {{
        video.addEventListener('click', () => allVideos(group)[0].paused ? playGroup(group) : pauseGroup(group));
        video.addEventListener('dblclick', () => video.requestFullscreen?.());
      }});
    }});
    document.querySelector('#pause-all').addEventListener('click', () => groups.forEach(pauseGroup));
    document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {{
      document.querySelectorAll('[data-filter]').forEach(item => item.classList.remove('active')); button.classList.add('active');
      groups.forEach(group => {{
        const visible = button.dataset.filter === 'all' || group.dataset.winner === button.dataset.filter ||
          (button.dataset.filter === 'conflict' && group.dataset.conflict === 'yes');
        group.hidden = !visible; if (!visible) pauseGroup(group);
      }});
    }}));
    if ('IntersectionObserver' in window) {{
      const observer = new IntersectionObserver(entries => entries.forEach(entry => {{
        if (entry.isIntersecting) {{ loadGroup(entry.target).catch(() => {{}}); observer.unobserve(entry.target); }}
      }}), {{rootMargin:'240px'}});
      groups.forEach(group => observer.observe(group));
    }}
  </script>
</body>
</html>
"""


def _transcode_one(
    *, ffmpeg: Path, input_path: Path, output_path: Path, threads: int,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    probe = _probe_video(input_path, ffmpeg)
    pts_scale = 3.2 / probe["duration"]
    video_filter = (
        f"setpts={pts_scale:.12f}*PTS,fps=25,"
        "tpad=stop_mode=clone:stop_duration=1,"
        "scale=960:-2:force_original_aspect_ratio=decrease,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(input_path), "-map", "0:v:0", "-an", "-vf", video_filter,
        "-frames:v", "81", "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-threads", str(threads), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {input_path}: {process.stderr[-2000:]}")
    output_probe = _probe_video(output_path, ffmpeg)
    if output_probe["frame_count"] != 81 or abs(output_probe["fps"] - 25.0) >= 1.0e-6:
        raise RuntimeError(f"review transcode geometry differs: {output_path}: {output_probe}")
    return {
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "input_probe": probe,
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "output_probe": output_probe,
    }


def _probe_video(path: Path, ffmpeg: Path) -> dict[str, Any]:
    try:
        return _opencv_probe_video(path)
    except ModuleNotFoundError:
        command = [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(path), "-map", "0:v:0", "-f", "null", "-",
            "-progress", "pipe:1", "-nostats",
        ]
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg probe failed for {path}: {process.stderr[-2000:]}")
        progress: dict[str, str] = {}
        for line in process.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                progress[key] = value
        frames = int(progress.get("frame", "0"))
        raw_duration = progress.get("out_time", "00:00:00")
        hours, minutes, seconds = raw_duration.split(":")
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if frames <= 0 or duration <= 0:
            raise RuntimeError(f"ffmpeg probe produced invalid geometry for {path}: {progress}")
        return {
            "duration": duration,
            "frame_count": frames,
            "fps": frames / duration,
            "width": None,
            "height": None,
            "backend": "ffmpeg-decode-progress",
        }


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve(strict=True)
    qwen_path = Path(args.qwen_summary).resolve(strict=True)
    sm_path = Path(args.semantic_moments_summary).resolve(strict=True)
    ffmpeg = Path(args.ffmpeg).resolve(strict=True)
    if not os.access(ffmpeg, os.X_OK):
        raise ValueError(f"ffmpeg is not executable: {ffmpeg}")
    output_dir = Path(args.output_dir).resolve(strict=False)
    assert_not_protected_write(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output directory already exists: {output_dir}")

    manifest = load_json(manifest_path)
    qwen = load_json(qwen_path)
    semantic_moments = load_json(sm_path)
    cases = build_cases(manifest, qwen, semantic_moments)
    output_dir.mkdir(parents=True)

    tasks = []
    for case in cases:
        for role in ROLE_ORDER:
            tasks.append((case, role, Path(case["source_paths"][role])))
    media_rows: dict[str, dict[str, Any]] = {case["pair_prefix"]: {} for case in cases}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _transcode_one,
                ffmpeg=ffmpeg,
                input_path=input_path,
                output_path=output_dir / "media" / case["pair_prefix"] / f"{role}.mp4",
                threads=args.ffmpeg_threads,
            ): (case["pair_prefix"], role)
            for case, role, input_path in tasks
        }
        for future in as_completed(futures):
            prefix, role = futures[future]
            media_rows[prefix][role] = future.result()
            print(json.dumps({"pair_prefix": prefix, "role": role, "status": "complete"}), flush=True)

    index_path = output_dir / "index.html"
    assert_not_protected_write(index_path)
    index_path.write_text(render_html(cases), encoding="utf-8")
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pair_count": len(cases),
        "roles": list(ROLE_ORDER),
        "review_media_contract": {
            "frame_count": 81,
            "fps": 25,
            "complete_event_retimed": True,
            "review_only": True,
            "evaluation_inputs_changed": False,
            "protected_mev_tree_modified": False,
        },
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
            "qwen_summary": {"path": str(qwen_path), "sha256": file_sha256(qwen_path)},
            "semantic_moments_summary": {"path": str(sm_path), "sha256": file_sha256(sm_path)},
        },
        "index_html_sha256": file_sha256(index_path),
        "media": media_rows,
    }
    write_json(output_dir / "bundle_manifest.json", bundle)
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "pairs": len(cases)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qwen-summary", required=True)
    parser.add_argument("--semantic-moments-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffmpeg-threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0 or args.ffmpeg_threads <= 0:
        raise ValueError("worker counts must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an offline HTML postmortem for the failed case01 exact5 run.

This publisher is downstream of the independent failure postflight.  It does
not accept a successful exact5 package and does not reinterpret the five
partial outputs as accepted experiment results.  The historical frozen case01
output is carried only as a byte-parity reference and is not a sixth task arm.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


if __package__:
    from . import case01_source_bone_exact5_failure_postflight_v1 as pf
    from .build_case01_source_bone_exact5_r64_html_v1 import (
        SiteBuildError,
        _copy_verified,
        _write_new,
        canonical_json_bytes,
        make_pair_sheet,
        object_sha256,
        resolve_tool,
        stable_file,
    )
else:
    import case01_source_bone_exact5_failure_postflight_v1 as pf
    from build_case01_source_bone_exact5_r64_html_v1 import (
        SiteBuildError,
        _copy_verified,
        _write_new,
        canonical_json_bytes,
        make_pair_sheet,
        object_sha256,
        resolve_tool,
        stable_file,
    )


SITE_SCHEMA = "case01-source-bone-exact5-failure-postmortem-site-v1"
FAILURE_SCHEMA = "case01-source-bone-exact5-runner-failure-v1"
CAMPAIGN = "case01-source-bone-exact5-r64-canary"
IID = "288545b9c031491a"
INSTRUCTION = "Make the dog pick up the bone and hold it in its mouth."
REFERENCE_SHA256 = (
    "e0d3c07d1d3e6ae4d45e59713d2af3f04786c305f8842c20d79172a9cae22403"
)
KEYFRAMES = (0, 20, 40, 60, 80)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

VARIANT_ORDER = (
    "exact_original",
    "codec_only_present",
    "bone_removed",
    "bone_translated_up150",
    "sham_control_up150",
)
TASK_IDS = tuple(f"case01-{variant}-full644" for variant in VARIANT_ORDER)

# These are concise renderings of the required independent Markdown audit.
# They are deliberately excluded from the site manifest's machine conclusions;
# the page also states that no structured review JSON accompanies the bundle.
VISUAL_AUDIT_FINDINGS = {
    "exact_original": (
        "人工全帧审计：source dog identity 被替换；原 bone 全程留在背景；"
        "另一个较大的橙色 prop 最终嘴持。"
    ),
    "codec_only_present": (
        "人工全帧审计：source dog identity 被替换；原 bone 全程留在背景；"
        "后段出现第二个橙色 prop 并最终嘴持。"
    ),
    "bone_removed": (
        "人工全帧审计：source dog identity 被替换；bone 按预期缺失；"
        "未见物体接触、抓取、抬起或最终嘴持。"
    ),
    "bone_translated_up150": (
        "人工全帧审计：source dog identity 被替换；上移后的实际 bone 全程留在"
        "背景；未见接近、接触、转移或可靠的第二个 bone。"
    ),
    "sham_control_up150": (
        "人工全帧审计：source dog identity 被替换；原 bone 全程留在背景；"
        "后段出现独立的橙色嘴持 prop。"
    ),
}


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SiteBuildError(f"{label} SHA-256 differs")
    return value


def _path(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise SiteBuildError(f"{label} path is absent")
    path = Path(value)
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise SiteBuildError(f"{label} path is unavailable") from error


def _member_path(
    verified: Mapping[str, Any], *names: str, label: str,
) -> Path:
    for name in names:
        if name in verified:
            return _path(verified[name], label=label)
    paths = verified.get("paths")
    if isinstance(paths, Mapping):
        aliases = {
            "plan_path": "plan",
            "failure_path": "failure",
            "failure_attestation_path": "failure",
            "manifest_path": "manifest",
            "postflight_path": "manifest",
            "reference_path": "reference",
            "reference_receipt_path": "reference_receipt",
        }
        for name in names:
            key = aliases.get(name)
            if key is not None and key in paths:
                return _path(paths[key], label=label)
    raise SiteBuildError(f"{label} path is absent")


def _member_sha(
    verified: Mapping[str, Any], *names: str, label: str,
) -> str:
    for name in names:
        if name in verified:
            return _sha(verified[name], label=label)
    raise SiteBuildError(f"{label} SHA-256 is absent")


def _probe(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SiteBuildError(f"{label} probe is absent")
    codec = value.get("codec", value.get("codec_name"))
    try:
        frame_count = value["frame_count"]
        fps_num = value["fps_num"]
        fps_den = value["fps_den"]
        width = value["width"]
        height = value["height"]
    except KeyError as error:
        raise SiteBuildError(f"{label} probe fields differ") from error
    if (
        codec != "h264"
        or type(frame_count) is not int
        or frame_count != 81
        or type(fps_num) is not int
        or fps_num != 25
        or type(fps_den) is not int
        or fps_den != 1
        or type(width) is not int
        or width <= 0
        or type(height) is not int
        or height <= 0
    ):
        raise SiteBuildError(f"{label} is not H.264 81f@25fps")
    return {
        "codec": "h264",
        "frame_count": frame_count,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "width": width,
        "height": height,
    }


def _load_visual_audit(
    *, bundle: Path, explicit_path: Path | None,
) -> dict[str, Any]:
    path = (
        bundle / "evidence/VISUAL_AUDIT.md"
        if explicit_path is None
        else explicit_path.expanduser()
    )
    path = _path(path, label="independent visual audit")
    raw, sha256, size = stable_file(path, label="independent visual audit")
    if size < 1_000 or size > 256_000 or b"\x00" in raw:
        raise SiteBuildError("independent visual audit size/content differs")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise SiteBuildError("independent visual audit is not UTF-8") from error
    required_fragments = (
        "# Case01 exact5 visual audit (independent, all 81 decoded frames)",
        "## Per-arm all-frame findings",
        "| exact_original |",
        "| codec_only_present |",
        "| bone_removed |",
        "| bone_translated_up150 |",
        "| sham_control_up150 |",
        "## Matched contrasts",
        "## Action-stage determination",
        "No arm satisfies the desired source-object chain",
        "## Bottom line and limitations",
        "failed diagnostic for object use and identity retention",
        "not a formal causal claim",
        "each condition has one sample",
        "exact-original parity check failed",
    )
    if any(fragment not in text for fragment in required_fragments):
        raise SiteBuildError("independent visual audit conclusion/limit closure differs")
    return {"path": path, "sha256": sha256, "size": size}


def _validate_verified_bundle(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SiteBuildError("failure postflight return value differs")
    verified = dict(value)
    plan = verified.get("plan")
    failure = verified.get("failure")
    manifest = verified.get("manifest")
    cases_value = verified.get("cases")
    if (
        not isinstance(plan, Mapping)
        or not isinstance(failure, Mapping)
        or not isinstance(manifest, Mapping)
        or not isinstance(cases_value, list)
        or len(cases_value) != 5
    ):
        raise SiteBuildError("failure postflight document closure differs")
    if (
        failure.get("schema_version") != FAILURE_SCHEMA
        or failure.get("status") != "FAILED_NO_RETRY"
        or failure.get("retry_allowed") is not False
        or failure.get("partial_outputs_are_not_results") is not True
        or failure.get("scientific_claim_authorized") is not False
    ):
        raise SiteBuildError("runner failure claim boundary differs")

    normalized_cases: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for index, raw_case in enumerate(cases_value):
        if not isinstance(raw_case, Mapping):
            raise SiteBuildError("postflight case row differs")
        required = {
            "id", "title", "title_zh", "group", "summary", "task_id",
            "task", "source_path", "output_path", "receipt_path",
            "source_sha256", "output_sha256", "source_probe", "output_probe",
        }
        if not required.issubset(raw_case):
            raise SiteBuildError("postflight case row is incomplete")
        case = dict(raw_case)
        variant = case["id"]
        expected_task_id = TASK_IDS[index]
        if (
            variant != VARIANT_ORDER[index]
            or case.get("task_id") != expected_task_id
            or case.get("group") not in {"controls", "interventions"}
            or not isinstance(case.get("title"), str)
            or not case["title"]
            or not isinstance(case.get("title_zh"), str)
            or not case["title_zh"]
            or not isinstance(case.get("summary"), str)
            or not case["summary"]
            or not isinstance(case.get("task"), Mapping)
            or case["task"].get("task_id") != expected_task_id
        ):
            raise SiteBuildError("postflight case identity/order differs")
        source_path = _path(case["source_path"], label=f"{variant} source")
        output_path = _path(case["output_path"], label=f"{variant} partial output")
        receipt_path = _path(case["receipt_path"], label=f"{variant} receipt")
        if any(path in seen_paths for path in (source_path, output_path, receipt_path)):
            raise SiteBuildError("postflight case paths are not distinct")
        seen_paths.update((source_path, output_path, receipt_path))
        case.update({
            "source_path": source_path,
            "output_path": output_path,
            "receipt_path": receipt_path,
            "source_sha256": _sha(
                case["source_sha256"], label=f"{variant} source"
            ),
            "output_sha256": _sha(
                case["output_sha256"], label=f"{variant} partial output"
            ),
            "source_probe": _probe(
                case["source_probe"], label=f"{variant} source"
            ),
            "output_probe": _probe(
                case["output_probe"], label=f"{variant} partial output"
            ),
        })
        normalized_cases.append(case)
    if normalized_cases[0]["output_sha256"] == REFERENCE_SHA256:
        raise SiteBuildError("failure site refuses an exact_original parity pass")

    reference_path = _member_path(
        verified, "reference_path", label="historical reference"
    )
    reference_sha256 = _member_sha(
        verified, "reference_sha256", label="historical reference"
    )
    if reference_sha256 != REFERENCE_SHA256:
        raise SiteBuildError("historical reference bytes differ")
    reference_probe = _probe(
        verified.get("reference_probe"), label="historical reference"
    )
    reference_receipt_path = _member_path(
        verified, "reference_receipt_path",
        label="historical reference receipt",
    )
    reference_receipt_sha256 = _member_sha(
        verified, "reference_receipt_sha256",
        label="historical reference receipt",
    )
    reference_receipt_size = verified.get("reference_receipt_size")
    if type(reference_receipt_size) is not int or reference_receipt_size <= 0:
        raise SiteBuildError("historical reference receipt size differs")

    return {
        "plan": dict(plan),
        "failure": dict(failure),
        "manifest": dict(manifest),
        "plan_path": _member_path(
            verified, "plan_path", label="exact5 plan"
        ),
        "plan_sha256": _member_sha(
            verified, "plan_sha256", label="exact5 plan"
        ),
        "failure_path": _member_path(
            verified,
            "failure_path",
            "failure_attestation_path",
            label="failure attestation",
        ),
        "failure_sha256": _member_sha(
            verified,
            "failure_sha256",
            "failure_attestation_sha256",
            label="failure attestation",
        ),
        "manifest_path": _member_path(
            verified,
            "manifest_path",
            "postflight_path",
            label="failure postflight manifest",
        ),
        "manifest_sha256": _member_sha(
            verified,
            "manifest_sha256",
            "postflight_sha256",
            label="failure postflight manifest",
        ),
        "cases": normalized_cases,
        "reference_path": reference_path,
        "reference_sha256": reference_sha256,
        "reference_probe": reference_probe,
        "reference_receipt_path": reference_receipt_path,
        "reference_receipt_sha256": reference_receipt_sha256,
        "reference_receipt_size": reference_receipt_size,
    }


def _media_meta(probe: Mapping[str, Any]) -> str:
    return (
        f"{probe['frame_count']}f · {probe['fps_num']}fps · "
        f"{probe['width']}×{probe['height']} · H.264"
    )


def render_html(
    cases: Sequence[Mapping[str, Any]], *,
    reference_probe: Mapping[str, Any],
    reference_sha256: str,
    reference_basename: str,
    reference_receipt_basename: str,
    plan_sha256: str,
    failure_sha256: str,
    postflight_sha256: str,
    visual_audit_sha256: str,
    build_time: str,
) -> str:
    cards: list[str] = []
    for index, case in enumerate(cases):
        variant = str(case["id"])
        source_basename = str(case["published_source_basename"])
        output_basename = str(case["published_output_basename"])
        receipt_basename = str(case["published_receipt_basename"])
        search = " ".join(
            str(item) for item in (
                variant, case["title"], case["title_zh"], case["group"],
                case["summary"], case["task_id"], VISUAL_AUDIT_FINDINGS[variant],
            )
        ).lower()
        parity_badge = (
            '<span class="badge danger">PARITY MISMATCH ARM</span>'
            if variant == "exact_original" else ""
        )
        cards.append(f'''<article class="case" id="variant-{_h(variant)}" data-group="{_h(case['group'])}" data-search="{_h(search)}">
  <header class="case-head">
    <div class="case-number">{index + 1:02d}</div>
    <div><div class="case-title"><h2>{_h(case['title_zh'])}</h2><span class="badge">{_h(variant)}</span>{parity_badge}</div><p>{_h(case['summary'])}</p><code>{_h(case['task_id'])}</code></div>
    <button class="sync-pair" type="button">两列同步从头播放</button>
  </header>
  <div class="review-note"><strong>独立人工 visual audit（Markdown 已绑定；review JSON 未附）</strong><span>{_h(VISUAL_AUDIT_FINDINGS[variant])}</span></div>
  <div class="video-grid">
    <article class="video-card">
      <div class="video-head"><div><h3>Source intervention</h3><p>{_h(case['title'])}</p></div><button class="play-one" type="button">播放 / 暂停</button></div>
      <video controls muted playsinline preload="metadata" aria-label="{_h(case['title_zh'])} source intervention" src="assets/media/{_h(source_basename)}"></video>
      <div class="media-meta">{_h(_media_meta(case['source_probe']))}</div>
      <code title="{_h(case['source_sha256'])}">sha256 {_h(case['source_sha256'])}</code>
      <div class="asset-links"><a href="assets/media/{_h(source_basename)}">source 文件</a></div>
    </article>
    <article class="video-card partial">
      <div class="video-head"><div><h3>Current partial output</h3><p>FAILED_NO_RETRY · postmortem artifact</p></div><button class="play-one" type="button">播放 / 暂停</button></div>
      <video controls muted playsinline preload="metadata" aria-label="{_h(case['title_zh'])} current partial output" src="assets/media/{_h(output_basename)}"></video>
      <div class="media-meta">{_h(_media_meta(case['output_probe']))}</div>
      <code title="{_h(case['output_sha256'])}">sha256 {_h(case['output_sha256'])}</code>
      <div class="asset-links"><a href="assets/media/{_h(output_basename)}">partial output</a> · <a href="evidence/receipts/{_h(receipt_basename)}">native receipt</a></div>
    </article>
  </div>
  <details class="sheets" open><summary>逐帧诊断对照（上：Source；下：current partial output）</summary><div class="sheet-grid">
    <button class="sheet" type="button" data-image="assets/sheets/{_h(variant)}-keyframes.jpg" data-title="{_h(case['title_zh'])} · frames 0/20/40/60/80"><span>关键帧 · 0 / 20 / 40 / 60 / 80</span><img loading="lazy" src="assets/sheets/{_h(variant)}-keyframes.jpg" alt="{_h(case['title_zh'])} source 与 partial output 五个关键帧诊断对照"></button>
    <button class="sheet" type="button" data-image="assets/sheets/{_h(variant)}-all81.jpg" data-title="{_h(case['title_zh'])} · all 81 frames"><span>全部 81 帧 · 9 × 9</span><img loading="lazy" src="assets/sheets/{_h(variant)}-all81.jpg" alt="{_h(case['title_zh'])} source 与 partial output 全部 81 帧诊断对照"></button>
  </div></details>
</article>''')

    exact_case = cases[0]
    exact_basename = str(exact_case["published_output_basename"])
    exact_sha256 = str(exact_case["output_sha256"])
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Case01 Exact5 · FAILED_NO_RETRY postmortem</title>
<style>
:root{{--bg:#090909;--panel:#15171b;--panel2:#1c2026;--line:#343941;--text:#f4f2ef;--muted:#aaaeb6;--red:#ff655d;--red2:#431b1b;--amber:#ffc75c;--blue:#8bc4ff;--max:1480px}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--text);background:radial-gradient(circle at 16% -10%,rgba(255,101,93,.17),transparent 38rem),radial-gradient(circle at 92% 8%,rgba(255,199,92,.08),transparent 32rem),var(--bg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}a{{color:#a9d3ff}}button,input,select{{font:inherit}}code{{font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}}.wrap{{width:min(calc(100% - 32px),var(--max));margin:auto}}
.topbar{{position:sticky;top:0;z-index:30;border-bottom:1px solid rgba(255,255,255,.09);background:rgba(9,9,9,.92);backdrop-filter:blur(16px)}}.topbar .wrap{{min-height:64px;display:flex;align-items:center;justify-content:space-between;gap:16px}}.brand{{font-weight:860}}.brand::before{{content:"";display:inline-block;width:10px;height:10px;margin-right:10px;border-radius:50%;background:var(--red);box-shadow:0 0 0 6px rgba(255,101,93,.14)}}.topmeta{{color:var(--muted);font-size:12px;text-align:right}}
.hero{{padding:58px 0 30px}}.eyebrow{{color:#ffaaa5;font-size:12px;font-weight:900;letter-spacing:.15em;text-transform:uppercase}}h1{{max-width:1050px;margin:13px 0 18px;font-size:clamp(39px,6vw,76px);line-height:1.02;letter-spacing:-.05em}}.lede{{max-width:1080px;margin:0;color:#d0d0d2;font-size:clamp(17px,2vw,21px)}}.hard-fail{{display:grid;grid-template-columns:auto 1fr;gap:15px;margin-top:27px;padding:22px;border:1px solid rgba(255,101,93,.7);border-radius:17px;background:linear-gradient(135deg,rgba(255,101,93,.19),rgba(255,101,93,.045))}}.hard-fail .icon{{width:39px;height:39px;display:grid;place-items:center;border-radius:50%;color:#210505;background:var(--red);font-weight:950}}.hard-fail strong{{display:block;color:#ffd0cd;font-size:18px}}.hard-fail p{{margin:5px 0 0;color:#e1c1bf}}.badges{{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}}.hero-badge{{padding:7px 11px;border:1px solid rgba(255,101,93,.55);border-radius:999px;color:#ffd0cd;background:rgba(255,101,93,.1);font-size:12px;font-weight:850;letter-spacing:.04em}}.hero-badge.amber{{color:#ffe1a4;border-color:rgba(255,199,92,.5);background:rgba(255,199,92,.09)}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}}.stat{{padding:17px;border:1px solid var(--line);border-radius:14px;background:rgba(21,23,27,.86)}}.stat b{{display:block;color:#ffaaa5;font-size:25px}}.stat span{{color:var(--muted);font-size:12px}}.instruction{{margin-top:16px;padding:16px 18px;border-left:4px solid var(--blue);border-radius:0 12px 12px 0;background:var(--panel2);font-size:17px}}.instruction small{{display:block;color:var(--blue);font-size:11px;font-weight:850;letter-spacing:.09em;text-transform:uppercase}}
.parity{{margin:12px auto 30px;padding:22px;border:1px solid rgba(255,199,92,.5);border-radius:18px;background:linear-gradient(145deg,rgba(50,39,18,.85),rgba(20,20,18,.96))}}.parity-head{{display:flex;flex-wrap:wrap;align-items:start;justify-content:space-between;gap:12px}}.parity h2{{margin:0;font-size:28px}}.parity p{{max-width:940px;margin:5px 0;color:#d3c8ae}}.not-arm{{padding:6px 10px;border:1px solid var(--amber);border-radius:999px;color:#ffe2a8;font-size:11px;font-weight:900}}.sha-compare{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:15px 0}}.sha-box{{padding:10px;border:1px solid #554a32;border-radius:9px;background:#15130e}}.sha-box strong{{display:block;color:#ffe2a8;font-size:11px}}.sha-box code{{display:block;margin-top:4px;color:#c8bea8}}.reference-note{{padding:10px 12px;border-left:3px solid var(--amber);color:#dacfae;background:rgba(255,199,92,.07)}}
.filters{{position:sticky;top:64px;z-index:20;padding:11px 0;border-block:1px solid rgba(255,255,255,.07);background:rgba(9,9,9,.9);backdrop-filter:blur(14px)}}.filter-row{{display:flex;flex-wrap:wrap;align-items:center;gap:8px}}.filter,.global-button,select{{padding:8px 12px;border:1px solid var(--line);border-radius:999px;color:var(--muted);background:var(--panel);cursor:pointer}}.filter.active,.filter[aria-pressed="true"]{{color:#240707;border-color:var(--red);background:var(--red);font-weight:800}}.global-button:hover,.filter:hover{{border-color:#65707d;color:var(--text)}}#search{{min-width:250px;flex:1;padding:9px 13px;border:1px solid var(--line);border-radius:10px;color:var(--text);background:var(--panel)}}#visible-count{{color:var(--muted);font-size:12px}}
.review-boundary{{margin:28px auto 0;padding:18px;border:1px dashed rgba(255,199,92,.65);border-radius:14px;background:rgba(255,199,92,.055)}}.review-boundary strong{{color:#ffe2a8}}.review-boundary p{{margin:5px 0 0;color:#cfc5ae}}
.case-list{{display:grid;gap:27px;padding:22px 0 32px}}.case{{scroll-margin-top:132px;overflow:hidden;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,rgba(28,32,38,.96),rgba(14,15,18,.97));box-shadow:0 24px 72px rgba(0,0,0,.25)}}.case[hidden]{{display:none}}.case-head{{display:grid;grid-template-columns:58px 1fr auto;gap:16px;align-items:center;padding:21px 22px 15px}}.case-number{{width:52px;height:52px;display:grid;place-items:center;border:1px solid var(--line);border-radius:14px;color:#ffaaa5;background:#0b0c0e;font-weight:880;font-size:18px}}.case-title{{display:flex;flex-wrap:wrap;align-items:center;gap:8px}}.case-title h2{{margin:0;font-size:26px}}.case-head p{{margin:4px 0;color:var(--muted)}}.badge{{padding:3px 8px;border:1px solid var(--line);border-radius:999px;color:var(--muted);font-size:11px}}.badge.danger{{color:#ffb0ab;border-color:rgba(255,101,93,.5);background:rgba(255,101,93,.09)}}.sync-pair,.play-one{{padding:7px 10px;border:1px solid #59616c;border-radius:8px;color:var(--text);background:#242931;cursor:pointer}}.sync-pair:hover,.play-one:hover{{background:#303741}}.review-note{{display:flex;flex-wrap:wrap;gap:9px 14px;margin:0 22px 18px;padding:10px 12px;border:1px dashed #665a3d;border-radius:9px;color:#d1c6ac;background:#19160f}}.review-note strong{{color:#ffe1a2}}.video-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;padding:0 12px 12px}}.video-card{{overflow:hidden;border:1px solid var(--line);border-radius:13px;background:var(--panel)}}.video-card.partial{{border-color:#6b3936}}.video-card.historical{{border-color:#675a37}}.video-head{{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:12px}}.video-head h3{{margin:0;font-size:17px}}.video-head p{{margin:2px 0 0;color:var(--muted);font-size:11px}}.play-one{{padding:5px 8px;font-size:11px}}video{{display:block;width:100%;aspect-ratio:704/736;max-height:72vh;background:#000;object-fit:contain}}.media-meta{{padding:9px 11px 4px;color:#bdc3cb;font-size:11px}}.video-card code{{display:block;padding:3px 11px;color:var(--muted)}}.asset-links{{padding:5px 11px 12px;color:var(--muted);font-size:11px}}
.sheets{{border-top:1px solid var(--line);background:#0a0b0d}}.sheets summary{{padding:11px 15px;color:var(--muted);cursor:pointer}}.sheet-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:0 12px 12px}}.sheet{{padding:0;overflow:hidden;border:1px solid var(--line);border-radius:10px;color:var(--text);background:#0d0f12;cursor:zoom-in;text-align:left}}.sheet span{{display:block;padding:7px 9px;font-size:11px}}.sheet img{{display:block;width:100%;height:auto}}.empty{{display:none;margin:30px 0;padding:30px;border:1px dashed var(--line);border-radius:14px;color:var(--muted);text-align:center}}footer{{padding:18px 0 42px;color:var(--muted);font-size:12px}}footer .evidence{{display:flex;flex-wrap:wrap;gap:8px 16px;margin-bottom:8px}}
dialog{{width:min(96vw,1800px);max-width:none;padding:0;border:1px solid #555d68;border-radius:14px;color:var(--text);background:#08090b;box-shadow:0 35px 120px rgba(0,0,0,.82)}}dialog::backdrop{{background:rgba(0,0,0,.87);backdrop-filter:blur(5px)}}dialog img{{display:block;width:100%;height:auto}}.dialog-bar{{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:10px 12px;border-bottom:1px solid var(--line)}}.dialog-bar span{{color:var(--muted)}}.dialog-bar button{{padding:6px 9px;border:1px solid var(--line);border-radius:7px;color:var(--text);background:var(--panel);cursor:pointer}}
@media(max-width:900px){{.stats{{grid-template-columns:repeat(2,1fr)}}.video-grid,.sheet-grid,.sha-compare{{grid-template-columns:1fr}}.case-head{{grid-template-columns:50px 1fr}}.sync-pair{{grid-column:2;justify-self:start}}}}@media(max-width:560px){{.wrap{{width:min(calc(100% - 20px),var(--max))}}.topmeta{{display:none}}.hero{{padding-top:42px}}.hard-fail{{grid-template-columns:1fr}}#search{{min-width:100%}}.case-head{{padding-inline:14px}}.review-note{{margin-inline:14px}}}}
</style>
</head>
<body>
<nav class="topbar"><div class="wrap"><div class="brand">Case01 Exact5 · FAILED_NO_RETRY</div><div class="topmeta">postmortem only · partial outputs are not results</div></div></nav>
<header class="hero wrap">
  <div class="eyebrow">Deterministic byte parity · HARD FAIL</div>
  <h1>五路失败态诊断产物<br>与冻结历史 reference</h1>
  <p class="lede">本页保留同一 case01 的五种 source intervention 与本次运行留下的五个 partial output，供失败复盘和人工检查。它不是完成态实验页面。</p>
  <div class="hard-fail"><div class="icon">!</div><div><strong>FAILED_NO_RETRY · exact_original 未通过冻结历史输出的 deterministic byte-parity gate。</strong><p>partial outputs are not results。页面仅用于 postmortem；不得据此发布 formal evaluation、scientific claim、causal intervention effect 或 generalization claim。</p></div></div>
  <div class="badges"><span class="hero-badge">PARITY HARD FAIL</span><span class="hero-badge">FAILED_NO_RETRY</span><span class="hero-badge">PARTIAL OUTPUTS ARE NOT RESULTS</span><span class="hero-badge amber">POSTMORTEM ONLY</span><span class="hero-badge amber">NON-FORMAL · NON-SCIENTIFIC</span></div>
  <div class="stats"><div class="stat"><b>5</b><span>source interventions</span></div><div class="stat"><b>5</b><span>current partial outputs</span></div><div class="stat"><b>1</b><span>historical reference · not a task arm</span></div><div class="stat"><b>81f · 25fps</b><span>postflight-verified media contract</span></div></div>
  <div class="instruction"><small>Instruction · IID {_h(IID)}</small>{_h(INSTRUCTION)}</div>
  <div class="hard-fail"><div class="icon">i</div><div><strong>Full644 membership / exposure audit 未随本 postmortem bundle 附带。</strong><p>本页不证明 IID-disjoint、content-disjoint、预训练无 exposure，也不把一个 intervention case 解释成多个独立数据集样本。</p></div></div>
</header>
<section class="parity wrap" id="historical-reference">
  <div class="parity-head"><div><h2>Current exact_original vs frozen historical reference</h2><p>上次冻结输出只用于定位 byte-parity mismatch。它不是本次 exact5 的第六个 task arm，也不替代失败态的 current exact_original。</p></div><div><span class="not-arm">NOT A CURRENT TASK ARM</span> <button class="sync-pair" id="sync-reference" type="button">两列同步从头播放</button></div></div>
  <div class="sha-compare"><div class="sha-box"><strong>Current exact_original · observed</strong><code>{_h(exact_sha256)}</code></div><div class="sha-box"><strong>Historical frozen reference · expected</strong><code>{_h(reference_sha256)}</code></div></div>
  <div class="video-grid">
    <article class="video-card partial"><div class="video-head"><div><h3>Current exact_original partial output</h3><p>observed bytes · parity mismatch</p></div><button class="play-one" type="button">播放 / 暂停</button></div><video controls muted playsinline preload="metadata" aria-label="current exact original partial output" src="assets/media/{_h(exact_basename)}"></video><div class="media-meta">{_h(_media_meta(exact_case['output_probe']))}</div><code>sha256 {_h(exact_sha256)}</code></article>
    <article class="video-card historical"><div class="video-head"><div><h3>Frozen historical reference</h3><p>reference only · not a current task arm</p></div><button class="play-one" type="button">播放 / 暂停</button></div><video controls muted playsinline preload="metadata" aria-label="frozen historical case01 reference" src="reference/{_h(reference_basename)}"></video><div class="media-meta">{_h(_media_meta(reference_probe))}</div><code>sha256 {_h(reference_sha256)}</code><div class="asset-links"><a href="reference/{_h(reference_basename)}">historical reference 文件</a> · <a href="reference/{_h(reference_receipt_basename)}">historical receipt</a></div></article>
  </div>
  <div class="reference-note">下面的 sheet 上半部是 current exact_original，下半部是 historical reference；差异本身不自动解释其视觉原因。</div>
  <details class="sheets" open><summary>Current / historical 帧对照</summary><div class="sheet-grid"><button class="sheet" type="button" data-image="assets/sheets/exact-original-current-vs-historical-keyframes.jpg" data-title="current exact_original vs historical · frames 0/20/40/60/80"><span>关键帧 · current / historical</span><img loading="lazy" src="assets/sheets/exact-original-current-vs-historical-keyframes.jpg" alt="current exact original 与 historical reference 关键帧对照"></button><button class="sheet" type="button" data-image="assets/sheets/exact-original-current-vs-historical-all81.jpg" data-title="current exact_original vs historical · all 81 frames"><span>全部 81 帧 · current / historical</span><img loading="lazy" src="assets/sheets/exact-original-current-vs-historical-all81.jpg" alt="current exact original 与 historical reference 全部帧对照"></button></div></details>
</section>
<section class="filters"><div class="wrap filter-row" aria-label="诊断产物筛选与播放控制"><button class="filter active" type="button" data-group="all" aria-pressed="true">全部 5 组</button><button class="filter" type="button" data-group="controls" aria-pressed="false">controls</button><button class="filter" type="button" data-group="interventions" aria-pressed="false">interventions</button><button class="global-button" id="play-visible" type="button">可见视频同步从头播放</button><button class="global-button" id="pause-all" type="button">全部暂停</button><label>速度 <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><input id="search" type="search" placeholder="筛选 variant / task / 待复核观察" aria-label="筛选诊断条件"><span id="visible-count" aria-live="polite">5 / 5</span></div></section>
<aside class="review-boundary wrap"><strong>独立人工 visual audit 已绑定 · 非自动 postflight 结论</strong><p>审计覆盖全部 81 个 decoded frames。结论：五臂均未完成对实际 source bone 的 approach → contact → grip → lift → hold；source dog identity 均被替换；exact / codec / sham 的终态 prop 是第二个生成物体，removed / translated 未形成终态 hold。<a href="evidence/visual-audit.md">阅读完整 visual audit</a>。</p><p><strong>Limits：</strong>固定执行顺序、每个条件仅一个样本、exact-original parity 已失败，因此不能干净区分 target representation、codec/composition sensitivity 或 run-state variation；不构成 formal causal claim。结构化 review JSON 仍未附带。</p></aside>
<main class="wrap"><div class="case-list">{''.join(cards)}</div><div class="empty" id="empty">没有匹配的诊断条件。</div></main>
<footer class="wrap"><div class="evidence"><a href="evidence/plan.json">exact5 plan</a><a href="evidence/failure-attestation.json">FAILED_NO_RETRY attestation</a><a href="evidence/postflight-manifest.json">independent postflight manifest</a><a href="evidence/visual-audit.md">independent visual audit</a><a href="site-manifest.json">site manifest</a></div><div>All-relative offline postmortem · plan {_h(plan_sha256)} · failure {_h(failure_sha256)} · postflight {_h(postflight_sha256)} · visual audit {_h(visual_audit_sha256)} · built {_h(build_time)}</div></footer>
<dialog id="lightbox"><div class="dialog-bar"><span id="dialog-title"></span><button type="button" id="dialog-close">关闭</button></div><img id="dialog-image" alt="放大的失败态帧诊断对照图"></dialog>
<script>
const cards=[...document.querySelectorAll('.case')],filters=[...document.querySelectorAll('.filter')],search=document.querySelector('#search'),speed=document.querySelector('#speed');let group='all';
function visibleVideos(){{return [...document.querySelectorAll('video')].filter(video=>{{const card=video.closest('.case');return !card||!card.hidden;}});}}
function applyFilter(){{const query=search.value.trim().toLowerCase();let visible=0;for(const card of cards){{const show=(group==='all'||card.dataset.group===group)&&(!query||card.dataset.search.includes(query));card.hidden=!show;if(show)visible++;else card.querySelectorAll('video').forEach(video=>video.pause());}}document.querySelector('#visible-count').textContent=`${{visible}} / 5`;document.querySelector('#empty').style.display=visible?'none':'block';}}
filters.forEach(button=>button.addEventListener('click',()=>{{group=button.dataset.group;filters.forEach(item=>{{const active=item===button;item.classList.toggle('active',active);item.setAttribute('aria-pressed',String(active));}});applyFilter();}}));search.addEventListener('input',applyFilter);
async function playTogether(videos){{videos.forEach(video=>{{video.pause();video.currentTime=0;video.playbackRate=Number(speed.value);}});await Promise.allSettled(videos.map(video=>video.play()));}}
document.querySelectorAll('.case .sync-pair').forEach(button=>button.addEventListener('click',()=>playTogether([...button.closest('.case').querySelectorAll('video')])));
document.querySelector('#sync-reference').addEventListener('click',()=>playTogether([...document.querySelector('#historical-reference').querySelectorAll('video')]));
document.querySelector('#play-visible').addEventListener('click',()=>playTogether(visibleVideos()));document.querySelector('#pause-all').addEventListener('click',()=>document.querySelectorAll('video').forEach(video=>video.pause()));
speed.addEventListener('change',()=>document.querySelectorAll('video').forEach(video=>{{video.playbackRate=Number(speed.value);}}));document.querySelectorAll('.play-one').forEach(button=>button.addEventListener('click',async()=>{{const video=button.closest('.video-card').querySelector('video');video.playbackRate=Number(speed.value);if(video.paused)await video.play().catch(()=>{{}});else video.pause();}}));
const dialog=document.querySelector('#lightbox');document.querySelectorAll('.sheet').forEach(button=>button.addEventListener('click',()=>{{document.querySelector('#dialog-image').src=button.dataset.image;document.querySelector('#dialog-title').textContent=button.dataset.title;dialog.showModal();}}));document.querySelector('#dialog-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{{if(event.target===dialog)dialog.close();}});document.addEventListener('keydown',event=>{{if(event.key==='Escape'&&dialog.open)dialog.close();}});
</script>
</body>
</html>
'''


def build_site(
    *, bundle: Path, output: Path, ffmpeg: Path, ffprobe: Path,
    visual_audit: Path | None = None,
) -> dict[str, Any]:
    verified = _validate_verified_bundle(
        pf.load_verified_bundle(bundle, ffprobe, ffmpeg)
    )
    cases = verified["cases"]
    audit = _load_visual_audit(bundle=bundle, explicit_path=visual_audit)
    reference_basename = verified["reference_path"].name
    if reference_basename in {"", ".", ".."}:
        raise SiteBuildError("historical reference basename differs")
    reference_receipt_path = verified["reference_receipt_path"]
    if reference_receipt_path.parent != verified["reference_path"].parent:
        raise SiteBuildError("historical reference receipt directory differs")
    if reference_receipt_path.name != reference_basename + ".receipt.json":
        raise SiteBuildError("historical reference receipt basename differs")
    reference_receipt_basename = reference_receipt_path.name
    _, observed_reference_receipt_sha256, observed_reference_receipt_size = stable_file(
        reference_receipt_path, label="historical reference receipt"
    )
    if (
        observed_reference_receipt_sha256
        != verified["reference_receipt_sha256"]
        or observed_reference_receipt_size
        != verified["reference_receipt_size"]
    ):
        raise SiteBuildError("historical reference receipt changed after postflight")
    verified.update({
        "reference_basename": reference_basename,
        "reference_receipt_basename": reference_receipt_basename,
    })

    # Re-read receipt bytes only to bind the portable copies.  Their semantic
    # validation belongs to the independent postflight producer.
    for case in cases:
        variant = str(case["id"])
        _, receipt_sha256, receipt_size = stable_file(
            case["receipt_path"], label=f"{variant} verified receipt"
        )
        if (
            case.get("receipt_sha256") != receipt_sha256
            or case.get("receipt_size") != receipt_size
        ):
            raise SiteBuildError(f"{variant} receipt changed after postflight")
        case.update({
            "receipt_sha256": receipt_sha256,
            "receipt_size": receipt_size,
            "published_source_basename": f"{variant}-source.mp4",
            "published_output_basename": f"{variant}-partial-output.mp4",
            "published_receipt_basename": f"{case['task_id']}.mp4.receipt.json",
        })

    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise SiteBuildError(f"output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-stage-", dir=output.parent)
    )
    try:
        assets = stage / "assets"
        media_out = assets / "media"
        sheets_out = assets / "sheets"
        evidence_out = stage / "evidence"
        receipts_out = evidence_out / "receipts"
        reference_out = stage / "reference"
        for directory in (
            assets, media_out, sheets_out, evidence_out, receipts_out,
            reference_out,
        ):
            directory.mkdir(mode=0o755)

        published: list[dict[str, Any]] = []

        def copy(
            source: Path, destination: Path, expected_sha256: str, *, label: str,
            relative: str,
        ) -> None:
            copied = _copy_verified(
                source, destination,
                expected_sha256=expected_sha256, label=label,
            )
            published.append({"path": relative, **copied})

        for case in cases:
            copy(
                case["source_path"],
                media_out / case["published_source_basename"],
                case["source_sha256"],
                label=f"published {case['id']} source",
                relative=f"assets/media/{case['published_source_basename']}",
            )
            copy(
                case["output_path"],
                media_out / case["published_output_basename"],
                case["output_sha256"],
                label=f"published {case['id']} partial output",
                relative=f"assets/media/{case['published_output_basename']}",
            )
            copy(
                case["receipt_path"],
                receipts_out / case["published_receipt_basename"],
                case["receipt_sha256"],
                label=f"published {case['id']} receipt",
                relative=f"evidence/receipts/{case['published_receipt_basename']}",
            )

        copy(
            verified["reference_path"],
            reference_out / verified["reference_basename"],
            verified["reference_sha256"], label="published historical reference",
            relative=f"reference/{verified['reference_basename']}",
        )
        copy(
            verified["reference_receipt_path"],
            reference_out / verified["reference_receipt_basename"],
            verified["reference_receipt_sha256"],
            label="published historical reference receipt",
            relative=f"reference/{verified['reference_receipt_basename']}",
        )
        for basename, source, sha256, label in (
            ("plan.json", verified["plan_path"], verified["plan_sha256"], "plan"),
            (
                "failure-attestation.json", verified["failure_path"],
                verified["failure_sha256"], "failure attestation",
            ),
            (
                "postflight-manifest.json", verified["manifest_path"],
                verified["manifest_sha256"], "postflight manifest",
            ),
        ):
            copy(
                source, evidence_out / basename, sha256,
                label=f"published {label}", relative=f"evidence/{basename}",
            )
        copy(
            audit["path"], evidence_out / "visual-audit.md", audit["sha256"],
            label="published independent visual audit",
            relative="evidence/visual-audit.md",
        )

        for case in cases:
            variant = str(case["id"])
            source_media = media_out / case["published_source_basename"]
            output_media = media_out / case["published_output_basename"]
            for suffix, all_frames in (("keyframes", False), ("all81", True)):
                sheet = sheets_out / f"{variant}-{suffix}.jpg"
                make_pair_sheet(
                    source_media, output_media, sheet, ffmpeg,
                    all_frames=all_frames,
                )
                os.chmod(sheet, 0o444)
                _, sheet_sha256, sheet_size = stable_file(
                    sheet, label=f"published {variant} {suffix} sheet"
                )
                published.append({
                    "path": f"assets/sheets/{sheet.name}",
                    "sha256": sheet_sha256,
                    "size": sheet_size,
                })

        exact_media = media_out / cases[0]["published_output_basename"]
        historical_media = reference_out / verified["reference_basename"]
        for suffix, all_frames in (("keyframes", False), ("all81", True)):
            sheet = sheets_out / f"exact-original-current-vs-historical-{suffix}.jpg"
            make_pair_sheet(
                exact_media, historical_media, sheet, ffmpeg,
                all_frames=all_frames,
            )
            os.chmod(sheet, 0o444)
            _, sheet_sha256, sheet_size = stable_file(
                sheet, label=f"published current/historical {suffix} sheet"
            )
            published.append({
                "path": f"assets/sheets/{sheet.name}",
                "sha256": sheet_sha256,
                "size": sheet_size,
            })

        build_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        index_raw = render_html(
            cases,
            reference_probe=verified["reference_probe"],
            reference_sha256=verified["reference_sha256"],
            reference_basename=verified["reference_basename"],
            reference_receipt_basename=verified["reference_receipt_basename"],
            plan_sha256=verified["plan_sha256"],
            failure_sha256=verified["failure_sha256"],
            postflight_sha256=verified["manifest_sha256"],
            visual_audit_sha256=audit["sha256"],
            build_time=build_time,
        ).encode("utf-8")
        _write_new(stage / "index.html", index_raw)
        os.chmod(stage / "index.html", 0o444)
        published.append({
            "path": "index.html",
            "sha256": hashlib.sha256(index_raw).hexdigest(),
            "size": len(index_raw),
        })

        site_manifest: dict[str, Any] = {
            "schema_version": SITE_SCHEMA,
            "status": "FAILED_NO_RETRY_POSTMORTEM_SITE_MATERIALIZED",
            "built_at_utc": build_time,
            "campaign_mode": CAMPAIGN,
            "iid": IID,
            "instruction": INSTRUCTION,
            "runner_status": "FAILED_NO_RETRY",
            "deterministic_reference_parity": {
                "policy": "HARD_FAIL",
                "status": "FAIL",
                "variant": "exact_original",
                "observed_output_sha256": cases[0]["output_sha256"],
                "expected_historical_reference_sha256": REFERENCE_SHA256,
                "historical_reference_is_current_task_arm": False,
            },
            "artifact_counts": {
                "source_interventions": 5,
                "current_partial_outputs": 5,
                "historical_references": 1,
                "historical_reference_receipts": 1,
                "videos": 11,
                "source_partial_pair_sheets": 10,
                "current_historical_pair_sheets": 2,
                "pair_sheets_total": 12,
            },
            "keyframes": list(KEYFRAMES),
            "all81_sheet_included_for_each_source_partial_pair": True,
            "all81_current_historical_sheet_included": True,
            "dataset_scope": {
                "independent_dataset_example_count": 1,
                "source_intervention_count": 5,
                "full644_membership_audit_included": False,
                "iid_disjoint_proven_by_this_bundle": False,
                "content_disjoint_proven": False,
                "pretraining_exposure_excluded": False,
            },
            "claim_limits": {
                "postmortem_only": True,
                "partial_outputs_are_not_results": True,
                "formal_evaluation_authorized": False,
                "scientific_claim_authorized": False,
                "causal_intervention_effect_claim_authorized": False,
                "generalization_claim_authorized": False,
                "historical_reference_is_not_current_task_arm": True,
                "manual_observations_bound_to_review_json": False,
                "independent_visual_audit_markdown_included": True,
                "automated_visual_conclusion_present": False,
            },
            "visual_audit": {
                "path": "evidence/visual-audit.md",
                "sha256": audit["sha256"],
                "size": audit["size"],
                "coverage": "all_81_decoded_frames_for_all_five_arms",
                "human_observation_not_postflight_automation": True,
                "structured_review_json_included": False,
            },
            "authorities": {
                "plan_sha256": verified["plan_sha256"],
                "failure_attestation_sha256": verified["failure_sha256"],
                "failure_postflight_manifest_sha256": verified["manifest_sha256"],
                "historical_reference_sha256": verified["reference_sha256"],
                "historical_reference_receipt_sha256": verified[
                    "reference_receipt_sha256"
                ],
                "independent_visual_audit_sha256": audit["sha256"],
            },
            "files_excluding_this_manifest": sorted(
                published, key=lambda row: row["path"]
            ),
        }
        site_manifest["manifest_digest"] = object_sha256(site_manifest)
        manifest_raw = canonical_json_bytes(site_manifest) + b"\n"
        _write_new(stage / "site-manifest.json", manifest_raw)
        os.chmod(stage / "site-manifest.json", 0o444)

        os.replace(stage, output)
        stage = None
        return {
            "output": str(output),
            "index": str(output / "index.html"),
            "manifest": str(output / "site-manifest.json"),
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "source_intervention_count": 5,
            "partial_output_count": 5,
            "historical_reference_count": 1,
            "visual_audit_sha256": audit["sha256"],
            "video_count": 11,
            "pair_sheet_count": 12,
        }
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an all-relative FAILED_NO_RETRY exact5 postmortem HTML site "
            "from a bundle admitted by the independent failure postflight."
        )
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--visual-audit",
        help=(
            "independent all-frame visual audit Markdown; defaults to "
            "BUNDLE/evidence/VISUAL_AUDIT.md and is always required"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_site(
            bundle=Path(args.bundle).expanduser().absolute(),
            output=Path(args.output).expanduser(),
            ffmpeg=resolve_tool(args.ffmpeg, label="ffmpeg"),
            ffprobe=resolve_tool(args.ffprobe, label="ffprobe"),
            visual_audit=(
                None
                if args.visual_audit is None
                else Path(args.visual_audit).expanduser()
            ),
        )
    except (
        OSError, SiteBuildError, pf.PostflightError, subprocess.SubprocessError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an offline HTML review page for frozen factorial calibration."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


BRANCHES = (
    "normalized_noop", "normalized_forward", "reverse_from_forward",
    "incomplete_phasewarp", "camera_right_push", "camera_center_push",
    "camera_vertical_push", "camera_center_pull", "appearance_hue_ramp",
)
MARGINS = (
    "forward_gt_noop", "forward_gt_reverse", "forward_gt_incomplete",
    "forward_gt_abs_nuisance",
)
BRANCH_LABELS = {
    "normalized_noop": "Noop：保持原姿态，不执行目标动作",
    "normalized_forward": "Forward：self-generated 目标动作锚点",
    "reverse_from_forward": "Reverse：将 forward 锚点倒放",
    "incomplete_phasewarp": "Incomplete：forward 前 41 帧拉伸到 81 帧",
    "camera_right_push": "Camera：右移 + push-in",
    "camera_center_push": "Camera：中心 push-in",
    "camera_vertical_push": "Camera：纵向移动 + push-in",
    "camera_center_pull": "Camera：中心 pull-out",
    "appearance_hue_ramp": "Appearance：noop 的渐变色相扰动",
}
MARGIN_LABELS = {
    "forward_gt_noop": "Forward − Noop",
    "forward_gt_reverse": "Forward − Reverse",
    "forward_gt_incomplete": "Forward − Incomplete",
    "forward_gt_abs_nuisance": "Forward − max |Camera / Appearance|",
}
ACTION_LABELS = {
    "dog-stand-to-sit": "让原始狗从四足站立自然坐下，并保持坐姿",
    "human-one-knee-to-stand": "让原始人物从单膝跪姿起身，并保持完全直立",
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def relative(path: Path, root: Path) -> str:
    return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()


def metric_card(name: str, count: int, worst: float) -> str:
    state = "pass" if count == 6 else "fail"
    return (
        f'<div class="metric {state}"><div class="metric-name">{esc(MARGIN_LABELS[name])}</div>'
        f'<div class="metric-value">{count}/6</div>'
        f'<div class="metric-note">最差 margin {worst:+.6f}</div></div>'
    )


def source_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = manifest.get("entries")
    if type(entries) is not list:
        raise RuntimeError("factorial manifest entries differ")
    result: dict[str, dict[str, Any]] = {}
    for item in entries:
        if item.get("analysis_split") != "calibration":
            continue
        source_id = item["source_id"]
        row = result.setdefault(
            source_id,
            {
                "source_id": source_id,
                "action_family": item["action_family"],
                "source_video": item["source_video"],
                "source_video_sha256": item["source_video_sha256"],
                "instructions": {},
            },
        )
        row["instructions"].setdefault(item["branch"], item["instruction"])
    return result


def cell_section(
    row: Mapping[str, Any], media_root: Path, source_media_root: Path,
    bundle_root: Path, sources: Mapping[str, Mapping[str, Any]],
) -> str:
    cell = str(row["cell"]).split(":", 1)[1]
    source_id = cell[:16]
    seed = cell.rsplit("-s", 1)[-1]
    source = sources.get(source_id)
    if source is None:
        raise RuntimeError(f"missing source metadata: {source_id}")
    cell_root = media_root / cell
    if not cell_root.is_dir():
        raise RuntimeError(f"missing cell media: {cell_root}")
    failed = [name for name in MARGINS if not row["passes"][name]]
    margin_rows = "".join(
        '<tr><td>{}</td><td class="{}">{:+.6f}</td><td>{}</td></tr>'.format(
            esc(MARGIN_LABELS[name]), "ok" if row["passes"][name] else "bad",
            float(row["margins"][name]), "PASS" if row["passes"][name] else "FAIL",
        )
        for name in MARGINS
    )
    score_rows = "".join(
        "<tr><td>{}</td><td>{:+.6f}</td></tr>".format(
            esc(BRANCH_LABELS[branch]), float(row["mixed_scores"][branch])
        )
        for branch in BRANCHES
    )
    videos = []
    for branch in BRANCHES:
        src = relative(cell_root / f"{branch}.mp4", bundle_root)
        videos.append(
            f'<figure><figcaption><strong>{esc(BRANCH_LABELS[branch])}</strong>'
            f'<span>S = {float(row["mixed_scores"][branch]):+.6f}</span></figcaption>'
            f'<video controls muted loop playsinline preload="metadata" src="{esc(src)}"></video></figure>'
        )
    source_src = relative(source_media_root / f"{source_id}.mp4", bundle_root)
    instructions = source["instructions"]
    badge = "PASS" if not failed else "FAIL: " + ", ".join(failed)
    state = "pass" if not failed else "fail"
    return f"""
    <section class="cell">
      <div class="cell-head">
        <div><div class="eyebrow">{esc(row['family'])} · source {esc(source_id)} · seed {esc(seed)}</div><h2>{esc(ACTION_LABELS.get(source['action_family'], source['action_family']))}</h2><div class="subtle">family: {esc(source['action_family'])}</div></div>
        <span class="badge {state}">{esc(badge)}</span>
      </div>
      <div class="cell-body">
        <div class="source-layout">
          <figure class="source-video"><figcaption><strong>原始 Source video</strong><span>{esc(source_id)}</span></figcaption><video controls muted loop playsinline preload="metadata" src="{esc(source_src)}"></video></figure>
          <div class="instructions">
            <div class="instruction primary"><div class="eyebrow">实际 Forward edit instruction</div><p>{esc(instructions['forward'])}</p></div>
            <div class="instruction"><div class="eyebrow">Noop instruction</div><p>{esc(instructions['noop'])}</p></div>
            <div class="warning"><strong>注意：</strong>下方 reverse / incomplete / camera / appearance 是从 forward 或 noop 视频确定性变换得到的控制分支，不是再次调用模型执行对应文字 prompt。当前 incomplete 使用前 41 帧，因此可能已经包含完整动作，这正是本轮 NOGO 的一个原因。</div>
          </div>
        </div>
        <div class="numbers">
          <div><h3>Ordinal margins</h3><p class="hint">margin = S(forward) − S(negative)。必须 &gt; 0；负数表示错误分支得分反而更高。</p><table><thead><tr><th>比较</th><th>margin</th><th>gate</th></tr></thead><tbody>{margin_rows}</tbody></table></div>
          <div><h3>各分支原始诊断分数 S</h3><p class="hint">S 不是概率、百分比或人工质量分；只用于同一个 cell 内的相对排序。</p><table><thead><tr><th>分支</th><th>S(branch)</th></tr></thead><tbody>{score_rows}</tbody></table></div>
        </div>
        <h3>视频逐分支对照</h3>
        <div class="video-grid">{''.join(videos)}</div>
      </div>
    </section>"""


def build(
    report: Mapping[str, Any], *, bundle_root: Path, media_root: Path,
    source_media_root: Path, manifest: Mapping[str, Any],
) -> str:
    calibration = report["calibration"]
    sources = source_index(manifest)
    metrics = "".join(
        metric_card(name, calibration["pass_counts"][name], calibration["minimum_margins"][name])
        for name in MARGINS
    )
    cells = "".join(
        cell_section(row, media_root, source_media_root, bundle_root, sources)
        for row in calibration["cells"]
    )
    frozen = report["frozen_selection"]
    verdict = "GO" if calibration["confirmation_evaluation_authorized"] else "NOGO"
    verdict_class = "pass" if verdict == "GO" else "fail"
    digest = report["receipt_digest"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frozen two-head calibration · {verdict}</title>
<style>
:root{{--bg:#081018;--panel:#101c28;--panel2:#142333;--ink:#edf5fa;--muted:#9bb0bf;--line:#294153;--good:#48d597;--bad:#ff6f7d;--blue:#77bdfb}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 15% 0,#153149 0,transparent 34rem),var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:auto;padding:40px 28px 100px}} h1{{font-size:clamp(34px,5vw,68px);line-height:1.02;margin:8px 0 18px;letter-spacing:-.045em}} h2{{margin:2px 0;font-size:22px}} .eyebrow{{color:var(--blue);font-size:12px;text-transform:uppercase;letter-spacing:.14em;font-weight:750}} .subtle{{color:var(--muted);font:12px ui-monospace,SFMono-Regular,Menlo,monospace}}
.lead{{max-width:900px;color:var(--muted);font-size:18px}} .verdict{{display:inline-flex;padding:7px 13px;border-radius:999px;font-weight:800;letter-spacing:.08em}} .pass{{color:var(--good)}} .fail{{color:var(--bad)}} .verdict.pass,.badge.pass{{background:#123d30}} .verdict.fail,.badge.fail{{background:#491d27}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:28px 0}} .metric{{padding:18px;border:1px solid var(--line);background:linear-gradient(145deg,var(--panel2),var(--panel));border-radius:16px}} .metric-name{{color:var(--muted);font-size:12px}} .metric-value{{font-size:32px;font-weight:850}} .metric-note{{font-variant-numeric:tabular-nums}}
.protocol{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:24px 0}} .protocol>div,.definition{{padding:18px;background:#0d1924;border:1px solid var(--line);border-radius:14px}} .definition{{margin:0 0 44px;border-color:#49718c}} .definition h2{{margin-bottom:10px}} .formula{{font-size:18px;color:#d7efff}} code{{color:#c6e6ff}} .cell{{margin:22px 0 52px;border:1px solid var(--line);border-radius:20px;overflow:hidden;background:rgba(12,24,34,.88)}} .cell-head{{display:flex;justify-content:space-between;align-items:center;padding:18px 20px;border-bottom:1px solid var(--line)}} .badge{{max-width:65%;padding:6px 10px;border-radius:10px;font-size:12px;font-weight:750;text-align:right}}
.cell-body{{padding:18px}} .source-layout{{display:grid;grid-template-columns:minmax(260px,420px) 1fr;gap:18px;margin-bottom:20px}} .source-video video{{aspect-ratio:432/544}} .instructions{{display:flex;flex-direction:column;gap:12px}} .instruction,.warning{{padding:14px 16px;border:1px solid var(--line);background:#0b1721;border-radius:12px}} .instruction.primary{{border-color:#3f7da8;background:#102537}} .instruction p{{margin:6px 0 0}} .warning{{color:#ffd6a6;border-color:#8a633c;background:#2a2117}} .numbers{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:20px 0}} h3{{margin:8px 0}} .hint{{color:var(--muted);font-size:12px;margin:0 0 8px}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{border-bottom:1px solid var(--line);padding:9px 7px;text-align:left;font-size:12px}} th{{color:var(--muted)}} td.ok{{color:var(--good)}} td.bad{{color:var(--bad);font-weight:750}}
.video-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}} figure{{margin:0;border:1px solid var(--line);background:#050a0e;border-radius:12px;overflow:hidden}} figcaption{{display:flex;justify-content:space-between;gap:8px;padding:8px 10px;color:var(--muted);font:11px ui-monospace,SFMono-Regular,Menlo,monospace}} figcaption strong{{color:#dbeaf3;font-weight:650}} video{{display:block;width:100%;aspect-ratio:432/544;background:#000}} details{{margin-top:42px;color:var(--muted)}}
@media(max-width:980px){{.metrics,.protocol{{grid-template-columns:repeat(2,1fr)}}.source-layout,.numbers{{grid-template-columns:1fr}}.video-grid{{grid-template-columns:repeat(2,1fr)}}}} @media(max-width:600px){{main{{padding:24px 12px 60px}}.metrics,.protocol,.video-grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">Action editing · independent calibration · 2026-08-14</div>
<h1>Frozen two-head calibration <span class="verdict {verdict_class}">{verdict}</span></h1>
<p class="lead">Fit-only 选择的 representation 在 6 个独立 calibration source 上原样执行。它能稳定区分 forward 与 noop，但不能稳定区分 reverse / incomplete，因而 confirmation 和 optimizer 权限保持关闭。</p>
<div class="metrics">{metrics}</div>
<div class="protocol">
 <div><div class="eyebrow">head A</div><code>{esc(frozen['head_a']['candidate_id'])}</code><br>weight {float(frozen['weight_a']):.2f}</div>
 <div><div class="eyebrow">head B</div><code>{esc(frozen['head_b']['candidate_id'])}</code><br>weight {float(frozen['weight_b']):.2f}</div>
 <div><div class="eyebrow">gate</div><code>each ordinal margin &gt; 0</code><br>no raw-score threshold search</div>
</div>
<section class="definition">
 <div class="eyebrow">这些 value 到底是什么</div><h2>它们是同一 cell 内的相对诊断量，不是成功率</h2>
 <p class="formula"><code>S(branch) = 0.61 · S_speed-profile(branch) + 0.39 · S_temporal-self-similarity(branch)</code></p>
 <p>两个 head 都来自 fit-only 的 frozen DINO temporal representation：先投影掉 fit controls 学到的 camera / appearance nuisance subspace，再计算该分支与 action basis 的对齐分数。</p>
 <p class="formula"><code>margin(forward &gt; negative) = S(forward) − S(negative)</code></p>
 <p><strong>margin &gt; 0 才通过。</strong>例如 −0.140806 表示 reverse 比 forward 高 0.140806；它不表示“质量下降 14%”。不同 cell 或不同 evaluator 的绝对 S 值不应直接横向比较。</p>
</section>
{cells}
<details><summary>Receipt / authority</summary><p><code>{esc(digest)}</code></p><p>representation reselection: false · optimizer step: false · confirmation evaluation: {str(calibration['confirmation_evaluation_authorized']).lower()}</p></details>
</main></body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--source-media-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema_version") != "bernini-frozen-multiscene-factorial-calibration-v1":
        raise RuntimeError("calibration report schema differs")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    document = build(
        report, bundle_root=args.bundle_root, media_root=args.media_root,
        source_media_root=args.source_media_root, manifest=manifest,
    )
    args.output.write_text(document, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

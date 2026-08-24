#!/usr/bin/env python3
"""Audit the legacy full644 exact28 panel and build a failure review.

Visual promotion is fail-closed on the decoded exact81 trajectory.  Phase-zero
and action-proxy metrics remain diagnostics only: they can never turn an
invalid decoded video into a passing result.  Passing this media-integrity
gate still cannot promote the legacy self-generated-anchor objective: that
route is not the formal exact160 contract.  Failed videos intentionally stay
visible in the scientific page, with an unmistakable failure label, so the
review cannot hide model collapse.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

try:
    from .visual_validity_gate_v1 import (
        SCHEMA_VERSION as VISUAL_VALIDITY_SCHEMA_VERSION,
        evaluate_visual_validity,
    )
except ImportError:  # Direct execution from this directory.
    from visual_validity_gate_v1 import (
        SCHEMA_VERSION as VISUAL_VALIDITY_SCHEMA_VERSION,
        evaluate_visual_validity,
    )


VARIANTS = ("frozen", "seed20260820", "seed20260821", "seed20260822")
TRAINED_VARIANTS = tuple(item for item in VARIANTS if item != "frozen")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def load_video(path: Path) -> np.ndarray:
    import decord

    reader = decord.VideoReader(str(path), width=248, height=240, num_threads=2)
    if len(reader) != 81 or not math.isclose(float(reader.get_avg_fps()), 25.0, abs_tol=1e-6):
        raise ValueError(f"video is not exact81/25fps: {path}")
    return reader.get_batch(list(range(81))).asnumpy().astype(np.float32) / 255.0


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1).astype(np.float64)
    b = right.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def metrics(source: np.ndarray, anchor: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    phase0_mse = float(np.mean((candidate[0] - source[0]) ** 2))
    phase0_psnr = 99.0 if phase0_mse == 0 else -10.0 * math.log10(phase0_mse)
    x, y = source[0].astype(np.float64), candidate[0].astype(np.float64)
    c1, c2 = 0.01**2, 0.03**2
    global_ssim = ((2*x.mean()*y.mean()+c1)*(2*np.mean((x-x.mean())*(y-y.mean()))+c2)) / (
        (x.mean()**2+y.mean()**2+c1)*(x.var()+y.var()+c2)
    )
    checkpoints = (20, 40, 60, 80)
    action_cos = np.mean([
        cosine(candidate[t] - candidate[0], anchor[t] - anchor[0]) for t in checkpoints
    ])
    candidate_motion = np.mean(np.abs(np.diff(candidate, axis=0)), axis=(1, 2, 3))
    anchor_motion = np.mean(np.abs(np.diff(anchor, axis=0)), axis=(1, 2, 3))
    motion_corr = float(np.corrcoef(candidate_motion, anchor_motion)[0, 1])
    if not math.isfinite(motion_corr):
        motion_corr = 0.0
    endpoint_l1 = float(np.mean(np.abs(candidate[-1] - anchor[-1])))
    return {
        "action_delta_cosine": float(action_cos),
        "anchor_endpoint_l1": endpoint_l1,
        "motion_profile_correlation": motion_corr,
        "source_phase0_global_ssim": float(global_ssim),
        "source_phase0_psnr_db": float(phase0_psnr),
    }


def evaluate_candidate(
    source: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one variant without allowing proxy metrics to grant success."""

    visual_validity = evaluate_visual_validity(
        candidate,
        reference_frames=source,
    )
    proxy_metrics = metrics(source, anchor, candidate)
    trajectory = visual_validity.get("trajectory", {})
    reference_metrics = trajectory.get("reference_metrics", {})
    if reference_metrics:
        proxy_metrics.update(
            source_full_trajectory_global_ssim_mean=float(
                reference_metrics["full_trajectory_global_ssim_mean"]
            ),
            source_full_trajectory_l1_mean=float(
                reference_metrics["full_trajectory_l1_mean"]
            ),
            source_post_onset_global_ssim_mean=float(
                reference_metrics["post_onset_global_ssim_mean"]
            ),
        )
    return {
        "metrics": proxy_metrics,
        "visual_validity_status": visual_validity["status"],
        "visual_validity_passed": bool(visual_validity["passed"]),
        "failure_codes": list(visual_validity["failure_codes"]),
        "visual_validity": visual_validity,
    }


def promotion_summary(
    results: list[dict[str, Any]],
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Summarize media integrity without promoting the invalid legacy route."""

    if len(results) != expected_rows:
        raise ValueError(
            f"promotion gate expected {expected_rows} rows, got {len(results)}"
        )
    variant_summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        values = [row["variants"][variant] for row in results]
        passed_rows = sum(bool(value["visual_validity_passed"]) for value in values)
        failure_counts: dict[str, int] = {}
        for value in values:
            for code in value["failure_codes"]:
                failure_counts[code] = failure_counts.get(code, 0) + 1
        variant_summaries[variant] = {
            "evaluated_rows": expected_rows,
            "full81_valid_rows": passed_rows,
            "full81_failed_rows": expected_rows - passed_rows,
            "full81_valid_fraction": float(passed_rows / expected_rows),
            "failure_code_row_counts": dict(sorted(failure_counts.items())),
            "all_rows_full81_valid": passed_rows == expected_rows,
        }

    panel_valid = all(
        variant_summaries[variant]["all_rows_full81_valid"] for variant in VARIANTS
    )
    trained_candidates_valid = all(
        variant_summaries[variant]["all_rows_full81_valid"]
        for variant in TRAINED_VARIANTS
    )
    return {
        "gate_schema_version": VISUAL_VALIDITY_SCHEMA_VERSION,
        "gate_scope": "all 81 decoded frames and all 80 temporal transitions",
        "expected_rows_per_variant": expected_rows,
        "required_variants": list(VARIANTS),
        "phase0_metrics_can_authorize_promotion": False,
        "every_variant_row_must_pass": True,
        "evaluation_panel_valid": panel_valid,
        "trained_candidates_full81_valid": trained_candidates_valid,
        "visual_media_gate_all_pass": bool(panel_valid and trained_candidates_valid),
        "legacy_objective_formal_box_compliant": False,
        "scientific_promotion_blockers": [
            "legacy_self_generated_anchor_objective_is_not_exact160_target_grounded",
            "legacy_full644_membership_is_not_formal_clean_paired_membership",
        ],
        "promotion_authorized": False,
        "variants": variant_summaries,
    }


def _render_reference_cell(title: str, iid: str, filename: str) -> str:
    return (
        f'<div class="cell reference"><h4>{html.escape(title)}</h4>'
        f'<video controls muted loop playsinline preload="metadata" '
        f'src="media/{html.escape(iid, quote=True)}/{html.escape(filename, quote=True)}"></video>'
        '<small>reference media (not a generated variant)</small></div>'
    )


def _render_variant_cell(
    title: str,
    iid: str,
    filename: str,
    values: dict[str, Any],
) -> str:
    """Render generated media even when invalid, with a prominent red failure."""

    passed = bool(values["visual_validity_passed"])
    status = "PASS" if passed else "FAIL"
    status_class = "validity-pass" if passed else "validity-fail"
    metrics_values = values["metrics"]
    full_ssim = metrics_values.get("source_full_trajectory_global_ssim_mean")
    full_ssim_text = "n/a" if full_ssim is None else f"{full_ssim:.3f}"
    codes = values["failure_codes"]
    failure_text = "none" if not codes else ", ".join(codes)
    diagnostic_text = (
        f"full81 source SSIM {full_ssim_text} · "
        f"action cos {metrics_values['action_delta_cosine']:.3f} · "
        f"motion corr {metrics_values['motion_profile_correlation']:.3f}"
    )
    return (
        f'<div class="cell {status_class}"><h4>{html.escape(title)} '
        f'<span class="badge {status_class}">FULL81 {status}</span></h4>'
        f'<video controls muted loop playsinline preload="metadata" '
        f'src="media/{html.escape(iid, quote=True)}/{html.escape(filename, quote=True)}"></video>'
        f'<strong class="gate-result">FULL81 VISUAL VALIDITY: {status}</strong>'
        f'<small>{html.escape(diagnostic_text)}</small>'
        f'<small class="failure-codes">failure codes: {html.escape(failure_text)}</small></div>'
    )


def copy_create_only(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"existing copied media differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as inp, destination.open("xb") as out:
        shutil.copyfileobj(inp, out, 1024 * 1024)
        out.flush()
        os.fsync(out.fileno())
    os.chmod(destination, 0o444)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    rows = selection["rows"]
    if len(rows) != 28 or len({row["family"] for row in rows}) != 28:
        raise ValueError("selection is not exact28 families")
    results: list[dict[str, Any]] = []
    for row in rows:
        iid = row["iid"]
        media_dir = args.eval_root / "media" / iid
        source_copy, anchor_copy = media_dir / "source.mp4", media_dir / "action_anchor.mp4"
        copy_create_only(Path(row["source_video_path"]), source_copy)
        copy_create_only(Path(row["action_anchor_video_path"]), anchor_copy)
        if sha256(source_copy) != row["source_video_sha256"]:
            raise ValueError(f"source SHA differs for {iid}")
        if sha256(anchor_copy) != row["action_anchor_video_sha256"]:
            raise ValueError(f"action anchor SHA differs for {iid}")
        source, anchor = load_video(source_copy), load_video(anchor_copy)
        variant_values = {}
        for variant in VARIANTS:
            video = media_dir / f"{variant}.mp4"
            receipt = Path(str(video) + ".receipt.json")
            if not video.is_file() or not receipt.is_file():
                raise FileNotFoundError(video)
            candidate = load_video(video)
            variant_values[variant] = evaluate_candidate(source, anchor, candidate)
            variant_values[variant].update(
                receipt_sha256=sha256(receipt),
                video_sha256=sha256(video),
            )
        results.append({**row, "variants": variant_values})
    aggregate = {}
    for variant in VARIANTS:
        keys = sorted(results[0]["variants"][variant]["metrics"])
        aggregate[variant] = {
            key: float(np.mean([row["variants"][variant]["metrics"][key] for row in results]))
            for key in keys
        }
    promotion = promotion_summary(results, expected_rows=28)
    report: dict[str, Any] = {
        "schema_version": "bernini-legacy-full644-self-generated-anchor-exact28-failure-audit-v3",
        "declared_legacy_training_coverage_rows": 644,
        "formal_box_training_rows": 0,
        "legacy_objective_formal_box_compliant": False,
        "evaluation_rows": 28,
        "action_family_count": 28,
        "generated_video_count": 112,
        "selection_sha256": sha256(args.selection),
        "metrics_are_engineering_proxies_not_identity_or_scientific_qualification": True,
        "phase0_metrics_are_diagnostic_only_and_never_authorize_promotion": True,
        "scientific_review_preserves_failed_media": True,
        "failed_generated_media_hidden": False,
        "promotion_authorized": promotion["promotion_authorized"],
        "promotion_gate": promotion,
        "aggregate": aggregate,
        "rows": results,
    }
    report["report_digest"] = canonical_digest(report)
    if args.output_metrics.exists() or args.output_html.exists():
        raise FileExistsError("report outputs must be fresh")
    args.output_metrics.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.chmod(args.output_metrics, 0o444)

    cards = []
    for row in results:
        iid, family = row["iid"], row["family"]
        videos = [
            _render_reference_cell("Source", iid, "source.mp4"),
            _render_reference_cell("Action anchor", iid, "action_anchor.mp4"),
        ]
        videos.extend(
            _render_variant_cell(
                variant,
                iid,
                f"{variant}.mp4",
                row["variants"][variant],
            )
            for variant in VARIANTS
        )
        row_has_failure = any(
            not row["variants"][variant]["visual_validity_passed"]
            for variant in VARIANTS
        )
        row_gate = (
            '<span class="row-gate validity-fail">FULL81 FAIL</span>'
            if row_has_failure
            else '<span class="row-gate validity-pass">FULL81 PASS</span>'
        )
        cards.append(
            f'<section class="row {"has-failure" if row_has_failure else ""}" '
            f'data-family="{html.escape(family, quote=True)}"><header><b>{html.escape(family)}</b> · '
            f'<code>{html.escape(iid)}</code> · strict={str(row["strict_selection_gates_all_true"]).lower()} '
            f'{row_gate}'
            f'<button onclick="syncRow(this)">同步播放本行</button></header>'
            f'<p>{html.escape(row["instruction"])}</p><div class="grid">{"".join(videos)}</div></section>'
        )
    agg_rows = "".join(
        f'<tr class="{"aggregate-pass" if promotion["variants"][variant]["all_rows_full81_valid"] else "aggregate-fail"}">'
        f"<th>{html.escape(variant)}</th>"
        f'<td><strong>{"PASS" if promotion["variants"][variant]["all_rows_full81_valid"] else "FAIL"}</strong></td>'
        f'<td>{promotion["variants"][variant]["full81_valid_rows"]}/28</td>'
        f'<td>{28 - promotion["variants"][variant]["full81_valid_rows"]}</td>'
        f'<td>{aggregate[variant].get("source_full_trajectory_global_ssim_mean", float("nan")):.4f}</td>'
        f'<td>{aggregate[variant]["action_delta_cosine"]:.4f}</td>'
        f'<td>{aggregate[variant]["motion_profile_correlation"]:.4f}</td>'
        f'<td>{aggregate[variant]["anchor_endpoint_l1"]:.4f}</td></tr>'
        for variant in VARIANTS
    )
    if promotion["visual_media_gate_all_pass"]:
        promotion_banner = (
            '<div class="promotion-banner promotion-blocked"><b>VISUAL MEDIA GATE PASSED; '
            'SCIENTIFIC PROMOTION STILL BLOCKED</b><br>All generated variants passed the '
            'catastrophic-media gate, but the legacy self-generated-anchor/full644 objective '
            'is not the formal exact160 target-grounded contract.</div>'
        )
    else:
        promotion_banner = (
            '<div class="promotion-banner promotion-fail"><b>PROMOTION BLOCKED — FULL81 '
            'VISUAL-VALIDITY FAILURE</b><br>At least one generated variant failed the '
            'full-trajectory gate. Failed videos remain visible below as scientific failure '
            'evidence; they are not hidden, replaced, or counted as success.</div>'
        )
    page = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Legacy Bernini full644 failure audit</title><style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#101216;color:#e9edf2}} h1{{margin-bottom:4px}} .note{{color:#aeb8c5;max-width:1100px}}
table{{border-collapse:collapse;margin:16px 0}}th,td{{border:1px solid #46505d;padding:7px 10px}} .row{{border-top:1px solid #3a424d;padding:18px 0}}
.row header{{font-size:18px}} button{{margin-left:12px}} .row p{{color:#bdc6d2;max-width:1300px}} .grid{{display:grid;grid-template-columns:repeat(6,minmax(210px,1fr));gap:10px;overflow-x:auto}}
.cell{{background:#1b1f26;padding:8px;border:2px solid transparent;border-radius:8px}} video{{width:100%;background:#000}} h4{{margin:0 0 6px}} small{{display:block;color:#9eadbd;margin-top:5px;min-height:1.2em}} code{{color:#9ad1ff}}
.cell.validity-fail{{background:#3b1015;border:3px solid #ff3b4f;box-shadow:0 0 0 2px #7e1722}} .cell.validity-pass{{border-color:#278c54}}
.badge,.row-gate{{display:inline-block;padding:2px 7px;border-radius:999px;font-size:12px;font-weight:800}} .validity-fail{{background:#ff2945;color:#fff}} .validity-pass{{background:#1f9d58;color:#fff}}
.gate-result{{display:block;margin-top:7px}} .cell.validity-fail .gate-result,.failure-codes{{color:#ff9aa7}} .row.has-failure{{border-left:5px solid #ff2945;padding-left:12px}}
.promotion-banner{{max-width:1100px;margin:14px 0;padding:14px 17px;border-radius:8px;font-size:16px}} .promotion-fail{{background:#4a0d16;border:3px solid #ff2945;color:#ffd9df}} .promotion-blocked{{background:#49340b;border:3px solid #e5a822;color:#fff1c4}}
.aggregate-fail{{background:#3b1015;color:#ffd9df}} .aggregate-pass{{background:#0d3822}}
</style></head><body><h1>旧 full644 self-generated-anchor 失败审计 · exact28</h1>
{promotion_banner}
<p class="note">旧 run 自报消费 644-row candidate catalog；这不等于 box 的 formal exact160 clean-paired membership，本页面也不授权任何模型。这里展示 28 rows（每个动作族一条，strict-first），列为 source、self-generated action anchor、frozen base、三个旧训练副本。每个生成视频的媒体门检查全部 81 帧和 80 个帧间过渡；phase0 SSIM 及其他 RGB 指标只是诊断代理，FULL81 PASS 也不等于动作正确或 scientific promotion。红色 FAIL 视频仍原样展示为失败证据。Qwen 不参与训练门。</p>
<table><thead><tr><th>variant</th><th>FULL81 gate</th><th>valid / 28</th><th>failed</th><th>source full81 SSIM ↑ (diagnostic)</th><th>action Δ cosine ↑ (diagnostic)</th><th>motion corr ↑ (diagnostic)</th><th>anchor endpoint L1 ↓ (diagnostic)</th></tr></thead><tbody>{agg_rows}</tbody></table>
{''.join(cards)}<script>function syncRow(b){{let vs=b.closest('.row').querySelectorAll('video');vs.forEach(v=>{{v.currentTime=0;v.play()}})}}</script></body></html>'''
    args.output_html.write_text(page)
    os.chmod(args.output_html, 0o444)
    print(json.dumps({"html": str(args.output_html), "html_sha256": sha256(args.output_html), "metrics": str(args.output_metrics), "metrics_sha256": sha256(args.output_metrics), "report_digest": report["report_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

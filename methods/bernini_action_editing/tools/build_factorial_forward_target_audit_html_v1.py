#!/usr/bin/env python3
"""Build a create-only HTML review for source-versus-forward-target pairs."""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "bernini-factorial-forward-target-audit-v1"
STATUSES = (
    "strict_eligible",
    "compound_instruction",
    "action_failure",
    "wrong_target_family",
)
MEDIA_FILES = (
    "comparison_source_target.mp4",
    "review_source_target_f0_20_40_60_80.jpg",
    "raw_candidate.json",
)


class ForwardTargetAuditHTMLError(RuntimeError):
    """Raised when review evidence is incomplete or mutable."""


def _plain_file(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise ForwardTargetAuditHTMLError(f"{label} must be an absolute plain file")
    return value.resolve(strict=True)


def _plain_dir(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or not value.is_dir() or value.is_symlink():
        raise ForwardTargetAuditHTMLError(f"{label} must be an absolute plain directory")
    return value.resolve(strict=True)


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ForwardTargetAuditHTMLError(f"cannot read {label}") from error
    if type(value) is not dict:
        raise ForwardTargetAuditHTMLError(f"{label} must contain one object")
    return value


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ForwardTargetAuditHTMLError(f"{label} must be non-empty text")
    return value


def _validate(review: Mapping[str, Any], media_root: Path) -> list[dict[str, Any]]:
    if review.get("schema_version") != SCHEMA:
        raise ForwardTargetAuditHTMLError("review schema differs")
    authority = review.get("authority")
    if not isinstance(authority, Mapping) or any(
        authority.get(field) is not False
        for field in (
            "factorial_negatives_present",
            "same_seed_pairing_verified",
            "training_target_authorized",
            "optimizer_step_authorized",
            "method_success_claimed",
        )
    ):
        raise ForwardTargetAuditHTMLError("review authority must remain fail-closed")
    raw_rows = review.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ForwardTargetAuditHTMLError("review rows must be non-empty")
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for index, raw in enumerate(raw_rows):
        if type(raw) is not dict or set(raw) != {
            "source_id",
            "action_family",
            "split",
            "status",
            "raw_prompt",
            "review_note",
        }:
            raise ForwardTargetAuditHTMLError(f"row {index} closure differs")
        row = dict(raw)
        source_id = _text(row["source_id"], label="source ID")
        if source_id in ids or "/" in source_id or source_id in {".", ".."}:
            raise ForwardTargetAuditHTMLError("source identity differs")
        ids.append(source_id)
        if row["status"] not in STATUSES:
            raise ForwardTargetAuditHTMLError("review status differs")
        for field in ("action_family", "split", "raw_prompt", "review_note"):
            _text(row[field], label=field)
        source_root = media_root / source_id
        if not source_root.is_dir() or source_root.is_symlink():
            raise ForwardTargetAuditHTMLError(f"media root missing for {source_id}")
        for basename in MEDIA_FILES:
            _plain_file(source_root / basename, label=f"{source_id}/{basename}")
        metadata = _load_object(source_root / "raw_candidate.json", label=source_id)
        if metadata.get("iid") != source_id or metadata.get("prompt") != row["raw_prompt"]:
            raise ForwardTargetAuditHTMLError("raw prompt binding differs")
        rows.append(row)
    if ids != sorted(ids):
        raise ForwardTargetAuditHTMLError("review rows must be sorted by source ID")
    counts = Counter(row["status"] for row in rows)
    expected = {"reviewed": len(rows), **{status: counts[status] for status in STATUSES}}
    if review.get("summary") != expected:
        raise ForwardTargetAuditHTMLError("review summary differs")
    return rows


def _badge(status: str) -> str:
    labels = {
        "strict_eligible": "strict eligible",
        "compound_instruction": "compound instruction",
        "action_failure": "action failure",
        "wrong_target_family": "wrong target family",
    }
    return f'<span class="badge {html.escape(status)}">{labels[status]}</span>'


def _page(review: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    summary = review["summary"]
    cards = []
    for row in rows:
        source_id = html.escape(str(row["source_id"]))
        cards.append(
            f"""
            <article class="case" id="case-{source_id}">
              <div class="case-head">
                <div><h2>{source_id}</h2><p>{html.escape(str(row['action_family']))} · {html.escape(str(row['split']))}</p></div>
                {_badge(str(row['status']))}
              </div>
              <p class="prompt"><strong>Raw prompt</strong> · {html.escape(str(row['raw_prompt']))}</p>
              <p class="note">{html.escape(str(row['review_note']))}</p>
              <p class="media-label">Video: source on the left · pre-existing edited target on the right</p>
              <video controls muted loop preload="metadata" src="{source_id}/comparison_source_target.mp4"></video>
              <details><summary>Frames 0 / 20 / 40 / 60 / 80</summary><img loading="lazy" src="{source_id}/review_source_target_f0_20_40_60_80.jpg" alt="source and target frame strip for {source_id}"></details>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Factorial prospective source ↔ existing forward-target audit</title>
<style>
:root{{--bg:#0b1020;--panel:#121a2d;--ink:#eef3ff;--muted:#9ca9c7;--line:#26334e;--good:#43d18d;--warn:#ffbe55;--bad:#ff6b7c;--violet:#a997ff}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(135deg,#080c17,#111a31);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}
main{{width:min(1480px,94vw);margin:auto;padding:42px 0 80px}} h1{{font-size:clamp(28px,4vw,52px);line-height:1.04;margin:.2em 0}} h2{{margin:0;font-size:20px}} p{{margin:.45em 0}} .lede{{max-width:1000px;color:#c7d2ec;font-size:17px}}
.summary{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:12px;margin:28px 0}} .metric{{padding:18px;border:1px solid var(--line);border-radius:14px;background:#10182a}} .metric b{{display:block;font-size:30px}} .metric span{{color:var(--muted)}}
.policy{{border-left:4px solid var(--warn);padding:14px 18px;background:#171a28;border-radius:8px;margin:18px 0 32px}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}
.case{{background:rgba(18,26,45,.94);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 18px 45px rgba(0,0,0,.18)}} .case-head{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}} .case-head p,.media-label{{color:var(--muted)}}
.badge{{white-space:nowrap;padding:5px 10px;border-radius:999px;font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.04em}} .strict_eligible{{background:rgba(67,209,141,.16);color:var(--good)}} .compound_instruction{{background:rgba(255,190,85,.15);color:var(--warn)}} .action_failure{{background:rgba(255,107,124,.15);color:var(--bad)}} .wrong_target_family{{background:rgba(169,151,255,.16);color:var(--violet)}}
.prompt{{min-height:48px}} .note{{min-height:48px;color:#d8e0f4}} video,img{{display:block;width:100%;border-radius:10px;background:#060911}} video{{aspect-ratio:2/1;object-fit:contain}} details{{margin-top:12px}} summary{{cursor:pointer;color:#b9c8e8;margin-bottom:10px}}
@media(max-width:900px){{.summary{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}.prompt,.note{{min-height:0}}}}
</style>
</head>
<body><main>
<p>2026-08-13 · prospective data audit</p>
<h1>Source ↔ existing synthetic forward target</h1>
<p class="lede">This page answers whether the pre-existing Goku edited clips can be reused as clean action targets for the newly frozen factorial population. The answer is mostly no: action completion is common, but prompt-level compound events contaminate the motion representation.</p>
<section class="summary">
  <div class="metric"><b>{summary['reviewed']}</b><span>reviewed</span></div>
  <div class="metric"><b>{summary['strict_eligible']}</b><span>strict eligible</span></div>
  <div class="metric"><b>{summary['compound_instruction']}</b><span>compound instruction</span></div>
  <div class="metric"><b>{summary['action_failure']}</b><span>action failure</span></div>
  <div class="metric"><b>{summary['wrong_target_family']}</b><span>wrong family</span></div>
</section>
<div class="policy"><strong>Authority remains closed.</strong> These pairs have no matched noop/reverse/incomplete/camera/appearance/wrong-owner branches and no verified same-seed Gaussian pairing. Even “strict eligible” means forward-target scouting only; it does not authorize training or a method claim.</div>
<section class="grid">{''.join(cards)}</section>
</main></body></html>"""


def build(*, review_path: str | Path, media_root: str | Path, output: str | Path) -> dict[str, Any]:
    review_file = _plain_file(review_path, label="review manifest")
    root = _plain_dir(media_root, label="media root")
    review = _load_object(review_file, label="review manifest")
    rows = _validate(review, root)
    destination = Path(output)
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise ForwardTargetAuditHTMLError("output must be a fresh absolute path")
    payload = _page(review, rows).encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {"output": str(destination), "row_count": len(rows), "bytes": len(payload)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(review_path=args.review, media_root=args.media_root, output=args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

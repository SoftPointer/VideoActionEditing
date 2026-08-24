#!/usr/bin/env python3
"""Build the direct, unranked R64 held-out preservation review page."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generic_source_carrier_r64_heldout_contract_v1 as contract  # noqa: E402


SCHEMA_VERSION = "bernini-generic-source-carrier-r64-heldout-html-v1"


class R64HeldoutHtmlError(RuntimeError):
    """Raised before a partial or mislabeled page is written."""


def fail(message: str) -> NoReturn:
    raise R64HeldoutHtmlError(message)


def _plain_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail("input-dir must be an absolute non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise R64HeldoutHtmlError("input-dir is unavailable") from error
    if resolved != path or not path.is_dir() or path.is_symlink():
        fail("input-dir must be one canonical plain directory")
    return path


def _strict_receipt(
    root: Path, *, runtime_source_revision: str,
    runtime_source_closure_sha256: str, launcher_sha256: str,
) -> Mapping[str, Any]:
    path = root / "receipt.json"
    if path.is_symlink() or not path.is_file():
        fail("inference receipt must be a plain file")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise R64HeldoutHtmlError("cannot read inference receipt") from error
    if not isinstance(value, Mapping):
        fail("inference receipt root must be an object")
    try:
        return contract.validate_receipt(
            value,
            expected_runtime_source_revision=runtime_source_revision,
            expected_runtime_source_closure_sha256=runtime_source_closure_sha256,
            expected_launcher_sha256=launcher_sha256,
            media_root=root,
            verify_media=True,
        )
    except contract.R64HeldoutContractError as error:
        raise R64HeldoutHtmlError(str(error)) from error


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _video(relative: str, *, label: str) -> str:
    return (
        f'<video controls loop muted playsinline preload="metadata" '
        f'aria-label="{_esc(label)}"><source src="{_esc(relative)}" '
        'type="video/mp4"></video>'
    )


def render_html(receipt: Mapping[str, Any]) -> str:
    rows = {
        (str(row["iid"]), str(row["arm"])): row for row in receipt["rows"]
    }
    sources = {str(row["iid"]): row for row in receipt["sources"]}
    sections: list[str] = []
    for iid in sorted(sources):
        source = sources[iid]
        base = rows[(iid, "frozen-base")]
        trained = rows[(iid, "trained-carrier-r64")]
        if (
            base["source_video_sha256"] != source["source_video_sha256"]
            or trained["source_video_sha256"] != source["source_video_sha256"]
            or base["seed"] != trained["seed"]
            or base["initial_gaussian_sha256"]
            != trained["initial_gaussian_sha256"]
        ):
            fail(f"{iid} page pair binding differs")
        cards = []
        for title, row, explanation in (
            (
                "Frozen base",
                base,
                "Same loaded Bernini model; trained carrier route authenticated but disabled.",
            ),
            (
                "Trained carrier · R64",
                trained,
                "Only the strictly loaded blocks 8/12/16/20 source carrier is enabled.",
            ),
        ):
            cards.append(
                '<article class="card">'
                f'<h3>{_esc(title)}</h3>'
                + _video(str(row["relative_mp4"]), label=f"{iid} {title}")
                + f'<p>{_esc(explanation)}</p>'
                + f'<p class="hash">MP4 <code>{_esc(row["mp4_sha256"])}</code></p>'
                '</article>'
            )
        sections.append(
            '<section class="example">'
            f'<header><p class="eyebrow">Held-out IID {_esc(iid)} · '
            f'family provenance only: {_esc(source["action_family_provenance_only"])}</p>'
            f'<h2>{_esc(iid)}</h2>'
            f'<p>Seed <code>{_esc(source["seed"])}</code> · source SHA '
            f'<code>{_esc(source["source_video_sha256"])}</code> · official Gaussian '
            f'<code>{_esc(base["initial_gaussian_sha256"])}</code></p></header>'
            '<div class="source-row"><article class="card source"><h3>Real source anchor</h3>'
            + _video(str(source["relative_mp4"]), label=f"{iid} real source anchor")
            + '<p>This exact real source is the native RV2V condition for both arms.</p></article></div>'
            '<div class="pair">' + ''.join(cards) + '</div>'
            '</section>'
        )
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    checkpoint_sha = receipt["r64_authority"]["checkpoint_file_sha256"]
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R64 held-out preservation · Frozen base vs trained carrier</title>
<style>
:root{{--ink:#17202a;--muted:#5d6875;--line:#d7dee7;--paper:#f4f7fb;--card:#fff;--accent:#155eef;--warn:#8a4b08}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1500px;margin:auto;padding:34px 24px 70px}} .hero,.example{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;margin-bottom:24px;box-shadow:0 4px 18px #23334d0d}}
h1,h2,h3{{line-height:1.16;margin:.2em 0 .55em}} h1{{font-size:clamp(28px,4vw,48px)}} h2{{font-size:24px}} h3{{font-size:18px}} .eyebrow{{text-transform:uppercase;letter-spacing:.08em;font-weight:750;color:var(--accent);font-size:12px}}
.warning{{border-left:5px solid #e38b2c;background:#fff6e9;padding:14px 16px;color:var(--warn);font-weight:650}} code{{overflow-wrap:anywhere;font-size:12px}} .pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}} .source-row{{max-width:720px;margin:16px 0 22px}}
.card{{border:1px solid var(--line);border-radius:12px;padding:15px;background:#fcfdff}} video{{display:block;width:100%;background:#090b0f;border-radius:9px;aspect-ratio:16/9;object-fit:contain}} .hash{{color:var(--muted)}}
.contract{{display:grid;grid-template-columns:max-content 1fr;gap:7px 14px}} .contract dt{{font-weight:700}} .contract dd{{margin:0}} footer{{color:var(--muted);font-size:13px}}
@media(max-width:820px){{.pair{{grid-template-columns:1fr}} main{{padding:18px 10px 50px}} .hero,.example{{padding:16px}}}}
</style></head><body><main>
<section class="hero"><p class="eyebrow">Manual preservation review · all 8 true held-out sources</p>
<h1>Frozen base vs Trained source carrier · R64</h1>
<p class="warning">This page is not an action-editing result. R64 trained only the source-retention carrier; planner/operator updates are exactly zero. No score, reward, ranking, selection, or automatic quality verdict is present.</p>
<dl class="contract"><dt>Comparison</dt><dd>8 IIDs × 2 arms = 16 exact81 MP4s</dd><dt>Pair lock</dt><dd>same real source · same no-op prompt · same seed · same observed official Gaussian · native exact40</dd><dt>Trainable scope</dt><dd>carrier blocks 8, 12, 16, 20 only</dd><dt>R64 checkpoint</dt><dd><code>{_esc(checkpoint_sha)}</code></dd><dt>No-op prompt</dt><dd>{_esc(contract.GENERIC_NOOP_INSTRUCTION)}</dd></dl>
</section>
{''.join(sections)}
<footer>Generated {_esc(created)} · receipt digest <code>{_esc(receipt['receipt_digest'])}</code> · manual review pending.</footer>
</main></body></html>'''


def _write_create_only(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite {path.name}")
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-closure-sha256", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    args = parser.parse_args(argv)
    root = _plain_root(args.input_dir)
    receipt = _strict_receipt(
        root,
        runtime_source_revision=args.runtime_source_revision,
        runtime_source_closure_sha256=args.runtime_source_closure_sha256,
        launcher_sha256=args.launcher_sha256,
    )
    html_path = root / "index.html"
    html_payload = render_html(receipt).encode("utf-8")
    _write_create_only(html_path, html_payload)
    html_sha = contract.file_sha256(html_path)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "input_receipt": "receipt.json",
        "input_receipt_file_sha256": contract.file_sha256(root / "receipt.json"),
        "input_receipt_digest": receipt["receipt_digest"],
        "html": "index.html", "html_sha256": html_sha,
        "heldout_rows": contract.HELDOUT_ROWS,
        "paired_mp4_rows": contract.MEDIA_ROWS,
        "source_video_rows": contract.HELDOUT_ROWS,
        "manual_review_pending": True,
        "score_present": False, "ranking_present": False,
        "complete_action_result": False, "action_claim_forbidden": True,
    }
    value = {**unsigned, "receipt_digest": contract.object_sha256(unsigned)}
    _write_create_only(
        root / "html_receipt.json", contract.canonical_json_bytes(value) + b"\n"
    )
    print(contract.canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["R64HeldoutHtmlError", "SCHEMA_VERSION", "main", "render_html"]

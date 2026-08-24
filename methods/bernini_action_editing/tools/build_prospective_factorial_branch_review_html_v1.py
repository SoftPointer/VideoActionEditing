#!/usr/bin/env python3
"""Build a create-only comparison HTML for released seven-branch cells."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import prospective_factorial_branch_manifest_v1 as branch_manifest  # noqa: E402
import run_prospective_factorial_branch_shard_v1 as runner  # noqa: E402


DISPLAY_ORDER = (
    "noop",
    "forward",
    "reverse",
    "incomplete",
    "camera_only",
    "appearance_only",
    "wrong_actor_or_object",
)


class FactorialBranchReviewHTMLError(RuntimeError):
    """Raised before incomplete branch evidence is exposed as a review packet."""


def _page(
    manifest: Mapping[str, Any],
    cells: Sequence[tuple[str, int]],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    by_cell = {
        cell: sorted(
            [row for row in rows if (row["source_id"], row["seed"]) == cell],
            key=lambda row: DISPLAY_ORDER.index(row["branch"]),
        )
        for cell in cells
    }
    cell_cards: list[str] = []
    for source_id, seed in cells:
        cell_rows = by_cell[(source_id, seed)]
        family = cell_rows[0]["action_family"]
        branch_cards: list[str] = []
        for row in cell_rows:
            branch = str(row["branch"])
            label = "noop / exact source" if branch == "noop" else branch.replace("_", "-")
            entry_id = html.escape(str(row["entry_id"]), quote=True)
            branch_cards.append(
                f"""
                <article class="branch branch-{html.escape(branch)}">
                  <div class="branch-head"><h3>{html.escape(label)}</h3><span>unreviewed</span></div>
                  <video controls muted loop preload="metadata" src="entries/{entry_id}/output.mp4"></video>
                  <details><summary>Frozen instruction</summary><p>{html.escape(str(row['instruction']))}</p></details>
                </article>
                """
            )
        cell_cards.append(
            f"""
            <section class="cell" id="cell-{source_id}-{seed}">
              <div class="cell-head"><div><h2>{html.escape(source_id)}</h2><p>{html.escape(str(family))} · fit · seed {seed}</p></div><b>complete 7-way cell</b></div>
              <p class="hint">Read noop first as the exact source, then compare forward and every negative against it. Review the full videos; cards are not ranked and no candidate has training authority.</p>
              <div class="grid">{''.join(branch_cards)}</div>
            </section>
            """
        )
    digest = html.escape(str(manifest["manifest_digest"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prospective seven-branch action editing review</title>
<style>
:root{{--bg:#080d18;--panel:#111a2c;--panel2:#162139;--ink:#edf3ff;--muted:#9baac8;--line:#273652;--green:#49d69a;--amber:#ffc56b;--purple:#ab9cff}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at top,#172544 0,#080d18 48%);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}}
main{{width:min(1540px,95vw);margin:auto;padding:42px 0 90px}} h1{{font-size:clamp(30px,4vw,54px);line-height:1.04;margin:.2em 0}} h2,h3,p{{margin:.35em 0}} .lede{{max-width:1050px;color:#c6d2ea;font-size:17px}}
.gate{{margin:24px 0 34px;padding:15px 18px;border-left:4px solid var(--amber);border-radius:9px;background:#191c28}} .cell{{margin:24px 0;padding:20px;border:1px solid var(--line);border-radius:18px;background:rgba(17,26,44,.95)}}
.cell-head,.branch-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:14px}} .cell-head p,.hint{{color:var(--muted)}} .cell-head b{{color:var(--green)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px;margin-top:18px}} .branch{{padding:14px;border:1px solid var(--line);border-radius:13px;background:var(--panel2)}} .branch:first-child{{grid-column:1/-1;border-color:#3b8068}} .branch-head span{{color:var(--amber);font-size:12px;text-transform:uppercase;letter-spacing:.06em}}
video{{display:block;width:100%;max-height:460px;object-fit:contain;background:#03060c;border-radius:9px;margin-top:9px}} details{{margin-top:9px;color:#cbd6ed}} summary{{cursor:pointer;color:var(--purple)}}
code{{overflow-wrap:anywhere}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}.branch:first-child{{grid-column:auto}}}}
</style>
</head>
<body><main>
<p>2026-08-13 · decoded fit smoke · frozen Bernini</p>
<h1>Exact source/noop ↔ six generated branches</h1>
<p class="lede">Every section is one source/seed factorial cell. The noop video is a byte-identical source copy. The remaining six videos use the same source and seed, so forward must be judged against noop, reverse, incomplete, camera-only, appearance-only, and wrong-owner together.</p>
<div class="gate"><strong>Review only — authority closed.</strong> Decoding completion does not make these clips training targets. Prompt semantics, action phase, owner, identity, camera, background, and artifacts still require full-video review. Manifest digest: <code>{digest}</code>.</div>
{''.join(cell_cards)}
</main></body></html>"""


def build(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    cells: Sequence[str],
    output_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    try:
        manifest, _ = runner._load_manifest(manifest_path, expected_manifest_sha256)
        parsed_cells = runner._cells(cells)
        rows = runner._released_entries(manifest, parsed_cells, "fit")
        root = runner._plain_directory(output_root, label="review output root")
        for row in rows:
            runner._verify_entry(row, root, expected_manifest_sha256)
    except (runner.FactorialBranchRunError, branch_manifest.FactorialBranchManifestError) as error:
        raise FactorialBranchReviewHTMLError(str(error)) from error
    destination = Path(output)
    if not destination.is_absolute() or destination.parent.resolve() != root:
        raise FactorialBranchReviewHTMLError("HTML output must be inside the release root")
    if destination.exists() or destination.is_symlink():
        raise FactorialBranchReviewHTMLError("HTML output must be fresh")
    payload = _page(manifest, parsed_cells, rows).encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "output": str(destination),
        "cell_count": len(parsed_cells),
        "video_link_count": len(rows),
        "bytes": len(payload),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build(
                manifest_path=args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                cells=args.cell,
                output_root=args.output_root,
                output=args.output,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FactorialBranchReviewHTMLError as error:
        print(f"[factorial-branch-review-html] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

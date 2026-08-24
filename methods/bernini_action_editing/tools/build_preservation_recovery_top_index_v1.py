#!/usr/bin/env python3
"""Publish a minimal top-level index over the fixed F0/A0/A1 review paths."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


ENTRIES = (
    {
        "stage": "F0",
        "relative_dir": "F0_checkpoint_dynamics",
        "marker": "F0 旧两样本 preservation-residual 诊断",
        "identity": "旧两条 real-source preservation-residual 连续 checkpoint dynamics",
        "training": True,
        "detail": "同一连续训练轨迹的 optimizer step 0 / 20 / 40；仅作旧方法动态诊断。",
    },
    {
        "stage": "A0",
        "relative_dir": "A0_native_v_axis",
        "marker": "Native full-video V-axis review",
        "identity": "Frozen native full-video V-axis causal probe",
        "training": False,
        "detail": "V-on / V-off / wrong-V；不含 optimizer 或参数更新。",
    },
    {
        "stage": "A1",
        "relative_dir": "A1_source_edge_formal_grid",
        "marker": "Formal source-edge grid",
        "identity": "Frozen formal schedule × block source-edge localization",
        "training": False,
        "detail": "4 schedules × 2 block bands × typed controls；不含 optimizer 或参数更新。",
    },
)


class PreservationTopIndexError(RuntimeError):
    """Raised before a top-level index can point at the wrong experiment."""


def fail(message: str) -> NoReturn:
    raise PreservationTopIndexError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_review_root(raw: str | Path) -> Path:
    requested = Path(raw).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail("review root must be an absolute non-symlink directory")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise PreservationTopIndexError(f"missing review root: {requested}") from error
    if root != requested or not root.is_dir():
        fail("review root directory differs")
    return root


def _validate_entry(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    relative_dir = str(spec["relative_dir"])
    directory = root / relative_dir
    if directory.is_symlink():
        fail(f"{spec['stage']} review directory must not be a symlink")
    try:
        resolved_dir = directory.resolve(strict=True)
    except OSError as error:
        raise PreservationTopIndexError(
            f"missing fixed {spec['stage']} review directory: {relative_dir}"
        ) from error
    if resolved_dir != directory or not directory.is_dir() or root not in directory.parents:
        fail(f"{spec['stage']} fixed review directory differs")
    index = directory / "index.html"
    if index.is_symlink():
        fail(f"{spec['stage']} child index must not be a symlink")
    try:
        resolved_index = index.resolve(strict=True)
        raw = index.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise PreservationTopIndexError(
            f"cannot read {spec['stage']} child index"
        ) from error
    if (
        resolved_index != index
        or not index.is_file()
        or directory not in index.parents
        or str(spec["marker"]) not in text
    ):
        fail(f"{spec['stage']} child index experiment identity differs")
    return {
        **dict(spec),
        "href": f"{relative_dir}/index.html",
        "index_sha256": hashlib.sha256(raw).hexdigest(),
    }


def render_html(entries: Sequence[Mapping[str, Any]]) -> str:
    cards = "".join(
        f"""
        <a class="card" href="{html.escape(str(item['href']), quote=True)}">
          <div class="head"><span class="stage">{html.escape(str(item['stage']))}</span><span class="training {'yes' if item['training'] else 'no'}">{'TRAINED' if item['training'] else 'FROZEN / NO TRAINING'}</span></div>
          <h2>{html.escape(str(item['identity']))}</h2><p>{html.escape(str(item['detail']))}</p>
          <dl><dt>固定相对目录</dt><dd><code>{html.escape(str(item['relative_dir']))}/</code></dd><dt>child index SHA-256</dt><dd><code>{item['index_sha256']}</code></dd></dl>
        </a>"""
        for item in entries
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Preservation recovery · F0 / A0 / A1</title><style>
:root{{--bg:#071018;--panel:#122230;--line:#315269;--ink:#f1f7fb;--muted:#abc0ce;--cyan:#64d9df;--green:#78d9a5;--amber:#ffca70}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(150deg,#061019,#0a1b29);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{width:min(1100px,94vw);margin:auto;padding:48px 0 72px}}h1{{font-size:clamp(34px,6vw,64px);line-height:1.04;margin:.1em 0}}.lede{{color:var(--muted);font-size:18px}}.grid{{display:grid;gap:14px;margin-top:28px}}.card{{display:block;padding:20px;border:1px solid var(--line);border-radius:15px;background:var(--panel);color:var(--ink);text-decoration:none}}.card:hover{{border-color:var(--cyan);transform:translateY(-1px)}}.head{{display:flex;justify-content:space-between;gap:12px}}.stage{{font-size:24px;font-weight:900;color:var(--cyan)}}.training{{padding:5px 9px;border-radius:999px;font-size:12px;font-weight:900}}.training.yes{{color:var(--amber);background:rgba(255,202,112,.12)}}.training.no{{color:var(--green);background:rgba(120,217,165,.12)}}h2{{margin:.35em 0}}p{{color:#d3e0e9}}dl{{display:grid;grid-template-columns:150px 1fr;gap:5px;margin:0}}dt{{color:var(--muted)}}dd{{margin:0}}code{{color:var(--cyan);overflow-wrap:anywhere;font-size:11px}}@media(max-width:650px){{.head{{display:block}}dl{{grid-template-columns:1fr}}}}
</style></head><body><main><p>2026-08-14 · strict preservation recovery</p><h1>F0 / A0 / A1 review index</h1><p class="lede">三个页面是不同实验，不能把结果互相归并：F0 是旧方法训练动态；A0 与 A1 是 frozen-model causal localization，没有训练。</p><section class="grid">{cards}</section></main></body></html>"""


def build(*, review_root: str | Path) -> Path:
    root = _plain_review_root(review_root)
    output = root / "index.html"
    if output.exists() or output.is_symlink():
        fail("top-level index.html must be fresh")
    entries = [_validate_entry(root, spec) for spec in ENTRIES]
    rendered = render_html(entries).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=root, prefix=".preservation-top-index.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if output.exists() and not output.is_symlink():
            output.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(build(review_root=args.review_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

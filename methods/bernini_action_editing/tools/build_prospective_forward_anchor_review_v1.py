#!/usr/bin/env python3
"""Build a 12-cell noop/forward review packet from two sealed releases."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, NoReturn


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

FORWARD_NAME = "run_prospective_forward_anchor_shard_v1.py"
FORWARD_SHA256 = "acad67318bbaf3c575ca288c654126012390b6226267e10bcdabfad250e1bd96"
FORWARD_PATH = METHOD_ROOT / FORWARD_NAME
if (
    not FORWARD_PATH.is_file() or FORWARD_PATH.is_symlink()
    or hashlib.sha256(FORWARD_PATH.read_bytes()).hexdigest() != FORWARD_SHA256
):
    raise RuntimeError("pinned forward-anchor runtime dependency differs")

import run_prospective_forward_anchor_shard_v1 as forward_runner  # noqa: E402


SCHEMA_VERSION = "bernini-prospective-forward-anchor-review-v1"
OLD_CELLS = (
    ("0b2fc177202e4d08", 2026082511),
    ("1367d5595ed641ae", 2026082601),
)
NEW_CELLS = (
    ("0b2fc177202e4d08", 2026082512),
    ("1367d5595ed641ae", 2026082602),
    ("173371bf8fa74785", 2026082521),
    ("173371bf8fa74785", 2026082522),
    ("31fcd6205efb4b84", 2026082631),
    ("31fcd6205efb4b84", 2026082632),
    ("5f4ba4fb4c6441e0", 2026082561),
    ("5f4ba4fb4c6441e0", 2026082562),
    ("7421728d949d40dd", 2026082671),
    ("7421728d949d40dd", 2026082672),
)
ALL_CELLS = tuple(sorted(OLD_CELLS + NEW_CELLS))
AUTHORITY = {
    "decoded_review_only": True,
    "training_target_authorized": False,
    "representation_selection_authorized": False,
    "optimizer_step_authorized": False,
    "method_success_claimed": False,
}


class ForwardAnchorReviewError(RuntimeError):
    """Raised before a partial or path-unbound review is written."""


def fail(message: str) -> NoReturn:
    raise ForwardAnchorReviewError(message)


def plain_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        fail(f"{label} must be one absolute plain directory")
    return path.resolve(strict=True)


def create_output(path: Path) -> Path:
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        fail("output root must be fresh, absolute, and non-root")
    path.mkdir(mode=0o700, parents=False)
    return path


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"{label} differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ForwardAnchorReviewError(f"cannot read {label}") from error
    if type(value) is not dict:
        fail(f"{label} must contain an object")
    return value


def entry(
    release: Path, source: str, seed: int, branch: str,
    *, manifest_sha256: str,
) -> dict[str, Any]:
    entry_id = f"{source}-s{seed}-{branch}"
    root = release / "entries" / entry_id
    receipt_path = root / "entry.receipt.json"
    receipt = read_json(receipt_path, label=f"{entry_id} receipt")
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest", None)
    video = root / "output.mp4"
    if (
        forward_runner.base.branch_manifest.object_sha256(unsigned) != digest
        or receipt.get("entry_id") != entry_id
        or receipt.get("source_id") != source
        or receipt.get("seed") != seed
        or receipt.get("branch") != branch
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("training_target_authorized") is not False
        or receipt.get("optimizer_step_authorized") is not False
        or not video.is_file() or video.is_symlink()
        or forward_runner.base.branch_manifest.file_sha256(video)
        != receipt.get("output_sha256")
    ):
        fail(f"{entry_id} binding differs")
    return {
        "entry_id": entry_id,
        "video": video,
        "video_sha256": receipt["output_sha256"],
        "receipt": receipt_path,
        "receipt_sha256": forward_runner.base.branch_manifest.file_sha256(receipt_path),
        "instruction_utf8_sha256": receipt["instruction_utf8_sha256"],
        "source_video_sha256": receipt["source_video_sha256"],
    }


def locate_new_cell(expanded_root: Path, source: str, seed: int) -> Path:
    candidates = [
        lane for lane in (expanded_root / "lanes").glob("lane*")
        if (lane / "entries" / f"{source}-s{seed}-noop").is_dir()
    ]
    if len(candidates) != 1:
        fail(f"new cell lane closure differs: {source}:{seed}")
    return candidates[0]


def action_family(manifest: Mapping[str, Any], source: str) -> str:
    state = manifest["source_states"].get(source)
    if state == {
        "initial_state": "stable_four_leg_stand",
        "terminal_state": "stable_seated_posture",
    }:
        return "dog stand→sit"
    if state == {
        "initial_state": "stable_one_knee_pose",
        "terminal_state": "fully_upright_stand",
    }:
        return "human one-knee→stand"
    fail(f"source state differs: {source}")


def build_html(rows: list[Mapping[str, Any]]) -> str:
    cards = []
    for row in rows:
        cards.append(
            f'''<article class="cell"><div><h2>{html.escape(row["family"])}</h2>
<p><code>{row["source_id"]}:{row["seed"]}</code> · {html.escape(row["origin"])}</p></div>
<section><h3>noop / exact source</h3><video controls muted loop src="{row["noop_path"]}"></video></section>
<section><h3>forward / frozen Bernini</h3><video controls muted loop src="{row["forward_path"]}"></video></section>
<aside><b>人工 verdict：</b> <span>待填 action / owner / scene / camera</span><br><small>只有全轨迹人工检查后，才可进入 v3 control 派生；仍不构成 training target。</small></aside></article>'''
        )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forward anchor fit12 review</title>
<style>:root{{--bg:#07101d;--p:#101d32;--c:#162640;--l:#2b4168;--t:#eef5ff;--m:#a5b4cb;--a:#67d9ff;--w:#ffc96c}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#173456,var(--bg) 48%);color:var(--t);font:15px/1.5 system-ui,sans-serif}}main{{width:min(1500px,96vw);margin:auto;padding:38px 0 80px}}h1{{font-size:42px;margin:.15em 0}}.lead{{max-width:1100px;font-size:18px;color:#d2ddef}}.gate{{border-left:5px solid var(--w);background:#2a2318;padding:15px 18px;border-radius:10px;margin:20px 0}}.cell{{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:14px;background:var(--p);border:1px solid var(--l);padding:16px;border-radius:15px;margin:16px 0}}section,aside{{background:var(--c);border:1px solid var(--l);border-radius:11px;padding:10px}}video{{width:100%;background:#02050a;border-radius:8px}}h2,h3{{margin:.2em 0 .5em}}h3{{color:var(--a);font-size:14px}}p,small,span{{color:var(--m)}}code{{color:#c4e8ff}}aside{{grid-column:1/-1}}@media(max-width:900px){{.cell{{grid-template-columns:1fr}}}}</style></head><body><main>
<p>2026-08-13 · prospective fit-only · self-generated anchors · no optimizer</p><h1>Fit12 forward anchor review</h1>
<p class="lead">6 个 source × 2 个 registered seed。每个 cell 只展示 exact noop 与 frozen-Bernini forward；没有 independently prompted reverse/incomplete/nuisance。</p>
<div class="gate"><b>先审 forward，再派生 controls。</b> 这 12 个 cell 是 decoded admission gate。失败的 forward 不进入 representation population；成功的 forward 仍只是 self-generated action anchor，不是 RGB target。</div>
{''.join(cards)}
</main></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-release", type=Path, required=True)
    parser.add_argument("--expanded-release", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    old = plain_directory(args.old_release, label="old release")
    expanded = plain_directory(args.expanded_release, label="expanded release")
    manifest_path = args.manifest.resolve(strict=True)
    if (
        manifest_path.is_symlink()
        or forward_runner.base.branch_manifest.file_sha256(manifest_path)
        != args.expected_manifest_sha256
    ):
        fail("manifest binding differs")
    manifest = read_json(manifest_path, label="manifest")
    forward_runner.base.branch_manifest.validate_manifest(manifest)
    output = create_output(args.output_root)
    rows = []
    for source, seed in ALL_CELLS:
        release = old if (source, seed) in OLD_CELLS else locate_new_cell(expanded, source, seed)
        origin = "initial smoke" if release == old else release.name
        values = {
            branch: entry(
                release, source, seed, branch,
                manifest_sha256=args.expected_manifest_sha256,
            )
            for branch in ("noop", "forward")
        }
        cell_root = output / f"{source}-s{seed}"
        cell_root.mkdir(mode=0o700)
        copied = {}
        for branch, value in values.items():
            destination = cell_root / f"{branch}.mp4"
            with value["video"].open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            if forward_runner.base.branch_manifest.file_sha256(destination) != value["video_sha256"]:
                fail("copied video SHA differs")
            copied[branch] = str(destination.relative_to(output))
        rows.append({
            "source_id": source, "seed": seed,
            "family": action_family(manifest, source), "origin": origin,
            "noop_path": copied["noop"], "forward_path": copied["forward"],
            "noop": {key: str(value) for key, value in values["noop"].items() if key != "video"},
            "forward": {key: str(value) for key, value in values["forward"].items() if key != "video"},
        })
    if len(rows) != 12:
        fail("review cell count differs")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": args.expected_manifest_sha256,
        "cells": rows,
        "cell_count": 12, "video_count": 24,
        "authority": dict(AUTHORITY),
    }
    receipt = {
        **unsigned,
        "receipt_digest": forward_runner.base.branch_manifest.object_sha256(unsigned),
    }
    forward_runner.base._write_create_only(output / "review.receipt.json", receipt)
    descriptor = os.open(output / "index.html", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(build_html(rows))
    print(json.dumps({"cells": 12, "videos": 24}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ForwardAnchorReviewError, forward_runner.ForwardAnchorShardError) as error:
        print(f"[forward-anchor-review] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

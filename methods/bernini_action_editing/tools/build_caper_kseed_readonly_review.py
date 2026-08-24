#!/usr/bin/env python3
"""Build a portable, read-only CAPER native K-seed review packet.

The experiment root is never modified.  The registered population order is
read from the sealed registry and checked against the completed population
receipt.  Every source and candidate MP4 is hash checked and probed as
exact81/25fps before being copied into a new, self-contained review directory.

Population completion is deliberately kept separate from semantic assessment:
this tool neither scores candidates nor selects a best seed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import quote


FRAME_COUNT = 81
FPS = 25
NATIVE_ARM = "native-source-video-only-v2v-endpoint"
REGISTRY_NAME = "sealed-caper-native-kseed-population-sit-v1.json"
MASTER_NAME = "fit-population-receipt.json"
REGISTRY_SCHEMA = "bernini-caper-native-kseed-population-sit-v1"
MASTER_SCHEMA = "bernini-caper-native-kseed-population-all8-receipt-v1"
ATTEMPT_SCHEMA = "bernini-caper-native-kseed-attempt-receipt-v1"
CELL_SCHEMA = "bernini-caper-native-kseed-cell-receipt-v1"
AUDIT_SCHEMA = "bernini-caper-kseed-portable-readonly-review-audit-v1"
SEMANTIC_STATUS = "UNASSESSED"
_CELL = re.compile(r"^(?P<phase>[a-z]+)-(?P<source>[0-9a-f]+)-s(?P<seed>[0-9]+)$")


class ReviewError(RuntimeError):
    """Raised when the registered population cannot be reviewed faithfully."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReviewError(f"not a plain file: {path}")
    return path.resolve(strict=True)


def _load_object(path: Path) -> dict[str, Any]:
    resolved = _plain_file(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReviewError(f"cannot decode JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReviewError(f"JSON root is not an object: {path}")
    return value


def _load_sealed(path: Path) -> tuple[dict[str, Any], str]:
    value = _load_object(path)
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha256(unsigned) != declared:
        raise ReviewError(f"receipt digest differs: {path}")
    return value, declared


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def _parse_cell(cell_id: str) -> tuple[str, str, int]:
    matched = _CELL.fullmatch(cell_id)
    if matched is None:
        raise ReviewError(f"invalid cell id: {cell_id!r}")
    return (
        matched.group("phase"),
        matched.group("source"),
        int(matched.group("seed")),
    )


def probe_exact81(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Require one exact81, 25fps video stream and return its dimensions."""

    video = _plain_file(path)
    command = (
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(video),
    )
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise ReviewError(f"cannot execute ffprobe for {video}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise ReviewError(f"ffprobe failed for {video}: {detail}")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        streams = payload["streams"]
        stream = streams[0]
    except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise ReviewError(f"ffprobe output differs for {video}") from error
    width = stream.get("width")
    height = stream.get("height")
    if (
        len(streams) != 1
        or type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or stream.get("nb_read_frames") != str(FRAME_COUNT)
        or stream.get("avg_frame_rate") != f"{FPS}/1"
    ):
        raise ReviewError(
            f"not exact81/25fps: {video} "
            f"(frames={stream.get('nb_read_frames')!r}, "
            f"fps={stream.get('avg_frame_rate')!r}, size={width!r}x{height!r})"
        )
    return {"frame_count": FRAME_COUNT, "fps": FPS, "width": width, "height": height}


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    source = _plain_file(source)
    _require(file_sha256(source) == expected_sha256, f"input SHA-256 differs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    _require(
        file_sha256(destination) == expected_sha256,
        f"portable copy SHA-256 differs: {destination}",
    )


def _validate_registry(
    path: Path, *, phase: str
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], list[int]]:
    registry = _load_object(path)
    _require(registry.get("schema_version") == REGISTRY_SCHEMA, "registry schema differs")
    design = registry.get("population_design", {}).get(phase)
    _require(isinstance(design, dict), f"registry phase is absent: {phase}")
    cells = design.get("cell_order")
    sources = registry.get("sources")
    seeds = design.get("seeds")
    source_ids = design.get("source_ids")
    _require(isinstance(cells, list) and all(isinstance(x, str) for x in cells), "cell order differs")
    _require(isinstance(seeds, list) and all(type(x) is int for x in seeds), "seed order differs")
    _require(isinstance(source_ids, list) and all(isinstance(x, str) for x in source_ids), "source order differs")
    _require(isinstance(sources, list), "source rows differ")
    _require(len(cells) == len(set(cells)), "duplicate registered cell")
    _require(design.get("expected_cell_count") == len(cells), "registered cell count differs")
    _require(design.get("cartesian_population_required") is True, "population is not Cartesian")
    _require(design.get("seed_filtering_or_best_of_k_authorized") is False, "best-of-K is authorized")
    expected_cells = [f"{phase}-{source_id}-s{seed}" for source_id in source_ids for seed in seeds]
    _require(cells == expected_cells, "cell order is not the registered source-major Cartesian order")
    indexed = {
        str(row.get("source_id")): row
        for row in sources
        if isinstance(row, dict) and row.get("split") == phase
    }
    _require(set(indexed) == set(source_ids), "phase source closure differs")
    ordered_sources = [indexed[source_id] for source_id in source_ids]
    return registry, list(cells), ordered_sources, list(seeds)


def _validate_master(
    path: Path, *, registry_path: Path, cells: list[str]
) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    master, digest = _load_sealed(path)
    _require(master.get("schema_version") == MASTER_SCHEMA, "population receipt schema differs")
    _require(master.get("registered_cell_order") == cells, "population receipt reordered cells")
    _require(master.get("registered_cell_count") == len(cells), "master registered count differs")
    _require(master.get("successful_cell_count") == len(cells), "master success count differs")
    _require(master.get("failed_cell_count") == 0 and master.get("failed_attempts") == [], "master contains failure")
    _require(master.get("population_complete") is True, "population is incomplete")
    _require(master.get("population_decision") == "PASS_COMPLETE", "population completion decision differs")
    _require(master.get("seed_filtering_or_best_of_k_authorized") is False, "master authorizes best-of-K")
    _require(master.get("retry_or_replacement_seed_authorized") is False, "master authorizes seed replacement")
    _require(master.get("partial_population_scientific_claim_authorized") is False, "master authorizes partial claims")
    _require(master.get("scientific_or_action_editing_claim_authorized") is not True, "master grants semantic authority")
    _require(master.get("training_performed") is False, "unexpected training claim")
    _require(master.get("optimizer_created") is False, "unexpected optimizer claim")
    _require(master.get("parameter_update") is False, "unexpected parameter update claim")
    _require(master.get("exact81") is True and master.get("fps") == FPS, "master exact81 contract differs")
    _require(master.get("registry_file_sha256") == file_sha256(registry_path), "master registry hash differs")
    children = master.get("children")
    _require(isinstance(children, list) and len(children) == len(cells), "master children differ")
    _require([row.get("cell_id") for row in children if isinstance(row, dict)] == cells, "master children reordered")
    return master, digest, {str(row["cell_id"]): row for row in children}


def _validate_candidate(
    root: Path,
    *,
    cell_id: str,
    phase: str,
    source_id: str,
    seed: int,
    master_child: Mapping[str, Any],
    ffprobe: str,
) -> dict[str, Any]:
    attempt_path = root / "attempts" / f"{cell_id}.json"
    attempt, attempt_digest = _load_sealed(attempt_path)
    _require(attempt.get("schema_version") == ATTEMPT_SCHEMA, f"attempt schema differs: {cell_id}")
    _require(attempt.get("cell_id") == cell_id, f"attempt cell differs: {cell_id}")
    _require(attempt.get("population_phase") == phase, f"attempt phase differs: {cell_id}")
    _require(attempt.get("source_id") == source_id, f"attempt source differs: {cell_id}")
    _require(attempt.get("seed") == seed, f"attempt seed differs: {cell_id}")
    _require(attempt.get("attempt_success") is True and attempt.get("process_exit_code") == 0, f"attempt failed: {cell_id}")
    _require(attempt.get("attempt_status") == "completed_success", f"attempt status differs: {cell_id}")
    _require(attempt.get("seed_discarded") is False, f"seed was discarded: {cell_id}")
    _require(attempt.get("retry_or_replacement_seed_authorized") is False, f"replacement seed authorized: {cell_id}")
    cell_root = root / cell_id
    receipt_path = cell_root / "receipt.json"
    receipt, receipt_digest = _load_sealed(receipt_path)
    receipt_file_sha = file_sha256(receipt_path)
    _require(attempt.get("child_receipt_file_sha256") == receipt_file_sha, f"attempt child file hash differs: {cell_id}")
    _require(attempt.get("child_receipt_digest") == receipt_digest, f"attempt child digest differs: {cell_id}")
    _require(master_child.get("attempt_receipt_sha256") == file_sha256(attempt_path), f"master attempt hash differs: {cell_id}")
    _require(master_child.get("child_receipt_file_sha256") == receipt_file_sha, f"master child file hash differs: {cell_id}")
    _require(master_child.get("child_receipt_digest") == receipt_digest, f"master child digest differs: {cell_id}")
    _require(receipt.get("schema_version") == CELL_SCHEMA, f"cell schema differs: {cell_id}")
    _require(receipt.get("cell_id") == cell_id, f"cell receipt id differs: {cell_id}")
    _require(receipt.get("population_phase") == phase, f"cell receipt phase differs: {cell_id}")
    _require(receipt.get("input", {}).get("source_id") == source_id, f"cell source differs: {cell_id}")
    sampling = receipt.get("sampling", {})
    _require(sampling.get("seed") == seed, f"cell seed differs: {cell_id}")
    _require(sampling.get("frame_count") == FRAME_COUNT and sampling.get("fps") == FPS, f"cell exact81 differs: {cell_id}")
    _require(receipt.get("seed_filtering_or_best_of_k_authorized") is False, f"cell authorizes best-of-K: {cell_id}")
    _require(receipt.get("scientific_or_action_editing_claim_authorized") is False, f"cell grants semantic authority: {cell_id}")
    _require(receipt.get("training_performed") is False, f"cell claims training: {cell_id}")
    _require(receipt.get("optimizer_created") is False, f"cell claims optimizer: {cell_id}")
    _require(receipt.get("parameter_update") is False, f"cell claims update: {cell_id}")
    output = receipt.get("outputs", {}).get(NATIVE_ARM)
    _require(isinstance(output, dict), f"cell output is absent: {cell_id}")
    video_path = cell_root / f"{NATIVE_ARM}.mp4"
    video_sha = file_sha256(_plain_file(video_path))
    _require(output.get("sha256") == video_sha, f"cell video hash differs: {cell_id}")
    _require(master_child.get("mp4_sha256") == video_sha, f"master video hash differs: {cell_id}")
    _require(output.get("frame_count") == FRAME_COUNT and output.get("fps") == FPS, f"output exact81 differs: {cell_id}")
    probe = probe_exact81(video_path, ffprobe=ffprobe)
    return {
        "cell_id": cell_id,
        "source_id": source_id,
        "seed": seed,
        "semantic_status": SEMANTIC_STATUS,
        "input_video_path": str(video_path),
        "video_sha256": video_sha,
        "probe": probe,
        "attempt_receipt_path": str(attempt_path),
        "attempt_receipt_sha256": file_sha256(attempt_path),
        "attempt_receipt_digest": attempt_digest,
        "cell_receipt_path": str(receipt_path),
        "cell_receipt_sha256": receipt_file_sha,
        "cell_receipt_digest": receipt_digest,
    }


def _url(path: str) -> str:
    return quote(path, safe="/:._-")


def _render_html(audit: Mapping[str, Any], *, job_id: str) -> str:
    sections: list[str] = []
    for row_index, row in enumerate(audit["rows"], start=1):
        source = row["source"]
        cards = [
            f'''<article class="card source-card"><header><span class="eyebrow">SOURCE</span><h3>Original source</h3><p>Registered input · exact81 / 25 fps</p></header><video data-group="row-{row_index}" data-leader controls muted playsinline preload="metadata" src="{_url(source['portable_video'])}"></video></article>'''
        ]
        for candidate_index, candidate in enumerate(row["candidates"], start=1):
            cards.append(
                f'''<article class="card"><header><span class="eyebrow">FIXED CANDIDATE {candidate_index:02d}</span><h3>Seed {candidate['seed']}</h3><p>Registered order #{candidate_index} · semantic status: <strong>UNASSESSED</strong></p></header><video data-group="row-{row_index}" controls muted playsinline preload="metadata" src="{_url(candidate['portable_video'])}"></video></article>'''
            )
        caption = html.escape(str(row.get("target_action_caption", "")))
        sections.append(
            f'''<section class="sample" id="source-{html.escape(row['source_id'])}"><div class="row-head"><div><span class="row-number">{row_index:02d}</span><h2>{html.escape(str(row.get('actor_kind', 'actor')).title())} · {html.escape(str(row.get('identity_id', row['source_id'])))}</h2><p class="meta">source {html.escape(row['source_id'])} · scene {html.escape(str(row.get('scene_id', 'unknown')))}</p></div><div class="row-controls"><button class="play" data-play="row-{row_index}">Play together from 0</button><button data-pause="row-{row_index}">Pause row</button><button data-reset="row-{row_index}">Reset row</button></div></div><details><summary>Registered action instruction</summary><p>{caption}</p></details><div class="grid">{''.join(cards)}</div></section>'''
        )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAPER fixed K-seed review · AUH {html.escape(job_id)}</title>
<style>
:root{{--ink:#17201c;--paper:#f4f1e9;--panel:#fffdf7;--line:#cfc9bb;--muted:#68706a;--accent:#1d6752;--warn:#8a421f;--warnbg:#fff0df}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:linear-gradient(180deg,#e9eee7 0,#f4f1e9 24rem);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1840px;margin:auto;padding:28px}}.hero,.sample{{background:rgba(255,253,247,.94);border:1px solid var(--line);border-radius:18px;box-shadow:0 12px 34px rgba(30,43,34,.08)}}.hero{{padding:26px;margin-bottom:18px}}.kicker,.eyebrow{{color:var(--accent);font-size:11px;font-weight:800;letter-spacing:.11em;text-transform:uppercase}}h1{{font-size:clamp(28px,4vw,52px);line-height:1.02;letter-spacing:-.035em;margin:7px 0 12px}}h2,h3,p{{margin-top:0}}h2{{display:inline;font-size:21px}}h3{{font-size:15px;margin:4px 0}}.lede{{max-width:980px;color:var(--muted);font-size:16px}}.warning{{max-width:1200px;padding:14px 16px;border:1px solid #e0a575;border-radius:11px;background:var(--warnbg);color:var(--warn)}}.facts{{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}}.fact{{padding:7px 10px;border:1px solid var(--line);border-radius:999px;background:#f8f6ef}}.fact strong{{color:var(--accent)}}.sample{{padding:17px;margin-bottom:18px}}.row-head{{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}}.row-number{{display:inline-block;margin-right:10px;color:#fff;background:var(--accent);border-radius:999px;padding:3px 8px;font-weight:800}}.meta{{margin:5px 0;color:var(--muted)}}.row-controls{{display:flex;flex-wrap:wrap;gap:7px}}button{{border:1px solid #a9afa8;border-radius:8px;background:#f5f2e9;color:var(--ink);padding:8px 11px;font-weight:650;cursor:pointer}}button:hover{{border-color:var(--accent)}}button.play{{background:var(--accent);border-color:var(--accent);color:#fff}}details{{margin:11px 0 14px;padding:9px 11px;border-radius:9px;background:#f2f3ed;color:var(--muted)}}details p{{margin:8px 0 0}}summary{{cursor:pointer;color:var(--ink);font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(5,minmax(210px,1fr));gap:10px}}.card{{overflow:hidden;border:1px solid var(--line);border-radius:12px;background:#fbfaf5}}.source-card{{border-color:#7ea899;background:#f0f7f3}}.card header{{padding:10px;min-height:104px}}.card header p{{color:var(--muted);font-size:12px;margin-bottom:0}}video{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#0c0e0d}}footer{{padding:8px 2px 30px;color:var(--muted)}}a{{color:var(--accent)}}@media(max-width:1280px){{.grid{{grid-template-columns:repeat(3,minmax(210px,1fr))}}}}@media(max-width:760px){{main{{padding:10px}}.grid{{grid-template-columns:1fr}}.row-head{{display:block}}.row-controls{{margin:10px 0}}.card header{{min-height:0}}}}
</style></head><body><main>
<section class="hero"><span class="kicker">AUH job {html.escape(job_id)} · read-only population review</span><h1>Four sources. Four fixed seeds each.</h1><p class="lede">Native Bernini V2V proposals are shown in the exact registered source-major and seed order. Every row has one source and all four candidates; no candidate is ranked, promoted, hidden, or substituted.</p><p class="warning"><strong>Population completion is not a semantic pass.</strong> It only confirms that all 16 registered renders completed and passed receipt/hash/exact81 checks. Action correctness, source identity, camera stability, consistency, and sharpness remain <strong>UNASSESSED</strong>.</p><div class="facts"><span class="fact"><strong>4</strong> source rows</span><span class="fact"><strong>16</strong> fixed candidates</span><span class="fact"><strong>81</strong> frames · 25 fps</span><span class="fact">best-of-K selection: <strong>disabled</strong></span><span class="fact">semantic labels: <strong>none assigned</strong></span><a class="fact" href="review-audit.json">machine-readable audit</a></div></section>
{''.join(sections)}
<footer>Portable review packet · media and evidence use relative paths · completion scope is engineering-only · no best-of-K selection.</footer>
</main><script>
const byGroup=g=>[...document.querySelectorAll(`video[data-group="${{g}}"]`)];
const ready=v=>v.readyState>=1?Promise.resolve():new Promise(resolve=>{{const done=()=>resolve();v.addEventListener('loadedmetadata',done,{{once:true}});v.addEventListener('error',done,{{once:true}})}});
async function setTime(v,t){{await ready(v);try{{v.currentTime=Math.min(t,Number.isFinite(v.duration)?Math.max(0,v.duration-.001):t)}}catch(_e){{}}}}
function pause(g){{byGroup(g).forEach(v=>v.pause())}}
async function reset(g){{const videos=byGroup(g);videos.forEach(v=>v.pause());await Promise.all(videos.map(v=>setTime(v,0)))}}
async function play(g){{document.querySelectorAll('video').forEach(v=>v.pause());const videos=byGroup(g);await Promise.all(videos.map(v=>setTime(v,0)));await Promise.allSettled(videos.map(v=>v.play()))}}
document.querySelectorAll('[data-play]').forEach(b=>b.onclick=()=>play(b.dataset.play));
document.querySelectorAll('[data-pause]').forEach(b=>b.onclick=()=>pause(b.dataset.pause));
document.querySelectorAll('[data-reset]').forEach(b=>b.onclick=()=>reset(b.dataset.reset));
document.querySelectorAll('video[data-leader]').forEach(leader=>leader.addEventListener('timeupdate',()=>{{if(leader.paused)return;byGroup(leader.dataset.group).forEach(v=>{{if(v!==leader&&!v.seeking&&Math.abs(v.currentTime-leader.currentTime)>.07)v.currentTime=leader.currentTime}})}}));
</script></body></html>'''


def build_review(
    *,
    input_root: Path,
    output_root: Path,
    job_id: str,
    phase: str = "fit",
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    """Validate one complete native population and publish a portable review."""

    input_root = input_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    _require(input_root.is_dir() and not input_root.is_symlink(), "input root is not a plain directory")
    _require(not output_root.exists(), f"output already exists: {output_root}")
    _require(output_root != input_root and input_root not in output_root.parents, "output must not be inside input")
    registry_path = input_root / REGISTRY_NAME
    master_path = input_root / MASTER_NAME
    registry, cells, source_rows, seeds = _validate_registry(registry_path, phase=phase)
    master, master_digest, master_children = _validate_master(
        master_path, registry_path=registry_path, cells=cells
    )
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(source_rows, start=1):
        source_id = str(source["source_id"])
        source_path = input_root / f"source-{source_id}.mp4"
        source_sha = str(source.get("source_video_sha256"))
        _require(len(source_sha) == 64, f"source SHA-256 is absent: {source_id}")
        _require(file_sha256(_plain_file(source_path)) == source_sha, f"source video hash differs: {source_id}")
        source_probe = probe_exact81(source_path, ffprobe=ffprobe)
        candidates = []
        for seed in seeds:
            cell_id = f"{phase}-{source_id}-s{seed}"
            candidate = _validate_candidate(
                input_root,
                cell_id=cell_id,
                phase=phase,
                source_id=source_id,
                seed=seed,
                master_child=master_children[cell_id],
                ffprobe=ffprobe,
            )
            candidate["registered_candidate_index"] = len(candidates) + 1
            candidate["portable_video"] = f"media/{source_index:02d}-{source_id}/candidate-{len(candidates)+1:02d}-seed-{seed}.mp4"
            candidate["portable_attempt_receipt"] = f"evidence/attempts/{cell_id}.json"
            candidate["portable_cell_receipt"] = f"evidence/cells/{cell_id}.json"
            candidates.append(candidate)
        rows.append(
            {
                "registered_source_index": source_index,
                "source_id": source_id,
                "actor_kind": source.get("actor_kind"),
                "identity_id": source.get("identity_id"),
                "scene_id": source.get("scene_id"),
                "target_action_caption": source.get("target_action_caption"),
                "source": {
                    "input_video_path": str(source_path),
                    "video_sha256": source_sha,
                    "probe": source_probe,
                    "portable_video": f"media/{source_index:02d}-{source_id}/source.mp4",
                },
                "candidates": candidates,
            }
        )
    _require([candidate["cell_id"] for row in rows for candidate in row["candidates"]] == cells, "render rows reordered registered cells")
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "job_id": str(job_id),
        "phase": phase,
        "population_complete": bool(master["population_complete"]),
        "population_completion_scope": "ENGINEERING_CLOSURE_ONLY",
        "semantic_status": SEMANTIC_STATUS,
        "semantic_pass_assigned": False,
        "scientific_or_action_editing_claim_authorized": False,
        "seed_filtering_or_best_of_k": False,
        "candidate_ranking_or_selection_performed": False,
        "source_count": len(rows),
        "candidate_count": sum(len(row["candidates"]) for row in rows),
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "registered_cell_order": cells,
        "registry": {
            "input_path": str(registry_path),
            "sha256": file_sha256(registry_path),
            "portable_path": f"evidence/{REGISTRY_NAME}",
        },
        "population_receipt": {
            "input_path": str(master_path),
            "sha256": file_sha256(master_path),
            "receipt_digest": master_digest,
            "portable_path": f"evidence/{MASTER_NAME}",
        },
        "rows": rows,
    }
    _require(audit["source_count"] == 4, "review requires exactly four registered sources")
    _require(audit["candidate_count"] == 16, "review requires exactly sixteen candidates")
    audit["audit_digest"] = object_sha256(audit)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        for row in rows:
            source = row["source"]
            _copy_verified(
                Path(source["input_video_path"]),
                staging / source["portable_video"],
                source["video_sha256"],
            )
            for candidate in row["candidates"]:
                _copy_verified(
                    Path(candidate["input_video_path"]),
                    staging / candidate["portable_video"],
                    candidate["video_sha256"],
                )
                _copy_verified(
                    Path(candidate["attempt_receipt_path"]),
                    staging / candidate["portable_attempt_receipt"],
                    candidate["attempt_receipt_sha256"],
                )
                _copy_verified(
                    Path(candidate["cell_receipt_path"]),
                    staging / candidate["portable_cell_receipt"],
                    candidate["cell_receipt_sha256"],
                )
        _copy_verified(registry_path, staging / "evidence" / REGISTRY_NAME, audit["registry"]["sha256"])
        _copy_verified(master_path, staging / "evidence" / MASTER_NAME, audit["population_receipt"]["sha256"])
        (staging / "review-audit.json").write_bytes(canonical_json_bytes(audit) + b"\n")
        (staging / "index.html").write_text(_render_html(audit, job_id=str(job_id)), encoding="utf-8")
        os.rename(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--phase", default="fit", choices=("fit", "lockbox"))
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = build_review(
        input_root=args.input_root,
        output_root=args.output_root,
        job_id=args.job_id,
        phase=args.phase,
        ffprobe=args.ffprobe,
    )
    print(
        canonical_json_bytes(
            {
                "audit_digest": audit["audit_digest"],
                "candidate_count": audit["candidate_count"],
                "population_complete": audit["population_complete"],
                "semantic_status": audit["semantic_status"],
                "source_count": audit["source_count"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA",
    "CELL_SCHEMA",
    "FRAME_COUNT",
    "FPS",
    "MASTER_SCHEMA",
    "NATIVE_ARM",
    "REGISTRY_SCHEMA",
    "ReviewError",
    "SEMANTIC_STATUS",
    "build_review",
    "canonical_json_bytes",
    "file_sha256",
    "main",
    "object_sha256",
    "probe_exact81",
]

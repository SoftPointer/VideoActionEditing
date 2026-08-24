#!/usr/bin/env python3
"""Build the synchronized MEV target/self-generated flow calibration review.

This publisher is intentionally fail-closed.  It does not create a review when
one of the eight videos is missing, a calibration receipt is inconsistent, or
the destination already contains files.  The four carrier columns are outputs
of a fixed, historically trained step-32 carrier; this program never labels
them as newly trained ``Ours`` results.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "mev-target-selfgen-flow-calibration-review-v1"
EXPECTED_MANIFEST_SCHEMA = "mev-target-selfgen-flow-calibration-manifest-v1"
EXPECTED_RECEIPT_SCHEMA = "bernini-mev-fixed-carrier-calibration-receipt-v1"
EXPECTED_HISTORICAL_STEPS = 32
EXPECTED_CURRENT_STEPS = 0
CLAIM_BOUNDARY = (
    "fixed_historically_trained_carrier_zero_new_update_admission_probe_"
    "not_new_ours_training"
)
ROUTES = ("real_forward", "temporal_shuffle", "reverse", "self_generated")
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

COLUMNS = (
    ("source", "Source"),
    ("real_target_reference", "Real target reference"),
    ("self_generated_anchor", "Self-generated anchor"),
    ("frozen", "Frozen"),
    ("real_forward_carrier", "real-forward carrier"),
    ("temporal_shuffle_carrier", "shuffle carrier"),
    ("reverse_carrier", "reverse carrier"),
    ("self_generated_carrier", "selfgen carrier"),
)


class BuildError(RuntimeError):
    """Raised when the review cannot be published without weakening its contract."""


@dataclass(frozen=True)
class Asset:
    role: str
    source: Path
    expected_sha256: str
    published_name: str
    receipt_source: Path | None = None
    receipt: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CaseInputs:
    case_id: str
    split: str
    action_family: str
    instruction: str
    seed: int
    assets: tuple[Asset, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise BuildError(f"{label} is missing or not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise BuildError(f"{label} must contain a JSON object: {path}")
    return value


def _sha_field(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise BuildError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _existing_media(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise BuildError(f"{label} does not resolve to a file: {path}: {error}") from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise BuildError(f"{label} is missing, not a file, or empty: {resolved}")
    return resolved


def _manifest_asset(
    case: Mapping[str, Any],
    key: str,
    *,
    role: str,
    published_name: str,
    manifest_dir: Path,
) -> Asset:
    spec = case.get(key)
    if not isinstance(spec, dict):
        raise BuildError(f"case {case.get('case_id')!r} is missing object {key!r}")
    raw_path = spec.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BuildError(f"case {case.get('case_id')!r} {key}.path is invalid")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return Asset(
        role=role,
        source=_existing_media(path, label=f"case {case.get('case_id')} {key}"),
        expected_sha256=_sha_field(
            spec.get("sha256"), label=f"case {case.get('case_id')} {key}.sha256"
        ),
        published_name=published_name,
    )


def _exact_int(value: Any, expected: int, *, label: str) -> None:
    if type(value) is not int or value != expected:  # bool must not pass as int
        raise BuildError(f"{label} must be exactly {expected}, got {value!r}")


def _validate_receipt(
    receipt_path: Path,
    *,
    output: Path,
    route: str,
    case_id: str,
    experiment_id: str,
) -> Mapping[str, Any]:
    receipt = _json_file(receipt_path, label=f"{case_id}/{route} calibration receipt")
    exact = {
        "schema_version": EXPECTED_RECEIPT_SCHEMA,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "route_kind": route,
        "anchor_kind": route,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            raise BuildError(
                f"{case_id}/{route} receipt {key} must be {expected!r}, "
                f"got {receipt.get(key)!r}"
            )
    _exact_int(
        receipt.get("historical_carrier_global_step"),
        EXPECTED_HISTORICAL_STEPS,
        label=f"{case_id}/{route} historical_carrier_global_step",
    )
    _exact_int(
        receipt.get("current_experiment_optimization_steps"),
        EXPECTED_CURRENT_STEPS,
        label=f"{case_id}/{route} current_experiment_optimization_steps",
    )
    if receipt.get("parameter_updates_in_current_experiment") is not False:
        raise BuildError(
            f"{case_id}/{route} receipt must set "
            "parameter_updates_in_current_experiment=false"
        )

    recorded_output = receipt.get("output")
    if not isinstance(recorded_output, str) or not recorded_output:
        raise BuildError(f"{case_id}/{route} receipt output is invalid")
    try:
        if Path(recorded_output).expanduser().resolve(strict=True) != output:
            raise BuildError(
                f"{case_id}/{route} receipt output does not identify {output}"
            )
    except (OSError, RuntimeError) as error:
        raise BuildError(f"{case_id}/{route} receipt output cannot be resolved: {error}") from error
    _sha_field(receipt.get("output_sha256"), label=f"{case_id}/{route} output_sha256")
    _sha_field(
        receipt.get("flow_bundle_sha256"), label=f"{case_id}/{route} flow_bundle_sha256"
    )
    _sha_field(
        receipt.get("carrier_receipt_sha256"),
        label=f"{case_id}/{route} carrier_receipt_sha256",
    )

    firewall = receipt.get("information_firewall")
    if not isinstance(firewall, dict):
        raise BuildError(f"{case_id}/{route} receipt information_firewall is missing")
    expected_firewall = {
        "target_video_accessed_by_extractor": route != "self_generated",
        "target_video_accessed_by_trainer": False,
        "target_video_accessed_by_renderer": False,
        "target_rgb_or_vae_target_used": False,
        "anchor_role": "detached_dense_flow_representation_only",
    }
    for key, expected in expected_firewall.items():
        if firewall.get(key) != expected:
            raise BuildError(
                f"{case_id}/{route} information_firewall.{key} must be "
                f"{expected!r}, got {firewall.get(key)!r}"
            )
    return receipt


def _route_asset(
    inference_root: Path,
    *,
    case_id: str,
    route: str,
    experiment_id: str,
) -> Asset:
    route_dir = inference_root / case_id / route
    output = _existing_media(route_dir / "output.mp4", label=f"{case_id}/{route} output")
    receipt_path = (route_dir / "calibration_receipt.json").resolve()
    receipt = _validate_receipt(
        receipt_path,
        output=output,
        route=route,
        case_id=case_id,
        experiment_id=experiment_id,
    )
    role = f"{route}_carrier"
    return Asset(
        role=role,
        source=output,
        expected_sha256=str(receipt["output_sha256"]),
        published_name=f"{role}.mp4",
        receipt_source=receipt_path,
        receipt=receipt,
    )


def _validate_inputs(manifest_path: Path, inference_root: Path) -> tuple[Mapping[str, Any], list[CaseInputs]]:
    manifest = _json_file(manifest_path, label="experiment manifest")
    if manifest.get("schema_version") != EXPECTED_MANIFEST_SCHEMA:
        raise BuildError(
            f"manifest schema_version must be {EXPECTED_MANIFEST_SCHEMA!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise BuildError("manifest experiment_id is missing")
    _exact_int(
        manifest.get("historical_carrier_global_step"),
        EXPECTED_HISTORICAL_STEPS,
        label="manifest historical_carrier_global_step",
    )
    _exact_int(
        manifest.get("current_experiment_optimization_steps"),
        EXPECTED_CURRENT_STEPS,
        label="manifest current_experiment_optimization_steps",
    )
    if tuple(manifest.get("flow_roles", ())) != ROUTES:
        raise BuildError(f"manifest flow_roles must be exactly {list(ROUTES)!r}")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise BuildError("manifest cases must be a non-empty list")

    seen: set[str] = set()
    checked: list[CaseInputs] = []
    for index, value in enumerate(raw_cases):
        if not isinstance(value, dict):
            raise BuildError(f"manifest cases[{index}] must be an object")
        case_id = value.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None:
            raise BuildError(f"manifest cases[{index}].case_id is unsafe or invalid")
        if case_id in seen:
            raise BuildError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        instruction = value.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise BuildError(f"case {case_id} instruction is missing")
        seed = value.get("seed")
        if type(seed) is not int:
            raise BuildError(f"case {case_id} seed must be an integer")
        split = value.get("split")
        action_family = value.get("action_family")
        if not isinstance(split, str) or not split:
            raise BuildError(f"case {case_id} split is missing")
        if not isinstance(action_family, str) or not action_family:
            raise BuildError(f"case {case_id} action_family is missing")

        assets = [
            _manifest_asset(
                value,
                "source",
                role="source",
                published_name="source.mp4",
                manifest_dir=manifest_path.parent,
            ),
            _manifest_asset(
                value,
                "real_forward",
                role="real_target_reference",
                published_name="real_target_reference.mp4",
                manifest_dir=manifest_path.parent,
            ),
            _manifest_asset(
                value,
                "self_generated",
                role="self_generated_anchor",
                published_name="self_generated_anchor.mp4",
                manifest_dir=manifest_path.parent,
            ),
            _manifest_asset(
                value,
                "frozen",
                role="frozen",
                published_name="frozen.mp4",
                manifest_dir=manifest_path.parent,
            ),
        ]
        assets.extend(
            _route_asset(
                inference_root,
                case_id=case_id,
                route=route,
                experiment_id=experiment_id,
            )
            for route in ROUTES
        )
        if tuple(asset.role for asset in assets) != tuple(key for key, _ in COLUMNS):
            raise AssertionError("internal review column order drifted")
        checked.append(
            CaseInputs(
                case_id=case_id,
                split=split,
                action_family=action_family,
                instruction=instruction.strip(),
                seed=seed,
                assets=tuple(assets),
            )
        )
    return manifest, checked


def _publish_media(source: Asset, target: Path, *, copy_mode: str) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise BuildError(f"internal create-only violation: {target}")
    if copy_mode == "copy":
        shutil.copy2(source.source, target)
    elif copy_mode == "symlink":
        os.symlink(source.source, target)
    else:  # guarded by both argparse and build()
        raise BuildError(f"unsupported copy mode: {copy_mode}")
    digest = _sha256(target)
    if digest != source.expected_sha256:
        raise BuildError(
            f"SHA-256 mismatch for {source.role}: expected {source.expected_sha256}, got {digest}"
        )
    return digest, target.stat().st_size


def _controls(scope: str, duration: float, *, global_controls: bool = False) -> str:
    escaped_scope = html.escape(scope, quote=True)
    label = "All cases" if global_controls else "This case"
    return f"""
    <div class="sync-controls {'global-controls' if global_controls else 'case-controls'}"
         data-controls-scope="{escaped_scope}">
      <strong>{label}</strong>
      <button type="button" data-command="play" data-scope="{escaped_scope}">Play</button>
      <button type="button" data-command="pause" data-scope="{escaped_scope}">Pause</button>
      <button type="button" data-command="restart" data-scope="{escaped_scope}">Restart</button>
      <label>Speed
        <select data-rate="{escaped_scope}">
          <option value="0.5">0.5×</option>
          <option value="0.75">0.75×</option>
          <option value="1" selected>1×</option>
          <option value="1.25">1.25×</option>
          <option value="1.5">1.5×</option>
          <option value="2">2×</option>
        </select>
      </label>
      <label class="seek-label">Seek
        <input type="range" min="0" max="{duration:.3f}" step="0.01" value="0"
               data-seek="{escaped_scope}">
      </label>
      <output data-time="{escaped_scope}">0.00 / {duration:.2f} s</output>
    </div>"""


def _render_html(manifest: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> str:
    frame_count = manifest.get("frame_count", 81)
    fps = manifest.get("fps", 25)
    duration = float(frame_count) / float(fps) if fps else 3.24
    headers = "".join(f"<div class=\"column-heading\">{html.escape(label)}</div>" for _, label in COLUMNS)
    case_sections: list[str] = []
    for case in cases:
        case_id = str(case["case_id"])
        escaped_id = html.escape(case_id, quote=True)
        cards = []
        for role, label in COLUMNS:
            media = case["media"][role]
            src = html.escape(str(media["published_path"]), quote=True)
            cards.append(
                f"""<article class="video-card" data-role="{html.escape(role, quote=True)}">
                  <h3>{html.escape(label)}</h3>
                  <video controls muted loop playsinline preload="metadata"
                         data-case-id="{escaped_id}" data-role="{html.escape(role, quote=True)}"
                         src="{src}"></video>
                </article>"""
            )
        case_sections.append(
            f"""<section class="case" id="case-{escaped_id}" data-case-id="{escaped_id}">
              <div class="case-title">
                <h2>{escaped_id} <span>{html.escape(str(case['split']))}</span></h2>
                <p><b>Action family:</b> {html.escape(str(case['action_family']))}
                   · <b>Seed:</b> {case['seed']}</p>
                <p><b>Instruction:</b> {html.escape(str(case['instruction']))}</p>
              </div>
              {_controls(case_id, duration)}
              <div class="comparison-grid">{''.join(cards)}</div>
            </section>"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MEV target vs self-generated flow calibration</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0a0d12; --card:#121722; --line:#30394b;
      --text:#f3f6fb; --muted:#aeb8ca; --warn:#ffd166; --accent:#67d4ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }}
    header {{ position:sticky; top:0; z-index:20; padding:16px 22px; background:#090c11f2;
      border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }}
    h1 {{ margin:0 0 8px; font-size:22px; }}
    .warning {{ margin:10px 0; padding:12px 15px; color:#17120a; background:var(--warn);
      border-radius:8px; font-weight:800; font-size:15px; }}
    .subtitle {{ color:var(--muted); max-width:1100px; }}
    main {{ padding:16px 22px 48px; }}
    button,select,input {{ accent-color:var(--accent); }}
    button,select {{ color:var(--text); background:#202839; border:1px solid #4b5870;
      border-radius:6px; padding:6px 10px; }}
    button:hover {{ border-color:var(--accent); cursor:pointer; }}
    .sync-controls {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
    .sync-controls label {{ display:flex; align-items:center; gap:6px; }}
    .seek-label {{ flex:1 1 260px; }}
    .seek-label input {{ width:100%; }}
    .global-controls {{ margin-top:10px; }}
    .column-headings,.comparison-grid {{ display:grid; grid-template-columns:repeat(8,minmax(260px,1fr));
      gap:10px; min-width:2160px; }}
    .column-headings {{ margin:18px 0 4px; overflow:hidden; color:var(--accent); font-weight:750; }}
    .column-heading {{ padding:0 8px; }}
    .case {{ margin:0 0 24px; padding:15px; border:1px solid var(--line); border-radius:12px;
      background:var(--card); overflow-x:auto; }}
    .case-title h2 {{ margin:0; font-size:18px; }}
    .case-title h2 span {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    .case-title p {{ margin:5px 0; max-width:1100px; }}
    .case-controls {{ position:sticky; left:0; max-width:1100px; margin:11px 0; padding:9px;
      background:#0b1019; border:1px solid var(--line); border-radius:8px; z-index:3; }}
    .video-card {{ min-width:0; border:1px solid var(--line); border-radius:8px; overflow:hidden;
      background:#080b10; }}
    .video-card h3 {{ margin:0; padding:8px 10px; font-size:13px; color:var(--muted); }}
    video {{ display:block; width:100%; background:#000; aspect-ratio:16/9; }}
    footer {{ color:var(--muted); padding:20px 22px 36px; }}
  </style>
</head>
<body>
  <header>
    <h1>MEV target / self-generated flow calibration</h1>
    <div class="warning">FIXED HISTORICAL STEP-32 CARRIER · CURRENT OPTIMIZATION STEPS = 0
      · THESE ARE NOT TRAINED “OURS” RESULTS</div>
    <div class="subtitle">Admission probe only. Real-target RGB is shown as a human reference;
      carrier inputs are detached dense-flow representations. All eight videos in a case use
      the same row controls and are continuously drift-corrected.</div>
    {_controls('all', duration, global_controls=True)}
  </header>
  <main>
    <div class="column-headings">{headers}</div>
    {''.join(case_sections)}
  </main>
  <footer>Provenance: <a href="review_manifest.json">review_manifest.json</a> ·
    receipts are under <code>receipts/&lt;case&gt;/&lt;route&gt;/</code>.</footer>
  <script>
  (() => {{
    'use strict';
    const videos = [...document.querySelectorAll('video[data-case-id]')];
    const duration = {duration:.6f};
    let activeScope = null;
    let activeVideos = [];
    let leader = null;
    const playGuard = new WeakSet(), pauseGuard = new WeakSet();
    const seekGuard = new WeakSet(), rateGuard = new WeakSet();
    const guardBriefly = (guard, video) => {{
      guard.add(video); window.setTimeout(() => guard.delete(video), 150);
    }};
    const scopeVideos = scope => scope === 'all'
      ? videos : videos.filter(video => video.dataset.caseId === scope);
    const scopeControls = scope => ({{
      seek: document.querySelector(`[data-seek="${{CSS.escape(scope)}}"]`),
      rate: document.querySelector(`[data-rate="${{CSS.escape(scope)}}"]`),
      time: document.querySelector(`[data-time="${{CSS.escape(scope)}}"]`)
    }});
    const ready = video => video.readyState >= 1 ? Promise.resolve() : new Promise(resolve => {{
      const done = () => resolve();
      video.addEventListener('loadedmetadata', done, {{once:true}});
      video.addEventListener('error', done, {{once:true}});
    }});
    const clamp = value => Math.max(0, Math.min(duration, Number(value) || 0));
    function updateControls(scope, time) {{
      const controls = scopeControls(scope); const value = clamp(time);
      if (controls.seek) controls.seek.value = String(value);
      if (controls.time) controls.time.textContent = `${{value.toFixed(2)}} / ${{duration.toFixed(2)}} s`;
    }}
    async function seekScope(scope, time, origin = null) {{
      const value = clamp(time);
      await Promise.all(scopeVideos(scope).map(async video => {{
        if (video === origin) return;
        await ready(video);
        const upper = Number.isFinite(video.duration) ? Math.max(0, video.duration - 0.001) : value;
        try {{ guardBriefly(seekGuard, video); video.currentTime = Math.min(value, upper); }}
        catch (_error) {{ seekGuard.delete(video); }}
      }}));
      updateControls(scope, value);
      if (scope !== 'all' && activeScope === 'all') updateControls('all', value);
    }}
    function setRate(scope, rate, origin = null) {{
      const value = Number(rate) || 1;
      scopeVideos(scope).forEach(video => {{
        if (video !== origin) {{ guardBriefly(rateGuard, video); video.playbackRate = value; }}
      }});
      const controls = scopeControls(scope); if (controls.rate) controls.rate.value = String(value);
    }}
    function pauseScope(scope) {{
      scopeVideos(scope).forEach(video => {{ guardBriefly(pauseGuard, video); video.pause(); }});
      if (scope === 'all' || activeScope === scope) {{ activeScope = null; activeVideos = []; leader = null; }}
    }}
    async function playScope(scope) {{
      pauseScope('all');
      const selected = scopeVideos(scope); const controls = scopeControls(scope);
      await seekScope(scope, controls.seek ? controls.seek.value : 0);
      setRate(scope, controls.rate ? controls.rate.value : 1);
      activeScope = scope; activeVideos = selected; leader = selected[0] || null;
      await Promise.allSettled(selected.map(video => {{
        guardBriefly(playGuard, video); return video.play();
      }}));
    }}
    async function restartScope(scope) {{
      pauseScope(scope); await seekScope(scope, 0);
    }}
    document.querySelectorAll('[data-command]').forEach(button => {{
      button.addEventListener('click', () => {{
        const scope = button.dataset.scope, command = button.dataset.command;
        if (command === 'play') void playScope(scope);
        else if (command === 'pause') pauseScope(scope);
        else void restartScope(scope);
      }});
    }});
    document.querySelectorAll('[data-seek]').forEach(input => {{
      input.addEventListener('input', () => {{ pauseScope(input.dataset.seek); void seekScope(input.dataset.seek, input.value); }});
    }});
    document.querySelectorAll('[data-rate]').forEach(select => {{
      select.addEventListener('change', () => setRate(select.dataset.rate, select.value));
    }});
    videos.forEach(video => {{
      video.addEventListener('play', () => {{ if (!playGuard.has(video)) void playScope(video.dataset.caseId); }});
      video.addEventListener('pause', () => {{ if (!pauseGuard.has(video) && activeScope === video.dataset.caseId) pauseScope(video.dataset.caseId); }});
      video.addEventListener('seeking', () => {{ if (!seekGuard.has(video)) void seekScope(video.dataset.caseId, video.currentTime, video); }});
      video.addEventListener('ratechange', () => {{ if (!rateGuard.has(video)) setRate(video.dataset.caseId, video.playbackRate, video); }});
      video.addEventListener('timeupdate', () => {{
        if (video !== leader || video.paused) return;
        const value = clamp(video.currentTime); updateControls(video.dataset.caseId, value);
        if (activeScope === 'all') updateControls('all', value);
      }});
    }});
    window.setInterval(() => {{
      if (!leader || leader.paused) return; const time = leader.currentTime;
      activeVideos.forEach(video => {{
        if (video !== leader && !video.seeking && Math.abs(video.currentTime - time) > 0.08)
          {{ guardBriefly(seekGuard, video); video.currentTime = time; }}
      }});
    }}, 200);
  }})();
  </script>
</body>
</html>
"""


def _readme(manifest: Mapping[str, Any], case_count: int, copy_mode: str) -> str:
    return f"""# MEV target / self-generated flow calibration review

This directory is a create-only review packet for `{manifest['experiment_id']}`.

**Scientific boundary:** every carrier output uses the fixed historically trained
step-{EXPECTED_HISTORICAL_STEPS} carrier. This calibration performed
**{EXPECTED_CURRENT_STEPS} current optimization steps** and is **not a trained Ours result**.

The page contains {case_count} cases and exactly eight synchronized columns per case:
Source, Real target reference, Self-generated anchor, Frozen, real-forward carrier,
shuffle carrier, reverse carrier, and selfgen carrier. Each row has independent
Play/Pause/Restart, seek, and speed controls; the header controls all rows.

The real target is visible only as a human review reference. Receipts require that
the renderer and trainer did not read target RGB/VAE data and that the carrier saw
only a detached dense-flow representation. Missing media and invalid receipts are
fatal; this packet contains no success placeholders.

- Media publication mode: `{copy_mode}`
- Machine-readable provenance: `review_manifest.json`
- Validated calibration receipts: `receipts/<case-id>/<route>/calibration_receipt.json`
"""


def build(
    manifest_path: str | Path,
    inference_root: str | Path,
    output_dir: str | Path,
    *,
    copy_mode: str = "copy",
) -> Mapping[str, Any]:
    """Validate inputs and atomically publish a synchronized review directory."""

    if copy_mode not in {"copy", "symlink"}:
        raise BuildError("copy_mode must be 'copy' or 'symlink'")
    manifest_file = Path(manifest_path).expanduser().resolve()
    inference = Path(inference_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise BuildError(f"output must be absent or an empty real directory: {output}")
    output_was_empty = output.is_dir() and not any(output.iterdir())
    if output.is_dir() and not output_was_empty:
        raise BuildError(f"refusing to overwrite non-empty output directory: {output}")

    manifest, cases = _validate_inputs(manifest_file, inference)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        review_cases: list[dict[str, Any]] = []
        for case in cases:
            media_dir = stage / "media" / case.case_id
            media: dict[str, Any] = {}
            receipts: dict[str, Any] = {}
            for asset in case.assets:
                target = media_dir / asset.published_name
                digest, size = _publish_media(asset, target, copy_mode=copy_mode)
                media[asset.role] = {
                    "source_path": str(asset.source),
                    "published_path": target.relative_to(stage).as_posix(),
                    "sha256": digest,
                    "size_bytes": size,
                }
                if asset.receipt_source is not None and asset.receipt is not None:
                    route = str(asset.receipt["route_kind"])
                    receipt_target = stage / "receipts" / case.case_id / route / "calibration_receipt.json"
                    receipt_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(asset.receipt_source, receipt_target)
                    receipts[route] = {
                        "source_path": str(asset.receipt_source),
                        "published_path": receipt_target.relative_to(stage).as_posix(),
                        "sha256": _sha256(receipt_target),
                        "validated": True,
                        "historical_carrier_global_step": EXPECTED_HISTORICAL_STEPS,
                        "current_experiment_optimization_steps": EXPECTED_CURRENT_STEPS,
                        "parameter_updates_in_current_experiment": False,
                    }
            review_cases.append(
                {
                    "case_id": case.case_id,
                    "split": case.split,
                    "action_family": case.action_family,
                    "instruction": case.instruction,
                    "seed": case.seed,
                    "media": media,
                    "receipts": receipts,
                }
            )

        review_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": manifest["experiment_id"],
            "claim_boundary": CLAIM_BOUNDARY,
            "method_success_claimed": False,
            "new_ours_training_claimed": False,
            "historical_carrier_global_step": EXPECTED_HISTORICAL_STEPS,
            "current_experiment_optimization_steps": EXPECTED_CURRENT_STEPS,
            "parameter_updates_in_current_experiment": False,
            "source_manifest": {
                "path": str(manifest_file),
                "sha256": _sha256(manifest_file),
            },
            "inference_root": str(inference),
            "copy_mode": copy_mode,
            "columns": [{"key": key, "label": label} for key, label in COLUMNS],
            "synchronization": {
                "per_case_play_pause_restart": True,
                "per_case_seek_and_playback_rate": True,
                "global_play_pause_restart": True,
                "global_seek_and_playback_rate": True,
                "drift_correction_seconds": 0.08,
            },
            "case_count": len(review_cases),
            "cases": review_cases,
        }
        (stage / "review_manifest.json").write_text(
            json.dumps(review_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (stage / "README.md").write_text(
            _readme(manifest, len(review_cases), copy_mode), encoding="utf-8"
        )
        (stage / "index.html").write_text(
            _render_html(manifest, review_cases), encoding="utf-8"
        )

        if output.exists():
            if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
                raise BuildError(f"output changed during build; refusing to overwrite: {output}")
            output.rmdir()
        stage.rename(output)
        return review_manifest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inference-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--copy-mode", choices=("copy", "symlink"), default="copy")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build(
            args.manifest,
            args.inference_root,
            args.output_dir,
            copy_mode=args.copy_mode,
        )
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "case_count": result["case_count"],
                "schema_version": result["schema_version"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

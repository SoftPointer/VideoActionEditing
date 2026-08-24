#!/usr/bin/env python3
"""Build the v14r3d2 gradient-geometry decode review.

This builder is intentionally fail-closed: it never renders a placeholder card.
Every requested decode MP4 and its audit sidecar must exist and satisfy the
expected checkpoint step, route state, preservation mode, frame count and FPS.

The ``--print-required`` mode is safe to run before decoding finishes.  It emits
the exact paths that must be copied from the remote ``dynaedit_fullgrid_v2``
directory into a local mirror before the one-shot review build.

The default ``full`` profile remains the closure review.  The
``route-on-progress`` profile is a fail-closed interim view: it renders only
the three route-ON arms which have completed for the requested events, plus
the immutable Source / Pure-T2V anchor / Frozen references.  It never relaxes
the MP4, sidecar, frame-rate, frame-count or hash checks.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REMOTE_DECODE_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/"
    "online_anchor_attention_training_v1/dynaedit_fullgrid_v2"
)


@dataclass(frozen=True)
class Event:
    ordinal: int
    slug: str
    name: str
    instruction: str
    aspect_ratio: str


@dataclass(frozen=True)
class DecodeArm:
    key: str
    experiment: str
    tag: str
    step: int
    transport_steps: int
    route_off: bool
    preservation_mode: str
    style: str
    label: str
    note: str
    events: tuple[int, ...] = (0, 2, 4, 7)


EVENTS = (
    Event(
        0,
        "pour-liquid-into-cup",
        "Pour pitcher into cup",
        "Lift and align the same pitcher, show a continuous stream and a rising "
        "cup level, then return the pitcher upright. Preserve source person, "
        "pitcher, cup, scene, camera and frame 0.",
        "2 / 3",
    ),
    Event(
        2,
        "twist-pull-mushroom",
        "Twist and pull mushroom",
        "Grasp and twist the same rooted mushroom, detach and lift it, and leave "
        "the original hole empty. Preserve source hand, mushroom, ground and camera.",
        "2 / 3",
    ),
    Event(
        4,
        "close-door-then-drawer",
        "Close door, then drawer",
        "Close the hinged lower door first, then push the separate upper drawer "
        "inward and hold both closed. Preserve the source cabinet and room.",
        "3 / 4",
    ),
    Event(
        7,
        "players-contact-then-separate",
        "Players contact, then separate",
        "Push off once, create a visible gap between the same two players, then "
        "continue on distinct paths. Preserve player identity, uniforms, court and camera.",
        "3 / 2",
    ),
)


ACTION_TAG = (
    "sameaction_global_actiononly_s8_v14r3_gradgeom_"
    "LASTVALID_S4_ABORTED_PREOPT_S5_DIAG"
)
NORM_TAG = (
    "sameaction_global_norm025_s8_v14r3_gradgeom_"
    "LASTVALID_S4_ABORTED_PREOPT_S5_DIAG"
)
PCGRAD_TAG = "sameaction_gate25_pcgrad010_s8_v14r3_gradgeom"


ARMS = (
    DecodeArm(
        "action_route_on",
        "sameaction_global_actiononly_s8_v14r3_gradgeom",
        f"{ACTION_TAG}_routeon_clean40_v14r3d2",
        4,
        40,
        False,
        "none",
        "diagnostic",
        "Action-only · last valid S4 · route ON",
        "Run aborted before the attempted S5 update. Diagnostic checkpoint only; "
        "not a completed or promotion-eligible run.",
    ),
    DecodeArm(
        "norm_route_on",
        "sameaction_global_norm025_s8_v14r3_gradgeom",
        f"{NORM_TAG}_routeon_clean40_v14r3d2",
        4,
        40,
        False,
        "none",
        "diagnostic",
        "Norm-balanced .25 · last valid S4 · route ON",
        "Run aborted before the attempted S5 update. Diagnostic checkpoint only; "
        "not a completed or promotion-eligible run.",
    ),
    DecodeArm(
        "pcgrad_route_on",
        "sameaction_gate25_pcgrad010_s8_v14r3_gradgeom",
        f"{PCGRAD_TAG}_routeon_clean40_v14r3d2",
        8,
        40,
        False,
        "none",
        "preservation-risk",
        "PCGrad .10 · completed S8 · route ON",
        "Action-priority conflict control. Source descent was not required by its "
        "update audit, so this checkpoint is explicitly not preservation-safe.",
    ),
    DecodeArm(
        "norm_preservation",
        "sameaction_global_norm025_s8_v14r3_gradgeom",
        f"{NORM_TAG}_preserve_motion_actionreward_v14r3d2",
        4,
        40,
        False,
        "source_motion_support",
        "diagnostic",
        "Norm-balanced .25 · last valid S4 · preservation decode",
        "Same aborted-run S4 diagnostic checkpoint with inference-time source-motion "
        "support; this is not evidence that the training run completed.",
        (0, 4),
    ),
    DecodeArm(
        "action_route_off",
        "sameaction_global_actiononly_s8_v14r3_gradgeom",
        f"{ACTION_TAG}_sameckpt_routeoff_control_v14r3d2",
        4,
        0,
        True,
        "none",
        "diagnostic",
        "Action-only · last valid S4 · route OFF",
        "Same diagnostic S4 adapter with online anchor transport disabled. The run "
        "aborted before the attempted S5 update.",
    ),
    DecodeArm(
        "norm_route_off",
        "sameaction_global_norm025_s8_v14r3_gradgeom",
        f"{NORM_TAG}_sameckpt_routeoff_control_v14r3d2",
        4,
        0,
        True,
        "none",
        "diagnostic",
        "Norm-balanced .25 · last valid S4 · route OFF",
        "Same diagnostic S4 adapter with online anchor transport disabled. The run "
        "aborted before the attempted S5 update.",
    ),
    DecodeArm(
        "pcgrad_route_off",
        "sameaction_gate25_pcgrad010_s8_v14r3_gradgeom",
        f"{PCGRAD_TAG}_sameckpt_routeoff_control_v14r3d2",
        8,
        0,
        True,
        "none",
        "preservation-risk",
        "PCGrad .10 · completed S8 · route OFF",
        "Same S8 action-priority checkpoint with online anchor transport disabled. "
        "Its training update audit did not require source descent; not preservation-safe.",
    ),
)


ARM_BY_KEY = {arm.key: arm for arm in ARMS}

PROFILE_FULL = "full"
PROFILE_ROUTE_ON_PROGRESS = "route-on-progress"
PROFILE_CHOICES = (PROFILE_FULL, PROFILE_ROUTE_ON_PROGRESS)
ROUTE_ON_PROGRESS_KEYS = (
    "action_route_on",
    "norm_route_on",
    "pcgrad_route_on",
)

BASELINE_SHA256 = {
    0: {
        "authority": "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de",
        "anchor": "e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa",
        "baseline": "a9fc55338ec4dbf5b338f20cda573c4e9670b0b0236fbd73703a4fdc774a51a7",
    },
    2: {
        "authority": "63dd620627e18d0f6836058bf725ffba7ab1f9b0b455e784c6a71dc68a8de46c",
        "anchor": "a1076cbe83c9dae4a4fddc25f73077288f6a3240324fd2d5e1854aa842b07b63",
        "baseline": "bbd5aeea46bd3c72870b9df5a449a601ef0b0f6eeff5e3cc4258868d43c254c8",
    },
    4: {
        "authority": "d699a8d5e35a57f09ae4ba5fc5124e733be9ed18a2bddb2ee90a1ba0232c53f5",
        "anchor": "c6d6a4e2835972609fcde8a8fbc2357eb36396f4a54aef7366adf809d6593f5e",
        "baseline": "b56c80902c30b3026747da7f5fb74af80e60d9f74ac6e4b9f8cc88d5ef8d6d50",
    },
    7: {
        "authority": "1164531fd34d3d1273d56930aed139eb1a5d8db708ac3cdc4f7434abc0080799",
        "anchor": "7b128ed47a7f6122d40a12711cf31535e39bcd5b92ce97d031cc2ff49424f4fc",
        "baseline": "15aa3b001e75381b14212982c387309e9668f9f0ba28dbc43a0fafab5c22fc20",
    },
}


class ReviewError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_by_ordinal(ordinal: int) -> Event:
    for event in EVENTS:
        if event.ordinal == ordinal:
            return event
    raise ReviewError(f"unsupported event: {ordinal}")


def decode_relative_path(arm: DecodeArm, event: Event) -> Path:
    label = (
        f"E{event.ordinal:02d}_{event.slug}_{arm.tag}_S{arm.step}_"
        "ONLINE_ANCHOR_REAL_SGA_ANC.mp4"
    )
    return (
        Path(arm.tag)
        / f"step_{arm.step:08d}"
        / f"e{event.ordinal:02d}"
        / label
    )


def arms_for_profile(profile: str) -> tuple[DecodeArm, ...]:
    if profile == PROFILE_FULL:
        return ARMS
    if profile == PROFILE_ROUTE_ON_PROGRESS:
        return tuple(ARM_BY_KEY[key] for key in ROUTE_ON_PROGRESS_KEYS)
    raise ReviewError(f"unsupported review profile: {profile}")


def required_mapping(
    decoded_root: Path,
    event_ordinals: Sequence[int],
    profile: str = PROFILE_FULL,
) -> dict[str, Any]:
    event_set = set(event_ordinals)
    rows: list[dict[str, Any]] = []
    for arm in arms_for_profile(profile):
        for ordinal in arm.events:
            if ordinal not in event_set:
                continue
            relative = decode_relative_path(arm, event_by_ordinal(ordinal))
            rows.append(
                {
                    "event": ordinal,
                    "arm": arm.key,
                    "step": arm.step,
                    "remote_mp4": f"{REMOTE_DECODE_ROOT}/{relative.as_posix()}",
                    "remote_sidecar": (
                        f"{REMOTE_DECODE_ROOT}/{relative.as_posix()}.receipt.json"
                    ),
                    "local_mp4": str(decoded_root / relative),
                    "local_sidecar": str(decoded_root / f"{relative}.receipt.json"),
                }
            )
    return {
        "schema_version": "bernini-v14r3d2-decode-review-required-files-v1",
        "profile": profile,
        "remote_decode_root": REMOTE_DECODE_ROOT,
        "local_decode_root": str(decoded_root),
        "required_decode_pairs": len(rows),
        "rows": rows,
    }


def read_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ReviewError(f"missing plain JSON sidecar: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReviewError(f"JSON sidecar root is not an object: {path}")
    return value


def nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ReviewError(f"sidecar is missing {'.'.join(keys)}")
        current = current[key]
    return current


def ffprobe_video(path: Path) -> None:
    command = (
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    )
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ReviewError(f"ffprobe failed for {path}: {exc}") from exc
    value = json.loads(result.stdout)
    streams = value.get("streams") if isinstance(value, dict) else None
    if not isinstance(streams, list) or len(streams) != 1:
        raise ReviewError(f"expected exactly one video stream: {path}")
    stream = streams[0]
    if stream.get("nb_read_frames") != "81" or stream.get("avg_frame_rate") != "25/1":
        raise ReviewError(f"video is not exact 81-frame 25-fps output: {path}")


def validate_decode(decoded_root: Path, arm: DecodeArm, event: Event) -> tuple[Path, Path, dict[str, Any]]:
    relative = decode_relative_path(arm, event)
    video = decoded_root / relative
    sidecar_path = decoded_root / f"{relative}.receipt.json"
    if not video.is_file() or video.is_symlink() or video.stat().st_size == 0:
        raise ReviewError(f"missing plain non-empty decode MP4: {video}")
    sidecar = read_object(sidecar_path)
    expected = {
        "complete": True,
        "loaded_trained_attention_checkpoint": True,
        "step": arm.step,
        "transport_steps": arm.transport_steps,
        "preservation_mode": arm.preservation_mode,
        "route_off": arm.route_off,
        "frames": 81,
        "fps": 25,
    }
    actual = {
        "complete": sidecar.get("complete"),
        "loaded_trained_attention_checkpoint": sidecar.get(
            "loaded_trained_attention_checkpoint"
        ),
        "step": nested(sidecar, "trained_attention_checkpoint", "global_step"),
        "transport_steps": nested(sidecar, "mechanism", "transport_steps"),
        "preservation_mode": nested(sidecar, "mechanism", "preservation_mode"),
        "route_off": nested(
            sidecar,
            "mechanism",
            "decode_audit_contract",
            "same_checkpoint_route_off_causal_control",
        ),
        "frames": nested(sidecar, "output", "frames"),
        "fps": nested(sidecar, "output", "fps"),
    }
    if actual != expected:
        raise ReviewError(
            f"decode sidecar contract differs for {video}: expected={expected}, actual={actual}"
        )
    output_sha = nested(sidecar, "output", "sha256")
    if output_sha != sha256(video):
        raise ReviewError(f"decode MP4 hash differs from sidecar: {video}")
    ffprobe_video(video)
    return video, sidecar_path, sidecar


def copy_plain(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink() or source.stat().st_size == 0:
        raise ReviewError(f"missing plain non-empty input: {source}")
    if destination.exists() or destination.is_symlink():
        raise ReviewError(f"duplicate review destination: {destination}")
    shutil.copy2(source, destination)
    return {
        "path": f"media/{destination.name}",
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def baseline_card(
    baseline_review: Path,
    media: Path,
    event: Event,
    role: str,
    source_name: str,
    destination_name: str,
    label: str,
    note: str,
) -> dict[str, Any]:
    source = baseline_review / "media" / source_name
    expected_sha = BASELINE_SHA256[event.ordinal][role]
    if not source.is_file() or source.is_symlink() or sha256(source) != expected_sha:
        raise ReviewError(
            f"baseline authority hash differs for event {event.ordinal} role {role}: {source}"
        )
    artifact = copy_plain(
        source,
        media / destination_name,
    )
    return {"role": role, "label": label, "note": note, "artifact": artifact}


def decode_card(
    decoded_root: Path,
    media: Path,
    event: Event,
    arm: DecodeArm,
) -> dict[str, Any]:
    video, sidecar, sidecar_value = validate_decode(decoded_root, arm, event)
    destination_name = f"e{event.ordinal:02d}-{arm.key}.mp4"
    artifact = copy_plain(video, media / destination_name)
    sidecar_artifact = copy_plain(
        sidecar, media / f"{destination_name}.receipt.json"
    )
    return {
        "role": arm.style,
        "label": arm.label,
        "note": arm.note,
        "arm": arm.key,
        "training_experiment": arm.experiment,
        "training_step": arm.step,
        "training_run_complete": arm.step == 8,
        "promotion_authorized": False,
        "artifact": artifact,
        "audit_sidecar": sidecar_artifact,
        "remote_output_path": nested(sidecar_value, "output", "path"),
    }


def selected_events(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("events must be comma-separated integers") from exc
    valid = {event.ordinal for event in EVENTS}
    if not result or len(result) != len(set(result)) or not set(result) <= valid:
        raise argparse.ArgumentTypeError("events must be a unique subset of 0,2,4,7")
    return result


def build(
    decoded_root: Path,
    baseline_review: Path,
    output: Path,
    event_ordinals: Sequence[int],
    profile: str = PROFILE_FULL,
) -> dict[str, Any]:
    if output.exists() or output == Path("/"):
        raise ReviewError("output must be a fresh non-root directory")
    if not decoded_root.is_dir() or decoded_root.is_symlink():
        raise ReviewError(f"decoded root must be a plain directory: {decoded_root}")
    if not baseline_review.is_dir() or baseline_review.is_symlink():
        raise ReviewError(f"baseline review must be a plain directory: {baseline_review}")

    profile_arms = arms_for_profile(profile)

    # Validate the entire requested profile before creating the output directory.
    for ordinal in event_ordinals:
        event = event_by_ordinal(ordinal)
        for arm in profile_arms:
            if ordinal in arm.events:
                validate_decode(decoded_root, arm, event)

    output.mkdir(parents=True)
    media = output / "media"
    media.mkdir()
    event_rows: list[dict[str, Any]] = []
    for ordinal in event_ordinals:
        event = event_by_ordinal(ordinal)
        prefix = f"e{ordinal:02d}"
        authority = baseline_card(
            baseline_review,
            media,
            event,
            "authority",
            f"{prefix}-source.mp4",
            f"{prefix}-source.mp4",
            "Source authority",
            "Sole authority for identity, appearance, objects, scene, camera and frame 0.",
        )
        anchor = baseline_card(
            baseline_review,
            media,
            event,
            "anchor",
            f"{prefix}-t2v-anchor-v0.mp4",
            f"{prefix}-t2v-anchor-v0.mp4",
            "Pure-T2V action anchor",
            "Action/temporal donor only. Its appearance, clothing, object instances and "
            "background are explicitly irrelevant.",
        )
        frozen = baseline_card(
            baseline_review,
            media,
            event,
            "baseline",
            f"{prefix}-frozen-s0.mp4",
            f"{prefix}-frozen-s0.mp4",
            "Frozen RV2V · 0 update",
            "Matched source/instruction baseline and failure control; not a target.",
        )
        route_on_keys = list(ROUTE_ON_PROGRESS_KEYS)
        if profile == PROFILE_FULL and ordinal in (0, 4):
            route_on_keys.append("norm_preservation")
        route_off_keys = ["action_route_off", "norm_route_off", "pcgrad_route_off"]
        rows = [
            {
                "title": "Authority and frozen control",
                "note": "Judge content only from Source; judge the requested temporal action from the instruction and anchor motion, never from anchor appearance.",
                "cards": [authority, anchor, frozen],
            },
            {
                "title": "Anchor route ON",
                "note": "Online self-generated anchor Q/K transport is active. Orange S4 cards are last-valid checkpoints from runs that aborted before the S5 update; red PCGrad is not preservation-safe.",
                "cards": [
                    decode_card(decoded_root, media, event, ARM_BY_KEY[key])
                    for key in route_on_keys
                ],
            },
        ]
        if profile == PROFILE_FULL:
            rows.append(
                {
                    "title": "Same-checkpoint route-OFF controls",
                    "note": "The corresponding trained adapter remains loaded, but online anchor transport is disabled. These are mechanism controls, not alternative ground truth.",
                    "cards": [
                        decode_card(decoded_root, media, event, ARM_BY_KEY[key])
                        for key in route_off_keys
                    ],
                }
            )
        event_rows.append(
            {
                "ordinal": ordinal,
                "name": event.name,
                "instruction": event.instruction,
                "aspect_ratio": event.aspect_ratio,
                "rows": rows,
            }
        )

    is_progress = profile == PROFILE_ROUTE_ON_PROGRESS
    receipt = {
        "schema_version": "bernini-v14r3d2-gradgeom-decode-review-v1",
        "profile": profile,
        "title": (
            "v14r3d2 route-ON progress review"
            if is_progress
            else "v14r3d2 gradient-geometry decode review"
        ),
        "interim_progress_review": is_progress,
        "machine_correct_answer_shown": False,
        "human_annotation_controls_shown": False,
        "source_is_content_authority": True,
        "pure_t2v_anchor_appearance_is_target": False,
        "s4_aborted_runs_are_diagnostic_only": True,
        "pcgrad_s8_is_preservation_safe": False,
        "promotion_claimed": False,
        "max_cards_per_row": 3 if is_progress else 4,
        "events": event_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(render(receipt), encoding="utf-8")
    (output / "COMPLETE").write_text("complete\n", encoding="ascii")
    return receipt


def render_card(card: Mapping[str, Any], group: str) -> str:
    role = html.escape(str(card["role"]), quote=True)
    label = html.escape(str(card["label"]))
    note = html.escape(str(card["note"]))
    path = html.escape(str(nested(card, "artifact", "path")), quote=True)
    badge = ""
    if role == "diagnostic":
        badge = '<span class="badge diagnostic">S4 ABORTED · DIAGNOSTIC</span>'
    elif role == "preservation-risk":
        badge = '<span class="badge risk">NOT PRESERVATION-SAFE</span>'
    return f'''<article class="card {role}">
<div class="label"><span>{label}</span>{badge}</div>
<video controls muted playsinline preload="metadata" data-group="{html.escape(group, quote=True)}" src="{path}"></video>
<div class="note">{note}</div></article>'''


def render(receipt: Mapping[str, Any]) -> str:
    sections: list[str] = []
    for event in receipt["events"]:
        ordinal = int(event["ordinal"])
        rows: list[str] = []
        for row_index, row in enumerate(event["rows"]):
            group = f"e{ordinal}-r{row_index}"
            cards = "".join(render_card(card, group) for card in row["cards"])
            columns = len(row["cards"])
            rows.append(
                f'''<div class="row" id="{group}">
<div class="rowhead"><div><h3>{html.escape(str(row["title"]))}</h3><p>{html.escape(str(row["note"]))}</p></div>
<button onclick="syncPlay('[data-group=&quot;{group}&quot;]',this)">同步播放本行</button></div>
<div class="grid" style="--columns:{columns};--event-aspect:{html.escape(str(event['aspect_ratio']), quote=True)}">{cards}</div></div>'''
            )
        sections.append(
            f'''<section class="event" id="event-{ordinal}"><header><div><h2>Event {ordinal:02d} · {html.escape(str(event["name"]))}</h2>
<p>{html.escape(str(event["instruction"]))}</p></div><button onclick="syncPlay('#event-{ordinal} video',this)">同步播放本事件</button></header>{''.join(rows)}</section>'''
        )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(receipt["title"]))}</title><style>
:root{{--bg:#f3efe7;--panel:#fffdf8;--ink:#17211e;--muted:#63706b;--line:#d2c7b6;--green:#176b57;--orange:#d77a13;--red:#b33b35}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.35 system-ui,-apple-system,sans-serif}}
.top{{position:sticky;top:0;z-index:5;display:flex;gap:9px;align-items:center;padding:8px 12px;background:#f3efe7f2;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
.top h1{{font-size:17px;margin:0}}.top .legend{{display:flex;gap:6px;align-items:center;color:var(--muted);font-size:12px}}.top button{{margin-left:auto}}
button{{border:1px solid #9e927f;background:#fffaf1;border-radius:8px;padding:7px 10px;font-weight:750;cursor:pointer;white-space:nowrap}}button:disabled{{opacity:.5}}
main{{padding:8px}}.warning{{display:flex;gap:8px;flex-wrap:wrap;padding:8px 10px;margin-bottom:8px;background:#fffdf8;border:1px solid var(--line);border-radius:10px;color:var(--muted);font-size:12px}}
.event{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:9px;margin-bottom:10px}}.event>header,.rowhead{{display:flex;align-items:start;gap:10px}}.event>header>div,.rowhead>div{{flex:1;min-width:0}}
h2{{font-size:18px;margin:0 0 2px}}h3{{font-size:14px;margin:0}}p{{margin:0;color:var(--muted);font-size:12px}}.event>header>div>p{{max-width:1500px}}
.row{{margin-top:8px;padding-top:7px;border-top:1px solid #e2d9cc}}.rowhead{{margin-bottom:5px}}.rowhead p{{max-width:1450px}}.grid{{display:grid;grid-template-columns:repeat(var(--columns),minmax(0,1fr));gap:7px;align-items:stretch;max-width:1600px}}
.card{{display:grid;grid-template-rows:66px auto 1fr;min-width:0;border:1px solid #78988d;border-radius:9px;overflow:hidden;background:#fff}}.card.diagnostic{{border-color:var(--orange)}}.card.preservation-risk{{border-color:var(--red)}}
.label{{padding:5px 7px;display:flex;flex-direction:column;justify-content:center;gap:3px;font-size:12px;line-height:1.16;font-weight:800;overflow:hidden}}.label>span:first-child{{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
.badge{{align-self:flex-start;padding:2px 5px;border-radius:999px;color:#fff;font-size:9px;line-height:1.2;letter-spacing:.03em}}.badge.diagnostic{{background:var(--orange)}}.badge.risk{{background:var(--red)}}
video{{display:block;width:100%;aspect-ratio:var(--event-aspect);object-fit:contain;background:#0a0b0a}}.note{{min-height:46px;padding:5px 7px;color:var(--muted);font-size:11px}}
@media(max-width:1050px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}.top .legend{{display:none}}.event>header,.rowhead{{align-items:center}}}}
</style></head><body><div class="top"><h1>{html.escape(str(receipt["title"]))}</h1><div class="legend"><span class="badge diagnostic">S4 ABORTED · DIAGNOSTIC</span><span class="badge risk">NOT PRESERVATION-SAFE</span></div><button onclick="pauseAll()">全部暂停</button></div>
<main><div class="warning"><strong>判读边界：</strong><span>Source 是内容/identity 唯一权威；Pure-T2V anchor 只提供动作，外观不是目标。</span><span>橙色 S4 来自 S5 更新前中止的 run，只能诊断。</span><span>红色 PCGrad S8 未要求 source descent，不可称为 preservation-safe。</span><span>页面不显示机器“正确答案”，也不包含无用标注表单。</span></div>{''.join(sections)}</main>
<script>function metadata(v){{if(v.readyState>=1)return Promise.resolve();return new Promise((ok,bad)=>{{v.addEventListener('loadedmetadata',ok,{{once:true}});v.addEventListener('error',()=>bad(Error('media load failed: '+v.currentSrc)),{{once:true}});v.load()}})}}
async function syncPlay(selector,button){{const vs=[...document.querySelectorAll(selector)];const old=button.textContent;button.disabled=true;button.textContent='加载并对齐…';try{{vs.forEach(v=>{{v.pause();v.muted=true}});await Promise.all(vs.map(metadata));vs.forEach(v=>v.currentTime=0);await Promise.all(vs.map(v=>new Promise((ok,bad)=>{{const done=()=>{{v.removeEventListener('seeked',done);ok()}};if(v.currentTime===0)ok();else{{v.addEventListener('seeked',done,{{once:true}});setTimeout(()=>bad(Error('seek timeout')),5000)}}}})));const r=await Promise.allSettled(vs.map(v=>v.play()));if(r.some(x=>x.status==='rejected'))throw Error('browser rejected part of playback')}}catch(e){{alert('同步播放失败：'+e.message+'。请通过 http://127.0.0.1 服务打开并检查 media。')}}finally{{button.disabled=false;button.textContent=old}}}}function pauseAll(){{document.querySelectorAll('video').forEach(v=>v.pause())}}</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-root", required=True)
    parser.add_argument("--baseline-review")
    parser.add_argument("--output")
    parser.add_argument("--events", type=selected_events, default=(0, 2, 4, 7))
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=PROFILE_FULL,
        help="full closure review (default) or fail-closed route-ON progress review",
    )
    parser.add_argument(
        "--print-required",
        action="store_true",
        help="print the exact remote/local decode mapping and do not build",
    )
    args = parser.parse_args()
    decoded_root = Path(args.decoded_root).expanduser().resolve()
    if args.print_required:
        print(
            json.dumps(
                required_mapping(decoded_root, args.events, args.profile),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    if not args.baseline_review or not args.output:
        parser.error("--baseline-review and --output are required unless --print-required is used")
    receipt = build(
        decoded_root,
        Path(args.baseline_review).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        args.events,
        args.profile,
    )
    print(
        json.dumps(
            {
                "events": len(receipt["events"]),
                "output": str(Path(args.output).expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

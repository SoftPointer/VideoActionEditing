#!/usr/bin/env python3
"""Build the future-joint-null six-view, reject-only v15c-r9 source audit.

Current r6 common-null replay remains diagnostic-only and produces no ownership
artifact; this builder cannot promote that NO-GO result.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_source_role_ownership_v15c_r9 as ownership_builder
import postflight_source_four_role_authority_v15c_r9 as postflight_module
import source_role_authority_v15c_r9 as core


SCHEMA = "bernini-source-four-role-overlay-v15c-r9-local"
MEDIA_SCHEMA = "bernini-source-four-role-media-v15c-r9-local"
DISPLAY_FRAMES = (0, 20, 40, 60, 80)
VIEW_KEYS = ("source", "all", *core.ROLE_NAMES)
LAYOUT_ROWS = (
    ("source", "all", "human_agent"),
    ("old_actor", "new_actor", "recipient"),
)
MEDIA_CONTRACT = {
    key: {
        "video": f"media/{key}.mp4",
        "contact_sheet": f"media/{key}_f00_20_40_60_80.jpg",
    }
    for key in VIEW_KEYS
}


class BuildSourceFourRoleReviewV15CR9Error(RuntimeError):
    """Reject-only review input, media, or layout differs."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise BuildSourceFourRoleReviewV15CR9Error("JSON input differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BuildSourceFourRoleReviewV15CR9Error("JSON input differs") from error
    if type(value) is not dict:
        raise BuildSourceFourRoleReviewV15CR9Error("JSON object differs")
    payload = dict(value)
    claimed = payload.pop("receipt_sha256", None)
    if (
        type(claimed) is not str
        or core.SHA256_PATTERN.fullmatch(claimed) is None
        or claimed != core.object_sha256(payload)
    ):
        raise BuildSourceFourRoleReviewV15CR9Error("JSON self hash differs")
    return value


def _decode_source(path: Path) -> list[Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise BuildSourceFourRoleReviewV15CR9Error("source media open failed")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if (
        len(frames) != 81
        or abs(fps - 25.0) > 1.0e-6
        or any(frame.shape[:2] != (1056, 704) for frame in frames)
    ):
        raise BuildSourceFourRoleReviewV15CR9Error("source media contract differs")
    return frames


def _draw_mask(canvas: Any, mask: Any, color: tuple[int, int, int], thickness: int) -> None:
    import cv2
    import numpy as np

    contours, _hierarchy = cv2.findContours(
        np.ascontiguousarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(canvas, contours, -1, color, thickness)


def _annotate(
    *,
    frame: Any,
    frame_index: int,
    label: str,
    raw_masks: Sequence[Any],
    ownership_mask: Any | None,
    contact_mask: Any | None,
    unassigned_mask: Any | None,
) -> Any:
    import cv2
    import numpy as np

    canvas = frame.copy()
    colors = ((40, 220, 255), (255, 150, 30), (80, 255, 80), (255, 80, 200))
    for index, mask in enumerate(raw_masks):
        _draw_mask(canvas, mask, colors[index % len(colors)], 2)
    if ownership_mask is not None:
        tint = np.zeros_like(canvas)
        tint[:, :] = (40, 210, 40)
        selector = np.asarray(ownership_mask, dtype=np.bool_)
        canvas[selector] = cv2.addWeighted(canvas, 0.45, tint, 0.55, 0)[selector]
        _draw_mask(canvas, selector, (40, 255, 40), 3)
    if contact_mask is not None:
        selector = np.asarray(contact_mask, dtype=np.bool_)
        canvas[selector] = (255, 40, 255)
    if unassigned_mask is not None:
        selector = np.asarray(unassigned_mask, dtype=np.bool_)
        canvas[selector] = (20, 20, 255)
    cv2.rectangle(canvas, (0, 0), (704, 68), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"{label} | f{frame_index:02d}",
        (10, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "raw contour | ownership green | contact magenta | unresolved red",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _write_video_and_sheet(
    *,
    frames: Sequence[Any],
    label: str,
    raw_masks: Sequence[Any],
    ownership_mask: Any | None,
    contact_mask: Any | None,
    unassigned_mask: Any | None,
    video_path: Path,
    sheet_path: Path,
) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (704, 1056)
    )
    if not writer.isOpened():
        raise BuildSourceFourRoleReviewV15CR9Error("overlay writer failed")
    selected = []
    for frame_index, frame in enumerate(frames):
        annotated = _annotate(
            frame=frame,
            frame_index=frame_index,
            label=label,
            raw_masks=[value[frame_index] for value in raw_masks],
            ownership_mask=(None if ownership_mask is None else ownership_mask[frame_index]),
            contact_mask=(None if contact_mask is None else contact_mask[frame_index]),
            unassigned_mask=(None if unassigned_mask is None else unassigned_mask[frame_index]),
        )
        writer.write(annotated)
        if frame_index in DISPLAY_FRAMES:
            selected.append(
                cv2.resize(annotated, (352, 528), interpolation=cv2.INTER_AREA)
            )
    writer.release()
    if len(selected) != len(DISPLAY_FRAMES):
        raise BuildSourceFourRoleReviewV15CR9Error("contact sheet frames differ")
    sheet = np.concatenate(selected, axis=1)
    if not cv2.imwrite(str(sheet_path), sheet):
        raise BuildSourceFourRoleReviewV15CR9Error("contact sheet write failed")
    reopened = cv2.imread(str(sheet_path), cv2.IMREAD_COLOR)
    if not isinstance(reopened, np.ndarray) or reopened.shape != (528, 1760, 3):
        raise BuildSourceFourRoleReviewV15CR9Error("contact sheet reopen differs")


def _reopen_video(path: Path) -> Mapping[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise BuildSourceFourRoleReviewV15CR9Error("rendered media open failed")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (1056, 704):
            capture.release()
            raise BuildSourceFourRoleReviewV15CR9Error("rendered frame differs")
        count += 1
    capture.release()
    if count != 81 or abs(fps - 25.0) > 1.0e-6 or (width, height) != (704, 1056):
        raise BuildSourceFourRoleReviewV15CR9Error("rendered media contract differs")
    return {
        "sha256": file_sha256(path),
        "frame_count": count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _evidence_html(result: Mapping[str, Any]) -> str:
    sections = []
    for role in core.ROLE_NAMES:
        rows = []
        for row in result["evidence"][role]:
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(row['proposal_id'])}</code></td>"
                f"<td>{row['eligible_before_proposal_competition']}</td>"
                f"<td>{row['track_real']}</td><td>{row['track_shuffled']}</td>"
                f"<td>{row['global_max_t_empirical_upper_p']}</td>"
                f"<td>{row['vessel_three_role_bonferroni_fwer_upper_p']}</td>"
                f"<td>{row['consistent_phase_count']}/{row['longest_consistent_run']}</td>"
                f"<td><pre>{html.escape(json.dumps(row['same_role_duplicate_or_nesting_neighbors'], sort_keys=True))}</pre></td>"
                f"<td><pre>{html.escape(json.dumps(row['gates'], sort_keys=True, indent=2))}</pre></td>"
                f"<td><details><summary>all fields</summary><pre>{html.escape(json.dumps(row, sort_keys=True, indent=2))}</pre></details></td>"
                "</tr>"
            )
        sections.append(
            f"<details open><summary>{html.escape(role)} = "
            f"{html.escape(result['assignments'][role] or 'UNASSIGNED')}</summary>"
            f"<pre>{html.escape(json.dumps(result['competition'][role], sort_keys=True, indent=2))}</pre>"
            "<table><thead><tr><th>proposal</th><th>eligible</th><th>real</th>"
            "<th>shuffle</th><th>global max-T p</th><th>vessel extra FWER</th>"
            "<th>consistent/run</th><th>duplicate/nesting</th><th>gates</th><th>full</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></details>"
        )
    return "".join(sections)


def _html(result: Mapping[str, Any]) -> str:
    rows = []
    titles = {
        "source": "Source authority",
        "all": "All SAM2 proposals",
        "human_agent": "human_agent",
        "old_actor": "old_actor #1",
        "new_actor": "new_actor #2",
        "recipient": "recipient #3",
    }
    for layout in LAYOUT_ROWS:
        cards = []
        for key in layout:
            media = MEDIA_CONTRACT[key]
            cards.append(
                f'<section class="card"><h2>{html.escape(titles[key])}</h2>'
                f'<video muted loop playsinline preload="metadata" src="{media["video"]}"></video>'
                f'<img loading="lazy" src="{media["contact_sheet"]}" alt="0 20 40 60 80"></section>'
            )
        rows.append(f'<div class="grid">{"".join(cards)}</div>')
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>v15c-r9 four-role source audit</title><style>
body{{margin:18px;background:#f6f3eb;color:#18201d;font:15px system-ui,sans-serif}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:10px;margin-bottom:12px}}
.card{{background:#fff;border:1px solid #b9ac93;border-radius:10px;overflow:hidden}}
.card h2{{font-size:15px;margin:0;padding:8px}} video,img{{display:block;width:100%;background:#111}}
.bar{{position:sticky;top:0;background:#f6f3eb;padding:8px 0;z-index:3}}
button{{padding:9px 14px;margin:3px;border:1px solid #8a3128;border-radius:8px;background:#fff;color:#81261f;font-weight:700}}
table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{border:1px solid #b9ac93;padding:7px;vertical-align:top}}
pre{{white-space:pre-wrap;font-size:10px}}code{{font-size:9px}}.warning{{font-weight:800;color:#8c251d}}
</style></head><body><h1>v15c-r9 source-only four-role observer</h1>
<p class="warning">仅可拒绝。绿色是互斥 ownership，黄色/蓝色轮廓是原始 proposal，洋红是独立 contact relation，红色是无法裁决像素。机械通过不是定位真值、动作成功或 route 许可。</p>
<div class="bar"><button id="play">同步播放全部</button><button id="pause">全部暂停</button><button id="export">导出拒绝</button></div>
{''.join(rows)}<h2>全部 null / shuffle / temporal / competition 失败证据</h2>
{_evidence_html(result)}<div>{''.join(f'<button class="reject" data-role="{role}">Reject {role}</button>' for role in core.ROLE_NAMES)}</div>
<script>const videos=[...document.querySelectorAll('video')];
play.onclick=async()=>{{videos.forEach(v=>v.currentTime=0);await Promise.allSettled(videos.map(v=>v.play()));}};
pause.onclick=()=>videos.forEach(v=>v.pause());let rejects={{}};
document.querySelectorAll('.reject').forEach(b=>b.onclick=()=>{{rejects[b.dataset.role]={{rejected:true,time:new Date().toISOString()}};b.textContent='Rejected '+b.dataset.role;}});
export.onclick=()=>{{const blob=new Blob([JSON.stringify({{schema_version:'v15c-r9-reject-only',rejects,approve_action_available:false,route_authorized:false}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='v15c_r9_rejections.json';a.click();URL.revokeObjectURL(a.href);}};</script>
</body></html>"""


def build_review(
    *,
    source_video: Path,
    track_receipt_path: Path,
    assignment_result_path: Path,
    ownership_dir: Path,
    postflight_path: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    import cv2
    import numpy as np
    from safetensors.numpy import load_file

    source = source_video.resolve(strict=True)
    track_path = track_receipt_path.resolve(strict=True)
    result_path = assignment_result_path.resolve(strict=True)
    ownership_root = ownership_dir.resolve(strict=True)
    postflight_json = postflight_path.resolve(strict=True)
    output = output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise BuildSourceFourRoleReviewV15CR9Error("review output is not fresh")
    result = read_json(result_path)
    ownership_receipt = read_json(ownership_root / "ownership_receipt.json")
    postflight = read_json(postflight_json)
    if (
        set(postflight) != set(postflight_module.POSTFLIGHT_KEYS)
        or set(ownership_receipt) != set(ownership_builder.RECEIPT_KEYS)
        or postflight.get("schema_version") != postflight_module.SCHEMA
        or postflight.get("human_audit_action") != "reject_only"
        or postflight.get("human_audit_may_authorize_route") is not False
        or postflight.get("remote_worker_execution_verified") is not False
        or postflight.get("observer_execution_authorized") is not False
        or postflight.get("scientific_claim_authorized") is not False
        or postflight.get("route_authorized") is not False
        or postflight.get("decode_authorized") is not False
        or postflight.get("training_authorized") is not False
        or postflight.get("file_sha256", {}).get("track_receipt")
        != file_sha256(track_path)
        or postflight.get("file_sha256", {}).get("assignment_result")
        != file_sha256(result_path)
        or postflight.get("file_sha256", {}).get("ownership_receipt")
        != file_sha256(ownership_root / "ownership_receipt.json")
        or postflight.get("file_sha256", {}).get("ownership_tensors")
        != file_sha256(ownership_root / "ownership.safetensors")
        or result.get("provenance", {}).get("source_video_sha256")
        != file_sha256(source)
        or set(result.get("assignments", {})) != set(core.ROLE_NAMES)
        or set(result.get("evidence", {})) != set(core.ROLE_NAMES)
        or set(result.get("competition", {})) != set(core.ROLE_NAMES)
        or ownership_receipt.get("raw_overlapping_proposals_passed_to_v15b") is not False
    ):
        raise BuildSourceFourRoleReviewV15CR9Error("reject-only authority differs")
    track = json.loads(track_path.read_text(encoding="utf-8"))
    proposal_ids = [row.get("proposal_id") for row in track.get("proposals", [])]
    if not proposal_ids or result.get("proposal_ids") != proposal_ids:
        raise BuildSourceFourRoleReviewV15CR9Error("proposal registry differs")
    frames = _decode_source(source)
    all_masks = []
    for proposal_id in proposal_ids:
        sequence = []
        for frame_index in range(81):
            value = cv2.imread(
                str(track_path.parent / "masks" / proposal_id / f"{frame_index:05d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if not isinstance(value, np.ndarray) or value.shape != (1056, 704):
                raise BuildSourceFourRoleReviewV15CR9Error("proposal mask differs")
            sequence.append(value > 0)
        all_masks.append(np.stack(sequence))
    tensors = load_file(str(ownership_root / "ownership.safetensors"))
    raw = tensors["raw_proposal_masks"] != 0
    owned = tensors["final_ownership_masks"] != 0
    unresolved = tensors["unassigned_occlusion_mask"] != 0
    contacts = {
        "old_actor": tensors["contact_human_old_actor"] != 0,
        "new_actor": tensors["contact_human_new_actor"] != 0,
        "recipient": tensors["contact_human_recipient"] != 0,
    }
    output.mkdir(mode=0o700, parents=True)
    media_root = output / "media"
    media_root.mkdir(mode=0o700)
    for key in VIEW_KEYS:
        video_path = output / MEDIA_CONTRACT[key]["video"]
        sheet_path = output / MEDIA_CONTRACT[key]["contact_sheet"]
        if key == "source":
            raw_views, ownership, contact, unassigned = [], None, None, None
        elif key == "all":
            raw_views, ownership, contact, unassigned = all_masks, None, None, unresolved
        else:
            role_index = core.ROLE_NAMES.index(key)
            raw_views = [raw[role_index]]
            ownership = owned[role_index]
            contact = (
                np.logical_or.reduce(list(contacts.values()))
                if key == "human_agent" else contacts[key]
            )
            unassigned = unresolved
        _write_video_and_sheet(
            frames=frames,
            label=key,
            raw_masks=raw_views,
            ownership_mask=ownership,
            contact_mask=contact,
            unassigned_mask=unassigned,
            video_path=video_path,
            sheet_path=sheet_path,
        )
    media_validation = {
        key: _reopen_video(output / MEDIA_CONTRACT[key]["video"])
        for key in VIEW_KEYS
    }
    media_payload = {
        "schema_version": MEDIA_SCHEMA,
        "view_keys": list(VIEW_KEYS),
        "layout_rows": [list(row) for row in LAYOUT_ROWS],
        "maximum_columns_per_row": max(len(row) for row in LAYOUT_ROWS),
        "display_frames": list(DISPLAY_FRAMES),
        "videos": media_validation,
        "all_media_gates_pass": True,
        "route_authorized": False,
    }
    media_payload["receipt_sha256"] = core.object_sha256(media_payload)
    media_path = output / "media_validation.json"
    media_path.write_bytes(core.canonical_bytes(media_payload))
    index_path = output / "index.html"
    index_path.write_text(_html(result), encoding="utf-8")
    files = {
        str(path.relative_to(output)): {
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "LOCAL_SYNCHRONIZED_SIX_VIEW_REJECT_ONLY_OVERLAY",
        "inputs": {
            "source_sha256": file_sha256(source),
            "track_receipt_sha256": file_sha256(track_path),
            "assignment_result_sha256": file_sha256(result_path),
            "ownership_receipt_sha256": file_sha256(
                ownership_root / "ownership_receipt.json"
            ),
            "ownership_tensors_sha256": file_sha256(
                ownership_root / "ownership.safetensors"
            ),
            "postflight_sha256": file_sha256(postflight_json),
        },
        "files": files,
        "media_contract": MEDIA_CONTRACT,
        "media_validation_file_sha256": file_sha256(media_path),
        "view_keys": list(VIEW_KEYS),
        "layout_rows": [list(row) for row in LAYOUT_ROWS],
        "maximum_columns_per_row": max(len(row) for row in LAYOUT_ROWS),
        "display_frames": list(DISPLAY_FRAMES),
        "all_six_contact_sheets_present": True,
        "raw_proposal_overlap_occlusion_evidence_visible": True,
        "final_ownership_visible": True,
        "independent_contact_relation_visible": True,
        "all_unassigned_rows_include_full_failure_evidence": True,
        "synchronized_playback": True,
        "human_audit_action": "reject_only",
        "approve_action_available": False,
        "threshold_mutation_available": False,
        "remote_worker_execution_verified": False,
        "observer_execution_authorized": False,
        "localization_semantically_certified": False,
        "scientific_claim_authorized": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    receipt["receipt_sha256"] = core.object_sha256(receipt)
    (output / "overlay_receipt.json").write_bytes(core.canonical_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--track-receipt", required=True, type=Path)
    parser.add_argument("--assignment-result", required=True, type=Path)
    parser.add_argument("--ownership-dir", required=True, type=Path)
    parser.add_argument("--postflight-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    receipt = build_review(
        source_video=args.source_video,
        track_receipt_path=args.track_receipt,
        assignment_result_path=args.assignment_result,
        ownership_dir=args.ownership_dir,
        postflight_path=args.postflight_json,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

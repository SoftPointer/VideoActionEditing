#!/usr/bin/env python3
"""Build a synchronized, reject-only audit page for v15c-r3 candidates."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import source_object_proposal_role_probe_v15c as core


SCHEMA = "bernini-source-object-proposal-role-overlay-v15c-r3"
POSTFLIGHT_SCHEMA = "bernini-source-sam2-proposal-role-postflight-v15c-r3"
POSTFLIGHT_GATE_KEYS = {
    "spec_raw_and_canonical_pins",
    "source_and_r6_pins",
    "track_receipt_exact_schema_and_self_hash",
    "track_output_and_artifact_manifests",
    "one_to_64_full_sha_sorted_proposals",
    "both_repeat_transcripts_rebuilt",
    "all_prompt_and_p_times_81_mask_bytes_reopened",
    "all_geometry_and_whole_object_gates_recomputed",
    "all_phase_coverage_recomputed",
    "all_logits_out_ids_shape_dtype_finite_order_evidence",
    "all_freeze_rng_repeat_evidence",
    "source_family_overlap_nesting_fail_closed",
    "r6_core_result_replayed",
}
POSTFLIGHT_KEYS = {
    "schema_version",
    "status",
    "gates",
    "file_sha256",
    "mechanical_candidate_qualified",
    "assignments_for_reject_only_audit",
    "human_audit_action",
    "human_audit_may_authorize_route",
    "localization_semantically_certified",
    "action_success_certified",
    "route_authorized",
    "decode_authorized",
    "training_authorized",
    "optimizer_updates",
    "renderer_forward_calls",
    "receipt_sha256",
}
DISPLAY_FRAMES = (0, 20, 40, 60, 80)
OVERLAY_RECEIPT_KEYS = (
    "schema_version",
    "status",
    "inputs",
    "files",
    "media_validation_receipt_sha256",
    "display_frames",
    "all_role_contact_sheets_present",
    "all_unassigned_rows_include_full_failure_evidence",
    "synchronized_playback",
    "human_audit_action",
    "approve_action_available",
    "threshold_mutation_available",
    "localization_semantically_certified",
    "route_authorized",
    "decode_authorized",
    "training_authorized",
    "receipt_sha256",
)


class BuildSourceProposalRoleReviewV15CR2Error(RuntimeError):
    """Reject-only overlay input/output closure differs."""


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise BuildSourceProposalRoleReviewV15CR2Error("JSON input differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise BuildSourceProposalRoleReviewV15CR2Error("JSON object differs")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str = "receipt_sha256") -> None:
    payload = dict(value)
    claimed = payload.pop(field, None)
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or any(character not in "0123456789abcdef" for character in claimed)
        or claimed != core.object_sha256(payload)
    ):
        raise BuildSourceProposalRoleReviewV15CR2Error("input self-hash differs")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_overlay(
    *,
    frames: list[Any],
    root: Path,
    proposal_ids: list[str],
    label: str,
    output: Path,
) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (704, 1056),
    )
    if not writer.isOpened():
        raise BuildSourceProposalRoleReviewV15CR2Error("overlay writer failed")
    colors = [
        (30, 220, 255),
        (255, 120, 30),
        (80, 255, 80),
        (255, 80, 190),
        (180, 100, 255),
        (60, 180, 255),
    ]
    for frame_index, source in enumerate(frames):
        canvas = source.copy()
        for proposal_index, proposal_id in enumerate(proposal_ids):
            mask_path = root / "masks" / proposal_id / f"{frame_index:05d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if not isinstance(mask, np.ndarray) or mask.shape != (1056, 704):
                raise BuildSourceProposalRoleReviewV15CR2Error("overlay mask differs")
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                canvas,
                contours,
                -1,
                colors[proposal_index % len(colors)],
                3,
            )
        cv2.rectangle(canvas, (0, 0), (704, 54), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            label,
            (12, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()


def _render_contact_sheet(
    *,
    frames: list[Any],
    root: Path,
    proposal_ids: list[str],
    label: str,
    output: Path,
) -> None:
    import cv2
    import numpy as np

    tiles = []
    colors = [(30, 220, 255), (255, 120, 30), (80, 255, 80), (255, 80, 190)]
    for frame_index in DISPLAY_FRAMES:
        canvas = frames[frame_index].copy()
        for proposal_index, proposal_id in enumerate(proposal_ids):
            mask = cv2.imread(
                str(root / "masks" / proposal_id / f"{frame_index:05d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if not isinstance(mask, np.ndarray) or mask.shape != (1056, 704):
                raise BuildSourceProposalRoleReviewV15CR2Error(
                    "contact-sheet mask differs"
                )
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                canvas, contours, -1, colors[proposal_index % len(colors)], 3
            )
        cv2.rectangle(canvas, (0, 0), (704, 58), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            f"{label} | f{frame_index:02d}",
            (10, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(cv2.resize(canvas, (352, 528), interpolation=cv2.INTER_AREA))
    sheet = np.concatenate(tiles, axis=1)
    if not cv2.imwrite(str(output), sheet):
        raise BuildSourceProposalRoleReviewV15CR2Error(
            "contact-sheet write failed"
        )
    reopened = cv2.imread(str(output), cv2.IMREAD_COLOR)
    if not isinstance(reopened, np.ndarray) or reopened.shape != (528, 1760, 3):
        raise BuildSourceProposalRoleReviewV15CR2Error(
            "contact-sheet reopen differs"
        )


def _reopen_video_receipt(path: Path) -> Mapping[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise BuildSourceProposalRoleReviewV15CR2Error("media reopen failed")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (1056, 704):
            capture.release()
            raise BuildSourceProposalRoleReviewV15CR2Error("media frame differs")
        frame_count += 1
    capture.release()
    gates = {
        "frame_count_81": frame_count == 81,
        "fps_25": abs(fps - 25.0) <= 1.0e-6,
        "width_704": width == 704,
        "height_1056": height == 1056,
    }
    if any(type(value) is not bool or value is not True for value in gates.values()):
        raise BuildSourceProposalRoleReviewV15CR2Error("media contract differs")
    return {
        "relative_path": str(path),
        "sha256": file_sha256(path),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "gates": gates,
    }


def _html(result: Mapping[str, Any], media: Mapping[str, Mapping[str, str]]) -> str:
    assignments = result["assignments"]
    evidence_sections = []
    for role in core.ROLE_NAMES:
        assigned = assignments[role]
        rows = result["evidence"][role]
        proposal_rows = []
        for row in rows:
            proposal_rows.append(
                "<tr>"
                f"<td><code>{html.escape(row['proposal_id'])}</code></td>"
                f"<td>{row['eligible_before_proposal_competition']}</td>"
                f"<td>{row['track_real']}</td>"
                f"<td>{row['track_shuffled']}</td>"
                f"<td>{row['proposal_max_null_required_quantile']}</td>"
                f"<td>{row['proposal_max_null_raw_upper_p']}</td>"
                f"<td>{row['three_role_bonferroni_fwer_upper_p']}</td>"
                f"<td>{row['consistent_phase_count']}/{row['longest_consistent_run']}</td>"
                f"<td>{row['real_over_permutation_phase_count']}</td>"
                f"<td><pre>{html.escape(json.dumps(row['source_family_overlap_or_nesting_neighbors'], sort_keys=True))}</pre></td>"
                f"<td><pre>{html.escape(json.dumps(row['gates'], sort_keys=True, indent=2))}</pre></td>"
                f"<td><details><summary>full evidence</summary><pre>{html.escape(json.dumps(row, sort_keys=True, indent=2))}</pre></details></td>"
                "</tr>"
            )
        competition = result["competition"][role]
        evidence_sections.append(
            f"<details open><summary>{html.escape(role)}: "
            f"{html.escape(assigned or 'UNASSIGNED')} — all proposal evidence</summary>"
            f"<pre class=competition>{html.escape(json.dumps(competition, sort_keys=True, indent=2))}</pre>"
            "<table><thead><tr><th>proposal</th><th>eligible</th><th>real</th>"
            "<th>shuffle</th><th>proposal-max null q</th><th>raw p</th>"
            "<th>3-role FWER</th><th>consistent/run</th><th>real&gt;perm phases</th>"
            "<th>family/overlap/nesting</th><th>all gates</th><th>complete row</th></tr></thead>"
            f"<tbody>{''.join(proposal_rows)}</tbody></table></details>"
        )
    cards = []
    for key, title in (
        ("source", "Source authority"),
        ("all", "All SAM2 proposals"),
        ("old_actor", "old_actor candidate"),
        ("new_actor", "new_actor candidate"),
        ("recipient", "recipient candidate"),
    ):
        cards.append(
            f'<section class="card"><h2>{html.escape(title)}</h2>'
            f'<video muted loop playsinline preload="metadata" src="{html.escape(media[key]["video"])}"></video>'
            f'<img loading="lazy" src="{html.escape(media[key]["sheet"])}" alt="frames 0 20 40 60 80">'
            "</section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>v15c-r3 reject-only audit</title>
<style>
body{{margin:18px;background:#f7f4ed;color:#17201d;font:16px system-ui,sans-serif}}
.bar{{position:sticky;top:0;background:#f7f4ed;padding:8px 0;z-index:3}}
button{{padding:10px 16px;margin:4px;border:1px solid #735f3f;border-radius:9px;background:#fffaf0;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:10px;align-items:start}}
.card{{border:1px solid #baad94;border-radius:10px;overflow:hidden;background:white}}
h2{{font-size:15px;min-height:38px;margin:0;padding:8px}} video,img{{display:block;width:100%;background:#111}}
table{{border-collapse:collapse;width:100%;margin-top:14px;background:white}}td,th{{border:1px solid #baad94;padding:8px;vertical-align:top}}pre{{white-space:pre-wrap;font-size:11px}}
.warning{{color:#8d261d;font-weight:800}} .reject{{border-color:#a52d23;color:#8d261d}}details{{margin-top:14px}}summary{{font-weight:800}}code{{font-size:10px}}
</style></head><body>
<h1>v15c-r3 source proposal / role observer</h1>
<p class="warning">Only rejection is allowed. Unassigned roles show every proposal's null, shuffle, temporal, competition and family-conflict evidence. No anchor or target instruction supplied material/transparency labels.</p>
<div class="bar"><button id="play">同步播放全部</button><button id="pause">全部暂停</button>
<button id="export">导出拒绝记录</button></div>
<div class="grid">{''.join(cards)}</div>
<h2>机械候选（不是 ground truth）</h2>
<div>{''.join(f'<button class="reject" data-role="{role}">Reject {role}</button>' for role in core.ROLE_NAMES)}</div>
{''.join(evidence_sections)}
<script>
const videos=[...document.querySelectorAll('video')];
document.getElementById('play').onclick=async()=>{{videos.forEach(v=>v.currentTime=0);await Promise.allSettled(videos.map(v=>v.play()));}};
document.getElementById('pause').onclick=()=>videos.forEach(v=>v.pause());
let rejects=JSON.parse(localStorage.getItem('v15c_r3_rejects')||'{{}}');
document.querySelectorAll('.reject').forEach(b=>{{const r=b.dataset.role;if(rejects[r])b.textContent='Rejected '+r;b.onclick=()=>{{rejects[r]={{rejected:true,time:new Date().toISOString()}};localStorage.setItem('v15c_r3_rejects',JSON.stringify(rejects));b.textContent='Rejected '+r;}}}});
document.getElementById('export').onclick=()=>{{const blob=new Blob([JSON.stringify({{schema_version:'v15c-r3-reject-only-human-audit',rejects,approve_action_available:false,route_authorized:false}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='v15c_r3_rejections.json';a.click();URL.revokeObjectURL(a.href);}};
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--track-receipt", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--postflight-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = args.source_video.resolve(strict=True)
    track_path = args.track_receipt.resolve(strict=True)
    result_path = args.result_json.resolve(strict=True)
    postflight_path = args.postflight_json.resolve(strict=True)
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise BuildSourceProposalRoleReviewV15CR2Error("review output is not fresh")
    track = read_json(track_path)
    result = read_json(result_path)
    postflight = read_json(postflight_path)
    verify_self_hash(track)
    verify_self_hash(result)
    verify_self_hash(postflight)
    if (
        postflight.get("schema_version")
        != POSTFLIGHT_SCHEMA
        or set(postflight) != POSTFLIGHT_KEYS
        or postflight.get("status") != "POSTFLIGHT_PASS_REJECT_ONLY_OVERLAY_PENDING"
        or not isinstance(postflight.get("gates"), Mapping)
        or not postflight["gates"]
        or set(postflight["gates"]) != POSTFLIGHT_GATE_KEYS
        or any(value is not True for value in postflight["gates"].values())
        or postflight.get("human_audit_action") != "reject_only"
        or postflight.get("human_audit_may_authorize_route") is not False
        or postflight.get("route_authorized") is not False
        or postflight.get("file_sha256", {}).get("source")
        != file_sha256(source)
        or postflight.get("file_sha256", {}).get("track_receipt")
        != file_sha256(track_path)
        or postflight.get("file_sha256", {}).get("result")
        != file_sha256(result_path)
        or result.get("route_authorized") is not False
        or result.get("training_authorized") is not False
    ):
        raise BuildSourceProposalRoleReviewV15CR2Error("reject-only authority differs")
    proposal_ids = [row.get("proposal_id") for row in track.get("proposals", [])]
    if (
        not proposal_ids
        or result.get("proposal_ids") != proposal_ids
        or set(result.get("assignments", {})) != set(core.ROLE_NAMES)
        or set(result.get("evidence", {})) != set(core.ROLE_NAMES)
        or set(result.get("competition", {})) != set(core.ROLE_NAMES)
    ):
        raise BuildSourceProposalRoleReviewV15CR2Error("role audit registry differs")
    for role in core.ROLE_NAMES:
        rows = result["evidence"][role]
        competition = result["competition"][role]
        if (
            type(rows) is not list
            or [row.get("proposal_id") for row in rows] != proposal_ids
            or type(competition) is not dict
            or not competition
            or not isinstance(competition.get("status"), str)
            or any(
                type(row) is not dict
                or type(row.get("gates")) is not dict
                or not row["gates"]
                or any(type(value) is not bool for value in row["gates"].values())
                or "track_real" not in row
                or "track_shuffled" not in row
                or "proposal_max_null_required_quantile" not in row
                or "proposal_max_null_raw_upper_p" not in row
                or "three_role_bonferroni_fwer_upper_p" not in row
                or "consistent_phase_count" not in row
                or "longest_consistent_run" not in row
                or "real_over_permutation_phase_count" not in row
                or "source_family_overlap_or_nesting_neighbors" not in row
                for row in rows
            )
            or (
                result["assignments"][role] is None
                and not competition["status"].startswith("unassigned_")
            )
        ):
            raise BuildSourceProposalRoleReviewV15CR2Error(
                "complete unassigned failure evidence differs"
            )

    import cv2

    capture = cv2.VideoCapture(str(source))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != 81 or any(frame.shape[:2] != (1056, 704) for frame in frames):
        raise BuildSourceProposalRoleReviewV15CR2Error("source decode differs")

    output.mkdir(mode=0o700, parents=True)
    media_root = output / "media"
    media_root.mkdir(mode=0o700)
    source_copy = media_root / "source.mp4"
    shutil.copy2(source, source_copy)
    media = {}
    source_sheet = media_root / "source_f00_20_40_60_80.jpg"
    _render_contact_sheet(
        frames=frames,
        root=track_path.parent,
        proposal_ids=[],
        label="SOURCE AUTHORITY",
        output=source_sheet,
    )
    media["source"] = {
        "video": "media/source.mp4",
        "sheet": "media/source_f00_20_40_60_80.jpg",
    }
    all_ids = proposal_ids
    all_path = media_root / "all_proposals.mp4"
    _render_overlay(
        frames=frames,
        root=track_path.parent,
        proposal_ids=all_ids,
        label=f"ALL PROPOSALS ({len(all_ids)}); NOT SEMANTIC GT",
        output=all_path,
    )
    all_sheet = media_root / "all_proposals_f00_20_40_60_80.jpg"
    _render_contact_sheet(
        frames=frames,
        root=track_path.parent,
        proposal_ids=all_ids,
        label=f"ALL {len(all_ids)} SOURCE PROPOSALS",
        output=all_sheet,
    )
    media["all"] = {
        "video": "media/all_proposals.mp4",
        "sheet": "media/all_proposals_f00_20_40_60_80.jpg",
    }
    for role in core.ROLE_NAMES:
        candidate = result["assignments"][role]
        role_path = media_root / f"{role}.mp4"
        _render_overlay(
            frames=frames,
            root=track_path.parent,
            proposal_ids=[] if candidate is None else [candidate],
            label=f"{role}: {candidate or 'UNASSIGNED'}; REJECT ONLY",
            output=role_path,
        )
        role_sheet = media_root / f"{role}_f00_20_40_60_80.jpg"
        _render_contact_sheet(
            frames=frames,
            root=track_path.parent,
            proposal_ids=[] if candidate is None else [candidate],
            label=f"{role}: {candidate or 'UNASSIGNED'}; REJECT ONLY",
            output=role_sheet,
        )
        media[role] = {
            "video": f"media/{role}.mp4",
            "sheet": f"media/{role}_f00_20_40_60_80.jpg",
        }
    media_validation = {}
    for key in ("source", "all", *core.ROLE_NAMES):
        relative = media[key]["video"]
        reopened = dict(_reopen_video_receipt(output / relative))
        reopened["relative_path"] = relative
        media_validation[key] = reopened
    media_validation_path = output / "media_validation.json"
    media_validation_payload = {
        "schema_version": "bernini-source-object-proposal-role-media-validation-v15c-r3",
        "required_contract": {
            "frame_count": 81,
            "fps": 25.0,
            "width": 704,
            "height": 1056,
        },
        "display_frames": list(DISPLAY_FRAMES),
        "videos": media_validation,
        "all_media_gates_pass": True,
    }
    media_validation_payload["receipt_sha256"] = core.object_sha256(
        media_validation_payload
    )
    media_validation_path.write_bytes(core.canonical_bytes(media_validation_payload))
    index = output / "index.html"
    index.write_text(_html(result, media), encoding="utf-8")
    files = {
        str(path.relative_to(output)): {
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    receipt = {
        "schema_version": SCHEMA,
        "status": "SYNCHRONIZED_REJECT_ONLY_OVERLAY_COMPLETE",
        "inputs": {
            "source_sha256": file_sha256(source),
            "track_receipt_sha256": file_sha256(track_path),
            "result_sha256": file_sha256(result_path),
            "postflight_sha256": file_sha256(postflight_path),
        },
        "files": files,
        "media_validation_receipt_sha256": file_sha256(media_validation_path),
        "display_frames": list(DISPLAY_FRAMES),
        "all_role_contact_sheets_present": True,
        "all_unassigned_rows_include_full_failure_evidence": True,
        "synchronized_playback": True,
        "human_audit_action": "reject_only",
        "approve_action_available": False,
        "threshold_mutation_available": False,
        "localization_semantically_certified": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    receipt["receipt_sha256"] = core.object_sha256(receipt)
    if set(receipt) != set(OVERLAY_RECEIPT_KEYS):
        raise BuildSourceProposalRoleReviewV15CR2Error("overlay receipt keys differ")
    for digest in (
        receipt["receipt_sha256"],
        receipt["media_validation_receipt_sha256"],
        *receipt["inputs"].values(),
        *(row["sha256"] for row in receipt["files"].values()),
    ):
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise BuildSourceProposalRoleReviewV15CR2Error("overlay SHA differs")
    (output / "overlay_receipt.json").write_bytes(core.canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the fail-closed local review page for the MEV840 QK diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "md/action_editing/20260821_mev840_action_repr_diagnostic_review"
MEDIA = ROOT / "media"
PROVENANCE = ROOT / "provenance"
SHEETS = ROOT / "contact_sheets"

EXPECTED_MEDIA = {
    "source.mp4": "ed8ced7ec6aa90a0ba0bee4db981d20a65fe6eb435771f818c81c6e5f3eebe55",
    "real_target.mp4": "355535f4f5ff83581c2286dfb70a64c7f5131f5ae81d76fbb6351b2aa972baf0",
    "self_anchor.mp4": "b2b3176ddf92a14dccfa49b1e4c8f4234a041df7a5827955d7941d4b1c0ed352",
    "native_frozen.mp4": "7a3da607ab381077c50c1daca5b1ef8e40b816f4743b17e7f92c8daf8feb77b7",
    "routeoff.mp4": "741ce1cc98a3ea9080f548dd88bf7670b96272b01cf90e579990789a44d7a302",
    "self_qk.mp4": "5a1662e50718e7d3f55eb0c40d801d621a103fdb7d6080461368ddb5d45e083b",
    "oracle_qk.mp4": "a56afd644dbdf211ea08c124b27e61d3e298bc25e37f583c8c82ab5ee157b06b",
}

RESULTS = {
    "routeoff": {
        "media": "routeoff.mp4",
        "stem": "MEV840_LEGACY_QK_MATCHED_ROUTEOFF_K0.mp4",
        "steps": 0,
        "anchor_sha": "412399051be25c19ec9ab7d1406b1e6982e31e75cfa1b3920259a6c22f66113b",
    },
    "self": {
        "media": "self_qk.mp4",
        "stem": "MEV840_LEGACY_QK_SELF_ANCHOR_ACTIVITY25_K40.mp4",
        "steps": 40,
        "anchor_sha": "412399051be25c19ec9ab7d1406b1e6982e31e75cfa1b3920259a6c22f66113b",
    },
    "oracle": {
        "media": "oracle_qk.mp4",
        "stem": "MEV840_LEGACY_QK_RETIMED_REAL_TARGET_ORACLE_ACTIVITY25_K40.mp4",
        "steps": 40,
        "anchor_sha": "355535f4f5ff83581c2286dfb70a64c7f5131f5ae81d76fbb6351b2aa972baf0",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"plain file required: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(plain(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    row = json.loads(completed.stdout)["streams"][0]
    result = {
        "width": int(row["width"]),
        "height": int(row["height"]),
        "fps": row["r_frame_rate"],
        "frames": int(row["nb_read_frames"]),
    }
    if result["fps"] != "25/1" or result["frames"] != 81:
        raise RuntimeError(f"81f/25fps contract differs: {path}: {result}")
    return result


def validate() -> dict[str, Any]:
    media_rows: dict[str, Any] = {}
    for name, expected in EXPECTED_MEDIA.items():
        path = plain(MEDIA / name)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"media SHA differs: {name}: {observed}")
        media_rows[name] = {"sha256": observed, **probe(path)}

    result_rows: dict[str, Any] = {}
    for arm, spec in RESULTS.items():
        receipt_path = PROVENANCE / f"{spec['stem']}.receipt.json"
        audit_path = PROVENANCE / f"{spec['stem']}.legacy-mev840-audit.json"
        complete_path = PROVENANCE / f"{spec['stem']}.complete.json"
        receipt = read_json(receipt_path)
        audit = read_json(audit_path)
        complete = read_json(complete_path)
        media_sha = EXPECTED_MEDIA[spec["media"]]
        if not (
            receipt.get("complete") is True
            and receipt.get("training_performed") is False
            and receipt.get("optimization_steps") == 0
            and receipt.get("loaded_trained_attention_checkpoint") is False
            and receipt.get("trained_attention_checkpoint") is None
            and receipt["source"]["sha256"]
            == "a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646"
            and receipt["pure_t2v_anchor"]["sha256"] == spec["anchor_sha"]
            and receipt["output"]["sha256"] == media_sha
            and receipt["mechanism"]["initial_noise_proposal_mode"] == "keyed_only"
            and receipt["mechanism"]["anchor_contrast_mode"] == "dynamic_static_same_caption"
            and receipt["mechanism"]["decode_audit_contract"]["transport_steps"] == spec["steps"]
            and receipt["freeze_before"] == receipt["freeze_after"]
            and receipt["freeze_before"]["trainable_parameter_tensors"] == 0
            and receipt["freeze_before"]["lora_module_count"] == 0
        ):
            raise RuntimeError(f"native receipt contract differs: {arm}")
        if not (
            audit.get("complete") is True
            and audit.get("zero_update") is True
            and audit["output"]["sha256"] == media_sha
            and audit["native_receipt"]["sha256"] == sha256(receipt_path)
            and audit["postflight_recovery"] == {
                "required": True,
                "reason": "compute node lacked ffprobe",
                "inference_was_not_rerun": True,
                "original_step_exit_code": 127,
            }
        ):
            raise RuntimeError(f"arm audit contract differs: {arm}")
        if not (
            complete.get("complete") is True
            and complete["output"]["sha256"] == media_sha
            and complete["audit"]["sha256"] == sha256(audit_path)
        ):
            raise RuntimeError(f"complete marker differs: {arm}")
        trace = receipt["mechanism"]["trace"]
        cache = trace["attention_cache"]
        expected = (0, 0, 0, 0) if spec["steps"] == 0 else (52, 104, 2288, 4576)
        observed = (
            trace["anchor_candidate_cells"],
            trace["anchor_model_forwards"],
            cache["qk_only_capture_count"],
            cache["qk_only_replay_count"],
        )
        if observed != expected or cache["pending_entries"] != 0:
            raise RuntimeError(f"QK trace closure differs: {arm}: {observed}")
        result_rows[arm] = {
            "media": spec["media"],
            "media_sha256": media_sha,
            "receipt": receipt_path.name,
            "receipt_sha256": sha256(receipt_path),
            "audit": audit_path.name,
            "audit_sha256": sha256(audit_path),
            "complete": complete_path.name,
            "complete_sha256": sha256(complete_path),
            "transport_steps": spec["steps"],
            "trace": {
                "candidate_cells": observed[0],
                "anchor_forwards": observed[1],
                "qk_captures": observed[2],
                "qk_replays": observed[3],
                "pending": 0,
            },
        }
    anchor_audit_path = PROVENANCE / "mev840_action_graph_multiappearance_anchor_execution_20260821.json"
    anchor_audit = read_json(anchor_audit_path)
    anchor_audit_sha = sha256(anchor_audit_path)
    if not (
        anchor_audit_sha == "303fdbc53d6191c285fe3a7fc390133844411248562720913359e6f9ea5a154f"
        and anchor_audit.get("terminal_status")
        == "blocked_by_per_node_host_memory_under_strict_sealed_runner"
        and anchor_audit.get("anchor_outputs_created") is False
        and anchor_audit.get("training_performed") is False
        and anchor_audit.get("parameter_updates") == 0
        and anchor_audit.get("real_target_read") is False
        and anchor_audit.get("r3_launched") is False
        and [row["state"] for row in anchor_audit["attempts"][1]["steps"]]
        == ["OUT_OF_MEMORY", "OUT_OF_MEMORY", "OUT_OF_MEMORY"]
    ):
        raise RuntimeError("multiappearance anchor-bank execution audit differs")
    return {
        "media": media_rows,
        "results": result_rows,
        "multiappearance_anchor_bank_attempt": {
            "audit": anchor_audit_path.name,
            "audit_sha256": anchor_audit_sha,
            "status": anchor_audit["terminal_status"],
            "anchor_outputs_created": False,
            "training_performed": False,
            "real_target_read": False,
            "reason": "All available allocations expose 64G host RAM; the strict sealed T2V runner OOMed before sampling.",
        },
    }


def card(title: str, badge: str, media: str, sheet: str, note: str, tone: str = "") -> str:
    return f"""
      <article class="card {tone}">
        <header><span class="badge">{badge}</span><h3>{title}</h3></header>
        <div class="video-shell"><video controls muted loop playsinline preload="metadata" data-sync src="media/{media}"></video></div>
        <img class="sheet" src="contact_sheets/{sheet}" alt="{title} frames 0,20,40,60,80">
        <p>{note}</p>
      </article>"""


def build_html(manifest_sha: str) -> str:
    cards_top = "".join(
        [
            card("Source", "input", "source.mp4", "source_f00_20_40_60_80.jpg", "Source identity and object authority: woman holds the same transparent bottle; no requested action."),
            card("Real target", "evaluation only", "real_target.mp4", "real_target_f00_20_40_60_80.jpg", "Same clip continuation: turn, place bottle on treadmill, release. Never used by the self arm."),
            card("Self-generated anchor", "T2V signal", "self_anchor.mp4", "self_anchor_f00_20_40_60_80.jpg", "Not a clean action exemplar: the person begins side-facing; the bottle appears only late instead of moving continuously from the hand.", "warn"),
            card("Native frozen RV2V", "legacy reference", "native_frozen.mp4", "native_frozen_f00_20_40_60_80.jpg", "Appearance is preserved, but the output stays close to the source and does not place the bottle."),
        ]
    )
    cards_bottom = "".join(
        [
            card("Matched route-off", "K0 · zero update", "routeoff.mp4", "routeoff_f00_20_40_60_80.jpg", "Rough late placement is already present without anchor Q/K: the hand/bottle morph reaches the treadmill around f65–75 and an upright bottle is released by f75–80, with late ghosting.", "partial"),
            card("Self-anchor Q/K", "activity25 · K40", "self_qk.mp4", "self_qk_f00_20_40_60_80.jpg", "Strict fail: severe face/body/hand/bottle ghosting begins at frame 1; no reliable placement.", "fail"),
            card("Real-target oracle Q/K", "oracle · activity25 · K40", "oracle_qk.mp4", "oracle_qk_f00_20_40_60_80.jpg", "Strict fail: even the real target cannot transfer the placement through this old operator; collapse resembles the self arm.", "fail"),
        ]
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MEV840 action representation diagnostic</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#9da7b3;--blue:#58a6ff;--red:#ff7b72;--amber:#d29922;--green:#3fb950}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1680px;margin:auto;padding:26px}}h1{{margin:0 0 8px;font-size:28px}}h2{{margin:26px 0 12px;font-size:19px}}p{{color:var(--muted)}}
.callout{{border:1px solid var(--red);background:#2a1517;padding:14px 16px;border-radius:10px}}.callout strong{{color:#ffaaa4}}
.contract{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin:14px 0}}.pill{{background:var(--panel);border:1px solid var(--line);padding:9px;border-radius:8px}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}}button{{background:#21262d;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:7px 10px;cursor:pointer}}button:hover{{border-color:var(--blue)}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}}.grid.bottom{{grid-template-columns:repeat(3,minmax(0,1fr))}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}}.card.fail{{border-color:#7d3030}}.card.warn{{border-color:#6e571e}}.card.partial{{border-color:#315d7d}}
.card header{{padding:10px 12px 8px}}h3{{font-size:15px;margin:6px 0 0}}.badge{{font-size:11px;text-transform:uppercase;color:var(--blue);letter-spacing:.06em}}
.video-shell{{aspect-ratio:16/9;background:#000;display:flex;align-items:center}}video{{width:100%;height:100%;object-fit:contain}}.sheet{{display:block;width:100%;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}.card p{{padding:0 12px 10px;min-height:60px}}
table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{border:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}}th{{color:var(--muted)}}.pass{{color:var(--green)}}.failtxt{{color:var(--red)}}code{{color:#c9d1d9}}
.foot{{font-size:12px;margin-top:18px}}a{{color:var(--blue)}}
@media(max-width:1100px){{.grid,.grid.bottom,.contract{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:650px){{.grid,.grid.bottom,.contract{{grid-template-columns:1fr}}main{{padding:14px}}}}
</style></head><body><main>
<h1>MEV840 · self-generated action representation diagnostic</h1>
<p>Instruction: <strong>the woman turns her head left and places the same water bottle onto the treadmill</strong>.</p>
<div class="callout"><strong>结论：当前 legacy Q/K operating point 失败。</strong> Matched route-off 本身已粗糙完成 late placement（f65–75 接近跑步机，f75–80 直立释放，但有 ghost/morph）；self-anchor 与 real-target oracle 在 <code>activity25 · strength 1 · K40</code> 都从 f1 开始严重崩坏，既没有超过 route-off，也无法证明增量 action transfer。这里没有训练或参数更新。</div>
<div class="contract"><div class="pill"><b>Noise</b><br>target-owned keyed_only</div><div class="pill"><b>Training</b><br>0 update · frozen base</div><div class="pill"><b>Route</b><br>current target V; donor Q/K only</div><div class="pill"><b>Oracle boundary</b><br>real target used only in oracle arm</div></div>
<div class="controls"><button data-action="play">Play all</button><button data-action="pause">Pause all</button><button data-frame="0">f0</button><button data-frame="20">f20</button><button data-frame="40">f40</button><button data-frame="60">f60</button><button data-frame="80">f80</button></div>
<h2>Inputs and references</h2><section class="grid">{cards_top}</section>
<h2>Matched zero-update outputs</h2><section class="grid bottom">{cards_bottom}</section>
<h2>Strict event audit</h2>
<table><thead><tr><th>Arm</th><th>Turn</th><th>Same-bottle placement</th><th>Property stability</th><th>Interpretation</th></tr></thead><tbody>
<tr><td>Real target</td><td class="pass">yes</td><td class="pass">yes; contact ≈f70, hand releases by f80</td><td class="pass">yes</td><td>Evaluation reference</td></tr>
<tr><td>Native frozen</td><td>no</td><td class="failtxt">no</td><td class="pass">strong</td><td>Appearance-only reference; essentially source replay</td></tr>
<tr><td>Route-off K0</td><td class="pass">yes, but the whole upper body turns</td><td class="pass">rough yes; hand/bottle morph f65–75, upright release f75–80</td><td>degrades late; ghosting</td><td>The rough full event is already present without anchor signal; gain must be incremental beyond this control</td></tr>
<tr><td>Self Q/K K40</td><td class="failtxt">unjudgeable</td><td class="failtxt">no</td><td class="failtxt">severe f1+ ghosting</td><td>No evidence that self action signal helps</td></tr>
<tr><td>Oracle Q/K K40</td><td class="failtxt">unjudgeable</td><td class="failtxt">no</td><td class="failtxt">severe f1+ ghosting</td><td>Old operator fails its oracle-capacity test</td></tr>
</tbody></table>
<h2>What this does—and does not—establish</h2>
<ul><li><b>Established:</b> broad feature leakage/over-injection affects face, clothing, hands, bottle and nearby background.</li><li><b>Not established:</b> a clean semantic copy of the anchor's sleeveless outfit, white-cap bottle or pillar colors. The corruption is broader than literal RGB/V copying.</li><li><b>Multi-appearance bank:</b> the strict sealed v1–v3 T2V generation reached no sampling step: every available allocation exposes 64G host RAM and the old runner OOMed during model/text-encoder load. No anchor, training, route or real-target read occurred. <a href="provenance/mev840_action_graph_multiappearance_anchor_execution_20260821.json">execution audit</a>.</li><li><b>Next gate:</b> first require a low-strength oracle route to perform the event without collapse; only then compare multi-appearance self anchors and a spatially removed role-relation graph.</li></ul>
<p class="foot">All three new outputs are 656×368, 81 frames, 25 fps. Native receipts record <code>training_performed=false</code>, <code>optimization_steps=0</code>. Route-off trace is 0/0; self/oracle each record 104 anchor forwards, 2288 Q/K captures, 4576 replays, pending=0. The original compute postflight exited 127 only because ffprobe was absent; login-node recovery re-opened artifacts and did not rerun inference. Manifest SHA: <code>{manifest_sha}</code>. <a href="manifest.json">manifest</a> · <a href="provenance/MEV840_LEGACY_QK_SELF_ANCHOR_ACTIVITY25_K40.mp4.receipt.json">self receipt</a> · <a href="provenance/MEV840_LEGACY_QK_RETIMED_REAL_TARGET_ORACLE_ACTIVITY25_K40.mp4.receipt.json">oracle receipt</a></p>
</main><script>
const videos=[...document.querySelectorAll('video[data-sync]')];
document.querySelector('[data-action="play"]').onclick=()=>videos.forEach(v=>v.play());
document.querySelector('[data-action="pause"]').onclick=()=>videos.forEach(v=>v.pause());
document.querySelectorAll('[data-frame]').forEach(b=>b.onclick=()=>{{const t=Number(b.dataset.frame)/25;videos.forEach(v=>{{v.pause();v.currentTime=t}})}});
</script></body></html>"""


def main() -> int:
    validated = validate()
    sheet_rows = {}
    for media_name in EXPECTED_MEDIA:
        stem = Path(media_name).stem
        sheet = plain(SHEETS / f"{stem}_f00_20_40_60_80.jpg")
        sheet_rows[sheet.name] = {"sha256": sha256(sheet), "size": sheet.stat().st_size}
    manifest = {
        "schema": "mev840-action-representation-diagnostic-review-v1",
        "date": "2026-08-21",
        "status": "LEGACY_QK_OPERATING_POINT_STRICT_FAIL",
        "claim_boundary": {
            "new_role_graph_validated": False,
            "legacy_dynamic_static_qk_diagnostic": True,
            "training_or_parameter_update": False,
            "real_target_read_by_self_arm": False,
            "oracle_arm_reads_retimed_real_target": True,
        },
        "visual_audit": {
            "real_target_action_success": True,
            "native_frozen_action_success": False,
            "routeoff_action_success": True,
            "routeoff_action_quality": "rough late placement with hand/bottle morph around f65-f75 and upright release by f75-f80; late ghosting",
            "self_qk_action_success": False,
            "oracle_qk_action_success": False,
            "self_anchor_clean_action_exemplar": False,
            "conclusion": "Routeoff already contains the rough late placement. Old activity25 strength1 K40 collapses even with the real-target oracle and provides no incremental action gain; self QK is harmful at this operating point.",
        },
        **validated,
        "contact_sheets": sheet_rows,
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha = sha256(manifest_path)
    (ROOT / "index.html").write_text(build_html(manifest_sha), encoding="utf-8")
    print(json.dumps({"manifest_sha256": manifest_sha, "index_sha256": sha256(ROOT / "index.html")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the fail-closed local review for the MEV840 legacy oracle sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "md/action_editing/20260821_mev840_legacy_oracle_lowstrength_sweep_review"
MEDIA = ROOT / "media"
PROVENANCE = ROOT / "provenance"
CONTACT = ROOT / "contact"
SOURCE_SHA = "a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646"
ORACLE_SHA = "355535f4f5ff83581c2286dfb70a64c7f5131f5ae81d76fbb6351b2aa972baf0"
SCHEDULE_SHA = "43cd53329945280dccea5c1a1aa3b5da05337a7f10cfec0ab5a727592ea77d25"
LAUNCHER_SHA = "20a132a82c502d1b4f557ce37f2e435da579594aa970290e00726de8cf4252e3"
FINALIZER_SHA = "0760171b93e0df4be752cb3dab4ccb85e3922bb1244547840cec0b5a04d51fb9"
RUN_MANIFEST_SHA = "24230cf087816a33c86a5c5846f92ed0b5519694272e99febda782fdd53c2b7f"
METRICS_SHA = "5ab24d5c23740b1f02864345e7dab2ac5bdc693f30f04e9af665d90bc4d7d2ea"

EXPECTED_MEDIA = {
    "source.mp4": ("ed8ced7ec6aa90a0ba0bee4db981d20a65fe6eb435771f818c81c6e5f3eebe55", 960, 540),
    "real_target.mp4": (ORACLE_SHA, 960, 540),
    "native_frozen.mp4": ("7a3da607ab381077c50c1daca5b1ef8e40b816f4743b17e7f92c8daf8feb77b7", 960, 538),
    "routeoff.mp4": ("741ce1cc98a3ea9080f548dd88bf7670b96272b01cf90e579990789a44d7a302", 656, 368),
    "oracle_qk.mp4": ("a56afd644dbdf211ea08c124b27e61d3e298bc25e37f583c8c82ab5ee157b06b", 656, 368),
    "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S005_K05.mp4": ("13aa54d70bb11eb40057ec1883972e681d53be2b37562ad0158b1dd70c736d0f", 656, 368),
    "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S010_K05.mp4": ("b14c85bedff918dd3c5d4f7301cf6b106958246242c390921c79511a3d0282d6", 656, 368),
    "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S025_K05.mp4": ("c0b781a0dae4c6eec2d5d9a68e2ff08bde09d30707e4615e26eed0699e241dcc", 656, 368),
    "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S005_K10.mp4": ("28e6657fc96d3b91e0d78e1b956a37f8c7ff9584c0d003531e755d39b1d52869", 656, 368),
    "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S010_K10.mp4": ("eb0250f656acad43d594343b1dad6674f6af471a93e76f98558578dcd5db3758", 656, 368),
    "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S025_K10.mp4": ("1ecdd4291bb507418e90937d53724a54407d3dd0379596580da78a12519bfff2", 656, 368),
}

ARMS = (
    {"arm": "s005_k05", "label": "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S005_K05", "strength": 0.05, "steps": 5, "job": "147881", "step": "2", "job_step": "147881.2", "node": "auh7-1b-gpu-213", "ssim_routeoff": 0.976003, "ssim_target": 0.746350},
    {"arm": "s010_k05", "label": "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S010_K05", "strength": 0.10, "steps": 5, "job": "147873", "step": "2", "job_step": "147873.2", "node": "auh7-1b-gpu-284", "ssim_routeoff": 0.959710, "ssim_target": 0.746249},
    {"arm": "s025_k05", "label": "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S025_K05", "strength": 0.25, "steps": 5, "job": "147871", "step": "2", "job_step": "147871.2", "node": "auh7-1b-gpu-232", "ssim_routeoff": 0.911442, "ssim_target": 0.741773},
    {"arm": "s005_k10", "label": "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S005_K10", "strength": 0.05, "steps": 10, "job": "143808", "step": "477", "job_step": "143808.477", "node": "auh7-1b-gpu-268", "ssim_routeoff": 0.969398, "ssim_target": 0.746464},
    {"arm": "s010_k10", "label": "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S010_K10", "strength": 0.10, "steps": 10, "job": "143808", "step": "476", "job_step": "143808.476", "node": "auh7-1b-gpu-315", "ssim_routeoff": 0.949832, "ssim_target": 0.746884},
    {"arm": "s025_k10", "label": "MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S025_K10", "strength": 0.25, "steps": 10, "job": "143808", "step": "478", "job_step": "143808.478", "node": "auh7-1b-gpu-233", "ssim_routeoff": 0.908613, "ssim_target": 0.741873},
)

COMPOSITE_HASHES = {
    "reference_grid.jpg": "c4a2b14e99c1738b9e0e94f157b72ea88f4cf25cffa008621e6ea2361fd5ab47",
    "six_arm_grid.jpg": "cc48ab40756d70c5979f1dfafc62419202336ad11937f216e5f1892c8b540415",
    "six_arm_early_and_phase_grid.jpg": "5220416a5731ebabf106bc8875ab7be18053fa5fe0354580410b2bbdedddd38e",
    "routeoff_and_sweep_late_event_grid.jpg": "c06fb1ec1d01a73f21a7bd9f4b79b4a50623d3aa5633d7e643bc21127a50459c",
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


def probe_video(path: Path) -> dict[str, Any]:
    run = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    row = json.loads(run.stdout)["streams"][0]
    return {"width": int(row["width"]), "height": int(row["height"]),
            "fps": row["r_frame_rate"], "frames": int(row["nb_read_frames"])}


def probe_image(path: Path) -> tuple[int, int]:
    run = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    row = json.loads(run.stdout)["streams"][0]
    return int(row["width"]), int(row["height"])


def expected_contact_names() -> set[str]:
    result = {f"{Path(name).stem}_f0_20_40_60_80.jpg" for name in EXPECTED_MEDIA}
    labels = [arm["label"] for arm in ARMS]
    result.update(f"{label}_early_and_phase.jpg" for label in labels)
    result.update(f"{label}_late_f60_65_70_75_80.jpg" for label in labels)
    result.add("routeoff_late_f60_65_70_75_80.jpg")
    result.update(COMPOSITE_HASHES)
    return result


def validate() -> dict[str, Any]:
    media_rows: dict[str, Any] = {}
    for name, (expected_sha, width, height) in EXPECTED_MEDIA.items():
        path = plain(MEDIA / name)
        observed_sha = sha256(path)
        observed_probe = probe_video(path)
        if observed_sha != expected_sha:
            raise RuntimeError(f"media SHA differs: {name}: {observed_sha}")
        if observed_probe != {"width": width, "height": height, "fps": "25/1", "frames": 81}:
            raise RuntimeError(f"media probe differs: {name}: {observed_probe}")
        media_rows[name] = {"sha256": observed_sha, **observed_probe}

    sealed = {
        "auh_launch_mev840_legacy_oracle_lowstrength_sweep_v1.sh": LAUNCHER_SHA,
        "auh_finalize_mev840_legacy_oracle_lowstrength_sweep_v1.sh": FINALIZER_SHA,
        "mev840_legacy_oracle_lowstrength_sweep_v1.json": RUN_MANIFEST_SHA,
    }
    provenance_authority = {}
    for name, expected_sha in sealed.items():
        path = plain(PROVENANCE / name)
        observed = sha256(path)
        if observed != expected_sha:
            raise RuntimeError(f"sealed authority differs: {name}: {observed}")
        provenance_authority[name] = observed

    metrics_path = plain(ROOT / "metrics.json")
    if sha256(metrics_path) != METRICS_SHA:
        raise RuntimeError("metrics SHA differs")
    metrics = read_json(metrics_path)
    visual = metrics.get("preliminary_visual_read", {})
    if not (
        visual.get("routeoff_rough_late_placement_already_present") is True
        and visual.get("strength_005_and_010_reproduce_routeoff_placement") is True
        and visual.get("strength_025_duplicate_bottle_instance_drift") is True
        and visual.get("incremental_oracle_action_gain_in_any_arm") is False
    ):
        raise RuntimeError("corrected visual conclusion differs")
    metric_by_arm = {row["arm"]: row for row in metrics.get("arms", [])}

    arm_rows: dict[str, Any] = {}
    forbidden = ["value", "hidden_state", "attention_output", "rgb", "latent", "absolute_spatial_coordinate"]
    for arm in ARMS:
        name = f"{arm['label']}.mp4"
        stem = name
        receipt_path = PROVENANCE / f"{stem}.receipt.json"
        worker_path = PROVENANCE / f"{stem}.worker.json"
        audit_path = PROVENANCE / f"{stem}.legacy-mev840-lowstrength-audit.json"
        complete_path = PROVENANCE / f"{stem}.complete.json"
        receipt = read_json(receipt_path)
        worker = read_json(worker_path)
        audit = read_json(audit_path)
        complete = read_json(complete_path)
        output_sha = EXPECTED_MEDIA[name][0]
        receipt_sha, worker_sha, audit_sha = sha256(receipt_path), sha256(worker_path), sha256(audit_path)
        cells = 17 if arm["steps"] == 5 else 22
        forwards, captures, replays = cells * 2, cells * 2 * 22, cells * 2 * 22 * 2
        mechanism, trace = receipt["mechanism"], receipt["mechanism"]["trace"]
        cache = trace["attention_cache"]
        if not (
            receipt.get("complete") is True
            and receipt.get("training_performed") is False
            and receipt.get("optimization_steps") == 0
            and receipt.get("loaded_trained_attention_checkpoint") is False
            and receipt.get("trained_attention_checkpoint") is None
            and receipt.get("anchor_generation_initial_gaussian") is None
            and receipt["source"]["sha256"] == SOURCE_SHA
            and receipt["pure_t2v_anchor"]["sha256"] == ORACLE_SHA
            and receipt["output"]["sha256"] == output_sha
            and mechanism["transport"] == "self_target_owned_activity_kernel25_attn_output_v14r2"
            and mechanism["transport_strength"] == arm["strength"]
            and mechanism["decode_audit_contract"]["transport_steps"] == arm["steps"]
            and mechanism["initial_noise_proposal_mode"] == "keyed_only"
            and mechanism["anchor_state_mode"] == "clean_noised"
            and mechanism["anchor_cfg_scope"] == "shared"
            and mechanism["anchor_contrast_mode"] == "dynamic_static_same_caption"
            and mechanism["field_guidance"] == "raw_cfg"
            and mechanism["field_model"] == "first_phase_caption_i2v"
            and mechanism["source_cfg_scale"] == mechanism["target_cfg_scale"] == 4.5
            and mechanism["preservation_mode"] == "none"
            and trace["outer_schedule_digest"] == SCHEDULE_SHA
            and [row["step_index"] for row in trace["anchor_active_schedule"]] == list(range(arm["steps"]))
            and trace["anchor_candidate_cells"] == cells
            and trace["anchor_model_forwards"] == forwards
            and cache["qk_only_capture_count"] == captures
            and cache["qk_only_replay_count"] == replays
            and cache["pending_entries"] == 0
            and cache["qk_only_forbidden_cached_fields"] == forbidden
            and receipt["freeze_before"] == receipt["freeze_after"]
            and receipt["freeze_before"]["base_frozen"] is True
            and receipt["freeze_before"]["trainable_parameter_tensors"] == 0
            and len({row["latent"]["content_sha256"] for row in receipt["rank_closure"]}) == 1
            and len({row["trace_digest"] for row in receipt["rank_closure"]}) == 1
        ):
            raise RuntimeError(f"native receipt closure differs: {arm['arm']}")
        if not (
            worker.get("complete") is True and worker.get("zero_update") is True
            and worker.get("training_performed") is False and worker.get("optimization_steps") == 0
            and worker["arm"] == arm["arm"] and worker["slurm"] == {"job_id": arm["job"], "step_id": arm["step"], "node": arm["node"]}
            and worker["output"]["sha256"] == output_sha
            and worker["native_receipt"]["sha256"] == receipt_sha
        ):
            raise RuntimeError(f"worker receipt closure differs: {arm['arm']}")
        if not (
            audit.get("complete") is True and audit.get("zero_update") is True
            and audit.get("training_performed") is False and audit.get("optimization_steps") == 0
            and audit.get("oracle_reads_real_target") is True
            and audit["arm"] == arm["arm"] and audit["slurm"] == worker["slurm"]
            and audit["mechanism"]["strength"] == arm["strength"]
            and audit["mechanism"]["transport_steps"] == arm["steps"]
            and audit["postflight"] == {"location": "login_node", "compute_ffprobe_invoked": False, "inference_rerun": False}
            and audit["authority"]["launcher"]["sha256"] == LAUNCHER_SHA
            and audit["output"]["sha256"] == output_sha
            and audit["native_receipt"]["sha256"] == receipt_sha
            and audit["worker_receipt"]["sha256"] == worker_sha
        ):
            raise RuntimeError(f"login audit closure differs: {arm['arm']}")
        if not (
            complete.get("complete") is True
            and complete["output"]["sha256"] == output_sha
            and complete["audit"]["sha256"] == audit_sha
        ):
            raise RuntimeError(f"complete marker differs: {arm['arm']}")
        metric = metric_by_arm.get(arm["arm"])
        if not metric or metric["output_sha256"] != output_sha or metric["job_step"] != arm["job_step"]:
            raise RuntimeError(f"metric authority differs: {arm['arm']}")
        if metric["ssim_all_to_routeoff"] != arm["ssim_routeoff"] or metric["ssim_all_to_scaled_real_target"] != arm["ssim_target"]:
            raise RuntimeError(f"metric value differs: {arm['arm']}")
        log_path = plain(PROVENANCE / f"{arm['arm']}.log")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if f"INFERENCE_COMPLETE {arm['arm']} " not in log_text or re.search(r"Traceback|OutOfMemory|out of memory|non-finite", log_text):
            raise RuntimeError(f"worker log closure differs: {arm['arm']}")
        arm_rows[arm["arm"]] = {
            **arm, "media": name, "output_sha256": output_sha,
            "receipt": receipt_path.name, "receipt_sha256": receipt_sha,
            "worker_receipt": worker_path.name, "worker_receipt_sha256": worker_sha,
            "audit": audit_path.name, "audit_sha256": audit_sha,
            "complete": complete_path.name, "complete_sha256": sha256(complete_path),
            "trace": {"candidate_cells": cells, "anchor_forwards": forwards,
                      "qk_captures": captures, "qk_replays": replays, "pending": 0},
        }

    expected_contacts = expected_contact_names()
    observed_contacts = {path.name for path in CONTACT.glob("*.jpg") if path.is_file() and not path.is_symlink()}
    if observed_contacts != expected_contacts:
        raise RuntimeError(f"contact inventory differs: missing={expected_contacts-observed_contacts}, extra={observed_contacts-expected_contacts}")
    contact_rows = {}
    composite_dims = {
        "reference_grid.jpg": (1640, 920), "six_arm_grid.jpg": (1640, 1104),
        "six_arm_early_and_phase_grid.jpg": (1640, 552),
        "routeoff_and_sweep_late_event_grid.jpg": (1640, 1288),
    }
    for name in sorted(expected_contacts):
        path = plain(CONTACT / name)
        dims = probe_image(path)
        expected_dims = composite_dims.get(name, (1640, 92) if name.endswith("_early_and_phase.jpg") else (1640, 184))
        if dims != expected_dims:
            raise RuntimeError(f"contact geometry differs: {name}: {dims}")
        observed_sha = sha256(path)
        if name in COMPOSITE_HASHES and observed_sha != COMPOSITE_HASHES[name]:
            raise RuntimeError(f"composite SHA differs: {name}: {observed_sha}")
        contact_rows[name] = {"sha256": observed_sha, "width": dims[0], "height": dims[1]}
    return {
        "media": media_rows, "arms": arm_rows, "contacts": contact_rows,
        "metrics": {"path": "metrics.json", "sha256": METRICS_SHA},
        "sealed_authority": provenance_authority,
    }


def video_card(title: str, badge: str, media: str, sheet: str, note: str, tone: str = "") -> str:
    return f'''<article class="card {tone}"><header><span>{badge}</span><h3>{title}</h3></header>
<video controls muted loop playsinline preload="metadata" data-sync src="media/{media}"></video>
<img src="contact/{sheet}" alt="{title}: frames 0, 20, 40, 60, 80"><p>{note}</p></article>'''


def build_html(manifest_sha: str) -> str:
    refs = "".join((
        video_card("Source", "source review copy", "source.mp4", "source_f0_20_40_60_80.jpg", "This displayed 960×540 file is a review copy, not the inference bytes. Native receipts pin the exact81 inference source SHA a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646."),
        video_card("Real target", "oracle / evaluation", "real_target.mp4", "real_target_f0_20_40_60_80.jpg", "Clean reference event. It is read only as the oracle anchor in this calibration.", "target"),
        video_card("Native frozen", "reference", "native_frozen.mp4", "native_frozen_f0_20_40_60_80.jpg", "Strong appearance retention; does not supply the requested event."),
        video_card("Matched route-off", "K0 · no anchor route", "routeoff.mp4", "routeoff_f0_20_40_60_80.jpg", "Corrected read: rough placement already occurs. Hand/bottle morph reaches the treadmill around f65–75; upright release appears by f75–80, with ghosting.", "partial"),
        video_card("Oracle Q/K K40", "strength 1 · failed", "oracle_qk.mp4", "oracle_qk_f0_20_40_60_80.jpg", "Catastrophic f1+ face/body/hand/bottle collapse. This rejects the old K40 operating point.", "bad"),
    ))
    cards = []
    rows = []
    for arm in ARMS:
        name = f"{arm['label']}.mp4"
        low = arm["strength"] < 0.25
        note = (
            "Rough late placement is reproduced, but it was already present in route-off; no incremental oracle action gain is visible."
            if low else
            "Strict property/instance fail: the original hand-held bottle persists while a second bottle appears on the treadmill."
        )
        cards.append(video_card(
            f"strength {arm['strength']:.2f} · K{arm['steps']}",
            "null vs route-off" if low else "duplicate-bottle fail", name,
            f"{arm['label']}_f0_20_40_60_80.jpg", note, "partial" if low else "bad",
        ))
        rows.append(
            f"<tr><td>{arm['strength']:.2f}</td><td>K{arm['steps']}</td><td>{arm['job_step']}</td>"
            f"<td>{arm['ssim_routeoff']:.6f}</td><td>{arm['ssim_target']:.6f}</td>"
            f"<td>{'route-off event reproduced; no increment' if low else 'original + second bottle; drift'}</td>"
            f"<td><a href=\"provenance/{arm['label']}.mp4.legacy-mev840-lowstrength-audit.json\">audit</a></td></tr>"
        )
    template = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MEV840 legacy oracle low-strength sweep</title><style>
:root{--bg:#091018;--panel:#111b27;--panel2:#162332;--line:#2b3a4c;--text:#eef5fb;--muted:#9eb0c1;--cyan:#58d5e8;--red:#ff776d;--amber:#f2ba55;--green:#71d49b}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% -10%,#183452 0,transparent 34%),var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1720px;margin:auto;padding:28px}h1{font-size:30px;margin:0 0 4px}h2{margin:30px 0 12px;font-size:20px}.sub{color:var(--muted);margin:0 0 16px}.verdict{border:1px solid #9b3f3b;background:linear-gradient(110deg,#321719,#1a1821);padding:16px 18px;border-radius:12px;font-size:15px}.verdict strong{color:#ffaaa4}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}.metric{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px}.metric b{display:block;font-size:20px;color:var(--cyan)}.metric small{color:var(--muted)}.controls{position:sticky;top:0;z-index:3;display:flex;gap:7px;flex-wrap:wrap;background:#091018e8;backdrop-filter:blur(8px);padding:9px 0}.controls button{color:var(--text);background:#172332;border:1px solid var(--line);border-radius:7px;padding:7px 10px;cursor:pointer}.controls button:hover{border-color:var(--cyan)}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}.sweep{grid-template-columns:repeat(3,minmax(0,1fr))}.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;overflow:hidden}.card.partial{border-color:#6c5a31}.card.bad{border-color:#763633}.card.target{border-color:#296472}.card header{padding:10px 12px}.card header span{color:var(--cyan);font-size:10px;letter-spacing:.09em;text-transform:uppercase}.card h3{margin:3px 0 0;font-size:15px}.card video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#000}.card img{display:block;width:100%;border-block:1px solid var(--line)}.card p{padding:0 12px 10px;color:var(--muted);min-height:68px}.figures{display:grid;grid-template-columns:1fr 1fr;gap:12px}.figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}.figure img{display:block;width:100%}.figure figcaption{padding:10px 12px;color:var(--muted)}.figure.wide{grid-column:1/-1}table{width:100%;border-collapse:collapse;background:var(--panel)}th,td{border:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}th{color:var(--muted)}a{color:var(--cyan)}.good{color:var(--green)}.warn{color:var(--amber)}.fail{color:var(--red)}.notes{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px 18px}.foot{font-size:12px;color:var(--muted);margin-top:18px}code{color:#cbe7f2}
@media(max-width:1200px){.grid{grid-template-columns:repeat(3,1fr)}.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:760px){main{padding:14px}.grid,.sweep,.figures,.metrics{grid-template-columns:1fr}.figure.wide{grid-column:auto}}
</style></head><body><main><h1>MEV840 · legacy oracle low-strength sweep</h1>
<p class="sub">Normalized real target oracle · activity25 target-owned Q/K · zero update · strength × early transport schedule</p>
<div class="verdict"><strong>STRICT FAIL — incremental oracle gain = 0/6.</strong> 关键修正：matched route-off 本身已经在 f65–80 粗糙完成“靠近跑步机 → 直立释放”，只是带有 hand/bottle morph 和 ghosting。strength 0.05/0.10 的四臂主要复现这条已有路径，不能归因于 oracle；strength 0.25 的两臂保留手中原瓶，同时在跑步机出现第二瓶，构成实例/属性漂移。strength 1 · K40 则灾难性崩坏。</div>
<section class="metrics"><div class="metric"><b>0 / 6</b><small>incremental oracle action gain</small></div><div class="metric"><b>4 / 4</b><small>0.05/0.10 arms: route-off-like event</small></div><div class="metric"><b>2 / 2</b><small>0.25 arms: duplicate bottle fail</small></div><div class="metric"><b>0 update</b><small>frozen base; no trained attention adapter/checkpoint</small></div></section>
<div class="controls"><button data-action="play">Play all 11</button><button data-action="pause">Pause all</button><button data-frame="0">f0</button><button data-frame="20">f20</button><button data-frame="40">f40</button><button data-frame="60">f60</button><button data-frame="65">f65</button><button data-frame="70">f70</button><button data-frame="75">f75</button><button data-frame="80">f80</button></div>
<h2>References and causal control</h2><section class="grid">@@REFS@@</section>
<h2>Six-arm oracle calibration</h2><section class="grid sweep">@@CARDS@@</section>
<h2>Frame evidence</h2><section class="figures"><figure class="figure"><img src="contact/reference_grid.jpg" alt="source, real target, native frozen, route-off and K40 oracle at frames 0,20,40,60,80"><figcaption>Rows: source · real target · native frozen · corrected route-off · failed oracle strength1/K40. Columns: f0/20/40/60/80.</figcaption></figure><figure class="figure"><img src="contact/six_arm_grid.jpg" alt="six sweep arms at frames 0,20,40,60,80"><figcaption>Rows: 0.05/K5 · 0.10/K5 · 0.25/K5 · 0.05/K10 · 0.10/K10 · 0.25/K10. Columns: f0/20/40/60/80.</figcaption></figure><figure class="figure"><img src="contact/six_arm_early_and_phase_grid.jpg" alt="six arms at early and phase frames"><figcaption>Early-collapse check: f0/1/2/4/8/10/20/40/60/80. All six avoid the catastrophic K40 blur, but this is not an action-gain result.</figcaption></figure><figure class="figure"><img src="contact/routeoff_and_sweep_late_event_grid.jpg" alt="route-off and six sweep arms from frame 60 to 80"><figcaption>Corrected late-event audit. Rows: route-off, then the six arms in the order above. Columns: f60/65/70/75/80. Low strengths reproduce route-off; 0.25 retains the original hand bottle while adding a second treadmill bottle.</figcaption></figure></section>
<h2>Quantitative displacement from controls</h2><table><thead><tr><th>Strength</th><th>Schedule</th><th>Slurm step</th><th>SSIM → route-off</th><th>SSIM → scaled target</th><th>Strict visual read</th><th>Evidence</th></tr></thead><tbody>@@ROWS@@</tbody></table>
<section class="notes"><h2>Interpretation</h2><ul><li><span class="good">Stability:</span> lowering strength/steps avoids the K40 global collapse.</li><li><span class="warn">Causal null:</span> the apparent action in 0.05/0.10 already exists in matched route-off; pixel displacement grows with dose, but target SSIM improves by at most 0.001404 and does not establish an action-specific gain.</li><li><span class="fail">Property seam:</span> 0.25 creates a second bottle while retaining the original, exactly the kind of source-object instance drift the representation must forbid.</li><li><b>Decision:</b> do not route self-generated anchors through this legacy operator. A new role/relation representation must first pass an incremental real-target oracle gate against this corrected route-off control.</li><li><b>Boundary:</b> this is frozen zero-update inference, not training and not the new v15 graph.</li></ul></section>
<p class="foot"><a href="metrics.json">metrics.json</a> · <a href="manifest.json">manifest.json</a> · manifest SHA <code>@@MANIFEST_SHA@@</code>. All six native receipts close 81f/25fps, exact source/oracle hashes, frozen parameters, keyed-only target noise, no anchor Gaussian, K5/K10 active schedules, Q/K-only capture/replay counts and pending=0. Login-node postflight did not rerun inference.</p>
</main><script>const vids=[...document.querySelectorAll('video[data-sync]')];document.querySelector('[data-action="play"]').onclick=()=>vids.forEach(v=>v.play());document.querySelector('[data-action="pause"]').onclick=()=>vids.forEach(v=>v.pause());document.querySelectorAll('[data-frame]').forEach(b=>b.onclick=()=>{const t=Number(b.dataset.frame)/25;vids.forEach(v=>{v.pause();v.currentTime=t})});</script></body></html>'''
    return (template.replace("@@REFS@@", refs).replace("@@CARDS@@", "".join(cards))
            .replace("@@ROWS@@", "".join(rows)).replace("@@MANIFEST_SHA@@", manifest_sha))


def validate_html(html: str) -> int:
    if html.count("<video ") != 11:
        raise RuntimeError("page must contain exactly 11 videos")
    if "incremental oracle gain = 0/6" not in html or "route-off 本身已经" not in html:
        raise RuntimeError("corrected strict-fail conclusion is absent")
    refs = re.findall(r'(?:src|href)="([^"#]+)"', html)
    for ref in refs:
        if ref == "manifest.json":
            continue
        if "://" not in ref and not (ROOT / ref).is_file():
            raise RuntimeError(f"missing local page reference: {ref}")
    return len(refs)


def main() -> int:
    validated = validate()
    manifest = {
        "schema": "mev840-legacy-oracle-lowstrength-review-v1",
        "date": "2026-08-21",
        "status": "STRICT_FAIL_INCREMENTAL_ORACLE_GAIN_0_OF_6",
        "claim_boundary": {
            "legacy_operator_calibration": True,
            "new_v15_action_graph": False,
            "training_or_parameter_update": False,
            "real_target_read": "oracle arms only",
        },
        "corrected_visual_audit": {
            "routeoff_rough_late_placement_present": True,
            "routeoff_timing": "hand/bottle morph reaches treadmill around f65-f75; upright release by f75-f80; late ghosting",
            "strength_005_010_incremental_oracle_gain": False,
            "strength_005_010_result": "reproduces routeoff-like rough placement",
            "strength_025_result": "original hand bottle persists while second treadmill bottle appears",
            "strength1_k40_result": "catastrophic f1+ collapse",
            "strict_decision": "NO_GO for legacy operator as an action-specific self/oracle route",
        },
        "quantitative": {
            "routeoff_to_scaled_target_ssim_all": 0.745480,
            "best_sweep_to_scaled_target_ssim_all": 0.746884,
            "largest_scaled_target_ssim_increment": 0.001404,
            "ssim_is_action_metric": False,
        },
        "builder": {"path": str(Path(__file__).resolve().relative_to(REPO)), "sha256": sha256(Path(__file__).resolve())},
        **validated,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    html = build_html(manifest_sha)
    link_count = validate_html(html)
    manifest_tmp = ROOT / ".manifest.json.tmp"
    index_tmp = ROOT / ".index.html.tmp"
    manifest_tmp.write_text(manifest_text, encoding="utf-8")
    index_tmp.write_text(html, encoding="utf-8")
    manifest_tmp.replace(ROOT / "manifest.json")
    index_tmp.replace(ROOT / "index.html")
    result = {
        "manifest_sha256": sha256(ROOT / "manifest.json"),
        "index_sha256": sha256(ROOT / "index.html"),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "video_count": 11,
        "local_reference_count": link_count,
        "contact_count": len(validated["contacts"]),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

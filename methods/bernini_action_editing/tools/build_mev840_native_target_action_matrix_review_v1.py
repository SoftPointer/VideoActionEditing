#!/usr/bin/env python3
"""Build and validate the MEV840 native target-action prompt-matrix review."""

from __future__ import annotations

import copy
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "md/action_editing/20260822_mev840_native_target_action_matrix_review"
AUTHORITY = REPO / "methods/bernini_action_editing/assets/mev840_native_rv2v_same_process_formal_v1.json"
SEALED_OBSERVER_MANIFEST = REPO / "methods/bernini_action_editing/assets/mev840_candidate_action_observer_formal6_v1.json"
SUMMARY = ROOT / "observer_formal6/summary.json"
OBSERVER_POSTFLIGHT = ROOT / "postflight/formal6_observer.json"
OBSERVER_CHECKSUMS = ROOT / "observer_formal6_REMOTE_SHA256SUMS.txt"
AUTHORITY_SHA = "e5d2f0a9a8bbf88df84264494e911e485e3bbff459b57a9bd32e7b39ffc79ab9"
SEALED_OBSERVER_MANIFEST_SHA = "57fe3594a1a23184e862d29afcf68417f0571bb9039eea8eab8740051a73226d"
SUMMARY_SHA = "dd73a270f3853295bb569d026d5ae409ea51f92f77128c205ed4dec89d0b3397"
OBSERVER_POSTFLIGHT_SHA = "2a0fac46ba346370dd4317f8cfdace04ceb4ac20eb8994fd72dd2391857953e6"
OBSERVER_CHECKSUMS_SHA = "0a8bf7a4c126d9ca0d0eb040e769c1977dcdb39525ac374aac3443e48107c455"
TARGET_REPRESENTATION_DIGEST = "7ce831a771d834817475445e38666775707557041d56a5ec5267c7d8bf0f86bd"
SUMMARY_OBJECT_DIGEST = "1054f39f7d3dac7837c48a4e142099774a3a2bc3a01a6156e8a50534eaf52f18"
ACTION_THRESHOLD = 0.78
IDS = ["p0_s2027", "p1_s2027", "p2_s2027", "p0_s2028", "p1_s2028", "p2_s2028"]
PROMPT_LABELS = {
    "p0": ("P0", "base-action baseline（不是 null）"),
    "p1": ("P1", "P0 + event-order supplement"),
    "p2": ("P2", "P0 + relation/contact/same-bottle supplement"),
}
CANDIDATES = {
    "p0_s2027": ("formal_runs/seed2027/p0a.mp4", "d84d848a6387871ddabde4d1cd65f9dfc0bd41f417cff1c0ecfded7f34a93478", "contacts/seed2027_p0a_f000_013_027_040_053_066_080.jpg"),
    "p1_s2027": ("formal_runs/seed2027/p1.mp4", "183420c2030fbd343b7f9c0d142b4f275bd1e7c5a53739937a6c71a957675b4f", "contacts/seed2027_p1_f000_013_027_040_053_066_080.jpg"),
    "p2_s2027": ("formal_runs/seed2027/p2.mp4", "2c0239e5ba1dd664bf54bbf57ba2e12500d948825c5aa3ed728164acfbca8eab", "contacts/seed2027_p2_f000_013_027_040_053_066_080.jpg"),
    "p0_s2028": ("formal_runs/seed2028/p0a.mp4", "80293af6a4abffb7f8eaea078b887c2134db5c054dec7da867ce28fe44f0969f", "contacts/seed2028_p0a_f000_013_027_040_053_066_080.jpg"),
    "p1_s2028": ("formal_runs/seed2028/p1.mp4", "de640d466a5baebd10caf319627cdee59345e063102be65b052a3c22103ae116", "contacts/seed2028_p1_f000_013_027_040_053_066_080.jpg"),
    "p2_s2028": ("formal_runs/seed2028/p2.mp4", "721c4044f4e77ce872a926df08be6405fc7535f89cf21e11f71baec04ae86c97", "contacts/seed2028_p2_f000_013_027_040_053_066_080.jpg"),
}
RECEIPTS = {
    2027: ("formal_runs/seed2027/receipt.json", "7f431c05e06b87d0ec217105962ad1c77d16c0d4d39d25ff7848ac3fd11c52f3", "143808.512", "auh7-1b-gpu-292"),
    2028: ("formal_runs/seed2028/receipt.json", "098bbb9919f2b3715bbcd46c9591f912d21d954d4964ad7811579e2309f6f9ae", "147873.11", "auh7-1b-gpu-284"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"plain file required: {path}")
    return path


def pinned(path: Path, expected: str) -> Path:
    plain(path)
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"SHA differs: {path}: {observed}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(plain(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def object_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    run = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(run.stdout)["streams"][0]
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": stream["r_frame_rate"], "frames": int(stream["nb_read_frames"]),
    }


def score_row(summary_row: dict[str, Any]) -> dict[str, Any]:
    identifier = summary_row["candidate_id"]
    prefix, seed_text = identifier.split("_s")
    label, semantics = PROMPT_LABELS[prefix]
    video, video_sha, contact = CANDIDATES[identifier]
    base = {
        "candidate_id": identifier,
        "seed": int(seed_text),
        "prompt_label": label,
        "prompt_semantics": semantics,
        "video": video,
        "video_sha256": video_sha,
        "contact_sheet": contact,
        "appearance_quality_gate_passed": None,
        "single_bottle_gate_passed": None,
        "selection_authorized": False,
    }
    if summary_row.get("action_score_assigned") is False:
        return {
            **base,
            "observer_status": "unassigned_fail_closed",
            "observer_reason": "role_mask_became_empty",
            "action_score": None,
            "action_gate_passed": None,
            "reject": f"observer_formal6/{identifier}/reject.json",
            "observer_spec": f"observer_formal6/{identifier}/observer.spec.json",
        }
    score_path = ROOT / f"observer_formal6/{identifier}/action.score.json"
    action_path = ROOT / f"observer_formal6/{identifier}/observer/action.json"
    score = read_json(score_path)
    if sha256(score_path) != summary_row["action_score_sha256"] or sha256(action_path) != summary_row["candidate_action_sha256"]:
        raise RuntimeError(f"observer artifact binding differs: {identifier}")
    if score["decision"]["action_gate_threshold"] != ACTION_THRESHOLD:
        raise RuntimeError(f"action threshold differs: {identifier}")
    return {
        **base,
        "observer_status": "scored_action_gate_failed",
        "observer_reason": "action_gate_failed",
        "action_score": score["scores"]["action"],
        "score_components": score["scores"],
        "action_gate_threshold": ACTION_THRESHOLD,
        "action_gate_passed": False,
        "action_score_file": f"observer_formal6/{identifier}/action.score.json",
        "candidate_action_file": f"observer_formal6/{identifier}/observer/action.json",
        "observer_overlay": f"observer_formal6/{identifier}/observer/AUDIT_ONLY_overlay.mp4",
        "observer_contact_sheet": f"observer_formal6/{identifier}/observer/AUDIT_ONLY_contact_sheet_f0_20_40_60_80.jpg",
    }


def build_metrics() -> dict[str, Any]:
    pinned(AUTHORITY, AUTHORITY_SHA)
    pinned(SEALED_OBSERVER_MANIFEST, SEALED_OBSERVER_MANIFEST_SHA)
    pinned(SUMMARY, SUMMARY_SHA)
    pinned(OBSERVER_POSTFLIGHT, OBSERVER_POSTFLIGHT_SHA)
    pinned(OBSERVER_CHECKSUMS, OBSERVER_CHECKSUMS_SHA)
    authority = read_json(AUTHORITY)
    summary = read_json(SUMMARY)
    digest_input = copy.deepcopy(summary)
    recorded_digest = digest_input.pop("summary_sha256")
    if recorded_digest != SUMMARY_OBJECT_DIGEST or object_digest(digest_input) != recorded_digest:
        raise RuntimeError("observer summary object digest differs")
    if not (
        summary["target_representation_digest"] == TARGET_REPRESENTATION_DIGEST
        and summary["candidate_count"] == 6
        and summary["generator_forward_calls"] == 0
        and summary["optimizer_updates"] == 0
        and summary["training_performed"] is False
        and summary["selection_authorized"] is False
        and summary["appearance_quality_gate_passed"] is None
    ):
        raise RuntimeError("observer summary authority differs")
    summary_by_id = {row["candidate_id"]: row for row in summary["candidates"]}
    if set(summary_by_id) != set(IDS):
        raise RuntimeError("observer candidate inventory differs")

    prompts = authority["prompts"]
    if not (
        prompts["P0"]["supplement_kind"] == "none"
        and prompts["P1"]["supplement_kind"] == "event_order"
        and prompts["P2"]["supplement_kind"] == "relation_contact"
        and [prompts[x]["untruncated_token_count"] for x in ("P0", "P1", "P2")] == [162, 231, 276]
    ):
        raise RuntimeError("prompt authority differs")

    receipt_rows = []
    for seed, (relative, expected_sha, step, node) in RECEIPTS.items():
        path = pinned(ROOT / relative, expected_sha)
        receipt = read_json(path)
        if not (
            receipt["schema_version"] == "mev840-native-rv2v-paired-prompt-matrix-formal-v1"
            and receipt["scientific_claim_authorized"] is False
            and receipt["production_claim_forbidden"] is True
            and receipt["interpretation"]["training_performed"] is False
            and receipt["freeze_certificate"]["trainable_parameter_elements"] == 0
            and receipt["paired_same_process_contract"]["p0_replay"]["generated_latent_bit_exact"] is True
        ):
            raise RuntimeError(f"formal receipt differs: seed{seed}")
        receipt_rows.append({"seed": seed, "path": relative, "sha256": expected_sha, "step": step, "node": node})

    media = {}
    for identifier, (relative, expected_sha, _contact) in CANDIDATES.items():
        path = pinned(ROOT / relative, expected_sha)
        probe = probe_video(path)
        if probe != {"width": 656, "height": 368, "fps": "25/1", "frames": 81}:
            raise RuntimeError(f"candidate media ABI differs: {identifier}: {probe}")
        media[identifier] = {"path": relative, "sha256": expected_sha, **probe}

    candidates = [score_row(summary_by_id[identifier]) for identifier in IDS]
    if sum(row["action_score"] is not None for row in candidates) != 5:
        raise RuntimeError("assigned score count differs")
    if sum(row["action_gate_passed"] is True for row in candidates) != 0:
        raise RuntimeError("unexpected action gate pass")
    return {
        "schema": "mev840-native-target-action-matrix-review-metrics-v1",
        "case_id": "MEV840",
        "status": "OBSERVER_ACTION_GATE_NO_GO__EXTERNAL_GATES_PENDING__SELECTION_FALSE",
        "terminology": {
            "P0": "base-action baseline; not null",
            "P1": "P0 plus event-order supplement",
            "P2": "P0 plus relation/contact/same-bottle supplement; visual similarity to P0 is not a null prompt",
        },
        "design": {
            "native_frozen_rv2v": True,
            "pairing_scope": "within_each_seed",
            "same_process_order": ["P0a", "P1", "P2", "P0b"],
            "p0_replay_bit_exact": True,
            "cross_seed_source_condition_bit_exact": False,
            "generator_reads_target_media_or_action": False,
            "generator_forward_calls_during_observer": 0,
            "training_performed": False,
            "optimizer_updates": 0,
        },
        "target_action_oracle": {
            "representation_digest": TARGET_REPRESENTATION_DIGEST,
            "coordinate_free": True,
            "appearance_free": True,
            "selection_only": True,
            "turn_onset_and_peak_are_low_confidence_mask_profile_proxies": True,
        },
        "prompt_authority": {
            label: {
                "full_prompt": prompts[label]["full_prompt_utf8"],
                "full_prompt_sha256": prompts[label]["full_prompt_utf8_sha256"],
                "final_task_prompt_sha256": prompts[label]["final_task_prompt_utf8_sha256"],
                "untruncated_token_count": prompts[label]["untruncated_token_count"],
            }
            for label in ("P0", "P1", "P2")
        },
        "formal_receipts": receipt_rows,
        "media": media,
        "observer": {
            "step": "143808.513", "node": "auh7-1b-gpu-292", "state": "COMPLETED",
            "exit_code": "0:0", "elapsed": "00:05:05", "max_rss_kib": 4016900,
            "summary_sha256": SUMMARY_SHA, "summary_object_digest": SUMMARY_OBJECT_DIGEST,
            "action_gate_threshold": ACTION_THRESHOLD,
            "score_assigned_count": 5, "unassigned_count": 1, "action_gate_pass_count": 0,
        },
        "candidates": candidates,
        "external_gates": {
            "appearance_quality_gate_passed": None,
            "single_bottle_gate_passed": None,
            "same_bottle_identity_gate_passed": None,
        },
        "decision": {
            "best_candidate_selected": False,
            "selection_authorized": False,
            "scientific_claim_authorized": False,
            "observer_action_success_established": False,
            "native_generation_provenance_released": True,
            "native_visual_quality_claim_from_observer": False,
            "note": "All five assigned action scores are below 0.78; p1_s2028 is unassigned after an empty role mask and is not a low score.",
        },
    }


def build_manifest(metrics: dict[str, Any], metrics_sha: str) -> dict[str, Any]:
    refs = {
        "metrics.json", "postflight/formal6_observer.json", "observer_formal6/summary.json",
        "observer_formal6_REMOTE_SHA256SUMS.txt", "observer_logs/formal6.run.log",
        "observer_logs/formal6.preallocation_failed.log", "formal_runs/REMOTE_SHA256SUMS.txt",
        "logs/REMOTE_SHA256SUMS.txt", "postflight/seed2027.json", "postflight/seed2028.json",
    }
    refs.update(relative for relative, _sha, _step, _node in RECEIPTS.values())
    for row in metrics["candidates"]:
        refs.update({row["video"], row["contact_sheet"]})
        for key in ("action_score_file", "candidate_action_file", "observer_overlay", "observer_contact_sheet", "reject", "observer_spec"):
            if key in row:
                refs.add(row[key])
    ref_rows = []
    for relative in sorted(refs):
        path = plain(ROOT / relative)
        ref_rows.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    return {
        "schema": "mev840-native-target-action-matrix-review-manifest-v1",
        "case_id": "MEV840",
        "builder": str(Path(__file__).resolve().relative_to(REPO)),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "metrics_sha256": metrics_sha,
        "reference_count": len(ref_rows),
        "missing_reference_count": 0,
        "video_count": sum(row["path"].endswith(".mp4") for row in ref_rows),
        "references": ref_rows,
        "claims": {
            "selection_authorized": False,
            "scientific_claim_authorized": False,
            "appearance_gate_pending": True,
            "single_bottle_gate_pending": True,
            "p1_s2028_unassigned_not_scored": True,
        },
    }


def fmt_score(value: Any) -> str:
    return "unassigned" if value is None else f"{float(value):.4f}"


def candidate_card(row: dict[str, Any]) -> str:
    cid = row["candidate_id"]
    score = fmt_score(row["action_score"])
    status = "UNASSIGNED" if row["action_score"] is None else "FAIL"
    detail = "SAM2 role mask became empty; fail-closed, no numeric score." if row["action_score"] is None else f"action={score} < {ACTION_THRESHOLD:.2f}"
    evidence = []
    for key, label in (("action_score_file", "score JSON"), ("candidate_action_file", "action JSON"), ("reject", "reject JSON"), ("observer_spec", "observer spec")):
        if key in row:
            evidence.append(f'<a href="{html.escape(row[key])}">{label}</a>')
    overlay = ""
    if "observer_overlay" in row:
        overlay = f'<details><summary>Observer overlay（audit only）</summary><video controls preload="metadata" src="{row["observer_overlay"]}"></video><img loading="lazy" src="{row["observer_contact_sheet"]}" alt="{cid} observer contact sheet"></details>'
    return f'''<article class="card" id="{cid}">
      <header><div><span class="prompt">{row["prompt_label"]}</span><h3>{cid}</h3></div><span class="chip {status.lower()}">{status}</span></header>
      <p class="semantics">{html.escape(row["prompt_semantics"])}</p>
      <video controls loop muted playsinline preload="metadata" src="{row["video"]}"></video>
      <img loading="lazy" src="{row["contact_sheet"]}" alt="{cid} seven-frame contact sheet">
      <div class="score"><strong>{score}</strong><span>{html.escape(detail)}</span></div>
      <p class="links">{' · '.join(evidence)}</p>{overlay}
      <p class="pending">appearance: pending · single/same bottle: pending · selection: false</p>
    </article>'''


def build_html(metrics: dict[str, Any], manifest: dict[str, Any], metrics_sha: str, manifest_sha: str) -> str:
    rows = "".join(
        f'<tr><td>{r["candidate_id"]}</td><td>{r["prompt_label"]}</td><td>{fmt_score(r["action_score"])}</td><td>{"unassigned" if r["action_score"] is None else "FAIL"}</td></tr>'
        for r in metrics["candidates"]
    )
    cards = "".join(candidate_card(row) for row in metrics["candidates"])
    prompt_items = "".join(
        f'<details><summary>{label}: {html.escape(metrics["terminology"][label])} · {value["untruncated_token_count"]} tokens</summary><p>{html.escape(value["full_prompt"])}</p><code>{value["full_prompt_sha256"]}</code></details>'
        for label, value in metrics["prompt_authority"].items()
    )
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MEV840 Native Target-Action Matrix Review</title>
<style>
:root{{--bg:#07111f;--panel:#0d1a2b;--panel2:#12233a;--text:#e8f0fa;--muted:#9fb0c5;--line:#28405e;--cyan:#69d2e7;--red:#ff7a85;--amber:#ffc76a;--green:#74d99f}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 20% 0,#142744,var(--bg) 45%);color:var(--text);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}main{{max-width:1500px;margin:auto;padding:34px 22px 70px}}h1{{font-size:clamp(30px,5vw,58px);line-height:1.05;margin:.1em 0}}h2{{margin-top:42px}}a{{color:var(--cyan)}}code{{font-size:11px;overflow-wrap:anywhere;color:#b7c7da}}.eyebrow{{color:var(--cyan);font-weight:800;letter-spacing:.12em;text-transform:uppercase}}.lead{{font-size:19px;color:#c5d3e3;max-width:1000px}}.banner{{padding:18px 20px;border:1px solid #73414b;border-left:6px solid var(--red);background:#211723;border-radius:12px;margin:24px 0}}.banner strong{{color:#ffadb5}}.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.fact,.panel{{background:color-mix(in srgb,var(--panel) 92%,transparent);border:1px solid var(--line);border-radius:14px;padding:16px}}.fact b{{display:block;font-size:24px}}.fact span,.muted{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:12px;overflow:hidden}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line)}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:14px;min-width:0}}.card header{{display:flex;justify-content:space-between;align-items:center}}.card h3{{margin:2px 0 8px}}.prompt{{font-weight:850;color:var(--cyan)}}.chip{{font-size:12px;font-weight:850;padding:5px 9px;border-radius:99px}}.chip.fail{{background:#4c2029;color:#ffbec4}}.chip.unassigned{{background:#4a3716;color:#ffe0a4}}video,img{{display:block;width:100%;border-radius:10px;background:#02070d;margin:10px 0}}.semantics,.pending{{color:var(--muted)}}.score{{display:flex;align-items:baseline;gap:12px;padding:8px 0}}.score strong{{font-size:28px}}.score span{{color:var(--muted)}}.links{{font-size:13px}}details{{border-top:1px solid var(--line);padding:10px 0}}summary{{cursor:pointer;font-weight:700}}.warning{{color:var(--amber)}}footer{{margin-top:42px;color:var(--muted);font-size:12px}}@media(max-width:980px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="eyebrow">Frozen native RV2V · coordinate-free observer · zero update</p>
<h1>MEV840 target-action prompt matrix</h1>
<p class="lead">在每个 seed 内，各自复用同一个 source condition 与同一个 official Gaussian，并在各自同一 WORLD4 进程中比较 P0 / P1 / P2。两个 seed 位于不同节点，不主张跨 seed condition bit-exact。target-derived 信息只以净化后的事件/关系文本进入 prompt；生成器从未读取 real-target pixels、features 或 action JSON。</p>
<div class="banner"><strong>当前结果：observer action gate NO-GO。</strong> 5 个可评分候选全部低于 0.78；P1 seed2028 因 role mask 变空而 fail-closed unassigned。没有候选被选择，appearance 与 single/same-bottle 外部门仍是 pending。</div>
<section class="facts"><div class="fact"><b>0 / 6</b><span>action gate pass</span></div><div class="fact"><b>5 + 1</b><span>scored + unassigned</span></div><div class="fact"><b>0</b><span>generator calls in observer</span></div><div class="fact"><b>false</b><span>selection authorized</span></div></section>
<h2>读法与边界</h2><div class="panel"><p><b>P0 是 base-action baseline，不是 null。</b> P1 保留 P0 并追加 event order；P2 保留 P0 并追加 relation/contact/same-bottle。P2 视觉上接近 P0 不能称为 prompt null。</p><p class="warning">action 分数是 coordinate-free observer 的诊断量，不是外观质量或单瓶守恒证明。target turn onset/peak 来自低置信 mask-profile proxy；P1 seed2028 无数值分数，不能参与排序。</p>{prompt_items}</div>
<h2>Observer summary</h2><table><thead><tr><th>Candidate</th><th>Prompt</th><th>Action score</th><th>Gate</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Six native candidates</h2><div class="grid">{cards}</div>
<h2>Provenance</h2><div class="panel"><p>Formal generation: same-process P0a→P1→P2→P0b, 40-step native RV2V; P0 replay latent bit-exact within each seed. Observer: <code>143808.513</code>, node292, single GPU serial, COMPLETED/0:0, 5:05, MaxRSS 4,016,900 KiB.</p><p><a href="metrics.json">metrics.json</a> · <a href="manifest.json">manifest.json</a> · <a href="observer_formal6/summary.json">raw observer summary</a> · <a href="postflight/formal6_observer.json">observer postflight</a> · <a href="observer_formal6_REMOTE_SHA256SUMS.txt">119-file checksum receipt</a></p><p>metrics SHA <code>{metrics_sha}</code><br>manifest SHA <code>{manifest_sha}</code><br>target representation digest <code>{TARGET_REPRESENTATION_DIGEST}</code></p></div>
<footer>Mechanical provenance released; scientific claim and candidate selection remain unauthorized. Reference inventory: {manifest["reference_count"]}, missing: {manifest["missing_reference_count"]}.</footer>
</main></body></html>'''


def validate_outputs(metrics: dict[str, Any], manifest: dict[str, Any]) -> None:
    html_text = plain(ROOT / "index.html").read_text(encoding="utf-8")
    if not all(token in html_text for token in ("P0 是 base-action baseline，不是 null", "selection authorized", "P1 seed2028 无数值分数")):
        raise RuntimeError("required claim boundary is absent from index")
    if re.search(r"P0\s*=.*null baseline|P2\s*=\s*(?:a\s+)?null", html_text, re.I):
        raise RuntimeError("forbidden null-arm claim appears")
    refs = re.findall(r'(?:src|href)="([^"#]+)"', html_text)
    missing = [ref for ref in refs if not (ROOT / ref).is_file()]
    if missing:
        raise RuntimeError(f"HTML references are missing: {missing}")
    if len(manifest["references"]) != manifest["reference_count"] or manifest["missing_reference_count"] != 0:
        raise RuntimeError("manifest inventory differs")
    for row in manifest["references"]:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"manifest reference SHA differs: {row['path']}")
    if metrics["decision"]["selection_authorized"] is not False or metrics["external_gates"]["single_bottle_gate_passed"] is not None:
        raise RuntimeError("claim gates differ")


def main() -> int:
    metrics = build_metrics()
    metrics_path = ROOT / "metrics.json"
    metrics_path.write_bytes(json_bytes(metrics))
    metrics_sha = sha256(metrics_path)
    manifest = build_manifest(metrics, metrics_sha)
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_bytes(json_bytes(manifest))
    manifest_sha = sha256(manifest_path)
    (ROOT / "index.html").write_text(build_html(metrics, manifest, metrics_sha, manifest_sha), encoding="utf-8")
    validate_outputs(metrics, manifest)
    print(json.dumps({"status": "PASS", "index_sha256": sha256(ROOT / "index.html"), "manifest_sha256": manifest_sha, "metrics_sha256": metrics_sha, "references": manifest["reference_count"], "missing": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

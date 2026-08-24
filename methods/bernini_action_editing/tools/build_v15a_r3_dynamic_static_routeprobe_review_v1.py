#!/usr/bin/env python3
"""Build the fail-closed v15a-r3 E00 matched-route review.

The page is deliberately narrow in scope: one immutable authority row and one
matched, zero-update causal triplet.  All videos, native receipts, v15a audit
sidecars, and the triplet manifest are validated before the output directory is
created.  This prevents a partial or provenance-mismatched page from looking
like a completed result.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping


EXPERIMENT_TAG = (
    "v15a_zero_update_dynamic_static_e00_maxstrength_routeprobe_r3_20260820"
)
FIXED_OUTPUT_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/"
    "online_anchor_attention_training_v1/dynaedit_maxstrength_routeprobe_v15a_r3/"
    f"{EXPERIMENT_TAG}"
)
GROUP_ID = "e00-v15a-r3-plain-frozen-max-strength-clean-noised-route-probe-v1"
PROBE_KIND = "max-strength clean-noised route probe"
SOURCE_SHA256 = "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de"
ANCHOR_SHA256 = "e485b2963d5ba191d8fc98158b7ab100bd1bd7584735221077c014ff0c3a19aa"
FROZEN_SHA256 = "a9fc55338ec4dbf5b338f20cda573c4e9670b0b0236fbd73703a4fdc774a51a7"
EXPECTED_CANDIDATE_COUNTS = (5, 5, 5) + (1,) * 37


@dataclass(frozen=True)
class Baseline:
    key: str
    filename: str
    output_name: str
    sha256: str
    label: str
    detail: str


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    role: str
    transport: str
    transport_steps: int
    qk_capture_count: int
    qk_replay_count: int
    anchor_model_forwards: int
    output_name: str
    card_label: str
    detail: str

    @property
    def mp4_name(self) -> str:
        return f"{self.label}.mp4"


BASELINES = (
    Baseline(
        "source",
        "e00-source.mp4",
        "e00-source-authority.mp4",
        SOURCE_SHA256,
        "Source authority",
        "Identity/content authority. In the observed source, object #1 pours into #2.",
    ),
    Baseline(
        "anchor",
        "e00-t2v-anchor-v0.mp4",
        "e00-pure-t2v-action-anchor.mp4",
        ANCHOR_SHA256,
        "Pure-T2V action anchor",
        "Action timing donor only by design. Its appearance must be ignored, but the outputs below show that this isolation is incomplete.",
    ),
    Baseline(
        "frozen",
        "e00-frozen-s0.mp4",
        "e00-frozen-rv2v-s0.mp4",
        FROZEN_SHA256,
        "Frozen RV2V · S0",
        "Matched editor baseline. It is a failure/control output, never a target.",
    ),
)


ARMS = (
    Arm(
        "route_off",
        "E00_V15AR3_FROZEN_ZEROUPDATE_DYNSTATIC_MATCHED_PLAIN_FROZEN_ROUTEOFF_K0_A100",
        "route_off_plain_frozen",
        "self_target_owned_temporal_kernel_attn_output_v14r2",
        0,
        0,
        0,
        0,
        "e00-v15a-r3-matched-routeoff-k0.mp4",
        "Matched Frozen weights · Q/K route OFF · K0",
        "Strict FAIL: #1→#2 persists and #2 becomes milky/opaque. Q/K injection is off, but this arm still shares the anchor-derived candidate-0 Gaussian and target caption; it is not a fully anchor-free baseline.",
    ),
    Arm(
        "temporal",
        "E00_V15AR3_FROZEN_ZEROUPDATE_DYNSTATIC_TARGETOWNED_TEMPORAL_ROUTEON_K40_A100",
        "route_on_temporal",
        "self_target_owned_temporal_kernel_attn_output_v14r2",
        40,
        2288,
        4576,
        104,
        "e00-v15a-r3-dynstatic-temporal-k40.mp4",
        "Dynamic−static Q/K · temporal · K40",
        "Strict FAIL: #1→#2 persists; #2 is not lifted and late whitening, ghosting, and structural damage increase.",
    ),
    Arm(
        "activity25",
        "E00_V15AR3_FROZEN_ZEROUPDATE_DYNSTATIC_TARGETOWNED_ACTIVITY25_ROUTEON_K40_A100",
        "route_on_activity25",
        "self_target_owned_activity_kernel25_attn_output_v14r2",
        40,
        2288,
        4576,
        104,
        "e00-v15a-r3-dynstatic-activity25-k40.mp4",
        "Dynamic−static Q/K · activity25 · K40",
        "Strict FAIL: transparency is somewhat better than temporal/K0, but #1→#2 persists and #3 is unused. No donor V/pixels are copied, yet indirect content-conditioned leakage remains.",
    ),
)


class ReviewError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_plain_file(path: Path, description: str) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ReviewError(f"missing plain non-empty {description}: {path}")


def read_object(path: Path, description: str) -> dict[str, Any]:
    require_plain_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"invalid {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewError(f"{description} root must be an object: {path}")
    return value


def nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ReviewError(f"provenance is missing {'.'.join(keys)}")
        current = current[key]
    return current


def expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ReviewError(f"{label}: expected {expected!r}, got {actual!r}")


def path_is_bound(remote_path: Any, root: str, basename: str, label: str) -> None:
    if not isinstance(remote_path, str):
        raise ReviewError(f"{label} must be a path string")
    path = PurePosixPath(remote_path)
    expect(str(path.parent), root, f"{label} root")
    expect(path.name, basename, f"{label} basename")


def ffprobe_video(path: Path) -> dict[str, Any]:
    command = (
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_read_frames,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    )
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        value = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ReviewError(f"ffprobe failed for {path}: {exc}") from exc
    streams = value.get("streams") if isinstance(value, dict) else None
    if not isinstance(streams, list) or len(streams) != 1:
        raise ReviewError(f"expected exactly one video stream: {path}")
    stream = streams[0]
    summary = {
        "frames": int(stream.get("nb_read_frames", -1)),
        "fps": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
    }
    if (
        summary["frames"] != 81
        or summary["fps"] != "25/1"
        or summary["r_frame_rate"] != "25/1"
        or summary["width"] <= 0
        or summary["height"] <= 0
    ):
        raise ReviewError(f"video is not exact 81-frame 25-fps output: {path}: {summary}")
    return summary


def validate_zero_update(value: Mapping[str, Any], label: str) -> None:
    expected = {
        "adapter_present": False,
        "base_frozen_before_and_after": True,
        "optimization_steps": 0,
        "trained_checkpoint_loaded": False,
        "training_performed": False,
    }
    actual = {key: value.get(key) for key in expected}
    expect(actual, expected, label)


def validate_triplet_bundle(decoded_root: Path) -> dict[str, Any]:
    if not decoded_root.is_dir() or decoded_root.is_symlink():
        raise ReviewError(f"decoded root must be a plain directory: {decoded_root}")
    marker = decoded_root / "V15A_MATCHED_TRIPLET_COMPLETE"
    if not marker.is_file() or marker.is_symlink():
        raise ReviewError(f"missing matched-triplet completion marker: {marker}")

    manifest_path = decoded_root / "v15a_matched_triplet_manifest.json"
    launch_path = decoded_root / "v15a_launch_receipt.json"
    manifest = read_object(manifest_path, "triplet manifest")
    launch = read_object(launch_path, "launch receipt")
    expect(manifest.get("schema_version"),
           "bernini-v15a-r3-zero-update-matched-triplet-manifest-v1",
           "triplet schema")
    expect(manifest.get("complete"), True, "triplet complete")
    expect(manifest.get("experiment_tag"), EXPERIMENT_TAG, "triplet experiment tag")
    expect(manifest.get("fixed_output_root"), FIXED_OUTPUT_ROOT, "triplet root")
    expect(manifest.get("comparison_group_id"), GROUP_ID, "triplet group")
    expect(manifest.get("probe_kind"), PROBE_KIND, "triplet probe kind")
    expect(nested(manifest, "launch_receipt", "sha256"), sha256(launch_path),
           "launch receipt SHA-256")
    path_is_bound(nested(manifest, "launch_receipt", "path"), FIXED_OUTPUT_ROOT,
                  "v15a_launch_receipt.json", "manifest launch receipt")

    contract = nested(manifest, "matched_contract")
    expected_contract = {
        "adapter": None,
        "anchor_action_reward_used_for_sga": False,
        "anchor_contrast_mode": "dynamic_static_same_caption",
        "anchor_sha256": ANCHOR_SHA256,
        "anchor_state_mode": "clean_noised",
        "arm": "AQK_SGA5",
        "event": 0,
        "fps": 25,
        "frames": 81,
        "frozen_base": True,
        "initial_noise_proposal_mode": "anchor_candidate0",
        "initial_phase_clamp": True,
        "optimization_steps": 0,
        "outer_seed": 2027,
        "sga_score_mode": "global_source_cosine",
        "sga_weights_forced_to_anchor_candidate0": False,
        "source_cfg_scale": 4.5,
        "source_sha256": SOURCE_SHA256,
        "target_cfg_scale": 4.5,
        "trained_checkpoint": None,
        "training_performed": False,
    }
    for key, expected_value in expected_contract.items():
        expect(contract.get(key), expected_value, f"triplet contract {key}")
    expect(tuple(contract.get("candidate_counts", ())), EXPECTED_CANDIDATE_COUNTS,
           "triplet candidate counts")

    expect(launch.get("schema_version"),
           "bernini-v15a-r3-max-strength-clean-noised-route-probe-launch-v1",
           "launch schema")
    expect(launch.get("complete"), True, "launch complete")
    expect(launch.get("experiment_tag"), EXPERIMENT_TAG, "launch experiment tag")
    expect(launch.get("fixed_output_root"), FIXED_OUTPUT_ROOT, "launch root")
    expect(launch.get("parent_job_id"), "143808", "launch parent job")
    expect(launch.get("compute_node"), "auh7-1b-gpu-292", "launch compute node")
    expect(launch.get("zero_update_frozen_base"), {
        "adapter": None,
        "checkpoint": None,
        "optimization_steps": 0,
        "training_performed": False,
    }, "launch zero-update contract")

    manifest_arms = manifest.get("arms")
    launch_arms = launch.get("arms")
    if not isinstance(manifest_arms, list) or not isinstance(launch_arms, list):
        raise ReviewError("triplet and launch arms must be arrays")
    expected_arm_rows = [
        (arm.label, arm.role, arm.transport_steps) for arm in ARMS
    ]
    expect([(row.get("label"), row.get("role"), row.get("transport_steps"))
            for row in manifest_arms], expected_arm_rows, "triplet arm labels/order")
    expect([(row.get("label"), row.get("role"), row.get("transport_steps"))
            for row in launch_arms], expected_arm_rows, "launch arm labels/order")

    validated_arms: list[dict[str, Any]] = []
    for arm, manifest_arm in zip(ARMS, manifest_arms):
        video_path = decoded_root / arm.mp4_name
        receipt_path = decoded_root / f"{arm.mp4_name}.receipt.json"
        audit_path = decoded_root / f"{arm.mp4_name}.v15a-audit.json"
        require_plain_file(video_path, "v15a-r3 MP4")
        receipt = read_object(receipt_path, "native receipt")
        audit = read_object(audit_path, "v15a audit")
        video_sha = sha256(video_path)
        receipt_sha = sha256(receipt_path)
        audit_sha = sha256(audit_path)
        expect(manifest_arm.get("audit_sha256"), audit_sha, f"{arm.key} audit SHA-256")
        path_is_bound(manifest_arm.get("audit_path"), FIXED_OUTPUT_ROOT,
                      audit_path.name, f"{arm.key} manifest audit")

        expect(receipt.get("schema_version"),
               "bernini-pure-t2v-anchor-sga-anc-event-canary-v47",
               f"{arm.key} receipt schema")
        expect(receipt.get("complete"), True, f"{arm.key} receipt complete")
        expect(receipt.get("training_performed"), False, f"{arm.key} training")
        expect(receipt.get("optimization_steps"), 0, f"{arm.key} optimization steps")
        expect(receipt.get("loaded_trained_attention_checkpoint"), False,
               f"{arm.key} trained checkpoint loaded")
        expect(receipt.get("trained_attention_checkpoint"), None,
               f"{arm.key} trained checkpoint")
        expect(nested(receipt, "source", "sha256"), SOURCE_SHA256,
               f"{arm.key} source SHA-256")
        expect(nested(receipt, "pure_t2v_anchor", "sha256"), ANCHOR_SHA256,
               f"{arm.key} anchor SHA-256")
        for freeze_key in ("freeze_before", "freeze_after"):
            freeze = nested(receipt, freeze_key)
            expect({key: freeze.get(key) for key in (
                "adapter_modules_absent", "base_frozen", "lora_module_count",
                "trainable_parameter_elements", "trainable_parameter_tensors")}, {
                "adapter_modules_absent": True,
                "base_frozen": True,
                "lora_module_count": 0,
                "trainable_parameter_elements": 0,
                "trainable_parameter_tensors": 0,
            }, f"{arm.key} {freeze_key}")
        output = nested(receipt, "output")
        expect(output.get("sha256"), video_sha, f"{arm.key} receipt video SHA-256")
        expect(output.get("frames"), 81, f"{arm.key} receipt frames")
        expect(output.get("fps"), 25, f"{arm.key} receipt FPS")
        path_is_bound(output.get("path"), FIXED_OUTPUT_ROOT, arm.mp4_name,
                      f"{arm.key} receipt output")

        mechanism = nested(receipt, "mechanism")
        for key, expected_value in {
            "anchor_contrast_mode": "dynamic_static_same_caption",
            "anchor_state_mode": "clean_noised",
            "initial_noise_proposal_mode": "anchor_candidate0",
            "initial_phase_clamp": True,
            "anchor_cfg_scope": "shared",
            "early_candidate_count": 5,
            "source_cfg_scale": 4.5,
            "target_cfg_scale": 4.5,
        }.items():
            expect(mechanism.get(key), expected_value, f"{arm.key} mechanism {key}")
        trace = nested(mechanism, "trace")
        cache = nested(trace, "attention_cache")
        expect(cache.get("qk_only_capture_count"), arm.qk_capture_count,
               f"{arm.key} native QK capture count")
        expect(cache.get("qk_only_replay_count"), arm.qk_replay_count,
               f"{arm.key} native QK replay count")
        expect(trace.get("anchor_model_forwards"), arm.anchor_model_forwards,
               f"{arm.key} anchor forwards")
        expect(trace.get("anchor_value_stream_copied"), False,
               f"{arm.key} no anchor value copy")
        expect(trace.get("anchor_absolute_qk_or_k_replacement"), False,
               f"{arm.key} no absolute QK replacement")
        causal = nested(receipt, "causal_control")
        expect(causal.get("transport_steps"), arm.transport_steps,
               f"{arm.key} native transport steps")
        expect(causal.get("anchor_injection_enabled"), arm.transport_steps > 0,
               f"{arm.key} native route state")

        expect(audit.get("schema_version"),
               "bernini-v15a-r3-max-strength-clean-noised-route-probe-audit-v1",
               f"{arm.key} audit schema")
        expect(audit.get("complete"), True, f"{arm.key} audit complete")
        expect(audit.get("experiment_tag"), EXPERIMENT_TAG,
               f"{arm.key} audit experiment tag")
        expect(audit.get("probe_kind"), PROBE_KIND, f"{arm.key} audit probe kind")
        expect(nested(audit, "native_sidecar", "sha256"), receipt_sha,
               f"{arm.key} native sidecar SHA-256")
        path_is_bound(nested(audit, "native_sidecar", "path"), FIXED_OUTPUT_ROOT,
                      receipt_path.name, f"{arm.key} audit native sidecar")
        audit_output = nested(audit, "output")
        expect(audit_output.get("sha256"), video_sha, f"{arm.key} audit video SHA-256")
        expect(audit_output.get("frames"), 81, f"{arm.key} audit frames")
        expect(audit_output.get("fps"), 25, f"{arm.key} audit FPS")
        path_is_bound(audit_output.get("path"), FIXED_OUTPUT_ROOT, arm.mp4_name,
                      f"{arm.key} audit output")
        comparison = nested(audit, "comparison")
        expect(comparison.get("exact_label"), arm.mp4_name,
               f"{arm.key} exact r3 label")
        expect(comparison.get("fixed_output_root"), FIXED_OUTPUT_ROOT,
               f"{arm.key} comparison root")
        expect(comparison.get("group_id"), GROUP_ID, f"{arm.key} comparison group")
        expect(comparison.get("arm_role"), arm.role, f"{arm.key} arm role")
        expect(comparison.get("pairable"), True, f"{arm.key} pairable")
        validate_zero_update(nested(audit, "zero_update_frozen_base"),
                             f"{arm.key} zero-update audit")
        qk = nested(audit, "qk_route_proof")
        expect(qk.get("active_steps"), arm.transport_steps, f"{arm.key} QK active steps")
        expect(qk.get("capture_count"), arm.qk_capture_count, f"{arm.key} QK captures")
        expect(qk.get("replay_count"), arm.qk_replay_count, f"{arm.key} QK replays")
        expect(qk.get("route_injection_enabled"), arm.transport_steps > 0,
               f"{arm.key} QK route state")
        expect(qk.get("strength"), 1.0, f"{arm.key} QK strength")
        expect(qk.get("transport"), arm.transport, f"{arm.key} QK transport")
        expect(qk.get("donor_value_or_pixels_used"), False,
               f"{arm.key} QK-only donor contract")
        expected_cached = ["query", "key"] if arm.transport_steps else []
        expect(qk.get("cached_fields"), expected_cached, f"{arm.key} cached fields")
        proof = nested(audit, "contrast_proof")
        expect(proof.get("contrast_pair_executed"), arm.transport_steps > 0,
               f"{arm.key} contrast execution")
        if arm.transport_steps:
            for key in ("same_caption", "same_exact_candidate_noise", "same_model_timestep"):
                expect(proof.get(key), True, f"{arm.key} contrast {key}")
        probe = ffprobe_video(video_path)
        validated_arms.append({
            "arm": arm,
            "video": video_path,
            "receipt": receipt_path,
            "audit": audit_path,
            "video_sha256": video_sha,
            "receipt_sha256": receipt_sha,
            "audit_sha256": audit_sha,
            "probe": probe,
        })
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256(manifest_path),
        "launch": launch,
        "launch_path": launch_path,
        "launch_sha256": sha256(launch_path),
        "marker_path": marker,
        "arms": validated_arms,
    }


def validate_baselines(
    baseline_media: Path,
    probe: Callable[[Path], dict[str, Any]] = ffprobe_video,
) -> list[dict[str, Any]]:
    if not baseline_media.is_dir() or baseline_media.is_symlink():
        raise ReviewError(f"baseline media must be a plain directory: {baseline_media}")
    result = []
    for baseline in BASELINES:
        path = baseline_media / baseline.filename
        require_plain_file(path, f"{baseline.key} baseline MP4")
        expect(sha256(path), baseline.sha256, f"{baseline.key} baseline SHA-256")
        result.append({"baseline": baseline, "path": path, "probe": probe(path)})
    return result


def copy_checked(source: Path, destination: Path, expected_sha: str) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise ReviewError(f"duplicate output artifact: {destination}")
    shutil.copy2(source, destination)
    actual_sha = sha256(destination)
    expect(actual_sha, expected_sha, f"copied artifact {destination.name} SHA-256")
    return {"path": f"media/{destination.name}", "sha256": actual_sha,
            "bytes": destination.stat().st_size}


def render_card(card: Mapping[str, Any]) -> str:
    return f'''<article class="card"><div class="card-title">{html.escape(str(card["label"]))}</div>
<div class="video-shell"><video controls muted playsinline preload="metadata" src="{html.escape(str(nested(card, "artifact", "path")), quote=True)}"></video></div>
<div class="detail">{html.escape(str(card["detail"]))}</div></article>'''


def render(receipt: Mapping[str, Any]) -> str:
    rows = []
    for row_index, row in enumerate(receipt["rows"]):
        group = f"row-{row_index}"
        cards = "".join(render_card(card) for card in row["cards"])
        rows.append(f'''<section class="comparison-row" id="{group}"><div class="row-head"><div><h2>{html.escape(str(row["title"]))}</h2><p>{html.escape(str(row["note"]))}</p></div>
<button type="button" onclick="syncPlay('#{group} video',this)">同步播放本行</button></div><div class="grid">{cards}</div></section>''')
    legend = receipt["object_legend"]
    legend_html = "".join(
        f'<li><strong>#{item["id"]}</strong> {html.escape(str(item["description"]))}</li>'
        for item in legend
    )
    result_warning = html.escape(str(receipt["strict_result_warning"]))
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(receipt["title"]))}</title><style>
:root{{--bg:#f4f0e8;--panel:#fffdf8;--ink:#17211e;--muted:#616d68;--line:#d2c6b5;--accent:#176b57;--warn:#8a4a16}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.38 system-ui,-apple-system,"PingFang SC",sans-serif}}
.top{{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:12px;padding:9px 14px;background:#f4f0e8f2;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
.top h1{{margin:0;font-size:18px}}.top .meta{{color:var(--muted);font-size:12px}}.top button{{margin-left:auto}}
button{{border:1px solid #9d907c;background:#fffaf1;border-radius:8px;padding:7px 11px;font-weight:750;cursor:pointer;white-space:nowrap}}button:disabled{{opacity:.5}}
main{{max-width:1260px;margin:0 auto;padding:10px}}.notice{{display:grid;grid-template-columns:minmax(250px,.85fr) minmax(420px,1.5fr);gap:10px;padding:10px 12px;background:var(--panel);border:1px solid #c99262;border-radius:11px}}
.notice h2{{margin:0 0 5px;font-size:15px;color:var(--warn)}}.notice p{{margin:0;color:var(--muted);font-size:12px}}.notice ul{{margin:0;padding-left:22px;color:var(--ink);font-size:12px}}.notice li+li{{margin-top:3px}}
.result-warning{{margin-top:8px;padding:9px 11px;border:1px solid #aa4b3f;border-radius:9px;background:#fff3ef;color:#6f211a;font-size:12px}}.result-warning strong{{font-size:13px}}
.comparison-row{{margin-top:10px;padding:9px;background:var(--panel);border:1px solid var(--line);border-radius:11px}}.row-head{{display:flex;align-items:start;gap:10px;margin-bottom:7px}}.row-head>div{{flex:1;min-width:0}}h2{{margin:0;font-size:15px}}.row-head p{{margin:1px 0 0;color:var(--muted);font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;align-items:stretch}}.card{{display:grid;grid-template-rows:48px auto minmax(50px,1fr);min-width:0;overflow:hidden;background:#fff;border:1px solid #719386;border-radius:9px}}.card-title{{display:flex;align-items:center;padding:6px 8px;font-weight:800;font-size:13px;line-height:1.18}}
.video-shell{{width:100%;aspect-ratio:2/3;background:#0b0c0b;overflow:hidden}}video{{display:block;width:100%;height:100%;object-fit:contain;background:#0b0c0b}}.detail{{padding:6px 8px;color:var(--muted);font-size:11px}}
.footer{{padding:9px 2px 2px;color:var(--muted);font-size:11px}}@media(max-width:720px){{main{{min-width:690px}}body{{overflow-x:auto}}.top{{min-width:690px}}}}
</style></head><body><div class="top"><h1>{html.escape(str(receipt["title"]))}</h1><span class="meta">E00 · 6 videos · 2×3 · zero update</span><button type="button" onclick="pauseAll()">全部暂停</button></div>
<main id="event-0"><section class="notice"><div><h2>这不是干净的主 benchmark</h2><p>{html.escape(str(receipt["benchmark_warning"]))}</p></div><div><h2>Source 对象编号与目标关系</h2><ul>{legend_html}</ul></div></section>
<div class="result-warning"><strong>严格结果：route-on 0/2。</strong> {result_warning}</div>
<div class="event-control"><button type="button" onclick="syncPlay('#event-0 video',this)">同步播放本事件（6条）</button></div>{''.join(rows)}
<div class="footer">页面只陈列 provenance 已锁定的视频，不提供表单，也不显示机器答案。Pure-T2V anchor 的外观不是目标；“只缓存 Q/K、未复制 donor V/像素”是数据流事实，不等于输出已经 appearance-invariant。</div></main>
<script>function metadata(v){{if(v.readyState>=1)return Promise.resolve();return new Promise((ok,bad)=>{{v.addEventListener('loadedmetadata',ok,{{once:true}});v.addEventListener('error',()=>bad(Error('media load failed: '+v.currentSrc)),{{once:true}});v.load()}})}}
async function syncPlay(selector,button){{const vs=[...document.querySelectorAll(selector)];const old=button.textContent;button.disabled=true;button.textContent='加载并对齐…';try{{vs.forEach(v=>{{v.pause();v.muted=true;v.currentTime=0}});await Promise.all(vs.map(metadata));vs.forEach(v=>v.currentTime=0);const result=await Promise.allSettled(vs.map(v=>v.play()));if(result.some(x=>x.status==='rejected'))throw Error('浏览器拒绝部分视频播放')}}catch(error){{alert('同步播放失败：'+error.message+'。请通过本地 HTTP 服务打开并检查 media。')}}finally{{button.disabled=false;button.textContent=old}}}}
function pauseAll(){{document.querySelectorAll('video').forEach(v=>v.pause())}}</script></body></html>'''


def build(decoded_root: Path, baseline_media: Path, output: Path) -> dict[str, Any]:
    decoded_root = decoded_root.resolve()
    baseline_media = baseline_media.resolve()
    output = output.resolve()
    if output == Path("/") or output.exists() or output.is_symlink():
        raise ReviewError("output must be a fresh non-root path")
    staging = output.with_name(f".{output.name}.building")
    if staging.exists() or staging.is_symlink():
        raise ReviewError(f"staging path must be absent: {staging}")
    if not output.parent.is_dir():
        raise ReviewError(f"output parent must already exist: {output.parent}")

    # Fail closed before creating either output or staging.
    bundle = validate_triplet_bundle(decoded_root)
    baselines = validate_baselines(baseline_media)

    staging.mkdir()
    media = staging / "media"
    provenance = staging / "provenance"
    media.mkdir()
    provenance.mkdir()
    authority_cards = []
    for item in baselines:
        baseline = item["baseline"]
        artifact = copy_checked(item["path"], media / baseline.output_name, baseline.sha256)
        authority_cards.append({
            "key": baseline.key,
            "label": baseline.label,
            "detail": baseline.detail,
            "artifact": artifact,
            "probe": item["probe"],
        })
    route_cards = []
    for item in bundle["arms"]:
        arm = item["arm"]
        artifact = copy_checked(item["video"], media / arm.output_name,
                                item["video_sha256"])
        route_cards.append({
            "key": arm.key,
            "label": arm.card_label,
            "detail": arm.detail,
            "artifact": artifact,
            "probe": item["probe"],
            "qk_capture_count": arm.qk_capture_count,
            "qk_replay_count": arm.qk_replay_count,
            "transport_steps": arm.transport_steps,
        })
        for kind in ("receipt", "audit"):
            copy_checked(item[kind], provenance / item[kind].name,
                         item[f"{kind}_sha256"])
    copy_checked(bundle["manifest_path"], provenance / bundle["manifest_path"].name,
                 bundle["manifest_sha256"])
    copy_checked(bundle["launch_path"], provenance / bundle["launch_path"].name,
                 bundle["launch_sha256"])
    shutil.copy2(bundle["marker_path"], provenance / bundle["marker_path"].name)

    receipt: dict[str, Any] = {
        "schema_version": "bernini-v15a-r3-dynamic-static-routeprobe-review-v1",
        "complete": True,
        "title": "v15a-r3 · E00 dynamic−static Q/K matched route probe",
        "experiment_tag": EXPERIMENT_TAG,
        "probe_kind": PROBE_KIND,
        "zero_update": True,
        "training_performed": False,
        "machine_correct_answer_shown": False,
        "human_annotation_controls_shown": False,
        "pure_t2v_anchor_appearance_is_target": False,
        "strict_route_on_success_count": 0,
        "strict_route_on_trial_count": 2,
        "indirect_anchor_conditioned_leakage_observed": True,
        "matched_route_off_is_fully_anchor_free": False,
        "shared_anchor_derived_candidate0_gaussian": True,
        "strict_result_warning": (
            "两条K40都没有完成#2→#3；它们仍沿用#1→#2。中央透明器皿#2的乳白化/不透明化"
            "与pure-T2V anchor的白色容器外观一致，属于需要警惕的间接anchor-conditioned content leakage；"
            "但matched K0也出现乳白化，所以不能把全部白化都因果归给Q/K route；K0仍共享anchor-derived candidate-0 Gaussian"
            "和target caption，并非完全anchor-free。temporal进一步加重重影和结构损坏，"
            "activity25只部分保住透明材质，仍未解决对象角色。Q/K-only不是appearance-free。"
        ),
        "benchmark_warning": (
            "当前请求不是普通的“壶倒入杯”：Source 初始/原动作是 #1→#2，而请求要 #2→#3。"
            "它要求把原 recipient 改成 actor，并与 Source 已在进行的倒水初态冲突。"
            "因此它是困难 role-switch / relation edit；运行 prompt 中的“glass pitcher”也有杯/壶形态歧义，"
            "后续主 benchmark 应用对象位置与原角色消歧。"
        ),
        "object_legend": [
            {"id": 1, "description": "左上白色陶瓷倒水器皿：Source actor（当前从它倒出）。"},
            {"id": 2, "description": "中央/右下透明带柄玻璃器皿：Source recipient；请求中的 desired actor。"},
            {"id": 3, "description": "左下小白茶杯：请求中的 desired recipient。"},
        ],
        "desired_relation": "#2 → #3",
        "source_relation": "#1 → #2",
        "layout": {
            "row_count": 2,
            "cards_per_row": [3, 3],
            "max_cards_per_row": 3,
            "equal_height_video_shells": True,
            "event_sync_controls": 1,
            "row_sync_controls": 2,
        },
        "input_provenance": {
            "triplet_manifest_sha256": bundle["manifest_sha256"],
            "launch_receipt_sha256": bundle["launch_sha256"],
            "fixed_output_root": FIXED_OUTPUT_ROOT,
        },
        "rows": [
            {
                "title": "Authority / action donor / matched frozen baseline",
                "note": "先用 Source 识别三个器皿与初态；anchor 外观无效。",
                "cards": authority_cards,
            },
            {
                "title": "Same frozen weights and sampler · matched route triplet",
                "note": "K0 与两条 K40 仅改变 dynamic−static Q/K 路由；三条均为 0 update，但都共享anchor-derived candidate-0 Gaussian和target caption。两条 route-on 严格 #2→#3 成功为 0/2。",
                "cards": route_cards,
            },
        ],
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (staging / "index.html").write_text(render(receipt), encoding="utf-8")
    (staging / "COMPLETE").write_text("complete\n", encoding="ascii")
    staging.rename(output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-root", required=True)
    parser.add_argument("--baseline-media", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    receipt = build(
        Path(args.decoded_root).expanduser(),
        Path(args.baseline_media).expanduser(),
        output,
    )
    index_path = output / "index.html"
    print(json.dumps({
        "output": str(output),
        "videos": sum(len(row["cards"]) for row in receipt["rows"]),
        "rows": receipt["layout"]["row_count"],
        "cards_per_row": receipt["layout"]["cards_per_row"],
        "index_sha256": sha256(index_path),
        "manifest_sha256": sha256(output / "manifest.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

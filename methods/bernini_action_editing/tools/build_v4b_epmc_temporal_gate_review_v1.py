#!/usr/bin/env python3
"""Build the detached two-seed v4-B/EPMC temporal-gate video review.

The source video is both the hash-pinned Bernini model input and a display
reference.  The action-anchor RGB enters only this detached display builder;
its already-extracted OOF action feature is the privileged gate-state input.
The page provides synchronized playback of Source | Anchor | B0 | Zero |
Correct | Reverse | Shuffle for render seeds 2028 and 2029.  Every label and
receipt keeps the scope at temporal-gating diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping, Optional, Sequence

from methods.bernini_action_editing import infer_v4b_epmc_temporal_gate_canary_v1 as runtime
from methods.bernini_action_editing import materialize_v4b_epmc_gate_state_v1 as gate_materializer


SCHEMA = "bernini-v4b-epmc-temporal-gate-review-packet-v1"
SEEDS = runtime.RENDER_SEEDS
ARMS = runtime.ARM_ORDER
EXPECTED_SOURCE_SHA256 = runtime.EXPECTED_SOURCE_SHA256
EXPECTED_ANCHOR_SHA256 = gate_materializer.EXPECTED_ANCHOR_VIDEO_SHA256
_MAX_RECEIPT_BYTES = 32 << 20


class V4BEPMCReviewError(RuntimeError):
    """A review input or detached-display boundary differed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise V4BEPMCReviewError("value is not canonical JSON") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(value: str | Path, *, label: str, maximum_bytes: int | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise V4BEPMCReviewError(f"{label} must be an absolute path")
    try:
        info = path.lstat()
    except OSError as error:
        raise V4BEPMCReviewError(f"cannot stat {label}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise V4BEPMCReviewError(f"{label} must be a plain nlink1 file")
    if maximum_bytes is not None and not 0 < info.st_size <= maximum_bytes:
        raise V4BEPMCReviewError(f"{label} size exceeds its bound")
    return path.resolve(strict=True)


def _strict_receipt(path: Path, *, seed: int) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4BEPMCReviewError(f"seed {seed} receipt is not ASCII JSON") from error
    if type(value) is not dict:
        raise V4BEPMCReviewError(f"seed {seed} receipt must be an object")
    digest = value.get("receipt_digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    claims = value.get("verified_claims")
    ancestry = value.get("end_to_end_data_ancestry")
    if (
        type(digest) is not str
        or _object_sha256(unsigned) != digest
        or value.get("schema_version") != runtime.RECEIPT_SCHEMA
        or value.get("status")
        != "V4B_EPMC_TEMPORAL_GATE_VIDEO_CANARY_COMPLETE_DIAGNOSTIC_ONLY"
        or value.get("iid") != runtime.EXPECTED_IID
        or value.get("temporal_gating_diagnostic_only") is not True
        or value.get("v4b_aggregate_gate_verified_true") is not True
        or value.get("renderer_qualified") is not False
        or value.get("video_editing_qualified") is not False
        or value.get("seeds") != {"proposal": runtime.PROPOSAL_SEED, "render": seed}
        or value.get("arms", {}).get("order") != list(ARMS)
        or not isinstance(claims, Mapping)
        or not all(claims.values())
        or not isinstance(ancestry, Mapping)
        or ancestry.get("gate_state_is_privileged_action_anchor_feature_derived")
        is not True
        or ancestry.get("heldout_action_anchor_rgb_consumed") is not False
        or ancestry.get("source_plus_instruction_only_end_to_end_claim") is not False
    ):
        raise V4BEPMCReviewError(f"seed {seed} receipt closure differs")
    outputs = value.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(ARMS):
        raise V4BEPMCReviewError(f"seed {seed} output manifest differs")
    return value


def _copy_sealed(source: Path, destination: Path, *, expected_sha256: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise V4BEPMCReviewError(f"refusing to overwrite {destination}")
    if _file_sha256(source) != expected_sha256:
        raise V4BEPMCReviewError(f"source hash differs for {source}")
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1 << 20)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.chmod(destination, 0o444)
    info = destination.stat()
    if (
        stat.S_IMODE(info.st_mode) != 0o444
        or info.st_nlink != 1
        or _file_sha256(destination) != expected_sha256
    ):
        raise V4BEPMCReviewError("sealed review copy differs")


def _video_card(label: str, relative: str, note: str, css: str = "") -> str:
    return (
        f'<article class="card {escape(css)}"><h3>{escape(label)}</h3>'
        f'<video controls muted loop playsinline preload="metadata" '
        f'src="{escape(relative)}"></video><p>{escape(note)}</p></article>'
    )


def _html(*, instruction: str, receipt_rows: Mapping[int, Mapping[str, Any]]) -> str:
    sections: list[str] = []
    for seed in SEEDS:
        prefix = f"media/seed{seed}"
        cards = [
            _video_card("Source (input + reference)", "media/source.mp4", "Bernini model input and display reference", "ref"),
            _video_card("Anchor (reference)", "media/anchor.mp4", "Display-only privileged action reference; not passed to Bernini", "ref anchor"),
            _video_card("B0", f"{prefix}/B0.mp4", "Frozen Bernini no branch", "base"),
            _video_card("Zero", f"{prefix}/zero.mp4", "Installed real hook with byte-exact zero gate", "control"),
            _video_card("Correct", f"{prefix}/correct.mp4", "v4-B decoded-residual temporal order", "candidate"),
            _video_card("Reverse", f"{prefix}/reverse.mp4", "Same gate multiset, reversed phase order", "negative"),
            _video_card("Shuffle", f"{prefix}/shuffle.mp4", "Same gate multiset, frozen shuffle", "negative"),
        ]
        receipt = receipt_rows[seed]
        sections.append(
            f'<section data-sync-group="seed{seed}"><header><div><p class="eyebrow">render seed {seed}</p>'
            f'<h2>Matched five-arm intervention</h2></div><div class="controls">'
            '<button data-action="play">同步播放 / 暂停</button>'
            '<button data-action="zero">全部归零</button>'
            '<button data-action="phase">按归一化进度对齐</button>'
            '<label>速度 <select data-action="rate"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label>'
            f'</div></header><div class="grid">{"".join(cards)}</div>'
            f'<details><summary>Receipt / gate provenance</summary><pre>{escape(json.dumps({"receipt_digest": receipt["receipt_digest"], "gate_state": receipt["gate_state"], "hook": receipt["hook"]}, indent=2, sort_keys=True))}</pre>'
            f'<p><a href="receipts/seed{seed}.json">seed {seed} full receipt</a></p></details></section>'
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v4-B → EPMC temporal-gating diagnostic</title>
<style>
:root{{--bg:#07101b;--panel:#101d2e;--line:#29425e;--text:#eef6ff;--muted:#a9bdd2;--cyan:#68d9e8;--amber:#ffc66d;--red:#ff7d8b}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#153a52,var(--bg) 44%);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{width:min(1900px,97vw);margin:auto;padding:34px 0 80px}}h1{{font-size:clamp(32px,5vw,62px);margin:.15em 0}}h2,h3,p{{margin-top:0}}.eyebrow{{color:var(--cyan);letter-spacing:.12em;text-transform:uppercase;font-weight:750}}
.warning{{border:1px solid #855e32;border-left:5px solid var(--amber);background:#2a2118;padding:16px 19px;border-radius:10px;margin:22px 0;font-size:17px}}
.instruction{{max-width:1200px;color:#d6e2ef;background:#0c1827;padding:14px 17px;border-radius:10px}}
section{{background:rgba(16,29,46,.96);border:1px solid var(--line);border-radius:16px;margin:24px 0;padding:18px}}section header{{display:flex;justify-content:space-between;gap:18px;align-items:center}}
.controls{{display:flex;gap:8px;flex-wrap:wrap}}button,select{{background:#17304a;color:var(--text);border:1px solid #3b6384;border-radius:8px;padding:8px 11px}}
.grid{{display:grid;grid-template-columns:repeat(7,minmax(210px,1fr));gap:9px;overflow-x:auto}}.card{{min-width:210px;background:#0a1522;border:1px solid var(--line);border-radius:11px;padding:9px}}.card h3{{font-size:14px;color:var(--cyan)}}.card p{{color:var(--muted);font-size:12px;min-height:38px}}
.card video{{width:100%;aspect-ratio:496/480;object-fit:contain;background:#020406;border-radius:7px}}.card.ref{{border-color:#526a7e}}.card.candidate{{border-color:#48a4b0}}.card.negative{{border-color:#79535d}}
details{{margin-top:13px;background:#0a1522;padding:11px;border-radius:9px}}summary{{cursor:pointer;color:var(--cyan)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#bed4e9;font-size:11px}}a{{color:var(--cyan)}}
@media(max-width:900px){{section header{{display:block}}}}
</style></head><body><main>
<p class="eyebrow">IID {escape(runtime.EXPECTED_IID)} · OOF fold 1 · proposal seed 2027</p>
<h1>v4-B → real Bernini EPMC temporal gating</h1>
<div class="warning"><strong>TEMPORAL-GATING DIAGNOSTIC ONLY.</strong> 这不是 action representation、renderer、画质或 video-editing qualification。Anchor 是页面中的 detached privileged reference；其 RGB 未进入 Bernini。Gate state 来自该 OOF anchor feature，因此也不能声称端到端 source+instruction-only。Block/head gate 为 0，所以 EPMC effective gate 是 0.5×profile；再乘 outer CPMR 0.10 后，实际 projected-motion residual 系数仅为 <strong>0.05×profile</strong>（source/phase0 为 0）。B0/Zero 只保证 final latent byte-exact，MP4 容器 byte parity 不作声明。</div>
<p class="instruction"><strong>Instruction:</strong> {escape(instruction)}</p>
{"".join(sections)}
</main><script>
for (const group of document.querySelectorAll('[data-sync-group]')) {{
  const videos=[...group.querySelectorAll('video')];
  group.querySelector('[data-action="play"]').onclick=async()=>{{const playing=videos.some(v=>!v.paused);if(playing)videos.forEach(v=>v.pause());else for(const v of videos){{try{{await v.play()}}catch(e){{}}}}}};
  group.querySelector('[data-action="zero"]').onclick=()=>videos.forEach(v=>{{v.pause();v.currentTime=0}});
  group.querySelector('[data-action="phase"]').onclick=()=>{{const leader=videos.find(v=>Number.isFinite(v.duration)&&v.duration>0);if(!leader)return;const p=leader.currentTime/leader.duration;videos.forEach(v=>{{if(Number.isFinite(v.duration)&&v.duration>0)v.currentTime=p*v.duration}})}};
  group.querySelector('[data-action="rate"]').onchange=e=>videos.forEach(v=>v.playbackRate=Number(e.target.value));
}}
</script></body></html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed2028-dir", required=True)
    parser.add_argument("--seed2029-dir", required=True)
    parser.add_argument("--expected-seed2028-receipt-sha256", required=True)
    parser.add_argument("--expected-seed2029-receipt-sha256", required=True)
    parser.add_argument("--source-video-ref", required=True)
    parser.add_argument("--anchor-video-ref", required=True)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--expected-anchor-sha256", default=EXPECTED_ANCHOR_SHA256)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_source_sha256 != EXPECTED_SOURCE_SHA256 or args.expected_anchor_sha256 != EXPECTED_ANCHOR_SHA256:
        raise V4BEPMCReviewError("source/detached-anchor authority pin differs")
    if hashlib.sha256(args.instruction.encode("utf-8")).hexdigest() != runtime.EXPECTED_INSTRUCTION_SHA256:
        raise V4BEPMCReviewError("instruction bytes differ")
    source = _plain_file(args.source_video_ref, label="source reference")
    anchor = _plain_file(args.anchor_video_ref, label="anchor reference")
    if _file_sha256(source) != EXPECTED_SOURCE_SHA256 or _file_sha256(anchor) != EXPECTED_ANCHOR_SHA256:
        raise V4BEPMCReviewError("source/detached-anchor SHA256 differs")

    run_dirs = {2028: Path(args.seed2028_dir).expanduser(), 2029: Path(args.seed2029_dir).expanduser()}
    receipt_hashes = {
        2028: args.expected_seed2028_receipt_sha256,
        2029: args.expected_seed2029_receipt_sha256,
    }
    receipts: dict[int, dict[str, Any]] = {}
    for seed, run_dir in run_dirs.items():
        if not run_dir.is_absolute() or run_dir.is_symlink() or not run_dir.is_dir():
            raise V4BEPMCReviewError(f"seed {seed} run directory differs")
        receipt_path = _plain_file(
            run_dir / "receipt.json",
            label=f"seed {seed} receipt",
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
        if _file_sha256(receipt_path) != receipt_hashes[seed]:
            raise V4BEPMCReviewError(f"seed {seed} receipt file SHA256 differs")
        receipts[seed] = _strict_receipt(receipt_path, seed=seed)
        if receipts[seed]["source"]["sha256"] != EXPECTED_SOURCE_SHA256:
            raise V4BEPMCReviewError(f"seed {seed} source join differs")
        if receipts[seed]["instruction_sha256"] != runtime.EXPECTED_INSTRUCTION_SHA256:
            raise V4BEPMCReviewError(f"seed {seed} instruction join differs")
    matched_fields = (
        "method_revision",
        "method_archive_sha256",
        "gate_state",
        "checkpoint",
        "source_revisions",
        "runtime_versions",
        "freeze_certificate",
        "proposal_latents",
        "carrier",
    )
    for field in matched_fields:
        if receipts[2028].get(field) != receipts[2029].get(field):
            raise V4BEPMCReviewError(
                f"two render seeds do not share matched {field} authority"
            )

    output = Path(args.output_dir).expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise V4BEPMCReviewError("output-dir must be a fresh absolute directory")
    output.mkdir(parents=False, exist_ok=False)
    media = output / "media"
    receipt_dir = output / "receipts"
    media.mkdir()
    receipt_dir.mkdir()
    _copy_sealed(source, media / "source.mp4", expected_sha256=EXPECTED_SOURCE_SHA256)
    _copy_sealed(anchor, media / "anchor.mp4", expected_sha256=EXPECTED_ANCHOR_SHA256)
    member_rows: list[dict[str, Any]] = [
        {
            "role": "source_input_and_reference",
            "relative_path": "media/source.mp4",
            "sha256": EXPECTED_SOURCE_SHA256,
        },
        {
            "role": "detached_anchor_reference",
            "relative_path": "media/anchor.mp4",
            "sha256": EXPECTED_ANCHOR_SHA256,
        },
    ]
    for seed in SEEDS:
        seed_media = media / f"seed{seed}"
        seed_media.mkdir()
        receipt_source = run_dirs[seed] / "receipt.json"
        _copy_sealed(
            receipt_source,
            receipt_dir / f"seed{seed}.json",
            expected_sha256=receipt_hashes[seed],
        )
        member_rows.append(
            {
                "role": "video_canary_receipt",
                "seed": seed,
                "relative_path": f"receipts/seed{seed}.json",
                "sha256": receipt_hashes[seed],
            }
        )
        for arm in ARMS:
            manifest = receipts[seed]["outputs"][arm]
            source_video = _plain_file(
                manifest["path"], label=f"seed {seed} {arm} output"
            )
            expected = manifest.get("mp4_sha256")
            if source_video.parent != run_dirs[seed].resolve(strict=True):
                raise V4BEPMCReviewError(f"seed {seed} {arm} escaped run directory")
            destination = seed_media / f"{arm}.mp4"
            _copy_sealed(source_video, destination, expected_sha256=expected)
            member_rows.append(
                {
                    "role": "generated_arm",
                    "seed": seed,
                    "arm": arm,
                    "relative_path": str(destination.relative_to(output)),
                    "sha256": expected,
                }
            )
    html = _html(instruction=args.instruction, receipt_rows=receipts).encode("utf-8")
    index = output / "index.html"
    with index.open("xb") as handle:
        handle.write(html)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(index, 0o444)
    packet: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "REVIEW_PACKET_COMPLETE_TEMPORAL_GATING_DIAGNOSTIC_ONLY",
        "iid": runtime.EXPECTED_IID,
        "render_seeds": list(SEEDS),
        "proposal_seed": runtime.PROPOSAL_SEED,
        "columns": ["Source", "Anchor", "B0", "Zero", "Correct", "Reverse", "Shuffle"],
        "source_reference_sha256": EXPECTED_SOURCE_SHA256,
        "anchor_reference_sha256": EXPECTED_ANCHOR_SHA256,
        "anchor_rgb_used_by_bernini_runtime": False,
        "gate_state_privileged_action_anchor_feature_derived": True,
        "effective_head_gate": "0.5*profile20",
        "total_projected_motion_residual_coefficient": "0.05*profile20",
        "b0_zero_byte_parity_scope": "final_latent_only_not_mp4_container",
        "temporal_gating_diagnostic_only": True,
        "action_representation_qualified": False,
        "renderer_qualified": False,
        "video_editing_qualified": False,
        "members": member_rows,
        "index_html_sha256": _file_sha256(index),
    }
    packet["receipt_digest"] = _object_sha256(packet)
    packet_path = output / "packet.json"
    raw = json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    with packet_path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(packet_path, 0o444)
    for directory in (media / "seed2028", media / "seed2029", media, receipt_dir, output):
        os.chmod(directory, 0o555)
    return {
        "index": str(index.resolve(strict=True)),
        "index_sha256": packet["index_html_sha256"],
        "packet": str(packet_path.resolve(strict=True)),
        "packet_sha256": _file_sha256(packet_path),
        "temporal_gating_diagnostic_only": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(_canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

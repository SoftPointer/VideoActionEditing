#!/usr/bin/env python3
"""Build the future-use detached v4-F/EPMC temporal-gate HTML review.

The page synchronizes Source | Anchor | B0 | Zero | Correct | Reverse |
Shuffle for render seeds 2028 and 2029.  Source is the hash-pinned Bernini
input and a display reference.  Anchor RGB is display-only and never enters
Bernini, while its previously extracted OOF V-JEPA feature is privileged input
to the gate materializer.  The packet therefore remains a temporal-gating
diagnostic and explicitly disclaims source+instruction-only inference.
The underlying codec gate is limited to known transform families exposed
during development; this review is not unseen-transform/action qualification.

This builder is intentionally unsealed and fails before parsing a path or
creating output until the v4-F canary runtime is fully pinned.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "bernini-v4f-epmc-temporal-gate-review-packet-v1"
STATUS = "REVIEW_PACKET_COMPLETE_TEMPORAL_GATING_DIAGNOSTIC_ONLY"
PIN_PLACEHOLDER = "TO_BE_PINNED"
RELEASE_SEALED = False
EXPECTED_RUNTIME_IMPLEMENTATION_SHA256 = PIN_PLACEHOLDER
EXPECTED_RUNTIME_RECEIPT_SCHEMA = PIN_PLACEHOLDER
EXPECTED_RUNTIME_RECEIPT_STATUS = PIN_PLACEHOLDER

SEEDS = (2028, 2029)
PROPOSAL_SEED = 2027
ARMS = ("B0", "zero", "correct", "reverse", "shuffle")
EXPECTED_IID = "7b88a1ca1f804f41"
EXPECTED_SOURCE_SHA256 = (
    "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
)
EXPECTED_ANCHOR_SHA256 = (
    "8234f5f35f7001134cf074263c481e3a8079c10f799370090d30e054aef02015"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "105ee8052a0f65d700736a8a25fdf02eb56f1b60d581403c328a8db3d500558c"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_RECEIPT_BYTES = 32 << 20


class V4FEPMCReviewError(RuntimeError):
    """A release authority, review input, or detached-display boundary differed."""


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
        raise V4FEPMCReviewError("value is not canonical JSON") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _required_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise V4FEPMCReviewError(f"{label} must be a lowercase SHA-256")
    return value


def _require_release_sealed() -> Any:
    """Fail before importing the runtime or touching a user-controlled path."""

    if (
        RELEASE_SEALED is not True
        or EXPECTED_RUNTIME_IMPLEMENTATION_SHA256 == PIN_PLACEHOLDER
        or EXPECTED_RUNTIME_RECEIPT_SCHEMA == PIN_PLACEHOLDER
        or EXPECTED_RUNTIME_RECEIPT_STATUS == PIN_PLACEHOLDER
        or _SHA256.fullmatch(EXPECTED_RUNTIME_IMPLEMENTATION_SHA256) is None
    ):
        raise V4FEPMCReviewError(
            "UNSEALED v4-F EPMC review builder: runtime pins are TO_BE_PINNED"
        )
    from methods.bernini_action_editing import (
        infer_v4f_epmc_temporal_gate_canary_v1 as runtime,
    )

    runtime._require_release_sealed()
    if (
        _file_sha256(Path(runtime.__file__).resolve(strict=True))
        != EXPECTED_RUNTIME_IMPLEMENTATION_SHA256
        or runtime.RECEIPT_SCHEMA != EXPECTED_RUNTIME_RECEIPT_SCHEMA
        or runtime.RECEIPT_STATUS != EXPECTED_RUNTIME_RECEIPT_STATUS
        or runtime.RENDER_SEEDS != SEEDS
        or runtime.ARM_ORDER != ARMS
    ):
        raise V4FEPMCReviewError("sealed v4-F runtime source/schema ABI differs")
    return runtime


def _plain_file(
    value: str | Path,
    *,
    label: str,
    maximum_bytes: int | None = None,
) -> Path:
    _require_release_sealed()
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise V4FEPMCReviewError(f"{label} must be an absolute path")
    try:
        info = path.lstat()
    except OSError as error:
        raise V4FEPMCReviewError(f"cannot stat {label}") from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise V4FEPMCReviewError(f"{label} must be a plain nlink1 file")
    if maximum_bytes is not None and not 0 < info.st_size <= maximum_bytes:
        raise V4FEPMCReviewError(f"{label} size exceeds its bound")
    return path.resolve(strict=True)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V4FEPMCReviewError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise V4FEPMCReviewError(f"non-finite JSON number: {value}")


def _strict_receipt(
    path: Path, *, seed: int, expected_sha256: str
) -> dict[str, Any]:
    runtime = _require_release_sealed()
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            raw = handle.read(_MAX_RECEIPT_BYTES + 1)
            after = os.fstat(handle.fileno())
            named = path.lstat()
        identity_fields = (
            "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_RECEIPT_BYTES
            or len(raw) != before.st_size
            or hashlib.sha256(raw).hexdigest() != expected_sha256
            or any(
                getattr(before, field) != getattr(after, field)
                or getattr(before, field) != getattr(named, field)
                for field in identity_fields
            )
            or stat.S_ISLNK(named.st_mode)
        ):
            raise V4FEPMCReviewError(
                f"seed {seed} receipt changed across single-FD read"
            )
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except V4FEPMCReviewError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4FEPMCReviewError(f"seed {seed} receipt is not ASCII JSON") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(value) is not dict:
        raise V4FEPMCReviewError(f"seed {seed} receipt must be an object")
    digest = value.get("receipt_digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    claims = value.get("verified_claims")
    ancestry = value.get("end_to_end_data_ancestry")
    closure = value.get("model_facing_input_closure")
    hook = value.get("hook")
    arms = value.get("arms")
    source = value.get("source")
    gate_state = value.get("gate_state")
    if (
        type(digest) is not str
        or _object_sha256(unsigned) != digest
        or value.get("schema_version") != runtime.RECEIPT_SCHEMA
        or value.get("status") != runtime.RECEIPT_STATUS
        or value.get("iid") != EXPECTED_IID
        or value.get("outer_fold") != 1
        or value.get("temporal_gating_diagnostic_only") is not True
        or value.get("v4f_aggregate_gate_verified_true") is not True
        or value.get("v4f_exact5_all_fold_inner_gates_passed") is not True
        or type(value.get("fold1_selected_rho")) is not float
        or value.get("fold1_selected_rho") not in (
            1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0,
            1.0 / 4.0, 1.0 / 2.0, 1.0,
        )
        or value.get("known_transform_families_exposed_during_model_fit") is not True
        or value.get("unseen_hostile_transform_gate") is not False
        or value.get("unseen_hostile_transform_gate_evaluated") is not False
        or value.get("unseen_action_qualification") is not False
        or value.get("scientific_claim") is not False
        or value.get("latent_metric_qualified") is not False
        or value.get("action_representation_qualified") is not False
        or value.get("identity_disentanglement_qualified") is not False
        or value.get("identity_preservation_qualified") is not False
        or value.get("prior_qualified") is not False
        or value.get("prior_generation_qualified") is not False
        or value.get("generation_qualified") is not False
        or value.get("renderer_qualified") is not False
        or value.get("video_editing_qualified") is not False
        or value.get("video_quality_claim") is not False
        or value.get("inference_authorized") is not False
        or value.get("web_evaluation_authorized") is not False
        or value.get("full644_refit_authorized") is not False
        or value.get("vae_necessary") is not None
        or value.get("seeds") != {"proposal": PROPOSAL_SEED, "render": seed}
        or not isinstance(arms, Mapping)
        or arms.get("order") != list(ARMS)
        or not isinstance(source, Mapping)
        or source.get("sha256") != EXPECTED_SOURCE_SHA256
        or value.get("instruction_sha256") != EXPECTED_INSTRUCTION_SHA256
        or not isinstance(gate_state, Mapping)
        or gate_state.get("file_sha256") != runtime.EXPECTED_GATE_STATE_FILE_SHA256
        or gate_state.get("receipt_digest")
        != runtime.EXPECTED_GATE_STATE_SELF_DIGEST
        or gate_state.get("v4f_aggregate_gate_verified_true") is not True
        or gate_state.get("v4f_exact5_all_fold_inner_gates_passed") is not True
        or gate_state.get("fold1_selected_rho") != value.get("fold1_selected_rho")
        or gate_state.get("known_exposed_transform_families_only") is not True
        or gate_state.get("unseen_hostile_transform_gate") is not False
        or gate_state.get("unseen_hostile_transform_gate_evaluated") is not False
        or not isinstance(claims, Mapping)
        or not claims
        or not all(item is True for item in claims.values())
        or claims.get(
            "aggregate_fold_receipt_preselection_selected_checkpoint_strong_join_before_render"
        )
        is not True
        or claims.get("v4f_exact5_all_fold_inner_gates_passed_before_render")
        is not True
        or claims.get("known_exposed_transform_boundary_preserved") is not True
        or not isinstance(ancestry, Mapping)
        or ancestry.get("gate_state_is_privileged_action_anchor_feature_derived")
        is not True
        or ancestry.get("heldout_action_anchor_rgb_consumed") is not False
        or ancestry.get("target_rgb_consumed") is not False
        or ancestry.get("source_plus_instruction_only_end_to_end_claim") is not False
        or not isinstance(closure, Mapping)
        or closure.get("anchor_video") is not False
        or closure.get("target_video") is not False
        or not isinstance(hook, Mapping)
        or hook.get("block_head_gates_all_exact_positive_zero") is not True
        or hook.get("total_coefficient_scale") != 0.05
        or hook.get("source_and_phase0_total_coefficient") != 0.0
    ):
        raise V4FEPMCReviewError(f"seed {seed} receipt closure differs")
    outputs = value.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(ARMS):
        raise V4FEPMCReviewError(f"seed {seed} output manifest differs")
    return value


def _copy_sealed(source: Path, destination: Path, *, expected_sha256: str) -> None:
    _require_release_sealed()
    _required_sha256(expected_sha256, label=f"{source} expected SHA256")
    if destination.exists() or destination.is_symlink():
        raise V4FEPMCReviewError(f"refusing to overwrite {destination}")
    if _file_sha256(source) != expected_sha256:
        raise V4FEPMCReviewError(f"source hash differs for {source}")
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
        raise V4FEPMCReviewError("sealed review copy differs")


def _video_card(label: str, relative: str, note: str, css: str = "") -> str:
    return (
        f'<article class="card {escape(css)}"><h3>{escape(label)}</h3>'
        f'<video controls muted loop playsinline preload="metadata" '
        f'src="{escape(relative)}"></video><p>{escape(note)}</p></article>'
    )


def _html(*, instruction: str, receipt_rows: Mapping[int, Mapping[str, Any]]) -> str:
    _require_release_sealed()
    sections: list[str] = []
    for seed in SEEDS:
        prefix = f"media/seed{seed}"
        cards = [
            _video_card(
                "Source (input + reference)",
                "media/source.mp4",
                "Bernini model input and display reference",
                "ref",
            ),
            _video_card(
                "Anchor (detached privileged reference)",
                "media/anchor.mp4",
                "RGB never passed to Bernini; its OOF V-JEPA feature created the gate",
                "ref anchor",
            ),
            _video_card("B0", f"{prefix}/B0.mp4", "Frozen Bernini no branch", "base"),
            _video_card(
                "Zero",
                f"{prefix}/zero.mp4",
                "Installed real hook with byte-exact zero gate",
                "control",
            ),
            _video_card(
                "Correct",
                f"{prefix}/correct.mp4",
                "v4-F decoded-residual temporal order",
                "candidate",
            ),
            _video_card(
                "Reverse",
                f"{prefix}/reverse.mp4",
                "Same gate multiset, reversed phase order",
                "negative",
            ),
            _video_card(
                "Shuffle",
                f"{prefix}/shuffle.mp4",
                "Same gate multiset, frozen shuffle",
                "negative",
            ),
        ]
        receipt = receipt_rows[seed]
        excerpt = {
            "receipt_digest": receipt["receipt_digest"],
            "gate_state": receipt["gate_state"],
            "fold1_selected_rho": receipt["fold1_selected_rho"],
            "hook": receipt["hook"],
            "end_to_end_data_ancestry": receipt["end_to_end_data_ancestry"],
        }
        sections.append(
            f'<section data-sync-group="seed{seed}"><header><div>'
            f'<p class="eyebrow">render seed {seed}</p>'
            f'<h2>Matched five-arm intervention</h2></div><div class="controls">'
            '<button data-action="play">同步播放 / 暂停</button>'
            '<button data-action="zero">全部归零</button>'
            '<button data-action="phase">按归一化进度对齐</button>'
            '<label>速度 <select data-action="rate"><option>0.5</option>'
            '<option selected>1</option><option>1.5</option><option>2</option>'
            f'</select></label></div></header><div class="grid">{"".join(cards)}</div>'
            f'<details><summary>Receipt / gate provenance</summary><pre>'
            f'{escape(json.dumps(excerpt, indent=2, sort_keys=True))}</pre>'
            f'<p><a href="receipts/seed{seed}.json">seed {seed} full receipt</a>'
            f'</p></details></section>'
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v4-F → EPMC temporal-gating diagnostic</title>
<style>
:root{{--bg:#07101b;--panel:#101d2e;--line:#29425e;--text:#eef6ff;--muted:#a9bdd2;--cyan:#68d9e8;--amber:#ffc66d;--red:#ff7d8b}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#153a52,var(--bg) 44%);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{width:min(1900px,97vw);margin:auto;padding:34px 0 80px}}h1{{font-size:clamp(32px,5vw,62px);margin:.15em 0}}h2,h3,p{{margin-top:0}}.eyebrow{{color:var(--cyan);letter-spacing:.12em;text-transform:uppercase;font-weight:750}}
.warning{{border:1px solid #855e32;border-left:5px solid var(--amber);background:#2a2118;padding:16px 19px;border-radius:10px;margin:22px 0;font-size:17px}}.danger{{border-color:#8e3945;border-left-color:var(--red);background:#2d171c}}
.instruction{{max-width:1200px;color:#d6e2ef;background:#0c1827;padding:14px 17px;border-radius:10px}}
section{{background:rgba(16,29,46,.96);border:1px solid var(--line);border-radius:16px;margin:24px 0;padding:18px}}section header{{display:flex;justify-content:space-between;gap:18px;align-items:center}}
.controls{{display:flex;gap:8px;flex-wrap:wrap}}button,select{{background:#17304a;color:var(--text);border:1px solid #3b6384;border-radius:8px;padding:8px 11px}}
.grid{{display:grid;grid-template-columns:repeat(7,minmax(210px,1fr));gap:9px;overflow-x:auto}}.card{{min-width:210px;background:#0a1522;border:1px solid var(--line);border-radius:11px;padding:9px}}.card h3{{font-size:14px;color:var(--cyan)}}.card p{{color:var(--muted);font-size:12px;min-height:38px}}
.card video{{width:100%;aspect-ratio:496/480;object-fit:contain;background:#020406;border-radius:7px}}.card.ref{{border-color:#526a7e}}.card.candidate{{border-color:#48a4b0}}.card.negative{{border-color:#79535d}}
details{{margin-top:13px;background:#0a1522;padding:11px;border-radius:9px}}summary{{cursor:pointer;color:var(--cyan)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#bed4e9;font-size:11px}}a{{color:var(--cyan)}}
@media(max-width:900px){{section header{{display:block}}}}
</style></head><body><main>
<p class="eyebrow">IID {escape(EXPECTED_IID)} · exact OOF fold 1 · proposal seed 2027</p>
<h1>v4-F → real Bernini EPMC temporal gating</h1>
<div class="warning danger"><strong>PRIVILEGED OOF TEMPORAL-GATING DIAGNOSTIC ONLY.</strong> 这不是 action representation、renderer、画质或 video-editing qualification，也不是 source+instruction-only 推理。它只覆盖训练中已暴露的 known transform families，绝不是 unseen-transform / unseen-action qualification。Anchor RGB 只作为 detached HTML reference，绝未进入 Bernini；但 gate 由该 OOF anchor 的 V-JEPA feature 生成，因此仍是 privileged side information。</div>
<div class="warning">只有 v4-F aggregate known-exposed development gate=true 且 exact5 所有 fold-local inner gate 均 PASS，才允许生成本页；任一 INNER_NO_GO 或 aggregate gate=false 都会在创建 HTML 前拒绝。v4-F 使用唯一 <code>[12,32]</code> global code（79,040 trainable parameters），并取 selected fold-local rho 下 <code>R=C(D(E(C(anchor))))-C(D(0))</code> 的 decoded residual，经 fold-1 model-fit-original-only p95 缩放、clamp 与 32→20 linear/align_corners 插值。16×12 block/head gates 与 phase0 均为正零，所以 EPMC effective gate 为 0.5×profile；再乘 outer CPMR 0.10 后，实际 projected-motion residual 系数仅为 <strong>0.05×profile</strong>。B0/Zero 只保证 final latent byte-exact，不声明 MP4 container byte parity。</div>
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
    _require_release_sealed()
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
    _require_release_sealed()
    if (
        args.expected_source_sha256 != EXPECTED_SOURCE_SHA256
        or args.expected_anchor_sha256 != EXPECTED_ANCHOR_SHA256
    ):
        raise V4FEPMCReviewError("source/detached-anchor authority pin differs")
    if (
        hashlib.sha256(args.instruction.encode("utf-8")).hexdigest()
        != EXPECTED_INSTRUCTION_SHA256
    ):
        raise V4FEPMCReviewError("instruction bytes differ")
    source = _plain_file(args.source_video_ref, label="source reference")
    anchor = _plain_file(args.anchor_video_ref, label="anchor reference")
    if (
        _file_sha256(source) != EXPECTED_SOURCE_SHA256
        or _file_sha256(anchor) != EXPECTED_ANCHOR_SHA256
    ):
        raise V4FEPMCReviewError("source/detached-anchor SHA256 differs")

    run_dirs = {
        2028: Path(args.seed2028_dir).expanduser(),
        2029: Path(args.seed2029_dir).expanduser(),
    }
    receipt_hashes = {
        2028: _required_sha256(
            args.expected_seed2028_receipt_sha256,
            label="seed2028 receipt SHA256",
        ),
        2029: _required_sha256(
            args.expected_seed2029_receipt_sha256,
            label="seed2029 receipt SHA256",
        ),
    }
    receipts: dict[int, dict[str, Any]] = {}
    receipt_paths: dict[int, Path] = {}
    video_sources: dict[tuple[int, str], tuple[Path, str]] = {}
    for seed, run_dir in run_dirs.items():
        if not run_dir.is_absolute() or run_dir.is_symlink() or not run_dir.is_dir():
            raise V4FEPMCReviewError(f"seed {seed} run directory differs")
        run_dir = run_dir.resolve(strict=True)
        run_dirs[seed] = run_dir
        receipt_path = _plain_file(
            run_dir / "receipt.json",
            label=f"seed {seed} receipt",
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
        if _file_sha256(receipt_path) != receipt_hashes[seed]:
            raise V4FEPMCReviewError(f"seed {seed} receipt file SHA256 differs")
        receipt_paths[seed] = receipt_path
        receipts[seed] = _strict_receipt(
            receipt_path, seed=seed, expected_sha256=receipt_hashes[seed]
        )
        for arm in ARMS:
            manifest = receipts[seed]["outputs"][arm]
            if not isinstance(manifest, Mapping):
                raise V4FEPMCReviewError(f"seed {seed} {arm} manifest differs")
            expected = _required_sha256(
                manifest.get("mp4_sha256"), label=f"seed {seed} {arm} SHA256"
            )
            source_video = _plain_file(
                manifest.get("path"), label=f"seed {seed} {arm} output"
            )
            if (
                source_video.parent != run_dir
                or _file_sha256(source_video) != expected
            ):
                raise V4FEPMCReviewError(f"seed {seed} {arm} output join differs")
            video_sources[(seed, arm)] = (source_video, expected)

    matched_fields = (
        "method_revision",
        "method_archive_sha256",
        "gate_state",
        "fold1_selected_rho",
        "checkpoint",
        "source_revisions",
        "runtime_versions",
        "freeze_certificate",
        "proposal_latents",
        "carrier",
    )
    for field in matched_fields:
        if receipts[2028].get(field) != receipts[2029].get(field):
            raise V4FEPMCReviewError(
                f"two render seeds do not share matched {field} authority"
            )

    output = Path(args.output_dir).expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise V4FEPMCReviewError("output-dir must be a fresh absolute directory")
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
        _copy_sealed(
            receipt_paths[seed],
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
            source_video, expected = video_sources[(seed, arm)]
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
    if stat.S_IMODE(index.stat().st_mode) != 0o444 or index.stat().st_nlink != 1:
        raise V4FEPMCReviewError("index seal differs")

    packet: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "iid": EXPECTED_IID,
        "render_seeds": list(SEEDS),
        "proposal_seed": PROPOSAL_SEED,
        "columns": [
            "Source",
            "Anchor",
            "B0",
            "Zero",
            "Correct",
            "Reverse",
            "Shuffle",
        ],
        "source_reference_sha256": EXPECTED_SOURCE_SHA256,
        "anchor_reference_sha256": EXPECTED_ANCHOR_SHA256,
        "anchor_rgb_used_by_bernini_runtime": False,
        "gate_state_privileged_action_anchor_feature_derived": True,
        "source_plus_instruction_only_end_to_end_claim": False,
        "sole_global_codec_shape": [12, 32],
        "exact_trainable_parameter_count": 79040,
        "fold1_selected_rho": receipts[2028]["fold1_selected_rho"],
        "v4f_aggregate_gate_verified_true": True,
        "v4f_exact5_all_fold_inner_gates_passed": True,
        "known_transform_families_exposed_during_model_fit": True,
        "unseen_hostile_transform_gate": False,
        "unseen_hostile_transform_gate_evaluated": False,
        "unseen_action_qualification": False,
        "aggregate_fold_receipt_preselection_selected_checkpoint_strong_join_verified_by_runtime": True,
        "decoded_residual_definition": "R=C(D(E(C(anchor))))-C(D(0))",
        "fit_only_scale": "fold1 model-fit-original-only p95",
        "profile_mapping": "32->20 torch linear align_corners=True",
        "effective_head_gate": "0.5*profile20",
        "total_projected_motion_residual_coefficient": "0.05*profile20",
        "block_head_gates_all_exact_positive_zero": True,
        "phase0_exact_positive_zero": True,
        "b0_zero_byte_parity_scope": "final_latent_only_not_mp4_container",
        "temporal_gating_diagnostic_only": True,
        "scientific_claim": False,
        "latent_metric_qualified": False,
        "action_representation_qualified": False,
        "identity_disentanglement_qualified": False,
        "identity_preservation_qualified": False,
        "prior_qualified": False,
        "prior_generation_qualified": False,
        "generation_qualified": False,
        "renderer_qualified": False,
        "video_editing_qualified": False,
        "video_quality_claim": False,
        "inference_authorized": False,
        "web_evaluation_authorized": False,
        "full644_refit_authorized": False,
        "vae_necessary": None,
        "members": member_rows,
        "index_html_sha256": _file_sha256(index),
    }
    packet["receipt_digest"] = _object_sha256(packet)
    packet_path = output / "packet.json"
    raw = (
        json.dumps(
            packet,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    with packet_path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(packet_path, 0o444)
    if (
        stat.S_IMODE(packet_path.stat().st_mode) != 0o444
        or packet_path.stat().st_nlink != 1
        or json.loads(packet_path.read_text(encoding="ascii")) != packet
    ):
        raise V4FEPMCReviewError("packet seal/readback differs")
    for directory in (
        media / "seed2028",
        media / "seed2029",
        media,
        receipt_dir,
        output,
    ):
        os.chmod(directory, 0o555)
    return {
        "index": str(index.resolve(strict=True)),
        "index_sha256": packet["index_html_sha256"],
        "packet": str(packet_path.resolve(strict=True)),
        "packet_sha256": _file_sha256(packet_path),
        "temporal_gating_diagnostic_only": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    _require_release_sealed()
    result = run(build_parser().parse_args(argv))
    print(_canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "EXPECTED_IID",
    "PIN_PLACEHOLDER",
    "RELEASE_SEALED",
    "SCHEMA",
    "SEEDS",
    "V4FEPMCReviewError",
    "build_parser",
    "main",
    "run",
]

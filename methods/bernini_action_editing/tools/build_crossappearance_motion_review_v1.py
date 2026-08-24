#!/usr/bin/env python3
"""Build the compact synchronized review for Complex8 motion transfer."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def card(
    *,
    root: Path,
    group: str,
    title: str,
    path: Path,
    note: str,
    kind: str,
) -> tuple[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = path.relative_to(root).as_posix()
    row = {
        "title": title,
        "path": relative,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "note": note,
        "kind": kind,
    }
    markup = f"""
<article class="card {html.escape(kind)}">
  <h3>{html.escape(title)}</h3>
  <video controls muted loop playsinline preload="metadata"
         data-group="{html.escape(group)}" src="{html.escape(relative)}"></video>
  <p>{html.escape(note)}</p>
</article>"""
    return markup, row


def section(
    *,
    root: Path,
    group: str,
    title: str,
    subtitle: str,
    specs: Sequence[tuple[str, Path, str, str]],
) -> tuple[str, dict[str, Any]]:
    cards = []
    rows = []
    for card_title, path, note, kind in specs:
        markup, row = card(
            root=root,
            group=group,
            title=card_title,
            path=path,
            note=note,
            kind=kind,
        )
        cards.append(markup)
        rows.append(row)
    markup = f"""
<section>
  <div class="section-head">
    <div><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div>
    <button type="button" onclick="playGroup('{html.escape(group)}')">同步播放本组</button>
    <button type="button" onclick="pauseGroup('{html.escape(group)}')">暂停本组</button>
  </div>
  <div class="grid">{''.join(cards)}</div>
</section>"""
    return markup, {"group": group, "title": title, "cards": rows}


def trained_specs(media: Path, case_id: str) -> list[tuple[str, Path, str, str]]:
    specs: list[tuple[str, Path, str, str]] = []
    audited_notes = {
        ("real", "matched", 1): "Turns, crouches and rises with a late object; source-object continuity remains uncertain.",
        ("real", "matched", 10): "Clear crouch/contact/rise/hold chain; strongest matched checkpoint, but held-object provenance is not strict.",
        ("real", "matched", 40): "A white round object appears while standing; required ground-contact onset is not preserved.",
        ("real", "matched", 64): "Subject disappears mid-video: categorical over-training failure.",
        ("real", "cross", 1): "Action-positive but oversaturated; a held object appears late without strict same-stone proof.",
        ("real", "cross", 10): "Crouches near the stones but remains stuck and does not complete rise/hold.",
        ("real", "cross", 40): "Strongest transfer: crouch, lift a large stone, rise and hold; stone instance/scale is still wrong.",
        ("real", "cross", 64): "Clothing drifts into a blue hoodie and body scale changes; strict identity failure.",
        ("synthetic", "matched", 10): "Fitted positive: contact/lift/hold occurs; this is a capacity check, not held-out transfer evidence.",
        ("synthetic", "matched", 64): "Long-step degradation: action becomes incomplete and the actor exits/disappears.",
        ("synthetic", "cross", 10): "Appearance becomes highly saturated and the lift sequence is unstable.",
        ("synthetic", "cross", 64): "Long-step degradation with incomplete action and severe appearance shift.",
    }
    for mode, label in (("matched", "Matched-flow upper bound"), ("cross", "Cross-appearance motion")):
        for step in (1, 10, 40, 64):
            path = media / mode / f"s{step:03d}" / case_id / "output.mp4"
            if path.is_file():
                specs.append(
                    (
                        f"{label} · step {step}",
                        path,
                        audited_notes.get(
                            (case_id, mode, step),
                            "Same source / instruction / inference seed; checkpoint step is the only changing coordinate.",
                        ),
                        "trained",
                    )
                )
    return specs


def hard_source_specs(
    media: Path, mode: str, baseline_step: int
) -> list[tuple[str, Path, str, str]]:
    labels = (
        ("cam_s05_t24_b18-22-26-29", "camera-flow · α .05 · start 24"),
        ("cam_s10_t28_b18-22-26-29", "camera-flow · α .10 · start 28"),
        ("cam_s15_t32_b18-22-26-29", "camera-flow · α .15 · start 32"),
        ("raw_s10_t28_b18-22-26-29", "raw-flow · α .10 · start 28"),
    )
    audited = {
        "cam_s05_t24_b18-22-26-29": "Preserves the baseline action; improves distant source background slightly, but object identity/scale is unchanged.",
        "cam_s10_t28_b18-22-26-29": "Stronger visible intervention and better distant background; still retains the same held-object error.",
        "cam_s15_t32_b18-22-26-29": "Best distant-background retention in this sweep; action survives, but same-stone continuity is not repaired.",
        "raw_s10_t28_b18-22-26-29": "Raw-flow control is nearly identical to camera-residual α .10; correspondence choice is not the bottleneck here.",
    }
    baseline = media / mode / f"s{baseline_step:03d}" / "real/output.mp4"
    donor = (
        media / "authority/t2v_v0_target_anchor.mp4"
        if mode == "matched"
        else media / "authority/t2v_v1_cross_motion_donor.mp4"
    )
    specs: list[tuple[str, Path, str, str]] = [
        (
            "Real source authority",
            media / "authority/real_source.mp4",
            "Authority for child, clothing, source stones, garden, camera and initial state.",
            "source",
        ),
        (
            f"{mode} T2V motion donor",
            donor,
            "Action demonstration only; its RGB appearance is never transported.",
            "anchor",
        ),
        (
            "Frozen RV2V",
            media / "frozen/real.mp4",
            "Same seed control: reaches toward foliage and does not lift the source stone.",
            "frozen",
        ),
        (
            f"No hard transport · {mode} step {baseline_step}",
            baseline,
            "Action-positive checkpoint and exact-seed control for this section.",
            "trained",
        )
    ]
    for directory, label in labels:
        path = media / "hard" / mode / directory / "output.mp4"
        if path.is_file():
            specs.append(
                (
                    label,
                    path,
                    audited[directory],
                    "trained",
                )
            )
    return specs


def source_attention_specs(
    media: Path, arm: str
) -> list[tuple[str, Path, str, str]]:
    arm_title = "all 30 blocks" if arm == "all30" else "late 15 blocks"
    variants = (
        ("s010_k100_t00", "step 10 · scale 1.0"),
        ("s040_k100_t00", "step 40 · scale 1.0"),
        ("s064_k100_t00", "step 64 · scale 1.0"),
        ("s040_k050_t00", "step 40 · scale 0.5"),
        ("s040_k025_t00", "step 40 · scale 0.25"),
    )
    audited = {
        ("all30", "s010_k100_t00"): "Completes contact/lift/hold, but darkens the garden and still creates an oversized stone.",
        ("all30", "s040_k100_t00"): "Background improves, while extra boulders appear before contact and subject/clothing drift remains.",
        ("all30", "s064_k100_t00"): "Over-trained failure: malformed late crouch and no complete lift/terminal hold.",
        ("all30", "s040_k050_t00"): "Best all-30 balance: action and scene survive, but the lifted boulder is not a preserved source instance.",
        ("all30", "s040_k025_t00"): "Closer to the no-source-attention action model; the same oversized-object error remains.",
        ("late15", "s010_k100_t00"): "Completes contact/lift/hold with less darkening than all-30, but still enlarges the stone.",
        ("late15", "s040_k100_t00"): "Strict collapse: the child disappears during the latter half of the video.",
        ("late15", "s064_k100_t00"): "Endpoint shortcut: a large stone appears in the hands without a ground-contact lift chain.",
        ("late15", "s040_k050_t00"): "Best visual balance overall; source identity and action survive, but same-stone continuity still fails.",
        ("late15", "s040_k025_t00"): "Stable action/identity, yet nearly reverts to the baseline and retains its oversized stone.",
    }
    specs: list[tuple[str, Path, str, str]] = [
        (
            "Real source authority",
            media / "authority/real_source.mp4",
            "Authority for the child, blue clothing, exact source stones, garden, camera and frame 0.",
            "source",
        ),
        (
            "Cross-appearance T2V donor",
            media / "authority/t2v_v1_cross_motion_donor.mp4",
            "Action demonstration; only its dense motion field enters the frozen action branch.",
            "anchor",
        ),
        (
            "Action model · cross step 40",
            media / "cross/s040/real/output.mp4",
            "No source-memory attention. It transfers the action but binds a wrong oversized stone.",
            "trained",
        ),
    ]
    for directory, title in variants:
        path = media / "sourceattn" / arm / directory / "output.mp4"
        if path.is_file():
            specs.append(
                (
                    f"Source attention {arm_title} · {title}",
                    path,
                    audited[(arm, directory)],
                    "trained",
                )
            )
    return specs


def source_correspondence_specs(
    media: Path, arm: str, weight_tag: str
) -> list[tuple[str, Path, str, str]]:
    arm_title = "all 30 blocks" if arm == "all30" else "late 15 blocks"
    weight = "0.02" if weight_tag == "w020" else "0.005"
    variants = (
        ("s010_k050_t00", "step 10 · source scale 0.5"),
        ("s040_k050_t00", "step 40 · source scale 0.5"),
        ("s064_k050_t00", "step 64 · source scale 0.5"),
        ("s040_k100_t00", "step 40 · source scale 1.0"),
    )
    audited = {
        ("all30", "w020", "s010_k050_t00"): "Completes crouch/lift/hold, but the held dark-grey rock is much larger than any source stone and lacks same-instance continuity.",
        ("all30", "w020", "s040_k050_t00"): "Completes the action while turning the object into an oversized faceted block; correspondence training worsens scale rather than source ownership.",
        ("all30", "w020", "s064_k050_t00"): "Long-step failure: a very large pale slab occupies the torso/shoulder region and no longer resembles a source instance.",
        ("all30", "w020", "s040_k100_t00"): "The held object is smaller than at scale .5, but remains a newly rendered grey chunk with no strict ground-to-hand source-instance trace.",
        ("all30", "w005", "s010_k050_t00"): "Action-positive, but the retrieved object becomes an oversized dark/green rock unrelated to the flat source stepping stones.",
        ("all30", "w005", "s040_k050_t00"): "A large white rectangular block appears during the lift; neither material nor scale preserves the contacted source stone.",
        ("all30", "w005", "s064_k050_t00"): "Severe long-step object drift: a pale slab is carried across the back/shoulder and dominates the frame.",
        ("all30", "w005", "s040_k100_t00"): "Best all-30 size control in this sweep, yet the terminal grey rock still cannot be traced to a vacated source stone.",
        ("late15", "w020", "s010_k050_t00"): "Clear crouch/contact/rise/hold, but the source object is replaced by a large grey rectangular rock.",
        ("late15", "w020", "s040_k050_t00"): "The held object expands into a block that covers much of the upper body; explicit cell labels do not preserve instance scale.",
        ("late15", "w020", "s064_k050_t00"): "Long-step degradation weakens the required lift/hold chain and does not recover a source-owned object.",
        ("late15", "w020", "s040_k100_t00"): "Stronger source attention changes slab thickness but still produces an oversized replacement object.",
        ("late15", "w005", "s010_k050_t00"): "Action-positive and identity-stable, but the terminal object is still an oversized grey block.",
        ("late15", "w005", "s040_k050_t00"): "A huge cube appears after contact and obscures the child; longer FM training amplifies the wrong object attractor.",
        ("late15", "w005", "s064_k050_t00"): "The action becomes unstable/incomplete at the long checkpoint; no same-stone terminal hold is established.",
        ("late15", "w005", "s040_k100_t00"): "A broad pale slab is held, but it is not the contacted flat source stone and no reliable vacancy chain is visible.",
    }
    specs: list[tuple[str, Path, str, str]] = [
        (
            "Real source authority",
            media / "authority/real_source.mp4",
            "Authority for the child, clothing, exact source stones, garden, camera and frame 0.",
            "source",
        ),
        (
            "Cross-appearance T2V donor",
            media / "authority/t2v_v1_cross_motion_donor.mp4",
            "Its motion enters the frozen action branch; its RGB appearance remains non-authoritative.",
            "anchor",
        ),
        (
            "Action model · cross step 40",
            media / "cross/s040/real/output.mp4",
            "Exact-seed action-positive control without dynamic source-token correspondence.",
            "trained",
        ),
    ]
    for directory, title in variants:
        path = media / "sourcecorr" / arm / weight_tag / directory / "output.mp4"
        if path.is_file():
            specs.append(
                (
                    f"Correspondence {arm_title} · w {weight} · {title}",
                    path,
                    audited[(arm, weight_tag, directory)],
                    "trained",
                )
            )
    return specs


def source_correspondence_refine_specs(
    media: Path,
) -> list[tuple[str, Path, str, str]]:
    """Decisive Round-80 convergence stop test, with no machine-positive label."""

    specs: list[tuple[str, Path, str, str]] = [
        (
            "Real source authority",
            media / "authority/real_source.mp4",
            "The contacted flat source stone must remain the same instance, leave a vacancy, and retain its scale/material while being lifted.",
            "source",
        ),
        (
            "Cross-appearance T2V donor",
            media / "authority/t2v_v1_cross_motion_donor.mp4",
            "Action demonstration only. Its actor, scene, RGB and object appearance are not output authority.",
            "anchor",
        ),
        (
            "Action model · cross step 40",
            media / "cross/s040/real/output.mp4",
            "No source-cell correspondence: action transfers, but the held object is an oversized replacement rather than a preserved source instance.",
            "trained",
        ),
        (
            "Round 79 warm start · late15 · step 10",
            media / "sourcecorr/late15/w020/s010_k050_t00/output.mp4",
            "Before Q/K-only refinement. Clear action chain, but the lifted object already has the wrong size and provenance.",
            "trained",
        ),
        (
            "Round 79 long train · late15 · step 64",
            media / "sourcecorr/late15/w020/s064_k050_t00/output.mp4",
            "Longer joint FM+CE training weakens the event and still does not establish source-object ownership.",
            "trained",
        ),
    ]
    refined = (
        media
        / "sourcecorr_refine_balanced/late15/lr500/s064_k050_t00/output.mp4"
    )
    if refined.is_file():
        specs.append(
            (
                "Round 80 Q/K refine · lr 5e-4 · step 64",
                refined,
                "CE is decisively below random, yet a new huge pale block replaces the contacted stone. Optimization GO; same-instance ownership and final edit NO-GO.",
                "failed",
            )
        )
    return specs


def build(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    media = root / "media"
    sections: list[str] = []
    manifest_sections: list[dict[str, Any]] = []

    specs = (
        (
            "Real source authority",
            media / "authority/real_source.mp4",
            "Only authority for child, blue clothing, stones, garden, camera and frame 0.",
            "source",
        ),
        (
            "Pure-T2V target action v0",
            media / "authority/t2v_v0_target_anchor.mp4",
            "Clean action target for the matched synthetic row; its appearance is not real-source authority.",
            "anchor",
        ),
        (
            "Pure-T2V cross donor v1",
            media / "authority/t2v_v1_cross_motion_donor.mp4",
            "Cross arm reads only this donor's RAFT motion features, never RGB or VAE latent.",
            "anchor",
        ),
        (
            "Frozen RV2V · real source",
            media / "frozen/real.mp4",
            "Same real source, instruction and seed; required baseline, not a target or ground truth.",
            "frozen",
        ),
    )
    markup, rows = section(
        root=root,
        group="authority",
        title="Authority and causal inputs",
        subtitle="First compare what appearance must be preserved with the two action demonstrations and the unchanged editor.",
        specs=specs,
    )
    sections.append(markup)
    manifest_sections.append(rows)

    synthetic_specs: list[tuple[str, Path, str, str]] = [
        (
            "Synthetic noop source v0",
            media / "authority/synthetic_noop_source.mp4",
            "Same actor/world as the clean v0 target, but the requested event is absent.",
            "source",
        ),
        (
            "Frozen RV2V · synthetic",
            media / "frozen/synthetic.mp4",
            "Capacity control on an in-bank appearance; already partly action-positive and must not be over-interpreted.",
            "frozen",
        ),
    ]
    synthetic_specs.extend(trained_specs(media, "synthetic"))
    markup, rows = section(
        root=root,
        group="synthetic",
        title="Synthetic fitted probe",
        subtitle="Checks whether the joint FM route learns at all. Success here alone is not evidence of real editing transfer.",
        specs=synthetic_specs,
    )
    sections.append(markup)
    manifest_sections.append(rows)

    real_specs: list[tuple[str, Path, str, str]] = [
        (
            "Real source authority",
            media / "authority/real_source.mp4",
            "The child must begin in this walking state; no endpoint may be pre-applied at frame 0.",
            "source",
        ),
        (
            "Frozen RV2V · real source",
            media / "frozen/real.mp4",
            "Fails same-stone contact/lift/hold; trained outputs must visibly differ in event semantics.",
            "frozen",
        ),
    ]
    real_specs.extend(trained_specs(media, "real"))
    markup, rows = section(
        root=root,
        group="real",
        title="Real held-out Event 01",
        subtitle="Primary decision row: same source stone must contact, leave a vacant spot, rise with the hand and remain held while source appearance stays fixed.",
        specs=real_specs,
    )
    sections.append(markup)
    manifest_sections.append(rows)

    for mode, baseline_step, label in (
        ("matched", 10, "Matched step-10 hard source transport"),
        ("cross", 40, "Cross-appearance step-40 hard source transport"),
    ):
        hard_specs = hard_source_specs(media, mode, baseline_step)
        if len(hard_specs) == 4:
            continue
        markup, rows = section(
            root=root,
            group=f"hard-{mode}",
            title=label,
            subtitle=(
                "Same source, instruction, checkpoint and seed. Compare whether "
                "late source hidden transport preserves identity without erasing lift/hold."
            ),
            specs=hard_specs,
        )
        sections.append(markup)
        manifest_sections.append(rows)

    for arm, label in (
        ("all30", "Learned source-memory attention · all 30 blocks"),
        ("late15", "Learned source-memory attention · late 15 blocks"),
    ):
        source_specs = source_attention_specs(media, arm)
        if len(source_specs) == 4:
            continue
        markup, rows = section(
            root=root,
            group=f"sourceattn-{arm}",
            title=label,
            subtitle=(
                "The action model is frozen at cross step 40. A separately learned "
                "target-query → source-phase-0 K/V route must improve same-instance "
                "binding without reverting to the no-action source."
            ),
            specs=source_specs,
        )
        sections.append(markup)
        manifest_sections.append(rows)

    for arm, arm_label in (
        ("all30", "all 30 blocks"),
        ("late15", "late 15 blocks"),
    ):
        for weight_tag, weight in (("w020", "0.02"), ("w005", "0.005")):
            correspondence_specs = source_correspondence_specs(
                media, arm, weight_tag
            )
            if len(correspondence_specs) == 3:
                continue
            markup, rows = section(
                root=root,
                group=f"sourcecorr-{arm}-{weight_tag}",
                title=(
                    "Dynamic source-token correspondence · "
                    f"{arm_label} · CE weight {weight}"
                ),
                subtitle=(
                    "Same source, instruction, action checkpoint and seed. Training-only "
                    "matched RAFT labels supervise which source phase-0 memory cell each "
                    "dynamic target token retrieves; no target or teacher is used at inference."
                ),
                specs=correspondence_specs,
            )
            sections.append(markup)
            manifest_sections.append(rows)

    refine_specs = source_correspondence_refine_specs(media)
    if len(refine_specs) > 5:
        markup, rows = section(
            root=root,
            group="sourcecorr-refine-stop",
            title="Round 80 convergence stop test · CE-low does not imply object ownership",
            subtitle=(
                "Fixed source, T2V donor, instruction and seed. The last card is not a "
                "machine-selected positive: it is the decisive failure showing that "
                "optimizing RAFT-cell retrieval still creates a replacement object."
            ),
            specs=refine_specs,
        )
        sections.append(markup)
        manifest_sections.append(rows)

    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross-appearance action-motion training review</title>
<style>
:root{{--bg:#f4f1e9;--panel:#fffdf8;--ink:#18221e;--muted:#65716c;--line:#cfc5b2;--green:#176b57;--orange:#a85e19}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,-apple-system,sans-serif}}
header{{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:10px;padding:12px 18px;background:rgba(244,241,233,.96);border-bottom:1px solid var(--line)}}
h1{{font-size:21px;margin:0 auto 0 0}} button{{border:1px solid #9f927b;border-radius:9px;background:#fffaf0;padding:8px 12px;font-weight:650;cursor:pointer}}
main{{padding:14px;max-width:1800px;margin:auto}} section{{margin:0 0 18px;padding:12px;background:var(--panel);border:1px solid var(--line);border-radius:14px}}
.section-head{{display:flex;align-items:center;gap:8px;margin-bottom:10px}} .section-head>div{{margin-right:auto}} h2{{font-size:19px;margin:0}} .section-head p{{margin:2px 0 0;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;align-items:stretch}} .card{{display:grid;grid-template-rows:auto auto 1fr;border:1px solid var(--line);border-radius:11px;overflow:hidden;background:#fff}}
.card h3{{display:flex;align-items:center;min-height:50px;margin:0;padding:8px 10px;font-size:15px}} video{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#080a09}}
.card p{{margin:0;padding:8px 10px;color:var(--muted);font-size:13px}} .source{{border-color:#368b78}} .anchor{{border-color:#c78037}} .trained{{border-color:#57907e}} .failed{{border:2px solid #a33b32}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}} @media(max-width:620px){{.grid{{grid-template-columns:1fr}} .section-head{{flex-wrap:wrap}}}}
</style></head><body>
<header><h1>Complex8 generation→editing · matched vs cross appearance</h1><button onclick="playAll()">全部从 0 同步播放</button><button onclick="pauseAll()">全部暂停</button></header>
<main>{''.join(sections)}</main>
<script>
const videos=(group)=>[...document.querySelectorAll(group?`video[data-group="${{group}}"]`:'video')];
async function ready(v){{if(v.readyState>=1)return;await new Promise((ok,bad)=>{{v.addEventListener('loadedmetadata',ok,{{once:true}});v.addEventListener('error',bad,{{once:true}});v.load();}})}}
async function start(list){{await Promise.all(list.map(ready));list.forEach(v=>{{v.pause();v.currentTime=0;v.muted=true}});await Promise.allSettled(list.map(v=>v.play()));}}
function playGroup(group){{start(videos(group))}} function pauseGroup(group){{videos(group).forEach(v=>v.pause())}}
function playAll(){{start(videos())}} function pauseAll(){{videos().forEach(v=>v.pause())}}
</script></body></html>"""
    (root / "index.html").write_text(body, encoding="utf-8")
    manifest = {
        "schema_version": "crossappearance-motion-review-v1",
        "layout_columns": 4,
        "human_ground_truth_labels": False,
        "machine_score_implies_correct": False,
        "sections": manifest_sections,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    build(args.root.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

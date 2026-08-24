#!/usr/bin/env python3
"""Materialize a compact, synchronized review for complex8 reward training.

The review deliberately separates source authority, pure-T2V motion anchors,
frozen RV2V proposals, preference endpoints, and trained checkpoint decodes.
Pure-T2V appearance is never labelled as a target or ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = (
    ("dpo_only_s4", "DPO only", 4),
    ("dpo_identity005_s4", "DPO + identity replay 0.05", 4),
    ("dpo_identity015_s4", "DPO + identity replay 0.15", 4),
    ("dpo_identity010_s8", "DPO + identity replay 0.10", 8),
)
EVENTS = tuple(range(8))


class ReviewError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ReviewError(f"missing plain JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReviewError(f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_video(source: Path, media: Path, name: str) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise ReviewError(f"missing plain video: {source}")
    destination = media / name
    if destination.exists():
        raise ReviewError(f"duplicate review media destination: {destination}")
    shutil.copy2(source, destination)
    return {
        "path": f"media/{destination.name}",
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
    }


def candidate_receipt(stage: Path, candidate_id: str) -> dict[str, Any]:
    return read_json(
        stage
        / "interaction_complex8_rv2v_candidates_v1"
        / candidate_id
        / "pair-v5-rollout-receipt.json"
    )


def event_anchor_directory(stage: Path, event: int) -> Path:
    matches = sorted(
        (stage / "interaction_complex8_multianchor_v2_r1").glob(f"e{event:02d}_*")
    )
    if len(matches) != 1 or not matches[0].is_dir():
        raise ReviewError(f"event {event} anchor directory is ambiguous")
    return matches[0]


def score_lookup(stage: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    reward_root = stage / "interaction_complex8_reward_v1"
    for path in sorted(reward_root.glob("group_*/reward-group.json")):
        receipt = read_json(path)
        for row in receipt.get("candidate_action_scores", ()):  # type: ignore[union-attr]
            if not isinstance(row, Mapping) or not isinstance(row.get("score"), Mapping):
                raise ReviewError(f"malformed candidate action row: {path}")
            result[str(row["candidate_id"])] = float(
                row["score"]["phase_conjunctive_reward"]
            )
    return result


def trained_score_lookup(stage: Path) -> dict[tuple[int, str, int], dict[str, float]]:
    result: dict[tuple[int, str, int], dict[str, float]] = {}
    root = stage / "interaction_complex8_large_lora_action_score_v1"
    for event in EVENTS:
        receipt = read_json(root / f"event_{event:02d}" / "trained-action-score.json")
        if receipt.get("complete") is not True or int(receipt["event_ordinal"]) != event:
            raise ReviewError(f"trained action score receipt differs: event {event}")
        for row in receipt.get("rows", ()):  # type: ignore[union-attr]
            if not isinstance(row, Mapping):
                raise ReviewError("malformed trained action score row")
            key = (event, str(row["arm"]), int(row["step"]))
            score = row.get("score")
            if key in result or not isinstance(score, Mapping):
                raise ReviewError("trained action score key/record differs")
            result[key] = {
                "reward": float(score["phase_conjunctive_reward"]),
                "delta": float(row["phase_reward_delta_from_frozen"]),
            }
    if len(result) != len(EVENTS) * 9:
        raise ReviewError("trained action score closure differs")
    return result


def preference_lookup(stage: Path) -> dict[int, dict[str, Any]]:
    manifest = read_json(stage / "interaction_complex8_preference_v1.json")
    result: dict[int, dict[str, Any]] = {}
    for row in manifest.get("pairs", ()):  # type: ignore[union-attr]
        if not isinstance(row, Mapping):
            raise ReviewError("malformed preference row")
        result[int(row["event_ordinal"])] = dict(row)
    return result


def binding_candidate_id(binding: Mapping[str, Any]) -> str:
    candidate_id = binding.get("candidate_id")
    if not isinstance(candidate_id, str):
        raise ReviewError("preference endpoint candidate ID is absent")
    return candidate_id


def materialize(stage: Path, output: Path) -> dict[str, Any]:
    if not stage.is_dir():
        raise ReviewError(f"stage does not exist: {stage}")
    if output.exists() or output == Path("/"):
        raise ReviewError("output must be a fresh non-root directory")
    output.mkdir(parents=True)
    media = output / "media"
    media.mkdir()
    scores = score_lookup(stage)
    trained_scores = trained_score_lookup(stage)
    preferences = preference_lookup(stage)
    events: list[dict[str, Any]] = []
    for event in EVENTS:
        baseline_id = f"complex8-e{event:02d}-rv2v-s0"
        baseline_receipt = candidate_receipt(stage, baseline_id)
        candidate = baseline_receipt["candidate"]
        if not isinstance(candidate, Mapping):
            raise ReviewError(f"candidate metadata absent: {baseline_id}")
        source_path = Path(str(candidate["source_video"]))
        anchor_path = event_anchor_directory(stage, event) / "v0" / "t2v.mp4"
        frozen_path = Path(str(baseline_receipt["artifacts"]["mp4"]["path"]))
        preference = preferences.get(event)
        if preference is None:
            raise ReviewError(f"review event {event} has no admitted preference")
        chosen_id = binding_candidate_id(preference["chosen_rollout"])
        rejected_id = binding_candidate_id(preference["rejected_rollout"])
        chosen_receipt = candidate_receipt(stage, chosen_id)
        rejected_receipt = candidate_receipt(stage, rejected_id)
        cards: list[dict[str, Any]] = []

        def add(label: str, role: str, path: Path, filename: str, note: str) -> None:
            cards.append(
                {
                    "label": label,
                    "role": role,
                    "note": note,
                    "media": copy_video(path, media, filename),
                }
            )

        add(
            "Source authority",
            "authority",
            source_path,
            f"e{event:02d}-source.mp4",
            "Identity, appearance, scene and frame-0 authority",
        )
        add(
            "Pure-T2V anchor v0",
            "anchor",
            anchor_path,
            f"e{event:02d}-t2v-anchor-v0.mp4",
            "Motion demonstration only; its appearance is irrelevant",
        )
        add(
            f"Frozen RV2V s0 · reward {trained_scores[(event, 'frozen', 0)]['reward']:.4f}",
            "baseline",
            frozen_path,
            f"e{event:02d}-frozen-s0.mp4",
            "Same source, instruction and seed used for checkpoint decoding",
        )
        add(
            f"Preference chosen · {chosen_id} · {scores[chosen_id]:.4f}",
            "preference",
            Path(str(chosen_receipt["artifacts"]["mp4"]["path"])),
            f"e{event:02d}-preference-chosen.mp4",
            "Machine-selected training endpoint; not ground truth",
        )
        add(
            f"Preference rejected · {rejected_id} · {scores[rejected_id]:.4f}",
            "preference",
            Path(str(rejected_receipt["artifacts"]["mp4"]["path"])),
            f"e{event:02d}-preference-rejected.mp4",
            "Lower action score after passing the same source hard gates",
        )
        for arm, arm_label, final_step in ARMS:
            for step, step_label in ((1, "step 1"), (final_step, f"final step {final_step}")):
                trained_score = trained_scores[(event, arm, step)]
                path = (
                    stage
                    / "interaction_complex8_large_lora_decode_v1"
                    / arm
                    / f"event_{event:02d}"
                    / f"step_{step:04d}"
                    / "rv2v.mp4"
                )
                add(
                    f"{arm_label} · {step_label} · reward {trained_score['reward']:.4f} · Δ {trained_score['delta']:+.4f}",
                    "trained",
                    path,
                    f"e{event:02d}-{arm}-step{step:04d}.mp4",
                    "Same source/instruction/seed; Δ is frozen-critic action reward, not human GT",
                )
        events.append(
            {
                "event": event,
                "caption": str(candidate["complete_caption"]),
                "strict_action_margin": float(preference["selection"]["strict_action_margin"]),
                "cards": cards,
            }
        )
    receipt = {
        "schema_version": "bernini-interaction-complex8-large-lora-review-v1",
        "stage": str(stage),
        "events": events,
        "interpretation": {
            "legacy_negative_control": True,
            "pure_t2v_anchor_enters_optimizer_or_student_forward": False,
            "pure_t2v_appearance_is_target": False,
            "preference_chosen_is_ground_truth": False,
            "source_is_identity_and_initial_state_authority": True,
            "all_checkpoint_decodes_share_frozen_baseline_source_instruction_seed": True,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def compact_label(label: str) -> str:
    """Keep review labels legible when all 13 comparison videos share one row."""
    if label == "Source authority":
        return "Source"
    if label == "Pure-T2V anchor v0":
        return "T2V anchor · motion only"
    preference = re.fullmatch(
        r"Preference (chosen|rejected) · complex8-e\d+-rv2v-s(\d+) · (-?\d+\.\d+)",
        label,
    )
    if preference is not None:
        role, step, reward = preference.groups()
        return f"{role.title()} s{step} · R {reward}"
    replacements = (
        ("Frozen RV2V s0 · reward ", "Frozen s0 · R "),
        ("DPO + identity replay 0.05", "Id replay .05"),
        ("DPO + identity replay 0.10", "Id replay .10"),
        ("DPO + identity replay 0.15", "Id replay .15"),
        ("DPO only", "DPO"),
        (" · final step ", " · final s"),
        (" · step ", " · s"),
        (" · reward ", " · R "),
    )
    for source, target in replacements:
        label = label.replace(source, target)
    return label


def card_html(event: int, card: Mapping[str, Any]) -> str:
    full_label = str(card["label"])
    label = html.escape(compact_label(full_label))
    note = html.escape(str(card["note"]))
    role = html.escape(str(card["role"]))
    path = html.escape(str(card["media"]["path"]), quote=True)
    tooltip = html.escape(f"{full_label} — {card['note']}", quote=True)
    return f"""
      <article class="card {role}" title="{tooltip}">
        <div class="label">{label}</div>
        <video class="event-{event}" controls muted playsinline preload="metadata" src="{path}"></video>
        <div class="note">{note}</div>
      </article>"""


def render(receipt: Mapping[str, Any]) -> str:
    sections = []
    for event in receipt["events"]:
        cards = "\n".join(card_html(int(event["event"]), card) for card in event["cards"])
        caption_text = str(event["caption"])
        caption = html.escape(caption_text)
        caption_attr = html.escape(caption_text, quote=True)
        sections.append(
            f"""
  <section class="event" id="event-{event['event']}">
    <header>
      <div><h2>Event {event['event']:02d}</h2><p title="{caption_attr}">{caption}</p></div>
      <button onclick="syncPlay('.event-{event['event']}', this)">同步播放本事件</button>
    </header>
    <div class="grid">{cards}</div>
  </section>"""
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LEGACY negative control · Complex8 RV2V-only DPO review</title>
<style>
:root{{--bg:#f5f1e9;--panel:#fffdf8;--ink:#17211e;--muted:#64706c;--line:#d7ccba;--green:#176b57;--brown:#8a5525;--review-cols:7}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,sans-serif}}
.top{{position:sticky;top:0;z-index:4;display:flex;gap:8px;align-items:center;padding:8px 10px;background:#f5f1e9ee;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
h1{{font-size:17px;margin:0}} button{{border:1px solid #9f927e;background:#fffaf1;border-radius:9px;padding:7px 10px;font-weight:700;cursor:pointer;white-space:nowrap}} button:disabled{{opacity:.55}}
button.density.active{{border-color:var(--green);background:var(--green);color:#fff}} .legend{{color:var(--muted);margin-left:auto;font-size:12px}} main{{padding:6px}} .event{{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin-bottom:7px;padding:7px}}
.event header{{display:flex;gap:8px;align-items:start;margin-bottom:6px}} .event header>div{{flex:1;min-width:0}} h2{{font-size:16px;margin:0 0 2px}} p{{margin:0;color:var(--muted);font-size:11px;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;overflow:hidden}}
.grid{{display:grid;grid-template-columns:repeat(var(--review-cols),minmax(0,1fr));gap:4px;align-items:start}} .card{{display:grid;grid-template-rows:55px auto;min-width:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff;align-self:start}}
.label{{height:55px;padding:5px 6px;font-weight:750;font-size:11px;line-height:1.25;word-break:break-word;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:4;overflow:hidden}} video{{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#111}}
.note{{display:none}} .authority{{border-color:#4e8f7e}} .anchor{{border-color:#b6844e}} .trained{{border-color:#759385}}
@media(max-width:720px){{.top{{overflow-x:auto}}.top h1,.legend,.density{{display:none}}.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.event header{{display:block}}.event header button{{margin-top:5px}}}}
</style></head><body>
<div class="top"><h1>LEGACY NO-GO · RV2V-only DPO</h1><button onclick="syncPlay('video', this)">全部同步播放</button><button onclick="pauseAll()">全部暂停</button><button class="density active" data-columns="7" onclick="setColumns(7,this)">7 列对齐</button><button class="density" data-columns="5" onclick="setColumns(5,this)">5 列放大</button><span class="legend">T2V anchor 只是展示/reward参考，不进入 optimizer 或 student forward · chosen≠人工真值</span></div>
<main>{''.join(sections)}</main>
<script>
function waitReady(v){{if(v.readyState>=1)return Promise.resolve();return new Promise((ok,bad)=>{{v.addEventListener('loadedmetadata',ok,{{once:true}});v.addEventListener('error',bad,{{once:true}});v.load();}})}}
async function syncPlay(selector,button){{const videos=[...document.querySelectorAll(selector)];button.disabled=true;button.textContent='加载并对齐…';try{{videos.forEach(v=>{{v.pause();v.muted=true;v.currentTime=0;}});await Promise.all(videos.map(waitReady));videos.forEach(v=>v.currentTime=0);const results=await Promise.allSettled(videos.map(v=>v.play()));if(results.some(r=>r.status==='rejected'))throw new Error('browser rejected playback');}}catch(error){{alert('同步播放失败：'+error.message+'。请用 http://127.0.0.1 服务打开，并检查 media 文件。');}}finally{{button.disabled=false;button.textContent=selector==='video'?'全部同步播放':'同步播放本事件';}}}}
function pauseAll(){{document.querySelectorAll('video').forEach(v=>v.pause())}}
function setColumns(count,button){{document.documentElement.style.setProperty('--review-cols',String(count));document.querySelectorAll('.density').forEach(item=>item.classList.toggle('active',item===button));}}
</script></body></html>"""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage")
    result.add_argument("--output", required=True)
    result.add_argument(
        "--rerender-existing",
        action="store_true",
        help="Rebuild only index.html from an existing output manifest.",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute():
        raise ReviewError("output must be absolute")
    if args.rerender_existing:
        output = output.resolve(strict=True)
        receipt = read_json(output / "manifest.json")
        (output / "index.html").write_text(render(receipt), encoding="utf-8")
        return 0
    if args.stage is None:
        raise ReviewError("--stage is required unless --rerender-existing is set")
    stage = Path(args.stage).resolve(strict=True)
    receipt = materialize(stage, output)
    (output / "index.html").write_text(render(receipt), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

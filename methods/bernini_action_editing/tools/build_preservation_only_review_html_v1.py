#!/usr/bin/env python3
"""Build a fail-closed HTML review for preservation-only action editing.

The page deliberately contains no aggregate success score.  It exposes the
source, the exact RV2V target caption, the native result, and the two trained
preservation variants side by side.  Training diagnostics are named and
defined as optimization diagnostics rather than rewards or semantic metrics.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "bernini-preservation-only-review-v1"
VARIANTS = ("rank8", "rank2")
CELL_FILES = (
    "source.mp4",
    "rank8-native-rv2v.mp4",
    "rank8-preservation-residual.mp4",
    "rank2-native-rv2v.mp4",
    "rank2-preservation-residual.mp4",
    "rank8-receipt.json",
    "rank2-receipt.json",
    "review_5x5.jpg",
)


class PreservationReviewHTMLError(RuntimeError):
    """Raised when a review packet is incomplete or overclaims its evidence."""


def _plain_file(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise PreservationReviewHTMLError(f"{label} must be an absolute plain file")
    return value.resolve(strict=True)


def _plain_dir(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or not value.is_dir() or value.is_symlink():
        raise PreservationReviewHTMLError(f"{label} must be an absolute plain directory")
    return value.resolve(strict=True)


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreservationReviewHTMLError(f"cannot read {label}") from error
    if type(value) is not dict:
        raise PreservationReviewHTMLError(f"{label} must contain one JSON object")
    return value


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise PreservationReviewHTMLError(f"{label} must be non-empty text")
    return value


def _number(value: Any, *, label: str) -> float:
    if type(value) not in (int, float):
        raise PreservationReviewHTMLError(f"{label} must be numeric")
    return float(value)


def _validate_receipt(path: Path, *, cell_id: str, expected_rank: int) -> None:
    receipt = _load_object(path, label=f"{cell_id} rank-{expected_rank} receipt")
    if receipt.get("cell_id") != cell_id:
        raise PreservationReviewHTMLError("inference receipt cell identity differs")
    bundle = receipt.get("training_bundle")
    if not isinstance(bundle, Mapping) or bundle.get("adapter_rank") != expected_rank:
        raise PreservationReviewHTMLError("inference receipt adapter rank differs")
    forbidden_claims = (
        "action_reward_consumed",
        "feature_reward_consumed",
        "vlm_reward_consumed",
        "synthetic_target_consumed",
        "scientific_or_action_editing_claim_authorized",
    )
    if any(receipt.get(field) is not False for field in forbidden_claims):
        raise PreservationReviewHTMLError("inference receipt authority is not fail-closed")
    sampling = receipt.get("sampling")
    patches = receipt.get("preservation_residual")
    outputs = receipt.get("outputs")
    freeze = receipt.get("freeze_certificate")
    if (
        not isinstance(sampling, Mapping)
        or sampling.get("same_official_gaussian_all_arms") is not True
        or sampling.get("num_inference_steps") != 40
        or sampling.get("frame_count") != 81
        or not isinstance(patches, Mapping)
        or not isinstance(outputs, Mapping)
        or not isinstance(freeze, Mapping)
        or freeze.get("all_ranks_sampling_model_unchanged") is not True
        or receipt.get("rank_zero_only_vae") is not True
    ):
        raise PreservationReviewHTMLError("inference execution contract differs")
    patch = patches.get("preservation-residual")
    if (
        not isinstance(patch, Mapping)
        or patch.get("composition")
        != "v_native_action+(v_adapted_noop-v_frozen_noop)"
        or patch.get("adapter_action_text_input") is not False
        or patch.get("unit_gain") is not True
        or patch.get("noop_forwards") != 80
        or patch.get("scheduler_steps") != 40
        or not isinstance(patch.get("trace"), list)
        or len(patch["trace"]) != 40
    ):
        raise PreservationReviewHTMLError("preservation residual trace differs")
    for arm in ("native-rv2v", "preservation-residual"):
        output = outputs.get(arm)
        if (
            not isinstance(output, Mapping)
            or output.get("frame_count") != 81
            or output.get("fps") != 25
        ):
            raise PreservationReviewHTMLError("decoded output metadata differs")


def _validate(
    manifest: Mapping[str, Any], media_root: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if manifest.get("schema_version") != SCHEMA:
        raise PreservationReviewHTMLError("review schema differs")
    authority = manifest.get("authority")
    if not isinstance(authority, Mapping) or any(
        authority.get(field) is not False
        for field in (
            "automatic_semantic_score_present",
            "reward_used",
            "synthetic_target_used",
            "manual_review_completed",
            "method_success_claimed",
        )
    ):
        raise PreservationReviewHTMLError("review authority must remain fail-closed")

    raw_experiments = manifest.get("experiments")
    if not isinstance(raw_experiments, Mapping) or set(raw_experiments) != set(VARIANTS):
        raise PreservationReviewHTMLError("experiment closure differs")
    experiments: dict[str, dict[str, Any]] = {}
    training_iids: set[str] = set()
    expected_ranks = {"rank8": 8, "rank2": 2}
    for variant in VARIANTS:
        raw = raw_experiments[variant]
        if type(raw) is not dict or raw.get("adapter_rank") != expected_ranks[variant]:
            raise PreservationReviewHTMLError(f"{variant} metadata differs")
        if raw.get("optimizer_steps") != 40 or raw.get("training_target") != "real_source_exact_noop":
            raise PreservationReviewHTMLError(f"{variant} training contract differs")
        raw_training_iids = raw.get("training_source_iids")
        if (
            raw.get("dataset_rows") != 2
            or not isinstance(raw_training_iids, list)
            or len(raw_training_iids) != 2
        ):
            raise PreservationReviewHTMLError(f"{variant} training scale differs")
        variant_training_iids = {
            _text(value, label=f"{variant} training source IID")
            for value in raw_training_iids
        }
        if len(variant_training_iids) != 2:
            raise PreservationReviewHTMLError(f"{variant} training source closure differs")
        if training_iids and variant_training_iids != training_iids:
            raise PreservationReviewHTMLError("variant training source IIDs differ")
        training_iids = variant_training_iids
        diagnostics = raw.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise PreservationReviewHTMLError(f"{variant} diagnostics missing")
        if diagnostics.get("semantic_success_assessed") is not False:
            raise PreservationReviewHTMLError("diagnostics cannot claim semantic success")
        for field in (
            "wrong_source_gap_positive",
            "wrong_source_gap_total",
            "wrong_source_gap_mean",
            "loss_min",
            "loss_max",
            "grad_norm_min",
            "grad_norm_max",
        ):
            _number(diagnostics.get(field), label=f"{variant} {field}")
        experiments[variant] = dict(raw)

    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise PreservationReviewHTMLError("review cells must be non-empty")
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_cells:
        if type(raw) is not dict:
            raise PreservationReviewHTMLError("review cell must be an object")
        cell_id = _text(raw.get("cell_id"), label="cell ID")
        if cell_id in seen or "/" in cell_id or cell_id in {".", ".."}:
            raise PreservationReviewHTMLError("review cell identity differs")
        seen.add(cell_id)
        for field in (
            "source_iid",
            "requested_action",
            "source_action_caption",
            "target_action_caption",
        ):
            _text(raw.get(field), label=f"{cell_id} {field}")
        exact_native = raw.get("cross_variant_native_mp4_byte_exact")
        exact_condition = raw.get("cross_variant_source_condition_raw_byte_exact")
        direct_authority = raw.get("cross_variant_direct_comparison_authorized")
        if (
            type(exact_native) is not bool
            or type(exact_condition) is not bool
            or type(direct_authority) is not bool
            or direct_authority is not (exact_native and exact_condition)
        ):
            raise PreservationReviewHTMLError(
                f"{cell_id} cross-variant comparison authority differs"
            )
        if str(raw["source_iid"]) in training_iids:
            raise PreservationReviewHTMLError(
                f"{cell_id} review source must be held out from training"
            )
        if type(raw.get("seed")) is not int:
            raise PreservationReviewHTMLError(f"{cell_id} seed differs")
        cell_root = media_root / cell_id
        if not cell_root.is_dir() or cell_root.is_symlink():
            raise PreservationReviewHTMLError(f"media root missing for {cell_id}")
        for basename in CELL_FILES:
            _plain_file(cell_root / basename, label=f"{cell_id}/{basename}")
        _validate_receipt(cell_root / "rank8-receipt.json", cell_id=cell_id, expected_rank=8)
        _validate_receipt(cell_root / "rank2-receipt.json", cell_id=cell_id, expected_rank=2)
        cells.append(dict(raw))
    return cells, experiments


def _experiment_card(label: str, experiment: Mapping[str, Any]) -> str:
    diagnostics = experiment["diagnostics"]
    return f"""
      <article class="experiment">
        <h3>{html.escape(label)}</h3>
        <p>adapter rank <b>{experiment['adapter_rank']}</b> · 2 real-source training clips · 40 exact-noop optimizer steps · holder {html.escape(str(experiment['holder_job']))} / {html.escape(str(experiment['node']))}</p>
        <dl>
          <div><dt>wrong-source gap &gt; 0</dt><dd>{int(diagnostics['wrong_source_gap_positive'])}/{int(diagnostics['wrong_source_gap_total'])}</dd></div>
          <div><dt>mean gap</dt><dd>{float(diagnostics['wrong_source_gap_mean']):.6f}</dd></div>
          <div><dt>training loss range</dt><dd>{float(diagnostics['loss_min']):.6f} – {float(diagnostics['loss_max']):.6f}</dd></div>
          <div><dt>gradient norm range</dt><dd>{float(diagnostics['grad_norm_min']):.6f} – {float(diagnostics['grad_norm_max']):.6f}</dd></div>
        </dl>
      </article>"""


def _video(label: str, filename: str, note: str) -> str:
    return f"""
      <figure>
        <figcaption><b>{html.escape(label)}</b><span>{html.escape(note)}</span></figcaption>
        <video controls muted loop playsinline preload="metadata" src="{html.escape(filename)}"></video>
      </figure>"""


def _cell_card(cell: Mapping[str, Any]) -> str:
    cell_id = html.escape(str(cell["cell_id"]))
    if cell["cross_variant_direct_comparison_authorized"]:
        matching_note = (
            '<p class="matching exact"><b>Cross-rank control exact:</b> '
            'rank-8 / rank-2 的 source condition 与 native MP4 均为 byte-exact；可直接横向比较。</p>'
        )
    else:
        matching_note = (
            '<p class="matching warning"><b>只允许 paired comparison：</b> '
            '两节点独立 ROCm VAE encode 的 source latent 有 low-bit 差异，native 也不相同。'
            '请分别比较 rank-8 native → rank-8 residual，以及 rank-2 native → rank-2 residual；'
            '不要把 rank-8 与 rank-2 直接归因于 adapter rank。</p>'
        )
    return f"""
    <article class="case" id="case-{cell_id}">
      <header><div><p class="eyebrow">cell {cell_id} · source IID {html.escape(str(cell['source_iid']))} · seed {int(cell['seed'])}</p><h2>{html.escape(str(cell['requested_action']))}</h2></div><span class="badge">manual verdict pending</span></header>
      <div class="prompts">
        <div><h3>Source state caption</h3><p>{html.escape(str(cell['source_action_caption']))}</p></div>
        <div><h3>Exact RV2V target caption</h3><p>{html.escape(str(cell['target_action_caption']))}</p></div>
      </div>
      {matching_note}
      <div class="videos">
        {_video('Source', f'{cell_id}/source.mp4', 'the real input video')}
        {_video('Native RV2V · rank-8 run', f'{cell_id}/rank8-native-rv2v.mp4', 'paired control for the rank-8 residual')}
        {_video('Preservation · rank 8', f'{cell_id}/rank8-preservation-residual.mp4', 'native RV2V + unit-gain exact-noop residual')}
        {_video('Native RV2V · rank-2 run', f'{cell_id}/rank2-native-rv2v.mp4', 'paired control for the rank-2 residual')}
        {_video('Preservation · rank 2', f'{cell_id}/rank2-preservation-residual.mp4', 'same run, lower adapter capacity')}
      </div>
      <details class="contact"><summary>5-frame overview（frames 0 / 20 / 40 / 60 / 80）</summary><p>从上到下：SOURCE、RANK-8 NATIVE、RANK-8 PRES、RANK-2 NATIVE、RANK-2 PRES。</p><img src="{cell_id}/review_5x5.jpg" alt="{cell_id} five-arm five-frame contact sheet"></details>
    </article>"""


def _page(
    manifest: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    experiments: Mapping[str, Mapping[str, Any]],
) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Preservation-only action editing review</title>
<style>
:root{{--bg:#09101c;--panel:#111b2d;--soft:#17243a;--line:#293954;--ink:#f0f5ff;--muted:#9eb0cb;--cyan:#61d8d8;--amber:#ffc76a}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#070c15,#0e1930);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{width:min(1680px,96vw);margin:auto;padding:40px 0 80px}}h1{{font-size:clamp(32px,5vw,64px);line-height:1;margin:.15em 0}}h2{{font-size:25px;margin:.2em 0}}h3{{margin:.2em 0 .5em}}p{{margin:.4em 0}}.lede{{max-width:1050px;color:#c8d5e9;font-size:17px}}.policy{{margin:25px 0;padding:16px 20px;border-left:4px solid var(--amber);background:#191b25;border-radius:9px}}.experiments{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:24px 0 34px}}.experiment,.case{{border:1px solid var(--line);background:rgba(17,27,45,.95);border-radius:16px;padding:20px}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:8px 18px}}dl div{{padding:9px;background:var(--soft);border-radius:8px}}dt{{color:var(--muted)}}dd{{margin:0;font-weight:750}}.case{{margin:20px 0}}header{{display:flex;justify-content:space-between;gap:15px;align-items:flex-start}}.eyebrow,figcaption span{{color:var(--muted)}}.badge{{white-space:nowrap;color:var(--amber);background:rgba(255,199,106,.12);padding:6px 10px;border-radius:999px;font-size:12px;font-weight:750;text-transform:uppercase}}.prompts{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}}.prompts>div{{background:var(--soft);border-radius:10px;padding:14px}}.matching{{margin:0 0 16px;padding:12px 14px;border-radius:9px}}.matching.exact{{background:rgba(97,216,216,.10);border:1px solid rgba(97,216,216,.35)}}.matching.warning{{background:rgba(255,199,106,.10);border:1px solid rgba(255,199,106,.45)}}.videos{{display:grid;grid-template-columns:repeat(5,minmax(240px,1fr));gap:12px;overflow-x:auto}}figure{{margin:0;min-width:240px}}figcaption{{min-height:56px;display:flex;flex-direction:column;padding:8px 2px}}video{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#03060c;border-radius:10px}}.contact{{margin-top:16px;padding:12px;background:var(--soft);border-radius:10px}}.contact summary{{cursor:pointer;font-weight:750;color:var(--cyan)}}.contact img{{display:block;width:100%;margin-top:10px;border-radius:8px}}.definitions{{margin-top:32px;border-top:1px solid var(--line);padding-top:22px}}code{{color:var(--cyan)}}@media(max-width:900px){{.experiments,.prompts{{grid-template-columns:1fr}}header{{display:block}}.badge{{display:inline-block;margin-top:9px}}}}
</style></head><body><main>
<p class="eyebrow">2026-08-14 · matched visual review packet</p>
<h1>Preservation-only residual × native action prior</h1>
<p class="lede">本页只回答一个可人工直接观察的问题：冻结的 native RV2V 已负责生成动作后，使用真实 source=no-op 训练的低秩 residual，是否更好地保留 source 的人物/物体、服装、背景与构图。训练规模只有 2 条真实 source clip；下方 2 个 review cell 均为未参与训练的 IID。每个 rank 都显示自己同一次运行中的 native control，确保 residual 只与严格配对的 native 比较。</p>
<div class="policy"><b>没有自动“成功 value”。</b> 本实验不使用 reward、frozen-feature scalar、VLM evaluator 或 synthetic target。下方数值只是训练与实现诊断，不能替代视频人工判断，也不能证明动作正确或 preservation 成功。</div>
<section class="experiments">{_experiment_card('Main', experiments['rank8'])}{_experiment_card('Low-capacity variant', experiments['rank2'])}</section>
{''.join(_cell_card(cell) for cell in cells)}
<section class="definitions"><h2>数值究竟是什么</h2>
<p><b>wrong-source gap</b>：同一个 exact-noop residual 目标下，错误 source 条件的 flow MSE 减去正确 source 条件的 flow MSE。它只测试网络输出是否依赖正确 source；它不是 reward，也不衡量动作完整性、camera 或 decoded-video preservation。</p>
<p><b>training loss</b>：<code>MSE((v_adapted_noop − v_frozen_noop), stopgrad(v_source_target − v_frozen_noop))</code>。target 是真实 source 自身，不是 synthetic edited target。</p>
<p><b>gradient norm</b>：每次更新的 adapter 梯度范数，仅用于确认优化没有断路/数值爆炸。</p>
<p><b>最终判定</b>：请分别人工看动作是否由 native prior 保持，以及 rank-8/rank-2 相对 native 是否减少 identity、appearance、background 漂移。当前页面故意保持 <code>manual verdict pending</code>。</p></section>
</main></body></html>"""


def build(*, manifest_path: str | Path, media_root: str | Path, output: str | Path) -> dict[str, Any]:
    manifest_file = _plain_file(manifest_path, label="review manifest")
    root = _plain_dir(media_root, label="media root")
    manifest = _load_object(manifest_file, label="review manifest")
    cells, experiments = _validate(manifest, root)
    destination = Path(output)
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise PreservationReviewHTMLError("output must be a fresh absolute path")
    payload = _page(manifest, cells, experiments).encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {"output": str(destination), "cell_count": len(cells), "bytes": len(payload)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(manifest_path=args.manifest, media_root=args.media_root, output=args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

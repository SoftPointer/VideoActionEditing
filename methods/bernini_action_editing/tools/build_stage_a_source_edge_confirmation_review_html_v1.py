#!/usr/bin/env python3
"""Build a fail-closed self-contained four-source confirmation review."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import stage_a_source_edge_confirmation_contract_v1 as contract  # noqa: E402


OUTPUT_SCHEMA = "bernini-stage-a-source-edge-four-source-confirmation-review-v1"
BRANCH_LABELS = {
    "forward": "forward · 目标动作",
    "noop": "noop · 不执行目标动作",
    "reverse": "reverse · 反向动作",
    "incomplete": "incomplete · 未完成动作",
    "camera_only": "camera-only · 仅相机控制",
    "appearance_only": "appearance-only · 仅外观控制",
}


class ConfirmationReviewError(RuntimeError):
    """Raised before an incomplete review directory is published."""


def fail(message: str) -> NoReturn:
    raise ConfirmationReviewError(message)


def _copy_verified(source: Path, target: Path, sha256: str) -> None:
    if source.is_symlink() or not source.is_file() or contract.file_sha256(source) != sha256:
        fail(f"source bytes differ before copy: {source}")
    target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if contract.file_sha256(target) != sha256:
        fail(f"copied bytes differ: {target}")


def _source_card(
    *,
    sentinel_id: str,
    receipt: Mapping[str, Any],
    wrong: bool,
    wrong_metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    source = receipt["source_snapshots"]["wrong_owner" if wrong else "correct"]
    identity = receipt["sentinel"]
    title = "Wrong-owner source · 等 latent geometry" if wrong else "Correct source · 实际编辑输入"
    note = (
        "这个 source 来自另一个预注册 sentinel，仅保证 latent geometry 兼容；action、scene、entity 同时变化，因此不是纯 identity control。"
        if wrong
        else "本 section 的 native、source-on 和六个 source-off 分支都以这个完整 81 帧 source 为编辑输入。"
    )
    iid = identity["wrong_owner_iid"] if wrong else identity["iid"]
    role = identity["wrong_owner_sentinel_id"] if wrong else identity["diversity_role"]
    description = (
        str(wrong_metadata["source_caption"])
        if wrong and isinstance(wrong_metadata, Mapping)
        else str(identity["source_caption"])
    )
    action_family = (
        str(wrong_metadata["action_family"])
        if wrong and isinstance(wrong_metadata, Mapping)
        else str(identity["action_family"])
    )
    filename = "source-wrong-owner.mp4" if wrong else "source-correct.mp4"
    return f"""
      <article class="card source-card"><div class="card-head"><h4>{html.escape(title)}</h4><span class="badge">{html.escape(str(role))}</span></div>
        <video controls playsinline preload="metadata" src="cells/{html.escape(sentinel_id, quote=True)}/{filename}"></video>
        <p><b>完整 source description</b><br>{html.escape(description)}</p><p>{html.escape(note)}</p>
        <dl><div><dt>source IID</dt><dd>{html.escape(str(iid))}</dd></div><div><dt>action family</dt><dd>{html.escape(action_family)}</dd></div><div><dt>MP4 SHA-256</dt><dd><code>{source['mp4_sha256']}</code></dd></div></dl>
      </article>"""


def _video_card(
    *,
    sentinel_id: str,
    record: Mapping[str, Any],
    title: str,
    badge: str,
    note: str,
) -> str:
    filename = Path(record["relative_mp4"]).name
    return f"""
      <article class="card video-card"><div class="card-head"><h4>{html.escape(title)}</h4><span class="badge">{html.escape(badge)}</span></div>
        <video controls playsinline preload="metadata" src="cells/{html.escape(sentinel_id, quote=True)}/{html.escape(filename, quote=True)}"></video>
        <p class="instruction"><b>完整 instruction</b><span>{html.escape(str(record['instruction']))}</span></p>
        <p>{html.escape(note)}</p>
        <dl><div><dt>seed</dt><dd>{record['seed']}</dd></div><div><dt>81-frame MP4 SHA-256</dt><dd><code>{record['mp4_sha256']}</code></dd></div></dl>
      </article>"""


def render_html(
    *, manifest: Mapping[str, Any], shards: Sequence[Mapping[str, Any]]
) -> str:
    cell = manifest["admitted_cell"]
    blocks = cell["block_indices"]
    sections: list[str] = []
    manifest_sentinels = {
        row["sentinel_id"]: row for row in manifest["sentinels"]
    }
    for receipt in shards:
        sentinel = receipt["sentinel"]
        sentinel_id = sentinel["sentinel_id"]
        wrong_metadata = manifest_sentinels[sentinel["wrong_owner_sentinel_id"]]
        records = {row["key"]: row for row in receipt["records"]}
        native_cards = "".join(
            _video_card(
                sentinel_id=sentinel_id,
                record=records[f"native-correct-{branch}"],
                title=BRANCH_LABELS[branch],
                badge="native · correct source",
                note="Frozen Bernini RV2V 原生输出；没有安装 source-edge 数值删除。",
            )
            for branch in contract.BRANCHES
        )
        parity_key = next(key for key in records if key.startswith("parity-source-on-"))
        parity_card = _video_card(
            sentinel_id=sentinel_id,
            record=records[parity_key],
            title="forward · source-on parity",
            badge=f"source-on · s{cell['schedule_index']}×{cell['block_band']}",
            note="Hook 已安装，但 target query 到 source K/V 的 edge 保留并委托官方 attn1；predecode latent 与 native forward bit-exact。",
        )
        wrong_card = _video_card(
            sentinel_id=sentinel_id,
            record=records["native-wrong-owner-forward"],
            title="forward · compatible wrong-owner",
            badge="native · wrong-owner",
            note="完整 source 与四帧 references 换成预注册的等 latent geometry source；action、scene 与 entity 混杂均明确保留。",
        )
        off_cards = "".join(
            _video_card(
                sentinel_id=sentinel_id,
                record=records[
                    f"off-s{cell['schedule_index']:02d}-{cell['block_band']}-{branch}"
                ],
                title=BRANCH_LABELS[branch],
                badge=f"source-off · s{cell['schedule_index']}×{cell['block_band']}",
                note=(
                    f"只在 denoise schedule index {cell['schedule_index']}、blocks "
                    f"{blocks[0]}–{blocks[-1]} 删除 target-query→source-prefix K/V edge；"
                    "source-query rows、target-to-target attention、prompt、noise 与其余 steps/blocks 保持原生。"
                ),
            )
            for branch in contract.BRANCHES
        )
        sections.append(
            f"""
    <section class="sentinel" id="{html.escape(sentinel_id, quote=True)}">
      <header><p class="eyebrow">{html.escape(str(sentinel['diversity_role']))}</p><h2>{html.escape(sentinel_id)}</h2>
        <p><b>source</b>: {html.escape(str(sentinel['source_caption']))}</p>
        <p><b>action family</b>: {html.escape(str(sentinel['action_family']))} · <b>correct IID</b>: {html.escape(str(sentinel['iid']))} · <b>seed</b>: {sentinel['seed']}</p></header>
      <h3>Source identity</h3><div class="grid source-grid">{_source_card(sentinel_id=sentinel_id, receipt=receipt, wrong=False)}{_source_card(sentinel_id=sentinel_id, receipt=receipt, wrong=True, wrong_metadata=wrong_metadata)}</div>
      <h3>Native typed controls · 6</h3><div class="grid">{native_cards}</div>
      <h3>Source-on parity 与 compatible wrong-owner</h3><div class="grid source-grid">{parity_card}{wrong_card}</div>
      <h3>Source-off typed controls · 6</h3><div class="grid">{off_cards}</div>
    </section>"""
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage-A winner · four-source confirmation</title>
<style>
:root{{--bg:#0a0d14;--panel:#121724;--line:#293249;--text:#eef3ff;--muted:#aab5ce;--accent:#6ce5c5;--warn:#ffc66d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}section{{margin:0 0 34px;padding:24px;border:1px solid var(--line);border-radius:18px;background:var(--panel)}}h1,h2,h3,h4,p{{margin-top:0}}h3{{margin-top:24px}}.lead{{font-size:17px;max-width:1050px}}.eyebrow,.badge{{color:var(--accent);font-weight:700}}.warning{{border-left:4px solid var(--warn);padding-left:14px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}.source-grid{{grid-template-columns:repeat(auto-fit,minmax(380px,1fr))}}.card{{padding:15px;border:1px solid var(--line);border-radius:13px;background:#0d1220}}.card-head{{display:flex;justify-content:space-between;gap:12px}}video{{display:block;width:100%;background:#000;border-radius:9px;margin:10px 0 13px}}.instruction span{{display:block;color:var(--text);margin-top:4px}}dl{{margin:10px 0 0}}dl div{{display:grid;grid-template-columns:145px 1fr;gap:8px}}dt{{color:var(--muted)}}dd{{margin:0;overflow-wrap:anywhere}}code{{color:#c7d8ff}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border:1px solid var(--line);text-align:left}}nav a{{color:var(--accent);margin-right:14px}}
</style></head><body><main>
<section class="overview"><p class="eyebrow">Stage-A · frozen inference · exact40 / exact81 / WORLD4</p><h1>单个 A1 winner 的 4-source-disjoint confirmation</h1>
<p class="lead">这里没有重新搜索 cell。唯一 cell 来自外部人工 authorization：<b>s{cell['schedule_index']} × {html.escape(str(cell['block_band']))}</b>，timestep {cell['timestep']}，sigma {html.escape(str(cell['sigma_decimal']))}，blocks {blocks[0]}–{blocks[-1]}。四个 correct source 两两不同，每个 source 在独立 node 上固定 seed，生成 14 个完整 81 帧视频。</p>
<p><b>Native</b> 是 correct source 下六条完整 instruction 的原生 RV2V；<b>source-on</b> 安装 hook 但保留该 edge，必须与 native forward predecode bit-exact；<b>source-off</b> 只在 admitted cell 删除 target-query→source-prefix K/V；<b>wrong-owner</b> 换成另一个等 latent geometry sentinel，包含 action/scene/entity confound。</p>
<p class="warning"><b>边界：</b>本页只提供 single-cell winner robustness evidence。它不是 Stage-B admission，也不改变 Stage-B 预注册的 two-band conjunctive rule。最终判断只能来自人工逐视频 review。</p>
<table><thead><tr><th>schedule index</th><th>timestep</th><th>sigma FP32 BE hex</th><th>sigma decimal</th><th>block band</th></tr></thead><tbody><tr><td>s{cell['schedule_index']}</td><td>{cell['timestep']}</td><td><code>{cell['sigma_float32_be_hex']}</code></td><td>{cell['sigma_decimal']}</td><td>{html.escape(str(cell['block_band']))} · blocks {blocks[0]}–{blocks[-1]}</td></tr></tbody></table>
<nav>{''.join(f'<a href="#{html.escape(item["sentinel"]["sentinel_id"], quote=True)}">{html.escape(item["sentinel"]["sentinel_id"])}</a>' for item in shards)}</nav></section>
{''.join(sections)}
</main></body></html>"""


def build(
    *,
    manifest_path_value: str | Path,
    expected_manifest_sha256: str,
    run_root_value: str | Path,
    output_dir_value: str | Path,
) -> Path:
    try:
        manifest_path = contract._plain_file(
            manifest_path_value, label="confirmation manifest"
        )
        manifest = contract.load_manifest(
            manifest_path,
            expected_file_sha256=expected_manifest_sha256,
            verify_files=True,
        )
    except Exception as error:
        raise ConfirmationReviewError(str(error)) from error
    run_root = Path(run_root_value).expanduser()
    output = Path(output_dir_value).expanduser()
    if not run_root.is_absolute() or not output.is_absolute() or output == Path("/"):
        fail("run root/output must be absolute")
    try:
        run_root = run_root.resolve(strict=True)
        parent = output.parent.resolve(strict=True)
    except OSError as error:
        raise ConfirmationReviewError("run root/output parent is unavailable") from error
    if (
        not run_root.is_dir()
        or run_root.is_symlink()
        or output.exists()
        or output.is_symlink()
        or parent != output.parent
    ):
        fail("review output must be a fresh canonical child")
    shards: list[Mapping[str, Any]] = []
    receipt_paths: list[Path] = []
    for sentinel_id in contract.SENTINEL_ORDER:
        shard_root = run_root / "outputs" / sentinel_id
        try:
            receipt, receipt_path, _ = contract.load_receipt(
                shard_root,
                manifest_value=manifest,
                manifest_path=manifest_path,
                manifest_file_sha256=expected_manifest_sha256,
                sentinel_id=sentinel_id,
                verify_media=True,
            )
        except Exception as error:
            raise ConfirmationReviewError(str(error)) from error
        shards.append(receipt)
        receipt_paths.append(receipt_path)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        (staging / "cells").mkdir(mode=0o755)
        (staging / "evidence").mkdir(mode=0o755)
        _copy_verified(
            manifest_path,
            staging / "evidence" / "confirmation-manifest.json",
            expected_manifest_sha256,
        )
        evidence_files: dict[str, Mapping[str, str]] = {
            "confirmation_manifest": {
                "path": "evidence/confirmation-manifest.json",
                "file_sha256": expected_manifest_sha256,
            }
        }
        authority_copies = (
            (
                "persistent_review_manifest",
                manifest["review_manifest"],
                "persistent-review-manifest.json",
                "path",
                "file_sha256",
            ),
            (
                "confirmation_authorization",
                manifest["confirmation_authorization"],
                "confirmation-authorization.json",
                "path",
                "file_sha256",
            ),
        )
        for key, record, filename, path_field, sha_field in authority_copies:
            source = contract._plain_file(record[path_field], label=key)
            _copy_verified(
                source, staging / "evidence" / filename, record[sha_field]
            )
            evidence_files[key] = {
                "path": f"evidence/{filename}",
                "file_sha256": record[sha_field],
            }
        for record in manifest["a1_formal_receipts"]:
            key = f"a1_{record['family']}_receipt"
            filename = f"a1-{record['family']}-receipt.json"
            source = contract._plain_file(
                record["receipt_path"], label=key
            )
            _copy_verified(
                source,
                staging / "evidence" / filename,
                record["receipt_file_sha256"],
            )
            evidence_files[key] = {
                "path": f"evidence/{filename}",
                "file_sha256": record["receipt_file_sha256"],
            }
        manifest_cells: list[Mapping[str, Any]] = []
        for receipt, receipt_path in zip(shards, receipt_paths):
            sentinel_id = receipt["sentinel"]["sentinel_id"]
            source_root = receipt_path.parent
            cell_root = staging / "cells" / sentinel_id
            cell_root.mkdir(mode=0o755)
            _copy_verified(
                receipt_path,
                cell_root / "receipt.json",
                contract.file_sha256(receipt_path),
            )
            local_outputs: list[Mapping[str, Any]] = []
            for source_name, filename in (
                ("correct", "source-correct.mp4"),
                ("wrong_owner", "source-wrong-owner.mp4"),
            ):
                record = receipt["source_snapshots"][source_name]
                _copy_verified(
                    source_root / record["relative_mp4"],
                    cell_root / filename,
                    record["mp4_sha256"],
                )
            for record in receipt["records"]:
                filename = Path(record["relative_mp4"]).name
                _copy_verified(
                    source_root / record["relative_mp4"],
                    cell_root / filename,
                    record["mp4_sha256"],
                )
                local_outputs.append(
                    {
                        "key": record["key"],
                        "video": f"cells/{sentinel_id}/{filename}",
                        "mp4_sha256": record["mp4_sha256"],
                        "instruction": record["instruction"],
                        "hook": record["hook"],
                    }
                )
            manifest_cells.append(
                {
                    "sentinel": receipt["sentinel"],
                    "receipt": f"cells/{sentinel_id}/receipt.json",
                    "correct_source": f"cells/{sentinel_id}/source-correct.mp4",
                    "wrong_owner_source": f"cells/{sentinel_id}/source-wrong-owner.mp4",
                    "outputs": local_outputs,
                }
            )
        output_manifest = {
            "schema_version": OUTPUT_SCHEMA,
            "evidence_role": contract.EVIDENCE_ROLE,
            "confirmation_manifest_file_sha256": expected_manifest_sha256,
            "confirmation_manifest_digest": manifest["manifest_digest"],
            "evidence_files": evidence_files,
            "admitted_cell": manifest["admitted_cell"],
            "sentinel_order": list(contract.SENTINEL_ORDER),
            "outputs_per_sentinel": contract.EXPECTED_OUTPUTS,
            "exact_steps": contract.NUM_STEPS,
            "frame_count": contract.FRAME_COUNT,
            "stage_b_admission": False,
            "manual_video_review_required": True,
            "cells": manifest_cells,
        }
        contract._walk_forbidden_keys(output_manifest)
        (staging / "manifest.json").write_text(
            json.dumps(
                output_manifest,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "index.html").write_text(
            render_html(manifest=manifest, shards=shards), encoding="utf-8"
        )
        for path in (staging / "manifest.json", staging / "index.html"):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output / "index.html"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        build(
            manifest_path_value=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            run_root_value=args.run_root,
            output_dir_value=args.output_dir,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

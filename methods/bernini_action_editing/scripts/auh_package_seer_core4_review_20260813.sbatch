#!/usr/bin/env bash
#SBATCH --job-name=seer-review
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=01:00:00

set -Eeuo pipefail
umask 022

exp=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_seer_event_erasure_v1_20260813
constrained="${exp}/heldout/constrained-core4-fresh256-exclusive-r5"
direct="${exp}/heldout/direct-core4-fresh256-exclusive-r5"
spec="${exp}/staging/seer-event-erasure-b0-r1/methods/bernini_action_editing/assets/self_generated_action_lora_heldout_core4_v1.json"
review="${exp}/review/seer-event-erasure-vs-direct-core4-r5"
archive="${exp}/review/seer-event-erasure-vs-direct-core4-r5.tar.gz"
iids=(99cde432839f4240 6ea45d35943742bb 311c82f83eca4a7f 6d346c38cf504493)

test ! -e "${review}"
test ! -e "${archive}"
for method_root in "${constrained}" "${direct}"; do
  test -f "${method_root}/task-0.COMPLETE"
  test -f "${method_root}/task-1.COMPLETE"
done
test "$(sha256sum "${spec}" | awk '{print $1}')" = 82fbe0f042d86f8d54aa254ce72a384e70aa5bdc3c1ac66d5422037cd4b4051c
mkdir -m 0755 -p "${review}"

for iid in "${iids[@]}"; do
  case_dir="${review}/${iid}"
  con_results="${constrained}/${iid}/results/${iid}"
  dir_results="${direct}/${iid}/results/${iid}"
  source="$(jq -r --arg iid "${iid}" '.cases[] | select(.iid == $iid) | .source_video' "${spec}")"
  test -f "${source}"
  for path in \
    "${con_results}/frozen_base.mp4" \
    "${con_results}/trained_adapter.mp4" \
    "${con_results}/paired-receipt.json" \
    "${dir_results}/frozen_base.mp4" \
    "${dir_results}/trained_adapter.mp4" \
    "${dir_results}/paired-receipt.json"; do
    test -f "${path}"
  done
  mkdir -m 0755 "${case_dir}"
  cp -- "${source}" "${case_dir}/source.mp4"
  cp -- "${con_results}/frozen_base.mp4" "${case_dir}/frozen_base.mp4"
  cp -- "${con_results}/trained_adapter.mp4" "${case_dir}/constrained_same_state.mp4"
  cp -- "${dir_results}/trained_adapter.mp4" "${case_dir}/direct_fm.mp4"
  cp -- "${con_results}/paired-receipt.json" "${case_dir}/constrained-paired-receipt.json"
  cp -- "${dir_results}/paired-receipt.json" "${case_dir}/direct-paired-receipt.json"

done

/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python -B - "${review}" "${spec}" <<'PY'
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
import sys

review = Path(sys.argv[1])
spec_path = Path(sys.argv[2])
spec = json.loads(spec_path.read_text(encoding="utf-8"))
rows = []
for case in spec["cases"]:
    iid = case["iid"]
    case_dir = review / iid
    files = {}
    for path in sorted(case_dir.iterdir()):
        if path.is_file():
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    rows.append({
        "iid": iid,
        "actor_family": case["actor_family"],
        "instruction": case["instruction"],
        "seed": case["seed"],
        "files": files,
    })
manifest = {
    "schema_version": "seer-event-erasure-vs-direct-core4-review-v1",
    "status": "decoded_pairs_ready_for_human_review_no_method_success_claim",
    "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    "constrained_training": "same-generated-coordinate FM + causal-motion + source-copy",
    "direct_training": "matched direct full-pair FM",
    "training_steps_each": 4,
    "fit_examples_each": 2,
    "heldout_examples": 4,
    "comparison_order": ["source", "frozen_base", "constrained_same_state", "direct_fm"],
    "rows": rows,
}
(review / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
cards = []
for row in rows:
    iid = row["iid"]
    labels = [
        ("Source", "source.mp4"),
        ("Frozen base", "frozen_base.mp4"),
        ("Constrained same-state", "constrained_same_state.mp4"),
        ("Direct full-pair FM", "direct_fm.mp4"),
    ]
    videos = "".join(
        f'<figure><figcaption>{html.escape(label)}</figcaption><video controls loop muted preload="metadata" src="{iid}/{name}"></video></figure>'
        for label, name in labels
    )
    cards.append(f'''<section class="card">
      <h2>{html.escape(row["actor_family"].title())} · {iid}</h2>
      <p class="instruction">{html.escape(row["instruction"])}</p>
      <div class="videos">{videos}</div>
      <p class="links"><a href="{iid}/constrained-paired-receipt.json">constrained receipt</a> · <a href="{iid}/direct-paired-receipt.json">direct receipt</a></p>
    </section>''')
document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SEER event-erasure vs direct FM · core-4</title><style>
:root{{--bg:#0b1020;--panel:#141b2d;--text:#eef2ff;--muted:#a8b2cc;--line:#2b3653;--accent:#7dd3fc}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin-bottom:6px}}.note{{color:var(--muted);max-width:1050px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:22px 0}}.instruction{{color:#d8def0}}.videos{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}figure{{margin:0}}figcaption{{font-weight:650;margin:0 0 6px}}video,img{{width:100%;background:#000;border-radius:8px}}details{{margin-top:14px}}summary{{cursor:pointer;color:var(--accent);margin-bottom:10px}}a{{color:var(--accent)}}.links{{margin-bottom:0}}@media(max-width:900px){{.videos{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:560px){{.videos{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>SEER event-erasure vs matched direct FM</h1><p class="note">顺序固定为 Source → Frozen base → constrained same-state objective → matched direct full-pair FM。两种 adapter 都只训练 4 steps、使用相同 2 个 self-generated fit examples；下列 4 个 confirmation sources 未进入 optimizer。页面只声明生成完成，方法是否成功需人工同时审查动作、身份、背景与相机。</p>{''.join(cards)}<p><a href="manifest.json">Machine-readable manifest</a></p></main></body></html>'''
(review / "index.html").write_text(document, encoding="utf-8")
PY

printf 'COMPLETE review=%s archive=%s\n' "${review}" "${archive}" >"${review}/COMPLETE"
find "${review}" -type f -print0 | sort -z | xargs -0 sha256sum >"${review}/SHA256SUMS"
tar -C "${review%/*}" -czf "${archive}" "${review##*/}"
sha256sum "${archive}" >"${archive}.sha256"

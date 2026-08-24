#!/usr/bin/env bash
#SBATCH --job-name=m7-qwenfinal-v2
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -Eeuo pipefail

run_root="${MOTIVE_RUN_ROOT:?set MOTIVE_RUN_ROOT}"
source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
expected_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?}"
python_bin="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
input_manifest="${run_root}/expansion/input/selected.jsonl"
qwen_root="${run_root}/expansion/qwen_v2/qwen_visual"
fused_output="${run_root}/expansion/fused_v2"
manifest_output="${run_root}/expansion/manifest_v2"
code_root="${source_snapshot}/methods/motive"

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${code_root}/motive/r7_qwen_merge.py" \
  "${code_root}/motive/r7_build_expansion_manifest.py" \
  "${input_manifest}" \
  "${qwen_root}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[r7-qwen-finalize] missing required path: ${required}" >&2
    exit 2
  fi
done

if [[ -L "${fused_output}" || ! -d "${fused_output}" ]]; then
  echo "[r7-qwen-finalize] fused output is absent: ${fused_output}" >&2
  exit 2
fi
if [[ -e "${manifest_output}" ]]; then
  echo "[r7-qwen-finalize] refusing existing output: ${manifest_output}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${code_root}:${source_snapshot}"
cd "${code_root}"

verify_source_closure() {
  "${python_bin}" -c '
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in (root / "SOURCE_FILES.jsonl").read_text().splitlines()
]
expected = {row["path"] for row in rows}
expected.update({"SOURCE_FILES.jsonl", "SOURCE_PROVENANCE.json"})
actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
}
links = [
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_symlink()
]
if actual != expected or links:
    raise SystemExit(
        "source snapshot closure differs: "
        f"missing={sorted(expected - actual)} "
        f"extra={sorted(actual - expected)} symlinks={links}"
    )
print(
    "[r7-qwen-finalize] source closure "
    f"files={len(actual)} extras=0 symlinks=0"
)
' "${source_snapshot}"
}

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${expected_tree_sha256}"
verify_source_closure

# The packed GPU job creates fused_v2 only after all eight inference shards
# finish.  Re-run the same strict merge in verification-only mode before
# classifying any pseudo-label.
for artifact in fused.jsonl summary.json done.json; do
  path="${fused_output}/${artifact}"
  if [[ -L "${path}" || ! -f "${path}" ]]; then
    echo "[r7-qwen-finalize] invalid fused artifact: ${path}" >&2
    exit 2
  fi
done

merge_validation="$(
  "${python_bin}" -m motive.r7_qwen_merge \
    --input "${input_manifest}" \
    --qwen-root "${qwen_root}" \
    --output-dir "${fused_output}" \
    --resume
)"
printf '%s\n' "${merge_validation}"
if [[ "${merge_validation}" != *"resume_verified=True"* ]]; then
  echo "[r7-qwen-finalize] merge did not report strict resume" >&2
  exit 2
fi

manifest_build="$(
  "${python_bin}" -m motive.r7_build_expansion_manifest \
    --input "${fused_output}/fused.jsonl" \
    --output-dir "${manifest_output}"
)"
printf '%s\n' "${manifest_build}"
if [[ "${manifest_build}" != *"resume_verified=False"* ]]; then
  echo "[r7-qwen-finalize] manifest build reported unexpected mode" >&2
  exit 2
fi

manifest_validation="$(
  "${python_bin}" -m motive.r7_build_expansion_manifest \
    --input "${fused_output}/fused.jsonl" \
    --output-dir "${manifest_output}" \
    --resume
)"
printf '%s\n' "${manifest_validation}"
if [[ "${manifest_validation}" != *"resume_verified=True"* ]]; then
  echo "[r7-qwen-finalize] manifest did not report strict resume" >&2
  exit 2
fi

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${expected_tree_sha256}"
verify_source_closure

sha256sum \
  "${fused_output}/fused.jsonl" \
  "${fused_output}/summary.json" \
  "${fused_output}/done.json" \
  "${manifest_output}/positives.jsonl" \
  "${manifest_output}/negatives.jsonl" \
  "${manifest_output}/review.jsonl" \
  "${manifest_output}/summary.json" \
  "${manifest_output}/done.json"

echo "[r7-qwen-finalize] strict fusion verification and manifest commit passed"

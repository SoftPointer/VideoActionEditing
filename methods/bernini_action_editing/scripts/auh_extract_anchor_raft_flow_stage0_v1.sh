#!/usr/bin/env bash
# Extract one Stage-0 anchor flow bundle inside Job 140846.

set -Eeuo pipefail

archive="${FLOW_STAGE0_SOURCE_ARCHIVE:?set source archive}"
archive_sha="${FLOW_STAGE0_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
manifest="${FLOW_STAGE0_MANIFEST:?set manifest}"
iid="${FLOW_STAGE0_IID:?set iid}"
output="${FLOW_STAGE0_FLOW_OUTPUT:?set fresh flow output}"

[[ -f "${archive}" && -f "${manifest}" && ! -e "${output}" ]]
[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
case "${iid}" in
  7b88a1ca1f804f41|841b5e0080a1441d|a35b590961d24694|a66e6818e4144928) ;;
  *) echo "unknown Stage-0 iid: ${iid}" >&2; exit 2 ;;
esac

python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
scratch="${SLURM_TMPDIR:-/tmp}/flow-stage0-extract-${SLURM_JOB_ID}-${SLURM_STEP_ID:-none}-${iid}-$$"
mkdir -p "${scratch}/source" "${scratch}/cache" "${output%/*}"
tar -xf "${archive}" -C "${scratch}/source"
method_root="${scratch}/source/methods/bernini_action_editing"
runner="${method_root}/extract_anchor_raft_flow_v1.py"

read -r source anchor < <("${python_bin}" -B - "${manifest}" "${iid}" <<'PY'
import json
import pathlib
import sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(manifest["remote_media_root"])
rows = {row["iid"]: row for row in manifest["rows"]}
if sys.argv[2] not in rows:
    raise SystemExit("iid absent from manifest")
print(root / sys.argv[2] / "source.mp4", root / sys.argv[2] / "anchor.mp4")
PY
)
[[ -f "${source}" && -f "${anchor}" && -f "${runner}" ]]

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
export XDG_CACHE_HOME="${scratch}/cache"
exec "${python_bin}" -B "${runner}" \
  --source "${source}" --anchor "${anchor}" --output "${output}"

#!/usr/bin/env bash
#SBATCH --job-name=saic-anchor-after-manifest
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

fail() { echo "[saic-anchor-after-manifest] ERROR: $*" >&2; exit 2; }

manifest="${SAIC_SOURCE_ANCHOR_MANIFEST:?}"
inner_launcher="${SAIC_SOURCE_ANCHOR_INNER_LAUNCHER:?}"
inner_launcher_sha256="${SAIC_SOURCE_ANCHOR_INNER_LAUNCHER_SHA256:?}"
python_bin="${SAIC_PYTHON_BIN:?}"

for value in "${manifest}" "${inner_launcher}" "${python_bin}"; do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
[[ "${inner_launcher_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "inner launcher SHA-256 differs"
manifest="$(realpath -e -- "${manifest}")"
inner_launcher="$(realpath -e -- "${inner_launcher}")"
python_bin="$(realpath -e -- "${python_bin}")"
[[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "manifest is not plain"
[[ -f "${inner_launcher}" && ! -L "${inner_launcher}" ]] || fail "inner launcher is not plain"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python differs"
[[ "$(sha256sum "${inner_launcher}" | awk '{print $1}')" == "${inner_launcher_sha256}" ]] || fail "inner launcher hash differs"

manifest_sha256="$("${python_bin}" -B - "${manifest}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
raw = path.read_bytes()

def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate manifest key: {key}")
        value[key] = item
    return value

try:
    value = json.loads(raw.decode("ascii"), object_pairs_hook=lambda pairs: _unique(pairs))
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"manifest JSON differs: {error}")

expected = {
    "schema_version", "optimizer_authorized", "source_root", "selection_seed",
    "frame_count", "fps", "train_count", "holdout_count",
    "selected_bucket_counts", "eligible_bucket_counts",
    "strict_action_iids_excluded", "wrong_source_policy",
    "holdout_used_by_optimizer", "train_rows", "holdout_rows",
    "input_closure", "manifest_digest",
}
if not isinstance(value, dict) or set(value) != expected:
    raise SystemExit("source-anchor manifest schema differs")
canonical = lambda item: json.dumps(
    item, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("ascii")
if raw != canonical(value) + b"\n":
    raise SystemExit("source-anchor manifest is not canonical")
unsigned = dict(value)
declared = unsigned.pop("manifest_digest")
if not isinstance(declared, str) or re.fullmatch(r"[0-9a-f]{64}", declared) is None:
    raise SystemExit("source-anchor manifest digest format differs")
if hashlib.sha256(canonical(unsigned)).hexdigest() != declared:
    raise SystemExit("source-anchor manifest digest differs")
train = value["train_rows"]
holdout = value["holdout_rows"]
train_iids = {row.get("iid") for row in train if isinstance(row, dict)}
holdout_iids = {row.get("iid") for row in holdout if isinstance(row, dict)}
if (
    value["schema_version"] != "bernini-saic-source-anchor-manifest-v1"
    or value["optimizer_authorized"] is not False
    or value["holdout_used_by_optimizer"] is not False
    or value["train_count"] != 64
    or value["holdout_count"] != 16
    or len(train) != 64
    or len(holdout) != 16
    or len(train_iids) != 64
    or len(holdout_iids) != 16
    or train_iids & holdout_iids
):
    raise SystemExit("source-anchor train/holdout closure differs")
print(hashlib.sha256(raw).hexdigest())
PY
)"
[[ "${manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "validated manifest SHA-256 differs"

export SAIC_SOURCE_ANCHOR_MANIFEST="${manifest}"
export SAIC_SOURCE_ANCHOR_MANIFEST_SHA256="${manifest_sha256}"
echo "[saic-anchor-after-manifest] PASS manifest_sha256=${manifest_sha256} entering=${inner_launcher}"
exec /usr/bin/bash "${inner_launcher}"

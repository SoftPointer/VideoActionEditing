#!/usr/bin/env bash
#SBATCH --job-name=bern-postvideo-qa359
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

archive_preflight_only="${BERNINI_POSTVIDEO_ARCHIVE_PREFLIGHT_ONLY:-0}"
source_archive="${BERNINI_POSTVIDEO_SOURCE_ARCHIVE:?set BERNINI_POSTVIDEO_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_POSTVIDEO_SOURCE_ARCHIVE_SHA256:?set BERNINI_POSTVIDEO_SOURCE_ARCHIVE_SHA256}"
tool_sha256="${BERNINI_POSTVIDEO_TOOL_SHA256:?set BERNINI_POSTVIDEO_TOOL_SHA256}"
python_bin="${BERNINI_POSTVIDEO_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

fail() {
  echo "[bernini-postvideo] ERROR: $*" >&2
  exit 2
}

require_absolute_nonroot() {
  local label="$1"
  local value="$2"
  [[ "${value}" == /* && "${value}" != "/" ]] || fail "${label} must be a non-root absolute path"
}

require_sha256() {
  local label="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "${label} must be one lowercase SHA-256"
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${label} must be a positive integer"
}

for binding in \
  "source_archive:${source_archive}" \
  "python_bin:${python_bin}"; do
  require_absolute_nonroot "${binding%%:*}" "${binding#*:}"
done
require_sha256 "tool_sha256" "${tool_sha256}"
require_sha256 "source_archive_sha256" "${source_archive_sha256}"
[[ "${archive_preflight_only}" == "0" || "${archive_preflight_only}" == "1" ]] || fail "archive preflight flag must be 0 or 1"
if [[ "${archive_preflight_only}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  fail "archive-only preflight is forbidden inside an allocated Slurm job"
fi

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive must be one non-symlink file"
[[ -x "${python_bin}" ]] || fail "Python executable is unavailable"

sha256_file() {
  "${python_bin}" -I - "$1" <<'PY'
from pathlib import Path
import hashlib
import sys

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
}

[[ "$(sha256_file "${source_archive}")" == "${source_archive_sha256}" ]] || fail "source archive hash differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
require_absolute_nonroot "scratch_parent" "${scratch_parent}"
[[ -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent must be one writable non-symlink directory"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-postvideo-${SLURM_JOB_ID:-manual}.XXXXXX")"
[[ -d "${task_scratch}" && ! -L "${task_scratch}" ]] || fail "private task scratch creation failed"
chmod 0700 "${task_scratch}"
mkdir "${task_scratch}/source" "${task_scratch}/hf" "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom"
archive_copy="${task_scratch}/source.archive"
cp "${source_archive}" "${archive_copy}"
chmod 0400 "${archive_copy}"
[[ -f "${archive_copy}" && ! -L "${archive_copy}" ]] || fail "private source archive copy differs"
[[ "$(sha256_file "${archive_copy}")" == "${source_archive_sha256}" ]] || fail "private source archive copy hash differs"

"${python_bin}" -I - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

archive_path = sys.argv[1]
try:
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
except (OSError, tarfile.TarError) as error:
    raise SystemExit(f"unsafe source archive: cannot read tar members: {error}")
if not members:
    raise SystemExit("unsafe source archive: archive is empty")
seen = set()
for member in members:
    name = member.name
    path = PurePosixPath(name)
    if not name or "\x00" in name or path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"unsafe source archive member path: {name!r}")
    normalized = path.as_posix()
    if normalized in seen:
        raise SystemExit(f"unsafe source archive duplicate member: {name!r}")
    seen.add(normalized)
    if not (member.isfile() or member.isdir()):
        raise SystemExit(
            "unsafe source archive member type (links/devices/fifos forbidden): "
            f"{name!r}"
        )
print(f"bernini_postvideo.archive_members={len(members)}")
PY

tar -xf "${archive_copy}" -C "${task_scratch}/source"
[[ "$(sha256_file "${archive_copy}")" == "${source_archive_sha256}" ]] || fail "private source archive changed during extraction"
method_root="${task_scratch}/source/methods/bernini_action_editing"
tool="${method_root}/tools/build_postvideo_quality_routing.py"
test_file="${method_root}/tests/test_build_postvideo_quality_routing.py"
[[ -f "${tool}" && ! -L "${tool}" ]] || fail "post-video audit tool is unavailable"
[[ -f "${test_file}" && ! -L "${test_file}" ]] || fail "post-video audit tests are unavailable"
[[ "$(sha256_file "${tool}")" == "${tool_sha256}" ]] || fail "post-video audit tool hash differs"

if [[ "${archive_preflight_only}" == "1" ]]; then
  echo "bernini_postvideo.status=VALID_ARCHIVE_PREFLIGHT_ONLY"
  exit 0
fi

source_revision="${BERNINI_POSTVIDEO_SOURCE_REVISION:?set BERNINI_POSTVIDEO_SOURCE_REVISION}"
input_manifest="${BERNINI_POSTVIDEO_INPUT_MANIFEST:?set BERNINI_POSTVIDEO_INPUT_MANIFEST}"
input_sha256="${BERNINI_POSTVIDEO_INPUT_SHA256:-706d835a8cdf924776000d69b229c272fd434a91abc8942c67dc6fd7732b7d1b}"
input_receipt="${BERNINI_POSTVIDEO_INPUT_RECEIPT:?set BERNINI_POSTVIDEO_INPUT_RECEIPT}"
input_receipt_sha256="${BERNINI_POSTVIDEO_INPUT_RECEIPT_SHA256:-1a90b1ab7c999eaca320a59ec110d7782bf9807ba1653e2d8aade0ad4ed9cce6}"
output_dir="${BERNINI_POSTVIDEO_OUTPUT_DIR:?set BERNINI_POSTVIDEO_OUTPUT_DIR}"
qwen_model="${BERNINI_POSTVIDEO_QWEN_MODEL:-/vast/users/guangyi.chen/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
expected_strict_rows="${BERNINI_POSTVIDEO_EXPECTED_STRICT_ROWS:-359}"
nframes="${BERNINI_POSTVIDEO_NFRAMES:-12}"
max_pixels="${BERNINI_POSTVIDEO_MAX_PIXELS:-589824}"
max_new_tokens="${BERNINI_POSTVIDEO_MAX_NEW_TOKENS:-768}"

for binding in \
  "input_manifest:${input_manifest}" \
  "input_receipt:${input_receipt}" \
  "output_dir:${output_dir}" \
  "qwen_model:${qwen_model}"; do
  require_absolute_nonroot "${binding%%:*}" "${binding#*:}"
done
require_sha256 "input_sha256" "${input_sha256}"
require_sha256 "input_receipt_sha256" "${input_receipt_sha256}"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ && "${source_revision}" != "0000000000000000000000000000000000000000" ]] || fail "source_revision must be one nonzero 40-hex commit"
require_positive_integer "expected_strict_rows" "${expected_strict_rows}"
require_positive_integer "nframes" "${nframes}"
require_positive_integer "max_pixels" "${max_pixels}"
require_positive_integer "max_new_tokens" "${max_new_tokens}"

[[ -f "${input_manifest}" && ! -L "${input_manifest}" ]] || fail "input manifest must be one non-symlink file"
[[ -f "${input_receipt}" && ! -L "${input_receipt}" ]] || fail "input receipt must be one non-symlink file"
[[ -d "${qwen_model}" && ! -L "${qwen_model}" ]] || fail "Qwen snapshot must be one non-symlink directory"
[[ "${qwen_model##*/}" =~ ^[0-9a-f]{40}$ ]] || fail "Qwen snapshot must end in a pinned 40-hex revision"
[[ -f "${qwen_model}/config.json" ]] || fail "Qwen config.json is unavailable"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "create-only audit output already exists"
output_parent="${output_dir%/*}"
[[ -d "${output_parent}" && -w "${output_parent}" ]] || fail "audit output parent must already exist and be writable"

"${python_bin}" -I - "${input_manifest}" "${input_sha256}" "${input_receipt}" "${input_receipt_sha256}" <<'PY'
from pathlib import Path
import hashlib
import sys

manifest, expected_manifest, receipt, expected_receipt = sys.argv[1:]
for path_raw, expected, label in (
    (manifest, expected_manifest, "input manifest"),
    (receipt, expected_receipt, "input receipt"),
):
    digest = hashlib.sha256(Path(path_raw).read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"{label} hash differs: observed={digest} expected={expected}")
PY

export PYTHONPATH="${task_scratch}/source:${task_scratch}/source/methods/motive:${method_root}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export HF_HOME="${task_scratch}/hf"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"

"${python_bin}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ROCm/CUDA device is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
if "MI210" not in name:
    raise SystemExit(f"expected MI210, got {name}")
print(f"bernini_postvideo.device={name}")
print(f"bernini_postvideo.torch={torch.__version__}")
print(f"bernini_postvideo.hip={torch.version.hip}")
PY

echo "bernini_postvideo.mode=audit_only"
echo "bernini_postvideo.input_manifest=${input_manifest}"
echo "bernini_postvideo.input_sha256=${input_sha256}"
echo "bernini_postvideo.input_receipt=${input_receipt}"
echo "bernini_postvideo.input_receipt_sha256=${input_receipt_sha256}"
echo "bernini_postvideo.source_revision=${source_revision}"
echo "bernini_postvideo.source_archive_sha256=${source_archive_sha256}"
echo "bernini_postvideo.qwen_model=${qwen_model}"
echo "bernini_postvideo.output_dir=${output_dir}"
echo "bernini_postvideo.expected_strict_rows=${expected_strict_rows}"
echo "bernini_postvideo.inference_input_forbidden=true"
echo "bernini_postvideo.route_emitted=false"
echo "bernini_postvideo.training_invoked=false"

"${python_bin}" "${test_file}" -q

"${python_bin}" "${tool}" audit \
  --input-manifest "${input_manifest}" \
  --expected-input-sha256 "${input_sha256}" \
  --input-receipt "${input_receipt}" \
  --expected-input-receipt-sha256 "${input_receipt_sha256}" \
  --model "${qwen_model}" \
  --output-dir "${output_dir}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --expected-strict-rows "${expected_strict_rows}" \
  --nframes "${nframes}" \
  --max-pixels "${max_pixels}" \
  --max-new-tokens "${max_new_tokens}" \
  --attn-implementation sdpa

done_sha256="$("${python_bin}" - "${output_dir}/done.json" <<'PY'
from pathlib import Path
import hashlib
import sys

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

"${python_bin}" "${tool}" validate-audit \
  --audit-dir "${output_dir}" \
  --expected-done-sha256 "${done_sha256}" \
  --require-production

"${python_bin}" - "${output_dir}" "${expected_strict_rows}" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
expected = int(sys.argv[2])
if {path.name for path in root.iterdir()} != {"records.jsonl", "summary.json", "done.json"}:
    raise SystemExit("audit output is not the exact audit-only three-file closure")
done = json.loads((root / "done.json").read_text(encoding="utf-8"))
if done.get("rows") != expected or done.get("status") != "complete":
    raise SystemExit("audit completion/count differs")
for field in (
    "offline_postvideo_audit_only",
    "inference_input_forbidden",
    "target_video_as_inference_condition_forbidden",
    "qa_observation_as_inference_condition_forbidden",
):
    if done.get(field) is not True:
        raise SystemExit(f"audit safety gate differs: {field}")
if done.get("training_route_emitted_by_audit") is not False:
    raise SystemExit("audit unexpectedly emitted a training route")
print("bernini_postvideo.status=VALID_AUDIT_ONLY")
PY

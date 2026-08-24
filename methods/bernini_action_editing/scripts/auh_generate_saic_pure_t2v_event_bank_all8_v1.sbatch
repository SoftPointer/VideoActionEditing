#!/usr/bin/env bash
#SBATCH --job-name=saic-t2v-events
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

# Two concurrent legal WORLD4/SP4 groups use all eight MI210s.  Each group
# serially launches a fresh torchrun for every preregistered attempt because
# this is the exact lifecycle that completed in owner job 131524.  The dog
# group runs on GPUs 0..3 and the human group on GPUs 4..7.

source_archive="${SAIC_T2V_SOURCE_ARCHIVE:?set git source archive}"
source_archive_sha256="${SAIC_T2V_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${SAIC_T2V_SOURCE_REVISION:?set archive git revision}"
source_manifest="${SAIC_T2V_SOURCE_MANIFEST:?set immutable SAIC source manifest}"
source_manifest_sha256="${SAIC_T2V_SOURCE_MANIFEST_SHA256:?set source manifest SHA-256}"
event_spec="${SAIC_T2V_EVENT_SPEC:?set sealed SAIC T2V event spec}"
event_spec_sha256="${SAIC_T2V_EVENT_SPEC_SHA256:?set event spec SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${SAIC_T2V_OUTPUT_ROOT:?set fresh persistent output root}"
python_bin="${SAIC_T2V_PYTHON_BIN:?set vace Python executable}"
ffmpeg_bin="${SAIC_T2V_FFMPEG:?set absolute ffmpeg executable}"

fail() {
  echo "[saic-t2v-events] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  [[ "$1" == /* ]] || fail "$2 must be absolute"
}

for name in source_archive source_manifest event_spec bernini_root veomni_root checkpoint checkpoint_manifest output_root python_bin ffmpeg_bin; do
  require_absolute "${!name}" "${name}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${source_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source manifest SHA-256 differs"
[[ "${event_spec_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "event spec SHA-256 differs"

source_archive="$(realpath -e -- "${source_archive}")"
source_manifest="$(realpath -e -- "${source_manifest}")"
event_spec="$(realpath -e -- "${event_spec}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
ffmpeg_bin="$(realpath -e -- "${ffmpeg_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is not plain"
[[ -f "${source_manifest}" && ! -L "${source_manifest}" ]] || fail "source manifest is not plain"
[[ -f "${event_spec}" && ! -L "${event_spec}" ]] || fail "event spec is not plain"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
[[ -x "${ffmpeg_bin}" && -f "${ffmpeg_bin}" ]] || fail "ffmpeg executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision changed"
[[ "$(sha256sum "${source_manifest}" | awk '{print $1}')" == "${source_manifest_sha256}" ]] || fail "source manifest changed"
[[ "$(sha256sum "${event_spec}" | awk '{print $1}')" == "${event_spec_sha256}" ]] || fail "event spec changed"

"${python_bin}" -B - "${checkpoint}/transformer/config.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_absolute() or not path.is_file() or path.is_symlink():
    raise SystemExit("checkpoint transformer config is absent or not plain")
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("num_attention_heads") != 12:
    raise SystemExit("Bernini-R 1.3B must expose exactly 12 attention heads")
if 12 % 4:
    raise SystemExit("12 attention heads are not divisible by SP4")
PY

[[ "${output_root}" != / && ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
output_parent="$(dirname -- "${output_root}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_root##*/}" ]] || fail "output root must be canonical"
[[ "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "unsafe output basename"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-t2v-events-${SLURM_JOB_ID}.XXXXXX")"
dog_pid=""
human_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/saic-t2v-events-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[saic-t2v-events] PASS all8=true topology=2xWORLD4-SP4 exact81=true attempts=60 event_audit=pending optimizer=false output=${output_root}"
  else
    echo "[saic-t2v-events] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then kill "${child}" 2>/dev/null || true; fi
  done
  for child in "${dog_pid}" "${human_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then wait "${child}" 2>/dev/null || true; fi
  done
  exit "${status}"
}

trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
    "methods/bernini_action_editing/saic_pure_t2v_event_bank_v1.py",
    "methods/bernini_action_editing/generate_saic_pure_t2v_event_bank_v1.py",
    "methods/bernini_action_editing/build_saic_reversible_source_set_v1.py",
    "methods/bernini_action_editing/assets/saic_reversible_source_set_v1.json",
    "methods/bernini_action_editing/assets/saic_pure_t2v_event_bank_v1.json",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_pair_v5_t2v_calibration_bank.py",
    "methods/bernini_action_editing/pair_v5_t2v_calibration_bank_spec.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/tools/build_renderer_dataset.py",
    "methods/bernini_action_editing/tools/ffprobe_pyav_saic.py",
    "methods/bernini_action_editing/tools/materialize_vae.py",
    "methods/bernini_action_editing/tests/test_saic_pure_t2v_event_bank_v1.py",
    "methods/bernini_action_editing/tests/test_generate_saic_pure_t2v_event_bank_v1.py",
    "methods/bernini_action_editing/tests/test_auh_generate_saic_pure_t2v_event_bank_launcher.py",
    "methods/bernini_action_editing/tests/test_ffprobe_pyav_saic.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("archive member escaped or is not a plain tree entry")
        seen.add(path.as_posix().lstrip("./"))
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks SAIC pure-T2V closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +
runtime="${method_root}/generate_saic_pure_t2v_event_bank_v1.py"
contract_runtime="${method_root}/saic_pure_t2v_event_bank_v1.py"
ffprobe_bin="${method_root}/tools/ffprobe_pyav_saic.py"
[[ -x "${ffprobe_bin}" && -f "${ffprobe_bin}" && ! -L "${ffprobe_bin}" ]] || fail "archive-bound PyAV ffprobe differs"

for test_file in test_saic_pure_t2v_event_bank_v1.py test_generate_saic_pure_t2v_event_bank_v1.py test_auh_generate_saic_pure_t2v_event_bank_launcher.py test_ffprobe_pyav_saic.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
done

mkdir -- "${output_root}"
mkdir -- "${output_root}/logs" "${output_root}/attempts"
cp -- "${source_manifest}" "${output_root}/sealed-saic-source-manifest.json"
cp -- "${event_spec}" "${output_root}/sealed-saic-t2v-event-spec.json"
chmod 0444 -- "${output_root}/sealed-saic-source-manifest.json" "${output_root}/sealed-saic-t2v-event-spec.json"
sealed_source="${output_root}/sealed-saic-source-manifest.json"
sealed_spec="${output_root}/sealed-saic-t2v-event-spec.json"

"${python_bin}" -B "${contract_runtime}" materialize-proxies \
  --source-manifest "${sealed_source}" \
  --spec "${sealed_spec}" \
  --expected-spec-raw-sha256 "${event_spec_sha256}" \
  --output-dir "${output_root}/geometry-proxies" \
  --ffmpeg "${ffmpeg_bin}" \
  --ffprobe "${ffprobe_bin}" \
  >"${output_root}/logs/materialize-proxies.json"

"${python_bin}" -B "${contract_runtime}" materialize-plan \
  --source-manifest "${sealed_source}" \
  --spec "${sealed_spec}" \
  --expected-spec-raw-sha256 "${event_spec_sha256}" \
  --proxy-receipt "${output_root}/geometry-proxies/geometry-proxy-receipt.json" \
  --output-dir "${output_root}/plan" \
  >"${output_root}/logs/materialize-plan.json"

echo "[saic-t2v-events] semantic_inputs=text_only real_source_rgb=false source_latent=false source_noise=false target=false donor=false mask=false flow=false pose=false"
echo "[saic-t2v-events] GPUs0-3=dog GPUs4-7=human candidates_per_group=30 branches=forward,reverse,noop attempts=60"

job_mod=$((SLURM_JOB_ID % 10000))
dog_port=$((41000 + 2 * job_mod))
human_port=$((dog_port + 1))

run_group() (
  set -Eeuo pipefail
  local group_id="$1" visible="$2" master_port="$3"
  local group_scratch="${task_scratch}/groups/${group_id}"
  mkdir -p -- "${group_scratch}/miopen-user" "${group_scratch}/miopen-custom" \
    "${group_scratch}/torch-extensions" "${group_scratch}/triton" \
    "${group_scratch}/xdg" "${group_scratch}/pycache" "${group_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_scratch}/torch-extensions"
  export TRITON_CACHE_DIR="${group_scratch}/triton"
  export XDG_CACHE_HOME="${group_scratch}/xdg"
  export PYTHONPYCACHEPREFIX="${group_scratch}/pycache"
  export TMPDIR="${group_scratch}/tmp"
  export TORCHELASTIC_ERROR_FILE="${group_scratch}/torch-elastic-error.json"
  local envelope candidate_id candidate_output
  while IFS= read -r envelope; do
    [[ -f "${envelope}" && ! -L "${envelope}" ]] || fail "candidate envelope differs"
    candidate_id="$("${python_bin}" -B - "${envelope}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="ascii"))["candidate"]["candidate_id"])
PY
)"
    [[ "${candidate_id}" =~ ^[a-z0-9][a-z0-9._-]{1,191}$ ]] || fail "candidate ID differs"
    candidate_output="${output_root}/attempts/${candidate_id}"
    [[ ! -e "${candidate_output}" && ! -L "${candidate_output}" ]] || fail "refusing attempt reuse"
    echo "[saic-t2v-events:${group_id}] START candidate=${candidate_id} visible=${visible}"
    "${python_bin}" -B -m torch.distributed.run \
      --nproc_per_node=4 \
      --master_addr=127.0.0.1 \
      --master_port="${master_port}" \
      "${runtime}" generate-attempt \
      --candidate-envelope "${envelope}" \
      --expected-root-spec-sha256 "${event_spec_sha256}" \
      --bernini-root "${bernini_root}" \
      --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" \
      --checkpoint-content-manifest "${checkpoint_manifest}" \
      --output-dir "${candidate_output}" \
      --method-source-revision "${source_revision}" \
      --method-source-archive-sha256 "${source_archive_sha256}"
  done < <(find "${output_root}/plan/${group_id}" -mindepth 1 -maxdepth 1 -type f -name '*.json' -print | sort)
)

run_group sp4-a "0,1,2,3" "${dog_port}" >"${output_root}/logs/sp4-a.log" 2>&1 &
dog_pid=$!
run_group sp4-b "4,5,6,7" "${human_port}" >"${output_root}/logs/sp4-b.log" 2>&1 &
human_pid=$!

dog_status=0
human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
  tail -n 200 -- "${output_root}/logs/sp4-a.log" >&2 || true
  tail -n 200 -- "${output_root}/logs/sp4-b.log" >&2 || true
  fail "one or more SP4 event groups failed dog=${dog_status} human=${human_status}"
fi

"${python_bin}" -B "${runtime}" audit-bank \
  --root-spec "${sealed_spec}" \
  --source-manifest "${sealed_source}" \
  --expected-root-spec-sha256 "${event_spec_sha256}" \
  --output-root "${output_root}" \
  >"${output_root}/logs/strong-audit.log"

[[ -f "${output_root}/saic-pure-t2v-event-bank-receipt.json" ]] || fail "master receipt missing"
echo "SAIC_PURE_T2V_EVENT_BANK_ALL8_V1_STRONG_AUDIT_OK"

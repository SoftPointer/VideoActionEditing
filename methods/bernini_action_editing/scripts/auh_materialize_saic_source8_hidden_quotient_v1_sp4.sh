#!/usr/bin/env bash

set -Eeuo pipefail

# Run one immutable dog/human source8 hidden materialization in an existing
# eight-GPU allocation while exposing only one four-GPU SP4 subgroup.

source_archive="${SAIC_SOURCE8_SOURCE_ARCHIVE:?set SAIC_SOURCE8_SOURCE_ARCHIVE}"
source_archive_sha256="${SAIC_SOURCE8_SOURCE_ARCHIVE_SHA256:?set SAIC_SOURCE8_SOURCE_ARCHIVE_SHA256}"
source_revision="${SAIC_SOURCE8_SOURCE_REVISION:?set SAIC_SOURCE8_SOURCE_REVISION}"
actor_family="${SAIC_SOURCE8_ACTOR_FAMILY:?set SAIC_SOURCE8_ACTOR_FAMILY}"
visible_gpus="${SAIC_SOURCE8_VISIBLE_GPUS:?set SAIC_SOURCE8_VISIBLE_GPUS}"
output_root="${SAIC_SOURCE8_OUTPUT_ROOT:?set SAIC_SOURCE8_OUTPUT_ROOT}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
python_bin="${SAIC_SOURCE8_PYTHON_BIN:?set SAIC_SOURCE8_PYTHON_BIN}"
ffprobe_bin="${SAIC_SOURCE8_FFPROBE_BIN:?set SAIC_SOURCE8_FFPROBE_BIN}"

readonly source_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/t2v-events-topup-r6-umaskfix-72f3a40-r1/sealed-saic-source-manifest.json
readonly source_manifest_sha256=899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9
readonly attempts_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/t2v-events-v1-533074b-r3/attempts
readonly required_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly required_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly required_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly required_ffprobe_sha256=356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5

fail() { echo "[source8-hidden:${actor_family:-unknown}] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be SHA-256"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be SHA-1"; }

[[ "${actor_family}" == dog || "${actor_family}" == human ]] || fail "actor family differs"
[[ "${visible_gpus}" =~ ^[0-7],[0-7],[0-7],[0-7]$ ]] || fail "visible GPU list differs"
[[ "$(tr ',' '\n' <<<"${visible_gpus}" | sort -u | wc -l)" == 4 ]] || fail "visible GPUs repeat"
for name in source_archive output_root bernini_root veomni_root checkpoint checkpoint_manifest python_bin ffprobe_bin; do
  require_absolute "${!name}" "${name}"
done
require_sha256 "${source_archive_sha256}" source_archive_sha256
require_sha1 "${source_revision}" source_revision

source_archive="$(realpath -e -- "${source_archive}")"
output_root="$(realpath -e -- "${output_root}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
ffprobe_bin="$(realpath -e -- "${ffprobe_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -d "${output_root}" && ! -L "${output_root}" ]] || fail "output root differs"
[[ ! -e "${output_root}/${actor_family}" && ! -L "${output_root}/${actor_family}" ]] || fail "group output is not fresh"
[[ -f "${source_manifest}" && ! -L "${source_manifest}" ]] || fail "source manifest differs"
[[ -d "${attempts_root}" && ! -L "${attempts_root}" ]] || fail "attempts root differs"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
[[ -x "${ffprobe_bin}" && -f "${ffprobe_bin}" && ! -L "${ffprobe_bin}" ]] || fail "ffprobe differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive SHA differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${source_manifest}" | awk '{print $1}')" == "${source_manifest_sha256}" ]] || fail "source manifest SHA differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${required_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest SHA differs"
[[ "$(sha256sum "${ffprobe_bin}" | awk '{print $1}')" == "${required_ffprobe_sha256}" ]] || fail "ffprobe SHA differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export PATH="$(dirname -- "${ffprobe_bin}"):${PATH}"
[[ "$(command -v ffprobe)" == "${ffprobe_bin}" ]] || fail "ffprobe PATH binding differs"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export ROCR_VISIBLE_DEVICES="${visible_gpus}"

[[ "${SLURM_JOB_ID:?requires an existing Slurm allocation}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/source8-hidden-${actor_family}-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/source8-hidden-${actor_family}-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[source8-hidden:${actor_family}] PASS sources=4 arms=12 forwards=24 optimizer=false output=${output_root}/${actor_family}"
  else
    echo "[source8-hidden:${actor_family}] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT TERM INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/cache/miopen-user" \
  "${task_scratch}/cache/miopen-custom" "${task_scratch}/cache/torch-extensions" \
  "${task_scratch}/cache/triton" "${task_scratch}/cache/xdg" \
  "${task_scratch}/cache/pycache" "${task_scratch}/tmp"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" \
  -C "${task_scratch}/source-tree" methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
materializer="${method_root}/materialize_saic_source8_hidden_quotient_v1.py"
starc_source="${method_root}/materialize_starc_core4_hidden_v1.py"
source_contract="${method_root}/build_saic_reversible_source_set_v1.py"
generation_contract="${method_root}/generate_saic_pure_t2v_event_bank_v1.py"
for source in "${materializer}" "${starc_source}" "${source_contract}" "${generation_contract}"; do
  [[ -f "${source}" && ! -L "${source}" ]] || fail "source closure differs: ${source}"
done
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +

materializer_sha="$(sha256sum "${materializer}" | awk '{print $1}')"
starc_sha="$(sha256sum "${starc_source}" | awk '{print $1}')"
source_contract_sha="$(sha256sum "${source_contract}" | awk '{print $1}')"
generation_contract_sha="$(sha256sum "${generation_contract}" | awk '{print $1}')"

export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"
export TMPDIR="${task_scratch}/tmp"
export TORCHELASTIC_ERROR_FILE="${task_scratch}/torch-elastic-error.json"
if [[ "${actor_family}" == dog ]]; then
  actor_port_offset=0
else
  actor_port_offset=1
fi
master_port=$((31000 + 2 * (SLURM_JOB_ID % 500) + actor_port_offset))

echo "[source8-hidden:${actor_family}] START node=$(hostname) visible=${visible_gpus} world=4 block=15 schedule=33"
PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${master_port}" \
  "${materializer}" materialize-group \
  --actor-family "${actor_family}" \
  --source-manifest "${source_manifest}" \
  --attempts-root "${attempts_root}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --output-root "${output_root}" \
  --expected-bernini-commit "${required_bernini_commit}" \
  --expected-veomni-commit "${required_veomni_commit}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --expected-materializer-source-sha256 "${materializer_sha}" \
  --expected-starc-source-sha256 "${starc_sha}" \
  --expected-source-contract-sha256 "${source_contract_sha}" \
  --expected-generation-contract-sha256 "${generation_contract_sha}" \
  --ack-hidden-diagnostic-only \
  --ack-no-generated-media-editor-use \
  --ack-no-optimizer-or-editor-update

group_receipt="${output_root}/${actor_family}/saic-source8-hidden-group-${actor_family}-v1.json"
[[ -f "${group_receipt}" && ! -L "${group_receipt}" ]] || fail "group receipt missing"
[[ "$(find "${output_root}/${actor_family}" -type f -name saic-source8-block15-hidden-residual.safetensors | wc -l)" == 12 ]] || fail "artifact count differs"
[[ "$(find "${output_root}/${actor_family}" -type f -name saic-source8-hidden-arm-receipt-v1.json | wc -l)" == 12 ]] || fail "arm receipt count differs"

#!/usr/bin/env bash
#SBATCH --job-name=saic-t2v-topup-r6
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
umask 077
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
  echo "[saic-t2v-topup-v3] ERROR: Bash 5.1+ is required for fail-fast wait -n -p" >&2
  exit 2
fi

# Two concurrent legal WORLD4/SP4 groups use all eight MI210s.  Each group
# serially launches a fresh torchrun for every preregistered attempt because
# this is the exact lifecycle that completed in owner job 131524.  This job is
# an additive v2 top-up and never reads, writes, or resumes an earlier output.
# The dog group runs on GPUs 0..3 and the human group on GPUs 4..7.
# Operational revision v3 replaces unreserved arithmetic ports with atomic
# c10d port-zero allocation plus a permanent job-local claim and WORLD4 gate.

source_archive="${SAIC_T2V_V3_SOURCE_ARCHIVE:?set v3 git source archive}"
source_archive_sha256="${SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256:?set v3 archive SHA-256}"
source_revision="${SAIC_T2V_V3_SOURCE_REVISION:?set v3 archive git revision}"
source_manifest="${SAIC_T2V_V3_SOURCE_MANIFEST:?set immutable SAIC source manifest}"
source_manifest_sha256="${SAIC_T2V_V3_SOURCE_MANIFEST_SHA256:?set source manifest SHA-256}"
event_spec="${SAIC_T2V_V3_EVENT_SPEC:?set sealed unchanged SAIC T2V v2 top-up spec}"
event_spec_sha256="${SAIC_T2V_V3_EVENT_SPEC_SHA256:?set unchanged v2 top-up spec SHA-256}"
rendezvous_guard_sha256="${SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256:?set rendezvous guard SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${SAIC_T2V_V3_OUTPUT_ROOT:?set fresh v3-only persistent output root}"
python_bin="${SAIC_T2V_V3_PYTHON_BIN:?set vace Python executable}"
ffmpeg_bin="${SAIC_T2V_V3_STATIC_FFMPEG:?set compute-visible static ffmpeg executable}"
group_select="${SAIC_T2V_V3_GROUP_SELECT:-all8}"
readonly expected_runtime_sha256=3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36

fail() {
  echo "[saic-t2v-topup-v3] ERROR: $*" >&2
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
[[ "${rendezvous_guard_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "rendezvous guard SHA-256 differs"
[[ "${group_select}" == all8 || "${group_select}" == sp4-a || "${group_select}" == sp4-b ]] || \
  fail "group select must be all8, sp4-a, or sp4-b"

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
# The archive-bound ffprobe helper is shared with v1 and deliberately accepts
# only this generic, already-validated runtime-Python authority.
export SAIC_T2V_PYTHON_BIN="${python_bin}"
[[ -x "${ffmpeg_bin}" && -f "${ffmpeg_bin}" ]] || fail "ffmpeg executable differs"
ffmpeg_version_line="$("${ffmpeg_bin}" -version | sed -n '1p')"
[[ "${ffmpeg_version_line}" == *static* ]] || fail "ffmpeg must be a compute-visible static build"
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

# Selected-group mode is only legal inside an srun that exposes one genuinely
# idle quartet.  The same guard also catches an accidentally partial all8
# allocation before the fresh persistent output root is created.
visible_gpu_preflight="$("${python_bin}" -I -B - "${group_select}" <<'PY'
import json
import sys
import torch

mode = sys.argv[1]
count = torch.cuda.device_count()
rows = []
for index in range(count):
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    rows.append({
        "index": index,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "used_bytes": int(total_bytes - free_bytes),
    })
if mode == "all8":
    allowed_counts = {8}
    checked_indices = list(range(8))
elif mode == "sp4-a":
    # A four-GPU Slurm allocation is already renumbered 0..3.  An existing
    # eight-GPU allocation may also expose the physical logical quartet.
    allowed_counts = {4, 8}
    checked_indices = list(range(4))
else:
    # The sealed scientific envelope names GPUs 4..7.  A four-GPU allocation
    # renumbered to 0..3 cannot honestly execute this group.
    allowed_counts = {8}
    checked_indices = list(range(4, 8))
print(json.dumps(
    {
        "mode": mode,
        "allowed_counts": sorted(allowed_counts),
        "checked_indices": checked_indices,
        "count": count,
        "devices": rows,
    },
    sort_keys=True, separators=(",", ":"),
))
if count not in allowed_counts:
    raise SystemExit(2)
if any(rows[index]["used_bytes"] > 1024**3 for index in checked_indices):
    raise SystemExit(3)
PY
)" || fail "launcher did not receive the expected unoccupied GPU topology"
echo "[saic-t2v-topup-v3] visible_gpu_preflight=${visible_gpu_preflight}"

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
export SAIC_COMPUTE_STATIC_FFMPEG="${ffmpeg_bin}"
export IMAGEIO_FFMPEG_EXE="${ffmpeg_bin}" FFMPEG_BINARY="${ffmpeg_bin}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=0

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-t2v-topup-v3-${SLURM_JOB_ID}.XXXXXX")"
dog_pid=""
human_pid=""
if [[ "${group_select}" == all8 ]]; then
  completion_topology="all8=true topology=2xWORLD4-SP4 attempts=60"
else
  completion_topology="all8=false topology=1xWORLD4-SP4 partial_group=${group_select} attempts=30"
fi

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/saic-t2v-topup-v3-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[saic-t2v-topup-v3] PASS top_up_only=true ${completion_topology} exact81=true rendezvous=kernel-port0-plus-permanent-claim merged_branches=6 event_audit=pending seed_selection=false optimizer=false output=${output_root}"
  else
    echo "[saic-t2v-topup-v3] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
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
    "methods/bernini_action_editing/saic_pure_t2v_event_bank_topup_v2.py",
    "methods/bernini_action_editing/generate_saic_pure_t2v_event_bank_topup_v2.py",
    "methods/bernini_action_editing/saic_t2v_rendezvous_guard_v1.py",
    "methods/bernini_action_editing/saic_t2v_rendezvous_guard_v2.py",
    "methods/bernini_action_editing/build_saic_reversible_source_set_v1.py",
    "methods/bernini_action_editing/assets/saic_reversible_source_set_v1.json",
    "methods/bernini_action_editing/assets/saic_pure_t2v_event_bank_v1.json",
    "methods/bernini_action_editing/assets/saic_pure_t2v_event_bank_topup_v2.json",
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
    "methods/bernini_action_editing/tests/test_saic_pure_t2v_event_bank_topup_v2.py",
    "methods/bernini_action_editing/tests/test_generate_saic_pure_t2v_event_bank_topup_v2.py",
    "methods/bernini_action_editing/tests/test_auh_generate_saic_pure_t2v_event_bank_topup_v2_launcher.py",
    "methods/bernini_action_editing/tests/test_saic_t2v_rendezvous_guard_v1.py",
    "methods/bernini_action_editing/tests/test_saic_t2v_rendezvous_guard_v2.py",
    "methods/bernini_action_editing/tests/test_auh_generate_saic_pure_t2v_event_bank_topup_v3_launcher.py",
    "methods/bernini_action_editing/tests/test_ffprobe_pyav_saic.py",
    "methods/bernini_action_editing/scripts/auh_generate_saic_pure_t2v_event_bank_all8_v1.sbatch",
    "methods/bernini_action_editing/scripts/auh_generate_saic_pure_t2v_event_bank_topup_all8_v2.sbatch",
    "methods/bernini_action_editing/scripts/auh_generate_saic_pure_t2v_event_bank_topup_all8_v3.sbatch",
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
    raise SystemExit(f"archive lacks SAIC pure-T2V v3 operational closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +
runtime="${method_root}/generate_saic_pure_t2v_event_bank_topup_v2.py"
contract_runtime="${method_root}/saic_pure_t2v_event_bank_topup_v2.py"
rendezvous_guard="${method_root}/saic_t2v_rendezvous_guard_v2.py"
base_v1_spec="${method_root}/assets/saic_pure_t2v_event_bank_v1.json"
ffprobe_bin="${method_root}/tools/ffprobe_pyav_saic.py"
[[ -x "${ffprobe_bin}" && -f "${ffprobe_bin}" && ! -L "${ffprobe_bin}" ]] || fail "archive-bound PyAV ffprobe differs"
[[ -f "${runtime}" && ! -L "${runtime}" && "$(sha256sum "${runtime}" | awk '{print $1}')" == "${expected_runtime_sha256}" ]] || fail "unchanged v2 generation runtime bytes differ"
[[ -f "${rendezvous_guard}" && ! -L "${rendezvous_guard}" && "$(sha256sum "${rendezvous_guard}" | awk '{print $1}')" == "${rendezvous_guard_sha256}" ]] || fail "rendezvous guard bytes differ"

for test_file in test_saic_pure_t2v_event_bank_v1.py test_generate_saic_pure_t2v_event_bank_v1.py test_auh_generate_saic_pure_t2v_event_bank_launcher.py test_saic_pure_t2v_event_bank_topup_v2.py test_generate_saic_pure_t2v_event_bank_topup_v2.py test_auh_generate_saic_pure_t2v_event_bank_topup_v2_launcher.py test_saic_t2v_rendezvous_guard_v1.py test_saic_t2v_rendezvous_guard_v2.py test_auh_generate_saic_pure_t2v_event_bank_topup_v3_launcher.py test_ffprobe_pyav_saic.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
done

mkdir -- "${output_root}"
mkdir -- "${output_root}/logs" "${output_root}/attempts"
mkdir -m 0700 -- "${output_root}/logs/rendezvous"
mkdir -m 0700 -- "${output_root}/logs/rendezvous/port-claims"
rendezvous_root="${output_root}/logs/rendezvous"
port_claim_root="${rendezvous_root}/port-claims"
port_claim_root_identity="$(stat -Lc '%d:%i' -- "${port_claim_root}")"
cp -- "${source_manifest}" "${output_root}/sealed-saic-source-manifest.json"
cp -- "${event_spec}" "${output_root}/sealed-saic-t2v-event-topup-v2-spec.json"
cp -- "${base_v1_spec}" "${output_root}/sealed-base-saic-t2v-event-v1-spec.json"
chmod 0444 -- "${output_root}/sealed-saic-source-manifest.json" "${output_root}/sealed-saic-t2v-event-topup-v2-spec.json" "${output_root}/sealed-base-saic-t2v-event-v1-spec.json"
sealed_source="${output_root}/sealed-saic-source-manifest.json"
sealed_spec="${output_root}/sealed-saic-t2v-event-topup-v2-spec.json"
sealed_base_v1_spec="${output_root}/sealed-base-saic-t2v-event-v1-spec.json"

"${python_bin}" -B "${contract_runtime}" materialize-proxies \
  --source-manifest "${sealed_source}" \
  --base-v1-spec "${sealed_base_v1_spec}" \
  --spec "${sealed_spec}" \
  --expected-spec-raw-sha256 "${event_spec_sha256}" \
  --output-dir "${output_root}/geometry-proxies" \
  --ffmpeg "${ffmpeg_bin}" \
  --ffprobe "${ffprobe_bin}" \
  >"${output_root}/logs/materialize-proxies.json"

"${python_bin}" -B "${contract_runtime}" materialize-plan \
  --source-manifest "${sealed_source}" \
  --base-v1-spec "${sealed_base_v1_spec}" \
  --spec "${sealed_spec}" \
  --expected-spec-raw-sha256 "${event_spec_sha256}" \
  --proxy-receipt "${output_root}/geometry-proxies/geometry-proxy-receipt.json" \
  --output-dir "${output_root}/plan" \
  >"${output_root}/logs/materialize-plan.json"

echo "[saic-t2v-topup-v3] semantic_inputs=text_only black_proxy_only=true real_source_rgb=false source_latent=false source_noise=false target=false reference=false donor=false mask=false flow=false pose=false"
if [[ "${group_select}" == all8 ]]; then
  echo "[saic-t2v-topup-v3] GPUs0-3=dog GPUs4-7=human candidates_per_group=30 branches=incomplete,camera_only,appearance_only attempts=60 top_up_only=true rendezvous=c10d@127.0.0.1:0"
else
  echo "[saic-t2v-topup-v3] logical_GPUs0-3=${group_select} candidates_per_group=30 branches=incomplete,camera_only,appearance_only attempts=30 partial_only=true top_up_only=true rendezvous=c10d@127.0.0.1:0"
fi

readonly candidates_per_group=30
readonly maximum_rendezvous_launches=16

run_group() (
  set -Eeuo pipefail
  local group_id="$1" visible="$2"
  local group_scratch="${task_scratch}/groups/${group_id}"
  local group_rendezvous_root="${rendezvous_root}/${group_id}"
  mkdir -p -- "${group_scratch}/miopen-user" "${group_scratch}/miopen-custom" \
    "${group_scratch}/torch-extensions" "${group_scratch}/triton" \
    "${group_scratch}/xdg" "${group_scratch}/pycache" "${group_scratch}/tmp"
  mkdir -m 0700 -- "${group_rendezvous_root}"
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
  local envelope candidate_id candidate_output candidate_digest rdzv_id_prefix rdzv_id
  local candidate_lifecycle_root launch_dir lifecycle_identity completion_receipt
  local torchrun_status launch_ordinal candidate_completed
  local candidate_index=0
  local rendezvous_plan="${output_root}/logs/${group_id}-rendezvous-dynamic-plan-v1.json"
  "${python_bin}" -B - \
    "${output_root}/plan/${group_id}" "${group_id}" "${SLURM_JOB_ID}" \
    "${candidates_per_group}" "${maximum_rendezvous_launches}" \
    >"${rendezvous_plan}" <<'PY'
from pathlib import Path
import hashlib
import json
import re
import sys

plan_dir = Path(sys.argv[1])
group_id = sys.argv[2]
slurm_job_id = sys.argv[3]
expected_count, maximum_launches = map(int, sys.argv[4:])
envelopes = sorted(plan_dir.glob("*.json"), key=lambda path: path.name)
if len(envelopes) != expected_count:
    raise SystemExit("rendezvous plan does not contain exactly 30 envelopes")
rows = []
for index, envelope in enumerate(envelopes):
    if not envelope.is_file() or envelope.is_symlink():
        raise SystemExit("rendezvous plan envelope is not plain")
    payload = envelope.read_bytes()
    value = json.loads(payload.decode("ascii"))
    candidate_id = value.get("candidate", {}).get("candidate_id")
    if not isinstance(candidate_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,191}", candidate_id) is None:
        raise SystemExit("rendezvous plan candidate ID differs")
    candidate_digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()
    rdzv_id_prefix = f"saic-{slurm_job_id}-{group_id}-c{index:02d}-{candidate_digest[:16]}"
    rows.append({
        "candidate_index": index,
        "candidate_id": candidate_id,
        "envelope_path": str(envelope),
        "envelope_sha256": hashlib.sha256(payload).hexdigest(),
        "candidate_id_sha256": candidate_digest,
        "rdzv_id_prefix": rdzv_id_prefix,
        "requested_rendezvous_endpoint": "127.0.0.1:0",
        "numeric_port_preregistered": False,
    })
receipt = {
    "schema_version": "saic-t2v-topup-rendezvous-dynamic-plan-v2",
    "group_id": group_id,
    "slurm_job_id": slurm_job_id,
    "candidate_count": expected_count,
    "fixed_order": "lexicographic_envelope_basename",
    "rdzv_backend": "c10d",
    "rdzv_endpoint_request": "127.0.0.1:0",
    "kernel_atomic_port_allocation": True,
    "torch_disable_share_rdzv_tcp_store": "0",
    "shared_tcp_store_bootstrap_required": True,
    "numeric_ports_preregistered": False,
    "permanent_create_only_claim_across_both_groups": True,
    "maximum_operational_launches_per_candidate": maximum_launches,
    "retry_condition": "sealed_same_job_prior_admitted_port_claim_and_candidate_output_absent_only",
    "scientific_candidate_set_or_order_changed_by_retry": False,
    "candidate_rows": rows,
    "authority": {
        "scientific_selection": False,
        "training": False,
        "optimizer": False,
        "resume_or_partial_reuse": False,
    },
}
canonical = lambda item: json.dumps(
    item, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("ascii")
receipt["receipt_digest"] = hashlib.sha256(canonical(receipt)).hexdigest()
sys.stdout.buffer.write(canonical(receipt) + b"\n")
PY
  chmod 0444 -- "${rendezvous_plan}"
  while IFS= read -r envelope; do
    [[ "${candidate_index}" -ge 0 && "${candidate_index}" -lt "${candidates_per_group}" ]] || \
      fail "candidate index escaped fixed 0..29 order"
    [[ -f "${envelope}" && ! -L "${envelope}" ]] || fail "candidate envelope differs"
    [[ "$(stat -c '%a' -- "${envelope}")" == "444" ]] || \
      fail "candidate envelope mode is not terminal-auditor compatible"
    candidate_id="$("${python_bin}" -B - "${envelope}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="ascii"))["candidate"]["candidate_id"])
PY
    )"
    [[ "${candidate_id}" =~ ^[a-z0-9][a-z0-9._-]{1,191}$ ]] || fail "candidate ID differs"
    candidate_digest="$(printf '%s' "${candidate_id}" | sha256sum | awk '{print $1}')"
    [[ "${candidate_digest}" =~ ^[0-9a-f]{64}$ ]] || fail "candidate ID digest differs"
    rdzv_id_prefix="saic-${SLURM_JOB_ID}-${group_id}-c$(printf '%02d' "${candidate_index}")-${candidate_digest:0:16}"
    [[ "${rdzv_id_prefix}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,180}$ ]] || fail "rendezvous ID prefix differs"
    candidate_output="${output_root}/attempts/${candidate_id}"
    [[ ! -e "${candidate_output}" && ! -L "${candidate_output}" ]] || fail "refusing attempt reuse"
    candidate_lifecycle_root="${group_rendezvous_root}/candidate-$(printf '%02d' "${candidate_index}")-${candidate_digest:0:16}"
    [[ ! -e "${candidate_lifecycle_root}" && ! -L "${candidate_lifecycle_root}" ]] || fail "candidate lifecycle reuse differs"
    mkdir -m 0700 -- "${candidate_lifecycle_root}"
    candidate_completed=false
    for ((launch_ordinal=1; launch_ordinal<=maximum_rendezvous_launches; launch_ordinal++)); do
      [[ ! -e "${candidate_output}" && ! -L "${candidate_output}" ]] || fail "candidate output appeared before admitted runtime"
      launch_dir="${candidate_lifecycle_root}/launch-$(printf '%02d' "${launch_ordinal}")"
      [[ ! -e "${launch_dir}" && ! -L "${launch_dir}" ]] || fail "rendezvous launch lifecycle reuse differs"
      mkdir -m 0700 -- "${launch_dir}"
      lifecycle_identity="$(stat -Lc '%d:%i' -- "${launch_dir}")"
      rdzv_id="${rdzv_id_prefix}-l$(printf '%02d' "${launch_ordinal}")"
      completion_receipt="${launch_dir}/completion.json"
      echo "[saic-t2v-topup-v3:${group_id}] START candidate_index=${candidate_index} candidate=${candidate_id} launch_ordinal=${launch_ordinal} endpoint=127.0.0.1:0 visible=${visible}"
      if "${python_bin}" -B -m torch.distributed.run \
        --nproc-per-node=4 \
        --max-restarts=0 \
        --rdzv-backend=c10d \
        --rdzv-endpoint=127.0.0.1:0 \
        --rdzv-id="${rdzv_id}" \
        --local-addr=127.0.0.1 \
        "${rendezvous_guard}" worker \
        --runtime "${runtime}" \
        --expected-runtime-sha256 "${expected_runtime_sha256}" \
        --slurm-job-id "${SLURM_JOB_ID}" \
        --group-id "${group_id}" \
        --candidate-index "${candidate_index}" \
        --candidate-id "${candidate_id}" \
        --launch-ordinal "${launch_ordinal}" \
        --expected-rdzv-id "${rdzv_id}" \
        --claim-root "${port_claim_root}" \
        --claim-root-identity "${port_claim_root_identity}" \
        --lifecycle-dir "${launch_dir}" \
        --lifecycle-dir-identity "${lifecycle_identity}" \
        -- generate-attempt \
        --candidate-envelope "${envelope}" \
        --root-spec "${sealed_spec}" \
        --base-v1-spec "${sealed_base_v1_spec}" \
        --source-manifest "${sealed_source}" \
        --expected-root-spec-sha256 "${event_spec_sha256}" \
        --bernini-root "${bernini_root}" \
        --veomni-root "${veomni_root}" \
        --checkpoint "${checkpoint}" \
        --checkpoint-content-manifest "${checkpoint_manifest}" \
        --output-dir "${candidate_output}" \
        --method-source-revision "${source_revision}" \
        --method-source-archive-sha256 "${source_archive_sha256}"; then
        torchrun_status=0
      else
        torchrun_status=$?
      fi
      if [[ "${torchrun_status}" == 0 ]]; then
        "${python_bin}" -B "${rendezvous_guard}" assemble \
          --expected-runtime-sha256 "${expected_runtime_sha256}" \
          --slurm-job-id "${SLURM_JOB_ID}" --group-id "${group_id}" \
          --candidate-index "${candidate_index}" --candidate-id "${candidate_id}" \
          --launch-ordinal "${launch_ordinal}" --expected-rdzv-id "${rdzv_id}" \
          --claim-root "${port_claim_root}" --claim-root-identity "${port_claim_root_identity}" \
          --lifecycle-dir "${launch_dir}" --lifecycle-dir-identity "${lifecycle_identity}" \
          --candidate-output "${candidate_output}" --completion-receipt "${completion_receipt}" \
          >"${launch_dir}/assemble.stdout"
        chmod 0444 -- "${launch_dir}/assemble.stdout"
        candidate_completed=true
        break
      fi
      if "${python_bin}" -B "${rendezvous_guard}" admit-collision \
        --expected-runtime-sha256 "${expected_runtime_sha256}" \
        --slurm-job-id "${SLURM_JOB_ID}" --group-id "${group_id}" \
        --candidate-index "${candidate_index}" --candidate-id "${candidate_id}" \
        --launch-ordinal "${launch_ordinal}" --expected-rdzv-id "${rdzv_id}" \
        --claim-root "${port_claim_root}" --claim-root-identity "${port_claim_root_identity}" \
        --lifecycle-dir "${launch_dir}" --lifecycle-dir-identity "${lifecycle_identity}" \
        --candidate-output "${candidate_output}" >/dev/null; then
        [[ "${launch_ordinal}" -lt "${maximum_rendezvous_launches}" ]] || fail "sealed port-claim collision retry budget exhausted"
        echo "[saic-t2v-topup-v3:${group_id}] RETRY candidate_index=${candidate_index} candidate=${candidate_id} completed_scientific_attempts_unchanged=true"
        continue
      fi
      fail "generation runtime failed without retry-authorizing collision candidate=${candidate_id} status=${torchrun_status}"
    done
    [[ "${candidate_completed}" == true && -f "${completion_receipt}" && ! -L "${completion_receipt}" ]] || fail "candidate lacked exact admitted completion"
    candidate_index=$((candidate_index + 1))
  done < <(find "${output_root}/plan/${group_id}" -mindepth 1 -maxdepth 1 -type f -name '*.json' -print | sort)
  [[ "${candidate_index}" -eq "${candidates_per_group}" ]] || \
    fail "group did not execute exactly 30 fixed-order candidates"
)

if [[ "${group_select}" == all8 ]]; then
  run_group sp4-a "0,1,2,3" >"${output_root}/logs/sp4-a.log" 2>&1 &
  dog_pid=$!
  run_group sp4-b "4,5,6,7" >"${output_root}/logs/sp4-b.log" 2>&1 &
  human_pid=$!

  finished_pid=""
  set +e
  wait -n -p finished_pid "${dog_pid}" "${human_pid}"
  first_status=$?
  set -e
  if (( first_status != 0 )); then
    kill "${dog_pid}" "${human_pid}" 2>/dev/null || true
    set +e
    wait "${dog_pid}" 2>/dev/null
    wait "${human_pid}" 2>/dev/null
    set -e
    tail -n 200 -- "${output_root}/logs/sp4-a.log" >&2 || true
    tail -n 200 -- "${output_root}/logs/sp4-b.log" >&2 || true
    fail "first completed WORLD4 group failed pid=${finished_pid} status=${first_status}; sibling terminated"
  fi
  if [[ "${finished_pid}" == "${dog_pid}" ]]; then
    remaining_pid="${human_pid}"
    remaining_group=sp4-b
  elif [[ "${finished_pid}" == "${human_pid}" ]]; then
    remaining_pid="${dog_pid}"
    remaining_group=sp4-a
  else
    kill "${dog_pid}" "${human_pid}" 2>/dev/null || true
    set +e
    wait "${dog_pid}" 2>/dev/null
    wait "${human_pid}" 2>/dev/null
    set -e
    fail "wait -n -p returned an unknown child; both groups terminated"
  fi
  set +e
  wait "${remaining_pid}"
  remaining_status=$?
  set -e
  if (( remaining_status != 0 )); then
    tail -n 200 -- "${output_root}/logs/${remaining_group}.log" >&2 || true
    fail "${remaining_group} WORLD4 failed after sibling completed status=${remaining_status}"
  fi

  "${python_bin}" -B "${rendezvous_guard}" audit-job \
    --expected-runtime-sha256 "${expected_runtime_sha256}" \
    --slurm-job-id "${SLURM_JOB_ID}" \
    --output-root "${output_root}" \
    --claim-root "${port_claim_root}" \
    --claim-root-identity "${port_claim_root_identity}" \
    --receipt "${output_root}/logs/rendezvous-job-audit-v1.json" \
    >"${output_root}/logs/rendezvous-job-audit-v1.stdout"
  chmod 0444 -- "${output_root}/logs/rendezvous-job-audit-v1.stdout"

  "${python_bin}" -B "${runtime}" audit-bank \
    --root-spec "${sealed_spec}" \
    --base-v1-spec "${sealed_base_v1_spec}" \
    --source-manifest "${sealed_source}" \
    --expected-root-spec-sha256 "${event_spec_sha256}" \
    --output-root "${output_root}" \
    >"${output_root}/logs/strong-audit.log"

  [[ -f "${output_root}/saic-pure-t2v-event-bank-topup-receipt.json" ]] || fail "v2 master receipt missing"
  [[ -f "${output_root}/logs/rendezvous-job-audit-v1.json" && ! -L "${output_root}/logs/rendezvous-job-audit-v1.json" ]] || fail "v3 rendezvous exact60 receipt missing"
  echo "SAIC_PURE_T2V_EVENT_BANK_TOPUP_ALL8_V3_STRONG_AUDIT_OK"
else
  # Existing-allocation mode keeps the sealed logical GPU IDs.  sp4-a may run
  # in a four-GPU allocation renumbered 0..3; sp4-b requires an eight-GPU
  # allocation with the physical logical 4..7 quartet idle.  The preflight
  # above rejects a dishonest four-GPU sp4-b remap before output creation.
  # A partial receipt is diagnostic-only.  It cannot authorize training until
  # an independent merger verifies both disjoint groups and runs audit-bank.
  if [[ "${group_select}" == sp4-a ]]; then
    selected_visible="0,1,2,3"
  else
    selected_visible="4,5,6,7"
  fi
  if ! run_group "${group_select}" "${selected_visible}" >"${output_root}/logs/${group_select}.log" 2>&1; then
    tail -n 200 -- "${output_root}/logs/${group_select}.log" >&2 || true
    fail "selected WORLD4 group failed group=${group_select}"
  fi
  partial_receipt="${output_root}/saic-pure-t2v-event-bank-topup-partial-${group_select}-receipt.json"
  "${python_bin}" -I -B - \
    "${output_root}" "${group_select}" "${SLURM_JOB_ID}" \
    "${source_revision}" "${source_archive_sha256}" "${event_spec_sha256}" \
    "${partial_receipt}" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import stat
import sys

root = Path(sys.argv[1])
group_id, slurm_job_id = sys.argv[2:4]
source_revision, archive_sha, spec_sha = sys.argv[4:7]
receipt_path = Path(sys.argv[7])
if group_id not in {"sp4-a", "sp4-b"}:
    raise SystemExit("partial group differs")
plan_paths = sorted((root / "plan" / group_id).glob("*.json"))
if len(plan_paths) != 30:
    raise SystemExit("partial plan is not exact30")
rows = []
for index, plan_path in enumerate(plan_paths):
    plan_raw = plan_path.read_bytes()
    candidate_id = json.loads(plan_raw.decode("ascii"))["candidate"]["candidate_id"]
    attempt_dir = root / "attempts" / candidate_id
    attempt_receipt = attempt_dir / "saic-event-topup-generation-receipt.json"
    if not attempt_dir.is_dir() or attempt_dir.is_symlink():
        raise SystemExit("partial attempt directory differs")
    if not attempt_receipt.is_file() or attempt_receipt.is_symlink():
        raise SystemExit("partial attempt receipt differs")
    attempt_raw = attempt_receipt.read_bytes()
    attempt_value = json.loads(attempt_raw.decode("ascii"))
    if (
        attempt_value.get("group_id") != group_id
        or attempt_value.get("candidate", {}).get("candidate_id") != candidate_id
        or attempt_value.get("training_target_authorized") is not False
        or stat.S_IMODE(attempt_receipt.stat().st_mode) != 0o444
    ):
        raise SystemExit("partial attempt binding differs")
    completion_paths = sorted(
        (root / "logs" / "rendezvous" / group_id).glob(
            f"candidate-{index:02d}-*/launch-*/completion.json"
        )
    )
    if len(completion_paths) != 1:
        raise SystemExit("partial completion coverage differs")
    completion_raw = completion_paths[0].read_bytes()
    completion = json.loads(completion_raw.decode("ascii"))
    if (
        completion.get("group_id") != group_id
        or completion.get("candidate_index") != index
        or completion.get("candidate_id") != candidate_id
        or completion.get("slurm_job_id") != slurm_job_id
    ):
        raise SystemExit("partial completion binding differs")
    rows.append(
        {
            "candidate_index": index,
            "candidate_id": candidate_id,
            "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "attempt_receipt_path": str(attempt_receipt),
            "attempt_receipt_sha256": hashlib.sha256(attempt_raw).hexdigest(),
            "completion_receipt_path": str(completion_paths[0]),
            "completion_receipt_sha256": hashlib.sha256(completion_raw).hexdigest(),
        }
    )
value = {
    "schema_version": "saic-t2v-topup-existing-allocation-partial-group-v1",
    "status": "exact30_world4_group_complete_pending_disjoint_merge_and_full_audit",
    "group_id": group_id,
    "slurm_job_id": slurm_job_id,
    "source_revision": source_revision,
    "source_archive_sha256": archive_sha,
    "root_spec_raw_sha256": spec_sha,
    "candidate_count": 30,
    "logical_visible_devices": [0, 1, 2, 3] if group_id == "sp4-a" else [4, 5, 6, 7],
    "rows": rows,
    "authority": {
        "scientific_selection": False,
        "training": False,
        "optimizer": False,
        "merge_or_partial_reuse": False,
    },
}
canonical = lambda item: json.dumps(
    item, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("ascii")
value["receipt_digest"] = hashlib.sha256(canonical(value)).hexdigest()
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
fd = os.open(receipt_path, flags, 0o444)
with os.fdopen(fd, "wb") as handle:
    handle.write(canonical(value) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())
    os.fchmod(handle.fileno(), 0o444)
PY
  [[ -f "${partial_receipt}" && ! -L "${partial_receipt}" && "$(stat -Lc '%a' -- "${partial_receipt}")" == 444 ]] || \
    fail "partial exact30 receipt differs"
  echo "SAIC_PURE_T2V_EVENT_BANK_TOPUP_${group_select}_V3_PARTIAL_EXACT30_OK training=false merge_required=true"
fi

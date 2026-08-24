#!/usr/bin/env bash
#SBATCH --job-name=starc-hid-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=06:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Frozen STARC hidden materialization only.  The two independent Ulysses-SP4
# groups use all eight allocated MI210s concurrently.  This job performs no
# critic/editor optimizer step and persists only the 52 fixed-sketch block-15
# action-minus-noop residuals plus fail-closed receipts.

source_archive="${STARC_V1_SOURCE_ARCHIVE:?set STARC_V1_SOURCE_ARCHIVE}"
source_archive_sha256="${STARC_V1_SOURCE_ARCHIVE_SHA256:?set STARC_V1_SOURCE_ARCHIVE_SHA256}"
source_revision="${STARC_V1_SOURCE_REVISION:?set STARC_V1_SOURCE_REVISION}"
formal_d541801_archive="${STARC_V1_D541801_SOURCE_ARCHIVE:?set STARC_V1_D541801_SOURCE_ARCHIVE}"
root_spec="${STARC_V1_ROOT_SPEC:?set STARC_V1_ROOT_SPEC}"
bank_output_dir="${STARC_V1_BANK_OUTPUT_DIR:?set STARC_V1_BANK_OUTPUT_DIR}"
bank_receipt="${STARC_V1_BANK_RECEIPT:?set STARC_V1_BANK_RECEIPT}"
detached_labels="${STARC_V1_DETACHED_LABEL_MANIFEST:?set STARC_V1_DETACHED_LABEL_MANIFEST}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_root="${STARC_V1_OUTPUT_ROOT:?set STARC_V1_OUTPUT_ROOT}"
python_bin="${STARC_V1_PYTHON_BIN:?set STARC_V1_PYTHON_BIN}"

readonly required_root_spec_sha256=a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95
readonly required_bank_receipt_sha256=8c4f77bdd24fa14786f3dff28a4044d819f444c0338484a2fa6df9588100cb59
readonly required_detached_labels_sha256=9246504e97e1ee46c2cdcf7dfac0f41364dca40f26e5c26f28f0968d0443808d
readonly required_sidecar_sha256=a71854673f64e027bd673cf4c74673bcd7de74dca6f5b7b3b2c429467055f215
readonly required_authority_sha256=c24e0193b29c7a8fa05cf9a25035ac01816fe54f3b80820b8e8de47418b90457
readonly required_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly required_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly required_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly required_d541801_revision=d541801a162796aacde34c2bfc2b1f0472d954d2
readonly required_d541801_archive_sha256=535aba9b5445e2a9b06cf3da267325c49d247ab2ef9f4a9dd129a51fdbb008c7
readonly required_d541801_scorer_sha256=3d7ce459ddb9a014873acd6384c7c4030b4e3aca9004c1b8486ebbc1f0f5d32e

fail() { echo "[starc-hidden-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be lowercase SHA-1"; }

for name in source_archive formal_d541801_archive root_spec bank_output_dir bank_receipt detached_labels bernini_root veomni_root checkpoint checkpoint_manifest output_root python_bin; do
  require_absolute "${!name}" "${name}"
done
require_sha256 "${source_archive_sha256}" source_archive_sha256
require_sha1 "${source_revision}" source_revision

source_archive="$(realpath -e -- "${source_archive}")"
formal_d541801_archive="$(realpath -e -- "${formal_d541801_archive}")"
root_spec="$(realpath -e -- "${root_spec}")"
bank_output_dir="$(realpath -e -- "${bank_output_dir}")"
bank_receipt="$(realpath -e -- "${bank_receipt}")"
detached_labels="$(realpath -e -- "${detached_labels}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

for plain_file in "${source_archive}" "${formal_d541801_archive}" "${root_spec}" "${bank_receipt}" "${detached_labels}" "${checkpoint_manifest}"; do
  [[ -f "${plain_file}" && ! -L "${plain_file}" ]] || fail "plain input file differs: ${plain_file}"
done
for plain_dir in "${bank_output_dir}" "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${plain_dir}" && ! -L "${plain_dir}" ]] || fail "plain input directory differs: ${plain_dir}"
done
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${formal_d541801_archive}" | awk '{print $1}')" == "${required_d541801_archive_sha256}" ]] || fail "formal d541801 source archive changed"
[[ "$(git get-tar-commit-id <"${formal_d541801_archive}")" == "${required_d541801_revision}" ]] || fail "formal d541801 source revision differs"
[[ "$(sha256sum "${root_spec}" | awk '{print $1}')" == "${required_root_spec_sha256}" ]] || fail "root spec authority differs"
[[ "$(sha256sum "${bank_receipt}" | awk '{print $1}')" == "${required_bank_receipt_sha256}" ]] || fail "bank receipt authority differs"
[[ "$(sha256sum "${detached_labels}" | awk '{print $1}')" == "${required_detached_labels_sha256}" ]] || fail "detached labels authority differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${required_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest authority differs"

[[ "${output_root}" != / && "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output root must be canonical non-root"
[[ "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
mkdir -p -- "$(dirname -- "${output_root}")"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/starc-hidden-v1-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""
sp4_b_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/starc-hidden-v1-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[starc-hidden-v1] PASS all8=true topology=2xSP4 cells=4 arms=52 forwards=104 geometries=P930x2,P928x1,P918x1 optimizer=false output=${output_root}"
  else
    echo "[starc-hidden-v1] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true; done
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/method-source.tar"
formal_archive_copy="${task_scratch}/formal-d541801-source.tar"
cp -- "${source_archive}" "${archive_copy}"
cp -- "${formal_d541801_archive}" "${formal_archive_copy}"
chmod 0444 -- "${archive_copy}"
chmod 0444 -- "${formal_archive_copy}"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile

required = {
    "md/action_editing/bernini_starc_core4_critic_use_authority_20260808.md",
    "methods/bernini_action_editing/assets/starc_core4_critic_only_use_v1.json",
    "methods/bernini_action_editing/materialize_starc_core4_hidden_v1.py",
    "methods/bernini_action_editing/internal_temporal_quotient_observer.py",
    "methods/bernini_action_editing/latent_temporal_event_critic_dataset.py",
    "methods/bernini_action_editing/temporal_counterfactual_action_scorer_v1.py",
    "methods/bernini_action_editing/temporal_counterfactual_contract_v1.py",
    "methods/bernini_action_editing/tools/author_pair_v5_core4_event_labels_d541801_v3.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        normalized = path.as_posix().lstrip("./")
        selected = (
            normalized.startswith("methods/bernini_action_editing/")
            or normalized
            == "md/action_editing/bernini_starc_core4_critic_use_authority_20260808.md"
        )
        if not selected:
            continue
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("source archive contains an unsafe member")
        seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks STARC materializer closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree" \
  methods/bernini_action_editing \
  md/action_editing/bernini_starc_core4_critic_use_authority_20260808.md
tar --no-same-owner --no-same-permissions -xf "${formal_archive_copy}" -C "${task_scratch}/source-tree" \
  methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py
source_tree="${task_scratch}/source-tree"
method_root="${source_tree}/methods/bernini_action_editing"
materializer="${method_root}/materialize_starc_core4_hidden_v1.py"
sidecar="${method_root}/assets/starc_core4_critic_only_use_v1.json"
authority="${source_tree}/md/action_editing/bernini_starc_core4_critic_use_authority_20260808.md"
temporal_scorer="${method_root}/temporal_counterfactual_action_scorer_v1.py"
temporal_contract="${method_root}/temporal_counterfactual_contract_v1.py"
fitq_observer="${method_root}/internal_temporal_quotient_observer.py"
dataset_contract="${method_root}/latent_temporal_event_critic_dataset.py"
label_author="${method_root}/tools/author_pair_v5_core4_event_labels_d541801_v3.py"
formal_d541801_scorer="${method_root}/score_pair_v5_t2v_energy_bank_v3.py"
launcher="${method_root}/scripts/auh_materialize_starc_core4_hidden_v1_dual4.sbatch"
for pinned_file in "${materializer}" "${sidecar}" "${authority}" "${formal_d541801_scorer}" "${temporal_scorer}" "${temporal_contract}" "${fitq_observer}" "${dataset_contract}" "${label_author}" "${launcher}"; do
  [[ -f "${pinned_file}" && ! -L "${pinned_file}" ]] || fail "extracted source closure differs: ${pinned_file}"
done
[[ "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" == "$(sha256sum "${launcher}" | awk '{print $1}')" ]] || fail "running launcher differs from source archive"
[[ "$(sha256sum "${sidecar}" | awk '{print $1}')" == "${required_sidecar_sha256}" ]] || fail "critic-only sidecar differs"
[[ "$(sha256sum "${authority}" | awk '{print $1}')" == "${required_authority_sha256}" ]] || fail "critic authority differs"
[[ "$(sha256sum "${formal_d541801_scorer}" | awk '{print $1}')" == "${required_d541801_scorer_sha256}" ]] || fail "formal d541801 scorer differs"
find "${source_tree}" -type f -exec chmod a-w -- {} +

materializer_sha256="$(sha256sum "${materializer}" | awk '{print $1}')"
temporal_scorer_sha256="$(sha256sum "${temporal_scorer}" | awk '{print $1}')"
temporal_contract_sha256="$(sha256sum "${temporal_contract}" | awk '{print $1}')"
fitq_observer_sha256="$(sha256sum "${fitq_observer}" | awk '{print $1}')"
dataset_contract_sha256="$(sha256sum "${dataset_contract}" | awk '{print $1}')"
label_author_sha256="$(sha256sum "${label_author}" | awk '{print $1}')"

mkdir -- "${output_root}"
mkdir -- "${output_root}/logs"
echo "[starc-hidden-v1] allocation=8xMI210 topology=sp4-a:0,1,2,3+sp4-b:4,5,6,7 exact81=true block=15 schedule_index=33"
echo "[starc-hidden-v1] generated_t2v_use=critic_hidden_query_only editor_target=false editor_condition=false mask=false flow=false pose=false track=false"

common_args=(
  materialize-group
  --root-spec "${root_spec}"
  --expected-root-spec-sha256 "${required_root_spec_sha256}"
  --bank-output-dir "${bank_output_dir}"
  --bank-receipt "${bank_receipt}"
  --expected-bank-receipt-sha256 "${required_bank_receipt_sha256}"
  --detached-label-manifest "${detached_labels}"
  --expected-detached-label-manifest-sha256 "${required_detached_labels_sha256}"
  --critic-use-sidecar "${sidecar}"
  --expected-critic-use-sidecar-sha256 "${required_sidecar_sha256}"
  --critic-use-authority-evidence "${authority}"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --output-root "${output_root}"
  --expected-bernini-commit "${required_bernini_commit}"
  --expected-veomni-commit "${required_veomni_commit}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
  --formal-d541801-source-revision "${required_d541801_revision}"
  --formal-d541801-source-archive-sha256 "${required_d541801_archive_sha256}"
  --expected-materializer-source-sha256 "${materializer_sha256}"
  --expected-formal-d541801-scorer-source-sha256 "${required_d541801_scorer_sha256}"
  --expected-temporal-scorer-source-sha256 "${temporal_scorer_sha256}"
  --expected-temporal-contract-source-sha256 "${temporal_contract_sha256}"
  --expected-fitq-observer-source-sha256 "${fitq_observer_sha256}"
  --expected-dataset-contract-source-sha256 "${dataset_contract_sha256}"
  --expected-label-author-source-sha256 "${label_author_sha256}"
  --ack-generated-t2v-hidden-critic-only
  --ack-no-generated-media-editor-use
  --ack-no-optimizer-or-editor-update
)

job_mod=$((SLURM_JOB_ID % 10000))
sp4_a_port=$((30000 + 2 * job_mod))
sp4_b_port=$((sp4_a_port + 1))

run_group() (
  set -Eeuo pipefail
  local group_id="$1" visible_gpus="$2" master_port="$3"
  local group_scratch="${task_scratch}/groups/${group_id}"
  mkdir -p -- "${group_scratch}/cache/miopen-user" "${group_scratch}/cache/miopen-custom" \
    "${group_scratch}/cache/torch-extensions" "${group_scratch}/cache/triton" \
    "${group_scratch}/cache/xdg" "${group_scratch}/cache/pycache" "${group_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_scratch}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${group_scratch}/cache/triton"
  export XDG_CACHE_HOME="${group_scratch}/cache/xdg"
  export PYTHONPYCACHEPREFIX="${group_scratch}/cache/pycache"
  export TMPDIR="${group_scratch}/tmp"
  export TORCHELASTIC_ERROR_FILE="${group_scratch}/torch-elastic-error.json"
  echo "[starc-hidden-v1:${group_id}] START visible=${visible_gpus} world=4"
  PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${master_port}" \
    "${materializer}" "${common_args[@]}" --group-id "${group_id}"
)

run_group sp4-a "0,1,2,3" "${sp4_a_port}" >"${output_root}/logs/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_group sp4-b "4,5,6,7" "${sp4_b_port}" >"${output_root}/logs/sp4-b.log" 2>&1 & sp4_b_pid=$!
sp4_a_status=0
sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" != 0 || "${sp4_b_status}" != 0 ]]; then
  tail -n 240 -- "${output_root}/logs/sp4-a.log" >&2 || true
  tail -n 240 -- "${output_root}/logs/sp4-b.log" >&2 || true
  fail "one or more STARC SP4 groups failed"
fi

sp4_a_manifest="${output_root}/sp4-a/starc-core4-hidden-group-sp4-a-v1.json"
sp4_b_manifest="${output_root}/sp4-b/starc-core4-hidden-group-sp4-b-v1.json"
[[ -f "${sp4_a_manifest}" && ! -L "${sp4_a_manifest}" ]] || fail "sp4-a manifest missing"
[[ -f "${sp4_b_manifest}" && ! -L "${sp4_b_manifest}" ]] || fail "sp4-b manifest missing"
sp4_a_manifest_sha256="$(sha256sum "${sp4_a_manifest}" | awk '{print $1}')"
sp4_b_manifest_sha256="$(sha256sum "${sp4_b_manifest}" | awk '{print $1}')"
PYTHONPATH="${method_root}" "${python_bin}" -B "${materializer}" aggregate-master \
  --output-root "${output_root}" \
  --sp4-a-group-manifest "${sp4_a_manifest}" \
  --expected-sp4-a-group-manifest-sha256 "${sp4_a_manifest_sha256}" \
  --sp4-b-group-manifest "${sp4_b_manifest}" \
  --expected-sp4-b-group-manifest-sha256 "${sp4_b_manifest_sha256}"

master_manifest="${output_root}/starc-core4-hidden-master-v1.json"
[[ -f "${master_manifest}" && ! -L "${master_manifest}" ]] || fail "master manifest missing"
[[ "$(find "${output_root}" -type f -name starc-block15-hidden-residual.safetensors | wc -l)" == 52 ]] || fail "residual artifact count differs"
[[ "$(find "${output_root}" -type f -name starc-block15-hidden-arm-receipt-v1.json | wc -l)" == 52 ]] || fail "arm receipt count differs"
"${python_bin}" -B - "${master_manifest}" <<'PY'
import json, sys

with open(sys.argv[1], "r", encoding="ascii") as handle:
    receipt = json.load(handle)
if receipt.get("episode_count") != 4 or receipt.get("arm_count") != 52:
    raise SystemExit("master population count differs")
if receipt.get("model_forward_count") != 104 or receipt.get("training_performed") is not False:
    raise SystemExit("master execution authority differs")
sketches = receipt.get("spatial_sketch_bindings_by_episode", {})
positions = sorted(row.get("patch_positions") for row in sketches.values())
if positions != [918, 928, 930, 930]:
    raise SystemExit("master native geometry population differs")
if receipt.get("editor_optimizer_authorized") is not False or receipt.get("scientific_critic_claim_authorized") is not False:
    raise SystemExit("master exceeded critic-only authority")
print("STARC_CORE4_HIDDEN_MASTER_OK episodes=4 arms=52 forwards=104 P=930x2,928x1,918x1")
PY

#!/usr/bin/env bash
#SBATCH --job-name=ltec-core4-hidden
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

# Frozen-hidden *materialization* only.  Two concurrent Ulysses-4 processes
# cover all four core4 cells.  This launcher never invokes the critic trainer,
# never consumes confirmation examples in an optimizer, and cannot authorize
# an RV2V/editor update.  A successful population receipt permits only the
# preregistered two-fit/two-confirmation pilot.  Native H/W is authenticated
# and handled per episode (P=930/928/918); no rank may resize/crop or reuse a
# sketch/layout from another episode.  Passing the pilot can at most
# motivate fixed top-up generation.

d541_archive="${LTEC_CORE4_D541801_SOURCE_ARCHIVE:?set LTEC_CORE4_D541801_SOURCE_ARCHIVE}"
d541_archive_sha256="${LTEC_CORE4_D541801_SOURCE_ARCHIVE_SHA256:?set LTEC_CORE4_D541801_SOURCE_ARCHIVE_SHA256}"
overlay_archive="${LTEC_CORE4_OVERLAY_SOURCE_ARCHIVE:?set LTEC_CORE4_OVERLAY_SOURCE_ARCHIVE}"
overlay_archive_sha256="${LTEC_CORE4_OVERLAY_SOURCE_ARCHIVE_SHA256:?set LTEC_CORE4_OVERLAY_SOURCE_ARCHIVE_SHA256}"
overlay_revision="${LTEC_CORE4_OVERLAY_SOURCE_REVISION:?set LTEC_CORE4_OVERLAY_SOURCE_REVISION}"
materializer_source_sha256="${LTEC_CORE4_MATERIALIZER_SOURCE_SHA256:?set LTEC_CORE4_MATERIALIZER_SOURCE_SHA256}"
observer_source_sha256="${LTEC_CORE4_OBSERVER_SOURCE_SHA256:?set LTEC_CORE4_OBSERVER_SOURCE_SHA256}"
critic_source_sha256="${LTEC_CORE4_CRITIC_SOURCE_SHA256:?set LTEC_CORE4_CRITIC_SOURCE_SHA256}"
dataset_source_sha256="${LTEC_CORE4_DATASET_SOURCE_SHA256:?set LTEC_CORE4_DATASET_SOURCE_SHA256}"
prompt_pair_scorer_source_sha256="${LTEC_CORE4_PROMPT_PAIR_SCORER_SOURCE_SHA256:?set LTEC_CORE4_PROMPT_PAIR_SCORER_SOURCE_SHA256}"
temporal_contract_source_sha256="${LTEC_CORE4_TEMPORAL_CONTRACT_SOURCE_SHA256:?set LTEC_CORE4_TEMPORAL_CONTRACT_SOURCE_SHA256}"
label_author_source_sha256="${LTEC_CORE4_LABEL_AUTHOR_SOURCE_SHA256:?set LTEC_CORE4_LABEL_AUTHOR_SOURCE_SHA256}"
bank_spec="${LTEC_CORE4_BANK_SPEC:?set LTEC_CORE4_BANK_SPEC}"
bank_spec_sha256="${LTEC_CORE4_BANK_SPEC_SHA256:?set LTEC_CORE4_BANK_SPEC_SHA256}"
bank_output="${LTEC_CORE4_BANK_OUTPUT_DIR:?set LTEC_CORE4_BANK_OUTPUT_DIR}"
bank_receipt="${LTEC_CORE4_BANK_RECEIPT:?set LTEC_CORE4_BANK_RECEIPT}"
bank_receipt_sha256="${LTEC_CORE4_BANK_RECEIPT_SHA256:?set LTEC_CORE4_BANK_RECEIPT_SHA256}"
event_labels="${LTEC_CORE4_EVENT_LABEL_MANIFEST:?set LTEC_CORE4_EVENT_LABEL_MANIFEST}"
event_labels_sha256="${LTEC_CORE4_EVENT_LABEL_MANIFEST_SHA256:?set LTEC_CORE4_EVENT_LABEL_MANIFEST_SHA256}"
critic_use_authority="${LTEC_CORE4_CRITIC_USE_AUTHORITY:?set LTEC_CORE4_CRITIC_USE_AUTHORITY}"
critic_use_authority_sha256="${LTEC_CORE4_CRITIC_USE_AUTHORITY_SHA256:?set LTEC_CORE4_CRITIC_USE_AUTHORITY_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
expected_bernini_commit="${LTEC_CORE4_EXPECTED_BERNINI_COMMIT:?set LTEC_CORE4_EXPECTED_BERNINI_COMMIT}"
expected_veomni_commit="${LTEC_CORE4_EXPECTED_VEOMNI_COMMIT:?set LTEC_CORE4_EXPECTED_VEOMNI_COMMIT}"
output_dir="${LTEC_CORE4_OUTPUT_DIR:?set LTEC_CORE4_OUTPUT_DIR}"
python_bin="${LTEC_CORE4_PYTHON_BIN:?set LTEC_CORE4_PYTHON_BIN}"

readonly required_d541_revision=d541801a162796aacde34c2bfc2b1f0472d954d2
readonly required_d541_archive_sha256=535aba9b5445e2a9b06cf3da267325c49d247ab2ef9f4a9dd129a51fdbb008c7
readonly required_d541_scorer_sha256=3d7ce459ddb9a014873acd6384c7c4030b4e3aca9004c1b8486ebbc1f0f5d32e
readonly required_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly required_bernini_revision=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly required_veomni_revision=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly required_bank_spec_sha256=a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95
readonly required_bank_receipt_sha256=8c4f77bdd24fa14786f3dff28a4044d819f444c0338484a2fa6df9588100cb59
readonly required_event_labels_sha256=9246504e97e1ee46c2cdcf7dfac0f41364dca40f26e5c26f28f0968d0443808d

fail() { echo "[ltec-core4-hidden] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be lowercase SHA-1"; }

for name in d541_archive overlay_archive bank_spec bank_output bank_receipt event_labels critic_use_authority bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
for name in d541_archive_sha256 overlay_archive_sha256 materializer_source_sha256 observer_source_sha256 critic_source_sha256 dataset_source_sha256 prompt_pair_scorer_source_sha256 temporal_contract_source_sha256 label_author_source_sha256 bank_spec_sha256 bank_receipt_sha256 event_labels_sha256 critic_use_authority_sha256 checkpoint_manifest_sha256; do
  require_sha256 "${!name}" "${name}"
done
for name in overlay_revision expected_bernini_commit expected_veomni_commit; do
  require_sha1 "${!name}" "${name}"
done
[[ "${d541_archive_sha256}" == "${required_d541_archive_sha256}" ]] || fail "formal d541801 archive hash differs"
[[ "${checkpoint_manifest_sha256}" == "${required_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest authority differs"
[[ "${expected_bernini_commit}" == "${required_bernini_revision}" ]] || fail "Bernini revision authority differs"
[[ "${expected_veomni_commit}" == "${required_veomni_revision}" ]] || fail "VeOmni revision authority differs"
[[ "${bank_spec_sha256}" == "${required_bank_spec_sha256}" ]] || fail "core4-v2 spec authority differs"
[[ "${bank_receipt_sha256}" == "${required_bank_receipt_sha256}" ]] || fail "core4-v2 bank authority differs"
[[ "${event_labels_sha256}" == "${required_event_labels_sha256}" ]] || fail "reviewed label authority differs"

d541_archive="$(realpath -e -- "${d541_archive}")"
overlay_archive="$(realpath -e -- "${overlay_archive}")"
bank_spec="$(realpath -e -- "${bank_spec}")"
bank_output="$(realpath -e -- "${bank_output}")"
bank_receipt="$(realpath -e -- "${bank_receipt}")"
event_labels="$(realpath -e -- "${event_labels}")"
critic_use_authority="$(realpath -e -- "${critic_use_authority}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

for archive in "${d541_archive}" "${overlay_archive}"; do
  [[ -f "${archive}" && ! -L "${archive}" ]] || fail "source archive differs: ${archive}"
done
for file in "${bank_spec}" "${bank_receipt}" "${event_labels}" "${critic_use_authority}" "${checkpoint_manifest}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "plain input file differs: ${file}"
done
for directory in "${bank_output}" "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${directory}" && ! -L "${directory}" ]] || fail "plain input directory differs: ${directory}"
done
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
[[ "$(sha256sum "${d541_archive}" | awk '{print $1}')" == "${d541_archive_sha256}" ]] || fail "d541801 archive changed"
[[ "$(git get-tar-commit-id <"${d541_archive}")" == "${required_d541_revision}" ]] || fail "d541801 archive revision differs"
[[ "$(sha256sum "${overlay_archive}" | awk '{print $1}')" == "${overlay_archive_sha256}" ]] || fail "overlay archive changed"
[[ "$(git get-tar-commit-id <"${overlay_archive}")" == "${overlay_revision}" ]] || fail "overlay archive revision differs"
[[ "$(sha256sum "${bank_spec}" | awk '{print $1}')" == "${bank_spec_sha256}" ]] || fail "bank spec changed"
[[ "$(sha256sum "${bank_receipt}" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "bank receipt changed"
[[ "$(sha256sum "${event_labels}" | awk '{print $1}')" == "${event_labels_sha256}" ]] || fail "event labels changed"
[[ "$(sha256sum "${critic_use_authority}" | awk '{print $1}')" == "${critic_use_authority_sha256}" ]] || fail "critic-use authority changed"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest changed"
[[ "${output_dir}" != / && "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output path differs"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "output must be fresh"
mkdir -p -- "$(dirname -- "${output_dir}")"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/ltec-core4-hidden-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""; sp4_b_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/ltec-core4-hidden-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[ltec-core4-hidden] PASS topology=2xSP4 cells=4 arms=52 main_forwards=104 parity_forwards=4 training=false editor_authority=false output=${output_dir}"
  else
    echo "[ltec-core4-hidden] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
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
"${python_bin}" -B - "${d541_archive}" "${overlay_archive}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile

required_d541 = {
    "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py",
    "methods/bernini_action_editing/pair_v5_native_bridge.py",
    "methods/bernini_action_editing/dclr_runtime_contract.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
}
required_overlay = {
    "methods/bernini_action_editing/materialize_latent_temporal_event_critic_core4.py",
    "methods/bernini_action_editing/internal_temporal_quotient_observer.py",
    "methods/bernini_action_editing/latent_temporal_event_critic.py",
    "methods/bernini_action_editing/latent_temporal_event_critic_dataset.py",
    "methods/bernini_action_editing/temporal_counterfactual_action_scorer_v1.py",
    "methods/bernini_action_editing/temporal_counterfactual_contract_v1.py",
    "methods/bernini_action_editing/tools/author_pair_v5_core4_event_labels_d541801_v3.py",
}
selected_prefix = "methods/bernini_action_editing/"
for archive_index, (archive, required) in enumerate(
    ((sys.argv[1], required_d541), (sys.argv[2], required_overlay))
):
    seen = set()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            normalized = path.as_posix().lstrip("./")
            selected = normalized.startswith(selected_prefix) if archive_index == 0 else normalized in required_overlay
            if not selected:
                continue
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise SystemExit("source archive contains an unsafe member")
            seen.add(normalized)
    missing = sorted(required - seen)
    if missing:
        raise SystemExit(f"source archive lacks required closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${d541_archive}" -C "${task_scratch}/source-tree" \
  methods/bernini_action_editing
tar --no-same-owner --no-same-permissions -xf "${overlay_archive}" -C "${task_scratch}/source-tree" \
  methods/bernini_action_editing/materialize_latent_temporal_event_critic_core4.py \
  methods/bernini_action_editing/internal_temporal_quotient_observer.py \
  methods/bernini_action_editing/latent_temporal_event_critic.py \
  methods/bernini_action_editing/latent_temporal_event_critic_dataset.py \
  methods/bernini_action_editing/temporal_counterfactual_action_scorer_v1.py \
  methods/bernini_action_editing/temporal_counterfactual_contract_v1.py \
  methods/bernini_action_editing/tools/author_pair_v5_core4_event_labels_d541801_v3.py
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
materializer="${method_root}/materialize_latent_temporal_event_critic_core4.py"
observer="${method_root}/internal_temporal_quotient_observer.py"
critic="${method_root}/latent_temporal_event_critic.py"
dataset="${method_root}/latent_temporal_event_critic_dataset.py"
prompt_pair_scorer="${method_root}/temporal_counterfactual_action_scorer_v1.py"
temporal_contract="${method_root}/temporal_counterfactual_contract_v1.py"
label_author="${method_root}/tools/author_pair_v5_core4_event_labels_d541801_v3.py"
formal_scorer="${method_root}/score_pair_v5_t2v_energy_bank_v3.py"
[[ "$(sha256sum "${formal_scorer}" | awk '{print $1}')" == "${required_d541_scorer_sha256}" ]] || fail "formal d541801 scorer bytes differ"
for entry in \
  "${materializer}:${materializer_source_sha256}" \
  "${observer}:${observer_source_sha256}" \
  "${critic}:${critic_source_sha256}" \
  "${dataset}:${dataset_source_sha256}" \
  "${prompt_pair_scorer}:${prompt_pair_scorer_source_sha256}" \
  "${temporal_contract}:${temporal_contract_source_sha256}" \
  "${label_author}:${label_author_source_sha256}"; do
  path="${entry%%:*}"; expected="${entry##*:}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || fail "overlay source bytes differ: ${path}"
done
find "${method_root}" -type f -exec chmod a-w -- {} +
mkdir -- "${output_dir}"

common_args=(
  --root-spec "${bank_spec}" --expected-root-spec-sha256 "${bank_spec_sha256}"
  --bank-receipt "${bank_receipt}" --expected-bank-receipt-sha256 "${bank_receipt_sha256}"
  --detached-event-label-manifest "${event_labels}"
  --expected-detached-event-label-manifest-sha256 "${event_labels_sha256}"
  --critic-use-authority "${critic_use_authority}"
  --expected-critic-use-authority-sha256 "${critic_use_authority_sha256}"
  --method-source-revision "${overlay_revision}"
  --method-source-archive-sha256 "${overlay_archive_sha256}"
  --expected-materializer-source-sha256 "${materializer_source_sha256}"
  --expected-observer-source-sha256 "${observer_source_sha256}"
  --expected-critic-source-sha256 "${critic_source_sha256}"
  --expected-dataset-source-sha256 "${dataset_source_sha256}"
  --expected-prompt-pair-scorer-source-sha256 "${prompt_pair_scorer_source_sha256}"
  --expected-temporal-contract-source-sha256 "${temporal_contract_source_sha256}"
  --expected-label-author-source-sha256 "${label_author_source_sha256}"
  --ack-core4-pilot-only-never-editor-target-or-authority
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
  "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port="${master_port}" \
    "${materializer}" group \
    "${common_args[@]}" \
    --bank-output-dir "${bank_output}" --group-id "${group_id}" \
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
    --expected-bernini-commit "${expected_bernini_commit}" \
    --expected-veomni-commit "${expected_veomni_commit}" \
    --output-dir "${output_dir}/${group_id}"
)

run_group sp4-a "0,1,2,3" "${sp4_a_port}" >"${output_dir}/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_group sp4-b "4,5,6,7" "${sp4_b_port}" >"${output_dir}/sp4-b.log" 2>&1 & sp4_b_pid=$!
sp4_a_status=0; sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" != 0 || "${sp4_b_status}" != 0 ]]; then
  tail -n 200 -- "${output_dir}/sp4-a.log" >&2 || true
  tail -n 200 -- "${output_dir}/sp4-b.log" >&2 || true
  fail "one or more hidden-query SP4 groups failed"
fi

PYTHONPATH="${method_root}" "${python_bin}" -B "${materializer}" merge \
  "${common_args[@]}" \
  --output-root "${output_dir}"

[[ -f "${output_dir}/latent-temporal-event-hidden-population-v1.json" ]] || fail "population receipt missing"
[[ "$(find "${output_dir}" -type f -name sketched-action-minus-noop.safetensors | wc -l)" == 52 ]] || fail "residual artifact count differs"
[[ "$(find "${output_dir}" -type f -name hidden-query-arm-v1.json | wc -l)" == 52 ]] || fail "arm receipt count differs"
[[ "$(find "${output_dir}" -type f -name episode-plan-v1.json | wc -l)" == 4 ]] || fail "episode plan count differs"
"${python_bin}" -B - "${output_dir}/${POPULATION_FILENAME:-latent-temporal-event-hidden-population-v1.json}" <<'PY'
import json, sys

with open(sys.argv[1], "r", encoding="ascii") as handle:
    receipt = json.load(handle)
geometries = receipt.get("native_geometries_by_episode", {})
sketches = receipt.get("spatial_sketches_by_episode", {})
if list(geometries) != receipt.get("episode_order") or list(sketches) != receipt.get("episode_order"):
    raise SystemExit("population per-episode geometry/sketch order differs")
observed = sorted(
    (
        tuple(binding.get("latent_shape", ())),
        tuple(binding.get("patch_grid_height_width", ())),
        binding.get("patch_positions"),
        sketches[episode].get("tensor_digest"),
    )
    for episode, binding in geometries.items()
)
expected_geometry_counts = {
    ((1, 16, 21, 60, 62), (30, 31), 930): 2,
    ((1, 16, 21, 64, 58), (32, 29), 928): 1,
    ((1, 16, 21, 68, 54), (34, 27), 918): 1,
}
actual_counts = {}
for shape, grid, positions, digest in observed:
    key = (shape, grid, positions)
    actual_counts[key] = actual_counts.get(key, 0) + 1
    if not isinstance(digest, str) or len(digest) != 64:
        raise SystemExit("population geometry-specific sketch digest differs")
if actual_counts != expected_geometry_counts or len({row[3] for row in observed}) != 3:
    raise SystemExit("population native geometry/dynamic-P sketch closure differs")
print("LTEC_CORE4_NATIVE_GEOMETRY_OK P=930x2,928x1,918x1 distinct_sketch_digests=3")
PY

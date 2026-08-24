#!/usr/bin/env bash
#SBATCH --job-name=temporal-cf-v1
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

# Two concurrent Ulysses-4 scorers.  A sealed d541801 source archive supplies
# the exact frozen bank/model plumbing; a separately sealed overlay archive
# supplies only the uniquely named temporal-counterfactual scorer and contract.
# This prevents a later generic/v4 scorer from silently replacing job 131177's
# authority while still allowing the new method to receive its own revision.

d541_archive="${TEMPORAL_CF_D541801_SOURCE_ARCHIVE:?set TEMPORAL_CF_D541801_SOURCE_ARCHIVE}"
d541_archive_sha256="${TEMPORAL_CF_D541801_SOURCE_ARCHIVE_SHA256:?set TEMPORAL_CF_D541801_SOURCE_ARCHIVE_SHA256}"
overlay_archive="${TEMPORAL_CF_OVERLAY_SOURCE_ARCHIVE:?set TEMPORAL_CF_OVERLAY_SOURCE_ARCHIVE}"
overlay_archive_sha256="${TEMPORAL_CF_OVERLAY_SOURCE_ARCHIVE_SHA256:?set TEMPORAL_CF_OVERLAY_SOURCE_ARCHIVE_SHA256}"
overlay_revision="${TEMPORAL_CF_OVERLAY_SOURCE_REVISION:?set TEMPORAL_CF_OVERLAY_SOURCE_REVISION}"
scorer_source_sha256="${TEMPORAL_CF_SCORER_SOURCE_SHA256:?set TEMPORAL_CF_SCORER_SOURCE_SHA256}"
contract_source_sha256="${TEMPORAL_CF_CONTRACT_SOURCE_SHA256:?set TEMPORAL_CF_CONTRACT_SOURCE_SHA256}"
bank_spec="${TEMPORAL_CF_BANK_SPEC:?set TEMPORAL_CF_BANK_SPEC}"
bank_spec_sha256="${TEMPORAL_CF_BANK_SPEC_SHA256:?set TEMPORAL_CF_BANK_SPEC_SHA256}"
bank_output="${TEMPORAL_CF_BANK_OUTPUT_DIR:?set TEMPORAL_CF_BANK_OUTPUT_DIR}"
bank_receipt="${TEMPORAL_CF_BANK_RECEIPT:?set TEMPORAL_CF_BANK_RECEIPT}"
bank_receipt_sha256="${TEMPORAL_CF_BANK_RECEIPT_SHA256:?set TEMPORAL_CF_BANK_RECEIPT_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
expected_bernini_commit="${TEMPORAL_CF_EXPECTED_BERNINI_COMMIT:?set TEMPORAL_CF_EXPECTED_BERNINI_COMMIT}"
expected_veomni_commit="${TEMPORAL_CF_EXPECTED_VEOMNI_COMMIT:?set TEMPORAL_CF_EXPECTED_VEOMNI_COMMIT}"
output_dir="${TEMPORAL_CF_OUTPUT_DIR:?set TEMPORAL_CF_OUTPUT_DIR}"
python_bin="${TEMPORAL_CF_PYTHON_BIN:?set TEMPORAL_CF_PYTHON_BIN}"

readonly required_d541_revision=d541801a162796aacde34c2bfc2b1f0472d954d2
readonly required_d541_archive_sha256=535aba9b5445e2a9b06cf3da267325c49d247ab2ef9f4a9dd129a51fdbb008c7
readonly required_d541_scorer_sha256=3d7ce459ddb9a014873acd6384c7c4030b4e3aca9004c1b8486ebbc1f0f5d32e
readonly required_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly required_bernini_revision=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly required_veomni_revision=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly required_bank_spec_sha256=a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95
readonly required_bank_receipt_sha256=8c4f77bdd24fa14786f3dff28a4044d819f444c0338484a2fa6df9588100cb59

fail() { echo "[temporal-cf-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be lowercase SHA-1"; }

for name in d541_archive overlay_archive bank_spec bank_output bank_receipt bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
for name in d541_archive_sha256 overlay_archive_sha256 scorer_source_sha256 contract_source_sha256 bank_spec_sha256 bank_receipt_sha256 checkpoint_manifest_sha256; do
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
[[ "${bank_receipt_sha256}" == "${required_bank_receipt_sha256}" ]] || fail "core4-v2 bank receipt authority differs"

d541_archive="$(realpath -e -- "${d541_archive}")"
overlay_archive="$(realpath -e -- "${overlay_archive}")"
bank_spec="$(realpath -e -- "${bank_spec}")"
bank_output="$(realpath -e -- "${bank_output}")"
bank_receipt="$(realpath -e -- "${bank_receipt}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

for archive in "${d541_archive}" "${overlay_archive}"; do
  [[ -f "${archive}" && ! -L "${archive}" ]] || fail "source archive differs: ${archive}"
done
[[ -f "${bank_spec}" && ! -L "${bank_spec}" ]] || fail "bank spec differs"
[[ -d "${bank_output}" && ! -L "${bank_output}" ]] || fail "bank output differs"
[[ -f "${bank_receipt}" && ! -L "${bank_receipt}" ]] || fail "bank receipt differs"
[[ "$(dirname -- "${bank_receipt}")" == "${bank_output}" ]] || fail "bank receipt outside bank output"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
[[ "$(sha256sum "${d541_archive}" | awk '{print $1}')" == "${d541_archive_sha256}" ]] || fail "d541801 archive changed"
[[ "$(git get-tar-commit-id <"${d541_archive}")" == "${required_d541_revision}" ]] || fail "d541801 archive revision differs"
[[ "$(sha256sum "${overlay_archive}" | awk '{print $1}')" == "${overlay_archive_sha256}" ]] || fail "overlay archive changed"
[[ "$(git get-tar-commit-id <"${overlay_archive}")" == "${overlay_revision}" ]] || fail "overlay archive revision differs"
[[ "$(sha256sum "${bank_spec}" | awk '{print $1}')" == "${bank_spec_sha256}" ]] || fail "bank spec hash differs"
[[ "$(sha256sum "${bank_receipt}" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "bank receipt hash differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"
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
task_scratch="$(mktemp -d "${scratch_parent%/}/temporal-cf-v1-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""; sp4_b_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/temporal-cf-v1-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[temporal-cf-v1] PASS topology=2xSP4 candidates=40 transforms=7 sigmas=3 prompts=2 t2v_calibration_only=true output=${output_dir}"
  else
    echo "[temporal-cf-v1] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
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
    "methods/bernini_action_editing/temporal_counterfactual_contract_v1.py",
    "methods/bernini_action_editing/temporal_counterfactual_action_scorer_v1.py",
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
            selected = (
                normalized.startswith(selected_prefix)
                if archive_index == 0
                else normalized in required_overlay
            )
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
  methods/bernini_action_editing/temporal_counterfactual_contract_v1.py \
  methods/bernini_action_editing/temporal_counterfactual_action_scorer_v1.py
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
formal_scorer="${method_root}/score_pair_v5_t2v_energy_bank_v3.py"
temporal_scorer="${method_root}/temporal_counterfactual_action_scorer_v1.py"
temporal_contract="${method_root}/temporal_counterfactual_contract_v1.py"
[[ "$(sha256sum "${formal_scorer}" | awk '{print $1}')" == "${required_d541_scorer_sha256}" ]] || fail "formal scorer bytes differ"
[[ "$(sha256sum "${temporal_scorer}" | awk '{print $1}')" == "${scorer_source_sha256}" ]] || fail "temporal scorer bytes differ"
[[ "$(sha256sum "${temporal_contract}" | awk '{print $1}')" == "${contract_source_sha256}" ]] || fail "temporal contract bytes differ"
find "${method_root}" -type f -exec chmod a-w -- {} +
mkdir -- "${output_dir}"

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
    "${temporal_scorer}" \
    --root-spec "${bank_spec}" --expected-root-spec-sha256 "${bank_spec_sha256}" \
    --bank-output-dir "${bank_output}" --bank-receipt "${bank_receipt}" \
    --expected-bank-receipt-sha256 "${bank_receipt_sha256}" --group-id "${group_id}" \
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
    --output-dir "${output_dir}/${group_id}" \
    --expected-bernini-commit "${expected_bernini_commit}" \
    --expected-veomni-commit "${expected_veomni_commit}" \
    --method-source-revision "${overlay_revision}" \
    --method-source-archive-sha256 "${overlay_archive_sha256}" \
    --expected-scorer-source-sha256 "${scorer_source_sha256}" \
    --expected-contract-source-sha256 "${contract_source_sha256}" \
    --ack-t2v-calibration-only-never-rv2v-input
)

run_group sp4-a "0,1,2,3" "${sp4_a_port}" >"${output_dir}/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_group sp4-b "4,5,6,7" "${sp4_b_port}" >"${output_dir}/sp4-b.log" 2>&1 & sp4_b_pid=$!
sp4_a_status=0; sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" != 0 || "${sp4_b_status}" != 0 ]]; then
  tail -n 160 -- "${output_dir}/sp4-a.log" >&2 || true
  tail -n 160 -- "${output_dir}/sp4-b.log" >&2 || true
  fail "one or more temporal-counterfactual SP4 groups failed"
fi

PYTHONPATH="${method_root}" "${python_bin}" -B - \
  "${bank_spec}" "${bank_spec_sha256}" "${output_dir}" \
  "${overlay_revision}" "${overlay_archive_sha256}" \
  "${scorer_source_sha256}" "${contract_source_sha256}" <<'PY'
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import temporal_counterfactual_contract_v1 as contract

spec_path, expected_spec_sha, output_path = sys.argv[1:4]
revision, archive_sha, scorer_sha, contract_sha = sys.argv[4:8]
raw = Path(spec_path).read_bytes()
if hashlib.sha256(raw).hexdigest() != expected_spec_sha:
    raise SystemExit("postflight spec hash differs")
spec = json.loads(raw)
output = Path(output_path)
for group in spec["groups"]:
    group_id = group["group_id"]
    expected_ids = [row["candidate_id"] for row in group["candidates"]]
    root = output / group_id
    group_name = f"temporal-counterfactual-action-score-{group_id}-v1.json"
    if {path.name for path in root.iterdir()} != set(expected_ids) | {group_name}:
        raise SystemExit(f"{group_id} output filesystem closure differs")
    receipts = []
    for candidate_id in expected_ids:
        candidate_root = root / candidate_id
        if {path.name for path in candidate_root.iterdir()} != {"temporal-counterfactual-action-score-v1.json"}:
            raise SystemExit(f"{candidate_id} output closure differs")
        value = json.loads((candidate_root / "temporal-counterfactual-action-score-v1.json").read_text("ascii"))
        receipts.append(contract.validate_candidate_score_receipt(value))
    group_receipt = contract.validate_group_receipt(
        json.loads((root / group_name).read_text("ascii")),
        candidate_receipts=receipts,
    )
    if (
        group_receipt["candidate_order"] != expected_ids
        or group_receipt["candidate_receipt_digests"] != [row["receipt_digest"] for row in receipts]
        or group_receipt["method_source_revision"] != revision
        or group_receipt["method_source_archive_sha256"] != archive_sha
        or group_receipt["scorer_source_sha256"] != scorer_sha
        or group_receipt["contract_source_sha256"] != contract_sha
        or group_receipt["optimizer_authorized"] is not False
    ):
        raise SystemExit(f"{group_id} group receipt binding differs")
print(json.dumps({"candidate_count": 40, "groups": 2, "optimizer_authorized": False}, sort_keys=True))
PY

#!/usr/bin/env bash
#SBATCH --job-name=pair5-t2v-score-v4
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

# Two concurrent persistent frozen-Bernini workers.  Each worker is one legal
# Ulysses-SP4 group and serially scores its sealed core4 cells/candidates.
# Primary score: global CandidateActionEnergyResult.reward at native exact40
# index 33: physical sigma=0.5161304473876953 and discrete model timestep=516
# from that same pinned schedule coordinate (never the legacy 1000*sigma map).
# Phase conjunction: diagnostic only.  No generation media enters the scorer.

source_archive="${PAIR_V5_T2V_SCORE_SOURCE_ARCHIVE:?set PAIR_V5_T2V_SCORE_SOURCE_ARCHIVE}"
source_archive_sha256="${PAIR_V5_T2V_SCORE_SOURCE_ARCHIVE_SHA256:?set PAIR_V5_T2V_SCORE_SOURCE_ARCHIVE_SHA256}"
source_revision="${PAIR_V5_T2V_SCORE_SOURCE_REVISION:?set PAIR_V5_T2V_SCORE_SOURCE_REVISION}"
bank_spec="${PAIR_V5_T2V_BANK_SPEC:?set PAIR_V5_T2V_BANK_SPEC}"
bank_spec_sha256="${PAIR_V5_T2V_BANK_SPEC_SHA256:?set PAIR_V5_T2V_BANK_SPEC_SHA256}"
bank_output="${PAIR_V5_T2V_BANK_OUTPUT_DIR:?set PAIR_V5_T2V_BANK_OUTPUT_DIR}"
bank_receipt="${PAIR_V5_T2V_BANK_RECEIPT:?set PAIR_V5_T2V_BANK_RECEIPT}"
bank_receipt_sha256="${PAIR_V5_T2V_BANK_RECEIPT_SHA256:?set PAIR_V5_T2V_BANK_RECEIPT_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
output_dir="${PAIR_V5_T2V_SCORE_OUTPUT_DIR:?set PAIR_V5_T2V_SCORE_OUTPUT_DIR}"
python_bin="${PAIR_V5_T2V_PYTHON_BIN:?set PAIR_V5_T2V_PYTHON_BIN}"

fail() { echo "[pair-v5-t2v-score-v4] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

for name in source_archive bank_spec bank_output bank_receipt bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${bank_spec_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "bank spec SHA-256 differs"
[[ "${bank_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "bank receipt SHA-256 differs"
[[ "${checkpoint_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "checkpoint manifest SHA-256 differs"

source_archive="$(realpath -e -- "${source_archive}")"
bank_spec="$(realpath -e -- "${bank_spec}")"
bank_output="$(realpath -e -- "${bank_output}")"
bank_receipt="$(realpath -e -- "${bank_receipt}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -f "${bank_spec}" && ! -L "${bank_spec}" ]] || fail "bank spec differs"
[[ -d "${bank_output}" && ! -L "${bank_output}" ]] || fail "bank output differs"
[[ -f "${bank_receipt}" && ! -L "${bank_receipt}" ]] || fail "bank receipt differs"
[[ "$(dirname -- "${bank_receipt}")" == "${bank_output}" ]] || fail "bank receipt is outside bank output"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source revision/archive differs"
[[ "$(sha256sum "${bank_spec}" | awk '{print $1}')" == "${bank_spec_sha256}" ]] || fail "bank spec hash differs"
[[ "$(sha256sum "${bank_receipt}" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "bank receipt hash differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"

# Refuse a v1/unknown bank even if its caller supplies self-consistent hashes.
# The scorer repeats the complete artifact audit; this preflight binds the
# launcher itself to the sealed core4-v2 root and its canonical bank receipt.
"${python_bin}" -B - "${bank_spec}" "${bank_spec_sha256}" "${bank_receipt}" "${bank_receipt_sha256}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def reject_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit("sealed JSON contains a duplicate key")
        value[key] = item
    return value


def reject_constant(value):
    raise SystemExit(f"sealed JSON contains non-finite constant: {value}")


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def load_sealed(raw_path, expected_sha256, label):
    path = Path(raw_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SystemExit(f"{label} raw SHA-256 differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        raise SystemExit(f"{label} is not canonical sealed JSON")
    return value


spec_sha = sys.argv[2]
spec = load_sealed(sys.argv[1], spec_sha, "core4-v2 root spec")
bank = load_sealed(sys.argv[3], sys.argv[4], "core4-v2 bank receipt")
if spec.get("schema_version") != "pair-v5-frozen-bernini-t2v-calibration-bank-spec-v2":
    raise SystemExit("launcher accepts only the sealed PAIR-v5 core4-v2 spec")
groups = spec.get("groups")
if not isinstance(groups, list) or [item.get("group_id") for item in groups] != ["sp4-a", "sp4-b"]:
    raise SystemExit("core4-v2 group topology differs")
if [len(item.get("candidates", [])) for item in groups] != [20, 20]:
    raise SystemExit("core4-v2 group candidate counts differ")
candidate_ids = [
    row.get("candidate_id")
    for group in groups
    for row in group.get("candidates", [])
]
if (
    len(candidate_ids) != 40
    or len(set(candidate_ids)) != 40
    or not all(
        isinstance(value, str) and value.startswith("pair5-t2v-core4-v2-")
        for value in candidate_ids
    )
):
    raise SystemExit("core4-v2 candidate identity closure differs")
if (
    bank.get("schema_version")
    != "pair-v5-frozen-bernini-t2v-calibration-bank-receipt-v2"
    or bank.get("root_spec_raw_sha256") != spec_sha
    or bank.get("candidate_count") != 40
    or bank.get("cell_count") != 4
):
    raise SystemExit("core4-v2 bank receipt binding differs")
bank_rows = bank.get("candidate_receipts")
if (
    not isinstance(bank_rows, list)
    or [row.get("candidate_id") for row in bank_rows] != candidate_ids
    or not isinstance(bank.get("same_cell_gaussian_proofs"), list)
    or len(bank["same_cell_gaussian_proofs"]) != 4
):
    raise SystemExit("core4-v2 bank candidate/proof closure differs")
unsigned = dict(bank)
declared_digest = unsigned.pop("receipt_digest", None)
if not isinstance(declared_digest, str) or hashlib.sha256(canonical(unsigned)).hexdigest() != declared_digest:
    raise SystemExit("core4-v2 embedded bank receipt digest differs")
PY

[[ "${output_dir}" != / ]] || fail "output cannot be root"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output must be canonical"
[[ "${output_dir##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing output reuse"
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
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v5-t2v-score-v4-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""; sp4_b_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v5-t2v-score-v4-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[pair-v5-t2v-score-v4] PASS topology=2xSP4 exact81=true exact40_index=33 native_model_timestep=516 canonical_decimal_score=true global_mace_diagnostic=true phase_gate=false output=${output_dir}"
  else
    echo "[pair-v5-t2v-score-v4] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
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
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || \
  fail "scratch source archive hash differs after copy"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || \
  fail "scratch source archive revision differs after copy"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
 "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py",
 "methods/bernini_action_editing/pair_v5_t2v_calibration_bank_spec.py",
 "methods/bernini_action_editing/infer_pair_v5_t2v_calibration_bank.py",
 "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
 "methods/bernini_action_editing/infer_lora.py",
 "methods/bernini_action_editing/train_lora.py",
 "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
 "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
 "methods/bernini_action_editing/source_kv_replay.py",
 "methods/bernini_action_editing/source_kv_route_batches.py",
 "methods/bernini_action_editing/source_value_residual.py",
 "methods/bernini_action_editing/pair_v5_native_bridge.py",
 "methods/bernini_action_editing/pair_v5_phase_conjunctive_energy.py",
 "methods/bernini_action_editing/mace_candidate_action_energy.py",
 "methods/bernini_action_editing/pair_v5_action_adapter.py",
 "methods/bernini_action_editing/inference_sigma_strata.py",
 "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
 "methods/bernini_action_editing/source_self_native_rv2v_guidance.py",
 "methods/bernini_action_editing/source_self_native_target_adapter.py",
 "methods/bernini_action_editing/dclr_runtime_contract.py",
}
seen=set()
with tarfile.open(sys.argv[1], "r:*") as handle:
 for member in handle.getmembers():
  path=PurePosixPath(member.name)
  if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
   raise SystemExit("archive member escaped or is not a plain tree entry")
  seen.add(path.as_posix().lstrip("./"))
missing=sorted(required-seen)
if missing: raise SystemExit(f"archive lacks PAIR-v5 scorer closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
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
    "${method_root}/score_pair_v5_t2v_energy_bank_v3.py" \
    --root-spec "${bank_spec}" --expected-root-spec-sha256 "${bank_spec_sha256}" \
    --bank-output-dir "${bank_output}" --bank-receipt "${bank_receipt}" \
    --expected-bank-receipt-sha256 "${bank_receipt_sha256}" --group-id "${group_id}" \
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
    --output-dir "${output_dir}/${group_id}" --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}" --ack-no-action-success-claim
)

run_group sp4-a "0,1,2,3" "${sp4_a_port}" >"${output_dir}/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_group sp4-b "4,5,6,7" "${sp4_b_port}" >"${output_dir}/sp4-b.log" 2>&1 & sp4_b_pid=$!
sp4_a_status=0; sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" != 0 || "${sp4_b_status}" != 0 ]]; then
  tail -n 160 -- "${output_dir}/sp4-a.log" >&2 || true
  tail -n 160 -- "${output_dir}/sp4-b.log" >&2 || true
  fail "one or more SP4 score groups failed"
fi
[[ -f "${output_dir}/sp4-a/pair-v5-t2v-global-energy-sp4-a-v4.json" ]] || fail "sp4-a group receipt absent"
[[ -f "${output_dir}/sp4-b/pair-v5-t2v-global-energy-sp4-b-v4.json" ]] || fail "sp4-b group receipt absent"
"${python_bin}" -B - \
  "${bank_spec}" "${bank_receipt}" "${bank_spec_sha256}" \
  "${checkpoint_manifest_sha256}" "${source_revision}" \
  "${source_archive_sha256}" "${output_dir}" "${method_root}" \
  "${bank_receipt_sha256}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

method_root = Path(sys.argv[8]).resolve(strict=True)
if not method_root.is_dir() or method_root.is_symlink():
    raise SystemExit("postflight scorer source root differs")
sys.path.insert(0, str(method_root))
import score_pair_v5_t2v_energy_bank_v3 as scorer


def reject_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit("score receipt contains a duplicate key")
        value[key] = item
    return value


def canonical(value, *, ensure_ascii=True):
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def load(path, *, ensure_ascii=True):
    raw = Path(path).read_bytes()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"score receipt is not strict JSON: {error}") from error
    if not isinstance(value, dict) or raw != canonical(
        value, ensure_ascii=ensure_ascii
    ) + b"\n":
        raise SystemExit("score receipt is not canonical JSON")
    return value


spec = load(sys.argv[1], ensure_ascii=False)
bank = load(sys.argv[2], ensure_ascii=False)
spec_sha, checkpoint_manifest_sha, revision, archive_sha = sys.argv[3:7]
output = Path(sys.argv[7])
bank_receipt_sha = sys.argv[9]
bank_digest = bank.get("receipt_digest")
bank_rows = bank.get("candidate_receipts")
if not isinstance(bank_rows, list):
    raise SystemExit("postflight bank candidate registry differs")
bank_by_id = {row.get("candidate_id"): row for row in bank_rows}
if len(bank_by_id) != len(bank_rows):
    raise SystemExit("postflight bank candidate IDs repeat")
for group in spec["groups"]:
    group_id = group["group_id"]
    expected_ids = [row["candidate_id"] for row in group["candidates"]]
    try:
        _checked_spec, checked_bank, bound_rows = scorer.load_group_bank(
            root_spec=sys.argv[1],
            root_spec_sha256=spec_sha,
            bank_output_dir=Path(sys.argv[2]).parent,
            bank_receipt=sys.argv[2],
            bank_receipt_sha256=bank_receipt_sha,
            group_id=group_id,
        )
    except scorer.PairV5T2VEnergyScoringError as error:
        raise SystemExit(
            f"{group_id} postflight generation-chain validation failed: {error}"
        ) from error
    if checked_bank.get("receipt_digest") != bank_digest:
        raise SystemExit(f"{group_id} postflight bank digest differs")
    bound_by_id = {
        row["candidate"]["candidate_id"]: row for row in bound_rows
    }
    if list(bound_by_id) != expected_ids:
        raise SystemExit(f"{group_id} postflight bound candidate order differs")
    expected_generation_registry_by_cell = {}
    for candidate in group["candidates"]:
        cell_id = candidate["calibration_group_id"]
        expected_generation_registry_by_cell.setdefault(cell_id, {})[
            candidate["semantic_branch"]
        ] = bound_by_id[candidate["candidate_id"]]["generation_runtime_binding"]
    receipt_name = f"pair-v5-t2v-global-energy-{group_id}-v4.json"
    group_root = output / group_id
    receipt = load(group_root / receipt_name)
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if not isinstance(declared, str) or hashlib.sha256(canonical(unsigned)).hexdigest() != declared:
        raise SystemExit(f"{group_id} embedded group receipt digest differs")
    coordinate = receipt.get("schedule_coordinate", {})
    checkpoint = receipt.get("checkpoint_content_binding", {})
    if (
        receipt.get("schema_version")
        != "bernini-pair-v5-frozen-t2v-global-energy-group-v4"
        or receipt.get("group_id") != group_id
        or receipt.get("root_spec_raw_sha256") != spec_sha
        or receipt.get("bank_receipt_digest") != bank_digest
        or receipt.get("candidate_count") != len(expected_ids)
        or len(receipt.get("candidate_receipt_digests", [])) != len(expected_ids)
        or len(set(receipt.get("candidate_receipt_digests", []))) != len(expected_ids)
        or receipt.get("method_source_revision") != revision
        or receipt.get("method_source_archive_sha256") != archive_sha
        or checkpoint.get("manifest_sha256") != checkpoint_manifest_sha
        or checkpoint.get("every_file_sha256_verified") is not True
        or checkpoint.get("all_loaded_parameters_frozen") is not True
        or coordinate.get("schedule_index") != 33
        or coordinate.get("physical_sigma") != 0.5161304473876953
        or coordinate.get("native_scheduler_timestep") != 516
        or coordinate.get("frozen_t2v_scorer_timestep") != 516.0
        or coordinate.get("frozen_t2v_scorer_timestep_mapping")
        != "direct_native_unipc40_discrete_timestep_same_schedule_index"
        or coordinate.get("physical_sigma_and_model_timestep_share_native_exact40_index") is not True
        or coordinate.get("legacy_1000_sigma_timestep_rejected") is not True
        or receipt.get("training_performed") is not False
        or receipt.get("optimizer_authorized") is not False
        or receipt.get("scientific_action_editing_claim") is not False
    ):
        raise SystemExit(f"{group_id} post-run provenance binding differs")
    expected_children = set(expected_ids) | {receipt_name}
    if {item.name for item in group_root.iterdir()} != expected_children:
        raise SystemExit(f"{group_id} output closure differs")
    actual_candidate_receipt_digests = []
    for candidate, candidate_id in zip(group["candidates"], expected_ids):
        candidate_root = group_root / candidate_id
        if candidate_root.is_symlink() or not candidate_root.is_dir():
            raise SystemExit(f"{group_id} candidate output root differs")
        score_path = candidate_root / "pair-v5-t2v-global-energy-score-v4.json"
        if (
            {item.name for item in candidate_root.iterdir()} != {score_path.name}
            or score_path.is_symlink()
            or not score_path.is_file()
        ):
            raise SystemExit(f"{group_id} candidate score output closure differs")
        raw_score = load(score_path)
        try:
            score = scorer.validate_score_receipt(raw_score)
        except scorer.PairV5T2VEnergyScoringError as error:
            raise SystemExit(
                f"{group_id}/{candidate_id} score receipt validation failed: {error}"
            ) from error
        bound = bound_by_id[candidate_id]
        bank_row = bank_by_id.get(candidate_id)
        identity_fields = (
            "candidate_id",
            "analysis_split",
            "action_family_id",
            "calibration_group_id",
            "actor_group_id",
            "scene_group_id",
            "action_group_id",
            "semantic_branch",
        )
        if (
            not isinstance(bank_row, dict)
            or any(score.get(name) != candidate.get(name) for name in identity_fields)
            or score.get("root_spec_raw_sha256") != spec_sha
            or score.get("bank_receipt_digest") != bank_digest
            or score.get("candidate_envelope_sha256")
            != bound["candidate_envelope_sha256"]
            or score.get("generation_receipt_digest")
            != bound["generation_receipt_digest"]
            or score.get("generation_receipt_digest")
            != bank_row.get("receipt_digest")
            or score.get("generation_receipt_file_sha256")
            != bound["generation_receipt_file_sha256"]
            or score.get("native_rollout_receipt_digest")
            != bound["native_rollout_receipt_digest"]
            or score.get("native_rollout_receipt_file_sha256")
            != bound["native_rollout_receipt_file_sha256"]
            or score.get("generated_mp4_sha256")
            != bound["artifacts"]["mp4"]["sha256"]
            or score.get("clean_latent_artifact_sha256")
            != bound["artifacts"]["predecode_clean_latent"]["sha256"]
            or score.get("official_gaussian_artifact_sha256")
            != bound["artifacts"]["official_initial_gaussian"]["sha256"]
            or score.get("official_gaussian_raw_value_sha256")
            != bound["artifacts"]["official_initial_gaussian"][
                "raw_value_sha256"
            ]
            or score.get("official_gaussian_content_sha256")
            != bound["artifacts"]["official_initial_gaussian"][
                "content_sha256"
            ]
            or score.get("checkpoint_content_binding") != checkpoint
            or score.get("frozen_checkpoint_receipt_digest")
            != receipt.get("frozen_checkpoint_receipt_digest")
            or score.get("schedule_coordinate") != coordinate
            or score.get("generation_runtime_binding_by_branch")
            != expected_generation_registry_by_cell[
                candidate["calibration_group_id"]
            ]
        ):
            raise SystemExit(
                f"{group_id}/{candidate_id} score/generation/group join differs"
            )
        actual_candidate_receipt_digests.append(score["receipt_digest"])
    if actual_candidate_receipt_digests != receipt.get(
        "candidate_receipt_digests"
    ):
        raise SystemExit(
            f"{group_id} ordered candidate receipt digest registry differs"
        )
PY
echo "PAIR_V5_T2V_GLOBAL_ENERGY_V4_DUAL4_OK"

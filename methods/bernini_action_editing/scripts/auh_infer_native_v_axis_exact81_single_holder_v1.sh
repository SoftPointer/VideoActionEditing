#!/usr/bin/env bash
# Stage-A0 full-video V-axis probe on retained holder 135412/gpu293.
# The controller launches one numbered all8 child and never releases/cancels
# the parent allocation.  Inside the child, one WORLD4/SP4 group is reused
# serially for dog then human on GPUs 0--3.  Parent 135412 has only 64G host
# RAM, so two simultaneous model replicas are deliberately forbidden.
set -Eeuo pipefail
umask 077

fail() { echo "[native-v-axis-holder] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly holder_job=135412
readonly holder_node=auh7-1b-gpu-293
readonly holder_user=guangyi.chen
readonly checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d

assert_rocm_idle() {
  local snapshot="$1" expected="$2" uses mems busy
  uses="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  mems="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if((v+0)!=0)print}' <<<"${snapshot}")"
  [[ "${uses}" == "${expected}" && "${mems}" == "${expected}" ]] || fail "GPU inventory differs"
  [[ -z "${busy}" ]] || fail "holder GPUs are not idle"
}

if [[ "${1:-}" == __exec ]]; then
  shift
  [[ $# == 0 ]] || fail "unexpected child arguments"
  [[ "${SLURM_JOB_ID:?numbered child requires Slurm}" == "${holder_job}" ]] || fail "child job differs"
  [[ "$(hostname -s)" == "${holder_node}" ]] || fail "child node differs"
  [[ "${SLURM_STEP_ID:?numbered step required}" =~ ^[0-9]+$ ]] || fail "child step differs"

  method_root="${NATIVE_V_AXIS_METHOD_ROOT:?set method root}"
  output_root="${NATIVE_V_AXIS_OUTPUT_ROOT:?set output root}"
  runtime_revision="${NATIVE_V_AXIS_RUNTIME_REVISION:?set runtime revision}"
  bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
  veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
  checkpoint="${BERNINI_ACTION_CHECKPOINT:?set checkpoint}"
  checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
  python_bin="${NATIVE_V_AXIS_PYTHON_BIN:?set Python}"
  dog_port="${NATIVE_V_AXIS_DOG_PORT:?set dog port}"
  human_port="${NATIVE_V_AXIS_HUMAN_PORT:?set human port}"

  for value in "${method_root}" "${output_root}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" "${python_bin}"; do
    [[ "${value}" == /* ]] || fail "all child paths must be absolute"
  done
  [[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "method root differs"
  [[ -d "${output_root}" && ! -L "${output_root}" ]] || fail "output root differs"
  [[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
  [[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision differs"
  [[ "${dog_port}" =~ ^[0-9]+$ && "${human_port}" =~ ^[0-9]+$ ]] || fail "ports differ"

  runtime_path="${method_root}/infer_native_v_axis_exact81_probe_v1.py"
  core_path="${method_root}/native_v_axis_guidance_v1.py"
  cell_spec="${method_root}/assets/native_v_axis_exact81_core2_v1.json"
  launcher_path="${method_root}/scripts/auh_infer_native_v_axis_exact81_single_holder_v1.sh"
  closure_files=(
    "${runtime_path}"
    "${core_path}"
    "${method_root}/native_i_axis_guidance.py"
    "${method_root}/infer_native_identity_generation_canary.py"
    "${method_root}/infer_orderless_source_frame_set_noise_canary.py"
    "${method_root}/tri_branch_unipc.py"
    "${method_root}/infer_lora.py"
    "${method_root}/train_lora.py"
    "${method_root}/infer_source_kv_carrier_oracle.py"
    "${method_root}/infer_source_value_residual_oracle.py"
    "${cell_spec}"
    "${launcher_path}"
  )
  for path in "${closure_files[@]}"; do
    [[ -f "${path}" && ! -L "${path}" ]] || fail "method closure file absent: ${path}"
  done
  [[ "$(sha256_file "${checkpoint_manifest}")" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest differs"
  cell_spec_sha="$(sha256_file "${cell_spec}")"
  launcher_sha="$(sha256_file "${launcher_path}")"
  closure_sha="$(sha256sum "${closure_files[@]}" | sha256sum | awk '{print $1}')"

  for test_file in test_native_v_axis_exact81_probe_v1.py test_auh_infer_native_v_axis_exact81_single_holder_v1.py; do
    [[ -f "${method_root}/tests/${test_file}" && ! -L "${method_root}/tests/${test_file}" ]] || fail "test absent: ${test_file}"
    PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
  done

  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONNOUSERSITE=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export MODELING_BACKEND=hf
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_SOCKET_IFNAME=bond0
  export GLOO_SOCKET_IFNAME=bond0
  export NCCL_IB_DISABLE=1

  scratch_parent="${SLURM_TMPDIR:-/tmp}"
  [[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
  task_scratch="$(mktemp -d "${scratch_parent%/}/native-v-axis-${SLURM_JOB_ID}.XXXXXX")"
  load_lock="${task_scratch}/renderer-load.lock"
  touch "${load_lock}"
  chmod 0600 "${load_lock}"
  export NATIVE_V_AXIS_LOAD_LOCK="${load_lock}"
  cleanup() {
    local status=$?
    trap - EXIT TERM INT
    case "${task_scratch:-}" in
      "${scratch_parent%/}/native-v-axis-${SLURM_JOB_ID}."*) ;;
      *) exit 2 ;;
    esac
    if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
      chmod -R u+w -- "${task_scratch}"
      rm -rf -- "${task_scratch}"
    fi
    [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || status=2
    exit "${status}"
  }
  trap cleanup EXIT

  launch_group() (
    set -Eeuo pipefail
    label="$1"
    visible_gpus="$2"
    master_port="$3"
    group_scratch="${task_scratch}/${label}"
    mkdir -p -- "${group_scratch}/miopen-user" "${group_scratch}/miopen-custom" "${group_scratch}/torch-extensions" "${group_scratch}/triton" "${group_scratch}/xdg" "${group_scratch}/pycache" "${group_scratch}/tmp"
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
    export ROCR_VISIBLE_DEVICES="${visible_gpus}"
    export MIOPEN_USER_DB_PATH="${group_scratch}/miopen-user"
    export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/miopen-custom"
    export TORCH_EXTENSIONS_DIR="${group_scratch}/torch-extensions"
    export TRITON_CACHE_DIR="${group_scratch}/triton"
    export XDG_CACHE_HOME="${group_scratch}/xdg"
    export PYTHONPYCACHEPREFIX="${group_scratch}/pycache"
    export TMPDIR="${group_scratch}/tmp"
    exec "${python_bin}" -B -m torch.distributed.run \
      --nproc_per_node=4 \
      --master_addr=127.0.0.1 \
      --master_port="${master_port}" \
      "${runtime_path}" \
      --cell-spec "${cell_spec}" \
      --expected-cell-spec-sha256 "${cell_spec_sha}" \
      --cell-id "${label}" \
      --bernini-root "${bernini_root}" \
      --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" \
      --checkpoint-content-manifest "${checkpoint_manifest}" \
      --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
      --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
      --output-dir "${output_root}/${label}" \
      --num-inference-steps 40 \
      --runtime-source-revision "${runtime_revision}" \
      --runtime-source-closure-sha256 "${closure_sha}" \
      --launcher-source-sha256 "${launcher_sha}" \
      --expected-bernini-commit "${bernini_commit}" \
      --expected-veomni-commit "${veomni_commit}"
  )

  echo "[native-v-axis-holder] serial WORLD4/SP4 dog then human GPUs0-3 arms=V-on,V-off,wrong-V seeds=2 exact40 exact81"
  launch_group dog 0,1,2,3 "${dog_port}" >"${output_root}/../logs/dog.log" 2>&1
  launch_group human 0,1,2,3 "${human_port}" >"${output_root}/../logs/human.log" 2>&1

  "${python_bin}" -I -S -B - "${output_root}" "${runtime_revision}" "${closure_sha}" "${launcher_sha}" "${cell_spec_sha}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
revision, closure_sha, launcher_sha, spec_sha = sys.argv[2:]
arms = ["V-on", "V-off", "wrong-V"]

def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()

children = {}
for label in ("dog", "human"):
    receipt_path = root / label / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    candidates = receipt.get("candidates", [])
    outputs = receipt.get("outputs", {})
    sampling = receipt.get("sampling", {})
    interpretation = receipt.get("interpretation", {})
    seeds = sampling.get("seeds", [])
    if not (
        digest(unsigned) == declared
        and receipt.get("schema_version") == "bernini-native-v-axis-exact81-probe-receipt-v1"
        and receipt.get("cell_spec", {}).get("file_sha256") == spec_sha
        and receipt.get("runtime_source", {}).get("revision") == revision
        and receipt.get("runtime_source", {}).get("closure_sha256") == closure_sha
        and receipt.get("runtime_source", {}).get("launcher_sha256") == launcher_sha
        and sampling.get("arm_order") == arms
        and sampling.get("exact40") is True
        and sampling.get("exact81") is True
        and len(seeds) == 2
        and len(candidates) == 6
        and len(outputs) == 6
        and interpretation.get("training_performed") is False
        and interpretation.get("feature_scorer_consumed") is False
        and interpretation.get("reward_computed") is False
        and interpretation.get("ranking_performed") is False
        and interpretation.get("best_arm_selected") is False
        and receipt.get("resource_lifetime", {}).get("rank_serialized_checkpoint_deserialize") is True
        and receipt.get("resource_lifetime", {}).get("model_moved_to_rank_device_before_load_lock_release") is True
        and receipt.get("resource_lifetime", {}).get("vae_instantiated_on_rank_zero_only") is True
        and receipt.get("resource_lifetime", {}).get("rank_zero_only_vae_observed") is True
        and receipt.get("resource_lifetime", {}).get("text_encoder_retired_before_vae_and_sampling") is True
        and receipt.get("resource_lifetime", {}).get("sampling_model_destroyed_without_cpu_offload_before_rank_zero_decode") is True
    ):
        raise SystemExit(f"{label} receipt gate failed")
    for seed in seeds:
        rows = [row for row in candidates if row.get("seed") == seed]
        if [row.get("arm") for row in rows] != arms:
            raise SystemExit(f"{label}/{seed} arm order differs")
        if len({row.get("official_initial_gaussian_raw_value_sha256") for row in rows}) != 1:
            raise SystemExit(f"{label}/{seed} Gaussian differs")
        if len({tuple(row.get("correct_reference_raw_storage_sha256_in_order", [])) for row in rows}) != 1:
            raise SystemExit(f"{label}/{seed} correct references differ")
        if len({row.get("action_caption_utf8_sha256") for row in rows}) != 1:
            raise SystemExit(f"{label}/{seed} instruction differs")
    for key, artifact in outputs.items():
        artifact_path = Path(artifact["path"])
        if not (artifact_path.is_file() and artifact["frame_count"] == 81 and artifact["fps"] == 25):
            raise SystemExit(f"{label}/{key} exact81 artifact differs")
    children[label] = {
        "receipt_path": str(receipt_path),
        "receipt_digest": declared,
        "candidate_count": len(candidates),
        "arms": arms,
        "seeds": seeds,
    }

unsigned = {
    "schema_version": "bernini-native-v-axis-all8-holder-receipt-v1",
    "holder_job": 135412,
    "holder_node": "auh7-1b-gpu-293",
    "parent_released": False,
    "exact40": True,
    "exact81": True,
    "generated_video_count": 12,
    "feature_scorer_consumed": False,
    "ranking_performed": False,
    "children": children,
}
value = {**unsigned, "receipt_digest": digest(unsigned)}
target = root.parent / "all8-receipt.json"
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
fd = os.open(target, flags, 0o400)
with os.fdopen(fd, "w", encoding="ascii") as handle:
    json.dump(value, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  echo "[native-v-axis-holder] PASS exact40=true exact81=true videos=12 output=${output_root}"
  exit 0
fi

[[ "${1:-}" == run && $# == 1 ]] || fail "usage: $0 run"
method_root="${NATIVE_V_AXIS_METHOD_ROOT:?set method root}"
run_root="${NATIVE_V_AXIS_RUN_ROOT:?set fresh run root}"
runtime_revision="${NATIVE_V_AXIS_RUNTIME_REVISION:?set full source revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
python_bin="${NATIVE_V_AXIS_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
base_port="${NATIVE_V_AXIS_BASE_PORT:-31412}"

for value in "${method_root}" "${run_root}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" "${python_bin}"; do
  [[ "${value}" == /* ]] || fail "all controller paths must be absolute"
done
[[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision differs"
[[ "${base_port}" =~ ^[0-9]+$ ]] && (( base_port >= 1024 && base_port <= 65533 )) || fail "base port differs"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "method root differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
[[ "${run_root}" != / && "$(realpath -m -- "${run_root}")" == "${run_root}" && ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh canonical"

record="$(scontrol show job -o "${holder_job}")"
[[ "${record}" == *"JobId=${holder_job} "* && "${record}" == *"JobState=RUNNING"* && "${record}" == *"NodeList=${holder_node}"* && "${record}" == *"UserId=${holder_user}"* ]] || fail "retained holder identity/state differs"
steps="$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/{print}')"
[[ -z "${steps}" ]] || fail "holder has active numbered child: ${steps}"
snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showuse --showmemuse')"
assert_rocm_idle "${snapshot}" 8
sleep 2
snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showuse --showmemuse')"
assert_rocm_idle "${snapshot}" 8

mkdir -m 0700 "${run_root}" "${run_root}/logs" "${run_root}/outputs"
set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --cpus-per-task=64 --mem=60G --gres=gpu:mi210:8 \
  env NATIVE_V_AXIS_METHOD_ROOT="${method_root}" \
    NATIVE_V_AXIS_OUTPUT_ROOT="${run_root}/outputs" \
    NATIVE_V_AXIS_RUNTIME_REVISION="${runtime_revision}" \
    NATIVE_V_AXIS_PYTHON_BIN="${python_bin}" \
    NATIVE_V_AXIS_DOG_PORT="${base_port}" \
    NATIVE_V_AXIS_HUMAN_PORT="$((base_port + 1))" \
    BERNINI_OFFICIAL_ROOT="${bernini_root}" \
    BERNINI_VEOMNI_ROOT="${veomni_root}" \
    BERNINI_ACTION_CHECKPOINT="${checkpoint}" \
    BERNINI_CHECKPOINT_CONTENT_MANIFEST="${checkpoint_manifest}" \
    bash "${method_root}/scripts/auh_infer_native_v_axis_exact81_single_holder_v1.sh" __exec \
  >"${run_root}/logs/controller-child.log" 2>&1
status=$?
set -e
printf 'child_exit=%s\nparent_job=%s\nparent_node=%s\nparent_not_released=true\n' "${status}" "${holder_job}" "${holder_node}" >"${run_root}/controller.status"
if (( status != 0 )); then
  tail -n 240 "${run_root}/logs/controller-child.log" >&2 || true
  exit "${status}"
fi
printf 'COMPLETE holder=%s node=%s parent_retained=true exact40=true exact81=true videos=12\n' "${holder_job}" "${holder_node}" >"${run_root}/controller.COMPLETE"

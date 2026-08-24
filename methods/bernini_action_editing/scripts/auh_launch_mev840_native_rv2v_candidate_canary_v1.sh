#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Two-seed, zero-update, official-native RV2V candidate canary for MEV840.
# The generation process accepts only the exact source video and one complete
# action caption.  Target media and target-derived action summaries are absent
# from this launch surface and from the deployed generation directory.

fail() { echo "[mev840-native-rv2v-canary] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }

readonly stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
readonly mev_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v1
readonly control_root="${stage}/mev840_native_rv2v_candidate_canary_v3_20260822_control"
readonly output_root="${stage}/mev840_native_rv2v_candidate_canary_v3_20260822"
readonly release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/generic-action-confirmation40-generation-r3-ac22e19f-r1
readonly runtime_archive="${release_root}/source.tar"
readonly runtime_manifest="${release_root}/source.manifest.json"
readonly runtime_tree="${release_root}/runtime"
readonly runtime_archive_sha=46ae7529d640a197006ab8d7d17c23ac81925dabd7fa1caf4b0bb261197e8115
readonly runtime_manifest_sha=e104031526236f16e94a4753c31ad8048b1a65345b1913212c35e421fcad48ae
readonly runtime_manifest_digest=4e78a935b2485e3f8c2c94aa5524a82ed25aa0b93aaf58dd81476dc5c9b48044
readonly content_revision=ac22e19ffd109a2d6b85c32c64463b0be8373792

readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly source_video="${mev_root}/preprocessed_sources/840b214afead/source-exact81.mp4"
readonly source_sha=a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646
readonly action_prompt='A young woman with long reddish-blonde hair, wearing a mint green athletic outfit, stands on a treadmill in a modern gym with blue and white striped pillars, holding a water bottle. In this continuous shot, The woman turns her head to the left and places her water bottle onto the treadmill. The same subject identity, scene, lighting, framing, and camera remain stable.'
readonly action_prompt_sha=effdf094385a4f2486391efc008150b7436a8137c1d5766864a678ed6e0c749f

readonly runner_rel=methods/bernini_action_editing/infer_native_identity_generation_canary.py
readonly runner_sha=bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42
readonly infer_lora_sha=babd6d63287723ccd14b2bbe43bd4550c30b4feaa794d17c66f5a5ddefe979fe
readonly train_lora_sha=630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5
readonly source_kv_oracle_sha=fcf77576735c89e685415b94b2dc0f0c5b8d1dd8dc1c55832538ff0daafb4604
readonly source_value_oracle_sha=40e581db7906f20103a16ad47fda76978cbad21c9277723f3e8e022d717ed2d8
readonly source_kv_replay_sha=45b43426dc7825dbd61280154fc35161c60476ec5cb9e53bc0225f3809c759f3
readonly source_kv_batches_sha=7f3ae0d27747ad58b3b195c712884641012eb836bb59963896c58518b8b5731e
readonly source_value_sha=420cadf3cb2824b2bf5a809c55086d81351db19f31743b0b77a957adf219e124
readonly materialize_vae_sha=a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0
readonly build_renderer_dataset_sha=afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5

usage() {
  echo "usage: $0 launch | worker 2027|2028" >&2
  exit 2
}

verify_plain_sha() {
  local path="$1" expected="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "plain file missing: ${path}"
  [[ "$(sha256_file "${path}")" == "${expected}" ]] || fail "SHA differs: ${path}"
}

verify_fixed_authority() {
  verify_plain_sha "${runtime_archive}" "${runtime_archive_sha}"
  verify_plain_sha "${runtime_manifest}" "${runtime_manifest_sha}"
  verify_plain_sha "${python_bin}" "${python_sha}"
  verify_plain_sha "${checkpoint_manifest}" "${checkpoint_manifest_sha}"
  verify_plain_sha "${source_video}" "${source_sha}"
  [[ "$(printf '%s' "${action_prompt}" | sha256sum | awk '{print $1}')" == "${action_prompt_sha}" ]] || fail "action prompt SHA differs"
  for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
    [[ -d "${path}" && ! -L "${path}" ]] || fail "directory authority differs: ${path}"
  done
  [[ -d "${runtime_tree}" && ! -L "${runtime_tree}" ]] || fail "released runtime tree differs"
  verify_release_closure
  verify_runtime_tree "${runtime_tree}"
}

verify_holder() {
  local job="$1" node="$2" nodes
  [[ "$(squeue -h -j "${job}" -o '%T')" == RUNNING ]] || fail "holder ${job} is not RUNNING"
  nodes="$(squeue -h -j "${job}" -o '%N')"
  scontrol show hostnames "${nodes}" | grep -Fqx -- "${node}" || fail "holder ${job} does not own ${node}"
}

verify_runtime_tree() {
  local root="$1"
  verify_plain_sha "${root}/${runner_rel}" "${runner_sha}"
  verify_plain_sha "${root}/methods/bernini_action_editing/infer_lora.py" "${infer_lora_sha}"
  verify_plain_sha "${root}/methods/bernini_action_editing/train_lora.py" "${train_lora_sha}"
  verify_plain_sha "${root}/methods/bernini_action_editing/infer_source_kv_carrier_oracle.py" "${source_kv_oracle_sha}"
  verify_plain_sha "${root}/methods/bernini_action_editing/infer_source_value_residual_oracle.py" "${source_value_oracle_sha}"
  verify_plain_sha "${root}/methods/bernini_action_editing/source_kv_replay.py" "${source_kv_replay_sha}"
  verify_plain_sha "${root}/methods/bernini_action_editing/source_kv_route_batches.py" "${source_kv_batches_sha}"
  verify_plain_sha "${root}/methods/bernini_action_editing/source_value_residual.py" "${source_value_sha}"
  verify_plain_sha "${root}/methods/bernini_action_editing/tools/materialize_vae.py" "${materialize_vae_sha}"
  verify_plain_sha "${root}/methods/bernini_action_editing/tools/build_renderer_dataset.py" "${build_renderer_dataset_sha}"
}

verify_release_closure() {
  "${python_bin}" -B - "${runtime_archive}" "${runtime_manifest}" "${runtime_tree}" "${runtime_manifest_digest}" <<'PY'
from pathlib import Path, PurePosixPath
import hashlib, json, os, stat, sys, tarfile

archive = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
runtime_root = Path(sys.argv[3])
expected_digest = sys.argv[4]

def pairs(items):
    out = {}
    for key, value in items:
        if key in out: raise SystemExit("duplicate manifest key")
        out[key] = value
    return out

value = json.loads(manifest_path.read_text(encoding="ascii"), object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
unsigned = dict(value); declared = unsigned.pop("manifest_digest", None)
canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
if declared != expected_digest or hashlib.sha256(canonical).hexdigest() != declared:
    raise SystemExit("release manifest digest differs")
if value.get("schema_version") != "bernini-generic-action-confirmation-data-prep-release-v3" or value.get("file_count") != 19 or value.get("content_closure_sha1") != "ac22e19ffd109a2d6b85c32c64463b0be8373792":
    raise SystemExit("release manifest identity differs")
rows = value.get("files")
if not isinstance(rows, list) or len(rows) != 19: raise SystemExit("release file rows differ")
expected = {}
for row in rows:
    rel = PurePosixPath(row["path"])
    if rel.is_absolute() or ".." in rel.parts or rel.as_posix() in expected: raise SystemExit("unsafe release path")
    expected[rel.as_posix()] = row["sha256"]
for rel, wanted in expected.items():
    path = runtime_root / "methods/bernini_action_editing" / rel
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
        raise SystemExit(f"runtime release file differs: {rel}")
with tarfile.open(archive, "r:") as handle:
    seen = {}
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("unsafe release archive member")
        if not member.isfile(): continue
        prefix = PurePosixPath("methods/bernini_action_editing")
        try: rel = path.relative_to(prefix).as_posix()
        except ValueError: raise SystemExit("release archive file escaped method root")
        payload = handle.extractfile(member).read()
        seen[rel] = hashlib.sha256(payload).hexdigest()
if seen != expected: raise SystemExit("release archive/manifest closure differs")
print("MEV840_PROVEN_RELEASE_CLOSURE_OK", len(expected))
PY
}

mode="${1:-}"
case "${mode}" in
  launch)
    [[ $# == 1 ]] || usage
    verify_fixed_authority
    [[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root is not fresh"
    mkdir "${output_root}"
    for seed in 2027 2028; do
      if [[ "${seed}" == 2027 ]]; then
        job=143808; node=auh7-1b-gpu-292
      else
        job=147873; node=auh7-1b-gpu-284
      fi
      verify_holder "${job}" "${node}"
      log="${control_root}/seed${seed}.log"
      pid_file="${control_root}/seed${seed}.pid"
      [[ ! -e "${log}" && ! -L "${log}" && ! -e "${pid_file}" && ! -L "${pid_file}" ]] || fail "launch sidecar is not fresh for seed ${seed}"
      nohup srun --jobid="${job}" --exclusive --nodes=1 --ntasks=1 \
        --cpus-per-task=64 --gres=gpu:4 --mem=0 --nodelist="${node}" \
        bash "${control_root}/auh_launch_mev840_native_rv2v_candidate_canary_v1.sh" worker "${seed}" \
        >"${log}" 2>&1 &
      printf '%s\n' "$!" >"${pid_file}"
    done
    echo "MEV840_NATIVE_RV2V_CANARY_LAUNCHED ${output_root}"
    ;;
  worker)
    [[ $# == 2 ]] || usage
    readonly seed="$2"
    case "${seed}" in
      2027) readonly expected_job=143808 expected_node=auh7-1b-gpu-292 ;;
      2028) readonly expected_job=147873 expected_node=auh7-1b-gpu-284 ;;
      *) usage ;;
    esac
    [[ "${SLURM_JOB_ID:-}" == "${expected_job}" ]] || fail "Slurm job differs"
    [[ "$(hostname -s)" == "${expected_node}" ]] || fail "compute node differs"
    [[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numbered Slurm step required"
    verify_fixed_authority
    readonly candidate_dir="${output_root}/seed${seed}"
    [[ ! -e "${candidate_dir}" && ! -L "${candidate_dir}" ]] || fail "candidate output is not fresh"

    scratch_parent="${SLURM_TMPDIR:-/tmp}"
    [[ "${scratch_parent}" == /* && "${scratch_parent}" != / && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
    scratch="$(mktemp -d "${scratch_parent%/}/mev840-native-rv2v-${seed}-${SLURM_STEP_ID}.XXXXXXXX")"
    cleanup() {
      local status=$?
      trap - EXIT INT TERM HUP
      if [[ -d "${scratch}" && ! -L "${scratch}" && "$(dirname "${scratch}")" == "${scratch_parent%/}" ]]; then
        find "${scratch}" -xdev -depth -mindepth 1 -delete || status=70
        rmdir "${scratch}" || status=70
      else
        status=70
      fi
      exit "${status}"
    }
    trap cleanup EXIT INT TERM HUP
    mkdir "${scratch}/cache" "${scratch}/tmp"
    "${python_bin}" -B -m py_compile "${runtime_tree}/${runner_rel}"

    model_load_lock="${scratch}/renderer-load.lock"
    : >"${model_load_lock}"
    chmod 0400 "${model_load_lock}"
    export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1
    export NATIVE_V_AXIS_LOAD_LOCK="${model_load_lock}"
    unset NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0
    export GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1
    export PYTHONPATH="${runtime_tree}/methods/bernini_action_editing"
    export TMPDIR="${scratch}/tmp"
    export MIOPEN_USER_DB_PATH="${scratch}/cache/miopen-user"
    export MIOPEN_CUSTOM_CACHE_DIR="${scratch}/cache/miopen-custom"
    export TORCH_EXTENSIONS_DIR="${scratch}/cache/torch-extensions"
    export TRITON_CACHE_DIR="${scratch}/cache/triton"
    export XDG_CACHE_HOME="${scratch}/cache/xdg"
    mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"

    "${python_bin}" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
      "${runtime_tree}/${runner_rel}" \
      --bernini-root "${bernini_root}" \
      --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" \
      --checkpoint-content-manifest "${checkpoint_manifest}" \
      --source-video "${source_video}" \
      --expected-source-sha256 "${source_sha}" \
      --action-prompt "${action_prompt}" \
      --expected-action-prompt-sha256 "${action_prompt_sha}" \
      --output-dir "${candidate_dir}" \
      --arms rv2v \
      --num-inference-steps 40 \
      --seed "${seed}" \
      --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793 \
      --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d \
      --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca \
      --method-source-revision "${content_revision}" \
      --method-source-archive-sha256 "${runtime_archive_sha}"

    readonly receipt="${candidate_dir}/receipt.json"
    readonly video="${candidate_dir}/rv2v.mp4"
    verify_plain_sha "${source_video}" "${source_sha}"
    [[ -f "${receipt}" && ! -L "${receipt}" && -f "${video}" && ! -L "${video}" ]] || fail "native output closure is incomplete"
    "${python_bin}" -B - "${receipt}" "${seed}" "${source_sha}" "${action_prompt_sha}" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
sampling=value.get("sampling",{}).get("rv2v",{})
output=value.get("outputs",{}).get("rv2v",{})
freeze=value.get("freeze_certificate",{})
if value.get("arms") != ["rv2v"]: raise SystemExit("arm closure differs")
if value.get("input",{}).get("source_video_sha256") != sys.argv[3]: raise SystemExit("source differs")
if value.get("input",{}).get("action_prompt_utf8_sha256") != sys.argv[4]: raise SystemExit("prompt differs")
if value.get("input",{}).get("target_video") is not False: raise SystemExit("target was admitted")
if sampling.get("seed") != int(sys.argv[2]) or sampling.get("num_inference_steps") != 40: raise SystemExit("sampling differs")
if sampling.get("guidance_mode") != "rv2v" or sampling.get("custom_sampler_or_scheduler") is not False: raise SystemExit("native route differs")
if freeze != {"base_frozen":True,"lora_module_count":0,"trainable_parameter_elements":0,"trainable_parameter_tensors":0}: raise SystemExit("freeze differs")
if output.get("frame_count") != 81 or output.get("fps") != 25: raise SystemExit("media metadata differs")
PY
    video_sha="$(sha256_file "${video}")"
    receipt_sha="$(sha256_file "${receipt}")"
    complete="${candidate_dir}/candidate.complete.json"
    "${python_bin}" -B - "${complete}" "${seed}" "${expected_job}" "${expected_node}" "${video}" "${video_sha}" "${receipt}" "${receipt_sha}" <<'PY'
import hashlib,json,os,sys,tempfile
path,seed,job,node,video,video_sha,receipt,receipt_sha=sys.argv[1:]
row={"schema":"mev840-native-rv2v-candidate-complete-v1","complete":True,"zero_update":True,"generator_target_video_read":False,"generator_action_json_read":False,"seed":int(seed),"slurm":{"job_id":job,"node":node},"video":{"path":video,"sha256":video_sha},"native_receipt":{"path":receipt,"sha256":receipt_sha}}
raw=json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")+b"\n"
fd,tmp=tempfile.mkstemp(prefix=".candidate-complete-",dir=os.path.dirname(path))
try:
 os.write(fd,raw); os.fsync(fd); os.close(fd); fd=-1; os.replace(tmp,path)
finally:
 if fd>=0: os.close(fd)
 if os.path.exists(tmp): os.unlink(tmp)
PY
    echo "MEV840_NATIVE_RV2V_CANDIDATE_COMPLETE seed=${seed} step=${SLURM_JOB_ID}.${SLURM_STEP_ID} video_sha=${video_sha}"
    ;;
  *) usage ;;
esac

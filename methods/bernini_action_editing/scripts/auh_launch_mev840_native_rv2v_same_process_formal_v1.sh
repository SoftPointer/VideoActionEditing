#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Two-seed formal 40-step P0a/P1/P2/P0b RV2V prompt matrix.
# Deployment and launch remain forbidden until a fresh independent GO.

fail() { echo "[mev840-same-process-formal-v1] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
usage() { echo "usage: $0 launch-formal SEED|worker-formal SEED|postflight-formal SEED" >&2; exit 2; }

readonly stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
readonly control_root="${stage}/mev840_native_rv2v_same_process_formal_v1_20260822_control"
readonly output_root="${stage}/mev840_native_rv2v_same_process_formal_v1_20260822"
readonly release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/generic-action-confirmation40-generation-r3-ac22e19f-r1
readonly runtime_archive="${release_root}/source.tar"
readonly runtime_manifest="${release_root}/source.manifest.json"
readonly runtime_archive_sha=46ae7529d640a197006ab8d7d17c23ac81925dabd7fa1caf4b0bb261197e8115
readonly runtime_manifest_sha=e104031526236f16e94a4753c31ad8048b1a65345b1913212c35e421fcad48ae
readonly runtime_manifest_digest=4e78a935b2485e3f8c2c94aa5524a82ed25aa0b93aaf58dd81476dc5c9b48044
readonly content_revision=ac22e19ffd109a2d6b85c32c64463b0be8373792
readonly base_runner_rel=methods/bernini_action_editing/infer_native_identity_generation_canary.py
readonly base_runner_sha=bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42
readonly paired_runner_rel=methods/bernini_action_editing/infer_mev840_native_rv2v_paired_prompt_matrix_formal_v1.py
readonly paired_runner="${control_root}/infer_mev840_native_rv2v_paired_prompt_matrix_formal_v1.py"
readonly paired_runner_sha=1f85e2a2444059161bc8ed073ad0565c315c97fa5e849fb5f6d2ac47c738a0ee

readonly authority="${control_root}/mev840_native_rv2v_same_process_formal_v1.json"
readonly authority_sha=e5d2f0a9a8bbf88df84264494e911e485e3bbff459b57a9bd32e7b39ffc79ab9
readonly prompt_matrix="${control_root}/mev840_native_rv2v_incremental_prompt_matrix_v1.json"
readonly prompt_matrix_sha=5c28e672bcdd86da3c7d3a94ba9e07b644421cea6c5945fb163fa7b871c2af0a
readonly postflight="${control_root}/audit_mev840_native_rv2v_same_process_formal_v1.py"
readonly postflight_sha=4c8c162f68127126f400cd07777dc841d3530b484ca4706e3ac5d691da697c28
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly unipc_source=/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/diffusers/schedulers/scheduling_unipc_multistep.py
readonly unipc_source_sha=5bfe1dcf55ebea6dbbf624d3af676b2529b81fbcaf493150d562ec9e1aba3872
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly source_video=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v1/preprocessed_sources/840b214afead/source-exact81.mp4
readonly source_sha=a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646
readonly mechanical_receipt=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/mev840_native_rv2v_same_process_prompt_matrix_v2_20260822_mechanical_seed2028/receipt.json
readonly mechanical_receipt_sha=40d5124281472eb89c7cb9bc8ee9b6436892ee4ec40c3f3af22698ddf5f43172

verify_file() {
  [[ -f "$1" && ! -L "$1" ]] || fail "plain file missing: $1"
  [[ "$(sha256_file "$1")" == "$2" ]] || fail "SHA differs: $1"
}

verify_release() {
  local extracted="${1:-}"
  "${python_bin}" -B - "${runtime_archive}" "${runtime_manifest}" "${runtime_manifest_digest}" "${extracted}" "${base_runner_rel}" "${base_runner_sha}" <<'PY'
from pathlib import Path, PurePosixPath
import hashlib,json,stat,sys,tarfile
a=Path(sys.argv[1]); m=Path(sys.argv[2]); want_digest=sys.argv[3]; root=sys.argv[4]; runner_rel=sys.argv[5]; runner_sha=sys.argv[6]
v=json.loads(m.read_text(encoding="ascii")); unsigned=dict(v); declared=unsigned.pop("manifest_digest",None)
raw=json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
if declared!=want_digest or hashlib.sha256(raw).hexdigest()!=want_digest: raise SystemExit("manifest digest differs")
expected={r["path"]:r["sha256"] for r in v["files"]}
if v.get("file_count")!=19 or len(expected)!=19 or v.get("content_closure_sha1")!="ac22e19ffd109a2d6b85c32c64463b0be8373792" or expected.get(Path(runner_rel).name)!=runner_sha: raise SystemExit("release identity differs")
if Path(runner_rel).name in v.get("allowed_entrypoints",[]): raise SystemExit("unexpected upstream entrypoint authorization")
seen={}
with tarfile.open(a,"r:") as h:
 for member in h.getmembers():
  p=PurePosixPath(member.name)
  if p.is_absolute() or ".." in p.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo(): raise SystemExit("unsafe archive member")
  if not member.isfile(): continue
  try: rel=p.relative_to(PurePosixPath("methods/bernini_action_editing")).as_posix()
  except ValueError: raise SystemExit("archive escaped method root")
  seen[rel]=hashlib.sha256(h.extractfile(member).read()).hexdigest()
if seen!=expected: raise SystemExit("archive exact closure differs")
if root:
 base=Path(root)/"methods/bernini_action_editing"
 actual={p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() or p.is_symlink()}
 if actual!=set(expected): raise SystemExit("extracted exact19 member set differs")
 for rel,want in expected.items():
  p=base/rel; s=p.lstat()
  if p.is_symlink() or not stat.S_ISREG(s.st_mode) or hashlib.sha256(p.read_bytes()).hexdigest()!=want: raise SystemExit("extracted file differs: "+rel)
print("MEV840_SAME_PROCESS_RELEASE_OK",len(expected),"scratch" if root else "archive")
PY
}

verify_authority() {
  [[ "${paired_runner_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "formal runner is not independently frozen"
  [[ "${authority_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "formal authority is not frozen"
  [[ "${postflight_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "formal postflight is not independently frozen"
  verify_file "${runtime_archive}" "${runtime_archive_sha}"
  verify_file "${runtime_manifest}" "${runtime_manifest_sha}"
  verify_file "${paired_runner}" "${paired_runner_sha}"
  verify_file "${authority}" "${authority_sha}"
  verify_file "${prompt_matrix}" "${prompt_matrix_sha}"
  verify_file "${postflight}" "${postflight_sha}"
  verify_file "${python_bin}" "${python_sha}"
  verify_file "${unipc_source}" "${unipc_source_sha}"
  verify_file "${checkpoint_manifest}" "${checkpoint_manifest_sha}"
  verify_file "${source_video}" "${source_sha}"
  verify_file "${mechanical_receipt}" "${mechanical_receipt_sha}"
  for p in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do [[ -d "$p" && ! -L "$p" ]] || fail "directory authority differs: $p"; done
  verify_release
  "${python_bin}" -B - "${authority}" "${prompt_matrix}" "${mechanical_receipt}" <<'PY'
import hashlib,json,sys
v=json.load(open(sys.argv[1],encoding="ascii"))
m=json.load(open(sys.argv[2],encoding="ascii"))
c=json.load(open(sys.argv[3],encoding="ascii")); unsigned=dict(c); declared=unsigned.pop("receipt_digest",None)
raw=json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
if declared!="98e053ad26144498e66e661088d4aaddf5f67cd969997c37240abda3d0eee2d8" or hashlib.sha256(raw).hexdigest()!=declared: raise SystemExit("mechanical receipt digest differs")
if c.get("schema_version")!="mev840-native-rv2v-paired-prompt-matrix-v1" or c.get("paired_same_process_contract",{}).get("slurm",{}).get("job_step_id")!="147873.10" or c.get("paired_same_process_contract",{}).get("p0_replay",{}).get("generated_latent_bit_exact") is not True or c.get("paired_same_process_contract",{}).get("target_media_or_action_json_read") is not False: raise SystemExit("mechanical gate differs")
if v.get("schema")!="mev840-native-rv2v-same-process-formal-v1": raise SystemExit("authority schema differs")
runtime={"unipc_source":{"path":"/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/diffusers/schedulers/scheduling_unipc_multistep.py","sha256":"5bfe1dcf55ebea6dbbf624d3af676b2529b81fbcaf493150d562ec9e1aba3872"},"formal_slurm_by_seed":{"2027":{"job_id":"143808","node":"auh7-1b-gpu-292","world_size":4},"2028":{"job_id":"147873","node":"auh7-1b-gpu-284","world_size":4}},"nearest_finite_cgroup_limit_bytes":68719476736,"minimum_cgroup_headroom_bytes":67108864}
if v.get("runtime_authority")!=runtime: raise SystemExit("runtime authority differs")
execution={"seeds":[2027,2028],"num_inference_steps":40,"decode_cells":["p0a","p1","p2"],"latent_only_replay_cells":["p0b"],"exact_regular_file_count_per_seed":13,"scientific_candidate":True,"requires_separate_independent_launch_go":True,"mechanical_gate_step":"147873.10","mechanical_gate_receipt_sha256":"40d5124281472eb89c7cb9bc8ee9b6436892ee4ec40c3f3af22698ddf5f43172","mechanical_gate_receipt_digest":"98e053ad26144498e66e661088d4aaddf5f67cd969997c37240abda3d0eee2d8"}
if v.get("execution_mode")!=execution: raise SystemExit("formal execution authority differs")
if v["prompt_matrix"]!={"basename":"mev840_native_rv2v_incremental_prompt_matrix_v1.json","sha256":"5c28e672bcdd86da3c7d3a94ba9e07b644421cea6c5945fb163fa7b871c2af0a","labels":["P0","P1","P2"],"only_registered_design_variable":"positive_prompt_utf8"}: raise SystemExit("prompt matrix reference differs")
if m.get("schema")!="mev840-native-rv2v-incremental-prompt-matrix-v1": raise SystemExit("prompt matrix schema differs")
if v["same_process_pairing"]["execution_order"]!=["p0a","p1","p2","p0b"]: raise SystemExit("execution order differs")
for key in ("same_scheduler_object_all_calls","no_manual_model_or_scheduler_state_reset_between_calls","rope_unregistered_state_observed_not_mutated"):
 if v["same_process_pairing"].get(key) is not True: raise SystemExit("same-process nonmutation gate differs: "+key)
for key in ("positive_tokens_and_embedding_world4_exact_per_cell","negative_tokens_and_embedding_world4_exact_across_cells"):
 if v["dynamic_observer_gates"].get(key) is not True: raise SystemExit("T5 WORLD4 gate differs: "+key)
for key in ("scheduler_effective_reset_fields_exact_across_calls","scheduler_stale_timestep_list_recorded","scheduler_stale_timestep_list_inactive_before_order2_overwrite_proved"):
 if v["dynamic_observer_gates"].get(key) is not True: raise SystemExit("effective scheduler gate differs: "+key)
if "scheduler_schedule_and_reset_state_exact_across_calls" in v["dynamic_observer_gates"]: raise SystemExit("impossible literal scheduler reset gate is present")
pins={"P0":(370,"effdf094385a4f2486391efc008150b7436a8137c1d5766864a678ed6e0c749f",162,"604bff69e9f43990de2efd7c26e64d15b4f1e92d9c165d182c7e2707f9299251"),"P1":(677,"248410295a0dd4226b478bedaa46cd23f0dd4d406d4d262c692c4006f4481aef",231,"ccd07d417c3a11ee698b11a07922a55a9f8c32d5bb40d69fa3d2541b4c7e0e0b"),"P2":(835,"63d4cda9cedca68487cdd9c5c951c2fe63226483d8975487114c221e38d1b4e5",276,"79293cc4c429e4b49734221d86800cc906726535901da2c0e5cc4dce648fbc11")}
for label,(n,d,t,fd) in pins.items():
 r=v["prompts"][label]; b=r["full_prompt_utf8"].encode()
 if len(b)!=n or r["full_prompt_utf8_bytes"]!=n or hashlib.sha256(b).hexdigest()!=d or r["full_prompt_utf8_sha256"]!=d: raise SystemExit("raw prompt differs: "+label)
 if r["untruncated_token_count"]!=t or r["final_task_prompt_utf8_sha256"]!=fd or r["terminal_token_id"]!=1: raise SystemExit("token pin differs: "+label)
 if m["prompts"][label]["full_prompt_utf8"]!=r["full_prompt_utf8"] or m["prompts"][label]["full_prompt_utf8_sha256"]!=d: raise SystemExit("authority/matrix prompt differs: "+label)
g=v["generator_contract"]
if g["accepted_external_conditions"]!=["source_video","positive_prompt_matrix"] or any(g[k] is not False for k in ("target_video_read","target_action_json_read","target_rgb_mask_box_xy_flow_feature_embedding_latent_qkv_gaussian_read","anchor_rgb_kv_latent_gaussian_read","legacy_activity25_qk_read")): raise SystemExit("generator input authority differs")
print("MEV840_SAME_PROCESS_FORMAL_AUTHORITY_OK")
PY
  "${python_bin}" -B - "${control_root}" <<'PY'
from pathlib import Path
import stat,sys
root=Path(sys.argv[1])
static={"auh_launch_mev840_native_rv2v_same_process_formal_v1.sh","infer_mev840_native_rv2v_paired_prompt_matrix_formal_v1.py","mev840_native_rv2v_same_process_formal_v1.json","mev840_native_rv2v_incremental_prompt_matrix_v1.json","audit_mev840_native_rv2v_same_process_formal_v1.py"}
pairs=[{"formal_seed2027.log","formal_seed2027.pid"},{"formal_seed2028.log","formal_seed2028.pid"}]
actual={p.name for p in root.iterdir()}
extras=actual-static
if not static<=actual or any(bool(extras & pair) and not pair<=extras for pair in pairs) or extras-set().union(*pairs): raise SystemExit("control-root exact member closure differs")
for p in root.iterdir():
 s=p.lstat()
 if p.is_symlink() or not stat.S_ISREG(s.st_mode): raise SystemExit("control-root member is not a plain file: "+p.name)
print("MEV840_SAME_PROCESS_CONTROL_ROOT_OK",len(actual))
PY
}

verify_holder() {
  local holder_job="$1" holder_node="$2"
  [[ "$(squeue -h -j "${holder_job}" -o '%T')" == RUNNING ]] || fail "holder is not running: ${holder_job}"
  scontrol show hostnames "$(squeue -h -j "${holder_job}" -o '%N')" | grep -Fqx -- "${holder_node}" || fail "holder/node differs: ${holder_job}/${holder_node}"
}

seed_binding() {
  case "$1" in
    2027) echo "143808 auh7-1b-gpu-292" ;;
    2028) echo "147873 auh7-1b-gpu-284" ;;
    *) fail "formal seed must be 2027 or 2028" ;;
  esac
}

require_seed2027_terminal_postflight_before_seed2028() {
  local first_log="${control_root}/formal_seed2027.log"
  local first_pid="${control_root}/formal_seed2027.pid"
  [[ -f "$first_log" && ! -L "$first_log" && -f "$first_pid" && ! -L "$first_pid" ]] || fail "seed2027 control sidecars are absent"
  [[ "$(grep -Ec '^MEV840_SAME_PROCESS_FORMAL_WORKER_DONE seed=2027 step=143808\.[0-9]+$' "$first_log")" == 1 ]] || fail "seed2027 terminal worker log gate differs"
  "${python_bin}" -B "${postflight}" \
    --output-dir "${output_root}/seed2027" --seed 2027 --authority "$authority" \
    --launcher "${control_root}/auh_launch_mev840_native_rv2v_same_process_formal_v1.sh" \
    --runner "$paired_runner"
}

mode="${1:-}"
case "${mode}" in
  launch-formal)
    [[ $# == 2 ]] || usage
    readonly seed="$2"; read -r holder_job holder_node < <(seed_binding "$seed")
    verify_authority; verify_holder "$holder_job" "$holder_node"
    if [[ "$seed" == 2028 ]]; then require_seed2027_terminal_postflight_before_seed2028; fi
    if [[ ! -e "${output_root}" && ! -L "${output_root}" ]]; then mkdir "${output_root}"; fi
    [[ -d "${output_root}" && ! -L "${output_root}" ]] || fail "formal output root differs"
    [[ ! -e "${output_root}/seed${seed}" && ! -L "${output_root}/seed${seed}" ]] || fail "fresh seed output required"
    readonly log="${control_root}/formal_seed${seed}.log" pid_file="${control_root}/formal_seed${seed}.pid"
    [[ ! -e "$log" && ! -L "$log" && ! -e "$pid_file" && ! -L "$pid_file" ]] || fail "launch sidecar not fresh"
    : >"$pid_file"
    : >"$log"
    nohup srun --jobid="${holder_job}" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:4 --mem=0 --nodelist="${holder_node}" \
      bash "${control_root}/auh_launch_mev840_native_rv2v_same_process_formal_v1.sh" worker-formal "$seed" >"$log" 2>&1 &
    printf '%s\n' "$!" >"$pid_file"
    echo "MEV840_SAME_PROCESS_FORMAL_LAUNCHED seed=${seed}"
    ;;
  worker-formal)
    [[ $# == 2 ]] || usage
    readonly seed="$2"; read -r holder_job holder_node < <(seed_binding "$seed")
    [[ "${SLURM_JOB_ID:-}" == "${holder_job}" && "$(hostname -s)" == "${holder_node}" && "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "formal Slurm authority differs"
    verify_authority
    [[ -d "${output_root}" && ! -L "${output_root}" && ! -e "${output_root}/seed${seed}" && ! -L "${output_root}/seed${seed}" ]] || fail "fresh seed output required"
    scratch_parent="${SLURM_TMPDIR:-/tmp}"; [[ "$scratch_parent" == /* && "$scratch_parent" != / && -d "$scratch_parent" && ! -L "$scratch_parent" && -w "$scratch_parent" ]] || fail "scratch parent differs"
    scratch="$(mktemp -d "${scratch_parent%/}/mev840-formal-${seed}-${SLURM_STEP_ID}.XXXXXXXX")"
    cleanup() { local s=$?; trap - EXIT INT TERM HUP; find "$scratch" -xdev -depth -mindepth 1 -delete || s=70; rmdir "$scratch" || s=70; exit "$s"; }; trap cleanup EXIT INT TERM HUP
    mkdir "$scratch/runtime" "$scratch/cache" "$scratch/tmp"
    tar -xf "${runtime_archive}" -C "$scratch/runtime" --no-same-owner
    verify_release "$scratch/runtime"
    readonly runtime_tree="$scratch/runtime" runtime_method="$scratch/runtime/methods/bernini_action_editing"
    cp -- "${paired_runner}" "${runtime_tree}/${paired_runner_rel}"
    verify_file "${runtime_tree}/${paired_runner_rel}" "${paired_runner_sha}"
    "${python_bin}" -B - "${runtime_method}" "${runtime_manifest}" "${paired_runner_rel}" "${paired_runner_sha}" <<'PY'
from pathlib import Path
import hashlib,json,stat,sys
root,manifest,runner_rel,runner_sha=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4]
expected={r["path"]:r["sha256"] for r in json.loads(manifest.read_text(encoding="ascii"))["files"]}; expected[Path(runner_rel).name]=runner_sha
actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
if actual!=set(expected): raise SystemExit("authorized exact20 closure differs")
for rel,want in expected.items():
 p=root/rel; s=p.lstat()
 if p.is_symlink() or not stat.S_ISREG(s.st_mode) or hashlib.sha256(p.read_bytes()).hexdigest()!=want: raise SystemExit("authorized file differs: "+rel)
print("MEV840_SAME_PROCESS_EXACT20_OK")
PY
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
    "${python_bin}" -B -c 'import py_compile,sys; py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)' "${runtime_tree}/${paired_runner_rel}" "$scratch/cache/runner.pyc"
    model_load_lock="$scratch/renderer-load.lock"; : >"$model_load_lock"; chmod 0400 "$model_load_lock"
    export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1 NATIVE_V_AXIS_LOAD_LOCK="$model_load_lock"
    unset NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1
    export PYTHONPATH="${runtime_method}" TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache/xdg" TORCH_EXTENSIONS_DIR="$scratch/cache/torch-extensions" TRITON_CACHE_DIR="$scratch/cache/triton"
    export MIOPEN_USER_DB_PATH="$scratch/cache/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/cache/miopen-custom"
    mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
    "${python_bin}" -B -m torch.distributed.run --standalone --nproc_per_node=4 "${runtime_tree}/${paired_runner_rel}" \
      --bernini-root "$bernini_root" --veomni-root "$veomni_root" --checkpoint "$checkpoint" --checkpoint-content-manifest "$checkpoint_manifest" \
      --source-video "$source_video" --expected-source-sha256 "$source_sha" --prompt-matrix-authority "$authority" --expected-prompt-matrix-authority-sha256 "$authority_sha" \
      --output-dir "${output_root}/seed${seed}" --num-inference-steps 40 --seed "$seed" \
      --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793 --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d \
      --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca \
      --method-source-revision "$content_revision" --method-source-archive-sha256 "$runtime_archive_sha"
    echo "MEV840_SAME_PROCESS_FORMAL_WORKER_DONE seed=${seed} step=${SLURM_JOB_ID}.${SLURM_STEP_ID}"
    ;;
  postflight-formal)
    [[ $# == 2 ]] || usage
    readonly seed="$2"; seed_binding "$seed" >/dev/null; verify_authority
    "${python_bin}" -B "${postflight}" \
      --output-dir "${output_root}/seed${seed}" --seed "$seed" --authority "$authority" --launcher "${control_root}/auh_launch_mev840_native_rv2v_same_process_formal_v1.sh" --runner "$paired_runner"
    ;;
  *) usage ;;
esac

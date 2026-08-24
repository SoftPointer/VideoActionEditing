#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Fresh paired P0/P1/P2 native RV2V matrix.  Each seed stays on one node and
# waves run sequentially.  Every worker extracts the pinned release tar into
# node-local scratch; the shared release runtime is never imported or written.

fail() { echo "[mev840-native-matrix] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }

readonly stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
readonly mev_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v1
readonly control_root="${stage}/mev840_native_rv2v_incremental_prompt_matrix_v1_20260822_control"
readonly output_root="${stage}/mev840_native_rv2v_incremental_prompt_matrix_v1_20260822"
readonly release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/generic-action-confirmation40-generation-r3-ac22e19f-r1
readonly runtime_archive="${release_root}/source.tar"
readonly runtime_manifest="${release_root}/source.manifest.json"
readonly runtime_archive_sha=46ae7529d640a197006ab8d7d17c23ac81925dabd7fa1caf4b0bb261197e8115
readonly runtime_manifest_sha=e104031526236f16e94a4753c31ad8048b1a65345b1913212c35e421fcad48ae
readonly runtime_manifest_digest=4e78a935b2485e3f8c2c94aa5524a82ed25aa0b93aaf58dd81476dc5c9b48044
readonly content_revision=ac22e19ffd109a2d6b85c32c64463b0be8373792
readonly runner_rel=methods/bernini_action_editing/infer_native_identity_generation_canary.py
readonly runner_sha=bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42

readonly matrix_authority="${control_root}/mev840_native_rv2v_incremental_prompt_matrix_v1.json"
readonly matrix_authority_sha=5c28e672bcdd86da3c7d3a94ba9e07b644421cea6c5945fb163fa7b871c2af0a
readonly p1_payload="${control_root}/mev840_action_only_p1_event_order_v1.txt"
readonly p1_payload_sha=2a334405d892434b8855d1a652c577c6caedf9bf63e1e0698ee4cd1973dd994b
readonly p2_payload="${control_root}/mev840_action_only_p2_relation_contact_v1.txt"
readonly p2_payload_sha=e22733bd003e77b0a914ce8a3a15f3b850285f7d721a04a8a51d81c1920e3f34

readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly source_video="${mev_root}/preprocessed_sources/840b214afead/source-exact81.mp4"
readonly source_sha=a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646

usage() { echo "usage: $0 launch-p0|launch-p1|launch-p2|worker P0|P1|P2 2027|2028|compare" >&2; exit 2; }
verify_file() { [[ -f "$1" && ! -L "$1" ]] || fail "plain file missing: $1"; [[ "$(sha256_file "$1")" == "$2" ]] || fail "SHA differs: $1"; }

verify_release() {
  local extracted="${1:-}"
  "${python_bin}" -B - "${runtime_archive}" "${runtime_manifest}" "${runtime_manifest_digest}" "${extracted}" <<'PY'
from pathlib import Path, PurePosixPath
import hashlib,json,stat,sys,tarfile
a,m,digest,root=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4]
v=json.loads(m.read_text(encoding="ascii")); unsigned=dict(v); declared=unsigned.pop("manifest_digest",None)
raw=json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
if declared != digest or hashlib.sha256(raw).hexdigest()!=digest: raise SystemExit("manifest digest differs")
if v.get("file_count")!=19 or v.get("content_closure_sha1")!="ac22e19ffd109a2d6b85c32c64463b0be8373792": raise SystemExit("release identity differs")
if "infer_native_identity_generation_canary.py" in v.get("allowed_entrypoints",[]): raise SystemExit("unexpected upstream entrypoint authorization")
expected={r["path"]:r["sha256"] for r in v["files"]}
if len(expected)!=19 or expected.get("infer_native_identity_generation_canary.py")!="bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42": raise SystemExit("manifest path closure differs")
seen={}
with tarfile.open(a,"r:") as h:
 for member in h.getmembers():
  p=PurePosixPath(member.name)
  if p.is_absolute() or ".." in p.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo(): raise SystemExit("unsafe archive member")
  if not member.isfile(): continue
  try: rel=p.relative_to(PurePosixPath("methods/bernini_action_editing")).as_posix()
  except ValueError: raise SystemExit("archive escaped method root")
  seen[rel]=hashlib.sha256(h.extractfile(member).read()).hexdigest()
if seen != expected: raise SystemExit("archive exact closure differs")
if root:
 base=Path(root)/"methods/bernini_action_editing"
 actual={p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() or p.is_symlink()}
 if actual != set(expected): raise SystemExit("extracted exact member set differs")
 for rel,want in expected.items():
  p=base/rel; s=p.lstat()
  if p.is_symlink() or not stat.S_ISREG(s.st_mode) or hashlib.sha256(p.read_bytes()).hexdigest()!=want: raise SystemExit("extracted file differs: "+rel)
print("MEV840_MATRIX_RELEASE_OK",len(expected),"scratch" if root else "archive")
PY
}

verify_authority() {
  verify_file "${runtime_archive}" "${runtime_archive_sha}"
  verify_file "${runtime_manifest}" "${runtime_manifest_sha}"
  verify_file "${matrix_authority}" "${matrix_authority_sha}"
  verify_file "${p1_payload}" "${p1_payload_sha}"
  verify_file "${p2_payload}" "${p2_payload_sha}"
  verify_file "${python_bin}" "${python_sha}"
  verify_file "${checkpoint_manifest}" "${checkpoint_manifest_sha}"
  verify_file "${source_video}" "${source_sha}"
  for p in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do [[ -d "$p" && ! -L "$p" ]] || fail "directory authority differs: $p"; done
  verify_release
  "${python_bin}" -B - "${matrix_authority}" "${p1_payload}" "${p2_payload}" <<'PY'
import hashlib,json,sys
m=json.load(open(sys.argv[1],encoding="ascii")); p1=open(sys.argv[2],"rb").read(); p2=open(sys.argv[3],"rb").read()
if p1[-1:]!=b"\n" or p1.count(b"\n")!=1 or p2[-1:]!=b"\n" or p2.count(b"\n")!=1: raise SystemExit("payload LF closure differs")
c=m["common"]; prompts=m["prompts"]
for key in ("source_context","base_action","preservation_suffix"):
 b=c[key+"_utf8"].encode();
 if hashlib.sha256(b).hexdigest()!=c[key+"_utf8_sha256"]: raise SystemExit("common prompt factor differs")
for label,payload,cue in (("P0",b"",b""),("P1",p1[:-1],b" Follow this event order: "),("P2",p2[:-1],b" Follow these contact relations: ")):
 full=c["source_context_utf8"].encode()+c["base_action_utf8"].encode()+cue+payload+c["preservation_suffix_utf8"].encode(); row=prompts[label]
 if full.decode()!=row["full_prompt_utf8"] or len(full)!=row["full_prompt_utf8_bytes"] or hashlib.sha256(full).hexdigest()!=row["full_prompt_utf8_sha256"]: raise SystemExit("full prompt differs: "+label)
print("MEV840_MATRIX_PROMPTS_OK")
PY
}

verify_holder() {
  [[ "$(squeue -h -j "$1" -o '%T')" == RUNNING ]] || fail "holder $1 is not running"
  scontrol show hostnames "$(squeue -h -j "$1" -o '%N')" | grep -Fqx -- "$2" || fail "holder/node differs"
}

cell_complete() { [[ -f "${output_root}/$1/seed$2/candidate.complete.json" && ! -L "${output_root}/$1/seed$2/candidate.complete.json" ]]; }

launch_wave() {
  local arm="$1" seed job node log pid_file arm_root
  verify_authority
  case "$arm" in
    P0) [[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "fresh P0 output root required"; mkdir "${output_root}" ;;
    P1) for seed in 2027 2028; do cell_complete P0 "$seed" || fail "P0 incomplete"; done ;;
    P2) for seed in 2027 2028; do cell_complete P0 "$seed" && cell_complete P1 "$seed" || fail "prior waves incomplete"; done ;;
    *) usage ;;
  esac
  arm_root="${output_root}/${arm}"
  [[ ! -e "$arm_root" && ! -L "$arm_root" ]] || fail "fresh arm root required"
  mkdir "$arm_root"
  for seed in 2027 2028; do
    if [[ "$seed" == 2027 ]]; then job=143808; node=auh7-1b-gpu-292; else job=147873; node=auh7-1b-gpu-284; fi
    verify_holder "$job" "$node"; cell_complete "$arm" "$seed" && fail "cell already complete"
    [[ ! -e "${output_root}/${arm}/seed${seed}" && ! -L "${output_root}/${arm}/seed${seed}" ]] || fail "cell path not fresh"
    log="${control_root}/${arm}_seed${seed}.log"; pid_file="${control_root}/${arm}_seed${seed}.pid"
    [[ ! -e "$log" && ! -L "$log" && ! -e "$pid_file" && ! -L "$pid_file" ]] || fail "launch sidecar not fresh"
    nohup srun --jobid="$job" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:4 --mem=0 --nodelist="$node" \
      bash "${control_root}/auh_launch_mev840_native_rv2v_incremental_prompt_matrix_v1.sh" worker "$arm" "$seed" >"$log" 2>&1 &
    printf '%s\n' "$!" >"$pid_file"
  done
  echo "MEV840_MATRIX_WAVE_LAUNCHED ${arm}"
}

mode="${1:-}"
case "$mode" in
  launch-p0) [[ $# == 1 ]] || usage; launch_wave P0 ;;
  launch-p1) [[ $# == 1 ]] || usage; launch_wave P1 ;;
  launch-p2) [[ $# == 1 ]] || usage; launch_wave P2 ;;
  worker)
    [[ $# == 3 ]] || usage; readonly arm="$2" seed="$3"
    case "$arm" in P0|P1|P2) ;; *) usage ;; esac
    case "$seed" in 2027) readonly expected_job=143808 expected_node=auh7-1b-gpu-292 ;; 2028) readonly expected_job=147873 expected_node=auh7-1b-gpu-284 ;; *) usage ;; esac
    [[ "${SLURM_JOB_ID:-}" == "$expected_job" && "$(hostname -s)" == "$expected_node" && "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "Slurm authority differs"
    verify_authority
    readonly candidate_dir="${output_root}/${arm}/seed${seed}"
    [[ ! -e "$candidate_dir" && ! -L "$candidate_dir" ]] || fail "candidate path not fresh"
    scratch_parent="${SLURM_TMPDIR:-/tmp}"; [[ "$scratch_parent" == /* && "$scratch_parent" != / && -d "$scratch_parent" && ! -L "$scratch_parent" && -w "$scratch_parent" ]] || fail "scratch parent differs"
    scratch="$(mktemp -d "${scratch_parent%/}/mev840-matrix-${arm}-${seed}-${SLURM_STEP_ID}.XXXXXXXX")"
    cleanup() { local s=$?; trap - EXIT INT TERM HUP; find "$scratch" -xdev -depth -mindepth 1 -delete || s=70; rmdir "$scratch" || s=70; exit "$s"; }; trap cleanup EXIT INT TERM HUP
    mkdir "$scratch/runtime" "$scratch/cache" "$scratch/tmp"
    tar -xf "${runtime_archive}" -C "$scratch/runtime" --no-same-owner
    verify_release "$scratch/runtime"
    readonly runtime_tree="$scratch/runtime"
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
    "${python_bin}" -B -c 'import py_compile,sys; py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)' "${runtime_tree}/${runner_rel}" "$scratch/cache/runner.pyc"
    action_prompt="$("${python_bin}" -B -c 'import json,sys; sys.stdout.write(json.load(open(sys.argv[1],encoding="ascii"))["prompts"][sys.argv[2]]["full_prompt_utf8"])' "${matrix_authority}" "$arm")"
    action_prompt_sha="$(printf '%s' "$action_prompt" | sha256sum | awk '{print $1}')"
    expected_prompt_sha="$("${python_bin}" -B -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="ascii"))["prompts"][sys.argv[2]]["full_prompt_utf8_sha256"])' "${matrix_authority}" "$arm")"
    [[ "$action_prompt_sha" == "$expected_prompt_sha" ]] || fail "rendered prompt differs"

    model_load_lock="$scratch/renderer-load.lock"; : >"$model_load_lock"; chmod 0400 "$model_load_lock"
    export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1 NATIVE_V_AXIS_LOAD_LOCK="$model_load_lock"
    unset NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1
    export PYTHONPATH="${runtime_tree}/methods/bernini_action_editing" TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache/xdg" TORCH_EXTENSIONS_DIR="$scratch/cache/torch-extensions" TRITON_CACHE_DIR="$scratch/cache/triton"
    export MIOPEN_USER_DB_PATH="$scratch/cache/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/cache/miopen-custom"
    mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
    "${python_bin}" -B -m torch.distributed.run --standalone --nproc_per_node=4 "${runtime_tree}/${runner_rel}" \
      --bernini-root "$bernini_root" --veomni-root "$veomni_root" --checkpoint "$checkpoint" --checkpoint-content-manifest "$checkpoint_manifest" \
      --source-video "$source_video" --expected-source-sha256 "$source_sha" --action-prompt "$action_prompt" --expected-action-prompt-sha256 "$action_prompt_sha" \
      --output-dir "$candidate_dir" --arms rv2v --num-inference-steps 40 --seed "$seed" \
      --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793 --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d \
      --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca \
      --method-source-revision "$content_revision" --method-source-archive-sha256 "$runtime_archive_sha"

    receipt="$candidate_dir/receipt.json"; video="$candidate_dir/rv2v.mp4"; [[ -f "$receipt" && ! -L "$receipt" && -f "$video" && ! -L "$video" ]] || fail "output closure incomplete"
    "${python_bin}" -B - "$receipt" "$arm" "$seed" "$action_prompt_sha" "$source_sha" <<'PY'
import hashlib,json,sys
p,arm,seed,prompt_sha,source_sha=sys.argv[1:]; v=json.load(open(p,encoding="utf-8")); declared=v.get("receipt_digest"); u=dict(v); u.pop("receipt_digest",None)
if hashlib.sha256(json.dumps(u,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest()!=declared: raise SystemExit("receipt digest differs")
i=v.get("input",{}); s=v.get("sampling",{}).get("rv2v",{}); n=v.get("initial_noise_artifacts",{}).get("rv2v",{}); f=v.get("freeze_certificate",{})
if v.get("arms") != ["rv2v"] or i.get("accepted_external_conditions") != ["source_video","action_prompt"]: raise SystemExit("input allowlist differs")
if i.get("source_video_sha256")!=source_sha or i.get("action_prompt_utf8_sha256")!=prompt_sha: raise SystemExit("input identity differs")
if i.get("target_video") is not False or i.get("external_first_frame_anchor") is not False or i.get("external_mask_flow_pose_track_trajectory") is not False or i.get("external_reference_image_or_video") is not False: raise SystemExit("external input admitted")
if s.get("seed")!=int(seed) or s.get("num_inference_steps")!=40 or s.get("guidance_mode")!="rv2v" or s.get("custom_sampler_or_scheduler") is not False or s.get("target_initialization")!="official_gen_wanx22_fresh_gaussian" or s.get("target_mixed_with_source_latent") is not False: raise SystemExit("native sampling differs")
if n.get("external_initial_noise_injection") is not False or n.get("source_or_target_derived") is not False or n.get("sampler_noise_replacement") is not False: raise SystemExit("Gaussian authority differs")
if f != {"base_frozen":True,"lora_module_count":0,"trainable_parameter_elements":0,"trainable_parameter_tensors":0}: raise SystemExit("freeze differs")
if v.get("method_source_archive_sha256")!="46ae7529d640a197006ab8d7d17c23ac81925dabd7fa1caf4b0bb261197e8115": raise SystemExit("method archive differs")
o=v.get("outputs",{}).get("rv2v",{}); 
if o.get("frame_count")!=81 or o.get("fps")!=25 or o.get("height")!=368 or o.get("width")!=656: raise SystemExit("media metadata differs")
PY
    video_sha="$(sha256_file "$video")"; receipt_sha="$(sha256_file "$receipt")"; complete="$candidate_dir/candidate.complete.json"
    "${python_bin}" -B - "$complete" "$arm" "$seed" "$expected_job" "$expected_node" "${SLURM_STEP_ID}" "$action_prompt_sha" "$video_sha" "$receipt_sha" <<'PY'
import json,os,sys,tempfile
p,arm,seed,job,node,step,prompt_sha,video_sha,receipt_sha=sys.argv[1:]
v={"schema":"mev840-native-rv2v-matrix-candidate-complete-v1","complete":True,"arm":arm,"seed":int(seed),"slurm":{"job_id":job,"step_id":step,"node":node},"prompt_utf8_sha256":prompt_sha,"video_sha256":video_sha,"native_receipt_sha256":receipt_sha,"zero_update":True,"generator_target_video_read":False,"generator_target_action_json_read":False,"generator_target_rgb_feature_embedding_latent_qkv_gaussian_read":False,"upstream_release_entrypoint_authorized":False,"immutable_release_bytes_reused_under_current_user_authorized_launcher":True,"runtime_extracted_to_node_local_scratch_exact19":True}
raw=json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")+b"\n"; fd,tmp=tempfile.mkstemp(prefix=".complete-",dir=os.path.dirname(p)); os.write(fd,raw); os.fsync(fd); os.close(fd); os.replace(tmp,p)
PY
    echo "MEV840_MATRIX_CELL_COMPLETE arm=${arm} seed=${seed} step=${SLURM_JOB_ID}.${SLURM_STEP_ID} video_sha=${video_sha}"
    ;;
  compare)
    [[ $# == 1 ]] || usage; verify_authority
    "${python_bin}" -B - "${output_root}" "${matrix_authority}" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]); auth=json.load(open(sys.argv[2],encoding="ascii")); rows=[]; cores=[]; noises={2027:[],2028:[]}; sources={2027:[],2028:[]}
for arm in ("P0","P1","P2"):
 for seed in (2027,2028):
  d=root/arm/f"seed{seed}"; c=json.load(open(d/"candidate.complete.json",encoding="ascii")); rpath=d/"receipt.json"; v=json.load(open(rpath,encoding="utf-8")); video=d/"rv2v.mp4"
  if c["arm"]!=arm or c["seed"]!=seed or c["prompt_utf8_sha256"]!=auth["prompts"][arm]["full_prompt_utf8_sha256"]: raise SystemExit("cell identity differs")
  if hashlib.sha256(video.read_bytes()).hexdigest()!=c["video_sha256"] or hashlib.sha256(rpath.read_bytes()).hexdigest()!=c["native_receipt_sha256"]: raise SystemExit("artifact SHA differs")
  i=v["input"]; s=v["sampling"]["rv2v"]; n=v["initial_noise_artifacts"]["rv2v"]
  core={"schema":v["schema_version"],"method":v["method"],"commits":[v["bernini_commit"],v["veomni_commit"]],"inference_files":v["bernini_inference_files"],"checkpoint":v["checkpoint"],"archive":[v["method_source_archive_sha256"],v["method_source_revision"]],"arms":v["arms"],"input":{"accepted":i["accepted_external_conditions"],"source":i["source_video_sha256"],"target":i["target_video"],"anchor":i["external_first_frame_anchor"],"controls":i["external_mask_flow_pose_track_trajectory"],"reference":i["external_reference_image_or_video"]},"freeze":v["freeze_certificate"],"conditioning":v["conditioning"],"preprocessing":v["preprocessing"],"latent_geometry":v["latent_geometry"],"runtime_versions":v["runtime_versions"],"sampling":{k:x for k,x in s.items() if k!="seed"}}
  cores.append(hashlib.sha256(json.dumps(core,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest())
  noises[seed].append((n["content_sha256"],n["raw_value_sha256"],n["tensor_value_sha256"],tuple(n["shape"])))
  identities=v["condition_identities"]; q=identities["full_source_video"]["identity"]
  ordered=[("full_source_video",q["content_sha256"],q["raw_storage_sha256"],tuple(q["shape"]),q["dtype"])]
  for index in ("0","27","53","80"):
   q=identities["references"][index]["identity"]; ordered.append((index,q["content_sha256"],q["raw_storage_sha256"],tuple(q["shape"]),q["dtype"]))
  sources[seed].append(tuple(ordered))
  rows.append({"arm":arm,"seed":seed,"node":c["slurm"]["node"],"step_id":c["slurm"]["step_id"],"prompt_sha256":c["prompt_utf8_sha256"],"video_sha256":c["video_sha256"],"receipt_sha256":c["native_receipt_sha256"],"receipt_digest":v["receipt_digest"]})
if len(set(cores))!=1: raise SystemExit("canonical native configuration differs")
for seed in (2027,2028):
 if len(set(noises[seed]))!=1: raise SystemExit(f"same-seed Gaussian differs: {seed}")
 if len(set(sources[seed]))!=1: raise SystemExit(f"same-seed source condition differs: {seed}")
out={"schema":"mev840-native-rv2v-paired-matrix-comparison-v1","complete":True,"canonical_configuration_sha256":cores[0],"same_seed_gaussian_exact":True,"same_seed_source_condition_exact":True,"cells":rows,"generator_target_reads":False,"upstream_release_entrypoint_authorized":False}
raw=json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")+b"\n"; path=root/"matrix.comparison.json"; path.write_bytes(raw); print("MEV840_MATRIX_COMPARE_OK",hashlib.sha256(raw).hexdigest())
PY
    ;;
  *) usage ;;
esac

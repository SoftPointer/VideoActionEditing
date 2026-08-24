#!/usr/bin/env bash
# Exact WORLD4/SP4 worker for the job141620 Full644 R64 capacity gate and run.

set -Eeuo pipefail
umask 077

readonly expected_job=141620
readonly expected_node=auh7-1b-gpu-226
readonly expected_owner=guangyi.chen
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha256=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly python_size=31490256
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_r64_job141620_v1
readonly launcher_root="${experiment_root}/launchers/full644-exploratory-r64-job141620-v1"
readonly helper="${launcher_root}/full644_exploratory_r64_release_v1.py"
readonly helper_sha256=3bfa6514b043b7a84edb208b1f9443dd304eb7e60c62697fa92fbb0871569446
readonly helper_size=66070
readonly release_root="${experiment_root}/releases/full644-exploratory-r64-source-95a8aadf6278"
readonly trainer="${release_root}/methods/bernini_action_editing/train_lora.py"
readonly trainer_sha256=8e8daf422548bc29e2c18f2d2c692af2dd3109aaad1897fc31e590a69d7e593e
readonly trainer_size=124045
readonly method_source_revision=f6dbb5c31b4550c8bae495c39a110ebe0b00bf15
readonly method_source_archive_sha256=95a8aadf627805bc3f08dd90b651274398af42b9328b8e2be622d4ce271d4a82
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly parquet_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/data/vae_full_81f_4d41e4c/shards
readonly dataset_summary=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/data/vae_full_81f_4d41e4c/dataset_summary.json
readonly dataset_summary_sha256=5dc45b4a6d700b3cd0108e941242ae364396458f20f41249744e74e00acc02dd
readonly dataset_index=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/data/vae_full_81f_4d41e4c/dataset_index.jsonl
readonly dataset_index_sha256=d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b
readonly source_authority="${release_root}/md/action_editing/20260814_man/evidence/stage_r64_joint_136309_r2/run_receipt.json"
readonly source_authority_sha256=0bcf24ce8aafabb37cf38eafe9da6b13c70043bb0f4c3146f16dc0bafd35618f
readonly source_authority_size=302520
readonly expected_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly expected_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly expected_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly capacity_output="${experiment_root}/runs/prelaunch-capacity-sft-r64-step1-v1"
readonly full644_output="${experiment_root}/runs/full644-r64-reference-dpo-preservation-one-pass-v1"
readonly attempt_root="${experiment_root}/attempts/prelaunch-capacity-then-full644-v1"
readonly capacity_completion="${attempt_root}/capacity-runner-completion.json"
readonly full644_completion="${attempt_root}/full644-runner-completion.json"
readonly capacity_cache_receipt="${attempt_root}/capacity-rank-cache.json"
readonly full644_cache_receipt="${attempt_root}/full644-rank-cache.json"
readonly capacity_step_record="${attempt_root}/capacity-step-id.txt"
readonly full644_step_record="${attempt_root}/full644-step-id.txt"

fail() {
  printf 'Full644 job141620 worker refused: %s\n' "$*" >&2
  exit 91
}

sha256_file() {
  /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'
}

require_plain_file_sha() {
  local path="$1" expected_sha="$2" expected_size="$3" expected_mode="$4" label="$5"
  local observed_sha observed_size observed_envelope
  [[ -f "${path}" && ! -L "${path}" ]] || fail "${label} is not a plain file"
  observed_size="$(/usr/bin/stat -c %s "${path}")" || fail "${label} size query failed"
  [[ "${observed_size}" == "${expected_size}" ]] || fail "${label} size differs"
  observed_envelope="$(/usr/bin/stat -c '%a:%h:%U' "${path}")" || fail "${label} envelope query failed"
  [[ "${observed_envelope}" == "${expected_mode}:1:${expected_owner}" ]] || fail "${label} mode/link/owner differs"
  observed_sha="$(sha256_file "${path}")" || fail "${label} SHA query failed"
  [[ "${observed_sha}" == "${expected_sha}" ]] || fail "${label} SHA differs"
}

output_for_mode() {
  case "$1" in
    PRELAUNCH_CAPACITY_ONLY) printf '%s\n' "${capacity_output}" ;;
    FULL644_EXPLORATORY) printf '%s\n' "${full644_output}" ;;
    *) fail "unknown training mode: $1" ;;
  esac
}

require_current_step_closure() {
  local label="$1" rows line step_id step_node step_user extra count=0 batch=0 extern=0 current=0
  rows="$(/usr/bin/timeout 15 /usr/bin/squeue --steps -h -j "${expected_job}" -o '%i|%N|%u' | \
    /usr/bin/awk '{gsub(/[[:space:]]/,""); if(length($0)) print $0}')" || \
    fail "${label}: step closure query failed"
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    (( count += 1 ))
    IFS='|' read -r step_id step_node step_user extra <<EOF
${line}
EOF
    [[ -z "${extra:-}" && "${step_node}" == "${expected_node}" && "${step_user}" == "${expected_owner}" ]] || \
      fail "${label}: hostile step row: ${line}"
    case "${step_id}" in
      "${expected_job}.batch") batch=1 ;;
      "${expected_job}.extern") extern=1 ;;
      "${expected_job}.${SLURM_STEP_ID}") current=1 ;;
      *) fail "${label}: sibling or malformed step row: ${line}" ;;
    esac
  done <<EOF
${rows}
EOF
  [[ "${count}" == 3 && "${batch}" == 1 && "${extern}" == 1 && "${current}" == 1 ]] || \
    fail "${label}: exact batch/extern/current closure differs"
}

prepare_rank_caches() {
  local mode="$1" effective_uid filesystem_type cache_root receipt_path
  effective_uid="$(/usr/bin/id -u)" || fail "effective uid query failed"
  filesystem_type="$(/usr/bin/stat -f -c %T /tmp)" || fail "node-local filesystem query failed"
  case "${filesystem_type}" in
    nfs*|lustre|gpfs|cifs|smb*|fuse*|autofs) fail "/tmp is not a permitted node-local filesystem: ${filesystem_type}" ;;
  esac
  [[ -d /tmp && ! -L /tmp ]] || fail "/tmp identity differs"
  cache_root="/tmp/cache/full644-r64-u${effective_uid}-j${expected_job}-s${SLURM_STEP_ID}-v1"
  case "${mode}" in
    PRELAUNCH_CAPACITY_ONLY) receipt_path="${capacity_cache_receipt}" ;;
    FULL644_EXPLORATORY) receipt_path="${full644_cache_receipt}" ;;
    *) fail "cache mode differs" ;;
  esac
  "${python_bin}" -I -B - "${cache_root}" "${receipt_path}" "${filesystem_type}" \
    "${mode}" "${expected_job}" "${SLURM_STEP_ID}" "${expected_node}" \
    "${scheduler_tmpdir_observed}" <<'PY'
import hashlib,json,os,stat,sys
from pathlib import Path

root=Path(sys.argv[1]); receipt=Path(sys.argv[2]); fs_type,mode,job,step,node,scheduler_tmpdir=sys.argv[3:]
assert scheduler_tmpdir in ("absent","/tmp")
uid=os.geteuid(); tmp=Path("/tmp"); base=Path("/tmp/cache")
tmp_info=tmp.lstat()
assert stat.S_ISDIR(tmp_info.st_mode) and not tmp.is_symlink()
try:
    os.mkdir(base,0o700)
except FileExistsError:
    pass
base_info=base.lstat()
assert stat.S_ISDIR(base_info.st_mode) and not base.is_symlink()
assert base_info.st_dev==tmp_info.st_dev and base_info.st_uid in (0,uid)
base_mode=stat.S_IMODE(base_info.st_mode)
assert (base_info.st_uid==uid and base_mode==0o700) or (base_info.st_uid==0 and base_mode in (0o755,0o1777))
flags=os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)
base_fd=os.open(base,flags)
try:
    opened_base=os.fstat(base_fd)
    assert (opened_base.st_dev,opened_base.st_ino)==(base_info.st_dev,base_info.st_ino)
    os.mkdir(root.name,0o700,dir_fd=base_fd)
    root_fd=os.open(root.name,flags,dir_fd=base_fd)
    try:
        root_info=os.fstat(root_fd)
        assert stat.S_ISDIR(root_info.st_mode) and stat.S_IMODE(root_info.st_mode)==0o700
        assert root_info.st_uid==uid and root_info.st_dev==tmp_info.st_dev
        rank_rows=[]
        leaves=("extensions","hf","home","inductor","miopen-custom","miopen-user","pycache","tmp","torch","triton","xdg")
        for rank in range(4):
            rank_name="rank-%d"%rank
            os.mkdir(rank_name,0o700,dir_fd=root_fd)
            rank_fd=os.open(rank_name,flags,dir_fd=root_fd)
            try:
                rank_info=os.fstat(rank_fd)
                assert stat.S_IMODE(rank_info.st_mode)==0o700 and rank_info.st_uid==uid
                for leaf in leaves:
                    os.mkdir(leaf,0o700,dir_fd=rank_fd)
                assert set(os.listdir(rank_fd))==set(leaves)
                rank_rows.append({"rank":rank,"path":str(root/rank_name),"device":rank_info.st_dev,
                                  "inode":rank_info.st_ino,"uid":rank_info.st_uid,"mode":"0700"})
            finally:
                os.close(rank_fd)
        assert set(os.listdir(root_fd))=={"rank-0","rank-1","rank-2","rank-3"}
    finally:
        os.close(root_fd)
finally:
    os.close(base_fd)
named=root.lstat()
assert stat.S_ISDIR(named.st_mode) and not root.is_symlink()
assert (named.st_dev,named.st_ino)==(root_info.st_dev,root_info.st_ino)
payload={"schema_version":"full644-r64-rank-cache-receipt-v1","mode":mode,"job_id":job,
         "step_id":step,"node":node,"filesystem_type":fs_type,"cache_root":str(root),
         "cache_root_device":root_info.st_dev,"cache_root_inode":root_info.st_ino,
         "cache_root_uid":uid,"cache_root_mode":"0700","rank_caches":rank_rows,
         "world_size":4,"rank_local":True,
         "scheduler_tmpdir_observed":scheduler_tmpdir,
         "scheduler_tmpdir_normalized_to_unset":True}
payload["receipt_digest"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest()
raw=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")+b"\n"
fd=os.open(receipt,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o400)
with os.fdopen(fd,"wb") as handle:
    view=memoryview(raw)
    while view:
        count=handle.write(view); assert count>0; view=view[count:]
    handle.flush(); os.fsync(handle.fileno())
assert receipt.read_bytes()==raw
PY
  export FULL644_RANK_CACHE_ROOT="${cache_root}"
}

publish_step_record() {
  local mode="$1" expected_record token
  case "${mode}" in
    PRELAUNCH_CAPACITY_ONLY) expected_record="${capacity_step_record}" ;;
    FULL644_EXPLORATORY) expected_record="${full644_step_record}" ;;
    *) fail "step-record mode differs" ;;
  esac
  [[ "${FULL644_STEP_ID_RECORD:-}" == "${expected_record}" ]] || fail "step-record authority differs"
  token="${expected_job}.${SLURM_STEP_ID}"
  "${python_bin}" -I -B - "${expected_record}" "${token}" <<'PY'
import os,re,stat,sys
path,token=sys.argv[1:]
assert re.fullmatch(r"141620[.][0-9]+",token)
raw=(token+"\n").encode("ascii")
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o400)
with os.fdopen(fd,"wb") as handle:
    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
info=os.lstat(path)
assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode)==0o400 and info.st_nlink==1
with open(path,"rb") as handle: assert handle.read()==raw
PY
}

validate_output() {
  local mode="$1" output checkpoint_step expected_step expected_objective helper_mode
  output="$(output_for_mode "${mode}")" || fail "validator output resolution failed"
  case "${mode}" in
    PRELAUNCH_CAPACITY_ONLY)
      checkpoint_step=checkpoint-00000001; expected_step=1; expected_objective=sft; helper_mode=capacity-smoke ;;
    FULL644_EXPLORATORY)
      checkpoint_step=checkpoint-00000644; expected_step=644; expected_objective=reference_dpo_preservation; helper_mode=full644 ;;
  esac
  "${python_bin}" -I -B - "${mode}" "${output}" "${checkpoint_step}" \
    "${expected_step}" "${expected_objective}" "${method_source_revision}" \
    "${method_source_archive_sha256}" <<'PY'
import hashlib,json,os,re,stat,sys
from pathlib import Path

mode,output,checkpoint_name,step_text,objective,revision,archive_sha=sys.argv[1:]
step=int(step_text); root=Path(output); checkpoint=root/checkpoint_name
assert root.is_absolute() and root.is_dir() and not root.is_symlink()
assert checkpoint.is_dir() and not checkpoint.is_symlink()

def unique(pairs):
    result={}
    for key,value in pairs:
        assert key not in result
        result[key]=value
    return result

def read_json(path):
    info=path.lstat()
    assert stat.S_ISREG(info.st_mode) and not path.is_symlink() and info.st_nlink==1
    raw=path.read_bytes(); assert raw.endswith(b"\n") and raw.count(b"\n")==1
    value=json.loads(raw[:-1].decode("utf-8"),object_pairs_hook=unique)
    expected=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")+b"\n"
    assert raw==expected
    return value,raw

receipt,receipt_raw=read_json(checkpoint/"receipt.json")
manifest,manifest_raw=read_json(checkpoint/"checkpoint_manifest.json")
declared=receipt["receipt_digest"]
unsigned=dict(receipt); del unsigned["receipt_digest"]
assert declared==hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")).hexdigest()
assert receipt["schema_version"]=="bernini-r-1p3b-action-lora-receipt-v2"
assert receipt["global_step"]==step and receipt["max_steps"]==step
assert receipt["method_source_revision"]==revision
assert receipt["method_source_archive_sha256"]==archive_sha
assert receipt["resumed_from"] is None
assert receipt["distributed"]["world_size"]==4
assert receipt["distributed"]["ulysses_size"]==4
assert receipt["distributed"]["backend"]=="nccl/rccl"
assert receipt["training_contract"]["lora_rank"]==64
assert receipt["training_contract"]["lora_alpha"]==64
assert receipt["training_contract"]["objective"]==objective
if mode=="PRELAUNCH_CAPACITY_ONLY":
    assert "exploratory_full644" not in receipt
else:
    profile=receipt["exploratory_full644"]
    assert profile["profile"]=="full644-r64-reference-dpo-preservation-one-pass-v1"
    assert profile["optimizer_rows_consumed"]==644
    assert profile["complete_one_pass"] is True
    assert profile["no_replacement_within_pass"] is True
    assert profile["resume_policy"]=="forbidden_for_this_profile"
    assert profile["source_authority"]["sha256"]=="0bcf24ce8aafabb37cf38eafe9da6b13c70043bb0f4c3146f16dc0bafd35618f"
assert manifest["schema_version"]=="bernini-r-action-lora-checkpoint-manifest-v1"
assert manifest["global_step"]==step and manifest["receipt_digest"]==declared
manifest_unsigned=dict(manifest); manifest_digest=manifest_unsigned.pop("manifest_digest")
assert manifest_digest==hashlib.sha256(json.dumps(manifest_unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")).hexdigest()
entries=manifest["entries"]; assert manifest["file_count"]==len(entries)
assert [entry["path"] for entry in entries]==sorted(entry["path"] for entry in entries)
assert len({entry["path"] for entry in entries})==len(entries)
by_path={entry["path"]:entry for entry in entries}
assert by_path["receipt.json"]["sha256"]==hashlib.sha256(receipt_raw).hexdigest()
for relative,entry in by_path.items():
    path=checkpoint/relative; info=path.lstat()
    assert stat.S_ISREG(info.st_mode) and not path.is_symlink() and info.st_nlink==1
    raw=path.read_bytes(); assert len(raw)==entry["size"]
    assert hashlib.sha256(raw).hexdigest()==entry["sha256"]
print(hashlib.sha256(manifest_raw).hexdigest())
PY
  require_plain_file_sha "${helper}" "${helper_sha256}" "${helper_size}" 444 release-helper
  "${python_bin}" -I -B "${helper}" verify-training-output \
    --mode "${helper_mode}" --output "${output}" >/dev/null
}

if [[ "${1:-}" == __validate_output__ ]]; then
  [[ $# == 2 ]] || fail "validator argv differs"
  validate_output "$2"
  exit 0
fi

if [[ "${1:-}" == __rank_worker__ ]]; then
  shift
  [[ $# -ge 2 ]] || fail "rank-worker argv differs"
  readonly rank_mode="$1"; shift
  readonly local_rank="${LOCAL_RANK:-}"
  readonly local_world="${LOCAL_WORLD_SIZE:-}"
  readonly world="${WORLD_SIZE:-}"
  [[ "${SLURM_JOB_ID:-}" == "${expected_job}" && "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "rank Slurm identity differs"
  [[ "${world}" == 4 && "${local_world}" == 4 && "${local_rank}" =~ ^[0-3]$ ]] || fail "rank WORLD4 identity differs"
  [[ "${ROCR_VISIBLE_DEVICES:-}" == 0,1,2,3 && "${HIP_VISIBLE_DEVICES:-}" == 0,1,2,3 \
    && "${CUDA_VISIBLE_DEVICES:-}" == 0,1,2,3 && "${GPU_DEVICE_ORDINAL:-}" == 0,1,2,3 ]] || \
    fail "rank GPU visibility differs"
  rank_effective_uid="$(/usr/bin/id -u)" || fail "rank effective uid query failed"
  readonly rank_effective_uid
  expected_cache_root="/tmp/cache/full644-r64-u${rank_effective_uid}-j${expected_job}-s${SLURM_STEP_ID}-v1"
  readonly expected_cache_root
  readonly cache_root="${FULL644_RANK_CACHE_ROOT:-}"
  readonly rank_cache="${cache_root}/rank-${local_rank}"
  [[ "${cache_root}" == "${expected_cache_root}" ]] || fail "rank cache authority differs"
  [[ -d "${rank_cache}" && ! -L "${rank_cache}" ]] || fail "precreated rank cache differs"
  rank_cache_envelope="$(/usr/bin/stat -c '%u:%a' "${rank_cache}")" || fail "rank cache envelope query failed"
  [[ "${rank_cache_envelope}" == "${rank_effective_uid}:700" ]] || fail "rank cache owner/mode differs"
  rank_cache_device="$(/usr/bin/stat -c %d "${rank_cache}")" || fail "rank cache device query failed"
  readonly rank_cache_device
  readonly expected_cache_entries=$'extensions\nhf\nhome\ninductor\nmiopen-custom\nmiopen-user\npycache\ntmp\ntorch\ntriton\nxdg'
  observed_cache_entries="$(/usr/bin/find "${rank_cache}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C /usr/bin/sort)" || \
    fail "rank cache leaf query failed"
  readonly observed_cache_entries
  [[ "${observed_cache_entries}" == "${expected_cache_entries}" ]] || fail "rank cache leaf closure differs"
  for cache_leaf in extensions hf home inductor miopen-custom miopen-user pycache tmp torch triton xdg; do
    cache_leaf_path="${rank_cache}/${cache_leaf}"
    [[ -d "${cache_leaf_path}" && ! -L "${cache_leaf_path}" ]] || fail "rank cache leaf type differs: ${cache_leaf}"
    cache_leaf_envelope="$(/usr/bin/stat -c '%u:%a:%d' "${cache_leaf_path}")" || \
      fail "rank cache leaf envelope query failed: ${cache_leaf}"
    [[ "${cache_leaf_envelope}" == "${rank_effective_uid}:700:${rank_cache_device}" ]] || \
      fail "rank cache leaf identity differs: ${cache_leaf}"
  done
  case "${rank_mode}" in
    PRELAUNCH_CAPACITY_ONLY) rank_cache_receipt="${capacity_cache_receipt}" ;;
    FULL644_EXPLORATORY) rank_cache_receipt="${full644_cache_receipt}" ;;
    *) fail "rank cache receipt mode differs" ;;
  esac
  "${python_bin}" -I -B - "${rank_cache_receipt}" "${cache_root}" "${rank_cache}" "${local_rank}" <<'PY'
import hashlib,json,os,stat,sys
from pathlib import Path
receipt_path,root_text,rank_text,rank_text_id=sys.argv[1:]
raw=Path(receipt_path).read_bytes(); assert raw.endswith(b"\n")
value=json.loads(raw[:-1].decode("ascii")); unsigned=dict(value); declared=unsigned.pop("receipt_digest")
assert hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest()==declared
root=Path(root_text); rank=Path(rank_text); rank_id=int(rank_text_id)
row=value["rank_caches"][rank_id]
root_info=root.lstat(); rank_info=rank.lstat()
assert stat.S_ISDIR(root_info.st_mode) and not root.is_symlink()
assert stat.S_ISDIR(rank_info.st_mode) and not rank.is_symlink()
assert (root_info.st_dev,root_info.st_ino,root_info.st_uid,stat.S_IMODE(root_info.st_mode))==(value["cache_root_device"],value["cache_root_inode"],value["cache_root_uid"],0o700)
assert (rank_info.st_dev,rank_info.st_ino,rank_info.st_uid,stat.S_IMODE(rank_info.st_mode))==(row["device"],row["inode"],row["uid"],0o700)
assert row["rank"]==rank_id and row["path"]==str(rank)
PY
  export HOME="${rank_cache}/home" TMPDIR="${rank_cache}/tmp" TMP="${rank_cache}/tmp" TEMP="${rank_cache}/tmp"
  export XDG_CACHE_HOME="${rank_cache}/xdg" HF_HOME="${rank_cache}/hf" TORCH_HOME="${rank_cache}/torch"
  export TRITON_CACHE_DIR="${rank_cache}/triton" TORCHINDUCTOR_CACHE_DIR="${rank_cache}/inductor"
  export TORCH_EXTENSIONS_DIR="${rank_cache}/extensions" PYTHONPYCACHEPREFIX="${rank_cache}/pycache"
  export MIOPEN_USER_DB_PATH="${rank_cache}/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${rank_cache}/miopen-custom"
  exec "${python_bin}" -I -B "${trainer}" "$@"
fi

[[ $# == 1 ]] || fail "expected exactly one mode"
readonly mode="$1"
output="$(output_for_mode "${mode}")" || fail "mode output resolution failed"
readonly output
[[ "${SLURM_JOB_ID:-}" == "${expected_job}" ]] || fail "holder job differs"
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numbered Slurm step is absent"
observed_node="$(/bin/hostname -s)" || fail "compute node query failed"
[[ "${observed_node}" == "${expected_node}" ]] || fail "compute node differs"
[[ "${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]] || fail "node count differs"
[[ "${SLURM_STEP_NUM_TASKS:-${SLURM_NTASKS:-}}" == 1 ]] || fail "outer task count differs"
[[ "${ROCR_VISIBLE_DEVICES:-}" == 0,1,2,3 && "${HIP_VISIBLE_DEVICES:-}" == 0,1,2,3 \
  && "${CUDA_VISIBLE_DEVICES:-}" == 0,1,2,3 && "${GPU_DEVICE_ORDINAL:-}" == 0,1,2,3 ]] || \
  fail "GPU 0-3 visibility differs"
[[ "${PATH:-}" == /usr/bin:/bin && "${LC_ALL:-}" == C && "${LANG:-}" == C \
  && "${BASH_ENV:-}" == /dev/null && "${HOME:-}" == /nonexistent/full644-job141620 ]] || \
  fail "clean step environment differs"
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" ]] || fail "inherited loader/Python environment differs"
scheduler_tmpdir_observed=absent
if [[ "${TMPDIR+x}" == x ]]; then
  [[ "${TMPDIR}" == /tmp ]] || fail "scheduler TMPDIR differs"
  scheduler_tmpdir_observed=/tmp
fi
readonly scheduler_tmpdir_observed
[[ "${TMP+x}" != x && "${TEMP+x}" != x && "${TEMPDIR+x}" != x \
  && "${MIOPEN_USER_DB_PATH+x}" != x && "${MIOPEN_CUSTOM_CACHE_DIR+x}" != x \
  && "${XDG_CACHE_HOME+x}" != x && "${HF_HOME+x}" != x && "${TORCH_HOME+x}" != x \
  && "${TRITON_CACHE_DIR+x}" != x && "${TORCHINDUCTOR_CACHE_DIR+x}" != x \
  && "${TORCH_EXTENSIONS_DIR+x}" != x && "${PYTHONPYCACHEPREFIX+x}" != x \
  && "${FULL644_RANK_CACHE_ROOT+x}" != x ]] || fail "inherited cache environment differs"
unset TMPDIR
[[ "${TMPDIR+x}" != x ]] || fail "scheduler TMPDIR normalization failed"
require_current_step_closure pre-runtime
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "training output is not fresh"
[[ -d "${output%/*}" && ! -L "${output%/*}" ]] || fail "training output parent differs"
[[ -d "${release_root}" && ! -L "${release_root}" ]] || fail "extracted source release differs"
require_plain_file_sha "${python_bin}" "${python_sha256}" "${python_size}" 755 Python
publish_step_record "${mode}"
require_plain_file_sha "${helper}" "${helper_sha256}" "${helper_size}" 444 release-helper
require_plain_file_sha "${trainer}" "${trainer_sha256}" "${trainer_size}" 444 trainer
require_plain_file_sha "${source_authority}" "${source_authority_sha256}" "${source_authority_size}" 444 source-authority
observed_dataset_summary_sha="$(sha256_file "${dataset_summary}")" || fail "dataset summary SHA query failed"
[[ "${observed_dataset_summary_sha}" == "${dataset_summary_sha256}" ]] || fail "dataset summary SHA differs"
observed_dataset_index_sha="$(sha256_file "${dataset_index}")" || fail "dataset index SHA query failed"
[[ "${observed_dataset_index_sha}" == "${dataset_index_sha256}" ]] || fail "dataset index SHA differs"
[[ -d "${parquet_root}" && ! -L "${parquet_root}" ]] || fail "parquet root differs"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
prepare_rank_caches "${mode}"
"${python_bin}" -I -B "${helper}" verify-runtime-inputs \
  --dataset-summary "${dataset_summary}" --dataset-index "${dataset_index}" \
  --source-authority "${source_authority}" >/dev/null

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1

command=(
  "${python_bin}" -I -B -m torch.distributed.run --standalone --nproc_per_node=4 --no_python
  "$0" __rank_worker__ "${mode}"
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}" --preprocessed-parquet-dir "${parquet_root}"
  --dataset-summary "${dataset_summary}" --output "${output}"
  --max-steps 1 --save-every 1 --lora-rank 64 --lora-alpha 64 --objective sft
  --learning-rate 1e-4 --weight-decay 0 --max-grad-norm 1 --seed 20260817
  --expected-bernini-commit "${expected_bernini_commit}"
  --expected-veomni-commit "${expected_veomni_commit}"
  --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}"
  --method-source-revision "${method_source_revision}"
  --method-source-archive-sha256 "${method_source_archive_sha256}"
)
if [[ "${mode}" == FULL644_EXPLORATORY ]]; then
  observed_source_authority_sha="$(sha256_file "${source_authority}")" || fail "full644 source authority SHA query failed"
  [[ "${observed_source_authority_sha}" == "${source_authority_sha256}" ]] || fail "full644 source authority SHA differs"
  command=(
    "${python_bin}" -I -B -m torch.distributed.run --standalone --nproc_per_node=4 --no_python
    "$0" __rank_worker__ "${mode}"
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}"
    --checkpoint "${checkpoint}" --preprocessed-parquet-dir "${parquet_root}"
    --dataset-summary "${dataset_summary}" --full644-source-authority-receipt "${source_authority}"
    --expected-full644-source-authority-sha256 "${source_authority_sha256}" --output "${output}"
    --exploratory-full644-one-pass --max-steps 644 --save-every 64
    --lora-rank 64 --lora-alpha 64 --objective reference_dpo_preservation
    --contrastive-negative-schedule rotate --preference-weight 1 --preference-margin 0.05
    --preference-temperature 20 --dpo-beta 10 --preservation-weight 0.25
    --learning-rate 1e-4 --weight-decay 0 --max-grad-norm 1 --seed 20260817
    --expected-bernini-commit "${expected_bernini_commit}"
    --expected-veomni-commit "${expected_veomni_commit}"
    --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}"
    --method-source-revision "${method_source_revision}"
    --method-source-archive-sha256 "${method_source_archive_sha256}"
  )
fi
"${command[@]}"
validate_output "${mode}"
require_current_step_closure post-training
case "${mode}" in
  PRELAUNCH_CAPACITY_ONLY)
    helper_mode=capacity-smoke; completion_path="${capacity_completion}"; cache_receipt="${capacity_cache_receipt}" ;;
  FULL644_EXPLORATORY)
    helper_mode=full644; completion_path="${full644_completion}"; cache_receipt="${full644_cache_receipt}" ;;
esac
[[ ! -e "${completion_path}" && ! -L "${completion_path}" ]] || fail "runner completion is not fresh"
"${python_bin}" -I -B "${helper}" publish-runner-completion \
  --mode "${helper_mode}" --output "${output}" --receipt-output "${completion_path}" \
  --slurm-job-id "${expected_job}" --slurm-step-id "${SLURM_STEP_ID}" --node "${expected_node}" \
  --cache-receipt "${cache_receipt}" >/dev/null
printf 'PASS mode=%s output=%s\n' "${mode}" "${output}"

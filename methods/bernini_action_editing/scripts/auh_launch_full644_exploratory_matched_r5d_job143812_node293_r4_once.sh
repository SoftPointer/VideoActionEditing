#!/bin/bash -p
set -euo pipefail
umask 077

readonly R5D_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5d_job143812_node293_case00_847b91a2_c91de7eb_85ccc17b_r4
readonly R5D_PAYLOAD="$R5D_ROOT/launch/root_launch_payload_auh_r5d.sh"
readonly R5D_STATIC="$R5D_ROOT/evidence/static_nomodel_probe_receipt_r5d.json"
readonly R5D_BOOTSTRAP="$R5D_ROOT/evidence/root_bootstrap_cpu_probe_receipt_r5d.json"
readonly R5D_CONSUMPTION="$R5D_ROOT/evidence/cpu_consumption_probe_work_r1/r5d-cpu-consumption-probe.json"

[[ "$0" == /bin/bash && "$#" -eq 0 && "$-" == *p* ]] || exit 91
[[ -z "${CUDA_VISIBLE_DEVICES-}" && -z "${HIP_VISIBLE_DEVICES-}" \
  && -z "${ROCR_VISIBLE_DEVICES-}" && -z "${GPU_DEVICE_ORDINAL-}" ]] || exit 92
if shopt -q varredir_close 2>/dev/null; then
  shopt -u varredir_close
fi

exec {R5D_PAYLOAD_FD}<"$R5D_PAYLOAD"
R5D_PAYLOAD_FD="$R5D_PAYLOAD_FD" /usr/bin/python3.10 -I -S -B -c '
import hashlib,json,os,stat,sys

fd=int(os.environ["R5D_PAYLOAD_FD"])
payload,static_path,bootstrap_path,consumption_path=sys.argv[1:]

def read_fd(descriptor,size):
    chunks=[]
    offset=0
    while offset<size:
        block=os.pread(descriptor,min(1048576,size-offset),offset)
        if not block:
            break
        chunks.append(block)
        offset+=len(block)
    raw=b"".join(chunks)
    if len(raw)!=size:
        raise RuntimeError("held payload read differs")
    return raw

held=os.fstat(fd)
named=os.lstat(payload)
raw=read_fd(fd,held.st_size)
identity=lambda value:(value.st_dev,value.st_ino,value.st_uid,value.st_gid,value.st_mode,value.st_nlink,value.st_rdev,value.st_size,getattr(value,"st_blocks",0),value.st_mtime_ns,value.st_ctime_ns)
if (
    not stat.S_ISREG(held.st_mode)
    or stat.S_IMODE(held.st_mode)!=0o444
    or held.st_nlink!=1
    or held.st_uid!=2012
    or held.st_gid!=2000
    or identity(held)!=identity(named)
    or held.st_size!=26615
    or hashlib.sha256(raw).hexdigest()!="f55e10faa6fe17617d7b6b684b91ba5a1460b8eee809903f57a30587721fc7cc"
    or os.lseek(fd,0,os.SEEK_CUR)!=0
):
    raise RuntimeError("held production payload differs")

expected=(
    (static_path,"f23ce3e264fe7ad982fec2980e188fff38e6e3ca92b46ac2339b85b375493f82","149"),
    (bootstrap_path,"f4a41aeb6c08d2e038bf878c375e51e60741dc0da0f94aea7d43cc3ad106f73c","150"),
    (consumption_path,"907bdb5a711b8dfd7973206a364660d0411b309daec9b0934b82e678eb515233",None),
)
for path,digest,step in expected:
    info=os.lstat(path)
    with open(path,"rb",buffering=0) as handle:
        receipt_raw=handle.read()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode)!=0o400
        or info.st_nlink!=1
        or info.st_uid!=2012
        or info.st_gid!=2000
        or hashlib.sha256(receipt_raw).hexdigest()!=digest
    ):
        raise RuntimeError("CPU gate receipt identity differs")
    value=json.loads(receipt_raw)
    if value.get("status")!="PASS" or (step is not None and value.get("slurm_step_id")!=step):
        raise RuntimeError("CPU gate receipt semantics differ")
if json.loads(open(consumption_path,"rb").read()).get("summary")!={
    "hostile_rejection_count":4,
    "rejected_hostiles":["digest","task","adapter_namespace","adapter_leaf"],
    "success_count":2,
    "successful_arms":["base","full"],
}:
    raise RuntimeError("consumption summary differs")
' "$R5D_PAYLOAD" "$R5D_STATIC" "$R5D_BOOTSTRAP" "$R5D_CONSUMPTION"

exec /usr/bin/srun \
  --jobid=143812 \
  --job-name=f644-r5d-case00-gpu \
  --exclusive --exact --immediate=10 --kill-on-bad-exit=1 \
  --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-293 \
  --cpus-per-task=64 --mem=64G --gpus-per-node=8 \
  --export=NONE --time=01:00:00 \
  /bin/bash -p -s <&"$R5D_PAYLOAD_FD"

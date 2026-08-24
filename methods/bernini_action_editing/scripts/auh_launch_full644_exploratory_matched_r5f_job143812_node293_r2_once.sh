#!/bin/bash -p
set -euo pipefail
umask 077

readonly R5F_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_case00_847b91a2_c91de7eb_d70eac5c_r1
readonly R5F_PAYLOAD="$R5F_ROOT/launch/root_launch_payload_auh_r5d.sh"
readonly R5F_STATIC="$R5F_ROOT/evidence/static_nomodel_probe_receipt_r5d.json"
readonly R5F_STATIC_EVIDENCE="$R5F_ROOT/evidence/r5f_static_nomodel_probe.sacct-and-replay.json"
readonly R5F_BOOTSTRAP="$R5F_ROOT/evidence/root_bootstrap_cpu_probe_receipt_r5d.json"
readonly R5F_BOOTSTRAP_EVIDENCE="$R5F_ROOT/evidence/r5f_root_bootstrap_cpu_probe.sacct-and-replay.json"
readonly R5F_CONSUMPTION="$R5F_ROOT/diagnostics/cpu_consumption_probe_work_r1/r5d-cpu-consumption-probe.json"
readonly R5F_CONSUMPTION_EVIDENCE="$R5F_ROOT/diagnostics/r5f_cpu_consumption_probe.sacct-and-replay.json"
readonly R5F_RANK_CACHE=/tmp/bernini-full644-r5f-job143812-node293-r1-rank-cache

[[ "$0" == /bin/bash && "$#" -eq 0 && "$-" == *p* ]] || exit 91
[[ -z "${BASH_ENV-}" && -z "${ENV-}" ]] || exit 92
[[ -z "${CUDA_VISIBLE_DEVICES-}" && -z "${HIP_VISIBLE_DEVICES-}" \
  && -z "${ROCR_VISIBLE_DEVICES-}" && -z "${GPU_DEVICE_ORDINAL-}" ]] || exit 93
if shopt -q varredir_close 2>/dev/null; then
  shopt -u varredir_close
fi

exec {R5F_PAYLOAD_FD}<"$R5F_PAYLOAD"
R5F_PAYLOAD_FD="$R5F_PAYLOAD_FD" /usr/bin/python3.10 -I -S -B -c '
import hashlib,json,os,stat,sys

fd=int(os.environ["R5F_PAYLOAD_FD"])
payload,root,cache,static_path,static_evidence_path,bootstrap_path,bootstrap_evidence_path,consumption_path,consumption_evidence_path=sys.argv[1:]

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")

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

def identity(value):
    return (value.st_dev,value.st_ino,value.st_uid,value.st_gid,value.st_mode,value.st_nlink,value.st_rdev,value.st_size,getattr(value,"st_blocks",0),value.st_mtime_ns,value.st_ctime_ns)

def stable_json(path,sha256):
    descriptor=os.open(path,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0))
    try:
        before=os.fstat(descriptor)
        raw=read_fd(descriptor,before.st_size)
        after=os.fstat(descriptor)
        named=os.lstat(path)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode)!=0o400
        or before.st_nlink!=1
        or before.st_uid!=2012
        or before.st_gid!=2000
        or identity(before)!=identity(after)
        or identity(before)!=identity(named)
        or hashlib.sha256(raw).hexdigest()!=sha256
    ):
        raise RuntimeError("CPU gate artifact identity differs")
    value=json.loads(raw)
    if type(value) is not dict or canonical(value)+b"\n"!=raw:
        raise RuntimeError("CPU gate artifact JSON differs")
    return value

held=os.fstat(fd)
named=os.lstat(payload)
raw=read_fd(fd,held.st_size)
if (
    not stat.S_ISREG(held.st_mode)
    or stat.S_IMODE(held.st_mode)!=0o444
    or held.st_nlink!=1
    or held.st_uid!=2012
    or held.st_gid!=2000
    or identity(held)!=identity(named)
    or held.st_size!=26460
    or hashlib.sha256(raw).hexdigest()!="aefb53b05976732c857a65730c0bcc16ab90764f8921ef7ec0377176049b8b6d"
    or os.lseek(fd,0,os.SEEK_CUR)!=0
):
    raise RuntimeError("held production payload differs")

static=stable_json(static_path,"28fb5c311a45b88106d40698c65d8edb88c5e9bdfb5dea8613aced76cb8ba7ee")
static_evidence=stable_json(static_evidence_path,"06279eed3f89546e914f372385c1225c4f7ade1ddf24ad33526fe8f2d7659355")
bootstrap=stable_json(bootstrap_path,"3defa55653714d37ed2c3040c96d60d59359521d95697986929156abfa5891cb")
consumption=stable_json(consumption_path,"8cc9f249e20223a81ead26ad97c25560d9222b3e5ca19be343f5e21555ae5b8e")
bootstrap_evidence=stable_json(bootstrap_evidence_path,"7242cd2385e3a35b73d658ae9efb0e3be200345c29a3e9be30b1d729a83dd5b8")
consumption_evidence=stable_json(consumption_evidence_path,"e3c17352d83b993e5b15c320e545a726640f871e9622ff80a67e698378b39681")
if (
    static.get("schema_version")!="full644-exploratory-matched-r5f-static-nomodel-probe-v1"
    or static.get("status")!="PASS"
    or static.get("slurm_step_id")!="167"
    or static.get("receipt_digest")!="bf460bbfc5d40760a43fb212e807aca97b539e9ed7e8a3cb5108cb61b40f8f92"
    or bootstrap.get("schema_version")!="full644-exploratory-matched-r5d-root-bootstrap-cpu-probe-v1"
    or bootstrap.get("status")!="PASS"
    or bootstrap.get("slurm_step_id")!="168"
    or bootstrap.get("receipt_digest")!="3ba27c7f024ecae5062a6ad1b7cbfebe3af96241686ece1ee6e35eed345d8bf1"
    or consumption.get("schema_version")!="full644-exploratory-matched-r5f-cpu-consumption-probe-v1"
    or consumption.get("status")!="PASS"
    or consumption.get("receipt_digest")!="0081b7e444fa4269dab90b686d7a17f0f39337fb30aa441e20414a7b37859980"
):
    raise RuntimeError("CPU gate receipt semantics differ")
if consumption.get("summary")!={
    "hostile_rejection_count":4,
    "rejected_hostiles":["digest","task","adapter_namespace","adapter_leaf"],
    "success_count":2,
    "successful_arms":["base","full"],
}:
    raise RuntimeError("consumption summary differs")
if (
    static_evidence.get("schema_version")!="full644-r5f-static-direct-step-replay-only-evidence-v2"
    or static_evidence.get("status")!="PASS"
    or static_evidence.get("numeric_step")!="143812.167"
    or static_evidence.get("evidence_digest")!="2e3984e43f6ae3729f77da3f007b8b29cd3ca5ee42c8c6e31708d3329ed61063"
    or bootstrap_evidence.get("schema_version")!="full644-r5f-bootstrap-heldfd-controller-evidence-v1"
    or bootstrap_evidence.get("status")!="PASS"
    or bootstrap_evidence.get("numeric_step")!="143812.168"
    or bootstrap_evidence.get("evidence_digest")!="0788ca06515db91af9c93adfcc8dbe3c5b4ac0f56f808165598071e651a88002"
    or consumption_evidence.get("schema_version")!="full644-r5f-consumption-replay-only-controller-evidence-v2"
    or consumption_evidence.get("status")!="PASS"
    or consumption_evidence.get("numeric_step")!="143812.169"
    or consumption_evidence.get("evidence_digest")!="cfb56e1e3a1fe92d988db53a62759bcb19939f838885705d1e12316b5f4323ea"
    or consumption_evidence.get("replay_only") is not True
    or consumption_evidence.get("srun_called_by_replay") is not False
):
    raise RuntimeError("CPU gate replay evidence differs")

if set(os.listdir(os.path.join(root,"evidence")))!={
    "static_nomodel_probe_receipt_r5d.json",
    "r5f_static_nomodel_probe.sacct-and-replay.json",
    "root_bootstrap_cpu_probe_receipt_r5d.json",
    "r5f_root_bootstrap_cpu_probe.stdout.log",
    "r5f_root_bootstrap_cpu_probe.stderr.log",
    "r5f_root_bootstrap_cpu_probe.sacct-and-replay.json",
}:
    raise RuntimeError("CPU gate evidence closure differs")

for relative in ("outputs/media","final","runtime"):
    path=os.path.join(root,relative)
    info=os.lstat(path)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode)!=0o755
        or info.st_uid!=2012
        or info.st_gid!=2000
        or os.listdir(path)
    ):
        raise RuntimeError("production result root is not fresh")
if os.listdir(os.path.join(root,"outputs")) != ["media"]:
    raise RuntimeError("production outputs closure differs")
for relative in (
    "final/case00_canary_report_auh_r5d.json",
    "final/case00_canary_runner_attestation_auh_r5d.json",
):
    if os.path.lexists(os.path.join(root,relative)):
        raise RuntimeError("production report is not fresh")
if os.path.lexists(cache):
    raise RuntimeError("production rank cache is not fresh")
if identity(os.fstat(fd))!=identity(held) or identity(os.lstat(payload))!=identity(held) or read_fd(fd,held.st_size)!=raw or os.lseek(fd,0,os.SEEK_CUR)!=0:
    raise RuntimeError("held production payload replay differs")
' "$R5F_PAYLOAD" "$R5F_ROOT" "$R5F_RANK_CACHE" "$R5F_STATIC" "$R5F_STATIC_EVIDENCE" "$R5F_BOOTSTRAP" "$R5F_BOOTSTRAP_EVIDENCE" "$R5F_CONSUMPTION" "$R5F_CONSUMPTION_EVIDENCE"

exec /usr/bin/srun \
  --jobid=143812 \
  --job-name=f644-r5f-case00-gpu \
  --exclusive --exact --immediate=10 --kill-on-bad-exit=1 \
  --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-293 \
  --cpus-per-task=64 --mem=64G --gpus-per-node=8 \
  --export=NONE --time=01:00:00 \
  /bin/bash -p -c '
[[ "${SLURM_JOB_ID-}" == 143812 && "${SLURM_STEP_ID-}" =~ ^[1-9][0-9]*$ ]] || exit 81
(( 10#$SLURM_STEP_ID > 169 )) || exit 82
[[ "${SLURM_GPUS_ON_NODE-}" == 8 && "${SLURM_GPUS_PER_NODE-}" == 8 && "${SLURM_STEP_GPUS-}" == 0,1,2,3,4,5,6,7 ]] || exit 83
[[ ! -e /tmp/bernini-full644-r5f-job143812-node293-r1-rank-cache && ! -L /tmp/bernini-full644-r5f-job143812-node293-r1-rank-cache ]] || exit 84
exec /bin/bash -p -s
' /bin/bash <&"$R5F_PAYLOAD_FD"

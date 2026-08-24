#!/bin/bash -p
# Submit exactly one runnable (not user-held) full81 job behind active14.

case "$-" in *p*) ;; *) echo "[graft-pa-full81-submit] ERROR: privileged Bash required" >&2; exit 2;; esac
set -Eeuo pipefail
umask 077
for poison_name in ${!LD_*} ${!PYTHON*} ${!SBATCH_*}; do unset "${poison_name}"; done
unset BASH_ENV ENV CDPATH GLOBIGNORE HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export PATH=/usr/bin:/bin LC_ALL=C LANG=C
fail() { echo "[graft-pa-full81-submit] ERROR: $*" >&2; exit 2; }

readonly required_launcher_sha256=0f8a099f89b7b5297632ee91d6b30f0a9ebff396f4dc4e9d675b6ce1012ebb80
readonly required_field14_plan_sha256=ec07f89c49dc545397d436eecbac03bb4de9126f5bf97704d41cde6a666782a2
readonly required_active14_plan_sha256=e24f9c5c62049da6400f40c71becac308434f9eb56a0ae22774e7b419d4e3541
readonly required_full81_plan_sha256=05c4f53279b6e23eb06e8aac8abe3ed8bc39d322d1acb9ebfdd0c2d77a44915a
readonly required_field14_job_id=133530
readonly required_active14_job_id=133534
readonly required_active14_receipt_path=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_active14_transaction_world8_v1/runs/source-037f7a8-launcher-8b6d08b-r2/receipt.json
readonly required_sbatch_path=/usr/bin/sbatch
readonly export_names_csv=GRAFT_FULL81_SOURCE_ARCHIVE,GRAFT_FULL81_SOURCE_ARCHIVE_SHA256,GRAFT_FULL81_RUNTIME_CLOSURE,GRAFT_FULL81_RUNTIME_CLOSURE_SHA256,GRAFT_FULL81_PYTHON_BIN,GRAFT_FULL81_PYTHON_SHA256,BERNINI_OFFICIAL_ROOT,BERNINI_VEOMNI_ROOT,BERNINI_ACTION_CHECKPOINT,BERNINI_CHECKPOINT_CONTENT_MANIFEST,BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256,GRAFT_FULL81_FIELD14_PLAN,GRAFT_FULL81_FIELD14_PLAN_SHA256,GRAFT_FULL81_ACTIVE14_PLAN,GRAFT_FULL81_ACTIVE14_PLAN_SHA256,GRAFT_FULL81_PLAN,GRAFT_FULL81_PLAN_SHA256,GRAFT_FULL81_TERMINAL_ADMISSION,GRAFT_FULL81_TERMINAL_ADMISSION_SHA256,GRAFT_FULL81_TERMINAL_MATERIALIZER_RUNTIME_SHA256,GRAFT_FULL81_UPSTREAM_FIELD14_RECEIPT,GRAFT_FULL81_UPSTREAM_ACTIVE14_RECEIPT,GRAFT_FULL81_UPSTREAM_FIELD14_JOB_ID,GRAFT_FULL81_ACTIVE14_JOB_ID,GRAFT_FULL81_OUTPUT_ROOT,GRAFT_FULL81_LOG_ROOT,GRAFT_FULL81_LAUNCHER_SOURCE,GRAFT_FULL81_LAUNCHER_SHA256

required_names=(
  GRAFT_FULL81_SOURCE_ARCHIVE GRAFT_FULL81_SOURCE_ARCHIVE_SHA256
  GRAFT_FULL81_RUNTIME_CLOSURE GRAFT_FULL81_RUNTIME_CLOSURE_SHA256
  GRAFT_FULL81_PYTHON_BIN GRAFT_FULL81_PYTHON_SHA256
  BERNINI_OFFICIAL_ROOT BERNINI_VEOMNI_ROOT BERNINI_ACTION_CHECKPOINT
  BERNINI_CHECKPOINT_CONTENT_MANIFEST BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
  GRAFT_FULL81_FIELD14_PLAN GRAFT_FULL81_FIELD14_PLAN_SHA256
  GRAFT_FULL81_ACTIVE14_PLAN GRAFT_FULL81_ACTIVE14_PLAN_SHA256
  GRAFT_FULL81_PLAN GRAFT_FULL81_PLAN_SHA256
  GRAFT_FULL81_TERMINAL_ADMISSION GRAFT_FULL81_TERMINAL_ADMISSION_SHA256
  GRAFT_FULL81_TERMINAL_MATERIALIZER_RUNTIME_SHA256
  GRAFT_FULL81_UPSTREAM_FIELD14_RECEIPT
  GRAFT_FULL81_UPSTREAM_ACTIVE14_RECEIPT
  GRAFT_FULL81_UPSTREAM_FIELD14_JOB_ID GRAFT_FULL81_ACTIVE14_JOB_ID
  GRAFT_FULL81_OUTPUT_ROOT GRAFT_FULL81_LOG_ROOT
  GRAFT_FULL81_LAUNCHER_SOURCE GRAFT_FULL81_LAUNCHER_SHA256
  GRAFT_FULL81_SUBMIT_WRAPPER_SHA256
)
for variable_name in "${required_names[@]}"; do [[ -n "${!variable_name:-}" ]] || fail "${variable_name} must be nonempty"; done
observed=0
for variable_name in ${!GRAFT_FULL81_*} ${!BERNINI_*}; do
  case " ${required_names[*]} " in *" ${variable_name} "*) ;; *) fail "unexpected interface variable ${variable_name}";; esac
  ((observed+=1))
done
[[ "${observed}" -eq 29 ]] || fail "exactly 29 interface variables are required"
[[ "${GRAFT_FULL81_ACTIVE14_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || fail "active14 dependency job ID is not frozen numeric"
[[ "${GRAFT_FULL81_UPSTREAM_FIELD14_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || fail "field14 job ID is not frozen numeric"
[[ "${GRAFT_FULL81_ACTIVE14_JOB_ID}" != "${GRAFT_FULL81_UPSTREAM_FIELD14_JOB_ID}" ]] || fail "dependency IDs alias"
[[ "${GRAFT_FULL81_ACTIVE14_JOB_ID}" == "${required_active14_job_id}" && "${GRAFT_FULL81_UPSTREAM_FIELD14_JOB_ID}" == "${required_field14_job_id}" ]] || fail "dependency job ID pins differ"
[[ "${GRAFT_FULL81_UPSTREAM_ACTIVE14_RECEIPT}" == "${required_active14_receipt_path}" ]] || fail "active14 receipt path pin differs"
[[ "${GRAFT_FULL81_FIELD14_PLAN_SHA256}" == "${required_field14_plan_sha256}" && "${GRAFT_FULL81_ACTIVE14_PLAN_SHA256}" == "${required_active14_plan_sha256}" && "${GRAFT_FULL81_PLAN_SHA256}" == "${required_full81_plan_sha256}" ]] || fail "plan SHA pins differ"
[[ "${GRAFT_FULL81_LAUNCHER_SHA256}" == "${required_launcher_sha256}" ]] || fail "launcher SHA differs from wrapper hardcode"
[[ "${GRAFT_FULL81_OUTPUT_ROOT}" != "${GRAFT_FULL81_LOG_ROOT}" ]] || fail "output and log roots alias"

exec /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 \
  "${GRAFT_FULL81_PYTHON_BIN}" -I -S -B - \
  "$0" "${GRAFT_FULL81_SUBMIT_WRAPPER_SHA256}" "${required_sbatch_path}" \
  "${required_launcher_sha256}" "${export_names_csv}" \
  "${GRAFT_FULL81_SOURCE_ARCHIVE}" "${GRAFT_FULL81_SOURCE_ARCHIVE_SHA256}" \
  "${GRAFT_FULL81_RUNTIME_CLOSURE}" "${GRAFT_FULL81_RUNTIME_CLOSURE_SHA256}" \
  "${GRAFT_FULL81_PYTHON_BIN}" "${GRAFT_FULL81_PYTHON_SHA256}" \
  "${BERNINI_OFFICIAL_ROOT}" "${BERNINI_VEOMNI_ROOT}" "${BERNINI_ACTION_CHECKPOINT}" \
  "${BERNINI_CHECKPOINT_CONTENT_MANIFEST}" "${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}" \
  "${GRAFT_FULL81_FIELD14_PLAN}" "${GRAFT_FULL81_FIELD14_PLAN_SHA256}" \
  "${GRAFT_FULL81_ACTIVE14_PLAN}" "${GRAFT_FULL81_ACTIVE14_PLAN_SHA256}" \
  "${GRAFT_FULL81_PLAN}" "${GRAFT_FULL81_PLAN_SHA256}" \
  "${GRAFT_FULL81_TERMINAL_ADMISSION}" "${GRAFT_FULL81_TERMINAL_ADMISSION_SHA256}" \
  "${GRAFT_FULL81_TERMINAL_MATERIALIZER_RUNTIME_SHA256}" \
  "${GRAFT_FULL81_UPSTREAM_FIELD14_RECEIPT}" \
  "${GRAFT_FULL81_UPSTREAM_ACTIVE14_RECEIPT}" \
  "${GRAFT_FULL81_UPSTREAM_FIELD14_JOB_ID}" "${GRAFT_FULL81_ACTIVE14_JOB_ID}" \
  "${GRAFT_FULL81_OUTPUT_ROOT}" "${GRAFT_FULL81_LOG_ROOT}" \
  "${GRAFT_FULL81_LAUNCHER_SOURCE}" "${GRAFT_FULL81_LAUNCHER_SHA256}" <<'PY'
from __future__ import annotations
import hashlib,json,os,re,stat,subprocess,sys
from pathlib import Path

(wrapper_path,wrapper_sha,sbatch_path,hardcoded_launcher_sha,export_csv,
 archive,archive_sha,closure,closure_sha,python_bin,python_sha,bernini_root,veomni_root,checkpoint,
 checkpoint_manifest,checkpoint_manifest_sha,field14_plan,field14_plan_sha,active14_plan,active14_plan_sha,
 full81_plan,full81_plan_sha,terminal,terminal_sha,materializer_runtime_sha,field14_receipt,active14_receipt,
 field14_job_id,active14_job_id,output_root,log_root,launcher,launcher_sha)=sys.argv[1:]

EXPORT_NAMES=tuple(export_csv.split(",")); SHA=re.compile(r"[0-9a-f]{64}\Z"); JOB=re.compile(r"([1-9][0-9]*)(?:;([A-Za-z0-9][A-Za-z0-9._-]*))?\Z")
SUPERVISOR_ENV={"PATH":"/usr/bin:/bin","LC_ALL":"C","LANG":"C","PYTHONDONTWRITEBYTECODE":"1"}
def die(message): raise SystemExit(message)
def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def identity(value): return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode,value.st_nlink)
def inode(value): return (value.st_dev,value.st_ino)
def require_sha(value,label):
    if SHA.fullmatch(value) is None: die(label+" is not lowercase SHA-256")
def open_plain(text,label,mode=None,nlink=None,executable=False):
    path=Path(text)
    if not path.is_absolute(): die(label+" is not absolute")
    before=path.lstat(); resolved=path.resolve(strict=True)
    if resolved!=path or stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode): die(label+" is not exact plain file")
    if mode is not None and stat.S_IMODE(before.st_mode)!=mode: die(label+" mode differs")
    if nlink is not None and before.st_nlink!=nlink: die(label+" link count differs")
    if executable and not before.st_mode&0o111: die(label+" is not executable")
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)); chunks=[]
    try:
        opened=os.fstat(fd)
        if identity(opened)!=identity(before): die(label+" changed while opening")
        while True:
            block=os.read(fd,1024*1024)
            if not block: break
            chunks.append(block)
        after=os.fstat(fd)
        if identity(opened)!=identity(after) or identity(after)!=identity(path.lstat()): die(label+" changed while reading")
        raw=b"".join(chunks); os.lseek(fd,0,os.SEEK_SET)
        return path,fd,raw,identity(after)
    except BaseException:
        os.close(fd); raise
def revalidate(item,label):
    path,fd,raw,wanted=item
    if identity(os.fstat(fd))!=wanted or identity(path.lstat())!=wanted: die(label+" identity changed")
    os.lseek(fd,0,os.SEEK_SET); chunks=[]
    while True:
        block=os.read(fd,1024*1024)
        if not block: break
        chunks.append(block)
    os.lseek(fd,0,os.SEEK_SET)
    if b"".join(chunks)!=raw: die(label+" bytes changed")
def open_fresh_parent(text,label):
    leaf=Path(text)
    if not leaf.is_absolute() or leaf==Path("/") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",leaf.name): die(label+" leaf differs")
    parent=leaf.parent; before=parent.lstat()
    if parent.resolve(strict=True)!=parent or stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode): die(label+" parent differs")
    fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); opened=os.fstat(fd)
    if inode(opened)!=inode(before): die(label+" parent changed")
    try: os.stat(leaf.name,dir_fd=fd,follow_symlinks=False)
    except FileNotFoundError: pass
    else: die(label+" already exists")
    return leaf,parent,fd,inode(opened)
def open_stable_directory(path,label):
    before=path.lstat()
    if path.resolve(strict=True)!=path or stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode): die(label+" differs")
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); opened=os.fstat(fd)
    if inode(opened)!=inode(before): os.close(fd); die(label+" changed")
    return fd,inode(opened)
def reserve_receipt(parent_fd,name):
    fd=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent_fd)
    opened=os.fstat(fd)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink!=1 or stat.S_IMODE(opened.st_mode)!=0o600: die("receipt reservation differs")
    return fd
def publish_receipt(fd,parent_fd,payload):
    offset=0
    while offset<len(payload):
        wrote=os.write(fd,payload[offset:])
        if wrote<=0: die("receipt write stalled")
        offset+=wrote
    os.fsync(fd); os.lseek(fd,0,os.SEEK_SET)
    if os.read(fd,len(payload)+1)!=payload: die("receipt reread differs")
    os.fchmod(fd,0o444); os.fsync(fd); os.fsync(parent_fd)

if sys.platform=="darwin": os.environ.pop("__CF_USER_TEXT_ENCODING",None)
if dict(os.environ)!=SUPERVISOR_ENV or sys.flags.isolated!=1 or sys.flags.no_site!=1 or sys.flags.ignore_environment!=1: die("isolated supervisor differs")
if len(EXPORT_NAMES)!=28 or len(set(EXPORT_NAMES))!=28: die("export interface differs")
for value,label in ((wrapper_sha,"wrapper"),(archive_sha,"archive"),(closure_sha,"closure"),(python_sha,"Python"),(checkpoint_manifest_sha,"checkpoint manifest"),(field14_plan_sha,"field14 plan"),(active14_plan_sha,"active14 plan"),(full81_plan_sha,"full81 plan"),(terminal_sha,"terminal"),(materializer_runtime_sha,"materializer runtime"),(launcher_sha,"launcher")): require_sha(value,label)
if launcher_sha!=hardcoded_launcher_sha or not active14_job_id.isdigit() or not field14_job_id.isdigit() or active14_job_id==field14_job_id: die("dependency/launcher pins differ")
if checkpoint_manifest_sha!="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831": die("checkpoint manifest pin differs")

output,parent,parent_fd,parent_identity=open_fresh_parent(output_root,"output")
logs,log_parent,log_parent_fd,log_parent_identity=open_fresh_parent(log_root,"log")
field_receipt_path=Path(field14_receipt)
if not field_receipt_path.is_absolute() or field_receipt_path.name!="receipt.json": die("future Field14 receipt path differs")
field_output=field_receipt_path.parent
if field_output in (output,logs) or not field_output.name: die("future Field14 output aliases this job")
field_anchor=field_output.parent
field_anchor_fd,field_anchor_identity=open_stable_directory(field_anchor,"Field14 output anchor")
try:
    field_output_info=field_output.lstat()
except FileNotFoundError:
    pass
else:
    if field_output.resolve(strict=True)!=field_output or stat.S_ISLNK(field_output_info.st_mode) or not stat.S_ISDIR(field_output_info.st_mode): die("preexisting Field14 output root differs")
    try:
        field_receipt_info=field_receipt_path.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(field_receipt_info.st_mode) or not stat.S_ISREG(field_receipt_info.st_mode) or stat.S_IMODE(field_receipt_info.st_mode)!=0o444 or field_receipt_info.st_nlink!=1: die("preexisting Field14 receipt is not sealed")
active_receipt_path=Path(active14_receipt)
if not active_receipt_path.is_absolute() or active_receipt_path.name!="receipt.json": die("future Active14 receipt path differs")
active_output=active_receipt_path.parent
if active_output in (output,logs,field_output) or not active_output.name: die("future Active14 output aliases another root")
active_anchor=active_output.parent
active_anchor_fd,active_anchor_identity=open_stable_directory(active_anchor,"Active14 output anchor")
try:
    active_output_info=active_output.lstat()
except FileNotFoundError:
    pass
else:
    if active_output.resolve(strict=True)!=active_output or stat.S_ISLNK(active_output_info.st_mode) or not stat.S_ISDIR(active_output_info.st_mode): die("preexisting Active14 output root differs")
    try:
        active_receipt_info=active_receipt_path.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(active_receipt_info.st_mode) or not stat.S_ISREG(active_receipt_info.st_mode) or stat.S_IMODE(active_receipt_info.st_mode)!=0o444 or active_receipt_info.st_nlink!=1: die("preexisting Active14 receipt is not sealed")
receipt_name=output.name+".submission.receipt.json"
try:
    try: os.stat(receipt_name,dir_fd=parent_fd,follow_symlinks=False)
    except FileNotFoundError: pass
    else: die("submission receipt already exists")
    receipt_fd=reserve_receipt(parent_fd,receipt_name)
except BaseException:
    os.close(active_anchor_fd); os.close(field_anchor_fd); os.close(parent_fd); os.close(log_parent_fd); raise

specs=(
 (wrapper_path,"wrapper",0o444,1,False,wrapper_sha),(archive,"archive",0o444,1,False,archive_sha),
 (closure,"closure",0o444,1,False,closure_sha),(python_bin,"Python",None,None,True,python_sha),
 (checkpoint_manifest,"checkpoint manifest",0o444,1,False,checkpoint_manifest_sha),
 (field14_plan,"field14 plan",0o444,1,False,field14_plan_sha),(active14_plan,"active14 plan",0o444,1,False,active14_plan_sha),
 (full81_plan,"full81 plan",0o444,1,False,full81_plan_sha),(terminal,"terminal",0o444,1,False,terminal_sha),
 (launcher,"launcher",0o555,1,True,launcher_sha),
 (sbatch_path,"sbatch",None,None,True,None),
)
opened={}; retained=[]; submitted=False
try:
    for text,label,mode,nlink,executable,wanted_sha in specs:
        item=open_plain(text,label,mode,nlink,executable); opened[label]=item; retained.append(item[1])
        if wanted_sha is not None and digest(item[2])!=wanted_sha: die(label+" SHA differs")
    if Path(sys.executable).resolve(strict=True)!=Path(python_bin): die("running Python differs")
    field_plan=json.loads(opened["field14 plan"][2].decode("ascii")); active_plan=json.loads(opened["active14 plan"][2].decode("ascii")); plan=json.loads(opened["full81 plan"][2].decode("ascii"))
    if opened["field14 plan"][2]!=canonical(field_plan)+b"\n" or opened["active14 plan"][2]!=canonical(active_plan)+b"\n" or opened["full81 plan"][2]!=canonical(plan)+b"\n": die("plan canonical bytes differ")
    if plan.get("dependency")!={"job_id":active14_job_id,"kind":"afterok","purpose":"queue-gate-only-no-weight-lineage","receipt_path":active14_receipt,"receipt_sha256_policy":"derive-from-stable-sealed-file-after-afterok"}: die("full81 plan dependency differs")
    if plan.get("checkpoint_policy",{}).get("dependency_checkpoint_consumed") is not False or any(plan.get("authority",{}).values()): die("full81 plan authority differs")
    dependency=active_plan.get("field14_dependency",active_plan.get("afterok_dependency",active_plan.get("dependency",{})))
    if dependency.get("job_id")!=field14_job_id or dependency.get("kind")!="afterok" or dependency.get("receipt_path")!=field14_receipt or any(active_plan.get("authority",{}).values()) or any(field_plan.get("authority",{}).values()): die("active14/field14 plan dependency or authority differs")
    fd_root=Path("/proc/self/fd") if Path("/proc/self/fd").is_dir() else Path("/dev/fd")
    launcher_fd=opened["launcher"][1]; sbatch_fd=opened["sbatch"][1]
    launcher_transport=str(fd_root/str(launcher_fd)); sbatch_transport=str(fd_root/str(sbatch_fd))
    if inode(os.stat(launcher_transport))!=inode(os.fstat(launcher_fd)) or inode(os.stat(sbatch_transport))!=inode(os.fstat(sbatch_fd)): die("retained fd transport differs")
    values=(archive,archive_sha,closure,closure_sha,python_bin,python_sha,bernini_root,veomni_root,checkpoint,checkpoint_manifest,checkpoint_manifest_sha,field14_plan,field14_plan_sha,active14_plan,active14_plan_sha,full81_plan,full81_plan_sha,terminal,terminal_sha,materializer_runtime_sha,field14_receipt,active14_receipt,field14_job_id,active14_job_id,output_root,log_root,launcher,launcher_sha)
    child_env={"PATH":"/usr/bin:/bin","LC_ALL":"C","LANG":"C",**dict(zip(EXPORT_NAMES,values))}
    argv=[sbatch_transport,"--parsable",f"--export={','.join(EXPORT_NAMES)}",f"--dependency=afterok:{active14_job_id}","--partition=faculty","--qos=bgqos","--nodes=1","--ntasks=1","--cpus-per-task=64","--mem=256G","--gres=gpu:mi210:8","--time=72:00:00","--job-name=graft-pa-full81-v1","--exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318",launcher_transport]
    completed=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,timeout=120,pass_fds=(launcher_fd,sbatch_fd),env=child_env)
    if completed.returncode!=0: die(f"sbatch failed exit={completed.returncode} stderr_sha256={digest(completed.stderr)}")
    try: token=completed.stdout.decode("ascii")
    except UnicodeDecodeError: die("sbatch stdout is not ASCII")
    if token.count("\n")!=1 or not token.endswith("\n"): die("sbatch stdout framing differs")
    match=JOB.fullmatch(token[:-1])
    if match is None: die("sbatch parsable job id differs")
    submitted=True
    for label,item in opened.items(): revalidate(item,label)
    if inode(parent.lstat())!=parent_identity or inode(log_parent.lstat())!=log_parent_identity or inode(field_anchor.lstat())!=field_anchor_identity or inode(active_anchor.lstat())!=active_anchor_identity: die("output/log/upstream parent changed")
    core={"schema_version":"bernini-graft-phase-a-full-exact81-submission-receipt-v1","status":"submitted","submission_success":True,"job_success":None,"submitted_job":{"job_id":match.group(1),"scheduler_cluster":match.group(2),"stdout_sha256":digest(completed.stdout),"stderr_sha256":digest(completed.stderr)},"request":{"job_name":"graft-pa-full81-v1","partition":"faculty","qos":"bgqos","nodes":1,"ntasks":1,"cpus_per_task":64,"memory":"256G","gpu_resource_requested":"gpu:mi210:8","walltime":"72:00:00","world_size":8,"dp_size":2,"sp_size":4,"dependency":{"kind":"afterok","job_id":active14_job_id,"queue_gate_only":True,"inherits_weights":False,"checkpoint_consumed":False},"user_hold":False},"bindings":{"launcher_sha256":launcher_sha,"field14_plan_sha256":field14_plan_sha,"active14_plan_sha256":active14_plan_sha,"full81_plan_sha256":full81_plan_sha,"upstream_field14_job_id":field14_job_id,"upstream_field14_receipt_path":field14_receipt,"upstream_field14_receipt_may_be_absent_at_submission":True,"upstream_field14_receipt_sha256_policy":"derive-after-afterok","upstream_active14_job_id":active14_job_id,"upstream_active14_receipt_path":active14_receipt,"upstream_active14_receipt_may_be_absent_at_submission":True,"upstream_active14_receipt_sha256_policy":"derive-after-afterok"},"outputs":{"logical_output_root":output_root,"log_root":log_root,"submission_receipt_path":str(parent/receipt_name),"create_only":True,"mode_0444_is_terminal_submission_success_transition":True},"authority":{"action_authority":False,"identity_authority":False,"quality_authority":False,"training_authority":False,"checkpoint_authority":False,"publication_authority":False,"production_authority":False,"scientific_success_claimed":False,"job_success_claimed":False},"failure_semantics":{"submission_success_is_not_job_success":True,"automatic_job_cancellation_on_receipt_failure":False,"successful_scheduler_submission_may_exist_if_receipt_publication_fails":True,"no_user_hold":True}}
    receipt={**core,"receipt_digest":digest(canonical(core))}; publish_receipt(receipt_fd,parent_fd,canonical(receipt)+b"\n")
    print(json.dumps({"job_id":match.group(1),"state_expected":"PD_dependency_or_resource","dependency":active14_job_id,"submission_receipt":str(parent/receipt_name)},sort_keys=True),flush=True)
finally:
    for fd in reversed(retained):
        try: os.close(fd)
        except OSError: pass
    try: os.close(receipt_fd)
    except OSError: pass
    if not submitted:
        try: os.unlink(receipt_name,dir_fd=parent_fd)
        except OSError: pass
    os.close(active_anchor_fd); os.close(field_anchor_fd); os.close(parent_fd); os.close(log_parent_fd)
PY

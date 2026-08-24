#!/bin/bash -p

# No-argument retained-FD submission boundary for the active14 afterok job.
# Submission success is never interpreted as job or scientific success.

case "$-" in *p*) ;; *) echo "[submit-graft-pa-a14-v1] ERROR: privileged Bash required" >&2; exit 2;; esac
set -Eeuo pipefail
umask 077
fail() { echo "[submit-graft-pa-a14-v1] ERROR: $*" >&2; exit 2; }
[[ "$#" -eq 0 ]] || fail "arbitrary arguments are forbidden"

readonly required_launcher_sha256=d896b87dbc95dbcb65b80a0d635bc1dfd577f6a30a0dfc1d726ca23e1432efdb
readonly export_names_csv=GRAFT_ACTIVE14_SOURCE_ARCHIVE,GRAFT_ACTIVE14_SOURCE_ARCHIVE_SHA256,GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST,GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST_SHA256,GRAFT_ACTIVE14_PYTHON_BIN,GRAFT_ACTIVE14_PYTHON_SHA256,BERNINI_OFFICIAL_ROOT,BERNINI_VEOMNI_ROOT,BERNINI_ACTION_CHECKPOINT,BERNINI_CHECKPOINT_CONTENT_MANIFEST,BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256,GRAFT_ACTIVE14_FIELD14_PLAN,GRAFT_ACTIVE14_FIELD14_PLAN_SHA256,GRAFT_ACTIVE14_PLAN,GRAFT_ACTIVE14_PLAN_SHA256,GRAFT_ACTIVE14_TERMINAL_ADMISSION,GRAFT_ACTIVE14_TERMINAL_ADMISSION_SHA256,GRAFT_ACTIVE14_TERMINAL_MATERIALIZER_RUNTIME_SHA256,GRAFT_ACTIVE14_UPSTREAM_FIELD14_RECEIPT,GRAFT_ACTIVE14_UPSTREAM_FIELD14_JOB_ID,GRAFT_ACTIVE14_OUTPUT_ROOT,GRAFT_ACTIVE14_LAUNCHER_SOURCE,GRAFT_ACTIVE14_LAUNCHER_SHA256
readonly -a required_names=(
  GRAFT_ACTIVE14_SOURCE_ARCHIVE GRAFT_ACTIVE14_SOURCE_ARCHIVE_SHA256
  GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST_SHA256
  GRAFT_ACTIVE14_PYTHON_BIN GRAFT_ACTIVE14_PYTHON_SHA256
  BERNINI_OFFICIAL_ROOT BERNINI_VEOMNI_ROOT BERNINI_ACTION_CHECKPOINT
  BERNINI_CHECKPOINT_CONTENT_MANIFEST BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
  GRAFT_ACTIVE14_FIELD14_PLAN GRAFT_ACTIVE14_FIELD14_PLAN_SHA256
  GRAFT_ACTIVE14_PLAN GRAFT_ACTIVE14_PLAN_SHA256
  GRAFT_ACTIVE14_TERMINAL_ADMISSION GRAFT_ACTIVE14_TERMINAL_ADMISSION_SHA256
  GRAFT_ACTIVE14_TERMINAL_MATERIALIZER_RUNTIME_SHA256
  GRAFT_ACTIVE14_UPSTREAM_FIELD14_RECEIPT GRAFT_ACTIVE14_UPSTREAM_FIELD14_JOB_ID
  GRAFT_ACTIVE14_OUTPUT_ROOT GRAFT_ACTIVE14_LAUNCHER_SOURCE GRAFT_ACTIVE14_LAUNCHER_SHA256
  GRAFT_ACTIVE14_SUBMIT_WRAPPER_SHA256
)
for name in "${required_names[@]}"; do [[ -n "${!name}" ]] || fail "${name} must be nonempty"; done
[[ "${GRAFT_ACTIVE14_LAUNCHER_SHA256}" == "${required_launcher_sha256}" ]] || fail "launcher SHA differs"

exec /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 \
  "${GRAFT_ACTIVE14_PYTHON_BIN}" -I -S -B - \
  "$0" "${GRAFT_ACTIVE14_SUBMIT_WRAPPER_SHA256}" "${required_launcher_sha256}" "${export_names_csv}" \
  "${GRAFT_ACTIVE14_SOURCE_ARCHIVE}" "${GRAFT_ACTIVE14_SOURCE_ARCHIVE_SHA256}" \
  "${GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST}" "${GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST_SHA256}" \
  "${GRAFT_ACTIVE14_PYTHON_BIN}" "${GRAFT_ACTIVE14_PYTHON_SHA256}" \
  "${BERNINI_OFFICIAL_ROOT}" "${BERNINI_VEOMNI_ROOT}" "${BERNINI_ACTION_CHECKPOINT}" \
  "${BERNINI_CHECKPOINT_CONTENT_MANIFEST}" "${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}" \
  "${GRAFT_ACTIVE14_FIELD14_PLAN}" "${GRAFT_ACTIVE14_FIELD14_PLAN_SHA256}" \
  "${GRAFT_ACTIVE14_PLAN}" "${GRAFT_ACTIVE14_PLAN_SHA256}" \
  "${GRAFT_ACTIVE14_TERMINAL_ADMISSION}" "${GRAFT_ACTIVE14_TERMINAL_ADMISSION_SHA256}" \
  "${GRAFT_ACTIVE14_TERMINAL_MATERIALIZER_RUNTIME_SHA256}" \
  "${GRAFT_ACTIVE14_UPSTREAM_FIELD14_RECEIPT}" "${GRAFT_ACTIVE14_UPSTREAM_FIELD14_JOB_ID}" \
  "${GRAFT_ACTIVE14_OUTPUT_ROOT}" "${GRAFT_ACTIVE14_LAUNCHER_SOURCE}" "${GRAFT_ACTIVE14_LAUNCHER_SHA256}" <<'PY'
from __future__ import annotations
import hashlib,json,os,re,stat,subprocess,sys
from pathlib import Path

(wrapper_path,wrapper_sha,hardcoded_launcher_sha,export_csv,
 archive_path,archive_sha,closure_path,closure_sha,python_path,python_sha,
 bernini_root,veomni_root,checkpoint,checkpoint_manifest,checkpoint_manifest_sha,
 field_plan_path,field_plan_sha,active_plan_path,active_plan_sha,terminal_path,terminal_sha,
 materializer_runtime_sha,field_receipt_path,field_job_id,output_root,launcher_path,launcher_sha)=sys.argv[1:]

EXPORT_NAMES=tuple(export_csv.split(",")); SHA=re.compile(r"[0-9a-f]{64}\Z"); JOB=re.compile(r"([1-9][0-9]*)(?:;([A-Za-z0-9][A-Za-z0-9._-]*))?\Z")
EXPECTED_EXPORTS=("GRAFT_ACTIVE14_SOURCE_ARCHIVE","GRAFT_ACTIVE14_SOURCE_ARCHIVE_SHA256","GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST","GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST_SHA256","GRAFT_ACTIVE14_PYTHON_BIN","GRAFT_ACTIVE14_PYTHON_SHA256","BERNINI_OFFICIAL_ROOT","BERNINI_VEOMNI_ROOT","BERNINI_ACTION_CHECKPOINT","BERNINI_CHECKPOINT_CONTENT_MANIFEST","BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256","GRAFT_ACTIVE14_FIELD14_PLAN","GRAFT_ACTIVE14_FIELD14_PLAN_SHA256","GRAFT_ACTIVE14_PLAN","GRAFT_ACTIVE14_PLAN_SHA256","GRAFT_ACTIVE14_TERMINAL_ADMISSION","GRAFT_ACTIVE14_TERMINAL_ADMISSION_SHA256","GRAFT_ACTIVE14_TERMINAL_MATERIALIZER_RUNTIME_SHA256","GRAFT_ACTIVE14_UPSTREAM_FIELD14_RECEIPT","GRAFT_ACTIVE14_UPSTREAM_FIELD14_JOB_ID","GRAFT_ACTIVE14_OUTPUT_ROOT","GRAFT_ACTIVE14_LAUNCHER_SOURCE","GRAFT_ACTIVE14_LAUNCHER_SHA256")
def die(message): raise SystemExit(message)
def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def identity(info): return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_mode,info.st_nlink)
def inode(info): return (info.st_dev,info.st_ino)
def require_sha(value,label):
    if SHA.fullmatch(value) is None: die(label+" is not SHA256")
def open_plain(text,label,mode=None,nlink=None,executable=False):
    path=Path(text)
    if not path.is_absolute(): die(label+" is not absolute")
    before=path.lstat(); resolved=path.resolve(strict=True)
    if resolved!=path or stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode): die(label+" is not exact plain file")
    if mode is not None and stat.S_IMODE(before.st_mode)!=mode: die(label+" mode differs")
    if nlink is not None and before.st_nlink!=nlink: die(label+" nlink differs")
    if executable and not before.st_mode&0o111: die(label+" is not executable")
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)); raw=b""
    try:
        opened=os.fstat(fd)
        if identity(opened)!=identity(before): die(label+" changed opening")
        while True:
            block=os.read(fd,1024*1024)
            if not block: break
            raw+=block
        after=os.fstat(fd)
        if identity(opened)!=identity(after) or identity(after)!=identity(path.lstat()): die(label+" changed reading")
        os.lseek(fd,0,os.SEEK_SET)
        return path,fd,raw,identity(after)
    except BaseException:
        os.close(fd); raise
def stable_parent(path,label):
    if not path.is_absolute() or not path.name: die(label+" path differs")
    parent=path.parent; info=parent.lstat()
    if parent.resolve(strict=True)!=parent or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): die(label+" parent differs")
    fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    if inode(os.fstat(fd))!=inode(info): die(label+" parent changed")
    return parent,fd,inode(info)
def absent_at(parent_fd,name,label):
    try: os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    except FileNotFoundError: return
    die(label+" already exists")
def reserve_receipt(parent_fd,name):
    fd=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent_fd)
    opened=os.fstat(fd); leaf=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    if not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode)!=0o600 or opened.st_nlink!=1 or opened.st_size!=0 or inode(opened)!=inode(leaf): die("submission reservation differs")
    os.fsync(parent_fd)
    return fd,identity(opened)
def publish_reserved_receipt(parent_fd,name,fd,reserved,payload):
    if identity(os.fstat(fd))!=reserved or inode(os.stat(name,dir_fd=parent_fd,follow_symlinks=False))!=inode(os.fstat(fd)): die("submission reservation changed")
    offset=0
    while offset<len(payload):
        wrote=os.write(fd,payload[offset:])
        if wrote<=0: die("receipt write stalled")
        offset+=wrote
    os.fsync(fd); os.lseek(fd,0,os.SEEK_SET); observed=b""
    while True:
        block=os.read(fd,1024*1024)
        if not block: break
        observed+=block
    provisional=os.fstat(fd); leaf=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    if observed!=payload or stat.S_IMODE(provisional.st_mode)!=0o600 or provisional.st_nlink!=1 or inode(provisional)!=inode(leaf): die("provisional submission receipt differs")
    os.fchmod(fd,0o444); os.fsync(fd); os.fsync(parent_fd)
    final=os.fstat(fd); leaf=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    if stat.S_IMODE(final.st_mode)!=0o444 or final.st_nlink!=1 or inode(final)!=inode(leaf): die("terminal submission receipt differs")

if EXPORT_NAMES!=EXPECTED_EXPORTS: die("export interface differs")
if sys.platform=="darwin": os.environ.pop("__CF_USER_TEXT_ENCODING",None)
if hardcoded_launcher_sha!="d896b87dbc95dbcb65b80a0d635bc1dfd577f6a30a0dfc1d726ca23e1432efdb" or launcher_sha!=hardcoded_launcher_sha: die("launcher hardcode differs")
if not field_job_id.isdecimal() or field_job_id.startswith("0"): die("Field14 job ID differs")
for value,label in ((wrapper_sha,"wrapper"),(archive_sha,"archive"),(closure_sha,"closure"),(python_sha,"python"),(checkpoint_manifest_sha,"checkpoint manifest"),(field_plan_sha,"field plan"),(active_plan_sha,"active plan"),(terminal_sha,"terminal"),(materializer_runtime_sha,"materializer runtime"),(launcher_sha,"launcher")): require_sha(value,label)
output=Path(output_root); output_parent,output_fd,output_parent_identity=stable_parent(output,"output")
absent_at(output_fd,output.name,"output root"); submission_name=output.name+".submission.receipt.json"; absent_at(output_fd,submission_name,"submission receipt")
field_receipt=Path(field_receipt_path)
if not field_receipt.is_absolute() or field_receipt.name!="receipt.json" or not field_receipt.parent.name: die("Field14 receipt path differs")
field_output=field_receipt.parent
field_anchor,field_anchor_fd,field_anchor_identity=stable_parent(field_output,"Field14 output root")
receipt_fd=None
try:
    try:
        output_info=field_output.lstat()
    except FileNotFoundError:
        pass
    else:
        if field_output.resolve(strict=True)!=field_output or stat.S_ISLNK(output_info.st_mode) or not stat.S_ISDIR(output_info.st_mode): die("preexisting Field14 output root differs")
        try:
            info=field_receipt.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o444 or info.st_nlink!=1: die("preexisting Field14 receipt is not sealed")
    specs=((wrapper_path,"wrapper",0o444,1,False,wrapper_sha),(archive_path,"archive",0o444,1,False,archive_sha),(closure_path,"closure",0o444,1,False,closure_sha),(python_path,"python",None,None,True,python_sha),(checkpoint_manifest,"checkpoint manifest",0o444,1,False,checkpoint_manifest_sha),(field_plan_path,"field plan",0o444,1,False,field_plan_sha),(active_plan_path,"active plan",0o444,1,False,active_plan_sha),(terminal_path,"terminal",0o444,1,False,terminal_sha),(launcher_path,"launcher",0o555,1,True,launcher_sha),("/usr/bin/sbatch","sbatch",None,None,True,None))
    retained=[]; opened={}
    for text,label,mode,nlink,executable,wanted in specs:
        item=open_plain(text,label,mode,nlink,executable); retained.append(item[1]); opened[label]=item
        if wanted is not None and digest(item[2])!=wanted: die(label+" SHA differs")
    plan=json.loads(opened["active plan"][2].decode("ascii"))
    if opened["active plan"][2]!=canonical(plan)+b"\n": die("active plan not canonical")
    dependency=plan.get("field14_dependency")
    if dependency!={"job_id":field_job_id,"kind":"afterok","receipt_path":field_receipt_path,"receipt_sha256_policy":"derive-from-stable-sealed-file-after-afterok"}: die("active plan dependency differs")
    if plan.get("inherits_weights_from_dependency") is not False or plan.get("dependency_is_queue_gate_only") is not True: die("dependency semantics differ")
    terminal=json.loads(opened["terminal"][2].decode("ascii")); unsigned=dict(terminal); claimed=unsigned.pop("receipt_digest",None)
    if opened["terminal"][2]!=canonical(terminal)+b"\n" or claimed!=digest(canonical(unsigned)) or terminal.get("status")!="admitted" or terminal.get("sacct_admission",{}).get("job_id")!="132549" or terminal.get("sacct_admission",{}).get("state")!="COMPLETED" or terminal.get("sacct_admission",{}).get("exit_code")!="0:0" or terminal.get("materializer",{}).get("runtime_sha256")!=materializer_runtime_sha: die("terminal admission differs")
    if inode(output_parent.lstat())!=output_parent_identity: die("output parent changed before reservation")
    receipt_fd,reserved_identity=reserve_receipt(output_fd,submission_name)
    launcher_fd=opened["launcher"][1]; sbatch_fd=opened["sbatch"][1]
    launcher_transport=f"/proc/self/fd/{launcher_fd}"; sbatch_transport=f"/proc/self/fd/{sbatch_fd}"
    if inode(os.stat(launcher_transport))!=inode(os.fstat(launcher_fd)) or inode(os.stat(sbatch_transport))!=inode(os.fstat(sbatch_fd)): die("retained fd differs")
    values=(archive_path,archive_sha,closure_path,closure_sha,python_path,python_sha,bernini_root,veomni_root,checkpoint,checkpoint_manifest,checkpoint_manifest_sha,field_plan_path,field_plan_sha,active_plan_path,active_plan_sha,terminal_path,terminal_sha,materializer_runtime_sha,field_receipt_path,field_job_id,output_root,launcher_path,launcher_sha)
    exported=dict(zip(EXPORT_NAMES,values)); child_env={"PATH":"/usr/bin:/bin","LC_ALL":"C","LANG":"C",**exported}
    argv=[sbatch_transport,"--parsable",f"--export={','.join(EXPORT_NAMES)}",f"--dependency=afterok:{field_job_id}","--partition=faculty","--qos=bgqos","--nodes=1","--ntasks=1","--cpus-per-task=64","--mem=256G","--gres=gpu:mi210:8","--time=72:00:00","--job-name=graft-pa-a14-v1","--exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318",launcher_transport]
    completed=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,timeout=120,pass_fds=(launcher_fd,sbatch_fd),env=child_env)
    if completed.returncode!=0: die(f"sbatch failed rc={completed.returncode} stderr_sha={digest(completed.stderr)}")
    try: token=completed.stdout.decode("ascii")
    except UnicodeDecodeError: die("sbatch stdout not ASCII")
    if token.count("\n")!=1 or not token.endswith("\n"): die("sbatch stdout framing differs")
    match=JOB.fullmatch(token[:-1])
    if match is None: die("sbatch job ID differs")
    if inode(output_parent.lstat())!=output_parent_identity or inode(field_anchor.lstat())!=field_anchor_identity: die("parent identity changed")
    core={"schema_version":"bernini-graft-phase-a-active14-submission-receipt-v1","status":"submitted_afterok","submission_success":True,"job_success":None,"submitted_job":{"job_id":match.group(1),"scheduler_cluster":match.group(2),"stdout_sha256":digest(completed.stdout),"stderr_sha256":digest(completed.stderr)},"dependency":{"kind":"afterok","job_id":field_job_id,"receipt_path":field_receipt_path,"receipt_may_be_absent_at_submission":True,"weights_inherited":False},"request":{"job_name":"graft-pa-a14-v1","partition":"faculty","qos":"bgqos","nodes":1,"ntasks":1,"cpus_per_task":64,"memory":"256G","gpu_resource_requested":"gpu:mi210:8","walltime":"72:00:00","world_size":8,"dp_size":2,"sp_size":4},"submission_boundary":{"contains_export_all":False,"exact_job_export_names":list(EXPORT_NAMES),"sbatch_executed_from_retained_fd":True,"launcher_submitted_from_retained_fd":True,"launcher_sha256":launcher_sha,"active14_plan_sha256":active_plan_sha,"field14_plan_sha256":field_plan_sha,"source_archive_sha256":archive_sha,"runtime_closure_manifest_sha256":closure_sha},"outputs":{"logical_output_root":output_root,"submission_receipt_path":str(output_parent/submission_name),"create_only_O_EXCL_reservation_before_sbatch":True,"provisional_mode":"0600","final_success_mode":"0444","mode_0444_is_terminal_success_transition":True},"failure_semantics":{"submission_success_is_not_job_success":True,"provisional_non_success_inode_prevents_ambiguous_resubmission":True,"automatic_job_cancellation_on_receipt_failure":False},"authority":{"action_authority":False,"identity_authority":False,"quality_authority":False,"training_authority":False,"checkpoint_authority":False,"publication_authority":False,"production_authority":False,"scientific_success_claimed":False,"job_success_claimed":False}}
    receipt={**core,"receipt_digest":digest(canonical(core))}; publish_reserved_receipt(output_fd,submission_name,receipt_fd,reserved_identity,canonical(receipt)+b"\n")
finally:
    if receipt_fd is not None:
        try: os.close(receipt_fd)
        except OSError: pass
    for fd in locals().get("retained",[]):
        try: os.close(fd)
        except OSError: pass
    os.close(field_anchor_fd); os.close(output_fd)
PY

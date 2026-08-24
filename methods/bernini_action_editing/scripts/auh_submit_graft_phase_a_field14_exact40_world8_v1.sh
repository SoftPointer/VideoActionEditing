#!/bin/bash -p

# Clean exact-once submission boundary for the frozen Field14 WORLD8 job.
# A mode-0600 create-only receipt inode is reserved immediately before sbatch;
# only a fully revalidated successful submission transitions that same inode to
# canonical mode 0444.  A provisional inode is deliberately not success and
# prevents an ambiguous retry from submitting a duplicate job.

case "$-" in
  *p*) ;;
  *) echo "[submit-graft-pa-f14-v1] ERROR: Bash privileged mode is required" >&2; exit 2 ;;
esac
set -Eeuo pipefail
umask 077
fail() { echo "[submit-graft-pa-f14-v1] ERROR: $*" >&2; exit 2; }
[[ "$#" -eq 0 ]] || fail "arbitrary arguments are forbidden"

readonly required_sbatch_path=/usr/bin/sbatch
readonly required_fd_root=/proc/self/fd
readonly required_fd_stat_identity=true
readonly required_execute_sbatch_from_fd=true
readonly dependency_job_id=133524
readonly required_launcher_sha256=307cf29453f828e2618aee720adbe04c9b82bacce888ed705eb93f3711c527bb
readonly export_names_csv=GRAFT_FIELD14_SOURCE_ARCHIVE,GRAFT_FIELD14_SOURCE_ARCHIVE_SHA256,GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST,GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST_SHA256,GRAFT_FIELD14_PYTHON_BIN,GRAFT_FIELD14_PYTHON_SHA256,BERNINI_OFFICIAL_ROOT,BERNINI_VEOMNI_ROOT,BERNINI_ACTION_CHECKPOINT,BERNINI_CHECKPOINT_CONTENT_MANIFEST,BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256,GRAFT_FIELD14_PLAN,GRAFT_FIELD14_PLAN_SHA256,GRAFT_FIELD14_TERMINAL_ADMISSION,GRAFT_FIELD14_TERMINAL_ADMISSION_SHA256,GRAFT_FIELD14_TERMINAL_MATERIALIZER_RUNTIME_SHA256,GRAFT_FIELD14_OUTPUT_ROOT,GRAFT_FIELD14_LAUNCHER_SOURCE,GRAFT_FIELD14_LAUNCHER_SHA256
readonly -a required_names=(
  GRAFT_FIELD14_SOURCE_ARCHIVE
  GRAFT_FIELD14_SOURCE_ARCHIVE_SHA256
  GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST
  GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST_SHA256
  GRAFT_FIELD14_PYTHON_BIN
  GRAFT_FIELD14_PYTHON_SHA256
  BERNINI_OFFICIAL_ROOT
  BERNINI_VEOMNI_ROOT
  BERNINI_ACTION_CHECKPOINT
  BERNINI_CHECKPOINT_CONTENT_MANIFEST
  BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
  GRAFT_FIELD14_PLAN
  GRAFT_FIELD14_PLAN_SHA256
  GRAFT_FIELD14_TERMINAL_ADMISSION
  GRAFT_FIELD14_TERMINAL_ADMISSION_SHA256
  GRAFT_FIELD14_TERMINAL_MATERIALIZER_RUNTIME_SHA256
  GRAFT_FIELD14_OUTPUT_ROOT
  GRAFT_FIELD14_LAUNCHER_SOURCE
  GRAFT_FIELD14_LAUNCHER_SHA256
  GRAFT_FIELD14_SUBMIT_WRAPPER_SHA256
)

[[ "${dependency_job_id}" =~ ^[1-9][0-9]*$ ]] || fail "Field14 dependency job ID is not frozen numeric"
[[ "${required_launcher_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "Field14 launcher SHA-256 is not frozen"
observed_count=0
for variable_name in ${!GRAFT_FIELD14_*} ${!BERNINI_*}; do
  case "${variable_name}" in
    GRAFT_FIELD14_SOURCE_ARCHIVE|GRAFT_FIELD14_SOURCE_ARCHIVE_SHA256|GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST|GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST_SHA256|GRAFT_FIELD14_PYTHON_BIN|GRAFT_FIELD14_PYTHON_SHA256|BERNINI_OFFICIAL_ROOT|BERNINI_VEOMNI_ROOT|BERNINI_ACTION_CHECKPOINT|BERNINI_CHECKPOINT_CONTENT_MANIFEST|BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256|GRAFT_FIELD14_PLAN|GRAFT_FIELD14_PLAN_SHA256|GRAFT_FIELD14_TERMINAL_ADMISSION|GRAFT_FIELD14_TERMINAL_ADMISSION_SHA256|GRAFT_FIELD14_TERMINAL_MATERIALIZER_RUNTIME_SHA256|GRAFT_FIELD14_OUTPUT_ROOT|GRAFT_FIELD14_LAUNCHER_SOURCE|GRAFT_FIELD14_LAUNCHER_SHA256|GRAFT_FIELD14_SUBMIT_WRAPPER_SHA256) ;;
    *) fail "unexpected Field14 interface variable: ${variable_name}" ;;
  esac
  ((observed_count+=1))
done
[[ "${observed_count}" -eq 20 ]] || fail "exactly twenty Field14 interface variables are required"
for variable_name in "${required_names[@]}"; do
  [[ -n "${!variable_name}" ]] || fail "${variable_name} must be nonempty"
done
[[ "${GRAFT_FIELD14_LAUNCHER_SHA256}" == "${required_launcher_sha256}" ]] || fail "launcher SHA-256 differs from wrapper hardcode"

exec /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 \
  "${GRAFT_FIELD14_PYTHON_BIN}" -I -S -B - \
  "$0" "${GRAFT_FIELD14_SUBMIT_WRAPPER_SHA256}" "${required_sbatch_path}" \
  "${required_fd_root}" "${required_fd_stat_identity}" "${required_execute_sbatch_from_fd}" \
  "${required_launcher_sha256}" "${dependency_job_id}" "${export_names_csv}" \
  "${GRAFT_FIELD14_SOURCE_ARCHIVE}" "${GRAFT_FIELD14_SOURCE_ARCHIVE_SHA256}" \
  "${GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST}" "${GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST_SHA256}" \
  "${GRAFT_FIELD14_PYTHON_BIN}" "${GRAFT_FIELD14_PYTHON_SHA256}" \
  "${BERNINI_OFFICIAL_ROOT}" "${BERNINI_VEOMNI_ROOT}" "${BERNINI_ACTION_CHECKPOINT}" \
  "${BERNINI_CHECKPOINT_CONTENT_MANIFEST}" "${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}" \
  "${GRAFT_FIELD14_PLAN}" "${GRAFT_FIELD14_PLAN_SHA256}" \
  "${GRAFT_FIELD14_TERMINAL_ADMISSION}" "${GRAFT_FIELD14_TERMINAL_ADMISSION_SHA256}" \
  "${GRAFT_FIELD14_TERMINAL_MATERIALIZER_RUNTIME_SHA256}" "${GRAFT_FIELD14_OUTPUT_ROOT}" \
  "${GRAFT_FIELD14_LAUNCHER_SOURCE}" "${GRAFT_FIELD14_LAUNCHER_SHA256}" <<'PY'
from __future__ import annotations
import hashlib,json,os,re,stat,subprocess,sys
from pathlib import Path

(wrapper_path,wrapper_sha_external,sbatch_path,fd_root,require_fd_identity,execute_sbatch_from_fd,
 hardcoded_launcher_sha,dependency_job_id,export_names_csv,archive_path,archive_sha,closure_path,
 closure_sha,python_path,python_sha,bernini_root,veomni_root,checkpoint,checkpoint_manifest,
 checkpoint_manifest_sha,plan_path,plan_sha,terminal_path,terminal_sha,materializer_runtime_sha,
 output_root,launcher_path,launcher_sha)=sys.argv[1:]

EXPORT_NAMES=("GRAFT_FIELD14_SOURCE_ARCHIVE","GRAFT_FIELD14_SOURCE_ARCHIVE_SHA256","GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST","GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST_SHA256","GRAFT_FIELD14_PYTHON_BIN","GRAFT_FIELD14_PYTHON_SHA256","BERNINI_OFFICIAL_ROOT","BERNINI_VEOMNI_ROOT","BERNINI_ACTION_CHECKPOINT","BERNINI_CHECKPOINT_CONTENT_MANIFEST","BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256","GRAFT_FIELD14_PLAN","GRAFT_FIELD14_PLAN_SHA256","GRAFT_FIELD14_TERMINAL_ADMISSION","GRAFT_FIELD14_TERMINAL_ADMISSION_SHA256","GRAFT_FIELD14_TERMINAL_MATERIALIZER_RUNTIME_SHA256","GRAFT_FIELD14_OUTPUT_ROOT","GRAFT_FIELD14_LAUNCHER_SOURCE","GRAFT_FIELD14_LAUNCHER_SHA256")
SUPERVISOR_ENV={"PATH":"/usr/bin:/bin","LC_ALL":"C","LANG":"C","PYTHONDONTWRITEBYTECODE":"1"}
SCHEDULER_ARGS=("--parsable",f"--export={','.join(EXPORT_NAMES)}",f"--dependency=afterok:{dependency_job_id}","--partition=faculty","--qos=bgqos","--nodes=1","--ntasks=1","--cpus-per-task=64","--mem=256G","--gres=gpu:mi210:8","--time=48:00:00","--job-name=graft-pa-f14-v1","--exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318")
SHA=re.compile(r"[0-9a-f]{64}\Z"); JOB=re.compile(r"([1-9][0-9]*)(?:;([A-Za-z0-9][A-Za-z0-9._-]*))?\Z")
AUTHORITY12=("action_authority","identity_authority","cross_clip_identity_authority","quality_authority","training_authority","checkpoint_authority","publication_authority","production_authority","data_governance_authority","data_license_authority","scientific_success_claimed","semantic_action_editing_success_claimed")
AUTHORITY9=("action_authority","identity_authority","cross_clip_identity_authority","quality_authority","training_authority","production_authority","data_governance_authority","data_license_authority","scientific_success_claimed")
def die(message): raise SystemExit(message)
def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def identity(value): return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode,value.st_nlink)
def inode(value): return (value.st_dev,value.st_ino)
def require_sha(value,label):
    if not isinstance(value,str) or SHA.fullmatch(value) is None: die(label+" is not lowercase SHA-256")
def open_plain(text,label,mode=None,nlink=None,executable=False):
    path=Path(text)
    if not path.is_absolute(): die(label+" is not absolute")
    before=path.lstat(); resolved=path.resolve(strict=True)
    if resolved!=path or stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode): die(label+" is not an exact plain file")
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
        after=os.fstat(fd); leaf=path.lstat()
        if identity(opened)!=identity(after) or identity(after)!=identity(leaf): die(label+" changed while reading")
        raw=b"".join(chunks); os.lseek(fd,0,os.SEEK_SET)
        return path,fd,raw,identity(after)
    except BaseException:
        os.close(fd); raise
def revalidate(path,fd,raw,wanted,label):
    if identity(os.fstat(fd))!=wanted or identity(path.lstat())!=wanted: die(label+" path/fd identity changed")
    os.lseek(fd,0,os.SEEK_SET); chunks=[]
    while True:
        block=os.read(fd,1024*1024)
        if not block: break
        chunks.append(block)
    os.lseek(fd,0,os.SEEK_SET)
    if b"".join(chunks)!=raw: die(label+" bytes changed")
def open_output(text):
    output=Path(text)
    if not output.is_absolute() or not output.name or output.name in {".",".."}: die("output root leaf differs")
    parent=output.parent; before=parent.lstat()
    if parent.resolve(strict=True)!=parent or stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode): die("output parent differs")
    parent_fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); opened=os.fstat(parent_fd)
    if inode(opened)!=inode(before): die("output parent changed")
    receipt_name=output.name+".submission.receipt.json"
    for name in (output.name,receipt_name):
        try: os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
        except FileNotFoundError: continue
        else: die("submission receipt already exists or output root is not fresh: "+name)
    return output,parent,parent_fd,inode(opened),receipt_name
def reserve_receipt(parent_fd,name):
    try: fd=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent_fd)
    except FileExistsError: die("submission receipt already exists")
    opened=os.fstat(fd); leaf=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    if not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode)!=0o600 or opened.st_nlink!=1 or opened.st_size!=0 or inode(opened)!=inode(leaf): die("submission reservation identity differs")
    os.fsync(parent_fd)
    return fd,identity(opened)
def publish_reserved_receipt(parent_fd,name,fd,reserved,payload):
    if identity(os.fstat(fd))!=reserved or inode(os.stat(name,dir_fd=parent_fd,follow_symlinks=False))!=inode(os.fstat(fd)): die("submission reservation changed")
    os.ftruncate(fd,0); os.lseek(fd,0,os.SEEK_SET); offset=0
    while offset<len(payload):
        wrote=os.write(fd,payload[offset:])
        if wrote<=0: die("submission receipt write stalled")
        offset+=wrote
    os.fsync(fd); os.lseek(fd,0,os.SEEK_SET); observed=b""
    while True:
        block=os.read(fd,1024*1024)
        if not block: break
        observed+=block
    provisional=os.fstat(fd); leaf=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    if observed!=payload or not stat.S_ISREG(provisional.st_mode) or stat.S_IMODE(provisional.st_mode)!=0o600 or provisional.st_nlink!=1 or provisional.st_size!=len(payload) or inode(provisional)!=inode(leaf): die("provisional submission receipt differs")
    os.fsync(parent_fd); os.fchmod(fd,0o444); os.fsync(fd); os.fsync(parent_fd)
    final=os.fstat(fd); leaf=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    if stat.S_IMODE(final.st_mode)!=0o444 or final.st_nlink!=1 or inode(final)!=inode(leaf): die("terminal submission receipt transition differs")
def sealed(value,label,field="digest"):
    if not isinstance(value,dict): die(label+" is not mapping")
    unsigned=dict(value); claimed=unsigned.pop(field,None)
    require_sha(claimed,label+" "+field)
    if claimed!=digest(canonical(unsigned)): die(label+" digest differs")
def false_authority(value,names,label):
    if not isinstance(value,dict) or set(value)!=set(names) or any(value.get(name) is not False for name in names): die(label+" authority differs")

if tuple(export_names_csv.split(","))!=EXPORT_NAMES: die("export interface differs")
if not dependency_job_id.isdigit() or dependency_job_id.startswith("0"): die("dependency job ID is not frozen numeric")
if sys.platform=="darwin": os.environ.pop("__CF_USER_TEXT_ENCODING",None)
if dict(os.environ)!=SUPERVISOR_ENV or sys.flags.isolated!=1 or sys.flags.no_site!=1 or sys.flags.dont_write_bytecode!=1 or sys.flags.ignore_environment!=1: die("isolated supervisor differs")
for value,label in ((wrapper_sha_external,"external wrapper bootstrap trust anchor"),(archive_sha,"archive"),(closure_sha,"closure"),(python_sha,"Python external bootstrap trust anchor"),(checkpoint_manifest_sha,"checkpoint manifest"),(plan_sha,"plan"),(terminal_sha,"terminal external trust anchor"),(materializer_runtime_sha,"materializer runtime external trust anchor"),(launcher_sha,"launcher")): require_sha(value,label)
if hardcoded_launcher_sha!=launcher_sha: die("launcher hardcode differs")

output,parent,parent_fd,parent_identity,receipt_name=open_output(output_root); retained=[]; receipt_fd=None
try:
    specs=((wrapper_path,"wrapper",0o444,1,False,wrapper_sha_external),(archive_path,"archive",0o444,1,False,archive_sha),(closure_path,"closure",0o444,1,False,closure_sha),(checkpoint_manifest,"checkpoint manifest",0o444,1,False,checkpoint_manifest_sha),(plan_path,"plan",0o444,1,False,plan_sha),(terminal_path,"terminal admission",0o444,1,False,terminal_sha),(python_path,"configured Python",None,None,True,python_sha),(launcher_path,"launcher",0o555,1,True,launcher_sha),(sbatch_path,"sbatch",None,None,True,None))
    opened={}
    for text,label,mode,nlink,executable,wanted_sha in specs:
        item=open_plain(text,label,mode=mode,nlink=nlink,executable=executable); retained.append(item[1]); opened[label]=item
        if wanted_sha is not None and digest(item[2])!=wanted_sha: die(label+" SHA-256 differs")
    if Path(sys.executable).resolve(strict=True)!=Path(python_path).resolve(strict=True): die("running Python path differs")
    plan=json.loads(opened["plan"][2].decode("ascii"))
    if opened["plan"][2]!=canonical(plan)+b"\n": die("plan is not canonical ASCII JSON newline")
    if plan.get("afterok_dependency")!={"job_id":dependency_job_id,"kind":"afterok","purpose":"queue-gate-only"} or plan.get("afterok_is_queue_gate_only") is not True or plan.get("inherits_weights_from_dependency") is not False or plan.get("resources")!={"cpus_per_task":64,"gpus":8,"memory_gib":256,"nodes":1,"ntasks":1,"time_limit_hours":48}: die("plan dependency/resources differ")
    false_authority(plan.get("authority"),AUTHORITY12,"plan")
    terminal=json.loads(opened["terminal admission"][2].decode("ascii"))
    if opened["terminal admission"][2]!=canonical(terminal)+b"\n": die("terminal admission is not canonical ASCII JSON newline")
    sealed(terminal,"terminal admission",field="receipt_digest")
    materializer=terminal.get("materializer",{}); sacct=terminal.get("sacct_admission",{}); bindings=terminal.get("artifact_bindings",{})
    false_authority(terminal.get("authority"),AUTHORITY9,"terminal admission")
    materializer_keys={"schema_version","implementation_sha256","runtime_sha256","independent_of_submitted_job_process","job_process_wrote_this_receipt","observed_after_job_became_terminal"}
    sacct_keys={"source","queried_fields","job_id","state","exit_code","terminal_state_observed","job_success","raw_stdout_sha256","raw_stdout_size_bytes","selected_record_sha256"}
    binding_keys={"manifest_file_sha256","producer_receipt_file_sha256","producer_receipt_digest","execution_receipt_file_sha256","execution_receipt_digest","submission_receipt_file_sha256","submission_receipt_digest"}
    if set(terminal)!={"schema_version","status","materializer","sacct_admission","artifact_bindings","authority","receipt_digest"} or set(materializer)!=materializer_keys or set(sacct)!=sacct_keys or set(bindings)!=binding_keys or terminal.get("schema_version")!="bernini-graft-a-lite-source-independent-sacct-admission-v1" or terminal.get("status")!="admitted" or materializer.get("schema_version")!="bernini-graft-independent-sacct-admission-materializer-v1" or materializer.get("implementation_sha256")!="4686463642a38e771c6858d1c10fc6aacb815a56e4f3eae951a336018d186cf4" or materializer.get("runtime_sha256")!=materializer_runtime_sha or materializer.get("independent_of_submitted_job_process") is not True or materializer.get("job_process_wrote_this_receipt") is not False or materializer.get("observed_after_job_became_terminal") is not True or sacct.get("source")!="sacct" or sacct.get("queried_fields")!=["JobIDRaw","State","ExitCode"] or sacct.get("job_id")!="132549" or sacct.get("state")!="COMPLETED" or sacct.get("exit_code")!="0:0" or sacct.get("terminal_state_observed") is not True or sacct.get("job_success") is not True or type(sacct.get("raw_stdout_size_bytes")) is not int or sacct["raw_stdout_size_bytes"]<=0: die("terminal admission contract differs")
    for name in ("raw_stdout_sha256","selected_record_sha256"): require_sha(sacct.get(name),"terminal "+name)
    for name in binding_keys: require_sha(bindings.get(name),"terminal "+name)
    release=plan.get("release",{}); artifacts=release.get("artifacts",{})
    if release.get("job_id")!="132549" or release.get("terminal_admission",{}).get("path")!=terminal_path: die("plan source release differs")
    expected_bindings={"manifest_file_sha256":artifacts.get("manifest",{}).get("sha256"),"producer_receipt_file_sha256":artifacts.get("producer",{}).get("sha256"),"execution_receipt_file_sha256":artifacts.get("execution",{}).get("sha256"),"submission_receipt_file_sha256":artifacts.get("submission",{}).get("sha256")}
    if any(bindings.get(name)!=value for name,value in expected_bindings.items()): die("terminal artifact bindings differ")
    for label,item in opened.items(): revalidate(item[0],item[1],item[2],item[3],label)
    if inode(parent.lstat())!=parent_identity: die("output parent changed before reservation")
    receipt_fd,reserved_identity=reserve_receipt(parent_fd,receipt_name)
    launcher_fd=opened["launcher"][1]; sbatch_fd=opened["sbatch"][1]; fdroot=Path(fd_root)
    if not fdroot.is_absolute() or not fdroot.is_dir(): die("retained fd transport unavailable")
    launcher_transport=f"{fd_root.rstrip('/')}/{launcher_fd}"; sbatch_transport=f"{fd_root.rstrip('/')}/{sbatch_fd}"
    if require_fd_identity=="true" and (inode(os.stat(launcher_transport))!=inode(os.fstat(launcher_fd)) or inode(os.stat(sbatch_transport))!=inode(os.fstat(sbatch_fd))): die("retained fd transport identity differs")
    values=(archive_path,archive_sha,closure_path,closure_sha,python_path,python_sha,bernini_root,veomni_root,checkpoint,checkpoint_manifest,checkpoint_manifest_sha,plan_path,plan_sha,terminal_path,terminal_sha,materializer_runtime_sha,output_root,launcher_path,launcher_sha)
    child_env={"PATH":"/usr/bin:/bin","LC_ALL":"C","LANG":"C",**dict(zip(EXPORT_NAMES,values))}
    executable=sbatch_transport if execute_sbatch_from_fd=="true" else sbatch_path
    completed=subprocess.run([executable,*SCHEDULER_ARGS,launcher_transport],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,timeout=120,pass_fds=(launcher_fd,sbatch_fd),env=child_env)
    if completed.returncode!=0: die(f"sbatch failed exit={completed.returncode} stderr_sha256={digest(completed.stderr)}")
    try: token=completed.stdout.decode("ascii")
    except UnicodeDecodeError: die("sbatch stdout is not ASCII")
    if token.count("\n")!=1 or not token.endswith("\n"): die("sbatch stdout framing differs")
    match=JOB.fullmatch(token[:-1])
    if match is None: die("sbatch parsable job ID differs")
    for label,item in opened.items(): revalidate(item[0],item[1],item[2],item[3],label)
    if dict(os.environ)!=SUPERVISOR_ENV or inode(parent.lstat())!=parent_identity: die("supervisor/output parent changed after sbatch")
    authority={name:False for name in AUTHORITY12}
    core={"schema_version":"bernini-graft-phase-a-field14-submission-receipt-v1","status":"submitted","submission_success":True,"job_success":None,"job_terminal_state_observed":False,"submitted_job":{"job_id":match.group(1),"scheduler_cluster":match.group(2),"stdout_sha256":digest(completed.stdout),"stderr_sha256":digest(completed.stderr)},"dependency":{"kind":"afterok","job_id":dependency_job_id,"queue_gate_only":True,"inherits_weights":False},"request":{"job_name":"graft-pa-f14-v1","partition":"faculty","qos":"bgqos","nodes":1,"ntasks":1,"cpus_per_task":64,"memory":"256G","gpu_resource_requested":"gpu:mi210:8","walltime":"48:00:00","world_size":8,"dp_size":2,"sp_size":4},"submission_boundary":{"environment_replaced_before_supervisor":True,"exact_supervisor_interface_names":[*EXPORT_NAMES,"GRAFT_FIELD14_SUBMIT_WRAPPER_SHA256"],"exact_job_export_names":list(EXPORT_NAMES),"contains_export_all":False,"sbatch_executed_from_retained_fd":execute_sbatch_from_fd=="true","launcher_submitted_from_retained_fd":True,"python_wrapper_sbatch_launcher_retained_and_revalidated":True,"launcher_sha256":launcher_sha,"plan_sha256":plan_sha},"outputs":{"logical_output_root":output_root,"submission_receipt_path":str(parent/receipt_name),"create_only_O_EXCL_reservation_before_sbatch":True,"provisional_mode":"0600","final_success_mode":"0444","mode_0444_is_terminal_success_transition":True},"checkpoint_written":False,"checkpoint_payload_returned":False,"publication_performed":False,"authority":authority,"failure_semantics":{"submission_success_is_not_job_success":True,"provisional_non_success_inode_prevents_ambiguous_resubmission":True,"automatic_job_cancellation_on_receipt_failure":False}}
    receipt={**core,"receipt_digest":digest(canonical(core))}; publish_reserved_receipt(parent_fd,receipt_name,receipt_fd,reserved_identity,canonical(receipt)+b"\n")
finally:
    if receipt_fd is not None:
        try: os.close(receipt_fd)
        except OSError: pass
    for fd in reversed(retained):
        try: os.close(fd)
        except OSError: pass
    os.close(parent_fd)
PY

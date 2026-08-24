#!/bin/bash -p

# Clean no-argument submission boundary.  The external terminal-admission SHA
# and materializer-runtime SHA are mandatory inputs; neither is learned from
# the terminal receipt.  Both sbatch and the sealed launcher are retained by
# descriptor through submission.  Success means submitted, never job success.

case "$-" in
  *p*) ;;
  *) echo "[submit-graft-pa-short-v1] ERROR: Bash privileged mode is required" >&2; exit 2 ;;
esac
set -Eeuo pipefail
umask 077
fail() { echo "[submit-graft-pa-short-v1] ERROR: $*" >&2; exit 2; }
[[ "$#" -eq 0 ]] || fail "arbitrary arguments are forbidden"

readonly required_sbatch_path=/usr/bin/sbatch
readonly required_fd_root=/proc/self/fd
readonly required_fd_stat_identity=true
readonly required_execute_sbatch_from_fd=true
readonly required_launcher_sha256=c62ee713e0309b6e0441b12375573d9e4cd7dc5ce94e5db652b8319dee2357a9
readonly export_names_csv=GRAFT_SHORT_SOURCE_ARCHIVE,GRAFT_SHORT_SOURCE_ARCHIVE_SHA256,GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST,GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST_SHA256,GRAFT_SHORT_PYTHON_BIN,GRAFT_SHORT_PYTHON_SHA256,BERNINI_OFFICIAL_ROOT,BERNINI_VEOMNI_ROOT,BERNINI_ACTION_CHECKPOINT,BERNINI_CHECKPOINT_CONTENT_MANIFEST,BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256,GRAFT_SHORT_PLAN,GRAFT_SHORT_PLAN_SHA256,GRAFT_SHORT_TERMINAL_ADMISSION,GRAFT_SHORT_TERMINAL_ADMISSION_SHA256,GRAFT_SHORT_TERMINAL_MATERIALIZER_RUNTIME_SHA256,GRAFT_SHORT_OUTPUT_ROOT,GRAFT_SHORT_LAUNCHER_SOURCE,GRAFT_SHORT_LAUNCHER_SHA256
readonly -a required_names=(
  GRAFT_SHORT_SOURCE_ARCHIVE
  GRAFT_SHORT_SOURCE_ARCHIVE_SHA256
  GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST
  GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST_SHA256
  GRAFT_SHORT_PYTHON_BIN
  GRAFT_SHORT_PYTHON_SHA256
  BERNINI_OFFICIAL_ROOT
  BERNINI_VEOMNI_ROOT
  BERNINI_ACTION_CHECKPOINT
  BERNINI_CHECKPOINT_CONTENT_MANIFEST
  BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
  GRAFT_SHORT_PLAN
  GRAFT_SHORT_PLAN_SHA256
  GRAFT_SHORT_TERMINAL_ADMISSION
  GRAFT_SHORT_TERMINAL_ADMISSION_SHA256
  GRAFT_SHORT_TERMINAL_MATERIALIZER_RUNTIME_SHA256
  GRAFT_SHORT_OUTPUT_ROOT
  GRAFT_SHORT_LAUNCHER_SOURCE
  GRAFT_SHORT_LAUNCHER_SHA256
  GRAFT_SHORT_SUBMIT_WRAPPER_SHA256
)

observed_count=0
for variable_name in ${!GRAFT_SHORT_*} ${!BERNINI_*}; do
  case "${variable_name}" in
    GRAFT_SHORT_SOURCE_ARCHIVE|GRAFT_SHORT_SOURCE_ARCHIVE_SHA256|GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST|GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST_SHA256|GRAFT_SHORT_PYTHON_BIN|GRAFT_SHORT_PYTHON_SHA256|BERNINI_OFFICIAL_ROOT|BERNINI_VEOMNI_ROOT|BERNINI_ACTION_CHECKPOINT|BERNINI_CHECKPOINT_CONTENT_MANIFEST|BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256|GRAFT_SHORT_PLAN|GRAFT_SHORT_PLAN_SHA256|GRAFT_SHORT_TERMINAL_ADMISSION|GRAFT_SHORT_TERMINAL_ADMISSION_SHA256|GRAFT_SHORT_TERMINAL_MATERIALIZER_RUNTIME_SHA256|GRAFT_SHORT_OUTPUT_ROOT|GRAFT_SHORT_LAUNCHER_SOURCE|GRAFT_SHORT_LAUNCHER_SHA256|GRAFT_SHORT_SUBMIT_WRAPPER_SHA256) ;;
    *) fail "unexpected short-run interface variable: ${variable_name}" ;;
  esac
  ((observed_count+=1))
done
[[ "${observed_count}" -eq 20 ]] || fail "exactly twenty short-run interface variables are required"
for variable_name in "${required_names[@]}"; do [[ -n "${!variable_name}" ]] || fail "${variable_name} must be nonempty"; done
[[ "${GRAFT_SHORT_LAUNCHER_SHA256}" == "${required_launcher_sha256}" ]] || fail "launcher SHA-256 differs from wrapper hardcode"

exec /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 \
  "${GRAFT_SHORT_PYTHON_BIN}" -I -S -B - \
  "$0" "${GRAFT_SHORT_SUBMIT_WRAPPER_SHA256}" "${required_sbatch_path}" "${required_fd_root}" "${required_fd_stat_identity}" "${required_execute_sbatch_from_fd}" "${required_launcher_sha256}" "${export_names_csv}" \
  "${GRAFT_SHORT_SOURCE_ARCHIVE}" "${GRAFT_SHORT_SOURCE_ARCHIVE_SHA256}" \
  "${GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST}" "${GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST_SHA256}" \
  "${GRAFT_SHORT_PYTHON_BIN}" "${GRAFT_SHORT_PYTHON_SHA256}" \
  "${BERNINI_OFFICIAL_ROOT}" "${BERNINI_VEOMNI_ROOT}" "${BERNINI_ACTION_CHECKPOINT}" \
  "${BERNINI_CHECKPOINT_CONTENT_MANIFEST}" "${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}" \
  "${GRAFT_SHORT_PLAN}" "${GRAFT_SHORT_PLAN_SHA256}" \
  "${GRAFT_SHORT_TERMINAL_ADMISSION}" "${GRAFT_SHORT_TERMINAL_ADMISSION_SHA256}" \
  "${GRAFT_SHORT_TERMINAL_MATERIALIZER_RUNTIME_SHA256}" "${GRAFT_SHORT_OUTPUT_ROOT}" \
  "${GRAFT_SHORT_LAUNCHER_SOURCE}" "${GRAFT_SHORT_LAUNCHER_SHA256}" <<'PY'
from __future__ import annotations
import hashlib,json,os,re,stat,subprocess,sys
from pathlib import Path

(wrapper_path,wrapper_sha_external,sbatch_path,fd_root,require_fd_identity,execute_sbatch_from_fd,hardcoded_launcher_sha,export_names_csv,
 archive_path,archive_sha,closure_path,closure_sha,python_path,python_sha,bernini_root,veomni_root,checkpoint,
 checkpoint_manifest,checkpoint_manifest_sha,plan_path,plan_sha,terminal_path,terminal_sha,materializer_runtime_sha,
 output_root,launcher_path,launcher_sha)=sys.argv[1:]

EXPORT_NAMES=("GRAFT_SHORT_SOURCE_ARCHIVE","GRAFT_SHORT_SOURCE_ARCHIVE_SHA256","GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST","GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST_SHA256","GRAFT_SHORT_PYTHON_BIN","GRAFT_SHORT_PYTHON_SHA256","BERNINI_OFFICIAL_ROOT","BERNINI_VEOMNI_ROOT","BERNINI_ACTION_CHECKPOINT","BERNINI_CHECKPOINT_CONTENT_MANIFEST","BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256","GRAFT_SHORT_PLAN","GRAFT_SHORT_PLAN_SHA256","GRAFT_SHORT_TERMINAL_ADMISSION","GRAFT_SHORT_TERMINAL_ADMISSION_SHA256","GRAFT_SHORT_TERMINAL_MATERIALIZER_RUNTIME_SHA256","GRAFT_SHORT_OUTPUT_ROOT","GRAFT_SHORT_LAUNCHER_SOURCE","GRAFT_SHORT_LAUNCHER_SHA256")
SUPERVISOR_ENV={"PATH":"/usr/bin:/bin","LC_ALL":"C","LANG":"C","PYTHONDONTWRITEBYTECODE":"1"}
SCHEDULER_ARGS=("--parsable",f"--export={','.join(EXPORT_NAMES)}","--partition=faculty","--qos=bgqos","--nodes=1","--ntasks=1","--cpus-per-task=32","--mem=256G","--gres=gpu:mi210:8","--time=08:00:00","--job-name=graft-pa-short-v1","--exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318")
SHA=re.compile(r"[0-9a-f]{64}\Z"); JOB=re.compile(r"([1-9][0-9]*)(?:;([A-Za-z0-9][A-Za-z0-9._-]*))?\Z")
def die(m): raise SystemExit(m)
def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def identity(s): return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_nlink)
def inode(s): return (s.st_dev,s.st_ino)
def require_sha(v,label):
    if SHA.fullmatch(v) is None: die(label+" is not lowercase SHA-256")
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
        return path,fd,raw,identity(after),{"path":str(path),"sha256":digest(raw),"size_bytes":len(raw),"mode":format(stat.S_IMODE(after.st_mode),"04o"),"link_count":after.st_nlink,"device":after.st_dev,"inode":after.st_ino}
    except BaseException:
        os.close(fd); raise
def revalidate(path,fd,raw,wanted,label):
    if identity(os.fstat(fd))!=wanted or identity(path.lstat())!=wanted: die(label+" path/fd identity changed")
    os.lseek(fd,0,os.SEEK_SET); observed=b""
    while True:
        block=os.read(fd,1024*1024)
        if not block: break
        observed+=block
    os.lseek(fd,0,os.SEEK_SET)
    if observed!=raw: die(label+" bytes changed")
def open_output(text):
    output=Path(text)
    if not output.is_absolute() or not output.name or output.name in {".",".."}: die("output root leaf differs")
    parent=output.parent; before=parent.lstat()
    if parent.resolve(strict=True)!=parent or stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode): die("output parent differs")
    fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); opened=os.fstat(fd)
    if inode(opened)!=inode(before): die("output parent changed")
    for name in (output.name,output.name+".submission.receipt.json"):
        try: os.stat(name,dir_fd=fd,follow_symlinks=False)
        except FileNotFoundError: continue
        else: die("create-only output exists: "+name)
    return output,parent,fd,inode(opened)
def create_receipt(fd,name,payload):
    out=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
    try:
        offset=0
        while offset<len(payload):
            wrote=os.write(out,payload[offset:])
            if wrote<=0: die("receipt write stalled")
            offset+=wrote
        os.fsync(out); os.lseek(out,0,os.SEEK_SET)
        if os.read(out,len(payload)+1)!=payload: die("receipt reread differs")
        opened=os.fstat(out); leaf=os.stat(name,dir_fd=fd,follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode)!=0o600 or opened.st_nlink!=1 or opened.st_size!=len(payload) or inode(opened)!=inode(leaf): die("provisional receipt identity differs")
        os.fsync(fd)
        os.fchmod(out,0o444)
        os._exit(0)
    except BaseException:
        try:
            opened=os.fstat(out); leaf=os.stat(name,dir_fd=fd,follow_symlinks=False)
            if inode(opened)==inode(leaf) and stat.S_IMODE(opened.st_mode)!=0o444: os.unlink(name,dir_fd=fd)
        except OSError: pass
        os.close(out)
        raise

if tuple(export_names_csv.split(","))!=EXPORT_NAMES: die("export interface differs")
if hardcoded_launcher_sha!="c62ee713e0309b6e0441b12375573d9e4cd7dc5ce94e5db652b8319dee2357a9" or launcher_sha!=hardcoded_launcher_sha: die("launcher hardcode differs")
# macOS may synthesize this CoreFoundation locale hint after env -i.  Remove
# only that platform-owned key before enforcing and forwarding the exact clean
# supervisor environment; Linux/AUH retains the stricter no-exception check.
if sys.platform=="darwin": os.environ.pop("__CF_USER_TEXT_ENCODING",None)
if dict(os.environ)!=SUPERVISOR_ENV or sys.flags.isolated!=1 or sys.flags.no_site!=1 or sys.flags.dont_write_bytecode!=1 or sys.flags.ignore_environment!=1: die("isolated supervisor differs")
for value,label in ((wrapper_sha_external,"external submit-wrapper bootstrap trust anchor"),(archive_sha,"archive"),(closure_sha,"closure"),(python_sha,"Python external bootstrap trust anchor"),(checkpoint_manifest_sha,"checkpoint manifest"),(plan_sha,"plan"),(terminal_sha,"external terminal trust anchor"),(materializer_runtime_sha,"external materializer runtime trust anchor"),(launcher_sha,"launcher")): require_sha(value,label)
if plan_sha!="ab2eb2e7b93341b47498184821761eb8e5c924f9dd8460284087e23a27ba34d8" or checkpoint_manifest_sha!="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831": die("fixed plan/checkpoint authority differs")

output,parent,parent_fd,parent_identity=open_output(output_root); retained=[]
try:
    specs=((wrapper_path,"wrapper",0o444,1,False,wrapper_sha_external),(archive_path,"archive",0o444,1,False,archive_sha),(closure_path,"closure",0o444,1,False,closure_sha),(checkpoint_manifest,"checkpoint manifest",0o444,1,False,checkpoint_manifest_sha),(plan_path,"plan",0o444,1,False,plan_sha),(terminal_path,"terminal admission",0o444,1,False,terminal_sha),(python_path,"configured Python",None,None,True,python_sha),(launcher_path,"launcher",0o555,1,True,launcher_sha),(sbatch_path,"sbatch",None,None,True,None))
    opened={}
    for text,label,mode,nlink,executable,wanted_sha in specs:
        item=open_plain(text,label,mode=mode,nlink=nlink,executable=executable); retained.append(item[1]); opened[label]=item
        if wanted_sha is not None and digest(item[2])!=wanted_sha: die(label+" SHA-256 differs")
    if Path(sys.executable).resolve(strict=True)!=Path(python_path): die("running Python path differs")
    terminal_raw=opened["terminal admission"][2]
    terminal=json.loads(terminal_raw.decode("ascii"))
    if terminal_raw!=canonical(terminal)+b"\n": die("terminal admission is not canonical ASCII JSON newline")
    unsigned=dict(terminal); claimed=unsigned.pop("receipt_digest",None)
    materializer=terminal.get("materializer",{}); sacct=terminal.get("sacct_admission",{})
    bindings=terminal.get("artifact_bindings",{}); terminal_authority=terminal.get("authority",{})
    terminal_keys={"schema_version","status","materializer","sacct_admission","artifact_bindings","authority","receipt_digest"}
    materializer_keys={"schema_version","implementation_sha256","runtime_sha256","independent_of_submitted_job_process","job_process_wrote_this_receipt","observed_after_job_became_terminal"}
    sacct_keys={"source","queried_fields","job_id","state","exit_code","terminal_state_observed","job_success","raw_stdout_sha256","raw_stdout_size_bytes","selected_record_sha256"}
    binding_keys={"manifest_file_sha256","producer_receipt_file_sha256","producer_receipt_digest","execution_receipt_file_sha256","execution_receipt_digest","submission_receipt_file_sha256","submission_receipt_digest"}
    terminal_authority_keys={"action_authority","identity_authority","cross_clip_identity_authority","quality_authority","training_authority","production_authority","data_governance_authority","data_license_authority","scientific_success_claimed"}
    if (set(terminal)!=terminal_keys or set(materializer)!=materializer_keys or set(sacct)!=sacct_keys or set(bindings)!=binding_keys or not isinstance(terminal_authority,dict) or set(terminal_authority)!=terminal_authority_keys or any(value is not False for value in terminal_authority.values()) or claimed!=digest(canonical(unsigned)) or terminal.get("schema_version")!="bernini-graft-a-lite-source-independent-sacct-admission-v1" or terminal.get("status")!="admitted" or materializer.get("schema_version")!="bernini-graft-independent-sacct-admission-materializer-v1" or materializer.get("implementation_sha256")!="4686463642a38e771c6858d1c10fc6aacb815a56e4f3eae951a336018d186cf4" or materializer.get("runtime_sha256")!=materializer_runtime_sha or materializer.get("independent_of_submitted_job_process") is not True or materializer.get("job_process_wrote_this_receipt") is not False or materializer.get("observed_after_job_became_terminal") is not True or sacct.get("source")!="sacct" or sacct.get("queried_fields")!=["JobIDRaw","State","ExitCode"] or sacct.get("job_id")!="132549" or sacct.get("state")!="COMPLETED" or sacct.get("exit_code")!="0:0" or sacct.get("terminal_state_observed") is not True or sacct.get("job_success") is not True or type(sacct.get("raw_stdout_size_bytes")) is not int or sacct["raw_stdout_size_bytes"]<=0): die("terminal admission did not prove fixed job132549 completion")
    for name in ("raw_stdout_sha256","selected_record_sha256"): require_sha(sacct.get(name),"terminal "+name)
    for name in ("producer_receipt_digest","execution_receipt_digest","submission_receipt_digest"): require_sha(bindings.get(name),"terminal "+name)
    plan=json.loads(opened["plan"][2].decode("ascii")); release=plan.get("release",{})
    if release.get("job_id")!="132549" or release.get("terminal_admission",{}).get("path")!=terminal_path or release.get("terminal_admission",{}).get("sha256_source")!="external-pre-submission-pin" or release.get("terminal_materializer",{}).get("runtime_sha256_source")!="external-pre-submission-pin": die("plan terminal external-pin contract differs")
    artifacts=release.get("artifacts",{})
    expected_bindings={"manifest_file_sha256":artifacts.get("manifest",{}).get("sha256"),"producer_receipt_file_sha256":artifacts.get("producer",{}).get("sha256"),"execution_receipt_file_sha256":artifacts.get("execution",{}).get("sha256"),"submission_receipt_file_sha256":artifacts.get("submission",{}).get("sha256")}
    if any(bindings.get(name)!=value for name,value in expected_bindings.items()): die("terminal admission artifact binding differs from fixed plan")
    launcher_fd=opened["launcher"][1]; sbatch_fd=opened["sbatch"][1]
    fdroot=Path(fd_root)
    if not fdroot.is_absolute() or not fdroot.is_dir(): die("retained fd transport unavailable")
    launcher_transport=f"{fd_root.rstrip('/')}/{launcher_fd}"; sbatch_transport=f"{fd_root.rstrip('/')}/{sbatch_fd}"
    if require_fd_identity=="true" and (inode(os.stat(launcher_transport))!=inode(os.fstat(launcher_fd)) or inode(os.stat(sbatch_transport))!=inode(os.fstat(sbatch_fd))): die("retained fd transport identity differs")
    values=(archive_path,archive_sha,closure_path,closure_sha,python_path,python_sha,bernini_root,veomni_root,checkpoint,checkpoint_manifest,checkpoint_manifest_sha,plan_path,plan_sha,terminal_path,terminal_sha,materializer_runtime_sha,output_root,launcher_path,launcher_sha)
    exported=dict(zip(EXPORT_NAMES,values)); child_env={"PATH":"/usr/bin:/bin","LC_ALL":"C","LANG":"C",**exported}
    executable=sbatch_transport if execute_sbatch_from_fd=="true" else sbatch_path
    argv=[executable,*SCHEDULER_ARGS,launcher_transport]
    completed=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,timeout=120,pass_fds=(launcher_fd,sbatch_fd),env=child_env)
    if completed.returncode!=0: die(f"sbatch failed exit={completed.returncode} stderr_sha256={digest(completed.stderr)}")
    try: token=completed.stdout.decode("ascii")
    except UnicodeDecodeError: die("sbatch stdout is not ASCII")
    if token.count("\n")!=1 or not token.endswith("\n"): die("sbatch stdout framing differs")
    match=JOB.fullmatch(token[:-1])
    if match is None: die("sbatch parsable job id differs")
    for path,fd,raw,wanted,obs in opened.values(): revalidate(path,fd,raw,wanted,obs["path"])
    if dict(os.environ)!=SUPERVISOR_ENV or inode(parent.lstat())!=parent_identity: die("supervisor/output parent changed after sbatch")
    core={"schema_version":"bernini-graft-phase-a-a-lite-short-submission-receipt-v1","status":"submitted","submission_success":True,"job_success":None,"job_terminal_state_observed":False,"submitted_job":{"job_id":match.group(1),"scheduler_cluster":match.group(2),"stdout_sha256":digest(completed.stdout),"stderr_sha256":digest(completed.stderr)},"request":{"job_name":"graft-pa-short-v1","partition":"faculty","qos":"bgqos","nodes":1,"ntasks":1,"cpus_per_task":32,"memory":"256G","gpu_resource_requested":"gpu:mi210:8","walltime":"08:00:00","world_size":8,"dp_size":2,"sp_size":4},"submission_boundary":{"environment_replaced_before_supervisor":True,"exact_supervisor_interface_names":[*EXPORT_NAMES,"GRAFT_SHORT_SUBMIT_WRAPPER_SHA256"],"exact_job_export_names":list(EXPORT_NAMES),"contains_export_all":False,"sbatch_executed_from_retained_fd":execute_sbatch_from_fd=="true","launcher_submitted_from_retained_fd":True,"launcher_sha256":launcher_sha,"plan_sha256":plan_sha,"bootstrap_trust_boundary":{"submit_wrapper_sha256_external_trust_anchor":wrapper_sha_external,"retained_submit_wrapper_bytes_matched_external_anchor":True,"submit_wrapper_pre_exec_formal_security_proven":False,"python_sha256_external_trust_anchor":python_sha,"running_python_bytes_matched_external_anchor":True,"python_pre_exec_formal_security_proven":False,"same_process_bootstrap_formal_security_proven":False},"terminal_admission":{"sha256_external_trust_anchor":terminal_sha,"materializer_runtime_sha256_external_trust_anchor":materializer_runtime_sha,"job_id":"132549","completed_0_0_observed_before_sbatch":True,"trust_anchor_computed_from_receipt":False}},"outputs":{"logical_output_root":output_root,"submission_receipt_path":str(parent/(output.name+".submission.receipt.json")),"create_only":True,"provisional_mode":"0600","final_success_mode":"0444","mode_0444_is_final_success_transition":True,"mode_0444_required_to_interpret_receipt_as_success":True},"authority":{"action_authority":False,"identity_authority":False,"quality_authority":False,"training_authority":False,"checkpoint_authority":False,"publication_authority":False,"production_authority":False,"scientific_success_claimed":False,"job_success_claimed":False},"failure_semantics":{"submission_success_is_not_job_success":True,"successful_scheduler_submission_may_exist_if_receipt_publication_fails":True,"success_receipt_may_survive_pre_finalization_failure":False,"failure_absence_of_any_receipt_inode_guaranteed":False,"provisional_non_success_inode_may_survive_cleanup_failure":True,"automatic_job_cancellation_on_receipt_failure":False,"bootstrap_trust_is_external_and_not_formally_proven_pre_exec":True}}
    receipt={**core,"receipt_digest":digest(canonical(core))}; create_receipt(parent_fd,output.name+".submission.receipt.json",canonical(receipt)+b"\n")
finally:
    for fd in reversed(retained):
        try: os.close(fd)
        except OSError: pass
    os.close(parent_fd)
PY

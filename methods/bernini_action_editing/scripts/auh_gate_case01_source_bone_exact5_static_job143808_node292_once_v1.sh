#!/bin/bash -p
# One-shot exact8 CPU-only static admission for the case01 exact5 package.

set -Eeuo pipefail
umask 077
[[ "$0" == /bin/bash && "$#" -eq 0 && "$-" == *p* && "$-" != *i* ]] || exit 96
[[ "${PATH:-}" == /usr/bin:/bin && "${LC_ALL:-}" == C && "${LANG:-}" == C \
  && "${HOME:-}" == /vast/users/guangyi.chen && "${BASH_ENV:-}" == /dev/null ]] || exit 96
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" \
  && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" \
  && -z "${HIP_VISIBLE_DEVICES:-}" && -z "${GPU_DEVICE_ORDINAL:-}" ]] || exit 96
if builtin declare -F | /usr/bin/grep . >/dev/null; then exit 96; fi
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi

readonly ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_r64_canary_v1
readonly PACKAGE_RECEIPT="$ROOT/authority/package_materialization_receipt_v1.json"
readonly PAYLOAD="$ROOT/diagnostics/exact5_static_probe_payload_v1.sh"
readonly RECEIPT="$ROOT/evidence/exact5_static_probe_receipt_v1.json"
readonly ATTEMPT="$ROOT/evidence/exact5_static_probe_attempt_v1.json"
readonly EVIDENCE="$ROOT/evidence/exact5_static_probe_controller_evidence_v1.json"
readonly STDOUT_LOG="$ROOT/logs/exact5_static_probe.stdout.log"
readonly STDERR_LOG="$ROOT/logs/exact5_static_probe.stderr.log"
readonly EVIDENCE_DIR="$ROOT/evidence"
readonly LOGS_DIR="$ROOT/logs"
readonly CACHE=/tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache
readonly ROOT_PYTHON=/usr/bin/python3.10

[[ -f "$ROOT_PYTHON" && -x "$ROOT_PYTHON" && ! -L "$ROOT_PYTHON" \
  && -f "$PACKAGE_RECEIPT" && ! -L "$PACKAGE_RECEIPT" \
  && -f "$PAYLOAD" && ! -L "$PAYLOAD" ]] || exit 96
for fresh in "$RECEIPT" "$ATTEMPT" "$EVIDENCE" "$STDOUT_LOG" "$STDERR_LOG" "$CACHE"; do
  [[ ! -e "$fresh" && ! -L "$fresh" ]] || exit 96
done
exec {ROOT_PYTHON_FD}<"$ROOT_PYTHON"
exec {PAYLOAD_FD}<"$PAYLOAD"
exec {EVIDENCE_DIR_FD}<"$EVIDENCE_DIR"
exec {LOGS_DIR_FD}<"$LOGS_DIR"

"/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B - \
  "$ROOT_PYTHON_FD" "$PAYLOAD_FD" "$EVIDENCE_DIR_FD" "$LOGS_DIR_FD" \
  "$ROOT" "$PACKAGE_RECEIPT" "$PAYLOAD" \
  "$RECEIPT" "$ATTEMPT" "$EVIDENCE" "$STDOUT_LOG" "$STDERR_LOG" "$CACHE" <<'PY'
import hashlib,json,os,stat,sys
def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def ident(v): return {"device":v.st_dev,"inode":v.st_ino,"uid":v.st_uid,"gid":v.st_gid,"mode":v.st_mode,"nlink":v.st_nlink,"rdev":v.st_rdev,"size":v.st_size,"blocks":getattr(v,"st_blocks",0),"mtime_ns":v.st_mtime_ns,"ctime_ns":v.st_ctime_ns}
def pread(fd,size):
 out=[]; off=0
 while off<size:
  block=os.pread(fd,min(1048576,size-off),off)
  if not block: break
  out.append(block);off+=len(block)
 raw=b"".join(out)
 if len(raw)!=size: raise RuntimeError("short read")
 return raw
def held(fd,path,digest_value,size,mode,uid,gid,process=False):
 before=os.fstat(fd);raw=pread(fd,before.st_size);after=os.fstat(fd);named=os.lstat(path)
 if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=mode or before.st_uid!=uid or before.st_gid!=gid or before.st_nlink!=1 or before.st_size!=size or ident(before)!=ident(after) or ident(before)!=ident(named) or hashlib.sha256(raw).hexdigest()!=digest_value or (process and ident(before)!=ident(os.stat("/proc/self/exe"))): raise RuntimeError("held authority differs")
 return raw
def stable(path,mode):
 fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0))
 try:
  before=os.fstat(fd);raw=pread(fd,before.st_size);after=os.fstat(fd);named=os.lstat(path)
 finally: os.close(fd)
 if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=mode or before.st_uid!=2012 or before.st_gid!=2000 or before.st_nlink!=1 or ident(before)!=ident(after) or ident(before)!=ident(named): raise RuntimeError("stable authority differs: "+path)
 return raw,ident(before)
def strict(raw):
 value=json.loads(raw)
 if type(value) is not dict or raw!=canonical(value)+b"\n": raise RuntimeError("canonical JSON differs")
 return value
def directory(fd,path,mode):
 before=os.fstat(fd);named=os.lstat(path)
 if os.path.realpath(path)!=path or not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode)!=mode or before.st_uid!=2012 or before.st_gid!=2000 or ident(before)!=ident(named): raise RuntimeError("directory authority differs: "+path)
def named_directory(path,mode):
 fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0))
 try:
  directory(fd,path,mode)
 finally: os.close(fd)
def create(parentfd,parent,path,value):
 if os.path.dirname(path)!=parent or os.path.basename(path) in ("",".",".."): raise RuntimeError("attempt target differs")
 raw=canonical(value)+b"\n";fd=os.open(os.path.basename(path),os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0),0,dir_fd=parentfd)
 try:
  offset=0
  while offset<len(raw):
   count=os.write(fd,raw[offset:])
   if count<=0: raise RuntimeError("attempt write made no progress")
   offset+=count
  os.fsync(fd);before=os.fstat(fd);named=os.stat(os.path.basename(path),dir_fd=parentfd,follow_symlinks=False)
  if stat.S_IMODE(before.st_mode)!=0 or before.st_uid!=2012 or before.st_gid!=2000 or before.st_nlink!=1 or ident(before)!=ident(named) or os.pread(fd,len(raw),0)!=raw: raise RuntimeError("attempt staging differs")
  os.fchmod(fd,0o400);os.fsync(fd);after=os.fstat(fd);named_after=os.stat(os.path.basename(path),dir_fd=parentfd,follow_symlinks=False)
  if stat.S_IMODE(after.st_mode)!=0o400 or ident(after)!=ident(named_after) or os.pread(fd,len(raw),0)!=raw: raise RuntimeError("attempt commit differs")
 finally: os.close(fd)
if len(sys.argv)!=14: raise RuntimeError("static controller preflight argv differs")
pyfd,payloadfd,evidencefd,logsfd=map(int,sys.argv[1:5]);root,package_path,payload_path,receipt,attempt,evidence,stdout,stderr,cache=sys.argv[5:]
held(pyfd,"/usr/bin/python3.10","11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96",5937800,0o755,0,0,True)
directory(evidencefd,root+"/evidence",0o755);directory(logsfd,root+"/logs",0o755)
for path,mode in ((root,0o755),(root+"/authority",0o555),(root+"/diagnostics",0o755),(root+"/launch",0o555),(root+"/plan",0o555),(root+"/release",0o555),(root+"/outputs",0o755),(root+"/outputs/media",0o755),(root+"/final",0o755),(root+"/runtime",0o755)):
 named_directory(path,mode)
package_raw,_=stable(package_path,0o400);package=strict(package_raw);unsigned=dict(package);claimed=unsigned.pop("receipt_digest",None)
row=package.get("cpu_admission",{}).get("static_probe",{})
payload_raw=held(payloadfd,payload_path,row.get("payload_sha256"),os.fstat(payloadfd).st_size,0o444,2012,2000)
if package.get("schema_version")!="case01-source-bone-exact5-r64-materialization-v1" or package.get("status")!="MATERIALIZED_NOT_SUBMITTED" or package.get("root")!=root or claimed!=digest(unsigned) or row.get("payload")!=payload_path or row.get("receipt")!=receipt or row.get("executed") is not False or hashlib.sha256(payload_raw).hexdigest()!=row.get("payload_sha256"): raise RuntimeError("package/static payload binding differs")
for path in (receipt,attempt,evidence,stdout,stderr,cache):
 if os.path.lexists(path): raise RuntimeError("static fresh target differs: "+path)
for rel in ("outputs/media","final","runtime"):
 if os.listdir(root+"/"+rel): raise RuntimeError("static production result root is not fresh")
value={"schema_version":"case01-source-bone-exact5-static-attempt-v1","status":"ATTEMPT_CLAIMED_BEFORE_SRUN","holder_job_id":"143808","node":"auh7-1b-gpu-292","package_receipt_sha256":hashlib.sha256(package_raw).hexdigest(),"package_receipt_digest":package["receipt_digest"],"payload_path":payload_path,"payload_sha256":hashlib.sha256(payload_raw).hexdigest(),"receipt_path":receipt,"single_srun_attempt":True,"retry_allowed":False,"renderer_executed":False}
value["attempt_digest"]=digest(value);create(evidencefd,root+"/evidence",attempt,value)
print("CASE01_EXACT5_STATIC_ATTEMPT_CLAIMED "+value["attempt_digest"],flush=True)
PY

set -o noclobber
exec {STDOUT_FD}>"/proc/self/fd/$LOGS_DIR_FD/exact5_static_probe.stdout.log"
exec {STDERR_FD}>"/proc/self/fd/$LOGS_DIR_FD/exact5_static_probe.stderr.log"
set +o noclobber
set +e
/usr/bin/srun --jobid=143808 --job-name=case01-exact5-static-v1 \
  --exclusive --exact --immediate=10 --kill-on-bad-exit=1 \
  --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-292 \
  --cpus-per-task=8 --mem=8G --gpus-per-node=8 --export=NONE --time=00:10:00 \
  /bin/bash -p -s <&"$PAYLOAD_FD" >&"$STDOUT_FD" 2>&"$STDERR_FD"
readonly SRUN_RC=$?
set -e
[[ "$SRUN_RC" -eq 0 ]] || exit "$SRUN_RC"

"/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B - \
  "$ROOT_PYTHON_FD" "$PAYLOAD_FD" "$EVIDENCE_DIR_FD" "$LOGS_DIR_FD" \
  "$STDOUT_FD" "$STDERR_FD" "$PACKAGE_RECEIPT" "$PAYLOAD" \
  "$RECEIPT" "$ATTEMPT" "$EVIDENCE" "$STDOUT_LOG" "$STDERR_LOG" "$CACHE" <<'PY'
import hashlib,json,os,stat,sys
def canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()
def ident(v): return {"device":v.st_dev,"inode":v.st_ino,"uid":v.st_uid,"gid":v.st_gid,"mode":v.st_mode,"nlink":v.st_nlink,"rdev":v.st_rdev,"size":v.st_size,"blocks":getattr(v,"st_blocks",0),"mtime_ns":v.st_mtime_ns,"ctime_ns":v.st_ctime_ns}
def stable(path,allow_empty=False):
 fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0))
 try: before=os.fstat(fd);raw=pread(fd,before.st_size);after=os.fstat(fd);named=os.lstat(path)
 finally: os.close(fd)
 if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=0o400 or before.st_uid!=2012 or before.st_gid!=2000 or before.st_nlink!=1 or ident(before)!=ident(after) or ident(before)!=ident(named) or (not allow_empty and not raw): raise RuntimeError("post stable authority differs")
 return raw,ident(before)
def pread(fd,size):
 out=[];off=0
 while off<size:
  block=os.pread(fd,min(1048576,size-off),off)
  if not block: break
  out.append(block);off+=len(block)
 raw=b"".join(out)
 if len(raw)!=size: raise RuntimeError("post held short read")
 return raw
def held(fd,path,digest_value,mode,uid,gid,process=False):
 before=os.fstat(fd);raw=pread(fd,before.st_size);after=os.fstat(fd);named=os.lstat(path)
 if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=mode or before.st_uid!=uid or before.st_gid!=gid or before.st_nlink!=1 or ident(before)!=ident(after) or ident(before)!=ident(named) or hashlib.sha256(raw).hexdigest()!=digest_value or (process and ident(before)!=ident(os.stat("/proc/self/exe"))): raise RuntimeError("post held authority differs")
 return raw,ident(before)
def directory(fd,path,mode):
 before=os.fstat(fd);named=os.lstat(path)
 if os.path.realpath(path)!=path or not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode)!=mode or before.st_uid!=2012 or before.st_gid!=2000 or ident(before)!=ident(named): raise RuntimeError("post directory authority differs: "+path)
def named_directory(path,mode):
 fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0))
 try: directory(fd,path,mode)
 finally: os.close(fd)
def stable_at(parentfd,parent,path,allow_empty=False):
 if os.path.dirname(path)!=parent or os.path.basename(path) in ("",".",".."): raise RuntimeError("post child target differs")
 fd=os.open(os.path.basename(path),os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0),dir_fd=parentfd)
 try:
  before=os.fstat(fd);raw=pread(fd,before.st_size);after=os.fstat(fd);named=os.stat(os.path.basename(path),dir_fd=parentfd,follow_symlinks=False)
 finally: os.close(fd)
 if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=0o400 or before.st_uid!=2012 or before.st_gid!=2000 or before.st_nlink!=1 or ident(before)!=ident(after) or ident(before)!=ident(named) or (not allow_empty and not raw): raise RuntimeError("post child authority differs")
 return raw,ident(before)
def commit_log(fd,parentfd,parent,path,allow_empty=False):
 if os.path.dirname(path)!=parent or os.path.basename(path) in ("",".",".."): raise RuntimeError("log target differs")
 os.fsync(fd);before=os.fstat(fd);named=os.stat(os.path.basename(path),dir_fd=parentfd,follow_symlinks=False)
 if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=0o600 or before.st_uid!=2012 or before.st_gid!=2000 or before.st_nlink!=1 or ident(before)!=ident(named): raise RuntimeError("held log staging differs")
 os.fchmod(fd,0o400);os.fsync(fd);after=os.fstat(fd);named_after=os.stat(os.path.basename(path),dir_fd=parentfd,follow_symlinks=False)
 if stat.S_IMODE(after.st_mode)!=0o400 or ident(after)!=ident(named_after): raise RuntimeError("held log commit differs")
 reader=os.open(os.path.basename(path),os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0),dir_fd=parentfd)
 try:
  if ident(os.fstat(reader))!=ident(after): raise RuntimeError("held log reader differs")
  raw=pread(reader,after.st_size)
  reader_after=os.fstat(reader)
 finally: os.close(reader)
 held_after=os.fstat(fd);named_final=os.stat(os.path.basename(path),dir_fd=parentfd,follow_symlinks=False)
 if ident(reader_after)!=ident(after) or ident(held_after)!=ident(after) or ident(named_final)!=ident(after): raise RuntimeError("held log final replay differs")
 if not allow_empty and not raw: raise RuntimeError("held log is empty")
 return raw,ident(after)
def strict(raw):
 value=json.loads(raw)
 if type(value) is not dict or raw!=canonical(value)+b"\n": raise RuntimeError("post canonical JSON differs")
 return value
def create(parentfd,parent,path,value):
 if os.path.dirname(path)!=parent or os.path.basename(path) in ("",".",".."): raise RuntimeError("evidence target differs")
 raw=canonical(value)+b"\n";fd=os.open(os.path.basename(path),os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0),0,dir_fd=parentfd)
 try:
  offset=0
  while offset<len(raw):
   count=os.write(fd,raw[offset:])
   if count<=0: raise RuntimeError("post write made no progress")
   offset+=count
  os.fsync(fd);before=os.fstat(fd);named=os.stat(os.path.basename(path),dir_fd=parentfd,follow_symlinks=False)
  if stat.S_IMODE(before.st_mode)!=0 or before.st_uid!=2012 or before.st_gid!=2000 or before.st_nlink!=1 or ident(before)!=ident(named) or os.pread(fd,len(raw),0)!=raw: raise RuntimeError("evidence staging differs")
  os.fchmod(fd,0o400);os.fsync(fd);after=os.fstat(fd);named_after=os.stat(os.path.basename(path),dir_fd=parentfd,follow_symlinks=False)
  if stat.S_IMODE(after.st_mode)!=0o400 or ident(after)!=ident(named_after) or os.pread(fd,len(raw),0)!=raw: raise RuntimeError("evidence commit differs")
 finally: os.close(fd)
if len(sys.argv)!=15: raise RuntimeError("static postflight argv differs")
pyfd,payloadfd,evidencefd,logsfd,stdoutfd,stderrfd=map(int,sys.argv[1:7]);package_path,payload_path,receipt_path,attempt_path,evidence_path,stdout_path,stderr_path,cache=sys.argv[7:]
held(pyfd,"/usr/bin/python3.10","11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96",0o755,0,0,True)
root=os.path.dirname(os.path.dirname(package_path));directory(evidencefd,root+"/evidence",0o755);directory(logsfd,root+"/logs",0o755)
for path,mode in ((root,0o755),(root+"/authority",0o555),(root+"/diagnostics",0o755),(root+"/launch",0o555),(root+"/plan",0o555),(root+"/release",0o555),(root+"/outputs",0o755),(root+"/outputs/media",0o755),(root+"/final",0o755),(root+"/runtime",0o755)):
 named_directory(path,mode)
for path in (root+"/outputs/media",root+"/final",root+"/runtime"):
 if os.listdir(path): raise RuntimeError("post production result root is not fresh")
package_raw,package_identity=stable(package_path);package=strict(package_raw);package_unsigned=dict(package);package_claimed=package_unsigned.pop("receipt_digest",None)
if package_claimed!=digest(package_unsigned): raise RuntimeError("package receipt digest differs postflight")
static_row=package.get("cpu_admission",{}).get("static_probe",{})
payload_raw,payload_identity=held(payloadfd,payload_path,static_row.get("payload_sha256"),0o444,2012,2000)
receipt_raw,receipt_identity=stable_at(evidencefd,os.path.dirname(receipt_path),receipt_path);receipt=strict(receipt_raw);unsigned=dict(receipt);claimed=unsigned.pop("receipt_digest",None)
attempt_raw,attempt_identity=stable_at(evidencefd,os.path.dirname(attempt_path),attempt_path);attempt=strict(attempt_raw);attempt_unsigned=dict(attempt);attempt_claimed=attempt_unsigned.pop("attempt_digest",None)
stdout_raw,stdout_identity=commit_log(stdoutfd,logsfd,os.path.dirname(stdout_path),stdout_path);stderr_raw,stderr_identity=commit_log(stderrfd,logsfd,os.path.dirname(stderr_path),stderr_path,True)
step=receipt.get("slurm_step_id","")
attempt_fields={"schema_version","status","holder_job_id","node","package_receipt_sha256","package_receipt_digest","payload_path","payload_sha256","receipt_path","single_srun_attempt","retry_allowed","renderer_executed","attempt_digest"}
receipt_fields={"schema_version","status","campaign_mode","holder_job_id","expected_node","slurm_step_id","task_count","selected_task_ids","release_file_count","launch_identity_count","plan_sha256","plan_digest","independent_audit_sha256","independent_audit_digest","checkpoint_manifest_sha256","launch_receipt_sha256","launch_receipt_digest","payload_sha256","ffprobe_path","ffprobe_sha256","rank_cache_root","production_outputs_fresh","rank_cache_fresh","pure_metadata_only","torch_imported","renderer_executed","receipt_digest"}
expected_tasks=["case01-exact_original-full644","case01-codec_only_present-full644","case01-bone_removed-full644","case01-bone_translated_up150-full644","case01-sham_control_up150-full644"]
if set(attempt)!=attempt_fields or attempt_claimed!=digest(attempt_unsigned) or attempt.get("status")!="ATTEMPT_CLAIMED_BEFORE_SRUN" or attempt.get("holder_job_id")!="143808" or attempt.get("node")!="auh7-1b-gpu-292" or attempt.get("package_receipt_sha256")!=hashlib.sha256(package_raw).hexdigest() or attempt.get("package_receipt_digest")!=package["receipt_digest"] or attempt.get("payload_path")!=payload_path or attempt.get("payload_sha256")!=hashlib.sha256(payload_raw).hexdigest() or attempt.get("receipt_path")!=receipt_path or attempt.get("single_srun_attempt") is not True or attempt.get("retry_allowed") is not False or attempt.get("renderer_executed") is not False: raise RuntimeError("static attempt replay differs")
if set(receipt)!=receipt_fields or receipt.get("schema_version")!="case01-source-bone-exact5-static-probe-v1" or receipt.get("status")!="PASS" or claimed!=digest(unsigned) or receipt.get("campaign_mode")!="case01-source-bone-exact5-r64-canary" or receipt.get("holder_job_id")!="143808" or receipt.get("expected_node")!="auh7-1b-gpu-292" or not step.isdecimal() or int(step)<=394 or receipt.get("task_count")!=5 or receipt.get("selected_task_ids")!=expected_tasks or receipt.get("release_file_count")!=19 or receipt.get("launch_identity_count")!=18 or receipt.get("plan_sha256")!=package["plan"]["sha256"] or receipt.get("plan_digest")!=package["plan"]["plan_digest"] or receipt.get("independent_audit_sha256")!=package["independent_audit"]["sha256"] or receipt.get("independent_audit_digest")!=package["independent_audit"]["audit_digest"] or receipt.get("checkpoint_manifest_sha256")!="7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2" or receipt.get("launch_receipt_sha256")!=package["launch"]["receipt_sha256"] or receipt.get("launch_receipt_digest")!=package["launch"]["receipt_digest"] or receipt.get("payload_sha256")!=package["launch"]["payload_sha256"] or receipt.get("ffprobe_path")!="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe" or receipt.get("ffprobe_sha256")!="356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5" or receipt.get("rank_cache_root")!=cache or receipt.get("production_outputs_fresh") is not True or receipt.get("rank_cache_fresh") is not True or receipt.get("pure_metadata_only") is not True or receipt.get("torch_imported") is not False or receipt.get("renderer_executed") is not False or os.path.lexists(cache) or stderr_raw!=b"" or stdout_raw!=b"CASE01_EXACT5_STATIC_PASS "+receipt["receipt_digest"].encode()+b"\n": raise RuntimeError("static terminal receipt/evidence differs")
value={"schema_version":"case01-source-bone-exact5-static-controller-evidence-v1","status":"PASS","holder_job_id":"143808","node":"auh7-1b-gpu-292","numeric_step":"143808."+step,"single_srun_attempt":True,"srun_returncode":0,"package_replay":{"path":package_path,"sha256":hashlib.sha256(package_raw).hexdigest(),"receipt_digest":package["receipt_digest"],"identity":package_identity},"payload_replay":{"path":payload_path,"sha256":hashlib.sha256(payload_raw).hexdigest(),"identity":payload_identity},"receipt_replay":{"path":receipt_path,"sha256":hashlib.sha256(receipt_raw).hexdigest(),"receipt_digest":receipt["receipt_digest"],"identity":receipt_identity,"canonical_json_plus_lf":True},"attempt_replay":{"path":attempt_path,"sha256":hashlib.sha256(attempt_raw).hexdigest(),"attempt_digest":attempt["attempt_digest"],"identity":attempt_identity},"stdout":{"path":stdout_path,"sha256":hashlib.sha256(stdout_raw).hexdigest(),"identity":stdout_identity},"stderr":{"path":stderr_path,"sha256":hashlib.sha256(stderr_raw).hexdigest(),"identity":stderr_identity,"empty":True},"retry_allowed":False,"renderer_executed":False}
value["evidence_digest"]=digest(value);create(evidencefd,os.path.dirname(evidence_path),evidence_path,value);print("CASE01_EXACT5_STATIC_CONTROLLER_PASS "+value["evidence_digest"])
PY

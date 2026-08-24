#!/bin/bash -p
set -euo pipefail
umask 077
[[ "$-" == *p* ]] || exit 90
[[ "$0" == "bash" || "$0" == "/bin/bash" || "$0" == "-bash" ]] || exit 91
[[ -z "${BASH_ENV-}" && -z "${ENV-}" ]] || exit 92
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi
readonly ROOT_PYTHON=/usr/bin/python3.10
exec {ROOT_PYTHON_FD}<"$ROOT_PYTHON"
[[ "$ROOT_PYTHON_FD" =~ ^[0-9]+$ ]] || exit 93
exec -c "/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B - "$ROOT_PYTHON_FD" <<'PY'
import hashlib,json,os,re,stat,subprocess,sys,time
ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_case00_847b91a2_c91de7eb_d70eac5c_r1"
JOB="143812"; STEP="167"; FULL="143812.167"; NODE="auh7-1b-gpu-293"; JOB_NAME="f644-r5f-static"
ROOT_PYTHON="/usr/bin/python3.10"; ROOT_PYTHON_SHA="11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"; ROOT_PYTHON_SIZE=5937800
SACCT="/usr/bin/sacct"; SACCT_SHA="fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"; SACCT_SIZE=85952
PAYLOAD=ROOT+"/diagnostics/static_nomodel_probe_payload_r5d.sh"; PAYLOAD_SHA="80dd0703a651f6e1634afbeb70007ffa304b93102556ba2d33b232e001af27ff"; PAYLOAD_SIZE=6263
RECEIPT=ROOT+"/evidence/static_nomodel_probe_receipt_r5d.json"; RECEIPT_SHA="28fb5c311a45b88106d40698c65d8edb88c5e9bdfb5dea8613aced76cb8ba7ee"; RECEIPT_SIZE=2285; RECEIPT_DIGEST="bf460bbfc5d40760a43fb212e807aca97b539e9ed7e8a3cb5108cb61b40f8f92"
PARENT=ROOT+"/evidence"; EVIDENCE="r5f_static_nomodel_probe.sacct-and-replay.json"
def canon(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v): return hashlib.sha256(canon(v)).hexdigest()
def ident(v): return {"device":v.st_dev,"inode":v.st_ino,"uid":v.st_uid,"gid":v.st_gid,"mode":v.st_mode,"nlink":v.st_nlink,"rdev":v.st_rdev,"size":v.st_size,"blocks":getattr(v,"st_blocks",0),"mtime_ns":v.st_mtime_ns,"ctime_ns":v.st_ctime_ns}
def oid(v): return (v.st_dev,v.st_ino,v.st_uid,v.st_gid,v.st_mode,v.st_nlink,v.st_rdev)
def pread(fd,size):
 out=[]; off=0
 while off<size:
  b=os.pread(fd,min(1048576,size-off),off)
  if not b: break
  out.append(b); off+=len(b)
 raw=b"".join(out)
 if len(raw)!=size: raise RuntimeError("short pread")
 return raw
def pin(path,sha,size,uid,gid,mode):
 fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0)); before=os.fstat(fd); raw=pread(fd,before.st_size); after=os.fstat(fd); named=os.lstat(path)
 if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=mode or before.st_uid!=uid or before.st_gid!=gid or before.st_nlink!=1 or before.st_size!=size or ident(before)!=ident(after) or ident(before)!=ident(named) or hashlib.sha256(raw).hexdigest()!=sha or os.lseek(fd,0,os.SEEK_CUR)!=0: raise RuntimeError("pin differs: "+path)
 return fd,before,raw
def replay(fd,path,before,sha):
 now=os.fstat(fd)
 if ident(now)!=ident(before) or ident(now)!=ident(os.lstat(path)) or hashlib.sha256(pread(fd,now.st_size)).hexdigest()!=sha or os.lseek(fd,0,os.SEEK_CUR)!=0: raise RuntimeError("post replay differs")
def strict(raw):
 def pairs(items):
  out={}
  for k,v in items:
   if k in out: raise ValueError("duplicate")
   out[k]=v
  return out
 value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in()).throw(ValueError(x)))
 if type(value) is not dict or raw!=canon(value)+b"\n": raise RuntimeError("canonical JSON differs")
 return value
def memory(raw):
 m=re.fullmatch(r"([1-9][0-9]*)([KMGT])",raw)
 if m is None: raise RuntimeError("memory differs")
 return int(m.group(1))*{"K":1024,"M":1024**2,"G":1024**3,"T":1024**4}[m.group(2)]
def tres(raw):
 out={}
 for token in raw.split(","):
  if token.count("=")!=1: raise RuntimeError("TRES differs")
  k,v=token.split("=",1)
  if not k or not v or k in out: raise RuntimeError("TRES differs")
  out[k]=v
 if not set(out).issubset({"cpu","mem","node","billing","gres/gpu","gres/gpu:mi210"}) or out.get("cpu")!="4" or out.get("node")!="1" or memory(out.get("mem",""))!=8*1024**3 or out.get("gres/gpu")!="8" or out.get("gres/gpu:mi210")!="8" or ("billing" in out and out["billing"]!="4"): raise RuntimeError("allocated TRES differs")
 return out
if len(sys.argv)!=2 or not sys.argv[1].isascii() or not sys.argv[1].isdecimal(): raise RuntimeError("argv differs")
rootfd=int(sys.argv[1])
if set(os.environ) not in (set(),{"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"]!="C.UTF-8"): raise RuntimeError("environment differs")
os.environ.clear(); rb=os.fstat(rootfd); rr=pread(rootfd,rb.st_size)
if rootfd<3 or not os.get_inheritable(rootfd) or os.lseek(rootfd,0,os.SEEK_CUR)!=0 or not stat.S_ISREG(rb.st_mode) or stat.S_IMODE(rb.st_mode)!=0o755 or rb.st_uid!=0 or rb.st_gid!=0 or rb.st_nlink!=1 or rb.st_size!=ROOT_PYTHON_SIZE or ident(rb)!=ident(os.lstat(ROOT_PYTHON)) or ident(rb)!=ident(os.stat("/proc/self/exe")) or hashlib.sha256(rr).hexdigest()!=ROOT_PYTHON_SHA: raise RuntimeError("root Python differs")
os.set_inheritable(rootfd,False)
pfd,pb,_=pin(PAYLOAD,PAYLOAD_SHA,PAYLOAD_SIZE,2012,2000,0o444); rfd,receipt_before,receipt_raw=pin(RECEIPT,RECEIPT_SHA,RECEIPT_SIZE,2012,2000,0o400); sfd,sb,_=pin(SACCT,SACCT_SHA,SACCT_SIZE,0,0,0o755)
rv=strict(receipt_raw); unsigned=dict(rv); claimed=unsigned.pop("receipt_digest",None)
if claimed!=RECEIPT_DIGEST or digest(unsigned)!=claimed or rv.get("schema_version")!="full644-exploratory-matched-r5f-static-nomodel-probe-v1" or rv.get("status")!="PASS" or rv.get("holder_job_id")!=JOB or rv.get("expected_node")!=NODE or rv.get("slurm_step_id")!=STEP or rv.get("torch_imported") is not False or rv.get("gpu_device_fd_observed_at_probe_end") is not False: raise RuntimeError("receipt semantics differ")
parent=os.open(PARENT,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)); parent_before=os.fstat(parent)
if not stat.S_ISDIR(parent_before.st_mode) or stat.S_IMODE(parent_before.st_mode)!=0o755 or parent_before.st_uid!=2012 or parent_before.st_gid!=2000 or oid(parent_before)!=oid(os.lstat(PARENT)): raise RuntimeError("parent differs")
origin=os.stat(EVIDENCE,dir_fd=parent,follow_symlinks=False)
if not stat.S_ISREG(origin.st_mode) or stat.S_IMODE(origin.st_mode)!=0 or origin.st_uid!=2012 or origin.st_gid!=2000 or origin.st_nlink!=1 or origin.st_size!=0: raise RuntimeError("prior evidence tombstone differs")
os.chmod(EVIDENCE,0o600,dir_fd=parent,follow_symlinks=False)
efd=os.open(EVIDENCE,os.O_RDWR|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0),dir_fd=parent); eb=os.fstat(efd); en=os.stat(EVIDENCE,dir_fd=parent,follow_symlinks=False)
if not stat.S_ISREG(eb.st_mode) or stat.S_IMODE(eb.st_mode)!=0o600 or eb.st_uid!=2012 or eb.st_gid!=2000 or eb.st_nlink!=1 or eb.st_size!=0 or (eb.st_dev,eb.st_ino,eb.st_uid,eb.st_gid,eb.st_nlink,eb.st_rdev,eb.st_size)!=(origin.st_dev,origin.st_ino,origin.st_uid,origin.st_gid,origin.st_nlink,origin.st_rdev,origin.st_size) or ident(eb)!=ident(en): raise RuntimeError("prior tombstone reopen differs")
query=[SACCT,"--jobs="+FULL,"--noheader","--parsable2","--noconvert","--format=JobIDRaw,JobName%64,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES%256,NodeList%64,MaxRSS"]
env={"LANG":"C","LC_ALL":"C"}; row=None; out=b""; err=b""; attempts=0
for attempts in range(1,31):
 q=subprocess.run(query,executable="/proc/self/fd/"+str(sfd),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True,pass_fds=(sfd,),cwd="/",env=env,check=False)
 if q.returncode!=0 or q.stderr!=b"": raise RuntimeError("sacct failed")
 out,err=q.stdout,q.stderr; rows=[]
 for line in out.decode("utf-8","strict").splitlines():
  c=line.split("|")
  if c and c[0]==FULL:
   if len(c)!=9: raise RuntimeError("sacct closure differs")
   rows.append(c)
 if len(rows)>1: raise RuntimeError("duplicate sacct row")
 if len(rows)==1 and rows[0][2] not in {"PENDING","RUNNING","COMPLETING","CONFIGURING","SUSPENDED","RESIZING"}: row=rows[0]; break
 time.sleep(1)
if row is None: raise RuntimeError("terminal sacct unavailable")
step,name,state,exitcode,elapsed,cpus,tres_raw,node,maxrss=row; tres_value=tres(tres_raw)
if step!=FULL or name!=JOB_NAME or state!="COMPLETED" or exitcode!="0:0" or elapsed!="1" or cpus!="4" or node!=NODE or maxrss!="1536000": raise RuntimeError("sacct semantics differ")
replay(pfd,PAYLOAD,pb,PAYLOAD_SHA); replay(rfd,RECEIPT,receipt_before,RECEIPT_SHA); replay(sfd,SACCT,sb,SACCT_SHA)
ra=os.fstat(rootfd)
if ident(ra)!=ident(rb) or ident(ra)!=ident(os.lstat(ROOT_PYTHON)) or ident(ra)!=ident(os.stat("/proc/self/exe")) or hashlib.sha256(pread(rootfd,ra.st_size)).hexdigest()!=ROOT_PYTHON_SHA or os.lseek(rootfd,0,os.SEEK_CUR)!=0: raise RuntimeError("root Python post differs")
if oid(os.fstat(parent))!=oid(parent_before) or oid(os.fstat(parent))!=oid(os.lstat(PARENT)): raise RuntimeError("parent post differs")
e={"schema_version":"full644-r5f-static-direct-step-replay-only-evidence-v2","status":"PASS","campaign_mode":"case00-pair-canary","holder_job_id":JOB,"node":NODE,"numeric_step":FULL,"replay_only":True,"srun_called_by_replay":False,"subprocess_calls_by_replay":["sacct"],"prior_tombstone_recovery":{"authenticated_mode":0,"temporary_write_mode":0o600,"same_inode_reopened":True,"final_commit_mode":0o400},"original_execution":{"entry":"direct trusted stdin","logs_unretained":True,"stdout_stderr_logs_retained":False,"receipt_and_sacct_are_replayed":True},"payload":{"path":PAYLOAD,"sha256":PAYLOAD_SHA,"size":PAYLOAD_SIZE,"retained_fd_replayed":True},"receipt":{"path":RECEIPT,"sha256":RECEIPT_SHA,"size":RECEIPT_SIZE,"receipt_digest":RECEIPT_DIGEST,"identity":ident(receipt_before),"canonical_json_plus_lf":True,"semantic_contract_replayed":True},"sacct_executable":{"path":SACCT,"sha256":SACCT_SHA,"executed_via_retained_fd":True},"root_python":{"path":ROOT_PYTHON,"sha256":ROOT_PYTHON_SHA,"executed_via_retained_fd":True},"sacct":{"query_argv":query,"query_attempts":attempts,"raw_stdout_sha256":hashlib.sha256(out).hexdigest(),"raw_stderr_sha256":hashlib.sha256(err).hexdigest(),"row":{"JobIDRaw":step,"JobName":name,"State":state,"ExitCode":exitcode,"ElapsedRaw":elapsed,"AllocCPUS":cpus,"AllocTRES":tres_value,"NodeList":node,"MaxRSS":maxrss}},"formal_report_generated":False,"html_generated":False,"external_trust_boundary":"root-owned default Slurm config, dynamic loader, shared libraries, plugins, and kernel"}
e["evidence_digest"]=digest(e); raw=canon(e)+b"\n"; current=os.fstat(efd); named=os.stat(EVIDENCE,dir_fd=parent,follow_symlinks=False)
if ident(current)!=ident(eb) or ident(current)!=ident(named): raise RuntimeError("tombstone differs")
off=0
while off<len(raw):
 n=os.write(efd,raw[off:])
 if n<=0: raise RuntimeError("write differs")
 off+=n
os.fsync(efd); staged=os.fstat(efd)
if pread(efd,len(raw))!=raw or ident(staged)!=ident(os.stat(EVIDENCE,dir_fd=parent,follow_symlinks=False)): raise RuntimeError("staged replay differs")
os.fchmod(efd,0o400); line=("R5F_STATIC_REPLAY_ONLY_PASS step="+FULL+" receipt_digest="+RECEIPT_DIGEST+" evidence_digest="+e["evidence_digest"]+"\n").encode("ascii")
try: os.write(1,line)
except OSError: pass
os._exit(0)
PY

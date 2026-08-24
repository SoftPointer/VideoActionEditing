#!/usr/bin/env python3
"""Externally admit a terminal compute-Bash retained-FD probe."""
from __future__ import annotations
import argparse,hashlib,json,os,re,stat,subprocess
from pathlib import Path

ROOT=Path("/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809")
WRAPPER_SHA="8283e73ddf240d1ed8946f5682910bcfafaf24a88ae6c175c80b6a4597a75016"
PYTHON=Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"); PYTHON_SHA="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
SACCT=Path("/usr/bin/sacct"); SACCT_SHA="fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
STEM="compute-bash-retained-fd-probe-8283e73d-r1"; RELEASE=ROOT/"releases"/STEM; WRAPPER=RELEASE/"auh_probe_compute_bash_retained_fd_v1.sbatch"; POSTFLIGHT=RELEASE/"postflight_compute_bash_retained_fd_probe_v1.py"; MANIFEST=RELEASE/"release-manifest.json"; OUTPUT=ROOT/"canaries"/STEM; SUBMISSION=OUTPUT/"submission-receipt.json"; ADMISSION=OUTPUT/"probe-admission.json"; LOGS=ROOT/"slurm"/STEM
AUTHORITY={"scientific":False,"generation":False,"training":False,"publication":False,"formal_job_authorized":False}
def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def die(x): raise SystemExit("postflight-compute-bash-fd-probe-v1: "+x)
def read_sealed(path,schema,fields):
 s=path.lstat(); before=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 if not stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode) or s.st_nlink!=1 or stat.S_IMODE(s.st_mode)!=0o444: die("receipt identity differs")
 raw=path.read_bytes(); after=path.lstat()
 if before!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns): die("receipt changed during read")
 v=json.loads(raw.decode("ascii")); u=dict(v); claimed=u.pop("receipt_digest",None)
 if set(v)!=fields or v.get("schema_version")!=schema or raw!=canonical(v)+b"\n" or claimed!=hashlib.sha256(canonical(u)).hexdigest(): die("receipt seal differs")
 return v,raw,s
def write_create(path,value):
 raw=canonical(value)+b"\n"; fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); view=memoryview(raw)
 while view:
  n=os.write(fd,view)
  if n<=0: die("write stalled")
  view=view[n:]
 os.fsync(fd); os.fchmod(fd,0o444); os.fsync(fd); os.lseek(fd,0,os.SEEK_SET); reread=os.read(fd,len(raw)+1); sealed=os.fstat(fd); public=path.lstat()
 if reread!=raw or not stat.S_ISREG(sealed.st_mode) or sealed.st_nlink!=1 or stat.S_IMODE(sealed.st_mode)!=0o444 or sealed.st_size!=len(raw) or (sealed.st_dev,sealed.st_ino)!=(public.st_dev,public.st_ino): die("published admission differs")
 os.close(fd); d=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); os.fsync(d); os.close(d)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--job-id",required=True); p.add_argument("--output-parent",required=True); p.add_argument("--submission-receipt",required=True); p.add_argument("--admission",required=True); p.add_argument("--wrapper",required=True); a=p.parse_args(); job=a.job_id
 if not re.fullmatch(r"[0-9]+",job) or Path(a.output_parent)!=OUTPUT or Path(a.submission_receipt)!=SUBMISSION or Path(a.admission)!=ADMISSION or Path(a.wrapper)!=WRAPPER: die("CLI identity differs")
 if ADMISSION.exists() or ADMISSION.is_symlink() or sha(WRAPPER)!=WRAPPER_SHA or Path(__file__)!=POSTFLIGHT or POSTFLIGHT.resolve(strict=True)!=POSTFLIGHT: die("fresh admission/input differs")
 release_info=RELEASE.lstat(); output_info=OUTPUT.lstat(); logs_info=LOGS.lstat()
 if not stat.S_ISDIR(release_info.st_mode) or stat.S_ISLNK(release_info.st_mode) or release_info.st_uid!=os.getuid() or stat.S_IMODE(release_info.st_mode)!=0o555 or set(RELEASE.iterdir())!={WRAPPER,POSTFLIGHT,MANIFEST}: die("release closure differs")
 if not stat.S_ISDIR(output_info.st_mode) or stat.S_ISLNK(output_info.st_mode) or output_info.st_uid!=os.getuid() or stat.S_IMODE(output_info.st_mode)!=0o700 or not stat.S_ISDIR(logs_info.st_mode) or stat.S_ISLNK(logs_info.st_mode) or logs_info.st_uid!=os.getuid() or stat.S_IMODE(logs_info.st_mode)!=0o700: die("output/log root differs")
 for leaf in (WRAPPER,POSTFLIGHT,MANIFEST):
  ls=leaf.lstat()
  if not stat.S_ISREG(ls.st_mode) or stat.S_ISLNK(ls.st_mode) or ls.st_uid!=os.getuid() or ls.st_nlink!=1 or stat.S_IMODE(ls.st_mode)!=0o444: die("release leaf differs")
 failure=OUTPUT/f"job-{job}.failure.json"; root=OUTPUT/f"job-{job}"; fixture=OUTPUT/f"fd-fixture-job-{job}"; evidence=root/"operational-evidence.json"
 if failure.exists() or failure.is_symlink() or set(OUTPUT.iterdir())!={SUBMISSION,root,fixture}: die("output closure differs")
 for directory in (root,fixture):
  ds=directory.lstat()
  if not stat.S_ISDIR(ds.st_mode) or stat.S_ISLNK(ds.st_mode) or ds.st_uid!=os.getuid() or stat.S_IMODE(ds.st_mode)!=0o700: die("job directory differs")
 sf={"schema_version","status","submission_success","job_success","submitted_job","request","submission_boundary","inputs","outputs","authority","receipt_digest"}
 sub,sraw,sinfo=read_sealed(SUBMISSION,"saic-compute-bash-retained-fd-probe-submission-v1",sf)
 submitted=sub.get("submitted_job",{}); boundary=sub.get("submission_boundary",{}); postflight_sha=sha(POSTFLIGHT)
 manifest_raw=MANIFEST.read_bytes(); manifest=json.loads(manifest_raw.decode("ascii")); mu=dict(manifest); manifest_digest=mu.pop("receipt_digest",None); manifest_sha=hashlib.sha256(manifest_raw).hexdigest()
 if (set(manifest)!={"schema_version","status","stem","release_root","output_parent","wrapper","postflight","executables","authority","receipt_digest"} or manifest.get("schema_version")!="saic-compute-bash-retained-fd-probe-release-v1" or manifest.get("status")!="sealed_before_submission" or manifest.get("stem")!=STEM or manifest.get("release_root")!=str(RELEASE) or manifest.get("output_parent")!=str(OUTPUT) or manifest.get("wrapper")!={"path":str(WRAPPER),"sha256":WRAPPER_SHA} or manifest.get("postflight")!={"path":str(POSTFLIGHT),"sha256":postflight_sha} or manifest.get("executables")!={"python":str(PYTHON),"python_sha256":PYTHON_SHA,"sacct":str(SACCT),"sacct_sha256":SACCT_SHA} or manifest.get("authority")!=AUTHORITY or manifest_raw!=canonical(manifest)+b"\n" or manifest_digest!=hashlib.sha256(canonical(mu)).hexdigest()): die("release manifest differs")
 exports=["SAIC_BASH_FD_PROBE_WRAPPER","SAIC_BASH_FD_PROBE_WRAPPER_SHA256","SAIC_BASH_FD_PROBE_PYTHON","SAIC_BASH_FD_PROBE_PYTHON_SHA256","SAIC_BASH_FD_PROBE_OUTPUT_PARENT","SAIC_BASH_FD_PROBE_OUTPUT_DEVICE","SAIC_BASH_FD_PROBE_OUTPUT_INODE","SAIC_BASH_FD_PROBE_SUBMISSION","SAIC_BASH_FD_PROBE_SUBMISSION_DEVICE","SAIC_BASH_FD_PROBE_SUBMISSION_INODE","SAIC_BASH_FD_PROBE_POSTFLIGHT","SAIC_BASH_FD_PROBE_POSTFLIGHT_SHA256","SAIC_BASH_FD_PROBE_RELEASE_MANIFEST","SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_SHA256","SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_DIGEST"]
 expected_request={"job_name":"saic-bash-fd-probe1","partition":"faculty","qos":"bgqos","nodes":1,"ntasks":1,"cpus_per_task":4,"memory":"8G","walltime":"00:05:00","gpu_resource_requested":"gpu:mi210:1","hold":False,"dependency":None,"scientific_generation":False}
 oi=OUTPUT.lstat(); retained_wrapper_fd=boundary.get("retained_wrapper_fd") if isinstance(boundary,dict) else None; expected_boundary={"environment_replaced":True,"exact_job_export_names":exports,"export_all":False,"reservation_created_before_sbatch":True,"same_inode_retained":True,"launcher_submitted_from_retained_fd":True,"retained_wrapper_fd":retained_wrapper_fd,"reservation_device":sinfo.st_dev,"reservation_inode":sinfo.st_ino,"output_parent_device":oi.st_dev,"output_parent_inode":oi.st_ino,"success_mode":"0444"}
 cluster=submitted.get("cluster") if isinstance(submitted,dict) else None; expected_stdout=(job+((';'+cluster) if cluster is not None else '')+'\n').encode("ascii") if isinstance(cluster,(str,type(None))) else b""
 if (not isinstance(submitted,dict) or set(submitted)!={"job_id","cluster","stdout_sha256","stderr_sha256"} or submitted.get("job_id")!=job or (cluster is not None and (not isinstance(cluster,str) or not cluster or "\n" in cluster or ";" in cluster)) or submitted.get("stdout_sha256")!=hashlib.sha256(expected_stdout).hexdigest() or submitted.get("stderr_sha256")!=hashlib.sha256(b"").hexdigest() or type(retained_wrapper_fd) is not int or retained_wrapper_fd<3 or sub.get("status")!="submitted" or sub.get("submission_success") is not True or sub.get("job_success") is not None or sub.get("request")!=expected_request or boundary!=expected_boundary or sub.get("inputs")!={"wrapper":str(WRAPPER),"wrapper_sha256":WRAPPER_SHA,"python":str(PYTHON),"python_sha256":PYTHON_SHA,"postflight":str(POSTFLIGHT),"postflight_sha256":postflight_sha,"release_manifest":str(MANIFEST),"release_manifest_file_sha256":manifest_sha,"release_manifest_digest":manifest_digest} or sub.get("outputs")!={"output_parent":str(OUTPUT),"job_output_root":str(root),"submission_receipt":str(SUBMISSION)} or sub.get("authority")!=AUTHORITY): die("submission binding differs")
 ef={"schema_version","status","slurm_job_id","job_success","slurm_terminal_verified","compute_bash","retained_fd","submission_binding","wrapper_sha256","scientific_generation_entered","authority","receipt_digest"}
 ev,eraw,_=read_sealed(evidence,"saic-compute-bash-retained-fd-probe-evidence-v1",ef)
 if ev["slurm_job_id"]!=job or ev["status"]!="job_completed_awaiting_external_slurm_admission" or ev["job_success"] is not None or ev["slurm_terminal_verified"] is not False or ev["scientific_generation_entered"] is not False or ev["authority"]!=AUTHORITY or ev.get("wrapper_sha256")!=WRAPPER_SHA: die("evidence state differs")
 expected_binding={"submission_receipt_path":str(SUBMISSION),"submission_receipt_file_sha256":hashlib.sha256(sraw).hexdigest(),"submission_receipt_digest":sub["receipt_digest"],"submission_receipt_identity":f"{sinfo.st_dev}:{sinfo.st_ino}","output_parent_identity":f"{oi.st_dev}:{oi.st_ino}","release_manifest_file_sha256":manifest_sha,"release_manifest_digest":manifest_digest,"postflight_sha256":postflight_sha}
 if ev.get("submission_binding")!=expected_binding: die("evidence submission binding differs")
 bash=ev["compute_bash"]; required={"path","identity","sha256","version_stdout_sha256","version_stdout_size","version_first_line","brace_fd_redirection_supported","retained_fd_survives_bash_script_handoff","varredir_close_option_required"}
 if set(bash)!=required or bash["path"]!="/usr/bin/bash" or not re.fullmatch(r"regular file:[0-9]+:[1-9][0-9]*:0:1:[0-9]+:[1-9][0-9]*",str(bash["identity"])) or not re.fullmatch(r"[0-9a-f]{64}",bash["sha256"]) or not re.fullmatch(r"[0-9a-f]{64}",bash["version_stdout_sha256"]) or not isinstance(bash["version_stdout_size"],int) or bash["version_stdout_size"]<=0 or not bash["version_first_line"] or bash["brace_fd_redirection_supported"] is not True or bash["retained_fd_survives_bash_script_handoff"] is not True or bash["varredir_close_option_required"] is not False: die("compute Bash evidence differs")
 fd=ev["retained_fd"]
 fd_fields={"child_proc_fd_path","original_retained_path","wrong_decoy_path","parent_fd_identity","child_fd_identity","original_retained_identity","wrong_decoy_identity","retained_payload_sha256","wrong_decoy_sha256","logical_leaf_replaced_after_open","parent_and_child_inode_equal","wrong_decoy_inode_differs","wrong_decoy_sha_differs","wrong_decoy_exit_code"}
 if not isinstance(fd,dict) or set(fd)!=fd_fields or fd.get("logical_leaf_replaced_after_open") is not True or fd["parent_and_child_inode_equal"] is not True or fd["wrong_decoy_inode_differs"] is not True or fd["wrong_decoy_sha_differs"] is not True or fd["wrong_decoy_exit_code"]!=97 or fd["parent_fd_identity"]!=fd["child_fd_identity"] or fd["child_fd_identity"]!=fd["original_retained_identity"]: die("retained FD evidence differs")
 if fd["wrong_decoy_identity"].split(":",2)[:2]==fd["child_fd_identity"].split(":",2)[:2] or fd["wrong_decoy_sha256"]==fd["retained_payload_sha256"] or fd["retained_payload_sha256"]!=WRAPPER_SHA: die("retained FD actuals differ")
 retained=fixture/"payload.retained"; decoy=fixture/"payload.logical"; version=fixture/"bash-version.stdout"
 if set(fixture.iterdir())!={retained,decoy,version} or set(root.iterdir())!={evidence}: die("artifact closure differs")
 def identity(path):
  s=path.lstat(); return f"{s.st_dev}:{s.st_ino}:{s.st_nlink}:{stat.S_IMODE(s.st_mode):o}:{s.st_size}"
 child_path_match=re.fullmatch(r"/proc/([1-9][0-9]*)/fd/([0-9]+)",str(fd.get("child_proc_fd_path")))
 if fd.get("original_retained_path")!=str(retained) or fd.get("wrong_decoy_path")!=str(decoy) or not child_path_match or str(int(child_path_match.group(2)))!=child_path_match.group(2) or int(child_path_match.group(2))<3 or identity(retained)!=fd["original_retained_identity"] or identity(decoy)!=fd["wrong_decoy_identity"] or sha(retained)!=WRAPPER_SHA or retained.lstat().st_uid!=os.getuid() or stat.S_IMODE(retained.lstat().st_mode)!=0o444 or decoy.read_bytes()!=b"#!/usr/bin/env bash\nexit 97\n" or sha(decoy)!=fd["wrong_decoy_sha256"] or decoy.lstat().st_uid!=os.getuid() or stat.S_IMODE(decoy.lstat().st_mode)!=0o444 or sha(version)!=bash["version_stdout_sha256"] or version.lstat().st_uid!=os.getuid() or stat.S_IMODE(version.lstat().st_mode)!=0o444 or version.stat().st_size!=bash["version_stdout_size"] or version.read_bytes().splitlines()[0].decode("ascii")!=bash["version_first_line"]: die("artifact actuals differ")
 ss=SACCT.lstat()
 if SACCT.resolve(strict=True)!=SACCT or not stat.S_ISREG(ss.st_mode) or stat.S_ISLNK(ss.st_mode) or ss.st_uid!=0 or ss.st_nlink!=1 or stat.S_IMODE(ss.st_mode)&0o022 or not os.access(SACCT,os.X_OK) or sha(SACCT)!=SACCT_SHA: die("sacct differs")
 fields="JobIDRaw,State,ExitCode,AllocTRES%256,NodeList,Start,End,Elapsed,SubmitLine%8192"; c=subprocess.run([str(SACCT),"-j",job,"-X","-n","-P","-o",fields],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,timeout=60,env={"PATH":"/usr/bin:/bin","LANG":"C","LC_ALL":"C"}); lines=c.stdout.decode("ascii").splitlines(); cols=lines[0].split("|") if len(lines)==1 else []
 tres={}
 for token in cols[3].split(",") if len(cols)==9 else []:
  if "=" not in token: die("AllocTRES differs")
  k,v=token.split("=",1)
  if not k or k in tres: die("AllocTRES closure differs")
  tres[k]=v
 expected_tres={"billing":"4","cpu":"4","gres/gpu:mi210":"1","gres/gpu":"1","mem":"8G","node":"1"}
 expected_env={exports[0]:str(WRAPPER),exports[1]:WRAPPER_SHA,exports[2]:str(PYTHON),exports[3]:PYTHON_SHA,exports[4]:str(OUTPUT),exports[5]:str(oi.st_dev),exports[6]:str(oi.st_ino),exports[7]:str(SUBMISSION),exports[8]:str(sinfo.st_dev),exports[9]:str(sinfo.st_ino),exports[10]:str(POSTFLIGHT),exports[11]:postflight_sha,exports[12]:str(MANIFEST),exports[13]:manifest_sha,exports[14]:manifest_digest}
 if list(expected_env)!=exports: die("SubmitLine export closure differs")
 submit_fixed=" ".join(["/usr/bin/sbatch","--parsable",f"--output={LOGS}/saic-bash-fd-probe1-%j.out",f"--error={LOGS}/saic-bash-fd-probe1-%j.err","--export=NONE,"+",".join(f"{k}={expected_env[k]}" for k in exports)])
 submit_match=re.fullmatch(re.escape(submit_fixed+" /proc/self/fd/")+r"([0-9]+)",cols[8] if len(cols)==9 else "")
 if c.returncode or c.stderr or len(cols)!=9 or cols[0]!=job or cols[1]!="COMPLETED" or cols[2]!="0:0" or tres!=expected_tres or not cols[4] or cols[4] in {"Unknown","None assigned"} or not cols[5] or cols[5] in {"Unknown","None"} or not cols[6] or cols[6] in {"Unknown","None"} or not re.fullmatch(r"[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}|[0-9]{2}:[0-9]{2}:[0-9]{2}",cols[7]) or not submit_match or str(int(submit_match.group(1)))!=submit_match.group(1) or int(submit_match.group(1))<3 or int(submit_match.group(1))!=retained_wrapper_fd: die("terminal sacct differs")
 expected_logs={LOGS/f"saic-bash-fd-probe1-{job}.out",LOGS/f"saic-bash-fd-probe1-{job}.err"}
 if set(LOGS.iterdir())!=expected_logs: die("log closure differs")
 for lp in expected_logs:
  ls=lp.lstat()
  if not stat.S_ISREG(ls.st_mode) or stat.S_ISLNK(ls.st_mode) or ls.st_uid!=os.getuid() or ls.st_nlink!=1 or stat.S_IMODE(ls.st_mode)&0o022: die("log differs")
 if any(lp.stat().st_size for lp in expected_logs): die("successful logs are not empty")
 sub2,sraw2,sinfo2=read_sealed(SUBMISSION,"saic-compute-bash-retained-fd-probe-submission-v1",sf); ev2,eraw2,_=read_sealed(evidence,"saic-compute-bash-retained-fd-probe-evidence-v1",ef)
 if sraw2!=sraw or eraw2!=eraw or (sinfo2.st_dev,sinfo2.st_ino)!=(sinfo.st_dev,sinfo.st_ino) or sub2!=sub or ev2!=ev or set(OUTPUT.iterdir())!={SUBMISSION,root,fixture}: die("post-sacct retained inputs changed")
 formal_bash={k:bash[k] for k in ("path","sha256","version_stdout_sha256","version_first_line","brace_fd_redirection_supported","retained_fd_survives_bash_script_handoff","varredir_close_option_required")}
 core={"schema_version":"saic-compute-bash-retained-fd-probe-admission-v1","status":"terminal_completed_compute_bash_retained_fd_admitted","slurm_job_id":job,"submission_receipt_sha256":hashlib.sha256(sraw).hexdigest(),"submission_receipt_digest":sub["receipt_digest"],"submission_receipt_identity":f"{sinfo.st_dev}:{sinfo.st_ino}","operational_evidence_sha256":hashlib.sha256(eraw).hexdigest(),"operational_evidence_digest":ev["receipt_digest"],"release_manifest_file_sha256":manifest_sha,"release_manifest_digest":manifest_digest,"wrapper_sha256":WRAPPER_SHA,"postflight_sha256":postflight_sha,"compute_bash":formal_bash,"compute_bash_observation":{"identity":bash["identity"],"version_stdout_size":bash["version_stdout_size"]},"retained_fd":fd,"job_success":True,"slurm_terminal_verified":True,"slurm":{"sacct_sha256":SACCT_SHA,"stdout_sha256":hashlib.sha256(c.stdout).hexdigest(),"exact_single_row":True,"state":"COMPLETED","exit_code":"0:0","alloc_tres":expected_tres,"node_list":cols[4],"start":cols[5],"end":cols[6],"elapsed":cols[7],"submit_line_sha256":hashlib.sha256(cols[8].encode("ascii")).hexdigest(),"retained_wrapper_fd":retained_wrapper_fd,"exact_submit_line":True},"authority":AUTHORITY}
 write_create(ADMISSION,{**core,"receipt_digest":hashlib.sha256(canonical(core)).hexdigest()}); os._exit(0)
if __name__=="__main__": main()

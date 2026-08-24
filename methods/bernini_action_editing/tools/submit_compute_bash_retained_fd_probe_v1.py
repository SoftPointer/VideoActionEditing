#!/usr/bin/env python3
"""Exactly once submit the zero-science compute-Bash retained-FD probe."""
from __future__ import annotations

import argparse, hashlib, json, os, re, stat, subprocess
from pathlib import Path

ROOT = Path("/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809")
WRAPPER_SHA = "8283e73ddf240d1ed8946f5682910bcfafaf24a88ae6c175c80b6a4597a75016"
POSTFLIGHT_SHA = "40c58b60afc6f20b569d15dd780a16e97722afdfefb8186444192c3dbf0868b9"
PYTHON = Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12")
PYTHON_SHA = "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
STEM = "compute-bash-retained-fd-probe-8283e73d-r1"
RELEASE = ROOT / "releases" / STEM
WRAPPER = RELEASE / "auh_probe_compute_bash_retained_fd_v1.sbatch"
POSTFLIGHT = RELEASE / "postflight_compute_bash_retained_fd_probe_v1.py"
MANIFEST = RELEASE / "release-manifest.json"
OUTPUT = ROOT / "canaries" / STEM
RECEIPT = OUTPUT / "submission-receipt.json"
LOGS = ROOT / "slurm" / STEM
AUTHORITY = {"scientific":False,"generation":False,"training":False,"publication":False,"formal_job_authorized":False}
EXPORTS = ["SAIC_BASH_FD_PROBE_WRAPPER","SAIC_BASH_FD_PROBE_WRAPPER_SHA256","SAIC_BASH_FD_PROBE_PYTHON","SAIC_BASH_FD_PROBE_PYTHON_SHA256","SAIC_BASH_FD_PROBE_OUTPUT_PARENT","SAIC_BASH_FD_PROBE_OUTPUT_DEVICE","SAIC_BASH_FD_PROBE_OUTPUT_INODE","SAIC_BASH_FD_PROBE_SUBMISSION","SAIC_BASH_FD_PROBE_SUBMISSION_DEVICE","SAIC_BASH_FD_PROBE_SUBMISSION_INODE","SAIC_BASH_FD_PROBE_POSTFLIGHT","SAIC_BASH_FD_PROBE_POSTFLIGHT_SHA256","SAIC_BASH_FD_PROBE_RELEASE_MANIFEST","SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_SHA256","SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_DIGEST"]

def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def sha_fd(fd):
    h=hashlib.sha256(); os.lseek(fd,0,os.SEEK_SET)
    while True:
        chunk=os.read(fd,1024*1024)
        if not chunk: break
        h.update(chunk)
    os.lseek(fd,0,os.SEEK_SET); return h.hexdigest()
def die(msg): raise SystemExit(f"submit-compute-bash-fd-probe-v1: {msg}")
def exact_file(path,expected,digest,mode):
    if path!=expected or not path.is_absolute() or path.resolve(strict=True)!=path: die("file identity differs")
    s=path.lstat()
    if not stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode) or s.st_uid!=os.getuid() or s.st_nlink!=1 or stat.S_IMODE(s.st_mode)!=mode or sha(path)!=digest: die("file bytes/mode differ")
def exact_executable(path,expected,digest):
    if path!=expected or path.resolve(strict=True)!=path: die("executable identity differs")
    s=path.lstat()
    if not stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode) or s.st_nlink!=1 or stat.S_IMODE(s.st_mode)&0o022 or not os.access(path,os.X_OK) or sha(path)!=digest: die("executable differs")
def exact_dir(path,expected,mode):
    if path!=expected or path.resolve(strict=True)!=path: die("directory identity differs")
    s=path.lstat()
    if not stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode) or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=mode: die("directory mode differs")
def write_all(fd,data):
    view=memoryview(data)
    while view:
        n=os.write(fd,view)
        if n<=0: die("receipt write stalled")
        view=view[n:]
def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); os.fsync(fd); os.close(fd)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--wrapper",required=True); p.add_argument("--wrapper-sha256",required=True); p.add_argument("--postflight",required=True); p.add_argument("--postflight-sha256",required=True); p.add_argument("--release-manifest",required=True); p.add_argument("--python",required=True); p.add_argument("--python-sha256",required=True); p.add_argument("--output-parent",required=True); p.add_argument("--receipt",required=True); p.add_argument("--slurm-log-dir",required=True); a=p.parse_args()
    if a.wrapper_sha256!=WRAPPER_SHA or a.postflight_sha256!=POSTFLIGHT_SHA or a.python_sha256!=PYTHON_SHA: die("SHA pin differs")
    exact_file(Path(a.wrapper),WRAPPER,WRAPPER_SHA,0o444); exact_file(Path(a.postflight),POSTFLIGHT,POSTFLIGHT_SHA,0o444); exact_executable(Path(a.python),PYTHON,PYTHON_SHA)
    if Path(a.release_manifest)!=MANIFEST: die("manifest path differs")
    exact_file(MANIFEST,MANIFEST,sha(MANIFEST),0o444)
    mraw=MANIFEST.read_bytes(); manifest=json.loads(mraw.decode("ascii")); mu=dict(manifest); mdigest=mu.pop("receipt_digest",None); msha=hashlib.sha256(mraw).hexdigest()
    if (set(manifest)!={"schema_version","status","stem","release_root","output_parent","wrapper","postflight","executables","authority","receipt_digest"} or manifest.get("schema_version")!="saic-compute-bash-retained-fd-probe-release-v1" or manifest.get("status")!="sealed_before_submission" or manifest.get("stem")!=STEM or manifest.get("release_root")!=str(RELEASE) or manifest.get("output_parent")!=str(OUTPUT) or manifest.get("wrapper")!={"path":str(WRAPPER),"sha256":WRAPPER_SHA} or manifest.get("postflight")!={"path":str(POSTFLIGHT),"sha256":POSTFLIGHT_SHA} or manifest.get("executables")!={"python":str(PYTHON),"python_sha256":PYTHON_SHA,"sacct":"/usr/bin/sacct","sacct_sha256":"fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"} or manifest.get("authority")!=AUTHORITY or mraw!=canonical(manifest)+b"\n" or mdigest!=hashlib.sha256(canonical(mu)).hexdigest()): die("release manifest differs")
    exact_dir(RELEASE,RELEASE,0o555)
    if set(RELEASE.iterdir())!={WRAPPER,POSTFLIGHT,MANIFEST}: die("release closure differs")
    out=Path(a.output_parent); logs=Path(a.slurm_log_dir); receipt=Path(a.receipt)
    exact_dir(out,OUTPUT,0o700); exact_dir(logs,LOGS,0o700)
    if any(out.iterdir()) or any(logs.iterdir()) or receipt!=RECEIPT or receipt.exists() or receipt.is_symlink(): die("fresh output/log differs")
    oi=out.lstat(); li=logs.lstat()
    wfd=os.open(WRAPPER,os.O_RDONLY|os.O_NOFOLLOW); wi=os.fstat(wfd)
    leaf=WRAPPER.lstat()
    if not stat.S_ISREG(wi.st_mode) or wi.st_nlink!=1 or stat.S_IMODE(wi.st_mode)!=0o444 or (wi.st_dev,wi.st_ino)!=(leaf.st_dev,leaf.st_ino) or sha_fd(wfd)!=WRAPPER_SHA: die("retained wrapper differs")
    rfd=os.open(receipt,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); ri=os.fstat(rfd)
    if not stat.S_ISREG(ri.st_mode) or ri.st_uid!=os.getuid() or ri.st_nlink!=1 or stat.S_IMODE(ri.st_mode)!=0o600 or (receipt.lstat().st_dev,receipt.lstat().st_ino)!=(ri.st_dev,ri.st_ino): die("reservation differs")
    provisional={"schema_version":"saic-compute-bash-retained-fd-probe-submission-v1","status":"reserved_before_sbatch","submission_success":False,"job_success":None,"wrapper_sha256":WRAPPER_SHA}
    write_all(rfd,canonical(provisional)+b"\n"); os.fsync(rfd); fsync_dir(out)
    env={"SAIC_BASH_FD_PROBE_WRAPPER":str(WRAPPER),"SAIC_BASH_FD_PROBE_WRAPPER_SHA256":WRAPPER_SHA,"SAIC_BASH_FD_PROBE_PYTHON":str(PYTHON),"SAIC_BASH_FD_PROBE_PYTHON_SHA256":PYTHON_SHA,"SAIC_BASH_FD_PROBE_OUTPUT_PARENT":str(OUTPUT),"SAIC_BASH_FD_PROBE_OUTPUT_DEVICE":str(oi.st_dev),"SAIC_BASH_FD_PROBE_OUTPUT_INODE":str(oi.st_ino),"SAIC_BASH_FD_PROBE_SUBMISSION":str(RECEIPT),"SAIC_BASH_FD_PROBE_SUBMISSION_DEVICE":str(ri.st_dev),"SAIC_BASH_FD_PROBE_SUBMISSION_INODE":str(ri.st_ino),"SAIC_BASH_FD_PROBE_POSTFLIGHT":str(POSTFLIGHT),"SAIC_BASH_FD_PROBE_POSTFLIGHT_SHA256":POSTFLIGHT_SHA,"SAIC_BASH_FD_PROBE_RELEASE_MANIFEST":str(MANIFEST),"SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_SHA256":msha,"SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_DIGEST":mdigest}
    if list(env)!=EXPORTS: die("export closure differs")
    current_reservation=os.fstat(rfd)
    if (out.lstat().st_dev,out.lstat().st_ino)!=(oi.st_dev,oi.st_ino) or (logs.lstat().st_dev,logs.lstat().st_ino)!=(li.st_dev,li.st_ino) or (receipt.lstat().st_dev,receipt.lstat().st_ino)!=(ri.st_dev,ri.st_ino) or (current_reservation.st_dev,current_reservation.st_ino)!=(ri.st_dev,ri.st_ino) or stat.S_IMODE(current_reservation.st_mode)!=0o600 or current_reservation.st_nlink!=1 or set(out.iterdir())!={receipt} or any(logs.iterdir()) or sha_fd(wfd)!=WRAPPER_SHA or set(RELEASE.iterdir())!={WRAPPER,POSTFLIGHT,MANIFEST}: die("pre-sbatch retained closure differs")
    cmd=["/usr/bin/sbatch","--parsable",f"--output={LOGS}/saic-bash-fd-probe1-%j.out",f"--error={LOGS}/saic-bash-fd-probe1-%j.err","--export=NONE,"+",".join(f"{k}={v}" for k,v in env.items()),f"/proc/self/fd/{wfd}"]
    c=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,timeout=120,pass_fds=(wfd,),env={"PATH":"/usr/bin:/bin","LANG":"C","LC_ALL":"C"})
    m=re.fullmatch(rb"([0-9]+)(?:;([^\n;]+))?\n?",c.stdout)
    if c.returncode or not m:
        os.close(wfd); os.close(rfd); die("sbatch failed; reservation retained")
    job=m.group(1).decode("ascii")
    boundary={"environment_replaced":True,"exact_job_export_names":EXPORTS,"export_all":False,"reservation_created_before_sbatch":True,"same_inode_retained":True,"launcher_submitted_from_retained_fd":True,"retained_wrapper_fd":wfd,"reservation_device":ri.st_dev,"reservation_inode":ri.st_ino,"output_parent_device":oi.st_dev,"output_parent_inode":oi.st_ino,"success_mode":"0444"}
    core={"schema_version":"saic-compute-bash-retained-fd-probe-submission-v1","status":"submitted","submission_success":True,"job_success":None,"submitted_job":{"job_id":job,"cluster":m.group(2).decode("ascii") if m.group(2) else None,"stdout_sha256":hashlib.sha256(c.stdout).hexdigest(),"stderr_sha256":hashlib.sha256(c.stderr).hexdigest()},"request":{"job_name":"saic-bash-fd-probe1","partition":"faculty","qos":"bgqos","nodes":1,"ntasks":1,"cpus_per_task":4,"memory":"8G","walltime":"00:05:00","gpu_resource_requested":"gpu:mi210:1","hold":False,"dependency":None,"scientific_generation":False},"submission_boundary":boundary,"inputs":{"wrapper":str(WRAPPER),"wrapper_sha256":WRAPPER_SHA,"python":str(PYTHON),"python_sha256":PYTHON_SHA,"postflight":str(POSTFLIGHT),"postflight_sha256":POSTFLIGHT_SHA,"release_manifest":str(MANIFEST),"release_manifest_file_sha256":msha,"release_manifest_digest":mdigest},"outputs":{"output_parent":str(OUTPUT),"job_output_root":str(OUTPUT/f"job-{job}"),"submission_receipt":str(RECEIPT)},"authority":AUTHORITY}
    value={**core,"receipt_digest":hashlib.sha256(canonical(core)).hexdigest()}; raw=canonical(value)+b"\n"
    os.lseek(rfd,0,os.SEEK_SET); os.ftruncate(rfd,0); write_all(rfd,raw); os.fsync(rfd); os.lseek(rfd,0,os.SEEK_SET)
    expected_logs={LOGS/f"saic-bash-fd-probe1-{job}.out",LOGS/f"saic-bash-fd-probe1-{job}.err"}
    actual_logs=set(logs.iterdir())
    if os.read(rfd,len(raw)+1)!=raw or (out.lstat().st_dev,out.lstat().st_ino)!=(oi.st_dev,oi.st_ino) or (receipt.lstat().st_dev,receipt.lstat().st_ino)!=(ri.st_dev,ri.st_ino) or (logs.lstat().st_dev,logs.lstat().st_ino)!=(li.st_dev,li.st_ino) or set(out.iterdir())!={receipt} or not actual_logs.issubset(expected_logs) or any(p.is_symlink() or not p.is_file() for p in actual_logs) or sha_fd(wfd)!=WRAPPER_SHA: die("terminal identity differs")
    os.fchmod(rfd,0o444); os.fsync(rfd)
    public=receipt.lstat()
    os.lseek(rfd,0,os.SEEK_SET)
    if os.read(rfd,len(raw)+1)!=raw or stat.S_IMODE(public.st_mode)!=0o444 or public.st_uid!=os.getuid() or public.st_nlink!=1 or public.st_size!=len(raw) or (public.st_dev,public.st_ino)!=(ri.st_dev,ri.st_ino): die("publication identity differs")
    os.close(rfd); os.close(wfd); fsync_dir(out); os._exit(0)
if __name__=="__main__": main()

#!/bin/bash -p

# Fresh nonhold/no-dependency submission boundary for the fixed WORLD8
# trajectory diagnostic.  Submission success is never job or science success.

case "$-" in
  *p*) ;;
  *) echo "[submit-graft-short-traj-r3] ERROR privileged bash required" >&2; exit 2 ;;
esac
set -Eeuo pipefail
umask 077
[[ "$#" -eq 0 ]] || { echo "arbitrary arguments are forbidden" >&2; exit 2; }

readonly names=(
  GRAFT_TRAJ_SOURCE_ARCHIVE GRAFT_TRAJ_SOURCE_ARCHIVE_SHA256
  GRAFT_TRAJ_RUNTIME_CLOSURE GRAFT_TRAJ_RUNTIME_CLOSURE_SHA256
  GRAFT_TRAJ_PYTHON_BIN GRAFT_TRAJ_PYTHON_SHA256
  BERNINI_OFFICIAL_ROOT BERNINI_VEOMNI_ROOT BERNINI_ACTION_CHECKPOINT
  BERNINI_CHECKPOINT_CONTENT_MANIFEST BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
  GRAFT_TRAJ_PLAN GRAFT_TRAJ_PLAN_SHA256
  GRAFT_TRAJ_TERMINAL_ADMISSION GRAFT_TRAJ_TERMINAL_ADMISSION_SHA256
  GRAFT_TRAJ_TERMINAL_MATERIALIZER_RUNTIME_SHA256
  GRAFT_TRAJ_OUTPUT_ROOT GRAFT_TRAJ_LAUNCHER_SOURCE GRAFT_TRAJ_LAUNCHER_SHA256
  GRAFT_TRAJ_RUNNER_SHA256
)
for name in "${names[@]}"; do [[ -n "${!name:-}" ]] || { echo "missing ${name}" >&2; exit 2; }; done

exec /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
  "${GRAFT_TRAJ_PYTHON_BIN}" -I -S -B - "$@" \
  "${GRAFT_TRAJ_SOURCE_ARCHIVE}" "${GRAFT_TRAJ_SOURCE_ARCHIVE_SHA256}" \
  "${GRAFT_TRAJ_RUNTIME_CLOSURE}" "${GRAFT_TRAJ_RUNTIME_CLOSURE_SHA256}" \
  "${GRAFT_TRAJ_PYTHON_BIN}" "${GRAFT_TRAJ_PYTHON_SHA256}" \
  "${BERNINI_OFFICIAL_ROOT}" "${BERNINI_VEOMNI_ROOT}" \
  "${BERNINI_ACTION_CHECKPOINT}" "${BERNINI_CHECKPOINT_CONTENT_MANIFEST}" \
  "${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}" \
  "${GRAFT_TRAJ_PLAN}" "${GRAFT_TRAJ_PLAN_SHA256}" \
  "${GRAFT_TRAJ_TERMINAL_ADMISSION}" "${GRAFT_TRAJ_TERMINAL_ADMISSION_SHA256}" \
  "${GRAFT_TRAJ_TERMINAL_MATERIALIZER_RUNTIME_SHA256}" \
  "${GRAFT_TRAJ_OUTPUT_ROOT}" "${GRAFT_TRAJ_LAUNCHER_SOURCE}" \
  "${GRAFT_TRAJ_LAUNCHER_SHA256}" "${GRAFT_TRAJ_RUNNER_SHA256}" <<'PY'
import hashlib,json,os,re,stat,subprocess,sys
from pathlib import Path
(archive,archive_sha,closure,closure_sha,python_bin,python_sha,bernini,veomni,
 checkpoint,checkpoint_manifest,checkpoint_manifest_sha,plan,plan_sha,terminal,
 terminal_sha,materializer_sha,output,launcher,launcher_sha,runner_sha)=sys.argv[1:]
SHA=re.compile(r"[0-9a-f]{64}\Z"); JOB=re.compile(r"([1-9][0-9]*)(?:;[^\n]+)?\n\Z")
def die(message): raise SystemExit(message)
def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
for value,label in ((archive_sha,"archive"),(closure_sha,"closure"),(python_sha,"Python"),(checkpoint_manifest_sha,"checkpoint manifest"),(plan_sha,"plan"),(terminal_sha,"terminal"),(materializer_sha,"materializer"),(launcher_sha,"launcher"),(runner_sha,"runner")):
 if SHA.fullmatch(value) is None: die(label+" SHA differs")
for value,label in ((archive,"archive"),(closure,"closure"),(python_bin,"Python"),(bernini,"Bernini"),(veomni,"VeOmni"),(checkpoint,"checkpoint"),(checkpoint_manifest,"checkpoint manifest"),(plan,"plan"),(terminal,"terminal"),(output,"output"),(launcher,"launcher")):
 if not Path(value).is_absolute(): die(label+" path is not absolute")
for value,wanted,label in ((archive,archive_sha,"archive"),(closure,closure_sha,"closure"),(python_bin,python_sha,"Python"),(checkpoint_manifest,checkpoint_manifest_sha,"checkpoint manifest"),(plan,plan_sha,"plan"),(terminal,terminal_sha,"terminal"),(launcher,launcher_sha,"launcher")):
 path=Path(value); info=path.lstat()
 if path.resolve(strict=True)!=path or stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode): die(label+" is not exact plain file")
 if digest(path.read_bytes())!=wanted: die(label+" bytes differ")
out=Path(output); parent=out.parent
receipt_path=Path(str(out)+".submission.receipt.json")
if not parent.is_dir() or out.exists() or receipt_path.exists(): die("fresh output/submission receipt unavailable")
exports={
"GRAFT_TRAJ_SOURCE_ARCHIVE":archive,"GRAFT_TRAJ_SOURCE_ARCHIVE_SHA256":archive_sha,
"GRAFT_TRAJ_RUNTIME_CLOSURE":closure,"GRAFT_TRAJ_RUNTIME_CLOSURE_SHA256":closure_sha,
"GRAFT_TRAJ_PYTHON_BIN":python_bin,"GRAFT_TRAJ_PYTHON_SHA256":python_sha,
"BERNINI_OFFICIAL_ROOT":bernini,"BERNINI_VEOMNI_ROOT":veomni,"BERNINI_ACTION_CHECKPOINT":checkpoint,
"BERNINI_CHECKPOINT_CONTENT_MANIFEST":checkpoint_manifest,"BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256":checkpoint_manifest_sha,
"GRAFT_TRAJ_PLAN":plan,"GRAFT_TRAJ_PLAN_SHA256":plan_sha,"GRAFT_TRAJ_TERMINAL_ADMISSION":terminal,
"GRAFT_TRAJ_TERMINAL_ADMISSION_SHA256":terminal_sha,"GRAFT_TRAJ_TERMINAL_MATERIALIZER_RUNTIME_SHA256":materializer_sha,
"GRAFT_TRAJ_OUTPUT_ROOT":output,"GRAFT_TRAJ_LAUNCHER_SOURCE":launcher,"GRAFT_TRAJ_LAUNCHER_SHA256":launcher_sha,
"GRAFT_TRAJ_RUNNER_SHA256":runner_sha}
export_arg=",".join(name+"="+value for name,value in exports.items())
argv=["/usr/bin/sbatch","--parsable","--export="+export_arg,"--job-name=graft-short-traj-r3","--partition=faculty","--qos=bgqos","--nodes=1","--ntasks=1","--cpus-per-task=32","--mem=256G","--gres=gpu:mi210:8","--time=24:00:00","--exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318","--output=/vast/users/guangyi.chen/slurm-%j.out",launcher]
fd=os.open(receipt_path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
try:
 reserved=os.fstat(fd); leaf=receipt_path.lstat()
 if not stat.S_ISREG(reserved.st_mode) or stat.S_IMODE(reserved.st_mode)!=0o600 or reserved.st_nlink!=1 or reserved.st_size!=0 or (reserved.st_dev,reserved.st_ino)!=(leaf.st_dev,leaf.st_ino): die("submission reservation differs")
 completed=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,env={"PATH":"/usr/bin:/bin","LC_ALL":"C","LANG":"C"},timeout=120)
 if completed.returncode!=0: die("sbatch failed exit=%d stderr_sha256=%s"%(completed.returncode,digest(completed.stderr)))
 try: token=completed.stdout.decode("ascii")
 except UnicodeDecodeError: die("sbatch output is not ASCII")
 match=JOB.fullmatch(token)
 if match is None: die("sbatch job id framing differs")
 job_id=match.group(1)
 authority={name:False for name in ("action_authority","identity_authority","quality_authority","training_authority","checkpoint_authority","publication_authority","production_authority","scientific_success_claimed","job_success_claimed")}
 core={"schema_version":"bernini-graft-phase-a-short-trajectory-submission-v1","status":"submitted","submission_success":True,"job_success":None,"job_terminal_state_observed":False,"submitted_job":{"job_id":job_id,"stdout_sha256":digest(completed.stdout),"stderr_sha256":digest(completed.stderr)},"request":{"job_name":"graft-short-traj-r3","partition":"faculty","qos":"bgqos","nodes":1,"ntasks":1,"cpus_per_task":32,"memory":"256G","gpu_resource_requested":"gpu:mi210:8","walltime":"24:00:00","world_size":8,"dp_size":2,"sp_size":4,"hold":False,"dependency":None},"inputs":{"runner_sha256":runner_sha,"plan_sha256":plan_sha,"source_archive_sha256":archive_sha,"runtime_closure_sha256":closure_sha,"launcher_sha256":launcher_sha},"outputs":{"logical_output_root":output,"submission_receipt_path":str(receipt_path),"create_only":True,"reservation_created_before_sbatch":True,"reservation_mode_before_finalization":"0600","success_mode":"0444","same_inode_retained_across_sbatch":True,"failed_submission_reservation_is_not_success":True},"authority":authority}
 receipt={**core,"digest":digest(canonical(core))}; payload=canonical(receipt)+b"\n"
 offset=0
 while offset<len(payload):
  wrote=os.write(fd,payload[offset:])
  if wrote<=0: die("submission receipt write stalled")
  offset+=wrote
 os.fsync(fd); os.lseek(fd,0,os.SEEK_SET)
 if os.read(fd,len(payload)+1)!=payload: die("submission receipt reread differs")
 final_leaf=receipt_path.lstat(); final_fd=os.fstat(fd)
 if (final_fd.st_dev,final_fd.st_ino)!=(reserved.st_dev,reserved.st_ino) or (final_leaf.st_dev,final_leaf.st_ino)!=(reserved.st_dev,reserved.st_ino) or stat.S_IMODE(final_fd.st_mode)!=0o600 or final_fd.st_nlink!=1: die("submission reservation inode changed")
 os.fchmod(fd,0o444); os.fsync(fd)
finally: os.close(fd)
print(job_id)
PY

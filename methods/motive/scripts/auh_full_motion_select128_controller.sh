#!/usr/bin/env bash
# Wait for the terminal 32-shard postcheck receipt, then invoke the frozen
# selector exactly once to publish an exact-128 dataset.  This controller does
# not use Slurm or SSH.  Run it under nohup with stdin redirected from
# /dev/null; all publications are create-only and a nonzero exit is terminal.

set -Eeuo pipefail
umask 077

snapshot="${MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT:?set MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT}"
python_bin="${MOTIVE_FULL_MOTION_SELECT_PYTHON:?set MOTIVE_FULL_MOTION_SELECT_PYTHON}"
generation_primary="${MOTIVE_FULL_MOTION_GENERATION_PRIMARY:?set MOTIVE_FULL_MOTION_GENERATION_PRIMARY}"
generation_done="${MOTIVE_FULL_MOTION_GENERATION_DONE:?set MOTIVE_FULL_MOTION_GENERATION_DONE}"
shard_manifest_root="${MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR:?set MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR}"
wan_output_root="${MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT:?set MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT}"
postcheck_root="${MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT:?set MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT}"
exact128_output="${MOTIVE_FULL_MOTION_EXACT128_OUTPUT:?set MOTIVE_FULL_MOTION_EXACT128_OUTPUT}"
controller_receipt="${MOTIVE_FULL_MOTION_EXACT128_RECEIPT:?set MOTIVE_FULL_MOTION_EXACT128_RECEIPT}"
dispatcher_receipt="${MOTIVE_FULL_MOTION_POSTCHECK_DISPATCH_RECEIPT:-${postcheck_root}/dispatcher_receipt.json}"
wait_seconds="${MOTIVE_FULL_MOTION_EXACT128_WAIT_SECONDS:-604800}"
poll_seconds="${MOTIVE_FULL_MOTION_EXACT128_POLL_SECONDS:-20}"
ffprobe_bin="${MOTIVE_FULL_MOTION_FFPROBE:-ffprobe}"
ffmpeg_bin="${MOTIVE_FULL_MOTION_FFMPEG:-ffmpeg}"
code_root="${snapshot}/methods/motive"
wan_shards_root="${wan_output_root}/wan_shards"
shard_manifest_leaf="${shard_manifest_root}/shards"
exact_size=128
min_multi_unit=32

fail() {
  echo "[full-motion-select128-controller] $*" >&2
  exit 2
}

require_absolute() {
  local label="$1"
  local path="$2"
  [[ "${path}" == /* && "${path}" != "/" ]] \
    || fail "${label} must be a non-root absolute path: ${path}"
}

require_plain_file() {
  local label="$1"
  local path="$2"
  [[ ! -L "${path}" && -f "${path}" ]] \
    || fail "${label} must be a regular non-symlink file: ${path}"
}

require_plain_directory() {
  local label="$1"
  local path="$2"
  [[ ! -L "${path}" && -d "${path}" ]] \
    || fail "${label} must be a non-symlink directory: ${path}"
}

for binding in \
  "snapshot:${snapshot}" \
  "python:${python_bin}" \
  "generation_primary:${generation_primary}" \
  "generation_done:${generation_done}" \
  "shard_manifest_root:${shard_manifest_root}" \
  "wan_output_root:${wan_output_root}" \
  "postcheck_root:${postcheck_root}" \
  "dispatcher_receipt:${dispatcher_receipt}" \
  "exact128_output:${exact128_output}" \
  "controller_receipt:${controller_receipt}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done
[[ "${wait_seconds}" =~ ^[1-9][0-9]*$ ]] \
  || fail "wait timeout must be a positive integer"
[[ "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] \
  || fail "poll interval must be a positive integer"

require_plain_directory "source snapshot" "${snapshot}"
require_plain_file "source snapshot closure" "${snapshot}/SOURCE_FILES.jsonl"
require_plain_file "selector implementation" \
  "${code_root}/motive/goku_full_motion_select128.py"
require_plain_file "selector Python" "${python_bin}"
[[ -x "${python_bin}" ]] || fail "selector Python is not executable"
require_plain_file "generation primary" "${generation_primary}"
require_plain_file "generation done" "${generation_done}"
require_plain_directory "shard manifest root" "${shard_manifest_root}"
require_plain_directory "shard manifest leaf" "${shard_manifest_leaf}"
require_plain_file "shard manifest summary" "${shard_manifest_root}/summary.json"
require_plain_file "shard manifest done" "${shard_manifest_root}/done.json"
require_plain_directory "Wan output root" "${wan_output_root}"
require_plain_directory "Wan shard roots" "${wan_shards_root}"
require_plain_directory "postcheck output root" "${postcheck_root}"

for shard_index in $(seq 0 31); do
  shard_id="$(printf 'shard_%03d' "${shard_index}")"
  require_plain_file "generation shard manifest ${shard_id}" \
    "${shard_manifest_leaf}/${shard_id}.jsonl"
  require_plain_directory "Wan run ${shard_id}" \
    "${wan_shards_root}/${shard_id}"
done

require_absolute "exact128 output parent" "${exact128_output%/*}"
require_absolute "controller receipt parent" "${controller_receipt%/*}"
require_plain_directory "exact128 output parent" "${exact128_output%/*}"
require_plain_directory "controller receipt parent" "${controller_receipt%/*}"
[[ -w "${exact128_output%/*}" ]] \
  || fail "exact128 output parent is not writable"
[[ -w "${controller_receipt%/*}" ]] \
  || fail "controller receipt parent is not writable"
[[ ! -e "${exact128_output}" && ! -L "${exact128_output}" ]] \
  || fail "create-only exact128 output already exists: ${exact128_output}"
[[ ! -e "${controller_receipt}" && ! -L "${controller_receipt}" ]] \
  || fail "create-only controller receipt already exists: ${controller_receipt}"
case "${controller_receipt}/" in
  "${exact128_output}/"*)
    fail "controller receipt must be outside the closed dataset directory"
    ;;
esac

for executable in "${ffprobe_bin}" "${ffmpeg_bin}"; do
  if [[ "${executable}" == /* ]]; then
    require_plain_file "media executable" "${executable}"
    [[ -x "${executable}" ]] || fail "media executable is not executable: ${executable}"
  else
    command -v "${executable}" >/dev/null \
      || fail "media executable is unavailable: ${executable}"
  fi
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${code_root}:${snapshot}"
# Python prepends the process working directory ahead of PYTHONPATH.  Enter
# the frozen code root so an unrelated caller checkout cannot shadow the
# snapshot's selector package.
cd "${code_root}"

temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/motive-select128-controller.XXXXXX")"
postcheck_list="${temporary_root}/postcheck_outputs.txt"
cleanup() {
  rm -f "${postcheck_list}"
  rmdir "${temporary_root}" 2>/dev/null || true
}
trap cleanup EXIT

deadline=$(( $(date +%s) + wait_seconds ))
while [[ ! -e "${dispatcher_receipt}" && ! -L "${dispatcher_receipt}" ]]; do
  (( $(date +%s) < deadline )) \
    || fail "timed out waiting for postcheck dispatcher terminal receipt"
  sleep "${poll_seconds}"
done
require_plain_file "postcheck dispatcher terminal receipt" "${dispatcher_receipt}"

validate_dispatch_receipt_py='import csv,hashlib,json,os,sys
from pathlib import Path

class Failure(RuntimeError): pass
def die(message): raise Failure(message)
def no_constant(value): die("non-finite JSON constant: "+value)
def no_duplicates(pairs):
    result={}
    for key,value in pairs:
        if key in result: die("duplicate JSON key: "+key)
        result[key]=value
    return result
def read(path,label):
    if path.is_symlink() or not path.is_file(): die(label+" is not a plain file: "+str(path))
    before=path.stat(); raw=path.read_bytes(); after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns): die(label+" changed while read")
    return raw
def load(raw,label):
    try: return json.loads(raw.decode("utf-8"),parse_constant=no_constant,object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError,json.JSONDecodeError) as error: raise Failure(label+" is not strict JSON") from error
def canon(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def sha(raw): return hashlib.sha256(raw).hexdigest()
receipt_path=Path(sys.argv[1]).resolve(strict=True)
root=Path(sys.argv[2]).resolve(strict=True)
snapshot=Path(sys.argv[3]).resolve(strict=True)
shard_leaf=Path(sys.argv[4]).resolve(strict=True)
wan_root=Path(sys.argv[5]).resolve(strict=True)
receipt_raw=read(receipt_path,"dispatcher receipt"); receipt=load(receipt_raw,"dispatcher receipt")
v2_keys={"schema_version","status","slurm_job_id","nodes","source_snapshot","generation_shard_dir","wan_shards_root","model","media_tools","status_tsv","status_tsv_sha256","completed_shards","failed_shards","shards","completed_at_utc","receipt_digest"}
if not isinstance(receipt,dict): die("dispatcher receipt schema differs")
if receipt.get("schema_version")!="motive-goku-full-motion-postcheck-dispatch-receipt-v2": die("dispatcher receipt identity differs")
if set(receipt)!=v2_keys: die("dispatcher v2 receipt schema differs")
media_tools=receipt.get("media_tools")
if not isinstance(media_tools,dict) or set(media_tools)!={"ffprobe","ffmpeg"}: die("dispatcher media-tools schema differs")
for label in ("ffprobe","ffmpeg"):
    record=media_tools.get(label)
    if not isinstance(record,dict) or set(record)!={"path","sha256"}: die("dispatcher "+label+" record schema differs")
    raw_path=record.get("path"); expected_sha=record.get("sha256")
    if not isinstance(raw_path,str): die("dispatcher "+label+" path differs")
    executable=Path(raw_path)
    if not executable.is_absolute() or executable==Path("/") or executable.is_symlink() or not executable.is_file() or not os.access(executable,os.X_OK): die("dispatcher "+label+" is not a plain absolute executable")
    if not isinstance(expected_sha,str) or len(expected_sha)!=64 or sha(read(executable,"dispatcher "+label+" executable"))!=expected_sha: die("dispatcher "+label+" executable SHA differs")
if receipt.get("status")!="complete" or receipt.get("completed_shards")!=32 or receipt.get("failed_shards")!=[]: die("dispatcher did not terminate successfully")
payload=dict(receipt); stored=payload.pop("receipt_digest",None)
if not isinstance(stored,str) or len(stored)!=64 or sha(canon(payload))!=stored: die("dispatcher receipt digest differs")
if Path(receipt.get("source_snapshot","")).resolve(strict=True)!=snapshot: die("dispatcher source snapshot differs")
if Path(receipt.get("generation_shard_dir","")).resolve(strict=True)!=shard_leaf: die("dispatcher shard-manifest directory differs")
if Path(receipt.get("wan_shards_root","")).resolve(strict=True)!=wan_root: die("dispatcher Wan root differs")
status_path=Path(receipt.get("status_tsv","")).resolve(strict=True)
if status_path!=root/"dispatcher_status.tsv": die("dispatcher status path differs")
status_raw=read(status_path,"dispatcher status")
if sha(status_raw)!=receipt.get("status_tsv_sha256"): die("dispatcher status SHA differs")
try: status_rows=list(csv.DictReader(status_raw.decode("utf-8").splitlines(),delimiter="\t"))
except UnicodeDecodeError as error: raise Failure("dispatcher status is not UTF-8") from error
shards=receipt.get("shards")
if not isinstance(shards,list) or len(shards)!=32 or status_rows!=shards: die("dispatcher receipt/status shard rows differ")
expected_fields={"shard","wave","slot","node","status","exit_code","output","receipt"}
outputs=[]
for index,row in enumerate(shards):
    shard=f"shard_{index:03d}"
    if not isinstance(row,dict) or set(row)!=expected_fields: die("dispatcher shard row schema differs: "+shard)
    if row.get("shard")!=shard or row.get("status")!="complete" or row.get("exit_code")!="0": die("dispatcher shard is not terminal-success: "+shard)
    output=(root/f"postcheck_{shard}.jsonl").resolve()
    shard_receipt=(root/f"postcheck_{shard}.receipt.json").resolve()
    if Path(row.get("output","")).resolve(strict=True)!=output or Path(row.get("receipt","")).resolve(strict=True)!=shard_receipt: die("dispatcher shard path differs: "+shard)
    read(output,"postcheck output "+shard); read(shard_receipt,"postcheck receipt "+shard)
    if any(ord(character)<32 for character in str(output)): die("control character in postcheck path")
    outputs.append(str(output))
if len(set(outputs))!=32: die("postcheck output paths are not unique")
if read(receipt_path,"dispatcher receipt closure")!=receipt_raw or read(status_path,"dispatcher status closure")!=status_raw: die("dispatcher closure changed during validation")
sys.stdout.write("\n".join(outputs)+"\n")'

if ! "${python_bin}" -c "${validate_dispatch_receipt_py}" \
  "${dispatcher_receipt}" "${postcheck_root}" "${snapshot}" \
  "${shard_manifest_leaf}" "${wan_shards_root}" >"${postcheck_list}"; then
  fail "postcheck dispatcher terminal receipt/32-output closure is invalid"
fi

selector_command=(
  "${python_bin}" -m motive.goku_full_motion_select128
  --generation-manifest "${generation_primary}"
  --finalizer-done "${generation_done}"
  --generation-shard-manifest-dir "${shard_manifest_root}"
  --generation-shard-index-dir "${wan_shards_root}"
  --output-dir "${exact128_output}"
  --exact-size "${exact_size}"
  --min-multi-unit "${min_multi_unit}"
  --ffprobe "${ffprobe_bin}"
  --ffmpeg "${ffmpeg_bin}"
)
postcheck_count=0
while IFS= read -r postcheck_output; do
  [[ -n "${postcheck_output}" ]] || fail "blank postcheck output path"
  selector_command+=(--postcheck-output "${postcheck_output}")
  postcheck_count=$(( postcheck_count + 1 ))
done <"${postcheck_list}"
(( postcheck_count == 32 )) || fail "expected exactly 32 postcheck outputs"

[[ ! -e "${exact128_output}" && ! -L "${exact128_output}" ]] \
  || fail "create-only exact128 output appeared before selection"
[[ ! -e "${controller_receipt}" && ! -L "${controller_receipt}" ]] \
  || fail "create-only controller receipt appeared before selection"
echo "[full-motion-select128-controller] postcheck closure complete; selecting exact=128 min_multi=32"
if ! "${selector_command[@]}"; then
  fail "goku_full_motion_select128 failed; no terminal controller receipt published"
fi

write_terminal_receipt_py='import hashlib,json,os,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path

class Failure(RuntimeError): pass
def die(message): raise Failure(message)
def no_constant(value): die("non-finite JSON constant: "+value)
def no_duplicates(pairs):
    result={}
    for key,value in pairs:
        if key in result: die("duplicate JSON key: "+key)
        result[key]=value
    return result
def read(path,label):
    if path.is_symlink() or not path.is_file(): die(label+" is not a plain file: "+str(path))
    before=path.stat(); raw=path.read_bytes(); after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns): die(label+" changed while read")
    return raw
def load(raw,label):
    try: return json.loads(raw.decode("utf-8"),parse_constant=no_constant,object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError,json.JSONDecodeError) as error: raise Failure(label+" is not strict JSON") from error
def canon(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def sha(raw): return hashlib.sha256(raw).hexdigest()
def file_record(path,label):
    raw=read(path,label)
    return {"path":str(path),"sha256":sha(raw),"bytes":len(raw)},raw
root=Path(sys.argv[1]).resolve(strict=True); receipt_path=Path(sys.argv[2])
primary=Path(sys.argv[3]).resolve(strict=True); generation_done=Path(sys.argv[4]).resolve(strict=True)
shards=Path(sys.argv[5]).resolve(strict=True); wan=Path(sys.argv[6]).resolve(strict=True)
dispatch=Path(sys.argv[7]).resolve(strict=True); output_list=Path(sys.argv[8]).resolve(strict=True)
snapshot=Path(sys.argv[9]).resolve(strict=True)
if root.is_symlink() or not root.is_dir(): die("exact128 output is not a plain directory")
done_record,done_raw=file_record(root/"done.json","dataset done"); done=load(done_raw,"dataset done")
if not isinstance(done,dict) or done.get("schema_version")!="motive-goku-full-motion-dataset-done-v1" or done.get("status")!="complete": die("dataset terminal receipt differs")
done_payload=dict(done); done_digest=done_payload.pop("done_digest",None)
if not isinstance(done_digest,str) or len(done_digest)!=64 or sha(canon(done_payload))!=done_digest: die("dataset done digest differs")
config=done.get("config"); counts=done.get("counts")
if not isinstance(config,dict) or config.get("exact_size")!=128 or config.get("min_multi_unit")!=32: die("dataset exact/min-multi policy differs")
if not isinstance(counts,dict) or counts.get("selected")!=128 or not isinstance(counts.get("multi_unit"),int) or counts.get("multi_unit")<32: die("dataset exact/multi counts differ")
manifest_record,manifest_raw=file_record(root/"dataset_manifest.jsonl","dataset manifest")
if len(manifest_raw.splitlines())!=128 or any(not line.strip() for line in manifest_raw.splitlines()): die("dataset manifest is not exact-128 JSONL")
artifacts=done.get("artifacts"); manifest_meta=artifacts.get("dataset_manifest.jsonl") if isinstance(artifacts,dict) else None
if not isinstance(manifest_meta,dict) or manifest_meta.get("sha256")!=manifest_record["sha256"] or manifest_meta.get("bytes")!=manifest_record["bytes"] or manifest_meta.get("rows")!=128: die("dataset manifest terminal binding differs")
dispatch_record,dispatch_raw=file_record(dispatch,"dispatcher receipt"); dispatch_value=load(dispatch_raw,"dispatcher receipt")
post_outputs=[]
for line in read(output_list,"postcheck output list").decode("utf-8").splitlines():
    output=Path(line).resolve(strict=True); output_record,_=file_record(output,"postcheck output")
    receipt=output.with_name(output.stem+".receipt.json").resolve(strict=True); receipt_record,_=file_record(receipt,"postcheck receipt")
    post_outputs.append({"output":output_record,"receipt":receipt_record})
if len(post_outputs)!=32: die("terminal receipt needs 32 postcheck outputs")
primary_record,_=file_record(primary,"generation primary"); generation_done_record,_=file_record(generation_done,"generation done")
payload={
 "schema_version":"motive-goku-full-motion-select128-controller-receipt-v1",
 "status":"complete",
 "config":{"exact_size":128,"min_multi_unit":32},
 "source_snapshot":str(snapshot),
 "generation":{"primary":primary_record,"done":generation_done_record},
 "shard_manifest_root":str(shards),
 "wan_shards_root":str(wan),
 "postcheck_dispatch":{"receipt":dispatch_record,"receipt_digest":dispatch_value.get("receipt_digest"),"outputs":post_outputs},
 "output":{"root":str(root),"dataset_manifest":manifest_record,"dataset_done":done_record,"done_digest":done_digest,"counts":counts},
 "completed_at_utc":datetime.now(timezone.utc).isoformat(),
}
receipt=dict(payload); receipt["receipt_digest"]=sha(canon(payload)); raw=canon(receipt)+b"\n"
parent=receipt_path.parent.resolve(strict=True)
if receipt_path.exists() or receipt_path.is_symlink(): die("controller receipt already exists")
descriptor,temporary_name=tempfile.mkstemp(prefix="."+receipt_path.name+".",suffix=".tmp",dir=parent)
temporary=Path(temporary_name)
try:
    with os.fdopen(descriptor,"wb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    os.chmod(temporary,0o400)
    os.link(temporary,receipt_path)
    directory=os.open(parent,os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
finally:
    if temporary.exists(): temporary.unlink()
published_raw=read(receipt_path,"published controller receipt"); published=load(published_raw,"published controller receipt")
published_payload=dict(published); published_digest=published_payload.pop("receipt_digest",None)
if published_raw!=raw or published.get("status")!="complete" or published.get("config")!={"exact_size":128,"min_multi_unit":32} or published_digest!=sha(canon(published_payload)): die("published controller receipt validation failed")'

if ! "${python_bin}" -c "${write_terminal_receipt_py}" \
  "${exact128_output}" "${controller_receipt}" \
  "${generation_primary}" "${generation_done}" \
  "${shard_manifest_root}" "${wan_shards_root}" \
  "${dispatcher_receipt}" "${postcheck_list}" "${snapshot}"; then
  fail "exact128 exists but terminal controller receipt validation/publication failed"
fi
require_plain_file "terminal controller receipt" "${controller_receipt}"
echo "[full-motion-select128-controller] complete exact=128 min_multi=32 output=${exact128_output} receipt=${controller_receipt}"

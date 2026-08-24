#!/usr/bin/env bash
# Fail-closed two-holder launcher: main on 136140/gpu215 and capacity control
# on 136141/gpu299.
# It starts one numbered child only after an explicit launch token.  Finishing
# or killing that child never releases the retained parent allocation.

set -Eeuo pipefail
umask 077

fail() { echo "[packed-preservation-v2] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly holder_job="${PRESV2_HOLDER_JOB:?set registered holder job}"
readonly holder_node="${PRESV2_HOLDER_NODE:?set registered holder node}"
readonly holder_user=guangyi.chen
readonly launch_token="${PRESV2_CONFIRM_LAUNCH:-}"
readonly execution_scope="${PRESV2_EXECUTION_SCOPE:?set optimizer-canary-2 or exact80}"
readonly lora_scope="${PRESV2_LORA_SCOPE:?set all-attention or self-attention}"
readonly method_revision="${PRESV2_METHOD_REVISION:?set exact 40-char source revision}"
readonly method_archive="${PRESV2_METHOD_ARCHIVE:?set sealed v2 method archive}"
readonly method_archive_sha="${PRESV2_METHOD_ARCHIVE_SHA256:?set sealed archive SHA-256}"
readonly method_manifest="${PRESV2_METHOD_MANIFEST:?set canonical v2 release manifest}"
readonly method_manifest_sha="${PRESV2_METHOD_MANIFEST_SHA256:?set manifest SHA-256}"
readonly run_root="${PRESV2_RUN_ROOT:?set one fresh run root}"
readonly master_port="${PRESV2_MASTER_PORT:?set one free TCP port}"

readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly source_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_preservation_recovery_20260814/runs/source-only-v3-64-16-8-e2b65a33690a-r1/source_only_split_v3.json
readonly source_manifest_sha=128064fd335c4e48c567217c6e7bae43555a904875625c9d1e21178e6f7fcc3d
readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831

case "${execution_scope}" in optimizer-canary-2|exact80) ;; *) fail "execution scope differs" ;; esac
case "${lora_scope}" in all-attention|self-attention) ;; *) fail "LoRA scope differs" ;; esac
case "${holder_job}:${holder_node}:${lora_scope}" in
  136140:auh7-1b-gpu-215:all-attention) ;;
  136141:auh7-1b-gpu-299:self-attention) ;;
  *) fail "holder/node/LoRA scope is outside the registered two-arm mapping" ;;
esac
[[ "${launch_token}" == "launch-job${holder_job}-on-${holder_node}" ]] || fail "explicit launch token is absent"
[[ "${method_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "method revision differs"
[[ "${method_archive_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "method archive SHA differs"
[[ "${method_manifest_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "method manifest SHA differs"
[[ "${master_port}" =~ ^[1-9][0-9]*$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || fail "master port differs"
[[ -z "${PRESV2_METHOD_ROOT+x}" ]] || fail "caller-supplied method root is forbidden"
for path_name in run_root method_archive method_manifest; do
  path_value="${!path_name}"
  [[ "${path_value}" == /vast/users/guangyi.chen/* && "${path_value}" != / ]] || fail "${path_name} path differs"
done
[[ -f "${method_archive}" && ! -L "${method_archive}" ]] || fail "sealed method archive differs"
[[ "$(sha256_file "${method_archive}")" == "${method_archive_sha}" ]] || fail "sealed method archive SHA differs"
[[ -f "${method_manifest}" && ! -L "${method_manifest}" ]] || fail "method manifest differs"
[[ "$(sha256_file "${method_manifest}")" == "${method_manifest_sha}" ]] || fail "method manifest SHA differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
for input_path in "${checkpoint_manifest}" "${source_manifest}"; do
  [[ -f "${input_path}" && ! -L "${input_path}" ]] || fail "pinned input differs: ${input_path}"
done
[[ "$(sha256_file "${checkpoint_manifest}")" == "${checkpoint_manifest_sha}" ]] || fail "checkpoint manifest SHA differs"
[[ "$(sha256_file "${source_manifest}")" == "${source_manifest_sha}" ]] || fail "source manifest SHA differs"
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh"
if [[ "${execution_scope}" == exact80 && "${run_root,,}" == *canary* ]]; then
  fail "exact80 must use a fresh non-canary output root"
fi

job_record="$(scontrol show job -o "${holder_job}")"
[[ "${job_record}" == *"JobId=${holder_job} "* && "${job_record}" == *"JobState=RUNNING"* ]] || fail "holder is not RUNNING"
[[ "${job_record}" == *"UserId=${holder_user}"* && "${job_record}" == *"NodeList=${holder_node}"* ]] || fail "holder owner/node differs"
[[ -z "$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')" ]] || fail "holder already has a numbered child"
[[ -z "$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" "ss -H -ltn 'sport = :${master_port}'")" ]] || fail "master port is occupied"

assert_gpu_idle() {
  local snapshot count busy kfd_holders
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showuse --showmemuse --showpids')"
  count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if ((v+0)!=0) print}' <<<"${snapshot}")"
  kfd_holders="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'fuser /dev/kfd 2>/dev/null || true')"
  [[ "${count}" == 8 && -z "${busy}" && -z "${kfd_holders}" ]] || fail "exact8 GPU/KFD inventory is not idle"
}
assert_gpu_idle
sleep 2
assert_gpu_idle

mkdir -m 0700 "${run_root}" "${run_root}/logs"
readonly release_extract_root="${run_root}/runtime-source"

# Bootstrap from bytes, never from a caller-selected source tree.  This is an
# intentionally self-contained stdlib verifier: importing any release member
# before this check would let archive A be paired with executable root B.
method_root="$("${python_bin}" -I -S - \
  "${method_archive}" "${method_archive_sha}" \
  "${method_manifest}" "${method_manifest_sha}" \
  "${method_revision}" "${release_extract_root}" <<'PY'
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile


SCHEMA_VERSION = "bernini-packed-preservation-release-v2"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-exact-modes-v2"
MEMBER_ROOT = "methods/bernini_action_editing"
FILES_AND_MODES = {
    "packed_preservation_lora_v2.py": 0o444,
    "packed_preservation_release_v2.py": 0o444,
    "train_packed_preservation_lora_v2.py": 0o444,
    "clean_source_visual_context_stage_b_contract_v1.py": 0o444,
    "clean_source_visual_context_training_v1.py": 0o444,
    "inference_sigma_strata.py": 0o444,
    "source_self_runtime.py": 0o444,
    "train_lora.py": 0o444,
    "scripts/auh_packed_preservation_rank_exec_v2.sh": 0o555,
    "scripts/auh_train_packed_preservation_lora_v2_job136140.sh": 0o555,
}


def fail(message):
    raise SystemExit(f"packed-preservation bootstrap: {message}")


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def plain_file(raw, label):
    requested = Path(raw).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} is not an absolute plain file")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not stat.S_ISREG(resolved.lstat().st_mode):
        fail(f"{label} is not canonical")
    return resolved


archive_path = plain_file(sys.argv[1], "archive")
expected_archive_sha = sys.argv[2]
manifest_path = plain_file(sys.argv[3], "manifest")
expected_manifest_sha = sys.argv[4]
expected_revision = sys.argv[5]
extract_root = Path(sys.argv[6]).expanduser()
if re.fullmatch(r"[0-9a-f]{64}", expected_archive_sha) is None:
    fail("archive SHA identity differs")
if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha) is None:
    fail("manifest SHA identity differs")
if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
    fail("content revision identity differs")
if (
    not extract_root.is_absolute()
    or extract_root.exists()
    or extract_root.is_symlink()
    or extract_root.parent.resolve(strict=True) != extract_root.parent
):
    fail("release extraction root is not one fresh canonical path")

archive_bytes = archive_path.read_bytes()
manifest_bytes = manifest_path.read_bytes()
if hashlib.sha256(archive_bytes).hexdigest() != expected_archive_sha:
    fail("archive SHA differs")
if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha:
    fail("manifest SHA differs")
try:
    manifest = json.loads(manifest_bytes.decode("ascii"))
except (UnicodeError, json.JSONDecodeError) as error:
    fail(f"manifest cannot be decoded: {error}")
if not isinstance(manifest, dict) or manifest_bytes != canonical_json(manifest) + b"\n":
    fail("manifest is not canonical JSON")
required = {
    "schema_version", "archive_format", "member_root", "revision_kind",
    "method_revision", "archive_sha256", "exact_member_closure",
    "file_count", "files", "manifest_digest",
}
unsigned = dict(manifest)
declared_digest = unsigned.pop("manifest_digest", None)
rows = manifest.get("files")
if (
    set(manifest) != required
    or manifest.get("schema_version") != SCHEMA_VERSION
    or manifest.get("archive_format") != ARCHIVE_FORMAT
    or manifest.get("member_root") != MEMBER_ROOT
    or manifest.get("revision_kind") != "content-closure-sha1"
    or manifest.get("method_revision") != expected_revision
    or manifest.get("archive_sha256") != expected_archive_sha
    or manifest.get("exact_member_closure") is not True
    or manifest.get("file_count") != len(FILES_AND_MODES)
    or not isinstance(rows, list)
    or declared_digest != hashlib.sha256(canonical_json(unsigned)).hexdigest()
):
    fail("manifest schema or digest differs")
expected_paths = sorted(FILES_AND_MODES)
if [row.get("path") if isinstance(row, dict) else None for row in rows] != expected_paths:
    fail("manifest exact file order or set differs")
for row, relative in zip(rows, expected_paths):
    if (
        set(row) != {"path", "mode", "size", "sha256"}
        or row["mode"] != FILES_AND_MODES[relative]
        or not isinstance(row["size"], int)
        or isinstance(row["size"], bool)
        or row["size"] < 0
        or not isinstance(row["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
    ):
        fail(f"manifest row differs: {relative}")
revision_projection = {
    "schema_version": SCHEMA_VERSION,
    "member_root": MEMBER_ROOT,
    "files": rows,
}
if hashlib.sha1(canonical_json(revision_projection)).hexdigest() != expected_revision:
    fail("content revision differs from canonical file rows")

payloads = {}
with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as bundle:
    members = bundle.getmembers()
    expected_names = [f"{MEMBER_ROOT}/{relative}" for relative in expected_paths]
    if [member.name for member in members] != expected_names:
        fail("archive exact member order or set differs")
    for member, row, relative in zip(members, rows, expected_paths):
        if (
            not member.isfile()
            or member.mode != row["mode"]
            or member.uid != 0
            or member.gid != 0
            or member.uname != ""
            or member.gname != ""
            or member.mtime != 0
            or member.size != row["size"]
            or member.pax_headers
        ):
            fail(f"archive metadata differs: {relative}")
        handle = bundle.extractfile(member)
        payload = b"" if handle is None else handle.read()
        if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            fail(f"archive bytes differ: {relative}")
        payloads[relative] = payload

extract_root.mkdir(mode=0o700)
method_root = extract_root / MEMBER_ROOT
(method_root / "scripts").mkdir(parents=True, mode=0o700)
for row in rows:
    relative = row["path"]
    destination = method_root / relative
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        row["mode"],
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payloads[relative])
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    if not stat.S_ISREG(destination.lstat().st_mode):
        fail(f"materialized entry is not a plain file: {relative}")
    os.chmod(destination, row["mode"])
    if hashlib.sha256(destination.read_bytes()).hexdigest() != row["sha256"]:
        fail(f"materialized bytes differ: {relative}")
    if stat.S_IMODE(destination.stat().st_mode) != row["mode"]:
        fail(f"materialized mode differs: {relative}")
for directory in (method_root / "scripts", method_root, method_root.parent, extract_root):
    if not stat.S_ISDIR(directory.lstat().st_mode):
        fail(f"materialized directory differs: {directory}")
    os.chmod(directory, 0o555)
print(method_root)
PY
)"
readonly method_root
readonly expected_derived_method_root="${release_extract_root}/methods/bernini_action_editing"
[[ "${method_root}" == "${expected_derived_method_root}" ]] || fail "derived method root differs"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "materialized method root differs"
readonly rank_exec="${method_root}/scripts/auh_packed_preservation_rank_exec_v2.sh"
[[ -f "${rank_exec}" && ! -L "${rank_exec}" && -x "${rank_exec}" ]] || fail "materialized rank wrapper differs"
readonly training_output="${run_root}/training"

set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --cpus-per-task=64 --mem=64G --gres=gpu:mi210:8 \
  env BERNINI_PRESV2_RANK_CACHE_TOKEN="presv2-${lora_scope}-${method_revision:0:10}" \
    BERNINI_PRESV2_PYTHON_BIN="${python_bin}" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="${method_root}" \
    "${python_bin}" -B -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
      --master_addr=127.0.0.1 --master_port="${master_port}" --no_python "${rank_exec}" \
      "${method_root}/train_packed_preservation_lora_v2.py" \
      --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --source-only-manifest "${source_manifest}" \
      --expected-source-only-manifest-sha256 "${source_manifest_sha}" \
      --output "${training_output}" --execution-scope "${execution_scope}" \
      --lora-scope "${lora_scope}" --method-source-revision "${method_revision}" \
      --method-source-archive "${method_archive}" \
      --expected-method-source-archive-sha256 "${method_archive_sha}" \
      --method-source-manifest "${method_manifest}" \
      --expected-method-source-manifest-sha256 "${method_manifest_sha}" \
      --ack-source-release-is-exploratory --ack-fresh-base-not-canary-resume \
  >"${run_root}/logs/train.log" 2>&1
status=$?
set -e

printf 'holder_job=%s\nholder_node=%s\nexecution_scope=%s\nlora_scope=%s\nchild_exit=%s\nparent_not_released=true\n' \
  "${holder_job}" "${holder_node}" "${execution_scope}" "${lora_scope}" "${status}" \
  >"${run_root}/controller.status"
if (( status != 0 )); then
  tail -n 240 "${run_root}/logs/train.log" >&2 || true
  exit "${status}"
fi
[[ -f "${training_output}/receipt.json" ]] || fail "training receipt is missing"
if [[ "${execution_scope}" == optimizer-canary-2 ]]; then
  required=(00000000 00000001 00000002)
else
  required=(00000000 00000020 00000040 00000060 00000080)
fi
for step in "${required[@]}"; do
  root="${training_output}/checkpoints/checkpoint-${step}"
  [[ -f "${root}/adapter.pt" && -f "${root}/optimizer.pt" && -f "${root}/metadata.json" ]] || fail "checkpoint ${step} differs"
done
printf 'TRAINING_COMPLETE_PRESV2 job=%s node=%s scope=%s lora=%s parent_retained=true\n' \
  "${holder_job}" "${holder_node}" "${execution_scope}" "${lora_scope}" \
  >"${run_root}/controller.TRAINING_COMPLETE"
echo "TRAINING_COMPLETE_PRESV2 output=${run_root} parent_retained=true"

#!/usr/bin/env bash
# Decode one immutable packed-preservation checkpoint on an exact retained holder.
# The script never cancels, requeues, releases, or signals the parent holder.

set -Eeuo pipefail
umask 077

fail() { echo "[presv2-review-136309] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly holder_user=guangyi.chen
readonly holder="${PRESV2_REVIEW_HOLDER:?set one exact allowlisted job@node}"
case "${holder}" in
  136140@auh7-1b-gpu-215)
    readonly holder_job=136140 holder_node=auh7-1b-gpu-215 holder_lane=all-attention ;;
  136309@auh7-1b-gpu-280)
    readonly holder_job=136309 holder_node=auh7-1b-gpu-280 holder_lane=neutral ;;
  136141@auh7-1b-gpu-299)
    readonly holder_job=136141 holder_node=auh7-1b-gpu-299 holder_lane=self-attention ;;
  135096@auh7-1b-gpu-246)
    readonly holder_job=135096 holder_node=auh7-1b-gpu-246 holder_lane=self-attention ;;
  *) fail "holder is outside the immutable decode allowlist" ;;
esac
readonly launch_token="${PRESV2_REVIEW_CONFIRM_LAUNCH:-}"
readonly method_archive="${PRESV2_REVIEW_METHOD_ARCHIVE:?set sealed review archive}"
readonly method_archive_sha="${PRESV2_REVIEW_METHOD_ARCHIVE_SHA256:?set review archive SHA}"
readonly method_manifest="${PRESV2_REVIEW_METHOD_MANIFEST:?set canonical review release manifest}"
readonly method_manifest_sha="${PRESV2_REVIEW_METHOD_MANIFEST_SHA256:?set review manifest SHA}"
readonly training_run="${PRESV2_REVIEW_TRAINING_RUN:?set packed-preservation training run}"
readonly training_receipt="${PRESV2_REVIEW_TRAINING_RECEIPT:-}"
readonly training_receipt_sha="${PRESV2_REVIEW_TRAINING_RECEIPT_SHA256:-}"
readonly review_manifest="${PRESV2_REVIEW_MANIFEST:?set fixed heldout review manifest}"
readonly review_manifest_sha="${PRESV2_REVIEW_MANIFEST_SHA256:?set review manifest SHA}"
readonly execution_scope="${PRESV2_REVIEW_EXECUTION_SCOPE:?set exact80 or optimizer-canary-2}"
readonly checkpoint_step="${PRESV2_REVIEW_CHECKPOINT_STEP:?set checkpoint step}"
readonly smoke_sentinel="${PRESV2_REVIEW_SMOKE_SENTINEL:-}"
readonly lora_scope="${PRESV2_REVIEW_LORA_SCOPE:?set LoRA scope}"
readonly output_root="${PRESV2_REVIEW_OUTPUT_ROOT:?set one fresh controller root}"
readonly runtime_revision="${PRESV2_REVIEW_RUNTIME_REVISION:?set exact runtime revision}"
readonly runtime_archive_sha="${PRESV2_REVIEW_RUNTIME_ARCHIVE_SHA256:?set runtime archive SHA}"
readonly launcher_sha="${PRESV2_REVIEW_LAUNCHER_SHA256:?set this launcher SHA}"
readonly master_port="${PRESV2_REVIEW_MASTER_PORT:?set one free port}"
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6
readonly base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly launcher_source="$(readlink -f -- "${BASH_SOURCE[0]}")"

[[ "${launch_token}" == "launch-job${holder_job}-on-${holder_node}" ]] || fail "explicit launch token is absent"
[[ "${holder_lane}" == neutral || "${lora_scope}" == "${holder_lane}" ]] || fail "holder/lane binding differs"
[[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision differs"
case "${execution_scope}" in exact80|optimizer-canary-2) ;; *) fail "execution scope differs" ;; esac
if [[ "${execution_scope}" == optimizer-canary-2 ]]; then
  [[ "${checkpoint_step}" == 2 && -n "${smoke_sentinel}" ]] || fail "smoke is restricted to P2 and one sentinel"
  [[ -z "${training_receipt}" && -z "${training_receipt_sha}" ]] || fail "smoke cannot claim a terminal exact80 receipt"
else
  case "${checkpoint_step}" in 0|20|40|60|80) ;; *) fail "formal checkpoint cadence differs" ;; esac
  [[ -z "${smoke_sentinel}" ]] || fail "formal decode cannot select one sentinel"
  [[ -n "${training_receipt}" && -n "${training_receipt_sha}" ]] || fail "formal decode requires the terminal exact80 receipt"
fi
for digest in review_manifest_sha method_archive_sha method_manifest_sha runtime_archive_sha launcher_sha; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"
done
if [[ "${execution_scope}" == exact80 ]]; then
  [[ "${training_receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "training receipt SHA differs"
fi
[[ "${master_port}" =~ ^[1-9][0-9]*$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || fail "master port differs"
[[ -z "${PRESV2_REVIEW_METHOD_ROOT+x}" && -z "${PRESV2_REVIEW_DEPENDENCY_ROOT+x}" ]] || fail "caller-supplied executable roots are forbidden"
for name in method_archive method_manifest training_run output_root; do
  value="${!name}"
  [[ "${value}" == /vast/users/guangyi.chen/* && "${value}" != / ]] || fail "${name} path differs"
done
[[ -d "${training_run}" && ! -L "${training_run}" && "$(readlink -f -- "${training_run}")" == "${training_run}" ]] || fail "training run differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
for path in "${method_archive}" "${method_manifest}" "${review_manifest}" "${checkpoint_manifest}" "${launcher_source}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed input differs: ${path}"
done
[[ "$(sha256_file "${method_archive}")" == "${method_archive_sha}" ]] || fail "method archive SHA differs"
[[ "${method_archive_sha}" == "${runtime_archive_sha}" ]] || fail "runtime/archive expected identities differ"
[[ "$(sha256_file "${method_manifest}")" == "${method_manifest_sha}" ]] || fail "method manifest SHA differs"
if [[ "${execution_scope}" == exact80 ]]; then
  [[ -f "${training_receipt}" && ! -L "${training_receipt}" && "$(readlink -f -- "${training_receipt}")" == "${training_receipt}" ]] || fail "terminal training receipt differs"
  [[ "$(dirname -- "${training_receipt}")" == "${training_run}" ]] || fail "terminal receipt is outside the exact training run"
  [[ "$(basename -- "${training_receipt}")" == receipt.json ]] || fail "terminal receipt filename differs"
  [[ "$(sha256_file "${training_receipt}")" == "${training_receipt_sha}" ]] || fail "terminal receipt SHA differs"
fi
[[ "$(sha256_file "${review_manifest}")" == "${review_manifest_sha}" ]] || fail "review manifest SHA differs"
[[ "$(sha256_file "${checkpoint_manifest}")" == "${checkpoint_manifest_sha}" ]] || fail "checkpoint manifest SHA differs"
[[ "$(sha256_file "${launcher_source}")" == "${launcher_sha}" ]] || fail "launcher SHA differs"

assert_parent() {
  local record allocated_nodes target_present node_child
  record="$(scontrol show job -o "${holder_job}")"
  [[ "${record}" == *"JobId=${holder_job} "* && "${record}" == *"JobState=RUNNING"* ]] || fail "holder is not RUNNING"
  [[ "${record}" == *"UserId=${holder_user}"* ]] || fail "holder owner differs"
  allocated_nodes="$(squeue -j "${holder_job}" -h -o '%N')"
  target_present="$(scontrol show hostnames "${allocated_nodes}" | awk -v target="${holder_node}" '$0 == target {print}')"
  [[ "${target_present}" == "${holder_node}" ]] || fail "target node is outside holder allocation"
  node_child="$(squeue -s -j "${holder_job}" -h -o '%i|%N' | awk -F'|' -v target="${holder_node}" '$1 ~ /[.][0-9]+$/ && $2 == target {print}')"
  [[ -z "${node_child}" ]] || fail "target node already has a numbered child"
}
assert_gpu_kfd_idle() {
  local snapshot count busy kfd_holders
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showuse --showmemuse --showpids')"
  count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if ((v+0)!=0) print}' <<<"${snapshot}")"
  kfd_holders="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'fuser /dev/kfd 2>/dev/null || true')"
  [[ "${count}" == 8 && -z "${busy}" && -z "${kfd_holders}" ]] || fail "exact8 GPU/KFD inventory is not idle"
}
assert_parent
assert_gpu_kfd_idle
sleep 2
assert_parent
assert_gpu_kfd_idle
[[ -z "$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" "ss -H -ltn 'sport = :${master_port}'")" ]] || fail "master port is occupied"

mkdir -m 0700 "${output_root}" "${output_root}/logs"
readonly release_extract_root="${output_root}/runtime-source"

# Verify canonical metadata and every ustar member with isolated stdlib Python,
# then write the only executable root.  No code from the release is imported
# and no caller-selected method/dependency directory is reachable beforehand.
method_root="$("${python_bin}" -I -S - \
  "${method_archive}" "${method_archive_sha}" \
  "${method_manifest}" "${method_manifest_sha}" \
  "${runtime_revision}" "${release_extract_root}" <<'PY'
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile

SCHEMA = "bernini-packed-preservation-checkpoint-review-release-v2"
FORMAT = "ustar-owner0-mtime0-exact-modes-v2"
ROOT = "methods/bernini_action_editing"
FILES = {
    "clean_source_visual_context_adapter_v1.py": 0o444,
    "clean_source_visual_context_checkpoint_review_contract_v1.py": 0o444,
    "clean_source_visual_context_training_v1.py": 0o444,
    "infer_lora.py": 0o444,
    "infer_native_identity_generation_canary.py": 0o444,
    "infer_native_v_axis_exact81_probe_v1.py": 0o444,
    "infer_orderless_source_frame_set_noise_canary.py": 0o444,
    "infer_packed_preservation_checkpoint_review_v2.py": 0o444,
    "infer_source_kv_carrier_oracle.py": 0o444,
    "infer_source_value_residual_oracle.py": 0o444,
    "native_i_axis_guidance.py": 0o444,
    "native_v_axis_guidance_v1.py": 0o444,
    "orderless_source_frame_set_noise.py": 0o444,
    "packed_preservation_checkpoint_review_release_v2.py": 0o444,
    "packed_preservation_checkpoint_review_v2.py": 0o444,
    "packed_preservation_lora_v2.py": 0o444,
    "source_kv_replay.py": 0o444,
    "source_kv_route_batches.py": 0o444,
    "source_value_residual.py": 0o444,
    "train_lora.py": 0o444,
    "tri_branch_unipc.py": 0o444,
    "tools/build_renderer_dataset.py": 0o444,
    "tools/materialize_vae.py": 0o444,
    "scripts/auh_decode_packed_preservation_checkpoint_v2_job136309.sh": 0o555,
    "scripts/auh_packed_preservation_review_rank_exec_v2.sh": 0o555,
}

def fail(message):
    raise SystemExit(f"presv2-review bootstrap: {message}")

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

def plain(raw, label):
    path = Path(raw).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} is not an absolute plain file")
    resolved = path.resolve(strict=True)
    if resolved != path or not stat.S_ISREG(path.lstat().st_mode):
        fail(f"{label} is not canonical")
    return path

archive = plain(sys.argv[1], "archive")
archive_sha = sys.argv[2]
manifest_path = plain(sys.argv[3], "manifest")
manifest_sha = sys.argv[4]
revision = sys.argv[5]
destination = Path(sys.argv[6]).expanduser()
if re.fullmatch(r"[0-9a-f]{64}", archive_sha) is None or re.fullmatch(r"[0-9a-f]{64}", manifest_sha) is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    fail("expected identities differ")
if not destination.is_absolute() or destination.exists() or destination.is_symlink() or destination.parent.resolve(strict=True) != destination.parent:
    fail("destination is not one fresh canonical path")
archive_bytes = archive.read_bytes()
manifest_bytes = manifest_path.read_bytes()
if hashlib.sha256(archive_bytes).hexdigest() != archive_sha or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha:
    fail("sealed release bytes differ")
try:
    manifest = json.loads(manifest_bytes.decode("ascii"))
except (UnicodeError, json.JSONDecodeError) as error:
    fail(f"manifest decode failed: {error}")
required = {
    "schema_version", "archive_format", "archive_file", "member_root",
    "revision_kind", "method_revision", "archive_sha256",
    "exact_member_closure", "executed_root_required", "runner_member",
    "launcher_member", "file_count", "files", "manifest_digest",
}
unsigned = dict(manifest) if isinstance(manifest, dict) else {}
declared = unsigned.pop("manifest_digest", None)
rows = manifest.get("files") if isinstance(manifest, dict) else None
if (
    not isinstance(manifest, dict) or manifest_bytes != canonical(manifest) + b"\n"
    or set(manifest) != required or manifest.get("schema_version") != SCHEMA
    or manifest.get("archive_format") != FORMAT or manifest.get("archive_file") != "method.tar"
    or manifest.get("member_root") != ROOT or manifest.get("revision_kind") != "content-closure-sha1"
    or manifest.get("method_revision") != revision or manifest.get("archive_sha256") != archive_sha
    or manifest.get("exact_member_closure") is not True or manifest.get("executed_root_required") is not True
    or manifest.get("runner_member") != "infer_packed_preservation_checkpoint_review_v2.py"
    or manifest.get("launcher_member") != "scripts/auh_decode_packed_preservation_checkpoint_v2_job136309.sh"
    or manifest.get("file_count") != len(FILES) or not isinstance(rows, list)
    or declared != hashlib.sha256(canonical(unsigned)).hexdigest()
):
    fail("manifest schema/digest differs")
paths = sorted(FILES)
if [row.get("path") if isinstance(row, dict) else None for row in rows] != paths:
    fail("manifest exact member order/set differs")
for row, relative in zip(rows, paths):
    if set(row) != {"path", "mode", "size", "sha256"} or row["mode"] != FILES[relative] or not isinstance(row["size"], int) or isinstance(row["size"], bool) or row["size"] < 0 or re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is None:
        fail(f"manifest row differs: {relative}")
projection = {"schema_version": SCHEMA, "member_root": ROOT, "files": rows}
if hashlib.sha1(canonical(projection)).hexdigest() != revision:
    fail("content revision differs")
payloads = {}
with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as bundle:
    members = bundle.getmembers()
    if [member.name for member in members] != [f"{ROOT}/{path}" for path in paths]:
        fail("archive exact member order/set differs")
    for member, row, relative in zip(members, rows, paths):
        if not member.isfile() or member.mode != row["mode"] or member.uid != 0 or member.gid != 0 or member.uname != "" or member.gname != "" or member.mtime != 0 or member.size != row["size"] or member.pax_headers:
            fail(f"archive metadata differs: {relative}")
        handle = bundle.extractfile(member)
        payload = b"" if handle is None else handle.read()
        if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            fail(f"archive bytes differ: {relative}")
        payloads[relative] = payload
destination.mkdir(mode=0o700)
method = destination / ROOT
for parent in sorted({(method / relative).parent for relative in paths}, key=lambda item: len(item.parts)):
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
for row in rows:
    target = method / row["path"]
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), row["mode"])
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payloads[row["path"]])
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(target, row["mode"])
    if not stat.S_ISREG(target.lstat().st_mode) or hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"] or stat.S_IMODE(target.stat().st_mode) != row["mode"]:
        fail(f"materialized member differs: {row['path']}")
for name, payload in (("method.tar", archive_bytes), ("manifest.json", manifest_bytes)):
    target = destination / name
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(target, 0o444)
for directory in sorted((path for path in destination.rglob("*") if path.is_dir()), key=lambda item: len(item.parts), reverse=True):
    os.chmod(directory, 0o555)
os.chmod(destination, 0o555)
print(method)
PY
)"
readonly method_root
readonly expected_method_root="${release_extract_root}/methods/bernini_action_editing"
[[ "${method_root}" == "${expected_method_root}" && -d "${method_root}" && ! -L "${method_root}" ]] || fail "derived method root differs"
readonly runner="${method_root}/infer_packed_preservation_checkpoint_review_v2.py"
readonly rank_exec="${method_root}/scripts/auh_packed_preservation_review_rank_exec_v2.sh"
readonly runtime_launcher="${method_root}/scripts/auh_decode_packed_preservation_checkpoint_v2_job136309.sh"
[[ -f "${runner}" && ! -L "${runner}" && -f "${rank_exec}" && ! -L "${rank_exec}" && -x "${rank_exec}" ]] || fail "materialized review runtime differs"
[[ -f "${runtime_launcher}" && ! -L "${runtime_launcher}" && "$(sha256_file "${runtime_launcher}")" == "${launcher_sha}" ]] || fail "executed bootstrap launcher differs from release launcher"
touch "${output_root}/adapter-load.lock"
chmod 0400 "${output_root}/adapter-load.lock"
readonly shard="${output_root}/shard"
smoke_args=()
terminal_args=()
if [[ "${execution_scope}" == optimizer-canary-2 ]]; then
  smoke_args=(--smoke-sentinel "${smoke_sentinel}")
else
  terminal_args=(--training-receipt "${training_receipt}" --expected-training-receipt-sha256 "${training_receipt_sha}")
fi
set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --cpus-per-task=32 --mem=64G --gres=gpu:mi210:4 \
  env ROCR_VISIBLE_DEVICES=0,1,2,3 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="${method_root}" \
    NATIVE_V_AXIS_LOAD_LOCK="${output_root}/adapter-load.lock" \
    PACKED_PRESERVATION_REVIEW_LOAD_LOCK="${output_root}/adapter-load.lock" \
    PACKED_PRESERVATION_REVIEW_CACHE_TOKEN="presv2-${execution_scope}-${checkpoint_step}" \
    PACKED_PRESERVATION_REVIEW_PYTHON_BIN="${python_bin}" \
    "${python_bin}" -B -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
      --master_addr=127.0.0.1 --master_port="${master_port}" --no_python "${rank_exec}" \
      "${runner}" --review-manifest "${review_manifest}" \
      --expected-review-manifest-sha256 "${review_manifest_sha}" \
      --training-run "${training_run}" --checkpoint-step "${checkpoint_step}" \
      "${terminal_args[@]}" --execution-scope "${execution_scope}" "${smoke_args[@]}" --lora-scope "${lora_scope}" \
      --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --base-checkpoint "${base_checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --runtime-source-manifest "${release_extract_root}/manifest.json" \
      --expected-runtime-source-manifest-sha256 "${method_manifest_sha}" \
      --runtime-source-launcher "${runtime_launcher}" \
      --runtime-source-revision "${runtime_revision}" \
      --runtime-source-archive-sha256 "${runtime_archive_sha}" \
      --launcher-source-sha256 "${launcher_sha}" --output-dir "${shard}" \
  >"${output_root}/logs/decode.log" 2>&1
child_status=$?
set -e
assert_parent
[[ "${child_status}" == 0 && -f "${shard}/receipt.json" && ! -L "${shard}/receipt.json" ]] || fail "decode child failed: ${child_status}"
if [[ "${execution_scope}" == optimizer-canary-2 ]]; then
  printf 'SMOKE_ONLY_P2_LOADER_DECODE_COMPLETE parent_retained=true scientific_claim=false\n' >"${output_root}/SMOKE.COMPLETE"
else
  printf 'FORMAL_CHECKPOINT_SHARD_COMPLETE step=%s parent_retained=true\n' "${checkpoint_step}" >"${output_root}/SHARD.COMPLETE"
fi
printf 'child_status=0\nparent_job=%s\nparent_node=%s\nparent_not_released=true\n' "${holder_job}" "${holder_node}" >"${output_root}/controller.status"
echo "PACKED_PRESERVATION_REVIEW_COMPLETE scope=${execution_scope} step=${checkpoint_step} output=${shard} parent_retained=true"

#!/usr/bin/env bash
# Run one exact ELAL-3 C1 simulator oracle-q diagnostic inside a held AUH node.
# Scope: actual-shape no-update preflight -> fresh 1-step smoke -> fresh 10-step
# overfit.  This never authorizes formal C1, exact160, source+instruction
# inference, real-video/generalization, production, or scientific claims.

set -Eeuo pipefail
umask 0027

fail() {
  echo "[elal3-c1-oracle-diagnostic] ERROR: $*" >&2
  exit 2
}

# These exact literals bind this launcher to the independently reviewed trainer
# and the deterministic source release produced from it.
expected_archive_sha256="631611a96a744025eb6e5b223958908c7dfccfb69bfaefa7432ea9c20afc8194"
expected_manifest_sha256="bb56f175f205b626f003c855260243a5c1a5fa3d8c7f0464ddea49931006a9f3"
expected_runner_sha256="521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3"
trainer_ack_flags=(
  --ack-simulator-oracle-q-overfit-only
  --ack-not-source-instruction-inference
  --ack-not-formal-c1
  --ack-not-exact160
  --ack-no-scientific-claim
)

sha_re='^[0-9a-f]{64}$'
[[ "${expected_archive_sha256}" =~ ${sha_re} ]] || fail "release archive pin is still PENDING"
[[ "${expected_manifest_sha256}" =~ ${sha_re} ]] || fail "release manifest pin is still PENDING"
[[ "${expected_runner_sha256}" =~ ${sha_re} ]] || fail "trainer source pin is still PENDING"

[[ -n "${ELAL3_C1_SOURCE_ARCHIVE:-}" ]] || fail "ELAL3_C1_SOURCE_ARCHIVE is required"
[[ -n "${ELAL3_C1_SOURCE_MANIFEST:-}" ]] || fail "ELAL3_C1_SOURCE_MANIFEST is required"
[[ -n "${ELAL3_C1_OUTPUT_ROOT:-}" ]] || fail "a fresh ELAL3_C1_OUTPUT_ROOT is required"
archive="${ELAL3_C1_SOURCE_ARCHIVE}"
manifest="${ELAL3_C1_SOURCE_MANIFEST}"
output_root="${ELAL3_C1_OUTPUT_ROOT}"

python_bin="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
bernini_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591"
veomni_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
checkpoint_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
experiment_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/elal3_c1_oracle_diagnostic_preflight_20260817_v1"
packet_root="${experiment_root}/simulator_gt_canary_v1"
latent_bundle="${experiment_root}/vae-c1-row-modelbound-v2/c1-latents.safetensors"

derivative_authority_sha256="298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b"
model_authority_sha256="4c2f4d28af646ab39bdeb775e1b651d523d83b3fc0b8e5c1dd4bc78fbd4f25ed"
latent_receipt_sha256="a400d11d0d1337daa61d74a25e040aab27b83cc75e62038b81b83f56075e4fcb"
latent_bundle_sha256="8fbd27abf7b6eea0593b236a0594dcfad38b3bedf46cf42e77391ec5648fdedf"
packet_manifest_sha256="2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"
train_lora_source_sha256="630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5"
elal3_core_source_sha256="70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862"
elal3_label_source_sha256="4fecea53a55376545614edfca8603184f5e6f91dc86baccf6fb980f8b8124aa2"
packed_lora_source_sha256="61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6"
runtime_source_sha256="62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f"
sigma_source_sha256="e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3"

[[ "${archive}" == /* && "${manifest}" == /* && "${output_root}" == /* ]] || fail "release/output paths must be absolute"
[[ "${output_root}" != / && "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || fail "unsafe output root"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
for path in "${archive}" "${manifest}" "${python_bin}" "${latent_bundle}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "required plain file unavailable: ${path}"
done
for path in "${bernini_root}" "${veomni_root}" "${checkpoint_root}" "${packet_root}"; do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "required canonical directory unavailable: ${path}"
done
[[ -x "${python_bin}" ]] || fail "pinned Python is not executable"
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${expected_archive_sha256}" ]] || fail "release archive SHA-256 differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${expected_manifest_sha256}" ]] || fail "release manifest SHA-256 differs"
[[ "$(sha256sum "${latent_bundle}" | awk '{print $1}')" == "${latent_bundle_sha256}" ]] || fail "model-bound latent bundle SHA-256 differs"
[[ "$(stat -c '%s:%a:%h' "${latent_bundle}")" == "39138208:444:1" ]] || fail "model-bound latent bundle size/mode/link differs"
[[ "$(sha256sum "${packet_root}/manifest.json" | awk '{print $1}')" == "${packet_manifest_sha256}" ]] || fail "simulator packet manifest SHA-256 differs"

job_id="${SLURM_JOB_ID:-}"
node="$(hostname -s)"
case "${job_id}:${node}" in
  141620:auh7-1b-gpu-226)
    seed=20260817
    arm="main-full-w64"
    master_port=29620
    ;;
  141618:auh7-1b-gpu-249)
    seed=20260818
    arm="replicate-full-w64-seed2"
    master_port=29618
    ;;
  141619:auh7-1b-gpu-257)
    seed=20260819
    arm="replicate-full-w64-seed3"
    master_port=29619
    ;;
  *) fail "unregistered holder job/node pair: ${job_id:-unset}:${node}" ;;
esac

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
scratch="$(mktemp -d "${scratch_parent%/}/elal3-c1-oracle.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  case "${scratch}" in
    "${scratch_parent%/}/elal3-c1-oracle."*) ;;
    *) echo "[elal3-c1-oracle-diagnostic] refusing unsafe scratch cleanup" >&2; exit 2 ;;
  esac
  if [[ -d "${scratch}" && ! -L "${scratch}" ]]; then
    chmod -R u+w -- "${scratch}" || status=2
    rm -rf -- "${scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p -- "$(dirname -- "${output_root}")"
mkdir -- "${output_root}"
chmod 0700 -- "${output_root}"
mkdir -- "${scratch}/source"

# torchrun starts this exact wrapper once per rank.  MIOpen's sqlite/cache and
# Torch/Triton compilation caches must never be shared through a read-only home
# DB (node257 already demonstrated that failure mode).
rank_wrapper="${scratch}/rank_exec.sh"
"${python_bin}" -I -B - "${rank_wrapper}" <<'PY'
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
raw = b'''#!/usr/bin/env bash
set -Eeuo pipefail
umask 0077
cache_base="${1:?rank cache base missing}"
shift
rank="${LOCAL_RANK:?LOCAL_RANK missing}"
[[ "${rank}" =~ ^[0-7]$ ]] || { echo "invalid LOCAL_RANK=${rank}" >&2; exit 2; }
rank_root="${cache_base}/rank_${rank}"
[[ ! -e "${rank_root}" && ! -L "${rank_root}" ]] || { echo "rank cache exists: ${rank_root}" >&2; exit 2; }
mkdir -m 0700 -- "${rank_root}" \
  "${rank_root}/miopen-user" "${rank_root}/miopen-custom" \
  "${rank_root}/torch" "${rank_root}/xdg" "${rank_root}/triton"
export MIOPEN_USER_DB_PATH="${rank_root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${rank_root}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${rank_root}/torch"
export XDG_CACHE_HOME="${rank_root}/xdg"
export TRITON_CACHE_DIR="${rank_root}/triton"
exec "$@"
'''
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o500)
try:
    remaining = memoryview(raw)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise SystemExit("rank wrapper write made no progress")
        remaining = remaining[written:]
    os.fchmod(descriptor, 0o500)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
rank_cache_base="${scratch}/rank-cache"
mkdir -m 0700 -- "${rank_cache_base}"

"${python_bin}" -I -B - \
  "${archive}" "${manifest}" "${scratch}/source" \
  "${job_id}" "${node}" "${seed}" "${arm}" \
  "${expected_runner_sha256}" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

archive_path, manifest_path, output = map(Path, sys.argv[1:4])
job_id, node, seed, arm, runner_sha = sys.argv[4:9]

def reject(message):
    raise SystemExit("release rejected: " + message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

manifest_raw = manifest_path.read_bytes()
try:
    value = json.loads(manifest_raw)
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    reject("manifest parse failed: " + str(error))
if manifest_raw != canonical(value) + b"\n":
    reject("manifest is not canonical JSON+newline")
unsigned = dict(value)
stored = unsigned.pop("manifest_digest", None)
if stored != hashlib.sha256(canonical(unsigned)).hexdigest():
    reject("manifest self-digest differs")
if value.get("schema_version") != "bernini-elal3-c1-oracle-diagnostic-release-v1":
    reject("manifest schema differs")
if value.get("archive_format") != "fixed-ustar-ascii-sorted-owner0-mtime0-record10240-v1":
    reject("archive format differs")
archive_raw = archive_path.read_bytes()
if hashlib.sha256(archive_raw).hexdigest() != value.get("archive_sha256") or len(archive_raw) != value.get("archive_size"):
    reject("archive binding differs")
if any(value.get(key) is not False for key in (
    "formal_c1_authorized", "exact160_authorized",
    "source_instruction_inference_authorized",
    "real_video_generalization_authorized", "production_model_authorized",
    "scientific_claim_authorized",
)):
    reject("forbidden claim became authorized")
if value.get("simulator_optimizer_diagnostic_authorized") is not True or value.get("teacher_forced_oracle_q_required") is not True:
    reject("narrow simulator/oracle authority differs")
if value.get("row_id") != "c1-two-entity-push-to-goal" or value.get("representation_variant") != "full" or value.get("attention_width") != 64 or value.get("lora_rank") != 256:
    reject("one-row/full-w64/r256 identity differs")
if value.get("optimizer_update_sequence") != [0, 1, 10] or value.get("maximum_authorized_optimizer_updates") != 20:
    reject("update sequence/ceiling differs")
if value.get("distributed_topology") != {
    "world_size": 8, "data_parallel_size": 2, "sequence_parallel_size": 4,
    "one_node_per_run": True,
}:
    reject("WORLD8=DP2xSP4 topology differs")
assignments = value.get("run_assignments")
expected_assignments = [
    {"holder_job_id": "141620", "node": "auh7-1b-gpu-226", "seed": 20260817, "arm": "main-full-w64"},
    {"holder_job_id": "141618", "node": "auh7-1b-gpu-249", "seed": 20260818, "arm": "replicate-full-w64-seed2"},
    {"holder_job_id": "141619", "node": "auh7-1b-gpu-257", "seed": 20260819, "arm": "replicate-full-w64-seed3"},
]
if assignments != expected_assignments or {
    "holder_job_id": job_id, "node": node, "seed": int(seed), "arm": arm
} not in assignments:
    reject("job/node/seed assignment differs")
bindings = value.get("authority_bindings")
if bindings != {
    "derivative_authority_sha256": "298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b",
    "derivative_authority_digest": "c1706ee5b3f8a3fa4c037dfa6dbdbc7d0b088d3682128e50e712e311dae35043",
    "model_authority_sha256": "4c2f4d28af646ab39bdeb775e1b651d523d83b3fc0b8e5c1dd4bc78fbd4f25ed",
    "model_authority_digest": "25255902f4c5ce6de94ce6c3666bcf85eae4bf8e360a217f327c6febd049d21b",
    "latent_receipt_sha256": "a400d11d0d1337daa61d74a25e040aab27b83cc75e62038b81b83f56075e4fcb",
    "latent_receipt_digest": "81f0ab734249651b00571e94a616de5a04fb13aa53fd711e45554b5a76251d61",
    "packet_manifest_sha256": "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc",
}:
    reject("authority binding closure differs")
if value.get("external_latent_bundle") != {
    "sha256": "8fbd27abf7b6eea0593b236a0594dcfad38b3bedf46cf42e77391ec5648fdedf",
    "size": 39138208, "mode": "0444", "nlink": 1,
}:
    reject("model-bound latent bundle pin differs")
runtime = value.get("runtime_pins")
runner_name = "methods/bernini_action_editing/train_elal3_c1_simulator_overfit_v1.py"
if not isinstance(runtime, dict) or runtime.get(runner_name) != runner_sha:
    reject("trainer source pin differs")
rows = value.get("files")
if not isinstance(rows, list) or len(rows) != 10:
    reject("release exact10 file closure differs")
expected = {row.get("path"): row for row in rows if isinstance(row, dict)}
if len(expected) != len(rows) or set(runtime) - set(expected):
    reject("release file/runtime closure differs")
with tarfile.open(archive_path, "r:") as release:
    members = release.getmembers()
    names = [member.name for member in members]
    if names != sorted(names, key=lambda item: item.encode("ascii")) or set(names) != set(expected):
        reject("archive member order/closure differs")
    for member in members:
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts or not member.isreg() or member.mode != 0o444 or member.uid != 0 or member.gid != 0 or member.mtime != 0:
            reject("unsafe archive member: " + member.name)
        raw = release.extractfile(member).read()
        row = expected[member.name]
        if set(row) != {"path", "sha256", "size", "mode"} or row["mode"] != "0444" or len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            reject("archive member bytes differ: " + member.name)
        target = output.joinpath(*pure.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
        try:
            remaining = memoryview(raw)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    reject("archive write made no progress: " + member.name)
                remaining = remaining[written:]
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
PY

extracted_root="${scratch}/source"
method_root="${extracted_root}/methods/bernini_action_editing"
trainer="${method_root}/train_elal3_c1_simulator_overfit_v1.py"
derivative_authority="${extracted_root}/md/action_editing/20260817_box/evidence/elal3_c1_simulator_optimizer_diagnostic_authority_v1.json"
model_authority="${extracted_root}/md/action_editing/20260817_box/evidence/elal3_c1_real_model_authority_v1.json"
latent_receipt="${extracted_root}/md/action_editing/20260817_box/evidence/elal3_c1_latent_bundle_receipt_authorized_v1.json"
[[ "$(sha256sum "${trainer}" | awk '{print $1}')" == "${expected_runner_sha256}" ]] || fail "extracted trainer SHA-256 differs"
[[ "$(sha256sum "${derivative_authority}" | awk '{print $1}')" == "${derivative_authority_sha256}" ]] || fail "extracted derivative authority differs"
[[ "$(sha256sum "${model_authority}" | awk '{print $1}')" == "${model_authority_sha256}" ]] || fail "extracted model authority differs"
[[ "$(sha256sum "${latent_receipt}" | awk '{print $1}')" == "${latent_receipt_sha256}" ]] || fail "extracted model-bound latent receipt differs"

# Static actual-shape contract admission.  This authenticates every external
# model file, all exact8 full-video VAE tensors, and the target-derived oracle-q
# label on Bernini's (21,26,35) patch grid.  It does not load the renderer or
# execute a forward pass, so only the later trainer PRECHECK receipt proves the
# WORLD8 runtime model preflight.  It creates no optimizer and performs no
# model update.
"${python_bin}" -I -B - \
  "${method_root}" "${packet_root}" "${latent_bundle}" "${latent_receipt}" \
  "${derivative_authority}" "${model_authority}" "${bernini_root}" \
  "${checkpoint_root}" "${output_root}/NO_UPDATE_PREFLIGHT.json" \
  "${job_id}" "${node}" "${seed}" "${arm}" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys

method_root, packet_root, bundle_path, receipt_path, derivative_path, model_path, bernini_root, checkpoint_root, output = map(Path, sys.argv[1:10])
job_id, node, seed, arm = sys.argv[10:14]
sys.path.insert(0, str(method_root))

import torch
from safetensors import safe_open
import elal3_simulator_label_v1 as labels

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

if torch.cuda.device_count() != 8:
    raise SystemExit(f"actual-shape preflight requires exact8 visible GPUs; found {torch.cuda.device_count()}")
devices = []
for index in range(8):
    row = torch.cuda.get_device_properties(index)
    devices.append({"index": index, "name": row.name, "total_memory": int(row.total_memory)})
receipt = json.loads(receipt_path.read_bytes())
receipt_unsigned = dict(receipt)
receipt_digest = receipt_unsigned.pop("receipt_digest", None)
if (
    receipt_digest != "81f0ab734249651b00571e94a616de5a04fb13aa53fd711e45554b5a76251d61"
    or hashlib.sha256(canonical(receipt_unsigned)).hexdigest() != receipt_digest
):
    raise SystemExit("latent receipt digest differs")
expected_rows = {row["role"]: row for row in receipt["tensor_rows"]}
if list(expected_rows) != receipt["tensor_order"] or len(expected_rows) != 8:
    raise SystemExit("latent exact8 order differs")
tensor_rows = []
with safe_open(bundle_path, framework="pt", device="cpu") as handle:
    # safetensors exposes its header map in lexical key order; the semantic
    # role order is authenticated separately by the receipt.
    if set(handle.keys()) != set(receipt["tensor_order"]) or len(handle.keys()) != 8:
        raise SystemExit("safetensors exact8 key closure differs")
    for role in receipt["tensor_order"]:
        tensor = handle.get_tensor(role).detach().contiguous()
        row = expected_rows[role]
        header = canonical({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        digest = hashlib.sha256(header + b"\0" + tensor.numpy().tobytes(order="C")).hexdigest()
        if list(tensor.shape) != [1, 16, 21, 52, 70] or str(tensor.dtype) != "torch.float32" or digest != row["sha256"]:
            raise SystemExit("latent tensor differs: " + role)
        tensor_rows.append({"role": role, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": digest})
authority = json.loads(model_path.read_bytes())
roots = {
    "bernini": bernini_root,
    "checkpoint": checkpoint_root,
    "python_env": Path(authority["python_env_root"]),
}
model_rows = []
for row in authority["files"]:
    pure = PurePosixPath(row["relative_path"])
    if pure.is_absolute() or ".." in pure.parts:
        raise SystemExit("unsafe model authority relative path")
    path = roots[row["root"]].joinpath(*pure.parts)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != row["mode"] or info.st_size != row["size"] or sha_file(path) != row["sha256"]:
        raise SystemExit("real model file differs: " + row["root"] + ":" + row["relative_path"])
    model_rows.append({"root": row["root"], "relative_path": row["relative_path"], "sha256": row["sha256"]})
oracle = labels.load_oracle_q_label_v1(
    packet_root,
    patch_grid=(21, 26, 35),
    external_authority_path=derivative_path,
    external_authority_sha256="298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b",
    device="cpu",
    dtype=torch.float32,
)
unsigned = {
    "schema_version": "bernini-elal3-c1-oracle-diagnostic-no-update-preflight-v1",
    "status": "STATIC_ACTUAL_SHAPE_CONTRACT_PASS_TRAINER_NOT_YET_RUN",
    "scope": "simulator_oracle_q_exact_one_row_optimizer_diagnostic_only",
    "job_id": job_id,
    "node": node,
    "seed": int(seed),
    "arm": arm,
    "row_id": "c1-two-entity-push-to-goal",
    "world_size": 8,
    "data_parallel_size": 2,
    "sequence_parallel_size": 4,
    "devices": devices,
    "latent_shape": [1, 16, 21, 52, 70],
    "patch_grid": [21, 26, 35],
    "tensor_rows": tensor_rows,
    "tensor_rows_digest": hashlib.sha256(canonical(tensor_rows)).hexdigest(),
    "oracle_label_digest": oracle.receipt["label_digest"],
    "event_cells": int(oracle.event_mask_patch.sum().item()),
    "context_cells": int(oracle.context_mask_patch.sum().item()),
    "model_file_rows": model_rows,
    "model_file_rows_digest": hashlib.sha256(canonical(model_rows)).hexdigest(),
    "optimizer_created": False,
    "optimizer_updates": 0,
    "renderer_loaded": False,
    "model_forward_executed": False,
    "trainer_runtime_precheck_completed": False,
    "frozen_teacher_used": False,
    "frozen_base_velocity_reference_used": False,
    "self_distillation_used": False,
    "reward_scalar_used": False,
    "teacher_forced_oracle_q": True,
    "formal_c1_authorized": False,
    "exact160_authorized": False,
    "source_instruction_inference_authorized": False,
    "real_video_generalization_authorized": False,
    "production_model_authorized": False,
    "scientific_claim_authorized": False,
}
value = {**unsigned, "receipt_digest": hashlib.sha256(canonical(unsigned)).hexdigest()}
raw = canonical(value) + b"\n"
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o440)
try:
    remaining = memoryview(raw)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise SystemExit("no-update receipt write made no progress")
        remaining = remaining[written:]
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
chmod 0444 -- "${output_root}/NO_UPDATE_PREFLIGHT.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

validate_stage_receipt() {
  local stage_root="$1" receipt_name="$2" mode="$3" steps="$4"
  "${python_bin}" -I -B - \
    "${stage_root}" "${receipt_name}" "${mode}" "${steps}" "${seed}" \
    "${train_lora_source_sha256}" "${elal3_core_source_sha256}" \
    "${elal3_label_source_sha256}" "${packed_lora_source_sha256}" \
    "${runtime_source_sha256}" "${sigma_source_sha256}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
receipt_name, mode = sys.argv[2:4]
steps, seed = map(int, sys.argv[4:6])
train_lora_sha, core_sha, label_sha, packed_sha, runtime_sha, sigma_sha = sys.argv[6:12]

def reject(message):
    raise SystemExit("trainer stage receipt rejected: " + message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

path = root / receipt_name
info = path.lstat()
if not stat.S_ISREG(info.st_mode) or path.is_symlink() or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
    reject("receipt file type/mode/link differs")
raw = path.read_bytes()
try:
    value = json.loads(raw)
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    reject("receipt parse failed: " + str(error))
if raw != canonical(value) + b"\n":
    reject("receipt is not canonical JSON+newline")
unsigned = dict(value)
stored = unsigned.pop("receipt_digest", None)
if stored != hashlib.sha256(canonical(unsigned)).hexdigest():
    reject("receipt self-digest differs")
if value.get("schema_version") != "bernini-elal3-c1-simulator-overfit-training-receipt-v1":
    reject("receipt schema differs")
if value.get("row_id") != "c1-two-entity-push-to-goal" or value.get("seed") != seed or value.get("max_steps") != steps:
    reject("row/seed/max_steps differs")
boundaries = {
    "oracle_q_teacher_forced": True,
    "fresh_official_base": True,
    "resume_consumed": False,
    "source_instruction_inference": False,
    "formal_c1_authorized": False,
    "exact160_authorized": False,
    "scientific_claim_authorized": False,
    "real_video_data": False,
    "action_encoder_qualified": False,
    "action_predictor_present": False,
    "frozen_teacher_used": False,
    "frozen_velocity_reference_used": False,
    "self_distillation_used": False,
    "reward_used": False,
}
if any(value.get(key) is not expected for key, expected in boundaries.items()):
    reject("authority/objective boundary differs")
if value.get("lora_affines") != 240 or value.get("lora_rank") != 256 or value.get("elal3_variant") != "full-w64" or value.get("trainable_parameter_count") != 198723614:
    reject("trainable model closure differs")
if (
    value.get("activation_checkpoint_profile") != "selective-nonreentrant-stride4-exact8"
    or value.get("activation_checkpointed_blocks") != [0, 4, 8, 12, 16, 20, 24, 28]
    or value.get("activation_uncheckpointed_blocks") != [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29]
    or value.get("activation_checkpoint_nonreentrant") is not True
    or value.get("activation_checkpoint_elal_route_context_replay") is not True
    or value.get("memory_gate_true_training_tensors_only") is not True
    or value.get("dummy_or_padding_memory_allocations") is not False
):
    reject("selective-checkpoint/no-dummy memory closure differs")
if value.get("latent_bundle_sha256") != "8fbd27abf7b6eea0593b236a0594dcfad38b3bedf46cf42e77391ec5648fdedf" or value.get("latent_bundle_receipt_sha256") != "a400d11d0d1337daa61d74a25e040aab27b83cc75e62038b81b83f56075e4fcb":
    reject("model-bound latent closure differs")
if value.get("external_optimizer_authority_sha256") != "298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b" or value.get("model_authority_sha256") != "4c2f4d28af646ab39bdeb775e1b651d523d83b3fc0b8e5c1dd4bc78fbd4f25ed":
    reject("authority file closure differs")
sources = value.get("local_source_closure")
expected_sources = {
    "train_lora": train_lora_sha,
    "elal3_core": core_sha,
    "elal3_label": label_sha,
    "packed_lora": packed_sha,
    "runtime": runtime_sha,
    "sigma": sigma_sha,
}
if not isinstance(sources, dict) or set(sources) != set(expected_sources):
    reject("local source closure differs")
for name, expected in expected_sources.items():
    row = sources[name]
    if not isinstance(row, dict) or row.get("sha256") != expected or not str(row.get("path", "")).endswith(".py") or type(row.get("size")) is not int or row["size"] <= 0:
        reject("local source row differs: " + name)
if mode == "preflight":
    if value.get("status") != "PRECHECK_COMPLETE_NO_OPTIMIZER_CONSTRUCTED_NO_UPDATE" or value.get("preflight_only") is not True or value.get("completed_optimizer_steps") != 0:
        reject("no-update preflight status differs")
    checkpoint_root = root / "checkpoints"
    if not checkpoint_root.is_dir() or list(checkpoint_root.iterdir()):
        reject("preflight checkpoint directory is absent or non-empty")
    if {item.name for item in root.iterdir()} != {"checkpoints", "PRECHECK_RECEIPT.json"}:
        reject("preflight output root closure differs")
elif mode == "train":
    if value.get("status") != "TRAINING_COMPLETE_SIMULATOR_ORACLE_Q_OVERFIT_DIAGNOSTIC_ONLY" or value.get("preflight_only") is not False or value.get("completed_optimizer_steps") != steps or value.get("requested_optimizer_steps") != steps:
        reject("training completion status/steps differs")
    for key in (
        "fresh_initialization_verified", "parameters_changed",
        "memory_gate_all_steps_all8_strictly_gt_half",
        "all_steps_finite_nonzero_synchronized_gradients",
        "all_steps_all30_elal_used_and_nonzero_after_sp_reduction",
    ):
        if value.get(key) is not True:
            reject("training gate false: " + key)
    if value.get("initial_parameter_sha256") == value.get("final_parameter_sha256"):
        reject("trainable parameters did not change")
    history = value.get("history")
    if not isinstance(history, list) or len(history) != steps or [row.get("step") for row in history] != list(range(1, steps + 1)):
        reject("training history closure differs")
    for row in history:
        memory = row.get("memory_world8")
        if row.get("optimizer_step_executed") is not True or row.get("memory_gate_all8_pass") is not True or not isinstance(memory, list) or len(memory) != 8:
            reject("per-step optimizer/memory closure differs")
        if [item.get("world_rank") for item in memory] != list(range(8)) or any(item.get("strictly_greater_than_half") is not True or not math.isfinite(item.get("peak_allocated_fraction", float("nan"))) or item["peak_allocated_fraction"] <= 0.5 for item in memory):
            reject("per-rank >50% memory gate differs")
    checkpoints = value.get("checkpoint_records")
    if not isinstance(checkpoints, list) or [row.get("step") for row in checkpoints] != [0, steps]:
        reject("checkpoint receipt list differs")
    if {item.name for item in root.iterdir()} != {"checkpoints", "TRAINING_RECEIPT.json"}:
        reject("training output root closure differs")
    checkpoint_root = root / "checkpoints"
    if {item.name for item in checkpoint_root.iterdir()} != {"checkpoint-00000000", f"checkpoint-{steps:08d}"}:
        reject("checkpoint directory closure differs")
    for step in (0, steps):
        directory = root / "checkpoints" / f"checkpoint-{step:08d}"
        metadata = directory / "CHECKPOINT_RECEIPT.json"
        if not directory.is_dir() or not metadata.is_file() or metadata.is_symlink():
            reject("checkpoint directory/receipt differs")
        expected_files = {"adapter-and-elal3.pt", "CHECKPOINT_RECEIPT.json"}
        if step == steps:
            expected_files.add("optimizer.pt")
        if {item.name for item in directory.iterdir()} != expected_files:
            reject("checkpoint file closure differs")
        metadata_raw = metadata.read_bytes()
        try:
            metadata_value = json.loads(metadata_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            reject("checkpoint receipt parse failed: " + str(error))
        metadata_unsigned = dict(metadata_value)
        metadata_digest = metadata_unsigned.pop("receipt_digest", None)
        if metadata_raw != canonical(metadata_value) + b"\n" or metadata_digest != hashlib.sha256(canonical(metadata_unsigned)).hexdigest():
            reject("checkpoint receipt canonical self-digest differs")
        if metadata_value.get("step") != step or metadata_value.get("strict_weights_only_reload_verified") is not True or metadata_value.get("oracle_q_teacher_forced") is not True or metadata_value.get("formal_c1_authorized") is not False or metadata_value.get("exact160_authorized") is not False or metadata_value.get("scientific_claim_authorized") is not False or metadata_value.get("source_instruction_inference") is not False:
            reject("checkpoint receipt semantics differ")
        if (
            metadata_value.get("activation_checkpoint_profile") != "selective-nonreentrant-stride4-exact8"
            or metadata_value.get("activation_checkpointed_blocks") != [0, 4, 8, 12, 16, 20, 24, 28]
            or metadata_value.get("activation_uncheckpointed_blocks") != [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29]
            or metadata_value.get("activation_checkpoint_nonreentrant") is not True
            or metadata_value.get("activation_checkpoint_elal_route_context_replay") is not True
            or metadata_value.get("memory_gate_true_training_tensors_only") is not True
            or metadata_value.get("dummy_or_padding_memory_allocations") is not False
        ):
            reject("checkpoint selective-checkpoint/no-dummy closure differs")
        if step == 0 and (metadata_value.get("optimizer_file") is not None or metadata_value.get("optimizer_sha256") is not None):
            reject("initial checkpoint unexpectedly contains optimizer")
        if step == steps and (metadata_value.get("optimizer_file") != "optimizer.pt" or not isinstance(metadata_value.get("optimizer_sha256"), str)):
            reject("final optimizer checkpoint binding differs")
else:
    reject("stage mode differs")
PY
}

run_stage() {
  local label="$1" steps="$2" mode="$3" stage_root log_path stage_status stage_cache_base receipt_name validation_status
  local -a mode_flags=()
  stage_root="${output_root}/${label}"
  log_path="${output_root}/${label}.log"
  stage_cache_base="${rank_cache_base}/${label}"
  if [[ "${mode}" == preflight ]]; then
    mode_flags=(--preflight-only)
    receipt_name="PRECHECK_RECEIPT.json"
  elif [[ "${mode}" == train ]]; then
    receipt_name="TRAINING_RECEIPT.json"
  else
    fail "unknown trainer stage mode: ${mode}"
  fi
  [[ ! -e "${stage_root}" && ! -L "${stage_root}" && ! -e "${log_path}" && ! -L "${log_path}" ]] || fail "stage output is not fresh: ${label}"
  mkdir -m 0700 -- "${stage_cache_base}"
  set +e
  "${python_bin}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    --master-port="${master_port}" \
    --no-python \
    "${rank_wrapper}" "${stage_cache_base}" "${python_bin}" -B "${trainer}" \
      --bernini-root "${bernini_root}" \
      --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint_root}" \
      --packet-root "${packet_root}" \
      --latent-bundle "${latent_bundle}" \
      --expected-latent-bundle-sha256 "${latent_bundle_sha256}" \
      --latent-bundle-receipt "${latent_receipt}" \
      --expected-latent-bundle-receipt-sha256 "${latent_receipt_sha256}" \
      --external-optimizer-authority "${derivative_authority}" \
      --expected-external-optimizer-authority-sha256 "${derivative_authority_sha256}" \
      --model-authority "${model_authority}" \
      --expected-model-authority-sha256 "${model_authority_sha256}" \
      --output "${stage_root}" \
      --expected-runner-source-sha256 "${expected_runner_sha256}" \
      --expected-train-lora-source-sha256 "${train_lora_source_sha256}" \
      --expected-elal3-core-source-sha256 "${elal3_core_source_sha256}" \
      --expected-elal3-label-source-sha256 "${elal3_label_source_sha256}" \
      --expected-packed-lora-source-sha256 "${packed_lora_source_sha256}" \
      --expected-runtime-source-sha256 "${runtime_source_sha256}" \
      --expected-sigma-source-sha256 "${sigma_source_sha256}" \
      --max-steps "${steps}" \
      --seed "${seed}" \
      "${mode_flags[@]}" \
      "${trainer_ack_flags[@]}" \
      >"${log_path}" 2>&1
  stage_status=$?
  set -e
  chmod 0444 -- "${log_path}"
  if [[ "${stage_status}" -ne 0 ]]; then
    if [[ -d "${stage_root}" && ! -L "${stage_root}" ]]; then
      chmod -R a-w -- "${stage_root}" || true
    fi
    fail "${label} torchrun failed with status ${stage_status}; failure log sealed"
  fi
  [[ -f "${stage_root}/${receipt_name}" && ! -L "${stage_root}/${receipt_name}" ]] || fail "missing stage receipt: ${label}"
  if [[ "${mode}" == train ]]; then
    [[ -d "${stage_root}/checkpoints/checkpoint-00000000" && -d "${stage_root}/checkpoints/checkpoint-$(printf '%08d' "${steps}")" ]] || fail "checkpoint closure differs: ${label}"
  fi
  set +e
  validate_stage_receipt "${stage_root}" "${receipt_name}" "${mode}" "${steps}"
  validation_status=$?
  set -e
  if [[ "${validation_status}" -ne 0 ]]; then
    chmod -R a-w -- "${stage_root}" || true
    fail "${label} receipt validation failed with status ${validation_status}; stage sealed"
  fi
  chmod -R a-w -- "${stage_root}"
}

# All three are fresh WORLD8 model loads.  No checkpoint is reused between the
# preflight, 1-step smoke, and 10-step diagnostic.
run_stage elal3_c1_preflight_no_update 1 preflight
run_stage elal3_c1_one_step_smoke 1 train
run_stage elal3_c1_ten_step_overfit 10 train

"${python_bin}" -I -B - \
  "${output_root}" "${job_id}" "${node}" "${seed}" "${arm}" \
  "${expected_archive_sha256}" "${expected_manifest_sha256}" \
  "${expected_runner_sha256}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
job_id, node, seed, arm, archive_sha, manifest_sha, runner_sha = sys.argv[2:9]

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

def row(path):
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}

rows = [
    row(root / "NO_UPDATE_PREFLIGHT.json"),
    row(root / "elal3_c1_preflight_no_update.log"),
    row(root / "elal3_c1_preflight_no_update/PRECHECK_RECEIPT.json"),
    row(root / "elal3_c1_one_step_smoke.log"),
    row(root / "elal3_c1_one_step_smoke/TRAINING_RECEIPT.json"),
    row(root / "elal3_c1_ten_step_overfit.log"),
    row(root / "elal3_c1_ten_step_overfit/TRAINING_RECEIPT.json"),
]
unsigned = {
    "schema_version": "bernini-elal3-c1-oracle-diagnostic-run-complete-v1",
    "status": "SIMULATOR_ORACLE_Q_DIAGNOSTIC_COMPLETE",
    "scope": "simulator_oracle_q_exact_one_row_optimizer_diagnostic_only",
    "job_id": job_id,
    "node": node,
    "seed": int(seed),
    "arm": arm,
    "world_size": 8,
    "data_parallel_size": 2,
    "sequence_parallel_size": 4,
    "optimizer_update_sequence": [0, 1, 10],
    "world8_preflight_no_optimizer": True,
    "fresh_ten_step_run_after_smoke": True,
    "release_archive_sha256": archive_sha,
    "release_manifest_sha256": manifest_sha,
    "trainer_source_sha256": runner_sha,
    "artifact_rows": rows,
    "artifact_rows_digest": hashlib.sha256(canonical(rows)).hexdigest(),
    "teacher_forced_oracle_q": True,
    "formal_c1_authorized": False,
    "exact160_authorized": False,
    "source_instruction_inference_authorized": False,
    "real_video_generalization_authorized": False,
    "production_model_authorized": False,
    "scientific_claim_authorized": False,
}
value = {**unsigned, "receipt_digest": hashlib.sha256(canonical(unsigned)).hexdigest()}
raw = canonical(value) + b"\n"
path = root / "RUN_COMPLETE.json"
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o440)
try:
    remaining = memoryview(raw)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise SystemExit("completion receipt write made no progress")
        remaining = remaining[written:]
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY

chmod 0444 -- "${output_root}/RUN_COMPLETE.json"
chmod -R a-w -- "${output_root}"
chmod 0555 -- "${output_root}"
echo "[elal3-c1-oracle-diagnostic] COMPLETE ${job_id} ${node} seed=${seed} output=${output_root}"

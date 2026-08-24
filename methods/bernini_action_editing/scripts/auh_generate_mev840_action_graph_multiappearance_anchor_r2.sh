#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# One frozen, source-free native T2V anchor per invocation.  The script is
# intentionally restricted to Job 143808 and to the preregistered node/variant
# map.  It never accepts a target, route, edit decoder, checkpoint update, or
# training argument.
if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 MANIFEST_JSON VARIANT_ID" >&2
  exit 2
fi

readonly manifest="$1"
readonly variant_id="$2"
readonly expected_manifest_sha256=be8dba8d32b63d79660f46f38fa3f66926b3fa29947d6898726556ce130336db
readonly expected_job_id=143808

fail() {
  echo "[mev840-anchor-bank] ERROR: $*" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha256() {
  local path="$1"
  local expected="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "plain file required: ${path}"
  local observed
  observed="$(sha256_file "${path}")"
  [[ "${observed}" == "${expected}" ]] || fail "sha256 differs for ${path}: ${observed}"
}

[[ "${manifest}" == /* ]] || fail "manifest must be absolute"
[[ "${variant_id}" =~ ^v[123]$ ]] || fail "variant must be v1, v2, or v3"
require_sha256 "${manifest}" "${expected_manifest_sha256}"
[[ "${SLURM_JOB_ID:-}" == "${expected_job_id}" ]] || fail "requires Slurm Job ${expected_job_id}"
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "requires a numbered Slurm step"
[[ "${SLURM_GPUS_ON_NODE:-}" == "4" ]] || fail "requires exactly four GPUs in this step"
[[ "${ROCR_VISIBLE_DEVICES:-}" == "0,1,2,3" ]] || fail "requires ROCR devices 0,1,2,3"

readonly release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1
readonly source_archive="${release_root}/inputs/runtime-source.tar"
readonly runtime_closure="${release_root}/inputs/runtime-closure.json"
readonly checkpoint_manifest="${release_root}/inputs/bernini_r13_ff4c5d4_checkpoint.sha256"
readonly bernini_root="${release_root}/vendor/Bernini-2d2b4591"
readonly veomni_root="${release_root}/vendor/VeOmni-f90b3dc6"
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly source_video=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v1/preprocessed_sources/840b214afead/source-exact81.mp4
readonly output_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev840_self_generated_action_graph_anchor_bank_20260821_r2

require_sha256 "${source_archive}" ef7b83a12e5dcb94274fa2e9aea5c4067f0c7e82c2f1c733af020d430ecbcbf0
require_sha256 "${runtime_closure}" 61cb4198613d497e0f858740322aa874a8c479ed89c8b123db6f2e8f74b38abb
require_sha256 "${checkpoint_manifest}" a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
require_sha256 "${python_bin}" 8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
require_sha256 "${source_video}" a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "sealed Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "sealed VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root differs"
[[ "$(git -C "${bernini_root}" rev-parse HEAD)" == 2d2b4591ac053ec25c6371b01a5a6746679e5793 ]] || fail "Bernini commit differs"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == f90b3dc6fbb0ce693745223cc7a94064123dbf4d ]] || fail "VeOmni commit differs"

mapfile -t fields < <("${python_bin}" -B - "${manifest}" "${variant_id}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
variant_id = sys.argv[2]
document = json.loads(manifest_path.read_text(encoding="utf-8"))
assert document["schema_version"] == "mev840-self-generated-action-graph-anchor-bank-r2"
failed = document["supersedes_failed_preflight"]
assert failed["slurm_steps"] == ["143808.468", "143808.469", "143808.470"]
assert failed["outputs_created"] is False
authorization = document["authorization"]
assert authorization == {
    "native_frozen_t2v_anchor_generation": True,
    "source_edit_route": False,
    "source_edit_decode": False,
    "training": False,
    "optimizer_steps": 0,
    "lora": False,
    "parameter_updates": False,
    "real_target_read": False,
    "external_reference_read": False,
    "mask_flow_pose_track_trajectory_read": False,
    "scientific_use_before_visual_review": "representation_preflight_only",
}
assert document["generic_action"]["appearance_fields_authorized_in_representation"] is False
assert document["geometry_source"]["native_t2v_conditioning_count"] == 0
assert document["output"]["native_arm"] == "t2v"
assert document["output"]["num_inference_steps"] == 40
rows = [row for row in document["variants"] if row["variant_id"] == variant_id]
assert len(rows) == 1
row = rows[0]
prompt = row["prompt"]
assert "\n" not in prompt and "\x00" not in prompt
assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == row["prompt_sha256"]
print(document["slurm"]["node_variant_map"][variant_id])
print(row["seed"])
print(prompt)
print(row["prompt_sha256"])
print(document["output"]["root"])
PY
)
[[ "${#fields[@]}" -eq 5 ]] || fail "manifest variant projection differs"
readonly expected_node="${fields[0]}"
readonly seed="${fields[1]}"
readonly prompt="${fields[2]}"
readonly prompt_sha256="${fields[3]}"
[[ "${fields[4]}" == "${output_root}" ]] || fail "output root differs from manifest"
[[ "$(hostname -s)" == "${expected_node}" ]] || fail "${variant_id} is restricted to ${expected_node}"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed differs"

readonly output_parent="${output_root}/anchors"
readonly output_dir="${output_parent}/${variant_id}"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse ${output_dir}"
mkdir -p "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"

readonly scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/mev840-anchor-${variant_id}-${SLURM_JOB_ID}-${SLURM_STEP_ID}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/mev840-anchor-${variant_id}-${SLURM_JOB_ID}-${SLURM_STEP_ID}."*)
      chmod -R u+w "${task_scratch}" 2>/dev/null || true
      rm -rf -- "${task_scratch}" 2>/dev/null || true
      ;;
    *)
      echo "[mev840-anchor-bank] refusing unsafe scratch cleanup" >&2
      [[ "${status}" -ne 0 ]] || status=2
      ;;
  esac
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p \
  "${task_scratch}/source-tree" \
  "${task_scratch}/cache/miopen-user" \
  "${task_scratch}/cache/miopen-custom" \
  "${task_scratch}/cache/torch-extensions" \
  "${task_scratch}/cache/triton" \
  "${task_scratch}/cache/xdg" \
  "${task_scratch}/cache/pycache" \
  "${task_scratch}/tmp"
tar --no-same-owner --no-same-permissions -xf "${source_archive}" -C "${task_scratch}/source-tree"
readonly method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
readonly runner="${method_root}/infer_native_identity_generation_canary.py"
require_sha256 "${runner}" a60c37591c40206c6130185f1a2d2a7a8e473f5af4425205e268ae4a8b58f334
find "${method_root}" -type f -exec chmod a-w -- {} +

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"
export TMPDIR="${task_scratch}/tmp"
export TORCHELASTIC_ERROR_FILE="${task_scratch}/torch-elastic-error.json"
export BERNINI_OUTPUT_TRANSACTION_ID="${SLURM_JOB_ID}-${SLURM_STEP_ID}-${variant_id}"
export PYTHONPATH="${method_root}:${bernini_root}:${veomni_root}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

echo "[mev840-anchor-bank] START variant=${variant_id} seed=${seed} job=${SLURM_JOB_ID} step=${SLURM_STEP_ID} node=$(hostname -s) topology=SP4"
echo "[mev840-anchor-bank] authority=frozen_native_t2v_only training=false updates=0 route=false source_edit_decode=false real_target_read=false"
"${python_bin}" -B -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${runner}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --source-video "${source_video}" \
  --expected-source-sha256 a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646 \
  --action-prompt "${prompt}" \
  --expected-action-prompt-sha256 "${prompt_sha256}" \
  --output-dir "${output_dir}" \
  --arms t2v \
  --num-inference-steps 40 \
  --seed "${seed}" \
  --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793 \
  --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d \
  --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca \
  --method-source-revision 00f7aba4bd58e67778542273cfca82fb785b648c \
  --method-source-archive-sha256 ef7b83a12e5dcb94274fa2e9aea5c4067f0c7e82c2f1c733af020d430ecbcbf0

"${python_bin}" -B - \
  "${manifest}" "${expected_manifest_sha256}" "${variant_id}" \
  "${output_dir}" "${prompt}" "${prompt_sha256}" "${seed}" \
  "$(hostname -s)" "${SLURM_JOB_ID}" "${SLURM_STEP_ID}" <<'PY'
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

import av


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


manifest_path = Path(sys.argv[1])
manifest_sha = sys.argv[2]
variant_id = sys.argv[3]
output_dir = Path(sys.argv[4])
prompt, prompt_sha, seed = sys.argv[5], sys.argv[6], int(sys.argv[7])
node, job_id, step_id = sys.argv[8], sys.argv[9], sys.argv[10]
assert sha256_file(manifest_path) == manifest_sha
document = json.loads(manifest_path.read_text(encoding="utf-8"))
native_path = output_dir / "receipt.json"
video_path = output_dir / "t2v.mp4"
native = json.loads(native_path.read_text(encoding="utf-8"))
declared_digest = native.pop("receipt_digest")
assert hashlib.sha256(canonical(native)).hexdigest() == declared_digest
native["receipt_digest"] = declared_digest
assert native["schema_version"] == "bernini-native-identity-generation-canary-v1"
assert native["arms"] == ["t2v"]
assert native["input"]["action_prompt_utf8_sha256"] == prompt_sha
assert native["input"]["target_video"] is False
assert native["input"]["external_reference_image_or_video"] is False
assert native["input"]["external_mask_flow_pose_track_trajectory"] is False
assert native["input"]["external_first_frame_anchor"] is False
assert native["conditioning"]["t2v"]["full_source_video_count"] == 0
assert native["conditioning"]["t2v"]["source_derived_reference_count"] == 0
assert native["sampling"]["t2v"]["seed"] == seed
assert native["freeze_certificate"]["trainable_parameter_elements"] == 0
assert native["interpretation"]["training_performed"] is False
assert native["outputs"]["t2v"]["frame_count"] == 81
assert native["outputs"]["t2v"]["fps"] == 25
with av.open(str(video_path), "r") as container:
    streams = list(container.streams.video)
    assert len(streams) == 1
    stream = streams[0]
    frame_count = sum(1 for _ in container.decode(stream))
    fps = Fraction(stream.average_rate)
    width = int(stream.codec_context.width)
    height = int(stream.codec_context.height)
assert frame_count == 81 and fps == Fraction(25, 1)
receipt = {
    "schema_version": "mev840-self-generated-action-graph-anchor-receipt-r2",
    "complete": True,
    "variant_id": variant_id,
    "generic_action": document["generic_action"],
    "appearance_family": next(
        row["appearance_family"]
        for row in document["variants"]
        if row["variant_id"] == variant_id
    ),
    "prompt": prompt,
    "prompt_sha256": prompt_sha,
    "seed": seed,
    "slurm": {"job_id": job_id, "step_id": step_id, "node": node, "topology": "SP4"},
    "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
    "authority": {
        "frozen_native_t2v_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "parameter_updates": False,
        "source_edit_route": False,
        "source_edit_decode": False,
        "native_t2v_media_decode": True,
        "real_target_read": False,
    },
    "media": {
        "path": str(video_path),
        "sha256": sha256_file(video_path),
        "frame_count": frame_count,
        "fps": "25/1",
        "width": width,
        "height": height,
    },
    "native_receipt": {
        "path": str(native_path),
        "sha256": sha256_file(native_path),
        "receipt_digest": declared_digest,
    },
}
receipt["receipt_digest"] = hashlib.sha256(canonical(receipt)).hexdigest()
receipt_path = output_dir / "representation_anchor_receipt.json"
temporary = receipt_path.with_name(receipt_path.name + ".partial")
temporary.write_bytes(canonical(receipt) + b"\n")
os.replace(temporary, receipt_path)
(output_dir / "REPRESENTATION_ONLY_ANCHOR_COMPLETE").write_text(
    receipt["receipt_digest"] + "\n", encoding="utf-8"
)
print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
PY

echo "[mev840-anchor-bank] PASS variant=${variant_id} frames=81 fps=25 training=false route=false real_target_read=false output=${output_dir}"

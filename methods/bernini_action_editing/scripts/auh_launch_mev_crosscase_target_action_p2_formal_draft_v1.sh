#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# DRAFT deployment/launch wrapper for the cross-case native RV2V matrix.
#
# One Slurm step owns one 64-GiB allocation and runs seed 2028 followed by
# seed 2027.  Each seed is one WORLD4 process with the fixed execution order
# P0a -> P1 -> P2 -> P0b.  The run class is the explicitly user-authorized
# target_oracle_diagnostic; these artifacts are not a formal scientific claim.
# This file intentionally does not deploy itself, the runner, or an authority.
# Deployment must seal their SHA-256 pins first.

fail() { echo "[mev-crosscase-formal] ERROR: $*" >&2; exit 2; }
usage() {
  echo "usage: $0 launch-case CASE_ID JOB_ID NODE|worker-case CASE_ID JOB_ID NODE" >&2
  exit 2
}
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }

readonly stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
readonly control_root="${stage}/mev_crosscase_target_action_p2_formal_v1_20260822_control"
readonly output_root="${stage}/mev_crosscase_target_action_p2_formal_v1_20260822"
readonly launcher="${control_root}/auh_launch_mev_crosscase_target_action_p2_formal_v1.sh"
readonly runner_name=infer_native_rv2v_crosscase_paired_prompt_matrix_formal_v1.py
readonly runner="${control_root}/${runner_name}"
readonly runner_pin="${control_root}/pins/${runner_name}.sha256"
readonly core_name=infer_mev840_native_rv2v_paired_prompt_matrix_formal_v1.py
readonly core="${control_root}/${core_name}"
readonly core_sha=1f85e2a2444059161bc8ed073ad0565c315c97fa5e849fb5f6d2ac47c738a0ee
readonly case_assets_root="${control_root}/assets/mev_crosscase_target_action_p2_v1"

readonly release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/generic-action-confirmation40-generation-r3-ac22e19f-r1
readonly runtime_archive="${release_root}/source.tar"
readonly runtime_manifest="${release_root}/source.manifest.json"
readonly runtime_archive_sha=46ae7529d640a197006ab8d7d17c23ac81925dabd7fa1caf4b0bb261197e8115
readonly runtime_manifest_sha=e104031526236f16e94a4753c31ad8048b1a65345b1913212c35e421fcad48ae
readonly runtime_manifest_digest=4e78a935b2485e3f8c2c94aa5524a82ed25aa0b93aaf58dd81476dc5c9b48044
readonly content_revision=ac22e19ffd109a2d6b85c32c64463b0be8373792
readonly base_runner_name=infer_native_identity_generation_canary.py
readonly base_runner_sha=bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42

readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly unipc_source=/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/diffusers/schedulers/scheduling_unipc_multistep.py
readonly unipc_source_sha=5bfe1dcf55ebea6dbbf624d3af676b2529b81fbcaf493150d562ec9e1aba3872
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831

plain_file() { [[ -f "$1" && ! -L "$1" ]] || fail "plain file missing: $1"; }
verify_file() {
  plain_file "$1"
  [[ "$(sha256_file "$1")" == "$2" ]] || fail "SHA-256 differs: $1"
}
read_sealed_pin() {
  local path="$1" value mode
  plain_file "$path"
  mode="$(stat -c '%a' -- "$path")"
  [[ "$mode" == 400 ]] || fail "pin must be mode 0400: $path"
  [[ "$(wc -c <"$path")" == 65 ]] || fail "pin byte length differs: $path"
  IFS= read -r value <"$path"
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail "pin format differs: $path"
  printf '%s\n' "$value"
}
validate_case_id() {
  [[ "$1" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] || fail "unsafe CASE_ID: $1"
}
validate_case_holder() {
  case "$1|$2|$3" in
    8b05aaf463db\|147873\|auh7-1b-gpu-284|40712e1341dc\|147881\|auh7-1b-gpu-213|5e83a9279951\|147871\|auh7-1b-gpu-232) ;;
    *) fail "case/holder binding differs: $1/$2/$3" ;;
  esac
}
validate_holder_binding() {
  local holder_job_arg="$1" holder_node_arg="$2" state nodes step
  state="$(squeue -h -j "$holder_job_arg" -o '%T')"
  [[ "$state" == RUNNING ]] || fail "holder is not RUNNING: $holder_job_arg ($state)"
  nodes="$(squeue -h -j "$holder_job_arg" -o '%N')"
  [[ "$(scontrol show hostnames "$nodes")" == "$holder_node_arg" ]] || fail "holder/node differs: ${holder_job_arg}/${holder_node_arg}"
  while IFS= read -r step; do
    [[ -z "$step" || "$step" == "${holder_job_arg}.batch" || "$step" == "${holder_job_arg}.extern" ]] \
      || fail "holder already has an active compute step: $step"
  done < <(squeue -s -h -j "$holder_job_arg" -o '%i')
}

verify_release() {
  local extracted="${1:-}"
  "${python_bin}" -B - "${runtime_archive}" "${runtime_manifest}" \
    "${runtime_manifest_digest}" "$extracted" "$base_runner_name" "$base_runner_sha" <<'PY'
from pathlib import Path, PurePosixPath
import hashlib, json, stat, sys, tarfile

archive, manifest = Path(sys.argv[1]), Path(sys.argv[2])
want_digest, extracted, base_name, base_sha = sys.argv[3:]
value = json.loads(manifest.read_text(encoding="ascii"))
unsigned = dict(value)
declared = unsigned.pop("manifest_digest", None)
raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
if declared != want_digest or hashlib.sha256(raw).hexdigest() != want_digest:
    raise SystemExit("runtime manifest digest differs")
expected = {row["path"]: row["sha256"] for row in value["files"]}
if value.get("file_count") != 19 or len(expected) != 19 or expected.get(base_name) != base_sha:
    raise SystemExit("runtime exact19 identity differs")
seen = {}
with tarfile.open(archive, "r:") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("unsafe runtime archive member")
        if not member.isfile():
            continue
        try:
            relative = path.relative_to(PurePosixPath("methods/bernini_action_editing")).as_posix()
        except ValueError as error:
            raise SystemExit("runtime archive escaped method root") from error
        seen[relative] = hashlib.sha256(handle.extractfile(member).read()).hexdigest()
if seen != expected:
    raise SystemExit("runtime archive content closure differs")
if extracted:
    root = Path(extracted) / "methods/bernini_action_editing"
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(expected):
        raise SystemExit("extracted runtime exact19 member set differs")
    for relative, digest in expected.items():
        path = root / relative
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit("extracted runtime member differs: " + relative)
print("MEV_CROSSCASE_RELEASE_OK", len(expected), "scratch" if extracted else "archive")
PY
}

case_bindings() {
  local case_arg="$1" case_root_path authority_path authority_pin_path authority_digest source_path source_digest
  case_root_path="${case_assets_root}/${case_arg}"
  authority_path="${case_root_path}/authority.json"
  authority_pin_path="${case_root_path}/authority.sha256"
  [[ -d "$case_root_path" && ! -L "$case_root_path" ]] || fail "case authority directory differs: $case_root_path"
  authority_digest="$(read_sealed_pin "$authority_pin_path")"
  verify_file "$authority_path" "$authority_digest"
  IFS=$'\t' read -r source_path source_digest < <("${python_bin}" -B - "$authority_path" "$case_arg" <<'PY'
import json, re, sys
value = json.load(open(sys.argv[1], encoding="ascii"))
case = value.get("case")
if not isinstance(case, dict) or case.get("case_id") != sys.argv[2]:
    raise SystemExit("authority case_id differs")
source = case.get("source_video")
if not isinstance(source, dict):
    raise SystemExit("authority source_video binding is absent")
path, digest = source.get("path"), source.get("sha256")
if not isinstance(path, str) or not path.startswith("/") or path == "/":
    raise SystemExit("authority source path differs")
if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise SystemExit("authority source SHA-256 differs")
print(path, digest, sep="\t")
PY
  )
  verify_file "$source_path" "$source_digest"
  printf '%s\t%s\t%s\t%s\n' "$authority_path" "$authority_digest" "$source_path" "$source_digest"
}

verify_static_authority() {
  local expected_runner_sha="$1"
  verify_file "$runtime_archive" "$runtime_archive_sha"
  verify_file "$runtime_manifest" "$runtime_manifest_sha"
  verify_file "$runner" "$expected_runner_sha"
  verify_file "$core" "$core_sha"
  verify_file "$python_bin" "$python_sha"
  verify_file "$unipc_source" "$unipc_source_sha"
  verify_file "$checkpoint_manifest" "$checkpoint_manifest_sha"
  for directory in "$bernini_root" "$veomni_root" "$checkpoint"; do
    [[ -d "$directory" && ! -L "$directory" ]] || fail "runtime directory authority differs: $directory"
  done
  verify_release
}

postflight_seed() {
  local candidate_arg="$1" case_arg="$2" seed_arg="$3" job_arg="$4" node_arg="$5" authority_arg="$6" source_arg="$7" runner_arg="$8"
  "${python_bin}" -B - "$candidate_arg" "$case_arg" "$seed_arg" "$job_arg" "$node_arg" "$authority_arg" "$source_arg" "$runner_arg" "$core_sha" <<'PY'
from pathlib import Path
import hashlib, json, stat, sys

root = Path(sys.argv[1])
case_id, seed, job, node, authority_sha, source_sha, runner_sha, core_sha = sys.argv[2], int(sys.argv[3]), *sys.argv[4:]
expected_names = {
    "p0a.mp4", "p1.mp4", "p2.mp4", "receipt.json",
    "source.normalized-clean-latent.safetensors",
    "p0a.normalized-clean-latent.safetensors", "p1.normalized-clean-latent.safetensors",
    "p2.normalized-clean-latent.safetensors", "p0b.normalized-clean-latent.safetensors",
    "p0a.official-initial-gaussian.safetensors", "p1.official-initial-gaussian.safetensors",
    "p2.official-initial-gaussian.safetensors", "p0b.official-initial-gaussian.safetensors",
}
if root.is_symlink() or not root.is_dir():
    raise SystemExit("candidate output directory differs")
actual = {p.name for p in root.iterdir()}
if actual != expected_names:
    raise SystemExit("candidate exact13 closure differs")
for path in root.iterdir():
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise SystemExit("candidate member is not a plain file: " + path.name)
receipt_path = root / "receipt.json"
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
declared = receipt.get("receipt_digest")
unsigned = dict(receipt)
unsigned.pop("receipt_digest", None)
raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
if not isinstance(declared, str) or hashlib.sha256(raw).hexdigest() != declared:
    raise SystemExit("receipt digest differs")
if receipt.get("case_id") != case_id:
    raise SystemExit("receipt case_id differs")
if receipt.get("scientific_claim_authorized") is not False:
    raise SystemExit("target-oracle diagnostic was mislabeled as a scientific claim")
mode = receipt.get("execution_mode", {})
if mode.get("seed") != seed or mode.get("num_inference_steps") != 40:
    raise SystemExit("receipt execution mode differs")
if (receipt.get("run_class") != "target_oracle_diagnostic"
        or mode.get("run_class") != "target_oracle_diagnostic"
        or mode.get("scientific_candidate") is not False
        or mode.get("formal_generation") is not False
        or mode.get("target_oracle_diagnostic") is not True
        or receipt.get("mechanical_gate") is not None):
    raise SystemExit("target-oracle diagnostic classification differs")
if receipt.get("execution_cells") != ["p0a", "p1", "p2", "p0b"]:
    raise SystemExit("receipt four-cell order differs")
inputs = receipt.get("input", {})
if inputs.get("case_authority_sha256") != authority_sha:
    raise SystemExit("receipt authority SHA-256 differs")
if inputs.get("source_video_sha256") != source_sha or inputs.get("accepted_external_conditions") != ["source_video", "positive_prompt_matrix"]:
    raise SystemExit("receipt source/input allowlist differs")
for key in ("target_video", "target_action_json", "target_rgb_mask_box_xy_flow_feature_embedding_latent_qkv_gaussian", "anchor_rgb_kv_latent_gaussian", "legacy_activity25_qk", "external_first_frame_anchor", "external_mask_flow_pose_track_trajectory", "external_reference_image_or_video"):
    if inputs.get(key) is not False:
        raise SystemExit("forbidden generator input admitted: " + key)
if receipt.get("freeze_certificate") != {"base_frozen": True, "lora_module_count": 0, "trainable_parameter_elements": 0, "trainable_parameter_tensors": 0}:
    raise SystemExit("freeze certificate differs")
pair = receipt.get("paired_same_process_contract", {})
slurm = pair.get("slurm", {})
if slurm.get("job_id") != job or slurm.get("node") != node or slurm.get("world_size") != 4:
    raise SystemExit("receipt Slurm binding differs")
if pair.get("execution_order") != ["p0a", "p1", "p2", "p0b"] or pair.get("target_media_or_action_json_read") is not False:
    raise SystemExit("same-process contract differs")
overlay = pair.get("current_authorized_overlay_runner", {})
if (overlay.get("sha256") != runner_sha
        or overlay.get("sampling_core_sha256") != core_sha
        or Path(overlay.get("path", "")).name != "infer_native_rv2v_crosscase_paired_prompt_matrix_formal_v1.py"
        or Path(overlay.get("sampling_core_path", "")).name != "infer_mev840_native_rv2v_paired_prompt_matrix_formal_v1.py"
        or overlay.get("upstream_release_entrypoint_authorized") is not False):
    raise SystemExit("wrapper/core overlay receipt differs")
p0 = pair.get("p0_replay", {})
if p0.get("generated_latent_bit_exact") is not True or p0.get("positive_tokens_and_embedding_bit_exact") is not True:
    raise SystemExit("P0 replay gate differs")
noise = receipt.get("initial_noise_artifacts", {})
identities = []
for cell in ("p0a", "p1", "p2", "p0b"):
    row = noise.get(cell, {})
    identities.append((row.get("content_sha256"), row.get("raw_value_sha256"), row.get("tensor_value_sha256"), tuple(row.get("shape", []))))
if len(set(identities)) != 1:
    raise SystemExit("same-seed official Gaussian differs")
outputs = receipt.get("outputs", {})
for cell in ("p0a", "p1", "p2"):
    row = outputs.get(cell, {})
    if (row.get("frame_count"), row.get("fps"), row.get("height"), row.get("width")) != (81, 25, 368, 656):
        raise SystemExit("decoded media metadata differs: " + cell)
print("MEV_CROSSCASE_POSTFLIGHT_OK", case_id, seed, declared)
PY
}

mode="${1:-}"
case "$mode" in
  launch-case)
    [[ $# == 4 ]] || usage
    readonly case_id="$2" job="$3" node="$4"
    validate_case_id "$case_id"
    validate_case_holder "$case_id" "$job" "$node"
    validate_holder_binding "$job" "$node"
    readonly runner_sha="$(read_sealed_pin "$runner_pin")"
    verify_static_authority "$runner_sha"
    IFS=$'\t' read -r authority authority_sha source_video source_sha < <(case_bindings "$case_id")
    [[ -d "$control_root/logs" && ! -L "$control_root/logs" ]] || fail "sealed log directory is absent"
    [[ -d "$output_root" && ! -L "$output_root" ]] || fail "output root must be deployed before launch"
    [[ ! -e "${output_root}/${case_id}" && ! -L "${output_root}/${case_id}" ]] || fail "fresh case output required"
    readonly log="${control_root}/logs/${case_id}.job${job}.log"
    readonly pid_file="${control_root}/logs/${case_id}.job${job}.pid"
    [[ ! -e "$log" && ! -L "$log" && ! -e "$pid_file" && ! -L "$pid_file" ]] || fail "fresh launch sidecars required"
    : >"$log"
    : >"$pid_file"
    nohup srun --jobid="$job" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=64 \
      --gres=gpu:4 --mem=0 --nodelist="$node" \
      env MEV_CROSSCASE_RUNNER_SHA="$runner_sha" MEV_CROSSCASE_AUTHORITY_SHA="$authority_sha" \
      MEV_CROSSCASE_SOURCE_SHA="$source_sha" \
      bash "$launcher" worker-case "$case_id" "$job" "$node" >"$log" 2>&1 &
    printf '%s\n' "$!" >"$pid_file"
    echo "MEV_CROSSCASE_LAUNCHED case=${case_id} job=${job} node=${node} runner_sha=${runner_sha} authority_sha=${authority_sha}"
    ;;
  worker-case)
    [[ $# == 4 ]] || usage
    readonly case_id="$2" job="$3" node="$4"
    validate_case_id "$case_id"
    validate_case_holder "$case_id" "$job" "$node"
    [[ "${SLURM_JOB_ID:-}" == "$job" && "$(hostname -s)" == "$node" && "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "worker Slurm authority differs"
    readonly runner_sha="${MEV_CROSSCASE_RUNNER_SHA:-}"
    readonly captured_authority_sha="${MEV_CROSSCASE_AUTHORITY_SHA:-}"
    readonly captured_source_sha="${MEV_CROSSCASE_SOURCE_SHA:-}"
    [[ "$runner_sha" =~ ^[0-9a-f]{64}$ && "$captured_authority_sha" =~ ^[0-9a-f]{64}$ && "$captured_source_sha" =~ ^[0-9a-f]{64}$ ]] || fail "captured deployment pins are absent"
    [[ "$(read_sealed_pin "$runner_pin")" == "$runner_sha" ]] || fail "runner pin changed after launch"
    verify_static_authority "$runner_sha"
    IFS=$'\t' read -r authority authority_sha source_video source_sha < <(case_bindings "$case_id")
    [[ "$authority_sha" == "$captured_authority_sha" && "$source_sha" == "$captured_source_sha" ]] || fail "case pins changed after launch"
    [[ -d "$output_root" && ! -L "$output_root" && ! -e "${output_root}/${case_id}" && ! -L "${output_root}/${case_id}" ]] || fail "fresh case output required"
    mkdir "${output_root}/${case_id}"

    scratch_parent="${SLURM_TMPDIR:-/tmp}"
    [[ "$scratch_parent" == /* && "$scratch_parent" != / && -d "$scratch_parent" && ! -L "$scratch_parent" && -w "$scratch_parent" ]] || fail "scratch parent differs"
    scratch="$(mktemp -d "${scratch_parent%/}/mev-crosscase-${case_id}-${SLURM_STEP_ID}.XXXXXXXX")"
    cleanup() {
      local status=$?
      trap - EXIT INT TERM HUP
      find "$scratch" -xdev -depth -mindepth 1 -delete || status=70
      rmdir "$scratch" || status=70
      exit "$status"
    }
    trap cleanup EXIT INT TERM HUP
    mkdir "$scratch/runtime" "$scratch/cache" "$scratch/tmp"
    tar -xf "$runtime_archive" -C "$scratch/runtime" --no-same-owner
    verify_release "$scratch/runtime"
    readonly runtime_method="$scratch/runtime/methods/bernini_action_editing"
    readonly runtime_runner="${runtime_method}/${runner_name}"
    readonly runtime_core="${runtime_method}/${core_name}"
    cp -- "$runner" "$runtime_runner"
    cp -- "$core" "$runtime_core"
    verify_file "$runtime_runner" "$runner_sha"
    verify_file "$runtime_core" "$core_sha"
    "${python_bin}" -B - "$runtime_method" "$runtime_manifest" "$runner_name" "$runner_sha" "$core_name" "$core_sha" <<'PY'
from pathlib import Path
import hashlib, json, stat, sys

root, manifest = Path(sys.argv[1]), Path(sys.argv[2])
wrapper_name, wrapper_sha, core_name, core_sha = sys.argv[3:]
expected = {row["path"]: row["sha256"] for row in json.loads(manifest.read_text(encoding="ascii"))["files"]}
if wrapper_name in expected or core_name in expected or wrapper_name == core_name:
    raise SystemExit("crosscase overlay collides with exact19 runtime")
expected[wrapper_name] = wrapper_sha
expected[core_name] = core_sha
actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() or path.is_symlink()}
if len(expected) != 21 or actual != set(expected):
    raise SystemExit("authorized scratch exact21 member closure differs")
for relative, digest in expected.items():
    path = root / relative
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit("authorized scratch member differs: " + relative)
print("MEV_CROSSCASE_EXACT21_OK")
PY
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
    "${python_bin}" -B -c 'import py_compile,sys; py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)' "$runtime_runner" "$scratch/cache/runner.pyc"
    "${python_bin}" -B -c 'import py_compile,sys; py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)' "$runtime_core" "$scratch/cache/core.pyc"
    model_load_lock="$scratch/renderer-load.lock"
    : >"$model_load_lock"
    chmod 0400 "$model_load_lock"
    export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1 NATIVE_V_AXIS_LOAD_LOCK="$model_load_lock"
    unset NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1
    export PYTHONPATH="$runtime_method" TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache/xdg" TORCH_EXTENSIONS_DIR="$scratch/cache/torch-extensions" TRITON_CACHE_DIR="$scratch/cache/triton"
    export MIOPEN_USER_DB_PATH="$scratch/cache/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/cache/miopen-custom"
    mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"

    for seed in 2028 2027; do
      candidate_dir="${output_root}/${case_id}/seed${seed}"
      [[ ! -e "$candidate_dir" && ! -L "$candidate_dir" ]] || fail "fresh seed output required: $candidate_dir"
      "${python_bin}" -B -m torch.distributed.run --standalone --nproc_per_node=4 "$runtime_runner" \
        --bernini-root "$bernini_root" --veomni-root "$veomni_root" \
        --checkpoint "$checkpoint" --checkpoint-content-manifest "$checkpoint_manifest" \
        --case-authority "$authority" --expected-case-authority-sha256 "$authority_sha" \
        --output-dir "$candidate_dir" --seed "$seed" \
        --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793 \
        --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d \
        --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca \
        --method-source-revision "$content_revision" --method-source-archive-sha256 "$runtime_archive_sha"
      postflight_seed "$candidate_dir" "$case_id" "$seed" "$job" "$node" "$authority_sha" "$source_sha" "$runner_sha"
      echo "MEV_CROSSCASE_SEED_DONE case=${case_id} seed=${seed} step=${SLURM_JOB_ID}.${SLURM_STEP_ID}"
    done
    echo "MEV_CROSSCASE_FORMAL_WORKER_DONE case=${case_id} job=${job} node=${node} seeds=2028,2027 run_class=target_oracle_diagnostic scientific_claim=false step=${SLURM_JOB_ID}.${SLURM_STEP_ID}"
    ;;
  *) usage ;;
esac

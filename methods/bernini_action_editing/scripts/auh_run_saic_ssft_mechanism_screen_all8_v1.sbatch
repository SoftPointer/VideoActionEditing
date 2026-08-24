#!/usr/bin/env bash
#SBATCH --job-name=saic-ssft-mech-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-289,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

# Formal, pre-registered T1/IAVG/I1/I1A mechanism screen.  Every wave uses all
# eight allocated MI210s as two legal WORLD4/Ulysses-SP4 groups.  The source
# clean coordinate is inherited read-only from job132387.  Each source frame0
# coordinate is freshly materialized exactly once before the four waves.

fail() { echo "[saic-ssft-mech-v1] ERROR: $*" >&2; exit 2; }
hash_file() { sha256sum -- "$1" | awk '{print $1}'; }
check_hash() { [[ "$(hash_file "$1")" == "$2" ]] || fail "$3 SHA-256 differs"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be a full lowercase Git SHA-1"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be a lowercase SHA-256"; }
require_absolute_nonroot() { [[ "$1" == /* && "$1" != / ]] || fail "$2 must be absolute and non-root"; }

[[ "${SAIC_SSFT_MECH_ACK_FROZEN_DIAGNOSTIC_ONLY:-NO}" == YES ]] || \
  fail "set SAIC_SSFT_MECH_ACK_FROZEN_DIAGNOSTIC_ONLY=YES"
[[ "${SAIC_SSFT_MECH_ACK_ZERO_SELECTION_TRAINING_OPTIMIZER:-NO}" == YES ]] || \
  fail "set SAIC_SSFT_MECH_ACK_ZERO_SELECTION_TRAINING_OPTIMIZER=YES"
[[ "${SAIC_SSFT_MECH_ACK_REUSE_ONLY_JOB132387_SOURCE_COORDINATE:-NO}" == YES ]] || \
  fail "set SAIC_SSFT_MECH_ACK_REUSE_ONLY_JOB132387_SOURCE_COORDINATE=YES"

source_archive="${SAIC_SSFT_MECH_SOURCE_ARCHIVE:?set immutable scoped Git archive}"
source_archive_sha256="${SAIC_SSFT_MECH_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${SAIC_SSFT_MECH_SOURCE_REVISION:?set archive Git revision}"
output_root="${SAIC_SSFT_MECH_OUTPUT_ROOT:?set fresh versioned output root}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set pinned Bernini-R checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
python_bin="${SAIC_SSFT_MECH_PYTHON_BIN:?set vace Python executable}"

readonly pinned_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly pinned_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly pinned_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly pinned_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly pinned_source_manifest_sha256=899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9
readonly pinned_event_bank_sha256=623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927
readonly pinned_ffprobe=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe
readonly pinned_ffprobe_sha256=356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5
readonly pinned_ffprobe_version_stdout_sha256=2271b81138bdaf07532b801ac7abd5b48d9e84dd66a6287a82fb44bc04c84f6b
readonly pinned_ffprobe_version_first_line='ffprobe version 9.0 Copyright (c) 2007-2026 the FFmpeg developers'
readonly old_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/ssft-t0-i0-ec4bfb6-j6
readonly source_manifest="${old_root}/sealed-inputs/saic_reversible_source_set_v1.json"
readonly event_bank="${old_root}/sealed-inputs/saic_pure_t2v_event_bank_v1.json"
readonly dog_row=fit-dog-00-7b88a1ca1f804f41
readonly dog_seed=2026082101
readonly dog_source_video_sha256=4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed
readonly dog_source_clean="${old_root}/groups/dog/source.clean-latent.safetensors"
readonly dog_source_clean_sha256=e3560e77546d3936f7e7231d5aceb78b8f29ea379b98748c26e4b37a5f277d7a
readonly dog_source_clean_receipt="${dog_source_clean}.receipt.json"
readonly dog_source_clean_receipt_sha256=bf1a817b6ab2f415311c19b0111f90c7a5537b3243df135d5fbcba06f63c37f1
readonly dog_source_clean_tensor_sha256=86c6b10800cfee13a1c3ca95b388a6ce8f310af6cbb84eea462728d8cd046343
readonly dog_job132387_frame0_tensor_sha256=995ac093c20fd07ef7018bdf942a1d7acb971c357effa53d4fddf5ac9e4e1f14
readonly human_row=fit-human-00-a35b590961d24694
readonly human_seed=2026082121
readonly human_source_video_sha256=6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed
readonly human_source_clean="${old_root}/groups/human/source.clean-latent.safetensors"
readonly human_source_clean_sha256=9cab2ac419833a3d8451b24c578c3ca3b466341f7bc17ea40e4067c8ebf6b7e9
readonly human_source_clean_receipt="${human_source_clean}.receipt.json"
readonly human_source_clean_receipt_sha256=ffdc8e4bce3181e0328f0714289b9d03d3c25ff5ac4ae57357d984101ddae0ba
readonly human_source_clean_tensor_sha256=2f7333434783bf681005c62200877aca1f95a395f21f228a2e83689ea0e1b263
readonly human_job132387_frame0_tensor_sha256=a6ce585d73bfa7a5e607356c9c06c25ec4f5e1d6f8c3b5e2c83d685b6910c6e1
readonly arms=(T1 IAVG I1 I1A)

for name in source_archive output_root bernini_root veomni_root checkpoint checkpoint_manifest python_bin pinned_ffprobe; do
  require_absolute_nonroot "${!name}" "${name}"
done
require_sha1 "${source_revision}" source_revision
require_sha256 "${source_archive_sha256}" source_archive_sha256
for name in pinned_checkpoint_tree_sha256 pinned_checkpoint_manifest_sha256 \
  pinned_source_manifest_sha256 pinned_event_bank_sha256 \
  pinned_ffprobe_sha256 pinned_ffprobe_version_stdout_sha256 \
  dog_source_video_sha256 dog_source_clean_sha256 dog_source_clean_receipt_sha256 \
  dog_source_clean_tensor_sha256 dog_job132387_frame0_tensor_sha256 \
  human_source_video_sha256 human_source_clean_sha256 human_source_clean_receipt_sha256 \
  human_source_clean_tensor_sha256 human_job132387_frame0_tensor_sha256; do
  require_sha256 "${!name}" "${name}"
done

for path in "${source_archive}" "${checkpoint_manifest}" "${source_manifest}" \
  "${event_bank}" "${dog_source_clean}" "${dog_source_clean_receipt}" \
  "${human_source_clean}" "${human_source_clean_receipt}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "required plain file is absent: ${path}"
done
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "required plain directory is absent: ${path}"
done
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "python executable differs"
[[ -x "${pinned_ffprobe}" && -f "${pinned_ffprobe}" && ! -L "${pinned_ffprobe}" ]] || fail "pinned ffprobe executable differs"
check_hash "${source_archive}" "${source_archive_sha256}" source_archive
check_hash "${checkpoint_manifest}" "${pinned_checkpoint_manifest_sha256}" checkpoint_manifest
check_hash "${source_manifest}" "${pinned_source_manifest_sha256}" source_manifest
check_hash "${event_bank}" "${pinned_event_bank_sha256}" event_bank
check_hash "${dog_source_clean}" "${dog_source_clean_sha256}" dog_source_clean
check_hash "${dog_source_clean_receipt}" "${dog_source_clean_receipt_sha256}" dog_source_clean_receipt
check_hash "${human_source_clean}" "${human_source_clean_sha256}" human_source_clean
check_hash "${human_source_clean_receipt}" "${human_source_clean_receipt_sha256}" human_source_clean_receipt
check_hash "${pinned_ffprobe}" "${pinned_ffprobe_sha256}" pinned_ffprobe
[[ "$("${pinned_ffprobe}" -version | sha256sum | awk '{print $1}')" == "${pinned_ffprobe_version_stdout_sha256}" ]] || fail "pinned ffprobe version output differs"
[[ "$("${pinned_ffprobe}" -version | sed -n '1p')" == "${pinned_ffprobe_version_first_line}" ]] || fail "pinned ffprobe version line differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive Git revision differs"
[[ "${SLURM_JOB_ID:?launcher requires Slurm job ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"

output_parent="$(dirname -- "${output_root}")"
output_name="${output_root##*/}"
[[ "${output_parent}" != / && -d "${output_parent}" && ! -L "${output_parent}" && -w "${output_parent}" ]] || fail "output parent differs"
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output root is not canonical"
[[ "${output_name}" =~ ^ssft-t1-iavg-i1-i1a-v1-[0-9a-f]{12}-r[0-9]+$ ]] || fail "output release name differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "fresh output root already exists"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
scratch_parent="$(realpath -e -- "${scratch_parent}")"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-ssft-mech-v1-${SLURM_JOB_ID}.XXXXXX")"
task_scratch_identity="$(stat -c '%d:%i' -- "${task_scratch}")"
published=false
owned_output=false
output_identity=""
dog_pid=""
human_pid=""

terminate_children() {
  local status="$1" child
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do
    [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true
  done
  for child in "${dog_pid}" "${human_pid}"; do
    [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true
  done
  exit "${status}"
}

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  if [[ "${published}" != true && "${owned_output}" == true && -d "${output_root}" && ! -L "${output_root}" ]]; then
    [[ "${output_root}" == "${output_parent}/${output_name}" ]] || cleanup_failed=1
    [[ "${output_identity}" =~ ^[0-9]+:[0-9]+$ ]] || cleanup_failed=1
    [[ "$(stat -c '%d:%i' -- "${output_root}")" == "${output_identity}" ]] || cleanup_failed=1
    if [[ "${cleanup_failed}" == 0 ]]; then
      chmod -R u+w -- "${output_root}" || cleanup_failed=1
      rm -rf -- "${output_root}" || cleanup_failed=1
    fi
  fi
  case "${task_scratch}" in
    "${scratch_parent%/}/saic-ssft-mech-v1-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" && "$(stat -c '%d:%i' -- "${task_scratch}")" == "${task_scratch_identity}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    [[ "${cleanup_failed}" != 0 ]] || rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  if [[ "${status}" == 0 && "${cleanup_failed}" == 0 && "${published}" == true ]]; then
    echo "[saic-ssft-mech-v1] PASS arms=T1,IAVG,I1,I1A exact81=true fps=25 exact40=true source_job=132387 frame0_materializations=2 consumers=8 training=false optimizer=false semantic_authority=false output=${output_root}"
  else
    [[ "${status}" != 0 ]] || status=2
    echo "[saic-ssft-mech-v1] EXIT status=${status} published=${published} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -- "${task_scratch}/source-tree" "${task_scratch}/logs" "${task_scratch}/runtime"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
check_hash "${archive_copy}" "${source_archive_sha256}" staged_archive
chmod 0444 -- "${archive_copy}"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
    "methods/bernini_action_editing/infer_saic_source_state_flow_transport_v2.py",
    "methods/bernini_action_editing/materialize_saic_frame0_latent_v1.py",
    "methods/bernini_action_editing/infer_saic_ssft_mechanism_screen_v1.py",
    "methods/bernini_action_editing/release_saic_ssft_mechanism_screen_v1.py",
    "methods/bernini_action_editing/assets/saic_ssft_mechanism_screen_v1.json",
    "methods/bernini_action_editing/scripts/auh_run_saic_ssft_mechanism_screen_all8_v1.sbatch",
    "methods/bernini_action_editing/scripts/auh_submit_saic_ssft_mechanism_screen_v1.sh",
    "methods/bernini_action_editing/tests/test_infer_saic_source_state_flow_transport_v2.py",
    "methods/bernini_action_editing/tests/test_materialize_saic_frame0_latent_v1.py",
    "methods/bernini_action_editing/tests/test_saic_ssft_mechanism_screen_v1.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        name = path.as_posix().rstrip("/")
        scoped = name in {"methods", "methods/bernini_action_editing"} or name.startswith("methods/bernini_action_editing/")
        if (not name or path.is_absolute() or ".." in path.parts or not scoped or name in seen
                or member.issym() or member.islnk() or member.isdev() or member.isfifo()
                or (not member.isfile() and not member.isdir())):
            raise SystemExit("unsafe/duplicate/out-of-scope archive member")
        seen.add(name)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks mechanism-screen closure: {missing}")
PY
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree" methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runner="${method_root}/infer_saic_ssft_mechanism_screen_v1.py"
materializer="${method_root}/materialize_saic_frame0_latent_v1.py"
release_auditor="${method_root}/release_saic_ssft_mechanism_screen_v1.py"
launcher="${method_root}/scripts/auh_run_saic_ssft_mechanism_screen_all8_v1.sbatch"
plan="${method_root}/assets/saic_ssft_mechanism_screen_v1.json"
[[ "$(hash_file "${BASH_SOURCE[0]}")" == "$(hash_file "${launcher}")" ]] || fail "running launcher differs from archive"
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +
find "${task_scratch}/source-tree" -type d -exec chmod a-w -- {} +

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

# The same hostile/contract suite is also run independently before submission.
# Re-run it inside the allocation before output creation, then prove vendor,
# checkpoint, plan, source-manifest, and inherited-coordinate closure.
for test_file in \
  test_materialize_saic_frame0_latent_v1.py \
  test_infer_saic_source_state_flow_transport_v2.py \
  test_saic_ssft_mechanism_screen_v1.py; do
  PYTHONPATH="${method_root}:${bernini_root}:${veomni_root}" \
    "${python_bin}" -B "${method_root}/tests/${test_file}"
done
PYTHONPATH="${method_root}:${bernini_root}:${veomni_root}" \
  "${python_bin}" -B - "${bernini_root}" "${veomni_root}" "${checkpoint}" \
  "${checkpoint_manifest}" "${plan}" "${source_revision}" "${source_archive_sha256}" <<'PY'
import hashlib, json, sys
from pathlib import Path
import infer_saic_ssft_mechanism_screen_v1 as runner
import infer_source_kv_carrier_oracle as source_audit
import train_lora

bernini, veomni, checkpoint, manifest, plan_path, revision, archive_sha = sys.argv[1:]
_, config = train_lora.validate_checkpoint(checkpoint)
if int(config.get("num_attention_heads", -1)) != 12 or 12 % 4:
    raise SystemExit("Bernini-R SP4 checkpoint closure differs")
_, _, br, vr = train_lora.validate_source_trees(
    bernini, veomni,
    expected_bernini_commit=train_lora.BERNINI_OFFICIAL_COMMIT,
    expected_veomni_commit=train_lora.VEOMNI_TESTED_COMMIT,
)
source_audit.validate_checkpoint_content(Path(checkpoint), Path(manifest))
plan = json.loads(Path(plan_path).read_text())
if tuple(runner.ARM_NAMES) != ("T1", "IAVG", "I1", "I1A"):
    raise SystemExit("mechanism arm registry differs")
if [x.arm for x in runner.ARM_SPECS.values()] != list(runner.ARM_NAMES):
    raise SystemExit("mechanism arm mapping order differs")
if runner._impl.RUNTIME_METHOD_FILES[-1] != "infer_saic_ssft_mechanism_screen_v1.py":
    raise SystemExit("runtime closure lacks mechanism wrapper")
deps = plan["implementation_dependencies"]
for name, expected in deps.items():
    actual = hashlib.sha256((Path(plan_path).parent.parent / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"pinned implementation dependency differs: {name}")
if (br, vr) != (runner._impl.legacy.trainer.BERNINI_OFFICIAL_COMMIT, runner._impl.legacy.trainer.VEOMNI_TESTED_COMMIT):
    raise SystemExit("vendor revision closure differs")
if len(revision) != 40 or len(archive_sha) != 64:
    raise SystemExit("method provenance identifiers differ")
PY

mkdir -- "${output_root}"
owned_output=true
output_identity="$(stat -c '%d:%i' -- "${output_root}")"
[[ "${output_identity}" =~ ^[0-9]+:[0-9]+$ ]] || fail "output identity differs"
mkdir -- "${output_root}/groups" "${output_root}/groups/dog" "${output_root}/groups/human" "${output_root}/logs" "${output_root}/sealed-inputs"
durable_archive="${output_root}/method-source.tar"
durable_plan="${output_root}/sealed-inputs/saic_ssft_mechanism_screen_v1.json"
cp -- "${archive_copy}" "${durable_archive}"
cp -- "${plan}" "${durable_plan}"
chmod 0444 -- "${durable_archive}" "${durable_plan}"
check_hash "${durable_archive}" "${source_archive_sha256}" durable_archive

dog_frame0="${output_root}/groups/dog/frame0.latent.safetensors"
human_frame0="${output_root}/groups/human/frame0.latent.safetensors"

materialize_frame0() (
  set -Eeuo pipefail
  local group="$1" row="$2" source_sha="$3" expected_raw="$4" visible="$5" output="$6"
  local scratch="${task_scratch}/runtime/${group}/frame0"
  mkdir -p -- "${scratch}/miopen-user" "${scratch}/miopen-custom" "${scratch}/torch-extensions" "${scratch}/triton" "${scratch}/xdg" "${scratch}/pycache" "${scratch}/tmp"
  [[ ! -e "${output}" && ! -L "${output}" && ! -e "${output}.receipt.json" && ! -L "${output}.receipt.json" ]] || fail "frame0 output reuse forbidden"
  exec env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    ROCR_VISIBLE_DEVICES="${visible}" MIOPEN_USER_DB_PATH="${scratch}/miopen-user" \
    MIOPEN_CUSTOM_CACHE_DIR="${scratch}/miopen-custom" TORCH_EXTENSIONS_DIR="${scratch}/torch-extensions" \
    TRITON_CACHE_DIR="${scratch}/triton" XDG_CACHE_HOME="${scratch}/xdg" \
    PYTHONPYCACHEPREFIX="${scratch}/pycache" TMPDIR="${scratch}/tmp" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    PYTHONPATH="${method_root}:${bernini_root}:${veomni_root}" \
    "${python_bin}" -B "${materializer}" \
      --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --source-manifest "${source_manifest}" --expected-source-manifest-sha256 "${pinned_source_manifest_sha256}" \
      --row-id "${row}" --expected-source-video-sha256 "${source_sha}" \
      --expected-reference-frame0-tensor-raw-sha256 "${expected_raw}" \
      --expected-bernini-commit "${pinned_bernini_commit}" --expected-veomni-commit "${pinned_veomni_commit}" \
      --expected-checkpoint-tree-sha256 "${pinned_checkpoint_tree_sha256}" \
      --method-source-revision "${source_revision}" --method-source-archive "${archive_copy}" \
      --durable-method-source-archive "${durable_archive}" --method-source-archive-sha256 "${source_archive_sha256}" \
      --device cuda:0 --output "${output}"
)

materialize_frame0 dog "${dog_row}" "${dog_source_video_sha256}" "${dog_job132387_frame0_tensor_sha256}" 0 "${dog_frame0}" >"${output_root}/logs/frame0.dog.log" 2>&1 &
dog_pid=$!
materialize_frame0 human "${human_row}" "${human_source_video_sha256}" "${human_job132387_frame0_tensor_sha256}" 4 "${human_frame0}" >"${output_root}/logs/frame0.human.log" 2>&1 &
human_pid=$!
dog_status=0; human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
dog_pid=""; human_pid=""
if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
  tail -n 240 -- "${output_root}/logs/frame0.dog.log" >&2 || true
  tail -n 240 -- "${output_root}/logs/frame0.human.log" >&2 || true
  fail "frame0 materialization failed dog=${dog_status} human=${human_status}"
fi

mapfile -t frame0_bindings < <(
  "${python_bin}" -B - \
    dog "${dog_frame0}" "${dog_frame0}.receipt.json" "${dog_job132387_frame0_tensor_sha256}" \
    human "${human_frame0}" "${human_frame0}.receipt.json" "${human_job132387_frame0_tensor_sha256}" <<'PY'
from pathlib import Path
import hashlib, json, stat, sys

def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

for offset in (1, 5):
    group, artifact_text, receipt_text, old_raw = sys.argv[offset:offset + 4]
    artifact, receipt_path = Path(artifact_text), Path(receipt_text)
    for path in (artifact, receipt_path):
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
            raise SystemExit("frame0 materialization is not a sealed plain file")
    payload = receipt_path.read_bytes()
    receipt = json.loads(payload.decode("ascii"))
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    if payload != canonical + b"\n":
        raise SystemExit("frame0 receipt is not canonical")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    unsigned_payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    if hashlib.sha256(unsigned_payload).hexdigest() != declared:
        raise SystemExit("frame0 receipt digest differs")
    section, encoding = receipt.get("artifact", {}), receipt.get("encoding", {})
    artifact_sha, receipt_sha = file_sha(artifact), file_sha(receipt_path)
    actual_raw = section.get("tensor_raw_sha256")
    if (
        receipt.get("schema_version") != "bernini-saic-frame0-latent-receipt-v1"
        or section.get("path") != str(artifact.resolve())
        or section.get("file_sha256") != artifact_sha
        or encoding.get("actual_reference_frame0_tensor_raw_sha256") != actual_raw
        or encoding.get("expected_job132387_frame0_tensor_raw_sha256") != old_raw
        or encoding.get("job132387_frame0_tensor_raw_sha256_match") is not (actual_raw == old_raw)
        or not isinstance(actual_raw, str) or len(actual_raw) != 64
    ):
        raise SystemExit("frame0 receipt binding differs")
    print("\t".join((group, artifact_sha, receipt_sha, actual_raw, str(actual_raw == old_raw).lower())))
PY
)
[[ "${#frame0_bindings[@]}" == 2 ]] || fail "frame0 binding count differs"
IFS=$'\t' read -r dog_frame0_group dog_frame0_sha256 dog_frame0_receipt_sha256 dog_frame0_tensor_sha256 dog_frame0_matches_job132387 <<<"${frame0_bindings[0]}"
IFS=$'\t' read -r human_frame0_group human_frame0_sha256 human_frame0_receipt_sha256 human_frame0_tensor_sha256 human_frame0_matches_job132387 <<<"${frame0_bindings[1]}"
[[ "${dog_frame0_group}" == dog && "${human_frame0_group}" == human ]] || fail "frame0 binding order differs"
[[ "${dog_frame0_matches_job132387}" == true || "${dog_frame0_matches_job132387}" == false ]] || fail "dog frame0 comparison differs"
[[ "${human_frame0_matches_job132387}" == true || "${human_frame0_matches_job132387}" == false ]] || fail "human frame0 comparison differs"
for name in dog_frame0_sha256 dog_frame0_receipt_sha256 dog_frame0_tensor_sha256 \
  human_frame0_sha256 human_frame0_receipt_sha256 human_frame0_tensor_sha256; do
  require_sha256 "${!name}" "${name}"
  readonly "${name}=${!name}"
done
readonly dog_frame0_matches_job132387 human_frame0_matches_job132387
echo "[saic-ssft-mech-v1] fresh frame0 comparison dog_matches_job132387=${dog_frame0_matches_job132387} human_matches_job132387=${human_frame0_matches_job132387}"

job_mod=$((SLURM_JOB_ID % 20000))
dog_port=$((20000 + 2 * job_mod))
human_port=$((dog_port + 1))

run_cell() (
  set -Eeuo pipefail
  local group="$1" row="$2" seed="$3" arm="$4" visible="$5" port="$6" output="$7"
  local source_clean="$8" source_receipt="$9" source_sha="${10}" source_receipt_sha="${11}" source_raw="${12}"
  local frame0="${13}" frame0_receipt="${14}" frame0_sha="${15}" frame0_receipt_sha="${16}" frame0_raw="${17}"
  local scratch="${task_scratch}/runtime/${group}"
  mkdir -p -- "${scratch}/miopen-user" "${scratch}/miopen-custom" "${scratch}/torch-extensions" "${scratch}/triton" "${scratch}/xdg" "${scratch}/pycache" "${scratch}/tmp"
  [[ ! -e "${output}" && ! -L "${output}" && ! -e "${output}.receipt.json" && ! -L "${output}.receipt.json" ]] || fail "runner output reuse forbidden"
  check_hash "${source_clean}" "${source_sha}" source_before_consumer
  check_hash "${source_receipt}" "${source_receipt_sha}" source_receipt_before_consumer
  check_hash "${frame0}" "${frame0_sha}" frame0_before_consumer
  check_hash "${frame0_receipt}" "${frame0_receipt_sha}" frame0_receipt_before_consumer
  exec env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    ROCR_VISIBLE_DEVICES="${visible}" MIOPEN_USER_DB_PATH="${scratch}/miopen-user" \
    MIOPEN_CUSTOM_CACHE_DIR="${scratch}/miopen-custom" TORCH_EXTENSIONS_DIR="${scratch}/torch-extensions" \
    TRITON_CACHE_DIR="${scratch}/triton" XDG_CACHE_HOME="${scratch}/xdg" \
    PYTHONPYCACHEPREFIX="${scratch}/pycache" TMPDIR="${scratch}/tmp" \
    TORCHELASTIC_ERROR_FILE="${scratch}/torch-elastic-${arm}.json" \
    "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
      --master_addr=127.0.0.1 --master_port="${port}" "${runner}" \
      --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --source-manifest "${source_manifest}" --expected-source-manifest-sha256 "${pinned_source_manifest_sha256}" \
      --event-bank "${event_bank}" --expected-event-bank-sha256 "${pinned_event_bank_sha256}" \
      --row-id "${row}" --branch forward --rollout-seed "${seed}" --arm "${arm}" \
      --source-clean-latent "${source_clean}" --source-clean-latent-receipt "${source_receipt}" \
      --expected-source-clean-latent-sha256 "${source_sha}" --expected-source-clean-latent-receipt-sha256 "${source_receipt_sha}" \
      --expected-source-clean-tensor-raw-sha256 "${source_raw}" \
      --reference-frame0-latent "${frame0}" --reference-frame0-latent-receipt "${frame0_receipt}" \
      --expected-reference-frame0-latent-sha256 "${frame0_sha}" --expected-reference-frame0-latent-receipt-sha256 "${frame0_receipt_sha}" \
      --expected-reference-frame0-tensor-raw-sha256 "${frame0_raw}" \
      --output "${output}" --expected-bernini-commit "${pinned_bernini_commit}" \
      --expected-veomni-commit "${pinned_veomni_commit}" --expected-checkpoint-tree-sha256 "${pinned_checkpoint_tree_sha256}" \
      --method-source-revision "${source_revision}" --method-source-archive "${archive_copy}" \
      --durable-method-source-archive "${durable_archive}" --method-source-archive-sha256 "${source_archive_sha256}"
)

run_wave() {
  local arm="$1"
  run_cell dog "${dog_row}" "${dog_seed}" "${arm}" "0,1,2,3" "${dog_port}" "${output_root}/groups/dog/${arm}.mp4" \
    "${dog_source_clean}" "${dog_source_clean_receipt}" "${dog_source_clean_sha256}" "${dog_source_clean_receipt_sha256}" "${dog_source_clean_tensor_sha256}" \
    "${dog_frame0}" "${dog_frame0}.receipt.json" "${dog_frame0_sha256}" "${dog_frame0_receipt_sha256}" "${dog_frame0_tensor_sha256}" \
    >"${output_root}/logs/${arm}.dog.log" 2>&1 &
  dog_pid=$!
  run_cell human "${human_row}" "${human_seed}" "${arm}" "4,5,6,7" "${human_port}" "${output_root}/groups/human/${arm}.mp4" \
    "${human_source_clean}" "${human_source_clean_receipt}" "${human_source_clean_sha256}" "${human_source_clean_receipt_sha256}" "${human_source_clean_tensor_sha256}" \
    "${human_frame0}" "${human_frame0}.receipt.json" "${human_frame0_sha256}" "${human_frame0_receipt_sha256}" "${human_frame0_tensor_sha256}" \
    >"${output_root}/logs/${arm}.human.log" 2>&1 &
  human_pid=$!
  local dog_status=0 human_status=0
  wait "${dog_pid}" || dog_status=$?
  wait "${human_pid}" || human_status=$?
  dog_pid=""; human_pid=""
  if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
    tail -n 240 -- "${output_root}/logs/${arm}.dog.log" >&2 || true
    tail -n 240 -- "${output_root}/logs/${arm}.human.log" >&2 || true
    fail "WORLD4/SP4 wave ${arm} failed dog=${dog_status} human=${human_status}"
  fi
}

echo "[saic-ssft-mech-v1] topology=dog:0-3+human:4-7 fixed_arms=T1,IAVG,I1,I1A"
for arm in "${arms[@]}"; do
  run_wave "${arm}"
done

PYTHONPATH="${method_root}:${bernini_root}:${veomni_root}" \
  "${python_bin}" -B "${release_auditor}" --plan "${durable_plan}" \
    --output-root "${output_root}" --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}" --job-id "${SLURM_JOB_ID}" \
    --ffprobe-bin "${pinned_ffprobe}" --ffprobe-sha256 "${pinned_ffprobe_sha256}" \
    --ffprobe-version-stdout-sha256 "${pinned_ffprobe_version_stdout_sha256}" \
    --ffprobe-version-first-line "${pinned_ffprobe_version_first_line}"

[[ -f "${output_root}/mechanism-screen-parent-receipt.json" && ! -L "${output_root}/mechanism-screen-parent-receipt.json" ]] || fail "parent receipt missing"
[[ "$(stat -c '%a' -- "${output_root}/mechanism-screen-parent-receipt.json")" == 444 ]] || fail "parent receipt mode differs"
find "${output_root}" -type f -exec chmod a-w -- {} +
find "${output_root}" -type d -exec chmod a-w -- {} +
published=true

#!/usr/bin/env bash
# BOX-EXP-014 source-only exact7 VAE re-encode on retained 136141/gpu299.
# One numbered one-GPU child is created only after explicit confirmation.  The
# script never cancels, releases, or requeues the retained parent allocation.

set -Eeuo pipefail
umask 077

fail() { echo "[full30-source7-reencode-r4] ERROR: $*" >&2; return 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
path_is_equal_or_descendant() { [[ "$1" == "$2" || "$1" == "$2/"* ]]; }

readonly confirm="${S7_CONFIRM:?explicit BOX-EXP-014 launch confirmation required}"
readonly run_root="${S7_RUN_ROOT:?set one fresh run root}"
readonly method_archive="${S7_METHOD_ARCHIVE:?set sealed source7 release archive}"
readonly method_archive_sha="${S7_METHOD_ARCHIVE_SHA256:?pin source7 release archive}"
readonly method_manifest="${S7_METHOD_MANIFEST:?set canonical source7 release manifest}"
readonly method_manifest_sha="${S7_METHOD_MANIFEST_SHA256:?pin source7 release manifest}"

readonly holder_job=136141
readonly holder_node=auh7-1b-gpu-299
readonly holder_user=guangyi.chen
readonly run_generation=r4
readonly launch_confirmation=launch-approved-BOX-EXP-014-source-only-exact7-reencode-r4-136141
readonly failed_r1_run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/data_prep/full30-action-source7-reencode-r1-a5f7e159-j136141-r1
readonly failed_r2_run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/data_prep/full30-action-source7-reencode-r2-4f188c71-j136141-r1
readonly failed_r1_release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/full30-action-source7-reencode-r1-a5f7e159-20260815-r1
readonly failed_r2_release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/full30-action-source7-reencode-r2-4f188c71-20260815-r1
readonly failed_r3_run_prefix=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/data_prep/full30-action-source7-reencode-r3-
readonly failed_r3_release_prefix=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/full30-action-source7-reencode-r3-
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831

case "${1:-}" in
  "") readonly role=parent ;;
  __child) [[ $# == 1 ]] || fail "child takes no extra arguments"; readonly role=child ;;
  *) fail "launcher arguments differ" ;;
esac

[[ "${confirm}" == "${launch_confirmation}" ]] || fail "launch confirmation differs"
for digest in method_archive_sha method_manifest_sha checkpoint_manifest_sha; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} is not SHA-256"
done
for name in run_root method_archive method_manifest python_bin checkpoint checkpoint_manifest; do
  value="${!name}"
  [[ "${value}" == /* && "${value}" != / ]] || fail "${name} must be a scoped absolute path"
  [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || fail "${name} contains a forbidden line break"
  [[ "$(realpath -m -- "${value}")" == "${value}" ]] || fail "${name} must already be canonical and may not traverse a symlink or dot component"
done
[[ "${run_root}" == /vast/users/guangyi.chen/* ]] || fail "run root scope differs"
path_is_equal_or_descendant "${run_root}" "${failed_r1_run_root}" && fail "failed r1 run root or descendant is permanently forbidden"
path_is_equal_or_descendant "${run_root}" "${failed_r2_run_root}" && fail "failed r2 run root or descendant is permanently forbidden"
[[ "${run_root}" != "${failed_r3_run_prefix}"* ]] || fail "failed r3 run generation is permanently forbidden"
for release_path in "${method_archive}" "${method_manifest}"; do
  path_is_equal_or_descendant "${release_path}" "${failed_r1_release_root}" && fail "failed r1 release or descendant is permanently forbidden"
  path_is_equal_or_descendant "${release_path}" "${failed_r2_release_root}" && fail "failed r2 release or descendant is permanently forbidden"
  [[ "${release_path}" != "${failed_r3_release_prefix}"* ]] || fail "failed r3 release generation is permanently forbidden"
done
for path in "${method_archive}" "${method_manifest}" "${python_bin}" "${checkpoint_manifest}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed file differs: ${path}"
done
[[ -x "${python_bin}" ]] || fail "Python differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" && "$(readlink -f -- "${checkpoint}")" == "${checkpoint}" ]] || fail "checkpoint differs"
[[ "$(sha256_file "${method_archive}")" == "${method_archive_sha}" ]] || fail "release archive SHA differs"
[[ "$(sha256_file "${method_manifest}")" == "${method_manifest_sha}" ]] || fail "release manifest SHA differs"
[[ "$(sha256_file "${checkpoint_manifest}")" == "${checkpoint_manifest_sha}" ]] || fail "checkpoint manifest SHA differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

readonly runtime_parent="${run_root}/runtime-source"
readonly method_root="${runtime_parent}/methods/bernini_action_editing"
readonly controller="${method_root}/full30_action_source7_reencode_controller_v1.py"
readonly release_builder="${method_root}/tools/build_full30_action_source7_reencode_release_v1.py"
readonly runtime_cache_tool="${method_root}/tools/full30_action_source7_reencode_runtime_cache_v1.py"
readonly launcher="${method_root}/scripts/auh_full30_action_source7_reencode_136141_v1.sh"

if [[ "${role}" == child ]]; then
  [[ "${SLURM_JOB_ID:?Slurm child is required}" == "${holder_job}" ]] || fail "child holder differs"
  [[ "$(hostname -s)" == "${holder_node}" ]] || fail "child node differs"
  [[ -f "${controller}" && ! -L "${controller}" ]] || fail "controller differs"
  [[ -f "${runtime_cache_tool}" && ! -L "${runtime_cache_tool}" ]] || fail "runtime cache tool differs"
  [[ -f "${method_manifest}" && ! -L "${method_manifest}" ]] || fail "release manifest differs in child"
  readonly child_job_token="${SLURM_JOB_ID}"
  readonly child_step_token="${SLURM_STEP_ID:?Slurm numbered step id is required}"
  [[ "${child_job_token}" =~ ^(0|[1-9][0-9]{0,19})$ ]] || fail "child job token differs"
  [[ "${child_step_token}" =~ ^(0|[1-9][0-9]{0,19})$ ]] || fail "child step token differs"
  readonly cache_root="/tmp/BOX-EXP-014-r4-${child_job_token}-${child_step_token}"
  readonly cache_prepare_receipt="${run_root}/logs/runtime-cache-prepare-r4.json"
  readonly cache_cleanup_receipt="${run_root}/logs/runtime-cache-cleanup-r4.json"
  readonly cache_retained_failure_receipt="${run_root}/logs/runtime-cache-retained-failure-r4.json"
  readonly cache_phase_failure_receipt="${run_root}/logs/runtime-cache-phase-failure-r4.json"
  readonly original_home="${HOME:?HOME must remain set}"
  export MIOPEN_USER_DB_PATH="${cache_root}/user-db"
  export MIOPEN_CUSTOM_CACHE_DIR="${cache_root}/kernel-cache"
  export XDG_CACHE_HOME="${cache_root}/xdg-cache"
  export TMPDIR="${cache_root}/tmp"
  child_phase=prepare-or-precontroller
  child_terminal_written=false
  record_child_phase_failure() {
    local failure_status="$1"
    trap - ERR
    set +e
    if [[ "${child_terminal_written}" != true ]]; then
      "${python_bin}" -I -S "${runtime_cache_tool}" record-phase-failure \
        --phase-failure-receipt-output "${cache_phase_failure_receipt}" \
        --phase "${child_phase}" \
        --failure-exit-status "${failure_status}" \
        --prepare-receipt "${cache_prepare_receipt}" \
        --controller-completion "${run_root}/controller-completion.json" \
        --cleanup-receipt "${cache_cleanup_receipt}" >&2
      local terminal_status=$?
      if (( terminal_status != 0 )); then
        echo "BOX-EXP-014_R4_PHASE_FAILURE_TERMINAL_WRITE_FAILED phase=${child_phase} original_status=${failure_status} terminal_status=${terminal_status}" >&2
        "${python_bin}" -I -S - \
          "${run_root}/logs/runtime-cache-phase-failure-r4.fallback" \
          "${child_phase}" "${failure_status}" <<'PY'
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
phase = sys.argv[2]
status = sys.argv[3]
raw = (
    "schema=bernini-full30-action-source7-reencode-phase-failure-shell-fallback-v1\n"
    "experiment_id=BOX-EXP-014\n"
    "run_generation=r4\n"
    f"phase={phase}\n"
    f"failure_exit_status={status}\n"
    "success_claimed=false\n"
    "final_marker_authorized=false\n"
    "cache_root_reusable=false\n"
).encode("ascii")
descriptor = os.open(
    path,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o600,
)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(raw)
    handle.flush()
    os.fsync(handle.fileno())
assert path.read_bytes() == raw
directory = os.open(
    path.parent,
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
      fi
    fi
    exit "${failure_status}"
  }
  trap 'record_child_phase_failure $?' ERR
  "${python_bin}" -I -S "${runtime_cache_tool}" prepare \
    --receipt-output "${cache_prepare_receipt}"
  readonly cache_prepare_sha="$(sha256_file "${cache_prepare_receipt}")"
  [[ -d "${cache_root}" && ! -L "${cache_root}" && "$(readlink -f -- "${cache_root}")" == "${cache_root}" ]] || fail "runtime cache root differs after prepare"
  [[ -d "${TMPDIR}" && ! -L "${TMPDIR}" && "$(readlink -f -- "${TMPDIR}")" == "${TMPDIR}" ]] || fail "scoped TMPDIR differs after prepare"
  [[ "${HOME}" == "${original_home}" ]] || fail "HOME changed during runtime cache preparation"
  child_phase=controller-or-retained-terminal
  trap - ERR
  set +e
  "${python_bin}" -B "${controller}" \
    --method-root "${method_root}" \
    --release-manifest "${method_manifest}" \
    --expected-release-manifest-sha256 "${method_manifest_sha}" \
    --plan-output "${run_root}/source7-plan.json" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha}" \
    --materialization-output-root "${run_root}/physical_source_posterior_index0_exact7" \
    --completion-output "${run_root}/controller-completion.json" \
    --runtime-cache-receipt "${cache_prepare_receipt}" \
    --expected-runtime-cache-receipt-sha256 "${cache_prepare_sha}"
  controller_status=$?
  set -e
  trap 'record_child_phase_failure $?' ERR
  if (( controller_status != 0 )); then
    "${python_bin}" -I -S "${runtime_cache_tool}" record-retained-failure \
      --prepare-receipt "${cache_prepare_receipt}" \
      --expected-prepare-receipt-sha256 "${cache_prepare_sha}" \
      --retained-failure-receipt-output "${cache_retained_failure_receipt}" \
      --controller-exit-status "${controller_status}"
    readonly cache_retained_failure_sha="$(sha256_file "${cache_retained_failure_receipt}")"
    "${python_bin}" -I -S "${runtime_cache_tool}" audit-retained-failure \
      --retained-failure-receipt "${cache_retained_failure_receipt}" \
      --expected-retained-failure-sha256 "${cache_retained_failure_sha}" \
      --prepare-receipt "${cache_prepare_receipt}" \
      --expected-prepare-receipt-sha256 "${cache_prepare_sha}" \
      >"${run_root}/logs/runtime-cache-retained-failure-audit-r4.json"
    child_terminal_written=true
    trap - ERR
    echo "BOX-EXP-014_R4_CONTROLLER_FAILED status=${controller_status} cache_retained=${cache_root} cache_reusable=false prepare_receipt=${cache_prepare_receipt} retained_failure_receipt=${cache_retained_failure_receipt}" >&2
    exit "${controller_status}"
  fi
  child_phase=cleanup-or-cleanup-receipt-publication
  readonly controller_completion_sha="$(sha256_file "${run_root}/controller-completion.json")"
  "${python_bin}" -I -S "${runtime_cache_tool}" cleanup \
    --prepare-receipt "${cache_prepare_receipt}" \
    --expected-prepare-receipt-sha256 "${cache_prepare_sha}" \
    --controller-completion "${run_root}/controller-completion.json" \
    --expected-controller-completion-sha256 "${controller_completion_sha}" \
    --cleanup-receipt-output "${cache_cleanup_receipt}" \
    --controller-exit-status 0
  readonly cache_cleanup_sha="$(sha256_file "${cache_cleanup_receipt}")"
  child_phase=post-cleanup-audit
  "${python_bin}" -I -S "${runtime_cache_tool}" audit-cleanup \
    --cleanup-receipt "${cache_cleanup_receipt}" \
    --expected-cleanup-receipt-sha256 "${cache_cleanup_sha}" \
    --prepare-receipt "${cache_prepare_receipt}" \
    --expected-prepare-receipt-sha256 "${cache_prepare_sha}" \
    --controller-completion "${run_root}/controller-completion.json" \
    --expected-controller-completion-sha256 "${controller_completion_sha}" \
    >"${run_root}/logs/runtime-cache-cleanup-audit-child-r4.json"
  child_terminal_written=true
  trap - ERR
  echo "BOX-EXP-014_R4_CONTROLLER_COMPLETE cache_cleaned_on_compute_node_before_step_exit=${cache_root} cache_reusable=false"
  exit 0
fi

[[ ! -e "${run_root}" && ! -L "${run_root}" && "$(realpath -m -- "${run_root}")" == "${run_root}" ]] || fail "run root must be fresh"
job_record="$(scontrol show job -o "${holder_job}")"
[[ "${job_record}" == *"JobId=${holder_job} "* && "${job_record}" == *"JobState=RUNNING"* ]] || fail "retained parent is not RUNNING"
[[ "${job_record}" == *"UserId=${holder_user}"* && "${job_record}" == *"NodeList=${holder_node}"* ]] || fail "retained parent owner/node differs"
[[ -z "$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')" ]] || fail "retained parent already has a numbered child"

mkdir -m 0700 "${run_root}" "${run_root}/logs" "${runtime_parent}"

# Verify the canonical manifest and every USTAR member with stdlib-only Python
# before importing any released Python module, then extract create-only.
"${python_bin}" -I -S - \
  "${method_archive}" "${method_archive_sha}" \
  "${method_manifest}" "${method_manifest_sha}" "${runtime_parent}" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile

archive_path, archive_sha, manifest_path, manifest_sha, output_root = sys.argv[1:]
archive_path = Path(archive_path)
manifest_path = Path(manifest_path)
output_root = Path(output_root)
archive_raw = archive_path.read_bytes()
manifest_raw = manifest_path.read_bytes()
assert hashlib.sha256(archive_raw).hexdigest() == archive_sha
assert hashlib.sha256(manifest_raw).hexdigest() == manifest_sha
manifest = json.loads(manifest_raw)
canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
assert canonical == manifest_raw
assert manifest["schema_version"] == "bernini-full30-action-source7-reencode-release-v1"
assert manifest["release_generation"] == "r4"
assert manifest["archive_format"] == "ustar-owner0-mtime0-exact-modes-zero-devfields-v2"
assert manifest["member_root"] == "methods/bernini_action_editing"
assert manifest["exact_member_closure"] is True
assert manifest["topology"]["holder_job_id"] == 136141
assert manifest["topology"]["holder_node"] == "auh7-1b-gpu-299"
assert manifest["topology"]["runtime_cache_prepare_cleanup_node"] == "auh7-1b-gpu-299"
assert manifest["topology"]["parent_must_not_inspect_or_remove_compute_node_tmp"] is True
assert manifest["authority"]["torch_version"] == "2.7.1+rocm6.3"
assert manifest["authority"]["torch_hip_version"] == "6.3.42131-fa1d09cbd"
assert manifest["authority"]["miopen_backend_version"] == 3003000
assert manifest["authority"]["miopen_custom_cache_fresh_gfx90a68_ukdb_required"] is True
assert manifest["authority"]["miopen_custom_cache_kern_db_nonempty_required"] is True
assert manifest["authority"]["miopen_custom_cache_wal_absent_or_empty_required"] is True
assert manifest["authority"]["miopen_custom_cache_sqlite_immutable_readonly_validation_required"] is True
assert manifest["authority"]["miopen_user_db_plaintext_main_mode_0777_required_if_present"] is True
assert manifest["authority"]["tmpdir_export_before_runtime_prepare_required"] is True
assert manifest["authority"]["cpp_temp_directory_path_scoped_lock_activity_required"] is True
assert manifest["authority"]["miopen_lock_basenames_path_hash_bound"] is True
assert manifest["authority"]["global_miopen_lock_root_authoritative"] is False
assert manifest["authority"]["global_miopen_lock_root_members_scanned"] is False
assert manifest["authority"]["global_miopen_lock_root_cleanup_allowed"] is False
assert manifest["authority"]["phase_failure_create_only_terminal_required"] is True
assert manifest["authority"]["retained_failure_scoped_lock_root_present_absent_observation_required"] is True
assert manifest["authority"]["post_srun_parent_recomputes_prepare_completion_cleanup_digests"] is True
assert manifest["authority"]["post_srun_parent_revalidates_completion_negative_access_and_exact7_authority"] is True
assert manifest["authority"]["final_marker_binds_release_and_all_runtime_receipt_hashes_and_digests"] is True
assert manifest["authority"]["ustar_header_fields_explicitly_normalized"] is True
assert manifest["authority"]["ustar_raw_headers_checksums_offsets_and_zero_trailer_reverified"] is True
unsigned = dict(manifest)
declared = unsigned.pop("manifest_digest")
assert hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest() == declared
expected = {f"methods/bernini_action_editing/{row['path']}": row for row in manifest["files"]}
offset = 0
for row in manifest["files"]:
    assert offset + 512 <= len(archive_raw)
    header = archive_raw[offset : offset + 512]
    assert header[329:345] == b"\x00" * 16
    checksum_header = bytearray(header)
    checksum_header[148:156] = b" " * 8
    checksum = sum(checksum_header)
    assert header[148:156] == f"{checksum:06o}\x00 ".encode("ascii")
    expected_name = f"methods/bernini_action_editing/{row['path']}".encode("ascii")
    assert len(expected_name) <= 100
    assert header[0:100].split(b"\x00", 1)[0] == expected_name
    data_start = offset + 512
    data_end = data_start + row["size"]
    padded_end = data_start + ((row["size"] + 511) // 512) * 512
    assert data_end <= len(archive_raw)
    assert hashlib.sha256(archive_raw[data_start:data_end]).hexdigest() == row["sha256"]
    assert archive_raw[data_end:padded_end] == b"\x00" * (padded_end - data_end)
    offset = padded_end
expected_archive_size = (
    (offset + 1024 + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE
) * tarfile.RECORDSIZE
assert len(archive_raw) == expected_archive_size
assert archive_raw[offset:] == b"\x00" * (len(archive_raw) - offset)
with tarfile.open(archive_path, "r:") as handle:
    members = handle.getmembers()
    assert [member.name for member in members] == list(expected)
    for member in members:
        row = expected[member.name]
        pure = PurePosixPath(member.name)
        assert not pure.is_absolute() and ".." not in pure.parts
        assert member.isfile() and not member.issym() and not member.islnk()
        assert member.uid == member.gid == member.mtime == 0
        assert member.devmajor == member.devminor == 0
        assert stat.S_IMODE(member.mode) == row["mode"] and member.size == row["size"]
        stream = handle.extractfile(member)
        assert stream is not None
        raw = stream.read()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        destination = output_root / pure
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), row["mode"])
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(destination, row["mode"])
PY

[[ -f "${release_builder}" && -f "${controller}" && -f "${runtime_cache_tool}" && -f "${launcher}" ]] || fail "extracted release entrypoints differ"
[[ "$(sha256_file "$0")" == "$(sha256_file "${launcher}")" ]] || fail "invoked launcher is not the sealed release launcher"
"${python_bin}" -B "${release_builder}" audit \
  --archive "${method_archive}" --expected-archive-sha256 "${method_archive_sha}" \
  --manifest "${method_manifest}" --expected-manifest-sha256 "${method_manifest_sha}" \
  >"${run_root}/logs/static-release-audit.json" || fail "release audit failed"

set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --immediate=5 \
  --cpus-per-task=8 --mem=48G --gpus-per-task=1 --gpu-bind=none --gres-flags=enforce-binding \
  env S7_CONFIRM="${confirm}" S7_RUN_ROOT="${run_root}" \
      S7_METHOD_ARCHIVE="${method_archive}" S7_METHOD_ARCHIVE_SHA256="${method_archive_sha}" \
      S7_METHOD_MANIFEST="${method_manifest}" S7_METHOD_MANIFEST_SHA256="${method_manifest_sha}" \
      bash "${launcher}" __child >"${run_root}/logs/source7-reencode-r4.log" 2>&1
status=$?
set -e
if (( status != 0 )); then
  tail -n 240 "${run_root}/logs/source7-reencode-r4.log" >&2 || true
  [[ ( -f "${run_root}/logs/runtime-cache-phase-failure-r4.json" && ! -L "${run_root}/logs/runtime-cache-phase-failure-r4.json" ) || ( -f "${run_root}/logs/runtime-cache-phase-failure-r4.fallback" && ! -L "${run_root}/logs/runtime-cache-phase-failure-r4.fallback" ) || ( -f "${run_root}/logs/runtime-cache-retained-failure-r4.json" && ! -L "${run_root}/logs/runtime-cache-retained-failure-r4.json" ) ]] || fail "compute child failed without a create-only phase terminal"
  fail "source7 re-encode child failed status=${status}"
fi

[[ -f "${run_root}/controller-completion.json" && ! -L "${run_root}/controller-completion.json" ]] || fail "controller completion is missing"
[[ -d "${run_root}/physical_source_posterior_index0_exact7" && ! -L "${run_root}/physical_source_posterior_index0_exact7" ]] || fail "physical exact7 output is missing"
readonly cache_prepare_receipt="${run_root}/logs/runtime-cache-prepare-r4.json"
readonly cache_cleanup_receipt="${run_root}/logs/runtime-cache-cleanup-r4.json"
[[ -f "${cache_prepare_receipt}" && ! -L "${cache_prepare_receipt}" ]] || fail "runtime cache prepare receipt is missing"
[[ -f "${cache_cleanup_receipt}" && ! -L "${cache_cleanup_receipt}" ]] || fail "compute-child runtime cache cleanup receipt is missing"
readonly cache_prepare_sha="$(sha256_file "${cache_prepare_receipt}")"
readonly controller_completion_sha="$(sha256_file "${run_root}/controller-completion.json")"
readonly cache_cleanup_sha="$(sha256_file "${cache_cleanup_receipt}")"
# Receipt-only shared-filesystem audit.  The login parent never inspects or
# removes the compute node's /tmp path.
"${python_bin}" -I -S "${runtime_cache_tool}" audit-cleanup \
  --cleanup-receipt "${cache_cleanup_receipt}" \
  --expected-cleanup-receipt-sha256 "${cache_cleanup_sha}" \
  --prepare-receipt "${cache_prepare_receipt}" \
  --expected-prepare-receipt-sha256 "${cache_prepare_sha}" \
  --controller-completion "${run_root}/controller-completion.json" \
  --expected-controller-completion-sha256 "${controller_completion_sha}" \
  >"${run_root}/logs/runtime-cache-cleanup-audit-r4.json"
"${python_bin}" -I -S - \
  "${run_root}/BOX-EXP-014_R4_COMPLETE" \
  "${method_archive}" "${method_archive_sha}" \
  "${method_manifest}" "${method_manifest_sha}" \
  "${cache_prepare_receipt}" "${cache_prepare_sha}" \
  "${run_root}/controller-completion.json" "${controller_completion_sha}" \
  "${cache_cleanup_receipt}" "${cache_cleanup_sha}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import sys

(
    marker_arg,
    archive_arg,
    archive_sha_arg,
    manifest_arg,
    manifest_sha_arg,
    prepare_arg,
    prepare_sha_arg,
    completion_arg,
    completion_sha_arg,
    cleanup_arg,
    cleanup_sha_arg,
) = sys.argv[1:]
sha_re = re.compile(r"[0-9a-f]{64}")

def read_bound(path_text, expected_sha, *, json_file):
    path = Path(path_text)
    assert path.is_absolute() and path.resolve(strict=True) == path
    raw_value = path.read_bytes()
    assert sha_re.fullmatch(expected_sha)
    assert hashlib.sha256(raw_value).hexdigest() == expected_sha
    if not json_file:
        return path, None
    value = json.loads(raw_value)
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii") + b"\n"
    assert canonical == raw_value
    return path, value

archive_path, _ = read_bound(archive_arg, archive_sha_arg, json_file=False)
manifest_path, manifest = read_bound(manifest_arg, manifest_sha_arg, json_file=True)
prepare_path, prepare = read_bound(prepare_arg, prepare_sha_arg, json_file=True)
completion_path, completion = read_bound(
    completion_arg, completion_sha_arg, json_file=True
)
cleanup_path, cleanup = read_bound(cleanup_arg, cleanup_sha_arg, json_file=True)
assert manifest["release_generation"] == "r4"
assert prepare["run_generation"] == completion["run_generation"] == cleanup["run_generation"] == "r4"
assert completion["complete"] is True
assert cleanup["controller_complete"] is True and cleanup["cache_root_removed"] is True
assert cleanup["prepare_digest"] == prepare["prepare_digest"]
assert cleanup["controller_completion_digest"] == completion["completion_digest"]
assert all(
    sha_re.fullmatch(value)
    for value in (
        manifest["manifest_digest"], prepare["prepare_digest"],
        completion["completion_digest"], cleanup["cleanup_digest"],
    )
)
assert re.fullmatch(r"[0-9a-f]{40}", manifest["content_closure_sha1"])
bindings = {
    "release_archive_path": str(archive_path),
    "release_archive_sha256": archive_sha_arg,
    "release_manifest_path": str(manifest_path),
    "release_manifest_sha256": manifest_sha_arg,
    "release_manifest_digest": manifest["manifest_digest"],
    "release_content_closure_sha1": manifest["content_closure_sha1"],
    "runtime_cache_prepare_receipt_path": str(prepare_path),
    "runtime_cache_prepare_receipt_sha256": prepare_sha_arg,
    "runtime_cache_prepare_digest": prepare["prepare_digest"],
    "controller_completion_path": str(completion_path),
    "controller_completion_sha256": completion_sha_arg,
    "controller_completion_digest": completion["completion_digest"],
    "runtime_cache_cleanup_receipt_path": str(cleanup_path),
    "runtime_cache_cleanup_receipt_sha256": cleanup_sha_arg,
    "runtime_cache_cleanup_digest": cleanup["cleanup_digest"],
}
fixed = [
    "schema=bernini-full30-action-source7-reencode-launch-status-v4",
    "experiment_id=BOX-EXP-014",
    "run_generation=r4",
    "holder_job=136141",
    "holder_node=auh7-1b-gpu-299",
    "source_mp4_count=7",
    "vae_encode_calls=7",
    "source_only_reencode_from_source_video=true",
    "vae_encode_calls_per_source=1",
    "physical_posterior_count=7",
    "external_2d2_reencoded=false",
    "paired_dataset_accessed=false",
    "legacy_source_target_container_opened=false",
    "synthetic_target_index1_path_read=false",
    "synthetic_target_index1_bytes_read=false",
    "synthetic_target_index1_decoded=false",
    "synthetic_target_index1_filtered_on=false",
    "synthetic_target_index1_hashed=false",
    "target_video_path_present=false",
    "target_video_accessed=false",
    "home_unchanged=true",
    "runtime_cache_ext4_create_only=true",
    "runtime_cache_exclusive_fsync_probe=true",
    "runtime_cache_sqlite_commit_reopen_probe=true",
    "cuda_miopen_wan_resample_three_geometry_smoke_before_source_open=true",
    "miopen_custom_cache_fresh_gfx90a68_ukdb_kern_db_nonempty=true",
    "miopen_custom_cache_wal_absent_or_empty=true",
    "miopen_custom_cache_sqlite_immutable_readonly_validation=true",
    "miopen_user_db_path_write_required=false",
    "miopen_user_db_plaintext_main_mode_0777_enforced=true",
    "miopen_user_db_time_sidecar_mode_recorded_not_pinned=true",
    "tmpdir_exported_before_runtime_prepare=true",
    "tmpdir_scoped_under_runtime_cache=true",
    "cpp_temp_directory_path_redirect_observed=true",
    "scoped_miopen_temp_lock_activity_observed=true",
    "scoped_miopen_lock_basenames_user_db_parent_md5_bound=true",
    "scoped_miopen_temp_locks_removed_with_cache_root=true",
    "global_miopen_lock_root_members_scanned=false",
    "global_miopen_lock_root_cleanup_attempted=false",
    "runtime_cache_cleaned_on_gpu299_after_controller_exit_before_numbered_step_exit=true",
    "srun_numbered_step_exit_status=0",
    "login_parent_compute_tmp_accessed=false",
    "runtime_cache_reusable=false",
    "optimizer_updates=0",
    "parent_retained=true",
]
raw = ("\n".join(fixed + [f"{key}={value}" for key, value in bindings.items()]) + "\n").encode("ascii")
path = Path(marker_arg)
descriptor = os.open(
    path,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o400,
)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(raw)
    handle.flush()
    os.fsync(handle.fileno())
assert path.read_bytes() == raw
directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
echo "BOX-EXP-014_R4_COMPLETE output=${run_root} parent_retained=true optimizer_updates=0"

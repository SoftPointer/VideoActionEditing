#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# AUH-side formal submitter.  This script never requests a hold and never adds
# a dependency; the returned state must be a schedulable PD or R state.

fail() { echo "[submit-saic-ssft-mech-v1] ERROR: $*" >&2; exit 2; }
hash_file() { sha256sum -- "$1" | awk '{print $1}'; }

base=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809
source_archive="${SAIC_SSFT_MECH_SOURCE_ARCHIVE:?set immutable scoped source archive}"
source_archive_sha256="${SAIC_SSFT_MECH_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
source_revision="${SAIC_SSFT_MECH_SOURCE_REVISION:?set source revision}"
launcher="${SAIC_SSFT_MECH_LAUNCHER:?set extracted authenticated launcher}"
launcher_sha256="${SAIC_SSFT_MECH_LAUNCHER_SHA256:?set launcher SHA-256}"
release_round="${SAIC_SSFT_MECH_RELEASE_ROUND:-1}"

[[ "${source_archive}" == /* && -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ "${launcher}" == /* && -f "${launcher}" && ! -L "${launcher}" ]] || fail "launcher differs"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "archive SHA differs"
[[ "${launcher_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "launcher SHA differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${release_round}" =~ ^[1-9][0-9]*$ ]] || fail "release round differs"
[[ "$(hash_file "${source_archive}")" == "${source_archive_sha256}" ]] || fail "archive bytes differ"
[[ "$(hash_file "${launcher}")" == "${launcher_sha256}" ]] || fail "launcher bytes differ"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive commit differs"

short_revision="${source_revision:0:12}"
output_root="${base}/runs/ssft-t1-iavg-i1-i1a-v1-${short_revision}-r${release_round}"
submission_dir="${base}/submissions"
log_dir="${base}/logs"
submission_receipt="${submission_dir}/ssft-t1-iavg-i1-i1a-v1-${short_revision}-r${release_round}.submission.json"
mkdir -p -- "${submission_dir}" "${log_dir}"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "versioned output root already exists"
[[ ! -e "${submission_receipt}" && ! -L "${submission_receipt}" ]] || fail "submission receipt already exists"

export SAIC_SSFT_MECH_ACK_FROZEN_DIAGNOSTIC_ONLY=YES
export SAIC_SSFT_MECH_ACK_ZERO_SELECTION_TRAINING_OPTIMIZER=YES
export SAIC_SSFT_MECH_ACK_REUSE_ONLY_JOB132387_SOURCE_COORDINATE=YES
export SAIC_SSFT_MECH_SOURCE_ARCHIVE="${source_archive}"
export SAIC_SSFT_MECH_SOURCE_ARCHIVE_SHA256="${source_archive_sha256}"
export SAIC_SSFT_MECH_SOURCE_REVISION="${source_revision}"
export SAIC_SSFT_MECH_OUTPUT_ROOT="${output_root}"
export BERNINI_OFFICIAL_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
export BERNINI_VEOMNI_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
export BERNINI_ACTION_CHECKPOINT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
export BERNINI_CHECKPOINT_CONTENT_MANIFEST=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/braid_source_tree_7608c43_20260809/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
export SAIC_SSFT_MECH_PYTHON_BIN=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

job_id="$(sbatch --parsable --export=ALL \
  --output="${log_dir}/saic-ssft-mech-v1-%j.out" \
  --error="${log_dir}/saic-ssft-mech-v1-%j.err" \
  "${launcher}")"
[[ "${job_id}" =~ ^[0-9]+$ ]] || fail "sbatch did not return one numeric job ID"

job_line="$(squeue -h -j "${job_id}" -o '%i|%T|%r|%j|%R' | head -n 1)"
[[ -n "${job_line}" ]] || fail "submitted job is absent from squeue"
IFS='|' read -r observed_id state reason job_name node_or_reason <<<"${job_line}"
[[ "${observed_id}" == "${job_id}" ]] || fail "queued job ID differs"
[[ "${state}" == PENDING || "${state}" == RUNNING ]] || fail "job is not schedulable PD/R: ${state}"
[[ "${reason}" != JobHeldUser && "${reason}" != JobHeldAdmin ]] || fail "job is held"
[[ "${job_name}" == saic-ssft-mech-v1 ]] || fail "job name differs"

/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12 -B - \
  "${submission_receipt}" "${job_id}" "${state}" "${reason}" "${node_or_reason}" \
  "${output_root}" "${source_archive}" "${source_archive_sha256}" "${source_revision}" \
  "${launcher}" "${launcher_sha256}" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
(
    receipt_path, job_id, state, reason, node_or_reason, output_root,
    archive_path, archive_sha, revision, launcher_path, launcher_sha,
) = sys.argv[1:]
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("ascii")
unsigned = {
    "schema_version": "bernini-saic-ssft-mechanism-screen-submission-v1",
    "job_id": job_id,
    "job_name": "saic-ssft-mech-v1",
    "state_at_receipt": state,
    "reason_at_receipt": reason,
    "node_or_reason_at_receipt": node_or_reason,
    "held": False,
    "dependency": None,
    "schedulable_pd_or_running": True,
    "source_archive_path": archive_path,
    "source_archive_sha256": archive_sha,
    "source_revision": revision,
    "launcher_path": launcher_path,
    "launcher_sha256": launcher_sha,
    "output_root": output_root,
    "fixed_arm_order": ["T1", "IAVG", "I1", "I1A"],
    "allocated_mi210": 8,
    "source_coordinate_upstream_job_id": "132387",
    "training_authority": False,
    "optimizer_authority": False,
    "semantic_action_success": False,
}
receipt = {**unsigned, "receipt_digest": hashlib.sha256(canonical(unsigned)).hexdigest()}
target = pathlib.Path(receipt_path)
fd, temporary_text = tempfile.mkstemp(prefix=".submission.", dir=str(target.parent))
temporary = pathlib.Path(temporary_text)
try:
    with os.fdopen(fd, "wb", closefd=True) as handle:
        handle.write(canonical(receipt) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    os.link(temporary, target)
finally:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
print(canonical(receipt).decode("ascii"))
PY

echo "[submit-saic-ssft-mech-v1] SUBMITTED job=${job_id} state=${state} reason=${reason} output=${output_root} receipt=${submission_receipt}"

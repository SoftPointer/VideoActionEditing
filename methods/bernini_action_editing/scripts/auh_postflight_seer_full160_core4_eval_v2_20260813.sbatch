#!/usr/bin/env bash
# Submit with --dependency=afterok:<full160-array-job-id> after the decode array.
#SBATCH --job-name=seer160-post
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -Eeuo pipefail
umask 077

fail() { echo "[seer160-postflight] ERROR: $*" >&2; exit 2; }

run_id="${SEER_FULL160_EVAL_RUN_ID:?set the exact decode run ID}"
array_job_id="${SEER_FULL160_EVAL_ARRAY_JOB_ID:?set completed decode array job ID}"
[[ "${run_id}" =~ ^[a-z0-9][a-z0-9-]{0,63}$ ]] || fail "run ID differs"
[[ "${array_job_id}" =~ ^[1-9][0-9]*$ ]] || fail "array job ID differs"

exp=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_seer_event_erasure_v1_20260813
stage="${exp}/staging/seer-full160-eval-overlay-v2"
output_root="${exp}/heldout/full160-core4-${run_id}"
postflight_launcher="${stage}/auh_postflight_seer_full160_core4_eval_v2_20260813.sbatch"
postflight_launcher_sha="${SEER_FULL160_POSTFLIGHT_LAUNCHER_SHA256:?set frozen postflight launcher SHA}"
postflight="${stage}/postflight_seer_full160_core4_eval_v2.py"
postflight_sha="${SEER_FULL160_POSTFLIGHT_SHA256:?set frozen postflight SHA}"
source_binder="${stage}/bind_seer_full160_eval_source_v2.py"
source_binder_sha="${SEER_FULL160_SOURCE_BINDER_SHA256:?set frozen source binder SHA}"
overlay_builder="${stage}/tools/build_seer_full160_eval_overlay_v1.py"
overlay_builder_sha="${SEER_FULL160_OVERLAY_BUILDER_SHA256:?set frozen overlay builder SHA}"
verdict_reducer="${stage}/seer_full160_review_verdict_v2.py"
verdict_reducer_sha="${SEER_FULL160_VERDICT_REDUCER_SHA256:?set frozen verdict reducer SHA}"
recovery_verifier="${stage}/tools/recover_seer_fm160_job135313_v1.py"
recovery_verifier_sha="${SEER_FM160_RECOVERY_VERIFIER_SHA256:?set frozen recovery verifier SHA}"
recovery_receipt="${SEER_FM160_RECOVERY_RECEIPT:?set absolute recovery receipt path}"
recovery_receipt_sha="${SEER_FM160_RECOVERY_RECEIPT_SHA256:?set frozen recovery receipt raw SHA}"
recovery_receipt_digest="${SEER_FM160_RECOVERY_RECEIPT_DIGEST:?set frozen recovery receipt digest}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
spec_sha=82fbe0f042d86f8d54aa254ce72a384e70aa5bdc3c1ac66d5422037cd4b4051c
master="${output_root}/full160-core4-eval-master-binding.json"
blank_review="${output_root}/full160-core4-strict-paired-review.blank.json"

[[ "$(sha256sum "${postflight_launcher}" | awk '{print $1}')" == "${postflight_launcher_sha}" ]] || fail "postflight launcher differs"
[[ "$(sha256sum "${postflight}" | awk '{print $1}')" == "${postflight_sha}" ]] || fail "postflight differs"
[[ "$(sha256sum "${source_binder}" | awk '{print $1}')" == "${source_binder_sha}" ]] || fail "source binder differs"
[[ "$(sha256sum "${overlay_builder}" | awk '{print $1}')" == "${overlay_builder_sha}" ]] || fail "overlay builder differs"
[[ "$(sha256sum "${verdict_reducer}" | awk '{print $1}')" == "${verdict_reducer_sha}" ]] || fail "verdict reducer differs"
[[ "$(sha256sum "${recovery_verifier}" | awk '{print $1}')" == "${recovery_verifier_sha}" ]] || fail "recovery verifier differs"
[[ "$(sha256sum "${recovery_receipt}" | awk '{print $1}')" == "${recovery_receipt_sha}" ]] || fail "recovery receipt differs"
"${python_bin}" -I -B "${recovery_verifier}" verify --receipt "${recovery_receipt}" >/dev/null
"${python_bin}" -I -B - "${recovery_receipt}" "${recovery_receipt_digest}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
assert r.get("receipt_digest") == sys.argv[2]
assert r.get("checkpoint_heldout_eligible") is True and r.get("slurm_job_success") is False
PY
[[ ! -e "${master}" && ! -L "${master}" ]] || fail "master output must be fresh"
[[ ! -e "${blank_review}" && ! -L "${blank_review}" ]] || fail "blank review must be fresh"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
"${python_bin}" -I -B "${postflight}" \
  --output-root "${output_root}" --array-job-id "${array_job_id}" \
  --python-bin "${python_bin}" --expected-spec-sha256 "${spec_sha}" \
  --recovery-receipt "${recovery_receipt}" \
  --expected-recovery-receipt-sha256 "${recovery_receipt_sha}" \
  --expected-recovery-receipt-digest "${recovery_receipt_digest}" \
  --output "${master}"
"${python_bin}" -I -B "${verdict_reducer}" blank \
  --master-binding "${master}" \
  --recovery-receipt "${recovery_receipt}" \
  --expected-recovery-receipt-sha256 "${recovery_receipt_sha}" \
  --expected-recovery-receipt-digest "${recovery_receipt_digest}" \
  --output "${blank_review}"
echo "PASS_EXECUTION_CLOSURE_PENDING_STRICT_PAIRED_FULL_VIDEO_REVIEW launcher_sha256=${postflight_launcher_sha} master=${master} blank_review=${blank_review}"

#!/usr/bin/env bash
set -euo pipefail

JOB_ID="140846"
NODE="auh7-1b-gpu-279"
PYTHON_BIN="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python"
ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817"
RELEASE_ROOT="${ROOT}/releases/goku-stage0-admission-d206beba2053-v3"
MODULE="${RELEASE_ROOT}/goku_paired_stage0_admission_0817_v1.py"
MODULE_SHA256="d206beba2053b9ee240a78d2eed2fff4aa8e864dd6db265f66c9a287229ba272"
ORIGINAL="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_atomic1000_round2_20260813T060000Z/source_census_v1/final/selected.jsonl"
STAGE0="${ROOT}/data_prep/goku-licensed-pair-target-stage0-1676-v1"
ATTEMPT="${ROOT}/data_prep/goku-licensed-pair-stage0-admission-1198-v3-attempt3"
OUTPUT="${ROOT}/data_prep/goku-licensed-pair-stage0-admission-1198-v3"
STATUS="${ATTEMPT}/controller.status.json"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256_file "$path")"
  if [[ "$observed" != "$expected" ]]; then
    echo "SHA mismatch path=${path} expected=${expected} observed=${observed}" >&2
    exit 41
  fi
}

if ! mkdir -m 0700 "$ATTEMPT"; then
  echo "attempt already claimed: $ATTEMPT" >&2
  exit 42
fi
if [[ -e "$OUTPUT" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 43
fi

[[ -f "$MODULE" && ! -L "$MODULE" ]] || exit 44
[[ "$(stat -c %a "$MODULE")" == "444" ]] || exit 45
[[ "$(stat -c %h "$MODULE")" == "1" ]] || exit 46
require_sha "$MODULE" "$MODULE_SHA256"
require_sha "$ORIGINAL" "7a76d6b2dec10203f1af00016c60f6bb654e2ebea19d5b08b74a49d31300cac2"
require_sha "${STAGE0}/summary.json" "b9ea9af7fa4bbe38f3386231073cd640f81111ba4ae1523e3256e3049f14c36e"
require_sha "${STAGE0}/audit.jsonl" "279c0aa70a3c45464dc7c622d501eb1803095ea9ac06a173fa73c95352c8a3e1"
require_sha "${STAGE0}/selected.jsonl" "9143d94830b09f60ee365043f53ce5a6d40689428c57528403f1654af114f190"
require_sha "${STAGE0}/descriptors.npz" "63999e578cf6ba4b1c78a03b8ce6306b2180d361a9501c863a01fc8e922a29cc"

parent_before="$(squeue -j "$JOB_ID" -h -o %T)"
if [[ "$parent_before" != "RUNNING" ]]; then
  echo "parent is not RUNNING: $parent_before" >&2
  exit 47
fi
while IFS='|' read -r step_id step_nodes; do
  [[ -z "$step_id" ]] && continue
  suffix="${step_id#*.}"
  if [[ "$suffix" == "batch" || "$suffix" == "extern" ]]; then
    continue
  fi
  if [[ "$step_nodes" == *"279"* ]]; then
    echo "node279 already has numeric child ${step_id} on ${step_nodes}" >&2
    exit 48
  fi
done < <(squeue --steps -j "$JOB_ID" -h -o '%i|%N')

set +e
/usr/bin/srun \
  --jobid="$JOB_ID" \
  --overlap \
  --exact \
  --nodes=1 \
  --ntasks=1 \
  --nodelist="$NODE" \
  --cpus-per-task=16 \
  --mem=60G \
  --gres=gpu:mi210:8 \
  --kill-on-bad-exit=1 \
  env \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" -B "$MODULE" \
      --stage0-dir "$STAGE0" \
      --original-selected "$ORIGINAL" \
      --output-dir "$OUTPUT" \
      --expected-original-sha256 7a76d6b2dec10203f1af00016c60f6bb654e2ebea19d5b08b74a49d31300cac2 \
      --expected-summary-sha256 b9ea9af7fa4bbe38f3386231073cd640f81111ba4ae1523e3256e3049f14c36e \
      --expected-audit-sha256 279c0aa70a3c45464dc7c622d501eb1803095ea9ac06a173fa73c95352c8a3e1 \
      --expected-selected-sha256 9143d94830b09f60ee365043f53ce5a6d40689428c57528403f1654af114f190 \
      --expected-descriptors-sha256 63999e578cf6ba4b1c78a03b8ce6306b2180d361a9501c863a01fc8e922a29cc
child_rc="$?"
set -e

if [[ "$child_rc" != "0" ]]; then
  jq -n \
    --argjson child_exit "$child_rc" \
    '{schema:"goku-stage0-admission-controller-v3",status:"failed",child_exit:$child_exit,training_authorized:false,formal_training_started:false,automatic_relaunch_authorized:false,parent_cancelled:false,parent_released:false,parent_requeued:false}' > "$STATUS"
  chmod 0400 "$STATUS"
  exit "$child_rc"
fi

receipt="${OUTPUT}/admission_receipt.json"
done_file="${OUTPUT}/DONE.json"
for visibility_probe in $(seq 1 30); do
  if [[ -f "$receipt" && -f "$done_file" ]]; then
    break
  fi
  /bin/sleep 1
done
if [[ ! -f "$receipt" || ! -f "$done_file" ]]; then
  jq -n \
    '{schema:"goku-stage0-admission-controller-v3",status:"postprocess_visibility_failed",child_exit:0,training_authorized:false,formal_training_started:false,automatic_relaunch_authorized:false,parent_cancelled:false,parent_released:false,parent_requeued:false}' > "$STATUS"
  chmod 0400 "$STATUS"
  exit 50
fi
jq -e '
  .qualification_status == "unqualified" and
  .training_authorized == false and
  .formal_d0_count_contribution == 0 and
  .counts.original_rows == 1676 and
  .counts.audit_rows == 1676 and
  .counts.selected_physical_rows == 1198 and
  .counts.formal_d0_qualified_rows == 0
' "$receipt" >/dev/null
jq -e '
  .closure_status == "PASS" and
  .qualification_status == "unqualified" and
  .training_authorized == false and
  .formal_d0_count_contribution == 0
' "$done_file" >/dev/null

parent_after="$(squeue -j "$JOB_ID" -h -o %T)"
if [[ "$parent_after" != "RUNNING" ]]; then
  echo "parent state changed after child: $parent_after" >&2
  exit 49
fi

chmod -R a-w "$OUTPUT"
receipt_sha="$(sha256_file "$receipt")"
done_sha="$(sha256_file "$done_file")"
candidate_sha="$(sha256_file "${OUTPUT}/candidate_manifest.jsonl")"
review_sha="$(sha256_file "${OUTPUT}/review_queue.jsonl")"
effective_n="$(jq -r '.counts.effective_diagnostic_candidates' "$receipt")"

jq -n \
  --arg receipt_sha256 "$receipt_sha" \
  --arg done_sha256 "$done_sha" \
  --arg candidate_sha256 "$candidate_sha" \
  --arg review_sha256 "$review_sha" \
  --argjson effective_diagnostic_candidates "$effective_n" \
  '{schema:"goku-stage0-admission-controller-v3",status:"complete",child_exit:0,input_rows:1676,stage0_selected:1198,effective_diagnostic_candidates:$effective_diagnostic_candidates,receipt_sha256:$receipt_sha256,done_sha256:$done_sha256,candidate_manifest_sha256:$candidate_sha256,review_queue_sha256:$review_sha256,qualification_status:"unqualified",training_authorized:false,formal_d0_count_contribution:0,formal_training_started:false,automatic_relaunch_authorized:false,parent_cancelled:false,parent_released:false,parent_requeued:false}' > "$STATUS"
chmod 0400 "$STATUS"
echo "GOKU_STAGE0_ADMISSION_COMPLETE effective=${effective_n} training_authorized=false output=${OUTPUT}"

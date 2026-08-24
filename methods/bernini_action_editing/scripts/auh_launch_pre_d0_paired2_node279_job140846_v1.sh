#!/usr/bin/env bash
# One-shot controller for the disposable 0817 PRE_D0 paired two-update smoke.
# It never retries and never cancels, releases, requeues, or signals job 140846.

set -Eeuo pipefail
umask 077

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly launch_root="${experiment_root}/launchers/pre-d0-paired2-1211e56185aa-v2"
readonly attempt_root="${experiment_root}/attempts/pre-d0-paired2-1211e56185aa-r1"
readonly output="${experiment_root}/runs/pre_d0_engineering_paired2-1211e56185aa-r1"
readonly step_payload="${launch_root}/auh_pre_d0_paired2_node279_step_v1.sh"
readonly step_payload_sha=a4dff014dd53f4b1bf46e46b2ca21957049a5e58899d30312063506711b6f732
readonly rank_exec="${launch_root}/auh_pre_d0_paired2_node279_rank_exec_v1.sh"
readonly rank_exec_sha=bac05573c035ce6770665304f54970ed1fefc2aa76f2801f1e4cb34929f5b638

fail() {
  printf 'PRE_D0 node279 controller refused: %s\n' "$*" >&2
  exit 95
}

[[ -d "${launch_root}" && ! -L "${launch_root}" ]] || fail "launch root differs"
[[ "$(stat -c %a "${launch_root}")" == 555 ]] || fail "launch root mode differs"
[[ -x "${step_payload}" && ! -L "${step_payload}" ]] || fail "step payload differs"
[[ "$(sha256sum "${step_payload}" | awk '{print $1}')" == "${step_payload_sha}" ]] || fail "step payload SHA differs"
[[ -x "${rank_exec}" && ! -L "${rank_exec}" ]] || fail "rank wrapper differs"
[[ "$(sha256sum "${rank_exec}" | awk '{print $1}')" == "${rank_exec_sha}" ]] || fail "rank wrapper SHA differs"
[[ -d "${attempt_root}" && ! -L "${attempt_root}" ]] || fail "attempt latch root differs"
[[ "$(stat -c %a "${attempt_root}")" == 700 ]] || fail "attempt latch mode differs"
mkdir -m 0700 "${attempt_root}/STARTED" 2>/dev/null || fail "attempt was already claimed; no retry is authorized"
printf 'schema=bernini-0817-pre-d0-attempt-claim-v1\njob_id=%s\nnode=%s\nautomatic_relaunch_authorized=false\n' \
  "${job_id}" "${node}" >"${attempt_root}/STARTED/intent"
chmod 0400 "${attempt_root}/STARTED/intent"
[[ ! -e "${attempt_root}/train.log" ]] || fail "attempt train log already exists"
[[ ! -e "${attempt_root}/controller.status" ]] || fail "attempt status already exists"
[[ ! -e "${attempt_root}/SUCCESS" ]] || fail "attempt success marker already exists"
[[ ! -e "${output}" ]] || fail "PRE_D0 output is not fresh"

readonly parent_state="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"
[[ "${parent_state}" == 'RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8' ]] || fail "parent allocation state differs: ${parent_state}"
readonly supported_steps="$(squeue --steps -h -j "${job_id}" -o '%i' | LC_ALL=C sort)"
[[ "${supported_steps}" == $'140846.batch\n140846.extern' ]] || fail "numeric or unexpected child already exists: ${supported_steps}"

printf 'PRE_D0 one-shot start job=%s node=%s output=%s parent_untouched=true\n' \
  "${job_id}" "${node}" "${output}"

set +e
/usr/bin/srun --jobid="${job_id}" --overlap --exact --nodes=1 --ntasks=1 \
  --nodelist="${node}" --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 \
  --kill-on-bad-exit=1 "${step_payload}" >"${attempt_root}/train.log" 2>&1
status=$?
set -e

readonly status_tmp="${attempt_root}/.controller.status.$$.tmp"
printf 'schema=bernini-0817-pre-d0-controller-status-v1\njob_id=%s\nnode=%s\nchild_exit=%s\nattempt_claimed=true\nparent_cancelled=false\nparent_released=false\nparent_requeued=false\nautomatic_relaunch_authorized=false\n' \
  "${job_id}" "${node}" "${status}" >"${status_tmp}"
chmod 0400 "${status_tmp}"
mv "${status_tmp}" "${attempt_root}/controller.status"

if (( status != 0 )); then
  printf 'PRE_D0 child failed rc=%s; no retry is authorized\n' "${status}" >&2
  tail -n 240 "${attempt_root}/train.log" >&2 || true
  exit "${status}"
fi

readonly receipt="${output}/receipt.json"
[[ -f "${receipt}" && ! -L "${receipt}" ]] || fail "terminal receipt is missing"
/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12 -I -B -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
assert value["authority"] == "PRE_D0_ENGINEERING_ONLY"
assert value["complete"] is True
assert value["promotable"] is False
assert value["formal_training_started"] is False
assert value["counts_as_d0"] is False
assert value["scientific_claim_authorized"] is False
assert value["optimizer_steps"] == 2
assert value["parent_allocation_released"] is False
print("PASS PRE_D0 terminal receipt", value["receipt_digest"], flush=True)
' "${receipt}"
readonly receipt_sha="$(sha256sum "${receipt}" | awk '{print $1}')"
readonly success_tmp="${attempt_root}/.SUCCESS.$$.tmp"
printf 'PRE_D0_ENGINEERING_COMPLETE=true\nreceipt_sha256=%s\npromotable=false\nformal_training_started=false\nparent_untouched=true\nautomatic_relaunch_authorized=false\n' \
  "${receipt_sha}" >"${success_tmp}"
chmod 0400 "${success_tmp}"
mv "${success_tmp}" "${attempt_root}/SUCCESS"
printf 'PRE_D0 two-update engineering smoke complete receipt_sha256=%s parent_untouched=true\n' "${receipt_sha}"

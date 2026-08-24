#!/usr/bin/env bash
# One-shot controller for the disposable 0817 PRE_D0 r2 paired two-update
# smoke. It never retries and never cancels, releases, requeues, or signals
# parent allocation 140846.

set -Eeuo pipefail
umask 077

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly release_root="${experiment_root}/releases/pre-d0-paired2-edf3d1d2a77c-r2"
readonly launch_root="${experiment_root}/launchers/pre-d0-paired2-edf3d1d2a77c-r2-v1"
readonly attempt_root="${experiment_root}/attempts/pre-d0-paired2-edf3d1d2a77c-r2"
readonly output="${experiment_root}/runs/pre_d0_engineering_paired2-edf3d1d2a77c-r2"
readonly runner_sha=edf3d1d2a77cb2f713968f537ce85a7d92f0b7347a0474419fe5562fbd319bd9
readonly release_manifest_sha=671179995a64f20ee773273e84b5eb3f1f0bbd018fbfa3c0c6dc41d56c5555f5
readonly release_member_set_sha=b2556330c45cc8db8b8b6497e821fd773fe724113c8bb1860a5b343301776306
readonly step_payload="${launch_root}/auh_pre_d0_paired2_node279_step_r2_v1.sh"
readonly step_payload_sha=9d9835edaa085c7d93bb744f7bca891a86aaddc264a07c1899ef91bcd30b8493
readonly rank_exec="${launch_root}/auh_pre_d0_paired2_node279_rank_exec_r2_v1.sh"
readonly rank_exec_sha=20d0b79c5c981c76cea08c1f85318e22c398a1440c0fb638f4b031e63948a543
readonly expected_parent_state='RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8'

fail() {
  printf 'PRE_D0 r2 node279 controller refused: %s\n' "$*" >&2
  exit 95
}

[[ -d "${launch_root}" && ! -L "${launch_root}" ]] || fail "launch root differs"
[[ "$(stat -c %a "${launch_root}")" == 555 ]] || fail "launch root mode differs"
[[ -x "${step_payload}" && ! -L "${step_payload}" ]] || fail "step payload differs"
[[ "$(stat -c %a "${step_payload}")" == 555 ]] || fail "step payload mode differs"
[[ "$(stat -c %h "${step_payload}")" == 1 ]] || fail "step payload link count differs"
[[ "$(sha256sum "${step_payload}" | awk '{print $1}')" == "${step_payload_sha}" ]] || fail "step payload SHA differs"
[[ -x "${rank_exec}" && ! -L "${rank_exec}" ]] || fail "rank wrapper differs"
[[ "$(stat -c %a "${rank_exec}")" == 555 ]] || fail "rank wrapper mode differs"
[[ "$(stat -c %h "${rank_exec}")" == 1 ]] || fail "rank wrapper link count differs"
[[ "$(sha256sum "${rank_exec}" | awk '{print $1}')" == "${rank_exec_sha}" ]] || fail "rank wrapper SHA differs"
[[ -d "${attempt_root}" && ! -L "${attempt_root}" ]] || fail "attempt latch root differs"
[[ "$(stat -c %a "${attempt_root}")" == 700 ]] || fail "attempt latch mode differs"
[[ ! -e "${attempt_root}/STARTED" ]] || fail "attempt was already claimed; no retry is authorized"
[[ ! -e "${attempt_root}/train.log" ]] || fail "attempt train log already exists"
[[ ! -e "${attempt_root}/controller.status" ]] || fail "attempt status already exists"
[[ ! -e "${attempt_root}/SUCCESS" ]] || fail "attempt success marker already exists"
[[ ! -e "${output}" ]] || fail "PRE_D0 r2 output is not fresh"

# mkdir is the attempt's atomic, persistent one-shot claim. Any later failure
# leaves STARTED in place and therefore cannot become an automatic relaunch.
mkdir -m 0700 "${attempt_root}/STARTED" 2>/dev/null || \
  fail "attempt was already claimed; no retry is authorized"
readonly intent_tmp="${attempt_root}/STARTED/.intent.$$.tmp"
printf 'schema=bernini-0817-pre-d0-attempt-claim-v2\njob_id=%s\nnode=%s\nrelease_root=%s\nlaunch_root=%s\noutput=%s\nrunner_sha256=%s\nrelease_manifest_sha256=%s\nrelease_member_set_sha256=%s\nstep_payload_sha256=%s\nrank_exec_sha256=%s\nautomatic_relaunch_authorized=false\n' \
  "${job_id}" "${node}" "${release_root}" "${launch_root}" "${output}" \
  "${runner_sha}" "${release_manifest_sha}" "${release_member_set_sha}" \
  "${step_payload_sha}" "${rank_exec_sha}" >"${intent_tmp}"
chmod 0400 "${intent_tmp}"
mv "${intent_tmp}" "${attempt_root}/STARTED/intent"

readonly parent_state_before="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"
[[ "${parent_state_before}" == "${expected_parent_state}" ]] || \
  fail "parent allocation state differs: ${parent_state_before}"
readonly node_numeric_steps_before="$(
  squeue --steps -h -j "${job_id}" -o '%i|%N' |
    awk -F '|' -v wanted="${node}" \
      '$1 ~ /^[0-9]+\.[0-9]+$/ && $2 == wanted { print $1 "|" $2 }' |
    LC_ALL=C sort
)"
[[ -z "${node_numeric_steps_before}" ]] || \
  fail "node279 already has a numeric child: ${node_numeric_steps_before}"

printf 'PRE_D0 r2 one-shot start job=%s node=%s output=%s parent_untouched=true\n' \
  "${job_id}" "${node}" "${output}"

set +e
set -o noclobber
/usr/bin/srun --jobid="${job_id}" --overlap --exact --nodes=1 --ntasks=1 \
  --nodelist="${node}" --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 \
  --kill-on-bad-exit=1 "${step_payload}" >"${attempt_root}/train.log" 2>&1
status=$?
set +o noclobber
set -e

readonly parent_state_after="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"
readonly status_tmp="${attempt_root}/.controller.status.$$.tmp"
printf 'schema=bernini-0817-pre-d0-controller-status-v2\njob_id=%s\nnode=%s\nchild_exit=%s\nattempt_claimed=true\nparent_state_before=%s\nparent_state_after=%s\nparent_cancelled=false\nparent_released=false\nparent_requeued=false\nautomatic_relaunch_authorized=false\n' \
  "${job_id}" "${node}" "${status}" "${parent_state_before}" \
  "${parent_state_after}" >"${status_tmp}"
chmod 0400 "${status_tmp}"
mv "${status_tmp}" "${attempt_root}/controller.status"

if (( status != 0 )); then
  printf 'PRE_D0 r2 child failed rc=%s; no retry is authorized\n' "${status}" >&2
  tail -n 240 "${attempt_root}/train.log" >&2 || true
  exit "${status}"
fi
[[ "${parent_state_after}" == "${expected_parent_state}" ]] || \
  fail "parent allocation terminal state differs: ${parent_state_after}"

readonly receipt="${output}/receipt.json"
[[ -f "${receipt}" && ! -L "${receipt}" ]] || fail "terminal receipt is missing"
/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12 -I -B -c '
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
runner_sha, manifest_sha, member_set_sha = sys.argv[2:]
value = json.loads(path.read_text(encoding="utf-8"))
claimed_digest = value["receipt_digest"]
unsigned = dict(value)
del unsigned["receipt_digest"]
payload = json.dumps(
    unsigned,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
assert hashlib.sha256(payload).hexdigest() == claimed_digest
assert value["authority"] == "PRE_D0_ENGINEERING_ONLY"
assert value["complete"] is True
assert value["promotable"] is False
assert value["formal_training_started"] is False
assert value["counts_as_d0"] is False
assert value["scientific_claim_authorized"] is False
assert value["optimizer_steps"] == 2
assert value["checkpoint_steps"] == [0, 1, 2]
assert value["all_checkpoints_rank0_full_trainable_optimizer_roundtrip_reloaded"] is True
assert value["all_checkpoints_all8_runtime_state_bytes_persisted"] is True
assert value["all_checkpoints_rank0_runtime_state_roundtrip_reloaded"] is True
assert value["parent_allocation_released"] is False
provenance = value["provenance"]
assert provenance["runner_source_sha256"] == runner_sha
closure = provenance["release_closure"]
assert closure["sha256"] == manifest_sha
assert closure["member_set_sha256"] == member_set_sha
print("PASS PRE_D0 r2 terminal receipt", claimed_digest, flush=True)
' "${receipt}" "${runner_sha}" "${release_manifest_sha}" "${release_member_set_sha}"
readonly receipt_sha="$(sha256sum "${receipt}" | awk '{print $1}')"
readonly success_tmp="${attempt_root}/.SUCCESS.$$.tmp"
printf 'PRE_D0_ENGINEERING_COMPLETE=true\nreceipt_sha256=%s\nrunner_sha256=%s\nrelease_manifest_sha256=%s\nrelease_member_set_sha256=%s\npromotable=false\nformal_training_started=false\nparent_untouched=true\nautomatic_relaunch_authorized=false\n' \
  "${receipt_sha}" "${runner_sha}" "${release_manifest_sha}" \
  "${release_member_set_sha}" >"${success_tmp}"
chmod 0400 "${success_tmp}"
mv "${success_tmp}" "${attempt_root}/SUCCESS"
printf 'PRE_D0 r2 two-update engineering smoke complete receipt_sha256=%s parent_untouched=true\n' \
  "${receipt_sha}"

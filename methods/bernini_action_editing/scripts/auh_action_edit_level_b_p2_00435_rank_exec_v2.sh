#!/usr/bin/env bash
# Exact per-rank boundary for the one-shot PRE_D0 Level-B P2 render.

set -Eeuo pipefail

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly tag=fresh-world8-level-b-p2-00435-v2
readonly launch_root="${experiment_root}/launchers/${tag}"
readonly rank_self="${launch_root}/auh_action_edit_level_b_p2_00435_rank_exec_v2.sh"
readonly bootstrap="${launch_root}/action_edit_level_b_p2_00435_bootstrap_0817_v2.py"
readonly bootstrap_sha=104ff054cdb695000c971bc2724ff36505735cd11d40127071dd9555b10a3af3

fail() {
  printf 'Level-B P2 rank wrapper refused: %s\n' "$*" >&2
  exit 97
}

[[ "${bootstrap_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launcher authority SHA differs"
[[ $# == 2 && "$1" == "${bootstrap}" && "$2" == run ]] || fail "exact bootstrap argv differs"
[[ "$0" == "${rank_self}" ]] || fail "rank wrapper absolute path differs"
[[ -x "${rank_self}" && ! -L "${rank_self}" ]] || fail "rank wrapper file differs"
[[ "$(stat -c %a "${rank_self}")" == 555 ]] || fail "rank wrapper mode differs"
[[ "$(stat -c %h "${rank_self}")" == 1 ]] || fail "rank wrapper link count differs"
[[ "${SLURM_JOB_ID:-}" == "${job_id}" ]] || fail "parent job identity differs"
[[ "$(hostname -s)" == "${node}" ]] || fail "physical node differs"
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numeric Slurm child is absent"
[[ "${WORLD_SIZE:-}" == 8 && "${LOCAL_WORLD_SIZE:-}" == 8 ]] || fail "WORLD8 differs"
[[ "${LOCAL_RANK:-}" =~ ^[0-7]$ && "${RANK:-}" =~ ^[0-7]$ ]] || fail "rank geometry differs"
[[ "${TORCHELASTIC_MAX_RESTARTS:-}" == 0 ]] || fail "elastic max restarts differs"
[[ "${TORCHELASTIC_RESTART_COUNT:-}" == 0 ]] || fail "elastic restart occurred"
[[ "${PYTHONHASHSEED:-}" == 0 ]] || fail "Python hash seed differs"
[[ "${HF_HUB_OFFLINE:-}" == 1 && "${TRANSFORMERS_OFFLINE:-}" == 1 ]] || fail "offline mode differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "pinned Python differs"
[[ "$(sha256sum "${python_bin}" | awk '{print $1}')" == "${python_sha}" ]] || fail "pinned Python SHA differs"
[[ -f "${bootstrap}" && ! -L "${bootstrap}" ]] || fail "bootstrap file differs"
[[ "$(stat -c %a "${bootstrap}")" == 444 && "$(stat -c %h "${bootstrap}")" == 1 ]] || fail "bootstrap topology differs"
[[ "$(sha256sum "${bootstrap}" | awk '{print $1}')" == "${bootstrap_sha}" ]] || fail "bootstrap SHA differs"

exec "${python_bin}" -I -B "$@"

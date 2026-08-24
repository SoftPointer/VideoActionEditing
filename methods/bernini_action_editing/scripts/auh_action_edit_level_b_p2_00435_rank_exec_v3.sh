#!/usr/bin/env bash
# Exact per-rank boundary for the one-shot PRE_D0 Level-B P2 render.

set -Eeuo pipefail

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly tag=fresh-world8-level-b-p2-00435-v3
readonly launch_root="${experiment_root}/launchers/${tag}"
readonly rank_self="${launch_root}/auh_action_edit_level_b_p2_00435_rank_exec_v3.sh"
readonly bootstrap="${launch_root}/action_edit_level_b_p2_00435_bootstrap_0817_v3.py"
readonly bootstrap_sha=0c7d0e28dedc9a22fe543faec5f5c4f4abba628445d1f8a7f72c9138ccc6fe00

fail() {
  printf 'Level-B P2 rank wrapper refused: %s\n' "$*" >&2
  exit 97
}

require_stat_value() {
  local stat_path="$1"
  local stat_format="$2"
  local stat_expected="$3"
  local stat_label="$4"
  local stat_observed
  if ! stat_observed="$(stat -c "${stat_format}" "${stat_path}")"; then
    fail "${stat_label}: stat query failed"
  fi
  [[ "${stat_observed}" == "${stat_expected}" ]] || fail "${stat_label} differs"
}

require_sha256() {
  local sha_path="$1"
  local sha_expected="$2"
  local sha_label="$3"
  local sha_observed
  if ! sha_observed="$(sha256sum "${sha_path}" | awk '{print $1}')"; then
    fail "${sha_label}: SHA query failed"
  fi
  [[ "${sha_observed}" =~ ^[0-9a-f]{64}$ ]] || fail "${sha_label}: SHA format differs"
  [[ "${sha_observed}" == "${sha_expected}" ]] || fail "${sha_label}: SHA differs"
}

[[ "${bootstrap_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launcher authority SHA differs"
[[ $# == 2 && "$1" == "${bootstrap}" && "$2" == run ]] || fail "exact bootstrap argv differs"
[[ "$0" == "${rank_self}" ]] || fail "rank wrapper absolute path differs"
[[ -x "${rank_self}" && ! -L "${rank_self}" ]] || fail "rank wrapper file differs"
require_stat_value "${rank_self}" %a 555 "rank wrapper mode"
require_stat_value "${rank_self}" %h 1 "rank wrapper link count"
[[ "${SLURM_JOB_ID:-}" == "${job_id}" ]] || fail "parent job identity differs"
observed_hostname=
if ! observed_hostname="$(hostname -s)"; then
  fail "physical hostname query failed"
fi
readonly observed_hostname
[[ "${observed_hostname}" == "${node}" ]] || fail "physical node differs"
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numeric Slurm child is absent"
[[ "${WORLD_SIZE:-}" == 8 && "${LOCAL_WORLD_SIZE:-}" == 8 ]] || fail "WORLD8 differs"
[[ "${LOCAL_RANK:-}" =~ ^[0-7]$ && "${RANK:-}" =~ ^[0-7]$ ]] || fail "rank geometry differs"
[[ "${TORCHELASTIC_MAX_RESTARTS:-}" == 0 ]] || fail "elastic max restarts differs"
[[ "${TORCHELASTIC_RESTART_COUNT:-}" == 0 ]] || fail "elastic restart occurred"
[[ "${PYTHONHASHSEED:-}" == 0 ]] || fail "Python hash seed differs"
[[ "${HF_HUB_OFFLINE:-}" == 1 && "${TRANSFORMERS_OFFLINE:-}" == 1 ]] || fail "offline mode differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "pinned Python differs"
require_sha256 "${python_bin}" "${python_sha}" "pinned Python"
[[ -f "${bootstrap}" && ! -L "${bootstrap}" ]] || fail "bootstrap file differs"
require_stat_value "${bootstrap}" %a 444 "bootstrap mode"
require_stat_value "${bootstrap}" %h 1 "bootstrap link count"
require_sha256 "${bootstrap}" "${bootstrap_sha}" "bootstrap"

exec "${python_bin}" -I -B "$@"

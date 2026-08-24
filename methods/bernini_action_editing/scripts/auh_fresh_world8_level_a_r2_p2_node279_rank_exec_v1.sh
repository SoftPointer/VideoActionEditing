#!/usr/bin/env bash
# Exact per-rank interpreter boundary for one frozen PRE_D0 Level-A consumer.

set -Eeuo pipefail

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly release_root="${experiment_root}/releases/fresh-world8-level-a-r2-p2-launchbound-v2"
readonly launch_root="${experiment_root}/launchers/fresh-world8-level-a-r2-p2-launchbound-v2"
readonly rank_self="${launch_root}/auh_fresh_world8_level_a_r2_p2_node279_rank_exec_v1.sh"
readonly driver="${release_root}/action_edit_fresh_world8_level_a_driver_0817_v1.py"
readonly driver_sha=6435c6bb06a79cfcb407c137571404e5962e0de50e8082e7bd600e4618c05ea4

fail() {
  printf 'Level-A P2 rank wrapper refused: %s\n' "$*" >&2
  exit 97
}

[[ $# -ge 2 ]] || fail "missing frozen driver command argv"
[[ "$0" == "${rank_self}" ]] || fail "rank wrapper must be invoked by its frozen absolute path"
[[ -x "${rank_self}" && ! -L "${rank_self}" ]] || fail "rank wrapper file differs"
[[ "$(stat -c %a "${rank_self}")" == 555 ]] || fail "rank wrapper mode differs"
[[ "$(stat -c %h "${rank_self}")" == 1 ]] || fail "rank wrapper link count differs"
[[ "$1" == "${driver}" ]] || fail "driver path differs"
[[ "$2" == run ]] || fail "rank wrapper accepts only the run command"
[[ "${SLURM_JOB_ID:-}" == "${job_id}" ]] || fail "parent job identity differs"
[[ "$(hostname -s)" == "${node}" ]] || fail "physical node differs"
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numeric Slurm step is absent"
[[ "${WORLD_SIZE:-}" == 8 ]] || fail "WORLD_SIZE is not 8"
[[ "${LOCAL_WORLD_SIZE:-}" == 8 ]] || fail "LOCAL_WORLD_SIZE is not 8"
[[ "${LOCAL_RANK:-}" =~ ^[0-7]$ ]] || fail "LOCAL_RANK is outside 0..7"
[[ "${RANK:-}" =~ ^[0-7]$ ]] || fail "RANK is outside 0..7"
[[ "${TORCHELASTIC_MAX_RESTARTS:-}" == 0 ]] || fail "elastic max restarts is not zero"
[[ "${TORCHELASTIC_RESTART_COUNT:-}" == 0 ]] || fail "elastic restart count is not zero"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "pinned Python differs"
[[ -f "${driver}" && ! -L "${driver}" ]] || fail "driver file differs"
[[ "$(stat -c %a "${driver}")" == 444 ]] || fail "driver mode differs"
[[ "$(stat -c %h "${driver}")" == 1 ]] || fail "driver link count differs"
[[ "$(sha256sum "${driver}" | awk '{print $1}')" == "${driver_sha}" ]] || fail "driver SHA differs"

exec "${python_bin}" -I -B "$@"

#!/usr/bin/env bash
set -Eeuo pipefail

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818
release="${root}/release_2ad469e5_9ea76025"
eval_root="${root}/evaluation/exact28_matched_seed3_v1"
log="${root}/logs/eval_exact28_matched_seed3_v1.log"
[[ ! -e "${log}" ]]

export FULL644_EVAL_SELECTION="${root}/data/exact28_eval_selection_v1.json"
export FULL644_EVAL_ROOT="${eval_root}"
export FULL644_EVAL_RUNNER="${release}/auh_infer_full644_exact28_one_v1.sh"
export FULL644_EVAL_INFER_ARCHIVE="${release}/full644_exact28_infer_source_v2.tar"
export FULL644_EVAL_INFER_ARCHIVE_SHA256=0d0c4df1edc68e69ebf535ff968eb63629d08018e5f8dd29abfde5746fc40383
export FULL644_EVAL_INFER_REVISION=0da23167e4158daf0875095243fe3816c4e16960

exec "${release}/auh_eval_full644_exact28_v2.sh" >"${log}" 2>&1

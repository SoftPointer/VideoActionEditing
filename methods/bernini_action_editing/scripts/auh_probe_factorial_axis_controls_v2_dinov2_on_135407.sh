#!/usr/bin/env bash
# Run the no-update factorial-axis v2 DINO shortcut probe inside holder 135407.

set -Eeuo pipefail
umask 077

readonly holder_job=135407
readonly expected_node=auh7-1b-gpu-260
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/factorial_branch_core18_v1/smoke2_20260813T144200Z
readonly input_root=${experiment_root}/axis-controls-v2-a3
readonly diagnostic=${experiment_root}/probe_factorial_axis_controls_v2_dinov2.py
readonly diagnostic_sha=3c97e23fb17022699c394293ce11549ab9a925e5cd39b01cf1596aed824893d0
readonly media_manifest_sha=3625e308a4fb6ddd55424bb8860568bc64161c046d808c090355ab4fa2ec2f17
readonly output_root=${experiment_root}/axis-controls-v2-a3-dinov2-probe-r3
readonly completion=${experiment_root}/AXIS_CONTROLS_V2_A3_DINOV2_PROBE_R3_COMPLETE
readonly runtime_scratch=/tmp/factorial-axis-v2-dino-r3-135407
readonly python_bin=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/runtime/venv-transformers-4.53.2/bin/python
readonly evaluator_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808
readonly checkpoint=${evaluator_root}/vendor/dinov2-base-f9e44c8
readonly checkpoint_manifest=${evaluator_root}/inputs/dinov2-base-f9e44c8.sha256
readonly evaluator_spec=${evaluator_root}/inputs/pair_v5_source_bound_preservation_evaluator_7c4c837_v1.json
readonly visual_scorer=${evaluator_root}/inputs/source-preservation-7c4c837-minimal/methods/bernini_action_editing/score_pair_v5_source_bound_preservation_v1.py

fail() { echo "[factorial-axis-v2-dino] ERROR: $*" >&2; exit 2; }
[[ "$(hostname)" == "${expected_node}" ]] || fail "node differs"
record="$(scontrol show job -o "${holder_job}")"
[[ "${record}" == *"JobState=RUNNING"* && "${record}" == *"NodeList=${expected_node}"* ]] || fail "holder differs"
[[ -f "${diagnostic}" && ! -L "${diagnostic}" ]] || fail "diagnostic source differs"
[[ "$(sha256sum "${diagnostic}" | awk '{print $1}')" == "${diagnostic_sha}" ]] || fail "diagnostic SHA differs"
[[ -f "${input_root}/media.sha256" && ! -L "${input_root}/media.sha256" ]] || fail "media manifest differs"
[[ "$(sha256sum "${input_root}/media.sha256" | awk '{print $1}')" == "${media_manifest_sha}" ]] || fail "media manifest SHA differs"
[[ -x "${python_bin}" && -d "${checkpoint}" ]] || fail "frozen DINO runtime differs"
for plain_file in "${checkpoint_manifest}" "${evaluator_spec}" "${visual_scorer}"; do
  [[ -f "${plain_file}" && ! -L "${plain_file}" ]] || fail "sealed evaluator input differs"
done
[[ ! -e "${output_root}" && ! -L "${output_root}" && ! -e "${completion}" && ! -L "${completion}" ]] || fail "output must be fresh"
[[ ! -e "${runtime_scratch}" && ! -L "${runtime_scratch}" ]] || fail "runtime scratch must be fresh"
mkdir -m 0700 "${runtime_scratch}"
for child in home tmp xdg-cache miopen-user miopen-custom torch-extensions triton torchinductor pycache; do
  mkdir -m 0700 "${runtime_scratch}/${child}"
done

env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 GPU_DEVICE_ORDINAL=0 \
  HOME="${runtime_scratch}/home" TMPDIR="${runtime_scratch}/tmp" \
  XDG_CACHE_HOME="${runtime_scratch}/xdg-cache" \
  MIOPEN_USER_DB_PATH="${runtime_scratch}/miopen-user" \
  MIOPEN_CUSTOM_CACHE_DIR="${runtime_scratch}/miopen-custom" \
  TORCH_EXTENSIONS_DIR="${runtime_scratch}/torch-extensions" \
  TRITON_CACHE_DIR="${runtime_scratch}/triton" \
  TORCHINDUCTOR_CACHE_DIR="${runtime_scratch}/torchinductor" \
  PYTHONPYCACHEPREFIX="${runtime_scratch}/pycache" \
  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${python_bin}" "${diagnostic}" \
    --input-root "${input_root}" \
    --expected-media-manifest-sha256 "${media_manifest_sha}" \
    --expected-source-sha256 "${diagnostic_sha}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-manifest "${checkpoint_manifest}" \
    --evaluator-spec "${evaluator_spec}" \
    --visual-scorer "${visual_scorer}" \
    --output-root "${output_root}"

sha256sum "${output_root}/report.json" "${output_root}/summary.md" >"${output_root}/artifacts.sha256"
printf '%s COMPLETE holder_retained=%s output=%s\n' "$(date -u +%FT%TZ)" "${holder_job}" "${output_root}" | tee "${completion}"

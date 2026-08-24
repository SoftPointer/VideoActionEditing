#!/usr/bin/env bash
set -Eeuo pipefail

# Submit the second, appearance-distinct member for three Bernini identity
# orbits.  Each Slurm job consumes one full 8-GPU node as two independent
# WORLD=4 Ulysses groups and renders native exact81/25fps/40-step T2V, R2V,
# and RV2V arms.  Only R2V/RV2V can become orbit candidates after an external
# same-motion/same-camera qualification seal; T2V is an audit control.

python_bin="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
experiment_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_identity_orbit_v2_20260808_74ed30c"
prior_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c"
source_archive="${prior_root}/runtime/bernini_appearance_cf_74ed30c.tar"
source_archive_sha256="9718a63734f6e2deb9f622c7b15dda5ff4f84a51f0ac647886a7a52ccafc9aa3"
source_revision="74ed30c7d6a95b5097b6cd8b3332e8108b671428"
launcher="${prior_root}/runtime/methods/bernini_action_editing/scripts/auh_infer_native_identity_generation_dual4.sbatch"
launcher_sha256="2b3c90a4216fdac48a6bf514bfa1ca758d3d7d91dab93bd1cc2ae7437ac529c5"
bernini_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591"
veomni_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
checkpoint="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
checkpoint_manifest="${prior_root}/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256"

fail() {
  echo "[identity-orbit-v2-submit] ERROR: $*" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

sha256_text() {
  "${python_bin}" -B - "$1" <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest())
PY
}

for path in \
  "${python_bin}" "${source_archive}" "${launcher}" "${bernini_root}" \
  "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}"
do
  [[ -e "${path}" && ! -L "${path}" ]] || fail "missing or linked dependency: ${path}"
done
[[ "$(sha256_file "${source_archive}")" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(sha256_file "${launcher}")" == "${launcher_sha256}" ]] || fail "launcher hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision differs"

mkdir -p -- "${experiment_root}/runs" "${experiment_root}/slurm"
[[ -d "${experiment_root}" && ! -L "${experiment_root}" ]] || fail "experiment root is invalid"

submit_one() {
  local iid="$1"
  local run_name="$2"
  local source_video="$3"
  local source_sha256="$4"
  local prompt="$5"
  local output_dir="${experiment_root}/runs/${run_name}"
  local prompt_sha256
  local job_id

  [[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source for ${iid}"
  [[ "$(sha256_file "${source_video}")" == "${source_sha256}" ]] || fail "source hash differs for ${iid}"
  [[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing existing output: ${output_dir}"
  prompt_sha256="$(sha256_text "${prompt}")"

  job_id="$(env \
    BERNINI_NATIVE_SOURCE_ARCHIVE="${source_archive}" \
    BERNINI_NATIVE_SOURCE_ARCHIVE_SHA256="${source_archive_sha256}" \
    BERNINI_NATIVE_SOURCE_REVISION="${source_revision}" \
    BERNINI_OFFICIAL_ROOT="${bernini_root}" \
    BERNINI_VEOMNI_ROOT="${veomni_root}" \
    BERNINI_ACTION_CHECKPOINT="${checkpoint}" \
    BERNINI_CHECKPOINT_CONTENT_MANIFEST="${checkpoint_manifest}" \
    BERNINI_NATIVE_SOURCE_VIDEO="${source_video}" \
    BERNINI_NATIVE_SOURCE_SHA256="${source_sha256}" \
    BERNINI_NATIVE_ACTION_PROMPT="${prompt}" \
    BERNINI_NATIVE_ACTION_PROMPT_SHA256="${prompt_sha256}" \
    BERNINI_NATIVE_OUTPUT_DIR="${output_dir}" \
    BERNINI_NATIVE_PYTHON_BIN="${python_bin}" \
    sbatch --parsable \
      --job-name="id2-${iid:0:8}" \
      --output="${experiment_root}/slurm/${run_name}-%j.out" \
      --error="${experiment_root}/slurm/${run_name}-%j.err" \
      "${launcher}")"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "unexpected sbatch response for ${iid}: ${job_id}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${iid}" "${run_name}" "${source_sha256}" "${prompt_sha256}"
}

case_selector="${1:-initial}"
if [[ "${case_selector}" == "person-b-retry" ]]; then
  person_b_retry_source="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/00435ad621c44fac/samples/00435ad621c44fac/source_video.mp4"
  person_b_retry_prompt="Edit only the appearance of the single female athlete seen from behind: replace her with one East Asian woman with long straight auburn hair, wearing a cobalt-blue fitted sports top and navy training pants. Preserve the exact original torso pose, both arm and hand trajectories while she handles her hair, the hair-motion timing, facing direction, spatial position, and full body motion. Preserve the window, room, every object and background detail, lighting, framing, and the complete original locked camera for all 81 frames. Keep the replacement identity and outfit consistent from the first through the last frame. Do not add or remove anyone and do not invent a new action or camera view; this is appearance replacement only."
  submit_one \
    "00435ad621c44fac" \
    "00435ad621c44fac-east-asian-woman-blue-longhair-retry-seed2027" \
    "${person_b_retry_source}" \
    "b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1" \
    "${person_b_retry_prompt}"
  echo "[identity-orbit-v2-submit] submitted=1 retry=person-b exact_frames=81 fps=25 steps=40 gpus_per_job=8"
  exit 0
fi
if [[ "${case_selector}" == "person-b-retry2" ]]; then
  person_b_retry2_source="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/00435ad621c44fac/samples/00435ad621c44fac/source_video.mp4"
  person_b_retry2_prompt="Edit only the appearance of the single female athlete seen from behind: replace her with one older dark-skinned woman with waist-length silver-gray hair, wearing an emerald-green fitted sports top and charcoal training pants. The replacement identity, skin, hair color, hair length, and outfit must already be present in frame 0 and remain unchanged through frame 80; no gradual identity or clothing transition is allowed. Preserve the exact original torso pose, both arm and hand trajectories while she handles her hair, the hair-motion timing, facing direction, spatial position, and full body motion. Preserve the window, room, every object and background detail, lighting, framing, and the complete original locked camera for all 81 frames. Do not add or remove anyone and do not invent a new action or camera view; this is appearance replacement only."
  submit_one \
    "00435ad621c44fac" \
    "00435ad621c44fac-older-woman-green-silverhair-retry2-seed2027" \
    "${person_b_retry2_source}" \
    "b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1" \
    "${person_b_retry2_prompt}"
  echo "[identity-orbit-v2-submit] submitted=1 retry=person-b-retry2 exact_frames=81 fps=25 steps=40 gpus_per_job=8"
  exit 0
fi
[[ "${case_selector}" == "initial" ]] || fail "case selector must be initial, person-b-retry, or person-b-retry2"

dog_source="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/288545b9c031491a/source.mp4"
dog_prompt="Edit only the dog's appearance: replace the tan pit bull with one white Siberian husky wearing a blue collar. Preserve the dog's exact original pose, head and mouth motion, gait, timing, spatial trajectory, and interaction with the bone. Preserve the bone and every other object, the ground and background, lighting, framing, the overhead camera pose, and the complete original camera path for all 81 frames. Do not add or remove any subject or object. Do not invent a new action or camera view; this is appearance replacement only."
submit_one \
  "288545b9c031491a" \
  "cdfdog-white-husky-seed2027" \
  "${dog_source}" \
  "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18" \
  "${dog_prompt}"

person_a_source="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/0014a41e55e44670/samples/0014a41e55e44670/source_video.mp4"
person_a_prompt="Edit only the appearance of the kneeling person in the brown blazer: replace that person with one young Black woman wearing a mustard-yellow suit. Preserve her exact source-derived kneeling body pose, hand placement on the pregnant woman's belly, head motion, facial orientation, timing, and every interaction. Keep the standing pregnant woman in the beige gown completely unchanged in identity, clothing, pose, motion, and timing. Preserve the stool, green curtain, every other scene detail, lighting, framing, and the complete original camera path for all 81 frames. Do not add or remove anyone or invent a new action or camera view; this is appearance replacement of the kneeling person only."
submit_one \
  "0014a41e55e44670" \
  "0014a41e55e44670-black-woman-yellow-suit-seed2027" \
  "${person_a_source}" \
  "b0255970cdbb42375cd783e8f2ab9b8099d5f02ec96f62085be77e24eb5f2437" \
  "${person_a_prompt}"

person_b_source="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/00435ad621c44fac/samples/00435ad621c44fac/source_video.mp4"
person_b_prompt="Edit only the appearance of the single athlete seen from behind: replace that person with one short-haired muscular East Asian man wearing a royal-blue compression shirt and black training pants. Preserve the exact original torso pose, arm and hand trajectory around the head, timing, facing direction, spatial position, and full body motion. Preserve the window, room, every object and background detail, lighting, framing, and the complete original locked camera for all 81 frames. Do not add or remove anyone and do not invent a new action or camera view; this is appearance replacement only."
submit_one \
  "00435ad621c44fac" \
  "00435ad621c44fac-east-asian-man-blue-shirt-seed2027" \
  "${person_b_source}" \
  "b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1" \
  "${person_b_prompt}"

echo "[identity-orbit-v2-submit] submitted=3 exact_frames=81 fps=25 steps=40 gpus_per_job=8"

#!/usr/bin/env bash
# Launch the four source-only activation-v2 materializers inside allocation 140846.
# This does not launch sampling, training, or an optimizer.

set -Eeuo pipefail

fail() {
  echo "[oracle-materializer-outer] ERROR: $*" >&2
  exit 2
}

stage="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1"
bundle="${stage}/oracle_activation_v2_materializer_r2_ef97259d_07b40cdd_5bba9f97_6da8c414"
run_root="${stage}/oracle_activation_v2_materializations_ef97259d_07b40cdd_5bba9f97_r4"
launcher="${bundle}/methods/bernini_action_editing/scripts/auh_materialize_oracle_regeneration_activation_v2_r2.sh"
bernini_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591"
veomni_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
checkpoint="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
checkpoint_manifest="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256"

[[ -z "${SLURM_JOB_ID:-}" ]] || fail "outer must run on the login node, not inside any allocation step"
[[ -d "${bundle}" && ! -L "${bundle}" ]] || fail "content-addressed bundle differs"
[[ "$(stat -c '%a' "${bundle}")" == "555" ]] || fail "bundle root mode differs"
[[ -x "${launcher}" && ! -L "${launcher}" ]] || fail "materializer launcher differs"
[[ "$(stat -c '%a:%h' "${launcher}")" == "555:1" ]] || fail "materializer launcher mode or link count differs"
[[ "$(sha256sum "${launcher}" | awk '{print $1}')" == "6da8c414006bfe72498df8ab0a1ad288541230cb8011f9fa14062c09122f77d0" ]] || fail "materializer launcher bytes differ"
declare -A transitive_sha256=(
  [oracle_regeneration_activation_v2.py]="ef97259dd181ff065267e32f1e5cca158e26ad5174457780163658a3db728bb0"
  [oracle_regeneration_canary_v1.py]="0148b137c200e426ff18571f71d373a9e6ef595c620664925dae0ab9d1d91081"
  [native_branch_homotopy_runtime_v1.py]="b81ee152e358e4d5a6638dfccf1232c4e221311ffb38937e61be3c6a799b84d5"
  [native_branch_homotopy_v1.py]="2585416e61935db62cc7534daf19b4bb851f9fdcdeb92f78e6152f55e034f3d0"
  [self_guided_action_field_v1.py]="2ad204c09f5eb60865017b1e596de25b777d8d6ed43774f4dcbc23a4ad58bc7e"
  [tri_branch_unipc.py]="58d2e0e8d56a500eea07ec20f0fb101539ac846bbd039c0d50a22506b58fb3d2"
  [infer_native_identity_generation_canary.py]="bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42"
  [infer_native_branch_homotopy_canary.py]="d6dab735ce52da151848c96f9e00775994dc281ace20afa6dcb9fb64709e5983"
  [infer_source_kv_carrier_oracle.py]="fcf77576735c89e685415b94b2dc0f0c5b8d1dd8dc1c55832538ff0daafb4604"
  [infer_source_value_residual_oracle.py]="40e581db7906f20103a16ad47fda76978cbad21c9277723f3e8e022d717ed2d8"
  [infer_native_self_guided_action_field_canary.py]="ad591fe5bd5943fab59603400fcf70a126f3461a39670f93a78aa24b1902d313"
  [infer_lora.py]="acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
  [train_lora.py]="8e8daf422548bc29e2c18f2d2c692af2dd3109aaad1897fc31e590a69d7e593e"
  [action_preservation_decoded_eval_model_authority_v2.py]="760ed9988147a44965fd47f68a08fd353ce1d900e661b55bb818088ec9ef848e"
  [source_kv_replay.py]="45b43426dc7825dbd61280154fc35161c60476ec5cb9e53bc0225f3809c759f3"
  [source_kv_route_batches.py]="7f3ae0d27747ad58b3b195c712884641012eb836bb59963896c58518b8b5731e"
  [source_value_residual.py]="420cadf3cb2824b2bf5a809c55086d81351db19f31743b0b77a957adf219e124"
  [tools/__init__.py]="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  [tools/materialize_vae.py]="a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
  [tools/build_renderer_dataset.py]="afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
  [tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py]="07b40cdd67771d257ce546ca4166301980c6768269acc5f097fc08973656bbde"
  [tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py]="5bba9f977fa40e5044053baaaf73eba779b3816ef6137457a06ceac82a3463af"
  [tests/test_oracle_regeneration_activation_v2_materializers_r2.py]="e8bfc32e4c5fb13b8ba4e6417c61ef1c7c6e7e204cec7a348f0d99bbd23da859"
)
for relative_path in "${!transitive_sha256[@]}"; do
  dependency="${bundle}/methods/bernini_action_editing/${relative_path}"
  [[ -f "${dependency}" && ! -L "${dependency}" ]] || fail "transitive dependency differs: ${relative_path}"
  [[ "$(stat -c '%a:%h' "${dependency}")" == "444:1" ]] || fail "transitive dependency mode or link count differs: ${relative_path}"
  [[ "$(sha256sum "${dependency}" | awk '{print $1}')" == "${transitive_sha256[${relative_path}]}" ]] || fail "transitive dependency bytes differ: ${relative_path}"
done
expected_members=(
  "d|555|methods"
  "d|555|methods/bernini_action_editing"
  "d|555|methods/bernini_action_editing/scripts"
  "d|555|methods/bernini_action_editing/tests"
  "d|555|methods/bernini_action_editing/tools"
  "f|555|methods/bernini_action_editing/scripts/auh_materialize_oracle_regeneration_activation_v2_r2.sh"
)
for relative_path in "${!transitive_sha256[@]}"; do
  expected_members+=("f|444|methods/bernini_action_editing/${relative_path}")
done
if ! expected_member_rows="$(printf '%s\n' "${expected_members[@]}" | LC_ALL=C sort)"; then
  fail "cannot construct expected bundle member closure"
fi
if ! actual_member_rows="$(find "${bundle}" -mindepth 1 -printf '%y|%m|%P\n' | LC_ALL=C sort)"; then
  fail "cannot inspect bundle member closure"
fi
[[ "${actual_member_rows}" == "${expected_member_rows}" ]] || fail "bundle member closure differs"
if ! step_rows="$(squeue -s -j 140846 -h -o '%i')"; then
  fail "cannot query active allocation steps"
fi
saw_batch=0
saw_extern=0
while read -r step_id; do
  [[ -n "${step_id}" ]] || continue
  case "${step_id}" in
    140846.batch) saw_batch=1 ;;
    140846.extern) saw_extern=1 ;;
    *) fail "unexpected active allocation step: ${step_id}" ;;
  esac
done <<< "${step_rows}"
(( saw_batch == 1 && saw_extern == 1 )) || fail "allocation step closure differs"
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root is not fresh"

mkdir "${run_root}"
mkdir "${run_root}/logs" "${run_root}/reservations"

labels=(e02-vae e02-prompt e03-vae e03-prompt)
nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279)
kinds=(vae prompt vae prompt)
cases=(e02 e02 e03 e03)
ports=(41821 41822 41823 41824)

set -o noclobber
for label in "${labels[@]}"; do
  [[ ! -e "${run_root}/${label}" && ! -L "${run_root}/${label}" ]] || fail "output is not fresh: ${label}"
  mkdir "${run_root}/reservations/${label}"
  : > "${run_root}/logs/${label}.log"
done
set +o noclobber

launch_arm() {
  local index="$1"
  local label="${labels[${index}]}"
  local node="${nodes[${index}]}"
  local kind="${kinds[${index}]}"
  local case_id="${cases[${index}]}"
  local port="${ports[${index}]}"
  local log="${run_root}/logs/${label}.log"

  srun --overlap --exact --jobid=140846 \
    --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=60G \
    --gres=gpu:mi210:8 --nodelist="${node}" \
    --kill-on-bad-exit=1 \
    env \
      ORACLE_ACTIVATION_V2_REPO_ROOT="${bundle}" \
      BERNINI_OFFICIAL_ROOT="${bernini_root}" \
      BERNINI_VEOMNI_ROOT="${veomni_root}" \
      BERNINI_ACTION_CHECKPOINT="${checkpoint}" \
      BERNINI_CHECKPOINT_CONTENT_MANIFEST="${checkpoint_manifest}" \
      ORACLE_MATERIALIZER_OUTPUT_DIR="${run_root}/${label}" \
      ORACLE_MATERIALIZER_KIND="${kind}" \
      ORACLE_CASE_ID="${case_id}" \
      ORACLE_VISIBLE_GPUS="0,1,2,3" \
      ORACLE_MASTER_PORT="${port}" \
      "${launcher}" >> "${log}" 2>&1
}

pids=()
for index in 0 1 2 3; do
  launch_arm "${index}" &
  pids+=("$!")
done

status=0
for index in 0 1 2 3; do
  if ! wait "${pids[${index}]}"; then
    echo "[oracle-materializer-outer] ${labels[${index}]} failed" >&2
    status=1
  fi
done

if (( status != 0 )); then
  fail "one or more materializers failed"
fi

echo "[oracle-materializer-outer] all four source-only materializers completed"

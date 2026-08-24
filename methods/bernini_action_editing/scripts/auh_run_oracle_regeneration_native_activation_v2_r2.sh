#!/usr/bin/env bash
# Frozen-bundle WORLD4 launcher for the e02 two-arm Round37 diagnostic.
#
# Run only from an already allocated AUH compute-node shell.  This script does
# not call ssh, srun, sbatch, or any optimizer/training entrypoint.  e03 is an
# authority-policy ABSTAIN row and is never sampled by this launcher.

set -Eeuo pipefail

fail() {
  echo "[oracle-regeneration-native-v2-r9] ERROR: $*" >&2
  exit 2
}

realpath_existing() {
  if realpath -e -- / >/dev/null 2>&1; then
    realpath -e -- "$1"
  else
    realpath "$1"
  fi
}

verify_local_bundle_only=false
if [[ "$#" == "1" && "$1" == "--verify-local-bundle-only" ]]; then
  verify_local_bundle_only=true
elif [[ "$#" != "0" ]]; then
  fail "unsupported launcher arguments"
fi

repo_root="${ORACLE_ACTIVATION_V2_REPO_ROOT:?set isolated frozen bundle root}"
[[ "${repo_root}" == /* ]] || fail "bundle root must be absolute"
repo_root="$(realpath_existing "${repo_root}")"
[[ -d "${repo_root}" && ! -L "${repo_root}" ]] || fail "bundle root differs"

method_root="${repo_root}/methods/bernini_action_editing"
runner="${method_root}/infer_oracle_regeneration_native_activation_v2_r2.py"
preflight="${method_root}/preflight_oracle_regeneration_native_activation_v2_r2.py"
static_test="${method_root}/tests/test_oracle_regeneration_native_activation_v2_r2.py"
lock_file="${method_root}/assets/oracle_regeneration_native_activation_v2_r2_host_load.lock"
launcher_relative="scripts/auh_run_oracle_regeneration_native_activation_v2_r2.sh"

declare -A expected_bundle_sha256=(
  ["oracle_regeneration_native_runtime_activation_v2.py"]="b8e0018893c9582d97d20446956c2bea0506fbc48c7d333a021f70f467edc0d0"
  ["infer_oracle_regeneration_native_activation_v2_r2.py"]="ee7fe068096231f222fe9cb153e754b35be3f9078dc00f3924bb7889142aa2f9"
  ["preflight_oracle_regeneration_native_activation_v2_r2.py"]="82a1b61e16d76a3ce411f708d222de16a59d46b817097d2da2dfbed11af366e0"
  ["assets/oracle_regeneration_native_activation_v2_r2_spec.json"]="edadfa5be1758aaed8b8c4c5f72354bd6becf9a3b999ad116814886e08487d7e"
  ["assets/oracle_regeneration_native_activation_v2_r2_host_load.lock"]="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  ["assets/oracle_regeneration_activation_v2_authoring_template.json"]="a4a6d3c75028c61198aae99a43367bc9f2b17f19be0c562dc2049eb6ef8299e9"
  ["oracle_regeneration_activation_v2.py"]="ef97259dd181ff065267e32f1e5cca158e26ad5174457780163658a3db728bb0"
  ["tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py"]="07b40cdd67771d257ce546ca4166301980c6768269acc5f097fc08973656bbde"
  ["tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py"]="5bba9f977fa40e5044053baaaf73eba779b3816ef6137457a06ceac82a3463af"
  ["oracle_regeneration_canary_v1.py"]="0148b137c200e426ff18571f71d373a9e6ef595c620664925dae0ab9d1d91081"
  ["native_branch_homotopy_runtime_v1.py"]="b81ee152e358e4d5a6638dfccf1232c4e221311ffb38937e61be3c6a799b84d5"
  ["native_branch_homotopy_v1.py"]="2585416e61935db62cc7534daf19b4bb851f9fdcdeb92f78e6152f55e034f3d0"
  ["self_guided_action_field_v1.py"]="2ad204c09f5eb60865017b1e596de25b777d8d6ed43774f4dcbc23a4ad58bc7e"
  ["tri_branch_unipc.py"]="58d2e0e8d56a500eea07ec20f0fb101539ac846bbd039c0d50a22506b58fb3d2"
  ["infer_native_identity_generation_canary.py"]="bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42"
  ["infer_native_branch_homotopy_canary.py"]="d6dab735ce52da151848c96f9e00775994dc281ace20afa6dcb9fb64709e5983"
  ["infer_source_kv_carrier_oracle.py"]="fcf77576735c89e685415b94b2dc0f0c5b8d1dd8dc1c55832538ff0daafb4604"
  ["infer_source_value_residual_oracle.py"]="40e581db7906f20103a16ad47fda76978cbad21c9277723f3e8e022d717ed2d8"
  ["infer_native_self_guided_action_field_canary.py"]="ad591fe5bd5943fab59603400fcf70a126f3461a39670f93a78aa24b1902d313"
  ["infer_lora.py"]="acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
  ["tools/materialize_vae.py"]="a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
  ["tools/build_renderer_dataset.py"]="afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
  ["tools/__init__.py"]="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  ["train_lora.py"]="8e8daf422548bc29e2c18f2d2c692af2dd3109aaad1897fc31e590a69d7e593e"
  ["action_preservation_decoded_eval_model_authority_v2.py"]="760ed9988147a44965fd47f68a08fd353ce1d900e661b55bb818088ec9ef848e"
  ["source_kv_replay.py"]="45b43426dc7825dbd61280154fc35161c60476ec5cb9e53bc0225f3809c759f3"
  ["source_kv_route_batches.py"]="7f3ae0d27747ad58b3b195c712884641012eb836bb59963896c58518b8b5731e"
  ["source_value_residual.py"]="420cadf3cb2824b2bf5a809c55086d81351db19f31743b0b77a957adf219e124"
  ["source_self_native_ref_contrastive_v3.py"]="d8825bc167c64e497f8d29c807d9b0a69d9a9a59de09afee863b7fc9df2bdeb0"
  ["tests/test_native_branch_homotopy_runtime_v1.py"]="b71a832a8f1cfcd2da2a2db34542eb4114aeefaada833d5005f345b36ceecc99"
  ["tests/test_oracle_regeneration_native_activation_v2_r2.py"]="93d44d7b4cc3d9af9e30a8db5e88a5075c70ce857a3c24b7a59b2c0903690a15"
)

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

stat_mode() {
  stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1"
}

stat_nlink() {
  stat -c '%h' "$1" 2>/dev/null || stat -f '%l' "$1"
}

stat_size() {
  stat -c '%s' "$1" 2>/dev/null || stat -f '%z' "$1"
}

stat_device() {
  stat -c '%d' "$1" 2>/dev/null || stat -f '%d' "$1"
}

stat_uid() {
  stat -c '%u' "$1" 2>/dev/null || stat -f '%u' "$1"
}

stat_gid() {
  stat -c '%g' "$1" 2>/dev/null || stat -f '%g' "$1"
}

stat_inode() {
  stat -c '%i' "$1" 2>/dev/null || stat -f '%i' "$1"
}

check_frozen() {
  local path="$1"
  local expected_sha="$2"
  [[ "${expected_sha}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "frozen expected SHA-256 is malformed: ${path}"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "frozen file differs: ${path}"
  [[ "$(sha256_file "${path}")" == "${expected_sha}" ]] || \
    fail "frozen bytes differ: ${path}"
  [[ "$(stat_mode "${path}")" == "444" ]] || fail "frozen mode differs: ${path}"
  [[ "$(stat_nlink "${path}")" == "1" ]] || fail "frozen nlink differs: ${path}"
}

for relative in "${!expected_bundle_sha256[@]}"; do
  check_frozen "${method_root}/${relative}" "${expected_bundle_sha256[${relative}]}"
done
[[ -f "${method_root}/${launcher_relative}" && ! -L "${method_root}/${launcher_relative}" ]] || \
  fail "launcher identity differs"
[[ "$(stat_mode "${method_root}/${launcher_relative}")" == "555" ]] || \
  fail "launcher mode differs"
[[ "$(stat_nlink "${method_root}/${launcher_relative}")" == "1" ]] || \
  fail "launcher nlink differs"
actual_bundle_files="$(
  cd "${method_root}" && find . -type f -print | sed 's#^\./##' | LC_ALL=C sort
)"
expected_bundle_files="$(
  printf '%s\n' "${!expected_bundle_sha256[@]}" "${launcher_relative}" | LC_ALL=C sort
)"
[[ "${actual_bundle_files}" == "${expected_bundle_files}" ]] || \
  fail "isolated bundle file set differs"
[[ -z "$(find "${method_root}" -type l -print -quit)" ]] || fail "bundle contains symlink"
actual_bundle_directories="$(
  cd "${method_root}" && find . -mindepth 1 -type d -print | \
    sed 's#^\./##' | LC_ALL=C sort
)"
expected_bundle_directories="$(printf '%s\n' assets scripts tests tools | LC_ALL=C sort)"
[[ "${actual_bundle_directories}" == "${expected_bundle_directories}" ]] || \
  fail "isolated bundle directory set differs"
while IFS= read -r directory; do
  [[ "$(stat_mode "${directory}")" == "555" ]] || \
    fail "bundle directory is not frozen: ${directory}"
done < <(find "${method_root}" -type d -print)

if [[ "${verify_local_bundle_only}" == "true" ]]; then
  echo "[oracle-regeneration-native-v2-r9] local frozen bundle verified"
  exit 0
fi

bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini-R checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_dir="${ORACLE_ACTIVATION_V2_OUTPUT_DIR:?set fresh e02 output directory}"
visible_gpus="${ORACLE_VISIBLE_GPUS:?set exact local GPU list}"
master_port="${ORACLE_MASTER_PORT:?set private torchrun port}"
python_bin="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
miopen_library="/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/torch/lib/libMIOpen.so"

for value in \
  "${python_bin}" "${bernini_root}" "${veomni_root}" \
  "${checkpoint}" "${checkpoint_manifest}" "${output_dir}"
do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
[[ "${visible_gpus}" == "0,1,2,3" ]] || fail "visible GPU list differs"
[[ "${master_port}" =~ ^[1-9][0-9]{3,4}$ ]] || fail "master port differs"
(( 10#${master_port} <= 65535 )) || fail "master port exceeds 65535"
[[ "${SLURM_JOB_ID:-}" == "141620" ]] || fail "allocation identity differs"
node_name="$(hostname -s)"
case "${node_name}" in
  auh7-1b-gpu-226) ;;
  *) fail "node is outside allocation 141620 allowlist" ;;
esac
[[ "${SLURM_STEP_ID:-}" =~ ^(0|[1-9][0-9]*)$ ]] || \
  fail "numbered Slurm step identity differs"

# AUH srun injects exactly TMPDIR=/tmp.  Authenticate the allocation, node, and
# numbered step above before accepting that scheduler fact.  Reject any caller
# attempt to pre-author the receipt fields, then remove TMPDIR before the first
# Python/Torch process and export an exact marker that every later receipt
# revalidates.  All other temporary-environment keys remain forbidden.
inherited_normalization="$(env | LC_ALL=C grep '^ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_' || true)"
[[ -z "${inherited_normalization}" ]] || \
  fail "caller scheduler TMPDIR normalization receipt is forbidden"
[[ ! -v ORACLE_ACTIVATION_V2_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_SHA256 ]] || \
  fail "caller launcher local-tmp empty proof is forbidden"
[[ -v TMPDIR && "${TMPDIR}" == "/tmp" ]] || \
  fail "scheduler-injected TMPDIR differs"
for temp_name in TMP TEMP TEMPDIR; do
  [[ ! -v "${temp_name}" ]] || fail "caller temporary environment is forbidden"
done
scheduler_observed_tmpdir="${TMPDIR}"
unset TMPDIR
[[ ! -v TMPDIR ]] || fail "scheduler TMPDIR normalization failed"
export ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_JOB_ID="${SLURM_JOB_ID}"
export ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_STEP_ID="${SLURM_STEP_ID}"
export ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_HOSTNAME="${node_name}"
export ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_OBSERVED="${scheduler_observed_tmpdir}"
export ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_ACTION="UNSET_BEFORE_ANY_PYTHON"
readonly \
  ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_JOB_ID \
  ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_STEP_ID \
  ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_HOSTNAME \
  ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_OBSERVED \
  ORACLE_ACTIVATION_V2_SCHEDULER_TMPDIR_ACTION
echo "[oracle-regeneration-native-v2-r9] normalized scheduler TMPDIR=/tmp before Python for Slurm step ${SLURM_STEP_ID}" >&2

python_bin="$(realpath_existing "${python_bin}")"
bernini_root="$(realpath_existing "${bernini_root}")"
veomni_root="$(realpath_existing "${veomni_root}")"
checkpoint="$(realpath_existing "${checkpoint}")"
checkpoint_manifest="$(realpath_existing "${checkpoint_manifest}")"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || \
  fail "Python executable differs"
[[ "${python_bin}" == "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12" ]] || \
  fail "Python executable path differs"
[[ "$(sha256_file "${python_bin}")" == \
  8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a ]] || \
  fail "Python executable bytes differ"
[[ "$(stat_size "${python_bin}")" == "31490256" ]] || \
  fail "Python executable size differs"
[[ -f "${miopen_library}" && ! -L "${miopen_library}" ]] || \
  fail "Torch-bundled MIOpen library path differs"
[[ "$(sha256_file "${miopen_library}")" == \
  1e6cc33ca21951dce12795e6c5d99578e8f2f1754b84a703508df44426b44b52 ]] || \
  fail "Torch-bundled MIOpen library bytes differ"
[[ "$(stat_size "${miopen_library}")" == "690355265" ]] || \
  fail "Torch-bundled MIOpen library size differs"
[[ "$(stat_mode "${miopen_library}")" == "755" ]] || \
  fail "Torch-bundled MIOpen library mode differs"
[[ "$(stat_nlink "${miopen_library}")" == "1" ]] || \
  fail "Torch-bundled MIOpen library nlink differs"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || \
  fail "checkpoint manifest differs"
[[ "$(sha256_file "${checkpoint_manifest}")" == \
  a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831 ]] || \
  fail "checkpoint manifest bytes differ"

authority_dir="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/oracle_regeneration_round37_authority_e02_r1"
packet="${authority_dir}/round37-activation-v2-authority-packet-r1.json"
ledger="${authority_dir}/round37-activation-v2-external-ledger-receipt-r1.json"
material_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/oracle_activation_v2_materializations_e02_only_ef97259d_07b40cdd_5bba9f97_r1"

declare -A expected_authority_sha256=(
  ["${authority_dir}/e02-manual-dck-gate-r3.json"]="f5c0e20f478ff11a63fa5c1ecb1d2bfb4a37e655e14f0e2c9739975a3e63dae1"
  ["${authority_dir}/e02-independent-gate-review-r1.json"]="4e060227512687060a32acc0543462be4dfe0364c46e5fe6e7e02416dac93962"
  ["${authority_dir}/e02-materializer-ai-agent-diagnostic-review-r1.json"]="788cf7b83851b79a662c0040e7470fd32a8aea2456ce47e6b53baff0b2a73c6e"
  ["${packet}"]="6ae5602350d54696e0ddcd716a311f96a3569c6f062622840ad130fcbba0baeb"
  ["${ledger}"]="5a9efae443bc8d3cb0886dee7f950204377f653f7dbc474f820d7abbbe437e51"
  ["${material_root}/e02-vae/vae-reference-receipt.json"]="b89840c3f87a0950e5d0634c3697eed7641ae06e840d1804ea30d2ebaf74611f"
  ["${material_root}/e02-vae/run-receipt.json"]="65b3acc8e2581bcc7c3311475c95bada1fd25067328627542572933fd930bbb7"
  ["${material_root}/e02-prompt/prompt-receipt.json"]="ed6e11b50651b6d30f067c4dc9d285a3b790658edfd82ac9e0e0df7c3d290924"
  ["${material_root}/e02-prompt/run-receipt.json"]="2fd4a232db890dde19cf26fa4596be1279c8756dfa60a943837705aece69da9a"
)
[[ -d "${authority_dir}" && ! -L "${authority_dir}" ]] || fail "authority dir differs"
[[ "$(stat -c '%a' "${authority_dir}")" == "555" ]] || fail "authority dir not frozen"
authority_names="$(find "${authority_dir}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
expected_authority_names="$(printf '%s\n' \
  e02-independent-gate-review-r1.json \
  e02-manual-dck-gate-r3.json \
  e02-materializer-ai-agent-diagnostic-review-r1.json \
  round37-activation-v2-authority-packet-r1.json \
  round37-activation-v2-external-ledger-receipt-r1.json | LC_ALL=C sort)"
[[ "${authority_names}" == "${expected_authority_names}" ]] || \
  fail "authority deployment exact file set differs"
for path in "${!expected_authority_sha256[@]}"; do
  check_frozen "${path}" "${expected_authority_sha256[${path}]}"
done

check_plain_hash() {
  local path="$1"
  local expected_sha="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "media file differs: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected_sha}" ]] || \
    fail "media bytes differ: ${path}"
}
check_plain_hash \
  "/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/10ed90644f81461d/source.mp4" \
  "63dd620627e18d0f6836058bf725ffba7ab1f9b0b455e784c6a71dc68a8de46c"
check_plain_hash \
  "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/interaction_complex8_multianchor_v2_r1/e02_twist-pull-mushroom/v0/t2v.mp4" \
  "a1076cbe83c9dae4a4fddc25f73077288f6a3240324fd2d5e1854aa842b07b63"
check_plain_hash \
  "/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/7a33b36459c84289/source.mp4" \
  "c1455b9b89d1f352da69e7bb07e955ee4495df94f5ef6f3f09fe7fd9eac035bb"
check_plain_hash \
  "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/interaction_complex8_multianchor_v2_r1/e03_release-harvest-into-basket/v0/t2v.mp4" \
  "1d0a0e8895ec976d3cb1f9ee3070ac36c75ca29247bb52f81677135f7786f12f"
check_plain_hash \
  "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/interaction_complex8_rv2v_candidates_v1/complex8-e03-rv2v-s0/rv2v.mp4" \
  "d75bbafbbc225ea3935c2d149be8b3969fffd6d8b645c5ec9edb5968bf25f654"

[[ "${output_dir}" != / && ! -e "${output_dir}" && ! -L "${output_dir}" ]] || \
  fail "output directory must be fresh"
output_parent="$(realpath_existing "$(dirname -- "${output_dir}")")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
[[ "${output_dir}" == "${output_parent}/${output_dir##*/}" ]] || \
  fail "output path is not canonical"
[[ "${output_dir##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || \
  fail "unsafe output basename"
case "${output_dir}/" in
  "${repo_root}/"*) fail "output/cache must be outside frozen bundle" ;;
esac

# No caller may select a MIOpen solver/cache behavior.  The launcher mints the
# only two MIOPEN_* keys after it creates a fresh private bootstrap namespace.
inherited_miopen="$(env | LC_ALL=C grep '^MIOPEN_' || true)"
[[ -z "${inherited_miopen}" ]] || fail "caller MIOPEN environment is forbidden"
for temp_name in TMPDIR TMP TEMP TEMPDIR; do
  [[ ! -v "${temp_name}" ]] || fail "caller temporary environment is forbidden"
done

cache_root="${output_dir}.miopen-cache-r9"
[[ "${cache_root}" == "${output_parent}/${output_dir##*/}.miopen-cache-r9" ]] || \
  fail "MIOpen cache sibling path differs"
[[ ! -e "${cache_root}" && ! -L "${cache_root}" ]] || \
  fail "MIOpen cache root must be fresh"
umask 077
mkdir -m 700 -- "${cache_root}" || fail "cannot create fresh MIOpen cache root"
[[ -d "${cache_root}" && ! -L "${cache_root}" ]] || \
  fail "created MIOpen cache root identity differs"
cache_directory_names=(launcher-bootstrap-user-db launcher-bootstrap-kernel-cache)
for rank in 0 1 2 3; do
  cache_directory_names+=(
    "rank-${rank}-user-db"
    "rank-${rank}-kernel-cache"
  )
done
for name in "${cache_directory_names[@]}"; do
  mkdir -m 700 -- "${cache_root}/${name}" || \
    fail "cannot create MIOpen cache role directory: ${name}"
done
expected_cache_names="$(printf '%s\n' "${cache_directory_names[@]}" | LC_ALL=C sort)"
actual_cache_names="$(find "${cache_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
[[ "${actual_cache_names}" == "${expected_cache_names}" ]] || \
  fail "initial MIOpen cache directory set differs"
cache_uid="$(id -u)"
for name in "${cache_directory_names[@]}"; do
  path="${cache_root}/${name}"
  [[ -d "${path}" && ! -L "${path}" && "$(stat_mode "${path}")" == "700" ]] || \
    fail "MIOpen cache role identity differs: ${name}"
  [[ "$(stat -c '%u' "${path}")" == "${cache_uid}" ]] || \
    fail "MIOpen cache role owner differs: ${name}"
  [[ -z "$(find "${path}" -mindepth 1 -print -quit)" ]] || \
    fail "MIOpen cache role is not initially empty: ${name}"
done
[[ "$(stat_mode "${cache_root}")" == "700" ]] || fail "MIOpen cache root mode differs"
[[ "$(stat -c '%u' "${cache_root}")" == "${cache_uid}" ]] || \
  fail "MIOpen cache root owner differs"

[[ -d /tmp && ! -L /tmp && "$(realpath_existing /tmp)" == "/tmp" ]] || \
  fail "node-local tmp parent identity differs"
[[ "$(stat_mode /tmp)" == "1777" && "$(stat_uid /tmp)" == "0" ]] || \
  fail "node-local tmp parent mode/owner differs"
local_tmp_domain="bernini-oracle-regeneration-native-activation-v2-local-tmp-r9"
local_tmp_output_id="$(
  printf '%s\0%s\0%s\0%s\0%s' \
    "${local_tmp_domain}" "${SLURM_JOB_ID}" "${SLURM_STEP_ID}" \
    "${cache_uid}" "${output_dir}" | sha256sum | awk '{print $1}'
)"
[[ "${local_tmp_output_id}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "node-local tmp output domain digest differs"
local_tmp_root="/tmp/oracle-regeneration-native-v2-r9-u${cache_uid}-j${SLURM_JOB_ID}-s${SLURM_STEP_ID}-o${local_tmp_output_id}"
[[ ! -e "${local_tmp_root}" && ! -L "${local_tmp_root}" ]] || \
  fail "node-local tmp root must be fresh"
mkdir -m 700 -- "${local_tmp_root}" || fail "cannot create node-local tmp root"
for rank in 0 1 2 3; do
  mkdir -m 700 -- "${local_tmp_root}/rank-${rank}" || \
    fail "cannot create node-local rank tmp: ${rank}"
done
expected_local_tmp_names="$(printf '%s\n' rank-0 rank-1 rank-2 rank-3 | LC_ALL=C sort)"
actual_local_tmp_names="$(find "${local_tmp_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
[[ "${actual_local_tmp_names}" == "${expected_local_tmp_names}" ]] || \
  fail "initial node-local tmp directory set differs"
[[ "$(stat_mode "${local_tmp_root}")" == "700" && \
   "$(stat_uid "${local_tmp_root}")" == "${cache_uid}" ]] || \
  fail "node-local tmp root mode/owner differs"
[[ "$(stat_device "${local_tmp_root}")" == "$(stat_device /tmp)" && \
   "$(stat_device "${local_tmp_root}")" != "$(stat_device "${cache_root}")" ]] || \
  fail "node-local tmp filesystem separation differs"
for rank in 0 1 2 3; do
  path="${local_tmp_root}/rank-${rank}"
  [[ -d "${path}" && ! -L "${path}" && "$(stat_mode "${path}")" == "700" && \
     "$(stat_uid "${path}")" == "${cache_uid}" && \
     "$(stat_device "${path}")" == "$(stat_device "${local_tmp_root}")" ]] || \
    fail "node-local rank tmp identity differs: ${rank}"
  [[ -z "$(find "${path}" -mindepth 1 -print -quit)" ]] || \
    fail "node-local rank tmp is not initially empty: ${rank}"
done

# This digest is a frozen-launcher observation, not an independent signature.
# CPU preflight and every worker rebuild it from the same no-symlink identities
# before Torch and bind it into the step-local engineering receipt.
launcher_local_tmp_empty_proof_payload="$(
  printf '%s\n' 'bernini-launcher-node-local-tmp-fresh-empty-proof-r11'
  printf 'root\troot\t%s\t%s\t%s\t%s\t%s\t%s\troles-only\n' \
    "${local_tmp_root}" "$(stat_mode "${local_tmp_root}")" \
    "$(stat_uid "${local_tmp_root}")" "$(stat_gid "${local_tmp_root}")" \
    "$(stat_device "${local_tmp_root}")" "$(stat_inode "${local_tmp_root}")"
  for rank in 0 1 2 3; do
    name="rank-${rank}"
    path="${local_tmp_root}/${name}"
    printf 'role\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tempty\n' \
      "${name}" "${path}" "$(stat_mode "${path}")" \
      "$(stat_uid "${path}")" "$(stat_gid "${path}")" \
      "$(stat_device "${path}")" "$(stat_inode "${path}")"
  done
)"
export ORACLE_ACTIVATION_V2_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_SHA256="$(
  printf '%s' "${launcher_local_tmp_empty_proof_payload}" | sha256sum | awk '{print $1}'
)"
[[ "${ORACLE_ACTIVATION_V2_LAUNCHER_LOCAL_TMP_EMPTY_PROOF_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "launcher local-tmp empty proof digest differs"

export ORACLE_ACTIVATION_V2_OUTPUT_DIR="${output_dir}"
export ORACLE_ACTIVATION_V2_MIOPEN_CACHE_ROOT="${cache_root}"
export ORACLE_ACTIVATION_V2_MIOPEN_LOCAL_TMP_ROOT="${local_tmp_root}"
export MIOPEN_USER_DB_PATH="${cache_root}/launcher-bootstrap-user-db"
export MIOPEN_CUSTOM_CACHE_DIR="${cache_root}/launcher-bootstrap-kernel-cache"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1
export NATIVE_V_AXIS_LOAD_LOCK="${lock_file}"
unset PYTHONPATH HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export ROCR_VISIBLE_DEVICES="${visible_gpus}"

"${python_bin}" -I -B -c '
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
import infer_oracle_regeneration_native_activation_v2_r2 as runner
import oracle_regeneration_native_runtime_activation_v2 as runtime
import preflight_oracle_regeneration_native_activation_v2_r2 as preflight
import tools
import tools.materialize_vae as materialize_vae

expected = {
    "runner": (runner, root / "infer_oracle_regeneration_native_activation_v2_r2.py"),
    "runtime": (runtime, root / "oracle_regeneration_native_runtime_activation_v2.py"),
    "preflight": (preflight, root / "preflight_oracle_regeneration_native_activation_v2_r2.py"),
    "tools": (tools, root / "tools/__init__.py"),
    "materialize_vae": (materialize_vae, root / "tools/materialize_vae.py"),
    "raw_builder": (materialize_vae.raw_builder, root / "tools/build_renderer_dataset.py"),
}
for label, (module, path) in expected.items():
    if pathlib.Path(module.__file__).resolve(strict=True) != path:
        raise SystemExit(label + " import origin differs")
if (
    [pathlib.Path(item).resolve(strict=True) for item in tools.__path__]
    != [root / "tools"]
    or materialize_vae.raw_builder is not sys.modules.get("tools.build_renderer_dataset")
    or "torch" in sys.modules
):
    raise SystemExit("isolated import graph differs")
' "${method_root}" || fail "isolated import closure probe failed"

env PYTHONPATH="${method_root}" "${python_bin}" -B "${static_test}" || \
  fail "runtime static/unit test failed"
env PYTHONPATH="${method_root}" "${python_bin}" -O -B "${static_test}" || \
  fail "runtime optimized static/unit test failed"
env PYTHONPATH="${method_root}" "${python_bin}" -B "${preflight}" \
  --authority-packet "${packet}" \
  --external-ledger "${ledger}" \
  --require-ready || fail "CPU release preflight failed"

exec env PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="${master_port}" \
  "${runner}" \
  --authority-packet "${packet}" \
  --external-ledger "${ledger}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --output-dir "${output_dir}" \
  --miopen-cache-root "${cache_root}" \
  --miopen-local-tmp-root "${local_tmp_root}" \
  --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793 \
  --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d \
  --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca

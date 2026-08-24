#!/usr/bin/env bash
# WORLD4 authoring launcher r2 for one activation-v2 VAE or prompt receipt.
#
# This script does not run the regeneration canary.  It materializes factual,
# source-only diagnostic provenance inside an existing one-node allocation.
# The output remains non-authority until a distinct AI-agent review, exact
# packet/ledger publication, and a later compiled activation revision.

set -Eeuo pipefail

fail() {
  echo "[oracle-activation-v2-materializer] ERROR: $*" >&2
  exit 2
}

repo_root="${ORACLE_ACTIVATION_V2_REPO_ROOT:?set repository root}"
python_bin="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini-R checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint content manifest}"
output_dir="${ORACLE_MATERIALIZER_OUTPUT_DIR:?set fresh materializer output dir}"
materializer_kind="${ORACLE_MATERIALIZER_KIND:?set vae or prompt}"
case_id="${ORACLE_CASE_ID:?set e02 or e03}"
visible_gpus="${ORACLE_VISIBLE_GPUS:?set four comma-separated local GPU indices}"
master_port="${ORACLE_MASTER_PORT:?set private torchrun port}"

for value in \
  "${repo_root}" "${python_bin}" "${bernini_root}" "${veomni_root}" \
  "${checkpoint}" "${checkpoint_manifest}" "${output_dir}"
do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
[[ "${materializer_kind}" == vae || "${materializer_kind}" == prompt ]] || \
  fail "materializer kind differs"
[[ "${case_id}" == e02 || "${case_id}" == e03 ]] || fail "case differs"
[[ "${visible_gpus}" == "0,1,2,3" ]] || fail "visible GPU list differs"
[[ "${master_port}" =~ ^[1-9][0-9]{3,4}$ ]] || fail "master port differs"
(( 10#${master_port} <= 65535 )) || fail "master port exceeds 65535"
[[ "${SLURM_JOB_ID:-}" == "140846" ]] || fail "launcher requires allocation 140846"
node_name="$(hostname -s)"
case "${node_name}" in
  auh7-1b-gpu-246|auh7-1b-gpu-247|auh7-1b-gpu-248|auh7-1b-gpu-279) ;;
  *) fail "node is outside allocation 140846 allowlist" ;;
esac

repo_root="$(realpath -e -- "${repo_root}")"
python_bin="$(realpath -e -- "${python_bin}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
[[ -d "${repo_root}" && ! -L "${repo_root}" ]] || fail "repository root differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || \
  fail "Python executable differs"
[[ "${python_bin}" == "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12" ]] || \
  fail "Python executable path differs"
[[ "$(sha256sum "${python_bin}" | awk '{print $1}')" == \
  8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a ]] || \
  fail "Python executable bytes differ"
[[ "$(stat -c '%s' "${python_bin}")" == "31490256" ]] || \
  fail "Python executable size differs"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || \
  fail "checkpoint manifest differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == \
  a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831 ]] || \
  fail "checkpoint content manifest bytes differ"

method_root="${repo_root}/methods/bernini_action_editing"
core="${method_root}/oracle_regeneration_activation_v2.py"
vae_tool="${method_root}/tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py"
prompt_tool="${method_root}/tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py"
static_test="${method_root}/tests/test_oracle_regeneration_activation_v2_materializers_r2.py"

declare -A expected_sha256=(
  ["${core}"]="ef97259dd181ff065267e32f1e5cca158e26ad5174457780163658a3db728bb0"
  ["${vae_tool}"]="07b40cdd67771d257ce546ca4166301980c6768269acc5f097fc08973656bbde"
  ["${prompt_tool}"]="5bba9f977fa40e5044053baaaf73eba779b3816ef6137457a06ceac82a3463af"
  ["${static_test}"]="e8bfc32e4c5fb13b8ba4e6417c61ef1c7c6e7e204cec7a348f0d99bbd23da859"
  ["${method_root}/oracle_regeneration_canary_v1.py"]="0148b137c200e426ff18571f71d373a9e6ef595c620664925dae0ab9d1d91081"
  ["${method_root}/native_branch_homotopy_runtime_v1.py"]="b81ee152e358e4d5a6638dfccf1232c4e221311ffb38937e61be3c6a799b84d5"
  ["${method_root}/native_branch_homotopy_v1.py"]="2585416e61935db62cc7534daf19b4bb851f9fdcdeb92f78e6152f55e034f3d0"
  ["${method_root}/self_guided_action_field_v1.py"]="2ad204c09f5eb60865017b1e596de25b777d8d6ed43774f4dcbc23a4ad58bc7e"
  ["${method_root}/tri_branch_unipc.py"]="58d2e0e8d56a500eea07ec20f0fb101539ac846bbd039c0d50a22506b58fb3d2"
  ["${method_root}/infer_native_identity_generation_canary.py"]="bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42"
  ["${method_root}/infer_native_branch_homotopy_canary.py"]="d6dab735ce52da151848c96f9e00775994dc281ace20afa6dcb9fb64709e5983"
  ["${method_root}/infer_source_kv_carrier_oracle.py"]="fcf77576735c89e685415b94b2dc0f0c5b8d1dd8dc1c55832538ff0daafb4604"
  ["${method_root}/infer_source_value_residual_oracle.py"]="40e581db7906f20103a16ad47fda76978cbad21c9277723f3e8e022d717ed2d8"
  ["${method_root}/infer_native_self_guided_action_field_canary.py"]="ad591fe5bd5943fab59603400fcf70a126f3461a39670f93a78aa24b1902d313"
  ["${method_root}/infer_lora.py"]="acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
  ["${method_root}/tools/materialize_vae.py"]="a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
  ["${method_root}/tools/build_renderer_dataset.py"]="afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
  ["${method_root}/tools/__init__.py"]="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  ["${method_root}/train_lora.py"]="8e8daf422548bc29e2c18f2d2c692af2dd3109aaad1897fc31e590a69d7e593e"
  ["${method_root}/action_preservation_decoded_eval_model_authority_v2.py"]="760ed9988147a44965fd47f68a08fd353ce1d900e661b55bb818088ec9ef848e"
  ["${method_root}/source_kv_replay.py"]="45b43426dc7825dbd61280154fc35161c60476ec5cb9e53bc0225f3809c759f3"
  ["${method_root}/source_kv_route_batches.py"]="7f3ae0d27747ad58b3b195c712884641012eb836bb59963896c58518b8b5731e"
  ["${method_root}/source_value_residual.py"]="420cadf3cb2824b2bf5a809c55086d81351db19f31743b0b77a957adf219e124"
)
for path in "${!expected_sha256[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "closure file differs: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected_sha256[${path}]}" ]] || \
    fail "closure bytes differ: ${path}"
done

case "${case_id}" in
  e02)
    source_iid="10ed90644f81461d"
    source_video="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/10ed90644f81461d/source.mp4"
    source_sha256="63dd620627e18d0f6836058bf725ffba7ab1f9b0b455e784c6a71dc68a8de46c"
    action_caption="The same pale bare hand firmly grips the same red mushroom at its lower stem, twists and pulls it free from the same soil, lifts the same intact mushroom above the newly empty root hole, and holds it there. Exactly one hand and one mushroom remain visible; do not duplicate, split, or fuse either one."
    ;;
  e03)
    source_iid="7a33b36459c84289"
    source_video="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/7a33b36459c84289/source.mp4"
    source_sha256="c1455b9b89d1f352da69e7bb07e955ee4495df94f5ef6f3f09fe7fd9eac035bb"
    action_caption="The same farm worker moves the same harvested root cluster over the same woven basket, lowers it past the rim, opens the same hand, and releases it. The cluster falls, bounces slightly, and settles inside while the now-empty hand withdraws. Do not duplicate the hand or cluster."
    ;;
esac
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "source video differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${source_sha256}" ]] || \
  fail "source video bytes differ"

[[ "${output_dir}" != / && ! -e "${output_dir}" && ! -L "${output_dir}" ]] || \
  fail "output directory must be fresh"
output_parent="$(realpath -e -- "$(dirname -- "${output_dir}")")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
[[ "${output_dir}" == "${output_parent}/${output_dir##*/}" ]] || \
  fail "output path is not canonical"
[[ "${output_dir##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
  fail "unsafe output basename"

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
unset PYTHONPATH HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export ROCR_VISIBLE_DEVICES="${visible_gpus}"

"${python_bin}" -I -B -c '
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
import tools
import tools.materialize_vae as materialize_vae
import tools.materialize_oracle_regeneration_vae_refs_activation_v2_r2 as vae_r2
import tools.materialize_oracle_regeneration_prompts_activation_v2_r2 as prompt_r2

expected = {
    "native_branch_homotopy_runtime_v1.py",
    "native_branch_homotopy_v1.py",
    "oracle_regeneration_activation_v2.py",
    "oracle_regeneration_canary_v1.py",
    "self_guided_action_field_v1.py",
    "tools/__init__.py",
    "tools/build_renderer_dataset.py",
    "tools/materialize_vae.py",
    "tools/materialize_oracle_regeneration_vae_refs_activation_v2_r2.py",
    "tools/materialize_oracle_regeneration_prompts_activation_v2_r2.py",
    "tri_branch_unipc.py",
}
loaded = set()
for module in tuple(sys.modules.values()):
    source = getattr(module, "__file__", None)
    if not source:
        continue
    try:
        loaded.add(str(pathlib.Path(source).resolve(strict=True).relative_to(root)))
    except ValueError:
        pass
if loaded != expected:
    raise SystemExit("isolated local import set differs: " + repr(sorted(loaded)))
tools_root = root / "tools"
if (
    pathlib.Path(tools.__file__).resolve(strict=True) != tools_root / "__init__.py"
    or [pathlib.Path(item).resolve(strict=True) for item in tools.__path__]
    != [tools_root]
    or pathlib.Path(materialize_vae.__file__).resolve(strict=True)
    != tools_root / "materialize_vae.py"
    or pathlib.Path(materialize_vae.raw_builder.__file__).resolve(strict=True)
    != tools_root / "build_renderer_dataset.py"
    or materialize_vae.raw_builder is not sys.modules.get("tools.build_renderer_dataset")
    or pathlib.Path(vae_r2.__file__).resolve(strict=True) != pathlib.Path(sys.argv[2])
    or pathlib.Path(prompt_r2.__file__).resolve(strict=True) != pathlib.Path(sys.argv[3])
    or "_omnivideo2_strict_action_preview_materializer" in sys.modules
):
    raise SystemExit("isolated materializer import origin differs")
' "${method_root}" "${vae_tool}" "${prompt_tool}" || fail "isolated import closure probe failed"

PYTHONPATH="${method_root}" "${python_bin}" -B "${static_test}"
PYTHONPATH="${method_root}" "${python_bin}" -O -B "${static_test}"

common_args=(
  --case-id "${case_id}"
  --source-iid "${source_iid}"
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --output-dir "${output_dir}"
  --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793
  --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d
)
if [[ "${materializer_kind}" == vae ]]; then
  runtime="${vae_tool}"
  common_args+=(--source-video "${source_video}")
else
  runtime="${prompt_tool}"
  common_args+=(--action-caption "${action_caption}")
fi

exec env PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="${master_port}" \
  "${runtime}" \
  "${common_args[@]}"

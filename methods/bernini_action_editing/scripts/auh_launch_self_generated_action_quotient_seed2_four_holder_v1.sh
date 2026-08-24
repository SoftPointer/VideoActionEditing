#!/usr/bin/env bash
# Full independent seed-20260817 replication of the frozen eight-arm action
# quotient screen.  Four retained holders each run two disjoint WORLD4/SP4
# islands.  This controller never cancels, releases, requeues, or retries a
# parent allocation.

set -Eeuo pipefail
umask 077

fail() { echo "[action-quotient-seed2] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly confirmation="${AQ2_CONFIRM:?set explicit seed2 confirmation}"
readonly release_root="${AQ2_RELEASE_ROOT:?set immutable release root}"
readonly experiment_root="${AQ2_EXPERIMENT_ROOT:?set fresh experiment root}"
readonly archive_sha="${AQ2_ARCHIVE_SHA256:?pin source archive SHA-256}"
readonly release_manifest_sha="${AQ2_RELEASE_MANIFEST_SHA256:?pin release manifest SHA-256}"
readonly controller_sha="${AQ2_CONTROLLER_SHA256:?pin detached controller SHA-256}"
readonly envelope_sha="${AQ2_DEPLOYMENT_ENVELOPE_SHA256:?pin deployment envelope SHA-256}"
readonly source_revision="${AQ2_SOURCE_REVISION:?pin source revision}"

readonly expected_confirmation=launch-approved-action-quotient-seed20260817-four-holder-r1
readonly replication_seed=20260817
readonly source_data_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_quotient_job140846_v4/source_only/manifest.json
readonly source_data_manifest_sha=62fee73b3d84015f2e72edcd4da14b51f7695980a4ba892420ca137aa50e9ad8
readonly source_data_manifest_digest=2fb367ed6f06275705e0b71020dd87fd68e13a010e80ef0bd2a122c94070f503
readonly forbidden_seed1_cache_sha=d96253fba3dac1b9602cc55bc71704f386c5ad17d4078992231df05da9b64a41
readonly forbidden_seed1_initial_digest=21505088817f2571fb49ba257d4d538055e87e6a5318fd6b4411eef9d76c8e0c
readonly expected_node_runner_sha=d2861c22c6758879cf842a57479eb82a2e6372a4f35a51ab1cc2d491fa6cb85f
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly archive="${release_root}/source.tar"
readonly release_manifest="${release_root}/source.manifest.json"
readonly controller="${release_root}/auh_launch_self_generated_action_quotient_seed2_four_holder_v1.sh"
readonly envelope="${release_root}/deployment-envelope.json"
readonly materialized="${experiment_root}/materialized"
readonly node_runner="${materialized}/methods/bernini_action_editing/scripts/auh_run_self_generated_action_quotient_v1.sh"
readonly cache="${experiment_root}/teacher-cache-seed20260817-row4-slot4.pt"

[[ "${confirmation}" == "${expected_confirmation}" ]] || fail "confirmation differs"
[[ "${release_root}" == /vast/users/guangyi.chen/* && "${experiment_root}" == /vast/users/guangyi.chen/* ]] || fail "root path differs"
[[ "${release_root}" != / && "${experiment_root}" != / && "${release_root}" != "${experiment_root}" ]] || fail "root topology differs"
[[ "${archive_sha}${release_manifest_sha}${controller_sha}${envelope_sha}" =~ ^[0-9a-f]{256}$ ]] || fail "release SHA pin differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ ! -e "${experiment_root}" && ! -L "${experiment_root}" ]] || fail "experiment root is not fresh"
[[ "$(realpath -m -- "${experiment_root}")" == "${experiment_root}" ]] || fail "experiment root is not canonical"
[[ -d "${release_root}" && ! -L "${release_root}" && "$(readlink -f -- "${release_root}")" == "${release_root}" ]] || fail "release root differs"
[[ "$(readlink -f -- "$0")" == "${controller}" ]] || fail "executing controller is not the sealed release copy"

readonly expected_release_entries=$'auh_launch_self_generated_action_quotient_seed2_four_holder_v1.sh\ndeployment-envelope.json\nsource.manifest.json\nsource.tar'
observed_release_entries="$(find "${release_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
[[ "${observed_release_entries}" == "${expected_release_entries}" ]] || fail "release entry closure differs"
for pair in \
  "${archive}:${archive_sha}:444" \
  "${release_manifest}:${release_manifest_sha}:444" \
  "${controller}:${controller_sha}:555" \
  "${envelope}:${envelope_sha}:444"; do
  path="${pair%%:*}"; rest="${pair#*:}"; digest="${rest%%:*}"; mode="${rest##*:}"
  [[ -f "${path}" && ! -L "${path}" && "$(stat -c '%h|%a|%u' "${path}")" == "1|${mode}|2012" ]] || fail "release file topology differs: ${path}"
  [[ "$(sha256_file "${path}")" == "${digest}" ]] || fail "release file SHA differs: ${path}"
done
[[ -f "${source_data_manifest}" && ! -L "${source_data_manifest}" ]] || fail "source data manifest differs"
[[ "$(sha256_file "${source_data_manifest}")" == "${source_data_manifest_sha}" ]] || fail "source data manifest SHA differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python executable differs"

jobs=(136719 136141 136309 136140)
nodes=(auh7-1b-gpu-306 auh7-1b-gpu-299 auh7-1b-gpu-280 auh7-1b-gpu-215)
holder_preflight() {
  local job="$1" node="$2" observed_steps snapshot
  [[ "$(squeue -j "${job}" -h -o '%T|%N|%u|%C|%m|%b')" == \
    "RUNNING|${node}|guangyi.chen|64|64G|gres/gpu:mi210:8" ]] || fail "holder state/resources differ: ${job}"
  observed_steps="$(squeue -s -j "${job}" -h -o '%i' | LC_ALL=C sort)"
  [[ "${observed_steps}" == "${job}.batch"$'\n'"${job}.extern" ]] || fail "holder has a numbered step: ${job}"
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -- "${node}" \
    '/usr/bin/rocm-smi --showuse --showmemuse --showpids --json')" || fail "GPU probe failed: ${node}"
  printf '%s' "${snapshot}" | "${python_bin}" -I -B -c '
import json,sys
x=json.load(sys.stdin); assert set(x)=={f"card{i}" for i in range(8)}|{"system"}
for i in range(8):
 c=x[f"card{i}"]; assert c["GPU use (%)"]=="0" and c["GPU Memory Allocated (VRAM%)"]=="0"
for key,value in x["system"].items():
 assert key.startswith("PID") and key[3:].isdigit()
 p=[v.strip() for v in value.split(",")]
 assert p[0]=="gpuagent" and p[1:]==["0","0","0","0"]
' || fail "GPU/KFD is not idle: ${node}"
}
for index in 0 1 2 3; do
  holder_preflight "${jobs[$index]}" "${nodes[$index]}"
done

mkdir -m 700 "${experiment_root}"
mkdir -m 700 "${experiment_root}/logs" "${experiment_root}/runs" "${materialized}"
tar -xf "${archive}" -C "${materialized}"
[[ -f "${node_runner}" && ! -L "${node_runner}" && -x "${node_runner}" ]] || fail "materialized node runner differs"
[[ "$(sha256_file "${node_runner}")" == "${expected_node_runner_sha}" ]] || fail "materialized node runner SHA differs"

common_env=(
  ACTION_QUOTIENT_SOURCE_ARCHIVE="${archive}"
  ACTION_QUOTIENT_SOURCE_ARCHIVE_SHA256="${archive_sha}"
  ACTION_QUOTIENT_SOURCE_REVISION="${source_revision}"
  ACTION_QUOTIENT_SOURCE_MANIFEST="${source_data_manifest}"
  ACTION_QUOTIENT_SOURCE_MANIFEST_SHA256="${source_data_manifest_sha}"
  ACTION_QUOTIENT_SEED="${replication_seed}"
  ACTION_QUOTIENT_SLOTS=4
)

run_cache_step() {
  local label="$1" limit="$2" output="$3"
  srun --jobid=136719 --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-306 \
    --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 --immediate=5 --kill-on-bad-exit=1 \
    --job-name="aq2-${label}" \
    env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
    "${common_env[@]}" ACTION_QUOTIENT_MODE=cache ACTION_QUOTIENT_LIMIT_CELLS="${limit}" \
    ACTION_QUOTIENT_CACHE="${experiment_root}/unused-${label}.pt" ACTION_QUOTIENT_OUTPUT="${output}" \
    "${node_runner}" >"${experiment_root}/logs/${label}.log" 2>&1
}

run_cache_step cache-canary 1 "${experiment_root}/teacher-cache-canary-seed20260817.pt"
[[ -f "${experiment_root}/teacher-cache-canary-seed20260817.pt" ]] || fail "one-cell cache canary is absent"
run_cache_step cache-full 0 "${cache}"
[[ -f "${cache}" && ! -L "${cache}" ]] || fail "full teacher cache is absent"
readonly cache_sha="$(sha256_file "${cache}")"
[[ "${cache_sha}" =~ ^[0-9a-f]{64}$ && "${cache_sha}" != "${forbidden_seed1_cache_sha}" ]] || fail "seed2 cache identity differs"

"${python_bin}" -I -B -c '
import hashlib, json, pathlib, sys, torch
canary_path=pathlib.Path(sys.argv[1]); full_path=pathlib.Path(sys.argv[2]); expected=sys.argv[3]
raw=full_path.read_bytes(); assert hashlib.sha256(raw).hexdigest()==expected
canary=torch.load(canary_path,map_location="cpu",weights_only=False)
x=torch.load(full_path,map_location="cpu",weights_only=False)
keys={"schema_version","manifest_digest","source_manifest_sha256","method_source_revision","method_source_archive_sha256","slots","seed","initialization_seed","teacher_cache_seed","cells","teacher_graph","anchor_role"}
assert set(x)==set(canary)==keys
assert x["manifest_digest"]==sys.argv[4] and x["source_manifest_sha256"]==sys.argv[5]
assert x["method_source_revision"]==sys.argv[6] and x["method_source_archive_sha256"]==sys.argv[7]
assert x["slots"]==4 and x["seed"]==x["initialization_seed"]==x["teacher_cache_seed"]==20260817
assert len(canary["cells"])==1 and len(x["cells"])==16
assert {(int(c["row_index"]),int(c["slot"])) for c in x["cells"]}=={(r,s) for r in range(4) for s in range(4)}
def same(a,b):
    if type(a) is not type(b): return False
    if isinstance(a,torch.Tensor): return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and torch.equal(a,b)
    if isinstance(a,dict): return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(same(i,j) for i,j in zip(a,b))
    return a==b
assert all(same(canary[k],x[k]) for k in keys-{"cells"})
assert same(canary["cells"][0],x["cells"][0])
' "${experiment_root}/teacher-cache-canary-seed20260817.pt" "${cache}" "${cache_sha}" \
  "${source_data_manifest_digest}" "${source_data_manifest_sha}" "${source_revision}" "${archive_sha}" || \
  fail "seed2 teacher cache validation failed"
chmod 0444 "${cache}" "${cache}.receipt.json" \
  "${experiment_root}/teacher-cache-canary-seed20260817.pt" \
  "${experiment_root}/teacher-cache-canary-seed20260817.pt.receipt.json"
for retained_cache_file in \
  "${cache}" "${cache}.receipt.json" \
  "${experiment_root}/teacher-cache-canary-seed20260817.pt" \
  "${experiment_root}/teacher-cache-canary-seed20260817.pt.receipt.json"; do
  [[ "$(stat -c '%h|%a|%u' "${retained_cache_file}")" == "1|444|2012" ]] || fail "retained cache topology differs"
done

# The cache step can take several minutes.  Re-establish that no other
# numbered work entered any retained holder before the eight formal arms.
for index in 0 1 2 3; do
  holder_preflight "${jobs[$index]}" "${nodes[$index]}"
done

arms=(
  action_only action_only_lowlr
  action_noop action_start
  action_nuisance action_start_nuisance
  action_start_nuisance_noop action_start_nuisance_border
)
arm_jobs=(136719 136719 136141 136141 136309 136309 136140 136140)
arm_nodes=(
  auh7-1b-gpu-306 auh7-1b-gpu-306
  auh7-1b-gpu-299 auh7-1b-gpu-299
  auh7-1b-gpu-280 auh7-1b-gpu-280
  auh7-1b-gpu-215 auh7-1b-gpu-215
)
arm_groups=(0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7)
pids=()
for index in 0 1 2 3 4 5 6 7; do
  arm="${arms[$index]}"; job="${arm_jobs[$index]}"; node="${arm_nodes[$index]}"; group="${arm_groups[$index]}"
  output="${experiment_root}/runs/${arm}"
  [[ ! -e "${output}" && ! -L "${output}" ]] || fail "arm output is not fresh: ${arm}"
  srun --jobid="${job}" --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
    --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 --immediate=5 --kill-on-bad-exit=1 \
    --job-name="aq2-${arm}" \
    env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES="${group}" \
    "${common_env[@]}" ACTION_QUOTIENT_MODE=train ACTION_QUOTIENT_LIMIT_CELLS=0 \
    ACTION_QUOTIENT_CACHE="${cache}" ACTION_QUOTIENT_EXPECTED_CACHE_SHA256="${cache_sha}" \
    ACTION_QUOTIENT_OUTPUT="${output}" ACTION_QUOTIENT_ARM="${arm}" ACTION_QUOTIENT_MAX_STEPS=160 \
    "${node_runner}" >"${experiment_root}/logs/train-${arm}.log" 2>&1 &
  pids+=("$!")
  printf 'LAUNCHED arm=%s job=%s node=%s gpus=%s pid=%s\n' "${arm}" "${job}" "${node}" "${group}" "${pids[-1]}"
done

status=0
for index in 0 1 2 3 4 5 6 7; do
  if wait "${pids[$index]}"; then
    printf 'COMPLETE arm=%s\n' "${arms[$index]}"
  else
    printf 'FAILED arm=%s\n' "${arms[$index]}" >&2
    status=1
  fi
done
(( status == 0 )) || fail "one or more formal arms failed; no retry was attempted"

"${python_bin}" -I -B -c '
import hashlib, json, math, pathlib, sys
root=pathlib.Path(sys.argv[1]); cache_sha=sys.argv[2]; old_init=sys.argv[3]
archive_sha=sys.argv[4]; revision=sys.argv[5]; manifest_sha=sys.argv[6]; arms=sys.argv[7:]
digests=[]
for arm in arms:
    arm_root=root/"runs"/arm
    expected={f"checkpoint-{step:08d}" for step in (10,20,40,80,160)}
    assert {p.name for p in arm_root.iterdir()}==expected
    arm_digest=None
    for step in (10,20,40,80,160):
        p=arm_root/f"checkpoint-{step:08d}"/"receipt.json"
        x=json.loads(p.read_text())
        declared=x.pop("receipt_digest")
        raw=json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        assert hashlib.sha256(raw).hexdigest()==declared
        assert x["global_step"]==step and x["max_steps"]==160
        assert x["initialization_seed"]==x["teacher_cache_seed"]==20260817
        assert x["teacher_cache_sha256"]==cache_sha
        assert x["method_source_archive_sha256"]==archive_sha and x["method_source_revision"]==revision
        assert x["source_manifest_sha256"]==manifest_sha
        assert x["training_contract"]["arm"]==arm
        assert x["experimental_training"] is True and x["scientific_claim_authorized"] is False
        assert math.isfinite(x["last_loss"]) and math.isfinite(x["last_preclip_gradient_norm"])
        current=x["distributed"]["lora_initialization_digest"]
        assert arm_digest in (None,current); arm_digest=current
    digests.append(arm_digest)
assert len(set(digests))==1 and digests[0]!=old_init
' "${experiment_root}" "${cache_sha}" "${forbidden_seed1_initial_digest}" \
  "${archive_sha}" "${source_revision}" "${source_data_manifest_sha}" "${arms[@]}" || \
  fail "formal checkpoint receipt validation failed"

printf 'seed2_training_complete=true\nseed=%s\ncache_sha256=%s\nparent_allocations_cancelled=false\nautomatic_retry=false\n' \
  "${replication_seed}" "${cache_sha}" >"${experiment_root}/TRAINING_COMPLETE"
echo "ALL_SEED2_TRAINING_COMPLETE parent_allocations_cancelled=false automatic_retry=false"

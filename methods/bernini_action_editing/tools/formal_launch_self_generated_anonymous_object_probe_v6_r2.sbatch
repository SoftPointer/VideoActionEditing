#!/usr/bin/env bash
#SBATCH --job-name=anonymous-object-v6-r2
#SBATCH --account=faculty-acc
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:mi210:4
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/self_generated_anonymous_object_probe_v6_20260823/formal-v6-r2-slurm-%j.log

set -euo pipefail
umask 077

readonly BASE="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/self_generated_anonymous_object_probe_v6_20260823"
readonly SNAPSHOT="${BASE}/snapshot_83aa45bfe4bd_r2"
readonly CONTROLLER_ROOT="${BASE}/completion_controller_604c600d246b_r2"
readonly CONTROLLER="${CONTROLLER_ROOT}/methods/bernini_action_editing/tools/complete_self_generated_anonymous_object_probe_v6.py"
readonly CONTROLLER_SHA256="604c600d246b0b01dd1c4fad3250ec2d580d71b4d1e4001c6f79490cab1bf339"
readonly CONTROLLER_TEST="${CONTROLLER_ROOT}/methods/bernini_action_editing/tests/test_anonymous_object_v6_completion_controller.py"
readonly CONTROLLER_TEST_SHA256="c1c8bb2bdce6ce33e23cb4326cfed067519f0ad5b65bc79439db4151e0e0f268"
readonly SNAPSHOT_SUMS_SHA256="349b793e436ad08219775348770b363ed2a74e96606d506a70dfd3c75d0234e7"
readonly PYTHON="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
readonly ENTRYPOINT="${SNAPSHOT}/methods/bernini_action_editing/auh_self_generated_anonymous_object_same_state_probe_v6.py"
readonly RANK_WRAPPER="${SNAPSHOT}/methods/bernini_action_editing/scripts/auh_self_generated_anonymous_object_same_state_probe_rank_wrapper_v6.sh"
readonly RUN_ROOT="${BASE}/real_anonymous_object_same_state_probe_v6_r2"
readonly CACHE_ROOT="${RUN_ROOT}/miopen-cache"
readonly CANDIDATE="${RUN_ROOT}/candidate.json"
readonly COMPLETION_SEAL="${RUN_ROOT}/completion.seal.json"
readonly LAUNCHER_LOG="${RUN_ROOT}/srun.log"
readonly STEP_ID_FILE="${RUN_ROOT}/slurm-step-id.txt"

die() {
  echo "formal V6 launch rejected: $*" >&2
  exit 70
}

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || die "missing numeric Slurm job allocation"
[[ "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || die "allocation must contain exactly one node"
[[ "${SLURM_CPUS_PER_TASK:-0}" == "32" ]] || die "allocation must reserve 32 CPUs per task"
[[ "${SLURM_MEM_PER_NODE:-0}" =~ ^[0-9]+$ ]] || die "allocation memory authority is absent"
(( SLURM_MEM_PER_NODE >= 262144 )) || die "allocation has less than 256 GiB host memory"
[[ -f "$CONTROLLER" && ! -L "$CONTROLLER" ]] || die "controller is not a plain file"
[[ -f "$CONTROLLER_TEST" && ! -L "$CONTROLLER_TEST" ]] || die "controller test is not a plain file"
[[ "$(sha256sum "$CONTROLLER" | awk '{print $1}')" == "$CONTROLLER_SHA256" ]] || die "controller SHA differs"
[[ "$(sha256sum "$CONTROLLER_TEST" | awk '{print $1}')" == "$CONTROLLER_TEST_SHA256" ]] || die "controller test SHA differs"
[[ "$(stat -c '%a' "$CONTROLLER")" == "444" ]] || die "controller mode differs"
[[ "$(stat -c '%a' "$CONTROLLER_TEST")" == "444" ]] || die "controller test mode differs"
if find "$CONTROLLER_ROOT" -perm /022 -print -quit | grep -q .; then
  die "external controller authority has a group/world-writable entry"
fi
[[ "$(sha256sum "${SNAPSHOT}/SHA256SUMS" | awk '{print $1}')" == "$SNAPSHOT_SUMS_SHA256" ]] || die "snapshot SHA256SUMS authority differs"
(
  cd "$SNAPSHOT"
  sha256sum --quiet -c SHA256SUMS
) || die "snapshot byte closure differs"
if find "$SNAPSHOT" -perm /022 -print -quit | grep -q .; then
  die "snapshot has a group/world-writable entry"
fi
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "fixed run root is not fresh"
mkdir "$RUN_ROOT"
chmod 0700 "$RUN_ROOT"
mkdir "$CACHE_ROOT"
chmod 0700 "$CACHE_ROOT"
for path in "$CANDIDATE" "$COMPLETION_SEAL" "$LAUNCHER_LOG" "$STEP_ID_FILE"; do
  [[ ! -e "$path" && ! -L "$path" ]] || die "fresh output path check failed: $path"
done

readonly STEP_EXEC='set -euo pipefail
umask 077
[[ ! -e "$V6_STEP_ID_FILE" && ! -L "$V6_STEP_ID_FILE" ]]
( set -o noclobber; printf "%s\n" "${SLURM_STEP_ID:?}" > "$V6_STEP_ID_FILE" )
exec "$@"'

srun_command=(
  srun
  --exclusive
  --exact
  --nodes=1
  --ntasks=1
  --cpus-per-task=32
  --gres=gpu:mi210:4
  --mem=240G
  --kill-on-bad-exit=1
  env
  MODELING_BACKEND=hf
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  PYTHONDONTWRITEBYTECODE=1
  PYTHONNOUSERSITE=1
  MIOPEN_CACHE_ROOT="$CACHE_ROOT"
  V6_STEP_ID_FILE="$STEP_ID_FILE"
  /bin/bash
  -c
  "$STEP_EXEC"
  v6-step-exec
  "$PYTHON"
  -B
  -m
  torch.distributed.run
  --standalone
  --nproc_per_node=4
  --no-python
  "$RANK_WRAPPER"
  "$CANDIDATE"
  "$ENTRYPOINT"
)
readonly SRUN_COMMAND_SHA256="$(
  printf '%s\0' "${srun_command[@]}" | sha256sum | awk '{print $1}'
)"

set +e
"${srun_command[@]}" 2>&1 | tee "$LAUNCHER_LOG"
pipeline_status=("${PIPESTATUS[@]}")
set -e
[[ "${#pipeline_status[@]}" -eq 2 ]] || die "srun/log pipeline status is incomplete"
readonly SRUN_RC="${pipeline_status[0]}"
readonly TEE_RC="${pipeline_status[1]}"
[[ "$TEE_RC" -eq 0 ]] || die "launcher log write failed with rc=${TEE_RC}"
if [[ "$SRUN_RC" -ne 0 ]]; then
  echo "formal V6 srun failed with rc=${SRUN_RC}; no completion controller invoked" >&2
  exit "$SRUN_RC"
fi

mapfile -t step_rows < "$STEP_ID_FILE"
[[ "${#step_rows[@]}" -eq 1 && "${step_rows[0]}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "captured Slurm step identity differs"
readonly CAPTURED_STEP_ID="${step_rows[0]}"

"$PYTHON" -B "$CONTROLLER" \
  --candidate "$CANDIDATE" \
  --completion-seal "$COMPLETION_SEAL" \
  --launcher-log "$LAUNCHER_LOG" \
  --srun-exit-code "$SRUN_RC" \
  --caller-attests-all-rank-wrappers-exit-zero \
  --slurm-job-id "$SLURM_JOB_ID" \
  --slurm-step-id "$CAPTURED_STEP_ID" \
  --srun-command-sha256 "$SRUN_COMMAND_SHA256"

[[ -f "$COMPLETION_SEAL" && ! -L "$COMPLETION_SEAL" ]] || die "completion seal was not created"
echo "formal V6 candidate: $CANDIDATE"
echo "formal V6 completion seal: $COMPLETION_SEAL"

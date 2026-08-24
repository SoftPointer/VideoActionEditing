#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RELEASE PYTHON" >&2
  exit 2
fi

v4b_release=$1
v4b_python=$2
v4b_runtime=$v4b_release/methods/bernini_action_editing/semantic_anchor_temporal_convae_v4b_fast.py
v4b_test=$v4b_release/methods/bernini_action_editing/tests/test_semantic_anchor_temporal_convae_v4b_fast.py
v4a_runtime=$v4b_release/methods/bernini_action_editing/semantic_anchor_linear_frontier_v4_fast.py
v2_runtime=$v4b_release/methods/bernini_action_editing/semantic_anchor_action_sequence_vae_v2.py
feature_runtime=$v4b_release/methods/bernini_action_editing/semantic_action_cvae_canary_v1.py

check_file() {
  local path=$1
  local expected=$2
  [[ -f $path ]]
  [[ $(stat -c '%a:%h' "$path") == 444:1 ]]
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]]
}

check_release() {
  check_file "$v4b_runtime" 6afa9fc39f993cedcb7ef672ca1297412ab95f5fdacbaf33a431fb49ef586ac4
  check_file "$v4b_test" 494c980fbdf8a86ab4c6e57a3c57371d3daf82fb15073478254ea2bf9e7e382d
  check_file "$v4a_runtime" e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973
  check_file "$v2_runtime" 46927772a1861354ad5edeb2072ae9b1b505d235de7c2615fb11a6648f2bddca
  check_file "$feature_runtime" 74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233
}

check_release

v4b_cache=/tmp/anchor-v4b-preflight-${SLURM_JOB_ID:?}-${SLURM_STEP_ID:?}
[[ ! -e $v4b_cache ]]
mkdir -m 0700 "$v4b_cache"
export TMPDIR=$v4b_cache/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export XDG_CACHE_HOME=$v4b_cache/xdg
export PYTHONPYCACHEPREFIX=$v4b_cache/pycache
export MIOPEN_USER_DB_PATH=$v4b_cache/miopen-db
export MIOPEN_CUSTOM_CACHE_DIR=$v4b_cache/miopen-custom
export TRITON_CACHE_DIR=$v4b_cache/triton
export TORCHINDUCTOR_CACHE_DIR=$v4b_cache/inductor
export TORCH_EXTENSIONS_DIR=$v4b_cache/extensions
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX" \
  "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
export PYTHONPATH=$v4b_release
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONSAFEPATH=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
export ROCR_VISIBLE_DEVICES=-1
export HIP_VISIBLE_DEVICES=-1
export CUDA_VISIBLE_DEVICES=-1
cd "$v4b_release"

"$v4b_python" -P -B -m unittest \
  methods.bernini_action_editing.tests.test_semantic_anchor_temporal_convae_v4b_fast
"$v4b_python" -O -P -B -m unittest \
  methods.bernini_action_editing.tests.test_semantic_anchor_temporal_convae_v4b_fast
"$v4b_python" -P -B -m py_compile \
  "$v4b_runtime" "$v4b_test" "$v4a_runtime" "$v2_runtime" "$feature_runtime"
"$v4b_python" -P -B "$v4b_runtime" --help
"$v4b_python" -P -B "$v4b_runtime" run-exact5 --help

check_release
echo V4B_CPU_PREFLIGHT_PASS

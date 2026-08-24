#!/usr/bin/env bash
set -Eeuo pipefail

: "${LOCAL_RANK:?torchrun must supply LOCAL_RANK}"
: "${V16R4_RANK_CACHE_ROOT:?task must supply V16R4_RANK_CACHE_ROOT}"
case "${LOCAL_RANK}" in
  0|1|2|3) ;;
  *) echo "invalid WORLD4 local rank: ${LOCAL_RANK}" >&2; exit 2 ;;
esac

rank_root="${V16R4_RANK_CACHE_ROOT}/rank-${LOCAL_RANK}"
if [[ -e "${rank_root}" || -L "${rank_root}" ]]; then
  echo "rank cache is not fresh: ${rank_root}" >&2
  exit 3
fi
mkdir -m 0700 "${rank_root}"
for leaf in miopen-user miopen-custom xdg tmp triton inductor extensions pycache hf torch; do
  mkdir -m 0700 "${rank_root}/${leaf}"
done

export MIOPEN_USER_DB_PATH="${rank_root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${rank_root}/miopen-custom"
export XDG_CACHE_HOME="${rank_root}/xdg"
export TMPDIR="${rank_root}/tmp"
export TMP="${rank_root}/tmp"
export TEMP="${rank_root}/tmp"
export TRITON_CACHE_DIR="${rank_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${rank_root}/inductor"
export TORCH_EXTENSIONS_DIR="${rank_root}/extensions"
export PYTHONPYCACHEPREFIX="${rank_root}/pycache"
export HF_HOME="${rank_root}/hf"
export TORCH_HOME="${rank_root}/torch"

exec "$@"
